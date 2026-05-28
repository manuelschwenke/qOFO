"""
DER Classification Module
=========================

Per-DER classification of grid-forming vs. grid-following control behavior,
independent of whether the DER sits at TSO or DSO level.

Two physical models are exposed:

* ``GRID_FORMING``  — modeled as ``pp.gen`` (PV bus, AVR-style voltage
  control). The supervising OFO commands a per-unit voltage setpoint
  ``vm_pu``. Q is computed by power flow and clipped to
  ``[min_q_mvar, max_q_mvar]`` when ``enforce_q_lims=True``.

* ``GRID_FOLLOWING`` — modeled as ``pp.sgen`` (current source) with an
  optional local Q(V) droop in the simulated plant. The supervising OFO
  commands either Q directly (legacy) or a V_ref setpoint (Stage 2 of the
  per-DER classification rollout). The local Q(V) loop, when present,
  enforces ``Q = clip(-k (V - V_ref), Q_min, Q_max)`` with ``k = 1/slope``.

Default classification reproduces the historical mapping in this codebase:
all sgens registered as TSO-connected DER (``meta.tso_der_indices``) are
grid-forming; all sgens registered as DSO-connected DER
(``meta.dso_der_indices``) are grid-following. Callers may override
per-DER via ``DERClassification.mode``.

The dictionary keys are the *original* pandapower sgen indices as created
by the network builder, before any promotion step. They are stable
identifiers across the build-time sgen→gen promotion: after promotion,
``gen_idx_of_der_id`` records the new ``net.gen`` row for each promoted
unit so controllers can look up the right pandapower table.

Author: Manuel Schwenke
Date: 2026-05-02
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List


class DERMode(Enum):
    """Physical control class for a single DER unit."""

    GRID_FORMING = "grid_forming"
    """Voltage source. Modeled as ``pp.gen``; OFO commands ``vm_pu``."""

    GRID_FOLLOWING = "grid_following"
    """Current source. Modeled as ``pp.sgen`` with optional local Q(V)
    droop; OFO commands Q directly (Stage 1) or V_ref (Stage 2)."""


@dataclass
class DERClassification:
    """Per-DER classification of grid-forming vs. grid-following behavior.

    Attributes
    ----------
    mode : Dict[int, DERMode]
        Classification keyed by *original* pandapower sgen index
        (``der_id``). DERs not present in this dict default to
        ``GRID_FOLLOWING``.
    gen_idx_of_der_id : Dict[int, int]
        Populated by the build-time promotion step. Maps each
        grid-forming ``der_id`` to its new ``net.gen`` row index after
        the original sgen row was dropped.
    qv_slope_pu : Dict[int, float]
        Optional per-DER Q(V) slope override [pu_q / pu_v]. Applies to
        grid-following DERs only. Empty dict ⇒ use the global default
        (``MultiTSOConfig.qv_slope_pu``).
    qv_v_ref_init_pu : Dict[int, float]
        Optional per-DER initial V_ref override [pu]. Applies to
        grid-following DERs only. Empty dict ⇒ use the global default
        (``MultiTSOConfig.qv_setpoint_pu``).
    """

    mode: Dict[int, DERMode] = field(default_factory=dict)
    gen_idx_of_der_id: Dict[int, int] = field(default_factory=dict)
    qv_slope_pu: Dict[int, float] = field(default_factory=dict)
    qv_v_ref_init_pu: Dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    #  Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_default(
        cls,
        *,
        tso_der_indices: Iterable[int] = (),
        dso_der_indices: Iterable[int] = (),
    ) -> "DERClassification":
        """Construct the historical default classification.

        TSO-connected DER → ``GRID_FORMING``.
        DSO-connected DER → ``GRID_FOLLOWING``.
        """
        mode: Dict[int, DERMode] = {}
        for s in tso_der_indices:
            mode[int(s)] = DERMode.GRID_FORMING
        for s in dso_der_indices:
            mode[int(s)] = DERMode.GRID_FOLLOWING
        return cls(mode=mode)

    def with_overrides(self, overrides: Dict[int, str]) -> "DERClassification":
        """Return a copy with per-DER mode overrides applied.

        Parameters
        ----------
        overrides : Dict[int, str]
            Map from sgen index → ``"grid_forming"`` or
            ``"grid_following"``. Unrecognised values raise
            ``ValueError``. Indices not present in the original map are
            added; indices present have their mode replaced.
        """
        new_mode = dict(self.mode)
        for sgen_idx, mode_str in overrides.items():
            try:
                new_mode[int(sgen_idx)] = DERMode(mode_str)
            except ValueError as exc:
                valid = [m.value for m in DERMode]
                raise ValueError(
                    f"Invalid DER mode {mode_str!r} for sgen "
                    f"{sgen_idx}; expected one of {valid}"
                ) from exc
        return DERClassification(
            mode=new_mode,
            gen_idx_of_der_id=dict(self.gen_idx_of_der_id),
            qv_slope_pu=dict(self.qv_slope_pu),
            qv_v_ref_init_pu=dict(self.qv_v_ref_init_pu),
        )

    # ------------------------------------------------------------------
    #  Lookup helpers
    # ------------------------------------------------------------------

    def is_grid_forming(self, der_id: int) -> bool:
        """True if the DER at original sgen index *der_id* is grid-forming.

        DERs not present in the classification default to grid-following.
        """
        return self.mode.get(int(der_id), DERMode.GRID_FOLLOWING) == DERMode.GRID_FORMING

    def is_grid_following(self, der_id: int) -> bool:
        """True if the DER is grid-following (the default for unknown ids)."""
        return not self.is_grid_forming(der_id)

    def gen_idx(self, der_id: int) -> int:
        """Return the ``net.gen`` row index for a promoted grid-forming DER.

        Raises ``KeyError`` if the DER is not grid-forming or has not
        been promoted yet (i.e. the build-time promotion step has not
        been called).
        """
        d = int(der_id)
        if not self.is_grid_forming(d):
            raise KeyError(
                f"DER id {d} is not classified grid-forming; gen_idx is "
                f"only defined for grid-forming units"
            )
        if d not in self.gen_idx_of_der_id:
            raise KeyError(
                f"DER id {d} is grid-forming but has no recorded gen_idx; "
                f"the promotion step has not run yet"
            )
        return int(self.gen_idx_of_der_id[d])

    def record_promotion(self, der_id: int, gen_idx: int) -> None:
        """Record that *der_id* (original sgen index) was promoted to
        ``net.gen`` row *gen_idx*. Called by the promotion helper in
        ``network/ieee39/build.py`` after dropping the sgen and creating
        the gen.
        """
        d = int(der_id)
        if not self.is_grid_forming(d):
            raise ValueError(
                f"Cannot record promotion for DER id {d}: it is not "
                f"classified grid-forming"
            )
        self.gen_idx_of_der_id[d] = int(gen_idx)

    def grid_forming_der_ids(self) -> List[int]:
        """All DER ids classified as grid-forming, in insertion order."""
        return [d for d, m in self.mode.items() if m == DERMode.GRID_FORMING]

    def grid_following_der_ids(self) -> List[int]:
        """All DER ids classified as grid-following, in insertion order."""
        return [d for d, m in self.mode.items() if m == DERMode.GRID_FOLLOWING]

    def qv_slope(self, der_id: int, default: float) -> float:
        """Per-DER Q(V) slope, falling back to *default*."""
        return float(self.qv_slope_pu.get(int(der_id), default))

    def qv_v_ref_init(self, der_id: int, default: float) -> float:
        """Per-DER initial V_ref, falling back to *default*."""
        return float(self.qv_v_ref_init_pu.get(int(der_id), default))
