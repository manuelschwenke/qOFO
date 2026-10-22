"""
Switched-shunt bank model (MSC / MSR) for the TSO OFO controller
================================================================

Holds the state of one mechanically switched shunt bank owned by the TSO at a
DSO tertiary winding. The *dispatch decision* is not here: the bank level is an
integer variable of the TSO MIQP, so the bank is chosen jointly with the
continuous actuators and the OLTCs. This module supplies what that decision
needs — the committed level, which directions are currently permitted, and the
predicted-voltage guard — plus the commit record the runner applies.

Two device classes are provided:

* :class:`MSC` — mechanically switched capacitor. Engaging a step *injects*
  reactive power at the tertiary (raises voltage), so the boundary voltage
  sensitivity per step is expected to be non-negative.
* :class:`MSR` — mechanically switched reactor. Engaging a step *absorbs*
  reactive power (lowers voltage); the expected sensitivity is non-positive.

Each bank carries ``n_levels`` discrete levels (lattice ``l in {0 ... N}``). The
device sign is carried by the measured sensitivity column, so the two classes
differ only in the pandapower step mapping and the expected sensitivity sign
used by the fail-fast guard.

Conventions
-----------
* ``Q_eq`` is the *nameplate* equivalent reactive injection in Mvar; the
  committed physical level satisfies ``Q_eq_phys = l * q_step_mvar`` and the
  per-step increment is ``q_step_mvar``.
* The boundary sensitivity ``h_H`` is ``dy_H / dQ_eq`` per Mvar of nameplate
  injection, evaluated by the online estimator with a unit step.
* Wear is limited by the minimum holding time alone. The daily switching budget
  was removed on 2026-10-22: it used reset windows rather than a rolling
  counter, and it was never a binding constraint in the studies.

Fail-fast: every constructor validates shapes and ranges and raises
``ValueError`` rather than silently degrading (see project CLAUDE.md).

Author: Manuel Schwenke
Date: 2026-06-22
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class ShuntBankConfig:
    """Static configuration for one switched-shunt bank.

    Parameters
    ----------
    shunt_idx :
        Pandapower ``net.shunt`` index of this bank.  Required (not just the
        bus) because a tertiary may host both an MSC and an MSR bank; the
        toggle, the rank-1 SMW refresh, and the disturbance message all target
        this specific device.
    bus_idx :
        Pandapower bus index of the tertiary node hosting the bank (ground
        truth lives here, on the explicit 3-winding tertiary).  Used by the
        boundary-sensitivity helpers, which read the bus voltage.
    interface_trafo3w_idx :
        ``net.trafo3w`` index of the coupling transformer whose tertiary hosts
        the bank.  Used by the runner to compute the DSO interface feedforward
        from the DSO's own three-winding model.
    dso_id :
        Controller identifier of the DSO whose tertiary hosts the bank.
    kind :
        ``"MSC"`` (capacitor) or ``"MSR"`` (reactor).
    q_step_mvar :
        Nameplate reactive power per level step at ``V = 1`` p.u. [Mvar],
        strictly positive (the sign / direction is encoded by the subclass and
        the measured sensitivity, not by this magnitude).
    n_levels :
        Number of discrete steps ``N`` (lattice ``ℓ ∈ {0 … N}``), ``≥ 1``.
    t_dwell_s :
        Minimum holding time between commits of this bank [s], ``≥ 0``.  The
        MIQP sees it through :meth:`ShuntBank.available_directions`, so a bank
        inside its holding time is simply bounded to its current level.
    y_h_min, y_h_max :
        HV-boundary voltage band [p.u.] used by the overshoot feasibility
        guard (uniform across the observed boundary buses).  ``y_h_min <
        y_h_max``.
    """

    shunt_idx: int
    bus_idx: int
    interface_trafo3w_idx: int
    dso_id: str
    kind: str
    q_step_mvar: float
    n_levels: int
    t_dwell_s: float
    y_h_min: float
    y_h_max: float

    def __post_init__(self) -> None:
        if self.kind not in ("MSC", "MSR"):
            raise ValueError(
                f"ShuntBankConfig.kind must be 'MSC' or 'MSR', got {self.kind!r}"
            )
        if not (self.q_step_mvar > 0.0):
            raise ValueError(
                f"q_step_mvar must be > 0, got {self.q_step_mvar}"
            )
        if int(self.n_levels) < 1:
            raise ValueError(
                f"n_levels must be >= 1, got {self.n_levels}"
            )
        if self.t_dwell_s < 0.0:
            raise ValueError(f"t_dwell_s must be >= 0, got {self.t_dwell_s}")
        if not (float(self.y_h_min) < float(self.y_h_max)):
            raise ValueError(
                f"y_h_min ({self.y_h_min}) must be strictly less than "
                f"y_h_max ({self.y_h_max})"
            )


@dataclass
class ShuntCommit:
    """Record of a single committed step change, returned to the runner.

    The runner uses this to (atomically) toggle the physical bank, step the DSO
    interface feedforward, and refresh the cached sensitivities via the rank-1
    SMW update — all in the same control instant, with no power flow.
    """

    shunt_idx: int
    bus_idx: int
    interface_trafo3w_idx: int
    dso_id: str
    kind: str
    old_level: int
    new_level: int
    pp_step_new: int
    q_step_mvar: float
    direction: int  # +1 = engage one more step, -1 = release one step


class ShuntBank:
    """One switched-shunt bank: committed level, holding time and guard.

    Subclassed by :class:`MSC` and :class:`MSR`; the base class holds the state
    and the only difference is :meth:`pp_step_for_level` and
    :attr:`expected_h_sign`.
    """

    #: Expected sign of ``∂V/∂Q_eq`` at the bank's own buses (+1 for a
    #: capacitor, -1 for a reactor).  Used only by the fail-fast sign guard;
    #: the update law itself reads the measured sensitivity directly.
    expected_h_sign: int = 0

    def __init__(self, config: ShuntBankConfig) -> None:
        self.config = config
        # Committed physical level ℓ ∈ {0 … N}; banks start de-energised.
        self.level: int = 0
        # -inf so the first commit is always outside the holding time.
        self._last_switch_t: float = -math.inf

    # -- Derived quantities ------------------------------------------------

    @property
    def q_eq_phys(self) -> float:
        """Committed equivalent nameplate injection [Mvar]: ``ℓ · q_step``."""
        return float(self.level) * self.config.q_step_mvar

    def pp_step_for_level(self, level: int) -> int:
        """Map a lattice level to the pandapower ``net.shunt['step']`` value.

        Overridden per device class.  The base implementation is the identity
        (level == pandapower step), which both MSC and MSR use because the
        capacitive / inductive direction is encoded by the sign of the shunt's
        ``q_mvar`` set at build time, not by the step value.
        """
        return int(level)

    # -- Internal helpers --------------------------------------------------

    def _dwell_elapsed(self, t_now: float) -> bool:
        return (t_now - self._last_switch_t) >= self.config.t_dwell_s

    def _feasible(self, direction: int, v_meas: NDArray[np.float64],
                  h_v: NDArray[np.float64]) -> bool:
        """HV overshoot guard for a candidate one-step move.

        Predicts the boundary voltages after the step using the cached
        sensitivity and checks them against the configured band:

            y_H_min ≤ v_meas + h_v · (direction · q_step) ≤ y_H_max

        ``h_v`` is ``∂V/∂Q_eq`` (per Mvar) at the observed boundary buses.
        """
        v_pred = v_meas + h_v * (float(direction) * self.config.q_step_mvar)
        return bool(
            np.all(v_pred >= self.config.y_h_min)
            and np.all(v_pred <= self.config.y_h_max)
        )

    # -- Public update -----------------------------------------------------

    def available_directions(self, t_now: float) -> List[int]:
        """Directions permitted by the level limits and the holding time.

        This is what the MIQP is given as bounds on the bank variable, so a
        bank inside its holding time cannot be selected at all.
        """
        if not math.isfinite(t_now):
            raise ValueError("Switch time must be finite")
        if not self._dwell_elapsed(t_now):
            return []
        return [d for d in (-1, 1) if 0 <= self.level+d <= self.config.n_levels]

    def commit_selected(self, direction: int, t_now: float) -> ShuntCommit:
        """Commit the step the MIQP selected for this bank.

        The MIQP has already checked the output limits and the one-step-per-zone
        restriction; this records the level change and restarts the holding
        clock.
        """
        if direction not in self.available_directions(t_now):
            raise ValueError("Selected shunt step violates the holding time or a level limit")
        old = self.level
        self.level += direction
        self._last_switch_t = t_now
        c = self.config
        return ShuntCommit(c.shunt_idx, c.bus_idx, c.interface_trafo3w_idx,
            c.dso_id, c.kind, old, self.level, self.pp_step_for_level(self.level),
            c.q_step_mvar, direction)

class MSC(ShuntBank):
    """Mechanically switched capacitor: engaging a step injects reactive power
    (raises voltage), so ``∂V/∂Q_eq ≥ 0`` is expected at the bank's buses."""

    expected_h_sign = +1


class MSR(ShuntBank):
    """Mechanically switched reactor: engaging a step absorbs reactive power
    (lowers voltage), so ``∂V/∂Q_eq ≤ 0`` is expected at the bank's buses."""

    expected_h_sign = -1


def make_bank(config: ShuntBankConfig) -> ShuntBank:
    """Construct the concrete bank instance for ``config.kind``."""
    if config.kind == "MSC":
        return MSC(config)
    if config.kind == "MSR":
        return MSR(config)
    raise ValueError(f"Unknown shunt kind {config.kind!r}")
