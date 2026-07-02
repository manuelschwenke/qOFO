"""
Horizontal TSO–TSO Tie-Coordinator Module  (gradient-exchange design)
====================================================================

This module defines :class:`HorizontalTieCoordinator`, the lightweight
**horizontal** coordinator that mediates reactive-power coordination between
neighbouring TSO control zones across their shared tie lines.

Design (two loops, gradient exchange)
-------------------------------------
Per tie line ``e = (i, j)`` the coordinator negotiates one scalar, the **agreed
boundary-voltage difference** ``ΔV_ref`` (= target ``V_i − V_j``).  It does *not*
force a common voltage and *not* inject a price into the zones' objectives.

* **Inner loop (fast, per zone):** each zone tracks its agreed per-side boundary
  setpoint through its *existing primary* objective::

      V_ref_i = V_anchor + κ·ΔV_ref ,   V_ref_j = V_anchor − (1−κ)·ΔV_ref
      V_anchor = ½(V_nom_i + V_nom_j)

  so ``V_ref_i − V_ref_j = ΔV_ref``.  An unreachable setpoint is just a bounded
  tracking error (cannot diverge the PF).

* **Outer loop (slow, coordinator):** each zone shares the marginal of its
  **full OFO objective** w.r.t. its boundary voltage,

      γ_i = (∇J_i · h_b) / (h_b · h_b)        # h_b = ∂V_b/∂u (boundary V row of H)

  the projection of the *iterating* control-space gradient ``∇J_i`` (voltage
  tracking + reserve + effort + …, whatever is weighted in the objective) onto
  the boundary-voltage direction.  Because it uses the iterating gradient — not
  the converged setpoint marginal — the envelope theorem does **not** zero the
  reserve (and, later, loss) terms: every objective component contributes.

  The coordinator then descends the **combined** marginal (the joint objective's
  gradient w.r.t. ``ΔV_ref``) with a per-zone safeguard::

      G   = κ·γ_i − (1−κ)·γ_j                          # ∂(J_i + J_j)/∂ΔV_ref
      Δ   = −grad_alpha·G − anchor·DB_Δ(ΔV_ref)        # joint descent + weak subsidiarity
      safeguard: dJ_i ≈ γ_i·κ·Δ , dJ_j ≈ −γ_j·(1−κ)·Δ
                 if max(dJ_i, dJ_j) > grad_eps:  Δ ← Δ · grad_eps / max(dJ_i, dJ_j)
      ΔV_ref ← Π_[±Δmax]{ ΔV_ref + Δ }

  ``G`` points to where the two zones *jointly* want the boundary; descending it
  helps whichever zone's marginal is larger, and the cap ``grad_eps`` bounds how
  far either zone is dragged uphill per round ("help the neighbour only if it's
  cheap for me").  The weak subsidiarity anchor biases ``ΔV_ref → 0`` when the
  zones are jointly indifferent.

The current voltage-only relaxation is the special case ``γ_i = −2g_v·err_i``
(``g_res = 0``); enriching ``∇J_i`` generalises it to reserve / losses / etc.

Author: Manuel Schwenke / Claude Code
Date: 2026-06-26
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.message import TieCoordinationMessage


@dataclass(frozen=True)
class TieLink:
    """Static description of one inter-zone tie line ``e = (i, j)``.

    Orientation is fixed (``zone_i < zone_j`` by convention) and stable across
    rounds, since the sign of ``ΔV_ref`` and the per-side split depend on it.

    Attributes
    ----------
    tie_id : int
        Identifier of the tie line (e.g. pandapower line index).
    zone_i, zone_j : int
        Zone identifiers of the two TSO zones sharing the line.
    bus_i, bus_j : int
        Pandapower bus indices of the in-zone endpoint (boundary bus) in
        ``zone_i`` and ``zone_j``; each must be a monitored voltage bus.
    controller_i, controller_j : str
        ``controller_id`` of the two zones' TSO controllers (message targets).
    v_nom_i, v_nom_j : float
        Each zone's nominal (scheduled) boundary voltage [p.u.].  Their midpoint
        is the anchor level the per-side setpoints are built around.
    """

    tie_id: int
    zone_i: int
    zone_j: int
    bus_i: int
    bus_j: int
    controller_i: str
    controller_j: str
    v_nom_i: float = 1.0
    v_nom_j: float = 1.0

    @property
    def v_anchor(self) -> float:
        """Common anchor level = midpoint of the two zones' schedules."""
        return 0.5 * (self.v_nom_i + self.v_nom_j)


@dataclass
class TieCoordinatorConfig:
    """Tuning parameters for :class:`HorizontalTieCoordinator`."""

    grad_alpha: float = 5e-8
    """Descent step on the combined boundary gradient ``G`` [p.u. / (objective
    per p.u.)].  The natural scale is ``≈ 1/(2·g_v)`` (the voltage-tracking
    curvature), so the runner sets it from a dimensionless ``tie_grad_step`` and
    ``g_v``; the default here matches ``g_v ≈ 1e7``."""

    grad_eps: float = 10.0
    """Per-zone, per-round cap on objective *worsening* [objective units].  The
    candidate move is shrunk so neither zone's predicted ``dJ`` exceeds this —
    the "don't worsen my own too much" safeguard.  Scales with the objective
    (``~ g_v·err²``); tune alongside ``g_v``.  ``0`` ⇒ only strictly-jointly-
    beneficial moves are allowed."""

    anchor: float = 0.2
    """Weak subsidiarity pull ``≥ 0`` biasing ``ΔV_ref → 0`` (zero exchange) when
    the zones are jointly indifferent; a tiebreaker, not the driver."""

    deadband_v_pu: float = 0.002
    """Deadband Δ [p.u.] on the anchor: ``|ΔV_ref| ≤ Δ`` is left untouched."""

    kappa: float = 0.5
    """Per-side split of ``ΔV_ref`` (0.5 = symmetric, recommended)."""

    dvref_max: float = 0.06
    """Clip on the agreed difference, ``|ΔV_ref| ≤ dvref_max`` [p.u.]."""

    def __post_init__(self) -> None:
        if self.grad_alpha <= 0.0:
            raise ValueError("grad_alpha must be positive")
        if self.grad_eps < 0.0:
            raise ValueError("grad_eps must be non-negative")
        if self.anchor < 0.0:
            raise ValueError("anchor must be non-negative")
        if self.deadband_v_pu < 0.0:
            raise ValueError("deadband_v_pu must be non-negative")
        if not (0.0 <= self.kappa <= 1.0):
            raise ValueError("kappa must lie in [0, 1]")
        if self.dvref_max <= 0.0:
            raise ValueError("dvref_max must be positive")


def _deadband(x: float, dz: float) -> float:
    """Soft-threshold ``sign(x)·max(0, |x| − dz)``."""
    if x > dz:
        return x - dz
    if x < -dz:
        return x + dz
    return 0.0


class HorizontalTieCoordinator:
    """Horizontal coordinator descending the combined boundary gradient.

    Holds the agreed difference ``ΔV_ref`` per tie; advances it from the two
    zones' boundary objective-gradients via :meth:`update`; emits per-side
    boundary setpoints via :meth:`generate_messages`.  Plant-free, unit-testable.

    Parameters
    ----------
    links : list of TieLink
        One entry per coordinated tie line (unique ``tie_id``).
    config : TieCoordinatorConfig
        Tuning parameters (shared across all ties).
    """

    def __init__(
        self,
        links: List[TieLink],
        config: TieCoordinatorConfig,
    ) -> None:
        ids = [lk.tie_id for lk in links]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate tie_id in links: {ids}")
        self.links: Dict[int, TieLink] = {lk.tie_id: lk for lk in links}
        self.config = config
        self.iteration: int = 0

        # Agreed difference per tie (starts at 0 = full subsidiarity).
        self.dvref: Dict[int, float] = {lk.tie_id: 0.0 for lk in links}
        # Diagnostics from the last update (for logging / live plot).
        self.grad_i: Dict[int, float] = {lk.tie_id: 0.0 for lk in links}
        self.grad_j: Dict[int, float] = {lk.tie_id: 0.0 for lk in links}
        self.grad_combined: Dict[int, float] = {lk.tie_id: 0.0 for lk in links}

    # ------------------------------------------------------------------
    #  Core update
    # ------------------------------------------------------------------

    def update(self, gradients: Dict[int, Tuple[float, float]]) -> None:
        """Advance the coordinator state by one outer round.

        Parameters
        ----------
        gradients : dict {tie_id: (γ_i, γ_j)}
            Each zone's marginal of its full OFO objective w.r.t. its boundary
            voltage (``γ = (∇J·h_b)/(h_b·h_b)``).  Ties absent keep their state.
        """
        cfg = self.config
        for tie_id, (g_i, g_j) in gradients.items():
            if tie_id not in self.links:
                raise KeyError(f"unknown tie_id {tie_id}")
            g_i = float(g_i)
            g_j = float(g_j)
            dvref = self.dvref[tie_id]

            # Combined gradient of (J_i + J_j) w.r.t. ΔV_ref.
            G = cfg.kappa * g_i - (1.0 - cfg.kappa) * g_j

            # Candidate move: joint descent + weak subsidiarity anchor.
            delta = (
                -cfg.grad_alpha * G
                - cfg.anchor * _deadband(dvref, cfg.deadband_v_pu)
            )

            # Safeguard: cap the per-round objective worsening of EITHER zone.
            dJ_i = g_i * cfg.kappa * delta
            dJ_j = -g_j * (1.0 - cfg.kappa) * delta
            worst = max(dJ_i, dJ_j, 0.0)
            if worst > cfg.grad_eps:
                delta *= cfg.grad_eps / worst

            self.dvref[tie_id] = float(
                np.clip(dvref + delta, -cfg.dvref_max, cfg.dvref_max)
            )
            self.grad_i[tie_id] = g_i
            self.grad_j[tie_id] = g_j
            self.grad_combined[tie_id] = G

        self.iteration += 1

    # ------------------------------------------------------------------
    #  Message generation
    # ------------------------------------------------------------------

    def generate_messages(self) -> List[TieCoordinationMessage]:
        """Emit one :class:`TieCoordinationMessage` per zone-controller.

        Carries each boundary bus's per-side voltage setpoint
        ``V_ref = V_anchor ± (split)·ΔV_ref`` (no price).  A zone touching
        several ties receives one aggregated message.
        """
        cfg = self.config
        per_zone: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
        for tie_id, lk in self.links.items():
            dvref = self.dvref[tie_id]
            v_ref_i = lk.v_anchor + cfg.kappa * dvref
            v_ref_j = lk.v_anchor - (1.0 - cfg.kappa) * dvref
            per_zone[lk.controller_i].append((tie_id, lk.bus_i, v_ref_i))
            per_zone[lk.controller_j].append((tie_id, lk.bus_j, v_ref_j))

        messages: List[TieCoordinationMessage] = []
        for ctrl_id, rows in per_zone.items():
            messages.append(
                TieCoordinationMessage(
                    source_controller_id="tie_coordinator",
                    target_controller_id=ctrl_id,
                    iteration=self.iteration,
                    tie_line_indices=np.array([r[0] for r in rows], dtype=np.int64),
                    boundary_bus_indices=np.array([r[1] for r in rows], dtype=np.int64),
                    v_ref_pu=np.array([r[2] for r in rows], dtype=np.float64),
                )
            )
        return messages

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def state(self) -> Dict[int, Dict[str, float]]:
        """Per-tie ``{dvref, grad_i, grad_j, grad_combined}`` snapshot."""
        return {
            tie_id: {
                "dvref": self.dvref[tie_id],
                "grad_i": self.grad_i[tie_id],
                "grad_j": self.grad_j[tie_id],
                "grad_combined": self.grad_combined[tie_id],
            }
            for tie_id in self.links
        }
