"""
sbx/capability.py
=================
Joint-box capability LP for SBX (plan v2 §2.4, REPLACED by amendment
v2.2 item 1 / D13 — STATUS_SBX.md).

Per area i with corridor set C_i, ONE LP per cycle:

    max  t ≥ 0
    s.t. for every sign vector σ ∈ {−1, +1}^{|C_i|}, exists Δu_σ:
         u_min ≤ u + Δu_σ ≤ u_max
         v_min + margin ≤ v_meas + H_loc · Δu_σ ≤ v_max − margin
         s_corr_c · (H_term_c · Δu_σ) = t · σ_c · dq_quant_c   ∀ c ∈ C_i

Offer per corridor c: ``offer_range_mvar = (−a_c, +a_c)`` with
``a_c = min(t, 1) · dq_quant_c``.  The box with these half-widths is
inscribed in the joint feasibility polytope (convexity: vertex
feasibility implies box feasibility), so ANY combination of accepted
deals across the area's corridors is jointly achievable.

If the measured point violates the margined voltage limits, the LP is
skipped and every corridor is offered (0, 0) — consistent with the need
flag being set (v2.2 item 1).

Symbol map (code ↔ plan/amendment)
----------------------------------
* ``CorridorCoupling.control_row`` ↔ s_corr_c · H_term_c ∈ R^{n_u}
  [Mvar per actuator unit]: the composed row d q_corr_c / d u — built by
  the scheduler from the per-line own-side sensitivities of
  ``sbx.corridor.corridor_sensitivities`` and the cached local H rows at
  the corridor terminal buses.
* ``H_loc``                       ↔ local sensitivity ∂v/∂u of the area's
  monitored voltage buses (LOCAL data only — the controllers never see
  the plant, only their cached model).
* ``t`` / offers                  ↔ D13 above.

Solver: the EXISTING wrapper ``optimisation.miqp_solver.MIQPSolver``
(hard rule 5 — no solver modification).  The joint-box LP maps exactly
onto the continuous ``_solve_qp`` branch with ``G_w = 0`` (pure LP),
``G_z = 0`` (hard output constraints), ``alpha = 1``:

* decision vector  w = [t, Δu_σ1, …, Δu_σm],  m = 2^{|C_i|};
* box bounds on w  = [0, T_CAP] × Π_σ [u_min − u, u_max − u];
* output rows      = per σ-block: n_v voltage rows (y = v_meas + H_loc·Δu_σ
  within margined bounds) and |C_i| equality rows
  (s_corr_c·H_term_c·Δu_σ − t·σ_c·dq_c = 0, encoded as y_lower = y_upper = 0);
* objective        = min −t.

The wrapper returns a status instead of raising on failure; this module
checks ``is_optimal`` and calls ``rep1`` otherwise — the capability LP
never fails silently (Phase 3 acceptance / v2.2 item 4).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 3)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from optimisation.miqp_solver import MIQPProblem, MIQPSolver
from sbx.fail import rep1

#: Upper box bound on t.  The LP is bounded through the equality rows and
#: the actuator box anyway; this cap only guards numerically degenerate
#: couplings.  Offers clip at min(t, 1), so any T_CAP ≥ 1 is equivalent.
T_CAP = 100.0

#: 2^n_corridors blocks are stacked into one LP; the triangle headline has
#: n = 2 per area.  The hard cap keeps a mis-wired registry from building
#: a monster problem.
MAX_CORRIDORS_PER_AREA = 6


@dataclass(frozen=True)
class CorridorCoupling:
    """One corridor's coupling data for the joint-box LP (v2.2 D13)."""

    key: Tuple[int, int]
    control_row: NDArray[np.float64]
    dq_quant_mvar: float

    def __post_init__(self) -> None:
        row = np.array(self.control_row, dtype=np.float64, copy=True)
        if row.ndim != 1:
            rep1("CorridorCoupling.control_row must be a 1-D vector",
                 key=self.key, shape=row.shape)
        if not np.all(np.isfinite(row)):
            rep1("CorridorCoupling.control_row contains non-finite entries",
                 key=self.key)
        row.flags.writeable = False
        object.__setattr__(self, "control_row", row)
        if not (math.isfinite(self.dq_quant_mvar)
                and self.dq_quant_mvar > 0.0):
            rep1("CorridorCoupling.dq_quant_mvar must be positive",
                 key=self.key, dq_quant_mvar=self.dq_quant_mvar)


@dataclass(frozen=True)
class CapabilityResult:
    """Joint-box outcome: t, per-corridor offers, and the skip marker."""

    t: float
    offers_mvar: Dict[Tuple[int, int], Tuple[float, float]]
    skipped_due_to_violation: bool


def joint_box_capability(
    u_now: Sequence[float],
    u_min: Sequence[float],
    u_max: Sequence[float],
    v_meas_pu: Sequence[float],
    v_min_pu: Sequence[float],
    v_max_pu: Sequence[float],
    h_loc: NDArray[np.float64],
    couplings: Sequence[CorridorCoupling],
    solver: MIQPSolver,
    *,
    voltage_margin_pu: float,
) -> CapabilityResult:
    """Solve the v2.2 D13 joint-box LP for one area (one call per cycle).

    Parameters are LOCAL data only: the area's actuator vector and box
    bounds, its monitored voltages with hard bounds, its cached local
    sensitivity ``h_loc`` (n_v × n_u), and one :class:`CorridorCoupling`
    per incident corridor.  Returns the scalar ``t`` and the per-corridor
    symmetric offers ``(−a_c, +a_c)``, ``a_c = min(t, 1) · dq_quant_c``.
    """
    u_now = np.asarray(u_now, dtype=np.float64)
    u_lo = np.asarray(u_min, dtype=np.float64)
    u_hi = np.asarray(u_max, dtype=np.float64)
    v = np.asarray(v_meas_pu, dtype=np.float64)
    v_lo = np.asarray(v_min_pu, dtype=np.float64)
    v_hi = np.asarray(v_max_pu, dtype=np.float64)
    h_loc = np.asarray(h_loc, dtype=np.float64)

    n_u = u_now.size
    n_v = v.size
    if not (u_lo.shape == u_hi.shape == u_now.shape):
        rep1("actuator vectors must share one shape",
             n_u=n_u, n_lo=u_lo.size, n_hi=u_hi.size)
    if not (v_lo.shape == v_hi.shape == v.shape):
        rep1("voltage vectors must share one shape",
             n_v=n_v, n_lo=v_lo.size, n_hi=v_hi.size)
    if h_loc.shape != (n_v, n_u):
        rep1("h_loc must be (n_v x n_u)", shape=h_loc.shape,
             n_v=n_v, n_u=n_u)
    for arr, name in ((u_now, "u_now"), (u_lo, "u_min"), (u_hi, "u_max"),
                      (v, "v_meas_pu"), (v_lo, "v_min_pu"),
                      (v_hi, "v_max_pu"), (h_loc.ravel(), "h_loc")):
        if not np.all(np.isfinite(arr)):
            rep1("capability input contains non-finite entries", field=name)
    if np.any(u_lo > u_now) or np.any(u_now > u_hi):
        rep1("current actuator vector lies outside its own box bounds",
             offenders=np.where((u_lo > u_now) | (u_now > u_hi))[0].tolist())
    if not couplings:
        rep1("joint_box_capability needs at least one corridor coupling")
    if len(couplings) > MAX_CORRIDORS_PER_AREA:
        rep1("too many corridors for one joint-box LP",
             n=len(couplings), cap=MAX_CORRIDORS_PER_AREA)
    keys = [c.key for c in couplings]
    if len(set(keys)) != len(keys):
        rep1("duplicate corridor keys in couplings", keys=keys)
    for c in couplings:
        if c.control_row.size != n_u:
            rep1("control_row length must equal n_u",
                 key=c.key, got=c.control_row.size, n_u=n_u)
    if voltage_margin_pu <= 0.0:
        rep1("voltage_margin_pu must be positive",
             voltage_margin_pu=voltage_margin_pu)

    lo_m = v_lo + voltage_margin_pu
    hi_m = v_hi - voltage_margin_pu
    if np.any(lo_m > hi_m):
        rep1("margined voltage bounds cross (margin too large for the "
             "configured corridor)",
             offenders=np.where(lo_m > hi_m)[0].tolist(),
             voltage_margin_pu=voltage_margin_pu)

    # Measured point outside the margined limits → skip the LP, offer
    # (0, 0) on all corridors (v2.2 item 1; consistent with the need flag).
    if np.any(v < lo_m) or np.any(v > hi_m):
        return CapabilityResult(
            t=0.0,
            offers_mvar={c.key: (0.0, 0.0) for c in couplings},
            skipped_due_to_violation=True,
        )

    n_c = len(couplings)
    vertices: List[Tuple[int, ...]] = list(
        itertools.product((-1, +1), repeat=n_c)
    )
    m = len(vertices)                       # 2^n_c blocks
    n_total = 1 + m * n_u                   # w = [t, du_1, ..., du_m]
    n_rows_per_block = n_v + n_c
    n_outputs = m * n_rows_per_block

    h_tilde = np.zeros((n_outputs, n_total), dtype=np.float64)
    y_current = np.zeros(n_outputs, dtype=np.float64)
    y_lower = np.zeros(n_outputs, dtype=np.float64)
    y_upper = np.zeros(n_outputs, dtype=np.float64)

    for blk, sigma in enumerate(vertices):
        col0 = 1 + blk * n_u
        row0 = blk * n_rows_per_block
        # Voltage rows: lo_m ≤ v_meas + H_loc·du_σ ≤ hi_m.
        h_tilde[row0:row0 + n_v, col0:col0 + n_u] = h_loc
        y_current[row0:row0 + n_v] = v
        y_lower[row0:row0 + n_v] = lo_m
        y_upper[row0:row0 + n_v] = hi_m
        # Equality rows: control_row·du_σ − t·σ_c·dq_c = 0.
        for k, c in enumerate(couplings):
            r = row0 + n_v + k
            h_tilde[r, col0:col0 + n_u] = c.control_row
            h_tilde[r, 0] = -float(sigma[k]) * c.dq_quant_mvar
            # y_current = 0, y_lower = y_upper = 0 already.

    w_lower = np.concatenate(
        [[0.0]] + [u_lo - u_now] * m
    )
    w_upper = np.concatenate(
        [[T_CAP]] + [u_hi - u_now] * m
    )

    grad_f = np.zeros(n_total, dtype=np.float64)
    grad_f[0] = -1.0                        # max t ≡ min −t

    problem = MIQPProblem(
        n_continuous=n_total,
        n_integer=0,
        n_outputs=n_outputs,
        alpha=1.0,
        G_w=np.zeros((n_total, n_total), dtype=np.float64),
        G_z=np.zeros((n_outputs, n_outputs), dtype=np.float64),
        grad_f=grad_f,
        H_tilde=h_tilde,
        u_current=np.zeros(n_total, dtype=np.float64),
        u_lower=w_lower,
        u_upper=w_upper,
        y_current=y_current,
        y_lower=y_lower,
        y_upper=y_upper,
        integer_indices=[],
    )

    result = solver.solve(problem)
    if not result.is_feasible:
        rep1("joint-box capability LP did not solve to (near-)optimality "
             "— the capability path never fails silently (v2.2 item 4)",
             status=result.status, n_total=n_total, n_outputs=n_outputs,
             corridors=keys)

    t = float(result.w_continuous[0])
    if not math.isfinite(t) or t < -1e-9:
        rep1("joint-box LP returned a non-physical t", t=t,
             status=result.status)
    t = max(t, 0.0)

    offers: Dict[Tuple[int, int], Tuple[float, float]] = {}
    scale = min(t, 1.0)
    for c in couplings:
        a_c = scale * c.dq_quant_mvar
        offers[c.key] = (-a_c, +a_c)
    return CapabilityResult(
        t=t, offers_mvar=offers, skipped_due_to_violation=False
    )
