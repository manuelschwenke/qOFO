"""
sbx_v/miqp_cost.py
=================
Piecewise-linear tier cost layer for the TSO MIQP (plan §5, V-D1, V-D9).

Mechanism
---------
Per AggregationArea and direction, the netted boundary quantity is
priced by a CONVEX piecewise-linear cost anchored at a band edge: the
free tier costs nothing, granted Vorhalteleistung the
Arbeits-Durchschnittspreis, everything beyond the Grenzpreis.  Grants
and postings change prices, never feasibility — the CAIR capability box
remains the only physical constraint (V-D1).

Integration without touching protected modules (hard rule 5)
-------------------------------------------------------------
The controllers call ``self.solver.solve(problem)`` with a fully built
:class:`optimisation.miqp_solver.MIQPProblem`.  SBX-V therefore wraps
the solver INSTANCE from outside (:class:`PricingSolver`): when a tier
specification is active, the problem is AUGMENTED — per priced side one
extra output row bounding the netted signed quantity by its anchor, and
per tier segment one extra continuous variable with a pure linear cost
that relaxes that row — solved, and the result STRIPPED back to the
original problem shape before the controller sees it.  With no active
specification the problem passes through untouched, which makes the
neutral-configuration regression R1 byte-identical by construction.

Encoding (per side ``d`` of area ``a``; ``s = sign_d · q_a`` is the side
coordinate, positive when operating on side ``d``):

    s − Σ_i x_i ≤ anchor_d          (new output row, slack weight g_z_tier)
    0 ≤ x_i ≤ width_i               (new continuous variables)
    objective += Σ_i slope_i · x_i  (slopes non-decreasing → convex)

Because slopes are non-decreasing, the solver fills cheap segments
first; the reconstruction invariant (plan §5) is asserted after every
solve.  The LAST segment of every side is an open tail at the
Grenzpreis slope, so the row can never tighten feasibility (V-D1) —
beyond the posted potential no higher tier exists, hence the marginal
price saturates at the Grenzpreis; the posted-potential cap governs
attribution and settlement, not feasibility (STATUS_SBXV.md Phase 1
design note).

Slack caveat: the solver's shared-slack encoding penalises tier-row
violations QUADRATICALLY (weight ``g_z_tier``), so up to
``slope/(2·g_z_tier)`` Mvar of excess rides the slack instead of the
segments (≈ 4e-4 Mvar at default prices).  The reconstruction assert
includes the observed slack and rejects anything beyond this bound.

α-scaling: the solver treats continuous variables as micro-steps
``w = Δu/α`` and applies outputs as ``Δy = α·H_c·w_c``.  Auxiliary
segment variables use ``u_current = 0``, ``u_upper = width`` and row
coefficient −1, so their EFFECTIVE Mvar value is ``x = α·w`` ∈
[0, width]; the linear objective entry is ``slope · α`` so that
``grad · w = slope · x``.  All algebra below is α-exact.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from optimisation.miqp_solver import MIQPProblem, MIQPResult
from sbx_h.fail import rep1
from sbx_v.band import NormalBand
from sbx_v.config import (COST_MODEL_LEITFADEN, COST_MODEL_NEAREST_EDGE,
                         SBXVConfig)
from sbx_v.directions import Direction

#: Open-tail segment width [Mvar] — large but FINITE (the solver asserts
#: finite problem data); far beyond any physical boundary Q.
OPEN_TAIL_MVAR = 1.0e6

#: Wide finite lower bound for the one-sided tier rows.
ROW_LOWER_MVAR = -1.0e9

#: Absolute tolerance [Mvar] of the reconstruction invariant (plan §5).
RECON_TOL_MVAR = 1.0e-3


@dataclass(frozen=True)
class TierSegment:
    """One cost segment: ``width_mvar`` at ``slope_obj_per_mvar``."""

    width_mvar: float
    slope_obj_per_mvar: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.width_mvar) or self.width_mvar <= 0.0:
            rep1("segment width must be positive and finite",
                 width_mvar=self.width_mvar)
        if not math.isfinite(self.slope_obj_per_mvar) or \
                self.slope_obj_per_mvar < 0.0:
            rep1("segment slope must be finite and non-negative",
                 slope_obj_per_mvar=self.slope_obj_per_mvar)


@dataclass(frozen=True)
class SideSpec:
    """Piecewise-linear price of one side (direction) of one area.

    ``anchor_mvar`` is where cost starts, in the SIDE COORDINATE
    ``s = sign_d · q_a`` (so the own band edge is ``+edge_d``; the
    opposite edge — V-D9 re-anchoring during an active grant — is
    ``−edge_opposite``).  ``segments`` must have non-decreasing slopes
    (convexity) and end with an open tail (V-D1: never tighten
    feasibility).
    """

    direction: Direction
    anchor_mvar: float
    segments: Tuple[TierSegment, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.anchor_mvar):
            rep1("anchor must be finite", anchor_mvar=self.anchor_mvar)
        if not self.segments:
            rep1("a priced side needs at least one segment",
                 direction=self.direction)
        slopes = [s.slope_obj_per_mvar for s in self.segments]
        if any(b < a - 1e-12 for a, b in zip(slopes, slopes[1:])):
            rep1("segment slopes must be non-decreasing (convexity, "
                 "plan §5 'ordered')", direction=self.direction,
                 slopes=tuple(slopes))
        if self.segments[-1].width_mvar < OPEN_TAIL_MVAR:
            rep1("the last segment must be the open tail — a bounded "
                 "final segment would tighten feasibility (V-D1)",
                 direction=self.direction,
                 last_width_mvar=self.segments[-1].width_mvar)


@dataclass(frozen=True)
class AreaTierSpec:
    """Per-window tier specification of ONE AggregationArea (DP5: the
    area nets its PCC NVPs; rows are the area's Q_PCC output rows in the
    UNAUGMENTED problem and are summed to the netted quantity)."""

    area_id: str
    pcc_output_rows: Tuple[int, ...]
    raising: SideSpec
    lowering: SideSpec

    def __post_init__(self) -> None:
        if not self.area_id:
            rep1("area_id must be non-empty", area_id=self.area_id)
        if not self.pcc_output_rows:
            rep1("an AggregationArea needs at least one PCC output row",
                 area_id=self.area_id)
        if len(set(self.pcc_output_rows)) != len(self.pcc_output_rows) or \
                any(r < 0 for r in self.pcc_output_rows):
            rep1("PCC output rows must be unique and non-negative",
                 area_id=self.area_id, rows=self.pcc_output_rows)
        if self.raising.direction is not Direction.RAISING or \
                self.lowering.direction is not Direction.LOWERING:
            rep1("side specs attached to the wrong direction slots",
                 area_id=self.area_id,
                 raising=self.raising.direction,
                 lowering=self.lowering.direction)


@dataclass(frozen=True)
class TierDecomposition:
    """Post-solve tier attribution of one side (plan §5 reconstruction).

    ``q_netted_signed_mvar`` is the PREDICTED post-step netted boundary
    Q (model prediction ``y + α·H·w``); ``excess_mvar`` its part beyond
    the anchor in the side coordinate; ``x_mvar`` the per-segment fill;
    ``slack_mvar`` the tier-row slack used; ``cost_obj`` the commercial
    objective contribution of this side.
    """

    area_id: str
    direction: Direction
    q_netted_signed_mvar: float
    excess_mvar: float
    x_mvar: Tuple[float, ...]
    slack_mvar: float
    cost_obj: float


# ----------------------------------------------------------------------
#  Side-spec construction (V-D9 incentive models)
# ----------------------------------------------------------------------

def build_side_spec(
    *,
    direction: Direction,
    band: NormalBand,
    grant_mvar: float,
    config: SBXVConfig,
) -> SideSpec:
    """Side spec per the configured incentive model (V-D9).

    No active grant (``grant_mvar == 0``, the Phase-1 state): both
    models coincide — anchor at the own band edge, everything beyond at
    the Grenzpreis (the CAIR posting doubles as Potenzialmeldung and its
    posting implies consent to Abruf [LF §6]; V-D4: no gating).

    Active grant:

    * ``nearest_edge`` — anchor at the own edge; ``grant_mvar`` at the
      Arbeits-Durchschnittspreis, Grenzpreis beyond.
    * ``leitfaden_exact_when_granted`` — anchor at the OPPOSITE band
      edge [LF §7.2 case 1]: the Durchschnittspreis slope runs from the
      opposite edge up to the grant maximum (own edge + grant), the
      Grenzpreis beyond.  Constant cost offsets are irrelevant to the
      argmin, so anchoring encodes the exact Leitfaden marginal prices.

    Capacity (Leistungs-) prices are sunk once granted and never enter
    the MIQP (V-D9).
    """
    if not math.isfinite(grant_mvar) or grant_mvar < 0.0:
        rep1("grant magnitude must be finite and non-negative",
             direction=direction, grant_mvar=grant_mvar)
    c_vh = config.c_vh_obj_per_mvar_step
    c_ug = config.c_ug_obj_per_mvar_step
    own_edge = band.edge_mvar(direction)

    if grant_mvar == 0.0:
        return SideSpec(
            direction=direction,
            anchor_mvar=own_edge,
            segments=(TierSegment(OPEN_TAIL_MVAR, c_ug),),
        )

    if config.miqp_cost_model == COST_MODEL_NEAREST_EDGE:
        return SideSpec(
            direction=direction,
            anchor_mvar=own_edge,
            segments=(TierSegment(grant_mvar, c_vh),
                      TierSegment(OPEN_TAIL_MVAR, c_ug)),
        )
    if config.miqp_cost_model == COST_MODEL_LEITFADEN:
        opp_edge = band.edge_mvar(direction.opposite)
        # Durchschnittspreis from the opposite edge (side coordinate
        # −opp_edge) up to the grant maximum (own_edge + grant).
        vh_width = opp_edge + own_edge + grant_mvar
        if vh_width <= 0.0:
            rep1("degenerate Leitfaden segment (zero band spread and "
                 "zero grant)", area_id=band.area_id, direction=direction)
        return SideSpec(
            direction=direction,
            anchor_mvar=-opp_edge,
            segments=(TierSegment(vh_width, c_vh),
                      TierSegment(OPEN_TAIL_MVAR, c_ug)),
        )
    rep1("unknown miqp_cost_model", miqp_cost_model=config.miqp_cost_model)


def area_tier_spec(
    *,
    area_id: str,
    pcc_output_rows: Sequence[int],
    band: NormalBand,
    grant_raise_mvar: float,
    grant_lower_mvar: float,
    config: SBXVConfig,
) -> AreaTierSpec:
    """Assemble the per-window spec of one AggregationArea.

    Simultaneous active grants in BOTH directions under
    ``leitfaden_exact_when_granted`` would re-anchor both sides across
    the band and double-charge the in-band region; the plan does not
    define this case — STOP per hard rule 5 spirit (report, do not
    improvise).
    """
    if (config.miqp_cost_model == COST_MODEL_LEITFADEN
            and grant_raise_mvar > 0.0 and grant_lower_mvar > 0.0):
        rep1("simultaneous grants in both directions are not defined "
             "under 'leitfaden_exact_when_granted' (opposite-edge "
             "re-anchoring would double-charge the band); blocked "
             "pending a plan decision", area_id=area_id,
             grant_raise_mvar=grant_raise_mvar,
             grant_lower_mvar=grant_lower_mvar)
    return AreaTierSpec(
        area_id=area_id,
        pcc_output_rows=tuple(int(r) for r in pcc_output_rows),
        raising=build_side_spec(direction=Direction.RAISING, band=band,
                                grant_mvar=grant_raise_mvar, config=config),
        lowering=build_side_spec(direction=Direction.LOWERING, band=band,
                                 grant_mvar=grant_lower_mvar, config=config),
    )


# ----------------------------------------------------------------------
#  Problem augmentation
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class _SideMap:
    """Bookkeeping of one augmented side."""

    area_id: str
    direction: Direction
    sign: float
    pcc_rows: Tuple[int, ...]
    anchor_mvar: float
    row_index: int                 # index of the tier row in the AUGMENTED outputs
    aux_start: int                 # first aux column (augmented u indexing)
    n_seg: int
    slopes: Tuple[float, ...]
    widths: Tuple[float, ...]


@dataclass(frozen=True)
class AugmentationMap:
    """Everything needed to strip a result and rebuild the decomposition."""

    n_continuous_orig: int
    n_integer: int
    n_outputs_orig: int
    n_aux: int
    alpha: float
    g_z_tier: float
    sides: Tuple[_SideMap, ...]
    problem_orig: MIQPProblem


def augment_problem(
    problem: MIQPProblem,
    specs: Sequence[AreaTierSpec],
    *,
    g_z_tier: float,
) -> Tuple[MIQPProblem, AugmentationMap]:
    """Return the augmented problem plus the stripping map.

    The input problem must be in the layout produced by
    ``build_miqp_problem`` (continuous block first, integer block last);
    auxiliary segment variables are appended to the continuous block,
    tier rows to the outputs.  The original problem object is not
    modified.
    """
    if not specs:
        rep1("augment_problem called with no specs — the neutral path "
             "must bypass augmentation entirely (R1)")
    if g_z_tier <= 0.0 or not math.isfinite(g_z_tier):
        rep1("g_z_tier must be positive and finite", g_z_tier=g_z_tier)
    n_c0 = problem.n_continuous
    n_i = problem.n_integer
    n_t0 = n_c0 + n_i
    n_o0 = problem.n_outputs
    alpha = problem.alpha
    if list(problem.integer_indices) != list(range(n_c0, n_t0)):
        rep1("expected build_miqp_problem layout (continuous first, "
             "integer block last)",
             integer_indices=tuple(problem.integer_indices),
             n_continuous=n_c0, n_total=n_t0)
    seen_rows: set = set()
    for spec in specs:
        for r in spec.pcc_output_rows:
            if r >= n_o0:
                rep1("PCC output row beyond the problem's outputs",
                     area_id=spec.area_id, row=r, n_outputs=n_o0)
            if r in seen_rows:
                rep1("PCC output row claimed by two AggregationAreas",
                     area_id=spec.area_id, row=r)
            seen_rows.add(r)

    sides: List[_SideMap] = []
    aux_widths: List[float] = []
    aux_slopes: List[float] = []
    for spec in specs:
        for side in (spec.raising, spec.lowering):
            start = len(aux_widths)
            for seg in side.segments:
                aux_widths.append(seg.width_mvar)
                aux_slopes.append(seg.slope_obj_per_mvar)
            sides.append(_SideMap(
                area_id=spec.area_id,
                direction=side.direction,
                sign=side.direction.q_hv_sign,
                pcc_rows=spec.pcc_output_rows,
                anchor_mvar=side.anchor_mvar,
                row_index=n_o0 + len(sides),
                aux_start=n_c0 + start,
                n_seg=len(side.segments),
                slopes=tuple(s.slope_obj_per_mvar for s in side.segments),
                widths=tuple(s.width_mvar for s in side.segments),
            ))
    n_aux = len(aux_widths)
    n_rows = len(sides)
    n_c1 = n_c0 + n_aux
    n_t1 = n_t0 + n_aux
    n_o1 = n_o0 + n_rows

    # --- H_tilde: insert zero aux columns, append tier rows ---
    H0 = problem.H_tilde
    H1 = np.zeros((n_o1, n_t1), dtype=np.float64)
    H1[:n_o0, :n_c0] = H0[:, :n_c0]
    H1[:n_o0, n_c1:] = H0[:, n_c0:]
    for sm in sides:
        row = np.zeros(n_t1, dtype=np.float64)
        for r in sm.pcc_rows:
            row[:n_c0] += sm.sign * H0[r, :n_c0]
            row[n_c1:] += sm.sign * H0[r, n_c0:]
        row[sm.aux_start:sm.aux_start + sm.n_seg] = -1.0
        H1[sm.row_index, :] = row

    # --- y vectors ---
    y_cur1 = np.concatenate([
        problem.y_current,
        [sum(float(sm.sign) * float(problem.y_current[r])
             for r in sm.pcc_rows) for sm in sides],
    ])
    y_lo1 = np.concatenate([
        problem.y_lower, np.full(n_rows, ROW_LOWER_MVAR),
    ])
    y_up1 = np.concatenate([
        problem.y_upper, [sm.anchor_mvar for sm in sides],
    ])

    # --- u vectors and gradient ---
    u_cur1 = np.concatenate([
        problem.u_current[:n_c0], np.zeros(n_aux),
        problem.u_current[n_c0:],
    ])
    u_lo1 = np.concatenate([
        problem.u_lower[:n_c0], np.zeros(n_aux), problem.u_lower[n_c0:],
    ])
    u_up1 = np.concatenate([
        problem.u_upper[:n_c0], np.asarray(aux_widths, dtype=np.float64),
        problem.u_upper[n_c0:],
    ])
    # w_aux = x/α  →  grad entry slope·α so that grad·w = slope·x.
    grad1 = np.concatenate([
        problem.grad_f[:n_c0],
        alpha * np.asarray(aux_slopes, dtype=np.float64),
        problem.grad_f[n_c0:],
    ])

    # --- weights: zero quadratic damping on aux (pure linear price) ---
    G_w1 = np.zeros((n_t1, n_t1), dtype=np.float64)
    G_w0 = problem.G_w
    G_w1[:n_c0, :n_c0] = G_w0[:n_c0, :n_c0]
    G_w1[:n_c0, n_c1:] = G_w0[:n_c0, n_c0:]
    G_w1[n_c1:, :n_c0] = G_w0[n_c0:, :n_c0]
    G_w1[n_c1:, n_c1:] = G_w0[n_c0:, n_c0:]
    G_z1 = np.diag(np.concatenate([
        np.diag(problem.G_z), np.full(n_rows, g_z_tier),
    ]))

    augmented = MIQPProblem(
        n_continuous=n_c1,
        n_integer=n_i,
        n_outputs=n_o1,
        alpha=alpha,
        G_w=G_w1,
        G_z=G_z1,
        grad_f=grad1,
        H_tilde=H1,
        u_current=u_cur1,
        u_lower=u_lo1,
        u_upper=u_up1,
        y_current=y_cur1,
        y_lower=y_lo1,
        y_upper=y_up1,
        integer_indices=list(range(n_c1, n_t1)),
    )
    amap = AugmentationMap(
        n_continuous_orig=n_c0,
        n_integer=n_i,
        n_outputs_orig=n_o0,
        n_aux=n_aux,
        alpha=alpha,
        g_z_tier=g_z_tier,
        sides=tuple(sides),
        problem_orig=problem,
    )
    return augmented, amap


def strip_result(
    result: MIQPResult,
    amap: AugmentationMap,
) -> Tuple[MIQPResult, List[TierDecomposition]]:
    """Strip an augmented result back to the original problem shape and
    return the tier decomposition (plan §5 reconstruction, asserted)."""
    n_c0 = amap.n_continuous_orig
    n_o0 = amap.n_outputs_orig
    if not result.is_feasible:
        # Preserve the failure verbatim in the original shape; the
        # controller raises on it (solver status asserted, plan §5).
        return MIQPResult(
            w_continuous=np.zeros(n_c0),
            w_integer=np.zeros(amap.n_integer, dtype=np.int64),
            z=np.zeros(n_o0),
            objective_value=result.objective_value,
            status=result.status,
            solve_time_s=result.solve_time_s,
        ), []

    if len(result.w_continuous) != n_c0 + amap.n_aux:
        rep1("augmented result has unexpected continuous length",
             got=len(result.w_continuous), expected=n_c0 + amap.n_aux)
    w_c0 = np.asarray(result.w_continuous[:n_c0], dtype=np.float64)
    w_aux = np.asarray(result.w_continuous[n_c0:], dtype=np.float64)
    x_all = amap.alpha * w_aux
    z0 = np.asarray(result.z[:n_o0], dtype=np.float64)
    z_tier = np.asarray(result.z[n_o0:], dtype=np.float64)

    # Predicted post-step outputs of the ORIGINAL problem.
    p0 = amap.problem_orig
    dy0 = (amap.alpha * (p0.H_tilde[:, :n_c0] @ w_c0)
           + p0.H_tilde[:, n_c0:] @ result.w_integer.astype(np.float64))
    y_pred0 = p0.y_current + dy0

    decompositions: List[TierDecomposition] = []
    for k, sm in enumerate(amap.sides):
        x = x_all[sm.aux_start - n_c0:sm.aux_start - n_c0 + sm.n_seg]
        if np.any(x < -RECON_TOL_MVAR) or \
                np.any(x > np.asarray(sm.widths) + RECON_TOL_MVAR):
            rep1("segment value outside its bounds (plan §5 assertion)",
                 area_id=sm.area_id, direction=sm.direction,
                 x_mvar=tuple(float(v) for v in x), widths=sm.widths)
        q_signed = float(sum(y_pred0[r] for r in sm.pcc_rows))
        s_coord = sm.sign * q_signed
        excess = max(0.0, s_coord - sm.anchor_mvar)
        slack = float(z_tier[k])
        fill = float(np.sum(x)) + slack
        if abs(excess - fill) > RECON_TOL_MVAR:
            rep1("tier reconstruction failed: excess beyond the anchor "
                 "does not match segment fill + slack (plan §5 "
                 "'q_band + q_vh + q_ug == q_pcc within solver "
                 "tolerance')", area_id=sm.area_id,
                 direction=sm.direction, excess_mvar=excess,
                 x_sum_mvar=float(np.sum(x)), slack_mvar=slack,
                 anchor_mvar=sm.anchor_mvar, s_coord_mvar=s_coord)
        max_slope = max(sm.slopes)
        if slack > max_slope / (2.0 * amap.g_z_tier) + RECON_TOL_MVAR:
            rep1("tier-row slack beyond the quadratic-crossover bound — "
                 "g_z_tier is too small relative to the tier prices",
                 area_id=sm.area_id, direction=sm.direction,
                 slack_mvar=slack, max_slope=max_slope)
        cost = float(np.dot(np.asarray(sm.slopes), x))
        decompositions.append(TierDecomposition(
            area_id=sm.area_id,
            direction=sm.direction,
            q_netted_signed_mvar=q_signed,
            excess_mvar=excess,
            x_mvar=tuple(float(v) for v in x),
            slack_mvar=slack,
            cost_obj=cost,
        ))

    stripped = MIQPResult(
        w_continuous=w_c0,
        w_integer=result.w_integer,
        z=z0,
        objective_value=result.objective_value,
        status=result.status,
        solve_time_s=result.solve_time_s,
    )
    return stripped, decompositions


class PricingSolver:
    """Solver proxy adding the SBX-V tier cost layer (plan §5).

    Wraps an existing ``MIQPSolver`` INSTANCE; installed from outside
    via ``controller.solver = PricingSolver(controller.solver, provider,
    g_z_tier=...)`` — no protected module is modified (hard rule 5).

    ``spec_provider`` is called once per solve and returns the currently
    active tier specification (``None`` or empty → NEUTRAL: the problem
    is passed through UNTOUCHED, byte-identical to the unwrapped solver;
    regression R1).  Specs change only at window commit instants
    (Phase 4 wiring; regression R3).

    The decompositions of the most recent priced solve are kept in
    :attr:`last_decompositions` for logging/metering.  Note that the
    frozen-integer companion solve of the BME discrete-hygiene gate also
    passes through here; consumers must read the attribute once per
    controller step, immediately after it.
    """

    def __init__(
        self,
        inner,
        spec_provider: Callable[[], Optional[Sequence[AreaTierSpec]]],
        *,
        g_z_tier: float,
    ) -> None:
        if g_z_tier <= 0.0 or not math.isfinite(g_z_tier):
            rep1("g_z_tier must be positive and finite", g_z_tier=g_z_tier)
        self._inner = inner
        self._spec_provider = spec_provider
        self._g_z_tier = float(g_z_tier)
        self.last_decompositions: Optional[List[TierDecomposition]] = None

    def solve(self, problem: MIQPProblem) -> MIQPResult:
        specs = self._spec_provider()
        if not specs:
            # NEUTRAL bypass: identical problem object, single inner
            # solve — R1 byte-identity by construction.
            return self._inner.solve(problem)
        augmented, amap = augment_problem(
            problem, specs, g_z_tier=self._g_z_tier)
        res = self._inner.solve(augmented)
        stripped, decomp = strip_result(res, amap)
        if decomp:
            self.last_decompositions = decomp
        return stripped

    def __getattr__(self, name: str):
        # Delegate everything else (verbose flags, limits, ...) to the
        # wrapped solver so the proxy is a drop-in replacement.
        return getattr(self._inner, name)
