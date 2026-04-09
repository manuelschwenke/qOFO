"""
PSO-based joint tuning of g_w for the multi-TSO + cascaded DSO
================================================================

A single Particle Swarm Optimisation entry point that tunes:

* per-TSO-actuator regulariser ``g_w`` (DER, PCC, V_gen, OLTC),
* per-DSO-actuator regulariser ``g_w`` (DER, OLTC, shunt).

The step-size parameter alpha has been removed (absorbed into g_w).

Fitness (minimised) is the **min-max convergence rate** across the
stability components

    f(x) = max( max_i rho_i^TSO,  max_d rho_d^TSO_period )

* ``f < 1``  means all stability conditions hold,
* ``f -> 0``  means fast contraction in every layer.

The PSO is hand-rolled (standard inertia-weight scheme) so the dependency
surface stays unchanged.  Particle 0 is seeded from the existing
Gershgorin warm start, which guarantees PSO >= the legacy tuner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from analysis.stability_analysis import (
    MultiZoneStabilityResult,
    analyse_multi_zone_stability,
)
from analysis.tune_ofo_params import (
    ZoneTuningResult,
    _DSO_COLUMN_ORDER,
    _TSO_COLUMN_ORDER,
    _apply_gw_floors,
    _build_actuator_labels,
    _compute_curvature_diagonal,
    _compute_curvature_matrix,
    _dominant_eigenpair,
    _effective_eigenspectrum,
    _zone_index_ranges,
)


# =============================================================================
#  Public dataclasses
# =============================================================================

@dataclass
class DSOTuneInput:
    """One DSO controller's snapshot for the PSO tuner.

    Attributes
    ----------
    dso_id : str
        Identifier matching ``dso_controllers`` keys in the runner.
    H : NDArray
        DSO sensitivity matrix (rows = outputs, cols = actuators).  The
        column order must match ``[DER | OLTC | shunt]`` (the DSO
        convention).
    q_obj_diag : NDArray
        Per-output objective weights (length = H.shape[0]).
    n_der, n_oltc, n_shunt : int
        Actuator counts in each block (must sum to H.shape[1]).
    """
    dso_id: str
    H: NDArray[np.float64]
    q_obj_diag: NDArray[np.float64]
    n_der: int
    n_oltc: int
    n_shunt: int


@dataclass
class PSOTuningResult:
    """Joint TSO + DSO PSO tuning result."""
    tso_zones: List[ZoneTuningResult]
    dso: Dict[str, Tuple[NDArray[np.float64], float, float]]
    """Per-DSO ``(g_w, rho_D, cascade_margin)``."""
    fitness: float
    """Final min-max objective value."""
    converged: bool
    """True iff fitness < 1.0 (all three stability conditions met)."""
    iterations: int
    history: List[Dict[str, float]] = field(default_factory=list)
    feasibility_warnings: List[str] = field(default_factory=list)
    stability_result: Optional[MultiZoneStabilityResult] = None
    swarm_size: int = 0
    warm_start_fitness: float = float("inf")
    """Fitness of the Gershgorin warm-start particle (regression guard)."""


# =============================================================================
#  Internal helpers
# =============================================================================

def _dso_cascade_decay(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    g_w: NDArray[np.float64],
    n_inner: int,
) -> Tuple[float, float, float]:
    """Return ``(rho_D, cascade_decay, alpha_opt)`` for one DSO.

    Replicates the cascade-margin maths in ``tune_dso`` (Phase 1) but as a
    pure function so the PSO does not depend on the full tuning machinery.

    * ``rho_D = (lam_max - lam_min) / (lam_max + lam_min)`` of the active
      eigenspectrum of M_dso = G_w^{-1/2} C G_w^{-1/2}.
    * ``cascade_decay = rho_D ** n_inner`` -- the contraction achieved
      across one TSO period (n_inner = T_TSO / T_DSO).  Stable iff < 1;
      ``cascade_margin = 1 - cascade_decay``.
    * ``alpha_opt = 2 / (lam_min + lam_max)`` -- optimal DSO step size.

    For degenerate spectra (no active mode, single active mode) the
    function returns conservative defaults that do not contribute to the
    PSO bottleneck.
    """
    C = _compute_curvature_matrix(H, q_obj_diag)
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M = (gw_inv_sqrt[:, None] * C) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M)

    if len(active) == 0:
        return 0.0, 0.0, 1.0
    if len(active) == 1:
        l = float(active[0])
        return 0.0, 0.0, 1.0 / l if l > 1e-14 else 1.0

    l_min, l_max = float(active[0]), float(active[-1])
    rho_D = (l_max - l_min) / (l_max + l_min)
    cascade_decay = rho_D ** max(int(n_inner), 1)
    alpha_opt = 2.0 / (l_min + l_max)
    return rho_D, cascade_decay, alpha_opt


def _dso_actual_decay(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    g_w: NDArray[np.float64],
    n_inner: int,
) -> Tuple[float, float, float]:
    """Return ``(rho_d, cascade_decay, lam_max)`` with alpha=1.

    Evaluates the contraction rate ``rho_d = max_l |1 - lambda_l(M_dso)|``
    that the controller will actually realise.  ``cascade_decay = rho_d
    ** n_inner`` is the contraction over one TSO period.

    Returns ``(rho_d, cascade_decay, lam_max)``.  ``lam_max`` is needed
    by the caller to enforce the hard stability bound ``lam_max < 2``.
    """
    C = _compute_curvature_matrix(H, q_obj_diag)
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M = (gw_inv_sqrt[:, None] * C) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M)

    if len(active) == 0:
        return 0.0, 0.0, 0.0

    l_max = float(active[-1])
    l_min = float(active[0]) if len(active) >= 2 else l_max
    # |1 - lambda| is maximised at one of the spectrum endpoints.
    rho_d = max(abs(1.0 - l_max), abs(1.0 - l_min))
    cascade_decay = rho_d ** max(int(n_inner), 1)
    return rho_d, cascade_decay, l_max


# ---------------------------------------------------------------------------
#  Variable layout helpers
# ---------------------------------------------------------------------------

def _tso_block_size(counts: Dict[str, int]) -> int:
    return (
        int(counts.get('n_der', 0))
        + int(counts.get('n_pcc', 0))
        + int(counts.get('n_gen', 0))
        + int(counts.get('n_oltc', 0))
    )


def _dso_block_size(d: DSOTuneInput) -> int:
    return int(d.n_der) + int(d.n_oltc) + int(d.n_shunt)


def _tso_floor_vector(
    counts: Dict[str, int],
    floors: Dict[str, float],
) -> NDArray[np.float64]:
    """Per-actuator floor vector laid out in TSO column order."""
    parts = []
    for type_name in _TSO_COLUMN_ORDER:
        n = int(counts.get(f'n_{type_name}', 0))
        if n <= 0:
            continue
        parts.append(np.full(n, float(floors.get(type_name, 0.0))))
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)


def _dso_floor_vector(
    d: DSOTuneInput,
    floors: Dict[str, float],
) -> NDArray[np.float64]:
    """Per-actuator floor vector laid out in DSO column order."""
    parts = []
    for type_name in _DSO_COLUMN_ORDER:
        n = int(getattr(d, f'n_{type_name}', 0))
        if n <= 0:
            continue
        parts.append(np.full(n, float(floors.get(type_name, 0.0))))
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)


def _layout(
    actuator_counts: List[Dict[str, int]],
    dso_inputs: List[DSOTuneInput],
) -> Tuple[int, int, int, List[int], List[int]]:
    """Compute the PSO vector layout.

    Returns
    -------
    n_zones, n_dsos, total_dim, tso_block_sizes, dso_block_sizes
    """
    n_zones = len(actuator_counts)
    n_dsos = len(dso_inputs)
    tso_sizes = [_tso_block_size(c) for c in actuator_counts]
    dso_sizes = [_dso_block_size(d) for d in dso_inputs]
    total = sum(tso_sizes) + sum(dso_sizes)
    return n_zones, n_dsos, total, tso_sizes, dso_sizes


def _decode(
    x: NDArray[np.float64],
    n_zones: int,
    n_dsos: int,
    tso_sizes: List[int],
    dso_sizes: List[int],
) -> Tuple[
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
]:
    """Decode a PSO particle into ``(gw_TSO, gw_DSO)``.

    Vector layout::

        [ log10 g_w (TSO zone 1) | ... | log10 g_w (TSO zone Z) |
          log10 g_w (DSO 1)      | ... | log10 g_w (DSO D)         ]
    """
    off = 0

    gw_tso: List[NDArray[np.float64]] = []
    for size in tso_sizes:
        gw_tso.append(np.power(10.0, x[off:off + size]))
        off += size

    gw_dso: List[NDArray[np.float64]] = []
    for size in dso_sizes:
        gw_dso.append(np.power(10.0, x[off:off + size]))
        off += size

    return gw_tso, gw_dso


def _build_bounds(
    actuator_counts: List[Dict[str, int]],
    dso_inputs: List[DSOTuneInput],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    g_w_upper_factor: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct full lower/upper bounds for the PSO vector.

    g_w entries are stored in log10 space.
    """
    lb_parts: List[NDArray[np.float64]] = []
    ub_parts: List[NDArray[np.float64]] = []

    log_factor = float(np.log10(max(g_w_upper_factor, 1.0)))

    # log10 g_w (TSO)
    for counts in actuator_counts:
        floor_vec = _tso_floor_vector(counts, floors_tso)
        log_lo = np.log10(np.maximum(floor_vec, 1e-30))
        lb_parts.append(log_lo)
        ub_parts.append(log_lo + log_factor)

    # log10 g_w (DSO)
    for d in dso_inputs:
        floor_vec = _dso_floor_vector(d, floors_dso)
        log_lo = np.log10(np.maximum(floor_vec, 1e-30))
        lb_parts.append(log_lo)
        ub_parts.append(log_lo + log_factor)

    return np.concatenate(lb_parts), np.concatenate(ub_parts)


def _eigenvector_pump(
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    dso_inputs: List[DSOTuneInput],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    safety_factor_tso: float,
    safety_factor_dso: float,
    n_inner: int,
    spectral_target: float,
    max_pump_iters: int = 50,
    verbose: bool = False,
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
    """Deterministic eigenvector-directed g_w tuner (TSO + DSO).

    Returns ``(gw_tso_list, gw_dso_list)`` — the tuned per-actuator
    weight arrays for every TSO zone and every DSO controller.

    **Phase 1 — Gershgorin preconditioning** (per-zone, per-DSO):
    ``g_w = safety · diag(C) / 2`` independently for each zone/DSO.

    **Phase 2a — TSO global pump**:
    Repeatedly evaluates ``M_sys``, finds its dominant eigenvector, and
    boosts g_w of the participating actuators until
    ``λ_max(M_sys) < 0.95 · spectral_target``.

    **Phase 2b — Per-DSO pump**:
    For each DSO whose ``λ_max(M_dso) ≥ 2``, boosts g_w along its
    dominant eigenvector until ``λ_max(M_dso) < 1.8``.

    This deterministic pump converges in ~5 iterations each (milliseconds
    total) and produces a fully feasible g_w set.  The PSO in
    ``tune_pso_all`` is an optional refinement step on top.
    """
    n_zones = len(zone_ids)

    # ── Phase 1: Gershgorin per-zone preconditioning ────────────────────
    gw_tso_arrays: List[NDArray[np.float64]] = []
    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        c_diag = _compute_curvature_diagonal(H_ii, Q_obj_list[idx])
        gw = safety_factor_tso * c_diag / 2.0
        gw = _apply_gw_floors(
            gw, actuator_counts[idx], floors_tso, _TSO_COLUMN_ORDER,
        )
        gw_tso_arrays.append(gw)

    gw_dso_arrays: List[NDArray[np.float64]] = []
    for d in dso_inputs:
        c_diag = _compute_curvature_diagonal(d.H, d.q_obj_diag)
        counts = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        gw = safety_factor_dso * c_diag / 2.0
        gw = _apply_gw_floors(gw, counts, floors_dso, _DSO_COLUMN_ORDER)
        gw_dso_arrays.append(gw)

    # ── Phase 2a: TSO global pump ───────────────────────────────────────
    target_inner = 0.95 * float(spectral_target)
    n_per_zone = [len(gw) for gw in gw_tso_arrays]
    zone_ranges = _zone_index_ranges(n_per_zone)

    lam_sys = float('inf')
    for pump_iter in range(max_pump_iters):
        stab = analyse_multi_zone_stability(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            G_w_list=gw_tso_arrays,
            zone_ids=zone_ids,
            actuator_counts=actuator_counts,
            verbose=False,
        )
        lam_sys = float(stab.M_sys_lambda_max)
        if verbose:
            print(f"  [pump TSO] iter {pump_iter}: "
                  f"lam_sys = {lam_sys:.4f}  (target < {target_inner:.2f})")
        if lam_sys < target_inner:
            break

        _, v_sys = _dominant_eigenpair(stab.M_sys)
        v_sq = v_sys * v_sys
        for k in range(n_zones):
            lo, hi = zone_ranges[k]
            v_zone = v_sq[lo:hi]
            v_zone_max = float(np.max(v_zone)) if v_zone.size else 0.0
            if v_zone_max < 1e-14:
                continue
            scale = 1.0 + 2.0 * (v_zone / v_zone_max)
            gw_tso_arrays[k] = _apply_gw_floors(
                gw_tso_arrays[k] * scale,
                actuator_counts[k],
                floors_tso, _TSO_COLUMN_ORDER,
            )

    if verbose and lam_sys >= target_inner:
        print(f"  [pump TSO] did not reach target after "
              f"{max_pump_iters} iters (lam_sys = {lam_sys:.4f})")

    # ── Phase 2b: Per-DSO pump ──────────────────────────────────────────
    for d_idx, d in enumerate(dso_inputs):
        counts = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        for dso_pump in range(max_pump_iters):
            C_d = _compute_curvature_matrix(d.H, d.q_obj_diag)
            gw_inv = 1.0 / np.sqrt(np.maximum(gw_dso_arrays[d_idx], 1e-12))
            M_d = (gw_inv[:, None] * C_d) * gw_inv[None, :]
            eigs_d = np.linalg.eigvalsh(M_d)
            lam_max_d = float(eigs_d[-1]) if len(eigs_d) > 0 else 0.0
            if verbose:
                print(f"  [pump DSO {d.dso_id}] iter {dso_pump}: "
                      f"lam_max = {lam_max_d:.4f}")
            if lam_max_d < 1.8:
                break
            # Boost along dominant eigenvector
            _, vecs_d = np.linalg.eigh(M_d)
            v_d = vecs_d[:, -1] ** 2
            v_max_d = float(np.max(v_d)) if v_d.size else 0.0
            if v_max_d < 1e-14:
                break
            scale_d = 1.0 + 2.0 * (v_d / v_max_d)
            gw_dso_arrays[d_idx] = _apply_gw_floors(
                gw_dso_arrays[d_idx] * scale_d,
                counts, floors_dso, _DSO_COLUMN_ORDER,
            )

    return gw_tso_arrays, gw_dso_arrays


# ---------------------------------------------------------------------------
#  Fitness
# ---------------------------------------------------------------------------

_INTEGER_ACTUATOR_TYPES = frozenset({'oltc', 'shunt'})


def _oltc_penalty(
    gw_tso: List[NDArray[np.float64]],
    gw_dso: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    dso_inputs: List[DSOTuneInput],
    gw_oltc_ref_tso: float,
    gw_oltc_ref_dso: float,
    penalty_weight_tso: float,
    penalty_weight_dso: float,
) -> float:
    """Soft penalty for OLTC g_w exceeding reference values.

    Separate reference and weight for TSO and DSO OLTCs.  DSO OLTCs
    (coupling transformers) need a stronger penalty because low g_w
    lets them switch taps wildly, destabilising the DSO voltage.

    Returns a non-negative additive penalty term.
    """
    total_penalty = 0.0

    # TSO OLTCs
    if gw_oltc_ref_tso > 0 and penalty_weight_tso > 0:
        tso_ratios: List[float] = []
        for idx, counts in enumerate(actuator_counts):
            n_oltc = int(counts.get('n_oltc', 0))
            if n_oltc <= 0:
                continue
            off = (int(counts.get('n_der', 0))
                   + int(counts.get('n_pcc', 0))
                   + int(counts.get('n_gen', 0)))
            for k in range(n_oltc):
                gw_val = float(gw_tso[idx][off + k])
                if gw_val < gw_oltc_ref_tso:
                    # ALSO penalise g_w BELOW the reference (too aggressive)
                    tso_ratios.append(np.log10(gw_oltc_ref_tso / gw_val))
                elif gw_val > gw_oltc_ref_tso * 10:
                    # Penalise g_w more than 1 decade above reference (frozen)
                    tso_ratios.append(np.log10(gw_val / (gw_oltc_ref_tso * 10)))
        if tso_ratios:
            total_penalty += penalty_weight_tso * float(np.mean(tso_ratios))

    # DSO OLTCs (coupling transformers) — stronger penalty
    if gw_oltc_ref_dso > 0 and penalty_weight_dso > 0:
        dso_ratios: List[float] = []
        for d_idx, d in enumerate(dso_inputs):
            n_oltc = int(d.n_oltc)
            if n_oltc <= 0:
                continue
            off = int(d.n_der)
            for k in range(n_oltc):
                gw_val = float(gw_dso[d_idx][off + k])
                if gw_val < gw_oltc_ref_dso:
                    dso_ratios.append(np.log10(gw_oltc_ref_dso / gw_val))
                elif gw_val > gw_oltc_ref_dso * 10:
                    dso_ratios.append(np.log10(gw_val / (gw_oltc_ref_dso * 10)))
        if dso_ratios:
            total_penalty += penalty_weight_dso * float(np.mean(dso_ratios))

    return total_penalty


def _evaluate_fitness(
    x: NDArray[np.float64],
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    dso_inputs: List[DSOTuneInput],
    n_zones: int,
    n_dsos: int,
    tso_sizes: List[int],
    dso_sizes: List[int],
    n_inner: int,
    spectral_target: float = 1.9,
    gw_oltc_ref_tso: float = 50.0,
    gw_oltc_ref_dso: float = 40.0,
    oltc_penalty_weight_tso: float = 0.01,
    oltc_penalty_weight_dso: float = 0.05,
) -> Tuple[float, Dict[str, float], Optional[MultiZoneStabilityResult]]:
    """Compute the contraction-rate fitness for one PSO particle.

    Objective (minimised):

        f(x) = max( max_i rho_i^TSO,  max_d rho_d^TSO_period )

    where rho_i^TSO = max_l |1 - alpha_i * lambda_l(M_ii)| is the per-zone
    TSO contraction rate at the chosen alpha (computed inside
    ``analyse_multi_zone_stability``), and rho_d^TSO_period = rho_d ** n_inner
    is the per-DSO contraction across one TSO period at the chosen DSO alpha.

    f < 1 iff every layer strictly contracts; smaller f → faster overall
    convergence.  The objective rewards the *actual* contraction rate, so
    PSO is incentivised to pick the textbook optimal alpha for each layer
    (alpha* = 2/(lam_min+lam_max)) instead of collapsing alpha to its lower
    bound to chase a row-sum bound that may be structurally infeasible.

    Hard stability constraints (any violation → infeasibility penalty):

    1. Global TSO spectral bound: lam_max(M_sys) < 2
       (necessary-and-sufficient for the closed-loop TSO iteration).
    2. Per-DSO local bound: lam_max(M_dso,d) < 2 for every DSO d.

    The conservative row-sum gamma is *not* in the fitness; it is reported
    in the metrics dict for diagnostics only.

    Returns ``(fitness, metrics_dict, stability_result_or_None)``.
    """
    # Default metrics dict layout -- every return path returns the same keys
    # so the convergence history has a uniform schema.
    metrics: Dict[str, float] = {
        'fitness':       100.0,  # large but not 1e6; smooth penalty scale
        'rho_max_tso':   1.0,
        'rho_max_dso':   1.0,
        'max_dso_decay': 1.0,
        'spectral':      float('nan'),
        'gamma':         float('nan'),
    }

    try:
        gw_tso, gw_dso = _decode(
            x, n_zones, n_dsos, tso_sizes, dso_sizes,
        )

        # TSO block: full multi-zone stability call (cheap, ~ms).
        stab = analyse_multi_zone_stability(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            G_w_list=gw_tso,
            zone_ids=zone_ids,
            actuator_counts=actuator_counts,
            verbose=False,
        )
        lam_sys = float(stab.M_sys_lambda_max)
        spectral = lam_sys   # alpha=1
        gamma = float(stab.small_gain_gamma)  # diagnostic only
        metrics['spectral'] = spectral
        metrics['gamma'] = gamma

        # ── Soft constraint #1: TSO spectral bound ─────────────────────
        # Instead of a hard 1e6 cliff, use a smooth quadratic penalty
        # that starts at 1.0 right at the spectral boundary:
        #   f_penalty = 1.0 + (spectral/target - 1)^2
        # This lets PSO particles near the boundary navigate smoothly
        # back to feasibility instead of being repelled by a cliff.
        if not np.isfinite(spectral) or spectral >= spectral_target:
            excess = (spectral / max(spectral_target, 1e-12)) - 1.0
            penalty = 1.0 + excess * excess
            if not np.isfinite(penalty):
                penalty = 1e6
            metrics['fitness'] = penalty
            return penalty, metrics, stab

        # ── TSO condition number κ(M_sys) = λ_max / λ_min ────────────────
        # This is the PRIMARY objective.  Lower κ → the optimal contraction
        # rate ρ* = (κ−1)/(κ+1) is tighter → faster convergence.
        # Use the ACTIVE eigenvalues of M_sys (after null-space filtering).
        sys_eigs = stab.M_sys_eigenvalues
        sys_eigs_active = sys_eigs[sys_eigs > 1e-10 * max(float(sys_eigs[-1]), 1e-14)]
        if len(sys_eigs_active) >= 2:
            kappa_sys = float(sys_eigs_active[-1] / sys_eigs_active[0])
        elif len(sys_eigs_active) == 1:
            kappa_sys = 1.0
        else:
            kappa_sys = 1e6  # degenerate

        rho_opt_sys = (kappa_sys - 1.0) / (kappa_sys + 1.0)
        metrics['kappa_sys'] = kappa_sys
        metrics['rho_opt_sys'] = rho_opt_sys

        # Per-zone contraction for diagnostics
        rho_max_tso = max(
            (float(zr.rho_i) for zr in stab.zones),
            default=0.0,
        )
        metrics['rho_max_tso'] = rho_max_tso

        # ── DSO condition numbers ───────────────────────────────────────
        max_kappa_dso = 1.0
        max_dso_decay = 0.0
        rho_max_dso = 0.0
        for d_idx, d in enumerate(dso_inputs):
            C_d = _compute_curvature_matrix(d.H, d.q_obj_diag)
            gw_inv = 1.0 / np.sqrt(np.maximum(gw_dso[d_idx], 1e-12))
            M_d = (gw_inv[:, None] * C_d) * gw_inv[None, :]
            eigs_d_all = np.linalg.eigvalsh(M_d)
            eigs_d = eigs_d_all[eigs_d_all > 1e-10 * max(float(eigs_d_all[-1]), 1e-14)]

            if len(eigs_d) == 0:
                continue

            lam_max_d = float(eigs_d[-1])
            lam_min_d = float(eigs_d[0]) if len(eigs_d) >= 2 else lam_max_d

            # Soft stability constraint on DSO
            if lam_max_d >= 2.0:
                excess = (lam_max_d / 2.0) - 1.0
                penalty = 1e3 + excess * excess  # large penalty, well above any κ
                metrics['fitness'] = penalty
                return penalty, metrics, stab

            kappa_d = lam_max_d / max(lam_min_d, 1e-14) if len(eigs_d) >= 2 else 1.0
            rho_d = (kappa_d - 1.0) / (kappa_d + 1.0) if kappa_d > 1.0 else 0.0
            decay_d = rho_d ** max(int(n_inner), 1)

            if kappa_d > max_kappa_dso:
                max_kappa_dso = kappa_d
            if decay_d > max_dso_decay:
                max_dso_decay = decay_d
            if rho_d > rho_max_dso:
                rho_max_dso = rho_d

        metrics['max_kappa_dso'] = max_kappa_dso
        metrics['rho_max_dso'] = rho_max_dso
        metrics['max_dso_decay'] = max_dso_decay

        # ── Combined fitness: worst condition number across layers ───────
        # Use kappa_sys for TSO and max_kappa_dso for DSO.
        # To make them comparable, convert both to optimal rho and take
        # the maximum — this weights the layer that converges slowest.
        rho_opt_dso = (max_kappa_dso - 1.0) / (max_kappa_dso + 1.0) if max_kappa_dso > 1 else 0.0
        # Account for DSO having n_inner steps per TSO period:
        # effective DSO rho over one TSO period = rho_opt_dso^n_inner
        rho_dso_per_tso = rho_opt_dso ** max(int(n_inner), 1)

        base_fitness = float(max(rho_opt_sys, rho_dso_per_tso))

        # Soft penalty for inflated OLTC g_w: steers PSO toward
        # flattening kappa via continuous actuators rather than by
        # freezing OLTCs with huge g_w.
        oltc_pen = _oltc_penalty(
            gw_tso, gw_dso, actuator_counts, dso_inputs,
            gw_oltc_ref_tso, gw_oltc_ref_dso,
            oltc_penalty_weight_tso, oltc_penalty_weight_dso,
        )
        metrics['oltc_penalty'] = oltc_pen

        fitness = base_fitness + oltc_pen
        if not np.isfinite(fitness):
            fitness = 100.0
        metrics['fitness'] = fitness
        return fitness, metrics, stab

    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        metrics['fitness'] = 100.0
        metrics['error'] = str(exc)  # type: ignore[assignment]
        return 100.0, metrics, None


# ---------------------------------------------------------------------------
#  PSO main loop
# ---------------------------------------------------------------------------

def _pso_loop(
    *,
    fitness_fn,
    lb: NDArray[np.float64],
    ub: NDArray[np.float64],
    warm_start: NDArray[np.float64],
    swarm_size: int,
    max_iterations: int,
    w_inertia: Tuple[float, float],
    c_cognitive: float,
    c_social: float,
    velocity_clamp_frac: float,
    rng: np.random.Generator,
    verbose: bool,
) -> Tuple[
    NDArray[np.float64],   # x_best
    float,                 # f_best
    int,                   # iterations executed
    List[Dict[str, float]],
]:
    """Run the inertia-weight PSO and return the best particle."""
    dim = lb.size
    width = ub - lb
    v_max = velocity_clamp_frac * width

    # Initialise positions
    X = np.empty((swarm_size, dim), dtype=np.float64)
    X[0] = warm_start
    if swarm_size > 1:
        # Particle 1: log-uniform random across the full bounds
        X[1] = lb + rng.random(dim) * width
    # Particles 2..N-1: warm start + Gaussian noise in normalised units.
    # sigma = 0.25 of the bound width per dimension in log-g_w space,
    # giving ~1.5-decade spread around the warm start.  Wider spread than
    # the original 0.15 to encourage exploration now that the warm start
    # is feasible (thanks to the feasibility pump).
    if swarm_size > 2:
        for k in range(2, swarm_size):
            noise = rng.normal(scale=0.25, size=dim) * width
            X[k] = warm_start + noise
    np.clip(X, lb, ub, out=X)

    # Initialise velocities ~ uniform in [-v_max, v_max]
    V = (rng.random((swarm_size, dim)) * 2.0 - 1.0) * v_max[None, :]

    # Evaluate initial fitness
    F = np.empty(swarm_size, dtype=np.float64)
    metrics_list: List[Dict[str, float]] = []
    for k in range(swarm_size):
        f, _m, _ = fitness_fn(X[k])
        F[k] = f

    # Personal bests
    P = X.copy()
    Pf = F.copy()

    # Global best
    g_idx = int(np.argmin(Pf))
    g_best = P[g_idx].copy()
    g_best_f = float(Pf[g_idx])

    history: List[Dict[str, float]] = []
    no_improve = 0
    last_best = g_best_f

    w_lo, w_hi = float(w_inertia[1]), float(w_inertia[0])  # decay hi -> lo

    for it in range(1, max_iterations + 1):
        # Linear inertia decay
        w = w_hi - (w_hi - w_lo) * (it - 1) / max(max_iterations - 1, 1)

        r1 = rng.random((swarm_size, dim))
        r2 = rng.random((swarm_size, dim))

        V = (
            w * V
            + c_cognitive * r1 * (P - X)
            + c_social * r2 * (g_best[None, :] - X)
        )
        np.clip(V, -v_max[None, :], v_max[None, :], out=V)

        X = X + V
        # Reflective boundary handling: bounce off the walls and zero
        # the offending velocity component
        below = X < lb[None, :]
        above = X > ub[None, :]
        if below.any():
            X = np.where(below, 2.0 * lb[None, :] - X, X)
            V = np.where(below, -V, V)
        if above.any():
            X = np.where(above, 2.0 * ub[None, :] - X, X)
            V = np.where(above, -V, V)
        np.clip(X, lb, ub, out=X)

        # Evaluate
        for k in range(swarm_size):
            f, _m, _ = fitness_fn(X[k])
            F[k] = f
            if f < Pf[k]:
                Pf[k] = f
                P[k] = X[k]

        g_idx = int(np.argmin(Pf))
        if Pf[g_idx] < g_best_f:
            g_best_f = float(Pf[g_idx])
            g_best = P[g_idx].copy()

        # Detailed metrics for the current global best (cheap: one eval)
        _, m_best, _ = fitness_fn(g_best)
        m_best['iter'] = it
        m_best['inertia'] = w
        history.append(m_best)

        if verbose:
            print(
                f"  [pso] iter {it:3d}: f_best={g_best_f:.4f}  "
                f"kappa_sys={m_best.get('kappa_sys', float('nan')):.1f}  "
                f"rho_opt={m_best.get('rho_opt_sys', float('nan')):.4f}  "
                f"kappa_dso={m_best.get('max_kappa_dso', float('nan')):.1f}  "
                f"spec={m_best.get('spectral', float('nan')):.4f}  "
                f"w={w:.3f}"
            )

        # Early termination: feasible and stalled
        if g_best_f < 0.5:
            if abs(last_best - g_best_f) < 1e-4:
                no_improve += 1
            else:
                no_improve = 0
        last_best = g_best_f
        if no_improve >= 20:
            if verbose:
                print(
                    f"  [pso] early stop at iter {it}: no improvement "
                    f"for 20 iterations and feasible (f_best={g_best_f:.4f})"
                )
            return g_best, g_best_f, it, history

    return g_best, g_best_f, max_iterations, history


# =============================================================================
#  Public entry point
# =============================================================================

def tune_pso_all(
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    dso_inputs: List[DSOTuneInput],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    g_w_upper_factor: float = 1e6,
    swarm_size: int = 30,
    max_iterations: int = 100,
    w_inertia: Tuple[float, float] = (0.7, 0.4),
    c_cognitive: float = 1.5,
    c_social: float = 1.5,
    velocity_clamp_frac: float = 0.2,
    cascade_margin_target: float = 0.3,
    spectral_target: float = 1.9,
    tso_period_s: float = 180.0,
    dso_period_s: float = 60.0,
    legacy_safety_factor_tso: float = 4.0,
    legacy_safety_factor_dso: float = 2.0,
    gw_oltc_ref_tso: float = 50.0,
    gw_oltc_ref_dso: float = 40.0,
    oltc_penalty_weight_tso: float = 0.01,
    oltc_penalty_weight_dso: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> PSOTuningResult:
    """Joint TSO + DSO PSO tuning.

    Parameters
    ----------
    H_blocks, Q_obj_list, actuator_counts, zone_ids :
        TSO sensitivity blocks and per-zone actuator/output metadata,
        identical to ``analyse_multi_zone_stability`` and
        ``tune_multi_zone``.
    dso_inputs :
        One :class:`DSOTuneInput` per DSO controller.
    floors_tso :
        Per-actuator-type lower bounds for the TSO log-g_w search.  Keys
        ``'der'``, ``'pcc'``, ``'gen'``, ``'oltc'``.
    floors_dso :
        Per-actuator-type lower bounds for the DSO log-g_w search.  Keys
        ``'der'``, ``'oltc'``, ``'shunt'``.
    g_w_upper_factor :
        Upper bound multiplier on every g_w entry: ``ub = lb * factor``.
        Default ``1e6``.
    swarm_size, max_iterations :
        PSO budget.  Default 30 × 100 ≈ 3000 fitness evals.
    w_inertia, c_cognitive, c_social, velocity_clamp_frac :
        PSO hyper-parameters.
    cascade_margin_target :
        Reporting threshold; not used as a hard constraint.
    spectral_target :
        Hard constraint headroom on the global TSO spectral bound.
        PSO penalises any particle whose ``lam_max(M_sys)``
        exceeds this.
        Default ``1.8`` (10% margin below the absolute stability bound
        of 2.0).
    tso_period_s, dso_period_s :
        Controller periods.  ``n_inner = T_TSO / T_DSO`` is the number of
        DSO iterations per TSO iteration; sets the cascade-decay exponent.
    legacy_safety_factor_tso, legacy_safety_factor_dso :
        Safety multipliers used by the Gershgorin warm start.  Match the
        defaults of ``tune_multi_zone`` and ``tune_dso``.
    seed :
        ``numpy.random.default_rng`` seed for reproducibility.
    verbose :
        If True, print one line per PSO iteration.

    Returns
    -------
    :class:`PSOTuningResult`.
    """
    if not actuator_counts:
        raise ValueError("actuator_counts must be non-empty")
    if len(actuator_counts) != len(zone_ids):
        raise ValueError(
            "actuator_counts and zone_ids must have the same length"
        )

    n_inner = max(int(round(tso_period_s / max(dso_period_s, 1e-9))), 1)

    n_zones, n_dsos, total_dim, tso_sizes, dso_sizes = _layout(
        actuator_counts, dso_inputs,
    )
    if verbose:
        print(
            f"  [pso] dim = {total_dim}  "
            f"({n_zones} TSO zones, {n_dsos} DSOs, "
            f"n_inner = {n_inner})"
        )

    # ── Step 1: Deterministic eigenvector pump (TSO + DSO) ─────────────
    # The pump is the PRIMARY tuner.  It uses Gershgorin preconditioning
    # followed by iterative eigenvector-directed boosting that converges
    # in ~5 iterations (milliseconds).  The PSO is an OPTIONAL refinement
    # step that only runs if the pump succeeded and there's budget for it.
    gw_tso_pumped, gw_dso_pumped = _eigenvector_pump(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        actuator_counts=actuator_counts,
        zone_ids=zone_ids,
        dso_inputs=dso_inputs,
        floors_tso=floors_tso,
        floors_dso=floors_dso,
        safety_factor_tso=legacy_safety_factor_tso,
        safety_factor_dso=legacy_safety_factor_dso,
        n_inner=n_inner,
        spectral_target=float(spectral_target),
        verbose=verbose,
    )

    # Encode pump result into the flat log-space vector
    pump_vec = np.concatenate(
        [np.log10(np.maximum(gw, 1e-30)) for gw in gw_tso_pumped]
        + [np.log10(np.maximum(gw, 1e-30)) for gw in gw_dso_pumped]
    )

    def fitness_fn(x: NDArray[np.float64]):
        return _evaluate_fitness(
            x,
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            actuator_counts=actuator_counts,
            zone_ids=zone_ids,
            dso_inputs=dso_inputs,
            n_zones=n_zones,
            n_dsos=n_dsos,
            tso_sizes=tso_sizes,
            dso_sizes=dso_sizes,
            n_inner=n_inner,
            spectral_target=float(spectral_target),
            gw_oltc_ref_tso=float(gw_oltc_ref_tso),
            gw_oltc_ref_dso=float(gw_oltc_ref_dso),
            oltc_penalty_weight_tso=float(oltc_penalty_weight_tso),
            oltc_penalty_weight_dso=float(oltc_penalty_weight_dso),
        )

    pump_f, pump_metrics, _ = fitness_fn(pump_vec)
    if verbose:
        print(f"  [pump] fitness = {pump_f:.4f}  "
              f"kappa_sys = {pump_metrics.get('kappa_sys', float('nan')):.1f}  "
              f"rho_opt = {pump_metrics.get('rho_opt_sys', float('nan')):.4f}  "
              f"kappa_dso = {pump_metrics.get('max_kappa_dso', float('nan')):.1f}  "
              f"spec = {pump_metrics.get('spectral', float('nan')):.4f}")

    # ── Step 2: Optional PSO refinement ──────────────────────────────────
    # Only runs if the pump succeeded (fitness < 1) AND max_iterations > 0.
    # Build bounds around the pump result: ±3 decades gives PSO room to
    # explore without drifting into deeply infeasible territory.
    if pump_f < 1.0 and int(max_iterations) > 0:
        lb = pump_vec - 3.0  # 3 decades below pump
        ub = pump_vec + 3.0  # 3 decades above pump
        # Never below the floor-based lower bounds
        lb_floor, _ = _build_bounds(
            actuator_counts=actuator_counts,
            dso_inputs=dso_inputs,
            floors_tso=floors_tso,
            floors_dso=floors_dso,
            g_w_upper_factor=g_w_upper_factor,
        )
        lb = np.maximum(lb, lb_floor)

        rng = np.random.default_rng(seed)
        g_best, g_best_f, iters_done, history = _pso_loop(
            fitness_fn=fitness_fn,
            lb=lb,
            ub=ub,
            warm_start=pump_vec,
            swarm_size=int(swarm_size),
            max_iterations=int(max_iterations),
            w_inertia=w_inertia,
            c_cognitive=float(c_cognitive),
            c_social=float(c_social),
            velocity_clamp_frac=float(velocity_clamp_frac),
            rng=rng,
            verbose=verbose,
        )
        # Regression guard
        if pump_f < g_best_f:
            if verbose:
                print(f"  [pso] PSO worse than pump ({g_best_f:.4f} vs "
                      f"{pump_f:.4f}); keeping pump result")
            g_best = pump_vec
            g_best_f = pump_f
    else:
        # Pump failed or PSO disabled — use pump result directly
        g_best = pump_vec
        g_best_f = pump_f
        iters_done = 0
        history = []
        if verbose and pump_f >= 1.0:
            print(f"  [pump] infeasible (fitness={pump_f:.4f}); "
                  f"skipping PSO refinement")

    # Decode the final solution
    gw_tso, gw_dso = _decode(
        g_best, n_zones, n_dsos, tso_sizes, dso_sizes,
    )

    # Final stability snapshot for diagnostics
    stab_final = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=False,
    )

    # Pack TSO results into the existing ZoneTuningResult shape so the
    # console-printing code in run_M_TSO_M_DSO can stay almost unchanged.
    zone_results: List[ZoneTuningResult] = []
    for idx, zr in enumerate(stab_final.zones):
        zone_results.append(ZoneTuningResult(
            zone_id=zone_ids[idx],
            g_w=gw_tso[idx],
            rho=zr.rho_i,
            kappa=zr.kappa_Mii,
            lambda_min=zr.lambda_min_Mii,
            lambda_max=zr.lambda_max_Mii,
            actuator_labels=_build_actuator_labels(actuator_counts[idx]),
        ))

    # DSO results: rebuild rho/cascade-margin from the final g_w
    dso_results: Dict[str, Tuple[NDArray[np.float64], float, float]] = {}
    feasibility_warnings: List[str] = []
    for d_idx, d in enumerate(dso_inputs):
        rho_d, decay_d, _ = _dso_cascade_decay(
            d.H, d.q_obj_diag, gw_dso[d_idx], n_inner,
        )
        cascade_margin = 1.0 - decay_d
        dso_results[d.dso_id] = (
            gw_dso[d_idx],
            float(rho_d),
            float(cascade_margin),
        )
        if cascade_margin < cascade_margin_target:
            feasibility_warnings.append(
                f"DSO {d.dso_id}: cascade_margin = {cascade_margin:.4f} "
                f"below target {cascade_margin_target:.2f} "
                f"(decay = {decay_d:.4f}, n_inner = {n_inner})"
            )

    # Top-level feasibility checks
    final_spectral = float(stab_final.M_sys_lambda_max)  # alpha=1
    if final_spectral >= 2.0:
        feasibility_warnings.append(
            f"TSO spectral metric lam_sys = {final_spectral:.4f} "
            f">= 2 (UNSTABLE)"
        )
    elif final_spectral >= float(spectral_target):
        feasibility_warnings.append(
            f"TSO spectral metric lam_sys = {final_spectral:.4f} "
            f">= spectral_target {spectral_target:.2f} (no headroom)"
        )
    if stab_final.small_gain_gamma >= 1.0:
        feasibility_warnings.append(
            f"TSO row-sum gamma = {stab_final.small_gain_gamma:.4f} "
            f">= 1 (sufficient bound only; spectral test is what matters)"
        )

    # 'converged' means PSO satisfied all hard constraints AND each layer
    # contracts strictly: max(rho_max_tso, max_dso_decay) < 1, equivalent
    # to fitness < 1 in the new objective.
    pso_converged = bool(g_best_f < 1.0)

    return PSOTuningResult(
        tso_zones=zone_results,
        dso=dso_results,
        fitness=float(g_best_f),
        converged=pso_converged,
        iterations=int(iters_done),
        history=history,
        feasibility_warnings=feasibility_warnings,
        stability_result=stab_final,
        swarm_size=int(swarm_size),
        warm_start_fitness=float(pump_f),
    )
