"""
PSO-based joint tuning of g_w and alpha for the multi-TSO + cascaded DSO
==========================================================================

A single Particle Swarm Optimisation entry point that tunes:

* per-TSO-zone step size ``alpha_i``,
* per-TSO-actuator regulariser ``g_w`` (DER, PCC, V_gen, OLTC),
* per-DSO-controller step size ``alpha_d``,
* per-DSO-actuator regulariser ``g_w`` (DER, OLTC, shunt).

Fitness (minimised) is the **min-max convergence rate** across the three
stability components

    s     = (1/2) * max(alpha_TSO) * lambda_max(M_sys)         # TSO spectral half-rate
    gamma = small_gain_gamma                                     # TSO row-sum
    delta_d = rho_D(d) ** (T_TSO / T_DSO)        per DSO d       # cascade decay

so that

    f(x) = max(s, gamma, max_d delta_d)

* ``f < 1``  ⇔  all three stability conditions hold,
* ``f → 0``  ⇔  fast contraction in every layer.

The PSO is hand-rolled (standard inertia-weight scheme) so the dependency
surface stays unchanged.  Particle 0 is seeded from the existing
Gershgorin warm start, which guarantees PSO ≥ the legacy tuner.
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
    _effective_eigenspectrum,
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
    dso: Dict[str, Tuple[NDArray[np.float64], float, float, float]]
    """Per-DSO ``(g_w, alpha, rho_D, cascade_margin)``."""
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
    alpha: float,
    n_inner: int,
) -> Tuple[float, float, float]:
    """Return ``(rho_d, cascade_decay, lam_max)`` at the **actual** alpha.

    Unlike :func:`_dso_cascade_decay` (which assumes the optimal alpha
    and is used only by the warm start), this function evaluates the
    contraction rate ``rho_d = max_l |1 - alpha * lambda_l(M_dso)|``
    that the controller will actually realise.  ``cascade_decay = rho_d
    ** n_inner`` is the contraction over one TSO period.

    The PSO fitness drives ``rho_d * n_inner`` down, which makes the
    chosen ``alpha`` a real decision variable -- in contrast to the
    legacy ``_dso_cascade_decay`` which ignored alpha entirely.

    Returns ``(rho_d, cascade_decay, lam_max)``.  ``lam_max`` is needed
    by the caller to enforce the hard stability bound
    ``alpha * lam_max < 2``.
    """
    C = _compute_curvature_matrix(H, q_obj_diag)
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M = (gw_inv_sqrt[:, None] * C) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M)

    if len(active) == 0:
        return 0.0, 0.0, 0.0

    l_max = float(active[-1])
    l_min = float(active[0]) if len(active) >= 2 else l_max
    # |1 - alpha*lambda| is maximised at one of the spectrum endpoints.
    rho_d = max(abs(1.0 - alpha * l_max), abs(1.0 - alpha * l_min))
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
    total = n_zones + n_dsos + sum(tso_sizes) + sum(dso_sizes)
    return n_zones, n_dsos, total, tso_sizes, dso_sizes


def _decode(
    x: NDArray[np.float64],
    n_zones: int,
    n_dsos: int,
    tso_sizes: List[int],
    dso_sizes: List[int],
) -> Tuple[
    List[float],
    List[NDArray[np.float64]],
    List[float],
    List[NDArray[np.float64]],
]:
    """Decode a PSO particle into ``(alpha_TSO, gw_TSO, alpha_DSO, gw_DSO)``.

    Vector layout::

        [ alpha_TSO_1..Z |
          alpha_DSO_1..D |
          log10 g_w (TSO zone 1) | ... | log10 g_w (TSO zone Z) |
          log10 g_w (DSO 1)      | ... | log10 g_w (DSO D)         ]
    """
    off = 0
    alpha_tso = [float(x[off + i]) for i in range(n_zones)]
    off += n_zones
    alpha_dso = [float(x[off + i]) for i in range(n_dsos)]
    off += n_dsos

    gw_tso: List[NDArray[np.float64]] = []
    for size in tso_sizes:
        gw_tso.append(np.power(10.0, x[off:off + size]))
        off += size

    gw_dso: List[NDArray[np.float64]] = []
    for size in dso_sizes:
        gw_dso.append(np.power(10.0, x[off:off + size]))
        off += size

    return alpha_tso, gw_tso, alpha_dso, gw_dso


def _build_bounds(
    actuator_counts: List[Dict[str, int]],
    dso_inputs: List[DSOTuneInput],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    g_w_upper_factor: float,
    alpha_bounds: Tuple[float, float],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct full lower/upper bounds for the PSO vector.

    Alpha entries are linear; g_w entries are stored in log10 space.
    """
    n_zones = len(actuator_counts)
    n_dsos = len(dso_inputs)
    a_lo, a_hi = float(alpha_bounds[0]), float(alpha_bounds[1])

    lb_parts: List[NDArray[np.float64]] = []
    ub_parts: List[NDArray[np.float64]] = []

    # alpha_TSO
    lb_parts.append(np.full(n_zones, a_lo))
    ub_parts.append(np.full(n_zones, a_hi))
    # alpha_DSO
    lb_parts.append(np.full(n_dsos, a_lo))
    ub_parts.append(np.full(n_dsos, a_hi))

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


def _encode_warm_start(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    dso_inputs: List[DSOTuneInput],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    legacy_safety_factor_tso: float,
    legacy_safety_factor_dso: float,
    n_inner: int,
    alpha_init_tso: Optional[Dict[int, float]],
    alpha_init_dso: float,
    spectral_target: float,
    lb: NDArray[np.float64],
    ub: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the warm-start particle from the existing Gershgorin path.

    The TSO block reproduces ``compute_optimal_gw`` (Phase 1 of
    ``tune_multi_zone``).  Local-optimal alpha values are derived from
    each zone's M_ii spectrum and then **clamped by the global spectral
    bound** ``alpha_eff * lam_max(M_sys) < spectral_target`` so the warm
    start is feasible by construction (the legacy ``tune_multi_zone``
    enforces the same cap on every iteration).

    The DSO block reproduces Phase 1 of ``tune_dso`` and the per-DSO
    alpha is set to the local optimum.

    This guarantees PSO starts from a point at least as good as what the
    legacy tuner would have produced *and* satisfies the hard fitness
    constraints, so the no-regression invariant of the search holds.
    """
    n_zones = len(zone_ids)
    n_dsos = len(dso_inputs)

    alpha_tso = np.array(
        [float((alpha_init_tso or {}).get(z, 0.1)) for z in zone_ids],
        dtype=np.float64,
    )
    alpha_dso = np.full(n_dsos, float(alpha_init_dso), dtype=np.float64)

    log_gw_tso: List[NDArray[np.float64]] = []
    gw_tso_arrays: List[NDArray[np.float64]] = []
    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        c_diag = _compute_curvature_diagonal(H_ii, Q_obj_list[idx])
        gw = legacy_safety_factor_tso * alpha_tso[idx] * c_diag / 2.0
        gw = _apply_gw_floors(
            gw, actuator_counts[idx], floors_tso, _TSO_COLUMN_ORDER,
        )
        gw_tso_arrays.append(gw)
        # Local-optimal alpha (will be capped against the global bound below)
        C_ii = _compute_curvature_matrix(H_ii, Q_obj_list[idx])
        gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw, 1e-12))
        M_ii = (gw_inv_sqrt[:, None] * C_ii) * gw_inv_sqrt[None, :]
        _, active, _ = _effective_eigenspectrum(M_ii)
        if len(active) >= 2:
            alpha_tso[idx] = 2.0 / (float(active[0]) + float(active[-1]))
        elif len(active) == 1:
            alpha_tso[idx] = 1.0 / float(active[0])

    # Global spectral cap so the warm-start particle is *strictly* feasible
    # under the hard constraint  alpha_eff * lam_max(M_sys) < spectral_target.
    # We aim for 0.95 * spectral_target so the warm start has a small inner
    # margin instead of landing exactly on the constraint boundary.
    try:
        stab_warm = analyse_multi_zone_stability(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            G_w_list=gw_tso_arrays,
            alpha_list=alpha_tso.tolist(),
            zone_ids=zone_ids,
            actuator_counts=actuator_counts,
            verbose=False,
        )
        lam_sys = float(stab_warm.M_sys_lambda_max)
        if lam_sys > 1e-14:
            alpha_cap = 0.95 * float(spectral_target) / lam_sys
            alpha_tso = np.minimum(alpha_tso, alpha_cap)
    except (np.linalg.LinAlgError, ValueError):
        pass  # leave alpha_tso untouched; PSO will handle infeasibility

    for gw in gw_tso_arrays:
        log_gw_tso.append(np.log10(np.maximum(gw, 1e-30)))

    log_gw_dso: List[NDArray[np.float64]] = []
    for d_idx, d in enumerate(dso_inputs):
        c_diag = _compute_curvature_diagonal(d.H, d.q_obj_diag)
        counts = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        gw = legacy_safety_factor_dso * alpha_dso[d_idx] * c_diag / 2.0
        gw = _apply_gw_floors(gw, counts, floors_dso, _DSO_COLUMN_ORDER)
        # Refine DSO alpha: start from local optimum 2/(lam_min+lam_max)
        # but cap by 0.95*spectral_target/lam_max so the warm start
        # satisfies the per-DSO hard stability bound  alpha*lam_max < target.
        # For ill-conditioned M_dso (lam_min << lam_max) the local optimum
        # approaches 2/lam_max which exceeds the cap; the cap is what
        # actually applies.
        _, _, alpha_local = _dso_cascade_decay(d.H, d.q_obj_diag, gw, n_inner)
        # Compute lam_max via the same M assembly used by _dso_actual_decay
        C_d = _compute_curvature_matrix(d.H, d.q_obj_diag)
        gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw, 1e-12))
        M_d = (gw_inv_sqrt[:, None] * C_d) * gw_inv_sqrt[None, :]
        _, active_d, _ = _effective_eigenspectrum(M_d)
        if len(active_d) > 0:
            lam_max_d = float(active_d[-1])
            alpha_cap_d = (0.95 * float(spectral_target) / lam_max_d
                           if lam_max_d > 1e-14 else float('inf'))
            alpha_dso[d_idx] = float(min(alpha_local, alpha_cap_d))
        else:
            alpha_dso[d_idx] = float(alpha_local)
        log_gw_dso.append(np.log10(np.maximum(gw, 1e-30)))

    parts: List[NDArray[np.float64]] = [alpha_tso, alpha_dso]
    parts.extend(log_gw_tso)
    parts.extend(log_gw_dso)
    x = np.concatenate(parts)
    return np.clip(x, lb, ub)


# ---------------------------------------------------------------------------
#  Fitness
# ---------------------------------------------------------------------------

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
    spectral_target: float = 1.8,
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

    1. Global TSO spectral bound: alpha_eff^TSO * lam_max(M_sys) < 2
       (necessary-and-sufficient for the closed-loop TSO iteration).
    2. Per-DSO local bound: alpha_d * lam_max(M_dso,d) < 2 for every DSO d.

    The conservative row-sum gamma is *not* in the fitness; it is reported
    in the metrics dict for diagnostics only.

    Returns ``(fitness, metrics_dict, stability_result_or_None)``.
    """
    # Default metrics dict layout -- every return path returns the same keys
    # so the convergence history has a uniform schema.
    metrics: Dict[str, float] = {
        'fitness':       1e6,
        'rho_max_tso':   1.0,
        'rho_max_dso':   1.0,
        'max_dso_decay': 1.0,
        'spectral':      float('nan'),
        'gamma':         float('nan'),
        'alpha_eff_tso': 0.0,
        'alpha_eff_dso': 0.0,
    }

    try:
        alpha_tso, gw_tso, alpha_dso, gw_dso = _decode(
            x, n_zones, n_dsos, tso_sizes, dso_sizes,
        )
        metrics['alpha_eff_tso'] = max(alpha_tso) if alpha_tso else 0.0
        metrics['alpha_eff_dso'] = max(alpha_dso) if alpha_dso else 0.0

        # TSO block: full multi-zone stability call (cheap, ~ms).
        # Returned ZoneStabilityResult.rho_i is computed at alpha_list[i].
        stab = analyse_multi_zone_stability(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            G_w_list=gw_tso,
            alpha_list=alpha_tso,
            zone_ids=zone_ids,
            actuator_counts=actuator_counts,
            verbose=False,
        )
        lam_sys = float(stab.M_sys_lambda_max)
        spectral = metrics['alpha_eff_tso'] * lam_sys
        gamma = float(stab.small_gain_gamma)  # diagnostic only
        metrics['spectral'] = spectral
        metrics['gamma'] = gamma

        # ── Hard constraint #1: TSO necessary-and-sufficient stability ──
        # spectral_target < 2 leaves headroom for the operating point
        # shifting between PSO and the post-PSO stability re-evaluation.
        if not np.isfinite(spectral) or spectral >= spectral_target:
            penalty = 1e6 + max(spectral, 0.0)
            metrics['fitness'] = penalty
            return penalty, metrics, stab

        # Per-zone TSO contraction rate at the chosen alpha
        rho_max_tso = max(
            (float(zr.rho_i) for zr in stab.zones),
            default=0.0,
        )
        metrics['rho_max_tso'] = rho_max_tso

        # ── DSO blocks: actual-alpha contraction + hard stability check ─
        rho_max_dso = 0.0
        max_dso_decay = 0.0
        for d_idx, d in enumerate(dso_inputs):
            rho_d, decay_d, lam_max_d = _dso_actual_decay(
                d.H, d.q_obj_diag, gw_dso[d_idx],
                float(alpha_dso[d_idx]), n_inner,
            )
            # Hard stability: alpha_d * lam_max < spectral_target
            if alpha_dso[d_idx] * lam_max_d >= spectral_target:
                penalty = 1e6 + float(alpha_dso[d_idx] * lam_max_d)
                metrics['fitness'] = penalty
                return penalty, metrics, stab
            if decay_d > max_dso_decay:
                max_dso_decay = decay_d
            if rho_d > rho_max_dso:
                rho_max_dso = rho_d
        metrics['rho_max_dso'] = rho_max_dso
        metrics['max_dso_decay'] = max_dso_decay

        fitness = float(max(rho_max_tso, max_dso_decay))
        if not np.isfinite(fitness):
            fitness = 1e6
        metrics['fitness'] = fitness
        return fitness, metrics, stab

    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        metrics['fitness'] = 1e6
        metrics['error'] = str(exc)  # type: ignore[assignment]
        return 1e6, metrics, None


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
    # Particles 2..N-1: warm start + Gaussian noise.  log-g_w stays in
    # log space (sigma = 0.3 dec); alpha entries are perturbed by ~10%
    # of the bound width to keep them inside.
    if swarm_size > 2:
        n_alpha = 0
        # We don't know the exact split here; treat the whole vector
        # uniformly with a single sigma in normalised units.  Since the
        # alpha entries occupy a much smaller numerical range than the
        # log-g_w entries, normalised noise on width gives a sensible
        # spread for both.
        for k in range(2, swarm_size):
            noise = rng.normal(scale=0.15, size=dim) * width
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
                f"rho_tso={m_best.get('rho_max_tso', float('nan')):.4f}  "
                f"rho_dso={m_best.get('rho_max_dso', float('nan')):.4f}  "
                f"spec={m_best.get('spectral', float('nan')):.4f}  "
                f"gamma={m_best.get('gamma', float('nan')):.4f}  "
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
    alpha_bounds: Tuple[float, float] = (1e-4, 1.0),
    swarm_size: int = 30,
    max_iterations: int = 100,
    w_inertia: Tuple[float, float] = (0.7, 0.4),
    c_cognitive: float = 1.5,
    c_social: float = 1.5,
    velocity_clamp_frac: float = 0.2,
    cascade_margin_target: float = 0.3,
    spectral_target: float = 1.8,
    tso_period_s: float = 180.0,
    dso_period_s: float = 60.0,
    legacy_safety_factor_tso: float = 4.0,
    legacy_safety_factor_dso: float = 2.0,
    alpha_init_tso: Optional[Dict[int, float]] = None,
    alpha_init_dso: float = 0.1,
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
    alpha_bounds :
        ``(alpha_min, alpha_max)`` for both TSO and DSO step sizes.
    swarm_size, max_iterations :
        PSO budget.  Default 30 × 100 ≈ 3000 fitness evals.
    w_inertia, c_cognitive, c_social, velocity_clamp_frac :
        PSO hyper-parameters.
    cascade_margin_target :
        Reporting threshold; not used as a hard constraint.
    spectral_target :
        Hard constraint headroom on the global TSO spectral bound.
        PSO penalises any particle whose ``alpha_eff * lam_max(M_sys)``
        exceeds this; the same cap is applied to the warm-start alpha.
        Default ``1.8`` (10% margin below the absolute stability bound
        of 2.0) -- matches the legacy ``tune_multi_zone`` default and
        leaves headroom for the operating point shifting between PSO
        and the post-PSO stability re-evaluation.
    tso_period_s, dso_period_s :
        Controller periods.  ``n_inner = T_TSO / T_DSO`` is the number of
        DSO iterations per TSO iteration; sets the cascade-decay exponent.
    legacy_safety_factor_tso, legacy_safety_factor_dso :
        Safety multipliers used by the Gershgorin warm start.  Match the
        defaults of ``tune_multi_zone`` and ``tune_dso``.
    alpha_init_tso, alpha_init_dso :
        Initial alphas seeded into the warm start (refined locally before
        encoding).
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

    lb, ub = _build_bounds(
        actuator_counts=actuator_counts,
        dso_inputs=dso_inputs,
        floors_tso=floors_tso,
        floors_dso=floors_dso,
        g_w_upper_factor=g_w_upper_factor,
        alpha_bounds=alpha_bounds,
    )
    assert lb.size == total_dim and ub.size == total_dim

    warm_start = _encode_warm_start(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        actuator_counts=actuator_counts,
        zone_ids=zone_ids,
        dso_inputs=dso_inputs,
        floors_tso=floors_tso,
        floors_dso=floors_dso,
        legacy_safety_factor_tso=legacy_safety_factor_tso,
        legacy_safety_factor_dso=legacy_safety_factor_dso,
        n_inner=n_inner,
        alpha_init_tso=alpha_init_tso,
        alpha_init_dso=alpha_init_dso,
        spectral_target=float(spectral_target),
        lb=lb,
        ub=ub,
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
        )

    warm_f, _, _ = fitness_fn(warm_start)
    if verbose:
        print(f"  [pso] warm-start fitness = {warm_f:.4f}")

    rng = np.random.default_rng(seed)
    g_best, g_best_f, iters_done, history = _pso_loop(
        fitness_fn=fitness_fn,
        lb=lb,
        ub=ub,
        warm_start=warm_start,
        swarm_size=int(swarm_size),
        max_iterations=int(max_iterations),
        w_inertia=w_inertia,
        c_cognitive=float(c_cognitive),
        c_social=float(c_social),
        velocity_clamp_frac=float(velocity_clamp_frac),
        rng=rng,
        verbose=verbose,
    )

    # Regression guard: never return worse than the warm start
    if warm_f < g_best_f:
        if verbose:
            print(
                f"  [pso] warning: PSO best ({g_best_f:.4f}) is worse than "
                f"warm start ({warm_f:.4f}); falling back to warm start"
            )
        g_best = warm_start
        g_best_f = warm_f

    # Decode the final solution
    alpha_tso, gw_tso, alpha_dso, gw_dso = _decode(
        g_best, n_zones, n_dsos, tso_sizes, dso_sizes,
    )

    # Final stability snapshot for diagnostics
    stab_final = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        alpha_list=alpha_tso,
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
            alpha=alpha_tso[idx],
            alpha_max_local=zr.alpha_max_local,
            alpha_max_coupled=zr.alpha_max_coupled,
            rho=zr.rho_i,
            kappa=zr.kappa_Mii,
            lambda_min=zr.lambda_min_Mii,
            lambda_max=zr.lambda_max_Mii,
            actuator_labels=_build_actuator_labels(actuator_counts[idx]),
        ))

    # DSO results: rebuild rho/cascade-margin from the final g_w
    dso_results: Dict[str, Tuple[NDArray[np.float64], float, float, float]] = {}
    feasibility_warnings: List[str] = []
    for d_idx, d in enumerate(dso_inputs):
        rho_d, decay_d, _ = _dso_cascade_decay(
            d.H, d.q_obj_diag, gw_dso[d_idx], n_inner,
        )
        cascade_margin = 1.0 - decay_d
        dso_results[d.dso_id] = (
            gw_dso[d_idx],
            float(alpha_dso[d_idx]),
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
    final_spectral = max(alpha_tso) * float(stab_final.M_sys_lambda_max)
    if final_spectral >= 2.0:
        feasibility_warnings.append(
            f"TSO spectral metric alpha_eff*lam_sys = {final_spectral:.4f} "
            f">= 2 (UNSTABLE)"
        )
    elif final_spectral >= float(spectral_target):
        feasibility_warnings.append(
            f"TSO spectral metric alpha_eff*lam_sys = {final_spectral:.4f} "
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
        warm_start_fitness=float(warm_f),
    )
