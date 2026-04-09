"""
Auto-Tuning for Multi-Zone OFO Controller Parameters
=====================================================

Computes per-actuator g_w weights that satisfy a stability condition for
the multi-zone (multi-TSO, multi-DSO) OFO controller hierarchy.

The step-size parameter alpha has been removed (absorbed into g_w).
The OFO update is now u^{k+1} = u^k + sigma^k with alpha=1.

Theory
------
The OFO iteration contracts iff all eigenvalues of the preconditioned
curvature matrix

    M = G_w^{-1/2} H^T Q_obj H G_w^{-1/2}

satisfy  0 < lambda_i(M) < 2.

Two stability bounds are available for the multi-zone case:

1. ROW-SUM (sufficient, conservative):
       gamma = max_i { rho_i + sum_{j!=i} sigma_ij } < 1
   where rho_i = max|1 - lambda_l(M_ii)| and
   sigma_ij = ||M_ij||_2.

2. SPECTRAL (necessary & sufficient, much tighter):
       lambda_max(M_sys) < 2
   where M_sys = [[M_TSO,ij]] is the full block matrix.

Functions
---------
compute_optimal_gw
    Phase 1 -- per-actuator Gershgorin preconditioning (fast, local).

tune_multi_zone
    Phase 2 -- g_w optimisation supporting either the spectral
    objective (default) or the row-sum objective.

tune_dso
    Per-actuator g_w tuning for DSO controllers with cascade margin
    enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    MultiZoneStabilityResult,
)


# =============================================================================
#  Result dataclasses
# =============================================================================

@dataclass
class ZoneTuningResult:
    """Tuning result for a single zone."""
    zone_id: int
    g_w: NDArray[np.float64]
    rho: float
    kappa: float
    lambda_min: float
    lambda_max: float
    actuator_labels: List[str]


@dataclass
class TuningResult:
    """Complete tuning result for the multi-zone system."""
    zones: List[ZoneTuningResult]
    small_gain_gamma: float
    converged: bool
    iterations: int
    stability_result: Optional[MultiZoneStabilityResult] = None
    objective: str = "spectral"
    spectral_metric: float = np.inf            # alpha_eff * lambda_max(M_sys)
    feasibility_warnings: List[str] = field(default_factory=list)

    def gw_vectors(self) -> List[NDArray[np.float64]]:
        """Return list of per-zone g_w vectors (same order as zones)."""
        return [z.g_w for z in self.zones]


# =============================================================================
#  Private helpers
# =============================================================================

def _compute_curvature_diagonal(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return diag(H^T Q_obj H) efficiently without forming the full matrix.

    Complexity: O(n_y * n_u) -- avoids the O(n_u^2 * n_y) full product.
    """
    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    HQ = q_sqrt[:, None] * H        # (n_y, n_u)
    return np.sum(HQ ** 2, axis=0)   # (n_u,)


def _compute_curvature_matrix(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return C = H^T Q_obj H (full matrix, needed for eigenvalue analysis)."""
    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    HQ = q_sqrt[:, None] * H        # (n_y, n_u)
    return HQ.T @ HQ                # (n_u, n_u)


#  Map actuator type name -> actuator_counts key.  Used by _apply_gw_floors
#  to resolve block sizes from the column-order list.
_TYPE_TO_COUNT_KEY = {
    'der':   'n_der',
    'pcc':   'n_pcc',
    'gen':   'n_gen',
    'oltc':  'n_oltc',
    'shunt': 'n_shunt',
}

# Default TSO column order (matches ZoneDefinition.gw_diagonal() in
# multi_tso_coordinator.py).
_TSO_COLUMN_ORDER = ('der', 'pcc', 'gen', 'oltc')

# Default DSO column order (matches DSOControllerConfig column ordering).
_DSO_COLUMN_ORDER = ('der', 'oltc', 'shunt')


def _apply_gw_floors(
    g_w: NDArray[np.float64],
    actuator_counts: Dict[str, int],
    floors: Dict[str, float],
    column_order: Tuple[str, ...] = _TSO_COLUMN_ORDER,
) -> NDArray[np.float64]:
    """Enforce per-actuator-type minimum g_w values.

    Parameters
    ----------
    g_w :
        Current per-actuator weight vector.  Modified in place on a copy.
    actuator_counts :
        Dict of actuator counts keyed by ``'n_der'``, ``'n_pcc'``,
        ``'n_gen'``, ``'n_oltc'``, ``'n_shunt'`` (missing keys default
        to 0).
    floors :
        Per-type minimum g_w values keyed by the type names ``'der'``,
        ``'pcc'``, ``'gen'``, ``'oltc'``, ``'shunt'``.  Missing keys are
        treated as 0 (no floor for that type).
    column_order :
        Tuple of actuator-type names describing how ``g_w`` is laid out.
        Defaults to the TSO layout ``('der', 'pcc', 'gen', 'oltc')``.
        Pass ``('der', 'oltc', 'shunt')`` for the DSO layout.

    Notes
    -----
    Each type block is clamped to ``g_w[block] >= floors[type]``; no type
    is hard-overridden (the V_gen block used to be a hard override, but
    is now a floor so the tuner can push it higher if the spectral
    analysis calls for it).
    """
    g_w = g_w.copy()

    off = 0
    for type_name in column_order:
        count_key = _TYPE_TO_COUNT_KEY.get(type_name)
        if count_key is None:
            continue
        n = int(actuator_counts.get(count_key, 0))
        if n <= 0:
            continue
        floor = float(floors.get(type_name, 0.0))
        if floor > 0.0:
            g_w[off:off + n] = np.maximum(g_w[off:off + n], floor)
        off += n

    return g_w


def _build_actuator_labels(actuator_counts: Dict[str, int]) -> List[str]:
    """Build human-readable actuator labels matching column ordering."""
    labels: List[str] = []
    for prefix, key in [('Q_DER', 'n_der'), ('Q_PCC', 'n_pcc'),
                        ('V_gen', 'n_gen'), ('OLTC', 'n_oltc')]:
        n = actuator_counts.get(key, 0)
        labels.extend(f'{prefix}_{k}' for k in range(n))
    return labels


def _effective_eigenspectrum(
    M: NDArray[np.float64],
    null_tol_factor: float = 1e-12,
    active_tol_factor: float = 0.01,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Compute eigenvalues of symmetric M, separating null/active modes.

    Returns (all_eigenvalues, active_eigenvalues, n_null).
    """
    eigs = np.linalg.eigvalsh(M)
    lam_max = float(np.maximum(eigs[-1], 0.0)) if len(eigs) > 0 else 0.0
    null_tol = null_tol_factor * max(lam_max, 1e-14)
    active_tol = active_tol_factor * max(lam_max, 1e-14)

    effective = eigs[eigs > null_tol]
    active = eigs[eigs > active_tol]
    n_null = len(eigs) - len(effective)

    return eigs, active, n_null


def _dominant_eigenpair(M: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
    """Return (lambda_max, unit eigenvector) of M (may be non-symmetric).

    For non-symmetric M the eigenvalues may be complex.  Returns the
    eigenvalue with the largest *real part* and the corresponding
    eigenvector (real part, unit-normalised).  Falls back to eigh for
    symmetric M (faster and numerically cleaner).
    """
    if M.size == 0:
        return 0.0, np.zeros(0)

    # Check symmetry — use fast symmetric solver when possible
    asym = float(np.linalg.norm(M - M.T, 'fro'))
    if asym < 1e-10 * max(float(np.linalg.norm(M, 'fro')), 1e-14):
        eigs, vecs = np.linalg.eigh(M)
        return float(eigs[-1]), np.asarray(vecs[:, -1], dtype=np.float64)

    # General (non-symmetric) eigensolver
    eigs, vecs = np.linalg.eig(M)
    # Pick eigenvalue with largest real part
    idx = int(np.argmax(eigs.real))
    lam = float(eigs[idx].real)
    v = np.asarray(vecs[:, idx].real, dtype=np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-14:
        v /= v_norm
    return lam, v


def _zone_index_ranges(n_per_zone: List[int]) -> List[Tuple[int, int]]:
    """Return [(start, stop)] half-open ranges for each zone in M_sys."""
    ranges: List[Tuple[int, int]] = []
    off = 0
    for n in n_per_zone:
        ranges.append((off, off + n))
        off += n
    return ranges


# =============================================================================
#  Phase 1: compute_optimal_gw  (Gershgorin preconditioning)
# =============================================================================

def compute_optimal_gw(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    *,
    safety_factor: float = 2.0,
    min_gw_der: float = 0.01,
    min_gw_pcc: float = 0.1,
    min_gw_gen: float = 1e4,
    min_gw_oltc: float = 40.0,
) -> List[NDArray[np.float64]]:
    """Compute per-actuator g_w vectors from local curvature (Phase 1).

    For each zone i and each actuator k:

        g_w[k] = safety_factor * C_ii[k,k] / 2

    where C_ii = H_ii^T Q_obj,i H_ii is the local curvature.

    This satisfies the per-actuator Gershgorin necessary condition
    g_w[k] > C_ii[k,k] / 2 with the specified safety factor.

    Parameters
    ----------
    H_blocks :
        Dict mapping (zone_i, zone_i) to diagonal sensitivity blocks H_ii.
    Q_obj_list :
        Per-zone Q_obj diagonal vectors (one per zone, ordered by zone_id).
    actuator_counts :
        Per-zone dicts with keys 'n_der', 'n_pcc', 'n_gen', 'n_oltc'.
    safety_factor :
        Multiplier on the theoretical Gershgorin bound. Default 2.0.
    min_gw_der, min_gw_pcc, min_gw_gen, min_gw_oltc :
        Per-actuator-type minimum g_w floors.

    Returns
    -------
    List of per-zone g_w vectors (numpy arrays), ordered by zone_id.
    """
    # Extract sorted zone ids from diagonal blocks
    zone_ids = sorted({i for (i, j) in H_blocks if i == j})

    floors = {
        'der':  min_gw_der,
        'pcc':  min_gw_pcc,
        'gen':  min_gw_gen,
        'oltc': min_gw_oltc,
    }

    result: List[NDArray[np.float64]] = []
    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        q_obj = Q_obj_list[idx]
        counts = actuator_counts[idx]

        # Per-actuator curvature diagonal
        c_diag = _compute_curvature_diagonal(H_ii, q_obj)

        # Gershgorin-based g_w (alpha=1, absorbed into g_w)
        g_w = safety_factor * c_diag / 2.0

        # Apply per-type floors
        g_w = _apply_gw_floors(g_w, counts, floors, _TSO_COLUMN_ORDER)

        result.append(g_w)

    return result


# =============================================================================
#  Infeasibility detection (#5)
# =============================================================================

def _row_sum_floor_estimate(
    stab: MultiZoneStabilityResult,
) -> Tuple[float, List[str]]:
    """Estimate the minimum achievable small-gain gamma for the *current*
    g_w preconditioning.

    For each zone i, the row sum  f_i(alpha) = rho_i(alpha) + alpha * C_i  is
    convex in alpha (rho is piecewise linear, the coupling term is linear).
    Its unconstrained minimum is:

        f_i_min = (lambda_max - lambda_min + 2 * coupling_sum)
                  / (lambda_max + lambda_min)        if C_i < lambda_min
        f_i_min = 1                                   otherwise

    where C_i = sum_{j != i} ||M_ij||_2.  The system floor is the worst row.

    The "C_i < lambda_min" branch is the regime where coupling stays below
    self-curvature; in the other branch the row sum cannot be pushed below 1
    by alpha alone, so the small-gain condition is structurally infeasible
    without re-preconditioning g_w (or accepting the system is rate-limited).

    Returns (gamma_floor, warnings).
    """
    warnings: List[str] = []
    floors: List[float] = []
    for zr in stab.zones:
        C_i = zr.coupling_sum
        l_min = max(zr.lambda_min_Mii, 0.0)
        l_max = max(zr.lambda_max_Mii, l_min)
        if l_max <= 1e-14:
            floors.append(0.0)
            continue
        if C_i >= l_min:
            floors.append(1.0)
            warnings.append(
                f"  Zone {zr.zone_id}: coupling_sum = {C_i:.4g} "
                f">= lambda_min(M_ii) = {l_min:.4g}.  "
                f"Row sum cannot be < 1 with current g_w preconditioning."
            )
        else:
            f_min = (l_max - l_min + 2.0 * C_i) / (l_max + l_min)
            floors.append(f_min)
    gamma_floor = max(floors) if floors else 0.0
    return gamma_floor, warnings


def _check_feasibility(
    stab: MultiZoneStabilityResult,
    objective: str,
    gamma_target: float,
    spectral_target: float,
    verbose: bool,
) -> List[str]:
    """Print early warnings about target feasibility.  Returns the warning list."""
    notes: List[str] = []

    # Spectral side -- always reported (it is the tight bound)
    lam_sys = stab.M_sys_lambda_max
    if lam_sys > 1e-14:
        alpha_eff_max = 2.0 / lam_sys
        notes.append(
            f"  spectral: lambda_max(M_sys) = {lam_sys:.4g}, "
            f"alpha_eff <= {alpha_eff_max:.4g} for stability."
        )
    else:
        notes.append("  spectral: lambda_max(M_sys) ~ 0; system is degenerate.")

    # Row-sum side -- check if gamma < gamma_target is reachable at all
    gamma_floor, row_notes = _row_sum_floor_estimate(stab)
    notes.append(
        f"  row-sum: gamma_floor (current g_w) ~ {gamma_floor:.4g}.  "
        f"target = {gamma_target:.4g}."
    )
    notes.extend(row_notes)

    if gamma_floor >= gamma_target:
        if objective == "row_sum":
            notes.append(
                "  WARNING: row-sum target may be infeasible with the current "
                "zone partition / preconditioning.  Consider:"
            )
            notes.append("    (a) switching to objective='spectral' (tighter bound),")
            notes.append("    (b) loosening gamma_target,")
            notes.append("    (c) re-partitioning zones to reduce cross-coupling,")
            notes.append("    (d) adding controllable actuators in the bottleneck zones.")
        else:
            notes.append(
                "  NOTE: row-sum gamma is unreachable below target with this "
                "preconditioning, but objective='spectral' may still succeed."
            )

    if verbose:
        for line in notes:
            print(line)
    return notes


# =============================================================================
#  Phase 2: tune_multi_zone  (joint alpha + g_w optimisation)
# =============================================================================

def tune_multi_zone(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    *,
    zone_ids: Optional[List[int]] = None,
    safety_factor: float = 4.0,
    min_gw_der: float = 1e-3,
    min_gw_pcc: float = 0.1,
    min_gw_gen: float = 1e5,
    min_gw_oltc: float = 40.0,
    objective: Literal["spectral", "row_sum"] = "spectral",
    spectral_target: float = 1.8,
    gamma_target: float = 0.9,
    max_iterations: int = 30,
    verbose: bool = True,
) -> TuningResult:
    """Tune g_w for a multi-zone TSO-DSO OFO system (alpha=1, absorbed).

    Two objectives are supported:

    * ``objective='spectral'`` (default, recommended):
        Target  lambda_max(M_sys) < spectral_target  (default 1.8,
        i.e. 10% margin below the stability bound 2.0).
        g_w boost direction is read off the dominant eigenvector of M_sys
        so that exactly the actuators participating in the worst global
        mode are preconditioned.

    * ``objective='row_sum'``:
        Target  gamma = max_i {rho_i + sum sigma_ij} < gamma_target.
        For the bottleneck row i, g_w is boosted on the *column zone* j
        contributing the most to sigma_ij.

    Parameters
    ----------
    H_blocks, Q_obj_list, actuator_counts, zone_ids :
        Same as analyse_multi_zone_stability.
    safety_factor :
        Phase-1 Gershgorin safety multiplier.
    min_gw_der, min_gw_pcc, min_gw_gen, min_gw_oltc :
        Per-actuator-type minimum g_w floors.
    objective :
        'spectral' (default) or 'row_sum'.
    spectral_target :
        Convergence target for lambda_max(M_sys).  Default 1.8.
    gamma_target :
        Convergence target for the row-sum metric (only used when
        objective='row_sum').
    max_iterations :
        Cap on the inner refinement loop.
    verbose :
        If True, print progress and final report.

    Returns
    -------
    TuningResult with per-zone g_w vectors, stability metrics, and
    feasibility warnings.
    """
    # ── Resolve zone ids ─────────────────────────────────────────────────────
    if zone_ids is None:
        zone_ids = sorted({i for (i, j) in H_blocks if i == j})
    n_zones = len(zone_ids)

    if objective not in ("spectral", "row_sum"):
        raise ValueError(f"Unknown objective {objective!r}; "
                         "expected 'spectral' or 'row_sum'.")

    # ── Build TSO per-type floor dict once; reused at every g_w update ─────
    floors = {
        'der':  min_gw_der,
        'pcc':  min_gw_pcc,
        'gen':  min_gw_gen,
        'oltc': min_gw_oltc,
    }

    # ── Phase 1: Gershgorin initialisation (alpha=1, absorbed into g_w) ─────
    C_matrices: List[NDArray[np.float64]] = []
    gw_list: List[NDArray[np.float64]] = []

    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        q_obj = Q_obj_list[idx]
        counts = actuator_counts[idx]

        C_ii = _compute_curvature_matrix(H_ii, q_obj)
        C_matrices.append(C_ii)

        c_diag = np.diag(C_ii)
        g_w = safety_factor * c_diag / 2.0
        g_w = _apply_gw_floors(g_w, counts, floors, _TSO_COLUMN_ORDER)
        gw_list.append(g_w)

    # ── Early infeasibility check (#5) ───────────────────────────────────────
    stab0 = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_list,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=False,
    )
    if verbose:
        print(f"  [tune] objective = '{objective}', "
              f"spectral_target = {spectral_target:.3g}, "
              f"gamma_target = {gamma_target:.3g}")
        print("  [tune] Initial preconditioning bounds:")
    feasibility_notes = _check_feasibility(
        stab0, objective, gamma_target, spectral_target, verbose,
    )

    # ── Phase 2: refinement loop ─────────────────────────────────────────────
    n_per_zone = [len(gw) for gw in gw_list]
    zone_ranges = _zone_index_ranges(n_per_zone)

    stab: Optional[MultiZoneStabilityResult] = stab0
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        stab = analyse_multi_zone_stability(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            G_w_list=gw_list,
            zone_ids=zone_ids,
            actuator_counts=actuator_counts,
            verbose=False,
        )
        gamma = stab.small_gain_gamma
        lam_sys = stab.M_sys_lambda_max
        spectral_metric = lam_sys  # alpha=1

        if verbose:
            print(f"  [tune] iter {iteration:2d}: "
                  f"gamma = {gamma:.4f}  "
                  f"lam_sys = {spectral_metric:.4f}")

        # Convergence check (#1)
        if objective == "spectral":
            if spectral_metric < spectral_target:
                converged = True
                break
        else:  # row_sum
            if gamma < gamma_target:
                converged = True
                break

        # ── Identify which g_w to boost (#2 / spectral mode) ─────────────
        if objective == "spectral":
            # Use the dominant eigenvector of M_sys: boost the actuators with
            # the largest squared participation in the worst global mode.
            _, v_sys = _dominant_eigenpair(stab.M_sys)
            v_sq = v_sys * v_sys

            # Determine which zone owns the most mass (used as a target for
            # in-zone scaling).  We boost g_w of THAT zone in the directions
            # where v_sq is large.
            zone_mass = np.array([
                float(np.sum(v_sq[lo:hi])) for (lo, hi) in zone_ranges
            ])
            target_idx = int(np.argmax(zone_mass))
            lo, hi = zone_ranges[target_idx]
            v_zone = v_sq[lo:hi]
            v_zone_max = float(np.max(v_zone)) if v_zone.size else 0.0

            if v_zone_max > 1e-14:
                # Scale in [1, 2.5]: actuators participating most get the
                # biggest boost.  We grow g_w_target so the rest of the
                # global eigenvector mass shifts to other modes.
                scale = 1.0 + 1.5 * (v_zone / v_zone_max)
                gw_list[target_idx] = _apply_gw_floors(
                    gw_list[target_idx] * scale,
                    actuator_counts[target_idx],
                    floors, _TSO_COLUMN_ORDER,
                )

            # Even global boost on the dominant zone is the primary lever,
            # but we also nudge zones with non-trivial mass to keep g_w
            # ratios from drifting (avoids the sqrt(g_w_i/g_w_j) trap).
            for k in range(n_zones):
                if k == target_idx:
                    continue
                if zone_mass[k] < 0.05 * zone_mass[target_idx]:
                    continue
                lo_k, hi_k = zone_ranges[k]
                v_k = v_sq[lo_k:hi_k]
                v_k_max = float(np.max(v_k)) if v_k.size else 0.0
                if v_k_max > 1e-14:
                    scale_k = 1.0 + 0.5 * (v_k / v_k_max)
                    gw_list[k] = _apply_gw_floors(
                        gw_list[k] * scale_k,
                        actuator_counts[k],
                        floors, _TSO_COLUMN_ORDER,
                    )

        else:  # row_sum  -- boost the COLUMN zone of the worst sigma_ij (#2)
            row_sums = [zr.lyapunov_row_sum for zr in stab.zones]
            bottleneck_idx = int(np.argmax(row_sums))
            bn = stab.zones[bottleneck_idx]
            i_id = zone_ids[bottleneck_idx]

            # Pick the j != i with the largest sigma_ij contribution.
            best_j_idx: Optional[int] = None
            best_sigma = -1.0
            for j_idx, zj_id in enumerate(zone_ids):
                if j_idx == bottleneck_idx:
                    continue
                sig = bn.sigma_ij.get(zj_id, 0.0)
                if sig > best_sigma:
                    best_sigma = sig
                    best_j_idx = j_idx

            if best_j_idx is not None and best_sigma > 0.0:
                # Boost g_w of zone j along the columns of M_ij that have
                # the largest column norms (those carry most of ||M_ij||_2).
                # We approximate via the column-wise squared norm of the
                # M_ij block, which is what the spectral norm acts on.
                # M_ij is not stored on the result, so reconstruct quickly:
                gw_i = gw_list[bottleneck_idx]
                gw_j = gw_list[best_j_idx]
                C_ij = _build_cross_curvature(
                    H_blocks, Q_obj_list, zone_ids,
                    bottleneck_idx, best_j_idx,
                )
                if C_ij is not None:
                    gw_i_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_i, 1e-12))
                    gw_j_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_j, 1e-12))
                    M_ij = (gw_i_inv_sqrt[:, None] * C_ij) * gw_j_inv_sqrt[None, :]
                    col_sq = np.sum(M_ij ** 2, axis=0)
                    col_max = float(np.max(col_sq)) if col_sq.size else 0.0
                    if col_max > 1e-14:
                        scale = 1.0 + 1.5 * (col_sq / col_max)
                        gw_list[best_j_idx] = _apply_gw_floors(
                            gw_list[best_j_idx] * scale,
                            actuator_counts[best_j_idx],
                            floors, _TSO_COLUMN_ORDER,
                        )

            # Also flatten the bottleneck's own M_ii kappa (helps rho_i).
            C_ii = C_matrices[bottleneck_idx]
            g_w_i = gw_list[bottleneck_idx]
            gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w_i, 1e-12))
            M_ii_loc = (gw_inv_sqrt[:, None] * C_ii) * gw_inv_sqrt[None, :]
            M_diag = np.diag(M_ii_loc)
            m_max = float(np.max(M_diag)) if M_diag.size else 0.0
            if m_max > 1e-14:
                scale_ii = 1.0 + 0.5 * (M_diag / m_max)
                gw_list[bottleneck_idx] = _apply_gw_floors(
                    g_w_i * scale_ii,
                    actuator_counts[bottleneck_idx],
                    floors, _TSO_COLUMN_ORDER,
                )

        # (alpha refresh removed: alpha is absorbed into g_w)

    # ── Fallback: if not converged, log a warning (no alpha to reduce) ────
    if not converged and verbose:
        print(f"  [tune] not converged after {max_iterations} iterations.  "
              f"Increase g_w or adjust floors.")

    # ── Final validation ─────────────────────────────────────────────────────
    stab_final = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_list,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=verbose,
    )
    final_spectral = stab_final.M_sys_lambda_max  # alpha=1

    if objective == "spectral":
        converged_final = final_spectral < spectral_target
    else:
        converged_final = stab_final.small_gain_gamma < gamma_target

    # ── Package results ──────────────────────────────────────────────────────
    zone_results: List[ZoneTuningResult] = []
    for idx, zr in enumerate(stab_final.zones):
        zone_results.append(ZoneTuningResult(
            zone_id=zone_ids[idx],
            g_w=gw_list[idx],
            rho=zr.rho_i,
            kappa=zr.kappa_Mii,
            lambda_min=zr.lambda_min_Mii,
            lambda_max=zr.lambda_max_Mii,
            actuator_labels=_build_actuator_labels(actuator_counts[idx]),
        ))

    return TuningResult(
        zones=zone_results,
        small_gain_gamma=stab_final.small_gain_gamma,
        converged=converged_final,
        iterations=iteration,
        stability_result=stab_final,
        objective=objective,
        spectral_metric=final_spectral,
        feasibility_warnings=feasibility_notes,
    )


def _build_cross_curvature(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    zone_ids: List[int],
    i_idx: int,
    j_idx: int,
) -> Optional[NDArray[np.float64]]:
    """Compute C_ij = H_ii^T Q_obj,i H_ij if both blocks are present."""
    i = zone_ids[i_idx]
    j = zone_ids[j_idx]
    H_ii = H_blocks.get((i, i))
    H_ij = H_blocks.get((i, j))
    if H_ii is None or H_ij is None:
        return None
    q_obj_i = Q_obj_list[i_idx]
    q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))
    QH_ii = q_sqrt_i[:, None] * H_ii
    QH_ij = q_sqrt_i[:, None] * H_ij
    return QH_ii.T @ QH_ij


# =============================================================================
#  DSO cascade tuning
# =============================================================================

def tune_dso(
    H_dso: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    n_der: int,
    n_oltc: int,
    n_shunt: int = 0,
    *,
    safety_factor: float = 2.0,
    min_gw_der: float = 0.01,
    min_gw_oltc: float = 40.0,
    min_gw_shunt: float = 40.0,
    tso_period_s: float = 180.0,
    dso_period_s: float = 60.0,
    cascade_margin_target: float = 0.3,
    max_refinement_iterations: int = 0,
    max_growth_per_call: float = 10.0,
) -> Tuple[NDArray[np.float64], float, float]:
    """Tune per-actuator g_w for a DSO controller with cascade margin.

    Ensures the DSO converges within the TSO period by checking:

        cascade_margin = 1 - rho_D^(T_T / T_D) > cascade_margin_target

    where rho_D is the spectral contraction rate of the DSO iteration.

    Two phases:

    1. **Phase 1 (always run):** per-actuator Gershgorin preconditioning
       ``g_w = safety_factor * diag(C_dso) / 2``, then per-type
       floors via ``_apply_gw_floors``.  This produces sensible per-MVA
       (DER) and per-tap (OLTC) regularisation values that match the
       physical scale of the controlled outputs.

    2. **Phase 2 (optional, off by default):** non-uniform iterative
       refinement that boosts ``g_w`` for high-curvature actuators to
       flatten the spectrum of ``M_dso`` and lower its condition number.
       For poorly-conditioned DSO networks (kappa >> 1) this loop tends
       to inflate ``g_w`` by orders of magnitude without actually
       improving ``rho_D`` -- on the IEEE-39 cascade it ran 20 iterations
       and produced ``g_w ~ 1e7``, well outside any usable range.  It is
       therefore disabled by default.  Set ``max_refinement_iterations``
       > 0 to enable it for well-conditioned topologies; the per-call
       growth is hard-capped by ``max_growth_per_call`` to prevent
       runaways.

    Parameters
    ----------
    H_dso :
        DSO sensitivity matrix (n_y_dso, n_u_dso).
    q_obj_diag :
        Per-output objective weight vector.
    n_der, n_oltc, n_shunt :
        Actuator counts (column order: ``[DER | OLTC | shunt]``).
    safety_factor :
        Phase-1 Gershgorin safety multiplier.
    min_gw_der, min_gw_oltc, min_gw_shunt :
        Per-actuator-type minimum g_w floors for the DSO column layout.
    tso_period_s, dso_period_s :
        Controller periods.  Their ratio determines how many DSO
        iterations must converge within one TSO period.
    cascade_margin_target :
        Target for ``1 - rho_D^(T_T / T_D)``.  Default 0.3.
    max_refinement_iterations :
        Cap on the Phase-2 refinement loop.  Default 0 -- only Phase 1
        runs and tune_dso returns immediately after the floors are
        applied.  Increase to enable refinement for well-conditioned
        DSOs.
    max_growth_per_call :
        Hard cap on the element-wise growth factor of g_w from Phase 1
        to the final return value.  Default 10x.  Only meaningful when
        ``max_refinement_iterations > 0``.

    Returns
    -------
    (g_w, rho_D, cascade_margin) : tuple
        The tuned per-actuator weights, the DSO spectral contraction
        rate at optimal alpha, and the achieved cascade margin.
    """
    n_inner = max(int(tso_period_s / dso_period_s), 1)

    C_dso = _compute_curvature_matrix(H_dso, q_obj_diag)
    c_diag = np.diag(C_dso)

    counts = {'n_der': n_der, 'n_oltc': n_oltc, 'n_shunt': n_shunt}
    floors = {
        'der':   min_gw_der,
        'oltc':  min_gw_oltc,
        'shunt': min_gw_shunt,
    }

    # ── Phase 1: Gershgorin per-actuator preconditioning + per-type floors
    g_w = safety_factor * c_diag / 2.0
    g_w = _apply_gw_floors(g_w, counts, floors, _DSO_COLUMN_ORDER)
    g_w_phase1 = g_w.copy()  # remembered for the per-call growth cap

    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M_dso = (gw_inv_sqrt[:, None] * C_dso) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M_dso)

    if len(active) == 0:
        return g_w, 0.0, 1.0

    l_min, l_max = float(active[0]), float(active[-1])
    alpha_opt = 2.0 / (l_min + l_max) if len(active) >= 2 else 1.0 / l_max
    rho_D = (l_max - l_min) / (l_max + l_min) if len(active) >= 2 else 0.0
    cascade_margin = 1.0 - rho_D ** n_inner

    # Phase-1 already meets the target, OR Phase-2 refinement is disabled.
    if (cascade_margin >= cascade_margin_target
            or max_refinement_iterations <= 0):
        return g_w, rho_D, cascade_margin

    # ── Phase 2 (optional): non-uniform iterative refinement ─────────────
    g_w_cap = g_w_phase1 * float(max_growth_per_call)
    for attempt in range(int(max_refinement_iterations)):
        M_diag = np.diag(M_dso)
        if np.max(M_diag) < 1e-14:
            break

        scale = 1.0 + 1.0 * (M_diag / np.max(M_diag))
        g_w_new = g_w * scale

        # Hard upper cap relative to the Phase-1 baseline so the loop
        # cannot run away on poorly-conditioned DSO networks.
        g_w_new = np.minimum(g_w_new, g_w_cap)
        g_w = _apply_gw_floors(g_w_new, counts, floors, _DSO_COLUMN_ORDER)

        gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
        M_dso = (gw_inv_sqrt[:, None] * C_dso) * gw_inv_sqrt[None, :]
        _, active, _ = _effective_eigenspectrum(M_dso)

        if len(active) < 2:
            rho_D = 0.0
            cascade_margin = 1.0
            break

        l_min, l_max = float(active[0]), float(active[-1])
        rho_D = (l_max - l_min) / (l_max + l_min)
        cascade_margin = 1.0 - rho_D ** n_inner

        if cascade_margin >= cascade_margin_target:
            break

    return g_w, rho_D, cascade_margin
