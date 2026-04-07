"""
Auto-Tuning for Multi-Zone OFO Controller Parameters
=====================================================

Computes per-actuator g_w weights and step sizes alpha that satisfy the
contraction condition for the multi-zone (multi-TSO, multi-DSO) OFO
controller hierarchy.

Theory
------
The OFO iteration u^{k+1} = u^k + alpha * sigma^k contracts iff all
eigenvalues of the preconditioned curvature matrix

    M = G_w^{-1/2} H^T Q_obj H G_w^{-1/2}

satisfy  0 < alpha * lambda_i(M) < 2.

For a multi-zone system with N TSO zones, the small-gain condition is:

    gamma = max_i { rho_i + sum_{j!=i} sigma_ij } < 1

where rho_i = max|1 - alpha_i * lambda_l(M_ii)| is the per-zone
contraction rate and sigma_ij = alpha_i * ||M_ij||_2 the cross-coupling
gain.

Functions
---------
compute_optimal_gw
    Phase 1 -- per-actuator Gershgorin preconditioning (fast, local).
    Matches the existing call site in run_M_TSO_M_DSO.py.

tune_multi_zone
    Phase 2 -- joint alpha + g_w optimisation using the multi-zone
    small-gain condition.

tune_dso
    Per-actuator g_w tuning for DSO controllers with cascade margin
    enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from core.cascade_config import CascadeConfig
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
    alpha: float
    alpha_max_local: float
    alpha_max_coupled: float
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

    def gw_vectors(self) -> List[NDArray[np.float64]]:
        """Return list of per-zone g_w vectors (same order as zones)."""
        return [z.g_w for z in self.zones]

    def alpha_dict(self) -> Dict[int, float]:
        """Return {zone_id: alpha} mapping."""
        return {z.zone_id: z.alpha for z in self.zones}


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


def _apply_gw_floors(
    g_w: NDArray[np.float64],
    actuator_counts: Dict[str, int],
    min_gw: float,
    min_gw_discrete: float,
    gen_gw_override: float,
) -> NDArray[np.float64]:
    """Enforce per-type minimum g_w values.

    Column order: [Q_DER | Q_PCC | V_gen | OLTC] matching
    ZoneDefinition.gw_diagonal() in multi_tso_coordinator.py.

    Generator weights are hard-overridden (not a floor) since their
    electromechanical dynamics are not captured by the steady-state H.
    """
    g_w = g_w.copy()
    n_der  = actuator_counts.get('n_der', 0)
    n_pcc  = actuator_counts.get('n_pcc', 0)
    n_gen  = actuator_counts.get('n_gen', 0)
    n_oltc = actuator_counts.get('n_oltc', 0)

    off = 0
    # DER block -- continuous, apply min_gw
    g_w[off:off + n_der] = np.maximum(g_w[off:off + n_der], min_gw)
    off += n_der

    # PCC block -- continuous, apply min_gw
    g_w[off:off + n_pcc] = np.maximum(g_w[off:off + n_pcc], min_gw)
    off += n_pcc

    # Generator block -- hard override
    g_w[off:off + n_gen] = gen_gw_override
    off += n_gen

    # OLTC block -- discrete, apply min_gw_discrete
    g_w[off:off + n_oltc] = np.maximum(g_w[off:off + n_oltc], min_gw_discrete)
    off += n_oltc

    return g_w


def _build_actuator_labels(actuator_counts: Dict[str, int]) -> List[str]:
    """Build human-readable actuator labels matching column ordering."""
    labels: List[str] = []
    for prefix, key in [('Q_DER', 'n_der'), ('Q_PCC', 'n_pcc'),
                        ('V_gen', 'n_gen'), ('OLTC', 'n_oltc')]:
        n = actuator_counts.get(key, 0)
        labels.extend(f'{prefix}_{k}' for k in range(n))
    return labels


def _optimal_alpha_for_zone(
    C_ii: NDArray[np.float64],
    g_w: NDArray[np.float64],
    alpha_fallback: float,
) -> float:
    """Compute optimal alpha for a zone given C_ii and g_w.

    Returns alpha* = 2 / (lambda_min + lambda_max) of the preconditioned
    curvature M_ii, which minimises the contraction rate rho_i.
    """
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M_ii = (gw_inv_sqrt[:, None] * C_ii) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M_ii)
    if len(active) >= 2:
        l_min, l_max = float(active[0]), float(active[-1])
        return 2.0 / (l_min + l_max)
    elif len(active) == 1:
        return 1.0 / float(active[0])
    return alpha_fallback


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


# =============================================================================
#  Phase 1: compute_optimal_gw  (Gershgorin preconditioning)
# =============================================================================

def compute_optimal_gw(
    config: CascadeConfig,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    *,
    safety_factor: float = 2.0,
    min_gw: float = 0.01,
    min_gw_discrete: float = 40.0,
) -> List[NDArray[np.float64]]:
    """Compute per-actuator g_w vectors from local curvature (Phase 1).

    For each zone i and each actuator k:

        g_w[k] = safety_factor * alpha * C_ii[k,k] / 2

    where C_ii = H_ii^T Q_obj,i H_ii is the local curvature.

    This satisfies the per-actuator Gershgorin necessary condition
    g_w[k] > alpha * C_ii[k,k] / 2 with the specified safety factor.

    Parameters
    ----------
    config :
        CascadeConfig providing alpha and gw_tso_v_gen.
    H_blocks :
        Dict mapping (zone_i, zone_i) to diagonal sensitivity blocks H_ii.
    Q_obj_list :
        Per-zone Q_obj diagonal vectors (one per zone, ordered by zone_id).
    actuator_counts :
        Per-zone dicts with keys 'n_der', 'n_pcc', 'n_gen', 'n_oltc'.
    safety_factor :
        Multiplier on the theoretical Gershgorin bound. Default 2.0.
    min_gw :
        Floor for continuous actuators (DER, PCC).
    min_gw_discrete :
        Floor for discrete actuators (OLTC).

    Returns
    -------
    List of per-zone g_w vectors (numpy arrays), ordered by zone_id.
    """
    # Extract sorted zone ids from diagonal blocks
    zone_ids = sorted({i for (i, j) in H_blocks if i == j})

    result: List[NDArray[np.float64]] = []
    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        q_obj = Q_obj_list[idx]
        counts = actuator_counts[idx]

        # Per-actuator curvature diagonal
        c_diag = _compute_curvature_diagonal(H_ii, q_obj)

        # Gershgorin-based g_w
        g_w = safety_factor * config.alpha * c_diag / 2.0

        # Apply type-specific floors and generator override
        g_w = _apply_gw_floors(
            g_w, counts, min_gw, min_gw_discrete,
            gen_gw_override=config.gw_tso_v_gen,
        )

        result.append(g_w)

    return result


# =============================================================================
#  Phase 2: tune_multi_zone  (joint alpha + g_w optimisation)
# =============================================================================

def tune_multi_zone(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    alpha_init: Optional[Dict[int, float]] = None,
    *,
    zone_ids: Optional[List[int]] = None,
    safety_factor: float = 2.0,
    min_gw: float = 0.01,
    min_gw_discrete: float = 40.0,
    gen_gw: float = 1e7,
    gamma_target: float = 0.8,
    max_iterations: int = 20,
    verbose: bool = True,
) -> TuningResult:
    """Jointly tune g_w and alpha for a multi-zone TSO-DSO OFO system.

    Iteratively adjusts per-actuator g_w and per-zone alpha to satisfy
    the multi-zone small-gain condition gamma < gamma_target, where

        gamma = max_i { rho_i + sum_{j!=i} sigma_ij }

    Algorithm:
        1. Initialise g_w via Gershgorin preconditioning (Phase 1).
        2. Compute per-zone optimal alpha from M_ii eigenspectrum.
        3. Check small-gain via analyse_multi_zone_stability().
        4. If gamma >= target: scale g_w of bottleneck zone, recompute.
        5. Repeat until convergence or max_iterations.

    Parameters
    ----------
    H_blocks :
        Sensitivity blocks (i,j) -> H_ij.  Must include all diagonal
        blocks; off-diagonal blocks are optional (zero if missing).
    Q_obj_list :
        Per-zone Q_obj diagonal vectors.
    actuator_counts :
        Per-zone dicts with keys 'n_der', 'n_pcc', 'n_gen', 'n_oltc'.
    alpha_init :
        Initial step sizes as {zone_id: alpha}.  If None, all zones
        start at alpha = 1.0.
    zone_ids :
        Zone IDs in the same order as Q_obj_list.  Defaults to sorted
        diagonal block keys.
    safety_factor :
        Multiplier on the Gershgorin g_w bound.
    min_gw, min_gw_discrete :
        Floors for continuous / discrete actuators.
    gen_gw :
        Hard override for generator g_w (not auto-tuned).
    gamma_target :
        Target for the small-gain contraction rate.  Values < 1.0
        provide robustness margin; 0.8 is recommended.
    max_iterations :
        Maximum tuning iterations.
    verbose :
        If True, print progress and final stability report.

    Returns
    -------
    TuningResult with per-zone g_w vectors, alpha values, and stability
    assessment.
    """
    # ── Resolve zone ids ─────────────────────────────────────────────────────
    if zone_ids is None:
        zone_ids = sorted({i for (i, j) in H_blocks if i == j})
    n_zones = len(zone_ids)

    # ── Phase 1: Gershgorin initialisation ───────────────────────────────────
    alpha_list = [
        (alpha_init or {}).get(z, 1.0) for z in zone_ids
    ]

    # Compute full curvature matrices (needed for eigenvalue analysis)
    C_matrices: List[NDArray[np.float64]] = []
    gw_list: List[NDArray[np.float64]] = []

    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        q_obj = Q_obj_list[idx]
        counts = actuator_counts[idx]

        C_ii = _compute_curvature_matrix(H_ii, q_obj)
        C_matrices.append(C_ii)

        # Initial g_w from curvature diagonal
        c_diag = np.diag(C_ii)
        g_w = safety_factor * alpha_list[idx] * c_diag / 2.0
        g_w = _apply_gw_floors(g_w, counts, min_gw, min_gw_discrete, gen_gw)
        gw_list.append(g_w)

    # ── Phase 1b: Compute optimal alpha from initial g_w ─────────────────────
    for idx in range(n_zones):
        alpha_list[idx] = _optimal_alpha_for_zone(
            C_matrices[idx], gw_list[idx], alpha_list[idx],
        )

    # ── Phase 2: Iterative coupling-aware refinement ─────────────────────────
    stab: Optional[MultiZoneStabilityResult] = None
    gamma = np.inf
    converged = False

    for iteration in range(1, max_iterations + 1):
        # Stability check via existing infrastructure
        stab = analyse_multi_zone_stability(
            H_blocks=H_blocks,
            Q_obj_list=Q_obj_list,
            G_w_list=gw_list,
            alpha_list=alpha_list,
            zone_ids=zone_ids,
            actuator_counts=actuator_counts,
            verbose=False,
        )
        gamma = stab.small_gain_gamma

        if verbose:
            print(f"  [tune] iter {iteration:2d}: "
                  f"gamma = {gamma:.4f} (target {gamma_target:.2f}), "
                  f"alpha = [{', '.join(f'{a:.4g}' for a in alpha_list)}]")

        if gamma < gamma_target:
            converged = True
            break

        # Identify bottleneck zone (highest Lyapunov row sum)
        row_sums = [zr.lyapunov_row_sum for zr in stab.zones]
        bottleneck_idx = int(np.argmax(row_sums))
        bn = stab.zones[bottleneck_idx]

        # Non-uniform g_w scaling to reduce condition number kappa.
        # Boost g_w for actuators with highest curvature (M_ii diagonal)
        # to flatten the eigenspectrum.
        C_ii = C_matrices[bottleneck_idx]
        g_w = gw_list[bottleneck_idx]
        gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
        M_ii_loc = (gw_inv_sqrt[:, None] * C_ii) * gw_inv_sqrt[None, :]
        M_diag = np.diag(M_ii_loc)
        m_max = np.max(M_diag)

        if m_max > 1e-14:
            # Scale more aggressively for high-curvature actuators
            scale = 1.0 + 1.0 * (M_diag / m_max)   # range [1, 2]
            gw_list[bottleneck_idx] = _apply_gw_floors(
                g_w * scale,
                actuator_counts[bottleneck_idx],
                min_gw, min_gw_discrete, gen_gw,
            )

        # Recompute optimal alpha for bottleneck zone
        alpha_list[bottleneck_idx] = _optimal_alpha_for_zone(
            C_ii, gw_list[bottleneck_idx], alpha_list[bottleneck_idx],
        )

        # Cap alpha by coupling-aware upper bound for all zones
        for i_idx in range(n_zones):
            zr = stab.zones[i_idx]
            if zr.alpha_max_coupled < alpha_list[i_idx]:
                alpha_list[i_idx] = 0.9 * zr.alpha_max_coupled

    # Fallback if not converged: reduce all alpha
    if not converged and gamma > 0:
        reduction = gamma_target / gamma
        alpha_list = [a * reduction for a in alpha_list]
        if verbose:
            print(f"  [tune] not converged after {max_iterations} iterations. "
                  f"Applying fallback alpha reduction by {reduction:.4f}.")

    # ── Final validation ─────────────────────────────────────────────────────
    stab_final = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_list,
        alpha_list=alpha_list,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=verbose,
    )

    # ── Package results ──────────────────────────────────────────────────────
    zone_results: List[ZoneTuningResult] = []
    for idx, zr in enumerate(stab_final.zones):
        zone_results.append(ZoneTuningResult(
            zone_id=zone_ids[idx],
            g_w=gw_list[idx],
            alpha=alpha_list[idx],
            alpha_max_local=zr.alpha_max_local,
            alpha_max_coupled=zr.alpha_max_coupled,
            rho=zr.rho_i,
            kappa=zr.kappa_Mii,
            lambda_min=zr.lambda_min_Mii,
            lambda_max=zr.lambda_max_Mii,
            actuator_labels=_build_actuator_labels(actuator_counts[idx]),
        ))

    n_iter = iteration if not converged else iteration  # noqa: F821
    return TuningResult(
        zones=zone_results,
        small_gain_gamma=stab_final.small_gain_gamma,
        converged=stab_final.small_gain_gamma < gamma_target,
        iterations=n_iter,
        stability_result=stab_final,
    )


# =============================================================================
#  DSO cascade tuning
# =============================================================================

def tune_dso(
    H_dso: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    n_der: int,
    n_oltc: int,
    n_shunt: int = 0,
    alpha: float = 1.0,
    *,
    safety_factor: float = 2.0,
    min_gw: float = 0.01,
    min_gw_discrete: float = 40.0,
    tso_period_s: float = 180.0,
    dso_period_s: float = 60.0,
    cascade_margin_target: float = 0.3,
) -> Tuple[NDArray[np.float64], float, float]:
    """Tune per-actuator g_w for a DSO controller with cascade margin.

    Ensures the DSO converges within the TSO period by checking:

        cascade_margin = 1 - rho_D^(T_T / T_D) > cascade_margin_target

    where rho_D is the spectral contraction rate of the DSO iteration.

    Parameters
    ----------
    H_dso :
        DSO sensitivity matrix (n_y_dso, n_u_dso).
    q_obj_diag :
        Per-output objective weight vector.
    n_der, n_oltc, n_shunt :
        Actuator counts (column order: [DER | OLTC | shunt]).
    alpha :
        DSO step size.
    safety_factor, min_gw, min_gw_discrete :
        Same semantics as compute_optimal_gw.
    tso_period_s, dso_period_s :
        Controller periods.  The ratio T_T / T_D determines how many
        DSO iterations must converge within one TSO period.
    cascade_margin_target :
        Target for 1 - rho_D^(T_T/T_D).  Default 0.3.

    Returns
    -------
    g_w : NDArray
        Per-actuator g_w vector.
    rho_D : float
        Spectral contraction rate.
    cascade_margin : float
        1 - rho_D^(n_inner).
    """
    n_inner = max(int(tso_period_s / dso_period_s), 1)

    # Curvature
    C_dso = _compute_curvature_matrix(H_dso, q_obj_diag)
    c_diag = np.diag(C_dso)

    # Initial g_w from Gershgorin
    g_w = safety_factor * alpha * c_diag / 2.0

    # Apply floors -- DSO column order: [DER | OLTC | shunt]
    n_u = len(g_w)
    g_w[:n_der] = np.maximum(g_w[:n_der], min_gw)
    g_w[n_der:n_der + n_oltc] = np.maximum(
        g_w[n_der:n_der + n_oltc], min_gw_discrete)
    g_w[n_der + n_oltc:] = np.maximum(
        g_w[n_der + n_oltc:], min_gw_discrete)

    # The cascade margin depends on rho_D = (kappa-1)/(kappa+1) at optimal
    # alpha.  Uniform g_w scaling doesn't change kappa.  To improve the
    # margin, we scale g_w non-uniformly: boost actuators with the
    # smallest M_dso eigenvalue contributions to reduce the condition
    # number.

    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M_dso = (gw_inv_sqrt[:, None] * C_dso) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M_dso)

    if len(active) == 0:
        return g_w, 0.0, 1.0

    # Compute optimal alpha and check initial margin
    l_min, l_max = float(active[0]), float(active[-1])
    alpha_opt = 2.0 / (l_min + l_max) if len(active) >= 2 else 1.0 / l_max
    rho_D = (l_max - l_min) / (l_max + l_min) if len(active) >= 2 else 0.0
    cascade_margin = 1.0 - rho_D ** n_inner

    if cascade_margin >= cascade_margin_target:
        return g_w, rho_D, cascade_margin

    # Non-uniform scaling: boost g_w for actuators with highest
    # curvature to flatten the eigenspectrum
    for attempt in range(20):
        # Identify actuators with highest M_dso diagonal (high curvature)
        M_diag = np.diag(M_dso)
        if np.max(M_diag) < 1e-14:
            break

        # Scale g_w proportionally to curvature: high-curvature
        # actuators get a bigger boost
        scale = 1.0 + 1.0 * (M_diag / np.max(M_diag))  # range [1, 2]
        g_w *= scale

        # Re-apply floors
        g_w[:n_der] = np.maximum(g_w[:n_der], min_gw)
        g_w[n_der:n_der + n_oltc] = np.maximum(
            g_w[n_der:n_der + n_oltc], min_gw_discrete)
        g_w[n_der + n_oltc:] = np.maximum(
            g_w[n_der + n_oltc:], min_gw_discrete)

        # Recompute
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
