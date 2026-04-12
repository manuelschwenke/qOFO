"""
Auto-Tuning for the Three-Condition Stability Framework
========================================================

Tunes g_w parameters to satisfy all three conditions of Theorem 3.3:

    C1: DSO inner loops  --  Gershgorin preconditioning
    C2: TSO continuous   --  Gershgorin + eigenvector-pump
    C3: TSO discrete     --  G sizing rule (Corollary 3.2, closed-form)

Functions
---------
auto_tune
    Main entry point.  Orchestrates DSO -> continuous TSO -> discrete TSO.

tune_continuous_gw
    Phase 2: Spectral-condition tuning for continuous actuators.

tune_discrete_gw
    Corollary 3.2: closed-form lower bound on discrete g_w.

tune_dso_gw
    Per-DSO g_w tuning with cascade-margin enforcement.

Author: Manuel Schwenke, TU Darmstadt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ── Constants ──────────────────────────────────────────────────────────────────

# Default TSO column order (matches ZoneDefinition.gw_diagonal())
_TSO_COLUMN_ORDER = ('der', 'pcc', 'gen', 'oltc', 'shunt')

# Default DSO column order
_DSO_COLUMN_ORDER = ('der', 'oltc', 'shunt')

_TYPE_TO_COUNT_KEY = {
    'der':   'n_der',
    'pcc':   'n_pcc',
    'gen':   'n_gen',
    'oltc':  'n_oltc',
    'shunt': 'n_shunt',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TuningResult:
    """Complete tuning result for the multi-zone system."""

    gw_tso_list: List[NDArray[np.float64]]
    """Tuned per-zone TSO g_w vectors (full, including discrete)."""

    gw_dso_list: List[NDArray[np.float64]]
    """Tuned per-DSO g_w vectors."""

    alpha_tso: float
    """Tuned TSO step-size (from continuous stability)."""

    alpha_dso_list: List[float]
    """Tuned per-DSO step-sizes."""

    c1_feasible: bool
    """True iff all DSOs satisfy C1 after tuning."""

    c2_feasible: bool
    """True iff C2 satisfied after tuning."""

    c3_feasible: bool
    """True iff C3 satisfied after tuning."""

    feasible: bool
    """True iff all three conditions satisfied."""

    discrete_gw_applied: Dict[int, Dict[str, float]] = field(default_factory=dict)
    """Per-zone, per-discrete-actuator: final g_w after C3 tuning."""

    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Private Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _curvature_diagonal(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return diag(H^T Q H) without forming the full matrix."""
    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    QH = q_sqrt[:, None] * H
    return np.sum(QH ** 2, axis=0)


def _curvature_matrix(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return C = H^T Q H."""
    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    QH = q_sqrt[:, None] * H
    return QH.T @ QH


def _apply_gw_floors(
    g_w: NDArray[np.float64],
    actuator_counts: Dict[str, int],
    floors: Dict[str, float],
    column_order: Tuple[str, ...] = _TSO_COLUMN_ORDER,
) -> NDArray[np.float64]:
    """Enforce per-actuator-type minimum g_w values."""
    g_w = g_w.copy()
    off = 0
    for atype in column_order:
        count_key = _TYPE_TO_COUNT_KEY.get(atype)
        if count_key is None:
            continue
        n = int(actuator_counts.get(count_key, 0))
        if n <= 0:
            continue
        floor = float(floors.get(atype, 0.0))
        if floor > 0.0:
            g_w[off:off + n] = np.maximum(g_w[off:off + n], floor)
        off += n
    return g_w


def _find_alpha_for_target_rho(
    eigs: NDArray,
    rho_target: float = 0.95,
) -> float:
    """Binary search for largest alpha with rho(I - alpha*M) <= rho_target.

    Handles complex eigenvalues correctly.
    """
    if len(eigs) == 0:
        return 1.0

    # Filter near-zero eigenvalues
    eig_max_abs = max(float(np.max(np.abs(eigs))), 1e-14)
    tol = 0.01 * eig_max_abs
    active = eigs[np.abs(eigs) > tol]
    if len(active) == 0:
        return 1.0

    def rho_at(alpha):
        return float(np.max(np.abs(1.0 - alpha * active)))

    # Upper bound: alpha_crit = 2 / max(Re(lambda))
    lam_max_re = float(np.max(active.real))
    if lam_max_re <= 1e-14:
        return 1.0
    alpha_hi = 2.0 / lam_max_re

    # Ternary search for alpha_opt (minimum rho)
    lo, hi = 0.0, alpha_hi
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if rho_at(m1) < rho_at(m2):
            hi = m2
        else:
            lo = m1
    alpha_opt = (lo + hi) / 2
    rho_opt = rho_at(alpha_opt)

    if rho_opt > rho_target:
        # Cannot achieve target -- return alpha_opt
        return alpha_opt

    # Binary search for largest alpha >= alpha_opt with rho <= target
    lo_b, hi_b = alpha_opt, alpha_hi
    for _ in range(60):
        mid = (lo_b + hi_b) / 2
        if rho_at(mid) <= rho_target:
            lo_b = mid
        else:
            hi_b = mid
    return lo_b


def _partition_indices(ac: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """(continuous_indices, discrete_indices) for TSO column ordering."""
    n_der = ac.get('n_der', 0)
    n_pcc = ac.get('n_pcc', 0)
    n_gen = ac.get('n_gen', 0)
    n_oltc = ac.get('n_oltc', 0)
    n_shunt = ac.get('n_shunt', 0)
    n_cont = n_der + n_pcc + n_gen
    n_disc = n_oltc + n_shunt
    return list(range(n_cont)), list(range(n_cont, n_cont + n_disc))


# ═══════════════════════════════════════════════════════════════════════════════
#  Eigenvector-pump iteration (shared by C1 and C2)
# ═══════════════════════════════════════════════════════════════════════════════

def _eigenvector_pump_gw(
    gw: NDArray[np.float64],
    M_builder,
    *,
    rho_target: float = 0.95,
    max_iters: int = 30,
    boost_factor: float = 1.5,
) -> Tuple[NDArray[np.float64], float, int]:
    """Iteratively boost g_w entries along dominant eigenvector of M.

    M_builder(gw) -> (M, eigs):
        Callable that builds the preconditioned matrix from g_w and
        returns (M_matrix, eigenvalues).

    Returns (gw_tuned, rho_achieved, n_iters).
    """
    gw = gw.copy()
    n = len(gw)

    for it in range(max_iters):
        M, eigs = M_builder(gw)

        # Filter active eigenvalues
        eig_max_abs = max(float(np.max(np.abs(eigs))), 1e-14)
        active_mask = np.abs(eigs) > 0.01 * eig_max_abs
        eigs_active = eigs[active_mask]

        if len(eigs_active) == 0:
            return gw, 0.0, it

        rho = float(np.max(np.abs(1.0 - eigs_active)))
        if rho <= rho_target:
            return gw, rho, it

        # Find dominant eigenvalue (largest |1 - lambda|)
        contraction = np.abs(1.0 - eigs)
        dom_idx = int(np.argmax(contraction))
        lam_dom = float(eigs[dom_idx].real)

        # Get eigenvector for direction
        if M.shape[0] == M.shape[1]:
            # Check symmetry for solver choice
            asym = float(np.linalg.norm(M - M.T, 'fro'))
            if asym < 1e-10 * max(float(np.linalg.norm(M, 'fro')), 1e-14):
                _, vecs = np.linalg.eigh(M)
                v = vecs[:, -1]  # eigenvector of largest eigenvalue
            else:
                all_eigs_full, vecs = np.linalg.eig(M)
                idx = int(np.argmax(all_eigs_full.real))
                v = np.abs(vecs[:, idx].real)
        else:
            v = np.ones(n)

        # Squared participation weights
        v_sq = v ** 2
        v_sq = v_sq / max(np.sum(v_sq), 1e-14)

        # Boost g_w proportional to participation
        # Actuators that participate most in the dominant mode get boosted
        boost = 1.0 + (boost_factor - 1.0) * v_sq / max(np.max(v_sq), 1e-14)
        gw = gw * boost

    # Final check
    _, eigs_final = M_builder(gw)
    eig_max_abs = max(float(np.max(np.abs(eigs_final))), 1e-14)
    active = eigs_final[np.abs(eigs_final) > 0.01 * eig_max_abs]
    rho_final = float(np.max(np.abs(1.0 - active))) if len(active) > 0 else 0.0
    return gw, rho_final, max_iters


# ═══════════════════════════════════════════════════════════════════════════════
#  C1: DSO g_w tuning
# ═══════════════════════════════════════════════════════════════════════════════

def tune_dso_gw(
    H_dso: NDArray[np.float64],
    Q_dso: NDArray[np.float64],
    G_w_dso_init: NDArray[np.float64],
    *,
    safety_factor_continuous: float = 2.0,
    safety_factor_discrete: float = 3.0,
    floors: Optional[Dict[str, float]] = None,
    actuator_counts: Optional[Dict[str, int]] = None,
    rho_target: float = 0.95,
) -> Tuple[NDArray[np.float64], float]:
    """Tune DSO g_w to satisfy C1: rho(M_cont) < 1.

    Targets the CONTINUOUS-ONLY M_cont = I - (G_c)^{-1} Phi_c,
    not the full DSO M.

    Phase 1: Gershgorin preconditioning on continuous curvature.
    Phase 2: Eigenvector pump if Gershgorin alone doesn't achieve target.

    Returns (g_w_tuned_full, alpha_dso).  The full g_w vector (continuous
    + discrete) is returned; discrete entries keep their init values
    boosted by safety_factor_discrete.
    """
    if floors is None:
        floors = {}

    n_u = H_dso.shape[1]

    # Partition into continuous / discrete
    if actuator_counts is not None:
        n_der = actuator_counts.get('n_der', 0)
        n_oltc = actuator_counts.get('n_oltc', 0)
        n_shunt = actuator_counts.get('n_shunt', 0)
        cont_idx = list(range(n_der))
        disc_idx = list(range(n_der, n_der + n_oltc + n_shunt))
    else:
        cont_idx = list(range(n_u))
        disc_idx = []

    n_cont = len(cont_idx)

    # --- Phase 1: Gershgorin on CONTINUOUS columns only ---
    K_c = H_dso[:, cont_idx]
    Q_sqrt = np.sqrt(np.maximum(Q_dso, 0.0))

    # Continuous curvature diagonal: diag((K_c)^T Q K_c)
    QK_c = Q_sqrt[:, None] * K_c
    c_diag_cont = np.sum(QK_c ** 2, axis=0)  # Phi_c diagonal (ignoring R)

    gw_cont = safety_factor_continuous * c_diag_cont / 2.0
    gw_cont = np.maximum(gw_cont, G_w_dso_init[cont_idx])

    # --- Phase 2: Eigenvector pump on M_cont ---
    def _build_M_cont(gw_c):
        gi = 1.0 / np.sqrt(np.maximum(gw_c, 1e-12))
        Phi_c = 2.0 * (QK_c.T @ QK_c)
        M_c = (gi[:, None] * Phi_c) * gi[None, :]
        eigs = np.linalg.eigvalsh(M_c)
        return M_c, eigs

    if n_cont > 0:
        gw_cont, rho, _ = _eigenvector_pump_gw(
            gw_cont, _build_M_cont, rho_target=rho_target,
        )

    # --- Discrete: just use safety-factored curvature ---
    gw_full = G_w_dso_init.copy()
    if n_cont > 0:
        gw_full[cont_idx] = gw_cont
    if disc_idx:
        K_d = H_dso[:, disc_idx]
        QK_d = Q_sqrt[:, None] * K_d
        c_diag_disc = np.sum(QK_d ** 2, axis=0)
        gw_disc = safety_factor_discrete * c_diag_disc / 2.0
        gw_disc = np.maximum(gw_disc, G_w_dso_init[disc_idx])
        gw_full[disc_idx] = gw_disc

    # Apply floors
    if actuator_counts is not None:
        gw_full = _apply_gw_floors(gw_full, actuator_counts, floors, _DSO_COLUMN_ORDER)

    # Compute alpha from full DSO M (including discrete)
    C_full = _curvature_matrix(H_dso, Q_dso)
    gw_inv_sqrt_full = 1.0 / np.sqrt(np.maximum(gw_full, 1e-12))
    M_full = (gw_inv_sqrt_full[:, None] * C_full) * gw_inv_sqrt_full[None, :]
    eigs_full = np.linalg.eigvalsh(M_full).astype(np.complex128)
    alpha = _find_alpha_for_target_rho(eigs_full, rho_target=rho_target)

    return gw_full, alpha


# ═══════════════════════════════════════════════════════════════════════════════
#  C2: Continuous g_w tuning (targets M_full^c, NOT full M_sys)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_continuous_gw(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    G_w_init_list: List[NDArray[np.float64]],
    *,
    zone_ids: Optional[List[int]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
    safety_factor: float = 2.0,
    floors: Optional[Dict[str, float]] = None,
    rho_target: float = 0.95,
    max_pump_iters: int = 30,
    verbose: bool = False,
) -> Tuple[List[NDArray[np.float64]], float]:
    """Tune continuous g_w entries to satisfy C2: rho(M_full^c) < 1.

    Targets the CONTINUOUS-ONLY system matrix M_full^c (Theorem 3.1).
    Discrete columns of H are excluded.  Discrete g_w entries in the
    returned vectors are copied from init (tuned separately in C3).

    Phase 1: Gershgorin preconditioning on continuous curvature.
    Phase 2: Eigenvector pump on M_full^c until rho < rho_target.

    Returns (gw_tuned_list, alpha_tso).
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))
    if floors is None:
        floors = {}

    # Extract continuous indices per zone
    cont_indices: List[List[int]] = []
    for k in range(n_zones):
        if actuator_counts is not None:
            ci, _ = _partition_indices(actuator_counts[k])
        else:
            ci = list(range(len(G_w_init_list[k])))
        cont_indices.append(ci)

    # --- Phase 1: Gershgorin on per-zone CONTINUOUS curvature ---
    gw_list = [gw.copy() for gw in G_w_init_list]

    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks.get((z, z))
        ci = cont_indices[idx]
        if H_ii is None or not ci:
            continue

        K_ii_c = H_ii[:, ci]
        Q_sqrt_i = np.sqrt(np.maximum(Q_obj_list[idx], 0.0))
        QK_c = Q_sqrt_i[:, None] * K_ii_c
        c_diag = np.sum(QK_c ** 2, axis=0)

        gw_cont = safety_factor * c_diag / 2.0
        gw_cont = np.maximum(gw_cont, G_w_init_list[idx][ci])
        gw_list[idx][ci] = gw_cont

    # Apply floors to full vectors
    for idx in range(n_zones):
        if actuator_counts is not None:
            gw_list[idx] = _apply_gw_floors(
                gw_list[idx], actuator_counts[idx], floors, _TSO_COLUMN_ORDER,
            )

    # --- Phase 2: Eigenvector pump on M_full^c ---
    n_c_per = [len(ci) for ci in cont_indices]
    n_total_c = sum(n_c_per)

    if n_total_c == 0:
        return gw_list, 1.0

    # Flatten continuous g_w into a single vector for the pump
    gw_c_flat = np.concatenate([gw_list[k][cont_indices[k]] for k in range(n_zones)])

    # Precompute Q-weighted H blocks (continuous columns only)
    QH_cache = {}
    for i_idx, i in enumerate(zone_ids):
        ci_i = cont_indices[i_idx]
        if not ci_i:
            continue
        Q_sqrt_i = np.sqrt(np.maximum(Q_obj_list[i_idx], 0.0))
        H_ii = H_blocks.get((i, i))
        if H_ii is not None:
            QH_cache[('QH_ii', i)] = Q_sqrt_i[:, None] * H_ii[:, ci_i]
        for j_idx, j in enumerate(zone_ids):
            H_ij = H_blocks.get((i, j))
            ci_j = cont_indices[j_idx]
            if H_ij is not None and ci_j:
                QH_cache[('QH', i, j)] = Q_sqrt_i[:, None] * H_ij[:, ci_j]

    def _build_M_full_c(gw_c_flat_in):
        """Build M_full^c from flattened continuous g_w."""
        M = np.zeros((n_total_c, n_total_c))
        # Unflatten
        off_list = []
        off = 0
        for k in range(n_zones):
            off_list.append(off)
            off += n_c_per[k]

        for i_idx, i in enumerate(zone_ids):
            if n_c_per[i_idx] == 0:
                continue
            r0 = off_list[i_idx]
            gw_i = gw_c_flat_in[r0:r0 + n_c_per[i_idx]]
            gi = 1.0 / np.sqrt(np.maximum(gw_i, 1e-12))
            QH_ii = QH_cache.get(('QH_ii', i))

            for j_idx, j in enumerate(zone_ids):
                if n_c_per[j_idx] == 0:
                    continue
                c0 = off_list[j_idx]
                gw_j = gw_c_flat_in[c0:c0 + n_c_per[j_idx]]
                gj = 1.0 / np.sqrt(np.maximum(gw_j, 1e-12))

                QH_ij = QH_cache.get(('QH', i, j))
                if QH_ii is None or QH_ij is None:
                    continue
                C_ij = 2.0 * (QH_ii.T @ QH_ij)
                M_ij = (gi[:, None] * C_ij) * gj[None, :]
                M[r0:r0+n_c_per[i_idx], c0:c0+n_c_per[j_idx]] = M_ij

        eigs = np.linalg.eigvals(M)
        return M, eigs

    gw_c_tuned, rho, n_iters = _eigenvector_pump_gw(
        gw_c_flat, _build_M_full_c,
        rho_target=rho_target, max_iters=max_pump_iters,
    )

    # Write tuned continuous g_w back into per-zone vectors
    off = 0
    for k in range(n_zones):
        ci = cont_indices[k]
        gw_list[k][ci] = gw_c_tuned[off:off + n_c_per[k]]
        off += n_c_per[k]

    # Compute alpha from the full system (all actuators)
    n_per_zone = [len(gw) for gw in gw_list]
    n_total = sum(n_per_zone)
    M_sys = np.zeros((n_total, n_total))
    row_off = 0
    for i_idx, i in enumerate(zone_ids):
        q_obj_i = Q_obj_list[i_idx]
        gw_i = gw_list[i_idx]
        gw_inv_sqrt_i = 1.0 / np.sqrt(np.maximum(gw_i, 1e-12))
        Q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))
        H_ii = H_blocks.get((i, i))
        if H_ii is None:
            row_off += n_per_zone[i_idx]
            continue
        QH_ii = Q_sqrt_i[:, None] * H_ii
        col_off = 0
        for j_idx, j in enumerate(zone_ids):
            gw_j = gw_list[j_idx]
            gw_inv_sqrt_j = 1.0 / np.sqrt(np.maximum(gw_j, 1e-12))
            H_ij = H_blocks.get((i, j))
            if H_ij is not None:
                QH_ij = Q_sqrt_i[:, None] * H_ij
                C_ij = QH_ii.T @ QH_ij
                M_ij = (gw_inv_sqrt_i[:, None] * C_ij) * gw_inv_sqrt_j[None, :]
                nr, nc = n_per_zone[i_idx], n_per_zone[j_idx]
                M_sys[row_off:row_off+nr, col_off:col_off+nc] = M_ij
            col_off += n_per_zone[j_idx]
        row_off += n_per_zone[i_idx]

    sys_eigs = np.linalg.eigvals(M_sys)
    alpha_tso = _find_alpha_for_target_rho(sys_eigs, rho_target=rho_target)

    if verbose:
        lam_max_re = float(np.max(sys_eigs.real)) if len(sys_eigs) > 0 else 0.0
        rho_at = float(np.max(np.abs(1.0 - alpha_tso * sys_eigs))) if len(sys_eigs) > 0 else 0.0
        print(f"  [C2 tune] rho(M_full^c) = {rho:.4f} after {n_iters} pump iters, "
              f"alpha_tso = {alpha_tso:.6f}, rho(full) = {rho_at:.4f}")

    return gw_list, alpha_tso


# ═══════════════════════════════════════════════════════════════════════════════
#  C3: Discrete g_w tuning (G sizing rule, Corollary 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_discrete_gw(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    G_w_list: List[NDArray[np.float64]],
    *,
    zone_ids: Optional[List[int]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
    safety_factor: float = 1.5,
    verbose: bool = False,
) -> List[NDArray[np.float64]]:
    """Tune discrete g_w entries to satisfy C3 via Corollary 3.2.

    The G sizing rule gives a closed-form lower bound:

        g_{i,a} >= safety_factor * 4 * sum_{j!=i} ||[P_ij]_{a,.}||_1

    where P_ij = (K_ii^d)^T Q_i K_ij^d.

    Only modifies discrete entries (OLTC, shunt) of G_w.  Continuous
    entries are untouched.

    Returns updated gw_list (copies).
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))
    if actuator_counts is None:
        return [gw.copy() for gw in G_w_list]

    gw_out = [gw.copy() for gw in G_w_list]

    # Get discrete indices per zone
    disc_indices = []
    for k in range(n_zones):
        _, di = _partition_indices(actuator_counts[k])
        disc_indices.append(di)

    # Build P_ij blocks and compute g_min per discrete actuator
    for i_idx, i in enumerate(zone_ids):
        di_i = disc_indices[i_idx]
        n_d_i = len(di_i)
        if n_d_i == 0:
            continue

        q_obj_i = Q_obj_list[i_idx]
        Q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))

        H_ii = H_blocks.get((i, i))
        if H_ii is None:
            continue

        K_ii_d = H_ii[:, di_i]
        QK_ii_d = Q_sqrt_i[:, None] * K_ii_d

        # Accumulate g_min per discrete actuator
        g_min = np.zeros(n_d_i)

        for j_idx, j in enumerate(zone_ids):
            if j == i:
                continue
            di_j = disc_indices[j_idx]
            n_d_j = len(di_j)
            if n_d_j == 0:
                continue

            H_ij = H_blocks.get((i, j))
            if H_ij is None:
                continue

            K_ij_d = H_ij[:, di_j]
            QK_ij_d = Q_sqrt_i[:, None] * K_ij_d

            P_ij = QK_ii_d.T @ QK_ij_d  # (n_d_i, n_d_j)
            row_l1 = np.sum(np.abs(P_ij), axis=1)  # (n_d_i,)
            g_min += 4.0 * row_l1

        # Apply safety factor and update
        g_min_safe = safety_factor * g_min
        for a_local, a_global in enumerate(di_i):
            gw_out[i_idx][a_global] = max(
                float(gw_out[i_idx][a_global]),
                float(g_min_safe[a_local]),
            )

        if verbose and np.any(g_min_safe > 0):
            n_oltc = actuator_counts[i_idx].get('n_oltc', 0)
            n_shunt = actuator_counts[i_idx].get('n_shunt', 0)
            names = ([f'OLTC_{k}' for k in range(n_oltc)]
                     + [f'Shunt_{k}' for k in range(n_shunt)])
            for a in range(n_d_i):
                name = names[a] if a < len(names) else f'd_{a}'
                old = float(G_w_list[i_idx][di_i[a]])
                new = float(gw_out[i_idx][di_i[a]])
                if new > old:
                    print(f"  [C3] Zone {i} {name}: g_w {old:.2f} -> {new:.2f} "
                          f"(g_min = {float(g_min_safe[a]):.2f})")

    return gw_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSOTuneInput:
    """Input data for one DSO."""
    H: NDArray[np.float64]
    q_obj_diag: NDArray[np.float64]
    n_der: int
    n_oltc: int
    n_shunt: int
    dso_id: str = "DSO"
    zone_id: int = 0


def auto_tune(
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    gw_tso_init: List[NDArray[np.float64]],
    # DSO data
    dso_inputs: Optional[List[DSOTuneInput]] = None,
    gw_dso_init: Optional[List[NDArray[np.float64]]] = None,
    # Tuning parameters
    safety_factor_continuous: float = 2.0,
    safety_factor_discrete: float = 1.5,
    rho_target: float = 0.95,
    floors_tso: Optional[Dict[str, float]] = None,
    floors_dso: Optional[Dict[str, float]] = None,
    # Cascade
    tso_period_s: float = 180.0,
    dso_period_s: float = 20.0,
    verbose: bool = False,
) -> TuningResult:
    """Orchestrate three-condition auto-tuning.

    Sequence: DSO (C1) -> continuous TSO (C2) -> discrete TSO (C3).

    Parameters
    ----------
    H_blocks : TSO sensitivity blocks.
    Q_obj_list : Per-zone Q_obj diagonals.
    actuator_counts : Per-zone actuator count dicts.
    zone_ids : Zone integer IDs.
    gw_tso_init : Initial per-zone TSO g_w (user hand-tuned, used as floor).
    dso_inputs : List of DSOTuneInput.
    gw_dso_init : Initial per-DSO g_w vectors.
    """
    if floors_tso is None:
        floors_tso = {}
    if floors_dso is None:
        floors_dso = {}
    if dso_inputs is None:
        dso_inputs = []
    if gw_dso_init is None:
        gw_dso_init = []

    warnings: List[str] = []

    # ── C1: DSO tuning ────────────────────────────────────────────────────
    gw_dso_tuned: List[NDArray[np.float64]] = []
    alpha_dso_list: List[float] = []
    c1_ok = True

    for d_idx, d in enumerate(dso_inputs):
        init = gw_dso_init[d_idx] if d_idx < len(gw_dso_init) else np.ones(d.H.shape[1])
        ac = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        gw, alpha = tune_dso_gw(
            H_dso=d.H, Q_dso=d.q_obj_diag, G_w_dso_init=init,
            safety_factor_continuous=safety_factor_continuous,
            safety_factor_discrete=safety_factor_discrete * 2,  # extra margin for DSO discrete
            floors=floors_dso,
            actuator_counts=ac,
            rho_target=rho_target,
        )
        gw_dso_tuned.append(gw)
        alpha_dso_list.append(alpha)

        if verbose:
            print(f"  [C1] {d.dso_id}: alpha = {alpha:.4f}")

    # ── C2: Continuous TSO tuning ─────────────────────────────────────────
    gw_tso, alpha_tso = tune_continuous_gw(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_init_list=gw_tso_init,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        safety_factor=safety_factor_continuous,
        floors=floors_tso,
        rho_target=rho_target,
        verbose=verbose,
    )

    # ── C3: Discrete TSO tuning ───────────────────────────────────────────
    gw_tso = tune_discrete_gw(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        safety_factor=safety_factor_discrete,
        verbose=verbose,
    )

    # ── Verify ────────────────────────────────────────────────────────────
    from analysis.stability_analysis import (
        analyse_multi_zone_stability,
        analyse_dso_stability,
    )

    # Verify C1
    for d_idx, d in enumerate(dso_inputs):
        ac = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        r = analyse_dso_stability(
            H_dso=d.H, Q_dso=d.q_obj_diag, G_w_dso=gw_dso_tuned[d_idx],
            dso_id=d.dso_id, actuator_counts=ac,
            tso_period_s=tso_period_s, dso_period_s=dso_period_s,
        )
        if not r.stable:
            c1_ok = False
            warnings.append(f"C1: {d.dso_id} still unstable after tuning")

    # Verify C2 + C3
    stab = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=False,
    )
    c2_ok = stab.c2_satisfied
    c3_ok = stab.c3_satisfied

    if not c2_ok:
        warnings.append("C2: continuous stability not achieved after tuning")
    if not c3_ok:
        warnings.append("C3: discrete small-gain not achieved after tuning")

    # Build discrete g_w report
    disc_gw_applied: Dict[int, Dict[str, float]] = {}
    for i_idx, i in enumerate(zone_ids):
        _, di = _partition_indices(actuator_counts[i_idx])
        if not di:
            continue
        n_oltc = actuator_counts[i_idx].get('n_oltc', 0)
        n_shunt = actuator_counts[i_idx].get('n_shunt', 0)
        names = ([f'OLTC_{k}' for k in range(n_oltc)]
                 + [f'Shunt_{k}' for k in range(n_shunt)])
        disc_gw_applied[i] = {}
        for a_local, a_global in enumerate(di):
            name = names[a_local] if a_local < len(names) else f'd_{a_local}'
            disc_gw_applied[i][name] = float(gw_tso[i_idx][a_global])

    return TuningResult(
        gw_tso_list=gw_tso,
        gw_dso_list=gw_dso_tuned,
        alpha_tso=alpha_tso,
        alpha_dso_list=alpha_dso_list,
        c1_feasible=c1_ok,
        c2_feasible=c2_ok,
        c3_feasible=c3_ok,
        feasible=c1_ok and c2_ok and c3_ok,
        discrete_gw_applied=disc_gw_applied,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Stability-analysis exclusion helpers (carried over from v1)
# ═══════════════════════════════════════════════════════════════════════════════

def build_exclusion_mask(
    actuator_counts: Dict[str, int],
    col_order: Tuple[str, ...],
    exclude_types: set,
) -> NDArray[np.bool_]:
    """Boolean mask: True = keep, False = exclude."""
    segments: List[NDArray[np.bool_]] = []
    for atype in col_order:
        count_key = _TYPE_TO_COUNT_KEY.get(atype, f'n_{atype}')
        n = int(actuator_counts.get(count_key, 0))
        if n > 0:
            segments.append(np.full(n, atype not in exclude_types))
    return np.concatenate(segments) if segments else np.array([], dtype=bool)


def filter_stability_inputs(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    G_w_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    exclude_types: set,
    col_order: Tuple[str, ...] = _TSO_COLUMN_ORDER,
) -> Tuple[
    Dict[Tuple[int, int], NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[Dict[str, int]],
    List[NDArray[np.bool_]],
]:
    """Remove excluded actuator columns from stability inputs."""
    if not exclude_types:
        masks = [np.ones(len(gw), dtype=bool) for gw in G_w_list]
        return H_blocks, G_w_list, actuator_counts, masks

    masks: List[NDArray[np.bool_]] = []
    for idx in range(len(zone_ids)):
        masks.append(build_exclusion_mask(
            actuator_counts[idx], col_order, exclude_types,
        ))

    zid_to_idx = {z: idx for idx, z in enumerate(zone_ids)}

    H_filtered: Dict[Tuple[int, int], NDArray[np.float64]] = {}
    for (i, j), H_ij in H_blocks.items():
        j_idx = zid_to_idx.get(j)
        if j_idx is None:
            H_filtered[(i, j)] = H_ij
            continue
        H_filtered[(i, j)] = H_ij[:, masks[j_idx]]

    G_w_filtered = [gw[m] for gw, m in zip(G_w_list, masks)]

    counts_filtered: List[Dict[str, int]] = []
    for idx in range(len(zone_ids)):
        new_counts = dict(actuator_counts[idx])
        for atype in exclude_types:
            count_key = _TYPE_TO_COUNT_KEY.get(atype, f'n_{atype}')
            new_counts[count_key] = 0
        counts_filtered.append(new_counts)

    return H_filtered, G_w_filtered, counts_filtered, masks


def expand_gw_with_excluded(
    gw_filtered: NDArray[np.float64],
    gw_original: NDArray[np.float64],
    keep_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Re-insert original g_w values at excluded positions."""
    result = gw_original.copy()
    result[keep_mask] = gw_filtered
    return result
