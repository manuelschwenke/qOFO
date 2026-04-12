"""
Three-Condition Stability Analysis for OFO-MIQP Controllers
============================================================

Implements the full stability framework from Theorem 3.3:

    Stable  iff  C1 /\\ C2 /\\ C3

where:

    C1: rho(M_cont,j) < 1          for each DSO j       (DSO inner loops)
    C2: rho(M_full^c) < 1                                (multi-zone TSO continuous)
    C3: rho(Gamma) < 1                                    (multi-zone TSO discrete small-gain)

Theory
------
Part I  -- Single DSO (Theorem 1.2):
    Continuous iteration matrix  M_cont = I - (G_c)^{-1} Phi_c
    where Phi_c = 2 R_c + 2 (K_c)^T Q K_c  is the continuous Hessian.
    Discrete variables settle finitely (Proposition 1.3).

Part II -- Hierarchical TSO-DSO (Theorem 2.1):
    Cascaded decomposition via timescale separation.
    DSO converges within N_inner = T_TSO/T_DSO steps.

Part III -- Multi-TSO/DSO (Theorem 3.3):
    C2: M_full^c = I - blkdiag(G_i^c)^{-1} Phi_full^c
    C3: Gamma_ij = max_a (4/g_{i,a}) sum_b |[P_ij]_{ab}|
        where P_ij = (K_ii^d)^T Q_i K_ij^d

    G sizing rule (Corollary 3.2):
        g_{i,a} > 4 sum_{j!=i} ||[P_ij]_{a,.}||_1

Author: Manuel Schwenke, TU Darmstadt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ── Constants ──────────────────────────────────────────────────────────────────

_EIG_NULL_TOL_FRAC = 1e-12   # null-space threshold relative to lambda_max
_EIG_ACTIVE_TOL_FRAC = 0.01  # active-mode threshold relative to lambda_max


# ═══════════════════════════════════════════════════════════════════════════════
#  Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSOStabilityResult:
    """C1 -- Per-DSO stability analysis (Theorem 1.2).

    The DSO operates an OFO-MIQP with continuous (DER Q) and discrete
    (OLTC, shunt) actuators.  Continuous stability is checked via the
    spectral radius of M_cont.  Discrete variables settle finitely by
    Proposition 1.3 (single-zone, no cross-zone interaction).
    """

    dso_id: str
    """Identifier for this DSO controller."""

    # Continuous stability (Theorem 1.1)
    n_continuous: int
    """Number of continuous control variables."""

    n_discrete: int
    """Number of discrete control variables."""

    Phi_c_eigenvalues: NDArray[np.float64]
    """Eigenvalues of the continuous Hessian Phi_c (active modes)."""

    M_cont_spectral_radius: float
    """rho(M_cont) = max |1 - lambda_i(M_cont)| on active modes."""

    stable: bool
    """True iff rho(M_cont) < 1."""

    # Cascade margin (Assumption 2.2)
    N_inner: float
    """T_TSO / T_DSO: number of DSO iterations per TSO step."""

    cascade_decay: float
    """rho(M_cont)^N_inner: residual error fraction after one TSO step."""

    cascade_margin: float
    """1 - cascade_decay.  Positive means DSO converges within TSO step."""

    # Per-actuator diagnostics
    per_actuator_margins: Dict[str, float] = field(default_factory=dict)
    """Per continuous actuator: g_w[i] - Phi_c[i,i]/2.  Positive = safe."""

    per_actuator_gw_min: Dict[str, float] = field(default_factory=dict)
    """Per continuous actuator: Phi_c[i,i]/2 (Gershgorin necessary condition)."""

    warnings: List[str] = field(default_factory=list)


@dataclass
class ContinuousStabilityResult:
    """C2 -- Multi-zone TSO continuous stability (Theorem 3.1).

    With all discrete variables frozen, the multi-zone continuous OFO
    iteration matrix is M_full^c = I - blkdiag(G_i^c)^{-1} Phi_full^c.
    """

    # Global continuous stability
    M_full_c_eigenvalues: NDArray[np.float64]
    """Eigenvalues of M_full^c (active modes only)."""

    spectral_radius: float
    """rho(M_full^c) on active modes."""

    stable: bool
    """True iff rho(M_full^c) < 1."""

    # Per-zone continuous metrics
    per_zone_rho: Dict[int, float] = field(default_factory=dict)
    """Per-zone: rho_i = max |1 - lambda(M_ii^c)| on active modes."""

    per_zone_lambda_max: Dict[int, float] = field(default_factory=dict)
    """Per-zone: lambda_max(M_ii^c)."""

    per_zone_kappa: Dict[int, float] = field(default_factory=dict)
    """Per-zone: condition number kappa(M_ii^c)."""

    # Continuous coupling (Corollary 3.1)
    coupling_norms: Dict[Tuple[int, int], float] = field(default_factory=dict)
    """||M_ij^c||_2 for each (i, j), i != j."""

    small_gain_gamma: float = np.inf
    """Continuous small-gain: gamma = max_i {rho_i + sum_{j!=i} ||M_ij^c||_2}."""

    small_gain_stable: bool = False
    """True iff gamma < 1 (sufficient for C2)."""

    n_null_filtered: int = 0
    """Number of null-space eigenvalues filtered from M_full^c."""

    warnings: List[str] = field(default_factory=list)


@dataclass
class DiscreteSmallGainResult:
    """C3 -- Multi-zone discrete small-gain (Theorem 3.2).

    Prevents cross-zone tap cycling via the normalised interaction
    gain matrix Gamma.
    """

    # Gamma matrix and spectral radius
    Gamma: NDArray[np.float64]
    """N x N normalised discrete interaction gain matrix."""

    Gamma_spectral_radius: float
    """rho(Gamma).  Must be < 1 for C3."""

    stable: bool
    """True iff rho(Gamma) < 1."""

    # Underlying interaction matrices
    P_blocks: Dict[Tuple[int, int], NDArray[np.float64]] = field(default_factory=dict)
    """P_ij = (K_ii^d)^T Q_i K_ij^d for each (i, j)."""

    # G sizing rule (Corollary 3.2)
    g_min_required: Dict[int, Dict[str, float]] = field(default_factory=dict)
    """Per-zone, per-actuator: minimum g_w for C3.
    g_min[zone_id][actuator_name] = 4 * sum_{j!=i} ||[P_ij]_{a,.}||_1."""

    g_current: Dict[int, Dict[str, float]] = field(default_factory=dict)
    """Per-zone, per-actuator: current g_w value."""

    g_margin: Dict[int, Dict[str, float]] = field(default_factory=dict)
    """Per-zone, per-actuator: g_current - g_min_required.  Positive = safe."""

    # Row-sum sufficient condition
    Gamma_row_sums: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    """Per-zone: sum_j Gamma_ij.  Must be < 1 for each zone (Gershgorin)."""

    warnings: List[str] = field(default_factory=list)


@dataclass
class ZoneStabilityResult:
    """Per-zone summary combining continuous and discrete metrics."""

    zone_id: int
    zone_name: str
    n_continuous: int
    n_discrete: int
    n_outputs: int

    # Continuous (from C2)
    lambda_max_c: float
    """Largest eigenvalue of M_ii^c (local continuous curvature)."""

    rho_c: float
    """Local continuous contraction rate."""

    coupling_sum_c: float
    """Continuous coupling: sum_{j!=i} ||M_ij^c||_2."""

    # Discrete (from C3)
    Gamma_row_sum: float
    """Discrete coupling: sum_{j!=i} Gamma_ij.  < 1 for Gershgorin."""

    n_discrete_violations: int
    """Number of discrete actuators with g_w < g_min."""

    # DSO cascade (from C1)
    dso_results: List[DSOStabilityResult] = field(default_factory=list)


@dataclass
class MultiZoneStabilityResult:
    """Top-level: Theorem 3.3 combined verdict."""

    # Three conditions
    c1_dso: List[DSOStabilityResult]
    """C1: Per-DSO stability results."""

    c2_continuous: ContinuousStabilityResult
    """C2: Multi-zone continuous stability."""

    c3_discrete: DiscreteSmallGainResult
    """C3: Multi-zone discrete small-gain."""

    # Per-zone summaries
    zones: List[ZoneStabilityResult]

    # Combined verdict
    c1_satisfied: bool
    """True iff all DSOs satisfy C1."""

    c2_satisfied: bool
    """True iff C2 is satisfied."""

    c3_satisfied: bool
    """True iff C3 is satisfied."""

    stable: bool
    """True iff C1 AND C2 AND C3."""

    summary: str = ""

    # System matrices (for diagnostics / auto-tuning)
    M_full_c: Optional[NDArray[np.float64]] = None
    """Full continuous system matrix (for tuning access)."""

    Gamma: Optional[NDArray[np.float64]] = None
    """Discrete interaction gain matrix (for tuning access)."""

    recommendations: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _partition_indices(ac: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """Return (continuous_indices, discrete_indices) from actuator counts.

    Column ordering: [Q_DER | Q_PCC | V_gen | OLTC | shunt].
    Continuous = DER + PCC + gen.  Discrete = OLTC + shunt.
    """
    n_der = ac.get('n_der', 0)
    n_pcc = ac.get('n_pcc', 0)
    n_gen = ac.get('n_gen', 0)
    n_oltc = ac.get('n_oltc', 0)
    n_shunt = ac.get('n_shunt', 0)

    n_cont = n_der + n_pcc + n_gen
    n_disc = n_oltc + n_shunt

    cont_idx = list(range(n_cont))
    disc_idx = list(range(n_cont, n_cont + n_disc))
    return cont_idx, disc_idx


def _partition_dso_indices(ac: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """Return (continuous_indices, discrete_indices) for a DSO.

    Column ordering: [Q_DER | OLTC | shunt].
    """
    n_der = ac.get('n_der', 0)
    n_oltc = ac.get('n_oltc', 0)
    n_shunt = ac.get('n_shunt', 0)

    cont_idx = list(range(n_der))
    disc_idx = list(range(n_der, n_der + n_oltc + n_shunt))
    return cont_idx, disc_idx


def _discrete_actuator_names(ac: Dict[str, int]) -> List[str]:
    """Build names for discrete actuators in a zone."""
    n_oltc = ac.get('n_oltc', 0)
    n_shunt = ac.get('n_shunt', 0)
    return ([f'OLTC_{k}' for k in range(n_oltc)]
            + [f'Shunt_{k}' for k in range(n_shunt)])


def _continuous_actuator_names(ac: Dict[str, int]) -> List[str]:
    """Build names for continuous actuators in a zone."""
    return ([f'Q_DER_{k}' for k in range(ac.get('n_der', 0))]
            + [f'Q_PCC_{k}' for k in range(ac.get('n_pcc', 0))]
            + [f'V_gen_{k}' for k in range(ac.get('n_gen', 0))])


def _filter_active_eigenvalues(
    eigs: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], int]:
    """Filter null-space eigenvalues.  Returns (active_eigs, n_filtered)."""
    if len(eigs) == 0:
        return eigs, 0
    lam_max = float(np.max(np.abs(eigs)))
    tol = _EIG_ACTIVE_TOL_FRAC * max(lam_max, 1e-14)
    active = eigs[np.abs(eigs) > tol]
    return active, len(eigs) - len(active)


def _spectral_radius_iteration(eigs_active: NDArray[np.float64]) -> float:
    """Compute rho = max |1 - lambda_i| on active eigenvalues."""
    if len(eigs_active) == 0:
        return 0.0
    return float(np.max(np.abs(1.0 - eigs_active)))


# ═══════════════════════════════════════════════════════════════════════════════
#  C1: DSO Inner-Loop Stability (Theorem 1.2)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_dso_stability(
    H_dso: NDArray[np.float64],
    Q_dso: NDArray[np.float64],
    G_w_dso: NDArray[np.float64],
    G_u_dso: Optional[NDArray[np.float64]] = None,
    *,
    dso_id: str = "DSO",
    actuator_counts: Optional[Dict[str, int]] = None,
    tso_period_s: float = 180.0,
    dso_period_s: float = 20.0,
) -> DSOStabilityResult:
    """Analyse stability of a single DSO (Theorem 1.2).

    Parameters
    ----------
    H_dso : (n_y, n_u) sensitivity matrix.
    Q_dso : (n_y,) per-output objective weights.
    G_w_dso : (n_u,) diagonal of the G_w matrix.
    G_u_dso : (n_u,) diagonal of the R (input regularisation) matrix.
              Typically zeros for DSO.
    actuator_counts : Dict with n_der, n_oltc, n_shunt.
    tso_period_s, dso_period_s : For cascade margin computation.
    """
    n_y, n_u = H_dso.shape
    if G_u_dso is None:
        G_u_dso = np.zeros(n_u)

    # Partition into continuous / discrete
    if actuator_counts is not None:
        cont_idx, disc_idx = _partition_dso_indices(actuator_counts)
    else:
        cont_idx = list(range(n_u))
        disc_idx = []

    n_cont = len(cont_idx)
    n_disc = len(disc_idx)

    if n_cont == 0:
        return DSOStabilityResult(
            dso_id=dso_id, n_continuous=0, n_discrete=n_disc,
            Phi_c_eigenvalues=np.array([]),
            M_cont_spectral_radius=0.0, stable=True,
            N_inner=tso_period_s / max(dso_period_s, 1e-6),
            cascade_decay=0.0, cascade_margin=1.0,
        )

    # Extract continuous sub-matrices
    K_c = H_dso[:, cont_idx]           # (n_y, n_cont)
    G_w_c = G_w_dso[cont_idx]          # (n_cont,)
    R_c = np.diag(G_u_dso[cont_idx])   # (n_cont, n_cont)

    # Phi_c = 2*R_c + 2*(K_c)^T Q K_c
    Q_sqrt = np.sqrt(np.maximum(Q_dso, 0.0))
    QK_c = Q_sqrt[:, None] * K_c       # (n_y, n_cont)
    Phi_c = 2.0 * R_c + 2.0 * (QK_c.T @ QK_c)  # (n_cont, n_cont)

    # M_cont = I - (G_c)^{-1} Phi_c
    # Via similarity: M = G_c^{-1/2} Phi_c G_c^{-1/2}
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(G_w_c, 1e-12))
    M_c = (gw_inv_sqrt[:, None] * Phi_c) * gw_inv_sqrt[None, :]

    eigs_all = np.linalg.eigvalsh(M_c)
    eigs_active, n_filt = _filter_active_eigenvalues(eigs_all)
    rho = _spectral_radius_iteration(eigs_active)
    stable = rho < 1.0

    # Cascade margin
    N_inner = tso_period_s / max(dso_period_s, 1e-6)
    cascade_decay = rho ** N_inner if rho < 1.0 else float('inf')
    cascade_margin = 1.0 - cascade_decay

    # Per-actuator Gershgorin margins
    cont_names = (_continuous_actuator_names(actuator_counts)
                  if actuator_counts else [f'u_{i}' for i in range(n_cont)])
    # Truncate/pad to match actual number of continuous variables
    while len(cont_names) < n_cont:
        cont_names.append(f'u_{len(cont_names)}')
    cont_names = cont_names[:n_cont]

    margins = {}
    gw_min = {}
    for k in range(n_cont):
        phi_kk = float(Phi_c[k, k])
        threshold = phi_kk / 2.0
        margins[cont_names[k]] = float(G_w_c[k]) - threshold
        gw_min[cont_names[k]] = threshold

    warnings: List[str] = []
    if not stable:
        warnings.append(f"{dso_id}: rho(M_cont) = {rho:.4f} >= 1 -- UNSTABLE")
    if cascade_margin < 0.1 and stable:
        warnings.append(
            f"{dso_id}: cascade margin = {cascade_margin:.4f} is thin "
            f"(rho^{N_inner:.0f} = {cascade_decay:.4f})"
        )

    return DSOStabilityResult(
        dso_id=dso_id,
        n_continuous=n_cont,
        n_discrete=n_disc,
        Phi_c_eigenvalues=eigs_active,
        M_cont_spectral_radius=rho,
        stable=stable,
        N_inner=N_inner,
        cascade_decay=cascade_decay,
        cascade_margin=cascade_margin,
        per_actuator_margins=margins,
        per_actuator_gw_min=gw_min,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  C2: Multi-Zone Continuous Stability (Theorem 3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_continuous_stability(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    G_w_list: List[NDArray[np.float64]],
    G_u_list: Optional[List[NDArray[np.float64]]] = None,
    *,
    zone_ids: Optional[List[int]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
) -> ContinuousStabilityResult:
    """Multi-zone continuous stability (Theorem 3.1).

    Uses only continuous columns of H_ij.  Discrete columns are excluded.

    Parameters
    ----------
    H_blocks : Sensitivity blocks {(i,j): H_ij}.
    Q_obj_list : Per-zone Q_obj diagonal.
    G_w_list : Per-zone G_w diagonal (full — will be partitioned).
    G_u_list : Per-zone G_u diagonal (input regularisation).
    actuator_counts : Per-zone dict with n_der, n_pcc, n_gen, n_oltc, n_shunt.
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))
    if G_u_list is None:
        G_u_list = [np.zeros_like(gw) for gw in G_w_list]

    # Extract continuous sub-indices per zone
    cont_indices: List[List[int]] = []
    for k in range(n_zones):
        if actuator_counts is not None:
            ci, _ = _partition_indices(actuator_counts[k])
        else:
            ci = list(range(len(G_w_list[k])))
        cont_indices.append(ci)

    # Build continuous G_w and G_u per zone
    G_w_c = [G_w_list[k][cont_indices[k]] for k in range(n_zones)]
    G_u_c = [G_u_list[k][cont_indices[k]] for k in range(n_zones)]

    # Build M blocks using only continuous columns
    M_blocks_c: Dict[Tuple[int, int], NDArray[np.float64]] = {}
    n_c_per_zone = [len(ci) for ci in cont_indices]

    for i_idx, i in enumerate(zone_ids):
        q_obj_i = Q_obj_list[i_idx]
        gw_c_i = G_w_c[i_idx]
        gu_c_i = G_u_c[i_idx]
        n_c_i = n_c_per_zone[i_idx]

        if n_c_i == 0:
            for j_idx, j in enumerate(zone_ids):
                M_blocks_c[(i, j)] = np.zeros((0, n_c_per_zone[j_idx]))
            continue

        gw_inv_sqrt_i = 1.0 / np.sqrt(np.maximum(gw_c_i, 1e-12))
        Q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))

        H_ii = H_blocks.get((i, i))
        if H_ii is None:
            raise ValueError(f"Diagonal block H_({i},{i}) required but missing.")

        # Extract continuous columns of H_ii
        K_ii_c = H_ii[:, cont_indices[i_idx]]
        QK_ii_c = Q_sqrt_i[:, None] * K_ii_c

        for j_idx, j in enumerate(zone_ids):
            n_c_j = n_c_per_zone[j_idx]
            if n_c_j == 0:
                M_blocks_c[(i, j)] = np.zeros((n_c_i, 0))
                continue

            gw_inv_sqrt_j = 1.0 / np.sqrt(np.maximum(G_w_c[j_idx], 1e-12))

            H_ij = H_blocks.get((i, j))
            if H_ij is None:
                M_blocks_c[(i, j)] = np.zeros((n_c_i, n_c_j))
                continue

            K_ij_c = H_ij[:, cont_indices[j_idx]]
            QK_ij_c = Q_sqrt_i[:, None] * K_ij_c

            # C_ij^c = 2*(K_ii^c)^T Q_i K_ij^c  (+ 2*R_i^c for diagonal)
            C_ij_c = 2.0 * (QK_ii_c.T @ QK_ij_c)
            if i == j:
                C_ij_c += 2.0 * np.diag(gu_c_i)

            # M_ij^c = G_w,i^{-1/2} C_ij^c G_w,j^{-1/2}
            M_ij_c = (gw_inv_sqrt_i[:, None] * C_ij_c) * gw_inv_sqrt_j[None, :]
            M_blocks_c[(i, j)] = M_ij_c

    # Assemble full M_full^c
    n_total_c = sum(n_c_per_zone)
    M_full_c = np.zeros((n_total_c, n_total_c))
    row_off = 0
    for i_idx, i in enumerate(zone_ids):
        col_off = 0
        for j_idx, j in enumerate(zone_ids):
            M_ij = M_blocks_c[(i, j)]
            r0, r1 = row_off, row_off + n_c_per_zone[i_idx]
            c0, c1 = col_off, col_off + n_c_per_zone[j_idx]
            ar = min(M_ij.shape[0], r1 - r0)
            ac = min(M_ij.shape[1], c1 - c0)
            M_full_c[r0:r0+ar, c0:c0+ac] = M_ij[:ar, :ac]
            col_off += n_c_per_zone[j_idx]
        row_off += n_c_per_zone[i_idx]

    # Global eigenvalues (M_full^c may be non-symmetric due to cross terms)
    if n_total_c > 0:
        eigs_all = np.linalg.eigvals(M_full_c)
        eigs_active, n_filt = _filter_active_eigenvalues(eigs_all)
        rho_global = _spectral_radius_iteration(eigs_active)
    else:
        eigs_active = np.array([])
        n_filt = 0
        rho_global = 0.0

    # Per-zone local metrics
    per_zone_rho: Dict[int, float] = {}
    per_zone_lmax: Dict[int, float] = {}
    per_zone_kappa: Dict[int, float] = {}
    coupling_norms: Dict[Tuple[int, int], float] = {}

    for i_idx, i in enumerate(zone_ids):
        M_ii = M_blocks_c[(i, i)]
        if M_ii.size == 0:
            per_zone_rho[i] = 0.0
            per_zone_lmax[i] = 0.0
            per_zone_kappa[i] = 1.0
            continue

        eig_ii = np.linalg.eigvalsh(M_ii)
        eig_act, _ = _filter_active_eigenvalues(eig_ii)
        if len(eig_act) > 0:
            per_zone_rho[i] = _spectral_radius_iteration(eig_act)
            per_zone_lmax[i] = float(eig_act[-1])
            lmin = float(eig_act[0])
            per_zone_kappa[i] = per_zone_lmax[i] / lmin if lmin > 1e-14 else np.inf
        else:
            per_zone_rho[i] = 0.0
            per_zone_lmax[i] = 0.0
            per_zone_kappa[i] = 1.0

        for j_idx, j in enumerate(zone_ids):
            if j == i:
                continue
            M_ij = M_blocks_c.get((i, j), np.zeros((0, 0)))
            if M_ij.size > 0:
                coupling_norms[(i, j)] = float(np.linalg.norm(M_ij, ord=2))
            else:
                coupling_norms[(i, j)] = 0.0

    # Small-gain: gamma = max_i {rho_i + sum_{j!=i} ||M_ij^c||_2}
    gamma = 0.0
    for i_idx, i in enumerate(zone_ids):
        row_sum = per_zone_rho[i]
        for j in zone_ids:
            if j != i:
                row_sum += coupling_norms.get((i, j), 0.0)
        gamma = max(gamma, row_sum)

    warnings: List[str] = []
    if not (rho_global < 1.0):
        warnings.append(
            f"C2 VIOLATED: rho(M_full^c) = {rho_global:.4f} >= 1"
        )

    return ContinuousStabilityResult(
        M_full_c_eigenvalues=eigs_active,
        spectral_radius=rho_global,
        stable=rho_global < 1.0,
        per_zone_rho=per_zone_rho,
        per_zone_lambda_max=per_zone_lmax,
        per_zone_kappa=per_zone_kappa,
        coupling_norms=coupling_norms,
        small_gain_gamma=gamma,
        small_gain_stable=gamma < 1.0,
        n_null_filtered=n_filt,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  C3: Multi-Zone Discrete Small-Gain (Theorem 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_discrete_small_gain(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    G_w_list: List[NDArray[np.float64]],
    *,
    zone_ids: Optional[List[int]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
) -> DiscreteSmallGainResult:
    """Multi-zone discrete small-gain analysis (Theorem 3.2).

    Builds the Gamma matrix and checks rho(Gamma) < 1.
    Reports per-actuator G sizing rule (Corollary 3.2).

    Parameters
    ----------
    H_blocks : Sensitivity blocks {(i,j): H_ij}.
    Q_obj_list : Per-zone Q_obj diagonal.
    G_w_list : Per-zone G_w diagonal (full — discrete entries extracted).
    actuator_counts : Per-zone dict with n_der, n_pcc, n_gen, n_oltc, n_shunt.
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))

    # Extract discrete sub-indices per zone
    disc_indices: List[List[int]] = []
    for k in range(n_zones):
        if actuator_counts is not None:
            _, di = _partition_indices(actuator_counts[k])
        else:
            di = []
        disc_indices.append(di)

    n_disc_per_zone = [len(di) for di in disc_indices]

    # Check: any discrete actuators?
    if sum(n_disc_per_zone) == 0:
        return DiscreteSmallGainResult(
            Gamma=np.zeros((n_zones, n_zones)),
            Gamma_spectral_radius=0.0,
            stable=True,
            Gamma_row_sums=np.zeros(n_zones),
        )

    # Build P_ij = (K_ii^d)^T Q_i K_ij^d for each (i, j)
    P_blocks: Dict[Tuple[int, int], NDArray[np.float64]] = {}

    for i_idx, i in enumerate(zone_ids):
        n_d_i = n_disc_per_zone[i_idx]
        if n_d_i == 0:
            continue

        q_obj_i = Q_obj_list[i_idx]
        Q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))

        H_ii = H_blocks.get((i, i))
        if H_ii is None:
            continue

        # K_ii^d: discrete columns of H_ii
        K_ii_d = H_ii[:, disc_indices[i_idx]]
        QK_ii_d = Q_sqrt_i[:, None] * K_ii_d  # Q^{1/2} K_ii^d

        for j_idx, j in enumerate(zone_ids):
            n_d_j = n_disc_per_zone[j_idx]
            if n_d_j == 0:
                P_blocks[(i, j)] = np.zeros((n_d_i, 0))
                continue

            H_ij = H_blocks.get((i, j))
            if H_ij is None:
                P_blocks[(i, j)] = np.zeros((n_d_i, n_d_j))
                continue

            K_ij_d = H_ij[:, disc_indices[j_idx]]
            QK_ij_d = Q_sqrt_i[:, None] * K_ij_d

            # P_ij = (K_ii^d)^T Q_i K_ij^d  (via Q^{1/2} factorisation)
            P_ij = QK_ii_d.T @ QK_ij_d  # (n_d_i, n_d_j)
            P_blocks[(i, j)] = P_ij

    # Build Gamma matrix (N x N)
    # Gamma_ij = max_a (4/g_{i,a}) sum_b |[P_ij]_{ab}|   for i != j
    # Gamma_ii = 0
    Gamma = np.zeros((n_zones, n_zones))

    # Also compute G sizing rule
    g_min_required: Dict[int, Dict[str, float]] = {}
    g_current_dict: Dict[int, Dict[str, float]] = {}
    g_margin_dict: Dict[int, Dict[str, float]] = {}

    for i_idx, i in enumerate(zone_ids):
        n_d_i = n_disc_per_zone[i_idx]
        if n_d_i == 0:
            g_min_required[i] = {}
            g_current_dict[i] = {}
            g_margin_dict[i] = {}
            continue

        gw_d_i = G_w_list[i_idx][disc_indices[i_idx]]
        disc_names = (_discrete_actuator_names(actuator_counts[i_idx])
                      if actuator_counts else [f'd_{k}' for k in range(n_d_i)])
        while len(disc_names) < n_d_i:
            disc_names.append(f'd_{len(disc_names)}')
        disc_names = disc_names[:n_d_i]

        g_min_required[i] = {}
        g_current_dict[i] = {}
        g_margin_dict[i] = {}

        for j_idx, j in enumerate(zone_ids):
            if j == i:
                continue

            P_ij = P_blocks.get((i, j))
            if P_ij is None or P_ij.size == 0:
                continue

            # Gamma_ij = max_a (4/g_{i,a}) sum_b |P_ij[a,b]|
            row_l1 = np.sum(np.abs(P_ij), axis=1)  # (n_d_i,)
            ratios = np.zeros(n_d_i)
            for a in range(n_d_i):
                g_ia = max(float(gw_d_i[a]), 1e-12)
                ratios[a] = (4.0 / g_ia) * row_l1[a]
            Gamma[i_idx, j_idx] = float(np.max(ratios)) if n_d_i > 0 else 0.0

            # G sizing rule: accumulate per-actuator minimum
            for a in range(n_d_i):
                name = disc_names[a]
                g_min_required[i][name] = (
                    g_min_required[i].get(name, 0.0) + 4.0 * float(row_l1[a])
                )

        # Fill current values and margins
        for a in range(n_d_i):
            name = disc_names[a]
            g_current_dict[i][name] = float(gw_d_i[a])
            g_min_val = g_min_required[i].get(name, 0.0)
            g_margin_dict[i][name] = float(gw_d_i[a]) - g_min_val

    # Spectral radius of Gamma
    if n_zones > 0:
        gamma_eigs = np.linalg.eigvals(Gamma)
        gamma_rho = float(np.max(np.abs(gamma_eigs)))
    else:
        gamma_rho = 0.0

    row_sums = np.sum(Gamma, axis=1)

    warnings: List[str] = []
    if gamma_rho >= 1.0:
        warnings.append(f"C3 VIOLATED: rho(Gamma) = {gamma_rho:.4f} >= 1")
    for i_idx, i in enumerate(zone_ids):
        for name, margin in g_margin_dict.get(i, {}).items():
            if margin < 0:
                g_min = g_min_required[i].get(name, 0.0)
                g_cur = g_current_dict[i].get(name, 0.0)
                warnings.append(
                    f"Zone {i} {name}: g_w = {g_cur:.2f} < g_min = {g_min:.2f} "
                    f"(deficit = {-margin:.2f})"
                )

    return DiscreteSmallGainResult(
        Gamma=Gamma,
        Gamma_spectral_radius=gamma_rho,
        stable=gamma_rho < 1.0,
        P_blocks=P_blocks,
        g_min_required=g_min_required,
        g_current=g_current_dict,
        g_margin=g_margin_dict,
        Gamma_row_sums=row_sums,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Combined: Theorem 3.3 Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_multi_zone_stability(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    G_w_list: List[NDArray[np.float64]],
    *,
    zone_ids: Optional[List[int]] = None,
    zone_names: Optional[List[str]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
    G_u_list: Optional[List[NDArray[np.float64]]] = None,
    # DSO data (for C1)
    dso_data: Optional[List[Dict]] = None,
    tso_period_s: float = 180.0,
    dso_period_s: float = 20.0,
    # Display
    verbose: bool = True,
    # Legacy compatibility (unused, kept for call-site compat)
    alpha: float = 1.0,
    configured_cooldown: Optional[int] = None,
    int_max_step: int = 1,
    dwell_time_epsilon: float = 0.01,
) -> MultiZoneStabilityResult:
    """Three-condition stability analysis (Theorem 3.3).

    Parameters
    ----------
    H_blocks : {(i,j): H_ij} sensitivity blocks.
    Q_obj_list : Per-zone Q_obj diagonal vectors.
    G_w_list : Per-zone G_w diagonal vectors.
    zone_ids : Zone integer IDs.
    zone_names : Human-readable zone labels.
    actuator_counts : Per-zone dict with n_der, n_pcc, n_gen, n_oltc, n_shunt.
    G_u_list : Per-zone input regularisation (R diagonal).
    dso_data : List of dicts, each with keys:
        'H': H_dso, 'Q': Q_dso, 'G_w': G_w_dso, 'G_u': G_u_dso,
        'id': str, 'actuator_counts': dict, 'zone_id': int.
    tso_period_s, dso_period_s : For cascade margin.
    verbose : Print formatted report.
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))
    if zone_names is None:
        zone_names = [f"Zone {z}" for z in zone_ids]
    if actuator_counts is None:
        actuator_counts = [{'n_der': len(gw)} for gw in G_w_list]

    # ── C1: DSO inner-loop stability ──────────────────────────────────────
    c1_results: List[DSOStabilityResult] = []
    if dso_data is not None:
        for dd in dso_data:
            r = analyse_dso_stability(
                H_dso=dd['H'],
                Q_dso=dd['Q'],
                G_w_dso=dd['G_w'],
                G_u_dso=dd.get('G_u'),
                dso_id=dd.get('id', 'DSO'),
                actuator_counts=dd.get('actuator_counts'),
                tso_period_s=tso_period_s,
                dso_period_s=dso_period_s,
            )
            c1_results.append(r)

    c1_ok = all(r.stable for r in c1_results) if c1_results else True

    # ── C2: Multi-zone TSO continuous stability ───────────────────────────
    c2_result = analyse_continuous_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=G_w_list,
        G_u_list=G_u_list,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
    )
    c2_ok = c2_result.stable

    # ── C3: Multi-zone TSO discrete small-gain ────────────────────────────
    c3_result = analyse_discrete_small_gain(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=G_w_list,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
    )
    c3_ok = c3_result.stable

    # ── Per-zone summaries ────────────────────────────────────────────────
    zone_results: List[ZoneStabilityResult] = []
    for i_idx, i in enumerate(zone_ids):
        ac = actuator_counts[i_idx]
        cont_idx, disc_idx = _partition_indices(ac)
        n_c = len(cont_idx)
        n_d = len(disc_idx)

        H_ii = H_blocks.get((i, i))
        n_out = H_ii.shape[0] if H_ii is not None else 0

        coupling_sum_c = sum(
            c2_result.coupling_norms.get((i, j), 0.0)
            for j in zone_ids if j != i
        )

        gamma_row = float(c3_result.Gamma_row_sums[i_idx]) if i_idx < len(c3_result.Gamma_row_sums) else 0.0

        n_viol = sum(1 for m in c3_result.g_margin.get(i, {}).values() if m < 0)

        dso_for_zone = [r for r in c1_results if hasattr(r, '_zone_id') and r._zone_id == i]

        zone_results.append(ZoneStabilityResult(
            zone_id=i,
            zone_name=zone_names[i_idx],
            n_continuous=n_c,
            n_discrete=n_d,
            n_outputs=n_out,
            lambda_max_c=c2_result.per_zone_lambda_max.get(i, 0.0),
            rho_c=c2_result.per_zone_rho.get(i, 0.0),
            coupling_sum_c=coupling_sum_c,
            Gamma_row_sum=gamma_row,
            n_discrete_violations=n_viol,
            dso_results=dso_for_zone,
        ))

    # ── Build M_full_c for external access ────────────────────────────────
    # Reconstruct from c2_result internals
    cont_indices_all = []
    for k in range(n_zones):
        ci, _ = _partition_indices(actuator_counts[k])
        cont_indices_all.append(ci)
    n_c_per = [len(ci) for ci in cont_indices_all]
    n_total_c = sum(n_c_per)
    M_full_c = np.zeros((n_total_c, n_total_c))
    # Re-assemble (already computed in c2; we do it once more for the result)
    row_off = 0
    for i_idx, i in enumerate(zone_ids):
        col_off = 0
        for j_idx, j in enumerate(zone_ids):
            q_obj_i = Q_obj_list[i_idx]
            Q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))
            gw_c_i = G_w_list[i_idx][cont_indices_all[i_idx]]
            gw_c_j = G_w_list[j_idx][cont_indices_all[j_idx]]
            gw_inv_sqrt_i = 1.0 / np.sqrt(np.maximum(gw_c_i, 1e-12))
            gw_inv_sqrt_j = 1.0 / np.sqrt(np.maximum(gw_c_j, 1e-12))

            H_ii = H_blocks.get((i, i))
            H_ij = H_blocks.get((i, j))
            if H_ii is not None and H_ij is not None and len(cont_indices_all[i_idx]) > 0 and len(cont_indices_all[j_idx]) > 0:
                K_ii_c = H_ii[:, cont_indices_all[i_idx]]
                K_ij_c = H_ij[:, cont_indices_all[j_idx]]
                QK_ii_c = Q_sqrt_i[:, None] * K_ii_c
                QK_ij_c = Q_sqrt_i[:, None] * K_ij_c
                C_ij_c = 2.0 * (QK_ii_c.T @ QK_ij_c)
                if i == j and G_u_list is not None:
                    gu_c_i = G_u_list[i_idx][cont_indices_all[i_idx]]
                    C_ij_c += 2.0 * np.diag(gu_c_i)
                M_ij_c = (gw_inv_sqrt_i[:, None] * C_ij_c) * gw_inv_sqrt_j[None, :]
                nr = n_c_per[i_idx]
                nc = n_c_per[j_idx]
                M_full_c[row_off:row_off+nr, col_off:col_off+nc] = M_ij_c
            col_off += n_c_per[j_idx]
        row_off += n_c_per[i_idx]

    # ── Combined verdict ──────────────────────────────────────────────────
    all_stable = c1_ok and c2_ok and c3_ok
    tag = "STABLE" if all_stable else "UNSTABLE"

    c1_str = f"C1(DSO): {'pass' if c1_ok else 'FAIL'}"
    if c1_results:
        worst_dso = max(c1_results, key=lambda r: r.M_cont_spectral_radius)
        c1_str += f" (worst rho = {worst_dso.M_cont_spectral_radius:.4f})"
    else:
        c1_str += " (no DSO data)"

    c2_str = f"C2(cont): {'pass' if c2_ok else 'FAIL'} (rho = {c2_result.spectral_radius:.4f})"
    c3_str = f"C3(disc): {'pass' if c3_ok else 'FAIL'} (rho(Gamma) = {c3_result.Gamma_spectral_radius:.4f})"

    summary = f"Theorem 3.3: {tag}.  {c1_str}.  {c2_str}.  {c3_str}."

    # ── Recommendations ───────────────────────────────────────────────────
    recommendations: List[str] = []
    if not c2_ok:
        worst_zone_id = max(c2_result.per_zone_rho, key=c2_result.per_zone_rho.get)
        worst_idx = zone_ids.index(worst_zone_id)
        recommendations.append(
            f"C2: Increase continuous g_w in {zone_names[worst_idx]} "
            f"(rho_c = {c2_result.per_zone_rho[worst_zone_id]:.4f}, "
            f"kappa = {c2_result.per_zone_kappa.get(worst_zone_id, np.inf):.1f})"
        )
    if not c3_ok:
        for i_idx, i in enumerate(zone_ids):
            for name, margin in c3_result.g_margin.get(i, {}).items():
                if margin < 0:
                    g_min = c3_result.g_min_required[i][name]
                    recommendations.append(
                        f"C3: {zone_names[i_idx]} {name}: "
                        f"increase g_w to >= {g_min:.2f} (currently {c3_result.g_current[i][name]:.2f})"
                    )
    if not c1_ok:
        for r in c1_results:
            if not r.stable:
                recommendations.append(
                    f"C1: {r.dso_id} unstable (rho = {r.M_cont_spectral_radius:.4f}). "
                    f"Increase DSO g_w."
                )
                for name, margin in r.per_actuator_margins.items():
                    if margin < 0:
                        recommendations.append(
                            f"  {name}: g_w deficit = {-margin:.4f}"
                        )

    result = MultiZoneStabilityResult(
        c1_dso=c1_results,
        c2_continuous=c2_result,
        c3_discrete=c3_result,
        zones=zone_results,
        c1_satisfied=c1_ok,
        c2_satisfied=c2_ok,
        c3_satisfied=c3_ok,
        stable=all_stable,
        summary=summary,
        M_full_c=M_full_c,
        Gamma=c3_result.Gamma,
        recommendations=recommendations,
    )

    if verbose:
        _print_report(result, zone_names)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Report Printing
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(result: MultiZoneStabilityResult, zone_names: List[str]) -> None:
    """Print compact three-condition stability report."""
    W = 80
    sep = "=" * W
    thin = "-" * W

    print()
    print(sep)
    print("  STABILITY ANALYSIS (Theorem 3.3)")
    print(sep)

    # ── C1: DSO inner loops ───────────────────────────────────────────────
    if result.c1_dso:
        c1_tag = "pass" if result.c1_satisfied else "FAIL"
        print(f"\n  C1  DSO inner-loop stability                        [{c1_tag}]")
        print(f"  {thin[2:]}")
        header = f"  {'DSO':<16s} {'rho(M_cont)':>11s} {'cascade':>9s} {'N_inner':>8s} {'status':>8s}"
        print(header)
        for r in result.c1_dso:
            st = "pass" if r.stable else "FAIL"
            cm = f"{r.cascade_margin:.3f}" if r.cascade_margin < 100 else "ok"
            print(f"  {r.dso_id:<16s} {r.M_cont_spectral_radius:>11.4f} "
                  f"{cm:>9s} {r.N_inner:>8.0f} {st:>8s}")
    else:
        print(f"\n  C1  DSO inner-loop stability                        [no data]")

    # ── C2: Multi-zone continuous ─────────────────────────────────────────
    c2 = result.c2_continuous
    c2_tag = "pass" if result.c2_satisfied else "FAIL"
    print(f"\n  C2  Multi-zone continuous stability                 [{c2_tag}]")
    print(f"  {thin[2:]}")
    header = f"  {'Zone':<10s} {'lam_max':>8s} {'kappa':>8s} {'rho_c':>7s} {'coupling':>9s}"
    print(header)
    for zr in result.zones:
        kappa_str = ("inf" if c2.per_zone_kappa.get(zr.zone_id, np.inf) >= 1e6
                     else f"{c2.per_zone_kappa.get(zr.zone_id, 1.0):>8.1f}")
        print(f"  {zr.zone_name:<10s} "
              f"{zr.lambda_max_c:>8.3f} "
              f"{kappa_str:>8s} "
              f"{zr.rho_c:>7.4f} "
              f"{zr.coupling_sum_c:>9.3f}")
    print(f"  {thin[2:]}")
    print(f"  Global rho(M_full^c) = {c2.spectral_radius:.4f}")
    print(f"  Small-gain gamma     = {c2.small_gain_gamma:.4f}  "
          f"[{'pass' if c2.small_gain_stable else 'FAIL'}]")

    # ── C3: Discrete small-gain ───────────────────────────────────────────
    c3 = result.c3_discrete
    c3_tag = "pass" if result.c3_satisfied else "FAIL"
    n_disc_total = sum(zr.n_discrete for zr in result.zones)
    print(f"\n  C3  Discrete small-gain (N_disc = {n_disc_total})             [{c3_tag}]")
    print(f"  {thin[2:]}")

    if n_disc_total > 0:
        n_zones = len(result.zones)
        # Print Gamma matrix
        print(f"  Gamma matrix ({n_zones}x{n_zones}):")
        for i in range(n_zones):
            row_str = "    " + "  ".join(f"{c3.Gamma[i, j]:7.4f}" for j in range(n_zones))
            print(row_str)
        print(f"  rho(Gamma) = {c3.Gamma_spectral_radius:.4f}")

        # G sizing rule violations
        n_violations = sum(zr.n_discrete_violations for zr in result.zones)
        if n_violations > 0:
            print(f"\n  G sizing rule violations ({n_violations}):")
            for zr in result.zones:
                for name, margin in c3.g_margin.get(zr.zone_id, {}).items():
                    if margin < 0:
                        g_min = c3.g_min_required[zr.zone_id][name]
                        g_cur = c3.g_current[zr.zone_id][name]
                        print(f"    {zr.zone_name} {name}: g_w = {g_cur:.1f} "
                              f"< g_min = {g_min:.1f} (need +{-margin:.1f})")
        else:
            print(f"  All discrete g_w satisfy sizing rule.")
    else:
        print(f"  No discrete actuators.")

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{sep}")
    tag = "STABLE" if result.stable else "UNSTABLE"
    print(f"  Theorem 3.3 verdict: {tag}")
    c1_sym = "ok" if result.c1_satisfied else "X "
    c2_sym = "ok" if result.c2_satisfied else "X "
    c3_sym = "ok" if result.c3_satisfied else "X "
    print(f"    [{c1_sym}] C1: DSO inner loops")
    print(f"    [{c2_sym}] C2: TSO continuous (rho = {c2.spectral_radius:.4f})")
    print(f"    [{c3_sym}] C3: TSO discrete   (rho(Gamma) = {c3.Gamma_spectral_radius:.4f})")

    # Recommendations
    if result.recommendations:
        print(f"\n  Recommendations:")
        for rec in result.recommendations:
            print(f"    {rec}")

    print(sep)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy compatibility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_gw_min(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    alpha: float = 1.0,
) -> float:
    """Conservative scalar g_w lower bound (Gershgorin diagonal)."""
    Q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    QH = Q_sqrt[:, None] * H
    C_diag = np.sum(QH ** 2, axis=0)
    return float(alpha * np.max(C_diag) / 2.0)
