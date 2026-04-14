"""
Auto-Tuning for the Three-Condition Stability Framework
========================================================

Tunes g_w (actuator weights) and alpha (step size) to satisfy all three
conditions of Theorem 3.3:

    C1: DSO inner loops  --  Q-only curvature sizing, Gershgorin fallback
    C2: TSO continuous   --  Gershgorin floor + eigenvalue init + pump
    C3: TSO discrete     --  G sizing rule (Corollary 3.2, closed-form)

All tuning parameters are centralised in the ``TuningConfig`` dataclass.

Functions
---------
auto_tune
    Main entry point.  Orchestrates DSO (C1) -> continuous TSO (C2) ->
    discrete TSO (C3), then verifies all conditions.

tune_dso_gw
    C1: per-DSO g_w + alpha.  Sizes alpha from Q-tracking curvature
    (not full objective) to decouple cascade speed from voltage weight.

tune_continuous_gw
    C2: multi-zone continuous g_w + alpha.  Filters inert actuators,
    applies eigenvalue-based init, conditioning pump, then alpha search.

tune_discrete_gw
    C3: discrete g_w via Corollary 3.2 (closed-form lower bound).

recommend_dso_weights
    Diagnostic: recommended g_q / g_v ratio from sensitivity norms.

Author: Manuel Schwenke, TU Darmstadt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from analysis.stability_analysis import (
    analyse_dso_stability,
    analyse_multi_zone_stability,
)

_log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_EIG_ACTIVE_FRAC = 0.01
"""Eigenvalues below this fraction of lambda_max are treated as null-space."""

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
#  Tuning Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TuningConfig:
    """Structured configuration for auto-tuning.

    Separates DSO (C1) and TSO (C2/C3) parameters so each layer can be
    tuned independently.
    """

    # ── DSO (C1) ──────────────────────────────────────────────────────────
    dso_kappa_target: float = 10.0
    """Condition-number target for DSO pump (rho_opt = (k-1)/(k+1))."""

    dso_rho_target: float = 0.85
    """Target spectral radius for DSO alpha search."""

    dso_alpha_target: float = 1.0
    """Expected alpha for DSO — used for warnings only."""

    dso_safety_factor_continuous: float = 1.5
    """Gershgorin safety factor for DSO continuous actuators."""

    dso_safety_factor_discrete: float = 2.0
    """Safety factor for DSO discrete actuators."""

    dso_alpha_min: float = 0.1
    """If alpha-first gives alpha below this, fall back to Gershgorin."""

    # ── TSO (C2) ──────────────────────────────────────────────────────────
    tso_kappa_target: float = 50.0
    """Condition-number target for TSO pump."""

    tso_rho_target: float = 0.95
    """Target spectral radius for TSO alpha search."""

    tso_alpha_target: float = 0.5
    """Expected alpha for TSO — used for pump lambda_max guard and warnings."""

    tso_safety_factor_continuous: float = 2.0
    """Gershgorin safety factor for TSO continuous actuators."""

    tso_safety_factor_discrete: float = 3.0
    """Safety factor for TSO discrete actuators."""

    tso_gersh_floor_fraction: float = 1.0
    """Fraction of Gershgorin bound used as floor for TSO g_w."""

    # ── C3 discrete ───────────────────────────────────────────────────────
    c3_safety_factor: float = 1.5
    """Safety factor for discrete G sizing rule (Corollary 3.2)."""

    # ── Pump control ──────────────────────────────────────────────────────
    lambda_target: float = 1.5
    """Eigenvalue-based init target: desired lambda_max(M) after init."""

    lambda_max_alpha_target: float = 1.8
    """Pump guard: alpha * lambda_max must stay below this."""

    pump_max_iters: int = 20
    """Maximum conditioning-pump iterations."""

    eigenvalue_init_max_boost: float = 10.0
    """Max per-actuator boost in eigenvalue-based g_w init."""

    sensitivity_filter_frac: float = 0.01
    """Actuators with Q-weighted column norm below this fraction of max
    are excluded from eigenanalysis (prevents spurious small eigenvalues)."""

    # ── Post-tuning boost ─────────────────────────────────────────────
    alpha_dso_boost: float = 1.0
    """Multiply DSO alpha after tuning (>1 = faster tracking, less margin).
    Stability is re-verified; the boost is backed off if C1 fails."""


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
    tol = _EIG_ACTIVE_FRAC * eig_max_abs
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
#  Conditioning pump (improves kappa, does NOT brute-force stability)
# ═══════════════════════════════════════════════════════════════════════════════

def _conditioning_pump(
    gw: NDArray[np.float64],
    M_builder,
    *,
    kappa_target: float = 50.0,
    alpha_target: float = 0.5,
    lambda_max_alpha_target: float = 1.8,
    max_iters: int = 20,
    boost_factor: float = 1.3,
) -> Tuple[NDArray[np.float64], float, int]:
    """Improve conditioning of M by boosting g_w for the fastest mode.

    Two exit criteria (both must be satisfied):
        1. kappa = lambda_max / lambda_min  <=  kappa_target
        2. alpha_target * lambda_max        <=  lambda_max_alpha_target

    Criterion 2 ensures absolute eigenvalue scale is controlled,
    preventing alpha*lambda_max >= 2 (instability boundary).

    Returns (gw_tuned, kappa_achieved, n_iters).
    """
    gw = gw.copy()

    for it in range(max_iters):
        M, eigs = M_builder(gw)

        # Filter active eigenvalues
        eig_max_abs = max(float(np.max(np.abs(eigs))), 1e-14)
        active_mask = np.abs(eigs) > _EIG_ACTIVE_FRAC * eig_max_abs
        eigs_active = eigs[active_mask].real if np.isrealobj(eigs) else eigs[active_mask]

        if len(eigs_active) < 2:
            return gw, 1.0, it

        lam_max = float(np.max(eigs_active.real))
        lam_min = float(np.min(eigs_active.real))
        if lam_min <= 1e-14:
            lam_min = 1e-14
        kappa = lam_max / lam_min

        # Check BOTH conditions
        kappa_ok = kappa <= kappa_target
        scale_ok = (alpha_target * lam_max) <= lambda_max_alpha_target
        if kappa_ok and scale_ok:
            return gw, kappa, it

        # Find eigenvector of LARGEST eigenvalue (the bottleneck)
        if M.shape[0] == M.shape[1]:
            asym = float(np.linalg.norm(M - M.T, 'fro'))
            if asym < 1e-10 * max(float(np.linalg.norm(M, 'fro')), 1e-14):
                all_eigs, vecs = np.linalg.eigh(M)
                v = vecs[:, -1]  # eigenvector of lambda_max
            else:
                all_eigs, vecs = np.linalg.eig(M)
                idx = int(np.argmax(all_eigs.real))
                v = np.abs(vecs[:, idx].real)
        else:
            v = np.ones(len(gw))

        # Boost g_w for actuators that participate in the fast mode.
        v_sq = v ** 2
        v_sq = v_sq / max(np.sum(v_sq), 1e-14)

        boost = 1.0 + (boost_factor - 1.0) * v_sq / max(np.max(v_sq), 1e-14)
        gw = gw * boost

    # Final kappa
    _, eigs_final = M_builder(gw)
    eig_max_abs = max(float(np.max(np.abs(eigs_final))), 1e-14)
    active = eigs_final[np.abs(eigs_final) > _EIG_ACTIVE_FRAC * eig_max_abs]
    if len(active) >= 2:
        lmax = float(np.max(active.real))
        lmin = float(np.min(active.real))
        kappa_final = lmax / max(lmin, 1e-14)
    else:
        kappa_final = 1.0
    return gw, kappa_final, max_iters


def _eigenvalue_init_gw(
    M_builder,
    gw_init: NDArray[np.float64],
    lambda_target: float = 1.5,
    max_boost: float = 10.0,
) -> NDArray[np.float64]:
    """Boost g_w for actuators participating in eigenvalues above lambda_target.

    Unlike Gershgorin (which is blind to eigenvalue accumulation from
    collinear sensitivities), this uses the actual eigenspectrum to
    selectively inflate only the actuators that drive lambda_max.

    The per-actuator boost is capped at ``max_boost`` to prevent extreme
    inflation when the starting g_w is far from the curvature scale.
    The conditioning pump handles the remainder iteratively.

    Parameters
    ----------
    M_builder : callable(gw) -> (M, eigs)
    gw_init : starting g_w vector
    lambda_target : desired lambda_max(M) after adjustment
    max_boost : maximum per-actuator multiplicative boost (default 10x)

    Returns g_w_new (never decreases any entry).
    """
    _, eigs = M_builder(gw_init)
    eigs_real = eigs.real if not np.isrealobj(eigs) else eigs
    lam_max = float(np.max(eigs_real))

    if lam_max <= lambda_target or lam_max <= 1e-14:
        return gw_init.copy()

    # Need eigenvectors to compute participation
    M, _ = M_builder(gw_init)
    asym = float(np.linalg.norm(M - M.T, 'fro'))
    if asym < 1e-10 * max(float(np.linalg.norm(M, 'fro')), 1e-14):
        all_eigs, vecs = np.linalg.eigh(M)
        v = vecs[:, -1]
    else:
        all_eigs, vecs = np.linalg.eig(M)
        idx = int(np.argmax(all_eigs.real))
        v = np.abs(vecs[:, idx].real)

    participation = v ** 2
    p_max = max(np.max(participation), 1e-14)

    overshoot = lam_max / lambda_target
    boost = 1.0 + (overshoot - 1.0) * participation / p_max
    boost = np.minimum(boost, max_boost)  # cap per-actuator boost
    return gw_init * boost


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
    alpha_min: float = 0.1,
    n_interfaces: int = 0,
) -> Tuple[NDArray[np.float64], float]:
    """Tune DSO to satisfy C1: rho(I - alpha*M_cont) < 1.

    When ``n_interfaces > 0``, alpha is sized for **Q-tracking curvature
    only** (the cascade-critical objective), then verified against the
    full objective.  This prevents the voltage weight g_v from dominating
    the Hessian and forcing a tiny alpha.

    Falls back to Gershgorin preconditioning when alpha < ``alpha_min``.

    Returns (g_w_tuned_full, alpha_dso).
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
    Q_sqrt = np.sqrt(np.maximum(Q_dso, 0.0))
    gw_full = G_w_dso_init.copy()

    if n_cont == 0:
        return gw_full, 1.0

    K_c = H_dso[:, cont_idx]

    # Full Q-weighted sensitivity (Q + V + I rows)
    QK_c_full = Q_sqrt[:, None] * K_c

    # Q-tracking-only sensitivity (first n_interfaces rows)
    # This is the cascade-critical part; voltage is secondary.
    if n_interfaces > 0:
        Q_sqrt_q = Q_sqrt.copy()
        Q_sqrt_q[n_interfaces:] = 0.0  # zero out V and I rows
        QK_c_q = Q_sqrt_q[:, None] * K_c
    else:
        QK_c_q = QK_c_full

    def _build_M_cont(gw_c, qk=None):
        if qk is None:
            qk = QK_c_full
        gi = 1.0 / np.sqrt(np.maximum(gw_c, 1e-12))
        Phi_c = qk.T @ qk
        M_c = (gi[:, None] * Phi_c) * gi[None, :]
        eigs = np.linalg.eigvalsh(M_c)
        return M_c, eigs

    # --- Step 1: Compute alpha from Q-tracking curvature ---
    # Size alpha for the cascade-critical Q rows, not the full objective.
    _, eigs_q = _build_M_cont(gw_full[cont_idx], QK_c_q)
    eig_max = max(float(np.max(np.abs(eigs_q))), 1e-14)
    active = eigs_q[eigs_q > _EIG_ACTIVE_FRAC * eig_max]
    alpha = _find_alpha_for_target_rho(
        active.astype(np.complex128), rho_target=rho_target,
    )

    # --- Step 2: If alpha too small, Gershgorin on Q-tracking curvature ---
    c_diag_q = np.sum(QK_c_q ** 2, axis=0)

    if alpha < alpha_min:
        gw_gersh = safety_factor_continuous * c_diag_q / 2.0
        gw_gersh = np.maximum(gw_gersh, 1e-6)  # avoid zero for V-only actuators
        gw_cont = np.maximum(gw_full[cont_idx], gw_gersh)
        gw_full[cont_idx] = gw_cont

        _, eigs_gersh = _build_M_cont(gw_full[cont_idx], QK_c_q)
        eig_max = max(float(np.max(np.abs(eigs_gersh))), 1e-14)
        active = eigs_gersh[eigs_gersh > _EIG_ACTIVE_FRAC * eig_max]
        alpha = _find_alpha_for_target_rho(
            active.astype(np.complex128), rho_target=rho_target,
        )

    # --- Step 3: Verify full-objective stability at this alpha ---
    # The alpha was sized for Q-only. Check the full M (Q+V) doesn't
    # have eigenvalues that make alpha*lambda > 2 (instability).
    _, eigs_full = _build_M_cont(gw_full[cont_idx], QK_c_full)
    eig_max_full = float(np.max(np.abs(eigs_full)))
    if alpha * eig_max_full >= 1.95:
        # Full objective would be unstable — reduce alpha to safe value
        alpha = min(alpha, 1.9 / max(eig_max_full, 1e-14))

    # --- Step 4: Discrete g_w (safety-factored curvature) ---
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

    return gw_full, alpha


# ═══════════════════════════════════════════════════════════════════════════════
#  C2: Continuous g_w tuning (targets M_full^c)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_continuous_gw(
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    G_w_init_list: List[NDArray[np.float64]],
    *,
    zone_ids: Optional[List[int]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
    safety_factor: float = 2.0,
    gersh_floor_fraction: float = 1.0,
    floors: Optional[Dict[str, float]] = None,
    rho_target: float = 0.95,
    kappa_target: float = 50.0,
    alpha_target: float = 0.5,
    lambda_target: float = 1.5,
    lambda_max_alpha_target: float = 1.8,
    max_pump_iters: int = 20,
    eigenvalue_init_max_boost: float = 10.0,
    sensitivity_filter_frac: float = 0.01,
    warnings_list: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[List[NDArray[np.float64]], float]:
    """Tune continuous g_w to satisfy C2: rho(I - alpha*M_full^c) < 1.

    g_w controls CONDITIONING.  alpha controls STABILITY.

    Phase 1: Gershgorin as FLOOR on continuous curvature (user init trusted).
    Phase 1b: Sensitivity filter — exclude near-zero actuators.
    Phase 1c: Eigenvalue-based selective boost for dominant modes.
    Phase 2: Conditioning pump on M_full^c to reduce kappa AND lambda_max.
    Phase 3: Compute alpha from M_full^c eigenspectrum.

    Returns (gw_tuned_list, alpha_tso).
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))
    if floors is None:
        floors = {}
    if warnings_list is None:
        warnings_list = []

    # Extract continuous indices per zone
    cont_indices: List[List[int]] = []
    for k in range(n_zones):
        if actuator_counts is not None:
            ci, _ = _partition_indices(actuator_counts[k])
        else:
            ci = list(range(len(G_w_init_list[k])))
        cont_indices.append(ci)

    # --- Phase 1: Gershgorin as FLOOR on per-zone CONTINUOUS curvature ---
    # Trust user's init g_w; only override if it violates the per-actuator
    # necessary condition by a margin controlled by gersh_floor_fraction.
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

        gw_gersh = safety_factor * c_diag / 2.0  # full Gershgorin bound
        gw_gersh_floor = gw_gersh * gersh_floor_fraction  # reduced floor
        gw_cont = np.maximum(G_w_init_list[idx][ci], gw_gersh_floor)
        gw_list[idx][ci] = gw_cont

    # Apply floors
    for idx in range(n_zones):
        if actuator_counts is not None:
            gw_list[idx] = _apply_gw_floors(
                gw_list[idx], actuator_counts[idx], floors, _TSO_COLUMN_ORDER,
            )

    # --- Phase 2: Conditioning pump on M_full^c ---
    n_c_per = [len(ci) for ci in cont_indices]
    n_total_c = sum(n_c_per)

    if n_total_c == 0:
        return gw_list, 1.0

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

    # --- Filter out near-zero-sensitivity actuators from eigenanalysis ---
    # Actuators with negligible Q-weighted sensitivity columns create
    # spurious near-zero eigenvalues that inflate kappa and alpha.
    col_norms = np.zeros(n_total_c)
    off = 0
    for k_idx, k in enumerate(zone_ids):
        ci = cont_indices[k_idx]
        if not ci:
            off += 0
            continue
        QH_kk = QH_cache.get(('QH_ii', k))
        if QH_kk is not None:
            for j_local in range(len(ci)):
                col_norms[off + j_local] = float(np.linalg.norm(QH_kk[:, j_local]))
        off += len(ci)

    norm_max = max(float(np.max(col_norms)), 1e-14)
    sens_threshold = sensitivity_filter_frac * norm_max
    active_mask = col_norms > sens_threshold
    n_active = int(np.sum(active_mask))
    n_inert = n_total_c - n_active

    if verbose and n_inert > 0:
        print(f"  [C2 filter] {n_inert}/{n_total_c} continuous actuators "
              f"have negligible sensitivity (< 1% of max) — excluded from "
              f"eigenanalysis")

    # Build active-only index mapping
    active_indices = np.where(active_mask)[0]

    def _build_M_full_c(gw_c_flat_in):
        """Build M from ACTIVE actuators only."""
        if n_active == 0:
            return np.zeros((1, 1)), np.array([0.0])

        # Extract active g_w
        gw_active = gw_c_flat_in[active_indices]

        # Build full M first, then extract active submatrix
        M_full = np.zeros((n_total_c, n_total_c))
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
                C_ij = QH_ii.T @ QH_ij
                M_ij = (gi[:, None] * C_ij) * gj[None, :]
                M_full[r0:r0+n_c_per[i_idx], c0:c0+n_c_per[j_idx]] = M_ij

        # Extract active-only submatrix
        M = M_full[np.ix_(active_indices, active_indices)]
        eigs = np.linalg.eigvals(M)
        return M, eigs

    # --- Phase 1b: Eigenvalue-based g_w init (selective boost) ---
    # Only boost active actuators; map back to full flat vector
    def _active_M_builder(gw_active_in):
        gw_full = gw_c_flat.copy()
        gw_full[active_indices] = gw_active_in
        return _build_M_full_c(gw_full)

    if n_active > 0:
        gw_active = gw_c_flat[active_indices].copy()
        gw_active = _eigenvalue_init_gw(
            _active_M_builder, gw_active,
            lambda_target=lambda_target,
            max_boost=eigenvalue_init_max_boost,
        )
        gw_c_flat[active_indices] = gw_active

        # Pump on active actuators only
        gw_active_tuned, kappa, n_iters = _conditioning_pump(
            gw_c_flat[active_indices], _active_M_builder,
            kappa_target=kappa_target,
            alpha_target=alpha_target,
            lambda_max_alpha_target=lambda_max_alpha_target,
            max_iters=max_pump_iters,
        )
        gw_c_flat[active_indices] = gw_active_tuned
    else:
        kappa = 1.0
        n_iters = 0

    gw_c_tuned = gw_c_flat

    # Write back
    off = 0
    for k in range(n_zones):
        ci = cont_indices[k]
        gw_list[k][ci] = gw_c_tuned[off:off + n_c_per[k]]
        off += n_c_per[k]

    # --- Phase 3: Compute alpha from M_full^c ---
    _, eigs_c = _build_M_full_c(gw_c_tuned)
    eig_max = max(float(np.max(np.abs(eigs_c))), 1e-14)
    active_c = eigs_c[np.abs(eigs_c) > _EIG_ACTIVE_FRAC * eig_max]
    alpha_tso = _find_alpha_for_target_rho(active_c, rho_target=rho_target)

    # --- Alpha divergence warning ---
    if alpha_tso > 3.0 * alpha_target:
        msg = (f"alpha_tso={alpha_tso:.3f} >> alpha_target={alpha_target:.3f}. "
               f"g_w is likely over-inflated — actuators will be sluggish.")
        _log.warning(msg)
        warnings_list.append(msg)
    elif alpha_tso < 0.3 * alpha_target:
        msg = (f"alpha_tso={alpha_tso:.3f} << alpha_target={alpha_target:.3f}. "
               f"g_w may be insufficient for target convergence rate.")
        _log.warning(msg)
        warnings_list.append(msg)

    if verbose:
        rho_at = float(np.max(np.abs(1.0 - alpha_tso * active_c))) if len(active_c) > 0 else 0.0
        lam_max = float(np.max(active_c.real)) if len(active_c) > 0 else 0.0
        lam_min = float(np.min(active_c.real[active_c.real > _EIG_ACTIVE_FRAC * max(lam_max, 1e-14)])) if len(active_c) > 0 else 0.0
        print(f"  [C2 tune] kappa = {kappa:.1f} after {n_iters} pump iters, "
              f"lam = [{lam_min:.4f}, {lam_max:.4f}], "
              f"alpha_tso = {alpha_tso:.4f}, rho = {rho_at:.4f}")

        # Effective step-size per actuator
        print("  [C2 effective gains]")
        off = 0
        for k in range(n_zones):
            ci = cont_indices[k]
            if not ci:
                if verbose:
                    print(f"    Zone {zone_ids[k]} [TRIVIAL — discrete-only]")
                continue
            gw_c_k = gw_c_tuned[off:off + n_c_per[k]]
            eff = alpha_tso / (2.0 * gw_c_k)
            sluggish = np.sum(eff < 1e-3)
            print(f"    Zone {zone_ids[k]}: eff_gain = [{np.min(eff):.4f}, {np.max(eff):.4f}]"
                  + (f"  ({sluggish} sluggish)" if sluggish else ""))
            off += n_c_per[k]

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
            g_min += 2.0 * row_l1

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
    n_interfaces: int = 0
    """Number of Q-interface rows in H (first n_interfaces rows)."""


def recommend_dso_weights(
    dso_inputs: List[DSOTuneInput],
    *,
    delta_q_typ_mvar: float = 10.0,
    delta_v_typ_pu: float = 0.01,
    n_inner: int = 9,
    cascade_target: float = 0.8,
) -> List[Dict[str, float]]:
    """Compute recommended g_q / g_v ratio for each DSO.

    **Gradient-balance normalisation** — sets g_q/g_v so that a typical
    Q error and a typical V error produce equal gradient magnitudes:

        g_q · ΔQ_typ · ||H_Q||_F  =  g_v · ΔV_typ · ||H_V||_F

        ⟹  g_q/g_v = (ΔV_typ · ||H_V||_F) / (ΔQ_typ · ||H_Q||_F)

    If the current g_q is already above the balanced value, Q tracking
    is already prioritised — the step-size problem is elsewhere (alpha/g_w).

    Parameters
    ----------
    dso_inputs : list of DSOTuneInput (must have n_interfaces > 0)
    delta_q_typ_mvar : typical Q tracking error [Mvar]
    delta_v_typ_pu : typical voltage deviation [p.u.]
    n_inner : DSO iterations per TSO step (T_TSO / T_DSO)
    cascade_target : desired Q convergence fraction within n_inner steps

    Returns list of dicts per DSO.
    """
    results: List[Dict[str, float]] = []

    for d in dso_inputs:
        n_q = d.n_interfaces
        if n_q == 0:
            results.append({'dso_id': d.dso_id, 'gq_gv_ratio': float('nan'),
                            'note': 'no Q-interface rows'})
            continue

        H_Q = d.H[:n_q, :]
        # Voltage rows start after Q rows; find how many have nonzero weight
        q_diag = d.q_obj_diag
        n_v = int(np.sum(q_diag[n_q:] > 0))
        H_V = d.H[n_q:n_q + n_v, :] if n_v > 0 else np.zeros((1, d.H.shape[1]))

        frob_Q = float(np.linalg.norm(H_Q, 'fro'))
        frob_V = float(np.linalg.norm(H_V, 'fro'))

        # Current weights from q_obj_diag
        g_q_cur = float(q_diag[0]) if n_q > 0 else 1.0
        g_v_cur = float(q_diag[n_q]) if n_v > 0 else 1.0

        # Gradient-balance ratio (linear, not squared)
        if frob_Q > 1e-14 and frob_V > 1e-14:
            ratio = (delta_v_typ_pu * frob_V) / (delta_q_typ_mvar * frob_Q)
        else:
            ratio = float('nan')

        g_q_balanced = ratio * g_v_cur if not np.isnan(ratio) else g_q_cur
        q_dominance = g_q_cur / g_q_balanced if g_q_balanced > 1e-14 else float('inf')

        results.append({
            'dso_id': d.dso_id,
            'gq_gv_ratio': ratio,
            'H_Q_frob': frob_Q,
            'H_V_frob': frob_V,
            'g_q_current': g_q_cur,
            'g_v_current': g_v_cur,
            'g_q_balanced': g_q_balanced,
            'q_dominance': q_dominance,
        })

    # Print diagnostic table
    print("\n  ── DSO Weight Diagnostic (g_q / g_v) ─────────────────────────")
    print(f"    Assumptions: ΔQ_typ = {delta_q_typ_mvar} Mvar, "
          f"ΔV_typ = {delta_v_typ_pu} p.u., "
          f"N_inner = {n_inner}")
    for r in results:
        if 'note' in r:
            print(f"    {r['dso_id']}: {r['note']}")
            continue
        print(f"    {r['dso_id']}:")
        print(f"      ||H_Q||_F = {r['H_Q_frob']:.4f},  ||H_V||_F = {r['H_V_frob']:.4f}")
        print(f"      Current:     g_q = {r['g_q_current']:.1f},  g_v = {r['g_v_current']:.1f}")
        print(f"      Balanced:    g_q = {r['g_q_balanced']:.2f}  "
              f"(g_q/g_v = {r['gq_gv_ratio']:.6f})")
        dom = r['q_dominance']
        if dom > 1.5:
            print(f"      Status:      Q already dominant ({dom:.0f}x above balanced) "
                  f"— step-size (alpha/g_w) is the bottleneck, not g_q/g_v")
        elif dom < 0.5:
            print(f"      Status:      V dominant — increase g_q to at least {r['g_q_balanced']:.1f}")
        else:
            print(f"      Status:      near balanced ({dom:.1f}x)")
    print("  ───────────────────────────────────────────────────────────────\n")

    return results


def auto_tune(
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    gw_tso_init: List[NDArray[np.float64]],
    dso_inputs: Optional[List[DSOTuneInput]] = None,
    gw_dso_init: Optional[List[NDArray[np.float64]]] = None,
    tuning_config: TuningConfig = TuningConfig(),
    floors_tso: Optional[Dict[str, float]] = None,
    floors_dso: Optional[Dict[str, float]] = None,
    tso_period_s: float = 180.0,
    dso_period_s: float = 20.0,
    verbose: bool = False,
) -> TuningResult:
    """Orchestrate three-condition auto-tuning.

    Sequence: DSO (C1) -> continuous TSO (C2) -> discrete TSO (C3).

    Parameters
    ----------
    H_blocks : dict[(i,j)] -> H_ij sensitivity blocks.
    Q_obj_list : per-zone Q_obj diagonals (g_q / g_v weights).
    actuator_counts : per-zone actuator count dicts.
    zone_ids : zone integer IDs.
    gw_tso_init : initial per-zone TSO g_w (user hand-tuned, used as floor).
    dso_inputs : list of DSOTuneInput (one per DSO).
    gw_dso_init : initial per-DSO g_w vectors.
    tuning_config : all tuning parameters (TuningConfig dataclass).
    floors_tso, floors_dso : per-actuator-type minimum g_w.
    tso_period_s, dso_period_s : cascade timing for margin computation.
    verbose : print tuning tables.
    """
    if floors_tso is None:
        floors_tso = {}
    if floors_dso is None:
        floors_dso = {}
    if dso_inputs is None:
        dso_inputs = []
    if gw_dso_init is None:
        gw_dso_init = []

    tc = tuning_config

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
            safety_factor_continuous=tc.dso_safety_factor_continuous,
            safety_factor_discrete=tc.dso_safety_factor_discrete,
            floors=floors_dso,
            actuator_counts=ac,
            rho_target=tc.dso_rho_target,
            alpha_min=tc.dso_alpha_min,
            n_interfaces=d.n_interfaces,
        )
        gw_dso_tuned.append(gw)
        alpha_dso_list.append(alpha)

        if verbose:
            print(f"  [C1] {d.dso_id}: alpha = {alpha:.4f}")

    # ── Apply DSO alpha boost (speed up tracking, then verify) ───────
    if tc.alpha_dso_boost != 1.0 and alpha_dso_list:
        for d_idx, d in enumerate(dso_inputs):
            alpha_base = alpha_dso_list[d_idx]
            alpha_boosted = alpha_base * tc.alpha_dso_boost
            ac = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}

            r = analyse_dso_stability(
                H_dso=d.H, Q_dso=d.q_obj_diag,
                G_w_dso=gw_dso_tuned[d_idx],
                dso_id=d.dso_id, actuator_counts=ac,
                alpha=alpha_boosted,
                tso_period_s=tso_period_s, dso_period_s=dso_period_s,
            )
            if r.stable:
                alpha_dso_list[d_idx] = alpha_boosted
                if verbose:
                    print(f"  [C1 boost] {d.dso_id}: alpha {alpha_base:.4f} "
                          f"x{tc.alpha_dso_boost:.1f} -> {alpha_boosted:.4f} "
                          f"(rho = {r.M_cont_spectral_radius:.4f}) [ok]")
            else:
                warnings.append(
                    f"C1 boost: {d.dso_id} x{tc.alpha_dso_boost:.1f} "
                    f"rejected (rho = {r.M_cont_spectral_radius:.4f})")
                if verbose:
                    print(f"  [C1 boost] {d.dso_id}: x{tc.alpha_dso_boost:.1f} "
                          f"rejected (rho = {r.M_cont_spectral_radius:.4f}), "
                          f"keeping alpha = {alpha_base:.4f}")

    # ── C2: Continuous TSO tuning ─────────────────────────────────────────
    gw_tso, alpha_tso = tune_continuous_gw(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_init_list=gw_tso_init,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        safety_factor=tc.tso_safety_factor_continuous,
        gersh_floor_fraction=tc.tso_gersh_floor_fraction,
        floors=floors_tso,
        rho_target=tc.tso_rho_target,
        kappa_target=tc.tso_kappa_target,
        alpha_target=tc.tso_alpha_target,
        lambda_target=tc.lambda_target,
        lambda_max_alpha_target=tc.lambda_max_alpha_target,
        max_pump_iters=tc.pump_max_iters,
        eigenvalue_init_max_boost=tc.eigenvalue_init_max_boost,
        sensitivity_filter_frac=tc.sensitivity_filter_frac,
        warnings_list=warnings,
        verbose=verbose,
    )

    # ── C3: Discrete TSO tuning ───────────────────────────────────────────
    gw_tso = tune_discrete_gw(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        safety_factor=tc.c3_safety_factor,
        verbose=verbose,
    )

    # ── Verify ────────────────────────────────────────────────────────────
    # Verify C1 (with tuned alpha)
    for d_idx, d in enumerate(dso_inputs):
        ac = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        alpha_d = alpha_dso_list[d_idx] if d_idx < len(alpha_dso_list) else 1.0
        r = analyse_dso_stability(
            H_dso=d.H, Q_dso=d.q_obj_diag, G_w_dso=gw_dso_tuned[d_idx],
            dso_id=d.dso_id, actuator_counts=ac, alpha=alpha_d,
            tso_period_s=tso_period_s, dso_period_s=dso_period_s,
        )
        if not r.stable:
            c1_ok = False
            warnings.append(f"C1: {d.dso_id} still unstable after tuning "
                            f"(rho = {r.M_cont_spectral_radius:.4f})")

    # Verify C2 + C3 (with tuned alpha)
    stab = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        alpha=alpha_tso,
        verbose=False,
    )
    c2_ok = stab.c2_satisfied
    c3_ok = stab.c3_satisfied

    if not c2_ok:
        warnings.append("C2: continuous stability not achieved after tuning")
    if not c3_ok:
        warnings.append("C3: discrete small-gain not achieved after tuning")

    # ── Post-tuning feasibility: check effective actuator speed ───────────
    for i_idx, i in enumerate(zone_ids):
        ci, _ = _partition_indices(actuator_counts[i_idx])
        if not ci:
            continue
        gw_ci = gw_tso[i_idx][ci]
        eff_gain = alpha_tso / (2.0 * gw_ci)
        for k, eg in enumerate(eff_gain):
            if eg < 1e-3:
                msg = (f"Zone {i} actuator {k}: effective gain = {eg:.4e} — "
                       f"too conservative (g_w = {gw_ci[k]:.1f})")
                _log.warning(msg)
                warnings.append(msg)

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
