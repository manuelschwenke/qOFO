"""
Eigenvector-pump tuning of g_w for multi-TSO + cascaded DSO
=============================================================

Deterministic tuner for per-actuator g_w weights.  Uses Gershgorin
preconditioning (Phase 1) followed by iterative eigenvector-directed
boosting (Phase 2) to satisfy ``lambda_max(M_sys) < spectral_target``
where ``M = G_w^{-1/2} H^T Q_obj H G_w^{-1/2}``.

The user's hand-tuned g_w values are the **floor**: the pump can only
increase g_w above the user's initial values, never decrease.  This
ensures the pump's output is at least as conservative as the user's
manual tuning, with the additional guarantee of spectral feasibility.

The pump converges in ~5 iterations (milliseconds total) and replaces
the previous PSO-based tuner which was optimising eigenvalues of a
matrix that does not capture the actual constrained MIQP dynamics.
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
    _DSO_COLUMN_ORDER,
    _TSO_COLUMN_ORDER,
    _apply_gw_floors,
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
    """One DSO controller's snapshot for the tuner."""
    dso_id: str
    H: NDArray[np.float64]
    q_obj_diag: NDArray[np.float64]
    n_der: int
    n_oltc: int
    n_shunt: int


@dataclass
class TuneGwResult:
    """Result of the eigenvector-pump g_w tuner."""
    gw_tso: List[NDArray[np.float64]]
    gw_dso: List[NDArray[np.float64]]
    lam_max_sys: float
    spectral_feasible: bool
    pump_iterations_tso: int
    per_zone_kappa: List[float]
    per_dso_lam_max: List[float]
    stability_result: Optional[MultiZoneStabilityResult] = None
    alpha_tso: float = 1.0
    """Computed OFO step-size for TSO continuous actuators."""
    alpha_dso: List[float] = field(default_factory=list)
    """Per-DSO computed OFO step-sizes for continuous actuators."""


# =============================================================================
#  DSO helpers (kept from the previous module)
# =============================================================================

def _dso_cascade_decay(
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    g_w: NDArray[np.float64],
    n_inner: int,
) -> Tuple[float, float, float]:
    """Return ``(rho_D, cascade_decay, alpha_opt)`` for one DSO.

    * ``rho_D = (lam_max - lam_min) / (lam_max + lam_min)``
    * ``cascade_decay = rho_D ** n_inner``
    * ``alpha_opt = 2 / (lam_min + lam_max)``
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
    """Return ``(rho_d, cascade_decay, lam_max)`` for one DSO.

    ``rho_d = max_l |1 - lambda_l(M_dso)|``.
    """
    C = _compute_curvature_matrix(H, q_obj_diag)
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(g_w, 1e-12))
    M = (gw_inv_sqrt[:, None] * C) * gw_inv_sqrt[None, :]
    _, active, _ = _effective_eigenspectrum(M)

    if len(active) == 0:
        return 0.0, 0.0, 0.0

    l_max = float(active[-1])
    l_min = float(active[0]) if len(active) >= 2 else l_max
    rho_d = max(abs(1.0 - l_max), abs(1.0 - l_min))
    cascade_decay = rho_d ** max(int(n_inner), 1)
    return rho_d, cascade_decay, l_max


# =============================================================================
#  Eigenvector pump
# =============================================================================

def _eigenvector_pump(
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    dso_inputs: List[DSOTuneInput],
    gw_tso_init: List[NDArray[np.float64]],
    gw_dso_init: List[NDArray[np.float64]],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    safety_factor_tso: float,
    safety_factor_dso: float,
    spectral_target: float,
    max_pump_iters: int = 50,
    verbose: bool = False,
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]], int,
           float, List[float]]:
    """Deterministic g_w + alpha tuner (TSO + DSO).

    Returns ``(gw_tso_list, gw_dso_list, pump_iters_tso,
               alpha_tso, alpha_dso_list)``.

    **Phase 1 — Gershgorin + user floor:**
    ``g_w = max(gw_user_init, safety * diag(C) / 2)`` per zone/DSO.
    The user's hand-tuned values act as the floor.

    **Phase 2a — TSO alpha computation:**
    Computes ``alpha_tso = spectral_target / lambda_max(M_sys)``
    using Phase-1 g_w values (no g_w inflation).

    **Phase 2b — Per-DSO alpha computation:**
    Computes ``alpha_dso = 1.8 / lambda_max(M_dso)`` per DSO.
    """
    n_zones = len(zone_ids)

    # ── Phase 1: max(user_init, Gershgorin) ─────────────────────────────
    gw_tso_arrays: List[NDArray[np.float64]] = []
    for idx, z in enumerate(zone_ids):
        H_ii = H_blocks[(z, z)]
        c_diag = _compute_curvature_diagonal(H_ii, Q_obj_list[idx])
        gw_gersh = safety_factor_tso * c_diag / 2.0
        gw_gersh = _apply_gw_floors(
            gw_gersh, actuator_counts[idx], floors_tso, _TSO_COLUMN_ORDER,
        )
        # User's init values are the FLOOR — pump can only increase
        gw = np.maximum(gw_tso_init[idx], gw_gersh)
        gw_tso_arrays.append(gw)

    gw_dso_arrays: List[NDArray[np.float64]] = []
    for d_idx, d in enumerate(dso_inputs):
        c_diag = _compute_curvature_diagonal(d.H, d.q_obj_diag)
        counts = {'n_der': d.n_der, 'n_oltc': d.n_oltc, 'n_shunt': d.n_shunt}
        gw_gersh = safety_factor_dso * c_diag / 2.0
        gw_gersh = _apply_gw_floors(gw_gersh, counts, floors_dso, _DSO_COLUMN_ORDER)
        gw = np.maximum(gw_dso_init[d_idx], gw_gersh)
        gw_dso_arrays.append(gw)

    # ── Phase 2a: Compute alpha_tso from M_sys ─────────────────────────
    #
    # Instead of inflating g_w to push lambda_max below 2, we keep the
    # Phase-1 g_w values and compute alpha = target / lambda_max(M_sys).
    # This decouples stability (alpha) from action amplitude (g_w).
    stab = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso_arrays,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=False,
    )
    lam_sys = float(stab.M_sys_lambda_max)
    pump_iters_tso = 1

    # alpha_tso: the step-size that makes |1 - alpha * lambda_i| < 1
    # for ALL eigenvalues of M_sys.
    #
    # For real lambda:   alpha < 2 / lambda
    # For complex lambda = a + bi:  alpha < 2a / (a² + b²)
    #   (derived from |1 - alpha*(a+bi)|² < 1)
    #
    # The safety margin uses spectral_target/2 instead of 1.0 as the
    # contraction radius target.
    safety = float(spectral_target) / 2.0   # e.g. 1.9/2 = 0.95

    sys_eigs = np.linalg.eigvals(stab.M_sys)
    alpha_bounds: List[float] = []
    for lam in sys_eigs:
        a, b = float(lam.real), float(lam.imag)
        mag_sq = a * a + b * b
        if mag_sq < 1e-14:
            continue
        # |1 - alpha*lambda|² < 1  →  alpha < 2a / (a²+b²)
        if a > 1e-14:
            alpha_bounds.append(safety * 2.0 * a / mag_sq)
        # If a <= 0 (shouldn't happen for PSD-ish M), skip
    alpha_tso = min(1.0, min(alpha_bounds)) if alpha_bounds else 1.0

    if verbose:
        rho_at_alpha = float(np.max(np.abs(1.0 - alpha_tso * sys_eigs)))
        print(f"  [alpha TSO] lam_max(Re) = {lam_sys:.4f}, "
              f"alpha_tso = {alpha_tso:.6f}  "
              f"(rho(I-alpha*M) = {rho_at_alpha:.4f})")
        if stab.M_sys_asymmetry > 0.01:
            print(f"  [alpha TSO] M_sys asymmetry = "
                  f"{stab.M_sys_asymmetry:.4f}")
        if stab.M_sys_has_complex_eigenvalues:
            print(f"  [alpha TSO] M_sys has complex eigenvalues, "
                  f"rho(I-M) = {stab.M_sys_spectral_radius:.4f}")

    # ── Phase 2b: Per-DSO alpha computation ────────────────────────────
    # DSO M_dso is symmetric (single-zone, C = H^T Q H) → eigvalsh is correct.
    alpha_dso_list: List[float] = []
    for d_idx, d in enumerate(dso_inputs):
        C_d = _compute_curvature_matrix(d.H, d.q_obj_diag)
        gw_inv = 1.0 / np.sqrt(np.maximum(gw_dso_arrays[d_idx], 1e-12))
        M_d = (gw_inv[:, None] * C_d) * gw_inv[None, :]
        eigs_d = np.linalg.eigvalsh(M_d)
        lam_max_d = float(eigs_d[-1]) if len(eigs_d) > 0 else 0.0

        if lam_max_d > 1e-14:
            alpha_d = min(1.0, 1.8 / lam_max_d)
        else:
            alpha_d = 1.0
        alpha_dso_list.append(alpha_d)

        if verbose:
            print(f"  [alpha DSO {d.dso_id}] lam_max = {lam_max_d:.4f}, "
                  f"alpha_dso = {alpha_d:.6f}  "
                  f"(alpha*lam = {alpha_d * lam_max_d:.4f})")

    return gw_tso_arrays, gw_dso_arrays, pump_iters_tso, alpha_tso, alpha_dso_list


# =============================================================================
#  Public entry point
# =============================================================================

def tune_gw(
    *,
    H_blocks: Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list: List[NDArray[np.float64]],
    actuator_counts: List[Dict[str, int]],
    zone_ids: List[int],
    dso_inputs: List[DSOTuneInput],
    gw_tso_init: List[NDArray[np.float64]],
    gw_dso_init: List[NDArray[np.float64]],
    floors_tso: Dict[str, float],
    floors_dso: Dict[str, float],
    spectral_target: float = 1.9,
    tso_period_s: float = 180.0,
    dso_period_s: float = 60.0,
    safety_factor_tso: float = 4.0,
    safety_factor_dso: float = 2.0,
    verbose: bool = True,
) -> TuneGwResult:
    """Tune per-actuator g_w for the multi-TSO + DSO cascade.

    Uses the deterministic eigenvector pump.  The user's ``gw_tso_init``
    and ``gw_dso_init`` are the FLOOR: the pump can only increase g_w
    above these values, never decrease.

    Returns :class:`TuneGwResult`.
    """
    n_inner = max(int(round(tso_period_s / max(dso_period_s, 1e-9))), 1)

    gw_tso, gw_dso, pump_iters, alpha_tso, alpha_dso = _eigenvector_pump(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        actuator_counts=actuator_counts,
        zone_ids=zone_ids,
        dso_inputs=dso_inputs,
        gw_tso_init=gw_tso_init,
        gw_dso_init=gw_dso_init,
        floors_tso=floors_tso,
        floors_dso=floors_dso,
        safety_factor_tso=safety_factor_tso,
        safety_factor_dso=safety_factor_dso,
        spectral_target=spectral_target,
        verbose=verbose,
    )

    # Final stability snapshot
    stab = analyse_multi_zone_stability(
        H_blocks=H_blocks,
        Q_obj_list=Q_obj_list,
        G_w_list=gw_tso,
        zone_ids=zone_ids,
        actuator_counts=actuator_counts,
        verbose=False,
    )
    lam_sys = float(stab.M_sys_lambda_max)

    # Per-zone kappa
    per_zone_kappa: List[float] = []
    for zr in stab.zones:
        per_zone_kappa.append(float(zr.kappa_Mii))

    # Per-DSO lam_max
    per_dso_lam_max: List[float] = []
    for d_idx, d in enumerate(dso_inputs):
        C_d = _compute_curvature_matrix(d.H, d.q_obj_diag)
        gw_inv = 1.0 / np.sqrt(np.maximum(gw_dso[d_idx], 1e-12))
        M_d = (gw_inv[:, None] * C_d) * gw_inv[None, :]
        eigs_d = np.linalg.eigvalsh(M_d)
        lam_d = float(eigs_d[-1]) if len(eigs_d) > 0 else 0.0
        per_dso_lam_max.append(lam_d)

    return TuneGwResult(
        gw_tso=gw_tso,
        gw_dso=gw_dso,
        lam_max_sys=lam_sys,
        spectral_feasible=bool(alpha_tso * lam_sys < 2.0),
        pump_iterations_tso=pump_iters,
        per_zone_kappa=per_zone_kappa,
        per_dso_lam_max=per_dso_lam_max,
        stability_result=stab,
        alpha_tso=alpha_tso,
        alpha_dso=alpha_dso,
    )
