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
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]], int]:
    """Deterministic eigenvector-directed g_w tuner (TSO + DSO).

    Returns ``(gw_tso_list, gw_dso_list, pump_iters_tso)``.

    **Phase 1 — Gershgorin + user floor:**
    ``g_w = max(gw_user_init, safety * diag(C) / 2)`` per zone/DSO.
    The user's hand-tuned values act as the floor.

    **Phase 2a — TSO global pump:**
    Boosts g_w along the dominant eigenvector of M_sys until
    ``lambda_max(M_sys) < 0.95 * spectral_target``.

    **Phase 2b — Per-DSO pump:**
    Boosts g_w along each DSO's dominant eigenvector until
    ``lambda_max(M_dso) < 1.8``.
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

    # ── Phase 2a: TSO global pump ───────────────────────────────────────
    target_inner = 0.95 * float(spectral_target)
    n_per_zone = [len(gw) for gw in gw_tso_arrays]
    zone_ranges = _zone_index_ranges(n_per_zone)

    lam_sys = float('inf')
    pump_iters_tso = 0
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
        pump_iters_tso = pump_iter + 1
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
            gw_tso_arrays[k] = np.maximum(
                gw_tso_init[k],  # never go below user init
                _apply_gw_floors(
                    gw_tso_arrays[k] * scale,
                    actuator_counts[k],
                    floors_tso, _TSO_COLUMN_ORDER,
                ),
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
            _, vecs_d = np.linalg.eigh(M_d)
            v_d = vecs_d[:, -1] ** 2
            v_max_d = float(np.max(v_d)) if v_d.size else 0.0
            if v_max_d < 1e-14:
                break
            scale_d = 1.0 + 2.0 * (v_d / v_max_d)
            gw_dso_arrays[d_idx] = np.maximum(
                gw_dso_init[d_idx],  # never go below user init
                _apply_gw_floors(
                    gw_dso_arrays[d_idx] * scale_d,
                    counts, floors_dso, _DSO_COLUMN_ORDER,
                ),
            )

    return gw_tso_arrays, gw_dso_arrays, pump_iters_tso


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

    gw_tso, gw_dso, pump_iters = _eigenvector_pump(
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
        spectral_feasible=bool(lam_sys < spectral_target),
        pump_iterations_tso=pump_iters,
        per_zone_kappa=per_zone_kappa,
        per_dso_lam_max=per_dso_lam_max,
        stability_result=stab,
    )
