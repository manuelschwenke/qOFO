"""
Stability Analysis for the Cascaded TSO-DSO OFO Controller
===========================================================

This module computes theoretical stability bounds for both the TSO and DSO
OFO controllers from a given :class:`~core.cascade_config.CascadeConfig` and
a pair of pre-computed sensitivity matrices H_TSO and H_DSO.

Theory  (Hauswirth et al., 2021)
---------------------------------
The OFO MIQP unconstrained step direction is:

    σ* = −G_w⁻¹ ∇f / 2

and the update is u^{k+1} = u^k + α σ*, i.e.

    u^{k+1} = u^k − (α/2) G_w⁻¹ ∇f(u^k)

For a quadratic tracking objective  f(u) = Σ_j q_j (y_j − y_set_j)²  with
y = H u + d,  the gradient is  ∇f = 2 H^T Q_obj (y − y_set)  and the Hessian
is  ∇²f = 2 H^T Q_obj H.  Define the *curvature matrix*

    C = H^T Q_obj H        (n_u × n_u,  symmetric PSD)

where Q_obj = diag(q_1, …, q_{n_y}) is the **per-output objective weight
matrix** — NOT a scalar!  For the TSO  Q_obj = diag(g_v … g_v, 0 … 0)
(voltage rows weighted, current rows zero).  For the DSO
Q_obj = diag(g_q … g_q, dso_g_v … dso_g_v, 0 … 0)
(Q-interface rows, voltage rows, current rows).

The contraction condition for the fixed-point iteration is:

    eigenvalues of  α G_w⁻¹ C  ∈ (0, 2)

Through the similarity transform  M = G_w^{−1/2} C G_w^{−1/2}  (symmetric),
this becomes:

    α λ_max(M) < 2        →        α_max = 2 / λ_max(M)

The *per-actuator necessary condition* (Gershgorin diagonal) is:

    g_w[i] > α C_{ii} / 2

where C_{ii} = (H^T Q_obj H)_{ii} is the per-actuator self-curvature.

For the *cascade*, the DSO must converge within T_T / T_D iterations.
The spectral contraction rate is:

    ρ_D = max_i |1 − α λ_i(M)|

and the cascade margin is  1 − ρ_D^(T_T/T_D) > 0.

Usage
-----
    from analysis.stability_analysis import analyse_stability
    from core.cascade_config import CascadeConfig
    import numpy as np

    config = CascadeConfig()
    H_tso = ...   # shape (n_y_tso, n_u_tso)
    H_dso = ...   # shape (n_y_dso, n_u_dso)

    result = analyse_stability(
        config, H_tso, H_dso,
        n_tso_der=2, n_tso_oltc=1, n_tso_shunt=1,
        n_dso_der=8, n_dso_oltc=3, n_dso_shunt=3,
        n_tso_v_out=10, n_tso_i_out=3,
        n_dso_q_out=3, n_dso_v_out=12, n_dso_i_out=9,
    )

Author: Manuel Schwenke, TU Darmstadt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from core.cascade_config import CascadeConfig


# ─── Result containers ─────────────────────────────────────────────────────────

@dataclass
class ControllerStabilityResult:
    """Stability analysis result for one OFO controller layer."""

    # SVD / Jacobian properties (raw H, for diagnostics)
    sigma_max: float
    """Largest singular value of the raw sensitivity matrix H."""

    sigma_min: float
    """Smallest singular value of H (0 if H is rank-deficient)."""

    condition_number: float
    """Condition number σ_max / σ_min (inf if σ_min = 0)."""

    # Curvature and Lipschitz bounds
    L_eff: float
    """Effective curvature: λ_max(H^T Q_obj H)."""

    alpha_max: float
    """Maximum stable step size: 2 / λ_max(G_w^{-1/2} C G_w^{-1/2})."""

    # Per-actuator margins  (positive → stable, negative → unstable)
    actuator_margins: Dict[str, float] = field(default_factory=dict)
    """Per-actuator margin: g_w[i] − α·C_{ii}/2.  Positive = safe."""

    actuator_thresholds: Dict[str, float] = field(default_factory=dict)
    """Per-actuator critical threshold: α·C_{ii}/2."""

    # DSO cascade margin (only meaningful for DSO)
    dso_convergence_rate: Optional[float] = None
    """Spectral contraction rate ρ_D = max_i |1 − α λ_i(M)| (memoryless)."""

    dso_cascade_margin: Optional[float] = None
    """1 − ρ_D^(T_T/T_D): positive means DSO converges within T_T/T_D steps."""

    # Augmented PI analysis (when integral Q-tracking is active)
    augmented_rho: Optional[float] = None
    """Spectral radius of the augmented (u, q_int) contraction matrix.
    Accounts for integral Q-tracking dynamics.  None if g_qi = 0."""

    augmented_cascade_margin: Optional[float] = None
    """1 − ρ_aug^(T_T/T_D) for the augmented system."""

    # Eigenvalue diagnostics (top modes of preconditioned curvature M)
    eigenvalue_diagnostics: list = field(default_factory=list)
    """List of dicts for the top eigenvalues of M = G_w^{-½} C G_w^{-½}.

    Each dict has:
        'eigenvalue': float — λ_i(M)
        'contraction': float — |1 − α λ_i|
        'participation': Dict[str, float] — squared eigenvector weight per actuator
        'type_contribution': Dict[str, float] — summed weight per actuator type
    """

    # Classification
    stable: bool = True
    """True iff all actuator margins are positive AND α ≤ α_max."""

    marginal: bool = False
    """True iff stable but some margin < 20 % of its threshold."""

    warnings: list = field(default_factory=list)
    """Human-readable warning strings."""


@dataclass
class CascadeStabilityResult:
    """Top-level result container for the full cascade."""

    tso: ControllerStabilityResult
    dso: ControllerStabilityResult

    cascade_stable: bool
    """True iff both layers are individually stable AND DSO cascade margin > 0."""

    summary: str = ""
    """Human-readable one-paragraph summary."""


# ─── Helper: build Q_obj diagonal ──────────────────────────────────────────────

def _build_q_obj(
    n_q: int,
    n_v: int,
    n_i: int,
    g_q: float,
    g_v: float,
) -> NDArray[np.float64]:
    """Build the per-output objective weight vector Q_obj.

    Row ordering in H is assumed to be  [Q_trafo | V_bus | I_line].
    Current rows get weight 0 (constraint-only, not tracked).

    Parameters
    ----------
    n_q : Number of Q-interface / trafo output rows.
    n_v : Number of voltage observation output rows.
    n_i : Number of current / branch output rows.
    g_q : Weight for Q-interface tracking (0 if not tracked).
    g_v : Weight for voltage tracking.

    Returns
    -------
    1-D array of length n_q + n_v + n_i.
    """
    return np.concatenate([
        np.full(n_q, g_q),
        np.full(n_v, g_v),
        np.full(n_i, 0.0),
    ])


# ─── Core analysis function ─────────────────────────────────────────────────────

def _analyse_layer(
    layer_name: str,
    H: NDArray[np.float64],
    alpha: float,
    q_obj_diag: NDArray[np.float64],
    gw_vector: NDArray[np.float64],
    actuator_names: list[str],
    gu_vector: Optional[NDArray[np.float64]] = None,
    tso_period_s: Optional[float] = None,
    dso_period_s: Optional[float] = None,
    # Integral Q-tracking parameters (augmented PI dynamics)
    n_q_integral: int = 0,
    g_qi: float = 0.0,
    lambda_qi: float = 0.0,
) -> ControllerStabilityResult:
    """Compute stability bounds for one controller layer.

    Parameters
    ----------
    layer_name :
        Human-readable name, e.g. 'TSO' or 'DSO'.
    H :
        Full sensitivity matrix (n_y × n_u).
    alpha :
        OFO step-size gain α.
    q_obj_diag :
        Per-output objective weight vector (length n_y).
        E.g. [g_q, g_q, g_q, g_v, …, g_v, 0, …, 0] for DSO.
    gw_vector :
        Per-actuator g_w values (length n_u).
    actuator_names :
        Names matching gw_vector entries (for reporting).
    gu_vector :
        Per-actuator g_u values (optional, defaults to zeros).
    tso_period_s :
        TSO sampling period [s] (required for DSO cascade margin).
    dso_period_s :
        DSO sampling period [s] (required for DSO cascade margin).
    n_q_integral :
        Number of Q-interface rows that feed the integral tracker.
        These must be the first n_q_integral rows of H.
    g_qi :
        Integral Q-tracking weight (0 = disabled).
    lambda_qi :
        Leaky integrator decay factor (0 < λ < 1).
    """
    if H.ndim != 2:
        raise ValueError(f'{layer_name}: H must be a 2-D array, got shape {H.shape}.')
    n_y, n_u = H.shape
    if len(q_obj_diag) != n_y:
        raise ValueError(
            f'{layer_name}: q_obj_diag length {len(q_obj_diag)} does not match '
            f'H.shape[0] = {n_y}.'
        )
    if len(gw_vector) != n_u:
        raise ValueError(
            f'{layer_name}: gw_vector length {len(gw_vector)} does not match '
            f'H.shape[1] = {n_u}.'
        )

    if gu_vector is None:
        gu_vector = np.zeros_like(gw_vector)
    if len(gu_vector) != len(gw_vector):
        raise ValueError(f'{layer_name}: gu_vector and gw_vector must have the same length.')

    # ── SVD of raw H (diagnostics) ──────────────────────────────────────────────
    singular_values = np.linalg.svd(H, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1]) if len(singular_values) > 0 else 0.0
    condition_number = float(sigma_max / sigma_min) if sigma_min > 0.0 else np.inf

    # ── Curvature matrix  C = H^T Q_obj H ───────────────────────────────────────
    # Efficiently computed as (Q^{1/2} H)^T (Q^{1/2} H).
    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    HQ = q_sqrt[:, None] * H          # (n_y, n_u) — each row scaled
    C = HQ.T @ HQ                     # (n_u, n_u) — symmetric PSD

    # Eigenvalues of C
    C_eig = np.linalg.eigvalsh(C)
    L_eff = float(np.maximum(C_eig[-1], 0.0))    # λ_max(C)

    # ── Preconditioned curvature  M = G_w^{-1/2} C G_w^{-1/2} ──────────────────
    gw_total = gw_vector + gu_vector
    gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_total, 1e-12))
    # M_{ij} = C_{ij} / (sqrt(gw_i) * sqrt(gw_j))
    M = (gw_inv_sqrt[:, None] * C) * gw_inv_sqrt[None, :]
    M_eig, M_vecs = np.linalg.eigh(M)        # eigenvalues + eigenvectors
    M_max = float(np.maximum(M_eig[-1], 0.0))

    alpha_max = (2.0 / M_max) if M_max > 0.0 else np.inf

    # ── Eigenvalue diagnostics ───────────────────────────────────────────────────
    # For each eigenvalue, decompose the eigenvector to show which actuators
    # participate.  |v_i|² gives the fraction from scaled-actuator i.
    # Group by type prefix to get per-type contribution.
    #
    # We report:
    #   - Top modes (largest λ): these determine α_max and overshoot risk
    #   - Slowest active modes (smallest active λ): these determine ρ_D
    eigenvalue_diagnostics = []
    rel_active_thresh = max(M_max * 0.01, 1e-10)

    def _build_mode_dict(idx: int) -> Optional[dict]:
        lam = float(M_eig[idx])
        if lam < 1e-14:
            return None
        v = M_vecs[:, idx]
        v_sq = v ** 2
        contraction = abs(1.0 - alpha * lam)
        participation = {
            name: float(v_sq[i]) for i, name in enumerate(actuator_names)
        }
        type_contrib: Dict[str, float] = {}
        for i, name in enumerate(actuator_names):
            parts = name.rsplit('_', 1)
            atype = parts[0] if len(parts) > 1 and parts[-1].isdigit() else name
            type_contrib[atype] = type_contrib.get(atype, 0.0) + float(v_sq[i])
        return {
            'eigenvalue': lam,
            'contraction': contraction,
            'participation': participation,
            'type_contribution': type_contrib,
            'active': lam > rel_active_thresh,
        }

    # Top modes (largest first)
    n_top = min(5, n_u)
    seen_indices = set()
    for k in range(n_top):
        idx = n_u - 1 - k
        mode = _build_mode_dict(idx)
        if mode is None:
            break
        eigenvalue_diagnostics.append(mode)
        seen_indices.add(idx)

    # Slowest active mode(s): find the smallest eigenvalue(s) above the
    # active threshold that aren't already in the top-5
    active_mask = M_eig > rel_active_thresh
    active_indices = np.where(active_mask)[0]
    for idx in active_indices[:3]:  # up to 3 slowest active modes
        if idx not in seen_indices:
            mode = _build_mode_dict(int(idx))
            if mode is not None:
                mode['_slowest_active'] = True
                eigenvalue_diagnostics.append(mode)
                seen_indices.add(int(idx))

    # ── Per-actuator stability margins ────────────────────────────────────────────
    # Necessary condition (Gershgorin diagonal):  g_w[i] > α C_{ii} / 2
    C_diag = np.diag(C)
    actuator_margins: Dict[str, float] = {}
    actuator_thresholds: Dict[str, float] = {}
    for i, (name, gw_i, gu_i) in enumerate(zip(actuator_names, gw_vector, gu_vector)):
        threshold_i = alpha * float(C_diag[i]) / 2.0
        margin_i = float(gw_i + gu_i) - threshold_i
        actuator_margins[name] = margin_i
        actuator_thresholds[name] = threshold_i

    warnings = []

    # ── Step-size check ───────────────────────────────────────────────────────────
    if alpha > alpha_max:
        warnings.append(
            f'{layer_name}: alpha = {alpha:.4g} exceeds alpha_max = {alpha_max:.4g} '
            f'(lam_max(M) = {M_max:.4g}).  Instability likely.'
        )

    # ── Per-actuator warnings ─────────────────────────────────────────────────────
    for idx, (name, margin) in enumerate(actuator_margins.items()):
        threshold_i = actuator_thresholds[name]
        if margin < 0.0:
            warnings.append(
                f'{layer_name}: actuator "{name}" has NEGATIVE margin {margin:.4g} '
                f'(g_w+g_u = {float(gw_vector[idx] + gu_vector[idx]):.4g}, '
                f'threshold = {threshold_i:.4g}).'
            )
        elif threshold_i > 0.0 and margin < 0.2 * threshold_i:
            warnings.append(
                f'{layer_name}: actuator "{name}" margin {margin:.4g} is '
                f'< 20 %% of threshold (marginal).'
            )

    # ── Condition number warning ──────────────────────────────────────────────────
    if condition_number > 1e6:
        warnings.append(
            f'{layer_name}: raw H is ill-conditioned (kappa = {condition_number:.3e}). '
            f'Common cause: co-located DERs (same bus -> near-identical columns). '
            f'Benign for stability (G_w regularises), but raw sigma_min is meaningless.'
        )

    # ── DSO cascade convergence rate ──────────────────────────────────────────────
    dso_convergence_rate = None
    dso_cascade_margin = None
    if tso_period_s is not None and dso_period_s is not None:
        ratio = tso_period_s / dso_period_s  # e.g. 9

        # Spectral contraction rate:  ρ = max_i |1 − α λ_i(M)|
        #
        # Near-zero eigenvalues of M (from rank-deficient H, e.g. co-located
        # DERs on the same bus) give |1 − α·0| ≈ 1.  These are BENIGN:
        # the gradient is zero in those directions, so the iterate doesn't
        # move — but it doesn't need to, because the error is also zero.
        #
        # We use a relative threshold of 1% of λ_max(M) to filter out these
        # degenerate modes and compute a physically meaningful ρ_D.
        rel_threshold = max(M_max * 0.01, 1e-10)
        M_eig_active = M_eig[M_eig > rel_threshold]
        n_active = len(M_eig_active)
        n_inactive = n_u - n_active

        if n_active > 0:
            rho = float(np.max(np.abs(1.0 - alpha * M_eig_active)))
        else:
            rho = 0.0  # degenerate case: C = 0
        dso_convergence_rate = rho
        dso_cascade_margin = 1.0 - rho ** ratio

        if n_inactive > 0:
            # Find the largest filtered eigenvalue for reporting
            M_eig_inactive = M_eig[(M_eig > 1e-14) & (M_eig <= rel_threshold)]
            if len(M_eig_inactive) > 0:
                rho_raw = float(np.max(np.abs(
                    1.0 - alpha * M_eig[M_eig > 1e-14]
                )))
                warnings.append(
                    f'{layer_name}: {n_inactive} near-degenerate modes filtered '
                    f'(lam < {rel_threshold:.2e}).  '
                    f'rho_D (active modes only) = {rho:.4f},  '
                    f'rho_D (all modes, incl. degenerate) = {rho_raw:.4f}.  '
                    f'Degenerate modes are benign (co-located DERs / rank-deficient H).'
                )

        if rho >= 1.0:
            warnings.append(
                f'{layer_name}: spectral radius rho_D = {rho:.4f} >= 1.0 '
                f'(DSO diverges in active modes).  alpha exceeds stability bound.'
            )
        elif dso_cascade_margin <= 0.0:
            warnings.append(
                f'{layer_name}: DSO may NOT converge within {ratio:.0f} iterations '
                f'(rho_D = {rho:.4f}, rho_D^{ratio:.0f} = {rho**ratio:.4f}). '
                f'Increase alpha or reduce g_w to ensure quasi-steady-state.'
            )

    # ── Augmented PI analysis (integral Q-tracking) ─────────────────────────────
    augmented_rho = None
    augmented_cascade_margin = None
    if g_qi > 0.0 and n_q_integral > 0 and 0 < lambda_qi < 1.0:

        H_Q = H[:n_q_integral, :]  # (n_qi, n_u)
        n_qi = n_q_integral

        # Work in the eigenbasis of M = G_w^{-1/2} C G_w^{-1/2}.
        # M is symmetric PSD; its eigenvectors V satisfy M = V diag(λ) V^T.
        # In this basis, the preconditioned iteration is:
        #   ê_u^{k+1} = (I - α diag(λ)) ê_u^k - α g_qi Ĝw^{-1/2} H_Q^T e_qi^k
        # where ê_u = G_w^{1/2} e_u and the integral evolves in measurement space.
        #
        # Active modes only: λ_i > 1% λ_max(M).
        rel_threshold = max(M_max * 0.01, 1e-10)
        active_mask_m = M_eig > rel_threshold
        r = int(np.sum(active_mask_m))

        if r > 0:
            # Eigenvectors of M restricted to active modes: V_r (n_u x r)
            V_r = M_vecs[:, active_mask_m]  # (n_u, r)
            lam_r = M_eig[active_mask_m]  # (r,)

            # In the eigenbasis of M:
            #   A_uu = diag(1 - alpha * lam_r)           (r x r, diagonal)
            #   The coupling from integral to actuators involves G_w^{-1/2}:
            #   A_uq_hat = -alpha * g_qi * V_r^T G_w^{-1/2} H_Q^T  (r x n_qi)
            gw_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_total, 1e-12))
            # G_w^{-1/2} H_Q^T  (n_u x n_qi)
            GwInvSqrt_HQt = gw_inv_sqrt[:, None] * H_Q.T
            A_uq_hat = -alpha * g_qi * (V_r.T @ GwInvSqrt_HQt)  # (r, n_qi)

            # Integral updates with NEW actuator value (as per actual controller):
            #   e_qi^{k+1} = H_Q e_u^{k+1} + lambda_qi e_qi^k
            # In the original space:
            #   e_u^{k+1} = G_w^{-1/2} V_r ê_u^{k+1}
            # So in mixed space (ê_u, e_qi):
            #   H_Q @ G_w^{-1/2} V_r  maps ê_u back to measurement space
            H_Q_hat = H_Q @ (gw_inv_sqrt[:, None] * V_r)  # (n_qi, r)
            # Note: G_w^{-1/2} V_r because ê_u = G_w^{1/2} e_u,
            #       so e_u = G_w^{-1/2} V_r ê_u

            A_uu_hat = np.diag(1.0 - alpha * lam_r)  # (r, r)

            # Augmented matrix in (ê_u, e_qi) space — (r + n_qi) × (r + n_qi)
            A_aug = np.block([
                [A_uu_hat, A_uq_hat],
                [H_Q_hat @ A_uu_hat, H_Q_hat @ A_uq_hat + lambda_qi * np.eye(n_qi)],
            ])

            eig_aug = np.linalg.eigvals(A_aug)
            augmented_rho = float(np.max(np.abs(eig_aug)))
        else:
            augmented_rho = lambda_qi

        if tso_period_s is not None and dso_period_s is not None:
            ratio = tso_period_s / dso_period_s
            augmented_cascade_margin = 1.0 - augmented_rho ** ratio

        if augmented_rho >= 1.0:
            warnings.append(
                f'{layer_name}: augmented PI spectral radius rho_aug = {augmented_rho:.4f} >= 1.0. '
                f'Integral Q-tracking destabilises the DSO.  '
                f'Reduce g_qi ({g_qi}) or lam_qi ({lambda_qi}).'
            )
        elif augmented_rho > 0.95:
            warnings.append(
                f'{layer_name}: augmented PI spectral radius rho_aug = {augmented_rho:.4f} '
                f'(near boundary).  Integral Q-tracking slows convergence.'
            )

    stable = (alpha <= alpha_max) and all(m >= 0.0 for m in actuator_margins.values())
    marginal = stable and (
        any(
            0.0 <= m < 0.2 * actuator_thresholds[name]
            for name, m in actuator_margins.items()
            if actuator_thresholds[name] > 0.0
        )
        or (alpha_max < np.inf and alpha > 0.8 * alpha_max)
    )

    # If augmented analysis shows instability, override
    if augmented_rho is not None and augmented_rho >= 1.0:
        stable = False

    return ControllerStabilityResult(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        condition_number=condition_number,
        L_eff=L_eff,
        alpha_max=alpha_max,
        actuator_margins=actuator_margins,
        actuator_thresholds=actuator_thresholds,
        eigenvalue_diagnostics=eigenvalue_diagnostics,
        dso_convergence_rate=dso_convergence_rate,
        dso_cascade_margin=dso_cascade_margin,
        augmented_rho=augmented_rho,
        augmented_cascade_margin=augmented_cascade_margin,
        stable=stable,
        marginal=marginal,
        warnings=warnings,
    )


def analyse_stability(
    config: CascadeConfig,
    H_tso: NDArray[np.float64],
    H_dso: NDArray[np.float64],
    n_tso_der: int = 0,
    n_tso_pcc: int = 0,
    n_tso_gen: int = 0,
    n_tso_oltc: int = 0,
    n_tso_shunt: int = 0,
    n_dso_der: int = 0,
    n_dso_oltc: int = 0,
    n_dso_shunt: int = 0,
    # ── Output row counts for building Q_obj ──────────────────────────────
    n_tso_v_out: int = 0,
    n_tso_i_out: int = 0,
    n_dso_q_out: int = 0,
    n_dso_v_out: int = 0,
    n_dso_i_out: int = 0,
    verbose: bool = True,
) -> CascadeStabilityResult:
    """Compute and report stability bounds for the full cascaded OFO controller.

    The actuator counts must match the column dimension of H_TSO and H_DSO
    respectively:  n_u_tso = n_tso_der + n_tso_pcc + n_tso_gen + n_tso_oltc + n_tso_shunt
    and           n_u_dso = n_dso_der + n_dso_oltc + n_dso_shunt.

    The output row counts determine how the per-output objective weight vector
    Q_obj is constructed:

    - **TSO**: H rows = [V_bus | I_line].
      Q_obj = [g_v × n_tso_v_out, 0 × n_tso_i_out].
    - **DSO**: H rows = [Q_interface | V_bus | I_line].
      Q_obj = [g_q × n_dso_q_out, dso_g_v × n_dso_v_out, 0 × n_dso_i_out].

    If output row counts are all zero (legacy mode), a scalar fallback
    Q_obj = g_obj · I is used (equivalent to the old behaviour).

    Parameters
    ----------
    config :
        Populated CascadeConfig instance.
    H_tso, H_dso :
        Sensitivity matrices ∂y/∂u for TSO and DSO respectively.
    n_tso_der .. n_dso_shunt :
        Actuator counts (columns of H).
    n_tso_v_out, n_tso_i_out :
        TSO output row counts: voltage buses, current lines.
    n_dso_q_out, n_dso_v_out, n_dso_i_out :
        DSO output row counts: Q-interface, voltage buses, current lines.
    verbose :
        If True, print a formatted stability report to stdout.
    """
    n_u_tso = H_tso.shape[1]
    n_u_dso = H_dso.shape[1]

    # ── Build TSO g_w / g_u vectors ───────────────────────────────────────────────
    if n_tso_der + n_tso_pcc + n_tso_gen + n_tso_oltc + n_tso_shunt == 0:
        gw_tso = np.full(n_u_tso, config.gw_tso_q_der)
        gu_tso = np.zeros(n_u_tso)
        tso_actuator_names = [f'u_tso_{i}' for i in range(n_u_tso)]
    else:
        gw_tso = config.build_gw_tso_diag(
            n_tso_der, n_tso_pcc, n_tso_gen, n_tso_oltc, n_tso_shunt
        )
        gu_tso = config.build_gu_tso(
            n_tso_der, n_tso_pcc, n_tso_gen, n_tso_oltc, n_tso_shunt
        )
        tso_actuator_names = (
            [f'Q_DER_TS_{i}' for i in range(n_tso_der)] +
            [f'Q_PCC_{i}'    for i in range(n_tso_pcc)] +
            [f'V_gen_{i}'    for i in range(n_tso_gen)] +
            [f'OLTC_{i}'     for i in range(n_tso_oltc)] +
            [f'Shunt_TS_{i}' for i in range(n_tso_shunt)]
        )
        if len(gw_tso) != n_u_tso:
            raise ValueError(
                f'TSO actuator counts sum to {len(gw_tso)}, '
                f'but H_tso has {n_u_tso} columns.'
            )

    # ── Build DSO g_w / g_u vectors ───────────────────────────────────────────────
    if n_dso_der + n_dso_oltc + n_dso_shunt == 0:
        gw_dso = np.full(n_u_dso, config.gw_dso_q_der)
        gu_dso = np.zeros(n_u_dso)
        dso_actuator_names = [f'u_dso_{i}' for i in range(n_u_dso)]
    else:
        gw_dso = config.build_gw_dso_diag(n_dso_der, n_dso_oltc, n_dso_shunt)
        gu_dso = config.build_gu_dso(n_dso_der, n_dso_oltc, n_dso_shunt)
        dso_actuator_names = (
            [f'Q_DER_DN_{i}' for i in range(n_dso_der)] +
            [f'OLTC_DN_{i}'  for i in range(n_dso_oltc)] +
            [f'Shunt_DN_{i}' for i in range(n_dso_shunt)]
        )
        if len(gw_dso) != n_u_dso:
            raise ValueError(
                f'DSO actuator counts sum to {len(gw_dso)}, '
                f'but H_dso has {n_u_dso} columns.'
            )

    # ── Build per-output Q_obj vectors ────────────────────────────────────────────
    # TSO: H rows = [V_bus | I_line], objective = g_v · ||V − V_set||²
    if n_tso_v_out + n_tso_i_out > 0:
        q_obj_tso = _build_q_obj(
            n_q=0, n_v=n_tso_v_out, n_i=n_tso_i_out,
            g_q=0.0, g_v=config.g_v,
        )
    else:
        # Legacy fallback: uniform weight
        q_obj_tso = np.full(H_tso.shape[0], config.g_v)

    # DSO: H rows = [Q_interface | V_bus | I_line],
    #       objective = g_q · ||Q − Q_set||² + dso_g_v · ||V − V_set||²
    if n_dso_q_out + n_dso_v_out + n_dso_i_out > 0:
        q_obj_dso = _build_q_obj(
            n_q=n_dso_q_out, n_v=n_dso_v_out, n_i=n_dso_i_out,
            g_q=config.g_q, g_v=config.dso_g_v,
        )
    else:
        # Legacy fallback: uniform weight
        q_obj_dso = np.full(H_dso.shape[0], config.g_q)

    # ── Analyse each layer ────────────────────────────────────────────────────────
    tso_result = _analyse_layer(
        layer_name='TSO',
        H=H_tso,
        alpha=config.alpha,
        q_obj_diag=q_obj_tso,
        gw_vector=gw_tso,
        actuator_names=tso_actuator_names,
        gu_vector=gu_tso,
    )

    tso_period_s = config.effective_tso_period_s
    dso_period_s = config.effective_dso_period_s

    dso_result = _analyse_layer(
        layer_name='DSO',
        H=H_dso,
        alpha=config.alpha,
        q_obj_diag=q_obj_dso,
        gw_vector=gw_dso,
        actuator_names=dso_actuator_names,
        gu_vector=gu_dso,
        tso_period_s=tso_period_s,
        dso_period_s=dso_period_s,
        # Integral Q-tracking (augmented PI dynamics)
        n_q_integral=n_dso_q_out,
        g_qi=config.g_qi,
        lambda_qi=config.lambda_qi,
    )

    cascade_stable = tso_result.stable and dso_result.stable
    if dso_result.dso_cascade_margin is not None:
        cascade_stable = cascade_stable and (dso_result.dso_cascade_margin > 0.0)

    # ── Build summary string ──────────────────────────────────────────────────────
    status = 'STABLE' if cascade_stable else 'UNSTABLE'
    if cascade_stable and (tso_result.marginal or dso_result.marginal):
        status = 'MARGINAL'
    summary = (
        f'Cascade stability: {status}.  '
        f'TSO alpha_max = {tso_result.alpha_max:.3g} '
        f'(L_eff = {tso_result.L_eff:.3g}),  '
        f'DSO alpha_max = {dso_result.alpha_max:.3g} '
        f'(L_eff = {dso_result.L_eff:.3g}).  '
        f'Current alpha = {config.alpha:.3g}.'
    )
    if dso_result.dso_cascade_margin is not None:
        ratio = tso_period_s / dso_period_s
        ratio_int = int(ratio)
        summary += (
            f'  DSO rho_D = {dso_result.dso_convergence_rate:.4f}, '
            f'cascade margin = {dso_result.dso_cascade_margin:.4f}.'
        )
    if dso_result.augmented_rho is not None:
        summary += f'  rho_aug(PI) = {dso_result.augmented_rho:.4f}.'

    result = CascadeStabilityResult(
        tso=tso_result,
        dso=dso_result,
        cascade_stable=cascade_stable,
        summary=summary,
    )

    if verbose:
        _print_report(config, result, q_obj_tso, q_obj_dso)

    return result


# ─── Formatted report ──────────────────────────────────────────────────────────

def _print_report(
    config: CascadeConfig,
    result: CascadeStabilityResult,
    q_obj_tso: NDArray[np.float64],
    q_obj_dso: NDArray[np.float64],
) -> None:
    """Print a formatted stability report to stdout."""
    sep  = '=' * 72
    thin = '-' * 72
    print(sep)
    print('  Cascaded OFO Stability Analysis Report')
    print('  (Hauswirth contraction condition,  alpha lam_max(M) < 2)')
    print(sep)
    tso_s = config.effective_tso_period_s
    dso_s = config.effective_dso_period_s
    def _fmt(s: float) -> str:
        return f'{int(s // 60)} min' if s >= 60 and s % 60 == 0 else f'{s:.1f} s'
    print(f'  alpha = {config.alpha}   T_TSO = {_fmt(tso_s)}   '
          f'T_DSO = {_fmt(dso_s)}')
    print()

    for layer_name, r, q_obj in (
        ('TSO', result.tso, q_obj_tso),
        ('DSO', result.dso, q_obj_dso),
    ):
        print(thin)
        # Show active output weights
        unique_w = sorted(set(q_obj[q_obj > 0]))
        w_str = ', '.join(f'{w:.4g}' for w in unique_w) if unique_w else 'none'
        n_active = int(np.sum(q_obj > 0))
        n_total = len(q_obj)
        print(f'  {layer_name} Layer   '
              f'(Q_obj: {n_active}/{n_total} active rows, weights: {w_str})')
        print(thin)

        print(f'  Raw SVD of H     : sigma_max = {r.sigma_max:.4g}, '
              f'sigma_min = {r.sigma_min:.4g}, kappa = {r.condition_number:.3e}')
        print(f'  Curvature        : L_eff = lam_max(H^T Q H) = {r.L_eff:.4g}')
        print(f'  alpha_max        : 2/lam_max(M) = {r.alpha_max:.4g}  '
              f'(M = G_w^{{-1/2}} C G_w^{{-1/2}})')
        alpha_margin = r.alpha_max - config.alpha
        print(f'  alpha margin     : {alpha_margin:.4g}  '
              f'({"OK" if config.alpha <= r.alpha_max else "EXCEEDED"})')
        print()

        print(f'  Per-actuator margins  (condition: g_w > alpha C_ii / 2):')
        for name, margin in r.actuator_margins.items():
            threshold = r.actuator_thresholds[name]
            flag = 'OK' if margin >= 0 else 'X'
            print(f'    {flag}  {name:<22s}  threshold = {threshold:>8.4g}  '
                  f'margin = {margin:+.4g}')
        print()

        # ── Eigenvalue diagnostics ────────────────────────────────────────────
        if r.eigenvalue_diagnostics:
            # Separate top modes from slowest-active modes
            top_modes = [m for m in r.eigenvalue_diagnostics
                         if not m.get('_slowest_active', False)]
            slow_modes = [m for m in r.eigenvalue_diagnostics
                          if m.get('_slowest_active', False)]

            def _print_mode_line(k_label, mode):
                lam = mode['eigenvalue']
                contraction = mode['contraction']
                alpha_lam = config.alpha * lam
                tc = mode['type_contribution']
                parts = sorted(tc.items(), key=lambda x: -x[1])
                parts_str = '  '.join(
                    f'{name}: {100*w:.0f}%' for name, w in parts if w >= 0.01
                )
                flag = ' << slow' if contraction >= 0.95 else (
                    ' << OVER' if alpha_lam > 2.0 else '')
                print(f'  {k_label:>6s}   {lam:>10.4g}   {alpha_lam:>8.4g}   '
                      f'{contraction:>8.4f}   {parts_str}{flag}')

            print(f'  Eigenvalue diagnostics  (M = G_w^{{-1/2}} C G_w^{{-1/2}}):')
            print(f'  {"mode":>6s}   {"lam(M)":>10s}   {"alpha*lam":>8s}   {"|1-a*l|":>8s}   '
                  f'actuator-type participation')

            for k, mode in enumerate(top_modes):
                _print_mode_line(str(k + 1), mode)

            if slow_modes:
                print(f'  {"":>6s}   {"...":>10s}')
                for k, mode in enumerate(slow_modes):
                    _print_mode_line(f'slow', mode)

            print()

        if r.dso_convergence_rate is not None:
            ratio = int(tso_s / dso_s)
            print(f'  Spectral contraction rate  rho_D     = {r.dso_convergence_rate:.4f}  '
                  f'(memoryless, active modes only)')
            print(f'  Cascade margin        1 - rho_D^{ratio}  = {r.dso_cascade_margin:.4f}  '
                  f'({"OK" if r.dso_cascade_margin > 0 else "INSUFFICIENT"})')

            if r.augmented_rho is not None:
                print()
                print(f'  Augmented PI analysis  (g_qi = {config.g_qi}, '
                      f'lam_qi = {config.lambda_qi}):')
                print(f'    rho_aug (u + q_int dynamics) = {r.augmented_rho:.4f}  '
                      f'({"< 1 OK" if r.augmented_rho < 1.0 else ">= 1 UNSTABLE"})')
                if r.augmented_cascade_margin is not None:
                    print(f'    Cascade margin   1 - rho_aug^{ratio} = '
                          f'{r.augmented_cascade_margin:.4f}  '
                          f'({"OK" if r.augmented_cascade_margin > 0 else "INSUFFICIENT"})')
            print()

        if r.warnings:
            print(f'  Warnings:')
            for w in r.warnings:
                print(f'    WARNING: {w}')
            print()

        status = 'STABLE' if r.stable else 'UNSTABLE'
        if r.stable and r.marginal:
            status = 'MARGINAL'
        print(f'  -> {layer_name} status: {status}')
        print()

    print(sep)
    print(f'  OVERALL: {result.summary}')
    print(sep)


# ─── Convenience: minimum g_w recommendation (per actuator type) ──────────────

def recommend_gw_min(
    config: CascadeConfig,
    H: NDArray[np.float64],
    g_obj: float = 0.0,
    safety_factor: float = 2.0,
    q_obj_diag: Optional[NDArray[np.float64]] = None,
) -> float:
    """Return the minimum recommended g_w scalar for a given H.

    Uses the per-actuator self-curvature C_{ii} = (H^T Q_obj H)_{ii}
    and returns safety_factor × α × max_i(C_{ii}) / 2.

    This is a conservative scalar bound over ALL actuator types.
    For per-type analysis use :func:`recommend_gw_min_per_type`.
    """
    if q_obj_diag is None:
        q_obj_diag = np.full(H.shape[0], g_obj)

    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    HQ = q_sqrt[:, None] * H
    C = HQ.T @ HQ
    C_max_diag = float(np.max(np.diag(C)))

    return safety_factor * config.alpha * C_max_diag / 2.0


def recommend_gw_min_per_type(
    config: CascadeConfig,
    H: NDArray[np.float64],
    q_obj_diag: NDArray[np.float64],
    actuator_blocks: list[tuple[str, int, float]],
    safety_factor: float = 2.0,
) -> list[dict]:
    """Return per-actuator-type minimum g_w recommendations.

    For each actuator type, the worst-case (maximum) diagonal entry of
    C = H^T Q_obj H within that type's column block is used:

        g_w_min[type] = safety_factor × α × max_{i in block}(C_{ii}) / 2

    This is compared against the actual g_w value configured for that type,
    so the output is meaningful and actionable.

    Parameters
    ----------
    config :
        CascadeConfig providing α.
    H :
        Sensitivity matrix (n_y × n_u).
    q_obj_diag :
        Per-output objective weight vector (length n_y).
    actuator_blocks :
        Ordered list of (type_name, count, gw_actual) tuples matching the
        column order of H.  count = 0 blocks are silently skipped.
        Example for TSO:
            [('Q_DER_TS', n_tso_der,   config.gw_tso_q_der),
             ('V_gen',    n_tso_gen,    config.gw_tso_v_gen),
             ('Shunt_TS', n_tso_shunt,  config.gw_tso_shunt)]
    safety_factor :
        Multiplier on the theoretical bound. Default 2.0.

    Returns
    -------
    List of dicts, one per non-empty block, with keys:
        'type'       : str   — actuator type name
        'gw_actual'  : float — configured g_w value
        'gw_min'     : float — required minimum (with safety factor)
        'margin'     : float — gw_actual − gw_min  (positive = safe)
        'margin_rel' : float — margin / gw_min  (relative margin)
        'ok'         : bool  — True iff gw_actual ≥ gw_min
    """
    # Verify that block counts sum to n_u.
    total = sum(count for _, count, _ in actuator_blocks)
    if total != H.shape[1]:
        raise ValueError(
            f'Sum of actuator_blocks counts ({total}) does not match '
            f'H.shape[1] = {H.shape[1]}.'
        )

    q_sqrt = np.sqrt(np.maximum(q_obj_diag, 0.0))
    HQ = q_sqrt[:, None] * H          # (n_y, n_u)
    C = HQ.T @ HQ                     # (n_u, n_u) symmetric PSD
    C_diag = np.diag(C)               # (n_u,)

    results = []
    col = 0
    for type_name, count, gw_actual in actuator_blocks:
        if count == 0:
            col += count
            continue

        # Maximum self-curvature within this actuator type's column block.
        C_block_max = float(np.max(C_diag[col : col + count]))
        gw_min = safety_factor * config.alpha * C_block_max / 2.0
        margin = gw_actual - gw_min
        margin_rel = margin / gw_min if gw_min > 0.0 else np.inf

        results.append({
            'type':       type_name,
            'gw_actual':  gw_actual,
            'gw_min':     gw_min,
            'margin':     margin,
            'margin_rel': margin_rel,
            'ok':         gw_actual >= gw_min,
        })
        col += count

    return results


# =============================================================================
#  Multi-Zone (Multi-TSO) Stability Analysis
# =============================================================================
#
# Extension of the single-zone analysis above to an N-zone system.
#
# Theory reference: Schwenke / CIGRE 2026 — "Multi-TSO-DSO OFO"
#
# For N independent TSO zones, each with its own OFO iteration, the overall
# system converges if the full block system matrix M_sys satisfies:
#
#   ρ(I − α_eff M_sys) < 1
#
# A sufficient (conservative) condition based on diagonal dominance is:
#
#   ∀ i:   0 < α_i · (λ_max(M_TSO,ii) + Σ_{j≠i} ||M_TSO,ij||₂)  < 2
#
# The global eigenvalue condition ρ(M_sys) is also computed, giving a tighter
# (necessary and sufficient) stability bound.
#
# See MultiTSOCoordinator.check_contraction() for runtime diagnostics during
# the simulation.  This module provides an offline, post-hoc analysis from
# precomputed H blocks.


@dataclass
class ZoneStabilityResult:
    """
    Stability analysis result for one zone in a multi-zone system.

    Attributes
    ----------
    zone_id : int
        Integer zone label.
    n_controls : int
        Number of control variables in this zone.
    n_outputs : int
        Number of output variables (rows in H_ii).
    sigma_max_Hii : float
        Largest singular value of the local sensitivity matrix H_ii.
    lambda_max_Mii : float
        Largest eigenvalue of the local preconditioned curvature M_TSO,ii.
    alpha_max_local : float
        Single-zone stability bound: 2 / λ_max(M_ii).
        This is the SAME as the existing α_max in ControllerStabilityResult.
    coupling_norms : Dict[int, float]
        ||M_TSO,ij||₂ for each coupling zone j ≠ i.
    coupling_sum : float
        Σ_{j≠i} ||M_TSO,ij||₂.  If large relative to λ_max(M_ii), the
        inter-zone coupling dominates and the system may diverge even if each
        zone is locally stable.
    contraction_lhs : float
        α_i · (λ_max(M_ii) + coupling_sum).  Must lie in (0, 2) for stability.
    alpha_max_coupled : float
        2 / (λ_max(M_ii) + coupling_sum) — tighter bound than alpha_max_local.
    diagonally_dominant : bool
        True iff contraction_lhs ∈ (0, 2).
    warnings : List[str]
        Human-readable warnings.
    """
    zone_id:            int
    n_controls:         int
    n_outputs:          int
    sigma_max_Hii:      float
    lambda_max_Mii:     float
    alpha_max_local:    float
    coupling_norms:     Dict[int, float]
    coupling_sum:       float
    contraction_lhs:    float
    alpha_max_coupled:  float
    diagonally_dominant: bool
    warnings:           list = field(default_factory=list)
    eigenvalue_diagnostics: list = field(default_factory=list)

    # ── Lyapunov / small-gain analysis (populated by analyse_multi_zone_stability) ──
    n_effective:        int = 0
    """Number of effective (non-null) eigenvalues of M_ii."""

    n_null:             int = 0
    """Number of null-space eigenvalues (co-located DERs, inactive actuators)."""

    lambda_min_Mii:     float = 0.0
    """Smallest *effective* eigenvalue of M_ii (after null-space filtering)."""

    kappa_Mii:          float = np.inf
    """Condition number κ(M_ii) = λ_max / λ_min."""

    rho_i:              float = 1.0
    """Per-zone contraction rate ρ_i = max_ℓ |1 − α_i λ_ℓ(M_ii)|."""

    rho_i_opt:          float = 1.0
    """Optimal contraction rate at α_i* = 2 / (λ_min + λ_max): ρ* = (κ−1)/(κ+1)."""

    alpha_i_opt:        float = 0.0
    """Optimal step size α_i* = 2 / (λ_min(M_ii) + λ_max(M_ii))."""

    sigma_ij:           Dict[int, float] = field(default_factory=dict)
    """Per-zone cross-coupling gains σ_ij = α_i · ‖M_ij‖₂."""

    sigma_ij_opt:       Dict[int, float] = field(default_factory=dict)
    """Cross-coupling gains at optimal α_i*: σ*_ij = α_i* · ‖M_ij‖₂."""

    lyapunov_row_sum:   float = 1.0
    """ρ_i + Σ_{j≠i} σ_ij.  Must be < 1 for N-zone small-gain (uniform weights)."""

    lyapunov_row_sum_opt: float = 1.0
    """ρ*_i + Σ_{j≠i} σ*_ij at optimal step size."""


@dataclass
class MultiZoneStabilityResult:
    """
    Top-level result container for the multi-zone (multi-TSO) stability analysis.

    Attributes
    ----------
    zones : List[ZoneStabilityResult]
        Per-zone results, ordered by zone_id.
    M_sys : NDArray
        Full assembled block system matrix of shape (Σ n_u_i, Σ n_u_i).
        This is the M_sys = [[M_TSO,ij]] block matrix.  NOTE: M_sys is
        generally NOT symmetric because C_ij = H_ii^T Q_i H_ij sums over
        zone i's observation buses, while C_ji^T sums over zone j's.
    M_sys_eigenvalues : NDArray
        Eigenvalues of M_sys (may be complex for non-symmetric M_sys;
        stored as complex128, use .real for the real parts).
    M_sys_lambda_max : float
        Stability-relevant metric: max Re(λ) of M_sys eigenvalues.
    alpha_max_global : float
        Global stability bound: 2 / λ_max(M_sys).
        This is the TIGHTEST stability bound — tighter than the per-zone
        diagonal-dominance bound.
    globally_stable : bool
        True iff α_i · λ_max(M_sys) < 2 for all i (here all α_i are used).
    all_zones_diagonally_dominant : bool
        True iff every zone satisfies the diagonal-dominance condition.
    summary : str
        Human-readable one-paragraph summary.
    """
    zones:                      List[ZoneStabilityResult]
    M_sys:                      NDArray[np.float64]
    M_sys_eigenvalues:          NDArray[np.float64]
    M_sys_lambda_max:           float
    alpha_max_global:           float
    globally_stable:            bool
    all_zones_diagonally_dominant: bool
    summary:                    str = ""

    # ── Lyapunov / small-gain analysis ───────────────────────────────────────
    small_gain_gamma:           float = np.inf
    """Contraction rate γ = max_i {ρ_i + Σ_{j≠i} σ_ij} (uniform weights).
    System converges iff γ < 1."""

    small_gain_gamma_opt:       float = np.inf
    """Contraction rate γ at optimal per-zone step sizes α_i*."""

    small_gain_stable:          bool = False
    """True iff γ < 1 with current step sizes."""

    small_gain_stable_opt:      bool = False
    """True iff γ < 1 would hold at optimal step sizes."""

    pairwise_small_gain:        Dict[Tuple[int, int], bool] = field(default_factory=dict)
    """For each ordered pair (i,j): True iff σ_ij·σ_ji < (1−ρ_i)(1−ρ_j)."""

    pairwise_small_gain_opt:    Dict[Tuple[int, int], bool] = field(default_factory=dict)
    """Same check at optimal step sizes."""

    recommendations:            List[str] = field(default_factory=list)
    """Actionable tuning recommendations."""

    M_sys_asymmetry:            float = 0.0
    """Relative Frobenius-norm asymmetry ‖M−M^T‖_F / ‖M‖_F.
    Values > 0.01 indicate eigvalsh is unreliable."""

    M_sys_spectral_radius:      float = 0.0
    """Spectral radius ρ(I − M_sys) = max|1 − λ_i(M_sys)| using general
    (non-symmetric) eigenvalues.  Must be < 1 for convergence."""

    M_sys_has_complex_eigenvalues: bool = False
    """True if M_sys has eigenvalues with |imag| > 1e-8."""


def analyse_multi_zone_stability(
    H_blocks:    Dict[Tuple[int, int], NDArray[np.float64]],
    Q_obj_list:  List[NDArray[np.float64]],
    G_w_list:    List[NDArray[np.float64]],
    *,
    zone_ids:    Optional[List[int]] = None,
    zone_names:  Optional[List[str]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
    verbose:     bool = True,
) -> "MultiZoneStabilityResult":
    """
    Offline stability analysis for a multi-zone TSO-DSO OFO system.

    This function takes pre-computed sensitivity blocks H_ij and tuning
    parameters and produces a full stability assessment including:

    * Per-zone local stability (same analysis as the single-zone case).
    * Cross-zone coupling norms ||M_TSO,ij||₂.
    * Per-zone diagonal-dominance check.
    * Global system-matrix M_sys eigenvalue analysis (necessary & sufficient).

    Mathematical background
    -----------------------
    Local curvature (zone i):
        C_ii = H_ii^T  Q_obj,i  H_ii

    Cross curvature (zone i observing zone j's control effect):
        C_ij = H_ii^T  Q_obj,i  H_ij          (H_ii^T on the LEFT → i's own Jacobian)

    Preconditioned local block:
        M_TSO,ii = G_w,i^{-½} C_ii G_w,i^{-½}  ∈ ℝ^{n_u_i × n_u_i}

    Preconditioned coupling block:
        M_TSO,ij = G_w,i^{-½} C_ij G_w,j^{-½}  ∈ ℝ^{n_u_i × n_u_j}

    Full system matrix:
        M_sys = [[M_TSO,ij]]                    ∈ ℝ^{Σn_u × Σn_u}

    Sufficient stability condition (diagonal dominance, alpha=1):
        (λ_max(M_ii) + Σ_{j≠i} ||M_ij||₂) < 2   ∀ i

    Tighter (necessary & sufficient) condition:
        λ_max(M_sys) < 2

    Parameters
    ----------
    H_blocks : Dict[(i, j), NDArray]
        Sensitivity blocks.  (i, j) → H_ij of shape (n_y_i, n_u_j).
        Must contain at least all diagonal blocks (i, i).
    Q_obj_list : List[NDArray]
        Per-zone Q_obj diagonal vectors (length n_y_i each).
        Row ordering assumed: [V_bus | I_line] (I rows get weight 0).
    G_w_list : List[NDArray]
        Per-zone G_w diagonal vectors (length n_u_i each).
        Column ordering: [Q_DER | Q_PCC | V_gen | OLTC | shunt].
    zone_ids : List[int], optional
        Zone IDs in the same order as Q_obj_list and G_w_list.
        Defaults to [0, 1, 2, …].
    zone_names : List[str], optional
        Human-readable zone labels for the report.
    verbose : bool
        If True, print a formatted report to stdout.

    Returns
    -------
    MultiZoneStabilityResult
    """
    n_zones = len(Q_obj_list)
    if zone_ids is None:
        zone_ids = list(range(n_zones))
    if zone_names is None:
        zone_names = [f"Zone {z}" for z in zone_ids]

    # ── Step 1: Build M blocks ────────────────────────────────────────────────
    #
    # For each (i, j) pair: C_ij = H_ii^T Q_obj,i H_ij → M_ij = G_w,i^{-½} C_ij G_w,j^{-½}
    M_blocks: Dict[Tuple[int, int], NDArray[np.float64]] = {}

    for i_idx, i in enumerate(zone_ids):
        q_obj_i = Q_obj_list[i_idx]
        gw_i    = G_w_list[i_idx]
        gw_i_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_i, 1e-12))

        H_ii = H_blocks.get((i, i))
        if H_ii is None:
            raise ValueError(
                f"Diagonal block H_({i},{i}) is required but not found in H_blocks."
            )
        # Q^{1/2}-weighted H_ii (used for all C_ij in this row i)
        q_sqrt_i = np.sqrt(np.maximum(q_obj_i, 0.0))
        QH_ii = q_sqrt_i[:, None] * H_ii    # (n_y_i, n_u_i)

        for j_idx, j in enumerate(zone_ids):
            gw_j = G_w_list[j_idx]
            gw_j_inv_sqrt = 1.0 / np.sqrt(np.maximum(gw_j, 1e-12))

            H_ij = H_blocks.get((i, j))
            if H_ij is None:
                # Missing off-diagonal block → treat as zero (no coupling)
                n_u_j = len(gw_j)
                n_u_i = len(gw_i)
                M_blocks[(i, j)] = np.zeros((n_u_i, n_u_j))
                continue

            # C_ij = H_ii^T Q_obj,i H_ij  (efficient via Q^{1/2} H)
            QH_ij = q_sqrt_i[:, None] * H_ij   # (n_y_i, n_u_j)
            C_ij = QH_ii.T @ QH_ij             # (n_u_i, n_u_j)

            # M_ij = G_w,i^{-½} C_ij G_w,j^{-½}
            M_ij = (gw_i_inv_sqrt[:, None] * C_ij) * gw_j_inv_sqrt[None, :]
            M_blocks[(i, j)] = M_ij

    # ── Step 2: Per-zone local analysis (reusing _analyse_layer logic) ────────
    zone_results: List[ZoneStabilityResult] = []

    for i_idx, i in enumerate(zone_ids):
        H_ii  = H_blocks[(i, i)]
        M_ii  = M_blocks[(i, i)]
        n_y_i, n_u_i = H_ii.shape

        # SVD of raw H_ii
        sv = np.linalg.svd(H_ii, compute_uv=False)
        sigma_max_Hii = float(sv[0]) if len(sv) > 0 else 0.0

        # Eigenvalues of M_ii
        eig_ii_all = np.linalg.eigvalsh(M_ii)
        lambda_max_all = float(np.maximum(eig_ii_all[-1], 0.0))

        # Filter near-zero eigenvalues (null-space of H^T Q H from co-located
        # DERs or inactive actuators).  These directions are neutrally stable
        # (σ* = 0) and must not dominate the contraction rate.
        #
        # For multi-zone analysis, we distinguish between:
        #   1. Null-space modes (lam < 1e-12 * lam_max) -> ignore
        #   2. Slow modes (1e-12 < lam < 0.01 * lam_max) -> report as 'slow'
        #   3. Active modes (lam > 0.01 * lam_max) -> determine rho_i
        eig_tol_null = 1e-12 * max(lambda_max_all, 1e-14)
        eig_tol_active = 0.01 * max(lambda_max_all, 1e-14)

        eig_ii_eff = eig_ii_all[eig_ii_all > eig_tol_null]
        eig_ii_active = eig_ii_all[eig_ii_all > eig_tol_active]

        n_effective = len(eig_ii_eff)
        n_active = len(eig_ii_active)
        n_null = len(eig_ii_all) - n_effective

        if n_active == 0:
            # Degenerate or only slow modes
            lambda_max = lambda_max_all
            lambda_min = float(eig_ii_eff[0]) if n_effective > 0 else 0.0
            rho_i = 1.0
        else:
            lambda_max = float(eig_ii_active[-1])
            lambda_min = float(eig_ii_active[0])
            # Contraction rate (alpha=1, absorbed into g_w)
            rho_i = max(abs(1.0 - lam) for lam in eig_ii_active)

        if n_effective == 0:
            kappa = 1.0
            alpha_max_local = np.inf
            alpha_i_opt = 1.0
            rho_i_opt = 0.0
        else:
            l_max_eff = float(eig_ii_eff[-1])
            l_min_eff = float(eig_ii_eff[0])
            alpha_max_local = 2.0 / l_max_eff if l_max_eff > 1e-14 else np.inf
            kappa = l_max_eff / l_min_eff if l_min_eff > 1e-14 else np.inf
            alpha_i_opt = 1.0   # alpha removed; kept for backward compat
            rho_i_opt = (kappa - 1.0) / (kappa + 1.0) if kappa < np.inf else 1.0

        # Coupling norms: ||M_ij||₂ for j ≠ i
        coupling_norms: Dict[int, float] = {}
        for j_idx2, j in enumerate(zone_ids):
            if j == i:
                continue
            M_ij = M_blocks.get((i, j), np.zeros((0, 0)))
            if M_ij.size == 0:
                coupling_norms[j] = 0.0
            else:
                coupling_norms[j] = float(np.linalg.norm(M_ij, ord=2))

        coupling_sum = sum(coupling_norms.values())

        # Cross-coupling gains σ_ij = ‖M_ij‖₂ (no alpha)
        sigma_ij_current = {j: norm for j, norm in coupling_norms.items()}
        sigma_ij_optimal = {j: norm for j, norm in coupling_norms.items()}

        # Lyapunov row sums: ρ_i + Σ_{j≠i} σ_ij
        lyap_row = rho_i + sum(sigma_ij_current.values())
        lyap_row_opt = rho_i_opt + sum(sigma_ij_optimal.values())

        # Diagonal-dominance condition: (λ_max + Σ||M_ij||₂) ∈ (0, 2)
        contraction_lhs = lambda_max + coupling_sum
        alpha_max_coupled = (
            2.0 / (lambda_max + coupling_sum)
            if (lambda_max + coupling_sum) > 1e-14 else np.inf
        )
        diag_dom = (contraction_lhs > 0.0) and (contraction_lhs < 2.0)

        warnings: List[str] = []
        if contraction_lhs >= 2.0:
            warnings.append(
                f"{zone_names[i_idx]}: contraction_lhs = {contraction_lhs:.4f} >= 2.0 -- "
                f"diagonal-dominance condition VIOLATED.  "
                f"Increase g_w to bring lambda_max below {alpha_max_coupled:.4g}."
            )
        elif contraction_lhs > 1.5:
            warnings.append(
                f"{zone_names[i_idx]}: contraction_lhs = {contraction_lhs:.4f} -- "
                f"marginal (> 1.5).  Coupling is significant."
            )
        if coupling_sum > lambda_max:
            warnings.append(
                f"{zone_names[i_idx]}: inter-zone coupling dominates "
                f"(sum ||M_ij||_2 = {coupling_sum:.4f} > lambda_max = {lambda_max:.4f}).  "
                f"Adding DERs to Zone {i} strengthens the diagonal block."
            )

        # ── Per-zone eigenvalue diagnostics ──────────────────────────────────
        # We reuse the mode-building logic to show actuator participation per zone.
        # This helps identify why rho_i is high.
        ev_diag = []
        if n_u_i > 0:
            # Re-run eigh to get vectors
            Mi_eig, Mi_vecs = np.linalg.eigh(M_ii)
            
            # Map column indices to names for diagnostics
            # Note: order matches G_w_list construction
            ac = actuator_counts[i_idx] if actuator_counts else {}
            a_names = (
                [f'Q_DER_{k}' for k in range(ac.get('n_der', 0))] +
                [f'Q_PCC_{k}' for k in range(ac.get('n_pcc', 0))] +
                [f'V_gen_{k}' for k in range(ac.get('n_gen', 0))] +
                [f'OLTC_{k}'  for k in range(ac.get('n_oltc', 0))] +
                [f'Shunt_{k}' for k in range(ac.get('n_shunt', 0))]
            )
            # Pad names if counts missing
            if len(a_names) < n_u_i:
                a_names += [f'u_{k}' for k in range(len(a_names), n_u_i)]

            def _build_mode(idx):
                lam = float(Mi_eig[idx])
                if lam < 1e-14: return None
                v_sq = Mi_vecs[:, idx]**2
                tc = {}
                for k, name in enumerate(a_names):
                    atype = name.rsplit('_', 1)[0]
                    tc[atype] = tc.get(atype, 0.0) + float(v_sq[k])
                return {
                    'eigenvalue': lam,
                    'contraction': abs(1.0 - lam),
                    'type_contribution': tc,
                    'active': lam > eig_tol_active,
                    '_slowest_active': lam <= eig_tol_active and lam > eig_tol_null
                }

            # Top 3 modes
            for k in range(min(3, n_u_i)):
                m = _build_mode(n_u_i - 1 - k)
                if m: ev_diag.append(m)
            # Slowest active mode
            if n_active > 0:
                first_active_idx = int(np.where(Mi_eig > eig_tol_active)[0][0])
                if n_u_i - 1 - first_active_idx >= 3: # if not already in top 3
                    m = _build_mode(first_active_idx)
                    if m: 
                        m['_slowest_active'] = True
                        ev_diag.append(m)

        zone_results.append(ZoneStabilityResult(
            zone_id=i,
            n_controls=n_u_i,
            n_outputs=n_y_i,
            sigma_max_Hii=sigma_max_Hii,
            lambda_max_Mii=lambda_max,
            alpha_max_local=alpha_max_local,
            coupling_norms=coupling_norms,
            coupling_sum=coupling_sum,
            contraction_lhs=contraction_lhs,
            alpha_max_coupled=alpha_max_coupled,
            diagonally_dominant=diag_dom,
            warnings=warnings,
            eigenvalue_diagnostics=ev_diag,
            # Lyapunov fields
            n_effective=n_effective,
            n_null=n_null,
            lambda_min_Mii=lambda_min,
            kappa_Mii=kappa,
            rho_i=rho_i,
            rho_i_opt=rho_i_opt,
            alpha_i_opt=alpha_i_opt,
            sigma_ij=sigma_ij_current,
            sigma_ij_opt=sigma_ij_optimal,
            lyapunov_row_sum=lyap_row,
            lyapunov_row_sum_opt=lyap_row_opt,
        ))

    # ── Step 3: Assemble full M_sys and global eigenvalue analysis ────────────
    #
    # M_sys = [[M_TSO,ij]] block matrix.
    # Row/column ordering: zone 0 block, zone 1 block, …, zone N-1 block.
    n_per_zone = [len(G_w_list[k]) for k in range(n_zones)]
    n_total = sum(n_per_zone)
    M_sys = np.zeros((n_total, n_total), dtype=np.float64)

    row_offset = 0
    for i_idx, i in enumerate(zone_ids):
        col_offset = 0
        for j_idx, j in enumerate(zone_ids):
            M_ij = M_blocks.get((i, j), np.zeros((n_per_zone[i_idx], n_per_zone[j_idx])))
            r0 = row_offset
            r1 = row_offset + n_per_zone[i_idx]
            c0 = col_offset
            c1 = col_offset + n_per_zone[j_idx]
            # Guard against shape mismatch (e.g. from zero-padded blocks)
            ar = min(M_ij.shape[0], r1 - r0)
            ac = min(M_ij.shape[1], c1 - c0)
            M_sys[r0:r0+ar, c0:c0+ac] = M_ij[:ar, :ac]
            col_offset += n_per_zone[j_idx]
        row_offset += n_per_zone[i_idx]

    # ── M_sys asymmetry diagnostic ──────────────────────────────────────────
    # M_sys is NOT generally symmetric: C_ij = H_ii^T Q_i H_ij sums over
    # zone i's observation buses, while C_ji^T sums over zone j's.  The
    # correct stability criterion is rho(I - M_sys) < 1 using general
    # (non-symmetric) eigenvalues.
    M_norm = float(np.linalg.norm(M_sys, 'fro'))
    M_asym = float(np.linalg.norm(M_sys - M_sys.T, 'fro'))
    M_sys_asymmetry = M_asym / max(M_norm, 1e-14)

    # General eigenvalue computation (handles non-symmetric M_sys).
    sys_eigs_all = np.linalg.eigvals(M_sys)

    # Check for complex eigenvalues (indicates oscillatory coupling modes)
    max_imag = float(np.max(np.abs(sys_eigs_all.imag)))
    has_complex = max_imag > 1e-8

    # Stability metric: spectral radius rho(I - M_sys) = max|1 - lambda_i|
    spectral_radius = float(np.max(np.abs(1.0 - sys_eigs_all)))

    # For lambda_max reporting: use max real part of eigenvalues
    # Filter near-zero eigenvalues (null-space directions)
    sys_eigs_real = sys_eigs_all.real
    sys_eig_tol = 1e-10 * max(float(np.max(np.abs(sys_eigs_real))), 1e-14)
    active_mask = np.abs(sys_eigs_real) > sys_eig_tol
    sys_eigs_active = sys_eigs_all[active_mask]
    M_sys_lambda_max = float(np.max(sys_eigs_real[active_mask])) if np.any(active_mask) else 0.0
    alpha_max_global = (2.0 / M_sys_lambda_max) if M_sys_lambda_max > 1e-14 else np.inf

    # Global stability: rho(I - M_sys) < 1 (true contraction condition)
    globally_stable = (spectral_radius < 1.0)

    all_diag_dom = all(zr.diagonally_dominant for zr in zone_results)

    # ── Step 4: Lyapunov / small-gain post-loop analysis ────────────────────

    # N-zone small-gain condition: γ = max_i {ρ_i + Σ_{j≠i} σ_ij} < 1
    small_gain_gamma = max(zr.lyapunov_row_sum for zr in zone_results)
    small_gain_gamma_opt = max(zr.lyapunov_row_sum_opt for zr in zone_results)
    small_gain_stable = small_gain_gamma < 1.0
    small_gain_stable_opt = small_gain_gamma_opt < 1.0

    # Pairwise small-gain: σ_ij · σ_ji < (1 − ρ_i)(1 − ρ_j)
    pairwise_sg: Dict[Tuple[int, int], bool] = {}
    pairwise_sg_opt: Dict[Tuple[int, int], bool] = {}
    for i_idx, zr_i in enumerate(zone_results):
        for j_idx, zr_j in enumerate(zone_results):
            if i_idx >= j_idx:
                continue
            zi, zj = zr_i.zone_id, zr_j.zone_id
            sig_ij = zr_i.sigma_ij.get(zj, 0.0)
            sig_ji = zr_j.sigma_ij.get(zi, 0.0)
            margin_i = max(1.0 - zr_i.rho_i, 0.0)
            margin_j = max(1.0 - zr_j.rho_i, 0.0)
            pairwise_sg[(zi, zj)] = (sig_ij * sig_ji) < (margin_i * margin_j)

            sig_ij_o = zr_i.sigma_ij_opt.get(zj, 0.0)
            sig_ji_o = zr_j.sigma_ij_opt.get(zi, 0.0)
            margin_i_o = max(1.0 - zr_i.rho_i_opt, 0.0)
            margin_j_o = max(1.0 - zr_j.rho_i_opt, 0.0)
            pairwise_sg_opt[(zi, zj)] = (sig_ij_o * sig_ji_o) < (margin_i_o * margin_j_o)

    # ── Step 5: Generate actionable tuning recommendations ───────────────
    recommendations: List[str] = []

    # Find bottleneck zone (highest Lyapunov row sum)
    worst_idx = int(np.argmax([zr.lyapunov_row_sum for zr in zone_results]))
    worst = zone_results[worst_idx]
    wname = zone_names[worst_idx]

    if not small_gain_stable:
        recommendations.append(
            f"Small-gain condition VIOLATED (gamma={small_gain_gamma:.4f} >= 1). "
            f"Bottleneck: {wname} (row sum = {worst.lyapunov_row_sum:.4f})."
        )

        # Suggest increasing g_w if rho_i is the dominant term
        if worst.rho_i > 0.5 * worst.lyapunov_row_sum:
            recommendations.append(
                f"  -> {wname}: rho_i = {worst.rho_i:.4f} dominates. "
                f"Increase g_w in this zone to improve contraction "
                f"(optimal rho* = {worst.rho_i_opt:.4f})."
            )

        # Suggest increasing g_w for zones with high condition number
        for i_idx, zr in enumerate(zone_results):
            if zr.kappa_Mii > 100:
                recommendations.append(
                    f"  -> {zone_names[i_idx]}: kappa(M_ii) = {zr.kappa_Mii:.1f} is large. "
                    f"Increase g_w for weakly-coupled actuators to reduce condition number. "
                    f"lam_min = {zr.lambda_min_Mii:.4g}, lam_max = {zr.lambda_max_Mii:.4g}."
                )

        # Suggest decoupling for large cross-coupling
        for i_idx, zr in enumerate(zone_results):
            for j, sig in zr.sigma_ij.items():
                if sig > 0.3:
                    recommendations.append(
                        f"  -> sigma_{zr.zone_id},{j} = {sig:.4f} is large. "
                        f"Increase g_w in {zone_names[i_idx]} to reduce ||M_{zr.zone_id},{j}||_2."
                    )

    if small_gain_stable and not small_gain_stable_opt:
        recommendations.append(
            f"System stable at current alpha but NOT at optimal alpha*. "
            f"Current gamma = {small_gain_gamma:.4f}, optimal gamma = {small_gain_gamma_opt:.4f}. "
            f"Cross-coupling limits the benefit of faster step sizes."
        )

    if small_gain_stable_opt and small_gain_stable:
        recommendations.append(
            f"Small-gain condition satisfied (gamma = {small_gain_gamma:.4f}). "
            f"At optimal step sizes: gamma* = {small_gain_gamma_opt:.4f}."
        )
        # (alpha removed: no per-zone step-size recommendations)

    # Pairwise violation recommendations
    for (zi, zj), ok in pairwise_sg.items():
        if not ok:
            recommendations.append(
                f"  -> Pairwise small-gain VIOLATED for zones ({zi}, {zj}). "
                f"Reduce coupling between these zones."
            )

    # ── Asymmetry and complex eigenvalue warnings ──────────────────────────
    if M_sys_asymmetry > 0.01:
        recommendations.append(
            f"  ⚠ M_sys asymmetry = {M_sys_asymmetry:.4f} (> 0.01).  "
            f"Off-diagonal blocks C_ij != C_ji^T.  "
            f"Stability assessment uses general eigenvalues (non-symmetric)."
        )
    if has_complex:
        recommendations.append(
            f"  ⚠ M_sys has complex eigenvalues (max |imag| = {max_imag:.4g}).  "
            f"This indicates oscillatory coupling modes between zones."
        )

    # ── Build summary string ──────────────────────────────────────────────────
    g_status = "STABLE" if globally_stable else "UNSTABLE"
    d_status = "satisfied" if all_diag_dom else "VIOLATED for some zones"
    sg_status = "satisfied" if small_gain_stable else "VIOLATED"
    asym_note = f"  M_sys asymmetry = {M_sys_asymmetry:.4g}." if M_sys_asymmetry > 1e-6 else ""
    rho_note = f"  rho(I-M_sys) = {spectral_radius:.4g}."
    summary = (
        f"Multi-zone stability: {g_status}.  "
        f"lam_max(M_sys) = {M_sys_lambda_max:.4g}, "
        f"rho(I-M_sys) = {spectral_radius:.4g}, "
        f"alpha_max_global = {alpha_max_global:.4g}.  "
        f"Diagonal-dominance condition {d_status}.  "
        f"Small-gain condition {sg_status} (gamma = {small_gain_gamma:.4f}).  "
        f"N_zones = {n_zones}, N_controls_total = {n_total}."
        f"{asym_note}"
    )

    result = MultiZoneStabilityResult(
        zones=zone_results,
        M_sys=M_sys,
        M_sys_eigenvalues=sys_eigs_active,
        M_sys_lambda_max=M_sys_lambda_max,
        alpha_max_global=alpha_max_global,
        globally_stable=globally_stable,
        all_zones_diagonally_dominant=all_diag_dom,
        summary=summary,
        small_gain_gamma=small_gain_gamma,
        small_gain_gamma_opt=small_gain_gamma_opt,
        small_gain_stable=small_gain_stable,
        small_gain_stable_opt=small_gain_stable_opt,
        pairwise_small_gain=pairwise_sg,
        pairwise_small_gain_opt=pairwise_sg_opt,
        recommendations=recommendations,
        M_sys_asymmetry=M_sys_asymmetry,
        M_sys_spectral_radius=spectral_radius,
        M_sys_has_complex_eigenvalues=has_complex,
    )

    if verbose:
        _print_multi_zone_report(result, zone_names,
                                 H_blocks, Q_obj_list, G_w_list,
                                 actuator_counts)

    return result


def _print_multi_zone_report(
    result: "MultiZoneStabilityResult",
    zone_names: List[str],
    H_blocks: Optional[Dict[Tuple[int, int], NDArray[np.float64]]] = None,
    Q_obj_list: Optional[List[NDArray[np.float64]]] = None,
    G_w_list: Optional[List[NDArray[np.float64]]] = None,
    actuator_counts: Optional[List[Dict[str, int]]] = None,
) -> None:
    """Print a compact multi-zone stability summary to stdout.

    The output is intentionally brief (~15-20 lines): one header line, a
    per-zone single-row table, a global-eigenvalue summary line, a
    small-gain summary line, a diagonal-dominance count, and any
    warnings.  The full diagnostic tables (eigenvalue breakdown,
    per-actuator-type contribution, sigma_ij matrix, pairwise small-gain
    detail, tuning recommendations) have been removed -- every value they
    displayed is still available on ``MultiZoneStabilityResult`` if
    programmatic access is needed.
    """
    sep  = "=" * 86
    thin = "-" * 86
    n_zones = len(result.zones)
    spectral_metric = result.M_sys_lambda_max

    print(sep)
    print("  Multi-Zone OFO Stability Summary")
    print(thin)

    # Per-zone compact table: one row per zone with the most load-bearing
    # numbers (alpha, eigenspectrum, coupling, contraction rate, row sum).
    header = (f"  {'Zone':<10s} {'lam_max':>9s} {'kappa':>9s} "
              f"{'Sum||Mij||':>11s} {'rho_i':>7s} {'row_sum':>9s} {'Status':>9s}")
    print(header)
    print(thin)

    n_diag_dom = 0
    for i_idx, zr in enumerate(result.zones):
        if zr.diagonally_dominant:
            n_diag_dom += 1
            status = "OK"
        else:
            status = "VIOLATED"
        kappa_str = "inf" if zr.kappa_Mii >= 1e6 else f"{zr.kappa_Mii:>9.3g}"
        print(
            f"  {zone_names[i_idx]:<10s} "
            f"{zr.lambda_max_Mii:>9.3g} "
            f"{kappa_str:>9s} "
            f"{zr.coupling_sum:>11.4g} "
            f"{zr.rho_i:>7.4f} "
            f"{zr.lyapunov_row_sum:>9.4f} "
            f"{status:>9s}"
        )
    print(thin)

    # Global spectral bound (necessary-and-sufficient)
    global_tag = "STABLE" if result.globally_stable else "VIOLATED"
    print(f"  Global spectral:  lam_max(M_sys) = {result.M_sys_lambda_max:.4g}   "
          f"[{global_tag}]")

    # Small-gain (sufficient only)
    sg_tag = "SATISFIED" if result.small_gain_stable else "VIOLATED"
    print(f"  Small-gain:       gamma = {result.small_gain_gamma:.4f}   "
          f"[{sg_tag}]")

    # Diagonal-dominance count
    print(f"  Diag. dominance:  {n_diag_dom}/{n_zones} zones pass")

    # Warnings (collected across all zones).  The recommendations block is
    # intentionally removed per the compact-report design.
    all_warnings = []
    for i_idx, zr in enumerate(result.zones):
        for w in zr.warnings:
            all_warnings.append(f"[{zone_names[i_idx]}] {w}")
    if all_warnings:
        print(thin)
        for w in all_warnings:
            print(f"  ! {w}")

    print(sep)
