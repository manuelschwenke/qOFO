"""
Stability Analysis for the Cascaded TSO-DSO OFO Controller
===========================================================

This script computes theoretical stability bounds for both the TSO and DSO
OFO controllers from a given :class:`~core.cascade_config.CascadeConfig` and
a pair of pre-computed sensitivity matrices H_TSO and H_DSO.

Theory
------
For the quadratic OFO objective

    f(w) = w^T G_w w  +  ∇f^T w  +  u^T G_u u

the gradient ∇_w f is linear in w, so its Lipschitz constant equals the
largest eigenvalue of the Hessian:

    L = λ_max(2 G_w + 2 H^T G_z H + 2 G_u)

In the practically relevant limit (G_z dominated by the objective weight
g_obj and the sensitivity magnitude σ_max(H)), the stability condition
α < 2/L translates to the per-actuator lower bound

    g_w  >  α · g_obj · σ_max(H)²

For the *cascade*, the DSO must converge within T_T/T_D = 3 iterations, i.e.

    ρ_D³ ≪ 1   where   ρ_D ≈ 1 − α / (g_w_DSO + ε)

This script:
  1. Accepts H_TSO and H_DSO (numpy arrays) and a CascadeConfig.
  2. Computes σ_max, L_eff, and per-actuator stability margins.
  3. Classifies the current config as STABLE / MARGINAL / UNSTABLE.
  4. Prints a detailed report and returns a structured result dict.

Usage
-----
    from analysis.stability_analysis import analyse_stability
    from core.cascade_config import CascadeConfig
    import numpy as np

    config = CascadeConfig()          # or load from JSON
    H_tso = ...                       # shape (n_y_tso, n_u_tso)
    H_dso = ...                       # shape (n_y_dso, n_u_dso)

    result = analyse_stability(config, H_tso, H_dso)

Author: Manuel Schwenke, TU Darmstadt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from core.cascade_config import CascadeConfig


# ─── Result containers ─────────────────────────────────────────────────────────

@dataclass
class ControllerStabilityResult:
    """Stability analysis result for one OFO controller layer."""

    # SVD / Jacobian properties
    sigma_max: float
    """Largest singular value of the sensitivity matrix H."""

    sigma_min: float
    """Smallest singular value of H (0 if H is rank-deficient)."""

    condition_number: float
    """Condition number σ_max / σ_min (inf if σ_min = 0)."""

    # Lipschitz bounds
    L_eff: float
    """Effective Lipschitz constant: 2 · g_obj · σ_max(H)²."""

    alpha_max: float
    """Maximum stable step size: α_max = 2 / L_eff."""

    # Per-actuator margins  (positive → stable, negative → unstable)
    actuator_margins: Dict[str, float] = field(default_factory=dict)
    """Per-actuator stability margin: g_w[i] − (α · L_eff / 2). Positive = safe."""

    # DSO cascade margin (only meaningful for DSO)
    dso_convergence_rate: Optional[float] = None
    """Estimated spectral radius ρ_D ≈ max(0, 1 − α / g_w_min)."""

    dso_cascade_margin: Optional[float] = None
    """1 − ρ_D^(T_T/T_D): positive means DSO converges within T_T/T_D steps."""

    # Classification
    stable: bool = True
    """True iff all actuator margins are positive AND α ≤ α_max."""

    marginal: bool = False
    """True iff stable but some margin < 20 % of the critical threshold."""

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


# ─── Core analysis function ─────────────────────────────────────────────────────

def _analyse_layer(
    layer_name: str,
    H: NDArray[np.float64],
    alpha: float,
    g_obj: float,
    gw_vector: NDArray[np.float64],
    actuator_names: list[str],
    gu_vector: Optional[NDArray[np.float64]] = None,
    tso_period_min: Optional[int] = None,
    dso_period_min: Optional[int] = None,
) -> ControllerStabilityResult:
    """Compute stability bounds for one controller layer.

    Parameters
    ----------
    layer_name:
        Human-readable name, e.g. 'TSO' or 'DSO'.
    H:
        Full sensitivity matrix (n_y × n_u). May include continuous and
        integer columns; the SVD is taken over the entire matrix.
    alpha:
        OFO step-size gain α.
    g_obj:
        The dominant objective weight (g_v for TSO, g_q for DSO).
    gw_vector:
        Per-actuator g_w values (length n_u).
    actuator_names:
        Names matching gw_vector entries (for reporting).
    gu_vector:
        Per-actuator g_u values (optional, defaults to zeros).
    tso_period_min:
        TSO sampling period [min] (required for DSO cascade margin).
    dso_period_min:
        DSO sampling period [min] (required for DSO cascade margin).
    """
    if H.ndim != 2:
        raise ValueError(f'{layer_name}: H must be a 2-D array, got shape {H.shape}.')
    if len(gw_vector) != H.shape[1]:
        raise ValueError(
            f'{layer_name}: gw_vector length {len(gw_vector)} does not match '
            f'H.shape[1] = {H.shape[1]}.'
        )

    if gu_vector is None:
        gu_vector = np.zeros_like(gw_vector)
    if len(gu_vector) != len(gw_vector):
        raise ValueError(f'{layer_name}: gu_vector and gw_vector must have the same length.')

    # ── SVD ──────────────────────────────────────────────────────────────────────
    singular_values = np.linalg.svd(H, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1]) if len(singular_values) > 0 else 0.0
    condition_number = float(sigma_max / sigma_min) if sigma_min > 0.0 else np.inf

    # ── Lipschitz constant and maximum stable step size ───────────────────────────
    # L_eff = 2 · g_obj · σ_max(H)² represents the dominant contribution
    # from the tracking objective term in the Hessian.
    L_eff = 2.0 * g_obj * sigma_max ** 2
    alpha_max = (2.0 / L_eff) if L_eff > 0.0 else np.inf

    # ── Per-actuator stability margins ────────────────────────────────────────────
    # Condition: g_w[i] + g_u[i]  >  α · L_eff / 2
    critical_threshold = alpha * L_eff / 2.0
    actuator_margins: Dict[str, float] = {}
    for name, gw_i, gu_i in zip(actuator_names, gw_vector, gu_vector):
        margin = float(gw_i + gu_i) - critical_threshold
        actuator_margins[name] = margin

    warnings = []

    # ── Step-size check ───────────────────────────────────────────────────────────
    if alpha > alpha_max:
        warnings.append(
            f'{layer_name}: α = {alpha:.4g} exceeds α_max = {alpha_max:.4g} '
            f'(L_eff = {L_eff:.4g}).  Instability likely.'
        )

    # ── Per-actuator warnings ─────────────────────────────────────────────────────
    for idx, (name, margin) in enumerate(actuator_margins.items()):
        if margin < 0.0:
            warnings.append(
                f'{layer_name}: actuator "{name}" has NEGATIVE margin {margin:.4g} '
                f'(g_w + g_u = {float(gw_vector[idx] + gu_vector[idx]):.4g}, '
                f'threshold = {critical_threshold:.4g}).'
            )
        elif critical_threshold > 0.0 and margin < 0.2 * critical_threshold:
            warnings.append(
                f'{layer_name}: actuator "{name}" margin {margin:.4g} is '
                f'< 20 %% of threshold (marginal).'
            )

    # ── Condition number warning ──────────────────────────────────────────────────
    if condition_number > 1e6:
        warnings.append(
            f'{layer_name}: H is ill-conditioned (κ = {condition_number:.3e}). '
            f'Sensitivity accuracy and gradient quality may be degraded.'
        )

    # ── DSO cascade convergence rate ──────────────────────────────────────────────
    dso_convergence_rate = None
    dso_cascade_margin = None
    if tso_period_min is not None and dso_period_min is not None:
        ratio = tso_period_min / dso_period_min  # e.g. 3
        gw_min = float(np.min(gw_vector))  # most conservative (smallest) actuator
        # Upper-bound spectral radius: ρ ≈ max(0, 1 − α / (g_w_min + ε))
        rho = max(0.0, 1.0 - alpha / (gw_min + 1e-12))
        dso_convergence_rate = rho
        dso_cascade_margin = 1.0 - rho ** ratio
        if dso_cascade_margin <= 0.0:
            warnings.append(
                f'{layer_name}: DSO may NOT converge within {ratio:.0f} iterations '
                f'(ρ_D = {rho:.4f}, ρ_D^{ratio:.0f} = {rho**ratio:.4f}). '
                f'Increase α or reduce gw_dso_q_der to ensure quasi-steady-state.'
            )

    stable = (alpha <= alpha_max) and all(m >= 0.0 for m in actuator_margins.values())
    marginal = stable and (
        any(
            0.0 <= m < 0.2 * critical_threshold
            for m in actuator_margins.values()
            if critical_threshold > 0.0
        )
        or (alpha_max < np.inf and alpha > 0.8 * alpha_max)
    )

    return ControllerStabilityResult(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        condition_number=condition_number,
        L_eff=L_eff,
        alpha_max=alpha_max,
        actuator_margins=actuator_margins,
        dso_convergence_rate=dso_convergence_rate,
        dso_cascade_margin=dso_cascade_margin,
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
    verbose: bool = True,
) -> CascadeStabilityResult:
    """Compute and report stability bounds for the full cascaded OFO controller.

    The actuator counts must match the column dimension of H_TSO and H_DSO
    respectively:  n_u_tso = n_tso_der + n_tso_pcc + n_tso_gen + n_tso_oltc + n_tso_shunt
    and           n_u_dso = n_dso_der + n_dso_oltc + n_dso_shunt.

    If all actuator counts are zero (default), uniform g_w vectors are built
    from the scalar g_w defaults in CascadeConfig.

    Parameters
    ----------
    config:
        Populated CascadeConfig instance.
    H_tso:
        TSO sensitivity matrix ∂y_TSO/∂u_TSO, shape (n_y_tso, n_u_tso).
    H_dso:
        DSO sensitivity matrix ∂y_DSO/∂u_DSO, shape (n_y_dso, n_u_dso).
    n_tso_der .. n_dso_shunt:
        Actuator counts for building per-actuator g_w and g_u vectors.
        Must sum to H.shape[1] for each controller.
    verbose:
        If True, print a formatted stability report to stdout.

    Returns
    -------
    CascadeStabilityResult
        Fully populated result with per-layer and cascade-level diagnostics.
    """
    n_u_tso = H_tso.shape[1]
    n_u_dso = H_dso.shape[1]

    # ── Build TSO g_w / g_u vectors ───────────────────────────────────────────────
    if n_tso_der + n_tso_pcc + n_tso_gen + n_tso_oltc + n_tso_shunt == 0:
        # No actuator counts given: use uniform g_w from config scalar default.
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

    # ── Analyse each layer ────────────────────────────────────────────────────────
    tso_result = _analyse_layer(
        layer_name='TSO',
        H=H_tso,
        alpha=config.alpha,
        g_obj=config.g_v,
        gw_vector=gw_tso,
        actuator_names=tso_actuator_names,
        gu_vector=gu_tso,
    )

    dso_result = _analyse_layer(
        layer_name='DSO',
        H=H_dso,
        alpha=config.alpha,
        g_obj=config.g_q,
        gw_vector=gw_dso,
        actuator_names=dso_actuator_names,
        gu_vector=gu_dso,
        tso_period_min=config.tso_period_min,
        dso_period_min=config.dso_period_min,
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
        f'TSO α_max = {tso_result.alpha_max:.3g} (σ_max = {tso_result.sigma_max:.3g}),  '
        f'DSO α_max = {dso_result.alpha_max:.3g} (σ_max = {dso_result.sigma_max:.3g}).  '
        f'Current α = {config.alpha:.3g}.'
    )
    if dso_result.dso_cascade_margin is not None:
        ratio = config.tso_period_min // config.dso_period_min
        summary += (
            f'  DSO cascade margin = {dso_result.dso_cascade_margin:.4f} '
            f'(ρ_D^{ratio} = '
            f'{dso_result.dso_convergence_rate ** ratio:.4f}).'
        )

    result = CascadeStabilityResult(
        tso=tso_result,
        dso=dso_result,
        cascade_stable=cascade_stable,
        summary=summary,
    )

    if verbose:
        _print_report(config, result)

    return result


# ─── Formatted report ──────────────────────────────────────────────────────────

def _print_report(config: CascadeConfig, result: CascadeStabilityResult) -> None:
    """Print a formatted stability report to stdout."""
    sep  = '=' * 72
    thin = '-' * 72
    print(sep)
    print('  Cascaded OFO Stability Analysis Report')
    print(sep)
    print(f'  α = {config.alpha}   T_TSO = {config.tso_period_min} min   '
          f'T_DSO = {config.dso_period_min} min')
    print()

    for layer_name, r, g_obj in (
        ('TSO', result.tso, config.g_v),
        ('DSO', result.dso, config.g_q),
    ):
        print(thin)
        print(f'  {layer_name} Layer   (g_obj = {g_obj})')
        print(thin)
        print(f'  SVD              : σ_max = {r.sigma_max:.4g}, '
              f'σ_min = {r.sigma_min:.4g}, κ = {r.condition_number:.3e}')
        print(f'  L_eff            : {r.L_eff:.4g}')
        print(f'  α_max (= 2/L)    : {r.alpha_max:.4g}')
        alpha_margin = r.alpha_max - config.alpha
        print(f'  α margin         : {alpha_margin:.4g}  '
              f'({"OK" if config.alpha <= r.alpha_max else "EXCEEDED"})')
        print()
        threshold = config.alpha * r.L_eff / 2.0
        print(f'  Per-actuator stability margins  '
              f'(critical threshold = {threshold:.4g}):')
        for name, margin in r.actuator_margins.items():
            flag = '✓' if margin >= 0 else '✗'
            print(f'    {flag}  {name:<22s}  margin = {margin:+.4g}')
        print()
        if r.dso_convergence_rate is not None:
            ratio = config.tso_period_min // config.dso_period_min
            print(f'  DSO convergence rate   ρ_D         = {r.dso_convergence_rate:.4f}')
            print(f'  DSO cascade margin   1 − ρ_D^{ratio}   = {r.dso_cascade_margin:.4f}  '
                  f'({"OK" if r.dso_cascade_margin > 0 else "INSUFFICIENT"})')
            print()
        if r.warnings:
            print(f'  Warnings:')
            for w in r.warnings:
                print(f'    ⚠  {w}')
            print()
        status = 'STABLE' if r.stable else 'UNSTABLE'
        if r.stable and r.marginal:
            status = 'MARGINAL'
        print(f'  → {layer_name} status: {status}')
        print()

    print(sep)
    print(f'  OVERALL: {result.summary}')
    print(sep)


# ─── Convenience: minimum g_w recommendation ──────────────────────────────────

def recommend_gw_min(
    config: CascadeConfig,
    H: NDArray[np.float64],
    g_obj: float,
    safety_factor: float = 2.0,
) -> float:
    """Return the minimum recommended g_w scalar for a given H and g_obj.

    The theoretical lower bound is:

        g_w_min = α · g_obj · σ_max(H)²

    A safety factor ≥ 2 is recommended to ensure aperiodic (non-oscillatory)
    convergence, consistent with the empirical finding gw ≥ 5 in the PSCC paper.

    Parameters
    ----------
    config:
        CascadeConfig providing α.
    H:
        Sensitivity matrix.
    g_obj:
        Dominant objective weight.
    safety_factor:
        Multiplier applied to the theoretical bound.  Default 2.0.

    Returns
    -------
    float
        Recommended minimum g_w value.
    """
    sigma_max = float(np.linalg.svd(H, compute_uv=False)[0])
    g_w_min_theoretical = config.alpha * g_obj * sigma_max ** 2
    return safety_factor * g_w_min_theoretical
