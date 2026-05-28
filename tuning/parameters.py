"""
tuning/parameters.py
====================
Declarative definition of the Bayesian-optimization decision space and
the mapping between BO param dicts and :class:`MultiTSOConfig` instances.

The BO search space is 8-dimensional.  Three integral-Q-tracking
parameters (``dso_g_qi``, ``dso_lambda_qi``, ``dso_q_integral_max_mvar``)
are excluded by user decision (integrator off in this thesis
configuration).

``FIXED_OVERRIDES`` lists fields that are always overwritten when
applying BO params, regardless of their value in the baseline config:
live plots disabled, verbose silenced, integral mode off, etc.  This
guarantees that BO trials are headless and deterministic w.r.t. the
baseline.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from configs.multi_tso_config import MultiTSOConfig
from tuning._types import BOParam, Ceilings


# 9 BO dimensions.
# `g_w_tso_shunt` is conditionally excluded while
# `install_tso_tertiary_shunts=False` in the baseline (no shunt actuators
# → vacuous coordinate). The slot is pinned in FIXED_OVERRIDES so the
# baseline value still flows through `apply_to_config`. Re-enable by
# uncommenting and removing the FIXED_OVERRIDES entry once shunts are
# installed.
#
# `tso_g_q_tie` (added 2026-04-29) is a tracking weight on the Q-tie-line
# rows of the TSO MIQP objective — analogous to `g_q` (Q-PCC tracking),
# scales the curvature ``Q`` block of the iteration matrix rather than
# the proximal ``G_w`` block.  Bounds are chosen from the field's own
# docstring (configs/multi_tso_config.py:294): validated 1.0 starting
# point, aggressive 1e2-1e4, numerically unstable above 1e6.  A
# log-uniform [1e-1, 1e3] range stays inside the validated envelope.
#
# Adaptive `g_w` meta-knobs (`g_w_adapt_beta1`, `g_w_adapt_beta2`,
# `g_w_adapt_t_min`, `g_w_adapt_t_max`, `g_w_adapt_deadband_rel`,
# added 2026-04-29) are NOT in BO_DIMS by default.  When the operator
# enables one or more `adapt_g_w_*` flags in the config, the existing
# `g_w_*` BO dims serve as the *initial* values for the online adapter
# (warm-start), and the meta-knobs take their config defaults.  Add the
# meta-knobs to `BO_DIMS` only after deciding to BO-tune them too —
# typical ranges: β₁, β₂ ∈ [1e-2, 3e-1] log-uniform; t_min ∈ [1e-3, 1]
# log-uniform; t_max ∈ [1, 1e6] log-uniform.  See paper Eq. 16
# (Zagorowska et al., IFAC WC 2026, arXiv:2604.12863).
BO_DIMS: tuple[BOParam, ...] = (
    BOParam("g_v",           log=True, low=1e2, high=1e5), #  high="ceil", fallback_high=1e7),
    BOParam("g_q",           log=True, low=1e-1,  high=1e3),
    BOParam("tso_g_q_tie",   log=True, low=1e-1, high=1e3),
    BOParam("dso_g_v",       log=True, low=1,  high=1e5),
    BOParam("g_w_der",       log=True, low=1e-1, high=1e3), # high="ceil", fallback_high=1e4),
    # `g_w_pcc` upper bound capped at 30 (≈10^1.5) on 2026-05-02:
    # a prior BO run converged to ``g_w_pcc ≈ 269.7`` by exploiting
    # the gameability of ``itae_q_pcc`` (very high ``g_w_pcc`` freezes
    # the PCC setpoint, making the DSO trivially track it).  See the
    # ``CostWeights`` docstring in ``tuning/metrics.py`` for the
    # paired objective-side fix (demoted ``w_q_track`` and new
    # ``w_pcc_underutil`` term).  Values above ~30 are sluggish without
    # a meaningful end-performance benefit.
    BOParam("g_w_pcc",       log=True, low=1e-1, high=100.0),
    BOParam("g_w_tso_oltc",  log=True, low=1, high=1e5), # high="ceil", fallback_high=1e4),
    #BOParam("g_w_tso_shunt", log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_dso_der",   log=True, low=1e-1, high=1e3), # high="ceil", fallback_high=1e4),
    BOParam("g_w_dso_oltc",  log=True, low=1, high=1e5), # high="ceil", fallback_high=1e4),
    # ── Stage-2 (grid-forming + Q(V) local loop) knobs ─────────────────
    # g_w_gridforming is the only stage-2 g_w that actually affects the
    # MIQP step in the current controller setup — it must dominate the
    # V_gf curvature ``g_v · (∂V/∂vm_pu)² ≈ 5·10⁵``.  Below ``1e6`` the
    # TSO V_gf chases DSO transients; above ``1e8`` the converter
    # response slows toward synch-machine timescales.  Search box
    # spans the validated band [1e6, 1e8].
    BOParam("g_w_gridforming", log=True, low=1e6, high=1e8),
    # g_w_dso_der_vref is empirically inert in the current setup
    # (curvature dominates).  Pinned in FIXED_OVERRIDES rather than
    # tuned to avoid wasting BO trials on a vacuous coordinate.
    # See ``tests/diag_stage2_steps.py`` for the proof sweep.
)


# Fields always pinned during tuning (override baseline config).
FIXED_OVERRIDES: dict[str, Any] = {
    # Per-user decision: g_w_gen excluded from stability tuning
    "g_w_gen":                 5e7,

    # Conditionally pinned: shunts not installed at TSO tertiary
    # (`install_tso_tertiary_shunts=False`), so this dim is vacuous.
    # Value matches the 002 baseline; remove this key when shunts are
    # re-installed and the BO_DIMS entry is uncommented.
    "g_w_tso_shunt":           50000.0,

    # Stage-2 knob that's empirically inert (curvature dominates the
    # MIQP step regardless of g_w_dso_der_vref across [1e-2, 1e8]).
    # Pinned at 1.0 to keep BO from wasting trials on a vacuous
    # coordinate.  Promote to BO_DIMS only after a controller change
    # makes it non-redundant (e.g. lowering g_q sufficiently or
    # disabling the K-transform).
    "g_w_dso_der_vref":        1.0,

    # Integral Q-tracking off (user excluded these from BO)
    "dso_g_qi":                0.0,
    "dso_lambda_qi":           0.9,
    "dso_q_integral_max_mvar": 50.0,

    # Structural choices, not tuned
    "dso_gamma_oltc_q":        0.0,
    "int_max_step":            1,
    "int_cooldown":            1,    # USER FIXED 2025-04-27

    # Headless / deterministic
    "verbose":                 0,
    "live_plot_controller":    False,
    "live_plot_cascade":       False,
    "live_plot_system":        False,
    "run_stability_analysis":  False,
}


def resolve_high(param: BOParam, ceilings: Ceilings | None) -> float:
    """Resolve the effective upper bound for one BO param.

    If ``param.high`` is the literal ``"ceil"``, look up the ceiling for
    ``param.name`` in ``ceilings``.  If the ceiling is missing, non-finite,
    less than or equal to ``param.low``, or ``ceilings is None``, return
    ``param.fallback_high``.
    """
    if isinstance(param.high, str):
        if param.high != "ceil":
            raise ValueError(
                f"BOParam.high must be a float or the literal 'ceil', "
                f"got {param.high!r}"
            )
        if ceilings is None:
            return float(param.fallback_high)
        c = ceilings.as_dict().get(param.name)
        if c is None or not math.isfinite(c) or c <= param.low:
            return float(param.fallback_high)
        return float(c)
    return float(param.high)


def search_space(ceilings: Ceilings | None) -> dict[str, tuple[float, float, bool]]:
    """Return ``{name: (low, high, log)}`` ready for any BO library.

    Used by tests and by ``objective.py`` (Task 3) to drive Optuna.
    """
    return {
        p.name: (float(p.low), resolve_high(p, ceilings), bool(p.log))
        for p in BO_DIMS
    }


def apply_to_config(cfg: MultiTSOConfig, params: dict[str, float]) -> MultiTSOConfig:
    """Return a new ``MultiTSOConfig`` with BO params overlaid plus
    ``FIXED_OVERRIDES``.

    Parameters
    ----------
    cfg
        Baseline config; not mutated.
    params
        Must contain exactly the keys in ``[p.name for p in BO_DIMS]``.
        Extra keys raise :class:`ValueError`.  Missing keys raise
        :class:`KeyError`.

    Returns
    -------
    MultiTSOConfig
        New instance with BO params + ``FIXED_OVERRIDES`` applied.  All
        other fields are unchanged from ``cfg``.

    Raises
    ------
    ValueError
        If ``params`` contains unknown keys, non-finite values, or
        negative values.
    KeyError
        If any expected BO param is missing from ``params``.
    """
    expected = {p.name for p in BO_DIMS}
    given = set(params.keys())
    extra = given - expected
    missing = expected - given
    if extra:
        raise ValueError(f"Unknown BO params: {sorted(extra)}")
    if missing:
        raise KeyError(f"Missing BO params: {sorted(missing)}")

    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"Non-numeric value for {k}: {v!r}")
        v_f = float(v)
        if not math.isfinite(v_f):
            raise ValueError(f"Non-finite value for {k}: {v}")
        if v_f < 0:
            raise ValueError(f"Negative value for {k}: {v}")

    overlay: dict[str, Any] = {**{k: float(v) for k, v in params.items()},
                               **FIXED_OVERRIDES}
    return dataclasses.replace(cfg, **overlay)


def params_from_config(cfg: MultiTSOConfig) -> dict[str, float]:
    """Inverse of :func:`apply_to_config`: extract the 8 BO fields from
    a config.

    Useful for warm-starting BO from a known-good config.
    """
    return {p.name: float(getattr(cfg, p.name)) for p in BO_DIMS}
