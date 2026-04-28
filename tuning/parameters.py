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


# 9 BO dimensions
BO_DIMS: tuple[BOParam, ...] = (
    BOParam("g_v",           log=True, low=1e2,  high="ceil", fallback_high=1e7),
    BOParam("g_q",           log=True, low=1.0,  high=1e4),
    BOParam("dso_g_v",       log=True, low=1e2,  high=1e6),
    BOParam("g_w_der",       log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_pcc",       log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_tso_oltc",  log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_tso_shunt", log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_dso_der",   log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_dso_oltc",  log=True, low=1e-1, high="ceil", fallback_high=1e4),
)


# Fields always pinned during tuning (override baseline config).
FIXED_OVERRIDES: dict[str, Any] = {
    # Per-user decision: g_w_gen excluded from stability tuning
    "g_w_gen":                 1e7,

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
