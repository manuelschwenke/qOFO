"""
tuning/parameters.py
====================
Declarative definition of the Bayesian-optimization decision space and
the mapping between BO param dicts and :class:`MultiTSOConfig` instances.

The BO search space is 8-dimensional.

``FIXED_OVERRIDES`` lists fields that are always overwritten when
applying BO params, regardless of their value in the baseline config:
live plots disabled, verbose silenced, etc.  This guarantees that BO
trials are headless and deterministic w.r.t. the baseline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

from configs.config import MultiTSOConfig
from tuning._types import BOParam, Ceilings


#: Bumped whenever the *meaning* of a coordinate changes in a way that makes
#: old trials incomparable, even if the bounds happen to stay the same.
#: Bounds changes are caught automatically by the fingerprint below; this is
#: for semantic changes the numbers cannot express.
SEARCH_SPACE_VERSION: int = 2


def search_space_fingerprint() -> str:
    """Stable digest of the decision space, for storage in ``study.user_attrs``.

    Resuming a study across a change of search space silently mixes trials that
    were never comparable.  That happened here: every persisted IEEE-39 study
    records a 9th parameter, ``tso_g_q_tie``, which is not in
    :data:`BO_DIMS` any more -- and is not even a field of
    :class:`~configs.config.MultiTSOConfig` (the field is ``tso_g_q_pcc``).
    Feeding such a study's best-params back through :func:`apply_to_config`
    raises ``ValueError: Unknown BO params``.

    :func:`tuning.tune.main` compares this digest against the one recorded on
    the study and refuses to resume on a mismatch.
    """
    payload = {
        "version": SEARCH_SPACE_VERSION,
        "dims": sorted(
            (p.name, float(p.low), str(p.high), bool(p.log)) for p in BO_DIMS
        ),
        "fixed": sorted(
            (k, str(v)) for k, v in FIXED_OVERRIDES.items()
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# 8 BO dimensions.
# `g_w_tso_shunt` is conditionally excluded while
# `install_tso_tertiary_shunts=False` in the baseline (no shunt actuators
# → vacuous coordinate). The slot is pinned in FIXED_OVERRIDES so the
# baseline value still flows through `apply_to_config`. Re-enable by
# uncommenting and removing the FIXED_OVERRIDES entry once shunts are
# installed.
#
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
    BOParam("g_w_pcc",       log=True, low=1e-1, high=30.0),
    BOParam("g_w_tso_oltc",  log=True, low=1, high=1e5), # high="ceil", fallback_high=1e4),
    #BOParam("g_w_tso_shunt", log=True, low=1e-1, high="ceil", fallback_high=1e4),
    BOParam("g_w_dso_der",   log=True, low=1e-1, high=1e3), # high="ceil", fallback_high=1e4),
    BOParam("g_w_dso_oltc",  log=True, low=1, high=1e5), # high="ceil", fallback_high=1e4),
    # ── Stage-2 (Q(V) local loop) knobs ─────────────────
)


# Fields always pinned during tuning (override baseline config).
FIXED_OVERRIDES: dict[str, Any] = {
    # g_w_gen is excluded from BO_DIMS and therefore inherits the baseline
    # exactly. It is deliberately not duplicated as a numeric override:
    # run_multi_system_ofo uses a very large value to keep AVR moves slow.

    # Conditionally pinned: shunts not installed at TSO tertiary
    # (`install_tso_tertiary_shunts=False`), so this dim is vacuous.
    # Value matches the 002 baseline; remove this key when shunts are
    # re-installed and the BO_DIMS entry is uncommented.
    "g_w_tso_shunt":           50000.0,


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

    This is a *pure extractor* and deliberately does not validate -- callers
    that need the config to be reachable by the optimiser should ask
    :func:`out_of_box_params` and decide what to do about it.
    """
    return {p.name: float(getattr(cfg, p.name)) for p in BO_DIMS}


def out_of_box_params(
    cfg: MultiTSOConfig,
    ceilings: Ceilings | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Config fields that the search space **cannot represent**.

    Returns ``{name: (value, low, high)}`` for every BO dimension whose value
    in ``cfg`` lies outside its bounds -- i.e. operating points no trial can
    ever propose.

    This is not hypothetical.  Three independent configurations are all
    outside the current box on the same two coordinates:

    =========================  ==========  ============  ==================
    config                     ``g_v``     ``g_w_pcc``   box
    =========================  ==========  ============  ==================
    hand-tuned ``make_config``  1e7         80            ``g_v``: [1e2, 1e5]
    ``baseline_002_ieee39``     1e5 (edge)  100           ``g_w_pcc``: [0.1, 30]
    ``tests/tuning/conftest``   1.2e5       100
    =========================  ==========  ============  ==================

    The hand-tuned point is the one you report as controlling well, so the
    optimiser was searching a region that excluded the only known-good answer.
    Hard-coding ``--no-warm-start-baseline`` suppressed the symptom.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for p in BO_DIMS:
        v = float(getattr(cfg, p.name))
        hi = resolve_high(p, ceilings)
        if v < p.low or v > hi:
            out[p.name] = (v, float(p.low), float(hi))
    return out
