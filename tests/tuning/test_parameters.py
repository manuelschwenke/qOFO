"""Unit tests for ``tuning/parameters.py``."""

from __future__ import annotations

import dataclasses
import math

import pytest

from configs.multi_tso_config import MultiTSOConfig
from tuning._types import BOParam, Ceilings
from tuning.parameters import (
    BO_DIMS,
    FIXED_OVERRIDES,
    apply_to_config,
    params_from_config,
    resolve_high,
    search_space,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_params() -> dict[str, float]:
    """A complete, valid BO param dict with one nominal value per BO_DIM."""
    return {p.name: 1.0 for p in BO_DIMS}


def _multi_tso_config_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(MultiTSOConfig)}


# ---------------------------------------------------------------------------
# 1. BO_DIMS structural integrity
# ---------------------------------------------------------------------------

def test_bo_dims_complete() -> None:
    """``BO_DIMS`` has exactly 9 entries (``g_w_tso_shunt`` excluded
    while shunts are not installed; ``tso_g_q_tie`` added 2026-04-29);
    every name exists on ``MultiTSOConfig``."""
    assert len(BO_DIMS) == 9
    cfg_fields = _multi_tso_config_fields()
    names = [p.name for p in BO_DIMS]
    assert len(set(names)) == len(names), "duplicate BO_DIMS names"
    for p in BO_DIMS:
        assert p.name in cfg_fields, (
            f"BO_DIMS entry '{p.name}' is not a field on MultiTSOConfig"
        )


# ---------------------------------------------------------------------------
# 2. FIXED_OVERRIDES structural integrity
# ---------------------------------------------------------------------------

def test_fixed_overrides_valid() -> None:
    """Every key in ``FIXED_OVERRIDES`` is an attribute on
    ``MultiTSOConfig``; ``int_cooldown`` is exactly ``1``."""
    cfg_fields = _multi_tso_config_fields()
    for k in FIXED_OVERRIDES:
        assert k in cfg_fields, f"FIXED_OVERRIDES key '{k}' is not a MultiTSOConfig field"
    assert FIXED_OVERRIDES["int_cooldown"] == 1


# ---------------------------------------------------------------------------
# 3. apply_to_config: param overlay
# ---------------------------------------------------------------------------

def test_apply_to_config_overlays_params(baseline_cfg: MultiTSOConfig) -> None:
    """``apply_to_config`` writes the param value to the config field;
    original config is unchanged (immutability via ``dataclasses.replace``)."""
    params = _valid_params()
    params["g_w_der"] = 42.0
    params["g_v"] = 7.5e3

    new_cfg = apply_to_config(baseline_cfg, params)

    assert new_cfg.g_w_der == pytest.approx(42.0)
    assert new_cfg.g_v == pytest.approx(7.5e3)
    # Baseline unchanged (apply_to_config does not mutate).
    assert baseline_cfg.g_w_der != pytest.approx(42.0)
    assert baseline_cfg.g_v != pytest.approx(7.5e3)
    # Returned object is a different instance.
    assert new_cfg is not baseline_cfg


# ---------------------------------------------------------------------------
# 4. apply_to_config: FIXED_OVERRIDES are always applied
# ---------------------------------------------------------------------------

def test_apply_to_config_applies_fixed_overrides(baseline_cfg: MultiTSOConfig) -> None:
    """After ``apply_to_config``, all ``FIXED_OVERRIDES`` values are
    present on the returned config -- even if the baseline differs."""
    # Construct a baseline whose fixed-override fields differ from the
    # values FIXED_OVERRIDES will pin.
    diverging = dataclasses.replace(
        baseline_cfg,
        g_w_gen=42.0,
        dso_g_qi=99.0,
        int_cooldown=9,
        verbose=2,
        live_plot_controller=True,
    )
    new_cfg = apply_to_config(diverging, _valid_params())
    for k, v in FIXED_OVERRIDES.items():
        assert getattr(new_cfg, k) == v, (
            f"FIXED_OVERRIDES['{k}']={v!r} not enforced; got {getattr(new_cfg, k)!r}"
        )


# ---------------------------------------------------------------------------
# 5. apply_to_config: unknown keys raise ValueError
# ---------------------------------------------------------------------------

def test_apply_to_config_rejects_unknown_keys(baseline_cfg: MultiTSOConfig) -> None:
    bad = _valid_params()
    bad["not_a_real_param"] = 1.0
    with pytest.raises(ValueError, match="Unknown BO params"):
        apply_to_config(baseline_cfg, bad)


# ---------------------------------------------------------------------------
# 6. apply_to_config: missing keys raise KeyError
# ---------------------------------------------------------------------------

def test_apply_to_config_rejects_missing_keys(baseline_cfg: MultiTSOConfig) -> None:
    bad = _valid_params()
    bad.pop("g_w_der")
    with pytest.raises(KeyError, match="Missing BO params"):
        apply_to_config(baseline_cfg, bad)


# ---------------------------------------------------------------------------
# 7. apply_to_config: rejects nan / inf / negative values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1e-3, -100.0])
def test_apply_to_config_rejects_negative_or_nan(
    baseline_cfg: MultiTSOConfig,
    bad_value: float,
) -> None:
    bad = _valid_params()
    bad["g_v"] = bad_value
    with pytest.raises(ValueError):
        apply_to_config(baseline_cfg, bad)


# ---------------------------------------------------------------------------
# 8. resolve_high: ceil resolution
# ---------------------------------------------------------------------------

def _make_ceilings(**kwargs: float) -> Ceilings:
    base = {
        "g_w_der":       math.inf,
        "g_w_pcc":       math.inf,
        "g_w_tso_oltc":  math.inf,
        "g_w_tso_shunt": math.inf,
        "g_w_dso_der":   math.inf,
        "g_w_dso_oltc":  math.inf,
        "g_v":           math.inf,
    }
    base.update(kwargs)
    return Ceilings(**base)


def test_resolve_high_with_ceil() -> None:
    """``BOParam(high='ceil')`` resolves to the ceilings-dict value when
    finite, else ``fallback_high``."""
    p = BOParam("g_w_der", log=True, low=1.0, high="ceil", fallback_high=12345.0)

    # Finite ceiling above low: returned.
    c_ok = _make_ceilings(g_w_der=999.0)
    assert resolve_high(p, c_ok) == pytest.approx(999.0)

    # Inf ceiling: fallback used.
    c_inf = _make_ceilings(g_w_der=math.inf)
    assert resolve_high(p, c_inf) == pytest.approx(12345.0)

    # Ceiling at-or-below low: fallback used (BO range would be empty).
    c_low = _make_ceilings(g_w_der=0.5)
    assert resolve_high(p, c_low) == pytest.approx(12345.0)

    # ceilings is None: fallback used.
    assert resolve_high(p, None) == pytest.approx(12345.0)

    # NaN ceiling: fallback used.
    c_nan = _make_ceilings(g_w_der=float("nan"))
    assert resolve_high(p, c_nan) == pytest.approx(12345.0)

    # Numeric (non-"ceil") high: returned regardless of ceilings.
    p_fixed = BOParam("g_q", log=True, low=1.0, high=5.0, fallback_high=999.0)
    assert resolve_high(p_fixed, c_ok) == pytest.approx(5.0)
    assert resolve_high(p_fixed, None) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 9. search_space: 8 entries, log on g_v + g_w_*
# ---------------------------------------------------------------------------

def test_search_space_returns_9_entries_with_log() -> None:
    space = search_space(None)
    assert len(space) == 9
    assert set(space.keys()) == {p.name for p in BO_DIMS}

    must_be_log = {"g_v", "tso_g_q_tie", "g_w_der", "g_w_pcc",
                   "g_w_tso_oltc", "g_w_dso_der", "g_w_dso_oltc"}
    for name, (low, high, log) in space.items():
        assert low > 0, f"{name}: low={low} must be > 0 for log-space search"
        assert high > low, f"{name}: high={high} must exceed low={low}"
        if name in must_be_log:
            assert log is True, f"{name} must use log-scale BO sampling"


# ---------------------------------------------------------------------------
# 9b. g_w_pcc upper bound is capped (degenerate-optimum guard)
# ---------------------------------------------------------------------------

def test_g_w_pcc_upper_bound_capped() -> None:
    """``g_w_pcc`` upper bound is capped at 30 to prevent BO from
    converging to the degenerate "freeze the PCC setpoint" optimum
    (see the rationale comment in ``tuning/parameters.py`` and the
    ``CostWeights`` docstring in ``tuning/metrics.py``).

    Asserted as a strict upper bound rather than equality so a future
    further-tightening of the bound does not break the regression."""
    low, high, log = search_space(None)["g_w_pcc"]
    assert log is True
    assert low == pytest.approx(1e-1)
    assert high <= 30.0, (
        f"g_w_pcc upper bound {high} > 30 — values above ~30 produce "
        f"sluggish PCC tracking with no end-performance benefit; see "
        f"tuning/parameters.py rationale comment."
    )


# ---------------------------------------------------------------------------
# 10. params_from_config / apply_to_config round-trip
# ---------------------------------------------------------------------------

def test_params_from_config_round_trip(baseline_cfg: MultiTSOConfig) -> None:
    """``apply_to_config(cfg, params_from_config(cfg))`` differs from
    ``cfg`` only in ``FIXED_OVERRIDES`` fields."""
    extracted = params_from_config(baseline_cfg)
    assert set(extracted.keys()) == {p.name for p in BO_DIMS}
    for name, value in extracted.items():
        assert value == pytest.approx(float(getattr(baseline_cfg, name)))

    round_trip = apply_to_config(baseline_cfg, extracted)
    overridden = set(FIXED_OVERRIDES.keys())
    for f in dataclasses.fields(MultiTSOConfig):
        baseline_val = getattr(baseline_cfg, f.name)
        rt_val = getattr(round_trip, f.name)
        if f.name in overridden:
            assert rt_val == FIXED_OVERRIDES[f.name], (
                f"{f.name}: expected fixed-override value {FIXED_OVERRIDES[f.name]!r}, "
                f"got {rt_val!r}"
            )
        else:
            assert rt_val == baseline_val, (
                f"{f.name}: round-trip mutated non-overridden field "
                f"({baseline_val!r} -> {rt_val!r})"
            )
