"""The canonical weight vector, and the holes the 2026-09-10 reset closed.

These tests exist because the previous search space drifted away from the
configs it was meant to describe: ``BO_DIMS`` carried bounds that excluded the
only known-good point (``g_v = 1e7`` against a box of ``[1e2, 1e5]``), and
``FIXED_OVERRIDES`` silently re-pinned weights on every trial.  The properties
asserted here are what stops either recurring.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from configs.config import MultiTSOConfig
from tuning.parameters import (
    FIXED_OVERRIDES,
    GAUGE_FIELD,
    SEARCHED_FIELDS,
    TIMING_FIELDS,
    WEIGHT_FIELDS,
    apply_to_config,
    params_from_config,
    weight_fingerprint,
)


@pytest.fixture()
def weights() -> dict[str, float]:
    return {
        "g_v": 1.0e4, "g_q": 100.0, "dso_g_v": 2.0e3,
        "g_w_der": 10.0, "g_w_pcc": 20.0, "g_w_gen": 1.0e8,
        "g_w_tso_oltc": 500.0, "g_w_dso_der": 100.0, "g_w_dso_oltc": 50.0,
    }


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------

def test_weight_fields_are_all_config_fields() -> None:
    names = {f.name for f in dataclasses.fields(MultiTSOConfig)}
    for w in WEIGHT_FIELDS:
        assert w in names, f"{w} is not a MultiTSOConfig field"
    for t in TIMING_FIELDS:
        assert t in names, f"{t} is not a MultiTSOConfig field"


def test_weight_fields_match_the_configs_required_set() -> None:
    """One list of required weights, not two.

    ``configs.config`` raises on an unset weight and ``tuning.parameters``
    decides what a parameter set must carry.  If these two ever disagree, a
    parameter set can be complete by one definition and incomplete by the
    other -- which is exactly the class of bug the reset removed.
    """
    assert set(WEIGHT_FIELDS) == set(MultiTSOConfig.REQUIRED_WEIGHTS)


def test_gauge_is_pinned_and_the_rest_are_searched() -> None:
    assert GAUGE_FIELD in WEIGHT_FIELDS
    assert GAUGE_FIELD not in SEARCHED_FIELDS
    assert set(SEARCHED_FIELDS) == set(WEIGHT_FIELDS) - {GAUGE_FIELD}
    assert len(SEARCHED_FIELDS) == len(WEIGHT_FIELDS) - 1 == 8


def test_fixed_overrides_pins_no_weight() -> None:
    """A weight pinned here would be a silent parameter set.

    The retired ``FIXED_OVERRIDES`` pinned ``g_w_tso_shunt`` and
    ``dso_gamma_oltc_q`` on every trial, so a candidate never actually ran at
    the weights it recorded.
    """
    for w in WEIGHT_FIELDS:
        assert w not in FIXED_OVERRIDES, f"{w} is pinned in FIXED_OVERRIDES"


def test_fixed_overrides_disables_the_runtime_preconditioner() -> None:
    """``precondition_g_w`` regenerates ``g_w`` at controller construction.

    Left on, the weights that run are not the weights that were selected.
    """
    assert FIXED_OVERRIDES["precondition_g_w"] is False


# ---------------------------------------------------------------------------
# 2. apply_to_config
# ---------------------------------------------------------------------------

def test_complete_overlay_applies_and_satisfies_the_config_guard(weights) -> None:
    cfg = apply_to_config(MultiTSOConfig(), weights)
    assert cfg.missing_weights() == []
    cfg.require_weights()
    for k, v in weights.items():
        assert float(getattr(cfg, k)) == pytest.approx(v)


def test_timing_fields_may_ride_along(weights) -> None:
    cfg = apply_to_config(
        MultiTSOConfig(),
        {**weights, "tso_period_s": 180.0, "dso_period_s": 20.0, "dt_s": 20.0},
    )
    assert cfg.tso_period_s == pytest.approx(180.0)
    assert cfg.dso_period_s == pytest.approx(20.0)


def test_partial_overlay_raises(weights) -> None:
    """The whole point: no weight may be inherited from anywhere."""
    partial = {k: v for k, v in weights.items() if k != "g_w_pcc"}
    with pytest.raises(KeyError, match="incomplete"):
        apply_to_config(MultiTSOConfig(), partial)


def test_unknown_key_raises(weights) -> None:
    with pytest.raises(ValueError, match="unknown parameter-set keys"):
        apply_to_config(MultiTSOConfig(), {**weights, "g_w_tso_shunt": 1.0})


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_non_positive_or_non_finite_raises(weights, bad) -> None:
    with pytest.raises(ValueError):
        apply_to_config(MultiTSOConfig(), {**weights, "g_w_der": bad})


def test_bool_is_rejected(weights) -> None:
    """``True`` is an ``int``; a weight of 1 arrived at by accident is a bug."""
    with pytest.raises(ValueError, match="non-numeric"):
        apply_to_config(MultiTSOConfig(), {**weights, "g_w_der": True})


def test_overlay_does_not_mutate_the_baseline(weights) -> None:
    base = MultiTSOConfig()
    apply_to_config(base, weights)
    assert base.missing_weights() == list(MultiTSOConfig.REQUIRED_WEIGHTS)


# ---------------------------------------------------------------------------
# 3. Round-trip and fingerprint
# ---------------------------------------------------------------------------

def test_params_from_config_round_trips(weights) -> None:
    cfg = apply_to_config(MultiTSOConfig(), weights)
    assert params_from_config(cfg) == pytest.approx(weights)


def test_fingerprint_is_stable_and_discriminating(weights) -> None:
    a = weight_fingerprint(weights)
    assert a == weight_fingerprint(dict(reversed(list(weights.items()))))
    assert a != weight_fingerprint({**weights, "g_w_der": weights["g_w_der"] * 1.01})
    assert len(a) == 12
