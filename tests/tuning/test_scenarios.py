"""Unit tests for ``tuning/scenarios.py``."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent
from tuning.scenarios import (
    ScenarioSpec,
    design_set,
    validation_set,
)


# ---------------------------------------------------------------------------
# 1. design_set: 5 named scenarios
# ---------------------------------------------------------------------------

def test_design_set_returns_5_scenarios() -> None:
    ds = design_set()
    assert len(ds) == 5
    names = {s.name for s in ds}
    assert names == {
        "nominal_quiet", "gen_trip_recovery", "load_step",
        "dual_disturbance", "winter_peak",
    }
    # No duplicate name
    assert len({s.name for s in ds}) == len(ds)


# ---------------------------------------------------------------------------
# 2. design_set durations sit in the documented range
# ---------------------------------------------------------------------------

def test_design_set_durations_reasonable() -> None:
    ds = design_set()
    for s in ds:
        assert 60 * 60 <= s.duration_s <= 120 * 60, (
            f"{s.name}: duration {s.duration_s}s outside [3600, 7200]"
        )


# ---------------------------------------------------------------------------
# 3. overlay_on must NOT touch controller weights
# ---------------------------------------------------------------------------

def test_overlay_on_preserves_controller_weights(
    baseline_cfg: MultiTSOConfig,
) -> None:
    cfg = dataclasses.replace(
        baseline_cfg,
        g_v=12345.0,
        g_q=67.0,
        g_w_der=99.0,
        g_w_pcc=11.0,
        g_w_tso_oltc=22.0,
        g_w_dso_der=33.0,
        g_w_dso_oltc=44.0,
    )
    for s in design_set():
        out = s.overlay_on(cfg)
        assert out.g_v == pytest.approx(12345.0)
        assert out.g_q == pytest.approx(67.0)
        assert out.g_w_der == pytest.approx(99.0)
        assert out.g_w_pcc == pytest.approx(11.0)
        assert out.g_w_tso_oltc == pytest.approx(22.0)
        assert out.g_w_dso_der == pytest.approx(33.0)
        assert out.g_w_dso_oltc == pytest.approx(44.0)


# ---------------------------------------------------------------------------
# 4. overlay_on replaces timing + start + contingencies
# ---------------------------------------------------------------------------

def test_overlay_on_replaces_timing(baseline_cfg: MultiTSOConfig) -> None:
    spec = ScenarioSpec(
        name="custom",
        start_time=datetime(2017, 7, 4, 9, 0),
        duration_s=42.0,
        contingencies=(
            ContingencyEvent(
                minute=1, element_type="gen", element_index=2, action="trip",
            ),
        ),
        scenario="base",
        use_profiles=False,
        tso_period_s=120.0,
        dso_period_s=15.0,
    )
    out = spec.overlay_on(baseline_cfg)

    assert out.n_total_s == pytest.approx(42.0)
    assert out.start_time == datetime(2017, 7, 4, 9, 0)
    assert out.tso_period_s == pytest.approx(120.0)
    assert out.dso_period_s == pytest.approx(15.0)
    assert out.scenario == "base"
    assert out.use_profiles is False
    assert len(out.contingencies) == 1
    ev = out.contingencies[0]
    assert ev.element_type == "gen"
    assert ev.element_index == 2
    assert ev.action == "trip"
    # Original baseline untouched
    assert baseline_cfg.n_total_s != pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 5. validation_set is reproducible for fixed seed
# ---------------------------------------------------------------------------

def test_validation_set_reproducible() -> None:
    a = validation_set(42, 10)
    b = validation_set(42, 10)
    assert len(a) == len(b) == 10
    for x, y in zip(a, b):
        assert x.start_time == y.start_time
        assert x.duration_s == y.duration_s
        assert x.scenario == y.scenario
        assert len(x.contingencies) == len(y.contingencies)
        for cx, cy in zip(x.contingencies, y.contingencies):
            assert cx.element_type == cy.element_type
            assert cx.element_index == cy.element_index
            assert cx.action == cy.action
            assert cx.minute == cy.minute
            assert cx.bus == cy.bus
            # NaN-aware p_mw / q_mvar comparison
            for fld in ("p_mw", "q_mvar"):
                vx, vy = getattr(cx, fld), getattr(cy, fld)
                if vx != vx:  # NaN
                    assert vy != vy
                else:
                    assert vx == pytest.approx(vy)


# ---------------------------------------------------------------------------
# 6. validation_set returns the requested count
# ---------------------------------------------------------------------------

def test_validation_set_size() -> None:
    assert len(validation_set(1, 50)) == 50
    assert len(validation_set(0, 1)) == 1
    assert len(validation_set(7, 0)) == 0


# ---------------------------------------------------------------------------
# 7. validation_set scenario-string distribution
# ---------------------------------------------------------------------------

def test_validation_set_distribution_sanity() -> None:
    n = 500
    vs = validation_set(0, n)
    n_wind = sum(1 for s in vs if s.scenario == "wind_replace")
    frac = n_wind / n
    assert 0.75 <= frac <= 0.85, (
        f"wind_replace fraction {frac:.3f} out of [0.75, 0.85] window"
    )
    valid_strings = {"wind_replace", "base"}
    assert all(s.scenario in valid_strings for s in vs)
