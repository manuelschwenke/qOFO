"""Unit tests for ``tuning/scenarios.py``."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from configs.config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent
from tuning.scenarios import (
    VALID_NETWORK_SCENARIOS,
    ScenarioSpec,
    design_set,
    holdout_set_v2,
    tune_set_v2,
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
        scenario="rural_700",
        use_profiles=False,
        tso_period_s=120.0,
        dso_period_s=15.0,
        dt_s=15.0,
    )
    out = spec.overlay_on(baseline_cfg)

    assert out.n_total_s == pytest.approx(42.0)
    assert out.start_time == datetime(2017, 7, 4, 9, 0)
    assert out.tso_period_s == pytest.approx(120.0)
    assert out.dso_period_s == pytest.approx(15.0)
    # dt_s is now part of the spec.  It used to be omitted, so it silently
    # inherited the baseline's 60 s while dso_period_s said 10 s — and since
    # the runner tests `time_s % period_s < 1`, the DSO then fired on *every*
    # plant step instead of every 10 s.  The cascade ran with no timescale
    # separation at all.
    assert out.dt_s == pytest.approx(15.0)
    assert out.scenario == "rural_700"
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
    """The 80/20 split now names two *real* networks.

    It used to draw ``"wind_replace"`` (a deprecated alias) and ``"base"``.
    ``"base"`` is not in ``SCENARIO_REGISTRY``, so ``build_ieee39_net`` raised
    for that fifth of the draws, ``run_one`` swallowed the exception into a
    sentinel cost, and ~20 % of every validation campaign was silently recorded
    as a power-flow failure.
    """
    n = 500
    vs = validation_set(0, n)
    frac = sum(1 for s in vs if s.scenario == "base_410") / n
    assert 0.75 <= frac <= 0.85, (
        f"base_410 fraction {frac:.3f} out of [0.75, 0.85] window"
    )
    assert all(s.scenario in VALID_NETWORK_SCENARIOS for s in vs)


def test_no_scenario_spec_can_name_an_unknown_network() -> None:
    """Regression guard for the ``"base"`` defect: fail loudly, not silently."""
    with pytest.raises(ValueError, match="Unknown network scenario"):
        ScenarioSpec(name="bad", start_time=datetime(2016, 1, 1),
                     duration_s=60.0, scenario="base")


def test_dso_period_below_dt_is_rejected() -> None:
    """A stated DSO period faster than the plant step is fiction.

    ``_is_period_hit`` tests ``time_s % period_s < 1``, so any
    ``dso_period_s <= dt_s`` makes the DSO fire every plant step regardless of
    the number written in the spec.  The design set claimed 10 s against a 60 s
    plant step for the entire history of the tuning package.
    """
    with pytest.raises(ValueError, match="below dt_s"):
        ScenarioSpec(name="bad", start_time=datetime(2016, 1, 1),
                     duration_s=60.0, dt_s=20.0, dso_period_s=10.0)


# ---------------------------------------------------------------------------
# v2 tune / holdout sets
# ---------------------------------------------------------------------------

def test_tune_and_holdout_calendars_are_disjoint() -> None:
    """No calendar-block leakage between tuning and holdout.

    SimBench profiles are strongly autocorrelated within a day, so a random
    day-level split would leak: a holdout scenario could share its load and
    generation profile with a tune scenario and the "held-out" score would be
    partly in-sample.  Splitting on ISO-week parity does not leak.

    This caught a real slip: the legacy ``_T_WINTER`` (2016-01-14) is ISO week
    2 — an even, i.e. holdout, week — so the first draft of the v2 tune set
    overlapped the holdout calendar.
    """
    tune_weeks = {s.start_time.isocalendar().week for s in tune_set_v2()}
    hold_weeks = {s.start_time.isocalendar().week
                  for s in holdout_set_v2(42, 40)}
    assert all(w % 2 == 1 for w in tune_weeks), sorted(tune_weeks)
    assert all(w % 2 == 0 for w in hold_weeks), sorted(hold_weeks)
    assert not (tune_weeks & hold_weeks)


def test_tune_set_v2_covers_both_networks() -> None:
    """``rural_700`` had never been tuned on: every legacy scenario was
    ``base_410`` via the deprecated ``wind_replace`` alias."""
    networks = {s.scenario for s in tune_set_v2()}
    assert networks == {"base_410", "rural_700"}


def test_tune_set_v2_has_sustained_ramps_without_restore() -> None:
    """Impulsive trip/restore is absorbed by the continuous actuators.

    Only sustained one-way drift exhausts reactive reserve and hands authority
    to the tap changers — which is what makes the OLTC weights identifiable at
    all.  A ramp scenario must therefore contain no ``trip`` undoing its
    ``connect`` events.
    """
    ramps = [s for s in tune_set_v2() if "ramp" in s.name]
    assert ramps
    for s in ramps:
        actions = [ev.action for ev in s.contingencies]
        assert actions.count("connect") >= 3
        assert "trip" not in actions


def test_event_times_anchor_the_settling_windows() -> None:
    quiet = next(s for s in tune_set_v2() if s.name == "v2_quiet_spring")
    assert quiet.event_times_s == ()      # nothing to settle from
    ramp = next(s for s in tune_set_v2() if "undervoltage" in s.name)
    assert len(ramp.event_times_s) == len(set(ramp.event_times_s))
    assert list(ramp.event_times_s) == sorted(ramp.event_times_s)


def test_holdout_is_reproducible_and_fixed_duration() -> None:
    """Fixed duration, unlike ``validation_set``: drawing {30,60,90} min
    reintroduces a 9x T^2 ITAE bias that has nothing to do with control."""
    a, b = holdout_set_v2(7, 12), holdout_set_v2(7, 12)
    assert [s.start_time for s in a] == [s.start_time for s in b]
    assert len({s.duration_s for s in a}) == 1
