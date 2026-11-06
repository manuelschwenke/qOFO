"""Tests for experiments/ch_11_case2_multisystem/variants.py (thesis ch:case3)."""

from __future__ import annotations

import dataclasses

import pytest

from experiments.ch_11_case2_multisystem import variants as v
from experiments.helpers.records import ContingencyEvent

_EVENTS = [ContingencyEvent(minute=0, time_s=1840.0, element_type="gen",
                            element_index=2, action="trip")]


def test_guard_passes_for_every_registered_step():
    v.assert_case2_controlled(minutes=10, events=_EVENTS, noise_seed=7)


def test_guard_rejects_a_stray_difference(monkeypatch):
    bad = dict(v.VARIANTS)
    bad["O3"] = dict(bad["O3"], g_w_gen=1.0)        # not part of O2 -> O3
    monkeypatch.setattr(v, "VARIANTS", bad)
    with pytest.raises(RuntimeError, match="O2->O3: g_w_gen"):
        v.assert_case2_controlled(minutes=10)


def test_no_event_is_inherited_from_make_config():
    assert v.resolved("O2", 10).contingencies == []


def test_events_and_noise_seed_reach_every_variant():
    for name in v.VARIANTS:
        cfg = v.resolved(name, 10, events=_EVENTS, noise_seed=123)
        assert cfg.contingencies == _EVENTS
        assert cfg.measurement_noise.enabled
        assert cfg.measurement_noise.apply_to_svr
        assert cfg.measurement_noise.apply_to_der_voltage
        assert cfg.measurement_noise.seed == 123
        assert cfg.n_total_s == 600.0


def test_o2_to_o3_changes_only_the_sts_controller():
    a, b = v.resolved("O2", 10), v.resolved("O3", 10)
    diff = {f.name for f in dataclasses.fields(a)
            if f.name not in ("result_dir", "verbose")
            and getattr(a, f.name) != getattr(b, f.name)}
    assert diff == {"dso_mode"}


def test_unknown_variant_is_refused():
    with pytest.raises(KeyError):
        v.resolved("S9", 10)


def test_m_is_ready_and_has_the_cascade_timescale():
    assert "M" in v.VARIANTS and "M" not in v.NOT_READY
    m = v.resolved("M", 10)
    n = m.central_ts_discrete_cooldown
    # free again after the same time as a TS-OFO device: (1 + c) * T_TS
    assert (1 + n) * m.dt_s == pytest.approx((1 + m.int_cooldown) * m.tso_period_s)
    assert n == 53
    assert m.central_lambda_rule == "continuous"


def test_m_weights_the_sts_buses_like_the_sts_controllers():
    m = v.resolved("M", 10)
    assert m.central_dso_g_v == pytest.approx(m.dso_g_v)
    assert m.control_scope == "central"


def test_m_targets_the_declared_loop_gain_below_the_selected_designs():
    m = v.resolved("M", 10)
    assert m.central_lambda_target == pytest.approx(0.90)
    assert 0.0 < m.central_lambda_target < v.selected_lambda_max() < 2.0


def test_msel_is_m_at_the_selected_designs_loop_gain():
    m, ms = v.resolved("M", 10), v.resolved("Msel", 10)
    assert ms.central_lambda_target == pytest.approx(v.selected_lambda_max())
    diff = {f.name for f in dataclasses.fields(m) if getattr(m, f.name) != getattr(ms, f.name)}
    assert diff == {"central_lambda_target"}


def test_s1_to_s2_changes_only_the_sts_controller_and_the_access():
    a, b = v.resolved("S1", 10), v.resolved("S2", 10)
    diff = {f.name for f in dataclasses.fields(a)
            if f.name not in ("result_dir", "verbose")
            and getattr(a, f.name) != getattr(b, f.name)}
    assert diff == {"dso_mode", "svr_sts_access"}
