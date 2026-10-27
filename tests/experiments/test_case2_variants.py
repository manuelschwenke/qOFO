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


def test_m_is_registered_but_not_ready():
    assert "M" in v.VARIANTS and "M" in v.NOT_READY
    assert "M" not in v.WAVE_1


def test_m_weights_the_sts_buses_like_the_sts_controllers():
    m = v.resolved("M", 10)
    assert m.central_dso_g_v == pytest.approx(m.dso_g_v)
    assert m.control_scope == "central"


def test_m_targets_the_selected_designs_loop_gain():
    m = v.resolved("M", 10)
    assert m.central_lambda_target == pytest.approx(v.selected_lambda_max())
    assert 0.0 < m.central_lambda_target < 2.0
