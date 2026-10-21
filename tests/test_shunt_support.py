"""Behavioural checks for confirmed demand and explicit STS shunt assistance."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from controller.shunt_integrator import MSC, ShuntBankConfig, ShuntIntegrator
from controller.shunt_support import (
    STSFeedbackTracker, STSShuntFeedback, STSShuntSupportConfig, support_request,
)
from controller.tso_controller import TSOController


def bank(**kwargs):
    values = dict(shunt_idx=0, bus_idx=0, interface_trafo3w_idx=0,
                  dso_id="STS", kind="MSC", q_step_mvar=25.0, n_levels=3,
                  g_w=1.0, delta=5.0, t_dwell_s=0.0, daily_switch_budget=10,
                  y_h_min=0.9, y_h_max=1.1, t_persistence_s=20.0,
                  max_sample_gap_s=15.0)
    values.update(kwargs)
    return MSC(ShuntBankConfig(**values))


def step(b, t, g, **kwargs):
    return b.step(grad_g=g, v_meas=np.array([1.0]), h_v=np.array([0.001]),
                  t_now=float(t), **kwargs)


@pytest.mark.parametrize("gradients", [
    [-40, 0, 0, 0], [-10, 0, -10, 0, -10, 0, -10, 0],
])
def test_disconnected_demand_does_not_switch(gradients):
    b = bank()
    assert all(step(b, 10*k, g) is None for k, g in enumerate(gradients))
    assert b.level == 0 and b.q_eq_aux == 0


def test_sustained_large_demand_waits_for_persistence_then_switches():
    b = bank()
    assert step(b, 0, -40) is None
    assert step(b, 10, -40) is None
    assert step(b, 20, -40).direction == 1
    assert b.q_eq_aux == b.q_eq_phys
    assert step(b, 30, -40) is None
    assert step(b, 40, -40) is None
    assert step(b, 50, -40).new_level == 2


def test_demand_disappearing_during_dwell_cancels_request():
    b = bank(t_dwell_s=60)
    b.level, b.q_eq_aux, b._last_switch_t = 1, 25, 0
    for t in (0, 10, 20):
        assert step(b, t, -40) is None
    for t in (30, 40, 50, 60):
        assert step(b, t, 0) is None
    assert b.level == 1 and b.q_eq_aux == 25


@pytest.mark.parametrize("change", ["direction", "kind", "sample_gap"])
def test_changed_demand_restarts_confirmation(change):
    b = bank()
    b.level, b.q_eq_aux = 1, 25
    assert step(b, 0, -40) is None
    assert step(b, 10, -40) is None
    t, g, kind = 20, -40, "objective"
    if change == "direction":
        g = 40
    elif change == "kind":
        kind = "sts_tracking"
    else:
        t = 40
    assert step(b, t, g, request_kind=kind) is None
    assert step(b, t+10, g, request_kind=kind) is None
    assert step(b, t+20, g, request_kind=kind) is not None


def test_deadband_and_voltage_guard_remain_effective():
    b = bank(gradient_deadband=1)
    assert step(b, 0, -40) is None
    assert step(b, 10, -0.5) is None
    assert b.q_eq_aux == 0
    b = bank(y_h_max=1.01)
    assert all(step(b, t, -40) is None for t in (0, 10, 20, 30))
    assert b.level == 0


def test_one_commit_limit_keeps_other_bank_pending():
    integ = ShuntIntegrator([bank(), bank()])
    for t in (0, 10):
        assert not integ.update([-40, -40], [np.array([1.0])]*2,
                                [np.array([0.001])]*2, t, max_commits=1)
    commits = integ.update([-40, -40], [np.array([1.0])]*2,
                           [np.array([0.001])]*2, 20, max_commits=1)
    assert len(commits) == 1
    assert [b.level for b in integ.banks] == [1, 0]


def feedback(measured=0.0, requested=40.0, lo=-50.0, hi=50.0, t=100.0, age=100.0):
    a = lambda x: np.atleast_1d(np.asarray(x, dtype=float))
    n = len(a(measured))
    return STSShuntFeedback(np.arange(n), a(measured), a(requested),
                            np.broadcast_to(a(lo), (n,)).copy(),
                            np.broadcast_to(a(hi), (n,)).copy(),
                            np.full(n, age), t)


def request(f, h=1.0, **kwargs):
    values = dict(h_q=np.atleast_1d(h), q_step_mvar=25.0, level=0,
                  n_levels=3, t_now=100.0, config=STSShuntSupportConfig())
    values.update(kwargs)
    return support_request(f, **values)


@pytest.mark.parametrize("sign", [-1, 1])
def test_tracking_assistance_uses_device_direction(sign):
    r = request(feedback(requested=40*sign), h=sign)
    assert r.kind == "sts_tracking" and r.gradient < 0
    np.testing.assert_allclose(r.predicted_delta_q, [25*sign])
    assert r.cost_after < r.cost_before


def test_inactive_bank_cannot_supply_opposite_direction():
    assert request(feedback(requested=-40), h=1) is None
    r = request(feedback(requested=-40), h=1, level=1)
    assert r.gradient > 0 and r.predicted_delta_q[0] == -25


def test_tracking_assistance_rejects_full_step_overshoot():
    assert request(feedback(requested=10)) is None


def test_updated_tso_reference_cancels_old_assistance_need():
    f = feedback(requested=40)
    assert request(f, current_q_requested=[0]) is None
    assert request(f, current_q_requested=[-40]) is None
    assert request(f, current_q_requested=[50]).kind == "sts_tracking"


def test_updated_tso_reference_cancels_old_reserve_need():
    f = feedback(measured=45, requested=45)
    assert request(f, current_q_requested=[0]) is None


@pytest.mark.parametrize("value", [-40, -20, 0, 20, 40])
def test_reserve_support_is_inactive_inside_central_band(value):
    assert request(feedback(measured=value, requested=value)) is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_reserve_support_near_limit_uses_correct_direction(sign):
    r = request(feedback(measured=45*sign, requested=45*sign), h=sign)
    assert r.kind == "sts_reserve" and r.gradient < 0
    assert r.cost_before > 0 and r.cost_after == 0


def test_tracking_has_priority_over_reserve_and_can_be_disabled():
    f = feedback(measured=45, requested=60)
    r = request(f, q_step_mvar=10)
    assert r.kind == "sts_tracking"
    cfg = replace(STSShuntSupportConfig(), tracking_weight=0)
    assert request(f, config=cfg) is None


def test_no_reserve_dispatch_on_collapsed_capability():
    assert request(feedback(measured=5, requested=5, lo=5, hi=5)) is None


def test_reserve_rejects_step_crossing_opposite_capability_limit():
    assert request(feedback(measured=45, requested=45), q_step_mvar=100) is None


@pytest.mark.parametrize("age,t_now", [(0, 100), (100, 161), (100, 99)])
def test_freshness_and_response_wait(age, t_now):
    assert request(feedback(age=age), t_now=t_now) is None


def test_reported_limits_need_not_be_reached_for_tracking_assistance():
    # An internal network constraint may prevent tracking inside the DER envelope.
    r = request(feedback(measured=0, requested=40, lo=-100, hi=100))
    assert r.kind == "sts_tracking"


def test_support_is_invariant_to_existing_shunt_offset():
    a = request(feedback(measured=45, requested=45))
    b = request(feedback(measured=145, requested=145, lo=50, hi=150))
    assert a.gradient == pytest.approx(b.gradient)
    assert a.cost_before == pytest.approx(b.cost_before)


@pytest.mark.parametrize("reserve", [False, True])
def test_support_gradients_match_cost_finite_difference(reserve):
    f = feedback(measured=[45, -46], requested=[45, -46]) if reserve else feedback(
        measured=[0, 0], requested=[40, -35])
    h = np.array([0.8, -0.6])
    r = request(f, h=h)
    def cost(x):
        if reserve:
            z = (f.q_requested - h*x - (f.q_min+f.q_max)/2) / ((f.q_max-f.q_min)/2)
            return np.maximum(np.abs(z)-0.8, 0) @ np.maximum(np.abs(z)-0.8, 0)
        z = np.maximum(np.abs(f.q_requested-f.q_measured-h*x)-5, 0)/25
        return z @ z
    fd = (cost(1e-4)-cost(-1e-4))/(2e-4)
    assert r.gradient == pytest.approx(fd, rel=1e-7)


def test_feedback_tracker_records_actual_reference_age_and_absolute_capability():
    tracker = STSFeedbackTracker()
    def obs(t, target):
        return tracker.observe(interface_indices=[3], q_measured=[10],
                               q_requested=[target], delta_min=[-20],
                               delta_max=[30], t_now=t)
    assert obs(0, 40).reference_age_s[0] == 0
    f = obs(60, 40)
    assert f.reference_age_s[0] == 60
    assert f.q_min[0] == -10 and f.q_max[0] == 40
    assert obs(80, 41.5).reference_age_s[0] == 0


def test_transfer_preserves_total_reference_and_other_actuators():
    tso = TSOController.__new__(TSOController)
    tso.config = SimpleNamespace(der_mapping=SimpleNamespace(n_der=2),
        der_indices=[9], pcc_trafo_indices=[10, 20], pcc_dso_controller_ids=["S", "S"])
    tso.controller_id, tso.iteration = "T", 1
    tso._u_current = np.array([1., 2., 40., -30., 1.03])
    offset = np.array([5., -7.])
    before = tso.generate_setpoint_messages()[0].q_setpoints_mvar + offset
    tso.transfer_pcc_request_to_shunt([20, 10], [-4., 25.])
    offset += [25., -4.]
    after = tso.generate_setpoint_messages()[0].q_setpoints_mvar + offset
    np.testing.assert_allclose(after, before)
    np.testing.assert_allclose(tso._u_current[[0, 1, 4]], [1, 2, 1.03])
    old = tso._u_current.copy()
    with pytest.raises(ValueError):
        tso.transfer_pcc_request_to_shunt([10, 99], [5., 5.])
    np.testing.assert_array_equal(tso._u_current, old)


def test_closed_loop_support_removes_unreachable_request_and_restores_headroom():
    # Reduced scalar plant: STS actuator x in [-20,20], interface Q=x+bank.
    # The controller sees only measured Q and reported capability increments.
    b = bank(g_w=0.01)
    target, x, shunt = 40.0, 20.0, 0.0
    commits = []
    for t in range(0, 80, 10):
        f = feedback(measured=x+shunt, requested=target,
                     lo=-20+shunt, hi=20+shunt, t=t)
        r = request(f, t_now=t)
        c = step(b, t, r.gradient if r else 0,
                 request_kind=r.kind if r else "objective")
        if c:
            shunt += c.direction*25
            commits.append(c)
        x = float(np.clip(target-shunt, -20, 20))
    assert len(commits) == 1 and commits[0].request_kind == "sts_tracking"
    assert x+shunt == target and x == 15


@pytest.mark.parametrize("field,value", [("t_persistence_s", -1),
    ("t_persistence_s", float("nan")), ("gradient_deadband", -1),
    ("max_sample_gap_s", 0)])
def test_invalid_persistence_settings(field, value):
    with pytest.raises(ValueError):
        bank(**{field:value})


def test_reserve_restoration_stops_inside_central_band():
    # Total interface Q is held at 45 Mvar. The continuous STS actuator starts
    # at 90% utilisation. A bank should offload it once and then remain idle.
    b = bank(g_w=0.001)
    target, x, shunt = 45.0, 45.0, 0.0
    commits = []
    for t in range(0, 100, 10):
        f = feedback(measured=x+shunt, requested=target,
                     lo=-50+shunt, hi=50+shunt, t=t)
        r = request(f, t_now=t, level=b.level)
        c = step(b, t, r.gradient if r else 0,
                 request_kind=r.kind if r else "objective")
        if c:
            shunt += c.direction*25
            commits.append(c)
        x = target-shunt
    assert len(commits) == 1 and commits[0].request_kind == "sts_reserve"
    assert x+shunt == target and x == 20


@pytest.mark.parametrize("settings", [
    {"shunt_int_t_persistence_s": 0},
    {"shunt_int_t_persistence_s": float("nan")},
    {"shunt_dispatch": "off"},
    {"control_scope": "central"},
    {"dso_mode": "local"},
    {"tso_mode": "local"},
    {"coordination_mode": "sbx_v"},
    {"dso_ids_to_run": ["DSO_1"]},
])
def test_unsupported_support_configurations_fail_early(settings):
    from configs.config import MultiTSOConfig
    values = dict(shunt_sts_support=True, shunt_int_t_persistence_s=360,
                  shunt_dispatch="integrator")
    values.update(settings)
    with pytest.raises(ValueError):
        MultiTSOConfig(**values).validate_integrator_mode()
