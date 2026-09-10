"""SVR availability regressions: outage, missing results, and restoration."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from controller.svr_controller import SVRZoneConfig, SVRZoneController


class CapabilityBounds:
    def compute_gen_q_bounds(self, p, v):
        # Missing outage measurements must not enter a capability calculation.
        assert np.isfinite(p).all()
        assert np.isfinite(v).all()
        q = np.sqrt(200.0**2 - np.asarray(p)**2)
        return -q, q

    def compute_der_q_bounds(self, p):
        return np.zeros(0), np.zeros(0)


def make_controller(*, second_in_service=False, withheld=()):
    net = SimpleNamespace(
        gen=pd.DataFrame({"vm_pu": [1.0, 1.012],
                          "in_service": [True, second_in_service]}, index=[10, 11]),
        res_gen=pd.DataFrame({"p_mw": [0.0, np.nan], "q_mvar": [40.0, np.nan]}, index=[10, 11]),
        res_bus=pd.DataFrame({"vm_pu": [1.02]}, index=[5]),
        sgen=pd.DataFrame(columns=["p_mw", "q_mvar"]),
        res_sgen=pd.DataFrame(columns=["q_mvar"]),
    )
    zone = SimpleNamespace(zone_id=1, gen_indices=[10, 11], tso_der_indices=[],
                           pcc_trafo_indices=[], oltc_trafo_indices=[], shunt_bus_indices=[])
    ctrl = SVRZoneController(zone, CapabilityBounds(), SVRZoneConfig(pilot_bus=5, deadband_pu=0.0),
                             non_dispatchable_gens=withheld)
    ctrl.dv_pilot_dq = np.array([1e-4, 1e-4])
    ctrl.dq_dvset = np.array([1000.0, 1000.0])
    ctrl.pilot_sensitivity = 0.04
    return ctrl, net


def test_outage_excludes_missing_results_from_capability_and_initial_level():
    ctrl, net = make_controller()
    np.testing.assert_allclose(ctrl._headroom(net, upper=True), [200.0, 0.0])
    np.testing.assert_allclose(ctrl._headroom(net, upper=False), [200.0, 0.0])
    ctrl._initialise_level(net)
    assert ctrl.state.q_level == pytest.approx(0.2)
    assert ctrl.state.integrator == pytest.approx(0.2)


def test_outage_holds_avr_reference_instead_of_integrating_towards_a_limit():
    ctrl, net = make_controller()
    ctrl.state.q_level = ctrl.state.integrator = 0.4
    ctrl._initialised = True
    for _ in range(20):
        result = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
        assert np.isfinite(result.u_new).all()
        assert result.u_new[1] == pytest.approx(1.012)
        net.gen.loc[:, "vm_pu"] = result.u_new


def test_restoration_resumes_rpr_without_modifying_static_dispatch_permission():
    ctrl, net = make_controller()
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    before = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    net.gen.at[11, "in_service"] = True
    net.res_gen.loc[11, ["p_mw", "q_mvar"]] = [0.0, 0.0]
    after = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert before.u_new[1] == pytest.approx(1.012)
    assert after.u_new[1] > before.u_new[1]
    np.testing.assert_array_equal(ctrl.gen_mask, [1.0, 1.0])


def test_online_withheld_machine_remains_excluded_even_with_missing_q():
    ctrl, net = make_controller(second_in_service=True, withheld=[11])
    result = ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    assert np.isfinite(result.u_new).all()
    assert result.u_new[1] == pytest.approx(1.012)
    assert ctrl._headroom(net, upper=True)[1] == 0.0


def test_rpr_calls_do_not_advance_the_regional_integrator_between_dispatches():
    ctrl, net = make_controller()
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 0.98
    ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert ctrl.state.q_level == pytest.approx(0.2)
    ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    assert ctrl.state.q_level > 0.2


@pytest.mark.parametrize("pilot_v,measured_q", [(1.00, 100.0), (1.06, 0.0)])
def test_voltage_priority_retains_support_when_q_alignment_would_worsen_pilot(pilot_v, measured_q):
    ctrl, net = make_controller()
    ctrl.cfg.rpr_voltage_priority = True
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = pilot_v
    net.res_gen.at[10, "q_mvar"] = measured_q
    out = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert out.u_new[0] == pytest.approx(net.gen.at[10, "vm_pu"])
    np.testing.assert_array_equal(out.rpr_blocked, [True, False])
    np.testing.assert_array_equal(out.gen_available, [True, False])
    np.testing.assert_allclose(out.q_ref_mvar, [40.0, 0.0])


@pytest.mark.parametrize("pilot_v,measured_q,sign", [(1.00, 0.0, 1), (1.06, 100.0, -1)])
def test_voltage_priority_allows_q_alignment_that_supports_pilot(pilot_v, measured_q, sign):
    ctrl, net = make_controller()
    ctrl.cfg.rpr_voltage_priority = True
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = pilot_v
    net.res_gen.at[10, "q_mvar"] = measured_q
    out = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert sign * (out.u_new[0] - net.gen.at[10, "vm_pu"]) > 0
    assert not out.rpr_blocked.any()


def test_voltage_priority_uses_cached_sensitivity_sign():
    ctrl, net = make_controller()
    ctrl.cfg.rpr_voltage_priority = True
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    ctrl.dv_pilot_dq[0] = -1e-4
    net.res_bus.at[5, "vm_pu"] = 1.00
    net.res_gen.at[10, "q_mvar"] = 0.0
    out = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert out.u_new[0] == pytest.approx(1.0)
    assert out.rpr_blocked[0]


def test_voltage_priority_respects_existing_supervisory_deadband():
    ctrl, net = make_controller()
    ctrl.cfg.rpr_voltage_priority = True
    ctrl.cfg.deadband_pu = 0.01
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.025
    net.res_gen.at[10, "q_mvar"] = 100.0
    out = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert out.pilot_error_pu == pytest.approx(0.0)
    assert out.u_new[0] < 1.0
    assert not out.rpr_blocked.any()


def test_voltage_priority_can_be_disabled_for_classical_reference_replay():
    ctrl, net = make_controller()
    ctrl.cfg.rpr_voltage_priority = False
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.00
    net.res_gen.at[10, "q_mvar"] = 100.0
    out = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert out.u_new[0] < 1.0
    assert not out.rpr_blocked.any()


def test_rvr_proportional_term_is_algebraic_and_uses_pilot_sensitivity():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rvr = 0.2
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.00
    out = ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    # I: 0.2 + 180/300 * 0.03/0.04 = 0.65; P: 0.2*0.03/0.04 = 0.15.
    assert ctrl.state.integrator == pytest.approx(0.65)
    assert out.rvr_p_term_level == pytest.approx(0.15)
    assert out.q_level == pytest.approx(0.8)


def test_rvr_pi_initialisation_is_bumpless():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rvr = 0.2
    ctrl._initialise_level(net)
    # q0=40/200=0.2; P0=0.2*(1.03-1.02)/0.04=0.05.
    assert ctrl.state.q_level == pytest.approx(0.2)
    assert ctrl.state.rvr_p_term == pytest.approx(0.05)
    assert ctrl.state.integrator == pytest.approx(0.15)
    ctrl._initialised = True
    held = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert held.q_level == pytest.approx(0.2)
    # At the first regional call the initial P term cancels out: the only
    # command change is the same integral increment as in the I-only case.
    due = ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    assert due.q_level == pytest.approx(0.35)


def test_rvr_proportional_output_is_held_between_regional_samples():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rvr = 0.2
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.00
    first = ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    net.res_bus.at[5, "vm_pu"] = 1.02
    held = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert held.q_level == pytest.approx(first.q_level)
    assert held.rvr_p_term_level == pytest.approx(first.rvr_p_term_level)


def test_optional_rvr_proportional_path_reacts_on_inner_grid_without_integrating():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rvr = 0.2
    ctrl.cfg.rvr_p_every_step = True
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.00
    before_x = ctrl.state.integrator
    out = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert ctrl.state.integrator == pytest.approx(before_x)
    assert out.rvr_p_term_level == pytest.approx(0.15)
    assert out.q_level == pytest.approx(0.35)


def test_rvr_conditional_anti_windup_allows_unwinding():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rvr = 0.2
    ctrl.state.q_level = ctrl.state.integrator = 0.95
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.00
    saturated = ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    assert saturated.q_level == pytest.approx(1.0)
    assert ctrl.state.integrator == pytest.approx(0.95)
    net.res_bus.at[5, "vm_pu"] = 1.06
    ctrl.step(net, 20.0, rvr_due=True, dt_rvr_s=180.0)
    assert ctrl.state.integrator < 0.95


def test_rpr_proportional_term_is_not_integrated_again():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rpr = 0.2
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_gen.at[10, "q_mvar"] = 0.0
    first = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    net.gen.loc[:, "vm_pu"] = first.u_new
    second = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    expected_integral_increment = (20.0 / 60.0) * 40.0 / 1000.0
    assert first.rpr_p_term_pu[0] == pytest.approx(0.008)
    assert second.u_new[0] - first.u_new[0] == pytest.approx(expected_integral_increment)


def test_voltage_priority_rolls_back_hidden_rpr_integrator():
    ctrl, net = make_controller()
    ctrl.cfg.k_p_rpr = 0.2
    ctrl.cfg.rpr_voltage_priority = True
    ctrl.state.q_level = ctrl.state.integrator = 0.2
    ctrl._initialised = True
    net.res_bus.at[5, "vm_pu"] = 1.00
    net.res_gen.at[10, "q_mvar"] = 100.0
    first = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    state_after_first = ctrl._rpr_integrator.copy()
    second = ctrl.step(net, 20.0, rvr_due=False, dt_rvr_s=180.0)
    assert first.rpr_blocked[0] and second.rpr_blocked[0]
    assert first.u_new[0] == pytest.approx(1.0)
    assert second.u_new[0] == pytest.approx(1.0)
    np.testing.assert_allclose(ctrl._rpr_integrator, state_after_first, equal_nan=True)


@pytest.mark.parametrize("field", ["k_p_rvr", "k_p_rpr"])
def test_pi_gains_must_be_nonnegative_and_finite(field):
    ctrl, net = make_controller()
    values = dict(pilot_bus=5, **{field: -0.1})
    with pytest.raises(ValueError, match=field):
        SVRZoneController(ctrl.zone_def, CapabilityBounds(), SVRZoneConfig(**values))

