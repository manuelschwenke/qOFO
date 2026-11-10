"""Set-point re-centring rule of the machine transformers (2026-11-10)."""

import math

import pandapower as pp
import pytest

from controller.local_rules import LocalDiscreteRules, SetpointTapRelay, TapRelay


def _net(v_set=1.0):
    """Grid (ext grid) -- line -- HV bus -- machine transformer -- machine terminal."""
    net = pp.create_empty_network()
    b_grid = pp.create_bus(net, 345.0)
    b_hv = pp.create_bus(net, 345.0)
    b_lv = pp.create_bus(net, 20.0)
    pp.create_ext_grid(net, b_grid, vm_pu=1.03)
    pp.create_line_from_parameters(net, b_grid, b_hv, 20, 0.01, 0.2, 0, 2.0)
    pp.create_transformer_from_parameters(
        net, b_hv, b_lv, sn_mva=500, vn_hv_kv=345, vn_lv_kv=20, vk_percent=12,
        vkr_percent=0.3, pfe_kw=0, i0_percent=0, tap_side="hv", tap_neutral=0,
        tap_min=-9, tap_max=9, tap_step_percent=1.25, tap_pos=0,
        tap_changer_type="Ratio")                 # as in the IEEE 39 build (pandapower 3)
    pp.create_gen(net, b_lv, p_mw=200.0, vm_pu=v_set, max_q_mvar=300, min_q_mvar=-300)
    pp.create_load(net, b_hv, p_mw=150.0, q_mvar=80.0)
    pp.runpp(net)
    return net, b_hv


def _relay(**kw):
    args = dict(trafo_idx=0, gen_idx=0, v_low_pu=0.96, v_high_pu=1.04, hold_s=900.0,
                confirm_steps=3)
    args.update(kw)
    return SetpointTapRelay(**args)


def test_high_setpoint_raises_grid_side_after_confirmation_then_holds():
    net, b_hv = _net(v_set=1.045)
    r = _relay()
    v_hv0 = float(net.res_bus.at[b_hv, "vm_pu"])
    assert r.step(net, 0.0) == 0 and r.step(net, 20.0) == 0      # not yet confirmed
    d = r.step(net, 40.0)
    assert d == r.direction(net) == 1                               # tap on HV side: +1
    assert int(net.trafo.at[0, "tap_pos"]) == 1
    pp.runpp(net)
    assert float(net.res_bus.at[b_hv, "vm_pu"]) > v_hv0             # grid side raised
    for t in (60.0, 80.0, 100.0, 900.0):                            # holding time
        assert r.step(net, t) == 0
    assert r.step(net, 960.0) == 1                                  # 3 confirmations since 900 s


def test_low_setpoint_lowers_grid_side():
    net, _ = _net(v_set=0.955)
    r = _relay(confirm_steps=1)
    assert r.step(net, 0.0) == -1
    assert int(net.trafo.at[0, "tap_pos"]) == -1


def test_inside_band_and_interrupted_confirmation_do_nothing():
    net, _ = _net(v_set=1.00)
    r = _relay()
    assert all(r.step(net, 20.0 * k) == 0 for k in range(10))
    net.gen.at[0, "vm_pu"] = 1.045
    r.step(net, 300.0)
    r.step(net, 320.0)
    net.gen.at[0, "vm_pu"] = 1.03                                   # leaves the band
    assert r.step(net, 340.0) == 0
    net.gen.at[0, "vm_pu"] = 1.045
    assert r.step(net, 360.0) == 0                                  # count restarted


def test_reserve_condition_blocks_raising_step():
    net, _ = _net(v_set=1.045)
    q = float(net.res_gen.at[0, "q_mvar"])
    assert q > 0.0
    blocked = _relay(confirm_steps=1, q_util_max=0.8, q_limits=lambda n: (-q, q / 0.9))
    assert blocked.step(net, 0.0) == 0                              # q = 0.9 Q_max
    free = _relay(confirm_steps=1, q_util_max=0.8, q_limits=lambda n: (-q, 2.0 * q))
    assert free.step(net, 0.0) == 1                                 # q = 0.5 Q_max


def test_machine_out_of_service_and_tap_limit():
    net, _ = _net(v_set=1.045)
    net.gen.at[0, "in_service"] = False
    r = _relay(confirm_steps=1)
    assert r.step(net, 0.0) == 0
    net.gen.at[0, "in_service"] = True
    net.trafo.at[0, "tap_pos"] = 9
    assert r.step(net, 0.0) == 0


def test_validation_and_rule_overlap():
    with pytest.raises(ValueError):
        _relay(v_low_pu=1.04, v_high_pu=0.96)
    with pytest.raises(ValueError):
        _relay(q_util_max=0.8)                                      # needs q_limits
    with pytest.raises(ValueError):
        LocalDiscreteRules([TapRelay(trafo_idx=0, v_ref_pu=1.03, tol_pu=0.01, hold_s=900.0)],
                           [], mutual_exclusion=True, setpoint_taps=[_relay()])


def test_rules_report_setpoint_action():
    net, _ = _net(v_set=1.045)
    rules = LocalDiscreteRules([], [], mutual_exclusion=True,
                               setpoint_taps=[_relay(confirm_steps=1)])
    acts = rules.step(net, 0.0)
    assert len(acts) == 1 and acts[0]["rule"] == "setpoint" and acts[0]["direction"] == 1
    assert math.isclose(acts[0]["v_set_pu"], 1.045)
