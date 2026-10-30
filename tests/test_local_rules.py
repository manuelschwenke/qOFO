"""Local tap and bank rules of the classical variants (2026-10-30)."""

import math

import pandapower as pp
import pytest

from controller.local_rules import BankRelay, LocalDiscreteRules, TapRelay
from controller.shunt_banks import ShuntBankConfig, make_bank


def _net():
    net = pp.create_empty_network()
    b_hv = pp.create_bus(net, 345.0)
    b_lv = pp.create_bus(net, 20.0)
    b_p2 = pp.create_bus(net, 345.0)
    pp.create_ext_grid(net, b_lv, vm_pu=1.0)
    pp.create_transformer_from_parameters(
        net, b_hv, b_lv, sn_mva=500, vn_hv_kv=345, vn_lv_kv=20, vk_percent=12,
        vkr_percent=0.3, pfe_kw=0, i0_percent=0, tap_side="hv", tap_neutral=0,
        tap_min=-9, tap_max=9, tap_step_percent=1.25, tap_pos=0)
    pp.create_line_from_parameters(net, b_hv, b_p2, 10, 0.01, 0.1, 0, 2.0)
    for b in (b_hv, b_p2):
        pp.create_shunt(net, b, q_mvar=-25.0, step=0, max_step=2)
        pp.create_shunt(net, b, q_mvar=25.0, step=0, max_step=2)
    pp.create_load(net, b_p2, p_mw=0, q_mvar=0)
    return net, b_hv, b_p2


def _bank(shunt_idx, bus, kind, dso="DSO_1"):
    return make_bank(ShuntBankConfig(
        shunt_idx=shunt_idx, bus_idx=bus, interface_trafo3w_idx=0, dso_id=dso,
        kind=kind, q_step_mvar=25.0, n_levels=2, t_dwell_s=1800.0))


def test_tap_relay_band_and_holding_time():
    net, b_hv, _ = _net()
    net.ext_grid.at[0, "vm_pu"] = 0.98
    pp.runpp(net)
    r = TapRelay(0, 1.03, 0.01, 900.0)
    assert net.res_bus.at[b_hv, "vm_pu"] < 1.02
    assert r.step(net, 0.0) == 1
    pp.runpp(net)
    assert r.step(net, 880.0) == 0          # inside the holding time
    assert r.step(net, 900.0) in (0, 1)
    with pytest.raises(ValueError):
        TapRelay(0, 1.03, 0.0, 900.0)


def test_bank_rule_thresholds_exclusion_and_release():
    net, b_hv, b_p2 = _net()
    pp.runpp(net)
    msc = BankRelay(_bank(0, b_hv, "MSC"), b_hv, 1.01, 1.05)
    msr = BankRelay(_bank(3, b_p2, "MSR"), b_p2, 1.01, 1.05)
    assert msc.wanted(1.00) == 1 and msc.wanted(1.03) == 0 and msc.wanted(1.06) == -1
    assert msr.wanted(1.06) == 1 and msr.wanted(1.03) == 0 and msr.wanted(1.00) == -1
    rules = LocalDiscreteRules([], [msc, msr], mutual_exclusion=True)
    net.res_bus.loc[[b_hv, b_p2], "vm_pu"] = [1.00, 1.06]
    acts = rules.step(net, 0.0)
    # both want to engage; the first engagement blocks the opposite device
    assert [a["kind"] for a in acts] == ["MSC"]
    assert int(net.shunt.at[0, "step"]) == 1
    net.res_bus.loc[[b_hv, b_p2], "vm_pu"] = [1.00, 1.03]
    assert rules.step(net, 600.0) == []      # MSC inside its holding time
    acts = rules.step(net, 1800.0)
    assert acts[0]["new_level"] == 2
    net.res_bus.loc[[b_hv, b_p2], "vm_pu"] = [1.06, 1.06]
    acts = rules.step(net, 3600.0)
    # release of the MSC is committed before the MSR engagement
    assert [(a["kind"], a["direction"]) for a in acts] == [("MSC", -1)]
    assert math.isclose(msc.bank.level, 1)
