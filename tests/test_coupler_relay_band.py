"""Run-time band of the local coupler relays, ``coupler_relay_tol_pu`` (2026-11-10)."""

import pandapower as pp
from pandapower.control import DiscreteTapControl

from configs.config import MultiTSOConfig
from experiments.runners._multi_tso_helpers import set_coupler_relay_band


def _net():
    """Grid (ext grid) -- 345 kV bus -- 3W coupler (345/110/20 kV) -- 110 kV load, plus a
    345/20 kV 2W machine transformer whose relay must stay untouched."""
    net = pp.create_empty_network()
    b_ehv = pp.create_bus(net, 345.0)
    b_hv = pp.create_bus(net, 110.0)
    b_lv = pp.create_bus(net, 20.0)
    b_mt = pp.create_bus(net, 20.0)
    pp.create_ext_grid(net, b_ehv, vm_pu=1.03)
    for _ in range(2):                                 # two couplers, as in an STS
        pp.create_transformer3w_from_parameters(
            net, b_ehv, b_hv, b_lv, vn_hv_kv=345, vn_mv_kv=110, vn_lv_kv=20,
            sn_hv_mva=300, sn_mv_mva=300, sn_lv_mva=100, vk_hv_percent=12, vk_mv_percent=8,
            vk_lv_percent=8, vkr_hv_percent=0.3, vkr_mv_percent=0.3, vkr_lv_percent=0.3,
            pfe_kw=0, i0_percent=0, tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
            tap_step_percent=1.25, tap_pos=0, tap_changer_type="Ratio")
    pp.create_transformer_from_parameters(
        net, b_ehv, b_mt, sn_mva=500, vn_hv_kv=345, vn_lv_kv=20, vk_percent=12,
        vkr_percent=0.3, pfe_kw=0, i0_percent=0, tap_side="hv", tap_neutral=0,
        tap_min=-9, tap_max=9, tap_step_percent=1.25, tap_pos=0, tap_changer_type="Ratio")
    pp.create_load(net, b_hv, p_mw=200.0, q_mvar=60.0)
    for t in net.trafo3w.index:
        DiscreteTapControl(net, element_index=t, vm_lower_pu=1.02, vm_upper_pu=1.04,
                           side="mv", element="trafo3w")
    DiscreteTapControl(net, element_index=0, vm_lower_pu=1.02, vm_upper_pu=1.04,
                       side="hv", element="trafo")
    return net


def _bands(net, element):
    return [(round(c.vm_lower_pu, 6), round(c.vm_upper_pu, 6))
            for c in net.controller.object if c.element == element]


def test_sets_band_of_listed_couplers_only():
    net = _net()
    n = set_coupler_relay_band(net, net.trafo3w.index, 1.03, 0.02)
    assert n == 2
    assert _bands(net, "trafo3w") == [(1.01, 1.05), (1.01, 1.05)]
    assert _bands(net, "trafo") == [(1.02, 1.04)]          # machine transformer untouched


def test_unlisted_coupler_untouched():
    net = _net()
    assert set_coupler_relay_band(net, [0], 1.03, 0.02) == 1
    assert _bands(net, "trafo3w") == [(1.01, 1.05), (1.02, 1.04)]


def test_relay_acts_on_the_new_band():
    """With the 110 kV side just below 1.02 pu, the +/- 0.01 relay steps; the
    +/- 0.02 relay leaves the taps where they are."""
    taps = {}
    for tol in (0.01, 0.02):
        net = _net()
        pp.runpp(net)
        net.ext_grid.vm_pu = 1.03 + (1.015 - float(net.res_bus.vm_pu.iloc[1]))
        set_coupler_relay_band(net, net.trafo3w.index, 1.03, tol)
        pp.runpp(net, run_control=True)
        taps[tol] = net.trafo3w.tap_pos.tolist()
    assert taps[0.02] == [0, 0]
    assert taps[0.01] != [0, 0]


def test_config_default_keeps_old_behaviour():
    assert MultiTSOConfig.__dataclass_fields__["coupler_relay_tol_pu"].default is None
