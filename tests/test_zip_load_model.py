"""
tests/test_zip_load_model.py
============================
The anchored ZIP load model (network/ieee39/load_model.py) must be the
exact exponent image (kpu, kqu) = (1, 2):

    P_served(V) = P_prof · (V / V_anchor)      Q_served(V) = Q_prof · (V / V_anchor)²

realised as 100 % constant-current P / 100 % constant-impedance Q with the
anchor folded into the bases.  Decision 2026-07-17 (RMS co-simulation).
"""

from __future__ import annotations

import pandapower as pp
import pytest

from export.make_snapshots import DEFAULT_T0, build_snapshot_state
from network.ieee39.load_model import apply_zip_load_model

ANCHOR = 1.03


def _tiny_net(load_at_slack: bool) -> tuple[pp.pandapowerNet, int]:
    """Slack at 1.03 pu; one 50 MW / 20 Mvar load, optionally behind a line."""
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b0, vm_pu=ANCHOR)
    if load_at_slack:
        bus = b0
    else:
        bus = pp.create_bus(net, vn_kv=110.0)
        pp.create_line_from_parameters(net, b0, bus, length_km=10.0,
                                       r_ohm_per_km=0.1, x_ohm_per_km=0.4,
                                       c_nf_per_km=10.0, max_i_ka=1.0)
    li = pp.create_load(net, bus=bus, p_mw=50.0, q_mvar=20.0)
    net.load["base_p_mw"] = net.load["p_mw"].astype(float)
    net.load["base_q_mvar"] = net.load["q_mvar"].astype(float)
    return net, int(li)


def test_anchor_identity_at_setpoint_voltage():
    """At V = anchor exactly, the served power equals the profile value."""
    net, li = _tiny_net(load_at_slack=True)
    apply_zip_load_model(net, anchor_vm_pu=ANCHOR)
    pp.runpp(net, calculate_voltage_angles=True)
    assert net.res_bus.at[int(net.load.at[li, "bus"]), "vm_pu"] == pytest.approx(ANCHOR)
    assert net.res_load.at[li, "p_mw"] == pytest.approx(50.0, abs=1e-9)
    assert net.res_load.at[li, "q_mvar"] == pytest.approx(20.0, abs=1e-9)


def test_exponent_image_off_anchor():
    """Away from the anchor: P = P_prof·(V/anchor), Q = Q_prof·(V/anchor)²."""
    net, li = _tiny_net(load_at_slack=False)
    apply_zip_load_model(net, anchor_vm_pu=ANCHOR)
    pp.runpp(net, calculate_voltage_angles=True)
    vm = float(net.res_bus.at[int(net.load.at[li, "bus"]), "vm_pu"])
    assert vm != pytest.approx(ANCHOR)  # the line drop must bite
    assert net.res_load.at[li, "p_mw"] == pytest.approx(50.0 * vm / ANCHOR,
                                                        abs=1e-9)
    assert net.res_load.at[li, "q_mvar"] == pytest.approx(
        20.0 * (vm / ANCHOR) ** 2, abs=1e-9)


def test_double_application_raises():
    net, _li = _tiny_net(load_at_slack=True)
    apply_zip_load_model(net, anchor_vm_pu=ANCHOR)
    with pytest.raises(ValueError, match="already applied"):
        apply_zip_load_model(net, anchor_vm_pu=ANCHOR)


def test_full_build_serves_zip_consistently():
    """Every load of the built network obeys P = p·V and Q = q·V² and the
    anchor bookkeeping column is set (snapshot state, base phase)."""
    state = build_snapshot_state("base", DEFAULT_T0, load_model="zip",
                                 verbose=0)
    net = state.net
    assert (net.load["zip_anchor_vm_pu"] == ANCHOR).all()
    assert (net.load["const_i_p_percent"] == 100.0).all()
    assert (net.load["const_z_q_percent"] == 100.0).all()
    for li in net.load.index:
        bus = int(net.load.at[li, "bus"])
        vm = float(net.res_bus.at[bus, "vm_pu"])
        assert net.res_load.at[li, "p_mw"] == pytest.approx(
            float(net.load.at[li, "p_mw"]) * vm, abs=1e-9), f"load {li}"
        assert net.res_load.at[li, "q_mvar"] == pytest.approx(
            float(net.load.at[li, "q_mvar"]) * vm ** 2, abs=1e-9), f"load {li}"


def test_const_pq_option_reproduces_legacy_plant():
    """load_model='const_pq' must leave the ZIP shares untouched (all zero)."""
    state = build_snapshot_state("base", DEFAULT_T0, load_model="const_pq",
                                 verbose=0)
    net = state.net
    assert "zip_anchor_vm_pu" not in net.load.columns
    for col in ("const_z_p_percent", "const_i_p_percent",
                "const_z_q_percent", "const_i_q_percent"):
        assert (net.load[col] == 0.0).all()
