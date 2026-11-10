"""Warm, staged re-solve after a diverged end-of-step power flow (core.plant.runpp_warm_staged, 2026-11-10)."""

import pandapower as pp

from core.plant import runpp_warm_staged, snapshot_results


def _net():
    """Ext grid -- line -- bus 1 (IBG) -- line -- bus 2 (machine with a tight Q limit, heavy Q load)."""
    net = pp.create_empty_network()
    b0, b1, b2 = (pp.create_bus(net, 110.0) for _ in range(3))
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_line_from_parameters(net, b0, b1, 30, 0.05, 0.4, 0, 1.0)
    pp.create_line_from_parameters(net, b1, b2, 30, 0.05, 0.4, 0, 1.0)
    pp.create_gen(net, b2, p_mw=20.0, vm_pu=1.02, min_q_mvar=-10.0, max_q_mvar=10.0)
    pp.create_load(net, b2, p_mw=30.0, q_mvar=40.0)
    pp.create_sgen(net, b1, p_mw=5.0, q_mvar=0.0)
    pp.runpp(net)
    return net


def test_q_limits_enforced_from_the_snapshot():
    net = _net()
    snap = snapshot_results(net)
    net.res_bus = net.res_bus.iloc[0:0]          # as if the diverged solve had cleared the results
    net.sgen["q_mvar"] = 1e3                     # and a Q(V) seed far off
    how = runpp_warm_staged(net, snap, enforce_q_lims=True, calculate_voltage_angles=True)
    assert how.startswith("warm, Q limits")
    assert net.res_gen.q_mvar.iloc[0] <= 10.0 + 1e-6
    assert net.converged


def test_without_q_limits_the_machine_exceeds_them():
    net = _net()
    snap = snapshot_results(net)
    how = runpp_warm_staged(net, snap, enforce_q_lims=False, calculate_voltage_angles=True)
    assert how == "warm, no Q limits"
    assert net.res_gen.q_mvar.iloc[0] > 10.0


def test_snapshot_restores_every_result_table_and_the_der_outputs():
    net = _net()
    snap = snapshot_results(net)
    assert {"res_bus", "res_gen", "res_line", "_sgen_q_mvar"} <= set(snap)
    q0 = float(net.sgen.q_mvar.iloc[0])
    net.sgen["q_mvar"] = 99.0
    runpp_warm_staged(net, snap, enforce_q_lims=False, calculate_voltage_angles=True)
    assert abs(float(net.sgen.q_mvar.iloc[0]) - q0) < 1e-9
