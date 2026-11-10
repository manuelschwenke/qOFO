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


def test_static_plant_rescues_a_diverged_solve(monkeypatch):
    """PandapowerStaticPlant.advance: a diverged solve is re-solved warm from the state before it."""
    import core.plant as cp
    from pandapower.auxiliary import LoadflowNotConverged
    net = _net()
    plant = cp.PandapowerStaticPlant(net, enforce_q_lims=True)
    calls = {"n": 0}
    real = cp.runpp_holding_taps

    def fail_once(n, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LoadflowNotConverged("forced")
        real(n, **kw)

    monkeypatch.setattr(cp, "runpp_holding_taps", fail_once)
    plant.advance()
    assert plant.warm_resolves == 1 and net.converged
    assert net.res_gen.q_mvar.iloc[0] <= 10.0 + 1e-6
    plant.advance()                      # converges normally: no rescue
    assert plant.warm_resolves == 1


def test_static_plant_converging_solve_is_unchanged():
    import core.plant as cp
    a, b = _net(), _net()
    cp.PandapowerStaticPlant(a, enforce_q_lims=True).advance()
    pp.runpp(b, enforce_q_lims=True, calculate_voltage_angles=True, run_control=True)
    assert abs(a.res_bus.vm_pu - b.res_bus.vm_pu).max() < 1e-9
