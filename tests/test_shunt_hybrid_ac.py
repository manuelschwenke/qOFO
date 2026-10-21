"""Directional AC checks with a constructed persistent interface request.

The subordinate contribution is held to isolate an unserved request. This is
not evidence that the full STS MIQP would fail to track that request.
"""
from dataclasses import replace
import numpy as np
import pandapower as pp
import pytest

from controller.shunt_hybrid import BankModel, HybridContext, HybridShuntSolver
from controller.shunt_integrator import ShuntBankConfig, make_bank
from controller.shunt_support import STSShuntFeedback, STSShuntSupportConfig
from network.ieee39.build import build_ieee39_net
from network.ieee39.hv_networks import add_hv_networks
from optimisation.miqp_solver import MIQPProblem, MIQPSolver
from sensitivity.jacobian import JacobianSensitivities


@pytest.mark.parametrize("kind,sign", [("MSC", -1), ("MSR", 1)])
@pytest.mark.parametrize("direction", [1, -1])
def test_joint_support_step_reduces_actual_ac_error(kind, sign, direction):
    net, meta = build_ieee39_net(scenario="rural_700", verbose=False)
    meta = add_hv_networks(net, meta, install_tso_tertiary_shunts=True,
        tso_shunt_kind="msc_msr", msc_n_levels=2, msr_n_levels=2,
        msc_q_step_mvar=25., msr_q_step_mvar=25., verbose=False)
    sh, bus = next((int(s), int(b)) for s, b, k in zip(
        meta.tso_tertiary_shunt_indices, meta.tso_tertiary_shunt_buses,
        meta.tso_tertiary_shunt_kinds) if k == kind)
    hv = next(h for h in meta.hv_networks if bus in h.coupling_lv_bus_indices)
    ids = np.asarray(hv.coupling_trafo_indices, dtype=int)
    host = int(ids[list(hv.coupling_lv_bus_indices).index(bus)])
    initial_level = 0 if direction == 1 else 1
    net.shunt.at[sh, "step"] = initial_level
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True)
    sens = JacobianSensitivities(net)
    obs = [int(b) for b in net.bus.index
           if float(net.bus.at[b, "vn_kv"]) >= 220
           and b not in set(net.gen.bus) and b not in set(net.ext_grid.bus)]
    dv, obs = sens.compute_dV_dQ_shunt(bus, obs, sign*25.)
    dq = np.array([sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=int(t), shunt_bus_idx=bus, q_step_mvar=sign*25.) for t in ids])
    voltage = net.res_bus.loc[obs, "vm_pu"].to_numpy().copy()
    q0 = net.res_trafo3w.loc[ids, "q_hv_mvar"].to_numpy().copy()
    target = q0+1.5*direction*dq
    b = make_bank(ShuntBankConfig(sh, bus, host, hv.net_id, kind,
        25., 2, .001, 5., 1800., 10, .9, 1.1,
        t_persistence_s=360., max_sample_gap_s=270.))
    b.level = initial_level
    nv, nq = len(obs), len(ids)
    f = STSShuntFeedback(ids, q0, target, q0-100, q0+100,
                        np.full(nq, 100.), 0.)
    model = BankModel(b, np.r_[dv, dq], ids, np.arange(nq), dq, np.zeros(nq), f)
    ctx = HybridContext(0., [model], np.zeros(nv+nq), np.arange(nv), voltage)
    p = MIQPProblem(nq, 0, nv+nq, .5, np.eye(nq), np.eye(nv+nq),
        np.zeros(nq), np.vstack([np.zeros((nv, nq)), np.eye(nq)]),
        target, target-100, target+100, np.r_[voltage, q0],
        np.r_[np.full(nv, .9), q0-200], np.r_[np.full(nv, 1.1), q0+200], [])
    solver = HybridShuntSolver(MIQPSolver(), lambda: ctx, movement_weight=100.,
        support_config=STSShuntSupportConfig(tracking_weight=10, reserve_weight=0),
        voltage_error_allowance_pu=.02)
    for t in (0., 180., 360.):
        ctx.time_s = t
        model.feedback = replace(f, observed_at_s=t)
        result = solver.solve(p)
        if t < 360:
            assert solver.take_commit() is None
    commit, affected, step = solver.take_commit()
    assert commit.direction == direction and commit.request_kind == "sts_tracking"
    assert np.array_equal(affected, ids)
    # Delegated request plus the newly transferred bank contribution stays fixed.
    assert np.allclose(p.u_current+p.alpha*result.w_continuous+step, target)
    net.shunt.at[sh, "step"] = commit.pp_step_new
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True)
    q1 = net.res_trafo3w.loc[ids, "q_hv_mvar"].to_numpy()
    assert np.linalg.norm(target-q1) < np.linalg.norm(target-q0)
    assert float((q1-q0) @ step) > 0
