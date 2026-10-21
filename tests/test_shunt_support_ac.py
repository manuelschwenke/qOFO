"""Check support direction against an AC plant with continuous actuators held.

The request is constructed for this test. It does not represent an observed
failure of the subordinate MIQP. The plant power flow checks that a confirmed
step actually reduces the specified interface error for both device types.
"""
import numpy as np
import pandapower as pp
import pytest

from controller.shunt_integrator import ShuntBankConfig, make_bank
from controller.shunt_support import (
    STSShuntFeedback, STSShuntSupportConfig, support_request,
)
from network.ieee39.build import build_ieee39_net
from network.ieee39.hv_networks import add_hv_networks
from sensitivity.jacobian import JacobianSensitivities


@pytest.mark.parametrize("kind,sign", [("MSC", -1), ("MSR", 1)])
def test_confirmed_support_step_reduces_actual_ac_interface_error(kind, sign):
    net, meta = build_ieee39_net(scenario="rural_700", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, tso_shunt_kind="msc_msr",
        msc_n_levels=2, msr_n_levels=2, msc_q_step_mvar=25.,
        msr_q_step_mvar=25., verbose=False,
    )
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True)
    sh, bus = next(
        (int(s), int(b)) for s, b, k in zip(
            meta.tso_tertiary_shunt_indices, meta.tso_tertiary_shunt_buses,
            meta.tso_tertiary_shunt_kinds) if k == kind)
    hv = next(h for h in meta.hv_networks if bus in h.coupling_lv_bus_indices)
    ids = np.asarray(hv.coupling_trafo_indices, dtype=int)
    host = int(ids[list(hv.coupling_lv_bus_indices).index(bus)])
    sens = JacobianSensitivities(net)
    hq = np.array([sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=int(t), shunt_bus_idx=bus, q_step_mvar=sign) for t in ids])
    q0 = net.res_trafo3w.loc[ids, "q_hv_mvar"].to_numpy().copy()
    target = q0 + 1.5 * 25 * hq
    # This test isolates interface-Q assistance. Real measured EHV voltages
    # and model columns still pass through the normal bank feasibility guard.
    obs = [int(b) for b in net.bus.index
           if float(net.bus.at[b, "vn_kv"]) >= 220
           and b not in set(net.gen.bus) and b not in set(net.ext_grid.bus)]
    hv_col, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=bus, observation_bus_indices=obs, q_step_mvar=sign)
    volts = net.res_bus.loc[obs_map, "vm_pu"].to_numpy()
    b = make_bank(ShuntBankConfig(
        sh, bus, host, hv.net_id, kind, 25., 2, .001, 5., 1800., 10,
        .9, 1.1, t_persistence_s=360., max_sample_gap_s=270.))
    commit = None
    for t in (0., 180., 360.):
        f = STSShuntFeedback(ids, q0, target, q0-100, q0+100,
                             np.full(len(ids), 100.), t)
        r = support_request(f, h_q=hq, q_step_mvar=25., level=b.level,
            n_levels=2, t_now=t, config=STSShuntSupportConfig())
        assert r is not None and r.kind == "sts_tracking"
        commit = b.step(grad_g=r.gradient, v_meas=volts, h_v=hv_col,
                        t_now=t, request_kind=r.kind)
        if t < 360:
            assert commit is None
    assert commit is not None and commit.direction == 1, (
        r.gradient, b.q_eq_aux, volts.min(), volts.max(),
        (volts+np.asarray(hv_col)*25).min(), (volts+np.asarray(hv_col)*25).max())
    net.shunt.at[sh, "step"] = commit.pp_step_new
    pp.runpp(net, calculate_voltage_angles=True, distributed_slack=True)
    q1 = net.res_trafo3w.loc[ids, "q_hv_mvar"].to_numpy()
    assert np.linalg.norm(target-q1) < np.linalg.norm(target-q0)
    assert float((q1-q0) @ hq) > 0
