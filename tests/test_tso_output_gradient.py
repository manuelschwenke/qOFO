"""
Consistency test for the factored TSO output-space gradient.

The switched-shunt integrator dots ``_compute_output_gradient`` (∇_y f) with a
bank's boundary sensitivity column.  For that to mean "the same objective f" as
the MIQP, the control-space gradient used by the MIQP must equal the output
gradient projected through the sensitivity matrix:

    _compute_objective_gradient(meas)  ==  _compute_output_gradient(meas) @ H

(for a configuration without the direct DER-reserve term, which acts on a
control variable rather than an output).  This test pins that invariant so the
two code paths cannot silently diverge.

Author: Manuel Schwenke
Date: 2026-06-22
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandapower as pp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _build():
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, verbose=False,
    )
    return net, meta


def test_output_gradient_projects_to_objective_gradient():
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from core.measurement import measure_zone_tso
    from controller.multi_tso_coordinator import ZoneDefinition
    from experiments.helpers.utils import _network_state

    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Zone-2 PCC couplers and HV observation buses (no DER / gen / OLTC / shunt
    # controls → the only controls are the Q_PCC setpoints).
    pcc_trafos, obs_buses = [], []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            pcc_trafos.extend(list(hv.coupling_trafo_indices))
            obs_buses.extend(list(hv.bus_indices))
    assert len(pcc_trafos) > 0

    n_pcc = len(pcc_trafos)
    n_v = len(obs_buses)
    cfg = TSOControllerConfig(
        der_indices=[],
        pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[f"DSO_{i}" for i in range(n_pcc)],
        oltc_trafo_indices=[],
        shunt_bus_indices=[],            # integrator mode: no shunt in the MIQP
        shunt_q_steps_mvar=[],
        voltage_bus_indices=obs_buses,
        current_line_indices=[],
        gen_indices=[],
        v_setpoints_pu=np.full(n_v, 1.03),
        g_v=2.0,
        g_q_tso=1.5,                     # exercise both V and Q_PCC blocks
        g_res_der=0.0,                   # no direct DER-reserve term
    )
    params = OFOParameters(
        alpha=1.0,
        g_w=np.full(n_pcc, 100.0),
        g_z=np.zeros(n_v + n_pcc),
        g_u=np.zeros(n_pcc),
    )
    ns = _network_state(net)
    bounds = ActuatorBounds(
        der_indices=np.array([], dtype=np.int64),
        der_s_rated_mva=np.array([], dtype=np.float64),
        der_p_max_mw=np.array([], dtype=np.float64),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_min=np.array([], dtype=np.int64),
        oltc_tap_max=np.array([], dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_q_mvar=np.array([], dtype=np.float64),
    )
    from sensitivity.jacobian import JacobianSensitivities
    sens = JacobianSensitivities(net)
    tso = TSOController("tso_grad_test", params, cfg, ns, bounds, sens)

    zd = ZoneDefinition(
        zone_id=2, bus_indices=obs_buses, gen_indices=[], gen_bus_indices=[],
        tso_der_indices=[], tso_der_buses=[], v_bus_indices=obs_buses,
        line_indices=[], line_max_i_ka=[], pcc_trafo_indices=pcc_trafos,
        pcc_dso_ids=[f"DSO_{i}" for i in range(n_pcc)],
        shunt_bus_indices=[], shunt_q_steps_mvar=[],
    )
    meas = measure_zone_tso(net, zd, 1)
    tso.initialise(meas)
    tso._last_measurement = meas

    # A non-trivial control point so the Q_PCC tracking error is non-zero
    # (shift the Q_PCC setpoints away from the measured interface Q).
    tso._u_current = tso._u_current + 7.5

    grad_f = tso._compute_objective_gradient(meas)
    grad_y = tso._compute_output_gradient(meas)
    H = tso._expand_H_to_der_level(tso._build_sensitivity_matrix())

    # Output gradient must have one entry per output row of H.
    assert grad_y.shape[0] == H.shape[0]
    np.testing.assert_allclose(grad_f, grad_y @ H, rtol=1e-9, atol=1e-9)

    # And the V / Q_PCC blocks must be non-trivial (test is actually exercising
    # both terms, not the degenerate all-zero case).
    assert np.linalg.norm(grad_y[:n_v]) > 0.0
    assert np.linalg.norm(grad_y[n_v:n_v + n_pcc]) > 0.0
