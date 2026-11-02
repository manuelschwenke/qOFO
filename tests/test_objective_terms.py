"""
Consistency of the plant-level objective Φ with the controllers' gradients.

``objective_terms`` evaluates Φ(y^k, u^k) for the live plot and the record;
``_compute_output_gradient`` / ``_compute_objective_gradient`` build the
gradient the MIQP descends.  Both encode the same terms and weights in two
places, so this test pins them together: the finite-difference gradient of Φ
in the outputs must equal the analytic output gradient (TSO), and projected
through H it must equal the control-space gradient (DSO, γ = 1).

Author: Manuel Schwenke / Claude Code
Date: 2026-11-02
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp

from tests.test_tso_output_gradient import _build


def _phi(ctrl, meas) -> float:
    terms = ctrl.objective_terms(meas)
    return sum(v for k, v in terms.items() if k != "u_reg")


def _fd(ctrl, meas, field: str, k: int, h: float) -> float:
    lo, hi = copy.deepcopy(meas), copy.deepcopy(meas)
    getattr(lo, field)[k] -= h
    getattr(hi, field)[k] += h
    return (_phi(ctrl, hi) - _phi(ctrl, lo)) / (2.0 * h)


def test_tso_objective_terms_match_output_gradient():
    from controller.base_controller import OFOParameters
    from controller.multi_tso_coordinator import ZoneDefinition
    from controller.tso_controller import TSOController, TSOControllerConfig
    from core.actuator_bounds import ActuatorBounds
    from core.measurement import measure_zone_tso
    from experiments.helpers.utils import _network_state
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             distributed_slack=True)
    pcc_trafos, obs_buses = [], []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            pcc_trafos.extend(list(hv.coupling_trafo_indices))
            obs_buses.extend(list(hv.bus_indices))
    n_pcc, n_v = len(pcc_trafos), len(obs_buses)
    g_u = np.linspace(0.1, 0.5, n_pcc)
    cfg = TSOControllerConfig(
        der_indices=[], pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[f"DSO_{i}" for i in range(n_pcc)],
        oltc_trafo_indices=[], shunt_bus_indices=[], shunt_q_steps_mvar=[],
        voltage_bus_indices=obs_buses, current_line_indices=[], gen_indices=[],
        v_setpoints_pu=np.full(n_v, 1.03), g_v=2.0, g_q_tso=1.5, g_res_der=0.0,
    )
    params = OFOParameters(alpha=1.0, g_w=np.full(n_pcc, 100.0),
                           g_z=np.zeros(n_v + n_pcc), g_u=g_u)
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
    tso = TSOController("tso_phi_test", params, cfg, _network_state(net),
                        bounds, JacobianSensitivities(net))
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
    tso._u_current = tso._u_current + 7.5

    terms = tso.objective_terms(meas)
    q_meas = meas.interface_q_hv_side_mvar
    np.testing.assert_allclose(
        terms["q_pcc"], 1.5 * np.sum((q_meas - tso._u_current[:n_pcc]) ** 2))
    np.testing.assert_allclose(
        terms["u_reg"], np.sum(g_u * tso._u_current ** 2))
    assert terms["v"] > 0.0 and terms["q_pcc"] > 0.0

    grad_y = tso._compute_output_gradient(meas)
    pos = {int(b): i for i, b in enumerate(meas.bus_indices)}
    for j in (0, n_v // 2, n_v - 1):
        fd = _fd(tso, meas, "voltage_magnitudes_pu", pos[obs_buses[j]], 1e-6)
        np.testing.assert_allclose(fd, grad_y[j], rtol=1e-5, atol=1e-8)
    for k in range(n_pcc):
        fd = _fd(tso, meas, "interface_q_hv_side_mvar", k, 1e-4)
        np.testing.assert_allclose(fd, grad_y[n_v + k], rtol=1e-6, atol=1e-8)


def test_dso_objective_terms_project_to_objective_gradient():
    from controller.base_controller import OFOParameters
    from controller.dso_controller import DSOController, DSOControllerConfig
    from tests.test_controller import (
        _make_actuator_bounds, _make_dso_measurement,
        _make_mock_sensitivities, _make_network_state,
    )

    der_buses, oltc_trafos, shunt_buses = [2, 3], [0], [4]
    ifaces, v_buses, lines = [0], [1, 2, 3], [0, 1]
    cfg = DSOControllerConfig(
        der_indices=der_buses, oltc_trafo_indices=oltc_trafos,
        shunt_bus_indices=shunt_buses, shunt_q_steps_mvar=[50.0],
        interface_trafo_indices=ifaces, voltage_bus_indices=v_buses,
        current_line_indices=lines,
        g_q=3.0, g_v=40.0, gamma_oltc_q=1.0,
        v_setpoints_pu=np.array([1.00, 1.01, 1.03]),
    )
    dso = DSOController(
        controller_id="dso_phi_test",
        params=OFOParameters(g_w=0.2, g_z=1000.0, g_u=0.01),
        config=cfg,
        network_state=_make_network_state(),
        actuator_bounds=_make_actuator_bounds(2, 1, 1),
        sensitivities=_make_mock_sensitivities(
            n_outputs=len(ifaces) + len(v_buses) + len(lines),
            n_der=2, n_oltc=1, n_shunt=1, der_indices=der_buses,
            oltc_trafo_indices=oltc_trafos, interface_trafo_indices=ifaces,
        ),
    )
    meas = _make_dso_measurement(
        der_indices=der_buses, oltc_trafo_indices=oltc_trafos,
        shunt_bus_indices=shunt_buses, interface_trafo_indices=ifaces,
        voltage_bus_indices=v_buses, current_line_indices=lines,
    )
    dso.initialise(meas)
    dso.q_setpoint_mvar = np.array([-4.0])

    terms = dso.objective_terms(meas)
    np.testing.assert_allclose(terms["q_iface"], 3.0 * (10.0 + 4.0) ** 2)
    assert terms["v"] > 0.0

    pos = {int(b): i for i, b in enumerate(meas.bus_indices)}
    grad_y = np.concatenate([
        [_fd(dso, meas, "interface_q_hv_side_mvar", 0, 1e-4)],
        [_fd(dso, meas, "voltage_magnitudes_pu", pos[b], 1e-6) for b in v_buses],
        np.zeros(len(lines)),
    ])
    H = dso._expand_H_to_der_level(dso._build_sensitivity_matrix())
    np.testing.assert_allclose(
        grad_y @ H, dso._compute_objective_gradient(meas), rtol=1e-5, atol=1e-8)
