"""
Integration tests for the TSOController horizontal tie-coordination hooks
(two-loop ΔV_ref design — no price term).

Verifies that:
  * ``receive_tie_coordination`` redirects the boundary-bus voltage setpoint
    (and rejects bad targets / unknown buses);
  * receiving coordination adds NO objective term — the gradient invariant
    ``_compute_objective_gradient == _compute_output_gradient @ H`` still holds
    (the price term was removed);
  * the Q_tie soft-cap band tightens the Q_tie output limits when configured.

Author: Manuel Schwenke / Claude Code
Date: 2026-06-25
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandapower as pp
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _build_net():
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, verbose=False,
    )
    return net, meta


def _make_controller(tie_line_indices=None, tie_endpoint_buses=None, q_tie_band=None):
    """Zone-2 TSOController whose only controls are PCC Q setpoints (mirrors
    tests/test_tso_output_gradient.py), optionally with a tie line + band."""
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from core.measurement import measure_zone_tso
    from controller.multi_tso_coordinator import ZoneDefinition
    from experiments.helpers.utils import _network_state
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build_net()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    pcc_trafos, obs_buses = [], []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            pcc_trafos.extend(list(hv.coupling_trafo_indices))
            obs_buses.extend(list(hv.bus_indices))
    assert len(pcc_trafos) > 0
    n_pcc, n_v = len(pcc_trafos), len(obs_buses)

    cfg_kw = dict(
        der_indices=[], pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[f"DSO_{i}" for i in range(n_pcc)],
        oltc_trafo_indices=[], shunt_bus_indices=[], shunt_q_steps_mvar=[],
        voltage_bus_indices=obs_buses, current_line_indices=[], gen_indices=[],
        v_setpoints_pu=np.full(n_v, 1.03), g_v=2.0, g_q_tso=1.5, g_res_der=0.0,
    )
    if tie_line_indices is not None:
        cfg_kw.update(tie_line_indices=tie_line_indices,
                      tie_line_endpoint_buses=tie_endpoint_buses,
                      g_q_tie=0.0, q_tie_band_mvar=q_tie_band)
    cfg = TSOControllerConfig(**cfg_kw)

    params = OFOParameters(
        alpha=1.0, g_w=np.full(n_pcc, 100.0),
        g_z=np.zeros(n_v + n_pcc + len(tie_line_indices or [])), g_u=np.zeros(n_pcc),
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
    sens = JacobianSensitivities(net)
    tso = TSOController("TSO_2", params, cfg, ns, bounds, sens)
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
    return tso, meas, obs_buses


def _tie_msg(target, bus, v_ref, tie_id=999):
    from core.message import TieCoordinationMessage
    return TieCoordinationMessage(
        source_controller_id="tie_coordinator", target_controller_id=target,
        iteration=0,
        tie_line_indices=np.array([tie_id], dtype=np.int64),
        boundary_bus_indices=np.array([bus], dtype=np.int64),
        v_ref_pu=np.array([v_ref], dtype=np.float64),
    )


# ── receive_tie_coordination ────────────────────────────────────────────

def test_receive_redirects_setpoint():
    tso, _, obs_buses = _make_controller()
    tso.receive_tie_coordination(_tie_msg("TSO_2", int(obs_buses[1]), 1.055))
    assert tso.config.v_setpoints_pu[1] == pytest.approx(1.055)
    assert tso.config.v_setpoints_pu[0] == pytest.approx(1.03)  # others untouched


def test_receive_wrong_target_raises():
    tso, _, obs_buses = _make_controller()
    with pytest.raises(ValueError):
        tso.receive_tie_coordination(_tie_msg("TSO_7", int(obs_buses[0]), 1.05))


def test_receive_unknown_boundary_bus_raises():
    tso, _, _ = _make_controller()
    with pytest.raises(ValueError):
        tso.receive_tie_coordination(_tie_msg("TSO_2", 999999, 1.05))


# ── gradient invariant: no hidden term from coordination ─────────────────

def test_receive_preserves_gradient_invariant():
    tso, meas, obs_buses = _make_controller()
    tso._u_current = tso._u_current + 7.5
    # Redirect a couple of boundary setpoints; this must NOT add any objective
    # term beyond the (existing) voltage tracking it feeds.
    tso.receive_tie_coordination(_tie_msg("TSO_2", int(obs_buses[1]), 1.05))
    grad_f = tso._compute_objective_gradient(meas)
    grad_y = tso._compute_output_gradient(meas)
    H = tso._expand_H_to_der_level(tso._build_sensitivity_matrix())
    np.testing.assert_allclose(grad_f, grad_y @ H, rtol=1e-9, atol=1e-9)


# ── Q_tie soft-cap band ──────────────────────────────────────────────────

def test_q_tie_band_tightens_output_limits():
    net, _ = _build_net()
    tie_line = int(net.line.index[0])
    endpoint = int(net.line.at[tie_line, "from_bus"])
    tso, _, _ = _make_controller(tie_line_indices=[tie_line],
                                 tie_endpoint_buses=[endpoint],
                                 q_tie_band=np.array([50.0]))
    y_lower, y_upper = tso._get_output_limits()
    assert y_lower[-1] == pytest.approx(-50.0)
    assert y_upper[-1] == pytest.approx(+50.0)


def test_no_band_leaves_q_tie_wide_open():
    net, _ = _build_net()
    tie_line = int(net.line.index[0])
    endpoint = int(net.line.at[tie_line, "from_bus"])
    tso, _, _ = _make_controller(tie_line_indices=[tie_line],
                                 tie_endpoint_buses=[endpoint], q_tie_band=None)
    y_lower, y_upper = tso._get_output_limits()
    assert y_lower[-1] <= -1e5 and y_upper[-1] >= 1e5


def test_report_tie_boundary_voltage():
    tso, meas, obs_buses = _make_controller()
    bus = int(obs_buses[0])
    v = tso.report_tie_boundary_voltage(meas, bus)
    midx = np.where(meas.bus_indices == bus)[0][0]
    assert v == pytest.approx(float(meas.voltage_magnitudes_pu[midx]))
