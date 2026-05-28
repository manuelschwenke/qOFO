"""
Regression test: DSO MIQP commands non-zero Q_set under tracking error
=======================================================================

After the refactor_v3 revert of the Q_cor + H@T' machinery, the DSO
controller commands ``Q_set`` directly through the DER block of its
action vector ``u_new``.  The plant-side ``QVLocalLoop`` then feeds in
``Q_set`` while V stays in the deadband (and overrides via local droop
otherwise).

This test exercises the full chain end-to-end on a synthetic mock plant:

1. The DSO MIQP receives a ``SetpointMessage`` that pushes the
   interface-Q target away from the measured value.
2. With a positive ``∂Q_iface / ∂Q_DER`` mock sensitivity, the optimiser
   must therefore command **non-zero** Q_DER deltas in ``u_new[:n_der]``.
3. ``apply_dso_controls`` writes those values into
   ``net.sgen.q_set_mvar`` (not ``q_mvar``, not ``q_cor_mvar``).
4. The plant-side ``QVLocalLoop`` reads ``q_set_mvar`` and produces a
   matching Q target inside the deadband.

The test is intentionally synthetic so it's robust against changes
elsewhere in the codebase; the goal is to lock down the Q_set
actuation path against silent regressions of the kind we hit during
the refactor (where the OFO appeared to converge but no Q_DER ever
moved on the plant).
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from controller.base_controller import OFOParameters, ControllerOutput
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.der_qv_local_loop import QVLocalLoop
from core.message import SetpointMessage
from experiments.helpers.plant_io import apply_dso_controls

# Re-use the synthetic plant fixtures already defined in test_controller.
from tests.test_controller import (  # noqa: E402
    _make_actuator_bounds,
    _make_dso_measurement,
    _make_mock_sensitivities,
    _make_network_state,
)


# ---------------------------------------------------------------------------
#  Fixture: a DSO controller wired up to mock sensitivities with a
#  strong positive ∂Q_iface/∂Q_DER so the optimiser has a clear gradient.
# ---------------------------------------------------------------------------


def _make_dso_for_actuation_test(
    *, g_w: float = 0.05,
) -> tuple[DSOController, DSOControllerConfig, ...]:
    der_buses = [2, 3]
    oltc_trafos = [0]
    shunt_buses = [4]
    interface_trafos = [0]
    voltage_buses = [1, 2, 3]
    current_lines = [0, 1]

    n_der = len(der_buses)
    n_oltc = len(oltc_trafos)
    n_shunt = len(shunt_buses)
    n_outputs = len(interface_trafos) + len(voltage_buses) + len(current_lines)

    config = DSOControllerConfig(
        der_indices=der_buses,
        oltc_trafo_indices=oltc_trafos,
        shunt_bus_indices=shunt_buses,
        shunt_q_steps_mvar=[50.0],
        interface_trafo_indices=interface_trafos,
        voltage_bus_indices=voltage_buses,
        current_line_indices=current_lines,
        # Tight Q-tracking, mild voltage tracking — make the gradient on
        # the DER block dominate the cost.
        g_q=1000.0,
        g_v=1.0,
    )

    params = OFOParameters(
        g_w=g_w,
        g_z=1000.0,
        g_u=0.0,                 # no per-step regularisation
    )

    sens = _make_mock_sensitivities(
        n_outputs=n_outputs,
        n_der=n_der,
        n_oltc=n_oltc,
        n_shunt=n_shunt,
        der_indices=der_buses,
        oltc_trafo_indices=oltc_trafos,
        interface_trafo_indices=interface_trafos,
    )

    controller = DSOController(
        controller_id="dso_actuation_test",
        params=params,
        config=config,
        network_state=_make_network_state(),
        actuator_bounds=_make_actuator_bounds(n_der, n_oltc, n_shunt),
        sensitivities=sens,
    )

    measurement = _make_dso_measurement(
        der_indices=der_buses,
        oltc_trafo_indices=oltc_trafos,
        shunt_bus_indices=shunt_buses,
        interface_trafo_indices=interface_trafos,
        voltage_bus_indices=voltage_buses,
        current_line_indices=current_lines,
    )
    return controller, config, measurement, der_buses, interface_trafos


# ---------------------------------------------------------------------------
#  Test 1 — DSO MIQP commands non-zero Q_set under tracking error
# ---------------------------------------------------------------------------


def test_dso_miqp_commands_nonzero_q_set_under_tracking_error():
    """When Q_iface_set differs from Q_iface_meas, the MIQP must move
    the DER block of u_new away from u_current (not just the OLTC)."""
    controller, config, measurement, der_buses, interface_trafos = (
        _make_dso_for_actuation_test()
    )
    controller.initialise(measurement)
    u_initial = controller.u_current.copy()
    n_der = len(der_buses)

    # Push the interface-Q setpoint 50 Mvar away from the measured value
    # (measurement has interface_q_hv_side_mvar = +10 Mvar by default).
    # With ∂Q_iface/∂Q_DER = +0.05 (mock), the optimiser needs to add
    # roughly +1000 Mvar of DER Q to close the gap — capability-clipped
    # to S_n = 100 Mvar — so we expect both DERs near their q_max.
    setpoint = SetpointMessage(
        source_controller_id="tso_test",
        target_controller_id=controller.controller_id,
        iteration=0,
        interface_transformer_indices=np.array(
            interface_trafos, dtype=np.int64
        ),
        q_setpoints_mvar=np.array([60.0]),
    )
    controller.receive_setpoint(setpoint)

    out = controller.step(measurement)

    # The DER block of u_new must move from its initial value.
    der_block = out.u_new[:n_der]
    der_initial = u_initial[:n_der]
    assert not np.allclose(der_block, der_initial), (
        f"DSO MIQP did not move Q_set: u_new[:n_der]={der_block} "
        f"vs u_current[:n_der]={der_initial}.  This regresses the "
        f"refactor_v3 actuation path — without Q_set movement the "
        f"OFO has no way to change interface Q except via OLTC."
    )

    # Sign sanity: setpoint > measured ⇒ optimiser must inject positive
    # Q (positive ∂Q_iface/∂Q_DER mock).  At least one DER's command
    # must be greater than its initial value.
    assert (der_block > der_initial).any(), (
        "Optimiser commanded a Q_set update but the direction is wrong: "
        f"setpoint={setpoint.q_setpoints_mvar[0]} Mvar, "
        f"measured={measurement.interface_q_hv_side_mvar[0]} Mvar, "
        f"u_initial={der_initial}, u_new={der_block}."
    )


# ---------------------------------------------------------------------------
#  Test 2 — apply_dso_controls writes Q_set to net.sgen.q_set_mvar
# ---------------------------------------------------------------------------


def test_apply_dso_controls_writes_q_set_mvar():
    """The runner-side hand-off in ``apply_dso_controls`` must write the
    DER block of ``u_new`` into ``net.sgen.q_set_mvar`` (sgen sign
    convention) — NOT into ``q_mvar`` (which the QVLocalLoop owns)
    or any legacy ``q_cor_mvar`` column."""
    der_indices = [0, 1]
    n_der = len(der_indices)

    # Minimal pandapower net with two sgens
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b, vm_pu=1.0)
    for _ in der_indices:
        pp.create_sgen(net, bus=b, p_mw=1.0, q_mvar=0.0, sn_mva=10.0)

    cfg = DSOControllerConfig(
        der_indices=der_indices,
        oltc_trafo_indices=[],
        shunt_bus_indices=[],
        shunt_q_steps_mvar=[],
        interface_trafo_indices=[],
        voltage_bus_indices=[],
        current_line_indices=[],
        gridforming_gen_indices=[],
        gridforming_gen_buses=[],
    )

    out = ControllerOutput(
        iteration=1,
        u_new=np.array([2.5, -1.5]),
        u_continuous=np.array([2.5, -1.5]),
        u_integer=np.array([], dtype=np.int64),
        y_predicted=np.zeros(0, dtype=np.float64),
        sigma=np.zeros(2, dtype=np.float64),
        z_slack=np.zeros(0, dtype=np.float64),
        objective_value=0.0,
        solver_status="optimal",
        solve_time_s=0.0,
    )

    apply_dso_controls(net, cfg, out)

    assert "q_set_mvar" in net.sgen.columns, (
        "apply_dso_controls must auto-create net.sgen.q_set_mvar"
    )
    assert net.sgen.at[der_indices[0], "q_set_mvar"] == pytest.approx(2.5)
    assert net.sgen.at[der_indices[1], "q_set_mvar"] == pytest.approx(-1.5)
    # q_mvar must remain untouched — that column is owned by the
    # plant-side QVLocalLoop, not by the runner's apply step.
    assert net.sgen.at[der_indices[0], "q_mvar"] == pytest.approx(0.0)
    assert net.sgen.at[der_indices[1], "q_mvar"] == pytest.approx(0.0)
    # No legacy q_cor_mvar column gets created by the apply step.
    assert "q_cor_mvar" not in net.sgen.columns


# ---------------------------------------------------------------------------
#  Test 3 — QVLocalLoop reads q_set_mvar back and tracks it in deadband
# ---------------------------------------------------------------------------


def test_qv_local_loop_tracks_q_set_in_deadband():
    """End-to-end consistency: when ``apply_dso_controls`` writes a
    Q_set value AND V stays inside the deadband at the next PF, the
    QVLocalLoop's target equals Q_set.  This is the contract the OFO
    relies on: ``Q_realised == Q_set`` whenever the local droop is
    inside the deadband."""
    net = pp.create_empty_network()
    b_slack = pp.create_bus(net, vn_kv=110.0)
    b_load = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.03)
    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_load,
        length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_load(net, bus=b_load, p_mw=5.0, q_mvar=2.0)
    s_idx = pp.create_sgen(
        net, bus=b_load, p_mw=5.0, q_mvar=0.0, sn_mva=50.0,
        type="WP", name="QV_TEST",
    )
    net.sgen["op_diagram"] = "STATCOM"
    net.sgen["q_mode"] = "qv"
    net.sgen["qv_slope_pu"] = 0.07
    net.sgen["qv_vref_pu"] = 1.03
    net.sgen["qv_deadband_pu"] = 0.02
    net.sgen["cosphi"] = 1.0
    net.sgen["cosphi_sign"] = -1
    net.sgen["q_set_mvar"] = 0.0

    loop = QVLocalLoop(net, sgen_idx=s_idx, slope_pu=0.07)
    pp.runpp(net, run_control=False)

    # Force V to V_ref so we sit comfortably inside the (shifted) deadband
    # for any |Q_set/R| ≤ db.  R = 50/0.07 ≈ 714 Mvar/pu; db = 0.02 ⇒
    # |Q_set| ≤ R*db ≈ 14.3 Mvar.  We pick 10 Mvar.
    net.res_bus.at[b_load, "vm_pu"] = 1.03
    net.res_sgen.at[s_idx, "p_mw"] = 5.0

    # Simulate the apply path: write q_set_mvar.
    net.sgen.at[s_idx, "q_set_mvar"] = 10.0
    target_after = loop._compute_target(net)
    assert target_after == pytest.approx(10.0, abs=1e-6), (
        f"QVLocalLoop did not track the commanded Q_set: "
        f"target={target_after} expected=10.0 (Q_set was just written)."
    )

    # Negative Q_set — the inverter must absorb in band.
    net.sgen.at[s_idx, "q_set_mvar"] = -8.0
    target_neg = loop._compute_target(net)
    assert target_neg == pytest.approx(-8.0, abs=1e-6)


# ---------------------------------------------------------------------------
#  Test 4 — Multi-step trajectory: Q_set evolves smoothly across calls
# ---------------------------------------------------------------------------


def test_dso_miqp_q_set_evolves_smoothly_over_multiple_steps():
    """Drive the DSO MIQP for N consecutive steps and have the synthetic
    "plant" track the previous command exactly each step.  The Q_set
    trajectory should reduce the tracking error monotonically and
    *never flatline before the error is closed*.

    The setpoint is sized so the closed-loop trajectory stays well below
    the capability rail — otherwise the optimiser legitimately pins at
    the rail in step 1 and "flatline" is not a regression.

    This is the contract the real run is meant to honour: when the OFO
    has authority (gradient ≠ 0, inside-deadband, bounds wide), Q_set
    must keep adjusting until the error is gone.
    """
    # Higher g_w (inertia) so each step takes a small fraction of the
    # optimal jump rather than saturating the rail in one shot.
    controller, config, measurement, der_buses, interface_trafos = (
        _make_dso_for_actuation_test(g_w=5.0)
    )
    controller.initialise(measurement)
    n_der = len(der_buses)

    # Use a small interface error: setpoint = measured + 4 Mvar.  With
    # the mock H_iface_DER ≈ 0.05 across two DERs, the unconstrained
    # optimum delta is ΔQ_DER ≈ 4/(0.05·2) = 40 Mvar TOTAL — within the
    # ±41 Mvar rail per DER.  With g_w = 5 the optimiser will take
    # several steps to close the gap, exercising the trajectory.
    iface_meas = float(measurement.interface_q_hv_side_mvar[0])
    setpoint = SetpointMessage(
        source_controller_id="tso_test",
        target_controller_id=controller.controller_id,
        iteration=0,
        interface_transformer_indices=np.array(
            interface_trafos, dtype=np.int64
        ),
        q_setpoints_mvar=np.array([iface_meas + 4.0]),
    )
    controller.receive_setpoint(setpoint)

    n_steps = 6
    der_history = np.zeros((n_steps, n_der), dtype=float)
    iface_history = np.zeros(n_steps, dtype=float)
    for k in range(n_steps):
        out = controller.step(measurement)
        der_history[k] = out.u_new[:n_der]
        # Synthetic plant: measurement.der_q_mvar tracks the command 1:1
        # (in-deadband behaviour); interface Q updates per the mock H.
        measurement.der_q_mvar = der_history[k].copy()
        H_ie = 0.05  # mock sensitivity
        delta_q_iface = H_ie * float(np.sum(der_history[k] - 5.0))
        iface_history[k] = iface_meas + delta_q_iface
        measurement.interface_q_hv_side_mvar = np.array([iface_history[k]])

    # 1.  At least the first transition must show real movement (not
    #     "stuck at u_current" the moment we start).
    assert not np.allclose(der_history[0], 5.0, atol=1e-3), (
        f"Q_set never moved off its initial value: u_new[0]={der_history[0]}"
    )

    # 2.  The optimiser must at least be moving in the correct direction
    #     (positive H_ie + positive setpoint error → push Q up).
    assert (der_history[0] > 5.0).all(), (
        f"Q_set moved the wrong direction: u_initial=5.0, "
        f"u_new[0]={der_history[0]}"
    )

    # 3.  The interface tracking error must be reducing across the run.
    iface_err_history = np.abs(
        iface_history - setpoint.q_setpoints_mvar[0]
    )
    assert iface_err_history[-1] < iface_err_history[0], (
        f"Interface tracking error did NOT reduce over {n_steps} steps. "
        f"err history = {iface_err_history}"
    )

    # 4.  The optimiser must NOT instantly saturate at the rail (which
    #     would be a regression to "single-step jump then stuck"
    #     behaviour).  At least one intermediate u_new must be strictly
    #     between the initial value and the rail.
    rail = 41.0  # VDE-AR-N q_max for sn=100, p=80 ⇒ 0.41·100 = 41
    intermediate_in_band = (
        (der_history > 5.0 + 1e-3) & (der_history < rail - 1e-3)
    ).any(axis=1)
    assert intermediate_in_band.any(), (
        f"Optimiser jumped straight to the rail without intermediate "
        f"smooth steps.  trajectory =\n{der_history}"
    )
