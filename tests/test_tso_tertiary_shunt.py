"""
Tests for TSO-owned bipolar shunts at DSO tertiary windings.

Verifies:

1. Pandapower bipolar step writes (-1 / 0 / +1) round-trip through the
   ``net.shunt['step']`` column and produce the expected ``Q_inj``
   under ``pp.runpp``.
2. ``add_hv_networks(install_tso_tertiary_shunts=True)`` installs one
   shunt per active DSO sub-network at the first tertiary bus and
   populates ``IEEE39NetworkMeta.tso_tertiary_shunt_*`` correctly.
3. The DSO controller config keeps ``shunt_bus_indices=[]`` after the
   change — the DSO sees the new shunts only as disturbances, not as
   control variables.
4. ``apply_zone_tso_controls`` writes the new step into ``net.shunt``
   and returns the previous step list so the caller can detect changes.
5. The Sherman–Morrison rank-1 update of ``dV_dQ_reduced`` /
   ``J_inv`` (``apply_shunt_step_change_smw``) preserves the cached
   operating point and matches a full Jacobian rebuild to numerical
   precision.
6. Routing through ``ShuntDisturbanceMessage`` updates the DSO's
   cached Jacobian without re-measuring or running ``pp.runpp``.

Author: Manuel Schwenke
Date: 2026-04-27
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandapower as pp
import pytest

# Allow tests to import project modules from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.message import ShuntDisturbanceMessage
from sensitivity.jacobian import JacobianSensitivities


# =====================================================================
#  Test 1 — pandapower bipolar step round-trip
# =====================================================================

def test_pp_bipolar_step():
    """Step ∈ {-1, 0, +1} on a 50 Mvar shunt → res_shunt.q_mvar ≈ 50·step·V²."""
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.0, name="t")
    pp.create_ext_grid(net, bus=b, vm_pu=1.0)
    sh = pp.create_shunt(
        net, bus=b,
        q_mvar=50.0, p_mw=0.0, vn_kv=20.0,
        step=0, max_step=1,
    )
    # Defensive cast (the runner does the same)
    if "step" in net.shunt.columns:
        net.shunt["step"] = net.shunt["step"].astype("int64")

    for s in [-1, 0, 1]:
        net.shunt.at[sh, "step"] = int(s)
        pp.runpp(net)
        v_pu = float(net.res_bus.at[b, "vm_pu"])
        q_obs = float(net.res_shunt.at[sh, "q_mvar"])
        q_expect = 50.0 * s * v_pu * v_pu
        assert abs(q_obs - q_expect) < 0.1, (
            f"step={s}: expected Q ≈ {q_expect:.3f} Mvar, got {q_obs:.3f}"
        )


# =====================================================================
#  Helpers for full IEEE 39 build
# =====================================================================

def _build_ieee39_with_shunts():
    """Build IEEE 39 with HV sub-networks and 4 TSO-owned tertiary shunts."""
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=True,
        tso_tertiary_shunt_q_mvar=50.0,
        verbose=False,
    )
    return net, meta


# =====================================================================
#  Test 2 — shunts installed at correct tertiary buses
# =====================================================================

def test_shunts_built_at_correct_buses():
    net, meta = _build_ieee39_with_shunts()

    n_dso = len(meta.hv_networks)
    assert len(meta.tso_tertiary_shunt_indices) == n_dso, (
        f"Expected {n_dso} shunts (one per DSO), got "
        f"{len(meta.tso_tertiary_shunt_indices)}"
    )
    assert (
        len(meta.tso_tertiary_shunt_buses)
        == len(meta.tso_tertiary_shunt_q_steps_mvar)
        == len(meta.tso_tertiary_shunt_zones)
        == n_dso
    )

    for i, hv in enumerate(meta.hv_networks):
        sh_idx = meta.tso_tertiary_shunt_indices[i]
        sh_bus = meta.tso_tertiary_shunt_buses[i]
        first_trafo3w = hv.coupling_trafo_indices[0]
        first_lv_bus = int(net.trafo3w.at[first_trafo3w, "lv_bus"])

        assert sh_bus == first_lv_bus, (
            f"DSO {hv.net_id}: shunt bus {sh_bus} ≠ first tertiary "
            f"{first_lv_bus}"
        )
        assert float(net.bus.at[sh_bus, "vn_kv"]) == 20.0
        assert float(net.shunt.at[sh_idx, "q_mvar"]) == 50.0
        assert int(net.shunt.at[sh_idx, "step"]) == 0
        assert meta.tso_tertiary_shunt_zones[i] == hv.zone


# =====================================================================
#  Test 3 — installation can be disabled
# =====================================================================

def test_install_flag_disables_shunts():
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=False,
        verbose=False,
    )
    assert len(meta.tso_tertiary_shunt_indices) == 0
    assert len(meta.tso_tertiary_shunt_buses) == 0


# =====================================================================
#  Test 4 — Q changes at PCC when shunt switches (sanity check)
# =====================================================================

def test_shunt_q_iface_delta_at_pcc():
    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Pick the first DSO's first 3W coupler and its tertiary shunt
    hv = meta.hv_networks[0]
    t_first = hv.coupling_trafo_indices[0]
    sh_idx = meta.tso_tertiary_shunt_indices[0]
    q0 = float(net.res_trafo3w.at[t_first, "q_hv_mvar"])

    deltas = {}
    for s in (+1, -1):
        net.shunt.at[sh_idx, "step"] = int(s)
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)
        q_s = float(net.res_trafo3w.at[t_first, "q_hv_mvar"])
        deltas[s] = q_s - q0
        # Restore for next iteration
        net.shunt.at[sh_idx, "step"] = 0
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Each step is ~50 Mvar at the tertiary.  The Q split between the HV
    # and MV windings of the 3W coupler depends on local impedances and
    # how much the HV (110 kV) sub-network absorbs through line losses /
    # parallel couplers — so the |ΔQ| at the HV port lands around 25-45
    # Mvar in this case.  Demand at least 20 Mvar so the test catches a
    # degenerate "no effect" outcome but does not over-constrain the
    # physical redistribution.
    assert abs(deltas[+1]) > 20.0, (
        f"Δq_iface at step=+1 too small: {deltas[+1]:.2f} Mvar"
    )
    assert abs(deltas[-1]) > 20.0, (
        f"Δq_iface at step=-1 too small: {deltas[-1]:.2f} Mvar"
    )
    # Opposite signs for opposite steps
    assert np.sign(deltas[+1]) == -np.sign(deltas[-1])


# =====================================================================
#  Test 5 — apply_zone_tso_controls writes step + returns prev_steps
# =====================================================================

def test_apply_zone_tso_controls_writes_step():
    from controller.multi_tso_coordinator import ZoneDefinition
    from experiments.helpers.plant_io import apply_zone_tso_controls

    net, meta = _build_ieee39_with_shunts()
    sh_bus = int(meta.tso_tertiary_shunt_buses[0])

    # Minimal zone def with just one shunt actuator (other actuator lists empty)
    zd = ZoneDefinition(
        zone_id=0,
        bus_indices=[],
        gen_indices=[],
        gen_bus_indices=[],
        tso_der_indices=[],
        tso_der_buses=[],
        v_bus_indices=[],
        line_indices=[],
        line_max_i_ka=[],
        pcc_trafo_indices=[],
        pcc_dso_ids=[],
        oltc_trafo_indices=[],
        shunt_bus_indices=[sh_bus],
        shunt_q_steps_mvar=[50.0],
    )

    class _Out:  # fake ControllerOutput
        def __init__(self, val):
            self.u_new = np.array([float(val)])

    # u_new layout for an empty zone with one shunt: u = [shunt_step]
    for s_target in (1, 0, -1):
        prev = apply_zone_tso_controls(net, zd, _Out(s_target))
        sh_idx = int(net.shunt.index[net.shunt["bus"] == sh_bus][0])
        assert int(net.shunt.at[sh_idx, "step"]) == int(s_target), (
            f"Step write failed: target {s_target}, got "
            f"{int(net.shunt.at[sh_idx, 'step'])}"
        )
        # prev list has length 1
        assert isinstance(prev, list) and len(prev) == 1
        # Try a runpp to confirm the network still solves with this step
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)


# =====================================================================
#  Test 6 — Sherman–Morrison rank-1 update vs. full rebuild
# =====================================================================

def test_smw_matches_direct_reinversion():
    """SMW-updated dV_dQ_reduced must match a direct re-inversion of the
    cached Jacobian at the SAME operating point with the (Q,V)_bb
    diagonal entry perturbed.

    This is the mathematically correct equivalence: SMW is exact for a
    rank-1 perturbation of the inverse.  The test deliberately does NOT
    re-run pp.runpp on the perturbed network — the operating point
    (V, θ) is held fixed and only the J entry changes.
    """
    from sensitivity.index_helper import get_jacobian_indices

    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    sh_bus = int(meta.tso_tertiary_shunt_buses[0])

    # Build initial sensitivity at step=0 (cached operating point)
    js_smw = JacobianSensitivities(net)

    # Cache the J before SMW perturbation so we can build the ground
    # truth by direct re-inversion at the same operating point.
    J_before = js_smw.J.copy()
    n_pv_pq = len(js_smw.pq_buses) + len(js_smw.pv_buses)
    _, v_idx = get_jacobian_indices(js_smw.net, sh_bus)
    assert v_idx is not None, "Shunt bus must be a PQ bus"
    n_b = n_pv_pq + int(v_idx)

    # SMW path: change step to +1 via the rank-1 update (no pp.runpp)
    applied = js_smw.apply_shunt_step_change_smw(sh_bus, +1)
    assert applied is True

    # Ground truth: take a fresh J = J_before, perturb the (Q,V)_bb
    # diagonal by the same ΔJ, recompute the reduced inverse from scratch.
    delta_J = float(js_smw.J[n_b, n_b] - J_before[n_b, n_b])
    assert delta_J != 0.0, "SMW did not modify J diagonal entry"

    J_gt = J_before.copy()
    J_gt[n_b, n_b] += delta_J
    J_Pt = J_gt[:n_pv_pq, :n_pv_pq]
    J_PV = J_gt[:n_pv_pq, n_pv_pq:]
    J_Qt = J_gt[n_pv_pq:, :n_pv_pq]
    J_QV = J_gt[n_pv_pq:, n_pv_pq:]
    schur = J_QV - J_Qt @ np.linalg.inv(J_Pt) @ J_PV
    Jr_gt = np.linalg.inv(schur)

    diff = js_smw.dV_dQ_reduced - Jr_gt
    err = float(np.max(np.abs(diff)))
    rel = err / float(max(np.max(np.abs(Jr_gt)), 1e-12))
    # SMW is mathematically exact for rank-1 perturbations; expect
    # numerical-precision-level agreement (1e-9 typical, conservative cap 1e-6).
    assert rel < 1e-6, (
        f"SMW vs. direct re-inversion mismatch: rel={rel:.3e}, abs={err:.3e}"
    )


# =====================================================================
#  Test 7 — disturbance message refreshes DSO Jacobian without runpp
# =====================================================================

def test_disturbance_message_smw_updates_jacobian_no_runpp():
    """End-to-end ShuntDisturbanceMessage path on a real
    JacobianSensitivities instance, with the DSOController.receive_*
    method invoked via a lightweight stand-in object.  Verifies:

    1. ``apply_shunt_step_change_smw`` updates ``dV_dQ_reduced`` (rank-1).
    2. The cached operating point (V, θ) is preserved.
    3. The cached shunt step in ``net.shunt`` reflects the new state.
    4. No ``pp.runpp`` is called during the handler (we count
       ``net._ppc`` identity to detect a re-solve).
    """
    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    sh_bus = int(meta.tso_tertiary_shunt_buses[0])
    sens = JacobianSensitivities(net)

    # Snapshot the cached operating point (V, θ) and dV_dQ_reduced
    v_before = sens.net.res_bus["vm_pu"].to_numpy().copy()
    th_before = sens.net.res_bus["va_degree"].to_numpy().copy()
    ppc_id_before = id(sens.net._ppc)
    Jr_before = sens.dV_dQ_reduced.copy()

    # Stand-in DSO controller with the receive_disturbance_message logic.
    # We mirror the real method — it lives on DSOController and depends
    # only on .sensitivities + .invalidate_sensitivity_cache.
    class _Stub:
        def __init__(self):
            self.sensitivities = sens
            self.controller_id = "DSO_test"
            self._H_cache = "DUMMY_NOT_NONE"  # so we can detect the clear
            self.invalidated = False
        def invalidate_sensitivity_cache(self):
            self._H_cache = None
            self.invalidated = True

    stub = _Stub()
    # Apply the same logic that DSOController.receive_disturbance_message uses
    msg = ShuntDisturbanceMessage(
        source_controller_id="tso_zone_0",
        target_controller_id="DSO_test",
        iteration=42,
        shunt_bus_indices=np.array([sh_bus], dtype=np.int64),
        shunt_steps=np.array([1], dtype=np.int64),
        shunt_q_steps_mvar=np.array([50.0], dtype=np.float64),
    )
    any_applied = False
    for bus, step in zip(msg.shunt_bus_indices, msg.shunt_steps):
        if stub.sensitivities.apply_shunt_step_change_smw(int(bus), int(step)):
            any_applied = True
    if any_applied:
        stub.invalidate_sensitivity_cache()

    # 1. SMW was applied
    assert any_applied is True
    # 2. H cache cleared
    assert stub._H_cache is None
    assert stub.invalidated is True
    # 3. Operating point UNCHANGED
    np.testing.assert_array_equal(
        sens.net.res_bus["vm_pu"].to_numpy(), v_before,
    )
    np.testing.assert_array_equal(
        sens.net.res_bus["va_degree"].to_numpy(), th_before,
    )
    # 4. No pp.runpp re-solve (net._ppc is not regenerated)
    assert id(sens.net._ppc) == ppc_id_before, (
        "Operating point cache was rebuilt — pp.runpp must not be called"
    )
    # 5. dV_dQ_reduced shifted (rank-1 perturbation)
    diff = sens.dV_dQ_reduced - Jr_before
    assert np.max(np.abs(diff)) > 0.0, "dV_dQ_reduced did not change"
    # 6. Cached shunt step now reflects the new state
    sh_idx = int(sens.net.shunt.index[sens.net.shunt["bus"] == sh_bus][0])
    assert int(sens.net.shunt.at[sh_idx, "step"]) == 1


def test_dso_receive_disturbance_message_method_directly():
    """Verify the actual DSOController.receive_disturbance_message
    method on a real controller built via the production constructor."""
    from controller.dso_controller import DSOController, DSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from experiments.helpers.utils import _network_state

    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    hv = meta.hv_networks[0]
    sh_bus = int(meta.tso_tertiary_shunt_buses[0])
    der_idx = list(hv.sgen_indices[:1])
    oltc_idx = list(hv.coupling_trafo_indices[:1])

    dso_cfg = DSOControllerConfig(
        der_indices=der_idx,
        oltc_trafo_indices=oltc_idx,
        shunt_bus_indices=[],
        shunt_q_steps_mvar=[],
        interface_trafo_indices=list(hv.coupling_trafo_indices),
        voltage_bus_indices=list(hv.bus_indices),
        current_line_indices=[],
    )
    params = OFOParameters(alpha=0.5, g_w=1.0, g_z=1.0)
    ns = _network_state(net)
    # Match the ActuatorBounds signature used elsewhere in tests
    bounds = ActuatorBounds(
        der_indices=np.array(der_idx, dtype=np.int64),
        der_s_rated_mva=np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in der_idx]
        ),
        der_p_max_mw=np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in der_idx]
        ),
        oltc_indices=np.array(oltc_idx, dtype=np.int64),
        oltc_tap_min=np.full(len(oltc_idx), -13, dtype=np.int64),
        oltc_tap_max=np.full(len(oltc_idx), +13, dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_q_mvar=np.array([], dtype=np.float64),
    )
    sens = JacobianSensitivities(net)
    dso = DSOController("DSO_test", params, dso_cfg, ns, bounds, sens)

    Jr_before = sens.dV_dQ_reduced.copy()
    v_before = sens.net.res_bus["vm_pu"].to_numpy().copy()
    ppc_id_before = id(sens.net._ppc)

    msg = ShuntDisturbanceMessage(
        source_controller_id="tso_zone_0",
        target_controller_id="DSO_test",
        iteration=42,
        shunt_bus_indices=np.array([sh_bus], dtype=np.int64),
        shunt_steps=np.array([1], dtype=np.int64),
        shunt_q_steps_mvar=np.array([50.0], dtype=np.float64),
    )
    dso.receive_disturbance_message(msg)

    # H cache cleared
    assert dso._H_cache is None
    # Operating point unchanged
    np.testing.assert_array_equal(
        sens.net.res_bus["vm_pu"].to_numpy(), v_before,
    )
    assert id(sens.net._ppc) == ppc_id_before
    # dV_dQ_reduced updated
    assert np.max(np.abs(sens.dV_dQ_reduced - Jr_before)) > 0.0
    # Step persisted
    sh_idx = int(sens.net.shunt.index[sens.net.shunt["bus"] == sh_bus][0])
    assert int(sens.net.shunt.at[sh_idx, "step"]) == 1


# =====================================================================
#  Test 8 — DSO config blind to TSO shunts (no shunt columns in H)
# =====================================================================

def test_dso_blind_to_tso_shunt_as_control():
    """DSOControllerConfig.shunt_bus_indices must remain empty so the
    DSO H matrix has no shunt columns even when TSO shunts exist in
    the plant."""
    from controller.dso_controller import DSOControllerConfig

    net, meta = _build_ieee39_with_shunts()
    hv = meta.hv_networks[0]
    cfg = DSOControllerConfig(
        der_indices=list(hv.sgen_indices),
        oltc_trafo_indices=list(hv.coupling_trafo_indices),
        shunt_bus_indices=[],
        shunt_q_steps_mvar=[],
        interface_trafo_indices=list(hv.coupling_trafo_indices),
        voltage_bus_indices=list(hv.bus_indices),
        current_line_indices=[],
    )
    assert cfg.shunt_bus_indices == []
    assert cfg.shunt_q_steps_mvar == []


# =====================================================================
#  Test 9 — sign convention of ∂V/∂s_shunt and the V-tracking gradient
# =====================================================================

def test_shunt_v_gradient_sign():
    """Sign-trace from physics → Jacobian column → MIQP V-tracking gradient.

    The chain that determines whether the MIQP picks the *correct*
    shunt switching direction is:

        ∂V_i/∂s_shunt  =  -∂V_i/∂Q_inj(shunt_bus) · q_step · V_pu²
                       <  0   (more step → lower V)

        ∂f/∂s = 2·g_v · Σ_i (V_i - V_set) · ∂V_i/∂s_shunt
              = 2·g_v · (V_err @ dV_dshunt_col)

    Under *undervoltage* (V < V_set), the gradient must be POSITIVE so
    the MIQP wants to DECREASE s (moving the shunt toward more
    capacitive, i.e. toward −1).

    Under *overvoltage* (V > V_set), the gradient must be NEGATIVE so
    the MIQP wants to INCREASE s (toward off / inductive).

    A sign error anywhere along this chain — in the load-convention
    flip, the V² scaling, or the column placement in H — would invert
    the optimiser's direction for shunt switching after a contingency.
    """
    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    sh_bus = int(meta.tso_tertiary_shunt_buses[0])
    hv = meta.hv_networks[0]
    # Pick observation buses on the HV (110 kV) sub-network — these are
    # PQ buses (no PV gens) so the Jacobian framework can compute
    # ∂V/∂Q at them.  Use the bus where the shunt's parent coupler's
    # MV winding lands (electrically closest non-tertiary bus).
    obs_buses = list(hv.bus_indices)

    sens = JacobianSensitivities(net)
    dV_dshunt, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=sh_bus,
        observation_bus_indices=obs_buses,
        q_step_mvar=50.0,
    )
    assert len(obs_map) > 0, "No PQ observation buses near the shunt"
    # 1) Direction of ∂V/∂s_shunt at every observation bus must be < 0.
    for v_bus, sens_val in zip(obs_map, dV_dshunt):
        assert sens_val < 0.0, (
            f"∂V/∂s_shunt at bus {v_bus} = {sens_val:.6e} should be "
            f"negative (increasing step → less Q injection → V drops)"
        )

    # 2) MIQP V-tracking gradient sign:
    #    Undervoltage (uniform V_err = -0.05): sum should be POSITIVE
    v_err_under = np.full(len(dV_dshunt), -0.05, dtype=np.float64)
    grad_under = float(v_err_under @ dV_dshunt)
    assert grad_under > 0.0, (
        f"Undervoltage V-tracking gradient should be > 0 (MIQP wants "
        f"to decrease s, i.e. keep / increase capacitive support); "
        f"got {grad_under:.6e}"
    )

    # 3) Overvoltage (uniform V_err = +0.05): sum should be NEGATIVE
    v_err_over = np.full(len(dV_dshunt), +0.05, dtype=np.float64)
    grad_over = float(v_err_over @ dV_dshunt)
    assert grad_over < 0.0, (
        f"Overvoltage V-tracking gradient should be < 0 (MIQP wants "
        f"to increase s, i.e. drop the capacitor toward 0 / +1); "
        f"got {grad_over:.6e}"
    )

    # 4) The two gradients must have opposite sign (sanity: signs come
    #    only from the V_err vector, not the column).
    assert grad_under * grad_over < 0.0


def test_q_pcc_rows_reenabled_basic_shape():
    """With Strategy D the TSO H matrix has Q_PCC rows in addition to
    V, I and Q_gen.  Verify the basic dimensions for a small zone."""
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from experiments.helpers.utils import _network_state

    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    sh_buses_z2 = [
        int(b) for b, z in zip(
            meta.tso_tertiary_shunt_buses, meta.tso_tertiary_shunt_zones,
        ) if z == 2
    ]
    assert len(sh_buses_z2) > 0
    pcc_trafos_z2 = []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            pcc_trafos_z2.extend(list(hv.coupling_trafo_indices))
    obs_buses = []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            obs_buses.extend(list(hv.bus_indices))

    n_pcc = len(pcc_trafos_z2)
    n_shunt = len(sh_buses_z2)
    n_v = len(obs_buses)

    cfg = TSOControllerConfig(
        der_indices=[],
        pcc_trafo_indices=pcc_trafos_z2,
        pcc_dso_controller_ids=[f"DSO_{i}" for i in range(n_pcc)],
        oltc_trafo_indices=[],
        shunt_bus_indices=sh_buses_z2,
        shunt_q_steps_mvar=[50.0] * n_shunt,
        voltage_bus_indices=obs_buses,
        current_line_indices=[],
        gen_indices=[],
        v_setpoints_pu=np.full(n_v, 1.03),
        g_q_tso=1.0,
        pcc_capability_on_output=True,
    )
    n_controls = 0 + n_pcc + 0 + 0 + n_shunt
    params = OFOParameters(
        alpha=1.0,
        g_w=np.full(n_controls, 100.0),
        g_z=np.zeros(n_v + n_pcc),
        g_u=np.zeros(n_controls),
    )
    ns = _network_state(net)
    bounds = ActuatorBounds(
        der_indices=np.array([], dtype=np.int64),
        der_s_rated_mva=np.array([], dtype=np.float64),
        der_p_max_mw=np.array([], dtype=np.float64),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_min=np.array([], dtype=np.int64),
        oltc_tap_max=np.array([], dtype=np.int64),
        shunt_indices=np.array(sh_buses_z2, dtype=np.int64),
        shunt_q_mvar=np.array([50.0] * n_shunt, dtype=np.float64),
    )
    sens = JacobianSensitivities(net)
    tso = TSOController("tso_test_qpcc", params, cfg, ns, bounds, sens)

    H = tso._build_sensitivity_matrix()
    # Expected shape: rows = n_v + n_pcc + n_i + n_gen = n_v + n_pcc
    # cols = n_pcc + n_shunt
    assert H.shape == (n_v + n_pcc, n_pcc + n_shunt), (
        f"H shape {H.shape}; expected {(n_v + n_pcc, n_pcc + n_shunt)}"
    )

    # Q_PCC,set columns: closed-loop identity (diagonal +1, off-diag 0)
    q_pcc_block = H[n_v:n_v + n_pcc, 0:n_pcc]
    np.testing.assert_array_equal(q_pcc_block, np.eye(n_pcc))


def test_q_pcc_rows_shunt_column_matches_jacobian():
    """The shunt column in the Q_PCC row block must match
    compute_dQtrafo3w_hv_dQ_shunt to numerical precision."""
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from experiments.helpers.utils import _network_state

    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Use DSO_1: pick its first 3W coupler (the one with the shunt) as
    # the only PCC for this zone setup.
    hv0 = meta.hv_networks[0]
    pcc_trafo = int(hv0.coupling_trafo_indices[0])
    sh_bus = int(meta.tso_tertiary_shunt_buses[0])
    obs_buses = list(hv0.bus_indices)

    cfg = TSOControllerConfig(
        der_indices=[],
        pcc_trafo_indices=[pcc_trafo],
        pcc_dso_controller_ids=["DSO_1"],
        oltc_trafo_indices=[],
        shunt_bus_indices=[sh_bus],
        shunt_q_steps_mvar=[50.0],
        voltage_bus_indices=obs_buses,
        current_line_indices=[],
        gen_indices=[],
        v_setpoints_pu=np.full(len(obs_buses), 1.03),
        g_q_tso=1.0,
        pcc_capability_on_output=True,
    )
    n_controls = 1 + 1
    params = OFOParameters(
        alpha=1.0,
        g_w=np.full(n_controls, 100.0),
        g_z=np.zeros(len(obs_buses) + 1),
        g_u=np.zeros(n_controls),
    )
    ns = _network_state(net)
    bounds = ActuatorBounds(
        der_indices=np.array([], dtype=np.int64),
        der_s_rated_mva=np.array([], dtype=np.float64),
        der_p_max_mw=np.array([], dtype=np.float64),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_min=np.array([], dtype=np.int64),
        oltc_tap_max=np.array([], dtype=np.int64),
        shunt_indices=np.array([sh_bus], dtype=np.int64),
        shunt_q_mvar=np.array([50.0], dtype=np.float64),
    )
    sens = JacobianSensitivities(net)
    tso = TSOController("tso_test_qpcc_col", params, cfg, ns, bounds, sens)

    H = tso._build_sensitivity_matrix()
    n_v = len(obs_buses)
    # Shunt column index: n_pcc=1 PCC col + 0 V_gen + 0 OLTC + 0 = position 1
    shunt_col = 1
    actual = H[n_v, shunt_col]  # Q_PCC row, shunt column

    expected = sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=pcc_trafo,
        shunt_bus_idx=sh_bus,
        q_step_mvar=50.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_q_pcc_output_band_when_capability_on_output_true():
    """With pcc_capability_on_output=True, the Q_PCC output bound is
    centred at the current Q_iface ± DSO capability; the control-side
    Q_PCC,set band is wide engineering."""
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from core.message import CapabilityMessage
    from experiments.helpers.utils import _network_state

    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    hv0 = meta.hv_networks[0]
    pcc_trafo = int(hv0.coupling_trafo_indices[0])
    obs_buses = list(hv0.bus_indices)

    cfg = TSOControllerConfig(
        der_indices=[],
        pcc_trafo_indices=[pcc_trafo],
        pcc_dso_controller_ids=["DSO_1"],
        oltc_trafo_indices=[],
        shunt_bus_indices=[],
        shunt_q_steps_mvar=[],
        voltage_bus_indices=obs_buses,
        current_line_indices=[],
        gen_indices=[],
        v_setpoints_pu=np.full(len(obs_buses), 1.03),
        g_q_tso=1.0,
        pcc_capability_on_output=True,
    )
    n_controls = 1
    params = OFOParameters(
        alpha=1.0,
        g_w=np.full(n_controls, 100.0),
        g_z=np.zeros(len(obs_buses) + 1),
        g_u=np.zeros(n_controls),
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
    tso = TSOController("tso_test_band", params, cfg, ns, bounds, sens)

    # Send a capability message: tight ±5 Mvar
    cap = CapabilityMessage(
        source_controller_id="DSO_1",
        target_controller_id="tso_test_band",
        iteration=0,
        interface_transformer_indices=np.array([pcc_trafo], dtype=np.int64),
        q_min_mvar=np.array([-5.0]),
        q_max_mvar=np.array([5.0]),
    )
    tso.receive_capability(cap)
    # Run a step so _last_measurement is populated for the output bound
    from core.measurement import measure_zone_tso
    from controller.multi_tso_coordinator import ZoneDefinition
    zd = ZoneDefinition(
        zone_id=0, bus_indices=obs_buses, gen_indices=[], gen_bus_indices=[],
        tso_der_indices=[], tso_der_buses=[], v_bus_indices=obs_buses,
        line_indices=[], line_max_i_ka=[], pcc_trafo_indices=[pcc_trafo],
        pcc_dso_ids=["DSO_1"], shunt_bus_indices=[], shunt_q_steps_mvar=[],
    )
    meas = measure_zone_tso(net, zd, 1)
    tso._last_measurement = meas

    y_lo, y_hi = tso._get_output_limits()
    # Output ordering: [V | Q_PCC | I | Q_gen]
    n_v = len(obs_buses)
    q_pcc_lo = y_lo[n_v]
    q_pcc_hi = y_hi[n_v]
    band = q_pcc_hi - q_pcc_lo
    # Band width should be ~ DSO_max - DSO_min = 10 Mvar
    np.testing.assert_allclose(band, 10.0, rtol=0.05)

    # Control-side Q_PCC,set bound should be wide engineering range
    q_iface = tso._extract_trafo_reactive_power(meas)
    der_p = tso._extract_der_active_power(meas)
    u_lo, u_hi = tso._compute_input_bounds(q_iface, der_p)
    # Only the PCC column (index 0 since no DERs)
    assert u_lo[0] <= -1000.0  # wide band
    assert u_hi[0] >= +1000.0


def test_shunt_v_gradient_sign_via_TSO_H():
    """Same sign trace but through the full TSOController._build_sensitivity_matrix
    pipeline (catches column-placement bugs).

    Builds the zone-2 TSO controller, forces an H rebuild, slices out
    the shunt column for the first tertiary shunt, and verifies the
    same gradient signs hold."""
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from experiments.helpers.utils import _network_state

    net, meta = _build_ieee39_with_shunts()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Pick zone-2 shunts: first 3 (DSO_1 / DSO_2 / DSO_3 all in zone 2)
    sh_buses_z2 = [
        int(b) for b, z in zip(
            meta.tso_tertiary_shunt_buses, meta.tso_tertiary_shunt_zones,
        ) if z == 2
    ]
    assert len(sh_buses_z2) > 0, "Zone 2 should have at least 1 shunt"
    q_steps_z2 = [50.0] * len(sh_buses_z2)

    # Build observation buses across all zone-2 HV sub-networks
    obs_buses = []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            obs_buses.extend(list(hv.bus_indices))

    cfg = TSOControllerConfig(
        der_indices=[],
        pcc_trafo_indices=[],
        pcc_dso_controller_ids=[],
        oltc_trafo_indices=[],
        shunt_bus_indices=sh_buses_z2,
        shunt_q_steps_mvar=q_steps_z2,
        voltage_bus_indices=obs_buses,
        current_line_indices=[],
        gen_indices=[],
        v_setpoints_pu=np.full(len(obs_buses), 1.03),
    )
    n_controls = len(sh_buses_z2)
    params = OFOParameters(
        alpha=1.0,
        g_w=np.full(n_controls, 100.0),
        g_z=np.zeros(len(obs_buses)),  # n_v rows; no I, no Q_gen here
        g_u=np.zeros(n_controls),
    )
    ns = _network_state(net)
    bounds = ActuatorBounds(
        der_indices=np.array([], dtype=np.int64),
        der_s_rated_mva=np.array([], dtype=np.float64),
        der_p_max_mw=np.array([], dtype=np.float64),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_min=np.array([], dtype=np.int64),
        oltc_tap_max=np.array([], dtype=np.int64),
        shunt_indices=np.array(sh_buses_z2, dtype=np.int64),
        shunt_q_mvar=np.array(q_steps_z2, dtype=np.float64),
    )
    sens = JacobianSensitivities(net)
    tso = TSOController("tso_zone_2_test", params, cfg, ns, bounds, sens)

    # Force H rebuild — this exercises the full _build_sensitivity_matrix
    H = tso._build_sensitivity_matrix()

    n_v = len(obs_buses)
    n_shunt = len(sh_buses_z2)
    # Column ordering: [DER | PCC_set | V_gen | OLTC | shunt]
    # With empty DER/PCC/V_gen/OLTC, shunt columns start at 0.
    shunt_col_start = 0  # = n_der + n_pcc + n_gen + n_oltc, all zero
    dV_du_shunt = H[:n_v, shunt_col_start:shunt_col_start + n_shunt]

    # Per-bus sensitivity for the FIRST shunt should be negative everywhere
    # (or zero at electrically distant buses).  At least at the shunt's own
    # parent HV-sub-network buses we expect strictly < 0.
    sh0_col = dV_du_shunt[:, 0]
    n_neg = int(np.sum(sh0_col < -1e-10))
    n_pos = int(np.sum(sh0_col > +1e-10))
    assert n_neg > 0, (
        f"First shunt column has no negative ∂V/∂s entries — "
        f"sign convention may be inverted.  col = {sh0_col}"
    )
    assert n_pos == 0, (
        f"First shunt column has POSITIVE ∂V/∂s entries: "
        f"{sh0_col[sh0_col > 0]} — increasing step should never raise V."
    )

    # Gradient sign under uniform undervoltage — should be > 0
    v_err = np.full(n_v, -0.05, dtype=np.float64)
    grad_shunt0 = float(v_err @ sh0_col)
    assert grad_shunt0 > 0.0, (
        f"Undervoltage V-tracking gradient on first shunt = {grad_shunt0:.6e} "
        f"should be POSITIVE (MIQP wants to keep capacitor on, i.e. s = -1)"
    )
