"""
Tests for the TSO transmission-loss objective term (form B).

The loss term ``g_loss · Σ_ℓ c_ℓ·|I_ℓ|²`` is summed over the monitored current
lines and realised through the *existing* current rows of the sensitivity
matrix: ``∂P_loss/∂I_ℓ = 2·g_loss·c_ℓ·|I_ℓ|`` projected via the cached
``∂I_ℓ/∂u``.  These tests pin:

  1. The output/objective gradient invariant ``grad_f == ∇_y f · H`` still holds
     with the loss term active (so the shunt integrator stays consistent).
  2. The default per-line coefficient is ``c_ℓ = 3·R_ℓ`` from the cached net.
  3. The loss term contributes a non-zero, current-reducing control gradient.
  4. ``g_loss == 0`` reproduces the legacy (loss-free) gradient exactly.
  5. The PMU (phasor) path is a guarded stub.

Author: Manuel Schwenke
Date: 2026-06-30
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


def _build():
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=True, verbose=False,
    )
    return net, meta


def _make_tso(net, meta, *, g_loss, loss_use_phasor=False, line_coeff=None):
    """Build a Zone-2 TSO controller that monitors a few EHV lines so the
    loss term has non-empty current rows to act on."""
    from controller.tso_controller import TSOController, TSOControllerConfig
    from controller.base_controller import OFOParameters
    from core.actuator_bounds import ActuatorBounds
    from core.measurement import measure_zone_tso
    from controller.multi_tso_coordinator import ZoneDefinition
    from experiments.helpers.utils import _network_state
    from sensitivity.jacobian import JacobianSensitivities

    pcc_trafos, obs_buses = [], []
    for hv in meta.hv_networks:
        if hv.zone == 2:
            pcc_trafos.extend(list(hv.coupling_trafo_indices))
            obs_buses.extend(list(hv.bus_indices))
    assert pcc_trafos

    # Pick a handful of EHV lines whose both endpoints are monitored buses,
    # so their current sensitivities are well-posed in the reduced query.
    obs_set = set(obs_buses)
    line_idx = [
        li for li in net.line.index
        if int(net.line.at[li, "from_bus"]) in obs_set
        and int(net.line.at[li, "to_bus"]) in obs_set
    ][:4]
    assert line_idx, "need at least one in-zone monitored line for the loss test"

    n_pcc = len(pcc_trafos)
    n_v = len(obs_buses)
    cfg = TSOControllerConfig(
        der_indices=[],
        pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[f"DSO_{i}" for i in range(n_pcc)],
        oltc_trafo_indices=[],
        shunt_bus_indices=[],
        shunt_q_steps_mvar=[],
        voltage_bus_indices=obs_buses,
        current_line_indices=line_idx,
        gen_indices=[],
        v_setpoints_pu=np.full(n_v, 1.03),
        g_v=2.0,
        g_q_tso=1.5,
        g_loss=g_loss,
        loss_use_phasor=loss_use_phasor,
        loss_line_coeff_mw_per_ka2=line_coeff,
    )
    params = OFOParameters(
        alpha=1.0,
        g_w=np.full(n_pcc, 100.0),
        g_z=np.zeros(n_v + n_pcc + len(line_idx)),
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
    sens = JacobianSensitivities(net)
    tso = TSOController("tso_loss_test", params, cfg, ns, bounds, sens)

    zd = ZoneDefinition(
        zone_id=2, bus_indices=obs_buses, gen_indices=[], gen_bus_indices=[],
        tso_der_indices=[], tso_der_buses=[], v_bus_indices=obs_buses,
        line_indices=line_idx, line_max_i_ka=[], pcc_trafo_indices=pcc_trafos,
        pcc_dso_ids=[f"DSO_{i}" for i in range(n_pcc)],
        shunt_bus_indices=[], shunt_q_steps_mvar=[],
    )
    meas = measure_zone_tso(net, zd, 1)
    tso.initialise(meas)
    tso._last_measurement = meas
    tso._u_current = tso._u_current + 7.5  # non-trivial Q_PCC error
    return tso, meas, line_idx


def test_loss_gradient_invariant_holds():
    """grad_f == ∇_y f · H with the loss term active (g_loss > 0)."""
    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    tso, meas, _ = _make_tso(net, meta, g_loss=5.0)

    grad_f = tso._compute_objective_gradient(meas)
    grad_y = tso._compute_output_gradient(meas)
    H = tso._expand_H_to_der_level(tso._build_sensitivity_matrix())

    assert grad_y.shape[0] == H.shape[0]
    np.testing.assert_allclose(grad_f, grad_y @ H, rtol=1e-9, atol=1e-9)


def test_loss_coeff_default_is_three_r():
    """Default per-line coefficient c_ℓ = 3·R_ℓ from the cached net."""
    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    tso, _, line_idx = _make_tso(net, meta, g_loss=1.0)

    c = tso._loss_line_coeffs()
    assert c.shape[0] == len(line_idx)
    for k, li in enumerate(line_idx):
        r_ohm = (
            float(net.line.at[li, "r_ohm_per_km"])
            * float(net.line.at[li, "length_km"])
            / max(float(net.line.at[li, "parallel"]), 1.0)
        )
        assert c[k] == pytest.approx(3.0 * r_ohm)


def test_loss_term_is_current_reducing_and_nonzero():
    """The I-block of ∇_y f equals 2·g_loss·c·|I_meas| (≥ 0, non-trivial),
    and zeroing g_loss removes exactly that contribution from grad_f."""
    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    g_loss = 5.0
    tso, meas, line_idx = _make_tso(net, meta, g_loss=g_loss)
    n_v = len(tso.config.voltage_bus_indices)
    n_pcc = len(tso.config.pcc_trafo_indices)
    n_i = len(line_idx)

    grad_y = tso._compute_output_gradient(meas)
    grad_i = grad_y[n_v + n_pcc:n_v + n_pcc + n_i]

    # Expected I-block: 2·g_loss·c·|I_meas|, all non-negative, not all zero.
    c = tso._loss_line_coeffs()
    i_meas = meas.current_magnitudes_ka
    np.testing.assert_allclose(grad_i, 2.0 * g_loss * c * i_meas, rtol=1e-12)
    assert np.all(grad_i >= 0.0)
    assert np.linalg.norm(grad_i) > 0.0

    # grad_f with the loss term minus grad_f without it must equal exactly the
    # projected loss contribution grad_i · H[I-rows].
    tso_off, meas_off, _ = _make_tso(net, meta, g_loss=0.0)
    grad_f_on = tso._compute_objective_gradient(meas)
    grad_f_off = tso_off._compute_objective_gradient(meas_off)
    H = tso._expand_H_to_der_level(tso._build_sensitivity_matrix())
    dI_du = H[n_v + n_pcc:n_v + n_pcc + n_i, :]
    np.testing.assert_allclose(
        grad_f_on - grad_f_off, grad_i @ dI_du, rtol=1e-9, atol=1e-9,
    )

    # Off-path I-block is exactly zero (legacy behaviour preserved).
    grad_y_off = tso_off._compute_output_gradient(meas_off)
    assert np.linalg.norm(grad_y_off[n_v + n_pcc:n_v + n_pcc + n_i]) == 0.0


def test_loss_line_coeff_override_can_exclude_a_line():
    """A zero override coefficient drops that line from the loss sum."""
    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # First discover how many lines the helper picks, then override with a
    # vector that zeroes the first line only.
    tso0, _, line_idx = _make_tso(net, meta, g_loss=1.0)
    coeff = [0.0] + [1.0] * (len(line_idx) - 1)

    tso, meas, _ = _make_tso(net, meta, g_loss=1.0, line_coeff=coeff)
    c = tso._loss_line_coeffs()
    np.testing.assert_allclose(c, np.asarray(coeff))
    grad_y = tso._compute_output_gradient(meas)
    n_v = len(tso.config.voltage_bus_indices)
    n_pcc = len(tso.config.pcc_trafo_indices)
    # First monitored line contributes nothing to the loss gradient.
    assert grad_y[n_v + n_pcc] == 0.0


def test_phasor_path_is_guarded_stub():
    """loss_use_phasor=True raises NotImplementedError at gradient time."""
    net, meta = _build()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    tso, meas, _ = _make_tso(net, meta, g_loss=5.0, loss_use_phasor=True)
    with pytest.raises(NotImplementedError):
        tso._compute_output_gradient(meas)


def test_g_loss_without_current_line_warns_and_is_inert():
    """Config guard: g_loss != 0 with no current lines warns (no crash) —
    the loss term is a silent no-op when there is nothing to sum over."""
    from controller.tso_controller import TSOControllerConfig

    with pytest.warns(RuntimeWarning):
        cfg = TSOControllerConfig(
            der_indices=[],
            pcc_trafo_indices=[],
            pcc_dso_controller_ids=[],
            oltc_trafo_indices=[],
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
            voltage_bus_indices=[0],
            current_line_indices=[],   # empty → loss term has nothing to act on
            g_loss=1.0,
        )
    assert cfg.g_loss == 1.0
