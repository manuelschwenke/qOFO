"""
Integration test: switched-shunt integrator against the REAL pandapower
sensitivities on the IEEE 39 + HV benchmark.

This bridges the pure integrator logic (``test_shunt_integrator.py``) and the
plant: it checks the boundary sensitivity SIGNS the integrator relies on (an MSC
step must raise voltage, an MSR step must lower it), the interface feedforward
sign, and that a bank driven by a sustained under-voltage gradient computed from
those real sensitivities commits one step — while a brief transient does not.

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

from controller.shunt_integrator import MSC, MSR, ShuntBankConfig  # noqa: E402


def _build():
    from network.ieee39.build import build_ieee39_net
    from network.ieee39.hv_networks import add_hv_networks

    net, meta = build_ieee39_net(scenario="base", verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=True,
        tso_shunt_kind="msc_msr",
        msc_n_levels=4, msr_n_levels=4,
        msc_q_step_mvar=50.0, msr_q_step_mvar=50.0,
        verbose=False,
    )
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    return net, meta


def _first_bank(meta, kind):
    for sh_idx, bus, t_zone, k in zip(
        meta.tso_tertiary_shunt_indices,
        meta.tso_tertiary_shunt_buses,
        meta.tso_tertiary_shunt_zones,
        meta.tso_tertiary_shunt_kinds,
    ):
        if k == kind:
            return int(sh_idx), int(bus)
    raise AssertionError(f"no {kind} bank found")


def test_boundary_sensitivity_signs():
    """∂V/∂Q_eq (per Mvar) at the HV sub-network buses: > 0 for MSC, < 0 for MSR."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    _, msc_bus = _first_bank(meta, "MSC")

    # MSC: pass the signed nameplate sign (-1 ⇒ capacitive) → V rises with step.
    h_msc, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=msc_bus,
        observation_bus_indices=list(hv0.bus_indices),
        q_step_mvar=-1.0,
    )
    assert len(obs_map) > 0
    assert np.all(np.asarray(h_msc) > 0.0), f"MSC ∂V/∂Q_eq must be > 0, got {h_msc}"

    # MSR: +1 ⇒ inductive → V falls with step.
    h_msr, _ = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=msc_bus,
        observation_bus_indices=list(hv0.bus_indices),
        q_step_mvar=+1.0,
    )
    assert np.all(np.asarray(h_msr) < 0.0), f"MSR ∂V/∂Q_eq must be < 0, got {h_msr}"


def test_interface_feedforward_finite_and_signed():
    """dQ_itf for an MSC engaging step is finite and opposite-signed to MSR."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    t_iface = int(hv0.coupling_trafo_indices[0])
    _, msc_bus = _first_bank(meta, "MSC")

    dq_msc = sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=t_iface, shunt_bus_idx=msc_bus, q_step_mvar=-50.0,
    )
    dq_msr = sens.compute_dQtrafo3w_hv_dQ_shunt(
        trafo3w_idx=t_iface, shunt_bus_idx=msc_bus, q_step_mvar=+50.0,
    )
    assert np.isfinite(dq_msc) and np.isfinite(dq_msr)
    assert abs(dq_msc) > 0.0
    # Opposite susceptance signs ⇒ opposite interface-Q contributions.
    assert np.sign(dq_msc) == -np.sign(dq_msr)


def test_msc_commits_on_sustained_undervoltage_from_real_sensitivities():
    """Drive an MSC bank with a sustained under-voltage gradient built from the
    real Jacobian; it must integrate to a commit, but a single transient must
    not."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    msc_idx, msc_bus = _first_bank(meta, "MSC")
    t_iface = int(hv0.coupling_trafo_indices[0])

    obs = list(hv0.bus_indices)
    h_col, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=msc_bus, observation_bus_indices=obs, q_step_mvar=-1.0,
    )
    h_col = np.asarray(h_col, dtype=np.float64)

    # Synthetic sustained under-voltage: V_err = -0.05 p.u. at every obs bus.
    # ∇_y f (V rows) = 2·g_v·V_err with g_v = 1.  g_H = ∇_y f · h_col < 0
    # (under-voltage + MSC raises V ⇒ engage).
    g_v = 1.0
    grad_v = 2.0 * g_v * np.full(len(obs_map), -0.05)
    grad_g = float(grad_v @ h_col)
    assert grad_g < 0.0, "under-voltage MSC gradient should drive engagement"

    # The boundary ∂V/∂Q_eq is small (~1e-4 p.u./Mvar), so the gain must be
    # scaled to the gradient magnitude for a bulk device to commit in a sane
    # number of iterations.  Size g_w so each iteration advances the auxiliary
    # state by ~5 Mvar (step = g_H/(2*g_w)) → ~6 iterations to cross the 30 Mvar
    # up-threshold, while a single transient (5 Mvar) stays well below it.
    # (This scaling is the tuning concern flagged in the status doc.)
    g_w = abs(grad_g) / 10.0
    # Wide band here to ISOLATE the integrator mechanism from the overshoot
    # guard.  At these HV buses ∂V/∂Q_eq is large (~1e-2 p.u./Mvar), so the
    # LINEAR predictor v + h·50 over-estimates the post-step voltage massively
    # (the linearisation is only valid locally) — the guard is exercised
    # separately in ``test_feasibility_guard_blocks_when_band_tight``.
    cfg = ShuntBankConfig(
        shunt_idx=msc_idx, bus_idx=msc_bus, interface_trafo3w_idx=t_iface,
        dso_id=hv0.net_id, kind="MSC", q_step_mvar=50.0, n_levels=4,
        g_w=g_w, delta=5.0, t_dwell_s=0.0, daily_switch_budget=10,
        y_h_min=0.0, y_h_max=10.0,
    )

    # Brief transient: a single iteration must NOT commit.
    bank_t = MSC(cfg)
    v_meas = np.array([
        float(net.res_bus.at[b, "vm_pu"]) for b in obs_map
    ], dtype=np.float64)
    assert bank_t.step(grad_g=grad_g, v_meas=v_meas, h_v=h_col, t_now=0.0) is None

    # Sustained: repeated iterations integrate to a commit (engage capacitor).
    bank = MSC(cfg)
    commit = None
    for i in range(50):
        c = bank.step(grad_g=grad_g, v_meas=v_meas, h_v=h_col, t_now=float(i))
        if c is not None:
            commit = c
            break
    assert commit is not None, "sustained under-voltage should commit an MSC step"
    assert commit.direction == +1 and commit.new_level == 1
    assert commit.shunt_idx == msc_idx


def test_feasibility_guard_blocks_when_band_tight():
    """With a band tight around the present voltage, the overshoot guard blocks
    the otherwise-ready commit."""
    from sensitivity.jacobian import JacobianSensitivities

    net, meta = _build()
    sens = JacobianSensitivities(net)
    hv0 = meta.hv_networks[0]
    msc_idx, msc_bus = _first_bank(meta, "MSC")
    t_iface = int(hv0.coupling_trafo_indices[0])
    obs = list(hv0.bus_indices)
    h_col, obs_map = sens.compute_dV_dQ_shunt(
        shunt_bus_idx=msc_bus, observation_bus_indices=obs, q_step_mvar=-1.0,
    )
    h_col = np.asarray(h_col, dtype=np.float64)
    v_meas = np.array([
        float(net.res_bus.at[b, "vm_pu"]) for b in obs_map
    ], dtype=np.float64)

    # Band upper rail just above the present max voltage so a capacitive step
    # (which raises V) would overshoot it.
    v_max_tight = float(v_meas.max()) + 1e-4
    cfg = ShuntBankConfig(
        shunt_idx=msc_idx, bus_idx=msc_bus, interface_trafo3w_idx=t_iface,
        dso_id=hv0.net_id, kind="MSC", q_step_mvar=50.0, n_levels=4,
        g_w=0.25, delta=5.0, t_dwell_s=0.0, daily_switch_budget=10,
        y_h_min=0.80, y_h_max=v_max_tight,
    )
    bank = MSC(cfg)
    bank.q_eq_aux = 100.0  # already past the up-threshold
    grad_g = float((2.0 * np.full(len(obs_map), -0.05)) @ h_col)
    assert bank.step(grad_g=grad_g, v_meas=v_meas, h_v=h_col, t_now=0.0) is None
    assert bank.level == 0
