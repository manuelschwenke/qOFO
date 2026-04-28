"""
Finite-difference validation for tap-position sensitivities.

Covers ``compute_dV_ds_2w`` (V-at-PQ vs tap) and
``compute_dQgen_ds_2w_matrix`` (Q_gen-at-PV vs tap, including the direct
term at the gen's own machine transformer).

Both sensitivities are algebra-derived from the pi-equivalent branch
model; we exercise them against central-difference power-flow solves on
a minimal PV-gen-behind-machine-trafo network — the canonical topology
for a TSO machine-transformer OLTC.

Topology::

    ext_grid ── line ── EHV_bus (PQ, 345 kV) ── machine trafo (tap_side=hv)
                                  │                 │
                                  load              │
                                                    │
                                              gen_bus (PV, 10.5 kV, V=1.0)

Tap on HV side, so raising ``tap_pos`` increases tau = 1 + s * delta_tau and
— with the gen pinning V_LV — should raise V_HV.  Expected sign: positive.

Regression context: ``compute_dQgen_ds_2w_matrix`` previously captured
only the indirect chain ``∂Q_gen/∂x · dx/dτ`` and missed the direct
τ-dependence of Q_calc at the gen's own LV bus, producing wrong-signed
diagonal entries on IEEE39 machine trafos.  The test below guards the
fix at :func:`sensitivity.jacobian.JacobianSensitivities._compute_dg_dtau_2w`
which now also returns the per-endpoint direct dQ/dτ.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from sensitivity.jacobian import JacobianSensitivities


@pytest.fixture
def machine_trafo_network() -> pp.pandapowerNet:
    """EHV slack ↔ PQ bus ↔ machine trafo ↔ PV gen."""
    net = pp.create_empty_network(sn_mva=100.0)

    b_slack = pp.create_bus(net, vn_kv=345.0, name="slack_ehv")
    b_ehv = pp.create_bus(net, vn_kv=345.0, name="ehv_pq")
    b_gen = pp.create_bus(net, vn_kv=10.5, name="gen_lv")

    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.00)

    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_ehv,
        length_km=30.0, r_ohm_per_km=0.05, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.5,
    )

    pp.create_load(net, bus=b_ehv, p_mw=60.0, q_mvar=20.0)

    t_idx = pp.create_transformer_from_parameters(
        net,
        hv_bus=b_ehv, lv_bus=b_gen,
        sn_mva=200.0, vn_hv_kv=345.0, vn_lv_kv=10.5,
        vk_percent=12.0, vkr_percent=0.3,
        pfe_kw=0.0, i0_percent=0.0,
        tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
        tap_pos=0, tap_step_percent=1.25, shift_degree=0.0,
        tap_changer_type="Ratio",
        name="machine_trafo",
    )

    pp.create_gen(net, bus=b_gen, p_mw=100.0, vm_pu=1.00, name="PV_GEN")

    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged
    # Stash the trafo index for tests to grab without re-querying.
    net.trafo_test_idx = int(t_idx)
    net.obs_bus_test_idx = int(b_ehv)
    return net


def _numerical_dV_ds_2w(
    net: pp.pandapowerNet,
    trafo_idx: int,
    obs_bus: int,
    delta_s: int = 1,
) -> float:
    """Central-difference approximation of ∂V_obs / ∂tap_pos via pp.runpp."""
    s0 = int(net.trafo.at[trafo_idx, "tap_pos"])

    net_pos = copy.deepcopy(net)
    net_pos.trafo.at[trafo_idx, "tap_pos"] = s0 + delta_s
    pp.runpp(net_pos, calculate_voltage_angles=True)
    v_pos = float(net_pos.res_bus.at[obs_bus, "vm_pu"])

    net_neg = copy.deepcopy(net)
    net_neg.trafo.at[trafo_idx, "tap_pos"] = s0 - delta_s
    pp.runpp(net_neg, calculate_voltage_angles=True)
    v_neg = float(net_neg.res_bus.at[obs_bus, "vm_pu"])

    return (v_pos - v_neg) / (2 * delta_s)


class TestComputeDVds2W:
    def test_shape_and_mapping(self, machine_trafo_network):
        sens = JacobianSensitivities(machine_trafo_network)
        dV_ds, obs_map = sens.compute_dV_ds_2w(
            trafo_idx=machine_trafo_network.trafo_test_idx,
            observation_bus_indices=[machine_trafo_network.obs_bus_test_idx],
        )
        assert dV_ds.shape == (1,)
        assert obs_map == [machine_trafo_network.obs_bus_test_idx]
        assert np.isfinite(dV_ds[0])

    def test_sign_positive_for_hv_side_tap(self, machine_trafo_network):
        """Raising HV-side tap with LV pinned by gen must raise V_HV."""
        sens = JacobianSensitivities(machine_trafo_network)
        dV_ds, _ = sens.compute_dV_ds_2w(
            trafo_idx=machine_trafo_network.trafo_test_idx,
            observation_bus_indices=[machine_trafo_network.obs_bus_test_idx],
        )
        numerical = _numerical_dV_ds_2w(
            machine_trafo_network,
            trafo_idx=machine_trafo_network.trafo_test_idx,
            obs_bus=machine_trafo_network.obs_bus_test_idx,
        )
        # Physics: gen pins V_LV=1.0; HV-side tap up → τ up → V_HV up.
        assert numerical > 0, f"Physics check: numerical should be > 0, got {numerical}"
        assert dV_ds[0] > 0, (
            f"Analytical ∂V/∂τ has wrong sign: analytical={dV_ds[0]}, "
            f"numerical={numerical}"
        )

    def test_numerical_agreement(self, machine_trafo_network):
        """Analytical vs central-difference ∂V/∂tap on a machine trafo."""
        sens = JacobianSensitivities(machine_trafo_network)
        dV_ds, _ = sens.compute_dV_ds_2w(
            trafo_idx=machine_trafo_network.trafo_test_idx,
            observation_bus_indices=[machine_trafo_network.obs_bus_test_idx],
        )
        analytical = float(dV_ds[0])
        numerical = _numerical_dV_ds_2w(
            machine_trafo_network,
            trafo_idx=machine_trafo_network.trafo_test_idx,
            obs_bus=machine_trafo_network.obs_bus_test_idx,
        )
        assert np.sign(analytical) == np.sign(numerical), (
            f"Sign disagreement: analytical={analytical}, numerical={numerical}"
        )
        if abs(numerical) > 1e-5:
            ratio = analytical / numerical
            assert 0.5 < ratio < 2.0, (
                f"Magnitude disagreement: analytical={analytical}, "
                f"numerical={numerical}, ratio={ratio}"
            )


def _numerical_dQgen_ds_2w(
    net: pp.pandapowerNet,
    trafo_idx: int,
    gen_bus: int,
    delta_s: int = 1,
) -> float:
    """Central-difference approximation of ∂Q_gen / ∂tap_pos via pp.runpp."""
    s0 = int(net.trafo.at[trafo_idx, "tap_pos"])
    gen_mask = net.gen["bus"] == gen_bus
    assert gen_mask.any(), f"No gen at bus {gen_bus}"
    gen_idx = int(net.gen.index[gen_mask][0])

    net_pos = copy.deepcopy(net)
    net_pos.trafo.at[trafo_idx, "tap_pos"] = s0 + delta_s
    pp.runpp(net_pos, calculate_voltage_angles=True)
    q_pos = float(net_pos.res_gen.at[gen_idx, "q_mvar"])

    net_neg = copy.deepcopy(net)
    net_neg.trafo.at[trafo_idx, "tap_pos"] = s0 - delta_s
    pp.runpp(net_neg, calculate_voltage_angles=True)
    q_neg = float(net_neg.res_gen.at[gen_idx, "q_mvar"])

    return (q_pos - q_neg) / (2 * delta_s)


class TestComputeDQgenDs2WDirectTerm:
    """Regression: ``compute_dQgen_ds_2w_matrix`` must include the direct
    τ-dependence of Q_calc at the gen's own trafo endpoint.  Without it,
    diagonal entries land with the wrong sign and magnitude, causing the
    TSO MIQP to ratchet machine-transformer OLTCs to saturation against
    the voltage-tracking objective."""

    def test_diagonal_entry_matches_numerical(self, machine_trafo_network):
        """Gen at LV side of its own machine trafo: analytical = FD."""
        net = machine_trafo_network
        t_idx = net.trafo_test_idx
        gen_bus = int(net.gen.at[0, "bus"])  # LV terminal of machine trafo

        sens = JacobianSensitivities(net)
        mat, gen_map, _ = sens.compute_dQgen_ds_2w_matrix(
            gen_bus_indices_pp=[gen_bus],
            oltc_trafo_indices=[t_idx],
        )
        analytical = float(mat[0, 0]) * net.sn_mva
        numerical = _numerical_dQgen_ds_2w(net, trafo_idx=t_idx, gen_bus=gen_bus)

        # Physics: raising HV-side tap pushes V_HV up; gen must inject more Q
        # through the trafo leakage to keep V_LV=1.0 → dQ_gen/dτ > 0.
        assert numerical > 0, (
            f"Physics sanity check: numerical should be > 0, got {numerical}"
        )
        assert np.sign(analytical) == np.sign(numerical), (
            f"Sign disagreement: analytical={analytical}, numerical={numerical}"
        )
        if abs(numerical) > 1e-3:
            ratio = analytical / numerical
            assert 0.8 < ratio < 1.2, (
                f"Magnitude disagreement: analytical={analytical}, "
                f"numerical={numerical}, ratio={ratio}"
            )
