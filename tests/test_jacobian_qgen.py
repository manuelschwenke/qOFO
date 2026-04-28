"""
Tests for Q_gen output sensitivities and PQ-mode (AVR saturation) variants.

Covers:
* ``compute_dQgen_dQder_matrix`` — numerical agreement on a PV-bus network
* ``compute_dQgen_dVgen_matrix`` — direct (self) term vs indirect (cross)
* ``compute_dV_dVgen_matrix_pqmode`` — saturated column is zero, free matches
* Shape / mapping sanity for the other compute_dQgen_* family members

Design: a minimal 3-bus network with one PV generator (non-slack) and one
DER is sufficient to exercise every code path.  The slack bus is kept
separate so the PV generator truly contributes an absorbed-Q row.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest

from sensitivity.jacobian import JacobianSensitivities


@pytest.fixture
def pv_gen_network() -> pp.pandapowerNet:
    """3-bus: slack — line — PV gen bus — line — PQ DER bus."""
    net = pp.create_empty_network(sn_mva=100.0)
    b_slack = pp.create_bus(net, vn_kv=110.0, name="slack")
    b_gen = pp.create_bus(net, vn_kv=110.0, name="gen")
    b_der = pp.create_bus(net, vn_kv=110.0, name="der")

    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.02)

    pp.create_line_from_parameters(
        net, from_bus=b_slack, to_bus=b_gen,
        length_km=10.0, r_ohm_per_km=0.05, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )
    pp.create_line_from_parameters(
        net, from_bus=b_gen, to_bus=b_der,
        length_km=8.0, r_ohm_per_km=0.05, x_ohm_per_km=0.3,
        c_nf_per_km=10.0, max_i_ka=1.0,
    )

    pp.create_gen(net, bus=b_gen, p_mw=30.0, vm_pu=1.03, name="PV_GEN")
    pp.create_load(net, bus=b_der, p_mw=15.0, q_mvar=5.0)
    pp.create_sgen(net, bus=b_der, p_mw=0.0, q_mvar=0.0, name="DER")

    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged
    return net


def _numerical_dQgen_dQder(
    net: pp.pandapowerNet,
    gen_bus: int,
    der_bus: int,
    delta_q: float = 0.5,
) -> float:
    """Central-difference approximation of ∂Q_gen_{gen_bus}/∂Q_der_{der_bus}."""
    sgen_mask = net.sgen['bus'] == der_bus
    assert sgen_mask.any(), f"No sgen at bus {der_bus}"
    sgen_idx = int(net.sgen.index[sgen_mask][0])
    q0 = float(net.sgen.at[sgen_idx, 'q_mvar'])

    gen_mask = net.gen['bus'] == gen_bus
    assert gen_mask.any(), f"No gen at bus {gen_bus}"
    gen_idx = int(net.gen.index[gen_mask][0])

    net_pos = copy.deepcopy(net)
    net_pos.sgen.at[sgen_idx, 'q_mvar'] = q0 + delta_q
    pp.runpp(net_pos, calculate_voltage_angles=True)
    q_pos = float(net_pos.res_gen.at[gen_idx, 'q_mvar'])

    net_neg = copy.deepcopy(net)
    net_neg.sgen.at[sgen_idx, 'q_mvar'] = q0 - delta_q
    pp.runpp(net_neg, calculate_voltage_angles=True)
    q_neg = float(net_neg.res_gen.at[gen_idx, 'q_mvar'])

    return (q_pos - q_neg) / (2 * delta_q)


def _numerical_dQgen_dVgen(
    net: pp.pandapowerNet,
    meas_gen_bus: int,
    chg_gen_bus: int,
    delta_v: float = 1e-3,
) -> float:
    """Central-difference approximation of ∂Q_gen_meas/∂V_gen_set_chg."""
    gen_mask_meas = net.gen['bus'] == meas_gen_bus
    meas_idx = int(net.gen.index[gen_mask_meas][0])
    gen_mask_chg = net.gen['bus'] == chg_gen_bus
    chg_idx = int(net.gen.index[gen_mask_chg][0])
    v0 = float(net.gen.at[chg_idx, 'vm_pu'])

    net_pos = copy.deepcopy(net)
    net_pos.gen.at[chg_idx, 'vm_pu'] = v0 + delta_v
    pp.runpp(net_pos, calculate_voltage_angles=True)
    q_pos = float(net_pos.res_gen.at[meas_idx, 'q_mvar'])

    net_neg = copy.deepcopy(net)
    net_neg.gen.at[chg_idx, 'vm_pu'] = v0 - delta_v
    pp.runpp(net_neg, calculate_voltage_angles=True)
    q_neg = float(net_neg.res_gen.at[meas_idx, 'q_mvar'])

    return (q_pos - q_neg) / (2 * delta_v)


class TestComputeDQgenDQder:
    def test_shape_and_mapping(self, pv_gen_network):
        sens = JacobianSensitivities(pv_gen_network)
        # Gen is at bus 1 (pandapower ordering), DER at bus 2.
        mat, gen_map, der_map = sens.compute_dQgen_dQder_matrix(
            gen_bus_indices_pp=[1], der_bus_indices=[2],
        )
        assert mat.shape == (1, 1)
        assert gen_map == [1]
        assert der_map == [2]
        assert np.isfinite(mat[0, 0])

    def test_numerical_agreement(self, pv_gen_network):
        """Analytical vs finite-difference Q_gen response to a DER Q kick.

        Both quantities are dimensionless (Mvar/Mvar).  The reduced Jacobian
        returns ∂Q_pu/∂Q_pu which equals ∂Q_Mvar/∂Q_Mvar — no ``sn_mva``
        scaling is needed here (unlike ``compute_dV_dQ_der`` where V stays
        in pu and only the Q axis is rescaled).
        """
        sens = JacobianSensitivities(pv_gen_network)
        mat, _, _ = sens.compute_dQgen_dQder_matrix(
            gen_bus_indices_pp=[1], der_bus_indices=[2],
        )
        analytical = float(mat[0, 0])
        numerical = _numerical_dQgen_dQder(pv_gen_network, gen_bus=1, der_bus=2)
        assert np.sign(analytical) == np.sign(numerical) or abs(numerical) < 1e-4
        if abs(numerical) > 1e-4:
            ratio = analytical / numerical
            assert 0.9 < ratio < 1.1, f"analytical={analytical}, numerical={numerical}, ratio={ratio}"


class TestComputeDQgenDVgen:
    def test_direct_term_matches_numerical(self, pv_gen_network):
        """Self-sensitivity ∂Q_gen_k/∂V_gen_k should match numerical."""
        sens = JacobianSensitivities(pv_gen_network)
        mat, _, _ = sens.compute_dQgen_dVgen_matrix(
            gen_bus_indices_pp_meas=[1],
            gen_bus_indices_pp_chg=[1],
        )
        analytical = float(mat[0, 0]) * pv_gen_network.sn_mva
        numerical = _numerical_dQgen_dVgen(
            pv_gen_network, meas_gen_bus=1, chg_gen_bus=1, delta_v=1e-3,
        )
        # Raising V_gen on an absorbing gen should raise its Q (+ sign).
        assert np.sign(analytical) == np.sign(numerical) or abs(numerical) < 1e-3
        if abs(numerical) > 1e-3:
            ratio = analytical / numerical
            assert 0.3 < ratio < 3.0, f"ratio={ratio}"


class TestPQModeVariants:
    def test_saturated_column_is_zero(self, pv_gen_network):
        """Mode = +1 → column for that gen is zeroed in the PQ-mode variant."""
        sens = JacobianSensitivities(pv_gen_network)
        mode = np.array([1], dtype=np.int8)  # saturated upper
        mat, _, _ = sens.compute_dV_dVgen_matrix_pqmode(
            gen_bus_indices_pp=[1],
            observation_bus_indices=[2],
            mode_vector=mode,
        )
        assert np.all(mat[:, 0] == 0.0)

    def test_free_column_matches_regular(self, pv_gen_network):
        """Mode = 0 → column equals the regular (non-PQ-mode) result."""
        sens = JacobianSensitivities(pv_gen_network)
        regular, _, _ = sens.compute_dV_dVgen_matrix(
            gen_bus_indices_pp=[1], observation_bus_indices=[2],
        )
        mode = np.array([0], dtype=np.int8)  # free
        pqmode, _, _ = sens.compute_dV_dVgen_matrix_pqmode(
            gen_bus_indices_pp=[1],
            observation_bus_indices=[2],
            mode_vector=mode,
        )
        np.testing.assert_allclose(pqmode, regular)


class TestComputeDQgenOther:
    """Sanity: remaining Q_gen matrix methods produce correctly-shaped, finite output."""

    def test_ds_2w_empty_safe(self, pv_gen_network):
        sens = JacobianSensitivities(pv_gen_network)
        mat, gen_map, t_map = sens.compute_dQgen_ds_2w_matrix(
            gen_bus_indices_pp=[1], oltc_trafo_indices=[],
        )
        assert mat.shape == (1, 0)

    def test_ds_3w_empty_safe(self, pv_gen_network):
        sens = JacobianSensitivities(pv_gen_network)
        mat, _, _ = sens.compute_dQgen_ds_3w_matrix(
            gen_bus_indices_pp=[1], oltc_trafo3w_indices=[],
        )
        assert mat.shape == (1, 0)

    def test_dQ_shunt_mismatched_steps_raises(self, pv_gen_network):
        sens = JacobianSensitivities(pv_gen_network)
        with pytest.raises(ValueError, match="equal length"):
            sens.compute_dQgen_dQ_shunt_matrix(
                gen_bus_indices_pp=[1],
                shunt_bus_indices=[2],
                shunt_q_steps_mvar=[],
            )
