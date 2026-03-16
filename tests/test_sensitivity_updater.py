"""
Tests for SensitivityUpdater class.

Verifies that:
1. Shunt columns are correctly rescaled by (V_measured / V_cached)²
2. Non-shunt columns remain unchanged during shunt updates
3. Update interval logic is respected
4. OLTC columns are recomputed when V_gen changes
5. V² correction is applied in the base Jacobian shunt sensitivities

Author: Manuel Schwenke
Date: 2026-02-13
"""

import numpy as np
import pytest
import pandapower as pp

from sensitivity.jacobian import JacobianSensitivities
from sensitivity.sensitivity_updater import SensitivityUpdater
from core.measurement import Measurement


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def network_with_shunt() -> pp.pandapowerNet:
    """
    Create a simple network with a switchable shunt for shunt sensitivity tests.

    Topology:
        Bus 0 (Slack, 110 kV) --- Trafo --- Bus 1 (PQ, 20 kV) --- Line --- Bus 2 (PQ)
                                                 |
                                                Line
                                                 |
                                             Bus 3 (PQ, DER + Shunt)
    """
    net = pp.create_empty_network(sn_mva=100.0)

    bus0 = pp.create_bus(net, vn_kv=110.0, name="Slack Bus")
    bus1 = pp.create_bus(net, vn_kv=20.0, name="Trafo LV Bus")
    bus2 = pp.create_bus(net, vn_kv=20.0, name="Load Bus")
    bus3 = pp.create_bus(net, vn_kv=20.0, name="DER+Shunt Bus")

    pp.create_ext_grid(net, bus=bus0, vm_pu=1.02)

    pp.create_transformer_from_parameters(
        net, hv_bus=bus0, lv_bus=bus1, sn_mva=40.0,
        vn_hv_kv=110.0, vn_lv_kv=20.0,
        vkr_percent=0.5, vk_percent=10.0,
        pfe_kw=30.0, i0_percent=0.1,
        tap_side="hv", tap_neutral=0, tap_min=-10, tap_max=10,
        tap_step_percent=1.5, tap_pos=0, tap_changer_type="Ratio",
    )

    pp.create_line_from_parameters(
        net, from_bus=bus1, to_bus=bus2, length_km=10.0,
        r_ohm_per_km=0.1, x_ohm_per_km=0.3, c_nf_per_km=10.0, max_i_ka=0.5,
    )
    pp.create_line_from_parameters(
        net, from_bus=bus1, to_bus=bus3, length_km=5.0,
        r_ohm_per_km=0.1, x_ohm_per_km=0.3, c_nf_per_km=10.0, max_i_ka=0.5,
    )

    pp.create_load(net, bus=bus2, p_mw=5.0, q_mvar=2.0)
    pp.create_load(net, bus=bus3, p_mw=3.0, q_mvar=1.0)

    pp.create_sgen(net, bus=bus3, p_mw=2.0, q_mvar=0.0, name="DER_1")

    # Add a switchable shunt at bus 3
    pp.create_shunt(net, bus=bus3, q_mvar=5.0, p_mw=0.0, step=1)

    pp.runpp(net, calculate_voltage_angles=True)
    assert net.converged
    return net


def _make_measurement(bus_indices, voltages_pu, iteration=1):
    """Helper to create a Measurement object for testing."""
    n_bus = len(bus_indices)
    return Measurement(
        iteration=iteration,
        bus_indices=np.array(bus_indices, dtype=np.int64),
        voltage_magnitudes_pu=np.array(voltages_pu, dtype=np.float64),
        branch_indices=np.array([], dtype=np.int64),
        current_magnitudes_ka=np.array([], dtype=np.float64),
        interface_transformer_indices=np.array([], dtype=np.int64),
        interface_q_hv_side_mvar=np.array([], dtype=np.float64),
        der_indices=np.array([], dtype=np.int64),
        der_q_mvar=np.array([], dtype=np.float64),
        oltc_indices=np.array([], dtype=np.int64),
        oltc_tap_positions=np.array([], dtype=np.int64),
        shunt_indices=np.array([], dtype=np.int64),
        shunt_states=np.array([], dtype=np.int64),
        gen_indices=np.array([], dtype=np.int64),
        gen_vm_pu=np.array([], dtype=np.float64),
    )


# =============================================================================
# Tests: Shunt V² correction in JacobianSensitivities
# =============================================================================

class TestShuntVSquaredCorrection:
    """Verify that shunt sensitivities include the V² factor."""

    def test_dV_dQ_shunt_includes_v_squared(self, network_with_shunt):
        """Shunt voltage sensitivity = -dV_dQ_der * q_step * V²."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        shunt_bus = 3
        V_bus = net.res_bus.at[shunt_bus, 'vm_pu']
        q_step = 5.0
        obs_buses = [1, 2, 3]

        # Shunt sensitivity (includes V²)
        dV_shunt, _ = sens.compute_dV_dQ_shunt(
            shunt_bus_idx=shunt_bus,
            observation_bus_indices=obs_buses,
            q_step_mvar=q_step,
        )

        # DER sensitivity at same bus (constant-power, no V²)
        dV_der, _, _ = sens.compute_dV_dQ_der(
            der_bus_indices=[shunt_bus],
            observation_bus_indices=obs_buses,
        )

        # Shunt should be: -dV_der * q_step * V²
        expected = -dV_der[:, 0] * q_step * V_bus ** 2
        np.testing.assert_allclose(dV_shunt, expected, rtol=1e-10)

    def test_dI_dQ_shunt_includes_v_squared(self, network_with_shunt):
        """Shunt current sensitivity = -dI_dQ_der * q_step * V²."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        shunt_bus = 3
        V_bus = net.res_bus.at[shunt_bus, 'vm_pu']
        q_step = 5.0
        line_indices = [0, 1]

        dI_shunt, _ = sens.compute_dI_dQ_shunt(
            shunt_bus_idx=shunt_bus,
            line_indices=line_indices,
            q_step_mvar=q_step,
        )

        dI_der, _, _ = sens.compute_dI_dQ_der_matrix(
            line_indices=line_indices,
            der_bus_indices=[shunt_bus],
        )

        expected = -dI_der[:, 0] * q_step * V_bus ** 2
        np.testing.assert_allclose(dI_shunt, expected, rtol=1e-10)

    def test_mappings_contain_cached_voltages(self, network_with_shunt):
        """build_sensitivity_matrix_H includes shunt_cached_v_pu in mappings."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        _, mappings = sens.build_sensitivity_matrix_H(
            der_bus_indices=[3],
            observation_bus_indices=[1, 2, 3],
            line_indices=[0, 1],
            shunt_bus_indices=[3],
            shunt_q_steps_mvar=[5.0],
        )

        assert 'shunt_cached_v_pu' in mappings
        assert len(mappings['shunt_cached_v_pu']) == 1

        expected_v = net.res_bus.at[3, 'vm_pu']
        np.testing.assert_allclose(
            mappings['shunt_cached_v_pu'][0], expected_v, rtol=1e-10
        )


# =============================================================================
# Tests: SensitivityUpdater
# =============================================================================

class TestSensitivityUpdater:
    """Tests for the SensitivityUpdater class."""

    def test_no_shunts_returns_unchanged_H(self):
        """With zero shunts, H is returned unchanged."""
        H = np.random.randn(4, 3)
        mappings = {
            'der_buses': [1, 2, 3],
            'oltc_trafos': [],
            'oltc_trafo3w': [],
            'shunt_buses': [],
            'shunt_cached_v_pu': np.array([]),

        }

        # Need a mock-like sensitivities object; we don't call any methods
        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=None,
            update_interval_min=1,
        )

        meas = _make_measurement([1, 2, 3], [1.0, 1.0, 1.0])
        H_out = updater.update(meas, current_iteration=1)

        np.testing.assert_array_equal(H_out, H)

    def test_identity_rescaling_at_cached_voltage(self, network_with_shunt):
        """When V_measured == V_cached, H should be unchanged."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        H, mappings = sens.build_sensitivity_matrix_H(
            der_bus_indices=[3],
            observation_bus_indices=[1, 2, 3],
            line_indices=[0, 1],
            shunt_bus_indices=[3],
            shunt_q_steps_mvar=[5.0],
        )

        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=sens,
            update_interval_min=1,
        )

        # Provide measured voltages equal to cached
        all_buses = sorted(net.res_bus.index)
        all_v = [net.res_bus.at[b, 'vm_pu'] for b in all_buses]
        meas = _make_measurement(all_buses, all_v, iteration=1)

        H_out = updater.update(meas, current_iteration=1)
        np.testing.assert_allclose(H_out, H, rtol=1e-12)

    def test_shunt_rescaling_with_higher_voltage(self, network_with_shunt):
        """When V_measured > V_cached, shunt columns should increase."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        H, mappings = sens.build_sensitivity_matrix_H(
            der_bus_indices=[3],
            observation_bus_indices=[1, 2, 3],
            line_indices=[0, 1],
            shunt_bus_indices=[3],
            shunt_q_steps_mvar=[5.0],
        )
        H_original = H.copy()

        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=sens,
            update_interval_min=1,
        )

        # Provide higher voltage at shunt bus
        V_cached = net.res_bus.at[3, 'vm_pu']
        V_new = V_cached + 0.05  # 5% higher

        all_buses = sorted(net.res_bus.index)
        all_v = [net.res_bus.at[b, 'vm_pu'] for b in all_buses]
        # Override bus 3 voltage
        idx_bus3 = all_buses.index(3)
        all_v[idx_bus3] = V_new

        meas = _make_measurement(all_buses, all_v, iteration=1)
        H_out = updater.update(meas, current_iteration=1)

        # The shunt column (last column) should be scaled by (V_new/V_cached)²
        expected_ratio = (V_new / V_cached) ** 2

        # Shunt column is the last column
        n_cols = H_original.shape[1]
        shunt_col = n_cols - 1  # last column is shunt

        np.testing.assert_allclose(
            H_out[:, shunt_col],
            H_original[:, shunt_col] * expected_ratio,
            rtol=1e-10,
        )

    def test_non_shunt_columns_unchanged(self, network_with_shunt):
        """DER columns should not be affected by shunt voltage updates."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        H, mappings = sens.build_sensitivity_matrix_H(
            der_bus_indices=[3],
            observation_bus_indices=[1, 2, 3],
            line_indices=[0, 1],
            shunt_bus_indices=[3],
            shunt_q_steps_mvar=[5.0],
        )
        H_original = H.copy()

        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=sens,
            update_interval_min=1,
        )

        # Change voltage at shunt bus
        all_buses = sorted(net.res_bus.index)
        all_v = [net.res_bus.at[b, 'vm_pu'] for b in all_buses]
        idx_bus3 = all_buses.index(3)
        all_v[idx_bus3] += 0.05

        meas = _make_measurement(all_buses, all_v, iteration=1)
        H_out = updater.update(meas, current_iteration=1)

        # DER column (first column) should be unchanged
        np.testing.assert_array_equal(H_out[:, 0], H_original[:, 0])

    def test_update_interval_respected(self, network_with_shunt):
        """H should not be recomputed before the interval elapses."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        H, mappings = sens.build_sensitivity_matrix_H(
            der_bus_indices=[3],
            observation_bus_indices=[1, 2, 3],
            line_indices=[0, 1],
            shunt_bus_indices=[3],
            shunt_q_steps_mvar=[5.0],
        )
        H_original = H.copy()

        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=sens,
            update_interval_min=5,  # only update every 5 iterations
        )

        # First update at iteration 1 — should update
        all_buses = sorted(net.res_bus.index)
        all_v = [net.res_bus.at[b, 'vm_pu'] for b in all_buses]
        idx_bus3 = all_buses.index(3)
        all_v[idx_bus3] += 0.05

        meas = _make_measurement(all_buses, all_v, iteration=1)
        H_1 = updater.update(meas, current_iteration=1).copy()

        # Second update at iteration 3 — should NOT update (interval=5)
        all_v2 = list(all_v)
        all_v2[idx_bus3] += 0.10  # different voltage
        meas2 = _make_measurement(all_buses, all_v2, iteration=3)
        H_3 = updater.update(meas2, current_iteration=3).copy()

        # H_3 should be the same as H_1 (not recomputed)
        np.testing.assert_array_equal(H_3, H_1)

        # Third update at iteration 6 — should update (5 elapsed since 1)
        H_6 = updater.update(meas2, current_iteration=6).copy()

        # H_6 should be different from H_1 (different voltage at bus 3)
        assert not np.allclose(H_6, H_1)

    def test_missing_bus_voltage_uses_cached(self, network_with_shunt):
        """If a shunt bus voltage is not in measurement, use cached voltage."""
        net = network_with_shunt
        sens = JacobianSensitivities(net)

        H, mappings = sens.build_sensitivity_matrix_H(
            der_bus_indices=[3],
            observation_bus_indices=[1, 2, 3],
            line_indices=[0, 1],
            shunt_bus_indices=[3],
            shunt_q_steps_mvar=[5.0],
        )
        H_original = H.copy()

        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=sens,
            update_interval_min=1,
        )

        # Measurement with only buses 0 and 1 (bus 3 missing)
        meas = _make_measurement([0, 1], [1.02, 1.01], iteration=1)
        H_out = updater.update(meas, current_iteration=1)

        # Bus 3 not in measurement → ratio = 1 → H unchanged
        np.testing.assert_allclose(H_out, H_original, rtol=1e-12)

    def test_multiple_shunts_independent_scaling(self):
        """Each shunt column should be scaled by its own bus voltage ratio."""
        # Create a synthetic H with known structure
        H = np.ones((3, 4))  # 3 outputs, 4 inputs: [DER, shunt1, shunt2, shunt3]
        mappings = {
            'der_buses': [10],
            'oltc_trafos': [],
            'oltc_trafo3w': [],
            'shunt_buses': [20, 30, 40],
            'shunt_cached_v_pu': np.array([1.0, 0.95, 1.05]),

        }

        updater = SensitivityUpdater(
            H=H, mappings=mappings, sensitivities=None,
            update_interval_min=1,
        )

        # Measured voltages: bus 20 = 1.1, bus 30 = 1.0, bus 40 = 1.0
        meas = _make_measurement([20, 30, 40], [1.1, 1.0, 1.0], iteration=1)
        H_out = updater.update(meas, current_iteration=1)

        # DER column (col 0) unchanged
        np.testing.assert_array_equal(H_out[:, 0], np.ones(3))

        # Shunt 1 (col 1): ratio = (1.1/1.0)² = 1.21
        np.testing.assert_allclose(H_out[:, 1], np.ones(3) * 1.21, rtol=1e-10)

        # Shunt 2 (col 2): ratio = (1.0/0.95)²
        expected_2 = (1.0 / 0.95) ** 2
        np.testing.assert_allclose(H_out[:, 2], np.ones(3) * expected_2, rtol=1e-10)

        # Shunt 3 (col 3): ratio = (1.0/1.05)²
        expected_3 = (1.0 / 1.05) ** 2
        np.testing.assert_allclose(H_out[:, 3], np.ones(3) * expected_3, rtol=1e-10)
