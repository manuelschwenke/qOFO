"""
Tests for Measurement class.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
import pytest

from core.measurement import Measurement


class TestMeasurement:
    """Test cases for Measurement class."""
    
    def test_create_measurement(self):
        """Test basic Measurement creation."""
        measurement = Measurement(
            iteration=0,
            bus_indices=np.array([0, 1, 2], dtype=np.int64),
            voltage_magnitudes_pu=np.array([1.02, 1.01, 0.99], dtype=np.float64),
            branch_indices=np.array([0, 1], dtype=np.int64),
            current_magnitudes_ka=np.array([0.5, 0.3], dtype=np.float64),
            interface_transformer_indices=np.array([0], dtype=np.int64),
            interface_q_hv_side_mvar=np.array([10.0], dtype=np.float64),
            der_indices=np.array([0, 1], dtype=np.int64),
            der_q_mvar=np.array([5.0, -3.0], dtype=np.float64),
            oltc_indices=np.array([0], dtype=np.int64),
            oltc_tap_positions=np.array([0], dtype=np.int64),
            shunt_indices=np.array([0], dtype=np.int64),
            shunt_states=np.array([0], dtype=np.int64),
        )
        
        assert measurement.iteration == 0
        assert measurement.n_bus_measurements == 3
        assert measurement.n_branch_measurements == 2
        assert measurement.n_interface_measurements == 1
    
    def test_empty_measurements(self):
        """Test Measurement with empty arrays (valid case)."""
        measurement = Measurement(
            iteration=5,
            bus_indices=np.array([], dtype=np.int64),
            voltage_magnitudes_pu=np.array([], dtype=np.float64),
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
        )
        
        assert measurement.iteration == 5
        assert measurement.n_bus_measurements == 0
        assert measurement.n_branch_measurements == 0
        assert measurement.n_interface_measurements == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
