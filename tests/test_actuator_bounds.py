"""
Tests for ActuatorBounds class.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
import pytest

from core.actuator_bounds import ActuatorBounds


class TestActuatorBounds:
    """Test cases for ActuatorBounds class."""
    
    def test_create_actuator_bounds(self):
        """Test basic ActuatorBounds creation."""
        bounds = ActuatorBounds(
            der_indices=np.array([0, 1], dtype=np.int64),
            der_s_rated_mva=np.array([100.0, 50.0], dtype=np.float64),
            der_p_max_mw=np.array([100.0, 50.0], dtype=np.float64),
            oltc_indices=np.array([0], dtype=np.int64),
            oltc_tap_min=np.array([-16], dtype=np.int64),
            oltc_tap_max=np.array([16], dtype=np.int64),
            shunt_indices=np.array([0, 1], dtype=np.int64),
            shunt_q_mvar=np.array([50.0, 50.0], dtype=np.float64),
        )
        
        assert bounds.n_ders == 2
        assert bounds.n_oltcs == 1
        assert bounds.n_shunts == 2
    
    def test_der_q_bounds_full_power(self):
        """Test DER Q bounds at full active power output."""
        bounds = ActuatorBounds(
            der_indices=np.array([0], dtype=np.int64),
            der_s_rated_mva=np.array([100.0], dtype=np.float64),
            der_p_max_mw=np.array([100.0], dtype=np.float64),
            oltc_indices=np.array([], dtype=np.int64),
            oltc_tap_min=np.array([], dtype=np.int64),
            oltc_tap_max=np.array([], dtype=np.int64),
            shunt_indices=np.array([], dtype=np.int64),
            shunt_q_mvar=np.array([], dtype=np.float64),
        )
        
        # At full power (P = P_max), Q capability should be ±0.33 * S_rated
        q_min, q_max = bounds.compute_der_q_bounds(
            der_p_current_mw=np.array([100.0], dtype=np.float64)
        )
        
        expected_q = 0.33 * 100.0  # 33 Mvar
        assert np.isclose(q_max[0], expected_q, rtol=0.01)
        assert np.isclose(q_min[0], -expected_q, rtol=0.01)
    
    def test_der_q_bounds_zero_power(self):
        """Test DER Q bounds at zero active power output."""
        bounds = ActuatorBounds(
            der_indices=np.array([0], dtype=np.int64),
            der_s_rated_mva=np.array([100.0], dtype=np.float64),
            der_p_max_mw=np.array([100.0], dtype=np.float64),
            oltc_indices=np.array([], dtype=np.int64),
            oltc_tap_min=np.array([], dtype=np.int64),
            oltc_tap_max=np.array([], dtype=np.int64),
            shunt_indices=np.array([], dtype=np.int64),
            shunt_q_mvar=np.array([], dtype=np.float64),
        )
        
        # At zero power, Q capability should be zero (or very small)
        q_min, q_max = bounds.compute_der_q_bounds(
            der_p_current_mw=np.array([0.0], dtype=np.float64)
        )
        
        assert np.isclose(q_max[0], 0.0, atol=0.01)
        assert np.isclose(q_min[0], 0.0, atol=0.01)
    
    def test_der_q_bounds_partial_power(self):
        """Test DER Q bounds at partial active power output."""
        bounds = ActuatorBounds(
            der_indices=np.array([0], dtype=np.int64),
            der_s_rated_mva=np.array([100.0], dtype=np.float64),
            der_p_max_mw=np.array([100.0], dtype=np.float64),
            oltc_indices=np.array([], dtype=np.int64),
            oltc_tap_min=np.array([], dtype=np.int64),
            oltc_tap_max=np.array([], dtype=np.int64),
            shunt_indices=np.array([], dtype=np.int64),
            shunt_q_mvar=np.array([], dtype=np.float64),
        )
        
        # At P = 0.5 * P_max (above threshold), full Q capability
        q_min, q_max = bounds.compute_der_q_bounds(
            der_p_current_mw=np.array([50.0], dtype=np.float64)
        )
        
        expected_q = 0.33 * 100.0  # 33 Mvar
        assert np.isclose(q_max[0], expected_q, rtol=0.01)
        assert np.isclose(q_min[0], -expected_q, rtol=0.01)
    
    def test_oltc_tap_bounds(self):
        """Test OLTC tap position bounds."""
        bounds = ActuatorBounds(
            der_indices=np.array([], dtype=np.int64),
            der_s_rated_mva=np.array([], dtype=np.float64),
            der_p_max_mw=np.array([], dtype=np.float64),
            oltc_indices=np.array([0, 1], dtype=np.int64),
            oltc_tap_min=np.array([-16, -10], dtype=np.int64),
            oltc_tap_max=np.array([16, 10], dtype=np.int64),
            shunt_indices=np.array([], dtype=np.int64),
            shunt_q_mvar=np.array([], dtype=np.float64),
        )
        
        tap_min, tap_max = bounds.get_oltc_tap_bounds()
        
        assert tap_min[0] == -16
        assert tap_max[0] == 16
        assert tap_min[1] == -10
        assert tap_max[1] == 10
    
    def test_shunt_state_bounds(self):
        """Test shunt state bounds."""
        bounds = ActuatorBounds(
            der_indices=np.array([], dtype=np.int64),
            der_s_rated_mva=np.array([], dtype=np.float64),
            der_p_max_mw=np.array([], dtype=np.float64),
            oltc_indices=np.array([], dtype=np.int64),
            oltc_tap_min=np.array([], dtype=np.int64),
            oltc_tap_max=np.array([], dtype=np.int64),
            shunt_indices=np.array([0, 1, 2], dtype=np.int64),
            shunt_q_mvar=np.array([50.0, 50.0, 100.0], dtype=np.float64),
        )
        
        state_min, state_max = bounds.get_shunt_state_bounds()
        
        assert len(state_min) == 3
        assert len(state_max) == 3
        assert all(state_min == -1)
        assert all(state_max == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
