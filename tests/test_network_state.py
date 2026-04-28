"""
Tests for NetworkState class.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
import pytest

from core.network_state import NetworkState


class TestNetworkState:
    """Test cases for NetworkState class."""
    
    def test_create_network_state(self):
        """Test basic NetworkState creation."""
        state = NetworkState(
            bus_indices=np.array([0, 1, 2], dtype=np.int64),
            voltage_magnitudes_pu=np.array([1.02, 1.01, 0.99], dtype=np.float64),
            voltage_angles_rad=np.array([0.0, -0.02, -0.05], dtype=np.float64),
            slack_bus_index=0,
            pv_bus_indices=np.array([1], dtype=np.int64),
            pq_bus_indices=np.array([2], dtype=np.int64),
            transformer_indices=np.array([0], dtype=np.int64),
            tap_positions=np.array([0.0], dtype=np.float64),
            source_case="test_network",
            timestamp="2025-02-05T10:30:00",
            cached_at_iteration=0,
        )
        
        assert state.n_buses == 3
        assert state.n_transformers == 1
        assert state.slack_bus_index == 0
    
    def test_n_buses_property(self):
        """Test n_buses property returns correct count."""
        state = NetworkState(
            bus_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            voltage_magnitudes_pu=np.ones(5, dtype=np.float64),
            voltage_angles_rad=np.zeros(5, dtype=np.float64),
            slack_bus_index=0,
            pv_bus_indices=np.array([], dtype=np.int64),
            pq_bus_indices=np.array([1, 2, 3, 4], dtype=np.int64),
            transformer_indices=np.array([], dtype=np.int64),
            tap_positions=np.array([], dtype=np.float64),
            source_case="test",
            timestamp="2025-02-05T10:30:00",
            cached_at_iteration=0,
        )
        
        assert state.n_buses == 5
    
    def test_n_transformers_property(self):
        """Test n_transformers property returns correct count."""
        state = NetworkState(
            bus_indices=np.array([0, 1], dtype=np.int64),
            voltage_magnitudes_pu=np.ones(2, dtype=np.float64),
            voltage_angles_rad=np.zeros(2, dtype=np.float64),
            slack_bus_index=0,
            pv_bus_indices=np.array([], dtype=np.int64),
            pq_bus_indices=np.array([1], dtype=np.int64),
            transformer_indices=np.array([0, 1, 2], dtype=np.int64),
            tap_positions=np.array([0.0, 1.0, -1.0], dtype=np.float64),
            source_case="test",
            timestamp="2025-02-05T10:30:00",
            cached_at_iteration=0,
        )
        
        assert state.n_transformers == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
