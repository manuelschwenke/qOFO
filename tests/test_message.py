"""
Tests for Message classes.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
import pytest

from core.message import SetpointMessage, CapabilityMessage


class TestSetpointMessage:
    """Test cases for SetpointMessage class."""
    
    def test_create_setpoint_message(self):
        """Test basic SetpointMessage creation."""
        message = SetpointMessage(
            source_controller_id="TSO_1",
            target_controller_id="DSO_1",
            iteration=10,
            interface_transformer_indices=np.array([0, 1], dtype=np.int64),
            q_setpoints_mvar=np.array([20.0, -15.0], dtype=np.float64),
        )
        
        assert message.source_controller_id == "TSO_1"
        assert message.target_controller_id == "DSO_1"
        assert message.iteration == 10
        assert message.n_interfaces == 2
    
    def test_single_interface(self):
        """Test SetpointMessage with single interface."""
        message = SetpointMessage(
            source_controller_id="TSO_1",
            target_controller_id="DSO_2",
            iteration=0,
            interface_transformer_indices=np.array([5], dtype=np.int64),
            q_setpoints_mvar=np.array([50.0], dtype=np.float64),
        )
        
        assert message.n_interfaces == 1
        assert message.q_setpoints_mvar[0] == 50.0


class TestCapabilityMessage:
    """Test cases for CapabilityMessage class."""
    
    def test_create_capability_message(self):
        """Test basic CapabilityMessage creation."""
        message = CapabilityMessage(
            source_controller_id="DSO_1",
            target_controller_id="TSO_1",
            iteration=10,
            interface_transformer_indices=np.array([0, 1], dtype=np.int64),
            q_min_mvar=np.array([-30.0, -20.0], dtype=np.float64),
            q_max_mvar=np.array([30.0, 20.0], dtype=np.float64),
        )
        
        assert message.source_controller_id == "DSO_1"
        assert message.target_controller_id == "TSO_1"
        assert message.iteration == 10
        assert message.n_interfaces == 2
    
    def test_asymmetric_bounds(self):
        """Test CapabilityMessage with asymmetric Q bounds."""
        message = CapabilityMessage(
            source_controller_id="DSO_1",
            target_controller_id="TSO_1",
            iteration=5,
            interface_transformer_indices=np.array([0], dtype=np.int64),
            q_min_mvar=np.array([-10.0], dtype=np.float64),
            q_max_mvar=np.array([50.0], dtype=np.float64),
        )
        
        assert message.q_min_mvar[0] == -10.0
        assert message.q_max_mvar[0] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
