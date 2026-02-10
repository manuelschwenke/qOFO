"""
Core Module
============

This module provides the core data structures for the cascaded OFO controller.

Classes
-------
NetworkState
    Cached network state for Jacobian-based sensitivity computation.
Measurement
    Runtime measurements from the physical system.
SetpointMessage
    Message from TSO to DSO containing reactive power setpoints.
CapabilityMessage
    Message from DSO to TSO containing reactive power capability bounds.
ActuatorBounds
    Calculator for operating-point-dependent actuator bounds.
"""

from core.network_state import NetworkState
from core.measurement import Measurement
from core.message import SetpointMessage, CapabilityMessage
from core.actuator_bounds import ActuatorBounds

__all__ = [
    "NetworkState",
    "Measurement",
    "SetpointMessage",
    "CapabilityMessage",
    "ActuatorBounds",
]
