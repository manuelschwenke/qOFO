"""
Network State Module
====================

This module defines the NetworkState class, which represents a cached snapshot
of the electrical network state at a specific operating point.

The NetworkState is used as the basis for computing Jacobian-based sensitivities.

Note on TSO-DSO Separation
--------------------------
TSO and DSO controllers maintain separate network state representations:

- TSO controllers represent DSO areas as PQ nodes (fixed P, Q injection)
- DSO controllers represent the TSO interface as PV nodes (fixed P, V magnitude)
  with one bus selected as slack (reference angle)

This reflects real-world operational boundaries where TSO and DSO do not exchange
detailed network models.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
from numpy.typing import NDArray


class NetworkState:
    """
    Snapshot of network state for Jacobian-based sensitivity computation.
    
    This class stores the voltage magnitudes and angles at all buses, along with
    transformer tap positions. The state must originate from a converged power
    flow solution to ensure consistency.
    
    Attributes
    ----------
    bus_indices : NDArray[np.int64]
        Array of pandapower bus indices corresponding to the state vectors.
    voltage_magnitudes_pu : NDArray[np.float64]
        Bus voltage magnitudes in per-unit, ordered according to bus_indices.
    voltage_angles_rad : NDArray[np.float64]
        Bus voltage angles in radians, ordered according to bus_indices.
    slack_bus_index : int
        The pandapower bus index of the slack (reference) bus.
    pv_bus_indices : NDArray[np.int64]
        Pandapower bus indices of PV buses (voltage-controlled).
    pq_bus_indices : NDArray[np.int64]
        Pandapower bus indices of PQ buses (load buses).
    transformer_indices : NDArray[np.int64]
        Pandapower transformer indices for OLTCs.
    tap_positions : NDArray[np.float64]
        Current tap positions of transformers, ordered according to transformer_indices.
    source_case : str
        Identifier string for the source network or case name.
    timestamp : str
        ISO 8601 formatted timestamp of when this state was captured.
    cached_at_iteration : int
        The OFO iteration index at which this state was cached.
    """
    
    def __init__(
        self,
        bus_indices: NDArray[np.int64],
        voltage_magnitudes_pu: NDArray[np.float64],
        voltage_angles_rad: NDArray[np.float64],
        slack_bus_index: int,
        pv_bus_indices: NDArray[np.int64],
        pq_bus_indices: NDArray[np.int64],
        transformer_indices: NDArray[np.int64],
        tap_positions: NDArray[np.float64],
        source_case: str,
        timestamp: str,
        cached_at_iteration: int,
    ) -> None:
        """
        Initialise a NetworkState instance.
        
        Parameters
        ----------
        bus_indices : NDArray[np.int64]
            Pandapower bus indices.
        voltage_magnitudes_pu : NDArray[np.float64]
            Voltage magnitudes in per-unit.
        voltage_angles_rad : NDArray[np.float64]
            Voltage angles in radians.
        slack_bus_index : int
            Index of the slack bus.
        pv_bus_indices : NDArray[np.int64]
            Indices of PV buses.
        pq_bus_indices : NDArray[np.int64]
            Indices of PQ buses.
        transformer_indices : NDArray[np.int64]
            Indices of transformers with OLTCs.
        tap_positions : NDArray[np.float64]
            Current tap positions.
        source_case : str
            Identifier for the source case.
        timestamp : str
            ISO 8601 timestamp.
        cached_at_iteration : int
            OFO iteration index.
        """
        self.bus_indices = bus_indices
        self.voltage_magnitudes_pu = voltage_magnitudes_pu
        self.voltage_angles_rad = voltage_angles_rad
        self.slack_bus_index = slack_bus_index
        self.pv_bus_indices = pv_bus_indices
        self.pq_bus_indices = pq_bus_indices
        self.transformer_indices = transformer_indices
        self.tap_positions = tap_positions
        self.source_case = source_case
        self.timestamp = timestamp
        self.cached_at_iteration = cached_at_iteration
    
    @property
    def n_buses(self) -> int:
        """Return the number of buses in this network state."""
        return len(self.bus_indices)
    
    @property
    def n_transformers(self) -> int:
        """Return the number of transformers with tap changers."""
        return len(self.transformer_indices)
