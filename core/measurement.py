"""
Measurement Module
==================

This module defines the Measurement class, which represents the measured
quantities received by a controller at each OFO iteration.

Measurements are obtained from the physical system (or simulation via pandapower)
and provide the feedback signal for the closed-loop OFO controller.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import numpy as np
from numpy.typing import NDArray


class Measurement:
    """
    Container for measurements received by a controller at iteration k.
    
    This class holds all measured quantities from the physical system that
    are needed for the OFO control loop. Measurements include bus voltages,
    branch currents, interface reactive power flows, and current actuator
    states.
    
    Attributes
    ----------
    iteration : int
        The OFO iteration index k at which this measurement was taken.
    bus_indices : NDArray[np.int64]
        Pandapower bus indices where voltage is measured.
    voltage_magnitudes_pu : NDArray[np.float64]
        Measured voltage magnitudes in per-unit.
    branch_indices : NDArray[np.int64]
        Pandapower branch (line) indices where current is measured.
    current_magnitudes_ka : NDArray[np.float64]
        Measured current magnitudes in kA.
    interface_transformer_indices : NDArray[np.int64]
        Pandapower transformer indices for TSO-DSO coupling transformers.
    interface_q_hv_side_mvar : NDArray[np.float64]
        Measured reactive power flow at HV side of interface transformers in Mvar.
        Positive value indicates Q flowing into the HV bus from the transformer.
    der_indices : NDArray[np.int64]
        Indices of DERs (as sgen or gen in pandapower).
    der_q_mvar : NDArray[np.float64]
        Current reactive power output of DERs in Mvar.
    oltc_indices : NDArray[np.int64]
        Pandapower transformer indices with OLTCs.
    oltc_tap_positions : NDArray[np.int64]
        Current tap positions of OLTCs.
    shunt_indices : NDArray[np.int64]
        Pandapower shunt indices.
    shunt_states : NDArray[np.int64]
        Current shunt switching states (-1: capacitor, 0: off, 1: reactor).
    """

    def __init__(
        self,
        iteration: int,
        bus_indices: NDArray[np.int64],
        voltage_magnitudes_pu: NDArray[np.float64],
        branch_indices: NDArray[np.int64],
        current_magnitudes_ka: NDArray[np.float64],
        interface_transformer_indices: NDArray[np.int64],
        interface_q_hv_side_mvar: NDArray[np.float64],
        der_indices: NDArray[np.int64],
        der_q_mvar: NDArray[np.float64],
        oltc_indices: NDArray[np.int64],
        oltc_tap_positions: NDArray[np.int64],
        shunt_indices: NDArray[np.int64],
        shunt_states: NDArray[np.int64],
        gen_indices: NDArray[np.int64],
        gen_vm_pu: NDArray[np.float64]
    ) -> None:
        """
        Initialise a Measurement instance.
        
        Parameters
        ----------
        iteration : int
            OFO iteration index.
        bus_indices : NDArray[np.int64]
            Bus indices for voltage measurements.
        voltage_magnitudes_pu : NDArray[np.float64]
            Voltage magnitudes in per-unit.
        branch_indices : NDArray[np.int64]
            Branch indices for current measurements.
        current_magnitudes_ka : NDArray[np.float64]
            Current magnitudes in kA.
        interface_transformer_indices : NDArray[np.int64]
            TSO-DSO coupling transformer indices.
        interface_q_hv_side_mvar : NDArray[np.float64]
            Reactive power flow at HV side of interface transformers in Mvar.
        der_indices : NDArray[np.int64]
            DER indices.
        der_q_mvar : NDArray[np.float64]
            DER reactive power outputs in Mvar.
        oltc_indices : NDArray[np.int64]
            OLTC transformer indices.
        oltc_tap_positions : NDArray[np.int64]
            OLTC tap positions.
        shunt_indices : NDArray[np.int64]
            Shunt indices.
        shunt_states : NDArray[np.int64]
            Shunt states.
        gen_indices : NDArray[np.int64]
            Generator indices.
        gen_vm_pu : NDArray[np.int64]
            Generator AVR voltage.
        """
        self.iteration = iteration
        self.bus_indices = bus_indices
        self.voltage_magnitudes_pu = voltage_magnitudes_pu
        self.branch_indices = branch_indices
        self.current_magnitudes_ka = current_magnitudes_ka
        self.interface_transformer_indices = interface_transformer_indices
        self.interface_q_hv_side_mvar = interface_q_hv_side_mvar
        self.der_indices = der_indices
        self.der_q_mvar = der_q_mvar
        self.oltc_indices = oltc_indices
        self.oltc_tap_positions = oltc_tap_positions
        self.shunt_indices = shunt_indices
        self.shunt_states = shunt_states
        self.gen_indices = gen_indices
        self.gen_vm_pu = gen_vm_pu
    
    @property
    def n_bus_measurements(self) -> int:
        """Return the number of bus voltage measurements."""
        return len(self.bus_indices)
    
    @property
    def n_branch_measurements(self) -> int:
        """Return the number of branch current measurements."""
        return len(self.branch_indices)
    
    @property
    def n_interface_measurements(self) -> int:
        """Return the number of interface transformer Q measurements."""
        return len(self.interface_transformer_indices)
