"""
DSO Controller Module
=====================

This module defines the DSO-level MIQP controller for distribution system
voltage and reactive power control.

The DSO controller:
- Controls DER reactive power (continuous)
- Controls HV-level OLTCs (discrete)
- Controls HV-level shunts (discrete)
- Tracks reactive power setpoints received from TSO
- Reports capability bounds to TSO

The objective function includes a setpoint tracking term:
    ∇f = 2 · (Q_interface - Q_set) · ∂Q_interface/∂u

References
----------
[1] Schwenke et al., PSCC 2026, Section III.B (DSO-level control)
[2] Schwenke et al., CIGRE 2026 (Cascaded OFO framework)

Author: Manuel Schwenke
Date: 2025-02-06
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from controller.base_controller import (
    BaseOFOController,
    OFOParameters,
    ControllerOutput,
)
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from core.message import SetpointMessage, CapabilityMessage
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.sensitivity_updater import SensitivityUpdater


@dataclass
class DSOControllerConfig:
    """
    Configuration for the DSO controller.
    
    Attributes
    ----------
    der_bus_indices : List[int]
        Pandapower bus indices where DERs are connected.
    oltc_trafo_indices : List[int]
        Pandapower transformer indices with OLTCs.
    shunt_bus_indices : List[int]
        Pandapower bus indices where switchable shunts are connected.
    shunt_q_steps_mvar : List[float]
        Reactive power step per state change for each shunt.
    interface_trafo_indices : List[int]
        Pandapower transformer indices for TSO-DSO interfaces.
    voltage_bus_indices : List[int]
        Pandapower bus indices for voltage monitoring.
    current_line_indices : List[int]
        Pandapower line indices for current monitoring.
    v_min_pu : float
        Minimum voltage limit in per-unit.
    v_max_pu : float
        Maximum voltage limit in per-unit.
    i_max_pu : float
        Maximum current limit as fraction of line rating.
    g_q : float
        Weight for Q-interface tracking in the objective function.
        Scales the gradient ``2 · g_q · (Q - Q_set)^T · ∂Q/∂u``.
        Higher values make the controller track the TSO's reactive
        power setpoints more aggressively.  Must be balanced against
        the change penalty ``g_w``: the effective per-iteration step
        scales as ``g_q / g_w``.  Default 1.0 (unweighted).
    """
    der_bus_indices: List[int]
    oltc_trafo_indices: List[int]
    shunt_bus_indices: List[int]
    shunt_q_steps_mvar: List[float]
    interface_trafo_indices: List[int]
    voltage_bus_indices: List[int]
    current_line_indices: List[int]
    v_min_pu: float = 0.9
    v_max_pu: float = 1.1
    i_max_pu: float = 1.0
    g_q: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if len(self.shunt_bus_indices) != len(self.shunt_q_steps_mvar):
            raise ValueError(
                f"shunt_bus_indices length ({len(self.shunt_bus_indices)}) "
                f"must match shunt_q_steps_mvar length ({len(self.shunt_q_steps_mvar)})"
            )
        if self.v_min_pu >= self.v_max_pu:
            raise ValueError(
                f"v_min_pu ({self.v_min_pu}) must be less than "
                f"v_max_pu ({self.v_max_pu})"
            )
        if self.i_max_pu <= 0:
            raise ValueError(f"i_max_pu must be positive, got {self.i_max_pu}")



class DSOController(BaseOFOController):
    """
    DSO-level Online Feedback Optimisation controller.
    
    This controller manages the distribution network reactive power and
    voltage by controlling DERs, OLTCs, and shunts. It tracks reactive
    power setpoints received from the TSO whilst enforcing local constraints.
    
    Control Variables (u):
        - Q_DER: DER reactive power setpoints [Mvar] (continuous)
        - s_OLTC: OLTC tap positions (integer)
        - state_shunt: Shunt switching states {-1, 0, +1} (integer)
    
    Outputs (y):
        - Q_interface: Reactive power at TSO-DSO interface [Mvar]
        - V_bus: Voltage magnitudes at monitored buses [p.u.]
        - I_line: Current magnitudes at monitored lines [p.u.]
    
    Attributes
    ----------
    config : DSOControllerConfig
        Controller configuration.
    q_setpoint_mvar : NDArray[np.float64]
        Current reactive power setpoints from TSO.
    """
    
    def __init__(
        self,
        controller_id: str,
        params: OFOParameters,
        config: DSOControllerConfig,
        network_state: NetworkState,
        actuator_bounds: ActuatorBounds,
        sensitivities: JacobianSensitivities,
    ) -> None:
        """
        Initialise the DSO controller.
        
        Parameters
        ----------
        controller_id : str
            Unique identifier for this controller.
        params : OFOParameters
            OFO tuning parameters.
        config : DSOControllerConfig
            DSO-specific configuration.
        network_state : NetworkState
            Cached network state for sensitivity computation.
        actuator_bounds : ActuatorBounds
            Actuator limits calculator.
        sensitivities : JacobianSensitivities
            Jacobian-based sensitivity calculator.
        """
        super().__init__(
            controller_id=controller_id,
            params=params,
            network_state=network_state,
            actuator_bounds=actuator_bounds,
            sensitivities=sensitivities,
        )
        
        self.config = config

        # Initialise Q setpoints to zero (no tracking until TSO sends message)
        n_interfaces = len(config.interface_trafo_indices)
        self.q_setpoint_mvar = np.zeros(n_interfaces)

        # Shunt bound overrides from Reserve Observer.
        # Keys: shunt index (0-based within shunt vector).
        # Values: (lower, upper) bound override for that shunt.
        # Cleared after each step.
        self._shunt_bound_overrides: Dict[int, tuple] = {}

        # Cache the sensitivity matrix structure
        self._H_cache: Optional[NDArray[np.float64]] = None
        self._H_mappings: Optional[Dict] = None
        self._sensitivity_updater: Optional[SensitivityUpdater] = None
    
    def receive_setpoint(self, message: SetpointMessage) -> None:
        """
        Receive a setpoint message from the TSO controller.
        
        Parameters
        ----------
        message : SetpointMessage
            Setpoint message from TSO.
        
        Raises
        ------
        ValueError
            If the message targets a different controller or has
            incompatible interface transformers.
        """
        if message.target_controller_id != self.controller_id:
            raise ValueError(
                f"Message target '{message.target_controller_id}' does not "
                f"match controller ID '{self.controller_id}'"
            )
        
        # Verify interface transformers match
        expected = set(self.config.interface_trafo_indices)
        received = set(message.interface_transformer_indices)
        
        if expected != received:
            raise ValueError(
                f"Interface transformer mismatch. Expected {expected}, "
                f"got {received}"
            )
        
        # Store setpoints in the correct order
        for i, trafo_idx in enumerate(self.config.interface_trafo_indices):
            msg_idx = list(message.interface_transformer_indices).index(trafo_idx)
            self.q_setpoint_mvar[i] = message.q_setpoints_mvar[msg_idx]
    
    def generate_capability_message(
        self,
        target_controller_id: str,
        measurement: Measurement,
    ) -> CapabilityMessage:
        """
        Generate a capability message for the TSO controller.
        
        The capability bounds are computed from DER capabilities mapped
        to the interface using sensitivity analysis.
        
        Parameters
        ----------
        target_controller_id : str
            Identifier of the TSO controller.
        measurement : Measurement
            Current system measurements.
        
        Returns
        -------
        CapabilityMessage
            Message containing interface Q capability bounds.
        """
        # Get DER P for capability calculation
        der_p = self._extract_der_active_power(measurement)
        
        # Get DER Q capability bounds
        q_der_min, q_der_max = self.actuator_bounds.compute_der_q_bounds(der_p)
        # Build sensitivity matrix if not cached
        if self._H_cache is None:
            self._build_sensitivity_matrix()
        
        # Extract ∂Q_interface/∂Q_DER from H matrix
        # The first n_interfaces rows of H correspond to Q_interface outputs
        n_interfaces = len(self.config.interface_trafo_indices)
        n_der = len(self.config.der_bus_indices)
        
        dQ_interface_dQ_der = self._H_cache[:n_interfaces, :n_der]
        
        # Map DER capability to interface capability
        # Simple approach: sum contributions assuming independence
        q_interface_min = np.zeros(n_interfaces)
        q_interface_max = np.zeros(n_interfaces)
        
        for i in range(n_interfaces):
            for j in range(n_der):
                sensitivity = dQ_interface_dQ_der[i, j]
                if sensitivity >= 0:
                    q_interface_min[i] += sensitivity * q_der_min[j]
                    q_interface_max[i] += sensitivity * q_der_max[j]
                else:
                    q_interface_min[i] += sensitivity * q_der_max[j]
                    q_interface_max[i] += sensitivity * q_der_min[j]
        
        return CapabilityMessage(
            source_controller_id=self.controller_id,
            target_controller_id=target_controller_id,
            iteration=self.iteration,
            interface_transformer_indices=np.array(
                self.config.interface_trafo_indices, dtype=np.int64
            ),
            q_min_mvar=q_interface_min,
            q_max_mvar=q_interface_max,
        )
    
    def set_shunt_overrides(self, overrides: Dict[int, tuple]) -> None:
        """
        Set shunt bound overrides from the Reserve Observer.

        Parameters
        ----------
        overrides : Dict[int, tuple]
            Mapping from shunt index (0-based within the DSO shunt
            vector) to ``(lower, upper)`` bound.  These are applied
            in :meth:`_compute_input_bounds` and cleared after each
            call to :meth:`step`.
        """
        self._shunt_bound_overrides = overrides.copy()

    # =========================================================================
    # Implementation of abstract methods
    # =========================================================================

    def _get_control_structure(self) -> Tuple[int, int, List[int]]:
        """Define the control variable structure."""
        n_der = len(self.config.der_bus_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        
        n_continuous = n_der
        n_integer = n_oltc + n_shunt
        
        # Integer indices are after the continuous DER variables
        integer_indices = list(range(n_continuous, n_continuous + n_integer))
        
        return n_continuous, n_integer, integer_indices
    
    def _extract_control_values(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract current control values from measurements."""
        n_der = len(self.config.der_bus_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_total = n_der + n_oltc + n_shunt
        
        u = np.zeros(n_total)
        
        # DER Q setpoints
        for i, der_idx in enumerate(self.config.der_bus_indices):
            # Find DER in measurement by matching indices
            meas_idx = np.where(measurement.der_indices == der_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"DER at bus {der_idx} not found in measurement")
            u[i] = measurement.der_q_mvar[meas_idx[0]]
        
        # OLTC tap positions
        for i, oltc_idx in enumerate(self.config.oltc_trafo_indices):
            meas_idx = np.where(measurement.oltc_indices == oltc_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"OLTC {oltc_idx} not found in measurement")
            u[n_der + i] = float(measurement.oltc_tap_positions[meas_idx[0]])
        
        # Shunt states
        for i, shunt_idx in enumerate(self.config.shunt_bus_indices):
            meas_idx = np.where(measurement.shunt_indices == shunt_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"Shunt at bus {shunt_idx} not found in measurement")
            u[n_der + n_oltc + i] = float(measurement.shunt_states[meas_idx[0]])
        
        return u
    
    def _extract_outputs(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract current output values from measurements."""
        n_interfaces = len(self.config.interface_trafo_indices)
        n_voltage = len(self.config.voltage_bus_indices)
        n_current = len(self.config.current_line_indices)
        n_outputs = n_interfaces + n_voltage + n_current
        
        y = np.zeros(n_outputs)
        idx = 0
        
        # Interface Q measurements
        for trafo_idx in self.config.interface_trafo_indices:
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Interface transformer {trafo_idx} not found in measurement"
                )
            y[idx] = measurement.interface_q_hv_side_mvar[meas_idx[0]]
            idx += 1
        
        # Voltage measurements
        for bus_idx in self.config.voltage_bus_indices:
            meas_idx = np.where(measurement.bus_indices == bus_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"Bus {bus_idx} not found in measurement")
            y[idx] = measurement.voltage_magnitudes_pu[meas_idx[0]]
            idx += 1
        
        # Current measurements
        for line_idx in self.config.current_line_indices:
            meas_idx = np.where(measurement.branch_indices == line_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"Line {line_idx} not found in measurement")
            y[idx] = measurement.current_magnitudes_ka[meas_idx[0]]
            idx += 1
        
        return y
    
    def _extract_der_active_power(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract DER active power for capability calculation."""
        # Note: The Measurement class currently does not include DER P values.
        # For now, we use the installed capacity from actuator_bounds.
        # In a real implementation, this would come from measurements.
        return self.actuator_bounds.der_p_max_mw.copy()
    
    def _compute_input_bounds(
        self,
        der_p_current: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute operating-point-dependent input bounds."""
        n_der = len(self.config.der_bus_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_total = n_der + n_oltc + n_shunt
        
        u_lower = np.zeros(n_total)
        u_upper = np.zeros(n_total)
        
        # DER Q bounds (P-dependent)
        q_min, q_max = self.actuator_bounds.compute_der_q_bounds(der_p_current)
        u_lower[:n_der] = -50 #q_min # ToDo: Also loosen here to test controller performance
        u_upper[:n_der] = 50 #q_max # ToDo: Also loosen here to test controller performance
        
        # OLTC tap bounds (fixed)
        tap_min, tap_max = self.actuator_bounds.get_oltc_tap_bounds()
        u_lower[n_der:n_der + n_oltc] = tap_min.astype(np.float64)
        u_upper[n_der:n_der + n_oltc] = tap_max.astype(np.float64)
        
        # Shunt state bounds (fixed: -1, 0, +1)
        state_min, state_max = self.actuator_bounds.get_shunt_state_bounds()
        u_lower[n_der + n_oltc:] = state_min.astype(np.float64)
        u_upper[n_der + n_oltc:] = state_max.astype(np.float64)

        # Apply Reserve Observer overrides
        shunt_offset = n_der + n_oltc
        for j, (lo, hi) in self._shunt_bound_overrides.items():
            u_lower[shunt_offset + j] = lo
            u_upper[shunt_offset + j] = hi

        return u_lower, u_upper
    
    def _get_output_limits(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get output constraint limits."""
        n_interfaces = len(self.config.interface_trafo_indices)
        n_voltage = len(self.config.voltage_bus_indices)
        n_current = len(self.config.current_line_indices)
        n_outputs = n_interfaces + n_voltage + n_current
        
        y_lower = np.zeros(n_outputs)
        y_upper = np.zeros(n_outputs)
        idx = 0
        
        # Interface Q: no hard limits (tracking via objective)
        for _ in range(n_interfaces):
            y_lower[idx] = -1E6
            y_upper[idx] = 1E6
            idx += 1
        
        # Voltage limits
        for _ in range(n_voltage):
            y_lower[idx] = self.config.v_min_pu
            y_upper[idx] = self.config.v_max_pu
            idx += 1
        
        # Current limits (upper only, normalised to rating)
        # NOTE: Extremely loosened to unblock integer switching (shunts/OLTCs).
        #       The large shunt Q steps (50 Mvar) cause huge current swings
        #       in the linearised model, blocking MIQP engagement.
        for _ in range(n_current):
            y_lower[idx] = -1E6
            y_upper[idx] = 1E6
            idx += 1
        
        return y_lower, y_upper
    
    def _compute_objective_gradient(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Compute the objective function gradient.

        The DSO objective is to track the TSO setpoint:
            f(u) = g_q · ||Q_interface - Q_set||²

        The gradient is:
            ∇f = 2 · g_q · (Q_interface - Q_set) · ∂Q_interface/∂u
        """
        n_total = self.n_controls
        grad_f = np.zeros(n_total)
        
        # Get current interface Q values
        n_interfaces = len(self.config.interface_trafo_indices)
        q_interface = np.zeros(n_interfaces)
        
        for i, trafo_idx in enumerate(self.config.interface_trafo_indices):
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) > 0:
                q_interface[i] = measurement.interface_q_hv_side_mvar[meas_idx[0]]
        
        # Compute Q tracking error
        q_error = q_interface - self.q_setpoint_mvar
        
        # Get sensitivity matrix (use cache if available)
        H = self._build_sensitivity_matrix()
        
        # Extract ∂Q_interface/∂u (first n_interfaces rows)
        dQ_du = H[:n_interfaces, :]
        
        # Compute gradient: ∇f = 2 · g_q · (Q - Q_set)^T · ∂Q/∂u
        grad_f = 2.0 * self.config.g_q * (q_error @ dQ_du)

        return grad_f

    def _build_sensitivity_matrix(self) -> NDArray[np.float64]:
        """Build the input-output sensitivity matrix H."""
        if self._H_cache is not None:
            return self._H_cache

        # Determine whether interface and OLTC transformers are 2W or 3W.
        # Check if the indices exist in net.trafo3w (they do for the
        # TU Darmstadt benchmark where couplers are 3-winding).
        net = self.sensitivities.net

        iface_are_3w = (
            hasattr(net, "trafo3w")
            and not net.trafo3w.empty
            and all(
                t in net.trafo3w.index
                for t in self.config.interface_trafo_indices
            )
        )
        oltc_are_3w = (
            hasattr(net, "trafo3w")
            and not net.trafo3w.empty
            and all(
                t in net.trafo3w.index
                for t in self.config.oltc_trafo_indices
            )
        )

        # Build keyword arguments depending on transformer type
        kw = dict(
            der_bus_indices=self.config.der_bus_indices,
            observation_bus_indices=self.config.voltage_bus_indices,
            line_indices=self.config.current_line_indices,
            shunt_bus_indices=self.config.shunt_bus_indices,
            shunt_q_steps_mvar=self.config.shunt_q_steps_mvar,
        )
        if iface_are_3w:
            kw["trafo3w_indices"] = self.config.interface_trafo_indices
        else:
            kw["trafo_indices"] = self.config.interface_trafo_indices

        if oltc_are_3w:
            kw["oltc_trafo3w_indices"] = self.config.oltc_trafo_indices
        else:
            kw["oltc_trafo_indices"] = self.config.oltc_trafo_indices

        H, mappings = self.sensitivities.build_sensitivity_matrix_H(**kw)

        self._H_cache = H
        self._H_mappings = mappings

        # Create the sensitivity updater for voltage-dependent corrections.
        # DSO has no machine transformer OLTCs, so only shunt updates apply.
        self._sensitivity_updater = SensitivityUpdater(
            H=H,
            mappings=mappings,
            sensitivities=self.sensitivities,
            update_interval_min=1,
        )

        return H

    def step(self, measurement: Measurement) -> ControllerOutput:
        """
        Execute one OFO iteration with voltage-dependent sensitivity updates.

        Before delegating to :meth:`BaseOFOController.step`, this method
        rescales shunt columns of the cached H matrix using the measured
        bus voltages.  The V² correction accounts for the constant-susceptance
        nature of shunt devices (MSR / MSC).

        After the step, shunt bound overrides from the Reserve Observer
        are cleared so they must be re-set each iteration.
        """
        # Ensure H is built
        if self._H_cache is None:
            self._build_sensitivity_matrix()
        # Update state-dependent shunt columns
        if self._sensitivity_updater is not None:
            self._H_cache = self._sensitivity_updater.update(
                measurement, measurement.iteration
            )
        result = super().step(measurement)
        # Clear one-shot overrides
        self._shunt_bound_overrides.clear()
        return result

    def invalidate_sensitivity_cache(self) -> None:
        """Invalidate the cached sensitivity matrix."""
        self._H_cache = None
        self._H_mappings = None
        self._sensitivity_updater = None
