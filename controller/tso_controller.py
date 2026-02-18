"""
TSO Controller Module
=====================

This module defines the TSO-level MIQP controller for transmission system
voltage and reactive power optimisation.

The TSO controller operates as the upper layer in the cascaded OFO hierarchy.
It dispatches:
    (i)   Generator reactive power via machine-transformer OLTC positions
    (ii)  Discrete shunt compensation switching (MSC/MSR)
    (iii) Reactive power from transmission-connected DER
    (iv)  Reactive power setpoints to subordinate DSO controllers at PCCs

Subordinate DSO controllers report operating-point-dependent capability
bounds which are used as input constraints for the PCC setpoints.

Control Variables (u):
    - Q_DER:     Transmission-connected DER reactive power [Mvar] (continuous)
    - Q_PCC_set: PCC reactive power setpoints to DSOs [Mvar] (continuous)
    - s_OLTC:    Machine-transformer OLTC tap positions (integer)
    - s_shunt:   Shunt compensation states {-1, 0, +1} (integer)

Outputs (y):
    - V_bus:  Voltage magnitudes at monitored EHV buses [p.u.]
    - Q_PCC:  Reactive power at TSO-DSO interfaces [Mvar]
    - I_line: Current magnitudes at monitored EHV lines [p.u.]

Objective:
    The TSO objective minimises a composite criterion consisting of voltage
    schedule tracking at EHV buses and regularisation of DER and PCC usage.
    Voltage setpoints are encoded as soft output constraints with a quadratic
    penalty on the per-bus voltage deviation, following the approach in
    Schwenke et al. (CIRED 2025).

References
----------
[1] Schwenke, Ruppert, Hanson — CIGRE 2026 Synopsis (ABSTRACT_BASE)
[2] Schwenke, Hanson — PSCC 2026, Section III
[3] Schwenke, Korff, Hanson — CIRED 2025

Author: Manuel Schwenke
Date: 2025-02-06
"""

from dataclasses import dataclass, field
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
class TSOControllerConfig:
    """
    Configuration for the TSO controller.

    Attributes
    ----------
    der_bus_indices : List[int]
        Pandapower bus indices where transmission-connected DERs are located.
    pcc_trafo_indices : List[int]
        Pandapower transformer indices for TSO-DSO interface (PCC) transformers.
        These are the transformers through which DSO areas are connected.
    pcc_dso_controller_ids : List[str]
        Controller identifiers of subordinate DSO controllers, ordered
        consistently with pcc_trafo_indices.
    oltc_trafo_indices : List[int]
        Pandapower transformer indices for machine-transformer OLTCs at EHV level.
    shunt_bus_indices : List[int]
        Pandapower bus indices where switchable shunts (MSC/MSR) are connected.
    shunt_q_steps_mvar : List[float]
        Reactive power step per state change for each shunt [Mvar].
    voltage_bus_indices : List[int]
        Pandapower bus indices for EHV voltage monitoring.
    current_line_indices : List[int]
        Pandapower line indices for EHV thermal monitoring.
    v_min_pu : float
        Minimum voltage limit at EHV buses in per-unit.
    v_max_pu : float
        Maximum voltage limit at EHV buses in per-unit.
    i_max_pu : float
        Maximum current limit as fraction of line rating.
    v_setpoints_pu : Optional[NDArray[np.float64]]
        Voltage setpoints at monitored EHV buses [p.u.]. If provided, the
        objective includes a voltage-schedule tracking term. Must have the
        same length as voltage_bus_indices.
    g_v : float
        Weight for voltage-schedule tracking in the objective function.
        Scales the gradient ``2 · g_v · (V - V_set)^T · ∂V/∂u``.
        Higher values make the controller track the voltage setpoints
        more aggressively.  Must be balanced against the change penalty
        ``g_w``: the effective per-iteration step scales as
        ``α · g_v / g_w``.  Default 1.0 (unweighted tracking).
    """
    der_bus_indices: List[int]
    pcc_trafo_indices: List[int]
    pcc_dso_controller_ids: List[str]
    oltc_trafo_indices: List[int]
    shunt_bus_indices: List[int]
    shunt_q_steps_mvar: List[float]
    voltage_bus_indices: List[int]
    current_line_indices: List[int]
    v_min_pu: float = 0.90
    v_max_pu: float = 1.10
    i_max_pu: float = 1.0
    v_setpoints_pu: Optional[NDArray[np.float64]] = None
    g_v: float = 1.0
    gen_indices: List[int] = field(default_factory=list)
    gen_bus_indices: List[int] = field(default_factory=list)
    gen_vm_min_pu: float = 0.95
    gen_vm_max_pu: float = 1.10

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if len(self.pcc_trafo_indices) != len(self.pcc_dso_controller_ids):
            raise ValueError(
                f"pcc_trafo_indices length ({len(self.pcc_trafo_indices)}) "
                f"must match pcc_dso_controller_ids length "
                f"({len(self.pcc_dso_controller_ids)})"
            )
        if len(self.shunt_bus_indices) != len(self.shunt_q_steps_mvar):
            raise ValueError(
                f"shunt_bus_indices length ({len(self.shunt_bus_indices)}) "
                f"must match shunt_q_steps_mvar length "
                f"({len(self.shunt_q_steps_mvar)})"
            )
        if self.v_min_pu >= self.v_max_pu:
            raise ValueError(
                f"v_min_pu ({self.v_min_pu}) must be less than "
                f"v_max_pu ({self.v_max_pu})"
            )
        if self.i_max_pu <= 0:
            raise ValueError(
                f"i_max_pu must be positive, got {self.i_max_pu}"
            )
        if self.v_setpoints_pu is not None:
            if len(self.v_setpoints_pu) != len(self.voltage_bus_indices):
                raise ValueError(
                    f"v_setpoints_pu length ({len(self.v_setpoints_pu)}) "
                    f"must match voltage_bus_indices length "
                    f"({len(self.voltage_bus_indices)})"
                )
        if len(self.gen_indices) != len(self.gen_bus_indices):
            raise ValueError(
                f"gen_indices length ({len(self.gen_indices)}) must match "
                f"gen_bus_indices length ({len(self.gen_bus_indices)})"
            )
        if self.gen_vm_min_pu >= self.gen_vm_max_pu:
            raise ValueError(
                f"gen_vm_min_pu ({self.gen_vm_min_pu}) must be less than "
                f"gen_vm_max_pu ({self.gen_vm_max_pu})"
            )


class TSOController(BaseOFOController):
    """
    TSO-level Online Feedback Optimisation controller.

    This controller operates as the upper-layer supervisor in the cascaded
    OFO hierarchy.  It optimises EHV bus voltages and reactive power flows
    by actuating transmission-connected DER, machine-transformer OLTCs,
    switchable shunts, and by issuing reactive power setpoints to
    subordinate DSO controllers via the TSO-DSO interfaces (PCCs).

    Control variable ordering within u:
        [ Q_DER (continuous) | Q_PCC_set (continuous) | s_OLTC (integer) | s_shunt (integer) ]

    Output variable ordering within y:
        [ V_bus (p.u.) | Q_PCC (Mvar) | I_line (p.u.) ]

    Attributes
    ----------
    config : TSOControllerConfig
        Controller configuration.
    pcc_capability_min_mvar : NDArray[np.float64]
        Current lower PCC capability bounds from DSO messages [Mvar].
    pcc_capability_max_mvar : NDArray[np.float64]
        Current upper PCC capability bounds from DSO messages [Mvar].
    """

    def __init__(
        self,
        controller_id: str,
        params: OFOParameters,
        config: TSOControllerConfig,
        network_state: NetworkState,
        actuator_bounds: ActuatorBounds,
        sensitivities: JacobianSensitivities,
    ) -> None:
        """
        Initialise the TSO controller.

        Parameters
        ----------
        controller_id : str
            Unique identifier for this controller.
        params : OFOParameters
            OFO tuning parameters.
        config : TSOControllerConfig
            TSO-specific configuration.
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

        # Initialise PCC capability bounds to large symmetric range
        # until DSO controllers report actual capabilities
        n_pcc = len(config.pcc_trafo_indices)
        self.pcc_capability_min_mvar = np.full(n_pcc, -1E6)
        self.pcc_capability_max_mvar = np.full(n_pcc, +1E6)

        # Cache for the sensitivity matrix
        self._H_cache: Optional[NDArray[np.float64]] = None
        self._H_mappings: Optional[Dict] = None
        self._sensitivity_updater: Optional[SensitivityUpdater] = None

    # =========================================================================
    # Public interface for cascaded hierarchy communication
    # =========================================================================

    def receive_capability(self, message: CapabilityMessage) -> None:
        """
        Receive a capability message from a subordinate DSO controller.

        The reported capability bounds are stored and used as input
        constraints for the corresponding PCC setpoint variables.

        Parameters
        ----------
        message : CapabilityMessage
            Capability message from a DSO controller.

        Raises
        ------
        ValueError
            If the message targets a different controller or references
            unknown interface transformers.
        """
        if message.target_controller_id != self.controller_id:
            raise ValueError(
                f"Message target '{message.target_controller_id}' does not "
                f"match controller ID '{self.controller_id}'"
            )

        # Map reported capabilities to local PCC ordering
        for i, trafo_idx in enumerate(self.config.pcc_trafo_indices):
            msg_idx_candidates = np.where(
                message.interface_transformer_indices == trafo_idx
            )[0]
            if len(msg_idx_candidates) == 0:
                # This PCC transformer is not covered by the message; skip
                continue
            msg_idx = msg_idx_candidates[0]
            self.pcc_capability_min_mvar[i] = message.q_min_mvar[msg_idx]
            self.pcc_capability_max_mvar[i] = message.q_max_mvar[msg_idx]

    def generate_setpoint_messages(
        self,
    ) -> List[SetpointMessage]:
        """
        Generate setpoint messages for all subordinate DSO controllers.

        The PCC setpoint values are taken from the current control vector.
        One SetpointMessage is generated per DSO controller.

        Returns
        -------
        messages : List[SetpointMessage]
            Setpoint messages, one per subordinate DSO controller.

        Raises
        ------
        RuntimeError
            If the controller is not yet initialised.
        """
        if self._u_current is None:
            raise RuntimeError(
                "Controller not initialised. Call initialise() first."
            )

        n_der = len(self.config.der_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)

        # PCC setpoint values sit after the DER entries in u
        q_pcc_setpoints = self._u_current[n_der:n_der + n_pcc]

        messages: List[SetpointMessage] = []
        for i, dso_id in enumerate(self.config.pcc_dso_controller_ids):
            msg = SetpointMessage(
                source_controller_id=self.controller_id,
                target_controller_id=dso_id,
                iteration=self.iteration,
                interface_transformer_indices=np.array(
                    [self.config.pcc_trafo_indices[i]], dtype=np.int64
                ),
                q_setpoints_mvar=np.array(
                    [q_pcc_setpoints[i]], dtype=np.float64
                ),
            )
            messages.append(msg)
        return messages

    def update_voltage_setpoints(
        self,
        v_setpoints_pu: NDArray[np.float64],
    ) -> None:
        """
        Update the voltage setpoints (voltage schedule).

        Parameters
        ----------
        v_setpoints_pu : NDArray[np.float64]
            New voltage setpoints in per-unit, one per monitored bus.

        Raises
        ------
        ValueError
            If the length does not match the number of monitored buses.
        """
        n_v = len(self.config.voltage_bus_indices)
        if len(v_setpoints_pu) != n_v:
            raise ValueError(
                f"v_setpoints_pu length ({len(v_setpoints_pu)}) must match "
                f"voltage_bus_indices length ({n_v})"
            )
        self.config.v_setpoints_pu = v_setpoints_pu.copy()

    # =========================================================================
    # Implementation of abstract methods
    # =========================================================================

    def _get_control_structure(self) -> Tuple[int, int, List[int]]:
        """
        Define the control variable structure.

        Ordering: [ Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]
        """
        n_der = len(self.config.der_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        n_continuous = n_der + n_pcc + n_gen
        n_integer = n_oltc + n_shunt

        # Integer indices start after all continuous variables
        integer_indices = list(range(n_continuous, n_continuous + n_integer))
        return n_continuous, n_integer, integer_indices

    def _extract_control_values(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract current control values from measurements."""
        n_der = len(self.config.der_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        n_total = n_der + n_pcc + n_gen + n_oltc + n_shunt
        u = np.zeros(n_total, dtype=np.float64)
        idx = 0

        # --- Transmission-connected DER Q setpoints ---
        for der_bus in self.config.der_bus_indices:
            meas_idx = np.where(measurement.der_indices == der_bus)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"DER at bus {der_bus} not found in measurement"
                )
            u[idx] = float(measurement.der_q_mvar[meas_idx[0]])
            idx += 1

        # --- PCC reactive power as initial setpoints ---
        for trafo_idx in self.config.pcc_trafo_indices:
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"PCC transformer {trafo_idx} not found in measurement"
                )
            u[idx] = float(
                measurement.interface_q_hv_side_mvar[meas_idx[0]]
            )
            idx += 1

        # --- Generator AVR voltage setpoints ---
        for g_idx in self.config.gen_indices:
            meas_idx = np.where(measurement.gen_indices == g_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Generator {g_idx} not found in measurement.gen_indices"
                )
            u[idx] = float(measurement.gen_vm_pu[meas_idx[0]])
            idx += 1

        # --- Machine-transformer OLTC tap positions ---
        for oltc_idx in self.config.oltc_trafo_indices:
            meas_idx = np.where(measurement.oltc_indices == oltc_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"OLTC {oltc_idx} not found in measurement"
                )
            u[idx] = float(measurement.oltc_tap_positions[meas_idx[0]])
            idx += 1

        # --- Shunt states ---
        for shunt_bus in self.config.shunt_bus_indices:
            meas_idx = np.where(measurement.shunt_indices == shunt_bus)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Shunt at bus {shunt_bus} not found in measurement"
                )
            u[idx] = float(measurement.shunt_states[meas_idx[0]])
            idx += 1

        return u

    def _extract_outputs(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract current output values from measurements.

        Output ordering: [ V_bus | Q_PCC | I_line ]
        """
        n_v = len(self.config.voltage_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_i = len(self.config.current_line_indices)
        n_outputs = n_v + n_pcc + n_i

        y = np.zeros(n_outputs)
        idx = 0

        # Voltage measurements
        for bus_idx in self.config.voltage_bus_indices:
            meas_idx = np.where(measurement.bus_indices == bus_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Bus {bus_idx} not found in measurement"
                )
            y[idx] = measurement.voltage_magnitudes_pu[meas_idx[0]]
            idx += 1

        # PCC reactive power measurements
        for trafo_idx in self.config.pcc_trafo_indices:
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"PCC transformer {trafo_idx} not found in measurement"
                )
            y[idx] = measurement.interface_q_hv_side_mvar[meas_idx[0]]
            idx += 1

        # Current measurements
        for line_idx in self.config.current_line_indices:
            meas_idx = np.where(measurement.branch_indices == line_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Line {line_idx} not found in measurement"
                )
            y[idx] = measurement.current_magnitudes_ka[meas_idx[0]]
            idx += 1

        return y

    def _extract_der_active_power(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract DER active power for capability calculation.

        Note: The Measurement class currently does not carry DER P values.
        The installed capacity from ActuatorBounds is used as a proxy.
        """
        return self.actuator_bounds.der_p_max_mw.copy()

    def _compute_input_bounds(
        self,
        der_p_current: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute operating-point-dependent input bounds.

        For PCC setpoints, the bounds are updated from capability messages.
        AVR setpoints are bounded by fixed min/max values in the config.
        """
        n_der = len(self.config.der_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        n_total = n_der + n_pcc + n_gen + n_oltc + n_shunt

        u_lower = np.zeros(n_total, dtype=np.float64)
        u_upper = np.zeros(n_total, dtype=np.float64)

        # --- DER Q bounds (P-dependent) ---
        q_min, q_max = self.actuator_bounds.compute_der_q_bounds(der_p_current)
        u_lower[:n_der] = q_min
        u_upper[:n_der] = q_max

        # --- PCC setpoint bounds (from DSO capability messages) ---
        u_lower[n_der:n_der + n_pcc] = self.pcc_capability_min_mvar
        u_upper[n_der:n_der + n_pcc] = self.pcc_capability_max_mvar

        # --- AVR setpoint bounds (fixed) ---
        avr_start = n_der + n_pcc
        avr_end = avr_start + n_gen
        u_lower[avr_start:avr_end] = self.config.gen_vm_min_pu
        u_upper[avr_start:avr_end] = self.config.gen_vm_max_pu

        # --- OLTC tap bounds (fixed mechanical limits) ---
        tap_min, tap_max = self.actuator_bounds.get_oltc_tap_bounds()
        tap_start = avr_end
        tap_end = tap_start + n_oltc
        u_lower[tap_start:tap_end] = tap_min.astype(np.float64)
        u_upper[tap_start:tap_end] = tap_max.astype(np.float64)

        # --- Shunt state bounds (fixed: -1, 0, +1) ---
        state_min, state_max = self.actuator_bounds.get_shunt_state_bounds()
        shunt_start = tap_end
        u_lower[shunt_start:] = state_min.astype(np.float64)
        u_upper[shunt_start:] = state_max.astype(np.float64)

        return u_lower, u_upper


    def _get_output_limits(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get output constraint limits.

        Voltage outputs are constrained to the permissible band
        [v_min_pu, v_max_pu].  Tracking toward voltage setpoints
        (if configured) is handled by the quadratic objective in
        ``_compute_objective_gradient()``, not by tight constraints.

        PCC Q outputs have no hard limits (±∞); tracking is handled
        by the DSO layer through the setpoint messages.
        """
        n_v = len(self.config.voltage_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_i = len(self.config.current_line_indices)
        n_outputs = n_v + n_pcc + n_i

        y_lower = np.zeros(n_outputs)
        y_upper = np.zeros(n_outputs)
        idx = 0

        # --- Voltage limits (band constraints) ---
        # Voltage tracking toward setpoints is handled by the quadratic
        # objective term in _compute_objective_gradient(), not by tight
        # output constraints.  The output constraints enforce the
        # permissible voltage band [v_min_pu, v_max_pu].
        for j in range(n_v):
            y_lower[idx] = self.config.v_min_pu
            y_upper[idx] = self.config.v_max_pu
            idx += 1

        # --- PCC Q: no hard limits (tracking is objective-based) ---
        for _ in range(n_pcc):
            y_lower[idx] = -1E6
            y_upper[idx] = 1E6
            idx += 1

        # --- Current limits (upper only) ---
        # NOTE: Extremely loosened to unblock integer switching (shunts/OLTCs).
        for _ in range(n_i):
            y_lower[idx] = -1E6
            y_upper[idx] = 1E6
            idx += 1

        return y_lower, y_upper

    def _compute_objective_gradient(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Compute the objective function gradient ∇f.

        The TSO objective combines:
            1. DER usage regularisation:  2 · g_u · Q_DER^k
            2. Voltage-schedule tracking: 2 · g_v · (V^k - V_set)
               projected onto the control space via sensitivities
            3. PCC setpoint terms are zero (tracking handled by DSOs)

        The gradient is assembled in the augmented form
            ∇f = [ ∇f_u ; ∇f_y ]^T · H̃
        as described in PSCC 2026, Eq. (30)-(31).  However, since the
        base class build_miqp_problem already multiplies grad_f with H̃
        inside the QP objective term  ∇f^T · H̃ · w ,  we provide the
        *un-augmented* gradient in control-variable space here, i.e.
        the product  ∇f_combined = ∇f_u + (∂y/∂u)^T · ∇f_y .

        This is consistent with the DSO controller implementation and
        the interface of build_miqp_problem.
        """
        n_total = self.n_controls
        grad_f = np.zeros(n_total)

        n_der = len(self.config.der_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_v = len(self.config.voltage_bus_indices)

        # --- Component 1: DER usage regularisation ---
        # Handled implicitly by g_u in build_miqp_problem; no explicit
        # gradient needed here (build_miqp_problem adds 2·α·g_u·u_current).

        # --- Component 2: Voltage-schedule tracking ---
        if self.config.v_setpoints_pu is not None:
            # Get current voltages
            v_current = np.zeros(n_v)
            for j, bus_idx in enumerate(self.config.voltage_bus_indices):
                meas_idx = np.where(measurement.bus_indices == bus_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"Bus {bus_idx} not found in measurement"
                    )
                v_current[j] = measurement.voltage_magnitudes_pu[meas_idx[0]]

            v_error = v_current - self.config.v_setpoints_pu

            # Map voltage error to control space via sensitivities
            H = self._build_sensitivity_matrix()
            # Voltage outputs are the first n_v rows of H
            dV_du = H[:n_v, :]

            # ∇f contribution: 2 · g_v · (V - V_set)^T · ∂V/∂u
            grad_f += 2.0 * self.config.g_v * (v_error @ dV_du)

        return grad_f

    def _build_sensitivity_matrix(self) -> NDArray[np.float64]:
        """
        Build the input-output sensitivity matrix H.

        The matrix maps control variable changes to output changes:
            Δy ≈ H · Δu

        Columns correspond to:  [ Q_DER | Q_PCC_set | s_OLTC | s_shunt ]
        Rows correspond to:     [ V_bus | Q_PCC     | I_line ]

        For PCC setpoint columns, the TSO does not have direct physical
        sensitivities (the PCC Q is realised by the DSO).  These columns
        are filled with identity-like entries: a unit change in Q_PCC_set
        is expected to produce a unit change in the measured Q_PCC output,
        i.e.  ∂Q_PCC_j / ∂Q_PCC_set_j = 1.  All other entries in the
        PCC setpoint columns are zero.  This is a first-order
        approximation assuming perfect DSO tracking.

        Returns
        -------
        H : NDArray[np.float64]
            Sensitivity matrix of shape (n_outputs, n_controls).
        """
        if self._H_cache is not None:
            return self._H_cache

        n_der = len(self.config.der_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_v = len(self.config.voltage_bus_indices)
        n_i = len(self.config.current_line_indices)

        n_controls = n_der + n_pcc + n_gen + n_oltc + n_shunt
        n_outputs = n_v + n_pcc + n_i

        H = np.zeros((n_outputs, n_controls), dtype=np.float64)

        # --- Physical sensitivities for DER / OLTC / shunt columns ---
        # Build sub-matrix from JacobianSensitivities for the physical
        # actuators: columns = [DER, OLTC, shunt], rows = [V, Q_PCC, I]
        #
        # Determine whether PCC transformers are 2W or 3W in the network.
        # After splitting, the TN network may not contain the PCC trafos
        # at all (they are replaced by boundary sgens).  In that case we
        # build H without Q_trafo rows and add PCC setpoint identity rows
        # manually below.
        net = self.sensitivities.net

        pcc_in_trafo = all(
            t in net.trafo.index for t in self.config.pcc_trafo_indices
        ) if not net.trafo.empty else False
        pcc_in_trafo3w = (
            hasattr(net, "trafo3w")
            and not net.trafo3w.empty
            and all(
                t in net.trafo3w.index
                for t in self.config.pcc_trafo_indices
            )
        )

        kw = dict(
            der_bus_indices=self.config.der_bus_indices,
            observation_bus_indices=self.config.voltage_bus_indices,
            line_indices=self.config.current_line_indices,
            oltc_trafo_indices=self.config.oltc_trafo_indices,
            shunt_bus_indices=self.config.shunt_bus_indices,
            shunt_q_steps_mvar=self.config.shunt_q_steps_mvar,
        )
        if pcc_in_trafo:
            kw["trafo_indices"] = self.config.pcc_trafo_indices
        elif pcc_in_trafo3w:
            kw["trafo3w_indices"] = self.config.pcc_trafo_indices
        # else: no trafo Q outputs — PCC handled via identity below

        H_physical, mappings = self.sensitivities.build_sensitivity_matrix_H(
            **kw,
        )

        # H_physical columns: [DER (n_der), OLTC (n_oltc), shunt (n_shunt)]
        # H_physical rows: depends on whether PCC trafos are in the network
        #   If present: [Q_trafo (n_pcc), V_bus (n_v), I_line (n_i)]
        #   If absent:  [V_bus (n_v), I_line (n_i)]
        #
        # Target H rows:      [V_bus (n_v), Q_PCC (n_pcc), I_line (n_i)]
        # Target H columns:   [DER (n_der), PCC_set (n_pcc), OLTC (n_oltc),
        #                       shunt (n_shunt)]

        # Determine how many Q_trafo rows are in H_physical
        has_pcc_rows = pcc_in_trafo or pcc_in_trafo3w
        n_q_phys = n_pcc if has_pcc_rows else 0

        # H_physical row ranges:
        #   Q_trafo : 0              .. n_q_phys - 1       (may be 0 rows)
        #   V_bus   : n_q_phys       .. n_q_phys + n_v - 1
        #   I_line  : n_q_phys + n_v .. end
        # H_physical column ranges:
        #   DER   : 0         .. n_der - 1
        #   OLTC  : n_der     .. n_der + n_oltc - 1
        #   shunt : n_der + n_oltc .. end

        # Map to target row ordering [V, Q_PCC, I]
        # DER columns → target column 0..n_der-1
        # V rows
        H[:n_v, :n_der] = H_physical[n_q_phys:n_q_phys + n_v, :n_der]
        # Q_PCC rows (from physical if available, else zero — will be
        # overridden by identity columns for PCC setpoints below)
        if has_pcc_rows:
            H[n_v:n_v + n_pcc, :n_der] = H_physical[:n_q_phys, :n_der]
        # I rows
        H[n_v + n_pcc:, :n_der] = H_physical[n_q_phys + n_v:, :n_der]

        # PCC setpoint columns → target column n_der..n_der+n_pcc-1
        #
        # Q_PCC_set uses *load convention* on the HV port: positive means
        # reactive power flowing INTO the coupler FROM the HV bus.  An
        # increase in Q_PCC_set therefore *removes* Q from the HV bus,
        # which is the opposite of a Q injection.  The Jacobian
        # sensitivities (dV/dQ, dI/dQ) are computed in generator
        # convention (positive = injection), so the PCC columns must be
        # negated:  ∂V/∂Q_PCC_set(load) = −∂V/∂Q_inj(gen).
        pcc_hv_buses = []
        for t in self.config.pcc_trafo_indices:
            if pcc_in_trafo3w:
                pcc_hv_buses.append(int(net.trafo3w.at[t, "hv_bus"]))
            elif pcc_in_trafo:
                pcc_hv_buses.append(int(net.trafo.at[t, "hv_bus"]))
        if pcc_hv_buses:
            # ∂V/∂Q_inj at PCC HV buses → negate for load convention
            dV_dQ_pcc, obs_map, pcc_map = self.sensitivities.compute_dV_dQ_der(
                der_bus_indices=pcc_hv_buses,
                observation_bus_indices=self.config.voltage_bus_indices,
            )
            for j_pcc, bus in enumerate(pcc_hv_buses):
                col = n_der + j_pcc
                if bus in pcc_map:
                    j_jac = pcc_map.index(bus)
                    for i_obs, obs_bus in enumerate(obs_map):
                        i_row = self.config.voltage_bus_indices.index(obs_bus)
                        H[i_row, col] = -dV_dQ_pcc[i_obs, j_jac]
            # ∂I/∂Q_inj at PCC HV buses → negate for load convention
            if self.config.current_line_indices:
                dI_dQ_pcc, line_map, pcc_map_i = \
                    self.sensitivities.compute_dI_dQ_der_matrix(
                        line_indices=self.config.current_line_indices,
                        der_bus_indices=pcc_hv_buses,
                    )
                for j_pcc, bus in enumerate(pcc_hv_buses):
                    col = n_der + j_pcc
                    if bus in pcc_map_i:
                        j_jac = pcc_map_i.index(bus)
                        for i_line in range(len(line_map)):
                            H[n_v + n_pcc + i_line, col] = -dI_dQ_pcc[i_line, j_jac]
        # Q_PCC tracking: ∂Q_PCC_j / ∂Q_PCC_set_j = 1
        for j in range(n_pcc):
            H[n_v + j, n_der + j] = 1.0

        # OLTC columns → target column n_der+n_pcc+n_gen..n_der+n_pcc+n_gen+n_oltc-1
        col_oltc_phys = slice(n_der, n_der + n_oltc)
        col_oltc_target = slice(n_der + n_pcc + n_gen, n_der + n_pcc + n_gen + n_oltc)
        H[:n_v, col_oltc_target] = H_physical[
            n_q_phys:n_q_phys + n_v, col_oltc_phys
        ]
        if has_pcc_rows:
            H[n_v:n_v + n_pcc, col_oltc_target] = H_physical[
                :n_q_phys, col_oltc_phys
            ]
        H[n_v + n_pcc:, col_oltc_target] = H_physical[
            n_q_phys + n_v:, col_oltc_phys
        ]

        # Shunt columns → target column n_der+n_pcc+n_gen+n_oltc..end
        col_sh_phys = slice(n_der + n_oltc, n_der + n_oltc + n_shunt)
        col_sh_target = slice(n_der + n_pcc + n_gen + n_oltc, n_controls)
        H[:n_v, col_sh_target] = H_physical[
            n_q_phys:n_q_phys + n_v, col_sh_phys
        ]
        if has_pcc_rows:
            H[n_v:n_v + n_pcc, col_sh_target] = H_physical[
                :n_q_phys, col_sh_phys
            ]
        H[n_v + n_pcc:, col_sh_target] = H_physical[
            n_q_phys + n_v:, col_sh_phys
        ]

        # --- AVR columns: ∂V_obs / ∂V_gen from Jacobian-based sensitivity ---
        avr_start = n_der + n_pcc
        # The PV terminal buses of the generators (needed for sensitivity calc)
        gen_terminal_buses = [
            int(net.gen.at[g, "bus"]) for g in self.config.gen_indices
        ]
        dV_dVgen, obs_map, gen_map = \
            self.sensitivities.compute_dV_dVgen_matrix(
                gen_bus_indices_pp=gen_terminal_buses,
                observation_bus_indices=self.config.voltage_bus_indices,
            )
        for k, gen_bus_pp in enumerate(gen_terminal_buses):
            col = avr_start + k
            if gen_bus_pp in gen_map:
                j_gen = gen_map.index(gen_bus_pp)
                for i_obs, obs_bus in enumerate(obs_map):
                    i_row = self.config.voltage_bus_indices.index(obs_bus)
                    H[i_row, col] = dV_dVgen[i_obs, j_gen]

        self._H_cache = H
        self._H_mappings = mappings

        # TSO H column layout: [DER | PCC_set | V_gen_set | OLTC | shunt]
        col_shunt_start = n_der + n_pcc + n_gen + n_oltc

        self._sensitivity_updater = SensitivityUpdater(
            H=H,
            mappings=mappings,
            sensitivities=self.sensitivities,
            update_interval_min=1,
            col_shunt_start=col_shunt_start,
        )

        return H

    def step(self, measurement: Measurement) -> ControllerOutput:
        """
        Execute one OFO iteration with voltage-dependent sensitivity updates.

        Before delegating to :meth:`BaseOFOController.step`, this method
        rescales shunt columns by ``(V_measured / V_cached)²`` to account
        for the constant-susceptance nature of shunt devices.
        """
        # Ensure H is built
        if self._H_cache is None:
            self._build_sensitivity_matrix()
        # Update state-dependent columns (shunt V² rescaling)
        if self._sensitivity_updater is not None:
            self._H_cache = self._sensitivity_updater.update(
                measurement, measurement.iteration,
            )
        return super().step(measurement)

    def invalidate_sensitivity_cache(self) -> None:
        """Invalidate the cached sensitivity matrix."""
        self._H_cache = None
        self._H_mappings = None
        self._sensitivity_updater = None
