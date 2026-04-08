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

from collections import defaultdict
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
from core.der_mapping import DERMapping
from core.message import SetpointMessage, CapabilityMessage
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.sensitivity_updater import SensitivityUpdater


@dataclass
class TSOControllerConfig:
    """
    Configuration for the TSO controller.

    Attributes
    ----------
    der_indices : List[int]
        Pandapower sgen indices for transmission-connected DERs.
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
    der_indices: List[int]
    pcc_trafo_indices: List[int]
    pcc_dso_controller_ids: List[str]
    oltc_trafo_indices: List[int]
    shunt_bus_indices: List[int]
    shunt_q_steps_mvar: List[float]
    voltage_bus_indices: List[int]
    current_line_indices: List[int]
    v_min_pu: float = 0.90
    v_max_pu: float = 1.10
    i_max_pu: float = 1.0 #1.0
    current_line_max_i_ka: Optional[List[float]] = None
    """Per-line thermal rating [kA]. Must have the same length as
    ``current_line_indices``. If ``None``, limits are not enforced."""
    v_setpoints_pu: Optional[NDArray[np.float64]] = None
    g_v: float = 1.0
    gen_indices: List[int] = field(default_factory=list)
    gen_bus_indices: List[int] = field(default_factory=list)
    gen_vm_min_pu: float = 1.00
    gen_vm_max_pu: float = 1.07

    k_t_avt: float = 0.0
    """Achieved-Value Tracking factor for PCC-Q reset.
    0.0 = no reset (current behaviour), 1.0 = full reset (recommended).
    Blends: u_pcc <- (1 - k_t) * u_cmd + k_t * q_measured."""

    der_mapping: Optional[DERMapping] = None
    """Per-DER mapping for individual sgen-level control.  When
    provided, enables per-DER decision variables in the MIQP
    and factorises the sensitivity matrix as H_der = H_bus @ E.
    If None, the controller uses the legacy sgen-index-based control."""

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        # When a DER mapping is provided, derive der_indices from it
        if self.der_mapping is not None:
            object.__setattr__(
                self, "der_indices",
                list(self.der_mapping.sgen_indices),
            )
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
        if self.current_line_max_i_ka is not None:
            if len(self.current_line_max_i_ka) != len(self.current_line_indices):
                raise ValueError(
                    f"current_line_max_i_ka length ({len(self.current_line_max_i_ka)}) "
                    f"must match current_line_indices length "
                    f"({len(self.current_line_indices)})"
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
        [ V_bus (p.u.) | I_line (p.u.) ]

    Note: Q_PCC rows have been removed from the output vector and H matrix.
    Q_PCC_set is a direct decision variable; the TSO does not need to
    observe Q_PCC as an output.  The code for Q_PCC rows is retained
    (commented out) in _extract_outputs, _get_output_limits, and
    _build_sensitivity_matrix so it can be re-enabled if needed.

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
        self.pcc_capability_min_mvar = np.full(n_pcc, -1E-6)
        self.pcc_capability_max_mvar = np.full(n_pcc, +1E-6)

        # Cache for the sensitivity matrix
        self._H_cache: Optional[NDArray[np.float64]] = None
        self._H_mappings: Optional[Dict] = None
        self._sensitivity_updater: Optional[SensitivityUpdater] = None

        # Cached measurement for capability-curve bounds (set in step())
        self._last_measurement: Optional[Measurement] = None

        # Achieved-Value Tracking verbosity (set from run_cascade)
        self._avt_verbose: int = 0

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

        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_pcc = len(self.config.pcc_trafo_indices)

        # PCC setpoint values sit after the DER entries in u
        q_pcc_setpoints = self._u_current[n_der:n_der + n_pcc]

        # Group PCC trafos by DSO ID (each DSO may have multiple PCC trafos)
        dso_trafos: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, dso_id in enumerate(self.config.pcc_dso_controller_ids):
            dso_trafos[dso_id].append((self.config.pcc_trafo_indices[i], i))

        messages: List[SetpointMessage] = []
        for dso_id, trafo_pairs in dso_trafos.items():
            trafo_indices = [t for t, _ in trafo_pairs]
            q_vals = [float(q_pcc_setpoints[idx]) for _, idx in trafo_pairs]
            msg = SetpointMessage(
                source_controller_id=self.controller_id,
                target_controller_id=dso_id,
                iteration=self.iteration,
                interface_transformer_indices=np.array(
                    trafo_indices, dtype=np.int64
                ),
                q_setpoints_mvar=np.array(
                    q_vals, dtype=np.float64
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

    def _get_der_mapping(self) -> Optional[DERMapping]:
        """Return the DER mapping from config."""
        return self.config.der_mapping

    def _get_n_der_bus(self) -> int:
        """Number of unique DER bus columns in H_bus."""
        mapping = self.config.der_mapping
        if mapping is not None:
            return mapping.n_unique_bus
        return len(self.config.der_indices)

    def _get_control_structure(self) -> Tuple[int, int, List[int]]:
        """
        Define the control variable structure.

        Ordering: [ Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
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
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        n_total = n_der + n_pcc + n_gen + n_oltc + n_shunt
        u = np.zeros(n_total, dtype=np.float64)
        idx = 0

        # --- Transmission-connected DER Q setpoints ---
        if mapping is not None:
            for sgen_idx in mapping.sgen_indices:
                meas_idx = np.where(measurement.der_indices == sgen_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"DER sgen {sgen_idx} not found in measurement"
                    )
                u[idx] = float(measurement.der_q_mvar[meas_idx[0]])
                idx += 1
        else:
            for der_idx in self.config.der_indices:
                meas_idx = np.where(measurement.der_indices == der_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"DER {der_idx} not found in measurement.der_indices"
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

        Output ordering: [ V_bus | I_line ]

        Note: Q_PCC rows have been removed.  Q_PCC_set is a direct
        decision variable and does not need to appear as an output.
        The commented-out block below can be re-enabled if Q_PCC
        output tracking is needed in the future.
        """
        n_v = len(self.config.voltage_bus_indices)
        n_i = len(self.config.current_line_indices)
        n_outputs = n_v + n_i

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

        # --- Q_PCC rows removed (Q_PCC_set is a direct decision variable) ---
        # To re-enable, uncomment this block and update n_outputs above:
        # n_pcc = len(self.config.pcc_trafo_indices)
        # n_outputs = n_v + n_pcc + n_i  # (also update above)
        # for trafo_idx in self.config.pcc_trafo_indices:
        #     meas_idx = np.where(
        #         measurement.interface_transformer_indices == trafo_idx
        #     )[0]
        #     if len(meas_idx) == 0:
        #         raise ValueError(
        #             f"PCC transformer {trafo_idx} not found in measurement"
        #         )
        #     y[idx] = measurement.interface_q_hv_side_mvar[meas_idx[0]]
        #     idx += 1

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
        """Extract DER active power from measurement for capability calculation."""
        p_current = measurement.der_p_mw.copy()

        return p_current

    def _extract_trafo_reactive_power(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Extract current trafo ractive power flow for capability calculation.
        """
        q_current = measurement.interface_q_hv_side_mvar.copy()
        return q_current

    def _compute_input_bounds(
        self,
        tso_dso_interface_q_current: NDArray[np.float64],
        der_p_current: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute operating-point-dependent input bounds.

        For PCC setpoints, the bounds are updated from capability messages.
        AVR setpoints start from the fixed ``[gen_vm_min, gen_vm_max]`` band
        and are then tightened using the detailed generator capability curve
        (Milano §12.2.1) when ``gen_params`` is available and the measured
        generator Q approaches a thermal limit.

        The tightening logic works as follows: if the current Q_gen is
        within a margin of Q_max (overexcitation / rotor limit), the upper
        V_gen bound is clamped to the current setpoint so the MIQP cannot
        increase voltage further (which would push Q_gen past the limit).
        Similarly, if Q_gen is near Q_min (underexcitation / stator limit),
        the lower V_gen bound is clamped to prevent further voltage decrease.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
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
        u_lower[n_der:n_der + n_pcc] = tso_dso_interface_q_current + self.pcc_capability_min_mvar
        u_upper[n_der:n_der + n_pcc] = tso_dso_interface_q_current + self.pcc_capability_max_mvar

        # DEBUG
        # print(f'ppc capability min: {u_lower[n_der:n_der + n_pcc]}')
        # print(f'pcc capability max: {u_upper[n_der:n_der + n_pcc]}')

        # --- AVR setpoint bounds ---
        avr_start = n_der + n_pcc
        avr_end = avr_start + n_gen
        u_lower[avr_start:avr_end] = self.config.gen_vm_min_pu
        u_upper[avr_start:avr_end] = self.config.gen_vm_max_pu

        # Tighten V_gen bounds using generator capability curve
        if (
            self.actuator_bounds.gen_params is not None
            and self._last_measurement is not None
            and len(self._last_measurement.gen_p_mw) == n_gen
            and len(self._last_measurement.gen_q_mvar) == n_gen
        ):
            meas = self._last_measurement
            gen_p = meas.gen_p_mw
            gen_v = meas.gen_vm_pu
            gen_q = meas.gen_q_mvar

            gen_q_min, gen_q_max = self.actuator_bounds.compute_gen_q_bounds(
                gen_p, gen_v,
            )

            # Margin: clamp V_gen when Q is within this fraction of the limit
            margin_frac = 0.1  # 10% of the capability range

            for k in range(n_gen):
                q_range = gen_q_max[k] - gen_q_min[k]
                if q_range <= 0:
                    continue
                margin = margin_frac * q_range

                # Near overexcitation limit → block V_gen increase
                # Clamp u_upper down to the current setpoint, but never
                # below the absolute lower bound gen_vm_min_pu (otherwise
                # u_lower > u_upper if the current setpoint happens to
                # sit below the configured lower bound, e.g. when the
                # bounds are tightened after the plant equilibrated at a
                # lower voltage).
                if gen_q[k] >= gen_q_max[k] - margin:
                    u_upper[avr_start + k] = max(
                        self.config.gen_vm_min_pu,
                        min(
                            u_upper[avr_start + k],
                            self._u_current[avr_start + k],
                        ),
                    )

                # Near underexcitation limit → block V_gen decrease.
                # Clamp u_lower up to the current setpoint, but never
                # above the absolute upper bound gen_vm_max_pu (symmetric
                # safety to the overexcitation branch above).
                if gen_q[k] <= gen_q_min[k] + margin:
                    u_lower[avr_start + k] = min(
                        self.config.gen_vm_max_pu,
                        max(
                            u_lower[avr_start + k],
                            self._u_current[avr_start + k],
                        ),
                    )

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

        Output ordering: [ V_bus | I_line ]

        Voltage outputs are constrained to the permissible band
        [v_min_pu, v_max_pu].  Tracking toward voltage setpoints
        (if configured) is handled by the quadratic objective in
        ``_compute_objective_gradient()``, not by tight constraints.

        Note: Q_PCC rows have been removed from the output vector.
        """
        n_v = len(self.config.voltage_bus_indices)
        n_i = len(self.config.current_line_indices)
        n_outputs = n_v + n_i

        y_lower = np.zeros(n_outputs)
        y_upper = np.zeros(n_outputs)
        idx = 0

        # --- Voltage limits (band constraints) ---
        for j in range(n_v):
            y_lower[idx] = self.config.v_min_pu
            y_upper[idx] = self.config.v_max_pu
            idx += 1

        # --- Q_PCC limits removed (Q_PCC_set is a direct decision variable) ---
        # To re-enable, uncomment this block and update n_outputs above:
        # n_pcc = len(self.config.pcc_trafo_indices)
        # n_outputs = n_v + n_pcc + n_i  # (also update above)
        # for _ in range(n_pcc):
        #     y_lower[idx] = -1E6
        #     y_upper[idx] = 1E6
        #     idx += 1

        # --- Current limits (upper only, kA) ---
        for j in range(n_i):
            if self.config.current_line_max_i_ka is not None:
                i_lim_ka = self.config.i_max_pu * self.config.current_line_max_i_ka[j]
            else:
                i_lim_ka = 1E6  # no limit if ratings not provided
            y_lower[idx] = 0.0
            y_upper[idx] = i_lim_ka
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

        n_v = len(self.config.voltage_bus_indices)

        # --- Component 1: DER usage regularisation ---
        # Handled implicitly by g_u in build_miqp_problem; no explicit
        # gradient needed here (build_miqp_problem adds 2*alpha*g_u*u_current).

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
            # Use expanded H (per-DER) for correct gradient dimensions
            H_bus = self._build_sensitivity_matrix()
            H = self._expand_H_to_der_level(H_bus)
            # Voltage outputs are the first n_v rows of H
            dV_du = H[:n_v, :]

            # grad_f contribution: 2 * g_v * (V - V_set)^T * dV/du
            grad_f += 2.0 * self.config.g_v * (v_error @ dV_du)

        return grad_f

    def _build_sensitivity_matrix(self) -> NDArray[np.float64]:
        """
        Build the input-output sensitivity matrix H.

        The matrix maps control variable changes to output changes:
            Δy ≈ H · Δu

        Columns correspond to:  [ Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]
        Rows correspond to:     [ V_bus | I_line ]

        Note: Q_PCC rows have been removed from the output vector.
        Q_PCC_set is a direct decision variable; its effect on the
        output vector is only through the voltage and current rows
        (via ∂V/∂Q_PCC_set and ∂I/∂Q_PCC_set).  The code for Q_PCC
        identity rows is retained (commented out) below.

        Returns
        -------
        H : NDArray[np.float64]
            Sensitivity matrix of shape (n_outputs, n_controls).
        """
        if self._H_cache is not None:
            return self._H_cache

        # When a DER mapping is active, the H matrix is built at bus-level
        # (one column per unique DER bus).  The base class _expand_H_to_der_level
        # will expand to per-DER via the E matrix.
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der_bus = mapping.n_unique_bus
        else:
            n_der_bus = len(self.config.der_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_v = len(self.config.voltage_bus_indices)
        n_i = len(self.config.current_line_indices)

        n_controls = n_der_bus + n_pcc + n_gen + n_oltc + n_shunt
        n_outputs = n_v + n_i  # Q_PCC rows removed

        H = np.zeros((n_outputs, n_controls), dtype=np.float64)

        # --- Physical sensitivities for DER / OLTC / shunt columns ---
        net = self.sensitivities.net

        if mapping is not None:
            der_bus_indices = mapping.unique_bus_indices
        else:
            der_bus_indices = [
                int(net.sgen.at[s, 'bus']) for s in self.config.der_indices
            ]
        n_der = len(der_bus_indices)  # alias for column indexing below

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

        # Note: We no longer request Q_trafo rows from the physical
        # sensitivity builder, since Q_PCC is not an output anymore.
        # However, we still pass trafo indices so the builder can
        # include them in the internal row mapping (may affect V/I rows).
        kw = dict(
            der_bus_indices=der_bus_indices,
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

        # build_sensitivity_matrix_H requires at least one physical input
        # (DER / OLTC / shunt).  When all are absent (e.g. add_tso_ders=False
        # and no machine OLTCs), skip the call — H_physical is all-zero.
        has_physical_inputs = (
            len(der_bus_indices) > 0
            or len(self.config.oltc_trafo_indices) > 0
            or len(self.config.shunt_bus_indices) > 0
        )
        mappings: dict = {}

        if has_physical_inputs:
            H_physical, mappings = self.sensitivities.build_sensitivity_matrix_H(
                **kw,
            )

            # H_physical columns: [DER (n_der), OLTC (n_oltc), shunt (n_shunt)]
            # H_physical rows (when PCC trafos present):
            #   [Q_trafo (n_pcc), V_bus (n_v), I_line (n_i)]
            # H_physical rows (when no PCC trafos):
            #   [V_bus (n_v), I_line (n_i)]
            #
            # Target H rows:      [V_bus (n_v), I_line (n_i)]
            # Target H columns:   [DER (n_der), PCC_set (n_pcc), V_gen (n_gen),
            #                       OLTC (n_oltc), shunt (n_shunt)]

            has_pcc_rows = pcc_in_trafo or pcc_in_trafo3w
            n_q_phys = n_pcc if has_pcc_rows else 0
            n_i_copy = 0

            # H_physical row ranges (skip Q_trafo rows):
            #   V_bus   : n_q_phys       .. n_q_phys + n_v - 1
            #   I_line  : n_q_phys + n_v .. end

            # DER columns -> target column 0..n_der-1
            n_i_phys = H_physical.shape[0] - n_q_phys - n_v
            n_i_copy = min(n_i, n_i_phys)
            H[:n_v, :n_der] = H_physical[n_q_phys:n_q_phys + n_v, :n_der]
            if n_i_copy > 0:
                H[n_v:n_v + n_i_copy, :n_der] = H_physical[n_q_phys + n_v:n_q_phys + n_v + n_i_copy, :n_der]

        # --- Q_PCC rows removed from H ---
        # Previously, Q_PCC occupied rows H[n_v:n_v+n_pcc, :].
        # The identity ∂Q_PCC_j/∂Q_PCC_set_j = 1 was placed at
        # H[n_v+j, n_der+j].  This is no longer needed because
        # Q_PCC_set is a pure decision variable without output tracking.
        #
        # To re-enable Q_PCC rows, change n_outputs = n_v + n_pcc + n_i,
        # shift I rows down by n_pcc, and uncomment:
        # for j in range(n_pcc):
        #     H[n_v + j, n_der + j] = 1.0

        # PCC setpoint columns → target column n_der..n_der+n_pcc-1
        # Q_PCC_set uses *load convention* on the HV port: positive means
        # reactive power flowing INTO the coupler FROM the HV bus.
        # Jacobian sensitivities use generator convention, so PCC columns
        # must be negated: ∂V/∂Q_PCC_set(load) = −∂V/∂Q_inj(gen).
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
                            H[n_v + i_line, col] = -dI_dQ_pcc[i_line, j_jac]

        if has_physical_inputs:
            # OLTC columns → target column n_der+n_pcc+n_gen..n_der+n_pcc+n_gen+n_oltc-1
            col_oltc_phys = slice(n_der, n_der + n_oltc)
            col_oltc_target = slice(n_der + n_pcc + n_gen, n_der + n_pcc + n_gen + n_oltc)
            H[:n_v, col_oltc_target] = H_physical[
                n_q_phys:n_q_phys + n_v, col_oltc_phys
            ]
            if n_i_copy > 0:
                H[n_v:n_v + n_i_copy, col_oltc_target] = H_physical[
                    n_q_phys + n_v:n_q_phys + n_v + n_i_copy, col_oltc_phys
                ]

            # Shunt columns -> target column n_der+n_pcc+n_gen+n_oltc..end
            col_sh_phys = slice(n_der + n_oltc, n_der + n_oltc + n_shunt)
            col_sh_target = slice(n_der + n_pcc + n_gen + n_oltc, n_controls)
            H[:n_v, col_sh_target] = H_physical[
                n_q_phys:n_q_phys + n_v, col_sh_phys
            ]
            if n_i_copy > 0:
                H[n_v:n_v + n_i_copy, col_sh_target] = H_physical[
                    n_q_phys + n_v:n_q_phys + n_v + n_i_copy, col_sh_phys
                ]

        # --- AVR columns: ∂V_obs / ∂V_gen from Jacobian-based sensitivity ---
        avr_start = n_der + n_pcc
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

        if self.config.current_line_indices:
            dI_dVgen, line_map, gen_map_i = \
                self.sensitivities.compute_dI_dVgen_matrix(
                    line_indices=self.config.current_line_indices,
                    gen_bus_indices_pp=gen_terminal_buses,
                )
            for k, gen_bus_pp in enumerate(gen_terminal_buses):
                col = avr_start + k
                if gen_bus_pp in gen_map_i:
                    j_gen = gen_map_i.index(gen_bus_pp)
                    for i_line, l_idx in enumerate(line_map):
                        # The lines are mapped to the lower part of H (after voltages)
                        H[n_v + i_line, col] = dI_dVgen[i_line, j_gen]

        # NOTE: ∂Q_PCC / ∂V_gen entries are NOT filled here.
        # Q_PCC rows have been removed entirely. The sensitivity method
        # compute_dQtrafo3w_hv_dVgen_matrix() is available if Q_PCC
        # rows are re-enabled in the future.

        if not np.all(np.isfinite(H)):
            nan_rows, nan_cols = np.where(~np.isfinite(H))
            raise ValueError(
                f"[TSOController {self.controller_id}] Non-finite entries in "
                f"sensitivity matrix H at (row, col): "
                f"{list(zip(nan_rows.tolist(), nan_cols.tolist()))}. "
                f"Check Jacobian computation for buses/lines in the config."
            )

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

    def apply_avt_reset(self, measurement: Measurement) -> None:
        """Replace PCC-Q entries in _u_current with measured achieved values.

        Implements the Achieved-Value Tracking (AVT) anti-windup mechanism.
        Before each MIQP solve, the PCC-Q components of the internal state
        are reset toward the physically realised Q at the TSO-DSO interface,
        controlled by ``config.k_t_avt`` (0 = no reset, 1 = full reset).
        """
        if self._u_current is None:
            return
        n_pcc = len(self.config.pcc_trafo_indices)
        if n_pcc == 0:
            return
        k_t = self.config.k_t_avt
        if k_t == 0.0:
            return

        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        pcc_slice = slice(n_der, n_der + n_pcc)

        # Extract measured Q at each PCC interface
        q_measured = np.empty(n_pcc, dtype=np.float64)
        for i, trafo_idx in enumerate(self.config.pcc_trafo_indices):
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) == 0:
                return  # measurement incomplete — skip reset
            q_measured[i] = measurement.interface_q_hv_side_mvar[meas_idx[0]]

        u_old = self._u_current[pcc_slice].copy()
        # Blend: (1 - k_t) * u_commanded + k_t * q_measured
        self._u_current[pcc_slice] = (1.0 - k_t) * u_old + k_t * q_measured

        if self._avt_verbose > 1:
            delta = self._u_current[pcc_slice] - u_old
            if np.any(np.abs(delta) > 1e-6):
                print(f"    [AVT] PCC-Q reset (k_t={k_t}):")
                for i, t in enumerate(self.config.pcc_trafo_indices):
                    print(f"      trafo {t}: {u_old[i]:.2f} -> "
                          f"{self._u_current[n_der + i]:.2f} Mvar")

    def step(self, measurement: Measurement) -> ControllerOutput:
        """
        Execute one OFO iteration with voltage-dependent sensitivity updates.

        Before delegating to :meth:`BaseOFOController.step`, this method
        rescales shunt columns by ``(V_measured / V_cached)²`` to account
        for the constant-susceptance nature of shunt devices, and caches
        the measurement for use in ``_compute_input_bounds`` (generator
        capability-curve bounds depend on measured P and V).
        """
        # Cache measurement for capability-curve bounds in _compute_input_bounds
        self._last_measurement = measurement
        # Ensure H is built
        if self._H_cache is None:
            self._build_sensitivity_matrix()
        # Update state-dependent columns (shunt V² rescaling)
        if self._sensitivity_updater is not None:
            self._H_cache = self._sensitivity_updater.update(
                measurement, measurement.iteration,
            )
        # Achieved-Value Tracking: reset PCC-Q to measured values
        self.apply_avt_reset(measurement)

        return super().step(measurement)

    def invalidate_sensitivity_cache(self) -> None:
        """Invalidate the cached sensitivity matrix.

        Also clears the per-DER expansion cache on the base class so the
        next ``step()`` rebuilds ``H_der`` from the freshly computed
        ``H_bus`` (e.g. after a contingency or topology change).
        """
        self._H_cache = None
        self._H_mappings = None
        self._sensitivity_updater = None
        self._H_der_cache = None
        self._H_der_cache_base_id = None
