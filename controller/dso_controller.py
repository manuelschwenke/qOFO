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
from core.message import SetpointMessage, CapabilityMessage, ShuntDisturbanceMessage
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.sensitivity_updater import SensitivityUpdater


@dataclass
class DSOControllerConfig:
    """
    Configuration for the DSO controller.
    
    Attributes
    ----------
    der_indices : List[int]
        Pandapower sgen indices for controllable DERs.
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
    v_setpoints_pu : Optional[NDArray[np.float64]]
        Voltage setpoints at monitored DN buses [p.u.].  If provided,
        the objective includes a soft voltage-schedule tracking term.
        Must have the same length as *voltage_bus_indices*.
        Default ``None`` (no voltage tracking).
    g_v : float
        Weight for voltage-schedule tracking in the objective function.
        Scales the gradient ``2 · g_v · (V - V_set)^T · ∂V/∂u``.
        Should be kept small relative to the TSO's g_v so that DSO
        voltage tracking remains a secondary, soft objective behind
        Q-interface tracking.  Default 1.0.
    """
    der_indices: List[int]
    oltc_trafo_indices: List[int]
    shunt_bus_indices: List[int]
    shunt_q_steps_mvar: List[float]
    interface_trafo_indices: List[int]
    voltage_bus_indices: List[int]
    current_line_indices: List[int]
    v_min_pu: float = 0.9
    v_max_pu: float = 1.1
    i_max_pu: float = 1.0 # 1.0
    current_line_max_i_ka: Optional[List[float]] = None
    """Per-line thermal rating [kA]. Must have the same length as
    ``current_line_indices``. If ``None``, limits are not enforced."""
    g_q: float = 1.0
    g_qi: float = 0.0
    """Weight for integral Q-tracking term.  When > 0, a leaky integrator
    accumulates Q-interface errors over iterations, building pressure for
    discrete switching actions (OLTC, shunts) when continuous DERs cannot
    satisfy the setpoint.  The integral gradient contribution is
    ``2 · g_qi · integral^T · ∂Q/∂u``.  Default 0.0 (disabled)."""
    lambda_qi: float = 0.9
    """Decay factor for the leaky integrator (0 ≤ λ ≤ 1).  Controls how
    fast past errors are forgotten.  1.0 = pure integration (no decay),
    0.9 = gradual decay.  Only used when *g_qi* > 0."""
    q_integral_max_mvar: float = 50.0
    """Anti-windup clamp for the integral accumulator [Mvar].  Limits each
    element of the integral to ``[-max, +max]`` to prevent excessive
    buildup when Q-capability limits are hit."""
    v_setpoints_pu: Optional[NDArray[np.float64]] = None
    g_v: float = 1.0
    gamma_oltc_q: float = 0.0
    """Role-based Q-tracking attenuation for OLTC columns.  Scales the
    Q-interface tracking gradient on OLTC decision variables:
        grad_f[oltc] += 2 · g_q · γ · (Q - Q_set)^T · ∂Q/∂s
    With γ = 0 (default), OLTCs are driven purely by voltage deviations
    and the physical Q effect of any tap change is still captured in the
    output constraints (H matrix).  With γ = 1, OLTCs are equally
    incentivised for Q-tracking (legacy behaviour).  Values in (0, 1)
    provide partial Q-sensitivity as a last-resort mechanism."""
    der_mapping: Optional[DERMapping] = None
    """Per-DER mapping for individual sgen-level control.  When
    provided, enables per-DER decision variables in the MIQP
    and factorises the sensitivity matrix as H_der = H_bus @ E.
    If None, the controller uses the legacy sgen-index-based control."""

    # ── Grid-forming converter gens (PV-bus voltage actuators) ──────────────
    gridforming_gen_indices: List[int] = field(default_factory=list)
    """``net.gen`` row indices for grid-forming converter gens managed by
    this DSO controller (e.g. STATCOM-capable units classified as
    grid-forming via ``MultiTSOConfig.der_mode_overrides``).  The DSO
    OFO commands ``vm_pu`` on each, in a separate ``V_gf`` actuator
    block.  Empty list ⇒ no grid-forming converters at this DSO
    (default — the legacy classification puts every DSO DER in
    grid-following mode)."""

    gridforming_gen_buses: List[int] = field(default_factory=list)
    """Bus index for each entry in ``gridforming_gen_indices``."""

    gridforming_gen_sn_mva: List[float] = field(default_factory=list)
    """Rated apparent power [MVA] per grid-forming gen.  Used to compute
    the STATCOM Q capability ``±sqrt(Sn² − P²)`` enforced as a soft
    output constraint on the realised Q."""

    gridforming_gen_op_diagrams: List[str] = field(default_factory=list)
    """Per-gen operating-diagram label (default ``"STATCOM"``)."""

    gridforming_vm_min_pu: float = 0.95
    """Lower bound on the V_gf voltage setpoint [pu]."""

    gridforming_vm_max_pu: float = 1.10
    """Upper bound on the V_gf voltage setpoint [pu]."""

    rho_q_gridforming: float = 1e2
    """Soft-constraint penalty for the Q_gf output."""

    # ── Stage-2: Q(V) plant model + how the OFO drives it ──────────────────
    use_qv_local_loop: bool = False
    """Stage-2 master switch — controls **plant model** only.  When True,
    the plant has a local Q(V) loop on every grid-following DSO sgen.
    See :attr:`qv_apply_mode` for how the OFO interacts with that
    plant model (V_ref-direct or Q+shim).

    Default ``False`` keeps the Stage-1 direct-Q path (no Q(V) plant)."""

    qv_apply_mode: str = "q_shim"
    """How the OFO drives the Q(V) plant when ``use_qv_local_loop=True``.

    * ``"v_ref"`` — OFO commands per-sgen V_ref (pu_v).  Activates the
      K-transform on the H matrix, Q_realized soft output rows, and
      V_ref bounds on the input vector.  Q tracking is structurally
      capped by the V_ref bound + closed-loop V_bus feedback; tuning
      ``g_q`` and ``g_w_dso_der_vref`` has limited effect (see the
      empirical sweeps in ``tests/diag_final_tuning_sweep.py``).
      Highest physical fidelity (the OFO output IS the V_ref the
      converter would receive in the real system).

    * ``"q_shim"`` (default) — OFO solves the same MIQP as Stage 1:
      DER block is Q (Mvar), no K-transform, no Q_realized rows,
      ``g_w_dso_der`` units are 1/Mvar², capability bounds are
      ``[Q_min(P), Q_max(P)]``.  The plant-side apply step inverts the
      droop locally (``vm_pu_ref = V_bus_meas + Q_cmd / k``) so the
      Q(V) loop converges to ``Q ≈ Q_cmd``.  Recovers Stage-1 Q
      tracking quality while keeping the realistic Q(V) plant model.

    Ignored when ``use_qv_local_loop=False``."""

    @property
    def ofo_in_v_ref_mode(self) -> bool:
        """True iff the OFO operates over V_ref (vs Q) for DSO DERs.

        Equivalent to ``use_qv_local_loop and qv_apply_mode == 'v_ref'``.
        Used throughout the controller to gate the K-transform,
        Q_realized rows, V_ref bounds, V_ref-from-measurement reads,
        and the ``"dso_der_vref"`` g_w-adapter class name."""
        return bool(self.use_qv_local_loop and self.qv_apply_mode == "v_ref")

    qv_slope_pu: float = 0.07
    """Q(V) droop slope [pu_q / pu_v].  Must match the value passed to
    :func:`controller.dso_qv_local_loop.install_qv_local_loops` so the
    OFO's K-transform agrees with the plant-side loop."""

    qv_v_ref_min_pu: float = 0.95
    """Lower bound on the per-DER V_ref MIQP control variable."""

    qv_v_ref_max_pu: float = 1.10
    """Upper bound on the per-DER V_ref MIQP control variable."""

    qv_saturation_eps_mvar: float = 1.0
    """Tolerance for the active-set saturation detector.  When realized
    Q is within this tolerance of either capability rail, the
    corresponding diagonal of K is zeroed in the closed-loop transform
    (the DER's slope effectively becomes 0 — pulling V_ref no longer
    moves Q).  See the architectural plan, Risks #5."""

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        # When a DER mapping is provided, derive der_indices from it
        if self.der_mapping is not None:
            object.__setattr__(
                self, "der_indices",
                list(self.der_mapping.sgen_indices),
            )
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
        if self.current_line_max_i_ka is not None:
            if len(self.current_line_max_i_ka) != len(self.current_line_indices):
                raise ValueError(
                    f"current_line_max_i_ka length ({len(self.current_line_max_i_ka)}) "
                    f"must match current_line_indices length "
                    f"({len(self.current_line_indices)})"
                )
        if self.g_qi < 0:
            raise ValueError(f"g_qi must be non-negative, got {self.g_qi}")
        if not (0.0 <= self.lambda_qi <= 1.0):
            raise ValueError(
                f"lambda_qi must be in [0, 1], got {self.lambda_qi}"
            )
        if self.q_integral_max_mvar <= 0:
            raise ValueError(
                f"q_integral_max_mvar must be positive, got "
                f"{self.q_integral_max_mvar}"
            )
        if not (0.0 <= self.gamma_oltc_q <= 1.0):
            raise ValueError(
                f"gamma_oltc_q must be in [0, 1], got {self.gamma_oltc_q}"
            )
        if self.v_setpoints_pu is not None:
            if len(self.v_setpoints_pu) != len(self.voltage_bus_indices):
                raise ValueError(
                    f"v_setpoints_pu length ({len(self.v_setpoints_pu)}) "
                    f"must match voltage_bus_indices length "
                    f"({len(self.voltage_bus_indices)})"
                )
        # ── Grid-forming gen field consistency ──────────────────────────────
        n_gf = len(self.gridforming_gen_indices)
        if len(self.gridforming_gen_buses) != n_gf:
            raise ValueError(
                f"gridforming_gen_buses length "
                f"({len(self.gridforming_gen_buses)}) must match "
                f"gridforming_gen_indices length ({n_gf})"
            )
        if len(self.gridforming_gen_sn_mva) != n_gf:
            raise ValueError(
                f"gridforming_gen_sn_mva length "
                f"({len(self.gridforming_gen_sn_mva)}) must match "
                f"gridforming_gen_indices length ({n_gf})"
            )
        if self.gridforming_gen_op_diagrams and \
                len(self.gridforming_gen_op_diagrams) != n_gf:
            raise ValueError(
                f"gridforming_gen_op_diagrams length "
                f"({len(self.gridforming_gen_op_diagrams)}) must match "
                f"gridforming_gen_indices length ({n_gf})"
            )
        if not self.gridforming_gen_op_diagrams and n_gf > 0:
            self.gridforming_gen_op_diagrams = ["STATCOM"] * n_gf
        if self.gridforming_vm_min_pu >= self.gridforming_vm_max_pu:
            raise ValueError(
                f"gridforming_vm_min_pu ({self.gridforming_vm_min_pu}) must "
                f"be less than gridforming_vm_max_pu "
                f"({self.gridforming_vm_max_pu})"
            )
        if self.rho_q_gridforming < 0:
            raise ValueError(
                f"rho_q_gridforming must be non-negative, got "
                f"{self.rho_q_gridforming}"
            )



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

        # Integral Q-error accumulator (leaky integrator for PI-like behaviour)
        self._q_error_integral = np.zeros(n_interfaces)

        # Shunt bound overrides from Reserve Observer.
        # Keys: shunt index (0-based within shunt vector).
        # Values: (lower, upper) bound override for that shunt.
        # Cleared after each step.
        self._shunt_bound_overrides: Dict[int, tuple] = {}

        # Cache the sensitivity matrix structure
        self._H_cache: Optional[NDArray[np.float64]] = None
        self._H_mappings: Optional[Dict] = None
        self._sensitivity_updater: Optional[SensitivityUpdater] = None

        # Last absolute interface-Q capability bounds (Mvar) reported upward
        # via :meth:`generate_capability_message`.  Stored for live-plot
        # diagnostics so the cascade plotter can show the envelope the TSO
        # solver was given.  ``None`` until the first capability message is
        # generated; one entry per ``interface_trafo_indices`` thereafter,
        # ordered to match.
        self._last_capability_q_iface_min_mvar: Optional[NDArray[np.float64]] = None
        self._last_capability_q_iface_max_mvar: Optional[NDArray[np.float64]] = None

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
    
    def reset_integral(self) -> None:
        """Reset the Q-error integral accumulator to zero."""
        self._q_error_integral[:] = 0.0

    def receive_disturbance_message(
        self,
        message: ShuntDisturbanceMessage,
    ) -> None:
        """Handle a TSO-owned shunt step change inside this DSO's network.

        For each shunt in the message, the cached reduced Jacobian inverse
        is updated via a rank-1 Sherman-Morrison correction at the shunt
        bus.  The cached operating point ``(V, θ)`` is preserved — no
        ``pp.runpp`` is called and no new measurement is taken.

        After the update(s), the H cache is invalidated so the next
        ``step(measurement)`` call rebuilds H from the updated
        ``dV_dQ_reduced``.  This refreshes the OLTC sensitivities of the
        3-winding transformer whose tertiary hosts the shunt with the
        correct shunt-coupling term — restoring an otherwise stale
        ``∂Q_HV-3W / ∂s_OLTC`` column.

        The DSO never sees the shunt as a *control variable*; this is
        purely a model-state refresh path.

        Parameters
        ----------
        message : ShuntDisturbanceMessage
            Message from the supervising TSO controller listing the
            shunts whose step has just changed and their new states.
        """
        if message.target_controller_id != self.controller_id:
            raise ValueError(
                f"Disturbance message target "
                f"'{message.target_controller_id}' does not match "
                f"controller ID '{self.controller_id}'"
            )
        any_applied = False
        for bus, step in zip(message.shunt_bus_indices, message.shunt_steps):
            applied = self.sensitivities.apply_shunt_step_change_smw(
                int(bus), int(step),
            )
            any_applied = any_applied or applied
        if any_applied:
            # Drop H caches so the next step(measurement) rebuilds H from the
            # SMW-updated dV_dQ_reduced.  The cached operating point (V, θ)
            # in self.sensitivities.net is preserved — no pp.runpp call.
            self.invalidate_sensitivity_cache()

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

        # Extract dQ_interface/dQ_DER from H matrix (expanded to per-DER)
        # The first n_interfaces rows of H correspond to Q_interface outputs
        n_interfaces = len(self.config.interface_trafo_indices)
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)

        H_expanded = self._expand_H_to_der_level(self._H_cache)
        dQ_interface_dQ_der = H_expanded[:n_interfaces, :n_der]
        
        # Map DER capability to interface capability using delta from current
        # operating point.  The TSO applies these as:
        #     u_bound = q_interface_current + q_cap
        # so q_cap must be a delta: S · (q_der_bound - q_der_current).
        q_der_current = measurement.der_q_mvar.copy()
        q_interface_min = np.zeros(n_interfaces)
        q_interface_max = np.zeros(n_interfaces)

        for i in range(n_interfaces):
            for j in range(n_der):
                sensitivity = dQ_interface_dQ_der[i, j]
                dq_min = q_der_min[j] - q_der_current[j]
                dq_max = q_der_max[j] - q_der_current[j]
                if sensitivity >= 0:
                    q_interface_min[i] += sensitivity * dq_min
                    q_interface_max[i] += sensitivity * dq_max
                else:
                    q_interface_min[i] += sensitivity * dq_max
                    q_interface_max[i] += sensitivity * dq_min

        # Cache the absolute interface-Q capability envelope (Mvar) so the
        # cascade live plot can overlay it on the per-PCC Q traces.  The
        # message itself ships deltas; the absolute bound the TSO will use
        # is q_iface_now + delta, so we store that form.
        q_iface_now = (
            measurement.interface_q_hv_side_mvar.copy()
            if measurement.interface_q_hv_side_mvar is not None
            else np.zeros(n_interfaces, dtype=np.float64)
        )
        self._last_capability_q_iface_min_mvar = q_iface_now + q_interface_min
        self._last_capability_q_iface_max_mvar = q_iface_now + q_interface_max

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

    def get_interface_der_sensitivity(self) -> NDArray[np.float64]:
        """
        Return the ∂Q_interface / ∂Q_DER sub-matrix, shape (n_interfaces, n_der).

        Builds the full H matrix on first call if not yet cached. Subsequent
        calls return the cached result without recomputation.

        When a DER mapping is active, the returned matrix has per-DER
        columns (expanded via H_bus @ E).
        """
        if self._H_cache is None:
            self._build_sensitivity_matrix()
        n_interfaces = len(self.config.interface_trafo_indices)
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        H_expanded = self._expand_H_to_der_level(self._H_cache)
        return H_expanded[:n_interfaces, :n_der]

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
        """Define the control variable structure.

        Ordering: [ Q_DER | V_gf | s_OLTC | s_shunt ]
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        n_continuous = n_der + n_gf
        n_integer = n_oltc + n_shunt

        # Integer indices are after the continuous DER + V_gf block
        integer_indices = list(range(n_continuous, n_continuous + n_integer))

        return n_continuous, n_integer, integer_indices

    def _get_oltc_integer_indices(self) -> List[int]:
        """OLTC slice within the integer block.

        DSO ordering ``[Q_DER | V_gf | s_OLTC | s_shunt]`` puts the OLTC
        integers immediately after the continuous block.  Used by
        :class:`BaseOFOController` to scope the wall-clock cooldown to
        OLTC indices only.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        return list(range(n_der + n_gf, n_der + n_gf + n_oltc))

    def _actuator_class_indices(self) -> Dict[str, NDArray[np.int64]]:
        """Per-class index map for adaptive ``g_w`` (paper Eq. 16).

        Class names match the BO dimension naming in
        ``tuning/parameters.py``: ``"dso_der"``, ``"dso_grid_forming"``,
        ``"dso_oltc"``, ``"dso_shunt"``.  Empty classes are dropped
        from the map.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        gf_start = n_der
        oltc_start = n_der + n_gf
        oltc_end = oltc_start + n_oltc
        shunt_end = oltc_end + n_shunt

        out: Dict[str, NDArray[np.int64]] = {}
        if n_der > 0:
            # Stage-2 V_ref mode uses different units (pu_v vs Mvar) so
            # the g_w_adapter must learn its step-size scale from
            # scratch — rename the class so any pickled BO state is
            # treated as fresh.
            der_class_name = (
                "dso_der_vref" if self.config.ofo_in_v_ref_mode else "dso_der"
            )
            out[der_class_name] = np.arange(0, n_der, dtype=np.int64)
        if n_gf > 0:
            out["dso_grid_forming"] = np.arange(gf_start, oltc_start, dtype=np.int64)
        if n_oltc > 0:
            out["dso_oltc"] = np.arange(oltc_start, oltc_end, dtype=np.int64)
        if n_shunt > 0:
            out["dso_shunt"] = np.arange(oltc_end, shunt_end, dtype=np.int64)
        return out
    
    def _extract_control_values(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract current control values from measurements.

        Stage 2 (``use_qv_local_loop=True``): the DER block reads V_ref
        from ``measurement.der_vm_pu_ref`` instead of Q from
        ``measurement.der_q_mvar``.  The plant-side
        :class:`controller.dso_qv_local_loop.QVLocalLoop` converges Q on
        each PF given the OFO-commanded V_ref.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_total = n_der + n_gf + n_oltc + n_shunt

        u = np.zeros(n_total)
        v_ref_mode = self.config.ofo_in_v_ref_mode

        # DER block: Q (legacy) or V_ref (Stage 2)
        if v_ref_mode:
            der_iter = (
                mapping.sgen_indices if mapping is not None
                else self.config.der_indices
            )
            for i, sgen_idx in enumerate(der_iter):
                meas_idx = np.where(measurement.der_indices == sgen_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"DER sgen {sgen_idx} not found in measurement"
                    )
                if len(measurement.der_vm_pu_ref) == 0:
                    raise ValueError(
                        "use_qv_local_loop=True but measurement.der_vm_pu_ref "
                        "is empty; check that measure_zone_dso/measure_dso "
                        "populated it from net.sgen.vm_pu_ref."
                    )
                u[i] = float(measurement.der_vm_pu_ref[meas_idx[0]])
        elif mapping is not None:
            for i, sgen_idx in enumerate(mapping.sgen_indices):
                meas_idx = np.where(measurement.der_indices == sgen_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"DER sgen {sgen_idx} not found in measurement"
                    )
                u[i] = measurement.der_q_mvar[meas_idx[0]]
        else:
            for i, der_idx in enumerate(self.config.der_indices):
                meas_idx = np.where(measurement.der_indices == der_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(f"DER {der_idx} not found in measurement")
                u[i] = measurement.der_q_mvar[meas_idx[0]]

        # Grid-forming converter gen voltage setpoints (V_gf)
        for i, gf_idx in enumerate(self.config.gridforming_gen_indices):
            meas_idx = np.where(
                measurement.gridforming_gen_indices == gf_idx
            )[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Grid-forming gen {gf_idx} not found in "
                    f"measurement.gridforming_gen_indices"
                )
            u[n_der + i] = float(
                measurement.gridforming_gen_vm_pu[meas_idx[0]]
            )

        # OLTC tap positions
        for i, oltc_idx in enumerate(self.config.oltc_trafo_indices):
            meas_idx = np.where(measurement.oltc_indices == oltc_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"OLTC {oltc_idx} not found in measurement")
            u[n_der + n_gf + i] = float(
                measurement.oltc_tap_positions[meas_idx[0]]
            )

        # Shunt states
        for i, shunt_idx in enumerate(self.config.shunt_bus_indices):
            meas_idx = np.where(measurement.shunt_indices == shunt_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"Shunt at bus {shunt_idx} not found in measurement")
            u[n_der + n_gf + n_oltc + i] = float(
                measurement.shunt_states[meas_idx[0]]
            )

        return u
    
    def _extract_outputs(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract current output values from measurements.

        Ordering: [ Q_interface | V_bus | I_line | Q_gf | Q_realized_DER ]

        ``Q_realized_DER`` is appended only when
        ``use_qv_local_loop=True`` (Stage 2).
        """
        n_interfaces = len(self.config.interface_trafo_indices)
        n_voltage = len(self.config.voltage_bus_indices)
        n_current = len(self.config.current_line_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        mapping = self.config.der_mapping
        n_der = (
            mapping.n_der if mapping is not None
            else len(self.config.der_indices)
        )
        n_q_real = n_der if self.config.ofo_in_v_ref_mode else 0
        n_outputs = (
            n_interfaces + n_voltage + n_current + n_gf + n_q_real
        )

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

        # Grid-forming converter Q (Q_gf, STATCOM capability output)
        for gf_idx in self.config.gridforming_gen_indices:
            meas_idx = np.where(
                measurement.gridforming_gen_indices == gf_idx
            )[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Grid-forming gen {gf_idx} not found in "
                    f"measurement.gridforming_gen_indices"
                )
            y[idx] = float(measurement.gridforming_gen_q_mvar[meas_idx[0]])
            idx += 1

        # Stage-2: realised Q at each grid-following DER (V_ref-mode only).
        # The plant's Q(V) loop produces this Q in response to OFO V_ref
        # commands; the soft output bound on this row enforces the
        # converter PQ envelope so the OFO eases off V_ref before
        # hitting the rail.
        if self.config.ofo_in_v_ref_mode:
            mapping = self.config.der_mapping
            der_iter = (
                mapping.sgen_indices if mapping is not None
                else self.config.der_indices
            )
            for sgen_idx in der_iter:
                meas_idx = np.where(measurement.der_indices == sgen_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"DER sgen {sgen_idx} not found in measurement"
                    )
                y[idx] = float(measurement.der_q_mvar[meas_idx[0]])
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
        """Compute operating-point-dependent input bounds."""
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_total = n_der + n_gf + n_oltc + n_shunt

        u_lower = np.zeros(n_total)
        u_upper = np.zeros(n_total)

        # DER block: Q-bounds (legacy) or V_ref-bounds (Stage 2)
        if self.config.ofo_in_v_ref_mode:
            u_lower[:n_der] = self.config.qv_v_ref_min_pu
            u_upper[:n_der] = self.config.qv_v_ref_max_pu
        else:
            q_min, q_max = self.actuator_bounds.compute_der_q_bounds(
                der_p_current,
            )
            u_lower[:n_der] = q_min
            u_upper[:n_der] = q_max

        # Grid-forming converter V_gf bounds (fixed physical band)
        gf_start = n_der
        gf_end = gf_start + n_gf
        u_lower[gf_start:gf_end] = self.config.gridforming_vm_min_pu
        u_upper[gf_start:gf_end] = self.config.gridforming_vm_max_pu

        # OLTC tap bounds (fixed)
        tap_min, tap_max = self.actuator_bounds.get_oltc_tap_bounds()
        u_lower[gf_end:gf_end + n_oltc] = tap_min.astype(np.float64)
        u_upper[gf_end:gf_end + n_oltc] = tap_max.astype(np.float64)

        # Shunt state bounds (fixed: -1, 0, +1)
        state_min, state_max = self.actuator_bounds.get_shunt_state_bounds()
        shunt_offset = gf_end + n_oltc
        u_lower[shunt_offset:] = state_min.astype(np.float64)
        u_upper[shunt_offset:] = state_max.astype(np.float64)

        # Apply Reserve Observer overrides
        for j, (lo, hi) in self._shunt_bound_overrides.items():
            u_lower[shunt_offset + j] = lo
            u_upper[shunt_offset + j] = hi

        return u_lower, u_upper
    
    def _get_output_limits(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get output constraint limits.

        Ordering: [ Q_interface | V_bus | I_line | Q_gf | Q_realized_DER ]

        ``Q_realized_DER`` rows are present only when
        ``use_qv_local_loop=True`` (Stage 2).  Their bounds come from
        the per-DER PQ envelope (op_diagram), so the soft output slack
        ``g_z_q_dso_der`` makes the OFO ease off V_ref before pinning
        the converter at its Q rail.
        """
        n_interfaces = len(self.config.interface_trafo_indices)
        n_voltage = len(self.config.voltage_bus_indices)
        n_current = len(self.config.current_line_indices)
        n_gf = len(self.config.gridforming_gen_indices)
        mapping = self.config.der_mapping
        n_der = (
            mapping.n_der if mapping is not None
            else len(self.config.der_indices)
        )
        n_q_real = n_der if self.config.ofo_in_v_ref_mode else 0
        n_outputs = (
            n_interfaces + n_voltage + n_current + n_gf + n_q_real
        )

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

        # Current limits (upper only, kA)
        for j in range(n_current):
            if self.config.current_line_max_i_ka is not None:
                i_lim_ka = self.config.i_max_pu * self.config.current_line_max_i_ka[j]
            else:
                i_lim_ka = 1E6  # no limit if ratings not provided
            y_lower[idx] = 0.0
            y_upper[idx] = i_lim_ka
            idx += 1

        # Q_gf bounds (STATCOM Q-circle, P-dependent — soft via rho_q_gridforming)
        if n_gf > 0:
            meas = getattr(self, "_last_measurement", None)
            if (
                meas is not None
                and len(meas.gridforming_gen_p_mw) == n_gf
            ):
                p_gf = meas.gridforming_gen_p_mw
                for k in range(n_gf):
                    sn = float(self.config.gridforming_gen_sn_mva[k])
                    op_diag = self.config.gridforming_gen_op_diagrams[k]
                    p_ratio = abs(float(p_gf[k])) / sn if sn > 0.0 else 0.0
                    q_min_k, q_max_k = (
                        self.actuator_bounds._compute_single_der_q_capability(
                            p_ratio=p_ratio,
                            s_rated_mva=sn,
                            op_diagram=op_diag,
                        )
                    )
                    y_lower[idx + k] = q_min_k
                    y_upper[idx + k] = q_max_k
            else:
                y_lower[idx:idx + n_gf] = -1e6
                y_upper[idx:idx + n_gf] = +1e6
            idx += n_gf

        # Q_realized_DER bounds (Stage-2 V_ref-mode only).  Per-DER
        # capability box from the op_diagram (STATCOM full circle or
        # VDE-AR-N piecewise linear).  Soft slack with weight
        # ``g_z_q_dso_der`` from the runner makes this a "ease off
        # before the rail" pressure.
        if n_q_real > 0:
            net = self.sensitivities.net
            mapping = self.config.der_mapping
            der_iter = (
                mapping.sgen_indices if mapping is not None
                else self.config.der_indices
            )
            meas = getattr(self, "_last_measurement", None)
            from controller.dso_qv_local_loop import _qv_capability
            for k, sgen_idx in enumerate(der_iter):
                sn = float(net.sgen.at[int(sgen_idx), "sn_mva"])
                if "op_diagram" in net.sgen.columns:
                    od = net.sgen.at[int(sgen_idx), "op_diagram"]
                    op_diag = (
                        str(od) if od is not None and str(od) != "nan"
                        else "VDE-AR-N-4120-v2"
                    )
                else:
                    op_diag = "VDE-AR-N-4120-v2"
                if (
                    meas is not None
                    and len(meas.der_p_mw) > k
                ):
                    p_mw = float(meas.der_p_mw[k])
                else:
                    p_mw = float(net.sgen.at[int(sgen_idx), "p_mw"])
                q_min_k, q_max_k = _qv_capability(sn, op_diag, p_mw)
                y_lower[idx + k] = q_min_k
                y_upper[idx + k] = q_max_k
            idx += n_q_real

        return y_lower, y_upper
    
    def _compute_objective_gradient(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """
        Compute the objective function gradient.

        The DSO objective combines two terms:

        1. **Q-interface tracking** (primary):
           f_q(u) = g_q · ||Q_interface - Q_set||²
           ∇f_q  = 2 · g_q · (Q - Q_set)^T · ∂Q/∂u

        2. **Voltage-schedule tracking** (secondary, optional):
           f_v(u) = g_v · ||V - V_set||²
           ∇f_v  = 2 · g_v · (V - V_set)^T · ∂V/∂u

        **Role-based OLTC attenuation** (``gamma_oltc_q``):
        The Q-tracking gradient on OLTC columns is scaled by γ ∈ [0, 1].
        With γ = 0 (default), OLTCs receive no Q-tracking incentive and
        are driven only by voltage deviations.  The full ∂Q/∂s_OLTC
        remains in H for output constraints, so the physical coupling
        between tap changes and reactive power is preserved.

        The voltage term is only active when *v_setpoints_pu* is
        configured.  It should be weighted softly (small g_v) so that
        Q-interface tracking remains the dominant objective.
        """
        n_total = self.n_controls
        grad_f = np.zeros(n_total)

        # Get sensitivity matrix (bus-level) and expand to per-DER
        H_bus = self._build_sensitivity_matrix()
        H = self._expand_H_to_der_level(H_bus)

        n_interfaces = len(self.config.interface_trafo_indices)
        n_v = len(self.config.voltage_bus_indices)

        # --- Component 1: Q-interface tracking ---
        q_interface = np.zeros(n_interfaces)
        for i, trafo_idx in enumerate(self.config.interface_trafo_indices):
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) > 0:
                q_interface[i] = measurement.interface_q_hv_side_mvar[meas_idx[0]]

        q_error = q_interface - self.q_setpoint_mvar
        dQ_du = H[:n_interfaces, :]

        # Role-based gradient attenuation: attenuate Q-tracking gradient
        # on OLTC columns so that OLTCs are driven primarily by voltage
        # deviations.  The full ∂Q/∂s remains in H for output constraints,
        # preserving the physical coupling in predicted outputs.
        gamma = self.config.gamma_oltc_q
        if gamma < 1.0:
            mapping = self.config.der_mapping
            n_der = mapping.n_der if mapping is not None else len(self.config.der_indices)
            n_oltc = len(self.config.oltc_trafo_indices)
            oltc_slice = slice(n_der, n_der + n_oltc)
            dQ_du_q = dQ_du.copy()
            dQ_du_q[:, oltc_slice] *= gamma
        else:
            dQ_du_q = dQ_du

        grad_f += 2.0 * self.config.g_q * (q_error @ dQ_du_q)

        # --- Integral Q-tracking component (leaky integrator) ---
        if self.config.g_qi > 0.0:
            # Leaky integrator: s_{k+1} = lambda * s_k + e_k
            self._q_error_integral = (
                self.config.lambda_qi * self._q_error_integral + q_error
            )
            # Anti-windup: clamp accumulator
            np.clip(
                self._q_error_integral,
                -self.config.q_integral_max_mvar,
                self.config.q_integral_max_mvar,
                out=self._q_error_integral,
            )
            grad_f += 2.0 * self.config.g_qi * (self._q_error_integral @ dQ_du_q)

        # --- Component 2: Voltage-schedule tracking (optional) ---
        if self.config.v_setpoints_pu is not None:
            v_current = np.zeros(n_v)
            for j, bus_idx in enumerate(self.config.voltage_bus_indices):
                meas_idx = np.where(measurement.bus_indices == bus_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(f"Bus {bus_idx} not found in measurement")
                v_current[j] = measurement.voltage_magnitudes_pu[meas_idx[0]]

            v_error = v_current - self.config.v_setpoints_pu

            # Voltage rows start after interface Q rows in H
            dV_du = H[n_interfaces:n_interfaces + n_v, :]
            grad_f += 2.0 * self.config.g_v * (v_error @ dV_du)

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

        # Map DER indices (sgen) to their corresponding buses for sensitivity
        der_bus_indices = [
            int(net.sgen.at[s, "bus"]) for s in self.config.der_indices
        ]

        # Deduplicate: the sensitivity builder works on unique buses.
        # After building H we expand columns back to one per DER.
        unique_buses: List[int] = []
        der_to_unique: List[int] = []  # maps each DER to its unique-bus column
        for b in der_bus_indices:
            if b not in unique_buses:
                unique_buses.append(b)
            der_to_unique.append(unique_buses.index(b))

        # Build keyword arguments depending on transformer type
        kw = dict(
            der_bus_indices=unique_buses,
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

        # ── Stage-2: Q(V) closed-loop transform on the DER column block ──
        # When the V_ref-mode is active, the DER columns of H represent
        # ∂y/∂Q_DER.  Map them to ∂y/∂V_ref via the closed-loop chain rule
        #     ∂y/∂V_ref = ∂y/∂Q · K (I + S_VQ K)^{-1}
        # where:
        #   * S_VQ is the bus-to-bus DER voltage-to-Q sensitivity (we
        #     query it explicitly because the DER buses are typically not
        #     in voltage_bus_indices).
        #   * K = diag(S_n,i / slope_pu) per unique bus, summed across
        #     DERs sharing a bus.
        # The transform is applied at the bus level, BEFORE the legacy
        # per-DER column duplication, so DERs sharing a bus get the same
        # transformed bus-column (matching the closed-loop invariance).
        # Saturated DERs zero their k contribution to capture the
        # active-set effect (V_ref shift no longer moves Q at the rail).
        T_qv: Optional[NDArray[np.float64]] = None
        if self.config.ofo_in_v_ref_mode and unique_buses:
            T_qv = self._compute_qv_transform_T(unique_buses, der_bus_indices)
            if T_qv is not None:
                n_b = len(unique_buses)
                H = H.copy()
                H[:, :n_b] = H[:, :n_b] @ T_qv
        # Stash T_qv for use by the post-splice Q_realized row block.
        self._qv_T_cache = T_qv

        # When a DER mapping is active, keep H at bus-level (unique buses).
        # The base class _expand_H_to_der_level will handle per-DER expansion
        # via the E matrix.  When no mapping is active, use legacy column
        # duplication so existing code keeps working.
        if self.config.der_mapping is None:
            if len(unique_buses) < len(der_bus_indices):
                n_unique = len(unique_buses)
                n_other = H.shape[1] - n_unique
                H_der = H[:, :n_unique][:, der_to_unique]  # expand
                H_rest = H[:, n_unique:]
                H = np.hstack([H_der, H_rest])
                mappings["der_buses"] = der_bus_indices

        # ── Splice V_gf columns + Q_gf rows for grid-forming converter gens ──
        # Layout shift:
        #   Cols (legacy):   [DER | OLTC | shunt]
        #   Cols (with gf):  [DER | V_gf | OLTC | shunt]
        #   Rows (legacy):   [Q_iface | V | I]
        #   Rows (with gf):  [Q_iface | V | I | Q_gf]
        n_gf_local = len(self.config.gridforming_gen_indices)
        if n_gf_local > 0:
            H = self._splice_gridforming_into_H(
                H_legacy=H,
                der_to_unique=(
                    der_to_unique
                    if self.config.der_mapping is None
                    else None
                ),
                unique_buses=unique_buses,
                der_bus_indices=der_bus_indices,
                iface_are_3w=iface_are_3w,
                oltc_are_3w=oltc_are_3w,
            )

        # ── Stage-2: append Q_realized output rows for V_ref-mode ──────
        # One row per DSO grid-following DER.  ∂Q_realized,i/∂V_ref,j =
        # T_ij (the K-transform); ∂Q_realized,i/∂(other actuator) = 0
        # (cross-coupling via local Q(V) loop deferred — would require
        # multiplying the V_gen / V_gf / OLTC / shunt columns by the
        # bus-voltage sensitivity at the DER bus times -K).  This row
        # block lets the OFO see the converter PQ envelope as a soft
        # output bound, so steepening g_q now actually trades V-tracking
        # against staying within the Q rail rather than running into
        # the rail and stalling.
        if (
            self.config.ofo_in_v_ref_mode
            and T_qv is not None
            and len(self.config.der_indices) > 0
        ):
            n_b = len(unique_buses)
            n_der_total = (
                self.config.der_mapping.n_der
                if self.config.der_mapping is not None
                else len(self.config.der_indices)
            )
            n_total_cols = H.shape[1]
            H_q_real = np.zeros((n_der_total, n_total_cols), dtype=np.float64)
            # T is bus-level (n_b × n_b); each DER inherits its bus row.
            # When der_mapping is active, the per-DER row is just the
            # bus-row repeated; when no mapping (legacy), the same.
            for d_idx, b_pp in enumerate(der_bus_indices):
                b_pos = unique_buses.index(int(b_pp))
                # The DER columns may be already per-DER (legacy
                # duplicated) or per-bus (with mapping).  In both cases
                # the bus-row of T applied to the first n_b columns
                # captures ∂Q_realized,i/∂V_ref,j for j in same bus.
                # Spread the bus-row across the per-DER columns sharing
                # that bus (legacy duplication path).
                if (
                    self.config.der_mapping is None
                    and len(unique_buses) < len(der_bus_indices)
                ):
                    # Duplicated DER columns: T row at bus b_pos applies
                    # to ALL DER columns at that bus.
                    for d_other_idx, b_other_pp in enumerate(der_bus_indices):
                        if int(b_other_pp) == int(b_pp):
                            H_q_real[d_idx, d_other_idx] = T_qv[b_pos, b_pos]
                else:
                    # Per-DER columns are 1:1 with unique buses.
                    H_q_real[d_idx, :n_b] = T_qv[b_pos, :]
            H = np.vstack([H, H_q_real])

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

    def step(
        self,
        measurement: Measurement,
        *,
        sim_time_s: Optional[float] = None,
    ) -> ControllerOutput:
        """
        Execute one OFO iteration with voltage-dependent sensitivity updates.

        Before delegating to :meth:`BaseOFOController.step`, this method
        rescales shunt columns of the cached H matrix using the measured
        bus voltages.  The V² correction accounts for the constant-susceptance
        nature of shunt devices (MSR / MSC).

        After the step, shunt bound overrides from the Reserve Observer
        are cleared so they must be re-set each iteration.

        Parameters
        ----------
        measurement : Measurement
            Current system measurements.
        sim_time_s : float, optional
            Wall-clock simulation time forwarded to
            :meth:`BaseOFOController.step` to drive the wall-clock OLTC
            cooldown (see :attr:`OFOParameters.int_cooldown_s`).
        """
        # Ensure H is built
        if self._H_cache is None:
            self._build_sensitivity_matrix()
        # Update state-dependent shunt columns
        if self._sensitivity_updater is not None:
            self._H_cache = self._sensitivity_updater.update(
                measurement, measurement.iteration
            )
        result = super().step(measurement, sim_time_s=sim_time_s)
        # Clear one-shot overrides
        self._shunt_bound_overrides.clear()
        return result

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

    # =========================================================================
    # Stage-2 Q(V) closed-loop sensitivity transform
    # =========================================================================

    def _compute_qv_transform_T(
        self,
        unique_buses: List[int],
        der_bus_indices: List[int],
    ) -> Optional[NDArray[np.float64]]:
        """Return the bus-level closed-loop Q(V) transform
        ``T = K (I + S_VQ K)^{-1}`` of shape ``(n_unique_bus,
        n_unique_bus)``.

        This is the matrix that maps a per-bus V_ref shift to the
        steady-state Q response of the local Q(V) loop, including the
        network coupling.  Saturated DERs contribute 0 to ``K_diag``
        (active-set: V_ref no longer moves Q at the rail).

        Returns ``None`` if the helper cannot compute S_VQ (e.g. one of
        the DER buses is a PV bus and the reduced Jacobian skips it).
        Callers should fall back to identity in that case.
        """
        n_b = len(unique_buses)
        slope = self.config.qv_slope_pu
        if slope <= 0.0:
            return None

        # K_diag at bus level: sum over DERs at that bus.  Drop saturated
        # DERs from the sum so their column-contribution becomes zero.
        K_diag = np.zeros(n_b, dtype=np.float64)
        net = self.sensitivities.net
        meas = getattr(self, "_last_measurement", None)
        eps = self.config.qv_saturation_eps_mvar
        for d_idx, sgen_idx in enumerate(self.config.der_indices):
            sn = float(net.sgen.at[int(sgen_idx), "sn_mva"])
            saturated = False
            if meas is not None and len(meas.der_q_mvar) > d_idx:
                if "op_diagram" in net.sgen.columns:
                    od = net.sgen.at[int(sgen_idx), "op_diagram"]
                    op_diag = (
                        str(od) if od is not None and str(od) != "nan"
                        else "VDE-AR-N-4120-v2"
                    )
                else:
                    op_diag = "VDE-AR-N-4120-v2"
                from controller.dso_qv_local_loop import _qv_capability
                p_mw = (
                    float(meas.der_p_mw[d_idx])
                    if d_idx < len(meas.der_p_mw) else 0.0
                )
                q_min, q_max = _qv_capability(sn, op_diag, p_mw)
                q_act = float(meas.der_q_mvar[d_idx])
                saturated = (q_act >= q_max - eps) or (q_act <= q_min + eps)
            if not saturated:
                b_pos = unique_buses.index(int(der_bus_indices[d_idx]))
                K_diag[b_pos] += sn / slope

        # Compute S_VQ (bus → bus) at the DER buses.
        try:
            S_VQ_full, obs_map, der_map = (
                self.sensitivities.compute_dV_dQ_der(
                    der_bus_indices=unique_buses,
                    observation_bus_indices=unique_buses,
                )
            )
        except (ValueError, KeyError):
            return None

        # Reorder the helper's output to match ``unique_buses``.
        if obs_map != unique_buses or der_map != unique_buses:
            try:
                obs_perm = [obs_map.index(b) for b in unique_buses]
                der_perm = [der_map.index(b) for b in unique_buses]
                S_VQ = S_VQ_full[np.ix_(obs_perm, der_perm)]
            except ValueError:
                return None
        else:
            S_VQ = S_VQ_full
        # Units fix: ``compute_dV_dQ_der`` returns dV_pu / dQ_pu where
        # Q_pu uses the system base ``net.sn_mva`` (typically 100 MVA).
        # The K matrix below is in Mvar/pu_v (k = S_n/slope), so S_VQ
        # must also be in pu_v/Mvar for the product k·S_VQ to be
        # dimensionless.  Divide by S_base.
        s_base = float(getattr(net, "sn_mva", 1.0))
        if s_base > 0:
            S_VQ = S_VQ / s_base

        K = np.diag(K_diag)
        M = np.eye(n_b) + S_VQ @ K
        # Closed-loop sensitivity:  ∂Q/∂V_ref = K (I + S_VQ K)^{-1} = K M^{-1}.
        # Solve M^T x = K^T column-wise and transpose: x = M^{-T} K^T,
        # so x^T = K M^{-1} = T.  Avoids an explicit inverse.
        try:
            return np.linalg.solve(M.T, K.T).T
        except np.linalg.LinAlgError:
            return None

    def _apply_qv_closed_loop_transform(
        self,
        H_legacy: NDArray[np.float64],
        unique_buses: List[int],
        der_bus_indices: List[int],
    ) -> NDArray[np.float64]:
        """Post-multiply the DER bus-columns of ``H_legacy`` by the
        closed-loop Q(V) transform ``T = K (I + S_VQ K)^{-1}``.

        Wraps :meth:`_compute_qv_transform_T`.  Falls back to the
        legacy H when T cannot be computed (PV bus, singular matrix).
        """
        T = self._compute_qv_transform_T(unique_buses, der_bus_indices)
        if T is None:
            return H_legacy
        n_b = len(unique_buses)
        H = H_legacy.copy()
        H[:, :n_b] = H_legacy[:, :n_b] @ T
        return H

    # =========================================================================
    # Grid-forming converter (V_gf / Q_gf) splicing helper
    # =========================================================================

    def _splice_gridforming_into_H(
        self,
        H_legacy: NDArray[np.float64],
        der_to_unique: Optional[List[int]],
        unique_buses: List[int],
        der_bus_indices: List[int],
        iface_are_3w: bool,
        oltc_are_3w: bool,
    ) -> NDArray[np.float64]:
        """Insert V_gf columns between DER and OLTC blocks, and Q_gf rows
        after the I block, into the legacy DSO sensitivity matrix.

        Parameters mirror those built up in :meth:`_build_sensitivity_matrix`
        before this helper is called.  The returned matrix has layout::

            Cols: [ DER (n_der) | V_gf (n_gf) | OLTC (n_oltc) | shunt (n_shunt) ]
            Rows: [ Q_iface (n_iface) | V (n_v) | I (n_i) | Q_gf (n_gf) ]

        See class-level docstring for the full controller picture.
        """
        gf_buses = list(self.config.gridforming_gen_buses)
        n_gf = len(gf_buses)
        n_iface = len(self.config.interface_trafo_indices)
        n_v = len(self.config.voltage_bus_indices)
        n_i = len(self.config.current_line_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_legacy_rows = H_legacy.shape[0]
        n_legacy_cols = H_legacy.shape[1]
        n_der = n_legacy_cols - n_oltc - n_shunt

        # ── V_gf column block (shape: n_legacy_rows × n_gf) ────────────
        H_vgf = np.zeros((n_legacy_rows, n_gf), dtype=np.float64)

        # Q_iface × V_gf — only for 3W interface trafos (the standard for HV couplers)
        if iface_are_3w and n_iface > 0:
            try:
                dQiface_dVgf, t3w_map, gf_map = (
                    self.sensitivities.compute_dQtrafo3w_hv_dVgen_matrix(
                        gen_bus_indices_pp=gf_buses,
                        trafo3w_indices=self.config.interface_trafo_indices,
                    )
                )
                for k, gf_bus in enumerate(gf_buses):
                    if gf_bus not in gf_map:
                        continue
                    j_gf = gf_map.index(gf_bus)
                    for i_t3w, t_idx in enumerate(t3w_map):
                        if t_idx not in self.config.interface_trafo_indices:
                            continue
                        i_row = self.config.interface_trafo_indices.index(t_idx)
                        H_vgf[i_row, k] = dQiface_dVgf[i_t3w, j_gf]
            except (AttributeError, ValueError):
                pass

        # V_bus × V_gf
        if n_v > 0:
            dV_dVgf, obs_map, gf_map = (
                self.sensitivities.compute_dV_dVgen_matrix(
                    gen_bus_indices_pp=gf_buses,
                    observation_bus_indices=self.config.voltage_bus_indices,
                )
            )
            for k, gf_bus in enumerate(gf_buses):
                if gf_bus not in gf_map:
                    continue
                j_gf = gf_map.index(gf_bus)
                for i_obs, obs_bus in enumerate(obs_map):
                    i_row = self.config.voltage_bus_indices.index(obs_bus)
                    H_vgf[n_iface + i_row, k] = dV_dVgf[i_obs, j_gf]

        # I_line × V_gf
        if n_i > 0:
            dI_dVgf, line_map, gf_map = (
                self.sensitivities.compute_dI_dVgen_matrix(
                    line_indices=self.config.current_line_indices,
                    gen_bus_indices_pp=gf_buses,
                )
            )
            for k, gf_bus in enumerate(gf_buses):
                if gf_bus not in gf_map:
                    continue
                j_gf = gf_map.index(gf_bus)
                for i_line, l_idx in enumerate(line_map):
                    H_vgf[n_iface + n_v + i_line, k] = dI_dVgf[i_line, j_gf]

        # Splice V_gf columns between DER and OLTC blocks
        H = np.hstack([H_legacy[:, :n_der], H_vgf, H_legacy[:, n_der:]])

        # ── Q_gf row block (shape: n_gf × full new column count) ───────
        n_full_cols = H.shape[1]
        H_qgf = np.zeros((n_gf, n_full_cols), dtype=np.float64)

        # ∂Q_gf/∂Q_DER  (DER columns are at positions [0, n_der))
        # The legacy matrix uses unique_buses for sensitivities; if no
        # mapping is active and DER columns were duplicated, we must
        # compute against unique_buses and re-duplicate.
        if unique_buses:
            dQgf_dQder, _, _ = (
                self.sensitivities.compute_dQgen_dQder_matrix(
                    gen_bus_indices_pp=gf_buses,
                    der_bus_indices=unique_buses,
                )
            )
            if der_to_unique is None:
                # DER mapping path: H DER columns are 1:1 with unique buses
                # (or expanded by the base-class E matrix later). Per-DER
                # expansion happens in _expand_H_to_der_level downstream.
                H_qgf[:, :n_der] = dQgf_dQder
            else:
                # Legacy path: DER columns were duplicated by der_to_unique.
                for d, u_idx in enumerate(der_to_unique):
                    H_qgf[:, d] = dQgf_dQder[:, u_idx]

        # ∂Q_gf/∂V_gf  (V_gf columns are at positions [n_der, n_der + n_gf))
        dQgf_dVgf, _, gf_map_qgf = (
            self.sensitivities.compute_dQgen_dVgen_matrix(
                gen_bus_indices_pp_meas=gf_buses,
                gen_bus_indices_pp_chg=gf_buses,
            )
        )
        for k, gf_bus in enumerate(gf_buses):
            col = n_der + k
            if gf_bus in gf_map_qgf:
                j_chg = gf_map_qgf.index(gf_bus)
                H_qgf[:, col] = dQgf_dVgf[:, j_chg]

        # ∂Q_gf/∂s_OLTC  (OLTC columns are at [n_der + n_gf, n_der + n_gf + n_oltc))
        if n_oltc > 0:
            try:
                if oltc_are_3w:
                    dQgf_dsOltc, _, _ = (
                        self.sensitivities.compute_dQgen_ds_3w_matrix(
                            gen_bus_indices_pp=gf_buses,
                            oltc_trafo3w_indices=self.config.oltc_trafo_indices,
                        )
                    )
                else:
                    dQgf_dsOltc, _, _ = (
                        self.sensitivities.compute_dQgen_ds_2w_matrix(
                            gen_bus_indices_pp=gf_buses,
                            oltc_trafo_indices=self.config.oltc_trafo_indices,
                        )
                    )
                col_oltc_start = n_der + n_gf
                H_qgf[:, col_oltc_start:col_oltc_start + n_oltc] = dQgf_dsOltc
            except (AttributeError, ValueError):
                # Helper missing or numeric failure — leave OLTC columns at 0.
                pass

        # ∂Q_gf/∂s_shunt
        if n_shunt > 0:
            dQgf_dShunt, _, _ = (
                self.sensitivities.compute_dQgen_dQ_shunt_matrix(
                    gen_bus_indices_pp=gf_buses,
                    shunt_bus_indices=self.config.shunt_bus_indices,
                    shunt_q_steps_mvar=self.config.shunt_q_steps_mvar,
                )
            )
            col_sh_start = n_der + n_gf + n_oltc
            H_qgf[:, col_sh_start:col_sh_start + n_shunt] = dQgf_dShunt

        # Append Q_gf rows
        H = np.vstack([H, H_qgf])
        return H
