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
import pandas as pd
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

    qv_slope_pu: float = 0.07
    """Q(V) droop slope [pu_q / pu_v].  Fallback when
    ``net.sgen.qv_slope_pu`` column is missing (legacy networks)."""

    use_qv_local_loop: bool = False
    qv_apply_mode: str = "q_shim"
    apply_qv_h_transform: bool = False
    """Apply the closed-loop sensitivity transform
    ``T' = (I + diag(K) . S_VQ)^{-1}`` to the DER columns of H (and the
    matching q_set input-bound scaling via ``T'_bb``).

    Default ``False`` => bare ``H = dy/dQ_DER`` and physical Q_DER input
    bounds, matching the reference-anchored DER model documented in the
    dissertation (Ch.4 sec.4.6.3): the reanchoring centres the deadband at
    every dispatch, so the commanded q_set is realised one-to-one
    (dQ_DER/dq_set = 1 at the dispatch point) and no closed-loop transform
    is needed.

    Set ``True`` to restore the legacy sloping-segment T' correction
    (e.g. to reproduce results generated before 2026-06-16)."""
    qv_v_ref_min_pu: float = 0.95
    qv_v_ref_max_pu: float = 1.10

    @property
    def ofo_in_v_ref_mode(self) -> bool:
        """Always False under refactor_v2 Q_cor mode (legacy V_ref-direct
        path is dead but its consumers still query this property)."""
        return bool(self.use_qv_local_loop and self.qv_apply_mode == "v_ref")

    qv_saturation_eps_mvar: float = 1.0
    """Tolerance for the active-set saturation detector.  When realized
    Q is within this tolerance of either capability rail, the
    corresponding diagonal of K is zeroed in the closed-loop transform
    (the DER's slope effectively becomes 0 — w-shift no longer moves
    Q).  See the architectural plan, Risks #5."""

    # ── DER actuator: w-shift (vertical shift + V_ref reanchoring) ──────────
    # The DER block of the OFO action vector is the OFO-commanded
    # ``q_set`` (Mvar) at the reanchored V_ref per DER.  The apply step
    # writes ``q_set_mvar`` and reanchors ``qv_vref_anchor_pu``.  The
    # H matrix's DER columns are post-multiplied by
    #     T' = (I + diag(K) · S_VQ)^{-1}
    # (same closed-loop transform as the earlier Q_cor formulation —
    # see :func:`controller.der_qv_local_loop.compute_w_shift_h_transform`).

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

        # Last measurement seen by this controller, used by
        # :meth:`_compute_input_bounds` to derive Q_cor bounds from the
        # current Q_DER measurement and the closed-loop sensitivity.
        # Set at the entry of :meth:`step` and
        # :meth:`generate_capability_message`.
        self._last_measurement: Optional[Measurement] = None

        # Bus-level open-loop interface-Q vs DER-Q sensitivity snapshot,
        # captured BEFORE the Q_cor / V_ref T-transform is applied to
        # ``_H_cache``.  Used by :meth:`generate_capability_message` so
        # the reported envelope is in actual ``ΔQ_DER × ΔQ_iface`` units
        # (capability is bounded by physical DER Q rails, not by Q_cor).
        # ``None`` until :meth:`_build_sensitivity_matrix` runs.
        self._H_iface_der_open_bus: Optional[NDArray[np.float64]] = None
        self._H_iface_der_unique_buses: Optional[List[int]] = None
        self._H_iface_der_to_unique: Optional[List[int]] = None

        # Cached bus-level w-shift closed-loop transform
        # ``T' = (I + diag(K)·S_VQ)^{-1}`` and the corresponding unique
        # DER-bus list.  Populated on each :meth:`_build_sensitivity_matrix`
        # call.  Used by :meth:`_compute_input_bounds` to map physical
        # Q_DER bounds to ``q_set`` bounds (since the actuator commanded
        # by the OFO is ``q_set_mvar`` and ``ΔQ_DER ≈ T'_bb · Δq_set`` at
        # the closed-loop equilibrium) and by
        # :meth:`_apply_w_shift_closedloop_to_non_der` to fold the local
        # Q(V) loop's reaction into the OLTC / shunt columns of H.
        # ``None`` when the transform was not built (e.g. singular ``M``).
        self._T_prime_cache: Optional[NDArray[np.float64]] = None
        self._T_prime_unique_buses: Optional[List[int]] = None
        self._T_prime_K_diag: Optional[NDArray[np.float64]] = None

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
        ``dV_dQ_reduced`` / ``J_inv``.  Through that updated Jacobian factor
        the rebuild refreshes the shunt-coupling term in the 3-winding
        transformer's interface sensitivities — both the DER column
        (``∂Q_HV-3W / ∂Q_DER`` via ``dV_dQ_reduced``) and the OLTC column
        (``∂Q_HV-3W / ∂s_OLTC`` via ``J_inv``), restoring otherwise stale
        columns.

        By design, the operating point ``(V, θ)`` is held fixed: NO power flow
        is run on a switch.  Consequently the operating-point-dependent branch
        partials of the 3W model (``∂Q_HV/∂V_hv``, ``∂g/∂τ``, the ``V_pu²``
        scaling) keep their last full-refresh values; only the linear Jacobian
        coupling is corrected here.  This is the intended first-order behaviour
        (controllers never see the plant) — the susceptance change is the
        dominant term and the rank-1 update captures it exactly.

        For a tertiary that hosts more than one bank (e.g. an MSC and an MSR),
        the message's ``shunt_indices`` disambiguates which device changed so
        the SMW reads the correct ``q_mvar`` / step.

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
        sh_indices = message.shunt_indices
        for k, (bus, step) in enumerate(
            zip(message.shunt_bus_indices, message.shunt_steps)
        ):
            sidx = int(sh_indices[k]) if sh_indices is not None else None
            applied = self.sensitivities.apply_shunt_step_change_smw(
                int(bus), int(step), shunt_idx=sidx,
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
        # Snapshot the measurement so downstream sensitivity / bound
        # computations triggered from this call see the same state.
        self._last_measurement = measurement

        # Get DER P for capability calculation
        der_p = self._extract_der_active_power(measurement)

        # Get DER Q capability bounds
        q_der_min, q_der_max = self.actuator_bounds.compute_der_q_bounds(der_p)
        # Build sensitivity matrix if not cached.  This also populates
        # ``_H_iface_der_open_bus`` -- the open-loop ∂Q_iface/∂Q_DER
        # block snapshot we use below.
        if self._H_cache is None:
            self._build_sensitivity_matrix()

        n_interfaces = len(self.config.interface_trafo_indices)
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)

        # Capability is bounded by the physical DER Q rail (ΔQ_DER), so
        # the sensitivity multiplied by ΔQ_DER must be the OPEN-LOOP
        # ``∂Q_iface/∂Q_DER`` (network-only), NOT the closed-loop
        # ``∂Q_iface/∂Q_cor`` that lives in ``_H_cache`` after the T'
        # transform.  Mixing post-T' sensitivity with ΔQ_DER under-
        # reports the envelope by a factor T'_bb (~0.3-0.5 in practice).
        #
        # Use the bus-level open-loop snapshot captured before T' was
        # applied, then expand to per-DER columns via the same E-matrix
        # / legacy duplication path used elsewhere.  Falls back to the
        # post-T' cache if the snapshot is unavailable (e.g. legacy
        # direct-Q mode where open-loop = closed-loop).
        if (
            self._H_iface_der_open_bus is not None
            and self._H_iface_der_unique_buses is not None
            and self._H_iface_der_to_unique is not None
        ):
            H_iface_open_bus = self._H_iface_der_open_bus
            if mapping is not None:
                # E-mapping: H_open_bus @ E -> per-DER columns
                dQ_interface_dQ_der = H_iface_open_bus @ mapping.E
            else:
                # Legacy duplication: per-DER column inherits its bus column.
                dQ_interface_dQ_der = H_iface_open_bus[
                    :, self._H_iface_der_to_unique
                ]
        else:
            # Fallback path: post-T' cache (legacy / direct-Q mode).
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
    
    def apply_qw_reset(self, measurement: Measurement) -> None:
        """Reset the DER block of ``_u_current`` to the measured Q per DER.

        Under the w-shift actuator (vertical shift + V_ref reanchoring),
        the OFO's effective per-step command is the *increment* sigma;
        the ``q_set`` value commanded to the plant is
        ``q_set = Q_meas + sigma``.  Implementing this as
        ``u_new = u_old + sigma`` (the generic OFO update in
        :meth:`BaseOFOController.step`) requires resetting ``u_old`` to
        ``Q_meas`` at the start of each step.

        Call from the runner immediately before :meth:`step`.  A no-op
        when the DER block is empty or when the DSO is operating in
        the vestigial V_ref-direct mode.
        """
        if self._u_current is None:
            return
        if self.config.ofo_in_v_ref_mode:
            return

        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
            der_iter = list(mapping.sgen_indices)
        else:
            n_der = len(self.config.der_indices)
            der_iter = list(self.config.der_indices)
        if n_der == 0:
            return

        q_measured = np.empty(n_der, dtype=np.float64)
        for i, sgen_idx in enumerate(der_iter):
            meas_idx = np.where(measurement.der_indices == int(sgen_idx))[0]
            if len(meas_idx) == 0:
                # Measurement incomplete — skip reset rather than corrupt state.
                return
            q_measured[i] = float(measurement.der_q_mvar[meas_idx[0]])

        self._u_current[:n_der] = q_measured

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

        Ordering: [ Q_DER | s_OLTC | s_shunt ]
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        n_continuous = n_der
        n_integer = n_oltc + n_shunt

        # Integer indices are after the continuous DER block
        integer_indices = list(range(n_continuous, n_continuous + n_integer))

        return n_continuous, n_integer, integer_indices

    def _get_oltc_integer_indices(self) -> List[int]:
        """OLTC slice within the integer block.

        DSO ordering ``[Q_DER | s_OLTC | s_shunt]`` puts the OLTC
        integers immediately after the continuous block.  Used by
        :class:`BaseOFOController` to scope the wall-clock cooldown to
        OLTC indices only.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        return list(range(n_der, n_der + n_oltc))

    def _actuator_class_indices(self) -> Dict[str, NDArray[np.int64]]:
        """Per-class index map for adaptive ``g_w`` (paper Eq. 16).

        Class names match the BO dimension naming in
        ``tuning/parameters.py``: ``"dso_der"``, ``"dso_oltc"``,
        ``"dso_shunt"``.  Empty classes are dropped from the map.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)

        oltc_start = n_der
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
        if n_oltc > 0:
            out["dso_oltc"] = np.arange(oltc_start, oltc_end, dtype=np.int64)
        if n_shunt > 0:
            out["dso_shunt"] = np.arange(oltc_end, shunt_end, dtype=np.int64)
        return out

    def voltage_curvature_inputs(
        self,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Voltage rows of H and per-bus ``g_v`` for curvature analysis.

        DSO output ordering is ``[ Q_interface | V_bus | ... ]``, so the
        voltage block is rows ``[n_interfaces : n_interfaces + n_v]`` (see
        ``_compute_objective_gradient``).  Returns ``None`` unless a
        voltage schedule is active with non-zero weight: a DSO whose
        objective is dominated by interface-Q tracking has no voltage
        curvature to precondition against, and the *voltage-only*
        preconditioner (this prototype) must skip it rather than mis-scale
        ``g_w`` from a non-objective block.  Preconditioning the DSO's
        interface-Q curvature ``H_Q G_w^{-1} H_Q^T diag(g_q)`` is the
        documented next extension.  See
        :meth:`BaseOFOController.voltage_curvature_inputs`.
        """
        n_v = len(self.config.voltage_bus_indices)
        if n_v == 0:
            return None
        if self.config.v_setpoints_pu is None or float(self.config.g_v) <= 0.0:
            return None
        n_interfaces = len(self.config.interface_trafo_indices)
        H = self._expand_H_to_der_level(self._build_sensitivity_matrix())
        H_v = np.ascontiguousarray(
            H[n_interfaces:n_interfaces + n_v, :], dtype=np.float64,
        )
        g_v_vec = np.full(n_v, float(self.config.g_v), dtype=np.float64)
        return H_v, g_v_vec

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
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_total = n_der + n_oltc + n_shunt

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

        # OLTC tap positions
        for i, oltc_idx in enumerate(self.config.oltc_trafo_indices):
            meas_idx = np.where(measurement.oltc_indices == oltc_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"OLTC {oltc_idx} not found in measurement")
            u[n_der + i] = float(
                measurement.oltc_tap_positions[meas_idx[0]]
            )

        # Shunt states
        for i, shunt_idx in enumerate(self.config.shunt_bus_indices):
            meas_idx = np.where(measurement.shunt_indices == shunt_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(f"Shunt at bus {shunt_idx} not found in measurement")
            u[n_der + n_oltc + i] = float(
                measurement.shunt_states[meas_idx[0]]
            )

        return u
    
    def _extract_outputs(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Extract current output values from measurements.

        Ordering: [ Q_interface | V_bus | I_line | Q_realized_DER ]

        ``Q_realized_DER`` is appended only when
        ``use_qv_local_loop=True`` (Stage 2).
        """
        n_interfaces = len(self.config.interface_trafo_indices)
        n_voltage = len(self.config.voltage_bus_indices)
        n_current = len(self.config.current_line_indices)
        mapping = self.config.der_mapping
        n_der = (
            mapping.n_der if mapping is not None
            else len(self.config.der_indices)
        )
        n_q_real = n_der if self.config.ofo_in_v_ref_mode else 0
        n_outputs = (
            n_interfaces + n_voltage + n_current + n_q_real
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

        # Stage-2: realised Q at each DER (V_ref-mode only).
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
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        n_total = n_der + n_oltc + n_shunt

        u_lower = np.zeros(n_total)
        u_upper = np.zeros(n_total)

        # DER block: V_ref bounds (vestigial Stage-2 path) or q_set bounds
        # (the active w-shift path).
        if self.config.ofo_in_v_ref_mode:
            u_lower[:n_der] = self.config.qv_v_ref_min_pu
            u_upper[:n_der] = self.config.qv_v_ref_max_pu
        else:
            # ``u[:n_der]`` is ``q_set`` (Mvar) commanded into the
            # plant-side QVLocalLoop under the w-shift / V_ref-reanchored
            # mode.  The physical bound is the Q_DER envelope from the
            # converter operating diagram.  Map it to ``q_set`` space
            # via the per-bus closed-loop sensitivity
            # ``T'_bb = ΔQ_DER_b / Δq_set_b`` so the optimiser can
            # actually push Q_DER to its rails:
            #
            #   Q_DER(u_new) ≈ Q_DER_current + T'_bb · (u_new - u_old)
            #   ⇒ u_new_max[j] = u_old[j] + (Q_DER_max[j] - Q_DER_current[j]) / T'_bb
            #
            # Falls back to direct Q_DER bounds when the T' cache is
            # unavailable (first call, or T' was singular).  Saturated
            # DERs end up with ``u_lower[j] ≈ u_upper[j]`` because the
            # difference (Q_max - Q_current) collapses, so the optimiser
            # naturally pins them -- no separate active-set logic needed.
            q_min, q_max = self.actuator_bounds.compute_der_q_bounds(
                der_p_current,
            )

            T_cache = self._T_prime_cache
            T_buses = self._T_prime_unique_buses
            meas = self._last_measurement
            net = self.sensitivities.net
            u_old = self._u_current
            der_indices_list = list(self.config.der_indices)

            if (
                T_cache is not None
                and T_buses is not None
                and meas is not None
                and u_old is not None
                and len(meas.der_q_mvar) >= len(der_indices_list)
            ):
                # Per-DER bus lookup -> T'_bb from cached T_prime.
                u_min_qset = np.empty(n_der, dtype=np.float64)
                u_max_qset = np.empty(n_der, dtype=np.float64)
                # Floor T'_bb away from zero to avoid pathological
                # blow-up when a bus is fully saturated.  0.05 is
                # roughly the smallest non-zero T'_bb seen in the
                # IEEE 39 multi-zone scenario; below that the bound
                # would balloon and the optimiser could over-shoot
                # into nonlinear territory.  ``T'_bb`` is bounded
                # above by 1.0 (passive network), so no upper clip.
                T_floor = 0.05
                for j, sgen_idx in enumerate(der_indices_list):
                    bus_j = int(net.sgen.at[int(sgen_idx), "bus"])
                    try:
                        b_pos = T_buses.index(bus_j)
                    except ValueError:
                        # Bus missing from T' (e.g. PV-classified):
                        # fall back to direct Q_DER bound for this DER.
                        u_min_qset[j] = q_min[j]
                        u_max_qset[j] = q_max[j]
                        continue
                    T_jj = max(float(T_cache[b_pos, b_pos]), T_floor)
                    q_now = float(meas.der_q_mvar[j])
                    u_now = float(u_old[j]) if j < len(u_old) else 0.0
                    u_min_qset[j] = u_now + (q_min[j] - q_now) / T_jj
                    u_max_qset[j] = u_now + (q_max[j] - q_now) / T_jj
                u_lower[:n_der] = u_min_qset
                u_upper[:n_der] = u_max_qset
            else:
                # No T' cache yet (first call, before any sensitivity
                # build).  Use the physical Q_DER envelope as a safe
                # initial bound; the next iteration will tighten with
                # the proper T' scaling.
                u_lower[:n_der] = q_min
                u_upper[:n_der] = q_max

        # OLTC tap bounds (fixed)
        tap_min, tap_max = self.actuator_bounds.get_oltc_tap_bounds()
        u_lower[n_der:n_der + n_oltc] = tap_min.astype(np.float64)
        u_upper[n_der:n_der + n_oltc] = tap_max.astype(np.float64)

        # Shunt state bounds (fixed: -1, 0, +1)
        state_min, state_max = self.actuator_bounds.get_shunt_state_bounds()
        shunt_offset = n_der + n_oltc
        u_lower[shunt_offset:] = state_min.astype(np.float64)
        u_upper[shunt_offset:] = state_max.astype(np.float64)

        # Apply Reserve Observer overrides
        for j, (lo, hi) in self._shunt_bound_overrides.items():
            u_lower[shunt_offset + j] = lo
            u_upper[shunt_offset + j] = hi

        return u_lower, u_upper
    
    def _get_output_limits(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get output constraint limits.

        Ordering: [ Q_interface | V_bus | I_line | Q_realized_DER ]

        ``Q_realized_DER`` rows are present only when
        ``use_qv_local_loop=True`` (Stage 2).  Their bounds come from
        the per-DER PQ envelope (op_diagram), so the soft output slack
        ``g_z_q_dso_der`` makes the OFO ease off V_ref before pinning
        the converter at its Q rail.
        """
        n_interfaces = len(self.config.interface_trafo_indices)
        n_voltage = len(self.config.voltage_bus_indices)
        n_current = len(self.config.current_line_indices)
        mapping = self.config.der_mapping
        n_der = (
            mapping.n_der if mapping is not None
            else len(self.config.der_indices)
        )
        n_q_real = n_der if self.config.ofo_in_v_ref_mode else 0
        n_outputs = (
            n_interfaces + n_voltage + n_current + n_q_real
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
            from controller.der_qv_local_loop import _qv_capability
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

        # ── Open-loop snapshot of the iface-Q vs DER-Q bus block ──────
        # Capture BEFORE any T_qv / T_prime transform is applied so
        # :meth:`generate_capability_message` can compute a physically-
        # correct envelope (capability is bounded by the ΔQ_DER rail,
        # not by ΔQ_cor; mixing post-T' sensitivity with ΔQ_DER under-
        # reports the envelope by a factor T'_bb).
        n_iface_local = len(self.config.interface_trafo_indices)
        n_b_local = len(unique_buses)
        if n_iface_local > 0 and n_b_local > 0:
            self._H_iface_der_open_bus = H[:n_iface_local, :n_b_local].copy()
        else:
            self._H_iface_der_open_bus = None
        self._H_iface_der_unique_buses = list(unique_buses)
        self._H_iface_der_to_unique = list(der_to_unique)

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
        # K_diag includes every in-service DER; saturation is enforced by
        # :meth:`_compute_input_bounds`, not by zeroing K.
        T_qv: Optional[NDArray[np.float64]] = None
        if self.config.ofo_in_v_ref_mode and unique_buses:
            T_qv = self._compute_qv_transform_T(unique_buses, der_bus_indices)
            if T_qv is not None:
                n_b = len(unique_buses)
                H = H.copy()
                H[:, :n_b] = H[:, :n_b] @ T_qv
        # Stash T_qv for use by the post-splice Q_realized row block.
        self._qv_T_cache = T_qv

        # ── w-shift: q_set closed-loop transform on DER columns ────────
        # Under the vertical-shift + V_ref-reanchored DER actuator,
        # post-multiply the DER columns of H by
        #     T' = (I + diag(K) · S_VQ)^{-1}
        # This maps the network-level ``∂y/∂Q`` to ``∂y/∂q_set``.
        # K_diag in T' includes every in-service DER -- saturation is
        # enforced by :meth:`_compute_input_bounds`, which maps the
        # physical Q_DER envelope to q_set bounds via T'_bb.
        if not self.config.ofo_in_v_ref_mode and unique_buses:
            # Gated by ``apply_qv_h_transform`` (default False => bare H).
            # When disabled, T_prime is None so the cache-clearing branch
            # below runs and _compute_input_bounds reverts to physical
            # Q_DER bounds -- matching the dissertation's bare-H model.
            T_prime = (
                self._compute_w_shift_transform_T_prime(
                    unique_buses, der_bus_indices,
                )
                if self.config.apply_qv_h_transform else None
            )
            if T_prime is not None:
                n_b = len(unique_buses)
                # Snapshot the FULL open-loop DER block (all output rows)
                # BEFORE T' is applied, so the closed-loop correction for
                # non-DER columns can use it.
                H_DER_open_full = H[:, :n_b].copy()
                if T_qv is None:
                    H = H.copy()
                # Apply T' to the DER columns: ∂y/∂Q -> ∂y/∂q_set.
                H[:, :n_b] = H[:, :n_b] @ T_prime

                # Closed-loop correction for non-DER columns (OLTC, shunt):
                # when an OLTC tap or shunt step changes, V at DER buses
                # shifts by ΔV_open, the local Q(V) loop responds with
                # ΔQ_DER = -T'·diag(K)·ΔV_open, and that DER reaction
                # propagates back into the network (and Q_iface).  The
                # open-loop H column captures only the network response,
                # not the DER reaction.  Add the DER-reaction term:
                #     ΔH[:, j] = -H_DER_open · T' · diag(K) · ∂V_DER/∂j
                # so that the OFO's predicted ΔQ_iface from a non-DER
                # action matches the actual closed-loop response.
                #
                # NOTE 2026-05-06: temporarily gated by env var while we
                # debug a sign / magnitude issue surfaced by FD validation
                # (V_obs × OLTC entries flip sign and over-shoot by ~600x).
                import os as _os
                if _os.environ.get("DSO_CLOSED_LOOP_OLTC", "0") == "1":
                    self._apply_w_shift_closedloop_to_non_der(
                        H, H_DER_open_full, T_prime,
                        self._T_prime_K_diag, unique_buses,
                        n_b=n_b, oltc_are_3w=oltc_are_3w,
                    )
            else:
                # Fall back: clear the cache so :meth:`_compute_input_bounds`
                # reverts to physical Q_DER bounds without scaling.
                self._T_prime_cache = None
                self._T_prime_unique_buses = None
                self._T_prime_K_diag = None

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

        # ── Stage-2: append Q_realized output rows for V_ref-mode ──────
        # One row per DSO DER.  ∂Q_realized,i/∂V_ref,j =
        # T_ij (the K-transform); ∂Q_realized,i/∂(other actuator) = 0
        # (cross-coupling via local Q(V) loop deferred — would require
        # multiplying the V_gen / OLTC / shunt columns by the
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
        # Snapshot the measurement so :meth:`_compute_input_bounds`
        # (called downstream from ``super().step``) can read the current
        # Q_DER state when mapping the physical Q_DER envelope to Q_cor
        # bounds via T'.
        self._last_measurement = measurement

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
        ``H_bus`` (e.g. after a contingency or topology change), plus
        the open-loop iface-DER snapshot and the Q_cor T_prime cache
        used by :meth:`_compute_input_bounds`.
        """
        self._H_cache = None
        self._H_mappings = None
        self._sensitivity_updater = None
        self._H_der_cache = None
        self._H_der_cache_base_id = None
        self._H_iface_der_open_bus = None
        self._H_iface_der_unique_buses = None
        self._H_iface_der_to_unique = None
        self._T_prime_cache = None
        self._T_prime_unique_buses = None
        self._T_prime_K_diag = None

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
        network coupling.

        ``K_diag`` includes every in-service DER at the bus regardless
        of whether the DER is currently at its Q rail.  Saturation is
        instead enforced by :meth:`_compute_input_bounds`.  This keeps
        the linearised model a clean Jacobian and lets the optimiser
        handle the active set.

        Returns ``None`` if the helper cannot compute S_VQ (e.g. one of
        the DER buses is a PV bus and the reduced Jacobian skips it).
        Callers should fall back to identity in that case.
        """
        n_b = len(unique_buses)
        slope = self.config.qv_slope_pu
        if slope <= 0.0:
            return None

        # K_diag at bus level: sum S_n/slope over every in-service DER
        # at the bus.  No saturation special-case -- bounds handle it.
        K_diag = np.zeros(n_b, dtype=np.float64)
        net = self.sensitivities.net
        for d_idx, sgen_idx in enumerate(self.config.der_indices):
            sn = float(net.sgen.at[int(sgen_idx), "sn_mva"])
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
        # ``compute_dV_dQ_der`` now returns S_VQ in pu_v/Mvar (it divides by
        # ``sn_mva`` internally); K is in Mvar/pu_v, so K·S_VQ is dimensionless
        # — no extra base scaling here.
        K = np.diag(K_diag)
        M = np.eye(n_b) + S_VQ @ K
        # Closed-loop sensitivity:  ∂Q/∂V_ref = K (I + S_VQ K)^{-1} = K M^{-1}.
        # Solve M^T x = K^T column-wise and transpose: x = M^{-T} K^T,
        # so x^T = K M^{-1} = T.  Avoids an explicit inverse.
        try:
            return np.linalg.solve(M.T, K.T).T
        except np.linalg.LinAlgError:
            return None

    def _compute_w_shift_transform_T_prime(
        self,
        unique_buses: List[int],
        der_bus_indices: List[int],
    ) -> Optional[NDArray[np.float64]]:
        """Return the bus-level closed-loop w-shift transform
        ``T' = (I + diag(K) · S_VQ)^{-1}`` of shape ``(n_unique_bus,
        n_unique_bus)``.

        This is the matrix that maps a per-bus ``q_set`` command (Mvar)
        to the steady-state realised Q (Mvar) of the local Q(V) loop,
        including network coupling.  The matrix is structurally
        identical to the earlier Q_cor transform (Soleimani & Van Cutsem,
        eq. 18); under the vertical-shift + V_ref-reanchored formulation
        the differential sensitivity ``∂Q_realised / ∂q_set`` is the
        same ``(I + R·S_VQ)^{-1}``.

        ``K_diag`` includes every in-service DER at the bus regardless
        of whether the DER is currently at its Q rail.  Saturation is
        instead enforced by :meth:`_compute_input_bounds`, which maps
        the physical Q_DER envelope to ``q_set`` input bounds via this
        T'.  That keeps the linearised model a clean Jacobian around
        the operating point and lets the optimiser handle the active
        set.

        Per-DER slope is read from ``net.sgen.qv_slope_pu`` if the column
        is present; otherwise falls back to ``self.config.qv_slope_pu``
        so legacy networks still get a sensible matrix.

        Returns ``None`` if ``S_VQ`` cannot be computed (e.g. one of the
        DER buses is a PV bus and the reduced Jacobian skips it) or
        ``M`` is singular.  Callers should fall back to identity (no
        transform) in that case.
        """
        from controller.der_qv_local_loop import compute_w_shift_h_transform

        n_b = len(unique_buses)
        if n_b == 0:
            return None

        net = self.sensitivities.net

        # Test / mock nets may lack ``sn_mva`` (e.g. the synthetic
        # actuator-bounds fixtures used in test_oltc_cooldown).  Return
        # None so the caller falls back to identity — the controller
        # still works, just without the closed-loop H correction.
        if "sn_mva" not in net.sgen.columns:
            return None

        # K_diag at bus level: sum S_n,i / slope_pu,i over every in-service
        # DER at the bus.  No saturation special-case -- bounds handle it.
        K_diag = np.zeros(n_b, dtype=np.float64)
        sgen_has_slope_col = "qv_slope_pu" in net.sgen.columns
        for d_idx, sgen_idx in enumerate(self.config.der_indices):
            sn = float(net.sgen.at[int(sgen_idx), "sn_mva"])
            if sgen_has_slope_col:
                slope_v = net.sgen.at[int(sgen_idx), "qv_slope_pu"]
                if pd.isna(slope_v) or float(slope_v) <= 0.0:
                    slope = self.config.qv_slope_pu
                else:
                    slope = float(slope_v)
            else:
                slope = self.config.qv_slope_pu
            if slope <= 0.0:
                continue
            b_pos = unique_buses.index(int(der_bus_indices[d_idx]))
            K_diag[b_pos] += sn / slope

        # Bus-to-bus S_VQ at the DER buses (units fix below).
        try:
            S_VQ_full, obs_map, der_map = (
                self.sensitivities.compute_dV_dQ_der(
                    der_bus_indices=unique_buses,
                    observation_bus_indices=unique_buses,
                )
            )
        except (ValueError, KeyError):
            return None

        if obs_map != unique_buses or der_map != unique_buses:
            try:
                obs_perm = [obs_map.index(b) for b in unique_buses]
                der_perm = [der_map.index(b) for b in unique_buses]
                S_VQ = S_VQ_full[np.ix_(obs_perm, der_perm)]
            except ValueError:
                return None
        else:
            S_VQ = S_VQ_full
        # ``compute_dV_dQ_der`` now returns S_VQ in pu_v/Mvar (divides by
        # ``sn_mva`` internally); no extra base scaling here.
        T_prime = compute_w_shift_h_transform(K_diag, S_VQ)
        # Cache for downstream consumers (input-bound calculation in
        # :meth:`_compute_input_bounds`, closed-loop correction for
        # non-DER columns in :meth:`_apply_w_shift_closedloop_to_non_der`).
        # Note: we cache the most recent successful computation; if T'
        # is None, leave the previous cache intact so the bound calc
        # can still operate.
        if T_prime is not None:
            self._T_prime_cache = T_prime
            self._T_prime_unique_buses = list(unique_buses)
            self._T_prime_K_diag = K_diag.copy()
        return T_prime

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

    def _apply_w_shift_closedloop_to_non_der(
        self,
        H: NDArray[np.float64],
        H_DER_open_full: NDArray[np.float64],
        T_prime: NDArray[np.float64],
        K_diag: NDArray[np.float64],
        unique_buses: List[int],
        *,
        n_b: int,
        oltc_are_3w: bool,
    ) -> None:
        """Apply the local-Q(V)-loop closed-loop correction to the
        non-DER columns (OLTC, shunt) of the OFO sensitivity matrix.

        Under the w-shift actuator, the DER columns of H have already
        been post-multiplied by ``T'`` to express ``∂y/∂q_set`` instead
        of ``∂y/∂Q``.  The non-DER columns (OLTC tap, shunt step) still
        carry their open-loop sensitivity, but in reality those actions
        also perturb V at the DER buses, the local Q(V) loops respond,
        and the resulting ΔQ_DER feeds back into Q_iface and the rest of
        ``y``.  Add that DER-reaction term:

        .. math::
            \\Delta H_{:,j} = -H_{\\rm DER,open} \\cdot T' \\cdot
                \\mathrm{diag}(K) \\cdot \\frac{\\partial V_{\\rm DER}}{\\partial j}

        The matrix :math:`H_{\\rm DER,open}` is the **pre-T'** snapshot of
        the DER bus-column block (every output row), captured by the caller
        right before :meth:`_compute_w_shift_transform_T_prime` was applied.
        Modifies ``H`` in place.
        """
        n_oltc = len(self.config.oltc_trafo_indices)
        n_shunt = len(self.config.shunt_bus_indices)
        if n_oltc == 0 and n_shunt == 0:
            return
        if K_diag is None or T_prime is None:
            return

        W = self._compute_dV_DER_per_non_der_action(
            unique_buses=unique_buses,
            oltc_are_3w=oltc_are_3w,
        )
        if W is None or W.shape[1] == 0:
            return

        # Correction = -H_DER_open · T' · diag(K) · W
        T_K = T_prime * K_diag.reshape(1, n_b)  # (n_b, n_b) -- right-mult by diag
        correction = -H_DER_open_full @ T_K @ W
        n_non_der = correction.shape[1]
        # The non-DER columns sit immediately after the DER block in H.
        # Layout at this point: [DER (n_b) | OLTC (n_oltc) | shunt (n_shunt)].
        H[:, n_b:n_b + n_non_der] += correction

    def _compute_dV_DER_per_non_der_action(
        self,
        unique_buses: List[int],
        oltc_are_3w: bool,
    ) -> Optional[NDArray[np.float64]]:
        """Return ``∂V_at_unique_DER_buses / ∂non_DER_actuator`` of shape
        ``(n_unique_bus, n_oltc + n_shunt)``.

        Concatenates OLTC and shunt columns in the same order they appear
        in the H matrix's non-DER block.  Falls back to zero columns for
        any actuator whose helper raises (e.g. shunt bus not in the
        reduced Jacobian).
        """
        n_b = len(unique_buses)
        if n_b == 0:
            return None

        blocks: List[NDArray[np.float64]] = []

        # ── OLTC block ─────────────────────────────────────────────────
        oltc_indices = list(self.config.oltc_trafo_indices)
        n_oltc = len(oltc_indices)
        if n_oltc > 0:
            try:
                if oltc_are_3w:
                    W_oltc, obs_map, t_map = (
                        self.sensitivities.compute_dV_ds_trafo3w_matrix(
                            trafo3w_indices=oltc_indices,
                            observation_bus_indices=unique_buses,
                        )
                    )
                else:
                    W_oltc, obs_map, t_map = (
                        self.sensitivities.compute_dV_ds_2w_matrix(
                            trafo_indices=oltc_indices,
                            observation_bus_indices=unique_buses,
                        )
                    )
                # Reorder rows to match unique_buses if helper returned
                # a different ordering.
                if obs_map and obs_map != unique_buses:
                    try:
                        perm = [obs_map.index(b) for b in unique_buses]
                        W_oltc = W_oltc[perm, :]
                    except ValueError:
                        W_oltc = np.zeros((n_b, n_oltc))
                # Pad columns if the helper dropped any OLTCs from t_map.
                if t_map and t_map != oltc_indices:
                    full = np.zeros((n_b, n_oltc), dtype=np.float64)
                    for j_full, t in enumerate(oltc_indices):
                        if t in t_map:
                            full[:, j_full] = W_oltc[:, t_map.index(t)]
                    W_oltc = full
                blocks.append(W_oltc)
            except (ValueError, AttributeError):
                blocks.append(np.zeros((n_b, n_oltc)))

        # ── Shunt block ────────────────────────────────────────────────
        shunt_buses = list(self.config.shunt_bus_indices)
        shunt_steps = list(self.config.shunt_q_steps_mvar)
        n_shunt = len(shunt_buses)
        if n_shunt > 0:
            W_shunt = np.zeros((n_b, n_shunt), dtype=np.float64)
            for k, sb in enumerate(shunt_buses):
                q_step = (
                    float(shunt_steps[k]) if k < len(shunt_steps) else 1.0
                )
                try:
                    col, obs_map = self.sensitivities.compute_dV_dQ_shunt(
                        shunt_bus_idx=int(sb),
                        observation_bus_indices=unique_buses,
                        q_step_mvar=q_step,
                    )
                    if obs_map and obs_map != unique_buses:
                        try:
                            perm = [obs_map.index(b) for b in unique_buses]
                            col = col[perm]
                        except ValueError:
                            col = np.zeros(n_b, dtype=np.float64)
                    W_shunt[:, k] = col
                except (ValueError, AttributeError, IndexError):
                    pass
            blocks.append(W_shunt)

        if not blocks:
            return None
        return np.hstack(blocks)
