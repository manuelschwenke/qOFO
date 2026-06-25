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
from typing import Optional, List, Set, Tuple, Dict
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
    shunt_sensitivity_bus_indices: Optional[List[int]] = None
    """Optional remapping of shunt bus indices for the *sensitivity*
    Jacobian only.  When ``None`` (default), the same indices in
    ``shunt_bus_indices`` are used both for the apply path
    (writing ``net.shunt.step``) and for ``JacobianSensitivities``
    lookups.  Under the local-network sensitivity option
    (``MultiTSOConfig.local_sensitivities_tso=True``) the runner sets
    this field to the synthetic 3W primary bus indices, so the per-zone
    reduced Jacobian's shunt column is built at the kept boundary bus
    rather than at the (dropped) tertiary bus.  Same length as
    ``shunt_bus_indices``."""

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
    gen_vm_min_pu: float = 0.95
    gen_vm_max_pu: float = 1.07

    gen_oltc_map: Dict[int, int] = field(default_factory=dict)
    """Maps generator position index (in gen_indices) to OLTC position
    index (in oltc_trafo_indices).  Only generators with a dedicated
    machine transformer are included.  Generators without an entry are
    ignored for capability-based OLTC blocking.  Example: {0: 2} means
    gen_indices[0] is connected to oltc_trafo_indices[2]."""

    rho_q_gen: float = 1e2
    """Soft-constraint penalty for Q_gen output violations in the MIQP
    objective (z^T G_z z contribution).  Q_gen is an operating-point-
    dependent capability output: bounds come from the generator PQ
    capability curve each iteration.  Order-of-magnitude guidance:
    set comparable to ``gz_tso_current`` (default 1e3) so Q_gen and
    current limits compete on similar footing with DER-Q changes."""

    sat_eps_enter_mvar: float = 5.0
    """Enter-saturation margin [Mvar]: a generator flips to saturated
    mode when its measured Q reaches ``Q_lim ∓ sat_eps_enter_mvar``.
    Small enter margin = quick response to saturation."""

    sat_eps_exit_mvar: float = 25.0
    """Exit-saturation margin [Mvar]: a saturated generator flips back
    to free mode only when its Q has retreated to
    ``Q_lim ∓ sat_eps_exit_mvar``.  Must be greater than
    ``sat_eps_enter_mvar`` to enforce hysteresis (no mode chatter)."""

    enable_saturation_mode: bool = False
    """Master gate for Feature B (hysteretic saturation classifier +
    asymmetric V_gen bound clamp + PQ-mode column zeroing).  When False
    (default), V_gen behaves as a plain continuous control; the cached
    ``_sat_mode`` stays zero and the saturation code paths short-circuit.
    Set True to reproduce the saturation-aware experiment."""

    k_t_avt: float = 0.0
    """Achieved-Value Tracking factor for PCC-Q reset.
    0.0 = no reset (current behaviour), 1.0 = full reset (recommended).
    Blends: u_pcc <- (1 - k_t) * u_cmd + k_t * q_measured."""

    der_mapping: Optional[DERMapping] = None
    """Per-DER mapping for individual sgen-level control.  When
    provided, enables per-DER decision variables in the MIQP
    and factorises the sensitivity matrix as H_der = H_bus @ E.
    If None, the controller uses the legacy sgen-index-based control."""

    g_q_tso: float = 1.0
    """Q_PCC tracking weight when the (re-enabled) Q_PCC output rows
    are present.  Scales the gradient contribution
    ``2 · g_q_tso · (Q_PCC_meas − Q_PCC,set)^T · ∂Q_PCC/∂u``.  Default
    1.0 (small) — the TSO mildly prefers to cancel shunt-induced Q
    displacement at the EHV–DSO interface via Q_PCC,set adjustments
    rather than overload the DSO.  Set 0.0 to keep the rows
    informational only (legacy behaviour)."""

    pcc_capability_on_output: bool = True
    """If True, the DSO-reported PCC capability bound is enforced on
    the *output* Q_PCC (soft slack with weight ``g_z_q_pcc``) and the
    control variable Q_PCC,set is given a wide engineering band.  If
    False (legacy), the same bound is enforced as a hard input bound
    on the control Q_PCC,set."""

    tie_line_indices: List[int] = field(default_factory=list)
    """Pandapower line indices for tie lines monitored by this zone
    (Phase A: monitored only).  A tie line is one whose two endpoints
    sit in two different TSO zones.  Each entry corresponds — by
    position — to an entry in ``tie_line_endpoint_buses``."""

    tie_line_endpoint_buses: List[int] = field(default_factory=list)
    """Pandapower bus indices identifying which endpoint of each tie line
    is INSIDE this zone.  Sign convention for the corresponding Q_tie row
    of H: positive Q_tie = reactive power flowing from this endpoint
    INTO the line (i.e. leaving the zone).  Same length as
    ``tie_line_indices``."""

    q_tie_setpoints_mvar: Optional[NDArray[np.float64]] = None
    """Tracking setpoints for tie-line Q [Mvar] (length = len(tie_line_indices)).
    The Phase B gradient term penalises ``(Q_tie_meas - q_tie_setpoints_mvar)^2``.
    Default ``None`` materialises as a zero vector at construction time."""

    g_q_tie: float = 0.0
    """Q_tie tracking weight.  Scales the gradient contribution
    ``2 * g_q_tie * (Q_tie_meas - Q_tie_set)^T * dQ_tie/du``.  Default 0.0
    keeps Q_tie informational only (Phase A behaviour: rows present in H,
    no objective contribution).  Mirrors ``g_q_tso`` for Q_PCC tracking;
    1.0 is a safe Phase-B starting point — see
    :class:`configs.multi_tso_config.MultiTSOConfig.tso_g_q_tie` for the
    full tuning sweep."""

    g_z_q_tie: float = 0.0
    """Soft-constraint slack penalty for Q_tie output bound (Phase B
    optional).  Mirrors ``g_z_q_pcc``.  Default 0.0 = no slack-based
    bound enforcement on Q_tie."""

    # ── Explicit reactive-reserve penalisation (optional, togglable) ────────
    g_res_sg: float = 0.0
    """Explicit reserve-penalisation weight for transmission-connected
    SYNCHRONOUS GENERATORS.  Adds the term ``g_res_sg · Σ_i (r_SG,i)²`` to
    the TSO objective, where
    ``r_SG,i = (Q_gen,i − Q_mid,i) / Q_half,i`` is the *normalised* distance
    of generator ``i``'s reactive output from the midpoint of its
    state-dependent PQ-capability band ``[Q_min,i, Q_max,i]``
    (``Q_mid = ½(Q_min+Q_max)``, ``Q_half = ½(Q_max−Q_min)``).  Minimising
    the term keeps synchronous machines centred in their capability band,
    preserving symmetric reactive reserve in both directions.  Q_gen is an
    *output*, so the gradient is mapped to control space via the
    sensitivities: ``2 · g_res_sg · (r_SG / Q_half)^T · ∂Q_gen/∂u``.
    Default ``0.0`` keeps the term out of the objective entirely (reserve is
    then minimised only *implicitly*, via the DSO cascade).  Toggle pattern
    mirrors ``g_q_tie``.  See
    :class:`configs.multi_tso_config.MultiTSOConfig.tso_g_res_sg`."""

    g_res_der: float = 0.0
    """Explicit reserve-penalisation weight for transmission-connected DER
    (continuous, Q-controlled sgens).  Adds ``g_res_der · Σ_i (r_DER,i)²``
    with ``r_DER,i = (Q_DER,i − Q_mid,i) / Q_half,i`` over each DER's
    VDE-AR-N-4120 capability band.  Because ``Q_DER`` is a *direct* control
    variable, the gradient lands straight on the DER control columns:
    ``2 · g_res_der · (Q_DER − Q_mid) / Q_half²``.  Default ``0.0`` (off).
    A weight separate from ``g_res_sg`` lets the operator prefer one resource
    class over the other.  NOTE: DSO-connected DER reserve is intentionally
    NOT penalised here — that belongs to the DSO (Layer-2) controllers.  See
    :class:`configs.multi_tso_config.MultiTSOConfig.tso_g_res_der`."""

    # ── DER actuator: w-shift (vertical shift + V_ref reanchoring) ──────────
    # The DER block of the TSO action vector is the OFO-commanded
    # ``q_set`` (Mvar) at the reanchored V_ref of each DER.  The apply
    # step (``apply_zone_tso_controls``) writes the value to
    # ``net.sgen.q_set_mvar`` and reanchors ``qv_vref_anchor_pu`` to
    # the most recent measured bus voltage.  The H matrix's DER columns
    # are post-multiplied by the closed-loop transform
    #
    #     T' = (I + diag(K) · S_VQ)^{-1}
    #
    # where ``K_b = Σ_i S_n,i / slope_pu,i`` per unique TSO DER bus
    # (Soleimani & Van Cutsem, eq. 18 — same matrix as the earlier
    # Q_cor formulation, see :func:`controller.der_qv_local_loop.
    # compute_w_shift_h_transform`).  Saturated DERs zero their K_diag
    # so the corresponding column of ``T'`` collapses to identity.

    qv_slope_pu: float = 0.07
    """Default Q(V) droop slope for TSO DERs (pu_q/pu_v) used when the
    ``net.sgen.qv_slope_pu`` column is missing (legacy networks).  When
    the column is present the per-DER value is read from there and this
    field is ignored."""

    apply_qv_h_transform: bool = False
    """Apply the closed-loop sensitivity transform
    ``T' = (I + diag(K) . S_VQ)^{-1}`` to the DER columns of H.

    Default ``False`` => bare ``H = dy/dQ_DER``, matching the
    reference-anchored DER model documented in the dissertation
    (Ch.4 sec.4.6.3): the reanchoring centres the deadband at every
    dispatch, so the commanded q_set is realised one-to-one and no
    closed-loop transform is needed.  Set ``True`` to restore the legacy
    sloping-segment T' correction (mirrors the DSO controller)."""

    qv_saturation_eps_mvar: float = 1.0
    """Tolerance for the active-set saturation detector.  When realised
    Q is within this tolerance of either capability rail, the
    corresponding diagonal of K is zeroed in the closed-loop transform
    (DER's slope effectively becomes 0 — w-shift no longer moves Q)."""

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
        if self.rho_q_gen < 0:
            raise ValueError(
                f"rho_q_gen must be non-negative, got {self.rho_q_gen}"
            )
        if self.enable_saturation_mode:
            if self.sat_eps_enter_mvar <= 0 or self.sat_eps_exit_mvar <= 0:
                raise ValueError(
                    "sat_eps_enter_mvar and sat_eps_exit_mvar must be positive, "
                    f"got {self.sat_eps_enter_mvar}, {self.sat_eps_exit_mvar}"
                )
            if self.sat_eps_exit_mvar <= self.sat_eps_enter_mvar:
                raise ValueError(
                    f"sat_eps_exit_mvar ({self.sat_eps_exit_mvar}) must be "
                    f"strictly greater than sat_eps_enter_mvar "
                    f"({self.sat_eps_enter_mvar}) to enforce hysteresis."
                )

        n_tie = len(self.tie_line_indices)
        if len(self.tie_line_endpoint_buses) != n_tie:
            raise ValueError(
                f"tie_line_endpoint_buses length ({len(self.tie_line_endpoint_buses)}) "
                f"must match tie_line_indices length ({n_tie})"
            )
        if self.q_tie_setpoints_mvar is None:
            self.q_tie_setpoints_mvar = np.zeros(n_tie, dtype=np.float64)
        elif len(self.q_tie_setpoints_mvar) != n_tie:
            raise ValueError(
                f"q_tie_setpoints_mvar length ({len(self.q_tie_setpoints_mvar)}) "
                f"must match tie_line_indices length ({n_tie})"
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
        # Wide default until first DSO capability message arrives.  With
        # the Strategy-D output-side bound on Q_PCC, a near-zero default
        # would forbid any actuator move on the very first TSO step
        # before any DSO has reported its capability.
        self.pcc_capability_min_mvar = np.full(n_pcc, -1E6)
        self.pcc_capability_max_mvar = np.full(n_pcc, +1E6)

        # Cache for the sensitivity matrix
        self._H_cache: Optional[NDArray[np.float64]] = None
        self._H_mappings: Optional[Dict] = None
        self._sensitivity_updater: Optional[SensitivityUpdater] = None

        # Cached measurement for capability-curve bounds (set in step())
        self._last_measurement: Optional[Measurement] = None

        # Achieved-Value Tracking verbosity (set from run_cascade)
        self._avt_verbose: int = 0

        # Out-of-service masks for generators and OLTCs (contingency handling).
        # True = element is out of service → freeze bounds, zero H columns.
        n_gen = len(config.gen_indices)
        n_oltc = len(config.oltc_trafo_indices)
        self._oos_gen_mask: NDArray[np.bool_] = np.zeros(n_gen, dtype=np.bool_)
        self._oos_oltc_mask: NDArray[np.bool_] = np.zeros(n_oltc, dtype=np.bool_)

        # AVR saturation mode per generator.  Values:
        #   0  = free (AVR regulating, gen acts as PV)
        #  +1  = saturated at upper Q limit (overexcitation)
        #  -1  = saturated at lower Q limit (underexcitation)
        # Persists across iterations to implement hysteresis (ε_exit > ε_enter).
        self._sat_mode: NDArray[np.int8] = np.zeros(n_gen, dtype=np.int8)

    # =========================================================================
    # Public interface for cascaded hierarchy communication
    # =========================================================================

    def update_outage_mask(
        self,
        oos_gen_indices: Set[int],
        oos_oltc_indices: Set[int],
    ) -> None:
        """
        Update out-of-service masks for generators and OLTCs.

        Called after a contingency changes element ``in_service`` status.
        Out-of-service actuators are frozen (bounds collapsed to current
        value) and their sensitivity columns are zeroed so the MIQP
        does not dispatch them.

        Parameters
        ----------
        oos_gen_indices : Set[int]
            ``net.gen`` indices that are currently out of service.
        oos_oltc_indices : Set[int]
            ``net.trafo`` indices of machine-transformer OLTCs that are
            currently out of service.
        """
        changed = False
        for k, g_idx in enumerate(self.config.gen_indices):
            new_val = g_idx in oos_gen_indices
            if self._oos_gen_mask[k] != new_val:
                changed = True
            self._oos_gen_mask[k] = new_val
        for k, t_idx in enumerate(self.config.oltc_trafo_indices):
            new_val = t_idx in oos_oltc_indices
            if self._oos_oltc_mask[k] != new_val:
                changed = True
            self._oos_oltc_mask[k] = new_val
        if changed:
            self.invalidate_sensitivity_cache()

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

    def voltage_curvature_inputs(
        self,
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Voltage rows of H and per-bus ``g_v`` for curvature analysis.

        TSO output ordering is ``[ V_bus | Q_PCC | I_line | Q_gen | Q_tie ]``,
        so the voltage block is the leading ``n_v`` rows.  ``g_v`` is the
        scalar :attr:`TSOControllerConfig.g_v` replicated per bus.  See
        :meth:`BaseOFOController.voltage_curvature_inputs`.
        """
        n_v = len(self.config.voltage_bus_indices)
        if n_v == 0:
            return None
        H = self._expand_H_to_der_level(self._build_sensitivity_matrix())
        H_v = np.ascontiguousarray(H[:n_v, :], dtype=np.float64)
        g_v_vec = np.full(n_v, float(self.config.g_v), dtype=np.float64)
        return H_v, g_v_vec

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

    def _get_oltc_integer_indices(self) -> List[int]:
        """OLTC slice within the integer block.

        TSO ordering ``[Q_DER | Q_PCC | V_gen | s_OLTC | s_shunt]``
        puts the OLTC integers immediately after the continuous block,
        before the shunts.  Used by :class:`BaseOFOController` to scope
        the wall-clock cooldown to OLTC indices only.
        """
        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_gen = len(self.config.gen_indices)
        n_oltc = len(self.config.oltc_trafo_indices)

        oltc_start = n_der + n_pcc + n_gen
        return list(range(oltc_start, oltc_start + n_oltc))

    def _actuator_class_indices(self) -> Dict[str, NDArray[np.int64]]:
        """Per-class index map for adaptive ``g_w`` (paper Eq. 16).

        Class names match the BO dimension naming in
        ``tuning/parameters.py``: ``"der"``, ``"pcc"``, ``"gen"``,
        ``"tso_oltc"``, ``"tso_shunt"``.  Empty
        classes are silently dropped from the map so the adapter does
        not waste mask bits on zero-length slices.
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

        oltc_start = n_der + n_pcc + n_gen
        oltc_end = oltc_start + n_oltc
        shunt_end = oltc_end + n_shunt

        out: Dict[str, NDArray[np.int64]] = {}
        if n_der > 0:
            out["der"] = np.arange(0, n_der, dtype=np.int64)
        if n_pcc > 0:
            out["pcc"] = np.arange(n_der, n_der + n_pcc, dtype=np.int64)
        if n_gen > 0:
            out["gen"] = np.arange(n_der + n_pcc, oltc_start, dtype=np.int64)
        if n_oltc > 0:
            out["tso_oltc"] = np.arange(oltc_start, oltc_end, dtype=np.int64)
        if n_shunt > 0:
            out["tso_shunt"] = np.arange(oltc_end, shunt_end, dtype=np.int64)
        return out

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

        Output ordering: [ V_bus | Q_PCC | I_line | Q_gen | Q_tie ]

        Q_PCC is the physical reactive power at the HV port of each PCC
        transformer (load convention into the coupler).  It is included
        as an output so the TSO MIQP sees ∂Q_PCC/∂s_shunt and ∂Q_PCC/∂*
        explicitly — letting the optimiser trade shunt switching
        against Q_PCC,set adjustments inside one optimisation.

        Q_gen is a monitored output with a state-dependent soft band drawn
        from the generator PQ capability curve each iteration.  The MIQP
        can trade off small Q_gen violations against voltage tracking and
        interface-Q objectives via the per-output slack variables.

        Q_tie is the reactive power flowing on each inter-zone tie line at
        the in-zone endpoint (load convention into the line — positive Q
        means reactive flowing OUT of this zone).  Phase A: rows present
        in H but not weighted in the objective (informational).
        """
        n_v = len(self.config.voltage_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_i = len(self.config.current_line_indices)
        n_gen = len(self.config.gen_indices)
        n_tie = len(self.config.tie_line_indices)
        n_outputs = n_v + n_pcc + n_i + n_gen + n_tie

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

        # Q_PCC (physical interface Q at HV port of each PCC trafo)
        for trafo_idx in self.config.pcc_trafo_indices:
            meas_idx = np.where(
                measurement.interface_transformer_indices == trafo_idx
            )[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"PCC transformer {trafo_idx} not found in measurement"
                )
            y[idx] = float(measurement.interface_q_hv_side_mvar[meas_idx[0]])
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

        # Generator Q measurements (monitored capability output)
        for g_idx in self.config.gen_indices:
            meas_idx = np.where(measurement.gen_indices == g_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Generator {g_idx} not found in measurement.gen_indices"
                )
            y[idx] = float(measurement.gen_q_mvar[meas_idx[0]])
            idx += 1

        # Tie-line Q at in-zone endpoint (Phase A monitoring)
        for tie_idx in self.config.tie_line_indices:
            meas_idx = np.where(measurement.tie_line_indices == tie_idx)[0]
            if len(meas_idx) == 0:
                raise ValueError(
                    f"Tie line {tie_idx} not found in measurement.tie_line_indices"
                )
            y[idx] = float(measurement.tie_line_q_mvar[meas_idx[0]])
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

        The generator capability curve enters the problem as a *soft
        output constraint* (Q_gen rows in H, penalised by ``rho_q_gen``)
        rather than as a pre-emptive tightening of the V_gen bounds.
        This lets the MIQP trade off small Q_gen violations against
        voltage tracking, interface Q tracking, and current limits.

        An asymmetric V_gen bound clamp is still applied when the AVR
        saturates (``self._sat_mode[k] != 0``): motion in the saturating
        direction is blocked so the MIQP can only attempt de-saturation.
        Saturation detection and mode transitions are hysteretic — see
        :meth:`_classify_saturation_modes`.

        OOS actuator bounds are collapsed to the current value
        (contingency masking).
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

        # --- PCC setpoint bounds ---
        # Two regimes:
        #   - pcc_capability_on_output=True (default, Strategy D):
        #       DSO capability is enforced as a soft output bound on the
        #       physical Q_PCC (see _get_output_limits).  The control
        #       variable Q_PCC,set gets a wide engineering band so the
        #       LP relaxation stays bounded.
        #   - pcc_capability_on_output=False (legacy):
        #       DSO capability is enforced as a hard input bound on the
        #       control Q_PCC,set, anchored at the current measured Q.
        if self.config.pcc_capability_on_output:
            _PCC_SET_WIDE_MVAR = 10_000.0
            u_lower[n_der:n_der + n_pcc] = -_PCC_SET_WIDE_MVAR
            u_upper[n_der:n_der + n_pcc] = +_PCC_SET_WIDE_MVAR
        else:
            u_lower[n_der:n_der + n_pcc] = (
                tso_dso_interface_q_current + self.pcc_capability_min_mvar
            )
            u_upper[n_der:n_der + n_pcc] = (
                tso_dso_interface_q_current + self.pcc_capability_max_mvar
            )

        # --- AVR setpoint bounds (fixed physical band) ---
        avr_start = n_der + n_pcc
        avr_end = avr_start + n_gen
        u_lower[avr_start:avr_end] = self.config.gen_vm_min_pu
        u_upper[avr_start:avr_end] = self.config.gen_vm_max_pu

        # --- AVR saturation: asymmetric clamp (Feature B, gated) ---
        # When the AVR is saturated, the MIQP can only move in the
        # de-saturating direction.  Saturation is detected with hysteresis
        # in _classify_saturation_modes; this clamp reads _sat_mode.
        if (
            self.config.enable_saturation_mode
            and self._u_current is not None
            and n_gen > 0
        ):
            for k in range(n_gen):
                mode = int(self._sat_mode[k])
                if mode == +1:
                    # Saturated at upper Q limit (overexcited): block V_gen raise
                    u_upper[avr_start + k] = self._u_current[avr_start + k]
                elif mode == -1:
                    # Saturated at lower Q limit (underexcited): block V_gen drop
                    u_lower[avr_start + k] = self._u_current[avr_start + k]

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

        # --- Freeze OOS generators and OLTCs (contingency masking) ---
        if self._u_current is not None:
            for k in range(n_gen):
                if self._oos_gen_mask[k]:
                    u_lower[avr_start + k] = self._u_current[avr_start + k]
                    u_upper[avr_start + k] = self._u_current[avr_start + k]
            for k in range(n_oltc):
                if self._oos_oltc_mask[k]:
                    u_lower[tap_start + k] = self._u_current[tap_start + k]
                    u_upper[tap_start + k] = self._u_current[tap_start + k]

        return u_lower, u_upper


    def _get_output_limits(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get output constraint limits.

        Output ordering: [ V_bus | Q_PCC | I_line | Q_gen | Q_tie ]

        Voltage outputs are constrained to the permissible band
        [v_min_pu, v_max_pu].  Tracking toward voltage setpoints
        (if configured) is handled by the quadratic objective in
        ``_compute_objective_gradient()``, not by tight constraints.

        Q_PCC bounds use the DSO-reported capability interval centred
        at the current measured Q at each PCC, when
        ``config.pcc_capability_on_output`` is True (default).  When
        False, the output bound is wide and the same DSO capability
        is enforced on the *control* Q_PCC,set in
        ``_compute_input_bounds`` (legacy behaviour).

        Q_gen bounds are state-dependent (drawn from the generator PQ
        capability curve each iteration) and enforced as soft outputs
        with penalty ``rho_q_gen`` — see the ``g_z`` plumbing for the
        corresponding slack weight.

        Q_tie bounds are wide-open in Phase A (informational rows only).
        Phase B may tighten them using ``g_z_q_tie`` slack penalty.
        """
        n_v = len(self.config.voltage_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_i = len(self.config.current_line_indices)
        n_gen = len(self.config.gen_indices)
        n_tie = len(self.config.tie_line_indices)
        n_outputs = n_v + n_pcc + n_i + n_gen + n_tie

        y_lower = np.zeros(n_outputs)
        y_upper = np.zeros(n_outputs)
        idx = 0

        # --- Voltage limits (band constraints) ---
        for j in range(n_v):
            y_lower[idx] = self.config.v_min_pu
            y_upper[idx] = self.config.v_max_pu
            idx += 1

        # --- Q_PCC band (DSO capability on output, when enabled) ---
        if n_pcc > 0:
            if self.config.pcc_capability_on_output:
                if self._last_measurement is not None:
                    q_iface_now = self._extract_trafo_reactive_power(
                        self._last_measurement,
                    )
                else:
                    q_iface_now = np.zeros(n_pcc)
                y_lower[idx:idx + n_pcc] = (
                    q_iface_now + self.pcc_capability_min_mvar
                )
                y_upper[idx:idx + n_pcc] = (
                    q_iface_now + self.pcc_capability_max_mvar
                )
            else:
                # Legacy: capability bound on *control* Q_PCC,set instead.
                y_lower[idx:idx + n_pcc] = -1e6
                y_upper[idx:idx + n_pcc] = +1e6
            idx += n_pcc

        # --- Current limits (upper only, kA) ---
        for j in range(n_i):
            if self.config.current_line_max_i_ka is not None:
                i_lim_ka = self.config.i_max_pu * self.config.current_line_max_i_ka[j]
            else:
                i_lim_ka = 1E6  # no limit if ratings not provided
            y_lower[idx] = 0.0
            y_upper[idx] = i_lim_ka
            idx += 1

        # --- Generator Q capability limits (state-dependent) ---
        if n_gen > 0:
            if (
                self.actuator_bounds.gen_params is not None
                and self._last_measurement is not None
                and len(self._last_measurement.gen_p_mw) == n_gen
                and len(self._last_measurement.gen_vm_pu) == n_gen
            ):
                meas = self._last_measurement
                q_min, q_max = self.actuator_bounds.compute_gen_q_bounds(
                    meas.gen_p_mw, meas.gen_vm_pu,
                )
                y_lower[idx:idx + n_gen] = q_min
                y_upper[idx:idx + n_gen] = q_max
            else:
                # No capability information yet (e.g. before first measurement).
                # Use permissive bounds so the constraint is effectively inactive.
                y_lower[idx:idx + n_gen] = -1e6
                y_upper[idx:idx + n_gen] = +1e6

            # OOS generators: open the bound so the soft constraint is inactive
            # (the Q_gen row is also zeroed in H, so the slack is unused).
            for k in range(n_gen):
                if self._oos_gen_mask[k]:
                    y_lower[idx + k] = -1e6
                    y_upper[idx + k] = +1e6
            idx += n_gen

        # --- Q_tie bounds (Phase A: wide-open, informational only) -------
        if n_tie > 0:
            y_lower[idx:idx + n_tie] = -1e6
            y_upper[idx:idx + n_tie] = +1e6
            idx += n_tie

        return y_lower, y_upper

    def _compute_output_gradient(
        self,
        measurement: Measurement,
    ) -> NDArray[np.float64]:
        """Output-space objective gradient ``∇_y f`` for this TSO zone.

        Returns the gradient of the TSO objective with respect to the *outputs*
        ``y``, in the canonical output ordering ``[V | Q_PCC | I | Q_gen | Q_tie]``
        (same ordering as :meth:`_extract_outputs` and the rows of
        :meth:`_build_sensitivity_matrix`).  The control-space gradient used by
        the MIQP is recovered by projecting through the sensitivity matrix,
        ``∇_u f = (∂y/∂u)ᵀ · ∇_y f`` (plus the direct DER-reserve term, which
        acts on a control variable, not an output).

        This is the SINGLE source of the objective gradient: the switched-shunt
        integrator (:mod:`controller.shunt_integrator`) dots this vector with a
        bank's boundary sensitivity column ``h_H`` to obtain ``g_H = h_Hᵀ ∇_y f``,
        so the integrator follows literally the same objective ``f`` as the rest
        of the TSO loop.  The consistency between this vector and
        :meth:`_compute_objective_gradient` is pinned by
        ``tests/test_tso_output_gradient.py``.

        Per-block definitions (zero where the corresponding term is inactive):

        * ``V``     : ``2 · g_v · (V − V_set)``                   (voltage tracking)
        * ``Q_PCC`` : ``2 · g_q_tso · (Q_PCC − Q_PCC,set)``       (interface-Q tracking)
        * ``I``     : ``0``                                       (current is a constraint, not an objective term)
        * ``Q_gen`` : ``2 · g_res_sg · (Q_gen − Q_mid)/Q_half²``  (SG reactive-reserve centring)
        * ``Q_tie`` : ``2 · g_q_tie · (Q_tie − Q_tie,set)``       (tie-line-Q tracking)

        NOTE: the DER reactive-reserve term (``g_res_der``) is intentionally NOT
        included here — ``Q_DER`` is a direct control variable, so that term
        lands straight on the control gradient in
        :meth:`_compute_objective_gradient`, not on an output.
        """
        n_v = len(self.config.voltage_bus_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        n_i = len(self.config.current_line_indices)
        n_gen = len(self.config.gen_indices)
        n_tie = len(self.config.tie_line_indices)

        grad_v = np.zeros(n_v, dtype=np.float64)
        grad_pcc = np.zeros(n_pcc, dtype=np.float64)
        grad_i = np.zeros(n_i, dtype=np.float64)
        grad_gen = np.zeros(n_gen, dtype=np.float64)
        grad_tie = np.zeros(n_tie, dtype=np.float64)

        mapping = self.config.der_mapping
        n_der_for_u = mapping.n_der if mapping is not None else len(self.config.der_indices)

        # --- Voltage-schedule tracking ------------------------------------
        if self.config.v_setpoints_pu is not None:
            v_current = np.zeros(n_v)
            for j, bus_idx in enumerate(self.config.voltage_bus_indices):
                meas_idx = np.where(measurement.bus_indices == bus_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(f"Bus {bus_idx} not found in measurement")
                v_current[j] = measurement.voltage_magnitudes_pu[meas_idx[0]]
            grad_v = 2.0 * self.config.g_v * (v_current - self.config.v_setpoints_pu)

        # --- Q_PCC tracking ------------------------------------------------
        if n_pcc > 0 and self.config.g_q_tso != 0.0:
            q_pcc_meas = self._extract_trafo_reactive_power(measurement)
            if self._u_current is not None:
                q_pcc_set = self._u_current[n_der_for_u:n_der_for_u + n_pcc]
            else:
                q_pcc_set = q_pcc_meas  # zero error on first call
            grad_pcc = 2.0 * self.config.g_q_tso * (q_pcc_meas - q_pcc_set)

        # --- Q_tie tracking ------------------------------------------------
        if n_tie > 0 and self.config.g_q_tie != 0.0:
            q_tie_meas = np.zeros(n_tie, dtype=np.float64)
            for k, tie_idx in enumerate(self.config.tie_line_indices):
                meas_idx = np.where(measurement.tie_line_indices == tie_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"Tie line {tie_idx} not found in measurement.tie_line_indices"
                    )
                q_tie_meas[k] = float(measurement.tie_line_q_mvar[meas_idx[0]])
            q_tie_set = (
                self.config.q_tie_setpoints_mvar
                if self.config.q_tie_setpoints_mvar is not None
                else np.zeros(n_tie, dtype=np.float64)
            )
            grad_tie = 2.0 * self.config.g_q_tie * (q_tie_meas - q_tie_set)

        # --- SG reactive-reserve centring ---------------------------------
        if (
            n_gen > 0
            and self.config.g_res_sg != 0.0
            and self.actuator_bounds.gen_params is not None
            and len(measurement.gen_p_mw) == n_gen
            and len(measurement.gen_vm_pu) == n_gen
            and len(measurement.gen_q_mvar) == n_gen
        ):
            _RES_HALF_EPS = 1e-6
            q_min_g, q_max_g = self.actuator_bounds.compute_gen_q_bounds(
                measurement.gen_p_mw, measurement.gen_vm_pu,
            )
            q_mid_g = 0.5 * (q_max_g + q_min_g)
            q_half_g = 0.5 * (q_max_g - q_min_g)
            q_gen = np.asarray(measurement.gen_q_mvar, dtype=np.float64)
            ok_g = q_half_g > _RES_HALF_EPS
            grad_gen[ok_g] = (
                2.0 * self.config.g_res_sg
                * (q_gen[ok_g] - q_mid_g[ok_g]) / (q_half_g[ok_g] ** 2)
            )
            for k in range(n_gen):
                if self._oos_gen_mask[k]:
                    grad_gen[k] = 0.0

        return np.concatenate([grad_v, grad_pcc, grad_i, grad_gen, grad_tie])

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
        n_pcc = len(self.config.pcc_trafo_indices)
        n_i = len(self.config.current_line_indices)
        n_gen = len(self.config.gen_indices)
        n_tie = len(self.config.tie_line_indices)

        # --- Component 1: DER usage regularisation ---
        # Handled implicitly by g_u in build_miqp_problem; no explicit
        # gradient needed here (build_miqp_problem adds 2*alpha*g_u*u_current).

        # Build H once and reuse for both V- and Q_PCC-tracking terms.
        H_built = False
        H = None

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
            H_built = True
            # Voltage outputs are the first n_v rows of H
            dV_du = H[:n_v, :]

            # grad_f contribution: 2 * g_v * (V - V_set)^T * dV/du
            grad_f += 2.0 * self.config.g_v * (v_error @ dV_du)

        # --- Component 3: Q_PCC tracking (Strategy D) ---
        # When the Q_PCC output rows are re-enabled (always, in this
        # build) and ``g_q_tso != 0``, push the MIQP toward solutions
        # that keep Q_PCC near Q_PCC,set.  In the closed-loop view
        # Q_PCC = Q_PCC,set when the DSO tracks perfectly; the gradient
        # only acts when an open-loop disturbance (e.g. a TSO shunt
        # switch the DSO is blind to) deviates physical Q_PCC from
        # the commanded value.
        if n_pcc > 0 and self.config.g_q_tso != 0.0:
            if not H_built:
                H_bus = self._build_sensitivity_matrix()
                H = self._expand_H_to_der_level(H_bus)
                H_built = True

            mapping = self.config.der_mapping
            n_der_for_u = (
                mapping.n_der if mapping is not None
                else len(self.config.der_indices)
            )
            q_pcc_meas = self._extract_trafo_reactive_power(measurement)
            if self._u_current is not None:
                q_pcc_set = self._u_current[
                    n_der_for_u:n_der_for_u + n_pcc
                ]
            else:
                q_pcc_set = q_pcc_meas  # zero error on first call
            q_pcc_err = q_pcc_meas - q_pcc_set
            dQpcc_du = H[n_v:n_v + n_pcc, :]
            grad_f += 2.0 * self.config.g_q_tso * (q_pcc_err @ dQpcc_du)

        # --- Component 4: Q_tie tracking (Phase B) ---
        # When ``g_q_tie != 0``, push the MIQP toward solutions that
        # keep the inter-zone tie-line reactive power close to its
        # configured setpoint (default 0 Mvar across all ties).  Both
        # zones touching a tie line independently steer their own
        # in-zone endpoint Q toward the target — decentralised, with no
        # real-time inter-zone exchange.  Mirrors the Q_PCC tracking
        # pattern above.
        if n_tie > 0 and self.config.g_q_tie != 0.0:
            if not H_built:
                H_bus = self._build_sensitivity_matrix()
                H = self._expand_H_to_der_level(H_bus)
                H_built = True

            # Q_tie row block sits after [V | Q_PCC | I | Q_gen].
            q_tie_row_start = n_v + n_pcc + n_i + n_gen

            # Read measured Q_tie at each in-zone endpoint.  Order matches
            # self.config.tie_line_indices because measure_zone_tso copies
            # zone_def.tie_line_indices into measurement.tie_line_indices.
            q_tie_meas = np.zeros(n_tie, dtype=np.float64)
            for k, tie_idx in enumerate(self.config.tie_line_indices):
                meas_idx = np.where(measurement.tie_line_indices == tie_idx)[0]
                if len(meas_idx) == 0:
                    raise ValueError(
                        f"Tie line {tie_idx} not found in measurement.tie_line_indices"
                    )
                q_tie_meas[k] = float(measurement.tie_line_q_mvar[meas_idx[0]])

            q_tie_set = (
                self.config.q_tie_setpoints_mvar
                if self.config.q_tie_setpoints_mvar is not None
                else np.zeros(n_tie, dtype=np.float64)
            )
            q_tie_err = q_tie_meas - q_tie_set
            dQtie_du = H[q_tie_row_start:q_tie_row_start + n_tie, :]
            grad_f += 2.0 * self.config.g_q_tie * (q_tie_err @ dQtie_du)

        # --- Component 5: Explicit reactive-reserve penalisation (optional) ---
        # Keep TS actuators centred in their (state-dependent) Q-capability
        # band so symmetric reactive reserve is retained.  Each actuator's
        # normalised reserve coordinate is
        #     r_i = (Q_i - Q_mid,i) / Q_half,i,
        # with Q_mid = ½(Q_min+Q_max), Q_half = ½(Q_max-Q_min); the penalty
        # g_res · Σ_i r_i² has gradient 2·g_res·(Q_i - Q_mid)/Q_half² w.r.t Q_i.
        # Two independent weights (SG vs DER) let the operator prefer one
        # resource class over the other.  When a weight is 0 its block is
        # skipped — the term is then not part of the objective (mirrors the
        # g_q_tie toggle).  DSO-connected DER reserve is intentionally NOT
        # penalised here — it belongs to the DSO (Layer-2) controllers.
        _RES_HALF_EPS = 1e-6  # Mvar; bands narrower than this are treated as
        #                       collapsed (no reserve preference, avoids /0).

        # SG reserve: Q_gen is an OUTPUT (rows q_gen_row_start..+n_gen of H),
        # so the per-gen coefficient is mapped to control space via ∂Q_gen/∂u.
        if (
            n_gen > 0
            and self.config.g_res_sg != 0.0
            and self.actuator_bounds.gen_params is not None
            and len(measurement.gen_p_mw) == n_gen
            and len(measurement.gen_vm_pu) == n_gen
            and len(measurement.gen_q_mvar) == n_gen
        ):
            if not H_built:
                H_bus = self._build_sensitivity_matrix()
                H = self._expand_H_to_der_level(H_bus)
                H_built = True

            q_min_g, q_max_g = self.actuator_bounds.compute_gen_q_bounds(
                measurement.gen_p_mw, measurement.gen_vm_pu,
            )
            q_mid_g = 0.5 * (q_max_g + q_min_g)
            q_half_g = 0.5 * (q_max_g - q_min_g)
            q_gen = np.asarray(measurement.gen_q_mvar, dtype=np.float64)

            coeff_sg = np.zeros(n_gen, dtype=np.float64)
            ok_g = q_half_g > _RES_HALF_EPS
            coeff_sg[ok_g] = (
                2.0 * self.config.g_res_sg
                * (q_gen[ok_g] - q_mid_g[ok_g]) / (q_half_g[ok_g] ** 2)
            )
            # OOS generators contribute nothing (their Q_gen row is already
            # zeroed in H, but guard the coefficient too for clarity/safety).
            for k in range(n_gen):
                if self._oos_gen_mask[k]:
                    coeff_sg[k] = 0.0

            q_gen_row_start = n_v + n_pcc + n_i
            dQgen_du = H[q_gen_row_start:q_gen_row_start + n_gen, :]
            reserve_contrib_sg = coeff_sg @ dQgen_du
            grad_f += reserve_contrib_sg
            self._debug_reserve_term("SG", reserve_contrib_sg, grad_f, extra={
                "r_norm": float(np.linalg.norm(
                    np.where(ok_g, (q_gen - q_mid_g) / np.where(ok_g, q_half_g, 1.0), 0.0)
                )),
                "max_abs_r": float(np.max(np.abs(
                    np.where(ok_g, (q_gen - q_mid_g) / np.where(ok_g, q_half_g, 1.0), 0.0)
                )) if n_gen else 0.0),
            })

        # DER reserve: Q_DER is a DIRECT control variable (first n_der_ctrl
        # columns of u), so the coefficient lands straight on grad_f — no
        # sensitivity mapping is needed.
        if self.config.g_res_der != 0.0:
            mapping = self.config.der_mapping
            n_der_ctrl = (
                mapping.n_der if mapping is not None
                else len(self.config.der_indices)
            )
            der_p = self._extract_der_active_power(measurement)
            if n_der_ctrl > 0 and len(der_p) == n_der_ctrl:
                q_min_d, q_max_d = self.actuator_bounds.compute_der_q_bounds(der_p)
                q_mid_d = 0.5 * (q_max_d + q_min_d)
                q_half_d = 0.5 * (q_max_d - q_min_d)
                if self._u_current is not None:
                    q_der = self._u_current[:n_der_ctrl]
                else:
                    # First call: anchor at the midpoint → zero reserve term.
                    q_der = q_mid_d
                coeff_der = np.zeros(n_der_ctrl, dtype=np.float64)
                ok_d = q_half_d > _RES_HALF_EPS
                coeff_der[ok_d] = (
                    2.0 * self.config.g_res_der
                    * (q_der[ok_d] - q_mid_d[ok_d]) / (q_half_d[ok_d] ** 2)
                )
                grad_f[:n_der_ctrl] += coeff_der
                _contrib_der = np.zeros_like(grad_f)
                _contrib_der[:n_der_ctrl] = coeff_der
                self._debug_reserve_term("DER", _contrib_der, grad_f)

        return grad_f

    def _debug_reserve_term(
        self,
        tag: str,
        contrib: NDArray[np.float64],
        grad_f: NDArray[np.float64],
        extra: Optional[dict] = None,
    ) -> None:
        """Opt-in diagnostic for the explicit reserve term (Component 5).

        Enabled only when the environment variable ``QOFO_DEBUG_RESERVE`` is
        set (to any non-empty value).  Prints, for the first few TS steps of
        this controller, the magnitude of the reserve gradient contribution
        and — crucially — the implied OFO step ``|contrib| / g_w`` per actuator
        class, so a *large* reserve gradient that is *frozen* by a large
        ``g_w`` (e.g. ``g_w_gen``) is visible.  No effect on the dispatch.
        """
        import os
        if not os.environ.get("QOFO_DEBUG_RESERVE"):
            return
        cnt = getattr(self, "_reserve_dbg_count", 0)
        if cnt >= 6:  # a handful of steps is enough to diagnose
            return
        self._reserve_dbg_count = cnt + 1
        contrib = np.asarray(contrib, dtype=np.float64)
        g_w = np.asarray(getattr(self.params, "g_w", np.ones_like(contrib)),
                         dtype=np.float64)
        if g_w.shape != contrib.shape:
            g_w = np.broadcast_to(g_w, contrib.shape)
        # Implied (un-clamped, un-alpha'd) OFO step magnitude per control.
        step = np.where(g_w > 0, contrib / g_w, 0.0)
        nz = np.flatnonzero(np.abs(contrib) > 0)
        msg = (
            f"[reserve:{tag}] {self.controller_id} call#{self._reserve_dbg_count} "
            f"||contrib||={np.linalg.norm(contrib):.4g} "
            f"||grad_f||={np.linalg.norm(grad_f):.4g} "
            f"max|contrib|={np.max(np.abs(contrib)) if contrib.size else 0.0:.4g} "
            f"max|step=contrib/g_w|={np.max(np.abs(step)) if step.size else 0.0:.4g} "
            f"nonzero_cols={nz.tolist()} "
            f"g_w@nz={[float(g_w[i]) for i in nz][:8]}"
        )
        if extra:
            msg += " " + " ".join(f"{k}={v:.4g}" for k, v in extra.items())
        print(msg)

    def _build_sensitivity_matrix(self) -> NDArray[np.float64]:
        """
        Build the input-output sensitivity matrix H.

        The matrix maps control variable changes to output changes:
            Δy ≈ H · Δu

        Columns correspond to:  [ Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]
        Rows correspond to:     [ V_bus | I_line | Q_gen ]

        Q_gen rows were added for the generator-capability soft
        constraint (Feature A).  Q_PCC rows remain removed — Q_PCC_set
        is a direct decision variable.

        For generators flagged as saturated in ``self._sat_mode``
        (Feature B), the corresponding V_gen column is zeroed across
        all row blocks (PQ-mode treatment): a V_gen move in the
        saturating direction has no effect on the network state while
        the AVR is rail-bound.  Motion in the de-saturating direction
        is caught at the next iteration's mode reclassification.

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
        n_tie = len(self.config.tie_line_indices)

        n_controls = n_der_bus + n_pcc + n_gen + n_oltc + n_shunt
        n_outputs = n_v + n_pcc + n_i + n_gen + n_tie  # +Q_tie (Phase A)

        # Row offsets for the five output blocks
        # (Layout: [V_bus | Q_PCC | I_line | Q_gen | Q_tie]).
        q_pcc_row_start = n_v
        i_row_start     = n_v + n_pcc
        q_row_start     = n_v + n_pcc + n_i  # Q_gen block
        q_tie_row_start = n_v + n_pcc + n_i + n_gen  # Q_tie block (Phase A)

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

        # Prefer 3W lookup: PCC trafos in this codebase are always 3W
        # couplers.  pp.trafo and pp.trafo3w are independent index spaces
        # so the same int (e.g. 0) can exist in both — we must choose
        # the 3W table when the PCCs live there to feed
        # ``compute_dQtrafo3w_hv_*`` rather than ``compute_dQtrafo_2w_*``.
        pcc_in_trafo3w = (
            hasattr(net, "trafo3w")
            and not net.trafo3w.empty
            and all(
                t in net.trafo3w.index
                for t in self.config.pcc_trafo_indices
            )
        )
        pcc_in_trafo = (
            (not pcc_in_trafo3w)
            and (not net.trafo.empty)
            and all(t in net.trafo.index for t in self.config.pcc_trafo_indices)
        )

        # Note: We no longer request Q_trafo rows from the physical
        # sensitivity builder, since Q_PCC is not an output anymore.
        # However, we still pass trafo indices so the builder can
        # include them in the internal row mapping (may affect V/I rows).
        # Only pass in-service OLTCs to the Jacobian builder.
        # OOS machine trafos cause ValueError in compute_dV_ds_2w
        # (disconnected branch → no admittance entry), and the builder
        # silently skips them, producing fewer columns than expected.
        is_oltc_indices = [
            t for k, t in enumerate(self.config.oltc_trafo_indices)
            if not self._oos_oltc_mask[k]
        ]
        # Under the local-net sensitivity mode the runner remaps tertiary
        # shunt indices to synthetic primary-bus shunts; honour that
        # remapping when present.  Falls back to ``shunt_bus_indices``
        # otherwise (legacy / full-net mode).
        _shunt_buses_for_sens = (
            self.config.shunt_sensitivity_bus_indices
            if self.config.shunt_sensitivity_bus_indices is not None
            else self.config.shunt_bus_indices
        )
        kw = dict(
            der_bus_indices=der_bus_indices,
            observation_bus_indices=self.config.voltage_bus_indices,
            line_indices=self.config.current_line_indices,
            oltc_trafo_indices=is_oltc_indices,
            shunt_bus_indices=_shunt_buses_for_sens,
            shunt_q_steps_mvar=self.config.shunt_q_steps_mvar,
        )
        if pcc_in_trafo:
            kw["trafo_indices"] = self.config.pcc_trafo_indices
        elif pcc_in_trafo3w:
            kw["trafo3w_indices"] = self.config.pcc_trafo_indices
        n_oltc_is = len(is_oltc_indices)

        # build_sensitivity_matrix_H requires at least one physical input
        # (DER / OLTC / shunt).  When all are absent, skip the call —
        # H_physical is all-zero.
        has_physical_inputs = (
            len(der_bus_indices) > 0
            or n_oltc_is > 0
            or len(self.config.shunt_bus_indices) > 0
        )
        mappings: dict = {}

        # Whether the physical builder emits leading Q_trafo (Q_PCC) rows.
        # Defined unconditionally (depends only on config) so the V_gen / Q_PCC
        # column-filling code below is safe even when ``has_physical_inputs`` is
        # False — e.g. a zone whose only controls are Q_PCC setpoints.
        has_pcc_rows = pcc_in_trafo or pcc_in_trafo3w

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
                H[i_row_start:i_row_start + n_i_copy, :n_der] = (
                    H_physical[n_q_phys + n_v:n_q_phys + n_v + n_i_copy, :n_der]
                )

            # Q_PCC rows: copy the H_physical Q_trafo block (rows 0..n_pcc-1)
            # for DER columns.  Strategy D re-enables these rows.
            if n_pcc > 0 and has_pcc_rows:
                H[q_pcc_row_start:q_pcc_row_start + n_pcc, :n_der] = (
                    H_physical[:n_pcc, :n_der]
                )

        # --- Q_PCC,set columns: closed-loop identity ----------------
        # The DSO tracks each Q_PCC,set independently.  Diagonal +1
        # captures "the commanded setpoint becomes the achieved Q at
        # this PCC".  Off-diagonal cross-PCC effects flow through the
        # other actuator columns (DER, V_gen, OLTC, shunt), so they are
        # zero on the Q_PCC,set columns by design.
        if n_pcc > 0:
            for j_pcc in range(n_pcc):
                H[q_pcc_row_start + j_pcc, n_der + j_pcc] = 1.0

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
                            H[i_row_start + i_line, col] = (
                                -dI_dQ_pcc[i_line, j_jac]
                            )

        if has_physical_inputs:
            # OLTC columns: H_physical has n_oltc_is columns (only in-service
            # OLTCs).  Map each back to the correct target column in H.
            # OOS OLTC target columns stay zero.
            is_pos = 0
            for k, t_idx in enumerate(self.config.oltc_trafo_indices):
                target_col = n_der + n_pcc + n_gen + k
                if not self._oos_oltc_mask[k]:
                    phys_col = n_der + is_pos
                    H[:n_v, target_col] = H_physical[
                        n_q_phys:n_q_phys + n_v, phys_col
                    ]
                    if n_i_copy > 0:
                        H[i_row_start:i_row_start + n_i_copy, target_col] = (
                            H_physical[
                                n_q_phys + n_v:n_q_phys + n_v + n_i_copy,
                                phys_col,
                            ]
                        )
                    # Q_PCC rows: copy from H_physical's Q_trafo block
                    if n_pcc > 0 and has_pcc_rows:
                        H[q_pcc_row_start:q_pcc_row_start + n_pcc, target_col] = (
                            H_physical[:n_pcc, phys_col]
                        )
                    is_pos += 1

            # Shunt columns: shifted in H_physical because we only passed
            # n_oltc_is OLTC columns instead of n_oltc.
            col_sh_phys = slice(n_der + n_oltc_is, n_der + n_oltc_is + n_shunt)
            col_sh_target = slice(n_der + n_pcc + n_gen + n_oltc, n_controls)
            H[:n_v, col_sh_target] = H_physical[
                n_q_phys:n_q_phys + n_v, col_sh_phys
            ]
            if n_i_copy > 0:
                H[i_row_start:i_row_start + n_i_copy, col_sh_target] = (
                    H_physical[
                        n_q_phys + n_v:n_q_phys + n_v + n_i_copy, col_sh_phys
                    ]
                )
            # Q_PCC rows: shunt columns from H_physical's Q_trafo block
            if n_shunt > 0 and n_pcc > 0 and has_pcc_rows:
                H[q_pcc_row_start:q_pcc_row_start + n_pcc, col_sh_target] = (
                    H_physical[:n_pcc, col_sh_phys]
                )

        # --- AVR columns: ∂V_obs / ∂V_gen from Jacobian-based sensitivity ---
        avr_start = n_der + n_pcc
        gen_terminal_buses = [
            int(net.gen.at[g, "bus"]) for g in self.config.gen_indices
        ]
        # Only query in-service generators — OOS gen terminal buses may be
        # isolated (machine trafo tripped), causing the Jacobian to fail.
        is_gen_buses = [
            int(net.gen.at[g, "bus"])
            for g, oos in zip(self.config.gen_indices, self._oos_gen_mask)
            if not oos
        ]
        if is_gen_buses:
            dV_dVgen, obs_map, gen_map = \
                self.sensitivities.compute_dV_dVgen_matrix(
                    gen_bus_indices_pp=is_gen_buses,
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
                        gen_bus_indices_pp=is_gen_buses,
                    )
                for k, gen_bus_pp in enumerate(gen_terminal_buses):
                    col = avr_start + k
                    if gen_bus_pp in gen_map_i:
                        j_gen = gen_map_i.index(gen_bus_pp)
                        for i_line, l_idx in enumerate(line_map):
                            H[i_row_start + i_line, col] = (
                                dI_dVgen[i_line, j_gen]
                            )

        # ∂Q_PCC / ∂V_gen — fills the Q_PCC rows for AVR columns.
        # Strategy D: re-enabled.  Skips OOS gens (their V_gen column is
        # already zeroed below).
        if n_pcc > 0 and has_pcc_rows and is_gen_buses:
            try:
                dQpcc_dVgen, t3w_map_g, gen_map_g = (
                    self.sensitivities.compute_dQtrafo3w_hv_dVgen_matrix(
                        gen_bus_indices_pp=is_gen_buses,
                        trafo3w_indices=self.config.pcc_trafo_indices,
                    )
                )
                for k, gen_bus_pp in enumerate(gen_terminal_buses):
                    col = avr_start + k
                    if gen_bus_pp not in gen_map_g:
                        continue
                    j_g = gen_map_g.index(gen_bus_pp)
                    for i_t3w, t_idx in enumerate(t3w_map_g):
                        if t_idx not in self.config.pcc_trafo_indices:
                            continue
                        i_row_local = self.config.pcc_trafo_indices.index(t_idx)
                        H[q_pcc_row_start + i_row_local, col] = (
                            dQpcc_dVgen[i_t3w, j_g]
                        )
            except (AttributeError, ValueError):
                # If the helper does not exist (unlikely) or PCC trafos
                # are not all 3W, leave Q_PCC × V_gen entries at 0.
                pass

        # =====================================================================
        # Q_gen rows: generator Q capability as a soft output (Feature A).
        # =====================================================================
        # Row offset: q_row_start = n_v + n_i (set earlier).
        # Columns: DER | PCC_set | V_gen | OLTC | shunt.
        if n_gen > 0:
            # --- DER Q → Q_gen ---
            if der_bus_indices:
                dQgen_dQder, _, _ = \
                    self.sensitivities.compute_dQgen_dQder_matrix(
                        gen_bus_indices_pp=gen_terminal_buses,
                        der_bus_indices=der_bus_indices,
                    )
                H[q_row_start:q_row_start + n_gen, :n_der] = dQgen_dQder

            # --- PCC setpoint (load-convention) → Q_gen ---
            # Same convention as V/I rows: Q_PCC_set is load-convention on the
            # HV side, so negate the generator-convention sensitivity.
            if pcc_hv_buses:
                dQgen_dQpcc, _, pcc_map_q = \
                    self.sensitivities.compute_dQgen_dQder_matrix(
                        gen_bus_indices_pp=gen_terminal_buses,
                        der_bus_indices=pcc_hv_buses,
                    )
                for j_pcc, bus in enumerate(pcc_hv_buses):
                    if bus in pcc_map_q:
                        j_jac = pcc_map_q.index(bus)
                        col = n_der + j_pcc
                        H[q_row_start:q_row_start + n_gen, col] = (
                            -dQgen_dQpcc[:, j_jac]
                        )

            # --- V_gen setpoint → Q_gen  (includes direct diagonal term) ---
            if is_gen_buses:
                dQgen_dVgen, _, gen_map_q = \
                    self.sensitivities.compute_dQgen_dVgen_matrix(
                        gen_bus_indices_pp_meas=gen_terminal_buses,
                        gen_bus_indices_pp_chg=is_gen_buses,
                    )
                for k, gen_bus_pp in enumerate(gen_terminal_buses):
                    col = avr_start + k
                    if gen_bus_pp in gen_map_q:
                        j_chg = gen_map_q.index(gen_bus_pp)
                        H[q_row_start:q_row_start + n_gen, col] = (
                            dQgen_dVgen[:, j_chg]
                        )

            # --- 2W OLTC tap → Q_gen ---
            if is_oltc_indices:
                dQgen_dsOltc, _, _ = \
                    self.sensitivities.compute_dQgen_ds_2w_matrix(
                        gen_bus_indices_pp=gen_terminal_buses,
                        oltc_trafo_indices=is_oltc_indices,
                    )
                # is_oltc_indices is a filtered (in-service) subset; remap to
                # the full OLTC column ordering (OOS columns remain zero).
                is_pos_q = 0
                for k, t_idx in enumerate(self.config.oltc_trafo_indices):
                    if not self._oos_oltc_mask[k]:
                        target_col = n_der + n_pcc + n_gen + k
                        H[q_row_start:q_row_start + n_gen, target_col] = (
                            dQgen_dsOltc[:, is_pos_q]
                        )
                        is_pos_q += 1

            # --- Shunt state → Q_gen ---
            if self.config.shunt_bus_indices:
                dQgen_dShunt, _, _ = \
                    self.sensitivities.compute_dQgen_dQ_shunt_matrix(
                        gen_bus_indices_pp=gen_terminal_buses,
                        shunt_bus_indices=_shunt_buses_for_sens,
                        shunt_q_steps_mvar=self.config.shunt_q_steps_mvar,
                    )
                col_sh_start = n_der + n_pcc + n_gen + n_oltc
                H[q_row_start:q_row_start + n_gen,
                  col_sh_start:col_sh_start + n_shunt] = dQgen_dShunt

            # OOS generators: zero the *row* for any out-of-service gen.
            # (The column is also zeroed below via the existing loop, but
            # row zeroing keeps Q_gen_oos unresponsive to every input.)
            for k in range(n_gen):
                if self._oos_gen_mask[k]:
                    H[q_row_start + k, :] = 0.0

        # =====================================================================
        # Q_tie rows: tie-line reactive power as a monitored output (Phase A).
        # =====================================================================
        # Row offset: q_tie_row_start = n_v + n_pcc + n_i + n_gen.
        # Columns: DER | PCC_set | V_gen | OLTC | shunt, same as the other
        # output blocks.  The Q_tie rows are populated only when the zone
        # actually owns tie lines (n_tie > 0).  In Phase A the rows are
        # informational (g_q_tie defaults to 0); the gradient uses them only
        # when g_q_tie != 0 (Phase B).
        if n_tie > 0:
            tie_indices = list(self.config.tie_line_indices)
            tie_endpoints = list(self.config.tie_line_endpoint_buses)

            # --- DER Q → Q_tie ---
            if der_bus_indices:
                dQtie_dQder, _, _ = (
                    self.sensitivities.compute_dQ_line_dQ_der_matrix(
                        tie_line_indices=tie_indices,
                        tie_line_endpoint_buses=tie_endpoints,
                        der_bus_indices=der_bus_indices,
                    )
                )
                H[q_tie_row_start:q_tie_row_start + n_tie, :n_der] = dQtie_dQder

            # --- PCC setpoint (load-convention) → Q_tie ---
            # Q_PCC,set is load-convention on the HV side; the underlying
            # primitive uses generator convention, so negate.
            if pcc_hv_buses:
                dQtie_dQpcc, _, pcc_map_tie = (
                    self.sensitivities.compute_dQ_line_dQ_der_matrix(
                        tie_line_indices=tie_indices,
                        tie_line_endpoint_buses=tie_endpoints,
                        der_bus_indices=pcc_hv_buses,
                    )
                )
                for j_pcc, bus in enumerate(pcc_hv_buses):
                    if bus in pcc_map_tie:
                        j_jac = pcc_map_tie.index(bus)
                        col = n_der + j_pcc
                        H[q_tie_row_start:q_tie_row_start + n_tie, col] = (
                            -dQtie_dQpcc[:, j_jac]
                        )

            # --- V_gen setpoint → Q_tie ---
            # Skips OOS generators (V_gen column zeroed below in any case).
            if is_gen_buses:
                dQtie_dVgen, _, gen_map_tie = (
                    self.sensitivities.compute_dQ_line_dVgen_matrix(
                        tie_line_indices=tie_indices,
                        tie_line_endpoint_buses=tie_endpoints,
                        gen_bus_indices_pp=is_gen_buses,
                    )
                )
                for k, gen_bus_pp in enumerate(gen_terminal_buses):
                    col = avr_start + k
                    if gen_bus_pp in gen_map_tie:
                        j_chg = gen_map_tie.index(gen_bus_pp)
                        H[q_tie_row_start:q_tie_row_start + n_tie, col] = (
                            dQtie_dVgen[:, j_chg]
                        )

            # --- 2W OLTC tap → Q_tie ---
            if is_oltc_indices:
                dQtie_dsOltc, _, _ = (
                    self.sensitivities.compute_dQ_line_2w_ds_matrix(
                        tie_line_indices=tie_indices,
                        tie_line_endpoint_buses=tie_endpoints,
                        chg_trafo_indices=is_oltc_indices,
                    )
                )
                is_pos_t = 0
                for k, t_idx in enumerate(self.config.oltc_trafo_indices):
                    if not self._oos_oltc_mask[k]:
                        target_col = n_der + n_pcc + n_gen + k
                        H[q_tie_row_start:q_tie_row_start + n_tie, target_col] = (
                            dQtie_dsOltc[:, is_pos_t]
                        )
                        is_pos_t += 1

            # --- Shunt state → Q_tie ---
            if self.config.shunt_bus_indices:
                dQtie_dShunt, _, _ = (
                    self.sensitivities.compute_dQ_line_dQ_shunt_matrix(
                        tie_line_indices=tie_indices,
                        tie_line_endpoint_buses=tie_endpoints,
                        shunt_bus_indices=_shunt_buses_for_sens,
                        shunt_q_steps_mvar=self.config.shunt_q_steps_mvar,
                    )
                )
                col_sh_start = n_der + n_pcc + n_gen + n_oltc
                H[q_tie_row_start:q_tie_row_start + n_tie,
                  col_sh_start:col_sh_start + n_shunt] = dQtie_dShunt

        # --- Feature B: PQ-mode V_gen column zeroing for saturated AVRs ---
        # Saturated gens have V_gen setpoint motion in the saturating
        # direction with no network-state effect.  The asymmetric u-bound
        # clamp in _compute_input_bounds allows only de-saturating motion;
        # zeroing the column here keeps the cached model consistent.
        if self.config.enable_saturation_mode:
            for k in range(n_gen):
                if self._sat_mode[k] != 0:
                    H[:, avr_start + k] = 0.0

        # --- Zero out columns for OOS generators and OLTCs ---
        # An out-of-service generator has no voltage control authority;
        # an OOS machine-transformer OLTC has no tap-change effect.
        # Zeroing the columns ensures the MIQP sees zero sensitivity
        # for these actuators, complementing the frozen bounds.
        for k in range(n_gen):
            if self._oos_gen_mask[k]:
                H[:, avr_start + k] = 0.0
        col_oltc_start = n_der_bus + n_pcc + n_gen
        for k in range(n_oltc):
            if self._oos_oltc_mask[k]:
                H[:, col_oltc_start + k] = 0.0

        # ── w-shift closed-loop transform on TSO DER columns ──
        # The DER actuator is the OFO-commanded ``q_set`` (with V_ref
        # reanchored to V_meas by the apply step).  Post-multiply the
        # DER columns of H by
        #     T' = (I + diag(K) · S_VQ)^{-1}
        # which maps the network-level ``∂y/∂Q`` to ``∂y/∂q_set`` under
        # the local Q(V) droop (mirrors the DSO controller).
        # Gated by ``apply_qv_h_transform`` (default False => bare H,
        # matching the dissertation's reference-anchored model).
        if n_der > 0 and self.config.apply_qv_h_transform:
            T_prime = self._compute_w_shift_transform_T_prime(der_bus_indices)
            if T_prime is not None:
                H[:, :n_der] = H[:, :n_der] @ T_prime

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

    def _compute_w_shift_transform_T_prime(
        self,
        der_bus_indices: List[int],
    ) -> Optional[NDArray[np.float64]]:
        """Return the w-shift closed-loop transform
        ``T' = (I + diag(K) · S_VQ)^{-1}`` of shape ``(n_der, n_der)``
        for the TSO DER block.

        The matrix is structurally identical to the earlier Q_cor
        transform (Soleimani & Van Cutsem, eq. 18) — under the
        vertical-shift + V_ref-reanchored local Q(V) loop, the
        differential sensitivity ``∂Q_realised / ∂q_set`` is the same
        ``(I + R·S_VQ)^{-1}`` matrix.

        Per-DER slope is read from ``net.sgen.qv_slope_pu`` if the
        column is present; otherwise the fallback
        ``self.config.qv_slope_pu`` is used.  Saturated DERs zero
        their K_diag entry so their column of ``T'`` collapses to
        identity.

        Returns ``None`` if S_VQ cannot be computed (e.g. DER bus is a
        PV bus the reduced Jacobian skips) or M is singular.  Callers
        fall back to identity (no transform) in that case.
        """
        from controller.der_qv_local_loop import (
            compute_w_shift_h_transform,
            _qv_capability,
        )
        import pandas as pd

        n_der = len(der_bus_indices)
        if n_der == 0:
            return None

        net = self.sensitivities.net
        meas = getattr(self, "_last_measurement", None)
        eps = self.config.qv_saturation_eps_mvar

        # Test / mock nets may lack ``sn_mva``.  Return None so the
        # caller falls back to identity.
        if "sn_mva" not in net.sgen.columns:
            return None

        # Build K_diag at DER-bus level.  When der_mapping is active each
        # entry of der_bus_indices is a unique DER bus and we sum the
        # per-DER S_n / slope; without mapping each entry is a per-DER
        # column so K_diag has one entry per DER.
        K_diag = np.zeros(n_der, dtype=np.float64)
        sgen_has_slope_col = "qv_slope_pu" in net.sgen.columns

        # Two paths: with / without der_mapping.
        if self.config.der_mapping is not None:
            # Map: der_bus_indices = unique_bus_indices.  Sum across DERs
            # at each unique bus.
            for d_idx, sgen_idx in enumerate(self.config.der_indices):
                sgen_idx = int(sgen_idx)
                sn = float(net.sgen.at[sgen_idx, "sn_mva"])
                if sgen_has_slope_col:
                    slope_v = net.sgen.at[sgen_idx, "qv_slope_pu"]
                    if pd.isna(slope_v) or float(slope_v) <= 0.0:
                        slope = self.config.qv_slope_pu
                    else:
                        slope = float(slope_v)
                else:
                    slope = self.config.qv_slope_pu
                if slope <= 0.0:
                    continue

                saturated = False
                if meas is not None and len(meas.der_q_mvar) > d_idx:
                    if "op_diagram" in net.sgen.columns:
                        od = net.sgen.at[sgen_idx, "op_diagram"]
                        op_diag = (
                            str(od) if od is not None and str(od) != "nan"
                            else "VDE-AR-N-4120-v2"
                        )
                    else:
                        op_diag = "VDE-AR-N-4120-v2"
                    p_mw = (
                        float(meas.der_p_mw[d_idx])
                        if d_idx < len(meas.der_p_mw) else 0.0
                    )
                    q_min, q_max = _qv_capability(sn, op_diag, p_mw)
                    q_act = float(meas.der_q_mvar[d_idx])
                    saturated = (q_act >= q_max - eps) or (q_act <= q_min + eps)

                if not saturated:
                    bus = int(net.sgen.at[sgen_idx, "bus"])
                    if bus in der_bus_indices:
                        b_pos = der_bus_indices.index(bus)
                        K_diag[b_pos] += sn / slope
        else:
            # No mapping: one column per DER, K_diag[i] = S_n,i / slope_i.
            for d_idx, sgen_idx in enumerate(self.config.der_indices):
                sgen_idx = int(sgen_idx)
                sn = float(net.sgen.at[sgen_idx, "sn_mva"])
                if sgen_has_slope_col:
                    slope_v = net.sgen.at[sgen_idx, "qv_slope_pu"]
                    if pd.isna(slope_v) or float(slope_v) <= 0.0:
                        slope = self.config.qv_slope_pu
                    else:
                        slope = float(slope_v)
                else:
                    slope = self.config.qv_slope_pu
                if slope <= 0.0:
                    continue

                saturated = False
                if meas is not None and len(meas.der_q_mvar) > d_idx:
                    if "op_diagram" in net.sgen.columns:
                        od = net.sgen.at[sgen_idx, "op_diagram"]
                        op_diag = (
                            str(od) if od is not None and str(od) != "nan"
                            else "VDE-AR-N-4120-v2"
                        )
                    else:
                        op_diag = "VDE-AR-N-4120-v2"
                    p_mw = (
                        float(meas.der_p_mw[d_idx])
                        if d_idx < len(meas.der_p_mw) else 0.0
                    )
                    q_min, q_max = _qv_capability(sn, op_diag, p_mw)
                    q_act = float(meas.der_q_mvar[d_idx])
                    saturated = (q_act >= q_max - eps) or (q_act <= q_min + eps)

                if not saturated and d_idx < n_der:
                    K_diag[d_idx] = sn / slope

        # S_VQ at the DER buses (units fix: divide by S_base).
        try:
            S_VQ_full, obs_map, der_map = (
                self.sensitivities.compute_dV_dQ_der(
                    der_bus_indices=der_bus_indices,
                    observation_bus_indices=der_bus_indices,
                )
            )
        except (ValueError, KeyError):
            return None

        if obs_map != der_bus_indices or der_map != der_bus_indices:
            try:
                obs_perm = [obs_map.index(b) for b in der_bus_indices]
                der_perm = [der_map.index(b) for b in der_bus_indices]
                S_VQ = S_VQ_full[np.ix_(obs_perm, der_perm)]
            except ValueError:
                return None
        else:
            S_VQ = S_VQ_full
        # ``compute_dV_dQ_der`` now returns S_VQ in pu_v/Mvar (divides by
        # ``sn_mva`` internally); no extra base scaling here.
        return compute_w_shift_h_transform(K_diag, S_VQ)

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

    def apply_qw_reset(self, measurement: Measurement) -> None:
        """Reset the DER block of ``_u_current`` to the measured Q per DER.

        Under the w-shift actuator (vertical shift + V_ref reanchoring),
        the OFO's effective per-step command is the *increment* sigma;
        the ``q_set`` value commanded to the plant is
        ``q_set = Q_meas + sigma``.  Implementing this as
        ``u_new = u_old + sigma`` (the generic OFO update in
        :meth:`BaseOFOController.step`) requires resetting ``u_old`` to
        ``Q_meas`` at the start of each step.  Mirrors
        :meth:`apply_avt_reset` for the PCC-Q block.

        Call from the runner immediately before :meth:`step`.  A no-op
        when the DER block is empty.
        """
        if self._u_current is None:
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

    def _classify_saturation_modes(
        self, measurement: Measurement,
    ) -> NDArray[np.int8]:
        """Update ``self._sat_mode`` in place using a hysteretic rule.

        A generator's mode transitions according to its measured Q and the
        current-operating-point PQ capability limits:

        * Free (0) → +1 (saturated upper) if  ``Q ≥ Q_max − eps_enter``
        * Free (0) → -1 (saturated lower) if  ``Q ≤ Q_min + eps_enter``
        * +1 → Free   if  ``Q < Q_max − eps_exit``  (ε_exit > ε_enter)
        * -1 → Free   if  ``Q > Q_min + eps_exit``

        Hysteresis (ε_exit > ε_enter) prevents mode chatter at the limit.
        The configured thresholds are validated in ``TSOControllerConfig``.

        Returns
        -------
        previous_modes : NDArray[np.int8]
            Copy of the mode vector *before* this call.  Useful for
            detecting transitions (e.g. to drive ``apply_avr_mode_reset``).
        """
        previous = self._sat_mode.copy()
        n_gen = len(self.config.gen_indices)
        if n_gen == 0:
            return previous
        if self.actuator_bounds.gen_params is None:
            return previous
        if (
            len(measurement.gen_p_mw) != n_gen
            or len(measurement.gen_q_mvar) != n_gen
            or len(measurement.gen_vm_pu) != n_gen
        ):
            return previous

        q_min, q_max = self.actuator_bounds.compute_gen_q_bounds(
            measurement.gen_p_mw, measurement.gen_vm_pu,
        )
        eps_enter = self.config.sat_eps_enter_mvar
        eps_exit = self.config.sat_eps_exit_mvar

        for k in range(n_gen):
            if self._oos_gen_mask[k]:
                # OOS generators stay in free mode — their column is zeroed
                # in H anyway so sat-mode has no meaning.
                self._sat_mode[k] = 0
                continue

            q = float(measurement.gen_q_mvar[k])
            mode = int(previous[k])

            if mode == 0:
                if q >= q_max[k] - eps_enter:
                    self._sat_mode[k] = +1
                elif q <= q_min[k] + eps_enter:
                    self._sat_mode[k] = -1
            elif mode == +1:
                if q < q_max[k] - eps_exit:
                    self._sat_mode[k] = 0
            elif mode == -1:
                if q > q_min[k] + eps_exit:
                    self._sat_mode[k] = 0

        return previous

    def apply_avr_mode_reset(
        self, measurement: Measurement, previous_modes: NDArray[np.int8],
    ) -> None:
        """Reset the cached V_gen command to the measured value on saturation onset.

        When a generator transitions from free mode (0) into a saturated
        mode (±1), the commanded V_gen stored in ``self._u_current`` may
        have diverged from the physically achieved terminal voltage while
        the AVR was rail-bound.  Reset the command to the measured value
        so the asymmetric bound clamp in ``_compute_input_bounds`` is
        applied relative to the achieved voltage (AVT-style).

        Transitions from saturated→free are left alone: the commanded
        voltage already equals the achieved voltage (because the clamp
        prevented moves in the saturating direction during saturation).

        Parameters
        ----------
        measurement : Measurement
            Current measurements (for ``gen_vm_pu``).
        previous_modes : NDArray[np.int8]
            Mode vector captured *before* :meth:`_classify_saturation_modes`
            was called this iteration.
        """
        if self._u_current is None:
            return
        n_gen = len(self.config.gen_indices)
        if n_gen == 0:
            return
        if len(measurement.gen_vm_pu) != n_gen:
            return

        mapping = self.config.der_mapping
        if mapping is not None:
            n_der = mapping.n_der
        else:
            n_der = len(self.config.der_indices)
        n_pcc = len(self.config.pcc_trafo_indices)
        avr_start = n_der + n_pcc

        for k in range(n_gen):
            prev = int(previous_modes[k])
            curr = int(self._sat_mode[k])
            if prev == 0 and curr != 0:
                # Free → saturated transition: realign u_current with achieved V.
                self._u_current[avr_start + k] = float(measurement.gen_vm_pu[k])

    def step(
        self,
        measurement: Measurement,
        *,
        sim_time_s: Optional[float] = None,
    ) -> ControllerOutput:
        """
        Execute one OFO iteration with voltage-dependent sensitivity updates.

        Pipeline (before delegating to :meth:`BaseOFOController.step`):

        1. Cache the measurement for use in ``_compute_input_bounds`` and
           ``_get_output_limits`` (capability-curve data).
        2. Ensure H is built (first call only).
        3. Rescale shunt columns for the new ``V²`` operating point.
        4. Achieved-Value Tracking: reset PCC-Q command to measured values.
        5. Classify AVR saturation modes with hysteresis (Feature B).
        6. If any generator transitioned from free to saturated, reset its
           commanded V_gen to the measured value so the asymmetric bound
           clamp is applied relative to the achieved voltage.
        7. If the mode vector changed, invalidate the sensitivity cache
           so the next H build reflects the new PQ-mode V_gen columns.

        Parameters
        ----------
        measurement : Measurement
            Current system measurements.
        sim_time_s : float, optional
            Wall-clock simulation time forwarded to
            :meth:`BaseOFOController.step` to drive the wall-clock OLTC
            cooldown (see :attr:`OFOParameters.int_cooldown_s`).
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
        # AVR saturation classification + mode-transition reset (Feature B, gated)
        if self.config.enable_saturation_mode:
            previous_modes = self._classify_saturation_modes(measurement)
            self.apply_avr_mode_reset(measurement, previous_modes)
            if np.any(previous_modes != self._sat_mode):
                # Rebuild H so V_gen columns reflect the new PQ-mode mask.
                self.invalidate_sensitivity_cache()
                self._build_sensitivity_matrix()

        return super().step(measurement, sim_time_s=sim_time_s)

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
