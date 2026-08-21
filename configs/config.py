"""
configs/multi_tso_config.py
===========================
Central configuration dataclass for the multi-TSO / multi-DSO OFO experiment
(``experiments/000_M_TSO_M_DSO.py``).

Extracted from the runner to keep the experiment script short and to allow
different experiments to re-use or override subsets of the configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

_MEASUREMENT_NOISE_PROFILE_COMPONENTS: Mapping[str, Mapping[str, float]] = {
    "minimum": {
        "ehv_voltage_transformer": 0.001,
        "hv_voltage_transformer": 0.002,
        "voltage_meter": 0.001,
        "current_transformer": 0.002,
        "current_meter": 0.001,
        "power_meter_gain": 0.002,
        "power_phase_angle_deg": 0.1,
    },
    "conservative": {
        "ehv_voltage_transformer": 0.001,
        "hv_voltage_transformer": 0.002,
        "voltage_meter": 0.002,
        "current_transformer": 0.005,
        "current_meter": 0.002,
        "power_meter_gain": 0.005,
        # atan(0.02): a 2 % Q/P error at unity power factor.
        "power_phase_angle_deg": 1.1457628381751035,
    },
}

_MEASUREMENT_NOISE_COMPONENT_KEYS = frozenset(
    next(iter(_MEASUREMENT_NOISE_PROFILE_COMPONENTS.values()))
)


@dataclass
class MeasurementNoiseConfig:
    """Component-wise measurement-chain error model.

    Accuracy-class half-widths are split between a persistent channel bias
    and a small per-sample term; they are not interpreted as Gaussian sigma.
    Voltage uses VT/CVT plus PMD errors. Current uses CT plus PMD errors.
    P and Q share gain and phase errors through one complex-power channel.
    """

    enabled: bool = True
    profile: str = "minimum"
    seed: int = 20260717

    # 90 % persistent bias and 10 % per-sample variation by default. The two
    # absolute half-widths sum to the stated component accuracy-class bound.
    sample_noise_fraction: float = 0.10

    # Covers the IEEE-39 345-kV level as the 380-kV TSO analogue.
    ehv_voltage_threshold_kv: float = 220.0

    # Current accuracy below this fraction of CT rating uses an absolute floor.
    current_rating_floor: float = 0.20
    # Metadata for the lower validity range of PMD power accuracy classes.
    power_rating_floor: float = 0.02

    # Prefer an explicit CT primary rating; max_i_ka is only a compatibility
    # fallback for existing pandapower networks.
    ct_rating_columns: tuple[str, ...] = (
        "ct_primary_i_ka",
        "measurement_i_nom_ka",
        "rated_i_ka",
    )
    allow_line_rating_as_ct_fallback: bool = True

    # Half-width overrides are relative except power_phase_angle_deg.
    component_half_width_overrides: Dict[str, float] = field(
        default_factory=dict
    )

    apply_during_initialisation: bool = True
    clip_nonnegative_magnitudes: bool = True

    def profile_components(self) -> Dict[str, float]:
        """Return the selected component half-widths with overrides."""
        key = str(self.profile).strip().lower()
        if key not in _MEASUREMENT_NOISE_PROFILE_COMPONENTS:
            choices = ", ".join(sorted(_MEASUREMENT_NOISE_PROFILE_COMPONENTS))
            raise ValueError(
                f"measurement-noise profile must be one of: {choices}; "
                f"got {self.profile!r}"
            )
        unknown = set(self.component_half_width_overrides).difference(
            _MEASUREMENT_NOISE_COMPONENT_KEYS
        )
        if unknown:
            raise ValueError(
                "unknown measurement-noise component override(s): "
                + ", ".join(sorted(unknown))
            )
        out = dict(_MEASUREMENT_NOISE_PROFILE_COMPONENTS[key])
        out.update(
            {
                name: float(value)
                for name, value in self.component_half_width_overrides.items()
            }
        )
        return out

    def equivalent_bounds(self) -> Dict[str, float]:
        """Return variance-equivalent rectangular bounds for reporting."""
        import math

        c = self.profile_components()
        phase_rad = math.radians(c["power_phase_angle_deg"])
        return {
            "voltage_ehv": math.hypot(
                c["ehv_voltage_transformer"], c["voltage_meter"]
            ),
            "voltage_hv": math.hypot(
                c["hv_voltage_transformer"], c["voltage_meter"]
            ),
            "current": math.hypot(
                c["current_transformer"], c["current_meter"]
            ),
            "active_power_ehv": math.sqrt(
                c["ehv_voltage_transformer"] ** 2
                + c["current_transformer"] ** 2
                + c["power_meter_gain"] ** 2
            ),
            "active_power_hv": math.sqrt(
                c["hv_voltage_transformer"] ** 2
                + c["current_transformer"] ** 2
                + c["power_meter_gain"] ** 2
            ),
            "reactive_power_at_unity_pf": abs(math.tan(phase_rad)),
        }

    def relative_bounds(self) -> Dict[str, float]:
        """Backward-compatible EHV aggregate bounds for older callers."""
        b = self.equivalent_bounds()
        return {
            "voltage": b["voltage_ehv"],
            "current": b["current"],
            "active_power": b["active_power_ehv"],
            "reactive_power": b["reactive_power_at_unity_pf"],
        }

    def validate(self) -> None:
        """Validate the selected profile and range parameters."""
        components = self.profile_components()
        if not isinstance(self.seed, int):
            raise TypeError("measurement-noise seed must be an int")
        for name in (
            "sample_noise_fraction",
            "current_rating_floor",
            "power_rating_floor",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if float(self.ehv_voltage_threshold_kv) <= 0.0:
            raise ValueError("ehv_voltage_threshold_kv must be positive")
        if not self.ct_rating_columns:
            raise ValueError("ct_rating_columns must not be empty")
        for name, value in components.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        if components["power_phase_angle_deg"] >= 90.0:
            raise ValueError("power_phase_angle_deg must be less than 90")


@dataclass
class MultiTSOConfig:
    """
    Central configuration for the multi-TSO-DSO simulation.

    All parameters have sensible defaults suitable for a first test run on
    the IEEE 39-bus.  Adjust to explore stability margins.

    Timing
    ------
    dt_s : float
        Simulation timestep [s].  Should be the GCD of tso_period_s and
        dso_period_s (e.g. 60 s if TSO fires every 3 min, DSO every 1 min).
    n_total_s : float
        Total simulation duration [s].
    tso_period_s : float
        TSO control period [s].  Controllers fire every tso_period_s seconds.
    dso_period_s : float
        DSO control period [s].  DSOs fire more frequently than the TSO.

    Objective weights
    -----------------
    v_setpoint_pu : float
        Voltage setpoint for all monitored buses in all zones.
    g_v : float
        Voltage tracking weight (Q_obj diagonal for V rows).
    g_q : float
        Q-interface tracking weight for DSO controllers.

    OFO tuning
    ----------
    g_w_der : float
        Regularisation on TSO DER Q changes (prevents large single-step moves).
    g_w_gen : float
        Regularisation on generator AVR setpoint changes (very cautious by default).
    g_w_pcc : float
        Regularisation on PCC Q setpoint changes (Zone 2 -> DSO).
    g_w_tso_oltc : float
        Regularisation on machine-transformer OLTC tap changes.
    g_w_dso_der : float
        Regularisation on DSO DER Q changes.

    DSO parameters
    --------------
    dso_g_q : float
        DSO Q-interface tracking weight.
    dso_g_v : float
        DSO voltage tracking weight (secondary objective).

    Stability analysis
    ------------------
    run_stability_analysis : bool
        If True, compute and print the multi-zone stability analysis at t=0.
    sensitivity_update_interval : int
        Recompute cross-sensitivities H_ij every N TSO steps.
        1 = every step (most accurate but slower), 0 = only at initialization.

    Output
    ------
    verbose : int
        0 = silent, 1 = summary, 2 = full diagnostic per step.
    result_dir : str
        Directory for HDF5/JSON result files.  Relative to script location.
    """

    # -- Timing ----------------------------------------------------------------
    n_total_s:      float = 60.0 * 60.0
    tso_period_s:   float = 60.0 * 3.0
    dso_period_s:   float = 20.0 * 1.0
    dt_s:           float = dso_period_s

    # -- Measurement uncertainty ---------------------------------------------
    measurement_noise: MeasurementNoiseConfig = field(
        default_factory=MeasurementNoiseConfig
    )
    """Noise applied to analog feedback before it reaches any controller.

    Example::

        cfg.measurement_noise.enabled = True
        cfg.measurement_noise.profile = "minimum"  # or "conservative"

    The plant state and result recording remain noise-free; only controller,
    capability-message and coordination feedback is perturbed.
    """

    # -- Voltage setpoint ------------------------------------------------------
    v_setpoint_pu:  float = 1.03

    zone_v_setpoints_pu: Optional[Dict[int, float]] = None
    """Optional per-zone voltage-schedule override ``{zone_id: V_set_pu}``.
    When provided, each zone tracks its own ``V_set`` instead of the global
    ``v_setpoint_pu`` (zones not listed fall back to ``v_setpoint_pu``).  Used to
    create deliberate inter-zone voltage divergence (e.g. 1.05 / 1.03 / 1.01)."""

    # -- Voltage bounds (EHV hard limits) --------------------------------------
    v_min_pu: float = 0.90
    """Global hard lower bound on EHV bus voltage [p.u.], applied uniformly to
    every zone's ``TSOControllerConfig.v_min_pu`` unless overridden per zone by
    :attr:`zone_v_min_pu`.  Matches ``TSOControllerConfig.v_min_pu``'s own
    dataclass default exactly: the runner previously never passed ``v_min_pu``
    at all, silently relying on that same default, so leaving this field
    untouched reproduces legacy behaviour byte-for-byte."""

    v_max_pu: float = 1.10
    """Global hard upper bound on EHV bus voltage [p.u.].  See :attr:`v_min_pu`;
    matches ``TSOControllerConfig.v_max_pu``'s dataclass default (1.10)."""

    zone_v_min_pu: Optional[Dict[int, float]] = None
    """Optional per-zone override of the EHV hard lower voltage bound
    ``{zone_id: v_min_pu}`` (zones not listed fall back to :attr:`v_min_pu`).
    Routed to that zone's ``TSOControllerConfig.v_min_pu`` at zone
    construction time.  Typical use: tighten a zone's corridor (e.g.
    ``1.00``) when its voltage-tracking weight (:attr:`zone_g_v`) is turned
    off and the hard corridor becomes that zone's sole voltage discipline."""

    zone_v_max_pu: Optional[Dict[int, float]] = None
    """Optional per-zone override of the EHV hard upper voltage bound
    ``{zone_id: v_max_pu}``.  See :attr:`zone_v_min_pu`; routed to
    ``TSOControllerConfig.v_max_pu``."""

    # -- Objective weights -----------------------------------------------------
    g_v:            float = 50000.0
    zone_g_v: Optional[Dict[int, float]] = None
    """Optional per-zone voltage-tracking-weight override ``{zone_id: g_v}``
    (zones not listed fall back to the global :attr:`g_v`).  Routed to that
    zone's ``ZoneDefinition.g_v`` / ``TSOControllerConfig.g_v`` at zone
    construction time.  Set a zone's entry to ``0.0`` to remove that zone's
    voltage-schedule term from the objective entirely (``g_v=0`` combined
    with any ``v_setpoints_pu`` value gives an identical zero gradient
    contribution), while the zone keeps its hard ``[v_min_pu, v_max_pu]``
    corridor (:attr:`zone_v_min_pu` / :attr:`zone_v_max_pu`).  Used to build
    zones that instead prioritise loss minimisation
    (:attr:`zone_tso_g_loss`) or reserve centering (:attr:`zone_tso_g_res_sg`
    / :attr:`zone_tso_g_res_der`) without a competing voltage-tracking
    gradient."""
    g_q:            float = 1.0
    dso_g_v:        float = 50000.0

    central_dso_g_v: float = 20000.0
    """Voltage-tracking weight applied to the HV (110 kV / STS) buses by the
    single centralized controller (``control_scope='central'``, CIGRE V5).
    EHV/TN buses use the existing :attr:`g_v`.  The central controller builds
    a per-bus voltage weight vector that assigns ``g_v`` to every TN PQ bus and
    ``central_dso_g_v`` to every HV sub-network bus, so the two voltage levels
    can be balanced independently in the monolithic objective.  Ignored unless
    ``control_scope == 'central'``."""

    # -- G_w regularisation weights (TSO) --------------------------------------
    g_w_der:        float = 2.0
    g_w_gen:        float = 1e7
    g_w_pcc:        float = 2.0
    g_w_tso_oltc:   float = 1.0
    g_w_tso_shunt:  float = 10000.0
    """Regularisation penalty on TSO bipolar shunt step changes.  Set
    relatively low (~ ``g_w_tso_oltc``) so the discrete actuator can
    engage when continuous DERs cannot satisfy voltage / Q targets, but
    high enough to discourage chattering.  Used by
    :class:`controller.multi_tso_coordinator.ZoneDefinition.gw_diagonal`
    to fill the ``s_shunt`` block of the regularisation vector."""

    # -- DSO objective tuning --------------------------------------------------
    dso_gamma_oltc_q: float = 0.0

    # -- G_w regularisation weights (DSO) --------------------------------------
    g_w_dso_der:    float = 10.0
    """Step-size penalty on the DSO DER block (1/Mvar², units of Q_cor)."""

    g_w_dso_oltc:   float = 1.0

    # -- Adaptive g_w (paper Eq. 16, sign-only rule) --------------------------
    # When any of the per-class flags below is True, the corresponding
    # ``g_w_*`` entries become the *initial* values for an online
    # adapter (:class:`controller.g_w_adapter.GwAdapter`).  The shared
    # meta parameters control the multiplicative rates and clip box; α
    # remains fixed at 1.0 (no step-size adaptation).  See
    # :ref:`paper Zagorowska et al. (IFAC WC 2026)` and the project plan
    # at ``~/.claude/plans/c-users-manuel-schwenke-desktop-2604-12-soft-starfish.md``.
    adapt_g_w_der:        bool = False
    adapt_g_w_pcc:        bool = False
    adapt_g_w_gen:        bool = False
    """Enable online adaptation for TSO V_gen_set (AVR) g_w entries.
    NOTE: ``g_w_gen`` is pinned at ``1e7`` in
    :data:`tuning.parameters.FIXED_OVERRIDES`, so adapting it is only
    meaningful when ``g_w_gen`` is removed from the FIXED_OVERRIDES
    or the BO study is bypassed (e.g. running the experiment script
    directly).  Otherwise the BO overlay re-pins ``g_w_gen`` after
    every trial."""
    adapt_g_w_tso_oltc:   bool = False
    adapt_g_w_dso_der:    bool = False
    adapt_g_w_dso_oltc:   bool = False
    g_w_adapt_beta1:      float = 0.05
    """Multiplicative shrink rate of ``g_w`` in the descent regime
    (paper Eq. 16 β₁, in the S-space convention the *grow* rate of S).
    Must be in ``[0, 1)``."""
    g_w_adapt_beta2:      float = 0.10
    """Multiplicative grow rate of ``g_w`` in the anti-descent regime
    (paper Eq. 16 β₂).  Must be ``≥ 0``."""
    g_w_adapt_t_min:      float = 1e-2
    """Absolute floor on adapted ``g_w`` entries (clip after Eq. 16)."""
    g_w_adapt_t_max:      float = 1e6
    """Absolute ceiling on adapted ``g_w`` entries."""
    g_w_adapt_deadband_rel: float = 1e-6
    """Relative tolerance on ``|s_i| = |grad·w|`` below which no update
    is applied.  Scaled by ``max(||grad_f|| · ||w||, 1.0)`` so the floor
    matches numerical noise rather than the gradient magnitude.  Always
    a single shared scalar — has no per-class semantics, since the
    deadband is a property of the adapter as a whole."""

    # ---- Per-class overrides (paper Eq. 16, v1.1) ----------------------
    # When non-empty, the corresponding scalar above is treated as a
    # *fallback* and the per-class value is used for the listed class.
    # Class names match the keys returned by
    # :meth:`controller.tso_controller.TSOController._actuator_class_indices`
    # and :meth:`controller.dso_controller.DSOController._actuator_class_indices`
    # (``"der"``, ``"pcc"``, ``"gen"``, ``"tso_oltc"``, ``"tso_shunt"``,
    # ``"dso_der"``, ``"dso_oltc"``, ``"dso_shunt"``).  Classes not
    # listed fall back to the shared scalar above.  Values for classes
    # whose adapt-flag is ``False`` are silently ignored.  Typical use:
    # ``g_w_adapt_t_min_per_class={"der": 250, "pcc": 140, ...}`` so
    # each class clips at its own stability floor instead of forcing a
    # single shared value to satisfy the most conservative class.
    g_w_adapt_beta1_per_class: Dict[str, float] = field(default_factory=dict)
    g_w_adapt_beta2_per_class: Dict[str, float] = field(default_factory=dict)
    g_w_adapt_t_min_per_class: Dict[str, float] = field(default_factory=dict)
    g_w_adapt_t_max_per_class: Dict[str, float] = field(default_factory=dict)

    # -- Integer switching logic -----------------------------------------------
    int_max_step:   int = 1
    int_cooldown:   int = 6

    # -- Local-mode OLTC rate limit --------------------------------------------
    local_oltc_max_step_per_dt: int = 1
    """Maximum number of tap-position changes any local-mode
    ``DiscreteTapControl`` (machine 2W gen-trafo OLTCs and coupler 3W
    OLTCs) may execute per simulation timestep ``dt_s``.  pandapower's
    ``DiscreteTapControl`` iterates internally inside
    ``pp.runpp(run_control=True)`` and can move many positions in a
    single PF call when a disturbance is large; real OLTC mechanics
    impose an inter-tap delay (typically 30-60 s).  The runner snapshots
    every ``DiscreteTapControl``'s ``tap_pos`` at the start of each
    simulation step and, after every plant PF in the step, clamps the
    delta-from-snapshot to ``±local_oltc_max_step_per_dt``, re-running
    the PF with ``run_control=False`` if any tap was clamped.  Only
    applied when ``_local_dso`` or ``_local_tso`` is active; the OFO
    MIQP path uses ``int_max_step`` / ``int_cooldown`` for the same
    purpose."""

    oltc_cooldown_s: float = 30.0
    """Minimum wall-clock interval (simulation seconds) between
    consecutive tap changes on the same OLTC.  Applied to BOTH the
    local-mode ``DiscreteTapControl`` post-clamp (machine 2W gen-trafo
    and coupler 3W OLTCs) and the OFO MIQP integer cooldown (scoped to
    OLTC indices only — shunt switching still uses the iteration-based
    ``int_cooldown``).  Default 30 s reflects mechanical inter-tap delay
    of real OLTCs.  Set to 0.0 to disable the wall-clock cooldown
    entirely (the iteration-based ``int_cooldown`` and per-step
    ``local_oltc_max_step_per_dt`` clamp remain active)."""

    oltc_cooldown_s_mt: Optional[float] = None
    """Per-type override of ``oltc_cooldown_s`` for local-mode **machine
    2-winding (MT) gen-transformer OLTCs** (``net.trafo``).  ``None`` falls
    back to ``oltc_cooldown_s``.  Wall-clock seconds, so it is independent of
    ``dt_s`` (e.g. 180 -> at most one MT tap per 3 min)."""

    oltc_cooldown_s_nc: Optional[float] = None
    """Per-type override of ``oltc_cooldown_s`` for local-mode **coupler
    3-winding (NC) OLTCs** (``net.trafo3w``) at the TS--STS interface.
    ``None`` falls back to ``oltc_cooldown_s`` (e.g. 60 -> at most one NC tap
    per minute)."""

    # -- AVR saturation handling (Feature B) -----------------------------------
    enable_avr_saturation_mode: bool = False
    """When True, enable the hysteretic AVR saturation classifier, the
    asymmetric V_gen bound clamp, and the PQ-mode V_gen column zeroing.
    False (default) keeps V_gen as a plain continuous control."""

    # -- DSO OLTC initialisation -----------------------------------------------
    oltc_init_v_target_pu: float = 1.03
    dso_oltc_init_tol_pu: float = 0.01

    # -- Zone partitioning -----------------------------------------------------
    use_fixed_zones:    bool  = True

    # -- Single-DSO experiment selection (refactor_v2, used by 003_CIGRE_2026) --
    dso_ids_to_run: List[str] = field(default_factory=list)
    """Allow-list of DSO IDs (matching ``HVNetworkInfo.net_id`` such as
    ``"DSO_2"``) for which the runner should construct an OFO
    :class:`controller.dso_controller.DSOController`.

    Empty list (default) means "build for every DSO in
    ``meta.hv_networks``" — the legacy multi-DSO behaviour.  A non-empty
    list restricts OFO construction to the listed DSOs; the remaining
    HV sub-networks still exist in the plant network and exchange power
    through their coupling trafos, but they have no OFO controller and
    their DERs run only their plant-side Q(V) / cos(phi) loop.

    Used by ``experiments/003_S_DSO_CIGRE_2026.py`` to focus the
    optimisation on a single distribution system."""

    q_pcc_setpoint_schedule_per_dso: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )
    """Time-varying exogenous Q-setpoints at the TSO-DSO interface, keyed by
    DSO ID: ``[{"t_s": 0.0, "q_mvar": [0.0, 0.0, 0.0]}, ...]``.

    The entry in force at simulation time ``t`` is the last one whose ``t_s``
    is ``<= t``; entries need not be sorted. ``q_mvar`` carries one value per
    coupling 3W transformer, in ``HVNetworkInfo.coupling_trafo_indices`` order,
    exactly like :attr:`q_pcc_setpoints_mvar_per_dso`.

    Added 2026-08-19 for the Sec. 9.1 isolated-STS ``N_inner`` measurement
    (eq. 9.2), which steps the interface setpoint **across** the reported
    capability band and needs the subordinate loop to have settled at setpoint
    A before the step to B. With only a constant setpoint the measurement would
    start mid-transient at ``t = 0`` and the first iteration count would be an
    artefact of the initial condition rather than of the step.

    Takes precedence over the constant :attr:`q_pcc_setpoints_mvar_per_dso`
    for any DSO listed in both. Empty (default) changes nothing."""

    q_pcc_injection_with_ofo_parent: bool = False
    """Deliver the exogenous interface-Q setpoints while ``tso_mode == 'ofo'``.

    The original injection path is gated on ``tso_mode == 'local'``, which for
    a *measurement* of the subordinate loop is the wrong baseline: local Q(V)
    is a different controller from the OFO the thesis studies, and the TS plant
    then moves in response to the STS, confounding the very loop being isolated.

    With this flag and ``tso_period_s > n_total_s``, the supervisory OFO solves
    once at ``t = 0`` to establish the operating point and never revises it
    (``run_tso`` is ``(step == 1) or _is_period_hit(...)``), so the TS actuators
    hold while the exogenous setpoint drives the subordinate layer. That is the
    "parent silent" condition eq. (9.2) actually describes.

    Added 2026-08-19; default ``False`` leaves every existing configuration,
    including ``experiments/003_M_DSO_CIGRE_2026``, bit-identical."""

    q_pcc_setpoints_mvar_per_dso: Dict[str, List[float]] = field(
        default_factory=dict
    )
    """Exogenous Q-setpoints at the TSO–DSO interface 3W transformers,
    keyed by DSO ID.  Each value is a list of one Mvar setpoint per
    coupling 3W transformer of that DSO, in the same order as
    ``HVNetworkInfo.coupling_trafo_indices``.

    Only consulted when ``tso_mode == 'local'`` (i.e., no TSO OFO is
    generating ``SetpointMessage``\\s).  In that branch the runner
    synthesises a :class:`core.message.SetpointMessage` from this dict
    every step and delivers it to the named DSO controller via
    ``receive_setpoint``.  Empty dict (default) skips the injection.

    Used by ``experiments/003_S_DSO_CIGRE_2026.py`` to drive the DSO_2
    controller toward ``[0, 0, 0]`` Mvar at its three interface
    transformers."""

    # -- Load pre-computed tuned params from a previous run --------------------
    load_tuned_params_path: Optional[str] = None
    """Path to a JSON file written by a previous run's delayed stability
    analysis.  When set the per-controller g_w values are warm-started from
    that file."""

    # -- Slack variable penalty (g_z) ------------------------------------------
    g_z_voltage:   float = 1E9
    zone_g_z_voltage: Optional[Dict[int, float]] = None
    """Optional per-zone override of :attr:`g_z_voltage`
    ``{zone_id: g_z_voltage}`` (zones not listed fall back to the global
    ``g_z_voltage``).  Routed to that zone's voltage-slack weight at zone
    construction time.  The global default (``1E-12``) is a near-inert
    placeholder that relies on ``g_v`` tracking to keep voltage inside
    ``[v_min_pu, v_max_pu]`` in the ordinary case; a zone running
    :attr:`zone_g_v` ``= 0`` (e.g. a "bounds-only" strategy, see
    :attr:`zone_v_min_pu` / :attr:`zone_v_max_pu`) has nothing else pulling
    it back inside its corridor, so that zone's ``[v_min_pu, v_max_pu]`` is
    otherwise a purely nominal (non-binding) constraint.  Raise this zone's
    entry (e.g. to the order of its competing objective weight, such as
    :attr:`zone_tso_g_loss`) to make the corridor actually bind."""
    g_z_current:   float = 0.0
    g_z_interface: float = 0.0
    g_z_q_gen:     float = 1E2
    """Soft-constraint penalty for TSO Q_gen outputs (generator PQ capability).

    Kept as a gentle nudge only — voltage tracking must dominate when a
    generator exceeds its capability curve, because in the real system
    the AVR will physically limit Q_gen anyway.  Prior default 1E2 gave
    the Q_gen slack a gradient contribution roughly 1000x the voltage
    tracker at realistic operating points (V~0.9, Q_gen~350 Mvar with
    Q_max=300), which drove the TSO to ratchet machine-trafo OLTCs to
    saturation against voltage tracking.  See the tap-sensitivity direct-
    term fix in :meth:`sensitivity.jacobian.JacobianSensitivities.compute_dQgen_ds_2w_matrix`
    for the related sensitivity correction."""

    # -- g_z warmup ------------------------------------------------------------
    g_z_warmup_s:     float = 0.0
    g_z_warmup_value: float = 1E-12

    # -- Stability analysis ----------------------------------------------------
    run_stability_analysis:       bool = False
    stability_analysis_at_s:      float = 0.0
    sensitivity_update_interval:  int  = int(1E6)

    # -- Per-controller local-network sensitivity (Ward-style reduction) ------
    # When False (default), every TSO and DSO controller shares the
    # full-network ``JacobianSensitivities`` built from the entire plant
    # network (the historical behaviour: each controller sees the whole
    # interconnected Jacobian and its own block is sliced out).
    #
    # When True, ``run_multi_tso_dso`` builds a *reduced* pandapower
    # network per controller via :mod:`sensitivity.network_reduction` and
    # feeds that reduced net into a per-controller
    # ``JacobianSensitivities`` instance.  The reduction replaces every
    # boundary by an equivalent PQ injection from the cached operating
    # point (Ward equivalent):
    #
    # * **TSO zone:** tie-line far-end buses and 3W-coupler primary buses
    #   become PQ-load stubs; the slack lives on a zone gen; TSO-owned
    #   tertiary shunts are represented by *synthetic shunts on the 3W
    #   primary bus* (the tertiary bus itself is dropped along with the
    #   3W trafo and HV sub-network).
    # * **DSO sub-network:** the 3W-coupler primary bus becomes a virtual
    #   slack-gen pinned to ``V_cached`` (no explicit PQ load there — the
    #   slack auto-dispatches the cached HV flow).  The HV sub-network,
    #   3W trafo, tertiary bus, and TSO-owned tertiary shunt are all
    #   kept.
    #
    # The coordinator's cross-zone H_ij blocks are zeroed when
    # ``local_sensitivities_tso=True`` so the contraction diagnostic is
    # consistent with each TSO controller's restricted model (decoupled
    # decentralised view).
    local_sensitivities_tso: bool = False
    """If True, each TSO controller uses a Jacobian built from its own
    reduced zone net only (tie-line far-end + 3W primary boundaries as
    equivalent PQ loads).  See module docstring of
    :mod:`sensitivity.network_reduction`."""

    local_sensitivities_dso: bool = False
    """If True, each DSO controller uses a Jacobian built from its own
    reduced HV sub-network only (3W primary as virtual slack-gen).  See
    module docstring of :mod:`sensitivity.network_reduction`."""

    tie_boundary_equivalent: str = "pq"
    """How a neighbouring TSO area is condensed at the tie-line far-end
    stub of a reduced zone net.  Only read when
    ``local_sensitivities_tso=True``.

    * ``"pq"`` (default, historical): constant PQ load at the cached
      corridor flow -- infinite Thevenin impedance behind the boundary.
    * ``"pv"``: PV gen at the cached far-end voltage and active in-feed,
      reactive power free -- zero Thevenin impedance.
    * ``"z"``: constant admittance matched to the cached flow; falls back
      to PQ on stubs where the equivalent is a net source.
    * ``"thevenin"``: voltage source behind ``tie_thevenin_k`` times the
      tie line's own series impedance, on an auxiliary bus.  The only
      variant that leaves the far-end bus an ordinary PQ bus, hence the
      only one under which it has a voltage sensitivity of its own.

    ``"pq"`` and ``"pv"`` bracket the true finite-impedance equivalent, so
    the spread between them measures the modelling uncertainty this
    boundary choice introduces.  See
    :func:`sensitivity.network_reduction.build_tso_local_net`."""

    tie_thevenin_k: Any = 1.0
    """Boundary impedance for ``tie_boundary_equivalent="thevenin"``, as a
    multiple of the tie line's own series impedance.  ``k -> 0`` approaches
    the ``"pv"`` limit, ``k -> inf`` the ``"pq"`` limit.

    Either a float (one value for every corridor) or a dict keyed by
    ``(line_idx, far_end_bus)`` for per-corridor tuning; unlisted corridors
    fall back to
    :data:`sensitivity.network_reduction.THEVENIN_K_DEFAULT`.  The measured
    per-corridor set for the IEEE 39-bus case is
    :data:`sensitivity.network_reduction.THEVENIN_K_PER_CORRIDOR`."""

    zone_g_w_scale: Optional[Dict[int, float]] = None
    """Per-zone multiplier on that TSO controller's whole ``params.g_w``
    vector, applied once after controller construction and before the main
    loop.

    The MIQP step goes as ``H / g_w``, so this is the per-area loop-gain knob:
    below 1 makes that zone more aggressive.  It scales every actuator class in
    the zone by the same factor, leaving the tuned ratios between classes
    intact.  Zones absent from the mapping are left unscaled; ``None`` leaves
    all of them at their tuned values.

    Needed because a change of boundary equivalent changes the magnitude of
    ``H`` by a different factor in each area, so comparing two boundary models
    fairly requires re-gaining each area independently."""

    zone_g_w_class: Optional[Dict[int, Dict[str, float]]] = None
    """Per-zone, per-actuator-class **absolute** ``g_w`` override
    ``{zone_id: {class_name: g_w}}``, applied once after controller construction
    and before the main loop.

    The refinement level between the global ``g_w_<class>`` scalars and
    :attr:`zone_g_w_scale`.  The scalars carry no area information at all; the
    scale carries one number per area and so cannot express an area whose
    classes want to move in *opposite* directions — measured 2026-08-14 on the
    Thevenin baseline, TSO zone 1's analytic design wants ``g_w_der`` at 0.61x
    the global value but ``g_w_tso_oltc`` at 1.70x, a spread no single factor
    absorbs (residual 11.8x).  This field expresses that design directly.

    Class names are the keys of
    :meth:`controller.tso_controller.TSOController._actuator_class_indices`
    (``"der"``, ``"pcc"``, ``"gen"``, ``"tso_oltc"``, ``"tso_shunt"``).  Classes
    and zones not listed keep the value the global scalar gave them.  Values are
    absolute weights, not factors, and they REPLACE the class block of that
    zone's ``params.g_w`` vector; :attr:`zone_g_w_scale`, if also set, is applied
    *after* this and multiplies the result.

    Derive with ``python -m tuning_mc.stage_0_preconditioning --per-area``."""

    dso_g_w_class: Optional[Dict[str, Dict[str, float]]] = None
    """Per-DSO-area, per-actuator-class absolute ``g_w`` override
    ``{dso_id: {class_name: g_w}}`` — the DSO counterpart of
    :attr:`zone_g_w_class`, keyed by ``HVNetworkInfo.net_id`` (e.g. ``"DSO_2"``).

    Class names are the keys of
    :meth:`controller.dso_controller.DSOController._actuator_class_indices`
    (``"dso_der"``, ``"dso_oltc"``, ``"dso_shunt"``).  There is no
    ``dso_g_w_scale`` counterpart to :attr:`zone_g_w_scale`, so this is the only
    per-area ``g_w`` hook on the DSO layer."""

    dso_gamma_oltc_q_per_area: Optional[Dict[str, float]] = None
    """Per-DSO-area override of :attr:`dso_gamma_oltc_q`
    ``{dso_id: gamma}``, keyed by ``HVNetworkInfo.net_id``.  DSOs not listed
    keep the global scalar.  Applied to ``DSOControllerConfig.gamma_oltc_q``
    once after controller construction and before the main loop, next to
    :attr:`dso_g_v_per_area`.

    **This is the instrument for "the tap should react to interface-Q sooner",
    and it is the only one that does not disturb the DER block.**  ``g_q`` and
    :attr:`dso_g_q_per_area` are a single objective weight shared by every
    column of that controller, so raising them moves the continuous DSO-DER
    block by the same factor -- measured 2026-08-20 at a x20 ``g_q``, Stage 0's
    designed ``g_w_dso_der`` went 1172 -> 5190 against 1097 in service and the
    block oscillated.  ``gamma_oltc_q`` multiplies only the OLTC columns of
    ``dQ/du``, so the DER columns are untouched.

    Per-area rather than global because the right gamma is a property of the
    area's own weights: an area carrying the x20 voltage relief needs a large
    gain to reach a sane threshold, while an unrelieved one is already there
    and the same gain would drive it below its own tracking error.  Designed
    values for this plant are in
    ``experiments/run_multi_system_ofo.py::DSO_GAMMA_OLTC_Q_PER_AREA``.

    Not monotone -- see :attr:`dso_gamma_oltc_q` and
    ``controller.dso_controller.GAMMA_OLTC_Q_MAX``."""

    dso_g_q_per_area: Optional[Dict[str, float]] = None
    """Per-DSO-area override of the interface-Q tracking weight
    ``{dso_id: g_q}``, keyed by ``HVNetworkInfo.net_id`` like
    :attr:`dso_g_v_per_area`.  DSOs not listed keep the global :attr:`g_q`.
    Applied to ``DSOControllerConfig.g_q`` once after controller construction
    and before the main loop, in the same place as :attr:`dso_g_v_per_area`.

    **Exists only to complete the voltage relief, and is off unless asked for.**
    :func:`apply_dso_v_relief` scales an area's ``dso_g_v`` and its ``dso_oltc``
    step weight by the same factor so the OLTC *voltage* loop gain
    ``dso_g_v / g_w_dso_oltc`` is preserved.  Nothing then compensates ``g_q``,
    so that area's OLTC *interface-Q* commit threshold

        (g_w_oltc + ||a_oltc||^2) / (2 g_q |dQ_tr/ds|)

    rises by the full factor.  Measured 2026-08-20 at ``dso_gamma_oltc_q = 1``
    and a x20 relief: DSO_1/DSO_3 engage at 2.9-5.0 Mvar while DSO_2/DSO_4 need
    108-244 Mvar, against a measured interface-Q RMSE of ~6 Mvar -- i.e. the
    two relieved areas are Q-inert while the unrelieved ones are not.  That
    asymmetry is invisible in the voltage reading, where all four areas engage
    at 2.1-3.4 %.

    Scaling ``g_q`` by the same factor removes it.  **This is not a gauge
    transformation**: ``g_w_dso_der`` is deliberately left alone, so raising
    ``dso_g_v``, ``g_q`` and ``g_w_dso_oltc`` together makes that area's whole
    objective heavier relative to its *continuous* DER block -- the DER now
    takes larger steps for interface-Q error too, not only for voltage error.
    The original relief moved authority to the DER block for voltage only; with
    this field it does so for both channels.  State which is intended.

    Irrelevant at ``dso_gamma_oltc_q = 0``, where the OLTC has no Q gradient at
    all and the threshold above is infinite for any ``g_q``.

    See ``docs/daily_log/08_2026/2026-08-20_dso_oltc_inactivity_at_the_tuned_point.md``."""

    dso_g_v_per_area: Optional[Dict[str, float]] = None
    """Per-DSO-area override of the voltage-tracking weight
    ``{dso_id: dso_g_v}``, keyed by ``HVNetworkInfo.net_id`` like
    :attr:`dso_g_w_class`.  DSOs not listed keep the global :attr:`dso_g_v`.
    Applied to ``DSOControllerConfig.g_v`` once after controller construction
    and before the main loop, alongside :attr:`dso_g_w_class`.

    Needed because a weak, long-line HV network needs more voltage authority on
    its DER block than a compact one, and ``dso_g_v`` was global-only.  Measured
    2026-08-18 on DSO_4 (586 km, X = 1.84 p.u.): its internal voltage spread is
    0.147 p.u., 73 % of the whole [0.90, 1.10] band, which pins ``V_max`` on the
    upper bound for 58 % of the day.

    **Raise this together with that area's ``dso_oltc`` entry in
    :attr:`dso_g_w_class`, by the same factor.**  With
    ``dso_gamma_oltc_q = 0.0`` the DSO OLTC is driven *only* by the voltage
    gradient, so ``dso_g_v / g_w_dso_oltc`` is the OLTC loop gain; raising
    ``dso_g_v`` alone pushes the integer commit threshold into a limit cycle
    (measured: 50.5 tap reversals/h at x6.7, against 0.00 at baseline).  Holding
    the ratio moves the extra authority onto the *continuous* DER block, where
    it reshapes which DER injects — shrinking the spread at unchanged aggregate
    Q_DER — and the tap rate falls *below* baseline.

    See ``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md``."""

    numerical_h_closed_loop: bool = True
    """When ``numerical_h=True``, controls the perturbation mode of
    :func:`sensitivity.numerical_h.compute_numerical_h_tso` /
    ``compute_numerical_h_dso``:

    * ``True`` (default): every perturbation uses
      ``pp.runpp(run_control=True)`` so the plant-side Q(V) loops respond
      during finite difference.  The numerical DER column then captures
      ``∂y/∂q_set`` directly; V_gen / OLTC / shunt columns implicitly
      include the QV-loop reaction to those moves.
    * ``False``: perturbations use ``run_control=False`` (pure
      algebraic plant response) and the analytical ``T_prime`` transform
      is applied post-hoc to the DER columns only — mirroring the
      analytical builder's structure.  This is the right setting for an
      apples-to-apples test of the analytical H computation."""

    numerical_h: bool = False
    """If True, replace every controller's analytical H matrix by a
    *finite-difference* H computed via :mod:`sensitivity.numerical_h`
    (perturb each plant-side actuator, run ``pp.runpp(run_control=True)``,
    measure the response).  The numerical H is pinned to the controller's
    ``_H_cache`` and invalidation is suppressed, so the cached matrix
    survives every subsequent step.

    Used only by ``experiments/004b_REFRESH_PROOF.py`` to test whether
    the FULL-mode Q-tracking gap to LOCAL stems from a *computation
    bias* in the analytical Jacobian-based formulas (numerical H would
    perform differently) or purely from the boundary-modeling choice
    (numerical H matches analytical).

    Mutually exclusive with ``local_sensitivities_tso`` /
    ``local_sensitivities_dso`` (no-op when either is True — under local
    mode the controllers' H is built from a reduced net, not from the
    full plant)."""

    refresh_shared_jac_on_tso: bool = False
    """If True, the runner rebuilds the full-network ``shared_jac`` (and
    reassigns it to every TSO and DSO controller) on every TSO tick,
    immediately before the TSO MIQP runs.  Default ``False`` (the
    historical behaviour) keeps ``shared_jac`` frozen at the post-Phase-2
    operating point for the whole simulation.

    No-op when either ``local_sensitivities_tso`` or
    ``local_sensitivities_dso`` is True — under local-net mode the
    affected controllers do not use ``shared_jac`` (they hold their own
    reduced Jacobians, which the runner intentionally keeps frozen as
    the decentralised cached-sensitivity assumption).

    Used by ``experiments/004_LOCAL_VS_FULL_SENS.py`` to disambiguate
    whether the FULL-mode steady-state Q-tracking drift comes from
    cached-Jacobian staleness (set this True → drift should collapse)
    or from the structural AVR-stiffness mismatch at the 3W primary
    bus (set this True → drift persists)."""

    # -- Output ----------------------------------------------------------------
    verbose:    int = 0
    result_dir: str = str(Path(__file__).resolve().parents[1] / "results")

    # -- Live plot -------------------------------------------------------------
    live_plot_controller: bool = False
    """Enable Figure 1 — MULTI-TSO CONTROLLER live plot."""

    live_plot_cascade:    bool = False
    """Enable Figure 2 — CASCADE-DSO CONTROLLER live plot."""

    live_plot_system:     bool = False
    """Enable Figure 3 — SYSTEM POWER FLOW live plot."""

    live_plot_tracking:   bool = False
    """Enable Figure 4 — TRACKING ERRORS & RESERVES live plot.

    Six tiles: per-zone and system-wide TS voltage tracking RMS error,
    per-DSO TSO-DSO interface-Q tracking RMS error, system-wide tie-line-Q
    tracking RMS error, plus synchronous-generator and TSO-DER reactive-power
    reserve r_Q(P) (one line per machine / DER).  See
    :class:`visualisation.plot_tracking.TrackingLivePlotter`."""

    live_plot_use_tex:    bool = False
    """When True, live plots enable ``text.usetex`` with a classicthesis-
    style mathpazo + eulervm preamble.  Requires a working LaTeX install
    and slows every redraw.  When False (default), rcParams select a
    Palatino-family serif font without any LaTeX dependency."""

    live_plot_show_line_currents: bool = False
    """When False, hide the TSO-line-currents tile on Figure 1 and the
    DSO-line-currents tile on Figure 2.  Useful to make more vertical
    room for the remaining tiles."""

    live_plot_show_reserves: bool = True
    """When True, show the TSO reactive-reserve tile on Figure 1 (per-zone
    mean normalised reserve of synchronous machines (solid) and TSO DER
    (dashed), 0 = at a capability limit, 0.5 = mid-band)."""

    live_plot_show_tie_flows: bool = True
    """When True, show the TSO inter-zone tie-line reactive-flow tile on
    Figure 1.  Set False to free vertical room when tie lines are not of
    interest (e.g. single-zone studies)."""


    sbx_v_std_schedule_path: Optional[str] = None
    """Optional path to a planning-anchored contract-voltage schedule
    (SBX v3, STATUS_SBX.md 2026-07-08): JSON mapping ``"i-j"`` corridor
    keys to ordered ``[t_from_s, [v_a per line], [v_b per line]]``
    intervals in scenario time, as written by
    ``experiments/017_SBX_PLANNING.py``. When set, planning overrides
    the default schedule assembled from each TSO controller's intended
    terminal-voltage setpoints."""

    live_plot_sbx: bool = False
    """Enable the SBX-H live thesis figure (requires
    ``coordination_mode="sbx_h"`` or its internal alias ``"sbx"``).
    Per corridor it shows measured Q, the scheduled-voltage baseline Q_0,
    the Q deadband, paid Q_sup, measured/scheduled terminal voltages,
    hold/sag state strips, and cumulative bilateral payments. See
    :class:`visualisation.plot_sbx.SBXMechanismLivePlotter`."""

    live_plot_sbxv: bool = False
    """Enable the SBX-V live figure (requires ``coordination_mode="sbx_v"``
    or ``"sbxv"``), showing Normalbereich bands, requests, grants, metering,
    and cumulative remuneration."""

    live_plot_layout: str = "dual_screen"
    """Window layout for the three live figures.
    ``"thirds"``      -- three figures side-by-side, 1/3 primary screen each.
    ``"dual_screen"`` -- Figures 1 and 2 half/half on the primary screen;
                         Figure 3 full-screen on the secondary screen
                         (falls back to ``"thirds"`` if no secondary).
    """

    # -- Time-series profiles --------------------------------------------------
    use_profiles: bool = False
    start_time:   datetime = field(default_factory=lambda: datetime(2016, 6, 10, 0, 0))
    profiles_csv: str = ""

    # -- Provenance ------------------------------------------------------------
    sensitivity_reduction_rev: int = 2
    """Generation of the reduced-network (Ward) sensitivity construction.

    Bumped whenever a change alters the cached sensitivities, so runs from
    either side of the boundary cannot be silently mixed in an analysis. This
    is a *code*-version marker: nothing else in ``config.json`` distinguishes
    the generations, which is precisely how a superseded run entered a study
    once before (see the 2026-07-31 daily log, run 0080).

    * **rev 1** — everything up to and including the 2026-08-01 overnight
      matrix. The DSO reductions solved on the wrong power-flow branch
      (0.10–0.36 pu from the combined solution, tertiary buses collapsed to
      0.0), omitted the sub-network's own ``internal_aux`` buses, and pinned
      every non-slack boundary coupler at zero active power. TSO zones carried
      generator *setpoints* instead of actual dispatch under distributed slack.
    * **rev 2** — 2026-08-01. DSO reductions reproduce the combined solution
      exactly (0.000000 pu); TSO zone 0 improved up to 5x. Verified by
      ``tools/check_reduction_fidelity.py``.

    Measured effect on results: at 2016-01-05 08:00, delta = 0.005, the
    interface-Q metric moved 0.4410 -> 0.4929 Mvar (~12%).
    """

    # -- Exogenous load step (disturbance-rejection studies) -------------------
    load_step_time_s: Optional[float] = None
    """Simulation time [s] at which a step is applied to the load profiles.

    ``None`` disables the feature entirely, which is the default and leaves
    every existing experiment bit-for-bit unchanged.

    The step multiplies the profile columns named in ``load_step_columns`` by
    ``load_step_factor`` from this instant onward.  It is applied to the
    *interpolated* profile DataFrame, i.e. at ``dt_s`` resolution, so it is a
    genuine step.  Editing the source CSV instead would not work:
    ``core.profiles.load_profiles`` linearly interpolates the 15-minute source
    to ``dt_s``, which would smear any step into a 15-minute ramp.

    Because both plants consume the same DataFrame -- the static plant through
    ``apply_profiles`` and the RMS plant through ``Plant.apply_exogenous``
    (EvtLod) -- the step reaches both legs through their existing, supported
    paths.  It is therefore NOT a contingency: no element is switched, and it
    does not go through the ``contingencies`` machinery, which the RMS plant
    rejects (``experiments/runners/multi_tso_dso.py``: "non-static plant does
    not support: contingency events").
    """

    load_step_factor: float = 1.0
    """Multiplier applied to the stepped load columns (1.0 = no change)."""

    load_step_columns: Tuple[str, ...] = (
        "mv_rural_pload", "HS4_pload", "HS5_pload",
    )
    """Profile columns the step scales.  Default: every active-load profile,
    i.e. a system-wide load step.  Reactive-load columns are deliberately
    excluded -- ``mv_rural_qload`` is signed, so scaling it would deepen a
    capacitive injection rather than increase a load."""

    load_step_bus: Optional[int] = None
    """Bus at which to apply a LOCALISED additive load step, instead of scaling
    every profile column.

    A uniform step is a poor instrument for probing the DER dead zone: it buys
    local voltage deviation at the price of a system-wide power imbalance.
    Measured 2026-08-02 at 2016-01-05 08:00, reaching |ΔV| =
    0.02 pu at the DER terminals required roughly TRIPLING system load
    (4938 -> 8770 MW), which destabilised the cascade before the threshold was
    reached. The same deviation at the most sensitive bus (119) needs
    **43 MW** -- about 90x less disturbance.

    A localised step is also the realistic event (a large consumer connecting, a
    feeder transferring) and it creates a spatial gradient: nearby parks see a
    large deviation and distant ones little, so a single run samples
    |ΔV| across a wide range (0.046 pu spread at bus 119) and
    probes the threshold park-by-park rather than in aggregate.

    ``None`` disables it and leaves the multiplicative behaviour untouched.
    Requires ``load_step_time_s``; combine with ``load_step_delta_mw``.
    """

    load_step_delta_mw: float = 0.0
    """Additive active-power step [MW] applied at ``load_step_bus``.

    Realised by giving the target load its own synthetic profile column equal to
    its original profile plus ``delta / base_p_mw`` from the step instant. That
    reuses the existing exogenous path exactly, so the step reaches the static
    plant through ``apply_profiles`` and the RMS plant through
    ``Plant.apply_exogenous`` (EvtLod) with no new plumbing.
    """

    use_zonal_gen_dispatch: bool = True

    scenario: str = "rural_700"
    """Network scenario registered in :mod:`network.ieee39.scenarios`.
    ``"base_410"`` and ``"rural_700"`` share the transmission-side wind
    replacement and select 410 or 700 MW installed DER per DSO."""

    dso_der_scale: Dict[str, float] = field(default_factory=dict)
    """Per-DSO installed-DER multiplier, e.g. ``{"DSO_3": 2.0}``.

    Scenario multiplier, NOT builder state: ``constants.py`` keeps defining the
    symmetric 410/700 MW networks and this scales one underlay on top.  Scales
    ``sgen.p_mw``, ``sgen.base_p_mw`` and ``sgen.sn_mva`` of that DSO's DER, so
    both the profile playback (which reads ``base_p_mw``) and the Q capability
    (which reads ``sn_mva``) follow.

    Applied straight after ``add_hv_networks`` and before any power flow, so the
    ZIP load model, droop tagging and the Phase 1/2/3 operating-point init all
    see the scaled network -- and the RMS snapshot therefore carries it into
    PowerFactory through ``pf_sync``.  What was applied is recorded in
    ``net["dso_overrides"]``."""

    dso_load_p_scale: Dict[str, float] = field(default_factory=dict)
    """Per-DSO active-load multiplier, e.g. ``{"DSO_3": 2.0}``.

    Scales ``load.p_mw`` and ``load.base_p_mw`` and the recorded
    ``total_ref_p_mw``.  Reactive load is deliberately NOT scaled -- see
    ``dso_load_q_profile_base_mvar`` for that."""

    dso_load_q_profile_base_mvar: Dict[str, float] = field(
        default_factory=dict)
    """Per-DSO aggregate reactive-load profile base [Mvar], e.g.
    ``{"DSO_3": 500.0}``.  Zeroes the constant-Q rows and rescales the
    profiled rows so the aggregate becomes ``Q_load(t) = base * profile(t)``."""

    dso_line_std_type: Dict[str, str] = field(default_factory=dict)
    """Per-DSO HV conductor override, e.g.
    ``{"DSO_3": "490-AL1/64-ST1A 110.0"}``.  The persistent per-DSO types live
    in ``constants.DSO_HV_LINE_STD_TYPES``; this is for one-off studies."""

    # -- Plant load model ------------------------------------------------------
    load_model: str = "zip"
    """Voltage dependency of every plant load.  ``"zip"`` (default since the
    2026-07-17 RMS co-simulation decision): exact exponent image
    (kpu, kqu) = (1, 2) anchored at ``load_zip_anchor_vm_pu`` — realised as
    100 % constant-current P / 100 % constant-impedance Q with the anchor
    folded into the base values (:func:`network.ieee39.load_model.
    apply_zip_load_model`).  ``"const_pq"`` reproduces the pre-2026-07-17
    constant-power plant (for replaying older experiments)."""

    load_zip_anchor_vm_pu: float = 1.03
    """Voltage [pu] at which the SimBench-profile powers are served exactly
    under ``load_model = "zip"``.  1.03 pu (the network voltage setpoint)
    preserves the pre-ZIP power balance at the operating setpoint."""

    # -- Contingencies ---------------------------------------------------------
    contingencies: List = field(default_factory=list)
    """List of :class:`experiments.records.ContingencyEvent` objects to inject
    during simulation.  Kept untyped to avoid a circular import with the
    experiment package."""

    distributed_slack: bool = True

    der_q_capability_override_pu: Optional[float] = None
    """TEMPORARY diagnostic: force every DER's Q capability to a symmetric,
    P-independent box ``[-x*S_n, +x*S_n]``, overriding ``op_diagram``.

    ``None`` (default) = real diagrams (STATCOM circle / VDE-AR-N-4120 box).

    Set 2026-07-21 to 0.5 for the Gate-E co-simulation while establishing
    whether DER capability saturation causes the DSO's steady-state
    interface-Q offset and its ~120 s delayed response.  A per-park audit
    found 12 of 44 parks pinned at an individual limit while *aggregate*
    headroom still looked ample, and the saturation count tracked the
    per-DSO tracking error.

    **This is not a physical model and must be reverted.**  ±0.5 pu at rated
    P implies S = sqrt(1 + 0.25) ~= 1.12 pu, i.e. a converter oversized
    beyond what the scenario declares.  The runner prints a warning whenever
    it is active, and the value is recorded in the Gate-E summary so a run
    made under it cannot be mistaken for one made under real limits.

    Honoured identically by ``ActuatorBounds._compute_single_der_q_capability``
    (the controller's bounds) and ``der_qv_local_loop._qv_capability`` (the
    plant's clip).  Those two MUST agree -- a mismatch makes the controller
    optimise against a plant it does not have."""

    rms_profile_settle_s: float = 0.0
    """Seconds the RMS plant advances to reflect a new profile *before* the
    controllers read, per dispatch interval (0 = disabled).

    The static plant applies a profile and re-solves to steady state before
    the controllers measure; the RMS plant applies it as events that fire
    during the advance, so at 0 the controllers read the *pre-profile* state
    -- a one-interval lag that seeded a DSO_4 coupler runaway on the first
    profiles-on run (2026-07-22).  A non-zero value splits the interval: the
    plant advances this long first (profile events fire, state partly
    settles, controllers read it), then the remaining ``dt_s - settle`` after
    the control commands are issued.  The total per interval stays ``dt_s``,
    so the plant clock is unchanged.  Ignored by the static plant (its
    advance is an instant re-solve regardless)."""

    dispatch_slack_gen_v_ref: bool = False
    """Let the TSO OFO dispatch the AVR setpoint of the slack generator.

    Default ``False`` (changed 2026-07-21).  The IEEE 39-bus slack is the
    10 GVA 'Rest of USA/Canada' network equivalent: pandapower models it as a
    slack bus with a settable ``vm_pu``, so the OFO was commanding it, while
    the PowerFactory RMS model has no AVR block for it (network equivalents
    ship without controllers) and silently skipped every write.  That made
    the static plant strictly more capable than the RMS one.  A TSO does not
    dispatch a neighbouring interconnection's voltage, so the actuator is
    withdrawn from both plants rather than added to the RMS.

    The machine remains fully observed: its Q still enters the reserve term
    and the Q_gen soft constraint; only the V-ref column and bounds are
    frozen.  Set ``True`` to reproduce pre-2026-07-21 results."""

    enforce_q_lims_plant: bool = True
    """Pass ``enforce_q_lims=True`` to every plant-side ``pp.runpp`` in the
    main run loop so synchronous generators that hit their static
    ``[min_q_mvar, max_q_mvar]`` box are converted from PV to PQ for that
    PF iteration (mirroring real AVR Q saturation).  Without this, the PF
    is free to settle the gen at any Q the AVR voltage setpoint demands,
    which produces capability-envelope violations in the post-PF state
    visible in ``compare_gen_q_headroom`` plots — most prominently in the
    local-droop scenarios (L0/L1/L2 of ``002_M_TSO_M_DSO_COMPARE.py``)
    where neither the TSO nor the DSO MIQP is stepped and the
    ``g_z_q_gen`` slack penalty has no effect.  Set to ``False`` to
    reproduce the pre-fix behaviour (e.g. for ablation)."""

    # -- Voltage-stability / nose-curve reachability guard ---------------------
    enable_reachability_guard: bool = True
    """Run the modal voltage-stability reachability check
    (:mod:`analysis.reachability`) at every main-loop step.  The quasi-static
    power flow can converge to points on the lower (unstable) voltage branch
    that the dynamic system could never reach; the guard records the stability
    margin every step and aborts the run (``ReachabilityViolation``) at the
    first equilibrium that is not on the stable upper branch.  Set ``False`` to
    disable the guard entirely (e.g. for ablation or to reproduce pre-guard
    behaviour)."""

    reach_tau_sigma: float = 1e-6
    """Proximity threshold on the smallest singular value of the full
    power-flow Jacobian; the step is rejected when ``sigma_min(J) <
    reach_tau_sigma``.  See :func:`analysis.reachability.check_reachability`."""

    reach_tau_eig: float = 1e-6
    """Proximity threshold on the minimum real eigenvalue of the reduced Q-V
    Jacobian ``J_R``; the step is rejected when ``min(real(eig(J_R))) <=
    reach_tau_eig``.  See :func:`analysis.reachability.check_reachability`."""

    # -- Control scope (cascaded distributed vs single centralized) -----------
    control_scope: str = "cascaded"
    """Selects the controller topology of a run.

    ``"cascaded"`` (default) -- the distributed multi-TSO / multi-DSO OFO
        framework: one :class:`controller.tso_controller.TSOController` per
        zone + one :class:`controller.dso_controller.DSOController` per HV
        sub-network, coordinated by capability / setpoint messages.  This is
        the V1--V4 path; ``tso_mode`` / ``dso_mode`` further select OFO vs
        local control per layer.

    ``"central"`` -- a single :class:`controller.central_controller.CentralOFOController`
        that owns **all** actuators (every gen AVR, every TSO+DSO DER, all 2W
        machine OLTCs, all 3W coupler OLTCs, all TSO shunts) and observes
        **all** measurements (every TN+HV bus voltage, all lines, gen-Q
        capability) across the whole interconnection.  Used as the CIGRE V5
        best-case upper-bound reference against the distributed proposed
        controller (V4).  The 3-zone partition and per-HV-network metadata are
        retained purely as a recording lens for the paper figures; the
        per-zone TSO controllers, DSO controllers, coordinator cross-
        sensitivities, and capability/setpoint messaging are all bypassed.
        The central controller fires every ``central_period_s`` (default:
        every step); the voltage objective uses ``g_v`` for TN buses and
        ``central_dso_g_v`` for HV buses (no interface-Q / tie-Q tracking)."""

    central_period_s: Optional[float] = None
    """Control period [s] of the single centralized controller
    (``control_scope='central'``, CIGRE V5).  ``None`` (default) fires the
    controller every simulation step (``dt_s``) — the correct best-case
    cadence, since V5 replaces the fast STS-OFO layer (which runs every step in
    the cascaded variants) as well as the slow TS-OFO layer.  Set to a larger
    value (e.g. ``tso_period_s``) only to deliberately slow the reference.
    Ignored unless ``control_scope == 'central'``."""

    debug_central_curvature: bool = False
    """Read-only tuning probe (CIGRE V5).  When True and
    ``control_scope == 'central'``, the runner prints the closed-loop
    curvature spectrum of the single central controller once, right after
    ``central_controller.initialise(...)``: the eigenvalues of
    ``M = H_V G_w^{-1} H_V^T diag(g_v)`` (the per-tick voltage-error map
    ``e_{k+1} = (I - M) e_k``).  OFO is stable iff ``eig(M) ⊂ (0, 2)`` and
    well-damped for ``lambda_max(M) ≲ 1``.  Used to pick the global ``g_w``
    scale ``kappa`` that turns V5 into a valid upper bound.  No effect on the
    simulation itself (the probe only reads the cached H / g_v / g_w)."""

    # -- Tier-2 curvature-based g_w preconditioning ---------------------------
    precondition_g_w: bool = False
    """Derive the proximal weights ``g_w`` of the continuous actuator classes
    from the cached sensitivities instead of from BO/config, by (i)
    column-norm preconditioning (Zagorowska Eq. 16 diagonal scaling) and
    (ii) solving one global ``kappa`` so ``lambda_max(M) ==
    precondition_lambda_target``.  Runs once per TSO/DSO controller right
    after ``initialise(...)`` and before the loop.  Integer classes
    (OLTC/shunt) are left at their config values (their tuning primitive is
    switching frequency, not curvature).  Default ``False`` keeps the
    BO/config ``g_w`` untouched, so the existing tuning path is unaffected.
    See :mod:`controller.gw_precondition`."""

    precondition_lambda_target: float = 0.9
    """Target ``lambda_max(M)`` for :attr:`precondition_g_w` (well-damped at
    ``~0.9``; OFO stable for ``< 2``).  One scalar fixes the closed-loop
    gain once the per-class shape is set by the column norms."""

    precondition_lambda_target_tso: Optional[float] = None
    """Per-layer override of :attr:`precondition_lambda_target` for the TSO
    controllers; ``None`` falls back to the shared value.

    Needed because the layers — and in fact the individual zones — sit in
    qualitatively different regimes.  Measured at the hand-tuned operating point
    (2026-07-31), the *continuous* curvature is 1.775 in TSO zone 2 (which is
    PCC-dominated, so ``g_w_pcc`` genuinely is its loop gain) but 0.021 in zone 1
    (OLTC-dominated), so a single shared target cannot be meaningful for both."""

    precondition_lambda_target_dso: Optional[float] = None
    """Per-layer override of :attr:`precondition_lambda_target` for the DSO
    controllers; ``None`` falls back to the shared value.

    Note the DSO was never preconditioned at all before ``objective_curvature_
    inputs`` was added: :meth:`DSOController.voltage_curvature_inputs` returns
    ``None`` unless a voltage schedule is active with non-zero weight, and the
    DSO objective is dominated by interface-Q tracking (~500x in priority terms).
    Measured over the full objective block, the DSO continuous loop runs at
    ``lambda = 0.91-1.15``."""

    precondition_mode: str = "cap"
    """``'cap'`` (default) or ``'set'`` — see
    :func:`controller.gw_precondition.precondition_g_w`.

    ``'cap'`` only ever *adds* damping and is the right production behaviour: a
    safety net must not make a stable loop more aggressive.  ``'set'`` makes
    ``lambda_max`` track the target in both directions, which is required when
    the target is a **tuning coordinate** — under ``'cap'`` every target above
    the current ``lambda_max`` collapses to the same no-op, leaving the
    coordinate flat over much of its range (98 % of it, for TSO zone 1)."""

    precondition_lambda_scope: str = "all"
    """``'all'`` (default) or ``'preconditioned'`` — which columns
    :attr:`precondition_lambda_target` refers to.

    ``'all'`` targets the true ``lambda_max(M)``, the right choice for a
    production safety cap.  ``'preconditioned'`` targets only the columns being
    scaled, which is the right choice when the target is a *tuning coordinate*:
    ``M`` treats the integer OLTC columns as continuous per-tick moves, so their
    contribution is an upper bound rather than a real effect, and under
    ``'all'`` they can block the target outright (TSO zone 1 reports
    ``integer_dominated`` at ``lambda_floor = 1.085`` while its continuous loop
    sits at 0.021)."""

    precondition_class_scales: dict = field(default_factory=dict)
    """Optional ``{actuator_class: factor}`` multiplying that class's
    provisional preconditioned weight before ``kappa`` is solved.

    This is the *shape* knob, orthogonal to the gain: with ``kappa`` placing
    ``lambda_max`` on target, the ratios between classes are the only remaining
    freedom and express actuator *preference*.  Use gauge-fixed factors
    (geometric mean 1) so shape and gain stay independent.  Empty = the analytic
    column-norm preconditioner, i.e. no preference."""

    precondition_granularity: str = "class"
    """``'class'`` (one shared ``g_w`` per actuator class, directly
    comparable to a BO-tuned ``g_w_<class>``) or ``'column'`` (per-variable
    ``g_w``, the full Zagorowska-``S`` diagonal, best conditioning)."""

    precondition_floor_frac: float = 1e-6
    """Relative floor on the column energy ``||a_i||^2`` (vs the max over
    all columns) below which an actuator is treated as near-uncontrollable
    and its preconditioned ``g_w`` is floored — prevents a tiny-sensitivity
    column from collapsing ``g_w`` toward zero (= infinite gain)."""

    precondition_exclude_classes: tuple = ("gen",)
    """Continuous actuator classes to EXCLUDE from :attr:`precondition_g_w`
    (in addition to the always-excluded integer classes).  Default
    ``("gen",)``: the AVR voltage setpoint is a *direct* strong voltage
    actuator (column energy ~10^10× a DER's) whose ``g_w_gen`` the user
    already pins out of stability tuning (``FIXED_OVERRIDES``); folding it
    into the shared curvature scaling would let it dominate ``kappa``.
    Excluded classes keep their config ``g_w``."""

    # -- DSO control mode ------------------------------------------------------
    dso_mode: str = "ofo"
    """DSO control mode for HV sub-networks.
    ``"ofo"``   -- Full OFO MIQP controllers (DSO tracks TSO Q setpoints).
    ``"local"`` -- Local controllers only: DiscreteTapControl for coupler OLTCs
                  and Q(V) droop for HV-connected DER.  No TSO->DSO coordination.
    """
    warmup_s:       float = 0.0

    local_der_mode: str = "cos_phi_1"
    """HV-connected DER control mode in ``dso_mode='local'`` baseline,
    diagnostic flag only -- the plant-side q_mode loops drive Q under
    the Q_cor path; this label is printed for scenario provenance.
    ``'cos_phi_1'`` -- unity power factor (Q=0 Mvar); no V-dependence.
    ``'qv'``        -- linear Q(V) droop (parameters in ``dso_qv_*``).
    """

    # -- TSO local-control baseline (for comparison experiments) --------------
    tso_mode: str = "ofo"
    """TSO control mode for transmission-network reactive power.
    ``"ofo"``   -- Multi-zone OFO MIQP controllers (default).
    ``"local"`` -- Skip OFO step; apply local Q(V) or cos phi=1 to
                   TSO-connected windparks via pandapower
                   ``CharacteristicControl`` (Q(V)) or static Q=0
                   (cos phi=1).  Used by ``002_M_TSO_M_DSO_COMPARE.py``."""
    tso_local_mode: str = "qv"
    """TSO windpark local-control mode when ``tso_mode='local'``.
    ``'qv'``        -- linear Q(V) droop via CharacteristicControl.
    ``'cos_phi_1'`` -- unity power factor (Q=0)."""
    tso_qv_setpoint_pu: float = 1.03
    """Voltage setpoint of the Q(V) droop applied to TSO windparks."""
    tso_qv_slope_pu: float = 0.06
    """Half-width of the Q(V) linear region (pu).  At V = setpoint+slope
    the windpark dispatches Q = q_min (full inductive); at V = setpoint-slope
    the windpark dispatches Q = q_max (full capacitive)."""

    # ------------------------------------------------------------------
    #  q_mode hierarchy (refactor_v2, Soleimani §III-B)
    # ------------------------------------------------------------------
    #  Each DER's steady-state Q response is one of two modes:
    #    "qv"     -- piecewise-linear Q(V) droop with optional symmetric
    #                deadband; OFO commands Q_cor (Mvar) which shifts the
    #                droop curve via V_cor = Q_cor / R.
    #    "cosphi" -- fixed power factor: Q = sign * |P| * tan(acos(cosphi)).
    #                Excluded from the OFO action vector (not an actuator).
    #
    #  Hierarchy: per-DER override > DSO/TSO default.
    #  Keys in *_overrides dicts are pandapower sgen indices.
    # ------------------------------------------------------------------
    tso_q_mode: str = "qv"
    """Default ``q_mode`` for every TSO-connected DER (sgen indices in
    ``meta.tso_der_indices``).  ``"qv"`` or ``"cosphi"``."""

    dso_q_mode: str = "qv"
    """Default ``q_mode`` for every DSO-connected DER (sgen indices in
    ``meta.dso_der_indices``).  ``"qv"`` or ``"cosphi"``."""

    der_q_mode_overrides: Dict[int, str] = field(default_factory=dict)
    """Per-DER override of the level default.  Map ``sgen_idx → "qv" | "cosphi"``."""

    # -- qv parameters (used when q_mode == "qv") ----------------------
    tso_qv_vref_pu: float = 1.03
    """Droop centre voltage for TSO DERs in qv mode."""

    dso_qv_vref_pu: float = 1.03
    """Droop centre voltage for DSO DERs in qv mode."""

    der_qv_vref_pu_overrides: Dict[int, float] = field(default_factory=dict)
    """Per-DER override of the qv droop centre voltage."""

    dso_qv_slope_pu: float = 0.06
    """Droop slope (pu_q/pu_v) for DSO DERs in qv mode.  TSO side uses
    the existing ``tso_qv_slope_pu`` field above."""

    der_qv_slope_pu_overrides: Dict[int, float] = field(default_factory=dict)
    """Per-DER override of the qv droop slope."""

    tso_qv_deadband_pu: float = 0.01
    """Half-width of the symmetric deadband around V_ref for TSO DERs.
    ``0.0`` disables the deadband (linear droop through V_ref)."""

    dso_qv_deadband_pu: float = 0.01
    """Half-width of the symmetric deadband around V_ref for DSO DERs."""

    der_qv_deadband_pu_overrides: Dict[int, float] = field(default_factory=dict)
    """Per-DER override of the qv deadband half-width."""

    der_qv_deadband_override_pu: Optional[float] = None
    """Diagnostic blanket override of the DER Q(V) deadband [pu] applied to
    BOTH the static and RMS plants (see
    ``core.actuator_bounds.set_der_qv_deadband_override``).  ``0.0`` removes the
    dead zone so the droop is single-valued -- fixes the deadband-edge
    multi-equilibrium that makes the RMS DERs settle in a different droop basin
    from the static plant under profiles (2026-07-24).  ``None`` keeps each
    park's own deadband."""

    disable_qv_seed: bool = False
    """Diagnostic: skip ``seed_qv_equilibrium`` (the linear, deadband-ignoring
    warm-start of the static plant's QVLocalLoops, both at init and each
    interval).  The static side then relies on ``run_control`` alone, so it
    settles to the natural (deadband-respecting) droop fixed point instead of
    the seeded strong-droop one -- the hypothesis being that the seed is what
    pushes the static into a different droop basin from the RMS at the deadband
    edge (2026-07-24 option-2 test)."""

    seed_der_anchor_to_local_v: bool = False
    """Initialise every DER's Q(V) anchor ``qv_vref_anchor_pu`` to its local
    ``res_bus.vm_pu`` at init, instead of leaving it unset (which cold-starts the
    static ``QVLocalLoop`` at the nominal ``qv_vref_pu`` = 1.03 on the first
    profiled re-solve).  The RMS plant already anchors to the local voltage
    ``v_lf`` at init (``PowerFactoryPlant._anchor_qv_precontrollers``); without
    this the two plants droop about different anchors on the FIRST profiled step
    (static ~1.03, RMS ~1.02) -- a one-interval mismatch that seeds the DSO_4
    static-vs-RMS divergence.  Both plants re-anchor to the common local voltage
    from the first dispatch onward, so this only affects interval 1
    (2026-07-24 anchor-seed hypothesis)."""

    # -- cosphi parameters (used when q_mode == "cosphi") --------------
    tso_cosphi: float = 1.0
    """Power factor magnitude for TSO DERs in cosphi mode (1.0 ⇒ Q = 0)."""

    dso_cosphi: float = 1.0
    """Power factor magnitude for DSO DERs in cosphi mode."""

    der_cosphi_overrides: Dict[int, float] = field(default_factory=dict)
    """Per-DER override of the cosphi value."""

    tso_cosphi_sign: int = -1
    """Sign convention for cosphi-mode Q on TSO DERs.  ``+1`` =
    over-excited (Q injected, capacitive); ``-1`` = under-excited
    (Q absorbed, inductive — typical DE LV grid-code default)."""

    dso_cosphi_sign: int = -1
    """Sign convention for cosphi-mode Q on DSO DERs."""

    der_cosphi_sign_overrides: Dict[int, int] = field(default_factory=dict)
    """Per-DER override of the cosphi sign."""

    # -- Plant-side Q(V) loop convergence tolerance (per level) --------
    tso_qv_tol_mvar: float = 0.1
    """Convergence tolerance for the plant-side QVLocalLoop on TSO
    DERs (Mvar).  Transmission STATCOMs are large (S_n ≈ 600 Mvar) so
    very tight tolerances cost iterations without operational benefit;
    0.1 Mvar is a reasonable T-side accuracy."""

    dso_qv_tol_mvar: float = 0.01
    """Convergence tolerance for the plant-side QVLocalLoop on DSO
    DERs (Mvar).  DSO sgens are smaller (S_n ≈ 30–50 Mvar) and the
    OFO benefits from sub-Mvar accuracy at the interface; keep tight
    (0.01 Mvar)."""

    qv_local_damping: float = 0.1
    """Damping factor for the Q(V) local loop iteration.

    Per-DER contraction: ``|1 − damping·(1 + K·S_VQ)|`` where K = S_n/slope.
    Multi-DER coupling makes the effective spectral radius of (R · S_VQ)
    larger than the diagonal — empirically ~3–4× on the IEEE 39-bus
    with 44 coupled DERs, so the per-DER damping that converges in
    isolation can diverge under coupling.

    Default 0.05 keeps the system stable at 24-hour profile sweeps with
    DSO STATCOMs (R·S_VQ ≈ 0.7) and is paired with an additional
    automatic clamp to 0.03 for TSO STATCOMs (R·S_VQ ≈ 8) inside the
    runner's [3c-deferred] step.

    Was 0.5 in the pre-refactor_v2 config.  That value was tuned for
    the DSO-only STATCOM regime exercised by the legacy stage2 smoke
    (where TSO converters were promoted to gens and never installed
    QVLocalLoops), but is unstable under the refactor_v2 Q_cor path
    where every DER — TSO + DSO — runs a QVLocalLoop concurrently."""

    qv_local_max_step_frac: float = 1.0
    """Per-iteration step cap on the Q(V) damped update, as a fraction
    of S_n.  Default ``1.0`` (= no effective cap, since ``|target| ≤ S_n``
    by the capability clip).  Lower values further restrict per-iteration
    swing; raise to disable."""

    qv_local_tol_mvar: float = 0.1
    """Convergence tolerance for the Q(V) local loop (Mvar).

    Tight default (0.01 Mvar) so that small Q-shim commands (e.g. 0.5
    Mvar) get tracked precisely.  Was 1.0 Mvar in the early Stage 2
    debug period — fine for V_ref direct mode but caused the QV loop to
    sit in dead band for tiny Q+shim commands, locking Q_realized at the
    pre-step value regardless of V_ref change."""

    # -- TSO-owned bipolar shunts at DSO tertiaries ----------------------------
    install_tso_tertiary_shunts: bool = False
    """Install one bipolar 50 Mvar shunt per active DSO sub-network at
    the first 20 kV tertiary, switched by the TSO controller.  DSOs see
    it as a disturbance (``DSOControllerConfig.shunt_bus_indices`` stays
    ``[]``).  Set ``False`` to revert to the legacy no-shunt IEEE 39
    topology."""

    tso_tertiary_shunt_q_mvar: float = 50.0
    """Per-shunt rated reactive power per step at V = 1 pu [Mvar].
    Sign convention follows pandapower load convention: ``step = +1``
    injects +q_mvar (reactor), ``step = -1`` injects −q_mvar (capacitor).
    Used only by the legacy bipolar (``shunt_dispatch='miqp'``) build."""

    # -- Switched-shunt dispatch mode + MSC/MSR integrator ---------------------
    shunt_dispatch: str = "off"
    """How TSO-owned tertiary shunts are dispatched:
      * ``'off'``        — no tertiary shunts (legacy no-shunt topology).
      * ``'miqp'``       — legacy bipolar ±1 bank as an integer variable inside
                           the OLTC MIQP (``install_tso_tertiary_shunts`` build).
      * ``'integrator'`` — N-step MSC + MSR banks dispatched by the separate
                           integrating mechanism in
                           :mod:`controller.shunt_integrator`, OUTSIDE the MIQP.
    The MIQP carries shunt integers only in ``'miqp'`` mode (see the TSO
    controller's ``n_shunt_miqp`` gating)."""

    tso_shunt_kind: str = "msc_msr"
    """Network-build device class for ``shunt_dispatch='integrator'`` — passed
    to :func:`network.ieee39.hv_networks.add_hv_networks`.  ``'msc_msr'``
    installs one capacitor (MSC) and one reactor (MSR) bank per DSO tertiary."""

    tso_shunt_msc_n_levels: int = 4
    """Number of MSC (capacitor) steps per bank (lattice ℓ ∈ {0 … N})."""

    tso_shunt_msr_n_levels: int = 4
    """Number of MSR (reactor) steps per bank (lattice ℓ ∈ {0 … N})."""

    tso_shunt_msc_q_step_mvar: float = 50.0
    """MSC nameplate reactive power per step at V = 1 pu [Mvar] (magnitude)."""

    tso_shunt_msr_q_step_mvar: float = 50.0
    """MSR nameplate reactive power per step at V = 1 pu [Mvar] (magnitude)."""

    shunt_int_g_w: float = 1.0
    """Quadratic step weight for the integrator's continuous-relaxation update,
    per bank.  Consistent with the rest of the controller (alpha fixed = 1,
    step amplitude tuned via ``g_w``): the relaxation advances by
    ``g_H / (2 * g_w)`` each TSO iteration, so a SMALLER weight gives a LARGER
    step.  Tune so a sustained reactive pressure crosses the half-step +
    hysteresis band over several TSO iterations (slow bulk commit), not in one.
    NOTE: the boundary voltage sensitivity is small, so the gradient ``g_H`` is
    small — expect ``g_w`` well below 1 in practice."""

    shunt_int_delta_mvar: float = 5.0
    """Hysteresis half-width [Mvar].  Must satisfy
    ``0 < delta < q_step/2`` for both the MSC and MSR step sizes."""

    shunt_int_t_dwell_s: float = 300.0
    """Minimum dwell time between commits of the same bank [s]."""

    shunt_int_daily_budget: int = 1E3
    """Maximum commits per bank within any rolling 24 h window."""

    shunt_int_v_min_pu: float = 0.95
    """Lower HV-boundary voltage limit [p.u.] used by the integrator's
    overshoot feasibility guard."""

    shunt_int_v_max_pu: float = 1.10
    """Upper HV-boundary voltage limit [p.u.] used by the integrator's
    overshoot feasibility guard."""

    def validate_integrator_mode(self) -> None:
        """Fail-fast consistency check for the switched-shunt dispatch mode.

        Raises ``ValueError`` on any inconsistent or out-of-range setting; the
        per-bank numeric guards (``delta ∈ (0, q_step/2)`` etc.) are additionally
        enforced by :class:`controller.shunt_integrator.ShuntBankConfig`."""
        if self.shunt_dispatch not in ("off", "miqp", "integrator"):
            raise ValueError(
                f"shunt_dispatch must be 'off', 'miqp' or 'integrator', "
                f"got {self.shunt_dispatch!r}"
            )
        if self.shunt_dispatch != "integrator":
            return
        if not self.install_tso_tertiary_shunts:
            raise ValueError(
                "shunt_dispatch='integrator' requires "
                "install_tso_tertiary_shunts=True"
            )
        if self.tso_shunt_kind != "msc_msr":
            raise ValueError(
                "shunt_dispatch='integrator' requires tso_shunt_kind='msc_msr', "
                f"got {self.tso_shunt_kind!r}"
            )
        q_min = min(self.tso_shunt_msc_q_step_mvar, self.tso_shunt_msr_q_step_mvar)
        if not (0.0 < self.shunt_int_delta_mvar < 0.5 * q_min):
            raise ValueError(
                f"shunt_int_delta_mvar must lie in (0, q_step/2) = "
                f"(0, {0.5 * q_min}), got {self.shunt_int_delta_mvar}"
            )
        if self.shunt_int_g_w <= 0.0:
            raise ValueError(
                f"shunt_int_g_w must be > 0, got {self.shunt_int_g_w}"
            )
        if self.shunt_int_t_dwell_s < 0.0:
            raise ValueError(
                f"shunt_int_t_dwell_s must be >= 0, got {self.shunt_int_t_dwell_s}"
            )
        if self.shunt_int_daily_budget < 0:
            raise ValueError(
                f"shunt_int_daily_budget must be >= 0, got "
                f"{self.shunt_int_daily_budget}"
            )
        if self.shunt_int_v_min_pu >= self.shunt_int_v_max_pu:
            raise ValueError(
                f"shunt_int_v_min_pu ({self.shunt_int_v_min_pu}) must be < "
                f"shunt_int_v_max_pu ({self.shunt_int_v_max_pu})"
            )
        if min(self.tso_shunt_msc_n_levels, self.tso_shunt_msr_n_levels) < 1:
            raise ValueError("MSC/MSR n_levels must be >= 1")

    tso_g_q_pcc: float = 0.0
    """Q-tracking weight on the (re-enabled) Q_PCC output rows of the TSO
    H matrix.  Scales the gradient contribution of
    ``(Q_PCC_actual − Q_PCC_set)^2`` in the TSO objective.  Default 1.0
    (small) — TSO mildly prefers to cancel shunt-induced Q displacement
    at the interface via ``Q_PCC_set`` adjustments rather than overload
    the DSO.  Set ``0.0`` to keep the rows informational only."""

    tso_pcc_capability_on_output: bool = False
    """If True, apply DSO-reported PCC capability bounds to the physical
    ``Q_PCC`` output (so a shunt switch is counted against DSO
    capability).  If False, bounds remain on the control variable
    ``Q_PCC_set`` as in the legacy formulation.  Recommended ``True``
    when ``install_tso_tertiary_shunts`` is True."""

    g_z_q_pcc: float = 1e6
    """Soft-constraint penalty for Q_PCC capability output bound.
    Mirrors ``g_z_q_gen``.  Engages when a shunt switch (or any other
    actuator move) would push physical Q_PCC outside the DSO-reported
    capability, providing the MIQP with a finite penalty for capability
    violation rather than a hard infeasibility.  Default 1e-2 is a
    gentle nudge; raise for tighter capability tracking."""



    # ── Inter-organisational coordination mode ─────────────────────────────
    coordination_mode: str = "none"
    """Inter-organisational coordination mode.

    Supported values are ``"none"`` (autonomous multi-zone baseline),
    ``"sbx_h"``/``"sbx"`` (horizontal Scheduled Boundary Exchange), and
    ``"sbx_v"``/``"sbxv"`` (vertical TSO–DSO band/request coordination).
    The runner normalises the public package-name forms to the stable internal
    aliases so existing SBX result files remain readable."""


    sbx_config: Optional[object] = None
    """SBX configuration (an ``sbx_h.config.SBXConfig`` instance) used when
    ``coordination_mode="sbx_h"``.  ``None`` (default) builds
    ``SBXConfig(tso_period_s=tso_period_s)`` with the v6 defaults.  Typed
    loosely so this config module does not import the ``sbx_h`` package;
    the runner validates the instance type and that its ``tso_period_s``
    matches this config's."""

    sbx_support_intervals: Optional[Dict[tuple, list]] = None
    """SBX-H v6 planned support agreed IN ADVANCE: per corridor key
    ``(area_a, area_b)``, a list of windows ``(t_from_s, t_to_s,
    dv_a_pu, dv_b_pu)`` during which the named side holds its corridor
    terminals SHIFTED by dv relative to the base schedule (e.g.
    "+0.002 pu on B's terminals from minute 60 to 120" = the neighbour
    delivers a raised boundary voltage).  Applied to the built
    contracts via ``sbx_h.contract.with_planned_support``; the
    settlement automatically references the raised promise."""

    sbxv_config: Optional[object] = None
    """SBX-V configuration (an ``sbx_v.config.SBXVConfig`` instance) used
    when ``coordination_mode="sbxv"``.  ``None`` (default) builds
    ``SBXVConfig(tso_period_s=tso_period_s)`` with the plan-§8 v1
    defaults.  Typed loosely so this config module does not import the
    ``sbxv`` package; the runner validates the instance type and the
    ``tso_period_s`` match (STATUS_SBXV.md §0.3)."""

    sbx_warmup_s: float = 0.0
    """Optional activation delay [s] before SBX metering and settlement
    start. The voltage schedule itself comes from controller intent or an
    explicit planning schedule; it is never inferred from the plant
    snapshot. The default therefore starts at the first TSO tick."""

    tso_g_res_sg: float = 0.0
    """Explicit reactive-RESERVE weight for TS synchronous generators.
    Routed to ``TSOControllerConfig.g_res_sg`` at zone construction time.
    Adds ``tso_g_res_sg · Σ_i r_SG,i²`` to each zone's TSO objective, where
    ``r_SG,i = (Q_gen,i − Q_mid,i)/Q_half,i`` is the normalised distance of
    generator ``i``'s reactive output from the midpoint of its
    state-dependent PQ-capability band.  Penalising it keeps synchronous
    machines centred in their band → symmetric reserve in both directions.
    Default ``0.0`` = term off (reserve minimised only implicitly via the
    DSO cascade).  Per-zone overrides are supported below."""

    zone_tso_g_res_sg: Optional[Dict[int, float]] = None
    """Optional per-zone override of :attr:`tso_g_res_sg`
    ``{zone_id: g_res_sg}`` (zones not listed fall back to the global
    ``tso_g_res_sg``).  Routed to that zone's ``TSOControllerConfig.g_res_sg``
    at zone construction time.  Lets one zone run "pure reserve
    optimisation" (large weight) while others keep the term off (``0.0``,
    the global default).  Toggle pattern mirrors :attr:`zone_v_setpoints_pu`."""

    tso_g_res_der: float = 0.0
    """Explicit reactive-RESERVE weight for TS-connected DER (continuous,
    Q-controlled sgens).  Routed to ``TSOControllerConfig.g_res_der``.
    Adds ``tso_g_res_der · Σ_i r_DER,i²`` with
    ``r_DER,i = (Q_DER,i − Q_mid,i)/Q_half,i`` over each DER's VDE-AR-N-4120
    capability band.  Kept separate from ``tso_g_res_sg`` so the operator
    can prefer one resource class over the other.  Default ``0.0`` (off).
    DSO-connected DER reserve is NOT covered here (it belongs to the DSO
    layer)."""

    zone_tso_g_res_der: Optional[Dict[int, float]] = None
    """Optional per-zone override of :attr:`tso_g_res_der`
    ``{zone_id: g_res_der}``.  See :attr:`zone_tso_g_res_sg`; routed to
    ``TSOControllerConfig.g_res_der``."""

    tso_g_loss: float = 0.0
    """Active transmission-loss weight for the TSO objective.  Routed to
    ``TSOControllerConfig.g_loss`` at zone construction.  Adds
    ``tso_g_loss · Σ_ℓ c_ℓ·|I_ℓ|²`` per zone (form B — current-magnitude
    form), summed over each zone's monitored current lines
    ``current_line_indices`` with default coefficient ``c_ℓ = 3·R_ℓ`` (MW).
    The term reuses the cached ∂I/∂u current rows (no new sensitivity).
    Default ``0.0`` = loss term off (legacy).  Toggle pattern mirrors
    ``tso_g_res_sg``.  Requires each loss-counting zone to monitor at least
    one current line; tune relative to ``tso_g_v`` / ``g_w`` so loss does not
    overwhelm voltage tracking (loss is a tertiary objective — see
    :attr:`TSOControllerConfig.g_loss`)."""

    zone_tso_g_loss: Optional[Dict[int, float]] = None
    """Optional per-zone override of :attr:`tso_g_loss`
    ``{zone_id: g_loss}`` (zones not listed fall back to the global
    ``tso_g_loss``).  Routed to that zone's ``TSOControllerConfig.g_loss``.
    Lets one zone (e.g. a DSO-rich zone with a large EHV current-carrying
    corridor) prioritise transmission-loss minimisation while others keep
    the term off.  The "biting" scale is scenario-dependent (see this
    general objective-weight tuning pattern — validate
    on a smoke run before trusting a full-horizon study."""


    # ---- Adaptive g_w helpers ------------------------------------------------
    def tso_adapt_g_w_classes(self) -> tuple:
        """Tuple of TSO actuator-class names whose ``g_w`` entries are
        adapted online.  Names match
        :meth:`controller.tso_controller.TSOController._actuator_class_indices`
        (``"der"``, ``"pcc"``, ``"tso_oltc"``).
        """
        out = []
        if self.adapt_g_w_der:
            out.append("der")
        if self.adapt_g_w_pcc:
            out.append("pcc")
        if self.adapt_g_w_gen:
            out.append("gen")
        if self.adapt_g_w_tso_oltc:
            out.append("tso_oltc")
        return tuple(out)

    def dso_adapt_g_w_classes(self) -> tuple:
        """Tuple of DSO actuator-class names whose ``g_w`` entries are
        adapted online.  Names match
        :meth:`controller.dso_controller.DSOController._actuator_class_indices`
        (``"dso_der"``, ``"dso_oltc"``).
        """
        out = []
        if self.adapt_g_w_dso_der:
            out.append("dso_der")
        if self.adapt_g_w_dso_oltc:
            out.append("dso_oltc")
        return tuple(out)

    def make_g_w_adapt_meta(self):
        """Build the meta-parameter object(s) for the adapter from the
        scalar fields and any per-class overrides on this config.

        Returns
        -------
        :class:`controller.g_w_adapter.GwAdaptMeta`
            When all per-class override dicts are empty, returns a single
            shared meta — the v1 behaviour.
        ``Mapping[str, GwAdaptMeta]``
            When **any** per-class override dict is non-empty, returns
            a dict spanning every adapted class (TSO ∪ DSO) with the
            shared scalars filling in any per-class field that was not
            explicitly overridden.  Classes whose adapt-flag is ``True``
            but for which no per-class entry is provided still appear
            in the dict so the adapter receives a complete map.

        ``deadband_rel`` is always taken from the shared scalar (it has
        no per-class meaning).

        Imported lazily so the config module stays free of controller
        dependencies at import time.
        """
        from controller.g_w_adapter import GwAdaptMeta

        shared = GwAdaptMeta(
            beta1=self.g_w_adapt_beta1,
            beta2=self.g_w_adapt_beta2,
            t_min=self.g_w_adapt_t_min,
            t_max=self.g_w_adapt_t_max,
            deadband_rel=self.g_w_adapt_deadband_rel,
        )

        per_class_dicts = (
            self.g_w_adapt_beta1_per_class,
            self.g_w_adapt_beta2_per_class,
            self.g_w_adapt_t_min_per_class,
            self.g_w_adapt_t_max_per_class,
        )
        any_override = any(d for d in per_class_dicts)
        if not any_override:
            return shared

        # Span all adapted classes plus any class mentioned only in an
        # override (so a typo'd class name surfaces visibly via the
        # adapter rather than silently falling back to defaults).
        adapted = set(self.tso_adapt_g_w_classes()) | set(
            self.dso_adapt_g_w_classes()
        )
        mentioned: set = set()
        for d in per_class_dicts:
            mentioned.update(d.keys())
        classes = adapted | mentioned

        out: Dict[str, GwAdaptMeta] = {}
        for cls in classes:
            out[cls] = GwAdaptMeta(
                beta1=self.g_w_adapt_beta1_per_class.get(
                    cls, self.g_w_adapt_beta1,
                ),
                beta2=self.g_w_adapt_beta2_per_class.get(
                    cls, self.g_w_adapt_beta2,
                ),
                t_min=self.g_w_adapt_t_min_per_class.get(
                    cls, self.g_w_adapt_t_min,
                ),
                t_max=self.g_w_adapt_t_max_per_class.get(
                    cls, self.g_w_adapt_t_max,
                ),
                deadband_rel=self.g_w_adapt_deadband_rel,
            )
        return out


# ---------------------------------------------------------------------------
#  Per-DSO voltage relief
# ---------------------------------------------------------------------------

def _q_relief_factor(
    scale_q: "bool | float | Mapping[str, float]",
    dso_id: str,
    voltage_factor: float,
) -> "float | None":
    """The Q-leg factor for ``dso_id``, or ``None`` to leave ``g_q`` alone.

    ``scale_q`` accepts, in increasing specificity:

    ``False`` / ``None``
        No Q leg.  The default, and what every study before 2026-08-20 ran.
    ``True``
        The same factor as the voltage relief.  This is the setting that makes
        the OLTC's interface-Q commit threshold *identical* to the unrelieved
        one, because the factor on ``g_w_dso_oltc`` and the factor on ``g_q``
        cancel in ``(g_w_oltc + ||a||^2) / (2 g_q |dQ/ds|)``.
    a number
        That factor for every relieved area.  **Decouples the two legs on
        purpose.**  Measured 2026-08-20: at the full factor the relieved area's
        continuous DER block sees its objective raised by the same amount while
        ``g_w_dso_der`` does not move, and the block oscillates.  A smaller Q
        factor buys back damping at the cost of a proportionally higher tap
        commit threshold -- the threshold scales as ``voltage_factor /
        q_factor`` against the unrelieved value.
    a mapping
        Per-area factors; areas absent from it get no Q leg.

    Kept separate from :func:`apply_dso_v_relief` so the resolution rule is
    testable on its own and reads in one place.
    """
    if scale_q is None or scale_q is False:
        return None
    if scale_q is True:
        return float(voltage_factor)
    if isinstance(scale_q, Mapping):
        raw = scale_q.get(dso_id)
        return None if raw is None else float(raw)
    return float(scale_q)


def apply_dso_v_relief(
    cfg: "MultiTSOConfig",
    factors: Mapping[str, float],
    *,
    scale_q: "bool | float | Mapping[str, float]" = False,
) -> "MultiTSOConfig":
    """Give each listed DSO ``factor`` x more voltage authority at UNCHANGED
    OLTC loop gain.

    Lives here rather than in an experiment module because two independent
    callers need it: the experiment entry point
    (``experiments/run_multi_system_ofo.py``) and the Stage-1 config builder
    (``tuning_mc/stage_1_search.build_config``).  The tuning path is the reason
    the factor is applied to ``cfg``'s *own* weights instead of being written as
    absolute numbers: ``dso_g_v_ratio`` is a search coordinate, so ``dso_g_v``
    moves from trial to trial, and an absolute relief would silently change the
    OLTC loop gain ``dso_g_v / g_w_dso_oltc`` as the search walked.

    With ``dso_gamma_oltc_q = 0`` that OLTC is voltage-driven only, so the ratio
    *is* its loop gain: raising ``dso_g_v`` alone drives the integer tap into a
    limit cycle (measured 2026-08-18, 50.5 tap reversals/h at factor 6.7 against
    0.00 at baseline).  Scaling both by the same factor moves the extra
    authority onto the continuous DER block instead, where it reshapes which DER
    injects and shrinks the network's internal voltage spread.

    Merges into any existing :attr:`MultiTSOConfig.dso_g_w_class` /
    :attr:`MultiTSOConfig.dso_g_v_per_area` rather than replacing them, so a
    per-area design for the other areas survives.  Entries with factor ``1.0``
    are skipped; an empty or all-unity mapping returns ``cfg`` unchanged.

    ``scale_q`` (default ``False``) additionally scales that area's interface-Q
    weight into :attr:`MultiTSOConfig.dso_g_q_per_area` by the same factor.
    **Off by default so every existing caller -- including
    ``tuning_mc.stage_1_search.build_config`` and therefore the whole 0815 /
    stage1 campaign -- reproduces bit-for-bit.**

    It exists because the relief's own arithmetic creates an asymmetry it does
    not fix.  Holding ``dso_g_v / g_w_dso_oltc`` preserves the OLTC's *voltage*
    commit threshold, but the x-factor on ``g_w_dso_oltc`` is uncompensated in
    the *interface-Q* threshold
    ``(g_w_oltc + ||a_oltc||^2) / (2 g_q |dQ_tr/ds|)``, which therefore rises by
    the full factor.  Measured 2026-08-20 at ``dso_gamma_oltc_q = 1``, x20
    relief: DSO_2/DSO_4 need 108-244 Mvar to commit while DSO_1/DSO_3 need
    2.9-5.0, against ~6 Mvar of measured interface-Q RMSE.

    Two things to be explicit about before using it:

    * It is **inert at** ``dso_gamma_oltc_q = 0``, where the OLTC carries no Q
      gradient and no ``g_q`` makes the threshold finite.
    * It is **not a gauge rescaling**.  ``g_w_dso_der`` is deliberately not
      scaled, so raising ``dso_g_v``, ``g_q`` and ``g_w_dso_oltc`` together
      makes the area's whole objective heavier against its continuous DER
      block: that DER now takes larger steps for interface-Q error as well as
      for voltage error.  The relief was specified as *voltage* authority; with
      ``scale_q=True`` it becomes authority on both channels.

    See ``docs/daily_log/08_2026/2026-08-18_dso4_voltage_relief.md`` and
    ``docs/daily_log/08_2026/2026-08-20_dso_oltc_inactivity_at_the_tuned_point.md``.
    """
    active = {d: float(f) for d, f in (factors or {}).items() if float(f) != 1.0}
    if not active:
        return cfg

    gv = dict(cfg.dso_g_v_per_area or {})
    gw = {k: dict(v) for k, v in (cfg.dso_g_w_class or {}).items()}
    gq = dict(getattr(cfg, "dso_g_q_per_area", None) or {})

    for dso_id, factor in active.items():
        if not (factor > 0.0):
            raise ValueError(
                f"voltage-relief factor for {dso_id!r} must be > 0, got {factor!r}"
            )
        gv[dso_id] = float(cfg.dso_g_v) * factor
        # Base the OLTC weight on this area's own per-area value when a per-area
        # design is present, else on the global scalar -- otherwise the factor
        # would be measured against the wrong reference and the loop gain would
        # move.
        oltc_base = gw.get(dso_id, {}).get("dso_oltc", float(cfg.g_w_dso_oltc))
        gw.setdefault(dso_id, {})["dso_oltc"] = float(oltc_base) * factor
        q_factor = _q_relief_factor(scale_q, dso_id, factor)
        if q_factor is not None:
            if not (q_factor > 0.0):
                raise ValueError(
                    f"Q-relief factor for {dso_id!r} must be > 0, got "
                    f"{q_factor!r}")
            # Same base rule as the OLTC half: an already-relieved area's own
            # per-area value, else the global scalar.  Without this, applying
            # the relief twice would measure the factor against the wrong
            # reference -- the bug the dso_g_v half avoids by reading
            # cfg.dso_g_v, which a second call leaves untouched.
            q_base = gq.get(dso_id, float(cfg.g_q))
            gq[dso_id] = float(q_base) * q_factor

    return dataclasses.replace(
        cfg, dso_g_v_per_area=gv, dso_g_w_class=gw,
        dso_g_q_per_area=(gq or None) if gq else cfg.dso_g_q_per_area,
    )
