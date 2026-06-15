"""
configs/multi_tso_config.py
===========================
Central configuration dataclass for the multi-TSO / multi-DSO OFO experiment
(``experiments/000_M_TSO_M_DSO.py``).

Extracted from the runner to keep the experiment script short and to allow
different experiments to re-use or override subsets of the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Union


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
    dt_s:           float = 60.0
    n_total_s:      float = 60.0 * 60.0
    tso_period_s:   float = 60.0 * 3.0
    dso_period_s:   float = 60.0 * 1.0

    # -- Voltage setpoint ------------------------------------------------------
    v_setpoint_pu:  float = 1.03

    # -- Objective weights -----------------------------------------------------
    g_v:            float = 50000.0
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
    dso_g_qi:       float = 0.0
    dso_lambda_qi:  float = 0.9
    dso_q_integral_max_mvar: float = 50.0
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

    # NOTE: the DER plant-side actuator is the w-shift mode (vertical
    # shift of q_0 + V_ref reanchoring) — implemented unconditionally
    # by :func:`network.ieee39.build.tag_der_q_modes` and
    # :func:`controller.der_qv_local_loop.install_der_q_loops`.  The
    # earlier ``use_q_cor_actuator`` master switch was removed when the
    # Q_cor horizontal-shift path was retired.

    # -- Load pre-computed tuned params from a previous run --------------------
    load_tuned_params_path: Optional[str] = None
    """Path to a JSON file written by a previous run's delayed stability
    analysis.  When set the per-controller g_w values are warm-started from
    that file."""

    # -- Slack variable penalty (g_z) ------------------------------------------
    g_z_voltage:   float = 1E-12
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
    g_z_warmup_s:     float = 900.0
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
    result_dir: str = "results"

    # -- Live plot -------------------------------------------------------------
    live_plot_controller: bool = False
    """Enable Figure 1 — MULTI-TSO CONTROLLER live plot."""

    live_plot_cascade:    bool = False
    """Enable Figure 2 — CASCADE-DSO CONTROLLER live plot."""

    live_plot_system:     bool = False
    """Enable Figure 3 — SYSTEM POWER FLOW live plot."""

    live_plot_use_tex:    bool = False
    """When True, live plots enable ``text.usetex`` with a classicthesis-
    style mathpazo + eulervm preamble.  Requires a working LaTeX install
    and slows every redraw.  When False (default), rcParams select a
    Palatino-family serif font without any LaTeX dependency."""

    live_plot_show_line_currents: bool = False
    """When False, hide the TSO-line-currents tile on Figure 1 and the
    DSO-line-currents tile on Figure 2.  Useful to make more vertical
    room for the remaining tiles."""

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

    use_zonal_gen_dispatch: bool = True

    scenario: str = "wind_replace"
    """Network scenario registered in :mod:`network.ieee39.scenarios`.
    ``"base"``, ``"reduced_gen_z2"``, or ``"wind_replace"``."""

    # -- Contingencies ---------------------------------------------------------
    contingencies: List = field(default_factory=list)
    """List of :class:`experiments.records.ContingencyEvent` objects to inject
    during simulation.  Kept untyped to avoid a circular import with the
    experiment package."""

    distributed_slack: bool = True

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

    qv_local_max_step_frac: float = 1E8
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
    injects +q_mvar (reactor), ``step = -1`` injects −q_mvar (capacitor)."""

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

    g_z_q_pcc: float = 1e-2
    """Soft-constraint penalty for Q_PCC capability output bound.
    Mirrors ``g_z_q_gen``.  Engages when a shunt switch (or any other
    actuator move) would push physical Q_PCC outside the DSO-reported
    capability, providing the MIQP with a finite penalty for capability
    violation rather than a hard infeasibility.  Default 1e-2 is a
    gentle nudge; raise for tighter capability tracking."""

    tso_g_q_tie: float = 10.0
    """Q-tracking weight on tie-line reactive power outputs.  Scales the
    gradient contribution ``2 * tso_g_q_tie * (Q_tie_meas - Q_tie_set)^T
    * dQ_tie/du`` in each zone's TSO objective.

    Sign convention: Q_tie measured at the tie-line endpoint inside the
    zone, into the line.  Both zones touching a tie line independently
    penalise their own Q_tie reading toward the setpoint (no inter-zone
    real-time exchange — symmetric decentralised tracking).

    Tuning (validated on IEEE 39-bus 'wind_replace', 30-min smoke):
        0.0   — Phase A: rows present in H but no objective contribution
                (informational only).  No effect on actuator dispatch.
        1.0   — Recommended Phase B starting point.  Per-Mvar gradient
                strength matches the Q_PCC tracker (``tso_g_q_pcc``); on
                the smoke run, mean voltage error IMPROVED from 2.22 to
                1.72 mpu and |Q_tie(2,3)| reduced 27%.  Contraction LHS
                essentially unchanged from baseline.
        1e2…1e4 — Aggressive tracking; observed to overshoot —
                voltage error grows >10x and some tie pairs carry MORE
                reactive power as zones reroute Q via other ties.  Avoid
                without careful per-zone weight rebalancing.
        >=1e6 — Numerically unstable in our smoke; voltage band collapses."""

    g_z_q_tie: float = 0.0
    """Soft-constraint slack penalty for Q_tie output bound (Phase B
    optional).  Mirrors ``g_z_q_pcc``.  Default 0.0 = no slack-based
    bound enforcement on Q_tie."""

    tso_g_res_sg: float = 0.0
    """Explicit reactive-RESERVE weight for TS synchronous generators.
    Routed to ``TSOControllerConfig.g_res_sg`` at zone construction time.
    Adds ``tso_g_res_sg · Σ_i r_SG,i²`` to each zone's TSO objective, where
    ``r_SG,i = (Q_gen,i − Q_mid,i)/Q_half,i`` is the normalised distance of
    generator ``i``'s reactive output from the midpoint of its
    state-dependent PQ-capability band.  Penalising it keeps synchronous
    machines centred in their band → symmetric reserve in both directions.
    Default ``0.0`` = term off (reserve minimised only implicitly via the
    DSO cascade).  Toggle pattern mirrors ``tso_g_q_tie``."""

    tso_g_res_der: float = 0.0
    """Explicit reactive-RESERVE weight for TS-connected DER (continuous,
    Q-controlled sgens).  Routed to ``TSOControllerConfig.g_res_der``.
    Adds ``tso_g_res_der · Σ_i r_DER,i²`` with
    ``r_DER,i = (Q_DER,i − Q_mid,i)/Q_half,i`` over each DER's VDE-AR-N-4120
    capability band.  Kept separate from ``tso_g_res_sg`` so the operator
    can prefer one resource class over the other.  Default ``0.0`` (off).
    DSO-connected DER reserve is NOT covered here (it belongs to the DSO
    layer)."""

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
