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
    """Step-size penalty on the DSO DER block when
    ``use_qv_local_loop=False`` (legacy Stage-1 path).  Units: 1/Mvar²
    — penalises Q dispatch deltas in Mvar.  Ignored under Stage 2."""

    g_w_dso_der_vref: float = 1.0
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  After the refactor the DSO DER actuator is Q_cor (Mvar)
    so the step-size penalty is the existing ``g_w_dso_der`` (in
    1/Mvar²); a separate V_ref-coordinate penalty is no longer needed.

    Step-size penalty on the DSO DER block when
    ``use_qv_local_loop=True`` (Stage 2).  Units: 1/pu_v².

    **NOTE — empirically inert in the current controller setup.**
    A 7-OOM sweep (``g_w ∈ [1e-2, 1e5]``) with ``g_q=200`` shows
    bit-for-bit identical Q-iface tracking across the range.  The MIQP
    step is dominated by curvature ``g_q · (∂Q_iface/∂V_ref)² ≈ 2·10⁸``
    from the K(I+SK)^{-1} closed-loop transform; ``g_w`` only matters
    once it exceeds that curvature (so `>> 1e8`).  Below 1e8 it is
    effectively redundant.

    Default ``1.0`` is a placeholder — feel free to override for
    ablation studies, but the actual lever for tighter Q-tracking is
    ``qv_slope_pu`` (steeper droop = more Q per V_ref shift).  See
    ``tests/diag_stage2_steps.py`` for the validating sweep."""

    g_w_gridforming: float = 1.0e7
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  No grid-forming actuator block exists post-refactor.

    Step-size penalty on the V_gf actuator block (grid-forming
    converter ``vm_pu`` setpoint).  Same units as ``g_w_gen`` (1/pu_v²).

    Default ``1e7`` matches ``g_w_gen``.  Empirically the V_gf curvature
    contribution from voltage tracking is ``g_v · 1² ≈ 5·10⁵``; for the
    OFO regularisation to actually constrain V_gf moves, ``g_w_gridforming``
    must dominate this curvature — so values below ``1e6`` produce visible
    chatter on V_gf (TSO chases DSO transients), and ``1e7`` is the
    sweet spot.  Lower it carefully; raising it to e.g. ``1e8`` slows
    the converter response toward synch-machine timescales."""

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

    Used by ``experiments/003_M_DSO_CIGRE_2026.py`` to focus the
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

    Used by ``experiments/003_M_DSO_CIGRE_2026.py`` to drive the DSO_2
    controller toward ``[0, 0, 0]`` Mvar at its three interface
    transformers."""

    # NOTE: the refactor_v2 master switch for the q_mode plant model is
    # ``use_q_cor_actuator`` (defined further below alongside the rest of
    # the q_cor / actuator block).  When that flag is True the runner
    # tags every DER's q_mode and installs the per-row plant-side loops
    # in a single pass; the legacy classification / Stage-1 / Stage-2
    # paths run only when ``use_q_cor_actuator`` is False.

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
    g_z_q_gridforming: float = 1E2
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  No Q_gf output exists post-refactor (every converter is
    a ``pp.sgen`` whose Q is determined by the local q_mode loop).

    Soft-constraint penalty for the Q_gf output (grid-forming converter
    realised Q vs the STATCOM ±sqrt(S^2-P^2) Q-circle).  Mirrors
    ``g_z_q_gen`` for synch machines.  Default 1E3 — a gentle nudge that
    lets voltage tracking dominate when the converter exceeds its Q
    envelope, since the physical converter will clip Q anyway via
    ``enforce_q_lims=True`` in the plant PF."""
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
    run_stability_analysis:       bool = True
    stability_analysis_at_s:      float = 0.0
    sensitivity_update_interval:  int  = int(1E6)

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

    # -- DSO control mode ------------------------------------------------------
    dso_mode: str = "ofo"
    """DSO control mode for HV sub-networks.
    ``"ofo"``   -- Full OFO MIQP controllers (DSO tracks TSO Q setpoints).
    ``"local"`` -- Local controllers only: DiscreteTapControl for coupler OLTCs
                  and Q(V) droop for HV-connected DER.  No TSO->DSO coordination.
    """
    qv_setpoint_pu: float = 1.03
    qv_slope_pu:    float = 0.07
    warmup_s:       float = 0.0

    local_der_mode: str = "cos_phi_1"
    """HV-connected DER control mode in ``dso_mode='local'`` baseline.
    ``'cos_phi_1'`` -- unity power factor (Q=0 Mvar); no V-dependence (standard).
    ``'qv'``        -- linear Q(V) droop around ``qv_setpoint_pu`` with
                       slope ``qv_slope_pu`` (previous baseline).
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
    tso_qv_slope_pu: float = 0.07
    """Half-width of the Q(V) linear region (pu).  At V = setpoint+slope
    the windpark dispatches Q = q_min (full inductive); at V = setpoint-slope
    the windpark dispatches Q = q_max (full capacitive)."""

    # ------------------------------------------------------------------
    #  Q_cor actuator master switch (refactor_v2, Soleimani §III-B)
    # ------------------------------------------------------------------
    use_q_cor_actuator: bool = False
    """Master switch for the refactor_v2 Q_cor path.  When True, the
    runner:

    * calls :func:`network.ieee39.build.tag_der_q_modes` to populate
      ``net.sgen.q_mode`` / ``qv_slope_pu`` / ``qv_vref_pu`` /
      ``qv_deadband_pu`` / ``cosphi`` / ``cosphi_sign`` / ``q_cor_mvar``;
    * skips :func:`network.ieee39.build.apply_der_classification` so
      every DER stays as ``pp.sgen`` (no sgen→gen promotion);
    * installs :func:`controller.der_qv_local_loop.install_der_q_loops`
      (which dispatches QVLocalLoop or CosPhiConstLoop per ``q_mode``)
      instead of the legacy ``install_qv_local_loops``;
    * sets ``use_q_cor_actuator=True`` on every
      :class:`controller.tso_controller.TSOControllerConfig` and
      :class:`controller.dso_controller.DSOControllerConfig`,
      activating the H-matrix ``T' = (I + diag(K)·S_VQ)^{-1}`` transform
      on the DER columns;
    * passes ``use_q_cor_actuator=True`` to
      :func:`experiments.helpers.plant_io.apply_zone_tso_controls`,
      which writes the OFO output into ``net.sgen.q_cor_mvar``.

    Default ``False`` keeps the legacy grid-forming/grid-following
    classification path (with sgen→gen promotion and the Stage-2 Q-shim
    apply step).  Set to ``True`` to exercise the Soleimani-style
    Q_cor formulation end-to-end."""

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
    tso_qv_vref_pu: float = 1.00
    """Droop centre voltage for TSO DERs in qv mode."""

    dso_qv_vref_pu: float = 1.00
    """Droop centre voltage for DSO DERs in qv mode."""

    der_qv_vref_pu_overrides: Dict[int, float] = field(default_factory=dict)
    """Per-DER override of the qv droop centre voltage."""

    dso_qv_slope_pu: float = 0.07
    """Droop slope (pu_q/pu_v) for DSO DERs in qv mode.  TSO side uses
    the existing ``tso_qv_slope_pu`` field above."""

    der_qv_slope_pu_overrides: Dict[int, float] = field(default_factory=dict)
    """Per-DER override of the qv droop slope."""

    tso_qv_deadband_pu: float = 0.0
    """Half-width of the symmetric deadband around V_ref for TSO DERs.
    ``0.0`` disables the deadband (linear droop through V_ref)."""

    dso_qv_deadband_pu: float = 0.0
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

    # ------------------------------------------------------------------
    #  Legacy fields (slated for removal in refactor_v2 commit 7)
    # ------------------------------------------------------------------

    tso_command_relaxation_alpha: float = 1.0
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7 once all controllers stop reading it.  The Q_cor-driven
    OFO does not use command-side relaxation; closed-loop stability
    comes from the ``(I + R·S_VQ)^-1`` H transform plus the plant-side
    Q(V) damping.

    Step-size relaxation factor on the TSO OFO command update for
    *continuous* actuators only (Q_DER, Q_PCC, V_gen, V_gf).  Discrete
    actuators (OLTC, shunt) always step by their full integer increment
    because they cannot move "a fraction of a step".

    Update rule:
        u^(k+1)[continuous] = u^(k)[continuous] + alpha * w^(k)[continuous]
        u^(k+1)[integer]    = u^(k)[integer]    +         w^(k)[integer]

    where ``w^(k)`` is the OFO MIQP step direction.

    Default ``1.0`` reproduces the unrelaxed update of the PSCC 2026
    paper.  Values in [0.1, 0.7] add stability margin against modelling
    error (e.g. when the cached open-loop sensitivity does not include
    the response of an installed plant-side Q(V) droop, as in T-OFO).
    Lower values converge more slowly but tolerate larger model error.
    """

    # -- Per-DER grid-forming / grid-following classification ----------------
    der_mode_overrides: Dict[int, str] = field(default_factory=dict)
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  Replaced by :attr:`der_q_mode_overrides` (with values
    ``"qv"`` / ``"cosphi"`` instead of ``"grid_forming"`` /
    ``"grid_following"``).

    Optional per-DER override of the default grid-forming /
    grid-following classification.  Keys are *original* pandapower sgen
    indices (before any build-time promotion); values are
    ``"grid_forming"`` or ``"grid_following"``.

    Empty dict ⇒ defaults apply: every sgen registered in
    ``meta.tso_der_indices`` is treated as ``GRID_FORMING`` (modeled as
    ``pp.gen``, OFO commands ``vm_pu``); every sgen in
    ``meta.dso_der_indices`` is treated as ``GRID_FOLLOWING`` (modeled
    as ``pp.sgen``, OFO commands Q or V_ref).

    Use this to flip individual DERs without rewriting the network
    builder.  Example::

        der_mode_overrides = {17: "grid_following", 22: "grid_forming"}

    See :class:`core.der_classification.DERClassification` for semantics
    and :mod:`network.ieee39.build` for where the classification is
    applied to the pandapower model."""

    force_all_der_grid_following: bool = False
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  After the refactor every DER is a ``pp.sgen`` by
    construction (no sgen→gen promotion), so the flag has no role.

    Convenience override: when True, every TSO-side DER in
    ``meta.tso_der_indices`` is forced to ``"grid_following"`` at
    classification time, regardless of ``der_mode_overrides``.  Combined
    with ``use_qv_local_loop=False`` this reproduces the pre-Stage-1
    architecture (every controllable converter is a current-source
    ``pp.sgen`` with direct Q dispatch from the OFO) for ablation /
    baseline comparison."""

    use_qv_local_loop: bool = True
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  After the refactor the local Q(V) loop is the *only*
    plant model for ``q_mode == "qv"`` DERs (no Stage-1 fallback), and
    cos(phi) DERs install a sibling controller — both are dispatched
    automatically by ``install_der_q_loops`` based on
    ``net.sgen.q_mode``.  The master switch has no role.

    Stage-2 master switch — controls **plant model** only.  When True,
    every grid-following DSO DER sgen gets a local Q(V) controller
    installed in the simulated plant
    (:class:`controller.dso_qv_local_loop.QVLocalLoop`), so the
    converter response to V_bus follows ``Q = -k(V_bus - V_ref)``
    clipped to the PQ envelope.

    The way the OFO interacts with that plant model is then chosen by
    :attr:`qv_apply_mode` (V_ref-direct or Q+shim).  Default ``False``
    keeps the legacy Stage-1 direct-Q dispatch (no Q(V) plant model).

    Historical V_ref-direct details (``qv_apply_mode='v_ref'``): the
    DSO MIQP sensitivity matrix gains the closed-loop transform
    ``K (I + S_VQ K)^{-1}`` on the DER column block, and the realized
    Q is enforced as a soft output constraint via ``g_z_q_dso_der``.
    See ``MultiTSOConfig.qv_setpoint_pu`` and
    ``MultiTSOConfig.qv_slope_pu`` for the global Q(V) parameters;
    per-DER overrides live on
    :attr:`core.der_classification.DERClassification.qv_slope_pu` and
    ``qv_v_ref_init_pu``."""

    qv_apply_mode: str = "q_shim"
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  After the refactor the OFO writes ``q_cor_mvar``
    directly into ``net.sgen``; there is no V_ref-direct or Q-shim
    branch to choose between.

    How the DSO OFO drives the Q(V) plant model when
    ``use_qv_local_loop=True``.  Two modes:

    * ``"v_ref"`` — OFO commands V_ref directly.  The MIQP solves over
      ``V_ref ∈ [qv_v_ref_min_pu, qv_v_ref_max_pu]`` and uses the
      ``K(I+SK)⁻¹`` closed-loop sensitivity transform plus Q_realized
      soft output rows (``g_z_q_dso_der``).  Highest physical fidelity,
      but the V_ref bound + bus-voltage feedback caps the achievable Q
      at roughly half the converter capability — see
      ``tests/diag_dso_sgen_state.py`` for the validating diagnostic.

    * ``"q_shim"`` (default) — OFO solves the same direct-Q MIQP as
      Stage 1 (no K-transform, no Q_realized rows; ``g_w_dso_der`` and
      ``g_q`` recover their Stage-1 meaning).  The apply step inverts
      the droop locally::

          vm_pu_ref = V_bus_meas + Q_cmd / k     # k = S_n / qv_slope_pu

      The plant's Q(V) loop converges to ``Q ≈ Q_cmd``, so Q tracking
      quality matches Stage 1 while the plant model still represents a
      real grid-following converter (Q(V) loop runs every PF; comm-loss
      keeps the local loop holding the last V_ref).

    Ignored when ``use_qv_local_loop=False`` (Stage 1 path)."""

    qv_v_ref_min_pu: float = 0.90
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  V_ref is no longer an OFO variable (Q_cor is).

    Lower bound on the DSO V_ref OFO variable (only used when
    ``qv_apply_mode='v_ref'``).  Q+shim sets V_ref via the apply-step
    formula and uses Q-bounds on the OFO variable instead."""

    qv_v_ref_max_pu: float = 1.10
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.

    Upper bound on the DSO V_ref OFO variable (Stage 2)."""

    qv_local_damping: float = 0.5
    """Damping factor for the Q(V) local loop iteration.

    Closed-loop iteration contraction factor: ``|1 − damping·(1 + K·S_VQ)|``.
    With the corrected ``S_VQ`` units (``dV_pu / dQ_Mvar``, divided by
    the system base ``net.sn_mva``), typical HV S_VQ ≈ 1e-3 pu/Mvar.
    For ``K = S_n/0.07`` and S_n ~ 50 Mvar, ``K·S_VQ ≈ 0.7``, so
    damping=0.5 gives contraction ≈ ``|1 − 0.5·1.7| = 0.15`` —
    converges in ~10 iterations.  Larger damping (up to ~1.17) is also
    stable but risks overshoot for STATCOM-class units (S_n ~600 Mvar)
    where ``K·S_VQ`` can reach ~3-4."""

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

    g_z_q_dso_der: float = 1e2
    """**DEPRECATED — refactor_v2 commit 1.**  Slated for removal in
    commit 7.  The Q_realized soft-constraint row only existed for the
    V_ref-direct OFO path, which goes away.

    Soft-constraint penalty for the realized-Q output of each
    grid-following DSO DER under the V_ref-mode (Stage 2).  The
    capability bound ``[Q_min, Q_max]`` from the op_diagram is enforced
    as a soft output via this slack penalty so the DSO OFO sees explicit
    saturation pressure rather than relying on the K-zeroing trick alone."""

    # -- TSO-owned bipolar shunts at DSO tertiaries ----------------------------
    install_tso_tertiary_shunts: bool = True
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
