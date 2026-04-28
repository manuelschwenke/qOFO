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
from typing import List, Optional


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
    g_w_dso_oltc:   float = 1.0

    # -- Integer switching logic -----------------------------------------------
    int_max_step:   int = 1
    int_cooldown:   int = 9

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

    # -- Load pre-computed tuned params from a previous run --------------------
    load_tuned_params_path: Optional[str] = None
    """Path to a JSON file written by a previous run's delayed stability
    analysis.  When set the per-controller g_w values are warm-started from
    that file."""

    # -- Slack variable penalty (g_z) ------------------------------------------
    g_z_voltage:   float = 1E-12
    g_z_current:   float = 0.0
    g_z_interface: float = 0.0
    g_z_q_gen:     float = 1E-2
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

    tso_g_q_pcc: float = 1.0
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

    g_z_q_pcc: float = 1e2
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
