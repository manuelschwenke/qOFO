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
    int_cooldown:   int = 3

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
    g_z_voltage:   float = 1E9
    g_z_current:   float = 0.0
    g_z_interface: float = 0.0

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
    live_plot:              bool = False
    live_plot_load_balance: bool = False
    live_plot_hv_power:     bool = False

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
    qv_slope_pu:    float = 0.05
    warmup_s:       float = 0.0

    local_der_mode: str = "cos_phi_1"
    """HV-connected DER control mode in ``dso_mode='local'`` baseline.
    ``'cos_phi_1'`` -- unity power factor (Q=0 Mvar); no V-dependence (standard).
    ``'qv'``        -- linear Q(V) droop around ``qv_setpoint_pu`` with
                       slope ``qv_slope_pu`` (previous baseline).
    """
