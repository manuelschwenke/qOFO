#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/003_S_DSO_CIGRE_2026.py
===================================
Single-DSO OFO experiment for the CIGRE 2026 paper (for you, Johannes)

Architecture
------------
* **Plant** -- Full IEEE 39-bus + every active HV sub-network (DSO_1..DSO_4)
  under the ``wind_replace`` scenario.  This is what the plant-side
  ``pp.runpp`` solves every step.
* **TSO layer** -- No OFO.  Every TSO-connected DER (the wind_replace
  STATCOMs) runs a plant-side ``QVLocalLoop`` with q_mode="qv",
  V_ref=1.03 pu, slope=0.06 pu, deadband=+-0.01 pu.  No zone partitioning is used in the
  control sense (the partitioning code still runs but no TSO controller
  is ever stepped).
* **DSO_2** -- One :class:`DSOController` is constructed for DSO_2 only
  (every other HV sub-network has no OFO).  Its actuator block on the
  DER side is ``Q_cor`` (refactor_Qcor_method, ``use_q_cor_actuator=True``):
  the OFO writes a correction term (sgen sign convention) that is
  communicated to each DER and added to the plant-side QVLocalLoop's
  droop response, so the DER's reactive output is
  ``Q_DER = Q_cor + K · (V_ref - V)`` (linear droop, deadband=0.01 in 003).
  The H matrix is therefore the closed-loop ``∂y/∂Q_cor``, obtained by
  post-multiplying the open-loop ``∂y/∂Q_DER`` block by
  ``T' = (I + diag(K) · S_VQ)^{-1}`` (Soleimani §III-B eq. 18).
* **DSO_2 outputs** -- Q at the HV side of the three coupling 3-winding
  transformers (``q_hv_mvar``) and DN voltage magnitudes.
* **DSO_2 actuators** -- ``Q_cor`` on every DSO_2 sgen plus the three
  coupling 3W trafo OLTC tap positions.
* **DSO_2 H matrix** -- Built from the full-grid Jacobian via
  :meth:`JacobianSensitivities.build_sensitivity_matrix_H` with DSO_2-only
  index lists, then post-multiplied by ``T'`` on the DER columns.
  Cross-coupling effects from the rest of the grid are implicit in
  the inverted full Jacobian.

Goal
----
Track ``Q_TSDS = [0, 0, 0]`` Mvar at the three coupling transformers
under the existing wind_replace profile + contingency timeline.  The
exogenous setpoint vector is wired through
``MultiTSOConfig.q_pcc_setpoints_mvar_per_dso``; under
``tso_mode='local'`` the runner synthesises a
:class:`core.message.SetpointMessage` from this dict every step and
delivers it to ``DSO_2``.

Notes for the sensitivity-corrector colleague
----------------------------------------------
This script exposes a read+write interface on the DSO_2 controller's
cached sensitivity matrix ``H`` so a Kalman filter or neural network
can run online: at every DSO step ``k`` the corrector reads ``H(k)``,
predicts ``H(k+1)``, and the prediction is used by the controller in
the next step.  Four pieces do the wiring (see below):

* :func:`get_dso_h_view`        -- read function: labeled, filtered
  view of ``H`` (the colleague's read access).
* :func:`install_h_corrector`   -- step wrapper: replaces
  ``dso_ctrl.step`` with a version that calls a user-supplied
  predictor after every step and writes its result into ``_H_cache``.
* :func:`install_pe_noise`      -- step wrapper: adds gaussian noise
  to the DER (Q_cor) actuator output of the OFO at every step
  (persistent excitation for sensitivity learning).  Toggled by
  ``cfg.dso_pe_noise_enabled`` in :func:`make_config`.
* :func:`_setup_h_predictor`    -- example startup function (passed to
  the runner as ``pre_loop_hook``); primes the cache, prints the view
  once, optionally installs PE noise, and installs the predictor.

Vector layouts (DSO_2 in this script, currents and shunts excluded):

``y`` -- output vector (rows of H), in this order::

    y = [ Q_HV @ coupling_3W_trafo[t]   for t in cfg.interface_trafo_indices ]   # Mvar, generator convention
        [ V[bus]                         for bus in cfg.voltage_bus_indices  ]   # p.u.

``u`` -- actuator vector (cols of H), in this order::

    u = [ Q_cor_DER[d]                   for d in cfg.der_indices            ]   # Mvar, sgen convention (>0 = injection); correction term added to the local Q(V) droop
        [ tap_position[t]                for t in cfg.oltc_trafo_indices     ]   # integer steps relative to neutral

The OLTC taps remain OFO actuators (columns of ``H``, entries of ``u``); the
DSO MIQP still moves them.  The online Kalman estimator, however, updates only
the DER (``Q_cor``) columns of ``H`` and leaves the OLTC columns at their
analytical value -- see ``_KalmanHPredictor``.

The DER's actual reactive output is ``Q_DER = Q_cor + K · (V_ref - V)``;
``Q_cor`` is what the OFO writes and what the controller's H is
linearized against.

``H = ∂y/∂u`` is the closed-loop linearization of the plant response
around the post-Phase-2 operating point.  Built from the inverse of
the full pandapower Jacobian projected onto DSO_2's actuator /
measurement sets, then post-multiplied on the DER columns by
``T' = (I + diag(K) · S_VQ)^{-1}`` to map the network-level
``∂y/∂Q_DER`` to ``∂y/∂Q_cor`` (see Soleimani §III-B eq. 18).
Cross-coupling from the rest of the grid is implicit in the inverted
full Jacobian.  Layout assumes ``use_q_cor_actuator=True`` and no
grid-forming DER (both hold in 003).

Predictor contract::

    predictor(dso_ctrl) -> Optional[np.ndarray]

is called once per DSO step, *after* ``dso_ctrl.step()`` returns.
* Return a numpy array of shape ``dso_ctrl._H_cache.shape`` (the FULL
  H, not the filtered view) to use it as ``H`` at step ``k+1``.
* Return ``None`` to leave ``_H_cache`` untouched (e.g. warm-up).

Per-step timing::

    step(k)  uses _H_cache (= H(k))
        -> step() returns
            -> predictor reads H(k) and any controller state
                -> predictor returns H(k+1)
                    -> _H_cache := H(k+1)
                        -> step(k+1) uses H(k+1)

For DSO_2 there are no shunts, so ``SensitivityUpdater.update()`` (which
runs inside ``step()`` and rescales shunt columns by V²) is a no-op
pass-through; the predictor's write is preserved through to the next
step.  When generalizing to DSO_1/3/4, beware that shunt columns may
get re-scaled inside ``step()`` and your predictor write would have to
account for that ordering.

Author: Manuel Schwenke, Johannes Ruppert
Date:   2026-05-05
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
import numpy as np

if TYPE_CHECKING:
    from controller.dso_controller import DSOController

# Type alias for the H-predictor a colleague wires in.  See
# :func:`install_h_corrector` for the contract.
HPredictor = Callable[["DSOController"], Optional[np.ndarray]]

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig  # noqa: E402
from experiments.helpers.records import MultiTSOIterationRecord  # noqa: E402
from experiments.helpers.plant_io import decouple_trafo3w_hv_with_slack  # noqa: E402
from experiments.runners import run_multi_tso_dso  # noqa: E402
from sensitivity.numerical_h import compute_numerical_h_dso  # noqa: E402

# ``002_M_TSO_M_DSO_COMPARE`` starts with a digit, so the import must go
# through importlib rather than a normal ``from ... import ...``.
_compare = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")
#make_002_base_config = _compare.make_config

# ---------------------------------------------------------------------------
#  Predictor selection
#  Set H_PREDICTOR_MODE to one of:
#    "identity"  -- _unity_multiply_predictor: returns H * 1.0 (wiring smoke-test)
#    "kalman"    -- _kalman_h_predictor:        Kalman-filter random-walk estimator
#    "rls"       -- _rls_h_predictor:           Per-row RLS with forgetting factor
#    "ann"       -- _ANN_h_predictor:           Keras ANN (requires trained model)
#    "numerical" -- _numerical_h_predictor:     ORACLE: true plant secant H,
#                   recomputed each DSO step by closed-loop finite difference on
#                   the live (decoupled) net.  Upper-bound baseline — the OFO
#                   acts with perfect knowledge of the realized sensitivity.
#
#  Set H_PREDICTOR_ROWS to one of:
#    "all"        -- replace the full H matrix with the predictor's output (default)
#    "q_trafo"    -- replace only the Q_trafo rows (∂Q_tr/∂u, the first n_q_trafo
#                    rows); all other rows (voltage, current) are kept from the
#                    current _H_cache unchanged
#    "q_trafo+v"  -- replace the Q_trafo rows AND the voltage rows (∂V/∂u, the
#                    next n_v = len(voltage_bus_indices) rows); the current
#                    (I_line) rows are kept from _H_cache.  Adds voltage-row
#                    online estimation on top of the interface-power rows.
#
#  Set FROZEN_OP_POINT = True to lock the plant at a real historical operating
#  point (FROZEN_OP_TIMESTAMP).  Profiles are enabled so the network is
#  initialised correctly at that timestamp; then cfg.frozen_at keeps the
#  profiles constant for the entire run.  Contingency events are suppressed.
#  PE noise is forced on so estimators remain persistently excited.
# ---------------------------------------------------------------------------
H_PREDICTOR_MODE: str = "kalman"
H_PREDICTOR_ROWS: str = "all"   # estimate Q_trafo + voltage rows (I_line rows kept analytical)

# ---------------------------------------------------------------------------
#  Kalman process-noise design (two-term split)
#  The predict-step process noise covariance is
#
#      Q_eff(k) = KALMAN_Q_SCALE · s²(k) · Q_excitation  +  Q_drift
#
#  with two physically distinct, additive terms:
#    Q_excitation -- INPUT-DRIVEN.  H is uncertain because the OFO is moving and
#                    exciting new directions.  Scaled by the normalised step
#                    energy s²(k) = ‖Δu_e / u_scale‖² / n_e, so it vanishes as
#                    Δu → 0 (no fictitious diffusion once the controller settles).
#    Q_drift      -- DISTURBANCE-DRIVEN.  H drifts due to exogenous load / profile
#                    / contingency changes that are independent of u.  ALWAYS on,
#                    so the filter never freezes once s² → 0 and keeps tracking.
#
#  Both matrices are estimated offline and loaded from kalman_matrices.npz (see
#  generate_kalman_matrices and experiments/_collect_train_mc.py).  KALMAN_Q_SCALE
#  is the only runtime knob: a multiplier on the excitation term (1.0 = as
#  trained).  This is the whole process-noise model — there is no forgetting
#  factor and no prior mean-reversion; Q_drift alone keeps the estimator alive.
# ---------------------------------------------------------------------------
KALMAN_Q_SCALE: float = 1.0

# ---------------------------------------------------------------------------
#  Measurement-noise R — input-dependent two-term model
#  The observation Δy ≈ H·Δu has a residual r = Δy − H·Δu.  In the Picallo
#  framework r is the 2nd-order Taylor remainder (∝ ‖Δu‖²) → variance ∝ ‖Δu‖⁴.
#  BUT a diagnostic on THIS plant (2026-06-19, instrumented excitation walk, no
#  tap motion) found r ∝ ‖Δu‖^0.99 — LINEAR, not quadratic: the dominant error is
#  a FIRST-ORDER model bias (the analytical H differs from the realised secant by
#  a roughly constant matrix B, so r ≈ B·Δu), and the curvature term is negligible.
#  A linear residual has variance ∝ ‖Δu‖², so:
#
#      R_eff(k) = R_sensor  +  KALMAN_R_MODEL_SCALE · ‖Δu(k)‖^KALMAN_R_DU_EXPONENT · R_model
#
#  with KALMAN_R_DU_EXPONENT = 2 (this plant; first-order model error).  Set it to
#  4 for a curvature-dominated plant (the Picallo case).
#    R_sensor -- INDEPENDENT per-channel SENSOR noise, Δu-independent floor
#                diag([σ_Q²]·n_q | [σ_V²]·n_v).  Also conditions R: the noise-free
#                MC residual cov is near rank-deficient (cross-row corr ≈0.99,
#                Mvar/p.u. scale split → cond ~1e10), so R_sensor + diagonal R_model
#                makes R_eff PD and well-conditioned for the gain.
#    R_model  -- the input-dependent MODEL error, ‖Δu‖^p-scaled.  R_model =
#                E[(r/‖Δu‖^{p/2})²] (diagonal) — normalise the residual by the
#                half-power so the size dependence is removed, then scale back up
#                by ‖Δu‖^p at runtime.  The filter then trusts small, near-linear
#                steps and distrusts large ones (secant ≠ Jacobian).  Note the
#                exponents: Q ∝ ‖Δu‖² (state increment), R_model ∝ ‖Δu‖² here
#                (linear model error; would be ‖Δu‖⁴ if curvature-dominated).
#  Set σ to the real SCADA spec (currents are dropped, so no σ_I).  Both terms are
#  estimated offline (generate_kalman_matrices / _collect_train_mc) and saved as
#  R_sensor, R_model; KALMAN_R_MODEL_SCALE is a runtime ablation knob (0 → fixed R).
# ---------------------------------------------------------------------------
KALMAN_R_SIGMA_Q:     float = 0.9    # interface-Q sensor noise std [Mvar]
KALMAN_R_SIGMA_V:     float = 0.01   # voltage     sensor noise std [p.u.]
KALMAN_R_DIAGONAL:    bool  = True   # drop noise-free-MC off-diagonal correlations
KALMAN_R_DU_EXPONENT: float = 2.0    # p in R_eff = R_sensor + scale·‖Δu‖^p·R_model (2 = first-order model error, this plant; 4 = Picallo curvature)
KALMAN_R_MODEL_SCALE: float = 0.0    # multiplier on the ‖Δu‖^p model-error term (0 = fixed R_sensor)
KALMAN_R_MODEL_MIN_DU: float = 0.1   # training guard: skip residuals with ‖Δu‖ < this when fitting R_model (avoids r/‖Δu‖^{p/2} blow-up)

# Step size for the "numerical" oracle's closed-loop finite difference.
# Matched to the deployment per-step OFO excitation (||du_DER|| ~= 3.4 Mvar) so
# the oracle returns the realized SECANT at the OFO scale, not the infinitesimal
# derivative.  The delta-sweep diagnostic confirmed the island is ~linear over
# [0.5, 3.4] Mvar, so the result is insensitive to the exact value in that band.
NUMERICAL_ORACLE_DELTA_MVAR: float = 3.4
FROZEN_OP_POINT:  bool = True
FROZEN_OP_TIMESTAMP: datetime = datetime(2016, 9, 7, 8, 0)  # real op-point to freeze at

# ---------------------------------------------------------------------------
#  Init-H bias injection
#  H_INIT_BIAS_STD > 0 multiplies each entry of the init H by an independent
#  log-normal(mean=1, std=H_INIT_BIAS_STD) factor — a multiplicative ±N%
#  perturbation of the analytical starting point.
#  H_INIT_BIAS_SEED is fixed so identity / kalman / ann runs all start from
#  exactly the same biased H, making comparisons directly meaningful.
#  Set H_INIT_BIAS_STD = 0.0 to disable (default).
# ---------------------------------------------------------------------------
H_INIT_BIAS_STD:  float = 0.3   # e.g. 0.10 for ±10% multiplicative noise; 0.0 = off
H_INIT_BIAS_SEED: int   = 64     # fixed seed → identical b
# ias for every run

# ---------------------------------------------------------------------------
#  DSO_2 interface mode
#  "slack"   -- replace the TN feed at each DSO_2 coupling 3W transformer's HV
#               side with a voltage-holding slack pinned to the operating point.
#               DSO_2 becomes its own island fed by 3 stiff (distributed) slacks,
#               so its online H estimation is NOT perturbed by TSO-side
#               transients (profiles, contingencies, TSO Q(V)).
#  "coupled" -- current behaviour: DSO_2 stays electrically coupled to the TSO
#               (use this to simulate the ANN in the present network state).
# ---------------------------------------------------------------------------
DSO2_INTERFACE_MODE: str = "slack"  # "slack" (decoupled, default) | "coupled"

# ---------------------------------------------------------------------------
#  Persistent-excitation (PE) mode toggle  (reversible)
#  Selects the DSO actuator excitation used for online H identification:
#    "white"      -- i.i.d. N(0, PE_WHITE_STD_MVAR²) per DER channel.  The
#                    original behaviour (isotropic in expectation).
#    "orthogonal" -- designed rotating orthonormal-basis probe of amplitude
#                    PE_AMPLITUDE_MVAR: each n_der-step cycle excites every input
#                    direction once, decorrelated from the OFO → balanced Fisher
#                    information → off-diagonal H columns identifiable far faster
#                    (see install_pe_noise).
#  Both settings give per-step ‖Δu‖≈1.0 Mvar and per-channel RMS≈0.3, so they
#  SHARE the same trained matrices (TARGET_STEP_MVAR=1.0) — flip freely, no
#  per-mode retrain.  Set PE_MODE="white" to reproduce the pre-orthogonal runs.
# ---------------------------------------------------------------------------
PE_MODE: str           = "white"  # "white" | "orthogonal"
PE_AMPLITUDE_MVAR: float = 1.0         # orthogonal-mode rotating-probe amplitude [Mvar]
PE_WHITE_STD_MVAR: float = 0.3         # white-mode per-channel noise std [Mvar]

# The Kalman estimator updates ONLY the DER (Q_cor) columns of H; the OLTC tap
# columns are held at their analytical (cache) value.  The OLTC taps remain full
# OFO actuators (columns of H, entries of u) — the DSO MIQP still moves them —
# they are simply a KNOWN control whose sensitivity the H-builder recomputes
# analytically at the current tap, so there is nothing to learn online.  The
# Kalman state is therefore vec(H[:, :n_der]) (n_state = n_y·n_der), matching the
# DER-only kalman_matrices.npz produced by _collect_train_mc.py.  The OLTC
# contribution to Δy is removed from the innovation using the analytical OLTC
# columns, so tap motion does not bias the DER estimate.


from experiments.helpers import ContingencyEvent
# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

def make_config() -> MultiTSOConfig:
    """Run configuration for the default multi-TSO / multi-DSO run (edit here).

    Single place to change the horizon, objective weights, OFO timing,
    profile and contingency schedule for ``main()``.  ``main_comparison()``
    keeps its own paired config.
    """
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 2,      # 36-hour (2160-min) simulation
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=20.0,            # DSO-OFO each plant step (dt_s=60 >= 10)
        g_v=3E5,                      # TSO voltage tracking; drives PCC Q dispatch
        g_q=150,                      # DSO Q-tracking
        tso_g_q_tie=0,
        #tso_g_res_sg=0,
        # ── DSO objective tuning ──
        dso_g_v=20000.0,              # reduced to avoid competing with Q tracking
        dso_g_qi=0,                   # integral Q-tracking (0 = off)
        dso_lambda_qi=0.95,           # leaky integrator decay
        dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,         # DER-primary, OLTC-backup
        # ── TSO weights (w-shift closed-loop curvature) ──
        g_w_der=100,
        g_w_gen=1e8,
        g_w_pcc=300,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=12000,
        # ── DSO weights ──
        g_w_dso_der=1000,
        g_w_dso_oltc=30,
        # ── Local-mode OLTC tap-rate limits (V1/V2 MT+NC, V3 NC) ──
        # max_step=1 (default) + wall-clock cooldown per OLTC type:
        #   MT (machine 2W gen-trafo) -> 1 tap / 180 s = once per TS interval.
        #   NC (coupler 3W interface) -> 1 tap / 60 s  = once per minute.
        # Cooldowns are wall-clock, hence robust to dt_s / dso_period_s.
        local_oltc_max_step_per_dt=1,
        #oltc_cooldown_s_mt=180.0,
        #oltc_cooldown_s_nc=60.0,
        use_fixed_zones=True,         # literature 3-area partition
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,
        verbose=1,
        tso_mode = "local",
        contingencies=[
            # Example: trip line 0 at t=30 min, restore at t=60 min
            # ContingencyEvent(minute=100, element_type="line", element_index=8, action="trip"),
            # (minute=150, element_type="line", element_index=8, action="restore"),
            # ContingencyEvent(minute=90, element_type="gen", element_index=5, action="trip"),
            # ContingencyEvent(minute=180, element_type="gen", element_index=5, action="restore"),
            # ContingencyEvent(minute=240, element_type="line", element_index=12, action="trip"),
            # ContingencyEvent(minute=150, element_type="line", element_index=12, action="restore"),
            # ContingencyEvent(minute=120, element_type="load", bus=5, p_mw=400, q_mvar=200, action="connect"),
            # ContingencyEvent(minute=300, element_type="load", bus=5, p_mw=400, q_mvar=200, action="trip"),
            # ContingencyEvent(minute=330, element_type="gen", element_index=4, action="trip"),
            # ContingencyEvent(minute=420, element_type="gen", element_index=4, action="restore"),
            # ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            # ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
            # ContingencyEvent(minute=720, element_type="load", bus=7, p_mw=300, q_mvar=100, action="connect"),
            # ContingencyEvent(minute=900, element_type="load", bus=7, p_mw=300, q_mvar=100, action="trip"),
        ],
    )

    # ---- DSO layer: only DSO_2 has an OFO controller -------------------
    cfg.dso_mode = "ofo"
    cfg.dso_ids_to_run = ["DSO_2"]

    # ---- Drop line-current rows from H (paper y = [Q_int, V]) ----------
    # With this off the DSO_2 sensitivity matrix reduces to [Q_interface | V]:
    # no ∂I/∂u rows enter the online estimator (Kalman/ANN) and the MIQP
    # enforces no line-current limits.  Regenerate kalman_matrices.npz after
    # toggling this — n_y (hence Q/R dimensions) changes.
    cfg.dso_monitor_currents = False

    # ---- DSO_2 interface: slack-decoupled vs. coupled (DSO2_INTERFACE_MODE)
    cfg.dso2_interface_slack = (DSO2_INTERFACE_MODE == "slack")
    if cfg.dso2_interface_slack:
        # Decoupling islands DSO_2 behind its own boundary slacks.  pandapower
        # cannot run a distributed slack across several islands at once, so the
        # whole run must use a per-island reference slack instead.
        cfg.distributed_slack = False

    # ---- Exogenous Q-setpoint vector at DSO_2's three 3W trafos --------
    # Order matches ``meta.hv_networks[1].coupling_trafo_indices``.
    cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [0.0, 0.0, 0.0]}

    # ---- Time-varying setpoint: bounded random walk per trafo ----------
    # Each DSO step the three interface-Q setpoints take an independent
    # Gaussian step (std below) around the base above, clipped to a per-trafo
    # band -> trafo bounds [40+-band, 20+-band, 80+-band].  Fixed seed so all
    # H-predictor modes in _run_comparison track the identical moving target.
    cfg.q_pcc_setpoint_random_enabled  = True
    cfg.q_pcc_setpoint_random_std_mvar = 1    # per-step walk increment [Mvar]
    cfg.q_pcc_setpoint_random_band_mvar = 40.0  # per-trafo half-band around base
    cfg.q_pcc_setpoint_random_seed     = 64

    # ---- Run length / timing -------------------------------------------
    cfg.n_total_s = 60.0 * 60 * 2  # 2 h smoke
    cfg.tso_period_s = 180.0  # cosmetic: TSO never steps anyway
    cfg.dso_period_s = 20  # 10
    cfg.warmup_s = 0.0
    cfg.start_time = datetime(2016, 9, 7, 10, 0)
    cfg.use_profiles = True

    # ---- Plant-side Q(V) loop damping ----------------------------------
    # 44 QVLocalLoops (4 TSO STATCOMs + 40 DSO sgens) iterate in parallel
    # inside every ``pp.runpp(run_control=True)``.  STATCOM-class units
    # (S_n ~600 Mvar with slope 0.07) have ~9 GVar/pu open-loop gain;
    # the published-stable damping is 0.1 (see QVLocalLoop docstring).
    # The 002 default of 0.5 is fine for fewer loops but oscillates here.
    # cfg.qv_local_damping = 0.1
    # cfg.qv_local_max_step_frac = None
    # cfg.qv_local_tol_mvar = 0.1

    # ---- Live plots off for diagnostic runs ---------------------------
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.live_plot_tracking = False

    # ---- Persistent excitation (sensitivity learning) -----------------
    # When enabled, gaussian noise N(0, sigma^2) is added to the DSO
    # OFO's continuous (DER Q_cor) actuator output AT EVERY DSO step,
    # AFTER the MIQP solves and BEFORE the command is sent to the plant.
    # The controller's internal u_current is also incremented by the
    # noise so the next OFO step starts from u(k+1)' = u(k+1) + eps and
    # does not "fight" the excitation.  OLTC tap commands are NOT
    # perturbed (integer actuators).  Useful for downstream H estimation
    # (Kalman filter / NN) where the input must be persistently exciting.
    cfg.dso_pe_noise_enabled = True
    cfg.dso_pe_noise_seed = 64  # RNG seed for reproducibility (also seeds the orthogonal basis)
    # Excitation mode + amplitudes from the top-of-file toggle (PE_MODE etc.).
    # "white" reproduces the pre-orthogonal behaviour; "orthogonal" is the designed
    # rotating excitation.  Both share the TARGET_STEP_MVAR=1.0 matrices (per-step
    # ‖Δu‖≈1.0 either way) — see the PE_MODE block above.
    cfg.dso_pe_mode           = PE_MODE
    cfg.dso_pe_amplitude_mvar = PE_AMPLITUDE_MVAR   # orthogonal-mode probe amplitude [Mvar]
    cfg.dso_pe_noise_std_mvar = PE_WHITE_STD_MVAR   # white-mode per-channel std [Mvar]

    # ---- Output directory ---------------------------------------------
    cfg.result_dir = os.path.join("results", "003_cigre_2026")

    # ---- Frozen operating point (estimator convergence mode) ----------
    # Profiles are kept ON so the network initialises at the real historical
    # operating point; cfg.frozen_at then holds that profile constant for the
    # entire run.  This avoids the artificial mean-profile base-case.
    if FROZEN_OP_POINT:
        cfg.use_profiles = False
        cfg.start_time = FROZEN_OP_TIMESTAMP
        cfg.frozen_at = FROZEN_OP_TIMESTAMP
        cfg.contingencies = []
        cfg.dso_pe_noise_enabled = True
        print(f"[003] FROZEN_OP_POINT=True: frozen at {FROZEN_OP_TIMESTAMP:%Y-%m-%d %H:%M}, "
              f"contingencies off, PE noise on")

    return cfg



# ---------------------------------------------------------------------------
#  H predictor interface  (read function + step wrappers + setup function)
# ---------------------------------------------------------------------------
#
# These pieces together let a Kalman filter / NN replace the controller's
# cached H matrix at every DSO step, and (optionally) inject gaussian
# input excitation for sensitivity learning.  See the module docstring
# above for the y / u / H layout and the predictor contract.  Workflow:
#
#       def my_h_predictor(dso_ctrl):
#           H_now = dso_ctrl._H_cache              # full shape, e.g. (24, 13)
#           meas  = dso_ctrl._last_measurement     # latest Measurement (V, Q, ...)
#           view  = get_dso_h_view(dso_ctrl)       # labeled view if you want named rows/cols
#
#           H_next = ...                           # KF / NN: predict H for the next DSO step
#           return H_next                          # MUST be same shape as H_now; or None to skip
#
#       # Install from the startup function passed to run_multi_tso_dso
#       # (see _setup_h_predictor below for an example):
#       install_h_corrector(state["dso_controllers"]["DSO_2"], my_h_predictor)
#
#       # Optional: gaussian persistent-excitation noise on Q_cor output.
#       # Toggle in make_config (cfg.dso_pe_noise_enabled = True/False);
#       # _setup_h_predictor reads the flag and installs the wrapper.


def get_dso_h_view(
    dso_ctrl: "DSOController",
    *,
    exclude_currents: bool = True,
    exclude_shunts: bool = True,
) -> Dict[str, Any]:
    """Return a labeled, optionally-filtered view of the DSO controller's H.

    Default filters drop current rows (``I_line``) and shunt columns
    (``shunt``), leaving voltages + Q at trafos as outputs and DER + OLTC
    as actuators -- the layout the corrector colleague is using.

    Parameters
    ----------
    dso_ctrl : DSOController
        The DSO controller to inspect.  ``dso_ctrl._H_cache`` is primed
        via ``_build_sensitivity_matrix()`` if it has not been built yet.
    exclude_currents : bool, default True
        Drop ``I_line`` rows.
    exclude_shunts : bool, default True
        Drop ``shunt`` columns.

    Returns
    -------
    dict
        Keys:
          ``H``              : np.ndarray, filtered (n_y x n_u) copy of ``_H_cache``.
          ``row_labels``     : list[str], one per row, e.g. ``"V[bus_42]"``.
          ``col_labels``     : list[str], one per column, e.g. ``"OLTC_3W[3]"``.
          ``row_units``      : list[str], physical unit for each row.
          ``col_units``      : list[str], physical unit for each column.
          ``row_kinds``      : list[str], one of ``{"Q_trafo","V_bus","I_line"}``.
          ``col_kinds``      : list[str], one of ``{"Q_cor","OLTC","shunt"}``.
          ``kept_row_mask``  : np.ndarray[bool], length = full-H rows.
          ``kept_col_mask``  : np.ndarray[bool], length = full-H cols.

    Notes
    -----
    The returned ``H`` is a copy; mutating it does not affect the
    controller cache.  To inject a corrected H, use
    :func:`install_h_corrector`.
    """
    if dso_ctrl._H_cache is None:
        dso_ctrl._build_sensitivity_matrix()
    H_full = dso_ctrl._H_cache
    m = dso_ctrl._H_mappings or {}
    cfg = dso_ctrl.config

    # ── Row metadata: order is [Q_trafo2w | Q_trafo3w_HV | V_bus | I_line] ──
    row_kinds: List[str] = []
    row_labels: List[str] = []
    row_units: List[str] = []
    for ti in m.get("trafos", []):
        row_kinds.append("Q_trafo")
        row_labels.append(f"Q_trafo2w[{ti}]")
        row_units.append("Mvar (gen conv)")
    for ti in m.get("trafo3w", []):
        row_kinds.append("Q_trafo")
        row_labels.append(f"Q_trafo3w_HV[{ti}]")
        row_units.append("Mvar (gen conv)")
    for bi in m.get("obs_buses", []):
        row_kinds.append("V_bus")
        row_labels.append(f"V[bus_{bi}]")
        row_units.append("p.u.")
    for li in m.get("lines", []):
        row_kinds.append("I_line")
        row_labels.append(f"I[line_{li}]")
        row_units.append("p.u.")

    # ── Column metadata: order is [Q_cor (per-DER) | OLTC_2W | OLTC_3W | shunt] ──
    # The DER columns of H are the closed-loop ∂y/∂Q_cor (T' applied) under
    # use_q_cor_actuator=True.  Q_cor is the correction term the OFO writes
    # to each DER; the DER's actual Q is Q_cor + K·(V_ref - V).
    col_kinds: List[str] = []
    col_labels: List[str] = []
    col_units: List[str] = []
    for d in cfg.der_indices:
        col_kinds.append("Q_cor")
        col_labels.append(f"Q_cor[sgen_{d}]")
        col_units.append("Mvar (sgen conv, +inj)")
    for ti in m.get("oltc_trafos", []):
        col_kinds.append("OLTC")
        col_labels.append(f"OLTC_2W[{ti}]")
        col_units.append("steps")
    for ti in m.get("oltc_trafo3w", []):
        col_kinds.append("OLTC")
        col_labels.append(f"OLTC_3W[{ti}]")
        col_units.append("steps")
    for bi in m.get("shunt_buses", []):
        col_kinds.append("shunt")
        col_labels.append(f"shunt[bus_{bi}]")
        col_units.append("steps")

    if len(row_kinds) != H_full.shape[0]:
        raise RuntimeError(
            f"H-view row count mismatch: counted {len(row_kinds)} from mappings "
            f"but H has {H_full.shape[0]} rows.  Mappings keys: {list(m.keys())}"
        )
    if len(col_kinds) != H_full.shape[1]:
        raise RuntimeError(
            f"H-view col count mismatch: counted {len(col_kinds)} (n_der="
            f"{len(cfg.der_indices)}) but H has {H_full.shape[1]} cols.  "
            f"Layout assumes w-shift actuator mode and no grid-forming DER; "
            f"V_gf or Q_realized splicing would invalidate this view."
        )

    # ── Filter masks ─────────────────────────────────────────────────────
    row_mask = np.array(
        [not (exclude_currents and k == "I_line") for k in row_kinds],
        dtype=bool,
    )
    col_mask = np.array(
        [not (exclude_shunts and k == "shunt") for k in col_kinds],
        dtype=bool,
    )
    kept_rows = np.flatnonzero(row_mask).tolist()
    kept_cols = np.flatnonzero(col_mask).tolist()

    # Slice once + copy so caller cannot mutate the controller's cache.
    H = H_full[np.ix_(kept_rows, kept_cols)].copy()

    return {
        "H": H,
        "row_labels": [row_labels[i] for i in kept_rows],
        "col_labels": [col_labels[j] for j in kept_cols],
        "row_units":  [row_units[i]  for i in kept_rows],
        "col_units":  [col_units[j]  for j in kept_cols],
        "row_kinds":  [row_kinds[i]  for i in kept_rows],
        "col_kinds":  [col_kinds[j]  for j in kept_cols],
        "kept_row_mask": row_mask,
        "kept_col_mask": col_mask,
    }


def install_h_corrector(
    dso_ctrl: "DSOController",
    predictor: HPredictor,
) -> None:
    """Wrap ``dso_ctrl.step`` so ``predictor(dso_ctrl)`` runs after every step.

    Per step ``k``:
      1. ``dso_ctrl.step(...)`` executes -- uses ``_H_cache`` (= ``H(k)``).
      2. ``predictor(dso_ctrl)`` is called.  It can read any controller
         attribute (``_H_cache``, ``_H_mappings``, ``_last_measurement``,
         ...) and returns either:
           * an ``np.ndarray`` of shape ``_H_cache.shape`` -- written into
             ``_H_cache`` and used as ``H(k+1)`` at the next step.
           * ``None`` -- ``_H_cache`` left untouched (e.g. warm-up window).
      3. Step ``k+1`` runs with the chosen ``H``.

    Calling this twice on the same controller swaps in the new predictor
    without double-wrapping.
    """
    # Detect a previous installation: restore the original step before
    # re-wrapping so we don't pile wrappers on top of each other.
    original_step = getattr(dso_ctrl, "_step_pre_h_corrector", None)
    if original_step is None:
        original_step = dso_ctrl.step
        dso_ctrl._step_pre_h_corrector = original_step  # type: ignore[attr-defined]

    def step_with_corrector(*args, **kwargs):
        result = original_step(*args, **kwargs)
        H_next = predictor(dso_ctrl)
        if H_next is None:
            return result
        if not isinstance(H_next, np.ndarray):
            raise TypeError(
                f"H-predictor must return np.ndarray or None, got "
                f"{type(H_next).__name__}"
            )
        if dso_ctrl._H_cache is None:
            raise RuntimeError(
                "H-predictor returned an array but _H_cache is None -- "
                "did step() fail to build H?"
            )
        if H_next.shape != dso_ctrl._H_cache.shape:
            raise ValueError(
                f"H-predictor returned shape {H_next.shape} but "
                f"_H_cache has shape {dso_ctrl._H_cache.shape}.  Predictor "
                f"must return the FULL-shape H, not the filtered view."
            )
        # Write into the sensitivity updater's base so the in-place reset
        # inside SensitivityUpdater.update() uses the predictor's H, not the
        # frozen init-time H.  Falls back to a plain reference swap when no
        # updater is present (e.g. unit tests).
        if dso_ctrl._sensitivity_updater is not None:
            dso_ctrl._sensitivity_updater.override_base(H_next)
        dso_ctrl._H_cache = H_next
        return result

    dso_ctrl.step = step_with_corrector  # type: ignore[method-assign]


def install_pe_noise(
    dso_ctrl: "DSOController",
    *,
    std_mvar: float = 0.0,
    rng: np.random.Generator,
    mode: str = "white",
    amplitude_mvar: float = 0.0,
) -> None:
    """Wrap ``dso_ctrl.step`` so a persistent-excitation (PE) increment ε(k) is
    added to the DER Q_cor commands.

    Per step k, after the OFO MIQP returns ``u(k+1) = u(k) + w``:
      * the DER slice of ``u_new`` / ``u_continuous`` (what the runner applies to
        the plant) and ``dso_ctrl._u_current`` (what the next OFO step starts from)
        are all incremented by ε(k), so the excitation becomes part of the
        controller's owned state (it does not "fight" the noise);
      * OLTC (integer) and any non-DER columns are NOT perturbed.

    Excitation modes
    ----------------
    ``"white"`` (default) -- ε(k) ~ N(0, std_mvar²) i.i.d. on each DER channel.
        Isotropic in expectation, but at the small ``std_mvar`` used in practice the
        off-diagonal H columns get little excitation (the OFO over-excites only its
        gradient direction → the rest is starved, see the identifiability budget).

    ``"orthogonal"`` -- DESIGNED excitation: ε(k) = amplitude_mvar · B[:, k mod n_der],
        where B is a fixed Haar-random orthonormal basis of the n_der DER space
        (drawn once from ``rng`` via QR).  Over every n_der steps each input
        direction gets one full-amplitude probe, decorrelated from the OFO gradient,
        so the Fisher information Σ Δu_PE Δu_PEᵀ = amplitude² · I is balanced across
        ALL H columns.  The time-averaged per-direction excitation std is
        ``amplitude_mvar / √n_der`` (vs ``std_mvar`` isotropic for white), and the
        per-step disturbance norm is exactly ``amplitude_mvar`` (one unit direction).
        Raising ``amplitude_mvar`` lifts the off-diagonal identifiability (smaller N
        to resolve a bias) at the cost of more tracking disturbance; keep it within
        the near-linear band (delta-sweep linear to ~3.4 Mvar) to avoid identifying a
        secant instead of the Jacobian.  Match the training step (TARGET_STEP_MVAR)
        to ``amplitude_mvar`` so R/Q/u_scale stay calibrated.

    Layout assumption: ``u_new = [Q_cor (n_der) | OLTC (n_oltc) | ...]`` (DER block
    at ``[0, n_der)``; holds for 003).

    Δu = ε(k) + w(k) carries an independent component spanning the input space, so
    the regression Δy ≈ H·Δu is full-rank — that is what makes H identifiable online.
    """
    n_der = len(dso_ctrl.config.der_indices)
    original_step = dso_ctrl.step
    _state = {"k": 0}

    basis: Optional[np.ndarray] = None
    if mode == "orthogonal" and n_der > 0:
        # Haar-random orthonormal basis of the DER input space (columns = probe
        # directions), fixed for the run so the rotation is reproducible.
        basis, _ = np.linalg.qr(rng.standard_normal((n_der, n_der)))

    def step_with_pe(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if n_der > 0:
            if mode == "orthogonal" and amplitude_mvar > 0.0 and basis is not None:
                eps = amplitude_mvar * basis[:, _state["k"] % n_der]
                _state["k"] += 1
            elif mode == "white" and std_mvar > 0.0:
                eps = rng.normal(0.0, std_mvar, size=n_der)
            else:
                return result
            # Mutate what the runner applies to the plant ...
            result.u_new[:n_der] += eps
            result.u_continuous[:n_der] += eps
            # ... and what the controller starts the next step from.
            if dso_ctrl._u_current is not None:
                dso_ctrl._u_current[:n_der] += eps
        return result

    dso_ctrl.step = step_with_pe  # type: ignore[method-assign]


# ─────────────────────────────────────────────────────────────────────────────
#  H-predictor implementations and training utilities
# ─────────────────────────────────────────────────────────────────────────────
#
# Both predictors are callable class instances that satisfy HPredictor exactly
# like plain functions but carry mutable state between consecutive DSO steps.
#
# Workflow
# --------
#   1. collect_training_data()     -- run 003 with PE noise; save (u, y, H)
#   2. generate_kalman_matrices()  -- estimate Q, R from that data
#   3. train_ann_model()           -- fit Keras model; save weights + stats
#   4. In _setup_h_predictor swap the predictor:
#        install_h_corrector(dso_ctrl, _kalman_h_predictor)
#        # or:
#        install_h_corrector(dso_ctrl, _ANN_h_predictor)

_RESULT_DIR_003       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "003_cigre_2026")
_TRAINING_DATA_PATH   = os.path.join(_RESULT_DIR_003, "training_data.npz")
_KALMAN_MATRICES_PATH = os.path.join(_RESULT_DIR_003, "kalman_matrices.npz")
_ANN_MODEL_PATH       = os.path.join(_RESULT_DIR_003, "ann_h_model.keras")
_ANN_STATS_PATH       = os.path.join(_RESULT_DIR_003, "ann_h_stats.npz")


def get_dso2_features(dso_ctrl: "DSOController") -> Dict[str, np.ndarray]:
    """Return all DSO_2-relevant measurements in a named dict.

    Call this inside any predictor — ``dso_ctrl._last_measurement`` is
    already populated when the predictor runs (it is set at the very start
    of ``dso_ctrl.step()`` and is not cleared afterwards).

    Keys
    ----
    q_trafo_mvar : Q at HV side of the three coupling 3W trafos [Mvar, load conv.]
                   shape (n_interface,) — these are the rows of H that track Q_set
    tap_pos      : OLTC tap positions of the coupling trafos, shape (n_oltc,)
                   integer steps; same ordering as cfg.oltc_trafo_indices
    der_q_mvar   : actual DER reactive output Q_DER = q_set + K·(V_anchor - V) [Mvar]
                   shape (n_der,) — this is what the plant actually produces
    der_p_mw     : DER active power output [MW], shape (n_der,)
    q_set_mvar   : q_set setpoint the OFO last commanded [Mvar], shape (n_der,)
                   = u[:n_der]; this is what H is linearised against
    v_buses_pu   : voltages at cfg.voltage_bus_indices [p.u.], shape (n_v_bus,)
    y            : OFO output vector [q_trafo | v_buses], shape (n_y,)
                   = rows of H; built by _extract_outputs (same as inside step())
    u            : OFO actuator vector [q_set | tap_pos], shape (n_u,)
                   = cols of H; = dso_ctrl._u_current
    """
    meas = dso_ctrl._last_measurement
    cfg  = dso_ctrl.config
    if meas is None or dso_ctrl._u_current is None:
        raise RuntimeError(
            "get_dso2_features() called before the controller has stepped once."
        )

    n_der = len(cfg.der_indices)

    v_buses = np.array(
        [
            float(meas.voltage_magnitudes_pu[np.where(meas.bus_indices == bi)[0][0]])
            for bi in cfg.voltage_bus_indices
        ],
        dtype=np.float64,
    )

    return {
        "q_trafo_mvar": meas.interface_q_hv_side_mvar.copy(),
        "tap_pos":      meas.oltc_tap_positions.astype(np.float64),
        "der_q_mvar":   meas.der_q_mvar.copy(),
        "der_p_mw":     meas.der_p_mw.copy(),
        "q_set_mvar":   dso_ctrl._u_current[:n_der].copy(),
        "v_buses_pu":   v_buses,
        "y":            dso_ctrl._extract_outputs(meas),
        "u":            dso_ctrl._u_current.copy(),
    }


class _KalmanHPredictor:
    """Tracks H as a random-walk state with noisy Δy ≈ H·Δu observations.

    State model :  h(k+1) = h(k) + w(k),        w ~ N(0, Q_eff(k))
    Observation :  Δỹ(k)  = C(Δu_e) h(k) + v(k), v ~ N(0, R)
    where h = vec_row(H[:, :n_e]) and C(Δu_e) = I_{n_y} ⊗ Δu_e^T.

    Two-term process noise
    ----------------------
    The predict-step covariance is the additive split (see module header)

        Q_eff(k) = q_scale · s²(k) · Q_excitation  +  Q_drift

    with s²(k) = ‖Δu_e / u_scale‖² / n_e.  The excitation term vanishes as the
    controller settles (Δu → 0); the always-on Q_drift then keeps the gain alive
    so the estimator never freezes.  This is the entire process-noise model —
    there is no forgetting factor, no prior mean-reversion, and no trace
    floor/ceiling: Q_drift carries the keep-alive role those hacks used to fake.

    DER-only column estimation
    --------------------------
    The filter estimates only the first ``n_e = n_der`` columns of H (the DER /
    ``Q_cor`` block); the OLTC columns are held at their analytical ``_H_cache``
    value and spliced back on every return.  The OLTC contribution to ``Δy`` is
    removed from the innovation (``Δỹ = Δy − H_rest·Δu_rest``) so tap motion does
    not bias the DER estimate.  The Kalman state is ``vec(H[:, :n_der])``
    (``n_state = n_y·n_der``), matching the DER-only ``kalman_matrices.npz``
    produced by ``_collect_train_mc.py``.

    Measurement noise is input-dependent:
    ``R_eff(k) = R_sensor + r_model_scale·‖Δu‖^p·R_model`` — a Δu-independent sensor
    floor plus an input-dependent model-error term.  On this plant the residual is
    first-order (r ∝ ‖Δu‖, p=2; not the curvature-dominated Picallo p=4), so large
    steps are distrusted (secant ≠ analytical Jacobian).

    Q_excitation, Q_drift, R_sensor and R_model are loaded from
    ``_KALMAN_MATRICES_PATH`` on the first call.  Falls back to identity defaults
    if absent; run :func:`generate_kalman_matrices` (or _collect_train_mc.py)
    first for data-driven matrices.
    """

    def __init__(self, min_delta_u_norm: float = 0.05,
                 q_scale: float = 1.0, r_model_scale: float = 1.0) -> None:
        self.min_delta_u_norm = min_delta_u_norm
        # Runtime multiplier on the excitation term (1.0 = as trained).
        self.q_scale = q_scale
        # Runtime multiplier on the ‖Δu‖^p model-error term of R (0 = fixed R_sensor).
        self.r_model_scale = r_model_scale
        # R input-dependence exponent p (R_eff = R_sensor + scale·‖Δu‖^p·R_model).
        # Overwritten from the npz on load; default = the module constant.
        self.r_du_exponent: float = KALMAN_R_DU_EXPONENT
        # Deployment DSO control period [s].  Set by _setup_h_predictor from
        # cfg.dso_period_s; Q_drift (a per-control-step drift variance) is
        # rescaled from the training cadence to this one at load time.  None ->
        # no rescale (ratio 1.0), e.g. when the predictor is used standalone.
        self.deploy_period_s: Optional[float] = None
        self._Q_exc: Optional[np.ndarray] = None      # input-driven, scaled by s²
        self._Q_drift: Optional[np.ndarray] = None    # disturbance-driven, always on
        self._R_sensor: Optional[np.ndarray] = None   # Δu-independent sensor floor
        self._R_model: Optional[np.ndarray] = None    # model-error cov, scaled by ‖Δu‖^p
        self._u_scale: Optional[np.ndarray] = None   # per-channel |Δu| RMS for the s² norm
        self._h: Optional[np.ndarray] = None         # vectorised H estimate
        self._P: Optional[np.ndarray] = None         # estimate covariance
        self._y_prev: Optional[np.ndarray] = None
        self._u_prev: Optional[np.ndarray] = None

    def _load_matrices(self, n_state: int, n_y: int) -> None:
        if os.path.exists(_KALMAN_MATRICES_PATH):
            d = np.load(_KALMAN_MATRICES_PATH)
            # New files store Q_excitation + Q_drift; legacy files store a single
            # "Q" (treated as the excitation term, drift = 0 → old behaviour).
            self._Q_exc = d["Q_excitation"] if "Q_excitation" in d else d["Q"]
            self._Q_drift = (d["Q_drift"] if "Q_drift" in d
                             else np.zeros_like(self._Q_exc))
            # Input-dependent R: R_eff = R_sensor + r_model_scale·‖Δu‖^p·R_model.
            # New files store R_sensor + R_model + r_du_exponent; legacy files store
            # a single "R" (fixed sensor term, R_model = 0 → old behaviour).  An
            # intermediate file may carry the old "R_curv" key (same role as R_model).
            if "R_sensor" in d:
                self._R_sensor = d["R_sensor"]
                if "R_model" in d:
                    self._R_model = d["R_model"]
                elif "R_curv" in d:
                    # Legacy ‖Δu‖⁴ key: built with p=4, so keep that exponent unless
                    # the file overrides it (regenerate for the p=2 model-error fit).
                    self._R_model = d["R_curv"]
                    self.r_du_exponent = 4.0
                else:
                    self._R_model = np.zeros_like(d["R_sensor"])
                if "r_du_exponent" in d:
                    self.r_du_exponent = float(d["r_du_exponent"])
            else:
                self._R_sensor = d["R"]
                self._R_model  = np.zeros_like(d["R"])
            # Per-channel |Δu| RMS used to normalise the s² = ‖Δu/u_scale‖² scaling.
            # Missing in legacy files -> ones (raw norm); regenerate for the scale.
            n_u = int(round(n_state / n_y))
            self._u_scale = d["u_scale"] if "u_scale" in d else np.ones(n_u)
            # Q_drift is a per-control-step drift covariance; random-walk variance
            # grows ∝ Δt, so rescale it linearly from the training cadence
            # (train_period_s) to the deployment cadence (deploy_period_s).  Only
            # Q_drift needs this: Q_excitation is already cadence-aware through s²
            # (which scales with the observed actuator step).  Legacy files lacking
            # train_period_s, or a predictor with no deploy_period_s set, skip it.
            train_period = float(d["train_period_s"]) if "train_period_s" in d else None
            ratio = 1.0
            if train_period and self.deploy_period_s and train_period > 0.0:
                ratio = float(self.deploy_period_s) / train_period
                self._Q_drift = self._Q_drift * ratio
            print(f"[KF] loaded Q_excitation {self._Q_exc.shape}, "
                  f"Q_drift {self._Q_drift.shape}, R_sensor {self._R_sensor.shape}, "
                  f"R_model {self._R_model.shape} (‖Δu‖^{self.r_du_exponent}), "
                  f"u_scale {self._u_scale.shape}")
            if ratio != 1.0:
                print(f"[KF] Q_drift rescaled by deploy/train period "
                      f"{self.deploy_period_s:.0f}/{train_period:.0f} = {ratio:.3f}")
            elif train_period is None:
                print("[KF] no train_period_s in matrices — Q_drift used as-is "
                      "(regenerate to enable period rescaling)")
        else:
            self._Q_exc = np.eye(n_state) * 1e-6
            self._Q_drift = np.eye(n_state) * 1e-9
            self._R_sensor = np.eye(n_y) * 1e-4
            self._R_model = np.zeros((n_y, n_y))
            self._u_scale = np.ones(int(round(n_state / n_y)))
            print("[KF] no kalman_matrices.npz found — using I defaults. "
                  "Run generate_kalman_matrices() for data-driven matrices.")

    def reset(self) -> None:
        """Reset KF state; re-seeds from the controller's H on next call."""
        self._h = self._P = self._y_prev = self._u_prev = None

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None

        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None

        # feats keys available here (replace manual extraction with these):
        #   feats["y"]           -- OFO output vector [q_trafo | v_buses]   (n_y,)
        #   feats["u"]           -- OFO actuator vector [q_cor | tap_pos]   (n_u,)
        #   feats["q_trafo_mvar"]-- Q at coupling 3W trafos                 (n_trafo,)
        #   feats["tap_pos"]     -- OLTC tap positions                       (n_oltc,)
        #   feats["der_q_mvar"]  -- actual DER Q output (Q_cor + K*(Vref-V))(n_der,)
        #   feats["der_p_mw"]    -- DER active power                         (n_der,)
        #   feats["q_set_mvar"]  -- q_set commanded by OFO (= u[:n_der])     (n_der,)
        #   feats["v_buses_pu"]  -- voltages at monitored buses               (n_v,)

        n_y, n_u = dso_ctrl._H_cache.shape
        # Estimated columns: the leading DER (Q_cor) block only; OLTC columns are
        # held analytical and spliced back on return (see class docstring).
        n_der = len(dso_ctrl.config.der_indices)
        n_e = n_der
        n_state = n_y * n_e

        if self._Q_exc is None or self._Q_exc.shape[0] != n_state:
            self._load_matrices(n_state, n_y)

        y = feats["y"]
        u = feats["u"]

        # Warm-up: seed h and P from the controller's analytical H (DER columns
        # only; OLTC columns stay in _H_cache and are spliced on return).
        if self._h is None:
            self._h = dso_ctrl._H_cache[:, :n_e].flatten().copy()
            # P_init reflects the actual initial uncertainty (bias on the seed H),
            # not the per-step Q magnitude: ~30% per-entry std matches the init bias.
            h_rms = float(np.sqrt(np.mean(self._h ** 2))) or 1e-4
            sigma_init = 0.30 * h_rms
            self._P = (sigma_init ** 2) * np.eye(n_state)
            self._y_prev, self._u_prev = y, u
            return None  # keep analytical H on the first call

        delta_u = u - self._u_prev
        delta_y = y - self._y_prev

        # Restrict the regression to the estimated (DER) columns.  The OLTC columns
        # are held at their analytical value in _H_cache; subtract their
        # contribution to Δy from the innovation so tap motion does not bias the
        # DER estimate (no-op when the taps do not move, Δu_rest = 0).
        delta_u_e = delta_u[:n_e]
        if n_e < n_u:
            H_rest = dso_ctrl._H_cache[:, n_e:]          # analytical OLTC columns
            delta_y = delta_y - H_rest @ delta_u[n_e:]

        # ── Predict: two-term process noise ────────────────────────────────────
        # Q_eff = q_scale · s² · Q_excitation  +  Q_drift
        #   s² = ‖Δu_e / u_scale‖² / n_e   (normalised step energy, dimensionless;
        #        s²≈1 at a typical full step, → 0 as the controller settles).
        # The excitation term ties assumed H drift to actuator motion (no fictitious
        # diffusion at convergence); the always-on Q_drift carries the keep-alive
        # role, so there is no forgetting factor, prior anchor, or trace bound.
        s2 = float(np.sum((delta_u_e / self._u_scale) ** 2)) / len(delta_u_e)
        Q_eff = self.q_scale * s2 * self._Q_exc + self._Q_drift
        h_p = self._h
        P_p = self._P + Q_eff

        # Input-dependent measurement noise R_eff = R_sensor + r_model_scale·‖Δu‖^p·R_model.
        # On this plant the residual is first-order (r ∝ ‖Δu‖, p=2): a larger step
        # carries more model error, so the filter trusts small near-linear steps and
        # distrusts large ones (avoids fitting a large-step secant into H).  ‖Δu_e‖ is
        # the RAW DER step (Mvar), matching the training normalisation (distinct from
        # the u_scale normalisation used by s²).
        du_raw = float(np.sqrt(delta_u_e @ delta_u_e))      # ‖Δu_e‖
        R_eff = self._R_sensor + self.r_model_scale * (du_raw ** self.r_du_exponent) * self._R_model

        # ── Update: skip when ‖Δu_e‖ is too small (ill-conditioned regression) ──
        if np.linalg.norm(delta_u_e) >= self.min_delta_u_norm:
            C = np.kron(np.eye(n_y), delta_u_e.reshape(1, -1))  # (n_y, n_state)
            S = C @ P_p @ C.T + R_eff                          # (n_y, n_y)
            K = np.linalg.solve(S, C @ P_p).T                  # (n_state, n_y)
            self._h = h_p + K @ (delta_y - C @ h_p)
            self._P = (np.eye(n_state) - K @ C) @ P_p
        else:
            self._h, self._P = h_p, P_p

        self._y_prev, self._u_prev = y, u
        # Splice the estimated DER columns back into the full analytical H; the
        # OLTC columns are returned unchanged (not estimated).
        H_out = dso_ctrl._H_cache.copy()
        H_out[:, :n_e] = self._h.reshape(n_y, n_e)
        return H_out


_kalman_h_predictor = _KalmanHPredictor()


class _PerRowRLSPredictor:
    """Per-row Recursive Least Squares H estimator with forgetting factor.

    Observation model: Δy_i = H[i,:] · Δu  (rows are physically decoupled)
    Each of the n_y rows maintains its own (n_u × n_u) covariance P_i.
    Forgetting factor λ < 1 exponentially discounts old observations.

    Advantages over full Kalman:
    - No Q/R calibration needed; λ is the only tuning knob
    - P stays bounded (λ prevents unbounded growth)
    - OLTC columns remain near their initial values when Δu_tap ≈ 0
    """

    def __init__(
        self,
        forgetting_factor: float = 0.995,
        delta_init: float = 1.0,
        min_delta_u_norm: float = 0.05,
    ):
        self.lam = forgetting_factor
        self.delta = delta_init
        self.min_du = min_delta_u_norm
        self._H: Optional[np.ndarray] = None      # (n_y, n_u)
        self._P: Optional[list] = None            # list of n_y (n_u, n_u) arrays
        self._y_prev: Optional[np.ndarray] = None
        self._u_prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._H = self._P = self._y_prev = self._u_prev = None

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None
        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None

        n_y, n_u = dso_ctrl._H_cache.shape
        y: np.ndarray = feats["y"]
        u: np.ndarray = feats["u"]

        if self._H is None:
            self._H = dso_ctrl._H_cache.copy()
            # Scale the initial covariance to the H magnitude so the first
            # update corrects ~delta*100 % of each entry rather than
            # instantly overshooting. delta=1 → ~1 % correction per step.
            h_rms_sq = float(np.mean(self._H ** 2)) or 1e-8
            delta_scaled = self.delta * h_rms_sq
            self._P = [delta_scaled * np.eye(n_u) for _ in range(n_y)]
            self._y_prev = y.copy()
            self._u_prev = u.copy()
            return None

        delta_u = u - self._u_prev
        delta_y = y - self._y_prev

        if np.linalg.norm(delta_u) >= self.min_du:
            c = delta_u  # shape (n_u,)
            for i in range(n_y):
                P_i = self._P[i]
                Pc = P_i @ c                              # (n_u,)
                denom = self.lam + float(c @ Pc)
                k_i = Pc / denom                          # gain (n_u,)
                e_i = float(delta_y[i]) - float(c @ self._H[i])
                self._H[i] += k_i * e_i
                self._P[i] = (P_i - np.outer(k_i, Pc)) / self.lam

        self._y_prev = y.copy()
        self._u_prev = u.copy()
        return self._H.copy()


_rls_h_predictor = _PerRowRLSPredictor()


class _ANNHPredictor:
    """ANN predictor: estimates H from the current I/O vectors [y, u].

    Feature vector (must match :func:`train_ann_model`): [y(k), u(k)]
    Output: H_analytical(k).flatten() reshaped to (n_y, n_u).

    Loads a Keras model from ``_ANN_MODEL_PATH`` on the first call.
    Returns ``None`` (leaves H untouched) if the model file is absent;
    run :func:`train_ann_model` first.
    """

    def __init__(self) -> None:
        self._model = None
        self._tried_load = False
        self._x_mean: Optional[np.ndarray] = None
        self._x_std:  Optional[np.ndarray] = None
        self._t_mean: Optional[np.ndarray] = None
        self._t_std:  Optional[np.ndarray] = None

    def _try_load(self) -> None:
        self._tried_load = True
        if not os.path.exists(_ANN_MODEL_PATH):
            print("[ANN] no ann_h_model.keras found — returning None. "
                  "Run train_ann_model() first.")
            return
        try:
            from tensorflow import keras
            self._model = keras.models.load_model(_ANN_MODEL_PATH)
            if os.path.exists(_ANN_STATS_PATH):
                st = np.load(_ANN_STATS_PATH)
                self._x_mean, self._x_std = st["x_mean"], st["x_std"]
                self._t_mean, self._t_std = st["t_mean"], st["t_std"]
            print(f"[ANN] loaded model from {_ANN_MODEL_PATH}")
        except Exception as exc:
            print(f"[ANN] failed to load model: {exc}")

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if not self._tried_load:
            self._try_load()
        if self._model is None:
            return None
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None

        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None

        # feats keys available for building the input feature vector:
        #   feats["y"]           -- OFO output vector [q_trafo | v_buses]   (n_y,)
        #   feats["u"]           -- OFO actuator vector [q_cor | tap_pos]   (n_u,)
        #   feats["q_trafo_mvar"]-- Q at coupling 3W trafos                 (n_trafo,)
        #   feats["tap_pos"]     -- OLTC tap positions                       (n_oltc,)
        #   feats["der_q_mvar"]  -- actual DER Q output (Q_cor + K*(Vref-V))(n_der,)
        #   feats["der_p_mw"]    -- DER active power                         (n_der,)
        #   feats["q_set_mvar"]  -- q_set commanded by OFO (= u[:n_der])     (n_der,)
        #   feats["v_buses_pu"]  -- voltages at monitored buses               (n_v,)

        n_y, n_u = dso_ctrl._H_cache.shape
        # Feature vector: [y(k), u(k)] — must match train_ann_model.
        x = np.concatenate([feats["y"], feats["u"]]).reshape(1, -1).astype(np.float32)

        if self._x_mean is not None:
            x = (x - self._x_mean) / (self._x_std + 1e-8)
        h_next = self._model.predict(x, verbose=0)[0]
        if self._t_mean is not None:
            h_next = h_next * (self._t_std + 1e-8) + self._t_mean

        return h_next.reshape(n_y, n_u)


_ANN_h_predictor = _ANNHPredictor()


# ── Training data collection ──────────────────────────────────────────────────

def collect_training_data(
    n_total_s: float = 7200.0,
    pe_noise_std_mvar: float = 0.001,
    output_path: str = _TRAINING_DATA_PATH,
) -> str:
    """Run the 003 simulation and record measurements at every DSO_2 step.

    PE noise is enabled so that Δu is persistently exciting and off-diagonal
    H entries are identifiable from the data.

    Saved .npz arrays  (N = number of DSO steps recorded)
    -----------------
    u            : (N, n_u)       OFO actuator vector [q_cor | tap_pos]
    y            : (N, n_y)       OFO output vector   [q_trafo | v_buses]
    H            : (N, n_y, n_u)  analytical sensitivity matrix
    q_trafo_mvar : (N, n_trafo)   Q at HV side of coupling 3W trafos [Mvar, load conv.]
    tap_pos      : (N, n_oltc)    OLTC tap positions
    der_q_mvar   : (N, n_der)     actual DER Q output  Q_DER = Q_cor + K·(Vref−V)
    der_p_mw     : (N, n_der)     DER active power [MW]
    q_set_mvar   : (N, n_der)     q_set commanded by OFO  (= u[:n_der])
    v_buses_pu   : (N, n_v_bus)   voltages at monitored buses [p.u.]

    Returns the absolute path of the saved file.
    """
    cfg = make_config()
    cfg.n_total_s = n_total_s
    cfg.dso_pe_noise_enabled = True
    cfg.dso_pe_noise_std_mvar = pe_noise_std_mvar
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False

    cfg.local_sensitivities_tso=True
    cfg.local_sensitivities_dso=True

    records: List[Dict[str, np.ndarray]] = []
    _net_ref: List[Any] = []

    def _collector(dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if dso_ctrl._H_cache is None or dso_ctrl._u_current is None:
            return None
        try:
            feats = get_dso2_features(dso_ctrl)
        except RuntimeError:
            return None
        feats["H"] = dso_ctrl._H_cache.copy()
        # Record the true Jacobian H at the current operating point as training target.
        if _net_ref:
            h_anal = dso_ctrl.compute_h_analytical(_net_ref[0])
            if h_anal is not None:
                feats["H_analytical"] = h_anal
        records.append(feats)
        return None  # keep the cached H unchanged; only observe

    def _pre_loop(state: dict) -> None:
        dso_ctrl = state["dso_controllers"].get("DSO_2")
        if dso_ctrl is None:
            return
        net = state.get("net")
        if net is not None:
            _net_ref.append(net)
        rng = np.random.default_rng(cfg.dso_pe_noise_seed)
        install_pe_noise(dso_ctrl, std_mvar=pe_noise_std_mvar, rng=rng)
        install_h_corrector(dso_ctrl, _collector)
        print(f"[collect] DSO_2 ready; PE std={pe_noise_std_mvar} Mvar, "
              f"duration={n_total_s / 3600:.1f} h")

    try:
        run_multi_tso_dso(cfg, pre_loop_hook=_pre_loop)
    except Exception as exc:
        if not records:
            raise RuntimeError(
                f"Simulation failed before any DSO steps were recorded: {exc}"
            ) from exc
        print(f"[collect] simulation ended early ({type(exc).__name__}: {exc}); "
              f"saving {len(records)} steps collected so far.")

    if not records:
        raise RuntimeError("No DSO steps recorded — did the simulation succeed?")

    # Only keep keys present in every record (H_analytical may be absent on warm-up).
    common_keys = set(records[0].keys())
    for r in records[1:]:
        common_keys &= set(r.keys())
    arrays = {k: np.stack([r[k] for r in records]) for k in common_keys}
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path, **arrays)
    abs_out = os.path.abspath(output_path)
    print(f"[collect] {len(records)} steps -> {abs_out}")
    for k, v in arrays.items():
        print(f"[collect]   {k}: {v.shape}")
    return abs_out


# ── Kalman Q/R matrix estimation ─────────────────────────────────────────────

def floor_measurement_noise_R(
    R: np.ndarray,
    n_q: int,
    sigma_q: Optional[float] = None,
    sigma_v: Optional[float] = None,
    diagonal: Optional[bool] = None,
) -> np.ndarray:
    """Condition the measurement-noise covariance R for use in the Kalman gain.

    The MC residual covariance Δy − H·Δu is noise-free (pure linearisation
    curvature), so it is near rank-deficient (cross-row corr ≈0.99, Mvar/p.u.
    scale split → cond ~1e10), which makes R⁻¹ — hence the gain / innovation —
    explode.  A well-posed KF R must be PD AND well-conditioned, representing
    INDEPENDENT per-channel sensor noise.  This:
      1. (``diagonal``) drops the spurious off-diagonal correlations, then
      2. ADDS a sensor-noise floor ``diag([σ_Q²]·n_q | [σ_V²]·n_v)`` — the
         residual (model) variance and the sensor variance are independent, so
         they add.

    Row order is ``[Q_trafo (n_q, Mvar) | V_bus (n_v, p.u.)]`` (currents dropped,
    so no σ_I).  Defaults pull ``KALMAN_R_SIGMA_Q/V`` and ``KALMAN_R_DIAGONAL``.
    """
    sigma_q  = KALMAN_R_SIGMA_Q  if sigma_q  is None else sigma_q
    sigma_v  = KALMAN_R_SIGMA_V  if sigma_v  is None else sigma_v
    diagonal = KALMAN_R_DIAGONAL if diagonal is None else diagonal
    R = np.asarray(R, dtype=float)
    n_y = R.shape[0]
    n_v = max(n_y - n_q, 0)
    R_base = np.diag(np.diag(R)) if diagonal else R.copy()
    floor  = np.concatenate([np.full(n_q, sigma_q ** 2),
                             np.full(n_v, sigma_v ** 2)])
    return R_base + np.diag(floor)


def estimate_measurement_noise_terms(
    residuals: np.ndarray,
    du_norm: np.ndarray,
    n_q: int,
    n_y: int,
    *,
    sigma_q: Optional[float] = None,
    sigma_v: Optional[float] = None,
    min_du: Optional[float] = None,
    exponent: Optional[float] = None,
) -> tuple:
    """Estimate the two terms of the input-dependent measurement noise R.

    Returns ``(R_sensor, R_model)`` (both (n_y, n_y), diagonal) for the runtime
    model ``R_eff(k) = R_sensor + ‖Δu(k)‖^p · R_model`` (see module header):

    * ``R_sensor`` -- the Δu-independent sensor floor
      ``diag([σ_Q²]·n_q | [σ_V²]·n_v)`` (built via
      :func:`floor_measurement_noise_R` on a zero matrix).
    * ``R_model``  -- the input-dependent model-error covariance.  The residual is
      empirically ``r ∝ ‖Δu‖`` on this plant (first-order analytical-H bias; the
      2nd-order curvature would give ``r ∝ ‖Δu‖²``), so with exponent ``p`` its
      variance is ``∝ ‖Δu‖^p`` (p=2 here, p=4 for the curvature case).  Normalising
      ``r̃ = r/‖Δu‖^{p/2}`` removes the size dependence, so
      ``R_model = E[r̃ r̃ᵀ]`` (diagonal 2nd moment).  Scaled back up by ``‖Δu‖^p``
      at runtime.

    Residuals with ``‖Δu‖ < min_du`` are dropped from the R_model fit so the
    ``1/‖Δu‖^{p/2}`` normalisation does not amplify tiny-step noise.
    """
    min_du   = KALMAN_R_MODEL_MIN_DU if min_du   is None else min_du
    exponent = KALMAN_R_DU_EXPONENT  if exponent is None else exponent
    R_sensor = floor_measurement_noise_R(
        np.zeros((n_y, n_y)), n_q, sigma_q=sigma_q, sigma_v=sigma_v, diagonal=True
    )
    res = np.asarray(residuals, dtype=float).reshape(-1, n_y)
    dun = np.asarray(du_norm, dtype=float).reshape(-1)
    mask = dun > float(min_du)
    if int(mask.sum()) >= 2:
        rn = res[mask] / (dun[mask][:, None] ** (exponent / 2.0))   # r̃ = r/‖Δu‖^{p/2}
        R_model = np.diag(np.mean(rn ** 2, axis=0))                 # diagonal 2nd moment
    else:
        R_model = np.zeros((n_y, n_y))
    return R_sensor, R_model


def generate_kalman_matrices(
    data_path: str = _TRAINING_DATA_PATH,
    output_path: str = _KALMAN_MATRICES_PATH,
    min_delta_u_norm: float = 0.05,
    settled_delta_u_norm: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Estimate the two-term Kalman process noise + measurement noise from data.

    Splits the per-step H change ΔH(k) = H(k+1) − H(k) by whether the controller
    was moving (the comment's settled-mask recipe), giving the two additive
    process-noise terms used by :class:`_KalmanHPredictor`:

    Q_excitation -- diagonal var(ΔH) over EXCITED steps (‖Δu‖ ≥
                    ``settled_delta_u_norm``): H change driven by actuator motion.
                    Scaled by s² = ‖Δu/u_scale‖²/n_e at runtime.
    Q_drift      -- diagonal var(ΔH) over SETTLED steps (‖Δu‖ <
                    ``settled_delta_u_norm``): H change from exogenous
                    disturbances (load / profiles) while the controller is at
                    rest.  Added unscaled (always on) at runtime — the keep-alive.
    R_sensor/R_model -- the input-dependent measurement noise
                    R_eff = R_sensor + ‖Δu‖^p·R_model (p=KALMAN_R_DU_EXPONENT),
                    estimated from the residual Δy(k) − H(k−1)·Δu(k) over excited
                    steps (‖Δu‖ ≥ ``min_delta_u_norm``) via
                    :func:`estimate_measurement_noise_terms`.
    u_scale      -- per-channel RMS of Δu (normalises s²).

    Saves a .npz with keys ``Q_excitation``, ``Q_drift``, ``R_sensor``, ``R_model``,
    ``R`` (representative, at the median step), ``r_du_exponent``, ``u_scale``,
    ``train_period_s``, ``n_q``.
    """
    d = np.load(data_path)
    U: np.ndarray = d["u"]            # (N, n_u)  OFO actuator vector
    Y: np.ndarray = d["y"]            # (N, n_y)  OFO output vector
    # Prefer H_analytical (true Jacobian) over the cached H when available.
    H: np.ndarray = d["H_analytical"] if "H_analytical" in d else d["H"]
    print(f"[kalman] using {'H_analytical' if 'H_analytical' in d else 'H (cached)'} "
          f"from training data")
    N, n_y, n_u = H.shape
    n_state = n_y * n_u

    dH = np.diff(H, axis=0).reshape(N - 1, n_state)   # (N-1, n_state)
    dU = np.diff(U, axis=0)                            # (N-1, n_u)
    du_norm = np.linalg.norm(dU, axis=1)              # (N-1,)

    # ── Two-term process noise: split ΔH by controller motion ──────────────
    excited = du_norm >= settled_delta_u_norm
    settled = ~excited
    if int(excited.sum()) >= 2:
        Q_excitation = np.diag(np.var(dH[excited], axis=0))
        print(f"[kalman] Q_excitation from {int(excited.sum())} excited "
              f"(‖Δu‖≥{settled_delta_u_norm}) steps")
    else:
        Q_excitation = np.diag(np.var(dH, axis=0))
        print("[kalman] too few excited steps — Q_excitation from all ΔH")
    if int(settled.sum()) >= 2:
        Q_drift = np.diag(np.var(dH[settled], axis=0))
        print(f"[kalman] Q_drift from {int(settled.sum())} settled "
              f"(‖Δu‖<{settled_delta_u_norm}) steps")
    else:
        Q_drift = np.eye(n_state) * 1e-9
        print("[kalman] too few settled steps — Q_drift = I*1e-9 (floor). "
              "(Expected for frozen / always-excited training data.)")

    # Per-channel actuator-step scale: RMS of Δu per input channel.  Used to
    # normalise the runtime s² scaling so DER (Mvar) and OLTC (taps) moves
    # contribute comparably and s≈1 at a typical full step.
    u_scale = np.sqrt(np.mean(dU ** 2, axis=0))        # (n_u,)
    u_scale = np.where(u_scale > 1e-9, u_scale, 1.0)   # guard zero-motion channels

    # Measurement noise: residual r = Δy - H(k-1)@Δu over excited steps, plus the
    # step norm ‖Δu‖ for the input-dependent model R_eff = R_sensor + ‖Δu‖^p·R_model.
    residuals, residual_du = [], []
    for k in range(1, N):
        du  = U[k] - U[k - 1]
        dun = float(np.linalg.norm(du))
        if dun < min_delta_u_norm:
            continue
        residuals.append((Y[k] - Y[k - 1]) - H[k - 1] @ du)
        residual_du.append(dun)

    # Row split [Q_trafo (n_q, Mvar) | V_bus (n_v, p.u.)]; n_q from the training
    # data's q_trafo_mvar block, else the make_config interface count.
    n_q = (int(d["q_trafo_mvar"].shape[1]) if "q_trafo_mvar" in d
           else len(make_config().q_pcc_setpoints_mvar_per_dso.get("DSO_2", [0, 0, 0])))
    if len(residuals) >= 2:
        RES = np.stack(residuals)                       # (M, n_y)
        DUN = np.asarray(residual_du)                   # (M,)
        R_sensor, R_model = estimate_measurement_noise_terms(RES, DUN, n_q, n_y)
        print(f"[kalman] R terms from {len(residuals)} residuals "
              f"({int((DUN > KALMAN_R_MODEL_MIN_DU).sum())} used for R_model, "
              f"p={KALMAN_R_DU_EXPONENT})")
    else:
        R_sensor = floor_measurement_noise_R(np.zeros((n_y, n_y)), n_q)
        R_model  = np.zeros((n_y, n_y))
        print("[kalman] too few excited steps for R — R_model=0, sensor floor only")

    # Representative R at the median step (diagnostic + legacy 'R' key).
    du_med = float(np.median(residual_du)) if residual_du else 0.0
    R = R_sensor + KALMAN_R_MODEL_SCALE * (du_med ** KALMAN_R_DU_EXPONENT) * R_model

    # Ensure PD (Q only; the R_sensor floor already makes R_eff PD)
    Q_excitation += 1e-12 * np.eye(n_state)
    Q_drift      += 1e-12 * np.eye(n_state)

    # train_period_s = the DSO control period the training run used (collect_
    # training_data runs make_config() without changing dso_period_s).  The runtime
    # rescales Q_drift from this cadence to its own deployment period.
    train_period_s = float(make_config().dso_period_s)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path, Q_excitation=Q_excitation, Q_drift=Q_drift,
             R_sensor=R_sensor, R_model=R_model, R=R, r_du_exponent=KALMAN_R_DU_EXPONENT,
             u_scale=u_scale, train_period_s=train_period_s, n_q=n_q)
    qe_d, qd_d = np.diag(Q_excitation), np.diag(Q_drift)
    rs_d, rm_d = np.diag(R_sensor), np.diag(R_model)
    print(f"[kalman] Q_excitation {Q_excitation.shape}, Q_drift {Q_drift.shape}, "
          f"R_sensor/R_model {R_sensor.shape}, u_scale {u_scale.shape} "
          f"-> {os.path.abspath(output_path)}")
    print(f"[kalman]   Q_excitation diag: [{qe_d.min():.2e}, {qe_d.max():.2e}]")
    print(f"[kalman]   Q_drift      diag: [{qd_d.min():.2e}, {qd_d.max():.2e}]")
    print(f"[kalman]   R_sensor     diag: [{rs_d.min():.2e}, {rs_d.max():.2e}]  "
          f"(σ_Q={KALMAN_R_SIGMA_Q}, σ_V={KALMAN_R_SIGMA_V})")
    print(f"[kalman]   R_model      diag: [{rm_d.min():.2e}, {rm_d.max():.2e}]  "
          f"(×‖Δu‖^{KALMAN_R_DU_EXPONENT}; median ‖Δu‖={du_med:.3f})")
    print(f"[kalman]   R_eff@median cond {np.linalg.cond(R):.2e}, "
          f"diag [{np.diag(R).min():.2e}, {np.diag(R).max():.2e}]")
    print(f"[kalman]   u_scale: [{u_scale.min():.3f}, {u_scale.max():.3f}]")
    print(f"[kalman]   train_period_s: {train_period_s:.0f} s")
    return {"Q_excitation": Q_excitation, "Q_drift": Q_drift,
            "R_sensor": R_sensor, "R_model": R_model, "R": R,
            "u_scale": u_scale, "train_period_s": train_period_s}


# ── ANN training ─────────────────────────────────────────────────────────────

def train_ann_model(
    data_path: str = _TRAINING_DATA_PATH,
    model_path: str = _ANN_MODEL_PATH,
    stats_path: str = _ANN_STATS_PATH,
    epochs: int = 200,
    batch_size: int = 32,
    validation_split: float = 0.15,
    hidden_units: tuple = (256, 128, 64),
) -> None:
    """Fit a dense ANN to predict H(k+1) from DSO_2 measurements at step k.

    Features : x = [y(k), u(k)]
    Targets  : t = H_analytical(k).flatten()

    Input/output pairs are (y(k), u(k)) → H_analytical(k) — the ANN
    estimates the true Jacobian at the current operating point directly
    from the I/O vectors.  ``H_analytical`` is used when present in the
    training file (generated by :func:`collect_training_data`); falls
    back to ``H`` (cached sensitivity) for compatibility with older data.

    Both x and t are z-score normalised; statistics are saved alongside
    the model in ``stats_path`` and applied by :class:`_ANNHPredictor`.

    Architecture : Dense(hidden[0], relu) → … → Dense(n_state, linear)
    Loss         : MSE on normalised targets, early stopping (patience=20)
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required: pip install tensorflow"
        ) from exc

    d = np.load(data_path)
    U: np.ndarray = d["u"]   # (N, n_u)
    Y: np.ndarray = d["y"]   # (N, n_y)
    # Prefer H_analytical (true Jacobian) as target when available.
    H_target: np.ndarray = (
        d["H_analytical"] if "H_analytical" in d else d["H"]
    )
    print(f"[ANN] training target: "
          f"{'H_analytical' if 'H_analytical' in d else 'H (cached)'}")
    N = len(U)
    if N < 1:
        raise ValueError(f"Need at least 1 training step, got {N}.")

    n_y, n_u = H_target.shape[1], H_target.shape[2]
    n_state = n_y * n_u

    # Samples: x[k] = [y(k), u(k)]  →  t[k] = H_analytical(k).flatten()
    X = np.concatenate([Y, U], axis=1).astype(np.float32)   # (N, n_y+n_u)
    T = H_target.reshape(N, n_state).astype(np.float32)     # (N, n_state)

    x_mean, x_std = X.mean(0), X.std(0)
    t_mean, t_std = T.mean(0), T.std(0)
    X_n = (X - x_mean) / (x_std + 1e-8)
    T_n = (T - t_mean) / (t_std + 1e-8)

    inputs = keras.Input(shape=(X_n.shape[1],))
    z = inputs
    for units in hidden_units:
        z = keras.layers.Dense(units, activation="relu")(z)
    outputs = keras.layers.Dense(n_state)(z)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    model.fit(
        X_n, T_n,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, verbose=1
            ),
        ],
        verbose=1,
    )

    out_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(out_dir, exist_ok=True)
    model.save(model_path)
    np.savez(stats_path, x_mean=x_mean, x_std=x_std, t_mean=t_mean, t_std=t_std)
    print(f"[ANN] model -> {os.path.abspath(model_path)}")
    print(f"[ANN] stats -> {os.path.abspath(stats_path)}")


class _NumericalHPredictor:
    """ORACLE predictor: returns the TRUE plant sensitivity at the current
    operating point, by closed-loop finite difference on the live combined net.

    This is the upper-bound baseline for the H-estimation comparison: the OFO
    acts with perfect knowledge of the realized secant at the OFO step scale
    (``delta_q_mvar`` = :data:`NUMERICAL_ORACLE_DELTA_MVAR`).  Recomputed every
    DSO step so it tracks operating-point drift.  Cost ~``2*(n_DER_bus +
    n_OLTC)`` plant power flows per step.

    ``combined_net`` is injected by :func:`_setup_h_predictor` (the same live
    net object the runner mutates each step).  ``compute_numerical_h_dso``
    deep-copies it internally, so the live converged state is never disturbed.
    """

    def __init__(self) -> None:
        self.combined_net: Optional[Any] = None
        self.delta_q_mvar: float = NUMERICAL_ORACLE_DELTA_MVAR

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        if self.combined_net is None or dso_ctrl._H_cache is None:
            return None
        try:
            H = compute_numerical_h_dso(
                self.combined_net, dso_ctrl,
                closed_loop=True, delta_q_mvar=self.delta_q_mvar,
            )
        except Exception as exc:  # noqa: BLE001 — oracle must never crash the run
            print(f"[numerical] H recompute failed: {type(exc).__name__}: {exc}")
            return None
        if H is None or H.shape != dso_ctrl._H_cache.shape:
            print(f"[numerical] shape mismatch / None "
                  f"(got {None if H is None else H.shape}, "
                  f"want {dso_ctrl._H_cache.shape}); keeping cache")
            return None
        return H


_numerical_h_predictor = _NumericalHPredictor()


def _unity_multiply_predictor(
    dso_ctrl: "DSOController",
) -> Optional[np.ndarray]:
    """Workflow-test predictor: returns ``H * 1.0`` (a fresh array).

    Functionally equivalent to ``_identity_predictor`` -- the H written
    back is identical to the H that came in -- but the array IS
    routed through the wrapper's full write path: type check, shape
    check, ``_H_cache`` reassignment, and use of the new object at step
    ``k+1``.  Use this to verify before plugging in a real KF / NN.

    Replace with the real predictor once the wiring is validated.
    """
    return dso_ctrl._H_cache * 1


# ---------------------------------------------------------------------------
#  Per-step observer (sidecar capture)
# ---------------------------------------------------------------------------

class _DSO2Observer:
    """Wraps a predictor and records H_used / H_predicted / H_analytical / y / u per step.

    ``h_used``       -- ``_H_cache`` just before the predictor fires; this is
                        the H matrix that was used in the MIQP at this step.
    ``h_predicted``  -- what the predictor returns (same as ``h_used`` when
                        the predictor returns ``None``, e.g. during warm-up).
    ``h_analytical`` -- H freshly recomputed from the current operating point
                        of the combined network (true Jacobian linearisation).
                        Requires *combined_net* to be passed at construction;
                        ``None`` entries are dropped at save time.
    ``y``            -- OFO output vector extracted from the measurement.
    ``u``            -- OFO actuator vector at this step.

    Call :meth:`save` after the simulation to dump all arrays to a ``.npz``
    sidecar file.
    """

    def __init__(
        self,
        predictor: HPredictor,
        combined_net: Optional[Any] = None,
    ) -> None:
        self._predictor    = predictor
        self._combined_net = combined_net
        self._h_used:       List[np.ndarray] = []
        self._h_predicted:  List[np.ndarray] = []
        self._h_analytical: List[np.ndarray] = []
        self._y:            List[np.ndarray] = []
        self._u:            List[np.ndarray] = []
        self._n_q_trafo:    int = 0   # set on first call from _H_mappings

    def __call__(self, dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        # Capture H_used = the H the MIQP just used (before prediction).
        h_used = dso_ctrl._H_cache.copy() if dso_ctrl._H_cache is not None else None

        # Latch Q_trafo row count from mappings (constant after init).
        if self._n_q_trafo == 0 and dso_ctrl._H_mappings is not None:
            m = dso_ctrl._H_mappings
            self._n_q_trafo = len(m.get("trafos", [])) + len(m.get("trafo3w", []))

        # Run the actual predictor.
        h_predicted = self._predictor(dso_ctrl)

        # If the predictor returned None (warm-up / no-op), treat as identity.
        h_for_next = h_predicted if h_predicted is not None else (
            h_used.copy() if h_used is not None else None
        )

        # Compute the true analytical H at the current operating point.
        h_analytical: Optional[np.ndarray] = None
        if self._combined_net is not None and h_used is not None:
            h_analytical = dso_ctrl.compute_h_analytical(self._combined_net)

        # Capture y and u from the controller's last measurement.
        try:
            feats = get_dso2_features(dso_ctrl)
            y, u  = feats["y"], feats["u"]
        except RuntimeError:
            y = u = None

        if h_used is not None and y is not None:
            self._h_used.append(h_used)
            self._h_predicted.append(h_for_next if h_for_next is not None else h_used.copy())
            if h_analytical is not None:
                self._h_analytical.append(h_analytical)
            self._y.append(y)
            self._u.append(u)

        return h_predicted  # None keeps _H_cache unchanged (observer is transparent)

    def save(self, path: str) -> None:
        """Save recorded arrays to a compressed ``.npz`` file."""
        if not self._h_used:
            print("[observer] no steps recorded; sidecar not saved.")
            return
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        arrays: Dict[str, np.ndarray] = dict(
            h_used      = np.stack(self._h_used),
            h_predicted = np.stack(self._h_predicted),
            y           = np.stack(self._y),
            u           = np.stack(self._u),
        )
        if self._h_analytical:
            if len(self._h_analytical) == len(self._h_used):
                arrays["h_analytical"] = np.stack(self._h_analytical)
            else:
                print(f"[observer] h_analytical has {len(self._h_analytical)} entries "
                      f"vs {len(self._h_used)} steps — skipping (shape mismatch).")
        if self._n_q_trafo > 0:
            arrays["n_q_trafo"] = np.array(self._n_q_trafo)
        np.savez_compressed(path, **arrays)
        keys_str = ", ".join(arrays.keys())
        print(f"[observer] {len(self._h_used)} steps [{keys_str}] -> {os.path.abspath(path)}")


def _make_row_masked_predictor(
    predictor: HPredictor,
    rows_mode: str,
) -> HPredictor:
    """Wrap *predictor* so only selected rows of H are written back to ``_H_cache``.

    rows_mode
    ---------
    ``"all"``       -- full H is written (pass-through, no mask applied).
    ``"q_trafo"``   -- only the Q_trafo rows (``H[:n_q_trafo, :]``) are taken
                       from the predictor's output; all remaining rows are kept
                       from the current ``_H_cache`` unchanged.  The number of
                       Q_trafo rows is read at call-time from
                       ``dso_ctrl._H_mappings`` so it adapts automatically to
                       the network topology.
    ``"q_trafo+v"`` -- the Q_trafo rows AND the following ``n_v`` voltage rows
                       (``H[:n_q_trafo + n_v, :]``, ``n_v = len(voltage_bus_indices)``)
                       are taken from the predictor; the remaining (I_line) rows
                       are kept from ``_H_cache``.

    The wrapped predictor always returns a full-shape array, so the shape
    check in :func:`install_h_corrector` continues to pass unchanged.
    """
    if rows_mode == "all":
        return predictor

    def _masked(dso_ctrl: "DSOController") -> Optional[np.ndarray]:
        H_next = predictor(dso_ctrl)
        if H_next is None:
            return None
        if dso_ctrl._H_cache is None:
            return H_next
        m = dso_ctrl._H_mappings or {}
        n_q_trafo = len(m.get("trafos", [])) + len(m.get("trafo3w", []))
        if n_q_trafo == 0:
            return H_next
        n_rows = n_q_trafo
        if rows_mode == "q_trafo+v":
            n_rows += len(dso_ctrl.config.voltage_bus_indices)   # add the voltage rows
        H_out = dso_ctrl._H_cache.copy()
        H_out[:n_rows] = H_next[:n_rows]
        return H_out

    return _masked


def _setup_h_predictor(cfg: MultiTSOConfig, state: Dict[str, Any]) -> Optional["_DSO2Observer"]:
    """Startup function: prime DSO_2's H, print view, install predictor (and PE noise).

    Returns the :class:`_DSO2Observer` so the caller (:func:`run`) can save
    the sidecar ``.npz`` after the simulation completes.

    Passed to the runner as ``pre_loop_hook`` (wrapped in a lambda that
    binds ``cfg``); runs once after Phase-2 init and before the main
    time loop.  See ``run()`` below for how it is wired in.

    If ``cfg.dso_pe_noise_enabled`` is True, gaussian persistent-
    excitation noise is installed on the DER actuator output
    (innermost wrapper, runs first); the H predictor wrapper is
    installed on top of it.  Wrapper call order at every DSO step
    therefore is:

        runner --> dso_ctrl.step (= H-corrector wrapper)
                     --> PE-noise wrapper
                       --> original step (OFO MIQP)
                       <-- ControllerOutput
                     <-- ControllerOutput, with noise added
                   <-- predictor called, _H_cache rewritten
                  --> result returned to runner --> applied to plant
    """
    dso_ctrl = state["dso_controllers"].get("DSO_2")
    if dso_ctrl is None:
        print("[H] DSO_2 not found in controllers; H predictor not installed.")
        return None

    view = get_dso_h_view(dso_ctrl)
    n_rows_full = len(view["kept_row_mask"])
    n_cols_full = len(view["kept_col_mask"])
    n_q = sum(k == "Q_trafo" for k in view["row_kinds"])
    n_v = sum(k == "V_bus"   for k in view["row_kinds"])
    n_der = sum(k == "Q_cor" for k in view["col_kinds"])
    n_oltc = sum(k == "OLTC" for k in view["col_kinds"])
    print(f"[H] DSO_2 view {view['H'].shape}  "
          f"(full was ({n_rows_full}, {n_cols_full}))")
    print(f"[H]   rows: {n_q} Q_trafo + {n_v} V_bus")
    print(f"[H]   cols: {n_der} Q_cor (DER) + {n_oltc} OLTC")

    # ── Init-H bias injection ─────────────────────────────────────────────
    # Apply a multiplicative perturbation to the init H so that all predictor
    # modes (identity / kalman / ann) start from exactly the same wrong H and
    # we can observe which one recovers faster.  Fixed seed guarantees that
    # every run with the same H_INIT_BIAS_SEED gets an identical bias matrix.
    if H_INIT_BIAS_STD > 0.0:
        combined_net = state.get("net")
        H_base = dso_ctrl.compute_h_analytical(combined_net) if combined_net is not None else None
        if H_base is None:
            H_base = dso_ctrl._H_cache
            print("[H] [warn] compute_h_analytical failed — falling back to split-network H_cache")
        rng_bias = np.random.default_rng(H_INIT_BIAS_SEED)
        noise = rng_bias.normal(1.0, H_INIT_BIAS_STD, size=H_base.shape)
        biased = H_base * noise
        if dso_ctrl._sensitivity_updater is not None:
            dso_ctrl._sensitivity_updater.override_base(biased)
        dso_ctrl._H_cache = biased
        print(f"[H] init bias applied: std={H_INIT_BIAS_STD:.0%}, "
              f"seed={H_INIT_BIAS_SEED} -> same bias for all predictor modes")

    # Install PE noise FIRST (innermost wrapper) so the H corrector
    # installed below sits on the outside.
    if getattr(cfg, "dso_pe_noise_enabled", False):
        rng = np.random.default_rng(cfg.dso_pe_noise_seed)
        pe_mode = getattr(cfg, "dso_pe_mode", "white")
        pe_amp  = float(getattr(cfg, "dso_pe_amplitude_mvar", 0.0))
        install_pe_noise(
            dso_ctrl,
            std_mvar=cfg.dso_pe_noise_std_mvar,
            rng=rng,
            mode=pe_mode,
            amplitude_mvar=pe_amp,
        )
        n_der = len(dso_ctrl.config.der_indices)
        if pe_mode == "orthogonal":
            print(f"[PE] installed ORTHOGONAL rotating excitation amp={pe_amp} Mvar "
                  f"on {n_der} DER channels (per-dir std≈{pe_amp/max(n_der,1)**0.5:.3f}, "
                  f"cycle={n_der} steps, seed={cfg.dso_pe_noise_seed})")
        else:
            print(f"[PE] installed WHITE noise std={cfg.dso_pe_noise_std_mvar} Mvar "
                  f"on {n_der} DER Q_cor channels (seed={cfg.dso_pe_noise_seed})")

    # Kalman runtime knobs.  The two-term Q (excitation + always-on drift) makes
    # the old keep-alive hacks (forgetting factor, prior anchor, P floor/ceiling,
    # dither subtraction) unnecessary, so only two knobs remain.
    _kalman_h_predictor.min_delta_u_norm = 0.0              # update gate (0 = update on every step; raise to skip noise-only steps)
    _kalman_h_predictor.q_scale          = KALMAN_Q_SCALE   # multiplier on the excitation term (1.0 = as trained)
    _kalman_h_predictor.r_model_scale    = KALMAN_R_MODEL_SCALE  # multiplier on the ‖Δu‖^p model-error term of R (0 = fixed R_sensor)
    # Deployment cadence: Q_drift is rescaled from the training period to this at load.
    _kalman_h_predictor.deploy_period_s  = float(getattr(cfg, "dso_period_s", 0.0)) or None
    print(f"[kalman] min_du={_kalman_h_predictor.min_delta_u_norm:.4f}  "
          f"q_scale={_kalman_h_predictor.q_scale}  r_model_scale={_kalman_h_predictor.r_model_scale}  "
          f"deploy_period_s={_kalman_h_predictor.deploy_period_s}  "
          f"(Q = q_scale·s²·Q_excitation + Q_drift;  R = R_sensor + r_model_scale·‖Δu‖^p·R_model)")

    # Inject the live combined net into the numerical oracle so it can
    # finite-difference the true plant sensitivity at each step.
    _numerical_h_predictor.combined_net = state.get("net")
    _numerical_h_predictor.delta_q_mvar = NUMERICAL_ORACLE_DELTA_MVAR

    _PREDICTORS = {
        "identity":  _unity_multiply_predictor,
        "kalman":    _kalman_h_predictor,
        "rls":       _rls_h_predictor,
        "ann":       _ANN_h_predictor,
        "numerical": _numerical_h_predictor,
    }
    if H_PREDICTOR_MODE not in _PREDICTORS:
        raise ValueError(
            f"Unknown H_PREDICTOR_MODE {H_PREDICTOR_MODE!r}. "
            f"Choose one of: {list(_PREDICTORS)}"
        )
    predictor = _PREDICTORS[H_PREDICTOR_MODE]
    if H_PREDICTOR_ROWS != "all":
        predictor = _make_row_masked_predictor(predictor, H_PREDICTOR_ROWS)
        _rows_desc = {"q_trafo": "Q_trafo rows",
                      "q_trafo+v": "Q_trafo + voltage rows"}.get(H_PREDICTOR_ROWS, H_PREDICTOR_ROWS)
        print(f"[H] row mask: {H_PREDICTOR_ROWS!r} — predictor updates {_rows_desc}; "
              f"remaining rows kept from cache")
    combined_net = state.get("net")
    observer  = _DSO2Observer(predictor, combined_net=combined_net)
    install_h_corrector(dso_ctrl, observer)
    print(f"[H] installed predictor: {H_PREDICTOR_MODE!r} ({type(predictor).__name__})"
          f" wrapped in _DSO2Observer")
    return observer


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def run() -> List[MultiTSOIterationRecord]:
    """Run the 003 experiment end-to-end and pickle the log."""
    cfg = make_config()
    out_dir = cfg.result_dir
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("=" * 72)
    print("  RUNNING EXPERIMENT 003_CIGRE_2026 (single-DSO OFO on DSO_2)")
    print("=" * 72)
    print(f"  scenario={cfg.scenario}")
    print(f"  tso_mode={cfg.tso_mode}, tso_local_mode={cfg.tso_local_mode}")
    print(f"  dso_mode={cfg.dso_mode}, dso_ids_to_run={cfg.dso_ids_to_run}")
    print(f"  q_pcc_setpoints={cfg.q_pcc_setpoints_mvar_per_dso}")
    print(f"  n_total_s={cfg.n_total_s:.0f}, dso_period_s={cfg.dso_period_s:.0f}")
    print(f"  H_PREDICTOR_MODE={H_PREDICTOR_MODE!r}  H_PREDICTOR_ROWS={H_PREDICTOR_ROWS!r}")
    print(f"  KALMAN_Q_SCALE={KALMAN_Q_SCALE}  "
          f"(process noise = q_scale·s²·Q_excitation + Q_drift; "
          f"Kalman estimates DER columns only, OLTC columns kept analytical)")
    print(f"  DSO2_INTERFACE_MODE={DSO2_INTERFACE_MODE!r}  "
          f"(dso2_interface_slack={cfg.dso2_interface_slack})")
    _pe_desc = (f"orthogonal rotating, amp={PE_AMPLITUDE_MVAR} Mvar"
                if PE_MODE == "orthogonal" else f"white, std={PE_WHITE_STD_MVAR} Mvar")
    print(f"  PE_MODE={PE_MODE!r}  ({_pe_desc})")
    if FROZEN_OP_POINT:
        print(f"  FROZEN_OP_POINT=True  frozen_at={FROZEN_OP_TIMESTAMP:%Y-%m-%d %H:%M}, contingencies=off")
    else:
        n_ev = len(cfg.contingencies)
        print(f"  FROZEN_OP_POINT=False  (profiles on, contingencies={n_ev})")
    if H_INIT_BIAS_STD > 0.0:
        print(f"  H_INIT_BIAS_STD={int(H_INIT_BIAS_STD*100)}%  H_INIT_BIAS_SEED={H_INIT_BIAS_SEED} "
              f"(same bias for all predictor modes)")
    print(f"  result_dir={out_dir}")

    _observer_ref: List[Optional[_DSO2Observer]] = [None]

    def _pre_loop(state: dict) -> None:
        if cfg.dso2_interface_slack:
            dso_ctrl = state["dso_controllers"].get("DSO_2")
            if dso_ctrl is None:
                print("[iface] DSO_2 not found; interface slack NOT installed.")
            else:
                ifaces = list(dso_ctrl.config.interface_trafo_indices)
                created = decouple_trafo3w_hv_with_slack(state["net"], ifaces)
                print(f"[iface] DSO_2 decoupled from TSO: replaced HV feed of "
                      f"3W trafos {ifaces} with pinned slacks (ext_grid {created}).")
        obs = _setup_h_predictor(cfg, state)
        _observer_ref[0] = obs

    try:
        log = run_multi_tso_dso(cfg, pre_loop_hook=_pre_loop)
    except Exception as exc:
        print(f"  [003] FAILED: {type(exc).__name__}: {exc}")
        log = []

    now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    _suffix = "_" + H_PREDICTOR_MODE
    if FROZEN_OP_POINT:
        _suffix += "_frozen"
    if H_INIT_BIAS_STD > 0.0:
        _suffix += f"_biased{int(H_INIT_BIAS_STD * 100)}pct"
    pkl_path = os.path.join(out_dir, now + _suffix + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [003] wrote {len(log)} records -> {pkl_path}")

    sidecar_path: Optional[str] = None
    observer = _observer_ref[0]
    if observer is not None:
        sidecar_path = os.path.join(out_dir, now + "_dso2_ctrl.npz")
        observer.save(sidecar_path)

    # ── Auto-analysis: show the analysis plot for this run immediately ──
    _analysis = importlib.import_module("experiments.003_analysis")
    _analysis.plot_comparison(pkl_path, sidecar_path)

    return log


def main() -> None:
    run()


if __name__ == "__main__":
    main()
