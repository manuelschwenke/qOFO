#!/usr/bin/env python3
"""
Monte Carlo training data collection for Kalman Q/R and ANN.

Following the approach from the old code (covariances_new):
  - N_OP random timestamps drawn from the full year of profiles give
    diverse load/wind operating points.
  - At each timestamp, a K_PERTURB-step random walk explores the
    Q_cor actuator space with a continuous random-direction step.  The
    OLTC taps are NOT perturbed randomly: a pandapower ``DiscreteTapControl``
    (the same AVR-style coupler controller installed by the runner's Phase-2
    init) moves them in response to the perturbed operating point, so the tap
    trajectory is the realistic voltage-driven response.
  - At every (timestamp, walk-step) pair, pp.runpp is run with
    run_control=True so the QV local loops + coupler OLTCs settle; the
    ANALYTICAL (tangent) sensitivity H is then computed at that operating point
    via ``dso_ctrl.compute_h_analytical`` — the SAME Jacobian-based H the
    controller deploys in its MIQP (not a numerical secant).  It is stored under
    the key "H_analytical" so the KF/ANN consumers (generate_kalman_matrices,
    train_ann_model) need no change.
  - The Kalman process noise is the TWO-TERM split used by _KalmanHPredictor,
    Q_eff = q_scale·s²·Q_excitation + Q_drift, and BOTH matrices are derived here:
      * Q_excitation = cov(ΔH) within each EXCITATION walk (actuator-driven H
        change at a fixed operating point) — the input-driven term, scaled by s²
        at runtime.  Estimated within-walk (not across operating-point boundaries,
        which would inflate it artificially).
      * Q_drift = cov(ΔH) over a DRIFT PROBE at each op-point: the DER q_set is
        held fixed while the profile timestamp is advanced by dso_period_s, so ΔH
        is the exogenous (load/profile-driven) H drift per control step with no
        actuator motion — the always-on keep-alive term.
  - Kalman R is estimated from cov(Δy − H·Δu) residuals (within-walk pairs
    with ‖Δu‖ > threshold) — the tangent-linearisation residual, i.e. exactly
    the model error the controller's tangent H incurs over a deployment step.
  - ANN training data: (y, u, H_analytical) from all samples.

Key differences vs the time-series collectors:
  - Timestamps span the full year instead of a single 4h window.
  - OLTC taps move via a DiscreteTapControl AVR loop (voltage-driven),
    not random ±1 steps, so they track the perturbed operating point.
  - Q is estimated only from within-walk consecutive pairs so the
    diagonal entries reflect genuine step-to-step H drift, not
    inter-episode jumps from profile changes.

Parallelism
-----------
Each operating-point walk is fully independent — the within-walk dH and
residual pairs never cross op-point boundaries — so the MC loop is
embarrassingly parallel.  The N_OP timestamps are split into
``min(n_jobs, N_OP)`` chunks; each chunk is processed in a separate worker
process (joblib ``loky`` backend) that rebuilds its own network + DSO_2
controller once via a 1-step init, then walks its assigned timestamps.  The
results are concatenated in the parent, which then estimates Q/R and trains
the ANN exactly as in the sequential version.

Because every walk uses an independent ``SeedSequence(SEED)`` child stream,
the collected statistics are reproducible given ``SEED`` and statistically
equivalent to the old sequential run, but NOT bit-identical to it (the RNG
draw order differs once the loop is parallelised).

Control:
  MC_N_JOBS     number of worker processes (default -1 = all cores).  Set 1
                to force the sequential path (also useful if loky interferes
                with the pandapower solver setup on a given machine).
  MC_N_OP       op-point timestamps (default 60).
  MC_K_PERTURB  excitation random-walk steps per op-point (default 15).
  MC_K_DRIFT    drift-probe steps per op-point (default 8; → Q_drift).
  MC_SEED       master seed (default 42).
  MC_PARTIAL_OUT  shard mode (see below).

Pipeline
--------
  1. 1-step simulation → initialised net + DSO_2 controller (per worker).
  2. Parallel MC loop per op-point: K_DRIFT drift-probe steps (u fixed, profile
     advanced → Q_drift) + K_PERTURB excitation walk steps (→ Q_excitation, R).
  3. Save training_data.npz (same format as the time-series collectors).
  4. Estimate Q_excitation / Q_drift / R / u_scale inline → kalman_matrices.npz,
     then train_ann_model().
"""

import sys
import os
import time
import importlib
from datetime import datetime, timedelta

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
sys.path.insert(0, _root)

import pandapower as pp
from pandapower.control import DiscreteTapControl
from core.profiles import DEFAULT_PROFILES_CSV, apply_profiles, load_profiles
from experiments.runners import run_multi_tso_dso
from experiments.helpers.plant_io import decouple_trafo3w_hv_with_slack

exp003 = importlib.import_module("experiments.003_S_DSO_CIGRE_2026")

# ── Configuration ──────────────────────────────────────────────────────────────
# Control cadence.  The collector runs at the DEPLOYMENT DSO control period so the
# drift probe measures Q_drift per deploy-step and train_period_s matches the 003
# comparison run (dso_period_s=20) — making the runtime Q_drift period rescale a
# no-op (ratio 1).  Set via STEP_S below (overrides 003.make_config's 60 s).
STEP_S = float(os.environ.get("MC_DT_S", 20.0))   # collector dt_s + dso_period_s [s]
#
# TARGET_STEP_MVAR — the MC walk step magnitude must match the actual per-step OFO
# excitation seen at deployment, because R = cov(Δy − H·Δu) is the linearisation
# residual at that step size and scales ∝ step² (an over-large step inflates R and
# collapses the Kalman gain), and because the runtime s² = ‖Δu/u_scale‖² is ≈1 only
# when the deploy step matches the training step.
#
# T (deployment per-step ‖Δu_DER‖) measured on the DER (Q_cor) columns of u from six
# recent 003 sidecars (results/003_cigre_2026/2026-06-19--*_dso2_ctrl.npz, N=150..600,
# post-transient): mean 0.60–0.67, median 0.52–0.60, per-channel RMS ≈0.24 Mvar —
# stable across N and cadence (PE-dither + setpoint-tracking floor, ~period-independent).
# The MC walk draws step_norm ~ Uniform(0, max_norm), so mean MC step = 0.5·max_norm;
# matching the deployment MEAN ⇒ max_norm = 2·T, i.e. Q_MAX_FRAC = 2·T/‖q_range‖.
# (The old value 10 was ~16× too large → R inflated ~280× → gain suppressed.)
# N_OP / K_PERTURB / SEED accept env overrides so this script can run as one of
# several PARALLEL SHARDS (see MC_PARTIAL_OUT below).
N_OP              = int(os.environ.get("MC_N_OP", 60))        # op-point timestamps (per shard)
K_PERTURB         = int(os.environ.get("MC_K_PERTURB", 15))   # excitation random-walk steps per op-point (Q_excitation)
K_DRIFT           = int(os.environ.get("MC_K_DRIFT", 60))      # drift-probe steps per op-point (Q_drift): u held fixed, profile time advanced
TARGET_STEP_MVAR  = 1.0       # deployment per-step ‖Δu_DER‖; matches cfg.dso_pe_amplitude_mvar (orthogonal PE, steady-state ‖Δu‖≈amplitude)
DU_MIN_NORM  = 0.0      # skip residual if norm(delta_u) < this (0 = keep all within-walk pairs)
SEED         = int(os.environ.get("MC_SEED", 64))            # distinct seeds -> distinct op-points per shard
N_JOBS       = int(os.environ.get("MC_N_JOBS", 15))          # worker processes (-1 = all cores, 1 = sequential)
HEARTBEAT_EVERY = int(os.environ.get("MC_HEARTBEAT_EVERY", 5))  # intra-op progress print every N walk steps (0 = off)
# Shard mode: when MC_PARTIAL_OUT is set, save a partial npz (records + within-walk
# dH/residual pairs) to that path and SKIP the Q/R + ANN steps; _merge_train_mc.py
# concatenates all shards.  Statistically identical to the sequential run because
# within-walk pairs never cross op-point boundaries.
_PARTIAL_OUT = os.environ.get("MC_PARTIAL_OUT", "").strip()

# Timestamps: Jan–Aug 2016 only; test window is Sep 7 which is outside this range.
# Using a strict seasonal split avoids operating-point leakage into the test day.
SAMPLE_START = datetime(2016,  1,  1,  0, 0)
SAMPLE_END   = datetime(2016,  8, 31, 23, 0)
EXCL_START   = datetime(2016,  9,  7,  8, 0)   # not reachable, kept as safety guard
EXCL_END     = datetime(2016,  9,  7, 22, 0)

_RUNPP_KW = dict(
    run_control=True,
    calculate_voltage_angles=True,
    max_iteration=50,   # power-flow Newton iterations
    max_iter=300,       # controller convergence iterations
    numba=False,
)


# ── Measurement / actuator extraction (stateless; explicit args) ────────────────

def _extract_y(net_, q_idx, v_idx, i_idx) -> np.ndarray:
    """Extract y = [Q_trafo | V_bus | I_line] from pandapower results."""
    q_tr  = np.array([net_.res_trafo3w.at[t, "q_hv_mvar"] for t in q_idx],
                     dtype=float)
    v_bus = net_.res_bus.loc[v_idx, "vm_pu"].values.astype(float)
    i_ka  = np.array([net_.res_line.at[li, "i_from_ka"] for li in i_idx],
                     dtype=float)
    return np.concatenate([q_tr, v_bus, i_ka])


def _seed_qv(net_) -> None:
    """Warm-start QV controllers by seeding q_mvar = q_set_mvar.

    With damping=0.1 and tol=0.02 Mvar, the QV loops need ~30 iterations to
    resolve a 0.3 Mvar residual but hundreds if started from a random cold
    state.  Seeding to q_set (= equilibrium at V = V_anchor) cuts the initial
    error to K·|V_anchor - V| which converges in ~20 iterations.
    """
    if "q_mode" not in net_.sgen.columns:
        return
    qv_mask = net_.sgen["q_mode"] == "qv"
    net_.sgen.loc[qv_mask, "q_mvar"] = net_.sgen.loc[qv_mask, "q_set_mvar"]


# ── Per-worker initialisation: build net + DSO_2 controller + parameters ────────

def _build_worker_state(quiet: bool = True) -> dict:
    """Initialise a fresh network + DSO_2 controller in THIS process.

    Runs the 1-step simulation, decouples DSO_2 behind pinned boundary slacks
    (exactly the deployment path in ``003.run()._pre_loop``), then extracts all
    parameters the MC walk needs.  Called once per worker (one chunk = one
    call), and once in the parent for the informative parameter print-out.

    Returns a dict ``S`` carrying the live ``net``/``dso_ctrl`` plus index
    arrays, actuator bounds and the derived ``Q_MAX_FRAC``.
    """
    # The collector drives the plant manually (no predictor/PE install), so the
    # init-bias / frozen flags are irrelevant here, but pin them for clarity.
    exp003.H_INIT_BIAS_STD = 0.0
    exp003.FROZEN_OP_POINT = False

    cfg_init = exp003.make_config()
    # Run the collector at the deployment control cadence (STEP_S, 20 s): the drift
    # probe then measures Q_drift per deploy-step and train_period_s = 20 s matches
    # the 003 comparison (dso_period_s=20) → no runtime Q_drift rescale.  Overrides
    # make_config's 60 s.  Both fields are set so dt_s and the DSO period agree.
    cfg_init.dt_s                 = STEP_S
    cfg_init.dso_period_s         = STEP_S
    cfg_init.n_total_s            = cfg_init.dso_period_s   # exactly one step
    cfg_init.live_plot_controller = False
    cfg_init.live_plot_cascade    = False
    cfg_init.live_plot_system     = False
    cfg_init.verbose              = 0 if quiet else cfg_init.verbose

    _ref: dict = {}

    def _init_hook(state: dict) -> None:
        net_ = state["net"]
        dso_ = state["dso_controllers"].get("DSO_2")
        _ref["net"]      = net_
        _ref["dso_ctrl"] = dso_
        if dso_ is not None:
            exp003.get_dso_h_view(dso_)   # prime H cache + mappings
            # Decouple DSO_2 from the TSO exactly like the deployment path
            # (003.run()._pre_loop): replace the TN feed at each coupling 3W's HV
            # side with a slack pinned to the init op-point, so the MC walk and the
            # estimated Q/R/ANN see the same islanded plant the predictors deploy on.
            if getattr(cfg_init, "dso2_interface_slack", False):
                ifaces = list(dso_.config.interface_trafo_indices)
                created = decouple_trafo3w_hv_with_slack(net_, ifaces)
                if not quiet:
                    print(f"[iface] DSO_2 decoupled from TSO: replaced HV feed of "
                          f"3W trafos {ifaces} with pinned slacks (ext_grid {created}).")

    run_multi_tso_dso(cfg_init, pre_loop_hook=_init_hook)

    net      = _ref.get("net")
    dso_ctrl = _ref.get("dso_ctrl")
    if net is None or dso_ctrl is None:
        raise RuntimeError("Failed to initialise network or DSO_2 controller.")

    # ── extract DSO_2 parameters ────────────────────────────────────────────
    dcfg     = dso_ctrl.config
    der_idx  = np.array(dcfg.der_indices,             dtype=int)
    oltc_idx = np.array(dcfg.oltc_trafo_indices,      dtype=int)
    v_idx    = np.array(dcfg.voltage_bus_indices,     dtype=int)
    q_idx    = np.array(dcfg.interface_trafo_indices, dtype=int)
    i_idx    = np.array(dcfg.current_line_indices,    dtype=int)

    n_der  = len(der_idx)
    n_oltc = len(oltc_idx)
    n_q_tr = len(q_idx)
    n_v    = len(v_idx)
    n_i    = len(i_idx)
    n_y    = n_q_tr + n_v + n_i
    # Input vector u = DER q_set only.  The OLTC tap is NOT an input of the
    # learned model: we estimate the dependency of [Q_trafo | V] on the DER
    # reactive injections alone.  (The taps still move via DiscreteTapControl
    # during the operating-point settling; they are simply not part of u/H.)
    n_u    = n_der

    # Q_cor actuator bounds per DER — use ±0.4·Sn (VDE max range) as fixed bounds.
    # ActuatorBounds computes these dynamically from P, but for the MC walk we want
    # a fixed conservative envelope that is valid across all operating points.
    sn_mva    = np.array(dso_ctrl.actuator_bounds.der_s_rated_mva, dtype=float)
    q_cor_min = -0.4 * sn_mva
    q_cor_max = +0.4 * sn_mva
    q_range   = q_cor_max - q_cor_min              # per-DER range (Mvar)

    # Derive Q_MAX_FRAC so the uniform-walk mean step (= 0.5 * Q_MAX_FRAC * ||q_range||)
    # matches the measured decoupled deployment step TARGET_STEP_MVAR.
    q_range_norm = float(np.linalg.norm(q_range))
    Q_MAX_FRAC   = 2.0 * TARGET_STEP_MVAR / q_range_norm

    # OLTC bounds (3W coupling trafos) — tap_neutral is needed to express the
    # actuator vector as a delta from neutral; tap_min/tap_max are enforced
    # internally by DiscreteTapControl (no manual clipping).
    tap_min     = net.trafo3w.loc[oltc_idx, "tap_min"].values.astype(int)
    tap_max     = net.trafo3w.loc[oltc_idx, "tap_max"].values.astype(int)
    tap_neutral = net.trafo3w.loc[oltc_idx, "tap_neutral"].values.astype(int)

    # ── Install DiscreteTapControl on DSO_2's coupler 3W OLTCs ──────────────
    # In ofo mode the runner DROPS the Phase-2 coupler DiscreteTapControls
    # (the OFO owns the taps), so we re-install them here for the MC walk: they
    # move the taps in response to the perturbed operating point (replacing the
    # old random ±1 perturbation), mirroring the runner's Phase-2 settings
    # (side="mv", element="trafo3w", band = oltc_init_v_target_pu ±
    # dso_oltc_init_tol_pu) so each sampled operating point has realistic,
    # voltage-regulated tap positions.  The analytical (tangent) H is then
    # evaluated at that settled point (no tap perturbation), so the tap
    # controllers stay active throughout.
    v_tgt = float(cfg_init.oltc_init_v_target_pu)
    v_tol = float(cfg_init.dso_oltc_init_tol_pu)
    _before = set(net.controller.index) if hasattr(net, "controller") else set()
    for t3w in oltc_idx:
        DiscreteTapControl(
            net, element_index=int(t3w),
            vm_lower_pu=v_tgt - v_tol,
            vm_upper_pu=v_tgt + v_tol,
            side="mv", element="trafo3w",
        )
    tap_ctrl_idx = sorted(set(net.controller.index) - _before)
    if not quiet:
        print(f"[oltc] installed {len(tap_ctrl_idx)} DiscreteTapControl(s) on DSO_2 "
              f"coupler 3W {list(map(int, oltc_idx))} -> V_mv {v_tgt:.3f} +-{v_tol:.3f} p.u.")

    _csv = cfg_init.profiles_csv if cfg_init.profiles_csv else DEFAULT_PROFILES_CSV
    profiles_df = load_profiles(_csv)

    # Drift-probe profiles at the DSO control cadence.  Holding u fixed and
    # advancing the timestamp by dso_period_s with these sub-15-min linearly
    # interpolated profiles produces the exogenous (load/profile-driven) per-step
    # H drift that Q_drift captures.  load_profiles interpolates the native 15-min
    # CSV down to dso_period_s (the runtime's control period).
    dso_period_s = float(cfg_init.dso_period_s)
    profiles_drift_df = load_profiles(_csv, timestep_s=dso_period_s)

    return dict(
        net=net, dso_ctrl=dso_ctrl, profiles_df=profiles_df,
        profiles_drift_df=profiles_drift_df, dso_period_s=dso_period_s,
        der_idx=der_idx, oltc_idx=oltc_idx, v_idx=v_idx, q_idx=q_idx, i_idx=i_idx,
        n_der=n_der, n_oltc=n_oltc, n_q_tr=n_q_tr, n_v=n_v, n_i=n_i,
        n_y=n_y, n_u=n_u,
        q_cor_min=q_cor_min, q_cor_max=q_cor_max, q_range=q_range,
        q_range_norm=q_range_norm, Q_MAX_FRAC=Q_MAX_FRAC,
        tap_min=tap_min, tap_max=tap_max, tap_neutral=tap_neutral,
        tap_ctrl_idx=tap_ctrl_idx,
    )


# ── Single operating-point walk ─────────────────────────────────────────────────

def _walk_one_op(S: dict, t_mc: datetime, seed, tag: str = "", step_every: int = 0) -> tuple:
    """Random-walk one operating point.  Returns
    (records, dh_within, dh_drift, residuals, du_within, skipped).

    Two sub-walks are run at each operating point, producing the TWO process-noise
    terms of the Kalman's split Q_eff = q_scale·s²·Q_excitation + Q_drift:

    1. Drift probe (``dh_drift``, Q_drift) -- the DER ``q_set`` is HELD at the OFO
       setpoint while the profile timestamp is advanced by ``dso_period_s`` for
       ``K_DRIFT`` steps (sub-15-min interpolated profiles).  ΔH then reflects the
       exogenous load/profile-driven H drift per control step, with NO actuator
       motion.  Time is restored to ``t_mc`` afterwards.
    2. Excitation walk (``dh_within``/``residuals``/``du_within``, Q_excitation/R)
       -- the continuous DER ``q_set`` is randomly perturbed at the fixed op-point;
       ΔH reflects actuator-driven H change and ``u`` records the DER q_set alone
       (the OLTC tap is NOT a model input).

    In both, the coupler OLTC taps are driven by the installed
    ``DiscreteTapControl`` (voltage band held by ``pp.runpp(run_control=True)``)
    so each point has realistic taps; the analytical (tangent) H is taken from
    :meth:`DSOController.compute_h_analytical` and sliced to its DER columns, so
    the learned model is ``[Q_trafo | V] = H_DER * q_set``.

    ``records``    : list of {"y", "u", "H_analytical"} dicts (excitation samples).
    ``dh_within``  : list of ΔH = vec(H_k) - vec(H_{k-1}) over excitation steps.
    ``dh_drift``   : list of ΔH over drift-probe steps (u fixed, profile advanced).
    ``residuals``  : list of Δy - C(Δu)·vec(H_{k-1}) for excited within-walk pairs.
    ``skipped``    : count of non-converged steps (incl. the op-point equilibrium).
    """
    net          = S["net"]
    dso_ctrl     = S["dso_ctrl"]
    profiles_df  = S["profiles_df"]
    profiles_drift_df = S["profiles_drift_df"]
    dso_period_s = S["dso_period_s"]
    der_idx      = S["der_idx"]
    oltc_idx     = S["oltc_idx"]
    v_idx, q_idx, i_idx = S["v_idx"], S["q_idx"], S["i_idx"]
    n_der, n_y   = S["n_der"], S["n_y"]
    q_cor_min, q_cor_max = S["q_cor_min"], S["q_cor_max"]
    q_range, Q_MAX_FRAC  = S["q_range"], S["Q_MAX_FRAC"]

    rng = np.random.default_rng(seed)

    records: list = []
    dh_within: list = []
    dh_drift: list = []    # drift-probe ΔH (u fixed, profile advanced) → Q_drift
    residuals: list = []
    residual_du: list = [] # ‖Δu‖ per residual (for the ‖Δu‖^p R_model term)
    du_within: list = []   # every within-walk Δu (for the per-channel u_scale)
    skipped = 0

    # ── Apply profiles and converge to operating-point equilibrium ───────────
    # DiscreteTapControl settles the coupler taps to the op-point voltage band.
    apply_profiles(net, profiles_df, t_mc)
    _seed_qv(net)
    try:
        pp.runpp(net, **_RUNPP_KW)
    except Exception:
        return records, dh_within, dh_drift, residuals, residual_du, du_within, 1   # op-point unusable

    # Initial q_set = OFO setpoint at this op-point (read from sgen table)
    q_set_now = net.sgen.loc[der_idx, "q_set_mvar"].values.astype(float).copy()

    # ── Drift probe (Q_drift): hold u fixed, advance profile time ─────────────
    # The DER q_set is held at the OFO setpoint while the profile timestamp is
    # advanced by dso_period_s per step.  ΔH then captures the exogenous
    # (load/profile-driven) H drift per control step with NO actuator motion —
    # the always-on Q_drift term.  The taps still track via DiscreteTapControl.
    # compute_h_analytical deep-copies the net, so only apply_profiles+runpp here
    # mutate live state; profiles are restored to t_mc before the excitation walk.
    net.sgen.loc[der_idx, "q_set_mvar"] = q_set_now
    prev_h_drift = None
    t_drift = t_mc
    for _ in range(K_DRIFT):
        t_drift = t_drift + timedelta(seconds=dso_period_s)
        apply_profiles(net, profiles_drift_df, t_drift)
        _seed_qv(net)
        try:
            pp.runpp(net, **_RUNPP_KW)
        except Exception:
            skipped += 1
            continue
        H_d = dso_ctrl.compute_h_analytical(net)
        if H_d is None:
            skipped += 1
            continue
        H_d = H_d[:, :n_der].flatten()
        if prev_h_drift is not None:
            dh_drift.append(H_d - prev_h_drift)
        prev_h_drift = H_d

    # Restore the op-point profiles + equilibrium for the excitation walk.
    apply_profiles(net, profiles_df, t_mc)
    net.sgen.loc[der_idx, "q_set_mvar"] = q_set_now
    _seed_qv(net)
    try:
        pp.runpp(net, **_RUNPP_KW)
    except Exception:
        return records, dh_within, dh_drift, residuals, residual_du, du_within, skipped + 1

    prev_rec: dict | None = None

    for k in range(K_PERTURB):
        # ── Random walk: continuous q_set step (DER only) ─────────────────
        direction  = rng.standard_normal(n_der)
        direction /= np.linalg.norm(direction) + 1e-12
        max_norm   = Q_MAX_FRAC * np.linalg.norm(q_range)
        step_norm  = rng.uniform(0.0, max_norm)
        q_set_new  = np.clip(q_set_now + direction * step_norm,
                             q_cor_min, q_cor_max)

        # ── Apply q_set; DiscreteTapControl moves the taps to hold the MV ──
        #    voltage band (run_control=True iterates the QV loops + OLTC AVR).
        net.sgen.loc[der_idx, "q_set_mvar"] = q_set_new
        _seed_qv(net)
        try:
            pp.runpp(net, **_RUNPP_KW)
        except Exception:
            skipped += 1
            continue

        # Settled operating point (taps moved by DiscreteTapControl to regulate
        # the MV-side voltage; the tap position is not recorded as a model input).
        y = _extract_y(net, q_idx, v_idx, i_idx)

        # ── Analytical (tangent) H — the SAME sensitivity the controller uses ──
        # Train the ANN (and estimate the KF Q/R) on the controller's analytical
        # Jacobian-based H, NOT a numerical secant, so the learned/estimated target
        # matches the tangent H the MIQP actually deploys against.
        # ``compute_h_analytical`` deep-copies + re-converges the net and applies
        # exactly the transforms of ``_build_sensitivity_matrix`` (i.e. the H that
        # ends up in the controller's ``_H_cache``).  It does NOT perturb tap_pos,
        # so the DiscreteTapControl needs no disabling.  Stored under the key
        # "H_analytical" (now accurate) — the KF/ANN consumers are unchanged.
        H = dso_ctrl.compute_h_analytical(net)
        if H is None:
            skipped += 1
            continue
        # Keep only the DER columns of H (drop OLTC + shunt).  H column order
        # is [DER | OLTC | shunt]; the first n_der columns are ∂[Q_trafo|V]/∂q_set.
        H = H[:, :n_der]

        # ── Measurements ──────────────────────────────────────────────────
        # u = DER q_set only (no OLTC tap columns).
        u = np.asarray(q_set_new, dtype=np.float64).copy()

        rec = {"y": y, "u": u, "H_analytical": H}
        records.append(rec)

        # ── Within-walk consecutive pairs for Q, R and u_scale ────────────
        if prev_rec is not None:
            dh = H.flatten() - prev_rec["H_analytical"].flatten()
            dh_within.append(dh)

            du = u - prev_rec["u"]
            du_within.append(du)   # all pairs (incl. small) → per-channel Δu RMS
            if np.linalg.norm(du) >= DU_MIN_NORM:
                dy  = y - prev_rec["y"]
                C   = np.kron(np.eye(n_y), du.reshape(1, -1))   # (n_y, n_state)
                res = dy - C @ prev_rec["H_analytical"].flatten()
                residuals.append(res)
                residual_du.append(float(np.linalg.norm(du)))

        prev_rec  = rec
        q_set_now = q_set_new

        # Intra-op heartbeat (flushed): a finer progress signal than the per-op
        # line in _process_chunk, since one op = K_PERTURB slow walk steps.
        if step_every and (k + 1) % step_every == 0:
            print(f"  {tag} step {k + 1}/{K_PERTURB} | {len(records)} ok, "
                  f"{skipped} skipped", flush=True)

    return records, dh_within, dh_drift, residuals, residual_du, du_within, skipped


# ── Chunk worker: one init, then walk a group of timestamps ─────────────────────

def _process_chunk(ts_chunk: list, seed_chunk: list) -> tuple:
    """Process a contiguous group of operating points in one worker process.

    Builds the network + DSO_2 controller ONCE (one init per chunk → one per
    worker), then walks every timestamp in the chunk, accumulating the results.
    Returns the same 6-tuple as :func:`_walk_one_op`, concatenated over the chunk.
    """
    pid = os.getpid()
    t_init = time.perf_counter()
    S = _build_worker_state(quiet=True)
    n_ops = len(ts_chunk)
    n_der = S["n_der"]
    print(f"[w{pid}] init done in {time.perf_counter() - t_init:.0f}s; "
          f"walking {n_ops} op-points", flush=True)

    records: list = []
    dh_within: list = []
    dh_drift: list = []
    residuals: list = []
    residual_du: list = []
    du_within: list = []
    skipped = 0
    t0 = time.perf_counter()
    for j, (t_mc, seed) in enumerate(zip(ts_chunk, seed_chunk), start=1):
        recs, dh, dhd, res, rdu, du, sk = _walk_one_op(
            S, t_mc, seed, tag=f"[w{pid}] op {j}/{n_ops}", step_every=HEARTBEAT_EVERY
        )
        records.extend(recs)
        dh_within.extend(dh)
        dh_drift.extend(dhd)
        residuals.extend(res)
        residual_du.extend(rdu)
        du_within.extend(du)
        skipped += sk
        # Per-op heartbeat (flushed): cumulative samples/skips, elapsed + ETA
        # for this worker's chunk.
        el  = time.perf_counter() - t0
        eta = el / j * (n_ops - j)
        print(f"[w{pid}] op {j}/{n_ops} | {len(records)} samples, "
              f"{len(dh_drift)} drift pairs, {skipped} skipped "
              f"| {el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left",
              flush=True)
    return records, dh_within, dh_drift, residuals, residual_du, du_within, skipped


# ── Timestamp sampling ──────────────────────────────────────────────────────────

def _sample_timestamps(rng: np.random.Generator) -> list:
    """Sample N_OP timestamps from the Jan–Aug 2016 window (excl. Sep 7)."""
    total_hours = int((SAMPLE_END - SAMPLE_START).total_seconds() / 3600)
    candidate_h = rng.integers(0, total_hours, size=N_OP * 4)
    timestamps: list = []
    for h in candidate_h:
        t = SAMPLE_START + timedelta(hours=int(h))
        if not (EXCL_START <= t <= EXCL_END):
            timestamps.append(t)
        if len(timestamps) == N_OP:
            break
    if len(timestamps) < N_OP:
        raise RuntimeError(f"Could not sample {N_OP} timestamps after exclusion window.")
    return timestamps


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    from joblib import Parallel, delayed, effective_n_jobs

    # ── Step 1: initialise once in the parent (parameters + informative print) ──
    print("\n" + "=" * 72)
    print("  STEP 1: initialising network (1-step run)")
    print("=" * 72)
    S = _build_worker_state(quiet=False)
    n_der, n_oltc, n_y, n_u = S["n_der"], S["n_oltc"], S["n_y"], S["n_u"]
    n_state = n_y * n_u
    print(f"  control cadence STEP_S = {STEP_S:.0f} s (dt_s + dso_period_s; "
          f"= 003 deploy period → no Q_drift rescale)")
    print(f"  ||q_range|| = {S['q_range_norm']:.3f} Mvar  ->  "
          f"Q_MAX_FRAC = 2*{TARGET_STEP_MVAR:.2f}/||q_range|| = {S['Q_MAX_FRAC']:.4f}")
    print(f"  n_der={n_der}  n_oltc={n_oltc}  n_y={n_y}  n_u={n_u}")
    print(f"  Q_cor range: [{S['q_cor_min'].min():.2f}, {S['q_cor_max'].max():.2f}] Mvar")
    print(f"  OLTC tap range: {S['tap_min']} .. {S['tap_max']}  neutral={S['tap_neutral']}")

    # ── Step 2: sample timestamps + per-op RNG streams ──────────────────────────
    # Timestamps from a SeedSequence(SEED) child; each walk gets its own independent
    # child stream so the parallel collection is reproducible given SEED (but not
    # bit-identical to the old sequential RNG order).
    root_ss  = np.random.SeedSequence(SEED)
    children = root_ss.spawn(N_OP + 1)
    timestamps = _sample_timestamps(np.random.default_rng(children[0]))
    walk_seeds = list(children[1:])

    # ── Step 3: parallel MC collection ──────────────────────────────────────────
    n_eff = max(1, min(effective_n_jobs(N_JOBS), N_OP))
    idx_chunks = [c for c in np.array_split(np.arange(N_OP), n_eff) if len(c) > 0]
    print("\n" + "=" * 72)
    print(f"  STEP 2: MC data collection ({n_eff} worker process(es))")
    print(f"  {N_OP} op-points × {K_PERTURB} walk steps = {N_OP * K_PERTURB} target samples")
    print(f"  {len(idx_chunks)} chunks (one network init per chunk)")
    print("=" * 72)

    if N_JOBS == 1:
        chunk_out = [
            _process_chunk([timestamps[i] for i in idx], [walk_seeds[i] for i in idx])
            for idx in idx_chunks
        ]
    else:
        chunk_out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
            delayed(_process_chunk)(
                [timestamps[i] for i in idx], [walk_seeds[i] for i in idx]
            )
            for idx in idx_chunks
        )

    # ── Merge (chunks returned in input order → records stay op-point ordered) ──
    records: list = []
    delta_h_within: list = []
    delta_h_drift: list = []
    residuals_list: list = []
    residual_du_list: list = []
    du_within_list: list = []
    skipped = 0
    for recs, dh, dhd, res, rdu, du, sk in chunk_out:
        records.extend(recs)
        delta_h_within.extend(dh)
        delta_h_drift.extend(dhd)
        residuals_list.extend(res)
        residual_du_list.extend(rdu)
        du_within_list.extend(du)
        skipped += sk

    # Per-channel actuator-step scale: RMS of within-walk Δu per input channel.
    # Mirrors generate_kalman_matrices so the runtime Σ_q = s²·Σ_q0 normalisation
    # (DER Mvar vs OLTC tap steps) is consistent with the MC-estimated Q/R; the
    # KF would otherwise fall back to u_scale = ones.  Zero-motion channels (e.g.
    # an OLTC that never moves under DiscreteTapControl) are guarded to 1.0.
    if du_within_list:
        DU      = np.stack(du_within_list)              # (M, n_u)
        u_scale = np.sqrt(np.mean(DU ** 2, axis=0))     # (n_u,)
        u_scale = np.where(u_scale > 1e-9, u_scale, 1.0)
    else:
        DU      = np.zeros((0, n_u))
        u_scale = np.ones(n_u)

    n_samples = len(records)
    print(f"\n  Collected {n_samples} samples, {len(delta_h_within)} excitation dH "
          f"pairs, {len(delta_h_drift)} drift dH pairs  "
          f"({skipped} skipped: non-convergence)")
    if n_samples < 10:
        raise RuntimeError("Too few samples — increase N_OP or check power flow convergence.")

    # ── Step 4: stack samples ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 3: stacking samples")
    print("=" * 72)
    arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
    for k, v in arrays.items():
        print(f"    {k}: {v.shape}")

    # ── Shard mode: save partial (records + within-walk pairs) and stop ──────────
    if _PARTIAL_OUT:
        DH  = np.stack(delta_h_within) if delta_h_within else np.zeros((0, n_state))
        DHD = np.stack(delta_h_drift)  if delta_h_drift  else np.zeros((0, n_state))
        RES = np.stack(residuals_list) if residuals_list else np.zeros((0, n_y))
        RDU = np.asarray(residual_du_list, dtype=float)
        _pdir = os.path.dirname(os.path.abspath(_PARTIAL_OUT)) or "."
        os.makedirs(_pdir, exist_ok=True)
        np.savez(_PARTIAL_OUT, _dh_within=DH, _dh_drift=DHD, _residuals=RES,
                 _residual_du=RDU, _du_within=DU, **arrays)
        print(f"\n  [shard] {n_samples} samples, {DH.shape[0]} excitation dH pairs, "
              f"{DHD.shape[0]} drift dH pairs, {RES.shape[0]} residuals, "
              f"{DU.shape[0]} du pairs -> {_PARTIAL_OUT}")
        print("  [shard] Q/R + ANN deferred to _merge_train_mc.py")
        return

    out_path = os.path.abspath(exp003._TRAINING_DATA_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"  {n_samples} samples -> {out_path}")

    # ── Step 5: two-term Kalman process noise + R from MC statistics ─────────────
    # Q_excitation (input-driven, scaled by s² at runtime)  ← excitation-walk ΔH.
    # Q_drift      (disturbance-driven, always on)          ← drift-probe ΔH.
    print("\n" + "=" * 72)
    print("  STEP 4: estimating Kalman Q_excitation / Q_drift / R from MC statistics")
    print("=" * 72)

    if len(delta_h_within) >= 2:
        DH  = np.stack(delta_h_within)               # (M, n_state)
        Q_excitation = np.diag(np.var(DH, axis=0))   # diagonal
        print(f"  Q_excitation from {len(delta_h_within)} excitation-walk dH pairs")
    else:
        Q_excitation = np.eye(n_state) * 1e-6
        print("  [warn] too few excitation pairs for Q_excitation — using I*1e-6")

    if len(delta_h_drift) >= 2:
        DHD = np.stack(delta_h_drift)                # (M, n_state)
        Q_drift = np.diag(np.var(DHD, axis=0))       # diagonal
        print(f"  Q_drift from {len(delta_h_drift)} drift-probe dH pairs "
              f"(u fixed, profile advanced by dso_period_s)")
    else:
        Q_drift = np.eye(n_state) * 1e-9
        print("  [warn] too few drift pairs for Q_drift — using I*1e-9 (floor)")

    # ── Input-dependent R: R_eff = R_sensor + ‖Δu‖^p·R_model ────────────────────
    # Raw MC residual cov is near rank-deficient (cond ~1e10); the diagonal R_sensor
    # floor + diagonal R_model make R_eff PD and well-conditioned at every step.
    # On this plant the residual is first-order (r ∝ ‖Δu‖, p=2).  Row split
    # [Q_trafo (n_q) | V_bus (n_v)].
    n_q = int(S["n_q_tr"])
    p_exp = exp003.KALMAN_R_DU_EXPONENT
    if len(residuals_list) >= 2:
        RES = np.stack(residuals_list)               # (M, n_y)
        RDU = np.asarray(residual_du_list)           # (M,)
        R_sensor, R_model = exp003.estimate_measurement_noise_terms(RES, RDU, n_q, n_y)
        n_used = int((RDU > exp003.KALMAN_R_MODEL_MIN_DU).sum())
        print(f"  R terms from {len(residuals_list)} residuals "
              f"({n_used} used for R_model, p={p_exp})")
    else:
        R_sensor = exp003.floor_measurement_noise_R(np.zeros((n_y, n_y)), n_q=n_q)
        R_model  = np.zeros((n_y, n_y))
        print("  [warn] too few excited steps for R — R_model=0, sensor floor only")

    du_med = float(np.median(residual_du_list)) if residual_du_list else 0.0
    R = R_sensor + exp003.KALMAN_R_MODEL_SCALE * (du_med ** p_exp) * R_model   # representative

    # Ensure PD (Q only; the R_sensor floor already makes R_eff PD)
    Q_excitation += 1e-12 * np.eye(n_state)
    Q_drift      += 1e-12 * np.eye(n_state)

    # train_period_s = the DSO control period the drift probe used (dso_period_s).
    # The runtime rescales Q_drift from this cadence to its own deployment period.
    train_period_s = float(S["dso_period_s"])
    kalman_path = os.path.abspath(exp003._KALMAN_MATRICES_PATH)
    os.makedirs(os.path.dirname(kalman_path), exist_ok=True)
    np.savez(kalman_path, Q_excitation=Q_excitation, Q_drift=Q_drift,
             R_sensor=R_sensor, R_model=R_model, R=R, r_du_exponent=p_exp,
             u_scale=u_scale, train_period_s=train_period_s, n_q=n_q)
    qe_d, qd_d = np.diag(Q_excitation), np.diag(Q_drift)
    rs_d, rm_d = np.diag(R_sensor), np.diag(R_model)
    print(f"  Q_excitation {Q_excitation.shape}  diag: [{qe_d.min():.2e}, {qe_d.max():.2e}]")
    print(f"  Q_drift      {Q_drift.shape}  diag: [{qd_d.min():.2e}, {qd_d.max():.2e}]")
    print(f"  R_sensor     {R_sensor.shape}  diag: [{rs_d.min():.2e}, {rs_d.max():.2e}]  "
          f"(σ_Q={exp003.KALMAN_R_SIGMA_Q}, σ_V={exp003.KALMAN_R_SIGMA_V})")
    print(f"  R_model      {R_model.shape}  diag: [{rm_d.min():.2e}, {rm_d.max():.2e}]  "
          f"(×‖Δu‖^{p_exp}; median ‖Δu‖={du_med:.3f})")
    print(f"  R_eff@median cond {np.linalg.cond(R):.2e}, "
          f"diag [{np.diag(R).min():.2e}, {np.diag(R).max():.2e}]")
    print(f"  u_scale ({u_scale.shape[0]}) from {DU.shape[0]} within-walk du: "
          f"[{u_scale.min():.3f}, {u_scale.max():.3f}]")
    print(f"  train_period_s = {train_period_s:.0f} s "
          f"(Q_drift cadence; runtime rescales to its deploy period)")
    print(f"  -> {kalman_path}")

    # ── Step 6: retrain ANN ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 5: training ANN on MC data")
    print("=" * 72)
    exp003.train_ann_model()
    print("[ann] done.")

    print("\n" + "=" * 72)
    print("  MC COLLECTION + TRAINING COMPLETE")
    print(f"  {n_samples} samples from {N_OP} operating points × {K_PERTURB} walk steps")
    print("  -> Run _batch_run_biased.py or _batch_run_changing.py to evaluate")
    print("=" * 72)


if __name__ == "__main__":
    main()
