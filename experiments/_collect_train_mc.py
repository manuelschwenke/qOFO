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
  - Kalman Q is estimated from cov(ΔH) within each walk (not across
    operating-point boundaries, which would inflate Q artificially).
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
  MC_K_PERTURB  random-walk steps per op-point (default 15).
  MC_SEED       master seed (default 42).
  MC_PARTIAL_OUT  shard mode (see below).

Pipeline
--------
  1. 1-step simulation → initialised net + DSO_2 controller (per worker).
  2. Parallel MC loop: N_OP timestamps × K_PERTURB actuator walk steps.
  3. Save training_data.npz (same format as the time-series collectors).
  4. Call generate_kalman_matrices() and train_ann_model().
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
# The MC walk step magnitude must match the actual per-step OFO excitation seen at
# deployment so that R captures the true linearisation residual at that step size.
# The residual Δy − H·Δu scales ∝ step², so a too-large training step inflates R and
# suppresses the runtime Kalman gain (the matched-step method is Mismatch 4).
#
# Timing (003.make_config): dso_period_s = 20 s, so the DSO/Kalman fires once per
# 20 s record.  The per-step DER motion is PE-floor-dominated (PE dither
# σ_u ≈ 0.25 Mvar/channel), hence roughly cadence-independent.
#
# T (deployment per-step ||delta_u_DER||) measured over the 10 DER channels from a
# 003 full-day run (results/003_cigre_2026/2026-06-18--11-54-34_dso2_ctrl.npz, 2160
# records, 20 s cadence): steady-state mean 0.615 / median 0.556 Mvar; per-channel
# RMS ≈ 0.245.  (An older 60 s reference gave 3.39 Mvar — that cadence is obsolete.)
# The MC walk draws step_norm ~ Uniform(0, max_norm), so mean MC step = 0.5 * max_norm.
# Matching the deployment mean => max_norm = 2 * T, i.e. Q_MAX_FRAC = 2*T / ||q_range||.
# Q_MAX_FRAC is derived from TARGET_STEP_MVAR below, once ||q_range|| is known.
# N_OP / K_PERTURB / SEED accept env overrides so this script can run as one of
# several PARALLEL SHARDS (see MC_PARTIAL_OUT below).  Defaults reproduce the
# single-process run (N_OP=60, K_PERTURB=15, SEED=42).
N_OP              = int(os.environ.get("MC_N_OP", 60))        # op-point timestamps (per shard)
K_PERTURB         = int(os.environ.get("MC_K_PERTURB", 15))   # random-walk steps per op-point
TARGET_STEP_MVAR  = 0.6       # measured deployment per-step ||delta_u_DER|| (mean) at 20 s cadence
DU_MIN_NORM  = 0.0      # skip residual if norm(delta_u) < this (0 = keep all within-walk pairs)
SEED         = int(os.environ.get("MC_SEED", 64))            # distinct seeds -> distinct op-points per shard
N_JOBS       = int(os.environ.get("MC_N_JOBS", 15))          # worker processes (-1 = all cores, 1 = sequential)
HEARTBEAT_EVERY = int(os.environ.get("MC_HEARTBEAT_EVERY", 5))  # intra-op progress print every N walk steps (0 = off)
# Shard mode: when MC_PARTIAL_OUT is set, save a partial npz (records + within-walk
# dH/residual pairs) to that path and SKIP the Q/R + ANN steps; _merge_train_mc.py
# concatenates all shards.  Statistically identical to the sequential run because
# within-walk pairs never cross op-point boundaries.
_PARTIAL_OUT = os.environ.get("MC_PARTIAL_OUT", "").strip()

# ── P_init / P_reset producer modes (gated by env; see _run_*_mode) ─────────────
# These two modes reuse the same _build_worker_state init but, instead of the
# random-walk Q/R collection, sample the spread of the analytical H under (a)
# admittance/model error → P_init_diag and (b) credible single outages →
# P_reset_diag.  Each LOADS the existing kalman_matrices.npz (Q/R/u_scale from a
# prior normal run) and merges its per-entry diagonal back in, so run the normal
# MC first, then these.  Values are modeling assumptions and accept env overrides.
def _envflag(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")

MC_ADM_LINE_FRAC  = float(os.environ.get("MC_ADM_LINE_FRAC", 0.15))   # ±frac on line r,x (correlated)
MC_ADM_TRAFO_FRAC = float(os.environ.get("MC_ADM_TRAFO_FRAC", 0.08))  # ±frac on trafo vk,vkr (per-element)
MC_ADM_RX_CORR    = _envflag("MC_ADM_RX_CORRELATED", True)            # scale r,x by the same per-line factor
MC_PERTURB_N_OP   = int(os.environ.get("MC_PERTURB_N_OP", 50))        # op-points for the admittance P_init MC
MC_PERTURB_K      = int(os.environ.get("MC_PERTURB_K", 6))            # perturbations per op-point
MC_OUTAGE_N_OP    = int(os.environ.get("MC_OUTAGE_N_OP", 20))         # op-points for the outage P_reset MC
MC_OUTAGE_INCLUDE_TRAFO3W = _envflag("MC_OUTAGE_INCLUDE_TRAFO3W", True)  # include interface 3W coupler trips
MC_OUTAGE_STAT    = os.environ.get("MC_OUTAGE_STAT", "var").strip()   # "var" (spread) | "max" (worst-case)

# ── R sensor-noise floor (noise-free MC → raw R is rank-deficient at small step) ──
# Defaults inherit the runtime constants in exp003; override per row-type via env.
MC_R_SIGMA_Q  = float(os.environ.get("MC_R_SIGMA_Q", exp003.KALMAN_R_SIGMA_Q))   # Mvar, interface-Q rows
MC_R_SIGMA_V  = float(os.environ.get("MC_R_SIGMA_V", exp003.KALMAN_R_SIGMA_V))   # p.u., voltage rows
MC_R_SIGMA_I  = float(os.environ.get("MC_R_SIGMA_I", exp003.KALMAN_R_SIGMA_I))   # kA, line-current rows
MC_R_DIAGONAL = _envflag("MC_R_DIAGONAL", exp003.KALMAN_R_DIAGONAL)              # drop off-diagonals first

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

    profiles_df = load_profiles(
        cfg_init.profiles_csv if cfg_init.profiles_csv else DEFAULT_PROFILES_CSV
    )

    return dict(
        net=net, dso_ctrl=dso_ctrl, profiles_df=profiles_df,
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
    """Random-walk one operating point.  Returns (records, dh_within, residuals, skipped).

    Only the continuous DER ``q_set`` is perturbed, and ``u`` records the DER
    q_set alone -- the OLTC tap is NOT a model input.  The coupler OLTC taps are
    driven by the installed ``DiscreteTapControl`` (voltage band held by
    ``pp.runpp(run_control=True)``) so each operating point has realistic taps;
    the analytical (tangent) H is then taken from
    :meth:`DSOController.compute_h_analytical` and sliced to its DER columns, so
    the learned model is ``[Q_trafo | V] = H_DER * q_set``.

    ``records``    : list of {"y", "u", "H_analytical"} dicts (converged samples).
    ``dh_within``  : list of ΔH = vec(H_k) - vec(H_{k-1}) over consecutive walk steps.
    ``residuals``  : list of Δy - C(Δu)·vec(H_{k-1}) for excited within-walk pairs.
    ``skipped``    : count of non-converged steps (incl. the op-point equilibrium).
    """
    net          = S["net"]
    dso_ctrl     = S["dso_ctrl"]
    profiles_df  = S["profiles_df"]
    der_idx      = S["der_idx"]
    oltc_idx     = S["oltc_idx"]
    v_idx, q_idx, i_idx = S["v_idx"], S["q_idx"], S["i_idx"]
    n_der, n_y   = S["n_der"], S["n_y"]
    q_cor_min, q_cor_max = S["q_cor_min"], S["q_cor_max"]
    q_range, Q_MAX_FRAC  = S["q_range"], S["Q_MAX_FRAC"]

    rng = np.random.default_rng(seed)

    records: list = []
    dh_within: list = []
    residuals: list = []
    du_within: list = []   # every within-walk Δu (for the Σ_q per-channel u_scale)
    skipped = 0

    # ── Apply profiles and converge to operating-point equilibrium ───────────
    # DiscreteTapControl settles the coupler taps to the op-point voltage band.
    apply_profiles(net, profiles_df, t_mc)
    _seed_qv(net)
    try:
        pp.runpp(net, **_RUNPP_KW)
    except Exception:
        return records, dh_within, residuals, du_within, 1   # whole op-point unusable

    # Initial q_set = OFO setpoint at this op-point (read from sgen table)
    q_set_now = net.sgen.loc[der_idx, "q_set_mvar"].values.astype(float).copy()

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

        prev_rec  = rec
        q_set_now = q_set_new

        # Intra-op heartbeat (flushed): a finer progress signal than the per-op
        # line in _process_chunk, since one op = K_PERTURB slow walk steps.
        if step_every and (k + 1) % step_every == 0:
            print(f"  {tag} step {k + 1}/{K_PERTURB} | {len(records)} ok, "
                  f"{skipped} skipped", flush=True)

    return records, dh_within, residuals, du_within, skipped


# ── Chunk worker: one init, then walk a group of timestamps ─────────────────────

def _process_chunk(ts_chunk: list, seed_chunk: list) -> tuple:
    """Process a contiguous group of operating points in one worker process.

    Builds the network + DSO_2 controller ONCE (one init per chunk → one per
    worker), then walks every timestamp in the chunk, accumulating the results.
    Returns the same 5-tuple as :func:`_walk_one_op`, concatenated over the chunk.
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
    residuals: list = []
    du_within: list = []
    skipped = 0
    t0 = time.perf_counter()
    for j, (t_mc, seed) in enumerate(zip(ts_chunk, seed_chunk), start=1):
        recs, dh, res, du, sk = _walk_one_op(
            S, t_mc, seed, tag=f"[w{pid}] op {j}/{n_ops}", step_every=HEARTBEAT_EVERY
        )
        records.extend(recs)
        dh_within.extend(dh)
        residuals.extend(res)
        du_within.extend(du)
        skipped += sk
        # Per-op heartbeat (flushed): cumulative samples/skips, elapsed + ETA
        # for this worker's chunk.
        el  = time.perf_counter() - t0
        eta = el / j * (n_ops - j)
        print(f"[w{pid}] op {j}/{n_ops} | {len(records)} samples, {skipped} skipped "
              f"| {el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left",
              flush=True)
    return records, dh_within, residuals, du_within, skipped


# ── Timestamp sampling ──────────────────────────────────────────────────────────

def _sample_timestamps(rng: np.random.Generator, n: int | None = None) -> list:
    """Sample ``n`` (default N_OP) timestamps from the Jan–Aug 2016 window (excl. Sep 7)."""
    n = N_OP if n is None else int(n)
    total_hours = int((SAMPLE_END - SAMPLE_START).total_seconds() / 3600)
    candidate_h = rng.integers(0, total_hours, size=n * 4)
    timestamps: list = []
    for h in candidate_h:
        t = SAMPLE_START + timedelta(hours=int(h))
        if not (EXCL_START <= t <= EXCL_END):
            timestamps.append(t)
        if len(timestamps) == n:
            break
    if len(timestamps) < n:
        raise RuntimeError(f"Could not sample {n} timestamps after exclusion window.")
    return timestamps


# ── P_init / P_reset producer modes ─────────────────────────────────────────────

def _merge_into_kalman_npz(**new_keys) -> str:
    """Load the existing kalman_matrices.npz, merge ``new_keys`` in, re-save.

    Keeps Q/R/u_scale (and any other previously-written diagonals) intact so the
    P_init/P_reset producers compose with the normal Q/R run and with each other.
    """
    path = os.path.abspath(exp003._KALMAN_MATRICES_PATH)
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} not found — run the normal MC (Q/R/u_scale) first, then this mode.")
    d = dict(np.load(path))
    d.update(new_keys)
    np.savez(path, **d)
    return path


def _run_admittance_pinit_mode() -> None:
    """Admittance-perturbation MC → per-entry ``P_init_diag`` (model-error prior).

    For each sampled operating point, compute the nominal analytical H, then draw
    K perturbations of the network model — line series impedances (r,x scaled by a
    correlated per-line factor 1±MC_ADM_LINE_FRAC) and transformer short-circuit
    impedances (vk/vkr scaled per-element by 1±MC_ADM_TRAFO_FRAC) — recompute H and
    accumulate ΔH = vec(H_pert − H_nom)[DER cols].  ``P_init_diag = var(ΔH)`` per
    entry is the spread of each H entry under credible model error.  Merged into
    kalman_matrices.npz next to Q/R/u_scale.
    """
    print("\n" + "=" * 72)
    print(f"  ADMITTANCE-PERTURBATION MC -> P_init_diag "
          f"(line +-{MC_ADM_LINE_FRAC:.0%} r,x; trafo +-{MC_ADM_TRAFO_FRAC:.0%} vk,vkr)")
    print("=" * 72)
    S = _build_worker_state(quiet=False)
    net, dso_ctrl = S["net"], S["dso_ctrl"]
    profiles_df, n_der, n_y = S["profiles_df"], S["n_der"], S["n_y"]
    n_state = n_y * n_der

    rng = np.random.default_rng(SEED)
    timestamps = _sample_timestamps(rng, MC_PERTURB_N_OP)

    # Snapshot the nominal admittance parameters we perturb (positional ndarrays).
    snap_line_r = net.line["r_ohm_per_km"].to_numpy(copy=True)
    snap_line_x = net.line["x_ohm_per_km"].to_numpy(copy=True)
    t3w_cols = [c for c in ("vk_hv_percent", "vkr_hv_percent", "vk_mv_percent",
                            "vkr_mv_percent", "vk_lv_percent", "vkr_lv_percent")
                if c in net.trafo3w.columns]
    snap_t3w = {c: net.trafo3w[c].to_numpy(copy=True) for c in t3w_cols}
    t2w_cols = [c for c in ("vk_percent", "vkr_percent") if c in net.trafo.columns]
    snap_t2w = {c: net.trafo[c].to_numpy(copy=True) for c in t2w_cols}
    n_line, n_t3w, n_t2w = len(net.line), len(net.trafo3w), len(net.trafo)

    def _restore_admittance() -> None:
        net.line["r_ohm_per_km"] = snap_line_r
        net.line["x_ohm_per_km"] = snap_line_x
        for c in t3w_cols:
            net.trafo3w[c] = snap_t3w[c]
        for c in t2w_cols:
            net.trafo[c] = snap_t2w[c]

    def _factor(frac: float, size: int) -> np.ndarray:
        return 1.0 + rng.uniform(-frac, frac, size=size)

    dH: list = []
    skipped = 0
    for ti, t_mc in enumerate(timestamps):
        apply_profiles(net, profiles_df, t_mc)
        _seed_qv(net)
        try:
            pp.runpp(net, **_RUNPP_KW)
            H_nom = dso_ctrl.compute_h_analytical(net)
        except Exception:
            H_nom = None
        if H_nom is None:
            skipped += 1
            continue
        H_nom = H_nom[:, :n_der].flatten()
        for _ in range(MC_PERTURB_K):
            f_line = _factor(MC_ADM_LINE_FRAC, n_line)         # correlated r,x per line
            net.line["r_ohm_per_km"] = snap_line_r * f_line
            net.line["x_ohm_per_km"] = snap_line_x * (
                f_line if MC_ADM_RX_CORR else _factor(MC_ADM_LINE_FRAC, n_line))
            f_t3w = _factor(MC_ADM_TRAFO_FRAC, n_t3w)          # one factor / trafo → keep vk/vkr ratio
            for c in t3w_cols:
                net.trafo3w[c] = snap_t3w[c] * f_t3w
            if n_t2w:
                f_t2w = _factor(MC_ADM_TRAFO_FRAC, n_t2w)
                for c in t2w_cols:
                    net.trafo[c] = snap_t2w[c] * f_t2w
            try:
                _seed_qv(net)
                pp.runpp(net, **_RUNPP_KW)
                H_pert = dso_ctrl.compute_h_analytical(net)
            except Exception:
                H_pert = None
            _restore_admittance()
            if H_pert is None:
                skipped += 1
                continue
            dH.append(H_pert[:, :n_der].flatten() - H_nom)
        print(f"  op {ti + 1}/{len(timestamps)}: {len(dH)} dH ({skipped} skipped)", flush=True)

    if len(dH) < 2:
        raise RuntimeError("Too few admittance-perturbation samples for P_init_diag "
                           "— increase MC_PERTURB_N_OP/K or check convergence.")
    P_init_diag = np.var(np.stack(dH), axis=0)
    P_init_diag += 1e-18                                       # strictly positive
    assert P_init_diag.shape[0] == n_state, (P_init_diag.shape, n_state)
    path = _merge_into_kalman_npz(P_init_diag=P_init_diag)
    print(f"\n  P_init_diag ({P_init_diag.shape[0]}) from {len(dH)} perturbations: "
          f"diag range [{P_init_diag.min():.2e}, {P_init_diag.max():.2e}]")
    print(f"  -> {path}")


def _run_outage_preset_mode() -> None:
    """Outage-topology MC → per-entry ``P_reset_diag`` (topology-jump spread).

    Enumerate credible single contingencies — every in-service line plus the
    interface 3W coupler transformers (q_idx) when MC_OUTAGE_INCLUDE_TRAFO3W — and,
    over sampled operating points, accumulate ΔH = vec(H_post_trip − H_nom)[DER cols].
    Non-convergent trips are pruned.  ``P_reset_diag = var(ΔH)`` (or max(ΔH²) when
    MC_OUTAGE_STAT="max") sizes the per-entry covariance the runtime re-opens by on
    a detected topology jump.  Merged into kalman_matrices.npz.
    """
    print("\n" + "=" * 72)
    print("  OUTAGE-TOPOLOGY MC -> P_reset_diag (lines + interface 3W couplers)")
    print("=" * 72)
    S = _build_worker_state(quiet=False)
    net, dso_ctrl = S["net"], S["dso_ctrl"]
    profiles_df, n_der, n_y, q_idx = S["profiles_df"], S["n_der"], S["n_y"], S["q_idx"]
    n_state = n_y * n_der

    rng = np.random.default_rng(SEED + 1)
    timestamps = _sample_timestamps(rng, MC_OUTAGE_N_OP)

    # Candidate single contingencies: in-service lines + interface 3W couplers.
    line_cands = [("line", int(i)) for i in net.line.index
                  if bool(net.line.at[i, "in_service"])]
    trafo_cands = ([("trafo3w", int(i)) for i in q_idx]
                   if MC_OUTAGE_INCLUDE_TRAFO3W else [])
    candidates = line_cands + trafo_cands
    print(f"  candidates: {len(line_cands)} lines + {len(trafo_cands)} couplers "
          f"= {len(candidates)}")

    def _set_in_service(etype: str, idx: int, val: bool) -> None:
        getattr(net, etype).at[idx, "in_service"] = val

    def _converge_nominal() -> None:
        _seed_qv(net)
        pp.runpp(net, **_RUNPP_KW)

    dH: list = []
    skipped = 0
    for ti, t_mc in enumerate(timestamps):
        apply_profiles(net, profiles_df, t_mc)
        try:
            _converge_nominal()
            H_nom = dso_ctrl.compute_h_analytical(net)
        except Exception:
            H_nom = None
        if H_nom is None:
            skipped += 1
            continue
        H_nom = H_nom[:, :n_der].flatten()
        for etype, idx in candidates:
            _set_in_service(etype, idx, False)
            try:
                _seed_qv(net)
                pp.runpp(net, **_RUNPP_KW)
                H_post = dso_ctrl.compute_h_analytical(net)
            except Exception:
                H_post = None
            _set_in_service(etype, idx, True)            # always restore
            try:
                _converge_nominal()                      # re-settle before next trip
            except Exception:
                pass
            if H_post is None:
                skipped += 1
                continue
            dH.append(H_post[:, :n_der].flatten() - H_nom)
        print(f"  op {ti + 1}/{len(timestamps)}: {len(dH)} dH ({skipped} skipped)", flush=True)

    if len(dH) < 2:
        raise RuntimeError("Too few outage samples for P_reset_diag "
                           "— increase MC_OUTAGE_N_OP or check convergence.")
    DH = np.stack(dH)
    P_reset_diag = np.max(DH ** 2, axis=0) if MC_OUTAGE_STAT == "max" else np.var(DH, axis=0)
    P_reset_diag += 1e-18
    assert P_reset_diag.shape[0] == n_state, (P_reset_diag.shape, n_state)
    path = _merge_into_kalman_npz(P_reset_diag=P_reset_diag)
    print(f"\n  P_reset_diag ({P_reset_diag.shape[0]}, stat={MC_OUTAGE_STAT}) from "
          f"{len(dH)} trips: diag range [{P_reset_diag.min():.2e}, {P_reset_diag.max():.2e}]")
    print(f"  -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    # P_init / P_reset producer modes (gated by env): each merges its per-entry
    # diagonal into an existing kalman_matrices.npz and returns.
    if os.environ.get("MC_ADMITTANCE_PINIT", "").strip():
        _run_admittance_pinit_mode()
        return
    if os.environ.get("MC_OUTAGE_PRESET", "").strip():
        _run_outage_preset_mode()
        return

    from joblib import Parallel, delayed, effective_n_jobs

    # ── Step 1: initialise once in the parent (parameters + informative print) ──
    print("\n" + "=" * 72)
    print("  STEP 1: initialising network (1-step run)")
    print("=" * 72)
    S = _build_worker_state(quiet=False)
    n_der, n_oltc, n_y, n_u = S["n_der"], S["n_oltc"], S["n_y"], S["n_u"]
    n_state = n_y * n_u
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
    residuals_list: list = []
    du_within_list: list = []
    skipped = 0
    for recs, dh, res, du, sk in chunk_out:
        records.extend(recs)
        delta_h_within.extend(dh)
        residuals_list.extend(res)
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
    print(f"\n  Collected {n_samples} samples  ({skipped} skipped: non-convergence)")
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
        RES = np.stack(residuals_list) if residuals_list else np.zeros((0, n_y))
        _pdir = os.path.dirname(os.path.abspath(_PARTIAL_OUT)) or "."
        os.makedirs(_pdir, exist_ok=True)
        np.savez(_PARTIAL_OUT, _dh_within=DH, _residuals=RES, _du_within=DU, **arrays)
        print(f"\n  [shard] {n_samples} samples, {DH.shape[0]} within-walk dH pairs, "
              f"{RES.shape[0]} residuals, {DU.shape[0]} du pairs -> {_PARTIAL_OUT}")
        print("  [shard] Q/R + ANN deferred to _merge_train_mc.py")
        return

    out_path = os.path.abspath(exp003._TRAINING_DATA_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, n_q_trafo=S["n_q_tr"], n_v=S["n_v"], n_i=S["n_i"], **arrays)
    print(f"  {n_samples} samples -> {out_path}")

    # ── Step 5: Kalman Q/R from MC-specific within-walk statistics ───────────────
    print("\n" + "=" * 72)
    print("  STEP 4: estimating Kalman Q/R from MC within-walk statistics")
    print("=" * 72)

    if len(delta_h_within) >= 2:
        DH  = np.stack(delta_h_within)           # (M, n_state)
        Q   = np.diag(np.var(DH, axis=0))        # diagonal, matching generate_kalman_matrices
        print(f"  Q estimated from {len(delta_h_within)} within-walk dH pairs")
    else:
        Q = np.eye(n_state) * 1e-6
        print("  [warn] too few within-walk pairs for Q — using I*1e-6")

    if len(residuals_list) >= 2:
        RES = np.stack(residuals_list)            # (M, n_y)
        R   = np.cov(RES.T) if n_y > 1 else np.atleast_2d(np.var(RES))
        print(f"  R estimated from {len(residuals_list)} within-walk residuals")
    else:
        R = np.eye(n_y) * 1e-4
        print("  [warn] too few excited steps for R — using I*1e-4")

    # Ensure PD
    Q += 1e-12 * np.eye(n_state)
    R += 1e-12 * np.eye(n_y)

    # Sensor-noise floor on R: the MC power flow is noise-free, so the raw residual
    # R is rank-deficient at the small deployment step (cond ~1e10) → R⁻¹ explodes
    # the runtime NIS and destabilises the gain.  Floor + (optionally) diagonalise.
    n_q_tr, n_v, n_i = S["n_q_tr"], S["n_v"], S["n_i"]
    R = exp003.floor_measurement_noise_R(R, n_q_tr, n_v, n_i,
                                         sigma_q=MC_R_SIGMA_Q, sigma_v=MC_R_SIGMA_V,
                                         sigma_i=MC_R_SIGMA_I, diagonal=MC_R_DIAGONAL)
    print(f"  R sensor-floor: split {n_q_tr}/{n_v}/{n_i}, diagonal={MC_R_DIAGONAL}, "
          f"sigma_Q/V/I={MC_R_SIGMA_Q}/{MC_R_SIGMA_V}/{MC_R_SIGMA_I} -> "
          f"cond={np.linalg.cond(R):.1e}")

    kalman_path = os.path.abspath(exp003._KALMAN_MATRICES_PATH)
    os.makedirs(os.path.dirname(kalman_path), exist_ok=True)
    save_kw = dict(Q=Q, R=R, u_scale=u_scale, n_q_trafo=n_q_tr, n_v=n_v, n_i=n_i)
    # Preserve per-entry P_init_diag/P_reset_diag from a prior admittance/outage
    # MC run (matching n_state) so re-running Q/R does not wipe them.
    if os.path.exists(kalman_path):
        _prev = np.load(kalman_path)
        for _k in ("P_init_diag", "P_reset_diag"):
            if _k in _prev and np.asarray(_prev[_k]).ravel().shape[0] == n_state:
                save_kw[_k] = _prev[_k]
    np.savez(kalman_path, **save_kw)
    q_d, r_d = np.diag(Q), np.diag(R)
    print(f"  Q {Q.shape}  diag: [{q_d.min():.2e}, {q_d.max():.2e}]")
    print(f"  R {R.shape}  diag: [{r_d.min():.2e}, {r_d.max():.2e}]")
    print(f"  u_scale ({u_scale.shape[0]}) from {DU.shape[0]} within-walk du: "
          f"[{u_scale.min():.3f}, {u_scale.max():.3f}]")
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
