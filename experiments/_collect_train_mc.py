#!/usr/bin/env python3
"""
Monte Carlo training data collection for Kalman Q/R and ANN.

Following the approach from the old code (covariances_new):
  - N_OP random timestamps drawn from the full year of profiles give
    diverse load/wind operating points.
  - At each timestamp, a K_PERTURB-step random walk explores the
    Q_cor / OLTC actuator space (continuous random-direction step +
    discrete ±1 OLTC perturbation), matching the old code pattern.
  - At every (timestamp, walk-step) pair, pp.runpp is run with
    run_control=True so the QV local loops converge; the REALIZED closed-loop
    secant H is computed via compute_numerical_h_dso(closed_loop=True,
    delta_q_mvar=TARGET_STEP_MVAR) — the true plant response over an OFO-sized
    step, which is what the predictors deploy against (not the δ→0 analytical
    Jacobian).  It is stored under the legacy key "H_analytical" so the KF/ANN
    consumers (generate_kalman_matrices, train_ann_model) need no change.
  - Kalman Q is estimated from cov(ΔH) within each walk (not across
    operating-point boundaries, which would inflate Q artificially).
  - Kalman R is estimated from cov(Δy − H·Δu) residuals (within-walk
    pairs with ‖Δu‖ > threshold).
  - ANN training data: (y, u, H_secant) from all samples.

Key differences vs the time-series collectors:
  - Timestamps span the full year instead of a single 4h window.
  - OLTC taps are explicitly perturbed discretely (±1 steps) rather
    than waiting for voltage-limit triggers.
  - Q is estimated only from within-walk consecutive pairs so the
    diagonal entries reflect genuine step-to-step H drift, not
    inter-episode jumps from profile changes.

Pipeline
--------
  1. 1-step simulation → initialised net + DSO_2 controller.
  2. MC loop: N_OP timestamps × K_PERTURB actuator walk steps.
  3. Save training_data.npz (same format as the time-series collectors).
  4. Call generate_kalman_matrices() and train_ann_model().
"""

import sys
import os
import importlib
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
sys.path.insert(0, _root)

import pandapower as pp
from core.profiles import DEFAULT_PROFILES_CSV, apply_profiles, load_profiles
from experiments.runners import run_multi_tso_dso
from experiments.helpers.plant_io import decouple_trafo3w_hv_with_slack
from sensitivity.numerical_h import compute_numerical_h_dso

exp003 = importlib.import_module("experiments.003_S_DSO_CIGRE_2026")

# ── Configuration ──────────────────────────────────────────────────────────────
# The MC walk step magnitude must match the actual per-step OFO excitation seen at
# deployment so that R captures the true linearisation residual at that step size.
#
# Timing (003.make_base_config): dt_s = 60 s master step, dso_period_s = 1 s, so the
# DSO/Kalman fires ONCE per 60 s record. The filter therefore sees the full 60 s OFO
# displacement plus the persistent PE excitation (~Mvar scale), NOT a 1 s increment.
#
# T (deployment per-step ||delta_u_DER||) measured on the DECOUPLED net from a 003 OFO
# reference run (results/003_cigre_2026/2026-06-03--09-15-47_dso2_ctrl.npz, 180 records):
#   transient (first ~4 records) decays 10.5 -> 6.6 -> 6.5 -> 4.1; steady-state floor is
#   mean 3.39 / median 3.31 Mvar (the PE-noise floor, robust to skipping 10..40 records).
# The MC walk draws step_norm ~ Uniform(0, max_norm), so mean MC step = 0.5 * max_norm.
# Matching the deployment mean => max_norm = 2 * T, i.e. Q_MAX_FRAC = 2*T / ||q_range||.
# Q_MAX_FRAC is derived from TARGET_STEP_MVAR below, once ||q_range|| is known.
DSO_PERIOD_S      = 1          # set to match dso_period_s in make_base_config
# N_OP / K_PERTURB / SEED accept env overrides so this script can run as one of
# several PARALLEL SHARDS (see MC_PARTIAL_OUT below).  Defaults reproduce the
# single-process run (N_OP=60, K_PERTURB=15, SEED=42).
N_OP              = int(os.environ.get("MC_N_OP", 60))        # op-point timestamps (per shard)
K_PERTURB         = int(os.environ.get("MC_K_PERTURB", 15))   # random-walk steps per op-point
TARGET_STEP_MVAR  = 1.0        # decoupled deployment per-step ||delta_u_DER||: measured 0.95 Mvar
                               # (changing, PE=0.25, 2026-06-17). Was 3.4 (stale June-3 calib, deep in
                               # the nonlinear regime -> secant ~0.5 off the Jacobian; see frozen PE sweep).
DU_MIN_NORM  = 0.005      # skip residual if norm(delta_u) < this (scaled for 1s steps)
SEED         = int(os.environ.get("MC_SEED", 42))            # distinct seeds -> distinct op-points per shard
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

rng = np.random.default_rng(SEED)

# ── Step 1: initialise network and DSO_2 controller ────────────────────────────
print("\n" + "=" * 72)
print("  STEP 1: initialising network (1-step run)")
print("=" * 72)


exp003.H_INIT_BIAS_STD = 0.0
exp003.FROZEN_OP_POINT = False

cfg_init = exp003.make_config()   # was make_base_config() (renamed; old name no longer exists)
cfg_init.n_total_s            = cfg_init.dso_period_s   # exactly one step
cfg_init.live_plot_controller = False
cfg_init.live_plot_cascade    = False
cfg_init.live_plot_system     = False

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
            print(f"[iface] DSO_2 decoupled from TSO: replaced HV feed of "
                  f"3W trafos {ifaces} with pinned slacks (ext_grid {created}).")

run_multi_tso_dso(cfg_init, pre_loop_hook=_init_hook)

net      = _ref.get("net")
dso_ctrl = _ref.get("dso_ctrl")
if net is None or dso_ctrl is None:
    raise RuntimeError("Failed to initialise network or DSO_2 controller.")

# ── Step 2: extract DSO_2 parameters ───────────────────────────────────────────
dcfg    = dso_ctrl.config
der_idx  = np.array(dcfg.der_indices,              dtype=int)
oltc_idx = np.array(dcfg.oltc_trafo_indices,       dtype=int)
v_idx    = np.array(dcfg.voltage_bus_indices,       dtype=int)
q_idx    = np.array(dcfg.interface_trafo_indices,   dtype=int)
i_idx    = np.array(dcfg.current_line_indices,      dtype=int)

n_der   = len(der_idx)
n_oltc  = len(oltc_idx)
n_q_tr  = len(q_idx)
n_v     = len(v_idx)
n_i     = len(i_idx)
n_y     = n_q_tr + n_v + n_i
n_u     = n_der + n_oltc

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
print(f"  ||q_range|| = {q_range_norm:.3f} Mvar  ->  "
      f"Q_MAX_FRAC = 2*{TARGET_STEP_MVAR:.2f}/||q_range|| = {Q_MAX_FRAC:.4f}")

# OLTC bounds (3W coupling trafos)
tap_min     = net.trafo3w.loc[oltc_idx, "tap_min"].values.astype(int)
tap_max     = net.trafo3w.loc[oltc_idx, "tap_max"].values.astype(int)
tap_neutral = net.trafo3w.loc[oltc_idx, "tap_neutral"].values.astype(int)

print(f"  n_der={n_der}  n_oltc={n_oltc}  n_y={n_y}  n_u={n_u}")
print(f"  Q_cor range: [{q_cor_min.min():.2f}, {q_cor_max.max():.2f}] Mvar")
print(f"  OLTC tap range: {tap_min} .. {tap_max}  neutral={tap_neutral}")

# ── Step 3: load profiles ──────────────────────────────────────────────────────
print("\n  Loading profiles...")
profiles_df = load_profiles(
    cfg_init.profiles_csv if cfg_init.profiles_csv else DEFAULT_PROFILES_CSV
)
# snapshot_base_values was called by the runner initialisation — do not repeat.

# ── Step 4: sample random timestamps ──────────────────────────────────────────
total_hours   = int((SAMPLE_END - SAMPLE_START).total_seconds() / 3600)
candidate_h   = rng.integers(0, total_hours, size=N_OP * 4)
timestamps: list[datetime] = []
for h in candidate_h:
    t = SAMPLE_START + timedelta(hours=int(h))
    if not (EXCL_START <= t <= EXCL_END):
        timestamps.append(t)
    if len(timestamps) == N_OP:
        break
if len(timestamps) < N_OP:
    raise RuntimeError(f"Could not sample {N_OP} timestamps after exclusion window.")

# ── Step 5: MC collection loop ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  STEP 2: MC data collection")
print(f"  {N_OP} op-points × {K_PERTURB} walk steps = {N_OP * K_PERTURB} target samples")
print("=" * 72)

records: list[dict] = []       # all successfully converged samples
delta_h_within: list[np.ndarray] = []  # ΔH within each walk only (for Q)
residuals_list:  list[np.ndarray] = []  # Δy − H·Δu within each walk (for R)
skipped = 0


def _extract_y(net_: "pp.pandapowerNet") -> np.ndarray:
    """Extract y = [Q_trafo | V_bus | I_line] from pandapower results."""
    q_tr  = np.array([net_.res_trafo3w.at[t, "q_hv_mvar"] for t in q_idx],
                     dtype=float)
    v_bus = net_.res_bus.loc[v_idx, "vm_pu"].values.astype(float)
    i_ka  = np.array([net_.res_line.at[li, "i_from_ka"] for li in i_idx],
                     dtype=float)
    return np.concatenate([q_tr, v_bus, i_ka])


def _build_u(q_cor: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Build u = [Q_cor | tap_delta_from_neutral]."""
    return np.concatenate([q_cor, (taps - tap_neutral).astype(float)])


def _seed_qv(net_: "pp.pandapowerNet") -> None:
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


_RUNPP_KW = dict(
    run_control=True,
    calculate_voltage_angles=True,
    max_iteration=50,   # power-flow Newton iterations
    max_iter=300,       # controller convergence iterations
    numba=False,
)

for t_mc in tqdm(timestamps, desc="Operating points"):
    # ── Apply profiles and converge to operating-point equilibrium ───────────
    apply_profiles(net, profiles_df, t_mc)
    _seed_qv(net)
    try:
        pp.runpp(net, **_RUNPP_KW)
    except Exception:
        skipped += 1
        continue

    # Initial q_set = OFO setpoint at this op-point (read from sgen table)
    q_set_now = net.sgen.loc[der_idx, "q_set_mvar"].values.astype(float).copy()
    tap_now   = net.trafo3w.loc[oltc_idx, "tap_pos"].values.astype(int).copy()

    prev_rec: dict | None = None

    for k in range(K_PERTURB):
        # ── Random walk: continuous q_set step ────────────────────────────
        direction  = rng.standard_normal(n_der)
        direction /= np.linalg.norm(direction) + 1e-12
        max_norm   = Q_MAX_FRAC * np.linalg.norm(q_range)
        step_norm  = rng.uniform(0.0, max_norm)
        q_set_new  = np.clip(q_set_now + direction * step_norm,
                             q_cor_min, q_cor_max)

        # ── Random walk: discrete OLTC perturbation ───────────────────────
        tap_delta = rng.choice([-1, 0, 0, 0, 1], size=n_oltc)  # 20/60/20 %
        tap_new   = np.clip(tap_now + tap_delta, tap_min, tap_max)

        # ── Apply to network ──────────────────────────────────────────────
        net.sgen.loc[der_idx, "q_set_mvar"]  = q_set_new
        net.trafo3w.loc[oltc_idx, "tap_pos"] = tap_new
        _seed_qv(net)

        try:
            pp.runpp(net, **_RUNPP_KW)
        except Exception:
            skipped += 1
            continue

        # ── Numerical secant H (the deployment target) ────────────────────
        # The OFO deploys against the REALIZED closed-loop secant at the OFO
        # step size, not the instantaneous analytical Jacobian.  Train the ANN
        # (and estimate KF Q/R) on that same secant so the target matches what
        # the plant actually does over a TARGET_STEP_MVAR move.  Stored under
        # the legacy key "H_analytical" so the KF/ANN consumers are unchanged.
        H = compute_numerical_h_dso(
            net, dso_ctrl, closed_loop=True, delta_q_mvar=TARGET_STEP_MVAR
        )
        if H is None:
            skipped += 1
            continue

        # ── Measurements ──────────────────────────────────────────────────
        y = _extract_y(net)
        u = _build_u(q_set_new, tap_new)

        rec = {"y": y, "u": u, "H_analytical": H}
        records.append(rec)

        # ── Within-walk consecutive pairs for Q and R ─────────────────────
        if prev_rec is not None:
            dh = H.flatten() - prev_rec["H_analytical"].flatten()
            delta_h_within.append(dh)

            du = u - prev_rec["u"]
            if np.linalg.norm(du) >= DU_MIN_NORM:
                dy  = y - prev_rec["y"]
                C   = np.kron(np.eye(n_y), du.reshape(1, -1))   # (n_y, n_state)
                res = dy - C @ prev_rec["H_analytical"].flatten()
                residuals_list.append(res)

        prev_rec  = rec
        q_set_now = q_set_new
        tap_now   = tap_new

n_samples = len(records)
print(f"\n  Collected {n_samples} samples  ({skipped} skipped: non-convergence)")
if n_samples < 10:
    raise RuntimeError("Too few samples — increase N_OP or check power flow convergence.")

# ── Step 6: save training data ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  STEP 3: stacking samples")
print("=" * 72)

arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
for k, v in arrays.items():
    print(f"    {k}: {v.shape}")

# ── Shard mode: save partial (records + within-walk pairs) and stop ────────────
if _PARTIAL_OUT:
    DH  = np.stack(delta_h_within) if delta_h_within else np.zeros((0, n_y * n_u))
    RES = np.stack(residuals_list) if residuals_list else np.zeros((0, n_y))
    _pdir = os.path.dirname(os.path.abspath(_PARTIAL_OUT)) or "."
    os.makedirs(_pdir, exist_ok=True)
    np.savez(_PARTIAL_OUT, _dh_within=DH, _residuals=RES, **arrays)
    print(f"\n  [shard] {n_samples} samples, {DH.shape[0]} within-walk dH pairs, "
          f"{RES.shape[0]} residuals -> {_PARTIAL_OUT}")
    print("  [shard] Q/R + ANN deferred to _merge_train_mc.py")
    sys.exit(0)

out_path = os.path.abspath(exp003._TRAINING_DATA_PATH)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(out_path, **arrays)
print(f"  {n_samples} samples -> {out_path}")

# ── Step 7: Kalman Q/R from MC-specific within-walk statistics ─────────────────
print("\n" + "=" * 72)
print("  STEP 4: estimating Kalman Q/R from MC within-walk statistics")
print("=" * 72)

n_state = n_y * n_u

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

kalman_path = os.path.abspath(exp003._KALMAN_MATRICES_PATH)
os.makedirs(os.path.dirname(kalman_path), exist_ok=True)
np.savez(kalman_path, Q=Q, R=R)
q_d, r_d = np.diag(Q), np.diag(R)
print(f"  Q {Q.shape}  diag: [{q_d.min():.2e}, {q_d.max():.2e}]")
print(f"  R {R.shape}  diag: [{r_d.min():.2e}, {r_d.max():.2e}]")
print(f"  -> {kalman_path}")

# ── Step 8: retrain ANN ────────────────────────────────────────────────────────
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
