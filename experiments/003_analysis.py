#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/003_analysis.py
===========================
Post-hoc analysis for 003_S_DSO_CIGRE_2026 runs.

Loads the ``.pkl`` log and the ``_dso2_ctrl.npz`` sidecar produced by
:func:`run` and compares:

  * **H prediction error** : ||H_pred(k) - H_used(k+1)||_F / ||H_used(k+1)||_F
  * **Q_trafo tracking**   : setpoint vs actual at the three coupling trafos
  * **Voltage deviations** : V_min / V_mean / V_max vs V_ref for DSO_2

Usage
-----
    # Interactive run selector (lists all pkl files, pick by number):
    python experiments/003_analysis.py

    # Explicit path:
    python experiments/003_analysis.py path/to/run.pkl --sidecar path/to/run_dso2_ctrl.npz

    # Save figure instead of showing:
    python experiments/003_analysis.py --save figures/003_analysis.png
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure the project root is on sys.path so pickle can deserialise
# experiments.helpers.records classes regardless of invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_run(
    pkl_path: str,
    sidecar_path: Optional[str] = None,
) -> Tuple[list, Optional[np.lib.npyio.NpzFile]]:
    """Load pkl log and optional ``_dso2_ctrl.npz`` sidecar.

    If ``sidecar_path`` is ``None``, looks for a ``.npz`` with the same
    timestamp stem alongside the pkl (e.g. ``2026-05-12--10-00-00_dso2_ctrl.npz``).
    """
    with open(pkl_path, "rb") as f:
        log = pickle.load(f)

    if sidecar_path is None:
        sidecar_path = _sidecar_for_pkl(pkl_path) or None

    sidecar = np.load(sidecar_path) if sidecar_path and os.path.exists(sidecar_path) else None
    if sidecar_path and sidecar is None:
        print(f"[analysis] sidecar not found: {sidecar_path}")
    return log, sidecar


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _sidecar_for_pkl(pkl_path: str) -> Optional[str]:
    """Return the sidecar path for *pkl_path*, or ``None`` if it doesn't exist.

    PKL files are named ``<timestamp>_<predictor_mode>.pkl``; sidecars are
    named ``<timestamp>_dso2_ctrl.npz``.  The timestamp is the first
    underscore-delimited segment of the basename.
    """
    basename = os.path.basename(pkl_path)
    timestamp = basename.split("_")[0]
    candidate = os.path.join(os.path.dirname(pkl_path), timestamp + "_dso2_ctrl.npz")
    return candidate if os.path.exists(candidate) else None

# Candidate result directories searched in order (first non-empty wins).
_RESULT_DIR_CANDIDATES = [
    os.path.join("results", "003_cigre_2026"),                        # CWD-relative (project root)
    os.path.join(_SCRIPT_DIR, "results", "003_cigre_2026"),           # script-dir-relative (legacy)
]


def find_all_runs(result_dir: Optional[str] = None) -> List[str]:
    """Return all timestamp pkl paths sorted by modification time (oldest first)."""
    if result_dir is None:
        seen: set = set()
        all_pkls: list = []
        for candidate in _RESULT_DIR_CANDIDATES:
            for p in glob.glob(os.path.join(candidate, "[0-9]*.pkl")):
                abs_p = os.path.abspath(p)
                if abs_p not in seen:
                    seen.add(abs_p)
                    all_pkls.append(abs_p)
        if not all_pkls:
            dirs = ", ".join(_RESULT_DIR_CANDIDATES)
            raise FileNotFoundError(f"No timestamp pkl files found in: {dirs}")
        all_pkls.sort(key=os.path.getmtime)
        return all_pkls
    pkls = sorted(
        glob.glob(os.path.join(result_dir, "[0-9]*.pkl")),
        key=os.path.getmtime,
    )
    if not pkls:
        raise FileNotFoundError(f"No pkl files found in {result_dir}")
    return pkls


def find_latest_run(result_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Return ``(pkl_path, sidecar_path)`` for the most recent run."""
    pkls = find_all_runs(result_dir)
    pkl = pkls[-1]
    return pkl, _sidecar_for_pkl(pkl)


def pick_run_interactive(result_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Print a numbered list of available runs and prompt for a selection.

    Returns ``(pkl_path, sidecar_path)``.  Pressing Enter (empty input)
    selects the most recent run.  Falls back to auto-latest when stdin is
    not a TTY (e.g. piped / CI).
    """
    pkls = find_all_runs(result_dir)

    print("\nAvailable runs (oldest → newest):")
    for i, p in enumerate(pkls):
        sidecar_mark = " + sidecar" if _sidecar_for_pkl(p) else ""
        print(f"  [{i}] {os.path.basename(p)}{sidecar_mark}")

    default_idx = len(pkls) - 1
    try:
        raw = input(f"\nSelect run [0-{len(pkls)-1}], or Enter for latest [{default_idx}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raw = ""

    if raw == "":
        idx = default_idx
    else:
        try:
            idx = int(raw)
            if not (0 <= idx < len(pkls)):
                print(f"[analysis] index {idx} out of range — using latest.")
                idx = default_idx
        except ValueError:
            print(f"[analysis] invalid input {raw!r} — using latest.")
            idx = default_idx

    pkl = pkls[idx]
    sidecar = _sidecar_for_pkl(pkl)
    print(f"[analysis] selected: {os.path.basename(pkl)}"
          + (f"  sidecar: {os.path.basename(sidecar)}" if sidecar else "  (no sidecar)"))
    return pkl, sidecar


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def h_analytical_error(sidecar: np.lib.npyio.NpzFile) -> Optional[Dict[str, np.ndarray]]:
    """Frobenius-norm error of H_used(k) vs H_analytical(k).

    ``h_analytical`` is the H matrix freshly recomputed from the current
    operating point at each step (true Jacobian linearisation), while
    ``h_used`` is the cached / corrected matrix the controller actually uses.

    Returns ``None`` when the sidecar does not contain ``h_analytical``.

    Returns a dict with:

    ``"absolute"``            (N,) -- ||H_used - H_analytical||_F  (full matrix)
    ``"relative"``            (N,) -- above / ||H_analytical||_F
    ``"absolute_q_trafo"``    (N,) -- same, restricted to Q_trafo rows only
    ``"relative_q_trafo"``    (N,) -- above / ||H_analytical[:n_q,:]||_F
    ``"n_q_trafo"``           int  -- number of Q_trafo rows (0 if unknown)
    ``"steps"``               (N,) -- step indices 0 … N-1
    """
    if "h_analytical" not in sidecar:
        return None
    H_used = sidecar["h_used"]        # (N, n_y, n_u)
    H_anal = sidecar["h_analytical"]  # (N, n_y, n_u)
    N = len(H_used)

    diff  = H_used - H_anal
    abs_e = np.linalg.norm(diff.reshape(N, -1), axis=1)
    ref   = np.linalg.norm(H_anal.reshape(N, -1), axis=1)
    rel_e = abs_e / (ref + 1e-12)

    result: Dict[str, np.ndarray] = {
        "absolute": abs_e,
        "relative": rel_e,
        "steps":    np.arange(N),
    }

    n_q = int(sidecar["n_q_trafo"]) if "n_q_trafo" in sidecar else 0
    result["n_q_trafo"] = np.array(n_q)
    if n_q > 0:
        diff_q = H_used[:, :n_q, :] - H_anal[:, :n_q, :]
        abs_q  = np.linalg.norm(diff_q.reshape(N, -1), axis=1)
        ref_q  = np.linalg.norm(H_anal[:, :n_q, :].reshape(N, -1), axis=1)
        result["absolute_q_trafo"] = abs_q
        result["relative_q_trafo"] = abs_q / (ref_q + 1e-12)

    return result


def one_step_qtrafo_error(
    sidecar: np.lib.npyio.NpzFile,
    which_h: str = "used",
    du_gate: float = 1e-3,
) -> Optional[Dict[str, np.ndarray]]:
    """One-step-ahead q_trafo prediction error against the *realised* plant.

    For each step ``k`` the deployed H matrix is used to predict the change in
    the q_trafo measurements that the applied control change actually produced::

        Δu(k)    = u(k)   - u(k-1)                   (n_u,)
        Δy_q(k)  = y(k)[:n_q] - y(k-1)[:n_q]         (n_q,)   realised
        pred(k)  = H_q(k) · Δu(k)                    (n_q,)   predicted
        resid(k) = Δy_q(k) - pred(k)                 (n_q,)
        e(k)     = ||resid(k)||                       [Mvar]

    The index pairing mirrors :class:`_KalmanHPredictor` exactly: the filter
    forms ``delta_y(k)=y(k)-y(k-1)``, ``delta_u(k)=u(k)-u(k-1)`` and scores them
    with its *prior* ``h`` — which is precisely ``h_used(k)`` (the cache before
    the predictor fires at step k).  Hence ``H_q(k) = h_used[k][:n_q]`` is paired
    with the (k-1 → k) deltas, i.e. ``H[1:]`` against ``diff(Y)``/``diff(U)``.
    With ``which_h="used"`` the returned residual therefore reproduces the
    filter's actual innovation sequence (genuine out-of-sample prediction).

    Unlike :func:`h_analytical_error`, this references the realised plant
    response — the *secant* the controller actually rides — not the analytical
    derivative.  It is the operationally meaningful convergence / functionality
    metric and is directly comparable across predictor modes (identity / kalman
    / ann), each evaluated on its own closed-loop trajectory.

    Parameters
    ----------
    which_h
        ``"used"``      -- H_used(k): the matrix held *before* observing Δy(k).
                           Genuine out-of-sample (innovation) form — recommended.
        ``"predicted"`` -- H_predicted(k): the posterior after the step's update
                           (in-sample; optimistic for adaptive predictors).
    du_gate
        Keep only steps with ``||Δu(k)|| > du_gate``; settled steps carry no
        information and would dilute the metric (and the skill score).

    Returns
    -------
    ``None`` when the sidecar lacks ``h_used`` / ``y`` / ``u`` / ``n_q_trafo``
    (or has < 2 steps), else a dict::

        "steps"        (M,)  gated absolute step indices k
        "resid_norm"   (M,)  ||Δy_q - H_q·Δu||            [Mvar]
        "zero_norm"    (M,)  ||Δy_q||  (zero-predictor baseline) [Mvar]
        "du_norm"      (M,)  ||Δu||
        "rmse"         float RMSE of resid_norm over gated steps [Mvar]
        "rmse_zero"    float RMSE of zero_norm   over gated steps [Mvar]
        "skill"        float 1 - rmse/rmse_zero  (fraction of zero-pred error removed)
        "n_q_trafo"    int
        "n_gated"      int
    """
    if any(k not in sidecar for k in ("h_used", "y", "u")):
        return None
    key = "h_predicted" if which_h == "predicted" else "h_used"
    if key not in sidecar:
        key = "h_used"
    H = sidecar[key]   # (N, n_y, n_u)
    Y = sidecar["y"]   # (N, n_y)
    U = sidecar["u"]   # (N, n_u)
    N = len(H)
    if N < 2:
        return None
    n_q = int(sidecar["n_q_trafo"]) if "n_q_trafo" in sidecar else 0
    if n_q <= 0:
        return None

    dU  = U[1:] - U[:-1]                # Δu_j = u[j+1]-u[j]                 (N-1, n_u)
    dYq = Y[1:, :n_q] - Y[:-1, :n_q]    # realised Δy_q_j                    (N-1, n_q)
    Hq  = H[1:, :n_q, :]               # H deployed for the j→j+1 transition (N-1, n_q, n_u)
    #                                    (= prior the KF scores in its innovation)
    pred  = np.einsum("kij,kj->ki", Hq, dU)   # H_q·Δu                       (N-1, n_q)
    resid = dYq - pred

    resid_norm = np.linalg.norm(resid, axis=1)
    zero_norm  = np.linalg.norm(dYq,   axis=1)
    du_norm    = np.linalg.norm(dU,    axis=1)

    mask = du_norm > du_gate
    rn = resid_norm[mask]
    zn = zero_norm[mask]
    rmse      = float(np.sqrt(np.mean(rn ** 2))) if rn.size else float("nan")
    rmse_zero = float(np.sqrt(np.mean(zn ** 2))) if zn.size else float("nan")

    return {
        "steps":      np.arange(1, N)[mask],   # absolute step k = j+1 (matches h_used[k])
        "resid_norm": rn,
        "zero_norm":  zn,
        "du_norm":    du_norm[mask],
        "rmse":       np.array(rmse),
        "rmse_zero":  np.array(rmse_zero),
        "skill":      np.array(1.0 - rmse / (rmse_zero + 1e-12)),
        "n_q_trafo":  np.array(n_q),
        "n_gated":    np.array(int(mask.sum())),
    }


def q_trafo_tracking(
    log: list,
    dso_id: str = "DSO_2",
) -> Dict[str, object]:
    """Per-step Q setpoint vs actual at the interface transformers.

    Returns a dict with:

    ``"setpoint"``   (N, n_trafo) -- Q setpoints [Mvar]
    ``"actual"``     (N, n_trafo) -- Q actuals   [Mvar]
    ``"error"``      (N, n_trafo) -- setpoint - actual  [Mvar]
    ``"trafo_keys"`` list[str]    -- trafo identifier strings
    ``"times_s"``    (N,)         -- simulation time [s]
    """
    setpoints: List[list] = []
    actuals:   List[list] = []
    times:     List[float] = []
    trafo_keys: Optional[List[str]] = None

    for rec in log:
        q_set_d: Dict[str, float] = rec.dso_trafo_q_set_mvar
        q_act_d: Dict[str, float] = rec.dso_trafo_q_actual_mvar

        if trafo_keys is None:
            # Pick keys that belong to dso_id, fall back to all keys.
            keys = [k for k in q_set_d if dso_id in k]
            trafo_keys = keys if keys else sorted(q_set_d.keys())

        setpoints.append([q_set_d.get(k, np.nan) for k in trafo_keys])
        actuals.append([q_act_d.get(k, np.nan) for k in trafo_keys])
        times.append(rec.time_s)

    Q_set = np.array(setpoints)
    Q_act = np.array(actuals)
    return {
        "setpoint":   Q_set,
        "actual":     Q_act,
        "error":      Q_set - Q_act,
        "trafo_keys": trafo_keys or [],
        "times_s":    np.array(times),
    }


def voltage_deviation(
    log: list,
    v_ref: float = 1.03,
    dso_id: str = "DSO_2",
) -> Dict[str, np.ndarray]:
    """Per-step V_min / V_mean / V_max and their deviation from v_ref.

    Returns a dict with:

    ``"v_min"``, ``"v_mean"``, ``"v_max"``     (N,) in p.u.
    ``"dev_min"``, ``"dev_mean"``, ``"dev_max"``(N,) in p.u.  (value - v_ref)
    ``"times_s"``  (N,)
    ``"v_ref"``    float
    """
    v_min:  List[float] = []
    v_mean: List[float] = []
    v_max:  List[float] = []
    times:  List[float] = []

    for rec in log:
        v_min.append(rec.dso_group_v_min_pu.get(dso_id, np.nan))
        v_mean.append(rec.dso_group_v_mean_pu.get(dso_id, np.nan))
        v_max.append(rec.dso_group_v_max_pu.get(dso_id, np.nan))
        times.append(rec.time_s)

    arr_min  = np.array(v_min)
    arr_mean = np.array(v_mean)
    arr_max  = np.array(v_max)
    return {
        "v_min":    arr_min,
        "v_mean":   arr_mean,
        "v_max":    arr_max,
        "dev_min":  arr_min  - v_ref,
        "dev_mean": arr_mean - v_ref,
        "dev_max":  arr_max  - v_ref,
        "times_s":  np.array(times),
        "v_ref":    v_ref,
    }


def print_summary(log: list, sidecar: Optional[np.lib.npyio.NpzFile] = None) -> None:
    """Print a compact numerical summary to stdout."""
    qt = q_trafo_tracking(log)
    vd = voltage_deviation(log)

    print("\n=== Q_trafo tracking [Mvar] ===")
    for i, key in enumerate(qt["trafo_keys"]):
        err = qt["error"][:, i]
        print(f"  {key}: mean_err={np.nanmean(err):+.3f}  "
              f"rms={np.sqrt(np.nanmean(err**2)):.3f}  "
              f"max|err|={np.nanmax(np.abs(err)):.3f}")

    print("\n=== Voltage deviations from V_ref=%.3f [mpu] ===" % vd["v_ref"])
    for key in ("dev_min", "dev_mean", "dev_max"):
        arr = vd[key] * 1e3
        print(f"  {key}: mean={np.nanmean(arr):+.2f}  "
              f"min={np.nanmin(arr):+.2f}  max={np.nanmax(arr):+.2f}")

    if sidecar is not None:
        ose = one_step_qtrafo_error(sidecar)
        if ose is not None and ose["steps"].size:
            n_q = int(ose["n_q_trafo"])
            print("\n=== One-step-ahead q_trafo prediction error  "
                  "||dy_q - H_q*du|| [Mvar] ===")
            print(f"  Q_trafo ({n_q} rows) over {int(ose['n_gated'])} active steps:")
            print(f"    predictor RMSE : {float(ose['rmse']):.4f} Mvar")
            print(f"    zero-pred RMSE : {float(ose['rmse_zero']):.4f} Mvar  "
                  f"(baseline ||Δy_q||)")
            print(f"    skill          : {100 * float(ose['skill']):+.1f} %  "
                  f"(fraction of zero-pred error removed)")
        else:
            print("\n[analysis] one-step q_trafo error unavailable "
                  "(sidecar lacks y / u / h_used).")

        herr = h_analytical_error(sidecar)
        if herr is not None:
            print("\n=== H analytical error  ||H_used - H_analytical|| / ||H_analytical|| "
                  "(secondary; references the derivative, not the realised secant) ===")
            print(f"  full H  rel [%]: mean={herr['relative'].mean()*100:.2f}  "
                  f"max={herr['relative'].max()*100:.2f}")
            print(f"  full H  abs:     mean={herr['absolute'].mean():.4e}  "
                  f"max={herr['absolute'].max():.4e}")
            if "relative_q_trafo" in herr:
                n_q = int(herr["n_q_trafo"])
                print(f"  Q_trafo ({n_q} rows) rel [%]: mean={herr['relative_q_trafo'].mean()*100:.2f}  "
                      f"max={herr['relative_q_trafo'].max()*100:.2f}")
                print(f"  Q_trafo ({n_q} rows) abs:     mean={herr['absolute_q_trafo'].mean():.4e}  "
                      f"max={herr['absolute_q_trafo'].max():.4e}")
        else:
            print("\n[analysis] no h_analytical in sidecar — re-run 003 to record it.")


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    pkl_path: str,
    sidecar_path: Optional[str] = None,
    v_ref: float = 1.03,
    save_path: Optional[str] = None,
) -> None:
    """Three-panel figure: H error | Q tracking | voltage deviations.

    The H-error panel is skipped if no sidecar is available.
    """
    log, sidecar = load_run(pkl_path, sidecar_path)

    if not log:
        print(f"[analysis] log is empty ({os.path.basename(pkl_path)}) — nothing to plot.")
        return

    print_summary(log, sidecar)

    has_sidecar = sidecar is not None
    n_panels = 4 if has_sidecar else 2   # one-step err | H err | Q tracking | V

    fig = plt.figure(figsize=(14, 4 * n_panels))
    gs  = GridSpec(n_panels, 1, figure=fig, hspace=0.50)
    panel = 0

    # ── Panel 1: one-step-ahead q_trafo prediction error ─────────────────────
    if has_sidecar:
        ax = fig.add_subplot(gs[panel]); panel += 1
        ose = one_step_qtrafo_error(sidecar)
        if ose is not None and ose["steps"].size:
            steps = ose["steps"]
            ax.plot(steps, ose["resid_norm"], color="tab:blue", lw=1.2, alpha=0.85,
                    label=r"$\|\Delta y_q - H_q\,\Delta u\|$  (predictor)")
            ax.plot(steps, ose["zero_norm"], color="tab:gray", lw=1.0, ls="--",
                    alpha=0.8, label=r"$\|\Delta y_q\|$  (zero-predictor baseline)")
            # running mean to expose the convergence trend through the noise
            if steps.size >= 10:
                w = max(5, steps.size // 20)
                rm = np.convolve(ose["resid_norm"], np.ones(w) / w, mode="valid")
                ax.plot(steps[w - 1:], rm, color="tab:red", lw=2.0,
                        label=f"predictor, running mean (w={w})")
            n_q = int(ose["n_q_trafo"])
            summary = (f"RMSE over {int(ose['n_gated'])} active steps\n"
                       f"predictor : {float(ose['rmse']):.3f} Mvar\n"
                       f"zero-pred : {float(ose['rmse_zero']):.3f} Mvar\n"
                       f"skill     : {100 * float(ose['skill']):.1f} %")
            ax.text(0.99, 0.97, summary, transform=ax.transAxes,
                    fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.set_xlabel("DSO step k")
            ax.set_ylabel(f"One-step q_trafo error [Mvar]  ({n_q} rows)")
            ax.set_title(
                r"One-step-ahead q$_\mathrm{trafo}$ prediction error  "
                r"$\|\Delta y_q(k) - H_q(k)\,\Delta u(k)\|$   (vs. realised plant)"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5,
                    "one-step q_trafo error unavailable\n(sidecar lacks y / u / h_used)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11,
                    color="gray")
            ax.set_title("One-step-ahead q_trafo prediction error (data not available)")

    # ── Panel 2: H-estimation error vs analytical ────────────────────────────
    if has_sidecar:
        ax = fig.add_subplot(gs[panel]); panel += 1
        he = h_analytical_error(sidecar)
        if he is not None:
            steps = he["steps"]
            if "relative_q_trafo" in he:
                ax.plot(steps, 100 * he["relative_q_trafo"], color="tab:purple", lw=1.6,
                        label=r"q$_\mathrm{trafo}$ rows  "
                              f"(final {100 * float(np.mean(he['relative_q_trafo'][-20:])):.1f}%)")
            ax.plot(steps, 100 * he["relative"], color="tab:gray", lw=1.0, ls="--",
                    alpha=0.7, label="full H")
            ax.set_xlabel("DSO step k")
            ax.set_ylabel("rel. H error [%]")
            ax.set_title(r"H-estimation error  "
                         r"$\|H_\mathrm{used}-H_\mathrm{analytical}\|_F/\|H_\mathrm{analytical}\|_F$"
                         "   (NB: analytical ref includes the $T'$ droop transform)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "H error unavailable (sidecar lacks h_analytical)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title("H-estimation error (data not available)")

    # ── Panel 3: Q_trafo setpoint vs actual ──────────────────────────────────
    ax = fig.add_subplot(gs[panel]); panel += 1
    qt = q_trafo_tracking(log)
    times_min = qt["times_s"] / 60.0
    n_trafo = qt["setpoint"].shape[1]
    cmap = plt.cm.tab10
    for i in range(n_trafo):
        col = cmap(i / max(n_trafo, 1))
        lbl = (qt["trafo_keys"][i] if qt["trafo_keys"] else f"trafo {i}").split("|")[-1]
        ax.plot(times_min, qt["setpoint"][:, i], "--", color=col, lw=1.2,
                label=f"{lbl} set")
        ax.plot(times_min, qt["actual"][:, i],   "-",  color=col, lw=1.8,
                label=f"{lbl} act")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Q [Mvar]")
    ax.set_title("Interface Q: setpoint (--) vs actual (—)")
    ax.legend(fontsize=7, ncol=n_trafo * 2)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Voltage deviations ──────────────────────────────────────────
    ax = fig.add_subplot(gs[panel]); panel += 1
    vd = voltage_deviation(log, v_ref=v_ref)
    times_min = vd["times_s"] / 60.0
    mpu = 1e3  # p.u. -> milli-p.u.
    ax.fill_between(
        times_min,
        vd["dev_min"]  * mpu,
        vd["dev_max"]  * mpu,
        alpha=0.20, color="tab:blue", label="V band [min, max]",
    )
    ax.plot(times_min, vd["dev_mean"] * mpu, color="tab:blue", lw=1.5, label="V_mean")
    ax.axhline( 0,  color="k",    lw=0.8)
    ax.axhline( 10, color="gray", lw=0.8, ls="--", label="+10 mpu")
    ax.axhline(-10, color="gray", lw=0.8, ls="--", label="-10 mpu")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("V - V_ref [mpu]")
    ax.set_title(f"DSO_2 voltage deviations  (V_ref = {v_ref:.3f} p.u.)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    title = f"003 analysis — {os.path.basename(pkl_path)}"
    if sidecar_path:
        title += f"\nsidecar: {os.path.basename(sidecar_path)}"
    fig.suptitle(title, fontsize=10)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[analysis] saved -> {os.path.abspath(save_path)}")
    else:
        plt.ioff()   # visualisation modules call plt.ion(); force blocking show
        plt.show(block=True)


# ---------------------------------------------------------------------------
#  Multi-run comparison plot
# ---------------------------------------------------------------------------

def plot_multi_comparison(
    runs: List[Tuple[str, Optional[str], str]],
    v_ref: float = 1.03,
    save_path: Optional[str] = None,
) -> None:
    """Overlay multiple runs on the same three-panel figure.

    Parameters
    ----------
    runs : list of (pkl_path, sidecar_path, label)
    v_ref : float
        Voltage reference in p.u.
    save_path : str, optional
        Save to file instead of showing.
    """
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(runs)))
    fig = plt.figure(figsize=(16, 22))
    gs  = GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.22)
    ax_he   = fig.add_subplot(gs[0, 0])   # H error vs analytical (q_trafo rows)
    ax_hef  = fig.add_subplot(gs[0, 1])   # NEW1: H error vs analytical (full matrix)
    ax_h    = fig.add_subplot(gs[1, 0])   # one-step-ahead prediction error
    ax_act  = fig.add_subplot(gs[1, 1])   # NEW6: predictor activity ||H_used-H_pred||
    ax_q    = fig.add_subplot(gs[2, 0])   # interface Q mean-abs error
    ax_qt   = fig.add_subplot(gs[2, 1])   # NEW2: per-trafo Q actual vs setpoint
    ax_v    = fig.add_subplot(gs[3, 0])   # voltage deviation
    ax_hent = fig.add_subplot(gs[3, 1])   # NEW5: single H-entry drift
    ax_u    = fig.add_subplot(gs[4, 0])   # NEW3: actuator excursion (q_set drift)
    ax_uT   = ax_u.twinx()                #        + OLTC tap excursion (right axis)
    ax_sk   = fig.add_subplot(gs[4, 1])   # NEW4: one-step prediction skill (bar)

    _ls = ["-", "--", ":", "-."]          # per-trafo line styles
    skills: List[Tuple[str, float, object]] = []   # (label, skill, colour) for the bar
    set_drawn = False                     # draw interface-Q setpoints once

    for (pkl_path, sidecar_path, label), col in zip(runs, colors):
        log, sidecar = load_run(pkl_path, sidecar_path)

        # ── Panel ax_he / ax_hef: H error vs analytical (q_trafo + full) ──
        if sidecar is not None:
            he = h_analytical_error(sidecar)
            if he is not None:
                if "relative_q_trafo" in he:
                    ax_he.plot(he["steps"], 100 * he["relative_q_trafo"], color=col, lw=1.8,
                               label=f"{label}  (final {100 * float(np.mean(he['relative_q_trafo'][-20:])):.1f}%)")
                # NEW1: full-matrix relative error (all rows, incl. voltage)
                ax_hef.plot(he["steps"], 100 * he["relative"], color=col, lw=1.8,
                            label=f"{label}  (final {100 * float(np.mean(he['relative'][-20:])):.1f}%)")

        # ── Panel ax_h: one-step-ahead q_trafo prediction error ──────────
        if sidecar is not None:
            ose = one_step_qtrafo_error(sidecar)
            if ose is not None and ose["steps"].size:
                steps = ose["steps"]
                rn = ose["resid_norm"]
                # running mean for readability; raw is too noisy when overlaid
                if steps.size >= 10:
                    w = max(5, steps.size // 20)
                    rm = np.convolve(rn, np.ones(w) / w, mode="valid")
                    ax_h.plot(steps[w - 1:], rm, color=col, lw=1.8,
                              label=f"{label}  (RMSE={float(ose['rmse']):.3f} / "
                                    f"zero={float(ose['rmse_zero']):.3f} Mvar)")
                else:
                    ax_h.plot(steps, rn, color=col, lw=1.5,
                              label=f"{label}  (RMSE={float(ose['rmse']):.3f} Mvar)")
                # NEW4: collect skill (1 - RMSE/RMSE_zero) for the bar panel
                skills.append((label, float(ose["skill"]), col))

        # ── Panel ax_act (NEW6): predictor activity ||H_used-H_pred||_F ──
        if sidecar is not None and "h_used" in sidecar and "h_predicted" in sidecar:
            Hu, Hp = sidecar["h_used"], sidecar["h_predicted"]
            n = len(Hu)
            activity = np.linalg.norm((Hp - Hu).reshape(n, -1), axis=1)
            ax_act.plot(np.arange(n), activity, color=col, lw=1.5, label=label)

        # ── Panel ax_q: interface Q mean-abs error ───────────────────────
        qt = q_trafo_tracking(log)
        err_arr = np.abs(qt["error"])
        if err_arr.ndim == 2 and err_arr.shape[0] > 0:
            times_min = qt["times_s"] / 60.0
            ax_q.plot(times_min, np.nanmean(err_arr, axis=1), color=col, lw=1.5, label=label)

        # ── Panel ax_qt (NEW2): per-trafo Q actual vs setpoint ───────────
        act_arr, set_arr = qt["actual"], qt["setpoint"]
        keys = qt["trafo_keys"]
        if isinstance(act_arr, np.ndarray) and act_arr.ndim == 2 and act_arr.shape[0] > 0:
            tmin = qt["times_s"] / 60.0
            for j in range(act_arr.shape[1]):
                tname = _trafo_shortname(keys[j]) if j < len(keys) else f"t{j}"
                ax_qt.plot(tmin, act_arr[:, j], color=col, lw=1.3,
                           ls=_ls[j % len(_ls)], label=f"{label} {tname}")
            if not set_drawn and isinstance(set_arr, np.ndarray) and set_arr.ndim == 2:
                for j in range(set_arr.shape[1]):
                    ax_qt.plot(tmin, set_arr[:, j], color="k", lw=0.9,
                               ls=_ls[j % len(_ls)], alpha=0.5,
                               label="setpoint" if j == 0 else None)
                set_drawn = True

        # ── Panel ax_v: voltage deviation (mean + min/max band) ──────────
        vd = voltage_deviation(log, v_ref=v_ref)
        if len(vd["times_s"]) > 0:
            times_min_v = vd["times_s"] / 60.0
            ax_v.plot(times_min_v, vd["dev_mean"] * 1e3, color=col, lw=1.5, label=label)
            ax_v.fill_between(times_min_v,
                              vd["dev_min"] * 1e3, vd["dev_max"] * 1e3,
                              color=col, alpha=0.08)

        # ── Panel ax_hent (NEW5): single H-entry drift dQ_int0/dq_set0 ───
        if sidecar is not None and "h_used" in sidecar:
            Hu = sidecar["h_used"]
            ax_hent.plot(np.arange(len(Hu)), Hu[:, 0, 0], color=col, lw=1.7,
                         label=f"{label} (used)")
            if "h_analytical" in sidecar:
                Ha = sidecar["h_analytical"]
                ax_hent.plot(np.arange(len(Ha)), Ha[:, 0, 0], color=col, lw=1.0,
                             ls=":", alpha=0.7)

        # ── Panel ax_u/ax_uT (NEW3): actuator excursion from initial ─────
        # u = [q_set (n_der) | tap (n_oltc)]; n_oltc == n_q_trafo (one OLTC per
        # interface trafo).  q_set drift (left, Mvar) exposes the null-space
        # wander; tap drift (right, integer steps) is usually ~0 on this island.
        if sidecar is not None and "u" in sidecar:
            U = sidecar["u"]
            n, n_u = U.shape
            n_q = int(sidecar["n_q_trafo"]) if "n_q_trafo" in sidecar else 0
            n_oltc = n_q if 0 < n_q < n_u else 0
            n_der = n_u - n_oltc
            q_drift = np.linalg.norm(U[:, :n_der] - U[0, :n_der], axis=1)
            ax_u.plot(np.arange(n), q_drift, color=col, lw=1.7, label=label)
            if n_oltc > 0:
                tap_drift = np.sum(np.abs(U[:, n_der:] - U[0, n_der:]), axis=1)
                ax_uT.plot(np.arange(n), tap_drift, color=col, lw=1.1, ls=":")

    # ── NEW4: one-step prediction skill bar (post-loop) ──────────────────
    if skills:
        xs = np.arange(len(skills))
        ax_sk.bar(xs, [s[1] for s in skills], color=[s[2] for s in skills])
        ax_sk.set_xticks(xs)
        ax_sk.set_xticklabels([s[0] for s in skills], fontsize=8)
        ax_sk.axhline(0, color="k", lw=0.8)
        for x, s in zip(xs, skills):
            ax_sk.text(x, s[1], f"{s[1]:.2f}", ha="center",
                       va="bottom" if s[1] >= 0 else "top", fontsize=8)

    # ── Cosmetics ────────────────────────────────────────────────────────
    ax_he.set_xlabel("DSO step k")
    ax_he.set_ylabel("rel. H error [%]  (q_trafo)")
    ax_he.set_title(r"H error (q$_\mathrm{trafo}$ rows)  "
                    r"$\|H_\mathrm{used}-H_\mathrm{anal}\|/\|H_\mathrm{anal}\|$")
    ax_he.legend(fontsize=8)
    ax_he.grid(True, alpha=0.3)

    ax_hef.set_xlabel("DSO step k")
    ax_hef.set_ylabel("rel. H error [%]  (full)")
    ax_hef.set_title(r"H error (full matrix, all rows)")
    ax_hef.legend(fontsize=8)
    ax_hef.grid(True, alpha=0.3)

    ax_h.set_xlabel("DSO step k")
    ax_h.set_ylabel("One-step q_trafo error [Mvar]")
    ax_h.set_title(r"One-step-ahead q$_\mathrm{trafo}$ prediction error  "
                   r"$\|\Delta y_q(k) - H_q(k)\,\Delta u(k)\|$  (running mean)")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    ax_act.set_xlabel("DSO step k")
    ax_act.set_ylabel(r"$\|H_\mathrm{used}-H_\mathrm{pred}\|_F$")
    ax_act.set_title("Predictor activity (how much each step moves H)")
    ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    ax_q.set_xlabel("Time [min]")
    ax_q.set_ylabel("Mean |Q_err| [Mvar]")
    ax_q.set_title("Interface Q tracking error (mean over trafos)")
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.3)

    ax_qt.set_xlabel("Time [min]")
    ax_qt.set_ylabel("Q at interface trafo [Mvar]")
    ax_qt.set_title("Per-trafo interface Q: actual (colour=mode, style=trafo) vs setpoint (black)")
    ax_qt.legend(fontsize=6, ncol=3)
    ax_qt.grid(True, alpha=0.3)

    ax_v.axhline(0,   color="k",    lw=0.8)
    ax_v.axhline( 10, color="gray", lw=0.8, ls="--", label="+10 mpu")
    ax_v.axhline(-10, color="gray", lw=0.8, ls="--", label="-10 mpu")
    ax_v.set_xlabel("Time [min]")
    ax_v.set_ylabel("V_mean - V_ref [mpu]")
    ax_v.set_title(f"DSO_2 voltage deviation (V_ref = {v_ref:.3f} p.u.)")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)

    ax_hent.set_xlabel("DSO step k")
    ax_hent.set_ylabel(r"$\partial Q_{\mathrm{int},0}/\partial q_{\mathrm{set},0}$")
    ax_hent.set_title(r"Single H entry drift: used (solid) vs analytical (dotted)")
    ax_hent.legend(fontsize=8)
    ax_hent.grid(True, alpha=0.3)

    ax_u.set_xlabel("DSO step k")
    ax_u.set_ylabel(r"$\|q_\mathrm{set}(k)-q_\mathrm{set}(0)\|$ [Mvar]")
    ax_uT.set_ylabel(r"$\sum|\Delta\,\mathrm{tap}|$ [steps] (dotted)")
    ax_u.set_title("Actuator excursion from initial: DER q_set (solid) / OLTC tap (dotted)")
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)

    ax_sk.set_ylabel("skill = 1 - RMSE/RMSE_zero")
    ax_sk.set_title("One-step prediction skill (>0 beats zero-predictor)")
    ax_sk.grid(True, alpha=0.3, axis="y")

    fig.suptitle("003 multi-run comparison", fontsize=12)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[analysis] saved -> {os.path.abspath(save_path)}")
    else:
        plt.ioff()
        plt.show(block=True)


# ---------------------------------------------------------------------------
#  CSV export  (for PGF plots)
# ---------------------------------------------------------------------------

def _trafo_shortname(key: str) -> str:
    """Shorten a trafo key like 'DSO_2|T3' → 'T3' for CSV column names."""
    return key.split("|")[-1].replace(" ", "_")


def export_csv(
    runs: List[Tuple[str, Optional[str], str]],
    out_dir: str,
    v_ref: float = 1.03,
) -> List[str]:
    """Write one CSV per run to *out_dir*.

    File name
    ---------
    ``<mode>.csv`` where mode = ``label.split("_")[0]``
    (e.g. label ``ann_frozen_biased30pct`` → ``ann.csv``).

    Columns
    -------
    time_min             -- simulation time [min]
    h_rel                -- ||H_used - H_analytical||_F / ||H_analytical||_F
    h_q_trafo_rel        -- same restricted to Q_trafo rows (NaN if unavailable)
    q_sum_abs_err_mvar   -- sum_i |q_set_i - q_act_i| over interface trafos [Mvar]

    H-error steps are aligned to the simulation time axis via linear
    interpolation when the sidecar step count differs from the log length.

    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    for pkl_path, sidecar_path, label in runs:
        log, sidecar = load_run(pkl_path, sidecar_path)
        if not log:
            print(f"[csv] skipping empty log: {os.path.basename(pkl_path)}")
            continue

        # ── simulation time axis and Q-sum error ─────────────────────────────
        qt = q_trafo_tracking(log)
        times_min = qt["times_s"] / 60.0
        N_sim = len(times_min)
        err = np.abs(qt["error"])
        if err.ndim == 1:
            err = err[:, np.newaxis]
        q_sum = np.nansum(err, axis=1)   # (N_sim,)  sum_i |q_set_i - q_act_i|

        # ── H error aligned to sim time ───────────────────────────────────────
        h_rel   = np.full(N_sim, np.nan)
        h_q_rel = np.full(N_sim, np.nan)

        if sidecar is not None:
            herr = h_analytical_error(sidecar)
            if herr is not None:
                N_h = len(herr["relative"])
                sim_idx = np.arange(N_sim, dtype=float)
                if N_h == N_sim:
                    h_rel = herr["relative"]
                else:
                    h_steps = np.linspace(0.0, float(N_sim - 1), N_h)
                    h_rel = np.interp(sim_idx, h_steps, herr["relative"])
                if "relative_q_trafo" in herr:
                    rq = herr["relative_q_trafo"]
                    if N_h == N_sim:
                        h_q_rel = rq
                    else:
                        h_q_rel = np.interp(sim_idx, h_steps, rq)

        # ── assemble and save ─────────────────────────────────────────────────
        df = pd.DataFrame({
            "time_min":           times_min,
            "h_rel":              h_rel,
            "h_q_trafo_rel":      h_q_rel,
            "q_sum_abs_err_mvar": q_sum,
        })

        mode = label.split("_")[0]
        ts   = os.path.basename(pkl_path).split("_")[0]   # e.g. 2026-05-28--13-42-10
        path = os.path.join(out_dir, f"{ts}_{mode}.csv")
        df.to_csv(path, index=False, float_format="%.6g")
        print(f"[csv] {path}  ({N_sim} rows)")
        written.append(path)

    return written


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyse a 003_S_DSO_CIGRE_2026 run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "pkl", nargs="?",
        help="Path to .pkl log. Omit to auto-select / pick interactively.",
    )
    ap.add_argument("--sidecar", default=None,
                    help="Path to *_dso2_ctrl.npz sidecar (auto-detected if omitted).")
    ap.add_argument("--result-dir", default=None,
                    help="Override result directory for auto-discovery.")
    ap.add_argument("--save", default=None,
                    help="Save figure to this path instead of showing it.")
    ap.add_argument("--v-ref", type=float, default=1.03,
                    help="Voltage reference [p.u.].")
    ap.add_argument("--last", type=int, default=None, metavar="N",
                    help="Compare the N most recent runs on one figure.")
    ap.add_argument("--filter", default=None, metavar="TEXT",
                    help="Only include pkl files whose basename contains TEXT "
                         "(applied before --last, e.g. 'frozen' or 'biased30pct').")
    ap.add_argument(
        "--csv", nargs="?", const="", metavar="DIR",
        help="Export one CSV per run to DIR (columns: time_min, h_rel, "
             "h_q_trafo_rel, q_sum_abs_err_mvar). "
             "Omit DIR to export alongside the pkl file (or into the "
             "result directory when used with --last).",
    )
    args = ap.parse_args()

    if args.last is not None:
        all_pkls = find_all_runs(args.result_dir)
        if args.filter:
            all_pkls = [p for p in all_pkls if args.filter in os.path.basename(p)]
        selected = all_pkls[-args.last:]
        runs = []
        for p in selected:
            s = _sidecar_for_pkl(p)
            # derive a label from everything after the timestamp segment
            stem = os.path.basename(p).replace(".pkl", "")
            parts = stem.split("_", 1)   # split on first "_" only
            label = parts[1] if len(parts) > 1 else stem
            runs.append((p, s, label))
            print(f"[analysis] comparing: {os.path.basename(p)}"
                  + (f" + sidecar" if s else ""))
        if args.csv is not None:
            csv_dir = args.csv if args.csv else os.path.dirname(os.path.abspath(selected[0]))
            export_csv(runs, csv_dir, v_ref=args.v_ref)
        else:
            plot_multi_comparison(runs, v_ref=args.v_ref, save_path=args.save)
    elif args.pkl:
        pkl_path     = args.pkl
        sidecar_path = args.sidecar
        if args.csv is not None:
            csv_dir = args.csv if args.csv else os.path.dirname(os.path.abspath(pkl_path))
            label = os.path.basename(pkl_path).replace(".pkl", "").split("_", 1)
            label = label[1] if len(label) > 1 else label[0]
            export_csv([(pkl_path, sidecar_path, label)], csv_dir, v_ref=args.v_ref)
        else:
            plot_comparison(pkl_path, sidecar_path, v_ref=args.v_ref, save_path=args.save)
    else:
        pkl_path, sidecar_path = pick_run_interactive(args.result_dir)
        if args.csv is not None:
            csv_dir = args.csv if args.csv else os.path.dirname(os.path.abspath(pkl_path))
            label = os.path.basename(pkl_path).replace(".pkl", "").split("_", 1)
            label = label[1] if len(label) > 1 else label[0]
            export_csv([(pkl_path, sidecar_path, label)], csv_dir, v_ref=args.v_ref)
        else:
            plot_comparison(pkl_path, sidecar_path, v_ref=args.v_ref, save_path=args.save)


if __name__ == "__main__":
    main()
