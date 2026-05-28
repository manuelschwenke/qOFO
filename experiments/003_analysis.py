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
        herr = h_analytical_error(sidecar)
        if herr is not None:
            print("\n=== H analytical error  ||H_used - H_analytical|| / ||H_analytical|| ===")
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
    n_panels = 3 if has_sidecar else 2

    fig = plt.figure(figsize=(14, 4 * n_panels))
    gs  = GridSpec(n_panels, 1, figure=fig, hspace=0.50)
    panel = 0

    # ── Panel 1: H analytical error ──────────────────────────────────────────
    if has_sidecar:
        ax = fig.add_subplot(gs[panel]); panel += 1
        herr = h_analytical_error(sidecar)
        if herr is not None:
            steps = herr["steps"]
            mean_full = herr["relative"].mean() * 100
            ax.plot(steps, herr["relative"] * 100, color="tab:blue", lw=1.5,
                    label=f"Full H (rel.)")
            if "relative_q_trafo" in herr:
                n_q = int(herr["n_q_trafo"])
                mean_q = herr["relative_q_trafo"].mean() * 100
                ax.plot(steps, herr["relative_q_trafo"] * 100,
                        color="tab:blue", lw=1.5, ls="--",
                        label=f"Q_trafo rows only ({n_q} rows, rel.)")
                summary = (f"Mean error\n"
                           f"Full H : {mean_full:.2f} %\n"
                           f"Q_trafo: {mean_q:.2f} %")
            else:
                summary = f"Mean error\nFull H: {mean_full:.2f} %"
            ax.text(0.99, 0.97, summary, transform=ax.transAxes,
                    fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.set_xlabel("DSO step k")
            ax.set_ylabel("Rel. H error [%]", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax.set_title(
                r"$\|H_\mathrm{used}(k) - H_\mathrm{analytical}(k)\|_F \;/\;"
                r"\|H_\mathrm{analytical}(k)\|_F$"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(steps, herr["absolute"], color="tab:orange", lw=0.8, alpha=0.7,
                     label="Full H (abs.)")
            if "absolute_q_trafo" in herr:
                ax2.plot(steps, herr["absolute_q_trafo"], color="tab:orange",
                         lw=0.8, alpha=0.7, ls="--", label="Q_trafo rows (abs.)")
            ax2.set_ylabel("Abs. error [p.u./Mvar]", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
        else:
            ax.text(0.5, 0.5, "h_analytical not in sidecar\n(re-run 003 to record it)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11,
                    color="gray")
            ax.set_title("H analytical error (data not available)")

    # ── Panel 2: Q_trafo setpoint vs actual ──────────────────────────────────
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
    fig = plt.figure(figsize=(14, 12))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.50)
    ax_h = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1])
    ax_v = fig.add_subplot(gs[2])

    for (pkl_path, sidecar_path, label), col in zip(runs, colors):
        log, sidecar = load_run(pkl_path, sidecar_path)

        # ── Panel 1: H analytical error ──────────────────────────────────
        if sidecar is not None:
            herr = h_analytical_error(sidecar)
            if herr is not None:
                mean_full = herr["relative"].mean() * 100
                ax_h.plot(herr["steps"], herr["relative"] * 100,
                          color=col, lw=1.5,
                          label=f"{label} full H  (mean={mean_full:.2f}%)")
                if "relative_q_trafo" in herr:
                    mean_q = herr["relative_q_trafo"].mean() * 100
                    ax_h.plot(herr["steps"], herr["relative_q_trafo"] * 100,
                              color=col, lw=1.5, ls="--",
                              label=f"{label} Q_trafo (mean={mean_q:.2f}%)")

        # ── Panel 2: Q_trafo mean absolute error ─────────────────────────
        qt = q_trafo_tracking(log)
        err_arr = np.abs(qt["error"])
        if err_arr.ndim == 2 and err_arr.shape[0] > 0:
            times_min = qt["times_s"] / 60.0
            mean_abs_err = np.nanmean(err_arr, axis=1)
            ax_q.plot(times_min, mean_abs_err, color=col, lw=1.5, label=label)

        # ── Panel 3: Voltage deviation (mean) ────────────────────────────
        vd = voltage_deviation(log, v_ref=v_ref)
        if len(vd["times_s"]) > 0:
            times_min_v = vd["times_s"] / 60.0
            ax_v.plot(times_min_v, vd["dev_mean"] * 1e3, color=col, lw=1.5, label=label)
            ax_v.fill_between(times_min_v,
                              vd["dev_min"] * 1e3, vd["dev_max"] * 1e3,
                              color=col, alpha=0.08)

    ax_h.set_xlabel("DSO step k")
    ax_h.set_ylabel("Rel. H error [%]")
    ax_h.set_title(r"$\|H_\mathrm{used} - H_\mathrm{analytical}\|_F \;/\; \|H_\mathrm{analytical}\|_F$")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    ax_q.set_xlabel("Time [min]")
    ax_q.set_ylabel("Mean |Q_err| [Mvar]")
    ax_q.set_title("Interface Q tracking error (mean over trafos)")
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.3)

    ax_v.axhline(0,   color="k",    lw=0.8)
    ax_v.axhline( 10, color="gray", lw=0.8, ls="--", label="+10 mpu")
    ax_v.axhline(-10, color="gray", lw=0.8, ls="--", label="-10 mpu")
    ax_v.set_xlabel("Time [min]")
    ax_v.set_ylabel("V_mean - V_ref [mpu]")
    ax_v.set_title(f"DSO_2 voltage deviation (V_ref = {v_ref:.3f} p.u.)")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)

    fig.suptitle("003 multi-run comparison", fontsize=11)

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
    """Write three CSVs to *out_dir* covering all *runs*.

    Files produced
    --------------
    h_error.csv   — step, h_rel[_label], h_abs[_label],
                    h_q_trafo_rel[_label], h_q_trafo_abs[_label]
    q_tracking.csv — time_min, q_set_<T>[_label], q_act_<T>[_label]
    voltage.csv   — time_min, v_min[_label], v_mean[_label], v_max[_label],
                    dev_min_mpu[_label], dev_mean_mpu[_label], dev_max_mpu[_label]

    Column names have a ``_<label>`` suffix only when there is more than one
    run, so single-run exports stay clean for pgfplots.

    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    multi = len(runs) > 1
    suffix = lambda lbl: f"_{lbl}" if multi else ""

    h_frames:   List[pd.DataFrame] = []
    q_frames:   List[pd.DataFrame] = []
    v_frames:   List[pd.DataFrame] = []

    for pkl_path, sidecar_path, label in runs:
        log, sidecar = load_run(pkl_path, sidecar_path)
        sfx = suffix(label)

        # ── H error ──────────────────────────────────────────────────────────
        if sidecar is not None:
            herr = h_analytical_error(sidecar)
            if herr is not None:
                d: Dict = {
                    "step":            herr["steps"],
                    f"h_rel{sfx}":     herr["relative"],
                    f"h_abs{sfx}":     herr["absolute"],
                }
                if "relative_q_trafo" in herr:
                    d[f"h_q_trafo_rel{sfx}"] = herr["relative_q_trafo"]
                    d[f"h_q_trafo_abs{sfx}"] = herr["absolute_q_trafo"]
                h_frames.append(pd.DataFrame(d).set_index("step"))

        # ── Q tracking ───────────────────────────────────────────────────────
        qt = q_trafo_tracking(log)
        n_tr = qt["setpoint"].shape[1]
        qd: Dict = {"time_min": qt["times_s"] / 60.0}
        for i in range(n_tr):
            name = _trafo_shortname(qt["trafo_keys"][i]) if qt["trafo_keys"] else f"T{i}"
            qd[f"q_set_{name}{sfx}"] = qt["setpoint"][:, i]
            qd[f"q_act_{name}{sfx}"] = qt["actual"][:, i]
        q_frames.append(pd.DataFrame(qd).set_index("time_min"))

        # ── Voltage ───────────────────────────────────────────────────────────
        vd = voltage_deviation(log, v_ref=v_ref)
        vdf: Dict = {
            "time_min":           vd["times_s"] / 60.0,
            f"v_min{sfx}":        vd["v_min"],
            f"v_mean{sfx}":       vd["v_mean"],
            f"v_max{sfx}":        vd["v_max"],
            f"dev_min_mpu{sfx}":  vd["dev_min"]  * 1e3,
            f"dev_mean_mpu{sfx}": vd["dev_mean"] * 1e3,
            f"dev_max_mpu{sfx}":  vd["dev_max"]  * 1e3,
        }
        v_frames.append(pd.DataFrame(vdf).set_index("time_min"))

    written: List[str] = []

    def _save(frames: List[pd.DataFrame], name: str) -> None:
        if not frames:
            return
        merged = frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)
        path = os.path.join(out_dir, name)
        merged.reset_index().to_csv(path, index=False, float_format="%.6g")
        print(f"[csv] {path}  ({len(merged)} rows, {len(merged.columns)} data cols)")
        written.append(path)

    _save(h_frames, "h_error.csv")
    _save(q_frames, "q_tracking.csv")
    _save(v_frames, "voltage.csv")
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
    ap.add_argument(
        "--csv", nargs="?", const="", metavar="DIR",
        help="Export h_error.csv / q_tracking.csv / voltage.csv to DIR. "
             "Omit DIR to export alongside the pkl file (or into the "
             "result directory when used with --last).",
    )
    args = ap.parse_args()

    if args.last is not None:
        all_pkls = find_all_runs(args.result_dir)
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
