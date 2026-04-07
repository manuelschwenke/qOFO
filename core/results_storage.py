"""
Results Storage
===============

Save and load cascade simulation results in a structured directory layout::

    results/
        001_20240610_120000/
            cascade_result.pkl      ← full CascadeResult (pickle)
            config.json             ← CascadeConfig as JSON
            tso_plot.svg / .png     ← TSO subplot figure
            dso_plot.svg / .png     ← DSO subplot figure
            summary.txt            ← human-readable run summary

Usage::

    from core.results_storage import save_results, load_results

    result = run_cascade(config)
    run_dir = save_results(result, config)          # → "results/001_..."
    result2, config2 = load_results(run_dir)

Author: Claude (generated)
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from run_cascade import CascadeResult
    from core.cascade_config import CascadeConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  Directory management
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _next_run_id(base_dir: str) -> int:
    """Return the next sequential run ID (1-based) in *base_dir*."""
    if not os.path.isdir(base_dir):
        return 1
    existing = []
    for name in os.listdir(base_dir):
        m = re.match(r"^(\d{3,})_", name)
        if m:
            existing.append(int(m.group(1)))
    return max(existing, default=0) + 1


def _make_run_dir(base_dir: str) -> str:
    """Create and return a new run directory ``NNN_YYYYMMDD_HHMMSS``."""
    os.makedirs(base_dir, exist_ok=True)
    run_id = _next_run_id(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{run_id:03d}_{timestamp}"
    run_dir = os.path.join(base_dir, dirname)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    result: "CascadeResult",
    config: "CascadeConfig",
    *,
    base_dir: str = DEFAULT_RESULTS_DIR,
    save_pickle: bool = True,
    save_plots: bool = True,
    wall_time_s: Optional[float] = None,
) -> str:
    """
    Persist simulation results to a structured directory.

    Parameters
    ----------
    result : CascadeResult
        The simulation result (log + controller configs).
    config : CascadeConfig
        The configuration that produced this result.
    base_dir : str
        Parent directory for all runs (default: ``<project>/results``).
    save_pickle : bool
        Whether to write the full CascadeResult as a pickle file.
    save_plots : bool
        Whether to generate and save TSO/DSO plots as SVG and PNG.
    wall_time_s : float, optional
        Wall-clock time for the simulation run (seconds).  If provided,
        it is recorded in the summary and config JSON.

    Returns
    -------
    str
        Path to the created run directory.
    """
    run_dir = _make_run_dir(base_dir)

    # ── 1) Pickle ─────────────────────────────────────────────────────────
    if save_pickle:
        pkl_path = os.path.join(run_dir, "cascade_result.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── 2) Config JSON ────────────────────────────────────────────────────
    config_dict = config.to_dict()
    if wall_time_s is not None:
        config_dict["_wall_time_s"] = round(wall_time_s, 2)
    config_dict["_saved_at"] = datetime.now().isoformat()
    config_dict["_run_dir"] = os.path.basename(run_dir)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # ── 3) Plots (SVG + PNG) ──────────────────────────────────────────────
    if save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend for file export
            from visualisation.plot_cascade import plot_all

            fig_tso, fig_dso = plot_all(
                result.log, result.tso_config, result.dso_config, show=False,
            )

            for fig, prefix in [(fig_tso, "tso_plot"), (fig_dso, "dso_plot")]:
                for ext in ("svg", "png"):
                    fig_path = os.path.join(run_dir, f"{prefix}.{ext}")
                    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)

        except Exception as e:
            # Don't let a plotting error prevent saving the other artefacts
            print(f"[results_storage] WARNING: plot export failed: {e}")

    # ── 4) Summary text ───────────────────────────────────────────────────
    summary_path = os.path.join(run_dir, "summary.txt")
    _write_summary(summary_path, result, config, wall_time_s)

    print(f"[results_storage] Saved to: {run_dir}")
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  Load
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(
    run_dir: str,
) -> Tuple["CascadeResult", "CascadeConfig"]:
    """
    Load a previously saved simulation run.

    Parameters
    ----------
    run_dir : str
        Path to the run directory (e.g. ``results/001_20240610_120000``).

    Returns
    -------
    (CascadeResult, CascadeConfig)
    """
    from core.cascade_config import CascadeConfig

    # Load pickle
    pkl_path = os.path.join(run_dir, "cascade_result.pkl")
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    # Load config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = CascadeConfig.from_dict(config_dict)

    return result, config


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary writer
# ═══════════════════════════════════════════════════════════════════════════════

def _write_summary(
    path: str,
    result: "CascadeResult",
    config: "CascadeConfig",
    wall_time_s: Optional[float],
) -> None:
    """Write a human-readable summary text file."""
    import numpy as np

    log = result.log
    lines = []
    a = lines.append

    a("=" * 72)
    a("  CASCADE SIMULATION SUMMARY")
    a("=" * 72)
    a(f"  Date:         {datetime.now():%Y-%m-%d %H:%M:%S}")
    a(f"  V setpoint TSO: {config.v_setpoint_pu:.3f} p.u.")
    a(f"  V setpoint DSO: {config.effective_dso_v_setpoint_pu:.3f} p.u.")
    total_s = config.effective_n_seconds
    if config.uses_sub_minute_timing:
        a(f"  Duration:     {total_s} s  ({total_s/60:.1f} min)")
        a(f"  TSO period:   {config.effective_tso_period_s} s")
        a(f"  DSO period:   {config.effective_dso_period_s} s")
    else:
        a(f"  Duration:     {total_s // 60} min  ({total_s/3600:.1f} h)")
        a(f"  TSO period:   {config.tso_period_min} min")
        a(f"  DSO period:   {config.dso_period_min} min")
    a(f"  Profiles:     {'ON' if config.use_profiles else 'OFF'}"
      f"  ({os.path.basename(config.profiles_csv)})")
    a(f"  Start time:   {config.start_time:%Y-%m-%d %H:%M}")
    if wall_time_s is not None:
        a(f"  Wall time:    {wall_time_s:.1f} s  ({wall_time_s/60:.1f} min)")
    a("")

    # Controller steps
    n_tso = sum(1 for r in log if r.tso_active)
    n_dso = sum(1 for r in log if r.dso_active)
    a(f"  TSO steps: {n_tso},  DSO steps: {n_dso}")
    a("")

    # Weights
    a("  ── Objective Weights ──")
    a(f"  g_v (TSO V tracking):   {config.g_v}")
    a(f"  g_q (DSO Q tracking):   {config.g_q}")
    a(f"  dso_g_v (DSO V track):  {config.dso_g_v}")
    a(f"  alpha:                  {config.alpha}")
    a(f"  g_z:                    {config.g_z}")
    a(f"  gz_tso_current:         {config.gz_tso_current}")
    a(f"  gz_dso_current:         {config.gz_dso_current}")
    a("")

    a("  ── TSO g_w (change damping) ──")
    a(f"  Q_DER:   {config.gw_tso_q_der}")
    a(f"  Q_PCC:   {config.gw_tso_q_pcc}")
    a(f"  V_gen:   {config.gw_tso_v_gen}")
    a(f"  OLTC:    {config.gw_tso_oltc}")
    a(f"  Shunt:   {config.gw_tso_shunt}")
    a(f"  OLTC cross: {config.gw_oltc_cross_tso}")
    a("")

    a("  ── DSO g_w (change damping) ──")
    a(f"  Q_DER:   {config.gw_dso_q_der}")
    a(f"  OLTC:    {config.gw_dso_oltc}")
    a(f"  Shunt:   {config.gw_dso_shunt}")
    a(f"  OLTC cross: {config.gw_oltc_cross_dso}")
    a("")

    a("  ── Generator Capability ──")
    a(f"  xd:        {config.gen_xd_pu} p.u.")
    a(f"  i_f_max:   {config.gen_i_f_max_pu} p.u.")
    a(f"  beta:      {config.gen_beta}")
    a(f"  q0:        {config.gen_q0_pu} p.u.")
    a("")

    a("  ── Reserve Observer ──")
    a(f"  Enabled:     {config.enable_reserve_observer}")
    a(f"  Threshold:   {config.reserve_q_threshold_mvar} Mvar")
    a(f"  Release:     {config.reserve_q_release_mvar} Mvar")
    a(f"  Cooldown:    {config.effective_reserve_cooldown_s:.0f} s  ({config.reserve_cooldown_min} min)")
    a("")

    # Contingencies
    if config.contingencies:
        a("  ── Contingencies ──")
        for c in config.contingencies:
            t = c.effective_time_s
            if t % 60 == 0:
                a(f"  min {int(t // 60):4d}: {c.action.upper()} "
                  f"{c.element_type} {c.element_index}")
            else:
                a(f"  t={t:.0f}s: {c.action.upper()} "
                  f"{c.element_type} {c.element_index}")
        a("")

    # Final state
    if log:
        final = log[-1]
        if final.plant_tn_voltages_pu is not None:
            v = final.plant_tn_voltages_pu
            a("  ── Final TN Voltages ──")
            a(f"  min={np.min(v):.4f}  mean={np.mean(v):.4f}  max={np.max(v):.4f}")
            a(f"  max |V - V_set| = {np.max(np.abs(v - config.v_setpoint_pu)):.4f} p.u.")
            a("")
        if final.plant_dn_voltages_pu is not None:
            v = final.plant_dn_voltages_pu
            a("  ── Final DN Voltages ──")
            a(f"  min={np.min(v):.4f}  mean={np.mean(v):.4f}  max={np.max(v):.4f}")
            a(f"  max |V - V_set| = {np.max(np.abs(v - config.effective_dso_v_setpoint_pu)):.4f} p.u.")
            a("")

    a("=" * 72)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
