#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/004b_REFRESH_PROOF.py
==================================
Side-experiment to :mod:`experiments.004_LOCAL_VS_FULL_SENS`.

Tests the hypothesis that the FULL-mode steady-state Q-tracking drift
observed in the 240-min 004 run comes from cached-Jacobian staleness
(the shared Jacobian is frozen at the post-Phase-2 operating point;
the profile evolves over 4 hours; the global Jacobian's wide-area
entries drift away from their cached values).

Three scenarios share the same 240-min ``wind_replace`` profile —
contingencies are *removed* here so the only disturbance is the slow
profile evolution.  This isolates the staleness mechanism from
contingency-induced topology changes (those introduce a separate
fragility in the controller's ``compute_dQgen_dQder_matrix`` path when
the Jacobian is rebuilt post-trip, which is orthogonal to the
staleness question).

* **FULL_FROZEN**   — ``shared_jac`` built once at post-Phase-2, never
  refreshed.  Expected to show steady-state Q-tracking drift if
  staleness is the cause.
* **FULL_REFRESH**  — ``shared_jac`` rebuilt every TSO tick (every
  3 min).  Expected to *not* show drift if staleness is the cause.
  Hypothesis falsified if drift persists.
* **LOCAL_FROZEN**  — per-controller reduced Jacobians, never
  refreshed.  Expected to *not* show drift (local terms are structurally
  insensitive to wide-area OP evolution).  Acts as control.

Outputs land in ``results/004b_refresh_proof/`` so they do not clobber
the contingency-included 004 results.

Author: Manuel Schwenke / Claude Code
Date: 2026-05-27
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import MultiTSOIterationRecord
from experiments.runners import run_multi_tso_dso

# ``004_LOCAL_VS_FULL_SENS`` starts with a digit, so import via importlib.
_sens = importlib.import_module("experiments.004_LOCAL_VS_FULL_SENS")
make_004_base_config = _sens.make_base_config


def make_base_config() -> MultiTSOConfig:
    """Same config as 004's FULL-vs-LOCAL base, but a 240-min profile and
    NO contingencies (refresh-vs-frozen proof needs a clean steady ramp)."""
    cfg = make_004_base_config()
    cfg.n_total_s = 60.0 * 60.0 * 4.0   # 240 min
    cfg.contingencies = []
    return cfg


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "FULL_FROZEN": dict(
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=False,
        numerical_h=False,
    ),
    "FULL_REFRESH": dict(
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=True,
        numerical_h=False,
    ),
    "FULL_NUM_FROZEN": dict(
        # Numerical (finite-difference) H, pinned once at post-Phase-2.
        # Uses run_control=True during perturbation — every column is
        # closed-loop including V_gen.  This gave excellent Q-tracking
        # but catastrophic V-tracking in the first round.
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=False,
        numerical_h=True,
        numerical_h_closed_loop=True,
    ),
    "FULL_NUM_OPENLOOP": dict(
        # Numerical H with run_control=False (no plant-side QV-loop
        # reaction during perturbation) plus analytical T_prime applied
        # to the DER block.  Mirrors the analytical structure exactly,
        # so only the base ``∂y/∂Q_DER`` block differs.  Isolates
        # whether the analytical bias lives in the base Schur-complement
        # computation (this scenario should match analytical FULL) or
        # somewhere else.
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=False,
        numerical_h=True,
        numerical_h_closed_loop=False,
    ),
    "FULL_NUM_REFRESH": dict(
        # Numerical H, recomputed on every TSO tick.  Tests whether
        # operating-point staleness in the numerical view contributes.
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=True,
        numerical_h=True,
    ),
    "LOCAL_FROZEN": dict(
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        refresh_shared_jac_on_tso=False,
        numerical_h=False,
    ),
}


def run_one_scenario(
    name: str,
    overrides: Dict[str, Any],
    out_root: str,
) -> List[MultiTSOIterationRecord]:
    cfg = make_base_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    scen_dir = os.path.join(out_root, name)
    cfg.result_dir = scen_dir
    os.makedirs(scen_dir, exist_ok=True)
    print()
    print("=" * 72)
    print(f"  RUNNING SCENARIO {name}")
    print("=" * 72)
    print(f"  refresh_shared_jac_on_tso={cfg.refresh_shared_jac_on_tso}")
    print(f"  local_sensitivities_tso={cfg.local_sensitivities_tso}")
    print(f"  local_sensitivities_dso={cfg.local_sensitivities_dso}")
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
        log = []
    pkl_path = os.path.join(scen_dir, "log.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [{name}] wrote {len(log)} records -> {pkl_path}")
    return log


def _ts_voltage_rms_error(log, v_set_pu):
    t_min, rms = [], []
    for rec in log:
        if not rec.zone_v_mean:
            continue
        errs = np.array([v - v_set_pu for v in rec.zone_v_mean.values()],
                        dtype=np.float64)
        t_min.append(rec.time_s / 60.0)
        rms.append(float(np.sqrt(np.mean(errs * errs))))
    return np.asarray(t_min), np.asarray(rms)


def _ds_q_setpoint_rms_error(log):
    last_qset: Dict[str, float] = {}
    t_min, rms = [], []
    for rec in log:
        for did, qs in rec.dso_q_set_mvar.items():
            if qs is not None:
                last_qset[did] = float(qs)
        errs: List[float] = []
        for did, qa in rec.dso_q_actual_mvar.items():
            if qa is None or did not in last_qset:
                continue
            errs.append(float(qa) - last_qset[did])
        if not errs:
            continue
        t_min.append(rec.time_s / 60.0)
        rms.append(float(np.sqrt(np.mean(np.asarray(errs) ** 2))))
    return np.asarray(t_min), np.asarray(rms)


SCEN_STYLE: Dict[str, Dict[str, Any]] = {
    "FULL_FROZEN":       dict(color="#1f77b4", linestyle="-",  label="FULL analytical frozen"),
    "FULL_REFRESH":      dict(color="#2ca02c", linestyle=":",  label="FULL analytical refresh"),
    "FULL_NUM_FROZEN":   dict(color="#9467bd", linestyle="-.", label="FULL numerical closed-loop"),
    "FULL_NUM_OPENLOOP": dict(color="#ff7f0e", linestyle=(0, (5, 1)), label="FULL numerical open-loop + T'"),
    "FULL_NUM_REFRESH":  dict(color="#bcbd22", linestyle=(0, (3, 1, 1, 1)), label="FULL numerical refresh"),
    "LOCAL_FROZEN":      dict(color="#d62728", linestyle="--", label="LOCAL frozen"),
}


def plot_comparison(logs: Dict[str, List[MultiTSOIterationRecord]], out_dir: str,
                    v_set_pu: float) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # V RMS
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name in SCEN_STYLE:
        log = logs.get(name, [])
        if not log:
            continue
        t, rms = _ts_voltage_rms_error(log, v_set_pu)
        if rms.size:
            ax.plot(t, rms * 1e3, linewidth=1.6, **SCEN_STYLE[name])
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"$\mathrm{RMS}_V \; [\mathrm{mp.u.}]$")
    ax.set_title("TS voltage tracking: refresh vs frozen sensitivities (no contingencies)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"proof_v_rms.{ext}"), dpi=150)
    plt.close(fig)

    # Q RMS
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name in SCEN_STYLE:
        log = logs.get(name, [])
        if not log:
            continue
        t, rms = _ds_q_setpoint_rms_error(log)
        if rms.size:
            ax.plot(t, rms, linewidth=1.6, **SCEN_STYLE[name])
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"$\mathrm{RMS}_Q \; [\mathrm{Mvar}]$")
    ax.set_title("DSO Q-setpoint tracking: refresh vs frozen sensitivities (no contingencies)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"proof_q_rms.{ext}"), dpi=150)
    plt.close(fig)

    # Console summary
    print()
    print("=" * 84)
    print("  AGGREGATE METRICS (time-averaged unless 'peak') — no contingencies")
    print("=" * 84)
    print(f"  {'scenario':<14s} | {'V_rms avg':>10s} | {'V_rms peak':>11s} | "
          f"{'Q_rms avg':>11s} | {'Q_rms peak':>12s}")
    print(f"  {'':<14s} | {'[mpu]':>10s} | {'[mpu]':>11s} | "
          f"{'[Mvar]':>11s} | {'[Mvar]':>12s}")
    print("  " + "-" * 80)
    for name in SCEN_STYLE:
        log = logs.get(name, [])
        if not log:
            continue
        _, v_rms = _ts_voltage_rms_error(log, v_set_pu)
        _, q_rms = _ds_q_setpoint_rms_error(log)
        # Compute "steady-state" averages from second-half too
        n_half = len(v_rms) // 2
        v_avg_steady = float(np.nanmean(v_rms[n_half:])) if v_rms.size else float("nan")
        q_avg_steady = float(np.nanmean(q_rms[n_half:])) if q_rms.size else float("nan")
        print(f"  {name:<14s} | "
              f"{float(np.nanmean(v_rms)) * 1e3:>10.3f} | "
              f"{float(np.nanmax(v_rms))  * 1e3:>11.3f} | "
              f"{float(np.nanmean(q_rms)):>11.3f} | "
              f"{float(np.nanmax(q_rms)):>12.3f}")
    print()
    print("  (Steady-state second-half averages, t > 120 min)")
    print(f"  {'scenario':<14s} | {'V_rms s.s.':>11s} | {'Q_rms s.s.':>11s}")
    print(f"  {'':<14s} | {'[mpu]':>11s} | {'[Mvar]':>11s}")
    print("  " + "-" * 45)
    for name in SCEN_STYLE:
        log = logs.get(name, [])
        if not log:
            continue
        _, v_rms = _ts_voltage_rms_error(log, v_set_pu)
        _, q_rms = _ds_q_setpoint_rms_error(log)
        n_half = len(v_rms) // 2
        v_avg = float(np.nanmean(v_rms[n_half:])) if v_rms.size else float("nan")
        q_avg = float(np.nanmean(q_rms[n_half:])) if q_rms.size else float("nan")
        print(f"  {name:<14s} | {v_avg * 1e3:>11.3f} | {q_avg:>11.3f}")
    print("=" * 84)


def main(only: Optional[List[str]] = None) -> None:
    out_root = os.path.join("results", "004b_refresh_proof")
    os.makedirs(out_root, exist_ok=True)
    selected = list(SCENARIOS.keys())
    if only is not None:
        selected = [s for s in selected if s in only]
    for name in selected:
        run_one_scenario(name, SCENARIOS[name], out_root)
    logs = {}
    for name in SCENARIOS:
        pkl_path = os.path.join(out_root, name, "log.pkl")
        if os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as f:
                logs[name] = pickle.load(f)
        else:
            logs[name] = []
    cfg = make_base_config()
    plot_comparison(logs, out_dir=out_root, v_set_pu=cfg.v_setpoint_pu)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--only" in args:
        idx = args.index("--only")
        only = [s.strip() for s in args[idx + 1].split(",") if s.strip()]
        main(only=only)
    else:
        main()
