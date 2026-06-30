#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/004_LOCAL_VS_FULL_SENS.py
=====================================
Compare full-network (centralised model) vs decentralised (per-controller
Ward-reduced) sensitivity Jacobians on the IEEE 39-bus multi-TSO /
multi-DSO scenario.

Two scenarios share the same IEEE 39 ``wind_replace`` setup, profiles
and contingency timeline; the only axis that varies is how each TSO /
DSO controller obtains its internal sensitivity Jacobian:

* **FULL** (baseline) — every TSO and DSO controller holds the same
  full-network :class:`sensitivity.jacobian.JacobianSensitivities` built
  from the entire plant net (IEEE 39 + four HV sub-networks).  Each
  controller's local H matrix is sliced out of the global Jacobian, so
  inter-area coupling is captured implicitly.
  Activated by ``local_sensitivities_tso=False`` and
  ``local_sensitivities_dso=False`` (the historical default).

* **LOCAL** (decentralised) — each TSO controller uses a Jacobian built
  from its own zone net only; each DSO controller uses one built from
  its own HV sub-network only.  Boundaries are replaced by equivalent
  PQ injections at the cached operating point (see
  :mod:`sensitivity.network_reduction` for the exact Ward-style
  reduction).
  Activated by ``local_sensitivities_tso=True`` and
  ``local_sensitivities_dso=True``.

Reported metrics
----------------
The script writes two scenario logs to disk and produces two comparison
figures plus one CSV summary:

* **TS voltage tracking RMS error vs time** (one line per scenario):

      RMS_V(t) = sqrt( mean over zones { ( V_mean_zone(t) − V_set )² } )

  Using the per-zone mean voltage `zone_v_mean[z]` recorded by the
  runner after every PF (so the time series is dense, not subsampled
  to the TSO tick).  Captures how well each scenario keeps the
  transmission voltages on schedule.

* **DS reactive-power setpoint tracking RMS error vs time** (one line
  per scenario):

      RMS_Q(t) = sqrt( mean over DSOs { ( Q_act_dso(t) − Q_set_dso(t) )² } )

  Using `dso_q_set_mvar[dso_id]` (the last TSO-issued setpoint, valid
  until the next TSO tick) and `dso_q_actual_mvar[dso_id]` (the
  realised interface Q at the end of the DSO step).  Quantifies how
  well the decentralised view tracks each DSO's commanded reactive
  power.

* **summary.csv** — one row per scenario with the *time-averaged* RMS
  errors and other aggregate diagnostics (loss energy, peak voltage
  excursion, etc.).

CLI flags
---------
* ``--replot``         — skip the simulation runs, regenerate plots
                         from existing ``log.pkl`` files in
                         ``results/004_local_vs_full/``.
* ``--only FULL,LOCAL``— run only the listed scenarios.

Author: Manuel Schwenke / Claude Code
Date: 2026-05-27
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent, MultiTSOIterationRecord
from experiments.paths import results_path
from experiments.runners import run_multi_tso_dso

# ``002_M_TSO_M_DSO_COMPARE`` starts with a digit, so import via importlib.
_compare = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")
make_002_base_config = _compare.make_base_config


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


def make_base_config() -> MultiTSOConfig:
    """Shared simulation conditions for the FULL-vs-LOCAL comparison.

    Derives from the cascade-OFO 'C' configuration in
    ``002_M_TSO_M_DSO_COMPARE.make_base_config()`` (same weights, ~180-min
    wind_replace profile, contingency-free base timing) and changes only the
    knobs this comparison needs.  The FULL/LOCAL sensitivity-model axis is set
    per scenario in ``SCENARIOS`` below; the base leaves both local-sensitivity
    flags OFF.
    """
    cfg = make_002_base_config()
    cfg.local_sensitivities_tso = False
    cfg.local_sensitivities_dso = False
    cfg.g_w_pcc = 50                  # NOTE: 002's C uses 10; kept at 50 for parity with prior 004 runs.
    cfg.live_plot_controller = False  # headless batch comparison
    cfg.live_plot_cascade = False
    # 004's own contingency timeline (gen-2 trip/restore + one load step).
    cfg.contingencies = [
        ContingencyEvent(minute=30, element_type="gen", element_index=2, action="trip"),
        ContingencyEvent(minute=120, element_type="gen", element_index=2, action="restore"),
        ContingencyEvent(minute=90, element_type="load", bus=27, p_mw=200, q_mvar=100, action="connect"),
        ContingencyEvent(minute=150, element_type="load", bus=27, p_mw=200, q_mvar=100, action="trip"),
    ]
    return cfg


SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ── Full-network sensitivity (baseline) ─────────────────────────────
    # Every controller shares ``shared_jac`` built from the full IEEE 39
    # + HV sub-networks plant.  Inter-area coupling is implicit in the
    # global Jacobian.  ``shared_jac`` stays frozen at the post-Phase-2
    # operating point for the whole simulation.
    "FULL":  dict(
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=False,
    ),
    "TSO-FULL": dict(
        local_sensitivities_tso=False,
        local_sensitivities_dso=True,
        refresh_shared_jac_on_tso=False,
    ),
    # ── Full-network sensitivity with periodic refresh ──────────────────
    # Same as FULL but ``shared_jac`` is rebuilt on every TSO tick from
    # the current operating point and reassigned to every controller.
    # Tests whether the FULL-mode steady-state Q-tracking drift comes
    # from cached-Jacobian staleness (this scenario should recover) or
    # from the structural AVR-stiffness mismatch at the 3W primary
    # (this scenario will still drift).
    "FULL_REFRESH": dict(
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        refresh_shared_jac_on_tso=True,
    ),
    # ── Decentralised sensitivity (Ward-style reduction) ────────────────
    # Per-controller reduced nets: tie-line far-end + 3W primary buses
    # become PQ-load boundaries for TSOs (with synthetic shunts at the
    # primary bus); 3W primary becomes a virtual slack-gen for DSOs.
    # Cross-zone H_ij blocks are zeroed in the coordinator (strict
    # decentralisation).
    "LOCAL": dict(
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
    ),
}


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------


def run_one_scenario(
    name: str,
    overrides: Dict[str, Any],
    out_root: str,
) -> List[MultiTSOIterationRecord]:
    """Run a single scenario and pickle its log.  Returns the log."""
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
    print(f"  local_sensitivities_tso={cfg.local_sensitivities_tso}")
    print(f"  local_sensitivities_dso={cfg.local_sensitivities_dso}")
    print(f"  result_dir={scen_dir}")

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


def load_logs(
    out_root: str,
    names: Optional[List[str]] = None,
) -> Dict[str, List[MultiTSOIterationRecord]]:
    """Reload pickled scenario logs from a previous run."""
    if names is None:
        names = list(SCENARIOS.keys())
    out: Dict[str, List[MultiTSOIterationRecord]] = {}
    for name in names:
        pkl_path = os.path.join(out_root, name, "log.pkl")
        if not os.path.isfile(pkl_path):
            print(f"  [load_logs] missing {pkl_path} -- treating {name} as diverged")
            out[name] = []
            continue
        with open(pkl_path, "rb") as f:
            out[name] = pickle.load(f)
        print(f"  [load_logs] {name}: {len(out[name])} records from {pkl_path}")
    return out


# ---------------------------------------------------------------------------
#  Metrics extraction
# ---------------------------------------------------------------------------


def _ts_voltage_rms_error(
    log: List[MultiTSOIterationRecord],
    v_set_pu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_min, rms_err_pu) over the per-zone mean voltages.

    RMS_V(t) = sqrt( mean_z { ( V_mean_zone_z(t) - v_set_pu )^2 } )

    Records without any zone voltage (e.g. very first step before the
    first PF) are skipped.  The time axis is in minutes for easy
    plotting alongside the contingency timeline.
    """
    t_min: List[float] = []
    rms: List[float] = []
    for rec in log:
        if not rec.zone_v_mean:
            continue
        errs = np.array(
            [v - v_set_pu for v in rec.zone_v_mean.values()],
            dtype=np.float64,
        )
        t_min.append(rec.time_s / 60.0)
        rms.append(float(np.sqrt(np.mean(errs * errs))))
    return np.asarray(t_min), np.asarray(rms)


def _ds_q_setpoint_rms_error(
    log: List[MultiTSOIterationRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_min, rms_err_mvar) over the DSO interface Q-tracking.

    RMS_Q(t) = sqrt( mean_d { ( Q_act_d(t) - Q_set_d(t) )^2 } )

    The TSO issues new Q setpoints only on TSO ticks (every
    ``tso_period_s``), so the setpoint is piecewise-constant.  We hold
    the most recent setpoint forward across off-cycle DSO ticks so the
    error is well-defined at every step the DSO produced a Q_actual.
    """
    last_qset: Dict[str, float] = {}
    t_min: List[float] = []
    rms: List[float] = []
    for rec in log:
        # Forward-fill the latest setpoint
        for did, qs in rec.dso_q_set_mvar.items():
            if qs is not None:
                last_qset[did] = float(qs)
        # Compute the error only when we have both a setpoint and a
        # measurement for the DSO at the current step.
        errs: List[float] = []
        for did, qa in rec.dso_q_actual_mvar.items():
            if qa is None or did not in last_qset:
                continue
            errs.append(float(qa) - last_qset[did])
        if not errs:
            continue
        errs_arr = np.asarray(errs, dtype=np.float64)
        t_min.append(rec.time_s / 60.0)
        rms.append(float(np.sqrt(np.mean(errs_arr * errs_arr))))
    return np.asarray(t_min), np.asarray(rms)


def _summarise(
    log: List[MultiTSOIterationRecord],
    v_set_pu: float,
) -> Dict[str, float]:
    """Aggregate per-scenario stats from a record list."""
    if not log:
        return {
            "n_records":             0,
            "v_rms_time_avg_pu":     float("nan"),
            "v_rms_peak_pu":         float("nan"),
            "q_rms_time_avg_mvar":   float("nan"),
            "q_rms_peak_mvar":       float("nan"),
            "tso_obj_time_avg":      float("nan"),
            "losses_energy_mwh":     float("nan"),
        }
    _, v_rms = _ts_voltage_rms_error(log, v_set_pu)
    _, q_rms = _ds_q_setpoint_rms_error(log)
    # TSO objective per-zone average across time
    obj_vals: List[float] = []
    for rec in log:
        if rec.zone_tso_objective:
            ovals = [
                v for v in rec.zone_tso_objective.values()
                if v is not None and np.isfinite(v)
            ]
            if ovals:
                obj_vals.append(float(np.mean(ovals)))
    # Loss energy (rectangle rule on per-step losses)
    losses_energy_mwh = float("nan")
    if log:
        loss_w_s: List[float] = []
        for r in log:
            if r.total_losses_mw is not None and np.isfinite(r.total_losses_mw):
                loss_w_s.append(float(r.total_losses_mw))
        if loss_w_s:
            # uniform dt_s assumed (from first two records' time delta)
            if len(log) > 1:
                dt_s = float(log[1].time_s - log[0].time_s)
            else:
                dt_s = 60.0
            losses_energy_mwh = float(np.mean(loss_w_s)) * (len(loss_w_s) * dt_s) / 3600.0

    return {
        "n_records":             len(log),
        "v_rms_time_avg_pu":     float(np.nanmean(v_rms)) if v_rms.size else float("nan"),
        "v_rms_peak_pu":         float(np.nanmax(v_rms))  if v_rms.size else float("nan"),
        "q_rms_time_avg_mvar":   float(np.nanmean(q_rms)) if q_rms.size else float("nan"),
        "q_rms_peak_mvar":       float(np.nanmax(q_rms))  if q_rms.size else float("nan"),
        "tso_obj_time_avg":      float(np.mean(obj_vals)) if obj_vals else float("nan"),
        "losses_energy_mwh":     losses_energy_mwh,
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


SCEN_STYLE: Dict[str, Dict[str, Any]] = {
    "FULL":         dict(color="#1f77b4", linestyle="-",  label="FULL (shared_jac, frozen)"),
    "FULL_REFRESH": dict(color="#2ca02c", linestyle=":",  label="FULL_REFRESH (rebuild every TSO tick)"),
    "LOCAL":        dict(color="#d62728", linestyle="--", label="LOCAL (Ward-reduced, frozen)"),
}


def plot_comparison(
    logs: Dict[str, List[MultiTSOIterationRecord]],
    out_dir: str,
    v_set_pu: float,
) -> None:
    """Produce comparison figures + summary.csv for the two scenarios."""
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: TS voltage tracking RMS error ─────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, log in logs.items():
        if not log:
            continue
        t, rms = _ts_voltage_rms_error(log, v_set_pu)
        if rms.size == 0:
            continue
        style = SCEN_STYLE.get(name, dict(label=name))
        ax.plot(t, rms * 1e3, linewidth=1.6, **style)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"$\mathrm{RMS}_V \; [\mathrm{mp.u.}]$")
    ax.set_title(
        "TS voltage tracking error: full-network vs decentralised sensitivities"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"compare_v_rms.{ext}"), dpi=150)
    plt.close(fig)

    # ── Figure 2: DS reactive-power setpoint tracking RMS error ─────────
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, log in logs.items():
        if not log:
            continue
        t, rms = _ds_q_setpoint_rms_error(log)
        if rms.size == 0:
            continue
        style = SCEN_STYLE.get(name, dict(label=name))
        ax.plot(t, rms, linewidth=1.6, **style)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(r"$\mathrm{RMS}_Q \; [\mathrm{Mvar}]$")
    ax.set_title(
        "DSO Q-setpoint tracking error: full-network vs decentralised sensitivities"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"compare_q_rms.{ext}"), dpi=150)
    plt.close(fig)

    # ── summary.csv ──────────────────────────────────────────────────────
    import csv

    rows: List[Dict[str, Any]] = []
    for name, log in logs.items():
        stats = _summarise(log, v_set_pu)
        row = {"scenario": name, **stats}
        rows.append(row)

    csv_path = os.path.join(out_dir, "summary.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"  Wrote {csv_path}")

    # ── Console summary table ───────────────────────────────────────────
    print()
    print("=" * 84)
    print("  AGGREGATE METRICS (time-averaged unless 'peak')")
    print("=" * 84)
    print(
        f"  {'scenario':<8s} | {'V_rms avg':>10s} | {'V_rms peak':>11s} | "
        f"{'Q_rms avg':>11s} | {'Q_rms peak':>12s} | {'losses':>10s}"
    )
    print(
        f"  {'':<8s} | {'[mpu]':>10s} | {'[mpu]':>11s} | "
        f"{'[Mvar]':>11s} | {'[Mvar]':>12s} | {'[MWh]':>10s}"
    )
    print("  " + "-" * 80)
    for row in rows:
        print(
            f"  {row['scenario']:<8s} | "
            f"{row['v_rms_time_avg_pu'] * 1e3:>10.3f} | "
            f"{row['v_rms_peak_pu']     * 1e3:>11.3f} | "
            f"{row['q_rms_time_avg_mvar']:>11.3f} | "
            f"{row['q_rms_peak_mvar']:>12.3f} | "
            f"{row['losses_energy_mwh']:>10.2f}"
        )
    print("=" * 84)


def replot(out_root: str) -> None:
    """Regenerate comparison figures from the persisted ``log.pkl`` files."""
    logs = load_logs(out_root)
    print()
    print("=" * 72)
    print(f"  REPLOTTING from {out_root}/")
    print("=" * 72)
    cfg = make_base_config()
    plot_comparison(logs, out_dir=out_root, v_set_pu=cfg.v_setpoint_pu)


# ---------------------------------------------------------------------------
#  Main entrypoint
# ---------------------------------------------------------------------------


def main(only: Optional[List[str]] = None) -> None:
    out_root = results_path("004_local_vs_full")
    os.makedirs(out_root, exist_ok=True)

    selected = list(SCENARIOS.keys())
    if only is not None:
        unknown = [s for s in only if s not in SCENARIOS]
        if unknown:
            raise ValueError(
                f"--only contains unknown scenarios: {unknown}; "
                f"valid names are {list(SCENARIOS.keys())}"
            )
        selected = [s for s in selected if s in only]

    if not selected:
        print("[main] No scenarios selected; nothing to run.")
    else:
        print(f"[main] Running scenarios: {selected}")
        for name in selected:
            run_one_scenario(name, SCENARIOS[name], out_root)

    # Always build the comparison from every available pickle.
    logs = load_logs(out_root)
    cfg = make_base_config()

    print()
    print("=" * 72)
    print("  AGGREGATING COMPARISON METRICS")
    print("=" * 72)
    plot_comparison(logs, out_dir=out_root, v_set_pu=cfg.v_setpoint_pu)


def _parse_csv_arg(argv: List[str], flag: str) -> Optional[List[str]]:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(
            f"{flag} requires a comma-separated value, e.g. {flag} FULL,LOCAL"
        )
    return [s.strip() for s in argv[idx + 1].split(",") if s.strip()]


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot(results_path("004_local_vs_full"))
    else:
        main(only=_parse_csv_arg(sys.argv, "--only"))


