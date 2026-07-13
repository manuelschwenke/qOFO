"""
diag_precond_sweep.py
=====================
A/B the Tier-2 curvature preconditioner against the BO/config baseline by
sweeping ``precondition_lambda_target`` and reporting the *controller-agnostic*
voltage-tracking KPI ``rms_v_ts_pu`` (the same unweighted metric used in the
V5 study — deliberately NOT the OFO objective, so no variant is favoured).

The point is to turn "guess a target" into "read the speed↔margin trade-off
curve and pick the knee":

* ``precondition_g_w = False``  → BO/config ``g_w`` (the baseline row).
* ``precondition_g_w = True``, one row per target → the cap-only rule
  (only ever *reduces* an over-hot loop; see
  :mod:`controller.gw_precondition`).

Usage
-----
    python experiments/diag_precond_sweep.py
    python experiments/diag_precond_sweep.py --targets 0.2,0.3,0.5,0.7 \
        --horizon-min 15
    python experiments/diag_precond_sweep.py \
        --module experiments.005_CIGRE_MULTI --scenario wind_replace

Notes
-----
* ``rms_v_ts_pu`` lower = better tracking.  ``n_sw`` = discrete switching ops.
  A wildly oscillating run shows up as a large ``rms_v_ts_pu`` and/or
  ``converged=False`` (PF failure / short log).
* Run ``--verbose`` to also see the per-controller ``[precond:...]`` lines
  (REDUCED / within-margin / INTEGER-DOMINATED) for each target.
* The default horizon is short (baseline operation only) for a fast first
  read; lengthen it (and pick a ``--scenario`` with contingencies) for the
  comparison you put in the thesis.

Author: Manuel Schwenke (with Claude Code)
Date: 2026-06-23
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Dict, List

import numpy as np

# Allow running as a plain script (`python experiments/diag_precond_sweep.py`):
# put the project root on sys.path before the `experiments.*` imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.helpers.comparison_metrics import cigre_summary_table
from experiments.helpers.records import MultiTSOIterationRecord
from experiments.runners.multi_tso_dso import run_multi_tso_dso


def _build_cfg(mod, horizon_s: float, scenario: str | None, verbose: int):
    cfg = mod.make_base_config()
    if scenario:
        scen = getattr(mod, "SCENARIOS", {})
        if scenario not in scen:
            raise SystemExit(
                f"scenario '{scenario}' not in {mod.__name__}.SCENARIOS "
                f"({sorted(scen)})"
            )
        for k, v in scen[scenario].items():
            setattr(cfg, k, v)
    cfg.n_total_s = horizon_s
    cfg.verbose = verbose
    # Headless: make_base_config() enables live plots; force them off.
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.live_plot_tracking = False
    cfg.run_stability_analysis = False
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", default="experiments.002_M_TSO_M_DSO_COMPARE",
                    help="experiment module exposing make_base_config()")
    ap.add_argument("--targets", default="0.2,0.3,0.5,0.7",
                    help="comma-separated lambda_max targets to sweep")
    ap.add_argument("--horizon-min", type=float, default=10.0,
                    help="simulation horizon in minutes")
    ap.add_argument("--scenario", default=None,
                    help="optional key into <module>.SCENARIOS")
    ap.add_argument("--granularity", default="class", choices=["class", "column"])
    ap.add_argument("--verbose", type=int, default=0,
                    help="0 quiet; 1 shows per-controller [precond:...] lines")
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    horizon_s = args.horizon_min * 60.0

    logs: Dict[str, List[MultiTSOIterationRecord]] = {}

    def _run(name: str, precond: bool, target: float | None) -> None:
        cfg = _build_cfg(mod, horizon_s, args.scenario, args.verbose)
        cfg.precondition_g_w = precond
        if precond:
            cfg.precondition_lambda_target = float(target)
            cfg.precondition_granularity = args.granularity
        if args.verbose:
            print(f"\n=== {name} (precondition_g_w={precond}, target={target}) ===")
        try:
            logs[name] = run_multi_tso_dso(cfg)
        except Exception as exc:  # diverged / solver failure
            print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
            logs[name] = []

    _run("BO_baseline", precond=False, target=None)
    for t in targets:
        _run(f"precond_{t:g}", precond=True, target=t)

    # KPI table (v_set from the last cfg; identical across runs).
    v_set = float(mod.make_base_config().v_setpoint_pu)
    df = cigre_summary_table(logs, v_set=v_set)

    base = df.loc["BO_baseline", "rms_v_ts_pu"] if "BO_baseline" in df.index else np.nan

    print()
    print(f"  module={args.module}  scenario={args.scenario}  "
          f"horizon={args.horizon_min:g} min  granularity={args.granularity}")
    print(f"  {'variant':<16}{'rms_v_ts_pu':>13}{'d vs BO':>11}"
          f"{'n_sw':>7}{'steps':>7}{'ok':>5}")
    print("  " + "-" * 59)
    for name in logs:
        rms = float(df.loc[name, "rms_v_ts_pu"]) if name in df.index else np.nan
        nsw = int(df.loc[name, "n_sw"]) if name in df.index else 0
        ok = bool(df.loc[name, "converged"]) if name in df.index else False
        if np.isfinite(base) and base > 0 and np.isfinite(rms):
            dpct = f"{(rms - base) / base * 100:+.1f}%"
        else:
            dpct = "--"
        rms_s = f"{rms:.6f}" if np.isfinite(rms) else "nan"
        print(f"  {name:<16}{rms_s:>13}{dpct:>11}{nsw:>7}"
              f"{len(logs[name]):>7}{('y' if ok else 'n'):>5}")
    print()
    print("  Lower rms_v_ts_pu = better tracking. Pick the knee: the largest")
    print("  target (fastest loop) that still tracks as well as the baseline.")


if __name__ == "__main__":
    main()
