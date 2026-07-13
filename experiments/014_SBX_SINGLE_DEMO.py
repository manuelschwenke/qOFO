#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/014_SBX_SINGLE_DEMO.py
==================================
SBX-H v6 single-run demonstration with the LIVE Figure 6
(``config.live_plot_sbx``).

Runs ONE simulation of one 015 cell (D2 / D1 / D0 — definitions and
timing imported from ``experiments/015_SBX_COMPARE`` so the two
experiments cannot drift apart) and shows, live per corridor:

* the measured corridor flow against the standard-flow staircase
  q_std with the tier-1 band shaded (v6: no deal schedule exists —
  the band is around q_std),
* escalation markers (the A4 re-planning indicator) and the
  violation-indicator strips,
* the per-cycle deviation staircase q_meas − q_std and the cumulative
  attributed settlement payments per area.

Options mirror the v6 mechanism: ``--support`` runs the
``sbx_support`` arm (planned support agreed in advance — the
supporters hold +2.5 mpu on their sides of the zone-3 corridors
during the stress window); ``--schedule`` consumes a
planning-anchored v_std/band schedule JSON from
``experiments/017_SBX_PLANNING.py`` (contracts then anchor to the
plan instead of the settled-state snapshot; the warmup is bypassed).

At the end the figure is saved to
``results/014_SBX_DEMO/<cell>/sbx_mechanism.png`` together with the
settlement ledger/summary.

The v5 deal-era version of this script (deal markers, calibrated
bands, ``--arm sbx_inert``) is archived in
``_archive/sbx_h_v5/experiments/``.

Run examples:
    python experiments/014_SBX_SINGLE_DEMO.py --cell D2
    python experiments/014_SBX_SINGLE_DEMO.py --cell D2 --support
    python experiments/014_SBX_SINGLE_DEMO.py --cell D1 --no-live \
        --schedule results/017_SBX_PLANNING/schedule_perfect_360min.json

Author: Manuel Schwenke / Claude Code
Date: 2026-07-13 (SBX-H v6)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sbx_h.fail import rep1  # noqa: E402

_015 = importlib.import_module("experiments.015_SBX_COMPARE")

RESULT_DIR = REPO / "results" / "014_SBX_DEMO"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SBX-H v6 single-run demonstration with live "
                    "Figure 6.")
    ap.add_argument("--cell", type=str, default="D2",
                    choices=sorted(_015.CELLS.keys()))
    ap.add_argument("--minutes", type=float,
                    default=_015.DEFAULT_MINUTES)
    ap.add_argument("--support", action="store_true",
                    help="run the sbx_support arm (planned support "
                         "agreed in advance) instead of the plain "
                         "contract arm")
    ap.add_argument("--schedule", type=str, default=None,
                    help="path to a planning-anchored v_std/band "
                         "schedule JSON (from experiments/017_SBX_"
                         "PLANNING.py); contracts then anchor to the "
                         "plan and the snapshot warmup is bypassed")
    ap.add_argument("--no-live", action="store_true",
                    help="disable the live figure (still saves the "
                         "final PNG via a headless redraw)")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    if args.no_live:
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")

    cell = args.cell
    arm = "sbx_support" if args.support else "sbx"
    out_dir = RESULT_DIR / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _015.CELLS[cell]
    print(f"=== 014 SBX-H v6 demo: {cell} ({spec['label']}), "
          f"arm {arm} ===")
    print(f"    {spec['expect']}")
    print(f"    horizon {args.minutes:.0f} min, live plot "
          f"{'OFF' if args.no_live else 'ON'}")

    cfg = _015.make_config(cell, arm, args.minutes)
    cfg.verbose = args.verbose
    cfg.live_plot_sbx = not args.no_live
    if args.schedule is not None:
        sched_path = Path(args.schedule)
        if not sched_path.exists():
            rep1("schedule JSON not found", path=str(sched_path))
        cfg.sbx_v_std_schedule_path = str(sched_path)
        # With a planning schedule the snapshot warmup is obsolete —
        # contracts anchor to the plan from t = 0 (v3 convention).
        cfg.sbx_warmup_s = 0.0

    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    wall = time.perf_counter() - t0
    print(f"  {len(recs)} steps in {wall:.0f} s wall")

    runtime = captured.get("sbx_runtime") or {}
    adapter = runtime.get("adapter")
    if adapter is None:
        rep1("run finished without a constructed SBX adapter — check "
             "sbx_warmup_s vs the horizon", cell=cell)

    from sbx_h.settlement import write_settlement_outputs
    csv_path, md_path = write_settlement_outputs(
        adapter.scheduler.settlement_engines, out_dir, f"{cell}_{arm}")
    print(f"  settlement ledger:  {csv_path}")
    print(f"  settlement summary: {md_path}")

    plotter = runtime.get("live_plotter")
    if plotter is not None:
        png = out_dir / "sbx_mechanism.png"
        plotter.save(png)
        print(f"  figure saved:       {png}")

    # Compact terminal summary: escalations + per-corridor deviations.
    esc = adapter.scheduler.escalations
    print(f"  escalations: {esc if esc else 'none'}")
    for key in sorted(adapter.scheduler.corridors):
        rl = adapter.scheduler.records[key]
        beyond = sum(1 for r in rl if r.beyond_band)
        print(f"  corridor {key}: {len(rl)} cycles, "
              f"{beyond} beyond-band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
