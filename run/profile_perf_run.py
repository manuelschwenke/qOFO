#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/profile_perf_run.py
=======================
Minimal profiling harness for the multi-TSO / multi-DSO OFO main loop.

Runs a deliberately short simulation (no profiles, no contingencies, no live
plot) under ``cProfile`` and prints the top-N cumulative-time functions.  Used
to benchmark per-iteration cost before and after the performance fixes in
worktree ``claude/ecstatic-newton``.

Usage
-----
    python run/profile_perf_run.py [--outfile OUTFILE] [--nmin MINUTES]

Author: Manuel Schwenke / Claude Code
Date:   2026-04-08
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from io import StringIO

# ── Ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_M_TSO_M_DSO import MultiTSOConfig, run_multi_tso_dso


def build_profile_config(n_total_min: float = 10.0) -> MultiTSOConfig:
    """Return a lightweight MultiTSOConfig suitable for profiling.

    * No time-series profiles (skip CSV load and profile application).
    * No contingencies (cache stays warm).
    * No live plot, minimal stdout.
    * Short horizon (default 10 minutes).
    """
    return MultiTSOConfig(
        n_total_s=60.0 * n_total_min,
        tso_period_s=60.0 * 1.0,   # TSO every minute (max iterations per min)
        dso_period_s=20.0,         # DSO every 20 s
        alpha={1: 0.01, 2: 0.01, 3: 0.01},
        dso_alpha=0.1,
        g_v=150000.0,
        g_q=5.0,
        dso_g_v=1000.0,
        g_w_der=10.0,
        g_w_gen=1e4,
        g_w_pcc=1.0,
        g_w_tso_oltc=50.0,
        g_w_dso_der=2.0,
        g_w_dso_oltc=10.0,
        use_fixed_zones=True,
        run_stability_analysis=False,
        sensitivity_update_interval=int(1e6),
        verbose=0,
        live_plot=False,
        add_tso_ders=True,
        use_profiles=False,
        use_zonal_gen_dispatch=False,
        contingencies=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nmin",
        type=float,
        default=10.0,
        help="Simulation horizon in minutes (default: 10)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="profile_perf.prof",
        help="pstats output file (default: profile_perf.prof)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top-cumtime rows to print (default: 30)",
    )
    args = parser.parse_args()

    cfg = build_profile_config(n_total_min=args.nmin)
    print(f"[profile] horizon = {args.nmin:.0f} min  "
          f"(TSO every {cfg.tso_period_s:.0f} s, DSO every {cfg.dso_period_s:.0f} s)")

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    log = run_multi_tso_dso(cfg)
    profiler.disable()
    wall = time.perf_counter() - t0

    profiler.dump_stats(args.outfile)

    # Print wall-clock and top functions
    n_steps = len(log)
    per_step_ms = 1000.0 * wall / max(n_steps, 1)
    print(f"[profile] wall = {wall:.3f} s  |  steps = {n_steps}  "
          f"|  {per_step_ms:.2f} ms/step")
    print(f"[profile] pstats saved to {args.outfile}")

    # Top by cumulative time, filtered to project code
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(args.top)
    print(buf.getvalue())

    # Also print top by tottime for hot-spot comparison
    buf2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=buf2).strip_dirs()
    stats2.sort_stats("tottime")
    stats2.print_stats(args.top)
    print(buf2.getvalue())


if __name__ == "__main__":
    main()
