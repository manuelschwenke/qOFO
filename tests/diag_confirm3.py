#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_confirm3.py
============================
Decide the V3 e_v sawtooth mechanism (Claude Code, 2026-06-24). If it is
inter-sample drift of the slow TSO loop (sample-and-hold against a between-step
disturbance), the tooth peak-to-trough should scale ~linearly with the TSO
update period. Re-run V3 at tso_period_s in {360,120,60} s and measure the
teeth in [60,120] min. DSO stays local (one-sided) — unchanged from V3.
Writes results/_diag_confirm/.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tests.diag_confirm import run
from tests.diag_voltage import ev_series, teeth

RUNS = [
    ("V3_tso360s", "V3", {"tso_period_s": 360.0}),
    ("V3_tso120s", "V3", {"tso_period_s": 120.0}),
    ("V3_tso60s",  "V3", {"tso_period_s": 60.0}),
]


def main():
    rows = []
    for tag, base, extra in RUNS:
        print("\n" + "=" * 72 + f"\n  RUN {tag} (extra {extra})\n" + "=" * 72,
              flush=True)
        log = run(tag, base, extra)
        t, e, a = ev_series(log)
        th = teeth(t, e, 60, 120)
        n_tso = int(a.sum())
        rows.append((tag, extra["tso_period_s"], th.get("n_peaks", 0),
                     th.get("mean_spacing_min", float("nan")),
                     th.get("mean_p2t", float("nan")), n_tso))

    print("\n" + "#" * 72 + "\n  CADENCE TEST: V3 e_v teeth vs TSO period\n" + "#" * 72)
    print(f"{'run':<14}{'Tso[s]':>8}{'peaks':>7}{'spacing[min]':>14}"
          f"{'p2t[mp.u.]':>12}{'#TSOsteps':>11}")
    for tag, T, npk, sp, p2t, nt in rows:
        print(f"{tag:<14}{T:>8.0f}{npk:>7}{sp:>14.1f}{p2t:>12.2f}{nt:>11}")
    print("\nPrediction: p2t falls ~linearly with Tso if it is inter-sample drift.")


if __name__ == "__main__":
    main()
