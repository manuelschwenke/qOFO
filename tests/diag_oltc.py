#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_oltc.py
========================
Honest OLTC trajectory audit (Claude Code, 2026-06-23).  rev_rate ignores flat
segments, so it inflates a tap that moves only a few times in alternating
directions.  This script reports, per OLTC tap signal, the ABSOLUTE move count,
up/down counts, direction reversals, net change vs total excursion, and the
timestamps of moves — so "hunting" (many reversals, ~zero net) is distinguished
from "tracking" (mostly one direction, net != 0).  Read-only.
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join("results", "005_cigre")


def load(name):
    with open(os.path.join(OUT_ROOT, name, "log.pkl"), "rb") as f:
        return pickle.load(f)


def tap_series(log, key):
    t, x = [], []
    for r in log:
        d = getattr(r, "dso_trafo_tap_pos", {})
        if key in d and d[key] is not None:
            t.append(getattr(r, "time_s", 0.0)); x.append(int(d[key]))
    return np.asarray(t), np.asarray(x)


def audit(t, x):
    d = np.diff(x)
    moves = np.flatnonzero(d != 0)
    md = d[moves]
    n_up = int(np.sum(md > 0)); n_dn = int(np.sum(md < 0))
    sgn = np.sign(md)
    n_rev = int(np.sum(sgn[1:] != sgn[:-1])) if sgn.size > 1 else 0
    net = int(x[-1] - x[0]); exc = int(np.sum(np.abs(d)))
    move_min = (t[moves + 1] / 60.0).astype(int) if moves.size else np.array([], int)
    return dict(n_moves=int(moves.size), n_up=n_up, n_dn=n_dn, n_rev=n_rev,
                net=net, exc=exc, lo=int(x.min()), hi=int(x.max()),
                move_min=move_min)


def main():
    for name in ["V3", "V4", "V5"]:
        log = load(name)
        keys = sorted({k for r in log
                       for k in getattr(r, "dso_trafo_tap_pos", {}).keys()})
        print(f"\n{'='*78}\n{name}: {len(log)} records, {len(keys)} interface OLTCs"
              f"\n{'='*78}")
        print(f"{'oltc':<20}{'moves':>6}{'up':>4}{'dn':>4}{'revs':>5}"
              f"{'net':>5}{'exc':>5}{'range':>8}   move-times[min]")
        for k in keys:
            t, x = tap_series(log, k)
            if x.size < 2:
                continue
            a = audit(t, x)
            mt = np.array2string(a["move_min"], threshold=20, max_line_width=120)
            print(f"{k:<20}{a['n_moves']:>6}{a['n_up']:>4}{a['n_dn']:>4}"
                  f"{a['n_rev']:>5}{a['net']:>5}{a['exc']:>5}"
                  f"   [{a['lo']},{a['hi']}]   {mt}")
        # full raw trace of the single worst (most reversals) tap
        worst, worst_rev = None, -1
        for k in keys:
            t, x = tap_series(log, k)
            if x.size < 2:
                continue
            a = audit(t, x)
            if a["n_rev"] > worst_rev:
                worst, worst_rev = k, a["n_rev"]
        if worst:
            t, x = tap_series(log, worst)
            print(f"\n  worst-by-reversals: {worst} ({worst_rev} reversals)")
            print("  tap trace:", np.array2string(x, threshold=400, max_line_width=160))


if __name__ == "__main__":
    main()
