#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_zigzag2.py
===========================
Follow-up (Claude Code, 2026-06-23): time-resolve the V3/V4 oscillation.
Windowed reversal-rate + step-rms vs simulation time, with the 005
contingency schedule overlaid, to distinguish:
  * intrinsic chatter (present from t=0, before any contingency), vs
  * stale-frozen-H signature (amplitude grows after a contingency).
Also prints the head of the V3 worst-OLTC trace to confirm period-2.
Read-only on the logs.
"""
from __future__ import annotations
import os, sys, pickle
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join("results", "005_cigre")

# 005 contingency schedule (minutes); window <= n_total = 300 min.
CONTINGENCIES = [(60, "gen2 TRIP"), (120, "load@b11 +300/150"),
                 (180, "gen2 RESTORE"), (260, "line25 TRIP")]


def load(name):
    with open(os.path.join(OUT_ROOT, name, "log.pkl"), "rb") as f:
        return pickle.load(f)


def series_of(log, getter):
    t, x = [], []
    for r in log:
        v = getter(r)
        if v is not None and np.isfinite(v):
            t.append(getattr(r, "time_s", 0.0)); x.append(float(v))
    return np.asarray(t), np.asarray(x)


def windowed(t, x, win=30):
    out = []
    for a in range(0, len(x) - win, win):
        seg = x[a:a + win]
        d = np.diff(seg)
        nz = d[np.abs(d) > 1e-12]
        rr = float(np.mean(np.sign(nz[1:]) != np.sign(nz[:-1]))) if nz.size > 1 else 0.0
        out.append((t[a] / 60.0, rr, float(np.sqrt(np.mean(d**2)))))
    return out


def pick(attr, key):
    return lambda r: getattr(r, attr, {}).get(key, None)


def main():
    print("Contingencies (min):", ", ".join(f"{m}:{lbl}" for m, lbl in CONTINGENCIES))

    specs = {
        "V3": [("oltcTap_DSO_3|t8", pick_tap("DSO_3|trafo_8")),
               ("ifaceQact_DSO_3|t6", pick("dso_trafo_q_actual_mvar", "DSO_3|trafo_6")),
               ("tieQ_1_3", pick("zone_tie_q_mvar", (1, 3)))],
        "V4": [("tieQ_2_3", pick("zone_tie_q_mvar", (2, 3))),
               ("ifaceQact_DSO_2|t3", pick("dso_trafo_q_actual_mvar", "DSO_2|trafo_3")),
               ("dsoQder_DSO_2", pick("dso_group_q_der_mvar", "DSO_2"))],
    }
    for name, sigs in specs.items():
        log = load(name)
        print(f"\n{'='*70}\n{name}: {len(log)} records\n{'='*70}")
        for label, getter in sigs:
            t, x = series_of(log, getter)
            if x.size < 40:
                print(f"  {label}: too short ({x.size})"); continue
            w = windowed(t, x, win=30)
            print(f"\n  -- {label} (n={x.size}, range={x.max()-x.min():.2f}) --")
            print(f"     {'t0_min':>7}{'rev_rate':>10}{'step_rms':>10}")
            for t0, rr, sr in w:
                flag = ""
                for m, _ in CONTINGENCIES:
                    if t0 <= m < t0 + 30 * np.median(np.diff(t)) / 60.0:
                        flag = "  <-- contingency window"
                print(f"     {t0:>7.0f}{rr:>10.2f}{sr:>10.3g}{flag}")

    # Confirm V3 OLTC period-2: print head of the raw trace.
    log = load("V3")
    t, x = series_of(log, pick_tap("DSO_3|trafo_8"))
    print(f"\n{'='*70}\nV3 oltcTap_DSO_3|trafo_8 raw head (confirm period-2)\n{'='*70}")
    print("  t_min:", np.array2string((t[:40] / 60.0).astype(int)))
    print("  tap  :", np.array2string(x[:40].astype(int)))


def pick_tap(key):
    return lambda r: (float(getattr(r, "dso_trafo_tap_pos", {}).get(key))
                      if key in getattr(r, "dso_trafo_tap_pos", {}) else None)


if __name__ == "__main__":
    main()
