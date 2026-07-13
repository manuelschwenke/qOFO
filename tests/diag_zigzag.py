#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_zigzag.py
==========================
One-off diagnostic (Claude Code, 2026-06-23): quantify the step-to-step
oscillation ("zigzag") in the 005_cigre V3/V4/V5 logs and localise which
signal oscillates and at what cadence.  Read-only on the logs; writes a
text report + per-signal CSV to results/005_cigre/_zigzag/.

Zigzag metric per scalar series x[t]:
  d            = diff(x)
  reversals    = #{ t : sign(d[t]) != sign(d[t-1]), both != 0 }
  rev_rate     = reversals / (len(d)-1)          # ~1.0 => alternates every step
  tv_ratio     = sum|d| / (max(x)-min(x) + eps)  # >>1 => lots of back-and-forth
  step_rms     = rms(d)
A clean ramp/settle has rev_rate ~ 0 and tv_ratio ~ 1.  Chatter has
rev_rate -> ~1 (or 0.5 for period-2 with rests) and tv_ratio >> 1.
"""
from __future__ import annotations

import os
import sys
import pickle
import csv
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_ROOT = os.path.join("results", "005_cigre")
DIAG_DIR = os.path.join(OUT_ROOT, "_zigzag")
VARIANTS = ["V3", "V4", "V5"]


def load(name: str) -> List[Any]:
    pkl = os.path.join(OUT_ROOT, name, "log.pkl")
    if not os.path.isfile(pkl):
        print(f"  [load] missing {pkl}")
        return []
    with open(pkl, "rb") as f:
        return pickle.load(f)


def zigzag_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return dict(n=x.size, rev_rate=np.nan, tv_ratio=np.nan,
                    step_rms=np.nan, rng=np.nan, step_max=np.nan)
    d = np.diff(x)
    nz = d[np.abs(d) > 1e-12]
    if nz.size < 2:
        rev_rate = 0.0
    else:
        s = np.sign(nz)
        rev_rate = float(np.mean(s[1:] != s[:-1]))
    rng = float(np.max(x) - np.min(x))
    tv = float(np.sum(np.abs(d)))
    tv_ratio = tv / (rng + 1e-9)
    return dict(n=int(x.size), rev_rate=rev_rate, tv_ratio=tv_ratio,
                step_rms=float(np.sqrt(np.mean(d**2))), rng=rng,
                step_max=float(np.max(np.abs(d))))


def collect_series(log: List[Any]) -> Dict[str, np.ndarray]:
    """Pull every interesting scalar/vector signal into named 1-D series."""
    series: Dict[str, List[float]] = {}

    def push(key: str, val: float):
        series.setdefault(key, []).append(float(val))

    for rec in log:
        # --- TSO PCC Q dispatch (command) per zone ---
        for z, arr in getattr(rec, "zone_q_pcc_set", {}).items():
            a = np.asarray(arr, dtype=float).ravel()
            for k in range(a.size):
                push(f"zoneQpcc_z{z}_{k}", a[k])
        # --- TSO DER command per zone (sum) ---
        for z, arr in getattr(rec, "zone_q_der", {}).items():
            a = np.asarray(arr, dtype=float).ravel()
            push(f"zoneQder_sum_z{z}", float(np.nansum(a)))
        # --- per-zone voltage tracking ---
        for z, v in getattr(rec, "zone_v_rms_err_pu", {}).items():
            push(f"zoneVrms_z{z}", v)
        for z, v in getattr(rec, "zone_v_max", {}).items():
            push(f"zoneVmax_z{z}", v)
        for z, v in getattr(rec, "zone_v_min", {}).items():
            push(f"zoneVmin_z{z}", v)
        # --- DSO interface Q (actual + set) per dso ---
        for d, v in getattr(rec, "dso_trafo_q_actual_mvar", {}).items():
            push(f"ifaceQact_{d}", v)
        for d, v in getattr(rec, "dso_trafo_q_set_mvar", {}).items():
            push(f"ifaceQset_{d}", v)
        # --- DSO DER reactive (group sum) ---
        for d, v in getattr(rec, "dso_group_q_der_mvar", {}).items():
            push(f"dsoQder_{d}", v)
        # --- DSO HV voltages ---
        for d, v in getattr(rec, "dso_group_v_max_pu", {}).items():
            push(f"dsoVmax_{d}", v)
        for d, v in getattr(rec, "dso_group_v_min_pu", {}).items():
            push(f"dsoVmin_{d}", v)
        # --- tie flows ---
        for pair, v in getattr(rec, "zone_tie_q_mvar", {}).items():
            push(f"tieQ_{pair[0]}_{pair[1]}", v)
        # --- OLTC taps (DSO interface) ---
        for d, v in getattr(rec, "dso_trafo_tap_pos", {}).items():
            push(f"oltcTap_{d}", v)

    return {k: np.asarray(v, dtype=float) for k, v in series.items()}


def main() -> None:
    os.makedirs(DIAG_DIR, exist_ok=True)
    report: List[str] = []
    for name in VARIANTS:
        log = load(name)
        if not log:
            report.append(f"\n=== {name}: EMPTY LOG ===")
            continue
        n = len(log)
        t = np.asarray([getattr(r, "time_s", i) for i, r in enumerate(log)])
        dt = float(np.median(np.diff(t))) if n > 1 else 0.0
        series = collect_series(log)
        rows = []
        for key, x in series.items():
            st = zigzag_stats(x)
            rows.append((key, st))
        # rank by a chatter score = rev_rate * log1p(tv_ratio)
        def score(st):
            rr = st["rev_rate"]
            tv = st["tv_ratio"]
            if not np.isfinite(rr) or not np.isfinite(tv):
                return -1.0
            return rr * np.log1p(max(tv - 1.0, 0.0)) * st["step_rms"]
        rows.sort(key=lambda r: score(r[1]), reverse=True)

        report.append(f"\n=== {name}: {n} records, dt={dt:.0f}s, "
                      f"{len(series)} signals ===")
        report.append(f"{'signal':<22}{'n':>5}{'rev_rate':>9}{'tv_ratio':>9}"
                      f"{'step_rms':>10}{'step_max':>10}{'range':>10}")
        for key, st in rows[:25]:
            report.append(f"{key:<22}{st['n']:>5}{st['rev_rate']:>9.2f}"
                          f"{st['tv_ratio']:>9.1f}{st['step_rms']:>10.3g}"
                          f"{st['step_max']:>10.3g}{st['rng']:>10.3g}")
        # dump full CSV
        csv_path = os.path.join(DIAG_DIR, f"{name}_zigzag.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["signal", "n", "rev_rate", "tv_ratio",
                        "step_rms", "step_max", "range"])
            for key, st in rows:
                w.writerow([key, st["n"], f"{st['rev_rate']:.4f}",
                            f"{st['tv_ratio']:.4f}", f"{st['step_rms']:.6g}",
                            f"{st['step_max']:.6g}", f"{st['rng']:.6g}"])
        # dump the worst signal's raw trace so cadence is inspectable
        if rows:
            worst_key = rows[0][0]
            np.savetxt(os.path.join(DIAG_DIR, f"{name}_worst_{worst_key}.csv"),
                       np.column_stack([t[:series[worst_key].size],
                                        series[worst_key]]),
                       delimiter=",", header=f"time_s,{worst_key}", comments="")

    txt = "\n".join(report)
    print(txt)
    with open(os.path.join(DIAG_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\n[done] wrote {DIAG_DIR}/report.txt + per-variant CSVs")


if __name__ == "__main__":
    main()
