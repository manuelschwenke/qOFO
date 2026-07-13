#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_confirm.py
===========================
Confirmatory runs for the zigzag investigation (Claude Code, 2026-06-23).

A) T' lever:   V3 with apply_qv_h_transform False vs True.
               Prediction: T' does NOT reduce the continuous churn (and may
               slightly worsen the in-deadband DER channel).
B) g_w lever:  V4 with g_w_pcc x1 vs x10 (stiffer PCC step = more damping).
               Prediction: raising g_w on the tie-driving columns reduces the
               inter-zone tie-flow oscillation -> confirms loop-gain, not T'.

Runs at dt_s=60 (matches the existing 005 pickles; 3x fewer steps than the
current dt_s=20 default) over the same 300-min horizon + contingency schedule.
Writes results/_diag_confirm/<tag>/log.pkl and prints per-family oscillation
medians (tie / interface-Q / DSO-DER).  Does NOT touch results/005_cigre.
"""
from __future__ import annotations
import os, sys, time, pickle, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# 005_CIGRE_MULTI has a digit-leading name -> load by path.
_spec = importlib.util.spec_from_file_location(
    "cigre005", os.path.join(os.path.dirname(HERE), "experiments", "005_CIGRE_MULTI.py"))
c5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c5)

from experiments.runners import run_multi_tso_dso
from tests.diag_zigzag import collect_series, zigzag_stats

OUT = os.path.join("results", "_diag_confirm")

RUNS = [
    ("V3_tprimeOFF",   "V3", {"apply_qv_h_transform": False}),
    ("V3_tprimeON",    "V3", {"apply_qv_h_transform": True}),
    ("V4_gw_pcc_x1",   "V4", {}),
    ("V4_gw_pcc_x10",  "V4", {"g_w_pcc": 2000.0}),
]


def run(tag, base, extra):
    cfg = c5.make_cigre_config()
    for k, v in c5.VARIANTS[base].items():
        setattr(cfg, k, v)
    cfg.dt_s = 60
    cfg.dso_period_s = 60.0
    cfg.verbose = 0
    for k, v in extra.items():
        setattr(cfg, k, v)
    d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
    cfg.result_dir = d
    t0 = time.time()
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] FAILED {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        log = []
    with open(os.path.join(d, "log.pkl"), "wb") as f:
        pickle.dump(log, f)
    print(f"[{tag}] {len(log)} records in {time.time()-t0:.0f}s", flush=True)
    return log


def summarize(tag, log):
    if not log:
        print(f"  [{tag}] EMPTY"); return {}
    s = collect_series(log)
    fams = {"tieQ": [], "ifaceQact": [], "dsoQder": [], "zoneQpcc": []}
    for k, x in s.items():
        for f in fams:
            if k.startswith(f):
                st = zigzag_stats(x)
                fams[f].append((st["rev_rate"], st["tv_ratio"], st["step_rms"]))
    out = {}
    for f, v in fams.items():
        if v:
            a = np.array(v, float)
            out[f] = (np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1]),
                      np.nanmedian(a[:, 2]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for tag, base, extra in RUNS:
        print("\n" + "=" * 72 + f"\n  RUN {tag}  (base {base}, extra {extra})\n"
              + "=" * 72, flush=True)
        log = run(tag, base, extra)
        results[tag] = summarize(tag, log)

    print("\n" + "#" * 72)
    print("  CONFIRMATORY SUMMARY  (median over each signal family)")
    print("#" * 72)
    hdr = f"{'run':<16}{'family':<12}{'rev_rate':>9}{'tv_ratio':>9}{'step_rms':>10}"
    print(hdr)
    for tag in results:
        for fam, (rr, tv, sr) in results[tag].items():
            print(f"{tag:<16}{fam:<12}{rr:>9.2f}{tv:>9.1f}{sr:>10.3g}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
