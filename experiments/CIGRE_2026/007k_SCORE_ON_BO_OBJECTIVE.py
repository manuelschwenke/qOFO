#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007k_SCORE_ON_BO_OBJECTIVE.py
====================================================
Is the "32 % tuning headroom" real, or did I just slide along the tuner's own
indifference curve?

007h swept a gain multiplier ``kappa`` and found TS voltage RMS improving 32 %
from kappa 1.0 to 0.2.  But that is ONE term.  The BO that produced the shipped
tuning optimises a scalar in which TS voltage RMS and interface-Q carry EQUAL
weight (``PerfWeights``: ``w_v_rms_ts = 1.0``, ``w_q_pcc = 1.0``, plus
worst-bus 0.5, band excess 1.0, DS voltage 0.3, PCC under-utilisation 0.3), and
over the same sweep interface-Q got ~90 % WORSE.

So the honest question is not "does rms_v_ts improve" but "does the BO's own
performance scalar improve".  If it does not, the shipped tuning is at or near
its intended optimum and there is no headroom -- only a different point on a
trade-off the tuner already considered and rejected.  If it does, the shipped
point is genuinely off its own optimum and that is worth knowing.

This scores every 007h frontier arm with the production
``tuning.metrics.extract_metrics`` + ``tuning.objectives_v2.performance_scalar``
and reports the full breakdown, so the trade can be read term by term rather
than inferred from two headline numbers.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuning.metrics import extract_metrics  # noqa: E402
from tuning.objectives_v2 import PerfWeights, performance_scalar  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

FRONTIER_DIR = _ROOT / "results" / "007_tie_boundary" / "frontier"
KAPPAS = (1.0, 0.5, 0.3, 0.2)


def _cfg():
    c = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(c, k, v)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0)
    args = ap.parse_args()

    cfg = _cfg()
    w = PerfWeights()
    print("=" * 96)
    print("FRONTIER ARMS SCORED ON THE BO'S OWN PERFORMANCE SCALAR")
    print("=" * 96)
    print(f"  weights: v_rms_ts {w.w_v_rms_ts}  v_rms_ds {w.w_v_rms_ds}  "
          f"v_worst_ts {w.w_v_worst_ts}  v_band_ts {w.w_v_band_ts}  "
          f"q_pcc {w.w_q_pcc}  pcc_underutil {w.w_pcc_underutil}")
    print("  (switching is NOT in this scalar -- it lives in the constraint "
          "vector)")

    rows: List[Dict[str, Any]] = []
    for b in ("pq", "th"):
        for k in KAPPAS:
            arm = f"{b}_k{k:g}"
            p = FRONTIER_DIR / f"{arm}_{args.hours:g}h.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as fh:
                recs = pickle.load(fh)
            try:
                m = extract_metrics(recs, cfg)
                total, parts = performance_scalar(m, w)
            except Exception as exc:
                print(f"  [{arm}] scoring failed "
                      f"({type(exc).__name__}: {exc})")
                continue
            rows.append(dict(arm=arm, boundary=b, kappa=k, total=total,
                             feasible=bool(getattr(m, "feasible", True)),
                             reason=getattr(m, "infeasible_reason", None),
                             **parts))

    if not rows:
        print("\nno arms scored")
        return 1

    keys = ["v_rms_ts", "v_rms_ds", "v_worst_ts", "v_band_ts", "q_pcc",
            "pcc_underutil"]
    hdr = (f"{'arm':>9} {'feas':>5} {'TOTAL':>9} " +
           " ".join(f"{k:>13}" for k in keys))
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>9} {str(r['feasible'])[:5]:>5} {r['total']:>9.3f} " +
              " ".join(f"{r.get(k, float('nan')):>13.3f}" for k in keys))

    print("\nRelative to the shipped tuning (kappa = 1.0) on its own boundary:")
    hdr2 = f"{'arm':>9} {'TOTAL vs k=1':>14} {'v_rms_ts':>12} {'q_pcc':>12}"
    print(hdr2)
    print("-" * len(hdr2))
    for b in ("pq", "th"):
        base = next((r for r in rows if r["boundary"] == b and r["kappa"] == 1.0),
                    None)
        if base is None:
            continue
        for k in KAPPAS:
            r = next((x for x in rows if x["boundary"] == b and x["kappa"] == k),
                     None)
            if r is None:
                continue
            def rel(key):
                a, c = r.get(key, np.nan), base.get(key, np.nan)
                return (a - c) / c * 100.0 if np.isfinite(c) and c != 0 else np.nan
            print(f"{r['arm']:>9} {rel('total'):>13.1f}% "
                  f"{rel('v_rms_ts'):>11.1f}% {rel('q_pcc'):>11.1f}%")

    print("\nVERDICT")
    for b, label in (("pq", "PQ"), ("th", "Thevenin")):
        sub = [r for r in rows if r["boundary"] == b and np.isfinite(r["total"])]
        if not sub:
            continue
        best = min(sub, key=lambda r: r["total"])
        base = next((r for r in sub if r["kappa"] == 1.0), None)
        if base is None:
            continue
        gain = (base["total"] - best["total"]) / base["total"] * 100.0
        if best["kappa"] == 1.0:
            print(f"  {label:>8}: shipped kappa=1.0 IS the best of the ladder "
                  f"-> no headroom on the BO's own objective; the rms_v_ts "
                  f"improvement was bought at a price the tuner counts.")
        else:
            print(f"  {label:>8}: best is kappa={best['kappa']:g}, "
                  f"{gain:.1f} % better than shipped on the BO scalar "
                  f"-> genuine headroom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
