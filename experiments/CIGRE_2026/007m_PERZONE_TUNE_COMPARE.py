#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007m_PERZONE_TUNE_COMPARE.py
===================================================
Best-tuned constant PQ against best-tuned calibrated Thevenin.

Every comparison so far has been confounded: the two boundaries imply different
loop gains, and a single scalar on the step weights narrows but cannot remove a
difference that is per zone (measured V-row gain factors 1.18 / 2.77 / 1.60 for
PQ against 1.05 / 1.05 / 0.97 for Thevenin).  PQ's apparent advantage may
therefore be nothing but an accidental, unevenly distributed gain boost landing
on the weakest area.

This gives BOTH boundaries the same per-zone tuning freedom and compares their
optima.  If Thevenin can match or beat PQ once tuned, the thesis can adopt the
physically argued boundary without paying for it.  If it cannot, PQ is retained
on evidence rather than on an untested confound.

Search
------
One scale factor per zone on that zone's ``params.g_w`` vector -- the step
weight the MIQP divides by, so the direct per-zone loop-gain knob.  Searched by
a single coordinate pass: hold the other zones fixed, sweep one zone over a
short ladder, keep its best, move on.  Coordinate descent is appropriate here
for the same reason the architecture is decentralised at all: cross-area
sensitivities are small (\\cref{ch:architectures:multitso:architecture:tuning}),
so the zones' gains are close to separable.

Not Bayesian optimisation, deliberately.  Three coordinates and a four-point
ladder is a ~10-evaluation problem per boundary; a surrogate model would cost
more to fit than the grid costs to evaluate.

Objective
---------
``tuning.objectives_v2.performance_scalar`` -- the same scalar the shipped
tuning was optimised against (TS voltage RMS and interface-Q equally weighted,
plus worst-bus, band excess, DS voltage and PCC under-utilisation).  Scoring on
``rms_v_ts`` alone is what produced the spurious "32 % headroom" earlier.

Search runs at 2 h for cost; the two winners are then re-run at 5 h so the
verdict is not drawn from the short horizon.

Every evaluation is cached, so an interrupted sweep resumes.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.network_reduction import THEVENIN_K_PER_CORRIDOR  # noqa: E402
from tuning.metrics import extract_metrics  # noqa: E402
from tuning.objectives_v2 import PerfWeights, performance_scalar  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

OUT_DIR = _ROOT / "results" / "007_tie_boundary" / "perzone"
LADDER: Tuple[float, ...] = (1.0, 0.6, 0.35, 0.2)
ZONES: Tuple[int, ...] = (1, 2, 3)


def _cfg(boundary: str, hours: float):
    c = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(c, k, v)
    c.n_total_s = 3600.0 * hours
    c.verbose = 0
    c.run_stability_analysis = False
    c.tie_boundary_equivalent = boundary
    if boundary == "thevenin":
        c.tie_thevenin_k = THEVENIN_K_PER_CORRIDOR
    return c


def _hook_factory(scales: Dict[int, float]):
    def _hook(state: Dict[str, Any]) -> bool:
        for z, ctrl in state.get("tso_controllers", {}).items():
            s = scales.get(int(z), 1.0)
            if s == 1.0:
                continue
            gw = getattr(getattr(ctrl, "params", None), "g_w", None)
            if gw is None:
                continue
            scaled = (np.asarray(gw, dtype=float) * float(s)
                      if np.ndim(gw) else float(gw) * float(s))
            ctrl.params = dataclasses.replace(ctrl.params, g_w=scaled)
        return False  # continue into the main loop
    return _hook


def _tag(boundary: str, scales: Dict[int, float], hours: float) -> str:
    s = "_".join(f"{scales.get(z, 1.0):g}" for z in ZONES)
    return f"{boundary}_{s}_{hours:g}h"


def _evaluate(boundary: str, scales: Dict[int, float], hours: float,
              cache: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Closed-loop run + BO performance scalar, cached on disk."""
    tag = _tag(boundary, scales, hours)
    pkl = OUT_DIR / f"{tag}.pkl"
    if tag in cache:
        return cache[tag], {}
    if pkl.exists():
        with open(pkl, "rb") as fh:
            recs = pickle.load(fh)
    else:
        t0 = time.time()
        recs = run_multi_tso_dso(_cfg(boundary, hours),
                                 pre_loop_hook=_hook_factory(scales))
        print(f"      ran in {time.time() - t0:.0f} s", flush=True)
        if recs:
            with open(pkl, "wb") as fh:
                pickle.dump(recs, fh)
    if not recs:
        cache[tag] = float("inf")
        return float("inf"), {}
    m = extract_metrics(recs, _cfg(boundary, hours))
    total, parts = performance_scalar(m, PerfWeights())
    cache[tag] = float(total)
    return float(total), parts


def _search(boundary: str, hours: float,
            cache: Dict[str, float]) -> Tuple[Dict[int, float], float]:
    """One coordinate pass over the per-zone scales."""
    scales: Dict[int, float] = {z: 1.0 for z in ZONES}
    best, _ = _evaluate(boundary, scales, hours, cache)
    print(f"  [{boundary}] baseline (all 1.0): {best:.4f}")

    for z in ZONES:
        z_best, z_val = scales[z], best
        for s in LADDER:
            if s == scales[z]:
                continue
            trial = dict(scales)
            trial[z] = s
            print(f"  [{boundary}] zone {z} scale {s:g} ...", flush=True)
            v, _ = _evaluate(boundary, trial, hours, cache)
            print(f"      -> {v:.4f}")
            if v < z_val:
                z_best, z_val = s, v
        scales[z] = z_best
        best = z_val
        print(f"  [{boundary}] zone {z} fixed at {z_best:g}  (best {best:.4f})")
    return scales, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-hours", type=float, default=2.0)
    ap.add_argument("--verify-hours", type=float, default=5.0)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("BEST-TUNED PQ vs BEST-TUNED THEVENIN -- per-zone step-weight scales")
    print(f"  ladder {LADDER}, zones {ZONES}, "
          f"search at {args.search_hours:g} h")
    print("  objective: the BO's own performance scalar (lower is better)")
    print("=" * 84)

    cache: Dict[str, float] = {}
    found: Dict[str, Tuple[Dict[int, float], float]] = {}
    for boundary in ("pq", "thevenin"):
        print(f"\n--- {boundary} ---")
        found[boundary] = _search(boundary, args.search_hours, cache)

    print("\n" + "=" * 84)
    print(f"SEARCH RESULT ({args.search_hours:g} h)")
    print("=" * 84)
    for b, (sc, v) in found.items():
        print(f"  {b:>9}: scales {[sc[z] for z in ZONES]}  ->  {v:.4f}")
    a, c = found["pq"][1], found["thevenin"][1]
    if np.isfinite(a) and a > 0:
        print(f"\n  Thevenin vs PQ at each one's own optimum: "
              f"{(c - a) / a * 100:+.1f} %")

    if not args.skip_verify:
        print("\n" + "=" * 84)
        print(f"VERIFICATION AT {args.verify_hours:g} h")
        print("=" * 84)
        vr: Dict[str, Tuple[float, Dict[str, float]]] = {}
        for b, (sc, _) in found.items():
            print(f"  [{b}] scales {[sc[z] for z in ZONES]} ...", flush=True)
            vr[b] = _evaluate(b, sc, args.verify_hours, cache)
            print(f"      -> {vr[b][0]:.4f}")
        keys = ["v_rms_ts", "v_rms_ds", "v_worst_ts", "v_band_ts", "q_pcc",
                "pcc_underutil"]
        hdr = f"{'boundary':>9} {'TOTAL':>8} " + " ".join(f"{k:>13}" for k in keys)
        print("\n" + hdr)
        print("-" * len(hdr))
        for b, (tot, parts) in vr.items():
            print(f"{b:>9} {tot:>8.3f} " +
                  " ".join(f"{parts.get(k, float('nan')):>13.3f}" for k in keys))
        a5, c5 = vr["pq"][0], vr["thevenin"][0]
        if np.isfinite(a5) and a5 > 0:
            d = (c5 - a5) / a5 * 100
            print(f"\n  Thevenin vs PQ, both at their own optimum: {d:+.1f} %")
            if d <= 2.0:
                print("  -> Thevenin MATCHES or BEATS PQ once tuned: the "
                      "physically argued boundary is affordable, and PQ's\n"
                      "     earlier advantage was the untuned gain difference.")
            else:
                print("  -> PQ still ahead after per-zone tuning: its "
                      "advantage is NOT merely an accidental gain boost.")

    with open(OUT_DIR / "search.csv", "w", encoding="utf-8") as fh:
        fh.write("tag,score\n")
        for k, v in sorted(cache.items()):
            fh.write(f"{k},{v}\n")
    print(f"\nwritten: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
