#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007p_PERZONE_REFINE.py
=============================================
Closes the two remaining asymmetries in the best-tuned PQ vs Thevenin
comparison, both of which favour PQ:

1.  Thevenin's best (1.806, 007o) is a UNIFORM scan; PQ's (1.672, 007m) is a
    per-zone search.  Thevenin gets a per-zone pass around 0.1.
2.  PQ's own optimum has zone 3 sitting ON the 007m ladder floor of 0.2, so its
    search was truncated too.  Zone 3 is probed below that floor.

Nine 5 h runs.  Artefacts go to a local directory (``--out``): two earlier runs
were lost to file-server outages mid-simulation.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-13
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.helpers.run_params import dump_params  # noqa: E402
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.network_reduction import THEVENIN_K_PER_CORRIDOR  # noqa: E402
from tuning.metrics import extract_metrics  # noqa: E402
from tuning.objectives_v2 import PerfWeights, performance_scalar  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py")))
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

_DEFAULT_OUT = os.environ.get(
    "QOFO_LOCAL_RESULTS",
    str(Path.home() / "AppData" / "Local" / "qOFO_runs" / "007_refine"))
HOURS = 5.0
ZONES = (1, 2, 3)

#: (start point, per-zone candidates, which zones to refine).
PLAN = {
    "thevenin": ({1: 0.1, 2: 0.1, 3: 0.1}, (0.15, 0.07), ZONES),
    "pq":       ({1: 0.35, 2: 0.35, 3: 0.2}, (0.1,), (3,)),
}


def _cfg(boundary: str):
    c = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(c, k, v)
    c.n_total_s = 3600.0 * HOURS
    c.verbose = 0
    c.run_stability_analysis = False
    c.tie_boundary_equivalent = boundary
    if boundary == "thevenin":
        c.tie_thevenin_k = THEVENIN_K_PER_CORRIDOR
    return c


def _hook(scales: Dict[int, float]):
    def _h(state: Dict[str, Any]) -> bool:
        for z, ctrl in state.get("tso_controllers", {}).items():
            s = float(scales.get(int(z), 1.0))
            gw = getattr(getattr(ctrl, "params", None), "g_w", None)
            if s != 1.0 and gw is not None:
                ctrl.params = dataclasses.replace(
                    ctrl.params, g_w=np.asarray(gw, dtype=float) * s)
        return False
    return _h


def _eval(boundary, scales, out: Path, cache) -> Tuple[float, dict]:
    tag = f"{boundary}_" + "_".join(f"{scales[z]:g}" for z in ZONES)
    if tag in cache:
        return cache[tag]
    pkl = out / f"{tag}_{HOURS:g}h.pkl"
    if pkl.exists():
        recs = pickle.load(open(pkl, "rb"))
    else:
        cfg = _cfg(boundary)
        # Snapshot resolved parameters before running: a later edit to
        # the config factory must not rewrite this result's history.
        dump_params(out / f"{tag}_{HOURS:g}h.params.json", cfg,
                    extra={"tag": tag, "boundary": boundary,
                           "zone_g_w_scale": scales})
        t0 = time.time()
        recs = run_multi_tso_dso(cfg, pre_loop_hook=_hook(scales))
        print(f"      ran in {time.time() - t0:.0f} s", flush=True)
        if recs:
            pickle.dump(recs, open(pkl, "wb"))
    if not recs:
        cache[tag] = (float("inf"), {})
    else:
        cache[tag] = performance_scalar(
            extract_metrics(recs, _cfg(boundary)), PerfWeights())
    return cache[tag]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_DEFAULT_OUT)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"PER-ZONE REFINEMENT -- {HOURS:g} h, BO performance scalar")
    print(f"  artefacts -> {out}")
    print("=" * 80)

    cache: Dict[str, Tuple[float, dict]] = {}
    res = {}
    for b, (start, ladder, zones) in PLAN.items():
        print(f"\n--- {b} ---")
        scales = dict(start)
        best, parts = _eval(b, scales, out, cache)
        print(f"  start {[scales[z] for z in ZONES]}: {best:.4f}")
        for z in zones:
            for s in ladder:
                if s == scales[z]:
                    continue
                trial = dict(scales)
                trial[z] = s
                print(f"  zone {z} -> {s:g} ...", flush=True)
                v, p = _eval(b, trial, out, cache)
                print(f"      {v:.4f}")
                if v < best:
                    best, parts, scales = v, p, trial
            print(f"  zone {z} fixed at {scales[z]:g}  (best {best:.4f})")
        res[b] = (scales, best, parts)

    keys = ["v_rms_ts", "v_worst_ts", "v_band_ts", "q_pcc"]
    hdr = (f"{'boundary':>9} {'scales':>18} {'TOTAL':>8} " +
           " ".join(f"{k:>12}" for k in keys))
    print("\n" + "=" * 80 + f"\nFINAL\n" + "=" * 80 + f"\n{hdr}\n" + "-" * len(hdr))
    for b in ("pq", "thevenin"):
        sc, tot, p = res[b]
        print(f"{b:>9} {str([sc[z] for z in ZONES]):>18} {tot:>8.3f} " +
              " ".join(f"{p.get(k, float('nan')):>12.3f}" for k in keys))

    a, c = res["pq"][1], res["thevenin"][1]
    d = (c - a) / a * 100.0
    print(f"\nThevenin vs PQ, each at its refined optimum: {d:+.1f} %")
    if abs(d) <= 3.0:
        print("-> indistinguishable; the boundary choice is not a "
              "control-performance question.")
    elif d < 0:
        print("-> the calibrated boundary is AHEAD; the deficit was search "
              "truncation throughout.")
    else:
        print(f"-> PQ keeps a {d:.1f} % edge with both searches refined.")

    with open(out / "refine.csv", "w", encoding="utf-8") as fh:
        fh.write("tag,score\n")
        for k, (v, _) in sorted(cache.items()):
            fh.write(f"{k},{v}\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
