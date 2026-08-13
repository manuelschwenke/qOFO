#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007o_LADDER_EXTEND.py
============================================
Was the Thevenin arm's "optimum" actually an optimum?

007m searched per-zone step-weight scales over the ladder (1.0, 0.6, 0.35, 0.2)
and reported best-tuned PQ at (0.35, 0.35, 0.2) against best-tuned Thevenin at
(0.2, 0.2, 0.2).  PQ's optimum is interior in two of three zones; **Thevenin's
sits on the edge of the ladder in all three**.  An optimum on the boundary of
the search box is not an optimum -- the arm wanted more gain than the box
allowed, so the reported +29 % is an upper bound on its deficit, not a
measurement of it.

This extends the ladder downward for the Thevenin arm only (it is the only one
that hit the edge) and re-scores at 5 h against the unchanged PQ reference.

Run artefacts are written OUTSIDE the project by default.  Two runs have already
been lost to file-server outages mid-simulation -- a blocking write to an
unreachable share wedges the process for hours -- so the pickles go to a local
directory given by ``--out`` (or ``$QOFO_LOCAL_RESULTS``).  They are
intermediate; the result worth keeping is the printed table.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
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
from typing import Any, Dict

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    str(Path.home() / "AppData" / "Local" / "qOFO_runs" / "007_extend"),
)
HOURS = 5.0
#: Uniform per-zone scales below the 007m ladder floor of 0.2.
EXTENSION = (0.1, 0.05, 0.025)
PQ_REFERENCE = 1.6715  # 007m, scales (0.35, 0.35, 0.2), 5 h


def _cfg():
    c = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(c, k, v)
    c.n_total_s = 3600.0 * HOURS
    c.verbose = 0
    c.run_stability_analysis = False
    c.tie_boundary_equivalent = "thevenin"
    c.tie_thevenin_k = THEVENIN_K_PER_CORRIDOR
    return c


def _hook(s: float):
    def _h(state: Dict[str, Any]) -> bool:
        for _z, ctrl in state.get("tso_controllers", {}).items():
            gw = getattr(getattr(ctrl, "params", None), "g_w", None)
            if gw is None:
                continue
            ctrl.params = dataclasses.replace(
                ctrl.params, g_w=np.asarray(gw, dtype=float) * float(s))
        return False
    return _h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_DEFAULT_OUT,
                    help="local directory for run pickles (keep OFF the share)")
    args = ap.parse_args()
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("LADDER EXTENSION -- Thevenin below the 007m floor of 0.2")
    print(f"  PQ reference (its own interior optimum): {PQ_REFERENCE:.4f}")
    print(f"  artefacts -> {OUT_DIR}")
    print("=" * 80)

    rows = []
    for s in EXTENSION:
        pkl = OUT_DIR / f"thevenin_s{s:g}_{HOURS:g}h.pkl"
        print(f"\n[thevenin] uniform scale {s:g} ...", flush=True)
        if pkl.exists():
            recs = pickle.load(open(pkl, "rb"))
        else:
            t0 = time.time()
            try:
                recs = run_multi_tso_dso(_cfg(), pre_loop_hook=_hook(s))
            except Exception as exc:
                print(f"    FAILED ({type(exc).__name__}: {exc})")
                continue
            print(f"    ran in {time.time() - t0:.0f} s", flush=True)
            if recs:
                with open(pkl, "wb") as fh:
                    pickle.dump(recs, fh)
        if not recs:
            print("    no records")
            continue
        m = extract_metrics(recs, _cfg())
        tot, parts = performance_scalar(m, PerfWeights())
        rows.append((s, tot, parts))
        print(f"    -> {tot:.4f}   ({(tot - PQ_REFERENCE) / PQ_REFERENCE * 100:+.1f} % vs PQ)")

    if not rows:
        return 1
    keys = ["v_rms_ts", "v_worst_ts", "v_band_ts", "q_pcc"]
    hdr = f"{'scale':>7} {'TOTAL':>8} {'vs PQ':>8} " + " ".join(f"{k:>12}" for k in keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    print(f"{0.2:>7.3f} {2.1581:>8.3f} {'+29.1%':>8} " +
          " ".join(f"{v:>12.3f}" for v in (0.308, 0.271, 0.149, 1.291)) +
          "   (007m edge point)")
    for s, tot, p in rows:
        print(f"{s:>7.3f} {tot:>8.3f} "
              f"{(tot - PQ_REFERENCE) / PQ_REFERENCE * 100:>7.1f}% " +
              " ".join(f"{p.get(k, float('nan')):>12.3f}" for k in keys))

    best = min(rows, key=lambda r: r[1])
    d = (best[1] - PQ_REFERENCE) / PQ_REFERENCE * 100
    print(f"\nbest extended Thevenin: scale {best[0]:g} -> {best[1]:.4f} "
          f"({d:+.1f} % vs PQ)")
    if best[1] > 2.1581 - 1e-9:
        print("-> 0.2 WAS the optimum after all; the edge was coincidental and "
              "the +29 % stands.")
    elif d <= 2.0:
        print("-> the ladder was the binding constraint: Thevenin reaches PQ "
              "once allowed enough gain.")
    else:
        print(f"-> Thevenin improves past the old floor but is still {d:+.1f} % "
              "behind; deficit real but smaller than reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
