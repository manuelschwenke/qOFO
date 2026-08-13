#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007n_EXPLICIT_MARGIN.py
==============================================
Is the constant-PQ boundary's advantage an implicit constraint back-off?

007m showed the calibrated Thevenin boundary still loses by ~29 % after both
boundaries are tuned per zone to their own optimum, and that the surviving gap
sits almost entirely in the voltage-band-excess term (0.002 against 0.149) while
interface utilisation is identical.  The proposed explanation: output
constraints are imposed on ``y + H w``, so they read H directly and no step
weight can compensate them; an over-stated H over-states how far a step moves
the monitored voltages, and the optimiser therefore keeps a wider margin from
the band than the linear prediction requires.  That margin absorbs the model
error and the neighbour's unmodelled control action arriving between iterations.

If that is right, giving the Thevenin arm the margin EXPLICITLY should recover
the gap.  The controller's band is tightened through the per-zone overrides;
the SCORING band is left at its nominal value, so the objective still measures
excursions past the real limit and the comparison stays honest.

Outcome
-------
* Thevenin + margin reaches PQ  -> PQ is an accurate-enough model bundled with
  an implicit margin; the two ingredients can be separated, and the boundary
  choice is a packaging decision rather than a modelling one.
* Thevenin + margin still short -> the margin explanation is wrong and should
  be dropped from the thesis.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import dataclasses
import importlib.util
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

OUT_DIR = _ROOT / "results" / "007_tie_boundary" / "margin"
HOURS = 5.0
#: Per-zone optima from 007m.
SCALES = {"pq": {1: 0.35, 2: 0.35, 3: 0.2},
          "thevenin": {1: 0.2, 2: 0.2, 3: 0.2}}
MARGINS = (0.0, 0.01, 0.02)


def _cfg(boundary: str, margin: float):
    c = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(c, k, v)
    c.n_total_s = 3600.0 * HOURS
    c.verbose = 0
    c.run_stability_analysis = False
    c.tie_boundary_equivalent = boundary
    if boundary == "thevenin":
        c.tie_thevenin_k = THEVENIN_K_PER_CORRIDOR
    if margin > 0.0:
        # Tighten what the CONTROLLER enforces.  c.v_min_pu / c.v_max_pu stay
        # at their nominal values, and those are what extract_metrics reads,
        # so the score still counts excursions past the real limit.
        c.zone_v_min_pu = {z: c.v_min_pu + margin for z in (1, 2, 3)}
        c.zone_v_max_pu = {z: c.v_max_pu - margin for z in (1, 2, 3)}
    return c


def _hook(scales: Dict[int, float]):
    def _h(state: Dict[str, Any]) -> bool:
        for z, ctrl in state.get("tso_controllers", {}).items():
            s = scales.get(int(z), 1.0)
            gw = getattr(getattr(ctrl, "params", None), "g_w", None)
            if s == 1.0 or gw is None:
                continue
            ctrl.params = dataclasses.replace(
                ctrl.params, g_w=np.asarray(gw, dtype=float) * float(s))
        return False
    return _h


def _score(boundary: str, margin: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = OUT_DIR / f"{boundary}_m{margin:g}_{HOURS:g}h.pkl"
    if pkl.exists():
        recs = pickle.load(open(pkl, "rb"))
    else:
        t0 = time.time()
        recs = run_multi_tso_dso(_cfg(boundary, margin),
                                 pre_loop_hook=_hook(SCALES[boundary]))
        print(f"    ran in {time.time() - t0:.0f} s", flush=True)
        if recs:
            pickle.dump(recs, open(pkl, "wb"))
    if not recs:
        return float("inf"), {}
    # Score against the NOMINAL band, never the tightened one.
    m = extract_metrics(recs, _cfg(boundary, 0.0))
    return performance_scalar(m, PerfWeights())


def main() -> int:
    print("=" * 88)
    print("EXPLICIT CONSTRAINT MARGIN -- can Thevenin recover the gap?")
    print(f"  {HOURS:g} h, each boundary at its 007m per-zone optimum")
    print("  controller band tightened; scoring band left nominal")
    print("=" * 88)

    rows = []
    for boundary, margins in (("pq", (0.0,)), ("thevenin", MARGINS)):
        for mg in margins:
            print(f"\n[{boundary}] margin {mg:g} pu ...", flush=True)
            tot, parts = _score(boundary, mg)
            rows.append((boundary, mg, tot, parts))
            print(f"    -> {tot:.4f}")

    keys = ["v_rms_ts", "v_worst_ts", "v_band_ts", "q_pcc", "pcc_underutil"]
    hdr = (f"{'boundary':>9} {'margin':>7} {'TOTAL':>8} " +
           " ".join(f"{k:>13}" for k in keys))
    print("\n" + hdr)
    print("-" * len(hdr))
    for b, mg, tot, p in rows:
        print(f"{b:>9} {mg:>7.3f} {tot:>8.3f} " +
              " ".join(f"{p.get(k, float('nan')):>13.3f}" for k in keys))

    base = next(t for b, m, t, _ in rows if b == "pq")
    best = min((r for r in rows if r[0] == "thevenin"), key=lambda r: r[2])
    d = (best[2] - base) / base * 100.0
    print(f"\nPQ (no margin)            : {base:.4f}")
    print(f"Thevenin best (margin {best[1]:g}) : {best[2]:.4f}   ({d:+.1f} %)")
    if d <= 2.0:
        print("\n-> RECOVERED. The gap was a missing constraint margin, not the "
              "boundary model.\n   PQ bundles model and margin; they separate.")
    elif best[2] < rows[1][2] - 1e-9:
        print("\n-> PARTIAL. An explicit margin helps but does not close the "
              "gap; the margin\n   is part of the story, not all of it.")
    else:
        print("\n-> NOT RECOVERED. The margin explanation does not hold and "
              "should be dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
