#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007q_SUFFICIENCY_ON_RMS_CONFIG.py
========================================================
Confirms the sufficiency claim on the WORKING config, not the paper one.

The claim the thesis makes is not "PQ beats Thevenin" -- that needs a fair
tuning comparison we could not settle -- but the weaker and more robust
statement that **the constant PQ equivalent is sufficient**: a 12-70x more
faithful boundary model buys only single-digit percent in achieved voltage
tracking, so the closed loop is insensitive to a model error of this size and
the feedback absorbs it.

That was measured against ``make_cigre_config``.  ``run_multi_system_ofo.py``
carries a different, empirically tuned operating point (lighter step weights:
g_w_der 50 vs 100, g_w_pcc 80 vs 200, g_w_tso_oltc 5000 vs 10000), and the
sufficiency claim should hold there too before it goes into the thesis
unqualified.

Arms
----
* ``pq``       -- the shipped empirical setting, unscaled.
* ``thevenin`` -- short per-zone pass around the translated scale
  (0.15, 0.15, 0.3), so the faithful model is not judged while under-driven.
  Under-tuning it would flatter the sufficiency claim, which is the wrong
  direction to err in.

What matters is not which wins but how far apart they are: a small spread
against a 12x model-fidelity difference is the evidence.

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

from experiments.helpers.run_params import (  # noqa: E402
    config_fingerprint, dump_params,
)
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.network_reduction import THEVENIN_K_PER_CORRIDOR  # noqa: E402
from tuning.metrics import extract_metrics  # noqa: E402
from tuning.objectives_v2 import PerfWeights, performance_scalar  # noqa: E402


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_RMS = _load(str(_ROOT / "experiments" / "run_multi_system_ofo.py"), "_rms")

_DEFAULT_OUT = os.environ.get(
    "QOFO_LOCAL_RESULTS",
    str(Path.home() / "AppData" / "Local" / "qOFO_runs" / "007_suff"))
ZONES = (1, 2, 3)
#: Both arms get a search of the same shape.  The first pass searched only the
#: Thevenin arm and left PQ at its shipped point, which is the same asymmetry --
#: pointing the other way -- that invalidated the earlier CIGRE comparison.
PLAN = {
    "pq":       ({1: 1.0, 2: 1.0, 3: 1.0}, {z: (0.7, 1.4) for z in ZONES}),
    "thevenin": ({1: 0.15, 2: 0.15, 3: 0.3},
                 {1: (0.1, 0.25), 2: (0.1, 0.25), 3: (0.2, 0.5)}),
}


def _cfg(boundary: str, scales):
    c = _RMS.make_config()
    c.verbose = 0
    c.run_stability_analysis = False
    for f in ("live_plot_controller", "live_plot_cascade", "live_plot_system",
              "live_plot_tracking", "live_plot_sbx"):
        setattr(c, f, False)
    c.tie_boundary_equivalent = boundary
    if boundary == "thevenin":
        c.tie_thevenin_k = THEVENIN_K_PER_CORRIDOR
    c.zone_g_w_scale = dict(scales) if scales else None
    return c


def _eval(boundary, scales, out: Path, cache) -> Tuple[float, dict]:
    cfg = _cfg(boundary, scales)
    # Fingerprint the WHOLE config into the key: keying on the swept
    # parameters alone lets a result computed under one baseline be reused
    # under another.  That is not hypothetical -- it corrupted this very
    # comparison once (2 h cached arm served against a fresh 5 h arm).
    tag = (f"{boundary}_" +
           ("none" if not scales else "_".join(f"{scales[z]:g}" for z in ZONES))
           + f"_{config_fingerprint(cfg)}")
    if tag in cache:
        return cache[tag]
    pkl = out / f"{tag}.pkl"
    if pkl.exists():
        recs = pickle.load(open(pkl, "rb"))
    else:
        # Snapshot the resolved parameters BEFORE running, so the record exists
        # even if the run dies, and so a later edit to make_config() cannot
        # rewrite the history of this result.
        dump_params(out / f"{tag}.params.json", cfg,
                    extra={"tag": tag, "boundary": boundary,
                           "zone_g_w_scale": scales})
        t0 = time.time()
        recs = run_multi_tso_dso(cfg)
        print(f"      ran in {time.time() - t0:.0f} s", flush=True)
        if recs:
            pickle.dump(recs, open(pkl, "wb"))
    cache[tag] = ((float("inf"), {}) if not recs else
                  performance_scalar(
                      extract_metrics(recs, _cfg(boundary, scales)),
                      PerfWeights()))
    return cache[tag]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_DEFAULT_OUT)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("SUFFICIENCY CHECK ON run_multi_system_ofo's OWN TUNING")
    print(f"  horizon {_RMS.make_config().n_total_s / 3600:g} h   -> {out}")
    print("=" * 78)

    cache: Dict[str, Tuple[float, dict]] = {}
    res = {}
    for boundary, (start, ladder) in PLAN.items():
        print(f"\n[{boundary}] per-zone pass ...")
        scales = dict(start)
        best, parts = _eval(boundary, scales, out, cache)
        print(f"  start {[scales[z] for z in ZONES]}: {best:.4f}")
        for z in ZONES:
            for s in ladder[z]:
                trial = dict(scales)
                trial[z] = s
                print(f"  zone {z} -> {s:g} ...", flush=True)
                v, p = _eval(boundary, trial, out, cache)
                print(f"      {v:.4f}")
                if v < best:
                    best, parts, scales = v, p, trial
            print(f"  zone {z} fixed at {scales[z]:g}  (best {best:.4f})")
        res[boundary] = (scales, best, parts)

    keys = ["v_rms_ts", "v_worst_ts", "v_band_ts", "q_pcc"]
    hdr = f"{'arm':>28} {'TOTAL':>8} " + " ".join(f"{k:>12}" for k in keys)
    print("\n" + "=" * 80 + "\nRESULT\n" + "=" * 80 + f"\n{hdr}\n" + "-" * len(hdr))
    for b in ("pq", "thevenin"):
        sc, tot, p = res[b]
        print(f"{b + ' ' + str([sc[z] for z in ZONES]):>28} {tot:>8.3f} " +
              " ".join(f"{p.get(k, float('nan')):>12.3f}" for k in keys))

    pq_tot, pq_parts = res["pq"][1], res["pq"][2]
    best, parts = res["thevenin"][1], res["thevenin"][2]
    dv = ((parts.get("v_rms_ts", np.nan) - pq_parts.get("v_rms_ts", np.nan))
          / pq_parts.get("v_rms_ts", np.nan) * 100.0)
    dt = (best - pq_tot) / pq_tot * 100.0
    print(f"\nachieved TS voltage tracking, Thevenin vs PQ: {dv:+.1f} %")
    print(f"overall objective,             Thevenin vs PQ: {dt:+.1f} %")
    print("\nSufficiency reads on the SPREAD, not the sign: a 12-70x more")
    print("faithful boundary model buying only a few percent of achieved")
    print("tracking is what makes the simple equivalent adequate.")
    if abs(dv) <= 15.0:
        print(f"-> spread {abs(dv):.1f} % against a 12x model difference: "
              "claim holds on this config too.")
    else:
        print(f"-> spread {abs(dv):.1f} % is large; the claim does NOT "
              "transfer unqualified to this config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
