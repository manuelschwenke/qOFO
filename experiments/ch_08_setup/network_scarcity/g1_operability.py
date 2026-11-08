#!/usr/bin/env python3
"""
experiments/ch_08_setup/network_scarcity/g1_operability.py
==========================================================

The G1 operability rule (``docs/analysis/2026-10-31_g1_operability_rule.md``,
section 1) applied to the scarcity redesign candidates and to tighter reactive
limits of the network equivalent G1 (author question 2026-11-08: should the
equivalent's Q be limited like an interconnection?).

Configuration as declared on 2026-10-31: LOC (``variants.resolved("L2")``),
design windows W1, W2, W3, W4, W6 at the start times of
``tuning/scenarios_mc_v2.py``, 12 h each, plant step 180 s, no events, noise
seed 20261027.  Pass = every step with a converged power flow and no
exception (the voltage band is not part of the rule).

Recorded per run: pass/fail and the failing step, G1's largest |Q| and the
share of steps at its limit, the lowest 345 kV voltage per area.

    python -m experiments.ch_08_setup.network_scarcity.g1_operability \
        --jobs 4 --scenarios rural_700,rural_700_north_res \
        --limits none,1200,600,400

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import contextlib
import dataclasses
import datetime as _dt
import io
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from experiments.thesis_runs import add_run_arguments, start_thesis_run  # noqa: E402

TARGET = ("ch:setup:multi:windreplace",)
WINDOWS = {"W1": "2016-01-05T08:00", "W2": "2016-02-22T13:00",
           "W3": "2016-01-04T08:00", "W4": "2016-11-26T10:00",
           "W6": "2016-07-18T14:00"}
HOURS = 12
DT_S = 180.0
NOISE_SEED = 20261027


def one(scenario: str, limit: Optional[float], window: str, out: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    lab = "none" if limit is None else str(int(limit))
    path = Path(out) / f"{window}__{scenario}__{lab}.json"
    rec: Dict[str, Any] = {"scenario": scenario, "limit": limit, "window": window,
                           "start": WINDOWS[window]}
    log = None
    try:
        from experiments.ch_11_case2_multisystem import variants as case2
        from experiments.runners.multi_tso_dso import run_multi_tso_dso
        cfg = case2.resolved("L2", HOURS * 60, [], NOISE_SEED)
        cfg = dataclasses.replace(
            cfg, scenario=scenario, slack_q_limit_mvar=limit, dt_s=DT_S,
            start_time=_dt.datetime.fromisoformat(WINDOWS[window]), verbose=0,
            result_dir=str(Path(out).parent / "_scratch"))
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                log = run_multi_tso_dso(cfg)
            rec["ok"] = True
        except Exception as exc:                  # noqa: BLE001
            log = getattr(exc, "partial_log", None)
            rec.update(ok=False, err=f"{type(exc).__name__}: {exc}"[:300])
        n_exp = int(round(HOURS * 3600 / DT_S))
        rec["n_steps"] = len(log or [])
        rec["n_expected"] = n_exp
        if log:
            q = np.array([float(getattr(r, "slack_q_mvar", np.nan) or np.nan) for r in log])
            lim = np.array([bool(getattr(r, "slack_q_at_limit", False)) for r in log])
            rec["g1_q_max"] = float(np.nanmax(q))
            rec["g1_q_min"] = float(np.nanmin(q))
            rec["g1_at_limit_share"] = float(lim.mean())
            for z in (1, 2, 3):
                v = [float(r.zone_v_min[z]) for r in log
                     if getattr(r, "zone_v_min", None) and z in r.zone_v_min]
                vx = [float(r.zone_v_max[z]) for r in log
                      if getattr(r, "zone_v_max", None) and z in r.zone_v_max]
                rec[f"v_min_{z}"] = float(min(v)) if v else None
                rec[f"v_max_{z}"] = float(max(vx)) if vx else None
            rec["uc_offline"] = list(getattr(log[0], "uc_offline_gens", None) or [])
        if rec.get("ok") and rec["n_steps"] < n_exp:
            rec["ok"] = False
            rec["err"] = f"stopped at step {rec['n_steps']} of {n_exp}"
    except Exception as exc:                      # noqa: BLE001
        rec.update(ok=False, err=f"{type(exc).__name__}: {exc}"[:300],
                   tb=traceback.format_exc()[-1200:])
    rec["wall_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(rec, default=str))
    return {k: rec.get(k) for k in ("window", "scenario", "limit", "ok", "n_steps",
                                    "g1_q_max", "g1_q_min", "wall_s")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scenarios", default="rural_700,rural_700_north_res")
    ap.add_argument("--limits", default="none,1200,600,400")
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--resume", default=None)
    add_run_arguments(ap)
    a = ap.parse_args()
    scen = a.scenarios.split(",")
    lims = [None if x == "none" else float(x) for x in a.limits.split(",")]
    wins = a.windows.split(",")
    run = start_thesis_run(
        __file__, target=TARGET,
        parameters={"scenarios": scen, "g1_limits_mvar": [x if x else "unlimited" for x in lims],
                    "windows": {w: WINDOWS[w] for w in wins}, "hours": HOURS,
                    "dt_s": DT_S, "noise_seed": NOISE_SEED, "variant": "L2 (LOC)",
                    "rule": "docs/analysis/2026-10-31_g1_operability_rule.md section 1"},
        time_series=f"{len(wins)} design windows x {HOURS} h",
        note=a.note, tag=a.tag, resume=a.resume)
    out = run.path("runs")
    out.mkdir(parents=True, exist_ok=True)
    run.path("_scratch").mkdir(exist_ok=True)
    jobs = [(s, l, w, str(out)) for s in scen for l in lims for w in wins
            if not (out / f"{w}__{s}__{'none' if l is None else int(l)}.json").exists()]
    print(f"{run.root}: {len(jobs)} runs", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                print(f"{_dt.datetime.now():%H:%M:%S} {k}/{len(jobs)} {f.result()}", flush=True)
            except Exception as exc:              # noqa: BLE001
                print(f"{k}/{len(jobs)} worker error {exc!r}", flush=True)
    run.describe_output("runs/", "one JSON per (window, scenario, G1 limit)")
    run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
