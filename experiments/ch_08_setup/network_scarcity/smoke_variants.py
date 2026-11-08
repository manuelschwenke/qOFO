#!/usr/bin/env python3
"""
experiments/ch_08_setup/network_scarcity/smoke_variants.py
==========================================================

Closed-loop smoke test of every Case-2 variant on a candidate network before
it becomes the base of a tuning campaign (2026-11-08, ``rural_700_north_res``:
zone 2 has no transmission IBG, so the zone-2 TS controller runs without a
continuous IBG column for the first time).

Per (operating point, variant): 60 min at the Case-2 step, the selected
paramset as it is (not re-tuned: indicative only), a +250 Mvar reactive load
step at IEEE bus 12 at minute 30 + 100 s.  Recorded: completed or the
exception, the number of steps, area voltage range, G1's Q range.

    python -m experiments.ch_08_setup.network_scarcity.smoke_variants --jobs 18

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
from typing import Any, Dict

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from experiments.thesis_runs import add_run_arguments, start_thesis_run  # noqa: E402

TARGET = ("ch:setup:multi:windreplace",)
OPS = {"W1": "2016-01-05T08:00", "W6": "2016-07-18T14:00",
       "G3off": "2016-05-27T13:00"}     # MC20261155: G3 decommitted, area 2 scarce
VARIANTS = ("L2", "S1", "S2", "O2", "O3", "M")
MINUTES = 60
LOAD_BUS = 11                            # IEEE 12 (the REF load step)
LOAD_Q = 250.0


def one(op: str, variant: str, scenario: str, out: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    path = Path(out) / f"{op}__{variant}.json"
    rec: Dict[str, Any] = {"op": op, "start": OPS[op], "variant": variant,
                           "scenario": scenario}
    try:
        from experiments.ch_11_case2_multisystem import variants as case2
        from experiments.helpers.records import ContingencyEvent
        from experiments.runners.multi_tso_dso import run_multi_tso_dso
        t_on = 30 * 60.0 + 100.0
        ev = [ContingencyEvent(minute=30, time_s=t_on, element_type="load",
                               bus=LOAD_BUS, p_mw=0.0, q_mvar=LOAD_Q,
                               action="connect")]
        cfg = case2.resolved(variant, MINUTES, ev, 20261108)
        cfg = dataclasses.replace(
            cfg, scenario=scenario, start_time=_dt.datetime.fromisoformat(OPS[op]),
            verbose=0, result_dir=str(Path(out).parent / "_scratch"))
        buf = io.StringIO()
        log = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                log = run_multi_tso_dso(cfg)
            rec["ok"] = True
        except BaseException as exc:              # noqa: BLE001
            log = getattr(exc, "partial_log", None)
            rec.update(ok=False, err=f"{type(exc).__name__}: {exc}"[:400],
                       tb=traceback.format_exc()[-1500:])
        rec["n_steps"] = len(log or [])
        rec["n_expected"] = int(round(MINUTES * 60 / cfg.dt_s))
        if log:
            for z in (1, 2, 3):
                v = [float(r.zone_v_min[z]) for r in log if r.zone_v_min and z in r.zone_v_min]
                vx = [float(r.zone_v_max[z]) for r in log if r.zone_v_max and z in r.zone_v_max]
                rec[f"v_min_{z}"] = min(v) if v else None
                rec[f"v_max_{z}"] = max(vx) if vx else None
            q = [float(r.slack_q_mvar) for r in log]
            rec["g1_q"] = [min(q), max(q)]
            rec["uc_offline"] = list(log[0].uc_offline_gens or [])
            rec["warnings"] = [l for l in buf.getvalue().splitlines()
                               if "did NOT converge" in l or "Traceback" in l][:5]
    except BaseException as exc:                  # noqa: BLE001
        rec.update(ok=False, err=f"{type(exc).__name__}: {exc}"[:400],
                   tb=traceback.format_exc()[-1500:])
    rec["wall_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(rec, default=str))
    return {k: rec.get(k) for k in ("op", "variant", "ok", "n_steps", "v_min_2", "wall_s")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=18)
    ap.add_argument("--scenario", default="rural_700_north_res")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    add_run_arguments(ap)
    a = ap.parse_args()
    vs = a.variants.split(",")
    run = start_thesis_run(
        __file__, target=TARGET,
        parameters={"scenario": a.scenario, "variants": vs, "ops": OPS,
                    "minutes": MINUTES, "event": f"+{LOAD_Q} Mvar at bus {LOAD_BUS + 1}, "
                                                 f"min 30 + 100 s",
                    "paramset": "selected (not re-tuned for this network)"},
        time_series=f"{len(OPS)} OPs x {MINUTES} min", note=a.note, tag=a.tag)
    out = run.path("runs")
    out.mkdir(parents=True, exist_ok=True)
    run.path("_scratch").mkdir(exist_ok=True)
    jobs = [(o, v, a.scenario, str(out)) for o in OPS for v in vs]
    print(f"{run.root}: {len(jobs)} runs", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                print(f"{_dt.datetime.now():%H:%M:%S} {k}/{len(jobs)} {f.result()}", flush=True)
            except Exception as exc:              # noqa: BLE001
                print(f"{k}/{len(jobs)} worker error {exc!r}", flush=True)
    run.describe_output("runs/", "one JSON per (OP, variant)")
    run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
