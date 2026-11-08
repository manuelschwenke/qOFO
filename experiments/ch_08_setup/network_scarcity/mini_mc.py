#!/usr/bin/env python3
"""
experiments/ch_08_setup/network_scarcity/mini_mc.py
===================================================

Case-2 MC episodes of a given seed list on the configured default network
(2026-11-08: ``rural_700_north_res``), with ``run_mc.variant_task`` -- the
MC's own episode draw, unit commitment, events, noise seed and outcome
classification.  The base network's outcomes for the same seeds are in the
MC of record (``mc/0002_2026-11-07_183649_run_mc_newbase``).

Purpose: robustness of the new network in the hours where it is scarce (G3
decommitted) before the tuning campaign; the paramset is the selected one,
not re-tuned for this network (indicative).

    python -m experiments.ch_08_setup.network_scarcity.mini_mc \
        --seeds <json list> --prescreen <dir with <seed>.json> \
        --variants L2,O2,O3,S2 --jobs 30

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import datetime as _dt
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from experiments.thesis_runs import add_run_arguments, start_thesis_run  # noqa: E402

TARGET = ("ch:setup:multi:windreplace",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--prescreen", required=True)
    ap.add_argument("--variants", default="L2,O2,O3,S2")
    ap.add_argument("--minutes", type=int, default=180)
    ap.add_argument("--jobs", type=int, default=30)
    add_run_arguments(ap)
    a = ap.parse_args()
    from configs.config import MultiTSOConfig
    from experiments.ch_11_case2_multisystem.mc import run_mc
    seeds = [int(s) for s in json.loads(Path(a.seeds).read_text())]
    vs = a.variants.split(",")
    run = start_thesis_run(
        __file__, target=TARGET,
        parameters={"network": MultiTSOConfig.scenario, "seeds": seeds,
                    "variants": vs, "minutes": a.minutes,
                    "paramset": run_mc.PARAMSET,
                    "draw": "run_mc.draw_episode (MC of record)"},
        time_series=f"{len(seeds)} MC episodes x {a.minutes} min",
        note=a.note, tag=a.tag)
    out = run.root
    d = run_mc._dirs(out)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        shutil.copy(Path(a.prescreen) / f"{s}.json", d["pre"] / f"{s}.json")
    jobs = [(s, v) for s in seeds for v in vs
            if not (d["runs"] / f"{s}_{v}.json").exists()]
    print(f"{out}: {len(jobs)} runs on {MultiTSOConfig.scenario}", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_mc.variant_task, s, v, str(out), a.minutes): (s, v)
                for s, v in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            s, v = futs[f]
            try:
                r = f.result()
                print(f"{_dt.datetime.now():%H:%M:%S} {k}/{len(jobs)} {s} {v} "
                      f"{r.get('outcome') if isinstance(r, dict) else r}", flush=True)
            except Exception as exc:              # noqa: BLE001
                print(f"{k}/{len(jobs)} {s} {v} worker error {exc!r}", flush=True)
    run.describe_output("runs/", "one JSON (+ npz, log) per (seed, variant), as run_mc")
    run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
