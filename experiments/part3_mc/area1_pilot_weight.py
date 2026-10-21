#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/part3_mc/area1_pilot_weight.py
==========================================

Check of the area-1 hypothesis of the first Monte-Carlo slice (2026-10-08).

Observation (THESIS_part3_mc, 92 paired episodes): O2 keeps the area-1 pilot
about 6 mpu ABOVE its reference, S1 keeps it on reference; in return O2 has a
third less error over the area's reference set and half the worst deviation.

Hypothesis: the OFO minimises the error over ALL referenced buses of the area
and does not treat the pilot as special; where the area's voltages cannot all
sit at 1.03 pu, the least-squares compromise leaves the pilot off reference.
It is a trade made by the objective, not weaker control.

Test (dose-response, paired): run O2 with the area-1 pilot's voltage-tracking
weight raised by factors k on the same episodes, and plain O2 as the same-
machine baseline.  The hypothesis predicts, as k grows, a smaller area-1 pilot
error AND a larger area-1 error over the reference set (and worst deviation),
moving towards S1.  If instead the pilot error does not fall, or falls without
the area error rising, the hypothesis is wrong.  The loop gain of zone 1 is
printed by the runner ("[g_v bus] ... lambda_max a -> b"); a run near or above
2 would confound the test and is flagged.

Episodes: the first N seeds (in seed order) of the campaign on which both O2
and S1 were ``ok`` -- no selection on the outcome of interest.

Run
---
    python -m experiments.part3_mc.area1_pilot_weight --mc-dir <THESIS_part3_mc> --jobs 12
    python -m experiments.part3_mc.area1_pilot_weight --mc-dir <...> --report

Author: Manuel Schwenke / Claude Code (2026-10-08)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import dataclasses
import json
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from experiments.case1_svr.report_case1 import area_metrics
from experiments.case1_svr.run_case1 import PILOT_BUSES, resolved
from experiments.part3_mc import run_mc as M
from experiments.paths import RESULTS_ROOT

OUT = Path(RESULTS_ROOT) / "THESIS_part3_mc_area1_pilot"
AREA = 1
#: 1.0 = plain O2, the same-machine baseline.  Capped at 6 by the loop gain:
#: zone-1 curvature lambda_max (stable iff < 2, budget ceiling 1.6) measured
#: 2026-10-08 on the first episode: k = 1 -> 0.886, 4 -> 1.287, 16 -> 2.893,
#: 64 -> 9.321.  Above about k = 6 the test would probe instability instead.
FACTORS = [1.0, 2.0, 4.0, 6.0]
N_EPISODES = 10


def _name(k: float) -> str:
    return "O2" if k == 1.0 else f"O2_pw{int(k)}"


def task(seed: int, k: float, out: str) -> Dict[str, Any]:
    out = Path(out)
    name = _name(k)
    try:
        start, events = M.draw_episode(seed)
        cfg = resolved("O2", M.HORIZON_MIN)
        over: Dict[str, Any] = dict(start_time=start, contingencies=list(events),
                                    result_dir=str(out / "_scratch" / f"{seed}_{name}"),
                                    verbose=0)
        if k != 1.0:
            over["tso_g_v_bus"] = {PILOT_BUSES[AREA]: float(cfg.g_v) * k}
        cfg = dataclasses.replace(cfg, **over)
        n_steps = int(round(cfg.n_total_s / cfg.dt_s))
        logf = out / "logs" / f"{seed}_{name}.log"
        t0 = time.time()
        with open(logf, "w", encoding="utf-8") as fh, \
                contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            log, exc = M._run(cfg)
        outcome, diag = M.classify(log, exc, n_steps, float(cfg.dt_s))
        lam = re.findall(r"lambda_max ([0-9.naN]+) -> ([0-9.naN]+)",
                         logf.read_text(encoding="utf-8", errors="replace"))
        rec: Dict[str, Any] = {"seed": seed, "variant": name, "k": k,
                               "outcome": outcome, "diag": diag,
                               "wall_s": time.time() - t0,
                               "lambda_zone1": [float(x) for x in lam[0]] if lam else None}
        if log and len(log) == n_steps:
            ref = float((cfg.zone_v_setpoints_pu or {}).get(AREA, cfg.v_setpoint_pu))
            m = area_metrics(log, AREA, ref, M.V_LO, M.V_HI)
            vp = np.array([(r.bus_vm_pu or {}).get(PILOT_BUSES[AREA], np.nan)
                           for r in log], dtype=float)
            m["pilot_signed_mean"] = float(np.nanmean(vp - ref))
            rec["metrics"] = m
        (out / "runs" / f"{seed}_{name}.json").write_text(
            json.dumps(rec, indent=2, default=float), encoding="utf-8")
        return {"seed": seed, "variant": name, "outcome": outcome}
    except Exception as exc:                        # noqa: BLE001
        (out / "runs" / f"{seed}_{name}.json").write_text(json.dumps(
            {"seed": seed, "variant": name, "k": k, "outcome": "task_crash",
             "diag": {"exception": repr(exc), "tb": traceback.format_exc()[-2000:]}},
            indent=2), encoding="utf-8")
        return {"seed": seed, "variant": name, "outcome": "task_crash"}


def select_seeds(mc_dir: Path, n: int) -> List[int]:
    oc = pd.read_csv(mc_dir / "outcomes_per_episode.csv")
    piv = oc.pivot_table(index="seed", columns="variant", values="outcome",
                         aggfunc="first")
    ok = piv[(piv.get("O2") == "ok") & (piv.get("S1") == "ok")].index
    return sorted(int(s) for s in ok)[:n]


def report(mc_dir: Path, out: Path) -> None:
    rows = []
    for p in sorted((out / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        m = r.get("metrics") or {}
        rows.append(dict(seed=r["seed"], variant=r["variant"], k=r.get("k"),
                         outcome=r["outcome"],
                         lam0=(r.get("lambda_zone1") or [np.nan, np.nan])[0],
                         lam1=(r.get("lambda_zone1") or [np.nan, np.nan])[1],
                         e_vp=m.get("e_vp"), e_v=m.get("e_v"),
                         worst=m.get("worst"), avr_trav=m.get("avr_trav"),
                         taps=m.get("taps"),
                         pilot_signed=m.get("pilot_signed_mean"),
                         rev=(r.get("diag") or {}).get("tap_reversals_end_ts")))
    df = pd.DataFrame(rows)
    df.to_csv(out / "area1_pilot_weight_runs.csv", index=False)
    # Server references on the same seeds.
    sm = pd.read_csv(mc_dir / "metrics_per_episode.csv")
    sm = sm[(sm.area == AREA) & (sm.variant.isin(["O2", "S1"]))]
    ref = sm.pivot_table(index="seed", columns="variant",
                         values=["e_vp", "e_v", "worst"])
    seeds = sorted(df.seed.unique())
    lines = [f"# Area-1 pilot-weight check ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"episodes: {seeds}", "",
             "| variant | k | n ok | lambda_max zone 1 | pilot e_vp med [mpu] | "
             "pilot signed mean med [mpu] | e_v area 1 med [mpu] | worst med [mpu] | "
             "AVR travel med [pu] | taps med |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v, g in df.groupby("variant", sort=False):
        g = g[g.outcome == "ok"]
        if g.empty:
            continue
        lines.append(
            f"| {v} | {g.k.iloc[0]:g} | {len(g)} | {g.lam0.median():.3f} -> {g.lam1.median():.3f} | "
            f"{1e3 * g.e_vp.median():.2f} | {1e3 * g.pilot_signed.median():+.2f} | "
            f"{1e3 * g.e_v.median():.2f} | {1e3 * g.worst.median():.1f} | "
            f"{g.avr_trav.median():.4f} | {g.taps.median():.0f} |")
    for v in ("S1", "O2"):
        if ("e_vp", v) in ref.columns:
            r = ref.loc[ref.index.isin(seeds)]
            lines.append(f"| server {v} | -- | {r[('e_vp', v)].notna().sum()} | -- | "
                         f"{1e3 * r[('e_vp', v)].median():.2f} | -- | "
                         f"{1e3 * r[('e_v', v)].median():.2f} | "
                         f"{1e3 * r[('worst', v)].median():.1f} | -- | -- |")
    # Reproduction check: local plain O2 vs server O2, per seed.
    loc = df[(df.variant == "O2") & (df.outcome == "ok")].set_index("seed")
    if not loc.empty and ("e_vp", "O2") in ref.columns:
        common = loc.index.intersection(ref.index)
        dd = (loc.loc[common, "e_vp"] - ref.loc[common, ("e_vp", "O2")]).abs()
        lines += ["", f"reproduction (local O2 vs server O2, area-1 e_vp): "
                      f"max |diff| = {1e3 * dd.max():.4f} mpu over {len(common)} seeds"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-dir", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n", type=int, default=N_EPISODES)
    ap.add_argument("--factors", default=",".join(str(f) for f in FACTORS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    mc_dir = Path(args.mc_dir)
    for sub in ("runs", "logs", "_scratch"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    if args.report:
        report(mc_dir, OUT)
        return
    seeds = select_seeds(mc_dir, args.n)
    factors = [float(x) for x in args.factors.split(",")]
    jobs = [(s, k, str(OUT)) for k in factors for s in seeds
            if not (OUT / "runs" / f"{s}_{_name(k)}.json").exists()]
    print(f"seeds {seeds}\nfactors {factors}\n{len(jobs)} run(s)", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(task, *j) for j in jobs]
        for f in as_completed(futs):
            print(time.strftime("%H:%M"), f.result(), flush=True)
    report(mc_dir, OUT)


if __name__ == "__main__":
    main()
