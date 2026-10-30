#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_5_avr_revalidation.py
================================================================
Re-validation of the selected weights after the device limits were harmonised
(author decision 2026-10-30; rule declared in
``docs/analysis/2026-10-30_weekend_plan.md`` §5 before this run):

* AVR set-point range of the TS-OFO [0.95, 1.07] -> [0.95, 1.05] pu
  (``MultiTSOConfig.gen_vm_max_pu``), as the classical SVR;
* TS tap lock 18 -> 15 min (``MultiTSOConfig.tso_int_cooldown`` = 4).

The rule, in one invocation (one run folder, one manifest):

1. **units** -- the 13 candidates admitted by run 0008 whose e_v(TS) lies within
   10 % of the best, and the Stage-I point (the reference of the interface-Q
   filter rule), on W1-W4 and W6 at 12 h, re-simulated with the unchanged
   Stage-II unit (``ch_9_3_stage2_screen._run_one_window``: frozen network,
   evented bank, bus metric).  Each unit is compared with its stored row; a
   bit-identical e_v(TS) means neither change acted in that unit.
2. **select** -- the UNCHANGED ``ch_9_3_select.py`` with the campaign's
   arguments (12 h, bank budget 12/day) on the re-simulated candidates.
3. **settle** -- in rank order as run 0008 (p95 <= 3 N_inner = 27, median
   residual <= 1 Mvar).  Results measured in runs 0001/0008 are reused: the
   isolated-STS bank freezes the TS parent after its single solve at t = 0, so
   the TS tap lock does not enter it and the AVR range only through that one
   operating point (stated assumption).  A candidate without a result gets
   Experiment C here.
4. **DECISION.json** -- ``unchanged`` iff the winner is c08b1d3d33ab, the
   weights the campaigns run on (``ch9_selected_band_2026_10_a1``).  Otherwise
   the winner is reported and nothing is written to ``configs/paramsets``
   (author decision required).

Usage::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_5_avr_revalidation.py --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PY = sys.executable
RUN0001 = (REPO_ROOT / "results" / "ch_09_parameter_selection"
           / "0001_2026-10-25_215433_weight_selection_driver_band")
UNION = RUN0001 / "selection" / "union_2b_round4"
RUN0008 = (REPO_ROOT / "results" / "ch_09_parameter_selection"
           / "0008_2026-10-28_103321_ch_9_3_amendment1")
START = "stage1_start_10_2026_banks"
WINDOWS = ["W1", "W2", "W3", "W4", "W6"]
HOURS = 12.0
BANK_BUDGET = 12.0
N_INNER = 9
SELECTED = "c08b1d3d33ab"
BASELINE = "3cb66225e1a7"
#: Run 0008's admitted candidates within 10 % of the best e_v(TS), rank order.
KEYS = ["301c285ee52b", "c08b1d3d33ab", "67ad1cf843ac", "9e48320e8a01",
        "ff3f747d3f96", "c9a5a0bb139e", "0efa83f8046a", "57f0162a5166",
        "4ccfa217c2b5", "e95cb58c9043", "34a9f0cf1444", "7935cc303372",
        "8323fe3f0c0b"]


def _log(root: Path, msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with (root / "revalidation.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def stage_units(root: Path, workers: int) -> Dict[str, Dict[str, Any]]:
    from experiments.ch_09_parameter_selection.ch_9_3_stage2_screen import _run_one_window
    udir = root / "units"
    udir.mkdir(parents=True, exist_ok=True)
    cands = {k: json.loads((UNION / f"cand_{k}.json").read_text("utf-8"))
             for k in KEYS + [BASELINE]}
    jobs = [{"key": k, "window": w, "hours": HOURS, "paramset": START,
             "weights": c["weights"], "lambda_max_B": c.get("lambda_max_B"),
             "is_baseline": k == BASELINE, "frozen_network": True,
             "evented": True, "v_metric": "bus",
             "out": str(udir / f"{k}__{w}.json")}
            for k, c in cands.items() for w in WINDOWS]
    _log(root, f"units: {len(jobs)} ({len(cands)} candidates x {WINDOWS}), "
               f"{workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_window, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
                _log(root, f"units: {j['key']} {j['window']} {r.get('status')} "
                           f"e_v_ts {r.get('e_v_ts')}")
            except Exception as exc:              # one unit never kills the rest
                _log(root, f"units: {j['key']} {j['window']} FAILED {exc!r}")
    return cands


def stage_assemble(root: Path, cands: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    from tuning.scenarios_mc_v2 import ZERO_CAPABILITY
    from tuning.stage2_search import aggregate
    cdir = root / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    repro: Dict[str, Any] = {}
    for k, c in cands.items():
        per_window: Dict[str, Any] = {}
        for w in WINDOWS:
            f = root / "units" / f"{k}__{w}.json"
            if f.exists():
                per_window[w] = json.loads(f.read_text("utf-8"))["row"]
        if len(per_window) != len(WINDOWS):
            _log(root, f"assemble: {k} has {len(per_window)}/{len(WINDOWS)} units; skipped")
            continue
        repro[k] = {w: {"new": per_window[w].get("e_v_ts"),
                        "stored": c["per_window"][w].get("e_v_ts"),
                        "identical": per_window[w].get("e_v_ts")
                        == c["per_window"][w].get("e_v_ts")} for w in WINDOWS}
        out = {kk: vv for kk, vv in c.items()
               if kk not in ("per_window", "aggregate",
                             "aggregate_with_zero_capability", "wall_s")}
        out.update({"hours": HOURS, "per_window": per_window,
                    "aggregate": aggregate(per_window, exclude=ZERO_CAPABILITY),
                    "aggregate_with_zero_capability": aggregate(per_window),
                    "wall_s": sum(float(r.get("wall_s", 0.0)) for r in per_window.values()),
                    "revalidation": "2026-10-30 AVR 1.05, TS lock 15 min"})
        (cdir / f"cand_{k}.json").write_text(json.dumps(out, indent=1, default=float),
                                              "utf-8")
    n_id = sum(v["identical"] for r in repro.values() for v in r.values())
    n_all = sum(len(r) for r in repro.values())
    (root / "reproduction.json").write_text(json.dumps(
        {"n_identical": n_id, "n_units": n_all, "per_candidate": repro},
        indent=1, default=float), "utf-8")
    _log(root, f"assemble: {n_id} / {n_all} units reproduce the stored e_v(TS) exactly")
    return repro


def stage_select(root: Path) -> Dict[str, Any]:
    out = root / "selection.json"
    with (root / "select.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.call(
            [PY, "-X", "utf8", "experiments/ch_09_parameter_selection/ch_9_3_select.py",
             "--candidates", _rel(root / "candidates"), "--hours", f"{HOURS:g}",
             "--shunt-ops-per-day", f"{BANK_BUDGET:g}", "--out", _rel(out)],
            cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"ch_9_3_select.py exit {rc}; see select.log")
    rep = json.loads(out.read_text("utf-8"))
    _log(root, f"select: {rep['n_feasible']} / {rep['n_total']} feasible; ranked "
               f"{[r['key'] for r in rep['ranked']]}")
    return rep


def stage_settle(root: Path, rep: Dict[str, Any], workers: int) -> Dict[str, Any]:
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import _criterion
    from tuning.parameters import WEIGHT_FIELDS
    from tuning.stage2_search import rank_lexicographic
    expc = root / "expC"
    expc.mkdir(parents=True, exist_ok=True)
    path = expc / "criterion_results.json"
    # Measured results of runs 0001 and 0008 (the latter holds 301c285ee52b);
    # an aborted run is not a result.
    results: Dict[str, Any] = {}
    for src in (RUN0001, RUN0008):
        prior = json.loads((src / "expC" / "criterion_results.json").read_text("utf-8"))
        results.update({k: {**v, "reused_from": v.get("reused_from", _rel(src))}
                        for k, v in prior.items() if "error" not in v})
    if path.exists():
        results.update(json.loads(path.read_text("utf-8")))
    feasible = {r["key"]: r for r in rep["ranked"]}
    while True:
        remaining = [r for k, r in feasible.items()
                     if not (k in results and not results[k]["pass"])]
        if not remaining:
            path.write_text(json.dumps(results, indent=1), "utf-8")
            return {"winner": None, "results": results}
        first = rank_lexicographic(remaining, indifference=0.02)[0]
        if first["key"] in results:
            path.write_text(json.dumps(results, indent=1), "utf-8")
            return {"winner": first, "results": results}
        label = f"cand_{first['key']}"
        wf = expc / f"{label}_weights.json"
        wf.write_text(json.dumps({k: float(first["weights"][k]) for k in WEIGHT_FIELDS},
                                 indent=1), "utf-8")
        _log(root, f"settle: Experiment C on {first['key']} ({workers} workers)")
        with (expc / f"{label}.log").open("a", encoding="utf-8") as fh:
            subprocess.call(
                [PY, "-X", "utf8",
                 "experiments/ch_09_parameter_selection/ch_9_1_ninner_isolated_sts.py",
                 "--paramset", START, "--weights-json", _rel(wf), "--label", label,
                 "--out", _rel(expc), "--workers", str(workers),
                 "--max-step-mvar", "10", "--save-trajectories"],
                cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
        results[first["key"]] = _criterion(expc, label)
        o = results[first["key"]]
        _log(root, f"settle: {first['key']}: {'PASS' if o['pass'] else 'FAIL'} "
                   f"p95 {o.get('p95')} median residual {o.get('median_residual_mvar')}")
        path.write_text(json.dumps(results, indent=1), "utf-8")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ch_9_5_avr_revalidation")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--c-workers", type=int, default=8,
                   help="workers of Experiment C, if a new candidate needs it")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--note", default="")
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")

    from configs.config import MultiTSOConfig
    from experiments.thesis_runs import start_thesis_run
    probe = MultiTSOConfig()
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved"],
            paramset=START,
            parameters={"candidates": ",".join(KEYS), "baseline": BASELINE,
                        "gen_vm_max_pu": probe.gen_vm_max_pu,
                        "tso_int_cooldown": probe.tso_int_cooldown,
                        "rule": "docs/analysis/2026-10-30_weekend_plan.md §5"},
            time_series=("W1-W4, W6 of tuning.scenarios_mc_v2.CONDITIONS, 12 h, one "
                         "disturbance each, evented, frozen network; T_TS 180 s, "
                         "T_STS 20 s, dt 20 s"),
            note=args.note or ("Re-validation of the selected weights under the "
                               "harmonised device limits (2026-10-30)"),
            resume=args.resume) as run:
        root = run.root
        _log(root, f"run folder {root}")
        cands = stage_units(root, args.workers)
        stage_assemble(root, cands)
        rep = stage_select(root)
        st = stage_settle(root, rep, args.c_workers)
        win = st["winner"]
        decision = {
            "winner": None if win is None else win["key"],
            "unchanged": win is not None and win["key"] == SELECTED,
            "selected_before": SELECTED,
            "ranked": [r["key"] for r in rep["ranked"]],
            "settling": {k: st["results"][k].get("pass") for k in st["results"]},
            "written": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        }
        (root / "DECISION.json").write_text(json.dumps(decision, indent=1), "utf-8")
        _log(root, f"DECISION: winner {decision['winner']}, "
                   f"unchanged {decision['unchanged']}")
    return 0 if decision["unchanged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
