#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_7_amendment3.py
==========================================================
Amendment 3 to the weight selection on the unit-commitment base (author
decision 2026-11-01, declared AFTER the results of run
``0001_2026-10-31_135000_ch_9_6_full_procedure``; note
``docs/tuning/2026-11-01_amendment3_theta_band.md``).

The amendment replaces the engagement bound of amendment 2 and changes nothing
else::

    0.02 pu <= theta_STS,min <= 0.03 pu        (amendment 2: theta_STS,min <= 0.02)

Under a uniform deviation the strongest STS coupler then commits between 1.05
and 1.06 pu, neither earlier nor later.  The limits (hunting tolerance 0), the
ranking (e_v(TS), e_v(DS) within 2 %) and the settling requirement in rank
order are unchanged: as in ``ch_9_6_full_procedure.settle``, a candidate that
fails settling leaves the set before the e_v(DS) tie-break is applied again.

No new search.  The candidates are the ones run 0001 simulated (Stage II-a and
the four II-b rounds); the II-b rounds were drawn around amendment-2 centres.

Stages, in one invocation:

1. **select** -- theta_STS,min for every candidate of the source run on its
   operating points.  The admitted ones, and the Stage-I point (the e_q
   reference of the limits, whatever its theta; it never enters the ranking
   when outside the band), are copied to ``selection/union_a3``; the UNCHANGED
   ``ch_9_3_select.py`` is run on them with the campaign's arguments.
2. **settle** -- Experiment C in rank order, two checks here and three on
   e5power at a time (``ch_9_6_full_procedure``'s call and transport).  Before
   every batch, completed checks of the same test (same arguments as the
   source run's checks, same weights) found in ``--reuse`` folders are taken
   over, so the diagnostics of 2026-11-01 count once they have finished.
3. **result** -- W5 for the report, then
   ``configs/paramsets/ch9_selected_uc_2026_10_a3.json``.  SELECTED_PARAMSET is
   NOT switched here (author decision).

Usage (from the repository root, on the master host)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_7_amendment3.py --note "..."
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_7_amendment3.py --resume <run dir>
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_7_amendment3.py --dry-run

Author: Manuel Schwenke / Claude Code (2026-11-01)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402

SOURCE = (ROOT / "results" / "ch_09_parameter_selection"
          / "0001_2026-10-31_135000_ch_9_6_full_procedure")
RESULT = "ch9_selected_uc_2026_10_a3"
THETA_MIN, THETA_MAX = 0.02, 0.03          # pu; amendment 3
REUSE = [SOURCE / "expC",
         ROOT / "results" / "diagnostics" / "2026-11-01_settle_ratio_candidates"]
#: Experiment-C arguments that are not part of the test itself.
VOLATILE_ARGS = ("label", "out", "workers", "weights_json", "save_trajectories")
#: Paths a change in which does not touch what an e5power shard runs.
SHARD_NEUTRAL = ("docs/", "tests/", "experiments/ch_11_case2_multisystem/",
                 "experiments/ch_09_parameter_selection/ch_9_7_amendment3.py",
                 "experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py")


def admitted(theta: float) -> bool:
    return THETA_MIN <= theta <= THETA_MAX


def source_candidates() -> Dict[str, Path]:
    """Every candidate record of the source run, by key."""
    files = sorted((SOURCE / "stage2a" / "candidates").glob("cand_*.json"))
    files += sorted(SOURCE.glob("stage2b/round*/candidates/cand_*.json"))
    return {f.stem[5:]: f for f in files}


# -- 1. select --------------------------------------------------------------------------

def load_ops():
    """The source run's operating points, copied (they are never rebuilt here)."""
    src, dst = SOURCE / "operating_points", P.ops_dir()
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("op_*.npz"))
    if len(files) != 6:
        raise RuntimeError(f"{src}: {len(files)} operating points, expected W1-W6")
    for f in files:
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
    return P.load_ops()


def stage_select(ops) -> dict:
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import theta_sts_min
    sel = P.RUN / "selection"
    union = sel / "union_a3"
    union.mkdir(parents=True, exist_ok=True)
    thetas: Dict[str, float] = {}
    base_key = None
    for key, f in source_candidates().items():
        c = json.loads(f.read_text("utf-8"))
        thetas[key] = theta_sts_min(c["weights"], ops)
        if c.get("is_baseline"):
            base_key = key
        if admitted(thetas[key]) or c.get("is_baseline"):
            shutil.copy2(f, union / f.name)
    if base_key is None:
        raise RuntimeError(f"no Stage-I point among the candidates of {P.rel(SOURCE)}")
    (sel / "theta_a3.json").write_text(json.dumps(
        {"theta_min_pu": THETA_MIN, "theta_max_pu": THETA_MAX, "theta_sts_min_pu": thetas,
         "admitted": sorted(k for k, v in thetas.items() if admitted(v)),
         "removed": sorted(k for k, v in thetas.items() if not admitted(v)),
         "baseline": base_key}, indent=1), "utf-8")
    out = sel / "selection_a3.json"
    P.run(["experiments/ch_09_parameter_selection/ch_9_3_select.py",
           "--candidates", P.rel(union), "--hours", f"{P.HOURS:g}",
           "--shunt-ops-per-day", f"{P.BANK_BUDGET:g}", "--out", P.rel(out)],
          sel / "select_a3.log")
    if not out.exists():
        raise RuntimeError("selection: no report (no feasible candidate?)")
    rep = json.loads(out.read_text("utf-8"))
    if not admitted(thetas[base_key]):
        rep["ranked"] = [r for r in rep["ranked"] if r["key"] != base_key]
    rep["_theta"] = thetas
    n_adm = sum(1 for v in thetas.values() if admitted(v))
    P.log(f"selection a3: {n_adm} / {len(thetas)} inside {THETA_MIN:g}-{THETA_MAX:g} pu "
          f"(Stage-I point {base_key}: theta {thetas[base_key]:.4f}, reference only); "
          f"{len(rep['ranked'])} feasible, ranked first "
          f"{[r['key'] for r in rep['ranked'][:4]]}")
    return rep


# -- 2. settle --------------------------------------------------------------------------

def _meta(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "run_meta.json").read_text("utf-8"))


def _latest(label_dir: Path) -> Optional[Path]:
    f = label_dir / "_latest.txt"
    return label_dir / f.read_text("utf-8").strip() if f.exists() else None


def reference_args() -> Dict[str, Any]:
    """The test as the source run called it, from one of its own checks."""
    for d in sorted((SOURCE / "expC").glob("cand_*")):
        rd = _latest(d) if d.is_dir() else None
        if rd is not None and (rd / "run_meta.json").exists():
            a = _meta(rd)["args"]
            return {k: v for k, v in a.items() if k not in VOLATILE_ARGS}
    raise RuntimeError(f"no Experiment-C run in {P.rel(SOURCE / 'expC')}")


def same_test(rd: Path, weights: Dict[str, float], ref: Dict[str, Any]) -> bool:
    """A finished check of exactly this test on exactly these weights."""
    from tuning.parameters import WEIGHT_FIELDS
    if not ((rd / "steps.csv").exists() and (rd / "run_meta.json").exists()):
        return False                                   # not finished (or aborted)
    meta = _meta(rd)
    args = {k: v for k, v in meta["args"].items() if k not in VOLATILE_ARGS}
    if args != ref:
        return False
    argv = meta["argv"]
    wf = ROOT / argv[argv.index("--weights-json") + 1]
    if not wf.exists():
        return False
    w = json.loads(wf.read_text("utf-8"))
    return all(abs(float(w[k]) - float(weights[k])) <= 1e-12 * max(1.0, abs(float(weights[k])))
               for k in WEIGHT_FIELDS)


def take_over(results: Dict[str, Any], feasible: Dict[str, dict], reuse: List[Path],
              ref: Dict[str, Any]) -> None:
    """Completed checks of the same test from earlier folders join the results."""
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import _criterion
    expc = P.RUN / "expC"
    for key, r in feasible.items():
        if key in results:
            continue
        label = f"cand_{key}"
        for d in reuse:
            rd = _latest(d / label)
            if rd is None or not same_test(rd, r["weights"], ref):
                continue
            dst = expc / label / rd.name
            if not dst.exists():
                shutil.copytree(rd, dst)
            (expc / label / "_latest.txt").write_text(rd.name, "utf-8")
            o = {**_criterion(expc, label), "reused_from": P.rel(rd)}
            results[key] = o
            P.log(f"  {key}: taken over from {P.rel(rd)}: "
                  f"{'PASS' if o['pass'] else 'FAIL'} p95 {o.get('p95')} "
                  f"median residual {o.get('median_residual_mvar')}")
            break


def hosts(local_slots: int, remote_slots: int) -> List[str]:
    """Alternating, so a short batch still uses both hosts."""
    out: List[str] = []
    while local_slots or remote_slots:
        if local_slots:
            out.append("local")
            local_slots -= 1
        if remote_slots:
            out.append("remote")
            remote_slots -= 1
    return out


def read_slots(local_slots: int, remote_slots: int) -> Tuple[int, int]:
    """``slots.json`` in the run folder, if present, overrides the command line
    from the next batch on (hosts that free up mid-run can be added)."""
    f = P.RUN / "slots.json"
    if f.exists():
        d = json.loads(f.read_text("utf-8"))
        return int(d.get("local", local_slots)), int(d.get("remote", remote_slots))
    return local_slots, remote_slots


def stage_settle(rep: dict, c_workers: int, local_slots: int, remote_slots: int,
                 reuse: List[Path]) -> Tuple[Optional[dict], dict]:
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import _criterion
    from tuning.parameters import WEIGHT_FIELDS
    from tuning.stage2_search import rank_lexicographic
    expc = P.RUN / "expC"
    expc.mkdir(parents=True, exist_ok=True)
    path = expc / "criterion_results.json"
    results = json.loads(path.read_text("utf-8")) if path.exists() else {}
    feasible = {r["key"]: r for r in rep["ranked"]}
    ref = reference_args()
    while True:
        slots = hosts(*read_slots(local_slots, remote_slots))
        take_over(results, feasible, reuse, ref)
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        remaining = [r for k, r in feasible.items()
                     if not (k in results and not results[k]["pass"])]
        if not remaining:
            return None, results
        order = rank_lexicographic(remaining, indifference=0.02)
        if order[0]["key"] in results:
            return order[0], results
        batch = [r for r in order if r["key"] not in results][:len(slots)]
        P.log(f"settling check on {[r['key'] for r in batch]} ({', '.join(slots[:len(batch)])})")

        def one(i_r: Tuple[int, dict]) -> dict:
            i, r = i_r
            label = f"cand_{r['key']}"
            wf = expc / f"{label}_weights.json"
            wf.write_text(json.dumps({k: float(r["weights"][k]) for k in WEIGHT_FIELDS},
                                     indent=1), "utf-8")
            argv = P.exp_c_argv(label, wf, c_workers)
            if slots[i] == "local":
                P.run(argv, expc / f"{label}.log")
            else:
                P.push(wf, P.remote_path(wf))
                P.run_remote(label, argv, expc, [(expc / label, True)])
            return _criterion(expc, label)

        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            outs = list(ex.map(one, enumerate(batch)))
        for r, o in zip(batch, outs):
            results[r["key"]] = o
            P.log(f"  {r['key']}: {'PASS' if o['pass'] else 'FAIL'} p95 "
                  f"{o.get('p95')} median residual {o.get('median_residual_mvar')} "
                  f"{o.get('error', '')}")
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")


def check_remote_code() -> None:
    """The shards run e5power's checkout: everything they execute must equal ours."""
    rh = P.remote_head()
    changed = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "diff", "--name-only", rh],
        text=True).split()
    changed += subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "ls-files", "--others",
         "--exclude-standard"], text=True).split()
    bad = [f for f in changed if not f.startswith(SHARD_NEUTRAL)]
    if bad:
        raise RuntimeError(f"code differs from {P.REMOTE} ({rh[:7]}) in {bad[:8]}")
    P.log(f"{P.REMOTE} at {rh[:7]}; differences here are shard-neutral "
          f"({len(changed)} file(s))")


# -- 3. result --------------------------------------------------------------------------

def write_result(win: dict, rep: dict, c_res: dict, w5: dict) -> Path:
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    path = PARAMSET_DIR / f"{RESULT}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a parameter set is never overwritten")
    fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
    doc = {
        "name": RESULT,
        "title": "Controller weights selected by the Ch 8 procedure (Experiment B) "
                 "on the unit-commitment base of 2026-10-31, amendment 3",
        "description": [
            "Selected by docs/tuning/2026-10-28_weight_selection_procedure.md "
            "(frozen) on the candidates of run 0001_2026-10-31_135000_ch_9_6_full_procedure "
            "(unit-commitment base, Stage II-a and four II-b rounds):",
            f"filter with the engagement band {THETA_MIN:g} <= theta_STS,min <= "
            f"{THETA_MAX:g} pu (amendment 3, declared after that run's results: "
            "the strongest STS coupler commits between 1.05 and 1.06 pu); ranking "
            "on the per-bus TS voltage deviation with a 2 % band on the STS "
            "deviation; settling requirement in rank order.",
            "No new search: the II-b rounds were drawn around amendment-2 centres.",
        ],
        "provenance": {
            "producer": P.rel(Path(__file__)),
            "run_dir": P.rel(P.RUN),
            "procedure": "docs/tuning/2026-10-28_weight_selection_procedure.md",
            "amendment3": "docs/tuning/2026-11-01_amendment3_theta_band.md",
            "source_run": P.rel(SOURCE),
            "starting_paramset": P.START,
            "dso_voltage_band_pu": list(P.BAND),
            "simulated_windows": P.WINDOWS,
            "w5_report": w5,
            "selected_key": win["key"],
            "n_candidates_admitted": rep.get("n_total"),
            "n_feasible": len(rep.get("ranked", [])),
            "theta_sts_min_pu": rep["_theta"].get(win["key"]),
            "theta_sts_band_pu": [THETA_MIN, THETA_MAX],
            "failed_settling": sorted(k for k, v in c_res.items() if not v["pass"]),
            "lambda_max_B": win.get("lambda_max_B"),
            "aggregate_ranked_windows_per_bus": win["aggregate"],
            "baseline_aggregate": rep.get("baseline", {}).get("aggregate"),
            "settling": c_res.get(win["key"]),
            "weight_fingerprint": weight_fingerprint(fields),
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {path}")
    return path


# -- main -------------------------------------------------------------------------------

def procedure(args) -> int:
    P.check_config()
    if args.remote_slots > 0 and not args.dry_run:
        check_remote_code()
    P.log("== 1. selection with the engagement band of amendment 3")
    rep = stage_select(load_ops())
    reuse = [Path(d) for d in args.reuse]
    if args.dry_run:
        results: Dict[str, Any] = {}
        take_over(results, {r["key"]: r for r in rep["ranked"]}, reuse, reference_args())
        P.log(f"dry run: {len(results)} check(s) taken over; first in rank order "
              f"{[r['key'] for r in rep['ranked'][:5]]}")
        return 0
    P.log("== 2. settling in rank order")
    win, c_res = stage_settle(rep, args.c_workers, args.local_slots, args.remote_slots,
                              reuse)
    if win is None:
        P.log("no admitted candidate passes the settling requirement")
        (P.RUN / "FAILED.txt").write_text("no candidate passes\n", "utf-8")
        return 2
    P.log(f"SELECTED {win['key']} (theta {rep['_theta'][win['key']]:.4f} pu)")
    P.log("== 3. W5 for the report, result")
    w5 = P.report_w5(win)
    P.log(f"  W5: e_v_ts {w5.get('e_v_ts')}, e_v_ds {w5.get('e_v_ds')}")
    write_result(win, rep, c_res, w5)
    (P.RUN / "DONE.txt").write_text(f"selected {win['key']}\n", "utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_7_amendment3")
    p.add_argument("--c-workers", type=int, default=10, help="workers per settling check")
    p.add_argument("--local-slots", type=int, default=2, help="checks at a time here")
    p.add_argument("--remote-slots", type=int, default=3,
                   help=f"checks at a time on {P.REMOTE}")
    p.add_argument("--reuse", action="append", default=[str(d) for d in REUSE],
                   help="folder with finished checks of the same test (repeatable)")
    p.add_argument("--dry-run", action="store_true",
                   help="selection and take-over only, in results/_dry_ch9_7; nothing runs")
    p.add_argument("--resume", type=Path, default=None)
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")

    if args.dry_run:
        P.RUN = ROOT / "results" / "_dry_ch9_7" / f"{datetime.now():%Y%m%d-%H%M%S}"
        P.RUN.mkdir(parents=True)
        return procedure(args)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            paramset=P.START,
            parameters={"procedure": "docs/tuning/2026-10-28_weight_selection_procedure.md",
                        "amendment": "docs/tuning/2026-11-01_amendment3_theta_band.md",
                        "theta_sts_band_pu": [THETA_MIN, THETA_MAX],
                        "source_run": P.rel(SOURCE), "result_paramset": RESULT,
                        "reuse": [P.rel(Path(d)) for d in args.reuse],
                        "settling": f"{args.c_workers} workers per check, "
                                    f"{args.local_slots} here + {args.remote_slots} on "
                                    f"{P.REMOTE} at a time"},
            time_series=("Re-selection from the 12-h W1-W4, W6 simulations of the source "
                         "run (no new search); Experiment C: 120 isolated interface steps "
                         "in the same windows; W5 report only"),
            note=args.note or "Ch 9 re-selection on the commitment base, amendment 3",
            tag=args.tag, resume=args.resume) as trun:
        P.RUN = trun.root
        P.log(f"run folder {P.RUN}")
        t0 = time.time()
        try:
            rc = procedure(args)
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            P.log(f"driver error: {type(exc).__name__}: {exc}")
            (P.RUN / "FAILED.txt").write_text(traceback.format_exc(), "utf-8")
            rc = 1
        trun.describe_output("driver.log", "every stage with times")
        trun.describe_output("selection/", "theta_STS,min per candidate, admitted union, "
                             "unchanged ch_9_3_select.py report")
        trun.describe_output("expC/", "Experiment C in the amended rank order; "
                             "criterion_results.json (reused checks marked)")
        P.log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
