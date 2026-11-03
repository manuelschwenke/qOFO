#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py
==============================================================
Weight selection by procedure v3 on the unit-commitment base
(``docs/tuning/2026-11-03_weight_selection_procedure_v3.md``; author decisions of
2026-11-03 in ``docs/daily_log/11_2026/2026-11-01b_ch9_settling_failure_diagnosis.md``).

v3 is v2 (``ch_9_8_procedure_v2``, imported as ``V2``) with these changes, and
otherwise the same code path:

* **Stage I.**  One STS variant, pi_q 10 (v2's verified choice), with the coupler
  engagement target Delta v_eng,STS = 0.01 pu instead of 0.03; bank engagement
  0.005 pu instead of 0.010 (price about halved); g_w_gen 5e6 instead of 1e8, so
  the II-a box spans 5e5..5e7.  The settling verification is that of v2.
* **Search windows** W1, W3, W4, W6.  A candidate's windows stop early at its
  2nd unexplained tap reversal or its 1st bank reversal (hunting tolerance 1,
  bank 0); the Stage-I point is never stopped.
* **Filter** v2's limits at hunting tolerance 1, theta_TS,min <= 0.009 pu, and two
  measured engagement filters: STS couplers move in every search window; banks
  commit within 2 h after an event in at least two search windows.  If no
  candidate passes both, the bank filter is dropped, then the coupler filter
  (each level is logged; the Stage-I point is the reference and never ranked
  against a filter it fails).
* **Completion.** The 2 % band of the final search ranking is completed on W2
  and ranked on all five windows; settling confirmation and W5 as in v2.
* **Go criteria** for the follow-ups (author 2026-11-03), written to GO.json or
  NOGO.json: e_v_ts <= 1.10 x 0.00827; couplers in every search window; banks
  after events in >= 2 windows; all limits met.

Usage (from the repository root, on e5wimi)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py --note "..."
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py --resume <run dir>
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_11_procedure_v3.py --dry-run

Author: Manuel Schwenke / Claude Code (2026-11-03)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V2  # noqa: E402

PROCEDURE = "docs/tuning/2026-11-03_weight_selection_procedure_v3.md"
RESULT = "ch9_selected_uc_2026_11_v3"
VARIANT_NAME = "stage1_v3_uc_2026_11_pq{:g}"
PI_Q = 10.0
G_W_GEN = 5.0e6
D_ENGAGE = {"ts": 0.0075, "sts": 0.01, "shunt": 0.005}       # pu
THETA_STS = (0.005, 0.015)                     # pu; 0.01 by construction, kept as a check
SEARCH = ("W1", "W3", "W4", "W6")
ALL = "W1,W2,W3,W4,W6"
TOLERANCE = 1                                  # unexplained tap reversals tolerated
AFTER_EVENT_S = 7200.0                         # a bank commit "after an event"
BANK_WINDOWS = 2
V2_E_V_TS, GO_FACTOR = 0.00827, 1.10           # go criterion on e_v_ts
STAGE = {"now": "search"}

# v2's design and verification code, with the v3 constants.
V2.PROCEDURE, V2.RESULT, V2.VARIANT_NAME = PROCEDURE, RESULT, VARIANT_NAME
V2.PI_Q_VARIANTS, V2.D_ENGAGE, V2.THETA_STS = (PI_Q,), D_ENGAGE, THETA_STS
P.WINDOWS = ALL

_screen_argv = P.screen_argv


def screen_argv(out: Path, cj: Path, workers: int, windows: Optional[str] = None,
                skip_baseline: bool = False) -> List[str]:
    """Search units: the four search windows with the early stop.  Completion:
    all five windows (only W2 is missing).  An explicit window list (W5 report)
    is passed through unchanged."""
    if windows is not None:
        return _screen_argv(out, cj, workers, windows=windows, skip_baseline=skip_baseline)
    if STAGE["now"] == "complete":
        return _screen_argv(out, cj, workers, windows=ALL, skip_baseline=skip_baseline)
    return _screen_argv(out, cj, workers, windows=",".join(SEARCH),
                        skip_baseline=skip_baseline) + [
        "--abort-tap-reversals", str(TOLERANCE + 1), "--abort-bank-reversals", "1"]


P.screen_argv = screen_argv


# -- Stage I --------------------------------------------------------------------------------

V2_STAGE1 = "stage1_v2_uc_2026_11_pq10"


def design(ops) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The STS weights and the bank price from the v3 engagement targets; g_v,
    g_w_der, g_w_pcc and g_w_tso_oltc from v2's Stage I; g_w_gen 5e6.

    Re-running the TS contraction design with the cheaper bank raises g_w_der and
    g_w_pcc about fivefold (0.09 -> 0.49), into the region where v2 found nothing
    feasible (0 of 40 candidates with g_w_der >= 0.3), so the TS continuous
    prices stay v2's; the screen bound lambda_max(B) <= 1.546 still applies."""
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    wd, rep = s1.design_stage1(
        ops, pi_q=PI_Q, pi_v_ds=V2.PI_V_DS, tau=V2.TAU, g_w_gen=V2.G_W_GEN,
        d_engage_ts=D_ENGAGE["ts"], d_engage_sts=D_ENGAGE["sts"],
        d_engage_shunt=D_ENGAGE["shunt"], lambda_target_ts=V2.LAMBDA_TARGET["ts"],
        lambda_target_sts=V2.LAMBDA_TARGET["sts"])
    w = {k: float(v) for k, v in load_paramset(V2_STAGE1)["fields"].items()}
    for k in (*V2.STS_FIELDS, "g_w_tso_shunt"):
        w[k] = float(wd[k])
    w["g_w_gen"] = G_W_GEN
    if not V2.passes_screen(w, ops):
        raise RuntimeError(f"the v3 Stage-I point fails the analytic screen: {w}")
    rep["op_horizon_s"] = 3600.0 * P.HOURS
    rep["ts_continuous_from"] = V2_STAGE1
    return w, rep


def write_variant(w: Dict[str, float], rep: Dict[str, Any], ops) -> str:
    from configs.paramsets import PARAMSET_DIR
    from tuning import stage1_screen as s1
    from tuning.parameters import WEIGHT_FIELDS
    name = VARIANT_NAME.format(PI_Q)
    path = PARAMSET_DIR / f"{name}.json"
    fields = {k: float(w[k]) for k in WEIGHT_FIELDS}
    if path.exists():
        old = json.loads(path.read_text("utf-8"))["fields"]
        if any(abs(float(old[k]) - v) > 1e-9 * max(1.0, abs(v)) for k, v in fields.items()):
            raise SystemExit(f"{path} exists with other weights; never overwritten")
        return name
    doc = {
        "name": name,
        "title": f"Procedure v3 Stage-I point (pi_q {PI_Q:g}, theta_STS 0.01 pu)",
        "description": [f"Designed by {P.rel(Path(__file__))} under {PROCEDURE}.",
                        "NOT a tuned point: the starting point of the v3 Stage-II search."],
        "provenance": {"producer": P.rel(Path(__file__)), "procedure": PROCEDURE,
                       "date": datetime.now().date().isoformat(),
                       "operating_points": P.rel(V2.OPS_SOURCE),
                       "operating_point_horizon_s": rep.get("op_horizon_s"),
                       "d_engage_pu": rep.get("d_engage"), "g_w_gen": G_W_GEN,
                       "lambda_targets": rep.get("targets"),
                       "select_ceiling": s1.select_ceiling(),
                       **V2.describe(fields, ops)},
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {P.rel(path)}")
    return name


# -- selection ------------------------------------------------------------------------------

def engagement(c: Dict[str, Any]) -> Dict[str, Any]:
    """Measured engagement over the search windows a record holds."""
    pw = c["per_window"]
    ws = [w for w in SEARCH if w in pw]
    sts = {w: sum(1 for m in pw[w].get("moves", []) if "DSO" in str(m[1])) for w in ws}
    after = {w: any(0.0 <= float(m[0]) - float(e) <= AFTER_EVENT_S
                    for m in pw[w].get("shunt_moves", []) for e in pw[w].get("events_s", []))
             for w in ws}
    return {"sts_moves": sts, "sts_all": bool(ws) and all(v > 0 for v in sts.values()),
            "bank_after_windows": sum(after.values()),
            "bank": sum(after.values()) >= BANK_WINDOWS}


LEVELS = (("sts+bank", lambda e: e["sts_all"] and e["bank"]),
          ("sts", lambda e: e["sts_all"]),
          ("none", lambda e: True))


def stage_select(dirs: List[Path], tag: str, ops) -> dict:
    """theta bounds, engagement level (strictest that leaves a ranked candidate),
    then the unchanged ch_9_3_select.py at hunting tolerance 1."""
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import theta_sts_min
    sel = P.RUN / "selection"
    sel.mkdir(parents=True, exist_ok=True)
    recs: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for d in dirs:
        for f in sorted((d / "candidates").glob("cand_*.json")):
            c = json.loads(f.read_text("utf-8"))
            recs.setdefault(c["key"], (f, c))
    th_sts = {k: theta_sts_min(c["weights"], ops) for k, (_, c) in recs.items()}
    th_ts = {k: V2.theta_ts_min(c["weights"], ops) for k, (_, c) in recs.items()}
    eng = {k: engagement(c) for k, (_, c) in recs.items()}
    base_key = next((k for k, (_, c) in recs.items() if c.get("is_baseline")), None)
    if base_key is None:
        raise RuntimeError(f"selection {tag}: no Stage-I point record")
    adm = {k for k in recs if THETA_STS[0] <= th_sts[k] <= THETA_STS[1]
           and th_ts[k] <= V2.THETA_TS_MAX}
    (sel / f"engagement_{tag}.json").write_text(json.dumps(
        {"theta_sts_band_pu": list(THETA_STS), "theta_ts_max_pu": V2.THETA_TS_MAX,
         "theta_sts_min_pu": th_sts, "theta_ts_min_pu": th_ts, "engagement": eng,
         "admitted": sorted(adm)}, indent=1), "utf-8")
    for level, ok in LEVELS:
        union = sel / f"union_{tag}_{level}"
        if union.exists():
            shutil.rmtree(union)
        union.mkdir(parents=True)
        keep = {k for k in adm if ok(eng[k])} | {base_key}
        for k in keep:
            shutil.copy2(recs[k][0], union / recs[k][0].name)
        out = sel / f"selection_{tag}_{level}.json"
        P.run(["experiments/ch_09_parameter_selection/ch_9_3_select.py",
               "--candidates", P.rel(union), "--hours", f"{P.HOURS:g}",
               "--shunt-ops-per-day", f"{P.BANK_BUDGET:g}",
               "--max-hunting", str(TOLERANCE), "--out", P.rel(out)],
              sel / f"select_{tag}_{level}.log")
        rep = json.loads(out.read_text("utf-8")) if out.exists() else None
        if rep and rep.get("ranked"):
            if base_key not in adm or not ok(eng[base_key]):
                rep["ranked"] = [r for r in rep["ranked"] if r["key"] != base_key]
            if rep["ranked"]:
                rep["_level"], rep["_max_hunting"] = level, TOLERANCE
                rep["_theta_sts"], rep["_theta_ts"], rep["_engagement"] = th_sts, th_ts, eng
                P.log(f"selection {tag}: {len(adm)} / {len(recs)} inside the theta bounds; "
                      f"engagement level '{level}' ({len(keep) - 1} candidates); "
                      f"{len(rep['ranked'])} feasible at hunting tolerance {TOLERANCE}; "
                      f"ranked first {[r['key'] for r in rep['ranked'][:4]]}")
                return rep
        P.log(f"selection {tag}: nothing ranked at engagement level '{level}'")
    raise RuntimeError(f"selection {tag}: no feasible candidate at any engagement level")


V2.stage_select = stage_select


def complete_band(rep: dict, dirs: List[Path], ops, w_local: int, w_remote: int,
                  split_local: float) -> dict:
    """The final search band (and the Stage-I point) on W2; ranked on all windows."""
    best = min(r["aggregate"]["e_v_ts"] for r in rep["ranked"])
    band = [r for r in rep["ranked"] if r["aggregate"]["e_v_ts"] <= 1.02 * best]
    keys = {r["key"] for r in band} | {rep["baseline"]["key"]}
    out = P.RUN / "complete"
    (out / "windows").mkdir(parents=True, exist_ok=True)
    for d in dirs:
        for f in (d / "windows").glob("*__W*.json"):
            if f.name.split("__")[0] in keys and not (out / "windows" / f.name).exists():
                shutil.copy2(f, out / "windows" / f.name)
    P.log(f"completion: {len(band)} band candidate(s) and the Stage-I point on W2")
    STAGE["now"] = "complete"
    try:
        P.simulate(out, [r["weights"] for r in band if r["key"] != rep["baseline"]["key"]],
                   w_local, w_remote, split_local)
    finally:
        STAGE["now"] = "search"
    return stage_select([out], "complete", ops)


# -- go criteria and result -------------------------------------------------------------------

def go_criteria(win: dict, rep: dict) -> Dict[str, Any]:
    e = rep["_engagement"][win["key"]]
    checks = {"e_v_ts": win["aggregate"]["e_v_ts"] <= GO_FACTOR * V2_E_V_TS,
              "sts_couplers_every_search_window": e["sts_all"],
              "banks_after_events_2_windows": e["bank"],
              "all_limits_met": True}           # it is in rep["ranked"]
    return {"go": all(checks.values()), "checks": checks,
            "e_v_ts": win["aggregate"]["e_v_ts"], "limit_e_v_ts": GO_FACTOR * V2_E_V_TS,
            "engagement": e, "engagement_level": rep.get("_level")}


def write_result(win: dict, rep: dict, c_res: dict, w5: dict, ver: Dict[float, Any],
                 go: Dict[str, Any], ops) -> Path:
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    path = PARAMSET_DIR / f"{RESULT}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a parameter set is never overwritten")
    fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
    doc = {
        "name": RESULT,
        "title": "Controller weights selected by procedure v3 on the unit-commitment base",
        "description": [f"Selected by {PROCEDURE}: v2 with coupler engagement 0.01 pu, "
                        "bank engagement 0.005 pu, g_w_gen box 5e5..5e7, search on W1, "
                        "W3, W4, W6 with an early stop, hunting tolerance 1, measured "
                        "engagement filters, the band completed on W2."],
        "provenance": {
            "producer": P.rel(Path(__file__)), "run_dir": P.rel(P.RUN),
            "procedure": PROCEDURE, "starting_paramset": P.START,
            "stage1_verification": {f"{k:g}": v for k, v in ver.items()},
            "dso_voltage_band_pu": list(P.BAND), "search_windows": list(SEARCH),
            "ranked_windows": ALL, "w5_report": w5, "selected_key": win["key"],
            "engagement_level": rep.get("_level"), "max_hunting_applied": TOLERANCE,
            "n_feasible": len(rep.get("ranked", [])),
            "theta_sts_min_pu": rep["_theta_sts"].get(win["key"]),
            "theta_ts_min_pu": rep["_theta_ts"].get(win["key"]),
            "engagement": rep["_engagement"].get(win["key"]),
            "failed_settling": sorted(k for k, v in c_res.items() if not v["pass"]),
            "aggregate_all_windows": win["aggregate"],
            "baseline_aggregate": rep.get("baseline", {}).get("aggregate"),
            "settling": c_res.get(win["key"]), "go_criteria": go,
            "weight_fingerprint": weight_fingerprint(fields),
            **{f"design_{k}": v for k, v in V2.describe(fields, ops).items()},
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"wrote {path}")
    return path


# -- code on e5power ----------------------------------------------------------------------------

def sync_remote_code() -> None:
    """Copy every Python file that differs from e5power's commit, then compare
    hashes: the shards must run exactly this checkout's code."""
    rh = P.remote_head()
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    names = set(subprocess.check_output(git + ["diff", "--name-only", rh], text=True).split())
    names |= set(subprocess.check_output(git + ["ls-files", "--others", "--exclude-standard"],
                                         text=True).split())
    files = sorted(n for n in names if n.endswith(".py") and (ROOT / n).exists()
                   and not n.startswith(("docs/", "tests/", "results/")))
    for n in files:
        P.push(ROOT / n, P.REMOTE_ROOT + "\\" + n.replace("/", "\\"))
    local = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest().upper() for n in files}
    paths = "', '".join(P.REMOTE_ROOT + "\\" + n.replace("/", "\\") for n in files)
    out = P.ps(f"Invoke-Command -ComputerName {P.REMOTE} -ScriptBlock {{ "
               f"foreach ($f in @('{paths}')) {{ (Get-FileHash -Algorithm SHA256 $f).Hash }} }}")
    remote = out.split()
    bad = [n for n, h in zip(files, remote) if local[n] != h]
    if len(remote) != len(files) or bad:
        raise RuntimeError(f"code on {P.REMOTE} differs after the copy: {bad[:8]}")
    P.log(f"{P.REMOTE}: {len(files)} Python file(s) copied over {rh[:7]}; hashes equal")


# -- main -----------------------------------------------------------------------------------

def procedure(args) -> int:
    from tuning import stage1_screen as s1
    cfg = P.check_config()
    if not P.base_config().shunt_ff_ledger:
        raise RuntimeError("v3 runs with shunt_ff_ledger = True")
    P.log("== 1. Stage I (v3): one STS variant, theta_STS 0.01, banks 0.005, g_w_gen 5e6")
    ops = V2.load_ops()
    w, rep1 = design(ops)
    d = V2.describe(w, ops)
    P.log(f"  {', '.join(f'{k} {v:.6g}' for k, v in w.items())}")
    P.log(f"  rho {d['rho']:.4g}, kappa {d['kappa']:.4g}, theta_STS {d['theta_sts_min']:.4f}, "
          f"theta_TS {d['theta_ts_min']:.4f}, lambda_max(B) {d['lambda_max_B']:.4f}")
    if args.dry_run:
        from tuning.stage2_search import Box, sobol_candidates
        draws = sobol_candidates(Box(centre=w, spans=V2.spans_v2()), V2.N_2A, seed=P.SEED)
        n = sum(1 for x in draws if V2.passes_screen(x, ops))
        P.log(f"dry run: {n} / {V2.N_2A} II-a draws pass the screen; nothing simulated")
        return 0
    sync_remote_code()
    name = write_variant(w, rep1, ops)
    P.log("== 2. settling verification of the Stage-I point")
    ver = V2.verify({PI_Q: name}, {PI_Q: w}, args.verify_workers)
    if V2.choose_variant(ver) is None:
        P.log("the Stage-I point fails the verification; the campaign stops")
        (P.RUN / "FAILED.txt").write_text("Stage-I verification failed\n", "utf-8")
        return 2
    P.START = name
    P.push_inputs()
    split_local = args.local_workers * args.local_speed
    P.log("== 3. Stage II-a (W1, W3, W4, W6; early stop)")
    rep = V2.stage2a(w, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 4. Stage II-b")
    rep, dirs = V2.stage2b(rep, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 5. completion of the band on W2, ranking on all windows")
    rep = complete_band(rep, dirs, ops, args.local_workers, args.remote_workers, split_local)
    P.log("== 6. settling confirmation")
    win, c_res = V2.confirm(rep, args.confirm_workers)
    if win is None:
        (P.RUN / "FAILED.txt").write_text("no candidate passes the settling test\n", "utf-8")
        return 2
    P.log(f"SELECTED {win['key']}")
    P.log("== 7. W5 for the report, result, go criteria")
    w5 = P.report_w5(win)
    go = go_criteria(win, rep)
    write_result(win, rep, c_res, w5, ver, go, ops)
    (P.RUN / ("GO.json" if go["go"] else "NOGO.json")).write_text(
        json.dumps(go, indent=1, default=float), "utf-8")
    P.log(f"go criteria: {'GO' if go['go'] else 'NO GO'} {go['checks']}")
    (P.RUN / "DONE.txt").write_text(f"selected {win['key']}\n", "utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_11_procedure_v3")
    p.add_argument("--local-workers", type=int, default=30)
    p.add_argument("--remote-workers", type=int, default=36)
    p.add_argument("--verify-workers", type=int, default=36)
    p.add_argument("--confirm-workers", type=int, default=36)
    p.add_argument("--local-speed", type=float, default=0.7)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    P.START = V2.V1_START                  # operating points only, until Stage I is written
    if args.dry_run:
        P.RUN = ROOT / "results" / "_dry_ch9_11" / f"{datetime.now():%Y%m%d-%H%M%S}"
        P.RUN.mkdir(parents=True)
        return procedure(args)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            parameters={"procedure": PROCEDURE, "pi_q": PI_Q, "g_w_gen": G_W_GEN,
                        "d_engage_pu": D_ENGAGE, "theta_sts_band_pu": list(THETA_STS),
                        "theta_ts_max_pu": V2.THETA_TS_MAX, "search_windows": list(SEARCH),
                        "hunting_tolerance": TOLERANCE, "n_2a": V2.N_2A,
                        "rounds_max": P.MAX_ROUNDS, "result_paramset": RESULT,
                        "workers": f"{args.local_workers} here + {args.remote_workers} on "
                                   f"{P.REMOTE}"},
            time_series=("Stage-I verification and settling confirmation: 120 isolated "
                         "interface steps in W1-W4, W6; Stage II: 12-h simulations of W1, "
                         "W3, W4, W6 (early stop); band completed on W2; W5 report only"),
            note=args.note or "Ch 9 weight selection, procedure v3",
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
        trun.describe_output("selection/", "theta, engagement, unions, ch_9_3_select reports")
        trun.describe_output("complete/", "the band completed on W2")
        trun.describe_output("expC/", "settling confirmation in rank order")
        P.log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
