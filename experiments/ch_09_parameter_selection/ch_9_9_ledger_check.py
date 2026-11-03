#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_9_ledger_check.py
============================================================
Does the weight selection of run 0004 hold with the shunt-feedforward ledger?

Run 0004 (procedure v2, ``ch_9_8_procedure_v2.py``) simulates its candidates on
``res/commitment-base-2026-10-31``, which predates ``ac69918``: a bank release
there is evaluated on the hosting STS's pre-switch model and comes out about
7 % too large (``shunt_ff_ledger = False``).  Every campaign after the
selection -- D1/D2, Case 1, the Case-2 REF and MC -- runs on
``minimal/thesis-core`` with the ledger on.  Author decision 2026-11-03: keep
the ledger on and check that the selection holds.

The ledger changes nothing until a bank level is released: an engagement adds
the same feedforward with or without it.  A window without a release therefore
runs identically under both settings, and only windows with a release need to
be simulated again.  In run 0004 no feasible candidate releases a bank (status
after II-b round 3), so the ledger can change the selection only by admitting a
rejected candidate whose rejection came from a release.

Two legs, both through ``ch_9_3_stage2_screen.py`` (the producer of run 0004's
windows) with run 0004's flags, Stage-I paramset and operating points:

``off``  the selected candidate, all windows, ``--config-set
         shunt_ff_ledger=false``.  It must reproduce run 0004.  If it does not,
         another code difference between the two checkouts changes the closed
         loop, and the copied windows of the ``on`` leg are not valid.
``on``   every candidate of run 0004's final union whose TS deviation lies
         within ``--band`` of the best feasible one, and the Stage-I point, on
         this checkout's default configuration (ledger on).  Windows with a bank
         release are simulated; the others are copied from run 0004 and marked
         ``copied_from``.  Re-selected with ``ch_9_3_select.py`` under run
         0004's limits.

Experiment C is not repeated: it holds the TS controllers, so no bank switches.
If another candidate ranks first with the ledger on, it has not passed the
settling confirmation, and the verdict says so.

Usage (after run 0004 has written its final selection)::

    python -m experiments.ch_09_parameter_selection.ch_9_9_ledger_check \\
        --workers 11 --note "..."

Writes ``compare.json`` and ``compare.md`` into the run folder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = Path(r"F:\qOFO_res\results\ch_09_parameter_selection"
              r"\0004_2026-11-02_091930_ch_9_8_procedure_v2")
WINDOWS = "W1,W2,W3,W4,W6"
HOURS = 12.0
BANK_BUDGET = 12.0
SCREEN = "experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py"
SELECT = "experiments/ch_09_parameter_selection/ch_9_3_select.py"
#: Window metrics the ``off`` leg must reproduce.
METRICS = ("e_v_ts", "e_v_ds", "e_q", "v_worst_ts", "v_worst_ds",
           "tap_ops_per_h_tso", "tap_ops_per_h_dso", "avr_max_step_pu",
           "n_hunting", "n_shunt_hunting", "shunt_commits", "shunt_down",
           "shunt_ops_max_bank", "pf_failures", "failed")
#: Aggregates compared per candidate (``tuning.stage2_search.aggregate``).
AGGREGATES = ("e_v_ts", "e_v_ds", "e_q", "n_hunting", "n_shunt_hunting",
              "worst_shunt_ops_per_day_bank", "worst_tap_ops_per_day_tso",
              "worst_tap_ops_per_day_dso", "worst_avr_max_step_pu", "shunt_commits")
REL_TOL = 1e-9


def log(run: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run / "driver.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_py(run: Path, argv: List[str], logfile: Path) -> None:
    cmd = [sys.executable, "-X", "utf8", *argv]
    log(run, "run: " + " ".join(argv) + f"  > {logfile.name}")
    env = dict(os.environ, PYTHONUTF8="1")
    with open(logfile, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    log(run, f"exit {rc}: {logfile.name}")
    if rc not in (0, 2):              # 2 = screen finished with a failed unit
        raise RuntimeError(f"{argv[0]} exited {rc}; see {logfile}")


# -- run 0004 ----------------------------------------------------------------------------

def final_selection(source: Path) -> Tuple[Path, dict]:
    """Run 0004's last selection report: highest II-b round, lowest hunting tolerance."""
    best: Optional[Tuple[Tuple[int, int, int], Path]] = None
    for f in (source / "selection").glob("selection_*_h*.json"):
        m = re.fullmatch(r"selection_(2a|2b_round(\d+))_h(\d+)\.json", f.name)
        if not m:
            continue
        if not json.loads(f.read_text("utf-8")).get("ranked"):
            continue
        rank = (1 if m.group(2) else 0, int(m.group(2) or 0), -int(m.group(3)))
        if best is None or rank > best[0]:
            best = (rank, f)
    if best is None:
        raise SystemExit(f"{source / 'selection'}: no selection report with a ranking")
    return best[1], json.loads(best[1].read_text("utf-8"))


def settling_failures(source: Path) -> List[str]:
    f = source / "expC" / "criterion_results.json"
    if not f.exists():
        return []
    return sorted(k for k, v in json.loads(f.read_text("utf-8")).items() if not v["pass"])


def selected_key(source: Path, ranked: List[dict], failed: List[str]) -> Tuple[str, str]:
    done = source / "DONE.txt"
    if done.exists():
        m = re.search(r"selected\s+([0-9a-f]{12})", done.read_text("utf-8"))
        if m:
            return m.group(1), "DONE.txt"
    for r in ranked:
        if r["key"] not in failed:
            return r["key"], "first ranked without a settling failure (run 0004 not done)"
    raise SystemExit("no candidate left to check")


def union_candidates(union: Path) -> Dict[str, dict]:
    return {c["key"]: c for c in (json.loads(f.read_text("utf-8"))
                                  for f in sorted(union.glob("cand_*.json")))}


def releases(cand: dict) -> List[str]:
    """Windows in which run 0004 released a bank level in this candidate."""
    return [w for w, r in cand["per_window"].items() if int(r.get("shunt_down", 0)) > 0]


def import_paramset(name: str, source: Path) -> str:
    """Bring run 0004's Stage-I variant into this checkout if it is missing.

    Run 0004 writes its variants untracked into its own checkout.  An existing
    file of the same name must be identical: a parameter set is never replaced.
    """
    from configs.paramsets import PARAMSET_DIR
    here = PARAMSET_DIR / f"{name}.json"
    there = source.parents[2] / "configs" / "paramsets" / f"{name}.json"
    if here.exists():
        if there.exists() and here.read_bytes() != there.read_bytes():
            raise SystemExit(f"{here} differs from {there}")
        return f"paramset {name} present here"
    if not there.exists():
        raise SystemExit(f"paramset {name} is in neither checkout")
    shutil.copy2(there, here)
    return f"paramset {name} imported from {there}"


def source_head(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(source.parents[2]),
             "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _stage1_point(source: Path) -> str:
    """The Stage-I variant run 0004 searched around (``--paramset`` of its screens)."""
    for f in sorted(source.glob("stage2b/round*/*/run_meta.json")) + \
            sorted(source.glob("stage2a/*/run_meta.json")):
        return json.loads(f.read_text("utf-8"))["paramset"]
    raise SystemExit(f"{source}: no screen run_meta.json names the Stage-I paramset")


# -- comparison --------------------------------------------------------------------------

def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def diff(a: Dict[str, Any], b: Dict[str, Any], names) -> Dict[str, Dict[str, Any]]:
    out = {}
    for n in names:
        x, y = _num(a.get(n)), _num(b.get(n))
        if x is None or y is None:
            out[n] = {"source": a.get(n), "check": b.get(n), "equal": a.get(n) == b.get(n)}
            continue
        d = y - x
        out[n] = {"source": x, "check": y, "delta": d,
                  "rel": d / x if x else None,
                  "equal": abs(d) <= REL_TOL * max(1.0, abs(x))}
    return out


def compare(run: Path, source: Path, src_sel_path: Path, src_rep: dict,
            union: Dict[str, dict], checked: List[str], resim: Dict[str, List[str]],
            off_key: str, winner: str, winner_how: str, failed: List[str]) -> dict:
    """``off_key``: the candidate of the off leg (the leader at launch); ``winner``
    and ``failed``: run 0004's selection and settling failures at the end of the
    check, which may differ when its confirmation was still running at launch."""
    src_rank = [r["key"] for r in src_rep["ranked"]]
    src_why = {r["key"]: r["why"] for r in src_rep.get("rejected", [])}
    on_dir, off_dir = run / "on", run / "off"
    out: Dict[str, Any] = {
        "source_run": str(source), "source_selection": src_sel_path.name,
        "winner": winner, "winner_from": winner_how, "off_leg_candidate": off_key,
        "settling_failures": failed,
        "checked": checked, "resimulated_windows": resim,
    }

    # off leg: reproduction of run 0004's windows (any candidate without a release
    # shows whether the two checkouts differ in more than the ledger)
    off = {}
    for w, row in union[off_key]["per_window"].items():
        f = off_dir / "windows" / f"{off_key}__{w}.json"
        if f.exists():
            off[w] = diff(row, json.loads(f.read_text("utf-8"))["row"], METRICS)
    out["off_leg"] = {
        "windows": off,
        "reproduced": len(off) == len(union[off_key]["per_window"])
                      and all(v["equal"] for m in off.values() for v in m.values()),
    }

    # on leg: the re-simulated candidates, and the re-selection over the band
    sel_on = json.loads((on_dir / "selection_h0.json").read_text("utf-8"))
    on_rank = [r["key"] for r in sel_on.get("ranked", [])]
    on_why = {r["key"]: r["why"] for r in sel_on.get("rejected", [])}
    cands = {}
    for k in [winner] + [k for k in resim if k != winner]:
        f = on_dir / "candidates" / f"cand_{k}.json"
        if not f.exists():
            cands[k] = {"missing": True}
            continue
        c_on = json.loads(f.read_text("utf-8"))
        cands[k] = {
            "aggregate": diff(union[k]["aggregate"], c_on["aggregate"], AGGREGATES),
            "resimulated": resim.get(k, []),
            "feasible_source": k in src_rank, "rejected_source": src_why.get(k),
            "feasible_on": k in on_rank, "rejected_on": on_why.get(k),
        }
    admitted = [k for k in resim if k in on_rank and k not in src_rank]
    # Candidates that failed run 0004's settling confirmation cannot be selected.
    eligible = [k for k in on_rank if k not in failed]
    out["on_leg"] = {
        "candidates": cands,
        "admitted_by_ledger": admitted,
        "ranking_on": on_rank[:10],
        "ranking_on_without_settling_failures": eligible[:10],
        "ranking_source_restricted": [k for k in src_rank if k in checked][:10],
        "winner_first_on": bool(eligible) and eligible[0] == winner,
    }
    provisional = "" if winner_how == "DONE.txt" else " (run 0004 not finished: provisional)"
    if not out["off_leg"]["reproduced"]:
        verdict = ("INCONCLUSIVE: the ledger-off leg does not reproduce run 0004, so the "
                   "two checkouts differ in more than the ledger")
    elif winner not in checked:
        verdict = (f"INCONCLUSIVE: the selected candidate {winner} lies outside the checked "
                   f"band; rerun with a larger --band")
    elif out["on_leg"]["winner_first_on"]:
        verdict = ("HOLDS: with the ledger on, the selected candidate stays feasible and "
                   "first" + provisional)
    elif winner in eligible:
        verdict = (f"CHANGES: {eligible[0]} ranks first with the ledger on (the selected "
                   f"candidate ranks {eligible.index(winner) + 1}); {eligible[0]} has no "
                   f"passed settling confirmation" + provisional)
    else:
        verdict = ("CHANGES: the selected candidate is infeasible with the ledger on"
                   + provisional)
    out["verdict"] = verdict
    return out


def write_md(run: Path, res: dict) -> None:
    n_units = sum(len(v) for v in res["resimulated_windows"].values())
    L = [f"# Ledger check of run 0004 -- {res['verdict']}", "",
         f"- Source: `{res['source_run']}` ({res['source_selection']})",
         f"- Selected candidate: `{res['winner']}` ({res['winner_from']}); off leg on "
         f"`{res['off_leg_candidate']}`; settling failures {res['settling_failures']}",
         f"- Re-selected over {len(res['checked'])} candidates inside the band; "
         f"{len(res['resimulated_windows'])} of them had a bank release, "
         f"{n_units} windows re-simulated with the ledger on", "",
         "## Ledger off: reproduction of run 0004 (off-leg candidate)", "",
         f"Reproduced: **{res['off_leg']['reproduced']}**", ""]
    bad = [(w, n, v) for w, m in res["off_leg"]["windows"].items()
           for n, v in m.items() if not v["equal"]]
    if bad:
        L += ["| window | metric | run 0004 | check | delta |", "|---|---|---:|---:|---:|"]
        L += [f"| {w} | {n} | {v['source']} | {v['check']} | {v.get('delta')} |"
              for w, n, v in bad]
    L += ["", "## Ledger on", "",
          f"Ranking with the ledger on (first 10): {res['on_leg']['ranking_on']}",
          f"Run 0004, same candidates: {res['on_leg']['ranking_source_restricted']}",
          f"Admitted by the ledger: {res['on_leg']['admitted_by_ledger'] or 'none'}", "",
          "| candidate | re-simulated | feasible 0004/on | e_v_ts 0004 | e_v_ts on "
          "| e_v_ds 0004 | e_v_ds on | hunting 0004/on | bank hunting 0004/on |",
          "|---|---|---|---:|---:|---:|---:|---|---|"]
    for k, c in res["on_leg"]["candidates"].items():
        if c.get("missing"):
            L.append(f"| `{k}` | missing | | | | | | | |")
            continue
        a = c["aggregate"]
        L.append(f"| `{k}` | {','.join(c['resimulated']) or '-'} | "
                 f"{c['feasible_source']}/{c['feasible_on']} | "
                 f"{a['e_v_ts']['source']:.5f} | {a['e_v_ts']['check']:.5f} | "
                 f"{a['e_v_ds']['source']:.5f} | {a['e_v_ds']['check']:.5f} | "
                 f"{a['n_hunting']['source']:g}/{a['n_hunting']['check']:g} | "
                 f"{a['n_shunt_hunting']['source']:g}/{a['n_shunt_hunting']['check']:g} |")
    (run / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# -- main --------------------------------------------------------------------------------

def plan(source: Path, band: float) -> dict:
    """Which candidates and windows the check covers; nothing is simulated here."""
    from tuning.parameters import weight_fingerprint
    src_sel_path, src_rep = final_selection(source)
    tag = src_sel_path.stem.removeprefix("selection_").rsplit("_h", 1)[0]
    union = union_candidates(source / "selection" / f"union_{tag}")
    for k, c in union.items():
        if weight_fingerprint(c["weights"]) != k:
            raise SystemExit(f"{k}: the weights do not reproduce the key")
    failed = settling_failures(source)
    winner, how = selected_key(source, src_rep["ranked"], failed)
    best = min(r["aggregate"]["e_v_ts"] for r in src_rep["ranked"])
    base = src_rep["baseline"]["key"]
    checked = [k for k, c in union.items()
               if c["aggregate"]["e_v_ts"] <= best * (1.0 + band) and k != base]
    if winner not in checked:
        checked.insert(0, winner)
    resim = {k: releases(union[k]) for k in checked + [base] if releases(union[k])}
    return {"sel_path": src_sel_path, "rep": src_rep, "union": union, "winner": winner,
            "how": how, "failed": failed, "best": best, "base": base,
            "checked": checked, "resim": resim}


def procedure(run: Path, args) -> int:
    P = plan(args.source, args.band)
    union, winner = P["union"], P["winner"]
    log(run, f"source selection {P['sel_path'].name}: winner {winner} ({P['how']}); "
             f"settling failures so far {P['failed']}")
    log(run, f"band {args.band:g} around e_v_ts {P['best']:.5f}: {len(P['checked'])} "
             f"candidates, {len(P['resim'])} with a release, "
             f"{sum(len(v) for v in P['resim'].values())} windows to re-simulate")

    ops = run / "operating_points"
    ops.mkdir(exist_ok=True)
    n_ops = 0
    for f in sorted((args.source / "operating_points").glob("op_*.npz")):
        shutil.copy2(f, ops / f.name)
        n_ops += 1
    log(run, f"{n_ops} operating points copied; {args.paramset_note}; source checkout "
             f"at {source_head(args.source)}")

    # Windows the ledger cannot change are taken from run 0004; the screen skips
    # every unit whose output exists.
    on_win = run / "on" / "windows"
    on_win.mkdir(parents=True, exist_ok=True)
    for k in P["checked"] + [P["base"]]:
        c = union[k]
        for w, row in c["per_window"].items():
            if w in P["resim"].get(k, []):
                continue
            (on_win / f"{k}__{w}.json").write_text(json.dumps(
                {"key": k, "window": w, "hours": c["hours"], "weights": c["weights"],
                 "lambda_max_B": c["lambda_max_B"], "is_baseline": bool(c["is_baseline"]),
                 "copied_from": f"{args.source.name}/{P['sel_path'].name} union "
                                f"(no bank release: the ledger cannot act)",
                 "row": row}, indent=1, default=float), encoding="utf-8")

    # Both legs at once, workers split in proportion to their units.
    common = ["--paramset", args.paramset, "--frozen-network", "--hours", f"{HOURS:g}",
              "--evented", "--v-metric", "bus", "--windows", WINDOWS,
              "--ops-dir", str(ops), "--op-horizon-h", f"{HOURS:g}"]
    u_off = len(WINDOWS.split(","))
    u_on = sum(len(v) for v in P["resim"].values())
    n_off = max(1, min(u_off, round(args.workers * u_off / (u_off + u_on)),
                       args.workers - 1))
    legs = (("off", [winner], n_off,
             ["--skip-baseline", "--config-set", "shunt_ff_ledger=false"]),
            ("on", P["checked"], max(1, args.workers - n_off), []))
    procs = []
    for leg, keys, workers, extra in legs:
        d = run / leg
        d.mkdir(exist_ok=True)
        cj = d / "candidates_in.json"
        cj.write_text(json.dumps([union[k]["weights"] for k in keys], indent=1), "utf-8")
        argv = [SCREEN, *common, "--workers", str(workers), "--candidates-json", str(cj),
                "--out-dir", str(d), *extra]
        log(run, f"start ({leg}): " + " ".join(argv) + "  > screen.log")
        fh = open(d / "screen.log", "w", encoding="utf-8")
        procs.append((leg, fh, subprocess.Popen(
            [sys.executable, "-X", "utf8", *argv], cwd=ROOT, stdout=fh,
            stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1"))))
    for leg, fh, proc in procs:
        rc = proc.wait()
        fh.close()
        log(run, f"exit {rc}: {leg}/screen.log")
        if rc not in (0, 2):
            raise RuntimeError(f"screen ({leg}) exited {rc}; see {run / leg / 'screen.log'}")
    built = sorted(f.name for f in ops.glob("op_*.npz"))
    if len(built) != n_ops:
        log(run, f"WARNING: the screen built operating points of its own: {built}")
    run_py(run, [SELECT, "--candidates", str(run / "on" / "candidates"),
                 "--hours", f"{HOURS:g}", "--shunt-ops-per-day", f"{BANK_BUDGET:g}",
                 "--max-hunting", "0", "--out", str(run / "on" / "selection_h0.json")],
           run / "on" / "select_h0.log")

    # Run 0004 may have finished its settling confirmation meanwhile.
    failed = settling_failures(args.source)
    final, how = selected_key(args.source, P["rep"]["ranked"], failed)
    log(run, f"run 0004 now: selected {final} ({how}); settling failures {failed}")
    res = compare(run, args.source, P["sel_path"], P["rep"], union, P["checked"],
                  P["resim"], winner, final, how, failed)
    res["band"] = args.band
    res["operating_points_rebuilt"] = len(built) != n_ops
    (run / "compare.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    write_md(run, res)
    log(run, res["verdict"])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_9_ledger_check")
    p.add_argument("--source", type=Path, default=SOURCE, help="run 0004's folder")
    p.add_argument("--band", type=float, default=0.02,
                   help="candidates with e_v_ts <= (1 + band) x the best feasible one "
                        "(0.02 = the indifference band of the selection rule)")
    p.add_argument("--workers", type=int, default=8,
                   help="workers of both legs together (default 8: with the 2 of run "
                        "0004's W5 report, e5wimi's budget of 10)")
    p.add_argument("--paramset", default=None,
                   help="Stage-I point (default: read from run 0004's screens)")
    p.add_argument("--plan-only", action="store_true",
                   help="print what would be simulated; no run folder, no simulation")
    p.add_argument("--resume", type=Path, default=None)
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    if args.plan_only:
        P = plan(args.source, args.band)
        print(f"{P['sel_path'].name}: winner {P['winner']} ({P['how']}); band "
              f"{args.band:g} around {P['best']:.5f}: {len(P['checked'])} candidates; "
              f"re-simulate {sum(len(v) for v in P['resim'].values())} windows:")
        for k, ws in P["resim"].items():
            a = P["union"][k]["aggregate"]
            print(f"  {k}  e_v_ts {a['e_v_ts']:.5f}  e_v_ds {a['e_v_ds']:.5f}  {ws}")
        return 0
    args.paramset = args.paramset or _stage1_point(args.source)
    args.paramset_note = import_paramset(args.paramset, args.source)
    print(args.paramset_note, flush=True)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved"],
            paramset=args.paramset,
            parameters={"source_run": str(args.source), "band": args.band,
                        "windows": WINDOWS, "hours": HOURS,
                        "legs": "off = selected candidate, shunt_ff_ledger false, all "
                                "windows; on = candidates inside the band, ledger on, "
                                "windows with a bank release re-simulated, others copied",
                        "workers": args.workers},
            time_series=("W1-W4, W6 of tuning.scenarios_mc_v2.CONDITIONS, 12 h, one "
                         "disturbance each, evented, frozen network; T_TS 180 s, "
                         "T_STS 20 s (as run 0004)"),
            note=args.note or "Shunt-feedforward ledger check of the run-0004 selection",
            tag=args.tag, resume=args.resume) as trun:
        run = trun.root
        t0 = time.time()
        try:
            rc = procedure(run, args)
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            log(run, f"driver error: {type(exc).__name__}: {exc}")
            (run / "FAILED.txt").write_text(traceback.format_exc(), "utf-8")
            rc = 1
        trun.describe_output("driver.log", "every stage with times")
        trun.describe_output("off/", "selected candidate, ledger off: reproduction of run 0004")
        trun.describe_output("on/", "candidates inside the band, ledger on (windows without "
                                    "a release copied from run 0004); selection_h0.json")
        trun.describe_output("compare.json", "per-window and per-candidate differences, verdict")
        trun.describe_output("compare.md", "the same, readable")
        log(run, f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
