#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_13_coupler_hold.py
=============================================================
Does holding the STS couplers after a TS bank commit remove the tap reversals
that priced the cheaper banks out in run 0006, Stage II-c?

Run 0006 (procedure v3, worktree ``F:/qOFO_res``) tuned the bank price on its
own in Stage II-c.  The prices 14.1 and 17.9 were rejected on a second counted
tap reversal; every one of those reversals sits at the same instant as a bank
commit in the same STS, on a coupler that hosts no bank (W3 at 17280/20880 s,
W4 at 19440 s).  At a TS instant the runner writes the TS decisions and then
lets the STS decide on a measurement taken before the switch, so both answer
the same deviation.  ``sts_coupler_hold_after_bank = n`` holds all couplers
of the hosting STS for the next ``n`` STS steps (author decision 2026-11-04:
"Yes, test the hold").

Three legs through ``ch_9_3_stage2_screen.py`` with run 0006's II-c flags,
Stage-I paramset and operating points, on the ``minimal/thesis-core`` code:

``h0``  hold off, one rejected candidate on W3 and W4.  It must reproduce run
        0006; otherwise the checkouts differ in more than the hold.
``h1``  hold 1 (the stale step only), the three II-c candidates, W1/W3/W4/W6.
``h6``  hold 6 (= the 120 s coupler lock), the same.

Writes ``compare.json`` and ``compare.md`` into the run folder.  The verdict
uses the II-c pick rule as far as it is recorded per window (counted reversals
<= 1, no bank reversal, e_v,TS within 2 % of the leader in the same leg); the
weight-based θ bounds are unchanged by the hold.

This checkout's screen has no ``--startup-exempt-s`` (procedure v3, amendment
2, on ``res/commitment-base``).  The exemption is a scoring rule only, so every
row -- run 0006's included -- is re-scored here from its persisted ``moves``,
``shunt_moves`` and ``events_s`` with :func:`split_reversals`, a copy of v3's
``tuning.stage2_search._split_reversals``; re-scoring run 0006 must give its
stored counts.

Usage::

    python -m experiments.ch_09_parameter_selection.ch_9_13_coupler_hold \\
        --workers 10 --note "..."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection.ch_9_9_ledger_check import (  # noqa: E402
    METRICS, REL_TOL, diff, import_paramset, log, source_head,
)

SOURCE = Path(r"F:\qOFO_res\results\ch_09_parameter_selection"
              r"\0006_2026-11-03_211954_ch_9_11_procedure_v3")
HOURS = 12.0
SCREEN = "experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py"
#: Stage II-c candidates: the two rejected bank prices and the leader.
CANDIDATES = ("58bcdc27c248", "29580fc8a748", "8ad82442269b")
LEADER = "8ad82442269b"
#: (leg, hold, candidates, windows)
LEGS = (("h0", 0, ("58bcdc27c248",), "W3,W4"),
        ("h1", 1, CANDIDATES, "W1,W3,W4,W6"),
        ("h6", 6, CANDIDATES, "W1,W3,W4,W6"))
#: Bank columns of the TS-MIQP -> hosting STS (build order of
#: ``add_hv_networks``: MSC, MSR per DSO; checked on the build 2026-11-04).
BANK_STS = {"z2[0]": "DSO_1", "z2[1]": "DSO_1", "z2[2]": "DSO_2", "z2[3]": "DSO_2",
            "z2[4]": "DSO_3", "z2[5]": "DSO_3", "z3[0]": "DSO_4", "z3[1]": "DSO_4"}
EV_BAND = 0.02
#: v3 start-up exemption [s] (run 0006: ``--startup-exempt-s 900``).
STARTUP_S = 900.0
#: Simulation metrics the h0 leg must reproduce (the reversal counts are
#: compared after re-scoring, below).
SIM_METRICS = tuple(m for m in METRICS if m not in ("n_hunting", "n_shunt_hunting"))


def split_reversals(moves, events_s, startup_s: float = STARTUP_S):
    """``(hunting, event_explained, startup)``: v3's ``_split_reversals``.

    Two consecutive opposite moves of one device within ``HUNT_DWELL_S`` are a
    reversal; it is event-explained when a disturbance arrives in between, and
    start-up when it is the device's first two moves with the first before
    ``startup_s``; otherwise it counts as hunting.
    """
    from tuning.stage2_search import HUNT_DWELL_S, merge_events
    events_s = merge_events(events_s)
    per: Dict[str, List[tuple]] = {}
    for t, key, s in moves:
        per.setdefault(key, []).append((float(t), int(s)))
    hunt: List[dict] = []
    legit: List[dict] = []
    startup: List[dict] = []
    for key, seq in per.items():
        seq.sort()
        for i, ((t0, s0), (t1, s1)) in enumerate(zip(seq, seq[1:])):
            if s1 == s0 or (t1 - t0) > HUNT_DWELL_S:
                continue
            row = {"transformer": key, "t0_s": t0, "t1_s": t1, "dwell_s": t1 - t0}
            if any(t0 + 1.0 < te <= t1 + 1.0 for te in events_s):
                legit.append(row)
            elif i == 0 and t0 < startup_s:
                startup.append(row)
            else:
                hunt.append(row)
    return hunt, legit, startup


def rescore(row: dict) -> dict:
    """Counted tap and bank reversals of one window under the v3 exemption."""
    ev = row.get("events_s", [])
    hunt, _, st = split_reversals(row.get("moves", []), ev)
    bhunt, _, bst = split_reversals(row.get("shunt_moves", []), ev)
    return {"n_hunting": len(hunt), "hunting_detail": hunt, "n_startup_exempt": len(st),
            "n_shunt_hunting": len(bhunt), "shunt_hunting_detail": bhunt,
            "n_shunt_startup_exempt": len(bst)}


def source_rows(source: Path) -> Dict[str, dict]:
    out = {}
    for k in CANDIDATES:
        out[k] = json.loads((source / "stage2c" / "candidates" / f"cand_{k}.json")
                            .read_text("utf-8"))
    return out


def same_instant_coupler_moves(row: dict) -> List[dict]:
    """Coupler moves of the hosting STS at the instant of a bank commit."""
    out = []
    for t, bank, d in row.get("shunt_moves", []):
        sts = BANK_STS.get(bank)
        for tm, dev, dm in row.get("moves", []):
            if tm == t and sts and dev.startswith(sts + "|"):
                out.append({"t_s": t, "bank": bank, "bank_dir": d,
                            "coupler": dev, "coupler_dir": dm})
    return out


def window_summary(row: dict) -> dict:
    return {k: row.get(k) for k in ("e_v_ts", "e_v_ds", "shunt_commits", "shunt_down",
                                    "n_moves", "shunt_moves", "failed")} | rescore(row) | {
        "same_instant": same_instant_coupler_moves(row)}


def leg_rows(run: Path, leg: str, keys, windows: str) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for k in keys:
        for w in windows.split(","):
            f = run / leg / "windows" / f"{k}__{w}.json"
            if f.exists():
                out.setdefault(k, {})[w] = json.loads(f.read_text("utf-8"))["row"]
    return out


def verdict_for(rows: Dict[str, dict], leader_ev: Optional[float]) -> dict:
    """II-c pick rule on the recorded windows (θ bounds are weight-based)."""
    scored = [rescore(r) for r in rows.values()]
    n_h = sum(s["n_hunting"] for s in scored)
    n_bh = sum(s["n_shunt_hunting"] for s in scored)
    # quadratic mean over the windows, as ``tuning.stage2_search.aggregate``
    evs = [float(r["e_v_ts"]) for r in rows.values() if r.get("e_v_ts") is not None]
    ev = (sum(e * e for e in evs) / len(evs)) ** 0.5 if evs else None
    within = (leader_ev is not None and ev is not None
              and ev <= (1.0 + EV_BAND) * leader_ev)
    return {"counted_reversals": n_h, "bank_reversals": n_bh, "e_v_ts_mean": ev,
            "within_2pct_of_leader": within,
            "feasible": n_h <= 1 and n_bh == 0 and within,
            "bank_commits": sum(int(r.get("shunt_commits", 0)) for r in rows.values()),
            "failed_windows": [w for w, r in rows.items() if r.get("failed")]}


def compare(run: Path, src: Dict[str, dict]) -> dict:
    res: Dict[str, Any] = {"source_run": str(SOURCE), "legs": {}}
    # h0: reproduction of run 0006
    _, _, keys0, win0 = LEGS[0]
    got0 = leg_rows(run, "h0", keys0, win0)
    rep = {}
    for k in keys0:
        for w in win0.split(","):
            if w in got0.get(k, {}):
                a, b = src[k]["per_window"][w], got0[k][w]
                m = diff(a, b, SIM_METRICS)
                for f in ("moves", "shunt_moves"):
                    m[f] = {"source": len(a.get(f, [])), "check": len(b.get(f, [])),
                            "equal": a.get(f) == b.get(f)}
                ra, rb = rescore(a), rescore(b)
                for f in ("n_hunting", "n_shunt_hunting"):
                    m[f] = {"source": ra[f], "check": rb[f], "equal": ra[f] == rb[f]}
                rep[f"{k}/{w}"] = m
    # The port of the v3 exemption must give run 0006's stored counts.
    port = {f"{k}/{w}": {f: (r.get(f), rescore(r)[f])
                         for f in ("n_hunting", "n_shunt_hunting")}
            for k in CANDIDATES for w, r in src[k]["per_window"].items()}
    res["rescoring_port"] = {
        "per_window": port,
        "matches_run_0006": all(a == b for d in port.values() for a, b in d.values())}
    res["reproduction"] = {
        "windows": rep,
        "reproduced": len(rep) == len(keys0) * len(win0.split(","))
                      and all(v["equal"] for m in rep.values() for v in m.values())}
    # source (run 0006, hold off) and the hold legs, same aggregation
    legs = {"0006": {k: src[k]["per_window"] for k in CANDIDATES}}
    for leg, hold, keys, wins in LEGS[1:]:
        legs[leg] = leg_rows(run, leg, keys, wins)
    for name, rows in legs.items():
        ev_leader = None
        if LEADER in rows and rows[LEADER]:
            ev_leader = verdict_for(rows[LEADER], None)["e_v_ts_mean"]
        res["legs"][name] = {
            k: {"verdict": verdict_for(r, ev_leader),
                "windows": {w: window_summary(x) for w, x in r.items()}}
            for k, r in rows.items()}
    return res


def write_md(run: Path, res: dict, src: Dict[str, dict]) -> None:
    price = {k: src[k]["weights"]["g_w_tso_shunt"] for k in CANDIDATES}
    L = ["# STS coupler hold after a TS bank commit -- run 0006 Stage II-c candidates", "",
         f"- Source: `{res['source_run']}` (stage2c)",
         f"- Re-scoring with the v3 start-up exemption reproduces run 0006's counts: "
         f"**{res['rescoring_port']['matches_run_0006']}**",
         f"- Reproduction of run 0006 with the hold off: "
         f"**{res['reproduction']['reproduced']}**", ""]
    bad = [(kw, n, v) for kw, m in res["reproduction"]["windows"].items()
           for n, v in m.items() if not v["equal"]]
    if bad:
        L += ["| unit | metric | run 0006 | here | delta |", "|---|---|---:|---:|---:|"]
        L += [f"| {kw} | {n} | {v['source']} | {v['check']} | {v.get('delta')} |"
              for kw, n, v in bad]
        L.append("")
    L += ["## II-c pick rule per leg (W1, W3, W4, W6)", "",
          "| candidate | bank price | leg | counted reversals | bank reversals | "
          "e_v,TS | within 2 % | bank commits | feasible |",
          "|---|---:|---|---:|---:|---:|---|---:|---|"]
    for k in CANDIDATES:
        for leg in ("0006", "h1", "h6"):
            c = res["legs"].get(leg, {}).get(k)
            if not c:
                L.append(f"| `{k}` | {price[k]:.2f} | {leg} | missing | | | | | |")
                continue
            v = c["verdict"]
            ev = f"{v['e_v_ts_mean']:.5f}" if v["e_v_ts_mean"] is not None else "-"
            L.append(f"| `{k}` | {price[k]:.2f} | {leg} | {v['counted_reversals']} | "
                     f"{v['bank_reversals']} | {ev} | {v['within_2pct_of_leader']} | "
                     f"{v['bank_commits']} | {v['feasible']} |")
    L += ["", "## Reversals and same-instant coupler moves per window", ""]
    for k in CANDIDATES:
        for leg in ("0006", "h1", "h6"):
            c = res["legs"].get(leg, {}).get(k)
            if not c:
                continue
            for w, s in c["windows"].items():
                if not (s["hunting_detail"] or s["same_instant"] or s["shunt_moves"]):
                    continue
                hd = "; ".join(f"{h['transformer']} {h['t0_s']:.0f}->{h['t1_s']:.0f} s"
                               for h in s["hunting_detail"] or []) or "none"
                si = "; ".join(f"{m['bank']}{m['bank_dir']:+d} & {m['coupler']}"
                               f"{m['coupler_dir']:+d} @ {m['t_s']:.0f} s"
                               for m in s["same_instant"]) or "none"
                bm = ", ".join(f"{b}{d:+d}@{t:.0f}" for t, b, d in s["shunt_moves"])
                L.append(f"- `{k}` {leg} {w}: reversals {hd}; same-instant {si}; "
                         f"banks {bm or 'none'}")
    (run / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def procedure(run: Path, args) -> int:
    src = source_rows(args.source)
    ops = run / "operating_points"
    ops.mkdir(exist_ok=True)
    n_ops = 0
    for f in sorted((args.source / "operating_points").glob("op_*.npz")):
        shutil.copy2(f, ops / f.name)
        n_ops += 1
    log(run, f"{n_ops} operating points copied; {args.paramset_note}; source checkout "
             f"at {source_head(args.source)}")

    units = {leg: len(keys) * len(wins.split(",")) for leg, _, keys, wins in LEGS}
    total = sum(units.values())
    procs = []
    left = args.workers
    for i, (leg, hold, keys, wins) in enumerate(LEGS):
        n = left if i == len(LEGS) - 1 else max(1, round(args.workers * units[leg] / total))
        n = max(1, min(n, units[leg], left - (len(LEGS) - 1 - i)))
        left -= n
        d = run / leg
        d.mkdir(exist_ok=True)
        cj = d / "candidates_in.json"
        cj.write_text(json.dumps([src[k]["weights"] for k in keys], indent=1), "utf-8")
        argv = [SCREEN, "--paramset", args.paramset, "--frozen-network",
                "--hours", f"{HOURS:g}", "--evented", "--v-metric", "bus",
                "--windows", wins, "--ops-dir", str(ops), "--op-horizon-h", f"{HOURS:g}",
                "--workers", str(n), "--candidates-json", str(cj), "--out-dir", str(d),
                "--skip-baseline",
                "--config-set", f"sts_coupler_hold_after_bank={hold}"]
        log(run, f"start ({leg}, {units[leg]} units, {n} workers): " + " ".join(argv))
        fh = open(d / "screen.log", "w", encoding="utf-8")
        procs.append((leg, fh, subprocess.Popen(
            [sys.executable, "-X", "utf8", *argv], cwd=ROOT, stdout=fh,
            stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1"))))
    rc_all = 0
    for leg, fh, proc in procs:
        rc = proc.wait()
        fh.close()
        log(run, f"exit {rc}: {leg}/screen.log")
        if rc not in (0, 2):              # 2 = screen finished with a failed unit
            rc_all = 1
    built = sorted(f.name for f in ops.glob("op_*.npz"))
    if len(built) != n_ops:
        log(run, f"WARNING: the screen built operating points of its own: {built}")
    res = compare(run, src)
    (run / "compare.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    write_md(run, res, src)
    log(run, f"reproduced: {res['reproduction']['reproduced']}")
    for leg in ("0006", "h1", "h6"):
        for k, c in res["legs"].get(leg, {}).items():
            v = c["verdict"]
            log(run, f"{leg} {k}: reversals {v['counted_reversals']}, bank reversals "
                     f"{v['bank_reversals']}, e_v_ts {v['e_v_ts_mean']}, "
                     f"feasible {v['feasible']}")
    return rc_all


def main(argv: Optional[List[str]] = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_13_coupler_hold")
    p.add_argument("--source", type=Path, default=SOURCE, help="run 0006's folder")
    p.add_argument("--workers", type=int, default=10,
                   help="workers of all legs together (default 10)")
    p.add_argument("--resume", type=Path, default=None)
    add_run_arguments(p)
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    meta = sorted((args.source / "stage2c").glob("*/run_meta.json"))
    if not meta:
        raise SystemExit(f"{args.source / 'stage2c'}: no run_meta.json")
    args.paramset = json.loads(meta[-1].read_text("utf-8"))["paramset"]
    args.paramset_note = import_paramset(args.paramset, args.source)
    print(args.paramset_note, flush=True)
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=[],
            paramset=args.paramset,
            parameters={"source_run": str(args.source),
                        "legs": {leg: {"sts_coupler_hold_after_bank": h,
                                       "candidates": list(k), "windows": w}
                                 for leg, h, k, w in LEGS},
                        "hours": HOURS, "workers": args.workers},
            time_series=("W1, W3, W4, W6 of tuning.scenarios_mc_v2.CONDITIONS, 12 h, one "
                         "disturbance each, evented, frozen network; T_TS 180 s, "
                         "T_STS 20 s (as run 0006 Stage II-c)"),
            note=args.note or "STS coupler hold after a TS bank commit (diagnostic)",
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
        trun.describe_output("h0/", "hold off, 58bcdc27c248 W3/W4: reproduction of run 0006")
        trun.describe_output("h1/", "hold 1 STS step, three II-c candidates, W1/W3/W4/W6")
        trun.describe_output("h6/", "hold 6 STS steps, the same")
        trun.describe_output("compare.json", "reproduction, II-c pick rule per leg, "
                                             "reversals and same-instant moves")
        trun.describe_output("compare.md", "the same, readable")
        log(run, f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
