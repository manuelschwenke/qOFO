#!/usr/bin/env python3
"""
experiments/ch_11_case2_multisystem/step/run_step.py
====================================================

Step tests of Case Study 2 (thesis ``ch:case3:step``), the campaign of
``docs/architecture/2026-10-28_case2_step_mc_plan.md`` §1-4, adopted by the
author on 2026-10-28 with the recommended options.  Every case starts from a
settled operating point with **held profiles** (``hold_profiles``): the test
disturbance is the only exogenous change.  Operating points and ramp buses
come from ``ops.json`` (``pick_ops.py``).

Parts
-----
R  load ramps (voltage tracking during a ramp).  Per OP:
   R-a, per area: +300 Mvar purely reactive at the area's ramp bus, ramp
   1800 s (10 Mvar/min) or 600 s (30 Mvar/min), up -- hold 30 min -- down at
   the same rate -- hold 30 min; R-b: every transmission load +10 % P and Q
   over 60 min, then hold 30 min.  All six variants.  21 cases.
P  event-phase sweep (the withdrawal between dispatches).  Per OP: the REF
   load step (250 Mvar at bus 11), the G4 trip, 250 Mvar at the area-1 and
   area-3 ramp buses, the tie 17-27 trip; at {20, 60, 140, 160} s after a
   dispatch (100 s is the E case); reversed 30 min later at the same phase.
   O2, O3, S2.  60 cases.
E  event battery, the Ch 7 N-1 catalogue (5 machines, the admissible lines,
   +250 Mvar at each transmission load bus), 100 s after a dispatch, reversed
   30 min later.  All six variants.  3 x 58 cases.

Every run settles for 30 min first (``SETTLE_S``).  One measurement-noise
realisation for all cases (``NOISE_SEED``): the persistent channel biases are
then the same in every case, so differences between cases and phases are not
noise draws; the MC varies the noise.

Each (case, variant) run writes ``runs/<case>/<variant>/`` -- ``log.pkl`` (full),
``status.json``, ``metrics.json`` (window metrics of ``metrics_step``).  The
pool runs the parts in the order R, P, E, the slowest variant first; a call
with ``--run-dir`` continues a campaign and skips every run with a status.

    python -m experiments.ch_11_case2_multisystem.step.run_step --jobs 30
    python -m experiments.ch_11_case2_multisystem.step.run_step --run-dir <run>
    python -m experiments.ch_11_case2_multisystem.step.run_step --smoke --jobs 6

Author: Manuel Schwenke / Claude Code (2026-10-28)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import contextlib
import dataclasses
import datetime as _dt
import json
import math
import pickle
import socket
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.step import metrics_step as ms
from experiments.helpers.records import ContingencyEvent
from experiments.thesis_runs import add_run_arguments, start_thesis_run

TARGET = "ch:case3:step"
FEEDS = ("tab:case3:step", "fig:case3:step:ramp", "fig:case3:step:phase",
         "fig:case3:step:recovery")
OPS_FILE = Path(__file__).resolve().parent / "ops.json"

T_TS = 180.0
SETTLE_S = 1800.0
PHASE_E_S = 100.0
PHASES_P_S = (20.0, 60.0, 140.0, 160.0)
EVENT_HOLD_S = 1800.0
TAIL_S = 1800.0
RAMP_Q_MVAR = 300.0
RAMP_RATES_S = {"slow": 1800.0, "fast": 600.0}
RAMP_HOLD_S = 1800.0
TS_RAMP_REL, TS_RAMP_S = 0.10, 3600.0
STEP_Q_MVAR = 250.0
NOISE_SEED = 20261028

#: Elements of the phase sweep (the REF events and the ramp-bus load steps).
P_GEN, P_LINE, P_LOAD_AREA2 = 2, 25, 11
ALL = ["L2", "S1", "S2", "O2", "O3", "M"]
P_VARIANTS = ["O2", "O3", "S2"]
#: Wall minutes per simulated hour on e5power (run 0009, 6 parallel): pool order.
COST = {"S1": 37.0, "L2": 11.5, "O2": 7.5, "O3": 3.2, "S2": 2.8, "M": 2.1}
PART_ORDER = {"R": 0, "P": 1, "E": 2}


# ---------------------------------------------------------------------------
#  Cases
# ---------------------------------------------------------------------------

def _ev(kind: str, idx: int, t: float, action: str) -> ContingencyEvent:
    if kind == "load":
        return ContingencyEvent(minute=int(t // 60), time_s=t, element_type="load",
                                bus=int(idx), p_mw=0.0, q_mvar=STEP_Q_MVAR,
                                action=action)
    return ContingencyEvent(minute=int(t // 60), time_s=t, element_type=kind,
                            element_index=int(idx), action=action)


def _event_pair(kind: str, idx: int, t_e: float, hold_s: float) -> List[ContingencyEvent]:
    on, off = {"gen": ("trip", "restore"), "line": ("trip", "restore"),
               "load": ("connect", "trip")}[kind]
    return [_ev(kind, idx, t_e, on), _ev(kind, idx, t_e + hold_s, off)]


def _minutes(end_s: float) -> int:
    return int(math.ceil(end_s / 60.0))


def build_cases(ops: Dict[str, Any], elements: Dict[str, List[int]],
                smoke: bool = False) -> List[Dict[str, Any]]:
    """Every case of the campaign (or a short smoke subset)."""
    settle = 120.0 if smoke else SETTLE_S
    hold = 240.0 if smoke else EVENT_HOLD_S
    tail = 240.0 if smoke else TAIL_S
    rates = {"fast": 120.0} if smoke else RAMP_RATES_S
    r_hold = 120.0 if smoke else RAMP_HOLD_S
    ts_ramp = 240.0 if smoke else TS_RAMP_S
    op_names = sorted(ops["ops"])[:1] if smoke else sorted(ops["ops"])
    cases: List[Dict[str, Any]] = []
    for op in op_names:
        start = ops["ops"][op]["start_time"]
        t0 = settle + PHASE_E_S
        # -- R: load ramps
        for area in ("1", "2", "3"):
            bus = int(ops["ramp_bus"][area])
            for rate, T in rates.items():
                spec = dict(kind="bus", bus=bus, q_mvar=RAMP_Q_MVAR, p_mw=0.0,
                            t0_s=t0, ramp_s=T, hold_s=r_hold, down=True)
                cases.append(dict(part="R", case=f"R_{op}_a{area}_{rate}", op=op,
                                  start_time=start, ramps=[spec], events=[],
                                  minutes=_minutes(t0 + 2 * T + r_hold + tail),
                                  variants=ALL, area=int(area), rate=rate,
                                  windows=_ramp_windows(t0, T, r_hold, True)))
            if smoke:
                break
        spec = dict(kind="ts_loads", rel=TS_RAMP_REL, t0_s=t0, ramp_s=ts_ramp,
                    hold_s=r_hold, down=False)
        cases.append(dict(part="R", case=f"R_{op}_sys", op=op, start_time=start,
                          ramps=[spec], events=[], variants=ALL, area=0, rate="sys",
                          minutes=_minutes(t0 + ts_ramp + r_hold),
                          windows=_ramp_windows(t0, ts_ramp, r_hold, False)))
        # -- P: phase sweep (O2, O3, S2)
        p_events = [("load", P_LOAD_AREA2), ("gen", P_GEN), ("line", P_LINE),
                    ("load", int(ops["ramp_bus"]["1"])), ("load", int(ops["ramp_bus"]["3"]))]
        for kind, idx in (p_events[:1] if smoke else p_events):
            for ph in (PHASES_P_S[:1] if smoke else PHASES_P_S):
                t_e = settle + ph
                cases.append(dict(part="P", case=f"P_{op}_{kind}{idx}_ph{int(ph)}", op=op,
                                  start_time=start, ramps=[], phase_s=ph,
                                  events=_event_pair(kind, idx, t_e, hold),
                                  t_e=t_e, t_rev=t_e + hold, kind=kind, element=idx,
                                  minutes=_minutes(t_e + hold + tail),
                                  variants=P_VARIANTS))
        # -- E: the catalogue at phase 100 s
        cat = ([("gen", g) for g in elements["gens"]] + [("line", li) for li in elements["lines"]]
               + [("load", b) for b in elements["load_buses"]])
        for kind, idx in (cat[:1] if smoke else cat):
            t_e = settle + PHASE_E_S
            cases.append(dict(part="E", case=f"E_{op}_{kind}{idx}", op=op,
                              start_time=start, ramps=[], phase_s=PHASE_E_S,
                              events=_event_pair(kind, idx, t_e, hold),
                              t_e=t_e, t_rev=t_e + hold, kind=kind, element=idx,
                              minutes=_minutes(t_e + hold + tail), variants=ALL))
    return cases


def _ramp_windows(t0: float, T: float, hold: float, down: bool) -> Dict[str, List[float]]:
    w = {"pre": [t0 - 300.0, t0], "up": [t0, t0 + T], "hold": [t0 + T, t0 + T + hold]}
    if down:
        w["down"] = [t0 + T + hold, t0 + 2 * T + hold]
        w["after"] = [t0 + 2 * T + hold, None]
    return w


# ---------------------------------------------------------------------------
#  One run
# ---------------------------------------------------------------------------

def _metrics(log: list, case: Dict[str, Any]) -> Dict[str, Any]:
    end = float(log[-1].time_s) + 1.0
    if case["part"] == "R":
        w = case["windows"]
        base = ms.baseline(log, *w["pre"])
        out = {"baseline": base}
        for k, (a, b) in w.items():
            out[k] = ms.window(log, a, end if b is None else b, T_TS, baseline=base)
        out["recovery_after_up"] = ms.recovery(log, w["up"][1], w["hold"][1], T_TS, tol=0.5e-3)
        return out
    t_e, t_rev = float(case["t_e"]), float(case["t_rev"])
    base = ms.baseline(log, t_e - 300.0, t_e)
    return {"baseline": base,
            "pre": ms.window(log, t_e - 300.0, t_e, T_TS, baseline=base),
            "event": ms.window(log, t_e, t_rev, T_TS, baseline=base),
            "after": ms.window(log, t_rev, end, T_TS, baseline=base),
            "recovery_event": ms.recovery(log, t_e, t_rev, T_TS),
            "recovery_reversal": ms.recovery(log, t_rev, end, T_TS),
            "event_rows": _event_rows(log)}


def _event_rows(log: list) -> List[Dict[str, Any]]:
    from experiments.ch_11_case2_multisystem import metrics_case2 as mc2
    rows = []
    for r in mc2.event_metrics(log, T_TS, float(log[1].time_s - log[0].time_s)):
        r = dict(r)
        r["w_total_mvar_min"] = float(sum(r["w_sts_mvar_min"].values()))
        rows.append(r)
    return rows


def run_one(case: Dict[str, Any], name: str, out: str) -> Dict[str, Any]:
    """Worker: one (case, variant); a failure is a result."""
    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    d = Path(out) / "runs" / case["case"] / name
    d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    try:
        cfg = v.resolved(name, int(case["minutes"]), case["events"], NOISE_SEED)
        cfg = dataclasses.replace(
            cfg, start_time=_dt.datetime.fromisoformat(case["start_time"]),
            hold_profiles=True, profile_ramps=tuple(case["ramps"]),
            result_dir=str(d), verbose=0)
        n_steps = int(round(cfg.n_total_s / cfg.dt_s))
        with open(d / "console.log", "w", encoding="utf-8") as fh, \
                contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            try:
                log = run_multi_tso_dso(cfg)
                status = "ok" if log is not None and len(log) == n_steps else \
                    f"short_log {len(log or [])}/{n_steps}"
            except Exception as exc:                # noqa: BLE001 -- an outcome
                log = list(getattr(exc, "partial_log", None) or [])
                status = f"failed: {type(exc).__name__}: {exc}"
                traceback.print_exc()
        with open(d / "log.pkl", "wb") as fh:
            pickle.dump(log, fh)
        met = None
        if log and len(log) > 10:
            try:
                met = _metrics(log, case)
            except Exception as exc:                # noqa: BLE001
                met = {"error": f"{type(exc).__name__}: {exc}"}
        if met is not None:
            (d / "metrics.json").write_text(json.dumps(met, indent=1, default=float),
                                            encoding="utf-8")
    except Exception as exc:                        # noqa: BLE001 -- task crash
        status, log = f"task_crash: {type(exc).__name__}: {exc}", []
        (d / "crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
    st = {"case": case["case"], "part": case["part"], "variant": name,
          "status": status, "n_records": len(log or []), "minutes": case["minutes"],
          "host": socket.gethostname(), "wall_s": time.time() - t_start}
    (d / "status.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return st


# ---------------------------------------------------------------------------
#  Campaign
# ---------------------------------------------------------------------------

def _tasks(cases: List[Dict[str, Any]], out: Path, parts: List[str]) -> List[tuple]:
    todo = []
    for c in cases:
        if c["part"] not in parts:
            continue
        for n in c["variants"]:
            st = out / "runs" / c["case"] / n / "status.json"
            if st.exists() and not json.loads(st.read_text(encoding="utf-8"))["status"] \
                    .startswith("task_crash"):
                continue
            todo.append((c, n))
    todo.sort(key=lambda cn: (PART_ORDER[cn[0]["part"]],
                              -COST[cn[1]] * cn[0]["minutes"]))
    return todo


def simulate(out: Path, cases: List[Dict[str, Any]], parts: List[str], jobs: int) -> Dict[str, int]:
    counts = {"ok": 0, "failed": 0}
    for attempt in range(4):
        todo = _tasks(cases, out, parts)
        print(f"  [pool] {len(todo)} run(s) to go, {jobs} workers "
              f"(attempt {attempt + 1})", flush=True)
        if not todo:
            break
        try:
            # No max_tasks_per_child: on Python 3.12 the executor stops replacing
            # recycled workers (the first campaign stalled after 30 x 4 = 120
            # runs, 2026-10-28 21:07); the MC driver's pool runs without it too.
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futs = {pool.submit(run_one, c, n, str(out)): (c["case"], n) for c, n in todo}
                for k, fut in enumerate(as_completed(futs), start=1):
                    st = fut.result()
                    ok = st["status"] == "ok"
                    counts["ok" if ok else "failed"] += 1
                    print(f"  [{k}/{len(todo)}] {st['case']} {st['variant']}: "
                          f"{st['status'][:80]} ({st['wall_s'] / 60:.1f} min)", flush=True)
        except BrokenProcessPool as exc:
            print(f"  [pool] broken ({exc}); restarting with the remaining runs", flush=True)
            continue
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default="R,P,E")
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--run-dir", default=None, help="continue this campaign")
    ap.add_argument("--smoke", action="store_true",
                    help="one short case per part on one OP (minutes, not hours)")
    add_run_arguments(ap)
    a = ap.parse_args()
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]

    from experiments.ch_11_case2_multisystem.mc.run_mc import enumerate_elements
    ops = json.loads(OPS_FILE.read_text(encoding="utf-8"))
    elements = enumerate_elements()
    cases = build_cases(ops, elements, smoke=a.smoke)
    v.assert_case2_controlled(10, [], NOISE_SEED)
    print("guard: every Case-2 step changes only what it is allowed to", flush=True)
    n_runs = {p: sum(len(c["variants"]) for c in cases if c["part"] == p) for p in parts}
    print(f"cases: {len(cases)}; runs per part: {n_runs}", flush=True)

    base = v.base_config(10, [], NOISE_SEED)
    with start_thesis_run(
            __file__, target=TARGET, feeds=FEEDS, config=base, paramset=v.PARAMSET,
            parameters={
                "parts": parts, "smoke": bool(a.smoke), "runs_per_part": n_runs,
                "ops": {k: o["start_time"] for k, o in ops["ops"].items()},
                "ramp_bus": ops["ramp_bus"], "settle_s": SETTLE_S,
                "phase_E_s": PHASE_E_S, "phases_P_s": list(PHASES_P_S),
                "event_hold_s": EVENT_HOLD_S, "step_q_mvar": STEP_Q_MVAR,
                "ramp": {"q_mvar": RAMP_Q_MVAR, "ramp_s": RAMP_RATES_S,
                         "hold_s": RAMP_HOLD_S, "ts_rel": TS_RAMP_REL,
                         "ts_ramp_s": TS_RAMP_S},
                "elements": {k: len(v_) for k, v_ in elements.items()},
                "noise": f"{base.measurement_noise.profile}, seed {NOISE_SEED}, "
                         "incl. SVR and droop; held profiles",
                "jobs": a.jobs, "host": socket.gethostname()},
            note=a.note, tag=a.tag or ("smoke" if a.smoke else None),
            resume=a.run_dir) as run:
        out = run.root
        (out / "cases.json").write_text(json.dumps(
            [dict(c, events=[e.__dict__ for e in c["events"]]) for c in cases],
            indent=1, default=str), encoding="utf-8")
        (out / "ops.json").write_text(OPS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"run dir: {out}", flush=True)
        counts = simulate(out, cases, parts, max(1, a.jobs))
        run.describe_output("runs/", "<case>/<variant>/: log.pkl, status.json, "
                                     "metrics.json, console.log")
        run.describe_output("cases.json", "every case: part, OP, events, ramps, horizon")
        run.describe_output("ops.json", "operating points and ramp buses (pick_ops.py)")
        print(f"done: {counts}", flush=True)
        if counts["failed"]:
            run.manifest.setdefault("notes", []).append(
                f"{counts['failed']} run(s) not ok -- outcomes, see runs/*/*/status.json")


if __name__ == "__main__":
    main()
