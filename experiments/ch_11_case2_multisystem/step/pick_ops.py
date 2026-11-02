#!/usr/bin/env python3
"""
experiments/ch_11_case2_multisystem/step/pick_ops.py
====================================================

Operating points and ramp buses of the Case-2 step tests (thesis
``ch:case3:step``; plan ``docs/architecture/2026-10-28_case2_step_mc_plan.md``
§1-2, criteria approved by the author 2026-10-28).

Operating points, from the 2016 profile year at 15-min resolution, without the
calendar days of the design windows W1--W6 (``run_mc.EXCLUDED_DAYS``):

========  ================================================  =================
OP-H      TS load >= P90 and renewable infeed (TS + STS)    the largest TS load
          <= P25
OP-L      TS load <= P25 and STS export >= P75              the largest export
                                                            with a solution
OP-M      TS load and STS net exchange within P45--P55      closest to (P50, P50)
========  ================================================  =================

Each candidate (one per calendar day, the best timestamp of the day) must pass
a base-case check -- ``L2``, held profiles, 10 min, no events, every step with
a converged power flow; the MC's discard rule.  OP-H and OP-M take the first
candidate that passes.  The largest STS exports have no power-flow solution in
this model at all (2026-10-28: every one of the 15 largest export days with
TS load <= P25 fails while the runner initialises, under L2 and O3 alike, held
or not), so OP-L is searched by bisection over the export ranking of the days
with export >= P75 and is the largest export found to pass (P90 was the plan).

Ramp buses (STEP-R a): area 2 takes IEEE bus 12 (pandapower 11), the bus of the
reference run's load step.  Areas 1 and 3 take the transmission load bus whose
+10 Mvar reactive injection moves the RMS voltage of the area's monitored
buses most (one power flow per candidate at OP-M, held profiles, no control).

Writes ``ops.json`` next to this file (an input of ``run_step.py``).

    python -m experiments.ch_11_case2_multisystem.step.pick_ops

Author: Manuel Schwenke / Claude Code (2026-10-28)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import copy
import dataclasses
import datetime as _dt
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandapower as pp

from experiments.ch_08_setup.reference_period import characterise_reference_period as crp
from experiments.ch_11_case2_multisystem import variants as v
from experiments.ch_11_case2_multisystem.mc.run_mc import (EXCLUDED_DAYS,
                                                           enumerate_elements)
from experiments.runners.multi_tso_dso import run_multi_tso_dso

OUT = Path(__file__).resolve().parent / "ops.json"
AREA2_RAMP_BUS = 11            # IEEE bus 12, the reference load step
N_CANDIDATES = 15
CHECK_MIN = 10


def _ranks(s) -> np.ndarray:
    return s.rank(pct=True).to_numpy() * 100.0


def yearly_table():
    cfg = v.resolved("L2", CHECK_MIN)
    net, meta, bus_zone = crp.build(cfg)
    level = crp.classify_buses(net, meta, bus_zone)
    df = crp.exogenous_year(net, level)
    df = df[~np.isin(df.index.date, list(EXCLUDED_DAYS))]
    df = df[df.index <= df.index.max() - _dt.timedelta(hours=4)]   # horizon fits
    df["res_P"] = df["der_P|TS"] + df["der_P|STS"]
    df["export_P|STS"] = -df["net_P|STS"]
    for k in ("load_P|TS", "res_P", "export_P|STS", "net_P|STS"):
        df[f"rank:{k}"] = _ranks(df[k])
    return df


def candidates(df) -> Dict[str, List]:
    h = df[(df["rank:load_P|TS"] >= 90) & (df["rank:res_P"] <= 25)]
    lo = df[(df["rank:load_P|TS"] <= 25) & (df["rank:export_P|STS"] >= 75)]
    m = df[df["rank:load_P|TS"].between(45, 55) & df["rank:net_P|STS"].between(45, 55)]
    m = m.assign(_d=np.hypot(m["rank:load_P|TS"] - 50, m["rank:net_P|STS"] - 50))

    def per_day(frame, cap=N_CANDIDATES):   # the best timestamp of each day
        frame = frame[~frame.index.normalize().duplicated(keep="first")]
        return list(frame.index[:cap])

    return {
        "OP-H": per_day(h.sort_values("load_P|TS", ascending=False)),
        "OP-M": per_day(m.sort_values("_d")),
        "OP-L": per_day(lo.sort_values("export_P|STS", ascending=False), cap=None),
    }


def base_case_ok(t0) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(v.resolved("L2", CHECK_MIN), start_time=t0.to_pydatetime(),
                                  hold_profiles=True, result_dir=tmp, verbose=0)
        n = int(round(cfg.n_total_s / cfg.dt_s))
        try:
            log = run_multi_tso_dso(cfg)
            ok = log is not None and len(log) == n
            return {"ok": bool(ok), "reason": None if ok else f"short log {len(log or [])}/{n}"}
        except Exception as exc:                       # noqa: BLE001 -- an outcome
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def ramp_buses(t0) -> Dict[str, Any]:
    """Per area, the load bus whose +10 Mvar moves the area voltages most."""
    loads = enumerate_elements()["load_buses"]
    result: Dict[str, Any] = {}

    def hook(st):
        net, zd = st["net"], st["zone_defs"]
        area_of = {int(b): int(z) for z, d in zd.items() for b in d.bus_indices}
        base = copy.deepcopy(net)
        pp.runpp(base, run_control=False, calculate_voltage_angles=True)
        for z, d in zd.items():
            rows = []
            vb = [int(b) for b in d.v_bus_indices]
            for b in (b for b in loads if area_of.get(int(b)) == int(z)):
                n2 = copy.deepcopy(base)
                pp.create_load(n2, int(b), p_mw=0.0, q_mvar=10.0)
                pp.runpp(n2, run_control=False, calculate_voltage_angles=True,
                         init="results")
                dv = n2.res_bus.vm_pu.loc[vb].to_numpy() - base.res_bus.vm_pu.loc[vb].to_numpy()
                rows.append((float(np.sqrt(np.mean(dv ** 2))) * 1e3, int(b)))
            rows.sort(reverse=True)
            result[str(int(z))] = [{"bus": b, "dv_rms_mpu_per_10mvar": round(x, 3)}
                                   for x, b in rows]
        return True

    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(v.resolved("L2", CHECK_MIN), start_time=t0.to_pydatetime(),
                                  hold_profiles=True, result_dir=tmp, verbose=0)
        run_multi_tso_dso(cfg, pre_loop_hook=hook)
    return result


def _op_record(df, t0, tried) -> Dict[str, Any]:
    row = df.loc[t0]
    return {"start_time": t0.isoformat(),
            "load_P_TS_mw": round(float(row["load_P|TS"]), 1),
            "load_Q_TS_mvar": round(float(row["load_Q|TS"]), 1),
            "res_P_mw": round(float(row["res_P"]), 1),
            "net_P_STS_mw": round(float(row["net_P|STS"]), 1),
            "rank_load_P_TS": round(float(row["rank:load_P|TS"]), 1),
            "rank_res_P": round(float(row["rank:res_P"]), 1),
            "rank_export_P_STS": round(float(row["rank:export_P|STS"]), 1),
            "tried": tried}


def _check(t0, tried, name) -> bool:
    chk = base_case_ok(t0)
    tried.append({"start_time": t0.isoformat(), **chk})
    print(f"{name} {t0}: {chk}", flush=True)
    return bool(chk["ok"])


def _write(ops, screen, ramp) -> None:
    out = {"criteria": __doc__.split("Operating points,")[1].split("Ramp buses")[0].strip(),
           "excluded_days": sorted(str(d) for d in EXCLUDED_DAYS),
           "ops": ops, "ramp_bus": ramp, "ramp_screen_at_OP-M": screen,
           "made": _dt.datetime.now().isoformat(timespec="seconds"),
           "by": "experiments/ch_11_case2_multisystem/step/pick_ops.py"}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("ops", "ramp_bus")}, indent=2), flush=True)


def main() -> None:
    df = yearly_table()
    cands = candidates(df)
    ops: Dict[str, Any] = {}
    for name in ("OP-H", "OP-M"):
        tried: List[Dict[str, Any]] = []
        for t0 in cands[name]:
            if _check(t0, tried, name):
                ops[name] = _op_record(df, t0, tried)
                break
        else:
            raise SystemExit(f"no feasible candidate for {name}: {tried}")
    screen = ramp_buses(_dt_index(ops["OP-M"]["start_time"]))
    ramp = {"1": screen["1"][0]["bus"], "2": AREA2_RAMP_BUS, "3": screen["3"][0]["bus"]}
    _write(ops, screen, ramp)                      # usable before OP-L is found

    # OP-L: bisection for the first day (largest export) that passes.
    days, tried = cands["OP-L"], []
    lo, hi = 0, len(days) - 1
    if not _check(days[hi], tried, "OP-L"):
        raise SystemExit(f"even the smallest export of the OP-L set fails: {tried}")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _check(days[mid], tried, "OP-L"):
            hi = mid
        else:
            lo = mid
    if hi == 1 and _check(days[0], tried, "OP-L"):
        hi = 0
    ops["OP-L"] = _op_record(df, days[hi], tried)
    ops["OP-L"]["bisection"] = {"n_days": len(days), "index": hi}
    _write(ops, screen, ramp)


def _dt_index(s: str):
    import pandas as pd
    return pd.Timestamp(s)


if __name__ == "__main__":
    main()
