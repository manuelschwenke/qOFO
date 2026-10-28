#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_3_amendment1.py
==========================================================
Amendment 1 to the weight-selection procedure of 2026-10-25 (author decision
2026-10-28, option (c1) of
``docs/tuning/2026-10-28_tap_engagement_at_selected_weights.md``).

The amendment adds one filter rule and changes nothing else::

    theta_STS,min <= V_max,band - V_ref = 1.08 - 1.03 = 0.05 pu

``theta_STS,min`` is the systematic commit threshold of the strongest STS
coupler over W1-W6, ``g_w_dso_oltc / (2 dso_g_v max_i |sum_j H_V[j, i]|)``
(``tuning.stage1_screen.engage_deviation`` on the cached operating points the
campaign loaded). A candidate above it cannot commit a coupler on the voltage
gradient while every STS bus is inside the operating band.

Stages, in one invocation (one run folder, one manifest writer):

1. **select** -- theta_STS,min for the 242 candidates of run 0001
   (``selection/union_2b_round4``); the admitted ones are copied to
   ``selection/union_amended`` and the UNCHANGED ``ch_9_3_select.py`` is run on
   them with the campaign's arguments (12 h, bank budget 12/day).
2. **settle** -- Experiment C in rank order, with the campaign's call of
   ``ch_9_1_ninner_isolated_sts.py`` and criterion (p95 <= 3 N_inner = 27,
   median residual <= 1 Mvar). Measured results of run 0001 are reused (same
   campaign, same closed loop); its aborted run of 301c285ee52b is not a
   result and is repeated.
3. **probe** (parallel to 2) -- W1-W4, W6 re-simulated for the listed keys
   exactly as the Stage-II screen built them, with a read-only probe on each
   controller's objective gradient: per iteration the measured voltages the
   controller sees and, per tap/bank column, the commit ratio
   ``|c_i + 2 g_u u_i| / (g_w,i + g_u)`` of ``build_miqp_problem`` (a free
   integer column commits one step iff it exceeds 1). The probe wraps
   ``_compute_objective_gradient`` and ``step`` on the instances from the
   ``pre_loop_hook`` and returns their values unchanged; the stored e_v(TS)
   row is the reproduction check.
4. **paramset** -- if a candidate passes, it is written as
   ``configs/paramsets/ch9_selected_band_2026_10_a1.json``. ``SELECTED_PARAMSET``
   is NOT switched here (author decision pending).

Usage::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_3_amendment1.py \
        --c-workers 8 --probe-workers 20
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
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
OPS_DIR = REPO_ROOT / "results" / "stage1" / "operating_points"
OPS_KEY = "op_rural_700_net82afed5b"           # the cache run 0001 loaded
START = "stage1_start_10_2026_banks"
RESULT = "ch9_selected_band_2026_10_a1"
V_REF, V_MAX_BAND = 1.03, 1.08
THETA_MAX = V_MAX_BAND - V_REF                 # 0.05 pu
BANK_BUDGET = 12.0
N_INNER = 9
WINDOWS = ["W1", "W2", "W3", "W4", "W6"]
SELECTED_AS_FROZEN = "fde2e81af6b1"
STAGE1_POINT = "3cb66225e1a7"

_LOG_LOCK = threading.Lock()


def _log(root: Path, msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with _LOG_LOCK, (root / "amendment.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _rel(p: Path) -> str:
    # Resolved on both sides: on the e5 hosts the repository is reached both
    # as a mapped drive (Z:) and as its UNC path, and relpath refuses to
    # relate the two (failed run 0008 on 2026-10-28 at the result write).
    return os.path.relpath(Path(p).resolve(), REPO_ROOT)


def _pin_threads() -> None:
    """One BLAS thread per process: a different count changes summation order,
    which can flip an integer decision and break the reproduction."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


# -- 1. select ------------------------------------------------------------------------

def theta_sts_min(weights: Dict[str, float], ops) -> float:
    """Strongest STS coupler's systematic commit threshold over all conditions."""
    import numpy as np

    from tuning import stage1_screen as s1
    best = float("inf")
    for op in ops:
        for lc in op.layers:
            if lc.layer != "STS":
                continue
            g_v, g_q = s1._layer_weights(lc, weights)
            dev = s1.engage_deviation(lc, g_v, g_q, lc.g_w_vector(weights))
            for i, c in enumerate(lc.classes):
                if c == "dso_oltc" and np.isfinite(dev[i]):
                    best = min(best, float(dev[i]))
    return best


def stage_select(root: Path) -> Dict[str, Any]:
    ops = [pickle.load((OPS_DIR / f"{OPS_KEY}_{w}.npz").open("rb"))
           for w in ("W1", "W2", "W3", "W4", "W5", "W6")]
    sel = root / "selection"
    union = sel / "union_amended"
    union.mkdir(parents=True, exist_ok=True)
    thetas: Dict[str, float] = {}
    for f in sorted(UNION.glob("cand_*.json")):
        c = json.loads(f.read_text("utf-8"))
        th = theta_sts_min(c["weights"], ops)
        thetas[c["key"]] = th
        if th <= THETA_MAX:
            shutil.copy2(f, union / f.name)
    admitted = sorted(k for k, v in thetas.items() if v <= THETA_MAX)
    (sel / "theta_sts_min.json").write_text(json.dumps(
        {"theta_max_pu": THETA_MAX, "theta_sts_min_pu": thetas,
         "admitted": admitted,
         "removed": sorted(k for k, v in thetas.items() if v > THETA_MAX)},
        indent=1), "utf-8")
    _log(root, f"select: {len(admitted)} / {len(thetas)} candidates have "
               f"theta_STS,min <= {THETA_MAX:g} pu")
    out = sel / "selection_amended.json"
    with (sel / "select_amended.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.call(
            [PY, "-X", "utf8", "experiments/ch_09_parameter_selection/ch_9_3_select.py",
             "--candidates", _rel(union), "--hours", "12",
             "--shunt-ops-per-day", f"{BANK_BUDGET:g}", "--out", _rel(out)],
            cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"ch_9_3_select.py exit {rc}; see select_amended.log")
    rep = json.loads(out.read_text("utf-8"))
    rep["_theta"] = thetas
    _log(root, f"select: {rep['n_feasible']} / {rep['n_total']} feasible; "
               f"ranked first {[r['key'] for r in rep['ranked'][:4]]}")
    return rep


# -- 2. settle ------------------------------------------------------------------------

def _criterion(expc: Path, label: str) -> Dict[str, Any]:
    """Run 0001's criterion, verbatim (driver.py::criterion)."""
    import numpy as np
    import pandas as pd
    latest = expc / label / "_latest.txt"
    if not latest.exists():
        return {"pass": False, "error": "no run"}
    rd = expc / label / latest.read_text("utf-8").strip()
    if not (rd / "steps.csv").exists():
        return {"pass": False, "error": f"no steps.csv in {rd.name}"}
    s = pd.read_csv(rd / "steps.csv")
    n, res = s.n_settle_final_1mvar, s.residual_mvar
    p95, med = float(np.percentile(n, 95)), float(res.median())
    return {"run": _rel(rd), "n_steps": int(len(n)), "p95": p95,
            "per_dso_p95": s.groupby("dso").n_settle_final_1mvar.apply(
                lambda x: float(np.percentile(x, 95))).round(2).to_dict(),
            "n_over": int((n > 3 * N_INNER).sum()),
            "censored": int(s.get("n_settle_censored", pd.Series([0])).sum()),
            "max": int(n.max()), "median_residual_mvar": med,
            "pass": bool(p95 <= 3 * N_INNER and med <= 1.0)}


def stage_settle(root: Path, rep: Dict[str, Any], workers: int) -> Dict[str, Any]:
    from tuning.parameters import WEIGHT_FIELDS
    from tuning.stage2_search import rank_lexicographic
    expc = root / "expC"
    expc.mkdir(parents=True, exist_ok=True)
    path = expc / "criterion_results.json"
    prior = json.loads((RUN0001 / "expC" / "criterion_results.json").read_text("utf-8"))
    # Measured results of the same campaign are reused; an aborted run is not one.
    results = {k: {**v, "reused_from": _rel(RUN0001)} for k, v in prior.items()
               if "error" not in v}
    if path.exists():                       # a resumed run keeps its own results
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
            rc = subprocess.call(
                [PY, "-X", "utf8",
                 "experiments/ch_09_parameter_selection/ch_9_1_ninner_isolated_sts.py",
                 "--paramset", START, "--weights-json", _rel(wf), "--label", label,
                 "--out", _rel(expc), "--workers", str(workers),
                 "--max-step-mvar", "10", "--save-trajectories"],
                cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
        results[first["key"]] = _criterion(expc, label)
        o = results[first["key"]]
        _log(root, f"settle: {first['key']} exit {rc}: "
                   f"{'PASS' if o['pass'] else 'FAIL'} p95 {o.get('p95')} "
                   f"median residual {o.get('median_residual_mvar')} {o.get('error', '')}")
        path.write_text(json.dumps(results, indent=1), "utf-8")


# -- 3. probe -------------------------------------------------------------------------

def _install_probe(state: Dict[str, Any], log: Dict[str, Any]) -> bool:
    import numpy as np

    def wrap(ctrl, key: str, cols: np.ndarray, names: List[str]) -> None:
        cfg = ctrl.config
        vbus = [int(b) for b in cfg.voltage_bus_indices]
        rec = log.setdefault(key, {"names": names, "v_buses": vbus, "t": [],
                                   "it": [], "v": [], "lin": [], "quad": []})
        orig_step, orig_grad = ctrl.step, ctrl._compute_objective_gradient
        now = {"t": float("nan")}

        def step(measurement, *a, **kw):
            t = kw.get("sim_time_s")
            now["t"] = float("nan") if t is None else float(t)
            return orig_step(measurement, *a, **kw)

        def grad(measurement):
            g = orig_grad(measurement)
            pos = {int(b): i for i, b in enumerate(measurement.bus_indices)}
            v = np.array([measurement.voltage_magnitudes_pu[pos[b]] if b in pos
                          else np.nan for b in vbus], dtype=float)
            gw, gu = ctrl._get_per_variable_weights()
            n = ctrl.n_controls
            gw = (np.broadcast_to(np.asarray(ctrl.params.g_w, float), (n,))
                  if gw is None else np.asarray(gw, float))
            if gu is None:
                gu_p = np.asarray(ctrl.params.g_u, float)
                gu = np.broadcast_to(gu_p, (n,)) if gu_p.ndim <= 1 else np.full(n, np.nan)
            u = np.asarray(ctrl._u_current, float)
            lin = np.asarray(g, float)[cols] + 2.0 * np.asarray(gu, float)[cols] * u[cols]
            quad = np.asarray(gw, float)[cols] + np.asarray(gu, float)[cols]
            rec["t"].append(now["t"])
            rec["it"].append(int(getattr(measurement, "iteration", -1)))
            rec["v"].append(v)
            rec["lin"].append(lin)
            rec["quad"].append(quad)
            return g

        ctrl.step, ctrl._compute_objective_gradient = step, grad

    for area, ctrl in sorted(state["dso_controllers"].items()):
        cfg = ctrl.config
        m = getattr(cfg, "der_mapping", None)
        n_der = m.n_der if m is not None else len(cfg.der_indices)
        n_oltc = len(cfg.oltc_trafo_indices)
        wrap(ctrl, f"STS|{area}", np.arange(n_der, n_der + n_oltc),
             [f"{area}|trafo_{int(t)}" for t in cfg.oltc_trafo_indices])
    for zid, ctrl in sorted(state["tso_controllers"].items()):
        cfg = ctrl.config
        m = getattr(cfg, "der_mapping", None)
        n_der = m.n_der if m is not None else len(cfg.der_indices)
        o0 = n_der + len(cfg.pcc_trafo_indices) + len(cfg.gen_indices)
        n_oltc, n_sh = len(cfg.oltc_trafo_indices), len(cfg.shunt_bus_indices)
        wrap(ctrl, f"TS|{zid}", np.arange(o0, o0 + n_oltc + n_sh),
             [f"z{zid}[{k}]" for k in range(n_oltc)]
             + [f"z{zid}b[{k}]" for k in range(n_sh)])
    return False                                     # continue the time loop


def _probe_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """One (key, window) re-simulation with the gradient probe. Module-level."""
    _pin_threads()
    import dataclasses as _dc
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np

    import tuning.runner as tr
    from configs.paramsets import apply_paramset
    from experiments.ch_09_parameter_selection._ch9_selected_design import (
        FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE)
    from experiments.run_multi_system_ofo import make_config
    from tuning.metrics import MetricScales
    from tuning.parameters import apply_to_config
    from tuning.scenarios_mc_v2 import condition_set_evented

    out = Path(job["out"])
    if out.exists():
        return {"key": job["key"], "window": job["window"], "status": "cached"}
    # Same construction as ch_9_3_stage2_screen._run_one_window (frozen, evented).
    base = apply_to_config(apply_paramset(make_config(), START, verbose=False),
                           job["weights"])
    base = _dc.replace(base, dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                       dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
    sc = next(s for s in condition_set_evented(hours=job["hours"])
              if s.name == job["window"])
    log: Dict[str, Any] = {}
    orig_get = tr.get_run_multi_tso_dso

    def patched():
        return functools.partial(orig_get(),
                                 pre_loop_hook=lambda st: _install_probe(st, log))

    tr.get_run_multi_tso_dso = patched
    try:
        result, records = tr.run_scenario_with_records(sc, base, MetricScales())
    finally:
        tr.get_run_multi_tso_dso = orig_get

    fields = ("dso_trafo_tap_pos", "dso_trafo_q_actual_mvar", "dso_trafo_q_set_mvar",
              "dso_group_v_min_pu", "dso_group_v_mean_pu", "dso_group_v_max_pu",
              "dso_group_v_meas_min_pu", "dso_group_v_meas_mean_pu",
              "dso_group_v_meas_max_pu", "dso_group_q_der_mvar",
              "dso_group_q_der_min_mvar", "dso_group_q_der_max_mvar",
              "zone_v_min", "zone_v_mean", "zone_v_max")
    series = [{"t": float(r.time_s),
               **{f: dict(getattr(r, f, {}) or {}) for f in fields},
               "zone_oltc_taps": {int(z): list(map(float, v))
                                  for z, v in (r.zone_oltc_taps or {}).items()},
               "zone_tso_shunt_states": {int(z): list(map(float, v)) for z, v in
                                         (r.zone_tso_shunt_states or {}).items()}}
              for r in records]
    probe = {k: {"names": v["names"], "v_buses": v["v_buses"],
                 "t": np.asarray(v["t"], float), "it": np.asarray(v["it"], int),
                 "v": np.asarray(v["v"], float), "lin": np.asarray(v["lin"], float),
                 "quad": np.asarray(v["quad"], float)} for k, v in log.items()}
    m = result.metrics
    row = {"e_v_ts": float(m.v_rms_ts), "e_v_ds": float(m.v_rms_ds),
           "e_q": float(m.itae_q_pcc), "failure": result.failure_reason}
    payload = {"key": job["key"], "window": job["window"], "hours": job["hours"],
               "row": row, "stored_row": job.get("stored_row"), "series": series,
               "probe": probe, "weights": job["weights"]}
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    return {"key": job["key"], "window": job["window"], "status": "done",
            **{k: row[k] for k in ("e_v_ts", "e_v_ds", "failure")}}


def stage_probe(root: Path, keys: List[str], workers: int, hours: float) -> None:
    pdir = root / "probe"
    pdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in keys:
        c = json.loads((UNION / f"cand_{k}.json").read_text("utf-8"))
        for w in WINDOWS:
            jobs.append({"key": k, "window": w, "hours": hours, "weights": c["weights"],
                         "stored_row": {x: c["per_window"][w].get(x) for x in
                                        ("e_v_ts", "e_v_ds", "e_q")},
                         "out": str(pdir / f"{k}__{w}.pkl")})
    _log(root, f"probe: {len(jobs)} units ({keys} x {WINDOWS}), {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe_one, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
                st = j["stored_row"]["e_v_ts"]
                rep = "" if r.get("status") != "done" or st is None else \
                    f" e_v_ts {r['e_v_ts']:.10g} stored {st:.10g} diff {r['e_v_ts'] - st:+.2e}"
                _log(root, f"probe: {j['key']} {j['window']} {r.get('status')}{rep} "
                           f"{r.get('failure') or ''}")
            except Exception as exc:                  # one unit never kills the rest
                _log(root, f"probe: {j['key']} {j['window']} FAILED {exc!r}")


# -- 4. paramset ----------------------------------------------------------------------

def write_result(root: Path, win: Dict[str, Any], rep: Dict[str, Any],
                 c_res: Dict[str, Any]) -> None:
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    path = PARAMSET_DIR / f"{RESULT}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a parameter set is never overwritten")
    fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
    doc = {
        "name": RESULT,
        "title": "Controller weights selected by the Ch 8 procedure "
                 "(Experiment B), final configuration with DSO operating band "
                 "and STS coupler engagement bound",
        "description": [
            "Selected by docs/tuning/2026-10-28_weight_selection_procedure.md "
            "(applied once, run 0008), with every DSO controller holding "
            "0.92-1.08 pu (make_config):",
            f"filter including theta_STS,min <= {THETA_MAX:g} pu (strongest STS "
            "coupler commits on a uniform offset inside the band around "
            "V_ref = 1.03 pu); ranking on the per-bus TS voltage deviation with "
            "a 2 % band on the STS deviation; settling requirement in rank order.",
            "Stage-II units from run 0001 (identical decision path, bit-exact "
            "simulations; analysis/tap_engagement/replay_path.py).",
            "The DSO band is configuration, not part of this parameter set.",
        ],
        "provenance": {
            "producer": _rel(Path(__file__)),
            "run": _rel(root),
            "procedure": "docs/tuning/2026-10-28_weight_selection_procedure.md",
            "bound_declared_in": "docs/tuning/2026-10-28_amendment1_sts_engagement_bound.md",
            "campaign_dir": _rel(RUN0001),
            "theta_sts_max_pu": THETA_MAX,
            "theta_sts_min_pu": rep["_theta"][win["key"]],
            "starting_paramset": START,
            "selected_key": win["key"],
            "as_frozen_result": SELECTED_AS_FROZEN,
            "n_candidates_admitted": rep.get("n_total"),
            "n_feasible": rep.get("n_feasible"),
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
    _log(root, f"wrote {path}")


# -- main -----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    from experiments.thesis_runs import add_run_arguments, start_thesis_run
    p = argparse.ArgumentParser(prog="ch_9_3_amendment1")
    p.add_argument("--c-workers", type=int, default=8,
                   help="Workers of the Experiment-C run (run 0001: 7 per candidate).")
    p.add_argument("--probe-workers", type=int, default=20)
    p.add_argument("--probe-keys", default="",
                   help="Comma-separated keys; default: the amended rank 1 and 2, "
                        "the as-frozen result and the Stage-I point.")
    p.add_argument("--hours", type=float, default=12.0,
                   help="Probe window length (12 h = the campaign's).")
    p.add_argument("--skip-settle", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--resume", type=Path, default=None,
                   help="Run folder to resume: selection is recomputed "
                        "(deterministic), settling continues from that run's "
                        "criterion_results.json, the result is written.")
    add_run_arguments(p)
    args = p.parse_args(argv)

    params = {"amendment": "theta_STS,min <= V_max,band - V_ref",
              "theta_sts_max_pu": THETA_MAX, "campaign": _rel(RUN0001),
              "operating_points": f"{_rel(OPS_DIR)}/{OPS_KEY}_W*.npz",
              "settling_workers": args.c_workers, "probe_hours": args.hours}
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "tab:param:timescales:ninner"],
            paramset=START, parameters=params,
            time_series=("W1-W4, W6 of tuning.scenarios_mc_v2.CONDITIONS, 12 h, "
                         "one disturbance each, evented, frozen network; T_TS "
                         "180 s, T_STS 20 s, dt 20 s (re-selection reads run 0001; "
                         "the probe re-simulates); Experiment C: 120 isolated "
                         "interface steps"),
            note=args.note, tag=args.tag, resume=args.resume) as run:
        root = run.root
        rep = stage_select(root)
        keys = [k for k in args.probe_keys.split(",") if k] or list(dict.fromkeys(
            [r["key"] for r in rep["ranked"][:2]] + [SELECTED_AS_FROZEN, STAGE1_POINT]))
        settle_out: Dict[str, Any] = {}
        th = None
        if not args.skip_settle:
            th = threading.Thread(target=lambda: settle_out.update(
                stage_settle(root, rep, args.c_workers)))
            th.start()
        if not args.skip_probe:
            stage_probe(root, keys, args.probe_workers, args.hours)
        if th is not None:
            th.join()
            win = settle_out.get("winner")
            if win is None:
                _log(root, "no admitted candidate passes the settling requirement")
            else:
                _log(root, f"RESULT: {win['key']}")
                write_result(root, win, rep, settle_out["results"])
        run.describe_output("amendment.log", "every stage with times")
        run.describe_output("selection/", "theta_STS,min per candidate, admitted "
                            "union, unchanged ch_9_3_select.py report")
        run.describe_output("expC/", "Experiment C of the amended rank order; "
                            "criterion_results.json (run 0001 results reused)")
        run.describe_output("probe/", "W1-W4, W6 re-simulations with the gradient "
                            "probe: records, measured voltages, commit ratios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
