"""
experiments/ch_09_parameter_selection/ch_9_4_deadband.py
========================================================

Experiments D1 and D2 of ``ch:timescales:deadband`` on the quasi-steady-state
plant: the half-width of the dead band of the dispatch-anchored droop (DAD)
layer, and the layer's response to a disturbance between dispatch instants.

Plan and decisions: ``docs/tuning/2026-10-27_D1_D2_deadband_plan.md``
(author decisions 2026-10-27).  Fixed throughout: the selected weights
(``SELECTED_PARAMSET``), T_TS = 180 s, T_STS = 20 s, the droop slope, the
frozen network, no measurement noise (the thesis states it), plant step
``dt = 10 s``.  The 15-min profiles are interpolated linearly to dt, so every
plant step is a profile step; dt = 10 s puts one solve with held anchors
between two STS dispatches and lets an event fall strictly between them.

D1 -- routine droop inactivity
    1. W1-W6, 2 h each, droop suppressed (dead band ``SUPPRESSED`` pu), full
       hierarchy.  Per converter i and anchor interval k (from one anchor
       update to the next, eq:param:deadband:drift):
       ``d_ik = max |U_f - U_anchor|`` over the solves of the interval.
       Intervals that start in the first ``WARMUP_S`` are the start-up phase,
       reported separately.
    2. ``Delta* = min{d in CANDIDATES : d >= max_drift d_ik + EPS_U}``
       (eq:param:deadband:detector); the candidate set is extended by 0.005 pu
       steps if no width qualifies.  "Drift" = the routine intervals that
       contain no discrete event -- a converter capability step at the
       P/S_n = 0.1 threshold, or an OLTC or bank operation (author decision
       2026-10-27, after run 0005 showed that these events alone set the
       maximum over all routine intervals; that maximum and the width it would
       give are reported alongside).
    3. Confirmation runs, droop active, at Delta* and at the adjacent narrower
       width: dead-zone activations, delivered support, and the deviation of
       the trajectories from the suppressed reference.

D2 -- disturbance response
    W1-W6 x {+-200 Mvar at TS bus 7, +-200 Mvar at TS bus 11} x {Delta*,
    2 Delta*, droop disabled}.  "+" = additional reactive load (inductive),
    "-" = capacitive.  The step is a dormant load switched in at
    ``EVENT_S`` (10 s after a TS and STS dispatch, so strictly between two STS
    dispatches and early in a TS period) and kept; it is not a step on a
    profiled load, which the next profile step would overwrite.  Every cell
    is checked for the event having reached the plant.

Start-up: every run starts with the droop suppressed, and a tested width
(D1 confirmation, D2 legs) takes effect only at ``WARMUP_S``.  The converters
start from cold-start anchors, so a width active from t = 0 lets the start-up
transient activate the droop, and runs that differ only in the width then
leave start-up in different states -- which would break the comparison with
the suppressed reference (D1) and the common pre-event state of the D2 legs.
(Found in run 0003, 2026-10-27; that run's stage 2 is superseded.)

The selected weights are reported in the run's README; the dt sensitivity is
checked by one extra suppressed W1 trace at dt = 20 s.

Run from the repository root::

    python -X utf8 -m experiments.ch_09_parameter_selection.ch_9_4_deadband --workers 30
    python -X utf8 -m experiments.ch_09_parameter_selection.ch_9_4_deadband --smoke

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""

from __future__ import annotations

import os

# One BLAS thread per worker, before numpy is imported (as
# ch_9_3_stage2_screen._pin_threads): the runs are parallel processes.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ch_09_parameter_selection._ch9_selected_design import (  # noqa: E402
    SELECTED_PARAMSET, build_selected_config)
from experiments.ch_09_parameter_selection._droop_probe import (  # noqa: E402
    DroopProbeFactory)
from experiments.helpers.records import ContingencyEvent  # noqa: E402
from experiments.thesis_runs import add_run_arguments, start_thesis_run  # noqa: E402
from tuning.scenarios import ScenarioSpec  # noqa: E402
from tuning.scenarios_mc_v2 import CONDITIONS, MC_NETWORK  # noqa: E402

# -- protocol (docs/tuning/2026-10-27_D1_D2_deadband_plan.md) -------------------
CANDIDATES: Sequence[float] = (0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020)
# pu: voltage-measurement accuracy of an HV converter, from the project's own
# noise model (configs/config.py, profile "minimum": HV voltage transformer
# 0.2 % + meter 0.1 %).  Was 0.001 (the EHV transformer alone) until
# 2026-10-27, when the author moved D1 to the drift-based rule below.
EPS_U = 0.003
# A converter's capability band that jumps by more than this between two
# solves marks a discrete event: the P/S_n = 0.1 threshold of the
# VDE-AR-N 4120 curve in controller.der_qv_local_loop._qv_capability, where
# the band steps from +-0.1 S_n to 0.  Inside the 0.1-0.2 ramp the band moves
# by far less than 1 Mvar per 10 s.
CAP_STEP_MVAR = 1.0
# Design margin in candidate steps above the rule's smallest admissible width
# (author decision 2026-10-28: 0.0075 -> 0.010 pu).
DESIGN_STEPS = 1
SUPPRESSED = 1.0       # pu half-width no routine excursion reaches = droop off
DT_S = 10.0
D1_HOURS = 2.0
WARMUP_S = 1800.0      # D1 start-up phase, reported separately
EVENT_S = 3610.0       # D2 event: 60 min + 10 s (TS/STS dispatch at 3600 s)
OBSERVE_S = 1800.0     # D2 observation after the event (10 TS periods)
Q_STEP_MVAR = 200.0
EVENTS: Dict[str, tuple] = {   # name -> (pandapower TS bus, Q of the new load)
    "b7_ind": (7, +Q_STEP_MVAR), "b7_cap": (7, -Q_STEP_MVAR),
    "b11_ind": (11, +Q_STEP_MVAR), "b11_cap": (11, -Q_STEP_MVAR),
}
WATCH_BUSES = (7, 11)
TOL_V_PU = 1e-6        # "identical trajectory" tolerances (D1 confirmation)
TOL_Q_MVAR = 1e-3

RECORD_FIELDS = (
    "zone_v_min", "zone_v_max", "zone_v_mean", "zone_v_rms_err_pu",
    "dso_group_v_min_pu", "dso_group_v_max_pu", "dso_group_v_mean_pu",
    "dso_trafo_q_actual_mvar", "dso_trafo_q_set_mvar", "dso_trafo_tap_pos",
    "dso_group_q_der_mvar")


# -- one simulation -----------------------------------------------------------------

def _config(width: float, start, duration_s: float, events=(), dt_s: float = DT_S):
    cfg, _prov = build_selected_config(paramset=SELECTED_PARAMSET)
    cfg = dataclasses.replace(
        cfg,
        measurement_noise=dataclasses.replace(cfg.measurement_noise, enabled=False),
        tso_qv_deadband_pu=float(width), dso_qv_deadband_pu=float(width),
        verbose=0)
    sc = ScenarioSpec(name="deadband", start_time=start, duration_s=duration_s,
                      contingencies=tuple(events), scenario=MC_NETWORK, dt_s=dt_s)
    return sc.overlay_on(cfg)


def _event(name: str, t_s: float) -> ContingencyEvent:
    bus, q = EVENTS[name]
    return ContingencyEvent(minute=int(t_s // 60), time_s=float(t_s),
                            element_type="load", action="connect", bus=bus,
                            p_mw=0.0, q_mvar=float(q))


def run_case(job: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate one case and write probe, watch and record files.  Picklable."""
    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    out = Path(job["out"])
    if (out / "done.json").exists():
        return json.loads((out / "done.json").read_text("utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    events = [_event(e, job["event_s"]) for e in job.get("events", ())]
    # A tested width starts after the start-up phase: until ``switch_s`` the run
    # is the suppressed reference, bit for bit, so start-up activations from
    # the cold-start anchors cannot make runs of different widths diverge.
    switch_s = job.get("switch_s")
    cfg = _config(SUPPRESSED if switch_s is not None else job["width"],
                  job["start"], job["duration_s"], events, job.get("dt_s", DT_S))
    factory = DroopProbeFactory(
        cfg, watch_buses=WATCH_BUSES,
        width_after=None if switch_s is None else (float(switch_s), job["width"]))
    t0 = time.perf_counter()
    failure, log = "", []
    with (out / "run.log").open("w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        try:
            log = run_multi_tso_dso(cfg, plant_factory=factory)
        except Exception:                                    # noqa: BLE001
            failure = traceback.format_exc(limit=5)
    plant = factory.plant
    if plant is not None:
        plant.frame().to_csv(out / "probe.csv.gz", index=False)
        plant.watch_frame().to_csv(out / "watch.csv", index=False)
    series = [{"t_s": float(r.time_s), "tso": bool(r.tso_active),
               "dso": bool(r.dso_active),
               **{f: dict(getattr(r, f, {}) or {}) for f in RECORD_FIELDS},
               "zone_oltc_taps": {int(z): list(map(float, v)) for z, v in
                                  (getattr(r, "zone_oltc_taps", None) or {}).items()},
               "zone_tso_shunt_states": {int(z): list(map(float, v)) for z, v in
                                         (getattr(r, "zone_tso_shunt_states", None)
                                          or {}).items()}}
              for r in log]
    with (out / "records.pkl").open("wb") as fh:
        pickle.dump(series, fh)
    done = {k: job[k] for k in ("case", "window", "width", "dt_s", "duration_s")
            if k in job}
    done.update(events=list(job.get("events", ())), n_records=len(series),
                width_switched_at_s=None if plant is None else plant.width_switched_at_s,
                failure=failure, wall_s=round(time.perf_counter() - t0, 1),
                pf_failed_steps=sum(1 for r in log if not getattr(r, "zone_v_mean", None)))
    (out / "done.json").write_text(json.dumps(done, indent=1, default=str), "utf-8")
    return done


def _pool(jobs: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    # Submit the worker by its importable module path: under ``python -m`` the
    # function defined here is ``__main__.run_case``, which spawned Windows
    # workers resolve less reliably than the package module.
    from experiments.ch_09_parameter_selection.ch_9_4_deadband import run_case as _run
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        futs = {ex.submit(_run, j): j["case"] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            print(f"  done {futs[f]}: {r.get('n_records')} records, "
                  f"{r.get('wall_s')} s{' FAILED' if r.get('failure') else ''}",
                  flush=True)
            results.append(r)
    return results


# -- D1 reduction -----------------------------------------------------------------------

def discrete_solves(run_dir: Path) -> set:
    """``(t_s, solve)`` of every solve at which a discrete event changes the plant.

    * a converter's capability band jumps by more than ``CAP_STEP_MVAR``,
      at the solve that first shows it;
    * an OLTC (TS or STS) or a bank operates, at the end-of-step solve of the
      dispatch that commanded it (the records carry the positions at dispatch
      steps only, so each is compared with the previous dispatch).

    Author decision 2026-10-27: D1 selects the width from the routine
    excursions WITHOUT these events; the excursions they cause are reported
    separately, together with the activations they produce.
    """
    p = pd.read_csv(run_dir / "probe.csv.gz").sort_values(["sgen", "t_s", "solve"])
    jump = (p.groupby("sgen")["q_max_mvar"].diff().abs() > CAP_STEP_MVAR) | \
           (p.groupby("sgen")["q_min_mvar"].diff().abs() > CAP_STEP_MVAR)
    keys = set(zip(p.loc[jump, "t_s"], p.loc[jump, "solve"]))
    last = p.groupby("t_s")["solve"].max()
    prev: Dict[str, Any] = {}
    for r in _records(run_dir):
        for f in ("zone_oltc_taps", "zone_tso_shunt_states", "dso_trafo_tap_pos"):
            cur = r[f]
            if not cur:
                continue
            old = prev.get(f)
            if old is not None and any(np.any(np.asarray(v) != np.asarray(old.get(k, v)))
                                       for k, v in cur.items()):
                if r["t_s"] in last.index:
                    keys.add((r["t_s"], int(last[r["t_s"]])))
            prev[f] = cur
    return keys


def excursions(probe: pd.DataFrame, discrete: set = frozenset()) -> pd.DataFrame:
    """One row per (converter, anchor interval): d_ik and its context.

    An interval runs from the solve that carries a new anchor to the last
    solve before the next one, so the pre-dispatch solve at t_k+1 (anchor still
    held) closes interval k and the post-dispatch solve opens k+1.
    ``discrete`` marks the intervals that contain a discrete-event solve
    (:func:`discrete_solves`): the step it causes persists until re-anchoring.
    ``after_discrete`` marks the converter's next interval, in which the
    controllers correct the step against an anchor set at the stepped voltage
    (author decision 2026-10-28: counted as part of the event).
    """
    df = probe.sort_values(["sgen", "t_s", "solve"]).reset_index(drop=True)
    new = df.groupby("sgen")["anchor_pu"].diff().fillna(1.0).ne(0.0)
    df["k"] = new.astype(int).groupby(df["sgen"]).cumsum()
    df["abs_v_eff"] = df["v_eff_pu"].abs()
    df["has_cap"] = (df["q_max_mvar"] - df["q_min_mvar"]) > 0.0
    df["disc"] = [(t, s) in discrete for t, s in zip(df["t_s"], df["solve"])]
    # Drift part of an interval: its solves before the first discrete event in
    # it (all of them if there is none), so excluding an event never hides the
    # drift that preceded it.
    after = df.groupby(["sgen", "k"])["disc"].cumsum() > 0
    df["abs_v_eff_drift"] = df["abs_v_eff"].where(~after)
    g = df.groupby(["sgen", "k"])
    out = pd.DataFrame({
        "level": g["level"].first(), "t_start_s": g["t_s"].first(),
        "t_end_s": g["t_s"].last(), "n_solves": g.size(),
        "d_pu": g["abs_v_eff"].max(), "has_cap": g["has_cap"].any(),
        "discrete": g["disc"].any(),
        "d_drift_pu": g["abs_v_eff_drift"].max().fillna(0.0),
    }).reset_index()
    out["after_discrete"] = out.groupby("sgen")["discrete"].shift(1, fill_value=False)
    return out


def select_width(d_max: float) -> Dict[str, Any]:
    cands = list(CANDIDATES)
    while not any(c >= d_max + EPS_U for c in cands):
        cands.append(round(cands[-1] + 0.005, 6))
    star = min(c for c in cands if c >= d_max + EPS_U)
    lower = [c for c in cands if c < star]
    narrower = max(lower) if lower else round(star - 0.0025, 6)
    return {"d_max_pu": d_max, "eps_u_pu": EPS_U, "candidates": cands,
            "delta_star_pu": star, "narrower_pu": narrower,
            "extended": len(cands) > len(CANDIDATES)}


def _records(path: Path) -> list:
    with (path / "records.pkl").open("rb") as fh:
        return pickle.load(fh)


def _ops(series: list) -> Dict[str, int]:
    """Tap operations (TS, STS) and bank commits over a run."""
    n = {"taps_ts": 0, "taps_sts": 0, "banks": 0}
    for a, b in zip(series[:-1], series[1:]):
        for z, v in b["zone_oltc_taps"].items():
            n["taps_ts"] += int(np.sum(np.abs(np.subtract(v, a["zone_oltc_taps"].get(z, v))) > 0))
        for k, v in b["dso_trafo_tap_pos"].items():
            n["taps_sts"] += int(abs(v - a["dso_trafo_tap_pos"].get(k, v)) > 0)
        for z, v in b["zone_tso_shunt_states"].items():
            n["banks"] += int(np.sum(np.abs(np.subtract(v, a["zone_tso_shunt_states"].get(z, v))) > 0))
    return n


def droop_delivered(p: pd.DataFrame) -> pd.Series:
    """Q delivered by the droop [Mvar]: realised Q minus the commanded Q as
    the capability admits it, where the dead zone is active, else 0.

    ``q_mvar - q_set_mvar`` alone also counts a command outside the
    capability band, which the plant clips with or without droop; and outside
    the dead zone the remainder is the convergence residual of the Q(V) loop
    (up to 0.1 Mvar), not droop action (fixed 2026-10-27 after run 0005's
    first reduction)."""
    dq = p["q_mvar"] - p["q_set_mvar"].clip(lower=p["q_min_mvar"],
                                            upper=p["q_max_mvar"])
    return dq.where(p["dz_active"].astype(bool), 0.0)


def confirm_metrics(run_dir: Path, ref_dir: Path, warmup_s: float) -> Dict[str, Any]:
    """Activations and deviation from the suppressed reference (D1 step 3)."""
    p = pd.read_csv(run_dir / "probe.csv.gz")
    r = pd.read_csv(ref_dir / "probe.csv.gz")
    cap = (p["q_max_mvar"] - p["q_min_mvar"]) > 0.0
    deliv = droop_delivered(p).abs() > TOL_Q_MVAR
    routine = p["t_s"] >= warmup_s
    # Rows in an anchor interval that contains a discrete event of this run.
    disc = discrete_solves(run_dir)
    q = p.sort_values(["sgen", "t_s", "solve"])
    k = q.groupby("sgen")["anchor_pu"].diff().fillna(1.0).ne(0.0).astype(int) \
        .groupby(q["sgen"]).cumsum()
    hit = pd.Series([(t, s) in disc for t, s in zip(q["t_s"], q["solve"])], index=q.index)
    iv = hit.groupby([q["sgen"], k]).any()
    # The correction interval after a discrete step counts with the event.
    iv = iv | iv.groupby(level=0).shift(1, fill_value=False)
    in_disc = pd.Series(iv.reindex(list(zip(q["sgen"], k))).to_numpy(),
                        index=q.index).reindex(p.index)
    out: Dict[str, Any] = {}
    for lev in ("TS", "STS"):
        m = p["level"] == lev
        out[f"activations_{lev}"] = int((p["dz_active"] & m).sum())
        out[f"activations_routine_{lev}"] = int((p["dz_active"] & m & routine).sum())
        out[f"activations_routine_drift_{lev}"] = int(
            (p["dz_active"] & m & routine & ~in_disc).sum())
        out[f"activations_routine_discrete_{lev}"] = int(
            (p["dz_active"] & m & routine & in_disc).sum())
        out[f"activations_with_capability_{lev}"] = int((p["dz_active"] & cap & m).sum())
        out[f"delivered_solves_{lev}"] = int((deliv & m).sum())
        if (p["dz_active"] & m & routine).any():
            out[f"first_routine_activation_{lev}_s"] = float(
                p.loc[p["dz_active"] & m & routine, "t_s"].min())
        out[f"max_abs_delivered_{lev}_mvar"] = float(
            droop_delivered(p[m]).abs().max() or 0.0)
    key = ["t_s", "solve", "sgen"]
    j = p.merge(r, on=key, suffixes=("", "_ref"))
    out["n_matched_solves"] = int(len(j))
    out["max_dv_der_bus_pu"] = float((j["v_pu"] - j["v_pu_ref"]).abs().max())
    out["max_dq_der_mvar"] = float((j["q_mvar"] - j["q_mvar_ref"]).abs().max())
    out["identical"] = bool(out["max_dv_der_bus_pu"] <= TOL_V_PU
                            and out["max_dq_der_mvar"] <= TOL_Q_MVAR)
    s, sr = _records(run_dir), _records(ref_dir)
    out["ops"], out["ops_ref"] = _ops(s), _ops(sr)
    dq = [abs(a["dso_trafo_q_actual_mvar"].get(k, 0.0) - v)
          for a, b in zip(s, sr) for k, v in b["dso_trafo_q_actual_mvar"].items()]
    out["max_d_interface_q_mvar"] = float(max(dq) if dq else 0.0)
    return out


# -- D2 reduction ------------------------------------------------------------------------

def d2_metrics(run_dir: Path, event: str, event_s: float, observe_s: float) -> Dict[str, Any]:
    bus, q_step = EVENTS[event]
    p = pd.read_csv(run_dir / "probe.csv.gz")
    w = pd.read_csv(run_dir / "watch.csv")
    pre = w[(w.t_s == event_s) & (w.solve == 0)]
    post = w[(w.t_s == event_s) & (w.solve > 0)]
    later = w[(w.t_s > event_s) & (w.t_s <= event_s + 60.0)]
    col = f"load_q_bus{bus}"
    out: Dict[str, Any] = {"event_bus": bus, "q_step_mvar": q_step}
    if pre.empty or post.empty:
        out["event_reached"] = False
        return out
    dq_load = float(post[col].iloc[-1] - pre[col].iloc[0])
    # At the event instant the profile is unchanged, so the step must be
    # exact; over the next minute the existing load follows its profile, so
    # persistence is judged with a tolerance far below the 200 Mvar step.
    out["event_reached"] = bool(abs(dq_load - q_step) < 1.0
                                and (later[col] - pre[col].iloc[0] - q_step).abs().max() < 10.0)
    out["load_q_change_mvar"] = dq_load
    out["dv_event_bus_pu"] = float(post[f"v_bus{bus}"].iloc[-1] - pre[f"v_bus{bus}"].iloc[0])
    ev = p[(p.t_s == event_s) & (p.solve > 0)]
    ev = ev[ev.solve == ev.solve.max()]
    for lev in ("TS", "STS"):
        e = ev[ev.level == lev]
        out[f"support_{lev}_mvar"] = float(droop_delivered(e).sum())
        out[f"support_unsat_{lev}_mvar"] = float(e.dq_unsat_mvar.sum())
        out[f"active_{lev}"] = int(e.dz_active.sum())
        out[f"max_abs_v_eff_{lev}_pu"] = float(e.v_eff_pu.abs().max())
    s = _records(run_dir)
    by_t = {round(r["t_s"], 3): r for r in s}
    r0, r1 = by_t.get(round(event_s - DT_S, 3)), by_t.get(round(event_s, 3))
    if r0 and r1:
        out["dv_zone_mean_pu"] = {z: r1["zone_v_mean"][z] - r0["zone_v_mean"].get(z, np.nan)
                                  for z in r1["zone_v_mean"]}
        out["dv_sts_mean_pu"] = {g: r1["dso_group_v_mean_pu"][g] - r0["dso_group_v_mean_pu"].get(g, np.nan)
                                 for g in r1["dso_group_v_mean_pu"]}
    obs = [r for r in s if event_s <= r["t_s"] <= event_s + observe_s]
    if obs:
        out["v_min_ts_pu"] = float(min(min(r["zone_v_min"].values()) for r in obs if r["zone_v_min"]))
        out["v_max_ts_pu"] = float(max(max(r["zone_v_max"].values()) for r in obs if r["zone_v_max"]))
        out["v_min_sts_pu"] = float(min(min(r["dso_group_v_min_pu"].values()) for r in obs if r["dso_group_v_min_pu"]))
        out["v_max_sts_pu"] = float(max(max(r["dso_group_v_max_pu"].values()) for r in obs if r["dso_group_v_max_pu"]))
    # hand-over: delivered droop support per end solve after the event
    h = p[(p.t_s >= event_s) & (p.t_s <= event_s + observe_s)]
    h = h.assign(dq=droop_delivered(h)).groupby(["t_s", "solve", "level"]).dq.sum().unstack("level")
    h.to_csv(run_dir / "support_series.csv")
    tot = h.sum(axis=1).abs()
    peak = float(tot.max()) if len(tot) else 0.0
    out["support_peak_mvar"] = peak
    if peak > 1.0:
        above = tot[tot > max(0.1 * peak, 1.0)]
        out["handover_s"] = float(above.index.get_level_values("t_s").max() - event_s + DT_S)
    return out


# -- orchestration ---------------------------------------------------------------------------

def _windows(names=None):
    return [(n, s) for n, s, *_ in CONDITIONS if names is None or n in names]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="W1 only, short runs, one event: a pipeline check")
    ap.add_argument("--stage1-from", default=None,
                    help="an earlier run of this script whose suppressed D1 "
                         "traces and dt check are copied and reused (same "
                         "protocol constants); stage 2 is simulated anew")
    ap.add_argument("--reduce", default=None,
                    help="an existing run of this script: append an invocation "
                         "and recompute its reductions; every simulation is "
                         "cached (done.json), so nothing is re-simulated")
    add_run_arguments(ap)
    args = ap.parse_args(argv)

    d1_s = 0.25 * 3600.0 if args.smoke else D1_HOURS * 3600.0
    ev_s = 610.0 if args.smoke else EVENT_S
    obs_s = 300.0 if args.smoke else OBSERVE_S
    win = _windows(("W1",) if args.smoke else None)
    events = ("b7_ind",) if args.smoke else tuple(EVENTS)
    warmup = 300.0 if args.smoke else WARMUP_S

    params = {"paramset": SELECTED_PARAMSET, "dt_s": DT_S, "candidates_pu": list(CANDIDATES),
              "eps_u_pu": EPS_U, "suppressed_pu": SUPPRESSED, "d1_hours": d1_s / 3600.0,
              "warmup_s": warmup, "event_s": ev_s, "observe_s": obs_s,
              "events": {k: EVENTS[k] for k in events}, "windows": [w for w, _ in win],
              "measurement_noise": False, "tested_width_from_s": warmup,
              "d1_rule": "drift (routine intervals without a discrete event)",
              "cap_step_mvar": CAP_STEP_MVAR, "design_margin_steps": DESIGN_STEPS,
              "stage1_from": args.stage1_from}
    with start_thesis_run(
            __file__, target=["ch:timescales:deadband"],
            feeds=["ch:timescales:deadband:choice", "ch:param:experiment:f2"],
            paramset=SELECTED_PARAMSET, parameters=params,
            time_series=(f"W1-W6 of tuning.scenarios_mc_v2.CONDITIONS, D1 "
                         f"{d1_s / 3600:g} h, D2 {(ev_s + obs_s) / 3600:g} h; "
                         f"dt {DT_S:g} s, T_TS 180 s, T_STS 20 s"),
            note=args.note, tag=args.tag or ("smoke" if args.smoke else None),
            resume=args.reduce) as run:
        root = run.root
        if args.stage1_from and not args.reduce:
            # Reuse the finished suppressed traces: their done.json makes the
            # stage-1 jobs below return as cached, and the reduction reruns.
            import shutil
            src = Path(args.stage1_from)
            for sub in ("suppressed", "dtcheck_W1_dt20"):
                shutil.copytree(src / "d1" / sub, root / "d1" / sub)
        # Stage 1: suppressed traces (+ the dt check: W1 at dt 20 s)
        jobs = [{"case": f"d1_ref_{w}", "window": w, "start": s, "width": SUPPRESSED,
                 "duration_s": d1_s, "out": str(root / "d1" / "suppressed" / w)}
                for w, s in win]
        jobs.append({"case": "dtcheck_W1_dt20", "window": "W1", "start": win[0][1],
                     "width": SUPPRESSED, "duration_s": d1_s, "dt_s": 20.0,
                     "out": str(root / "d1" / "dtcheck_W1_dt20")})
        print(f"stage 1: {len(jobs)} runs", flush=True)
        _pool(jobs, args.workers)

        ex = []
        for w, _s in win:
            d = root / "d1" / "suppressed" / w
            e = excursions(pd.read_csv(d / "probe.csv.gz"), discrete_solves(d))
            ex.append(e.assign(window=w))
        ex = pd.concat(ex, ignore_index=True)
        ex["startup"] = ex["t_start_s"] < warmup
        ex.to_csv(root / "d1" / "excursions.csv.gz", index=False)
        routine = ex[~ex.startup]
        drift = routine.assign(d_pu=routine["d_drift_pu"].where(~routine["after_discrete"]))
        # Drift-based rule (author decisions 2026-10-27/28): the width covers the
        # routine excursions up to the first discrete event of each interval;
        # the converter's next interval, the correction of the step, counts with
        # the event.  The rule as first written (all routine intervals) is
        # reported alongside.
        sel = select_width(float(drift["d_pu"].max()))
        sel["rule"] = ("drift: routine intervals up to a discrete event (capability "
                       "step > CAP_STEP_MVAR, OLTC or bank operation); the "
                       "correction interval after it counts with the event")
        sel["n_correction_intervals"] = int((routine.after_discrete & ~routine.discrete).sum())
        sel["cap_step_mvar"] = CAP_STEP_MVAR
        sel["d_max_all_routine_pu"] = float(routine["d_pu"].max())
        sel["delta_rule_as_written_pu"] = select_width(sel["d_max_all_routine_pu"])["delta_star_pu"]
        sel["n_routine_intervals"] = int(len(routine))
        sel["n_discrete_intervals"] = int(routine.discrete.sum())
        sel["discrete_above_delta_star"] = int((routine.discrete
                                                & (routine.d_pu > sel["delta_star_pu"])).sum())
        sel["d_max_with_capability_pu"] = float(drift.loc[drift.has_cap, "d_pu"].max())
        sel["d_max_startup_pu"] = float(ex.loc[ex.startup, "d_pu"].max()) if ex.startup.any() else None
        sel["per_window_level"] = (drift.groupby(["window", "level"]).d_pu.max()
                                   .unstack("level").round(6).to_dict(orient="index"))
        sel["per_window_level_all_routine"] = (
            routine.groupby(["window", "level"]).d_pu.max()
            .unstack("level").round(6).to_dict(orient="index"))
        dt20 = excursions(pd.read_csv(root / "d1" / "dtcheck_W1_dt20" / "probe.csv.gz"))
        sel["dtcheck_W1_d_max_dt20_pu"] = float(dt20.loc[dt20.t_start_s >= warmup, "d_pu"].max())
        sel["dtcheck_W1_d_max_dt10_pu"] = float(routine.loc[routine.window == "W1", "d_pu"].max())
        # Design value (author decision 2026-10-28): the rule gives the smallest
        # admissible width; the design width is DESIGN_STEPS candidates wider,
        # a declared margin for what the QSS traces do not contain (infeed
        # fluctuations below the 15-min profile step, measurement noise) and
        # the width at which Experiment B selected the weights.  The confirmation
        # runs and D2 use the design width; its narrower neighbour is the rule's.
        cands = sel["candidates"]
        sel["delta_rule_pu"] = sel["delta_star_pu"]
        i = cands.index(sel["delta_rule_pu"]) + DESIGN_STEPS
        while i >= len(cands):
            cands.append(round(cands[-1] + 0.005, 6))
        sel["design_margin_steps"] = DESIGN_STEPS
        sel["delta_design_pu"] = cands[i]
        sel["delta_star_pu"] = sel["delta_design_pu"]      # what stage 2 tests
        sel["narrower_pu"] = cands[i - 1]
        (root / "d1" / "selection.json").write_text(json.dumps(sel, indent=1), "utf-8")
        star, narrow = sel["delta_star_pu"], sel["narrower_pu"]
        print(f"D1: d_max {sel['d_max_pu']:.5f} pu -> rule {sel['delta_rule_pu']} pu, "
              f"design {star} pu (narrower {narrow})", flush=True)

        # Stage 2: D1 confirmation and D2, one pool
        legs = {"star": star, "wide": round(2.0 * star, 6), "off": SUPPRESSED}
        jobs = [{"case": f"d1_{tag}_{w}", "window": w, "start": s, "width": width,
                 "switch_s": warmup, "duration_s": d1_s,
                 "out": str(root / "d1" / f"confirm_{tag}" / w)}
                for tag, width in (("star", star), ("narrower", narrow)) for w, s in win]
        jobs += [{"case": f"d2_{w}_{e}_{leg}", "window": w, "start": s, "width": width,
                  "switch_s": warmup, "duration_s": ev_s + obs_s, "events": (e,),
                  "event_s": ev_s, "out": str(root / "d2" / w / e / leg)}
                 for w, s in win for e in events for leg, width in legs.items()]
        print(f"stage 2: {len(jobs)} runs", flush=True)
        _pool(jobs, args.workers)

        conf = {tag: {w: confirm_metrics(root / "d1" / f"confirm_{tag}" / w,
                                         root / "d1" / "suppressed" / w, warmup)
                      for w, _ in win}
                for tag in ("star", "narrower")}
        (root / "d1" / "confirmation.json").write_text(json.dumps(conf, indent=1), "utf-8")
        rows = [dict(window=w, event=e, leg=leg, width=width,
                     **d2_metrics(root / "d2" / w / e / leg, e, ev_s, obs_s))
                for w, _s in win for e in events for leg, width in legs.items()]
        d2 = pd.DataFrame(rows)
        # "All legs start from the same pre-event state": the converter-bus
        # voltages at the pre-event solve, each leg against the droop-off leg.
        pre_dv = {}
        for w, _s in win:
            for e in events:
                pre = {leg: pd.read_csv(root / "d2" / w / e / leg / "probe.csv.gz")
                       .query("t_s == @ev_s and solve == 0").set_index("sgen").v_pu
                       for leg in legs}
                for leg in legs:
                    pre_dv[(w, e, leg)] = float((pre[leg] - pre["off"]).abs().max())
        d2["pre_event_max_dv_vs_off_pu"] = [pre_dv[(r.window, r.event, r.leg)]
                                            for r in d2.itertuples()]
        d2.to_csv(root / "d2" / "summary.csv", index=False)
        n_bad = int((~d2["event_reached"].astype(bool)).sum())
        if n_bad:
            run.fail(f"{n_bad} D2 cell(s) whose event did not reach the plant")
        failed = [p.parent.name for p in root.rglob("done.json")
                  if json.loads(p.read_text("utf-8")).get("failure")]
        if failed:
            run.fail(f"{len(failed)} run(s) failed: {failed[:5]}")

        run.describe_output("d1/", "D1: suppressed traces, dt check, excursions.csv.gz "
                                   "(d_ik), selection.json (Delta*), confirmation runs "
                                   "and confirmation.json")
        run.describe_output("d2/", "D2: <window>/<event>/<leg>/ runs; summary.csv "
                                   "(event check, support, voltage deviation, extrema, "
                                   "hand-over); support_series.csv per cell")
        print(json.dumps({"delta_star_pu": star, "narrower_pu": narrow,
                          "d2_events_not_reached": n_bad, "failed_runs": failed}),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
