#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_11_case2_multisystem/mc/run_mc.py
================================================

Monte-Carlo ensemble of Part III of the thesis (``ch:setup:montecarlo``,
``ch:case3:montecarlo``).  The first slice runs L2, S1 and O2 -- the
"no sub-transmission access" row of Case 2, which also carries the paired
statistics of Case 1 (``tab:case_svr:stats``).  S2, O3 and M are added later on
the SAME episodes (same master seed), once implemented.

Successor of ``experiments/ch_11_case2_multisystem/ladder/ladder_mc.py`` (machinery:
seeded paired episodes, slot-based event draw, base-case prescreen, process
pool, compact per-episode files).  What changed, and why:

* configuration from ``experiments.ch_10_case1_svr.run_case1.resolved`` (paramset
  ``ch9_selected``, DSO_3 x2) instead of ``make_thesis_config`` with the old
  pinned ``g_w_gen = 1e9``: the ensemble must test the controller Chapter 8
  selects and Chapter 9 shows;
* discard rule (declared 2026-10-07, before any run): a draw is discarded ONLY
  if it fails the base-case prescreen.  A failing variant is an OUTCOME, never
  a reason to discard -- the old rule (discard if any of seven variants fails)
  rejected two thirds of the draws and conditioned the sample on the weakest
  variant surviving;
* N = 100 valid episodes, fixed in advance (no extension after looking at the
  confidence intervals);
* start times exclude the calendar days of the design windows W1--W6 (App. B
  rule: validation conditions are disjoint from the design conditions); the
  reference period (05 Jan 2016, 08:00) is W1's day and is excluded with it;
* every disturbance is reverted after ``REVERT_MIN`` if that falls inside the
  horizon, as in the reference period (trip -> restore, connect -> trip);
* load events are purely reactive, as in the reference period;
* metrics are those of Chapter 9/10 (``report_case1.area_metrics``), with the
  normal operating range [0.95, 1.05] pu and the non-inferiority margin
  delta = 1 mpu per area (both declared 2026-10-07).

Changes of 2026-10-28 (author-approved plan
``docs/architecture/2026-10-28_case2_step_mc_plan.md`` §5), before any Case-2
ensemble run:

* configuration from the Case-2 registry ``variants.resolved`` (paramset
  of record, minimum-class noise for every controller and the droop) for all
  six variants L2, S1, S2, O2, O3, M; O3 on 300 episodes;
* every disturbance and its reversal between two TS dispatches: phase uniform
  on {20, 40, ..., 160} s, drawn per event from ``default_rng([seed, 1])`` --
  start times and event lists of the declared draw are unchanged;
* the episode seed is also the noise seed (one realisation per episode,
  shared by all variants);
* metrics from ``metrics_case2`` (area incl. u_q and reserve, STS, interface,
  Delta e_post and withdrawn support per event); the full log is kept
  (``runs/<seed>_<variant>.log.pkl``).

Declared campaign constants (do not change once results exist)
--------------------------------------------------------------
MASTER_SEED 20261007, episode k has seed MASTER_SEED + k; N_VALID 100;
HORIZON_MIN 180; SLOT_MIN 30, SLOT_PROB 0.25; REVERT_MIN 90; reactive load
U[100, 400] Mvar; DELTA_MPU 1.0; V_LO/V_HI 0.95/1.05; hard band 0.90/1.10.

Outcomes per (episode, variant)
-------------------------------
``ok``                completed, none of the failure criteria met
``fail_unreachable``  aborted by the reachability guard (``ch:setup:framework:
                      reachability``) -- reported as its own category
``fail_abort``        any other exception, or a short log
``fail_pf``           at least one step without a converged power flow
``fail_v_end``        a monitored voltage outside [0.90, 1.10] pu at the end
``fail_osc``          a transformer tap reverses direction twice or more in
                      the last 30 min (actuator cycling at the horizon end)
An episode on which every variant run so far fails is reported as
``infeasible`` and kept out of the failure bounds.

Run (see README_server.md)
--------------------------
    python -m experiments.ch_11_case2_multisystem.mc.run_mc --jobs 30        # simulate
    python -m experiments.ch_11_case2_multisystem.mc.run_mc --jobs 30        # resume = same call
    python -m experiments.ch_11_case2_multisystem.mc.run_mc --report         # tables only
    python -m experiments.ch_11_case2_multisystem.mc.run_mc --smoke --jobs 4 # 2 episodes x 10 min
    python -m experiments.ch_11_case2_multisystem.mc.run_mc --new --jobs 30  # fresh campaign

The campaign is ONE documented run folder,
``results/ch_11_case2_multisystem/mc/<NNNN>_<stamp>_run_mc[_smoke]/``
(``experiments/thesis_runs.py``).  A call without ``--new`` continues the
newest campaign with the same tag and is recorded as a further invocation in
its README -- that is how S2, O3 and M join the same episodes later.

Author: Manuel Schwenke / Claude Code (2026-10-07)
"""
from __future__ import annotations

import os

# Headless + UTF-8 + one BLAS thread per worker; must precede numpy imports.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import dataclasses
import datetime as _dt
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))           # the project root

from experiments.ch_10_case1_svr.run_case1 import PILOT_BUSES  # noqa: E402
from experiments.ch_11_case2_multisystem import metrics_case2 as mc2  # noqa: E402
from experiments.ch_11_case2_multisystem import variants as case2  # noqa: E402

PARAMSET = case2.PARAMSET


def resolved(name: str, minutes: int, events=(), noise_seed: Optional[int] = None):
    """The Case-2 configuration of ``name`` (variants.py): selected paramset,
    noise for every controller and the droop, the given events and seed."""
    return case2.resolved(name, minutes, events, noise_seed)
from experiments.helpers.records import ContingencyEvent  # noqa: E402
from experiments.thesis_runs import (  # noqa: E402
    MANIFEST, add_run_arguments, annotate_run, latest_run, start_thesis_run,
)

#: Where the ensemble is reported, and what it feeds (thesis labels).
TARGET = ("ch:case3:montecarlo", "ch:setup:montecarlo")
FEEDS = ("tab:case3:mc", "fig:case3:mc", "tab:case_svr:stats")

# ---------------------------------------------------------------------------
#  Declared campaign constants (2026-10-07)
# ---------------------------------------------------------------------------
MASTER_SEED = 20261007
N_VALID = 100
HORIZON_MIN = 180

SLOT_MIN = 30                 # candidate event instants 30, 60, ..., < horizon
SLOT_PROB = 0.25              # firing probability per slot
REVERT_MIN = 90               # every disturbance is reverted after this long
LOAD_Q_MIN_MVAR = 100.0       # purely reactive load step, as in the reference
LOAD_Q_MAX_MVAR = 400.0

#: Event phase (author decision 2026-10-27/28): every disturbance and its
#: reversal fall between two TS dispatches, uniformly on this grid of seconds
#: after the slot's dispatch instant, drawn per event from the child stream
#: ``default_rng([seed, 1])`` so the declared draw of start times and events
#: is unchanged.
PHASE_GRID_S = tuple(float(x) for x in range(20, 180, 20))
#: O3 runs on more episodes (failure bound below 1 % at 95 % with 0 of 300).
N_VALID_O3 = 300

DELTA_MPU = 1.0               # non-inferiority margin, per area, pilot error
V_LO, V_HI = 0.95, 1.05       # normal operating range ("time outside")
V_HARD_LO, V_HARD_HI = 0.90, 1.10
OSC_WINDOW_MIN = 30
OSC_REVERSALS = 2

#: Coarse prescreen step: equal to T_TS and a multiple of T_STS, so no period
#: of the configuration falls between two steps.
PRESCREEN_DT_S = 180.0

#: Calendar days of the design windows W1--W6 (App. B, tab:timescales:
#: conditions).  W1 (05 Jan) is also the day of the reference period.
EXCLUDED_DAYS = {_dt.date(2016, 1, 4), _dt.date(2016, 1, 5),
                 _dt.date(2016, 2, 3), _dt.date(2016, 2, 22),
                 _dt.date(2016, 7, 18), _dt.date(2016, 11, 26)}
PROFILE_START = _dt.datetime(2016, 1, 1, 0, 0)
PROFILE_LAST = _dt.datetime(2016, 12, 31, 23, 45)

#: Variants this driver can run.  Case-2 variants (S2, O3, M) are added here
#: when implemented; their episodes are the same seeds.
VARIANTS: Dict[str, Dict[str, Any]] = dict(case2.VARIANTS)
ALL_VARIANTS = ["S1", "L2", "O2", "S2", "O3", "M"]   # longest first: pool packing

#: Pre-declared comparisons: (candidate, comparator) -- the family of the
#: plan of 2026-10-28 §5 (NI on e_vp; superiority on e_v, worst, t_out).
COMPARISONS: List[Tuple[str, str]] = [("O2", "S1"), ("O3", "S1"), ("O3", "S2"),
                                      ("O3", "O2"), ("S2", "S1")]


def _dirs(out: Path) -> Dict[str, Path]:
    return {"out": out, "pre": out / "prescreen", "runs": out / "runs",
            "logs": out / "logs", "scratch": out / "_scratch"}


# ---------------------------------------------------------------------------
#  Episode draw (deterministic given the seed)
# ---------------------------------------------------------------------------
_ELEMENTS: Optional[Dict[str, List[int]]] = None


def enumerate_elements() -> Dict[str, List[int]]:
    """Admissible event elements: non-slack machines, transmission lines whose
    outage keeps the transmission graph connected, transmission load buses.

    Built exactly as ``run_multi_tso_dso`` builds the network; transmission
    element indices are created before the sub-transmission systems and do not
    depend on the variant (shunt flag) or on the DSO_3 scaling.
    """
    global _ELEMENTS
    if _ELEMENTS is not None:
        return _ELEMENTS
    import pandapower.topology as top
    from network.ieee39 import build_ieee39_net, add_hv_networks
    from network.zone_partition import fixed_zone_partition_ieee39

    cfg = resolved("L2", HORIZON_MIN)
    net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                                 verbose=False)
    fixed_zone_partition_ieee39(net, verbose=False)
    meta = add_hv_networks(net, meta, install_tso_tertiary_shunts=False,
                           verbose=False)
    gen_trafo = {int(g) for g in meta.machine_trafo_gen_map
                 if g is not None and int(g) >= 0}
    slack = {int(g) for g in net.gen.index if bool(net.gen.at[g, "slack"])}
    gens = sorted(g for g in gen_trafo
                  if g not in slack and bool(net.gen.at[g, "in_service"]))

    # Transmission level by nominal voltage (345 kV).  ``hv.bus_indices`` lists
    # only part of each 110 kV network, so it cannot be used to exclude it.
    import networkx as nx
    tn_buses = {int(b) for b in net.bus.index
                if float(net.bus.at[b, "vn_kv"]) >= 200.0
                and bool(net.bus.at[b, "in_service"])}
    tn_lines = sorted(int(li) for li in net.line.index
                      if bool(net.line.at[li, "in_service"])
                      and int(net.line.at[li, "from_bus"]) in tn_buses
                      and int(net.line.at[li, "to_bus"]) in tn_buses)
    # N-1 admissible: the outage must not split the 345 kV network.  Paths
    # through the 110 kV networks do not count as a connection.
    g_all = top.create_nxgraph(net, multi=True, include_trafo3ws=False)
    g_tn = g_all.subgraph(tn_buses).copy()
    lines = []
    for li in tn_lines:
        g = g_tn.copy()
        fb, tb = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
        for key in [k for k in g[fb][tb] if k == ("line", li)]:
            g.remove_edge(fb, tb, key)
        if nx.number_connected_components(g) == 1:
            lines.append(li)

    pv = ({int(net.gen.at[g, "bus"]) for g in net.gen.index}
          | {int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index})
    tn_pq = {b for b in tn_buses if b not in pv}
    load_buses = sorted({int(net.load.at[i, "bus"]) for i in net.load.index
                         if int(net.load.at[i, "bus"]) in tn_pq})
    _ELEMENTS = {"gens": gens, "lines": lines, "load_buses": load_buses,
                 "lines_excluded_islanding": sorted(set(tn_lines) - set(lines))}
    return _ELEMENTS


def draw_start(rng: np.random.Generator) -> _dt.datetime:
    """Uniform over the profile year in 15-min steps; the horizon must fit and
    must not touch a design-window day."""
    span = int((PROFILE_LAST - PROFILE_START).total_seconds() // 60) - HORIZON_MIN
    for _ in range(1000):
        off = int(rng.integers(0, span))
        off -= off % 15
        t0 = PROFILE_START + _dt.timedelta(minutes=off)
        t1 = t0 + _dt.timedelta(minutes=HORIZON_MIN)
        days = {t0.date(), t1.date()}
        if not days & EXCLUDED_DAYS:
            return t0
    raise RuntimeError("no admissible start time in 1000 draws")


def _phase(rng_phase: Optional[np.random.Generator]) -> float:
    if rng_phase is None:
        return 0.0
    return PHASE_GRID_S[int(rng_phase.integers(len(PHASE_GRID_S)))]


def draw_events(rng: np.random.Generator, el: Dict[str, List[int]],
                rng_phase: Optional[np.random.Generator] = None
                ) -> List[ContingencyEvent]:
    """Slot draw.  Each slot fires with SLOT_PROB; the kind is uniform over the
    kinds that are admissible at that instant (at most one element of each
    kind away from normal at any time); every disturbance is reverted after
    REVERT_MIN if that falls inside the horizon."""
    ev: List[ContingencyEvent] = []
    busy_until = {"gen": -1, "line": -1, "load": -1}
    for m in range(SLOT_MIN, HORIZON_MIN, SLOT_MIN):
        if rng.random() >= SLOT_PROB:
            continue
        kinds = [k for k in ("gen", "line", "load")
                 if busy_until[k] <= m and el[{"gen": "gens", "line": "lines",
                                               "load": "load_buses"}[k]]]
        if not kinds:
            continue
        kind = kinds[int(rng.integers(len(kinds)))]
        back = m + REVERT_MIN
        ph = _phase(rng_phase)
        t_on, t_off = 60.0 * m + ph, 60.0 * back + ph
        if kind == "gen":
            g = int(el["gens"][int(rng.integers(len(el["gens"])))])
            ev.append(ContingencyEvent(minute=m, time_s=t_on, element_type="gen",
                                       element_index=g, action="trip"))
            if back < HORIZON_MIN:
                ev.append(ContingencyEvent(minute=back, time_s=t_off,
                                           element_type="gen",
                                           element_index=g, action="restore"))
        elif kind == "line":
            li = int(el["lines"][int(rng.integers(len(el["lines"])))])
            ev.append(ContingencyEvent(minute=m, time_s=t_on, element_type="line",
                                       element_index=li, action="trip"))
            if back < HORIZON_MIN:
                ev.append(ContingencyEvent(minute=back, time_s=t_off,
                                           element_type="line",
                                           element_index=li, action="restore"))
        else:
            b = int(el["load_buses"][int(rng.integers(len(el["load_buses"])))])
            q = round(float(rng.uniform(LOAD_Q_MIN_MVAR, LOAD_Q_MAX_MVAR)), 1)
            ev.append(ContingencyEvent(minute=m, time_s=t_on, element_type="load",
                                       bus=b, p_mw=0.0, q_mvar=q, action="connect"))
            if back < HORIZON_MIN:
                ev.append(ContingencyEvent(minute=back, time_s=t_off,
                                           element_type="load",
                                           bus=b, p_mw=0.0, q_mvar=q,
                                           action="trip"))
        busy_until[kind] = back
    ev.sort(key=lambda e: (e.effective_time_s, e.element_type, e.action))
    return ev


def draw_episode(seed: int) -> Tuple[_dt.datetime, List[ContingencyEvent]]:
    rng = np.random.default_rng(seed)
    t0 = draw_start(rng)
    return t0, draw_events(rng, enumerate_elements(),
                           np.random.default_rng([seed, 1]))


def events_to_rows(events: List[ContingencyEvent]) -> List[Dict[str, Any]]:
    return [{"minute": int(e.minute), "time_s": float(e.effective_time_s),
             "element_type": str(e.element_type),
             "element_index": int(getattr(e, "element_index", -1)
                                  if getattr(e, "element_index", None) is not None
                                  else -1),
             "bus": int(getattr(e, "bus", -1) if getattr(e, "bus", None)
                        is not None else -1),
             "q_mvar": float(getattr(e, "q_mvar", 0.0) or 0.0),
             "action": str(e.action)} for e in events]


def episode_class(events: List[ContingencyEvent]) -> str:
    n = sum(1 for e in events if e.action in ("trip", "connect")
            and not (e.element_type == "load" and e.action == "trip"))
    return "undisturbed" if n == 0 else ("single" if n == 1 else "multiple")


# ---------------------------------------------------------------------------
#  One task = one (seed, variant) run, or one prescreen
# ---------------------------------------------------------------------------
def _config(name: str, start: _dt.datetime, events: List[ContingencyEvent],
            scratch: Path, minutes: int, dt_s: Optional[float] = None,
            noise_seed: Optional[int] = None):
    """Case-2 configuration; the episode seed is also its noise seed, so all
    variants of an episode share one realisation (author decision 2026-10-27)."""
    cfg = resolved(name, minutes, list(events), noise_seed)
    over = dict(start_time=start, result_dir=str(scratch), verbose=0)
    if dt_s is not None:
        over["dt_s"] = float(dt_s)
    return dataclasses.replace(cfg, **over)


def _run(cfg) -> Tuple[Optional[list], Optional[BaseException]]:
    from experiments.runners.multi_tso_dso import run_multi_tso_dso
    try:
        return run_multi_tso_dso(cfg), None
    except BaseException as exc:          # noqa: BLE001 -- recorded as outcome
        return getattr(exc, "partial_log", None), exc


def prescreen_task(seed: int, out: str, minutes: int) -> Dict[str, Any]:
    """Base-case probe: L2, no events, coarse step.  Always writes its JSON
    (a crash of the task itself counts as an invalid draw), so the seed
    sequence never stalls."""
    try:
        return _prescreen(seed, out, minutes)
    except Exception as exc:                       # noqa: BLE001
        d = _dirs(Path(out))
        rec = {"seed": seed, "valid": False, "start_time": None, "events": [],
               "class": None, "reason": f"task_crash: {type(exc).__name__}: {exc}",
               "wall_s": 0.0}
        (d["pre"] / f"{seed}.json").write_text(json.dumps(rec, indent=2),
                                               encoding="utf-8")
        return rec


def _prescreen(seed: int, out: str, minutes: int) -> Dict[str, Any]:
    d = _dirs(Path(out))
    start, events = draw_episode(seed)
    scratch = d["scratch"] / f"pre_{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = _config("L2", start, [], scratch, minutes, dt_s=PRESCREEN_DT_S,
                  noise_seed=seed)
    n_steps = int(round(cfg.n_total_s / cfg.dt_s))
    logf = d["logs"] / f"pre_{seed}.log"
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        log, exc = _run(cfg)
    ok = exc is None and log is not None and len(log) == n_steps
    rec = {"seed": seed, "valid": bool(ok), "start_time": start.isoformat(),
           "events": events_to_rows(events), "class": episode_class(events),
           "reason": None if ok else (type(exc).__name__ if exc else
                                      f"short_log_{len(log or [])}of{n_steps}"),
           "wall_s": time.time() - t0}
    (d["pre"] / f"{seed}.json").write_text(json.dumps(rec, indent=2),
                                           encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    return rec


def _tap_reversals(series: List[np.ndarray]) -> int:
    """Largest number of direction reversals of any single tap position."""
    if len(series) < 3:
        return 0
    try:
        a = np.vstack([np.asarray(s, dtype=float) for s in series])
    except ValueError:
        return 0
    worst = 0
    for j in range(a.shape[1]):
        d = np.sign(np.diff(a[:, j]))
        d = d[d != 0]
        worst = max(worst, int(np.sum(d[1:] != d[:-1])) if d.size > 1 else 0)
    return worst


def classify(log: Optional[list], exc: Optional[BaseException], n_steps: int,
             dt_s: float) -> Tuple[str, Dict[str, Any]]:
    diag: Dict[str, Any] = {}
    if exc is not None:
        from analysis.reachability import ReachabilityViolation
        diag["exception"] = f"{type(exc).__name__}: {exc}"[:500]
        if isinstance(exc, ReachabilityViolation):
            return "fail_unreachable", diag
        return "fail_abort", diag
    if log is None or len(log) != n_steps:
        diag["n_records"] = len(log or [])
        return "fail_abort", diag
    n_gap = sum(1 for r in log if not getattr(r, "bus_vm_pu", None))
    diag["n_pf_fail_steps"] = n_gap
    last = log[-1]
    vmin = (float(np.nanmin(list(last.zone_v_min.values())))
            if last.zone_v_min else np.nan)
    vmax = (float(np.nanmax(list(last.zone_v_max.values())))
            if last.zone_v_max else np.nan)
    diag["v_end_min"], diag["v_end_max"] = float(vmin), float(vmax)
    k = int(round(OSC_WINDOW_MIN * 60.0 / dt_s))
    tail = log[-k:]
    ts_taps = {}
    for r in tail:
        for z, a in (getattr(r, "zone_oltc_taps", None) or {}).items():
            ts_taps.setdefault(z, []).append(np.asarray(a, dtype=float))
    sts_taps = [np.asarray([r.dso_trafo_tap_pos[key] for key in
                            sorted(r.dso_trafo_tap_pos)], dtype=float)
                for r in tail if getattr(r, "dso_trafo_tap_pos", None)]
    rev = max([_tap_reversals(s) for s in ts_taps.values()] + [0])
    rev_sts = _tap_reversals(sts_taps)
    diag["tap_reversals_end_ts"], diag["tap_reversals_end_sts"] = rev, rev_sts
    if n_gap:
        return "fail_pf", diag
    if not (V_HARD_LO <= vmin and vmax <= V_HARD_HI):
        return "fail_v_end", diag
    if max(rev, rev_sts) >= OSC_REVERSALS:
        return "fail_osc", diag
    return "ok", diag


def _series(log: list) -> Dict[str, np.ndarray]:
    """Compact per-step series for later criteria (float32)."""
    z_all = sorted(PILOT_BUSES)
    out = {"t_min": np.array([r.time_s / 60.0 for r in log], dtype=np.float32)}
    for z in z_all:
        p = PILOT_BUSES[z]
        out[f"e_v_{z}"] = np.array([(r.zone_v_rms_err_pu or {}).get(z, np.nan)
                                    for r in log], dtype=np.float32)
        out[f"v_pilot_{z}"] = np.array([(r.bus_vm_pu or {}).get(p, np.nan)
                                        for r in log], dtype=np.float32)
        out[f"v_min_{z}"] = np.array([(r.zone_v_min or {}).get(z, np.nan)
                                      for r in log], dtype=np.float32)
        out[f"v_max_{z}"] = np.array([(r.zone_v_max or {}).get(z, np.nan)
                                      for r in log], dtype=np.float32)
    return out


def variant_task(seed: int, name: str, out: str, minutes: int) -> Dict[str, Any]:
    """One variant on one valid episode.  Always writes its JSON; a crash of
    the task itself (not of the simulation) is recorded as ``task_crash`` and
    is NOT an outcome of the controller -- rerun those after fixing."""
    try:
        return _variant(seed, name, out, minutes)
    except Exception as exc:                       # noqa: BLE001
        d = _dirs(Path(out))
        rec = {"seed": seed, "variant": name, "outcome": "task_crash",
               "diag": {"exception": f"{type(exc).__name__}: {exc}"[:500],
                        "traceback": traceback.format_exc()[-2000:]},
               "wall_s": 0.0, "n_records": 0, "start_time": None, "class": None}
        (d["runs"] / f"{seed}_{name}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
        return {"seed": seed, "variant": name, "outcome": "task_crash",
                "wall_s": 0.0}


def _variant(seed: int, name: str, out: str, minutes: int) -> Dict[str, Any]:
    d = _dirs(Path(out))
    pre = json.loads((d["pre"] / f"{seed}.json").read_text(encoding="utf-8"))
    start, events = draw_episode(seed)
    assert start.isoformat() == pre["start_time"], "episode draw not reproducible"
    scratch = d["scratch"] / f"{seed}_{name}"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = _config(name, start, events, scratch, minutes, noise_seed=seed)
    n_steps = int(round(cfg.n_total_s / cfg.dt_s))
    logf = d["logs"] / f"{seed}_{name}.log"
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        log, exc = _run(cfg)
    wall = time.time() - t0
    outcome, diag = classify(log, exc, n_steps, float(cfg.dt_s))
    rec: Dict[str, Any] = {"seed": seed, "variant": name, "outcome": outcome,
                           "diag": diag, "wall_s": wall,
                           "n_records": len(log or []),
                           "start_time": pre["start_time"], "class": pre["class"]}
    if outcome == "ok" or (log and len(log) == n_steps):
        refs = {z: float((cfg.zone_v_setpoints_pu or {}).get(z, cfg.v_setpoint_pu))
                for z in PILOT_BUSES}
        t_ts, dt = float(cfg.tso_period_s), float(cfg.dt_s)
        ev_rows = mc2.event_metrics(log, t_ts, dt)
        rec["metrics"] = {}
        for z in sorted(PILOT_BUSES):
            m = mc2.area_metrics(log, z, refs[z], V_LO, V_HI)
            g = [r["epost"] for r in ev_rows if int(r["area"]) == int(z)]
            m["epost_max"] = float(max(g)) if g else float("nan")
            rec["metrics"][str(z)] = m
        sts_ids = next((sorted(r.dso_group_v_min_pu) for r in log
                        if r.dso_group_v_min_pu), [])
        rec["sts"] = {g: mc2.sts_metrics(log, g, float(cfg.dso_v_min_pu),
                                         float(cfg.dso_v_max_pu)) for g in sts_ids}
        rec["interface"] = mc2.interface_metrics(log, t_ts, dt)
        rec["events"] = [dict(r, w_total_mvar_min=float(sum(r["w_sts_mvar_min"].values())))
                         for r in ev_rows]
        np.savez_compressed(d["runs"] / f"{seed}_{name}.npz", **_series(log))
    if log:
        import pickle
        with open(d["runs"] / f"{seed}_{name}.log.pkl", "wb") as fh:
            pickle.dump(log, fh)
    (d["runs"] / f"{seed}_{name}.json").write_text(
        json.dumps(rec, indent=2, default=float), encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    return {"seed": seed, "variant": name, "outcome": outcome, "wall_s": wall}


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------
def _cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper--Pearson upper bound on a failure probability."""
    from scipy.stats import beta
    if n == 0:
        return float("nan")
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def _boot_upper_median(d: np.ndarray, conf: float = 0.95, n_boot: int = 10000,
                       seed: int = MASTER_SEED) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    med = np.median(d[idx], axis=1)
    return float(np.quantile(med, conf))


def report(out: Path, n_valid: int) -> None:
    from scipy.stats import wilcoxon
    d = _dirs(out)
    pre = [json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(d["pre"].glob("*.json"))]
    valid = sorted(r["seed"] for r in pre if r["valid"])[:n_valid]
    n_disc = sum(1 for r in pre if not r["valid"]
                 and r["seed"] < (max(valid) if valid else 0))
    rows, outc = [], []
    for p in sorted(d["runs"].glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["seed"] not in valid:
            continue
        if r["outcome"] == "task_crash":
            print(f"  WARNING: task crash, rerun needed: seed {r['seed']} {r['variant']}")
            continue
        outc.append({k: r[k] for k in ("seed", "variant", "outcome", "wall_s",
                                       "class")})
        for z, m in (r.get("metrics") or {}).items():
            rows.append(dict(seed=r["seed"], variant=r["variant"], area=int(z),
                             outcome=r["outcome"], cls=r["class"], **m))
    if not outc:
        print("no variant runs on valid episodes yet")
        return
    met = pd.DataFrame(rows)
    oc = pd.DataFrame(outc)
    met.to_csv(out / "metrics_per_episode.csv", index=False)
    oc.to_csv(out / "outcomes_per_episode.csv", index=False)

    lines = [f"# Part III Monte-Carlo report ({_dt.datetime.now():%Y-%m-%d %H:%M})",
             "", f"valid episodes used: {len(valid)} (target {n_valid}); "
             f"draws discarded by the prescreen before the last used seed: {n_disc}",
             f"classes: {pd.Series([r['class'] for r in pre if r['seed'] in valid]).value_counts().to_dict()}",
             ""]

    # Infeasible: every variant run on that episode failed.
    piv = oc.pivot_table(index="seed", columns="variant", values="outcome",
                         aggfunc="first")
    infeasible = [s for s, row in piv.iterrows()
                  if row.notna().all() and (row != "ok").all()]
    lines += ["## Outcomes", "",
              f"infeasible (every variant run fails): {len(infeasible)} {infeasible}", ""]
    lines.append("| variant | runs | ok | failures (excl. infeasible) | CP 95 % upper | by type |")
    lines.append("|---|---|---|---|---|---|")
    for v, g in oc.groupby("variant"):
        g2 = g[~g["seed"].isin(infeasible)]
        k = int((g2["outcome"] != "ok").sum())
        n = int(len(g2))
        lines.append(f"| {v} | {len(g)} | {int((g['outcome'] == 'ok').sum())} | "
                     f"{k}/{n} | {_cp_upper(k, n):.3f} | "
                     f"{g['outcome'].value_counts().to_dict()} |")
    lines.append("")

    # Distributions (ok runs only).
    okm = met[met["outcome"] == "ok"]
    cols = ["e_vp", "e_v", "worst", "t_out", "avr_trav", "taps", "epost_max"]
    lines += ["## Distributions (ok runs; e in mpu, t_out as share)", ""]
    lines.append("| variant | area | n | " + " | ".join(f"{c} med / p95" for c in cols) + " |")
    lines.append("|---|---|---|" + "---|" * len(cols))
    for (v, a), g in okm.groupby(["variant", "area"]):
        cells = []
        for c in cols:
            x = g[c].astype(float)
            s = 1e3 if c in ("e_vp", "e_v", "worst", "epost_max") else 1.0
            cells.append(f"{s * x.median():.3g} / {s * x.quantile(.95):.3g}")
        lines.append(f"| {v} | {a} | {len(g)} | " + " | ".join(cells) + " |")
    lines.append("")

    # Paired tests on episodes where both members are ok.
    lines += ["## Paired comparisons (candidate - comparator, both ok)", "",
              f"non-inferiority: e_vp, H0 median(d) >= delta = {DELTA_MPU} mpu; "
              "decision by the one-sided 95 % bootstrap upper bound of the median "
              "(< delta) -- per area; the aggregate claim needs all three areas "
              "(intersection-union, no multiplicity correction).",
              "superiority: e_v, worst, t_out, H0 median(d) >= 0; one-sided "
              "Wilcoxon signed-rank, Holm over the 3 metrics x 3 areas.", ""]
    for cand, comp in COMPARISONS:
        if not {cand, comp} <= set(okm["variant"]):
            continue
        lines.append(f"### {cand} vs {comp}")
        lines.append("")
        lines.append("| area | metric | n | median d | upper 95 % | test | p | decision |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sup = []
        for a in sorted(okm["area"].unique()):
            A = okm[(okm["area"] == a) & (okm["variant"] == cand)].set_index("seed")
            B = okm[(okm["area"] == a) & (okm["variant"] == comp)].set_index("seed")
            common = A.index.intersection(B.index)
            for c in ("e_vp", "e_v", "worst", "t_out"):
                dd = (A.loc[common, c].astype(float) - B.loc[common, c].astype(float)).to_numpy()
                if c in ("e_vp", "e_v", "worst"):
                    dd = dd * 1e3                                  # mpu
                if dd.size < 5:
                    continue
                ub = _boot_upper_median(dd)
                if c == "e_vp":
                    try:
                        p = float(wilcoxon(dd - DELTA_MPU, alternative="less").pvalue)
                    except ValueError:
                        p = float("nan")
                    dec = "non-inferior" if ub < DELTA_MPU else "NOT shown"
                    lines.append(f"| {a} | e_vp | {dd.size} | {np.median(dd):+.3g} | "
                                 f"{ub:+.3g} | NI (delta {DELTA_MPU}) | {p:.3g} | {dec} |")
                else:
                    try:
                        p = float(wilcoxon(dd, alternative="less").pvalue)
                    except ValueError:
                        p = float("nan")
                    sup.append([a, c, dd.size, np.median(dd), ub, p])
        # Holm over the superiority family.
        if sup:
            order = np.argsort([s[5] if np.isfinite(s[5]) else 1.0 for s in sup])
            m = len(sup)
            adj = [np.nan] * m
            run = 0.0
            for rank, i in enumerate(order):
                pv = sup[i][5] if np.isfinite(sup[i][5]) else 1.0
                run = max(run, min(1.0, (m - rank) * pv))
                adj[i] = run
            for (a, c, n, med, ub, p), pa in zip(sup, adj):
                dec = "superior" if pa < 0.05 else "not shown"
                lines.append(f"| {a} | {c} | {n} | {med:+.3g} | {ub:+.3g} | "
                             f"sup (Holm) | {pa:.3g} | {dec} |")
        lines.append("")

    # Wall times.
    lines += ["## Wall time per run [min]", ""]
    for v, g in oc.groupby("variant"):
        lines.append(f"- {v}: median {g['wall_s'].median() / 60:.1f}, "
                     f"max {g['wall_s'].max() / 60:.1f}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out / 'report.md'}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def _campaign_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """The declared constants, for the run's README and provenance block.

    Code state (commit, dirty flag, diff) is recorded per invocation by
    ``thesis_runs``; the drawable elements go to ``elements.json``.
    """
    return {"master_seed": MASTER_SEED, "n_valid": args.n,
            "n_valid_o3": getattr(args, "n_o3", N_VALID_O3),
            "event_phase_s": list(PHASE_GRID_S),
            "phase_stream": "default_rng([seed, 1]), one draw per event",
            "noise_seed": "the episode seed (shared by all variants)",
            "configuration": "experiments/ch_11_case2_multisystem/variants.py",
            "horizon_min": args.minutes, "variants": args.variants,
            "slot_min": SLOT_MIN, "slot_prob": SLOT_PROB,
            "revert_min": REVERT_MIN,
            "load_q_mvar": [LOAD_Q_MIN_MVAR, LOAD_Q_MAX_MVAR],
            "delta_mpu": DELTA_MPU, "v_range": [V_LO, V_HI],
            "v_hard": [V_HARD_LO, V_HARD_HI],
            "excluded_days": sorted(str(x) for x in EXCLUDED_DAYS)}


def _campaign_folder(args: argparse.Namespace) -> Optional[Path]:
    """The campaign to continue: ``--out``, else the newest one with this tag."""
    if args.out:
        return Path(args.out)
    if args.new:
        return None
    try:
        return latest_run(__file__, producer="run_mc", tag=args.tag,
                          require_complete=False)
    except FileNotFoundError:
        return None


def _check_gurobi(allow: bool) -> None:
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
        print("  [solver] Gurobi licence OK")
    except Exception as exc:                       # noqa: BLE001
        msg = (f"Gurobi is not usable here ({type(exc).__name__}: {exc}). The "
               "MIQP would fall back to SCIP, so O2 would not be the controller "
               "of the reference runs.")
        if not allow:
            raise SystemExit(msg + " Fix the licence or pass --allow-no-gurobi.")
        print("  [solver] WARNING: " + msg)


def _pool_map(fn, jobs: List[Tuple], n_workers: int, label: str) -> List[Any]:
    """Run ``fn(*job)`` over a process pool; survive a broken pool by
    resubmitting the jobs that were in flight (Gurobi licence hiccups killed
    workers natively in September)."""
    results, todo, restarts = [], list(jobs), 0
    while todo:
        ex = ProcessPoolExecutor(max_workers=max(1, n_workers))
        futs = {ex.submit(fn, *j): j for j in todo}
        todo = []
        try:
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception:                  # noqa: BLE001
                    print(f"  [{label}] task {j[:2]} crashed:", flush=True)
                    traceback.print_exc()
                    continue
                results.append(res)
                print(f"  [{label}] {_dt.datetime.now():%H:%M} {res}", flush=True)
        except BrokenProcessPool:
            restarts += 1
            done = {(r.get("seed"), r.get("variant")) for r in results}
            todo = [j for j in futs.values()
                    if (j[0], j[1] if len(j) > 3 else None) not in done]
            print(f"  [{label}] pool broken; restart {restarts}, "
                  f"{len(todo)} task(s) requeued", flush=True)
            if restarts > 20:
                raise
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    return results


def main() -> None:
    global HORIZON_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--n", type=int, default=N_VALID, help="valid episodes")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--n-o3", type=int, default=N_VALID_O3,
                    help="valid episodes for O3 (the failure bound)")
    ap.add_argument("--minutes", type=int, default=HORIZON_MIN)
    ap.add_argument("--out", default=None,
                    help="explicit campaign folder outside the documented "
                         "results tree (no README is written there)")
    ap.add_argument("--new", action="store_true",
                    help="start a fresh campaign instead of continuing the newest")
    ap.add_argument("--report", action="store_true", help="tables only")
    ap.add_argument("--smoke", action="store_true",
                    help="2 episodes x 10 min, a campaign tagged 'smoke'")
    ap.add_argument("--allow-no-gurobi", action="store_true")
    ap.add_argument("--also-seeds", default="",
                    help="comma-separated valid seeds beyond the first --n on which "
                         "--variants also run (to classify a failure of O3 on the "
                         "episodes only O3 covers)")
    add_run_arguments(ap)
    args = ap.parse_args()

    if args.smoke:
        args.n, args.n_o3, args.minutes, args.tag = 2, 3, 10, "smoke"
    if args.minutes != HORIZON_MIN:
        # The draw depends on the horizon; children read it from the env.
        os.environ["PART3_MC_HORIZON_MIN"] = str(args.minutes)
        HORIZON_MIN = args.minutes
    found = _campaign_folder(args)
    if args.report:
        if found is None:
            raise SystemExit("no campaign to report on; run one first")
        report(found, args.n)
        annotate_run(found, by=__file__, outputs={"report.md": "campaign tables"})
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; available {sorted(VARIANTS)}")
    _check_gurobi(args.allow_no_gurobi)

    if found is not None and not (found / MANIFEST).exists():
        print(f"  [campaign] WARNING: {found} is not a documented run folder; "
              f"no README or provenance block is written", flush=True)
        documented = contextlib.nullcontext(None)
    else:
        documented = start_thesis_run(
            __file__, target=TARGET, feeds=FEEDS, config=resolved("L2", args.minutes),
            paramset=PARAMSET, parameters=_campaign_parameters(args),
            time_series=(f"ensemble of {args.n} episodes x {args.minutes} min, "
                         f"start uniform over 2016 on a 15-min grid without the "
                         f"W1--W6 days; per-episode start and events in "
                         f"prescreen/<seed>.json"),
            note=args.note, tag=args.tag, resume=found)
    with documented as run:
        _simulate(run.root if run is not None else found, args, variants)
        if run is not None:
            for name, meaning in (
                    ("prescreen", "one JSON per drawn seed: start, events, valid"),
                    ("runs", "one JSON per (seed, variant): outcome and metrics"),
                    ("logs", "worker logs"),
                    ("elements.json", "the drawable machines, lines, load buses"),
                    ("report.md", "campaign tables")):
                run.describe_output(name, meaning)


def _simulate(out: Path, args: argparse.Namespace, variants: List[str]) -> None:
    """Prescreen until N valid seeds are known, then run what is missing."""
    d = _dirs(out)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    el = enumerate_elements()
    (out / "elements.json").write_text(
        json.dumps(el, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    print(f"  [campaign] {out}\n  [campaign] N={args.n} horizon={args.minutes} min "
          f"variants={variants} jobs={args.jobs} master_seed={MASTER_SEED}")
    print(f"  [elements] gens={el['gens']} lines={len(el['lines'])} "
          f"(excluded, islanding: {el['lines_excluded_islanding']}) "
          f"load buses={len(el['load_buses'])}", flush=True)

    # Phase 1: prescreen in seed order until the first N valid seeds are known
    # (N_O3 when O3 is part of the call).
    n_need = max(args.n, getattr(args, "n_o3", 0) if "O3" in variants else 0)
    k = 0
    while True:
        done = {int(p.stem): json.loads(p.read_text(encoding="utf-8"))["valid"]
                for p in d["pre"].glob("*.json")}
        seeds = [MASTER_SEED + i for i in range(k)]
        valid = [s for s in seeds if done.get(s)]
        if len(valid) >= n_need and all(s in done for s in seeds):
            break
        need = n_need - len(valid)
        batch = [MASTER_SEED + i for i in range(k, k + max(need + need // 4 + 1,
                                                           args.jobs))]
        k += len(batch)
        todo = [(s, str(out), args.minutes) for s in batch if s not in done]
        if todo:
            print(f"  [prescreen] {len(todo)} draw(s)", flush=True)
            _pool_map(prescreen_task, todo, args.jobs, "prescreen")
    valid = sorted(s for s, ok in done.items() if ok)[:n_need]
    print(f"  [prescreen] {len(valid)} valid of {k} drawn", flush=True)

    # Phase 2: variant runs, longest variant first; skip what exists (resume).
    def _missing(s: int, v: str) -> bool:
        p = d["runs"] / f"{s}_{v}.json"
        if not p.exists():
            return True
        return json.loads(p.read_text(encoding="utf-8"))["outcome"] == "task_crash"

    also = [int(x) for x in args.also_seeds.split(",") if x.strip()]
    if set(also) - set(valid):
        raise SystemExit(f"--also-seeds not valid here: {sorted(set(also) - set(valid))}")
    todo = [(s, v, str(out), args.minutes) for v in variants
            for s in (valid if v == "O3" else valid[:args.n] + also) if _missing(s, v)]
    print(f"  [runs] {len(todo)} run(s) to go", flush=True)
    if todo:
        _pool_map(variant_task, todo, args.jobs, "run")
    try:
        report(out, args.n)
    except Exception:                              # noqa: BLE001 -- tables later
        print("  [report] failed; the runs are complete, rerun with --report", flush=True)
        traceback.print_exc()


# Spawned workers re-import this module: honour a non-default horizon.
if os.environ.get("PART3_MC_HORIZON_MIN"):
    HORIZON_MIN = int(os.environ["PART3_MC_HORIZON_MIN"])

if __name__ == "__main__":
    main()
