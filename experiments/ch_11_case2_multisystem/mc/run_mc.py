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

Changes of 2026-11-07 (author-approved plan
``docs/architecture/2026-11-07_case2_mc_new_base_plan.md``), before any
ensemble run on the commitment base:

* an outage event of a machine the unit commitment keeps off-line is void,
  with its restore (``void_events``; the runner applies the same rule and
  reports it in ``uc_offline_gens`` / ``uc_void_events`` of its first
  record).  The draw is unchanged; the prescreen records the commitment, the
  void events and the class counted on the events that apply (``class``;
  ``class_drawn`` is the declared draw's);
* ``--buffer K`` runs the variants of the call on the next K valid episodes
  after the first N from the start, so the replacements the window rule may
  need run in the first wave (scheduling only: window and counts unchanged);
* a campaign may be split by variant over two hosts (``run_mc_remote.py``)
  and merged (``merge_mc.py``) before the final invocation with all six.

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
``fail_osc``          a transformer tap reverses direction twice or more in
                      the last 30 min (actuator cycling at the horizon end)
A monitored voltage outside [0.90, 1.10] pu at the horizon end is NOT a failure
(amendment of 2026-11-01, author decision); it is flagged in the run's
``diag`` and listed as an observation in ``failures.md``.  Runs of older code
recorded as ``fail_v_end`` count as ``fail_osc`` or ``ok`` from their tap
reversals (``effective_outcome``).  An episode on which every variant run so
far fails is reported as ``infeasible`` and kept out of the failure bounds.

Comparison window (amendment of 2026-11-01, author decision)
-------------------------------------------------------------
The paired comparisons run over the first ``N_VALID`` valid episodes in draw
order on which ALL six variants completed (``ok``): an episode on which any
variant failed is replaced by the next valid one once all six have run on it
(``comparison_window``).  The replaced episodes stay in the campaign: every
variant runs on every valid episode up to the last one used, so the failure
counts and bounds are taken over all of them and reported separately with
their causes (``failures.md``, ``failures_per_run.csv``).  The comparison
metrics are conditional on all six completing.  Stop condition: no episode
beyond the first ``N_VALID_MAX`` valid ones is used; a window still short then
goes back to the author.  Earlier rules, for re-reporting earlier campaigns
(``--rule``): ``2026-10-30`` replaces only an episode on which all six failed;
``2026-10-07`` (``--no-top-up``) takes the first ``N_VALID`` valid episodes.

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

#: Comparison-window rules, named by the date of the author decision; the
#: first is the default (see the module docstring).
WINDOW_RULES = ("2026-11-01", "2026-10-30", "2026-10-07")
WINDOW_RULE = WINDOW_RULES[0]
#: Stop condition: the window never reaches beyond the first N_VALID_MAX
#: valid episodes.
N_VALID_MAX = 150
#: Plant step and TS period of the Case-2 configuration, for runs recorded
#: before ``dt_s`` / ``t_ts_s`` were stored in their JSON.
DT_S_CASE2, T_TS_S_CASE2 = 20.0, 180.0

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


def comparison_window(valid: List[int], outcome: Dict[Tuple[int, str], str],
                      n: int, rule: str = WINDOW_RULE) -> Tuple[List[int], List[int]]:
    """The episodes the paired comparisons run over, and the ones skipped.

    ``valid`` are the valid seeds in draw order, ``outcome`` maps (seed,
    variant) to the outcome of a finished run (no ``task_crash``).  A seed is
    skipped only once all six variants of ``ALL_VARIANTS`` have run on it: with
    rule ``2026-11-01`` if any of them failed, with ``2026-10-30`` if all of
    them failed.  A seed not yet fully run counts as usable, so the window only
    moves on when its runs show otherwise.  The rule depends on the outcomes of
    all variants alike and keeps the draw order.  Rule ``2026-10-07`` is the
    declared rule: the first ``n`` valid seeds, none skipped.
    """
    if rule not in WINDOW_RULES:
        raise ValueError(f"unknown window rule {rule!r}; known {WINDOW_RULES}")
    if rule == "2026-10-07":
        return list(valid[:n]), []
    window: List[int] = []
    skipped: List[int] = []
    for s in valid:
        res = [outcome.get((s, v)) for v in ALL_VARIANTS]
        if all(r is not None for r in res):
            failed = [r != "ok" for r in res]
            if any(failed) if rule == "2026-11-01" else all(failed):
                skipped.append(s)
                continue
        window.append(s)
        if len(window) == n:
            break
    return window, skipped


def effective_outcome(outcome: str, diag: Optional[Dict[str, Any]] = None) -> str:
    """The outcome under the criteria of 2026-11-01.  A run of older code
    recorded as ``fail_v_end`` (voltage outside the hard band at the horizon
    end, no longer a failure) counts as ``fail_osc`` if its tail shows tap
    hunting -- the criterion ``classify`` checked after that one -- else ``ok``."""
    if outcome != "fail_v_end":
        return outcome
    d = diag or {}
    rev = max(int(d.get("tap_reversals_end_ts") or 0),
              int(d.get("tap_reversals_end_sts") or 0))
    return "fail_osc" if rev >= OSC_REVERSALS else "ok"


def _outcomes(runs: Path, rule: str = WINDOW_RULE) -> Dict[Tuple[int, str], str]:
    """(seed, variant) -> outcome of every finished run in ``runs``; the
    recorded one for the earlier rules, ``effective_outcome`` for 2026-11-01."""
    out: Dict[Tuple[int, str], str] = {}
    for p in runs.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["outcome"] != "task_crash":
            o = r["outcome"]
            if rule == "2026-11-01":
                o = effective_outcome(o, r.get("diag"))
            out[(int(r["seed"]), r["variant"])] = o
    return out


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


def void_events(events: List[ContingencyEvent], offline) -> Tuple[
        List[ContingencyEvent], List[ContingencyEvent]]:
    """(effective, void) events of an episode under its unit commitment.

    A drawn outage of a machine the commitment keeps off-line is void, with
    its restore (author 2026-11-07): the trip would change nothing and the
    restore would start a machine the commitment keeps off.  The runner
    applies the same rule (``uc_void_events`` of its first record); the draw
    itself is unchanged, so the episodes stay those of the declared draw.
    """
    off = {int(g) for g in (offline or ())}
    void = [e for e in events if e.element_type == "gen"
            and getattr(e, "element_index", None) is not None
            and int(e.element_index) in off]
    return [e for e in events if e not in void], void


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
    # The commitment of the episode (the prescreen spans the same window as
    # the variant runs, so it is the same) and the events it voids; the
    # class is counted on the events that apply.
    offline = list(getattr(log[0], "uc_offline_gens", ()) or ()) if log else None
    effective, void = void_events(events, offline or ())
    rec = {"seed": seed, "valid": bool(ok), "start_time": start.isoformat(),
           "events": events_to_rows(events),
           "class": episode_class(effective if offline is not None else events),
           "class_drawn": episode_class(events),
           "uc_offline": offline,
           "events_void": events_to_rows(void) if offline is not None else None,
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
    # An observation, not a failure (amendment of 2026-11-01).
    diag["v_end_outside_hard_band"] = not (V_HARD_LO <= vmin and vmax <= V_HARD_HI)
    if n_gap:
        return "fail_pf", diag
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
    if log:
        # The runner's commitment and voided events; they must equal the
        # prescreen's (same window), else the episode is not the screened one.
        off = sorted(int(g) for g in getattr(log[0], "uc_offline_gens", ()) or ())
        diag["uc_offline"] = off
        diag["uc_void_events"] = [list(v) for v in
                                  getattr(log[0], "uc_void_events", ()) or ()]
        if pre.get("uc_offline") is not None and off != sorted(pre["uc_offline"]):
            diag["uc_mismatch_prescreen"] = sorted(pre["uc_offline"])
    rec: Dict[str, Any] = {"seed": seed, "variant": name, "outcome": outcome,
                           "diag": diag, "wall_s": wall,
                           "n_records": len(log or []),
                           "dt_s": float(cfg.dt_s), "t_ts_s": float(cfg.tso_period_s),
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


_RULE_TEXT = {
    "2026-11-01": "rule 2026-11-01: episodes on which any variant failed replaced",
    "2026-10-30": "rule 2026-10-30: episodes on which all six failed replaced",
    "2026-10-07": "rule 2026-10-07: the first valid episodes, none replaced",
}


def applied_events(pre: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The drawn event rows of a prescreen record without the void ones
    (``void_events``); older records have no ``events_void``."""
    void = pre.get("events_void") or []
    return [e for e in pre.get("events") or [] if e not in void]


def _event_text(pre: Dict[str, Any]) -> str:
    parts = []
    void = pre.get("events_void") or []
    for e in pre.get("events") or []:
        what = f"{e['element_type']} {e['action']}"
        if e["element_type"] == "load":
            what += f" bus {e['bus']}"
            if e["action"] == "connect":
                what += f" {e['q_mvar']:.0f} Mvar"
        else:
            what += f" {e['element_index']}"
        if e in void:
            what += " (void: off-line)"
        parts.append(f"{e['time_s'] / 60.0:.1f} min {what}")
    return "; ".join(parts)


def failure_row(rec: Dict[str, Any], pre: Dict[str, Any]) -> Dict[str, Any]:
    """What failed, when and why, for one run that is not ``ok``.

    ``t_fail_min`` is the step that stopped the run (aborted runs only);
    ``after_event_min`` the time since the last disturbance or reversal before
    it; ``at_event`` / ``at_dispatch`` whether it stopped in an event's step or
    on the TS dispatch grid.
    """
    diag = rec.get("diag") or {}
    exc = str(diag.get("exception", ""))
    o = effective_outcome(rec["outcome"], diag)
    dt = float(rec.get("dt_s") or DT_S_CASE2)
    t_ts = float(rec.get("t_ts_s") or T_TS_S_CASE2)
    stopped = o in ("fail_abort", "fail_unreachable")
    t_fail = (int(rec.get("n_records") or 0) + 1) * dt if stopped else float("nan")
    times = [float(e["time_s"]) for e in applied_events(pre)]
    if o == "fail_unreachable":
        cause = "reachability guard"
    elif o == "fail_abort" and "single-slack" in exc:
        cause = "guard power flow (single-slack copy)"
    elif o == "fail_abort" and ("LoadflowNotConverged" in exc or "did not converge" in exc):
        cause = "power flow not converged, run stopped"
    elif o == "fail_abort":
        cause = f"exception {exc.split(':')[0]}" if exc else "short log"
    elif o == "fail_pf":
        cause = (f"power flow not converged in {diag.get('n_pf_fail_steps')} "
                 f"step(s), run continued")
    elif o == "fail_osc":
        cause = "tap hunting at the horizon end"
    else:
        cause = o
    before = [t for t in times if t <= t_fail]
    return {"seed": int(rec["seed"]), "variant": rec["variant"], "outcome": o,
            "cause": cause, "t_fail_min": t_fail / 60.0,
            "after_event_min": ((t_fail - max(before)) / 60.0 if stopped and before
                                else float("nan")),
            "at_event": bool(stopped and any(abs(t - t_fail) < dt / 2 for t in times)),
            "at_dispatch": bool(stopped and abs(t_fail % t_ts) < 1e-6),
            "tap_reversals_end": max(int(diag.get("tap_reversals_end_ts") or 0),
                                     int(diag.get("tap_reversals_end_sts") or 0)),
            "class": rec.get("class"), "start_time": rec.get("start_time"),
            "events": _event_text(pre), "exception": exc[:200]}


def _short(row: Dict[str, Any]) -> str:
    """Compact cell of the episode table: cause and stop time."""
    c = row["cause"]
    tag = ("PF" if c.startswith("power flow not converged, run") else
           "PF gaps" if c.startswith("power flow not converged in") else
           "guard PF" if c.startswith("guard") else
           "reach" if c.startswith("reachability") else
           "osc" if c.startswith("tap hunting") else c)
    t = row["t_fail_min"]
    return f"{tag} @{t:.0f} min" if np.isfinite(t) else tag


def _v_end_flag(rec: Dict[str, Any]) -> bool:
    d = rec.get("diag") or {}
    if "v_end_outside_hard_band" in d:
        return bool(d["v_end_outside_hard_band"])
    if rec["outcome"] == "fail_v_end":
        return True
    lo, hi = d.get("v_end_min"), d.get("v_end_max")
    return lo is not None and hi is not None and not (V_HARD_LO <= lo and hi <= V_HARD_HI)


def failure_report(out: Path, window: List[int], skipped: List[int],
                   rule: str) -> None:
    """``failures.md`` and ``failures_per_run.csv``: every failed run with its
    cause, the failure counts per variant over the episodes used (window and
    replaced), the replaced episodes variant by variant, and the runs that end
    with a voltage outside the hard band (an observation, not a failure)."""
    d = _dirs(out)
    pre = {int(p.stem): json.loads(p.read_text(encoding="utf-8"))
           for p in d["pre"].glob("*.json")}
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(d["runs"].glob("*.json"))]
    recs = [r for r in recs if r["outcome"] != "task_crash"]
    used = set(window) | set(skipped)

    ran = {}
    for r in recs:
        ran.setdefault(int(r["seed"]), set()).add(r["variant"])

    def where(s: int) -> str:
        if s in window:
            return "window" if ran.get(s, set()) >= set(ALL_VARIANTS) else \
                "window, not yet run by all six"
        return "replaced" if s in skipped else "outside the window"

    fails = [dict(failure_row(r, pre.get(int(r["seed"]), {})), set=where(int(r["seed"])))
             for r in recs if effective_outcome(r["outcome"], r.get("diag")) != "ok"]
    cols = ["seed", "set", "variant", "outcome", "cause", "t_fail_min",
            "after_event_min", "at_event", "at_dispatch", "tap_reversals_end",
            "class", "start_time", "events", "exception"]
    fl = pd.DataFrame(fails, columns=cols)
    fl.to_csv(out / "failures_per_run.csv", index=False)

    by = {(int(r["seed"]), r["variant"]): r for r in recs if int(r["seed"]) in used}
    infeasible = sorted(s for s in used
                        if all((s, v) in by for v in ALL_VARIANTS)
                        and all(effective_outcome(by[(s, v)]["outcome"],
                                                  by[(s, v)].get("diag")) != "ok"
                                for v in ALL_VARIANTS))
    lines = [f"# Failures and their causes ({_dt.datetime.now():%Y-%m-%d %H:%M})", "",
             f"{_RULE_TEXT[rule]}.  Comparison window: {len(window)} episodes; "
             f"replaced: {len(skipped)} {skipped}.",
             "Every variant runs on every valid episode up to the last one used, so "
             "the counts below are over the window and the replaced episodes; an "
             "episode on which all six failed (infeasible) is kept out of the bound.  "
             "Runs on episodes outside the window (O3's further episodes, "
             "--also-seeds) are listed, not counted.", "",
             f"infeasible: {len(infeasible)} {infeasible}", "",
             "## Failures per variant", "",
             "| variant | episodes | failed | CP 95 % upper | by cause |",
             "|---|---|---|---|---|"]
    fu = fl[fl["seed"].isin(used) & ~fl["seed"].isin(infeasible)]
    for v in ALL_VARIANTS:
        n = sum(1 for (s, w) in by if w == v and s not in infeasible)
        g = fu[fu["variant"] == v]
        lines.append(f"| {v} | {n} | {len(g)} | {_cp_upper(len(g), n):.3f} | "
                     f"{g['cause'].value_counts().to_dict() if len(g) else {}} |")
    lines += ["", "## Replaced episodes", "",
              "| seed | class | start | events | " + " | ".join(ALL_VARIANTS) + " |",
              "|---|---|---|---|" + "---|" * len(ALL_VARIANTS)]
    cell = {(r["seed"], r["variant"]): _short(r) for r in fails}
    for s in skipped:
        p = pre.get(s, {})
        lines.append(f"| {s} | {p.get('class')} | {p.get('start_time')} | "
                     f"{_event_text(p)} | " +
                     " | ".join(cell.get((s, v), "ok" if (s, v) in by else "--")
                                for v in ALL_VARIANTS) + " |")
    lines += ["", "## Every failed run", "",
              "| seed | set | variant | cause | stop [min] | after event [min] | "
              "at event | at dispatch | exception |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(fails, key=lambda x: (x["seed"], x["variant"])):
        lines.append(f"| {r['seed']} | {r['set']} | {r['variant']} | {r['cause']} | "
                     f"{r['t_fail_min']:.1f} | {r['after_event_min']:.1f} | "
                     f"{r['at_event']} | {r['at_dispatch']} | {r['exception'][:90]} |")
    flagged = [(int(r["seed"]), r["variant"]) for r in recs
               if int(r["seed"]) in used and _v_end_flag(r)]
    lines += ["", f"## Observation, not a failure: a monitored voltage outside "
              f"[{V_HARD_LO:.2f}, {V_HARD_HI:.2f}] pu at the horizon end", "",
              "| variant | runs | seeds |", "|---|---|---|"]
    for v in ALL_VARIANTS:
        ss = sorted(s for s, w in flagged if w == v)
        lines.append(f"| {v} | {len(ss)} | {ss} |")
    (out / "failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out / 'failures.md'}")


def report(out: Path, n_valid: int, rule: str = WINDOW_RULE,
           max_valid: int = N_VALID_MAX) -> None:
    from scipy.stats import wilcoxon
    d = _dirs(out)
    pre = [json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(d["pre"].glob("*.json"))]
    cap = max(max_valid, n_valid)
    window, skipped = comparison_window(
        sorted(r["seed"] for r in pre if r["valid"])[:cap],
        _outcomes(d["runs"], rule), n_valid, rule)
    valid = sorted(window + skipped)
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
        if rule == "2026-11-01":
            r["outcome"] = effective_outcome(r["outcome"], r.get("diag"))
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
    failure_report(out, window, skipped, rule)
    n_ran = oc.groupby("seed")["variant"].nunique().to_dict()
    incomplete = [s for s in window if n_ran.get(s, 0) < len(ALL_VARIANTS)]

    lines = [f"# Part III Monte-Carlo report ({_dt.datetime.now():%Y-%m-%d %H:%M})",
             "", f"comparison window: {len(window)} valid episodes (target {n_valid}), "
             f"{len(skipped)} replaced {skipped} ({_RULE_TEXT[rule]}; "
             f"at most the first {cap} valid episodes); "
             f"draws discarded by the prescreen before the last used seed: {n_disc}",
             f"window episodes not yet run by all six variants: {len(incomplete)} {incomplete}",
             f"classes: {pd.Series([r['class'] for r in pre if r['seed'] in valid]).value_counts().to_dict()}",
             "", "Failures and their causes: failures.md.", ""]

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

    # Distributions (ok runs on the comparison window only).
    okm = met[(met["outcome"] == "ok") & met["seed"].isin(window)]
    cols = ["e_vp", "e_v", "worst", "t_out", "avr_trav", "taps", "epost_max"]
    lines += ["## Distributions (ok runs in the window; e in mpu, t_out as share)", ""]
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
            "window_rule": getattr(args, "rule", WINDOW_RULE),
            "n_valid_max": getattr(args, "max_valid", N_VALID_MAX),
            "v_end_is_failure": False,
            "buffer": getattr(args, "buffer", 0),
            "uc_void_rule": ("an outage event of a machine the unit commitment "
                             "keeps off-line is void with its restore; class "
                             "from the events that apply (2026-11-07)"),
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
    ap.add_argument("--rule", choices=WINDOW_RULES, default=WINDOW_RULE,
                    help="comparison-window rule (module docstring); the earlier "
                         "ones re-report earlier campaigns")
    ap.add_argument("--no-top-up", action="store_true",
                    help="same as --rule 2026-10-07: compare on the first --n valid "
                         "episodes, none replaced")
    ap.add_argument("--max-valid", type=int, default=N_VALID_MAX,
                    help="stop condition: the window uses at most this many valid "
                         "episodes")
    ap.add_argument("--buffer", type=int, default=0,
                    help="also run --variants on the next BUFFER valid episodes "
                         "after the first --n from the start, so that the "
                         "episodes the window rule may need as replacements run "
                         "in the first wave (scheduling only: the window and the "
                         "counts are unchanged; runs outside the window are "
                         "listed, not counted)")
    add_run_arguments(ap)
    args = ap.parse_args()
    if args.no_top_up:
        args.rule = "2026-10-07"

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
        report(found, args.n, args.rule, args.max_valid)
        annotate_run(found, by=__file__, outputs={"report.md": "campaign tables",
                                                  "failures.md": "failures and causes"})
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
                    ("report.md", "campaign tables"),
                    ("failures.md", "failures and their causes, replaced episodes")):
                run.describe_output(name, meaning)


def _prescreen_until(d: Dict[str, Path], out: Path, args: argparse.Namespace,
                     n_need: int) -> List[int]:
    """Prescreen in seed order until the first ``n_need`` valid seeds are
    known; return them in draw order."""
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
    return valid


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
    n_o3 = getattr(args, "n_o3", 0) if "O3" in variants else 0
    n_buf = max(0, int(getattr(args, "buffer", 0) or 0))
    valid = _prescreen_until(d, out, args, max(args.n + n_buf, n_o3))

    # Phase 2: variant runs, longest variant first; skip what exists (resume).
    # The comparison window moves past an episode once all six variants have
    # run on it and the rule replaces it; the loop then runs the next one,
    # prescreening further seeds if the known valid ones are used up -- but
    # never beyond the first ``--max-valid`` valid seeds (stop condition).
    def _missing(s: int, v: str) -> bool:
        p = d["runs"] / f"{s}_{v}.json"
        if not p.exists():
            return True
        return json.loads(p.read_text(encoding="utf-8"))["outcome"] == "task_crash"

    rule, cap = args.rule, max(args.max_valid, args.n)
    while True:
        window, skipped = comparison_window(valid[:cap], _outcomes(d["runs"], rule),
                                            args.n, rule)
        if len(window) < args.n and len(valid) < cap:
            valid = _prescreen_until(d, out, args,
                                     min(len(valid) + args.n - len(window), cap))
            continue
        also = [int(x) for x in args.also_seeds.split(",") if x.strip()]
        if set(also) - set(valid):
            raise SystemExit(f"--also-seeds not valid here: {sorted(set(also) - set(valid))}")
        o3_seeds = sorted(set(valid[:n_o3]) | set(window))
        # The buffer: the next valid episodes after the declared first N,
        # fixed in draw order (so it never moves with the window).
        buf = valid[args.n:args.n + n_buf]
        paired = sorted(set(window) | set(also) | set(buf))
        todo = [(s, v, str(out), args.minutes) for v in variants
                for s in (sorted(set(o3_seeds) | set(paired)) if v == "O3" else paired)
                if _missing(s, v)]
        span = f"{window[0]}..{window[-1]}" if window else "empty"
        print(f"  [runs] {len(todo)} run(s) to go; comparison window {span} "
              f"({len(window)} episodes, {len(skipped)} replaced, rule {rule})",
              flush=True)
        if not todo:
            if len(window) < args.n:
                print(f"  [runs] STOP: {len(window)} of {args.n} complete episodes "
                      f"within the first {cap} valid ones (--max-valid); the window "
                      f"stays short -- author decision", flush=True)
            break
        _pool_map(variant_task, todo, args.jobs, "run")
    try:
        report(out, args.n, rule, args.max_valid)
    except Exception:                              # noqa: BLE001 -- tables later
        print("  [report] failed; the runs are complete, rerun with --report", flush=True)
        traceback.print_exc()


# Spawned workers re-import this module: honour a non-default horizon.
if os.environ.get("PART3_MC_HORIZON_MIN"):
    HORIZON_MIN = int(os.environ["PART3_MC_HORIZON_MIN"])

if __name__ == "__main__":
    main()
