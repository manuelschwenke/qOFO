#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch_10_6_ladder_mc.py
====================

Monte-Carlo robustness campaign for the dissertation's seven-rung control
ladder (thesis \\cref{ch:case3}).

Where ``ch_10_1_variant_ladder`` runs the ladder once on the authoritative
deterministic 360-min trajectory, this driver repeats that experiment over many
randomised scenarios, so the ladder's central claim -- that every step keeps
its sign -- becomes a measured probability with a spread instead of an
observation about a single run.

Relationship to 006_CIGRE_MONTECARLO
------------------------------------
The scenario machinery (paired runs, slot-based random schedule, drop-and-
replace, base-case prescreen, seeded reproducibility) is modelled directly on
``experiments/CIGRE_2026/006_CIGRE_MONTECARLO.py``.  That file is FROZEN for a
paper and is neither imported nor modified: it derives from
``make_cigre_config``, which differs from the authoritative
``run_multi_system_ofo.make_config`` in 15 fields, three of them structural.
Importing it would drag the paper's parameter set into a thesis experiment --
exactly the failure ``ch_10_1_variant_ladder``'s docstring exists to prevent.
The reusable *logic* is therefore restated here against
``make_thesis_config``; the reusable *weights* are, as always, inherited and
never restated.

What differs from 006, and why
------------------------------
* **Seven variants, not five.**  L1..M from ``ch_10_1_variant_ladder.VARIANTS``.
  A scenario is accepted only if ALL SEVEN converge, so the discard rate
  compounds faster than it does over five.  Measured 2026-09-05, roughly
  two thirds of draws are rejected, and an ungated diagnostic showed the
  rejected multi-contingency draws are infeasible for every rung rather
  than merely hard for the weak ones (see ``NON_GATING``).
* **360-min horizon**, matching the deterministic ladder.
* **Event-relative peak metrics.**  ``ch_10_4_pi_screen.evaluate`` takes peaks
  over windows hard-coded to the deterministic schedule ([30,60] and
  [210,240]).  Under a random schedule those minutes are meaningless -- the
  metric would silently measure a quiet stretch of a run whose contingency
  fired at minute 150.  Here every drawn event opens its own window and peaks
  pool by event TYPE.  Horizon RMS needs no such change; it is
  schedule-agnostic.
* **Actuator-effort metrics** (``tap_ops``, ``avr_travel``) join the set: they
  are ladder-relevant, and discrete actuator effort is where the 2026-09-04
  guard finding surfaced.
* **No full logs kept.**  700 pickles would be ~4.5 GB on a network home
  folder; a compact per-scenario ``npz`` plus a metrics row is stored instead.

Gauge
-----
``g_w_gen`` is pinned to the ``make_config`` default (1e9) and NOT swept.  The
1e8 tier's stored S1 is still the pre-2026-09-05 classical run, so a gauge
sweep would compare unlike references.

Run::

    python -m experiments.ch_10_case_study.ch_10_6_ladder_mc --runs 100 --jobs 18
    python -m experiments.ch_10_case_study.ch_10_6_ladder_mc --resume --jobs 18
    python -m experiments.ch_10_case_study.ch_10_6_ladder_mc --replot

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

import os

# Headless + UTF-8 must precede any matplotlib / numpy import.  A shunt-commit
# log line carries non-cp1252 glyphs; without this a redirected stdout kills the
# run mid-simulation with UnicodeEncodeError.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# One BLAS thread per worker: with scenario-level process parallelism, P workers
# times many BLAS threads oversubscribes the cores and throughput collapses.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib as mpl
mpl.use("Agg")

import argparse
import contextlib
import importlib.util
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from configs.config import MultiTSOConfig                      # noqa: E402
from experiments.helpers.records import ContingencyEvent       # noqa: E402
from experiments.paths import RESULTS_ROOT                     # noqa: E402
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
OUT_DIR = Path(RESULTS_ROOT) / "ch10_ladder_mc"
#: Worker processes are SPAWNED, so module globals set in main() do not reach
#: them.  The run tag and the gating switch therefore travel by environment,
#: which the children inherit at spawn time.
_TAG = os.environ.get("CH10_MC_TAG", "")
if _TAG:
    # A tagged campaign gets its OWN results root.  The 2026-09-05 campaign ran
    # unscaled with O3 uncoordinated; mixing its rows with a DSO_3 x2 run would
    # pool two different networks into one distribution.
    OUT_DIR = Path(RESULTS_ROOT) / f"ch10_ladder_mc_{_TAG}"
RUNS_DIR = OUT_DIR / "runs"
SCRATCH_DIR = OUT_DIR / "_scratch"

# ---------------------------------------------------------------------------
#  Scenario parameters (mirrored from 006; see module docstring)
# ---------------------------------------------------------------------------
BASE_SEED_DEFAULT = 20260905
N_RUNS_DEFAULT = 100
HORIZON_MIN = 360

SLOT_INTERVAL_MIN = 30        # candidate event slots at 30, 60, ...
SLOT_PROB = 0.25              # per-slot firing probability (~2-3 events/run)
GEN_RESTORE_DELAY_MIN = 180   # a tripped machine reconnects after this long
LOAD_P_MW = 150.0
LOAD_Q_MIN_MVAR = 100.0
LOAD_Q_MAX_MVAR = 400.0
MAX_GENS_OUT = 1
MAX_LINES_OUT = 1
MAX_LOADS_CONNECTED = 1

PROFILE_START = datetime(2016, 1, 1, 0, 0)
PROFILE_END_USABLE = datetime(2016, 12, 31, 18, 0)

PRESCREEN_DT_S = 300.0        # coarse probe step for the base-case feasibility

#: Rungs whose divergence is RECORDED but does not reject the scenario.
#: EMPTY -- every rung gates, so a scenario is rejected at the first failure.
#:
#: This was briefly relaxed (L1, then L1+L2) on the hypothesis that gating
#: selected away the stressed multi-contingency scenarios where coordination
#: matters.  An ungated diagnostic on the three rejected seeds (2026-09-05,
#: results/ch10_ladder_mc/runs_diag/) refuted it outright: ALL SEVEN rungs
#: diverge on all three, 3/4/5 events each.  Those scenarios have no feasible
#: operating point -- no controller holds them -- so rejecting them is correct
#: and gating on the WEAKEST rung first is also the cheapest way to find out
#: (~37 min to reject at L1 versus ~171 min to reach S1).
#:
#: Do not relax this again without re-running that diagnostic.  The failing
#: rung is recorded in every discarded scenario's JSON, so controller-specific
#: failures remain visible -- see seed 20260925, where L1, L2 and S1 all
#: converged and O1 diverged.  That one is NOT infeasibility and is worth a
#: separate look.
NON_GATING: tuple = ()
if os.environ.get("CH10_MC_NO_GATING"):
    # Diagnostic mode: nothing rejects a scenario, every rung is run to
    # completion and its outcome recorded.  Answers the question no gated run
    # can -- whether the upper rungs survive a scenario the lower ones fail.
    NON_GATING = ("L1", "L2", "S1", "O1", "O2", "O3", "M")

#: Minutes after an event over which its peak is taken.
PEAK_WINDOW_MIN = 30
PILOT_BUSES = {1: 25, 2: 7, 3: 20}
V_SET = 1.03


# ---------------------------------------------------------------------------
#  Ladder module (loaded by path, as ch_10_4_pi_screen does)
# ---------------------------------------------------------------------------
def _load_ladder():
    spec = importlib.util.spec_from_file_location(
        "ch10_lad_mc", _HERE / "ch_10_1_variant_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HORIZON_MIN = HORIZON_MIN
    return mod


_LAD = None


def lad():
    """Per-process ladder module cache (workers are spawned on Windows)."""
    global _LAD
    if _LAD is None:
        _LAD = _load_ladder()
    return _LAD


# ---------------------------------------------------------------------------
#  Element enumeration  (logic restated from 006.enumerate_elements)
# ---------------------------------------------------------------------------
def enumerate_elements(cfg: MultiTSOConfig) -> Dict[str, List[int]]:
    """Admissible contingency elements for the random schedule.

    Rebuilds the combined net exactly as ``run_multi_tso_dso`` does so element
    indices line up with what every variant run sees.  The TN elements used
    here (machine gens, TN lines, TN load buses) are all created by
    ``build_ieee39_net`` before ``add_hv_networks``, so their indices do not
    depend on the per-variant shunt flag.
    """
    from network.ieee39 import build_ieee39_net, add_hv_networks
    from network.zone_partition import fixed_zone_partition_ieee39

    net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                                 verbose=False)
    fixed_zone_partition_ieee39(net, verbose=False)
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
        verbose=False,
    )

    gen_trafo_map = {
        int(g): int(t)
        for t, g in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map)
        if g is not None and int(g) >= 0
    }
    slack_gens = set()
    if "slack" in net.gen.columns:
        slack_gens = {int(g) for g in net.gen.index if bool(net.gen.at[g, "slack"])}
    gens = sorted(g for g in gen_trafo_map
                  if g not in slack_gens and bool(net.gen.at[g, "in_service"]))

    hv_lines = {int(li) for hv in meta.hv_networks for li in hv.line_indices}
    lines = sorted(int(li) for li in net.line.index
                   if int(li) not in hv_lines and bool(net.line.at[li, "in_service"]))

    pv_slack = ({int(net.gen.at[g, "bus"]) for g in net.gen.index}
                | {int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index})
    hv_buses = {int(b) for hv in meta.hv_networks for b in hv.bus_indices}
    tn_pq = {int(b) for b in net.bus.index if b not in pv_slack and b not in hv_buses}
    load_buses = sorted({int(net.load.at[li, "bus"]) for li in net.load.index
                         if int(net.load.at[li, "bus"]) in tn_pq})
    if not load_buses:
        load_buses = sorted(tn_pq)
    return {"gens": gens, "lines": lines, "load_buses": load_buses}


#: Store every Nth step's voltages in the per-scenario npz.  Full logs are not
#: kept (700 pickles is ~4.5 GB), but storing NO voltages made a nodal-voltage
#: distribution unrecoverable from the 2026-09-08 campaign -- the boxplot had
#: to fall back to the single deterministic run.  Subsampled float32 costs
#: ~50 kB per rung per scenario, which is affordable where the full log is not.
V_SUBSAMPLE = 3

_ELEMENTS: Optional[Dict[str, List[int]]] = None
_HV_BUSES: Optional[set] = None


def get_hv_buses() -> set:
    """Bus indices belonging to the HV sub-networks (per-process cache)."""
    global _HV_BUSES
    if _HV_BUSES is None:
        from network.ieee39 import build_ieee39_net, add_hv_networks
        from network.zone_partition import fixed_zone_partition_ieee39
        cfg = lad().make_thesis_config()
        net, meta = build_ieee39_net(ext_grid_vm_pu=1.03, scenario=cfg.scenario,
                                     verbose=False)
        fixed_zone_partition_ieee39(net, verbose=False)
        meta = add_hv_networks(
            net, meta,
            install_tso_tertiary_shunts=cfg.install_tso_tertiary_shunts,
            verbose=False)
        _HV_BUSES = {int(b) for hv in meta.hv_networks for b in hv.bus_indices}
    return _HV_BUSES


def voltage_samples(log) -> Tuple[np.ndarray, np.ndarray]:
    """(TS nodal voltages, DS group envelope) as flat float32 samples.

    TS comes from ``bus_vm_pu`` minus the HV buses.  DS uses the per-group
    min/mean/max rather than nodal values because O3 (``dso_mode='ofo'``)
    records only the 30 TS buses in ``bus_vm_pu`` while every other rung
    records 70 -- so nodal DS voltage is unavailable for exactly the rung the
    DSO layer is being judged on.  The envelope is recorded by all rungs.
    """
    hv = get_hv_buses()
    ts: List[float] = []
    ds: List[float] = []
    for i, r in enumerate(log):
        if i % V_SUBSAMPLE:
            continue
        for b, val in (getattr(r, "bus_vm_pu", None) or {}).items():
            if int(b) not in hv:
                ts.append(float(val))
        for f in ("dso_group_v_min_pu", "dso_group_v_mean_pu", "dso_group_v_max_pu"):
            for val in (getattr(r, f, None) or {}).values():
                ds.append(float(val))
    return (np.asarray(ts, dtype=np.float32), np.asarray(ds, dtype=np.float32))


def get_elements() -> Dict[str, List[int]]:
    global _ELEMENTS
    if _ELEMENTS is None:
        _ELEMENTS = enumerate_elements(lad().make_thesis_config())
    return _ELEMENTS


# ---------------------------------------------------------------------------
#  Scenario draw
# ---------------------------------------------------------------------------
def random_start_time(rng: np.random.Generator) -> datetime:
    """Random profile-year start, quantised to 15 min, leaving the horizon tail."""
    total_min = int((PROFILE_END_USABLE - PROFILE_START).total_seconds() // 60)
    total_min -= HORIZON_MIN
    off = int(rng.integers(0, max(total_min, 1)))
    off -= off % 15
    return PROFILE_START + timedelta(minutes=off)


def draw_schedule(rng: np.random.Generator, elements: Dict[str, List[int]],
                  n_total_min: int) -> List[ContingencyEvent]:
    """Slot-based random contingency schedule, lightly constrained for PF
    feasibility: at most one line and one generator out at a time, the slack
    machine never tripped, load connections capped."""
    gens, lines, load_buses = (elements["gens"], elements["lines"],
                               elements["load_buses"])
    events: List[ContingencyEvent] = []
    gen_out_intervals: List[Tuple[int, int]] = []
    used_lines: set = set()
    used_load_buses: set = set()
    loads_connected = 0

    def gen_window_free(m: int) -> bool:
        new_s, new_e = m, min(m + GEN_RESTORE_DELAY_MIN, n_total_min)
        for (s, e) in gen_out_intervals:
            if not (new_e <= s or e <= new_s):
                return False
        return True

    for m in range(SLOT_INTERVAL_MIN, n_total_min, SLOT_INTERVAL_MIN):
        if rng.random() >= SLOT_PROB:
            continue
        allowed: List[str] = []
        if gens and gen_window_free(m):
            allowed.append("gen")
        if (len(used_lines) < MAX_LINES_OUT) and (set(lines) - used_lines):
            allowed.append("line")
        if (loads_connected < MAX_LOADS_CONNECTED) and (set(load_buses) - used_load_buses):
            allowed.append("load")
        if not allowed:
            continue
        kind = allowed[int(rng.integers(len(allowed)))]

        if kind == "gen":
            g = int(gens[int(rng.integers(len(gens)))])
            events.append(ContingencyEvent(minute=m, element_type="gen",
                                           element_index=g, action="trip"))
            end = m + GEN_RESTORE_DELAY_MIN
            if end <= n_total_min:
                events.append(ContingencyEvent(minute=end, element_type="gen",
                                               element_index=g, action="restore"))
            gen_out_intervals.append((m, min(end, n_total_min)))
        elif kind == "line":
            avail = sorted(set(lines) - used_lines)
            li = int(avail[int(rng.integers(len(avail)))])
            events.append(ContingencyEvent(minute=m, element_type="line",
                                           element_index=li, action="trip"))
            used_lines.add(li)
        else:
            avail = sorted(set(load_buses) - used_load_buses)
            b = int(avail[int(rng.integers(len(avail)))])
            q = round(float(rng.uniform(LOAD_Q_MIN_MVAR, LOAD_Q_MAX_MVAR)), 1)
            events.append(ContingencyEvent(minute=m, element_type="load",
                                           element_index=-1, bus=b,
                                           p_mw=LOAD_P_MW, q_mvar=q,
                                           action="connect"))
            used_load_buses.add(b)
            loads_connected += 1

    events.sort(key=lambda e: (e.minute, e.element_type, e.action))
    return events


def schedule_to_rows(seed: int, start_time: datetime,
                     schedule: List[ContingencyEvent]) -> List[Dict[str, Any]]:
    return [{"seed": seed, "start_time": start_time.isoformat(),
             "minute": int(e.minute), "element_type": str(e.element_type),
             "element_index": int(getattr(e, "element_index", -1)),
             "bus": int(getattr(e, "bus", -1) or -1),
             "p_mw": float(getattr(e, "p_mw", 0.0) or 0.0),
             "q_mvar": float(getattr(e, "q_mvar", 0.0) or 0.0),
             "action": str(e.action)} for e in schedule]


# ---------------------------------------------------------------------------
#  Metrics -- event-relative, schedule-agnostic
# ---------------------------------------------------------------------------
def _rms(v) -> float:
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def agg_error_series(log) -> Tuple[np.ndarray, np.ndarray]:
    """(t_min, zone-aggregated voltage RMS error) -- the peak-tracking signal."""
    t = np.array([r.time_s / 60.0 for r in log], float)
    agg = np.array([
        _rms([r.zone_v_rms_err_pu.get(z, np.nan) for z in (1, 2, 3)])
        if getattr(r, "zone_v_rms_err_pu", None) else np.nan
        for r in log], float)
    return t, agg


#: DS voltage band used for the violation fraction [pu].  The HV sub-networks
#: run on a 0.9-1.1 band, not the 0.95-1.05 one: DSO_4's known internal spread
#: of 0.147 pu is recorded as 73 % of the band, which fixes the band at 0.2 pu
#: wide.  A 0.95-1.05 check would score the reference-tracking target of
#: 1.03 pu as near-violating by construction.
DS_V_HI, DS_V_LO = 1.10, 0.90


def ds_metrics(log) -> Dict[str, float]:
    """Distribution-side quality metrics.

    These are the outputs the DSO layer actually controls.  ``e_v`` scores
    only TS-zone nodal voltage, and zone 1 carries no DSO at all, so an
    aggregate over the three TS zones cannot show what O2->O3 buys.

    * ``ds_v_*``   -- voltage quality in the HV sub-networks (the quality
      metric proper).
    * ``ds_q_*``   -- DIAGNOSTIC ONLY.  EHV-HV interface Q tracking error,
      actual minus setpoint, per machine transformer.

    ``ds_q_*`` must NOT be used to compare rungs.  Interface Q is the actuator
    path to voltage, not an objective: O2 runs local q(v) droop and has no
    interface-Q tracking objective at all, so the setpoint recorded under O2 is
    one nothing in the DS ever acts on.  "O3 tracks Q 30x better than O2" is
    true and vacuous -- O3 wins a contest only O3 entered.  Its legitimate use
    is as a MECHANISM CHECK: if it does not fall at O3, the vertical cascade is
    not taking effect.  Performance claims rest on ``e_v`` and ``ds_v_*``.

    The same caution applies to ``e_p`` (pilot-node RMS) elsewhere in this
    module: the pilot buses are S1's own control targets, selected for S1 by
    008_PILOT_BUS_SELECT, and are arbitrary buses for every OFO rung.  It is
    biased toward S1 and is likewise diagnostic.  ``tap_ops`` and
    ``avr_travel`` are actuator-effort diagnostics, not objectives.
    """
    out: Dict[str, float] = {}
    v_dev, v_worst, viol, n_samp = [], 0.0, 0, 0
    per_group: Dict[str, List[float]] = {}
    q_err: List[float] = []
    per_group_q: Dict[str, List[float]] = {}

    for r in log:
        vmean = getattr(r, "dso_group_v_mean_pu", None) or {}
        vmax = getattr(r, "dso_group_v_max_pu", None) or {}
        vmin = getattr(r, "dso_group_v_min_pu", None) or {}
        for g, vm in vmean.items():
            d = float(vm) - V_SET
            v_dev.append(d)
            per_group.setdefault(g, []).append(d)
            hi, lo = float(vmax.get(g, vm)), float(vmin.get(g, vm))
            v_worst = max(v_worst, abs(hi - V_SET), abs(lo - V_SET))
            n_samp += 1
            if hi > DS_V_HI or lo < DS_V_LO:
                viol += 1

        qset = getattr(r, "dso_trafo_q_set_mvar", None) or {}
        qact = getattr(r, "dso_trafo_q_actual_mvar", None) or {}
        grp = getattr(r, "dso_trafo_group", None) or {}
        for k, qs in qset.items():
            if k not in qact:
                continue
            e = float(qact[k]) - float(qs)
            q_err.append(e)
            per_group_q.setdefault(str(grp.get(k, "?")), []).append(e)

    out["ds_v_rms"] = _rms(v_dev)
    out["ds_v_worst"] = float(v_worst) if n_samp else float("nan")
    out["ds_v_viol_frac"] = (viol / n_samp) if n_samp else float("nan")
    out["ds_q_rmse_mvar"] = _rms(q_err)
    out["ds_q_max_mvar"] = float(np.max(np.abs(q_err))) if q_err else float("nan")
    for g in sorted(per_group):
        out[f"ds_v_rms_{g}"] = _rms(per_group[g])
    for g in sorted(per_group_q):
        if g != "?":
            out[f"ds_q_rmse_{g}"] = _rms(per_group_q[g])
    return out


def evaluate_mc(log, schedule: List[ContingencyEvent]) -> Dict[str, Any]:
    """Horizon RMS (schedule-agnostic) plus EVENT-RELATIVE peaks.

    The deterministic ``evaluate()`` cannot be reused: its PEAK_WINDOWS are
    hard-coded to the fixed schedule.  Here each drawn event opens its own
    ``PEAK_WINDOW_MIN`` window and peaks are keyed by event type, so they
    remain comparable across scenarios whose events fire at different minutes.
    """
    L = lad()
    ev = [_rms([r.zone_v_rms_err_pu.get(z, np.nan) for r in log
                if getattr(r, "zone_v_rms_err_pu", None)]) for z in (1, 2, 3)]
    ep = [_rms([float(r.bus_vm_pu[PILOT_BUSES[z]]) - V_SET for r in log
                if getattr(r, "bus_vm_pu", None) and PILOT_BUSES[z] in r.bus_vm_pu])
          for z in (1, 2, 3)]

    t, agg = agg_error_series(log)
    horizon = float(np.nanmax(t)) if t.size else 0.0

    peaks: Dict[str, List[float]] = {}
    for e in schedule:
        if str(e.action) not in ("trip", "connect"):
            continue                      # restores are not disturbance onsets
        a = float(e.minute)
        if a > horizon:
            continue
        b = min(a + PEAK_WINDOW_MIN, horizon)
        m = (t >= a) & (t <= b) & np.isfinite(agg)
        if m.any():
            peaks.setdefault(f"{e.element_type}_{e.action}", []).append(float(agg[m].max()))

    out: Dict[str, Any] = {
        "e_v": _rms(ev), "e_p": _rms(ep),
        "e_v_z1": ev[0], "e_v_z2": ev[1], "e_v_z3": ev[2],
        "tap_ops": int(L.tap_ops(log)),
        "avr_travel_z1": float(L.avr_travel(log, 1)),
        "avr_travel_z2": float(L.avr_travel(log, 2)),
        "avr_travel_z3": float(L.avr_travel(log, 3)),
        "n_records": len(log),
    }
    out.update(ds_metrics(log))
    for kind in ("gen_trip", "line_trip", "load_connect"):
        vals = peaks.get(kind, [])
        out[f"peak_{kind}"] = float(np.max(vals)) if vals else float("nan")
        out[f"n_{kind}"] = len(vals)
    return out


# ---------------------------------------------------------------------------
#  Single variant run
# ---------------------------------------------------------------------------
def run_variant(name: str, start_time: datetime,
                schedule: List[ContingencyEvent], tag: str
                ) -> Tuple[Optional[list], bool, Optional[str]]:
    """Run one ladder rung on one scenario.  Returns (log, converged, reason)."""
    L = lad()
    cfg = L.make_thesis_config()
    for k, v in L.VARIANTS[name].items():
        setattr(cfg, k, v)
    cfg.start_time = start_time
    cfg.contingencies = list(schedule)
    cfg.verbose = 0
    d = SCRATCH_DIR / f"{tag}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    cfg.result_dir = str(d)
    n_steps = int(round(cfg.n_total_s / cfg.dt_s))
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:                       # noqa: BLE001
        return None, False, type(exc).__name__
    if len(log) != n_steps:
        return log, False, f"short_log_{len(log)}of{n_steps}"
    return log, True, None


def prescreen_base_case(start_time: datetime) -> Tuple[bool, Optional[str]]:
    """Cheap base-case probe: weakest rung (L1, cos(phi)=1), no contingencies,
    coarse step.  A failure means the base operating point collapses at this
    start_time, so the scenario is resampled rather than burning SEVEN full
    runs -- the dominant rejection cause in 006, and costlier here."""
    L = lad()
    cfg = L.make_thesis_config()
    for k, v in L.VARIANTS["L1"].items():
        setattr(cfg, k, v)
    cfg.start_time = start_time
    cfg.contingencies = []
    cfg.dt_s = PRESCREEN_DT_S
    cfg.verbose = 0
    d = SCRATCH_DIR / f"pre_pid{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    cfg.result_dir = str(d)
    n_steps = int(round(cfg.n_total_s / cfg.dt_s))
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:                       # noqa: BLE001
        return False, type(exc).__name__
    return (len(log) == n_steps), None


# ---------------------------------------------------------------------------
#  One scenario = all seven rungs on one draw (drop-and-replace unit)
# ---------------------------------------------------------------------------
def run_one_scenario(seed: int) -> Dict[str, Any]:
    """Module-level so it is picklable for ProcessPoolExecutor."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    L = lad()
    t_start = datetime.now()
    rng = np.random.default_rng(seed)
    start_time = random_start_time(rng)
    # The slot draw yields an EMPTY schedule for roughly a fifth of seeds, and a
    # scenario with no disturbance costs seven full runs (~21 h) while
    # contributing nothing to any peak metric.  Redraw from the same generator
    # until at least one event lands: deterministic given the seed, at the cost
    # of conditioning the schedule distribution on >= 1 event.  State that
    # conditioning wherever the event statistics are reported.
    n_draws = 0
    while True:
        schedule = draw_schedule(rng, get_elements(), HORIZON_MIN)
        n_draws += 1
        if schedule or n_draws >= 20:
            break

    rec: Dict[str, Any] = {
        "seed": seed,
        "start_time": start_time.isoformat(),
        "n_events": len(schedule),
        "n_schedule_draws": n_draws,
        "schedule": schedule_to_rows(seed, start_time, schedule),
        "accepted": False,
        "failing_variant": None,
        "metrics": {},
        "wall_s": {},
    }

    ok, why = prescreen_base_case(start_time)
    if not ok:
        rec["failing_variant"] = "base_infeasible"
        rec["reason"] = why
        (RUNS_DIR / f"scenario_{seed}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
        return {"seed": seed, "accepted": False,
                "failing_variant": "base_infeasible", "n_events": len(schedule)}

    series: Dict[str, np.ndarray] = {}
    converged: Dict[str, bool] = {}
    for name in L.LADDER:
        v0 = datetime.now()
        log, conv, reason = run_variant(name, start_time, schedule, f"s{seed}")
        rec["wall_s"][name] = (datetime.now() - v0).total_seconds()
        converged[name] = bool(conv)
        if not conv:
            rec.setdefault("failures", {})[name] = reason
            if name in NON_GATING:
                # L1's divergence is an OUTCOME, not a rejected sample.  Gating
                # on it would discard precisely the multi-contingency scenarios
                # where coordination matters, leaving a sample conditioned on
                # the weakest rung surviving -- which flatters L1 and
                # understates every step above it.  Carry on and record it.
                continue
            rec["failing_variant"] = name
            rec["reason"] = reason
            rec["converged"] = converged
            (RUNS_DIR / f"scenario_{seed}.json").write_text(
                json.dumps(rec, indent=2), encoding="utf-8")
            return {"seed": seed, "accepted": False, "failing_variant": name,
                    "n_events": len(schedule)}
        rec["metrics"][name] = evaluate_mc(log, schedule)
        t, agg = agg_error_series(log)
        series["t"] = t
        series[name] = agg
        v_ts, v_ds = voltage_samples(log)
        series[f"v_ts_{name}"] = v_ts
        series[f"v_ds_{name}"] = v_ds
        del log                                     # keep worker RSS bounded

    rec["converged"] = converged
    rec["accepted"] = True
    rec["wall_s"]["total"] = (datetime.now() - t_start).total_seconds()
    (RUNS_DIR / f"scenario_{seed}.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    np.savez_compressed(RUNS_DIR / f"ts_{seed}.npz", **series)
    return {"seed": seed, "accepted": True, "failing_variant": None,
            "n_events": len(schedule)}


# ---------------------------------------------------------------------------
#  Aggregation
# ---------------------------------------------------------------------------
METRIC_COLS = ["e_v", "e_p", "e_v_z1", "e_v_z2", "e_v_z3",
               "peak_gen_trip", "peak_line_trip", "peak_load_connect",
               "tap_ops", "avr_travel_z1", "avr_travel_z2", "avr_travel_z3",
               "ds_v_rms", "ds_v_worst", "ds_v_viol_frac",
               "ds_q_rmse_mvar", "ds_q_max_mvar",
               "ds_v_rms_DSO_1", "ds_v_rms_DSO_2", "ds_v_rms_DSO_3", "ds_v_rms_DSO_4",
               "ds_q_rmse_DSO_1", "ds_q_rmse_DSO_2", "ds_q_rmse_DSO_3", "ds_q_rmse_DSO_4"]


def collect_rows() -> pd.DataFrame:
    rows = []
    for p in sorted(RUNS_DIR.glob("scenario_*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            continue
        if not rec.get("accepted"):
            continue
        for variant, m in rec["metrics"].items():
            row = {"seed": rec["seed"], "variant": variant,
                   "start_time": rec["start_time"], "n_events": rec["n_events"],
                   "wall_s": rec.get("wall_s", {}).get(variant, float("nan"))}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> None:
    if df.empty:
        print("no accepted scenarios yet -- nothing to aggregate")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "metrics_per_run.csv", index=False)

    L = lad()
    order = [v for v in L.LADDER if v in set(df["variant"])]
    present = [c for c in METRIC_COLS if c in df.columns]
    summ = (df.groupby("variant")[present]
              .agg(["mean", "std", "median",
                    lambda s: s.quantile(.25), lambda s: s.quantile(.75)]))
    summ.columns = ["_".join([a, {"<lambda_0>": "q25",
                                  "<lambda_1>": "q75"}.get(b, b)])
                    for a, b in summ.columns]
    summ = summ.reindex(order)
    summ.to_csv(OUT_DIR / "ladder_distribution.csv")

    # Step-sign stability: the payoff the single run cannot give.
    steps = list(zip(L.LADDER, L.LADDER[1:]))
    piv = df.pivot_table(index="seed", columns="variant", values="e_v")
    out = []
    for a, b in steps:
        if a not in piv.columns or b not in piv.columns:
            continue
        d = (piv[b] - piv[a]).dropna()
        if d.empty:
            continue
        out.append({"step": f"{a}->{b}", "n": int(d.size),
                    "mean_delta_e_v": float(d.mean()),
                    "median_delta_e_v": float(d.median()),
                    "frac_improving": float((d < 0).mean()),
                    "frac_keeps_sign": float((np.sign(d) == np.sign(d.mean())).mean())})
    if out:
        pd.DataFrame(out).to_csv(OUT_DIR / "step_sign_stability.csv", index=False)

    # Per-rung survival over ACCEPTED scenarios (L1 may be absent from some).
    surv = []
    seeds_acc = set()
    for p_ in RUNS_DIR.glob("scenario_*.json"):
        try:
            r_ = json.loads(p_.read_text(encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            continue
        if r_.get("accepted"):
            seeds_acc.add(r_["seed"])
    for v in order:
        n_ok = int(df[df["variant"] == v]["seed"].nunique())
        surv.append({"variant": v, "n_converged": n_ok,
                     "n_accepted_scenarios": len(seeds_acc),
                     "survival_frac": (n_ok / len(seeds_acc)) if seeds_acc else float("nan")})
    if surv:
        pd.DataFrame(surv).to_csv(OUT_DIR / "variant_survival.csv", index=False)
        print("" + chr(10) + "per-rung survival over accepted scenarios:")
        for r_ in surv:
            print(f"  {r_['variant']:<4} {r_['n_converged']:>4}/{r_['n_accepted_scenarios']:<4}"
                  f"  {r_['survival_frac']:.2f}")

    n_acc = df["seed"].nunique()
    print(f"\naccepted scenarios: {n_acc}")
    print(f"\n{'variant':<8}" + "".join(f"{c:>14}" for c in
                                        ("e_v mean", "e_v std", "e_p mean",
                                         "peak_line", "tap_ops")))
    print("-" * 78)
    for v in order:
        s = df[df["variant"] == v]
        print(f"{v:<8}{s['e_v'].mean():>14.5f}{s['e_v'].std():>14.5f}"
              f"{s['e_p'].mean():>14.5f}{s['peak_line_trip'].mean():>14.5f}"
              f"{s['tap_ops'].mean():>14.1f}")
    if out:
        print(f"\n{'step':<10}{'n':>5}{'mean d e_v':>13}{'frac improving':>17}")
        print("-" * 46)
        for r in out:
            print(f"{r['step']:<10}{r['n']:>5}{r['mean_delta_e_v']:>13.5f}"
                  f"{r['frac_improving']:>17.2f}")
    print(f"\nwritten: {OUT_DIR}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def _done_seeds() -> set:
    return {int(p.stem.split("_")[1]) for p in RUNS_DIR.glob("scenario_*.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=N_RUNS_DEFAULT,
                    help="accepted (paired-valid) scenarios to collect")
    ap.add_argument("--jobs", type=int, default=1,
                    help="scenarios in flight; each runs its 7 rungs serially")
    ap.add_argument("--seed", type=int, default=BASE_SEED_DEFAULT)
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds already attempted on disk")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild tables from runs/ without simulating")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds to run instead of sweeping")
    ap.add_argument("--no-gating", action="store_true",
                    help="diagnostic: run every rung, never discard")
    ap.add_argument("--tag", default=None,
                    help="write to runs_<tag>/ instead of runs/")
    args = ap.parse_args()

    # Env must be set BEFORE the pool spawns, so children see the same settings.
    global OUT_DIR, RUNS_DIR, NON_GATING
    if args.tag:
        os.environ["CH10_MC_TAG"] = args.tag
        OUT_DIR = Path(RESULTS_ROOT) / f"ch10_ladder_mc_{args.tag}"
        RUNS_DIR = OUT_DIR / "runs"
    if args.no_gating:
        os.environ["CH10_MC_NO_GATING"] = "1"
        NON_GATING = ("L1", "L2", "S1", "O1", "O2", "O3", "M")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    if args.replot:
        aggregate(collect_rows())
        return

    L = lad()
    L.assert_ladder_controlled()           # pre-flight, not post-mortem
    el = get_elements()
    print(f"  [elements] gens={el['gens']}  {len(el['lines'])} TN lines  "
          f"{len(el['load_buses'])} load buses")
    print(f"  [ladder]   {L.LADDER}")
    print(f"  [config]   horizon={HORIZON_MIN} min  g_w_gen="
          f"{L.make_thesis_config().g_w_gen:.3g}")
    print(f"  [campaign] target={args.runs} accepted  jobs={args.jobs}  "
          f"base_seed={args.seed}", flush=True)

    attempted = _done_seeds() if args.resume else set()
    accepted = len([p for p in RUNS_DIR.glob("scenario_*.json")
                    if json.loads(p.read_text(encoding="utf-8")).get("accepted")]) \
        if args.resume else 0
    print(f"  [resume]   {len(attempted)} attempted, {accepted} accepted on disk",
          flush=True)

    if args.seeds:
        want = [int(x) for x in args.seeds.split(",")]
        print(f"  [seeds]    explicit: {want}  gating="
              f"{'OFF' if args.no_gating else 'S1..M'}", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex2:
            futs = {ex2.submit(run_one_scenario, sd): sd for sd in want}
            for fut in as_completed(futs):
                sd = futs[fut]
                try:
                    res = fut.result()
                except Exception:                   # noqa: BLE001
                    print(f"  [seed {sd}] WORKER CRASH", flush=True)
                    traceback.print_exc()
                    continue
                print(f"  [seed {sd}] accepted={res['accepted']} "
                      f"fail={res['failing_variant']}", flush=True)
        aggregate(collect_rows())
        return

    attempt = 0
    restarts = 0
    MAX_RESTARTS = 20
    in_flight: Dict[Any, int] = {}
    ex = ProcessPoolExecutor(max_workers=max(1, args.jobs))

    def _rebuild(reason: str):
        """Recover from a child dying natively (no Python exception).

        A single abrupt child death breaks the whole pool and, before this,
        ended a multi-day campaign outright.  Observed 2026-09-08 at
        ``--jobs 18`` while a parallel session held ~19 other Python
        processes; the shared node-locked Gurobi academic licence is the
        prime suspect.  Seeds that were in flight are un-marked so they are
        retried rather than silently skipped -- they wrote no scenario JSON,
        so ``--resume`` would not have caught them either.
        """
        nonlocal ex, in_flight, restarts
        restarts += 1
        lost = sorted(in_flight.values())
        for sd in lost:
            attempted.discard(sd)
        print(f"  [pool] {reason}; restart {restarts}/{MAX_RESTARTS}, "
              f"requeueing {len(lost)} seed(s)", flush=True)
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:                           # noqa: BLE001
            pass
        in_flight = {}
        ex = ProcessPoolExecutor(max_workers=max(1, args.jobs))

    try:
        while accepted < args.runs:
            while len(in_flight) < max(1, args.jobs) and accepted + len(in_flight) < args.runs * 3:
                seed = args.seed + attempt
                attempt += 1
                if seed in attempted:
                    continue
                attempted.add(seed)
                try:
                    in_flight[ex.submit(run_one_scenario, seed)] = seed
                except BrokenProcessPool:
                    attempted.discard(seed)
                    if restarts >= MAX_RESTARTS:
                        raise
                    _rebuild("pool broken on submit")
                    break
            if not in_flight:
                break
            for fut in as_completed(list(in_flight)):
                seed = in_flight.pop(fut)
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    if restarts >= MAX_RESTARTS:
                        raise
                    attempted.discard(seed)
                    _rebuild(f"child died during seed {seed}")
                    break
                except Exception:                   # noqa: BLE001
                    print(f"  [seed {seed}] WORKER CRASH", flush=True)
                    traceback.print_exc()
                    break
                if res["accepted"]:
                    accepted += 1
                    print(f"  [seed {seed}] accepted  ({res['n_events']} events)  "
                          f"-> {accepted}/{args.runs}", flush=True)
                else:
                    print(f"  [seed {seed}] discarded at {res['failing_variant']}",
                          flush=True)
                break                                # refill the pool
    finally:
        ex.shutdown(wait=True)

    aggregate(collect_rows())


if __name__ == "__main__":
    main()
