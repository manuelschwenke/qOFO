#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/006_CIGRE_MONTECARLO.py
===================================
Monte-Carlo extension of ``experiments/005_CIGRE_MULTI.py`` for the CIGRE
Energy Forum 2026 paper.  Where 005 runs the four/five control variants on a
**single** deterministic 300-min time series with one fixed contingency
schedule, this driver repeats that experiment over **many randomized
scenarios** so the Table-3 ranking V1->V5 can be reported as a *distribution*
(mean +/- spread) and shown to be stable across scenarios -- the single most
convincing robustness addition to the case study.

Design (locked with the user)
------------------------------
* **Paired runs.**  Each scenario draws a random ``start_time`` (anywhere in the
  2016 SimBench profile year) and a random contingency schedule; **all five
  variants V1..V5 are run on that identical scenario** so the per-run ranking is
  a clean paired comparison.
* **Runtime** always 300 min (``make_cigre_config`` -> ``n_total_s = 60*60*5``).
* **Random schedule.**  Candidate slots every 30 min (min 30,60,...,270); each
  slot fires a contingency with 25 % probability (~2-3 events/run).  An event is
  a line trip, a generator trip, or a load connection, chosen uniformly among
  the *currently admissible* types.
* **Lightly constrained** for power-flow feasibility: at most one line and one
  generator out at a time, the slack machine is never tripped, and load
  connections use the proven-stable 200 MW / 100 Mvar step.  A tripped generator
  reconnects 180 min later (omitted if that exceeds the 300-min horizon); line
  outages and load connections persist to the end.
* **Drop-and-replace.**  A run is *accepted* only if **all five** variants
  converge (full 300-step log).  If any variant diverges the whole scenario is
  discarded and a fresh random scenario (next seed) is drawn, until ``--runs``
  paired-valid scenarios have been collected.

Reproducibility
---------------
Run ``r`` uses ``numpy.random.default_rng(BASE_SEED + attempt)``; the seed is
stored in every CSV, so any scenario can be regenerated.  The full contingency
schedule is also written verbatim to ``contingency_schedules.csv`` and can be
reloaded with :func:`load_schedule_csv`.

Outputs (under ``results/006_cigre_mc/``)
----------------------------------------
* ``contingency_schedules.csv`` -- one row per event of every accepted run.
* ``metrics_per_run.csv``       -- one row per (run_id, variant): the six
  Table-3 metrics (``cigre_summary_table``) + convergence flag.
* ``failed_attempts.csv``       -- discarded (non-converging) scenarios.
* ``timeseries/run_XXXX.npz``   -- compact per-run series for the aggregate
  figures (voltage-RMS error per variant; selected-generator P/Q per variant).
* ``table3_distribution.csv``   -- per-variant mean/std/median/IQR of each
  metric over the accepted runs (the populated Table 3).
* ``ranking_stability.csv``     -- per-metric fraction of runs in which V4 is
  best among V1-V4, plus mean rank of each variant.
* ``Fig_mc_voltage_band.pdf``   -- TS voltage tracking error, mean +/- 1 sigma
  band across runs (one panel per variant).
* ``Fig_mc_capability.pdf``     -- generator P-Q operating-point clouds pooled
  over all runs, against the Milano capability envelope.
* ``Fig_mc_table3_box.pdf``     -- box plots of each metric across variants.

CLI
---
* ``--runs N``   number of paired-valid scenarios to collect (default 50).
* ``--seed S``   base seed (default fixed; ``seed = S + attempt``).
* ``--resume``   continue an interrupted batch (skip already-collected /
                 already-failed seeds; keep appending).
* ``--replot``   skip all simulation; rebuild tables + figures from the CSVs and
                 ``timeseries/*.npz`` already on disk.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

# Headless rendering for the batch (must precede any matplotlib import, which is
# pulled in transitively via visualisation.style / plot_cigre).
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
# Pin BLAS/OpenMP to a single thread per process.  With scenario-level process
# parallelism (``--jobs P``) this prevents P workers x (many BLAS threads) from
# oversubscribing the cores; harmless for the single-process path.  Must precede
# the numpy import below.  Override by exporting these before launching.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib as mpl
mpl.use("Agg")

import argparse
import contextlib
import copy
import glob
import json
import shutil
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Force UTF-8 console (controller logs print a Greek Delta in capability msgs).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent, MultiTSOIterationRecord
from experiments.runners import run_multi_tso_dso
from analysis.reachability import ReachabilityViolation
from experiments.helpers.comparison_metrics import (
    cigre_summary_table,
    gen_s_rated_by_zone,
    voltage_rms_err_all,
)


# ===========================================================================
#  User-facing knobs
# ===========================================================================

#: Output directory inside the repo.
OUT_ROOT = os.path.join("results", "006_cigre_mc")
#: Per-run compact time-series store (feeds the aggregate figures + --replot).
TS_DIR = os.path.join(OUT_ROOT, "timeseries")
#: Throwaway result_dir for the per-variant runs (controllers may write here).
SCRATCH_DIR = os.path.join(OUT_ROOT, "_scratch")
#: The CIGRE paper's figure directory (PDFs are written here too, best-effort).
PAPER_FIG_DIR = r"C:\Users\Manuel Schwenke\Desktop\CIGRE_2026\Figures"

#: Voltage setpoint (matches MultiTSOConfig.v_setpoint_pu / paper V_ref).
V_SET = 1.03

#: Generators shown in the aggregate P-Q capability figure (match by name).
GEN_SELECT: List = ["G3_bus31", "G4_bus32", "G7_bus35"]

#: Variant order (canonical; V4 = proposed, V5 = centralized upper bound).
VARIANT_ORDER: List[str] = ["V1", "V2", "V3", "V4", "V5"]

#: Table-3 metric columns produced by ``cigre_summary_table``.
METRIC_COLS: List[str] = [
    "rms_v_ts_pu", "m_bar_mvar", "m_bar_pu", "res_util",
    "rms_e_sts_mvar", "rms_q_tie_mvar", "n_sw",
]
#: Direction per metric: True = lower is better (for ranking-stability scoring).
LOWER_BETTER: Dict[str, bool] = {
    "rms_v_ts_pu": True,      # voltage tracking error
    "m_bar_mvar": False,      # gen Q headroom preserved [Mvar] (higher = better)
    "m_bar_pu": False,        # gen Q headroom preserved [p.u. S_n] (higher = better)
    "res_util": True,         # gen reactive-capability utilisation (lower = more reserve)
    "rms_e_sts_mvar": True,   # STS interface tracking error
    "rms_q_tie_mvar": True,   # inter-zone tie-line Q
    "n_sw": True,             # discrete switching operations
}
#: Metrics drawn in the one-row box-plot figure (subset of METRIC_COLS, order
#: preserved).  Override at the CLI with ``--box-metrics a,b,c`` (also honoured
#: by ``--replot``).
BOX_METRICS_DEFAULT: List[str] = ["rms_v_ts_pu", "res_util"] #, "rms_q_tie_mvar", "n_sw"]
#: Mutable selection set by the CLI; read by the box-plot figure.
BOX_METRICS: List[str] = list(BOX_METRICS_DEFAULT)
#: Short axis labels for the box-plot panels.
METRIC_LABELS: Dict[str, str] = {
    "rms_v_ts_pu":    "RMS |dV| TS [p.u.]",
    "m_bar_mvar":     "gen Q headroom [Mvar]",
    "m_bar_pu":       "gen Q reserve [p.u. $S_n$]",
    "res_util":       "gen Q utilisation [-]",
    "rms_e_sts_mvar": "STS iface err [Mvar]",
    "rms_q_tie_mvar": "tie-line Q RMS [Mvar]",
    "n_sw":           "switching ops [-]",
}

# ── Random-schedule parameters ────────────────────────────────────────────
BASE_SEED_DEFAULT = 20260602
N_RUNS_DEFAULT = 100
SLOT_INTERVAL_MIN = 30        # candidate slots at 30,60,...
SLOT_PROB = 0.25              # per-slot contingency probability
GEN_RESTORE_DELAY_MIN = 180   # tripped gen reconnects after this many minutes
LOAD_P_MW = 150.0             # active component of the connected load [MW]
#: Reactive component of the connected contingency load is drawn uniformly in
#: [LOAD_Q_MIN_MVAR, LOAD_Q_MAX_MVAR] per event, to inject more severe (and
#: variable) reactive disturbances than the old fixed 100 Mvar step.
LOAD_Q_MIN_MVAR = 100.0
LOAD_Q_MAX_MVAR = 400.0
MAX_GENS_OUT = 1              # concurrency cap (gen)
MAX_LINES_OUT = 1            # concurrency cap (line; persists -> 1 line/run)
#: Cumulative cap on connected contingency loads.  Kept at 1 so the total added
#: load never exceeds the proven-stable 200 MW / 100 Mvar step: two concurrent
#: 200/100 connects (= 400/200) collapse the V1 cos(phi)=1 baseline (005 notes),
#: which would needlessly reject scenarios in the drop-and-replace loop.
MAX_LOADS_CONNECTED = 1
#: Safety cap on total scenario attempts so the drop-and-replace loop cannot
#: spin forever if the constrained randomness still diverges too often.
MAX_ATTEMPT_FACTOR = 4

# ── Base-case pre-screen ───────────────────────────────────────────────────
#: Before running the 5 variants, cheaply test whether the *base* operating
#: point (no contingency) is feasible across the window, by running the weakest
#: variant (V1, cos(phi)=1) with no contingencies at a coarse step.  If V1's
#: base case diverges (e.g. winter-peak start_time where cos(phi)=1 cannot hold
#: voltages), the start_time is resampled instead of wasting five full runs.
#: The OFO variants add reactive support, so a V1-feasible base implies the
#: others are feasible at base too; contingency-induced divergence is still
#: caught by the full run -> drop-and-replace.  Disable with ``--no-prescreen``.
PRESCREEN_BASE_DEFAULT = True
#: Coarse step for the pre-screen run (15 min -> 20 probes over the 300-min
#: window, spanning the load peak).  V1 is local-control only (no MIQP), so this
#: is far cheaper than a full 300-step run.
PRESCREEN_DT_S = 900.0

# ── Profile-year window for the random start_time ─────────────────────────
PROFILE_START = datetime(2016, 1, 1, 0, 0)
#: Latest start that still leaves > 5 h (the 300-min horizon) of profile.
PROFILE_END_USABLE = datetime(2016, 12, 31, 18, 0)


# ===========================================================================
#  Configuration + variant overrides (COPIED VERBATIM from 005_CIGRE_MULTI.py;
#  keep in sync if 005's tuning changes).
# ===========================================================================


def make_cigre_config() -> MultiTSOConfig:
    """Shared run configuration for all variants (identical to 005)."""
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 5,  # 36-hour (2160-min) simulation
        tso_period_s=60.0 * 3,  # TS-OFO every 3 min
        dso_period_s=20.0,  # DSO-OFO each plant step (dt_s=60 >= 10)
        dt_s=20,
        g_v=1E7,  # TSO voltage tracking; drives PCC Q dispatch
        g_q=200,  # DSO Q-tracking
        tso_g_q_tie=0,
        tso_g_res_sg=0,
        # ── DSO objective tuning ──
        dso_g_v=1E5,  # reduced to avoid competing with Q tracking
        dso_g_qi=0,  # integral Q-tracking (0 = off)
        dso_lambda_qi=0.95,  # leaky integrator decay
        dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,  # DER-primary, OLTC-backup
        # ── TSO weights (w-shift closed-loop curvature) ──
        g_w_der=100,
        g_w_gen=5e9,
        g_w_pcc=200,
        g_w_tso_oltc=1E4,
        # shunt
        install_tso_tertiary_shunts=False,
        shunt_dispatch="off",  # "integrator"
        g_w_tso_shunt=12000,
        tso_shunt_kind="msc_msr",  # one capacitor + one reactor bank per DSO
        tso_shunt_msc_n_levels=2,  # MSC steps 0..N
        tso_shunt_msr_n_levels=2,  # MSR steps 0..N
        tso_shunt_msc_q_step_mvar=25.0,  # Mvar per MSC step
        tso_shunt_msr_q_step_mvar=25.0,  # Mvar per MSR step
        # integrator tuning
        shunt_int_g_w=150,  # step = g_H/(2*g_w); SMALLER = bigger step — TUNE THIS
        shunt_int_delta_mvar=10.0,  # hysteresis half-width (must be < q_step/2 = 25)
        shunt_int_t_dwell_s=30 * 60.0,  # min seconds between commits per bank (anti-chatter)
        shunt_int_v_min_pu=0.90,  # HV feasibility band (overshoot guard)
        shunt_int_v_max_pu=1.10,
        # ── DSO weights ──
        g_w_dso_der=1000,
        g_w_dso_oltc=200,
        # ── Local-mode OLTC tap-rate limits (V1/V2 MT+NC, V3 NC) ──
        # max_step=1 (default) + wall-clock cooldown per OLTC type:
        #   MT (machine 2W gen-trafo) -> 1 tap / 180 s = once per TS interval.
        #   NC (coupler 3W interface) -> 1 tap / 60 s  = once per minute.
        # Cooldowns are wall-clock, hence robust to dt_s / dso_period_s.
        local_oltc_max_step_per_dt=1,
        oltc_cooldown_s_mt=180.0,
        oltc_cooldown_s_nc=60.0,
        use_fixed_zones=True,  # literature 3-area partition
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,
        verbose=1,
        tso_qv_vref_pu=1.03,
        tso_qv_slope_pu=0.06,
        tso_qv_deadband_pu=0.01,
        dso_qv_vref_pu=1.03,
        dso_qv_slope_pu=0.06,
        dso_qv_deadband_pu=0.01,
        # Live plots OFF for the batch sweep (see module docstring).
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        live_plot_tracking=False,
        live_plot_show_reserves=True,
        live_plot_show_tie_flows=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # Preconditioning of g_w
        precondition_g_w=False,  # turn it on
        precondition_lambda_target=0.5,  # target λ_max(M); 0.9 = well-damped, <2 stable
        # optional:
        precondition_granularity="class",  # or "column"
        precondition_exclude_classes=("gen",),  # AVR setpoint left at config
        # ── Profile & contingency settings ──
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
        ],
    )
    cfg.scenario = "wind_replace"
    cfg.warmup_s = 0.0
    # Distributed slack: the active-power imbalance after a disturbance (load
    # connection, generator trip) is shared across all machines weighted by
    # ``slack_weight`` (~ rating), rather than dumped on one slack machine.
    # This requires the zonal-dispatch spill bug fix (network/ieee39/
    # zonal_balancing.py, 2026-06-02): previously the spill loop re-counted the
    # cross-zone deficit each iteration and dispatched the slack gen ~2x the real
    # residual, which distributed_slack then honoured -> 200%+ overloads and
    # collapse at high load.  With total dispatch == residual the mismatch
    # distributed is just losses, so high-load start_times now converge.
    cfg.distributed_slack = True
    return cfg


#: Per-variant control-mode overrides (identical to 005_CIGRE_MULTI.VARIANTS).
VARIANTS: Dict[str, Dict[str, Any]] = {
    "V1": dict(  # 002 "L0" -- TS Q(V), STS cos phi = 1
        tso_mode="local", tso_local_mode="qv",
        dso_mode="local", local_der_mode="cos_phi_1",
        tso_q_mode="qv", dso_q_mode="cosphi",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
    ),
    "V2": dict(  # 002 "L1" -- TS Q(V), STS local Q(V)
        tso_mode="local", tso_local_mode="qv",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv", dso_q_mode="qv",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
        dso_qv_vref_pu=1.03, dso_qv_slope_pu=0.06, dso_qv_deadband_pu=0.01,
    ),
    "V3": dict(  # 002 "T1" -- TS-OFO, STS local Q(V); g_w_pcc pin
        tso_mode="ofo",
        dso_mode="local", local_der_mode="qv",
        tso_q_mode="qv", dso_q_mode="qv",
        tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
        dso_qv_vref_pu=1.03, dso_qv_slope_pu=0.06, dso_qv_deadband_pu=0.01,
        g_w_pcc=1.0e8,
    ),
    "V4": dict(  # 002 "C" -- cascaded TS-OFO + STS-OFO (proposed)
        tso_mode="ofo",
        dso_mode="ofo",
    ),
    "V5": dict(  # single centralized OFO -- best-case upper-bound reference
        control_scope="central",
        tso_mode="ofo",
        dso_mode="local",
        tso_q_mode="qv", dso_q_mode="qv",
        local_sensitivities_tso=False,
        local_sensitivities_dso=False,
        # ── Tuning (plan golden-popping-hartmanis): make V5 a VALID upper
        #    bound (>= V4 on rms_v_ts_pu).  Strategy: match V4's per-loop gain
        #    ratios g_v/g_w, match V4's control cadence, then cool the whole g_w
        #    block by one global KAPPA_V5 (set from the lambda_max probe below).
        central_dso_g_v=1E5,        # HV voltage weight = V4 dso_g_v
        central_period_s=20,       # = V4 tso_period_s: cadence-matched comparison
        # -- V4 gain ratios restored (g_v/g_w per class), then x KAPPA_V5 on g_w --
        # g_v=1E7,                    # TN voltage weight = V4 g_v (was 5E7)
        # g_w_der=100,                # = V4 (TSO DER);  ratio g_v/g_w_der = 1e5
        # g_w_dso_der=1000,           # = V4 (DSO DER);  ratio cdso_g_v/g_w = 1e2
        # g_w_gen=5E9,                # = V4 (very cautious AVR)
        # g_w_tso_oltc=1E4,           # = V4 (was 2E4)
        # g_w_dso_oltc=200,           # = V4 (was 1E3)
        # debug_central_curvature=True,  # enable to print lambda_max(M) at t=0
    ),
}
KAPPA_V5: float = 1.25  # lambda_max(M)=1.237 at kappa=1 -> 1.25 gives ~0.99 (well-damped)
_V5_GW_KEYS = ("g_w_der", "g_w_dso_der", "g_w_gen", "g_w_tso_oltc", "g_w_dso_oltc")
if KAPPA_V5 != 1.0:
    # Only scale the g_w keys that V5 sets *explicitly*; when a key is commented
    # out above, V5 inherits it from make_cigre_config() and KAPPA_V5 does not
    # apply to it (run_variant only setattr()s keys present in VARIANTS["V5"]).
    for _k in _V5_GW_KEYS:
        if _k in VARIANTS["V5"]:
            VARIANTS["V5"][_k] = VARIANTS["V5"][_k] * KAPPA_V5

# ===========================================================================
#  Valid-element enumeration (built once)
# ===========================================================================


def enumerate_elements(cfg: MultiTSOConfig) -> Dict[str, List[int]]:
    """Enumerate the admissible contingency elements for the random schedule.

    Rebuilds the combined net exactly as ``run_multi_tso_dso`` does
    (``build_ieee39_net`` -> ``fixed_zone_partition_ieee39`` ->
    ``add_hv_networks``) so element indices line up with what every variant run
    sees, then derives:

    * ``gens``       -- machine-generator indices (those with a machine
                        transformer), excluding the slack machine.
    * ``lines``      -- in-service TN transmission lines (not HV feeders).
    * ``load_buses`` -- TN PQ buses that host an existing load (realistic load
                        connection targets).
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

    # Machine generators (gen<->trafo map), excluding the slack machine.
    gen_trafo_map = {
        int(g): int(t)
        for t, g in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map)
        if g is not None and int(g) >= 0
    }
    slack_gens = set()
    if "slack" in net.gen.columns:
        slack_gens = {int(g) for g in net.gen.index if bool(net.gen.at[g, "slack"])}
    gens = sorted(
        g for g in gen_trafo_map
        if g not in slack_gens and bool(net.gen.at[g, "in_service"])
    )

    # TN transmission lines (exclude HV feeders so we never island a radial DSO).
    hv_lines = {int(li) for hv in meta.hv_networks for li in hv.line_indices}
    lines = sorted(
        int(li) for li in net.line.index
        if int(li) not in hv_lines and bool(net.line.at[li, "in_service"])
    )

    # TN PQ buses (exclude PV/slack gen + ext-grid buses, exclude HV buses).
    pv_slack = (
        {int(net.gen.at[g, "bus"]) for g in net.gen.index}
        | {int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index}
    )
    hv_buses = {int(b) for hv in meta.hv_networks for b in hv.bus_indices}
    tn_pq = {int(b) for b in net.bus.index if b not in pv_slack and b not in hv_buses}
    load_buses = sorted(
        {int(net.load.at[li, "bus"]) for li in net.load.index
         if int(net.load.at[li, "bus"]) in tn_pq}
    )
    if not load_buses:           # defensive fallback
        load_buses = sorted(tn_pq)

    print(f"  [elements] gens={gens}  |  {len(lines)} TN lines  |  "
          f"{len(load_buses)} candidate load buses")
    return {"gens": gens, "lines": lines, "load_buses": load_buses}


# ===========================================================================
#  Random contingency schedule
# ===========================================================================


def build_random_schedule(rng: np.random.Generator,
                          elements: Dict[str, List[int]],
                          n_total_min: int) -> List[ContingencyEvent]:
    """Draw one lightly-constrained random contingency schedule.

    See the module docstring for the rules.  Returns the events sorted by
    ``(minute, element_type, action)``.
    """
    gens = elements["gens"]
    lines = elements["lines"]
    load_buses = elements["load_buses"]

    events: List[ContingencyEvent] = []
    gen_out_intervals: List[Tuple[int, int]] = []  # (start_min, end_min)
    used_lines: set = set()
    used_load_buses: set = set()
    loads_connected = 0

    def gen_window_free(m: int) -> bool:
        """True iff a gen-out interval [m, m+180] overlaps no existing one
        (enforces MAX_GENS_OUT = 1 over the whole horizon)."""
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
            events.append(ContingencyEvent(
                minute=m, element_type="gen", element_index=g, action="trip"))
            end = m + GEN_RESTORE_DELAY_MIN
            if end <= n_total_min:
                events.append(ContingencyEvent(
                    minute=end, element_type="gen", element_index=g,
                    action="restore"))
            gen_out_intervals.append((m, min(end, n_total_min)))

        elif kind == "line":
            avail = sorted(set(lines) - used_lines)
            li = int(avail[int(rng.integers(len(avail)))])
            events.append(ContingencyEvent(
                minute=m, element_type="line", element_index=li, action="trip"))
            used_lines.add(li)

        else:  # load
            avail = sorted(set(load_buses) - used_load_buses)
            b = int(avail[int(rng.integers(len(avail)))])
            q = round(float(rng.uniform(LOAD_Q_MIN_MVAR, LOAD_Q_MAX_MVAR)), 1)
            events.append(ContingencyEvent(
                minute=m, element_type="load", element_index=-1, bus=b,
                p_mw=LOAD_P_MW, q_mvar=q, action="connect"))
            used_load_buses.add(b)
            loads_connected += 1

    events.sort(key=lambda e: (e.minute, e.element_type, e.action))
    return events


def schedule_to_rows(run_id: int, seed: int, start_time: datetime,
                     schedule: List[ContingencyEvent]) -> List[Dict[str, Any]]:
    """Flatten a schedule to CSV rows (one per event)."""
    rows = []
    for ev in schedule:
        rows.append({
            "run_id": run_id,
            "seed": seed,
            "start_time": start_time.isoformat(),
            "minute": ev.minute,
            "element_type": ev.element_type,
            "element_index": ev.element_index,
            "action": ev.action,
            "bus": "" if ev.bus is None else int(ev.bus),
            "p_mw": "" if not np.isfinite(ev.p_mw) else float(ev.p_mw),
            "q_mvar": "" if not np.isfinite(ev.q_mvar) else float(ev.q_mvar),
        })
    return rows


def load_schedule_csv(path: str, run_id: int) -> List[ContingencyEvent]:
    """Reload one run's schedule from ``contingency_schedules.csv``.

    Round-trips :func:`schedule_to_rows`; used by the verification step and for
    re-running a specific scenario later.
    """
    df = pd.read_csv(path)
    sub = df[df["run_id"] == run_id]
    out: List[ContingencyEvent] = []
    for _, r in sub.iterrows():
        bus = r["bus"]
        bus = None if (pd.isna(bus) or bus == "") else int(bus)
        p = r["p_mw"]
        p = float("nan") if (pd.isna(p) or p == "") else float(p)
        q = r["q_mvar"]
        q = float("nan") if (pd.isna(q) or q == "") else float(q)
        out.append(ContingencyEvent(
            minute=int(r["minute"]),
            element_type=str(r["element_type"]),
            element_index=int(r["element_index"]),
            action=str(r["action"]),
            bus=bus, p_mw=p, q_mvar=q,
        ))
    out.sort(key=lambda e: (e.minute, e.element_type, e.action))
    return out


# ===========================================================================
#  Run one variant + extract compact time series
# ===========================================================================


def _dump_mc_reach_margins(seed: int, variant: str,
                           margins: List[Dict[str, Any]]) -> None:
    """Persist a rejected variant's voltage-stability margin trajectory to
    ``RUNS_DIR/reach_margins_<seed>_<variant>.csv`` (per-seed/variant name so
    parallel workers never collide).  Never blocks the sweep on an I/O error."""
    if not margins:
        return
    try:
        pd.DataFrame(margins).to_csv(
            os.path.join(RUNS_DIR, f"reach_margins_{seed}_{variant}.csv"),
            index=False,
        )
    except (PermissionError, OSError):
        pass


def run_variant(name: str, start_time: datetime,
                schedule: List[ContingencyEvent]
                ) -> Tuple[List[MultiTSOIterationRecord], bool,
                           Optional[str], Optional[List[Dict[str, Any]]]]:
    """Run a single variant on the given scenario.

    Returns ``(log, converged, reason, margins)``:

    * ``converged`` -- True iff the full 300-step log was produced.
    * ``reason`` -- rejection cause when not converged: ``"voltage_unstable"``
      for a nose-curve / lower-branch reachability violation,
      ``"diverged"`` for a power-flow divergence, ``None`` on success.  This
      separates the (physically distinct) voltage-instability rejections from
      genuine non-convergence in the Monte-Carlo feasibility statistics.
    * ``margins`` -- the per-step voltage-stability margin trajectory up to the
      violation (``None`` unless ``reason == "voltage_unstable"``).
    """
    cfg = make_cigre_config()
    cfg.start_time = start_time
    cfg.contingencies = copy.deepcopy(schedule)  # runner mutates element_index
    for k, v in VARIANTS[name].items():
        setattr(cfg, k, v)
    # Per-process scratch dir so parallel workers never collide on result_dir.
    cfg.result_dir = os.path.join(SCRATCH_DIR, f"pid{os.getpid()}")
    os.makedirs(cfg.result_dir, exist_ok=True)
    cfg.verbose = 0
    n_steps = int(round(cfg.n_total_s / cfg.dt_s))
    reason: Optional[str] = None
    margins: Optional[List[Dict[str, Any]]] = None
    try:
        log = run_multi_tso_dso(cfg)
    except ReachabilityViolation as rv:
        # Converged power flow but on/beyond the nose (lower branch): reject as
        # voltage-unstable, distinct from a divergence.  Keep the records and
        # the margin trajectory computed before the violation.
        print(f"      [{name}] VOLTAGE-UNSTABLE (nose): {rv}")
        log = list(rv.partial_log) if rv.partial_log else []
        reason = "voltage_unstable"
        margins = rv.margins
    except Exception as exc:  # noqa: BLE001 - divergence -> reject scenario
        print(f"      [{name}] diverged: {type(exc).__name__}: {exc}")
        log = []
        reason = "diverged"
    return log, (len(log) == n_steps), reason, margins


def _gen_selection_info() -> List[Dict[str, Any]]:
    """Resolve ``GEN_SELECT`` to gen_info dicts annotated with (zone, k_in_zone)
    and capability parameters, via the plot_cigre helpers."""
    from visualisation.plot_cigre import _gen_info_with_k, _select_gens
    gen_info = _gen_info_with_k("wind_replace")
    return _select_gens(gen_info, GEN_SELECT)


def extract_timeseries(log: List[MultiTSOIterationRecord],
                       sel_info: List[Dict[str, Any]]
                       ) -> Tuple[np.ndarray, np.ndarray,
                                  Dict[str, Tuple[np.ndarray, np.ndarray]],
                                  Dict[int, Dict[str, np.ndarray]]]:
    """Pull the compact per-run series needed for the aggregate figures.

    Returns ``(t_min, vrms_pu, pq, zv)`` where ``vrms_pu`` is the system-wide TS
    voltage-RMS-error series (``voltage_rms_err_all``); ``pq`` maps each selected
    generator name to ``(P_mw[n_steps], Q_mvar[n_steps])`` read from
    ``zone_p_gen`` / ``zone_q_gen``; and ``zv`` maps each TS zone to
    ``{'min','mean','max': array[n_steps]}`` -- the per-step envelope of the
    voltage-observed EHV buses in that zone (``zone_v_min/mean/max``), used by the
    per-zone voltage-range box plot.
    """
    d = voltage_rms_err_all(log, V_SET)
    t_min = np.asarray(d["t_min"], dtype=np.float64)
    vrms = np.asarray(d["rms_err_pu"], dtype=np.float64)

    pq: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    n = len(log)
    for gi in sel_info:
        z, k, nm = gi["zone"], gi["k_in_zone"], str(gi["name"])
        P = np.full(n, np.nan, dtype=np.float64)
        Q = np.full(n, np.nan, dtype=np.float64)
        for i, r in enumerate(log):
            pv = r.zone_p_gen.get(z)
            qv = r.zone_q_gen.get(z)
            if pv is not None and qv is not None and k < len(pv):
                P[i] = float(pv[k])
                Q[i] = float(qv[k])
        pq[nm] = (P, Q)

    # Per-zone EHV bus-voltage envelope (min/mean/max per step) for the
    # voltage-range box plot (Fig_mc_voltage_zone_box).
    zones_seen: set = set()
    for r in log:
        zones_seen.update(r.zone_v_mean.keys())
    zv: Dict[int, Dict[str, np.ndarray]] = {}
    for z in sorted(int(zz) for zz in zones_seen):
        vmin = np.full(n, np.nan, dtype=np.float64)
        vmean = np.full(n, np.nan, dtype=np.float64)
        vmax = np.full(n, np.nan, dtype=np.float64)
        for i, r in enumerate(log):
            if z in r.zone_v_min:
                vmin[i] = float(r.zone_v_min[z])
            if z in r.zone_v_mean:
                vmean[i] = float(r.zone_v_mean[z])
            if z in r.zone_v_max:
                vmax[i] = float(r.zone_v_max[z])
        zv[z] = {"min": vmin, "mean": vmean, "max": vmax}
    return t_min, vrms, pq, zv


def _write_ts_npz(path: str, t_min: np.ndarray,
                  vrms_by_variant: Dict[str, np.ndarray],
                  pq_by_variant: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                  zv_by_variant: Optional[Dict[str, Dict[int, Dict[str, np.ndarray]]]] = None,
                  ) -> None:
    """Persist one accepted run's compact series to ``path`` (.npz)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs: Dict[str, np.ndarray] = {"t_min": t_min}
    for V, a in vrms_by_variant.items():
        arrs[f"vrms__{V}"] = a
    for V, gd in pq_by_variant.items():
        for nm, (P, Q) in gd.items():
            arrs[f"P__{V}__{nm}"] = P
            arrs[f"Q__{V}__{nm}"] = Q
    # Per-zone voltage envelope: key "Vz__{variant}__{zone}__{min|mean|max}".
    for V, zd in (zv_by_variant or {}).items():
        for z, stats in zd.items():
            for stat, arr in stats.items():
                arrs[f"Vz__{V}__{z}__{stat}"] = arr
    np.savez_compressed(path, **arrs)


# ===========================================================================
#  Output paths + per-seed staging
# ===========================================================================

SCHED_CSV = os.path.join(OUT_ROOT, "contingency_schedules.csv")
METRICS_CSV = os.path.join(OUT_ROOT, "metrics_per_run.csv")
FAILED_CSV = os.path.join(OUT_ROOT, "failed_attempts.csv")
#: Per-seed staging directory.  Each tried scenario writes one
#: ``scenario_<seed>.json`` (+ ``ts_<seed>.npz`` when accepted, ``log_<seed>.txt``
#: when run in a worker).  The collector derives the master CSVs + canonical
#: ``timeseries/run_XXXX.npz`` from these, so the run is restart-safe and the
#: parallel workers never contend on a shared file.
RUNS_DIR = os.path.join(OUT_ROOT, "_runs")


def random_start_time(rng: np.random.Generator) -> datetime:
    """Random profile-year start, quantized to 15 min, leaving the 300-min tail."""
    total_min = int((PROFILE_END_USABLE - PROFILE_START).total_seconds() // 60)
    off = int(rng.integers(0, total_min))
    off -= off % 15
    return PROFILE_START + timedelta(minutes=off)


# ── Per-process caches (rebuilt lazily in each worker under spawn) ──────────
_ELEMENTS: Optional[Dict[str, List[int]]] = None
_SEL_INFO: Optional[List[Dict[str, Any]]] = None
_N_TOTAL_MIN: Optional[int] = None
_GEN_SRATED: Optional[Dict[int, np.ndarray]] = None


def _get_elements() -> Dict[str, List[int]]:
    global _ELEMENTS
    if _ELEMENTS is None:
        _ELEMENTS = enumerate_elements(make_cigre_config())
    return _ELEMENTS


def _get_gen_srated() -> Optional[Dict[int, np.ndarray]]:
    """Per-zone machine ratings for the size-comparable ``m_bar_pu`` metric;
    cached per process.  ``None`` if the gen-info build fails (m_bar_pu -> NaN)."""
    global _GEN_SRATED
    if _GEN_SRATED is None:
        try:
            _GEN_SRATED = gen_s_rated_by_zone(make_cigre_config().scenario)
        except Exception as exc:  # noqa: BLE001 - m_bar_pu is optional
            print(f"  [m_bar_pu] ratings unavailable: {type(exc).__name__}: {exc}")
            _GEN_SRATED = {}
    return _GEN_SRATED or None


def _get_sel_info() -> List[Dict[str, Any]]:
    global _SEL_INFO
    if _SEL_INFO is None:
        _SEL_INFO = _gen_selection_info()
    return _SEL_INFO


def _get_n_total_min() -> int:
    global _N_TOTAL_MIN
    if _N_TOTAL_MIN is None:
        _N_TOTAL_MIN = int(round(make_cigre_config().n_total_s / 60.0))
    return _N_TOTAL_MIN


def _scan_runs_dir() -> Tuple[set, int]:
    """Return ``(tried_seeds, n_accepted)`` from the per-seed JSONs in RUNS_DIR."""
    tried: set = set()
    n_acc = 0
    for f in glob.glob(os.path.join(RUNS_DIR, "scenario_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            continue
        tried.add(int(j["seed"]))
        if j.get("accepted"):
            n_acc += 1
    return tried, n_acc


def _prescreen_base_case(start_time: datetime, capture_to: Optional[str] = None
                         ) -> Tuple[bool, Optional[str]]:
    """Cheap base-case feasibility probe for ``start_time``.

    Runs the weakest variant (V1, cos(phi)=1) with **no contingencies** at the
    coarse ``PRESCREEN_DT_S`` step over the full window.  Returns
    ``(ok, reason)``: ``ok`` is True iff the full coarse log is produced (the
    base operating point is feasible); ``reason`` is ``"voltage_unstable"`` for
    a nose-curve reachability violation, ``"diverged"`` for a power-flow
    divergence, and ``None`` when feasible.  A failure here means the
    cos(phi)=1 base case collapses at this start_time (the dominant rejection
    cause), so the scenario should be resampled rather than burning five full
    runs.
    """
    cfg = make_cigre_config()
    cfg.start_time = start_time
    cfg.contingencies = []
    for k, v in VARIANTS["V1"].items():
        setattr(cfg, k, v)
    cfg.dt_s = PRESCREEN_DT_S
    cfg.verbose = 0
    cfg.result_dir = os.path.join(SCRATCH_DIR, f"pid{os.getpid()}_pre")
    os.makedirs(cfg.result_dir, exist_ok=True)
    n_steps = int(round(cfg.n_total_s / cfg.dt_s))
    try:
        log = run_multi_tso_dso(cfg)
    except ReachabilityViolation:
        return False, "voltage_unstable"
    except Exception:  # noqa: BLE001 - any failure => treat base as infeasible
        return False, "diverged"
    return (len(log) == n_steps), None


def run_one_scenario(seed: int, capture: bool = False,
                     prescreen: bool = True) -> Dict[str, Any]:
    """Run all five variants on the scenario defined by ``seed`` (drop-and-replace
    unit; module-level so it is picklable for ``ProcessPoolExecutor``).

    When ``prescreen`` (default), a cheap V1 base-case probe runs first; if it
    fails the scenario is recorded as ``failing_variant="base_infeasible"`` and
    the five full runs are skipped (resample).

    Writes ``RUNS_DIR/scenario_<seed>.json`` (always) and ``ts_<seed>.npz`` (when
    accepted); when ``capture`` the per-variant runner output goes to
    ``log_<seed>.txt`` (keeps parallel console output readable).  Returns a small
    summary dict ``{seed, accepted, failing_variant, n_events}``.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    elements = _get_elements()
    sel_info = _get_sel_info()
    n_total_min = _get_n_total_min()

    rng = np.random.default_rng(seed)
    start_time = random_start_time(rng)
    schedule = build_random_schedule(rng, elements, n_total_min)

    metric_rows: List[Dict[str, Any]] = []
    vrms_by_variant: Dict[str, np.ndarray] = {}
    pq_by_variant: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    zv_by_variant: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    t_min_ref: Optional[np.ndarray] = None
    all_converged = True
    failing_variant: Optional[str] = None
    failing_reason: Optional[str] = None

    logf = open(os.path.join(RUNS_DIR, f"log_{seed}.txt"), "w",
                encoding="utf-8", errors="replace") if capture else None
    try:
        with contextlib.ExitStack() as stack:
            if logf is not None:
                stack.enter_context(contextlib.redirect_stdout(logf))
                stack.enter_context(contextlib.redirect_stderr(logf))
            print(f"=== scenario seed={seed}  start={start_time:%Y-%m-%d %H:%M}  "
                  f"{len(schedule)} events ===")

            # Base-case pre-screen: skip the five full runs if cos(phi)=1 cannot
            # even hold the base operating point at this start_time (resample).
            if prescreen:
                print("  -> pre-screen (V1 base case, no contingency, coarse) ...")
                pre_ok, pre_reason = _prescreen_base_case(start_time)
                if not pre_ok:
                    print(f"     base case INFEASIBLE ({pre_reason}) -> resample")
                    all_converged = False
                    failing_variant = "base_infeasible"
                    failing_reason = pre_reason
                else:
                    print("     base case OK")

            if failing_variant is None:
                for V in VARIANT_ORDER:
                    print(f"  -> running {V} ...")
                    log, conv, reason, margins = run_variant(V, start_time, schedule)
                    if not conv:
                        all_converged = False
                        failing_variant = V
                        failing_reason = reason
                        if margins:
                            _dump_mc_reach_margins(seed, V, margins)
                        del log
                        break
                    s = cigre_summary_table(
                        {V: log}, v_set=V_SET,
                        gen_s_rated_mva=_get_gen_srated(),
                    ).iloc[0]
                    metric_rows.append({
                        "variant": V, "converged": True,
                        **{m: float(s[m]) for m in METRIC_COLS},
                    })
                    t_min, vrms, pq, zv = extract_timeseries(log, sel_info)
                    t_min_ref = t_min
                    vrms_by_variant[V] = vrms
                    pq_by_variant[V] = pq
                    zv_by_variant[V] = zv
                    del log
    finally:
        if logf is not None:
            logf.close()

    res = {
        "seed": int(seed),
        "accepted": bool(all_converged),
        "start_time": start_time.isoformat(),
        "n_events": len(schedule),
        "failing_variant": failing_variant,
        "failing_reason": failing_reason,
        "schedule": schedule_to_rows(-1, seed, start_time, schedule),
        "metrics": metric_rows if all_converged else [],
    }
    with open(os.path.join(RUNS_DIR, f"scenario_{seed}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh)
    if all_converged and t_min_ref is not None:
        _write_ts_npz(os.path.join(RUNS_DIR, f"ts_{seed}.npz"),
                      t_min_ref, vrms_by_variant, pq_by_variant, zv_by_variant)

    return {"seed": int(seed), "accepted": bool(all_converged),
            "failing_variant": failing_variant, "failing_reason": failing_reason,
            "n_events": len(schedule)}


# ===========================================================================
#  Re-run a SINGLE variant on the already-accepted scenarios (in place)
# ===========================================================================
#  Use case: V1..V5 were all simulated once (expensive).  Afterwards only one
#  variant's *parameters* change (e.g. retuning V5 via ``VARIANTS["V5"]`` /
#  ``KAPPA_V5``).  Instead of re-running every variant on every scenario, this
#  path reloads the exact paired scenarios (``start_time`` + contingency
#  ``schedule``) that V1..V4 already saw from ``RUNS_DIR/scenario_<seed>.json``,
#  re-runs ONLY the chosen variant with the current parameters, and overwrites
#  that variant's slice of the per-seed staging (metrics row in the JSON, arrays
#  in ``ts_<seed>.npz``).  Every other variant is left byte-identical, so the
#  paired comparison is preserved.  ``collect_and_finalize`` then rebuilds the
#  master CSVs / canonical timeseries / tables / figures from the staging.
#
#  Divergence policy: a scenario was accepted because all variants (incl. the
#  *old* variant params) converged.  If the *new* params diverge on a scenario
#  we KEEP the scenario (V1..V4 untouched) but DROP that variant for that run,
#  rather than mixing two parameter sets in one column or discarding V1..V4.  The
#  variant therefore reports n <= N runs; the dropped count is summarised.


def _events_from_rows(rows: List[Dict[str, Any]]) -> List[ContingencyEvent]:
    """Reconstruct a contingency schedule from the per-seed JSON ``schedule``
    rows (the dicts produced by :func:`schedule_to_rows`).  This is the exact
    schedule V1..V4 ran on, so re-running a variant against it is a clean paired
    comparison -- it does not depend on regenerating the schedule from the seed
    (which would silently differ if the schedule-generation constants changed)."""
    out: List[ContingencyEvent] = []
    for r in rows:
        bus = r.get("bus", "")
        bus = None if (bus is None or bus == "") else int(bus)
        p = r.get("p_mw", "")
        p = float("nan") if (p is None or p == "") else float(p)
        q = r.get("q_mvar", "")
        q = float("nan") if (q is None or q == "") else float(q)
        out.append(ContingencyEvent(
            minute=int(r["minute"]), element_type=str(r["element_type"]),
            element_index=int(r["element_index"]), action=str(r["action"]),
            bus=bus, p_mw=p, q_mvar=q,
        ))
    out.sort(key=lambda e: (e.minute, e.element_type, e.action))
    return out


def _rerun_one_scenario_variant(seed: int, variant: str,
                                capture: bool = False) -> Dict[str, Any]:
    """Re-run a single ``variant`` on the accepted scenario ``seed``, in place.

    Reloads ``(start_time, schedule)`` from ``RUNS_DIR/scenario_<seed>.json``,
    runs only ``variant`` with the CURRENT ``VARIANTS[variant]`` parameters, and
    rewrites just that variant's slice of the JSON metrics and ``ts_<seed>.npz``.
    Module-level so it is picklable for ``ProcessPoolExecutor``.

    Returns ``{seed, variant, status, reason}`` with ``status`` in
    ``{"converged", "diverged", "skipped"}``.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    json_path = os.path.join(RUNS_DIR, f"scenario_{seed}.json")
    ts_path = os.path.join(RUNS_DIR, f"ts_{seed}.npz")

    try:
        with open(json_path, encoding="utf-8") as fh:
            j = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"seed": int(seed), "variant": variant, "status": "skipped",
                "reason": f"unreadable scenario json ({type(exc).__name__})"}

    if not j.get("accepted"):
        return {"seed": int(seed), "variant": variant, "status": "skipped",
                "reason": "scenario not accepted (no V1..V4 baseline)"}

    sel_info = _get_sel_info()
    start_time = datetime.fromisoformat(j["start_time"])
    schedule = _events_from_rows(j.get("schedule", []))

    logf = open(os.path.join(RUNS_DIR, f"rerun_{variant}_{seed}.txt"), "w",
                encoding="utf-8", errors="replace") if capture else None
    try:
        with contextlib.ExitStack() as stack:
            if logf is not None:
                stack.enter_context(contextlib.redirect_stdout(logf))
                stack.enter_context(contextlib.redirect_stderr(logf))
            print(f"=== rerun {variant}  seed={seed}  "
                  f"start={start_time:%Y-%m-%d %H:%M}  {len(schedule)} events ===")
            log, conv, reason, margins = run_variant(variant, start_time, schedule)
            if conv:
                s = cigre_summary_table(
                    {variant: log}, v_set=V_SET,
                    gen_s_rated_mva=_get_gen_srated(),
                ).iloc[0]
                new_metric = {"variant": variant, "converged": True,
                              **{m: float(s[m]) for m in METRIC_COLS}}
                t_min, vrms, pq, zv = extract_timeseries(log, sel_info)
            else:
                new_metric = None
                if margins:
                    _dump_mc_reach_margins(seed, variant, margins)
            del log
    finally:
        if logf is not None:
            logf.close()

    # ── Rewrite the JSON metrics: drop the old `variant` row, add the new one
    #    (if converged).  Keep V1..V4 (and any other variant) untouched. ──────
    j["metrics"] = [m for m in j.get("metrics", []) if m.get("variant") != variant]
    if conv and new_metric is not None:
        j["metrics"].append(new_metric)
    _ord = {V: i for i, V in enumerate(VARIANT_ORDER)}
    j["metrics"].sort(key=lambda m: _ord.get(m.get("variant"), 99))
    j.setdefault("rerun", {})[variant] = {
        "converged": bool(conv), "reason": reason,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(j, fh)

    # ── Rewrite ts_<seed>.npz: drop this variant's arrays, add the new ones. ──
    arrs: Dict[str, np.ndarray] = {}
    if os.path.isfile(ts_path):
        with np.load(ts_path, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
    # Drop every key belonging to `variant` (keys are "<kind>__<variant>__...";
    # "t_min" has no "__" and is always kept).
    arrs = {k: v for k, v in arrs.items()
            if "__" not in k or k.split("__")[1] != variant}
    if conv:
        arrs.setdefault("t_min", t_min)
        arrs[f"vrms__{variant}"] = vrms
        for nm, (P, Q) in pq.items():
            arrs[f"P__{variant}__{nm}"] = P
            arrs[f"Q__{variant}__{nm}"] = Q
        for z_, stats in zv.items():
            for stat, arr in stats.items():
                arrs[f"Vz__{variant}__{z_}__{stat}"] = arr
    np.savez_compressed(ts_path, **arrs)

    return {"seed": int(seed), "variant": variant,
            "status": ("converged" if conv else "diverged"), "reason": reason}


def rerun_variant_only(variant: str, n_runs: int, jobs: int = 1,
                       backup: bool = True) -> None:
    """Re-simulate a single ``variant`` on the existing accepted scenarios and
    merge it back in, keeping all other variants' results as-is.

    Operates on the same set ``collect_and_finalize`` would keep: the first
    ``n_runs`` accepted scenarios by ascending seed (or all accepted when
    ``n_runs <= 0``).  Re-finalizes at the end so the master CSVs, canonical
    ``timeseries/run_XXXX.npz``, Table-3 and figures are rebuilt.
    """
    if variant not in VARIANTS:
        raise SystemExit(f"--rerun-variant: unknown variant {variant!r} "
                         f"(valid: {', '.join(VARIANT_ORDER)})")
    os.makedirs(RUNS_DIR, exist_ok=True)

    recs: List[Dict[str, Any]] = []
    for f in glob.glob(os.path.join(RUNS_DIR, "scenario_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except (OSError, ValueError):
            continue
    accepted = sorted((r for r in recs if r.get("accepted")),
                      key=lambda r: int(r["seed"]))
    chosen = accepted[:n_runs] if (n_runs and n_runs > 0) else accepted
    if not chosen:
        raise SystemExit(
            f"--rerun-variant: no accepted scenarios found in {RUNS_DIR}. "
            "Run the full batch first (the per-seed staging must exist).")
    seeds = [int(r["seed"]) for r in chosen]

    print("=" * 72)
    print(f"  RE-RUN VARIANT {variant} ONLY over {len(seeds)} accepted "
          f"scenario(s); all other variants kept as-is.")
    print(f"  Active {variant} overrides (edit VARIANTS[{variant!r}] / "
          f"KAPPA_V5 to change):")
    for k, v in VARIANTS[variant].items():
        print(f"      {k} = {v}")
    print("=" * 72)

    # ── Safety backup of the per-seed staging we are about to overwrite ───────
    if backup:
        bdir = os.path.join(
            OUT_ROOT, f"_runs_backup_{variant}_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(bdir, exist_ok=True)
        for s in seeds:
            for nm in (f"scenario_{s}.json", f"ts_{s}.npz"):
                src = os.path.join(RUNS_DIR, nm)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(bdir, nm))
        print(f"  [backup] pre-rerun staging copied to {bdir}")

    # Warm parent caches (validates the element/gen build before forking).
    _get_elements()
    _get_sel_info()
    _get_gen_srated()

    results: List[Dict[str, Any]] = []
    if jobs <= 1:
        print(f"[serial] re-running {variant} on {len(seeds)} scenarios ...")
        for s in seeds:
            r = _rerun_one_scenario_variant(s, variant, capture=False)
            results.append(r)
            print(f"  [seed {r['seed']}] {variant}: {r['status']}"
                  + (f" ({r['reason']})" if r.get("reason") else ""))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[parallel] {jobs} workers re-running {variant} "
              f"(per-worker output -> {RUNS_DIR}/rerun_{variant}_<seed>.txt) ...")
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_rerun_one_scenario_variant, s, variant, True): s
                    for s in seeds}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {"seed": int(s), "variant": variant,
                         "status": "skipped", "reason":
                         f"worker error: {type(exc).__name__}: {exc}"}
                results.append(r)
                print(f"  [seed {r['seed']}] {variant}: {r['status']}"
                      + (f" ({r['reason']})" if r.get("reason") else ""))

    n_conv = sum(r["status"] == "converged" for r in results)
    n_div = sum(r["status"] == "diverged" for r in results)
    n_skip = sum(r["status"] == "skipped" for r in results)
    print("\n" + "#" * 72)
    print(f"  {variant} re-run complete: {n_conv} converged, "
          f"{n_div} diverged (variant dropped for those runs), {n_skip} skipped.")
    if n_div:
        print(f"  -> {variant} now has {n_conv}/{len(seeds)} runs; other "
              "variants unchanged.")
    print("#" * 72)

    collect_and_finalize(n_runs)


# ===========================================================================
#  Orchestrator: collect N paired-valid scenarios (sequential or parallel)
# ===========================================================================


def run_monte_carlo(n_runs: int, base_seed: int, resume: bool, jobs: int = 1,
                    prescreen: bool = True) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(TS_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    if not resume:
        for f in glob.glob(os.path.join(RUNS_DIR, "*")):
            os.remove(f)
        for f in glob.glob(os.path.join(TS_DIR, "run_*.npz")):
            os.remove(f)
        for p in (SCHED_CSV, METRICS_CSV, FAILED_CSV):
            if os.path.isfile(p):
                os.remove(p)

    # Warm the parent caches (also validates the element/gen build before forking).
    _get_elements()
    _get_sel_info()

    tried_seeds, n_accepted = _scan_runs_dir()
    if resume and tried_seeds:
        print(f"[resume] {len(tried_seeds)} scenarios already computed, "
              f"{n_accepted} accepted.")

    attempt = 0
    max_attempts = n_runs * MAX_ATTEMPT_FACTOR + 25

    def _next_seeds(k: int) -> List[int]:
        nonlocal attempt
        out: List[int] = []
        while len(out) < k and attempt < max_attempts:
            s = base_seed + attempt
            attempt += 1
            if s in tried_seeds:
                continue
            tried_seeds.add(s)
            out.append(s)
        return out

    def _report(r: Dict[str, Any]) -> None:
        nonlocal n_accepted
        n_accepted += int(r["accepted"])
        tag = ("ACCEPT" if r["accepted"]
               else f"reject ({r['failing_variant']}"
                    + (f"/{r['failing_reason']}" if r.get("failing_reason") else "")
                    + ")")
        print(f"  [seed {r['seed']}] {tag}  ({r['n_events']} events)  "
              f"-> {n_accepted}/{n_runs} accepted")

    print(f"  base-case pre-screen: {'ON' if prescreen else 'OFF'}")
    if jobs <= 1:
        print(f"[serial] collecting {n_runs} paired-valid scenarios ...")
        while n_accepted < n_runs and attempt < max_attempts:
            seeds = _next_seeds(1)
            if not seeds:
                break
            _report(run_one_scenario(seeds[0], capture=False, prescreen=prescreen))
    else:
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        print(f"[parallel] {jobs} workers over scenarios, continuously fed "
              f"(per-worker output -> {RUNS_DIR}/log_<seed>.txt) ...")
        # Continuous-feed pool: keep exactly `jobs` scenarios in flight and
        # submit the next seed the instant any finishes.  Avoids a per-wave
        # barrier, which matters because task durations vary by ~100x
        # (base_infeasible pre-screen reject ~6 s vs full accept ~1000 s) and a
        # barrier would leave workers idle waiting for the slowest in a batch.
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            inflight: Dict[Any, int] = {
                ex.submit(run_one_scenario, s, True, prescreen): s
                for s in _next_seeds(jobs)
            }
            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    s = inflight.pop(fut)
                    try:
                        _report(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [seed {s}] worker error: "
                              f"{type(exc).__name__}: {exc}")
                    # Refill only while more accepted runs are still needed;
                    # once the target is met, let the in-flight tasks drain
                    # (the collector keeps the first n_runs accepts by seed).
                    if n_accepted < n_runs:
                        for ns in _next_seeds(1):
                            inflight[ex.submit(run_one_scenario, ns, True, prescreen)] = ns

    print("\n" + "#" * 72)
    if n_accepted < n_runs:
        print(f"  WARNING: collected only {n_accepted}/{n_runs} paired-valid runs "
              f"in {attempt} attempts (cap {max_attempts}). Re-run with --resume "
              f"or relax the schedule constraints.")
    else:
        print(f"  Collected >= {n_runs} paired-valid runs ({attempt} attempts).")
    print("#" * 72)

    collect_and_finalize(n_runs)


# ===========================================================================
#  Collector: per-seed staging -> master CSVs + canonical timeseries + outputs
# ===========================================================================


def collect_and_finalize(n_runs: int) -> None:
    """Build the master CSVs, canonical ``timeseries/run_XXXX.npz``, tables, and
    figures from the per-seed JSON/npz in RUNS_DIR.

    The first ``n_runs`` accepted scenarios *by ascending seed* are kept and
    numbered ``run_id = 0..n_runs-1`` (deterministic, independent of the order in
    which parallel workers finished).  ``n_runs <= 0`` keeps every accepted run.
    Falls back to aggregating any pre-existing master CSV/timeseries when RUNS_DIR
    has no staged scenarios (e.g. an old sequential result tree).
    """
    recs: List[Dict[str, Any]] = []
    for f in glob.glob(os.path.join(RUNS_DIR, "scenario_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except (OSError, ValueError):
            continue

    if not recs:
        print("  [collect] no staged scenarios in RUNS_DIR; "
              "aggregating existing CSV/timeseries if present.")
        aggregate_table3()
        make_aggregate_figures()
        return

    accepted = sorted((r for r in recs if r.get("accepted")),
                      key=lambda r: int(r["seed"]))
    failed = sorted((r for r in recs if not r.get("accepted")),
                    key=lambda r: int(r["seed"]))
    chosen = accepted[:n_runs] if n_runs and n_runs > 0 else accepted

    # Rebuild the canonical artifacts from scratch (idempotent).
    for p in (SCHED_CSV, METRICS_CSV, FAILED_CSV):
        if os.path.isfile(p):
            os.remove(p)
    for f in glob.glob(os.path.join(TS_DIR, "run_*.npz")):
        os.remove(f)
    os.makedirs(TS_DIR, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    sched_rows: List[Dict[str, Any]] = []
    for run_id, r in enumerate(chosen):
        seed = int(r["seed"])
        for mr in r["metrics"]:
            metric_rows.append({"run_id": run_id, "seed": seed, **mr})
        for sr in r["schedule"]:
            row = dict(sr)
            row["run_id"] = run_id
            sched_rows.append(row)
        src = os.path.join(RUNS_DIR, f"ts_{seed}.npz")
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(TS_DIR, f"run_{run_id:04d}.npz"))

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(METRICS_CSV, index=False)
    if sched_rows:
        pd.DataFrame(sched_rows).to_csv(SCHED_CSV, index=False)
    if failed:
        pd.DataFrame([{
            "seed": int(r["seed"]), "start_time": r["start_time"],
            "n_events": r["n_events"], "failing_variant": r["failing_variant"],
            "failing_reason": r.get("failing_reason"),
        } for r in failed]).to_csv(FAILED_CSV, index=False)

    print(f"\n[collect] finalized {len(chosen)} run(s) "
          f"({len(accepted)} accepted total, {len(failed)} rejected).")
    aggregate_table3()
    make_aggregate_figures()


# ===========================================================================
#  Aggregation -> Table 3 + ranking stability
# ===========================================================================


def aggregate_table3() -> None:
    """Build ``table3_distribution.csv`` + ``ranking_stability.csv`` from
    ``metrics_per_run.csv`` and echo a mean +/- std summary."""
    if not os.path.isfile(METRICS_CSV):
        print("  [aggregate] no metrics_per_run.csv -- nothing to aggregate")
        return
    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        print("  [aggregate] metrics_per_run.csv is empty")
        return
    n_runs = int(df["run_id"].nunique())
    # Tolerate older metrics_per_run.csv that predate a metric column (e.g.
    # m_bar_pu added 2026-06-04): aggregate only the columns actually present.
    metric_cols = [m for m in METRIC_COLS if m in df.columns]
    missing = [m for m in METRIC_COLS if m not in df.columns]
    if missing:
        print(f"  [aggregate] metrics_per_run.csv lacks {missing} "
              f"(re-run the batch to populate); skipping those columns.")

    # ── Per-variant distribution of each metric ──────────────────────────────
    dist_rows: List[Dict[str, Any]] = []
    for V in VARIANT_ORDER:
        sub = df[df["variant"] == V]
        if sub.empty:
            continue
        for m in metric_cols:
            vals = sub[m].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                dist_rows.append({"variant": V, "metric": m, "n": 0,
                                  "mean": np.nan, "std": np.nan, "median": np.nan,
                                  "q25": np.nan, "q75": np.nan,
                                  "min": np.nan, "max": np.nan})
                continue
            dist_rows.append({
                "variant": V, "metric": m, "n": int(vals.size),
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if vals.size > 1 else 0.0),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
            })
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(os.path.join(OUT_ROOT, "table3_distribution.csv"), index=False)

    # ── Ranking stability ────────────────────────────────────────────────────
    rank_rows: List[Dict[str, Any]] = []
    run_ids = sorted(df["run_id"].unique())
    for m in metric_cols:
        better_low = LOWER_BETTER[m]
        v4_best = 0
        total = 0
        rank_accum: Dict[str, List[int]] = {V: [] for V in VARIANT_ORDER}
        for rid in run_ids:
            sub = df[df["run_id"] == rid].set_index("variant")
            # "best among V1-V4"
            comp4 = {V: float(sub.loc[V, m]) for V in ("V1", "V2", "V3", "V4")
                     if V in sub.index and np.isfinite(sub.loc[V, m])}
            if len(comp4) == 4:
                total += 1
                best = (min(comp4, key=comp4.get) if better_low
                        else max(comp4, key=comp4.get))
                if best == "V4":
                    v4_best += 1
            # mean rank across all present variants
            comp = {V: float(sub.loc[V, m]) for V in VARIANT_ORDER
                    if V in sub.index and np.isfinite(sub.loc[V, m])}
            order = sorted(comp, key=comp.get, reverse=not better_low)
            for rank, V in enumerate(order, start=1):
                rank_accum[V].append(rank)
        row: Dict[str, Any] = {
            "metric": m, "direction": "lower" if better_low else "higher",
            "n_runs_full": total,
            "frac_V4_best_among_V1toV4": (v4_best / total) if total else np.nan,
        }
        for V in VARIANT_ORDER:
            row[f"mean_rank_{V}"] = (float(np.mean(rank_accum[V]))
                                     if rank_accum[V] else np.nan)
        rank_rows.append(row)
    ranking = pd.DataFrame(rank_rows)
    ranking.to_csv(os.path.join(OUT_ROOT, "ranking_stability.csv"), index=False)

    # ── Console summary (mean +/- std, wide) ─────────────────────────────────
    print(f"\n== Table 3 over {n_runs} paired-valid runs (mean +/- std) ==")
    wide = {}
    for V in VARIANT_ORDER:
        d = dist[dist["variant"] == V].set_index("metric")
        if d.empty:
            continue
        wide[V] = {m: f"{d.loc[m, 'mean']:.4g} +/- {d.loc[m, 'std']:.2g}"
                   for m in METRIC_COLS if m in d.index}
    print(pd.DataFrame(wide).to_string())
    print("\n== Ranking stability (fraction of runs V4 is best among V1-V4) ==")
    print(ranking[["metric", "direction", "n_runs_full",
                   "frac_V4_best_among_V1toV4"]].to_string(index=False))


# ===========================================================================
#  Aggregate figures
# ===========================================================================


def _load_all_timeseries() -> List[Dict[str, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(TS_DIR, "run_*.npz")))
    runs = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            runs.append({k: z[k] for k in z.files})
    return runs


def plot_voltage_band(runs: List[Dict[str, np.ndarray]]) -> None:
    """TS voltage tracking error across runs: per variant, the mean line with a
    +/-1 standard-deviation shaded band (one panel per variant)."""
    if not runs:
        print("  [fig] voltage band: no timeseries -- skipped")
        return
    import matplotlib.pyplot as plt
    from visualisation.plot_cigre import apply_cigre_style, CIGRE_PALETTE, _save_pdf

    apply_cigre_style()
    t_min = runs[0]["t_min"]
    variants = [V for V in VARIANT_ORDER if any(f"vrms__{V}" in r for r in runs)]
    if not variants:
        print("  [fig] voltage band: no vrms series -- skipped")
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(2.3 * len(variants) + 0.6, 2.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, V in zip(axes[0], variants):
        stack = np.vstack([r[f"vrms__{V}"] for r in runs if f"vrms__{V}" in r])
        stack = stack * 1e3  # p.u. -> mp.u.
        n = stack.shape[0]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
            sigma = np.nanstd(stack, axis=0, ddof=1 if n > 1 else 0)
        # RMS error is non-negative, so clip the lower band at 0.
        lo = np.clip(mean - sigma, 0.0, None)
        hi = mean + sigma
        col = CIGRE_PALETTE.get(V, "0.4")
        ax.fill_between(t_min, lo, hi, color=col, alpha=0.20, linewidth=0,
                        zorder=2, label="$\\pm1\\sigma$")
        ax.plot(t_min, mean, color=col, linewidth=1.3, zorder=4, label="mean")
        ax.set_title(f"{V}  (n={n})", loc="left")
        ax.set_xlabel("Time / min")
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.0)
    axes[0, 0].set_ylabel("$\\mathrm{RMS}\\,\\|\\Delta V\\|$ / mp.u.")
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9)
    fig.suptitle("TS voltage tracking error across runs (mean $\\pm\\,1\\sigma$)",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pdf(fig, "Fig_mc_voltage_band", [OUT_ROOT, PAPER_FIG_DIR], also_png_dir=OUT_ROOT)
    plt.close(fig)


def plot_capability_mc(runs: List[Dict[str, np.ndarray]],
                       sel_info: List[Dict[str, Any]],
                       excluded_variants: Tuple[str, ...] = ("V1", "V5"),
                       p_zero_tol_mw: float = 20.0) -> None:
    """Generator P-Q operating-point clouds pooled over all runs, against the
    Milano capability envelope.  Mirrors plot_cigre.Fig.4 but aggregates every
    run's operating points; excludes V1/V5 by default (as in 005)."""
    if not runs:
        print("  [fig] capability: no timeseries -- skipped")
        return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from core.actuator_bounds import GeneratorParameters, compute_generator_q_limits
    from visualisation.plot_cigre import apply_cigre_style, CIGRE_PALETTE, _save_pdf

    apply_cigre_style()
    variants = [V for V in VARIANT_ORDER if V not in set(excluded_variants)
                and any(any(k.startswith(f"P__{V}__") for k in r) for r in runs)]
    if not variants or not sel_info:
        print("  [fig] capability: no variants / gens -- skipped")
        return

    ncol = len(sel_info)
    fig, axes = plt.subplots(1, ncol, figsize=(2.55 * ncol - 0.5, 2.6), squeeze=False)
    for ax, gi in zip(axes[0], sel_info):
        gp = GeneratorParameters(
            s_rated_mva=gi["s_rated_mva"], p_max_mw=gi["p_max_mw"],
            xd_pu=gi["xd_pu"], i_f_max_pu=gi["i_f_max_pu"],
            beta=gi["beta"], q0_pu=gi["q0_pu"],
        )
        nm = str(gi["name"])
        z = gi["zone"]
        v = 1.03
        # Capability envelope (P-dependent Q limits).
        p_min = gi.get("p_min_mw", 0.0)
        p_sweep = np.linspace(p_min, gi["p_max_mw"], 300)
        q_lo = np.empty_like(p_sweep)
        q_hi = np.empty_like(p_sweep)
        for ip, ppv in enumerate(p_sweep):
            q_lo[ip], q_hi[ip] = compute_generator_q_limits(gp, p_mw=ppv, v_pu=v)
        ax.fill_between(p_sweep, q_lo, q_hi, color="0.85", alpha=0.6, zorder=1,
                        label="Capability")
        ax.plot(p_sweep, q_hi, color="0.35", linewidth=0.9, zorder=1)
        ax.plot(p_sweep, q_lo, color="0.35", linewidth=0.9, zorder=1)
        ax.axvline(gi["p_max_mw"], color="0.35", linewidth=0.6, linestyle="--",
                   alpha=0.6, zorder=1)

        # Pooled operating points per variant (all runs, all steps).
        for V in variants:
            pk, qk = f"P__{V}__{nm}", f"Q__{V}__{nm}"
            Ps, Qs = [], []
            for r in runs:
                if pk not in r or qk not in r:
                    continue
                P = r[pk]
                Q = r[qk]
                for pp_, qq_ in zip(P, Q):
                    if not (np.isfinite(pp_) and np.isfinite(qq_)):
                        continue
                    if abs(pp_) < p_zero_tol_mw:
                        continue
                    qmn, qmx = compute_generator_q_limits(gp, p_mw=pp_, v_pu=v)
                    if qq_ < qmn or qq_ > qmx:
                        continue
                    Ps.append(pp_)
                    Qs.append(qq_)
            if Ps:
                ax.scatter(Ps, Qs, color=CIGRE_PALETTE.get(V, "0.5"), alpha=0.18,
                           s=6, zorder=3, edgecolors="none", label=V)
        ax.axhline(0.0, color="black", linewidth=0.3, linestyle=":")
        ax.set_title("%s ($\\mathcal{A}_{\\mathrm{T,%d}}$)" %
                     (nm.replace("_bus", "@B"), z), loc="left")
        ax.set_xlabel("$P$ / MW")
        ax.set_ylabel("$Q$ / Mvar")

    handles = [Line2D([], [], marker="o", linestyle="none",
                      color=CIGRE_PALETTE.get(V, "0.5"), markersize=5, alpha=0.7)
               for V in variants]
    handles.append(Line2D([], [], color="0.6", linewidth=6, alpha=0.6))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(handles, list(variants) + ["Capability"], loc="lower center",
               ncol=len(variants) + 1, frameon=True, bbox_to_anchor=(0.5, 0.01))
    _save_pdf(fig, "Fig_mc_capability", [OUT_ROOT, PAPER_FIG_DIR], also_png_dir=OUT_ROOT)
    plt.close(fig)


def plot_table3_boxplots(metrics: Optional[List[str]] = None) -> None:
    """Box plots of selected Table-3 metrics across variants, in a single row.

    ``metrics`` is a subset of ``METRIC_COLS`` (order preserved); defaults to the
    module-level ``BOX_METRICS`` (set by ``--box-metrics``).  One panel per
    metric, all in one row.
    """
    if not os.path.isfile(METRICS_CSV):
        return
    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        return
    sel = [m for m in (metrics if metrics else BOX_METRICS)
           if m in METRIC_COLS and m in df.columns]
    dropped = [m for m in (metrics if metrics else BOX_METRICS)
               if m in METRIC_COLS and m not in df.columns]
    if dropped:
        print(f"  [fig] box plot: {dropped} not in metrics_per_run.csv "
              f"(re-run the batch to populate); plotting the rest.")
    if not sel:
        print(f"  [fig] box plot: no valid metrics in {metrics or BOX_METRICS} "
              f"(valid: {METRIC_COLS})")
        return
    import matplotlib.pyplot as plt
    from visualisation.plot_cigre import apply_cigre_style, CIGRE_PALETTE, _save_pdf

    apply_cigre_style()
    variants = [V for V in VARIANT_ORDER if (df["variant"] == V).any()]
    n = len(sel)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 0.4, 2.7), squeeze=False)
    for idx, mname in enumerate(sel):
        ax = axes[0][idx]
        data = [df[df["variant"] == V][mname].to_numpy(dtype=float) for V in variants]
        data = [d[np.isfinite(d)] for d in data]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black"))
        ax.set_xticks(range(1, len(variants) + 1))
        ax.set_xticklabels(variants)
        for patch, V in zip(bp["boxes"], variants):
            patch.set_facecolor(CIGRE_PALETTE.get(V, "0.6"))
            patch.set_alpha(0.55)
        ax.set_title(METRIC_LABELS.get(mname, mname), loc="left", fontsize=8.5)
        ax.margins(y=0.05)
    n_runs = int(df["run_id"].nunique())
    fig.suptitle(f"Table-3 metrics across {n_runs} runs", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_pdf(fig, "Fig_mc_table3_box", [OUT_ROOT, PAPER_FIG_DIR], also_png_dir=OUT_ROOT)
    plt.close(fig)


def plot_voltage_zone_boxplots(runs: List[Dict[str, np.ndarray]]) -> None:
    """Per-zone TS bus-voltage range across the ensemble.

    Five variant groups (V1..V5), three boxes each (one per TS zone) -> 15 boxes.
    Each box pools the per-zone EHV bus-voltage envelope (min/mean/max per step)
    over all steps and all runs, so its whiskers show the voltage range the zone
    experiences under that variant.  Reads the ``Vz__{V}__{zone}__{stat}`` arrays
    written by :func:`_write_ts_npz`; skips with a hint if the timeseries predate
    that data (re-run the batch to populate).
    """
    if not runs:
        print("  [fig] voltage-zone box: no timeseries -- skipped")
        return
    # Discover (variant, zone) present from the npz keys "Vz__{V}__{zone}__{stat}".
    zones_set: set = set()
    vfound: set = set()
    for r in runs:
        for k in r:
            p = k.split("__")
            if len(p) == 4 and p[0] == "Vz":
                vfound.add(p[1])
                try:
                    zones_set.add(int(p[2]))
                except ValueError:
                    pass
    if not zones_set:
        print("  [fig] voltage-zone box: npz has no per-zone voltage "
              "(re-run the batch to populate) -- skipped")
        return
    zones = sorted(zones_set)
    variants = [V for V in VARIANT_ORDER if V in vfound]
    if not variants:
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from visualisation.plot_cigre import apply_cigre_style, _save_pdf

    apply_cigre_style()

    def _pool(V: str, z: int) -> np.ndarray:
        vals = []
        for r in runs:
            for stat in ("min", "mean", "max"):
                a = r.get(f"Vz__{V}__{z}__{stat}")
                if a is not None:
                    vals.append(np.asarray(a, dtype=np.float64))
        if not vals:
            return np.array([], dtype=np.float64)
        v = np.concatenate(vals)
        return v[np.isfinite(v)]

    zone_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    width = 0.8 / len(zones)
    data, positions, colors = [], [], []
    for gi, V in enumerate(variants):
        for zi, z in enumerate(zones):
            data.append(_pool(V, z))
            positions.append(gi + (zi - (len(zones) - 1) / 2.0) * width)
            colors.append(zone_colors[zi % len(zone_colors)])

    fig, ax = plt.subplots(figsize=(1.5 * len(variants) + 1.0, 3.0))
    # Whiskers = central-99.8% interval (0.1th/99.9th percentile) rather than the
    # default Tukey 1.5*IQR, so the whisker caps are a fixed, reportable quantile
    # of the pooled per-zone voltage envelope (matches the paper's V min/max table).
    bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                    whis=(0.1, 99.9),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(V_SET, color="black", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("TS bus voltage / p.u.")
    ax.margins(x=0.02)
    handles = [Patch(facecolor=zone_colors[zi % len(zone_colors)], alpha=0.6,
                     label=r"$\mathcal{A}_{\mathrm{T,%d}}$" % z)
               for zi, z in enumerate(zones)]
    handles.append(plt.Line2D([], [], color="black", linewidth=0.7,
                              linestyle="--", label=f"$V_{{\\mathrm{{ref}}}}={V_SET}$"))
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=True,
              ncol=len(zones) + 1)
    fig.suptitle(f"TS voltage range by zone across {len(runs)} runs",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pdf(fig, "Fig_mc_voltage_zone_box", [OUT_ROOT, PAPER_FIG_DIR],
              also_png_dir=OUT_ROOT)
    plt.close(fig)


def plot_paper_combined_box(runs: List[Dict[str, np.ndarray]]) -> None:
    """Final paper figure -- single row, three panels, width ratios 1:2:1.

    Panel 1: RMS dV TS / p.u. box plot over all variants (V1-V5).
    Panel 2: TS voltage range by zone (double width) -- per-zone EHV voltage
             boxes for V2-V5, three zones per group.
    Panel 3: gen Q utilisation box plot over V2-V4 only.

    Panels 1 and 3 read scalar metrics from ``metrics_per_run.csv``; panel 2
    reads per-zone voltage envelopes ``Vz__{V}__{zone}__{stat}`` from the
    timeseries npz (``runs``).  Skips gracefully if an input is missing.
    """
    if not os.path.isfile(METRICS_CSV):
        print("  [fig] paper combined: no metrics_per_run.csv -- skipped")
        return
    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        print("  [fig] paper combined: metrics_per_run.csv empty -- skipped")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from visualisation.plot_cigre import (apply_cigre_style, CIGRE_PALETTE,
                                          _save_pdf)
    apply_cigre_style()

    FS = 11  # Uniform font size for all text elements in this figure.

    # Per-zone voltage availability from the npz.
    zones_set: set = set()
    vfound: set = set()
    for r in runs:
        for k in r:
            p = k.split("__")
            if len(p) == 4 and p[0] == "Vz":
                vfound.add(p[1])
                try:
                    zones_set.add(int(p[2]))
                except ValueError:
                    pass
    zones = sorted(zones_set)

    def _scalar_box(ax, variants, col):
        data = [df[df["variant"] == V][col].to_numpy(dtype=float)
                for V in variants]
        data = [d[np.isfinite(d)] for d in data]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black"))
        for patch, V in zip(bp["boxes"], variants):
            patch.set_facecolor(CIGRE_PALETTE.get(V, "0.6"))
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(variants) + 1))
        ax.set_xticklabels(variants, fontsize=FS)
        ax.tick_params(axis="y", labelsize=FS)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(9.2, 2.7), gridspec_kw={"width_ratios": [1, 2, 1]})

    # ── Panel 1: RMS dV TS / p.u. (all variants) ──
    v1 = [V for V in VARIANT_ORDER if (df["variant"] == V).any()]
    _scalar_box(ax1, v1, "rms_v_ts_pu")
    ax1.set_title(r"RMS $\Delta V$ TS / p.u.", loc="left", fontsize=FS)

    # ── Panel 2: TS voltage range by zone (double width, V2-V5) ──
    v2 = [V for V in VARIANT_ORDER if V in vfound and V != "V1"]
    if zones and v2:
        zone_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
        width = 0.8 / len(zones)

        def _pool(V, z):
            vals = []
            for r in runs:
                for stat in ("min", "mean", "max"):
                    a = r.get(f"Vz__{V}__{z}__{stat}")
                    if a is not None:
                        vals.append(np.asarray(a, dtype=np.float64))
            if not vals:
                return np.array([], dtype=np.float64)
            v = np.concatenate(vals)
            return v[np.isfinite(v)]

        data2, pos2, col2 = [], [], []
        for gi, V in enumerate(v2):
            for zi, z in enumerate(zones):
                data2.append(_pool(V, z))
                pos2.append(gi + (zi - (len(zones) - 1) / 2.0) * width)
                col2.append(zone_colors[zi % len(zone_colors)])
        # Whiskers = central-99.8% interval (0.1th/99.9th percentile), see
        # plot_voltage_zone_boxplots; keeps this panel consistent with the
        # standalone voltage-range figure and the paper's V min/max table.
        bp2 = ax2.boxplot(data2, positions=pos2, widths=width * 0.9,
                          whis=(0.1, 99.9),
                          showfliers=False, patch_artist=True,
                          medianprops=dict(color="black"))
        for patch, c in zip(bp2["boxes"], col2):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax2.axhline(V_SET, color="black", linewidth=0.7, linestyle="--",
                    zorder=0)
        ax2.set_xticks(range(len(v2)))
        ax2.set_xticklabels(v2, fontsize=FS)
        ax2.tick_params(axis="y", labelsize=FS)
        handles = [Patch(facecolor=zone_colors[zi % len(zone_colors)],
                         alpha=0.6, label=r"$\mathcal{A}_{\mathrm{T,%d}}$" % z)
                   for zi, z in enumerate(zones)]
        leg = ax2.legend(handles=handles, loc="best", fontsize=FS, frameon=True,
                   ncol=len(zones))
        leg.get_frame().set_alpha(0.3)
    else:
        ax2.text(0.5, 0.5, "no per-zone voltage\n(re-run to populate)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=FS)
    ax2.set_title("TS voltage range by zone / p.u.", loc="left", fontsize=FS)

    # ── Panel 3: gen Q utilisation (V2-V4 only) ──
    v3 = [V for V in ("V2", "V3", "V4") if (df["variant"] == V).any()]
    _scalar_box(ax3, v3, "res_util")
    ax3.set_title("gen Q utilisation", loc="left", fontsize=FS)

    fig.tight_layout()
    _save_pdf(fig, "Fig_mc_paper_combined", [OUT_ROOT, PAPER_FIG_DIR],
              also_png_dir=OUT_ROOT)
    plt.close(fig)


def make_aggregate_figures() -> None:
    runs = _load_all_timeseries()
    sel_info = _gen_selection_info()
    for fn, args in (
        (plot_voltage_band, (runs,)),
        (plot_capability_mc, (runs, sel_info)),
        (plot_table3_boxplots, ()),
        (plot_voltage_zone_boxplots, (runs,)),
        (plot_paper_combined_box, (runs,)),
    ):
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - never lose the CSVs over a plot
            print(f"  [fig] {fn.__name__} failed: {type(exc).__name__}: {exc}")


# ===========================================================================
#  CLI
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte-Carlo CIGRE driver (006).")
    ap.add_argument("--runs", type=int, default=N_RUNS_DEFAULT,
                    help="number of paired-valid scenarios to collect")
    ap.add_argument("--seed", type=int, default=BASE_SEED_DEFAULT,
                    help="base seed; scenario seed = seed + attempt")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted batch (skip already-tried seeds)")
    ap.add_argument("--jobs", type=int, default=20,
                    help="parallel worker processes over scenarios. The work is "
                         "memory-bandwidth bound (sparse power flow), not CPU- or "
                         "solver-bound, so throughput plateaus ~2-2.3x and peaks "
                         "around 6 on this 8-core machine; it REGRESSES past ~8 "
                         "(jobs=10 was slower than jobs=6). 1 = serial.")
    ap.add_argument("--no-prescreen", action="store_true",
                    help="disable the cheap V1 base-case feasibility pre-screen "
                         "(by default, start_times where cos-phi=1 cannot hold the "
                         "base operating point are resampled before running the "
                         "five full variants)")
    ap.add_argument("--box-metrics", type=str, default=None,
                    help="comma-separated subset of the 6 Table-3 metrics to draw "
                         "in the one-row box-plot figure (default: "
                         f"{','.join(BOX_METRICS_DEFAULT)}). Valid: "
                         f"{','.join(METRIC_COLS)}. Honoured by --replot.")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild tables + figures from the per-seed staging "
                         "(RUNS_DIR) / existing CSVs only, no simulation")
    ap.add_argument("--rerun-variant", type=str, default=None, metavar="V",
                    help="re-simulate ONLY this variant (e.g. V5) on the already "
                         "accepted scenarios with the CURRENT VARIANTS[V] / "
                         "KAPPA_V5 parameters, keeping all other variants' results "
                         "as-is; then rebuild tables + figures. Requires the "
                         "per-seed staging from a prior full batch.")
    ap.add_argument("--no-backup", action="store_true",
                    help="with --rerun-variant, skip the safety copy of the "
                         "per-seed staging before overwriting it")
    args = ap.parse_args()

    if args.box_metrics:
        sel = [s.strip() for s in args.box_metrics.split(",") if s.strip()]
        bad = [s for s in sel if s not in METRIC_COLS]
        if bad:
            ap.error(f"--box-metrics unknown {bad}; valid: {METRIC_COLS}")
        global BOX_METRICS
        BOX_METRICS = sel

    os.makedirs(OUT_ROOT, exist_ok=True)
    if args.replot:
        print("=" * 72)
        print(f"  REPLOT from {OUT_ROOT}/ (no simulation)")
        print("=" * 72)
        collect_and_finalize(args.runs)
        print(f"\n[done] tables + figures rebuilt in {OUT_ROOT}/")
        return

    if args.rerun_variant:
        rerun_variant_only(args.rerun_variant, args.runs,
                           jobs=max(1, args.jobs), backup=not args.no_backup)
        print(f"\n[done] {args.rerun_variant} re-merged; outputs in {OUT_ROOT}/ "
              f"(and figures mirrored to {PAPER_FIG_DIR})")
        return

    run_monte_carlo(args.runs, args.seed, args.resume, jobs=max(1, args.jobs),
                    prescreen=not args.no_prescreen)
    print(f"\n[done] outputs in {OUT_ROOT}/ (and figures mirrored to {PAPER_FIG_DIR})")


if __name__ == "__main__":
    main()
