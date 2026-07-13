#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/015_SBX_COMPARE.py
==============================
SBX-H v6 comparison: none vs contract vs contract + PLANNED SUPPORT.

History of this experiment (baselines preserved on disk):

* v4/v5 campaigns (2026-07-10/12, ``v4_baseline/`` and ``v5_baseline/``
  under the results directory) answered Manuel's standing question "is
  SBX-H useful over no explicit communication?" with the G1–G7
  findings: the CONTRACT layer (scheduled boundary voltages, priority-
  tracked) carries essentially all the value; the runtime deal layer
  was unverifiable (G3), physically marginal (G4) and never armed by an
  honest exhaustion test (G7).  The deal layer was removed on
  2026-07-12 (v6; ``docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md``).
* This v6 version evaluates what REMAINS plus the new planning-time
  product: support agreed IN ADVANCE — the supporters hold a RAISED
  boundary voltage during the anticipated stress window
  (``sbx_h.contract.with_planned_support``; Manuel 2026-07-12: "planned
  support could be agreed upon in advance e.g. by demanding a higher
  boundary voltage from the neighbour").

Cells (deficit levels; zone A = zone 3, v_min = 1.00, sink at bus 15
from minute 60 to the horizon end):

* D2 — 900 Mvar (deep; recovers under the contract in ~30 min),
* D1 — 500 Mvar (misdirected regime: pinning redirects A's reserves),
* D0 — 150 Mvar (self-manageable; non-inferiority check).

Arms per cell (identical scenario, only the mechanism differs):

* ``none``        - autonomous zones, no coordination.
* ``sbx``         - controller-intent terminal schedules from minute 0,
                    ordinary-weight tracking, metering and settlement.
* ``sbx_support`` - base SBX plus planned support: z1 and z2 hold
                    +2.5 mpu on their zone-3 corridor terminals during
                    minutes 60-120 (agreed in advance).

Decomposition per cell (zone-3 violation exposure over the stress
window): base-SBX effect = expo(none) - expo(sbx), expected near zero;
planned-support benefit = expo(sbx) - expo(sbx_support).

Flags V1-V6: indicator fires iff deficit; base SBX is control-equivalent
to none within tolerance; planned support does not hurt; escalation and
solver/settlement health are checked.

CLI:
  --run           run the cells (arms × cells, sequential)
  --evaluate      rebuild table/plots/report from stored pickles
  --cells a,b     subset of {D2, D1, D0}
  --minutes M     horizon (default 120, calibration-horizon rule)

Outputs: ``results/015_SBX_COMPARE/<cell>_v6/`` (per-arm pickles,
settlement CSV/MD, FIG_B mechanism) and
``results/015_SBX_COMPARE/{matrix_v6.csv, FIG_A_v6.png, REPORT_v6.md}``.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-12 (SBX-H v6)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import csv
import importlib
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.helpers.records import ContingencyEvent  # noqa: E402
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sbx_h.config import SBXConfig  # noqa: E402
from sbx_h.fail import rep1  # noqa: E402

_005 = importlib.import_module("experiments.005_CIGRE_MULTI")

RESULT_DIR = REPO / "results" / "015_SBX_COMPARE"

SBX_START_MIN = 0.0      # controller-intent contracts start immediately
STRESS_ON_MIN = 60.0
DEFAULT_MINUTES = 120.0  # calibration-horizon rule (Manuel 2026-07-03)

#: v6 experiment knobs (defaults are Manuel's live knobs — pinned).
SBX_KNOBS = dict(k_sched=2, n_need=2, release_threshold_pu=0.001,
                 escalation_cycles=4, w_track_factor=1.0)

#: Planned support (agreed in advance): the supporters' sides (end A of
#: both zone-3 corridors) held +2.5 mpu during the stress window.
SUPPORT_DV_PU = 0.0025
Z3_CORRIDORS = ((1, 3), (2, 3))

STRESSED_ZONE = 3
SUPPORTERS = (1, 2)
ZONES = (1, 2, 3)
Z3_V_MIN = 1.00

C_ARM = {"none": "#888888", "sbx": "#4477aa", "sbx_support": "#117733"}
C_BOUND = "#cc3311"
C_STRESS = "#f0efec"

CELLS: Dict[str, dict] = {
    "D2": dict(sink_mvar=900.0,
               label="deep deficit (900 Mvar)",
               expect="base SBX matches none; planned support is explicit"),
    "D1": dict(sink_mvar=500.0,
               label="misdirected deficit (500 Mvar)",
               expect="base SBX matches none; support effect is measured"),
    "D0": dict(sink_mvar=150.0,
               label="self-manageable (150 Mvar)",
               expect="base SBX matches none; no hidden control effect"),
}

ARMS = ("none", "sbx", "sbx_support")


# ───────────────────────────────────────────────────────────────────────
#  Configuration and runs
# ───────────────────────────────────────────────────────────────────────


def make_config(cell: str, arm: str, minutes: float):
    spec = CELLS[cell]
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    cfg.enable_tie_coordination = False
    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.refresh_shared_jac_on_tso = False
    cfg.record_bme_phi = False

    if arm == "none":
        cfg.coordination_mode = "none"
    elif arm in ("sbx", "sbx_support"):
        cfg.coordination_mode = "sbx_h"
        cfg.sbx_config = SBXConfig(tso_period_s=float(cfg.tso_period_s),
                                   **SBX_KNOBS)
        if arm == "sbx_support":
            window = (60.0 * STRESS_ON_MIN, 60.0 * minutes)
            cfg.sbx_support_intervals = {
                key: [(window[0], window[1], SUPPORT_DV_PU, 0.0)]
                for key in Z3_CORRIDORS
            }
    else:
        rep1("unknown arm", arm=arm)

    cfg.sbx_warmup_s = 60.0 * SBX_START_MIN
    cfg.zone_v_min_pu = {STRESSED_ZONE: Z3_V_MIN}
    cfg.contingencies = [
        ContingencyEvent(minute=STRESS_ON_MIN, element_type="load",
                         bus=15, p_mw=0.0, q_mvar=spec["sink_mvar"],
                         action="connect"),
    ]
    return cfg


def extract_sbx_runtime(adapter) -> Optional[dict]:
    """Picklable extract of the v6 scheduler state after a run."""
    if adapter is None:
        return None
    sched = adapter.scheduler
    return {
        "records": {k: list(v) for k, v in sched.records.items()},
        "settlements": {k: list(v) for k, v in sched.settlements.items()},
        "escalations": list(sched.escalations),
        "terminal_history": list(adapter.terminal_history),
        "contracts": dict(sched.contracts),
        "config": adapter.config,
        "corridor_keys": sorted(sched.corridors.keys()),
        "area_ids": list(sched.area_ids),
    }


def run_arm(cell: str, arm: str, minutes: float, out_dir: Path):
    cfg = make_config(cell, arm, minutes)
    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    wall = time.perf_counter() - t0
    print(f"  [{cell}/{arm}] {len(recs)} steps in {wall:.0f} s wall")
    runtime = captured.get("sbx_runtime") or {}
    adapter = runtime.get("adapter")
    if arm != "none" and adapter is not None:
        from sbx_h.settlement import write_settlement_outputs
        out_dir.mkdir(parents=True, exist_ok=True)
        write_settlement_outputs(adapter.scheduler.settlement_engines,
                                 out_dir, f"{cell}_{arm}")
    return cfg, recs, extract_sbx_runtime(adapter)


def run_cell(cell: str, minutes: float) -> None:
    spec = CELLS[cell]
    sdir = RESULT_DIR / f"{cell}_v6"
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== cell {cell}: {spec['label']} (expect: {spec['expect']})")
    for arm in ARMS:
        cfg, recs, ext = run_arm(cell, arm, minutes, sdir)
        with open(sdir / f"arm_{arm}.pkl", "wb") as fh:
            pickle.dump({"cell": cell, "arm": arm, "minutes": minutes,
                         "records": recs, "sbx": ext}, fh)


def load_cell(cell: str) -> Dict[str, dict]:
    sdir = RESULT_DIR / f"{cell}_v6"
    out: Dict[str, dict] = {}
    for arm in ARMS:
        pkl = sdir / f"arm_{arm}.pkl"
        if not pkl.exists():
            rep1("stored arm missing — run with --run first",
                 cell=cell, arm=arm, path=str(pkl))
        with open(pkl, "rb") as fh:
            d = pickle.load(fh)
        out[arm] = {"recs": d["records"], "sbx": d["sbx"],
                    "minutes": d["minutes"]}
    return out


# ───────────────────────────────────────────────────────────────────────
#  Metrics
# ───────────────────────────────────────────────────────────────────────


def violation_exposure(recs, zone: int, t_lo_s: float, t_hi_s: float,
                       v_min: float, v_max: float = 1.10
                       ) -> Tuple[float, int, float]:
    """(Σ depth [pu·step], violating steps, max depth [pu]) in window."""
    depth_sum, count, worst = 0.0, 0, 0.0
    for r in recs:
        if not (t_lo_s <= r.time_s <= t_hi_s) or zone not in r.zone_v_min:
            continue
        d = max(v_min - r.zone_v_min[zone],
                r.zone_v_max[zone] - v_max, 0.0)
        if d > 0.0:
            depth_sum += d
            count += 1
            worst = max(worst, d)
    return depth_sum, count, worst


def tie_import_shift_z3(recs, t_lo_s: float, t_hi_s: float) -> float:
    """Mean tie-flow-proxy shift into zone 3 during the window vs the
    pre-stress mean [Mvar] (F7 caveat: proxy identical across arms —
    differences and shifts are valid, absolute levels are not)."""
    pre_lo, pre_hi = 60.0 * SBX_START_MIN, 60.0 * STRESS_ON_MIN

    def mean_flow(lo: float, hi: float) -> float:
        tot, n = 0.0, 0
        for r in recs:
            if not (lo <= r.time_s <= hi):
                continue
            if not all(k in r.zone_tie_q_mvar for k in Z3_CORRIDORS):
                continue
            tot += sum(r.zone_tie_q_mvar[k] for k in Z3_CORRIDORS)
            n += 1
        return tot / n if n else float("nan")

    return mean_flow(t_lo_s, t_hi_s) - mean_flow(pre_lo, pre_hi)


def solver_all_optimal(recs) -> bool:
    for r in recs:
        for s in r.zone_tso_status.values():
            if s is not None and s not in ("optimal",
                                           "optimal_inaccurate"):
                return False
    return True


def voltage_tracking_rms_mpu(recs, t_lo_s: float, t_hi_s: float) -> float:
    """Mean recorded zone RMS error to the global voltage target [mpu]."""
    values = [
        float(value)
        for record in recs
        if t_lo_s <= record.time_s <= t_hi_s
        for value in record.zone_v_rms_err_pu.values()
    ]
    return 1e3 * float(np.mean(values)) if values else float("nan")


def tso_oltc_activity(recs, t_lo_s: float,
                      t_hi_s: float) -> Tuple[float, int]:
    """Total tap travel and zone-event count whose new sample is in-window."""
    travel = 0.0
    zone_events = 0
    for previous, current in zip(recs[:-1], recs[1:]):
        if not (t_lo_s <= current.time_s <= t_hi_s):
            continue
        zones = set(previous.zone_oltc_taps) | set(current.zone_oltc_taps)
        for zone in zones:
            before = np.asarray(previous.zone_oltc_taps.get(zone, []),
                                dtype=float)
            after = np.asarray(current.zone_oltc_taps.get(zone, []),
                               dtype=float)
            if before.shape != after.shape or before.size == 0:
                continue
            delta = after - before
            if np.any(np.abs(delta) > 0.0):
                zone_events += 1
                travel += float(np.sum(np.abs(delta)))
    return travel, zone_events


def base_control_equivalence(none_recs, sbx_recs) -> Tuple[float, float]:
    """Maximum voltage-statistic and TSO-tap differences between arms."""
    if len(none_recs) != len(sbx_recs):
        return float("inf"), float("inf")
    max_voltage = 0.0
    max_tap = 0.0
    for none, sbx in zip(none_recs, sbx_recs):
        if abs(float(none.time_s) - float(sbx.time_s)) > 1e-9:
            return float("inf"), float("inf")
        for field in ("zone_v_min", "zone_v_max", "zone_v_mean"):
            left = getattr(none, field)
            right = getattr(sbx, field)
            for zone in set(left) | set(right):
                max_voltage = max(
                    max_voltage,
                    abs(float(left[zone]) - float(right[zone])),
                )
        for zone in set(none.zone_oltc_taps) | set(sbx.zone_oltc_taps):
            left = np.asarray(none.zone_oltc_taps.get(zone, []), dtype=float)
            right = np.asarray(sbx.zone_oltc_taps.get(zone, []), dtype=float)
            if left.shape != right.shape:
                return float("inf"), float("inf")
            if left.size:
                max_tap = max(
                    max_tap, float(np.max(np.abs(left - right)))
                )
    return max_voltage, max_tap


def evaluate_cell(cell: str, arm_data: Dict[str, dict]
                  ) -> Tuple[List[dict], dict]:
    spec = CELLS[cell]
    minutes = arm_data["sbx"]["minutes"]
    lo_s, hi_s = 60.0 * STRESS_ON_MIN, 60.0 * minutes
    from_sbx_start_s = 60.0 * SBX_START_MIN
    deficit = cell != "D0"

    expo = {arm: violation_exposure(arm_data[arm]["recs"], STRESSED_ZONE,
                                    lo_s, hi_s, Z3_V_MIN)
            for arm in ARMS}
    base_sbx_effect = expo["none"][0] - expo["sbx"][0]
    support_benefit = expo["sbx"][0] - expo["sbx_support"][0]
    max_base_v_diff, max_base_tap_diff = base_control_equivalence(
        arm_data["none"]["recs"], arm_data["sbx"]["recs"]
    )

    flags: Dict[str, Tuple[str, str]] = {}

    def flag(tag: str, ok: Optional[bool], detail: str) -> None:
        verdict = "n-a" if ok is None else ("PASS" if ok else "FAIL")
        flags[tag] = (verdict, detail)
        print(f"    {tag}: {verdict} — {detail}")

    ext = arm_data["sbx"]["sbx"]
    needs = sorted({r.cycle for key in Z3_CORRIDORS
                    for r in ext["records"].get(key, []) if r.need_b})
    escalated = sorted({c for c, z in ext["escalations"]
                        if z == STRESSED_ZONE})

    # V1 — the violation indicator tracks the deficit.
    if deficit:
        flag("V1", len(needs) >= 1,
             f"z3 indicator set in {len(needs)} cycle(s) {needs[:6]}")
    else:
        flag("V1", len(needs) == 0,
             f"z3 indicator set in {len(needs)} cycle(s) (expected 0)")
    # V2 - controller-intent base SBX must be physically neutral.
    flag("V2", max_base_v_diff <= 1e-10 and max_base_tap_diff == 0.0,
         f"max voltage-statistic difference {max_base_v_diff:.3e} pu; "
         f"max TSO tap difference {max_base_tap_diff:.1f} step")
    # V3 — planned support must not hurt; its benefit is reported.
    flag("V3", support_benefit > -0.05,
         f"planned-support benefit {support_benefit:+.3f} pu·step "
         f"(sbx {expo['sbx'][0]:.3f} → support "
         f"{expo['sbx_support'][0]:.3f})")
    # V4 — the A4 escalation indicator (re-planning signal) fires on a
    # persistent indicator run.
    if cell == "D2":
        flag("V4", len(escalated) >= 1,
             f"escalation flagged at cycle(s) {escalated[:4]} "
             f"(> {ext['config'].escalation_cycles} flagged boundaries)")
    else:
        flag("V4", len(escalated) == 0,
             f"{len(escalated)} escalation(s) (expected 0)")
    # V5 — solver health on every arm.
    bad = [a for a in ARMS if not solver_all_optimal(arm_data[a]["recs"])]
    flag("V5", not bad, f"non-optimal solves in {bad}" if bad
         else "all TSO solves optimal on all arms")
    # V6 — settlement completed (conservation asserted inside).
    n_settled = sum(len(v) for v in ext["settlements"].values())
    flag("V6", n_settled > 0, f"{n_settled} settled corridor-cycles")

    rows: List[dict] = []
    for arm in ARMS:
        recs = arm_data[arm]["recs"]
        d, n, worst = expo[arm]
        oltc_travel, oltc_zone_events = tso_oltc_activity(
            recs, from_sbx_start_s, hi_s)
        row: Dict[str, object] = dict(
            cell=cell, label=spec["label"], arm=arm, minutes=minutes,
            expo_z3_pustep=round(d, 4), expo_z3_steps=n,
            expo_z3_maxdepth_mpu=round(1e3 * worst, 2),
            tie_import_shift_z3_mvar=round(
                tie_import_shift_z3(recs, lo_s, hi_s), 2),
            losses_mean_mw=round(float(np.mean(
                [r.total_losses_mw for r in recs])), 3),
            v_rms_from_sbx_start_mpu=round(
                voltage_tracking_rms_mpu(recs, from_sbx_start_s, hi_s), 3),
            tso_oltc_travel_from_sbx_start=round(oltc_travel, 1),
            tso_oltc_zone_events_from_sbx_start=oltc_zone_events,
            solver_ok=int(solver_all_optimal(recs)),
        )
        aext = arm_data[arm]["sbx"]
        if aext is not None:
            row["n_indicator_cycles"] = len(sorted(
                {r.cycle for key in Z3_CORRIDORS
                 for r in aext["records"].get(key, []) if r.need_b}))
            row["n_escalations_z3"] = len(
                [1 for _, z in aext["escalations"] if z == STRESSED_ZONE])
            row["n_beyond_band"] = sum(
                1 for rl in aext["records"].values() for r in rl
                if r.beyond_band)
            pay: Dict[int, float] = {}
            for sl in aext["settlements"].values():
                for s in sl:
                    for z, x in s.payments_eur.items():
                        pay[z] = pay.get(z, 0.0) + x
            for z in sorted(pay):
                row[f"pay_z{z}_eur"] = round(pay[z], 2)
        if arm == "sbx":
            row["base_sbx_effect_pustep"] = round(base_sbx_effect, 4)
        if arm == "sbx_support":
            row["support_benefit_pustep"] = round(support_benefit, 4)
        rows.append(row)

    return rows, {"flags": flags, "expo": {a: expo[a][0] for a in ARMS},
                  "needs": needs, "escalated": escalated,
                  "max_base_v_diff_pu": max_base_v_diff,
                  "max_base_tap_diff": max_base_tap_diff,
                  "window_s": (lo_s, hi_s)}


# ───────────────────────────────────────────────────────────────────────
#  Figures
# ───────────────────────────────────────────────────────────────────────


def _shade(ax, minutes: float) -> None:
    ax.axvspan(STRESS_ON_MIN, minutes, color=C_STRESS, zorder=0)
    ax.axvline(SBX_START_MIN, color="0.75", lw=0.7, ls=":")
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8)


def fig_cells(all_extra: Dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    cells = [c for c in CELLS if c in all_extra]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.26
    xs = np.arange(len(cells))
    for j, arm in enumerate(ARMS):
        vals = [all_extra[c]["expo"][arm] for c in cells]
        ax.bar(xs + (j - 1) * width, vals, width, color=C_ARM[arm],
               label=arm)
    y_top = max(max(e["expo"].values()) for e in all_extra.values())
    ax.set_ylim(0, 1.18 * max(y_top, 1e-6))
    for i, c in enumerate(cells):
        e = all_extra[c]
        ax.annotate(f"esc: {len(e['escalated'])}",
                    (xs[i], max(e["expo"].values())),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{CELLS[c]['label']}" for c in cells],
                       fontsize=8)
    ax.set_ylabel("zone-3 violation exposure [pu·step]")
    ax.set_title("SBX-H v6: none vs contract vs contract + planned "
                 "support (stress window)", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_mechanism(cell: str, arm_data: Dict[str, dict],
                  out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    minutes = arm_data["sbx"]["minutes"]
    sup_ext = arm_data["sbx_support"]["sbx"]
    cfg = sup_ext["config"]
    cyc_min = cfg.t_cycle_min

    fig, axes = plt.subplots(4, 1, figsize=(10, 10.5), sharex=True)

    # (a) zone-3 minimum bus voltage, all arms.
    ax = axes[0]
    for arm in ARMS:
        t = [r.time_s / 60.0 for r in arm_data[arm]["recs"]]
        v = [r.zone_v_min.get(STRESSED_ZONE, np.nan)
             for r in arm_data[arm]["recs"]]
        ax.plot(t, v, color=C_ARM[arm], lw=1.2, label=arm)
    ax.axhline(Z3_V_MIN, color=C_BOUND, lw=1.0, ls="--", label="v_min")
    _shade(ax, minutes)
    ax.set_ylabel("min V (zone 3) [pu]")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    ax.set_title(f"{cell}: {CELLS[cell]['label']}", fontsize=10,
                 loc="left")

    # (b)/(c) corridors: q_meas vs the (support-shifted) q_std ± band.
    for ax, key in zip(axes[1:3], Z3_CORRIDORS):
        for arm in ("sbx", "sbx_support"):
            rl = arm_data[arm]["sbx"]["records"][key]
            t = [SBX_START_MIN + r.cycle * cyc_min for r in rl]
            ax.plot(t, [r.q_meas_mvar for r in rl], color=C_ARM[arm],
                    lw=1.1, label=f"q_meas ({arm})")
        rl = sup_ext["records"][key]
        t = [SBX_START_MIN + r.cycle * cyc_min for r in rl]
        band = [r.q_band_mvar for r in rl]
        ax.fill_between(t,
                        [r.q_std_mvar - b for r, b in zip(rl, band)],
                        [r.q_std_mvar + b for r, b in zip(rl, band)],
                        alpha=0.15, color=C_ARM["sbx_support"], lw=0,
                        label="q_std ± band (support arm)")
        ax.plot(t, [r.q_std_mvar for r in rl],
                color=C_ARM["sbx_support"], ls="--", lw=1.0)
        _shade(ax, minutes)
        ax.set_ylabel(f"({key[0]},{key[1]}) [Mvar]")
        ax.legend(fontsize=7, frameon=False, ncol=3)

    # (d) support into zone 3 relative to the pre-stress mean, all arms.
    ax = axes[3]
    for arm in ARMS:
        recs = arm_data[arm]["recs"]
        pre = [sum(r.zone_tie_q_mvar[k] for k in Z3_CORRIDORS)
               for r in recs
               if 60.0 * SBX_START_MIN <= r.time_s <= 60.0 * STRESS_ON_MIN
               and all(k in r.zone_tie_q_mvar for k in Z3_CORRIDORS)]
        base = float(np.mean(pre)) if pre else 0.0
        t, q = [], []
        for r in recs:
            if not all(k in r.zone_tie_q_mvar for k in Z3_CORRIDORS):
                continue
            t.append(r.time_s / 60.0)
            q.append(sum(r.zone_tie_q_mvar[k] for k in Z3_CORRIDORS)
                     - base)
        ax.plot(t, q, color=C_ARM[arm], lw=1.2, label=arm)
    ax.axhline(0.0, color="0.6", lw=0.7)
    _shade(ax, minutes)
    ax.set_ylabel("ΔQ import into z3 [Mvar]")
    ax.set_xlabel("time [min]")
    ax.legend(fontsize=8, frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out_dir / "FIG_B_mechanism.png", dpi=160)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
#  Report / driver
# ───────────────────────────────────────────────────────────────────────


def write_report(all_rows: List[dict], all_extra: Dict[str, dict],
                 out_path: Path) -> None:
    rep = [
        "# 015 SBX COMPARE — SBX-H v6 (contract + planned support)",
        "",
        "Generated 2026-07-12.  The deal-layer campaigns and findings "
        "G1–G7 are preserved in `v4_baseline/` and `v5_baseline/`; the "
        "v6 mechanism keeps the contract layer, the attributed "
        "settlement and the A4 escalation indicator, and adds PLANNED "
        "SUPPORT as a schedule product (the neighbour holds "
        f"+{1e3 * SUPPORT_DV_PU:.1f} mpu on its corridor terminals "
        "during the anticipated stress window, agreed in advance).",
        "",
        "## Result: zone-3 violation exposure [pu·step] (stress window)",
        "",
        "| cell | none | sbx (base) | sbx_support | base-SBX effect "
        "| support benefit | escalations |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in CELLS:
        if c not in all_extra:
            continue
        e = all_extra[c]
        sup_row = next(r for r in all_rows
                       if r["cell"] == c and r["arm"] == "sbx_support")
        sbx_row = next(r for r in all_rows
                       if r["cell"] == c and r["arm"] == "sbx")
        rep.append(
            f"| {c} | {e['expo']['none']:.3f} | {e['expo']['sbx']:.3f} "
            f"| {e['expo']['sbx_support']:.3f} | "
            f"{sbx_row['base_sbx_effect_pustep']:+.3f} | "
            f"{sup_row['support_benefit_pustep']:+.3f} | "
            f"{len(e['escalated'])} |")
    rep.append("")
    for c in CELLS:
        if c not in all_extra:
            continue
        e = all_extra[c]
        rep.append(f"## {c} — {CELLS[c]['label']}")
        rep.append("")
        rep.append("| flag | verdict | detail |")
        rep.append("|---|---|---|")
        for tag, (verdict, detail) in e["flags"].items():
            rep.append(f"| {tag} | {verdict} | {detail} |")
        rep.append("")
    rep.append("Figures: `FIG_A_v6.png`, per-cell "
               "`<cell>_v6/FIG_B_mechanism.png`; metrics in "
               "`matrix_v6.csv`; settlement ledgers per cell directory.")
    out_path.write_text("\n".join(rep), encoding="utf-8")


def evaluate_all(cells: List[str]) -> None:
    all_rows: List[dict] = []
    all_extra: Dict[str, dict] = {}
    for cell in cells:
        arm_data = load_cell(cell)
        print(f"\n--- evaluating {cell} ---")
        rows, extra = evaluate_cell(cell, arm_data)
        all_rows.extend(rows)
        all_extra[cell] = extra
        fig_mechanism(cell, arm_data, RESULT_DIR / f"{cell}_v6")

    cols: List[str] = []
    for row in all_rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(RESULT_DIR / "matrix_v6.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    fig_cells(all_extra, RESULT_DIR / "FIG_A_v6.png")
    write_report(all_rows, all_extra, RESULT_DIR / "REPORT_v6.md")
    print(f"\nmatrix:  {RESULT_DIR / 'matrix_v6.csv'}")
    print(f"figure:  {RESULT_DIR / 'FIG_A_v6.png'}")
    print(f"report:  {RESULT_DIR / 'REPORT_v6.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SBX-H v6 comparison: none / contract / contract + "
                    "planned support.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--cells", type=str, default=",".join(CELLS))
    ap.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in CELLS:
            rep1("unknown cell", cell=c, known=sorted(CELLS.keys()))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.run:
        for c in cells:
            run_cell(c, args.minutes)
    if args.run or args.evaluate:
        evaluate_all(cells)
    if not (args.run or args.evaluate):
        print("nothing to do: pass --run and/or --evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
