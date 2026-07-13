#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/013_SBX_LADDER.py
=============================
SBX Minimal Phase 7 — the mechanism DEMONSTRATION campaign
(plan v2 §4 Phase 7 as reframed by Manuel, 2026-07-07):

    "Primary acceptance = mechanism behaviour, generalising the smoke
    criteria [...]. Metrics table: mechanism-centric quantities with
    both Φ and violation exposure as descriptive columns — no
    acceptance gate on either. BME arm: optional context on one or two
    scenarios only, presented as 'for orientation', not as the
    benchmark SBX must beat."

Arms per scenario (identical net/scenario/seeds; ONLY the mechanism
differs):

* ``none``      — autonomous baseline,
* ``sbx_inert`` — contract pinning, no deals (need threshold
  unreachable): isolates the pinning price AND supplies the no-deal
  deviation sample for the tier-1 band calibration (finding F6 /
  D-P7-5),
* ``sbx``       — SBX Minimal, `q_band_mvar` calibrated per scenario
  from the inert arm's clean-cycle deviation (2 × RMS, integer-ceiled,
  floored at the §5 default) — config-level only, `sbx/` untouched,
* ``bme``       — OPTIONAL context (``--with-bme``), 011-calibrated
  constants, run only on scenarios listed in ``BME_CONTEXT_SCENARIOS``.
  Kept fully isolated: 011 constants are imported inside the branch, so
  this script survives a future removal of BME from the codebase.

Scenario families (constants below; timing identical to the Phase 5/6
smoke: contracts freeze at min 30, stress window min 60–210, unwind
tail to the horizon):

* ``asym_z3`` — the smoke-calibrated reference (500 Mvar sink, bus 15,
  zone 3, v_min 1.00). Exercises corridors (1,3)/(2,3), requester z3.
* ``asym_z1`` — zone-1 stress (border-actuator watch, finding F5:
  corridor (1,2) deals with AVR generators one trafo from terminals).
* ``asym_z2`` — zone-2 stress (F2 illustration: zone 3 offers (0,0),
  so (2,3) cannot deal; relief only via (1,2)).
* ``sym_z1z2`` — symmetric scarcity: zones 1+2 stressed in the same
  direction → opposite-sign requests on (1,2) → ScarcityEvents; zone 3
  supports neither (F2). Graceful-degradation demonstration.
* ``compl_z1z3`` — complementary needs: zone-1 OVERVOLTAGE (capacitive
  sink + tightened v_max) with zone-3 undervoltage → same-sign requests
  on (1,3) → MUTUAL (unpaid) deal demonstration; also exercises the
  D-P7-3 tier-2 payer rule commentary.

Mechanism-acceptance flags per scenario (generalised smoke criteria;
printed PASS/FAIL/n-a, recorded in the table, NOT process-fatal):

  M1 need flags fire only inside the stress window (and do fire);
  M2 family-expected outcome: asym → ≥ 1 unilateral paid deal with
     requester = the stressed zone; sym → ≥ 1 ScarcityEvent and no
     executed deal on the contested corridor; compl → ≥ 1 mutual deal;
  M3 supporters stay violation-free from first deal to trip (joint-box
     guarantee, C8);
  M4 q_sched settles (no mixed deal signs per corridor under stress);
  M5 unwind to zero within ceil(peak/quantum) + m_release + 1 cycles of
     the trip, references back at v_std at the end;
  M6 every TSO solve of every arm optimal / optimal_inaccurate;
  M7 settlement ran to completion (conservation is asserted inside
     sbx_h.settlement — reaching the end IS the pass) and the ledger is
     written.

Metrics table (``metrics.csv``, one row per scenario × arm):
mechanism counters (deals uni/mutual, scarcity, unwinds, exchanged
|ΔQ|, peak surplus, dust/cap rejections), relief decomposition
(exposure of the stressed zone: none / inert / sbx → pinning cost and
deal benefit), spillover per corridor (v2.2 item 6: q_meas(sbx) −
q_meas(sbx_inert) on no-deal corridors, stress cycles), settlement
(netting, tier-2 EUR, tier-3 count, UNATTRIBUTED, per-area payments,
calibrated band), capability health (t per area: median/min — F2
visible), consistency counts, and the descriptive columns Φ (uniform
metric, ``record_bme_phi``) and violation exposure. Plots per plan §4:
q_sched vs q_meas with band shading, terminal voltages vs references,
need flags + surplus, cumulative payments.

CLI:
  --calibrate          120-min sbx-arm-only pass per scenario (Manuel's
                       calibration-horizon rule): prints need-flag /
                       deal summary so stress magnitudes can be locked.
  --run                full campaign (arms × scenarios, sequential).
  --evaluate           rebuild table/plots/report from stored pickles.
  --scenarios a,b,...  subset (default: all five).
  --minutes M          horizon (default 360; the full protocol arc).
  --with-bme           add the BME context arm on BME_CONTEXT_SCENARIOS.

Outputs: ``results/013_SBX_LADDER/<scenario>/`` (per-arm pickles,
settlement CSV/MD, plots) and ``results/013_SBX_LADDER/{metrics.csv,
REPORT.md}``.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 7)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import csv
import importlib
import math
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

RESULT_DIR = REPO / "results" / "013_SBX_LADDER"

# ── Shared timing (identical to the Phase 5/6 smoke) ───────────────────
WARMUP_MIN = 30.0        # SBX contracts freeze at the settled state (A7)
STRESS_ON_MIN = 60.0
STRESS_OFF_MIN = 150.0
DEFAULT_MINUTES = 360.0
CAL_MINUTES = 120.0      # calibration-horizon rule (Manuel 2026-07-03)

#: Tier-1 band calibration (F6 / D-P7-5): band = ceil(2 × RMS of the
#: inert arm's clean-cycle |q_meas − q_sched| deviation), floored at the
#: §5 default. Config-level per scenario; sbx/ modules untouched.
BAND_RMS_FACTOR = 2.0
BAND_FLOOR_MVAR = 5.0

#: Scenarios that carry the OPTIONAL BME context arm under --with-bme.
BME_CONTEXT_SCENARIOS = ("asym_z3",)

# ── Scenario definitions ───────────────────────────────────────────────
# stress: list of (bus, p_mw, q_mvar) dormant loads, connected at
# STRESS_ON_MIN and tripped at STRESS_OFF_MIN. Bound overrides create a
# persistent violation against the zone's own actuators (smoke-style
# calibration; asym_z3 magnitudes are the VALIDATED smoke values, the
# other scenarios' magnitudes are locked via --calibrate).
SCENARIOS: Dict[str, dict] = {
    "asym_z3": dict(
        family="asym", stressed_zones=(3,),
        stress=[(15, 0.0, 500.0)],
        zone_v_min={3: 1.00}, zone_v_max={},
        description="Smoke-calibrated reference: 500 Mvar inductive sink "
                    "at bus 15 (zone 3), zone-3 v_min = 1.00.",
    ),
    "asym_z1": dict(
        family="asym", stressed_zones=(1,),
        stress=[(27, 0.0, 400.0)],
        zone_v_min={1: 1.02}, zone_v_max={},
        description="Zone-1 stress (border-actuator watch F5): 400 Mvar "
                    "sink at bus 27, zone-1 v_min = 1.01.",
    ),
    "asym_z2": dict(
        family="asym", stressed_zones=(2,),
        stress=[(6, 0.0, 600.0)],
        zone_v_min={2: 1.01}, zone_v_max={},
        description="Zone-2 stress (F2 illustration: zone 3 cannot "
                    "support): 600 Mvar sink at bus 6, zone-2 "
                    "v_min = 1.01.",
    ),
    "sym_z1z2": dict(
        family="sym", stressed_zones=(1, 2),
        stress=[(27, 0.0, 600.0), (6, 0.0, 500.0)],
        zone_v_min={1: 1.01, 2: 1.01}, zone_v_max={},
        description="Symmetric scarcity: zones 1+2 stressed in the same "
                    "direction; opposite-sign requests on (1,2) expected "
                    "(ScarcityEvents), zone 3 supports neither (F2).",
    ),
    "compl_z1z3": dict(
        family="compl", stressed_zones=(1, 3),
        stress=[(27, 0.0, -400.0), (15, 0.0, 400.0)],
        zone_v_min={3: 1.00}, zone_v_max={1: 1.045},
        description="Complementary needs: zone-1 overvoltage (capacitive "
                    "sink at bus 27, v_max = 1.045) + zone-3 "
                    "undervoltage → mutual (unpaid) deal on (1,3).",
    ),
}

ARMS_STANDARD = ("none", "sbx_inert", "sbx")


# ───────────────────────────────────────────────────────────────────────
#  Configuration and runs
# ───────────────────────────────────────────────────────────────────────


def make_config(scenario: str, arm: str, minutes: float,
                q_band_mvar: Optional[float] = None):
    """Shared scenario; ONLY the coordination mechanism differs per arm."""
    spec = SCENARIOS[scenario]
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    # Identical controller/model path on all arms (sbx v1 requirements).
    cfg.enable_tie_coordination = False
    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.refresh_shared_jac_on_tso = False
    # Uniform Φ metric — DESCRIPTIVE column only (no acceptance gate);
    # remove these four lines if/when BME leaves the codebase.
    cfg.record_bme_phi = False
    cfg.bme_metric_w_band = 1.0e4
    cfg.bme_v_soft_min_pu = 1.02
    cfg.bme_v_soft_max_pu = 1.04

    if arm == "none":
        cfg.coordination_mode = "none"
    elif arm == "sbx":
        cfg.coordination_mode = "sbx"
        kwargs = dict(tso_period_s=float(cfg.tso_period_s))
        if q_band_mvar is not None:
            kwargs["q_band_mvar"] = float(q_band_mvar)
        cfg.sbx_config = SBXConfig(**kwargs)
    elif arm == "sbx_inert":
        # Contract pinning without deals: need threshold unreachable.
        cfg.coordination_mode = "sbx"
        cfg.sbx_config = SBXConfig(
            tso_period_s=float(cfg.tso_period_s),
            v_viol_threshold_pu=0.5,
        )
    elif arm == "bme":
        # OPTIONAL context arm ("for orientation", not a benchmark).
        # 011 constants imported HERE so the script has no top-level BME
        # dependency and survives a future BME removal.
        _011 = importlib.import_module("experiments.011_BME_LADDER")
        cfg.coordination_mode = "bme"
        cfg.bme_gradient_scale = float(_011.BME_GRADIENT_SCALE)
        cfg.bme_w_band = float(_011.BME_W_BAND)
        cfg.bme_epsilon_switch = float(_011.BME_EPSILON_SWITCH)
        cfg.bme_switch_cost_oltc = float(_011.BME_SWITCH_COST_OLTC)
        cfg.bme_switch_cost_shunt = float(_011.BME_SWITCH_COST_SHUNT)
        cfg.bme_v_soft_min_pu = float(_011.BME_V_SOFT_MIN)
        cfg.bme_v_soft_max_pu = float(_011.BME_V_SOFT_MAX)
    else:
        rep1("unknown arm", arm=arm)

    cfg.sbx_warmup_s = 60.0 * WARMUP_MIN
    if spec["zone_v_min"]:
        cfg.zone_v_min_pu = dict(spec["zone_v_min"])
    if spec["zone_v_max"]:
        cfg.zone_v_max_pu = dict(spec["zone_v_max"])
    cfg.contingencies = []
    for bus, p_mw, q_mvar in spec["stress"]:
        cfg.contingencies += [
            ContingencyEvent(minute=STRESS_ON_MIN, element_type="load",
                             bus=bus, p_mw=p_mw, q_mvar=q_mvar,
                             action="connect"),
            ContingencyEvent(minute=STRESS_OFF_MIN, element_type="load",
                             bus=bus, p_mw=p_mw, q_mvar=q_mvar,
                             action="trip"),
            ContingencyEvent(minute=210, element_type="gen", element_index=2, action="trip"),
            ContingencyEvent(minute=270, element_type="gen", element_index=2, action="restore"),
        ]
    return cfg


def extract_sbx_runtime(adapter) -> Optional[dict]:
    """Picklable extract of the adapter's scheduler state after a run."""
    if adapter is None:
        return None
    sched = adapter.scheduler
    return {
        "records": {k: list(v) for k, v in sched.records.items()},
        "settlements": {k: list(v) for k, v in sched.settlements.items()},
        "scarcity_events": list(sched.scarcity_events),
        "terminal_history": list(adapter.terminal_history),
        "border_actuators": list(adapter.border_actuators),
        "contracts": dict(sched.contracts),
        "config": adapter.config,
        "corridor_keys": sorted(sched.corridors.keys()),
        "corridors_of_area": {z: list(v) for z, v
                              in sched.corridors_of_area.items()},
        "area_ids": list(sched.area_ids),
    }


def run_arm(scenario: str, arm: str, minutes: float,
            q_band_mvar: Optional[float] = None):
    cfg = make_config(scenario, arm, minutes, q_band_mvar)
    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    wall = time.perf_counter() - t0
    print(f"  [{scenario}/{arm}] {len(recs)} steps in {wall:.0f} s wall")
    runtime = captured.get("sbx_runtime") or {}
    adapter = runtime.get("adapter")
    # Settlement outputs while the engines are alive (M7).
    if arm in ("sbx", "sbx_inert") and adapter is not None:
        from sbx_h.settlement import write_settlement_outputs
        sdir = RESULT_DIR / scenario
        sdir.mkdir(parents=True, exist_ok=True)
        write_settlement_outputs(adapter.scheduler.settlement_engines,
                                 sdir, f"{scenario}_{arm}")
    return cfg, recs, extract_sbx_runtime(adapter)


def calibrate_band_from_inert(sbx_extract: dict, minutes: float) -> float:
    """F6/D-P7-5: tier-1 band from the inert arm's clean-cycle deviation.

    Clean cycles = boundaries strictly before the stress onset plus the
    settled tail (≥ 4 cycles after the trip, unwind budget); the stress
    window itself is excluded — the band prices ORDINARY deviation, the
    stress-driven flow shift is exactly what tier 3 / the ΔP attribution
    must classify.
    """
    if sbx_extract is None:
        rep1("band calibration needs the sbx_inert extract")
    cfg = sbx_extract["config"]
    cyc_min = cfg.t_cycle_min
    on_c = (STRESS_ON_MIN - WARMUP_MIN) / cyc_min
    off_c = (STRESS_OFF_MIN - WARMUP_MIN) / cyc_min
    devs: List[float] = []
    for key, recs in sbx_extract["records"].items():
        for r in recs:
            if r.consistency == "n/a":
                continue
            # Elapsed cycle r.cycle covers (r.cycle−1, r.cycle] in cycle
            # units after the freeze.
            if on_c < r.cycle <= off_c + 4:
                continue
            devs.append(r.q_meas_mvar - r.q_sched_mvar)
    if not devs:
        rep1("no clean cycles available for the band calibration",
             minutes=minutes)
    rms = float(np.sqrt(np.mean(np.square(devs))))
    band = max(BAND_FLOOR_MVAR, float(math.ceil(BAND_RMS_FACTOR * rms)))
    print(f"  band calibration: {len(devs)} clean-cycle deviations, "
          f"RMS = {rms:.2f} Mvar -> q_band = {band:.0f} Mvar")
    return band


# ───────────────────────────────────────────────────────────────────────
#  Evaluation
# ───────────────────────────────────────────────────────────────────────


def violation_exposure(recs, zone: int, v_min: float, v_max: float,
                       t_lo_s: float, t_hi_s: float) -> Tuple[float, int]:
    """(Σ depth, count) of per-step bound violations in a window."""
    depth_sum, count = 0.0, 0
    for r in recs:
        if not (t_lo_s <= r.time_s <= t_hi_s) or zone not in r.zone_v_min:
            continue
        d = max(v_min - r.zone_v_min[zone], r.zone_v_max[zone] - v_max, 0.0)
        if d > 0.0:
            depth_sum += d
            count += 1
    return depth_sum, count


def phi_stats(recs) -> Tuple[float, float]:
    """(mean Φ, last-hour mean Φ) [MW]; NaN when Φ was not recorded."""
    vals = [(r.time_s, r.bme_phi_mw) for r in recs
            if getattr(r, "bme_phi_mw", None) is not None]
    if not vals:
        return float("nan"), float("nan")
    t_end = vals[-1][0]
    all_phi = [p for _, p in vals]
    last = [p for t, p in vals if t >= t_end - 3600.0]
    return float(np.mean(all_phi)), float(np.mean(last))


def zone_bounds(cfg, spec, zone: int) -> Tuple[float, float]:
    lo = spec["zone_v_min"].get(zone, cfg.v_min_pu)
    hi = spec["zone_v_max"].get(zone, cfg.v_max_pu)
    return lo, hi


def solver_all_optimal(recs) -> bool:
    for r in recs:
        for s in r.zone_tso_status.values():
            if s is not None and s not in ("optimal",
                                           "optimal_inaccurate"):
                return False
    return True


def evaluate_scenario(scenario: str, minutes: float,
                      arm_data: Dict[str, dict]) -> Tuple[List[dict], dict]:
    """Mechanism flags + metric rows for one scenario (all arms)."""
    spec = SCENARIOS[scenario]
    stressed = spec["stressed_zones"]
    lo_s, hi_s = 60.0 * STRESS_ON_MIN, 60.0 * STRESS_OFF_MIN

    sbx = arm_data["sbx"]["sbx"]
    inert = arm_data["sbx_inert"]["sbx"]
    cfg_ref = arm_data["sbx"]["cfg"]
    sbx_cfg = sbx["config"]
    cyc_s = 60.0 * sbx_cfg.t_cycle_min
    warm_s = 60.0 * WARMUP_MIN

    records = sbx["records"]
    keys = sbx["corridor_keys"]
    deal_cycles = {key: [r for r in records[key]
                         if r.deal.dq_deal_mvar != 0.0] for key in keys}
    all_deals = [r for rl in deal_cycles.values() for r in rl]
    deal_cycle_set = {r.cycle for r in all_deals}
    stress_cycles = set(range(
        int(math.ceil((lo_s - warm_s) / cyc_s)),
        int((hi_s - warm_s) // cyc_s) + 1))

    flags: Dict[str, Tuple[str, str]] = {}

    def flag(tag: str, ok: Optional[bool], detail: str) -> None:
        verdict = "n-a" if ok is None else ("PASS" if ok else "FAIL")
        flags[tag] = (verdict, detail)
        print(f"    {tag}: {verdict} — {detail}")

    # M1 — need flags fire only inside the stress window.
    pre_flags, in_flags = 0, 0
    for key in keys:
        for r in records[key]:
            t_r = warm_s + r.cycle * cyc_s
            flagged = r.need_a or r.need_b
            if flagged and t_r < lo_s:
                pre_flags += 1
            if flagged and lo_s <= t_r <= hi_s + 2 * cyc_s:
                in_flags += 1
    flag("M1", pre_flags == 0 and in_flags > 0,
         f"{pre_flags} pre-stress flag cycle(s), {in_flags} in-window")

    # M2 — family-expected outcome.
    uni = [r for r in all_deals if r.deal.kind == "unilateral"]
    mut = [r for r in all_deals if r.deal.kind == "mutual"]
    scarcity = sbx["scarcity_events"]
    if spec["family"] == "asym":
        ok = any(r.deal.paid and r.deal.requester in stressed for r in uni)
        flag("M2", ok, f"{len(uni)} unilateral / {len(mut)} mutual deals; "
             f"requesters {sorted({r.deal.requester for r in uni})}")
    elif spec["family"] == "sym":
        contested = (1, 2)
        dealt_contested = bool(deal_cycles.get(contested))
        flag("M2", len(scarcity) > 0 and not dealt_contested,
             f"{len(scarcity)} ScarcityEvent(s); contested corridor "
             f"{contested} executed {len(deal_cycles.get(contested, []))} "
             f"deal(s)")
    else:  # compl
        flag("M2", len(mut) > 0,
             f"{len(mut)} mutual deal(s), {len(uni)} unilateral, "
             f"{len(scarcity)} scarcity")

    # M3 — supporters violation-free from first deal to trip (C8).
    if deal_cycle_set:
        t_deal = warm_s + min(deal_cycle_set) * cyc_s
        supporters = [z for z in sbx["area_ids"] if z not in stressed]
        worst = {}
        for z in supporters:
            v_lo, v_hi = zone_bounds(cfg_ref, spec, z)
            d, _ = violation_exposure(arm_data["sbx"]["recs"], z,
                                      v_lo, v_hi, t_deal, hi_s)
            worst[z] = d
        flag("M3", all(v == 0.0 for v in worst.values()),
             "supporter exposure [pu·step]: "
             + ", ".join(f"z{z}={v:.4f}" for z, v in sorted(worst.items())))
    else:
        flag("M3", None, "no deals executed")

    # M4 — settling: constant deal sign per corridor under stress.
    mixed = []
    for key in keys:
        signs = {int(math.copysign(1, r.deal.dq_deal_mvar))
                 for r in deal_cycles[key] if r.cycle in stress_cycles}
        if len(signs) > 1:
            mixed.append(key)
    flag("M4", not mixed,
         f"mixed-sign corridors: {mixed}" if mixed
         else "deal signs constant per corridor under stress")

    # M5 — unwind on budget, refs at v_std at the end.
    trip_cycle = int((hi_s - warm_s) // cyc_s)
    unwind_ok, details = True, []
    for key in keys:
        recs_c = records[key]
        peak = max((abs(r.surplus_mvar) for r in recs_c), default=0.0)
        if peak == 0.0:
            continue
        quantum = sbx["contracts"][key].dq_quant_mvar
        budget = math.ceil(peak / quantum) + sbx_cfg.m_release + 1
        zero_after = [r.cycle for r in recs_c
                      if r.cycle > trip_cycle and r.surplus_mvar == 0.0]
        final = recs_c[-1]
        if not zero_after or zero_after[0] - trip_cycle > budget:
            unwind_ok = False
            details.append(f"{key}: peak {peak:.1f}, budget {budget}, "
                           f"zero at {zero_after[:1]}")
        if final.surplus_mvar != 0.0:
            unwind_ok = False
            details.append(f"{key}: final surplus "
                           f"{final.surplus_mvar:+.2f}")
    flag("M5", unwind_ok if deal_cycle_set else None,
         "; ".join(details) if details else "all surpluses unwound "
         "within budget; final surplus zero everywhere")

    # M6 — solver feasibility on every arm.
    bad_arms = [a for a, d in arm_data.items()
                if not solver_all_optimal(d["recs"])]
    flag("M6", not bad_arms,
         f"non-optimal solves in arms {bad_arms}" if bad_arms
         else "all TSO solves optimal on all arms")

    # M7 — settlement completed (conservation asserted inside sbx).
    n_settled = sum(len(v) for v in sbx["settlements"].values())
    flag("M7", n_settled > 0,
         f"{n_settled} settled corridor-cycles, ledger written")

    # ── Metric rows ────────────────────────────────────────────────────
    rows: List[dict] = []
    for arm, data in arm_data.items():
        recs = data["recs"]
        row: Dict[str, object] = dict(
            scenario=scenario, family=spec["family"], arm=arm,
            minutes=minutes,
        )
        phi_mean, phi_last = phi_stats(recs)
        row["phi_mean_mw"] = round(phi_mean, 3)
        row["phi_lasthour_mw"] = round(phi_last, 3)
        row["losses_mean_mw"] = round(float(np.mean(
            [r.total_losses_mw for r in recs])), 3)
        for z in (1, 2, 3):
            v_lo, v_hi = zone_bounds(cfg_ref, spec, z)
            d, n = violation_exposure(recs, z, v_lo, v_hi, lo_s, hi_s)
            row[f"expo_z{z}_pustep"] = round(d, 4)
            row[f"expo_z{z}_steps"] = n
        ext = data["sbx"]
        if ext is not None:
            deals_ext = [r for rl in ext["records"].values()
                         for r in rl if r.deal.dq_deal_mvar != 0.0]
            row["n_deals_uni"] = sum(
                1 for r in deals_ext if r.deal.kind == "unilateral")
            row["n_deals_mutual"] = sum(
                1 for r in deals_ext if r.deal.kind == "mutual")
            row["n_scarcity"] = len(ext["scarcity_events"])
            row["n_unwind_cycles"] = sum(
                1 for rl in ext["records"].values()
                for r in rl if r.unwound_mvar != 0.0)
            row["exchanged_abs_dq_mvar"] = round(sum(
                abs(r.deal.dq_deal_mvar) for r in deals_ext), 2)
            row["n_reject_dust"] = sum(
                1 for rl in ext["records"].values() for r in rl
                if r.deal.reason == "below_dust_threshold")
            row["n_reject_cap"] = sum(
                1 for rl in ext["records"].values() for r in rl
                if r.deal.reason == "contract_cap")
            row["peak_surplus_mvar"] = round(max(
                (abs(r.surplus_mvar) for rl in ext["records"].values()
                 for r in rl), default=0.0), 2)
            row["q_band_mvar"] = ext["config"].q_band_mvar
            t_by_area = {z: [] for z in ext["area_ids"]}
            for key, rl in ext["records"].items():
                za, zb = key
                for r in rl:
                    t_by_area[za].append(r.t_a)
                    t_by_area[zb].append(r.t_b)
            for z, ts in t_by_area.items():
                row[f"t_z{z}_median"] = round(float(np.median(ts)), 3) \
                    if ts else float("nan")
            cons: Dict[str, int] = {}
            for rl in ext["records"].values():
                for r in rl:
                    cons[r.consistency] = cons.get(r.consistency, 0) + 1
            row["consistency"] = ";".join(
                f"{k}:{v}" for k, v in sorted(cons.items()))
            row["netting_mvarh"] = round(sum(
                s.netting_mvarh for sl in ext["settlements"].values()
                for s in sl), 4)
            row["tier2_eur"] = round(sum(
                s.tier2_eur for sl in ext["settlements"].values()
                for s in sl), 2)
            row["tier3_n_charged"] = sum(
                1 for sl in ext["settlements"].values()
                for s in sl if s.tier3_eur > 0.0)
            row["n_unattributed"] = sum(
                1 for sl in ext["settlements"].values()
                for s in sl if s.attribution == "UNATTRIBUTED")
            pay: Dict[int, float] = {}
            for sl in ext["settlements"].values():
                for s in sl:
                    for z, x in s.payments_eur.items():
                        pay[z] = pay.get(z, 0.0) + x
            for z in sorted(pay):
                row[f"pay_z{z}_eur"] = round(pay[z], 2)
        rows.append(row)

    # Relief decomposition + spillover on the sbx row.
    z0 = stressed[0]
    v_lo, v_hi = zone_bounds(cfg_ref, spec, z0)
    d_none, _ = violation_exposure(arm_data["none"]["recs"], z0,
                                   v_lo, v_hi, lo_s, hi_s)
    d_inert, _ = violation_exposure(arm_data["sbx_inert"]["recs"], z0,
                                    v_lo, v_hi, lo_s, hi_s)
    d_sbx, _ = violation_exposure(arm_data["sbx"]["recs"], z0,
                                  v_lo, v_hi, lo_s, hi_s)
    spill = spillover(sbx, inert, stress_cycles)
    for row in rows:
        if row["arm"] == "sbx":
            row["pinning_cost_pustep"] = round(d_inert - d_none, 4)
            row["deal_benefit_pustep"] = round(d_inert - d_sbx, 4)
            for key, (mx, mn) in spill.items():
                tag = f"{key[0]}{key[1]}"
                row[f"spill_{tag}_max_mvar"] = mx
                row[f"spill_{tag}_mean_mvar"] = mn

    return rows, {"flags": flags, "spillover": spill,
                  "deal_cycles": {k: [r.cycle for r in v]
                                  for k, v in deal_cycles.items()}}


def spillover(sbx: dict, inert: dict,
              stress_cycles: set) -> Dict[Tuple[int, int],
                                          Tuple[float, float]]:
    """v2.2 item 6: q_meas(sbx) − q_meas(inert) on NO-DEAL corridors.

    Returns {corridor: (max |spill|, mean |spill|)} over stress cycles in
    which that corridor itself executed no deal (the shift is then the
    footprint of OTHER corridors' deals plus arm-identical noise).
    """
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for key in sbx["corridor_keys"]:
        sbx_by_c = {r.cycle: r for r in sbx["records"][key]}
        inert_by_c = {r.cycle: r for r in inert["records"][key]}
        vals = []
        for c in sorted(set(sbx_by_c) & set(inert_by_c)):
            if c not in stress_cycles:
                continue
            if sbx_by_c[c].deal.dq_deal_mvar != 0.0:
                continue
            vals.append(sbx_by_c[c].q_meas_mvar
                        - inert_by_c[c].q_meas_mvar)
        if vals:
            out[key] = (round(float(np.max(np.abs(vals))), 3),
                        round(float(np.mean(np.abs(vals))), 3))
        else:
            out[key] = (float("nan"), float("nan"))
    return out


# ───────────────────────────────────────────────────────────────────────
#  Plots (plan §4: q_sched vs q_meas with band, terminals vs refs,
#  need flags, cumulative payments)
# ───────────────────────────────────────────────────────────────────────


def make_plots(scenario: str, sbx: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cfg = sbx["config"]
    warm_min = WARMUP_MIN
    cyc_min = cfg.t_cycle_min
    keys = sbx["corridor_keys"]

    def t_of(cycle: int) -> float:
        return warm_min + cycle * cyc_min

    # P1 — q_sched vs q_meas vs q_std with band shading + deal markers.
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 2.8 * len(keys)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        rl = sbx["records"][key]
        t = [t_of(r.cycle) for r in rl]
        ax.fill_between(
            t, [r.q_sched_mvar - cfg.q_band_mvar for r in rl],
            [r.q_sched_mvar + cfg.q_band_mvar for r in rl],
            alpha=0.18, color="#4477aa", linewidth=0,
            label=f"band ±{cfg.q_band_mvar:.0f} Mvar")
        ax.plot(t, [r.q_std_mvar for r in rl], color="#999999",
                ls="--", lw=1.0, label="q_std")
        ax.plot(t, [r.q_sched_mvar for r in rl], color="#4477aa",
                lw=1.4, label="q_sched")
        ax.plot(t, [r.q_meas_mvar for r in rl], color="#cc6677",
                lw=1.2, label="q_meas")
        deals = [(t_of(r.cycle), r.q_sched_mvar) for r in rl
                 if r.deal.dq_deal_mvar != 0.0]
        if deals:
            ax.scatter(*zip(*deals), marker="v", color="#117733",
                       zorder=5, label="deal")
        ax.axvspan(STRESS_ON_MIN, STRESS_OFF_MIN, color="#f0efec",
                   zorder=0)
        ax.set_ylabel(f"({key[0]},{key[1]}) [Mvar]")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend(loc="best", fontsize=8, ncol=5)
    axes[-1].set_xlabel("time [min]")
    fig.suptitle(f"{scenario}: corridor schedule vs measurement (sbx)")
    fig.tight_layout()
    fig.savefig(out_dir / "P1_q_sched_vs_meas.png", dpi=150)
    plt.close(fig)

    # P2 — terminal voltages vs frozen references.
    hist = sbx["terminal_history"]
    buses = sorted(hist[0][1].keys()) if hist else []
    if buses:
        tso_min = 3.0  # one TSO tick = 3 min (Phase 0 semantics)
        fig, axes = plt.subplots(len(buses), 1,
                                 figsize=(9, 1.7 * len(buses)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        t = [warm_min + i * tso_min for i in range(len(hist))]
        for ax, bus in zip(axes, buses):
            ax.plot(t, [tv[bus] for _, tv, _ in hist], lw=1.0,
                    color="#cc6677", label="v_meas")
            ax.plot(t, [tr[bus] for _, _, tr in hist], lw=1.0,
                    color="#4477aa", ls="--", label="v_ref")
            ax.axvspan(STRESS_ON_MIN, STRESS_OFF_MIN, color="#f0efec",
                       zorder=0)
            ax.set_ylabel(f"bus {bus}", fontsize=8)
            ax.grid(alpha=0.3, lw=0.5)
        axes[0].legend(loc="best", fontsize=8, ncol=2)
        axes[-1].set_xlabel("time [min]")
        fig.suptitle(f"{scenario}: corridor terminals vs references (sbx)")
        fig.tight_layout()
        fig.savefig(out_dir / "P2_terminals_vs_refs.png", dpi=150)
        plt.close(fig)

    # P3 — need flags and surplus.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for key in keys:
        rl = sbx["records"][key]
        t = [t_of(r.cycle) for r in rl]
        ax1.plot(t, [r.surplus_mvar for r in rl], lw=1.3,
                 label=f"({key[0]},{key[1]})")
        ax2.step(t, [int(r.need_a) + 2 * int(r.need_b) for r in rl],
                 where="post", lw=1.0, label=f"({key[0]},{key[1]})")
    ax1.set_ylabel("surplus [Mvar]")
    ax2.set_ylabel("need flags (A=1, B=2, both=3)")
    ax2.set_xlabel("time [min]")
    for ax in (ax1, ax2):
        ax.axvspan(STRESS_ON_MIN, STRESS_OFF_MIN, color="#f0efec",
                   zorder=0)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8)
    fig.suptitle(f"{scenario}: surplus and need flags (sbx)")
    fig.tight_layout()
    fig.savefig(out_dir / "P3_need_surplus.png", dpi=150)
    plt.close(fig)

    # P4 — cumulative payments per area.
    pay_cum: Dict[int, List[float]] = {}
    t_pay: List[float] = []
    cycles = sorted({s.cycle for sl in sbx["settlements"].values()
                     for s in sl})
    running: Dict[int, float] = {}
    for c in cycles:
        for sl in sbx["settlements"].values():
            for s in sl:
                if s.cycle != c:
                    continue
                for z, x in s.payments_eur.items():
                    running[z] = running.get(z, 0.0) + x
        t_pay.append(t_of(c))
        for z in sbx["area_ids"]:
            pay_cum.setdefault(z, []).append(running.get(z, 0.0))
    if t_pay:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        for z, series in sorted(pay_cum.items()):
            ax.plot(t_pay, series, lw=1.4, label=f"zone {z}")
        ax.axvspan(STRESS_ON_MIN, STRESS_OFF_MIN, color="#f0efec",
                   zorder=0)
        ax.set_xlabel("time [min]")
        ax.set_ylabel("cumulative payments [EUR]")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8)
        fig.suptitle(f"{scenario}: cumulative settlement payments (sbx)")
        fig.tight_layout()
        fig.savefig(out_dir / "P4_payments.png", dpi=150)
        plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
#  Campaign driver
# ───────────────────────────────────────────────────────────────────────


def run_scenario(scenario: str, minutes: float, with_bme: bool) -> None:
    spec = SCENARIOS[scenario]
    sdir = RESULT_DIR / scenario
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== scenario {scenario} ({spec['family']}): "
          f"{spec['description']}")

    arm_payloads: Dict[str, dict] = {}
    band: Optional[float] = None
    arms = list(ARMS_STANDARD)
    if with_bme and scenario in BME_CONTEXT_SCENARIOS:
        arms.append("bme")
    for arm in arms:
        q_band = band if arm == "sbx" else None
        cfg, recs, ext = run_arm(scenario, arm, minutes, q_band)
        arm_payloads[arm] = {"cfg": cfg, "recs": recs, "sbx": ext}
        if arm == "sbx_inert":
            band = calibrate_band_from_inert(ext, minutes)
        with open(sdir / f"arm_{arm}.pkl", "wb") as fh:
            pickle.dump({"scenario": scenario, "arm": arm,
                         "minutes": minutes, "records": recs,
                         "sbx": ext, "q_band_mvar": q_band}, fh)


def load_scenario(scenario: str) -> Dict[str, dict]:
    sdir = RESULT_DIR / scenario
    out = {}
    for pkl in sorted(sdir.glob("arm_*.pkl")):
        with open(pkl, "rb") as fh:
            d = pickle.load(fh)
        # Rebuild the reference config lazily for bound lookups only.
        cfg = make_config(scenario, "none", d["minutes"])
        out[d["arm"]] = {"cfg": cfg, "recs": d["records"],
                         "sbx": d["sbx"], "minutes": d["minutes"]}
    if not out:
        rep1("no stored arms for scenario — run with --run first",
             scenario=scenario)
    return out


def evaluate_all(scenarios: List[str]) -> None:
    all_rows: List[dict] = []
    report: List[str] = [
        "# 013 SBX LADDER — Phase 7 demonstration report",
        "",
        f"Generated 2026-07-07. Framing (Manuel): mechanism demonstration; "
        f"Φ and violation exposure are DESCRIPTIVE columns, no acceptance "
        f"gate on either; `sbx_inert` isolates the pinning cost; BME is "
        f"orientation context only.",
        "",
    ]
    for scenario in scenarios:
        arm_data = load_scenario(scenario)
        minutes = arm_data["sbx"].get("minutes", DEFAULT_MINUTES)
        print(f"\n--- evaluating {scenario} ---")
        rows, extra = evaluate_scenario(scenario, minutes, arm_data)
        all_rows.extend(rows)
        make_plots(scenario, arm_data["sbx"]["sbx"],
                   RESULT_DIR / scenario)
        report.append(f"## {scenario} — {SCENARIOS[scenario]['family']}")
        report.append("")
        report.append(SCENARIOS[scenario]["description"])
        report.append("")
        report.append("| flag | verdict | detail |")
        report.append("|---|---|---|")
        for tag, (verdict, detail) in extra["flags"].items():
            report.append(f"| {tag} | {verdict} | {detail} |")
        report.append("")
        report.append(f"Deal cycles: {extra['deal_cycles']}; spillover "
                      f"(max, mean |Mvar|) on no-deal stress cycles: "
                      f"{extra['spillover']}")
        report.append("")

    # metrics.csv — union of all row keys, stable order.
    cols: List[str] = []
    for row in all_rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(RESULT_DIR / "metrics.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    (RESULT_DIR / "REPORT.md").write_text("\n".join(report),
                                          encoding="utf-8")
    print(f"\nmetrics: {RESULT_DIR / 'metrics.csv'}")
    print(f"report:  {RESULT_DIR / 'REPORT.md'}")


def calibrate(scenarios: List[str]) -> None:
    """120-min sbx-only pass per scenario: do the flags/deals fire?"""
    for scenario in scenarios:
        print(f"\n=== calibrate {scenario} ===")
        _cfg, _recs, ext = run_arm(scenario, "sbx", CAL_MINUTES)
        if ext is None:
            rep1("calibration run produced no SBX runtime",
                 scenario=scenario)
        for key, rl in ext["records"].items():
            deals = [(r.cycle, r.deal.kind, round(r.deal.dq_deal_mvar, 1))
                     for r in rl if r.deal.dq_deal_mvar != 0.0]
            needs = [(r.cycle, int(r.need_a), int(r.need_b)) for r in rl
                     if r.need_a or r.need_b]
            print(f"  {key}: needs {needs[:8]}{'...' if len(needs) > 8 else ''}"
                  f" deals {deals}")
        print(f"  scarcity: {len(ext['scarcity_events'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    ap.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    ap.add_argument("--with-bme", action="store_true")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            rep1("unknown scenario", scenario=s,
                 known=sorted(SCENARIOS.keys()))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        calibrate(scenarios)
    if args.run:
        for s in scenarios:
            run_scenario(s, args.minutes, args.with_bme)
    if args.run or args.evaluate:
        evaluate_all(scenarios)
    if not (args.run or args.calibrate or args.evaluate):
        print("nothing to do: pass --calibrate, --run and/or --evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
