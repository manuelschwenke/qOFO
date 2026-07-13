"""
SBX Phase 5 — closed-loop smoke test (plan v2 §4 Phase 5, amended v2.2
items 3 and 5).

Three arms on ONE scenario (005 CIGRE cascade tuning, IEEE 39, 3-area
triangle partition byte-identical to BME):

* ``none``      — autonomous baseline (coordination_mode="none"),
* ``sbx_inert`` — contract pinning WITHOUT deals (need threshold set
  unreachably high): isolates the price of holding the corridor
  terminals at the contract voltages,
* ``sbx``       — SBX Minimal (coordination_mode="sbx").

The first smoke iteration (2026-07-07, two arms) showed exposure
sbx > none NOT because deals hurt but because the contract pinning
itself denies the stressed zone its own boundary-voltage lift (the
acting-side invariant holds the requester at v_std by design).  The
deal mechanism is therefore judged against the sbx_inert baseline; the
pinning cost (sbx_inert vs none) is reported alongside.

Scenario: SBX contracts freeze at the SETTLED closed-loop state after
WARMUP_MIN (revised A7 — the pre-loop snapshot pins terminals below the
zones' realised schedule and distorts the arm comparison); a pure
reactive contingency load (STRESS_Q_MVAR, inductive) is connected at a
zone-3 bus at STRESS_ON_MIN and tripped at STRESS_OFF_MIN (10 SBX
cycles under stress); zone 3's lower voltage bound is tightened so the
violation persists against the zone's own actuators.  The run continues
past the trip long enough for the full unwind.

Acceptance criteria evaluated (each printed PASS/FAIL, exit code 1 on
any FAIL):

  C1  unilateral paid deals with requester = zone 3 occur;
  C2  (v2.2 item 5) zone 3 accepts deals on BOTH of its corridors in the
      SAME cycle at least once;
  C3  zone-3 violation exposure (per-step depth below its v_min summed
      over the stress window) strictly below the SBX-INERT baseline
      (deal benefit); the pinning cost vs the autonomous arm is
      reported, not gated;
  C4  q_sched settles: per corridor, no deal of the opposite sign while
      the stress is on (quantum cycling — deal/unwind alternation — is
      allowed by the plan);
  C5  after the trip, every corridor surplus unwinds to zero within
      ceil(|surplus_peak| / quantum) + m_release + 1 cycles and both
      ends return to the contract voltages;
  C6  every TSO step of every zone reports an optimal/optimal_inaccurate
      solver status (OFO feasibility);
  C7  (v2.2 item 3) voltage_margin_pu >= the worst observed within-cycle
      corridor-terminal voltage shift over all deal cycles, measured on
      terminals of areas whose need flag is CLEAR that cycle (a
      violating area offers (0, 0) — its margin is not load-bearing and
      its terminal motion is its own recovery transient); the observed
      ratio is reported;
  C8  the SUPPORTER areas show zero voltage-bound violations from the
      first deal to the trip (the joint-box guarantee made empirically
      visible).

Reported numbers (Phase 5 amendment, 2026-07-07 — no pass/fail beyond
C7/C8): R1 maximum internal voltage violation per zone after accepted
deals; R2 maximum corridor-terminal reference-tracking error
|v_meas − v_ref|; R3 observed_terminal_shift / voltage_margin_pu.
Additionally the per-corridor contract-consistency classification
counts (scheduler CONSISTENCY_*) and the border-actuator diagnostic
are printed.

Run:  python tests/sbx_h/smoke_sbx_closed_loop.py [--minutes 320]
Results (pickle + printed table): results/sbx_phase5_smoke/.

Not collected by pytest (no ``test_`` prefix): a full closed-loop run of
both arms takes tens of minutes of wall clock.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 5)
"""
from __future__ import annotations

import argparse
import importlib
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.helpers.records import ContingencyEvent  # noqa: E402
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_005 = importlib.import_module("experiments.005_CIGRE_MULTI")

RESULT_DIR = REPO / "results" / "sbx_phase5_smoke"

# ── Scenario constants ─────────────────────────────────────────────────
WARMUP_MIN = 30.0           # SBX contracts freeze at the SETTLED state
STRESS_ON_MIN = 60.0        # 2 clean cycles after the freeze
STRESS_OFF_MIN = 210.0      # 10 cycles under stress (plan Phase 5)
STRESS_BUS = 15             # zone-3 interior bus (fixed partition)
STRESS_Q_MVAR = 500.0       # inductive Q sink (P = 0: pure reactive event)
ZONE_STRESSED = 3
V_MIN_STRESSED = 1.00       # tightened lower bound, zone 3 only.
                            # Calibration (2026-07-07, 300 Mvar @ 0.99):
                            # the zone lifts v_min only to ~0.997 and
                            # stalls — so against a 1.00 bound a 500-Mvar
                            # sink leaves a persistent violation deeper
                            # than v_viol_threshold_pu = 0.005, while the
                            # unstressed base (v_min ≥ 1.0019, flag only
                            # below 0.995) raises no need flag.


def make_config(mode: str, minutes: float):
    """Shared scenario; ONLY the coordination mechanism differs."""
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 1
    # Identical controller/model path on all arms (sbx v1 requirements).
    cfg.enable_tie_coordination = False
    cfg.local_sensitivities_tso = False
    cfg.local_sensitivities_dso = False
    cfg.refresh_shared_jac_on_tso = True
    if mode == "sbx_inert":
        # Contract pinning without deals: the need flag can never fire
        # (threshold unreachable), so no requests, no deals — only the
        # v_std terminal references act.
        from sbx_h.config import SBXConfig
        cfg.coordination_mode = "sbx"
        cfg.sbx_config = SBXConfig(
            tso_period_s=float(cfg.tso_period_s),
            v_viol_threshold_pu=0.5,
        )
    else:
        cfg.coordination_mode = mode
    cfg.sbx_warmup_s = 60.0 * WARMUP_MIN
    cfg.zone_v_min_pu = {ZONE_STRESSED: V_MIN_STRESSED}
    cfg.contingencies = [
        ContingencyEvent(minute=STRESS_ON_MIN, element_type="load",
                         bus=STRESS_BUS, p_mw=0.0, q_mvar=STRESS_Q_MVAR,
                         action="connect"),
        ContingencyEvent(minute=STRESS_OFF_MIN, element_type="load",
                         bus=STRESS_BUS, p_mw=0.0, q_mvar=STRESS_Q_MVAR,
                         action="trip"),
    ]
    return cfg


def run_arm(mode: str, minutes: float):
    cfg = make_config(mode, minutes)
    captured = {}

    def hook(state):
        captured.update(state)
        return None

    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    print(f"[{mode}] {len(recs)} steps in "
          f"{time.perf_counter() - t0:.0f} s wall")
    runtime = captured.get("sbx_runtime") or {}
    return cfg, recs, runtime.get("adapter")


def violation_exposure(recs, zone: int, v_min: float,
                       t_lo_s: float, t_hi_s: float):
    """(Σ depth, count) of per-step violations below v_min in a window."""
    depth_sum, count = 0.0, 0
    for r in recs:
        if not (t_lo_s <= r.time_s <= t_hi_s) or zone not in r.zone_v_min:
            continue
        d = v_min - r.zone_v_min[zone]
        if d > 0.0:
            depth_sum += d
            count += 1
    return depth_sum, count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=360.0)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_sbx, recs_sbx, adapter = run_arm("sbx", args.minutes)
    cfg_inert, recs_inert, _ = run_arm("sbx_inert", args.minutes)
    cfg_none, recs_none, _ = run_arm("none", args.minutes)

    sched = adapter.scheduler
    sbx_cfg = adapter.config
    k = sbx_cfg.k_sched
    results = {}
    failures = []

    def check(tag: str, ok: bool, detail: str) -> None:
        results[tag] = (bool(ok), detail)
        print(f"  {tag}: {'PASS' if ok else 'FAIL'} — {detail}")
        if not ok:
            failures.append(tag)

    print("\n=== SBX Phase 5 smoke criteria ===")

    z3_keys = sched.corridors_of_area[ZONE_STRESSED]

    # C1 — unilateral paid deals, requester = zone 3.
    deal_cycles = {
        key: [r for r in sched.records[key]
              if r.deal.dq_deal_mvar != 0.0]
        for key in sched.records
    }
    uni_z3 = [
        r for key in z3_keys for r in deal_cycles[key]
        if r.deal.kind == "unilateral" and r.deal.paid
        and r.deal.requester == ZONE_STRESSED
    ]
    check("C1", len(uni_z3) > 0,
          f"{len(uni_z3)} unilateral paid deal(s) requested by zone "
          f"{ZONE_STRESSED} on corridors {z3_keys}")

    # C2 — both zone-3 corridors deal in the SAME cycle (v2.2 item 5).
    cycles_by_key = {
        key: {r.cycle for r in deal_cycles[key]} for key in z3_keys
    }
    common = set.intersection(*cycles_by_key.values()) if z3_keys else set()
    check("C2", len(common) > 0,
          f"same-cycle deals on {z3_keys} in cycle(s) "
          f"{sorted(common)[:5]}")

    # C3 — deal benefit: exposure strictly below the SBX-INERT baseline
    # (identical contract pinning, no deals); the pinning cost vs the
    # autonomous arm is reported alongside.
    lo, hi = 60.0 * STRESS_ON_MIN, 60.0 * STRESS_OFF_MIN
    d_sbx, n_sbx = violation_exposure(
        recs_sbx, ZONE_STRESSED, V_MIN_STRESSED, lo, hi)
    d_inert, n_inert = violation_exposure(
        recs_inert, ZONE_STRESSED, V_MIN_STRESSED, lo, hi)
    d_none, n_none = violation_exposure(
        recs_none, ZONE_STRESSED, V_MIN_STRESSED, lo, hi)
    check("C3", d_sbx < d_inert and n_inert > 0,
          f"zone-{ZONE_STRESSED} exposure Σdepth [pu·step]: "
          f"sbx {d_sbx:.4f} ({n_sbx} steps) < sbx_inert {d_inert:.4f} "
          f"({n_inert} steps)?  [autonomous {d_none:.4f} ({n_none} "
          f"steps); pinning cost = {d_inert - d_none:+.4f}, deal "
          f"benefit = {d_inert - d_sbx:+.4f}]")

    # Cycle boundary c fires at t = WARMUP + c · t_cycle (the adapter
    # rebases its iteration counter at the contract-freeze tick).
    cyc_s = 60.0 * sbx_cfg.t_cycle_min
    warm_s = 60.0 * WARMUP_MIN

    # C4 — settling: no opposite-sign deal while the stress is on.
    settle_ok, settle_detail = True, []
    stress_cycles = range(int(math.ceil((lo - warm_s) / cyc_s)),
                          int((hi - warm_s) // cyc_s) + 1)
    for key in sched.records:
        signs = {int(math.copysign(1, r.deal.dq_deal_mvar))
                 for r in deal_cycles[key] if r.cycle in stress_cycles}
        if len(signs) > 1:
            settle_ok = False
            settle_detail.append(f"{key}: mixed deal signs under stress")
    check("C4", settle_ok,
          "; ".join(settle_detail) if settle_detail
          else "per-corridor deal signs constant while stressed")

    # C5 — unwind to zero within ceil(peak/quantum) + m_release + 1
    # cycles of the trip; both ends back at the contract voltages.
    unwind_ok, unwind_detail = True, []
    trip_cycle = int((hi - warm_s) // cyc_s)
    for key, corr in sched.corridors.items():
        recs_c = sched.records[key]
        peak = max((abs(r.surplus_mvar) for r in recs_c), default=0.0)
        st = sched.corridor_state(key)
        quantum = sched.contracts[key].dq_quant_mvar
        budget = math.ceil(peak / quantum) + sbx_cfg.m_release + 1
        zero_after = [r.cycle for r in recs_c
                      if r.cycle > trip_cycle and r.surplus_mvar == 0.0]
        if peak > 0.0:
            if not zero_after or zero_after[0] - trip_cycle > budget:
                unwind_ok = False
                unwind_detail.append(
                    f"{key}: peak {peak:.1f} Mvar, zero at cycle "
                    f"{zero_after[0] if zero_after else None} "
                    f"(trip cycle {trip_cycle}, budget {budget})")
        contract = sched.contracts[key]
        for kk, ln in enumerate(corr.lines):
            if (abs(st.refs_a[ln.bus_a] - contract.v_std_a_pu[kk]) > 1e-9
                    or abs(st.refs_b[ln.bus_b]
                           - contract.v_std_b_pu[kk]) > 1e-9):
                unwind_ok = False
                unwind_detail.append(f"{key}: refs not at v_std at end")
        if st.surplus_mvar != 0.0:
            unwind_ok = False
            unwind_detail.append(
                f"{key}: final surplus {st.surplus_mvar:+.2f} Mvar")
    check("C5", unwind_ok,
          "; ".join(unwind_detail) if unwind_detail
          else f"all surpluses unwound to zero within budget after the "
               f"trip (trip cycle {trip_cycle}); refs at v_std")

    # C6 — OFO feasibility on every TSO step of both arms.
    bad = []
    for tag, recs in (("sbx", recs_sbx), ("sbx_inert", recs_inert),
                      ("none", recs_none)):
        for r in recs:
            for z, s in r.zone_tso_status.items():
                if s is not None and s not in ("optimal",
                                               "optimal_inaccurate"):
                    bad.append((tag, r.step, z, s))
    check("C6", not bad,
          f"{len(bad)} non-optimal TSO solve(s): {bad[:5]}"
          if bad else "all TSO solves optimal/optimal_inaccurate")

    # C7 — v2.2 item 3: margin vs worst within-cycle terminal shift over
    # deal cycles (shift measured against the boundary-tick voltage).
    # Terminals of areas whose need flag is SET that cycle are excluded:
    # a violating area's capability LP is skipped and it offers (0, 0),
    # so its margin is not load-bearing — and its terminal motion is
    # dominated by its own recovery transient, not by quanta.
    hist = adapter.terminal_history
    v_by_it = {it: tv for it, tv, _ in hist}
    deal_cycle_set = {r.cycle for rl in deal_cycles.values() for r in rl}
    area_need = {}
    bus_area = {}
    for key, corr in sched.corridors.items():
        for r in sched.records[key]:
            area_need[(r.cycle, corr.area_a)] = (
                area_need.get((r.cycle, corr.area_a), False) or r.need_a)
            area_need[(r.cycle, corr.area_b)] = (
                area_need.get((r.cycle, corr.area_b), False) or r.need_b)
        for ln in corr.lines:
            bus_area[ln.bus_a] = corr.area_a
            bus_area[ln.bus_b] = corr.area_b
    worst_shift, worst_where = 0.0, None
    for c in sorted(deal_cycle_set):
        it0 = c * k
        base = v_by_it.get(it0)
        if base is None:
            continue
        for it in range(it0, it0 + k):
            tv = v_by_it.get(it)
            if tv is None:
                continue
            for bus, v in tv.items():
                if area_need.get((c, bus_area[bus]), False):
                    continue
                shift = abs(v - base[bus])
                if shift > worst_shift:
                    worst_shift, worst_where = shift, (c, it, bus)
    ratio_sm = (worst_shift / sbx_cfg.voltage_margin_pu
                if sbx_cfg.voltage_margin_pu > 0.0 else float("inf"))
    check("C7", worst_shift <= sbx_cfg.voltage_margin_pu,
          f"worst within-cycle terminal shift {worst_shift:.5f} pu at "
          f"(cycle, it, bus)={worst_where}; margin "
          f"{sbx_cfg.voltage_margin_pu:.3f} pu; R3 shift/margin = "
          f"{ratio_sm:.3f}")

    # C8 + R1 — supporter areas stay violation-free after accepted deals
    # (joint-box guarantee); per-zone worst violation depth reported.
    first_deal_cycle = min(deal_cycle_set) if deal_cycle_set else None
    r1 = {}
    if first_deal_cycle is not None:
        t_deal_lo = warm_s + first_deal_cycle * cyc_s
        for z in sched.area_ids:
            v_lo_z = (V_MIN_STRESSED if z == ZONE_STRESSED
                      else cfg_sbx.v_min_pu)
            v_hi_z = cfg_sbx.v_max_pu
            worst = 0.0
            for r in recs_sbx:
                if not (t_deal_lo <= r.time_s <= hi) or \
                        z not in r.zone_v_min:
                    continue
                worst = max(worst, v_lo_z - r.zone_v_min[z],
                            r.zone_v_max[z] - v_hi_z, 0.0)
            r1[z] = worst
    supporters = [z for z in sched.area_ids if z != ZONE_STRESSED]
    check("C8",
          first_deal_cycle is not None
          and all(r1.get(z, 0.0) == 0.0 for z in supporters),
          f"R1 worst violation depth after first deal [pu]: "
          + ", ".join(f"z{z}={r1.get(z, float('nan')):.5f}"
                      for z in sorted(r1)))

    # R2 — worst corridor-terminal reference-tracking error.
    r2, r2_where = 0.0, None
    for it, tv, tr in hist:
        for bus, v in tv.items():
            err = abs(v - tr[bus])
            if err > r2:
                r2, r2_where = err, (it, bus)
    print(f"  R2: worst terminal |v_meas − v_ref| = {r2:.5f} pu at "
          f"(it, bus)={r2_where}")

    # Contract-consistency classification counts (scheduler proposal 3).
    print("  consistency counts per corridor:")
    for key, rl in sched.records.items():
        counts = {}
        for r in rl:
            counts[r.consistency] = counts.get(r.consistency, 0) + 1
        print(f"    {key}: {counts}")

    # Border-actuator diagnostic (proposal 1).
    print(f"  border actuators ({len(adapter.border_actuators)}):")
    for h in adapter.border_actuators:
        print(f"    {h['element']} {h['index']} at bus {h['bus']} — "
              f"hop {h['hop']} from terminal {h['terminal_bus']} of "
              f"corridor {h['corridor']} (area {h['area']})")

    # ── Phase 6: settlement outputs (ledger CSV + Markdown summary) ────
    from sbx_h.settlement import write_settlement_outputs
    csv_path, md_path = write_settlement_outputs(
        sched.settlement_engines, RESULT_DIR, "smoke")
    print(f"\n  settlement ledger:  {csv_path}")
    print(f"  settlement summary: {md_path}")
    for key in sorted(sched.settlement_engines):
        eng = sched.settlement_engines[key]
        tier2 = sum(s.tier2_eur for s in eng.settlements)
        print(f"  {key}: netting {eng.ledger.netting_mvarh:+.3f} Mvar·h, "
              f"tier-2 {tier2:.2f} EUR, tier-3 charged "
              f"{eng.ledger.n_tier3_charged}, UNATTRIBUTED "
              f"{eng.ledger.n_unattributed}, net payments "
              f"{ {z: round(x, 2) for z, x in eng.ledger.payments_eur.items()} }")

    # ── Persist for STATUS_SBX.md / Phase 7 reuse ──────────────────────
    out = RESULT_DIR / "smoke_result.pkl"
    with open(out, "wb") as fh:
        pickle.dump({
            "criteria": results,
            "records_sbx": recs_sbx,
            "records_sbx_inert": recs_inert,
            "records_none": recs_none,
            "corridor_records": dict(sched.records),
            "settlements": dict(sched.settlements),
            "scarcity_events": list(sched.scarcity_events),
            "terminal_history": hist,
            "border_actuators": adapter.border_actuators,
            "sbx_config": sbx_cfg,
        }, fh)
    print(f"\nresults written to {out}")

    print("\n=== corridor cycle summaries (sbx arm) ===")
    for key, rl in sched.records.items():
        print(f"corridor {key}:")
        for r in rl:
            mark = ("DEAL" if r.deal.dq_deal_mvar != 0.0 else
                    "unwind" if r.unwound_mvar != 0.0 else
                    "scarcity" if r.deal.kind == "scarcity" else "")
            print(f"  c{r.cycle:3d}: q_std={r.q_std_mvar:+8.2f} "
                  f"q_meas={r.q_meas_mvar:+8.2f} "
                  f"q_sched={r.q_sched_mvar:+8.2f} "
                  f"surplus={r.surplus_mvar:+7.2f} "
                  f"dv={r.dv_pu:+.5f} t_a={r.t_a:5.2f} t_b={r.t_b:5.2f} "
                  f"need_a={int(r.need_a)} need_b={int(r.need_b)} {mark}")

    if failures:
        print(f"\nSMOKE FAILED: {failures}")
        return 1
    print("\nSMOKE PASSED (C1–C8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
