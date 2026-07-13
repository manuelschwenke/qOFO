"""
018_SBXV_E1 — SBX-V parity + economics (build plan §9 Phase 5, E1)
==================================================================
CAIR baseline (``coordination_mode="none"``) vs SBX-V
(``coordination_mode="sbxv"``) on the fixed 3-area IEEE 39 partition of
the shared 005 scenario (byte-identical partition — regression R2 holds
because BOTH arms build through ``make_cigre_config()`` unchanged).

Arms
----
* ``none``          — CAIR baseline.
* ``sbxv_neutral``  — full wiring, unreachable need flag (regression
                      R1: dispatch byte-identical to ``none``).
* ``sbxv``          — active mechanism (v1 defaults, STATUS_SBXV.md).

Metrics (plan E1)
-----------------
Per arm: TS voltage-violation energy [pu·s] and duration [s] (zone
min/max depth beyond ``[v_min_pu, v_max_pu]``); minimum DER reserve
margin; DSO interface tracking error (mean/max, plus the commit-instant
subset — the Phase-4 deferred invariant check).  For the sbxv arm
additionally: Mvarh and payments per tier and direction (settlement
CSVs), request count / acceptance ratio / grant utilisation.

Outputs: ``results/018_SBXV_E1/`` — settlement CSVs
(``e1_windows/days/totals.csv``), ``e1_summary.json``, and a findings
printout.

Usage:  python experiments/018_SBXV_E1.py [minutes]   (default 360)

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 5)
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from sbx_v.config import SBXVConfig
from sbx_v.settlement import write_settlement_csv

_005 = importlib.import_module("experiments.005_CIGRE_MULTI")
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

RESULT_DIR = REPO / "results" / "018_SBXV_E1"

ARMS = ("none", "sbxv_neutral", "sbxv")


def make_config(arm: str, minutes: float):
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    cfg.enable_tie_coordination = False
    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.refresh_shared_jac_on_tso = False
    if arm == "none":
        cfg.coordination_mode = "none"
    elif arm == "sbxv_neutral":
        cfg.coordination_mode = "sbxv"
        cfg.sbxv_config = SBXVConfig(
            tso_period_s=float(cfg.tso_period_s),
            miqp_pricing_enabled=False,       # explicit R1 bypass
            v_dev_threshold_pu=0.5,           # unreachable need flag
        )
    elif arm == "sbxv":
        cfg.coordination_mode = "sbxv"
        # Default config = the AR 4141-4 preset Normalbereich
        # (V-D2 as revised 2026-07-10; 5 %/10 % of contracted P).
        cfg.sbxv_config = SBXVConfig(
            tso_period_s=float(cfg.tso_period_s))
    else:
        raise ValueError(arm)
    return cfg


def run_arm(arm: str, minutes: float):
    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    cfg = make_config(arm, minutes)
    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    wall = time.perf_counter() - t0
    print(f"  [{arm}] {len(recs)} steps in {wall:.0f} s wall")
    runtime = captured.get("sbxv_runtime") or {}
    adapter = runtime.get("adapter")
    final = adapter.finalise() if adapter is not None else None
    return cfg, recs, final


# ----------------------------------------------------------------------
#  Metrics
# ----------------------------------------------------------------------

def voltage_violation_metrics(cfg, recs) -> dict:
    """Zone-level violation energy [pu·s] and duration [s] beyond the
    permissible band (upper + lower, from the recorded zone extrema)."""
    energy = 0.0
    duration = 0.0
    for r in recs:
        depth = 0.0
        for z in r.zone_v_max:
            depth += max(0.0, r.zone_v_max[z] - cfg.v_max_pu)
            depth += max(0.0, cfg.v_min_pu - r.zone_v_min.get(z, 1.0))
        if depth > 0.0:
            energy += depth * cfg.dt_s
            duration += cfg.dt_s
    return {"viol_energy_pu_s": energy, "viol_duration_s": duration}


def reserve_margin_metric(recs) -> float:
    """Minimum normalised DER reserve margin over the run (NaN-safe)."""
    m = np.inf
    for r in recs:
        for arr in r.tso_der_q_reserve.values():
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size:
                m = min(m, float(a.min()))
    return m if np.isfinite(m) else float("nan")


def dso_tracking_metrics(cfg, recs) -> dict:
    """DSO interface |Q_set − Q_act| stats, plus the commit-instant
    subset (t = multiples of the 900-s window: the Phase-4 deferred
    tracking-error invariant — commit instants must not stand out)."""
    err_all, err_commit = [], []
    for r in recs:
        for dso_id, qs in r.dso_q_set_mvar.items():
            qa = r.dso_q_actual_mvar.get(dso_id)
            if qs is None or qa is None:
                continue
            e = abs(float(qs) - float(qa))
            err_all.append(e)
            if r.time_s % 900.0 < cfg.dt_s:       # first step of a window
                err_commit.append(e)
    def _stats(v):
        return ({"mean": float(np.mean(v)), "max": float(np.max(v))}
                if v else {"mean": float("nan"), "max": float("nan")})
    return {"dso_track_all": _stats(err_all),
            "dso_track_commit_instants": _stats(err_commit)}


def pipeline_metrics(final) -> dict:
    """Request count, acceptance ratio, grant utilisation (plan E1)."""
    if final is None:
        return {}
    events = [e for log in final["pipeline_logs"].values() for e in log]
    n_req = sum(1 for e in events if e[0] == "request")
    replies = [e for e in events if e[0] == "reply"]
    n_ok = sum(1 for e in replies if e[4] in ("ACCEPT", "PARTIAL"))
    grants = final["grant_records"]
    granted_mvarh = sum(g.q_grant_mvar * 0.25 *
                        (g.window_end - g.window_first) for g in grants)
    used_mvarh = 0.0
    if final["settlement"] is not None:
        used_mvarh = sum(
            r.e_avg_mvarh for r in final["settlement"].window_rows
            if r.case.startswith("8.2"))
    return {
        "n_requests": n_req,
        "acceptance_ratio": (n_ok / len(replies)) if replies else None,
        "n_grants": len(grants),
        "n_dropped_grants": len(final["dropped_grants"]),
        "granted_mvarh_beyond_band": granted_mvarh,
        "settled_avg_tier_mvarh": used_mvarh,
    }


def settlement_summary(final) -> dict:
    if final is None or final["settlement"] is None:
        return {}
    result = final["settlement"]
    per_tier = {}
    for t in result.totals:
        key = f"{t.area_id}:{t.direction.value}:{t.world}"
        per_tier[key] = {
            "energy_avg_eur": t.pay_energy_avg_eur,
            "energy_grenz_eur": t.pay_energy_grenz_eur,
            "cap_avg_eur": t.pay_cap_avg_eur,
            "cap_grenz_eur": t.pay_cap_grenz_eur,
            "total_eur": t.pay_total_eur,
        }
    cases = {}
    for r in result.window_rows:
        cases[r.case] = cases.get(r.case, 0) + 1
    return {"totals": per_tier, "case_counts": cases,
            "grand_total_eur": sum(t.pay_total_eur
                                   for t in result.totals)}


def assert_r1(recs_none, recs_neutral) -> int:
    n = 0
    assert len(recs_none) == len(recs_neutral), "R1: record count differs"
    for ra, rb in zip(recs_none, recs_neutral):
        for field in ("zone_q_pcc_set", "zone_q_der", "zone_v_gen",
                      "zone_oltc_taps"):
            da, db = getattr(ra, field), getattr(rb, field)
            for z in da:
                assert np.array_equal(np.asarray(da[z]),
                                      np.asarray(db[z])), \
                    f"R1 VIOLATION: {field} z{z} step {ra.step}"
                n += 1
    return n


def main(minutes: float = 360.0) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"E1: parity + economics, horizon {minutes:.0f} min")

    results = {}
    recs_by_arm = {}
    final_by_arm = {}
    for arm in ARMS:
        cfg, recs, final = run_arm(arm, minutes)
        recs_by_arm[arm] = recs
        final_by_arm[arm] = final
        m = {}
        m.update(voltage_violation_metrics(cfg, recs))
        m["min_reserve_margin"] = reserve_margin_metric(recs)
        m.update(dso_tracking_metrics(cfg, recs))
        if arm == "sbxv":
            m.update(pipeline_metrics(final))
            m["settlement"] = settlement_summary(final)
            m["bands_mvar"] = final.get("bands") if final else None
        results[arm] = m

    # R1 (plan §10) — re-run every phase.
    n = assert_r1(recs_by_arm["none"], recs_by_arm["sbxv_neutral"])
    results["r1_arrays_checked"] = n
    print(f"  R1 OK ({n} arrays byte-identical)")

    if final_by_arm["sbxv"] is not None \
            and final_by_arm["sbxv"]["settlement"] is not None:
        write_settlement_csv(final_by_arm["sbxv"]["settlement"],
                             str(RESULT_DIR / "e1"))
        print(f"  settlement CSVs -> {RESULT_DIR}/e1_*.csv")

    with open(RESULT_DIR / "e1_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  summary -> {RESULT_DIR}/e1_summary.json")

    # ── Findings printout ────────────────────────────────────────────
    print("\n=== E1 findings ===")
    for arm in ARMS:
        m = results[arm]
        print(f"  {arm:13s} viol_energy={m['viol_energy_pu_s']:.4f} pu·s "
              f"({m['viol_duration_s']:.0f} s)  "
              f"min_reserve={m['min_reserve_margin']:.3f}  "
              f"dso_err mean={m['dso_track_all']['mean']:.2f} / "
              f"commit={m['dso_track_commit_instants']['mean']:.2f} Mvar")
    m = results["sbxv"]
    if "n_requests" in m:
        print(f"  sbxv pipeline: {m['n_requests']} request(s), "
              f"acceptance={m['acceptance_ratio']}, "
              f"{m['n_grants']} grant(s) "
              f"({m['n_dropped_grants']} beyond horizon)")
    if m.get("settlement"):
        print(f"  sbxv payments: {m['settlement']['grand_total_eur']:.2f} "
              f"EUR total; cases {m['settlement']['case_counts']}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 360.0)
