"""
019_SBXV_E2 — SBX-V band-width Pareto sweep (build plan §9 Phase 5, E2)
=======================================================================
Sweep the standard band over ``±{0, 25, 50, 75, 100}`` Mvar plus the
``ar41414_default`` preset (5 %/10 % of contracted P, AR Anhang C
spread assertion), on the Monte-Carlo scenario harness of the 012/006
campaign: each seed deterministically draws a random profile-year start
time and a lightly-constrained random contingency schedule — the
stressed draws are what fire the request pipeline (condition B needs a
real transmission-voltage violation; the benign base profile never
produces one, E1 finding 5).

Arms per seed
-------------
* ``none`` — CAIR baseline (reference violation energy).
* ``b0 | b25 | b50 | b75 | b100`` — SBX-V, symmetric band ±X Mvar.
* ``ar41414`` — SBX-V, the AR preset (contracted P from the rated
  interface capacity Σ sn_hv_mva; the spread assertion may legitimately
  reject it for small areas — recorded as a finding, not a crash).

Metrics per run (plan E2)
-------------------------
Request frequency (requests / metered window), acceptance ratio, grant
count, payments per tier [€], TS violation energy [pu·s] and duration
[s], the persistent-exceedance metric (windows priced beyond the band
WITHOUT a grant — cases ``8.1-4`` / ``8.2-4b_adhoc`` — their count, the
longest consecutive run per area, and the Grenzpreis-tier energy), DSO
tracking error, minimum reserve margin.

Outputs
-------
``results/019_SBXV_E2/runs/run_<seed>_<arm>.json`` (resume unit —
re-running skips completed cells), ``e2_sweep.csv`` (tidy long table),
``e2_pareto.png`` (mean over seeds vs band width), and a findings
paragraph printed at the end.

Usage:  python experiments/019_SBXV_E2.py [--seeds N] [--minutes M]
        (defaults: 3 seeds × 120 min — the calibration-horizon idiom)

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 5, E2)
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from sbx_h.fail import SBXError
from sbx_v.config import SBXVConfig

_005 = importlib.import_module("experiments.005_CIGRE_MULTI")
_006 = importlib.import_module("experiments.006_CIGRE_MONTECARLO")
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

RESULT_DIR = REPO / "results" / "019_SBXV_E2"
RUNS_DIR = RESULT_DIR / "runs"

#: Sweep arms: label → symmetric band half-width [Mvar] (None = preset).
BAND_ARMS = {"b0": 0.0, "b25": 25.0, "b50": 50.0, "b75": 75.0,
             "b100": 100.0, "ar41414": None}
ARMS = ("none",) + tuple(BAND_ARMS)

_ELEMENTS = None   # 006 contingency-element cache


# ----------------------------------------------------------------------
#  Scenario + config construction
# ----------------------------------------------------------------------

def _scenario(seed: int, minutes: float):
    """Deterministic (start_time, contingency schedule) per seed —
    the 012 Monte-Carlo idiom on the shared 005 configuration."""
    global _ELEMENTS
    rng = np.random.default_rng(seed)
    start_time = _006.random_start_time(rng)
    if _ELEMENTS is None:
        _ELEMENTS = _006.enumerate_elements(_base_config(minutes))
    schedule = _006.build_random_schedule(rng, _ELEMENTS,
                                          int(round(minutes)))
    return start_time, schedule


def _base_config(minutes: float):
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    cfg.enable_tie_coordination = False
    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.refresh_shared_jac_on_tso = False
    return cfg


def make_config(arm: str, seed: int, minutes: float):
    cfg = _base_config(minutes)
    start_time, schedule = _scenario(seed, minutes)
    cfg.start_time = start_time
    # Deep copy — the runner mutates schedule entries (006 idiom).
    cfg.contingencies = copy.deepcopy(schedule)
    if arm == "none":
        cfg.coordination_mode = "none"
        return cfg
    cfg.coordination_mode = "sbxv"
    half = BAND_ARMS[arm]
    if half is None:                      # ar41414_default preset
        cfg.sbxv_config = SBXVConfig(
            tso_period_s=float(cfg.tso_period_s),
            band_preset="ar41414_default",
        )
    else:
        cfg.sbxv_config = SBXVConfig(
            tso_period_s=float(cfg.tso_period_s),
            band_preset="fixed",     # explicit: sweep arms use the
            band_q_raise_mvar=half,  # symmetric widths, not the
            band_q_lower_mvar=half,  # ar41414 default (V-D2 rev.)
        )
    return cfg


# ----------------------------------------------------------------------
#  Metrics (shared with 018 where identical)
# ----------------------------------------------------------------------

def voltage_violation_metrics(cfg, recs) -> dict:
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
    m = np.inf
    for r in recs:
        for arr in r.tso_der_q_reserve.values():
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size:
                m = min(m, float(a.min()))
    return m if np.isfinite(m) else float("nan")


def dso_tracking_metric(recs) -> dict:
    err = [abs(float(qs) - float(r.dso_q_actual_mvar[d]))
           for r in recs
           for d, qs in r.dso_q_set_mvar.items()
           if qs is not None and r.dso_q_actual_mvar.get(d) is not None]
    return ({"mean": float(np.mean(err)), "max": float(np.max(err))}
            if err else {"mean": float("nan"), "max": float("nan")})


def sbxv_metrics(final) -> dict:
    """Pipeline + settlement + persistent-exceedance (plan E2/V-D4)."""
    events = [e for log in final["pipeline_logs"].values() for e in log]
    replies = [e for e in events if e[0] == "reply"]
    n_req = sum(1 for e in events if e[0] == "request")
    n_windows = len(final["observations"])
    result = final["settlement"]
    pay = {"energy_avg": 0.0, "energy_grenz": 0.0,
           "cap_avg": 0.0, "cap_grenz": 0.0}
    exceed_windows = 0
    exceed_mvarh = 0.0
    longest_run = 0
    cases = {}
    if result is not None:
        for t in result.totals:
            pay["energy_avg"] += t.pay_energy_avg_eur
            pay["energy_grenz"] += t.pay_energy_grenz_eur
            pay["cap_avg"] += t.pay_cap_avg_eur
            pay["cap_grenz"] += t.pay_cap_grenz_eur
        runs: dict = {}
        for r in sorted(result.window_rows,
                        key=lambda r: (r.area_id, r.window_index)):
            cases[r.case] = cases.get(r.case, 0) + 1
            if r.case in ("8.1-4", "8.2-4b_adhoc"):
                exceed_windows += 1
                exceed_mvarh += r.e_grenz_mvarh
                prev = runs.get(r.area_id)
                cur = (prev[1] + 1 if prev is not None
                       and prev[0] == r.window_index - 1 else 1)
                runs[r.area_id] = (r.window_index, cur)
                longest_run = max(longest_run, cur)
            elif r.area_id in runs and r.direction is not None:
                runs.pop(r.area_id, None)
    return {
        "n_requests": n_req,
        "req_per_window": (n_req / n_windows) if n_windows else 0.0,
        "acceptance_ratio": (
            sum(1 for e in replies if e[4] in ("ACCEPT", "PARTIAL"))
            / len(replies)) if replies else None,
        "n_grants": len(final["grant_records"]),
        "n_dropped_grants": len(final["dropped_grants"]),
        "pay_eur": pay,
        "pay_total_eur": sum(pay.values()),
        "exceed_windows_no_grant": exceed_windows,
        "exceed_longest_consecutive": longest_run,
        "exceed_grenz_mvarh": exceed_mvarh,
        "case_counts": cases,
        "n_windows": n_windows,
    }


# ----------------------------------------------------------------------
#  One cell (seed, arm) with resume
# ----------------------------------------------------------------------

def run_cell(seed: int, arm: str, minutes: float) -> dict:
    path = RUNS_DIR / f"run_{seed}_{arm}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        if row.get("minutes") == minutes:
            print(f"  [{seed}/{arm}] cached")
            return row
    row = {"seed": seed, "arm": arm, "minutes": minutes,
           "band_half_mvar": BAND_ARMS.get(arm), "status": "ok"}
    t0 = time.perf_counter()
    try:
        cfg = make_config(arm, seed, minutes)
        captured: dict = {}

        def hook(state):
            captured.update(state)
            return None

        recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
        row.update(voltage_violation_metrics(cfg, recs))
        row["min_reserve_margin"] = reserve_margin_metric(recs)
        row["dso_track"] = dso_tracking_metric(recs)
        runtime = captured.get("sbxv_runtime") or {}
        adapter = runtime.get("adapter")
        if adapter is not None:
            row.update(sbxv_metrics(adapter.finalise()))
    except SBXError as exc:
        # A legitimate fail-fast (e.g. the AR Anhang C spread assertion
        # rejecting the preset for small areas) is a FINDING.
        row["status"] = f"sbxv_rejected: {exc}"
    row["wall_s"] = round(time.perf_counter() - t0, 1)
    print(f"  [{seed}/{arm}] {row['status']}  ({row['wall_s']:.0f} s)")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=1, default=str)
    return row


# ----------------------------------------------------------------------
#  Aggregation + figure
# ----------------------------------------------------------------------

def write_sweep_csv(rows) -> None:
    keys = ["seed", "arm", "band_half_mvar", "status", "minutes",
            "viol_energy_pu_s", "viol_duration_s", "min_reserve_margin",
            "n_requests", "req_per_window", "acceptance_ratio",
            "n_grants", "pay_total_eur", "exceed_windows_no_grant",
            "exceed_longest_consecutive", "exceed_grenz_mvarh",
            "n_windows", "wall_s"]
    with open(RESULT_DIR / "e2_sweep.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])


def make_figure(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({r["band_half_mvar"] for r in rows
                     if r["arm"] in BAND_ARMS
                     and r["band_half_mvar"] is not None
                     and r["status"] == "ok"})
    if not widths:
        print("  (no successful band arms — figure skipped)")
        return

    def _mean(metric, w):
        vals = [r.get(metric) for r in rows
                if r.get("band_half_mvar") == w and r["status"] == "ok"
                and r.get(metric) is not None]
        return float(np.mean(vals)) if vals else np.nan

    base_viol = [r["viol_energy_pu_s"] for r in rows
                 if r["arm"] == "none"]
    panels = [
        ("req_per_window", "requests / window"),
        ("pay_total_eur", "payments [EUR]"),
        ("viol_energy_pu_s", "violation energy [pu·s]"),
        ("exceed_grenz_mvarh", "no-grant exceedance [Mvarh]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        ax.plot(widths, [_mean(metric, w) for w in widths],
                marker="o")
        if metric == "viol_energy_pu_s" and base_viol:
            ax.axhline(float(np.mean(base_viol)), ls="--", c="grey",
                       label="baseline (none)")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("band half-width [Mvar]")
    fig.suptitle("E2: SBX-V band-width Pareto "
                 f"(mean over seeds, {int(rows[0]['minutes'])} min)")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "e2_pareto.png", dpi=150)
    print(f"  figure -> {RESULT_DIR / 'e2_pareto.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=120.0)
    args = ap.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"E2 band-width Pareto: {args.seeds} seed(s) × {len(ARMS)} "
          f"arm(s) × {args.minutes:.0f} min")
    rows = [run_cell(seed, arm, args.minutes)
            for seed in range(1, args.seeds + 1)
            for arm in ARMS]
    write_sweep_csv(rows)
    print(f"  sweep table -> {RESULT_DIR / 'e2_sweep.csv'}")
    make_figure(rows)

    # ── Findings printout ────────────────────────────────────────────
    print("\n=== E2 findings (means over seeds) ===")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        ok = [r for r in sub if r["status"] == "ok"]
        if not ok:
            print(f"  {arm:8s} REJECTED: {sub[0]['status'][:100]}")
            continue
        def m(k):
            vals = [r[k] for r in ok if r.get(k) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        extra = ""
        if arm != "none":
            extra = (f"  req/w={m('req_per_window'):.3f} "
                     f"grants={m('n_grants'):.1f} "
                     f"pay={m('pay_total_eur'):8.0f} EUR "
                     f"exceed={m('exceed_grenz_mvarh'):.1f} Mvarh "
                     f"(longest {m('exceed_longest_consecutive'):.1f} w)")
        print(f"  {arm:8s} viol={m('viol_energy_pu_s'):8.4f} pu·s "
              f"reserve={m('min_reserve_margin'):.3f}{extra}")


if __name__ == "__main__":
    main()
