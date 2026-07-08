#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/012_BME_MONTECARLO.py
=================================
BME Monte-Carlo campaign (spec §5 Phase 6 item 6 / §6): sweep load
scenarios × delay d × β × drop × ε_switch × sensitivity error σ_H with
fixed seeds; results to parquet + a generated summary markdown.

Primary purposes (spec §6): populate the switching ledger across
conditions to validate the finite-switching premise (§3.10.2 —
P(realised ΔΦ ≤ −ε/2 | accepted)), and characterise delay sensitivity
(§3.10.3, "degrades gracefully with d and sensitivity error").

Design (Manuel 2026-07-05: recalibrated D2 edges (1.02, 1.04); H-error
= H_{b,i}-slice scope; full ~75-run design, run autonomously):

* **Paired scenarios.** Each scenario seed deterministically draws a
  random profile-year start time + a lightly-constrained random
  contingency schedule (both REUSED from 006's generator). All arms of
  a scenario run on that identical scenario; paired differences remove
  scenario variance.
* **Base arms** (N_BASE scenarios, drop-and-replace: a scenario is
  accepted only if all three converge): ``none`` | ``bme`` nominal
  (d=1, β=0.3, drop=0, ε=5.2e3, σ_H=0) | ``oracle`` (single-zone D8).
* **Sweep cells** (first SWEEP_N accepted scenarios, one factor at a
  time around the bme nominal): d ∈ {0,2,5}, β ∈ {0.1,0.6,1.0},
  drop ∈ {0.05,0.2}, ε ∈ {0, 1e3, 2.6e4}, σ_H ∈ {0.05,0.15,0.3}, and
  the **selfish-Φ_i ablation** drop=1.0 (price term never arrives →
  each zone descends its own Φ_i only; isolates the μ-exchange
  contribution).
* Horizon 120 min (calibration-horizon rule); coordination config and
  metric definition identical to ``011_BME_LADDER`` (same constants).

Outputs under ``results/012_BME_MC/``:
  * ``runs/run_<seed>_<arm>.json``  — per-run metrics (resume unit),
    ``ledger_<seed>_<arm>.csv``, ``ts_<seed>_<arm>.npz`` (compact
    series: Φ, losses, boundary V — enough to rebuild figures);
  * ``runs.parquet`` / ``ledger.parquet`` — aggregated tables;
  * ``schedules.csv`` — verbatim contingency events per scenario;
  * ``MC_SUMMARY.md`` — generated summary (base distribution, paired
    gaps, per-axis sweep tables, ledger premise statistics).

CLI:
  --run           execute the campaign (phases A base + B sweeps)
  --jobs N        process parallelism (default 1)
  --smoke         1 scenario, 30 min, no sweeps (pre-launch check)
  --summarize     rebuild parquet + MC_SUMMARY.md from runs/ only

Author: Manuel Schwenke / Claude Code
Date: 2026-07-05/06 (BME Phase 6 item 6)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import importlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULT_DIR = REPO / "results" / "012_BME_MC"
RUNS_DIR = RESULT_DIR / "runs"

# ── Campaign constants ─────────────────────────────────────────────────
BASE_SEED = 20260705
N_BASE = 10          # accepted base scenarios (drop-and-replace)
SWEEP_N = 3          # sweeps run on the first SWEEP_N accepted seeds
MINUTES = 120.0
MAX_ATTEMPTS = 40    # scenario-seed attempts before giving up

#: bme nominal = the calibrated Phase 6 operating point.
NOMINAL = {"d": 1, "beta": 0.3, "drop": 0.0, "eps": None, "h_sigma": 0.0}
# eps=None → keep the ladder's calibrated BME_EPSILON_SWITCH.

#: One-factor sweep cells: arm name -> override dict.
SWEEP_ARMS = {
    "bme_d0":      {"d": 0},
    "bme_d2":      {"d": 2},
    "bme_d5":      {"d": 5},
    "bme_beta01":  {"beta": 0.1},
    "bme_beta06":  {"beta": 0.6},
    "bme_beta10":  {"beta": 1.0},
    "bme_drop005": {"drop": 0.05},
    "bme_drop020": {"drop": 0.2},
    "bme_eps0":    {"eps": 0.0},
    "bme_eps1e3":  {"eps": 1.0e3},
    "bme_eps26e3": {"eps": 2.6e4},
    "bme_hs005":   {"h_sigma": 0.05},
    "bme_hs015":   {"h_sigma": 0.15},
    "bme_hs030":   {"h_sigma": 0.3},
    "bme_selfish": {"drop": 1.0},   # price term suppressed (ablation)
}
BASE_ARMS = ("none", "bme", "oracle")

# ── Lazy imports of the reused experiment modules (heavy) ──────────────
_L = None    # 011 ladder module (config factory + metric helpers)
_M = None    # 006 MC module (scenario generator)


def _mods():
    global _L, _M
    if _L is None:
        _L = importlib.import_module("experiments.011_BME_LADDER")
        _M = importlib.import_module("experiments.006_CIGRE_MONTECARLO")
    return _L, _M


_ELEMENTS = None  # per-process cache for 006's enumerate_elements


def _scenario(seed: int, minutes: float):
    """Deterministic (start_time, schedule) for a scenario seed."""
    global _ELEMENTS
    L, M = _mods()
    rng = np.random.default_rng(seed)
    start_time = M.random_start_time(rng)
    if _ELEMENTS is None:
        _ELEMENTS = M.enumerate_elements(L.make_ladder_config("none", minutes))
    schedule = M.build_random_schedule(rng, _ELEMENTS, int(round(minutes)))
    return start_time, schedule


def _cfg_for(arm: str, seed: int, minutes: float):
    L, _ = _mods()
    rung = arm if arm in ("none", "oracle") else "bme"
    cfg = L.make_ladder_config(rung, minutes)
    cfg.verbose = 0
    start_time, schedule = _scenario(seed, minutes)
    cfg.start_time = start_time
    cfg.contingencies = schedule
    if arm not in ("none", "oracle"):
        ov = dict(NOMINAL)
        ov.update(SWEEP_ARMS.get(arm, {}))    # "bme" nominal = no update
        cfg.bme_delay_steps = int(ov["d"])
        cfg.bme_beta_filter = float(ov["beta"])
        cfg.bme_drop_probability = float(ov["drop"])
        if ov["drop"] > 0.0:
            cfg.bme_seed = seed * 10 + 1
        if ov["eps"] is not None:
            cfg.bme_epsilon_switch = float(ov["eps"])
        if ov["h_sigma"] > 0.0:
            cfg.bme_h_error_rel_sigma = float(ov["h_sigma"])
            cfg.bme_h_error_seed = seed * 10 + 2
    return cfg


# ── Single run (top-level: picklable for ProcessPoolExecutor) ──────────

def run_single(spec: dict) -> dict:
    """Run one (scenario seed, arm); write staging JSON/CSV/NPZ; return
    the metrics row. Never raises — failures land in the row."""
    seed, arm, minutes = spec["seed"], spec["arm"], spec["minutes"]
    L, _ = _mods()
    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    row = {"seed": seed, "arm": arm, "minutes": minutes,
           "converged": False, "error": ""}
    t0 = time.perf_counter()
    try:
        cfg = _cfg_for(arm, seed, minutes)
        captured = {}

        def hook(state):
            captured.update(state)
            return None

        recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
        expected = int(round(cfg.n_total_s / cfg.dt_s))
        row["steps"] = len(recs)
        row["converged"] = len(recs) == expected
        if not row["converged"]:
            row["error"] = f"short log {len(recs)}/{expected}"
    except Exception:
        row["error"] = traceback.format_exc(limit=3)
        recs, captured = [], {}
    row["runtime_s"] = round(time.perf_counter() - t0, 1)

    if recs:
        t = L._t_min(recs)
        m = L._last_hour_mask(t)
        phi = L._phi_series(recs)
        loss = L._loss_series(recs)
        row.update({
            "losses_lh_mw": float(np.mean(loss[m])),
            "phi_lh_mw": float(np.nanmean(phi[m])),
            "phi_full_mw": float(np.nanmean(phi)),
            "band_viol_frac": float(L._band_violation_fraction(recs)),
            "oltc_switches": int(L._oltc_switch_count(recs)),
        })
        row.update(L._oscillation_indicator(recs))
        for z, v in L._zone_phi_last_hour(recs).items():
            row[f"phi_z{z}_mw"] = v
        # Compact series for later figures (no full pickles in the MC).
        vb, vb_cols = L._boundary_v_matrix(recs)
        np.savez_compressed(
            RUNS_DIR / f"ts_{seed}_{arm}.npz",
            t_min=t, phi=phi, losses=loss,
            v_boundary=(vb if vb is not None else np.empty((0, 0))),
            v_boundary_cols=np.array(vb_cols, dtype=object),
        )

    ledger = captured.get("bme_ledger")
    if ledger is not None and len(ledger) > 0:
        entries = ledger.to_records()
        row["ledger_accepted"] = sum(
            1 for e in entries if e["reason"] == "accepted")
        row["ledger_eps_reject"] = sum(
            1 for e in entries if e["reason"] == "epsilon_reject")
        row["ledger_slot_blocked"] = sum(
            1 for e in entries if e["reason"] == "slot_blocked")
        with open(RUNS_DIR / f"ledger_{seed}_{arm}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
            w.writeheader()
            for e in entries:
                w.writerow(e)

    with open(RUNS_DIR / f"run_{seed}_{arm}.json", "w") as f:
        json.dump(row, f, indent=1, default=str)
    return row


def _done(seed: int, arm: str, minutes: float) -> dict | None:
    p = RUNS_DIR / f"run_{seed}_{arm}.json"
    if p.exists():
        with open(p) as f:
            row = json.load(f)
        # Horizon-aware resume: a staged run only counts if it was run
        # at the requested horizon (a 30-min --smoke artefact must never
        # satisfy a 120-min campaign spec for the same seed/arm).
        if float(row.get("minutes", -1.0)) == float(minutes):
            return row
    return None


# ── Campaign driver ────────────────────────────────────────────────────

def _run_specs(specs, jobs: int):
    """Run specs (skipping already-staged ones); return rows."""
    rows, todo = [], []
    for s in specs:
        prior = _done(s["seed"], s["arm"], s["minutes"])
        if prior is not None:
            rows.append(prior)
        else:
            todo.append(s)
    if not todo:
        return rows
    if jobs <= 1:
        for s in todo:
            rows.append(run_single(s))
            r = rows[-1]
            print(f"  [{r['seed']}/{r['arm']}] conv={r['converged']} "
                  f"loss_lh={r.get('losses_lh_mw', float('nan')):.3f} "
                  f"({r['runtime_s']:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_single, s): s for s in todo}
            for fu in as_completed(futs):
                r = fu.result()
                rows.append(r)
                print(f"  [{r['seed']}/{r['arm']}] conv={r['converged']} "
                      f"loss_lh={r.get('losses_lh_mw', float('nan')):.3f} "
                      f"({r['runtime_s']:.0f}s)", flush=True)
    return rows


def run_campaign(jobs: int, smoke: bool) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    L, M = _mods()
    n_base = 1 if smoke else N_BASE
    minutes = 30.0 if smoke else MINUTES

    # Phase A — base triples with drop-and-replace.
    accepted, attempt = [], 0
    sched_rows = []
    while len(accepted) < n_base and attempt < MAX_ATTEMPTS:
        seed = BASE_SEED + attempt
        attempt += 1
        print(f"=== scenario seed {seed} "
              f"({len(accepted)}/{n_base} accepted) ===", flush=True)
        rows = _run_specs(
            [{"seed": seed, "arm": a, "minutes": minutes}
             for a in BASE_ARMS], jobs)
        if all(r["converged"] for r in rows):
            accepted.append(seed)
            start_time, schedule = _scenario(seed, minutes)
            sched_rows += M.schedule_to_rows(
                len(accepted) - 1, seed, start_time, schedule)
        else:
            bad = [r["arm"] for r in rows if not r["converged"]]
            print(f"  DROPPED (failed arms: {bad})", flush=True)
    if sched_rows:
        # Rewritten whole each campaign invocation (accepted seeds are
        # re-derived deterministically on resume → complete, deduped).
        with open(RESULT_DIR / "schedules.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sched_rows[0].keys()))
            w.writeheader()
            for r in sched_rows:
                w.writerow(r)
    print(f"Phase A done: {len(accepted)} accepted scenarios "
          f"({attempt} attempts)", flush=True)

    # Phase B — one-factor sweeps on the first SWEEP_N accepted seeds.
    if not smoke:
        sweep_seeds = accepted[:SWEEP_N]
        specs = [
            {"seed": s, "arm": arm, "minutes": minutes}
            for s in sweep_seeds for arm in SWEEP_ARMS
        ]
        print(f"=== Phase B: {len(specs)} sweep runs on seeds "
              f"{sweep_seeds} ===", flush=True)
        _run_specs(specs, jobs)

    summarize()


# ── Aggregation: parquet + MC_SUMMARY.md ───────────────────────────────

def _load_rows() -> list:
    rows = []
    for p in sorted(RUNS_DIR.glob("run_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def _agg(vals):
    if vals is None:
        return "—"
    v = np.asarray([x for x in vals
                    if x is not None and np.isfinite(float(x))],
                   dtype=float)
    if v.size == 0:
        return "—"
    return f"{v.mean():.3f} ± {v.std(ddof=1) if v.size > 1 else 0.0:.3f}"


def summarize() -> None:
    import pandas as pd
    rows = _load_rows()
    if not rows:
        print("no runs staged yet")
        return
    df = pd.DataFrame(rows)
    df.to_parquet(RESULT_DIR / "runs.parquet", index=False)

    led = []
    for p in sorted(RUNS_DIR.glob("ledger_*.csv")):
        seed, arm = p.stem.split("_", 2)[1:]
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["seed"], r["arm"] = int(seed), arm
                led.append(r)
    if led:
        pd.DataFrame(led).to_parquet(
            RESULT_DIR / "ledger.parquet", index=False)

    ok = df[df.converged].set_index(["seed", "arm"])
    lines = ["# BME MC campaign — generated summary", "",
             f"runs staged: {len(df)} (converged {int(df.converged.sum())}"
             f", failed {int((~df.converged).sum())})", ""]

    # Base-arm distribution + paired gaps.
    base = df[df.arm.isin(BASE_ARMS) & df.converged]
    seeds = sorted(set(base.seed))
    lines += ["## Base arms (paired scenarios)", "",
              "| arm | losses last-h [MW] | Φ last-h [MW] | OLTC | "
              "AR pole |", "|---|---|---|---|---|"]
    for arm in BASE_ARMS:
        g = base[base.arm == arm]
        lines.append(
            f"| {arm} | {_agg(g.losses_lh_mw)} | {_agg(g.phi_lh_mw)} | "
            f"{_agg(g.oltc_switches)} | {_agg(g.osc_pole_mod)} |")
    pair_loss_red, pair_gap = [], []
    for s in seeds:
        try:
            n = ok.loc[(s, "none")]
            b = ok.loc[(s, "bme")]
            o = ok.loc[(s, "oracle")]
        except KeyError:
            continue
        pair_loss_red.append(
            100.0 * (1.0 - b.losses_lh_mw / n.losses_lh_mw))
        pair_gap.append(b.losses_lh_mw - o.losses_lh_mw)
    lines += ["", f"paired bme loss reduction vs none [%]: "
                  f"{_agg(pair_loss_red)}",
              f"paired bme − oracle losses [MW]: {_agg(pair_gap)}", ""]

    # Sweep tables (Δ vs the nominal bme on the SAME scenarios).
    axes = {"delay d": ["bme_d0", "bme", "bme_d2", "bme_d5"],
            "β filter": ["bme_beta01", "bme", "bme_beta06", "bme_beta10"],
            "drop prob": ["bme", "bme_drop005", "bme_drop020",
                          "bme_selfish"],
            "ε_switch": ["bme_eps0", "bme_eps1e3", "bme", "bme_eps26e3"],
            "H error σ": ["bme", "bme_hs005", "bme_hs015", "bme_hs030"]}
    for title, arms in axes.items():
        lines += [f"## Sweep — {title}", "",
                  "| arm | losses last-h [MW] | Δloss vs nominal [MW] | "
                  "Φ last-h [MW] | OLTC | ledger acc/εrej/slot |",
                  "|---|---|---|---|---|---|"]
        for arm in arms:
            g = df[(df.arm == arm) & df.converged]
            g = g[g.seed.isin(seeds[:SWEEP_N])]
            dl = []
            for s in seeds[:SWEEP_N]:
                try:
                    dl.append(float(ok.loc[(s, arm)].losses_lh_mw
                                    - ok.loc[(s, "bme")].losses_lh_mw))
                except KeyError:
                    pass
            la = (f"{_agg(g.get('ledger_accepted'))}/"
                  f"{_agg(g.get('ledger_eps_reject'))}/"
                  f"{_agg(g.get('ledger_slot_blocked'))}")
            lines.append(
                f"| {arm} | {_agg(g.losses_lh_mw)} | {_agg(dl)} | "
                f"{_agg(g.phi_lh_mw)} | {_agg(g.oltc_switches)} | {la} |")
        lines.append("")

    # §3.10.2 finite-switching premise from the pooled ledgers.
    if led:
        ld = pd.DataFrame(led)
        acc = ld[ld.reason == "accepted"].copy()
        if len(acc):
            L, _ = _mods()
            # to_numeric with coerce: last-tick entries never receive the
            # deferred realised-ΔΦ fill and round-trip as empty strings.
            acc["pred_mw"] = (pd.to_numeric(acc.predicted_dphi,
                                            errors="coerce")
                              / L.BME_GRADIENT_SCALE)
            acc["real_mw"] = pd.to_numeric(acc.realised_dphi,
                                           errors="coerce")
            acc = acc[np.isfinite(acc.real_mw)]
            eps_mw = L.BME_EPSILON_SWITCH / L.BME_GRADIENT_SCALE
            p_premise = float(np.mean(acc.real_mw <= -eps_mw / 2.0))
            sign_agree = float(np.mean(np.sign(acc.pred_mw)
                                       == np.sign(acc.real_mw)))
            lines += ["## Finite-switching premise (§3.10.2, pooled "
                      "accepted switches)", "",
                      f"accepted switches with realised ΔΦ: {len(acc)}",
                      f"P(realised ΔΦ ≤ −ε/2 | accepted) = "
                      f"{p_premise:.3f}  (ε = {eps_mw:.4f} MW)",
                      f"sign agreement predicted vs realised: "
                      f"{sign_agree:.3f}", ""]

    (RESULT_DIR / "MC_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"summary → {RESULT_DIR / 'MC_SUMMARY.md'}")
    print(f"parquet → {RESULT_DIR / 'runs.parquet'}"
          + (f" + ledger.parquet ({len(led)} entries)" if led else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 scenario, 30 min, no sweeps")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    elif args.run or args.smoke:
        run_campaign(jobs=args.jobs, smoke=args.smoke)
    else:
        ap.error("pass --run, --smoke or --summarize")


if __name__ == "__main__":
    main()
