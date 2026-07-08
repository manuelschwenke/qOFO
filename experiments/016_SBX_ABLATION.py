#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/016_SBX_ABLATION.py
===============================
SBX quantum / cycle-length ablation (plan v2 §4 Phase 7 optional
ablation, deferred as D-P7-4 and activated by Manuel 2026-07-08):
does a LARGER QUANTUM or a SHORTER CYCLE make the mechanism's relief
material?

Variants on one 013 scenario (default ``asym_z3``), all sharing the
scenario, the calibrated tier-1 band and the contract voltages; ONLY
the exchange dynamics differ:

* ``sbx``            — the campaign reference (15-min cycle, quantum
                       rate 10 Mvar/15 min, cap 50 Mvar) — loaded from
                       the stored 013 pickle, not re-run.
* ``sbx_fast``       — 3-min cycle (``k_sched = 1``): per-cycle quantum
                       2 Mvar (rate-scaled, same Mvar/h ramp), deal
                       latency cut from ≤ 15 min to ≤ 3 min; settlement
                       on a rolling 5-cycle window (``n_settle_cycles``)
                       so the 15-min averaging is preserved.
* ``sbx_bigq``       — 15-min cycle, quantum rate 30 Mvar/15 min and
                       cap 100 Mvar (a big quantum saturates the 50-Mvar
                       cap after two deals, so the cap is raised with
                       it): triple ramp rate, triple per-deal terminal
                       shift.
* ``sbx_fast_bigq``  — both: 3-min cycle at rate 30 (6 Mvar per cycle),
                       cap 100.

Baselines ``none`` / ``sbx_inert`` are loaded from the 013 pickles.

Outputs (``results/016_SBX_ABLATION/<scenario>/``): per-variant pickles,
``ABLATION.md`` (exposure / deals / latency / payments / margin table)
and ``F1_ablation.png`` (stressed-zone exposure bars + the scheduled
import trajectory of every variant).

Run:
    python experiments/016_SBX_ABLATION.py --run [--scenario asym_z3]
    python experiments/016_SBX_ABLATION.py            (evaluate only)

Author: Manuel Schwenke / Claude Code
Date: 2026-07-08 (SBX Phase 7 ablation, D-P7-4)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sbx.config import SBXConfig  # noqa: E402
from sbx.fail import rep1  # noqa: E402

_013 = importlib.import_module("experiments.013_SBX_LADDER")
_015 = importlib.import_module("experiments.015_SBX_COMPARE")

RESULT_DIR = REPO / "results" / "016_SBX_ABLATION"

#: Variant definitions: SBXConfig overrides beyond the scenario band.
VARIANTS: Dict[str, dict] = {
    "sbx_fast": dict(k_sched=1, n_settle_cycles=6),
    # "sbx_bigq": dict(dq_quant_rate_mvar_per_15min=30.0,
    #                  dq_contract_max_mvar=100.0),
    "sbx_fast_bigq": dict(k_sched=1, n_settle_cycles=6,
                          dq_quant_rate_mvar_per_15min=30.0,
                          dq_contract_max_mvar=100.0),
}

#: Fixed variant colours (identity never repainted across figures).
COLOURS = {
    "none": "#888888",
    "sbx_inert": "#bbbbbb",
    "sbx": "#4477aa",
    "sbx_fast": "#117733",
    "sbx_bigq": "#cc6677",
    "sbx_fast_bigq": "#aa4499",
}


def make_variant_config(scenario: str, variant: str, minutes: float,
                        band: float):
    cfg = _013.make_config(scenario, "sbx", minutes, q_band_mvar=band)
    overrides = VARIANTS[variant]
    cfg.sbx_config = SBXConfig(
        tso_period_s=float(cfg.tso_period_s),
        q_band_mvar=float(band),
        **overrides,
    )
    return cfg


def run_variant(scenario: str, variant: str, minutes: float,
                band: float) -> dict:
    """One closed-loop run; returns the 013-style payload dict."""
    import time as _time
    from experiments.runners.multi_tso_dso import run_multi_tso_dso

    cfg = make_variant_config(scenario, variant, minutes, band)
    cfg.verbose = 0
    captured: dict = {}

    def hook(state):
        captured.update(state)
        return None

    t0 = _time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    print(f"  [{scenario}/{variant}] {len(recs)} steps in "
          f"{_time.perf_counter() - t0:.0f} s")
    adapter = (captured.get("sbx_runtime") or {}).get("adapter")
    return {"scenario": scenario, "arm": variant, "minutes": minutes,
            "records": recs, "sbx": _013.extract_sbx_runtime(adapter),
            "q_band_mvar": band}


def scheduled_import_series(ext: dict, stressed_zone: int):
    """(t_min, Mvar) of the total SCHEDULED import into the stressed
    zone: the signed sum of the corridor surpluses, oriented so that
    positive = flow towards the stressed zone."""
    cfg = ext["config"]
    per_cycle: Dict[int, float] = {}
    for key, rl in ext["records"].items():
        if stressed_zone not in key:
            continue
        sign = +1.0 if stressed_zone == key[1] else -1.0
        for r in rl:
            per_cycle[r.cycle] = per_cycle.get(r.cycle, 0.0) \
                + sign * r.surplus_mvar
    cycles = sorted(per_cycle)
    t = [_013.WARMUP_MIN + c * cfg.t_cycle_min for c in cycles]
    return np.asarray(t), np.asarray([per_cycle[c] for c in cycles])


def evaluate(scenario: str, data: Dict[str, dict]) -> None:
    spec = _013.SCENARIOS[scenario]
    z0 = spec["stressed_zones"][0]
    cfg_ref = _013.make_config(scenario, "none", _013.DEFAULT_MINUTES)
    v_lo = spec["zone_v_min"].get(z0, cfg_ref.v_min_pu)
    v_hi = spec["zone_v_max"].get(z0, cfg_ref.v_max_pu)
    lo_s, hi_s = 60.0 * _013.STRESS_ON_MIN, 60.0 * _013.STRESS_OFF_MIN

    rows = []
    for name, d in data.items():
        expo, n = _013.violation_exposure(d["records"], z0, v_lo, v_hi,
                                          lo_s, hi_s)
        row = {"variant": name, "expo_pustep": expo, "expo_steps": n,
               "n_deals": "", "exchanged_mvar": "", "peak_surplus": "",
               "first_deal_min": "", "tier2_eur": "", "quantum": "",
               "cycle_min": ""}
        ext = d.get("sbx")
        if ext is not None:
            cfg = ext["config"]
            deals = [(key, r) for key, rl in ext["records"].items()
                     for r in rl if r.deal.dq_deal_mvar != 0.0]
            row["n_deals"] = len(deals)
            row["exchanged_mvar"] = round(sum(
                abs(r.deal.dq_deal_mvar) for _, r in deals), 1)
            row["peak_surplus"] = round(max(
                (abs(r.surplus_mvar) for rl in ext["records"].values()
                 for r in rl), default=0.0), 1)
            if deals:
                first = min(r.cycle for _, r in deals)
                row["first_deal_min"] = _013.WARMUP_MIN \
                    + first * cfg.t_cycle_min
            row["tier2_eur"] = round(sum(
                s.tier2_eur for sl in ext["settlements"].values()
                for s in sl), 2)
            row["quantum"] = cfg.dq_quant_mvar
            row["cycle_min"] = cfg.t_cycle_min
        rows.append(row)

    out_dir = RESULT_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Markdown table ──────────────────────────────────────────────────
    cols = ["variant", "cycle_min", "quantum", "expo_pustep",
            "expo_steps", "n_deals", "exchanged_mvar", "peak_surplus",
            "first_deal_min", "tier2_eur"]
    lines = [f"# 016 quantum/cycle ablation — {scenario} "
             f"(stressed zone {z0})", "",
             "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines += ["",
              f"Exposure = Σ violation depth of zone {z0} over the "
              f"stress window [pu·step]. first_deal_min = simulation "
              f"minute of the first executed deal (stress starts at "
              f"minute {_013.STRESS_ON_MIN:.0f}; the need flag needs "
              f"15 min of persistence in every variant)."]
    (out_dir / "ABLATION.md").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")

    # ── Figure: exposure bars + scheduled-import trajectories ──────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), height_ratios=[1, 1.4])
    names = [r["variant"] for r in rows]
    expos = [r["expo_pustep"] for r in rows]
    bars = ax1.bar(range(len(names)), expos,
                   color=[COLOURS.get(n, "0.4") for n in names],
                   width=0.62)
    for b, e in zip(bars, expos):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{e:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel(f"zone-{z0} exposure / pu·step", fontsize=9)
    ax1.set_title(f"{scenario}: violation exposure in the stress window",
                  fontsize=10, loc="left")
    ax1.grid(alpha=0.25, lw=0.4, axis="y")

    for name, d in data.items():
        ext = d.get("sbx")
        if ext is None or not any(
                r.deal.dq_deal_mvar != 0.0
                for rl in ext["records"].values() for r in rl):
            continue
        t, imp = scheduled_import_series(ext, z0)
        ax2.step(t, imp, where="post", lw=1.5,
                 color=COLOURS.get(name, "0.4"), label=name)
    ax2.axvspan(_013.STRESS_ON_MIN, _013.STRESS_OFF_MIN,
                color="#f0efec", zorder=0)
    ax2.set_xlabel("time / min", fontsize=9)
    ax2.set_ylabel(f"scheduled import into zone {z0} / Mvar", fontsize=9)
    ax2.set_title("scheduled support: how much, how fast, and the "
                  "release", fontsize=10, loc="left")
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "F1_ablation.png", dpi=160)
    plt.close(fig)
    print(f"table + figure written to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SBX quantum / cycle-length ablation (D-P7-4).")
    ap.add_argument("--scenario", type=str, default="asym_z3",
                    choices=sorted(_013.SCENARIOS.keys()))
    ap.add_argument("--run", action="store_true",
                    help="execute the three ablation variants "
                         "(≈ 10-15 min each); otherwise evaluate "
                         "stored pickles only")
    ap.add_argument("--minutes", type=float, default=_013.DEFAULT_MINUTES)
    args = ap.parse_args()

    scenario = args.scenario
    _014 = importlib.import_module("experiments.014_SBX_SINGLE_DEMO")
    band = _014.CALIBRATED_BAND_MVAR[scenario]
    out_dir = RESULT_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        for variant in VARIANTS:
            payload = run_variant(scenario, variant, args.minutes, band)
            with open(out_dir / f"arm_{variant}.pkl", "wb") as fh:
                pickle.dump(payload, fh)

    # Baselines from 013; variants from this experiment's pickles.
    data: Dict[str, dict] = {}
    for arm in ("none", "sbx_inert", "sbx"):
        data[arm] = _015.load_arm(scenario, arm)
    for variant in VARIANTS:
        pkl = out_dir / f"arm_{variant}.pkl"
        if not pkl.exists():
            rep1("ablation variant pickle missing — run with --run",
                 scenario=scenario, variant=variant, path=str(pkl))
        with open(pkl, "rb") as fh:
            data[variant] = pickle.load(fh)

    evaluate(scenario, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
