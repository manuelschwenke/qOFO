"""
011_BME_LADDER — BME evaluation ladder (spec §5 Phase 6, §6).

Rungs (ONE shared scenario definition; only the coordination
configuration differs — spec §6):

  none      (a)  uncoordinated baseline (current non-cooperative behaviour)
  vref      (b)  existing two-loop ΔV_ref tie coordinator
  bme       (c)  Boundary Marginal Exchange (calibrated w_Φ, D2 band on)
  bme_loss  (c') bme with w_band = 0 — the losses-only ablation rung (D2)
  oracle    (d)  centralised per-step OFO-MIQP with Φ — NOT YET WIRED
                 (D8; V5/central-controller machinery — Phase 6 follow-up)

Usage (one command per rung, spec §5 Phase 6 acceptance):

    python experiments/011_BME_LADDER.py --rung none  [--minutes 360]
    python experiments/011_BME_LADDER.py --rung all

Outputs under ``results/011_BME_LADDER/``: per-rung record pickle,
``metrics.csv`` (appended per rung; recomputed rows replace old ones),
switching-ledger CSV for bme rungs, and a printed comparison table.

Metrics (uniform across rungs, spec §5 Phase 6):
  * Φ trajectory (``bme_phi_mw`` — recorded for EVERY rung via
    ``record_bme_phi``), settling mean over the last hour;
  * total network losses (MW, whole combined net — sanity reference);
  * per-zone voltage-band violation time (fraction of steps with any
    EHV bus outside the soft band, from zone_v_min/max);
  * discrete switch counts per device class (OLTC tap trajectory
    diffs; MIQP shunts when present);
  * bme rungs: ledger statistics (accept / ε-reject / slot-blocked
    counts, predicted-vs-realised ΔΦ pairs — §3.10.2 premise data).

Gap-to-oracle and the Phulpin fairness metric activate once rung (d)
lands.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-03 (BME Phase 6)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_005 = importlib.import_module("experiments.005_CIGRE_MULTI")

RESULT_DIR = REPO / "results" / "011_BME_LADDER"

# ── Calibrated constants (Phase 6a, docs/BME_STATUS.md) ────────────────
# w_Φ: bme_gradient_scale from the 2026-07-03 calibration sweep
# (60-min CIGRE scenario, losses-only Φ, vs the mode="none" reference):
#   1e4 → inert-to-noise (33.49 MW sustained vs none 32.92);
#   1e5 → −5.8 % sustained losses, V ∈ [0.986, 1.059] (contained);
#   1e6 → −8.4 % but V_max 1.140 (losses-only Φ escapes the band);
#   1e7 → over-driven: V_max 1.179, 3.7× runtime (solver stress).
# Chosen: 1e5 — the largest swept scale whose voltage envelope stays
# contained WITHOUT the band hinge. The Phase 6 D2 sweep revisits the
# (w_Φ × w_band) pairing jointly; 1e6 is only admissible with a binding
# band mechanism.
BME_GRADIENT_SCALE = 1.0e5
# D2 starting band weight (±3 % edges are the config defaults); the
# w_band sweep is a later Phase 6 item — bme_loss is the w_band=0 rung.
BME_W_BAND = 1.0e3

RUNGS = ("none", "vref", "bme", "bme_loss")


def make_ladder_config(rung: str, minutes: float):
    """The SHARED scenario (005 CIGRE cascade tuning) + the rung's
    coordination configuration. Nothing else may differ between rungs
    (spec §6)."""
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    cfg.record_bme_phi = True          # uniform Φ metric on every rung
    cfg.enable_tie_coordination = False
    cfg.coordination_mode = "none"

    if rung == "none":
        pass
    elif rung == "vref":
        cfg.enable_tie_coordination = True
        cfg.coordination_mode = "vref"
    elif rung in ("bme", "bme_loss"):
        if BME_GRADIENT_SCALE is None:
            raise RuntimeError(
                "BME_GRADIENT_SCALE is not calibrated yet — run the "
                "Phase 6a calibration and fill the constant "
                "(docs/BME_STATUS.md Phase 6 section)."
            )
        cfg.coordination_mode = "bme"
        cfg.refresh_shared_jac_on_tso = True
        cfg.local_sensitivities_tso = False
        cfg.local_sensitivities_dso = False
        cfg.bme_gradient_scale = float(BME_GRADIENT_SCALE)
        cfg.bme_w_band = BME_W_BAND if rung == "bme" else 0.0
    else:
        raise ValueError(f"unknown rung '{rung}' (choose from {RUNGS})")
    return cfg


def _band_violation_fraction(recs, v_lo=0.97, v_hi=1.03):
    """Fraction of steps with ANY zone's EHV extreme outside the band."""
    viol = 0
    for r in recs:
        if not r.zone_v_min:
            continue
        vmin = min(r.zone_v_min.values())
        vmax = max(r.zone_v_max.values())
        if vmin < v_lo or vmax > v_hi:
            viol += 1
    return viol / max(1, len(recs))


def _oltc_switch_count(recs):
    """Total whole-tap moves over the run, summed over zones."""
    count = 0
    prev = {}
    for r in recs:
        for z, taps in r.zone_oltc_taps.items():
            taps = np.asarray(taps, dtype=float)
            if z in prev and prev[z].shape == taps.shape:
                count += int(np.sum(np.abs(np.round(taps - prev[z]))))
            prev[z] = taps
    return count


def run_rung(rung: str, minutes: float) -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_ladder_config(rung, minutes)

    captured = {}

    def hook(state):
        captured.update(state)
        return None  # continue the run

    t0 = time.perf_counter()
    recs = run_multi_tso_dso(cfg, pre_loop_hook=hook)
    runtime = time.perf_counter() - t0

    phi = np.array([
        np.nan if r.bme_phi_mw is None else float(r.bme_phi_mw)
        for r in recs
    ])
    losses = np.array([float(r.total_losses_mw) for r in recs])
    n_last = max(1, int(round(60.0 * 60.0 / cfg.dt_s)))  # last hour
    metrics = {
        "rung": rung,
        "minutes": minutes,
        "steps": len(recs),
        "runtime_s": round(runtime, 1),
        "phi_first_mw": float(phi[0]),
        "phi_last_mw": float(phi[-1]),
        "phi_mean_last_hour_mw": float(np.nanmean(phi[-n_last:])),
        "losses_mean_last_hour_mw": float(np.mean(losses[-n_last:])),
        "band_violation_frac": round(
            _band_violation_fraction(recs), 4,
        ),
        "oltc_switches": _oltc_switch_count(recs),
    }

    # Ledger statistics + export (bme rungs)
    ledger = captured.get("bme_ledger")
    if ledger is not None and len(ledger) > 0:
        entries = ledger.entries()
        metrics["ledger_accepted"] = sum(
            1 for e in entries if e.reason == "accepted"
        )
        metrics["ledger_eps_reject"] = sum(
            1 for e in entries if e.reason == "epsilon_reject"
        )
        metrics["ledger_slot_blocked"] = sum(
            1 for e in entries if e.reason == "slot_blocked"
        )
        led_path = RESULT_DIR / f"ledger_{rung}.csv"
        rows = ledger.to_records()
        with open(led_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"  ledger → {led_path} ({len(rows)} entries)")

    with open(RESULT_DIR / f"records_{rung}.pkl", "wb") as f:
        pickle.dump(recs, f)

    _append_metrics_row(metrics)
    print(f"[{rung}] " + "  ".join(
        f"{k}={v}" for k, v in metrics.items() if k != "rung"
    ))
    return metrics


def _append_metrics_row(metrics: dict) -> None:
    """metrics.csv keeps one row per rung (re-runs replace the row)."""
    path = RESULT_DIR / "metrics.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)]
        rows = [r for r in rows if r.get("rung") != metrics["rung"]]
    rows.append({k: str(v) for k, v in metrics.items()})
    fields = sorted({k for r in rows for k in r})
    ordered = ["rung"] + [k for k in fields if k != "rung"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rung", required=True,
                    choices=list(RUNGS) + ["all"])
    ap.add_argument("--minutes", type=float, default=360.0,
                    help="scenario horizon (default 360 = the 005 "
                         "case-study length)")
    args = ap.parse_args()

    rungs = list(RUNGS) if args.rung == "all" else [args.rung]
    for rung in rungs:
        print(f"=== rung {rung} ({args.minutes:.0f} min) ===")
        run_rung(rung, args.minutes)


if __name__ == "__main__":
    main()
