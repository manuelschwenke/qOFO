"""
011_BME_LADDER — BME evaluation ladder (spec §5 Phase 6, §6).

Rungs (ONE shared scenario definition; only the coordination
configuration differs — spec §6):

  none      (a)  uncoordinated baseline (current non-cooperative behaviour)
  vref      (b)  existing two-loop ΔV_ref tie coordinator
  bme       (c)  Boundary Marginal Exchange (calibrated w_Φ, D2 band on)
  bme_loss  (c') bme with CONTROL w_band = 0 — losses-only ablation (D2;
                 the recorded Φ METRIC keeps the D2 band via
                 ``bme_metric_w_band`` — spec §6 uniform metrics)
  oracle    (d)  single-zone BME oracle (D8 "both"): one MIQP per step
                 over the union actuator universe, exact global dΦ/du by
                 the single-area identity; dispatch stays 3-area

Usage (one command per rung, spec §5 Phase 6 acceptance):

    python experiments/011_BME_LADDER.py --rung none  [--minutes 360]
    python experiments/011_BME_LADDER.py --rung all
    python experiments/011_BME_LADDER.py --plot        # figures only, from pickles

Outputs under ``results/011_BME_LADDER/``: per-rung record pickle,
``metrics.csv`` (appended per rung; recomputed rows replace old ones),
switching-ledger CSV for bme rungs, a printed comparison table, and the
evaluation figures (PNG + PDF; regenerated after every run and standalone
via ``--plot``):

  * ``fig1_phi_losses``       — Φ_global and plant-loss trajectories for
    all rungs with contingency markers and last-hour means: does BME
    lower the common objective vs none, and how close to the oracle;
  * ``fig2_voltage_envelope`` — per-rung system voltage envelope (min/max
    over zones) against the D2 soft band: what the loss gain costs in
    voltage security (exposes the bme_loss escape);
  * ``fig3_discrete``         — cumulative OLTC switching, ledger decision
    breakdown, predicted-vs-realised ΔΦ per accepted switch (§3.10.2);
  * ``fig4_summary``          — last-hour Φ / Δlosses-vs-none / switch
    counts / V-extreme bars per rung, gap-to-oracle annotated once the
    oracle rung's records exist;
  * ``fig5_zone_phi``         — per-zone Φ_i last-hour means + Phulpin
    normalised overcost vs none (net-loser check).

Every figure includes exactly the rungs whose ``records_<rung>.pkl`` is
on disk, so the oracle rung (d) joins all comparisons automatically once
it lands — no plotting change needed then.

Metrics (uniform across rungs, spec §5 Phase 6 / §6):
  * Φ trajectory (``bme_phi_mw`` — recorded for EVERY rung via
    ``record_bme_phi``; identical functional + 3-area partition on all
    rungs), settling mean over the last hour;
  * total network losses (MW, whole combined net — sanity reference);
  * D2-band violation time (fraction of steps with any EHV bus outside
    the soft band, from zone_v_min/max);
  * discrete switch counts per device class (OLTC tap trajectory
    diffs; MIQP shunts when present);
  * bme rungs: ledger statistics (accept / ε-reject / slot-blocked
    counts, predicted-vs-realised ΔΦ pairs — §3.10.2 premise data);
  * derived cross-rung metrics in ``metrics_derived.csv`` (recomputed
    with the figures): gap to oracle (terminal / integral / closure),
    Phulpin normalised overcost per zone (net-loser check), dominant
    AR(2) pole of the boundary-voltage series (oscillation indicator;
    tie-Q proxy for pickles predating ``bme_v_boundary``).

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

# D6 (Phase 6b, 2026-07-03): calibrated from the 360-min bme rung's
# 57-entry switching ledger (spans gen trip/restore, load step, tie-line
# trip), in the w_Φ-SCALED Φ̂ units the gate uses:
#   anchor (b) = median per-step |ΔΦ| on no-commit steps = 1039 scaled
#                (0.0104 MW);
#   ε = 5×(b) = 5193 ≈ the independent sanity cap 0.5×median|ΔΦ̂_prop|
#                = 5256 — the two derivations agree;
#   c_oltc = 1×(b); c_shunt = 5×(b) (breaker wear vs tap wear, mirrors
#                the integrator's stricter treatment of bulk shunts).
BME_EPSILON_SWITCH = 5.2e3
BME_SWITCH_COST_OLTC = 1.0e3
BME_SWITCH_COST_SHUNT = 5.2e3
# D2 pairing (Phase 6c sweep, 2026-07-03, chosen by Manuel): band edges
# centred on the ~1.03 pu operating schedule, w_band = 1e4. REVISED
# 2026-07-05 (Manuel, after the 6e finding): edges tightened
# (1.01, 1.05) → (1.02, 1.04). Under the wide band the coordinated
# rungs rode the upper edge for loss harvest and the realised hinge
# cost exceeded the in-scope loss gain (last-hour Φ ranking inverted,
# none < bme < oracle — BME_STATUS.md §6e). The ±0.01 pu band prices
# edge-riding immediately and keeps φ_band a genuine security margin.
# Full 6c table: docs/BME_STATUS.md §6c. bme_loss stays the w_band = 0
# ablation rung.
BME_W_BAND = 1.0e4
BME_V_SOFT_MIN = 1.02
BME_V_SOFT_MAX = 1.04

RUNGS = ("none", "vref", "bme", "bme_loss", "oracle")


def make_ladder_config(rung: str, minutes: float):
    """The SHARED scenario (005 CIGRE cascade tuning) + the rung's
    coordination configuration. Nothing else may differ between rungs
    (spec §6)."""
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    cfg.record_bme_phi = True          # uniform Φ metric on every rung
    # The Φ METRIC is the IDENTICAL functional on the IDENTICAL 3-area
    # partition on every rung (spec §6): bme_metric_w_band pins the
    # recorded metric even where the CONTROL objective differs — the
    # bme_loss ablation zeroes only the control-side w_band below, and
    # the oracle's single-zone flag collapses only the control
    # partition (runner metric-objective split).
    cfg.bme_w_band = BME_W_BAND
    cfg.bme_metric_w_band = BME_W_BAND
    cfg.bme_v_soft_min_pu = BME_V_SOFT_MIN
    cfg.bme_v_soft_max_pu = BME_V_SOFT_MAX
    cfg.enable_tie_coordination = False
    cfg.coordination_mode = "none"

    if rung == "none":
        pass
    elif rung == "vref":
        cfg.enable_tie_coordination = True
        cfg.coordination_mode = "vref"
    elif rung in ("bme", "bme_loss", "oracle"):
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
        cfg.bme_w_band = 0.0 if rung == "bme_loss" else BME_W_BAND
        # D6 hygiene calibration — identical across the bme-family rungs
        # (bme_loss isolates w_band; oracle isolates the decomposition).
        cfg.bme_epsilon_switch = float(BME_EPSILON_SWITCH)
        cfg.bme_switch_cost_oltc = float(BME_SWITCH_COST_OLTC)
        cfg.bme_switch_cost_shunt = float(BME_SWITCH_COST_SHUNT)
        if rung == "oracle":
            # Rung (d), D8 (single-zone interpretation, Manuel
            # 2026-07-03): ONE zone = union of the 3-area actuator
            # universe, one MIQP per step, global Φ exact (single-area
            # identity, spec §3.5 test 1), no communication; DSO cascade
            # unchanged. Same solver, step logic and hygiene as the
            # distributed bme rung. The V5-style full-set Φ oracle is
            # the separate optional bound (docs/BME_STATUS.md §6d).
            cfg.single_zone_partition = True
    else:
        raise ValueError(f"unknown rung '{rung}' (choose from {RUNGS})")
    return cfg


def _band_violation_fraction(recs, v_lo=BME_V_SOFT_MIN, v_hi=BME_V_SOFT_MAX):
    """Fraction of steps with ANY zone's EHV extreme outside the D2 band."""
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


# ── Derived cross-rung metrics (spec §6 metrics module, item 5) ────────
#
# Computed from the on-disk pickles ONLY (like the figures), so they are
# regenerated by ``--plot`` and always reflect exactly the rungs on
# disk. Written to ``metrics_derived.csv`` + printed as a table.
#
#   * gap to oracle (spec §6): terminal = last-hour Φ means,
#     integral = full-horizon Φ means; plus gap closure
#     100·(Φ_none − Φ_r)/(Φ_none − Φ_oracle) — the spec's headline
#     claim (2) quantifier.
#   * Phulpin normalised overcost (fairness): per zone i,
#     100·(Φ_i^rung − Φ_i^none)/|Φ_i^none| on last-hour means —
#     negative = the zone gains; a positive maximum identifies a net
#     loser. Requires the SAME 3-area metric partition on both rungs
#     (runner metric-objective split guarantees it for new pickles;
#     old single-zone oracle pickles are skipped).
#   * oscillation indicator (spec §6): dominant AR(2) pole (Yule-
#     Walker) per boundary-voltage series over the last-hour window,
#     reported as the maximum modulus over the registry; falls back to
#     the inter-zone tie-Q series for pickles predating
#     ``bme_v_boundary`` (labelled ``tie_q_proxy``).


def _boundary_v_matrix(recs):
    """(T, n_b) boundary-voltage matrix + bus labels from
    ``bme_v_boundary`` (None, [] if the pickles predate the field)."""
    buses = sorted({
        b for r in recs
        for b in (getattr(r, "bme_v_boundary", None) or {})
    })
    if not buses:
        return None, []
    mat = np.full((len(recs), len(buses)), np.nan)
    for i, r in enumerate(recs):
        d = getattr(r, "bme_v_boundary", None) or {}
        for j, b in enumerate(buses):
            if b in d:
                mat[i, j] = d[b]
    return mat, [f"bus{b}" for b in buses]


def _tie_q_matrix(recs):
    """(T, n_pairs) inter-zone tie-Q matrix (recorded unconditionally
    on every rung) — the oscillation-indicator fallback signal."""
    pairs = sorted({p for r in recs for p in r.zone_tie_q_mvar})
    if not pairs:
        return None, []
    mat = np.full((len(recs), len(pairs)), np.nan)
    for i, r in enumerate(recs):
        for j, p in enumerate(pairs):
            if p in r.zone_tie_q_mvar:
                mat[i, j] = r.zone_tie_q_mvar[p]
    return mat, [f"tie{p[0]}-{p[1]}" for p in pairs]


def _ar2_dominant_pole(y):
    """Dominant pole of an AR(2) fit (least-squares covariance method)
    to the mean-removed, linearly detrended series.

    Returns ``(modulus, complex_pair, cycles_per_step)``. Modulus → 1
    with a complex pair = sustained oscillation; small modulus or a
    real pair = damped / drift-dominated. Linear detrending keeps slow
    load-ramp drift from masquerading as a real pole at +1. The
    covariance method (regress y_k on y_{k-1}, y_{k-2}) is exact for
    (damped) sinusoids on short windows, where Yule-Walker's biased
    autocovariances are not."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 16:
        return np.nan, False, np.nan
    t = np.arange(len(y), dtype=float)
    y = y - y.mean()
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    if float(np.dot(y, y)) / len(y) <= 1e-18:   # numerically constant
        return 0.0, False, np.nan
    x = np.column_stack([y[1:-1], y[:-2]])
    (a1, a2), *_ = np.linalg.lstsq(x, y[2:], rcond=None)
    roots = np.roots([1.0, -a1, -a2])
    p = roots[int(np.argmax(np.abs(roots)))]
    complex_pair = bool(abs(p.imag) > 1e-12)
    cycles = (
        float(abs(np.angle(p)) / (2.0 * np.pi)) if complex_pair else 0.0
    )
    return float(abs(p)), complex_pair, cycles


def _oscillation_indicator(recs) -> dict:
    """Max dominant-AR(2)-pole modulus over the boundary signals,
    last-hour window (spec §6). Keys: osc_pole_mod, osc_complex_pair,
    osc_period_min (only for a complex pair), osc_signal."""
    t = _t_min(recs)
    m = _last_hour_mask(t)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    mat, labels = _boundary_v_matrix(recs)
    signal = "boundary_v"
    if mat is None:
        mat, labels = _tie_q_matrix(recs)
        signal = "tie_q_proxy"
    if mat is None:
        return {"osc_pole_mod": np.nan, "osc_complex_pair": False,
                "osc_period_min": np.nan, "osc_signal": "none"}
    best_mod, best_cx, best_cyc, best_lbl = -np.inf, False, np.nan, "?"
    for j, lbl in enumerate(labels):
        mod, cx, cyc = _ar2_dominant_pole(mat[m, j])
        if np.isfinite(mod) and mod > best_mod:
            best_mod, best_cx, best_cyc, best_lbl = mod, cx, cyc, lbl
    if not np.isfinite(best_mod):
        return {"osc_pole_mod": np.nan, "osc_complex_pair": False,
                "osc_period_min": np.nan, "osc_signal": signal}
    period = (
        round(dt / best_cyc, 1)
        if best_cx and best_cyc > 0 and np.isfinite(dt) else np.nan
    )
    return {
        "osc_pole_mod": round(best_mod, 4),
        "osc_complex_pair": best_cx,
        "osc_period_min": period,
        "osc_signal": f"{signal}:{best_lbl}",
    }


def _zone_phi_last_hour(recs) -> dict:
    """zone -> last-hour mean Φ_i (empty for pickles without the field)."""
    t = _t_min(recs)
    m = _last_hour_mask(t)
    zones = sorted({
        z for r in recs
        for z in (getattr(r, "bme_phi_zone_mw", None) or {})
    })
    return {
        z: float(np.nanmean(np.array([
            (getattr(r, "bme_phi_zone_mw", None) or {}).get(z, np.nan)
            for r in recs
        ])[m]))
        for z in zones
    }


def compute_derived_metrics(data: dict) -> list:
    """Cross-rung metric rows (spec §6) from the loaded pickles; writes
    ``metrics_derived.csv`` and prints the comparison table."""
    stats = {}
    for rung, recs in data.items():
        t = _t_min(recs)
        m = _last_hour_mask(t)
        phi = _phi_series(recs)
        stats[rung] = {
            "phi_lh": float(np.nanmean(phi[m])),
            "phi_full": float(np.nanmean(phi)),
            "loss_lh": float(np.mean(_loss_series(recs)[m])),
            "zones": _zone_phi_last_hour(recs),
        }

    none_s = stats.get("none")
    oracle_s = stats.get("oracle")
    rows = []
    for rung, recs in data.items():
        s = stats[rung]
        row = {
            "rung": rung,
            "phi_mean_last_hour_mw": round(s["phi_lh"], 3),
            "phi_mean_full_mw": round(s["phi_full"], 3),
            "losses_mean_last_hour_mw": round(s["loss_lh"], 3),
            "band_violation_frac": round(
                _band_violation_fraction(recs), 4),
            "oltc_switches": _oltc_switch_count(recs),
        }
        if none_s is not None:
            row["loss_red_vs_none_pct"] = round(
                100.0 * (1.0 - s["loss_lh"] / none_s["loss_lh"]), 2)
        # Gap to oracle (terminal + integral) and gap closure.
        if oracle_s is not None and rung != "oracle":
            row["gap_oracle_terminal_pct"] = round(
                100.0 * (s["phi_lh"] / oracle_s["phi_lh"] - 1.0), 3)
            row["gap_oracle_integral_pct"] = round(
                100.0 * (s["phi_full"] / oracle_s["phi_full"] - 1.0), 3)
            if none_s is not None and rung != "none":
                denom = none_s["phi_lh"] - oracle_s["phi_lh"]
                # Closure is only meaningful when the oracle IMPROVES
                # on none (denom > 0). When the realised Φ ranking
                # inverts (e.g. band-hinge cost of edge-riding exceeds
                # the in-scope loss gain), the signed gap columns tell
                # the story and a "closure %" would be nonsense.
                if denom > 1e-9:
                    row["gap_closure_pct"] = round(
                        100.0 * (none_s["phi_lh"] - s["phi_lh"]) / denom,
                        1)
        # Phulpin normalised overcost vs the non-cooperative baseline.
        # Needs the SAME zone set (3-area metric partition) on both
        # rungs — old oracle pickles (single-zone Φ_i) drop out here.
        if (rung != "none" and none_s is not None
                and s["zones"] and none_s["zones"]
                and set(s["zones"]) == set(none_s["zones"])):
            over = {
                z: 100.0 * (s["zones"][z] - none_s["zones"][z])
                / abs(none_s["zones"][z])
                for z in s["zones"]
            }
            for z in sorted(over):
                row[f"overcost_z{z}_pct"] = round(over[z], 2)
            row["overcost_max_pct"] = round(max(over.values()), 2)
        row.update(_oscillation_indicator(recs))
        rows.append(row)

    path = RESULT_DIR / "metrics_derived.csv"
    fields = ["rung"] + sorted(
        {k for r in rows for k in r} - {"rung"})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  derived metrics → {path}")
    print("  " + " | ".join(
        f"{'rung':<8}" if k == "rung" else k
        for k in ("rung", "phi_mean_last_hour_mw", "loss_red_vs_none_pct",
                  "gap_closure_pct", "overcost_max_pct", "osc_pole_mod")))
    for r in rows:
        print("  " + " | ".join([
            f"{r['rung']:<8}",
            f"{r['phi_mean_last_hour_mw']:>21.3f}",
            f"{r.get('loss_red_vs_none_pct', float('nan')):>20.2f}",
            f"{r.get('gap_closure_pct', float('nan')):>15.1f}",
            f"{r.get('overcost_max_pct', float('nan')):>16.2f}",
            f"{r.get('osc_pole_mod', float('nan')):>12.4f}",
        ]))
    return rows


# ── Evaluation figures (spec §5 Phase 6 / Phase 7 analysis artefacts) ──
#
# Colour identity is FIXED per rung (a missing rung never repaints the
# others); the oracle is the black dashed reference bound. Categorical
# hues follow the validated data-viz palette; chrome/annotation text
# stays in ink greys. Figures are built from the on-disk pickles only,
# so ``--plot`` never re-simulates.

RUNG_STYLE = {
    # rung: (colour, linestyle, display label)
    "none":     ("#52514e", "-",  "none (uncoordinated)"),
    "vref":     ("#1baf7a", "-",  "vref (ΔV_ref coordinator)"),
    "bme":      ("#2a78d6", "-",  "bme (Φ, D2 band)"),
    "bme_loss": ("#eb6834", "-",  "bme_loss (Φ losses-only)"),
    "oracle":   ("#0b0b0b", "--", "oracle (central Φ, D8)"),
}
PLOT_RUNG_ORDER = ("none", "vref", "bme", "bme_loss", "oracle")
_INK, _SEC, _MUTED = "#0b0b0b", "#52514e", "#898781"
_GRID, _BASE = "#e1e0d9", "#c3c2b7"


def _plt():
    """Headless matplotlib with the ladder's chrome defaults."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": _BASE,
        "axes.labelcolor": _SEC,
        "axes.titlecolor": _INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.6,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt


def _save_fig(plt, fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  figure → {RESULT_DIR / (name + '.png')} (+ .pdf)")


def _load_ladder_records() -> dict:
    """rung -> record list for every rung with a pickle on disk,
    in fixed plot order (oracle joins automatically once it lands)."""
    data = {}
    for rung in PLOT_RUNG_ORDER:
        path = RESULT_DIR / f"records_{rung}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data[rung] = pickle.load(f)
    return data


def _t_min(recs):
    return np.array([float(r.time_s) for r in recs]) / 60.0


def _phi_series(recs):
    return np.array([
        np.nan if r.bme_phi_mw is None else float(r.bme_phi_mw)
        for r in recs
    ])


def _loss_series(recs):
    return np.array([float(r.total_losses_mw) for r in recs])


def _v_envelope(recs):
    vmin = np.array([
        min(r.zone_v_min.values()) if r.zone_v_min else np.nan
        for r in recs
    ])
    vmax = np.array([
        max(r.zone_v_max.values()) if r.zone_v_max else np.nan
        for r in recs
    ])
    return vmin, vmax


def _rolling_mean(y, w: int):
    """Centred moving average, edge-renormalised by actual coverage."""
    if w <= 1 or len(y) < w:
        return np.asarray(y, dtype=float).copy()
    kernel = np.ones(w)
    out = np.convolve(np.nan_to_num(y, nan=0.0), kernel, mode="same")
    cover = np.convolve(np.isfinite(y).astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / cover


def _last_hour_mask(t_min):
    return t_min >= (t_min[-1] - 60.0)


def _spread_positions(values, min_sep: float):
    """Label positions >= min_sep apart, order-preserving (collision
    avoidance for right-margin annotations of near-tied series)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    pos = values[order].copy()
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_sep)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _smoothing_window(t_min, minutes: float = 15.0) -> int:
    dt = float(np.median(np.diff(t_min))) if len(t_min) > 1 else 1.0
    return max(1, int(round(minutes / dt)))


def _cum_switch_series(recs):
    """Cumulative whole-tap OLTC moves vs time (all zones)."""
    per_step = np.zeros(len(recs))
    prev = {}
    for i, r in enumerate(recs):
        moved = 0
        for z, taps in r.zone_oltc_taps.items():
            taps = np.asarray(taps, dtype=float)
            if z in prev and prev[z].shape == taps.shape:
                moved += int(np.sum(np.abs(np.round(taps - prev[z]))))
            prev[z] = taps
        per_step[i] = moved
    return np.cumsum(per_step)


def _scenario_events(horizon_min: float):
    """(minute, short label) for the shared 005 contingency schedule,
    restricted to the plotted horizon."""
    cfg = _005.make_cigre_config()
    events = []
    for ev in getattr(cfg, "contingencies", None) or []:
        t = ev.effective_time_s / 60.0
        if not (0.0 < t < horizon_min):
            continue
        if ev.element_type == "load":
            what = (f"load {ev.p_mw:+.0f} MW/{ev.q_mvar:+.0f} Mvar"
                    if ev.element_index < 0 else f"load {ev.element_index}")
        else:
            what = f"{ev.element_type} {ev.element_index}"
        events.append((t, f"{what} {ev.action}"))
    return events


def _ledger_decision_counts(rung: str):
    """reason -> count from the exported ledger CSV (None if absent)."""
    path = RESULT_DIR / f"ledger_{rung}.csv"
    if not path.exists():
        return None
    counts = {"accepted": 0, "epsilon_reject": 0, "slot_blocked": 0}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["reason"] in counts:
                counts[row["reason"]] += 1
    return counts


def _ledger_pred_real_pairs(rung: str):
    """(predicted ΔΦ̂ [MW], realised ΔΦ [MW]) of ACCEPTED switches.
    Predictions are stored in w_Φ-scaled units (D6) — converted here;
    realised is the whole-round Φ_global difference (§3.10.2)."""
    path = RESULT_DIR / f"ledger_{rung}.csv"
    if not path.exists():
        return np.empty(0), np.empty(0)
    pred, real = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["reason"] != "accepted":
                continue
            try:
                p = float(row["predicted_dphi"]) / BME_GRADIENT_SCALE
                r = float(row["realised_dphi"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(p) and np.isfinite(r):
                pred.append(p)
                real.append(r)
    return np.array(pred), np.array(real)


def _mark_events(ax, events, label_panel: bool):
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData,
                                                  ax.transAxes)
    for t_ev, lbl in events:
        ax.axvline(t_ev, color=_BASE, lw=0.8, zorder=1)
        if label_panel:
            ax.text(t_ev, 0.98, lbl, transform=trans, rotation=90,
                    va="top", ha="right", fontsize=6.5, color=_MUTED)


def fig1_phi_losses(data: dict) -> None:
    """Φ_global(t) and plant losses(t), all rungs (15-min rolling
    mean over faint raw traces); last hour shaded, means annotated."""
    plt = _plt()
    fig, (ax_phi, ax_loss) = plt.subplots(
        2, 1, figsize=(9.2, 6.6), sharex=True, constrained_layout=True)
    horizon = max(_t_min(recs)[-1] for recs in data.values())
    events = _scenario_events(horizon)

    means = {ax_phi: [], ax_loss: []}
    for rung, recs in data.items():
        col, ls, label = RUNG_STYLE[rung]
        t = _t_min(recs)
        w = _smoothing_window(t)
        m = _last_hour_mask(t)
        for ax, y in ((ax_phi, _phi_series(recs)),
                      (ax_loss, _loss_series(recs))):
            ax.plot(t, y, color=col, ls=ls, lw=0.7, alpha=0.22)
            ax.plot(t, _rolling_mean(y, w), color=col, ls=ls, lw=2.0,
                    label=label)
            means[ax].append((t[-1], float(np.nanmean(y[m]))))

    for ax in (ax_phi, ax_loss):
        ax.axvspan(horizon - 60.0, horizon, color="#f0efec", zorder=0)
        ax.margins(x=0.02)
        # last-hour means at the right margin, staggered so near-ties
        # (none vs vref) stay legible
        vals = np.array([v for _, v in means[ax]])
        y0, y1 = ax.get_ylim()
        ypos = _spread_positions(vals, 0.045 * (y1 - y0))
        for (t_end, v), yp in zip(means[ax], ypos):
            ax.annotate(f"{v:.1f}", xy=(t_end, yp),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8, color=_SEC, va="center")
    _mark_events(ax_phi, events, label_panel=True)
    _mark_events(ax_loss, events, label_panel=False)

    ax_phi.set_ylabel("Φ_global [MW]")
    ax_loss.set_ylabel("total plant losses [MW]")
    ax_loss.set_xlabel("time [min]")
    ax_phi.set_title(
        "BME ladder — Φ and plant losses (15-min rolling mean; "
        "shading / right labels = last-hour window / means)",
        fontsize=10, loc="left")
    ax_phi.legend(loc="upper left", fontsize=8, ncol=2, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.85)
    _save_fig(plt, fig, "fig1_phi_losses")


def fig2_voltage_envelope(data: dict) -> None:
    """Small multiples: per-rung system V envelope (min/max over
    zones) vs the D2 soft band — shared y-scale for comparability."""
    plt = _plt()
    rungs = list(data)
    ncols = 2
    nrows = (len(rungs) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.2, 2.7 * nrows),
        sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    horizon = max(_t_min(recs)[-1] for recs in data.values())
    events = _scenario_events(horizon)

    for ax, rung in zip(axes, rungs):
        recs = data[rung]
        col, _, label = RUNG_STYLE[rung]
        t = _t_min(recs)
        vmin, vmax = _v_envelope(recs)
        ax.fill_between(t, vmin, vmax, color=col, alpha=0.28, lw=0)
        ax.plot(t, vmax, color=col, lw=1.3)
        ax.plot(t, vmin, color=col, lw=1.3)
        for edge in (BME_V_SOFT_MIN, BME_V_SOFT_MAX):
            ax.axhline(edge, color=_MUTED, lw=0.9, ls="--", zorder=1)
        _mark_events(ax, events, label_panel=False)
        ax.set_title(
            f"{label} — V ∈ [{np.nanmin(vmin):.3f}, "
            f"{np.nanmax(vmax):.3f}] pu", fontsize=9, loc="left")
        ax.margins(x=0.02)
    for ax in axes[len(rungs):]:
        ax.set_visible(False)
    for k in range(len(rungs)):
        if k % ncols == 0:
            axes[k].set_ylabel("EHV bus voltage [pu]")
        if k // ncols == nrows - 1:
            axes[k].set_xlabel("time [min]")
    axes[0].annotate(
        f"D2 soft band [{BME_V_SOFT_MIN:.2f}, {BME_V_SOFT_MAX:.2f}] pu",
        xy=(0.01, 0.03), xycoords="axes fraction",
        fontsize=7.5, color=_MUTED)
    fig.suptitle("BME ladder — system voltage envelope "
                 "(min/max over all zones' EHV buses)",
                 fontsize=10, x=0.01, ha="left")
    _save_fig(plt, fig, "fig2_voltage_envelope")


def fig3_discrete(data: dict) -> None:
    """Discrete hygiene: cumulative OLTC switching (all rungs), ledger
    decision breakdown and predicted-vs-realised ΔΦ (bme rungs)."""
    plt = _plt()
    fig = plt.figure(figsize=(9.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_cum = fig.add_subplot(gs[0, :])
    ax_led = fig.add_subplot(gs[1, 0])
    ax_sca = fig.add_subplot(gs[1, 1])
    horizon = max(_t_min(recs)[-1] for recs in data.values())
    _mark_events(ax_cum, _scenario_events(horizon), label_panel=False)

    for rung, recs in data.items():
        col, ls, label = RUNG_STYLE[rung]
        t = _t_min(recs)
        cum = _cum_switch_series(recs)
        ax_cum.step(t, cum, where="post", color=col, ls=ls, lw=2.0,
                    label=f"{label} ({int(cum[-1])})")
    ax_cum.set_ylabel("cumulative OLTC tap moves")
    ax_cum.set_xlabel("time [min]")
    ax_cum.set_title("Discrete activity — cumulative whole-tap OLTC "
                     "moves (totals in legend)", fontsize=10, loc="left")
    ax_cum.legend(loc="upper left", fontsize=8)
    ax_cum.margins(x=0.02)

    # Ledger decision breakdown (grouped horizontal bars per bme rung).
    reasons = (("accepted", "#008300"),
               ("epsilon_reject", "#eda100"),
               ("slot_blocked", _MUTED))
    led_rungs = [r for r in data
                 if _ledger_decision_counts(r) is not None]
    if led_rungs:
        h = 0.8 / len(reasons)
        for i, rung in enumerate(led_rungs):
            counts = _ledger_decision_counts(rung)
            for j, (reason, rcol) in enumerate(reasons):
                bar = ax_led.barh(i + (j - 1) * h, counts[reason],
                                  height=h * 0.9, color=rcol,
                                  label=reason if i == 0 else None)
                ax_led.bar_label(bar, padding=3, fontsize=7.5,
                                 color=_SEC)
        ax_led.set_yticks(range(len(led_rungs)),
                          labels=led_rungs)
        ax_led.invert_yaxis()
        ax_led.margins(x=0.12)
        ax_led.legend(fontsize=7.5, loc="upper right", frameon=True,
                      facecolor="white", edgecolor="none",
                      framealpha=0.85)
        ax_led.set_xlabel("ledger entries")
        ax_led.set_title("Hygiene-gate decisions (§3.8.3)",
                         fontsize=10, loc="left")
    else:
        ax_led.set_visible(False)

    # Predicted vs realised ΔΦ per ACCEPTED switch (§3.10.2).
    any_pairs = False
    for rung in led_rungs:
        pred, real = _ledger_pred_real_pairs(rung)
        if len(pred) == 0:
            continue
        any_pairs = True
        col = RUNG_STYLE[rung][0]
        agree = float(np.mean(np.sign(pred) == np.sign(real)))
        ax_sca.plot(pred, real, "o", ms=5, mec="white", mew=0.6,
                    color=col, alpha=0.85,
                    label=f"{rung} (sign agree {agree:.2f})")
    if any_pairs:
        lims = np.array(ax_sca.get_xlim() + ax_sca.get_ylim())
        lo, hi = float(np.min(lims)), float(np.max(lims))
        ax_sca.plot([lo, hi], [lo, hi], color=_BASE, lw=1.0, zorder=1)
        ax_sca.axhline(0.0, color=_GRID, lw=0.8)
        ax_sca.axvline(0.0, color=_GRID, lw=0.8)
        ax_sca.set_xlabel("predicted ΔΦ̂ per accepted switch [MW]")
        ax_sca.set_ylabel("realised ΔΦ (whole round) [MW]")
        ax_sca.set_title("Switch-prediction premise data (§3.10.2)",
                         fontsize=10, loc="left")
        ax_sca.legend(fontsize=7.5, loc="upper left")
    else:
        ax_sca.set_visible(False)
    _save_fig(plt, fig, "fig3_discrete")


def fig4_summary(data: dict) -> None:
    """The headline dashboard: last-hour Φ, Δlosses vs none, switch
    counts, V extremes — one horizontal bar row per rung; gap-to-oracle
    annotated when the oracle records exist."""
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 0.75 * len(data) + 2.2),
                             constrained_layout=True)
    ax_phi, ax_dls, ax_sw, ax_v = axes
    rungs = list(data)
    ypos = np.arange(len(rungs))
    cols = [RUNG_STYLE[r][0] for r in rungs]

    phi_mean, loss_mean, switches, v_lo, v_hi = {}, {}, {}, {}, {}
    for rung, recs in data.items():
        t = _t_min(recs)
        m = _last_hour_mask(t)
        phi_mean[rung] = float(np.nanmean(_phi_series(recs)[m]))
        loss_mean[rung] = float(np.nanmean(_loss_series(recs)[m]))
        switches[rung] = int(_cum_switch_series(recs)[-1])
        vmin, vmax = _v_envelope(recs)
        v_lo[rung], v_hi[rung] = float(np.nanmin(vmin)), float(np.nanmax(vmax))

    bars = ax_phi.barh(ypos, [phi_mean[r] for r in rungs], color=cols,
                       height=0.62)
    if "oracle" in phi_mean:
        for i, r in enumerate(rungs):
            if r == "oracle" or phi_mean["oracle"] == 0.0:
                continue
            gap = 100.0 * (phi_mean[r] / phi_mean["oracle"] - 1.0)
            ax_phi.annotate(f"{gap:+.1f} % vs oracle",
                            xy=(phi_mean[r], i), xytext=(4, 10),
                            textcoords="offset points",
                            fontsize=7, color=_SEC)
    ax_phi.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color=_SEC)
    ax_phi.set_title("Φ mean, last hour [MW]", fontsize=9.5, loc="left")

    if "none" in loss_mean:
        # reduction (positive = better) so the bars grow rightwards and
        # value labels never collide with the y tick labels
        dvals = [100.0 * (1.0 - loss_mean[r] / loss_mean["none"])
                 for r in rungs]
        bars = ax_dls.barh(ypos, dvals, color=cols, height=0.62)
        ax_dls.bar_label(bars, fmt="%+.1f %%", padding=3, fontsize=8,
                         color=_SEC)
        ax_dls.axvline(0.0, color=_BASE, lw=1.0)
        ax_dls.margins(x=0.18)
        ax_dls.set_title("loss reduction vs none, last hour",
                         fontsize=9.5, loc="left")
    else:
        bars = ax_dls.barh(ypos, [loss_mean[r] for r in rungs],
                           color=cols, height=0.62)
        ax_dls.bar_label(bars, fmt="%.1f", padding=3, fontsize=8,
                         color=_SEC)
        ax_dls.set_title("losses mean, last hour [MW]",
                         fontsize=9.5, loc="left")

    bars = ax_sw.barh(ypos, [switches[r] for r in rungs], color=cols,
                      height=0.62)
    ax_sw.bar_label(bars, padding=3, fontsize=8, color=_SEC)
    ax_sw.set_title("OLTC tap moves (whole run)", fontsize=9.5,
                    loc="left")

    for i, r in enumerate(rungs):
        ax_v.plot([v_lo[r], v_hi[r]], [i, i], color=RUNG_STYLE[r][0],
                  lw=3.0, solid_capstyle="round")
        ax_v.plot([v_lo[r], v_hi[r]], [i, i], "o", ms=5,
                  color=RUNG_STYLE[r][0])
        ax_v.annotate(f"{v_hi[r]:.3f}", xy=(v_hi[r], i), xytext=(5, 0),
                      textcoords="offset points", fontsize=7.5,
                      color=_SEC, va="center")
    for edge in (BME_V_SOFT_MIN, BME_V_SOFT_MAX):
        ax_v.axvline(edge, color=_MUTED, lw=0.9, ls="--", zorder=1)
    ax_v.set_title("V range, whole run [pu]", fontsize=9.5, loc="left")
    ax_v.margins(x=0.18)

    for k, ax in enumerate(axes):
        ax.set_yticks(ypos, labels=rungs if k == 0 else [""] * len(rungs))
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    fig.suptitle("BME ladder — last-hour summary "
                 "(Φ = common objective; dashed = D2 soft band "
                 f"[{BME_V_SOFT_MIN:.2f}, {BME_V_SOFT_MAX:.2f}] pu)",
                 fontsize=10, x=0.01, ha="left")
    _save_fig(plt, fig, "fig4_summary")


def fig5_zone_phi(data: dict) -> None:
    """Per-zone Φ_i last-hour means (top) and the Phulpin normalised
    overcost vs the non-cooperative baseline (bottom): who pays for the
    common-objective gain, and is any TSO a net loser."""
    # getattr fallback inside the helpers: pickles written before the
    # bme_phi_zone_mw record field was added restore without it.
    zone_means = {
        rung: zm for rung, recs in data.items()
        if (zm := _zone_phi_last_hour(recs))
    }
    if not zone_means:
        print("  fig5_zone_phi skipped (records carry no "
              "bme_phi_zone_mw — re-run the rungs to populate it)")
        return

    none_zm = zone_means.get("none")
    over_rungs = {
        rung: zm for rung, zm in zone_means.items()
        if rung != "none" and none_zm
        and set(zm) == set(none_zm)
    }

    plt = _plt()
    nrows = 2 if over_rungs else 1
    fig, axes = plt.subplots(
        nrows, 1, figsize=(7.4, 3.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    zones = sorted({z for zm in zone_means.values() for z in zm})
    x = np.arange(len(zones), dtype=float)
    width = 0.8 / len(zone_means)
    for j, (rung, zm) in enumerate(zone_means.items()):
        col = RUNG_STYLE[rung][0]
        vals = [zm.get(z, np.nan) for z in zones]
        bars = ax.bar(x + (j - (len(zone_means) - 1) / 2) * width, vals,
                      width * 0.9, color=col, label=rung)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7,
                     color=_SEC)
    ax.set_xticks(x, labels=[f"zone {z}" for z in zones])
    ax.set_ylabel("Φ_i mean, last hour [MW]")
    ax.set_title("Per-zone common-objective share (D1 ownership; "
                 "Σ_i Φ_i = Φ_global)", fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=len(zone_means))
    ax.grid(axis="x", visible=False)

    if over_rungs:
        ax2 = axes[1]
        width = 0.8 / len(over_rungs)
        for j, (rung, zm) in enumerate(over_rungs.items()):
            col = RUNG_STYLE[rung][0]
            vals = [
                100.0 * (zm[z] - none_zm[z]) / abs(none_zm[z])
                for z in zones
            ]
            bars = ax2.bar(
                x + (j - (len(over_rungs) - 1) / 2) * width, vals,
                width * 0.9, color=col, label=rung)
            ax2.bar_label(bars, fmt="%+.1f", padding=2, fontsize=7,
                          color=_SEC)
        ax2.axhline(0.0, color=_BASE, lw=1.0)
        ax2.set_xticks(x, labels=[f"zone {z}" for z in zones])
        ax2.set_ylabel("normalised overcost vs none [%]")
        ax2.set_title(
            "Phulpin fairness — 100·(Φ_i − Φ_i^none)/|Φ_i^none| "
            "(negative = the zone gains; a positive bar = net loser)",
            fontsize=10, loc="left")
        ax2.legend(fontsize=8, ncol=len(over_rungs))
        ax2.grid(axis="x", visible=False)
        ax2.margins(y=0.15)
    _save_fig(plt, fig, "fig5_zone_phi")


def fig0_concepts() -> None:
    """Concept schematic of the five rungs (spec §6): what information
    crosses zone boundaries and which objective each zone descends —
    everything else (scenario, solver, step logic, hygiene) identical.
    Static figure: needs no pickles, regenerates with ``--plot``."""
    plt = _plt()
    import matplotlib.patches as mpatches

    Z_POS = {1: (0.22, 0.66), 2: (0.78, 0.66), 3: (0.50, 0.20)}
    R = 0.13

    def _zone_circles(ax, accent, obj_label):
        for (zi, zj) in ((1, 2), (2, 3), (1, 3)):
            (x1, y1), (x2, y2) = Z_POS[zi], Z_POS[zj]
            ax.plot([x1, x2], [y1, y2], color=_BASE, lw=1.4, zorder=1)
        for z, (x, y) in Z_POS.items():
            ax.add_patch(mpatches.Circle(
                (x, y), R, facecolor="white", edgecolor=accent,
                lw=1.8, zorder=2))
            ax.text(x, y + 0.035, f"TSO {z}", ha="center", va="center",
                    fontsize=8.5, color=_INK, zorder=3)
            ax.text(x, y - 0.045, obj_label, ha="center", va="center",
                    fontsize=7.0, color=_SEC, zorder=3)

    def _setup(ax, rung, subtitle):
        col, _, label = RUNG_STYLE[rung]
        ax.set_title(f"({rung}) {label.split('(')[-1].rstrip(')')}",
                     fontsize=9.5, loc="left", color=_INK)
        ax.text(0.0, 1.005, "", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, -0.02, subtitle, ha="center", va="top",
                fontsize=7.2, color=_MUTED, wrap=True,
                transform=ax.transAxes)
        return col

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0),
                             constrained_layout=True)
    ax_a, ax_b, ax_c, ax_cp, ax_d, ax_s = axes.ravel()

    # (a) none — private objectives, no exchange.
    col = _setup(ax_a, "none",
                 "no information exchange — zones interact only\n"
                 "through the plant (non-cooperative game)")
    _zone_circles(ax_a, col, "min $J_i$ private")

    # (b) vref — pairwise ΔV_ref agreement on each tie.
    col = _setup(ax_b, "vref",
                 "pairwise ΔV_ref agreement per tie (two-loop,\n"
                 "outer gradient + inner tracking); objectives stay private")
    _zone_circles(ax_b, col, "min $J_i$ private")
    for (zi, zj) in ((1, 2), (2, 3), (1, 3)):
        (x1, y1), (x2, y2) = Z_POS[zi], Z_POS[zj]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax_b.annotate("", xy=(xm + 0.06 * (x2 - x1), ym + 0.06 * (y2 - y1)),
                      xytext=(xm - 0.06 * (x2 - x1), ym - 0.06 * (y2 - y1)),
                      arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
    ax_b.text(0.5, 0.72, r"$\Delta V_\mathrm{ref}$", ha="center",
              fontsize=8, color=col)

    # (c) bme — common Φ, boundary marginals broadcast.
    col = _setup(ax_c, "bme",
                 "ONE common Φ = Σ Φ_i (w_loss·P_loss + φ_band, D1/D2);\n"
                 "boundary marginals μ_j exchanged (delay d, filter β) —\n"
                 "each zone descends the EXACT global dΦ/du_i")
    _zone_circles(ax_c, col, r"min $\Phi$ (common)")
    xb, yb = 0.50, 0.55
    ax_c.add_patch(mpatches.FancyBboxPatch(
        (xb - 0.09, yb - 0.035), 0.18, 0.07,
        boxstyle="round,pad=0.012", facecolor="white",
        edgecolor=col, lw=1.4, zorder=4))
    ax_c.text(xb, yb, "bus", ha="center", va="center", fontsize=7.5,
              color=col, zorder=5)
    for z, (x, y) in Z_POS.items():
        dx, dy = xb - x, yb - y
        n = (dx * dx + dy * dy) ** 0.5
        sx, sy = x + dx / n * R, y + dy / n * R
        ex, ey = xb - dx / n * 0.105, yb - dy / n * 0.055
        ax_c.annotate("", xy=(ex, ey), xytext=(sx, sy),
                      arrowprops=dict(arrowstyle="<->", color=col,
                                      lw=1.3, alpha=0.85), zorder=3)
    ax_c.text(xb + 0.13, yb + 0.05, r"$\mu_i$", fontsize=9, color=col)

    # (c') bme_loss — ablation: control drops the band term.
    col = _setup(ax_cp, "bme_loss",
                 "ablation: CONTROL w_band = 0 (losses-only descent) —\n"
                 "isolates what the D2 band buys; recorded METRIC keeps\n"
                 "the band (uniform functional across rungs)")
    _zone_circles(ax_cp, col, r"min $P_\mathrm{loss}$ only")
    ax_cp.text(0.5, 0.55, r"$w_\mathrm{band}=0$" + "\n(V escape expected)",
               ha="center", va="center", fontsize=7.5, color=col)

    # (d) oracle — one central MIQP over the union actuator set.
    col = _setup(ax_d, "oracle",
                 "decomposition bound (D8): ONE MIQP per step over the\n"
                 "union actuator universe, exact global dΦ/du (single-area\n"
                 "identity); no communication; dispatch stays 3-area")
    _zone_circles(ax_d, col, r"min $\Phi$ (common)")
    ax_d.add_patch(mpatches.FancyBboxPatch(
        (0.06, 0.06), 0.88, 0.82, boxstyle="round,pad=0.015",
        facecolor="none", edgecolor=col, lw=1.6, ls="--", zorder=1))
    ax_d.text(0.5, 0.93, "one central controller", ha="center",
              fontsize=8, color=col)

    # (s) shared-scenario panel — the ladder's design invariant.
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
    ax_s.set_title("shared across ALL rungs (spec §6)", fontsize=9.5,
                   loc="left", color=_INK)
    ax_s.text(0.02, 0.92, (
        "• ONE scenario: IEEE 39 3-area + CIGRE HV cascades,\n"
        "   same contingencies, same 3-area P dispatch\n"
        "• same MIQP solver, step logic, $G_w$ blocks\n"
        "• same discrete hygiene (D6 ε-acceptance, slotting,\n"
        "   switch costs) on the bme-family rungs\n"
        "• same DSO cascade underneath every zone\n"
        "• same RECORDED metric: Φ functional + 3-area\n"
        "   partition + D2 band on every rung\n\n"
        "⇒ only the coordination configuration differs;\n"
        "   every performance delta is attributable to it"),
        va="top", fontsize=7.8, color=_SEC, linespacing=1.55)

    fig.suptitle("BME evaluation ladder — coordination concepts per rung "
                 "(what crosses the zone boundary, and which objective "
                 "each zone descends)", fontsize=10.5, x=0.01, ha="left")
    _save_fig(plt, fig, "fig0_concepts")


def make_all_figures() -> None:
    """Render every ladder figure from the on-disk record pickles."""
    data = _load_ladder_records()
    if not data:
        print(f"no records_<rung>.pkl under {RESULT_DIR} — "
              "run rungs first")
        return
    print(f"figures over rungs: {', '.join(data)}"
          + ("" if "oracle" in data else "  (oracle rung not on disk "
             "yet — gap-to-oracle deferred)"))
    compute_derived_metrics(data)
    fig0_concepts()
    fig1_phi_losses(data)
    fig2_voltage_envelope(data)
    fig3_discrete(data)
    fig4_summary(data)
    fig5_zone_phi(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rung", required=False, default=None,
                    choices=list(RUNGS) + ["all"])
    ap.add_argument("--minutes", type=float, default=360.0,
                    help="scenario horizon (default 360 = the 005 "
                         "case-study length)")
    ap.add_argument("--plot", action="store_true",
                    help="(re)build the evaluation figures from the "
                         "on-disk pickles; with --rung, figures are "
                         "regenerated after the runs anyway")
    args = ap.parse_args()
    if args.rung is None and not args.plot:
        ap.error("--rung is required unless --plot is given")

    if args.rung is not None:
        rungs = list(RUNGS) if args.rung == "all" else [args.rung]
        for rung in rungs:
            print(f"=== rung {rung} ({args.minutes:.0f} min) ===")
            run_rung(rung, args.minutes)

    make_all_figures()


if __name__ == "__main__":
    main()
