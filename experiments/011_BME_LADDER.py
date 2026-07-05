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
  * ``fig5_zone_phi``         — per-zone Φ_i last-hour means (Phulpin
    fairness premise data).

Every figure includes exactly the rungs whose ``records_<rung>.pkl`` is
on disk, so the oracle rung (d) joins all comparisons automatically once
it lands — no plotting change needed then.

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
# centred on the ~1.03 pu operating schedule, (1.01, 1.05), w_band = 1e4.
# Nine-point sweep at the 120-min horizon: this pairing gave −5.4 %
# last-hour losses (vs −3.9 % for the spec-default (0.97, 1.03)×1e3),
# V ∈ [1.002, 1.062] with the best post-trip voltage support (lower
# hinge lifts the dip to 1.002 vs baseline 0.978) and the healthiest
# solve times. Full table: docs/BME_STATUS.md §6c. bme_loss stays the
# w_band = 0 ablation rung.
BME_W_BAND = 1.0e4
BME_V_SOFT_MIN = 1.01
BME_V_SOFT_MAX = 1.05

RUNGS = ("none", "vref", "bme", "bme_loss", "oracle")


def make_ladder_config(rung: str, minutes: float):
    """The SHARED scenario (005 CIGRE cascade tuning) + the rung's
    coordination configuration. Nothing else may differ between rungs
    (spec §6)."""
    cfg = _005.make_cigre_config()
    cfg.n_total_s = 60.0 * minutes
    cfg.verbose = 0
    cfg.record_bme_phi = True          # uniform Φ metric on every rung
    # The Φ METRIC must be the identical functional on every rung
    # (same D2 band definition); for non-bme rungs these fields affect
    # recording only. The bme_loss CONTROL ablation is realised below by
    # zeroing w_band for the bme gradient — its recorded Φ metric then
    # differs by definition (documented; compare it on losses).
    cfg.bme_w_band = BME_W_BAND
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
    """Per-zone Φ_i last-hour means, grouped by rung (Phulpin fairness
    premise: who pays for the common-objective gain)."""
    # getattr fallback: pickles written before the bme_phi_zone_mw
    # record field was added restore without the attribute.
    zone_means = {}
    for rung, recs in data.items():
        t = _t_min(recs)
        m = _last_hour_mask(t)
        zones = sorted({
            z for r in recs
            for z in (getattr(r, "bme_phi_zone_mw", None) or {})
        })
        if not zones:
            continue
        zone_means[rung] = {
            z: float(np.nanmean(np.array([
                (getattr(r, "bme_phi_zone_mw", None) or {}).get(z, np.nan)
                for r in recs
            ])[m]))
            for z in zones
        }
    if not zone_means:
        print("  fig5_zone_phi skipped (records carry no "
              "bme_phi_zone_mw — re-run the rungs to populate it)")
        return

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
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
                 "Σ_i Φ_i = Φ_global) — Phulpin fairness premise",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=len(zone_means))
    ax.grid(axis="x", visible=False)
    _save_fig(plt, fig, "fig5_zone_phi")


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
