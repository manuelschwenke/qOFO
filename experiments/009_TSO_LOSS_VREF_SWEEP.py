#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/009_TSO_LOSS_TIE_SWEEP.py
=====================================
Simple comparison of the TSO transmission-loss objective (``tso_g_loss``,
form B — current-magnitude) across a small weight sweep, crossed with the
horizontal TSO-TSO tie coordinator OFF / ON.

Design (2 factors)
------------------
* **Loss weight** ``tso_g_loss`` ∈ :data:`LOSS_WEIGHTS` (includes ``0.0`` =
  loss term off — the baseline).  The term adds
  ``tso_g_loss · Σ_ℓ (3·R_ℓ)·|I_ℓ|²`` per zone, summed over each zone's
  monitored EHV current lines, realised through the cached ``∂I/∂u`` rows of
  the sensitivity matrix (no new sensitivity).  See
  :attr:`controller.tso_controller.TSOControllerConfig.g_loss`.
* **Tie coordinator** ``enable_tie_coordination`` ∈ {False, True}
  (:class:`controller.tie_coordinator.HorizontalTieCoordinator`).  The
  gradient-exchange ΔV_ref scheme that redirects boundary-bus voltage
  setpoints between adjacent zones.

The cross product gives ``len(LOSS_WEIGHTS) × 2`` scenarios.  Everything else
(grid, profile, weights, timing, cascade OFO at both layers) is held fixed by
:func:`make_base_config`, so the only varied factors are the loss weight and
the tie toggle → a fair comparison.

Ground-truth metric
--------------------
The primary metric is :attr:`MultiTSOIterationRecord.total_losses_mw` — the
whole-network active loss ``Σ res_line.pl_mw + res_trafo.pl_mw +
res_trafo3w.pl_mw`` from the converged power flow at every step.  This is the
*plant* loss, independent of the controller's internal (monitored-line) loss
proxy, so it is a fair scorer of whether minimising the proxy actually lowers
real losses.  Secondary metrics (voltage tracking RMSE, V envelope, mean tie
reactive exchange) expose the loss-vs-voltage trade-off and the coordinator's
effect.

Outputs (under ``results/009_loss_tie_sweep/``)
-----------------------------------------------
* ``<scenario>/log.pkl``            -- pickled record list per scenario.
* ``summary.csv``                   -- one row per scenario.
* ``losses_vs_gloss.{png,pdf}``     -- mean plant loss vs weight, tie off/on.
* ``losses_timeseries.{png,pdf}``   -- loss trajectories, all scenarios.
* ``tie_q_vs_gloss.{png,pdf}``      -- mean |tie Q| vs weight, tie off/on.

CLI flags
---------
* ``--replot``  -- skip simulation, rebuild summary + figures from pickles.
* ``--only g0_tieOff,g1e4_tieOn``  -- run only the named scenarios.
* ``--smoke``   -- tiny 2-step horizon for a fast wiring check.

NOTE on weight scale: the absolute ``tso_g_loss`` that "bites" is scenario-
dependent (it competes with ``g_v`` and the ``g_w`` move penalties through the
relative magnitudes of ``∂I/∂u`` vs ``∂V/∂u``).  If the sweep shows a
negligible or over-strong effect, widen / shift :data:`LOSS_WEIGHTS`.

Author: Manuel Schwenke / Claude Code
Date: 2026-06-30
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Project root on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import MultiTSOIterationRecord
from experiments.paths import results_path
from experiments.runners import run_multi_tso_dso


# ---------------------------------------------------------------------------
#  Sweep definition
# ---------------------------------------------------------------------------

# Loss weights to sweep.  0.0 is the baseline (loss term off).  Spread over a
# few orders of magnitude because the biting scale is scenario-dependent.
LOSS_WEIGHTS: List[float] = [0.0, 1.0e3, 1.0e4, 1.0e5]

# Tie-coordinator toggle.
TIE_MODES: Dict[str, bool] = {"tieOff": False, "tieOn": True}


def _weight_label(w: float) -> str:
    """Compact label for a weight, e.g. 0.0 -> 'g0', 1e4 -> 'g1e4'."""
    if w == 0.0:
        return "g0"
    exp = int(round(np.log10(w)))
    mant = w / (10.0 ** exp)
    if abs(mant - 1.0) < 1e-9:
        return f"g1e{exp}"
    return f"g{mant:g}e{exp}"


def build_scenarios() -> Dict[str, Dict[str, Any]]:
    """Cross product LOSS_WEIGHTS × TIE_MODES -> {scenario_name: overrides}."""
    scen: Dict[str, Dict[str, Any]] = {}
    for w in LOSS_WEIGHTS:
        for tie_name, tie_on in TIE_MODES.items():
            name = f"{_weight_label(w)}_{tie_name}"
            scen[name] = dict(
                tso_g_loss=w,
                enable_tie_coordination=tie_on,
                _g_loss=w,          # carried for the summary (not a config key)
                _tie_on=tie_on,
            )
    return scen


SCENARIOS: Dict[str, Dict[str, Any]] = build_scenarios()

# Keys that are summary metadata, not MultiTSOConfig fields.
_META_KEYS = {"_g_loss", "_tie_on"}


# ---------------------------------------------------------------------------
#  Base configuration (held fixed across all scenarios)
# ---------------------------------------------------------------------------


def make_base_config(smoke: bool = False) -> MultiTSOConfig:
    """Shared conditions: cascade OFO at both layers on IEEE 39-bus
    ``wind_replace``, a clean 60-min profile (no contingencies) so loss
    differences are attributable purely to the reactive re-dispatch.

    Weights/timing mirror ``002_M_TSO_M_DSO_COMPARE.make_base_config`` so the
    operating point is realistic; only ``tso_g_loss`` and
    ``enable_tie_coordination`` are varied per scenario.
    """
    cfg = MultiTSOConfig(
        n_total_s=(2 * 180.0) if smoke else 60.0 * 60.0,   # 60-min run
        tso_period_s=180.0,        # TSO every 3 minutes
        dso_period_s=20.0,
        g_v=3e5,                   # TSO voltage tracking (primary objective)
        g_q=250,                   # DSO Q-tracking
        tso_g_q_tie=1,             # tie-Q tracking (Phase B; same in both modes)
        # ── DSO objective tuning (from 002 base) ──
        dso_g_v=20000.0,
        dso_g_qi=0,
        dso_lambda_qi=0.95,
        dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,
        # ── TSO move penalties ──
        g_w_der=10,
        g_w_gen=5e7,
        g_w_pcc=10,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        # ── DSO move penalties ──
        g_w_dso_der=1000,
        g_w_dso_oltc=40,
        use_fixed_zones=True,      # literature 3-area partition
        run_stability_analysis=False,
        sensitivity_update_interval=1e6,
        verbose=0,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        live_plot_tie_coordination=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # ── Profile settings (no contingencies for a clean comparison) ──
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[],
    )
    cfg.scenario = "wind_replace"
    cfg.warmup_s = 0.0
    return cfg


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------


def run_one_scenario(name: str, overrides: Dict[str, Any], out_root: str,
                     smoke: bool = False) -> List[MultiTSOIterationRecord]:
    """Run a single scenario and pickle its log.  Returns the log."""
    cfg = make_base_config(smoke=smoke)
    for k, v in overrides.items():
        if k in _META_KEYS:
            continue
        setattr(cfg, k, v)

    scen_dir = os.path.join(out_root, name)
    cfg.result_dir = scen_dir
    os.makedirs(scen_dir, exist_ok=True)

    print()
    print("=" * 72)
    print(f"  RUNNING SCENARIO {name}")
    print("=" * 72)
    print(f"  tso_g_loss={cfg.tso_g_loss}, "
          f"enable_tie_coordination={cfg.enable_tie_coordination}")
    print(f"  result_dir={scen_dir}")

    try:
        log = run_multi_tso_dso(cfg)
    except Exception as exc:  # noqa: BLE001 — isolate one scenario's failure
        print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
        log = []

    pkl_path = os.path.join(scen_dir, "log.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [{name}] wrote {len(log)} records -> {pkl_path}")
    return log


def load_logs(out_root: str,
              names: Optional[List[str]] = None,
              ) -> Dict[str, List[MultiTSOIterationRecord]]:
    """Reload pickled scenario logs from a previous run."""
    if names is None:
        names = list(SCENARIOS.keys())
    out: Dict[str, List[MultiTSOIterationRecord]] = {}
    for name in names:
        pkl_path = os.path.join(out_root, name, "log.pkl")
        if not os.path.isfile(pkl_path):
            print(f"  [load_logs] missing {pkl_path} -- treating as diverged")
            out[name] = []
            continue
        with open(pkl_path, "rb") as f:
            out[name] = pickle.load(f)
        print(f"  [load_logs] {name}: {len(out[name])} records")
    return out


# ---------------------------------------------------------------------------
#  Aggregation
# ---------------------------------------------------------------------------


def _scalar_metrics(log: List[MultiTSOIterationRecord]) -> Dict[str, float]:
    """Reduce one scenario's record list to scalar comparison metrics.

    Only TSO-active steps are used for the loss mean (the OFO layer is the
    thing the loss term acts on); falls back to all records if no step is
    flagged active.
    """
    if not log:
        return dict(n_records=0, mean_loss_mw=np.nan, final_loss_mw=np.nan,
                    std_loss_mw=np.nan, vmin=np.nan, vmax=np.nan,
                    mean_vrmse_pu=np.nan, mean_abs_tie_q_mvar=np.nan)

    losses = np.array([float(r.total_losses_mw) for r in log], dtype=float)
    active = np.array([bool(getattr(r, "tso_active", False)) for r in log])
    loss_sel = losses[active] if active.any() else losses

    # Voltage envelope across all zones / steps.
    vmins, vmaxs = [], []
    for r in log:
        if r.zone_v_min:
            vmins.append(min(r.zone_v_min.values()))
        if r.zone_v_max:
            vmaxs.append(max(r.zone_v_max.values()))

    # Voltage-tracking RMSE: mean over steps of the cross-zone mean RMSE.
    vrmse_per_step = []
    for r in log:
        if r.zone_v_rms_err_pu:
            vrmse_per_step.append(float(np.mean(list(r.zone_v_rms_err_pu.values()))))

    # Mean total absolute tie reactive exchange (sum |Q| over tie pairs).
    tie_per_step = []
    for r in log:
        if r.zone_tie_q_mvar:
            tie_per_step.append(
                float(np.sum(np.abs(list(r.zone_tie_q_mvar.values()))))
            )

    return dict(
        n_records=len(log),
        mean_loss_mw=float(np.mean(loss_sel)),
        final_loss_mw=float(losses[-1]),
        std_loss_mw=float(np.std(loss_sel)),
        vmin=float(min(vmins)) if vmins else np.nan,
        vmax=float(max(vmaxs)) if vmaxs else np.nan,
        mean_vrmse_pu=float(np.mean(vrmse_per_step)) if vrmse_per_step else np.nan,
        mean_abs_tie_q_mvar=float(np.mean(tie_per_step)) if tie_per_step else np.nan,
    )


def _loss_timeseries(log: List[MultiTSOIterationRecord]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """(time_min, total_losses_mw) trajectory for one scenario."""
    if not log:
        return np.array([]), np.array([])
    t = np.array([r.time_s / 60.0 for r in log], dtype=float)
    p = np.array([float(r.total_losses_mw) for r in log], dtype=float)
    return t, p


def write_summary_csv(logs: Dict[str, List[MultiTSOIterationRecord]],
                      out_root: str) -> List[Dict[str, Any]]:
    """Write summary.csv (one row per scenario) and return the rows."""
    rows: List[Dict[str, Any]] = []
    for name, ov in SCENARIOS.items():
        m = _scalar_metrics(logs.get(name, []))
        rows.append(dict(
            scenario=name,
            g_loss=ov["_g_loss"],
            tie_on=int(bool(ov["_tie_on"])),
            **m,
        ))
    # Sort: tie mode, then weight.
    rows.sort(key=lambda d: (d["tie_on"], d["g_loss"]))

    csv_path = os.path.join(out_root, "summary.csv")
    fieldnames = ["scenario", "g_loss", "tie_on", "n_records",
                  "mean_loss_mw", "final_loss_mw", "std_loss_mw",
                  "vmin", "vmax", "mean_vrmse_pu", "mean_abs_tie_q_mvar"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})
    print(f"  wrote {csv_path}")

    # Console table.
    print()
    print(f"  {'scenario':<14}{'g_loss':>10}{'tie':>5}"
          f"{'mean_loss':>12}{'final_loss':>12}{'vmin':>8}{'vmax':>8}"
          f"{'vrmse':>10}{'|tieQ|':>10}")
    for row in rows:
        print(f"  {row['scenario']:<14}{row['g_loss']:>10.0f}"
              f"{row['tie_on']:>5}{row['mean_loss_mw']:>12.4f}"
              f"{row['final_loss_mw']:>12.4f}{row['vmin']:>8.4f}"
              f"{row['vmax']:>8.4f}{row['mean_vrmse_pu']:>10.5f}"
              f"{row['mean_abs_tie_q_mvar']:>10.2f}")
    return rows


# ---------------------------------------------------------------------------
#  Plotting (self-contained, no external comparison helper)
# ---------------------------------------------------------------------------


def _save(fig, out_root: str, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(out_root, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=140)
    print(f"  wrote {stem}.png / .pdf")


def make_plots(logs: Dict[str, List[MultiTSOIterationRecord]],
               rows: List[Dict[str, Any]], out_root: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = LOSS_WEIGHTS
    xlabels = [_weight_label(w) for w in weights]
    x = np.arange(len(weights))

    def series(metric: str, tie_on: int) -> np.ndarray:
        by_w = {r["g_loss"]: r for r in rows if r["tie_on"] == tie_on}
        return np.array([by_w[w][metric] if w in by_w else np.nan
                         for w in weights], dtype=float)

    # ── Figure 1: mean plant loss vs weight, tie off/on ──────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(x, series("mean_loss_mw", 0), "o-", label="tie coord OFF")
    ax1.plot(x, series("mean_loss_mw", 1), "s--", label="tie coord ON")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel(r"TSO loss weight $g_\mathrm{loss}$")
    ax1.set_ylabel("mean whole-network active loss [MW]")
    ax1.set_title("Plant losses vs loss weight (tie coordinator off/on)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    _save(fig1, out_root, "losses_vs_gloss")
    plt.close(fig1)

    # ── Figure 2: loss trajectories, all scenarios ───────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    for name in SCENARIOS:
        t, p = _loss_timeseries(logs.get(name, []))
        if t.size == 0:
            continue
        g = SCENARIOS[name]["_g_loss"]
        tie_on = SCENARIOS[name]["_tie_on"]
        frac = (weights.index(g) / max(len(weights) - 1, 1)) if g in weights else 0.0
        ax2.plot(t, p, color=cmap(frac),
                 linestyle="--" if tie_on else "-",
                 label=name, linewidth=1.4)
    ax2.set_xlabel("time [min]")
    ax2.set_ylabel("whole-network active loss [MW]")
    ax2.set_title("Loss trajectories (solid = tie OFF, dashed = tie ON; "
                  "darker = larger $g_\\mathrm{loss}$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=7)
    _save(fig2, out_root, "losses_timeseries")
    plt.close(fig2)

    # ── Figure 3: mean |tie Q| vs weight, tie off/on ─────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 4.2))
    ax3.plot(x, series("mean_abs_tie_q_mvar", 0), "o-", label="tie coord OFF")
    ax3.plot(x, series("mean_abs_tie_q_mvar", 1), "s--", label="tie coord ON")
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels)
    ax3.set_xlabel(r"TSO loss weight $g_\mathrm{loss}$")
    ax3.set_ylabel(r"mean $\sum |Q_\mathrm{tie}|$ [Mvar]")
    ax3.set_title("Inter-zone reactive exchange vs loss weight")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    _save(fig3, out_root, "tie_q_vs_gloss")
    plt.close(fig3)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main(only: Optional[List[str]] = None, smoke: bool = False) -> None:
    out_root = results_path("009_loss_tie_sweep")
    os.makedirs(out_root, exist_ok=True)

    selected = list(SCENARIOS.keys())
    if only is not None:
        unknown = [s for s in only if s not in SCENARIOS]
        if unknown:
            raise ValueError(
                f"--only contains unknown scenarios: {unknown}; "
                f"valid names are {list(SCENARIOS.keys())}"
            )
        selected = [s for s in selected if s in only]

    if not selected:
        print("[main] No scenarios selected; nothing to run.")
    else:
        print(f"[main] Running scenarios: {selected}")
        for name in selected:
            run_one_scenario(name, SCENARIOS[name], out_root, smoke=smoke)

    logs = load_logs(out_root)
    print()
    print("=" * 72)
    print("  AGGREGATING COMPARISON METRICS")
    print("=" * 72)
    rows = write_summary_csv(logs, out_root)
    try:
        make_plots(logs, rows, out_root)
    except Exception as exc:  # noqa: BLE001 — plotting must not lose the CSV
        print(f"  [plot] FAILED: {type(exc).__name__}: {exc}")
    print(f"  Done. Results in {out_root}/")


def replot() -> None:
    out_root = results_path("009_loss_tie_sweep")
    logs = load_logs(out_root)
    rows = write_summary_csv(logs, out_root)
    make_plots(logs, rows, out_root)


def _parse_csv_arg(argv: List[str], flag: str) -> Optional[List[str]]:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(f"{flag} requires a comma-separated value")
    return [s.strip() for s in argv[idx + 1].split(",") if s.strip()]


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot()
    else:
        main(
            only=_parse_csv_arg(sys.argv, "--only"),
            smoke="--smoke" in sys.argv,
        )
