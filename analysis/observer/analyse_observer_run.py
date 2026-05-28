"""Phase-3 post-processing for a single observer-equipped simulation run.

Runs the canonical scenario-B config for `HORIZON_MIN` minutes (default 60),
captures the observer instance via a monkey-patch on `attach_observer`,
and emits:

  1. Slack-ratio table:  g_w_current / g_w_spectral_gap_p95 per zone x block.
     Classified as Spectral-gap-certified / box-regularised / strongly box-regularised.
  2. ||M||_op distribution per zone:  mean, p95, max from observer.trajectories.
  3. Empirical contraction rate rho_k per zone, from MultiTSOIterationRecord
     u_k = concat(zone_q_der, zone_q_pcc_set, zone_v_gen, zone_oltc_taps).
     Per-zone histogram + p95.
  4. JSON summary  + PNG plot.

This is read-only with respect to controller logic; the only state change
is observer's own trajectory recording (passive) and the monkey-patch on
the local `attach_observer` reference (does not affect other runs).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.multi_tso_config import MultiTSOConfig

# Import the experiment module by path (filename starts with a digit).
spec = importlib.util.spec_from_file_location(
    "experiment_000",
    ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


HORIZON_MIN = 60
RESULT_DIR = ROOT / "results" / "observer_full_scenarioB"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Canonical scenario-B config: validated-stable g_w tuning per Manuel
# (2026-04-20).  This is the box-regularised operating point that the
# observer's spectral-gap recommendation is intentionally orders of
# magnitude above; the slack-ratio table below is the headline finding.
CFG = MultiTSOConfig(
    n_total_s=60.0 * HORIZON_MIN,
    tso_period_s=60.0 * 3,
    dso_period_s=10.0,
    g_v=100000.0,
    g_q=200.0,
    dso_g_v=20000.0,
    dso_g_qi=0.0,
    dso_lambda_qi=0.9,
    dso_q_integral_max_mvar=50.0,
    dso_gamma_oltc_q=0.1,
    g_w_der=40.0,
    g_w_gen=1e7,
    g_w_pcc=100.0,
    g_w_tso_oltc=200.0,
    g_w_dso_der=1200.0,
    g_w_dso_oltc=40.0,
    use_fixed_zones=True,
    run_stability_analysis=False,
    sensitivity_update_interval=int(1e6),
    verbose=0,
    result_dir=str(RESULT_DIR),
    start_time=datetime(2016, 1, 5, 3, 0),
    use_profiles=True,
    use_zonal_gen_dispatch=True,
    contingencies=[],
)


# ---------- Capture observer via monkey-patch ----------
_captured: Dict[str, Any] = {}
_original_attach = mod.attach_observer

def _capturing_attach(*args, **kwargs):
    obs = _original_attach(*args, **kwargs)
    _captured["observer"] = obs
    return obs

mod.attach_observer = _capturing_attach

print(f"Running scenario-B for {HORIZON_MIN} min ({RESULT_DIR}) ...")
t0 = time.time()
try:
    log = mod.run_multi_tso_dso(CFG)
    print(f"  done in {time.time()-t0:.1f}s; {len(log)} records")
except Exception as exc:
    log = None
    print(f"  FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}")
finally:
    mod.attach_observer = _original_attach

if "observer" not in _captured:
    raise RuntimeError("observer was never instantiated; check integration")
observer = _captured["observer"]


# ---------- Helpers ----------
def fmt(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "  NaN"
    if abs(x) >= 1e6:
        return f"{x:>10.2e}"
    return f"{x:>10.{digits}f}"


def fmt_ratio(x: float) -> str:
    if not np.isfinite(x):
        return "    NaN"
    if x >= 100:
        return f"{x:>7.1e}"
    return f"{x:>7.4f}"


def classify(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    if ratio >= 1.0:
        return "Spectral-gap-certified"
    if ratio >= 1e-3:
        return "box-regularised"
    return "strongly box-regularised"


# ---------- 1. Slack-ratio table ----------
g_w_current = {
    "DER":   CFG.g_w_der,
    "PCC":   CFG.g_w_pcc,
    "V_gen": CFG.g_w_gen,        # alias used in the observer
    "OLTC":  CFG.g_w_tso_oltc,   # TSO machine-OLTC weight
}

slack_rows: List[Dict[str, Any]] = []
print()
print("=" * 110)
print(f"  Slack-ratio table: g_w_current vs Spectral-gap (unconstrained) vs Haeberle (box-constrained)")
print("=" * 110)
print(f"  {'Z':>2s}  {'Block':>5s}  "
      f"{'g_w_current':>13s}  "
      f"{'g_w_specgap_p95':>16s}  {'ratio_S':>9s}  "
      f"{'g_w_Haberle_p95':>16s}  {'ratio_H':>9s}  Regime (vs SG)")
for z, traj in sorted(observer.trajectories.items()):
    if not traj.records:
        continue
    p95 = traj.aggregate(statistic="percentile", percentile=95.0)
    gw_haberle_p95 = traj.aggregate_haberle(statistic="percentile", percentile=95.0)
    for k, name in enumerate(traj.layout.names):
        sl = traj.layout.block_slice(k)
        if sl.stop == sl.start:
            continue
        gw_s = float(p95[sl][0])    # spectral-gap floor (formerly "Bianchi")
        gw_h = float(gw_haberle_p95)
        gw_c = float(g_w_current.get(name, np.nan))
        ratio_s = gw_c / gw_s if gw_s > 0 else np.nan
        ratio_h = gw_c / gw_h if gw_h > 0 else np.nan
        cls = classify(ratio_s)
        slack_rows.append({
            "zone": z, "block": name,
            "g_w_current": gw_c,
            "g_w_spectral_gap_p95": gw_s, "ratio_spectral_gap": ratio_s,
            "g_w_haberle_p95": gw_h, "ratio_haberle": ratio_h,
            "regime": cls,
        })
        print(f"  {z:>2d}  {name:>5s}  {fmt(gw_c)}  "
              f"{fmt(gw_s)}  {fmt_ratio(ratio_s)}  "
              f"{fmt(gw_h)}  {fmt_ratio(ratio_h)}  {cls}")
print("=" * 110)
print("  Spectral-gap = unconstrained sufficient floor:  g_w >= ||M||_op - lam_min(M)  [naming, not citation]")
print("  Haeberle     = box-constrained projected-gradient floor:  g_w >= ||M||_op / 2  [Haeberle 2021 L-CSS, verify]")
print("  ratio_X      = g_w_current / g_w_X_p95.  ratio < 1 => current g_w is below that floor.")


# ---------- 2. ||M||_op stats ----------
print()
print("=" * 60)
print("  ||M||_op distribution per zone (from observer.trajectories)")
print("=" * 60)
print(f"  {'Z':>2s}  {'n':>3s}  {'mean':>12s}  {'p95':>12s}  {'max':>12s}")
m_op_rows: List[Dict[str, Any]] = []
for z, traj in sorted(observer.trajectories.items()):
    if not traj.records:
        continue
    op = np.array([r.result.op_norm for r in traj.records])
    row = {
        "zone": z, "n": int(op.size),
        "mean": float(op.mean()), "p95": float(np.percentile(op, 95)),
        "max": float(op.max()),
    }
    m_op_rows.append(row)
    print(f"  {z:>2d}  {row['n']:>3d}  {fmt(row['mean'], 4)}  "
          f"{fmt(row['p95'], 4)}  {fmt(row['max'], 4)}")
print("=" * 60)


# ---------- 3. Empirical contraction rho_k ----------
def _u_vector(rec, z: int) -> np.ndarray:
    """Concatenate the four actuator-output blocks for zone z, or return [] if missing."""
    pieces = []
    for attr in ("zone_q_der", "zone_q_pcc_set", "zone_v_gen", "zone_oltc_taps"):
        d = getattr(rec, attr, {})
        v = d.get(z)
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size:
            pieces.append(a)
    if not pieces:
        return np.zeros(0)
    return np.concatenate(pieces)


zones = sorted(observer.tracked_zone_ids)
rho_per_zone: Dict[int, np.ndarray] = {}
print()
print("=" * 72)
print("  Empirical contraction rate rho_k = ||u_k - u_{k-1}|| / ||u_{k-1} - u_{k-2}||")
print("=" * 72)

if log is None:
    print("  (no log captured; sim crashed before completion)")
else:
    for z in zones:
        u_seq: List[np.ndarray] = []
        for rec in log:
            if not rec.tso_active:
                continue
            u = _u_vector(rec, z)
            if u.size:
                u_seq.append(u)
        if len(u_seq) < 3:
            print(f"  Zone {z}: only {len(u_seq)} u-vectors; cannot compute rho_k")
            rho_per_zone[z] = np.zeros(0)
            continue
        # Make sure all u-vectors have the same length (dimensions can change
        # if a contingency disables an actuator; pad/skip on mismatch).
        m = min(u.size for u in u_seq)
        U = np.stack([u[:m] for u in u_seq], axis=0)
        diffs = np.linalg.norm(np.diff(U, axis=0), axis=1)
        # rho_k for k >= 2 (need three consecutive u-vectors).
        # Avoid division by zero with a tiny epsilon (treat 0/0 as undefined).
        denom = diffs[:-1]
        numer = diffs[1:]
        valid = denom > 1e-12
        rho = np.where(valid, numer / np.where(valid, denom, 1.0), np.nan)
        rho = rho[np.isfinite(rho)]
        rho_per_zone[z] = rho
        if rho.size:
            p50 = float(np.median(rho))
            p95 = float(np.percentile(rho, 95))
            mx  = float(rho.max())
            frac_lt1 = float((rho < 1.0).mean())
            print(f"  Zone {z}: n={rho.size:>3d}  median={p50:6.3f}  "
                  f"p95={p95:6.3f}  max={mx:6.3f}  P(rho<1)={frac_lt1:5.2%}")
        else:
            print(f"  Zone {z}: rho_k vector empty after filtering")
print("=" * 72)


# ---------- 4. JSON dump ----------
summary = {
    "horizon_min": HORIZON_MIN,
    "config": {
        "g_v": CFG.g_v, "g_q": CFG.g_q,
        "g_w_current": g_w_current,
    },
    "slack_ratios": slack_rows,
    "haberle_floor_per_zone": {
        str(z): float(traj.aggregate_haberle(statistic="percentile", percentile=95.0))
        for z, traj in observer.trajectories.items() if traj.records
    },
    "M_op": m_op_rows,
    "rho_k": {
        str(z): {
            "n": int(r.size),
            "median": float(np.median(r)) if r.size else float("nan"),
            "p95":    float(np.percentile(r, 95)) if r.size else float("nan"),
            "max":    float(r.max()) if r.size else float("nan"),
            "p_lt_1": float((r < 1.0).mean()) if r.size else float("nan"),
        }
        for z, r in rho_per_zone.items()
    },
}
SUMMARY_JSON = RESULT_DIR / "observer_analysis_summary.json"
with open(SUMMARY_JSON, "w") as fh:
    json.dump(summary, fh, indent=2, allow_nan=True, default=lambda o: float(o))
print(f"\nWrote {SUMMARY_JSON}")


# ---------- 5. rho_k histogram plot ----------
if any(r.size for r in rho_per_zone.values()):
    fig, axes = plt.subplots(1, len(zones), figsize=(5 * len(zones), 4), dpi=110, sharey=True)
    if len(zones) == 1:
        axes = [axes]
    for ax, z in zip(axes, zones):
        rho = rho_per_zone.get(z, np.zeros(0))
        if rho.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(f"Zone {z}")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.hist(rho, bins=max(8, min(30, rho.size // 2)),
                color="#4c72b0", alpha=0.78, edgecolor="white", linewidth=0.5)
        p50 = float(np.median(rho))
        p95 = float(np.percentile(rho, 95))
        ax.axvline(1.0, color="k", lw=1.0, ls="-",  label="rho=1 (contraction boundary)")
        ax.axvline(p50, color="#55a868", lw=1.0, ls="--", label=f"median = {p50:.3f}")
        ax.axvline(p95, color="#c44e52", lw=1.0, ls="--", label=f"p95 = {p95:.3f}")
        ax.set_xlabel(r"$\rho_k = \|\Delta u_k\| / \|\Delta u_{k-1}\|$")
        if z == zones[0]:
            ax.set_ylabel("count")
        ax.set_title(f"Zone {z}  (n={rho.size},  P(rho<1)={(rho<1.0).mean():.0%})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Empirical contraction rate distribution  ({HORIZON_MIN} min, scenario B)",
        y=1.02,
    )
    plt.tight_layout()
    PLOT_PATH = RESULT_DIR / "contraction_rho_histograms.png"
    plt.savefig(str(PLOT_PATH), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_PATH}")
else:
    print("(no rho_k samples; skipping histogram plot)")
