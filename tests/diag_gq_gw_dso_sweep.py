"""
3 x 3 sweep of (g_q, g_w_dso_der) for Stage 2 q_shim mode (post matrix-form
shim patch, 2026-05-04).

Measures Q-iface tracking error and DER-Q chatter so we can pick a tuning
that closes more of the Stage 1 / Stage 2 gap without inducing oscillation.

Configuration mirrors the user's main() in
``experiments/000_M_TSO_M_DSO.py`` but on a 40-min horizon with the same
gen-trip + restore disturbance used by ``diag_qshim_v_tracking.py``.

Sweep grid (default):
    g_q          ∈ {200, 400, 800}      [1/Mvar²]
    g_w_dso_der  ∈ {500, 1000, 2000}    [1/Mvar²]

Outputs (per cell):
    Q-iface RMS over all DSOs / time   [Mvar]
    Q-iface peak per DSO                [Mvar]
    V-tracking RMS over zones / time   [pu]
    DER-Q chatter: per-DSO mean / max |ΔQ_DER| between consecutive log
                   records (only counted on records where the DSO was active).

Usage:
    python tests/diag_gq_gw_dso_sweep.py
"""

from __future__ import annotations

import io
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import ContingencyEvent

spec = importlib.util.spec_from_file_location(
    "exp_000_m_tso_m_dso",
    ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_multi_tso_dso = mod.run_multi_tso_dso


# ---------------------------------------------------------------------------
#  Sweep grid
# ---------------------------------------------------------------------------

GQ_GRID: List[float] = [200.0, 400.0, 800.0]
GW_GRID: List[float] = [500.0, 1000.0, 2000.0]


# ---------------------------------------------------------------------------
#  Config builder
# ---------------------------------------------------------------------------


def make_cfg(*, g_q: float, g_w_dso_der: float) -> MultiTSOConfig:
    return MultiTSOConfig(
        dt_s=60.0,
        n_total_s=40.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        g_v=500000.0,
        g_q=g_q,
        use_qv_local_loop=True,
        qv_apply_mode="q_shim",
        force_all_der_grid_following=False,
        dso_g_v=20000.0,
        dso_g_qi=0,
        dso_lambda_qi=0.9,
        dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10,
        g_w_gridforming=5e7,
        g_w_gen=1e8,
        g_w_pcc=50,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        g_w_dso_der=g_w_dso_der,
        g_w_dso_der_vref=1e-4,    # inert in q_shim — kept identical across cells
        g_w_dso_oltc=20,
        adapt_g_w_der=False,
        adapt_g_w_pcc=False,
        adapt_g_w_gen=False,
        adapt_g_w_dso_der=False,
        g_w_adapt_t_max=5e7,
        g_w_adapt_deadband_rel=1e-4,
        g_w_adapt_t_min_per_class={
            "der": 10.0, "pcc": 10.0, "gen": 1.0e4, "dso_der": 100.0,
        },
        g_w_adapt_t_max_per_class={
            "der": 1e4, "pcc": 1e4, "dso_der": 1e4,
        },
        g_w_adapt_beta1_per_class={
            "der": 0.05, "pcc": 0.02, "gen": 0.1, "dso_der": 0.002,
        },
        g_w_adapt_beta2_per_class={
            "der": 0.3, "pcc": 0.15, "gen": 0.2, "dso_der": 0.03,
        },
        qv_slope_pu=0.07,
        qv_v_ref_min_pu=0.95,
        qv_v_ref_max_pu=1.10,
        g_z_warmup_s=0.0,
        verbose=0,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        run_stability_analysis=False,
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        use_fixed_zones=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 8, 0),
        contingencies=[
            ContingencyEvent(minute=15, element_type="gen", element_index=5, action="trip"),
            ContingencyEvent(minute=30, element_type="gen", element_index=5, action="restore"),
        ],
    )


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def _q_iface_metrics(log: list) -> Tuple[float, Dict[str, float]]:
    """Return (RMS over all DSOs / time, per-DSO peak |actual - set|)."""
    err: Dict[str, List[float]] = {}
    for rec in log:
        a_dict = getattr(rec, "dso_q_actual_mvar", {}) or {}
        t_dict = getattr(rec, "dso_q_set_mvar", {}) or {}
        for did, a in a_dict.items():
            t = t_dict.get(did)
            if a is None or t is None:
                continue
            err.setdefault(did, []).append(abs(float(a) - float(t)))
    if not err:
        return float("nan"), {}
    all_err = []
    peaks: Dict[str, float] = {}
    for did, vals in err.items():
        arr = np.asarray(vals, dtype=float)
        peaks[did] = float(arr.max()) if arr.size else float("nan")
        all_err.extend(vals)
    a = np.asarray(all_err, dtype=float)
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")
    return rms, peaks


def _v_tracking_rms(log: list, v_set: float) -> float:
    """Return RMS V-tracking error over zones / time."""
    errs: List[float] = []
    for rec in log:
        zmean = getattr(rec, "zone_v_mean", {}) or {}
        for vmean in zmean.values():
            if vmean is None or not np.isfinite(float(vmean)):
                continue
            errs.append(abs(float(vmean) - v_set))
    if not errs:
        return float("nan")
    a = np.asarray(errs, dtype=float)
    return float(np.sqrt(np.mean(a ** 2)))


def _q_chatter(log: list) -> Dict[str, Tuple[float, float]]:
    """Per-DSO (mean, max) |ΔQ_DER| step-to-step across all DERs.

    Skips records where the DSO was inactive (no Q_DER recorded).
    """
    series: Dict[str, List[np.ndarray]] = {}
    for rec in log:
        for did, arr in (getattr(rec, "dso_q_der", {}) or {}).items():
            if arr is None:
                continue
            arr_np = np.asarray(arr, dtype=float).ravel()
            if arr_np.size == 0:
                continue
            series.setdefault(did, []).append(arr_np)
    out: Dict[str, Tuple[float, float]] = {}
    for did, frames in series.items():
        if len(frames) < 2:
            out[did] = (0.0, 0.0)
            continue
        # Align lengths in case of size changes (shouldn't happen).
        n = min(f.shape[0] for f in frames)
        if n == 0:
            out[did] = (0.0, 0.0)
            continue
        mat = np.stack([f[:n] for f in frames], axis=0)  # (T, n_der)
        d = np.abs(np.diff(mat, axis=0))                 # (T-1, n_der)
        # Max over DERs at each step, then mean / max over steps.
        per_step_max = d.max(axis=1)
        out[did] = (float(per_step_max.mean()), float(per_step_max.max()))
    return out


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def _fmt_dict(d: Dict[str, float], width: int = 6, prec: int = 2) -> str:
    items = ", ".join(f"{k}={v:>{width}.{prec}f}" for k, v in sorted(d.items()))
    return items if items else "(none)"


def main() -> None:
    out_path = ROOT / "tests" / "diag_gq_gw_dso_sweep.out"
    rows: List[Dict] = []

    with out_path.open("w", encoding="utf-8") as out:
        print("Stage 2 q_shim (post matrix-form shim patch) — (g_q, g_w_dso_der) sweep", file=out)
        print(f"  horizon: 40 min  |  contingency: gen5 trip @15m, restore @30m", file=out)
        print(f"  grid: g_q ∈ {GQ_GRID},  g_w_dso_der ∈ {GW_GRID}", file=out)
        print("", file=out)

        for g_q in GQ_GRID:
            for g_w in GW_GRID:
                label = f"g_q={g_q:>5.0f}  g_w_dso_der={g_w:>5.0f}  (ratio={g_q/g_w:.3f})"
                print(f"  RUN: {label}")
                cfg = make_cfg(g_q=g_q, g_w_dso_der=g_w)
                v_set = cfg.v_setpoint_pu
                try:
                    log = run_multi_tso_dso(cfg)
                except Exception as exc:
                    print(f"\n=== {label} ===\n  FAILED: {exc!r}", file=out)
                    print(f"  FAILED: {exc!r}")
                    continue

                rms_q, peak_q = _q_iface_metrics(log)
                rms_v = _v_tracking_rms(log, v_set)
                chatter = _q_chatter(log)

                print(f"\n=== {label} ===", file=out)
                print(f"  Q-iface RMS over DSOs / time : {rms_q:.3f} Mvar", file=out)
                print(f"  V-tracking RMS over zones / time: {rms_v:.5f} pu", file=out)
                print(f"  Q-iface peak per DSO [Mvar] : {_fmt_dict(peak_q, width=5, prec=2)}", file=out)
                print(f"  DER Q chatter mean |ΔQ| [Mvar]: "
                      f"{_fmt_dict({d: c[0] for d, c in chatter.items()}, width=5, prec=2)}", file=out)
                print(f"  DER Q chatter max  |ΔQ| [Mvar]: "
                      f"{_fmt_dict({d: c[1] for d, c in chatter.items()}, width=5, prec=2)}", file=out)
                rows.append(dict(
                    g_q=g_q,
                    g_w_dso_der=g_w,
                    ratio=g_q / g_w,
                    rms_q=rms_q,
                    rms_v=rms_v,
                    peak_q_max=max(peak_q.values()) if peak_q else float("nan"),
                    chatter_max=max(c[1] for c in chatter.values()) if chatter else float("nan"),
                ))

        # ---- Compact summary table at the bottom ----
        print("\n=== SUMMARY (sorted by Q-iface RMS) ===", file=out)
        print(f"  {'g_q':>5}  {'g_w_dso':>7}  {'ratio':>6}  "
              f"{'Q_RMS':>7}  {'Q_peak':>7}  {'V_RMS':>9}  {'chat_max':>9}", file=out)
        print(f"  {'':>5}  {'':>7}  {'':>6}  {'[Mvar]':>7}  {'[Mvar]':>7}  "
              f"{'[pu]':>9}  {'[Mvar]':>9}", file=out)
        for r in sorted(rows, key=lambda x: (x["rms_q"]
                                             if np.isfinite(x["rms_q"]) else 1e9)):
            print(
                f"  {r['g_q']:>5.0f}  {r['g_w_dso_der']:>7.0f}  "
                f"{r['ratio']:>6.3f}  {r['rms_q']:>7.3f}  "
                f"{r['peak_q_max']:>7.2f}  {r['rms_v']:>9.5f}  "
                f"{r['chatter_max']:>9.3f}",
                file=out,
            )

    print(f"\nFull summary: {out_path}")
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
