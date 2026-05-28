"""
Compare DSO Q-tracking across the three modes:

  1. Stage 1               — use_qv_local_loop=False           (direct Q, no Q(V) plant)
  2. Stage 2 V_ref-direct  — use_qv_local_loop=True, qv_apply_mode='v_ref'
  3. Stage 2 Q+shim        — use_qv_local_loop=True, qv_apply_mode='q_shim'

Same OFO weights, same horizon, same scenario.  Q+shim should recover
Stage 1 tracking quality while keeping the Q(V) plant model.

Usage:
    python tests/diag_three_modes_sweep.py
"""

from __future__ import annotations

import io
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from configs.multi_tso_config import MultiTSOConfig

spec = importlib.util.spec_from_file_location(
    "exp_000_m_tso_m_dso",
    ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_multi_tso_dso = mod.run_multi_tso_dso


def make_cfg(*, use_qv_local_loop: bool, qv_apply_mode: str) -> MultiTSOConfig:
    return MultiTSOConfig(
        dt_s=60.0,
        n_total_s=20.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        g_v=500000.0, g_q=200,
        use_qv_local_loop=use_qv_local_loop,
        qv_apply_mode=qv_apply_mode,
        dso_g_v=25000.0,
        dso_g_qi=0, dso_lambda_qi=0.9, dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10, g_w_gridforming=1e7, g_w_gen=5e7, g_w_pcc=50,
        g_w_tso_oltc=100, install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        g_w_dso_der=1000,            # used by Stage 1 + Q+shim
        g_w_dso_der_vref=1.0,        # used by V_ref-direct (inert)
        g_w_dso_oltc=40,
        qv_slope_pu=0.07, qv_v_ref_min_pu=0.95, qv_v_ref_max_pu=1.10,
        g_z_warmup_s=0.0,
        verbose=0,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        run_stability_analysis=False,
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 10, 0),
        contingencies=[],
    )


def summarize(label: str, log: list, out) -> None:
    print(f"\n=== {label} ===", file=out)
    if not log:
        print("  (no log records)", file=out)
        return

    iface_err: dict[str, list[float]] = {}
    actuals_per_dso: dict[str, list[float]] = {}
    for rec in log:
        actuals = getattr(rec, "dso_q_actual_mvar", {}) or {}
        targets = getattr(rec, "dso_q_set_mvar", {}) or {}
        for did in actuals:
            a, t = actuals.get(did), targets.get(did)
            if a is None or t is None:
                continue
            iface_err.setdefault(did, []).append(abs(float(a) - float(t)))
            actuals_per_dso.setdefault(did, []).append(float(a))

    if iface_err:
        all_err = []
        all_q = []
        print("  DSO Q_iface tracking |actual - setpoint|:", file=out)
        for did in sorted(iface_err):
            arr = np.array(iface_err[did])
            qarr = np.array(actuals_per_dso[did])
            print(
                f"    {did}: err mean={arr.mean():6.2f}  max={arr.max():6.2f}  "
                f"final={arr[-1]:6.2f}  |  Q range=[{qarr.min():+6.1f}, "
                f"{qarr.max():+6.1f}]",
                file=out,
            )
            all_err.extend(iface_err[did])
            all_q.extend(actuals_per_dso[did])
        rms = np.sqrt(np.mean(np.array(all_err) ** 2))
        q_arr = np.array(all_q)
        print(
            f"    -> RMS error: {rms:6.2f} Mvar  |  "
            f"Q swing across all DSOs: [{q_arr.min():+6.1f}, {q_arr.max():+6.1f}]",
            file=out,
        )


def main() -> None:
    sweeps: List[Tuple[str, dict]] = [
        ("STAGE 1 direct-Q     (use_qv_local_loop=False)",
         dict(use_qv_local_loop=False, qv_apply_mode="q_shim")),
        ("STAGE 2 V_ref direct (use_qv_local_loop=True, qv_apply_mode='v_ref')",
         dict(use_qv_local_loop=True, qv_apply_mode="v_ref")),
        ("STAGE 2 Q + shim     (use_qv_local_loop=True, qv_apply_mode='q_shim')",
         dict(use_qv_local_loop=True, qv_apply_mode="q_shim")),
    ]

    summary_path = ROOT / "tests" / "diag_three_modes_sweep.out"
    with summary_path.open("w", encoding="utf-8") as out:
        for label, kwargs in sweeps:
            cfg = make_cfg(**kwargs)
            try:
                log = run_multi_tso_dso(cfg)
            except Exception as exc:
                print(f"\n=== {label} ===\n  FAILED: {exc!r}", file=out)
                print(f"FAILED: {label} ({exc!r})")
                continue
            summarize(label, log, out=out)
            print(f"DONE:   {label}")
    print(f"\nFull summary: {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
