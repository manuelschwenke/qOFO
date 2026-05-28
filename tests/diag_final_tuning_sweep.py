"""
Final tuning sweep: validate the recommended Stage-2 settings against
the user's current main() baseline, isolating the effects of the only
two levers that actually move Q-tracking quality (slope + V_ref bounds).

Usage:
    python tests/diag_final_tuning_sweep.py
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


def make_cfg(*,
             slope: float = 0.07,
             vref_lo: float = 0.95,
             vref_hi: float = 1.10) -> MultiTSOConfig:
    """Mirror production main() with overrides for slope + V_ref bounds."""
    return MultiTSOConfig(
        dt_s=60.0,
        n_total_s=20.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        # Production main() values
        g_v=500000.0,
        g_q=200,
        use_qv_local_loop=True,
        dso_g_v=25000.0,
        dso_g_qi=0,
        dso_lambda_qi=0.9,
        dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10,
        g_w_gridforming=1e7,
        g_w_gen=5e7,
        g_w_pcc=50,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        g_w_dso_der=1000,
        g_w_dso_der_vref=1.0,
        g_w_dso_oltc=40,
        # Diagnostic-swept knobs
        qv_slope_pu=slope,
        qv_v_ref_min_pu=vref_lo,
        qv_v_ref_max_pu=vref_hi,
        g_z_q_dso_der=1e2,
        g_z_warmup_s=0.0,                     # skip warmup so g_z bites immediately
        # Headless
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

    # DSO Q_iface tracking error (the headline metric)
    iface_err: dict[str, list[float]] = {}
    for rec in log:
        actuals = getattr(rec, "dso_q_actual_mvar", {}) or {}
        targets = getattr(rec, "dso_q_set_mvar", {}) or {}
        for did in actuals:
            a, t = actuals.get(did), targets.get(did)
            if a is None or t is None:
                continue
            iface_err.setdefault(did, []).append(abs(float(a) - float(t)))

    if iface_err:
        # Print per-DSO mean & final, plus a single aggregate (RMS across all)
        all_err: list[float] = []
        for did in sorted(iface_err):
            arr = np.array(iface_err[did])
            print(
                f"    {did}: mean={arr.mean():6.2f}  max={arr.max():6.2f}  "
                f"final={arr[-1]:6.2f} Mvar",
                file=out,
            )
            all_err.extend(iface_err[did])
        rms = np.sqrt(np.mean(np.array(all_err) ** 2))
        print(f"    -> RMS across all DSOs over time: {rms:6.2f} Mvar", file=out)

    # Total realised DSO DER Q infeed (sum of |Q| per DSO across last record)
    last = log[-1]
    print(f"    Final realised DSO DER |Q| sums:", file=out)
    for did, q_arr in sorted((getattr(last, "dso_q_der", {}) or {}).items()):
        if q_arr is None or len(q_arr) == 0:
            continue
        print(
            f"    {did}: |Q|_sum = {float(np.abs(q_arr).sum()):6.2f} Mvar  "
            f"({len(q_arr)} sgens)",
            file=out,
        )


def main() -> None:
    sweeps: List[Tuple[str, dict]] = [
        ("BASELINE        : slope=0.07  bounds=[0.95, 1.10]  (your main)",
         dict(slope=0.07, vref_lo=0.95, vref_hi=1.10)),
        ("STEEPER         : slope=0.04  bounds=[0.95, 1.10]",
         dict(slope=0.04, vref_lo=0.95, vref_hi=1.10)),
        ("WIDER           : slope=0.07  bounds=[0.85, 1.20]",
         dict(slope=0.07, vref_lo=0.85, vref_hi=1.20)),
        ("RECOMMENDED     : slope=0.04  bounds=[0.85, 1.20]",
         dict(slope=0.04, vref_lo=0.85, vref_hi=1.20)),
        ("AGGRESSIVE      : slope=0.02  bounds=[0.80, 1.25]",
         dict(slope=0.02, vref_lo=0.80, vref_hi=1.25)),
        ("STEEP+VERY_WIDE : slope=0.04  bounds=[0.70, 1.30]",
         dict(slope=0.04, vref_lo=0.70, vref_hi=1.30)),
    ]

    summary_path = ROOT / "tests" / "diag_final_tuning_sweep.out"
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
    # Also dump to console for convenience.
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
