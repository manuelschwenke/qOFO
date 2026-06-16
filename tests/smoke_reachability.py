"""Smoke driver for the voltage-stability reachability guard wired into the
multi-TSO/DSO runner.

NOT a pytest test — a fast driver that runs a few steps of the real time-series
loop with ``enable_reachability_guard=True`` and reports the recorded per-step
nose-curve margin.  Pass criteria:

* the runner returns without a ``ReachabilityViolation`` (normal operation is on
  the stable upper branch);
* every iteration record carries a finite ``reach_lambda_min_JR`` > 0 and
  ``reach_sigma_min_J`` > 0.

Run from the project root:

    python tests/smoke_reachability.py
"""

from __future__ import annotations

import io
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.multi_tso_config import MultiTSOConfig
from experiments.runners.multi_tso_dso import run_multi_tso_dso
from analysis.reachability import ReachabilityViolation


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=3.0 * 60.0,          # 3 steps
        tso_period_s=3.0 * 60.0,
        dso_period_s=1.0 * 60.0,
        verbose=1,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        run_stability_analysis=False,
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 10, 0),
        contingencies=[],
        enable_reachability_guard=True,
    )
    print(f"[smoke_reachability] guard={cfg.enable_reachability_guard} "
          f"tau_sigma={cfg.reach_tau_sigma:.1e} tau_eig={cfg.reach_tau_eig:.1e}")
    t0 = time.perf_counter()
    try:
        log = run_multi_tso_dso(cfg)
    except ReachabilityViolation as exc:
        print(f"\n[smoke_reachability] GUARD ABORTED THE RUN:\n  {exc}")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        print("\n[smoke_reachability] FAILED — see traceback above.")
        sys.exit(1)
    dt = time.perf_counter() - t0
    print(f"\n[smoke_reachability] OK — {len(log)} records in {dt:.1f}s "
          f"({dt / max(len(log), 1):.2f}s/step incl. guard).")
    print("  step | sigma_min(J) | lambda_min(J_R) | critical_bus")
    for rec in log:
        print(f"  {rec.step:4d} | {rec.reach_sigma_min_J!s:>12} | "
              f"{rec.reach_lambda_min_JR!s:>15} | {rec.reach_critical_bus}")
    # Sanity: all recorded margins finite and on the stable branch.
    bad = [r.step for r in log
           if r.reach_lambda_min_JR is None or r.reach_lambda_min_JR <= 0.0]
    if bad:
        print(f"[smoke_reachability] WARNING: non-positive lambda at steps {bad}")
        sys.exit(3)
    print("[smoke_reachability] all steps on the stable upper branch.")


if __name__ == "__main__":
    main()
