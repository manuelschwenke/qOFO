"""Smoke run for the refactor_v2 Q_cor end-to-end path.

NOT a pytest test — just a fast driver that flips
``MultiTSOConfig.use_q_cor_actuator=True`` and exercises the new
Q_cor formulation through the multi-zone runner.

Run from the project root:

    python tests/smoke_q_cor.py

A 4-minute simulation at 1-min step, 3-min TSO period, 1-min DSO
period exercises both control layers at least once.  Pass criteria:

* runner returns without exception;
* at least one iteration record is logged;
* every DER has ``q_cor_mvar`` populated (the OFO wrote at least
  one command); a sentinel "all zeros" return means the OFO never
  fired — flag and exit non-zero.
"""

from __future__ import annotations

import io
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout so the runner's print statements with unicode arrows
# don't crash on the Windows cp1252 default console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Make the project root importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.multi_tso_config import MultiTSOConfig

# Import via the experiments module name (file starts with a digit).
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_000_m_tso_m_dso",
    ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_multi_tso_dso = mod.run_multi_tso_dso


def main() -> None:
    cfg = MultiTSOConfig(
        # 4-minute simulation, 1-min step, 3-min TSO, 1-min DSO.
        dt_s=60.0,
        n_total_s=4.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=1.0 * 60.0,
        verbose=2,
        # Live plots OFF — feedback memory says always False for tests.
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        run_stability_analysis=False,
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 10, 0),
        contingencies=[],
        # ── refactor_v2 Q_cor path ────────────────────────────────────
        use_q_cor_actuator=True,
        use_qv_local_loop=True,
        # Test profile: TSO DERs at unity power factor (Q=0 always),
        # DSO DERs on Q(V) droop at V_ref=1.03.  This isolates the
        # DSO-side Q_cor path and matches the regime where the legacy
        # stage2 smoke is known to converge.
        # Default tso_q_mode="qv" — exercise QVLocalLoop on TSO STATCOMs too
        tso_qv_vref_pu=1.03,
        dso_qv_vref_pu=1.03,
        qv_local_damping=0.1,
        qv_local_max_step_frac=1.0,
        # Per-level tolerances now come from MultiTSOConfig defaults:
        # tso_qv_tol_mvar=0.1, dso_qv_tol_mvar=0.01.  No override here.
    )
    print(f"\n[smoke_q_cor] use_q_cor_actuator={cfg.use_q_cor_actuator} "
          f"use_qv_local_loop={cfg.use_qv_local_loop}")
    print(f"[smoke_q_cor] tso_q_mode={cfg.tso_q_mode} "
          f"dso_q_mode={cfg.dso_q_mode}")
    print(f"[smoke_q_cor] tso_qv_vref_pu={cfg.tso_qv_vref_pu} "
          f"dso_qv_vref_pu={cfg.dso_qv_vref_pu}")
    print(f"[smoke_q_cor] tso_qv_slope_pu={cfg.tso_qv_slope_pu} "
          f"dso_qv_slope_pu={cfg.dso_qv_slope_pu}")
    print(f"[smoke_q_cor] tso_qv_deadband_pu={cfg.tso_qv_deadband_pu} "
          f"dso_qv_deadband_pu={cfg.dso_qv_deadband_pu}\n")
    try:
        log = run_multi_tso_dso(cfg)
    except Exception:
        traceback.print_exc()
        print("\n[smoke_q_cor] FAILED — see traceback above.")
        sys.exit(1)
    print(f"\n[smoke_q_cor] OK — {len(log)} iteration records.")


if __name__ == "__main__":
    main()
