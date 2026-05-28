"""Smoke runs for Stage 1 and Stage 2 — verify the experiment can
build, step, and tear down end-to-end without errors.

NOT a pytest test — just a fast end-to-end driver.  Two modes:

    python tests/smoke_run_stage1.py                # Stage 1 (V_gf only)
    python tests/smoke_run_stage1.py --stage2       # Stage 1 + Stage 2 (Q(V) loops)
"""

from __future__ import annotations

import io
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout so the runner's print statements with unicode arrows
# (→) don't crash on the Windows cp1252 default console.
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
    stage2 = "--stage2" in sys.argv
    cfg = MultiTSOConfig(
        # Run only 4 minutes of sim time at 1-min step, 3-min TSO period,
        # 1-min DSO period — enough to exercise both layers once each.
        dt_s=60.0,
        n_total_s=4.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=1.0 * 60.0,
        verbose=2,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        run_stability_analysis=False,
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 10, 0),
        contingencies=[],
        use_qv_local_loop=stage2,
    )
    label = "stage2" if stage2 else "stage1"
    print(f"\n[smoke_run_{label}] use_qv_local_loop={cfg.use_qv_local_loop}\n")
    try:
        log = run_multi_tso_dso(cfg)
    except Exception:
        traceback.print_exc()
        print(f"\n[smoke_run_{label}] FAILED — see traceback above.")
        sys.exit(1)
    print(f"\n[smoke_run_{label}] OK — {len(log)} iteration records.")


if __name__ == "__main__":
    main()
