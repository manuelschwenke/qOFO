"""Full multi-zone TSO+DSO OFO run with contingencies, using the actual
000_M_TSO_M_DSO main() configuration.

Two variants:
  KNOB=stock_main      -- exact main() defaults (g_qi=0).  Baseline.
  KNOB=tuned_integrator -- main() defaults + dso_g_qi=50 + lambda=0.95 +
                           anti-windup raised to 200.  Recommended config
                           that hit <1 Mvar in the isolated DSO test.

Other knobs:
  N_TOTAL_S=600        -- override simulation length (default 240 min,
                           covers contingencies at min 90 / 120 / 180).
"""

from __future__ import annotations

import io
import os
import sys
import importlib
import traceback
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.multi_tso_config import MultiTSOConfig  # noqa: E402
from experiments.helpers.contingency import ContingencyEvent  # noqa: E402

_runner = importlib.import_module("experiments.000_M_TSO_M_DSO")
run_multi_tso_dso = _runner.run_multi_tso_dso


def main() -> None:
    knob = os.environ.get("KNOB", "stock_main")
    n_total_s = float(os.environ.get("N_TOTAL_S", str(60.0 * 60 * 4)))  # 4h default

    no_cont = os.environ.get("NO_CONT", "0") == "1"
    base_kwargs = dict(
        n_total_s=n_total_s,
        tso_period_s=60.0 * 3,
        dso_period_s=10.0,
        g_v=150000.0,
        dso_g_v=20000.0,
        dso_g_qi=0,
        dso_lambda_qi=0.9,
        dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10,
        g_w_gen=4e7,
        g_w_dso_der=1000,
        g_w_dso_oltc=30,
        use_fixed_zones=True,
        run_stability_analysis=False,  # skip the markdown-write path
        sensitivity_update_interval=int(1e6),
        verbose=1,
        live_plot_system=False,
        live_plot_controller=False,
        live_plot_cascade=False,
        start_time=datetime(2016, 4, 15, 12, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=([] if no_cont else [
            ContingencyEvent(minute=90, element_type="gen", element_index=5, action="trip"),
            ContingencyEvent(minute=180, element_type="gen", element_index=5, action="restore"),
            ContingencyEvent(minute=120, element_type="load", bus=5, p_mw=400, q_mvar=200, action="connect"),
            ContingencyEvent(minute=300, element_type="load", bus=5, p_mw=400, q_mvar=200, action="trip"),
            ContingencyEvent(minute=330, element_type="gen", element_index=4, action="trip"),
            ContingencyEvent(minute=420, element_type="gen", element_index=4, action="restore"),
            ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
        ]),
    )

    cfg = MultiTSOConfig(
        **base_kwargs,
        g_q=200,           # main() Scenario A coordinated TSO-DSO
        g_w_pcc=100,
        g_w_tso_oltc=100,
    )

    if knob == "tuned_integrator":
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
        cfg.g_w_dso_der = 1000.0  # already at this value
    elif knob == "tuned_no_v":
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
        cfg.dso_g_v = 0.0  # disable V-tracking entirely
    elif knob == "direct_q":
        # refactor_v3: legacy direct-Q mode (no QVLocalLoop) is gone.
        # Approximate by switching to cosphi=1 — the plant-side
        # CosPhiConstLoop pins Q=0 regardless of voltage; the OFO can
        # still command via Q_set but it is overridden by the cosphi
        # constraint.  H is open-loop in either case.
        cfg.tso_q_mode = "cosphi"
        cfg.dso_q_mode = "cosphi"
        cfg.dso_g_qi = 50.0  # same integrator as tuned config
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
    elif knob == "direct_q_no_int":
        # See note above on direct_q.
        cfg.tso_q_mode = "cosphi"
        cfg.dso_q_mode = "cosphi"

    print(f"\n[full_run] knob={knob}")
    print(f"[full_run] n_total_s={cfg.n_total_s:.0f}s ({cfg.n_total_s/60:.0f} min)")
    print(f"[full_run] g_q={cfg.g_q}  g_w_dso_der={cfg.g_w_dso_der}  "
          f"dso_g_v={cfg.dso_g_v}  dso_g_qi={cfg.dso_g_qi}")
    print(f"[full_run] contingencies: {len(cfg.contingencies)} scheduled")

    try:
        log = run_multi_tso_dso(cfg)
    except Exception:
        traceback.print_exc()
        print("[full_run] FAILED")
        sys.exit(1)
    print(f"\n[full_run] OK -- {len(log)} iteration records.")


if __name__ == "__main__":
    main()
