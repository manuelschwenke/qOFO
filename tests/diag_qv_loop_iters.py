"""
Count how many control_step calls the QVLocalLoop actually receives
during a single ``pp.runpp(run_control=True)`` call.  If it's only ~5,
that's the root cause of the Q undershoot.  If it's 30+, the loop is
truly converging at an interior closed-loop equilibrium (architectural).

Usage:
    python tests/diag_qv_loop_iters.py
"""

from __future__ import annotations

import io
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

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

# Patch QVLocalLoop to count control_step calls per instance + log Q evol
import controller.dso_qv_local_loop as qvl_mod

iter_counts: dict[int, int] = {}     # sgen_idx → count of control_step calls
last_q: dict[int, float] = {}         # sgen_idx → last Q
q_evolution: dict[int, list] = {}    # sgen_idx → list of (count, q_target, q_actual)

orig_init = qvl_mod.QVLocalLoop.__init__
orig_control_step = qvl_mod.QVLocalLoop.control_step

def patched_control_step(self, net):
    target = self._compute_target(net)
    q_before = float(net.sgen.at[self.sgen_idx, "q_mvar"])
    orig_control_step(self, net)
    q_after = float(net.sgen.at[self.sgen_idx, "q_mvar"])
    iter_counts[self.sgen_idx] = iter_counts.get(self.sgen_idx, 0) + 1
    if self.sgen_idx == 6 or self.sgen_idx == 7:  # log a couple of representative sgens
        q_evolution.setdefault(self.sgen_idx, []).append(
            (iter_counts[self.sgen_idx], target, q_before, q_after)
        )

qvl_mod.QVLocalLoop.control_step = patched_control_step


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=2.0 * 60.0,           # very short — just want a few PF calls
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        g_v=500000.0, g_q=200,
        use_qv_local_loop=True,
        dso_g_v=25000.0,
        dso_g_qi=0, dso_lambda_qi=0.9, dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10, g_w_gridforming=1e7, g_w_gen=5e7, g_w_pcc=50,
        g_w_tso_oltc=100, install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000, g_w_dso_der=1000, g_w_dso_der_vref=1.0,
        g_w_dso_oltc=40,
        qv_slope_pu=0.07, qv_v_ref_min_pu=0.95, qv_v_ref_max_pu=1.10,
        g_z_warmup_s=0.0,
        verbose=0,
        live_plot_controller=False, live_plot_cascade=False, live_plot_system=False,
        run_stability_analysis=False,
        use_profiles=True, use_zonal_gen_dispatch=True,
        scenario="wind_replace",
        start_time=datetime(2016, 4, 15, 10, 0),
        contingencies=[],
    )

    log = run_multi_tso_dso(cfg)
    print(f"\n=== After 2-min run ===")
    print(f"  Total control_step calls per sgen (across ALL pp.runpp invocations):")
    if not iter_counts:
        print("  (none — QVLocalLoop never called)")
        return
    counts = sorted(iter_counts.values())
    print(f"    min={min(counts)}, mean={sum(counts)/len(counts):.0f}, max={max(counts)}")
    print(f"    n sgens tracked: {len(iter_counts)}")

    # Detailed evolution for sgen 6 (first ~30 control_step calls)
    if 6 in q_evolution:
        print(f"\n  control_step trace for sgen 6 (first 30 calls):")
        print(f"    {'#':>4}  {'target':>8}  {'before':>8}  {'after':>8}  {'Δ':>7}")
        for entry in q_evolution[6][:30]:
            count, target, q_before, q_after = entry
            print(
                f"    {count:>4d}  {target:>+8.2f}  {q_before:>+8.2f}  "
                f"{q_after:>+8.2f}  {q_after-q_before:>+7.2f}"
            )


if __name__ == "__main__":
    main()
