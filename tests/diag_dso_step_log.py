"""
Log every DSO controller step():
  - The Q-iface setpoint received from TSO
  - The Q-iface actual measured
  - u_old (V_ref before the step)
  - u_new (V_ref the OFO wants)
  - net.sgen.vm_pu_ref AFTER apply_dso_controls

If u_new == u_old every step → OFO stuck at KKT (no gradient drive).
If u_new ≠ u_old but vm_pu_ref unchanged → apply step bug.
If everything changes but Q_actual flat → Q(V) loop bug or arch issue.

Usage:
    python tests/diag_dso_step_log.py
"""

from __future__ import annotations

import io
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

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

# Patch DSOController.step to log every call
from controller.dso_controller import DSOController
orig_step = DSOController.step

step_log: list = []
call_counts: dict = {}

def patched_step(self, measurement, *, sim_time_s=None):
    n_der = len(self.config.der_indices)
    # Snapshot u_old (V_ref currently in net.sgen.vm_pu_ref)
    net = self.sensitivities.net
    u_old = np.array(
        [float(net.sgen.at[int(s), "vm_pu_ref"])
         for s in self.config.der_indices],
        dtype=float,
    )
    q_set = self.q_setpoint_mvar.copy()
    q_actual = measurement.interface_q_hv_side_mvar.copy()
    out = orig_step(self, measurement, sim_time_s=sim_time_s)
    u_new_vref = out.u_new[:n_der]
    call_counts[self.controller_id] = call_counts.get(self.controller_id, 0) + 1
    if self.controller_id == "DSO_2":  # focus on one DSO
        step_log.append({
            "call": call_counts[self.controller_id],
            "t": sim_time_s,
            "q_set_sum": float(q_set.sum()),
            "q_actual_sum": float(q_actual.sum()),
            "q_err_sum": float((q_actual - q_set).sum()),
            "u_old_mean": float(u_old.mean()),
            "u_old_min": float(u_old.min()),
            "u_old_max": float(u_old.max()),
            "u_new_mean": float(u_new_vref.mean()),
            "u_new_min": float(u_new_vref.min()),
            "u_new_max": float(u_new_vref.max()),
            "u_change": float(np.linalg.norm(u_new_vref - u_old)),
        })
    return out

DSOController.step = patched_step


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=10.0 * 60.0,
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
    print(f"\n=== DSO_2 step log ({len(step_log)} calls) ===\n")
    print(f"  {'#':>4} {'t/s':>5} {'Q_set':>7} {'Q_act':>7} {'err':>7}  "
          f"{'u_old':>16}  {'u_new':>16}  {'|Δu|':>7}")
    for entry in step_log:
        u_old_str = (
            f"[{entry['u_old_min']:.3f}, {entry['u_old_mean']:.3f}, "
            f"{entry['u_old_max']:.3f}]"
        )
        u_new_str = (
            f"[{entry['u_new_min']:.3f}, {entry['u_new_mean']:.3f}, "
            f"{entry['u_new_max']:.3f}]"
        )
        t = entry["t"] if entry["t"] is not None else 0.0
        print(
            f"  {entry['call']:>4d} {t:>5.0f} {entry['q_set_sum']:>+7.1f} "
            f"{entry['q_actual_sum']:>+7.1f} {entry['q_err_sum']:>+7.1f}  "
            f"{u_old_str:>16s}  {u_new_str:>16s}  {entry['u_change']:>7.4f}"
        )
    print(f"\nTotal call counts: {call_counts}")


if __name__ == "__main__":
    main()
