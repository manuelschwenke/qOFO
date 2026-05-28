"""Diagnostic: inspect the TSO H matrix's Q_PCC,set columns under
local-sensitivity mode to verify whether the recent fix populates
the V_bus rows (and thus gives the OFO V-tracking leverage from PCC).
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs.multi_tso_config import MultiTSOConfig
from experiments.runners import run_multi_tso_dso


def _hook(state):
    tso_controllers = state["tso_controllers"]
    for z, ctrl in tso_controllers.items():
        # Force a fresh H build
        ctrl._H_cache = None
        H = ctrl._build_sensitivity_matrix()
        n_der = len(ctrl.config.der_indices)
        n_pcc = len(ctrl.config.pcc_trafo_indices)
        n_gen = len(ctrl.config.gen_indices)
        n_gf = len(getattr(ctrl.config, "gridforming_gen_indices", []) or [])
        n_oltc = len(ctrl.config.oltc_trafo_indices)
        n_shunt = len(ctrl.config.shunt_bus_indices)
        n_v = len(ctrl.config.voltage_bus_indices)
        n_i = len(ctrl.config.current_line_indices)
        n_tie = len(getattr(ctrl.config, "tie_line_indices", []) or [])
        print(f"\n=== Zone {z} ({ctrl.controller_id}) ===")
        print(f"H.shape = {H.shape}  (rows=V+Qpcc+I+Qgen+Qgf+Qtie = "
              f"{n_v}+{n_pcc}+{n_i}+{n_gen}+{n_gf}+{n_tie})")
        print(f"          (cols=DER+PCC+Vgen+Vgf+OLTC+shunt = "
              f"{n_der}+{n_pcc}+{n_gen}+{n_gf}+{n_oltc}+{n_shunt})")

        if n_pcc == 0:
            print("  no PCC trafos in this zone — skip")
            continue

        pcc_col_start = n_der
        pcc_col_end = n_der + n_pcc

        # V_bus row block of PCC columns
        H_V_PCC = H[:n_v, pcc_col_start:pcc_col_end]
        # Q_PCC row block of PCC columns  (should be identity)
        H_QPCC_PCC = H[n_v:n_v + n_pcc, pcc_col_start:pcc_col_end]
        # I_line row block of PCC columns
        H_I_PCC = H[n_v + n_pcc:n_v + n_pcc + n_i, pcc_col_start:pcc_col_end]
        # Q_gen row block of PCC columns
        q_gen_start = n_v + n_pcc + n_i
        H_Qgen_PCC = H[q_gen_start:q_gen_start + n_gen, pcc_col_start:pcc_col_end]
        # Q_tie row block of PCC columns
        q_tie_start = n_v + n_pcc + n_i + n_gen + n_gf
        H_Qtie_PCC = H[q_tie_start:q_tie_start + n_tie, pcc_col_start:pcc_col_end]

        print(f"||H_V_PCC||_F  = {np.linalg.norm(H_V_PCC):.4e}   "
              f"max|·|={np.max(np.abs(H_V_PCC)):.4e}")
        print(f"||H_QPCC_PCC|| = {np.linalg.norm(H_QPCC_PCC):.4e}  "
              f"(should be sqrt(n_pcc) ~ {np.sqrt(n_pcc):.2f})")
        print(f"||H_I_PCC||_F  = {np.linalg.norm(H_I_PCC):.4e}   "
              f"max|·|={np.max(np.abs(H_I_PCC)):.4e}")
        print(f"||H_Qgen_PCC|| = {np.linalg.norm(H_Qgen_PCC):.4e}   "
              f"max|·|={np.max(np.abs(H_Qgen_PCC)):.4e}")
        print(f"||H_Qtie_PCC|| = {np.linalg.norm(H_Qtie_PCC):.4e}   "
              f"max|·|={np.max(np.abs(H_Qtie_PCC)):.4e}")
        # First column of V/PCC block for inspection
        if H_V_PCC.shape[1] > 0:
            print(f"H_V_PCC[:, 0] head: "
                  f"{np.array2string(H_V_PCC[:min(5, n_v), 0], precision=3)}")

    return True  # truthy returns from hook short-circuit the main loop


def main():
    for label, flags in [
        ("FULL", dict(local_sensitivities_tso=False, local_sensitivities_dso=False)),
        ("LOCAL", dict(local_sensitivities_tso=True, local_sensitivities_dso=True)),
    ]:
        from datetime import datetime
        cfg = MultiTSOConfig(
            n_total_s=60.0, dt_s=60.0,
            tso_period_s=180.0, dso_period_s=60.0,
            verbose=0, live_plot_controller=False, live_plot_cascade=False,
            live_plot_system=False, run_stability_analysis=False,
            use_profiles=True, use_zonal_gen_dispatch=True,
            scenario="wind_replace",
            start_time=datetime(2016, 4, 15, 10, 0),
            contingencies=[],
            **flags,
        )
        print("\n" + "=" * 72)
        print(f"  Scenario: {label}")
        print("=" * 72)
        run_multi_tso_dso(cfg, pre_loop_hook=_hook)


if __name__ == "__main__":
    main()
