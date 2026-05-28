"""FD validation of the V_obs and Q_iface rows of H against actual OLTC
tap perturbation, with the closed-loop correction applied.

If ``H_V_obs[:, OLTC_j]` has the wrong sign vs FD ground truth, the OFO
will tap the wrong way for V tracking.  This script computes:

  predicted[i, j] = H_iface_or_V[i, OLTC_j]   (DSO controller's H, post-T'
                                              and post-closed-loop correction)
  actual[i, j]    = (y_after - y_before) / Δs_j   (FD via real PF)

and reports the sign + ratio per (output_row, OLTC) pair.  Both the
Q_iface and V_obs blocks are tested.
"""

from __future__ import annotations

import io
import sys
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandapower as pp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.multi_tso_config import MultiTSOConfig  # noqa: E402
from controller.der_qv_local_loop import seed_qv_equilibrium  # noqa: E402
from core.measurement import measure_zone_dso  # noqa: E402

_runner = importlib.import_module("experiments.000_M_TSO_M_DSO")
run_multi_tso_dso = _runner.run_multi_tso_dso


def _runpp_closed(net, cfg, all_der_indices, shared_jac):
    seed_qv_equilibrium(net, all_der_indices, shared_jac)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, max_iter=500,
             distributed_slack=cfg.distributed_slack,
             enforce_q_lims=cfg.enforce_q_lims_plant)


def _runpp_open(net, cfg):
    """Open-loop PF: do NOT iterate QVLocalLoop (Q_DER stays put)."""
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             max_iteration=50,
             distributed_slack=cfg.distributed_slack,
             enforce_q_lims=cfg.enforce_q_lims_plant)


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=60.0,
        tso_period_s=180.0,
        dso_period_s=60.0,
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
    )

    captured: Dict[str, Any] = {}

    def hook(state):
        captured.update(state)
        return True

    run_multi_tso_dso(cfg, pre_loop_hook=hook)
    net = captured["net"]
    meta = captured["meta"]
    shared_jac = captured["shared_jac"]
    all_der = list(meta.tso_der_indices) + list(meta.dso_der_indices)

    for dso_id in ["DSO_1", "DSO_2", "DSO_3", "DSO_4"]:
        dso_ctrl = captured["dso_controllers"][dso_id]
        cfg_d = dso_ctrl.config
        oltc_indices = list(cfg_d.oltc_trafo_indices)
        if not oltc_indices:
            continue
        v_buses = list(cfg_d.voltage_bus_indices)
        iface_indices = list(cfg_d.interface_trafo_indices)
        n_iface = len(iface_indices)
        n_v = len(v_buses)
        n_oltc = len(oltc_indices)

        print()
        print("=" * 78)
        print(f"  OLTC FD check -- DSO '{dso_id}'  ({n_oltc} OLTCs, {n_v} V-obs)")
        print("=" * 78)

        # H matrix (post-T' + closed-loop correction) -- build by triggering
        # the controller's own pipeline.
        meas = measure_zone_dso(net, cfg_d, 0)
        dso_ctrl._last_measurement = meas
        dso_ctrl.invalidate_sensitivity_cache()
        H = dso_ctrl._build_sensitivity_matrix()
        H = dso_ctrl._expand_H_to_der_level(H)
        # Layout: [DER (n_der_total) | V_gf (n_gf) | OLTC | shunt]
        # Rows: [Q_iface (n_iface) | V (n_v) | I | Q_gf | Q_real?]
        from controller.dso_controller import DSOControllerConfig
        n_der_total = (
            cfg_d.der_mapping.n_der if cfg_d.der_mapping is not None
            else len(cfg_d.der_indices)
        )
        n_gf = len(cfg_d.gridforming_gen_indices)
        oltc_col_start = n_der_total + n_gf
        oltc_col_end = oltc_col_start + n_oltc

        H_iface_OLTC = H[:n_iface, oltc_col_start:oltc_col_end]
        H_V_OLTC = H[n_iface:n_iface + n_v, oltc_col_start:oltc_col_end]

        # FD baseline: re-converge then snapshot
        _runpp_closed(net, cfg, all_der, shared_jac)
        q_iface_base = np.array(
            [float(net.res_trafo3w.at[t, "q_hv_mvar"]) for t in iface_indices]
        )
        v_obs_base = np.array(
            [float(net.res_bus.at[b, "vm_pu"]) for b in v_buses]
        )
        # Save tap positions to restore.
        is_3w_oltc = oltc_indices[0] in net.trafo3w.index
        if is_3w_oltc:
            tap_old = {t: int(net.trafo3w.at[t, "tap_pos"]) for t in oltc_indices}
        else:
            tap_old = {t: int(net.trafo.at[t, "tap_pos"]) for t in oltc_indices}

        # Snapshot open-loop baseline too (no run_control)
        _runpp_open(net, cfg)
        q_iface_base_open = np.array(
            [float(net.res_trafo3w.at[t, "q_hv_mvar"]) for t in iface_indices]
        )
        v_obs_base_open = np.array(
            [float(net.res_bus.at[b, "vm_pu"]) for b in v_buses]
        )
        # Restore closed-loop equilibrium
        _runpp_closed(net, cfg, all_der, shared_jac)

        delta = 1  # +1 tap step
        for j, t_idx in enumerate(oltc_indices):
            # ---- Closed-loop FD ----
            if is_3w_oltc:
                net.trafo3w.at[t_idx, "tap_pos"] = int(tap_old[t_idx] + delta)
            else:
                net.trafo.at[t_idx, "tap_pos"] = int(tap_old[t_idx] + delta)
            try:
                _runpp_closed(net, cfg, all_der, shared_jac)
            except pp.LoadflowNotConverged:
                if is_3w_oltc:
                    net.trafo3w.at[t_idx, "tap_pos"] = int(tap_old[t_idx])
                else:
                    net.trafo.at[t_idx, "tap_pos"] = int(tap_old[t_idx])
                continue
            q_iface_new = np.array(
                [float(net.res_trafo3w.at[t, "q_hv_mvar"]) for t in iface_indices]
            )
            v_obs_new = np.array(
                [float(net.res_bus.at[b, "vm_pu"]) for b in v_buses]
            )
            d_iface_closed = (q_iface_new - q_iface_base) / delta
            d_v_closed = (v_obs_new - v_obs_base) / delta

            # ---- Open-loop FD ----
            try:
                _runpp_open(net, cfg)
            except pp.LoadflowNotConverged:
                d_iface_open = np.full_like(d_iface_closed, np.nan)
                d_v_open = np.full_like(d_v_closed, np.nan)
            else:
                q_iface_new_open = np.array(
                    [float(net.res_trafo3w.at[t, "q_hv_mvar"]) for t in iface_indices]
                )
                v_obs_new_open = np.array(
                    [float(net.res_bus.at[b, "vm_pu"]) for b in v_buses]
                )
                d_iface_open = (q_iface_new_open - q_iface_base_open) / delta
                d_v_open = (v_obs_new_open - v_obs_base_open) / delta

            # Restore + reconverge for next perturbation
            if is_3w_oltc:
                net.trafo3w.at[t_idx, "tap_pos"] = int(tap_old[t_idx])
            else:
                net.trafo.at[t_idx, "tap_pos"] = int(tap_old[t_idx])
            _runpp_closed(net, cfg, all_der, shared_jac)
            d_iface = d_iface_closed
            d_v = d_v_closed

            # Report Q_iface row -- show open-loop, closed-loop, predicted
            print(f"  OLTC[{j}] (trafo3w {t_idx}):")
            for i in range(n_iface):
                pred = H_iface_OLTC[i, j]
                act_o = d_iface_open[i]
                act_c = d_iface[i]
                print(f"    Q_iface[{i}] (trafo3w {iface_indices[i]}): "
                      f"pred={pred:+.4f}  open_FD={act_o:+.4f}  "
                      f"closed_FD={act_c:+.4f}")
            # Report V_obs sample (first 3 buses) -- open-loop FD vs closed-loop FD vs pred
            n_sample = min(3, n_v)
            print(f"    V_obs (first {n_sample} of {n_v}):")
            for i in range(n_sample):
                pred = H_V_OLTC[i, j]
                act_o = d_v_open[i]
                act_c = d_v[i]
                print(f"      V[{i}] (bus {v_buses[i]}): "
                      f"pred={pred:+.5f}  open_FD={act_o:+.5f}  "
                      f"closed_FD={act_c:+.5f}")

    print()
    print("[diag_oltc] OK")


if __name__ == "__main__":
    main()
