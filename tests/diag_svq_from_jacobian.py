"""
Read the actual S_VQ (= dV_bus / dQ_inj at each DSO DER bus) from the
JacobianSensitivities cached at DSO-controller init.  If S_VQ is much
smaller than my back-of-envelope claim (~0.012), my whole "closed-loop
attenuation" story is wrong and the issue is convergence, not architecture.

Usage:
    python tests/diag_svq_from_jacobian.py
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
from sensitivity.jacobian import JacobianSensitivities

spec = importlib.util.spec_from_file_location(
    "exp_000_m_tso_m_dso",
    ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_multi_tso_dso = mod.run_multi_tso_dso

# Capture the controllers and sensitivities
captured: dict = {"dsocontrollers": None, "sgen_indices": []}

import controller.dso_qv_local_loop as qvl_mod
orig_install = qvl_mod.install_qv_local_loops

def install_and_snapshot(net, sgen_indices, **kwargs):
    captured["sgen_indices"] = list(sgen_indices)
    captured["net_at_install"] = net
    return orig_install(net, sgen_indices, **kwargs)

qvl_mod.install_qv_local_loops = install_and_snapshot
mod.install_qv_local_loops = install_and_snapshot

# Hook the runner just after the DSO controllers are constructed
from controller.dso_controller import DSOController
orig_step = DSOController.step

def patched_step(self, measurement, *, sim_time_s=None):
    if "dsocontrollers" not in captured or captured["dsocontrollers"] is None:
        captured["dsocontrollers"] = {}
    if self.controller_id not in captured["dsocontrollers"]:
        captured["dsocontrollers"][self.controller_id] = self
    return orig_step(self, measurement, sim_time_s=sim_time_s)

DSOController.step = patched_step


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=2.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        g_v=500000.0, g_q=200,
        use_qv_local_loop=True,
        qv_apply_mode="v_ref",
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

    print(f"\n=== S_VQ matrix entries from cached Jacobian per DSO ===\n")

    if not captured.get("dsocontrollers"):
        print("  (no DSO controllers captured)")
        return

    for dso_id in sorted(captured["dsocontrollers"]):
        dso_ctrl = captured["dsocontrollers"][dso_id]
        sens = dso_ctrl.sensitivities
        net = sens.net  # the DSO controller's snapshot of net
        der_indices = list(dso_ctrl.config.der_indices)
        if not der_indices:
            continue
        der_buses = [int(net.sgen.at[s, "bus"]) for s in der_indices]

        try:
            S_VQ_full, obs_map, der_map = sens.compute_dV_dQ_der(
                der_bus_indices=der_buses,
                observation_bus_indices=der_buses,
            )
            obs_perm = [obs_map.index(b) for b in der_buses]
            der_perm = [der_map.index(b) for b in der_buses]
            S_VQ = S_VQ_full[np.ix_(obs_perm, der_perm)]
        except Exception as exc:
            print(f"  {dso_id}: compute_dV_dQ_der failed: {exc!r}")
            continue

        print(f"  {dso_id}: {len(der_buses)} DERs, "
              f"S_VQ matrix {S_VQ.shape}")
        diag = np.diag(S_VQ)
        print(f"    S_VQ diagonal (self-bus) values [pu_v / Mvar]:")
        for s_idx, b, val in zip(der_indices, der_buses, diag):
            sn = float(net.sgen.at[s_idx, "sn_mva"])
            slope = cfg.qv_slope_pu
            k = sn / slope if slope > 0 else 0.0
            kS = k * val
            print(
                f"      sgen {s_idx:3d} (bus {b:3d}, S_n={sn:5.1f}): "
                f"S_VQ = {val:.6f}   "
                f"k = {k:6.0f} Mvar/pu_v   "
                f"k·S_VQ = {kS:6.3f}   "
                f"closed-loop gain factor 1/(1+k·S_VQ) = {1/(1+kS):.3f}"
            )
        print(f"    S_VQ off-diagonal (max): "
              f"{np.max(np.abs(S_VQ - np.diag(diag))):.6f}")
        print()


if __name__ == "__main__":
    main()
