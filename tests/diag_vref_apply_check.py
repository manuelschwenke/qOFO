"""
Pinpoint where vm_pu_ref gets reset.

Hooks both apply_dso_controls AND DSOController.step to log net.sgen.vm_pu_ref
for one specific sgen at every entry/exit, so we can see exactly which
function call resets the value.

Usage:
    python tests/diag_vref_apply_check.py
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

# Track sgen 14 (first DSO_2 sgen, per per-sgen diagnostic indexing)
TRACK_SGEN = 14
log = []


def _vref(net, idx):
    if "vm_pu_ref" not in net.sgen.columns:
        return None
    return float(net.sgen.at[idx, "vm_pu_ref"])


# Wrap apply_dso_controls
from experiments.helpers import plant_io
orig_apply = plant_io.apply_dso_controls

def patched_apply(net, dso_cfg, dso_out):
    if TRACK_SGEN in list(dso_cfg.der_indices):
        v_before = _vref(net, TRACK_SGEN)
        orig_apply(net, dso_cfg, dso_out)
        v_after = _vref(net, TRACK_SGEN)
        log.append(("APPLY", v_before, v_after, list(dso_cfg.der_indices)[:3]))
    else:
        orig_apply(net, dso_cfg, dso_out)

plant_io.apply_dso_controls = patched_apply
mod.apply_dso_controls = patched_apply

# Wrap pp.runpp
import pandapower as pp
orig_runpp = pp.runpp

def patched_runpp(*args, **kwargs):
    net = args[0] if args else kwargs.get("net")
    v_before = _vref(net, TRACK_SGEN) if net is not None else None
    ret = orig_runpp(*args, **kwargs)
    v_after = _vref(net, TRACK_SGEN) if net is not None else None
    log.append(("RUNPP", v_before, v_after,
                f"rc={kwargs.get('run_control', False)}"))
    return ret

pp.runpp = patched_runpp


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=3.0 * 60.0,    # short — 3 minutes is enough
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

    run_multi_tso_dso(cfg)
    print(f"\n=== vm_pu_ref trace for sgen {TRACK_SGEN} (all events) ===")
    print(f"  {'event':>6}  {'before':>8}  {'after':>8}  notes")
    for kind, vb, va, extra in log:
        vb_s = f"{vb:.4f}" if vb is not None else " None  "
        va_s = f"{va:.4f}" if va is not None else " None  "
        print(f"  {kind:>6}  {vb_s:>8s}  {va_s:>8s}  {extra}")
    print(f"\n  Total events: {len(log)}")


if __name__ == "__main__":
    main()
