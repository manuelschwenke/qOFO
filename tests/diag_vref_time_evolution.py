"""
Time-evolution diagnostic: track V_ref, Q_actual, and Q_setpoint per DSO
across every timestep so we can see whether the OFO is actually
commanding new V_ref values when the setpoint changes, or whether V_ref
is stuck at one bound the entire run.

Distinguishes three failure modes:
  A) V_ref constant across run     → OFO not commanding (bug in step/apply)
  B) V_ref oscillates with setpoint → OFO works; Q tracking architectural
  C) V_ref interior, doesn't move   → multi-objective stalemate

Usage:
    python tests/diag_vref_time_evolution.py
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


def main() -> None:
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=20.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        g_v=500000.0,
        g_q=200,
        use_qv_local_loop=True,
        dso_g_v=25000.0,
        dso_g_qi=0,
        dso_lambda_qi=0.9,
        dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10,
        g_w_gridforming=1e7,
        g_w_gen=5e7,
        g_w_pcc=50,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        g_w_dso_der=1000,
        g_w_dso_der_vref=1.0,
        g_w_dso_oltc=40,
        qv_slope_pu=0.07,
        qv_v_ref_min_pu=0.95,
        qv_v_ref_max_pu=1.10,
        g_z_warmup_s=0.0,
        verbose=0,
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

    # Capture per-timestep V_ref snapshots by hooking the QV install
    # AND wrapping pp.runpp to snapshot net.sgen.vm_pu_ref each call.
    captured = {"snapshots": [], "dso_sgens": []}
    import pandapower as pp
    orig_runpp = pp.runpp

    def runpp_with_snapshot(*args, **kwargs):
        ret = orig_runpp(*args, **kwargs)
        net = args[0] if args else kwargs.get("net")
        if (
            net is not None
            and "vm_pu_ref" in net.sgen.columns
            and captured["dso_sgens"]
        ):
            snap = {
                int(s): float(net.sgen.at[int(s), "vm_pu_ref"])
                for s in captured["dso_sgens"]
            }
            q_snap = {
                int(s): float(
                    net.res_sgen.at[int(s), "q_mvar"]
                    if hasattr(net, "res_sgen") and int(s) in net.res_sgen.index
                    else 0.0
                )
                for s in captured["dso_sgens"]
            }
            captured["snapshots"].append({"vref": snap, "q": q_snap})
        return ret

    pp.runpp = runpp_with_snapshot

    import controller.dso_qv_local_loop as qvl_mod
    orig_install = qvl_mod.install_qv_local_loops

    def install_and_snapshot(net, sgen_indices, **kwargs):
        captured["dso_sgens"] = list(sgen_indices)
        return orig_install(net, sgen_indices, **kwargs)

    qvl_mod.install_qv_local_loops = install_and_snapshot
    mod.install_qv_local_loops = install_and_snapshot

    try:
        log = run_multi_tso_dso(cfg)
    finally:
        qvl_mod.install_qv_local_loops = orig_install
        mod.install_qv_local_loops = orig_install
        pp.runpp = orig_runpp

    sgens = sorted(captured["dso_sgens"])
    snaps = captured["snapshots"]
    print(f"\n=== Captured {len(snaps)} PF snapshots, {len(sgens)} DSO sgens ===\n")

    if not snaps or not sgens:
        print("  (no snapshots captured)")
        return

    # Pick a representative subset of sgens (one per DSO ideally, but any 4)
    sample_sgens = sgens[: min(8, len(sgens))]

    # V_ref evolution: per snapshot, show min/mean/max V_ref + the sample.
    print("V_ref(per-sgen) evolution across PF snapshots:")
    print(f"  {'snap#':>5}  {'min':>6} {'mean':>6} {'max':>6}  "
          + "  ".join(f"sgen{s:>3}" for s in sample_sgens))
    for i, snap in enumerate(snaps[:: max(1, len(snaps) // 25)]):  # decimate to ~25 rows
        v_arr = np.array([snap["vref"][s] for s in sgens])
        sample = "  ".join(f"{snap['vref'][s]:>7.4f}" for s in sample_sgens)
        print(
            f"  {i:>5d}  {v_arr.min():>6.3f} {v_arr.mean():>6.3f} {v_arr.max():>6.3f}  "
            f"{sample}"
        )

    # How many sgens MOVED at any point?
    initial = snaps[0]["vref"]
    final = snaps[-1]["vref"]
    n_moved = sum(1 for s in sgens if abs(final[s] - initial[s]) > 1e-4)
    n_at_upper = sum(1 for s in sgens if abs(final[s] - cfg.qv_v_ref_max_pu) < 1e-4)
    n_at_lower = sum(1 for s in sgens if abs(final[s] - cfg.qv_v_ref_min_pu) < 1e-4)
    n_interior = len(sgens) - n_at_upper - n_at_lower

    # How many sgens FLIPPED bound (from upper to lower or vice versa across run)?
    n_flipped = 0
    for s in sgens:
        v_series = np.array([snap["vref"][s] for snap in snaps])
        was_at_upper = (np.abs(v_series - cfg.qv_v_ref_max_pu) < 1e-4)
        was_at_lower = (np.abs(v_series - cfg.qv_v_ref_min_pu) < 1e-4)
        if was_at_upper.any() and was_at_lower.any():
            n_flipped += 1

    print(f"\nFinal V_ref distribution:")
    print(f"  {n_at_upper}/{len(sgens)} at upper bound ({cfg.qv_v_ref_max_pu})")
    print(f"  {n_at_lower}/{len(sgens)} at lower bound ({cfg.qv_v_ref_min_pu})")
    print(f"  {n_interior}/{len(sgens)} interior")
    print(f"\nMovement summary across run:")
    print(f"  {n_moved}/{len(sgens)} sgens changed V_ref by > 1e-4")
    print(f"  {n_flipped}/{len(sgens)} sgens visited BOTH bounds at some point")

    # Also dump Q tracking stats for sanity
    if log:
        print("\nDSO Q_iface tracking trajectory (every TSO step):")
        print(f"  {'t/min':>6}  ", end="")
        dso_ids = sorted({d for r in log
                          for d in (getattr(r, "dso_q_actual_mvar", {}) or {})})
        for d in dso_ids:
            print(f"{d}_act/{d}_set  ", end="")
        print()
        for r in log[:: max(1, len(log) // 15)]:
            actuals = getattr(r, "dso_q_actual_mvar", {}) or {}
            targets = getattr(r, "dso_q_set_mvar", {}) or {}
            print(f"  {r.t_min:>6.1f}  ", end="")
            for d in dso_ids:
                a = actuals.get(d)
                t = targets.get(d)
                if a is None or t is None:
                    print(f"   ---/---     ", end="")
                else:
                    print(f"  {float(a):>+5.1f}/{float(t):>+5.1f}  ", end="")
            print()


if __name__ == "__main__":
    main()
