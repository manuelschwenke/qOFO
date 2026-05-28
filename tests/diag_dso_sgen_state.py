"""
Diagnostic: print per-DSO-sgen state (V_ref, V_bus, P, Q, Q_cap) at the
end of a short run that mirrors the production main() config, to
resolve the puzzle of why Q ≈ 30 Mvar per DSO seems "stuck" — i.e. not
at V_ref bound, not at converter PQ cap.

Usage:
    python tests/diag_dso_sgen_state.py
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


def main() -> None:
    # Mirror production main() but short (15 min, past warmup) and no
    # contingencies (avoid the trip).  Capture the net via the install
    # hook so we can inspect sgen state at end.
    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=20.0 * 60.0,                # 20 min — past 15-min warmup
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        # Production main() values (mirrored from line 3299+):
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
        g_w_dso_der_vref=1.0,                  # the inert one
        g_w_dso_oltc=40,
        # Stage 2 q(v) defaults
        qv_slope_pu=0.07,
        qv_v_ref_min_pu=0.95,
        qv_v_ref_max_pu=1.10,
        # ── headless ─────────────────────────────────────────────────
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

    captured = {}
    import controller.dso_qv_local_loop as qvl_mod
    orig_install = qvl_mod.install_qv_local_loops

    def install_and_snapshot(net, sgen_indices, **kwargs):
        captured["net"] = net
        captured["dso_sgen_indices"] = list(sgen_indices)
        captured["v_ref_initial"] = {
            int(s): float(net.sgen.at[int(s), "vm_pu_ref"])
            for s in sgen_indices
        }
        return orig_install(net, sgen_indices, **kwargs)

    qvl_mod.install_qv_local_loops = install_and_snapshot
    mod.install_qv_local_loops = install_and_snapshot
    try:
        log = run_multi_tso_dso(cfg)
    finally:
        qvl_mod.install_qv_local_loops = orig_install
        mod.install_qv_local_loops = orig_install

    net = captured["net"]
    sgens = captured["dso_sgen_indices"]
    init_map = captured["v_ref_initial"]

    print(f"\n=== Per-DSO-sgen state at t={cfg.n_total_s/60:.0f} min ===")
    print(
        f"  {'sgen':>5}  {'subnet':>6}  {'S_n':>6}  {'V_ref':>7}  "
        f"{'ΔV_ref':>8}  {'V_bus':>7}  {'V-V_ref':>9}  "
        f"{'P':>7}  {'Q_act':>8}  {'Q_cap':>8}  {'|Q|/Q_cap':>10}  "
        f"{'-k(V-Vr)':>10}  {'state':>9}"
    )
    print("  " + "-" * 122)

    from controller.dso_qv_local_loop import _qv_capability
    slope = cfg.qv_slope_pu
    n_at_qcap = 0
    n_at_vbnd = 0
    n_qsmall = 0
    for s in sorted(sgens):
        sn = float(net.sgen.at[int(s), "sn_mva"])
        bus = int(net.sgen.at[int(s), "bus"])
        v_ref = float(net.sgen.at[int(s), "vm_pu_ref"])
        v_ref_init = init_map[int(s)]
        v_bus = float(net.res_bus.at[bus, "vm_pu"])
        p = float(net.res_sgen.at[int(s), "p_mw"])
        q = float(net.res_sgen.at[int(s), "q_mvar"])
        op_diag = (
            str(net.sgen.at[int(s), "op_diagram"])
            if "op_diagram" in net.sgen.columns else "VDE-AR-N-4120-v2"
        )
        subnet = (
            str(net.sgen.at[int(s), "subnet"])
            if "subnet" in net.sgen.columns else "?"
        )
        q_min, q_max = _qv_capability(sn, op_diag, p)
        q_cap = max(abs(q_min), abs(q_max))
        q_droop = -(sn / slope) * (v_bus - v_ref)
        at_qcap = abs(q) >= q_cap - 0.5 if q_cap > 0 else False
        at_vbnd = (
            abs(v_ref - cfg.qv_v_ref_min_pu) < 1e-4
            or abs(v_ref - cfg.qv_v_ref_max_pu) < 1e-4
        )
        qsmall = abs(q) < 1.0  # less than 1 Mvar
        if at_qcap:
            n_at_qcap += 1
        if at_vbnd:
            n_at_vbnd += 1
        if qsmall:
            n_qsmall += 1
        state = []
        if at_qcap:
            state.append("Qcap")
        if at_vbnd:
            state.append("Vbnd")
        if qsmall and not at_qcap:
            state.append("Qsm")
        state_s = ",".join(state) or "-"
        ratio = abs(q) / q_cap if q_cap > 0 else float("nan")
        print(
            f"  {int(s):>5d}  {subnet:>6s}  {sn:>6.1f}  {v_ref:>7.4f}  "
            f"{v_ref - v_ref_init:>+8.4f}  {v_bus:>7.4f}  "
            f"{v_bus - v_ref:>+9.4f}  "
            f"{p:>7.2f}  {q:>+8.2f}  {q_cap:>8.2f}  {ratio:>10.2%}  "
            f"{q_droop:>+10.2f}  {state_s:>9s}"
        )
    print("  " + "-" * 122)
    print(
        f"  Summary: {n_at_qcap}/{len(sgens)} at Q-cap;  "
        f"{n_at_vbnd}/{len(sgens)} at V_ref bound;  "
        f"{n_qsmall}/{len(sgens)} with |Q| < 1 Mvar"
    )

    # Overall DSO Q-iface tracking
    if log:
        last = log[-1]
        actuals = getattr(last, "dso_q_actual_mvar", {}) or {}
        targets = getattr(last, "dso_q_set_mvar", {}) or {}
        print(f"\n  DSO Q_iface tracking @ t={cfg.n_total_s/60:.0f} min:")
        for did in sorted(actuals):
            a = actuals.get(did)
            t = targets.get(did)
            if a is not None and t is not None:
                print(
                    f"    {did}: actual={float(a):+8.2f}  "
                    f"setpoint={float(t):+8.2f}  err={abs(float(a) - float(t)):6.2f} Mvar"
                )


if __name__ == "__main__":
    main()
