"""Diagnostic sweep: run Stage 2 with several (V_ref bound, slope, g_w)
combinations and report Q-tracking + V_ref motion at the end.

Mirrors the values currently set in ``main()`` of
``experiments/000_M_TSO_M_DSO.py`` so the comparison is apples-to-apples.

Usage:
    python tests/diag_stage2_steps.py
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from configs.multi_tso_config import MultiTSOConfig

import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_000_m_tso_m_dso",
    ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_multi_tso_dso = mod.run_multi_tso_dso


def make_cfg(label: str, *,
             vref_lo: float = 0.95, vref_hi: float = 1.10,
             slope: float = 0.07,
             g_w_dso_der_vref: float = 1.0,
             g_w_gridforming: float = 1e7,
             g_q: float = 200,
             dso_g_v: float = 25000.0,
             g_z_q_dso_der: float = 1e2,
             g_z_warmup_s: float = 0.0) -> MultiTSOConfig:
    """Mirror the production main() config but override selected knobs.

    ``g_z_warmup_s=0`` defaults the slacks to their target values
    immediately so the diagnostic's short 10-minute window actually
    exercises the soft Q_realized output bound (the production main()
    uses 900s warmup to ramp slacks slowly).
    """
    return MultiTSOConfig(
        # ── timing (short for diagnostic) ────────────────────────────
        dt_s=60.0,
        n_total_s=10.0 * 60.0,
        tso_period_s=3.0 * 60.0,
        dso_period_s=10.0,
        g_z_warmup_s=g_z_warmup_s,
        g_z_q_dso_der=g_z_q_dso_der,
        # ── production main() values ─────────────────────────────────
        g_v=500000.0,
        g_q=g_q,
        use_qv_local_loop=True,
        dso_g_v=dso_g_v,
        dso_g_qi=0,
        dso_lambda_qi=0.9,
        dso_q_integral_max_mvar=50.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10,
        g_w_gen=5e7,
        g_w_pcc=50,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        g_w_dso_der=1000,
        g_w_dso_oltc=40,
        # ── diagnostic-swept knobs ───────────────────────────────────
        qv_slope_pu=slope,
        qv_v_ref_min_pu=vref_lo,
        qv_v_ref_max_pu=vref_hi,
        g_w_dso_der_vref=g_w_dso_der_vref,
        g_w_gridforming=g_w_gridforming,
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


def summarize(label: str, log: list, out=sys.stdout) -> None:
    """Print Q-tracking + actuator-motion summary across the run.

    Writes Q_iface tracking error, V_ref motion (proxied by Q_DER motion
    via the K-transform), and final Q_DER state per zone.
    """
    print(f"\n=== {label} ===", file=out)
    if not log:
        print("  (no log records)", file=out)
        return

    # ── DSO Q_iface tracking quality per DSO ────────────────────────
    # rec.dso_q_actual_mvar[dso_id] : realised Q at the DSO interface (Mvar)
    # rec.dso_q_set_mvar[dso_id]    : setpoint sent by the TSO (Mvar)
    iface_err: dict[str, list[float]] = {}
    final_err: dict[str, float] = {}
    for rec in log:
        actuals = getattr(rec, "dso_q_actual_mvar", {}) or {}
        targets = getattr(rec, "dso_q_set_mvar", {}) or {}
        for dso_id in actuals:
            a = actuals.get(dso_id)
            t = targets.get(dso_id)
            if a is None or t is None:
                continue
            err = abs(float(a) - float(t))
            iface_err.setdefault(dso_id, []).append(err)
            final_err[dso_id] = err
    if iface_err:
        print("  DSO Q_iface |actual - setpoint| (Mvar):", file=out)
        for dso_id in sorted(iface_err):
            arr = np.array(iface_err[dso_id])
            print(
                f"    {dso_id}: mean={arr.mean():7.2f}  max={arr.max():7.2f}  "
                f"final={final_err[dso_id]:7.2f}",
                file=out,
            )
    else:
        print("  (no DSO Q_iface tracking data found in log)", file=out)

    # ── Q_DER final state per zone ──────────────────────────────────
    final_q_per_zone: dict[int, np.ndarray] = {}
    for rec in log:
        for z, q_arr in rec.zone_q_der.items():
            if q_arr is None or len(q_arr) == 0:
                continue
            final_q_per_zone[z] = q_arr.copy()
    print("  final Q_DER per zone:", file=out)
    for z, q_arr in sorted(final_q_per_zone.items()):
        print(
            f"    zone {z}: n_DER={len(q_arr)}  "
            f"Q range=[{q_arr.min():+8.2f}, {q_arr.max():+8.2f}] Mvar  "
            f"mean={q_arr.mean():+8.2f}  std={q_arr.std():6.2f}",
            file=out,
        )

    # ── Q_DER step-to-step Δ (proxy for V_ref jitter) ───────────────
    prev_q: dict[int, np.ndarray] = {}
    max_dq_per_zone: dict[int, list[float]] = {}
    for rec in log:
        for z, q_arr in rec.zone_q_der.items():
            if q_arr is None or len(q_arr) == 0:
                continue
            if z in prev_q and prev_q[z].shape == q_arr.shape:
                dq = q_arr - prev_q[z]
                max_dq_per_zone.setdefault(z, []).append(float(np.abs(dq).max()))
            prev_q[z] = q_arr.copy()
    print("  max |ΔQ_DER|/step (chatter proxy):", file=out)
    for z, dq_list in sorted(max_dq_per_zone.items()):
        if not dq_list:
            continue
        arr = np.array(dq_list)
        print(
            f"    zone {z}: mean={arr.mean():6.2f}  max={arr.max():6.2f}  "
            f"final={arr[-1]:6.2f} Mvar",
            file=out,
        )


def vref_diagnostic(label: str, **cfg_kwargs) -> None:
    """Run one config and dump per-DSO V_ref snapshots before/after to
    confirm whether the DSO OFO is actually moving V_ref, AND confirm
    the DSOControllerConfig.use_qv_local_loop flag flowed through."""
    import importlib
    cfg = make_cfg(label, **cfg_kwargs)

    # Patch run_multi_tso_dso to capture V_ref before/after
    captured = {}

    # We'll use the experiment module's own globals
    orig_run = mod.run_multi_tso_dso

    def instrumented_run(c):
        # Use weakref of net through pp directly: we'll snapshot via
        # monkey-patching install_qv_local_loops (called in the runner
        # right after apply_der_classification).
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
        # Also patch the runner's local import
        mod.install_qv_local_loops = install_and_snapshot
        try:
            log = orig_run(c)
        finally:
            qvl_mod.install_qv_local_loops = orig_install
            mod.install_qv_local_loops = orig_install
        # After run, capture final V_ref state
        net = captured["net"]
        captured["v_ref_final"] = {
            int(s): float(net.sgen.at[int(s), "vm_pu_ref"])
            for s in captured["dso_sgen_indices"]
        }
        return log

    log = instrumented_run(cfg)

    print(f"\n=== {label} ===")
    init_map = captured.get("v_ref_initial", {})
    final_map = captured.get("v_ref_final", {})
    print(f"  DSO V_ref movement summary:")
    if not init_map:
        print("    (no DSO sgens captured — install_qv_local_loops not called!)")
        return
    deltas = []
    for s in sorted(init_map):
        d = final_map[s] - init_map[s]
        deltas.append(d)
        if abs(d) > 1e-6:
            print(f"    sgen {s}: init={init_map[s]:.4f}  final={final_map[s]:.4f}  Δ={d:+.4f}")
    arr = np.array(deltas)
    print(
        f"  ΔV_ref stats over {len(deltas)} sgens: "
        f"mean={arr.mean():+.4f}  max|={np.abs(arr).max():.4f}  "
        f"# moved={int((np.abs(arr) > 1e-6).sum())}/{len(deltas)}"
    )
    if np.abs(arr).max() < 1e-5:
        print("    → DSO OFO is NOT moving V_ref.  Gradient/apply path bug.")
    else:
        print("    → DSO OFO IS moving V_ref.  Tuning issue, not plumbing.")


def main() -> None:
    # NOW with Q_realized soft output rows + steeper droop, g_q should
    # actually move Q-tracking quality.  Sweep g_q at slope=0.02 to
    # confirm the bound saturation is no longer the binding constraint.
    sweeps: List[Tuple[str, dict]] = [
        # Confirm Q_realized soft output rows are now active by sweeping
        # g_q AND the new g_z_q_dso_der (which weights the Q-rail
        # violation slack).  All runs use slope=0.02 (steepest).
        ("g_q=200    slope=0.02  g_z_qreal=1e2  (default soft rail)",
         dict(g_q=200, slope=0.02, g_z_q_dso_der=1e2)),
        ("g_q=200    slope=0.02  g_z_qreal=1e6  (hard rail)",
         dict(g_q=200, slope=0.02, g_z_q_dso_der=1e6)),
        ("g_q=1000   slope=0.02  g_z_qreal=1e6  (5x Q + hard rail)",
         dict(g_q=1000, slope=0.02, g_z_q_dso_der=1e6)),
        ("g_q=10000  slope=0.02  g_z_qreal=1e6  (50x Q + hard rail)",
         dict(g_q=10000, slope=0.02, g_z_q_dso_der=1e6)),
    ]

    summary_path = ROOT / "tests" / "diag_stage2_steps.out"
    with summary_path.open("w", encoding="utf-8") as out:
        for label, kwargs in sweeps:
            cfg = make_cfg(label, **kwargs)
            try:
                log = run_multi_tso_dso(cfg)
            except Exception as exc:
                print(f"\n=== {label} ===\n  FAILED: {exc!r}", file=out)
                print(f"FAILED: {label} ({exc!r})")
                continue
            summarize(label, log, out=out)
            print(f"DONE:   {label}")
    print(f"\nFull summary: {summary_path}")

    summary_path = ROOT / "tests" / "diag_stage2_steps.out"
    with summary_path.open("w", encoding="utf-8") as out:
        for label, kwargs in sweeps:
            cfg = make_cfg(label, **kwargs)
            try:
                log = run_multi_tso_dso(cfg)
            except Exception as exc:
                print(f"\n=== {label} ===\n  FAILED: {exc!r}", file=out)
                print(f"FAILED: {label} ({exc!r})")
                continue
            summarize(label, log, out=out)
            print(f"DONE:   {label}")
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
