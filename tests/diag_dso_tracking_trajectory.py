"""Detailed DSO Q-tracking diagnostic with full trajectory logging.

Reuses the 003_M_DSO_CIGRE_2026 setup (TSO in local Q(V) mode, only
DSO_2 running OFO with fixed Q_pcc setpoints) so the DSO controller is
fully isolated from any TSO MIQP behaviour.  Logs Q_iface, Q_set,
q_error, Q_DER totals, OLTC tap positions, V_gf commands per step --
to nail down the exact source of any residual tracking error.

Run:  python tests/diag_dso_tracking_trajectory.py
"""

from __future__ import annotations

import io
import sys
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandapower as pp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.measurement import measure_zone_dso  # noqa: E402

_runner = importlib.import_module("experiments.000_M_TSO_M_DSO")
run_multi_tso_dso = _runner.run_multi_tso_dso

_003 = importlib.import_module("experiments.003_M_DSO_CIGRE_2026")
make_003_config = _003.make_base_config


def main() -> None:
    cfg = make_003_config()
    # Force live plots OFF (memory rule for tests).
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False

    # Run length: 30 min should be enough for convergence on a fixed setpoint.
    cfg.n_total_s = 30.0 * 60.0
    cfg.dso_period_s = 10.0  # match 003 -- DSO steps every 10 s

    # Targeted experimental knobs to test convergence quality.
    # Defaults from 003: g_q=500, g_w_dso_der=100, dso_g_v=20000, dso_g_qi=0
    # The bias toward V_set=1.03 (g_v=20000) competes with Q tracking and
    # produces a steady-state offset.  Test variants:
    import os
    knob = os.environ.get("KNOB", "default")
    if knob == "no_v_track":
        cfg.dso_g_v = 0.0  # disable V tracking entirely
    elif knob == "with_integrator":
        cfg.dso_g_qi = 50.0  # enable integral action
        cfg.dso_lambda_qi = 0.9
        cfg.dso_q_integral_max_mvar = 200.0
    elif knob == "high_g_q":
        cfg.g_q = 5000.0  # 10x larger Q-tracking weight
    elif knob == "less_aggressive":
        cfg.g_w_dso_der = 1000.0  # 10x larger action change penalty
    elif knob == "feasible_simple":
        # Single-PCC small swing setpoint to isolate convergence quality.
        cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [10.0, 10.0, 40.0]}
        # ~ current values, should converge to zero error trivially.
    elif knob == "best":
        # g_w_dso_der raised to kill oscillation, integrator on to remove
        # steady-state bias from V tracking.
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
    elif knob == "best_no_v":
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_v = 0.0
    elif knob == "best_with_int":
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
        cfg.dso_g_v = 0.0  # disable V-tracking too
    elif knob == "feasible_small_swing":
        # Setpoint that's clearly feasible: small swing from natural state.
        # DSO_2 natural q_iface ~ [+10, +6, +33].  Ask for [+5, +0, +25] --
        # well within capability and not requiring any DER saturation.
        cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [5.0, 0.0, 25.0]}
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
    elif knob == "stay_put":
        # Setpoint = current natural state -> ideal tracking should be 0.
        # Pre-Phase-2 state has q_iface ~[+10, +6, +33] at DSO_2.
        cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [10.6, 5.85, 33.5]}
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
    elif knob == "direct_q_stay_put":
        # refactor_v3: direct-Q ablation no longer exists.  Approximate
        # the old "no QVLocalLoop intervention" behaviour by switching
        # both levels to cosphi=1 mode (Q held at 0 by the plant-side
        # CosPhiConstLoop, OFO can still command via Q_set).
        cfg.tso_q_mode = "cosphi"
        cfg.dso_q_mode = "cosphi"
        cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [10.6, 5.85, 33.5]}
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0
    elif knob == "direct_q_small_swing":
        # See note above on direct_q_stay_put.
        cfg.tso_q_mode = "cosphi"
        cfg.dso_q_mode = "cosphi"
        cfg.q_pcc_setpoints_mvar_per_dso = {"DSO_2": [5.0, 0.0, 25.0]}
        cfg.g_w_dso_der = 1000.0
        cfg.dso_g_qi = 50.0
        cfg.dso_lambda_qi = 0.95
        cfg.dso_q_integral_max_mvar = 200.0

    # Use the 003 default setpoints.  These are well within capability:
    # DSO_2 has 3 PCCs, each with envelope ~[-60, +70] Mvar -- the
    # commanded setpoint [0, -20, +20] is comfortably feasible.
    print(f"\n[diag_traj] knob=                      {knob}")
    print(f"[diag_traj] q_pcc setpoints:           {cfg.q_pcc_setpoints_mvar_per_dso}")
    print(f"[diag_traj] g_q={cfg.g_q}  g_w_dso_der={cfg.g_w_dso_der}  "
          f"dso_g_v={cfg.dso_g_v}  dso_g_qi={cfg.dso_g_qi}")
    print(f"[diag_traj] qv_local_damping={cfg.qv_local_damping}")
    print(f"[diag_traj] n_total_s={cfg.n_total_s:.0f}, dso_period_s={cfg.dso_period_s:.0f}")

    # Per-step state captured by the post-step hook.
    trajectory = {
        "step": [],
        "q_iface": [],   # (n_iface,) per step
        "q_set": [],     # (n_iface,) per step
        "q_der_sum": [], # scalar per step
        "u_der_sum": [], # scalar per step (Q_cor sum)
        "u_oltc": [],    # (n_oltc,) per step
        "u_vgf": [],     # (n_vgf,) per step (if any)
        "q_der_min_count": [],  # number of DERs at min rail
        "q_der_max_count": [],  # number of DERs at max rail
    }

    def post_step_hook(state):
        """Pre-loop hook: replace later via monkey-patching the runner."""
        # We can't easily get per-step data from a pre-loop hook -- need
        # to patch the runner.  Instead, print the post-Phase-2 setup
        # info and then let the run proceed.
        net = state["net"]
        dso_ctrl = state["dso_controllers"].get("DSO_2")
        if dso_ctrl is None:
            print("[diag_traj] DSO_2 controller not found")
            return False  # let the loop run

        cfg = state["config"]
        meas = measure_zone_dso(net, dso_ctrl.config, 0)
        print()
        print("=" * 78)
        print("  POST-PHASE-2 INITIAL STATE -- DSO_2")
        print("=" * 78)
        print(f"  Q_setpoint requested: {cfg.q_pcc_setpoints_mvar_per_dso['DSO_2']}")
        print(f"  Q_iface initial:      {meas.interface_q_hv_side_mvar.tolist()}")
        print(f"  Q_DER initial sum:    {float(np.sum(meas.der_q_mvar)):.2f} Mvar "
              f"(per-DER: {[round(float(q), 2) for q in meas.der_q_mvar]})")
        # Capability envelope.
        cap_msg = dso_ctrl.generate_capability_message(
            target_controller_id="exogenous",
            measurement=meas,
        )
        print(f"  Capability delta:     min={cap_msg.q_min_mvar.tolist()}  "
              f"max={cap_msg.q_max_mvar.tolist()}")
        q_iface_now = meas.interface_q_hv_side_mvar.copy()
        env_min = q_iface_now + cap_msg.q_min_mvar
        env_max = q_iface_now + cap_msg.q_max_mvar
        print(f"  Capability absolute:  min={env_min.tolist()}  max={env_max.tolist()}")
        q_set_arr = np.array(cfg.q_pcc_setpoints_mvar_per_dso["DSO_2"])
        feasible = (env_min - 1.0 <= q_set_arr).all() and (q_set_arr <= env_max + 1.0).all()
        print(f"  Setpoint feasible?    {feasible}  "
              f"(margin: low={(q_set_arr - env_min).tolist()}  "
              f"high={(env_max - q_set_arr).tolist()})")

        return False  # let the main loop proceed

    # We want trajectory data from inside the loop.  The cleanest path:
    # monkey-patch the runner to print a one-line trajectory record per
    # DSO step.  Locate the apply_dso_controls call site inside the
    # runner module and wrap it.
    import experiments.helpers.plant_io as plant_io
    _orig_apply_dso_controls = plant_io.apply_dso_controls

    last_step_record = {"step": [-1]}

    def _apply_dso_controls_logged(net, dso_cfg, dso_out, *args, **kwargs):
        result = _orig_apply_dso_controls(net, dso_cfg, dso_out, *args, **kwargs)
        # Only log once per step (apply_dso_controls is called once per
        # DSO per step; we inspect the global step counter via the
        # caller's frame).
        import inspect
        frame = inspect.currentframe().f_back
        step = frame.f_locals.get("step", -1)
        time_s = frame.f_locals.get("time_s", -1)
        if step == last_step_record["step"][0]:
            return result
        last_step_record["step"][0] = step

        # Run a fresh PF so q_iface reflects the just-applied control.
        # NOTE: this is invasive but read-only on the OFO state.
        try:
            pp.runpp(net, run_control=True, calculate_voltage_angles=True,
                     max_iteration=50, max_iter=500, distributed_slack=False,
                     enforce_q_lims=False)
        except pp.LoadflowNotConverged:
            return result

        # Read the DSO_2 trafos for q_iface.  Uses the dso_cfg we just
        # got passed, which is DSO_2's config.
        n_iface = len(dso_cfg.interface_trafo_indices)
        q_iface = np.array(
            [float(net.res_trafo3w.at[t, "q_hv_mvar"])
             for t in dso_cfg.interface_trafo_indices],
            dtype=np.float64,
        )
        u = np.asarray(dso_out.u_new, dtype=np.float64)
        n_der_local = len(dso_cfg.der_indices)
        u_der = u[:n_der_local]
        # Q_DER totals.
        q_der = np.array(
            [float(net.sgen.at[s, "q_mvar"]) for s in dso_cfg.der_indices],
            dtype=np.float64,
        )
        sn = np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in dso_cfg.der_indices],
            dtype=np.float64,
        )
        n_at_max = int(np.sum(q_der > 0.40 * sn))   # roughly at +Q rail
        n_at_min = int(np.sum(q_der < -0.32 * sn))  # roughly at -Q rail

        trajectory["step"].append(int(step))
        trajectory["q_iface"].append(q_iface.copy())
        trajectory["q_der_sum"].append(float(np.sum(q_der)))
        trajectory["u_der_sum"].append(float(np.sum(u_der)))
        trajectory["q_der_max_count"].append(n_at_max)
        trajectory["q_der_min_count"].append(n_at_min)

        # Print one-liner.
        if step % 20 == 0 or step <= 5:
            print(f"  [t={time_s:6.0f}s step={step:4d}] "
                  f"q_iface={q_iface.tolist()}  "
                  f"Q_DER_sum={np.sum(q_der):+.2f}  "
                  f"u_der_sum={np.sum(u_der):+.2f}  "
                  f"sat_max={n_at_max}/{n_der_local}  sat_min={n_at_min}/{n_der_local}")
        return result

    plant_io.apply_dso_controls = _apply_dso_controls_logged
    # Re-bind the runner's local reference too (it imported the symbol).
    _runner.apply_dso_controls = _apply_dso_controls_logged

    print("\n[diag_traj] running full simulation with trajectory logging ...")
    print(f"  (config: scenario={cfg.scenario}, profiles={cfg.use_profiles}, "
          f"tso_mode={cfg.tso_mode})\n")

    try:
        log = run_multi_tso_dso(cfg, pre_loop_hook=post_step_hook)
    finally:
        # Restore.
        plant_io.apply_dso_controls = _orig_apply_dso_controls
        _runner.apply_dso_controls = _orig_apply_dso_controls

    print()
    print("=" * 78)
    print("  TRAJECTORY ANALYSIS")
    print("=" * 78)

    # Final values.
    if not trajectory["step"]:
        print("  No trajectory captured -- log hook did not fire.")
        return

    q_set_arr = np.array(cfg.q_pcc_setpoints_mvar_per_dso["DSO_2"])
    q_iface_arr = np.array(trajectory["q_iface"])  # (n_steps, n_iface)
    n_steps = q_iface_arr.shape[0]
    print(f"  steps logged:              {n_steps}")
    print(f"  Q_set:                     {q_set_arr.tolist()}")
    print(f"  Q_iface initial:           {q_iface_arr[0].tolist()}")
    print(f"  Q_iface final:             {q_iface_arr[-1].tolist()}")
    print(f"  q_error final:             "
          f"{(q_iface_arr[-1] - q_set_arr).tolist()}")
    print(f"  ||q_error|| final (Mvar):  "
          f"{np.linalg.norm(q_iface_arr[-1] - q_set_arr):.3f}")

    # Last-25%-of-run statistics (steady-state).
    tail = q_iface_arr[max(1, 3 * n_steps // 4):]
    if len(tail) >= 2:
        std = tail.std(axis=0)
        mean = tail.mean(axis=0)
        print(f"  Last 25%: Q_iface mean   {mean.tolist()}")
        print(f"  Last 25%: Q_iface std    {std.tolist()}")
        print(f"  Last 25%: q_error mean   "
              f"{(mean - q_set_arr).tolist()}")

    # Saturation summary.
    last_satmax = trajectory["q_der_max_count"][-1] if trajectory["q_der_max_count"] else 0
    last_satmin = trajectory["q_der_min_count"][-1] if trajectory["q_der_min_count"] else 0
    last_qder = trajectory["q_der_sum"][-1] if trajectory["q_der_sum"] else 0.0
    last_uder = trajectory["u_der_sum"][-1] if trajectory["u_der_sum"] else 0.0
    print(f"  Final Q_DER_sum:           {last_qder:+.2f} Mvar")
    print(f"  Final u_der_sum (Q_cor):   {last_uder:+.2f} Mvar")
    print(f"  Final saturated DERs:      max-rail {last_satmax}, min-rail {last_satmin}")


if __name__ == "__main__":
    main()
