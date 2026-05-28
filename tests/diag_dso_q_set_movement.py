"""
Diagnostic: trace DSO Q_set movement on every controller step
==============================================================

Refactor_v3 introduced a Q_set actuator: the OFO writes the central Q
setpoint into ``net.sgen.q_set_mvar``; the plant-side QVLocalLoop tracks
it 1:1 inside the deadband and overrides via the local droop outside.

This diagnostic patches :meth:`DSOController.step` to log, for each DSO
controller call:

  * ``sim_time_s`` / call counter
  * ``u_current[:n_der]`` — the per-DER Q_set the optimiser starts from
    (set from ``measurement.der_q_mvar`` ⇒ this is the *realised* Q
    coming back from the plant at the previous step)
  * ``u_new[:n_der]`` — the Q_set the optimiser commands this step
  * ``Δu = u_new - u_current`` (per DER)
  * ``q_min, q_max`` — capability envelope at the current op point
  * ``V_meas`` at each DER bus
  * ``in_deadband`` — bool per DER: ``|V_meas - V_ref + Q_set/R| ≤ db``;
    when False the local droop forces Q regardless of Q_set, so the OFO
    has no authority over realised Q at that DER until V re-enters the
    band
  * ``Q_iface_set / Q_iface_meas`` — interface-Q tracking error
  * ``H_iface_DER``\ ``_max`` — max abs gradient on the DER block (if 0,
    optimiser sees no incentive to move Q_set)
  * ``q_set_mvar_post`` — value of ``net.sgen.q_set_mvar`` after
    ``apply_dso_controls`` writes through (sanity-checks the apply path)

Interpretation:

* If ``|Δu|`` is consistently ~0 across many steps → optimiser sees no
  gradient (check ``H_iface_DER_max``) or hits a tight bound (``q_min``
  ≈ ``q_max`` ≈ ``u_current``).
* If ``|Δu|`` is non-zero but ``in_deadband`` is False at most DERs →
  the local droop is overriding Q; the OFO has no control authority
  until V is brought back into the deadband.  Either widen the deadband
  in config, or rely on the OLTC + AVR to bring V near V_ref first.
* If ``|Δu|`` is non-zero AND ``in_deadband`` is True but
  ``q_set_mvar_post`` doesn't equal ``u_new`` → ``apply_dso_controls``
  is broken (regression).

Usage (with live plots OFF per the project's testing convention):

    python tests/diag_dso_q_set_movement.py
    python tests/diag_dso_q_set_movement.py --short          # 5-min run
    python tests/diag_dso_q_set_movement.py --dso DSO_2      # one DSO
"""

from __future__ import annotations

import argparse
import io
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout so the tabular output renders cleanly on Windows.
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

from controller.dso_controller import DSOController


# ---------------------------------------------------------------------------
#  Per-step capture
# ---------------------------------------------------------------------------

step_log: List[Dict] = []
call_counts: Dict[str, int] = {}
dso_filter: str = ""


def patched_dso_step(self, measurement, *, sim_time_s=None):
    """Wrapper that captures DSO state before and after every step."""
    n_der = len(self.config.der_indices)
    net = self.sensitivities.net

    # u_current is set by initialise()/step() from measurement.der_q_mvar
    # so this captures the realised Q each DER fed in at the previous PF.
    u_current = (
        np.asarray(self._u_current[:n_der], dtype=float).copy()
        if self._u_current is not None
        else np.zeros(n_der)
    )

    # Per-DER bus indices and V_ref / db / R for the in-deadband check.
    der_bus = np.array(
        [int(net.sgen.at[int(s), "bus"]) for s in self.config.der_indices],
        dtype=np.int64,
    )

    def _read_or(col: str, fallback: float) -> np.ndarray:
        if col not in net.sgen.columns:
            return np.full(n_der, fallback, dtype=float)
        return np.array(
            [
                float(net.sgen.at[int(s), col])
                if not __import__("pandas").isna(net.sgen.at[int(s), col])
                else fallback
                for s in self.config.der_indices
            ],
            dtype=float,
        )

    sn = np.array(
        [float(net.sgen.at[int(s), "sn_mva"]) for s in self.config.der_indices],
        dtype=float,
    )
    slope = _read_or("qv_slope_pu", 0.07)
    v_ref = _read_or("qv_vref_pu", 1.03)
    db = _read_or("qv_deadband_pu", 0.02)
    R = np.where(slope > 0, sn / np.maximum(slope, 1e-12), 0.0)

    v_meas = np.array(
        [
            float(net.res_bus.at[int(b), "vm_pu"])
            if int(b) in net.res_bus.index else float("nan")
            for b in der_bus
        ],
        dtype=float,
    )

    # Setpoints the OFO is currently tracking (interface Q targets).
    q_set_iface = self.q_setpoint_mvar.copy()
    q_meas_iface = measurement.interface_q_hv_side_mvar.copy()

    # Per-DER capability at the current operating point.
    der_p = self._extract_der_active_power(measurement)
    q_min, q_max = self.actuator_bounds.compute_der_q_bounds(der_p)

    # Run the actual step — this populates self._H_cache and returns u_new.
    out = orig_step(self, measurement, sim_time_s=sim_time_s)
    u_new = np.asarray(out.u_new[:n_der], dtype=float).copy()

    # Pull max-abs gradient on the iface×DER block from the now-cached H.
    n_iface = len(self.config.interface_trafo_indices)
    h_max = float("nan")
    if (
        self._H_cache is not None
        and n_iface > 0
        and self._H_cache.shape[1] >= n_der
    ):
        H_expanded = self._expand_H_to_der_level(self._H_cache)
        if H_expanded.shape[0] >= n_iface and H_expanded.shape[1] >= n_der:
            h_block = H_expanded[:n_iface, :n_der]
            h_max = float(np.max(np.abs(h_block))) if h_block.size else float("nan")

    # In-deadband check uses the *commanded* Q_set (u_new) since that's
    # what the plant will see on the next PF.
    v_eff = v_meas - v_ref + np.where(R > 0, u_new / np.maximum(R, 1e-12), 0.0)
    in_db = np.abs(v_eff) <= db

    call_counts[self.controller_id] = call_counts.get(self.controller_id, 0) + 1

    if not dso_filter or self.controller_id == dso_filter:
        step_log.append({
            "controller_id": self.controller_id,
            "call": call_counts[self.controller_id],
            "t": float(sim_time_s) if sim_time_s is not None else 0.0,
            "n_der": n_der,
            "u_current": u_current,
            "u_new": u_new,
            "delta_u": u_new - u_current,
            "q_min": np.asarray(q_min, dtype=float),
            "q_max": np.asarray(q_max, dtype=float),
            "v_meas": v_meas,
            "v_ref": v_ref,
            "db": db,
            "in_deadband": in_db,
            "q_set_iface": q_set_iface,
            "q_meas_iface": q_meas_iface,
            "h_max": h_max,
        })

    return out


orig_step = DSOController.step
DSOController.step = patched_dso_step


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _summary_line(entry: Dict) -> str:
    delta_u_l1 = float(np.sum(np.abs(entry["delta_u"])))
    delta_u_max = float(np.max(np.abs(entry["delta_u"])))
    n_in = int(np.sum(entry["in_deadband"]))
    n_total = int(entry["n_der"])
    iface_err_l1 = float(
        np.sum(np.abs(entry["q_set_iface"] - entry["q_meas_iface"]))
    )
    v_dev_max = float(
        np.max(np.abs(entry["v_meas"] - entry["v_ref"]))
        if entry["v_meas"].size else 0.0
    )
    return (
        f"  call={entry['call']:>3}  t={entry['t']:>5.0f}s  "
        f"|Δu|_1={delta_u_l1:>7.3f}  |Δu|_∞={delta_u_max:>6.3f}  "
        f"in_db={n_in}/{n_total}  "
        f"|q_iface_err|_1={iface_err_l1:>7.2f}  "
        f"|V−V_ref|_∞={v_dev_max:>5.3f}  "
        f"|H_ie|_∞={entry['h_max']:>7.4f}"
    )


def _detail_line(entry: Dict) -> str:
    """Per-DER detail (compact, only first/last few if many DERs)."""
    n = entry["n_der"]
    if n == 0:
        return ""
    idx_show = list(range(min(3, n)))
    if n > 6:
        idx_show += [n - 1]
    parts = []
    for i in idx_show:
        parts.append(
            f"    DER#{i}: V={entry['v_meas'][i]:.3f}  "
            f"u={entry['u_current'][i]:>+7.2f}→{entry['u_new'][i]:>+7.2f}  "
            f"[Δ={entry['delta_u'][i]:>+6.2f}]  "
            f"caps=[{entry['q_min'][i]:>+5.1f},{entry['q_max'][i]:>+5.1f}]  "
            f"db={'IN' if entry['in_deadband'][i] else 'OUT'}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--short", action="store_true",
        help="5-minute run for fast iteration (default: 30 min).",
    )
    parser.add_argument(
        "--dso", type=str, default="",
        help="Filter step log to a single DSO controller id "
             "(e.g. 'DSO_2').  Default: log every DSO.",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Print per-DER detail rows (V, u, caps, in_deadband) under "
             "every step summary line.",
    )
    args = parser.parse_args()

    global dso_filter
    dso_filter = args.dso

    cfg = MultiTSOConfig(
        dt_s=60.0,
        n_total_s=60.0 * (5 if args.short else 30),
        tso_period_s=60.0 * 3,
        dso_period_s=20.0,
        g_v=5e5, g_q=200, tso_g_q_tie=1,
        dso_g_v=30000.0,
        dso_g_qi=0.0, dso_lambda_qi=0.95, dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,
        g_w_der=10, g_w_gen=5e7, g_w_pcc=50,
        g_w_tso_oltc=100, install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        g_w_dso_der=1000, g_w_dso_oltc=40,
        use_fixed_zones=True,
        run_stability_analysis=False,
        sensitivity_update_interval=1e6,
        verbose=0,
        live_plot_controller=False,
        live_plot_cascade=False,
        live_plot_system=False,
        start_time=datetime(2016, 4, 15, 10, 0),
        use_profiles=True, use_zonal_gen_dispatch=True,
        contingencies=[],
    )

    print(f"\n[diag] Running {cfg.n_total_s:.0f}s simulation "
          f"(dso_period={cfg.dso_period_s:.0f}s)...")
    if dso_filter:
        print(f"[diag] Filtering log to controller_id == {dso_filter!r}")
    run_multi_tso_dso(cfg)

    if not step_log:
        print("\n[diag] No DSO steps were captured — check the filter or "
              "the dso_period_s setting.")
        return

    # Group by controller and print a summary table per controller.
    by_dso: Dict[str, List[Dict]] = {}
    for e in step_log:
        by_dso.setdefault(e["controller_id"], []).append(e)

    for did, entries in by_dso.items():
        print(f"\n=== {did}: {len(entries)} step(s) ===")
        for e in entries:
            print(_summary_line(e))
            if args.detail:
                print(_detail_line(e))

    # Aggregate movement metrics across all logged steps.
    print("\n=== Aggregate ===")
    for did, entries in by_dso.items():
        all_du = np.concatenate([e["delta_u"] for e in entries])
        n_zero = int(np.sum(np.abs(all_du) < 1e-6))
        n_total = int(all_du.size)
        n_in_db_calls = int(sum(int(e["in_deadband"].any()) for e in entries))
        n_full_in_db_calls = int(
            sum(int(e["in_deadband"].all()) for e in entries)
        )
        print(
            f"  {did}: {len(entries)} step(s), "
            f"{n_zero}/{n_total} per-DER Δu were exactly 0, "
            f"max |Δu| = {float(np.max(np.abs(all_du))):.3f} Mvar, "
            f"steps with at least one DER in deadband = {n_in_db_calls}, "
            f"steps with ALL DERs in deadband = {n_full_in_db_calls}"
        )

    print(f"\nTotal call counts: {call_counts}")


if __name__ == "__main__":
    main()
