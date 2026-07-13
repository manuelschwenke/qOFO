"""
diag_objective_ratio_sweep.py
=============================
Tier-1 objective-weight sweep, in **normalised priority space** (Step A/B of
the weight-tuning methodology; see ``docs/daily_log/2026-06-23_*``).

The controller weights are derived, not chosen:

    g_i = pi_i / sigma_i**2

where ``sigma_i`` is a fixed engineering tolerance (Step A) and ``pi_i`` is a
dimensionless priority (Step B) — the only thing we sweep.  The Tier-2
curvature preconditioner is kept **ON** throughout, so changing an objective
ratio re-sizes ``g_w`` automatically and can never destabilise the loop: the
sweep measures *performance trade-offs only*.

Reports, per priority combination, the controller-agnostic KPIs from
``cigre_summary_table`` and marks the **Pareto front** of voltage tracking
(``rms_v_ts_pu``, lower better) vs reserve (``m_bar_pu``, higher better).

This re-parameterises the *config*, not the controllers — non-invasive.
Edit ``SIGMA`` / ``PI_BASE`` below to change the tolerances / baseline
priorities.

Usage
-----
    python experiments/diag_objective_ratio_sweep.py
    python experiments/diag_objective_ratio_sweep.py \
        --pi-qpcc 0,10,20 --pi-res 0,1,4 --horizon-min 20 \
        --module experiments.005_CIGRE_MULTI --scenario wind_replace

Author: Manuel Schwenke (with Claude Code)
Date: 2026-06-23
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.helpers.comparison_metrics import cigre_summary_table
from experiments.helpers.records import MultiTSOIterationRecord
from experiments.runners.multi_tso_dso import run_multi_tso_dso


# ── Step A: engineering tolerances (fixed; edit to retune the units) ────────
SIGMA: Dict[str, float] = {
    "v_ts":  0.010,   # pu  — TSO bus voltage tracking tolerance (10 mpu)
    "v_ds":  0.020,   # pu  — DSO bus (looser)
    "q_pcc": 5.0,     # Mvar — interface-Q tracking tolerance
    "q_tie": 50.0,    # Mvar — tie-Q (larger flows, less critical)
    "res":   1.0,     # —    — reserve already normalised (r=1 at the rail)
}

# ── Step B: baseline priorities (dimensionless; the things we sweep) ────────
PI_BASE: Dict[str, float] = {
    # TSO
    "v_ts":  100.0,   # primary controlled output
    "q_pcc": 10.0,    # interface-Q management (secondary)
    "q_tie": 10.0,    # inter-zone coordination
    "res":   1.0,     # reserve centring (soft, lowest)
    # DSO
    "q_dso": 100.0,   # primary: track the TSO setpoint
    "v_ds":  10.0,    # soft voltage nudge
}


# Stage-E: scale integer (OLTC/shunt) g_w with the objective scale so the
# discrete-actuator switching cost stays balanced against the (rescaled)
# objective benefit.  Without this, raising the objective scale makes the
# OLTCs relatively cheap and they chatter (n_sw blew up to ~190 when g_v was
# 13x the empirical scale; see docs/daily_log/2026-06-23_gw_precondition.md).
SCALE_INTEGER_GW: bool = True


def apply_priorities(cfg, pi: Dict[str, float]) -> None:
    """Set the config's objective weights from priorities ``pi`` and the
    fixed tolerances ``SIGMA`` via ``g_i = pi_i / sigma_i**2``.

    When ``SCALE_INTEGER_GW`` (Stage-E fix), also scales the integer ``g_w``
    by the factor its dominant objective term moved relative to the config
    baseline: TSO OLTC/shunt by ``g_v`` (they regulate EHV voltage), DSO
    OLTC by ``dso_g_v``.  The continuous ``g_w`` is handled separately by the
    curvature preconditioner; this keeps the *discrete* actuators in the same
    cost balance they had at the baseline objective scale.
    """
    # Reference objective scales at which the config's integer g_w were
    # calibrated (captured before we overwrite the objective weights).
    g_v_ref = cfg.g_v
    dso_g_v_ref = cfg.dso_g_v

    cfg.g_v           = pi["v_ts"]  / SIGMA["v_ts"] ** 2
    cfg.tso_g_q_pcc   = pi["q_pcc"] / SIGMA["q_pcc"] ** 2
    cfg.tso_g_q_tie   = pi["q_tie"] / SIGMA["q_tie"] ** 2
    cfg.tso_g_res_sg  = pi["res"]   / SIGMA["res"] ** 2
    cfg.tso_g_res_der = pi["res"]   / SIGMA["res"] ** 2
    cfg.g_q           = pi["q_dso"] / SIGMA["q_pcc"] ** 2   # DSO interface-Q
    cfg.dso_g_v       = pi["v_ds"]  / SIGMA["v_ds"] ** 2

    if SCALE_INTEGER_GW:
        tso_scale = cfg.g_v / g_v_ref if g_v_ref > 0 else 1.0
        dso_scale = cfg.dso_g_v / dso_g_v_ref if dso_g_v_ref > 0 else 1.0
        cfg.g_w_tso_oltc  = cfg.g_w_tso_oltc * tso_scale
        cfg.g_w_tso_shunt = cfg.g_w_tso_shunt * tso_scale
        cfg.g_w_dso_oltc  = cfg.g_w_dso_oltc * dso_scale


def _build_cfg(mod, horizon_s, scenario, target):
    cfg = mod.make_base_config()
    if scenario:
        scen = getattr(mod, "SCENARIOS", {})
        if scenario not in scen:
            raise SystemExit(f"scenario '{scenario}' not in SCENARIOS "
                             f"({sorted(scen)})")
        for k, v in scen[scenario].items():
            setattr(cfg, k, v)
    cfg.n_total_s = horizon_s
    cfg.verbose = 0
    # Headless: make_base_config() enables live plots; force them off so the
    # sweep runs without GUI windows (much faster, no blocking).
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.live_plot_tracking = False
    cfg.run_stability_analysis = False
    # Tier-2 ON so stability is held while objective ratios vary.
    cfg.precondition_g_w = True
    cfg.precondition_lambda_target = target
    return cfg


def _pareto_mask(rms_v: np.ndarray, m_bar: np.ndarray) -> np.ndarray:
    """Pareto front for (minimise rms_v, maximise m_bar)."""
    n = len(rms_v)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(rms_v[i]) or not np.isfinite(m_bar[i]):
            keep[i] = False
            continue
        for j in range(n):
            if j == i or not np.isfinite(rms_v[j]) or not np.isfinite(m_bar[j]):
                continue
            # j dominates i: no worse on both, strictly better on one
            if (rms_v[j] <= rms_v[i] and m_bar[j] >= m_bar[i]
                    and (rms_v[j] < rms_v[i] or m_bar[j] > m_bar[i])):
                keep[i] = False
                break
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", default="experiments.002_M_TSO_M_DSO_COMPARE")
    ap.add_argument("--pi-qpcc", default="0,10,20",
                    help="comma list of TSO interface-Q priorities to sweep")
    ap.add_argument("--pi-res", default="0,1,4",
                    help="comma list of TSO reserve priorities to sweep")
    ap.add_argument("--horizon-min", type=float, default=10.0)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--target", type=float, default=0.3,
                    help="preconditioner lambda_max target (held fixed)")
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    qpcc_list = [float(x) for x in args.pi_qpcc.split(",") if x.strip()]
    res_list = [float(x) for x in args.pi_res.split(",") if x.strip()]
    horizon_s = args.horizon_min * 60.0

    logs: Dict[str, List[MultiTSOIterationRecord]] = {}
    combos: List[tuple] = []
    for q in qpcc_list:
        for r in res_list:
            name = f"qpcc{q:g}_res{r:g}"
            combos.append((name, q, r))
            cfg = _build_cfg(mod, horizon_s, args.scenario, args.target)
            pi = dict(PI_BASE, q_pcc=q, res=r)
            apply_priorities(cfg, pi)
            try:
                logs[name] = run_multi_tso_dso(cfg)
            except Exception as exc:
                print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
                logs[name] = []

    v_set = float(mod.make_base_config().v_setpoint_pu)
    df = cigre_summary_table(logs, v_set=v_set)

    names = [c[0] for c in combos]
    rms_v = np.array([float(df.loc[n, "rms_v_ts_pu"]) if n in df.index else np.nan
                      for n in names])
    # m_bar_mvar = fleet-mean SG Q headroom [Mvar]; needs no MVA ratings
    # (unlike m_bar_pu), so it is finite here and usable as the reserve axis.
    m_bar = np.array([float(df.loc[n, "m_bar_mvar"]) if n in df.index else np.nan
                      for n in names])
    pareto = _pareto_mask(rms_v, m_bar)

    print()
    print(f"  module={args.module}  scenario={args.scenario}  "
          f"horizon={args.horizon_min:g} min  lambda_target={args.target:g}")
    print(f"  priorities swept: pi_qpcc in {qpcc_list}, pi_res in {res_list} "
          f"(pi_v_ts=100 ref)")
    print(f"  {'combo':<16}{'rms_v_ts':>10}{'m_bar_Mvar':>11}{'res_util':>9}"
          f"{'rms_e_sts':>10}{'rms_q_tie':>10}{'n_sw':>6}{'pareto':>8}")
    print("  " + "-" * 79)
    for i, (name, q, r) in enumerate(combos):
        def g(col):
            return float(df.loc[name, col]) if name in df.index else np.nan
        ok = name in df.index and bool(df.loc[name, "converged"])
        star = "*" if pareto[i] else ""
        print(f"  {name:<16}{rms_v[i]:>10.5f}{m_bar[i]:>11.2f}"
              f"{g('res_util'):>9.3f}{g('rms_e_sts_mvar'):>10.3f}"
              f"{g('rms_q_tie_mvar'):>10.2f}{int(g('n_sw')):>6}"
              f"{(star if ok else 'FAIL'):>8}")
    print()
    print("  '*' = Pareto-optimal (rms_v_ts down, m_bar_mvar up). Pick the knee.")
    print("  Stability is held by the preconditioner, so differences here are")
    print("  pure objective-ratio trade-offs.")


if __name__ == "__main__":
    main()
