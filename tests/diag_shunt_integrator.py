"""
End-to-end smoke / validation for the switched-shunt integrator dispatch
(``shunt_dispatch='integrator'``) through the full multi-TSO/multi-DSO runner.

Purpose
-------
Prove that the integrator path is wired correctly into ``run_multi_tso_dso``:
the MSC/MSR banks are built, the per-TSO-instant integrator update runs on the
cached sensitivities, and commits (if any) toggle the plant + step the DSO
interface feedforward + refresh the cached Jacobians via SMW — all WITHOUT a
power flow on the switch.

This is a *smoke* run on a short horizon (it asserts the run completes and the
banks are built); the deterministic commit / sign / dwell / feasibility / no-PF
behaviour is pinned by the unit + integration tests
(``tests/test_shunt_integrator*.py``).  See the status note
``docs/daily_log/2026-06-22_shunt_integrator.md`` for the tuning caveats (the
boundary ∂V/∂Q_eq is small at the EHV buses, so the gain must be sized to it).

Run:
    python experiments/diag_shunt_integrator.py

Author: Manuel Schwenke
Date: 2026-06-22
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_mod = importlib.import_module("experiments.002_M_TSO_M_DSO_COMPARE")


def make_integrator_config():
    cfg = _mod.make_base_config()
    # Short, headless, full-net sensitivities (integrator mode requirement).
    cfg.n_total_s = 60 * 12          # 12 min → a few TSO instants, pre-contingency
    cfg.verbose = 1
    cfg.live_plot_controller = False
    cfg.live_plot_cascade = False
    cfg.live_plot_system = False
    cfg.run_stability_analysis = False
    # Integrator mode now supports local sensitivities (the per-zone reduced net
    # keeps the PCC 3W couplers + tertiary buses + their shunts).
    cfg.local_sensitivities_tso = True
    cfg.local_sensitivities_dso = True
    cfg.result_dir = os.path.join("results", "_diag_shunt_integrator")
    os.makedirs(cfg.result_dir, exist_ok=True)

    # Switched-shunt integrator dispatch.
    cfg.shunt_dispatch = "integrator"
    cfg.install_tso_tertiary_shunts = True
    cfg.tso_shunt_kind = "msc_msr"
    cfg.tso_shunt_msc_n_levels = 4
    cfg.tso_shunt_msr_n_levels = 4
    cfg.tso_shunt_msc_q_step_mvar = 50.0
    cfg.tso_shunt_msr_q_step_mvar = 50.0
    # Gain weight sized down: step = g_H/(2*g_w), and the boundary gradient g_H
    # is small (small ∂V/∂Q_eq in physical units), so g_w must be small for the
    # bulk device to commit (see status note).  Enable Q_PCC tracking so the
    # interface term also contributes to the gradient.
    cfg.shunt_int_g_w = 1.0e-3
    cfg.shunt_int_delta_mvar = 5.0
    cfg.shunt_int_t_dwell_s = 120.0
    cfg.shunt_int_daily_budget = 8
    cfg.shunt_int_v_min_pu = 0.90
    cfg.shunt_int_v_max_pu = 1.12
    cfg.tso_g_q_pcc = 1.0
    cfg.tso_pcc_capability_on_output = True
    return cfg


def main() -> int:
    cfg = make_integrator_config()
    print("[diag] running integrator-mode smoke "
          f"(n_total_s={cfg.n_total_s}, tso_period_s={cfg.tso_period_s}) ...")
    log = run_multi_tso_dso(cfg)
    assert len(log) > 0, "runner returned an empty log"
    print(f"[diag] OK — runner completed {len(log)} steps in integrator mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
