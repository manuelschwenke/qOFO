#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/001_S_TSO_S_DSO.py
==============================
Cascaded TSO-DSO OFO Controller — main simulation entry point.

TSO every 3 min, DSO every 1 min on a single combined network (plant,
sensitivities, state).  TSO sends Q setpoints to the DSO at the PCCs.

TSO actuators: gen AVR voltages, machine trafo OLTCs (2W), TS-DER Q,
380 kV shunts.
DSO actuators: DN-DER Q, 3W coupler OLTCs, tertiary winding shunts.

The simulation loop itself lives in :mod:`experiments.runners.cascade`
so it can be reused by other scripts; this module only defines the
experiment-specific :func:`main` configuration and CLI entry point.

Author: Manuel Schwenke
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module=r"mosek")

from experiments.helpers import print_summary
from experiments.runners import run_cascade


# =============================================================================
#  Entry point
# =============================================================================


def main():
    import time as _time

    from configs.cascade_config import CascadeConfig
    from core.results_storage import save_results

    start_min = 600
    duration = 120
    step = 1
    start_sp = 1.05
    end_sp = 1.07

    num_steps = duration // step
    delta = (end_sp - start_sp) / num_steps

# ── Configuration (single place for ALL parameters) ───────────────────
    config = CascadeConfig(
        # Simulation
        v_setpoint_pu=1.05,
        n_minutes=6 * 60,
        tso_period_min=3,
        dso_period_s=30,
        start_time=datetime(2016, 1, 7, 6, 0),
        use_profiles=True,
        verbose=2,
        live_plot=True,
        # Objective weights
        g_v=250000,
        g_q=1,
        dso_g_v=100000,
        # OFO
        alpha=1.0,
        g_z=1e12,
        # TSO g_w
        gw_tso_q_der=0.2,
        gw_tso_q_pcc=0.1,
        gw_tso_v_gen=2e6,
        gw_tso_oltc=40.0,
        gw_tso_shunt=4000.0,
        gw_oltc_cross_tso=0,
        # DSO g_w
        gw_dso_q_der=10,
        gw_dso_oltc=140.0,
        gw_dso_shunt=5000.0,
        gw_oltc_cross_dso=0,
        # DSO integral Q-tracking
        g_qi=0.1,
        lambda_qi=0.9,
        q_integral_max_mvar=50.0,
        # Generator capability
        gen_xd_pu=1.2,
        gen_i_f_max_pu=2.65,
        gen_beta=0.15,
        gen_q0_pu=0.4,
        # Reserve Observer
        enable_reserve_observer=False,
        reserve_q_threshold_mvar=45.0,
        reserve_q_release_mvar=-45.0,
        reserve_cooldown_min=15,
        # Contingencies
        # contingencies = [
        #     ContingencyEvent(minute=90, element_type="line", element_index=3),
        #     ContingencyEvent(minute=60, element_type="gen", element_index=0),
        #     ContingencyEvent(
        #         minute=270, element_type="line", element_index=3, action="restore"
        #     ),
        #     ContingencyEvent(
        #         minute=300, element_type="gen", element_index=0, action="restore"
        #     ),
        #     ContingencyEvent(
        #         minute=360,
        #         element_type="ext_grid",
        #         element_index=0,
        #         action="setpoint_change",
        #         new_setpoint=1.05,
        #     ),
        #     # ramp of setpoints from 1.04 to 1.06
        #     # *[
        #     #     ContingencyEvent(
        #     #         minute=start_min + i * step,
        #     #         element_type="ext_grid",
        #     #         element_index=0,
        #     #         action="setpoint_change",
        #     #         new_setpoint=start_sp + (i + 1) * delta,
        #     #     )
        #     #     for i in range(num_steps)
        #     # ],
        #     ContingencyEvent(minute=420, element_type="line", element_index=11),
        #     ContingencyEvent(
        #         minute=480, element_type="line", element_index=11, action="restore"
        #     ),
        # ],
        contingencies=[],
    )

    # ── Run ────────────────────────────────────────────────────────────────
    t0 = _time.perf_counter()
    result = run_cascade(config)
    wall_time = _time.perf_counter() - t0

    print_summary(config.v_setpoint_pu, result.log)

    # ── Save results ──────────────────────────────────────────────────────
    run_dir = save_results(result, config, wall_time_s=wall_time)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
