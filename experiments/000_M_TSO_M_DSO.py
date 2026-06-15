#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/000_M_TSO_M_DSO.py
==============================
Multi-TSO / Multi-DSO OFO simulation entry point on the IEEE 39-bus network.

This script is the multi-zone analogue of ``experiments/001_S_TSO_S_DSO.py``.
It uses the same OFO controller infrastructure (TSOController, DSOController)
but orchestrates N=3 independent TSO zones via the MultiTSOCoordinator.  Each
zone has its own TSO controller and underlying DSO controllers, one per HV
sub-network (5 total: DSO_1..DSO_5 from add_hv_networks).

The simulation loop itself lives in :mod:`experiments.runners.multi_tso_dso`
so it can be shared with 002, 003, and the tuning pipeline.  This script
only defines the experiment-specific :func:`main` and :func:`main_comparison`
configurations and routes between them via ``--compare``.

Architecture (matches the multi-TSO theory in Schwenke / CIGRE 2026)
---------------------------------------------------------------------

    ┌──────────────────────────────────────────────────────────┐
    │              IEEE 39-bus network (plant)                 │
    │  Zone 1        │  Zone 2 (w/ DSOs) │  Zone 3             │
    │  TSOCtrl_1     │  TSOCtrl_2        │  TSOCtrl_3          │
    │  (4 gen incl.  │  ├── DSOCtrl_2_0  │  (4 gen)            │
    │   slack)       │  └── DSOCtrl_2_1  │                     │
    └──────────────────────────────────────────────────────────┘

Model of Q_PCC_set:

    TN backbone ─── primary bus ─── 3W trafo ─── MV bus ─── (dropped HV sub-network)
                       ▲                            ▲
                       │                            │
        Controllable Q_PCC,set                 Ward load
        actuator (DSO dispatch              = (-p_mv, -q_mv) cached
        commanded by TSO)                     (represents the dropped sub-net's
        ↑ what you asked about                 static draw at the MV side)

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Show every column
pd.set_option('display.max_columns', None)
# Show every row
pd.set_option('display.max_rows', None)
# Ensure the width is wide enough to prevent wrapping
pd.set_option('display.width', None)
# Show full content within a cell (don't truncate long strings)
pd.set_option('display.max_colwidth', None)

# ── Ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers import ContingencyEvent
from experiments.runners import run_multi_tso_dso
from network.ieee39 import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39


# =============================================================================
#  Comparison: coordinated vs. uncoordinated Q_PCC
# =============================================================================

def main_comparison() -> None:
    """Run coordinated vs. uncoordinated Q_PCC scenarios and compare.

    Invoke from the project root::

        python experiments/000_M_TSO_M_DSO.py --compare
    """
    import dataclasses

    # ── Shared parameters (identical for both scenarios) ─────────────────
    base_kwargs = dict(
        n_total_s=60.0 * 60 * 6,      # 720-min full simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=10.0,    # DSO every 5 seconds (more inner iterations)
        g_v=150000.0,  # TSO voltage tracking; drives PCC Q dispatch
        # ── DSO objective tuning ──
        dso_g_v=20000.0,  # reduced to avoid competing with Q tracking
        dso_g_qi=0,  # integral Q-tracking (0 = off)
        dso_lambda_qi=0.9,  # leaky integrator decay
        dso_q_integral_max_mvar=50.0,  # anti-windup clamp
        dso_gamma_oltc_q=0.0,  # OLTC Q-tracking attenuation: DER-primary, OLTC-backup
        # ── TSO weights (alpha=1, spectral rho(C)/2) ──
        g_w_der=10,   # single-DER zones; rho~C_jj=396 -> min 198
        g_w_gen=4e7,   # excluded from stability
        # ── DSO weights (alpha=1, rho(C_DER)=790 -> min 395) ──
        g_w_dso_der=1000,  # 8 correlated DER; sf~2.5 for smooth tracking
        g_w_dso_oltc=30,   # rho(C_OLTC)~1.1; higher for switching suppression
        use_fixed_zones=True,      # literature 3-area partition (not spectral)
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,  # refresh H_ij every N TSO steps
        verbose=1,
        live_plot_system=False,
        # ── Profile & contingency settings ───────────────────────────────
        start_time=datetime(2016, 4, 15, 12, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            # Example: trip line 0 at t=30 min, restore at t=60 min
            # ContingencyEvent(minute=100, element_type="line", element_index=8, action="trip"),
            # ContingencyEvent(minute=150, element_type="line", element_index=8, action="restore"),
            ContingencyEvent(minute=90, element_type="gen", element_index=5, action="trip"),
            ContingencyEvent(minute=180, element_type="gen", element_index=5, action="restore"),
            ContingencyEvent(minute=120, element_type="load", bus=5, p_mw=400, q_mvar=200, action="connect"),
            ContingencyEvent(minute=300, element_type="load", bus=5, p_mw=400, q_mvar=200, action="trip"),
            ContingencyEvent(minute=330, element_type="gen", element_index=4, action="trip"),
            ContingencyEvent(minute=420, element_type="gen", element_index=4, action="restore"),
            ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
            ContingencyEvent(minute=720, element_type="load", bus=7, p_mw=300, q_mvar=100, action="connect"),
            ContingencyEvent(minute=900, element_type="load", bus=7, p_mw=300, q_mvar=100, action="trip"),
        ],
    )

    # ── Scenario A: coordinated TSO-DSO ─────────────────────────────────
    cfg_a = MultiTSOConfig(
        **base_kwargs,
        g_q=200,
        g_w_pcc=100,
        g_w_tso_oltc=100,
        live_plot_controller=True,
        live_plot_cascade=True,
    )

    # ── Scenario B: local DSO control (DiscreteTapControl + cos phi=1) ──
    cfg_b = dataclasses.replace(
        cfg_a,
        dso_mode="local",       # local controllers instead of OFO
        g_w_pcc=1e5,           # TSO cannot dispatch Q_PCC (no coordination)
        g_q=0,
        g_w_tso_oltc=250,
        local_der_mode="cos_phi_1",  # unity power factor for HV-connected DER
        warmup_s=900.0,         # 15 min: let TSO OFO settle before baseline activates
        live_plot_controller=True,
        live_plot_cascade=True,
    )

    print("=" * 72)
    print("  Scenario A: Coordinated TSO-DSO (OFO, g_q=200)")
    print("=" * 72)
    log_a = run_multi_tso_dso(cfg_a)

    print()
    print("=" * 72)
    print("  Scenario B: Local DSO (DiscreteTapControl + cos φ = 1 HV-DER)")
    print("=" * 72)
    log_b = run_multi_tso_dso(cfg_b)

    # ── Summary statistics ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Comparison Summary")
    print("=" * 72)
    v_sp = cfg_a.v_setpoint_pu
    for label, log in [("Coordinated", log_a), ("Local DSO", log_b)]:
        v_devs = []
        for r in log:
            for z in r.zone_v_min:
                v_devs.append(abs(r.zone_v_min[z] - v_sp))
                v_devs.append(abs(r.zone_v_max[z] - v_sp))
        v_devs = np.array(v_devs)
        violations_5pct = np.sum(v_devs > 0.05)
        print(f"  {label:15s}  mean|ΔV|={v_devs.mean():.4f} p.u.  "
              f"max|ΔV|={v_devs.max():.4f} p.u.  "
              f"steps>5%={violations_5pct}")

    # ── Extract generator limits for capability curve plot ──────────────
    # Build the network once to read gen table limits and zone assignments.
    net_tmp, _ = build_ieee39_net(
        ext_grid_vm_pu=cfg_a.v_setpoint_pu,
        scenario=cfg_a.scenario,
    )
    _, bus_zone_tmp = fixed_zone_partition_ieee39(net_tmp, verbose=False)
    # gen_info: list of dicts, one per generator, sorted by (zone, gen_idx)
    gen_info: List[Dict[str, Any]] = []
    for g_idx in net_tmp.gen.index:
        g_bus = int(net_tmp.gen.at[g_idx, 'bus'])
        # Generator may be on LV terminal bus behind a machine trafo;
        # walk up through trafos to find the TN bus that has a zone.
        zone = bus_zone_tmp.get(g_bus)
        if zone is None:
            # Check if connected via a 2W trafo (machine trafo)
            for ti in net_tmp.trafo.index:
                if int(net_tmp.trafo.at[ti, 'lv_bus']) == g_bus:
                    hv_bus = int(net_tmp.trafo.at[ti, 'hv_bus'])
                    zone = bus_zone_tmp.get(hv_bus)
                    if zone is not None:
                        break
        if zone is None:
            continue  # skip generators not assigned to any zone
        # Same capability parameters as run_multi_tso_dso (nameplate read
        # directly; build_ieee39_net guarantees sn_mva and max_p_mw are set).
        sn       = float(net_tmp.gen.at[g_idx, 'sn_mva'])
        p_max_mw = float(net_tmp.gen.at[g_idx, 'max_p_mw'])
        gen_info.append(dict(
            zone=zone,
            gen_idx=int(g_idx),
            name=net_tmp.gen.at[g_idx, 'name'] or f"Gen_{g_idx}",
            s_rated_mva=sn,
            p_max_mw=p_max_mw,
            p_min_mw=0.0,
            xd_pu=1.8,
            i_f_max_pu=2.7,
            beta=0.15,
            q0_pu=0.4,
        ))
    gen_info.sort(key=lambda g: (g["zone"], g["gen_idx"]))

    # ── Plot comparison ─────────────────────────────────────────────────
    from visualisation.plot_multi_tso import plot_coordination_comparison
    plot_coordination_comparison(
        log_a, log_b,
        label_a="Coordinated",
        label_b="Local DSO",
        v_setpoint_pu=cfg_a.v_setpoint_pu,
        contingencies=cfg_a.contingencies,
        gen_info=gen_info,
    )


# =============================================================================
#  Entry point
# =============================================================================

def main() -> None:
    """
    Run the multi-TSO-DSO simulation with default settings and print results.

    Invoke from the project root:
        python experiments/000_M_TSO_M_DSO.py
    """

    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 36,      # 300-min simulation
        tso_period_s=60.0 * 3,        # TS-OFO every 6 min
        dso_period_s=10.0,            # STS-OFO each step (dt_s=60 >= 10)
        g_v=3E5,                      # TSO voltage tracking; drives PCC Q dispatch
        g_q=300,                      # DSO Q-tracking
        tso_g_q_tie=0,
        tso_g_res_sg=0,
        # ── DSO objective tuning ──
        dso_g_v=15000.0,              # reduced to avoid competing with Q tracking
        dso_g_qi=0,                   # integral Q-tracking (0 = off)
        dso_lambda_qi=0.95,           # leaky integrator decay
        dso_q_integral_max_mvar=200.0,
        dso_gamma_oltc_q=0.0,         # DER-primary, OLTC-backup
        # ── TSO weights (w-shift closed-loop curvature) ──
        g_w_der=20,
        g_w_gen=5e7,
        g_w_pcc=150,
        g_w_tso_oltc=100,
        install_tso_tertiary_shunts=False,
        g_w_tso_shunt=10000,
        # ── DSO weights ──
        g_w_dso_der=1000,
        g_w_dso_oltc=30,
        # ── Local-mode OLTC tap-rate limits (V1/V2 MT+NC, V3 NC) ──
        # max_step=1 (default) + wall-clock cooldown per OLTC type:
        #   MT (machine 2W gen-trafo) -> 1 tap / 180 s = once per TS interval.
        #   NC (coupler 3W interface) -> 1 tap / 60 s  = once per minute.
        # Cooldowns are wall-clock, hence robust to dt_s / dso_period_s.
        local_oltc_max_step_per_dt=1,
        oltc_cooldown_s_mt=180.0,
        oltc_cooldown_s_nc=60.0,
        use_fixed_zones=True,         # literature 3-area partition
        run_stability_analysis=False,
        sensitivity_update_interval=1E6,
        verbose=1,
        # Live plots OFF for the batch sweep (see module docstring).
        live_plot_controller=True,
        live_plot_cascade=True,
        live_plot_system=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # ── Profile & contingency settings ──
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            # NOTE: the original schedule also tripped lines 18 (min 45-150) and
            # 5 (min 60-150).  With gen 2 (G4_bus32, ~680 MW in zone 3) tripped
            # at min 120 *while both lines are out*, zone 3 loses its local
            # source and its remaining ties simultaneously -> the power flow is
            # genuinely infeasible (pp.diagnostic: converges only at 0.1 % load).
            # Per the user's instruction the line trips are removed; the gen and
            # load events are retained.  gen-2 trip alone at min 120 converges.
            # Generator trip + restore.
            ContingencyEvent(minute=60, element_type="gen",  element_index=2,  action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=2,  action="restore"),
            # Additional load connected then disconnected at bus 11.
            # Reduced from the originally-requested 400 MW / 200 Mvar: with gen 2
            # still tripped (out 120-270) the 200 Mvar step caused the cos phi=1
            # baseline (V1) to collapse at min 180.  200 MW / 100 Mvar (the
            # proven-stable magnitude from the 002 comparison) keeps the event
            # while all four variants converge.
            ContingencyEvent(minute=90, element_type="load", bus=15, p_mw=0, q_mvar=300, action="connect"),
            ContingencyEvent(minute=360, element_type="load", bus=15, p_mw=0, q_mvar=300, action="trip"),
            ContingencyEvent(minute=150, element_type="load", bus=11, p_mw=150, q_mvar=100, action="connect"),
            ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=150, q_mvar=100, action="trip"),
            ContingencyEvent(minute=260, element_type="line", element_index=25, action="trip"),
            ContingencyEvent(minute=360, element_type="line", element_index=25, action="restore"),
        ],
    )
    log = run_multi_tso_dso(cfg)
    print(f"\nSimulation complete. {len(log)} steps recorded.")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        main_comparison()
    else:
        main()
