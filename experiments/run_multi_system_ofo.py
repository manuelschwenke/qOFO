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
so it can be shared with 002, 003, and the tuning pipeline.  This script only
defines the experiment-specific configuration.

There is exactly one config factory, :func:`make_config`, and it takes no
command-line switches.  Everything it does NOT set is a
:class:`configs.config.MultiTSOConfig` default; alternative weight sets are
JSON overlays in ``configs/paramsets/``, applied with
:func:`configs.paramsets.apply_paramset`.  :func:`main_comparison` is a
separate coordinated-vs-uncoordinated study with its own paired config; call
it directly.

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

import dataclasses
import os
import pickle
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

from configs.config import MultiTSOConfig
from experiments.helpers import ContingencyEvent
from experiments.results_io import new_run_dir
from experiments.runners import run_multi_tso_dso
from network.ieee39 import build_ieee39_net
from network.zone_partition import fixed_zone_partition_ieee39
from configs.paramsets import apply_paramset

# =============================================================================
#  Comparison: coordinated vs. uncoordinated Q_PCC
# =============================================================================

def main_comparison() -> None:
    """Run coordinated vs. uncoordinated Q_PCC scenarios and compare.

    Called directly (there is no command-line switch for it)::

        python -c "from experiments.run_multi_system_ofo import main_comparison; main_comparison()"
    """
    import dataclasses

    # ── Shared parameters (identical for both scenarios) ─────────────────
    base_kwargs = dict(
        n_total_s=60.0 * 60 * 6,      # 720-min full simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=20.0,    # DSO every 5 seconds (more inner iterations)
        dt_s=20.0,
        g_v=150000.0,  # TSO voltage tracking; drives PCC Q dispatch
        # ── DSO objective tuning ──
        dso_g_v=20000.0,  # reduced to avoid competing with Q tracking
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
            ContingencyEvent(minute=10, element_type="gen", element_index=5, action="trip"),
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

    run_dir = new_run_dir(
        "run_multi_system_ofo_comparison",
        {"coordinated": cfg_a, "local_dso": cfg_b},
    )
    cfg_a.result_dir = str(run_dir.root)
    cfg_b.result_dir = str(run_dir.root)

    print("=" * 72)
    print("  Scenario A: Coordinated TSO-DSO (OFO, g_q=200)")
    print("=" * 72)
    log_a = run_multi_tso_dso(cfg_a)

    print()
    print("=" * 72)
    print("  Scenario B: Local DSO (DiscreteTapControl + cos φ = 1 HV-DER)")
    print("=" * 72)
    log_b = run_multi_tso_dso(cfg_b)
    with (run_dir.root / "records.pkl").open("wb") as handle:
        pickle.dump(
            {"coordinated": log_a, "local_dso": log_b},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

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

def make_config() -> MultiTSOConfig:
    """Run configuration for the multi-TSO / multi-DSO run -- **edit here**.

    This is the only config factory in the project.  It carries the horizon,
    the OFO timing, the objective weights, the shunt dispatch, the boundary
    equivalent and the profile / contingency schedule; **everything else lives
    in the** :class:`configs.config.MultiTSOConfig` **defaults**, which were
    overwritten on 2026-08-21 with the values this file used to restate.

    Two consequences worth knowing before editing:

    * A setting that is NOT below is not "unset" -- it is set in
      ``configs/config.py``.  Look there before adding a line here.
    * Alternative weight sets are **data**, not code: JSON overlays in
      ``configs/paramsets/``, applied with
      :func:`configs.paramsets.apply_paramset`.  ``make_config_tuned`` and
      ``make_config_per_area`` are now ``tuned`` and ``per_area`` there.
      ``make_config_dso_oltc_active`` is gone entirely: its Stage-1 candidate
      ``ac8941a46134`` -- the point at which the DSO OLTCs are actually used
      to control DSO voltage -- **is** the weight set below, to the rounding
      of its full-precision literals (g_w_der 14.430127, g_w_pcc 54.778782,
      g_w_dso_der 1097.158636, g_w_tso_oltc 3783.055025,
      g_w_dso_oltc 183.112129, dso_g_v 1e5).

    The per-DSO voltage relief is likewise a config field now
    (:attr:`MultiTSOConfig.dso_v_relief_factors`, default
    ``{"DSO_2": 20.0, "DSO_4": 20.0}``), re-derived from ``dso_g_v`` /
    ``g_w_dso_oltc`` on every ``dataclasses.replace``.  Editing either base
    weight below therefore moves the relieved pair with it, and the old
    call-it-twice-and-square-the-factor trap is gone.
    """
    return MultiTSOConfig(
        # -- horizon and cadence ---------------------------------------------
        n_total_s=60.0 * 60 * 1,
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=15.0,            # DSO-OFO every plant step
        dt_s=15.0,
        # -- objective weights -----------------------------------------------
        g_v=1E7,                      # TSO voltage tracking; drives PCC Q
        g_q=250,                      # DSO interface-Q tracking
        dso_g_v=1.0e5,                # DSO voltage; keep below the Q weight
        dso_gamma_oltc_q=0.0,
        zone_v_setpoints_pu={1: 1.03, 2: 1.03, 3: 1.03},
        # -- TSO actuator step weights ---------------------------------------
        g_w_gen=1e9,
        g_w_der=14.4,
        g_w_pcc=54.8,
        g_w_tso_oltc=3783,
        # -- TSO tertiary shunts (MSC/MSR) -----------------------------------
        install_tso_tertiary_shunts=True,
        shunt_dispatch="integrator",
        # step = g_H/(2*g_w); SMALLER = bigger step -- TUNE THIS.
        shunt_int_g_w=100,
        # -- DSO actuator step weights ---------------------------------------
        g_w_dso_der=1097.2,
        g_w_dso_oltc=183,
        # -- Diagnostics -----------------------------------------------------
        live_plot_controller=True,
        live_plot_cascade=True,
        live_plot_system=False,
        live_plot_tracking=False,
        live_plot_sbx=False,
        local_sensitivities_tso=True,
        local_sensitivities_dso=True,
        # -- Boundary equivalent for neighbouring TS areas -------------------
        tie_boundary_equivalent="thevenin",
        # -- Profile & contingency schedule ----------------------------------
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        contingencies=[
            ContingencyEvent(minute=30,  element_type="gen",  element_index=2, action="trip"),
            ContingencyEvent(minute=180, element_type="gen",  element_index=2, action="restore"),
            ContingencyEvent(minute=120,  element_type="load", bus=11, p_mw=0,   q_mvar=250, action="connect"),
            ContingencyEvent(minute=330, element_type="load", bus=11, p_mw=0,   q_mvar=250, action="trip"),
            # ContingencyEvent(minute=150, element_type="load", bus=11, p_mw=150, q_mvar=100, action="connect"),
            # ContingencyEvent(minute=360, element_type="load", bus=11, p_mw=150, q_mvar=100, action="trip"),
            ContingencyEvent(minute=210, element_type="line", element_index=25, action="trip"),
            ContingencyEvent(minute=300, element_type="line", element_index=25, action="restore"),
        ],
    )


def main() -> None:
    """Run the multi-TSO / multi-DSO simulation and record the trace.

    Invoke from the project root::

        python experiments/run_multi_system_ofo.py

    No command-line switches: :func:`make_config` is the configuration, and an
    alternative weight set is applied in code with
    :func:`configs.paramsets.apply_paramset`::

        from configs.paramsets import apply_paramset
        cfg = apply_paramset(make_config(), "per_area")

    :func:`main_comparison` is the coordinated-vs-uncoordinated study and
    builds its own paired config; call it directly.
    """
    cfg = apply_paramset(make_config(), "ch9_frozen")
    run_dir = new_run_dir("run_multi_system_ofo", cfg)
    log = run_multi_tso_dso(cfg)
    with (run_dir.root / "records.pkl").open("wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSimulation complete. {len(log)} steps recorded.")
    print(f"Results: {run_dir.root}")


def make_config_ch9_frozen() -> MultiTSOConfig:
    """The Chapter-9 frozen configuration, for analyses that take a factory.

    ``tuning_mc.stage_0_preconditioning --from-runner make_config_ch9_frozen``
    then analyses exactly what the Chapter-9 batch runs, rather than this
    module's own weights.  That distinction matters: the design depends on H,
    and H depends on the network, the boundary equivalent, the zone partition
    and the start time the config declares -- and the batch is frozen on the
    **doubled-DSO_3** network, which ``make_config()`` does not set.

    Returns the selected weight design, its per-area voltage relief, the DSO-4
    damping correction and the frozen network, all asserted by
    ``build_selected_config`` before it returns.  The import is local because
    that module reaches into ``tuning_mc``, which imports this one.
    """
    import sys
    from pathlib import Path as _Path
    _sel = _Path(__file__).resolve().parent / "ch_9_parameter_selection"
    if str(_sel) not in sys.path:
        sys.path.insert(0, str(_sel))
    from _ch9_selected_design import build_selected_config

    cfg, _prov = build_selected_config()
    return cfg


if __name__ == "__main__":
    main()
