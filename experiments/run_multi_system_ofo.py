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
        dso_gamma_oltc_q=0.0,  # OLTC Q-tracking attenuation: DER-primary, OLTC-backup
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
    # Weights come from the selected parameter set, never from this file.
    # Before the 2026-09-10 reset this study carried its own nine-weight set
    # -- a second, independent controller living in the same module as
    # make_config, and differing from it on every weight.
    from configs.paramsets import apply_paramset
    from experiments.ch_9_parameter_selection._ch9_selected_design import (
        SELECTED_PARAMSET,
    )

    _selected = apply_paramset(make_config(), SELECTED_PARAMSET, verbose=False)
    cfg_a = dataclasses.replace(
        _selected,
        **base_kwargs,
        live_plot_controller=True,
        live_plot_cascade=True,
    )

    # ── Scenario B: local DSO control (DiscreteTapControl + cos phi=1) ──
    cfg_b = dataclasses.replace(
        cfg_a,
        dso_mode="local",       # local controllers instead of OFO
        # These three are what DEFINES the uncoordinated variant, not a
        # tuning choice: pinning Q_PCC removes the TSO ability to dispatch it
        # and g_q = 0 removes the subordinate Q-tracking objective entirely.
        g_w_pcc=1e5,           # TSO cannot dispatch Q_PCC (no coordination)
        g_q=0,
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
    print("  Scenario A: Coordinated TSO-DSO (OFO, interface-Q tracked)")
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
    """The STRUCTURAL run configuration -- horizon, cadence, network, events.

    **It carries no controller weights.**  Since the 2026-09-10 reset the nine
    weights live in exactly one place, a JSON parameter set under
    ``configs/paramsets/``, and are applied on top of this config with
    :func:`configs.paramsets.apply_paramset`.  A config returned from here is
    therefore weight-INCOMPLETE by design: running it raises from
    :meth:`MultiTSOConfig.require_weights` rather than silently using a
    default.

    That is the whole point of the reset.  This factory previously stated its
    own nine weights, ``configs/config.py`` restated them as dataclass
    defaults, two JSON parameter sets restated them again, six YAML baselines
    disagreed with all of the above, and the Chapter-9 design regenerated them
    from search coordinates and then patched the result twice -- so "the
    weights the controller ran at" had no single answer.

    The cadence below is ``T_STS = 20 s`` (Experiment A) and
    ``T_TS = 180 s``; the ratio ``N_inner = 9`` is a MEASURED property
    (Experiment B), not a setting, and is re-measured whenever the weights
    change because it is a joint property of the period ratio and ``G_w``.
    """
    return MultiTSOConfig(
        # -- horizon and cadence ---------------------------------------------
        n_total_s=60.0 * 60 * 6,
        tso_period_s=60.0 * 3,        # TS-OFO every 3 min
        dso_period_s=20.0,            # T_STS, fixed by Experiment A
        dt_s=20.0,
        dso_gamma_oltc_q=0.0,
        zone_v_setpoints_pu={1: 1.03, 2: 1.03, 3: 1.03},
        # -- TSO tertiary shunts (MSC/MSR) -----------------------------------
        install_tso_tertiary_shunts=True,
        shunt_dispatch="integrator",
        # step = g_H/(2*g_w); SMALLER = bigger step -- TUNE THIS.
        shunt_int_g_w=10,
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
            ContingencyEvent(minute=150, element_type="load", bus=5, p_mw=150, q_mvar=100, action="connect"),
            ContingencyEvent(minute=400, element_type="load", bus=5, p_mw=150, q_mvar=100, action="trip"),
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
    # 2026-10-20 (author decision): back to the chapter's controller identity,
    # now the re-run Experiment B result ch9_selected_10_2026
    # (docs/tuning/2026-10-16_protocol_experiment_B_rerun.md).  The hand-set
    # variants used here 2026-10-14..19 ("ch9_selected_shunt_gw006",
    # "ch9_selected_pcc27", "ch9_selected_pcc27_gqceil",
    # "ch9_selected_pcc27_gqceil_comp_q_off") remain as parameter sets.
    from experiments.ch_9_parameter_selection._ch9_selected_design import (
        SELECTED_PARAMSET,
    )
    cfg = apply_paramset(make_config(), SELECTED_PARAMSET)
    run_dir = new_run_dir("run_multi_system_ofo", cfg)
    log = run_multi_tso_dso(cfg)
    with (run_dir.root / "records.pkl").open("wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSimulation complete. {len(log)} steps recorded.")
    print(f"Results: {run_dir.root}")


def make_config_ch9_frozen() -> MultiTSOConfig:
    """The Chapter-9 frozen configuration, for analyses that take a factory.

    ``tuning_mc.stage_0_preconditioning --from-runner make_config_ch9_frozen``
    then analyses exactly what the Chapter-9 batch runs.  That distinction
    mattered while the two lineages differed: the design depends on H, and H
    depends on the network, the boundary equivalent, the zone partition and the
    start time the config declares.

    Since 2026-09-15 ``build_selected_config`` builds on :func:`make_config`,
    so this is ``apply_paramset(make_config(), SELECTED_PARAMSET)`` plus the
    frozen **doubled-DSO_3** multipliers, which :func:`make_config` does not
    set.  Those two things -- the Chapter-9 weights and the Chapter-9 network
    -- are the whole difference from a bare :func:`make_config`, and naming
    them in one factory is why this function still exists: the
    ``--from-runner`` interface takes a factory name.

    Note that the network is the one respect in which the Chapter-9 batch
    differs from the Stage-I/II weight campaign, which ran symmetric.  The
    import is local because that module reaches into ``tuning_mc``, which
    imports this one.
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
