#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/runners/multi_tso_dso.py
=====================================
Multi-TSO / multi-DSO OFO simulation entry point.

This module hosts :func:`run_multi_tso_dso`, the main loop that builds the
IEEE 39-bus network, instantiates the multi-zone TSO and DSO OFO
controllers, and steps the cascaded outer/inner loop in time.  It was
extracted from ``experiments/000_M_TSO_M_DSO.py`` so that experiment
scripts (000, 002, 003) can share a single canonical implementation.

Architecture (matches the multi-TSO theory in Schwenke / CIGRE 2026)
---------------------------------------------------------------------

    ┌──────────────────────────────────────────────────────────┐
    │              IEEE 39-bus network (plant)                 │
    │  Zone 1        │  Zone 2 (w/ DSOs) │  Zone 3             │
    │  TSOCtrl_1     │  TSOCtrl_2        │  TSOCtrl_3          │
    │  (4 gen incl.  │  ├── DSOCtrl_2_0  │  (4 gen)            │
    │   slack)       │  └── DSOCtrl_2_1  │                     │
    └──────────────────────────────────────────────────────────┘

Step sequence (each simulation step dt_s):
    1.  Apply time-series profiles to plant network.
    2.  Run power flow on plant network.
    3.  If TSO step: call coordinator.step(measurements_per_zone).
        * Each TSOController.step() solves its local MIQP independently.
        * Coordinator optionally recomputes H_ij and checks contraction.
    4.  If DSO step: call DSOController.step() for each HV sub-network DSO.
    5.  Apply all new setpoints to plant network.
    6.  Run power flow, record results.

Sensitivity matrices
--------------------
* H_ii (local, zone i): computed by TSOController._build_sensitivity_matrix()
  using generator terminal buses + DER sgens as column inputs and zone buses
  as row outputs.
* H_ij (cross-zone, i≠j): computed by MultiTSOCoordinator.compute_cross_sensitivities()
  using zone_j's inputs and zone_i's observed outputs.
* H_DSO (per HV sub-network): computed by DSOController._build_sensitivity_matrix()
  using DSO DER Q + 3 coupling OLTC as inputs, interface Q + HV voltages + line
  currents as outputs.

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from numpy.typing import NDArray

# ── Ensure project root is on sys.path when imported as a package module ─────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    MultiZoneStabilityResult,
)
from controller.base_controller import OFOParameters
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.multi_tso_coordinator import MultiTSOCoordinator, ZoneDefinition
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from core.measurement import Measurement, measure_zone_tso, measure_zone_dso
from core.network_state import NetworkState
from core.reporting import (
    load_and_apply_tuned_params,
    write_stability_analysis_markdown,
    write_tuned_params_json,
)
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from network.ieee39.zonal_balancing import apply_gen_dispatch, compute_zonal_gen_dispatch
from network.ieee39 import (
    add_hv_networks,
    build_ieee39_net,
    HVNetworkInfo,
    IEEE39NetworkMeta,
    remove_generators,
    tag_der_q_modes,
)
from network.zone_partition import (
    fixed_zone_partition_ieee39,
    spectral_zone_partition,
    relabel_zones_by_generator_count,
    get_zone_lines,
    get_tie_lines,
    get_zone_tie_lines,
)
from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers import (
    ContingencyEvent,
    MultiTSOIterationRecord,
    _apply_contingency,
    _network_state,
    apply_dso_controls,
    apply_zone_tso_controls,
    prepare_load_contingencies,
)
from sensitivity.jacobian import JacobianSensitivities
from core.message import SetpointMessage, ShuntDisturbanceMessage
from controller.der_qv_local_loop import (
    CosPhiConstLoop,
    QVLocalLoop,
    clear_seed_lu_cache,
    install_der_q_loops,
    install_qv_local_loops,
    seed_qv_equilibrium,
)

# Helpers live in a sibling module; import at the top so callers (e.g.
# ``tuning.ceilings``) can monkey-patch them on this module's namespace.
from experiments.runners._multi_tso_helpers import (
    _OLTCRateLimiter,
    _clamp_oltc_taps,
    _collect_contingency_watch_buses,
    _dump_contingency_diagnostics,
    _record_dso_group_and_transformer_data,
    _record_hv_group_observables,
    _record_local_dso_trafo_data,
    _record_zone_live_plot_observables,
    _run_delayed_stability_analysis,
    _snapshot_oltc_taps,
)


# =============================================================================
#  Main simulation function
# =============================================================================

def run_multi_tso_dso(
    config: MultiTSOConfig,
    pre_loop_hook: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> List[MultiTSOIterationRecord]:
    """
    Execute the multi-TSO / multi-DSO OFO simulation.

    Parameters
    ----------
    config : MultiTSOConfig
        All simulation parameters.  See dataclass docstring for details.
    pre_loop_hook : optional
        If provided, called once with the post-Phase-2 state dict
        (``net``, ``meta``, ``tso_controllers``, ``dso_controllers``,
        ``shared_jac``, ``dso_to_tso_id``, ``zone_defs``,
        ``coordinator``) right before the main time loop. If the hook
        returns a truthy value, the runner returns immediately with an
        empty log -- used by diagnostic scripts that only need the
        post-init state.

    Returns
    -------
    log : List[MultiTSOIterationRecord]
        One record per simulation step.
    """
    v_set = config.v_setpoint_pu
    verbose = config.verbose

    if verbose >= 1:
        print("=" * 72)
        zone_method = "fixed 3-area" if config.use_fixed_zones else "spectral"
        print("  MULTI-TSO / MULTI-DSO OFO -- IEEE 39-bus New England")
        print(f"  V_set = {v_set:.3f} p.u.  |  N_zones = 3")
        print(f"  Zone partition: {zone_method}  |  4 HV sub-networks (DSO_1..DSO_4)")
        print("=" * 72)

    # =========================================================================
    # STEP 1: Build the base IEEE 39-bus network (no DSO feeders yet)
    # =========================================================================
    if verbose >= 1:
        print("[1] Building IEEE 39-bus network ...")

    net, meta = build_ieee39_net(
        ext_grid_vm_pu=1.03,
        scenario=config.scenario,
        verbose=(verbose >= 1),
    )

    # =========================================================================
    # STEP 2: Zone partitioning
    # =========================================================================
    if config.use_fixed_zones:
        if verbose >= 1:
            print()
            print("[2] Fixed 3-area zone partition (literature) ...")
        zone_map, bus_zone = fixed_zone_partition_ieee39(
            net, verbose=(verbose >= 2)
        )
    else:
        if verbose >= 1:
            print("[2] Spectral zone partitioning (N=3) ...")
        zone_map, bus_zone = spectral_zone_partition(
            net, n_zones=3, verbose=(verbose >= 2)
        )
        # ToDo: Delete
        # Relabel: Zone 0 = most generators, Zone 2 = fewest (prime candidate for DSO)
        # _gen_grid = list(meta.gen_grid_bus_indices or meta.gen_bus_indices)
        # zone_map, bus_zone = relabel_zones_by_generator_count(
        #     zone_map, bus_zone, _gen_grid
        # )

    # CONSOLE OUTPUT ##########################################################
    if verbose >= 1:
        _gen_grid_set = set(meta.gen_grid_bus_indices or meta.gen_bus_indices)
        for z in sorted(zone_map.keys()):
            n_gen_z = sum(1 for b in zone_map[z] if b in _gen_grid_set)
            n_load_z = sum(
                1 for li in net.load.index if int(net.load.at[li, "bus"]) in zone_map[z]
            )
            print(f"  Zone {z}: {len(zone_map[z])} buses, "
                  f"{n_gen_z} generators, {n_load_z} loads")
    ###########################################################################

    # =========================================================================
    # STEP 3: Attach 3 HV sub-networks (110 kV, TUDA topology)
    # =========================================================================
    if verbose >= 1:
        print()
        print("[3] Attaching 3 HV sub-networks (DSO_1..DSO_3) ...")

    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=config.install_tso_tertiary_shunts,
        verbose=(verbose >= 2),
    )

    # add_hv_networks() may remove buses (e.g. bus 11/0-idx = IEEE bus 12).
    # Purge any removed buses from zone_map so downstream logic stays consistent.
    existing_buses = set(net.bus.index)
    for z in zone_map:
        zone_map[z] = [b for b in zone_map[z] if b in existing_buses]

    # =========================================================================
    # STEP 4: tag every DER with its q_mode (Soleimani §III-B Q_cor path)
    # =========================================================================
    # All DERs stay as pp.sgen.  Tag every TSO and DSO sgen with its
    # q_mode and parameters from the runner-level MultiTSOConfig
    # hierarchy.  The plant-side controllers (QVLocalLoop /
    # CosPhiConstLoop) read these columns each PF iteration.
    # ToDo: Cleanup
    if verbose >= 1:
        print()
        print("[4] Tagging DER q_modes ...")
    meta = tag_der_q_modes(
        net, meta,
        tso_q_mode=config.tso_q_mode,
        dso_q_mode=config.dso_q_mode,
        q_mode_overrides=config.der_q_mode_overrides,
        tso_qv_slope_pu=config.tso_qv_slope_pu,
        dso_qv_slope_pu=config.dso_qv_slope_pu,
        qv_slope_pu_overrides=config.der_qv_slope_pu_overrides,
        tso_qv_vref_pu=config.tso_qv_vref_pu,
        dso_qv_vref_pu=config.dso_qv_vref_pu,
        qv_vref_pu_overrides=config.der_qv_vref_pu_overrides,
        tso_qv_deadband_pu=config.tso_qv_deadband_pu,
        dso_qv_deadband_pu=config.dso_qv_deadband_pu,
        qv_deadband_pu_overrides=config.der_qv_deadband_pu_overrides,
        tso_cosphi=config.tso_cosphi,
        dso_cosphi=config.dso_cosphi,
        cosphi_overrides=config.der_cosphi_overrides,
        tso_cosphi_sign=config.tso_cosphi_sign,
        dso_cosphi_sign=config.dso_cosphi_sign,
        cosphi_sign_overrides=config.der_cosphi_sign_overrides,
        verbose=(verbose >= 1),
    )

    # NB: install_der_q_loops is deferred to step [10.3], AFTER the
    # Phase 1/2 OLTC init.  If we install now, the plant-side QVLocalLoops
    # (one per TSO + DSO DER) would run during Phase 1 alongside the
    # temp PV gens used to seed STATCOM Q, and the inner-loop dynamics
    # destabilise the init.
    if verbose >= 1:
        print(
            f"[4b] Deferred plant-side loop install for "
            f"{len(meta.tso_der_indices)} TSO + "
            f"{len(meta.dso_der_indices)} DSO DERs until after "
            f"step [10] OLTC init."
        )

    # =========================================================================
    # STEP 5: Build ZoneDefinitions and TSOControllerConfigs
    # =========================================================================
    if verbose >= 1:
        print()
        print("[5] Building zone definitions and controller configs ...")

    # ── Partition generator indices per zone ──────────────────────────────────
    # Use gen_grid_bus_indices (original 345 kV bus) for zone assignment,
    # but store gen_bus_indices (terminal bus, possibly 10.5 kV) for
    # sensitivity computation.
    zone_gen_indices: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_gen_buses:   Dict[int, List[int]] = {z: [] for z in zone_map}
    gen_grid_buses = meta.gen_grid_bus_indices if meta.gen_grid_bus_indices else meta.gen_bus_indices
    for g, g_bus, g_grid_bus in zip(meta.gen_indices, meta.gen_bus_indices, gen_grid_buses):
        for z, buses in zone_map.items():
            if g_grid_bus in set(buses):
                zone_gen_indices[z].append(g)
                zone_gen_buses[z].append(g_bus)  # terminal bus for sensitivity
                break

    # ── Partition TSO DER indices per zone ────────────────────────────────────
    zone_der_indices: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_der_buses:   Dict[int, List[int]] = {z: [] for z in zone_map}
    for s_idx, s_bus in zip(meta.tso_der_indices, meta.tso_der_buses):
        for z, buses in zone_map.items():
            if s_bus in set(buses):
                zone_der_indices[z].append(s_idx)
                zone_der_buses[z].append(s_bus)
                break

    # ── Partition machine-transformer OLTCs per zone ───────────────────────────
    # Exclude the slack gen's machine trafo from the controllable OLTC set.
    # Its LV bus is the PYPOWER angle reference, so
    # :func:`sensitivity.jacobian.JacobianSensitivities.compute_dV_ds_2w`
    # cannot produce a sensitivity column for it (the reference bus is not
    # in the Jacobian).  The slack gen's ``vm_pu`` setpoint already gives
    # the TSO a direct voltage control at that terminal, so losing the
    # redundant OLTC degree of freedom is acceptable.
    slack_gen_term_buses: Set[int] = set()
    if "slack" in net.gen.columns:
        for g in net.gen.index[net.gen["slack"].astype(bool)]:
            slack_gen_term_buses.add(int(net.gen.at[g, "bus"]))

    zone_oltc_trafos: Dict[int, List[int]] = {z: [] for z in zone_map}
    for t_idx, g_idx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        lv_bus = int(net.trafo.at[t_idx, "lv_bus"])
        if lv_bus in slack_gen_term_buses:
            continue  # slack-gen OLTC excluded (see comment above)
        # Machine trafo's grid bus = hv_bus of the 2W transformer
        grid_bus = int(net.trafo.at[t_idx, "hv_bus"])
        for z, buses in zone_map.items():
            if grid_bus in set(buses):
                zone_oltc_trafos[z].append(t_idx)
                break

    # ── Build gen→trafo map for contingency handling ────────────────────────
    # Maps net.gen index → net.trafo index of the associated machine trafo.
    gen_trafo_map: Dict[int, int] = {
        g_idx: t_idx
        for t_idx, g_idx in zip(meta.machine_trafo_indices,
                                meta.machine_trafo_gen_map)
        if g_idx >= 0  # skip non-machine OLTCs (marked -1)
    }

    # ── Group HV sub-networks by zone ────────────────────────────────────────
    zone_hv_networks: Dict[int, List[HVNetworkInfo]] = {z: [] for z in zone_map}
    for hv in meta.hv_networks:
        zone_hv_networks[hv.zone].append(hv)

    # All DSO IDs (one per HV sub-network)
    dso_ids: List[str] = [hv.net_id for hv in meta.hv_networks]

    # Per-zone PCC trafos and DSO IDs (parallel lists)
    zone_pcc_trafos: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_pcc_dso_ids: Dict[int, List[str]] = {z: [] for z in zone_map}
    for hv in meta.hv_networks:
        for trafo_idx in hv.coupling_trafo_indices:
            zone_pcc_trafos[hv.zone].append(trafo_idx)
            zone_pcc_dso_ids[hv.zone].append(hv.net_id)

    # Save original TN-only zone map for TSO monitoring (before HV extension).
    # The TSO monitors TN-level voltages and line currents only; HV elements
    # are the DSO's domain.
    tn_zone_map: Dict[int, List[int]] = {
        z: [b for b in buses if b in net.bus.index]
        for z, buses in zone_map.items()
    }

    # Extend zone bus indices to include HV sub-network buses (for dispatch / ownership)
    for hv in meta.hv_networks:
        zone_map[hv.zone] = sorted(set(zone_map[hv.zone]) | set(hv.bus_indices))

    # Extend zone map with machine transformer LV buses so that
    # compute_zonal_gen_dispatch() can assign generators on LV terminal
    # buses (e.g. 10.5 kV) to the correct zone via the HV (grid) bus.
    for tidx, gidx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        if gidx < 0:
            continue
        lv_bus = int(net.trafo.at[tidx, "lv_bus"])
        hv_bus = int(net.trafo.at[tidx, "hv_bus"])
        for z, buses in zone_map.items():
            if hv_bus in set(buses):
                if lv_bus not in set(buses):
                    zone_map[z] = sorted(set(zone_map[z]) | {lv_bus})
                break

    # HV-network lookup for DSO controller init
    hv_info_map: Dict[str, HVNetworkInfo] = {hv.net_id: hv for hv in meta.hv_networks}

    # Map each TSO-owned tertiary shunt bus to its parent DSO id.  Used at
    # run-time to dispatch ``ShuntDisturbanceMessage`` to the affected DSO
    # whenever the TSO MIQP switches a shunt.  The shunt sits at the
    # tertiary of the first coupling 3-winding transformer (see add_hv_networks).
    shunt_bus_to_dso_id: Dict[int, str] = {}
    for hv in meta.hv_networks:
        if hv.coupling_lv_bus_indices:
            shunt_bus_to_dso_id[int(hv.coupling_lv_bus_indices[0])] = hv.net_id

    # ── Build ZoneDefinition for each zone ────────────────────────────────────
    # TSO monitoring uses TN-only buses and lines (tn_zone_map).
    # Line filtering: TSOController's sensitivity builder (build_sensitivity_matrix_H)
    # only computes ∂I/∂u for lines where BOTH endpoints are PQ buses.  Lines
    # touching a PV generator bus are excluded from the I-rows of H_physical.
    # To avoid a shape mismatch we pre-filter zone lines to PQ-bus endpoints only.
    pv_and_slack_buses_run = (
        set(int(net.gen.at[g, "bus"]) for g in net.gen.index) |
        set(int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index)
    )

    # ── Partition TSO-owned tertiary shunts per zone ──────────────────────────
    # Each shunt is owned by the TSO zone hosting the parent DSO sub-network
    # (see meta.tso_tertiary_shunt_zones, populated by add_hv_networks).  The
    # DSO controllers are blind to these shunts; the bus indices flow into
    # ZoneDefinition.shunt_bus_indices and from there into TSOControllerConfig.
    zone_shunt_buses:  Dict[int, List[int]]   = {z: [] for z in zone_map}
    zone_shunt_qsteps: Dict[int, List[float]] = {z: [] for z in zone_map}
    for sb, q_step, sz in zip(
        meta.tso_tertiary_shunt_buses,
        meta.tso_tertiary_shunt_q_steps_mvar,
        meta.tso_tertiary_shunt_zones,
    ):
        if sz in zone_shunt_buses:
            zone_shunt_buses[sz].append(int(sb))
            zone_shunt_qsteps[sz].append(float(q_step))

    zone_defs: Dict[int, ZoneDefinition] = {}
    for z in sorted(zone_map.keys()):
        # TSO monitors TN-level elements only
        tn_bus_set = set(tn_zone_map[z])
        all_z_lines = get_zone_lines(net, tn_bus_set)
        # Keep only lines with both endpoints at PQ buses (Jacobian builder requirement)
        z_lines = [
            li for li in all_z_lines
            if int(net.line.at[li, "from_bus"]) not in pv_and_slack_buses_run
            and int(net.line.at[li, "to_bus"]) not in pv_and_slack_buses_run
        ]
        z_line_max_i_ka = [
            float(net.line.at[li, "max_i_ka"]) for li in z_lines
        ]

        # Voltage observation buses: only TN PQ buses (not PV/slack, not HV).
        v_bus_indices_z = [
            b for b in tn_zone_map[z] if b not in pv_and_slack_buses_run
        ]

        zone_defs[z] = ZoneDefinition(
            zone_id=z,
            bus_indices=zone_map[z],
            gen_indices=zone_gen_indices[z],
            gen_bus_indices=zone_gen_buses[z],
            tso_der_indices=zone_der_indices[z],
            tso_der_buses=zone_der_buses[z],
            v_bus_indices=v_bus_indices_z,  # PQ buses only (V-observable)
            line_indices=z_lines,
            line_max_i_ka=z_line_max_i_ka,
            pcc_trafo_indices=zone_pcc_trafos[z],
            pcc_dso_ids=zone_pcc_dso_ids[z],
            oltc_trafo_indices=zone_oltc_trafos[z],
            shunt_bus_indices=zone_shunt_buses[z],
            shunt_q_steps_mvar=zone_shunt_qsteps[z],
            v_setpoint_pu=v_set,
            # alpha removed (absorbed into g_w)
            g_v=config.g_v,
            g_w_der=config.g_w_der,
            g_w_gen=config.g_w_gen,
            g_w_pcc=config.g_w_pcc,
            g_w_oltc=config.g_w_tso_oltc,
            g_w_shunt=config.g_w_tso_shunt,
            g_q_tso=config.tso_g_q_pcc,
        )

    # Populate per-zone tie-line sets (Phase A: monitoring only).
    # A tie line is one whose two endpoints sit in two different TSO zones.
    # For each zone we record the tie lines touching its bus set together
    # with the IN-ZONE endpoint bus (sign anchor for Q_tie measurement
    # and sensitivity).  Both zones touching the same line own it, each
    # at its own end — symmetric decentralised monitoring.
    _tn_zone_buses_set = {z: set(tn_zone_map[z]) for z in zone_defs}
    for z, zd in zone_defs.items():
        other_lists = [
            _tn_zone_buses_set[zj] for zj in zone_defs if zj != z
        ]
        pairs = get_zone_tie_lines(net, _tn_zone_buses_set[z], other_lists)
        zd.tie_line_indices = [li for li, _ in pairs]
        zd.tie_line_endpoint_buses = [endp for _, endp in pairs]
        zd.q_tie_setpoints_mvar = (
            np.zeros(len(pairs), dtype=np.float64) if pairs else None
        )
        zd.g_q_tie = config.tso_g_q_tie

    if verbose >= 1:
        for z, zd in zone_defs.items():
            hv_names = [hv.net_id for hv in zone_hv_networks.get(z, [])]
            print(f"  Zone {z}: {len(zd.gen_indices)} gen, {len(zd.tso_der_indices)} DER, "
                  f"{len(zd.oltc_trafo_indices)} OLTC, "
                  f"{len(zd.shunt_bus_indices)} shunt, "
                  f"{len(zd.line_indices)} lines, {len(zd.pcc_trafo_indices)} PCC trafos, "
                  f"{len(zd.tie_line_indices)} tie lines  "
                  f"DSOs: {hv_names}")

    # ── Live-plot statics (tie-line map, gen P/Q limits) ─────────────────────
    # The inter-zone tie-line map feeds the TSO-CONTROLLER tie-line Q tile.
    # Generator P/Q limits feed the SYSTEM-POWER-FLOW generator tiles.
    tie_line_map: Dict[Tuple[int, int], List[int]] = {}
    zone_ids_sorted = sorted(zone_defs.keys())
    for i, z_i in enumerate(zone_ids_sorted):
        for z_j in zone_ids_sorted[i + 1:]:
            ties = get_tie_lines(
                net, set(tn_zone_map[z_i]), set(tn_zone_map[z_j]),
            )
            if ties:
                tie_line_map[(z_i, z_j)] = list(ties)

    gen_limits_static: Dict[int, Dict[str, float]] = {} # ToDo: what are these limits? we want to use op-diagram!
    for g_idx in net.gen.index:
        limits: Dict[str, float] = {}
        for key in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"):
            limits[key] = (
                float(net.gen.at[g_idx, key])
                if key in net.gen.columns else float("nan")
            )
        gen_limits_static[int(g_idx)] = limits

    # =========================================================================
    # STEP 6: Initialise TSOControllers (one per zone)
    # =========================================================================
    if verbose >= 1:
        print()
        print("[6] Initialising TSOControllers ...")

    ns0 = _network_state(net)  # initial network state snapshot

    # Build one full-network Jacobian at the current (pre-profile) operating
    # point and share it across every TSO and DSO controller, plus the
    # coordinator.  This snapshot is replaced by a fresh post-Phase-2 one
    # below (see "Rebuild shared Jacobian"), so all controllers eventually
    # operate on the same post-init cached plant model.  Avoids 8 redundant
    # deep-copy + pp.runpp + dense-inversion calls inside the construction
    # loops.
    _t_jac_initial = perf_counter()
    shared_jac = JacobianSensitivities(net)
    if verbose >= 1:
        print(f"  [T] initial shared JacobianSensitivities: {perf_counter() - _t_jac_initial:.2f} s")

    _t_step5 = perf_counter()
    tso_controllers: Dict[int, TSOController] = {}
    for z, zd in zone_defs.items():

        # ── G_w diagonal for this zone's u vector ────────────────────────────
        gw_diag = zd.gw_diagonal()
        # Q_cor mode: no V_gf actuator → no Q_gf slacks.
        gz_diag_target = np.concatenate([
            np.full(len(zd.v_bus_indices),     config.g_z_voltage),  # V slacks
            np.full(len(zd.pcc_trafo_indices), config.g_z_q_pcc),    # Q_PCC slacks
            np.full(len(zd.line_indices),      config.g_z_current),  # current slacks
            np.full(len(zd.gen_indices),       config.g_z_q_gen),    # Q_gen slacks
            np.full(len(zd.tie_line_indices),  config.g_z_q_tie),    # Q_tie slacks
        ])
        # During warmup use a tiny g_z; after warmup switch to gz_diag_target
        if config.g_z_warmup_s > 0:
            gz_diag = np.where(gz_diag_target > 0, config.g_z_warmup_value, 0.0)
        else:
            gz_diag = gz_diag_target

        ofo_params = OFOParameters(
            g_w=gw_diag,
            g_z=gz_diag,
            g_u=np.zeros_like(gw_diag),
            alpha=1.0,  # Q_cor mode does not use command relaxation
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
            int_cooldown_s=config.oltc_cooldown_s,
            adapt_g_w_classes=config.tso_adapt_g_w_classes(),
            g_w_adapt_meta=config.make_g_w_adapt_meta(),
        )

        # Build gen→OLTC position mapping for capability-based OLTC blocking.
        # gen_trafo_map: net.gen index → net.trafo index (machine trafo).
        # We need: position in gen_indices → position in oltc_trafo_indices.
        _oltc_pos = {t: k for k, t in enumerate(zd.oltc_trafo_indices)}
        _gen_oltc_map: Dict[int, int] = {}
        for gen_pos, g_idx in enumerate(zd.gen_indices):
            t_idx = gen_trafo_map.get(g_idx)
            if t_idx is not None and t_idx in _oltc_pos:
                _gen_oltc_map[gen_pos] = _oltc_pos[t_idx]

        # TSOControllerConfig: pass zone-specific index sets
        tso_cfg = TSOControllerConfig(
            der_indices=zd.tso_der_indices,
            pcc_trafo_indices=zd.pcc_trafo_indices,
            pcc_dso_controller_ids=zd.pcc_dso_ids,
            oltc_trafo_indices=zd.oltc_trafo_indices,
            shunt_bus_indices=zd.shunt_bus_indices,
            shunt_q_steps_mvar=zd.shunt_q_steps_mvar,
            voltage_bus_indices=zd.v_bus_indices,
            current_line_indices=zd.line_indices,
            current_line_max_i_ka=zd.line_max_i_ka if zd.line_indices else None,
            v_setpoints_pu=np.full(len(zd.v_bus_indices), zd.v_setpoint_pu),
            g_v=zd.g_v,
            gen_indices=zd.gen_indices,
            gen_bus_indices=zd.gen_bus_indices,
            gen_oltc_map=_gen_oltc_map,
            enable_saturation_mode=config.enable_avr_saturation_mode,
            g_q_tso=config.tso_g_q_pcc,
            pcc_capability_on_output=config.tso_pcc_capability_on_output,
            tie_line_indices=zd.tie_line_indices,
            tie_line_endpoint_buses=zd.tie_line_endpoint_buses,
            q_tie_setpoints_mvar=zd.q_tie_setpoints_mvar,
            g_q_tie=config.tso_g_q_tie,
            g_z_q_tie=config.g_z_q_tie,
            use_q_cor_actuator=config.use_q_cor_actuator,
            qv_slope_pu=config.tso_qv_slope_pu,
        )

        # ActuatorBounds for DERs in this zone
        if zd.tso_der_indices:
            s_rated = np.array(
                [float(net.sgen.at[s, "sn_mva"]) for s in zd.tso_der_indices],
                dtype=np.float64,
            )
            p_max = np.array(
                [float(net.sgen.at[s, "p_mw"]) for s in zd.tso_der_indices],
                dtype=np.float64,
            )
        else:
            s_rated = np.array([], dtype=np.float64)
            p_max   = np.array([], dtype=np.float64)

        # Generator capability parameters for this zone.
        # Nameplate is set unconditionally in build_ieee39_net (see
        # network/ieee39/constants.NAMEPLATE_FACTOR).
        gen_params = []
        for g in zd.gen_indices:
            sn       = float(net.gen.at[g, "sn_mva"])
            p_max_mw = float(net.gen.at[g, "max_p_mw"])
            gen_params.append(
                GeneratorParameters(
                    s_rated_mva=sn,
                    p_max_mw=p_max_mw,
                    p_min_mw=0.0,
                    xd_pu=1.8,       # Milano: 1.0-1.8 for turbo-gen
                    i_f_max_pu=2.7,  # Milano eq. 12.10: 2.6-2.73
                    beta=0.15,       # Milano p. 293
                    q0_pu=0.4,       # Milano p. 293
                )
            )

        # Read per-DER operating diagram type (STATCOM vs VDE-AR-N-4120-v2)
        der_op_diagrams = []
        for s in zd.tso_der_indices:
            od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
            der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")

        bounds = ActuatorBounds(
            der_indices=np.array(zd.tso_der_indices, dtype=np.int64),
            der_s_rated_mva=s_rated,
            der_p_max_mw=p_max,
            oltc_indices=np.array(zd.oltc_trafo_indices, dtype=np.int64),
            oltc_tap_min=np.array(
                [int(net.trafo.at[t, "tap_min"]) for t in zd.oltc_trafo_indices],
                dtype=np.int64,
            ),
            oltc_tap_max=np.array(
                [int(net.trafo.at[t, "tap_max"]) for t in zd.oltc_trafo_indices],
                dtype=np.int64,
            ),
            shunt_indices=np.array(zd.shunt_bus_indices, dtype=np.int64),
            shunt_q_mvar=np.array(zd.shunt_q_steps_mvar, dtype=np.float64),
            gen_params=gen_params,
            der_op_diagrams=der_op_diagrams,
        )

        ctrl = TSOController(
            controller_id=f"tso_zone_{z}",
            params=ofo_params,
            config=tso_cfg,
            network_state=ns0,
            actuator_bounds=bounds,
            sensitivities=shared_jac,
        )
        # _u_current is initialised later (step 7e), after profiles and
        # OLTC/STATCOM init have settled the operating point.
        tso_controllers[z] = ctrl

    if verbose >= 1:
        print(f"  [T] step [5] TSO controller construction: {perf_counter() - _t_step5:.2f} s")


    # =========================================================================
    # STEP 7: Initialise DSO controllers (one per HV sub-network, all zones)
    #         (skipped when dso_mode='local'; see local-mode print branch)
    # =========================================================================
    dso_controllers: Dict[str, DSOController] = {}

    if config.dso_mode == "local":
        if verbose >= 1:
            print()
            print("[7] DSO mode = 'local' — skipping OFO DSO controllers.")
            print("    Coupler OLTCs: pandapower DiscreteTapControl (AVR)")
            n_der_total = sum(len(hv.sgen_indices) for hv in meta.hv_networks)
            if config.dso_q_mode == "qv":
                print(f"    DER Q control: Q(V) linear droop, "
                      f"V_set={config.dso_qv_vref_pu:.3f} p.u., "
                      f"slope={config.dso_qv_slope_pu:.3f}, "
                      f"deadband={config.dso_qv_deadband_pu:.3f}  "
                      f"({n_der_total} DER)")
            else:
                print(f"    DER Q control: cos phi = {config.dso_cosphi:.2f} "
                      f"({n_der_total} DER)")

    if config.dso_mode == "local":
        # No OFO DSO controllers — skip to step 7.
        pass
    else:
        if verbose >= 1:
            print()
            print("[7] Initialising DSO controllers (5 HV sub-networks) ...")

    _t_step6 = perf_counter()
    for hv in meta.hv_networks if config.dso_mode != "local" else []:
        # Allow-list filter: when ``config.dso_ids_to_run`` is non-empty the
        # runner constructs OFO controllers only for the listed DSOs.  The
        # remaining HV sub-networks still exist in the plant network and
        # exchange power through their coupling 3W transformers, but they
        # have no OFO controller — their DERs run only the plant-side
        # Q(V) / cos(phi) loop and their OLTC taps stay at the value
        # computed during the OLTC initialisation phase.  Used by
        # ``003_M_DSO_CIGRE_2026.py`` to focus the optimisation on DSO_2.
        if config.dso_ids_to_run and hv.net_id not in config.dso_ids_to_run:
            if verbose >= 1:
                print(f"  [7] {hv.net_id}: skipped (not in dso_ids_to_run)")
            continue
        dso_id = hv.net_id  # e.g. "DSO_1"
        interface_trafos = list(hv.coupling_trafo_indices)
        # Every DSO DER stays as pp.sgen under Q_cor mode (no promotion).
        der_indices = list(hv.sgen_indices)
        v_buses = list(hv.bus_indices)

        # HV lines — filter to PQ-bus endpoints only (same as TN lines)
        hv_lines = [
            li for li in hv.line_indices
            if int(net.line.at[li, "from_bus"]) not in pv_and_slack_buses_run
            and int(net.line.at[li, "to_bus"]) not in pv_and_slack_buses_run
        ]
        hv_line_max = [float(net.line.at[li, "max_i_ka"]) for li in hv_lines]

        # G_w diagonal: [Q_cor_DER | OLTC_tap].  Q_cor units are Mvar,
        # so g_w_dso_der (1/Mvar²) is the right knob.
        dso_gw_diag = np.concatenate([
            np.full(len(der_indices), config.g_w_dso_der),
            np.full(len(interface_trafos), config.g_w_dso_oltc),
        ])

        dso_cfg = DSOControllerConfig(
            der_indices=der_indices,
            oltc_trafo_indices=interface_trafos,
            shunt_bus_indices=[],
            shunt_q_steps_mvar=[],
            interface_trafo_indices=interface_trafos,
            voltage_bus_indices=v_buses,
            current_line_indices=hv_lines,
            current_line_max_i_ka=hv_line_max if hv_lines else None,
            g_q=config.g_q,
            g_qi=config.dso_g_qi,
            lambda_qi=config.dso_lambda_qi,
            q_integral_max_mvar=config.dso_q_integral_max_mvar,
            v_setpoints_pu=np.full(len(v_buses), v_set),
            g_v=config.dso_g_v,
            gamma_oltc_q=config.dso_gamma_oltc_q,
            qv_slope_pu=config.dso_qv_slope_pu,
            use_q_cor_actuator=config.use_q_cor_actuator,
        )

        dso_s_rated = np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in der_indices],
            dtype=np.float64,
        )
        dso_p_max = np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in der_indices],
            dtype=np.float64,
        )

        dso_der_op_diagrams = []
        for s in der_indices:
            od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
            dso_der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")

        dso_bounds = ActuatorBounds(
            der_indices=np.array(der_indices, dtype=np.int64),
            der_s_rated_mva=dso_s_rated,
            der_p_max_mw=dso_p_max,
            oltc_indices=np.array(interface_trafos, dtype=np.int64),
            oltc_tap_min=np.array(
                [int(net.trafo3w.at[t, "tap_min"]) for t in interface_trafos],
                dtype=np.int64,
            ),
            oltc_tap_max=np.array(
                [int(net.trafo3w.at[t, "tap_max"]) for t in interface_trafos],
                dtype=np.int64,
            ),
            shunt_indices=np.array([], dtype=np.int64),
            shunt_q_mvar=np.array([], dtype=np.float64),
            der_op_diagrams=dso_der_op_diagrams,
        )

        n_iface = len(interface_trafos)
        n_v = len(v_buses)
        n_i = len(hv_lines)
        # Q_cor mode: no V_gf, no Q_realized soft rows.
        dso_gz_target = np.concatenate([
            np.full(n_iface, config.g_z_interface),  # interface-Q slacks
            np.full(n_v,     config.g_z_voltage),    # voltage slacks
            np.full(n_i,     config.g_z_current),    # current slacks
        ])
        if config.g_z_warmup_s > 0:
            dso_gz = np.where(dso_gz_target > 0, config.g_z_warmup_value, 0.0)
        else:
            dso_gz = dso_gz_target
        dso_ofo = OFOParameters(
            g_w=dso_gw_diag,
            g_z=dso_gz,
            g_u=np.zeros_like(dso_gw_diag),
            alpha=1.0,
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
            int_cooldown_s=config.oltc_cooldown_s,
            adapt_g_w_classes=config.dso_adapt_g_w_classes(),
            g_w_adapt_meta=config.make_g_w_adapt_meta(),
        )

        dso_ctrl = DSOController(
            controller_id=dso_id,
            params=dso_ofo,
            config=dso_cfg,
            network_state=ns0,
            actuator_bounds=dso_bounds,
            sensitivities=shared_jac,
        )
        # _u_current is initialised later (step 7e), after profiles and
        # OLTC/STATCOM init have settled the operating point.
        dso_controllers[dso_id] = dso_ctrl

        if verbose >= 1:
            print(f"  {dso_id} (zone {hv.zone}): {len(der_indices)} DER, "
                  f"{n_iface} PCC trafos, {n_v} V-buses, {n_i} lines")

    if verbose >= 1 and config.dso_mode != "local":
        print(f"  [T] step [7] DSO controller construction: {perf_counter() - _t_step6:.2f} s")

    # Map each DSO controller ID to the ID of its supervising TSO controller.
    # TSO controller IDs follow the pattern "tso_zone_{z}" (see TSOController init above).
    dso_to_tso_id: Dict[str, str] = {
        hv.net_id: f"tso_zone_{hv.zone}"
        for hv in meta.hv_networks
    }

    # DSO group map (trivial: each DSO = its own group)
    dso_group_map: Dict[str, str] = {hv.net_id: hv.net_id for hv in meta.hv_networks}
    last_dso_q_set_mvar: Dict[str, Optional[NDArray]] = {
        dso_id: None for dso_id in dso_ids
    }

    # =========================================================================
    # STEP 8: Initialise MultiTSOCoordinator
    # =========================================================================
    if verbose >= 1:
        print()
        print("[8] Initialising MultiTSOCoordinator ...")

    coordinator = MultiTSOCoordinator(
        zones=list(zone_defs.values()),
        net=net,
        verbose=verbose,
    )
    for z, ctrl in tso_controllers.items():
        coordinator.register_tso_controller(z, ctrl)

    # =========================================================================
    # STEP 9: Load profiles and compute zonal generator dispatch
    # =========================================================================
    use_profiles = config.use_profiles
    start_time = config.start_time
    contingencies = list(config.contingencies) if config.contingencies else []

    profiles = None
    gen_dispatch = None

    _t_init_total = perf_counter()

    if use_profiles:
        profiles_csv = config.profiles_csv or DEFAULT_PROFILES_CSV
        if verbose >= 1:
            print()
            print(f"[9] Loading profiles from {profiles_csv}")
            print(f"     start_time = {start_time:%d.%m.%Y %H:%M}")

        _t = perf_counter()
        profiles = load_profiles(profiles_csv, timestep_s=config.dt_s)
        snapshot_base_values(net)
        if verbose >= 1:
            print(f"  [T] load_profiles + snapshot_base_values: {perf_counter() - _t:.2f} s")

        # Pre-create dormant loads for load-contingency events (must be
        # after snapshot_base_values so base columns exist).
        if contingencies:
            prepare_load_contingencies(net, contingencies, verbose=verbose)

        # Clip profile DataFrame to the simulation window only.
        # Without this, compute_zonal_gen_dispatch iterates the full profile
        # horizon (up to 525 600 rows at 60 s resolution) unnecessarily.
        # Note: must clip BOTH start and end — load_profiles returns the
        # full year (e.g. 2016-01-01 .. 2016-12-31).  ``profiles.loc[:t_end]``
        # alone would still iterate every row from the CSV start through
        # ``start_time``, which for an April start_time is ~3.5 months of
        # rows that compute_zonal_gen_dispatch would scan in vain.
        t_end = start_time + timedelta(seconds=config.n_total_s)
        profiles = profiles.loc[start_time:t_end]

        # Apply initial profiles
        _t = perf_counter()
        apply_profiles(net, profiles, start_time)
        if verbose >= 1:
            print(f"  [T] apply_profiles: {perf_counter() - _t:.2f} s")

        if config.use_zonal_gen_dispatch:
            # Per-generator P_min: 20% of P_max (consistent with
            # GeneratorParameters.p_min_mw construction above).
            _gen_p_min_dict: Dict[int, float] = {
                int(g): float(net.gen.at[g, "p_mw"]) * 0.0
                for g in net.gen.index
            }
            _t = perf_counter()
            gen_dispatch = compute_zonal_gen_dispatch(
                net, profiles, zone_map,
                gen_p_min_mw=_gen_p_min_dict,
            )
            apply_gen_dispatch(net, gen_dispatch, start_time)
            if verbose >= 1:
                print(f"  [T] compute+apply zonal gen dispatch: {perf_counter() - _t:.2f} s")

        # Re-converge after profile application
        _t = perf_counter()
        pp.runpp(net, max_iteration=50, run_control=False, calculate_voltage_angles=True, init='auto',
                 distributed_slack=config.distributed_slack,
                 enforce_q_lims=config.enforce_q_lims_plant)
        if verbose >= 1:
            print(f"  [T] post-profile pp.runpp: {perf_counter() - _t:.2f} s")

    # ── STEP 10: Combined operating-point init (three phases) ────────────
    # After profiles, bring STATCOM Q, OLTC taps, and plant-side Q(V)
    # loops to a self-consistent state at the profile-scaled operating
    # point.  Done in three phases so the TN settles *before* the coupler
    # 3W OLTCs adjust, and the plant-side q_mode loops install AFTER both:
    #
    #   Phase 1 (TSO):  STATCOM Q (temp-PV-gen trick) + machine 2W OLTC
    #                   → one run_control PF at v_setpoint_pu.
    #   Phase 2 (DSO):  coupler 3W OLTC
    #                   → one run_control PF at oltc_init_v_target_pu.
    #   Phase 3 (DER):  install QVLocalLoop / CosPhiConstLoop per DER
    #                   (deferred from step [4]) and seed q_mvar with
    #                   the analytical closed-loop equilibrium.
    #
    # In "cascade" DSO mode the coupler controllers are dropped after
    # Phase 2 (OFO takes over).  In "local" DSO mode they stay active
    # as local AVR for the rest of the simulation.
    from pandapower.control import DiscreteTapControl

    v_init_mt  = v_set                         # machine trafos → v_setpoint
    v_init_dso = config.oltc_init_v_target_pu  # coupler MV-side → 1.03
    tol_pu     = config.dso_oltc_init_tol_pu
    _local_dso = config.dso_mode == "local"
    _local_tso = config.tso_mode == "local"
    # Plant-side Q(V) loops (QVLocalLoop / CosPhiConstLoop) plus any
    # DiscreteTapControl on couplers / machine trafos must be iterated
    # by pp.runpp(run_control=True) every step.  Always True under the
    # Q_cor plant model.
    _run_control = True

    # -- Phase 1: STATCOM Q + machine 2W OLTC -----------------------------
    # TSO-side only: HV-side (subnet=="DN") STATCOMs stay at q_mvar=0
    # here; the DSO controller dispatches their Q at run time.
    _statcom_mask = (
        net.sgen["name"].astype(str).str.contains("STATCOM")
        & (net.sgen["subnet"].astype(str) != "DN")
    )
    _statcom_idxs = net.sgen.index[_statcom_mask].tolist()

    _tmp_map: Dict[int, int] = {}
    for si in _statcom_idxs:
        bus = int(net.sgen.at[si, "bus"])
        p = float(net.sgen.at[si, "p_mw"])
        sn = float(net.sgen.at[si, "sn_mva"])
        net.sgen.at[si, "in_service"] = False
        gi = pp.create_gen(
            net, bus=bus, p_mw=p, vm_pu=v_set, sn_mva=sn,
            max_q_mvar=sn, min_q_mvar=-sn,
            in_service=True, name="_TEMP_INIT",
        )
        _tmp_map[int(gi)] = si

    for tidx in meta.machine_trafo_indices:
        DiscreteTapControl(
            net, element_index=tidx,
            vm_lower_pu=v_init_mt - tol_pu,
            vm_upper_pu=v_init_mt + tol_pu,
            side="hv", element="trafo",
        )

    if verbose >= 1:
        print(f"[10.1] Phase 1 (TSO): {len(_tmp_map)} STATCOM Q via temp-PV-gens + "
              f"{len(meta.machine_trafo_indices)} machine OLTC "
              f"-> target {v_init_mt:.3f} +-{tol_pu:.3f} p.u.")

    _t = perf_counter()
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] Phase 1 pp.runpp(run_control=True): {perf_counter() - _t:.2f} s")

    # Transfer Q from temp-PV-gens to STATCOM sgens, then drop temp gens.
    for gi, si in _tmp_map.items():
        net.sgen.at[si, "q_mvar"] = float(net.res_gen.at[gi, "q_mvar"])
        net.sgen.at[si, "in_service"] = True
    if _tmp_map:
        net.gen.drop(index=list(_tmp_map.keys()), inplace=True)

    # Drop machine-trafo controllers — but preserve any plant-side
    # QVLocalLoops if they were already installed (defensive: under the
    # current flow they install at [10.3], after this drop, so this is a
    # no-op unless an external pre_loop_hook seeded them earlier).
    if hasattr(net, "controller") and len(net.controller) > 0:
        drop_idx = [
            idx for idx, row in net.controller.iterrows()
            if not isinstance(row["object"], QVLocalLoop)
        ]
        if drop_idx:
            net.controller.drop(index=drop_idx, inplace=True)

    if verbose >= 2:
        print("[10.1] Phase 1 result (machine 2W OLTC):")
        for tidx, gidx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
            tap = int(net.trafo.at[tidx, "tap_pos"])
            hv_bus = int(net.trafo.at[tidx, "hv_bus"])
            vm = float(net.res_bus.at[hv_bus, "vm_pu"])
            print(f"    trafo {tidx} (gen {gidx}): tap_pos={tap:+d}, "
                  f"V_hv={vm:.4f} p.u.")

    # -- Phase 2: coupler 3W OLTC -----------------------------------------
    for hv in meta.hv_networks:
        for t3w in hv.coupling_trafo_indices:
            DiscreteTapControl(
                net, element_index=t3w,
                vm_lower_pu=v_init_dso - tol_pu,
                vm_upper_pu=v_init_dso + tol_pu,
                side="mv", element="trafo3w",
            )

    if verbose >= 1:
        n_coup = sum(len(hv.coupling_trafo_indices) for hv in meta.hv_networks)
        print(f"[10.2] Phase 2 (DSO): {n_coup} coupler 3W OLTC "
              f"-> target {v_init_dso:.3f} +-{tol_pu:.3f} p.u.")

    _t = perf_counter()
    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=100, distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] Phase 2 pp.runpp(run_control=True): {perf_counter() - _t:.2f} s")

    if verbose >= 2:
        for hv in meta.hv_networks:
            for t3w in hv.coupling_trafo_indices:
                tap = int(net.trafo3w.at[t3w, "tap_pos"])
                mv_bus = int(net.trafo3w.at[t3w, "mv_bus"])
                vm = float(net.res_bus.at[mv_bus, "vm_pu"])
                print(f"  {hv.net_id} trafo3w {t3w}: tap_pos={tap:+d}, "
                      f"V_mv={vm:.4f} p.u.")

    # In "cascade" DSO mode, drop coupler controllers (OFO takes over).
    # In "local" DSO mode, keep them active as local AVR.  Either way
    # preserve the plant-side QVLocalLoops if any are present (defensive;
    # under the current flow they install at [10.3] AFTER this drop).
    if not _local_dso:
        if hasattr(net, "controller") and len(net.controller) > 0:
            drop_idx = [
                idx for idx, row in net.controller.iterrows()
                if not isinstance(row["object"], (QVLocalLoop, CosPhiConstLoop))
            ]
            if drop_idx:
                net.controller.drop(index=drop_idx, inplace=True)
    elif verbose >= 1:
        print(f"  [local DSO] Kept {len(net.controller)} coupler OLTC "
              f"DiscreteTapControl(s) active for simulation.")

    # ── Phase 3 (DER): install plant-side q_mode loops ──
    # Phase 1/2 init has settled; now install QVLocalLoop / CosPhiConstLoop
    # on every tagged DER so the local Q(V) feedback runs through the
    # main loop alongside the OFO.  TSO and DSO DERs install separately
    # so each gets its level-appropriate convergence tolerance
    # (TSO: 0.1 Mvar, DSO: 0.01 Mvar by default).
    #
    # Seed each DER's q_mvar with the *exact linear closed-loop
    # equilibrium* (Soleimani §IV-B eq. 18) computed from the post-
    # Phase-2 V via ``seed_qv_equilibrium``.  This bypasses the
    # multi-DER Gauss-Jacobi instability that broke the 24-hour main()
    # run with 500-iteration controller loops: the controllers install
    # already at their attractor and only need to refine residual
    # nonlinearity over a handful of iterations.
    tso_sgens = [int(s) for s in meta.tso_der_indices
                 if int(s) in net.sgen.index]
    dso_sgens = [int(s) for s in meta.dso_der_indices
                 if int(s) in net.sgen.index]
    # Reset q_cor_mvar (controller has not commanded yet) before
    # the analytical seed.
    for s in tso_sgens + dso_sgens:
        net.sgen.at[s, "q_cor_mvar"] = 0.0
    # Seed using a fresh Jacobian at the post-Phase-2 operating
    # point so S_VQ matches the network state we'll iterate from.
    _seed_jac = JacobianSensitivities(net)
    seed_qv_equilibrium(
        net, tso_sgens + dso_sgens, _seed_jac,
        verbose=(verbose >= 1),
    )
    # TSO STATCOMs have R*S_VQ ~ 8 vs DSO ~ 0.7, so cap TSO damping
    # to 0.03 to keep the per-DER contraction stable under multi-DER
    # coupling (commit e2746fe rationale; previously documented in
    # qv_local_damping docstring but not enforced in code).
    _tso_damp = min(float(config.qv_local_damping), 0.03)
    n_tso = len(install_der_q_loops(
        net, tso_sgens,
        qv_damping=_tso_damp,
        qv_max_step_frac=config.qv_local_max_step_frac,
        qv_tol_mvar=config.tso_qv_tol_mvar,
    )) if tso_sgens else 0
    n_dso = len(install_der_q_loops(
        net, dso_sgens,
        qv_damping=config.qv_local_damping,
        qv_max_step_frac=config.qv_local_max_step_frac,
        qv_tol_mvar=config.dso_qv_tol_mvar,
    )) if dso_sgens else 0
    if verbose >= 1:
        print(
            f"[10.3] Phase 3 (DER): installed {n_tso + n_dso} DER q_mode loops "
            f"({n_tso} TSO @ tol={config.tso_qv_tol_mvar} Mvar damp={_tso_damp:.3f} + "
            f"{n_dso} DSO @ tol={config.dso_qv_tol_mvar} Mvar damp={float(config.qv_local_damping):.3f}) "
            f"post Phase 2; seeded q_mvar with closed-loop equilibrium."
        )

    # Re-converge with final Q and tap positions.  Run *without* the
    # plant-side controllers here: Phase 2 left the network at a
    # converged operating point and the QVLocalLoops install at a damped
    # droop seed (above), then iterate inside the first main-loop runpp.
    # Running run_control=True here would force NR to digest a 44-DER Q
    # step in one call, making inner NR ill-conditioned on large profiles
    # (long sims with contingencies).
    _t = perf_counter()
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             max_iteration=50,
             distributed_slack=config.distributed_slack,
             enforce_q_lims=config.enforce_q_lims_plant)
    if verbose >= 1:
        print(f"  [T] final re-converge pp.runpp: {perf_counter() - _t:.2f} s")

    # ── Slack bus diagnostic after OLTC init ──────────────────────────────
    if verbose >= 1:
        # Prefer the slack-gen form (IEEE 39 distributed slack); fall back
        # to the legacy ext_grid form (TUDA benchmark, other networks).
        _slack_p, _slack_q = float("nan"), float("nan")
        if "slack" in net.gen.columns and len(net.gen) > 0:
            _slack_gens = net.gen.index[net.gen["slack"].astype(bool)].tolist()
            if _slack_gens:
                _sg = _slack_gens[0]
                _slack_p = float(net.res_gen.at[_sg, "p_mw"])
                _slack_q = float(net.res_gen.at[_sg, "q_mvar"])
        if (not np.isfinite(_slack_p)) and not net.ext_grid.empty:
            _sg = net.ext_grid.index[0]
            _slack_p = float(net.res_ext_grid.at[_sg, "p_mw"])
            _slack_q = float(net.res_ext_grid.at[_sg, "q_mvar"])
        print(f"  Slack bus: P = {_slack_p:.1f} MW, Q = {_slack_q:.1f} Mvar")
        # Warn on extreme machine trafo taps
        for tidx in meta.machine_trafo_indices:
            tap = int(net.trafo.at[tidx, "tap_pos"])
            if abs(tap) >= 7:
                hv_bus = int(net.trafo.at[tidx, "hv_bus"])
                print(f"  WARNING: Machine trafo {tidx} at tap {tap:+d} (bus {hv_bus})")

    # ── Voltage feasibility check after OLTC init ──────────────────────
    _v_violations_found = False
    if verbose >= 1:
        print(f"  {'Network':<12s} {'V_min':>7s} {'V_max':>7s} {'Headroom':>10s}")
        _items: list = []
        for hv in meta.hv_networks:
            v_buses = list(hv.bus_indices)
            vm_pu = net.res_bus.loc[v_buses, "vm_pu"].to_numpy(dtype=float)
            _items.append((hv.net_id, vm_pu))
        for z, zd in zone_defs.items():
            vm_pu = net.res_bus.loc[zd.v_bus_indices, "vm_pu"].to_numpy(dtype=float)
            _items.append((f"TSO Zone {z}", vm_pu))
        for label, vm_pu in _items:
            vm_min = float(np.min(vm_pu))
            vm_max = float(np.max(vm_pu))
            headroom = min(1.1 - vm_max, vm_min - 0.9)
            n_viol = int(np.sum(vm_pu < 0.9) + np.sum(vm_pu > 1.1))
            flag = "⚠" if n_viol > 0 or headroom < 0.02 else " "
            if n_viol > 0:
                _v_violations_found = True
            print(f"  {flag}{label:<11s} {vm_min:>7.4f} {vm_max:>7.4f} "
                  f"{headroom:>+9.4f} p.u.")
    if _v_violations_found:
        print("  ⚠ Voltage violations — MIQP may be infeasible with hard constraints.")

    # Rebuild shared Jacobian at the post-Phase-2 operating point and replace
    # the pre-profile snapshot held by every controller.  The H matrices
    # cached in each controller's _H_cache (and _sensitivity_updater) were
    # built from the stale Jacobian, so invalidate them — the next call to
    # _build_sensitivity_matrix will rebuild from the fresh shared_jac.
    _t = perf_counter()
    shared_jac = JacobianSensitivities(net)
    # Invalidate the seed_qv_equilibrium LU cache: a new sensitivities
    # object was constructed, so its id() changes and S_VQ_der may move.
    clear_seed_lu_cache()
    for ctrl in tso_controllers.values():
        ctrl.sensitivities = shared_jac
        ctrl.invalidate_sensitivity_cache()
    for dso_ctrl in dso_controllers.values():
        dso_ctrl.sensitivities = shared_jac
        dso_ctrl.invalidate_sensitivity_cache()
    if verbose >= 1:
        print(f"  [T] post-Phase-2 shared JacobianSensitivities rebuild + reassign: "
              f"{perf_counter() - _t:.2f} s")

    # Re-initialise all controllers so _u_current reflects the updated
    # operating point (profiles + correct tap positions).
    _t = perf_counter()
    for z, ctrl in tso_controllers.items():
        ctrl.initialise(measure_zone_tso(net, zone_defs[z], 0))
    for dso_id, dso_ctrl in dso_controllers.items():
        dso_ctrl.initialise(measure_zone_dso(net, dso_ctrl.config, 0))
    if verbose >= 1:
        print(f"  [T] controller .initialise() loop: {perf_counter() - _t:.2f} s")

    # ── Send initial DSO capability messages to TSO controllers ──────────
    # Without this, PCC capability bounds stay at the default ±1e-6 Mvar
    # until the first DSO step inside the loop.  The first TSO step then
    # sees near-zero capability and locks q_pcc; the second TSO step
    # (with real bounds) produces a large corrective jump.
    _t = perf_counter()
    for dso_id, dso_ctrl in dso_controllers.items():
        meas_init_dso = measure_zone_dso(net, dso_ctrl.config, 0)
        tso_id = dso_to_tso_id[dso_id]
        cap_msg = dso_ctrl.generate_capability_message(
            target_controller_id=tso_id,
            measurement=meas_init_dso,
        )
        target_tso = next(
            ctrl for ctrl in tso_controllers.values()
            if ctrl.controller_id == tso_id
        )
        target_tso.receive_capability(cap_msg)
    if verbose >= 1:
        print(f"  [T] DSO capability messages loop: {perf_counter() - _t:.2f} s")
        for z, ctrl in tso_controllers.items():
            n_pcc = len(zone_defs[z].pcc_trafo_indices)
            if n_pcc > 0:
                print(f"  Zone {z}: initial PCC capability "
                      f"[{ctrl.pcc_capability_min_mvar[0]:.1f}, "
                      f"{ctrl.pcc_capability_max_mvar[0]:.1f}] Mvar")

    # ── Cross-sensitivity computation (needed by stability analysis) ──────
    # Reuse the same shared Jacobian to avoid yet another deep-copy + PF +
    # dense inversion inside the coordinator.
    _t = perf_counter()
    coordinator.compute_cross_sensitivities(jac=shared_jac)
    if verbose >= 1:
        print(f"  [T] coordinator.compute_cross_sensitivities: {perf_counter() - _t:.2f} s")
    _t = perf_counter()
    coordinator.compute_M_blocks()
    if verbose >= 1:
        print(f"  [T] coordinator.compute_M_blocks: {perf_counter() - _t:.2f} s")

    _t = perf_counter()
    contraction_info = coordinator.check_contraction()
    if verbose >= 1:
        print(f"  [T] coordinator.check_contraction: {perf_counter() - _t:.2f} s")
        print(f"  [T] TOTAL init after [9]: {perf_counter() - _t_init_total:.2f} s")

    # Stability analysis is deferred until ``config.stability_analysis_at_s``
    # simulated seconds.  Running it at t=0 with an uncontrolled initial
    # operating point produces misleading curvature matrices.
    stab_result = None
    _stability_analysis_done = False

    # ── Optionally load tuned params from a previous run ────────────────
    # If ``config.load_tuned_params_path`` is set and points to a valid
    # JSON snapshot, apply those g_w / alpha values directly to the
    # controllers.  The delayed stability analysis still runs for
    # documentation.
    _tuned_params_loaded = False
    if config.load_tuned_params_path:
        if verbose >= 1:
            print(f"[11] Loading tuned params from "
                  f"{config.load_tuned_params_path} ...")
        try:
            _tuned_params_loaded = load_and_apply_tuned_params(
                config.load_tuned_params_path,
                zone_defs=zone_defs,
                tso_controllers=tso_controllers,
                dso_controllers=dso_controllers,
                verbose=verbose,
            )
        except Exception as _exc:
            if verbose >= 1:
                print(f"  ERROR loading tuned params: {_exc}")
            _tuned_params_loaded = False

    if contingencies and verbose >= 1:
        print(f"  Scheduled contingencies ({len(contingencies)}):")
        for ev in contingencies:
            t_label = f"t={ev.effective_time_s:.0f}s" if ev.time_s is not None else f"min {ev.minute}"
            print(f"    {t_label}: {ev.action} {ev.element_type}[{ev.element_index}]")


    # =========================================================================
    # STEP 12: Q-tracking capacity diagnostic
    # =========================================================================
    if verbose >= 2:
        import math as _math
        print()
        print("[12] Q-tracking capacity diagnostic")
        # DER Q capacity (VDE-AR-N 4120 v2)
        for did, hv in hv_info_map.items():
            sgens = net.sgen.loc[list(hv.sgen_indices)]
            res_sg = net.res_sgen.loc[list(hv.sgen_indices)]
            tot_qmin, tot_qmax, tot_qact = 0.0, 0.0, 0.0
            for idx in hv.sgen_indices:
                sn = net.sgen.at[idx, "sn_mva"]
                p_act = abs(net.res_sgen.at[idx, "p_mw"])
                p_ratio = p_act / sn if sn > 0 else 0
                if p_ratio < 0.1:
                    qmin, qmax = 0.0, 0.0
                elif p_ratio < 0.2:
                    t = (p_ratio - 0.1) / 0.1
                    qmin = (-0.10 + t * (-0.23)) * sn
                    qmax = ( 0.10 + t * ( 0.31)) * sn
                else:
                    qmin, qmax = -0.33 * sn, 0.41 * sn
                tot_qmin += qmin; tot_qmax += qmax
                tot_qact += net.res_sgen.at[idx, "q_mvar"]
            # Load Q
            q_load = net.res_load.loc[list(hv.load_indices), "q_mvar"].sum()
            p_load = net.res_load.loc[list(hv.load_indices), "p_mw"].sum()
            # Line Q losses
            q_line_loss = net.res_line.loc[list(hv.line_indices), "ql_mvar"].sum()
            # Interface Q
            q_iface = sum(net.res_trafo3w.at[t, "q_hv_mvar"]
                          for t in hv.coupling_trafo_indices)
            print(f"  {did}:")
            print(f"    DER Q capacity:  [{tot_qmin:+.0f}, {tot_qmax:+.0f}] Mvar  "
                  f"(actual: {tot_qact:+.1f} Mvar)")
            print(f"    Load Q:          {q_load:+.0f} Mvar  "
                  f"(P={p_load:.0f} MW)")
            print(f"    Line Q losses:   {q_line_loss:+.1f} Mvar")
            print(f"    Interface Q(HV): {q_iface:+.1f} Mvar")
            print(f"    Required DER Q for Q_iface=0: "
                  f"{q_load + q_line_loss:.0f} Mvar (to compensate loads+losses)")

        # IEEE 39-bus line lengths between coupling buses
        print()
        print("  Transmission line lengths between coupling buses:")
        for did, hv in hv_info_map.items():
            cbs = list(hv.coupling_ieee_buses)
            for i, b1 in enumerate(cbs):
                for b2 in cbs[i+1:]:
                    mask = ((net.line.from_bus == b1) & (net.line.to_bus == b2)) | \
                           ((net.line.from_bus == b2) & (net.line.to_bus == b1))
                    if mask.any():
                        for lidx in net.line.index[mask]:
                            L = net.line.at[lidx, "length_km"]
                            print(f"    {did}: TN line {b1}-{b2}: {L:.1f} km (345 kV)")
        print("  HV sub-network line lengths:")
        for did, hv in hv_info_map.items():
            lines = net.line.loc[list(hv.line_indices)]
            print(f"    {did} (scale={hv.line_length_scale}): "
                  f"range [{lines.length_km.min():.1f}, {lines.length_km.max():.1f}] km, "
                  f"total={lines.length_km.sum():.0f} km (110 kV)")


    # =========================================================================
    # STEP 13: Main simulation loop
    # =========================================================================
    if verbose >= 1:
        n_steps = int(config.n_total_s / config.dt_s)
        dur_str = f"start={start_time:%d.%m.%Y %H:%M}  " if use_profiles else ""
        print()
        warmup_str = f", warmup={config.warmup_s:.0f}s" if config.warmup_s > 0 else ""
        print(f"[13] Starting simulation: {n_steps} steps  "
              f"({dur_str}dt={config.dt_s:.0f}s, TSO/{config.tso_period_s/60:.0f}min, "
              f"DSO/{config.dso_period_s/60:.0f}min{warmup_str})")
        print()

    log: List[MultiTSOIterationRecord] = []

    # ── Optionally create live plot windows (three figures, 1/3 screen each) ─
    _plotter_tso = None
    _plotter_dso = None
    _plotter_sys = None

    if config.live_plot_controller:
        from visualisation.plot_tso_controller import TSOControllerLivePlotter
        _plotter_tso = TSOControllerLivePlotter(
            zone_ids=zone_ids_sorted,
            tie_line_pairs=sorted(tie_line_map.keys()),
            n_oltc_per_zone={z: len(zd.oltc_trafo_indices) for z, zd in zone_defs.items()},
            n_shunt_per_zone={
                z: len(getattr(zd, "shunt_bus_indices", []) or [])
                for z, zd in zone_defs.items()
            },
            n_gridforming_per_zone={z: 0 for z in zone_defs},
            v_setpoint_pu=config.v_setpoint_pu,
            v_min_pu=0.9, v_max_pu=1.1,
            sub_minute=False, update_every=1, slot_idx=0,
            layout=config.live_plot_layout,
            show_line_currents=config.live_plot_show_line_currents,
            use_tex=config.live_plot_use_tex,
        )

    if config.live_plot_cascade and dso_ids:
        from visualisation.plot_cascade_dso import CascadeDSOLivePlotter
        _plotter_dso = CascadeDSOLivePlotter(
            dso_ids=dso_ids,
            v_setpoint_pu=config.v_setpoint_pu,
            v_min_pu=0.9, v_max_pu=1.1,
            sub_minute=False, update_every=1, slot_idx=1,
            layout=config.live_plot_layout,
            show_line_currents=config.live_plot_show_line_currents,
            use_tex=config.live_plot_use_tex,
        )

    if config.live_plot_system:
        from visualisation.plot_system_power_flow import SystemPowerFlowLivePlotter
        # Interface trafo IDs mirror the record's trafo_key convention.
        # In OFO mode keys are "{dso_id}|trafo_{idx}"; in local mode they fall
        # back to "{group_id}|trafo_{idx}".  Build the OFO form when DSO
        # controllers are present, else the local form.
        if dso_controllers:
            _interface_trafo_ids = [
                f"{did}|trafo_{t}"
                for did, ctrl in dso_controllers.items()
                for t in ctrl.config.interface_trafo_indices
            ]
        else:
            _interface_trafo_ids = [
                f"{hv.net_id}|trafo_{t}"
                for hv in meta.hv_networks
                for t in hv.coupling_trafo_indices
            ]
        _plotter_sys = SystemPowerFlowLivePlotter(
            zone_ids=zone_ids_sorted,
            dso_ids=dso_ids,
            interface_trafo_ids=_interface_trafo_ids,
            zone_gen_indices={z: list(zd.gen_indices) for z, zd in zone_defs.items()},
            gen_limits_static=gen_limits_static,
            sub_minute=False, update_every=1, slot_idx=2,
            layout=config.live_plot_layout,
            use_tex=config.live_plot_use_tex,
        )

    def _is_period_hit(time_s: float, period_s: float) -> bool:
        """True if time_s is a multiple of period_s (within 1 s tolerance)."""
        rem = time_s % period_s
        return rem < 1.0 or abs(rem - period_s) < 1.0

    tso_step_count = 0  # count TSO steps for sensitivity refresh logic

    # ── g_z warmup: build target g_z vectors for the switch ──────────────
    _gz_warmup_done = (config.g_z_warmup_s <= 0)
    _gz_targets_tso: Dict[int, NDArray[np.float64]] = {}
    _gz_targets_dso: Dict[str, NDArray[np.float64]] = {}
    if not _gz_warmup_done:
        for z, zd in zone_defs.items():
            _gz_targets_tso[z] = np.concatenate([
                np.full(len(zd.v_bus_indices),     config.g_z_voltage),
                np.full(len(zd.pcc_trafo_indices), config.g_z_q_pcc),
                np.full(len(zd.line_indices),      config.g_z_current),
                np.full(len(zd.gen_indices),       config.g_z_q_gen),
                np.full(len(zd.tie_line_indices),  config.g_z_q_tie),
            ])
        for dso_id_tmp, dso_ctrl_tmp in dso_controllers.items():
            cfg_tmp = dso_ctrl_tmp.config
            n_iface_tmp = len(cfg_tmp.interface_trafo_indices)
            n_v_tmp = len(cfg_tmp.voltage_bus_indices)
            n_i_tmp = len(cfg_tmp.current_line_indices)
            _gz_targets_dso[dso_id_tmp] = np.concatenate([
                np.full(n_iface_tmp, config.g_z_interface),
                np.full(n_v_tmp,     config.g_z_voltage),
                np.full(n_i_tmp,     config.g_z_current),
            ])

    n_steps = int(config.n_total_s / config.dt_s)

    # ── TSO local-mode setup (one-shot, before main loop) ─────────────────
    # When the OFO TSO controller is skipped, two pieces keep TSO-side
    # primary voltage control alive (the windpark sgens already have
    # QVLocalLoop / CosPhiConstLoop installed by ``install_der_q_loops``
    # in step [10.3]):
    #
    #   (1) Generator AVR setpoints pinned to ``config.v_setpoint_pu``
    #       (1.03 pu by default).  Without OFO, nothing else writes
    #       net.gen.vm_pu, but we re-pin defensively in case a profile
    #       update touches it.
    #   (2) DiscreteTapControl on every machine 2W trafo, V_target =
    #       v_setpoint_pu, controlling the HV (grid) side.  These are
    #       the same controllers used in the Phase 1 OLTC init at
    #       lines ~1323; they were dropped after that init phase but
    #       must be re-installed to stay active for the simulation.
    _tso_der_idx_list: List[int] = [int(s) for s in meta.tso_der_indices]
    if _local_tso:
        # (1) Pin generator AVR setpoints
        net.gen.loc[:, "vm_pu"] = float(config.v_setpoint_pu)
        if verbose >= 1:
            print(f"  [local TSO] Pinned net.gen.vm_pu = {config.v_setpoint_pu:.3f} "
                  f"on {len(net.gen)} synchronous machines")

        # (2) Machine 2W OLTC DiscreteTapControl, HV side -> v_setpoint_pu
        _mt_tol_pu = config.dso_oltc_init_tol_pu
        for _tidx in meta.machine_trafo_indices:
            DiscreteTapControl(
                net, element_index=int(_tidx),
                vm_lower_pu=config.v_setpoint_pu - _mt_tol_pu,
                vm_upper_pu=config.v_setpoint_pu + _mt_tol_pu,
                side="hv", element="trafo",
            )
        if verbose >= 1:
            print(f"  [local TSO] Re-installed DiscreteTapControl on "
                  f"{len(meta.machine_trafo_indices)} machine 2W trafos "
                  f"(target HV side = {config.v_setpoint_pu:.3f} +/- "
                  f"{_mt_tol_pu:.3f} p.u.)")

    # ── Persistent OLTC rate-limiter for local-control mode ──────────────
    # Wraps the existing per-step ±max_step clamp with a wall-clock
    # cooldown (config.oltc_cooldown_s) so each DiscreteTapControl-managed
    # OLTC is locked for that many seconds after every actual tap
    # movement.  Only active when at least one local-mode tap controller
    # is present; the OFO MIQP path enforces the same cooldown via
    # OFOParameters.int_cooldown_s on its own integer block.
    _oltc_limiter = _OLTCRateLimiter(
        max_step=config.local_oltc_max_step_per_dt,
        cooldown_s=(
            config.oltc_cooldown_s if (_local_dso or _local_tso) else 0.0
        ),
    )
    _oltc_limiter_active = (_local_dso or _local_tso) and _oltc_limiter.active

    if pre_loop_hook is not None:
        _hook_state = {
            "net": net,
            "meta": meta,
            "tso_controllers": tso_controllers,
            "dso_controllers": dso_controllers,
            "shared_jac": shared_jac,
            "dso_to_tso_id": dso_to_tso_id,
            "zone_defs": zone_defs,
            "coordinator": coordinator,
            "config": config,
        }
        _hook_result = pre_loop_hook(_hook_state)
        if _hook_result:
            if verbose >= 1:
                print("[pre_loop_hook] returned truthy -- skipping main loop.")
            return []

    for step in range(1, n_steps + 1):
        _t_step = perf_counter()
        time_s  = step * config.dt_s
        run_tso = (step == 1) or _is_period_hit(time_s, config.tso_period_s)
        run_dso = _is_period_hit(time_s, config.dso_period_s)
        _in_warmup = time_s <= config.warmup_s
        # Track whether anything wrote new actuator commands this step.
        # Used to decide whether the end-of-step PF is needed: if no
        # MIQP fired and no contingency was applied, the post-profile
        # PF already reflects the final state.
        _contingency_fired_this_step = False

        # ── g_z warmup → activate output constraints ─────────────────────
        if not _gz_warmup_done and time_s >= config.g_z_warmup_s:
            _gz_warmup_done = True
            for z, ctrl in tso_controllers.items():
                ctrl.update_g_z(_gz_targets_tso[z])
            for did, dctrl in dso_controllers.items():
                dctrl.update_g_z(_gz_targets_dso[did])
            if verbose >= 1:
                print(f"  -- g_z warmup complete at t={time_s:.0f}s: "
                      f"output constraints activated "
                      f"(g_z_voltage={config.g_z_voltage:.0e}) --")

        rec = MultiTSOIterationRecord(
            step=step, time_s=time_s, tso_active=run_tso, dso_active=run_dso
        )

        # ── Local-mode OLTC rate-limit snapshot ──────────────────────────────
        # Snapshot every DiscreteTapControl tap_pos at the start of the
        # step so plant PFs in the step (post-profile, post-contingency,
        # end-of-step) can be clamped to ±config.local_oltc_max_step_per_dt
        # of these values AND blocked from moving twice within
        # config.oltc_cooldown_s seconds.  Only relevant when local-mode
        # tap controllers are present (cascade-DSO local mode and/or TSO
        # local mode); the OFO MIQP path manages its OLTCs via
        # int_max_step / int_cooldown / int_cooldown_s in the controller.
        if _oltc_limiter_active:
            _oltc_limiter.snapshot(net)

        # ── Apply time-series profiles ────────────────────────────────────────
        if use_profiles and profiles is not None:
            t_now = start_time + timedelta(seconds=time_s)
            t_profile = config.frozen_at if config.frozen_at is not None else t_now
            apply_profiles(net, profiles, t_profile)
            if gen_dispatch is not None:
                apply_gen_dispatch(net, gen_dispatch, t_profile)
            # Converge PF so that measurements (q_pcc, voltages) reflect the
            # new profiles/dispatch BEFORE controllers read them.
            # Warm-start the QVLocalLoops with the linear closed-loop
            # equilibrium so the run_control iteration only has to refine
            # the nonlinear residual.  Bypasses the multi-DER Gauss-Jacobi
            # coupling instability.
            # ToDo: pre seed currently disabled
            # seed_qv_equilibrium(
            #     net,
            #     list(meta.tso_der_indices) + list(meta.dso_der_indices),
            #     shared_jac,
            # )
            pp.runpp(net, run_control=_run_control, calculate_voltage_angles=True,
                     max_iteration=50,
                     max_iter=300,
                     distributed_slack=config.distributed_slack,
                     enforce_q_lims=config.enforce_q_lims_plant)
            if _oltc_limiter_active:
                _moved = _oltc_limiter.clamp(net, time_s)
                if _moved:
                    if verbose >= 1:
                        _pretty = ", ".join(
                            f"{tab}#{tid} {prev:+d}->{new:+d}"
                            for tab, tid, prev, new in _moved
                        )
                        print(
                            f"  [Step {step}] post-profile OLTC tap-rate limit "
                            f"({len(_moved)}): {_pretty}; re-running PF "
                            f"with run_control=False..."
                        )
                    pp.runpp(
                        net, run_control=False, calculate_voltage_angles=True,
                        max_iteration=100,
                        distributed_slack=config.distributed_slack,
                        enforce_q_lims=config.enforce_q_lims_plant,
                    )

        # ── Apply contingency events ──────────────────────────────────────────
        if contingencies:
            fired = [
                ev for ev in contingencies
                if abs(ev.effective_time_s - time_s) < 1e-9
            ]
            if fired:
                # Collect grid-side neighbourhood of any gen-trip event, BEFORE
                # applying the trip (after the trip, gen_trafo_map's trafo may
                # be OOS but still has hv_bus / lv_bus — still usable).
                watch_buses = _collect_contingency_watch_buses(
                    net, fired, gen_trafo_map
                )

                if verbose > 1:
                    _dump_contingency_diagnostics(
                        net, label=f"PRE-TRIP t={time_s:.0f}s",
                        watch_bus_0idx=watch_buses,
                    )

                _contingency_fired_this_step = True
                for ev in fired:
                    _apply_contingency(net, ev, verbose,
                                       gen_trafo_map=gen_trafo_map)

                # Re-converge PF with new topology so measurements
                # reflect the post-contingency operating point.
                # Re-seed the QVLocalLoops with the analytical closed-loop
                # equilibrium at the post-contingency operating point.
                # Without this the per-DER damped iteration starts from
                # the pre-contingency Q values and can hit the 501-
                # iteration controller cap on severe topology changes
                # (gen+trafo trips).  Mirrors the post-profile seed at the
                # top of the step.
                # ToDo: pre seed currently disabled
                # seed_qv_equilibrium(
                #     net,
                #     list(meta.tso_der_indices) + list(meta.dso_der_indices),
                #     shared_jac,
                # )
                try:
                    # First attempt: no Q-limit enforcement so that gens
                    # transiently producing Q outside their static box
                    # immediately after a topology change (gen trip) can
                    # converge in PV mode rather than cascading into
                    # PV→PQ flips that stall Newton-Raphson.  The
                    # subsequent end-of-step PF (with
                    # enforce_q_lims=config.enforce_q_lims_plant) clamps
                    # any out-of-box Q on the same step.  Keeping the
                    # asymmetry preserves the legacy retry path below
                    # (which adds enforce_q_lims=True as a recovery
                    # action when this unclipped attempt diverges).
                    pp.runpp(net, run_control=_run_control,
                             calculate_voltage_angles=True,
                             max_iteration=50,
                             max_iter=300,
                             distributed_slack=config.distributed_slack)
                    pf_converged = True
                except LoadflowNotConverged:
                    pf_converged = False
                    print("\n  *** Post-contingency PF did NOT converge "
                          "with default settings. ***")
                    print("  *** Running pp.diagnostic() to identify "
                          "topology / balance issues ***\n")
                    try:
                        pp.diagnostic(net, report_style="compact")
                    except Exception as exc:
                        print(f"  pp.diagnostic failed: {exc}")

                    print("\n  *** Retrying PF with enforce_q_lims=True, "
                          "init='flat', max_iteration=100 ***\n")
                    try:
                        pp.runpp(net, run_control=_run_control,
                                 calculate_voltage_angles=True,
                                 max_iteration=100,
                                 max_iter=300,
                                 distributed_slack=config.distributed_slack,
                                 enforce_q_lims=True,
                                 init="flat")
                        pf_converged = True
                        print("  *** Retry converged "
                              "→ original failure is Q-limit / warm-start "
                              "related. ***\n")
                    except LoadflowNotConverged:
                        print("  *** Retry with enforce_q_lims + flat start "
                              "ALSO diverged — structural issue. ***\n")
                        if verbose > 1:
                        # Dump what diagnostics we can without res_* tables
                            _dump_contingency_diagnostics(
                                net, label=f"POST-TRIP FAILED t={time_s:.0f}s",
                                watch_bus_0idx=watch_buses,
                            )
                        raise
                if verbose > 1:
                    _dump_contingency_diagnostics(
                        net, label=f"POST-TRIP t={time_s:.0f}s",
                        watch_bus_0idx=watch_buses,
                    )

                if _oltc_limiter_active:
                    _moved = _oltc_limiter.clamp(net, time_s)
                    if _moved:
                        if verbose >= 1:
                            _pretty = ", ".join(
                                f"{tab}#{tid} {prev:+d}->{new:+d}"
                                for tab, tid, prev, new in _moved
                            )
                            print(
                                f"  [Step {step}] post-contingency OLTC "
                                f"tap-rate limit ({len(_moved)}): {_pretty}; "
                                f"re-running PF with run_control=False..."
                            )
                        # Match the post-contingency PF's enforce_q_lims=False
                        # behaviour to keep convergence robust on the
                        # transient post-trip operating point.
                        pp.runpp(
                            net, run_control=False,
                            calculate_voltage_angles=True,
                            max_iteration=100,
                            distributed_slack=config.distributed_slack,
                            enforce_q_lims=False,
                        )

                # Notify controllers: freeze OOS actuator bounds, zero
                # their H columns, and invalidate sensitivity caches.
                coordinator.update_outage_masks(net)
                coordinator.invalidate_sensitivity_cache()

        # ── TSO step ──────────────────────────────────────────────────────────
        # Skipped entirely in TSO local mode: the CharacteristicControllers
        # (Q(V)) or the static cos phi=1 setting (Q=0) take over from the OFO
        # coordinator.  The DSO loop below is also disabled for L0/L1/L2 because
        # those scenarios use dso_mode='local'.
        if run_tso and not _local_tso:
            tso_step_count += 1
            # Decide whether to refresh cross-sensitivities this step
            refresh_H = (config.sensitivity_update_interval > 0
                         and tso_step_count % config.sensitivity_update_interval == 0)

            # Build per-zone measurements from plant network
            measurements: Dict[int, Measurement] = {
                z: measure_zone_tso(net, zd, step)
                for z, zd in zone_defs.items()
            }

            # Run decentralised TSO step for all zones
            tso_outputs = coordinator.step(
                measurements,
                step,
                recompute_cross_sensitivities=refresh_H,
                sim_time_s=time_s,
            )

            # Apply TSO controls to plant network
            for z, tso_out in tso_outputs.items():
                prev_shunt_steps = apply_zone_tso_controls(
                    net, zone_defs[z], tso_out,
                    use_q_cor_actuator=config.use_q_cor_actuator,
                )

                # Record per-zone results.
                # Q_cor mode actuator order on u (TSOControllerConfig._continuous_block):
                #   [ Q_cor_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt ]
                u = tso_out.u_new
                n_der = len(zone_defs[z].tso_der_indices)
                n_pcc = len(zone_defs[z].pcc_trafo_indices)
                n_gen = len(zone_defs[z].gen_indices)
                n_oltc = len(zone_defs[z].oltc_trafo_indices)
                n_shunt = len(zone_defs[z].shunt_bus_indices)
                off = 0
                rec.zone_q_der[z]         = u[off:off+n_der].copy(); off += n_der
                rec.zone_q_pcc_set[z]     = u[off:off+n_pcc].copy(); off += n_pcc
                rec.zone_v_gen[z]         = u[off:off+n_gen].copy(); off += n_gen
                rec.zone_v_gf[z]          = np.zeros(0)  # no V_gf actuator
                rec.zone_oltc_taps[z]     = u[off:off+n_oltc].copy(); off += n_oltc
                rec.zone_tso_objective[z] = tso_out.objective_value
                rec.zone_tso_status[z]    = tso_out.solver_status
                rec.zone_tso_solve_s[z]   = tso_out.solve_time_s

                # Record contraction diagnostic
                diag = coordinator.last_coupling_diagnostics.get(z, {})
                rec.zone_contraction_lhs[z] = diag.get("contraction_lhs", float("nan"))

                # ── TSO-owned shunt switching: detect and propagate ───────
                # When the MIQP switches a shunt step, apply a rank-1 SMW
                # update to the TSO's own cached J⁻¹ (no pp.runpp), drop
                # the H caches so the next TSO step rebuilds H from the
                # updated dV_dQ_reduced, then dispatch a
                # ``ShuntDisturbanceMessage`` to the DSO whose tertiary
                # hosts the shunt so the DSO can refresh its own model.
                if n_shunt > 0:
                    # Shunt block sits after [Q_cor_DER | Q_PCC | V_gen | OLTC].
                    shunt_offset = n_der + n_pcc + n_gen + n_oltc
                    new_shunt_steps = [
                        int(round(float(u[shunt_offset + k])))
                        for k in range(n_shunt)
                    ]
                    changed = [
                        (int(zone_defs[z].shunt_bus_indices[k]), s_new, k)
                        for k, (s_new, s_prev) in enumerate(
                            zip(new_shunt_steps, prev_shunt_steps)
                        )
                        if s_new != s_prev
                    ]
                    if changed:
                        tso_ctrl_z = tso_controllers[z]

                        # ── Diagnostic: which gradient component drove the
                        # switch?  Decompose the shunt column of H into V,
                        # I, Q_gen contributions and weight by the matching
                        # tracking error / soft-slack error.  Helpful for
                        # debugging "why did the shunt switch from -1 to 0?"
                        # — if the V-tracking term has the same sign as
                        # the actual move, the optimiser was tracking V;
                        # otherwise another term dominated.
                        H_z = getattr(tso_ctrl_z, "_H_cache", None)
                        last_meas = getattr(tso_ctrl_z, "_last_measurement", None)
                        if H_z is not None and last_meas is not None:
                            n_v_z = len(zone_defs[z].v_bus_indices)
                            n_pcc_z = len(zone_defs[z].pcc_trafo_indices)
                            n_i_z = len(zone_defs[z].line_indices)
                            n_gen_z = len(zone_defs[z].gen_indices)
                            v_set = float(config.v_setpoint_pu)
                            v_pu = np.asarray(
                                [float(last_meas.voltage_magnitudes_pu[
                                    np.where(last_meas.bus_indices == b)[0][0]
                                ]) for b in zone_defs[z].v_bus_indices],
                                dtype=np.float64,
                            )
                            v_err = v_pu - v_set
                            for sb, s_new, k_sh in changed:
                                col = shunt_offset + k_sh
                                s_prev_k = prev_shunt_steps[k_sh]
                                ds = s_new - s_prev_k
                                col_v_part = H_z[:n_v_z, col]
                                grad_v = float(v_err @ col_v_part)
                                # Q_gen contribution
                                # New row layout: [V | Q_PCC | I | Q_gen]
                                q_row_start = n_v_z + n_pcc_z + n_i_z
                                col_qg_part = H_z[q_row_start:q_row_start + n_gen_z, col]
                                # The gradient contribution from V tracks (V-V_set);
                                # for Q_gen it tracks (Q_gen - Q_gen_target) — but
                                # we don't have Q_target handy here; report magnitude
                                # so user can see if Q_gen column is non-trivial.
                                qg_norm = float(np.linalg.norm(col_qg_part))
                                v_min = float(v_pu.min())
                                v_max = float(v_pu.max())
                                # Expected V-driven move direction:
                                #   v_err·col_v < 0 → MIQP wants Δs > 0 (s up)
                                #   v_err·col_v > 0 → MIQP wants Δs < 0 (s down)
                                v_wants_up = grad_v < 0.0
                                actual_up = ds > 0
                                consistent = (v_wants_up == actual_up)
                                tag = "OK" if consistent else "INCONSISTENT-with-V"
                                print(
                                    f"  [shunt-switch z{z}] bus={sb}: "
                                    f"s {s_prev_k:+d}→{s_new:+d} (Δ={ds:+d})  "
                                    f"V[{v_min:.4f},{v_max:.4f}]  "
                                    f"V-grad={grad_v:+.3e}  "
                                    f"|Q_gen-col|={qg_norm:.3e}  "
                                    f"[{tag}]"
                                )

                        any_smw = False
                        for sb, s_new, _ in changed:
                            applied = tso_ctrl_z.sensitivities.apply_shunt_step_change_smw(
                                sb, s_new,
                            )
                            any_smw = any_smw or applied
                        if any_smw:
                            tso_ctrl_z.invalidate_sensitivity_cache()

                        # Dispatch a per-DSO ShuntDisturbanceMessage so each
                        # affected DSO updates its own cached J⁻¹.
                        per_dso: Dict[str, List[Tuple[int, int, float]]] = {}
                        for sb, s_new, k_idx in changed:
                            dso_id_aff = shunt_bus_to_dso_id.get(sb)
                            if dso_id_aff is None:
                                continue
                            q_step = float(zone_defs[z].shunt_q_steps_mvar[k_idx])
                            per_dso.setdefault(dso_id_aff, []).append(
                                (sb, s_new, q_step)
                            )
                        for dso_id_aff, items in per_dso.items():
                            dso_ctrl_aff = dso_controllers.get(dso_id_aff)
                            if dso_ctrl_aff is None:
                                continue
                            msg = ShuntDisturbanceMessage(
                                source_controller_id=tso_ctrl_z.controller_id,
                                target_controller_id=dso_id_aff,
                                iteration=step,
                                shunt_bus_indices=np.array(
                                    [it[0] for it in items], dtype=np.int64,
                                ),
                                shunt_steps=np.array(
                                    [it[1] for it in items], dtype=np.int64,
                                ),
                                shunt_q_steps_mvar=np.array(
                                    [it[2] for it in items], dtype=np.float64,
                                ),
                            )
                            dso_ctrl_aff.receive_disturbance_message(msg)

            # TSO sends Q setpoints to DSOs via grouped setpoint messages
            for z, ctrl in tso_controllers.items():
                for msg in ctrl.generate_setpoint_messages():
                    if msg.target_controller_id in dso_controllers:
                        dso_controllers[msg.target_controller_id].receive_setpoint(msg)
                        # Record total Q setpoint (sum over interface trafos)
                        rec.dso_q_set_mvar[msg.target_controller_id] = float(
                            msg.q_setpoints_mvar.sum()
                        )
                        last_dso_q_set_mvar[msg.target_controller_id] = (
                            msg.q_setpoints_mvar.copy()
                        )

        # Exogenous Q-setpoint injection for the no-TSO-OFO branch.  Used
        # by 003_M_DSO_CIGRE_2026 to push externally-defined Q_PCC setpoints
        # to one or more DSOs while ``tso_mode='local'`` keeps the TSO
        # layer purely under local Q(V) control.  Only runs when the TSO
        # OFO setpoint dispatch above is inactive (``_local_tso=True``)
        # AND the runner config supplies a setpoint dictionary.
        if (
            run_tso
            and _local_tso
            and config.q_pcc_setpoints_mvar_per_dso
        ):
            for dso_id, q_vec in config.q_pcc_setpoints_mvar_per_dso.items():
                if dso_id not in dso_controllers:
                    continue
                dso_ctrl_t = dso_controllers[dso_id]
                msg = SetpointMessage(
                    source_controller_id="exogenous",
                    target_controller_id=dso_id,
                    iteration=step,
                    interface_transformer_indices=np.array(
                        dso_ctrl_t.config.interface_trafo_indices,
                        dtype=np.int64,
                    ),
                    q_setpoints_mvar=np.asarray(q_vec, dtype=np.float64),
                )
                dso_ctrl_t.receive_setpoint(msg)
                rec.dso_q_set_mvar[dso_id] = float(msg.q_setpoints_mvar.sum())
                last_dso_q_set_mvar[dso_id] = msg.q_setpoints_mvar.copy()

        # ── DSO step (all zones) ──────────────────────────────────────────────
        if run_dso and not _local_dso:
            for dso_id, dso_ctrl in dso_controllers.items():
                # meas_dso reflects the current operating point BEFORE this DSO step.
                # This is the correct basis for the capability message: it tells the TSO
                # what the DSO can still do from its present dispatch, not what it just did.
                meas_dso = measure_zone_dso(net, dso_ctrl.config, step)

                # --- Capability message: DSO → TSO (must precede TSO solve) ----------
                tso_id = dso_to_tso_id[dso_id]
                cap_msg = dso_ctrl.generate_capability_message(
                    target_controller_id=tso_id,
                    measurement=meas_dso,
                )
                # Deliver to the responsible TSO controller.
                target_tso = next(
                    ctrl for ctrl in tso_controllers.values()
                    if ctrl.controller_id == tso_id
                )
                target_tso.receive_capability(cap_msg)

                # --- DSO optimisation step -------------------------------------------
                dso_out = dso_ctrl.step(meas_dso, sim_time_s=time_s)
                # Pass the live JacobianSensitivities handle so the
                # Q+shim apply step can read the **full** per-DSO S_VQ
                # block (cross-DER coupling included) directly off the
                # cached ``dV_dQ_reduced`` — no recompute, no extra PF.
                apply_dso_controls(
                    net, dso_ctrl.config, dso_out,
                    sensitivities=getattr(dso_ctrl, "sensitivities", None),
                )

        # ── End-of-step Power flow ─────────────────────────────────────
        # Skip when nothing wrote new actuator commands this step: the
        # post-profile (and post-contingency, if any) PFs already left
        # the network at its final operating point.  This is the typical
        # case for L0/L1 (no MIQP at all) and the off-cycle steps of
        # T0/T1 where DSO is local.  An MIQP fired ⇒ end-of-step PF is
        # required to propagate the new q_cor / V_set / OLTC commands.
        _miqp_acted = (
            (run_tso and not _local_tso)
            or (run_dso and not _local_dso)
        )
        _needs_end_pf = _miqp_acted or _contingency_fired_this_step
        if _needs_end_pf:
            # Warm-start QVLocalLoops from the linear closed-loop
            # equilibrium so run_control only refines residuals.
            seed_qv_equilibrium(
                net,
                list(meta.tso_der_indices) + list(meta.dso_der_indices),
                shared_jac,
            )
        try:
            if _needs_end_pf:
                try:
                    pp.runpp(net, run_control=_run_control, calculate_voltage_angles=True,
                             max_iteration=50,
                             max_iter=300,
                             distributed_slack=config.distributed_slack,
                             enforce_q_lims=config.enforce_q_lims_plant)
                except LoadflowNotConverged:
                    # End-of-step PF with enforce_q_lims=True can fail to
                    # converge on stressful events (e.g., a heavy load
                    # connect) when run_control=True is also active: the
                    # local Q(V) droop and the PV→PQ flips from
                    # enforce_q_lims interact and oscillate within the 50-NR
                    # iteration budget.  Fall back to a flat-start retry
                    # without Q-limit enforcement so the timestep still
                    # records a converged state; the next step's PFs (post-
                    # profile, post-contingency, or the next end-of-step PF
                    # itself) re-attempt with the clamp on a more relaxed
                    # operating point.
                    if verbose >= 1:
                        print(f"  [Step {step}] end-of-step PF with "
                              f"enforce_q_lims={config.enforce_q_lims_plant} "
                              f"diverged; retrying flat-start unclipped...")
                    pp.runpp(net, run_control=_run_control,
                             calculate_voltage_angles=True,
                             max_iteration=100,
                             max_iter=300,
                             distributed_slack=config.distributed_slack,
                             enforce_q_lims=False,
                             init='flat')
            if _oltc_limiter_active and _needs_end_pf:
                _moved = _oltc_limiter.clamp(net, time_s)
                if _moved:
                    if verbose >= 1:
                        _pretty = ", ".join(
                            f"{tab}#{tid} {prev:+d}->{new:+d}"
                            for tab, tid, prev, new in _moved
                        )
                        print(
                            f"  [Step {step}] end-of-step OLTC tap-rate "
                            f"limit ({len(_moved)}): {_pretty}; re-running "
                            f"PF with run_control=False..."
                        )
                    pp.runpp(
                        net, run_control=False, calculate_voltage_angles=True,
                        max_iteration=100,
                        distributed_slack=config.distributed_slack,
                        enforce_q_lims=config.enforce_q_lims_plant,
                    )
        except Exception as e:
            print(f"  [Step {step}] Power flow failed: {e}")
            log.append(rec)
            continue

        # ── Record post-PF observables (require converged res_* tables) ──────
        if run_dso and not _local_dso:
            for dso_id, dso_ctrl in dso_controllers.items():
                # Actual Q at PCC (sum over all 3W interface trafos)
                q_actual_sum = sum(
                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                    for t in dso_ctrl.config.interface_trafo_indices
                    if t in net.res_trafo3w.index
                )
                rec.dso_q_actual_mvar[dso_id] = q_actual_sum

            # Iterate only over DSOs with constructed OFO controllers
            # (``config.dso_ids_to_run`` may have restricted the set).
            _record_dso_group_and_transformer_data(
                rec=rec,
                net=net,
                dso_ids=list(dso_controllers.keys()),
                dsocontrollers=dso_controllers,
                dso_group_map=dso_group_map,
                last_dso_q_set_mvar=last_dso_q_set_mvar,
                hv_info_map=hv_info_map,
            )

        if _local_dso:
            # Record PCC Q actuals and HV group voltage stats even without OFO
            # DSO controllers so that comparison plots have the same dso_group_ids
            # as the coordinated scenario.
            for hv in meta.hv_networks:
                q_actual_sum = sum(
                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                    for t in hv.coupling_trafo_indices
                    if t in net.res_trafo3w.index
                )
                rec.dso_q_actual_mvar[hv.net_id] = q_actual_sum

                vm_hv = np.array(
                    [float(net.res_bus.at[b, "vm_pu"]) for b in hv.bus_indices
                     if b in net.res_bus.index],
                    dtype=np.float64,
                )
                if vm_hv.size:
                    rec.dso_group_v_min_pu[hv.net_id]  = float(vm_hv.min())
                    rec.dso_group_v_max_pu[hv.net_id]  = float(vm_hv.max())
                    rec.dso_group_v_mean_pu[hv.net_id] = float(vm_hv.mean())
                rec.dso_controller_group[hv.net_id] = hv.net_id

                # DER reactive power per HV group (post-PF measurement of
                # local-mode Q(V) droop or cos phi=1 dispatch).  Without
                # this the cascade-DSO live plot's "DER Q per HV group"
                # tile shows "no DSO DER dispatch available" in any
                # scenario with dso_mode='local' (L0/L1/L2 and T-OFO).
                # The ±sn_mva headroom band is a static box approximation
                # to the VDE-AR-N 4120 capability used by the OFO path
                # (compute_der_q_bounds); it is sufficient for live
                # visualisation in local mode where no controller drives
                # the DER toward the precise envelope.
                valid_der = [
                    s for s in hv.sgen_indices if s in net.res_sgen.index
                ]
                if valid_der:
                    rec.dso_group_q_der_mvar[hv.net_id] = float(
                        net.res_sgen.loc[valid_der, "q_mvar"].sum()
                    )
                    sn_total = float(
                        net.sgen.loc[valid_der, "sn_mva"].sum()
                    )
                    rec.dso_group_q_der_min_mvar[hv.net_id] = -sn_total
                    rec.dso_group_q_der_max_mvar[hv.net_id] = +sn_total
            _record_local_dso_trafo_data(rec, net, hv_info_map)
            _record_hv_group_observables(rec, net, hv_info_map)

        # ── Record plant voltages per zone ────────────────────────────────────
        for z, zd in zone_defs.items():
            vm_zone = np.array(
                [float(net.res_bus.at[b, "vm_pu"]) for b in zd.v_bus_indices
                 if b in net.res_bus.index],
                dtype=np.float64,
            )
            if vm_zone.size > 0:
                rec.zone_v_min[z]  = float(vm_zone.min())
                rec.zone_v_max[z]  = float(vm_zone.max())
                rec.zone_v_mean[z] = float(vm_zone.mean())

            # Generator P, Q from converged power flow (every step).
            # Live plots consume these each update, so they cannot be gated
            # on run_tso.
            if zd.gen_indices:
                rec.zone_q_gen[z] = np.array(
                    [net.res_gen.at[idx, "q_mvar"] for idx in zd.gen_indices],
                    dtype=np.float64,
                )
                rec.zone_p_gen[z] = np.array(
                    [net.res_gen.at[idx, "p_mw"] for idx in zd.gen_indices],
                    dtype=np.float64,
                )
                # Synchronous-machine Q headroom from the Milano §12.2.1
                # capability curve — matches the bound that the TSO MIQP
                # actually enforces (see TSOController._build_constraint_bounds
                # at controller/tso_controller.py:1000).  At each step the
                # bound is recomputed from the current P and terminal V via
                # ActuatorBounds.compute_gen_q_bounds.
                # Headroom = signed min margin: min(q_max - q, q - q_min).
                # Positive = inside envelope; negative = capability violated.
                gen_vm = np.array(
                    [float(net.gen.at[g, "vm_pu"]) for g in zd.gen_indices],
                    dtype=np.float64,
                )
                q_min_cap, q_max_cap = tso_controllers[z].actuator_bounds.compute_gen_q_bounds(
                    rec.zone_p_gen[z], gen_vm,
                )
                q_act = rec.zone_q_gen[z]
                rec.gen_q_headroom_mvar[z] = np.minimum(
                    q_max_cap - q_act, q_act - q_min_cap,
                )

            # ── Live-plot ACTUATORS tiles (TSO controller live plot) ──────
            # Populated from net state every step in BOTH OFO and local
            # modes:
            #   - zone_q_der:     net.sgen.q_mvar at each TSO DER index
            #   - zone_v_gen:     net.gen.vm_pu (AVR setpoint, constant in
            #                     local mode; OFO writes it from u_new)
            #   - zone_oltc_taps: net.trafo.tap_pos at machine 2W indices
            # The OFO TSO step also writes these from u_new on TSO ticks
            # (every 3 min); reading from net state every step gives smooth
            # time series across both modes.  Values reflect the converged
            # PF (PF does not modify sgen.q_mvar / gen.vm_pu / trafo.tap_pos
            # so for OFO they equal the commanded values).
            # zone_q_der: concatenate grid-following sgen Q (legacy path)
            # with the realised Q of any TSO grid-forming converter gen
            # in the zone (Stage-1 promoted units now live in net.gen).
            # This keeps the live plot showing the FULL TSO DER reactive
            # infeed, not just the un-promoted slice.  Sgen indices that
            # were promoted are no longer in zd.tso_der_indices, so the
            # two reads cover disjoint sets of physical units.
            der_q_following = (
                np.array(
                    [float(net.res_sgen.at[idx, "q_mvar"])
                     for idx in zd.tso_der_indices],
                    dtype=np.float64,
                )
                if zd.tso_der_indices
                else np.array([], dtype=np.float64)
            )
            if der_q_following.size:
                rec.zone_q_der[z] = der_q_following
            if zd.gen_indices:
                rec.zone_v_gen[z] = np.array(
                    [float(net.gen.at[idx, "vm_pu"]) for idx in zd.gen_indices],
                    dtype=np.float64,
                )
            if zd.oltc_trafo_indices:
                rec.zone_oltc_taps[z] = np.array(
                    [int(net.trafo.at[idx, "tap_pos"]) for idx in zd.oltc_trafo_indices],
                    dtype=np.int64,
                )

        # ── Total network losses (single scalar per record) ──────────────────
        rec.total_losses_mw = (
            float(net.res_line["pl_mw"].sum())
            + float(net.res_trafo["pl_mw"].sum())
            + float(net.res_trafo3w["pl_mw"].sum())
        )

        # ── Slack saturation diagnostic (added 2026-05-02) ───────────────────
        # Records the slack's P/Q every step plus a flag for whether |Q| is
        # within 1 % of max_q_mvar (saturation indicator).  Helps post-hoc
        # diagnosis of L0 / cos-phi-1 divergence: if the slack pegs at its
        # capability limit before NR fails, that is the proximate cause.
        if "slack" in net.gen.columns and len(net.gen) > 0:
            _slack_idxs = net.gen.index[net.gen["slack"].astype(bool)].tolist()
            if _slack_idxs:
                _sg = _slack_idxs[0]
                rec.slack_p_mw   = float(net.res_gen.at[_sg, "p_mw"])
                rec.slack_q_mvar = float(net.res_gen.at[_sg, "q_mvar"])
                _qmax = float(net.gen.at[_sg, "max_q_mvar"])
                _qmin = float(net.gen.at[_sg, "min_q_mvar"])
                _qabs_lim = max(abs(_qmax), abs(_qmin), 1.0)
                rec.slack_q_at_limit = bool(
                    abs(rec.slack_q_mvar) >= 0.99 * _qabs_lim
                )
        elif not net.ext_grid.empty:
            _sg = net.ext_grid.index[0]
            rec.slack_p_mw   = float(net.res_ext_grid.at[_sg, "p_mw"])
            rec.slack_q_mvar = float(net.res_ext_grid.at[_sg, "q_mvar"])
            rec.slack_q_at_limit = False  # ext_grid is unbounded

        # ── Record per-zone live-plot observables (loadings, balances,
        #    tie-line Q, shunt states) every step.
        _record_zone_live_plot_observables(
            rec=rec, net=net,
            zone_defs=zone_defs, tn_zone_map=tn_zone_map,
            tie_line_map=tie_line_map,
        )

        # ── Print progress ────────────────────────────────────────────────────
        if verbose >= 1 and run_tso:
            min_num = int(time_s / 60)
            v_info = "  ".join(
                f"Z{z}: [{rec.zone_v_min.get(z, float('nan')):.3f}, "
                f"{rec.zone_v_max.get(z, float('nan')):.3f}] p.u."
                for z in sorted(zone_defs.keys())
            )
            print(f"  t={min_num:3d} min | {v_info}")
            # ── Adaptive g_w live state per controller ─────────────────────
            # For each adapted class, show ``min..max`` of g_w_live across
            # the variables in that class.  Single-actuator classes (or
            # classes whose entries are all equal) collapse to one number.
            # Lines are skipped entirely when no controller has an active
            # adapter so the baseline log stays clean.
            def _fmt_class(arr):
                vmin = float(arr.min())
                vmax = float(arr.max())
                if arr.size == 1 or vmin == vmax:
                    return f"{vmin:.3g}"
                return f"{vmin:.3g}..{vmax:.3g}"

            tso_parts = []
            for z in sorted(tso_controllers.keys()):
                s = tso_controllers[z].adapter_summary()
                if s:
                    kv = " ".join(f"{k}={_fmt_class(v)}" for k, v in s.items())
                    tso_parts.append(f"Z{z}[{kv}]")
            if tso_parts:
                print(f"  t={min_num:3d} min | g_w TSO  {' '.join(tso_parts)}")
            dso_parts = []
            for did in sorted(dso_controllers.keys()):
                s = dso_controllers[did].adapter_summary()
                if s:
                    kv = " ".join(f"{k}={_fmt_class(v)}" for k, v in s.items())
                    dso_parts.append(f"{did}[{kv}]")
            if dso_parts:
                print(f"  t={min_num:3d} min | g_w DSO  {' '.join(dso_parts)}")
            if verbose >= 2:
                for z in sorted(zone_defs.keys()):
                    lhs = rec.zone_contraction_lhs.get(z, float("nan"))
                    print(f"    Zone {z}: contraction_lhs={lhs:.3f}  "
                          f"obj={rec.zone_tso_objective.get(z, float('nan')):.4e}")

        # ── Collect load-balance aggregates ──────────────────────────────
        _non_bound = ~net.sgen["name"].astype(str).str.startswith("BOUND_")
        rec.total_load_p_mw    = float(net.load["p_mw"].sum())
        rec.total_load_q_mvar  = float(net.load["q_mvar"].sum())
        rec.total_sgen_p_mw    = float(net.sgen.loc[_non_bound, "p_mw"].sum())
        rec.total_gen_p_mw     = float(net.res_gen["p_mw"].sum()) + float(net.res_ext_grid["p_mw"].sum())
        rec.total_gen_q_mvar   = float(net.res_gen["q_mvar"].sum()) + float(net.res_ext_grid["q_mvar"].sum())
        rec.residual_load_p_mw = rec.total_load_p_mw - rec.total_sgen_p_mw

        if _plotter_tso is not None:
            _plotter_tso.update(rec)
        if _plotter_dso is not None:
            _plotter_dso.update(rec)
        if _plotter_sys is not None:
            _plotter_sys.update(rec)

        if not _in_warmup:
            log.append(rec)

        if verbose >= 1:
            _dt_step = perf_counter() - _t_step
            _flags = []
            if run_tso:
                _flags.append("T")
            if run_dso:
                _flags.append("D")
            if _contingency_fired_this_step:
                _flags.append("X")
            if not _needs_end_pf:
                _flags.append("skip-endPF")
            _flag_str = ",".join(_flags) if _flags else "-"
            print(f"  [T] step {step:4d} t={time_s/60.0:6.1f} min  "
                  f"wall={_dt_step:5.2f} s  [{_flag_str}]")

        # ── Delayed auto-tune + stability analysis ──────────────────────
        # Triggered once when the simulated time crosses
        # ``config.stability_analysis_at_s``.  By default this is t=60
        # min, giving the controller time to equilibrate before we
        # auto-tune and analyse the operating point.  Running either at
        # t=0 would produce misleading results because the uncontrolled
        # initial state still has large tracking gradients.
        #
        # Sequence:
        #   1. (if config.run_stability_analysis) run the multi-zone
        #      stability report, print the compact summary, and write a
        #      markdown report in ``config.result_dir``.
        if (not _stability_analysis_done
                and time_s >= config.stability_analysis_at_s):
            _stability_analysis_done = True
            # NOTE: g_w tuning now runs at t=0 (before the main loop),
            # not here.  Only the delayed stability report remains.
            # Skip stability analysis entirely in TSO local mode: the
            # multi-zone OFO controllers are bypassed, so the spectral-gap
            # analysis is not meaningful (and would dereference state that
            # the local-mode runner never populates).
            if config.run_stability_analysis and not _local_tso:
                try:
                    stab_result = _run_delayed_stability_analysis(
                        config=config,
                        time_s=time_s,
                        net=net,
                        coordinator=coordinator,
                        zone_defs=zone_defs,
                        tso_controllers=tso_controllers,
                        dso_controllers=dso_controllers,
                        hv_info_map=hv_info_map,
                        verbose=verbose,
                    )
                except Exception as _exc:
                    if verbose >= 1:
                        print(f"  WARNING: delayed stability analysis failed: {_exc}")

    # =========================================================================
    # STEP 14: Print final summary
    # =========================================================================
    if verbose >= 1:
        print()
        print("=" * 72)
        print("  FINAL SUMMARY")
        print("=" * 72)
        for z, zd in zone_defs.items():
            last_rec = next((r for r in reversed(log) if z in r.zone_v_mean), None)
            if last_rec is None:
                continue
            v_mean = last_rec.zone_v_mean.get(z, float("nan"))
            v_err  = abs(v_mean - v_set)
            print(f"  Zone {z}: V_mean={v_mean:.4f} p.u.  |V - V_set|={v_err:.4f}")
        print("=" * 72)

        # ── DSO tracking quality ─────────────────────────────────────────────
        print()
        print("=" * 72)
        print("  DSO Q-TRACKING QUALITY")
        print("=" * 72)
        for dso_id in sorted(set().union(*(r.dso_q_set_mvar.keys() for r in log))):
            q_sets = []
            q_acts = []
            for r in log:
                qs = r.dso_q_set_mvar.get(dso_id)
                qa = r.dso_q_actual_mvar.get(dso_id)
                if qs is not None and qa is not None:
                    q_sets.append(qs)
                    q_acts.append(qa)
            if q_sets:
                errors = [abs(s - a) for s, a in zip(q_sets, q_acts)]
                print(f"  {dso_id}: Q_set={q_sets[-1]:+8.2f} Mvar, "
                      f"Q_act={q_acts[-1]:+8.2f} Mvar, "
                      f"|err|={errors[-1]:.2f} Mvar, "
                      f"mean|err|={np.mean(errors):.2f} Mvar, "
                      f"max|err|={max(errors):.2f} Mvar")
        print("=" * 72)

    return log
