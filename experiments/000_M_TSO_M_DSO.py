#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/run_M_TSO_M_DSO.py
========================
Multi-TSO / Multi-DSO OFO simulation loop on the IEEE 39-bus network.

This script is the multi-zone analogue of ``run/run_S_TSO_M_DSO.py``.  It uses
the same OFO controller infrastructure (TSOController, DSOController) but
orchestrates N=3 independent TSO zones via the MultiTSOCoordinator.  Each
zone has its own TSO controller and underlying DSO controllers, one per HV
sub-network (5 total: DSO_1..DSO_5 from add_hv_networks).

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

Network state caching
---------------------
Each controller's JacobianSensitivities caches the Jacobian at the current
operating point.  On each TSO step the cache is invalidated so the new
operating point is used.

Key differences from run_S_TSO_M_DSO.py
-------------------------------------
* Network: IEEE 39-bus + 5 HV sub-networks (build_ieee39_net + add_hv_networks).
* Zones: fixed_zone_partition_ieee39 (literature 3-area) or spectral_zone_partition.
* N TSOControllers instead of 1, coordinated by MultiTSOCoordinator.
* DSOs: 5 HV sub-networks with 3 PCC trafos each (2-winding 345/110 kV).
* No machine OLTCs: generators in case39 have direct AVR control (gen.vm_pu).
* Measurement functions are custom (handle multi-PCC and multi-zone).

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from numpy.typing import NDArray

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

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    MultiZoneStabilityResult,
)
from analysis.observer.stability_integration_ieee39 import (
    attach_observer,
    observer_record_fresh,
    write_observer_results_alongside_report,
    derive_tuned_gw,
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
)
from network.zone_partition import (
    fixed_zone_partition_ieee39,
    spectral_zone_partition,
    relabel_zones_by_generator_count,
    get_zone_lines,
    get_tie_lines,
)
from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers import (
    ContingencyEvent,
    MultiTSOIterationRecord,
    _apply_contingency,
    _network_state,
    apply_cos_phi_one_local_control,
    apply_dso_controls,
    apply_qv_local_control,
    apply_zone_tso_controls,
    install_cos_phi_one,
    install_qv_characteristic_controllers,
    prepare_load_contingencies,
)
from sensitivity.jacobian import JacobianSensitivities


def _collect_contingency_watch_buses(
    net: pp.pandapowerNet,
    events: List["ContingencyEvent"],
    gen_trafo_map: Dict[int, int],
) -> List[int]:
    """Grid-bus + first-order line neighbours for every fired gen-trip event."""
    watch: set[int] = set()
    for ev in events:
        if ev.element_type == "gen" and ev.action == "trip" \
                and ev.element_index in gen_trafo_map:
            t_idx = gen_trafo_map[ev.element_index]
            if t_idx in net.trafo.index:
                grid_bus = int(net.trafo.at[t_idx, "hv_bus"])
                watch.add(grid_bus)
                line_mask = (
                    (net.line["from_bus"] == grid_bus)
                    | (net.line["to_bus"] == grid_bus)
                )
                for li in net.line.index[line_mask]:
                    watch.add(int(net.line.at[li, "from_bus"]))
                    watch.add(int(net.line.at[li, "to_bus"]))
    return sorted(watch)


def _dump_contingency_diagnostics(
    net: pp.pandapowerNet,
    label: str,
    watch_bus_0idx: Optional[List[int]] = None,
) -> None:
    """Print gen P/Q utilisation, ext_grid load, and watched-bus voltages.

    Used around contingency events to identify whether PF divergence is a
    reactive-power / Q-limit issue vs. a slack redistribution issue.
    """
    print(f"\n  -- Contingency diagnostics: {label} --")

    # Generators: P, Q, slack weight, in_service, Q limits
    if not net.res_gen.empty:
        gen_df = pd.DataFrame(index=net.gen.index)
        gen_df["bus"] = net.gen["bus"]
        gen_df["in_srv"] = net.gen["in_service"]
        gen_df["slack_w"] = net.gen.get("slack_weight", np.nan)
        gen_df["p_mw"] = net.res_gen["p_mw"]
        gen_df["q_mvar"] = net.res_gen["q_mvar"]
        if "min_q_mvar" in net.gen.columns:
            gen_df["q_min"] = net.gen["min_q_mvar"]
        if "max_q_mvar" in net.gen.columns:
            gen_df["q_max"] = net.gen["max_q_mvar"]
        if "vm_pu" in net.gen.columns:
            gen_df["vm_set"] = net.gen["vm_pu"]
        print("  Generators:")
        print(gen_df.to_string(float_format="%.2f"))

    # ext_grid
    if not net.res_ext_grid.empty:
        eg_df = pd.DataFrame(index=net.ext_grid.index)
        eg_df["bus"] = net.ext_grid["bus"]
        eg_df["in_srv"] = net.ext_grid["in_service"]
        eg_df["slack_w"] = net.ext_grid.get("slack_weight", np.nan)
        eg_df["p_mw"] = net.res_ext_grid["p_mw"]
        eg_df["q_mvar"] = net.res_ext_grid["q_mvar"]
        if "vm_pu" in net.ext_grid.columns:
            eg_df["vm_set"] = net.ext_grid["vm_pu"]
        print("  ext_grid:")
        print(eg_df.to_string(float_format="%.2f"))

    # Aggregate: slack weight sum of in-service participants
    if "slack_weight" in net.gen.columns:
        in_srv_mask = net.gen["in_service"]
        sw_sum = float(net.gen.loc[in_srv_mask, "slack_weight"].sum())
        total_sw = float(net.gen["slack_weight"].sum())
        print(f"  slack_weight sum (in-service gens): {sw_sum:.3f} "
              f"/ total configured: {total_sw:.3f}")

    # Watched bus voltages
    if watch_bus_0idx and not net.res_bus.empty:
        buses_in = [b for b in watch_bus_0idx if b in net.res_bus.index]
        if buses_in:
            v_df = net.res_bus.loc[buses_in, ["vm_pu", "va_degree"]]
            print("  Watched bus voltages:")
            print(v_df.to_string(float_format="%.4f"))

    # Aggregate load + gen balance
    p_load = float(net.res_load["p_mw"].sum()) if not net.res_load.empty else 0.0
    q_load = float(net.res_load["q_mvar"].sum()) if not net.res_load.empty else 0.0
    p_gen = float(net.res_gen["p_mw"].sum()) if not net.res_gen.empty else 0.0
    q_gen = float(net.res_gen["q_mvar"].sum()) if not net.res_gen.empty else 0.0
    p_eg = float(net.res_ext_grid["p_mw"].sum()) if not net.res_ext_grid.empty else 0.0
    q_eg = float(net.res_ext_grid["q_mvar"].sum()) if not net.res_ext_grid.empty else 0.0
    p_sgen = float(net.res_sgen["p_mw"].sum()) if not net.res_sgen.empty else 0.0
    q_sgen = float(net.res_sgen["q_mvar"].sum()) if not net.res_sgen.empty else 0.0
    print(f"  Balance: P load={p_load:.1f}  gen={p_gen:.1f}  "
          f"ext_grid={p_eg:.1f}  sgen={p_sgen:.1f}  "
          f"Δ={p_gen + p_eg + p_sgen - p_load:.1f} MW")
    print(f"  Balance: Q load={q_load:.1f}  gen={q_gen:.1f}  "
          f"ext_grid={q_eg:.1f}  sgen={q_sgen:.1f}  "
          f"Δ={q_gen + q_eg + q_sgen - q_load:.1f} Mvar")
    print(f"  -- end diagnostics: {label} --\n")


def _record_dso_group_and_transformer_data(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    dso_ids: List[str],
    dsocontrollers: Dict[str, DSOController],
    dso_group_map: Dict[str, str],
    last_dso_q_set_mvar: Dict[str, Optional[NDArray]],
    hv_info_map: Dict[str, HVNetworkInfo],
) -> None:
    """
    Write DSO transformer- and network-group-level observables into rec.

    Each DSO may have multiple interface transformers (3 per HV sub-network).
    Per-trafo data is keyed by ``"{dso_id}|trafo_{trafo_idx}"``.
    """
    group_q_der: Dict[str, List[float]] = {}
    group_q_der_min: Dict[str, List[float]] = {}
    group_q_der_max: Dict[str, List[float]] = {}
    group_v_min: Dict[str, List[float]] = {}
    group_v_mean: Dict[str, List[float]] = {}
    group_v_max: Dict[str, List[float]] = {}

    for dso_id in dso_ids:
        if dso_id not in dsocontrollers:
            raise KeyError(f"Missing DSO controller '{dso_id}'.")
        if dso_id not in dso_group_map:
            raise KeyError(f"Missing network-group mapping for DSO '{dso_id}'.")

        dso_ctrl = dsocontrollers[dso_id]
        dsocfg = dso_ctrl.config
        group_id = dso_group_map[dso_id]

        # Retrieve per-trafo Q setpoints (vector or None)
        q_set_vec = last_dso_q_set_mvar.get(dso_id)

        rec.dso_controller_group[dso_id] = group_id

        # Per-trafo recording
        for k, trafo_idx in enumerate(dsocfg.interface_trafo_indices):
            trafo_idx = int(trafo_idx)
            trafo_key = f"{dso_id}|trafo_{trafo_idx}"

            rec.dso_trafo_group[trafo_key] = group_id

            if q_set_vec is not None and k < len(q_set_vec):
                rec.dso_trafo_q_set_mvar[trafo_key] = float(q_set_vec[k])
            elif q_set_vec is not None:
                rec.dso_trafo_q_set_mvar[trafo_key] = float(q_set_vec[0])

            if trafo_idx in net.res_trafo3w.index:
                rec.dso_trafo_q_actual_mvar[trafo_key] = float(
                    net.res_trafo3w.at[trafo_idx, "q_hv_mvar"]
                )
                rec.dso_trafo_p_actual_mw[trafo_key] = float(
                    net.res_trafo3w.at[trafo_idx, "p_hv_mw"]
                )
            if trafo_idx in net.trafo3w.index:
                rec.dso_trafo_tap_pos[trafo_key] = int(
                    net.trafo3w.at[trafo_idx, "tap_pos"]
                )

        # DER and voltage group data
        q_der_total = float(net.res_sgen.loc[dsocfg.der_indices, "q_mvar"].sum())
        vm_pu = net.res_bus.loc[dsocfg.voltage_bus_indices, "vm_pu"].to_numpy(dtype=float)

        # Per-DSO DER reactive power capability (sum over the DSO's DERs).
        # Used by the live plotter to draw a shaded headroom band around the
        # DER Q line. Bounds come from the VDE-AR-N 4120 capability curve in
        # actuator_bounds; they depend on the current active power dispatch.
        der_p = net.res_sgen.loc[dsocfg.der_indices, "p_mw"].to_numpy(dtype=float)
        q_min_arr, q_max_arr = dso_ctrl.actuator_bounds.compute_der_q_bounds(der_p)

        group_q_der.setdefault(group_id, []).append(q_der_total)
        group_q_der_min.setdefault(group_id, []).append(float(q_min_arr.sum()))
        group_q_der_max.setdefault(group_id, []).append(float(q_max_arr.sum()))
        if vm_pu.size > 0:
            group_v_min.setdefault(group_id, []).append(float(np.min(vm_pu)))
            group_v_mean.setdefault(group_id, []).append(float(np.mean(vm_pu)))
            group_v_max.setdefault(group_id, []).append(float(np.max(vm_pu)))

    for group_id, values in group_q_der.items():
        rec.dso_group_q_der_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_q_der_min.items():
        rec.dso_group_q_der_min_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_q_der_max.items():
        rec.dso_group_q_der_max_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_v_min.items():
        rec.dso_group_v_min_pu[group_id] = float(np.min(values))
    for group_id, values in group_v_mean.items():
        rec.dso_group_v_mean_pu[group_id] = float(np.mean(values))
    for group_id, values in group_v_max.items():
        rec.dso_group_v_max_pu[group_id] = float(np.max(values))

    # ── HV-group live-plot aggregates (line loading %, DER P, load P/Q) ─────
    _record_hv_group_observables(rec, net, hv_info_map)


def _record_hv_group_observables(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    hv_info_map: Dict[str, HVNetworkInfo],
) -> None:
    """Populate per-HV-group line-loading %, DER P, and load P/Q on the record.

    Works independently of controller state so it can be called from both
    the OFO and local-DSO paths.
    """
    for group_id, hv in hv_info_map.items():
        valid_lines = [li for li in hv.line_indices if li in net.res_line.index]
        if valid_lines:
            loadings = net.res_line.loc[valid_lines, "loading_percent"].to_numpy(dtype=float)
            rec.dso_group_i_max_pct[group_id]  = float(np.nanmax(loadings))
            rec.dso_group_i_mean_pct[group_id] = float(np.nanmean(loadings))
            rec.dso_group_i_min_pct[group_id]  = float(np.nanmin(loadings))
        if hv.sgen_indices:
            sgens = [s for s in hv.sgen_indices if s in net.res_sgen.index]
            if sgens:
                rec.dso_group_der_p_mw[group_id] = float(
                    net.res_sgen.loc[sgens, "p_mw"].sum()
                )
        if hv.load_indices:
            loads = [l for l in hv.load_indices if l in net.res_load.index]
            if loads:
                rec.dso_group_load_p_mw[group_id]    = float(
                    net.res_load.loc[loads, "p_mw"].sum()
                )
                rec.dso_group_load_q_mvar[group_id]  = float(
                    net.res_load.loc[loads, "q_mvar"].sum()
                )


def _record_local_dso_trafo_data(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    hv_info_map: Dict[str, HVNetworkInfo],
) -> None:
    """Populate per-trafo Q/P actuals and tap positions in local-DSO mode."""
    for group_id, hv in hv_info_map.items():
        for k, trafo_idx in enumerate(hv.coupling_trafo_indices):
            t = int(trafo_idx)
            trafo_key = f"{group_id}|trafo_{t}"
            rec.dso_trafo_group[trafo_key] = group_id
            if t in net.res_trafo3w.index:
                rec.dso_trafo_q_actual_mvar[trafo_key] = float(
                    net.res_trafo3w.at[t, "q_hv_mvar"]
                )
                rec.dso_trafo_p_actual_mw[trafo_key] = float(
                    net.res_trafo3w.at[t, "p_hv_mw"]
                )
            if t in net.trafo3w.index:
                rec.dso_trafo_tap_pos[trafo_key] = int(
                    net.trafo3w.at[t, "tap_pos"]
                )


def _record_zone_live_plot_observables(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    zone_defs: Dict[int, ZoneDefinition],
    tn_zone_map: Dict[int, List[int]],
    tie_line_map: Dict[Tuple[int, int], List[int]],
) -> None:
    """Populate per-zone line loadings, balance aggregates, tie-line Q, shunts.

    Called every step (after PF, regardless of run_tso/run_dso) to keep the
    live plotters fed with plant measurements.
    """
    # Per-zone line loadings + zone balance aggregates
    for z, zd in zone_defs.items():
        valid_lines = [li for li in zd.line_indices if li in net.res_line.index]
        if valid_lines:
            loadings = net.res_line.loc[valid_lines, "loading_percent"].to_numpy(dtype=float)
            rec.zone_line_loading_max_pct[z]  = float(np.nanmax(loadings))
            rec.zone_line_loading_mean_pct[z] = float(np.nanmean(loadings))
            rec.zone_line_loading_min_pct[z]  = float(np.nanmin(loadings))

        if zd.tso_der_indices:
            ders = [s for s in zd.tso_der_indices if s in net.res_sgen.index]
            if ders:
                p_arr = net.res_sgen.loc[ders, "p_mw"].to_numpy(dtype=float)
                q_arr = net.res_sgen.loc[ders, "q_mvar"].to_numpy(dtype=float)
                rec.zone_tso_der_p_mw[z]        = p_arr
                rec.zone_balance_der_p_mw[z]    = float(p_arr.sum())
                rec.zone_balance_der_q_mvar[z]  = float(q_arr.sum())

        if zd.gen_indices:
            gens = [g for g in zd.gen_indices if g in net.res_gen.index]
            if gens:
                rec.zone_balance_gen_p_mw[z]   = float(net.res_gen.loc[gens, "p_mw"].sum())
                rec.zone_balance_gen_q_mvar[z] = float(net.res_gen.loc[gens, "q_mvar"].sum())

        tn_bus_set = set(tn_zone_map.get(z, []))
        if tn_bus_set and len(net.load.index) > 0:
            zone_loads = net.load.index[net.load["bus"].isin(tn_bus_set)].tolist()
            zone_loads = [l for l in zone_loads if l in net.res_load.index]
            if zone_loads:
                rec.zone_balance_load_p_mw[z]   = float(
                    net.res_load.loc[zone_loads, "p_mw"].sum()
                )
                rec.zone_balance_load_q_mvar[z] = float(
                    net.res_load.loc[zone_loads, "q_mvar"].sum()
                )

        if zd.pcc_trafo_indices:
            pccs = [t for t in zd.pcc_trafo_indices if t in net.res_trafo3w.index]
            if pccs:
                rec.zone_balance_tso_dso_p_out_mw[z]   = float(
                    net.res_trafo3w.loc[pccs, "p_hv_mw"].sum()
                )
                rec.zone_balance_tso_dso_q_out_mvar[z] = float(
                    net.res_trafo3w.loc[pccs, "q_hv_mvar"].sum()
                )

        # Shunt states — ZoneDefinition currently carries shunt_bus_indices
        # but IEEE39 has no shunts so this is almost always empty.
        if getattr(zd, "shunt_bus_indices", None):
            shunt_rows = [s for s in zd.shunt_bus_indices if s in net.shunt.index]
            if shunt_rows:
                rec.zone_tso_shunt_states[z] = net.shunt.loc[
                    shunt_rows, "step"
                ].to_numpy(dtype=np.int64)
            else:
                rec.zone_tso_shunt_states[z] = np.array([], dtype=np.int64)
        else:
            rec.zone_tso_shunt_states[z] = np.array([], dtype=np.int64)

    # Inter-zone tie-line Q flow (positive = Q leaves zi toward zj)
    for (zi, zj), line_ids in tie_line_map.items():
        total = 0.0
        any_val = False
        bus_set_i = set(tn_zone_map.get(zi, []))
        for li in line_ids:
            if li not in net.res_line.index:
                continue
            q_from = float(net.res_line.at[li, "q_from_mvar"])
            fb = int(net.line.at[li, "from_bus"])
            total += q_from if fb in bus_set_i else -q_from
            any_val = True
        if any_val:
            rec.zone_tie_q_mvar[(zi, zj)] = total

# =============================================================================
#  Apply controls to plant network
# =============================================================================

# =============================================================================
#  Delayed stability analysis helpers
# =============================================================================

def _run_delayed_stability_analysis(
    *,
    config: "MultiTSOConfig",
    time_s: float,
    net,
    coordinator: "MultiTSOCoordinator",
    zone_defs,
    tso_controllers,
    dso_controllers,
    hv_info_map: Dict[str, "HVNetworkInfo"],
    verbose: int,
) -> "MultiZoneStabilityResult":
    """Refresh cross-sensitivities at the current operating point, run
    the multi-zone stability analysis, and save the results (per-zone +
    per-DSO g_w / alpha tables) to a markdown file under
    ``config.result_dir``.  Returns the stability result.
    """
    import os

    if verbose >= 1:
        print()
        print(f"[9] Running multi-zone stability analysis at t = {time_s/60.0:.1f} min ...")

    # Refresh the coordinator's cross-sensitivity blocks so the analysis
    # reflects the current operating point (profiles + controller state).
    coordinator.compute_cross_sensitivities()
    coordinator.compute_M_blocks()

    zone_ids_sorted = sorted(zone_defs.keys())
    H_blocks_stab = {k: coordinator.get_H_block(*k)
                     for k in coordinator._H_blocks}
    Q_obj_list = [zone_defs[z].q_obj_diagonal() for z in zone_ids_sorted]
    G_w_list   = [tso_controllers[z].params.g_w for z in zone_ids_sorted]
    actuator_counts = [
        {
            'n_der':  len(zone_defs[z].tso_der_indices),
            'n_pcc':  len(zone_defs[z].pcc_trafo_indices),
            'n_gen':  len(zone_defs[z].gen_indices),
            'n_oltc': len(zone_defs[z].oltc_trafo_indices),
            'n_shunt': len(zone_defs[z].shunt_bus_indices),
        }
        for z in zone_ids_sorted
    ]

    # Build DSO data for C1 analysis
    dso_data_list = []
    for dso_id_key, dso_ctrl in dso_controllers.items():
        dso_cfg_local = dso_ctrl.config
        n_interfaces = len(dso_cfg_local.interface_trafo_indices)
        n_voltage    = len(dso_cfg_local.voltage_bus_indices)
        n_current    = len(dso_cfg_local.current_line_indices)
        q_obj_dso = np.zeros(n_interfaces + n_voltage + n_current)
        q_obj_dso[:n_interfaces] = float(config.g_q)
        if dso_cfg_local.v_setpoints_pu is not None and n_voltage > 0:
            q_obj_dso[n_interfaces:n_interfaces + n_voltage] = float(config.dso_g_v)
        try:
            H_bus_dso = dso_ctrl._build_sensitivity_matrix()
            H_dso = dso_ctrl._expand_H_to_der_level(H_bus_dso)
        except Exception:
            continue
        dso_data_list.append({
            'H': H_dso, 'Q': q_obj_dso,
            'G_w': np.asarray(dso_ctrl.params.g_w).ravel(),
            'id': dso_id_key,
            'alpha': float(dso_ctrl.params.alpha),
            'actuator_counts': {
                'n_der': len(dso_cfg_local.der_indices),
                'n_oltc': len(dso_cfg_local.interface_trafo_indices),
                'n_shunt': len(dso_cfg_local.shunt_bus_indices),
            },
        })

    alpha_tso = tso_controllers[zone_ids_sorted[0]].params.alpha
    stab_result = analyse_multi_zone_stability(
        H_blocks=H_blocks_stab,
        Q_obj_list=Q_obj_list,
        G_w_list=G_w_list,
        zone_ids=zone_ids_sorted,
        zone_names=[f"Zone {z}" for z in zone_ids_sorted],
        actuator_counts=actuator_counts,
        alpha=alpha_tso,
        verbose=(verbose >= 1),
        dso_data=dso_data_list,
        tso_period_s=config.tso_period_s,
        dso_period_s=config.dso_period_s,
    )

    # Write markdown report + machine-readable JSON snapshot
    minutes = int(round(time_s / 60.0))
    md_path = os.path.join(config.result_dir,
                           f"stability_analysis_t{minutes}min.md")
    json_path = os.path.join(config.result_dir,
                             f"tuned_params_t{minutes}min.json")
    try:
        write_stability_analysis_markdown(
            md_path,
            time_s=time_s,
            config=config,
            net=net,
            zone_ids_sorted=zone_ids_sorted,
            zone_defs=zone_defs,
            tso_controllers=tso_controllers,
            dso_controllers=dso_controllers,
            hv_info_map=hv_info_map,
            stab_result=stab_result,
        )
        if verbose >= 1:
            print(f"  Stability report written to: {md_path}")
    except Exception as exc:
        if verbose >= 1:
            print(f"  WARNING: failed to write stability report to {md_path}: {exc}")

    try:
        write_tuned_params_json(
            json_path,
            time_s=time_s,
            zone_ids_sorted=zone_ids_sorted,
            zone_defs=zone_defs,
            tso_controllers=tso_controllers,
            dso_controllers=dso_controllers,
            hv_info_map=hv_info_map,
            stab_result=stab_result,
            pump_result=getattr(coordinator, "_last_pump_result", None),
        )
        if verbose >= 1:
            print(f"  Tuned params snapshot:       {json_path}")
            print(f"  (set config.load_tuned_params_path to this file "
                  f"to skip auto-tune next run)")
            print()
    except Exception as exc:
        if verbose >= 1:
            print(f"  WARNING: failed to write tuned params JSON to {json_path}: {exc}")

    return stab_result


# =============================================================================
#  Main simulation function
# =============================================================================

def run_multi_tso_dso(config: MultiTSOConfig) -> List[MultiTSOIterationRecord]:
    """
    Execute the multi-TSO / multi-DSO OFO simulation.

    Parameters
    ----------
    config : MultiTSOConfig
        All simulation parameters.  See dataclass docstring for details.

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

    #pp.runpp(net, run_control=False, calculate_voltage_angles=True)

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
        # Relabel: Zone 0 = most generators, Zone 2 = fewest (prime candidate for DSO)
        _gen_grid = list(meta.gen_grid_bus_indices or meta.gen_bus_indices)
        zone_map, bus_zone = relabel_zones_by_generator_count(
            zone_map, bus_zone, _gen_grid
        )

    if verbose >= 1:
        _gen_grid_set = set(meta.gen_grid_bus_indices or meta.gen_bus_indices)
        for z in sorted(zone_map.keys()):
            n_gen_z = sum(1 for b in zone_map[z] if b in _gen_grid_set)
            n_load_z = sum(
                1 for li in net.load.index if int(net.load.at[li, "bus"]) in zone_map[z]
            )
            print(f"  Zone {z}: {len(zone_map[z])} buses, "
                  f"{n_gen_z} generators, {n_load_z} loads")

    # =========================================================================
    # STEP 3: Attach 3 HV sub-networks (110 kV, TUDA topology)
    # =========================================================================
    if verbose >= 1:
        print()
        print("[3] Attaching 3 HV sub-networks (DSO_1..DSO_3) ...")

    meta = add_hv_networks(net, meta, verbose=(verbose >= 2))

    # add_hv_networks() may remove buses (e.g. bus 11/0-idx = IEEE bus 12).
    # Purge any removed buses from zone_map so downstream logic stays consistent.
    existing_buses = set(net.bus.index)
    for z in zone_map:
        zone_map[z] = [b for b in zone_map[z] if b in existing_buses]

    # =========================================================================
    # STEP 4: Build ZoneDefinitions and TSOControllerConfigs
    # =========================================================================
    if verbose >= 1:
        print()
        print("[4] Building zone definitions and controller configs ...")

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

    # ── Build ZoneDefinition for each zone ────────────────────────────────────
    #
    # TSO monitoring uses TN-only buses and lines (tn_zone_map).
    # Line filtering: TSOController's sensitivity builder (build_sensitivity_matrix_H)
    # only computes ∂I/∂u for lines where BOTH endpoints are PQ buses.  Lines
    # touching a PV generator bus are excluded from the I-rows of H_physical.
    # To avoid a shape mismatch we pre-filter zone lines to PQ-bus endpoints only.
    pv_and_slack_buses_run = (
        set(int(net.gen.at[g, "bus"]) for g in net.gen.index) |
        set(int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index)
    )

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
            v_setpoint_pu=v_set,
            # alpha removed (absorbed into g_w)
            g_v=config.g_v,
            g_w_der=config.g_w_der,
            g_w_gen=config.g_w_gen,
            g_w_pcc=config.g_w_pcc,
            g_w_oltc=config.g_w_tso_oltc,
        )

    if verbose >= 1:
        for z, zd in zone_defs.items():
            hv_names = [hv.net_id for hv in zone_hv_networks.get(z, [])]
            print(f"  Zone {z}: {len(zd.gen_indices)} gen, {len(zd.tso_der_indices)} DER, "
                  f"{len(zd.oltc_trafo_indices)} OLTC, "
                  f"{len(zd.line_indices)} lines, {len(zd.pcc_trafo_indices)} PCC trafos  "
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

    gen_limits_static: Dict[int, Dict[str, float]] = {}
    for g_idx in net.gen.index:
        limits: Dict[str, float] = {}
        for key in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"):
            limits[key] = (
                float(net.gen.at[g_idx, key])
                if key in net.gen.columns else float("nan")
            )
        gen_limits_static[int(g_idx)] = limits

    # =========================================================================
    # STEP 5: Initialise TSOControllers (one per zone)
    # =========================================================================
    if verbose >= 1:
        print()
        print("[5] Initialising TSOControllers ...")

    ns0 = _network_state(net)  # initial network state snapshot

    tso_controllers: Dict[int, TSOController] = {}
    for z, zd in zone_defs.items():

        # ── G_w diagonal for this zone's u vector ────────────────────────────
        gw_diag = zd.gw_diagonal()
        gz_diag_target = np.concatenate([
            np.full(len(zd.v_bus_indices), config.g_z_voltage),   # voltage slacks
            np.full(len(zd.line_indices),  config.g_z_current),   # current slacks
            np.full(len(zd.gen_indices),   config.g_z_q_gen),     # Q_gen slacks
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
            alpha=1.0,
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
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
            sensitivities=JacobianSensitivities(net),
        )
        # _u_current is initialised later (step 7e), after profiles and
        # OLTC/STATCOM init have settled the operating point.
        tso_controllers[z] = ctrl


    # =========================================================================
    # STEP 6: Initialise DSO controllers (one per HV sub-network, all zones)
    # =========================================================================
    dso_controllers: Dict[str, DSOController] = {}

    if config.dso_mode == "local":
        if verbose >= 1:
            print()
            print("[6] DSO mode = 'local' — skipping OFO DSO controllers.")
            print("    Coupler OLTCs: pandapower DiscreteTapControl (AVR)")
            n_der_total = sum(len(hv.sgen_indices) for hv in meta.hv_networks)
            if config.local_der_mode == "qv":
                print(f"    DER Q control: Q(V) linear droop, "
                      f"V_set={config.qv_setpoint_pu:.3f} p.u., "
                      f"slope={config.qv_slope_pu:.3f}  ({n_der_total} DER)")
            else:
                print(f"    DER Q control: cos phi = 1 (Q = 0 Mvar)  "
                      f"({n_der_total} DER)")

    if config.dso_mode == "local":
        # No OFO DSO controllers — skip to step 7.
        pass
    else:
        if verbose >= 1:
            print()
            print("[6] Initialising DSO controllers (5 HV sub-networks) ...")

    for hv in meta.hv_networks if config.dso_mode != "local" else []:
        dso_id = hv.net_id  # e.g. "DSO_1"
        interface_trafos = list(hv.coupling_trafo_indices)
        der_indices = list(hv.sgen_indices)
        v_buses = list(hv.bus_indices)

        # HV lines — filter to PQ-bus endpoints only (same as TN lines)
        hv_lines = [
            li for li in hv.line_indices
            if int(net.line.at[li, "from_bus"]) not in pv_and_slack_buses_run
            and int(net.line.at[li, "to_bus"]) not in pv_and_slack_buses_run
        ]
        hv_line_max = [float(net.line.at[li, "max_i_ka"]) for li in hv_lines]

        # G_w diagonal: [DER_Q | OLTC_tap]
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
        dso_gz_target = np.concatenate([
            np.full(n_iface, config.g_z_interface),   # interface-Q slacks
            np.full(n_v,     config.g_z_voltage),     # voltage slacks
            np.full(n_i,     config.g_z_current),     # current slacks
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
        )

        dso_ctrl = DSOController(
            controller_id=dso_id,
            params=dso_ofo,
            config=dso_cfg,
            network_state=ns0,
            actuator_bounds=dso_bounds,
            sensitivities=JacobianSensitivities(net),
        )
        # _u_current is initialised later (step 7e), after profiles and
        # OLTC/STATCOM init have settled the operating point.
        dso_controllers[dso_id] = dso_ctrl

        if verbose >= 1:
            print(f"  {dso_id} (zone {hv.zone}): {len(der_indices)} DER, "
                  f"{n_iface} PCC trafos, {n_v} V-buses, {n_i} lines")

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
    # STEP 7: Initialise MultiTSOCoordinator
    # =========================================================================
    if verbose >= 1:
        print()
        print("[7] Initialising MultiTSOCoordinator ...")

    coordinator = MultiTSOCoordinator(
        zones=list(zone_defs.values()),
        net=net,
        verbose=verbose,
    )
    for z, ctrl in tso_controllers.items():
        coordinator.register_tso_controller(z, ctrl)

    # Attach passive stability observer.  Records spectral-gap g_w^min per
    # zone at every TSO step using a freshly-computed H; the controller
    # continues to use its own cached H per
    # ``config.sensitivity_update_interval``.  The reported floor is a
    # DIAGNOSTIC of the unconstrained-OFO spectral-gap condition, not a
    # tuning suggestion for this MIQP-OFO controller (see
    # analysis/observer/DISCUSSION.md "Note on naming").
    # Gated by ``config.run_stability_observer``; when False the observer
    # is skipped entirely (no per-step recording, no end-of-run artefacts).
    if config.run_stability_observer:
        observer = attach_observer(coordinator, zone_defs, config, verbose=verbose)
    else:
        observer = None

    # =========================================================================
    # STEP 7b: Load profiles and compute zonal generator dispatch
    # =========================================================================
    use_profiles = config.use_profiles
    start_time = config.start_time
    contingencies = list(config.contingencies) if config.contingencies else []

    profiles = None
    gen_dispatch = None

    if use_profiles:
        profiles_csv = config.profiles_csv or DEFAULT_PROFILES_CSV
        if verbose >= 1:
            print()
            print(f"[7b] Loading profiles from {profiles_csv}")
            print(f"     start_time = {start_time:%d.%m.%Y %H:%M}")

        profiles = load_profiles(profiles_csv, timestep_s=config.dt_s)
        snapshot_base_values(net)

        # Pre-create dormant loads for load-contingency events (must be
        # after snapshot_base_values so base columns exist).
        if contingencies:
            prepare_load_contingencies(net, contingencies, verbose=verbose)

        # Clip profile DataFrame to the simulation window only.
        # Without this, compute_zonal_gen_dispatch iterates the full profile
        # horizon (up to 525 600 rows at 60 s resolution) unnecessarily.
        t_end = start_time + timedelta(seconds=config.n_total_s)
        profiles = profiles.loc[:t_end]  # keep only rows up to simulation end

        # Apply initial profiles
        apply_profiles(net, profiles, start_time)

        if config.use_zonal_gen_dispatch:
            # Per-generator P_min: 20% of P_max (consistent with
            # GeneratorParameters.p_min_mw construction above).
            _gen_p_min_dict: Dict[int, float] = {
                int(g): float(net.gen.at[g, "p_mw"]) * 0.0
                for g in net.gen.index
            }
            gen_dispatch = compute_zonal_gen_dispatch(
                net, profiles, zone_map,
                gen_p_min_mw=_gen_p_min_dict,
            )
            apply_gen_dispatch(net, gen_dispatch, start_time)

        # Re-converge after profile application
        pp.runpp(net, max_iteration=50, run_control=False, calculate_voltage_angles=True, init='auto',
                 distributed_slack=config.distributed_slack)

    # ── STEP 7c: Combined operating-point init (two phases) ─────────────
    # After profiles, bring STATCOM Q and OLTC taps to a self-consistent
    # state at the profile-scaled operating point.  Done in two phases so
    # the TN operating point settles *before* the coupler 3W OLTCs adjust:
    #
    #   Phase 1 (TSO):  STATCOM Q (temp-PV-gen trick) + machine 2W OLTC
    #                   → one run_control PF at v_setpoint_pu.
    #   Phase 2 (DSO):  coupler 3W OLTC
    #                   → one run_control PF at oltc_init_v_target_pu.
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
    # Combined "any local controller in net.controller table that should be
    # iterated by pp.runpp(run_control=...)".  True for cascade-DSO local mode
    # (DiscreteTapControl on couplers) AND for TSO local Q(V) mode
    # (CharacteristicControl on TSO windparks).
    _run_control = _local_dso or _local_tso

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
        print(f"[7c.1] Phase 1 (TSO): {len(_tmp_map)} STATCOM Q via temp-PV-gens + "
              f"{len(meta.machine_trafo_indices)} machine OLTC "
              f"-> target {v_init_mt:.3f} +-{tol_pu:.3f} p.u.")

    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, distributed_slack=config.distributed_slack)

    # Transfer Q from temp-PV-gens to STATCOM sgens, then drop temp gens.
    for gi, si in _tmp_map.items():
        net.sgen.at[si, "q_mvar"] = float(net.res_gen.at[gi, "q_mvar"])
        net.sgen.at[si, "in_service"] = True
    if _tmp_map:
        net.gen.drop(index=list(_tmp_map.keys()), inplace=True)

    # Drop machine-trafo controllers (only ones present at this point).
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    if verbose >= 2:
        print("[7c.1] Phase 1 result (machine 2W OLTC):")
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
        print(f"[7c.2] Phase 2 (DSO): {n_coup} coupler 3W OLTC "
              f"-> target {v_init_dso:.3f} +-{tol_pu:.3f} p.u.")

    pp.runpp(net, run_control=True, calculate_voltage_angles=True,
             max_iteration=50, distributed_slack=config.distributed_slack)

    if verbose >= 2:
        for hv in meta.hv_networks:
            for t3w in hv.coupling_trafo_indices:
                tap = int(net.trafo3w.at[t3w, "tap_pos"])
                mv_bus = int(net.trafo3w.at[t3w, "mv_bus"])
                vm = float(net.res_bus.at[mv_bus, "vm_pu"])
                print(f"  {hv.net_id} trafo3w {t3w}: tap_pos={tap:+d}, "
                      f"V_mv={vm:.4f} p.u.")

    # In "cascade" DSO mode, drop coupler controllers (OFO takes over).
    # In "local" DSO mode, keep them active as local AVR.
    if not _local_dso:
        if hasattr(net, "controller") and len(net.controller) > 0:
            net.controller.drop(net.controller.index, inplace=True)
    elif verbose >= 1:
        print(f"  [local DSO] Kept {len(net.controller)} coupler OLTC "
              f"DiscreteTapControl(s) active for simulation.")

    if _local_dso:
        n_dso_sgen = sum(len(hv.sgen_indices) for hv in meta.hv_networks)
        if config.local_der_mode == "qv":
            _v_set = config.qv_setpoint_pu
            _slope = config.qv_slope_pu
            # Converge Q(V) + OLTC iteratively: the Q(V) droop gain is too
            # high for pandapower's CharacteristicControl (inner NR diverges).
            # Instead, alternate between: (1) set Q from Q(V) characteristic
            # at current V, (2) run PF with DiscreteTapControl active.  This
            # converges because each step makes a bounded Q adjustment and
            # the OLTC settles taps to the resulting voltage.
            if verbose >= 1:
                print(f"  [local DSO] Converging Q(V) + OLTC iteratively "
                      f"({n_dso_sgen} DER, V_set={_v_set:.3f}, slope={_slope:.3f})")
            for _qv_iter in range(20):
                # Run PF with OLTC active (adjusts taps to current Q/V)
                pp.runpp(net, run_control=True, calculate_voltage_angles=True,
                         max_iteration=50)
                # Compute Q(V) from post-PF voltages
                max_dq = 0.0
                for hv in meta.hv_networks:
                    for s_idx in hv.sgen_indices:
                        bus = int(net.sgen.at[s_idx, "bus"])
                        v_pu = float(net.res_bus.at[bus, "vm_pu"])
                        sn = float(net.sgen.at[s_idx, "sn_mva"])
                        od = (net.sgen.at[s_idx, "op_diagram"]
                              if "op_diagram" in net.sgen.columns else None)
                        if str(od) == "STATCOM":
                            q_min, q_max = -sn, sn
                        else:
                            q_min, q_max = -0.33 * sn, 0.41 * sn
                        dv = v_pu - _v_set
                        frac = max(min(dv / _slope, 1.0), -1.0) if _slope > 0 else 0.0
                        q_new = q_min * frac if frac > 0 else q_max * (-frac)
                        max_dq = max(max_dq, abs(q_new - float(net.sgen.at[s_idx, "q_mvar"])))
                        net.sgen.at[s_idx, "q_mvar"] = float(q_new)
                if max_dq < 1.0:  # converged within 1 Mvar
                    break
            if verbose >= 1:
                total_q = sum(float(net.sgen.at[s, "q_mvar"])
                              for hv in meta.hv_networks for s in hv.sgen_indices)
                print(f"  [local DSO] Q(V)+OLTC converged in {_qv_iter+1} iter "
                      f"(max_dq={max_dq:.2f} Mvar, total Q={total_q:.1f} Mvar)")
        else:  # "cos_phi_1" (default)
            # Unity power factor: force Q=0 on every HV DER, then let
            # DiscreteTapControl settle OLTC taps to the resulting voltages.
            apply_cos_phi_one_local_control(net, meta.hv_networks)
            pp.runpp(net, run_control=True, calculate_voltage_angles=True,
                     max_iteration=50)
            if verbose >= 1:
                print(f"  [local DSO] cos phi=1 init: Q=0 Mvar on "
                      f"{n_dso_sgen} HV DER; OLTCs settled via DiscreteTapControl")

    # Re-converge with final Q and tap positions.
    pp.runpp(net, run_control=_run_control, calculate_voltage_angles=True,
             max_iteration=50,
             distributed_slack=config.distributed_slack)

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

    # Re-initialise all controllers so _u_current reflects the updated
    # operating point (profiles + correct tap positions).
    for z, ctrl in tso_controllers.items():
        ctrl.initialise(measure_zone_tso(net, zone_defs[z], 0))
    for dso_id, dso_ctrl in dso_controllers.items():
        dso_ctrl.initialise(measure_zone_dso(net, dso_ctrl.config, 0))

    # ── Send initial DSO capability messages to TSO controllers ──────────
    # Without this, PCC capability bounds stay at the default ±1e-6 Mvar
    # until the first DSO step inside the loop.  The first TSO step then
    # sees near-zero capability and locks q_pcc; the second TSO step
    # (with real bounds) produces a large corrective jump.
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
        for z, ctrl in tso_controllers.items():
            n_pcc = len(zone_defs[z].pcc_trafo_indices)
            if n_pcc > 0:
                print(f"  Zone {z}: initial PCC capability "
                      f"[{ctrl.pcc_capability_min_mvar[0]:.1f}, "
                      f"{ctrl.pcc_capability_max_mvar[0]:.1f}] Mvar")

    # ── Cross-sensitivity computation (needed by stability analysis) ──────
    coordinator.compute_cross_sensitivities()
    coordinator.compute_M_blocks()

    contraction_info = coordinator.check_contraction()

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
            print(f"[7.3] Loading tuned params from "
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
    # STEP 7d: Q-tracking capacity diagnostic
    # =========================================================================
    if verbose >= 2:
        import math as _math
        print()
        print("[7d] Q-tracking capacity diagnostic")
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
    # STEP 8: Main simulation loop
    # =========================================================================
    if verbose >= 1:
        n_steps = int(config.n_total_s / config.dt_s)
        dur_str = f"start={start_time:%d.%m.%Y %H:%M}  " if use_profiles else ""
        print()
        warmup_str = f", warmup={config.warmup_s:.0f}s" if config.warmup_s > 0 else ""
        print(f"[8] Starting simulation: {n_steps} steps  "
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
                np.full(len(zd.v_bus_indices), config.g_z_voltage),
                np.full(len(zd.line_indices),  config.g_z_current),
                np.full(len(zd.gen_indices),   config.g_z_q_gen),
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
    # Local-mode HV DER baseline (Q(V) droop or cos phi=1); active immediately
    # when no warmup, else enabled once warmup expires.
    _local_der_active = _local_dso and config.warmup_s <= 0

    def _apply_local_der() -> None:
        """Dispatch HV-DER baseline per ``config.local_der_mode``."""
        if config.local_der_mode == "qv":
            apply_qv_local_control(
                net, meta.hv_networks,
                v_set=config.qv_setpoint_pu,
                slope=config.qv_slope_pu,
            )
        else:  # "cos_phi_1" (default)
            apply_cos_phi_one_local_control(net, meta.hv_networks)

    # ── TSO local-mode setup (one-shot, before main loop) ─────────────────
    # In TSO local mode (L0/L1/L2 of the comparison experiment) the OFO
    # controllers are skipped entirely.  To keep the TSO-side primary
    # voltage control alive, three local-AVR pieces are installed here:
    #
    #   (1) Q(V) or cos phi=1 on every TSO-connected windpark sgen.
    #   (2) Generator AVR setpoints pinned to ``config.v_setpoint_pu``
    #       (1.03 pu by default).  Without OFO, nothing else writes
    #       net.gen.vm_pu, but we re-pin defensively in case a profile
    #       update touches it.
    #   (3) DiscreteTapControl on every machine 2W trafo, V_target =
    #       v_setpoint_pu, controlling the HV (grid) side.  These are
    #       the same controllers used in the Phase 1 OLTC init at
    #       lines ~1323; they were dropped after that init phase but
    #       must be re-installed to stay active for the simulation.
    _tso_der_idx_list: List[int] = [int(s) for s in meta.tso_der_indices]
    if _local_tso:
        # (1) Windpark Q control
        if _tso_der_idx_list:
            if config.tso_local_mode == "qv":
                install_qv_characteristic_controllers(
                    net, _tso_der_idx_list,
                    v_set=config.tso_qv_setpoint_pu,
                    slope=config.tso_qv_slope_pu,
                    name_prefix="qv_tso",
                )
                if verbose >= 1:
                    print(f"  [local TSO] Installed Q(V) CharacteristicControl on "
                          f"{len(_tso_der_idx_list)} windpark sgens "
                          f"(V_set={config.tso_qv_setpoint_pu:.3f}, "
                          f"slope={config.tso_qv_slope_pu:.3f})")
            else:  # "cos_phi_1"
                install_cos_phi_one(net, _tso_der_idx_list)
                if verbose >= 1:
                    print(f"  [local TSO] Forced cos phi=1 (Q=0) on "
                          f"{len(_tso_der_idx_list)} windpark sgens")

        # (2) Pin generator AVR setpoints
        net.gen.loc[:, "vm_pu"] = float(config.v_setpoint_pu)
        if verbose >= 1:
            print(f"  [local TSO] Pinned net.gen.vm_pu = {config.v_setpoint_pu:.3f} "
                  f"on {len(net.gen)} synchronous machines")

        # (3) Machine 2W OLTC DiscreteTapControl, HV side -> v_setpoint_pu
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

    def _apply_local_tso() -> None:
        """Re-apply TSO local-mode constraints after profile updates.

        For Q(V) mode this is a no-op: CharacteristicControl iterates inside
        every pp.runpp(run_control=True) using the current bus voltage.
        For cos phi=1 mode the profile may have rewritten q_mvar (e.g. via
        apply_profiles); re-zero it here so the next PF sees Q=0.
        """
        if not _local_tso or not _tso_der_idx_list:
            return
        if config.tso_local_mode == "cos_phi_1":
            install_cos_phi_one(net, _tso_der_idx_list)

    for step in range(1, n_steps + 1):
        time_s  = step * config.dt_s
        run_tso = (step == 1) or _is_period_hit(time_s, config.tso_period_s)
        run_dso = _is_period_hit(time_s, config.dso_period_s)
        _in_warmup = time_s <= config.warmup_s

        # ── Local-DER activation after warmup ─────────────────────────────
        if _local_dso and not _local_der_active and not _in_warmup:
            _local_der_active = True
            # Re-evaluate HV-DER baseline once from the settled TSO state
            _apply_local_der()
            pp.runpp(net, run_control=True, calculate_voltage_angles=True,
                     max_iteration=50,
                     distributed_slack=config.distributed_slack)
            if verbose >= 1:
                print(f"  -- local DER mode '{config.local_der_mode}' "
                      f"activated at t={time_s:.0f}s after "
                      f"{config.warmup_s:.0f}s warmup --")

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

        # ── Apply time-series profiles ────────────────────────────────────────
        if use_profiles and profiles is not None:
            t_now = start_time + timedelta(seconds=time_s)
            apply_profiles(net, profiles, t_now)
            if gen_dispatch is not None:
                apply_gen_dispatch(net, gen_dispatch, t_now)
            # Converge PF so that measurements (q_pcc, voltages) reflect the
            # new profiles/dispatch BEFORE controllers read them.
            # In local-DSO mode, re-evaluate the HV-DER baseline before the PF
            # so that Q is consistent with the current state after the profile
            # change. During warmup the baseline is inactive (DER Q follows
            # the profile).
            if _local_der_active:
                _apply_local_der()
            _apply_local_tso()
            pp.runpp(net, run_control=_run_control, calculate_voltage_angles=True,
                     max_iteration=50,
                     max_iter=100,
                     distributed_slack=config.distributed_slack)

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
                _dump_contingency_diagnostics(
                    net, label=f"PRE-TRIP t={time_s:.0f}s",
                    watch_bus_0idx=watch_buses,
                )

                for ev in fired:
                    _apply_contingency(net, ev, verbose,
                                       gen_trafo_map=gen_trafo_map)

                # Re-converge PF with new topology so measurements
                # reflect the post-contingency operating point.
                if _local_der_active:
                    _apply_local_der()
                _apply_local_tso()
                try:
                    pp.runpp(net, run_control=_run_control,
                             calculate_voltage_angles=True,
                             max_iteration=50,
                             max_iter=100,
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
                                 max_iter=100,
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
                        # Dump what diagnostics we can without res_* tables
                        _dump_contingency_diagnostics(
                            net, label=f"POST-TRIP FAILED t={time_s:.0f}s",
                            watch_bus_0idx=watch_buses,
                        )
                        raise

                _dump_contingency_diagnostics(
                    net, label=f"POST-TRIP t={time_s:.0f}s",
                    watch_bus_0idx=watch_buses,
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
            )

            # Passive stability recording.  Refreshes H for the observer only;
            # restores the controller's cached H_blocks immediately after.
            if observer is not None:
                observer_record_fresh(observer, coordinator, time_s=time_s)

            # Apply TSO controls to plant network
            for z, tso_out in tso_outputs.items():
                apply_zone_tso_controls(net, zone_defs[z], tso_out)

                # Record per-zone results
                u = tso_out.u_new
                n_der = len(zone_defs[z].tso_der_indices)
                n_pcc = len(zone_defs[z].pcc_trafo_indices)
                n_gen = len(zone_defs[z].gen_indices)
                n_oltc = len(zone_defs[z].oltc_trafo_indices)
                rec.zone_q_der[z]         = u[:n_der].copy()
                rec.zone_q_pcc_set[z]     = u[n_der:n_der+n_pcc].copy()
                rec.zone_v_gen[z]         = u[n_der+n_pcc:n_der+n_pcc+n_gen].copy()
                rec.zone_oltc_taps[z]     = u[n_der+n_pcc+n_gen:n_der+n_pcc+n_gen+n_oltc].copy()
                rec.zone_tso_objective[z] = tso_out.objective_value
                rec.zone_tso_status[z]    = tso_out.solver_status
                rec.zone_tso_solve_s[z]   = tso_out.solve_time_s

                # Record contraction diagnostic
                diag = coordinator.last_coupling_diagnostics.get(z, {})
                rec.zone_contraction_lhs[z] = diag.get("contraction_lhs", float("nan"))

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
                dso_out = dso_ctrl.step(meas_dso)
                apply_dso_controls(net, dso_ctrl.config, dso_out)

        # ── Power flow ────────────────────────────────────────────────────────
        if _local_der_active:
            # Local DSO mode: apply HV-DER baseline (Q(V) or cos phi=1) before
            # running the final PF with DiscreteTapControl for coupler OLTCs.
            _apply_local_der()
        _apply_local_tso()
        try:
            pp.runpp(net, run_control=_run_control, calculate_voltage_angles=True,
                     max_iteration=50,
                     max_iter=100,
                     distributed_slack=config.distributed_slack)
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

            _record_dso_group_and_transformer_data(
                rec=rec,
                net=net,
                dso_ids=dso_ids,
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
                # Synchronous-machine Q headroom: q_max - |q_actual|.
                # Positive = remaining capability, negative = capability
                # violated.  Used by 002_M_TSO_M_DSO_COMPARE.py.
                q_act = np.abs(rec.zone_q_gen[z])
                q_max = np.array(
                    [float(net.gen.at[g, "max_q_mvar"]) for g in zd.gen_indices],
                    dtype=np.float64,
                )
                rec.gen_q_headroom_mvar[z] = q_max - q_act

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
            if zd.tso_der_indices:
                rec.zone_q_der[z] = np.array(
                    [float(net.sgen.at[idx, "q_mvar"]) for idx in zd.tso_der_indices],
                    dtype=np.float64,
                )
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
    # STEP 9: Print final summary
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

    # =========================================================================
    # STEP 10: Stability observer trajectory report
    # =========================================================================
    if observer is not None:
        write_observer_results_alongside_report(
            observer, config.result_dir, verbose=verbose,
        )
        tuned = derive_tuned_gw(observer, statistic="percentile", percentile=95.0)
        if verbose >= 1 and tuned.per_zone:
            print()
            print("[9b] Spectral-gap floor (p95, unconstrained-OFO equivalent g_w):")
            print("     [DIAGNOSTIC ONLY -- not a tuning recommendation for the MIQP loop]")
            for z, vals in sorted(tuned.per_zone.items()):
                parts = [f"{k}={v:.2f}" for k, v in vals.items()]
                print(f"  Zone {z}: " + ", ".join(parts))

    return log


# =============================================================================
#  Comparison: coordinated vs. uncoordinated Q_PCC
# =============================================================================

def main_comparison() -> None:
    """Run coordinated vs. uncoordinated Q_PCC scenarios and compare.

    Invoke from the project root::

        python run/run_M_TSO_M_DSO.py --compare
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
        run_stability_analysis=True,
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
        python run/run_M_TSO_M_DSO.py
    """
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 60 * 16,      # 720-min full simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=10.0,    # DSO every 5 seconds (more inner iterations)
        g_v=120000.0,  # TSO voltage tracking; drives PCC Q dispatch
        g_q=200,  # DSO Q-tracking
        # ── DSO objective tuning ──
        dso_g_v=20000.0,  # reduced to avoid competing with Q tracking
        dso_g_qi=0,  # integral Q-tracking (0 = off)
        dso_lambda_qi=0.9,  # leaky integrator decay
        dso_q_integral_max_mvar=50.0,  # anti-windup clamp
        dso_gamma_oltc_q=0.0,  # OLTC Q-tracking attenuation: DER-primary, OLTC-backup
        # ── TSO weights (alpha=1, spectral rho(C)/2) ──
        g_w_der=10,   # single-DER zones; rho~C_jj=396 -> min 198
        g_w_gen=5e7,   # excluded from stability
        g_w_pcc=100,   # 9 correlated PCCs; rho(C)=221 -> min 111
        g_w_tso_oltc=100,
        # ── DSO weights (alpha=1, rho(C_DER)=790 -> min 395) ──
        g_w_dso_der=1000,  # 8 correlated DER; sf~2.5 for smooth tracking
        g_w_dso_oltc=50,   # rho(C_OLTC)~1.1; higher for switching suppression
        use_fixed_zones=True,      # literature 3-area partition (not spectral)
        run_stability_analysis=True,
        sensitivity_update_interval=1E6,  # refresh H_ij every N TSO steps
        verbose=1,
        live_plot_controller=True,
        live_plot_cascade=True,
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
            ContingencyEvent(minute=120, element_type="load", bus=5, p_mw=300, q_mvar=100, action="connect"),
            ContingencyEvent(minute=300, element_type="load", bus=5, p_mw=300, q_mvar=100, action="trip"),
            ContingencyEvent(minute=330, element_type="gen", element_index=2, action="trip"),
            ContingencyEvent(minute=420, element_type="gen", element_index=2, action="restore"),
            ContingencyEvent(minute=480, element_type="load", bus=27, p_mw=300, q_mvar=150, action="connect"),
            ContingencyEvent(minute=560, element_type="load", bus=27, p_mw=300, q_mvar=150, action="trip"),
            ContingencyEvent(minute=720, element_type="load", bus=7, p_mw=300, q_mvar=100, action="connect"),
            ContingencyEvent(minute=900, element_type="load", bus=7, p_mw=300, q_mvar=100, action="trip"),
        ],
    )
    log = run_multi_tso_dso(cfg)
    print(f"\nSimulation complete. {len(log)} steps recorded.")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        main_comparison()
    else:
        main()
