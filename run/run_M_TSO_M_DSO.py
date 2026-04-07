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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pandapower as pp
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
from tune_ofo_params import compute_optimal_gw
from controller.base_controller import OFOParameters
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.multi_tso_coordinator import MultiTSOCoordinator, ZoneDefinition
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from core.measurement import Measurement
from core.network_state import NetworkState
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_gen_dispatch,
    apply_profiles,
    compute_zonal_gen_dispatch,
    load_profiles,
    snapshot_base_values,
)
from network.build_ieee39_net import (build_ieee39_net, add_hv_networks, HVNetworkInfo,
                                      IEEE39NetworkMeta, remove_generators)
from network.zone_partition import (
    fixed_zone_partition_ieee39,
    spectral_zone_partition,
    relabel_zones_by_generator_count,
    get_zone_lines,
    get_tie_lines,
)
from run.contingency import _apply_contingency
from run.helpers import _network_state
from run.records import ContingencyEvent
from sensitivity.jacobian import JacobianSensitivities


# =============================================================================
#  Configuration dataclass
# =============================================================================

@dataclass
class MultiTSOConfig:
    """
    Central configuration for the multi-TSO-DSO simulation.

    All parameters have sensible defaults suitable for a first test run on
    the IEEE 39-bus.  Adjust to explore stability margins.

    Timing
    ------
    dt_s : float
        Simulation timestep [s].  Should be the GCD of tso_period_s and
        dso_period_s (e.g. 60 s if TSO fires every 3 min, DSO every 1 min).
    n_total_s : float
        Total simulation duration [s].
    tso_period_s : float
        TSO control period [s].  Controllers fire every tso_period_s seconds.
    dso_period_s : float
        DSO control period [s].  DSOs fire more frequently than the TSO.

    Objective weights
    -----------------
    v_setpoint_pu : float
        Voltage setpoint for all monitored buses in all zones.
    g_v : float
        Voltage tracking weight (Q_obj diagonal for V rows).
    g_q : float
        Q-interface tracking weight for DSO controllers.

    OFO tuning
    ----------
    alpha : float
        Step-size gain α (same for all zones in this simplified config).
        Must satisfy α · λ_max(M_sys) < 2 for global stability.
    g_w_der : float
        Regularisation on TSO DER Q changes (prevents large single-step moves).
    g_w_gen : float
        Regularisation on generator AVR setpoint changes (very cautious by default).
    g_w_pcc : float
        Regularisation on PCC Q setpoint changes (Zone 2 → DSO).
    g_w_tso_oltc : float
        Regularisation on machine-transformer OLTC tap changes.
    g_w_dso_der : float
        Regularisation on DSO DER Q changes.

    DSO parameters
    --------------
    dso_g_q : float
        DSO Q-interface tracking weight.
    dso_g_v : float
        DSO voltage tracking weight (secondary objective).

    Stability analysis
    ------------------
    run_stability_analysis : bool
        If True, compute and print the multi-zone stability analysis at t=0.
    sensitivity_update_interval : int
        Recompute cross-sensitivities H_ij every N TSO steps.
        1 = every step (most accurate but slower), 0 = only at initialization.

    Output
    ------
    verbose : int
        0 = silent, 1 = summary, 2 = full diagnostic per step.
    result_dir : str
        Directory for HDF5/JSON result files.  Relative to script location.
    """
    # ── Timing ────────────────────────────────────────────────────────────────
    dt_s:           float = 60.0     # 1-minute simulation timestep
    n_total_s:      float = 60.0 * 60.0  # 60 minutes
    tso_period_s:   float = 60.0 * 3.0  # TSO fires every 3 minutes
    dso_period_s:   float = 60.0 * 1.0  # DSO fires every 1 minute

    # ── Voltage setpoint ──────────────────────────────────────────────────────
    v_setpoint_pu:  float = 1.05     # nominal voltage target for all zones

    # ── Objective weights ─────────────────────────────────────────────────────
    g_v:            float = 50000.0      # TSO voltage tracking weight
    g_q:            float = 1.0      # DSO Q-interface tracking weight
    dso_g_v:        float = 10000.0      # DSO secondary voltage weight (0 = off)

    # ── OFO step-size ─────────────────────────────────────────────────────────
    alpha: Dict[int, float] = field(default_factory=lambda: {1: 0.1, 2: 0.1, 3: 0.1})     # OFO step-size gain per zone
    dso_alpha: float = 0.1              # OFO step-size gain for all DSO controllers
    # ── G_w regularisation weights (TSO) ─────────────────────────────────────
    #
    # With α=1 and Q_DER in [Mvar], the step is clamped by g_w such that
    #   w* ≈ −gradient / (2 · g_w)
    # So g_w=100 limits a single step to ~0.5/g_v Mvar for a 1-pu voltage error.
    g_w_der:        float = 20.0    # [Mvar]² cost per DER Q step
    g_w_gen:        float = 1e7      # [p.u.]² cost per AVR step (very cautious)
    g_w_pcc:        float = 10.0    # [Mvar]² cost per PCC-Q setpoint step
    g_w_tso_oltc:   float = 1.0   # [tap²] cost per machine-trafo OLTC tap step

    # ── G_w regularisation weights (DSO) ─────────────────────────────────────
    g_w_dso_der:    float = 10.0     # DSO DER Q regularisation
    g_w_dso_oltc:   float = 40.0    # DSO OLTC tap regularisation

    # ── Integer switching logic ──────────────────────────────────────────────
    int_max_step:   int = 1         # max tap/shunt change per iteration
    int_cooldown:   int = 1         # iterations to lock after switching

    # ── DSO OLTC initialisation ──────────────────────────────────────────────
    dso_oltc_init_tol_pu: float = 0.01
    """Voltage deadband half-width for the initial DiscreteTapControl run
    that sets coupling-transformer tap positions before OFO starts."""

    # ── Zone partitioning ─────────────────────────────────────────────────────
    use_fixed_zones:    bool  = True   # True = literature 3-area partition; False = spectral clustering

    # ── TSO DER actuators (sgens at PQ load buses) ─────────────────────────
    add_tso_ders: bool = True  # False → controller uses only gen AVR (V_gen)

    # ── Auto-tune g_w based on local curvature ────────────────────────────────
    auto_tune_gw: bool = False  # If True, calculates per-actuator g_w at t=0

    # ── Stability analysis ─────────────────────────────────────────────────────
    run_stability_analysis:       bool = True
    sensitivity_update_interval:  int  = 1E6  # recompute H_ij every N TSO steps

    # ── Output ────────────────────────────────────────────────────────────────
    verbose:        int   = 1
    result_dir:     str   = "results"

    # ── Live plot ─────────────────────────────────────────────────────────────
    live_plot:      bool  = False  # show real-time plot windows during simulation

    # ── Time-series profiles ─────────────────────────────────────────────────
    use_profiles:   bool  = False  # apply time-varying load/sgen profiles
    start_time:     datetime = field(default_factory=lambda: datetime(2016, 6, 10, 0, 0))
    """Simulation start time (for profile lookup).  Only used when use_profiles=True."""

    profiles_csv:   str   = ""
    """Path to profiles CSV.  Empty string → use DEFAULT_PROFILES_CSV."""

    use_zonal_gen_dispatch: bool = True
    """When True and use_profiles=True, pre-compute generator P dispatch
    from zonal residual load (load - sgen) and apply each timestep."""

    # ── Contingencies ────────────────────────────────────────────────────────
    contingencies:  List = field(default_factory=list)
    """List of ContingencyEvent objects to inject during simulation.
    Each event specifies a time, element type, index, and action."""


# =============================================================================
#  Per-step result record
# =============================================================================

@dataclass
class MultiTSOIterationRecord:
    """
    One timestep's worth of simulation data.

    Stores per-zone TSO outputs and plant measurements.
    DSO outputs are stored in the dso_* fields (indexed by dso_id string).
    """
    step:           int
    time_s:         float
    tso_active:     bool = False
    dso_active:     bool = False

    # Per-zone TSO outputs: zone_id → array
    zone_q_der:         Dict[int, NDArray] = field(default_factory=dict)
    zone_q_pcc_set:     Dict[int, NDArray] = field(default_factory=dict)
    zone_v_gen:         Dict[int, NDArray] = field(default_factory=dict)
    zone_q_gen:         Dict[int, Any] = field(default_factory=dict)
    zone_oltc_taps:     Dict[int, NDArray] = field(default_factory=dict)
    zone_tso_objective: Dict[int, Optional[float]] = field(default_factory=dict)
    zone_tso_status:    Dict[int, Optional[str]]  = field(default_factory=dict)
    zone_tso_solve_s:   Dict[int, Optional[float]] = field(default_factory=dict)

    # Plant voltages per zone (after PF)
    zone_v_min:  Dict[int, float] = field(default_factory=dict)
    zone_v_max:  Dict[int, float] = field(default_factory=dict)
    zone_v_mean: Dict[int, float] = field(default_factory=dict)

    # Per-zone stability diagnostic from coordinator
    zone_contraction_lhs: Dict[int, float] = field(default_factory=dict)

    # DSO outputs
    dso_q_der:         Dict[str, NDArray] = field(default_factory=dict)
    dso_q_actual_mvar: Dict[str, Optional[float]] = field(default_factory=dict)
    dso_q_set_mvar:    Dict[str, Optional[float]] = field(default_factory=dict)
    dso_objective:     Dict[str, Optional[float]] = field(default_factory=dict)
    dso_status:        Dict[str, Optional[str]]   = field(default_factory=dict)

    # DSO network-group aggregates
    dso_group_q_der_mvar: Dict[str, float] = field(default_factory=dict)
    dso_group_v_min_pu:   Dict[str, float] = field(default_factory=dict)
    dso_group_v_mean_pu:  Dict[str, float] = field(default_factory=dict)
    dso_group_v_max_pu:   Dict[str, float] = field(default_factory=dict)

    # Transformer-level DSO interface and OLTC data
    dso_trafo_q_set_mvar:    Dict[str, float] = field(default_factory=dict)
    dso_trafo_q_actual_mvar: Dict[str, float] = field(default_factory=dict)
    dso_trafo_tap_pos:       Dict[str, int] = field(default_factory=dict)

    # Explicit grouping metadata for fail-fast plotting
    dso_trafo_group:      Dict[str, str] = field(default_factory=dict)
    dso_controller_group: Dict[str, str] = field(default_factory=dict)

    # Global multi-zone stability snapshot (computed periodically)
    stability_result: Optional[MultiZoneStabilityResult] = None


# =============================================================================
#  Measurement helpers (custom for IEEE 39-bus multi-zone)
# =============================================================================

def _measure_for_zone_tso(
    net: pp.pandapowerNet,
    zone_def: ZoneDefinition,
    it: int,
) -> Measurement:
    """
    Build a Measurement object for one TSO zone from the combined plant network.

    This function reads all relevant quantities from the ALREADY CONVERGED
    pandapower result tables (net.res_bus, net.res_line, etc.).

    Column-to-row correspondence (must match TSOController's u vector ordering):
        bus_indices:                all network buses (TSOController needs full bus array)
        voltage_magnitudes_pu:      all bus voltages [p.u.]
        branch_indices:             zone's line indices (for current constraints)
        current_magnitudes_ka:      line i_from_ka [kA]
        interface_transformer_idx:  PCC 2W trafo indices (Zone 2 only)
        interface_q_hv_side_mvar:   Q at PCC HV bus (2W: net.res_trafo q_hv_mvar)
        der_indices:                TSO DER sgen indices in this zone
        der_q_mvar:                 DER Q output [Mvar]
        der_p_mw:                   DER P output [MW]
        oltc_indices:               [] (no machine OLTCs in IEEE 39-bus)
        oltc_tap_positions:         []
        shunt_indices:              zone shunt bus indices
        shunt_states:               shunt step values
        gen_indices:                zone generator indices
        gen_vm_pu:                  generator AVR setpoint (gen.vm_pu in pandapower)
        gen_p_mw:                   generator active power output
        gen_q_mvar:                 generator reactive power output

    Note: bus_indices and voltage_magnitudes_pu cover ALL buses (not just the zone)
    because TSOController.step() indexes into measurement.voltage_magnitudes_pu
    using bus_indices to extract the zone-specific voltages.

    Parameters
    ----------
    net : pandapowerNet
        The combined plant network after a converged power flow.
    zone_def : ZoneDefinition
        Index sets for this zone.
    it : int
        Current iteration (step) number.

    Returns
    -------
    Measurement
    """
    # ── All bus voltages (TSOController looks up by bus index) ────────────────
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm_all  = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # ── Line currents in this zone ─────────────────────────────────────────────
    i_ka = np.array(
        [float(net.res_line.at[li, "i_from_ka"]) for li in zone_def.line_indices],
        dtype=np.float64,
    ) if zone_def.line_indices else np.array([], dtype=np.float64)

    # ── PCC interface Q (3W trafo, HV side, load convention) ─────────────────
    #
    # For 3-winding trafos: net.res_trafo3w.q_hv_mvar gives Q at the HV port
    # with LOAD CONVENTION (positive = power flows INTO the HV bus from the
    # external grid side).
    #
    # TSOController expects "interface_q_hv_side_mvar" in the same convention.
    q_iface = np.zeros(len(zone_def.pcc_trafo_indices), dtype=np.float64)
    for k, t in enumerate(zone_def.pcc_trafo_indices):
        if t in net.res_trafo3w.index:
            q_iface[k] = float(net.res_trafo3w.at[t, "q_hv_mvar"])

    # ── TSO DER sgens ─────────────────────────────────────────────────────────
    der_q = np.array(
        [float(net.res_sgen.at[s, "q_mvar"]) for s in zone_def.tso_der_indices],
        dtype=np.float64,
    ) if zone_def.tso_der_indices else np.array([], dtype=np.float64)
    der_p = np.array(
        [float(net.res_sgen.at[s, "p_mw"]) for s in zone_def.tso_der_indices],
        dtype=np.float64,
    ) if zone_def.tso_der_indices else np.array([], dtype=np.float64)

    # ── Shunts ────────────────────────────────────────────────────────────────
    shunt_states = np.zeros(len(zone_def.shunt_bus_indices), dtype=np.int64)
    for k, sb in enumerate(zone_def.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            shunt_states[k] = int(net.shunt.at[net.shunt.index[mask][0], "step"])

    # ── Generators ────────────────────────────────────────────────────────────
    gen_vm = np.array(
        [float(net.gen.at[g, "vm_pu"]) for g in zone_def.gen_indices],
        dtype=np.float64,
    ) if zone_def.gen_indices else np.array([], dtype=np.float64)
    gen_p = np.array(
        [float(net.res_gen.at[g, "p_mw"]) for g in zone_def.gen_indices],
        dtype=np.float64,
    ) if zone_def.gen_indices else np.array([], dtype=np.float64)
    gen_q = np.array(
        [float(net.res_gen.at[g, "q_mvar"]) for g in zone_def.gen_indices],
        dtype=np.float64,
    ) if zone_def.gen_indices else np.array([], dtype=np.float64)

    # ── OLTC taps (Machine transformers) ──────────────────────────────────────
    oltc_taps = np.array(
        [int(net.trafo.at[t, "tap_pos"]) for t in zone_def.oltc_trafo_indices],
        dtype=np.int64,
    ) if zone_def.oltc_trafo_indices else np.array([], dtype=np.int64)

    return Measurement(
        iteration=it,
        bus_indices=all_bus,
        voltage_magnitudes_pu=vm_all,
        branch_indices=np.array(zone_def.line_indices, dtype=np.int64),
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=np.array(zone_def.pcc_trafo_indices, dtype=np.int64),
        interface_q_hv_side_mvar=q_iface,
        der_indices=np.array(zone_def.tso_der_indices, dtype=np.int64),
        der_q_mvar=der_q,
        der_p_mw=der_p,
        oltc_indices=np.array(zone_def.oltc_trafo_indices, dtype=np.int64),
        oltc_tap_positions=oltc_taps,
        shunt_indices=np.array(zone_def.shunt_bus_indices, dtype=np.int64),
        shunt_states=shunt_states,
        gen_indices=np.array(zone_def.gen_indices, dtype=np.int64),
        gen_vm_pu=gen_vm,
        gen_p_mw=gen_p,
        gen_q_mvar=gen_q,
    )


def _measure_for_dso(
    net: pp.pandapowerNet,
    dso_cfg: DSOControllerConfig,
    it: int,
) -> Measurement:
    """
    Build a Measurement for one DSO HV sub-network from the plant network.

    Reads interface Q from all coupling transformers (3-winding, 345/110/20 kV),
    voltages at all HV buses, and currents on all HV lines.

    Parameters
    ----------
    net : pandapowerNet
    dso_cfg : DSOControllerConfig
        The DSO controller's configuration (der_indices, interface trafos, etc.).
    it : int
        Current step number.

    Returns
    -------
    Measurement
    """
    # All bus voltages (DSOController looks up by bus index)
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm_all  = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Line currents (HV lines within the sub-network)
    i_ka = np.array(
        [float(net.res_line.at[li, "i_from_ka"]) for li in dso_cfg.current_line_indices],
        dtype=np.float64,
    ) if dso_cfg.current_line_indices else np.array([], dtype=np.float64)

    # Interface Q at all coupling trafos (HV side, load convention).
    # 3W trafos: q_hv_mvar is in res_trafo3w.
    q_iface = np.array(
        [float(net.res_trafo3w.at[t, "q_hv_mvar"]) for t in dso_cfg.interface_trafo_indices],
        dtype=np.float64,
    )

    # DSO DER sgens
    der_q = np.array(
        [float(net.res_sgen.at[s, "q_mvar"]) for s in dso_cfg.der_indices],
        dtype=np.float64,
    )
    der_p = np.array(
        [float(net.res_sgen.at[s, "p_mw"]) for s in dso_cfg.der_indices],
        dtype=np.float64,
    )

    # OLTC taps (3W coupling trafos)
    oltc_taps = np.array(
        [int(net.trafo3w.at[t, "tap_pos"]) for t in dso_cfg.oltc_trafo_indices],
        dtype=np.int64,
    ) if dso_cfg.oltc_trafo_indices else np.array([], dtype=np.int64)

    # Shunt states (empty for HV sub-networks)
    shunt_states = np.zeros(len(dso_cfg.shunt_bus_indices), dtype=np.int64)

    return Measurement(
        iteration=it,
        bus_indices=all_bus,
        voltage_magnitudes_pu=vm_all,
        branch_indices=np.array(dso_cfg.current_line_indices, dtype=np.int64),
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=np.array(dso_cfg.interface_trafo_indices, dtype=np.int64),
        interface_q_hv_side_mvar=q_iface,
        der_indices=np.array(dso_cfg.der_indices, dtype=np.int64),
        der_q_mvar=der_q,
        der_p_mw=der_p,
        oltc_indices=np.array(dso_cfg.oltc_trafo_indices, dtype=np.int64),
        oltc_tap_positions=oltc_taps,
        shunt_indices=np.array(dso_cfg.shunt_bus_indices, dtype=np.int64),
        shunt_states=shunt_states,
        gen_indices=np.array([], dtype=np.int64),
        gen_vm_pu=np.array([], dtype=np.float64),
    )


def _record_dso_group_and_transformer_data(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    dso_ids: List[str],
    dsocontrollers: Dict[str, DSOController],
    dso_group_map: Dict[str, str],
    last_dso_q_set_mvar: Dict[str, Optional[NDArray]],
) -> None:
    """
    Write DSO transformer- and network-group-level observables into rec.

    Each DSO may have multiple interface transformers (3 per HV sub-network).
    Per-trafo data is keyed by ``"{dso_id}|trafo_{trafo_idx}"``.
    """
    group_q_der: Dict[str, List[float]] = {}
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
            if trafo_idx in net.trafo3w.index:
                rec.dso_trafo_tap_pos[trafo_key] = int(
                    net.trafo3w.at[trafo_idx, "tap_pos"]
                )

        # DER and voltage group data
        q_der_total = float(net.res_sgen.loc[dsocfg.der_indices, "q_mvar"].sum())
        vm_pu = net.res_bus.loc[dsocfg.voltage_bus_indices, "vm_pu"].to_numpy(dtype=float)

        group_q_der.setdefault(group_id, []).append(q_der_total)
        if vm_pu.size > 0:
            group_v_min.setdefault(group_id, []).append(float(np.min(vm_pu)))
            group_v_mean.setdefault(group_id, []).append(float(np.mean(vm_pu)))
            group_v_max.setdefault(group_id, []).append(float(np.max(vm_pu)))

    for group_id, values in group_q_der.items():
        rec.dso_group_q_der_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_v_min.items():
        rec.dso_group_v_min_pu[group_id] = float(np.min(values))
    for group_id, values in group_v_mean.items():
        rec.dso_group_v_mean_pu[group_id] = float(np.mean(values))
    for group_id, values in group_v_max.items():
        rec.dso_group_v_max_pu[group_id] = float(np.max(values))

# =============================================================================
#  Apply controls to plant network
# =============================================================================

def _apply_zone_tso_controls(
    net: pp.pandapowerNet,
    zone_def: ZoneDefinition,
    tso_out,
) -> None:
    """
    Write TSO control output for one zone back to the pandapower plant network.

    Control variable ordering in u (must match TSOControllerConfig / TSOController):
        u = [ Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt ]

    For IEEE 39-bus: s_OLTC and s_shunt are absent (lists are empty).

    Parameters
    ----------
    net : pandapowerNet
        Combined plant network (modified IN-PLACE).
    zone_def : ZoneDefinition
        Provides index lists in the same order as u.
    tso_out : ControllerOutput
        Output of TSOController.step().  tso_out.u_new is the new setpoint vector.
    """
    u = tso_out.u_new
    n_der = len(zone_def.tso_der_indices)
    n_pcc = len(zone_def.pcc_trafo_indices)
    n_gen = len(zone_def.gen_indices)
    off = 0

    # ── DER Q setpoints ───────────────────────────────────────────────────────
    for k, s_idx in enumerate(zone_def.tso_der_indices):
        net.sgen.at[s_idx, "q_mvar"] = float(u[off + k])
    off += n_der

    # ── PCC Q setpoints (stored but not applied here — DSO is responsible) ────
    # The PCC Q setpoints are communicated to the DSO controller via the
    # generate_setpoint_messages() mechanism of TSOController.
    off += n_pcc

    # ── Generator AVR voltage setpoints ───────────────────────────────────────
    for k, g_idx in enumerate(zone_def.gen_indices):
        new_vm = float(u[off + k])
        net.gen.at[g_idx, "vm_pu"] = new_vm
    off += n_gen

    # ── OLTC tap positions (Machine transformers) ──────────────────────────────
    n_oltc = len(zone_def.oltc_trafo_indices)
    for k, t_idx in enumerate(zone_def.oltc_trafo_indices):
        net.trafo.at[t_idx, "tap_pos"] = int(round(u[off + k]))
    off += n_oltc

    # shunt application omitted (not used in base IEEE 39-bus setup)


def _apply_dso_controls(
    net: pp.pandapowerNet,
    dso_cfg: DSOControllerConfig,
    dso_out,
) -> None:
    """
    Write DSO control output to the pandapower plant network.

    DSO u = [ Q_DER | s_OLTC | s_shunt ].

    Parameters
    ----------
    net : pandapowerNet
    dso_cfg : DSOControllerConfig
    dso_out : ControllerOutput
    """
    u = dso_out.u_new
    n_der  = len(dso_cfg.der_indices)
    n_oltc = len(dso_cfg.oltc_trafo_indices)
    off = 0

    # DSO DER Q setpoints
    for k, s_idx in enumerate(dso_cfg.der_indices):
        net.sgen.at[s_idx, "q_mvar"] = float(u[off + k])
    off += n_der

    # DSO OLTC tap positions (3W trafo)
    for k, t_idx in enumerate(dso_cfg.oltc_trafo_indices):
        net.trafo3w.at[t_idx, "tap_pos"] = int(round(u[off + k]))
    off += n_oltc

    # Shunt switching (skipped in base config — shunts initialised separately)


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
        print(f"  V_set = {v_set:.3f} p.u.  |  N_zones = 3  |  alpha = {config.alpha}")
        print(f"  Zone partition: {zone_method}  |  5 HV sub-networks (DSO_1..DSO_5)")
        print("=" * 72)

    # =========================================================================
    # STEP 1: Build the base IEEE 39-bus network (no DSO feeders yet)
    # =========================================================================
    if verbose >= 1:
        print("[1] Building IEEE 39-bus network ...")

    net, meta = build_ieee39_net(
        ext_grid_vm_pu=v_set,
        add_der_at_gen_buses=config.add_tso_ders,
    )

    # Remove generators
    #meta = remove_generators(net, meta, gen_indices_to_remove=[2])

    #pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # =========================================================================
    # STEP 2: Zone partitioning
    # =========================================================================
    if config.use_fixed_zones:
        if verbose >= 1:
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
    # STEP 3: Attach 5 HV sub-networks (110 kV, TUDA topology)
    # =========================================================================
    if verbose >= 1:
        print("[3] Attaching 5 HV sub-networks (DSO_1..DSO_5) ...")

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
    zone_oltc_trafos: Dict[int, List[int]] = {z: [] for z in zone_map}
    for t_idx, g_idx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        # Machine trafo's grid bus = hv_bus of the 2W transformer
        grid_bus = int(net.trafo.at[t_idx, "hv_bus"])
        for z, buses in zone_map.items():
            if grid_bus in set(buses):
                zone_oltc_trafos[z].append(t_idx)
                break

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
            alpha=config.alpha[z],
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

    # =========================================================================
    # STEP 5: Initialise TSOControllers (one per zone)
    # =========================================================================
    if verbose >= 1:
        print("[5] Initialising TSOControllers ...")

    ns0 = _network_state(net)  # initial network state snapshot

    tso_controllers: Dict[int, TSOController] = {}
    for z, zd in zone_defs.items():

        # ── G_w diagonal for this zone's u vector ────────────────────────────
        gw_diag = zd.gw_diagonal()
        gz_diag = np.concatenate([
            np.full(len(zd.v_bus_indices), 0.0),   # voltage constraint slacks (g_z=0)
            np.full(len(zd.line_indices),  0.0),    # current constraint slacks
        ])
        # TSOController expects g_z as a flat array (one entry per output variable)
        # A value of 0 means "no slack variable", i.e. hard constraint.
        # For soft constraints set g_z > 0.

        ofo_params = OFOParameters(
            alpha=zd.alpha,
            g_w=gw_diag,    # 1-D vector → no OLTC cross-coupling
            g_z=gz_diag,
            g_u=np.zeros_like(gw_diag),  # no level penalty for now
            int_max_step=config.int_max_step,
            int_cooldown=config.int_cooldown,
        )

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
        # case39 generators may have NaN for sn_mva; use 1.2 * p_mw as fallback.
        gen_params = []
        for g in zd.gen_indices:
            p_mw = float(net.gen.at[g, "p_mw"])
            sn = net.gen.at[g, "sn_mva"]
            if pd.isna(sn) or sn <= 0:
                sn = p_mw * 1.2
            gen_params.append(
                GeneratorParameters(
                    s_rated_mva=float(sn),
                    p_max_mw=p_mw,
                    xd_pu=1.0,
                    i_f_max_pu=2.0,
                    beta=0.1,
                    q0_pu=0.1,
                )
            )

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
        )

        ctrl = TSOController(
            controller_id=f"tso_zone_{z}",
            params=ofo_params,
            config=tso_cfg,
            network_state=ns0,
            actuator_bounds=bounds,
            sensitivities=JacobianSensitivities(net),
        )
        # Initialise u_current from the converged PF
        meas_init = _measure_for_zone_tso(net, zd, 0)
        ctrl.initialise(meas_init)
        tso_controllers[z] = ctrl


    # =========================================================================
    # STEP 6: Initialise DSO controllers (one per HV sub-network, all zones)
    # =========================================================================
    if verbose >= 1:
        print("[6] Initialising DSO controllers (5 HV sub-networks) ...")

    dso_controllers: Dict[str, DSOController] = {}

    for hv in meta.hv_networks:
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
            g_qi=0.0,
            lambda_qi=0.0,
            q_integral_max_mvar=1.0,
            v_setpoints_pu=np.full(len(v_buses), v_set),
            g_v=config.dso_g_v,
        )

        dso_s_rated = np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in der_indices],
            dtype=np.float64,
        )
        dso_p_max = np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in der_indices],
            dtype=np.float64,
        )

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
        )

        n_iface = len(interface_trafos)
        n_v = len(v_buses)
        n_i = len(hv_lines)
        dso_ofo = OFOParameters(
            alpha=config.dso_alpha,
            g_w=dso_gw_diag,
            g_z=np.zeros(n_iface + n_v + n_i),
            g_u=np.zeros_like(dso_gw_diag),
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
        meas_dso_init = _measure_for_dso(net, dso_cfg, 0)
        dso_ctrl.initialise(meas_dso_init)
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
        print("[7] Initialising MultiTSOCoordinator ...")

    coordinator = MultiTSOCoordinator(
        zones=list(zone_defs.values()),
        net=net,
        verbose=verbose,
    )
    for z, ctrl in tso_controllers.items():
        coordinator.register_tso_controller(z, ctrl)

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
            print(f"[7b] Loading profiles from {profiles_csv}")
            print(f"     start_time = {start_time:%d.%m.%Y %H:%M}")

        profiles = load_profiles(profiles_csv, timestep_s=config.dt_s)
        snapshot_base_values(net)

        # Clip profile DataFrame to the simulation window only.
        # Without this, compute_zonal_gen_dispatch iterates the full profile
        # horizon (up to 525 600 rows at 60 s resolution) unnecessarily.
        t_end = start_time + timedelta(seconds=config.n_total_s)
        profiles = profiles.loc[:t_end]  # keep only rows up to simulation end

        # Apply initial profiles
        apply_profiles(net, profiles, start_time)

        if config.use_zonal_gen_dispatch:
            gen_dispatch = compute_zonal_gen_dispatch(
                net, profiles, zone_map,
            )
            apply_gen_dispatch(net, gen_dispatch, start_time)

        # Re-converge after profile application
        pp.runpp(net, max_iteration=50, run_control=False, calculate_voltage_angles=True, init='auto')

    # ── Initialise 2W and 3W OLTC tap positions via DiscreteTapControl ────────────
    # Run AFTER profiles/dispatch so taps are found for the actual operating point.
    from pandapower.control import DiscreteTapControl

    tol_pu = config.dso_oltc_init_tol_pu
    for hv in meta.hv_networks:
        for t3w in hv.coupling_trafo_indices:
            DiscreteTapControl(
                net,
                element_index=t3w,
                vm_lower_pu=v_set - tol_pu,
                vm_upper_pu=v_set + tol_pu,
                side="mv",
                element="trafo3w",
            )
    if meta.machine_trafo_indices:
        mt_tol_pu = config.dso_oltc_init_tol_pu
        for tidx in meta.machine_trafo_indices:
            DiscreteTapControl(
                net,
                element_index=tidx,
                vm_lower_pu=v_set - mt_tol_pu,
                vm_upper_pu=v_set + mt_tol_pu,
                side="hv",
                element="trafo",
            )
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if verbose >= 2:
        print("[7c] OLTC tap initialisation (DiscreteTapControl):")
        for hv in meta.hv_networks:
            for t3w in hv.coupling_trafo_indices:
                tap = int(net.trafo3w.at[t3w, "tap_pos"])
                mv_bus = int(net.trafo3w.at[t3w, "mv_bus"])
                vm = float(net.res_bus.at[mv_bus, "vm_pu"])
                print(f"  {hv.net_id} trafo3w {t3w}: tap_pos={tap:+d}, "
                      f"V_mv={vm:.4f} p.u.")
        print("  Machine transformer tap initialisation:")
        for tidx, gidx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
            tap = int(net.trafo.at[tidx, "tap_pos"])
            hv_bus = int(net.trafo.at[tidx, "hv_bus"])
            vm = float(net.res_bus.at[hv_bus, "vm_pu"])
            print(f"    trafo {tidx} (gen {gidx}): tap_pos={tap:+d}, "
                  f"V_hv={vm:.4f} p.u.")
    # Remove pandapower controllers so they don't interfere with OFO
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Re-converge with found tap positions (no control)
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Re-initialise all controllers so _u_current reflects the updated
    # operating point (profiles + correct tap positions).
    for z, ctrl in tso_controllers.items():
        ctrl.initialise(_measure_for_zone_tso(net, zone_defs[z], 0))
    for dso_id, dso_ctrl in dso_controllers.items():
        dso_ctrl.initialise(_measure_for_dso(net, dso_ctrl.config, 0))

    # ── Optional: Auto-tune g_w based on local curvature ──────────────────────
    if config.auto_tune_gw:
        if verbose >= 1:
            print("[7.1] Auto-tuning per-actuator g_w weights (preconditioning) ...")
        
        # 1. Gather required data for tuning
        zone_ids_sorted = sorted(zone_defs.keys())
        
        # Ensure TSO controllers have fresh sensitivities from the current PF
        H_blocks_tune = {}
        for z in zone_ids_sorted:
            ctrl = tso_controllers[z]
            ctrl.invalidate_sensitivity_cache() # Force recompute for current state
            H_blocks_tune[(z, z)] = ctrl._build_sensitivity_matrix()
            
        Q_obj_list = [zone_defs[z].q_obj_diagonal() for z in zone_ids_sorted]
        
        actuator_counts = [
            {
                'n_der':  len(zone_defs[z].tso_der_indices),
                'n_pcc':  len(zone_defs[z].pcc_trafo_indices),
                'n_gen':  len(zone_defs[z].gen_indices),
                'n_oltc': len(zone_defs[z].oltc_trafo_indices),
            }
            for z in zone_ids_sorted
        ]

        # 2. Compute vectors
        # Create a temporary CascadeConfig for the tuner since it expects one
        from core.cascade_config import CascadeConfig
        dummy_cfg = CascadeConfig(alpha=max(config.alpha.values()), 
                                 gw_tso_v_gen=config.g_w_gen)
        
        gw_vectors = compute_optimal_gw(
            dummy_cfg, H_blocks_tune, Q_obj_list, actuator_counts,
            safety_factor=2, min_gw=0.01, min_gw_discrete=40
        )

        # 3. Apply overrides
        import dataclasses
        for idx, z in enumerate(zone_ids_sorted):
            gw_vec = gw_vectors[idx]
            tso_controllers[z].params = dataclasses.replace(tso_controllers[z].params, g_w=gw_vec)
            
            if verbose >= 1:
                zd = zone_defs[z]
                print(f"\n  Zone {z} Tuned Weights (g_w):")
                print(f"    {'Type':<8s} {'Name':<20s} {'Bus':>5s} {'Weight':>12s}")
                print(f"    {'-'*48}")
                
                off = 0
                # DERs
                for k, s_idx in enumerate(zd.tso_der_indices):
                    name = net.sgen.at[s_idx, 'name'] or f"SGen_{s_idx}"
                    bus = net.sgen.at[s_idx, 'bus']
                    print(f"    {'Q_DER':<8s} {str(name):<20.20s} {int(bus):>5d} {gw_vec[off+k]:>12.4f}")
                off += len(zd.tso_der_indices)
                
                # PCCs
                for k, t_idx in enumerate(zd.pcc_trafo_indices):
                    # Check trafo or trafo3w
                    if t_idx in net.trafo.index:
                        name = net.trafo.at[t_idx, 'name'] or f"T2W_{t_idx}"
                        bus = net.trafo.at[t_idx, 'hv_bus']
                    else:
                        name = net.trafo3w.at[t_idx, 'name'] or f"T3W_{t_idx}"
                        bus = net.trafo3w.at[t_idx, 'hv_bus']
                    print(f"    {'Q_PCC':<8s} {str(name):<20.20s} {int(bus):>5d} {gw_vec[off+k]:>12.4f}")
                off += len(zd.pcc_trafo_indices)
                
                # V_gen
                for k, g_idx in enumerate(zd.gen_indices):
                    name = net.gen.at[g_idx, 'name'] or f"Gen_{g_idx}"
                    bus = net.gen.at[g_idx, 'bus']
                    print(f"    {'V_gen':<8s} {str(name):<20.20s} {int(bus):>5d} {gw_vec[off+k]:>12.4e}")
                off += len(zd.gen_indices)
                
                # OLTCs
                for k, t_idx in enumerate(zd.oltc_trafo_indices):
                    name = net.trafo.at[t_idx, 'name'] or f"T2W_{t_idx}"
                    bus = net.trafo.at[t_idx, 'hv_bus']
                    print(f"    {'OLTC':<8s} {str(name):<20.20s} {int(bus):>5d} {gw_vec[off+k]:>12.4f}")
                off += len(zd.oltc_trafo_indices)
                print()

    # ── Cross-sensitivity & stability analysis at the final operating point ──
    # Recompute H_ij and M_ij from the current Jacobian (after profiles + taps).
    coordinator.compute_cross_sensitivities()
    coordinator.compute_M_blocks()
    contraction_info = coordinator.check_contraction()

    if config.run_stability_analysis:
        if verbose >= 1:
            print("[7d] Running multi-zone stability analysis at t=0 ...")
        zone_ids_sorted = sorted(zone_defs.keys())
        H_blocks_stab = {k: coordinator.get_H_block(*k)
                         for k in coordinator._H_blocks}
        Q_obj_list = [zone_defs[z].q_obj_diagonal() for z in zone_ids_sorted]
        
        # USE THE ACTUAL WEIGHTS FROM CONTROLLERS (may be tuned)
        G_w_list = [tso_controllers[z].params.g_w for z in zone_ids_sorted]
        alpha_list  = [zone_defs[z].alpha            for z in zone_ids_sorted]

        actuator_counts = [
            {
                'n_der':  len(zone_defs[z].tso_der_indices),
                'n_pcc':  len(zone_defs[z].pcc_trafo_indices),
                'n_gen':  len(zone_defs[z].gen_indices),
                'n_oltc': len(zone_defs[z].oltc_trafo_indices),
            }
            for z in zone_ids_sorted
        ]
        stab_result = analyse_multi_zone_stability(
            H_blocks=H_blocks_stab,
            Q_obj_list=Q_obj_list,
            G_w_list=G_w_list,
            alpha_list=alpha_list,
            zone_ids=zone_ids_sorted,
            zone_names=[f"Zone {z}" for z in zone_ids_sorted],
            actuator_counts=actuator_counts,
            verbose=True,
        )
    else:
        stab_result = None

    if contingencies and verbose >= 1:
        print(f"  Scheduled contingencies ({len(contingencies)}):")
        for ev in contingencies:
            t_label = f"t={ev.effective_time_s:.0f}s" if ev.time_s is not None else f"min {ev.minute}"
            print(f"    {t_label}: {ev.action} {ev.element_type}[{ev.element_index}]")


    # =========================================================================
    # STEP 8: Main simulation loop
    # =========================================================================
    if verbose >= 1:
        n_steps = int(config.n_total_s / config.dt_s)
        dur_str = f"start={start_time:%d.%m.%Y %H:%M}  " if use_profiles else ""
        print(f"[8] Starting simulation: {n_steps} steps  "
              f"({dur_str}dt={config.dt_s:.0f}s, TSO/{config.tso_period_s/60:.0f}min, "
              f"DSO/{config.dso_period_s/60:.0f}min)")
        print()

    log: List[MultiTSOIterationRecord] = []

    # ── Optionally create live plot windows ───────────────────────────────────
    _live_plotter = None
    if config.live_plot:
        from visualisation.plot_multi_tso import MultiTSOLivePlotter
        _live_plotter = MultiTSOLivePlotter(
            zone_ids=sorted(zone_defs.keys()),
            dso_ids=dso_ids,
            v_setpoint_pu=config.v_setpoint_pu,
            v_min_pu=0.95,
            v_max_pu=1.05,
            sub_minute=False,
            update_every=1,
            tso_update_every=1,
        )

    def _is_period_hit(time_s: float, period_s: float) -> bool:
        """True if time_s is a multiple of period_s (within 1 s tolerance)."""
        rem = time_s % period_s
        return rem < 1.0 or abs(rem - period_s) < 1.0

    tso_step_count = 0  # count TSO steps for sensitivity refresh logic

    n_steps = int(config.n_total_s / config.dt_s)
    for step in range(1, n_steps + 1):
        time_s  = step * config.dt_s
        run_tso = (step == 1) or _is_period_hit(time_s, config.tso_period_s)
        run_dso = _is_period_hit(time_s, config.dso_period_s)

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
            pp.runpp(net, run_control=False, calculate_voltage_angles=True)

        # ── Apply contingency events ──────────────────────────────────────────
        if contingencies:
            fired = [
                ev for ev in contingencies
                if abs(ev.effective_time_s - time_s) < 1e-9
            ]
            if fired:
                for ev in fired:
                    _apply_contingency(net, ev, verbose)

                # Re-converge PF with new topology so Jacobian is valid
                pp.runpp(net, run_control=False, calculate_voltage_angles=True)

                # Refresh sensitivity matrices for all TSO controllers
                for z_id, ctrl in tso_controllers.items():
                    ctrl.sensitivities = JacobianSensitivities(net)
                    ctrl.invalidate_sensitivity_cache()
                for dso_ctrl in dso_controllers.values():
                    dso_ctrl.sensitivities = JacobianSensitivities(net)
                    dso_ctrl.invalidate_sensitivity_cache()

        # ── TSO step ──────────────────────────────────────────────────────────
        if run_tso:
            tso_step_count += 1
            # Decide whether to refresh cross-sensitivities this step
            refresh_H = (config.sensitivity_update_interval > 0
                         and tso_step_count % config.sensitivity_update_interval == 0)

            # Build per-zone measurements from plant network
            measurements: Dict[int, Measurement] = {
                z: _measure_for_zone_tso(net, zd, step)
                for z, zd in zone_defs.items()
            }

            # Run decentralised TSO step for all zones
            tso_outputs = coordinator.step(
                measurements,
                step,
                recompute_cross_sensitivities=refresh_H,
            )

            # Apply TSO controls to plant network
            for z, tso_out in tso_outputs.items():
                _apply_zone_tso_controls(net, zone_defs[z], tso_out)

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
        if run_dso:
            for dso_id, dso_ctrl in dso_controllers.items():
                # meas_dso reflects the current operating point BEFORE this DSO step.
                # This is the correct basis for the capability message: it tells the TSO
                # what the DSO can still do from its present dispatch, not what it just did.
                meas_dso = _measure_for_dso(net, dso_ctrl.config, step)

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
                _apply_dso_controls(net, dso_ctrl.config, dso_out)

        # ── Power flow ────────────────────────────────────────────────────────
        try:
            pp.runpp(net, run_control=False, calculate_voltage_angles=True)
        except Exception as e:
            print(f"  [Step {step}] Power flow failed: {e}")
            log.append(rec)
            continue

        # ── Record post-PF observables (require converged res_* tables) ──────
        if run_dso:
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
            )

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

            # Generator Q from converged power flow
            if run_tso and zd.gen_indices:
                rec.zone_q_gen[z] = np.array(
                    [net.res_gen.at[idx, "q_mvar"] for idx in zd.gen_indices],
                    dtype=np.float64,
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

        if _live_plotter is not None:
            _live_plotter.update(rec)

        log.append(rec)

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

    return log


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
        n_total_s=60.0 * 720,      # 720-minute simulation
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=20.0 * 1,    # DSO every 30 seconds
        alpha={1: 0.01, 2: 0.01, 3: 0.01},
        dso_alpha=0.1,
        g_v=150000.0,
        g_q=5,
        dso_g_v=1000.0,
        g_w_der=10.0,
        g_w_gen=1e4,
        g_w_pcc=1.0,
        g_w_tso_oltc=50,
        g_w_dso_der = 2.0,     # DSO DER Q regularisation
        g_w_dso_oltc = 10.0,    # DSO OLTC tap regularisation
        use_fixed_zones=True,      # literature 3-area partition (not spectral)
        run_stability_analysis=True,
        sensitivity_update_interval=1E6,  # refresh H_ij every N TSO steps
        verbose=1,
        live_plot=True,
        add_tso_ders=True,
        # ── Profile & contingency settings ───────────────────────────────
        start_time=datetime(2016, 1, 5, 8, 0),
        use_profiles=True,
        use_zonal_gen_dispatch=True,
        contingencies=[
            # Example: trip line 0 at t=30 min, restore at t=60 min
            ContingencyEvent(minute=90, element_type="gen", element_index=3, action="trip"),
            ContingencyEvent(minute=120, element_type="gen", element_index=3, action="restore"),
            ContingencyEvent(minute=180, element_type="gen", element_index=2, action="trip"),
            ContingencyEvent(minute=240, element_type="gen", element_index=2, action="restore"),
        ],
    )
    log = run_multi_tso_dso(cfg)
    print(f"\nSimulation complete. {len(log)} steps recorded.")


if __name__ == "__main__":
    main()
