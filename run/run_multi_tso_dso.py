#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/run_multi_tso_dso.py
========================
Multi-TSO / Multi-DSO OFO simulation loop on the IEEE 39-bus network.

This script is the multi-zone analogue of ``run/run_cascade.py``.  It uses
the same OFO controller infrastructure (TSOController, DSOController) but
orchestrates N=3 independent TSO zones via the MultiTSOCoordinator, and adds
DSO feeders at Zone-2 load buses.

Architecture (matches the multi-TSO theory in Schwenke / CIGRE 2026)
---------------------------------------------------------------------

    ┌──────────────────────────────────────────────────────────┐
    │              IEEE 39-bus network (plant)                 │
    │  Zone 0        │  Zone 1        │  Zone 2 (w/ DSOs)      │
    │  TSOCtrl_0     │  TSOCtrl_1     │  TSOCtrl_2             │
    │                │                │  ├── DSOCtrl_20a        │
    │                │                │  └── DSOCtrl_20b        │
    └──────────────────────────────────────────────────────────┘

Step sequence (each simulation step dt_s):
    1.  Apply time-series profiles to plant network.
    2.  Run power flow on plant network.
    3.  If TSO step: call coordinator.step(measurements_per_zone).
        * Each TSOController.step() solves its local MIQP independently.
        * Coordinator optionally recomputes H_ij and checks contraction.
    4.  If DSO step: call DSOController.step() for each Zone-2 DSO.
    5.  Apply all new setpoints to plant network.
    6.  Run power flow, record results.

Sensitivity matrices
--------------------
* H_ii (local, zone i): computed by TSOController._build_sensitivity_matrix()
  using generator terminal buses + DER sgens as column inputs and zone buses
  as row outputs.
* H_ij (cross-zone, i≠j): computed by MultiTSOCoordinator.compute_cross_sensitivities()
  using zone_j's inputs and zone_i's observed outputs.
* H_DSO (per DSO feeder): computed by DSOController._build_sensitivity_matrix()
  using DSO DER Q and OLTC as inputs, 2W-trafo Q + LV voltages as outputs.

Network state caching
---------------------
Each controller's JacobianSensitivities caches the Jacobian at the current
operating point.  On each TSO step the cache is invalidated so the new
operating point is used.

Key differences from run_cascade.py
-------------------------------------
* Network: IEEE 39-bus (build_ieee39_net) instead of TU-Darmstadt.
* Zones: spectral_zone_partition defines 3 TSO zones dynamically.
* N TSOControllers instead of 1, coordinated by MultiTSOCoordinator.
* DSOs: 2-winding interface trafos (net.trafo) instead of 3-winding.
* No machine OLTCs: generators in case39 have direct AVR control (gen.vm_pu).
* No pandapower DiscreteTapControl: DSO OLTC initialized to tap_pos = 0.
* Measurement functions are custom (handle 2W PCC trafos and multi-zone).

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
from numpy.typing import NDArray

warnings.filterwarnings("ignore", category=UserWarning, module=r"mosek")

# ── Ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    MultiZoneStabilityResult,
)
from controller.base_controller import OFOParameters
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.multi_tso_coordinator import MultiTSOCoordinator, ZoneDefinition
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from core.measurement import Measurement
from core.network_state import NetworkState
from network.build_ieee39_net import build_ieee39_net, add_dso_feeders, IEEE39NetworkMeta
from network.zone_partition import (
    spectral_zone_partition,
    relabel_zones_by_generator_count,
    get_zone_lines,
    get_tie_lines,
)
from run.helpers import _network_state
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
    g_w_dso_der : float
        Regularisation on DSO DER Q changes.

    DSO parameters
    --------------
    n_dso_buses : int
        Number of Zone-2 load buses to replace with DSO feeders (default 2).
    n_der_per_feeder : int
        DER sgens per DSO feeder.
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
    g_v:            float = 1.0      # TSO voltage tracking weight
    g_q:            float = 1.0      # DSO Q-interface tracking weight
    dso_g_v:        float = 0.0      # DSO secondary voltage weight (0 = off)

    # ── OFO step-size ─────────────────────────────────────────────────────────
    alpha:          float = 1.0      # OFO step-size gain

    # ── G_w regularisation weights (TSO) ─────────────────────────────────────
    #
    # With α=1 and Q_DER in [Mvar], the step is clamped by g_w such that
    #   w* ≈ −gradient / (2 · g_w)
    # So g_w=100 limits a single step to ~0.5/g_v Mvar for a 1-pu voltage error.
    g_w_der:        float = 10.0    # [Mvar]² cost per DER Q step
    g_w_gen:        float = 1e8      # [p.u.]² cost per AVR step (very cautious)
    g_w_pcc:        float = 5.0    # [Mvar]² cost per PCC-Q setpoint step

    # ── G_w regularisation weights (DSO) ─────────────────────────────────────
    g_w_dso_der:    float = 5.0     # DSO DER Q regularisation
    g_w_dso_oltc:   float = 120.0    # DSO OLTC tap regularisation (unused for now)

    # ── DSO parameters ─────────────────────────────────────────────────────────
    n_dso_buses:    int   = 2        # number of Zone-2 buses replaced by DSO feeders
    n_der_per_feeder: int = 3        # DER sgens per feeder
    der_s_mva:      float = 80.0     # rated MVA per DSO DER
    der_p_mw:       float = 20.0     # fixed active power per DSO DER [MW]
    shunt_q_mvar:   float = 30.0     # switchable shunt step [Mvar]

    # ── Stability analysis ─────────────────────────────────────────────────────
    run_stability_analysis:       bool = True
    sensitivity_update_interval:  int  = 5  # recompute H_ij every N TSO steps

    # ── Output ────────────────────────────────────────────────────────────────
    verbose:        int   = 1
    result_dir:     str   = "results"

    # ── Live plot ─────────────────────────────────────────────────────────────
    live_plot:      bool  = False  # show real-time plot windows during simulation


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

    # ── PCC interface Q (2W trafo, HV side, load convention) ─────────────────
    #
    # For 2-winding trafos: net.res_trafo.q_hv_mvar gives Q at the HV port
    # with LOAD CONVENTION (positive = power flows INTO the HV bus from the
    # external grid side).
    #
    # TSOController expects "interface_q_hv_side_mvar" in the same convention.
    q_iface = np.zeros(len(zone_def.pcc_trafo_indices), dtype=np.float64)
    for k, t in enumerate(zone_def.pcc_trafo_indices):
        if t in net.res_trafo.index:
            q_iface[k] = float(net.res_trafo.at[t, "q_hv_mvar"])
        # (trafo3w fallback omitted — IEEE 39-bus uses only 2W)

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
        oltc_indices=np.array([], dtype=np.int64),        # no machine OLTCs in IEEE 39-bus
        oltc_tap_positions=np.array([], dtype=np.int64),
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
    lv_bus: int,
    pcc_trafo_idx: int,
    it: int,
) -> Measurement:
    """
    Build a Measurement for one DSO feeder from the plant network.

    The DSO interface uses 2-winding transformers (net.res_trafo) rather than
    the 3-winding couplers in the TU-Darmstadt network.

    Parameters
    ----------
    net : pandapowerNet
    dso_cfg : DSOControllerConfig
        The DSO controller's configuration (der_indices, oltc, shunt, etc.).
    lv_bus : int
        The 20 kV distribution bus for this feeder.
    pcc_trafo_idx : int
        The 2W PCC transformer index for this feeder (net.trafo).
    it : int
        Current step number.

    Returns
    -------
    Measurement
    """
    # All bus voltages (DSOController looks up by bus index)
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm_all  = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Line currents (DN lines; for simple radial feeders this may be empty)
    i_ka = np.array(
        [float(net.res_line.at[li, "i_from_ka"]) for li in dso_cfg.current_line_indices],
        dtype=np.float64,
    ) if dso_cfg.current_line_indices else np.array([], dtype=np.float64)

    # Interface Q at 2W PCC trafo (HV side, load convention)
    #
    # DSOController expects "interface_q_hv_side_mvar" to be the Q the TSO
    # wants the DSO to track.  Here we read the ACTUAL Q from the PF result.
    q_iface = np.array(
        [float(net.res_trafo.at[pcc_trafo_idx, "q_hv_mvar"])],
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

    # OLTC taps (2W DSO trafo)
    oltc_taps = np.array(
        [int(net.trafo.at[t, "tap_pos"]) for t in dso_cfg.oltc_trafo_indices],
        dtype=np.int64,
    ) if dso_cfg.oltc_trafo_indices else np.array([], dtype=np.int64)

    # Shunt states
    shunt_states = np.zeros(len(dso_cfg.shunt_bus_indices), dtype=np.int64)
    for k, sb in enumerate(dso_cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            shunt_states[k] = int(net.shunt.at[net.shunt.index[mask][0], "step"])

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

    # OLTC and shunt application omitted (not used in base IEEE 39-bus setup)


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

    # DSO OLTC tap positions (2W trafo)
    for k, t_idx in enumerate(dso_cfg.oltc_trafo_indices):
        net.trafo.at[t_idx, "tap_pos"] = int(round(u[off + k]))
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
        print("  MULTI-TSO / MULTI-DSO OFO  —  IEEE 39-bus New England")
        print(f"  V_set = {v_set:.3f} p.u.  |  N_zones = 3  |  α = {config.alpha}")
        print("=" * 72)

    # =========================================================================
    # STEP 1: Build the base IEEE 39-bus network (no DSO feeders yet)
    # =========================================================================
    if verbose >= 1:
        print("[1] Building IEEE 39-bus network ...")

    net, meta = build_ieee39_net(ext_grid_vm_pu=v_set)
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # =========================================================================
    # STEP 2: Spectral zone partitioning
    # =========================================================================
    if verbose >= 1:
        print("[2] Spectral zone partitioning (N=3) ...")

    zone_map, bus_zone = spectral_zone_partition(
        net, n_zones=3, verbose=(verbose >= 2)
    )
    # Relabel: Zone 0 = most generators, Zone 2 = fewest (prime candidate for DSO)
    zone_map, bus_zone = relabel_zones_by_generator_count(
        zone_map, bus_zone, list(meta.gen_bus_indices)
    )

    if verbose >= 1:
        for z in sorted(zone_map.keys()):
            n_gen_z = sum(1 for b in zone_map[z] if b in set(meta.gen_bus_indices))
            n_load_z = sum(
                1 for li in net.load.index if int(net.load.at[li, "bus"]) in zone_map[z]
            )
            print(f"  Zone {z}: {len(zone_map[z])} buses, "
                  f"{n_gen_z} generators, {n_load_z} loads")

    # =========================================================================
    # STEP 3: Add DSO feeders at Zone-2 load buses
    # =========================================================================
    #
    # We pick the load buses in Zone 2 with the highest load (largest P_mw)
    # as the DSO interface points.  Only config.n_dso_buses are used.
    zone2_buses   = set(zone_map[2])
    zone2_load_candidates = [
        (float(net.load.at[li, "p_mw"]), int(net.load.at[li, "bus"]))
        for li in net.load.index
        if int(net.load.at[li, "bus"]) in zone2_buses
    ]
    # Sort descending by load size; take the top n_dso_buses unique buses
    zone2_load_candidates.sort(reverse=True)
    dso_hv_buses: List[int] = []
    seen = set()
    for _, bus in zone2_load_candidates:
        if bus not in seen:
            dso_hv_buses.append(bus)
            seen.add(bus)
        if len(dso_hv_buses) >= config.n_dso_buses:
            break

    if verbose >= 1:
        print(f"[3] Adding DSO feeders at Zone-2 buses: {dso_hv_buses}")

    meta = add_dso_feeders(
        net, meta, dso_hv_buses,
        n_der_per_feeder=config.n_der_per_feeder,
        der_s_mva=config.der_s_mva,
        der_p_mw=config.der_p_mw,
        shunt_q_mvar=config.shunt_q_mvar,
    )

    # Re-converge power flow after adding DSO feeders
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # =========================================================================
    # STEP 4: Build ZoneDefinitions and TSOControllerConfigs
    # =========================================================================
    if verbose >= 1:
        print("[4] Building zone definitions and controller configs ...")

    # ── Partition generator indices per zone ──────────────────────────────────
    zone_gen_indices: Dict[int, List[int]] = {z: [] for z in zone_map}
    zone_gen_buses:   Dict[int, List[int]] = {z: [] for z in zone_map}
    for g, g_bus in zip(meta.gen_indices, meta.gen_bus_indices):
        for z, buses in zone_map.items():
            if g_bus in set(buses):
                zone_gen_indices[z].append(g)
                zone_gen_buses[z].append(g_bus)
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

    # ── DSO interface trafos belong to Zone 2 ─────────────────────────────────
    #
    # Each DSO feeder is connected at a Zone-2 bus.  The PCC trafo and the
    # DSO controller ID are assigned to Zone 2's TSOController.
    dso_ids: List[str] = [f"dso_zone2_{k}" for k in range(len(meta.dso_pcc_trafo_indices))]

    # ── Build ZoneDefinition for each zone ────────────────────────────────────
    #
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
        z_bus_set = set(zone_map[z])
        all_z_lines = get_zone_lines(net, z_bus_set)
        # Keep only lines with both endpoints at PQ buses (Jacobian builder requirement)
        z_lines = [
            li for li in all_z_lines
            if int(net.line.at[li, "from_bus"]) not in pv_and_slack_buses_run
            and int(net.line.at[li, "to_bus"]) not in pv_and_slack_buses_run
        ]
        z_line_max_i_ka = [
            float(net.line.at[li, "max_i_ka"]) for li in z_lines
        ]

        # PCC trafo and DSO IDs: only for Zone 2
        if z == 2:
            pcc_trafos = list(meta.dso_pcc_trafo_indices)
            pcc_dso_ids = dso_ids
        else:
            pcc_trafos   = []
            pcc_dso_ids  = []

        # Voltage observation buses: only PQ buses (not PV/slack).
        # build_sensitivity_matrix_H filters out PV/slack buses from
        # the V_bus rows of H_physical, so we must pre-filter here to
        # ensure n_v matches between the zone definition and H_physical.
        v_bus_indices_z = [
            b for b in zone_map[z] if b not in pv_and_slack_buses_run
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
            pcc_trafo_indices=pcc_trafos,
            pcc_dso_ids=pcc_dso_ids,
            v_setpoint_pu=v_set,
            alpha=config.alpha,
            g_v=config.g_v,
            g_w_der=config.g_w_der,
            g_w_gen=config.g_w_gen,
            g_w_pcc=config.g_w_pcc,
        )

    if verbose >= 1:
        for z, zd in zone_defs.items():
            print(f"  Zone {z}: {len(zd.gen_indices)} gen, {len(zd.tso_der_indices)} DER, "
                  f"{len(zd.line_indices)} lines, {len(zd.pcc_trafo_indices)} PCC trafos")

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
        )

        # TSOControllerConfig: pass zone-specific index sets
        tso_cfg = TSOControllerConfig(
            der_indices=zd.tso_der_indices,
            pcc_trafo_indices=zd.pcc_trafo_indices,
            pcc_dso_controller_ids=zd.pcc_dso_ids,
            oltc_trafo_indices=[],          # no machine OLTCs in case39
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

        # Generator capability parameters for this zone
        gen_params = [
            GeneratorParameters(
                s_rated_mva=float(net.gen.at[g, "sn_mva"]),
                p_max_mw=float(net.gen.at[g, "p_mw"]),
                # Default IEEE parameters (no detailed xd data in case39)
                xd_pu=1.0,
                i_f_max_pu=2.0,
                beta=0.1,
                q0_pu=0.1,
            )
            for g in zd.gen_indices
        ]

        bounds = ActuatorBounds(
            der_indices=np.array(zd.tso_der_indices, dtype=np.int64),
            der_s_rated_mva=s_rated,
            der_p_max_mw=p_max,
            oltc_indices=np.array([], dtype=np.int64),
            oltc_tap_min=np.array([], dtype=np.int64),
            oltc_tap_max=np.array([], dtype=np.int64),
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
    # STEP 6: Initialise DSO controllers (Zone 2 only)
    # =========================================================================
    if verbose >= 1:
        print("[6] Initialising DSO controllers (Zone 2) ...")

    dso_controllers: Dict[str, DSOController] = {}
    dso_lv_buses_map: Dict[str, int] = {}   # dso_id → LV bus
    dso_pcc_trafo_map: Dict[str, int] = {}  # dso_id → PCC trafo index

    for k, (dso_id, pcc_trafo, lv_bus) in enumerate(zip(
        dso_ids,
        meta.dso_pcc_trafo_indices,
        meta.dso_lv_buses,
    )):
        dso_lv_buses_map[dso_id]   = lv_bus
        dso_pcc_trafo_map[dso_id]  = pcc_trafo

        # DSO DERs and shunts at this feeder's LV bus
        feeder_der_idx = [
            s for s, sb in zip(meta.dso_der_indices, meta.dso_der_buses)
            if sb == lv_bus
        ]
        feeder_shunt_idx = [
            sh for sh, sb in zip(meta.dso_shunt_indices, meta.dso_shunt_buses)
            if sb == lv_bus
        ]
        feeder_shunt_q = [
            float(net.shunt.at[sh, "q_mvar"]) for sh in feeder_shunt_idx
        ]

        dso_gw_diag = np.concatenate([
            np.full(len(feeder_der_idx), config.g_w_dso_der),
            np.full(len([pcc_trafo]),    config.g_w_dso_oltc),  # OLTC
            np.full(len(feeder_shunt_idx), config.g_w_dso_oltc),  # shunts
        ]) if feeder_der_idx else np.array([config.g_w_dso_der])

        # Interfaces: the single PCC trafo (2W)
        # For DSOControllerConfig, interface_trafo_indices stores the 2W trafo
        # index (matched by the custom _measure_for_dso function).
        # Use feeder shunts if available (switchable capacitor bank)
        shunt_buses_cfg = [net.shunt.at[sh, "bus"] for sh in feeder_shunt_idx]

        dso_cfg = DSOControllerConfig(
            der_indices=feeder_der_idx,
            oltc_trafo_indices=[pcc_trafo],            # use the PCC trafo as OLTC
            shunt_bus_indices=list(shunt_buses_cfg),   # switchable shunts
            shunt_q_steps_mvar=[abs(q) for q in feeder_shunt_q],  # |q| = step size
            interface_trafo_indices=[pcc_trafo],
            voltage_bus_indices=[lv_bus],
            current_line_indices=[],            # no distribution lines in radial feeder
            current_line_max_i_ka=None,
            g_q=config.g_q,
            g_qi=0.0,                           # no integral term for simplicity
            lambda_qi=0.0,
            q_integral_max_mvar=1.0,            # placeholder (g_qi=0 disables it)
            v_setpoints_pu=np.array([v_set]),
            g_v=config.dso_g_v,
        )

        if feeder_der_idx:
            dso_s_rated = np.array(
                [float(net.sgen.at[s, "sn_mva"]) for s in feeder_der_idx],
                dtype=np.float64,
            )
            dso_p_max = np.array(
                [float(net.sgen.at[s, "p_mw"]) for s in feeder_der_idx],
                dtype=np.float64,
            )
        else:
            dso_s_rated = np.array([], dtype=np.float64)
            dso_p_max   = np.array([], dtype=np.float64)

        dso_bounds = ActuatorBounds(
            der_indices=np.array(feeder_der_idx, dtype=np.int64),
            der_s_rated_mva=dso_s_rated,
            der_p_max_mw=dso_p_max,
            oltc_indices=np.array([pcc_trafo], dtype=np.int64),
            oltc_tap_min=np.array([int(net.trafo.at[pcc_trafo, "tap_min"])], dtype=np.int64),
            oltc_tap_max=np.array([int(net.trafo.at[pcc_trafo, "tap_max"])], dtype=np.int64),
            shunt_indices=np.array(feeder_shunt_idx, dtype=np.int64),
            shunt_q_mvar=np.array([abs(q) for q in feeder_shunt_q], dtype=np.float64),
        )

        dso_ofo = OFOParameters(
            alpha=config.alpha,
            g_w=dso_gw_diag,
            g_z=np.zeros(1 + len([lv_bus]) + 0),  # interface + voltage + current
            g_u=np.zeros_like(dso_gw_diag),
        )

        dso_ctrl = DSOController(
            controller_id=dso_id,
            params=dso_ofo,
            config=dso_cfg,
            network_state=ns0,
            actuator_bounds=dso_bounds,
            sensitivities=JacobianSensitivities(net),
        )
        meas_dso_init = _measure_for_dso(net, dso_cfg, lv_bus, pcc_trafo, 0)
        dso_ctrl.initialise(meas_dso_init)
        dso_controllers[dso_id] = dso_ctrl

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

    # Initial cross-sensitivity and stability analysis
    coordinator.compute_cross_sensitivities()
    coordinator.compute_M_blocks()
    contraction_info = coordinator.check_contraction()

    if config.run_stability_analysis:
        if verbose >= 1:
            print("[7b] Running multi-zone stability analysis at t=0 ...")
        # Collect H blocks and G_w / Q_obj for stability analysis
        zone_ids_sorted = sorted(zone_defs.keys())
        H_blocks_stab = {k: coordinator.get_H_block(*k)
                         for k in coordinator._H_blocks}
        Q_obj_list = [zone_defs[z].q_obj_diagonal() for z in zone_ids_sorted]
        G_w_list   = [zone_defs[z].gw_diagonal()    for z in zone_ids_sorted]
        alpha_list  = [zone_defs[z].alpha            for z in zone_ids_sorted]

        stab_result = analyse_multi_zone_stability(
            H_blocks=H_blocks_stab,
            Q_obj_list=Q_obj_list,
            G_w_list=G_w_list,
            alpha_list=alpha_list,
            zone_ids=zone_ids_sorted,
            zone_names=[f"Zone {z}" for z in zone_ids_sorted],
            verbose=True,
        )
    else:
        stab_result = None

    # =========================================================================
    # STEP 8: Main simulation loop
    # =========================================================================
    if verbose >= 1:
        n_steps = int(config.n_total_s / config.dt_s)
        print(f"[8] Starting simulation: {n_steps} steps  "
              f"(dt={config.dt_s:.0f}s, TSO/{config.tso_period_s/60:.0f}min, "
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
                rec.zone_q_der[z]         = u[:n_der].copy()
                rec.zone_q_pcc_set[z]     = u[n_der:n_der+n_pcc].copy()
                rec.zone_v_gen[z]         = u[n_der+n_pcc:n_der+n_pcc+n_gen].copy()
                rec.zone_tso_objective[z] = tso_out.objective_value
                rec.zone_tso_status[z]    = tso_out.solver_status
                rec.zone_tso_solve_s[z]   = tso_out.solve_time_s

                # Record contraction diagnostic
                diag = coordinator.last_coupling_diagnostics.get(z, {})
                rec.zone_contraction_lhs[z] = diag.get("contraction_lhs", float("nan"))

            # TSO sends Q setpoints to Zone-2 DSOs via setpoint messages
            for z, ctrl in tso_controllers.items():
                for msg in ctrl.generate_setpoint_messages():
                    if msg.target_controller_id in dso_controllers:
                        dso_controllers[msg.target_controller_id].receive_setpoint(msg)
                        # Record the Q setpoint the TSO sent
                        if hasattr(msg, "q_setpoint_mvar"):
                            rec.dso_q_set_mvar[msg.target_controller_id] = float(
                                msg.q_setpoint_mvar
                            )

        # ── DSO step (Zone 2 only) ────────────────────────────────────────────
        if run_dso:
            for dso_id, dso_ctrl in dso_controllers.items():
                lv_bus    = dso_lv_buses_map[dso_id]
                pcc_trafo = dso_pcc_trafo_map[dso_id]
                meas_dso  = _measure_for_dso(
                    net, dso_ctrl.config, lv_bus, pcc_trafo, step
                )
                dso_out = dso_ctrl.step(meas_dso)
                _apply_dso_controls(net, dso_ctrl.config, dso_out)

                # Record DSO outputs
                n_dso_der = len(dso_ctrl.config.der_indices)
                rec.dso_q_der[dso_id]     = dso_out.u_new[:n_dso_der].copy()
                rec.dso_objective[dso_id] = dso_out.objective_value
                rec.dso_status[dso_id]    = dso_out.solver_status

                # Actual Q at PCC (read after applying setpoints but before PF)
                if pcc_trafo in net.res_trafo.index:
                    rec.dso_q_actual_mvar[dso_id] = float(
                        net.res_trafo.at[pcc_trafo, "q_hv_mvar"]
                    )

        # ── Power flow ────────────────────────────────────────────────────────
        try:
            pp.runpp(net, run_control=False, calculate_voltage_angles=True)
        except Exception as e:
            print(f"  [Step {step}] Power flow failed: {e}")
            log.append(rec)
            continue

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
        python run/run_multi_tso_dso.py
    """
    cfg = MultiTSOConfig(
        n_total_s=60.0 * 180,      # 30-minute simulation for quick test
        tso_period_s=60.0 * 3,    # TSO every 3 minutes
        dso_period_s=30.0 * 1,    # DSO every 1 minute
        alpha=1.0,
        g_v=50000.0,
        g_w_der=10.0,
        g_w_gen=1e7,
        run_stability_analysis=True,
        sensitivity_update_interval=1E6,  # refresh H_ij every 3 TSO steps
        verbose=1,
        n_dso_buses=2,
        live_plot=True
    )
    log = run_multi_tso_dso(cfg)
    print(f"\nSimulation complete. {len(log)} steps recorded.")


if __name__ == "__main__":
    main()
