#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascaded OFO Controller Simulation
====================================

This script runs the full TSO-DSO cascaded Online Feedback Optimisation
controller loop with pandapower as the plant model.

Key Architecture:
    - **combined_net** = the 'real' plant (measurements, power flow, control application)
    - **tn_net, dn_net** = controller models only (for sensitivities/Jacobians)
    - Controllers use tn_net/dn_net for computing sensitivities
    - All measurements come from combined_net (realistic feedback)
    - All control actions applied to combined_net
    - Power flow runs only on combined_net

Sensitivity Methods:
    - **Analytical (default)**: Fast Jacobian-based sensitivities
    - **Numerical (toggle)**: Finite-difference validation/debugging

Timing:
    - DSO controller executes every 1 minute
    - TSO controller executes every 3 minutes

The TSO controller tracks a uniform voltage setpoint at all monitored EHV
buses.  Multiple scenarios are run with setpoints from 1.00 to 1.10 p.u.

After each scenario, a summary of control variables and outputs at every
iteration is printed (and optionally plotted).

Author: Manuel Schwenke
Date: 2026-02-10
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Set
import copy

import numpy as np
from numpy.typing import NDArray
import pandapower as pp

# -- qOFO imports --------------------------------------------------------------
from network.build_tuda_net import build_tuda_net, NetworkMetadata
from network.split_tn_dn_net import split_network, SplitResult
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from controller.base_controller import OFOParameters, ControllerOutput
from controller.tso_controller import TSOController, TSOControllerConfig
from controller.dso_controller import DSOController, DSOControllerConfig
from core.message import SetpointMessage

# -- Sensitivity modules (both analytical and numerical) ----------------------
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.numerical import NumericalSensitivities

# Type alias for sensitivity calculator
SensitivityCalculator = Union[JacobianSensitivities, NumericalSensitivities]


# ===============================================================================
#  HELPER: Identify DN-only buses from combined network
# ===============================================================================

def _get_dn_only_buses(combined_net: pp.pandapowerNet) -> Set[int]:
    """
    Identify DN-only buses from the combined network by checking the 'subnet' column.
    
    Returns a set of bus indices where subnet == 'DN'.
    """
    if "subnet" not in combined_net.bus.columns:
        raise RuntimeError("Combined network buses have no 'subnet' column.")
    
    dn_mask = combined_net.bus["subnet"].astype(str) == "DN"
    return set(int(b) for b in combined_net.bus.index[dn_mask])


# ===============================================================================
#  HELPER: Build NetworkState from a converged pandapower network
# ===============================================================================

def network_state_from_net(
    net: pp.pandapowerNet,
    trafo_indices: NDArray[np.int64],
    source_case: str = "runtime",
    iteration: int = 0,
) -> NetworkState:
    """Create a NetworkState snapshot from a converged pandapower network."""
    bus_indices = np.array(net.bus.index, dtype=np.int64)
    vm = net.res_bus.loc[bus_indices, "vm_pu"].values.astype(np.float64)
    va = np.deg2rad(net.res_bus.loc[bus_indices, "va_degree"].values.astype(np.float64))

    # Slack bus
    slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])

    # PV buses (generators with vm_pu set)
    pv_buses = np.array(
        [int(net.gen.at[g, "bus"]) for g in net.gen.index if net.gen.at[g, "in_service"]],
        dtype=np.int64,
    )

    # PQ buses = everything else
    all_special = set([slack_bus]) | set(pv_buses.tolist())
    pq_buses = np.array(
        [int(b) for b in bus_indices if int(b) not in all_special],
        dtype=np.int64,
    )

    # Tap positions
    tap_pos = np.array(
        [float(net.trafo.at[t, "tap_pos"]) for t in trafo_indices],
        dtype=np.float64,
    )

    return NetworkState(
        bus_indices=bus_indices,
        voltage_magnitudes_pu=vm,
        voltage_angles_rad=va,
        slack_bus_index=slack_bus,
        pv_bus_indices=pv_buses,
        pq_bus_indices=pq_buses,
        transformer_indices=trafo_indices,
        tap_positions=tap_pos,
        source_case=source_case,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cached_at_iteration=iteration,
    )


# ===============================================================================
#  HELPER: Extract Measurement from COMBINED plant
# ===============================================================================

def measurement_from_combined_tn_side(
    combined_net: pp.pandapowerNet,
    meta: NetworkMetadata,
    dn_buses: Set[int],
    tso_config: TSOControllerConfig,
    iteration: int,
) -> Measurement:
    """
    Build a TSO Measurement from the COMBINED plant network.
    
    Extracts TN-side quantities from the combined network (real plant).
    All measurements come from the physical system (combined_net).
    """
    # All bus indices from combined network (real plant)
    all_bus = np.array(sorted(combined_net.res_bus.index), dtype=np.int64)
    vm = combined_net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Branch currents -- TN lines only (from combined plant)
    line_idx = np.array(tso_config.current_line_indices, dtype=np.int64)
    i_ka = np.zeros(len(line_idx), dtype=np.float64)
    for k, li in enumerate(line_idx):
        if li not in combined_net.res_line.index:
            raise ValueError(f"TN line {li} not found in combined_net.res_line.")
        i_ka[k] = float(combined_net.res_line.at[li, "i_from_ka"])

    # Interface Q at coupler HV buses -- read from 3W trafo HV side (combined plant)
    iface_trafo = np.array(tso_config.pcc_trafo_indices, dtype=np.int64)
    q_iface = np.zeros(len(iface_trafo), dtype=np.float64)
    for k, t in enumerate(iface_trafo):
        if t not in combined_net.res_trafo3w.index:
            raise ValueError(f"Interface trafo {t} not found in combined_net.res_trafo3w.")
        q_iface[k] = float(combined_net.res_trafo3w.at[t, "q_hv_mvar"])

    # DER Q (transmission-connected sgens) -- exclude boundary sgens (combined plant)
    der_bus = np.array(tso_config.der_bus_indices, dtype=np.int64)
    der_q = np.zeros(len(der_bus), dtype=np.float64)
    for i, bus in enumerate(der_bus):
        sgen_mask = combined_net.sgen["bus"] == bus
        # Only count sgens that are not boundary injections
        for sidx in combined_net.sgen.index[sgen_mask]:
            # Check if this sgen has the boundary marker
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                der_q[i] += float(combined_net.res_sgen.at[sidx, "q_mvar"])

    # OLTC tap positions (2W machine transformers) from combined plant
    oltc_idx = np.array(tso_config.oltc_trafo_indices, dtype=np.int64)
    oltc_taps = np.zeros(len(oltc_idx), dtype=np.int64)
    for k, t in enumerate(oltc_idx):
        if t not in combined_net.trafo.index:
            raise ValueError(f"OLTC trafo {t} not found in combined_net.trafo.")
        oltc_taps[k] = int(combined_net.trafo.at[t, "tap_pos"])

    # Shunt states from combined plant
    shunt_bus = np.array(tso_config.shunt_bus_indices, dtype=np.int64)
    shunt_states = np.zeros(len(shunt_bus), dtype=np.int64)
    for i, sb in enumerate(shunt_bus):
        mask = combined_net.shunt["bus"] == sb
        if mask.any():
            sidx = combined_net.shunt.index[mask][0]
            shunt_states[i] = int(combined_net.shunt.at[sidx, "step"])

    # Generator AVR setpoints from combined plant
    gen_idx = np.array(tso_config.gen_indices, dtype=np.int64)
    gen_vm = np.zeros(len(gen_idx), dtype=np.float64)
    for k, g in enumerate(gen_idx):
        if g not in combined_net.gen.index:
            raise ValueError(f"Generator index {g} not found in combined_net.gen.")
        gen_vm[k] = float(combined_net.gen.at[g, "vm_pu"])

    return Measurement(
        iteration=iteration,
        bus_indices=all_bus,
        voltage_magnitudes_pu=vm,
        branch_indices=line_idx,
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=iface_trafo,
        interface_q_hv_side_mvar=q_iface,
        der_indices=der_bus,
        der_q_mvar=der_q,
        oltc_indices=oltc_idx,
        oltc_tap_positions=oltc_taps,
        shunt_indices=shunt_bus,
        shunt_states=shunt_states,
        gen_indices=gen_idx,
        gen_vm_pu=gen_vm,
    )


def measurement_from_combined_dn_side(
    combined_net: pp.pandapowerNet,
    meta: NetworkMetadata,
    dn_buses: Set[int],
    dso_config: DSOControllerConfig,
    iteration: int,
) -> Measurement:
    """
    Build a DSO Measurement from the COMBINED plant network.
    
    Extracts DN-side quantities from the combined network (real plant).
    All measurements come from the physical system (combined_net).
    """
    # All bus indices from combined network (real plant)
    all_bus = np.array(sorted(combined_net.res_bus.index), dtype=np.int64)
    vm = combined_net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Branch currents -- DN lines only (from combined plant)
    line_idx = np.array(dso_config.current_line_indices, dtype=np.int64)
    i_ka = np.zeros(len(line_idx), dtype=np.float64)
    for k, li in enumerate(line_idx):
        if li not in combined_net.res_line.index:
            raise ValueError(f"DN line {li} not found in combined_net.res_line.")
        i_ka[k] = float(combined_net.res_line.at[li, "i_from_ka"])

    # Interface Q -- from 3W transformer HV side result (combined plant)
    iface_trafo = np.array(dso_config.interface_trafo_indices, dtype=np.int64)
    q_iface = np.zeros(len(iface_trafo), dtype=np.float64)
    for k, t in enumerate(iface_trafo):
        if t not in combined_net.res_trafo3w.index:
            raise ValueError(f"Interface trafo {t} not found in combined_net.res_trafo3w.")
        q_iface[k] = float(combined_net.res_trafo3w.at[t, "q_hv_mvar"])

    # DER Q -- sum Q from all sgens at each (unique) bus in DN (combined plant)
    der_bus = np.array(dso_config.der_bus_indices, dtype=np.int64)
    der_q = np.zeros(len(der_bus), dtype=np.float64)
    for i, bus in enumerate(der_bus):
        sgen_mask = combined_net.sgen["bus"] == bus
        for sidx in combined_net.sgen.index[sgen_mask]:
            # Only count DN sgens (not boundary)
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                der_q[i] += float(combined_net.res_sgen.at[sidx, "q_mvar"])

    # OLTC tap positions (3W transformers in DN) from combined plant
    oltc_idx = np.array(dso_config.oltc_trafo_indices, dtype=np.int64)
    oltc_taps = np.zeros(len(oltc_idx), dtype=np.int64)
    for k, t in enumerate(oltc_idx):
        if t not in combined_net.trafo3w.index:
            raise ValueError(f"OLTC trafo {t} not found in combined_net.trafo3w.")
        oltc_taps[k] = int(combined_net.trafo3w.at[t, "tap_pos"])

    # Shunt states from combined plant
    shunt_bus = np.array(dso_config.shunt_bus_indices, dtype=np.int64)
    shunt_states = np.zeros(len(shunt_bus), dtype=np.int64)
    for i, sb in enumerate(shunt_bus):
        mask = combined_net.shunt["bus"] == sb
        if mask.any():
            sidx = combined_net.shunt.index[mask][0]
            shunt_states[i] = int(combined_net.shunt.at[sidx, "step"])

    return Measurement(
        iteration=iteration,
        bus_indices=all_bus,
        voltage_magnitudes_pu=vm,
        branch_indices=line_idx,
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=iface_trafo,
        interface_q_hv_side_mvar=q_iface,
        der_indices=der_bus,
        der_q_mvar=der_q,
        oltc_indices=oltc_idx,
        oltc_tap_positions=oltc_taps,
        shunt_indices=shunt_bus,
        shunt_states=shunt_states,
    )


# ===============================================================================
#  HELPER: Apply control outputs to COMBINED plant
# ===============================================================================

def apply_tso_controls_to_combined(
    combined_net: pp.pandapowerNet,
    tso_output: ControllerOutput,
    tso_config: TSOControllerConfig,
) -> None:
    """
    Apply TSO control outputs to the COMBINED pandapower plant network.
    """
    u = tso_output.u_new
    n_der = len(tso_config.der_bus_indices)
    n_pcc = len(tso_config.pcc_trafo_indices)
    n_gen = len(tso_config.gen_indices)
    n_oltc = len(tso_config.oltc_trafo_indices)
    n_shunt = len(tso_config.shunt_bus_indices)

    # DER Q setpoints
    for i, bus in enumerate(tso_config.der_bus_indices):
        sgen_mask = combined_net.sgen["bus"] == bus
        for sidx in combined_net.sgen.index[sgen_mask]:
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                combined_net.sgen.at[sidx, "q_mvar"] = u[i]
                break  # Only set first non-boundary sgen

    # PCC setpoints are NOT applied here -- they are communicated to DSO
    # The DSO will actuate DERs in the DN to track the PCC setpoints

    # AVR setpoints
    avr_start = n_der + n_pcc
    avr_end = avr_start + n_gen
    avr_values = u[avr_start:avr_end]
    for g_idx, vm in zip(tso_config.gen_indices, avr_values):
        if g_idx not in combined_net.gen.index:
            raise ValueError(f"Generator index {g_idx} not found in combined_net.gen.")
        combined_net.gen.at[g_idx, "vm_pu"] = float(vm)

    # OLTC taps (2W machine transformers)
    tap_start = avr_end
    for i, trafo_idx in enumerate(tso_config.oltc_trafo_indices):
        combined_net.trafo.at[trafo_idx, "tap_pos"] = int(
            np.round(u[tap_start + i])
        )

    # Shunt states
    shunt_start = tap_start + n_oltc
    for i, shunt_bus in enumerate(tso_config.shunt_bus_indices):
        mask = combined_net.shunt["bus"] == shunt_bus
        if mask.any():
            sidx = combined_net.shunt.index[mask][0]
            combined_net.shunt.at[sidx, "step"] = int(
                np.round(u[shunt_start + i])
            )


def apply_dso_controls_to_combined(
    combined_net: pp.pandapowerNet,
    dso_output: ControllerOutput,
    dso_config: DSOControllerConfig,
) -> None:
    """
    Apply DSO control outputs to the COMBINED pandapower plant network.
    """
    u = dso_output.u_new
    n_der = len(dso_config.der_bus_indices)
    n_oltc = len(dso_config.oltc_trafo_indices)

    # DER Q setpoints -- distribute evenly across all sgens at each bus
    for i, bus in enumerate(dso_config.der_bus_indices):
        sgen_mask = combined_net.sgen["bus"] == bus
        # Count non-boundary sgens
        valid_sgens = []
        for sidx in combined_net.sgen.index[sgen_mask]:
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                valid_sgens.append(sidx)
        
        n_sgens = len(valid_sgens)
        if n_sgens > 0:
            q_per_sgen = u[i] / n_sgens
            for sidx in valid_sgens:
                combined_net.sgen.at[sidx, "q_mvar"] = q_per_sgen

    # OLTC taps (3W transformers)
    for i, trafo_idx in enumerate(dso_config.oltc_trafo_indices):
        combined_net.trafo3w.at[trafo_idx, "tap_pos"] = int(np.round(u[n_der + i]))

    # Shunt states
    for i, shunt_bus in enumerate(dso_config.shunt_bus_indices):
        mask = combined_net.shunt["bus"] == shunt_bus
        if mask.any():
            sidx = combined_net.shunt.index[mask][0]
            combined_net.shunt.at[sidx, "step"] = int(np.round(u[n_der + n_oltc + i]))


# ===============================================================================
#  ITERATION LOG
# ===============================================================================

@dataclass
class IterationRecord:
    """Record of one cascade iteration."""
    minute: int
    tso_active: bool
    dso_active: bool

    # TSO outputs
    tso_voltages_pu: Optional[NDArray[np.float64]] = None
    tso_q_der_mvar: Optional[NDArray[np.float64]] = None
    tso_q_pcc_set_mvar: Optional[NDArray[np.float64]] = None
    tso_oltc_taps: Optional[NDArray[np.int64]] = None
    tso_shunt_states: Optional[NDArray[np.int64]] = None
    tso_objective: Optional[float] = None
    tso_solver_status: Optional[str] = None
    tso_solve_time_s: Optional[float] = None
    tso_slack: Optional[NDArray[np.float64]] = None

    # DSO outputs
    dso_q_interface_mvar: Optional[NDArray[np.float64]] = None
    dso_q_der_mvar: Optional[NDArray[np.float64]] = None
    dso_oltc_taps: Optional[NDArray[np.int64]] = None
    dso_shunt_states: Optional[NDArray[np.int64]] = None
    dso_voltages_pu: Optional[NDArray[np.float64]] = None
    dso_objective: Optional[float] = None
    dso_solver_status: Optional[str] = None
    dso_solve_time_s: Optional[float] = None
    dso_slack: Optional[NDArray[np.float64]] = None

    # Plant measurements after power flow
    plant_tn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_dn_voltages_pu: Optional[NDArray[np.float64]] = None


# ===============================================================================
#  MAIN CASCADE RUNNER
# ===============================================================================

def run_cascade(
    v_setpoint_pu: float,
    n_minutes: int = 30,
    dso_period_min: int = 1,
    tso_period_min: int = 3,
    alpha: float = 0.05,
    use_numerical_sensitivities: bool = False,
    verbose: bool = True,
) -> List[IterationRecord]:
    """
    Run the cascaded TSO-DSO OFO controller loop.

    Architecture:
        - combined_net: the 'real' plant (measurements, power flow, control application)
        - tn_net, dn_net: controller models only (sensitivities)

    Parameters
    ----------
    v_setpoint_pu : float
        Uniform voltage setpoint for all monitored EHV buses [p.u.].
    n_minutes : int
        Total simulation time in minutes.
    dso_period_min : int
        DSO controller execution period in minutes.
    tso_period_min : int
        TSO controller execution period in minutes.
    alpha : float
        OFO step size (gain).
    use_numerical_sensitivities : bool, optional
        If True, use numerical finite-difference sensitivities instead of
        analytical Jacobian-based sensitivities (default: False).
        Numerical method is slower but can help validate/debug analytical method.
    verbose : bool
        Print per-iteration details.

    Returns
    -------
    log : list[IterationRecord]
        One record per simulated minute.
    """
    sens_method = "NUMERICAL" if use_numerical_sensitivities else "ANALYTICAL"
    
    if verbose:
        print("=" * 72)
        print(f"  CASCADED OFO SIMULATION -- V_setpoint = {v_setpoint_pu:.3f} p.u.")
        print("=" * 72)
        print("  Architecture: combined_net (plant) + tn_net/dn_net (models)")
        print(f"  Sensitivity method: {sens_method}")
        print("=" * 72)

    # -- 1) Build combined network (plant) and model networks ---------------
    if verbose:
        print("[1/5] Building combined 380/110/20 kV network (real plant) ...")
    combined_net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

    if verbose:
        print("[2/5] Creating TN and DN model networks for controllers ...")
    # Create split for identifying TN/DN elements and creating model networks
    split = split_network(combined_net, meta, dn_slack_coupler_index=0)
    
    # Identify DN-only buses from combined network
    dn_buses = _get_dn_only_buses(combined_net)
    
    # Create model networks (deep copies for sensitivity computation)
    tn_net = copy.deepcopy(split.tn_net)
    dn_net = copy.deepcopy(split.dn_net)
    
    if verbose:
        print(f"       combined_net (plant): {len(combined_net.bus)} buses")
        print(f"       tn_net (model): {len(tn_net.bus)} buses")
        print(f"       dn_net (model): {len(dn_net.bus)} buses")
        print(f"       DN-only buses: {len(dn_buses)}")

    # -- 2) Identify element indices for controller configs FROM COMBINED NET ----
    
    if verbose:
        print("[3/5] Identifying TSO and DSO elements from combined network ...")

    # TSO DER: transmission-connected sgens (not boundary sgens) from COMBINED NET
    tso_der_buses = []
    for sidx in combined_net.sgen.index:
        bus = int(combined_net.sgen.at[sidx, "bus"])
        # TN sgens are those not in DN-only buses
        if bus not in dn_buses:
            # Not a boundary sgen
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                if bus not in tso_der_buses:
                    tso_der_buses.append(bus)

    # TSO voltage monitoring: all EHV (380 kV) buses from COMBINED NET
    tso_v_buses = sorted(
        int(b) for b in combined_net.bus.index
        if float(combined_net.bus.at[b, "vn_kv"]) >= 300.0
    )

    # TSO current monitoring: all TN lines from COMBINED NET
    tso_lines = [li for li in combined_net.line.index 
                 if li in tn_net.line.index]

    # TSO machine transformers (OLTCs) from metadata
    tso_oltc_candidates = list(meta.machine_trafo_indices)

    # Synchronous generators and their associated grid-side buses from COMBINED NET
    tso_gen_indices: List[int] = []
    tso_gen_bus_indices: List[int] = []
    for g in combined_net.gen.index:
        tso_gen_indices.append(int(g))
        gen_lv_bus = int(combined_net.gen.at[g, "bus"])
        mt_mask = (
            (combined_net.trafo["lv_bus"] == gen_lv_bus)
            & combined_net.trafo["name"].astype(str).str.startswith("MachineTrf|")
        )
        if not mt_mask.any():
            raise RuntimeError(
                f"No machine transformer found for generator {g} "
                f"(bus {gen_lv_bus}) in combined network."
            )
        mt_idx = int(combined_net.trafo.index[mt_mask][0])
        hv_bus = int(combined_net.trafo.at[mt_idx, "hv_bus"])
        tso_gen_bus_indices.append(hv_bus)

    # TSO shunts from metadata
    tso_shunt_bus_candidates = []
    tso_shunt_q_candidates = []
    for sidx in meta.tn_shunt_indices:
        if sidx in combined_net.shunt.index:
            tso_shunt_bus_candidates.append(int(combined_net.shunt.at[sidx, "bus"]))
            tso_shunt_q_candidates.append(float(combined_net.shunt.at[sidx, "q_mvar"]))

    # Probe TN MODEL to find which elements survive sensitivity computation
    if use_numerical_sensitivities:
        _probe_sens: SensitivityCalculator = NumericalSensitivities(tn_net)
    else:
        _probe_sens = JacobianSensitivities(tn_net)
    
    _H_probe, _m_probe = _probe_sens.build_sensitivity_matrix_H(
        der_bus_indices=tso_der_buses,
        observation_bus_indices=tso_v_buses,
        line_indices=tso_lines,
        oltc_trafo_indices=tso_oltc_candidates,
        shunt_bus_indices=tso_shunt_bus_candidates,
        shunt_q_steps_mvar=tso_shunt_q_candidates,
    )
    # Use only elements that actually survived sensitivity computation
    tso_oltc = list(_m_probe.get("oltc_trafos", []))
    tso_v_buses = list(_m_probe.get("obs_buses", tso_v_buses))
    tso_shunt_buses = list(_m_probe.get("shunt_buses", []))
    tso_shunt_q_steps = [
        tso_shunt_q_candidates[tso_shunt_bus_candidates.index(b)]
        for b in tso_shunt_buses
    ]

    # PCC info (one per coupler) from metadata
    pcc_trafo_indices = list(meta.coupler_trafo3w_indices)
    n_pcc = len(pcc_trafo_indices)
    dso_ids = ["dso_0"] * n_pcc

    # Voltage setpoints
    v_setpoints = np.full(len(tso_v_buses), v_setpoint_pu)

    tso_config = TSOControllerConfig(
        der_bus_indices=tso_der_buses,
        pcc_trafo_indices=pcc_trafo_indices,
        pcc_dso_controller_ids=dso_ids,
        oltc_trafo_indices=tso_oltc,
        shunt_bus_indices=tso_shunt_buses,
        shunt_q_steps_mvar=tso_shunt_q_steps,
        voltage_bus_indices=tso_v_buses,
        current_line_indices=tso_lines,
        v_setpoints_pu=v_setpoints,
        gamma_v_tracking=1.0,
        gen_indices=tso_gen_indices,
        gen_bus_indices=tso_gen_bus_indices,
        gen_vm_min_pu=0.95,
        gen_vm_max_pu=1.10,
    )

    # DSO configs - identify from COMBINED NET
    dso_der_buses_raw = []
    for sidx in combined_net.sgen.index:
        bus = int(combined_net.sgen.at[sidx, "bus"])
        # DN sgens are those in DN-only buses
        if bus in dn_buses:
            # Not a boundary sgen
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                dso_der_buses_raw.append(bus)
    
    # Deduplicate while preserving order
    seen = set()
    dso_der_buses = []
    for b in dso_der_buses_raw:
        if b not in seen:
            seen.add(b)
            dso_der_buses.append(b)

    # DSO voltage monitoring: all DN (110 kV) buses from COMBINED NET
    dso_v_buses = sorted(
        int(b) for b in combined_net.bus.index
        if 100.0 <= float(combined_net.bus.at[b, "vn_kv"]) <= 120.0
        and b in dn_buses
    )

    # DSO current monitoring: DN lines from COMBINED NET
    dso_lines = [li for li in combined_net.line.index 
                 if li in dn_net.line.index]

    # DSO OLTCs: the 3W coupler transformers
    dso_oltc_candidates = list(meta.coupler_trafo3w_indices)

    # DSO shunts (exclude tertiary 20 kV shunts)
    dso_shunt_buses = []
    dso_shunt_q_steps = []

    # Probe DSO MODEL to find surviving elements
    if use_numerical_sensitivities:
        _dso_probe_sens: SensitivityCalculator = NumericalSensitivities(dn_net)
    else:
        _dso_probe_sens = JacobianSensitivities(dn_net)
    
    _H_dso_probe, _m_dso_probe = _dso_probe_sens.build_sensitivity_matrix_H(
        der_bus_indices=dso_der_buses,
        observation_bus_indices=dso_v_buses,
        line_indices=dso_lines,
        trafo3w_indices=list(meta.coupler_trafo3w_indices),
        oltc_trafo3w_indices=dso_oltc_candidates,
    )
    dso_oltc = list(_m_dso_probe.get("oltc_trafo3w", dso_oltc_candidates))
    dso_v_buses = list(_m_dso_probe.get("obs_buses", dso_v_buses))

    dso_config = DSOControllerConfig(
        der_bus_indices=dso_der_buses,
        oltc_trafo_indices=dso_oltc,
        shunt_bus_indices=dso_shunt_buses,
        shunt_q_steps_mvar=dso_shunt_q_steps,
        interface_trafo_indices=pcc_trafo_indices,
        voltage_bus_indices=dso_v_buses,
        current_line_indices=dso_lines,
        gamma_q_tracking=100.0,
    )

    # -- 3) Create controller instances ------------------------------------
    if verbose:
        print("[4/5] Creating TSO and DSO controllers ...")

    ofo_params_tso = OFOParameters(
        alpha=0.01,
        g_w=0.00001,
        g_z=1000000.0,
        g_s=100.0,
        g_u=0.0001,
    )
    ofo_params_dso = OFOParameters(
        alpha=0.01,
        g_w=0.001,
        g_z=1000000.0,
        g_s=50.0,
        g_u=0.00001,
    )

    # TSO sensitivities & state (from TN MODEL)
    tso_trafo_idx = np.array(tso_oltc, dtype=np.int64)
    tso_ns = network_state_from_net(tn_net, tso_trafo_idx, source_case="TN_model")
    
    if use_numerical_sensitivities:
        tso_sens: SensitivityCalculator = NumericalSensitivities(tn_net)
    else:
        tso_sens = JacobianSensitivities(tn_net)

    # TSO bounds (from COMBINED PLANT)
    tso_bounds = ActuatorBounds(
        der_indices=np.array(tso_der_buses, dtype=np.int64),
        der_s_rated_mva=np.array(
            [float(combined_net.sgen.at[s, "sn_mva"])
             for s in combined_net.sgen.index
             if (int(combined_net.sgen.at[s, "bus"]) in tso_der_buses
                 and not str(combined_net.sgen.at[s, "name"]).startswith("BOUND_"))],
            dtype=np.float64,
        )[:len(tso_der_buses)],  # Ensure correct length
        der_p_max_mw=np.array(
            [float(combined_net.sgen.at[s, "p_mw"])
             for s in combined_net.sgen.index
             if (int(combined_net.sgen.at[s, "bus"]) in tso_der_buses
                 and not str(combined_net.sgen.at[s, "name"]).startswith("BOUND_"))],
            dtype=np.float64,
        )[:len(tso_der_buses)],
        oltc_indices=np.array(tso_oltc, dtype=np.int64),
        oltc_tap_min=np.array(
            [int(combined_net.trafo.at[t, "tap_min"]) for t in tso_oltc],
            dtype=np.int64,
        ),
        oltc_tap_max=np.array(
            [int(combined_net.trafo.at[t, "tap_max"]) for t in tso_oltc],
            dtype=np.int64,
        ),
        shunt_indices=np.array(tso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(tso_shunt_q_steps, dtype=np.float64),
    )

    tso = TSOController(
        controller_id="tso_main",
        params=ofo_params_tso,
        config=tso_config,
        network_state=tso_ns,
        actuator_bounds=tso_bounds,
        sensitivities=tso_sens,
    )

    # DSO sensitivities & state (from DN MODEL)
    dso_trafo3w_idx = np.array(dso_oltc, dtype=np.int64)
    dso_ns = network_state_from_net(
        dn_net,
        np.array([], dtype=np.int64),
        source_case="DN_model",
    )
    
    if use_numerical_sensitivities:
        dso_sens: SensitivityCalculator = NumericalSensitivities(dn_net)
    else:
        dso_sens = JacobianSensitivities(dn_net)

    # Aggregate DER ratings per unique bus from COMBINED PLANT
    dso_der_s_rated = {b: 0.0 for b in dso_der_buses}
    dso_der_p_max = {b: 0.0 for b in dso_der_buses}
    for sidx in combined_net.sgen.index:
        bus = int(combined_net.sgen.at[sidx, "bus"])
        if bus in dso_der_buses:
            if not str(combined_net.sgen.at[sidx, "name"]).startswith("BOUND_"):
                dso_der_s_rated[bus] += float(combined_net.sgen.at[sidx, "sn_mva"])
                dso_der_p_max[bus] += float(combined_net.sgen.at[sidx, "p_mw"])

    dso_bounds = ActuatorBounds(
        der_indices=np.array(dso_der_buses, dtype=np.int64),
        der_s_rated_mva=np.array(
            [dso_der_s_rated[b] for b in dso_der_buses],
            dtype=np.float64,
        ),
        der_p_max_mw=np.array(
            [dso_der_p_max[b] for b in dso_der_buses],
            dtype=np.float64,
        ),
        oltc_indices=dso_trafo3w_idx,
        oltc_tap_min=np.array(
            [int(combined_net.trafo3w.at[t, "tap_min"]) for t in dso_oltc],
            dtype=np.int64,
        ),
        oltc_tap_max=np.array(
            [int(combined_net.trafo3w.at[t, "tap_max"]) for t in dso_oltc],
            dtype=np.int64,
        ),
        shunt_indices=np.array(dso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(dso_shunt_q_steps, dtype=np.float64),
    )

    dso = DSOController(
        controller_id=dso_ids[0],
        params=ofo_params_dso,
        config=dso_config,
        network_state=dso_ns,
        actuator_bounds=dso_bounds,
        sensitivities=dso_sens,
    )

    # -- 4) Initialise controllers from COMBINED PLANT ---------------------
    if verbose:
        print("[5/5] Initialising controllers from combined plant power flow ...")

    # Run initial power flow on COMBINED PLANT
    pp.runpp(combined_net, run_control=False, calculate_voltage_angles=True)

    tso_meas = measurement_from_combined_tn_side(
        combined_net, meta, dn_buses, tso_config, iteration=0
    )
    dso_meas = measurement_from_combined_dn_side(
        combined_net, meta, dn_buses, dso_config, iteration=0
    )

    tso.initialise(tso_meas)
    dso.initialise(dso_meas)

    # -- 5) Run the cascade loop -------------------------------------------
    if verbose:
        print(f"\nRunning cascade for {n_minutes} minutes ...")
        print(f"  DSO period = {dso_period_min} min,  TSO period = {tso_period_min} min")
        print("  • Measurements from: COMBINED network (real plant)")
        print("  • Controls applied to: COMBINED network (real plant)")
        print("  • Power flow on: COMBINED network (real plant)")
        print("  • Sensitivities from: TN/DN model networks")
        print()

    log: List[IterationRecord] = []

    for minute in range(1, n_minutes + 1):
        run_tso = (minute % tso_period_min == 0)
        run_dso = (minute % dso_period_min == 0)

        rec = IterationRecord(minute=minute, tso_active=run_tso, dso_active=run_dso)

        # -- TSO step (every tso_period_min minutes) -----------------------
        if run_tso:
            # Measure from COMBINED PLANT
            tso_meas = measurement_from_combined_tn_side(
                combined_net, meta, dn_buses, tso_config, iteration=minute
            )
            try:
                tso_output = tso.step(tso_meas)

                n_der_t = len(tso_config.der_bus_indices)
                n_pcc_t = len(tso_config.pcc_trafo_indices)
                n_gen_t = len(tso_config.gen_indices)
                n_oltc_t = len(tso_config.oltc_trafo_indices)
                n_shunt_t = len(tso_config.shunt_bus_indices)

                rec.tso_q_der_mvar = tso_output.u_new[:n_der_t].copy()
                rec.tso_q_pcc_set_mvar = tso_output.u_new[n_der_t:n_der_t + n_pcc_t].copy()
                
                oltc_start = n_der_t + n_pcc_t + n_gen_t
                rec.tso_oltc_taps = np.round(
                    tso_output.u_new[oltc_start:oltc_start + n_oltc_t]
                ).astype(np.int64)
                
                shunt_start = oltc_start + n_oltc_t
                rec.tso_shunt_states = np.round(
                    tso_output.u_new[shunt_start:shunt_start + n_shunt_t]
                ).astype(np.int64)
                
                rec.tso_objective = tso_output.objective_value
                rec.tso_solver_status = tso_output.solver_status
                rec.tso_solve_time_s = tso_output.solve_time_s
                rec.tso_slack = tso_output.z_slack.copy()

                # Predicted voltages
                n_v = len(tso_config.voltage_bus_indices)
                rec.tso_voltages_pu = tso_output.y_predicted[:n_v].copy()

                # Apply TSO controls to COMBINED PLANT
                apply_tso_controls_to_combined(
                    combined_net, tso_output, tso_config
                )

                # Generate setpoint messages for DSO
                setpoint_msgs = tso.generate_setpoint_messages()
                dso_msgs = [m for m in setpoint_msgs
                            if m.target_controller_id == dso.controller_id]
                if dso_msgs:
                    merged_trafo_idx = np.concatenate(
                        [m.interface_transformer_indices for m in dso_msgs]
                    )
                    merged_q_set = np.concatenate(
                        [m.q_setpoints_mvar for m in dso_msgs]
                    )
                    merged_msg = SetpointMessage(
                        source_controller_id="tso_main",
                        target_controller_id=dso.controller_id,
                        iteration=minute,
                        interface_transformer_indices=merged_trafo_idx,
                        q_setpoints_mvar=merged_q_set,
                    )
                    dso.receive_setpoint(merged_msg)

            except RuntimeError as e:
                if verbose:
                    print(f"  [min {minute:3d}] TSO FAILED: {e}")

        # -- DSO step (every dso_period_min minutes) -----------------------
        if run_dso:
            # Measure from COMBINED PLANT
            dso_meas = measurement_from_combined_dn_side(
                combined_net, meta, dn_buses, dso_config, iteration=minute
            )
            try:
                dso_output = dso.step(dso_meas)

                n_der_d = len(dso_config.der_bus_indices)
                n_oltc_d = len(dso_config.oltc_trafo_indices)
                n_shunt_d = len(dso_config.shunt_bus_indices)

                rec.dso_q_der_mvar = dso_output.u_new[:n_der_d].copy()
                rec.dso_oltc_taps = np.round(
                    dso_output.u_new[n_der_d:n_der_d + n_oltc_d]
                ).astype(np.int64)
                rec.dso_shunt_states = np.round(
                    dso_output.u_new[n_der_d + n_oltc_d:n_der_d + n_oltc_d + n_shunt_d]
                ).astype(np.int64)
                rec.dso_objective = dso_output.objective_value
                rec.dso_solver_status = dso_output.solver_status
                rec.dso_solve_time_s = dso_output.solve_time_s
                rec.dso_slack = dso_output.z_slack.copy()

                # Interface Q and voltages from predicted outputs
                n_iface = len(dso_config.interface_trafo_indices)
                n_v_d = len(dso_config.voltage_bus_indices)
                rec.dso_q_interface_mvar = dso_output.y_predicted[:n_iface].copy()
                rec.dso_voltages_pu = dso_output.y_predicted[n_iface:n_iface + n_v_d].copy()

                # Apply DSO controls to COMBINED PLANT
                apply_dso_controls_to_combined(
                    combined_net, dso_output, dso_config
                )

                # Send capability back to TSO
                cap_msg = dso.generate_capability_message("tso_main", dso_meas)
                tso.receive_capability(cap_msg)

            except RuntimeError as e:
                if verbose:
                    print(f"  [min {minute:3d}] DSO FAILED: {e}")

        # -- Re-run power flow on COMBINED PLANT after control actions -----
        try:
            pp.runpp(combined_net, run_control=False, calculate_voltage_angles=True)
        except Exception as e:
            if verbose:
                print(f"  [min {minute:3d}] Combined plant power flow failed: {e}")

        # Record plant measurements from COMBINED network
        tn_bus_mask = [b for b in combined_net.bus.index if b not in dn_buses]
        rec.plant_tn_voltages_pu = combined_net.res_bus.loc[
            [b for b in tso_v_buses if b in tn_bus_mask], "vm_pu"
        ].values.copy()
        
        dn_bus_mask = [b for b in combined_net.bus.index if b in dn_buses]
        rec.plant_dn_voltages_pu = combined_net.res_bus.loc[
            [b for b in dso_v_buses if b in dn_bus_mask], "vm_pu"
        ].values.copy()

        log.append(rec)

        # -- Verbose per-iteration output ----------------------------------
        if verbose and (run_tso or run_dso):
            controllers_active = []
            if run_tso:
                controllers_active.append("TSO")
            if run_dso:
                controllers_active.append("DSO")
            print(f"  [min {minute:3d}]  Active: {'+'.join(controllers_active)}")

            if run_tso and rec.tso_voltages_pu is not None:
                v_mean = np.mean(rec.plant_tn_voltages_pu)
                v_max = np.max(rec.plant_tn_voltages_pu)
                v_min = np.min(rec.plant_tn_voltages_pu)
                print(f"           TN V (plant):  min={v_min:.4f}  mean={v_mean:.4f}  max={v_max:.4f} p.u."
                      f"   (target={v_setpoint_pu:.3f})")
                print(f"           TSO obj={rec.tso_objective:.4e}  status={rec.tso_solver_status}"
                      f"  t_solve={rec.tso_solve_time_s:.3f}s")
                if rec.tso_q_der_mvar is not None:
                    print(f"           TSO Q_DER = {np.array2string(rec.tso_q_der_mvar, precision=2, suppress_small=True)} Mvar")
                if rec.tso_q_pcc_set_mvar is not None:
                    print(f"           TSO Q_PCC_set = {np.array2string(rec.tso_q_pcc_set_mvar, precision=2, suppress_small=True)} Mvar")
                if rec.tso_oltc_taps is not None and len(rec.tso_oltc_taps) > 0:
                    print(f"           TSO OLTC taps = {rec.tso_oltc_taps}")
                if rec.tso_shunt_states is not None and len(rec.tso_shunt_states) > 0:
                    print(f"           TSO shunt states = {rec.tso_shunt_states}")

            if run_dso and rec.dso_voltages_pu is not None:
                v_d_mean = np.mean(rec.plant_dn_voltages_pu)
                v_d_max = np.max(rec.plant_dn_voltages_pu)
                v_d_min = np.min(rec.plant_dn_voltages_pu)
                print(f"           DN V (plant):  min={v_d_min:.4f}  mean={v_d_mean:.4f}  max={v_d_max:.4f} p.u.")
                print(f"           DSO obj={rec.dso_objective:.4e}  status={rec.dso_solver_status}"
                      f"  t_solve={rec.dso_solve_time_s:.3f}s")
                if rec.dso_q_der_mvar is not None:
                    print(f"           DSO Q_DER = {np.array2string(rec.dso_q_der_mvar, precision=2, suppress_small=True)} Mvar")
                if rec.dso_q_interface_mvar is not None:
                    print(f"           DSO Q_iface (plant) = {np.array2string(rec.dso_q_interface_mvar, precision=2, suppress_small=True)} Mvar")
                if rec.dso_oltc_taps is not None and len(rec.dso_oltc_taps) > 0:
                    print(f"           DSO OLTC taps = {rec.dso_oltc_taps}")

            print()

    return log


# ===============================================================================
#  SUMMARY PRINTING
# ===============================================================================

def print_summary(
    v_setpoint: float,
    log: List[IterationRecord],
    tso_v_buses: List[int],
) -> None:
    """Print a summary table for a completed scenario."""
    print()
    print("=" * 72)
    print(f"  SCENARIO SUMMARY  --  V_setpoint = {v_setpoint:.3f} p.u.")
    print("=" * 72)

    # Final TN voltages
    final = log[-1]
    if final.plant_tn_voltages_pu is not None:
        v = final.plant_tn_voltages_pu
        print(f"  Final TN voltages (plant):  min={np.min(v):.4f}  mean={np.mean(v):.4f}"
              f"  max={np.max(v):.4f} p.u.")
        print(f"  Voltage error (max): {np.max(np.abs(v - v_setpoint)):.4f} p.u.")

    # Count active iterations
    n_tso = sum(1 for r in log if r.tso_active)
    n_dso = sum(1 for r in log if r.dso_active)
    print(f"  TSO steps: {n_tso},  DSO steps: {n_dso}")

    # Final control variables
    tso_recs = [r for r in log if r.tso_active and r.tso_voltages_pu is not None]
    if tso_recs:
        last_tso = tso_recs[-1]
        if last_tso.tso_q_der_mvar is not None:
            print(f"  Final TSO Q_DER:       {np.array2string(last_tso.tso_q_der_mvar, precision=2)} Mvar")
        if last_tso.tso_q_pcc_set_mvar is not None:
            print(f"  Final TSO Q_PCC_set:   {np.array2string(last_tso.tso_q_pcc_set_mvar, precision=2)} Mvar")
        if last_tso.tso_oltc_taps is not None and len(last_tso.tso_oltc_taps) > 0:
            print(f"  Final TSO OLTC taps:   {last_tso.tso_oltc_taps}")
        if last_tso.tso_shunt_states is not None and len(last_tso.tso_shunt_states) > 0:
            print(f"  Final TSO shunt:       {last_tso.tso_shunt_states}")

    dso_recs = [r for r in log if r.dso_active and r.dso_voltages_pu is not None]
    if dso_recs:
        last_dso = dso_recs[-1]
        if last_dso.dso_q_der_mvar is not None:
            print(f"  Final DSO Q_DER:       {np.array2string(last_dso.dso_q_der_mvar, precision=2)} Mvar")
        if last_dso.dso_oltc_taps is not None and len(last_dso.dso_oltc_taps) > 0:
            print(f"  Final DSO OLTC taps:   {last_dso.dso_oltc_taps}")

    # Convergence trace: voltage error over TSO iterations
    if tso_recs:
        print()
        print("  TSO voltage convergence trace:")
        print(f"  {'min':>5s}  {'V_min':>8s}  {'V_mean':>8s}  {'V_max':>8s}  {'|err|_max':>10s}  {'obj':>12s}  {'status':>18s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*18}")
        for r in tso_recs:
            v = r.plant_tn_voltages_pu
            err = np.max(np.abs(v - v_setpoint))
            print(f"  {r.minute:5d}  {np.min(v):8.4f}  {np.mean(v):8.4f}  "
                  f"{np.max(v):8.4f}  {err:10.4f}  {r.tso_objective:12.4e}  "
                  f"{r.tso_solver_status:>18s}")

    print("=" * 72)
    print()


# ===============================================================================
#  ENTRY POINT
# ===============================================================================

def main() -> None:
    """Run cascade scenarios for voltage setpoints."""
    scenarios = [1.06]
    use_numerical = False  # Toggle: True for numerical, False for analytical

    all_results: Dict[float, List[IterationRecord]] = {}

    for v_set in scenarios:
        print()
        print("#" * 72)
        print(f"#  SCENARIO: V_setpoint = {v_set:.2f} p.u.")
        print("#" * 72)
        print()

        log = run_cascade(
            v_setpoint_pu=v_set,
            n_minutes=180,
            dso_period_min=15,
            tso_period_min=1,
            alpha=0.001,
            use_numerical_sensitivities=use_numerical,
            verbose=True,
        )
        all_results[v_set] = log

        print_summary(v_set, log, tso_v_buses=[])

    # -- Cross-scenario comparison -----------------------------------------
    print()
    print("=" * 72)
    print("  CROSS-SCENARIO COMPARISON  (final TN voltages from plant)")
    print("=" * 72)
    print(f"  {'V_set':>6s}  {'V_min':>8s}  {'V_mean':>8s}  {'V_max':>8s}  {'|err|_max':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for v_set, log in all_results.items():
        final = log[-1]
        if final.plant_tn_voltages_pu is not None:
            v = final.plant_tn_voltages_pu
            err = np.max(np.abs(v - v_set))
            print(f"  {v_set:6.3f}  {np.min(v):8.4f}  {np.mean(v):8.4f}  "
                  f"{np.max(v):8.4f}  {err:10.4f}")

    print("=" * 72)


if __name__ == "__main__":
    main()
