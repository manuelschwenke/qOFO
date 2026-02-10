#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSO-Only OFO Voltage Controller
===============================

This script runs a *single-layer* Online Feedback Optimisation (OFO)
controller on the **transmission system only**.  It does *not* include
any DSO controllers and it does *not* control reactive power at the
TSO–DSO interfaces (PCCs).

Network Architecture
--------------------
1. **Combined network** (TN + DN): The "real plant"
   - Controls are applied here
   - Measurements are taken from here
   - Power flow is executed here
   
2. **TN network model**: Used for sensitivity calculations
   - Network states are created from this
   - Jacobian sensitivities are computed from this
   - Not used for actual control or measurement

Control objective
-----------------
Track a *uniform* voltage setpoint of 1.05 p.u. at all monitored
380 kV buses by actuating:

    - Reactive power of transmission-connected DER (TN sgens)
    - OLTC tap positions of synchronous machine transformers
    - States of transmission-level switchable shunts (MSC / MSR)
    - AVR setpoints of synchronous generators

The controller is executed in closed loop with pandapower as the
plant model, using Jacobian-based sensitivities as in the PSCC 2026
and CIGRÉ 2026 formulations.

Author: Manuel Schwenke
Date: 2026-02-10
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union

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
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.numerical import NumericalSensitivities

# Type alias for sensitivity calculator
SensitivityCalculator = Union[JacobianSensitivities, NumericalSensitivities]


# ==============================================================================
#  HELPER: Build NetworkState from a converged pandapower network
# ==============================================================================

def network_state_from_net(
    net: pp.pandapowerNet,
    trafo_indices: NDArray[np.int64],
    source_case: str = "TN",
    iteration: int = 0,
) -> NetworkState:
    """
    Create a NetworkState snapshot from a converged pandapower network.

    This mirrors the helper in run_cascade.py but is restricted to the
    transmission system use case.
    """
    bus_indices = np.array(net.bus.index, dtype=np.int64)

    vm = net.res_bus.loc[bus_indices, "vm_pu"].values.astype(np.float64)
    va = np.deg2rad(
        net.res_bus.loc[bus_indices, "va_degree"].values.astype(np.float64)
    )

    # Slack bus (ext_grid) index
    if net.ext_grid.empty:
        raise RuntimeError("No external grid defined in TN network.")
    slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])

    # PV buses: synchronous generators (gen) with vm_pu set and in service
    pv_buses = np.array(
        [
            int(net.gen.at[g, "bus"])
            for g in net.gen.index
            if bool(net.gen.at[g, "in_service"])
        ],
        dtype=np.int64,
    )

    # PQ buses = remaining buses
    all_special = set([slack_bus]) | set(pv_buses.tolist())
    pq_buses = np.array(
        [int(b) for b in bus_indices if int(b) not in all_special],
        dtype=np.int64,
    )

    # Tap positions for specified transformers
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


# ==============================================================================
#  HELPER: Extract Measurement from COMBINED network (real plant)
# ==============================================================================

def measurement_from_combined(
    combined_net: pp.pandapowerNet,
    split: SplitResult,
    tso_config: TSOControllerConfig,
    iteration: int,
) -> Measurement:
    """
    Build a TSO Measurement from the converged COMBINED network (real plant).

    PCC reactive power is *not* included as an output or control
    variable in this TSO-only test.  The corresponding fields in
    Measurement are returned as empty arrays.
    
    This represents taking measurements from the real, physical system.
    """
    # All TN bus indices from combined network (exclude DN-only buses)
    tn_bus_indices = [
        b for b in combined_net.bus.index
        if b not in split.dn_only_bus_indices
    ]
    all_bus = np.array(sorted(tn_bus_indices), dtype=np.int64)
    vm = combined_net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Branch currents: monitored TN lines from combined network
    line_idx = np.array(tso_config.current_line_indices, dtype=np.int64)
    i_ka = np.zeros(len(line_idx), dtype=np.float64)
    for k, li in enumerate(line_idx):
        if li not in combined_net.res_line.index:
            raise ValueError(f"Line {li} not found in combined_net.res_line.")
        i_ka[k] = float(combined_net.res_line.at[li, "i_from_ka"])

    # Interface transformers (PCC) are intentionally *ignored* here.
    iface_trafo = np.zeros(0, dtype=np.int64)
    q_iface = np.zeros(0, dtype=np.float64)

    # DER Q (transmission-connected sgens, exclude boundary sgens)
    der_bus = np.array(tso_config.der_bus_indices, dtype=np.int64)
    der_q = np.zeros(len(der_bus), dtype=np.float64)
    for k, bus in enumerate(der_bus):
        # Find sgen at this bus that is NOT a boundary sgen
        sgen_mask = (
            (combined_net.sgen["bus"] == bus) &
            ~combined_net.sgen["name"].astype(str).str.startswith("BoundarySgen|")
        )
        if not sgen_mask.any():
            raise ValueError(
                f"DER sgen at bus {bus} not found in combined_net.sgen."
            )
        # Sum all valid sgens at this bus
        for sidx in combined_net.sgen.index[sgen_mask]:
            der_q[k] += float(combined_net.res_sgen.at[sidx, "q_mvar"])

    # OLTC tap positions (machine transformers)
    oltc_idx = np.array(tso_config.oltc_trafo_indices, dtype=np.int64)
    oltc_taps = np.zeros(len(oltc_idx), dtype=np.int64)
    for k, tidx in enumerate(oltc_idx):
        if tidx not in combined_net.trafo.index:
            raise ValueError(
                f"Machine transformer {tidx} not found in combined_net.trafo."
            )
        oltc_taps[k] = int(combined_net.trafo.at[tidx, "tap_pos"])

    # Shunt states at TN shunt buses
    shunt_bus = np.array(tso_config.shunt_bus_indices, dtype=np.int64)
    shunt_states = np.zeros(len(shunt_bus), dtype=np.int64)
    for k, sb in enumerate(shunt_bus):
        mask = combined_net.shunt["bus"] == sb
        if not mask.any():
            raise ValueError(
                f"TN shunt at bus {sb} not found in combined_net.shunt."
            )
        sidx = combined_net.shunt.index[mask][0]
        shunt_states[k] = int(combined_net.shunt.at[sidx, "step"])

    # generator AVR setpoints
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


# ==============================================================================
#  HELPER: Apply TSO controls to COMBINED network (real plant)
# ==============================================================================

def apply_tso_controls(
    combined_net: pp.pandapowerNet,
    tso_output: ControllerOutput,
    tso_config: TSOControllerConfig,
) -> None:
    """
    Apply TSO control outputs to the COMBINED pandapower network (real plant).

    This implementation *does not* write any PCC setpoints, in line
    with the TSO-only test setup.
    """
    u = tso_output.u_new

    n_der = len(tso_config.der_bus_indices)
    n_pcc = len(tso_config.pcc_trafo_indices)  # will be zero here
    n_gen = len(tso_config.gen_indices)
    n_oltc = len(tso_config.oltc_trafo_indices)

    # 1) Transmission-connected DER Q setpoints
    for k, bus in enumerate(tso_config.der_bus_indices):
        sgen_mask = (
            (combined_net.sgen["bus"] == bus) &
            ~combined_net.sgen["name"].astype(str).str.startswith("BoundarySgen|")
        )
        if not sgen_mask.any():
            raise ValueError(
                f"DER sgen at bus {bus} not found in combined_net.sgen."
            )
        # Set first valid sgen at this bus
        sidx = combined_net.sgen.index[sgen_mask][0]
        combined_net.sgen.at[sidx, "q_mvar"] = float(u[k])

    # 2) PCC Q setpoints are intentionally omitted (n_pcc = 0).

    # 3) Generator AVR setpoints
    avr_start = n_der + n_pcc
    avr_end = avr_start + n_gen
    avr_values = tso_output.u_new[avr_start:avr_end]
    for g_idx, vm in zip(tso_config.gen_indices, avr_values):
        if g_idx not in combined_net.gen.index:
            raise ValueError(f"Generator index {g_idx} not found in combined_net.gen.")
        combined_net.gen.at[g_idx, "vm_pu"] = float(vm)

    # 4) Machine transformer OLTC tap positions
    oltc_start = n_der + n_pcc + n_gen
    for k, trafo_idx in enumerate(tso_config.oltc_trafo_indices):
        if trafo_idx not in combined_net.trafo.index:
            raise ValueError(
                f"Machine transformer {trafo_idx} not found in combined_net.trafo."
            )
        tap_val = int(np.round(u[oltc_start + k]))
        combined_net.trafo.at[trafo_idx, "tap_pos"] = tap_val

    # 5) Shunt states
    shunt_start = oltc_start + n_oltc
    n_shunt = len(tso_config.shunt_bus_indices)
    for k, shunt_bus in enumerate(tso_config.shunt_bus_indices):
        mask = combined_net.shunt["bus"] == shunt_bus
        if not mask.any():
            raise ValueError(
                f"TN shunt at bus {shunt_bus} not found in combined_net.shunt."
            )
        sidx = combined_net.shunt.index[mask][0]
        step_val = int(np.round(u[shunt_start + k]))
        combined_net.shunt.at[sidx, "step"] = step_val


# ==============================================================================
#  ITERATION LOG
# ==============================================================================

@dataclass
class TSOIterationRecord:
    """
    Record of one TSO-only OFO iteration.
    """
    iteration: int

    # Controller outputs
    tso_voltages_pu: Optional[NDArray[np.float64]] = None
    tso_q_der_mvar: Optional[NDArray[np.float64]] = None
    tso_gen_avr_setpoints_pu: Optional[NDArray[np.float64]] = None
    tso_oltc_taps: Optional[NDArray[np.int64]] = None
    tso_shunt_states: Optional[NDArray[np.int64]] = None
    tso_objective: Optional[float] = None
    tso_solver_status: Optional[str] = None
    tso_solve_time_s: Optional[float] = None
    tso_slack: Optional[NDArray[np.float64]] = None

    # Plant measurements after power flow
    plant_tn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_gen_q_mvar: Optional[NDArray[np.float64]] = None


# ==============================================================================
#  MAIN TSO-ONLY RUNNER
# ==============================================================================

def run_tso_voltage_control(
    v_setpoint_pu: float = 1.05,
    n_iterations: int = 60,
    alpha: float = 0.01,
    use_numerical_sensitivities: bool = False,
    verbose: bool = True,
) -> List[TSOIterationRecord]:
    """
    Run a TSO-only OFO voltage controller loop.
    
    Architecture:
    - Combined network (TN+DN): Real plant (apply controls, measure, run PF)
    - TN network model: Used only for sensitivity calculations

    Parameters
    ----------
    v_setpoint_pu : float
        Uniform voltage setpoint for all monitored 380 kV buses [p.u.].
    n_iterations : int
        Number of OFO iterations to simulate.
    alpha : float
        OFO step size (gain).
    use_numerical_sensitivities : bool, optional
        If True, use numerical finite-difference sensitivities instead of
        analytical Jacobian-based sensitivities (default: False).
    verbose : bool
        If True, print per-iteration summary.

    Returns
    -------
    log : list[TSOIterationRecord]
        One record per OFO iteration.
    """
    sens_method = "NUMERICAL" if use_numerical_sensitivities else "ANALYTICAL"
    
    if verbose:
        print("=" * 72)
        print(f"  TSO-ONLY OFO SIMULATION -- V_setpoint = {v_setpoint_pu:.3f} p.u.")
        print("=" * 72)
        print(f"  Sensitivity method: {sens_method}")
        print("=" * 72)
        print("[1/6] Building combined 380/110/20 kV network (real plant) ...")

    # 1) Build combined network - this is the REAL PLANT
    combined_net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

    if verbose:
        print("[2/6] Running converged power flow on combined network ...")

    # Ensure base-case power flow is converged
    pp.runpp(combined_net, run_control=True, calculate_voltage_angles=True)

    if verbose:
        print("[3/6] Splitting network to create TN model (for sensitivities) ...")

    # 2) Split to create TN network MODEL (for sensitivities only)
    split_result = split_network(
        combined_net,
        meta,
        dn_slack_coupler_index=0,
    )

    tn_net_model = split_result.tn_net  # This is the MODEL, not the plant!

    if verbose:
        print("[4/6] Identifying TSO actuators and monitored quantities ...")

    # Transmission-connected DER: TN sgens (exclude boundary sgens)
    # Identify from combined network (real plant)
    tso_der_buses: List[int] = []
    for sidx in combined_net.sgen.index:
        bus = int(combined_net.sgen.at[sidx, "bus"])
        # TN sgens are those not in DN-only buses
        if bus not in split_result.dn_only_bus_indices:
            # Not a boundary sgen
            if not combined_net.sgen.at[sidx, "name"].startswith("BoundarySgen|"):
                if bus not in tso_der_buses:
                    tso_der_buses.append(bus)

    # Monitored voltages: all 380 kV buses (EHV) in TN
    tso_v_buses: List[int] = sorted(
        int(b)
        for b in combined_net.bus.index
        if float(combined_net.bus.at[b, "vn_kv"]) >= 300.0
        and b not in split_result.dn_only_bus_indices
    )

    if len(tso_v_buses) == 0:
        raise RuntimeError("No 380 kV buses found for TSO voltage monitoring.")

    # Monitored currents: all TN lines
    tso_lines: List[int] = sorted(
        int(li)
        for li in combined_net.line.index
        if li in tn_net_model.line.index
    )

    # Machine transformers (OLTCs) from metadata
    tso_oltc_candidates = list(meta.machine_trafo_indices)

    # TN shunts (MSC/MSR) from metadata
    tso_shunt_bus_candidates: List[int] = []
    tso_shunt_q_candidates: List[float] = []
    for sidx in meta.tn_shunt_indices:
        if sidx not in combined_net.shunt.index:
            raise RuntimeError(
                f"TN shunt index {sidx} from metadata not found in combined_net.shunt."
            )
        tso_shunt_bus_candidates.append(int(combined_net.shunt.at[sidx, "bus"]))
        tso_shunt_q_candidates.append(float(combined_net.shunt.at[sidx, "q_mvar"]))

    # synchronous generators and their associated grid-side buses
    tso_gen_indices: List[int] = []
    tso_gen_bus_indices: List[int] = []
    for g in combined_net.gen.index:
        tso_gen_indices.append(int(g))
        gen_lv_bus = int(combined_net.gen.at[g, "bus"])
        # Find the machine transformer that connects this LV bus to the grid
        mt_mask = (
            (combined_net.trafo["lv_bus"] == gen_lv_bus)
            & combined_net.trafo["name"].astype(str).str.startswith("MachineTrf|")
        )
        if not mt_mask.any():
            raise RuntimeError(
                f"No machine transformer found for generator {g} "
                f"(bus {gen_lv_bus})."
            )
        mt_idx = int(combined_net.trafo.index[mt_mask][0])
        hv_bus = int(combined_net.trafo.at[mt_idx, "hv_bus"])
        tso_gen_bus_indices.append(hv_bus)

    if verbose:
        print("[5/6] Computing sensitivities from TN model ...")

    # Probe sensitivities from TN MODEL (not combined plant)
    if use_numerical_sensitivities:
        probe_sens: SensitivityCalculator = NumericalSensitivities(tn_net_model)
    else:
        probe_sens = JacobianSensitivities(tn_net_model)
    
    H_probe, m_probe = probe_sens.build_sensitivity_matrix_H(
        der_bus_indices=tso_der_buses,
        observation_bus_indices=tso_v_buses,
        line_indices=tso_lines,
        oltc_trafo_indices=tso_oltc_candidates,
        shunt_bus_indices=tso_shunt_bus_candidates,
        shunt_q_steps_mvar=tso_shunt_q_candidates,
    )

    # Select only elements that produced valid sensitivities
    tso_oltc = list(m_probe.get("oltc_trafos", []))
    tso_v_buses = list(m_probe.get("obs_buses", tso_v_buses))
    tso_shunt_buses = list(m_probe.get("shunt_buses", []))
    tso_shunt_q_steps = [
        tso_shunt_q_candidates[tso_shunt_bus_candidates.index(b)]
        for b in tso_shunt_buses
    ]

    # No PCC control in this TSO-only test
    pcc_trafo_indices: List[int] = []
    pcc_dso_ids: List[str] = []

    # Voltage setpoints: uniform schedule at all monitored buses
    v_setpoints = np.full(len(tso_v_buses), v_setpoint_pu, dtype=np.float64)

    tso_config = TSOControllerConfig(
        der_bus_indices=tso_der_buses,
        pcc_trafo_indices=pcc_trafo_indices,
        pcc_dso_controller_ids=pcc_dso_ids,
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

    if verbose:
        print("[6/6] Creating TSO controller ...")

    # OFO tuning for TSO (same structure as in run_cascade.py)
    ofo_params_tso = OFOParameters(
        alpha=alpha,
        g_w=0.00001,
        g_z=1_000_000.0,
        g_s=100.0,
        g_u=0.0001,
    )

    # Network state from TN MODEL (for sensitivities)
    tso_trafo_idx = np.array(tso_oltc, dtype=np.int64)
    tso_ns = network_state_from_net(tn_net_model, tso_trafo_idx, source_case="TN")

    # Sensitivities from TN MODEL
    if use_numerical_sensitivities:
        tso_sens: SensitivityCalculator = NumericalSensitivities(tn_net_model)
    else:
        tso_sens = JacobianSensitivities(tn_net_model)

    # Actuator bounds from combined network (real plant ratings)
    der_indices_arr = np.array(tso_der_buses, dtype=np.int64)
    der_s_rated = []
    der_p_max = []
    for sidx in combined_net.sgen.index:
        bus = int(combined_net.sgen.at[sidx, "bus"])
        if bus not in tso_der_buses:
            continue
        if combined_net.sgen.at[sidx, "name"].startswith("BoundarySgen|"):
            continue
        der_s_rated.append(float(combined_net.sgen.at[sidx, "sn_mva"]))
        der_p_max.append(float(combined_net.sgen.at[sidx, "p_mw"]))

    if len(der_s_rated) != len(tso_der_buses):
        raise RuntimeError(
            "Mismatch between DER buses and DER rating list length."
        )

    tso_bounds = ActuatorBounds(
        der_indices=der_indices_arr,
        der_s_rated_mva=np.array(der_s_rated, dtype=np.float64),
        der_p_max_mw=np.array(der_p_max, dtype=np.float64),
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

    if verbose:
        print("Initialising TSO controller from combined network ...")

    # Initial TSO measurement from COMBINED network (real plant)
    tso_meas0 = measurement_from_combined(combined_net, split_result, tso_config, iteration=0)
    tso.initialise(tso_meas0)

    if verbose:
        print()
        print(
            f"Running TSO-only OFO for {n_iterations} iterations "
            f"(alpha = {alpha:.4f}) ..."
        )
        print("  • Controls applied to: COMBINED network (real plant)")
        print("  • Measurements from: COMBINED network (real plant)")
        print("  • Sensitivities from: TN model")
        print()

    log: List[TSOIterationRecord] = []

    for it in range(1, n_iterations + 1):
        rec = TSOIterationRecord(iteration=it)

        # New measurement from COMBINED network (real plant)
        meas = measurement_from_combined(combined_net, split_result, tso_config, iteration=it)

        # OFO step
        try:
            tso_output = tso.step(meas)
        except RuntimeError as e:
            # Explicit failure is desired in this scientific context
            raise RuntimeError(f"TSO OFO step failed at iteration {it}: {e}")

        # Extract control variables
        n_der_t = len(tso_config.der_bus_indices)
        n_pcc_t = len(tso_config.pcc_trafo_indices)  # zero here
        n_gen_t = len(tso_config.gen_indices)
        n_oltc_t = len(tso_config.oltc_trafo_indices)

        rec.tso_q_der_mvar = tso_output.u_new[:n_der_t].copy()
        
        # Extract generator AVR setpoints
        avr_start = n_der_t + n_pcc_t
        avr_end = avr_start + n_gen_t
        rec.tso_gen_avr_setpoints_pu = tso_output.u_new[avr_start:avr_end].copy()
        
        # Extract OLTC taps
        oltc_start = avr_end
        oltc_end = oltc_start + n_oltc_t
        rec.tso_oltc_taps = np.round(
            tso_output.u_new[oltc_start:oltc_end]
        ).astype(np.int64)
        
        # Extract shunt states
        rec.tso_shunt_states = np.round(
            tso_output.u_new[oltc_end:]
        ).astype(np.int64)
        
        rec.tso_objective = tso_output.objective_value
        rec.tso_solver_status = tso_output.solver_status
        rec.tso_solve_time_s = tso_output.solve_time_s
        rec.tso_slack = tso_output.z_slack.copy()

        # Predicted voltages (first n_v outputs)
        n_v = len(tso_config.voltage_bus_indices)
        rec.tso_voltages_pu = tso_output.y_predicted[:n_v].copy()

        # Apply controls to COMBINED network (real plant)
        apply_tso_controls(combined_net, tso_output, tso_config)

        # Re-run power flow on COMBINED network (real plant)
        try:
            pp.runpp(combined_net, run_control=False, calculate_voltage_angles=True)
        except Exception as e:
            raise RuntimeError(
                f"Combined network power flow failed at iteration {it}: {e}"
            )

        # Record plant voltages from COMBINED network at monitored EHV buses
        rec.plant_tn_voltages_pu = combined_net.res_bus.loc[
            tso_v_buses, "vm_pu"
        ].values.astype(np.float64).copy()
        
        # Record generator reactive power outputs from power flow results
        rec.plant_gen_q_mvar = np.zeros(len(tso_config.gen_indices), dtype=np.float64)
        for k, g in enumerate(tso_config.gen_indices):
            if g not in combined_net.res_gen.index:
                raise ValueError(f"Generator {g} not found in combined_net.res_gen.")
            rec.plant_gen_q_mvar[k] = float(combined_net.res_gen.at[g, "q_mvar"])

        log.append(rec)

        # Verbose summary
        if verbose:
            v = rec.plant_tn_voltages_pu
            v_min = float(np.min(v))
            v_mean = float(np.mean(v))
            v_max = float(np.max(v))
            err_max = float(np.max(np.abs(v - v_setpoint_pu)))

            print(
                f"[it {it:3d}] TN V: min={v_min:.4f}  mean={v_mean:.4f}  "
                f"max={v_max:.4f} p.u.  (target={v_setpoint_pu:.3f}, "
                f"|err|_max={err_max:.4f})"
            )
            print(
                f"          obj={rec.tso_objective:.4e}  "
                f"status={rec.tso_solver_status}  "
                f"t_solve={rec.tso_solve_time_s:.3f}s"
            )
            if rec.tso_q_der_mvar is not None:
                print(
                    "          Q_DER = "
                    f"{np.array2string(rec.tso_q_der_mvar, precision=2, suppress_small=True)} "
                    "Mvar"
                )
            if rec.tso_gen_avr_setpoints_pu is not None:
                print(
                    "          Gen AVR setpoints = "
                    f"{np.array2string(rec.tso_gen_avr_setpoints_pu, precision=4, suppress_small=True)} "
                    "p.u."
                )
            if rec.plant_gen_q_mvar is not None:
                print(
                    "          Gen Q output = "
                    f"{np.array2string(rec.plant_gen_q_mvar, precision=2, suppress_small=True)} "
                    "Mvar"
                )
            if rec.tso_oltc_taps is not None:
                print(f"          OLTC taps = {rec.tso_oltc_taps}")
            if rec.tso_shunt_states is not None:
                print(f"          Shunt states = {rec.tso_shunt_states}")
            print()

    return log


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Run a single TSO-only voltage control scenario with V_set = 1.05 p.u.
    """
    v_set = 1.05
    n_it = 120
    use_numerical = False  # Toggle: True for numerical, False for analytical

    print()
    print("#" * 72)
    print(f"#  TSO-ONLY SCENARIO: V_setpoint = {v_set:.2f} p.u.")
    print("#" * 72)
    print()

    log = run_tso_voltage_control(
        v_setpoint_pu=v_set,
        n_iterations=n_it,
        alpha=0.01,
        use_numerical_sensitivities=use_numerical,
        verbose=True,
    )

    final = log[-1]
    if final.plant_tn_voltages_pu is not None:
        v = final.plant_tn_voltages_pu
        v_min = float(np.min(v))
        v_mean = float(np.mean(v))
        v_max = float(np.max(v))
        err_max = float(np.max(np.abs(v - v_set)))

        print()
        print("=" * 72)
        print("  FINAL TSO-ONLY VOLTAGE SUMMARY")
        print("=" * 72)
        print(
            f"  V_set = {v_set:.3f} p.u.,  "
            f"V_min = {v_min:.4f} p.u.,  "
            f"V_mean = {v_mean:.4f} p.u.,  "
            f"V_max = {v_max:.4f} p.u."
        )
        print(f"  Max |V - V_set| = {err_max:.4f} p.u.")
        print("=" * 72)
        print()


if __name__ == "__main__":
    main()
