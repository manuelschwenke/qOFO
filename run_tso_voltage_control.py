#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSO-Only OFO Voltage Controller
===============================

This script runs a *single-layer* Online Feedback Optimisation (OFO)
controller on the **transmission system only**.  It does *not* include
any DSO controllers and it does *not* control reactive power at the
TSO–DSO interfaces (PCCs).

Control objective
-----------------
Track a *uniform* voltage setpoint of 1.05 p.u. at all monitored
380 kV buses by actuating:

    - Reactive power of transmission-connected DER (TN sgens)
    - OLTC tap positions of synchronous machine transformers
    - States of transmission-level switchable shunts (MSC / MSR)

The controller is executed in closed loop with pandapower as the
plant model, using Jacobian-based sensitivities as in the PSCC 2026
and CIGRÉ 2026 formulations.

Author: Manuel Schwenke / Claude Code
Date: 2026-02-10
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
import pandapower as pp

# -- qOFO imports --------------------------------------------------------------
from network.build_tuda_net import build_tuda_net, NetworkMetadata
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from controller.base_controller import OFOParameters, ControllerOutput
from controller.tso_controller import TSOController, TSOControllerConfig
from sensitivity.jacobian import JacobianSensitivities


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
#  HELPER: Extract Measurement from a converged TN network
# ==============================================================================

def measurement_from_tn(
    net: pp.pandapowerNet,
    tso_config: TSOControllerConfig,
    iteration: int,
) -> Measurement:
    """
    Build a TSO Measurement from the converged *combined* TN network.

    PCC reactive power is *not* included as an output or control
    variable in this TSO-only test.  The corresponding fields in
    Measurement are returned as empty arrays.
    """
    # All bus indices and voltage magnitudes
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Branch currents: monitored TN lines
    line_idx = np.array(tso_config.current_line_indices, dtype=np.int64)
    i_ka = np.zeros(len(line_idx), dtype=np.float64)
    for k, li in enumerate(line_idx):
        if li not in net.res_line.index:
            raise ValueError(f"Line {li} not found in net.res_line.")
        i_ka[k] = float(net.res_line.at[li, "i_from_ka"])

    # Interface transformers (PCC) are intentionally *ignored* here.
    iface_trafo = np.zeros(0, dtype=np.int64)
    q_iface = np.zeros(0, dtype=np.float64)

    # DER Q (transmission-connected sgens)
    der_bus = np.array(tso_config.der_bus_indices, dtype=np.int64)
    der_q = np.zeros(len(der_bus), dtype=np.float64)
    for k, bus in enumerate(der_bus):
        sgen_mask = net.sgen["bus"] == bus
        if not sgen_mask.any():
            raise ValueError(
                f"DER sgen at bus {bus} not found in TN net.sgen."
            )
        sidx = net.sgen.index[sgen_mask][0]
        der_q[k] = float(net.res_sgen.at[sidx, "q_mvar"])

    # OLTC tap positions (machine transformers)
    oltc_idx = np.array(tso_config.oltc_trafo_indices, dtype=np.int64)
    oltc_taps = np.zeros(len(oltc_idx), dtype=np.int64)
    for k, tidx in enumerate(oltc_idx):
        if tidx not in net.trafo.index:
            raise ValueError(
                f"Machine transformer {tidx} not found in net.trafo."
            )
        oltc_taps[k] = int(net.trafo.at[tidx, "tap_pos"])

    # Shunt states at TN shunt buses
    shunt_bus = np.array(tso_config.shunt_bus_indices, dtype=np.int64)
    shunt_states = np.zeros(len(shunt_bus), dtype=np.int64)
    for k, sb in enumerate(shunt_bus):
        mask = net.shunt["bus"] == sb
        if not mask.any():
            raise ValueError(
                f"TN shunt at bus {sb} not found in net.shunt."
            )
        sidx = net.shunt.index[mask][0]
        shunt_states[k] = int(net.shunt.at[sidx, "step"])

    # generator AVR setpoints
    gen_idx = np.array(tso_config.gen_indices, dtype=np.int64)
    gen_vm = np.zeros(len(gen_idx), dtype=np.float64)
    for k, g in enumerate(gen_idx):
        if g not in net.gen.index:
            raise ValueError(f"Generator index {g} not found in net.gen.")
        gen_vm[k] = float(net.gen.at[g, "vm_pu"])

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
#  HELPER: Apply TSO controls back to pandapower network
# ==============================================================================

def apply_tso_controls(
    net: pp.pandapowerNet,
    tso_output: ControllerOutput,
    tso_config: TSOControllerConfig,
) -> None:
    """
    Apply TSO control outputs to the combined TN pandapower network.

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
        sgen_mask = net.sgen["bus"] == bus
        if not sgen_mask.any():
            raise ValueError(
                f"DER sgen at bus {bus} not found in TN net.sgen."
            )
        sidx = net.sgen.index[sgen_mask][0]
        net.sgen.at[sidx, "q_mvar"] = float(u[k])

    # 2) PCC Q setpoints are intentionally omitted (n_pcc = 0).

    # 3) Generator AVR setpoints
    avr_start = n_der + n_pcc
    avr_end = avr_start + n_gen
    avr_values = tso_output.u_new[avr_start:avr_end]
    for g_idx, vm in zip(tso_config.gen_indices, avr_values):
        if g_idx not in net.gen.index:
            raise ValueError(f"Generator index {g_idx} not found in net.gen.")
        net.gen.at[g_idx, "vm_pu"] = float(vm)

    # 4) Machine transformer OLTC tap positions
    oltc_start = n_der + n_pcc + n_gen
    for k, trafo_idx in enumerate(tso_config.oltc_trafo_indices):
        if trafo_idx not in net.trafo.index:
            raise ValueError(
                f"Machine transformer {trafo_idx} not found in net.trafo."
            )
        tap_val = int(np.round(u[oltc_start + k]))
        net.trafo.at[trafo_idx, "tap_pos"] = tap_val

    # 5) Shunt states
    shunt_start = oltc_start + n_oltc
    n_shunt = len(tso_config.shunt_bus_indices)
    for k, shunt_bus in enumerate(tso_config.shunt_bus_indices):
        mask = net.shunt["bus"] == shunt_bus
        if not mask.any():
            raise ValueError(
                f"TN shunt at bus {shunt_bus} not found in net.shunt."
            )
        sidx = net.shunt.index[mask][0]
        step_val = int(np.round(u[shunt_start + k]))
        net.shunt.at[sidx, "step"] = step_val


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
    verbose: bool = True,
) -> List[TSOIterationRecord]:
    """
    Run a TSO-only OFO voltage controller loop on the combined network.

    Parameters
    ----------
    v_setpoint_pu : float
        Uniform voltage setpoint for all monitored 380 kV buses [p.u.].
    n_iterations : int
        Number of OFO iterations to simulate.
    alpha : float
        OFO step size (gain).
    verbose : bool
        If True, print per-iteration summary.

    Returns
    -------
    log : list[TSOIterationRecord]
        One record per OFO iteration.
    """
    if verbose:
        print("=" * 72)
        print(f"  TSO-ONLY OFO SIMULATION -- V_setpoint = {v_setpoint_pu:.3f} p.u.")
        print("=" * 72)
        print("[1/4] Building combined 380/110/20 kV network ...")

    # 1) Build combined network (TN+DN) but operate only on TN actuators
    net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)
    # We keep the full network; the TSO controller will only see
    # transmission-connected actuators and 380 kV voltage measurements.

    if verbose:
        print("[2/4] Identifying TSO actuators and monitored quantities ...")

    # Transmission-connected DER: TN sgens (exclude DN subnet)
    tso_der_buses: List[int] = []
    for sidx in net.sgen.index:
        subnet = str(net.sgen.at[sidx, "subnet"])
        if subnet != "TN":
            continue
        tso_der_buses.append(int(net.sgen.at[sidx, "bus"]))

    # Monitored voltages: all 380 kV buses (EHV, subnet "TN")
    tso_v_buses: List[int] = sorted(
        int(b)
        for b in net.bus.index
        if float(net.bus.at[b, "vn_kv"]) >= 300.0
        and str(net.bus.at[b, "subnet"]) == "TN"
    )

    if len(tso_v_buses) == 0:
        raise RuntimeError("No 380 kV buses found for TSO voltage monitoring.")

    # Monitored currents: all TN lines
    tso_lines: List[int] = sorted(
        int(li)
        for li in net.line.index
        if str(net.line.at[li, "subnet"]) == "TN"
    )

    # Machine transformers (OLTCs) from metadata
    tso_oltc_candidates = list(meta.machine_trafo_indices)

    # TN shunts (MSC/MSR) from metadata
    tso_shunt_bus_candidates: List[int] = []
    tso_shunt_q_candidates: List[float] = []
    for sidx in meta.tn_shunt_indices:
        if sidx not in net.shunt.index:
            raise RuntimeError(
                f"TN shunt index {sidx} from metadata not found in net.shunt."
            )
        tso_shunt_bus_candidates.append(int(net.shunt.at[sidx, "bus"]))
        tso_shunt_q_candidates.append(float(net.shunt.at[sidx, "q_mvar"]))

    # synchronous generators and their associated grid-side buses
    tso_gen_indices: List[int] = []
    tso_gen_bus_indices: List[int] = []
    for g in net.gen.index:
        tso_gen_indices.append(int(g))
        gen_lv_bus = int(net.gen.at[g, "bus"])
        # Find the machine transformer that connects this LV bus to the grid
        mt_mask = (
            (net.trafo["lv_bus"] == gen_lv_bus)
            & net.trafo["name"].astype(str).str.startswith("MachineTrf|")
        )
        if not mt_mask.any():
            raise RuntimeError(
                f"No machine transformer found for generator {g} "
                f"(bus {gen_lv_bus})."
            )
        mt_idx = int(net.trafo.index[mt_mask][0])
        hv_bus = int(net.trafo.at[mt_idx, "hv_bus"])
        tso_gen_bus_indices.append(hv_bus)

    # Probe Jacobian-based sensitivities to select valid actuators and buses
    probe_sens = JacobianSensitivities(net)
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
        print("[3/4] Creating TSO controller and sensitivity model ...")

    # OFO tuning for TSO (same structure as in run_cascade.py)
    ofo_params_tso = OFOParameters(
        alpha=alpha,
        g_w=0.00001,
        g_z=1_000_000.0,
        g_s=100.0,
        g_u=0.0001,
    )

    # Network state and sensitivities
    tso_trafo_idx = np.array(tso_oltc, dtype=np.int64)
    tso_ns = network_state_from_net(net, tso_trafo_idx, source_case="TN")

    tso_sens = JacobianSensitivities(net)

    # Actuator bounds for DER, OLTC and shunts
    der_indices_arr = np.array(tso_der_buses, dtype=np.int64)
    der_s_rated = []
    der_p_max = []
    for sidx in net.sgen.index:
        subnet = str(net.sgen.at[sidx, "subnet"])
        if subnet != "TN":
            continue
        der_s_rated.append(float(net.sgen.at[sidx, "sn_mva"]))
        der_p_max.append(float(net.sgen.at[sidx, "p_mw"]))

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
            [int(net.trafo.at[t, "tap_min"]) for t in tso_oltc],
            dtype=np.int64,
        ),
        oltc_tap_max=np.array(
            [int(net.trafo.at[t, "tap_max"]) for t in tso_oltc],
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

    # Initial TSO measurement and controller initialisation
    if verbose:
        print("[4/4] Initialising TSO controller from converged power flow ...")

    # Ensure base-case power flow is converged on the combined network
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    tso_meas0 = measurement_from_tn(net, tso_config, iteration=0)
    tso.initialise(tso_meas0)

    if verbose:
        print()
        print(
            f"Running TSO-only OFO for {n_iterations} iterations "
            f"(alpha = {alpha:.4f}) ..."
        )
        print()

    log: List[TSOIterationRecord] = []

    for it in range(1, n_iterations + 1):
        rec = TSOIterationRecord(iteration=it)

        # New measurement at current operating point
        meas = measurement_from_tn(net, tso_config, iteration=it)

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

        # Apply controls to plant
        apply_tso_controls(net, tso_output, tso_config)

        # Re-run power flow
        try:
            pp.runpp(net, run_control=False, calculate_voltage_angles=True)
        except Exception as e:
            raise RuntimeError(
                f"TN power flow failed at iteration {it}: {e}"
            )

        # Record plant voltages at monitored EHV buses
        rec.plant_tn_voltages_pu = net.res_bus.loc[
            tso_v_buses, "vm_pu"
        ].values.astype(np.float64).copy()
        
        # Record generator reactive power outputs from power flow results
        rec.plant_gen_q_mvar = np.zeros(len(tso_config.gen_indices), dtype=np.float64)
        for k, g in enumerate(tso_config.gen_indices):
            if g not in net.res_gen.index:
                raise ValueError(f"Generator {g} not found in net.res_gen.")
            rec.plant_gen_q_mvar[k] = float(net.res_gen.at[g, "q_mvar"])

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

    print()
    print("#" * 72)
    print(f"#  TSO-ONLY SCENARIO: V_setpoint = {v_set:.2f} p.u.")
    print("#" * 72)
    print()

    log = run_tso_voltage_control(
        v_setpoint_pu=v_set,
        n_iterations=n_it,
        alpha=0.01,
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
