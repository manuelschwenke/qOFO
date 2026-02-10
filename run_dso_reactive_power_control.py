#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSO-Only OFO Reactive Power Controller
=======================================

This script runs a *single-layer* Online Feedback Optimisation (OFO)
controller on the **distribution system only**.  It does *not* include
the TSO controller and receives fixed reactive power setpoints at the
TSO–DSO interfaces (coupler transformers).

Control objective
-----------------
Track fixed reactive power setpoints Q_set = [0, 0, 0] Mvar at the
three TSO-DSO interface transformers (3-winding couplers) by actuating:

    - Reactive power of distribution-connected DER (DN sgens)
    - OLTC tap positions of 3-winding coupler transformers
    - States of distribution-level switchable shunts

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
from network.split_tn_dn_net import split_network, SplitResult
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from controller.base_controller import OFOParameters, ControllerOutput
from controller.dso_controller import DSOController, DSOControllerConfig
from sensitivity.jacobian import JacobianSensitivities


# ==============================================================================
#  HELPER: Build NetworkState from a converged pandapower network
# ==============================================================================

def network_state_from_net(
    net: pp.pandapowerNet,
    trafo3w_indices: NDArray[np.int64],
    source_case: str = "DN",
    iteration: int = 0,
) -> NetworkState:
    """
    Create a NetworkState snapshot from a converged pandapower network.

    For the DSO, the transformers with OLTCs are 3-winding couplers,
    so we use trafo3w tap positions.
    """
    bus_indices = np.array(net.bus.index, dtype=np.int64)

    vm = net.res_bus.loc[bus_indices, "vm_pu"].values.astype(np.float64)
    va = np.deg2rad(
        net.res_bus.loc[bus_indices, "va_degree"].values.astype(np.float64)
    )

    # Slack bus (ext_grid) index
    if net.ext_grid.empty:
        raise RuntimeError("No external grid defined in DN network.")
    slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])

    # PV buses: none expected in DN (all generators are at TN level)
    pv_buses = np.array([], dtype=np.int64)

    # PQ buses = all non-slack buses
    pq_buses = np.array(
        [int(b) for b in bus_indices if int(b) != slack_bus],
        dtype=np.int64,
    )

    # Tap positions for 3-winding transformers
    tap_pos = np.zeros(len(trafo3w_indices), dtype=np.float64)
    for i, tidx in enumerate(trafo3w_indices):
        if tidx not in net.trafo3w.index:
            raise ValueError(
                f"3-winding transformer {tidx} not found in net.trafo3w."
            )
        tap_pos[i] = float(net.trafo3w.at[tidx, "tap_pos"])

    return NetworkState(
        bus_indices=bus_indices,
        voltage_magnitudes_pu=vm,
        voltage_angles_rad=va,
        slack_bus_index=slack_bus,
        pv_bus_indices=pv_buses,
        pq_bus_indices=pq_buses,
        transformer_indices=trafo3w_indices,
        tap_positions=tap_pos,
        source_case=source_case,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cached_at_iteration=iteration,
    )


# ==============================================================================
#  HELPER: Extract Measurement from a converged DN network
# ==============================================================================

def measurement_from_dn(
    net: pp.pandapowerNet,
    dso_config: DSOControllerConfig,
    iteration: int,
) -> Measurement:
    """
    Build a DSO Measurement from the converged DN network.

    Interface reactive power is measured at the HV side of the 3-winding
    coupler transformers.
    """
    # All bus indices and voltage magnitudes
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # Branch currents: monitored DN lines
    line_idx = np.array(dso_config.current_line_indices, dtype=np.int64)
    i_ka = np.zeros(len(line_idx), dtype=np.float64)
    for k, li in enumerate(line_idx):
        if li not in net.res_line.index:
            raise ValueError(f"Line {li} not found in net.res_line.")
        i_ka[k] = float(net.res_line.at[li, "i_from_ka"])

    # Interface transformers (3-winding couplers)
    iface_trafo = np.array(dso_config.interface_trafo_indices, dtype=np.int64)
    q_iface = np.zeros(len(iface_trafo), dtype=np.float64)
    for k, tidx in enumerate(iface_trafo):
        if tidx not in net.res_trafo3w.index:
            raise ValueError(
                f"Interface transformer {tidx} not found in net.res_trafo3w."
            )
        # Reactive power at HV side (transmission interface)
        q_iface[k] = float(net.res_trafo3w.at[tidx, "q_hv_mvar"])

    # DER Q (distribution-connected sgens, exclude boundary sgens)
    der_bus = np.array(dso_config.der_bus_indices, dtype=np.int64)
    der_q = np.zeros(len(der_bus), dtype=np.float64)
    for k, bus in enumerate(der_bus):
        # Find sgen at this bus that is NOT a boundary sgen
        sgen_mask = (
            (net.sgen["bus"] == bus) &
            ~net.sgen["name"].astype(str).str.startswith("BOUND_")
        )
        if not sgen_mask.any():
            raise ValueError(
                f"DER sgen at bus {bus} not found in DN net.sgen."
            )
        sidx = net.sgen.index[sgen_mask][0]
        der_q[k] = float(net.res_sgen.at[sidx, "q_mvar"])

    # OLTC tap positions (3-winding coupler transformers)
    oltc_idx = np.array(dso_config.oltc_trafo_indices, dtype=np.int64)
    oltc_taps = np.zeros(len(oltc_idx), dtype=np.int64)
    for k, tidx in enumerate(oltc_idx):
        if tidx not in net.trafo3w.index:
            raise ValueError(
                f"OLTC transformer {tidx} not found in net.trafo3w."
            )
        oltc_taps[k] = int(net.trafo3w.at[tidx, "tap_pos"])

    # Shunt states at DN shunt buses
    shunt_bus = np.array(dso_config.shunt_bus_indices, dtype=np.int64)
    shunt_states = np.zeros(len(shunt_bus), dtype=np.int64)
    for k, sb in enumerate(shunt_bus):
        mask = net.shunt["bus"] == sb
        if not mask.any():
            raise ValueError(
                f"DN shunt at bus {sb} not found in net.shunt."
            )
        sidx = net.shunt.index[mask][0]
        shunt_states[k] = int(net.shunt.at[sidx, "step"])

    # No generators in DN (all are at TN level)
    gen_idx = np.array([], dtype=np.int64)
    gen_vm = np.array([], dtype=np.float64)

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
#  HELPER: Apply DSO controls back to pandapower network
# ==============================================================================

def apply_dso_controls(
    net: pp.pandapowerNet,
    dso_output: ControllerOutput,
    dso_config: DSOControllerConfig,
) -> None:
    """
    Apply DSO control outputs to the DN pandapower network.

    Control variables are ordered as:
        [DER_Q (continuous), OLTC_taps (integer), Shunt_states (integer)]
    """
    u = dso_output.u_new

    n_der = len(dso_config.der_bus_indices)
    n_oltc = len(dso_config.oltc_trafo_indices)
    n_shunt = len(dso_config.shunt_bus_indices)

    # 1) Distribution-connected DER Q setpoints
    for k, bus in enumerate(dso_config.der_bus_indices):
        sgen_mask = (
            (net.sgen["bus"] == bus) &
            ~net.sgen["name"].astype(str).str.startswith("BOUND_")
        )
        if not sgen_mask.any():
            raise ValueError(
                f"DER sgen at bus {bus} not found in DN net.sgen."
            )
        sidx = net.sgen.index[sgen_mask][0]
        net.sgen.at[sidx, "q_mvar"] = float(u[k])

    # 2) OLTC tap positions (3-winding transformers)
    for k, trafo_idx in enumerate(dso_config.oltc_trafo_indices):
        if trafo_idx not in net.trafo3w.index:
            raise ValueError(
                f"OLTC transformer {trafo_idx} not found in net.trafo3w."
            )
        tap_val = int(np.round(u[n_der + k]))
        net.trafo3w.at[trafo_idx, "tap_pos"] = tap_val

    # 3) Shunt states
    for k, shunt_bus in enumerate(dso_config.shunt_bus_indices):
        mask = net.shunt["bus"] == shunt_bus
        if not mask.any():
            raise ValueError(
                f"DN shunt at bus {shunt_bus} not found in net.shunt."
            )
        sidx = net.shunt.index[mask][0]
        step_val = int(np.round(u[n_der + n_oltc + k]))
        net.shunt.at[sidx, "step"] = step_val


# ==============================================================================
#  ITERATION LOG
# ==============================================================================

@dataclass
class DSOIterationRecord:
    """
    Record of one DSO-only OFO iteration.
    """
    iteration: int

    # Controller outputs
    dso_q_interface_mvar: Optional[NDArray[np.float64]] = None
    dso_voltages_pu: Optional[NDArray[np.float64]] = None
    dso_q_der_mvar: Optional[NDArray[np.float64]] = None
    dso_oltc_taps: Optional[NDArray[np.int64]] = None
    dso_shunt_states: Optional[NDArray[np.int64]] = None
    dso_objective: Optional[float] = None
    dso_solver_status: Optional[str] = None
    dso_solve_time_s: Optional[float] = None
    dso_slack: Optional[NDArray[np.float64]] = None

    # Plant measurements after power flow
    plant_dn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_q_interface_mvar: Optional[NDArray[np.float64]] = None


# ==============================================================================
#  MAIN DSO-ONLY RUNNER
# ==============================================================================

def run_dso_reactive_power_control(
    q_setpoints_mvar: NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    n_iterations: int = 60,
    alpha: float = 0.01,
    verbose: bool = True,
) -> List[DSOIterationRecord]:
    """
    Run a DSO-only OFO reactive power controller loop on the DN network.

    Parameters
    ----------
    q_setpoints_mvar : NDArray[np.float64]
        Fixed reactive power setpoints for the three TSO-DSO interfaces [Mvar].
    n_iterations : int
        Number of OFO iterations to simulate.
    alpha : float
        OFO step size (gain).
    verbose : bool
        If True, print per-iteration summary.

    Returns
    -------
    log : list[DSOIterationRecord]
        One record per OFO iteration.
    """
    if verbose:
        print("=" * 72)
        print("  DSO-ONLY OFO SIMULATION -- Fixed Interface Setpoints")
        print(f"  Q_setpoints = {q_setpoints_mvar} Mvar")
        print("=" * 72)
        print("[1/5] Building combined 380/110/20 kV network ...")

    # 1) Build combined network
    combined_net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

    if verbose:
        print("[2/5] Running converged power flow on combined network ...")

    # Ensure base-case power flow is converged
    pp.runpp(combined_net, run_control=True, calculate_voltage_angles=True)

    if verbose:
        print("[3/5] Splitting network into TN and DN ...")

    # 2) Split into TN and DN networks
    split_result = split_network(
        combined_net,
        meta,
        dn_slack_coupler_index=0,
    )

    dn_net = split_result.dn_net

    if verbose:
        print("[4/5] Identifying DSO actuators and monitored quantities ...")

    # Distribution-connected DER: DN sgens (exclude boundary sgens)
    dso_der_buses: List[int] = []
    for sidx in dn_net.sgen.index:
        name = str(dn_net.sgen.at[sidx, "name"])
        if name.startswith("BOUND_"):
            continue  # Skip boundary sgens
        subnet = str(dn_net.sgen.at[sidx, "subnet"])
        if subnet == "DN":
            dso_der_buses.append(int(dn_net.sgen.at[sidx, "bus"]))

    # Monitored voltages: all 110 kV buses (HV, subnet "DN")
    dso_v_buses: List[int] = sorted(
        int(b)
        for b in dn_net.bus.index
        if 100.0 <= float(dn_net.bus.at[b, "vn_kv"]) < 200.0
        and str(dn_net.bus.at[b, "subnet"]) == "DN"
    )

    if len(dso_v_buses) == 0:
        raise RuntimeError("No 110 kV buses found for DSO voltage monitoring.")

    # Monitored currents: all DN lines
    dso_lines: List[int] = sorted(
        int(li)
        for li in dn_net.line.index
        if str(dn_net.line.at[li, "subnet"]) == "DN"
    )

    # Interface transformers: 3-winding couplers
    dso_interface_trafos = list(meta.coupler_trafo3w_indices)

    # OLTC transformers: same as interface transformers
    dso_oltc_trafos = dso_interface_trafos.copy()

    # DN shunts: all shunts in DN subnet (if any)
    dso_shunt_bus_candidates: List[int] = []
    dso_shunt_q_candidates: List[float] = []
    for sidx in dn_net.shunt.index:
        subnet = str(dn_net.shunt.at[sidx, "subnet"])
        if subnet == "DN":
            dso_shunt_bus_candidates.append(int(dn_net.shunt.at[sidx, "bus"]))
            dso_shunt_q_candidates.append(float(dn_net.shunt.at[sidx, "q_mvar"]))

    # Probe Jacobian-based sensitivities to select valid actuators
    probe_sens = JacobianSensitivities(dn_net)
    H_probe, m_probe = probe_sens.build_sensitivity_matrix_H(
        der_bus_indices=dso_der_buses,
        observation_bus_indices=dso_v_buses,
        line_indices=dso_lines,
        trafo3w_indices=dso_interface_trafos,
        oltc_trafo3w_indices=dso_oltc_trafos,
        shunt_bus_indices=dso_shunt_bus_candidates,
        shunt_q_steps_mvar=dso_shunt_q_candidates,
    )

    # Select only elements that produced valid sensitivities
    # FIXED: Use correct keys returned by build_sensitivity_matrix_H
    dso_oltc = list(m_probe.get("oltc_trafo3w", dso_oltc_trafos))
    dso_v_buses = list(m_probe.get("obs_buses", dso_v_buses))
    dso_shunt_buses = list(m_probe.get("shunt_buses", []))
    dso_shunt_q_steps = [
        dso_shunt_q_candidates[dso_shunt_bus_candidates.index(b)]
        for b in dso_shunt_buses
    ]
    dso_interface_trafos = list(m_probe.get("trafo3w", dso_interface_trafos))

    dso_config = DSOControllerConfig(
        der_bus_indices=dso_der_buses,
        oltc_trafo_indices=dso_oltc,
        shunt_bus_indices=dso_shunt_buses,
        shunt_q_steps_mvar=dso_shunt_q_steps,
        interface_trafo_indices=dso_interface_trafos,
        voltage_bus_indices=dso_v_buses,
        current_line_indices=dso_lines,
        v_min_pu=0.95,
        v_max_pu=1.05,
        i_max_pu=1.0,
        gamma_q_tracking=1.0,
    )

    if verbose:
        print("[5/5] Creating DSO controller and sensitivity model ...")

    # OFO tuning for DSO
    ofo_params_dso = OFOParameters(
        alpha=alpha,
        g_w=0.00001,
        g_z=1_000_000.0,
        g_s=100.0,
        g_u=0.0001,
    )

    # Network state and sensitivities
    dso_trafo_idx = np.array(dso_oltc, dtype=np.int64)
    dso_ns = network_state_from_net(dn_net, dso_trafo_idx, source_case="DN")

    dso_sens = JacobianSensitivities(dn_net)

    # Actuator bounds for DER, OLTC and shunts
    der_indices_arr = np.array(dso_der_buses, dtype=np.int64)
    der_s_rated = []
    der_p_max = []
    for sidx in dn_net.sgen.index:
        name = str(dn_net.sgen.at[sidx, "name"])
        if name.startswith("BOUND_"):
            continue
        subnet = str(dn_net.sgen.at[sidx, "subnet"])
        if subnet != "DN":
            continue
        der_s_rated.append(float(dn_net.sgen.at[sidx, "sn_mva"]))
        der_p_max.append(float(dn_net.sgen.at[sidx, "p_mw"]))

    if len(der_s_rated) != len(dso_der_buses):
        raise RuntimeError(
            "Mismatch between DER buses and DER rating list length."
        )

    dso_bounds = ActuatorBounds(
        der_indices=der_indices_arr,
        der_s_rated_mva=np.array(der_s_rated, dtype=np.float64),
        der_p_max_mw=np.array(der_p_max, dtype=np.float64),
        oltc_indices=np.array(dso_oltc, dtype=np.int64),
        oltc_tap_min=np.array(
            [int(dn_net.trafo3w.at[t, "tap_min"]) for t in dso_oltc],
            dtype=np.int64,
        ),
        oltc_tap_max=np.array(
            [int(dn_net.trafo3w.at[t, "tap_max"]) for t in dso_oltc],
            dtype=np.int64,
        ),
        shunt_indices=np.array(dso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(dso_shunt_q_steps, dtype=np.float64),
    )

    dso = DSOController(
        controller_id="dso_main",
        params=ofo_params_dso,
        config=dso_config,
        network_state=dso_ns,
        actuator_bounds=dso_bounds,
        sensitivities=dso_sens,
    )

    # Set fixed Q setpoints
    dso.q_setpoint_mvar = q_setpoints_mvar.copy()

    if verbose:
        print("Initialising DSO controller from converged power flow ...")

    # Initial DSO measurement and controller initialisation
    dso_meas0 = measurement_from_dn(dn_net, dso_config, iteration=0)
    dso.initialise(dso_meas0)

    if verbose:
        print()
        print(
            f"Running DSO-only OFO for {n_iterations} iterations "
            f"(alpha = {alpha:.4f}) ..."
        )
        print()

    log: List[DSOIterationRecord] = []

    for it in range(1, n_iterations + 1):
        rec = DSOIterationRecord(iteration=it)

        # New measurement at current operating point
        meas = measurement_from_dn(dn_net, dso_config, iteration=it)

        # OFO step
        try:
            dso_output = dso.step(meas)
        except RuntimeError as e:
            raise RuntimeError(f"DSO OFO step failed at iteration {it}: {e}")

        # Extract control variables
        n_der_d = len(dso_config.der_bus_indices)
        n_oltc_d = len(dso_config.oltc_trafo_indices)

        rec.dso_q_der_mvar = dso_output.u_new[:n_der_d].copy()
        rec.dso_oltc_taps = np.round(
            dso_output.u_new[n_der_d:n_der_d + n_oltc_d]
        ).astype(np.int64)
        rec.dso_shunt_states = np.round(
            dso_output.u_new[n_der_d + n_oltc_d:]
        ).astype(np.int64)

        rec.dso_objective = dso_output.objective_value
        rec.dso_solver_status = dso_output.solver_status
        rec.dso_solve_time_s = dso_output.solve_time_s
        rec.dso_slack = dso_output.z_slack.copy()

        # Predicted outputs
        n_iface = len(dso_config.interface_trafo_indices)
        n_v = len(dso_config.voltage_bus_indices)
        rec.dso_q_interface_mvar = dso_output.y_predicted[:n_iface].copy()
        rec.dso_voltages_pu = dso_output.y_predicted[n_iface:n_iface + n_v].copy()

        # Apply controls to plant
        apply_dso_controls(dn_net, dso_output, dso_config)

        # Re-run power flow
        try:
            pp.runpp(dn_net, run_control=False, calculate_voltage_angles=True)
        except Exception as e:
            raise RuntimeError(
                f"DN power flow failed at iteration {it}: {e}"
            )

        # Record plant measurements
        rec.plant_dn_voltages_pu = dn_net.res_bus.loc[
            dso_v_buses, "vm_pu"
        ].values.astype(np.float64).copy()

        rec.plant_q_interface_mvar = np.zeros(n_iface, dtype=np.float64)
        for k, tidx in enumerate(dso_config.interface_trafo_indices):
            rec.plant_q_interface_mvar[k] = float(
                dn_net.res_trafo3w.at[tidx, "q_hv_mvar"]
            )

        log.append(rec)

        # Verbose summary
        if verbose:
            v = rec.plant_dn_voltages_pu
            v_min = float(np.min(v))
            v_mean = float(np.mean(v))
            v_max = float(np.max(v))

            q_if = rec.plant_q_interface_mvar
            q_err = np.abs(q_if - q_setpoints_mvar[:n_iface])
            q_err_max = float(np.max(q_err))

            print(
                f"[it {it:3d}] DN V: min={v_min:.4f}  mean={v_mean:.4f}  "
                f"max={v_max:.4f} p.u."
            )
            print(
                f"          Q_interface = "
                f"{np.array2string(q_if, precision=2, suppress_small=True)} Mvar  "
                f"(setpoint={np.array2string(q_setpoints_mvar[:n_iface], precision=1)}, "
                f"|err|_max={q_err_max:.2f})"
            )
            print(
                f"          obj={rec.dso_objective:.4e}  "
                f"status={rec.dso_solver_status}  "
                f"t_solve={rec.dso_solve_time_s:.3f}s"
            )
            if rec.dso_q_der_mvar is not None:
                print(
                    "          Q_DER = "
                    f"{np.array2string(rec.dso_q_der_mvar, precision=2, suppress_small=True)} "
                    "Mvar"
                )
            if rec.dso_oltc_taps is not None:
                print(f"          OLTC taps = {rec.dso_oltc_taps}")
            if rec.dso_shunt_states is not None and len(rec.dso_shunt_states) > 0:
                print(f"          Shunt states = {rec.dso_shunt_states}")
            print()

    return log


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Run a single DSO-only reactive power control scenario with Q_set = [0, 0, 0] Mvar.
    """
    q_set = np.array([0.0, 0.0, 0.0])
    n_it = 120

    print()
    print("#" * 72)
    print(f"#  DSO-ONLY SCENARIO: Q_setpoints = {q_set} Mvar")
    print("#" * 72)
    print()

    log = run_dso_reactive_power_control(
        q_setpoints_mvar=q_set,
        n_iterations=n_it,
        alpha=0.01,
        verbose=True,
    )

    final = log[-1]
    if final.plant_q_interface_mvar is not None:
        q_if = final.plant_q_interface_mvar
        q_err = np.abs(q_if - q_set[:len(q_if)])
        q_err_max = float(np.max(q_err))

        print()
        print("=" * 72)
        print("  FINAL DSO-ONLY INTERFACE REACTIVE POWER SUMMARY")
        print("=" * 72)
        print(
            f"  Q_setpoints = {q_set} Mvar"
        )
        print(
            f"  Q_interface = {q_if} Mvar"
        )
        print(f"  Max |Q - Q_set| = {q_err_max:.4f} Mvar")
        print("=" * 72)
        print()


if __name__ == "__main__":
    main()
