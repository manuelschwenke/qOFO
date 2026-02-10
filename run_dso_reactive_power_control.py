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
    plant_q_interface_mvar: Optional[NDArray[np.float64]] = Non