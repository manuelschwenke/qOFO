#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascaded TSO-DSO OFO Controller
================================
TSO every 3 min, DSO every 1 min. Combined network for everything
(plant, sensitivities, state). TSO sends Q setpoints to DSO at PCCs.

TSO actuators: gen AVR voltages, machine trafo OLTCs (2W), TS-DER Q, 380 kV shunts
DSO actuators: DN-DER Q, 3W coupler OLTCs, tertiary winding shunts

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
import pandapower as pp

from network.build_tuda_net import build_tuda_net, NetworkMetadata
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds
from controller.base_controller import OFOParameters, ControllerOutput
from controller.tso_controller import TSOController, TSOControllerConfig
from controller.dso_controller import DSOController, DSOControllerConfig
from core.message import SetpointMessage
from core.profiles import load_profiles, snapshot_base_values, apply_profiles, DEFAULT_PROFILES_CSV
from sensitivity.jacobian import JacobianSensitivities

# =============================================================================
#  Helpers
# =============================================================================

def _network_state(net: pp.pandapowerNet, source: str = "COMBINED") -> NetworkState:
    """Snapshot NetworkState from a converged combined network."""
    buses = np.array(net.bus.index, dtype=np.int64)
    vm = net.res_bus.loc[buses, "vm_pu"].values.astype(np.float64)
    va = np.deg2rad(net.res_bus.loc[buses, "va_degree"].values.astype(np.float64))
    slack = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
    pv = np.array([int(net.gen.at[g, "bus"]) for g in net.gen.index
                    if net.gen.at[g, "in_service"]], dtype=np.int64)
    pq = np.array([int(b) for b in buses
                    if int(b) != slack and int(b) not in pv], dtype=np.int64)
    return NetworkState(
        bus_indices=buses, voltage_magnitudes_pu=vm, voltage_angles_rad=va,
        slack_bus_index=slack, pv_bus_indices=pv, pq_bus_indices=pq,
        transformer_indices=np.array([], dtype=np.int64),
        tap_positions=np.array([], dtype=np.float64),
        source_case=source, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cached_at_iteration=0,
    )


def _sgen_at_bus(net, bus: int, *, exclude_bound: bool = True):
    """Return first non-boundary sgen index at a bus."""
    for s in net.sgen.index[net.sgen["bus"] == bus]:
        if exclude_bound and str(net.sgen.at[s, "name"]).startswith("BOUND_"):
            continue
        return s
    raise ValueError(f"No sgen at bus {bus}")


def _measure_tso(net: pp.pandapowerNet, cfg: TSOControllerConfig, it: int) -> Measurement:
    """TSO measurement from combined network."""
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # TN line currents
    i_ka = np.array([float(net.res_line.at[li, "i_from_ka"])
                      for li in cfg.current_line_indices], dtype=np.float64)

    # Interface Q (3W HV side)
    q_iface = np.array([float(net.res_trafo3w.at[t, "q_hv_mvar"])
                         for t in cfg.pcc_trafo_indices], dtype=np.float64)

    # TS-connected DER Q
    der_q = np.zeros(len(cfg.der_bus_indices), dtype=np.float64)
    for k, bus in enumerate(cfg.der_bus_indices):
        for s in net.sgen.index[net.sgen["bus"] == bus]:
            if not str(net.sgen.at[s, "name"]).startswith("BOUND_"):
                der_q[k] += float(net.res_sgen.at[s, "q_mvar"])

    # 2W OLTC taps
    oltc_taps = np.array([int(net.trafo.at[t, "tap_pos"])
                           for t in cfg.oltc_trafo_indices], dtype=np.int64)

    # 380 kV shunt states
    shunt_states = np.zeros(len(cfg.shunt_bus_indices), dtype=np.int64)
    for k, sb in enumerate(cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            shunt_states[k] = int(net.shunt.at[net.shunt.index[mask][0], "step"])

    # Generator AVR
    gen_vm = np.array([float(net.gen.at[g, "vm_pu"])
                        for g in cfg.gen_indices], dtype=np.float64)

    return Measurement(
        iteration=it, bus_indices=all_bus, voltage_magnitudes_pu=vm,
        branch_indices=np.array(cfg.current_line_indices, dtype=np.int64),
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=np.array(cfg.pcc_trafo_indices, dtype=np.int64),
        interface_q_hv_side_mvar=q_iface,
        der_indices=np.array(cfg.der_bus_indices, dtype=np.int64), der_q_mvar=der_q,
        oltc_indices=np.array(cfg.oltc_trafo_indices, dtype=np.int64),
        oltc_tap_positions=oltc_taps,
        shunt_indices=np.array(cfg.shunt_bus_indices, dtype=np.int64),
        shunt_states=shunt_states,
        gen_indices=np.array(cfg.gen_indices, dtype=np.int64), gen_vm_pu=gen_vm,
    )


def _measure_dso(net: pp.pandapowerNet, cfg: DSOControllerConfig, it: int) -> Measurement:
    """DSO measurement from combined network."""
    all_bus = np.array(sorted(net.res_bus.index), dtype=np.int64)
    vm = net.res_bus.loc[all_bus, "vm_pu"].values.astype(np.float64)

    # DN line currents
    i_ka = np.array([float(net.res_line.at[li, "i_from_ka"])
                      for li in cfg.current_line_indices], dtype=np.float64)

    # Interface Q (3W HV side)
    q_iface = np.array([float(net.res_trafo3w.at[t, "q_hv_mvar"])
                         for t in cfg.interface_trafo_indices], dtype=np.float64)

    # DN DER Q
    der_q = np.zeros(len(cfg.der_bus_indices), dtype=np.float64)
    for k, bus in enumerate(cfg.der_bus_indices):
        for s in net.sgen.index[net.sgen["bus"] == bus]:
            if not str(net.sgen.at[s, "name"]).startswith("BOUND_"):
                der_q[k] += float(net.res_sgen.at[s, "q_mvar"])

    # 3W OLTC taps
    oltc_taps = np.array([int(net.trafo3w.at[t, "tap_pos"])
                           for t in cfg.oltc_trafo_indices], dtype=np.int64)

    # Tertiary shunt states
    shunt_states = np.zeros(len(cfg.shunt_bus_indices), dtype=np.int64)
    for k, sb in enumerate(cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            shunt_states[k] = int(net.shunt.at[net.shunt.index[mask][0], "step"])

    return Measurement(
        iteration=it, bus_indices=all_bus, voltage_magnitudes_pu=vm,
        branch_indices=np.array(cfg.current_line_indices, dtype=np.int64),
        current_magnitudes_ka=i_ka,
        interface_transformer_indices=np.array(cfg.interface_trafo_indices, dtype=np.int64),
        interface_q_hv_side_mvar=q_iface,
        der_indices=np.array(cfg.der_bus_indices, dtype=np.int64), der_q_mvar=der_q,
        oltc_indices=np.array(cfg.oltc_trafo_indices, dtype=np.int64),
        oltc_tap_positions=oltc_taps,
        shunt_indices=np.array(cfg.shunt_bus_indices, dtype=np.int64),
        shunt_states=shunt_states,
        gen_indices=np.array([], dtype=np.int64),
        gen_vm_pu=np.array([], dtype=np.float64),
    )


def _apply_tso(net: pp.pandapowerNet, out: ControllerOutput, cfg: TSOControllerConfig):
    """Apply TSO controls to combined network. PCC Q setpoints are NOT applied here."""
    u = out.u_new
    n_der = len(cfg.der_bus_indices)
    n_pcc = len(cfg.pcc_trafo_indices)
    n_gen = len(cfg.gen_indices)
    n_oltc = len(cfg.oltc_trafo_indices)

    # TS-DER Q
    for k, bus in enumerate(cfg.der_bus_indices):
        net.sgen.at[_sgen_at_bus(net, bus), "q_mvar"] = float(u[k])

    # AVR setpoints
    off = n_der + n_pcc
    for k, g in enumerate(cfg.gen_indices):
        net.gen.at[g, "vm_pu"] = float(u[off + k])

    # 2W machine trafo OLTCs
    off += n_gen
    for k, t in enumerate(cfg.oltc_trafo_indices):
        net.trafo.at[t, "tap_pos"] = int(np.round(u[off + k]))

    # 380 kV shunts
    off += n_oltc
    for k, sb in enumerate(cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = int(np.round(u[off + k]))


def _apply_dso(net: pp.pandapowerNet, out: ControllerOutput, cfg: DSOControllerConfig):
    """Apply DSO controls to combined network."""
    u = out.u_new
    n_der = len(cfg.der_bus_indices)
    n_oltc = len(cfg.oltc_trafo_indices)

    # DN DER Q
    for k, bus in enumerate(cfg.der_bus_indices):
        sgens = [s for s in net.sgen.index[net.sgen["bus"] == bus]
                 if not str(net.sgen.at[s, "name"]).startswith("BOUND_")]
        q_each = u[k] / len(sgens) if sgens else 0.0
        for s in sgens:
            net.sgen.at[s, "q_mvar"] = float(q_each)

    # 3W coupler OLTCs
    for k, t in enumerate(cfg.oltc_trafo_indices):
        net.trafo3w.at[t, "tap_pos"] = int(np.round(u[n_der + k]))

    # Tertiary shunts
    for k, sb in enumerate(cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = int(np.round(u[n_der + n_oltc + k]))



# =============================================================================
#  Iteration log
# =============================================================================

A2S = lambda a: np.array2string(a, precision=2, suppress_small=True)


@dataclass
class IterationRecord:
    minute: int
    tso_active: bool
    dso_active: bool
    # TSO optimisation variables [Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]
    tso_q_der_mvar: Optional[NDArray[np.float64]] = None
    tso_q_pcc_set_mvar: Optional[NDArray[np.float64]] = None
    tso_v_gen_pu: Optional[NDArray[np.float64]] = None
    tso_oltc_taps: Optional[NDArray[np.int64]] = None
    tso_shunt_states: Optional[NDArray[np.int64]] = None
    tso_objective: Optional[float] = None
    tso_solver_status: Optional[str] = None
    tso_solve_time_s: Optional[float] = None
    # DSO optimisation variables [Q_DER | s_OLTC | s_shunt]
    dso_q_der_mvar: Optional[NDArray[np.float64]] = None
    dso_oltc_taps: Optional[NDArray[np.int64]] = None
    dso_shunt_states: Optional[NDArray[np.int64]] = None
    dso_q_setpoint_mvar: Optional[NDArray[np.float64]] = None  # Q set by TSO
    dso_q_actual_mvar: Optional[NDArray[np.float64]] = None    # actual Q at interface after PF
    dso_objective: Optional[float] = None
    dso_solver_status: Optional[str] = None
    dso_solve_time_s: Optional[float] = None
    # Plant voltages after PF
    plant_tn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_dn_voltages_pu: Optional[NDArray[np.float64]] = None


# =============================================================================
#  Main cascade runner
# =============================================================================

def run_cascade(
    v_setpoint_pu: float,
    n_minutes: int = 120,
    tso_period_min: int = 3,
    dso_period_min: int = 1,
    start_time: datetime = datetime(2016, 7, 15, 10, 0),
    profiles_csv: str = DEFAULT_PROFILES_CSV,
    verbose: bool = True,
) -> List[IterationRecord]:
    """Run cascaded TSO-DSO OFO. Combined network for everything."""

    if verbose:
        print("=" * 72)
        print(f"  CASCADED OFO  --  V_set = {v_setpoint_pu:.3f} p.u.")
        print(f"  Profiles: {os.path.basename(profiles_csv)}  "
              f"start={start_time:%d.%m.%Y %H:%M}  {n_minutes} min")
        print("=" * 72)

    # 1) Build combined network
    net, meta = build_tuda_net(ext_grid_vm_pu=1.05, pv_nodes=True)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if hasattr(net, 'controller') and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Load time-series profiles and snapshot base load/gen values
    profiles = load_profiles(profiles_csv, timestep_min=dso_period_min)
    snapshot_base_values(net)

    # Apply profiles at t=0 and re-run PF so sensitivities + init use realistic operating point
    apply_profiles(net, profiles, start_time)
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # 2) Identify elements
    dn_buses = {int(b) for b in net.bus.index if str(net.bus.at[b, "subnet"]) == "DN"}

    # -- TSO DER: TS-connected sgens (not DN, not boundary)
    tso_der_buses = list(dict.fromkeys(
        int(net.sgen.at[s, "bus"]) for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) not in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ))

    # -- TSO monitored: 380 kV buses, TN lines
    tso_v_buses = sorted(int(b) for b in net.bus.index
                         if float(net.bus.at[b, "vn_kv"]) >= 300.0)
    tso_lines = sorted(int(li) for li in net.line.index
                       if str(net.line.at[li, "subnet"]) == "TN")

    # -- TSO generators + machine trafos
    tso_gen_indices, tso_gen_bus_indices = [], []
    for g in net.gen.index:
        tso_gen_indices.append(int(g))
        lv = int(net.gen.at[g, "bus"])
        mt = net.trafo.index[
            (net.trafo["lv_bus"] == lv) &
            net.trafo["name"].astype(str).str.startswith("MachineTrf|")
        ]
        if mt.empty:
            raise RuntimeError(f"No machine trafo for gen {g} bus {lv}")
        tso_gen_bus_indices.append(int(net.trafo.at[mt[0], "hv_bus"]))

    # -- TSO 380 kV shunts
    tso_shunt_buses_cand = [int(net.shunt.at[s, "bus"]) for s in meta.tn_shunt_indices
                            if s in net.shunt.index]
    tso_shunt_q_cand = [float(net.shunt.at[s, "q_mvar"]) for s in meta.tn_shunt_indices
                        if s in net.shunt.index]

    # -- DSO DER: DN-connected sgens (not boundary)
    dso_der_buses = list(dict.fromkeys(
        int(net.sgen.at[s, "bus"]) for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ))

    # -- DSO monitored: 110 kV DN buses, DN lines
    dso_v_buses = sorted(int(b) for b in net.bus.index
                         if 100.0 <= float(net.bus.at[b, "vn_kv"]) < 200.0
                         and int(b) in dn_buses)
    dso_lines = sorted(int(li) for li in net.line.index
                       if str(net.line.at[li, "subnet"]) == "DN")

    # -- DSO tertiary shunts
    dso_shunt_buses_cand = [int(net.shunt.at[s, "bus"]) for s in meta.tertiary_shunt_indices
                            if s in net.shunt.index]
    dso_shunt_q_cand = [float(net.shunt.at[s, "q_mvar"]) for s in meta.tertiary_shunt_indices
                        if s in net.shunt.index]

    pcc_trafos = list(meta.coupler_trafo3w_indices)

    # 3) Sensitivities from combined network
    if verbose:
        print("Computing sensitivities from combined network ...")

    # TSO probe
    tso_sens = JacobianSensitivities(net)
    _, m_tso = tso_sens.build_sensitivity_matrix_H(
        der_bus_indices=tso_der_buses, observation_bus_indices=tso_v_buses,
        line_indices=tso_lines, oltc_trafo_indices=list(meta.machine_trafo_indices),
        shunt_bus_indices=tso_shunt_buses_cand, shunt_q_steps_mvar=tso_shunt_q_cand,
    )
    tso_oltc = list(m_tso.get("oltc_trafos", []))
    tso_v_buses = list(m_tso.get("obs_buses", tso_v_buses))
    tso_shunt_buses = list(m_tso.get("shunt_buses", []))
    tso_shunt_q = [tso_shunt_q_cand[tso_shunt_buses_cand.index(b)] for b in tso_shunt_buses]

    # DSO probe
    dso_sens = JacobianSensitivities(net)
    _, m_dso = dso_sens.build_sensitivity_matrix_H(
        der_bus_indices=dso_der_buses, observation_bus_indices=dso_v_buses,
        line_indices=dso_lines, trafo3w_indices=pcc_trafos,
        oltc_trafo3w_indices=list(meta.coupler_trafo3w_indices),
        shunt_bus_indices=dso_shunt_buses_cand, shunt_q_steps_mvar=dso_shunt_q_cand,
    )
    dso_oltc = list(m_dso.get("oltc_trafo3w", list(meta.coupler_trafo3w_indices)))
    dso_v_buses = list(m_dso.get("obs_buses", dso_v_buses))
    dso_shunt_buses = list(m_dso.get("shunt_buses", []))
    dso_shunt_q = [dso_shunt_q_cand[dso_shunt_buses_cand.index(b)] for b in dso_shunt_buses]
    dso_iface_trafos = list(m_dso.get("trafo3w", pcc_trafos))

    # 4) Controller configs
    v_setpoints = np.full(len(tso_v_buses), v_setpoint_pu)
    dso_id = "dso_0"

    tso_config = TSOControllerConfig(
        der_bus_indices=tso_der_buses, pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[dso_id] * len(pcc_trafos),
        oltc_trafo_indices=tso_oltc, shunt_bus_indices=tso_shunt_buses,
        shunt_q_steps_mvar=tso_shunt_q,
        voltage_bus_indices=tso_v_buses, current_line_indices=tso_lines,
        v_setpoints_pu=v_setpoints, gamma_v_tracking=1,
        gen_indices=tso_gen_indices, gen_bus_indices=tso_gen_bus_indices,
    )
    dso_config = DSOControllerConfig(
        der_bus_indices=dso_der_buses, oltc_trafo_indices=dso_oltc,
        shunt_bus_indices=dso_shunt_buses, shunt_q_steps_mvar=dso_shunt_q,
        interface_trafo_indices=dso_iface_trafos,
        voltage_bus_indices=dso_v_buses, current_line_indices=dso_lines,
        gamma_q_tracking=1.0,
    )

    # 5) Actuator bounds (from combined network)
    def _der_bounds(buses):
        s_rated, p_max = {b: 0.0 for b in buses}, {b: 0.0 for b in buses}
        for s in net.sgen.index:
            b = int(net.sgen.at[s, "bus"])
            if b in s_rated and not str(net.sgen.at[s, "name"]).startswith("BOUND_"):
                s_rated[b] += float(net.sgen.at[s, "sn_mva"])
                p_max[b] += float(net.sgen.at[s, "p_mw"])
        return (np.array([s_rated[b] for b in buses], dtype=np.float64),
                np.array([p_max[b] for b in buses], dtype=np.float64))

    tso_s, tso_p = _der_bounds(tso_der_buses)
    tso_bounds = ActuatorBounds(
        der_indices=np.array(tso_der_buses, dtype=np.int64),
        der_s_rated_mva=tso_s, der_p_max_mw=tso_p,
        oltc_indices=np.array(tso_oltc, dtype=np.int64),
        oltc_tap_min=np.array([int(net.trafo.at[t, "tap_min"]) for t in tso_oltc], dtype=np.int64),
        oltc_tap_max=np.array([int(net.trafo.at[t, "tap_max"]) for t in tso_oltc], dtype=np.int64),
        shunt_indices=np.array(tso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(tso_shunt_q, dtype=np.float64),
    )

    dso_s, dso_p = _der_bounds(dso_der_buses)
    dso_bounds = ActuatorBounds(
        der_indices=np.array(dso_der_buses, dtype=np.int64),
        der_s_rated_mva=dso_s, der_p_max_mw=dso_p,
        oltc_indices=np.array(dso_oltc, dtype=np.int64),
        oltc_tap_min=np.array([int(net.trafo3w.at[t, "tap_min"]) for t in dso_oltc], dtype=np.int64),
        oltc_tap_max=np.array([int(net.trafo3w.at[t, "tap_max"]) for t in dso_oltc], dtype=np.int64),
        shunt_indices=np.array(dso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(dso_shunt_q, dtype=np.float64),
    )

    # 6) Create controllers
    # Per-actuator-type g_w weights on the diagonal of G_w.
    # TSO u = [Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt]
    gw_tso = np.concatenate([
        np.full(len(tso_der_buses),   0.01),   # Q_DER
        np.full(len(pcc_trafos),      0.01),   # Q_PCC_set
        np.full(len(tso_gen_indices), 100),   # V_gen_set
        np.full(len(tso_oltc),        0.1),   # s_OLTC
        np.full(len(tso_shunt_buses), 0.1),   # s_shunt
    ])
    # DSO u = [Q_DER | s_OLTC | s_shunt]
    gw_dso = np.concatenate([
        np.full(len(dso_der_buses),   0.02),   # Q_DER
        np.full(len(dso_oltc),        10),   # s_OLTC
        np.full(len(dso_shunt_buses), 10),   # s_shunt
    ])
    ofo_tso = OFOParameters(alpha=0.002, g_w=gw_tso, g_z=1e10, g_u=1e-8)
    ofo_dso = OFOParameters(alpha=0.003, g_w=gw_dso, g_z=1e10, g_u=1e-8)

    ns = _network_state(net)

    tso = TSOController("tso_main", ofo_tso, tso_config, ns, tso_bounds,
                        JacobianSensitivities(net))
    dso = DSOController(dso_id, ofo_dso, dso_config, ns, dso_bounds,
                        JacobianSensitivities(net))

    # 7) Initialise from converged PF
    tso.initialise(_measure_tso(net, tso_config, 0))
    dso.initialise(_measure_dso(net, dso_config, 0))

    if verbose:
        print(f"Running cascade: {n_minutes} min  (TSO/{tso_period_min}m, DSO/{dso_period_min}m)")
        print(f"  TSO: {len(tso_der_buses)} DER, {len(tso_oltc)} OLTC, "
              f"{len(tso_gen_indices)} gen, {len(tso_shunt_buses)} shunt")
        print(f"  DSO: {len(dso_der_buses)} DER, {len(dso_oltc)} OLTC, "
              f"{len(dso_shunt_buses)} shunt")
        print()

    # 8) Cascade loop
    log: List[IterationRecord] = []

    for minute in range(1, n_minutes + 1):
        run_tso = (minute % tso_period_min == 0)
        run_dso = (minute % dso_period_min == 0)
        rec = IterationRecord(minute=minute, tso_active=run_tso, dso_active=run_dso)

        # Apply time-series profiles for this minute
        t_now = start_time + timedelta(minutes=minute)
        apply_profiles(net, profiles, t_now)

        # TSO step
        if run_tso:
            try:
                tso_out = tso.step(_measure_tso(net, tso_config, minute))
                u = tso_out.u_new
                n_d = len(tso_config.der_bus_indices)
                n_p = len(tso_config.pcc_trafo_indices)
                n_g = len(tso_config.gen_indices)
                n_o = len(tso_config.oltc_trafo_indices)
                n_s = len(tso_config.shunt_bus_indices)
                off = 0
                rec.tso_q_der_mvar = u[off:off + n_d].copy();           off += n_d
                rec.tso_q_pcc_set_mvar = u[off:off + n_p].copy();       off += n_p
                rec.tso_v_gen_pu = u[off:off + n_g].copy();             off += n_g
                rec.tso_oltc_taps = np.round(u[off:off + n_o]).astype(np.int64);  off += n_o
                rec.tso_shunt_states = np.round(u[off:off + n_s]).astype(np.int64)
                rec.tso_objective = tso_out.objective_value
                rec.tso_solver_status = tso_out.solver_status
                rec.tso_solve_time_s = tso_out.solve_time_s

                _apply_tso(net, tso_out, tso_config)

                # Send Q setpoints to DSO
                msgs = [m for m in tso.generate_setpoint_messages()
                        if m.target_controller_id == dso_id]
                if msgs:
                    merged = SetpointMessage(
                        source_controller_id="tso_main", target_controller_id=dso_id,
                        iteration=minute,
                        interface_transformer_indices=np.concatenate(
                            [m.interface_transformer_indices for m in msgs]),
                        q_setpoints_mvar=np.concatenate(
                            [m.q_setpoints_mvar for m in msgs]),
                    )
                    dso.receive_setpoint(merged)
            except RuntimeError as e:
                if verbose:
                    print(f"  [min {minute:3d}] TSO FAILED: {e}")

        # DSO step
        if run_dso:
            try:
                dso_meas = _measure_dso(net, dso_config, minute)
                dso_out = dso.step(dso_meas)
                u = dso_out.u_new
                n_dd = len(dso_config.der_bus_indices)
                n_do = len(dso_config.oltc_trafo_indices)
                n_ds = len(dso_config.shunt_bus_indices)
                rec.dso_q_der_mvar = u[:n_dd].copy()
                rec.dso_oltc_taps = np.round(u[n_dd:n_dd + n_do]).astype(np.int64)
                rec.dso_shunt_states = np.round(u[n_dd + n_do:n_dd + n_do + n_ds]).astype(np.int64)
                rec.dso_q_setpoint_mvar = dso.q_setpoint_mvar.copy()
                rec.dso_objective = dso_out.objective_value
                rec.dso_solver_status = dso_out.solver_status
                rec.dso_solve_time_s = dso_out.solve_time_s

                _apply_dso(net, dso_out, dso_config)
                tso.receive_capability(dso.generate_capability_message("tso_main", dso_meas))
            except RuntimeError as e:
                if verbose:
                    print(f"  [min {minute:3d}] DSO FAILED: {e}")

        # Power flow
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)

        # Record plant voltages + actual interface Q after PF
        rec.plant_tn_voltages_pu = net.res_bus.loc[tso_v_buses, "vm_pu"].values.copy()
        rec.plant_dn_voltages_pu = net.res_bus.loc[dso_v_buses, "vm_pu"].values.copy()
        if run_dso:
            rec.dso_q_actual_mvar = np.array(
                [float(net.res_trafo3w.at[t, "q_hv_mvar"])
                 for t in dso_config.interface_trafo_indices], dtype=np.float64)
        log.append(rec)

        if verbose and (run_tso or run_dso):
            tags = "TSO+DSO" if (run_tso and run_dso) else ("TSO" if run_tso else "DSO")
            v_tn = rec.plant_tn_voltages_pu
            v_dn = rec.plant_dn_voltages_pu
            print(f"  [min {minute:3d}] {tags}")
            print(f"    TN V: min={np.min(v_tn):.4f} mean={np.mean(v_tn):.4f} max={np.max(v_tn):.4f}")
            print(f"    DN V: min={np.min(v_dn):.4f} mean={np.mean(v_dn):.4f} max={np.max(v_dn):.4f}")

            if run_tso and rec.tso_q_der_mvar is not None:
                print(f"    TSO  obj={rec.tso_objective:.4e}  {rec.tso_solver_status}  "
                      f"t={rec.tso_solve_time_s:.3f}s")
                print(f"      Q_DER   = {A2S(rec.tso_q_der_mvar)} Mvar")
                print(f"      Q_PCC_s = {A2S(rec.tso_q_pcc_set_mvar)} Mvar")
                print(f"      V_gen   = {A2S(rec.tso_v_gen_pu)} pu")
                if len(rec.tso_oltc_taps) > 0:
                    print(f"      OLTC    = {rec.tso_oltc_taps}")
                if len(rec.tso_shunt_states) > 0:
                    print(f"      Shunt   = {rec.tso_shunt_states}")

            if run_dso and rec.dso_q_der_mvar is not None:
                print(f"    DSO  obj={rec.dso_objective:.4e}  {rec.dso_solver_status}  "
                      f"t={rec.dso_solve_time_s:.3f}s")
                print(f"      Q_DER   = {A2S(rec.dso_q_der_mvar)} Mvar")
                if len(rec.dso_oltc_taps) > 0:
                    print(f"      OLTC    = {rec.dso_oltc_taps}")
                if len(rec.dso_shunt_states) > 0:
                    print(f"      Shunt   = {rec.dso_shunt_states}")
                print(f"      Q_set   = {A2S(rec.dso_q_setpoint_mvar)} Mvar  (from TSO)")
                print(f"      Q_act   = {A2S(rec.dso_q_actual_mvar)} Mvar  (plant)")
            print()

    return log


# =============================================================================
#  Summary
# =============================================================================

def print_summary(v_set: float, log: List[IterationRecord]):
    final = log[-1]
    print()
    print("=" * 72)
    print(f"  SUMMARY  --  V_set = {v_set:.3f} p.u.")
    print("=" * 72)

    if final.plant_tn_voltages_pu is not None:
        v = final.plant_tn_voltages_pu
        print(f"  Final TN V: min={np.min(v):.4f}  mean={np.mean(v):.4f}  max={np.max(v):.4f}")
        print(f"  Max |V - V_set| = {np.max(np.abs(v - v_set)):.4f} p.u.")

    n_tso = sum(1 for r in log if r.tso_active)
    n_dso = sum(1 for r in log if r.dso_active)
    print(f"  TSO steps: {n_tso},  DSO steps: {n_dso}")

    # Convergence trace
    tso_recs = [r for r in log if r.tso_active and r.plant_tn_voltages_pu is not None]
    if tso_recs:
        print()
        print(f"  {'min':>5s}  {'V_min':>8s}  {'V_mean':>8s}  {'V_max':>8s}  "
              f"{'|err|_max':>10s}  {'obj':>12s}  {'status':>10s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}")
        for r in tso_recs:
            v = r.plant_tn_voltages_pu
            err = np.max(np.abs(v - v_set))
            print(f"  {r.minute:5d}  {np.min(v):8.4f}  {np.mean(v):8.4f}  "
                  f"{np.max(v):8.4f}  {err:10.4f}  "
                  f"{r.tso_objective or 0:12.4e}  {r.tso_solver_status or '':>10s}")

    print("=" * 72)


# =============================================================================
#  Entry point
# =============================================================================

def main():
    for v_set in [1.03]:
        log = run_cascade(v_setpoint_pu=v_set, n_minutes=int(60*12),
                          tso_period_min=3, dso_period_min=1, verbose=True)
        print_summary(v_set, log)


if __name__ == "__main__":
    main()
