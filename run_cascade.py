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

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"mosek"
)

from network.build_tuda_net import build_tuda_net, NetworkMetadata
from core.network_state import NetworkState
from core.measurement import Measurement
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from controller.base_controller import OFOParameters, ControllerOutput
from controller.tso_controller import TSOController, TSOControllerConfig
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.reserve_observer import ReserveObserver, ReserveObserverConfig
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

    # Generator AVR setpoint, P and Q from power flow
    gen_vm = np.array([float(net.gen.at[g, "vm_pu"])
                        for g in cfg.gen_indices], dtype=np.float64)
    gen_p = np.array([float(net.res_gen.at[g, "p_mw"])
                       for g in cfg.gen_indices], dtype=np.float64)
    gen_q = np.array([float(net.res_gen.at[g, "q_mvar"])
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
        gen_p_mw=gen_p, gen_q_mvar=gen_q,
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
A3S = lambda a: np.array2string(a, precision=3, suppress_small=True)


@dataclass
class ContingencyEvent:
    """
    A scheduled network contingency (line trip or generator outage).

    Parameters
    ----------
    minute : int
        Simulation minute at which the event occurs.
    element_type : str
        Pandapower element table: ``"line"`` or ``"gen"``.
    element_index : int
        Row index in the corresponding ``net.<element_type>`` table.
    action : str
        ``"trip"`` sets ``in_service = False``;
        ``"restore"`` sets ``in_service = True``.
    """
    minute: int
    element_type: str       # "line" or "gen"
    element_index: int
    action: str = "trip"    # "trip" | "restore"


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
    # Plant measurements after PF
    plant_tn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_dn_voltages_pu: Optional[NDArray[np.float64]] = None
    plant_tn_currents_ka: Optional[NDArray[np.float64]] = None   # TN line i_from_ka
    plant_dn_currents_ka: Optional[NDArray[np.float64]] = None   # DN line i_from_ka
    tso_q_gen_mvar: Optional[NDArray[np.float64]] = None  # synchronous gen Q output
    # Penalty terms (computed from plant measurements after PF)
    tso_v_penalty: Optional[float] = None    # g_v * sum((V - V_set)^2)
    dso_q_penalty: Optional[float] = None    # g_q * sum((Q - Q_set)^2)
    # Contingency events that fired this minute (human-readable descriptions)
    contingency_events: Optional[List[str]] = None


# =============================================================================
#  Contingency helper
# =============================================================================

def _apply_contingency(
    net,
    ev: ContingencyEvent,
    verbose: int,
) -> str:
    """
    Apply a single contingency event to the pandapower network.

    Returns a human-readable description string for logging.
    """
    trip = (ev.action == "trip")
    in_service = not trip
    tag = "TRIP" if trip else "RESTORE"

    if ev.element_type == "line":
        net.line.at[ev.element_index, "in_service"] = in_service
        name = net.line.at[ev.element_index, "name"]
        desc = f"{tag} line {ev.element_index} ({name})"
    elif ev.element_type == "gen":
        net.gen.at[ev.element_index, "in_service"] = in_service
        name = net.gen.at[ev.element_index, "name"]
        desc = f"{tag} gen {ev.element_index} ({name})"
    else:
        raise ValueError(
            f"Unknown element_type '{ev.element_type}' "
            f"(expected 'line' or 'gen')"
        )

    if verbose > 0:
        print(f"\n  {'='*60}")
        print(f"  *** CONTINGENCY min {ev.minute}: {desc}")
        print(f"  {'='*60}\n")
    return desc


# =============================================================================
#  Main cascade runner
# =============================================================================

@dataclass
class CascadeResult:
    """Container for all cascade simulation results."""
    log: List[IterationRecord]
    tso_config: TSOControllerConfig
    dso_config: DSOControllerConfig


def run_cascade(
    config: "CascadeConfig",
    *,
    live_plotter=None,
) -> CascadeResult:
    """
    Run cascaded TSO-DSO OFO on a combined network.

    Parameters
    ----------
    config : CascadeConfig
        Central configuration object holding every tunable parameter.
        See :class:`core.cascade_config.CascadeConfig` for all fields.
    live_plotter : optional
        Object with an ``update(rec)`` method called after each iteration,
        e.g. :class:`visualisation.plot_cascade.LivePlotter`.  Takes
        precedence over ``config.live_plot``.
    """
    from core.cascade_config import CascadeConfig

    # ── Unpack frequently-used config fields for readability ──────────────
    v_setpoint_pu = config.v_setpoint_pu
    dso_v_setpoint_pu = config.effective_dso_v_setpoint_pu
    n_minutes = config.n_minutes
    tso_period_min = config.tso_period_min
    dso_period_min = config.dso_period_min
    start_time = config.start_time
    profiles_csv = config.profiles_csv
    verbose = config.verbose
    live_plot = config.live_plot
    g_v = config.g_v
    g_q = config.g_q
    use_profiles = config.use_profiles
    enable_reserve_observer = config.enable_reserve_observer
    gw_oltc_cross_tso = config.gw_oltc_cross_tso
    gw_oltc_cross_dso = config.gw_oltc_cross_dso
    contingencies = list(config.contingencies) if config.contingencies else []
    reserve_cooldown_min = config.reserve_cooldown_min

    if verbose > 0:
        print("=" * 72)
        if dso_v_setpoint_pu != v_setpoint_pu:
            print(f"  CASCADED OFO  --  V_set TSO={v_setpoint_pu:.3f}  DSO={dso_v_setpoint_pu:.3f} p.u.")
        else:
            print(f"  CASCADED OFO  --  V_set = {v_setpoint_pu:.3f} p.u.")
        print(f"  Profiles: {os.path.basename(profiles_csv)}  "
              f"start={start_time:%d.%m.%Y %H:%M}  {n_minutes} min")
        print("=" * 72)
        if contingencies:
            print(f"  Scheduled contingencies ({len(contingencies)}):")
            for ev in contingencies:
                print(f"    min {ev.minute:4d}: {ev.action.upper()} "
                      f"{ev.element_type} {ev.element_index}")

    # contingencies already unpacked as list above

    # 1) Build combined network
    net, meta = build_tuda_net(ext_grid_vm_pu=v_setpoint_pu, pv_nodes=True)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if hasattr(net, 'controller') and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Load time-series profiles and snapshot base load/gen values
    profiles = load_profiles(profiles_csv, timestep_min=dso_period_min)
    snapshot_base_values(net)

    # Apply profiles at t=0 and re-run PF so sensitivities + init use realistic operating point
    if use_profiles:
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
    if verbose > 1:
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

    # Per-line thermal ratings [kA] for current constraints
    tso_line_max_i_ka = [float(net.line.at[li, "max_i_ka"]) for li in tso_lines]
    dso_line_max_i_ka = [float(net.line.at[li, "max_i_ka"]) for li in dso_lines]

    tso_config = TSOControllerConfig(
        der_bus_indices=tso_der_buses, pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[dso_id] * len(pcc_trafos),
        oltc_trafo_indices=tso_oltc, shunt_bus_indices=tso_shunt_buses,
        shunt_q_steps_mvar=tso_shunt_q,
        voltage_bus_indices=tso_v_buses, current_line_indices=tso_lines,
        current_line_max_i_ka=tso_line_max_i_ka,
        v_setpoints_pu=v_setpoints, g_v=g_v,
        gen_indices=tso_gen_indices, gen_bus_indices=tso_gen_bus_indices,
    )
    dso_v_setpoints = np.full(len(dso_v_buses), dso_v_setpoint_pu)
    dso_config = DSOControllerConfig(
        der_bus_indices=dso_der_buses, oltc_trafo_indices=dso_oltc,
        shunt_bus_indices=dso_shunt_buses, shunt_q_steps_mvar=dso_shunt_q,
        interface_trafo_indices=dso_iface_trafos,
        voltage_bus_indices=dso_v_buses, current_line_indices=dso_lines,
        current_line_max_i_ka=dso_line_max_i_ka,
        g_q=g_q,
        v_setpoints_pu=dso_v_setpoints, g_v=config.dso_g_v,
    )

    # -- Live plotter (created here because configs are now available)
    if live_plot and live_plotter is None:
        from visualisation.plot_cascade import LivePlotter
        live_plotter = LivePlotter(
            tso_config, dso_config,
            tso_line_max_i_ka=np.array(tso_line_max_i_ka, dtype=np.float64)*tso_config.i_max_pu,
            dso_line_max_i_ka=np.array(dso_line_max_i_ka, dtype=np.float64)*dso_config.i_max_pu,
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

    # Generator capability parameters (Milano §12.2.1, from config)
    tso_gen_params = [
        GeneratorParameters(
            s_rated_mva=float(net.gen.at[g, "sn_mva"]),
            p_max_mw=float(net.gen.at[g, "p_mw"]),
            xd_pu=config.gen_xd_pu,
            i_f_max_pu=config.gen_i_f_max_pu,
            beta=config.gen_beta,
            q0_pu=config.gen_q0_pu,
        )
        for g in tso_gen_indices
    ]

    tso_bounds = ActuatorBounds(
        der_indices=np.array(tso_der_buses, dtype=np.int64),
        der_s_rated_mva=tso_s, der_p_max_mw=tso_p,
        oltc_indices=np.array(tso_oltc, dtype=np.int64),
        oltc_tap_min=np.array([int(net.trafo.at[t, "tap_min"]) for t in tso_oltc], dtype=np.int64),
        oltc_tap_max=np.array([int(net.trafo.at[t, "tap_max"]) for t in tso_oltc], dtype=np.int64),
        shunt_indices=np.array(tso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(tso_shunt_q, dtype=np.float64),
        gen_params=tso_gen_params,
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
    # =====================================================================
    #  g_w calibration (MIQP quadratic weighting on control changes)
    # =====================================================================
    #
    # With alpha = 1, continuous and integer variables are treated
    # identically: w = Δu (direct change).  The MIQP objective is
    #
    #   min  w^T G_w w  +  grad_f^T w
    #
    # where G_w = diag(g_w + g_u).
    #
    # The unconstrained optimum for a single variable is:
    #   w* = −grad_f / (2 * (g_w + g_u))
    #
    # So g_w directly controls the step size in physical units:
    #   • Q_DER / Q_PCC_set [Mvar]: g_w=100 → 1 Mvar costs 100
    #   • V_gen [p.u.]:             g_w=1e9 → very cautious AVR moves
    #   • s_OLTC [taps]:            g_w=100 → 1-tap costs 100
    #   • s_shunt [states]:         g_w=500 → 1-step costs 500
    #
    # =====================================================================
    # TSO u = [Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt]
    gw_tso_diag = config.build_gw_tso_diag(
        n_der=len(tso_der_buses), n_pcc=len(pcc_trafos),
        n_gen=len(tso_gen_indices), n_oltc=len(tso_oltc),
        n_shunt=len(tso_shunt_buses),
    )
    # DSO u = [Q_DER | s_OLTC | s_shunt]
    gw_dso_diag = config.build_gw_dso_diag(
        n_der=len(dso_der_buses), n_oltc=len(dso_oltc),
        n_shunt=len(dso_shunt_buses),
    )

    # Build G_w matrices — full 2-D if OLTC cross-coupling is requested,
    # otherwise keep as 1-D diagonal vector (cheaper for solver).
    #
    # Cross-coupling penalises *simultaneous* OLTC switching:
    #
    #   g_cross · (Σ_i w_i)²  =  g_cross · Σ_i Σ_j w_i · w_j
    #
    # This adds +g_cross to EVERY element (diagonal AND off-diagonal)
    # of the OLTC sub-block.  The resulting sub-block is
    #
    #   G_oltc = diag(d) + g_cross · 𝟏𝟏ᵀ
    #
    # which is PSD (rank-1 update of a positive diagonal).  Eigenvalues
    # are  d  (multiplicity n−1)  and  d + n·g_cross  (multiplicity 1).
    #
    # Effect on switching decisions:
    #   • Single OLTC taps by ±1:  cost = d + g_cross  (modest extra)
    #   • Two OLTCs both tap ±1:   cost = 2d + 4·g_cross  (heavy extra)
    # So the solver is steered toward tapping one OLTC at a time.
    def _build_Gw(diag_vec, n_pre_oltc, n_oltc, cross_weight):
        """Return 1-D diag vector or full 2-D G_w matrix."""
        if cross_weight == 0.0 or n_oltc <= 1:
            return diag_vec          # no coupling needed
        n = len(diag_vec)
        G = np.diag(diag_vec.copy())
        i0 = n_pre_oltc
        i1 = n_pre_oltc + n_oltc
        # Add g_cross · 𝟏𝟏ᵀ to the OLTC sub-block (all elements)
        G[i0:i1, i0:i1] += cross_weight
        return G

    n_pre_oltc_tso = len(tso_der_buses) + len(pcc_trafos) + len(tso_gen_indices)
    gw_tso = _build_Gw(gw_tso_diag, n_pre_oltc_tso, len(tso_oltc),
                        gw_oltc_cross_tso)

    n_pre_oltc_dso = len(dso_der_buses)
    gw_dso = _build_Gw(gw_dso_diag, n_pre_oltc_dso, len(dso_oltc),
                        gw_oltc_cross_dso)


    # Per-actuator g_u: penalises DER Q *level* (deviation from zero) to
    # incentivise freeing up DER headroom via shunt/OLTC switching.
    # With alpha=1 the g_u value adds directly to the G_w diagonal.
    gu_tso = config.build_gu_tso(
        n_der=len(tso_der_buses), n_pcc=len(pcc_trafos),
        n_gen=len(tso_gen_indices), n_oltc=len(tso_oltc),
        n_shunt=len(tso_shunt_buses),
    )
    gu_dso = config.build_gu_dso(
        n_der=len(dso_der_buses), n_oltc=len(dso_oltc),
        n_shunt=len(dso_shunt_buses),
    )
    gz_tso = config.build_gz_tso(n_v=len(tso_v_buses), n_i=len(tso_lines))
    gz_dso = config.build_gz_dso(
        n_iface=len(dso_iface_trafos), n_v=len(dso_v_buses), n_i=len(dso_lines))
    ofo_tso = OFOParameters(alpha=config.alpha, g_w=gw_tso, g_z=gz_tso, g_u=gu_tso)
    ofo_dso = OFOParameters(alpha=config.alpha, g_w=gw_dso, g_z=gz_dso, g_u=gu_dso)

    ns = _network_state(net)

    tso = TSOController("tso_main", ofo_tso, tso_config, ns, tso_bounds,
                        JacobianSensitivities(net))
    dso = DSOController(dso_id, ofo_dso, dso_config, ns, dso_bounds,
                        JacobianSensitivities(net))

    # Reserve Observer: monitors per-interface DER Q contribution
    # (Jacobian-weighted) and forces shunt engagement at the
    # tertiary winding when the DER burden at that interface
    # exceeds the threshold.  1:1 mapping between couplers and shunts.
    reserve_obs = ReserveObserver(
        config.build_reserve_observer_config(shunt_q_steps_mvar=dso_shunt_q)
    )

    # 7) Initialise actuators to a neutral operating point
    #    - DER Q = 0, TSO OLTC = 0, shunts = 0
    #    - Generator AVR setpoints = 1.05 p.u.
    #    - DSO 3W OLTCs: regulated via pandapower tap controllers to 1.05 p.u.
    #      on the 110 kV (MV) side, then controllers are removed.
    for s in net.sgen.index:
        if not str(net.sgen.at[s, "name"]).startswith("BOUND_"):
            net.sgen.at[s, "q_mvar"] = 0.0
    for t in tso_oltc:
        net.trafo.at[t, "tap_pos"] = 0
    for g in tso_gen_indices:
        net.gen.at[g, "vm_pu"] = v_setpoint_pu #v_setpoint_pu
    for sb in tso_shunt_buses:
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = 0
    for sb in dso_shunt_buses:
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = 0

    # Use pandapower DiscreteTapControl to find the DSO OLTC positions that
    # regulate the 110 kV side to ~dso_v_setpoint_pu, then remove the controllers.
    from pandapower.control import DiscreteTapControl
    tol_pu = config.dso_oltc_init_tol_pu
    for t3w in dso_oltc:
        DiscreteTapControl(
            net, element_index=t3w,
            vm_lower_pu=dso_v_setpoint_pu - tol_pu,
            vm_upper_pu=dso_v_setpoint_pu + tol_pu,
            side="mv", element="trafo3w",
        )
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if verbose > 1:
        for t3w in dso_oltc:
            tap = int(net.trafo3w.at[t3w, "tap_pos"])
            print(f"  DSO OLTC trafo3w {t3w}: initial tap_pos = {tap}")
    # Remove pandapower controllers so they don't interfere with OFO
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Run a clean PF with the initial timestep (t=0) and neutral actuators
    # so that the measurement used for initialise() is fully consistent
    # with the actuator state set above (DER Q=0, OLTC=neutral, shunts=0).
    if use_profiles:
        apply_profiles(net, profiles, start_time)
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Initialise controller u_current from converged PF results
    tso.initialise(_measure_tso(net, tso_config, 0))
    dso.initialise(_measure_dso(net, dso_config, 0))

    if verbose > 0:
        print(f"Running cascade: {n_minutes} min  (TSO/{tso_period_min}m, DSO/{dso_period_min}m)")
        print(f"  TSO: {len(tso_der_buses)} DER, {len(tso_oltc)} OLTC, "
              f"{len(tso_gen_indices)} gen, {len(tso_shunt_buses)} shunt")
        print(f"  DSO: {len(dso_der_buses)} DER, {len(dso_oltc)} OLTC, "
              f"{len(dso_shunt_buses)} shunt")
        if gw_oltc_cross_tso != 0.0:
            n_o = len(tso_oltc)
            eff_diag = 50 + gw_oltc_cross_tso
            print(f"  TSO OLTC cross-coupling: g_cross={gw_oltc_cross_tso}, "
                  f"eff. diag={eff_diag}, off-diag=+{gw_oltc_cross_tso}, "
                  f"G_w shape={np.asarray(gw_tso).shape}")
        if gw_oltc_cross_dso != 0.0:
            n_o = len(dso_oltc)
            eff_diag = 40 + gw_oltc_cross_dso
            print(f"  DSO OLTC cross-coupling: g_cross={gw_oltc_cross_dso}, "
                  f"eff. diag={eff_diag}, off-diag=+{gw_oltc_cross_dso}, "
                  f"G_w shape={np.asarray(gw_dso).shape}")

        # --- Print initialised u_current for both controllers ---
        u_tso = tso.u_current
        n_d = len(tso_config.der_bus_indices)
        n_p = len(tso_config.pcc_trafo_indices)
        n_g = len(tso_config.gen_indices)
        n_o = len(tso_config.oltc_trafo_indices)
        n_s = len(tso_config.shunt_bus_indices)
        off = 0
        print(f"  TSO u_init ({len(u_tso)} vars):")
        print(f"    Q_DER       = {A2S(u_tso[off:off+n_d])} Mvar");  off += n_d
        print(f"    Q_PCC_set   = {A2S(u_tso[off:off+n_p])} Mvar");  off += n_p
        print(f"    V_gen_set   = {A3S(u_tso[off:off+n_g])} pu");    off += n_g
        print(f"    s_OLTC      = {u_tso[off:off+n_o].astype(int).tolist()}");  off += n_o
        print(f"    s_shunt     = {u_tso[off:off+n_s].astype(int).tolist()}")

        u_dso = dso.u_current
        n_dd = len(dso_config.der_bus_indices)
        n_do = len(dso_config.oltc_trafo_indices)
        n_ds = len(dso_config.shunt_bus_indices)
        print(f"  DSO u_init ({len(u_dso)} vars):")
        print(f"    Q_DER       = {A2S(u_dso[:n_dd])} Mvar")
        print(f"    s_OLTC      = {u_dso[n_dd:n_dd+n_do].astype(int).tolist()}")
        print(f"    s_shunt     = {u_dso[n_dd+n_do:n_dd+n_do+n_ds].astype(int).tolist()}")
        print()

    # 8) Cascade loop
    log: List[IterationRecord] = []

    for minute in range(1, n_minutes + 1):
        run_tso = (minute == 1) or (minute % tso_period_min == 0)
        run_dso = (minute % dso_period_min == 0)
        rec = IterationRecord(minute=minute, tso_active=run_tso, dso_active=run_dso)

        # Apply time-series profiles for this minute
        if use_profiles:
            t_now = start_time + timedelta(minutes=minute)
            apply_profiles(net, profiles, t_now)

        # ── Apply contingency events ──────────────────────────────────
        if contingencies:
            fired = [ev for ev in contingencies if ev.minute == minute]
            if fired:
                descs = []
                for ev in fired:
                    descs.append(_apply_contingency(net, ev, verbose))
                rec.contingency_events = descs

                # Re-converge PF with new topology so Jacobian is valid
                pp.runpp(net, run_control=False,
                         calculate_voltage_angles=True)

                # Refresh sensitivity matrices for both controllers
                tso.sensitivities = JacobianSensitivities(net)
                dso.sensitivities = JacobianSensitivities(net)
                tso.invalidate_sensitivity_cache()
                dso.invalidate_sensitivity_cache()

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
                if verbose > 0:
                    print(f"  [min {minute:3d}] TSO FAILED: {e}")

        # DSO step
        if run_dso:
            try:
                dso_meas = _measure_dso(net, dso_config, minute)

                # Reserve Observer: per-interface DER Q burden → shunt overrides
                if enable_reserve_observer and dso._u_current is not None and len(dso_config.shunt_bus_indices) > 0:
                    n_dd = len(dso_config.der_bus_indices)
                    n_do = len(dso_config.oltc_trafo_indices)
                    n_ds = len(dso_config.shunt_bus_indices)
                    n_iface = len(dso_config.interface_trafo_indices)
                    der_q = dso._u_current[:n_dd]
                    shunt_st = np.round(dso._u_current[n_dd + n_do:n_dd + n_do + n_ds]).astype(np.int64)

                    # Extract dQ_interface/dQ_DER sub-matrix from DSO H cache
                    dQ_dQder = dso.get_interface_der_sensitivity()

                    # --- Debug: state BEFORE evaluate ---
                    if verbose > 2:
                        print(f"    [ReserveObs min {minute}] --- BEFORE evaluate ---")
                        print(f"      engaged_flags = {reserve_obs._engaged}")
                        print(f"      shunt_states  = {shunt_st.tolist()}")
                        print(f"      der_q (sum={np.sum(der_q):.2f} Mvar):")
                        # Show top-5 DERs by |Q|
                        top5 = np.argsort(np.abs(der_q))[-5:][::-1]
                        for k in top5:
                            print(f"        DER[{k}] bus={dso_config.der_bus_indices[k]}: "
                                  f"Q={der_q[k]:.2f} Mvar")
                        if dQ_dQder is not None:
                            q_contribution = dQ_dQder @ der_q
                            print(f"      dQ_dQder shape = {dQ_dQder.shape}")
                            for j in range(n_iface):
                                q_step_j = reserve_obs.config.shunt_q_steps_mvar[j]
                                burden_j = q_contribution[j] if q_step_j > 0 else -q_contribution[j]
                                row_nz = np.count_nonzero(dQ_dQder[j, :])
                                row_sum = np.sum(dQ_dQder[j, :])
                                print(f"      iface[{j}] trafo3w={dso_config.interface_trafo_indices[j]}:")
                                print(f"        q_contribution = {q_contribution[j]:.2f} Mvar  "
                                      f"(dQ_dQder row: {row_nz} nonzero, sum={row_sum:.4f})")
                                print(f"        q_step = {q_step_j:.1f} Mvar  "
                                      f"-> burden = {burden_j:.2f} Mvar")
                                print(f"        thresholds: engage={reserve_obs.config.q_threshold_mvar:.1f}, "
                                      f"release={reserve_obs.config.q_release_mvar:.1f}")
                                engaged_before = reserve_obs._engaged[j]
                                last_act = reserve_obs._last_action_min[j]
                                cd_remaining = max(0, reserve_obs.config.cooldown_min - (minute - last_act))
                                in_cd = cd_remaining > 0
                                cd_str = (f"COOLDOWN {cd_remaining}min remaining"
                                          if in_cd else "no cooldown")
                                print(f"        cooldown: {cd_str}  "
                                      f"(last_action=min {last_act})")
                                if not engaged_before:
                                    would_engage = burden_j > reserve_obs.config.q_threshold_mvar and not in_cd
                                    reason = ("BLOCKED by cooldown" if in_cd and burden_j > reserve_obs.config.q_threshold_mvar
                                              else ('WILL ENGAGE' if would_engage else 'stays disengaged'))
                                    print(f"        state=DISENGAGED  "
                                          f"-> {reason} "
                                          f"(burden {'>' if burden_j > reserve_obs.config.q_threshold_mvar else '<='} threshold)")
                                else:
                                    would_release = burden_j < reserve_obs.config.q_release_mvar and not in_cd
                                    reason = ("BLOCKED by cooldown" if in_cd and burden_j < reserve_obs.config.q_release_mvar
                                              else ('WILL RELEASE' if would_release else 'stays engaged'))
                                    print(f"        state=ENGAGED  "
                                          f"-> {reason} "
                                          f"(burden {'<' if burden_j < reserve_obs.config.q_release_mvar else '>='} release)")
                        else:
                            print(f"      dQ_dQder = None (fallback: aggregate sum)")

                    obs_result = reserve_obs.evaluate(der_q, shunt_st, dQ_dQder, minute=minute)

                    # --- Debug: result AFTER evaluate ---
                    if verbose > 2:
                        print(f"      --- AFTER evaluate ---")
                        print(f"      engaged_flags = {reserve_obs._engaged}")
                        print(f"      force_engage  = {obs_result.force_engage}")
                        print(f"      force_release = {obs_result.force_release}")

                    overrides = {}
                    for idx in obs_result.force_engage:
                        overrides[idx] = (1, 1)   # force shunt ON
                    for idx in obs_result.force_release:
                        overrides[idx] = (0, 0)   # force shunt OFF
                    if overrides:
                        if verbose > 2:
                            print(f"      -> APPLYING overrides: {overrides}")
                        dso.set_shunt_overrides(overrides)
                    elif verbose > 2:
                        print(f"      -> no overrides")

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
                if verbose > 0:
                    print(f"  [min {minute:3d}] DSO FAILED: {e}")

        # Power flow
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)

        # Record plant voltages, line currents, generator Q, and actual interface Q after PF
        rec.plant_tn_voltages_pu = net.res_bus.loc[tso_v_buses, "vm_pu"].values.copy()
        rec.plant_dn_voltages_pu = net.res_bus.loc[dso_v_buses, "vm_pu"].values.copy()
        rec.plant_tn_currents_ka = np.array(
            [float(net.res_line.at[li, "i_from_ka"]) for li in tso_lines],
            dtype=np.float64)
        rec.plant_dn_currents_ka = np.array(
            [float(net.res_line.at[li, "i_from_ka"]) for li in dso_lines],
            dtype=np.float64)
        rec.tso_q_gen_mvar = np.array(
            [float(net.res_gen.at[g, "q_mvar"]) for g in tso_gen_indices],
            dtype=np.float64)
        if run_dso:
            rec.dso_q_actual_mvar = np.array(
                [float(net.res_trafo3w.at[t, "q_hv_mvar"])
                 for t in dso_config.interface_trafo_indices], dtype=np.float64)

        # Penalty terms from plant measurements
        if tso_config.v_setpoints_pu is not None:
            v_err = rec.plant_tn_voltages_pu - tso_config.v_setpoints_pu
            rec.tso_v_penalty = float(tso_config.g_v * np.sum(v_err ** 2))
        if rec.dso_q_actual_mvar is not None and dso.q_setpoint_mvar is not None:
            q_err = rec.dso_q_actual_mvar - dso.q_setpoint_mvar
            rec.dso_q_penalty = float(dso_config.g_q * np.sum(q_err ** 2))

        log.append(rec)

        if live_plotter is not None:
            live_plotter.update(rec)

        if verbose > 0 and (run_tso or run_dso):
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
                print(f"      V_gen   = {A3S(rec.tso_v_gen_pu)} pu")
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

            # DEBUG ########################################################################
            # # --- DER active power injections ---
            # tso_der_p = np.array(
            #     [float(net.res_sgen.at[s, 'p_mw'])
            #      for s in net.sgen.index
            #      if int(net.sgen.at[s, 'bus']) not in dn_buses
            #      and not str(net.sgen.at[s, 'name']).startswith('BOUND')],
            #     dtype=np.float64,
            # )
            # dso_der_p = np.array(
            #     [float(net.res_sgen.at[s, 'p_mw'])
            #      for s in net.sgen.index
            #      if int(net.sgen.at[s, 'bus']) in dn_buses
            #      and not str(net.sgen.at[s, 'name']).startswith('BOUND')],
            #     dtype=np.float64,
            # )
            # gen_p = np.array(
            #     [float(net.res_gen.at[g, 'p_mw']) for g in tso_gen_indices],
            #     dtype=np.float64,
            # )
            #
            # # --- Load active power consumption ---
            # tso_load_p = np.array(
            #     [float(net.res_load.at[l, 'p_mw'])
            #      for l in net.load.index
            #      if int(net.load.at[l, 'bus']) not in dn_buses],
            #     dtype=np.float64,
            # )
            # dso_load_p = np.array(
            #     [float(net.res_load.at[l, 'p_mw'])
            #      for l in net.load.index
            #      if int(net.load.at[l, 'bus']) in dn_buses],
            #     dtype=np.float64,
            # )
            #
            # print(f"  TSO DER P  sum={tso_der_p.sum():.1f} MW  {A2S(tso_der_p)} MW")
            # print(f"  DSO DER P  sum={dso_der_p.sum():.1f} MW  {A2S(dso_der_p)} MW")
            # if len(gen_p):
            #     print(f"  TSO Gen P  sum={gen_p.sum():.1f} MW  {A2S(gen_p)} MW")
            # print(f"  TSO Load P sum={tso_load_p.sum():.1f} MW  {A2S(tso_load_p)} MW")
            # print(f"  DSO Load P sum={dso_load_p.sum():.1f} MW  {A2S(dso_load_p)} MW")
            # --- DEBUG: Print Actual Currents and Limits ---
            # tso_i_actual = np.array(
            #     [float(net.res_line.at[li, 'i_from_ka']) for li in tso_lines],
            #     dtype=np.float64,
            # )
            # dso_i_actual = np.array(
            #     [float(net.res_line.at[li, 'i_from_ka']) for li in dso_lines],
            #     dtype=np.float64,
            # )
            #
            # tso_i_lim = np.array(tso_config.current_line_max_i_ka, dtype=np.float64)
            # dso_i_lim = np.array(dso_config.current_line_max_i_ka, dtype=np.float64)
            #
            # tso_violated = np.any(tso_i_actual > tso_i_lim * tso_config.i_max_pu)
            # dso_violated = np.any(dso_i_actual > dso_i_lim * dso_config.i_max_pu)
            #
            # tso_flag = "  *** VIOLATION ***" if tso_violated else ""
            # dso_flag = "  *** VIOLATION ***" if dso_violated else ""
            #
            # # Calculate maximum currents
            # tso_max_i = np.max(tso_i_actual) if len(tso_i_actual) > 0 else 0.0
            # dso_max_i = np.max(dso_i_actual) if len(dso_i_actual) > 0 else 0.0
            #
            # print(
            #     f"  TN I actual [kA]  {A2S(tso_i_actual)}")
            # print(
            #     f"  (lim×{tso_config.i_max_pu:.2f}: {A2S(tso_i_lim * tso_config.i_max_pu)}){tso_flag}")
            # print(f"  TN I max    [kA]  {tso_max_i:.3f}")
            # print(
            #     f"  DN I actual [kA]  {A2S(dso_i_actual)}")
            # print(
            #     f"  (lim×{dso_config.i_max_pu:.2f}: {A2S(dso_i_lim * dso_config.i_max_pu)}){dso_flag}")
            # print(f"  DN I max    [kA]  {dso_max_i:.3f}")

    if live_plotter is not None:
        live_plotter.finish()

    return CascadeResult(log=log, tso_config=tso_config, dso_config=dso_config)


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
    import time as _time
    from core.cascade_config import CascadeConfig
    from core.results_storage import save_results

    # ── Configuration (single place for ALL parameters) ───────────────────
    config = CascadeConfig(
        # Simulation
        v_setpoint_pu=1.05,
        n_minutes=2 * 60,
        tso_period_min=3,
        dso_period_min=1,
        start_time=datetime(2016, 5, 1, 0, 0),
        use_profiles=True,
        verbose=1,
        live_plot=True,

        # Objective weights
        g_v=200000,
        g_q=1,
        dso_g_v=10000.0,

        # OFO
        alpha=1.0,
        g_z=1e12,

        # TSO g_w
        gw_tso_q_der=0.4,
        gw_tso_q_pcc=0.2,
        gw_tso_v_gen=5e6,
        gw_tso_oltc=40.0,
        gw_tso_shunt=2000.0,
        gw_oltc_cross_tso=0.0,

        # DSO g_w
        gw_dso_q_der=4.0,
        gw_dso_oltc=100.0,
        gw_dso_shunt=3000.0,
        gw_oltc_cross_dso=0.0,

        # Generator capability
        gen_xd_pu=1.2,
        gen_i_f_max_pu=2.65,
        gen_beta=0.15,
        gen_q0_pu=0.4,

        # Reserve Observer
        enable_reserve_observer=True,
        reserve_q_threshold_mvar=40.0,
        reserve_q_release_mvar=-40.0,
        reserve_cooldown_min=3,

        # Contingencies
        contingencies=[
            ContingencyEvent(minute=90, element_type="line", element_index=3),
            ContingencyEvent(minute=120, element_type="line", element_index=11),
            #ContingencyEvent(minute=120, element_type="line", element_index=16),
            ContingencyEvent(minute=180, element_type="gen", element_index=0),
            ContingencyEvent(minute=210, element_type="line", element_index=3, action="restore"),
            #ContingencyEvent(minute=210, element_type="line", element_index=16, action="restore"),
            ContingencyEvent(minute=240, element_type="line", element_index=11, action="restore"),
            ContingencyEvent(minute=400, element_type="gen",  element_index=0, action="restore"),
        ],
    )

    # ── Run ────────────────────────────────────────────────────────────────
    t0 = _time.perf_counter()
    result = run_cascade(config)
    wall_time = _time.perf_counter() - t0

    print_summary(config.v_setpoint_pu, result.log)

    # ── Save results ──────────────────────────────────────────────────────
    run_dir = save_results(result, config, wall_time_s=wall_time)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
