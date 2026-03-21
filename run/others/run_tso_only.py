#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSO-Only OFO Voltage Controller
================================

Runs the TSO layer of the cascaded OFO on the **combined** network, but
**without** a DSO controller.  Key differences from ``run_cascade``:

1. **No DSO controller** — Q_PCC is removed from TSO optimisation variables.
2. **DN DER Q = 0** — distribution-connected DERs inject zero reactive power.
3. **Coupling-transformer OLTCs** are operated by pandapower
   ``DiscreteTapControl`` (local AVR) regulating the 110 kV side to the
   DSO voltage setpoint.  These controllers remain active during the
   simulation loop (``run_control=True``).

Everything else (network, profiles, sensitivities, TSO config/weights,
contingencies, timing) matches ``run_cascade`` exactly.

Author: Manuel Schwenke
"""

from __future__ import annotations

import os
import sys
import time as _time
import warnings
from datetime import datetime, timedelta
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import pandapower as pp
from pandapower.control import DiscreteTapControl

warnings.filterwarnings("ignore", category=UserWarning, module=r"mosek")

from controller.base_controller import OFOParameters
from controller.dso_controller import DSOControllerConfig
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from core.cascade_config import CascadeConfig
from core.measurement import measure_tso as _measure_tso
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from network.build_tuda_net import build_tuda_net
from run.contingency import _apply_contingency
from run.helpers import _build_Gw, _network_state
from run.plant_io import _apply_tso
from run.records import A2S, A3S, ContingencyEvent, IterationRecord
from sensitivity.jacobian import JacobianSensitivities


# ==============================================================================
#  MAIN TSO-ONLY RUNNER
# ==============================================================================

def run_tso_only(
    config: CascadeConfig,
    *,
    live_plotter=None,
) -> List[IterationRecord]:
    """
    Run a TSO-only OFO voltage controller on the combined network.

    Uses the same combined-network architecture as ``run_cascade``:
    - Controls applied to combined network
    - Measurements taken from combined network
    - Sensitivities computed from combined network

    Coupling-transformer OLTCs are regulated by pandapower
    ``DiscreteTapControl`` (local AVR, 110 kV side).

    Parameters
    ----------
    config : CascadeConfig
        Central configuration (same object as for ``run_cascade``).
    live_plotter : optional
        Object with ``update(rec)`` / ``finish()`` methods, e.g.
        :class:`visualisation.plot_cascade.LivePlotter`.

    Returns
    -------
    log : list[IterationRecord]
        One record per simulation minute.
    """
    v_setpoint_pu = config.v_setpoint_pu
    dso_v_setpoint_pu = config.effective_dso_v_setpoint_pu
    n_minutes = config.n_minutes
    tso_period_min = config.tso_period_min
    start_time = config.start_time
    profiles_csv = config.profiles_csv
    verbose = config.verbose
    live_plot = config.live_plot
    use_profiles = config.use_profiles
    contingencies = list(config.contingencies) if config.contingencies else []

    if verbose > 0:
        print("=" * 72)
        print(f"  TSO-ONLY OFO  --  V_set = {v_setpoint_pu:.3f} p.u.")
        print(f"  Coupling OLTCs: local AVR -> {dso_v_setpoint_pu:.3f} p.u. (110 kV)")
        print(f"  DN DER Q: forced to 0 Mvar")
        print(
            f"  Profiles: {os.path.basename(profiles_csv)}  "
            f"start={start_time:%d.%m.%Y %H:%M}  {n_minutes} min"
        )
        print("=" * 72)
        if contingencies:
            print(f"  Scheduled contingencies ({len(contingencies)}):")
            for ev in contingencies:
                print(
                    f"    min {ev.minute:4d}: {ev.action.upper()} "
                    f"{ev.element_type} {ev.element_index}"
                )

    # 1) Build combined network
    net, meta = build_tuda_net(ext_grid_vm_pu=v_setpoint_pu, pv_nodes=True)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Load time-series profiles and snapshot base load/gen values
    profiles = load_profiles(profiles_csv, timestep_min=1)
    snapshot_base_values(net)

    # Apply profiles at t=0 and re-run PF
    if use_profiles:
        apply_profiles(net, profiles, start_time)
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # 2) Identify elements
    dn_buses = {int(b) for b in net.bus.index if str(net.bus.at[b, "subnet"]) == "DN"}

    # -- TSO DER: individual sgen indices (not DN, not boundary)
    tso_der_indices = [
        int(s)
        for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) not in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ]
    tso_der_buses = [int(net.sgen.at[s, "bus"]) for s in tso_der_indices]

    # -- TSO monitored: 380 kV buses, TN lines
    tso_v_buses = sorted(
        int(b) for b in net.bus.index if float(net.bus.at[b, "vn_kv"]) >= 300.0
    )
    tso_lines = sorted(
        int(li) for li in net.line.index if str(net.line.at[li, "subnet"]) == "TN"
    )

    # -- TSO generators + machine trafos
    tso_gen_indices, tso_gen_bus_indices = [], []
    for g in net.gen.index:
        tso_gen_indices.append(int(g))
        lv = int(net.gen.at[g, "bus"])
        mt = net.trafo.index[
            (net.trafo["lv_bus"] == lv)
            & net.trafo["name"].astype(str).str.startswith("MachineTrf|")
        ]
        if mt.empty:
            raise RuntimeError(f"No machine trafo for gen {g} bus {lv}")
        tso_gen_bus_indices.append(int(net.trafo.at[mt[0], "hv_bus"]))

    # -- TSO 380 kV shunts
    tso_shunt_buses_cand = [
        int(net.shunt.at[s, "bus"])
        for s in meta.tn_shunt_indices
        if s in net.shunt.index
    ]
    tso_shunt_q_cand = [
        float(net.shunt.at[s, "q_mvar"])
        for s in meta.tn_shunt_indices
        if s in net.shunt.index
    ]

    # -- DSO DER: individual sgen indices in DN (for recording, not controlled)
    dso_der_indices = [
        int(s)
        for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ]

    # -- DSO monitored: 110 kV DN buses, DN lines (for live plot)
    dso_v_buses = sorted(
        int(b)
        for b in net.bus.index
        if 100.0 <= float(net.bus.at[b, "vn_kv"]) < 200.0 and int(b) in dn_buses
    )
    dso_lines = sorted(
        int(li) for li in net.line.index if str(net.line.at[li, "subnet"]) == "DN"
    )

    # -- DSO tertiary shunts (for live plot recording)
    dso_shunt_buses_cand = [
        int(net.shunt.at[s, "bus"])
        for s in meta.tertiary_shunt_indices
        if s in net.shunt.index
    ]
    dso_shunt_q_cand = [
        float(net.shunt.at[s, "q_mvar"])
        for s in meta.tertiary_shunt_indices
        if s in net.shunt.index
    ]

    # -- No PCC control in TSO-only mode
    pcc_trafos: List[int] = []
    coupler_3w_indices = list(meta.coupler_trafo3w_indices)

    # 3) Sensitivities from combined network
    if verbose > 1:
        print("Computing sensitivities from combined network ...")

    tso_sens = JacobianSensitivities(net)
    _, m_tso = tso_sens.build_sensitivity_matrix_H(
        der_bus_indices=tso_der_buses,
        observation_bus_indices=tso_v_buses,
        line_indices=tso_lines,
        oltc_trafo_indices=list(meta.machine_trafo_indices),
        shunt_bus_indices=tso_shunt_buses_cand,
        shunt_q_steps_mvar=tso_shunt_q_cand,
    )
    tso_oltc = list(m_tso.get("oltc_trafos", []))
    tso_v_buses = list(m_tso.get("obs_buses", tso_v_buses))
    tso_shunt_buses = list(m_tso.get("shunt_buses", []))
    tso_shunt_q = [
        tso_shunt_q_cand[tso_shunt_buses_cand.index(b)] for b in tso_shunt_buses
    ]

    # 4) Controller config — same as run_cascade but pcc_trafo_indices = []
    v_setpoints = np.full(len(tso_v_buses), v_setpoint_pu)
    tso_line_max_i_ka = [float(net.line.at[li, "max_i_ka"]) for li in tso_lines]
    dso_line_max_i_ka = [float(net.line.at[li, "max_i_ka"]) for li in dso_lines]

    tso_config = TSOControllerConfig(
        der_indices=tso_der_indices,
        pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[],
        oltc_trafo_indices=tso_oltc,
        shunt_bus_indices=tso_shunt_buses,
        shunt_q_steps_mvar=tso_shunt_q,
        voltage_bus_indices=tso_v_buses,
        current_line_indices=tso_lines,
        current_line_max_i_ka=tso_line_max_i_ka,
        v_setpoints_pu=v_setpoints,
        g_v=config.g_v,
        gen_indices=tso_gen_indices,
        gen_bus_indices=tso_gen_bus_indices,
        k_t_avt=config.k_t_avt,
    )

    # DSO config — not used for control, only to drive LivePlotter axes
    dso_v_setpoints = np.full(len(dso_v_buses), dso_v_setpoint_pu)
    dso_config = DSOControllerConfig(
        der_indices=dso_der_indices,
        oltc_trafo_indices=coupler_3w_indices,
        shunt_bus_indices=dso_shunt_buses_cand,
        shunt_q_steps_mvar=dso_shunt_q_cand,
        interface_trafo_indices=coupler_3w_indices,
        voltage_bus_indices=dso_v_buses,
        current_line_indices=dso_lines,
        current_line_max_i_ka=dso_line_max_i_ka,
        g_q=config.g_q,
        v_setpoints_pu=dso_v_setpoints,
        g_v=config.dso_g_v,
    )

    # -- Live plotter
    if live_plot and live_plotter is None:
        from visualisation.plot_cascade import LivePlotter

        live_plotter = LivePlotter(
            tso_config,
            dso_config,
            tso_line_max_i_ka=np.array(tso_line_max_i_ka, dtype=np.float64)
            * tso_config.i_max_pu,
            dso_line_max_i_ka=np.array(dso_line_max_i_ka, dtype=np.float64)
            * dso_config.i_max_pu,
        )

    # 5) Actuator bounds (from combined network)
    tso_s = np.array(
        [float(net.sgen.at[s, "sn_mva"]) for s in tso_der_indices], dtype=np.float64
    )
    tso_p = np.array(
        [float(net.sgen.at[s, "p_mw"]) for s in tso_der_indices], dtype=np.float64
    )

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
        der_indices=np.array(tso_der_indices, dtype=np.int64),
        der_s_rated_mva=tso_s,
        der_p_max_mw=tso_p,
        oltc_indices=np.array(tso_oltc, dtype=np.int64),
        oltc_tap_min=np.array(
            [int(net.trafo.at[t, "tap_min"]) for t in tso_oltc], dtype=np.int64
        ),
        oltc_tap_max=np.array(
            [int(net.trafo.at[t, "tap_max"]) for t in tso_oltc], dtype=np.int64
        ),
        shunt_indices=np.array(tso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(tso_shunt_q, dtype=np.float64),
        gen_params=tso_gen_params,
    )

    # 6) Create TSO controller with same g_w / g_u / g_z as run_cascade
    gw_tso_diag = config.build_gw_tso_diag(
        n_der=len(tso_der_buses),
        n_pcc=0,  # no PCC
        n_gen=len(tso_gen_indices),
        n_oltc=len(tso_oltc),
        n_shunt=len(tso_shunt_buses),
    )
    n_pre_oltc_tso = len(tso_der_buses) + 0 + len(tso_gen_indices)
    gw_tso = _build_Gw(gw_tso_diag, n_pre_oltc_tso, len(tso_oltc), config.gw_oltc_cross_tso)

    gu_tso = config.build_gu_tso(
        n_der=len(tso_der_buses),
        n_pcc=0,
        n_gen=len(tso_gen_indices),
        n_oltc=len(tso_oltc),
        n_shunt=len(tso_shunt_buses),
    )
    gz_tso = config.build_gz_tso(n_v=len(tso_v_buses), n_i=len(tso_lines))
    ofo_tso = OFOParameters(alpha=config.alpha, g_w=gw_tso, g_z=gz_tso, g_u=gu_tso)

    ns = _network_state(net)

    tso = TSOController(
        "tso_main", ofo_tso, tso_config, ns, tso_bounds, JacobianSensitivities(net)
    )
    tso._avt_verbose = verbose

    # 7) Initialise actuators to neutral operating point
    #    - All DER Q = 0 (both TN and DN)
    #    - TSO OLTC = 0, shunts = 0
    #    - Generator AVR setpoints = v_setpoint_pu
    for s in net.sgen.index:
        if not str(net.sgen.at[s, "name"]).startswith("BOUND_"):
            net.sgen.at[s, "q_mvar"] = 0.0
    for t in tso_oltc:
        net.trafo.at[t, "tap_pos"] = 0
    for g in tso_gen_indices:
        net.gen.at[g, "vm_pu"] = v_setpoint_pu
    for sb in tso_shunt_buses:
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = 0
    # Tertiary shunts to 0 as well
    for sb in dso_shunt_buses_cand:
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = 0

    # 8) Coupling-transformer OLTCs: local AVR via pandapower DiscreteTapControl
    #    Regulate 110 kV (MV) side to dso_v_setpoint_pu.
    #    These controllers stay active throughout the simulation (run_control=True).
    tol_pu = config.dso_oltc_init_tol_pu
    for t3w in coupler_3w_indices:
        DiscreteTapControl(
            net,
            element_index=t3w,
            vm_lower_pu=dso_v_setpoint_pu - tol_pu,
            vm_upper_pu=dso_v_setpoint_pu + tol_pu,
            side="mv",
            element="trafo3w",
        )
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    if verbose > 1:
        for t3w in coupler_3w_indices:
            tap = int(net.trafo3w.at[t3w, "tap_pos"])
            print(f"  Coupler OLTC trafo3w {t3w}: initial tap_pos = {tap}")

    # NOTE: We do NOT remove the pandapower controllers here.
    # They act as local AVR for the coupling transformers throughout
    # the simulation, since there is no DSO OFO to control them.

    # Re-run PF with initial timestep and neutral actuators
    if use_profiles:
        apply_profiles(net, profiles, start_time)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    # Initialise TSO controller from converged PF
    tso.initialise(_measure_tso(net, tso_config, 0))

    if verbose > 0:
        print(
            f"Running TSO-only: {n_minutes} min  (TSO every {tso_period_min} min)"
        )
        print(
            f"  TSO: {len(tso_der_buses)} DER, {len(tso_oltc)} OLTC, "
            f"{len(tso_gen_indices)} gen, {len(tso_shunt_buses)} shunt"
        )
        print(
            f"  Coupler OLTCs: {len(coupler_3w_indices)} (local AVR, "
            f"target {dso_v_setpoint_pu:.3f} p.u. on 110 kV)"
        )
        print(
            f"  DSO DER: {len(dso_der_indices)} (Q=0, not controlled)"
        )
        print()

    # Reusable zero arrays for DSO DER Q (not controlled)
    _zero_dso_q = np.zeros(len(dso_der_indices), dtype=np.float64)

    # 9) Simulation loop
    log: List[IterationRecord] = []

    for minute in range(1, n_minutes + 1):
        run_tso = (minute == 1) or (minute % tso_period_min == 0)
        # Mark dso_active=True so LivePlotter records coupler OLTC / DN voltages
        rec = IterationRecord(minute=minute, tso_active=run_tso, dso_active=True)

        # Apply time-series profiles
        if use_profiles:
            t_now = start_time + timedelta(minutes=minute)
            apply_profiles(net, profiles, t_now)

        # Apply contingency events
        if contingencies:
            fired = [ev for ev in contingencies if ev.minute == minute]
            if fired:
                events = []
                for ev in fired:
                    desc, short_label = _apply_contingency(net, ev, verbose)
                    events.append((desc, short_label))
                rec.contingency_events = events

                # Re-converge PF with new topology
                pp.runpp(net, run_control=True, calculate_voltage_angles=True)

                # Refresh sensitivity matrices
                tso.sensitivities = JacobianSensitivities(net)
                tso.invalidate_sensitivity_cache()

        # TSO step
        if run_tso:
            try:
                tso_out = tso.step(_measure_tso(net, tso_config, minute))
                u = tso_out.u_new
                n_d = len(tso_config.der_indices)
                n_p = len(tso_config.pcc_trafo_indices)  # 0
                n_g = len(tso_config.gen_indices)
                n_o = len(tso_config.oltc_trafo_indices)
                n_s = len(tso_config.shunt_bus_indices)
                off = 0
                rec.tso_q_der_mvar = u[off : off + n_d].copy()
                off += n_d
                rec.tso_q_pcc_set_mvar = u[off : off + n_p].copy()
                off += n_p
                rec.tso_v_gen_pu = u[off : off + n_g].copy()
                off += n_g
                rec.tso_oltc_taps = np.round(u[off : off + n_o]).astype(np.int64)
                off += n_o
                rec.tso_shunt_states = np.round(u[off : off + n_s]).astype(np.int64)
                rec.tso_objective = tso_out.objective_value
                rec.tso_solver_status = tso_out.solver_status
                rec.tso_solve_time_s = tso_out.solve_time_s

                _apply_tso(net, tso_out, tso_config)
            except RuntimeError as e:
                if verbose > 0:
                    print(f"  [min {minute:3d}] TSO FAILED: {e}")

        # Power flow with run_control=True so coupling AVRs stay active
        pp.runpp(net, run_control=True, calculate_voltage_angles=True)

        # ── Record plant measurements (TN) ──
        rec.plant_tn_voltages_pu = net.res_bus.loc[tso_v_buses, "vm_pu"].values.copy()
        rec.plant_tn_currents_ka = np.array(
            [float(net.res_line.at[li, "i_from_ka"]) for li in tso_lines],
            dtype=np.float64,
        )
        rec.tso_q_gen_mvar = np.array(
            [float(net.res_gen.at[g, "q_mvar"]) for g in tso_gen_indices],
            dtype=np.float64,
        )

        # ── Record plant measurements (DN) — for LivePlotter DSO figure ──
        rec.plant_dn_voltages_pu = net.res_bus.loc[dso_v_buses, "vm_pu"].values.copy()
        rec.plant_dn_currents_ka = np.array(
            [float(net.res_line.at[li, "i_from_ka"]) for li in dso_lines],
            dtype=np.float64,
        )

        # DSO "control" record: DER Q = 0, coupler OLTC from AVR, shunt states
        rec.dso_q_der_mvar = _zero_dso_q.copy()
        rec.dso_oltc_taps = np.array(
            [int(net.trafo3w.at[t3w, "tap_pos"]) for t3w in coupler_3w_indices],
            dtype=np.int64,
        )
        rec.dso_shunt_states = np.array(
            [
                int(net.shunt.at[net.shunt.index[net.shunt["bus"] == sb][0], "step"])
                for sb in dso_shunt_buses_cand
            ],
            dtype=np.int64,
        ) if dso_shunt_buses_cand else np.array([], dtype=np.int64)

        # Actual interface Q (for plotting)
        rec.dso_q_actual_mvar = np.array(
            [float(net.res_trafo3w.at[t, "q_hv_mvar"]) for t in coupler_3w_indices],
            dtype=np.float64,
        )
        # No TSO setpoint in TSO-only → show zero line
        rec.dso_q_setpoint_mvar = np.zeros(len(coupler_3w_indices), dtype=np.float64)

        # Penalty from plant measurements
        if tso_config.v_setpoints_pu is not None:
            v_err = rec.plant_tn_voltages_pu - tso_config.v_setpoints_pu
            rec.tso_v_penalty = float(tso_config.g_v * np.sum(v_err**2))

        log.append(rec)

        if live_plotter is not None:
            live_plotter.update(rec)

        if verbose > 0 and run_tso:
            v_tn = rec.plant_tn_voltages_pu
            v_dn = rec.plant_dn_voltages_pu
            print(f"  [min {minute:3d}] TSO")
            print(
                f"    TN V: min={np.min(v_tn):.4f} mean={np.mean(v_tn):.4f} max={np.max(v_tn):.4f}"
            )
            print(
                f"    DN V: min={np.min(v_dn):.4f} mean={np.mean(v_dn):.4f} max={np.max(v_dn):.4f}"
            )
            if rec.tso_q_der_mvar is not None:
                print(
                    f"    TSO  obj={rec.tso_objective:.4e}  {rec.tso_solver_status}  "
                    f"t={rec.tso_solve_time_s:.3f}s"
                )
                print(f"      Q_DER   = {A2S(rec.tso_q_der_mvar)} Mvar")
                print(f"      V_gen   = {A3S(rec.tso_v_gen_pu)} pu")
                if len(rec.tso_oltc_taps) > 0:
                    print(f"      OLTC    = {rec.tso_oltc_taps}")
                if len(rec.tso_shunt_states) > 0:
                    print(f"      Shunt   = {rec.tso_shunt_states}")
                print(f"      Coupler OLTC = {rec.dso_oltc_taps.tolist()} (AVR)")
            print()

    if live_plotter is not None:
        live_plotter.finish()

    return log


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main() -> None:
    from core.cascade_config import CascadeConfig
    from run.records import ContingencyEvent

    start_min = 390
    duration = 120
    step = 1
    start_sp = 1.05
    end_sp = 1.07

    num_steps = duration // step
    delta = (end_sp - start_sp) / num_steps

    # Use exactly the same config as run_cascade main()
    config = CascadeConfig(
        # Simulation
        v_setpoint_pu=1.05,
        n_minutes=12 * 60,
        tso_period_min=3,
        dso_period_min=1,
        start_time=datetime(2016, 5, 1, 8, 0),
        use_profiles=True,
        verbose=1,
        live_plot=True,
        # Objective weights
        g_v=250000,
        g_q=1,
        dso_g_v=100000.0,
        # OFO
        alpha=1.0,
        g_z=1e12,
        # TSO g_w
        gw_tso_q_der=0.3,
        gw_tso_q_pcc=0.1,
        gw_tso_v_gen=2e6,
        gw_tso_oltc=30.0,
        gw_tso_shunt=5000.0,
        gw_oltc_cross_tso=0.0,
        # DSO g_w (unused but kept for config consistency)
        gw_dso_q_der=10.0,
        gw_dso_oltc=120.0,
        gw_dso_shunt=5000.0,
        gw_oltc_cross_dso=0.0,
        # DSO integral Q-tracking (unused)
        g_qi=0.3,
        lambda_qi=0.95,
        q_integral_max_mvar=50.0,
        # Generator capability
        gen_xd_pu=1.2,
        gen_i_f_max_pu=2.65,
        gen_beta=0.15,
        gen_q0_pu=0.4,
        # Reserve Observer (unused in TSO-only)
        enable_reserve_observer=True,
        reserve_q_threshold_mvar=45.0,
        reserve_q_release_mvar=-45.0,
        reserve_cooldown_min=15,
        # Contingencies
        contingencies = [
            ContingencyEvent(minute=90, element_type="line", element_index=3),
            ContingencyEvent(minute=60, element_type="gen", element_index=0),
            ContingencyEvent(
                minute=270, element_type="line", element_index=3, action="restore"
            ),
            ContingencyEvent(
                minute=300, element_type="gen", element_index=0, action="restore"
            ),
            ContingencyEvent(
                minute=360,
                element_type="ext_grid",
                element_index=0,
                action="setpoint_change",
                new_setpoint=1.06,
            ),
            # ramp of setpoints from 1.04 to 1.06
            # *[
            #     ContingencyEvent(
            #         minute=start_min + i * step,
            #         element_type="ext_grid",
            #         element_index=0,
            #         action="setpoint_change",
            #         new_setpoint=start_sp + (i + 1) * delta,
            #     )
            #     for i in range(num_steps)
            # ],
            ContingencyEvent(minute=420, element_type="line", element_index=11),
            ContingencyEvent(
                minute=480, element_type="line", element_index=11, action="restore"
            ),
        ],
    )

    print()
    print("#" * 72)
    print(f"#  TSO-ONLY SCENARIO: V_setpoint = {config.v_setpoint_pu:.2f} p.u.")
    print(f"#  DN DER Q = 0, Coupling OLTCs = local AVR")
    print("#" * 72)
    print()

    t0 = _time.perf_counter()
    log = run_tso_only(config)
    wall_time = _time.perf_counter() - t0

    # Summary
    final = log[-1]
    if final.plant_tn_voltages_pu is not None:
        v = final.plant_tn_voltages_pu
        v_set = config.v_setpoint_pu
        print()
        print("=" * 72)
        print(f"  TSO-ONLY FINAL SUMMARY  ({wall_time:.1f}s wall time)")
        print("=" * 72)
        print(
            f"  V_set = {v_set:.3f} p.u.,  "
            f"V_min = {np.min(v):.4f} p.u.,  "
            f"V_mean = {np.mean(v):.4f} p.u.,  "
            f"V_max = {np.max(v):.4f} p.u."
        )
        print(f"  Max |V - V_set| = {np.max(np.abs(v - v_set)):.4f} p.u.")
        n_tso = sum(1 for r in log if r.tso_active)
        print(f"  TSO steps: {n_tso}")
        print("=" * 72)
        print()


if __name__ == "__main__":
    main()
