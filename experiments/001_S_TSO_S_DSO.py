#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/run_S_TSO_M_DSO.py
==================
Cascaded TSO-DSO OFO Controller — main simulation loop.

TSO every 3 min, DSO every 1 min. Combined network for everything
(plant, sensitivities, state). TSO sends Q setpoints to DSO at PCCs.

TSO actuators: gen AVR voltages, machine trafo OLTCs (2W), TS-DER Q, 380 kV shunts
DSO actuators: DN-DER Q, 3W coupler OLTCs, tertiary winding shunts

Author: Manuel Schwenke / Claude Code

Note: All helper functions have been moved to sibling modules in the ``run/``
package.  This module contains only the main ``run_cascade`` and ``main``
functions.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore", category=UserWarning, module=r"mosek")

from controller.base_controller import OFOParameters
from controller.dso_controller import DSOController, DSOControllerConfig
from controller.reserve_observer import ReserveObserver
from controller.tso_controller import TSOController, TSOControllerConfig
from core.actuator_bounds import ActuatorBounds, GeneratorParameters
from core.measurement import measure_dso as _measure_dso
from core.measurement import measure_tso as _measure_tso
from core.message import SetpointMessage
from core.profiles import (
    DEFAULT_PROFILES_CSV,
    apply_profiles,
    load_profiles,
    snapshot_base_values,
)
from network.build_tuda_net import build_tuda_net
from experiments.helpers import (
    A2S,
    A3S,
    CascadeResult,
    ContingencyEvent,
    IterationRecord,
    _apply_contingency,
    _apply_dso,
    _apply_tso,
    _build_Gw,
    _network_state,
    _sgen_at_bus,
    prepare_load_contingencies,
    print_summary,
)
from sensitivity.jacobian import JacobianSensitivities


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
        See :class:`configs.cascade_config.CascadeConfig` for all fields.
    live_plotter : optional
        Object with an ``update(rec)`` method called after each iteration,
        e.g. :class:`visualisation.plot_cascade.LivePlotter`.  Takes
        precedence over ``config.live_plot``.
    """
    from configs.cascade_config import CascadeConfig

    # ── Unpack frequently-used config fields for readability ──────────────
    v_setpoint_pu = config.v_setpoint_pu
    dso_v_setpoint_pu = config.effective_dso_v_setpoint_pu
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

    # ── Timing: work in seconds internally ─────────────────────────────────
    tso_period_s = config.effective_tso_period_s
    dso_period_s = config.effective_dso_period_s
    dt_s = config.effective_sim_step_s          # simulation timestep [s]
    n_total_s = config.effective_n_seconds       # total duration [s]
    n_steps = int(n_total_s / dt_s)
    sub_minute = config.uses_sub_minute_timing

    def _fmt_period(s: float) -> str:
        """Format a period in seconds to a human-readable string."""
        if s >= 60.0 and s % 60 == 0:
            return f"{int(s // 60)} min"
        return f"{s:.1f} s"

    if verbose > 0:
        print("=" * 72)
        if dso_v_setpoint_pu != v_setpoint_pu:
            print(
                f"  CASCADED OFO  --  V_set TSO={v_setpoint_pu:.3f}  DSO={dso_v_setpoint_pu:.3f} p.u."
            )
        else:
            print(f"  CASCADED OFO  --  V_set = {v_setpoint_pu:.3f} p.u.")
        dur_str = _fmt_period(n_total_s) if sub_minute else f"{n_total_s // 60} min"
        print(
            f"  Profiles: {os.path.basename(profiles_csv)}  "
            f"start={start_time:%d.%m.%Y %H:%M}  {dur_str}"
        )
        print("=" * 72)
        if contingencies:
            print(f"  Scheduled contingencies ({len(contingencies)}):")
            for ev in contingencies:
                if ev.time_s is not None:
                    print(
                        f"    t={ev.time_s:.0f}s: {ev.action.upper()} "
                        f"{ev.element_type} {ev.element_index}"
                    )
                else:
                    print(
                        f"    min {ev.minute:4d}: {ev.action.upper()} "
                        f"{ev.element_type} {ev.element_index}"
                    )

    # contingencies already unpacked as list above

    # 1) Build combined network
    net, meta = build_tuda_net(ext_grid_vm_pu=v_setpoint_pu, pv_nodes=True)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Load time-series profiles and snapshot base load/gen values
    profiles = load_profiles(profiles_csv, timestep_s=dt_s)
    snapshot_base_values(net)

    # Pre-create dormant loads for load-contingency events
    if contingencies:
        prepare_load_contingencies(net, contingencies, verbose=verbose)

    # Apply profiles at t=0 and re-run PF so sensitivities + init use realistic operating point
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

    # -- DSO DER: individual sgen indices (not boundary)
    dso_der_indices = [
        int(s)
        for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ]
    dso_der_buses = [int(net.sgen.at[s, "bus"]) for s in dso_der_indices]

    # -- DSO monitored: 110 kV DN buses, DN lines
    dso_v_buses = sorted(
        int(b)
        for b in net.bus.index
        if 100.0 <= float(net.bus.at[b, "vn_kv"]) < 200.0 and int(b) in dn_buses
    )
    dso_lines = sorted(
        int(li) for li in net.line.index if str(net.line.at[li, "subnet"]) == "DN"
    )

    # -- DSO tertiary shunts
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

    pcc_trafos = list(meta.coupler_trafo3w_indices)

    # 3) Sensitivities from combined network
    if verbose > 1:
        print("Computing sensitivities from combined network ...")

    # TSO probe
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

    # DSO probe
    dso_sens = JacobianSensitivities(net)
    _, m_dso = dso_sens.build_sensitivity_matrix_H(
        der_bus_indices=dso_der_buses,
        observation_bus_indices=dso_v_buses,
        line_indices=dso_lines,
        trafo3w_indices=pcc_trafos,
        oltc_trafo3w_indices=list(meta.coupler_trafo3w_indices),
        shunt_bus_indices=dso_shunt_buses_cand,
        shunt_q_steps_mvar=dso_shunt_q_cand,
    )
    dso_oltc = list(m_dso.get("oltc_trafo3w", list(meta.coupler_trafo3w_indices)))
    dso_v_buses = list(m_dso.get("obs_buses", dso_v_buses))
    dso_shunt_buses = list(m_dso.get("shunt_buses", []))
    dso_shunt_q = [
        dso_shunt_q_cand[dso_shunt_buses_cand.index(b)] for b in dso_shunt_buses
    ]
    dso_iface_trafos = list(m_dso.get("trafo3w", pcc_trafos))

    # 4) Controller configs
    v_setpoints = np.full(len(tso_v_buses), v_setpoint_pu)
    dso_id = "dso_0"

    # Per-line thermal ratings [kA] for current constraints
    tso_line_max_i_ka = [float(net.line.at[li, "max_i_ka"]) for li in tso_lines]
    dso_line_max_i_ka = [float(net.line.at[li, "max_i_ka"]) for li in dso_lines]

    tso_config = TSOControllerConfig(
        der_indices=tso_der_indices,
        pcc_trafo_indices=pcc_trafos,
        pcc_dso_controller_ids=[dso_id] * len(pcc_trafos),
        oltc_trafo_indices=tso_oltc,
        shunt_bus_indices=tso_shunt_buses,
        shunt_q_steps_mvar=tso_shunt_q,
        voltage_bus_indices=tso_v_buses,
        current_line_indices=tso_lines,
        current_line_max_i_ka=tso_line_max_i_ka,
        v_setpoints_pu=v_setpoints,
        g_v=g_v,
        gen_indices=tso_gen_indices,
        gen_bus_indices=tso_gen_bus_indices,
        k_t_avt=config.k_t_avt,
    )
    dso_v_setpoints = np.full(len(dso_v_buses), dso_v_setpoint_pu)
    dso_config = DSOControllerConfig(
        der_indices=dso_der_indices,
        oltc_trafo_indices=dso_iface_trafos,
        shunt_bus_indices=dso_shunt_buses,
        shunt_q_steps_mvar=dso_shunt_q,
        interface_trafo_indices=dso_iface_trafos,
        voltage_bus_indices=dso_v_buses,
        current_line_indices=dso_lines,
        current_line_max_i_ka=dso_line_max_i_ka,
        g_q=g_q,
        g_qi=config.g_qi,
        lambda_qi=config.lambda_qi,
        q_integral_max_mvar=config.q_integral_max_mvar,
        v_setpoints_pu=dso_v_setpoints,
        g_v=config.dso_g_v,
    )

    # -- Live plotter (created here because configs are now available)
    if live_plot and live_plotter is None:
        from visualisation.plot_cascade import LivePlotter

        # Resolve DER display names from the network
        _tso_der_names = [
            str(net.sgen.at[s, "name"]) for s in tso_config.der_indices
        ]
        _dso_der_names = [
            str(net.sgen.at[s, "name"]) for s in dso_config.der_indices
        ]

        live_plotter = LivePlotter(
            tso_config,
            dso_config,
            tso_line_max_i_ka=np.array(tso_line_max_i_ka, dtype=np.float64)
            * tso_config.i_max_pu,
            dso_line_max_i_ka=np.array(dso_line_max_i_ka, dtype=np.float64)
            * dso_config.i_max_pu,
            sub_minute=sub_minute,
            tso_der_names=_tso_der_names,
            dso_der_names=_dso_der_names,
        )

    # 5) Actuator bounds (from combined network)
    def _der_bounds(indices):
        s_rated = np.array(
            [float(net.sgen.at[s, "sn_mva"]) for s in indices], dtype=np.float64
        )
        p_max = np.array(
            [float(net.sgen.at[s, "p_mw"]) for s in indices], dtype=np.float64
        )
        return s_rated, p_max

    tso_s, tso_p = _der_bounds(tso_der_indices)

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

    tso_der_op_diagrams = []
    for s in tso_der_indices:
        od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
        tso_der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")

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
        der_op_diagrams=tso_der_op_diagrams,
    )

    dso_s, dso_p = _der_bounds(dso_der_indices)
    dso_der_op_diagrams = []
    for s in dso_der_indices:
        od = net.sgen.at[s, "op_diagram"] if "op_diagram" in net.sgen.columns else None
        dso_der_op_diagrams.append(str(od) if od and str(od) != "nan" else "VDE-AR-N-4120-v2")

    dso_bounds = ActuatorBounds(
        der_indices=np.array(dso_der_indices, dtype=np.int64),
        der_s_rated_mva=dso_s,
        der_p_max_mw=dso_p,
        oltc_indices=np.array(dso_oltc, dtype=np.int64),
        oltc_tap_min=np.array(
            [int(net.trafo3w.at[t, "tap_min"]) for t in dso_oltc], dtype=np.int64
        ),
        oltc_tap_max=np.array(
            [int(net.trafo3w.at[t, "tap_max"]) for t in dso_oltc], dtype=np.int64
        ),
        shunt_indices=np.array(dso_shunt_buses, dtype=np.int64),
        shunt_q_mvar=np.array(dso_shunt_q, dtype=np.float64),
        der_op_diagrams=dso_der_op_diagrams,
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
        n_der=len(tso_der_buses),
        n_pcc=len(pcc_trafos),
        n_gen=len(tso_gen_indices),
        n_oltc=len(tso_oltc),
        n_shunt=len(tso_shunt_buses),
    )
    # DSO u = [Q_DER | s_OLTC | s_shunt]
    gw_dso_diag = config.build_gw_dso_diag(
        n_der=len(dso_der_buses),
        n_oltc=len(dso_oltc),
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
    n_pre_oltc_tso = len(tso_der_buses) + len(pcc_trafos) + len(tso_gen_indices)
    gw_tso = _build_Gw(gw_tso_diag, n_pre_oltc_tso, len(tso_oltc), gw_oltc_cross_tso)

    n_pre_oltc_dso = len(dso_der_buses)
    gw_dso = _build_Gw(gw_dso_diag, n_pre_oltc_dso, len(dso_oltc), gw_oltc_cross_dso)

    # Per-actuator g_u: penalises DER Q *level* (deviation from zero) to
    # incentivise freeing up DER headroom via shunt/OLTC switching.
    # With alpha=1 the g_u value adds directly to the G_w diagonal.
    gu_tso = config.build_gu_tso(
        n_der=len(tso_der_buses),
        n_pcc=len(pcc_trafos),
        n_gen=len(tso_gen_indices),
        n_oltc=len(tso_oltc),
        n_shunt=len(tso_shunt_buses),
    )
    gu_dso = config.build_gu_dso(
        n_der=len(dso_der_buses),
        n_oltc=len(dso_oltc),
        n_shunt=len(dso_shunt_buses),
    )
    gz_tso = config.build_gz_tso(n_v=len(tso_v_buses), n_i=len(tso_lines))
    gz_dso = config.build_gz_dso(
        n_iface=len(dso_iface_trafos), n_v=len(dso_v_buses), n_i=len(dso_lines)
    )
    ofo_tso = OFOParameters(alpha=config.alpha, g_w=gw_tso, g_z=gz_tso, g_u=gu_tso)
    ofo_dso = OFOParameters(alpha=config.alpha, g_w=gw_dso, g_z=gz_dso, g_u=gu_dso)

    ns = _network_state(net)

    tso = TSOController(
        "tso_main", ofo_tso, tso_config, ns, tso_bounds, JacobianSensitivities(net)
    )
    tso._avt_verbose = verbose
    dso = DSOController(
        dso_id, ofo_dso, dso_config, ns, dso_bounds, JacobianSensitivities(net)
    )

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
        net.gen.at[g, "vm_pu"] = v_setpoint_pu  # v_setpoint_pu
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
            net,
            element_index=t3w,
            vm_lower_pu=dso_v_setpoint_pu - tol_pu,
            vm_upper_pu=dso_v_setpoint_pu + tol_pu,
            side="mv",
            element="trafo3w",
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
        print(
            f"Running cascade: {_fmt_period(n_total_s)}  "
            f"(TSO/{_fmt_period(tso_period_s)}, DSO/{_fmt_period(dso_period_s)})  "
            f"[{n_steps} steps, dt={_fmt_period(dt_s)}]"
        )
        print(
            f"  TSO: {len(tso_der_buses)} DER, {len(tso_oltc)} OLTC, "
            f"{len(tso_gen_indices)} gen, {len(tso_shunt_buses)} shunt"
        )
        print(
            f"  DSO: {len(dso_der_buses)} DER, {len(dso_oltc)} OLTC, "
            f"{len(dso_shunt_buses)} shunt"
        )
        if gw_oltc_cross_tso != 0.0:
            n_o = len(tso_oltc)
            eff_diag = 50 + gw_oltc_cross_tso
            print(
                f"  TSO OLTC cross-coupling: g_cross={gw_oltc_cross_tso}, "
                f"eff. diag={eff_diag}, off-diag=+{gw_oltc_cross_tso}, "
                f"G_w shape={np.asarray(gw_tso).shape}"
            )
        if gw_oltc_cross_dso != 0.0:
            n_o = len(dso_oltc)
            eff_diag = 40 + gw_oltc_cross_dso
            print(
                f"  DSO OLTC cross-coupling: g_cross={gw_oltc_cross_dso}, "
                f"eff. diag={eff_diag}, off-diag=+{gw_oltc_cross_dso}, "
                f"G_w shape={np.asarray(gw_dso).shape}"
            )

        # --- Print initialised u_current for both controllers ---
        u_tso = tso.u_current
        n_d = len(tso_config.der_indices)
        n_p = len(tso_config.pcc_trafo_indices)
        n_g = len(tso_config.gen_indices)
        n_o = len(tso_config.oltc_trafo_indices)
        n_s = len(tso_config.shunt_bus_indices)
        off = 0
        print(f"  TSO u_init ({len(u_tso)} vars):")
        print(f"    Q_DER       = {A2S(u_tso[off : off + n_d])} Mvar")
        off += n_d
        print(f"    Q_PCC_set   = {A2S(u_tso[off : off + n_p])} Mvar")
        off += n_p
        print(f"    V_gen_set   = {A3S(u_tso[off : off + n_g])} pu")
        off += n_g
        print(f"    s_OLTC      = {u_tso[off : off + n_o].astype(int).tolist()}")
        off += n_o
        print(f"    s_shunt     = {u_tso[off : off + n_s].astype(int).tolist()}")

        u_dso = dso.u_current
        n_dd = len(dso_config.der_indices)
        n_do = len(dso_config.oltc_trafo_indices)
        n_ds = len(dso_config.shunt_bus_indices)
        print(f"  DSO u_init ({len(u_dso)} vars):")
        print(f"    Q_DER       = {A2S(u_dso[:n_dd])} Mvar")
        print(f"    s_OLTC      = {u_dso[n_dd : n_dd + n_do].astype(int).tolist()}")
        print(
            f"    s_shunt     = {u_dso[n_dd + n_do : n_dd + n_do + n_ds].astype(int).tolist()}"
        )
        print()

    # 8) Cascade loop
    log: List[IterationRecord] = []

    def _is_period_hit(time_s: float, period_s: float) -> bool:
        """Check if *time_s* is a multiple of *period_s* (with tolerance)."""
        remainder = time_s % period_s
        return remainder < 1e-9 or abs(remainder - period_s) < 1e-9

    for step in range(1, n_steps + 1):
        time_s = step * dt_s
        minute = int(time_s / 60)
        run_tso = (step == 1) or _is_period_hit(time_s, tso_period_s)
        run_dso = _is_period_hit(time_s, dso_period_s)
        rec = IterationRecord(
            minute=minute, time_s=time_s, tso_active=run_tso, dso_active=run_dso,
        )

        # Apply time-series profiles for this timestep
        if use_profiles:
            t_now = start_time + timedelta(seconds=time_s)
            apply_profiles(net, profiles, t_now)

        # ── Apply contingency events ──────────────────────────────────
        if contingencies:
            fired = [
                ev for ev in contingencies
                if abs(ev.effective_time_s - time_s) < 1e-9
            ]
            if fired:
                events = []
                for ev in fired:
                    desc, short_label = _apply_contingency(net, ev, verbose)
                    events.append((desc, short_label))
                rec.contingency_events = events

                # Re-converge PF with new topology so measurements
                # reflect the post-contingency operating point.
                # Controllers keep pre-outage sensitivities — OFO adapts
                # via measurement feedback (model-mismatch robustness).
                pp.runpp(net, run_control=False, calculate_voltage_angles=True)

        # TSO step
        if run_tso:
            try:
                tso_out = tso.step(_measure_tso(net, tso_config, step))
                u = tso_out.u_new
                n_d = len(tso_config.der_indices)
                n_p = len(tso_config.pcc_trafo_indices)
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

                # Send Q setpoints to DSO
                msgs = [
                    m
                    for m in tso.generate_setpoint_messages()
                    if m.target_controller_id == dso_id
                ]
                if msgs:
                    merged = SetpointMessage(
                        source_controller_id="tso_main",
                        target_controller_id=dso_id,
                        iteration=step,
                        interface_transformer_indices=np.concatenate(
                            [m.interface_transformer_indices for m in msgs]
                        ),
                        q_setpoints_mvar=np.concatenate(
                            [m.q_setpoints_mvar for m in msgs]
                        ),
                    )
                    dso.receive_setpoint(merged)
            except RuntimeError as e:
                if verbose > 0:
                    print(f"  [t={time_s:.0f}s] TSO FAILED: {e}")

        # DSO step
        if run_dso:
            try:
                dso_meas = _measure_dso(net, dso_config, step)

                # Reserve Observer: per-interface DER Q burden → shunt overrides
                if (
                    enable_reserve_observer
                    and dso._u_current is not None
                    and len(dso_config.shunt_bus_indices) > 0
                ):
                    n_dd = len(dso_config.der_indices)
                    n_do = len(dso_config.oltc_trafo_indices)
                    n_ds = len(dso_config.shunt_bus_indices)
                    n_iface = len(dso_config.interface_trafo_indices)
                    der_q = dso._u_current[:n_dd]
                    shunt_st = np.round(
                        dso._u_current[n_dd + n_do : n_dd + n_do + n_ds]
                    ).astype(np.int64)

                    # Extract dQ_interface/dQ_DER sub-matrix from DSO H cache
                    dQ_dQder = dso.get_interface_der_sensitivity()

                    # --- Debug: state BEFORE evaluate ---
                    if verbose > 2:
                        print(f"    [ReserveObs t={time_s:.0f}s] --- BEFORE evaluate ---")
                        print(f"      engaged_flags = {reserve_obs._engaged}")
                        print(f"      shunt_states  = {shunt_st.tolist()}")
                        print(f"      der_q (sum={np.sum(der_q):.2f} Mvar):")
                        # Show top-5 DERs by |Q|
                        top5 = np.argsort(np.abs(der_q))[-5:][::-1]
                        for k in top5:
                            print(
                                f"        DER[{k}] bus={dso_config.der_indices[k]}: "
                                f"Q={der_q[k]:.2f} Mvar"
                            )
                        if dQ_dQder is not None:
                            q_contribution = dQ_dQder @ der_q
                            print(f"      dQ_dQder shape = {dQ_dQder.shape}")
                            for j in range(n_iface):
                                q_step_j = reserve_obs.config.shunt_q_steps_mvar[j]
                                burden_j = (
                                    q_contribution[j]
                                    if q_step_j > 0
                                    else -q_contribution[j]
                                )
                                row_nz = np.count_nonzero(dQ_dQder[j, :])
                                row_sum = np.sum(dQ_dQder[j, :])
                                print(
                                    f"      iface[{j}] trafo3w={dso_config.interface_trafo_indices[j]}:"
                                )
                                print(
                                    f"        q_contribution = {q_contribution[j]:.2f} Mvar  "
                                    f"(dQ_dQder row: {row_nz} nonzero, sum={row_sum:.4f})"
                                )
                                print(
                                    f"        q_step = {q_step_j:.1f} Mvar  "
                                    f"-> burden = {burden_j:.2f} Mvar"
                                )
                                print(
                                    f"        thresholds: engage={reserve_obs.config.q_threshold_mvar:.1f}, "
                                    f"release={reserve_obs.config.q_release_mvar:.1f}"
                                )
                                engaged_before = reserve_obs._engaged[j]
                                last_act_s = reserve_obs._last_action_time_s[j]
                                cd_remaining_s = max(
                                    0.0,
                                    reserve_obs.config.effective_cooldown_s
                                    - (time_s - last_act_s),
                                )
                                in_cd = cd_remaining_s > 0
                                cd_str = (
                                    f"COOLDOWN {cd_remaining_s:.0f}s remaining"
                                    if in_cd
                                    else "no cooldown"
                                )
                                print(
                                    f"        cooldown: {cd_str}  "
                                    f"(last_action=t={last_act_s:.0f}s)"
                                )
                                if not engaged_before:
                                    would_engage = (
                                        burden_j > reserve_obs.config.q_threshold_mvar
                                        and not in_cd
                                    )
                                    reason = (
                                        "BLOCKED by cooldown"
                                        if in_cd
                                        and burden_j
                                        > reserve_obs.config.q_threshold_mvar
                                        else (
                                            "WILL ENGAGE"
                                            if would_engage
                                            else "stays disengaged"
                                        )
                                    )
                                    print(
                                        f"        state=DISENGAGED  "
                                        f"-> {reason} "
                                        f"(burden {'>' if burden_j > reserve_obs.config.q_threshold_mvar else '<='} threshold)"
                                    )
                                else:
                                    would_release = (
                                        burden_j < reserve_obs.config.q_release_mvar
                                        and not in_cd
                                    )
                                    reason = (
                                        "BLOCKED by cooldown"
                                        if in_cd
                                        and burden_j < reserve_obs.config.q_release_mvar
                                        else (
                                            "WILL RELEASE"
                                            if would_release
                                            else "stays engaged"
                                        )
                                    )
                                    print(
                                        f"        state=ENGAGED  "
                                        f"-> {reason} "
                                        f"(burden {'<' if burden_j < reserve_obs.config.q_release_mvar else '>='} release)"
                                    )
                        else:
                            print(f"      dQ_dQder = None (fallback: aggregate sum)")

                    obs_result = reserve_obs.evaluate(
                        der_q, shunt_st, dQ_dQder, time_s=time_s,
                    )

                    # --- Debug: result AFTER evaluate ---
                    if verbose > 2:
                        print(f"      --- AFTER evaluate ---")
                        print(f"      engaged_flags = {reserve_obs._engaged}")
                        print(f"      force_engage  = {obs_result.force_engage}")
                        print(f"      force_release = {obs_result.force_release}")

                    overrides = {}
                    for idx in obs_result.force_engage:
                        overrides[idx] = (1, 1)  # force shunt ON
                    for idx in obs_result.force_release:
                        overrides[idx] = (0, 0)  # force shunt OFF
                    if overrides:
                        if verbose > 2:
                            print(f"      -> APPLYING overrides: {overrides}")
                        dso.set_shunt_overrides(overrides)
                    elif verbose > 2:
                        print(f"      -> no overrides")

                dso_out = dso.step(dso_meas)
                u = dso_out.u_new
                n_dd = len(dso_config.der_indices)
                n_do = len(dso_config.oltc_trafo_indices)
                n_ds = len(dso_config.shunt_bus_indices)
                rec.dso_q_der_mvar = u[:n_dd].copy()
                rec.dso_oltc_taps = np.round(u[n_dd : n_dd + n_do]).astype(np.int64)
                rec.dso_shunt_states = np.round(
                    u[n_dd + n_do : n_dd + n_do + n_ds]
                ).astype(np.int64)
                rec.dso_q_setpoint_mvar = dso.q_setpoint_mvar.copy()
                rec.dso_objective = dso_out.objective_value
                rec.dso_solver_status = dso_out.solver_status
                rec.dso_solve_time_s = dso_out.solve_time_s

                _apply_dso(net, dso_out, dso_config)
                tso.receive_capability(
                    dso.generate_capability_message("tso_main", dso_meas)
                )
            except RuntimeError as e:
                if verbose > 0:
                    print(f"  [t={time_s:.0f}s] DSO FAILED: {e}")

        # Power flow
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)

        # Record plant voltages, line currents, generator Q, and actual interface Q after PF
        rec.plant_tn_voltages_pu = net.res_bus.loc[tso_v_buses, "vm_pu"].values.copy()
        rec.plant_dn_voltages_pu = net.res_bus.loc[dso_v_buses, "vm_pu"].values.copy()
        rec.plant_tn_currents_ka = np.array(
            [float(net.res_line.at[li, "i_from_ka"]) for li in tso_lines],
            dtype=np.float64,
        )
        rec.plant_dn_currents_ka = np.array(
            [float(net.res_line.at[li, "i_from_ka"]) for li in dso_lines],
            dtype=np.float64,
        )
        rec.tso_q_gen_mvar = np.array(
            [float(net.res_gen.at[g, "q_mvar"]) for g in tso_gen_indices],
            dtype=np.float64,
        )
        if run_dso:
            rec.dso_q_actual_mvar = np.array(
                [
                    float(net.res_trafo3w.at[t, "q_hv_mvar"])
                    for t in dso_config.interface_trafo_indices
                ],
                dtype=np.float64,
            )

        # Penalty terms from plant measurements
        if tso_config.v_setpoints_pu is not None:
            v_err = rec.plant_tn_voltages_pu - tso_config.v_setpoints_pu
            rec.tso_v_penalty = float(tso_config.g_v * np.sum(v_err**2))
        if rec.dso_q_actual_mvar is not None and dso.q_setpoint_mvar is not None:
            q_err = rec.dso_q_actual_mvar - dso.q_setpoint_mvar
            rec.dso_q_penalty = float(dso_config.g_q * np.sum(q_err**2))

        log.append(rec)

        if live_plotter is not None:
            live_plotter.update(rec)

        if verbose > 0 and (run_tso or run_dso):
            tags = "TSO+DSO" if (run_tso and run_dso) else ("TSO" if run_tso else "DSO")
            v_tn = rec.plant_tn_voltages_pu
            v_dn = rec.plant_dn_voltages_pu
            t_label = f"min {minute:3d}" if not sub_minute else f"t={time_s:.0f}s"
            print(f"  [{t_label}] {tags}")
            print(
                f"    TN V: min={np.min(v_tn):.4f} mean={np.mean(v_tn):.4f} max={np.max(v_tn):.4f}"
            )
            print(
                f"    DN V: min={np.min(v_dn):.4f} mean={np.mean(v_dn):.4f} max={np.max(v_dn):.4f}"
            )

            if run_tso and rec.tso_q_der_mvar is not None:
                print(
                    f"    TSO  obj={rec.tso_objective:.4e}  {rec.tso_solver_status}  "
                    f"t={rec.tso_solve_time_s:.3f}s"
                )
                print(f"      Q_DER   = {A2S(rec.tso_q_der_mvar)} Mvar")
                dso_der_p = np.array(
                    [
                        float(net.res_sgen.at[s, "p_mw"])
                        for s in net.sgen.index
                        if int(net.sgen.at[s, "bus"]) in dn_buses
                        and not str(net.sgen.at[s, "name"]).startswith("BOUND")
                    ],
                    dtype=np.float64,
                )
                DSO_bounds = dso_bounds.compute_der_q_bounds(dso_der_p)
                print(f"      Q_DER_min   = {A2S(DSO_bounds[0])} Mvar")
                print(f"      Q_DER_max   = {A2S(DSO_bounds[1])} Mvar")
                print(f"      Q_PCC_s = {A2S(rec.tso_q_pcc_set_mvar)} Mvar")
                print(f"      V_gen   = {A3S(rec.tso_v_gen_pu)} pu")
                if len(rec.tso_oltc_taps) > 0:
                    print(f"      OLTC    = {rec.tso_oltc_taps}")
                if len(rec.tso_shunt_states) > 0:
                    print(f"      Shunt   = {rec.tso_shunt_states}")

            if run_dso and rec.dso_q_der_mvar is not None:
                print(
                    f"    DSO  obj={rec.dso_objective:.4e}  {rec.dso_solver_status}  "
                    f"t={rec.dso_solve_time_s:.3f}s"
                )
                print(f"      Q_DER   = {A2S(rec.dso_q_der_mvar)} Mvar")
                dso_der_p = np.array(
                    [
                        float(net.res_sgen.at[s, "p_mw"])
                        for s in net.sgen.index
                        if int(net.sgen.at[s, "bus"]) in dn_buses
                        and not str(net.sgen.at[s, "name"]).startswith("BOUND")
                    ],
                    dtype=np.float64,
                )
                DSO_bounds = dso_bounds.compute_der_q_bounds(dso_der_p)
                print(f"      Q_DER_min   = {A2S(DSO_bounds[0])} Mvar")
                print(f"      Q_DER_max   = {A2S(DSO_bounds[1])} Mvar")
                if len(rec.dso_oltc_taps) > 0:
                    print(f"      OLTC    = {rec.dso_oltc_taps}")
                if len(rec.dso_shunt_states) > 0:
                    print(f"      Shunt   = {rec.dso_shunt_states}")
                print(
                    f"      Q_set   = {A2S(rec.dso_q_setpoint_mvar)} Mvar  (from TSO)"
                )
                print(f"      Q_act   = {A2S(rec.dso_q_actual_mvar)} Mvar  (plant)")
            print()

    if live_plotter is not None:
        live_plotter.finish()

    return CascadeResult(log=log, tso_config=tso_config, dso_config=dso_config)


# =============================================================================
#  Entry point
# =============================================================================


def main():
    import time as _time

    from configs.cascade_config import CascadeConfig
    from core.results_storage import save_results

    start_min = 600
    duration = 120
    step = 1
    start_sp = 1.05
    end_sp = 1.07

    num_steps = duration // step
    delta = (end_sp - start_sp) / num_steps

# ── Configuration (single place for ALL parameters) ───────────────────
    config = CascadeConfig(
        # Simulation
        v_setpoint_pu=1.05,
        n_minutes=6 * 60,
        tso_period_min=3,
        dso_period_s=30,
        start_time=datetime(2016, 1, 7, 6, 0),
        use_profiles=True,
        verbose=2,
        live_plot=True,
        # Objective weights
        g_v=250000,
        g_q=1,
        dso_g_v=100000,
        # OFO
        alpha=1.0,
        g_z=1e12,
        # TSO g_w
        gw_tso_q_der=0.2,
        gw_tso_q_pcc=0.1,
        gw_tso_v_gen=2e6,
        gw_tso_oltc=40.0,
        gw_tso_shunt=4000.0,
        gw_oltc_cross_tso=0,
        # DSO g_w
        gw_dso_q_der=10,
        gw_dso_oltc=140.0,
        gw_dso_shunt=5000.0,
        gw_oltc_cross_dso=0,
        # DSO integral Q-tracking
        g_qi=0.1,
        lambda_qi=0.9,
        q_integral_max_mvar=50.0,
        # Generator capability
        gen_xd_pu=1.2,
        gen_i_f_max_pu=2.65,
        gen_beta=0.15,
        gen_q0_pu=0.4,
        # Reserve Observer
        enable_reserve_observer=False,
        reserve_q_threshold_mvar=45.0,
        reserve_q_release_mvar=-45.0,
        reserve_cooldown_min=15,
        # Contingencies
        # contingencies = [
        #     ContingencyEvent(minute=90, element_type="line", element_index=3),
        #     ContingencyEvent(minute=60, element_type="gen", element_index=0),
        #     ContingencyEvent(
        #         minute=270, element_type="line", element_index=3, action="restore"
        #     ),
        #     ContingencyEvent(
        #         minute=300, element_type="gen", element_index=0, action="restore"
        #     ),
        #     ContingencyEvent(
        #         minute=360,
        #         element_type="ext_grid",
        #         element_index=0,
        #         action="setpoint_change",
        #         new_setpoint=1.05,
        #     ),
        #     # ramp of setpoints from 1.04 to 1.06
        #     # *[
        #     #     ContingencyEvent(
        #     #         minute=start_min + i * step,
        #     #         element_type="ext_grid",
        #     #         element_index=0,
        #     #         action="setpoint_change",
        #     #         new_setpoint=start_sp + (i + 1) * delta,
        #     #     )
        #     #     for i in range(num_steps)
        #     # ],
        #     ContingencyEvent(minute=420, element_type="line", element_index=11),
        #     ContingencyEvent(
        #         minute=480, element_type="line", element_index=11, action="restore"
        #     ),
        # ],
        contingencies=[],
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
