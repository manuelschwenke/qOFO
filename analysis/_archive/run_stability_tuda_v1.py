#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/run_stability_tuda.py
==============================
Integration script that builds the TUDa network, extracts the TSO and DSO
sensitivity matrices H, and runs the Lipschitz-based stability analysis from
:mod:`analysis.stability_analysis`.

Usage
-----
    python -m analysis.run_stability_tuda          # default CascadeConfig
    python -m analysis.run_stability_tuda --alpha 0.5 --g_v 80 --g_q 80

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore", category=UserWarning, module=r"mosek")

from analysis.stability_analysis import analyse_stability
from core.cascade_config import CascadeConfig
from core.profiles import DEFAULT_PROFILES_CSV, apply_profiles, load_profiles, snapshot_base_values
from network.build_tuda_net import build_tuda_net
from sensitivity.jacobian import JacobianSensitivities


def run_stability_analysis(
    config: CascadeConfig | None = None,
    *,
    verbose: bool = True,
):
    """Build the TUDa network, extract H matrices, and run stability analysis.

    Parameters
    ----------
    config :
        CascadeConfig with tuning parameters.  If *None*, a default is created.
    verbose :
        Print detailed report.

    Returns
    -------
    CascadeStabilityResult
        Stability result from :func:`analyse_stability`.
    """
    if config is None:
        config = CascadeConfig()

    v_setpoint_pu = config.v_setpoint_pu

    # ── 1) Build combined network ──────────────────────────────────────────
    if verbose:
        print("Building TUDa network …")
    net, meta = build_tuda_net(ext_grid_vm_pu=v_setpoint_pu, pv_nodes=True)
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    # Remove pandapower controllers (not needed for sensitivity probing)
    if hasattr(net, "controller") and len(net.controller) > 0:
        net.controller.drop(net.controller.index, inplace=True)

    # Optionally apply profiles at t=0 for a realistic operating point
    if config.use_profiles:
        profiles_csv = config.profiles_csv
        dt_s = config.effective_sim_step_s
        profiles = load_profiles(profiles_csv, timestep_s=dt_s)
        snapshot_base_values(net)
        apply_profiles(net, profiles, config.start_time)
        pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # ── 2) Identify elements (mirrors run_S_TSO_M_DSO.py logic) ────────────────
    dn_buses = {
        int(b) for b in net.bus.index if str(net.bus.at[b, "subnet"]) == "DN"
    }

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
    tso_gen_indices = []
    tso_gen_bus_indices = []
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

    # Machine transformer indices (one per generator): these control v_gen_pu.
    # They must be separated from ordinary network OLTCs for correct g_w assignment.
    machine_trafo_indices = list(meta.machine_trafo_indices)
    n_tso_gen_oltc = len(machine_trafo_indices)  # = len(tso_gen_indices)

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

    # ── 3) Build sensitivity matrices H ────────────────────────────────────
    if verbose:
        print("Computing TSO sensitivity matrix H_TSO …")

    tso_sens = JacobianSensitivities(net)
    H_tso, m_tso = tso_sens.build_sensitivity_matrix_H(
        der_bus_indices=tso_der_buses,
        observation_bus_indices=tso_v_buses,
        line_indices=tso_lines,
        oltc_trafo_indices=machine_trafo_indices,  # machine trafos only — no net OLTCs
        shunt_bus_indices=tso_shunt_buses_cand,
        shunt_q_steps_mvar=tso_shunt_q_cand,
    )
    tso_oltc_net = []
    tso_gen_oltc = list(m_tso.get("oltc_trafos", machine_trafo_indices))
    tso_oltc = list(m_tso.get("oltc_trafos", []))
    tso_shunt_buses = list(m_tso.get("shunt_buses", []))

    if verbose:
        print("Computing DSO sensitivity matrix H_DSO …")

    dso_sens = JacobianSensitivities(net)
    H_dso, m_dso = dso_sens.build_sensitivity_matrix_H(
        der_bus_indices=dso_der_buses,
        observation_bus_indices=dso_v_buses,
        line_indices=dso_lines,
        trafo3w_indices=pcc_trafos,
        oltc_trafo3w_indices=list(meta.coupler_trafo3w_indices),
        shunt_bus_indices=dso_shunt_buses_cand,
        shunt_q_steps_mvar=dso_shunt_q_cand,
    )
    dso_oltc = list(m_dso.get("oltc_trafo3w", list(meta.coupler_trafo3w_indices)))
    dso_shunt_buses = list(m_dso.get("shunt_buses", []))

    # ── 4) Determine actuator and output counts ───────────────────────────
    # Actuator columns: [Q_DER | OLTC taps | Shunt states]
    # PCC Q setpoints and Gen V setpoints are separate control channels
    # NOT represented as columns in the sensitivity matrix H.
    n_tso_der = len(tso_der_indices)
    n_tso_gen = len(tso_gen_oltc)
    n_tso_oltc = len(tso_oltc_net)
    n_tso_shunt = len(tso_shunt_buses)

    n_dso_der = len(dso_der_indices)
    n_dso_oltc = len(dso_oltc)
    n_dso_shunt = len(dso_shunt_buses)

    # Output row counts (from H mappings).
    # H rows: [Q_trafo2w | Q_trafo3w | V_bus | I_line]
    # TSO has no trafo Q outputs → rows are [V_bus | I_line]
    _n_tso_v = len(m_tso.get("obs_buses", []))
    _n_tso_i = len(m_tso.get("lines", []))
    # DSO has trafo3w Q outputs (PCC interface) → rows are [Q_PCC | V_bus | I_line]
    _n_dso_q = len(m_dso.get("trafos", [])) + len(m_dso.get("trafo3w", []))
    _n_dso_v = len(m_dso.get("obs_buses", []))
    _n_dso_i = len(m_dso.get("lines", []))

    if verbose:
        print()
        print(f"  TSO H shape: {H_tso.shape} "
              f"cols: DER={n_tso_der}, V_gen={n_tso_gen}, "
              f"OLTC={n_tso_oltc}, Shunt={n_tso_shunt}")
        print(f"  TSO H rows : V={_n_tso_v}, I={_n_tso_i}  "
              f"(Q_obj: g_v={config.g_v})")
        print(f"  DSO H shape: {H_dso.shape}  "
              f"cols: DER={n_dso_der}, OLTC={n_dso_oltc}, Shunt={n_dso_shunt}")
        print(f"  DSO H rows : Q={_n_dso_q}, V={_n_dso_v}, I={_n_dso_i}  "
              f"(Q_obj: g_q={config.g_q}, dso_g_v={config.dso_g_v})")
        print(f"  (PCC={len(pcc_trafos)}, Gen={len(tso_gen_indices)} "
              f"— not in H, separate control channels)")
        print()

        # Print DER names for reference
        tso_der_names = [str(net.sgen.at[s, "name"]) for s in tso_der_indices]
        dso_der_names = [str(net.sgen.at[s, "name"]) for s in dso_der_indices]
        print(f"  TSO DER: {tso_der_names}")
        print(f"  DSO DER: {dso_der_names}")
        print()

    # ── 5) Run stability analysis ──────────────────────────────────────────
    result = analyse_stability(
        config=config,
        H_tso=H_tso,
        H_dso=H_dso,
        # Actuator column counts
        n_tso_der=n_tso_der,
        n_tso_pcc=0,        # PCC Q not in H
        n_tso_gen=n_tso_gen,
        n_tso_oltc=n_tso_oltc,
        n_tso_shunt=n_tso_shunt,
        n_dso_der=n_dso_der,
        n_dso_oltc=n_dso_oltc,
        n_dso_shunt=n_dso_shunt,
        # Output row counts
        n_tso_v_out=_n_tso_v,
        n_tso_i_out=_n_tso_i,
        n_dso_q_out=_n_dso_q,
        n_dso_v_out=_n_dso_v,
        n_dso_i_out=_n_dso_i,
        verbose=verbose,
    )

    # ── 6) Per-type g_w recommendations ───────────────────────────────────────────
    if verbose:
        from analysis.stability_analysis import _build_q_obj, recommend_gw_min_per_type

        q_tso = _build_q_obj(0, _n_tso_v, _n_tso_i, 0.0, config.g_v)
        q_dso = _build_q_obj(_n_dso_q, _n_dso_v, _n_dso_i, config.g_q, config.dso_g_v)

        # Column order must match H exactly: [Q_DER | V_gen | Shunt] for TSO,
        # [Q_DER | OLTC | Shunt] for DSO.
        tso_blocks = [
            ('Q_DER_TS', n_tso_der, config.gw_tso_q_der),
            ('V_gen', n_tso_gen, config.gw_tso_v_gen),
            ('OLTC', n_tso_oltc, config.gw_tso_oltc),
            ('Shunt_TS', n_tso_shunt, config.gw_tso_shunt),
        ]
        dso_blocks = [
            ('Q_DER_DN', n_dso_der, config.gw_dso_q_der),
            ('OLTC_DN', n_dso_oltc, config.gw_dso_oltc),
            ('Shunt_DN', n_dso_shunt, config.gw_dso_shunt),
        ]

        rec_tso = recommend_gw_min_per_type(config, H_tso, q_tso, tso_blocks)
        rec_dso = recommend_gw_min_per_type(config, H_dso, q_dso, dso_blocks)

        print()
        print('  Recommended minimum g_w per actuator type (safety factor 2×):')
        print(f'  {"Type":<12s} {"g_w (actual)":>14s} {"g_w_min":>10s} '
              f'{"margin":>10s} {"rel. margin":>12s}  status')
        print('  ' + '-' * 64)
        for layer_name, recs in (('TSO', rec_tso), ('DSO', rec_dso)):
            for r in recs:
                flag = '✓' if r['ok'] else '✗'
                print(
                    f'  {flag} {layer_name} {r["type"]:<10s} '
                    f'{r["gw_actual"]:>14.4g} '
                    f'{r["gw_min"]:>10.4g} '
                    f'{r["margin"]:>+10.4g} '
                    f'{100 * r["margin_rel"]:>10.1f} %'
                )
        print()

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _tuda_config() -> CascadeConfig:
    """Return the CascadeConfig matching the actual TUDa run in run_S_TSO_M_DSO.py."""
    return CascadeConfig(
        # Simulation
        v_setpoint_pu=1.05,
        tso_period_min=3,
        dso_period_s=30,
        use_profiles=True,

        # Objective weights
        g_v=200000,
        g_q=0.8,
        dso_g_v=100000,

        # OFO algorithm
        alpha=1,
        g_z=1e12,

        # TSO g_w
        gw_tso_q_der=0.1,
        gw_tso_q_pcc=0.1,
        gw_tso_v_gen=2e6,
        gw_tso_oltc=40.0,
        gw_tso_shunt=4000.0,

        # DSO g_w
        gw_dso_q_der=10,
        gw_dso_oltc=150.0,
        gw_dso_shunt=5000.0,

        # DSO integral Q-tracking
        g_qi=0.2,
        lambda_qi=0.95,
        q_integral_max_mvar=50.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Lipschitz stability analysis on the TUDa network."
    )
    parser.add_argument("--alpha", type=float, default=None,
                        help="OFO step-size α  (default: from TUDa config)")
    parser.add_argument("--g_v", type=float, default=None,
                        help="TSO voltage objective weight")
    parser.add_argument("--g_q", type=float, default=None,
                        help="DSO reactive-power objective weight")
    parser.add_argument("--dso_g_v", type=float, default=None,
                        help="DSO voltage objective weight")
    parser.add_argument("--gw_tso", type=float, default=None,
                        help="TSO g_w_q_der (per-DER regularisation)")
    parser.add_argument("--gw_dso", type=float, default=None,
                        help="DSO g_w_q_der (per-DER regularisation)")
    parser.add_argument("--no-profiles", action="store_true",
                        help="Skip applying load/gen profiles (use flat operating point)")

    args = parser.parse_args()

    config = _tuda_config()
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.g_v is not None:
        config.g_v = args.g_v
    if args.g_q is not None:
        config.g_q = args.g_q
    if args.dso_g_v is not None:
        config.dso_g_v = args.dso_g_v
    if args.gw_tso is not None:
        config.gw_tso_q_der = args.gw_tso
    if args.gw_dso is not None:
        config.gw_dso_q_der = args.gw_dso
    if args.no_profiles:
        config.use_profiles = False

    run_stability_analysis(config, verbose=True)


if __name__ == "__main__":
    main()
