"""
Scenario: wind_replace
======================
Replace selected synchronous generators with STATCOM-capable wind parks.

Zone 2:  ALL generators removed; wind parks at HALF the original P_mw.
Zone 1:  G1 (term 29) + G8 (term 36) removed; wind at SAME P_mw.
         Retains G9 (term 37, ~830 MW) as synchronous anchor.
Zone 3:  G4 (term 32) + G5 (term 33) removed; wind at SAME P_mw.
         Retains G6 (term 34) + G7 (term 35) as synchronous anchors.

Each replacement wind park sgen has:
  - S_n = 1.2 * P_wp  (20 % oversized converter for Q headroom)
  - op_diagram = 'STATCOM'  (full-circle capability, no dead zone at P=0)
  - profile = 'WP10'

A temporary PV-generator trick initialises the STATCOM Q injection so
that the power-flow solution starts from a physically reasonable
operating point.

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandapower as pp

from network.ieee39.helpers import remove_generators
from network.ieee39.meta import IEEE39NetworkMeta


def apply_wind_replace(net, meta, *, ext_grid_vm_pu=1.03, **kwargs):
    """Apply the *wind_replace* scenario.

    Parameters
    ----------
    net : pp.pandapowerNet
        IEEE 39-bus network (modified in-place).
    meta : IEEE39NetworkMeta
        Current metadata catalogue.
    ext_grid_vm_pu : float, optional
        Voltage setpoint [pu] used for the temporary PV-generator
        initialisation trick (default 1.03).

    Returns
    -------
    (net, meta)
        The modified network and updated metadata.
    """
    # Zone bus sets (0-indexed, from _FIXED_ZONES_IEEE39):
    _zone1_buses = {0, 1, 24, 25, 26, 27, 28, 29, 36, 37, 38}
    _zone2_buses = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31}
    _zone3_buses = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35}

    # Terminal buses of the specific generators to remove in zones 1 and 3
    _z1_gens_to_remove_term = {29, 36}   # G1 + G8
    _z3_gens_to_remove_term = {32, 33}   # G4 + G5

    # Classify each generator and decide whether to remove it
    _gens_to_remove: List[int] = []
    # (gen_idx, grid_bus, p_mw, zone) -- zone needed for scaling
    _removed_gen_info: List[Tuple[int, int, float, int]] = []
    for g, gb in zip(meta.gen_indices, meta.gen_grid_bus_indices):
        term_bus = int(net.gen.at[g, "bus"])
        p_mw = float(net.gen.at[g, "p_mw"])

        if gb in _zone2_buses:
            # Zone 2: remove ALL generators
            _gens_to_remove.append(g)
            _removed_gen_info.append((g, gb, p_mw, 2))
        elif gb in _zone1_buses and term_bus in _z1_gens_to_remove_term:
            _gens_to_remove.append(g)
            _removed_gen_info.append((g, gb, p_mw, 1))
        elif gb in _zone3_buses and term_bus in _z3_gens_to_remove_term:
            _gens_to_remove.append(g)
            _removed_gen_info.append((g, gb, p_mw, 3))

    meta = remove_generators(net, meta, _gens_to_remove)

    # Create replacement wind park sgens at the same grid buses.
    # Zone 2: P_wp = P_gen / 2   (half the original capacity)
    # Zone 1, 3: P_wp = P_gen    (same installed power)
    # S_n = 1.2 * P_rated (20% oversized converter for Q headroom).
    #   Q_available = sqrt(S_n^2 - P^2) = sqrt(0.44) * P ~ 0.66 * P
    # Operating diagram: STATCOM (full circle, no dead zone at P=0).
    _wp_sgen_indices: List[int] = []
    _wp_sgen_buses: List[int] = []
    _wp_info: List[Tuple[int, int, float, float]] = []  # (sgen_idx, bus, p, sn)
    for _g_idx, gb, gen_p, zone in _removed_gen_info:
        wp_p = gen_p / 2.0 if zone == 2 else gen_p # ToDo: We can set STATCOM wind park capacities here
        wp_sn = wp_p * 1.2
        idx = pp.create_sgen(
            net,
            bus=gb,
            p_mw=wp_p,
            q_mvar=0.0,
            sn_mva=wp_sn,
            name=f"WP_STATCOM|grid_bus{gb}",
            subnet='TN',
            op_diagram='STATCOM',
        )
        net.sgen.at[idx, "profile"] = "WP10"
        _wp_sgen_indices.append(int(idx))
        _wp_sgen_buses.append(gb)
        _wp_info.append((int(idx), gb, wp_p, wp_sn))

    # ── Initialize STATCOM Q via temporary PV generators ─────────────────
    # Place temporary gens at each STATCOM bus with V_target so the
    # PF solver finds the Q needed to maintain voltage.  Then transfer
    # that Q to the sgen and remove the temporary gen.
    _temp_gen_map: Dict[int, int] = {}  # temp_gen_idx -> sgen_idx
    for sidx, bus, wp_p, wp_sn in _wp_info:
        # Disable the sgen during initialization
        net.sgen.at[sidx, "in_service"] = False
        gidx = pp.create_gen(
            net, bus=bus, p_mw=wp_p, vm_pu=ext_grid_vm_pu,
            sn_mva=wp_sn,
            max_q_mvar=wp_sn, min_q_mvar=-wp_sn,
            in_service=True,
            name=f"_TEMP_INIT|bus{bus}",
        )
        _temp_gen_map[int(gidx)] = sidx

    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             max_iteration=50)

    # Transfer Q from temp gens to STATCOMs, then clean up
    for gidx, sidx in _temp_gen_map.items():
        q_init = float(net.res_gen.at[gidx, "q_mvar"])
        net.sgen.at[sidx, "q_mvar"] = q_init
        net.sgen.at[sidx, "in_service"] = True
    net.gen.drop(index=list(_temp_gen_map.keys()), inplace=True)

    # Re-run PF with sgens active to verify convergence
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # Update meta to include the new wind park sgens as TSO DERs
    meta = IEEE39NetworkMeta(
        tn_bus_indices        = meta.tn_bus_indices,
        tn_line_indices       = meta.tn_line_indices,
        gen_indices           = meta.gen_indices,
        gen_bus_indices        = meta.gen_bus_indices,
        gen_grid_bus_indices  = meta.gen_grid_bus_indices,
        machine_trafo_indices = meta.machine_trafo_indices,
        machine_trafo_gen_map = meta.machine_trafo_gen_map,
        tso_der_indices       = tuple(list(meta.tso_der_indices) + _wp_sgen_indices),
        tso_der_buses         = tuple(list(meta.tso_der_buses) + _wp_sgen_buses),
        dso_pcc_trafo_indices = meta.dso_pcc_trafo_indices,
        dso_pcc_hv_buses      = meta.dso_pcc_hv_buses,
        dso_lv_buses          = meta.dso_lv_buses,
        dso_der_indices       = meta.dso_der_indices,
        dso_der_buses         = meta.dso_der_buses,
        dso_shunt_indices     = meta.dso_shunt_indices,
        dso_shunt_buses       = meta.dso_shunt_buses,
        dn_bus_indices        = meta.dn_bus_indices,
        dn_line_indices       = meta.dn_line_indices,
        hv_networks           = meta.hv_networks,
    )
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    return net, meta
