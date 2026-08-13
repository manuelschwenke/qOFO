"""
Scenario: wind_replace
======================
Replace selected synchronous generators with STATCOM-capable wind parks.

Generator labels follow the standard IEEE-39 textbook convention
(G1 = aggregated equivalent at IEEE bus 39; G10 = Hydro at IEEE bus 30).
Terminal-bus indices are 0-indexed pandapower buses (= IEEE 1-idx − 1).

Zone 1:  G8 (term 36) removed.
         Retains G10 (term 29, Hydro, 1 GVA), G9 (term 37, ~830 MW), and
         G1 (term 38, slack anchor, 10 GVA equivalent) as synchronous
         anchors.
Zone 2:  G2 (term 30, Nuclear, ex-slack) removed.
         Retains G3 (pandapower gen_idx 1, term 31, grid 9, 650 MW) as
         the Zone-2 synchronous anchor.
Zone 3:  G5 (term 33) + G6 (term 34) removed.
         Retains G4 (term 32, grid 18) + G7 (term 35, grid 35) as
         synchronous anchors.  This split keeps the two STATCOM wind
         parks at *different* grid buses (18 and 21) instead of
         collapsing both onto grid bus 18 via the two-trafo chain that
         makes G4 and G5 share a grid bus after build_ieee39_net().

Each replacement wind park sgen is sized as:
  - P_n = WIND_REPLACE_SCALE * P_removed_gen  (default 1.0 → full
    removed-gen P; tune via the module-level ``WIND_REPLACE_SCALE``)
  - S_n = P_n  (no converter oversize)
  - op_diagram = 'VDE-AR-N-4120-v2'  (grid-code box −0.33/+0.41·S_n above
    P/S_n = 0.2).  This was 'STATCOM' until 2026-07-21; with S_n = P_n the
    circular diagram collapses to Q = 0 at rated P.
  - profile = 'WP10'

A temporary PV-generator trick seeds the STATCOM Q injection at the
base (pre-profile) operating point so that the downstream base PF
(``add_hv_networks`` verification run, etc.) converges from a physically
plausible state.  This is a **seed only** — the authoritative Q + OLTC
init happens once more at the profile-scaled operating point in the
caller (see ``run_multi_tso_dso`` step 7c in
``experiments/000_M_TSO_M_DSO.py``).

Author: Manuel Schwenke / Claude Code
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandapower as pp

from network.ieee39.aux_load_buses import (
    AUX_LOAD_LINK_LENGTH_KM,
    AUX_LOAD_LINK_MAX_I_KA,
    AUX_LOAD_LINK_R_OHM,
    AUX_LOAD_LINK_X_OHM,
    separate_colocated_zip_loads,
)
from network.ieee39.helpers import remove_generators
from network.ieee39.meta import IEEE39NetworkMeta


# ── Tunable wind-park sizing ─────────────────────────────────────────────
# Multiplicative scale on the removed generator's base P_mw.  Default 1.0
# means "wind park rated at the FULL removed-gen P" (per the docstring
# convention).  Increase to oversize wind parks for additional zonal
# headroom; decrease to study under-replacement scenarios.
WIND_REPLACE_SCALE: float = 1
def _separate_colocated_wind_and_tn_loads(
    net: pp.pandapowerNet,
    replacement_sgen_indices: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    """Move TN ZIP loads off buses carrying fixed-power wind injections."""
    tn_load_indices = [
        int(i) for i in net.load.index
        if str(net.load.at[i, "subnet"]) == "TN"
    ]
    return separate_colocated_zip_loads(
        net,
        load_indices=tn_load_indices,
        injection_sgen_indices=replacement_sgen_indices,
        aux_subnet="TN_AUX",
    )

def apply_wind_replace(net, meta, *, ext_grid_vm_pu=1.03, **kwargs):
    """Apply the *wind_replace* scenario.

    Parameters
    ----------
    net : pp.pandapowerNet
        IEEE 39-bus network (modified in-place).
    meta : IEEE39NetworkMeta
        Current metadata catalogue.
    ext_grid_vm_pu : float, optional
        Voltage setpoint [pu] for the temporary PV-generator STATCOM
        **seed** (default 1.03).  Not the final operating point; the
        caller re-initialises Q + OLTCs after profiles are applied.

    Returns
    -------
    (net, meta)
        The modified network and updated metadata.
    """
    # Zone bus sets (0-indexed, from _FIXED_ZONES_IEEE39):
    _zone1_buses = {0, 1, 24, 25, 26, 27, 28, 29, 36, 37, 38}
    _zone2_buses = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31}
    _zone3_buses = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35}

    # Terminal buses of the specific generators to remove per zone.
    # Zone 1 keeps G10 (term 29, Hydro), G9 (term 37) and G1 (term 38,
    # slack anchor) as synchronous anchors — only G8 is replaced by a
    # STATCOM wind park.  Zone 2 keeps G3 (pandapower gen_idx 1, term 31,
    # grid 9, 650 MW) as the synchronous anchor; G2 (ex-slack) is replaced.
    _z1_gens_to_remove_term = {36}       # G8 only — G10 (term 29, Hydro) retained (decision 2026-07-17)
    _z2_gens_to_remove_term = {30}       # G2 (ex-slack); G3 (term 31) retained
    _z3_gens_to_remove_term = {33, 34}   # G5 (grid 18, shares with G4) + G6 (grid 21)

    # Classify each generator and decide whether to remove it
    _gens_to_remove: List[int] = []
    # (gen_idx, grid_bus, p_mw, zone) -- zone needed for scaling
    _removed_gen_info: List[Tuple[int, int, float, int]] = []
    for g, gb in zip(meta.gen_indices, meta.gen_grid_bus_indices):
        term_bus = int(net.gen.at[g, "bus"])
        p_mw = float(net.gen.at[g, "p_mw"])

        if gb in _zone2_buses and term_bus in _z2_gens_to_remove_term:
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
    # All zones: P_wp = WIND_REPLACE_SCALE * P_removed_gen   (default 1.0 →
    # full removed-gen P, restoring the docstring convention; tune via the
    # module-level ``WIND_REPLACE_SCALE``).  S_n = P_wp (no converter
    # oversize).
    #
    # op_diagram: VDE-AR-N-4120-v2, NOT 'STATCOM' (changed 2026-07-21).
    # With S_n = P_n the circular STATCOM diagram gives
    # Q = ±sqrt(S_n² − P²) = 0 at rated P, i.e. the park supplies no
    # reactive power at all whenever P is not profile-scaled below rated
    # (e.g. a Gate-E replay with ``use_profiles=False``).  The grid-code
    # box −0.33/+0.41·S_n is P-independent above P/S_n = 0.2 and therefore
    # well-defined at rated P.  See
    # docs/daily_log/07_2026/2026-07-21_rms_gate_e_qv_rollout_and_capability_finding.md
    _wp_sgen_indices: List[int] = []
    _wp_sgen_buses: List[int] = []
    _wp_info: List[Tuple[int, int, float, float]] = []  # (sgen_idx, bus, p, sn)
    for _g_idx, gb, gen_p, zone in _removed_gen_info:
        wp_p = gen_p * WIND_REPLACE_SCALE
        wp_sn = wp_p * 1.0
        idx = pp.create_sgen(
            net,
            bus=gb,
            p_mw=wp_p,
            q_mvar=0.0,
            sn_mva=wp_sn,
            name=f"WP_STATCOM|grid_bus{gb}",
            subnet='TN',
            op_diagram='VDE-AR-N-4120-v2',
        )
        net.sgen.at[idx, "profile"] = "WP10"
        _wp_sgen_indices.append(int(idx))
        _wp_sgen_buses.append(gb)
        _wp_info.append((int(idx), gb, wp_p, wp_sn))

    _aux_bus_indices, _aux_parent_buses, _aux_line_indices = (
        _separate_colocated_wind_and_tn_loads(net, _wp_sgen_indices)
    )

    # ── Seed STATCOM Q via temporary PV generators ──────────────────────
    # This is a *seed* so downstream base PFs converge; the caller
    # re-initialises Q + OLTCs once more at the profile-scaled operating
    # point (see run_multi_tso_dso step 7c).
    _temp_gen_map: Dict[int, int] = {}
    for sidx, bus, wp_p, wp_sn in _wp_info:
        net.sgen.at[sidx, "in_service"] = False
        gidx = pp.create_gen(
            net, bus=bus, p_mw=wp_p, vm_pu=ext_grid_vm_pu,
            sn_mva=wp_sn,
            max_q_mvar=wp_sn, min_q_mvar=-wp_sn,
            in_service=True,
            name=f"_TEMP_SEED|bus{bus}",
        )
        _temp_gen_map[int(gidx)] = sidx

    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             max_iteration=50)

    for gidx, sidx in _temp_gen_map.items():
        net.sgen.at[sidx, "q_mvar"] = float(net.res_gen.at[gidx, "q_mvar"])
        net.sgen.at[sidx, "in_service"] = True
    net.gen.drop(index=list(_temp_gen_map.keys()), inplace=True)

    # Update meta to include the new wind park sgens as TSO DERs
    meta = IEEE39NetworkMeta(
        tn_bus_indices        = meta.tn_bus_indices,
        tn_line_indices       = meta.tn_line_indices,
        gen_indices           = meta.gen_indices,
        gen_bus_indices        = meta.gen_bus_indices,
        gen_grid_bus_indices  = meta.gen_grid_bus_indices,
        machine_trafo_indices = meta.machine_trafo_indices,
        machine_trafo_gen_map = meta.machine_trafo_gen_map,
        internal_aux_bus_indices = tuple(
            list(meta.internal_aux_bus_indices) + _aux_bus_indices
        ),
        internal_aux_parent_buses = tuple(
            list(meta.internal_aux_parent_buses) + _aux_parent_buses
        ),
        internal_aux_line_indices = tuple(
            list(meta.internal_aux_line_indices) + _aux_line_indices
        ),
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
