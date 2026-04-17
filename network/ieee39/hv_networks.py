"""
network/ieee39/hv_networks.py
=============================
HV (110 kV) sub-network attachment for the IEEE 39-bus New England test case.

This module creates copies of the TUDA 110 kV meshed topology, couples them
to the 345 kV transmission network via 3-winding transformers, and handles
load redistribution, EHV profile wiring, and Q-load compensation.

Public entry point
------------------
``add_hv_networks(net, meta, ...)`` -- attach HV sub-networks and return
updated :class:`~network.ieee39.meta.IEEE39NetworkMeta`.

Internal helpers
----------------
``_create_hv_subnetwork`` -- build one 10-bus HV copy with lines, loads, DER.
``_wire_ehv_profiles``    -- assign HS4/HS5 simbench profiles to TN loads.
``_compute_reference_loads`` -- pool and cap coupling-bus loads per sub-net.
``_print_hv_summary``     -- formatted debug table of connections/DER.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandapower as pp

from network.ieee39.meta import IEEE39NetworkMeta, HVNetworkInfo
from network.ieee39.constants import (
    HV_LINE_TOPOLOGY,
    HV_COUPLING_WP_MVA,
    HV_Q_LOAD_FACTOR,
    TUDA_WIND_PARKS,
    TUDA_PV_PLANTS,
    ZONE3_BUSES_0IDX,
    SUBNET_DEFS,
)
from network.ieee39.helpers import get_load_at_bus, reduce_loads_at_bus


# =====================================================================
#  Internal: single HV sub-network builder
# =====================================================================

def _create_hv_subnetwork(
    net: pp.pandapowerNet,
    net_id: str,
    coupling_map: List[Tuple[int, int]],
    *,
    line_length_scale: float = 1.0,
    total_p_mw: float = 500.0,
    total_q_mvar: float = 50.0,
    gen_type: str = "mixed",
) -> HVNetworkInfo:
    """
    Create one copy of the TUDA 110 kV HV network and couple it to the
    IEEE 39-bus 345 kV network via 2-winding transformers.

    Parameters
    ----------
    net : pandapowerNet
        The IEEE 39-bus network (modified in-place).
    net_id : str
        Unique prefix for naming (e.g. ``"DSO_1"``).
    coupling_map : list of (ieee_bus_0idx, hv_bus_no)
        Each tuple connects a 0-indexed IEEE TN bus to a HV bus number
        (0--9) in this sub-network.
    line_length_scale : float
        Multiplicative factor for all HV line lengths (default 1.0).
    total_p_mw : float
        Target total active power across all 10 HV loads [MW].
        Each load gets ``total_p_mw / 10``.
    total_q_mvar : float
        Target total reactive power across all 10 HV loads [Mvar].
    gen_type : str
        ``"mixed"`` -- standard TUDA (4 wind + 4 PV).
        ``"pv"``    -- all wind replaced by PV of same capacity.
        ``"wind"``  -- all PV removed except 30 MW at HV bus 7.

    Returns
    -------
    HVNetworkInfo
    """
    # ── 1. Create 10 HV buses at 110 kV ──────────────────────────────────────
    bus_map: Dict[int, int] = {}
    bus_indices: List[int] = []
    for i in range(10):
        b = pp.create_bus(
            net, vn_kv=110.0,
            name=f"{net_id}|Bus_{i}",
            type="b", subnet="DN",
        )
        bus_map[i] = int(b)
        bus_indices.append(int(b))

    # ── 2. Create HV lines (TUDA topology, scaled lengths) ──────────────────
    line_indices: List[int] = []
    for f, t, base_km in HV_LINE_TOPOLOGY:
        li = pp.create_line(
            net,
            from_bus=bus_map[f],
            to_bus=bus_map[t],
            length_km=base_km * line_length_scale,
            std_type="184-AL1/30-ST1A 110.0",
            name=f"{net_id}|Line_({f}-{t})",
            subnet="DN",
        )
        line_indices.append(int(li))

    # ── 3. Create coupling transformers (3W, 345/110/20 kV, 300 MVA) ──────────
    coupling_trafo_indices: List[int] = []
    coupling_ieee_buses: List[int] = []
    coupling_hv_bus_indices: List[int] = []

    for ieee_bus, hv_no in coupling_map:
        hv_bus = bus_map[hv_no]

        # Create tertiary (LV) bus at 20 kV — star point for the 3W model
        lv_bus = pp.create_bus(
            net,
            vn_kv=20.0,
            name=f"{net_id}|Tertiary_TN{ieee_bus}_HV{hv_no}",
            subnet="DN",
        )

        vn_hv = float(net.bus.at[ieee_bus, "vn_kv"])  # 345 kV
        tidx = pp.create_transformer3w_from_parameters(
            net,
            hv_bus=ieee_bus,
            mv_bus=hv_bus,
            lv_bus=int(lv_bus),
            sn_hv_mva=300.0,
            sn_mv_mva=300.0,
            sn_lv_mva=75.0,
            vn_hv_kv=vn_hv,
            vn_mv_kv=110.0,
            vn_lv_kv=20.0,
            vk_hv_percent=12.0,
            vk_mv_percent=8.0,
            vk_lv_percent=10.0,
            vkr_hv_percent=0.30,
            vkr_mv_percent=0.20,
            vkr_lv_percent=0.25,
            pfe_kw=80.0,
            i0_percent=0.04,
            shift_mv_degree=0.0,
            shift_lv_degree=150.0,
            tap_side="hv",
            tap_neutral=0,
            tap_min=-13,
            tap_max=13,
            tap_pos=0,
            tap_step_percent=1.25,
            tap_changer_type="Ratio",
            name=f"{net_id}|Coupler3W_TN{ieee_bus}_HV{hv_no}",
        )
        coupling_trafo_indices.append(int(tidx))
        coupling_ieee_buses.append(ieee_bus)
        coupling_hv_bus_indices.append(hv_bus)

    # ── 4. Create loads (total P/Q distributed evenly across 10 buses) ───────
    load_indices: List[int] = []
    p_per_load = total_p_mw / 10.0
    q_per_load = total_q_mvar / 10.0
    sn_per_load = max(abs(p_per_load), abs(q_per_load), 1.0)

    for i in range(10):
        lidx = pp.create_load(
            net,
            bus=bus_map[i],
            sn_mva=sn_per_load,
            p_mw=p_per_load,
            q_mvar=q_per_load,
            name=f"{net_id}|HV_MV_Sub_{i}",
            subnet="DN",
            profile_p="mv_rural_pload",
            profile_q="mv_rural_qload",
        )
        load_indices.append(int(lidx))

    # ── 5. Create DER static generators ──────────────────────────────────────
    sgen_indices: List[int] = []
    cos_phi = 0.98
    tan_phi = np.tan(np.arccos(cos_phi))

    if gen_type == "mixed":
        # Standard TUDA: all 4 wind parks + all 4 PV plants
        for i, (bus_no, p_mw, profile) in enumerate(TUDA_WIND_PARKS):
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="WP", profile=profile,
                name=f"{net_id}|Wind_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))
        for i, (bus_no, p_mw) in enumerate(TUDA_PV_PLANTS):
            q_mvar = -p_mw * tan_phi
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=q_mvar, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))

    elif gen_type == "pv":
        # PV-dominated: original PV + wind locations as PV
        for i, (bus_no, p_mw) in enumerate(TUDA_PV_PLANTS):
            q_mvar = -p_mw * tan_phi
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=q_mvar, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))
        for i, (bus_no, p_mw, _) in enumerate(TUDA_WIND_PARKS):
            q_mvar = -p_mw * tan_phi
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=q_mvar, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_ex_wind_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))

    elif gen_type == "wind":
        # Wind-dominated: all wind + single PV at bus 7
        for i, (bus_no, p_mw, profile) in enumerate(TUDA_WIND_PARKS):
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="WP", profile=profile,
                name=f"{net_id}|Wind_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))
        for bus_no, p_mw in TUDA_PV_PLANTS:
            if bus_no != 7:
                continue
            q_mvar = -p_mw * tan_phi
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=q_mvar, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_0",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))

    else:
        raise ValueError(
            f"Unknown gen_type={gen_type!r}; use 'mixed', 'pv', or 'wind'."
        )

    # ── 6. STATCOM wind parks at each coupling bus ───────────────────────
    #   Place a 100 MVA STATCOM-capable wind park at every HV bus that
    #   couples to the 345 kV backbone via a 3W transformer.  This gives
    #   the DSO controller a continuously controllable Q actuator at each
    #   transformer node, improving reactive-power controllability.
    for ieee_bus, hv_no in coupling_map:
        hv_bus = bus_map[hv_no]
        wp_sn = HV_COUPLING_WP_MVA
        wp_p = wp_sn                    # rated P = S_n; Q headroom from profile < 1
        sidx = pp.create_sgen(
            net, bus=hv_bus,
            p_mw=wp_p, q_mvar=0.0, sn_mva=wp_sn,
            type="WP", profile="WP10",
            name=f"{net_id}|WP_STATCOM_HV{hv_no}",
            subnet="DN",
            op_diagram="STATCOM",
        )
        sgen_indices.append(int(sidx))

    return HVNetworkInfo(
        net_id=net_id,
        bus_indices=tuple(bus_indices),
        line_indices=tuple(line_indices),
        sgen_indices=tuple(sgen_indices),
        load_indices=tuple(load_indices),
        coupling_trafo_indices=tuple(coupling_trafo_indices),
        coupling_ieee_buses=tuple(coupling_ieee_buses),
        coupling_hv_bus_indices=tuple(coupling_hv_bus_indices),
        line_length_scale=line_length_scale,
        total_ref_p_mw=total_p_mw,
        total_ref_q_mvar=total_q_mvar,
        gen_type=gen_type,
    )


# =====================================================================
#  Internal: EHV profile wiring
# =====================================================================

def _wire_ehv_profiles(net: pp.pandapowerNet) -> None:
    """
    Add simbench profile columns to IEEE 39-bus loads that lack them.

    Assignment rule (mirrors the TUDA EHV load convention):
      - Loads at Zone 1 + Zone 2 buses  ->  HS4_pload / HS4_qload
      - Loads at Zone 3 buses           ->  HS5_pload / HS5_qload
    """
    if "profile_p" not in net.load.columns:
        net.load["profile_p"] = None
    if "profile_q" not in net.load.columns:
        net.load["profile_q"] = None

    for li in net.load.index:
        existing = net.load.at[li, "profile_p"]
        if existing is not None and str(existing) not in ("", "nan", "None", "NaN"):
            continue
        bus = int(net.load.at[li, "bus"])
        if bus in ZONE3_BUSES_0IDX:
            net.load.at[li, "profile_p"] = "HS5_pload"
            net.load.at[li, "profile_q"] = "HS5_qload"
        else:
            net.load.at[li, "profile_p"] = "HS4_pload"
            net.load.at[li, "profile_q"] = "HS4_qload"


# =====================================================================
#  Internal: reference load computation
# =====================================================================

def _compute_reference_loads(
    net: pp.pandapowerNet,
    *,
    coupler_sn_mva: float = 300.0,
    n_couplers: int = 3,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute the reference (total_p_mw, total_q_mvar) for each sub-network,
    capped at the aggregate coupling transformer capacity.

    Each HV sub-network connects to the TN via *n_couplers* 3-winding
    transformers rated at *coupler_sn_mva* each.  The DN load is capped at
    ``n_couplers * coupler_sn_mva`` (apparent power).  Any excess remains
    at the TN coupling buses as a separate load — see
    ``reduce_loads_at_bus``.

    For buses without a load, the average of the buses that DO have loads
    in the same 3-bus set is used.  DSO_3 is special: it uses the average
    of DSO_1 and DSO_2 totals (since its coupling buses have no IEEE load).

    Returns
    -------
    dict : net_id -> (total_p_mw, total_q_mvar)
    """
    max_s_mva = n_couplers * coupler_sn_mva          # 900 MVA default
    n_nets = len(SUBNET_DEFS)

    # ── Step 1: Pool all real loads at ALL coupling buses ───────────────
    pool_p = 0.0
    pool_q = 0.0
    for sdef in SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            p, q = get_load_at_bus(net, b1 - 1)
            pool_p += p
            pool_q += q

    # ── Step 2: Distribute equally, cap per network ────────────────────
    share_p = pool_p / n_nets if n_nets > 0 else 0.0
    share_q = pool_q / n_nets if n_nets > 0 else 0.0

    # Cap each share at coupler capacity (preserving power factor)
    share_s = (share_p ** 2 + share_q ** 2) ** 0.5
    if share_s > max_s_mva and share_s > 0:
        cap_scale = max_s_mva / share_s
        share_p *= cap_scale
        share_q *= cap_scale

    ref: Dict[str, Tuple[float, float]] = {}
    for sdef in SUBNET_DEFS:
        ref[sdef["net_id"]] = (share_p, share_q)

    return ref


# =====================================================================
#  Internal: debug summary printer
# =====================================================================

def _print_hv_summary(
    hv_nets: List[HVNetworkInfo],
    net: pp.pandapowerNet,
) -> None:
    """Print a formatted debug table of all HV sub-network connections."""
    print()
    print("=" * 80)
    print("  HV Sub-Network Connections (TUDA 110 kV topology copies)")
    print("=" * 80)
    print(f"  {'Sub-net':<8s} {'Zone':>4s}   {'IEEE(1-idx) -> HV bus':<28s} "
          f"{'Scale':>5s}  {'P_ref(MW)':>9s}  {'Q_ref':>8s}  {'Gen':>5s}")
    print("  " + "-" * 76)

    for hv in hv_nets:
        sdef = next(
            (s for s in SUBNET_DEFS if s["net_id"] == hv.net_id), None
        )
        if sdef is None:
            continue
        ieee_1 = sdef["ieee_1idx"]
        hv_b = sdef["hv_buses"]
        coupling_str = ", ".join(
            f"{i1}->{h}" for i1, h in zip(ieee_1, hv_b)
        )
        print(f"  {hv.net_id:<8s} {hv.zone:>4d}   {coupling_str:<28s} "
              f"{hv.line_length_scale:>5.2f}  {hv.total_ref_p_mw:>9.1f}  "
              f"{hv.total_ref_q_mvar:>8.1f}  {hv.gen_type:>5s}")

    print()
    print("  Coupling Transformers:")
    for hv in hv_nets:
        for tidx, ieee_b, hv_b in zip(
            hv.coupling_trafo_indices,
            hv.coupling_ieee_buses,
            hv.coupling_hv_bus_indices,
        ):
            tname = str(net.trafo3w.at[tidx, "name"])
            hv_name = str(net.bus.at[hv_b, "name"])
            print(f"    {tname:<35s}  TN bus {ieee_b} (345 kV)"
                  f"  <->  {hv_name} (110 kV)")

    # DER summary per sub-net
    print()
    print("  DER Generation Summary:")
    for hv in hv_nets:
        n_wp = sum(1 for s in hv.sgen_indices
                   if str(net.sgen.at[s, "type"]) == "WP")
        n_pv = sum(1 for s in hv.sgen_indices
                   if str(net.sgen.at[s, "type"]) == "PV")
        total_p = sum(float(net.sgen.at[s, "p_mw"]) for s in hv.sgen_indices)
        print(f"    {hv.net_id}: {n_wp} wind + {n_pv} PV = "
              f"{len(hv.sgen_indices)} sgens, {total_p:.0f} MW installed")

    print("=" * 80)
    print()


# =====================================================================
#  Public: attach all HV sub-networks
# =====================================================================

def add_hv_networks(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    *,
    q_compensation: bool = False,
    q_load_scale: float = 4.0,
    q_profile_max_factor: float = 0.329,
    verbose: bool = True,
) -> IEEE39NetworkMeta:
    """
    Attach five 110 kV HV sub-networks (copies of the TUDA DN topology) to
    the IEEE 39-bus 345 kV network.

    Each sub-network replaces the IEEE loads at its 3 coupling buses with
    a full meshed 110 kV network carrying the equivalent total load.
    Line lengths are scaled per sub-network, and generation mix varies by zone.

    Sub-network definitions
    -----------------------
    ====== ====  ===============  =====  ========
    ID     Zone  IEEE (1-idx)     Scale  Gen type
    ====== ====  ===============  =====  ========
    DSO_1   2    7, 8, 5          0.75   mixed
    DSO_2   2    14, 4, 3         1.50   mixed
    DSO_3   2    11, 10, 13       0.75   mixed
    DSO_4   3    24, 21, 23       2.00   pv
    DSO_5   1    27, 26, 25       3.00   wind
    ====== ====  ===============  =====  ========

    All sub-networks connect to TUDA HV buses (3, 0, 8) in that order.

    Special handling
    ----------------
    * **DSO_3** coupling buses have no IEEE load after removal; its reference
      load is set to the average of DSO_1 and DSO_2 totals.
    * EHV profiles (HS4/HS5) are wired to all remaining IEEE 39-bus loads.

    Parameters
    ----------
    net : pandapowerNet
        IEEE 39-bus network from ``build_ieee39_net()`` (modified in-place).
    meta : IEEE39NetworkMeta
        Existing metadata (replaced with updated copy).
    q_compensation : bool
        When True, add constant (non-profile-scaled) Q-consuming loads at
        the 345 kV coupling buses to compensate for the low Q-load profiles.
        The HS4/HS5 Q-load profiles peak at ~33% of the IEEE base case,
        making the reactive-power control problem unrealistically easy.
        These compensation loads raise the peak Q to approximately the
        original IEEE 39-bus level.
    q_load_scale : float
        Scale factor for the Q-load base values (default 2.0).  Applied
        to ``base_q_mvar`` of all profile-scaled loads to increase the
        profile-driven Q variation.  The actual ``q_mvar`` (used by the
        initial PF) is NOT changed; only ``base_q_mvar`` is set to the
        scaled value so that ``apply_profiles()`` uses the larger base.
    q_profile_max_factor : float
        Maximum Q-load profile scaling factor (default 0.329 from HS4/HS5).
        Used to compute the Q deficit:
        ``Q_comp = original_Q - scaled_Q * q_profile_max_factor``.
    verbose : bool
        Print connection summary table (default True).

    Returns
    -------
    meta : IEEE39NetworkMeta
        Updated metadata with ``hv_networks`` populated.
    """

    # =====================================================================
    # 1. Compute reference loads BEFORE modifying anything
    # =====================================================================
    ref_loads = _compute_reference_loads(net)

    if verbose:
        print("[add_hv_networks] Reference loads from IEEE 39-bus:")
        for net_id, (p, q) in ref_loads.items():
            print(f"  {net_id}: P={p:.1f} MW, Q={q:.1f} Mvar")

    # =====================================================================
    # 2. Reduce IEEE loads at coupling buses by the amount moved to DN
    # =====================================================================
    # The total DN load (pooled across all DSOs) was distributed equally
    # among the HV networks.  We now reduce the original TN loads at the
    # coupling buses by a total of (n_nets * share) = pool.  The reduction
    # is distributed proportionally across all loaded coupling buses.
    all_coupling_buses_0idx = set()

    # Collect all coupling buses and their original loads
    _original_bus_loads: Dict[int, Tuple[float, float]] = {}
    for sdef in SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            b0 = b1 - 1
            all_coupling_buses_0idx.add(b0)
            if b0 not in _original_bus_loads:
                _original_bus_loads[b0] = get_load_at_bus(net, b0)

    # Total load at all coupling buses (= pool) and total DN load
    pool_p = sum(p for p, q in _original_bus_loads.values())
    pool_q = sum(q for p, q in _original_bus_loads.values())
    n_nets = len(SUBNET_DEFS)
    total_dn_p = sum(p for p, q in ref_loads.values())
    total_dn_q = sum(q for p, q in ref_loads.values())

    # Reduce each loaded coupling bus proportionally
    for b in sorted(all_coupling_buses_0idx):
        bp, bq = _original_bus_loads[b]
        if bp == 0.0 and bq == 0.0:
            continue
        frac_p = bp / pool_p if pool_p > 0 else 0.0
        frac_q = bq / pool_q if pool_q > 0 else 0.0
        remove_p = total_dn_p * frac_p
        remove_q = total_dn_q * frac_q
        reduce_loads_at_bus(net, b, remove_p, remove_q)

    if verbose:
        print("[add_hv_networks] Reduced IEEE loads at coupling buses "
              f"(0-idx): {sorted(all_coupling_buses_0idx)}")
        for b in sorted(all_coupling_buses_0idx):
            orig_p, orig_q = _original_bus_loads[b]
            now_p, now_q = get_load_at_bus(net, b)
            if orig_p > 0 or orig_q > 0:
                print(f"  Bus {b}: {orig_p:.1f} -> {now_p:.1f} MW "
                      f"(kept {now_p:.1f} MW at TN)")

    # Also remove TN-DER sgens at coupling buses (they were placed before
    # HV sub-networks replaced the loads).
    sgens_to_remove = net.sgen.index[net.sgen["bus"].isin(all_coupling_buses_0idx)].tolist()

    tso_der_indices_updated = list(meta.tso_der_indices)
    tso_der_buses_updated = list(meta.tso_der_buses)

    if sgens_to_remove:
        net.sgen.drop(index=sgens_to_remove, inplace=True)
        # Update meta to remove these sgens from tso_der lists
        removed_set = set(sgens_to_remove)
        tso_der_indices_updated = [s for s in tso_der_indices_updated if s not in removed_set]
        tso_der_buses_updated = [
            b for s, b in zip(meta.tso_der_indices, meta.tso_der_buses)
            if s not in removed_set
        ]

    # =====================================================================
    # 3. Create 5 HV sub-networks
    # =====================================================================
    hv_nets: List[HVNetworkInfo] = []

    for sdef in SUBNET_DEFS:
        net_id = sdef["net_id"]
        ieee_0idx = [b - 1 for b in sdef["ieee_1idx"]]
        hv_buses = sdef["hv_buses"]
        coupling_map = list(zip(ieee_0idx, hv_buses))
        total_p, total_q = ref_loads[net_id]
        total_q_scaled = total_q * HV_Q_LOAD_FACTOR

        if verbose:
            print(f"[add_hv_networks] Creating {net_id} (zone {sdef['zone']}, "
                  f"{sdef['gen']}, scale {sdef['scale']:.2f}x, "
                  f"P={total_p:.1f} MW, Q={total_q_scaled:.1f} Mvar "
                  f"[Q x{HV_Q_LOAD_FACTOR:.1f}]) ...")

        hv = _create_hv_subnetwork(
            net, net_id, coupling_map,
            line_length_scale=sdef["scale"],
            total_p_mw=total_p,
            total_q_mvar=total_q_scaled,
            gen_type=sdef["gen"],
        )
        # Attach zone metadata
        hv.zone = sdef["zone"]
        hv_nets.append(hv)

    # =====================================================================
    # 4. (EHV profile wiring is performed during build_ieee39_net's 50/50
    #    load split; no extra wiring needed here.)
    # =====================================================================

    # =====================================================================
    # 5. Verification power flow (at scaled Q level)
    # =====================================================================
    # init='auto' re-uses the TN solution from build_ieee39_net; new HV
    # buses fall back to flat start.  This converges better than init='flat'
    # when HV_Q_LOAD_FACTOR > 1.
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             init='auto', max_iteration=100)

    # =====================================================================
    # 6. Debug output
    # =====================================================================
    if verbose:
        _print_hv_summary(hv_nets, net)

    # =====================================================================
    # 6b. Q-load scaling + constant Q compensation at coupling buses
    # =====================================================================
    # The HS4/HS5 Q-load profiles peak at ~33% of the IEEE 39 base case.
    #
    # base_q_mvar scaling
    # -------------------
    # q_load_scale amplifies the profile-driven Q variation so that at
    # peak the Q is closer to the IEEE 39 base value.
    #
    # Layer-specific treatment:
    #   - TN loads: base_q_mvar = q_mvar * q_load_scale  (profile peak too
    #     low otherwise).
    #   - DN/HV loads: base_q_mvar = q_mvar  (already elevated by
    #     HV_Q_LOAD_FACTOR, no extra scaling needed).
    #
    # Optional constant Q compensation (q_compensation=True) adds non-
    # profile-scaled loads at the 345 kV coupling buses to fill the
    # remaining gap.  With HV_Q_LOAD_FACTOR >= 3 this is typically not
    # needed (default: q_compensation=False).

    # Always pre-set base values so snapshot_base_values() won't overwrite
    # Original IEEE 39 loads have no "subnet" column; HV loads have "DN".
    # Detect DN explicitly; everything else is TN.
    is_dn = (net.load["subnet"].astype(str) == "DN") if "subnet" in net.load.columns else False
    is_tn = ~is_dn
    total_q = float(net.load["q_mvar"].sum())

    net.load["base_q_mvar"] = net.load["q_mvar"].copy()
    net.load["base_p_mw"] = net.load["p_mw"].copy()

    # TN loads: apply q_load_scale to increase profile amplitude
    if "profile_q" in net.load.columns:
        tn_has_profile = is_tn & net.load["profile_q"].notna() & (
            net.load["profile_q"].astype(str) != ""
        )
        net.load.loc[tn_has_profile, "base_q_mvar"] = (
            net.load.loc[tn_has_profile, "q_mvar"] * q_load_scale
        )
    # DN/HV loads: base_q_mvar = q_mvar (already x HV_Q_LOAD_FACTOR)

    scaled_base_q = float(net.load["base_q_mvar"].sum())
    tn_q_total = float(net.load.loc[is_tn, "q_mvar"].sum())

    if q_compensation:
        # Constant Q compensation (TN loads only, DN already elevated)
        q_comp_total = tn_q_total * (
            1.0 - q_load_scale * q_profile_max_factor
        )
        q_comp_total = max(q_comp_total, 0.0)

        n_coupling = len(all_coupling_buses_0idx)
        q_per_bus = q_comp_total / n_coupling if n_coupling > 0 else 0.0

        for b in sorted(all_coupling_buses_0idx):
            lidx = pp.create_load(
                net,
                bus=b,
                p_mw=0.0,
                q_mvar=q_per_bus,
                sn_mva=abs(q_per_bus) if q_per_bus != 0 else 1.0,
                name=f"Q_COMP|bus{b}",
                subnet="TN",
            )
            net.load.at[lidx, "base_p_mw"] = 0.0
            net.load.at[lidx, "base_q_mvar"] = q_per_bus
    else:
        q_comp_total = 0.0

    # ── Re-initialise STATCOM Q for the current operating point ──────
    _statcom_mask = net.sgen["name"].astype(str).str.contains("STATCOM")
    _statcom_idxs = net.sgen.index[_statcom_mask].tolist()
    if _statcom_idxs:
        _tmp_map: Dict[int, int] = {}
        for si in _statcom_idxs:
            bus = int(net.sgen.at[si, "bus"])
            p = float(net.sgen.at[si, "p_mw"])
            sn = float(net.sgen.at[si, "sn_mva"])
            net.sgen.at[si, "in_service"] = False
            gi = pp.create_gen(
                net, bus=bus, p_mw=p, vm_pu=1.03, sn_mva=sn,
                max_q_mvar=sn, min_q_mvar=-sn,
                in_service=True, name="_TEMP_REINIT",
            )
            _tmp_map[int(gi)] = si
        pp.runpp(net, run_control=False, calculate_voltage_angles=True,
                 init='auto', max_iteration=100)
        for gi, si in _tmp_map.items():
            net.sgen.at[si, "q_mvar"] = float(net.res_gen.at[gi, "q_mvar"])
            net.sgen.at[si, "in_service"] = True
        net.gen.drop(index=list(_tmp_map.keys()), inplace=True)
        pp.runpp(net, run_control=False, calculate_voltage_angles=True,
                 init='auto', max_iteration=100)

    if verbose:
        print(f"[add_hv_networks] Q scaling: HV_Q_LOAD_FACTOR={HV_Q_LOAD_FACTOR:.1f}, "
              f"q_load_scale(TN)={q_load_scale:.1f}x, "
              f"q_compensation={q_compensation}")
        print(f"  TN Q: {tn_q_total:.1f} Mvar, total Q: {total_q:.1f} Mvar")
        print(f"  Scaled base_q_mvar total: {scaled_base_q:.1f} Mvar "
              f"(profile peak ~ {scaled_base_q * q_profile_max_factor:.0f} Mvar)")
        if q_compensation:
            print(f"  Constant Q compensation: {q_comp_total:.1f} Mvar")
        peak_q_est = scaled_base_q * q_profile_max_factor + q_comp_total
        print(f"  Estimated peak total Q: {peak_q_est:.0f} Mvar")

    # =====================================================================
    # 7. Update metadata
    # =====================================================================
    all_dn_buses = sorted(
        int(b) for b in net.bus.index
        if str(net.bus.at[b, "subnet"]) == "DN"
    )
    all_dn_lines = sorted(
        int(li) for li in net.line.index
        if str(net.line.at[li, "subnet"]) == "DN"
    )

    return IEEE39NetworkMeta(
        tn_bus_indices=meta.tn_bus_indices,
        tn_line_indices=meta.tn_line_indices,
        gen_indices=meta.gen_indices,
        gen_bus_indices=meta.gen_bus_indices,
        gen_grid_bus_indices=meta.gen_grid_bus_indices,
        machine_trafo_indices=meta.machine_trafo_indices,
        machine_trafo_gen_map=meta.machine_trafo_gen_map,
        tso_der_indices=tuple(tso_der_indices_updated),
        tso_der_buses=tuple(tso_der_buses_updated),
        # DSO fields carried over
        dso_pcc_trafo_indices=meta.dso_pcc_trafo_indices,
        dso_pcc_hv_buses=meta.dso_pcc_hv_buses,
        dso_lv_buses=meta.dso_lv_buses,
        dso_der_indices=meta.dso_der_indices,
        dso_der_buses=meta.dso_der_buses,
        dso_shunt_indices=meta.dso_shunt_indices,
        dso_shunt_buses=meta.dso_shunt_buses,
        # DN indices cover all HV sub-network elements
        dn_bus_indices=tuple(all_dn_buses),
        dn_line_indices=tuple(all_dn_lines),
        # HV sub-network tracking
        hv_networks=tuple(hv_nets),
    )
