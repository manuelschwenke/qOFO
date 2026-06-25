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
    HV_HIGH_LOAD_BUS_NOS,
    HV_HIGH_LOAD_FACTOR,
    LOAD_CONST_FRACTION,
    LOAD_PEAK_BOOST,
    LOAD_VAR_FRACTION,
    PROFILE_MAX,
    PROFILE_MEAN,
    TUDA_WIND_PARKS,
    TUDA_PV_PLANTS,
    ZONE3_BUSES_0IDX,
    SUBNET_DEFS,
)
from network.ieee39.helpers import get_load_at_bus


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
    coupling_lv_bus_indices: List[int] = []

    for ieee_bus, hv_no in coupling_map:
        hv_bus = bus_map[hv_no]

        # Create tertiary (LV) bus at 20 kV — star point for the 3W model
        lv_bus = pp.create_bus(
            net,
            vn_kv=20.0,
            name=f"{net_id}|Tertiary_TN{ieee_bus}_HV{hv_no}",
            subnet="DN",
        )
        coupling_lv_bus_indices.append(int(lv_bus))

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

    # ── 4. Create loads — two rows per HV bus, mirroring the TN convention:
    #       * Constant row: no profile, carries time-mean.  For P this is
    #         0.5 * per-bus share.  For Q this is the FULL per-bus share:
    #         ``mv_rural_qload`` has near-zero mean (≈ -0.050), so the
    #         variable row contributes ≈ 0 on time average, and the
    #         constant row must carry the whole HV Q mean to match
    #         ``total_q_mvar``.
    #       * Variable row: profile-driven.  For P, ``base_p_mw`` is mean-
    #         normalised so the time mean equals 0.5 * per-bus share.  For
    #         Q, ``base_q_mvar = 0.5 * q_per_bus`` sets the reactive swing
    #         amplitude while its time mean adds a small capacitive bias
    #         (≈ -2.5 % of HV Q), accepted as a design tradeoff.
    #
    #       Load concentration on ``HV_HIGH_LOAD_BUS_NOS`` is expressed
    #       as a relative weight so that the weighted sum across all 10
    #       buses equals ``total_p_mw`` / ``total_q_mvar`` (no hidden
    #       inflation).
    load_indices: List[int] = []
    _high_load_set = set(HV_HIGH_LOAD_BUS_NOS)
    raw_weights = [HV_HIGH_LOAD_FACTOR if i in _high_load_set else 1.0
                   for i in range(10)]
    weight_sum = sum(raw_weights)
    weights = [10.0 * w / weight_sum for w in raw_weights]

    p_per_bus_uniform = total_p_mw / 10.0
    q_per_bus_uniform = total_q_mvar / 10.0
    c = LOAD_CONST_FRACTION
    v = LOAD_VAR_FRACTION
    max_mv_p = PROFILE_MAX["mv_rural_pload"]
    mean_mv_p = PROFILE_MEAN["mv_rural_pload"]

    for i in range(10):
        w = weights[i]
        p_per_bus = p_per_bus_uniform * w
        q_per_bus = q_per_bus_uniform * w
        sn_bus = max(abs(p_per_bus), abs(q_per_bus), 1.0)

        # Constant row: c * P (so peak = (c+v) * p_per_bus = p_per_bus when
        # the variable profile hits its maximum), FULL Q (the mv_rural_qload
        # profile has near-zero mean so the variable Q row swings around 0
        # and the constant row carries the per-bus mean Q).
        lidx_c = pp.create_load(
            net,
            bus=bus_map[i],
            sn_mva=sn_bus,
            p_mw=c * p_per_bus,
            q_mvar=q_per_bus,
            name=f"{net_id}|HV_MV_Sub_{i}_const",
            subnet="DN",
        )
        net.load.at[lidx_c, "base_p_mw"] = c * p_per_bus
        net.load.at[lidx_c, "base_q_mvar"] = q_per_bus
        net.load.at[lidx_c, "profile_p"] = None
        net.load.at[lidx_c, "profile_q"] = None
        load_indices.append(int(lidx_c))

        # Variable row: max-normalised P (peak at LOAD_PEAK_BOOST * v *
        # p_per_bus when profile hits max); swing-amplitude Q around zero
        # (kept as 0.5 * q_per_bus to preserve the previous Q swing behaviour,
        # since mv_rural_qload has near-zero mean and ~0.34 max amplitude --
        # the boost is intentionally not applied to HV Q).
        base_p_var = LOAD_PEAK_BOOST * v * p_per_bus / max_mv_p
        base_q_var = 0.5 * q_per_bus
        # Initial p_mw at the per-bus IEEE share at peak (c+v=1.0); this
        # plus the c*p_per_bus constant row gives a startup PF that matches
        # the original 0.5 + 0.5 = 1.0 * p_per_bus convention.
        # apply_profiles() overwrites this every step.
        lidx_v = pp.create_load(
            net,
            bus=bus_map[i],
            sn_mva=sn_bus,
            p_mw=v * p_per_bus,
            q_mvar=0.0,
            name=f"{net_id}|HV_MV_Sub_{i}_var",
            subnet="DN",
            profile_p="mv_rural_pload",
            profile_q="mv_rural_qload",
        )
        net.load.at[lidx_v, "base_p_mw"] = base_p_var
        net.load.at[lidx_v, "base_q_mvar"] = base_q_var
        load_indices.append(int(lidx_v))

    # ── 5. Create DER static generators ──────────────────────────────────────
    # All HV-side DER (WP, PV, STATCOM) initialise with q_mvar=0.  The
    # DSO controller dispatches Q at run time.
    sgen_indices: List[int] = []

    if gen_type == "mixed":
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
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))

    elif gen_type == "pv":
        for i, (bus_no, p_mw) in enumerate(TUDA_PV_PLANTS):
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))
        for i, (bus_no, p_mw, _) in enumerate(TUDA_WIND_PARKS):
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="PV", profile="PV3",
                name=f"{net_id}|PV_ex_wind_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))

    elif gen_type == "wind":
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
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
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
        coupling_lv_bus_indices=tuple(coupling_lv_bus_indices),
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
    Compute the reference (total_p_mw, total_q_mvar) for each sub-network.

    Each HV sub-network carries **half** of the pooled coupling-bus load
    (the other half stays at the 345 kV coupling bus as the constant row
    produced by ``_split_tn_loads``).  The half that moves is distributed
    equally across the HV sub-networks and capped at ``n_couplers *
    coupler_sn_mva`` per DSO.

    Pool is reconstructed from the constant rows: each bus contributes
    ``(1/c) * base_p_mw`` (resp. ``base_q_mvar``) of its constant-row,
    which equals the original pre-split nominal load (since the constant
    row stores ``c * p_orig``).  The previous 2.0x factor assumed the
    fixed 50/50 split; now ``c = LOAD_CONST_FRACTION`` is configurable.

    Returns
    -------
    dict : net_id -> (total_p_mw, total_q_mvar)  -- the HALF that moves to HV
    """
    max_s_mva = n_couplers * coupler_sn_mva          # 900 MVA default
    n_nets = len(SUBNET_DEFS)

    inv_c = 1.0 / LOAD_CONST_FRACTION

    pool_p_full = 0.0
    pool_q_full = 0.0
    for sdef in SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            b0 = b1 - 1
            mask = (net.load["bus"] == b0) & (
                net.load["subnet"].astype(str) == "TN"
            ) & net.load["profile_p"].isna()
            pool_p_full += inv_c * float(net.load.loc[mask, "base_p_mw"].sum())
            pool_q_full += inv_c * float(net.load.loc[mask, "base_q_mvar"].sum())

    # HALF of the full pool moves to HV (mirrors the original 50/50 split
    # convention; the per-coupling-bus distribution and HV per-bus split
    # remain symmetric so total system mean is preserved).
    pool_p = 0.5 * pool_p_full
    pool_q = 0.5 * pool_q_full

    share_p = pool_p / n_nets if n_nets > 0 else 0.0
    share_q = pool_q / n_nets if n_nets > 0 else 0.0

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
    install_tso_tertiary_shunts: bool = True,
    tso_tertiary_shunt_q_mvar: float = 50.0,
    tso_shunt_kind: str = "bipolar",
    msc_n_levels: int = 4,
    msr_n_levels: int = 4,
    msc_q_step_mvar: float = 50.0,
    msr_q_step_mvar: float = 50.0,
    verbose: bool = True,
) -> IEEE39NetworkMeta:
    """
    Attach 110 kV HV sub-networks (copies of the TUDA DN topology) to the
    IEEE 39-bus 345 kV network.

    Load redistribution convention
    ------------------------------
    ``build_ieee39_net`` first splits every 345 kV load into a constant
    half and a profile-driven half (see ``_split_tn_loads``).  For each
    coupling bus belonging to a sub-network, this function

      * keeps the constant half at 345 kV (unchanged), and
      * deletes the profile half; the equivalent power is moved into the
        HV sub-network as loads with their own 50 % const + 50 %
        mv_rural-driven split (see ``_create_hv_subnetwork``).

    As a result the time mean of the aggregate P and Q matches the IEEE
    39 base case (up to small biases from profile-mean rounding).

    Sub-network definitions come from :data:`SUBNET_DEFS`.

    Parameters
    ----------
    net : pandapowerNet
        IEEE 39-bus network from ``build_ieee39_net()`` (modified in-place).
    meta : IEEE39NetworkMeta
        Existing metadata (replaced with updated copy).
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
    # 2. Delete the profile-half TN rows at coupling buses
    # =====================================================================
    # The equivalent power is moved into the HV sub-network (step 3).
    # The constant-half rows stay untouched at 345 kV.
    all_coupling_buses_0idx = set()
    _original_bus_loads: Dict[int, Tuple[float, float]] = {}
    for sdef in SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            b0 = b1 - 1
            all_coupling_buses_0idx.add(b0)
            if b0 not in _original_bus_loads:
                _original_bus_loads[b0] = get_load_at_bus(net, b0)

    for b in sorted(all_coupling_buses_0idx):
        mask = (
            (net.load["bus"] == b)
            & (net.load["subnet"].astype(str) == "TN")
            & net.load["profile_p"].notna()
        )
        if mask.any():
            net.load.drop(index=net.load.index[mask], inplace=True)

    if verbose:
        print("[add_hv_networks] Dropped profile-half TN rows at coupling "
              f"buses (0-idx): {sorted(all_coupling_buses_0idx)}")
        for b in sorted(all_coupling_buses_0idx):
            orig_p, orig_q = _original_bus_loads[b]
            now_p, now_q = get_load_at_bus(net, b)
            if orig_p > 0 or orig_q > 0:
                print(f"  Bus {b}: {orig_p:.1f} MW total "
                      f"-> {now_p:.1f} MW constant-half at TN")

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

        if verbose:
            print(f"[add_hv_networks] Creating {net_id} (zone {sdef['zone']}, "
                  f"{sdef['gen']}, scale {sdef['scale']:.2f}x, "
                  f"P={total_p:.1f} MW, Q={total_q:.1f} Mvar) ...")

        hv = _create_hv_subnetwork(
            net, net_id, coupling_map,
            line_length_scale=sdef["scale"],
            total_p_mw=total_p,
            total_q_mvar=total_q,
            gen_type=sdef["gen"],
        )
        hv.zone = sdef["zone"]
        hv_nets.append(hv)

    # =====================================================================
    # 4. TSO-owned bipolar shunts at first tertiary of each DSO
    # =====================================================================
    # One bipolar 50 Mvar shunt per active DSO sub-network at the 20 kV
    # tertiary bus of the FIRST coupling 3-winding transformer.  These are
    # TSO actuators (state ∈ {-1, 0, +1}); the DSO controllers see them as
    # disturbances only (DSOControllerConfig.shunt_bus_indices stays []).
    # Installed at step=0 so the operating point is unchanged at build time.
    tso_sh_idx: List[int] = []
    tso_sh_buses: List[int] = []
    tso_sh_q: List[float] = []
    tso_sh_zone: List[int] = []
    tso_sh_kind: List[str] = []
    tso_sh_nlev: List[int] = []
    if install_tso_tertiary_shunts:
        if tso_shunt_kind not in ("bipolar", "msc_msr"):
            raise ValueError(
                f"tso_shunt_kind must be 'bipolar' or 'msc_msr', "
                f"got {tso_shunt_kind!r}"
            )
        for hv in hv_nets:
            tert_bus = int(hv.coupling_lv_bus_indices[0])
            if tso_shunt_kind == "bipolar":
                # Legacy MIQP-dispatched bipolar bank (state ∈ {-1, 0, +1}).
                sh = pp.create_shunt(
                    net, bus=tert_bus,
                    q_mvar=float(tso_tertiary_shunt_q_mvar),
                    p_mw=0.0, vn_kv=20.0,
                    step=0, max_step=1,
                    name=f"{hv.net_id}|TSOShunt_Tertiary",
                    in_service=True,
                )
                tso_sh_idx.append(int(sh))
                tso_sh_buses.append(tert_bus)
                tso_sh_q.append(float(tso_tertiary_shunt_q_mvar))
                tso_sh_zone.append(int(hv.zone))
                tso_sh_kind.append("BIPOLAR")
                tso_sh_nlev.append(1)
            else:
                # Integrator-dispatched MSC (capacitor) + MSR (reactor) banks,
                # both at the same tertiary bus.  Sign convention follows the
                # pandapower load convention (see test_pp_bipolar_step):
                #   res_shunt.q = q_mvar · step · V².
                #   MSC injects Q (raises V)  -> q_mvar < 0, step ∈ {0 … N_C}
                #   MSR absorbs Q (lowers V)  -> q_mvar > 0, step ∈ {0 … N_R}
                # The two banks are distinguished downstream by their net.shunt
                # index (NOT just the bus), so the SMW refresh and toggle target
                # the correct device.
                sh_c = pp.create_shunt(
                    net, bus=tert_bus,
                    q_mvar=-float(msc_q_step_mvar),
                    p_mw=0.0, vn_kv=20.0,
                    step=0, max_step=int(msc_n_levels),
                    name=f"{hv.net_id}|TSO_MSC_Tertiary",
                    in_service=True,
                )
                tso_sh_idx.append(int(sh_c))
                tso_sh_buses.append(tert_bus)
                tso_sh_q.append(float(msc_q_step_mvar))
                tso_sh_zone.append(int(hv.zone))
                tso_sh_kind.append("MSC")
                tso_sh_nlev.append(int(msc_n_levels))

                sh_r = pp.create_shunt(
                    net, bus=tert_bus,
                    q_mvar=+float(msr_q_step_mvar),
                    p_mw=0.0, vn_kv=20.0,
                    step=0, max_step=int(msr_n_levels),
                    name=f"{hv.net_id}|TSO_MSR_Tertiary",
                    in_service=True,
                )
                tso_sh_idx.append(int(sh_r))
                tso_sh_buses.append(tert_bus)
                tso_sh_q.append(float(msr_q_step_mvar))
                tso_sh_zone.append(int(hv.zone))
                tso_sh_kind.append("MSR")
                tso_sh_nlev.append(int(msr_n_levels))
        # Defensive: pandapower may declare net.shunt['step'] as uint;
        # bipolar writes need int64 so step = -1 round-trips through pandas.
        if "step" in net.shunt.columns:
            net.shunt["step"] = net.shunt["step"].astype("int64")
        if verbose:
            print(f"[add_hv_networks] Installed {len(tso_sh_idx)} TSO-owned "
                  f"tertiary shunts (kind={tso_shunt_kind}).")

    # =====================================================================
    # 5. Re-initialise TSO STATCOM Q via temp PV-gens, then verify PF
    # =====================================================================
    # TSO-side wind park sgens carry a Q value seeded by ``wind_replace`` at
    # the *pre-HV* operating point.  Adding the HV sub-networks shifts that
    # operating point (load redistribution + new HV gens/loads), so the
    # seeded Q is no longer self-consistent.  Temporarily disable the
    # STATCOM sgens and replace them with PV-gens that fix vm_pu=1.03 at
    # their grid bus; one PF then yields the Q each STATCOM must carry to
    # hold that voltage at the new state.  The PF is robust because the
    # PV-gens absorb mismatch as Q.
    #
    # HV-side (subnet=="DN") STATCOMs stay at q_mvar=0 at build time; the
    # DSO controller dispatches their Q at run time.
    _statcom_mask = (
        net.sgen["name"].astype(str).str.contains("STATCOM")
        & (net.sgen["subnet"].astype(str) != "DN")
    )
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

    # Verification power flow (with STATCOM Q already self-consistent if
    # the reinit ran above).  Runs unconditionally so scenarios without
    # STATCOM sgens still get a final convergence check.
    pp.runpp(net, run_control=False, calculate_voltage_angles=True,
             init='auto', max_iteration=100)

    # =====================================================================
    # 6. Debug output
    # =====================================================================
    if verbose:
        _print_hv_summary(hv_nets, net)

    if verbose:
        is_dn = net.load["subnet"].astype(str) == "DN"
        is_tn = ~is_dn
        print(f"[add_hv_networks] Load summary (after redistribution):")
        print(f"  TN P={net.load.loc[is_tn,'p_mw'].sum():.1f} MW, "
              f"Q={net.load.loc[is_tn,'q_mvar'].sum():.1f} Mvar "
              f"(sum of base: P={net.load.loc[is_tn,'base_p_mw'].sum():.1f}, "
              f"Q={net.load.loc[is_tn,'base_q_mvar'].sum():.1f})")
        print(f"  HV P={net.load.loc[is_dn,'p_mw'].sum():.1f} MW, "
              f"Q={net.load.loc[is_dn,'q_mvar'].sum():.1f} Mvar "
              f"(sum of base: P={net.load.loc[is_dn,'base_p_mw'].sum():.1f}, "
              f"Q={net.load.loc[is_dn,'base_q_mvar'].sum():.1f})")

    # =====================================================================
    # 6b. Explicit init: TSO-owned shunts at step=0 (off)
    # =====================================================================
    # Defensive reset right before metadata return.  The shunts were
    # created at step=0 in section 4, but the verification PF and the
    # STATCOM reinit pass run between then and now — be explicit so the
    # init contract holds regardless of intervening code.
    for sh_idx in tso_sh_idx:
        net.shunt.at[sh_idx, "step"] = 0

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

    # Aggregate per-HV sgen indices into the flat dso_der_indices /
    # dso_der_buses registries so downstream steps (DER classification,
    # generic per-DER iteration) can find every DSO-controlled converter
    # without recursing into hv_networks.
    flat_dso_der_indices: List[int] = list(meta.dso_der_indices)
    flat_dso_der_buses: List[int] = list(meta.dso_der_buses)
    for hv in hv_nets:
        for s in hv.sgen_indices:
            flat_dso_der_indices.append(int(s))
            flat_dso_der_buses.append(int(net.sgen.at[int(s), "bus"]))

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
        dso_der_indices=tuple(flat_dso_der_indices),
        dso_der_buses=tuple(flat_dso_der_buses),
        dso_shunt_indices=meta.dso_shunt_indices,
        dso_shunt_buses=meta.dso_shunt_buses,
        # TSO-owned bipolar tertiary shunts (one per active DSO sub-network)
        tso_tertiary_shunt_indices=tuple(tso_sh_idx),
        tso_tertiary_shunt_buses=tuple(tso_sh_buses),
        tso_tertiary_shunt_q_steps_mvar=tuple(tso_sh_q),
        tso_tertiary_shunt_zones=tuple(tso_sh_zone),
        tso_tertiary_shunt_kinds=tuple(tso_sh_kind),
        tso_tertiary_shunt_n_levels=tuple(tso_sh_nlev),
        # DN indices cover all HV sub-network elements
        dn_bus_indices=tuple(all_dn_buses),
        dn_line_indices=tuple(all_dn_lines),
        # HV sub-network tracking
        hv_networks=tuple(hv_nets),
    )
