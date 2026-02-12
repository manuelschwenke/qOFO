#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Network Builder — TU Darmstadt TSO-DSO Benchmark
=========================================================

Builds a combined 380/110/20 kV benchmark network for cascaded OFO studies.

Network topology
----------------
*  7 EHV buses at 380 kV  (TN|Bus_0 … TN|Bus_6)
* 10 HV  buses at 110 kV  (DN|Bus_0 … DN|Bus_9)
*  3 tertiary buses at 20 kV (one per 3-winding coupler transformer)
*  1 generator-terminal bus at 15 kV (one per conventional generator)

TSO–DSO interface
-----------------
Three 3-winding transformers (380/110/20 kV) couple the transmission and
distribution grids.  Their 20 kV tertiary windings carry switchable shunt
reactors for reactive-power compensation.

Public API
----------
``build_tuda_net(pv_nodes, ext_grid_vm_pu, load_scaling)``
    → ``(pp.pandapowerNet, NetworkMetadata)``

``NetworkMetadata``
    Structured, immutable record of every element index created during the
    build process.  Used downstream by the split and controller modules.

Author: Manuel Schwenke
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandapower as pp


# ═══════════════════════════════════════════════════════════════════════════════
#  NETWORK METADATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NetworkMetadata:
    """Structured record of all element indices created during the build.

    Every field is a ``list[int]`` holding pandapower element indices.
    The dataclass is frozen so that downstream code cannot accidentally mutate
    the metadata that was produced at build time.

    Attributes
    ----------
    coupler_trafo3w_indices : list[int]
        Indices into ``net.trafo3w`` for the three 380/110/20 kV couplers.
    coupler_hv_buses : list[int]
        380 kV (HV) bus of each coupler (same order as *coupler_trafo3w_indices*).
    coupler_mv_buses : list[int]
        110 kV (MV) bus of each coupler.
    coupler_lv_buses : list[int]
        20 kV tertiary bus of each coupler.
    tertiary_shunt_indices : list[int]
        Indices into ``net.shunt`` for the reactors at each tertiary winding.
    machine_trafo_indices : list[int]
        Indices into ``net.trafo`` for each machine transformer
        (generator terminal → grid bus).
    tn_shunt_indices : list[int]
        Indices into ``net.shunt`` for transmission-level shunt compensation.
    """

    coupler_trafo3w_indices: List[int] = field(default_factory=list)
    coupler_hv_buses: List[int] = field(default_factory=list)
    coupler_mv_buses: List[int] = field(default_factory=list)
    coupler_lv_buses: List[int] = field(default_factory=list)
    tertiary_shunt_indices: List[int] = field(default_factory=list)
    machine_trafo_indices: List[int] = field(default_factory=list)
    tn_shunt_indices: List[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping:  (EHV bus number, HV bus number) for each coupler transformer.
_COUPLER_MAP: List[Tuple[int, int]] = [
    (0, 3),   # Coupler 0: TN|Bus_0 (380 kV) ↔ DN|Bus_3 (110 kV)
    (3, 0),   # Coupler 1: TN|Bus_3 (380 kV) ↔ DN|Bus_0 (110 kV)
    (5, 8),   # Coupler 2: TN|Bus_5 (380 kV) ↔ DN|Bus_8 (110 kV)
]

# 3-winding transformer nameplate data (380/110/20 kV).
_TR3W: Dict[str, float] = {
    "sn_hv_mva": 300.0,
    "sn_mv_mva": 300.0,
    "sn_lv_mva": 75.0,
    "vn_hv_kv": 380.0,
    "vn_mv_kv": 110.0,
    "vn_lv_kv": 20.0,
    "uk_hv_mv_percent": 18.6,
    "uk_mv_lv_percent": 10.0,
    "uk_lv_hv_percent": 15.1,
    "p_cu_hv_mv_kw": 781.0,
    "p_cu_mv_lv_kw": 172.6,
    "p_cu_lv_hv_kw": 175.1,
    "pfe_kw": 91.9,
    "i0_percent": 0.036,
    "shift_mv_degree": 0.0,
    "shift_lv_degree": 150.0,
    "tap_side": "hv",
    "tap_neutral": 0,
    "tap_min": -9,
    "tap_max": 9,
    "tap_pos": 0,
    "tap_step_percent": 1.222222,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  3-WINDING TRANSFORMER IMPEDANCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _z_base(v_kv: float, s_mva: float) -> float:
    """Base impedance in ohms:  Z_base = V² / S."""
    return v_kv ** 2 / s_mva


def _i_base(s_mva: float, v_kv: float) -> float:
    """Base current in amperes:  I_base = S / (√3 · V)."""
    return s_mva * 1e6 / (np.sqrt(3) * v_kv * 1e3)


def _pairwise_z(
    uk_pct: float, pcu_kw: float, v_ref_kv: float, s_base_mva: float,
) -> complex:
    """Pairwise impedance R + jX referred to *v_ref_kv*."""
    z_abs = (uk_pct / 100.0) * _z_base(v_ref_kv, s_base_mva)
    i = _i_base(s_base_mva, v_ref_kv)
    r = (pcu_kw * 1e3) / (3.0 * i ** 2)
    x = np.sqrt(max(z_abs ** 2 - r ** 2, 0.0))
    return complex(r, x)


def _refer_z(z: complex, v_from_kv: float, v_to_kv: float) -> complex:
    """Refer impedance *z* from *v_from_kv* to *v_to_kv*."""
    return z * (v_to_kv / v_from_kv) ** 2


def _pairwise_to_star(
    z_hm: complex, z_ml: complex, z_lh: complex,
) -> Tuple[complex, complex, complex]:
    """Convert pairwise impedances to star (Y) equivalents."""
    z_h = 0.5 * (z_hm + z_lh - z_ml)
    z_m = 0.5 * (z_hm + z_ml - z_lh)
    z_l = 0.5 * (z_ml + z_lh - z_hm)
    return z_h, z_m, z_l


# ═══════════════════════════════════════════════════════════════════════════════
#  BUS, LINE, AND ELEMENT CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def _create_buses(net: pp.pandapowerNet) -> None:
    """Create 7 EHV buses (380 kV) and 10 HV buses (110 kV)."""
    for i in range(7):
        pp.create_bus(net, name=f"TN|Bus_{i}", vn_kv=380.0, type="b", subnet="TN")
    for i in range(10):
        pp.create_bus(net, name=f"DN|Bus_{i}", vn_kv=110.0, type="b", subnet="DN")


def _create_lines(net: pp.pandapowerNet) -> None:
    """Create EHV (380 kV) and HV (110 kV) lines."""
    # EHV lines
    for (f, t), length_km in zip(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (1, 6)],
        [50, 30, 50, 80, 30, 100, 100],
    ):
        pp.create_line(
            net,
            from_bus=pp.get_element_index(net, "bus", f"TN|Bus_{f}"),
            to_bus=pp.get_element_index(net, "bus", f"TN|Bus_{t}"),
            length_km=length_km,
            std_type="490-AL1/64-ST1A 380.0",
            name=f"TN|Line_(TN|{f}-TN|{t})",
            subnet="TN",
        )

    # HV lines
    for (f, t), length_km in zip(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
         (2, 6), (6, 7), (7, 8), (8, 9), (6, 9)],
        [15, 25, 20, 30, 40, 30, 20, 15, 10, 20, 15],
    ):
        pp.create_line(
            net,
            from_bus=pp.get_element_index(net, "bus", f"DN|Bus_{f}"),
            to_bus=pp.get_element_index(net, "bus", f"DN|Bus_{t}"),
            length_km=length_km,
            std_type="184-AL1/30-ST1A 110.0",
            name=f"DN|Line_(DN|{f}-DN|{t})",
            subnet="DN",
        )


def _create_static_generators(net: pp.pandapowerNet) -> None:
    """Create DER static generators (wind, PV) in both TSO and DSO grids."""
    # DSO wind parks
    for i, (bus_no, p_mw, profile) in enumerate(
        [(4, 60.0, "WP7"), (5, 130.0, "WP10"), (6, 110.0, "WP7"), (9, 110.0, "WP10")]
    ):
        pp.create_sgen(
            net,
            bus=pp.get_element_index(net, "bus", f"DN|Bus_{bus_no}"),
            p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw, type="WP",
            name=f"DN|Wind_{i}", profile=profile, subnet="DN",
            op_diagram="VDE-AR-N-4120-v2",
        )

    # DSO PV plants (cos φ = 0.98 inductive as initial setpoint)
    for i, (bus_no, p_mw) in enumerate([(3, 100.0), (4, 60.0), (5, 40.0), (7, 30.0)]):
        q_mvar = -p_mw * np.tan(np.arccos(0.98))
        pp.create_sgen(
            net,
            bus=pp.get_element_index(net, "bus", f"DN|Bus_{bus_no}"),
            p_mw=p_mw, q_mvar=q_mvar, sn_mva=p_mw, type="PV",
            name=f"DN|PV_{i}", profile="PV3", subnet="DN",
            op_diagram="VDE-AR-N-4120-v2",
        )

    # TSO large-scale PV
    pp.create_sgen(
        net,
        bus=pp.get_element_index(net, "bus", "TN|Bus_2"),
        p_mw=500.0, q_mvar=0.0, sn_mva=500.0, type="PV",
        name="TN|PV_1", profile="PV3", subnet="TN",
        op_diagram="VDE-AR-N-4120-v2",
    )

    # TSO large-scale wind farm
    pp.create_sgen(
        net,
        bus=pp.get_element_index(net, "bus", "TN|Bus_1"),
        p_mw=300.0, q_mvar=0.0, sn_mva=300.0, type="WP",
        name="TN|Wind_1", profile="WP10", subnet="TN",
        op_diagram="VDE-AR-N-4120-v2",
    )


def _create_loads(net: pp.pandapowerNet, load_scaling: float) -> None:
    """Create HV loads at every DN bus (scaled by *load_scaling*)."""
    for i in range(10):
        pp.create_load(
            net,
            bus=pp.get_element_index(net, "bus", f"DN|Bus_{i}"),
            sn_mva=50.0 * load_scaling,
            p_mw=50.0 * load_scaling,
            q_mvar=5.0 * load_scaling,
            name=f"HV/MV_Substation_{i}",
            subnet="DN",
            profile_p="mv_rural_pload",
            profile_q="mv_rural_qload",
        )


def _create_external_grid(net: pp.pandapowerNet, vm_pu: float) -> None:
    """Create the slack bus (external grid) at TN|Bus_6."""
    pp.create_ext_grid(
        net,
        bus=pp.get_element_index(net, "bus", "TN|Bus_6"),
        vm_pu=vm_pu, va_degree=0.0,
        name="External grid",
        s_sc_max_mva=10000.0, rx_max=0.1, rx_min=0.1,
        subnet="TN",
    )


def _create_conventional_generator(net: pp.pandapowerNet) -> None:
    """Create a single conventional generator (PV node) at TN|Bus_4."""
    pp.create_gen(
        net,
        bus=pp.get_element_index(net, "bus", "TN|Bus_4"),
        sn_mva=600.0, p_mw=500.0, vm_pu=1.05,
        name="TN_PP_0", subnet="TN",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  3-WINDING COUPLER TRANSFORMER CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def _create_single_coupler(
    net: pp.pandapowerNet,
    hv_bus: int,
    mv_bus: int,
    label: str,
) -> Tuple[int, int, List[int]]:
    """Create one 3W coupler with a 20 kV tertiary bus and a shunt reactor.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network to modify in place.
    hv_bus : int
        380 kV bus index.
    mv_bus : int
        110 kV bus index.
    label : str
        Human-readable label, e.g. ``"TN0_DN3"``.

    Returns
    -------
    trafo3w_idx : int
        Index of the created 3-winding transformer.
    lv_bus : int
        Index of the created 20 kV tertiary bus.
    shunt_indices : list[int]
        Indices of created shunt element(s) at the tertiary bus.
    """
    # Tertiary (20 kV) bus
    lv_bus = pp.create_bus(
        net, vn_kv=_TR3W["vn_lv_kv"],
        name=f"Tertiary|{label}|20kV", type="b", subnet="TERTIARY",
    )

    # --- Impedance calculation (pairwise → star, referred to each winding) ---
    s_hm = min(_TR3W["sn_hv_mva"], _TR3W["sn_mv_mva"])
    s_ml = min(_TR3W["sn_mv_mva"], _TR3W["sn_lv_mva"])
    s_lh = min(_TR3W["sn_lv_mva"], _TR3W["sn_hv_mva"])

    z_hm_hv = _pairwise_z(
        _TR3W["uk_hv_mv_percent"], _TR3W["p_cu_hv_mv_kw"],
        _TR3W["vn_hv_kv"], s_hm,
    )
    z_lh_hv = _pairwise_z(
        _TR3W["uk_lv_hv_percent"], _TR3W["p_cu_lv_hv_kw"],
        _TR3W["vn_hv_kv"], s_lh,
    )
    z_ml_mv = _pairwise_z(
        _TR3W["uk_mv_lv_percent"], _TR3W["p_cu_mv_lv_kw"],
        _TR3W["vn_mv_kv"], s_ml,
    )
    z_ml_hv = _refer_z(z_ml_mv, _TR3W["vn_mv_kv"], _TR3W["vn_hv_kv"])

    z_h, z_m, z_l = _pairwise_to_star(z_hm_hv, z_ml_hv, z_lh_hv)

    # Convert star impedances to per-winding vk/vkr parameters
    zb_hv = _z_base(_TR3W["vn_hv_kv"], _TR3W["sn_hv_mva"])
    vk_hv = abs(z_h) / zb_hv * 100.0
    vkr_hv = z_h.real / zb_hv * 100.0

    z_m_mv = _refer_z(z_m, _TR3W["vn_hv_kv"], _TR3W["vn_mv_kv"])
    zb_mv = _z_base(_TR3W["vn_mv_kv"], _TR3W["sn_mv_mva"])
    vk_mv = abs(z_m_mv) / zb_mv * 100.0
    vkr_mv = z_m_mv.real / zb_mv * 100.0

    z_l_lv = _refer_z(z_l, _TR3W["vn_hv_kv"], _TR3W["vn_lv_kv"])
    zb_lv = _z_base(_TR3W["vn_lv_kv"], _TR3W["sn_lv_mva"])
    vk_lv = abs(z_l_lv) / zb_lv * 100.0
    vkr_lv = z_l_lv.real / zb_lv * 100.0

    # Create the 3-winding transformer element
    trafo3w_idx = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=hv_bus, mv_bus=mv_bus, lv_bus=lv_bus,
        sn_hv_mva=_TR3W["sn_hv_mva"],
        sn_mv_mva=_TR3W["sn_mv_mva"],
        sn_lv_mva=_TR3W["sn_lv_mva"],
        vn_hv_kv=_TR3W["vn_hv_kv"],
        vn_mv_kv=_TR3W["vn_mv_kv"],
        vn_lv_kv=_TR3W["vn_lv_kv"],
        vk_hv_percent=vk_hv, vk_mv_percent=vk_mv, vk_lv_percent=vk_lv,
        vkr_hv_percent=vkr_hv, vkr_mv_percent=vkr_mv, vkr_lv_percent=vkr_lv,
        pfe_kw=_TR3W["pfe_kw"], i0_percent=_TR3W["i0_percent"],
        shift_mv_degree=_TR3W["shift_mv_degree"],
        shift_lv_degree=_TR3W["shift_lv_degree"],
        tap_side=_TR3W["tap_side"], tap_neutral=_TR3W["tap_neutral"],
        tap_min=_TR3W["tap_min"], tap_max=_TR3W["tap_max"],
        tap_pos=_TR3W["tap_pos"], tap_step_percent=_TR3W["tap_step_percent"],
        tap_changer_type="Ratio",
        name=f"Coupler3W|{label}",
    )

    # Shunt reactor at tertiary winding (50 Mvar inductive, initially switched off)
    reactor_idx = pp.create_shunt(
        net, bus=lv_bus,
        q_mvar=50.0, p_mw=0.0, vn_kv=_TR3W["vn_lv_kv"],
        step=0, max_step=1,
        name=f"Reactor|Tertiary|{label}",
        in_service=True, type="reactor", subnet='DN'
    )

    return trafo3w_idx, lv_bus, [reactor_idx]


def _create_all_couplers(net: pp.pandapowerNet) -> dict:
    """Replace the temporary 2W placeholder transformers with 3W couplers.

    Returns a dict whose keys match the ``NetworkMetadata`` field names.
    """
    trafo3w_indices: List[int] = []
    hv_buses: List[int] = []
    mv_buses: List[int] = []
    lv_buses: List[int] = []
    shunt_indices: List[int] = []

    for ehv_no, hv_no in _COUPLER_MAP:
        hv = pp.get_element_index(net, "bus", f"TN|Bus_{ehv_no}")
        mv = pp.get_element_index(net, "bus", f"DN|Bus_{hv_no}")
        tidx, lv, sh = _create_single_coupler(net, hv, mv, f"TN{ehv_no}_DN{hv_no}")
        trafo3w_indices.append(tidx)
        hv_buses.append(hv)
        mv_buses.append(mv)
        lv_buses.append(lv)
        shunt_indices.extend(sh)

    return dict(
        coupler_trafo3w_indices=trafo3w_indices,
        coupler_hv_buses=hv_buses,
        coupler_mv_buses=mv_buses,
        coupler_lv_buses=lv_buses,
        tertiary_shunt_indices=shunt_indices,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MACHINE TRANSFORMERS
# ═══════════════════════════════════════════════════════════════════════════════

def _add_machine_transformers(
    net: pp.pandapowerNet,
    gen_terminal_kv: float = 15.0,
    vk_percent: float = 12.0,
    vkr_percent: float = 0.3,
) -> List[int]:
    """Add a step-up transformer for every conventional generator.

    A new bus at *gen_terminal_kv* is created, the generator is moved
    there, and a 2-winding transformer connects the terminal bus to
    the original grid bus.

    Returns
    -------
    list[int]
        Indices of the created machine transformers in ``net.trafo``.
    """
    if net.gen.empty:
        return []

    created: List[int] = []
    for gen_idx in list(net.gen.index):
        grid_bus = int(net.gen.at[gen_idx, "bus"])
        grid_vn = float(net.bus.at[grid_bus, "vn_kv"])
        sn = float(net.gen.at[gen_idx, "sn_mva"])
        gen_name = str(net.gen.at[gen_idx, "name"])

        # Generator-terminal bus
        term_bus = pp.create_bus(
            net, vn_kv=gen_terminal_kv,
            name=f"GEN_TERM|{gen_name}", type="b", subnet="GEN_TERM",
        )
        net.gen.at[gen_idx, "bus"] = term_bus

        tidx = pp.create_transformer_from_parameters(
            net,
            hv_bus=grid_bus, lv_bus=term_bus,
            sn_mva=sn, vn_hv_kv=grid_vn, vn_lv_kv=gen_terminal_kv,
            vk_percent=vk_percent, vkr_percent=vkr_percent,
            pfe_kw=0.0, i0_percent=0.0,
            tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
            tap_pos=0, tap_step_percent=1.25, shift_degree=0.0,
            tap_changer_type="Ratio",
            name=f"MachineTrf|{gen_name}",
        )
        created.append(tidx)

    return created


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSMISSION-LEVEL SHUNT COMPENSATION
# ═══════════════════════════════════════════════════════════════════════════════

def _add_tn_shunt(
    net: pp.pandapowerNet,
    ehv_bus_no: int = 4,
    q_mvar: float = -150.0,
) -> int:
    """Add a switchable shunt at an EHV bus (negative Q → capacitive)."""
    bus = pp.get_element_index(net, "bus", f"TN|Bus_{ehv_bus_no}")
    return pp.create_shunt(
        net, bus=bus,
        q_mvar=q_mvar, p_mw=0.0,
        vn_kv=float(net.bus.at[bus, "vn_kv"]),
        step=0, max_step=1,
        name=f"TN_Shunt@EHV{ehv_bus_no}",
        in_service=True,
        type="capacitor" if q_mvar < 0 else "reactor", subnet='TN'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def build_tuda_net(
    *,
    pv_nodes: bool = True,
    ext_grid_vm_pu: float = 1.05,
    load_scaling: float = 1.0,
) -> Tuple[pp.pandapowerNet, NetworkMetadata]:
    """Build the combined 380/110/20 kV TU Darmstadt benchmark network.

    The function creates the full topology including 3-winding coupler
    transformers, machine transformers, and shunt compensation, then
    runs an initial power flow to obtain a converged operating point.

    Parameters
    ----------
    pv_nodes : bool
        If ``True`` (default), create conventional generators as PV nodes.
    ext_grid_vm_pu : float
        Voltage setpoint of the external grid / slack bus [p.u.].
    load_scaling : float
        Multiplicative scaling factor applied to every load.

    Returns
    -------
    net : pp.pandapowerNet
        Converged combined network.
    meta : NetworkMetadata
        Immutable record of all created element indices.

    Raises
    ------
    pandapower.powerflow.LoadflowNotConverged
        If the initial power flow does not converge.
    """
    net = pp.create_empty_network()

    # --- Base topology (buses, lines, external grid) ---
    _create_buses(net)
    _create_lines(net)
    _create_external_grid(net, ext_grid_vm_pu)

    # --- Generators and loads ---
    _create_static_generators(net)
    if pv_nodes:
        _create_conventional_generator(net)
    _create_loads(net, load_scaling)

    # --- Temporary 2W placeholder transformers (needed for base power flow) ---
    #     These are immediately replaced by 3W couplers below.
    _placeholder_2w_indices: List[int] = []
    for ehv_no, hv_no in _COUPLER_MAP:
        tidx = pp.create_transformer_from_parameters(
            net,
            hv_bus=pp.get_element_index(net, "bus", f"TN|Bus_{ehv_no}"),
            lv_bus=pp.get_element_index(net, "bus", f"DN|Bus_{hv_no}"),
            sn_mva=300.0, vn_hv_kv=380.0, vn_lv_kv=110.0,
            vkr_percent=0.25, vk_percent=15.0, pfe_kw=80.0, i0_percent=0.04,
            tap_pos=0, shift_degree=0.0, tap_step_percent=1.25,
            tap_min=-13, tap_max=13, tap_side="hv", tap_neutral=0,
            tap_changer_type="Ratio",
            name=f"TMP_2W_placeholder",
        )
        _placeholder_2w_indices.append(tidx)

    # --- Replace 2W placeholders with 3W coupler transformers ---
    coupler_data = _create_all_couplers(net)
    net.trafo.drop(index=_placeholder_2w_indices, inplace=True)

    # --- Machine transformers for conventional generators ---
    mt_indices = _add_machine_transformers(net) if pv_nodes else []

    # --- Transmission-level shunt compensation ---
    tn_shunt_idx = _add_tn_shunt(net, ehv_bus_no=4, q_mvar=-150.0)

    # --- Assemble metadata ---
    meta = NetworkMetadata(
        **coupler_data,
        machine_trafo_indices=mt_indices,
        tn_shunt_indices=[tn_shunt_idx],
    )

    # --- Initial power flow (voltage angles required due to 3W phase shift) ---
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    return net, meta
