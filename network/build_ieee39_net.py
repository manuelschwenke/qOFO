#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network/build_ieee39_net.py
===========================
Build the IEEE 39-bus New England test network and optionally attach synthetic
Distribution System Operator (DSO) feeders at selected Zone-2 load buses.

Network topology
----------------
The standard IEEE 39-bus (New England) system is a 345 kV transmission network
with 39 buses, 10 generators (1 slack, 9 PV), 19 loads, and ~46 branches.

All buses are tagged with ``subnet = "TN"`` (Transmission Network).

When DSO feeders are requested (``dso_load_buses`` parameter), this function:
  1. Adds a 20 kV distribution bus for each selected load bus.
  2. Connects it via a 345/20 kV 2-winding transformer (the "PCC trafo").
  3. Moves the original load to the 20 kV side.
  4. Adds ``n_der_per_feeder`` DER sgens at the 20 kV bus.
  5. Tags the new bus with ``subnet = "DN"`` (Distribution Network).

This mimics the structure of the TU-Darmstadt cascade network but re-uses the
publicly available IEEE 39-bus test case, which makes the multi-zone scenario
easy to reproduce and compare with the literature.

DER sgens added at generator buses
-----------------------------------
To give each TSO zone controllable Q actuators, a DER sgen is added co-located
with each PV generator.  Its rated MVA is set to 20 % of the generator's rated
MVA.  The control variable is Q_DER (continuous); the active power is fixed.

Sensitivity semantics
---------------------
* DER Q columns (∂V/∂Q_DER): computed via ``compute_dV_dQ_der`` with generator
  buses as "DER buses".  Because PV generators fix their terminal voltage, the
  Q injection from the sgen shifts reactive flows in the network and thus
  changes voltages at remote PQ buses.
* Generator AVR columns (∂V/∂V_gen): computed via ``compute_dV_dVgen_matrix``,
  which perturbs the PV-bus voltage setpoint.  This is the dominant long-range
  control action of TSOs.

Usage
-----
    from network.build_ieee39_net import build_ieee39_net, add_dso_feeders

    # Step 1: base network (no DSO feeders)
    net, meta = build_ieee39_net()

    # Step 2: (after zone partition) attach feeders to Zone-2 load buses
    meta = add_dso_feeders(net, meta, dso_load_buses=[bus_a, bus_b])

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandapower as pp
import pandapower.networks as pn


# ---------------------------------------------------------------------------
#  Network metadata
# ---------------------------------------------------------------------------

@dataclass
class IEEE39NetworkMeta:
    """
    Immutable index catalogue for the IEEE 39-bus network.

    All index lists refer to pandapower element indices (integer row labels
    in the respective DataFrame, e.g. net.sgen.index).

    The separation into TN (345 kV, "subnet = TN") and DN (20 kV,
    "subnet = DN") mirrors the convention used in the TU-Darmstadt benchmark
    (build_tuda_net.py) so that the rest of the code-base can be reused.
    """

    # ── Transmission-network (345 kV) ────────────────────────────────────────
    tn_bus_indices: Tuple[int, ...]
    """All 345 kV bus indices (the 39 original buses)."""

    tn_line_indices: Tuple[int, ...]
    """All line indices (between 345 kV buses)."""

    # ── Generators ───────────────────────────────────────────────────────────
    gen_indices: Tuple[int, ...]
    """pandapower ``net.gen`` indices for the 9 PV generators.
    (The slack generator is in ``net.ext_grid`` and is NOT controllable by OFO.)
    """

    gen_bus_indices: Tuple[int, ...]
    """Bus index of each generator (terminal bus = 345 kV bus directly)."""

    # ── TSO-level DER sgens ───────────────────────────────────────────────────
    tso_der_indices: Tuple[int, ...]
    """Indices of sgen elements co-located at generator buses.
    These represent the TSO's controllable reactive-power actuators.
    """

    tso_der_buses: Tuple[int, ...]
    """Bus indices of the TSO DER sgens (same order as tso_der_indices)."""

    # ── DSO feeders (populated by add_dso_feeders) ────────────────────────────
    dso_pcc_trafo_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Indices of the 2-winding PCC trafos connecting each DSO feeder (345/20 kV)."""

    dso_pcc_hv_buses: Tuple[int, ...] = field(default_factory=tuple)
    """345 kV buses at which DSO feeders are attached."""

    dso_lv_buses: Tuple[int, ...] = field(default_factory=tuple)
    """20 kV distribution buses (one per DSO feeder)."""

    dso_der_indices: Tuple[int, ...] = field(default_factory=tuple)
    """sgen indices for DERs inside the DSO feeders (20 kV level)."""

    dso_der_buses: Tuple[int, ...] = field(default_factory=tuple)
    """Bus indices of DSO DERs (same order as dso_der_indices)."""

    dso_shunt_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Shunt indices inside DSO feeders (one switchable shunt per feeder)."""

    dso_shunt_buses: Tuple[int, ...] = field(default_factory=tuple)
    """Bus indices of DSO shunts."""

    dn_bus_indices: Tuple[int, ...] = field(default_factory=tuple)
    """All 20 kV distribution bus indices."""

    dn_line_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Line indices within distribution networks (empty for simple feeders)."""

    hv_networks: Tuple = field(default_factory=tuple)
    """HVNetworkInfo objects for attached 110 kV sub-networks (see add_hv_networks)."""


# ---------------------------------------------------------------------------
#  Main build function
# ---------------------------------------------------------------------------

def build_ieee39_net(
    *,
    ext_grid_vm_pu: float = 1.04,
    der_mva_fraction: float = 0.20,
    der_p_mw_fraction: float = 0.30,
    add_der_at_gen_buses: bool = True,
) -> Tuple[pp.pandapowerNet, IEEE39NetworkMeta]:
    """
    Build the IEEE 39-bus New England test network with TSO DER sgens.

    Parameters
    ----------
    ext_grid_vm_pu : float
        Voltage setpoint of the slack external grid [p.u.].
    der_mva_fraction : float
        TSO DER rated MVA as a fraction of the co-located generator's rated MVA.
        E.g. 0.20 → a 600 MVA generator gets a 120 MVA DER sgen.
    der_p_mw_fraction : float
        Active power of each TSO DER as a fraction of its rated MVA [p.u.].
        Q is initialised to zero; P is kept fixed during the simulation.
    add_der_at_gen_buses : bool
        If True (default), add one DER sgen at each generator bus.
        These are the TSO's primary reactive-power actuators in multi-zone OFO.

    Returns
    -------
    net : pp.pandapowerNet
        Fully initialised pandapower network ready for power flow.
    meta : IEEE39NetworkMeta
        Immutable index catalogue.  DSO feeder fields are empty tuples;
        call :func:`add_dso_feeders` to populate them.
    """
    # ── Load standard IEEE 39-bus case ────────────────────────────────────────
    # pandapower.networks.case39() returns the standard New England test system.
    # All buses are at 345 kV; generators at buses 30, 31, …, 38, 39 (1-indexed).
    net = pn.case39()
    net.gen["vm_pu"] = 1.05

    # Adjust slack voltage
    net.ext_grid.at[net.ext_grid.index[0], "vm_pu"] = ext_grid_vm_pu

    # ── Metadata: label all original buses as TN (transmission network) ───────
    #
    # We add a "subnet" column matching the convention in build_tuda_net.py so
    # that helper functions that filter by subnet work unchanged.
    net.bus["subnet"] = "TN"
    net.bus["vn_kv"]  # already set by case39()

    # ── Tag lines with subnet for later filtering ─────────────────────────────
    # case39 has no trafo3w; all connections between TN buses are net.line or
    # net.trafo (there are a few step-up trafos in some versions).
    for li in net.line.index:
        net.line.at[li, "subnet"] = "TN"

    # ── Generators ────────────────────────────────────────────────────────────
    # case39() puts the slack at ext_grid (bus 38 in 0-indexed = bus 39 in
    # 1-indexed).  All remaining generators are in net.gen as PV buses.
    # We collect their indices and bus indices.
    gen_indices: List[int] = sorted(int(g) for g in net.gen.index)
    gen_bus_indices: List[int] = [int(net.gen.at[g, "bus"]) for g in gen_indices]

    # ── TN buses and lines ────────────────────────────────────────────────────
    tn_buses: List[int] = sorted(int(b) for b in net.bus.index
                                  if str(net.bus.at[b, "subnet"]) == "TN")
    tn_lines: List[int] = sorted(int(li) for li in net.line.index
                                  if str(net.line.at[li, "subnet"]) == "TN")

    # ── Add TSO DER sgens at PQ load buses ───────────────────────────────────
    #
    # TSO DERs must sit at PQ (load) buses so that the Q sensitivity
    # ∂V/∂Q_DER can be computed via the reduced Jacobian.  Generator buses
    # are PV buses — their voltage is fixed by the AVR and ∂V/∂Q_DER = 0
    # in the linearised model (the generator absorbs the injected Q).
    #
    # Strategy: place one DER sgen at each unique PQ bus that carries a load.
    # This gives the OFO direct Q-injection controllability at load centres,
    # which is the primary use-case for distributed reactive-power resources
    # in a transmission network (static VAR compensators, STATCOMs, etc.).
    #
    # The generator AVR setpoints (V_gen) are separate control variables
    # handled by the "gen_indices" pathway in TSOController.
    tso_der_indices: List[int] = []
    tso_der_buses: List[int]  = []

    gen_bus_set = set(gen_bus_indices)
    # ext_grid bus is also PV; exclude it
    ext_grid_buses = set(int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index)
    pv_and_slack_buses = gen_bus_set | ext_grid_buses

    if add_der_at_gen_buses:
        # Collect unique PQ load buses (not PV or slack)
        load_buses_pq: List[int] = sorted(
            {int(net.load.at[li, "bus"]) for li in net.load.index
             if int(net.load.at[li, "bus"]) not in pv_and_slack_buses}
        )
        for bus in load_buses_pq:
            # Default DER rating: 50 MVA (representative for a large transmission
            # network shunt compensator).  Scale with p_mw fraction for a small
            # active component (solar / BESS).
            sn_mva = 150.0
            p_mw   = sn_mva * der_p_mw_fraction
            idx = pp.create_sgen(
                net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=0.0,
                sn_mva=sn_mva,
                name=f"TN_DER|load_bus{bus}",
            )
            tso_der_indices.append(int(idx))
            tso_der_buses.append(bus)

    # ── Run initial power flow ────────────────────────────────────────────────
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    meta = IEEE39NetworkMeta(
        tn_bus_indices   = tuple(tn_buses),
        tn_line_indices  = tuple(tn_lines),
        gen_indices      = tuple(gen_indices),
        gen_bus_indices  = tuple(gen_bus_indices),
        tso_der_indices  = tuple(tso_der_indices),
        tso_der_buses    = tuple(tso_der_buses),
        # DSO fields are empty until add_dso_feeders() is called
    )
    return net, meta


# ---------------------------------------------------------------------------
#  DSO feeder attachment
# ---------------------------------------------------------------------------

def add_dso_feeders(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    dso_load_buses: List[int],
    *,
    n_der_per_feeder: int = 3,
    der_s_mva: float = 50.0,
    der_p_mw: float = 15.0,
    shunt_q_mvar: float = 30.0,
    mv_kv: float = 20.0,
    trafo_sn_mva: float = 400.0,
) -> IEEE39NetworkMeta:
    """
    Attach synthetic DSO feeders at the specified Zone-2 load buses.

    For each bus in ``dso_load_buses`` this function:

    1. Adds a new MV bus at ``mv_kv`` (default 20 kV) tagged ``subnet = "DN"``.
    2. Adds a 2-winding PCC transformer (345 kV / mv_kv).
       * HV side  = original load bus (345 kV).
       * LV side  = the new 20 kV bus.
    3. Moves the original load to the LV bus (so the 345 kV bus becomes lightly
       loaded — the TSO sees the DSO as a PQ load through the PCC trafo).
    4. Adds ``n_der_per_feeder`` DER sgens at the LV bus (Q = 0 initially).
    5. Adds one switchable shunt at the LV bus for reactive-power reserve.

    The resulting structure is:

        TN bus X (345 kV)
            │  PCC trafo  (2W, 345/20 kV)
        DN bus Y (20 kV)
            ├── Load (original P + Q)
            ├── DER sgen 0   (DSO Q control)
            ├── DER sgen 1
            └── Shunt        (optional switchable Q reserve)

    Why 2-winding instead of 3-winding?
    ------------------------------------
    The TU-Darmstadt benchmark uses 3-winding (380/110/20 kV) couplers because
    it models an intermediate 110 kV distribution level.  The IEEE 39-bus has
    only 345 kV; a 2-winding trafo is the natural single-step interface.
    TSOController already handles ``pcc_trafo_indices`` for both 2W and 3W
    transformers (detected automatically via ``net.trafo`` vs ``net.trafo3w``).

    Parameters
    ----------
    net : pp.pandapowerNet
        The network returned by ``build_ieee39_net()`` (modified in-place).
    meta : IEEE39NetworkMeta
        Existing metadata (will be replaced with an updated copy).
    dso_load_buses : List[int]
        Pandapower bus indices at which feeders should be attached.
        These should be load buses inside Zone 2 (determined by zone_partition).
    n_der_per_feeder : int
        Number of DER sgens per feeder (default 3).
    der_s_mva : float
        Rated MVA of each DSO DER sgen.
    der_p_mw : float
        Fixed active power of each DSO DER [MW] (Q controlled by DSO OFO).
    shunt_q_mvar : float
        Reactive power step of the switchable shunt [Mvar].  Positive = capacitive.
    mv_kv : float
        MV voltage level [kV] of the new distribution bus.
    trafo_sn_mva : float
        Rated MVA of each 2-winding PCC transformer.

    Returns
    -------
    meta : IEEE39NetworkMeta
        Updated metadata with all DSO feeder index lists populated.
    """
    # Accumulators for the new DSO elements
    pcc_trafo_indices: List[int]  = []
    pcc_hv_buses:     List[int]  = []
    lv_buses:         List[int]  = []
    dso_der_idx:      List[int]  = []
    dso_der_bus:      List[int]  = []
    dso_shunt_idx:    List[int]  = []
    dso_shunt_bus:    List[int]  = []

    for hv_bus in dso_load_buses:
        # ── 1. Add 20 kV distribution bus ────────────────────────────────────
        lv_bus = pp.create_bus(
            net,
            vn_kv=mv_kv,
            name=f"DN|Bus_HV{hv_bus}",
        )
        net.bus.at[lv_bus, "subnet"] = "DN"

        # ── 2. Add 2-winding PCC transformer ─────────────────────────────────
        #
        # Standard parameters for a 345/20 kV power transformer.
        # vk_percent = 10 % short-circuit impedance is typical for large units.
        # vkr_percent = 0.5 % resistance (low loss for transmission-level trafo).
        # No OLTC taps by default; a DiscreteTapControl can be added later.
        trafo_idx = pp.create_transformer_from_parameters(
            net,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            sn_mva=trafo_sn_mva,
            vn_hv_kv=float(net.bus.at[hv_bus, "vn_kv"]),
            vn_lv_kv=mv_kv,
            vkr_percent=0.5,
            vk_percent=10.0,
            pfe_kw=0.0,
            i0_percent=0.0,
            name=f"DSO_PCC|HV{hv_bus}",
            tap_neutral=0,
            tap_min=-5,
            tap_max=5,
            tap_step_percent=2.0,
            tap_pos=0,
        )
        pcc_trafo_indices.append(int(trafo_idx))
        pcc_hv_buses.append(int(hv_bus))
        lv_buses.append(int(lv_bus))

        # ── 3. Move existing load to LV bus ──────────────────────────────────
        #
        # Loads at the HV bus are transferred to the LV bus so the 345 kV bus
        # becomes the TSO-side PCC (lightly loaded; load is "behind the trafo").
        load_mask = net.load["bus"] == hv_bus
        for load_idx in net.load.index[load_mask]:
            net.load.at[load_idx, "bus"] = lv_bus

        # ── 4. Add DSO DER sgens at the LV bus ───────────────────────────────
        #
        # These are the DSO's controllable reactive-power actuators.
        # DSOController will dispatch Q_DER to track the TSO's Q_PCC setpoint.
        for k in range(n_der_per_feeder):
            sgen_idx = pp.create_sgen(
                net,
                bus=lv_bus,
                p_mw=der_p_mw,
                q_mvar=0.0,
                sn_mva=der_s_mva,
                name=f"DN_DER|HV{hv_bus}_k{k}",
            )
            dso_der_idx.append(int(sgen_idx))
            dso_der_bus.append(int(lv_bus))

        # ── 5. Add switchable shunt ───────────────────────────────────────────
        #
        # One shunt per feeder acts as a "last resort" reactive-power reserve.
        # In the current implementation the shunt is monitored by the DSO
        # controller but not actively switched (shunt_bus_indices=[] in config).
        # Enable it by passing the shunt index to DSOControllerConfig.
        shunt_idx = pp.create_shunt(
            net,
            bus=lv_bus,
            q_mvar=-shunt_q_mvar,   # negative = capacitive (generates Q)
            p_mw=0.0,
            name=f"DN_Shunt|HV{hv_bus}",
            step=0,                  # initially off
            max_step=1,
        )
        dso_shunt_idx.append(int(shunt_idx))
        dso_shunt_bus.append(int(lv_bus))

    # ── Run power flow to converge with DSO feeders ───────────────────────────
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # ── Assemble updated metadata ─────────────────────────────────────────────
    dn_buses = sorted(
        int(b) for b in net.bus.index if str(net.bus.at[b, "subnet"]) == "DN"
    )

    return IEEE39NetworkMeta(
        # Carry over TN fields unchanged
        tn_bus_indices   = meta.tn_bus_indices,
        tn_line_indices  = meta.tn_line_indices,
        gen_indices      = meta.gen_indices,
        gen_bus_indices  = meta.gen_bus_indices,
        tso_der_indices  = meta.tso_der_indices,
        tso_der_buses    = meta.tso_der_buses,
        # New DSO fields
        dso_pcc_trafo_indices = tuple(pcc_trafo_indices),
        dso_pcc_hv_buses      = tuple(pcc_hv_buses),
        dso_lv_buses          = tuple(lv_buses),
        dso_der_indices       = tuple(dso_der_idx),
        dso_der_buses         = tuple(dso_der_bus),
        dso_shunt_indices     = tuple(dso_shunt_idx),
        dso_shunt_buses       = tuple(dso_shunt_bus),
        dn_bus_indices        = tuple(dn_buses),
        dn_line_indices       = (),   # no 20 kV lines in radial feeders
    )


# ---------------------------------------------------------------------------
#  110 kV HV sub-network attachment (TUDA topology copies)
# ---------------------------------------------------------------------------

@dataclass
class HVNetworkInfo:
    """Tracking information for one attached 110 kV sub-network.

    Each HV sub-network replicates the TUDA 110 kV distribution topology
    (10 buses, 11 lines) and is coupled to the IEEE 39-bus 345 kV
    transmission network via 2-winding 345/110 kV transformers at three
    coupling points.
    """
    net_id: str
    """Unique sub-network identifier (e.g. ``"DSO_1"``)."""

    bus_indices: Tuple[int, ...]
    """Pandapower bus indices of the 10 HV (110 kV) buses."""

    line_indices: Tuple[int, ...]
    """Pandapower line indices within this sub-network."""

    sgen_indices: Tuple[int, ...]
    """Sgen indices for DER plants (PV / wind) in this sub-network."""

    load_indices: Tuple[int, ...]
    """Load indices (HV/MV substations, 10 loads)."""

    coupling_trafo_indices: Tuple[int, ...]
    """2-winding transformer indices coupling 345 kV TN to 110 kV HV."""

    coupling_ieee_buses: Tuple[int, ...]
    """IEEE 39-bus TN bus (0-indexed) at the HV side of each coupling trafo."""

    coupling_hv_bus_indices: Tuple[int, ...]
    """HV bus index at the LV side of each coupling transformer."""

    zone: int = 0
    """IEEE zone this sub-network belongs to."""

    line_length_scale: float = 1.0
    """Scale factor applied to all line lengths relative to TUDA base."""

    total_ref_p_mw: float = 0.0
    """Total reference active power (sum of replaced IEEE loads) [MW]."""

    total_ref_q_mvar: float = 0.0
    """Total reference reactive power [Mvar]."""

    gen_type: str = "mixed"
    """Generation type: 'mixed', 'pv', or 'wind'."""


# ── TUDA HV network topology (reused for all sub-networks) ───────────────

_HV_LINE_TOPOLOGY: List[Tuple[int, int, float]] = [
    # (from_bus_no, to_bus_no, length_km)  -- matches build_tuda_net._create_lines
    (0, 1, 15),  (1, 2, 25),  (2, 3, 20),  (3, 4, 30),
    (4, 5, 40),  (5, 6, 30),  (2, 6, 20),  (6, 7, 15),
    (7, 8, 10),  (8, 9, 20),  (6, 9, 15),
]

# TUDA DER data: (hv_bus_no, p_mw, profile_name)
_TUDA_WIND_PARKS: List[Tuple[int, float, str]] = [
    (4,  60.0, "WP7"),
    (5, 130.0, "WP10"),
    (6, 110.0, "WP7"),
    (9, 110.0, "WP10"),
]

# TUDA PV plants: (hv_bus_no, p_mw)  -- all use profile "PV3"
_TUDA_PV_PLANTS: List[Tuple[int, float]] = [
    (3, 100.0),
    (4,  60.0),
    (5,  40.0),
    (7,  30.0),
]

# Zone-3 buses for EHV profile assignment (0-indexed pandapower)
_ZONE3_BUSES_0IDX = set(range(14, 24)) | {32, 33, 34, 35}

# ── Sub-network configuration table ──────────────────────────────────────
#
# Each entry: (net_id, zone, ieee_buses_1idx, hv_coupling, line_scale, gen_type)
# ieee_buses_1idx are 1-indexed IEEE bus labels matching the picture.

_SUBNET_DEFS: List[dict] = [
    dict(net_id="DSO_1", zone=2,
         ieee_1idx=(7, 8, 5),    hv_buses=(3, 0, 8), scale=0.75, gen="mixed"),
    dict(net_id="DSO_2", zone=2,
         ieee_1idx=(14, 4, 3),   hv_buses=(3, 0, 8), scale=1.50, gen="mixed"),
    dict(net_id="DSO_3", zone=2,
         ieee_1idx=(11, 10, 13), hv_buses=(3, 0, 8), scale=0.75, gen="mixed"),
    dict(net_id="DSO_4", zone=3,
         ieee_1idx=(24, 21, 23), hv_buses=(3, 0, 8), scale=2.00, gen="pv"),
    dict(net_id="DSO_5", zone=1,
         ieee_1idx=(27, 26, 25), hv_buses=(3, 0, 8), scale=3.00, gen="wind"),
]

# Bus 12 (1-indexed = bus 11, 0-indexed) is removed for DSO_3.
# Its load is redistributed 50/50 to buses 11 and 13 (1-idx) = 10 and 12 (0-idx).
_BUS_TO_REMOVE_0IDX = 11   # IEEE Bus 12 (1-indexed)
_REDIST_BUSES_0IDX = (10, 12)  # buses 11 and 13 (1-indexed)


def _get_load_at_bus(net: pp.pandapowerNet, bus: int) -> Tuple[float, float]:
    """Return total (p_mw, q_mvar) at a bus, or (0, 0) if no load."""
    mask = net.load["bus"] == bus
    if not mask.any():
        return 0.0, 0.0
    return (
        float(net.load.loc[mask, "p_mw"].sum()),
        float(net.load.loc[mask, "q_mvar"].sum()),
    )


def _delete_loads_at_bus(net: pp.pandapowerNet, bus: int) -> None:
    """Remove all loads connected to a bus."""
    mask = net.load["bus"] == bus
    if mask.any():
        net.load.drop(index=net.load.index[mask], inplace=True)


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
    for f, t, base_km in _HV_LINE_TOPOLOGY:
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

    # ── 3. Create coupling transformers (2W, 345/110 kV, 300 MVA) ────────────
    coupling_trafo_indices: List[int] = []
    coupling_ieee_buses: List[int] = []
    coupling_hv_bus_indices: List[int] = []

    for ieee_bus, hv_no in coupling_map:
        hv_bus = bus_map[hv_no]
        tidx = pp.create_transformer_from_parameters(
            net,
            hv_bus=ieee_bus,
            lv_bus=hv_bus,
            sn_mva=300.0,
            vn_hv_kv=float(net.bus.at[ieee_bus, "vn_kv"]),
            vn_lv_kv=110.0,
            vkr_percent=0.3,
            vk_percent=12.0,
            pfe_kw=80.0,
            i0_percent=0.04,
            tap_side="hv",
            tap_neutral=0,
            tap_min=-9,
            tap_max=9,
            tap_pos=0,
            tap_step_percent=1.25,
            name=f"{net_id}|Coupler_TN{ieee_bus}_HV{hv_no}",
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
        for i, (bus_no, p_mw, profile) in enumerate(_TUDA_WIND_PARKS):
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="WP", profile=profile,
                name=f"{net_id}|Wind_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))
        for i, (bus_no, p_mw) in enumerate(_TUDA_PV_PLANTS):
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
        for i, (bus_no, p_mw) in enumerate(_TUDA_PV_PLANTS):
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
        for i, (bus_no, p_mw, _) in enumerate(_TUDA_WIND_PARKS):
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
        for i, (bus_no, p_mw, profile) in enumerate(_TUDA_WIND_PARKS):
            sidx = pp.create_sgen(
                net, bus=bus_map[bus_no],
                p_mw=p_mw, q_mvar=0.0, sn_mva=p_mw,
                type="WP", profile=profile,
                name=f"{net_id}|Wind_{i}",
                subnet="DN",
                op_diagram="VDE-AR-N-4120-v2",
            )
            sgen_indices.append(int(sidx))
        for bus_no, p_mw in _TUDA_PV_PLANTS:
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
        if bus in _ZONE3_BUSES_0IDX:
            net.load.at[li, "profile_p"] = "HS5_pload"
            net.load.at[li, "profile_q"] = "HS5_qload"
        else:
            net.load.at[li, "profile_p"] = "HS4_pload"
            net.load.at[li, "profile_q"] = "HS4_qload"


def _compute_reference_loads(
    net: pp.pandapowerNet,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute the reference (total_p_mw, total_q_mvar) for each sub-network
    by summing the IEEE loads at the coupling buses.

    For buses without a load, the average of the buses that DO have loads
    in the same 3-bus set is used.  DSO_3 is special: it uses the average
    of DSO_1 and DSO_2 totals (since its coupling buses have no IEEE load).

    Returns
    -------
    dict : net_id -> (total_p_mw, total_q_mvar)
    """
    ref: Dict[str, Tuple[float, float]] = {}

    for sdef in _SUBNET_DEFS:
        net_id = sdef["net_id"]
        ieee_0idx = [b - 1 for b in sdef["ieee_1idx"]]

        # Read loads at each coupling bus
        bus_loads = []
        for b in ieee_0idx:
            p, q = _get_load_at_bus(net, b)
            bus_loads.append((p, q, p != 0.0 or q != 0.0))

        # Buses with load
        loaded = [(p, q) for p, q, has in bus_loads if has]

        if loaded:
            avg_p = sum(p for p, q in loaded) / len(loaded)
            avg_q = sum(q for p, q in loaded) / len(loaded)
            total_p = sum(
                p if has else avg_p for p, q, has in bus_loads
            )
            total_q = sum(
                q if has else avg_q for p, q, has in bus_loads
            )
        else:
            # No loads at any coupling bus -- will be overridden for DSO_3
            total_p, total_q = 0.0, 0.0

        ref[net_id] = (total_p, total_q)

    # DSO_3 special case: use average of DSO_1 and DSO_2
    if "DSO_3" in ref and ref["DSO_3"] == (0.0, 0.0):
        p1, q1 = ref.get("DSO_1", (0.0, 0.0))
        p2, q2 = ref.get("DSO_2", (0.0, 0.0))
        ref["DSO_3"] = ((p1 + p2) / 2.0, (q1 + q2) / 2.0)

    return ref


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
            (s for s in _SUBNET_DEFS if s["net_id"] == hv.net_id), None
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
            tname = str(net.trafo.at[tidx, "name"])
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


def add_hv_networks(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    *,
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
    * **Bus 12 (1-idx = bus 11, 0-idx)** is removed entirely.  Its load
      (8.53 MW, 88 Mvar) is redistributed 50/50 to buses 11 and 13 (1-idx).
    * **DSO_3** coupling buses have no IEEE load after removal; its reference
      load is set to the average of DSO_1 and DSO_2 totals.
    * EHV profiles (HS4/HS5) are wired to all remaining IEEE 39-bus loads.

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
    # 2. Remove IEEE Bus 12 (1-idx = bus 11, 0-idx)
    # =====================================================================
    bus_rm = _BUS_TO_REMOVE_0IDX  # bus 11 (0-idx)
    p_rm, q_rm = _get_load_at_bus(net, bus_rm)

    if verbose:
        print(f"\n[add_hv_networks] Removing IEEE Bus 12 (0-idx {bus_rm}):")
        print(f"  Load: {p_rm:.2f} MW + j{q_rm:.1f} Mvar")

    # Redistribute load 50/50 to neighbouring buses
    for rb in _REDIST_BUSES_0IDX:
        mask = net.load["bus"] == rb
        if mask.any():
            for li in net.load.index[mask]:
                net.load.at[li, "p_mw"] += p_rm / 2.0
                net.load.at[li, "q_mvar"] += q_rm / 2.0
        else:
            # Create a small load if none exists
            pp.create_load(
                net, bus=rb,
                p_mw=p_rm / 2.0, q_mvar=q_rm / 2.0,
                sn_mva=max(abs(p_rm / 2.0), 1.0),
                name=f"Redist_from_bus{bus_rm}",
                subnet="TN",
            )
        if verbose:
            print(f"  +{p_rm/2:.2f} MW, +{q_rm/2:.1f} Mvar -> bus {rb}")

    # Delete loads at the removed bus
    _delete_loads_at_bus(net, bus_rm)

    # Delete lines connected to the removed bus
    lines_to_drop = net.line.index[
        (net.line["from_bus"] == bus_rm) | (net.line["to_bus"] == bus_rm)
    ]
    if verbose and len(lines_to_drop) > 0:
        for li in lines_to_drop:
            fb = int(net.line.at[li, "from_bus"])
            tb = int(net.line.at[li, "to_bus"])
            print(f"  Removing line {li}: bus {fb} -> bus {tb}")
    net.line.drop(index=lines_to_drop, inplace=True)

    # Delete trafos connected to the removed bus
    trafos_to_drop = net.trafo.index[
        (net.trafo["hv_bus"] == bus_rm) | (net.trafo["lv_bus"] == bus_rm)
    ]
    if verbose and len(trafos_to_drop) > 0:
        for ti in trafos_to_drop:
            hv = int(net.trafo.at[ti, "hv_bus"])
            lv = int(net.trafo.at[ti, "lv_bus"])
            print(f"  Removing trafo {ti}: bus {hv} -> bus {lv}")
    net.trafo.drop(index=trafos_to_drop, inplace=True)

    # Delete the bus itself
    net.bus.drop(index=[bus_rm], inplace=True)

    # Update TN metadata (bus 11 is no longer a TN bus)
    tn_buses_updated = tuple(b for b in meta.tn_bus_indices if b != bus_rm)
    tn_lines_updated = tuple(
        li for li in meta.tn_line_indices if li not in set(lines_to_drop)
    )

    # Also remove any TSO DERs that were at the removed bus
    tso_der_indices_updated = []
    tso_der_buses_updated = []
    for s, b in zip(meta.tso_der_indices, meta.tso_der_buses):
        if b != bus_rm:
            tso_der_indices_updated.append(s)
            tso_der_buses_updated.append(b)

    if verbose:
        print(f"  Bus {bus_rm} removed from network.\n")

    # =====================================================================
    # 3. Delete IEEE loads at all coupling buses (they are replaced by HV nets)
    # =====================================================================
    all_coupling_buses_0idx = set()
    for sdef in _SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            b0 = b1 - 1
            if b0 != bus_rm:  # already removed
                all_coupling_buses_0idx.add(b0)

    if verbose:
        print("[add_hv_networks] Deleting IEEE loads at coupling buses "
              f"(0-idx): {sorted(all_coupling_buses_0idx)}")

    for b in all_coupling_buses_0idx:
        _delete_loads_at_bus(net, b)

    # =====================================================================
    # 4. Create 5 HV sub-networks
    # =====================================================================
    hv_nets: List[HVNetworkInfo] = []

    for sdef in _SUBNET_DEFS:
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
        # Attach zone metadata
        hv.zone = sdef["zone"]
        hv_nets.append(hv)

    # =====================================================================
    # 5. Wire EHV profiles to remaining IEEE 39-bus loads
    # =====================================================================
    _wire_ehv_profiles(net)

    # =====================================================================
    # 6. Power flow
    # =====================================================================
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    # =====================================================================
    # 7. Debug output
    # =====================================================================
    if verbose:
        _print_hv_summary(hv_nets, net)

    # =====================================================================
    # 8. Update metadata
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
        tn_bus_indices=tn_buses_updated,
        tn_line_indices=tn_lines_updated,
        gen_indices=meta.gen_indices,
        gen_bus_indices=meta.gen_bus_indices,
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
