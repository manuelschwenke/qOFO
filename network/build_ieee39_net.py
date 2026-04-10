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
import pandas as pd
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
    """Bus index of each generator (terminal bus — may be 10.5 kV if machine
    trafos are present, or 345 kV if directly connected)."""

    # ── TSO-level DER sgens ───────────────────────────────────────────────────
    tso_der_indices: Tuple[int, ...]
    """Indices of sgen elements co-located at generator buses.
    These represent the TSO's controllable reactive-power actuators.
    """

    tso_der_buses: Tuple[int, ...]
    """Bus indices of the TSO DER sgens (same order as tso_der_indices)."""

    # ── Generators: grid buses and machine transformers (optional) ─────────────
    gen_grid_bus_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Original 345 kV grid bus of each generator (before machine trafo
    insertion).  Used for zone partitioning.  Same as gen_bus_indices when
    no machine transformers exist."""

    machine_trafo_indices: Tuple[int, ...] = field(default_factory=tuple)
    """Indices in ``net.trafo`` of 2W machine transformers connecting
    generator terminals to the grid.  OLTC actuators in the TSO OFO."""

    machine_trafo_gen_map: Tuple[int, ...] = field(default_factory=tuple)
    """For each machine trafo, the ``net.gen`` index of its generator
    (same order as machine_trafo_indices)."""

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
    # Set all PV generators to the same voltage setpoint as the slack so
    # the reactive power burden is distributed evenly.  The case39 defaults
    # range from 0.984 to 1.064 which concentrates Q at the slack.
    net.gen["vm_pu"] = ext_grid_vm_pu

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

    # ── Update original machine transformers (gen step-up trafos) ───────────
    #
    # Strategy: keep existing case39 trafos that connect generator buses to the
    # 345 kV grid, but change their LV winding to 10.5 kV (generator terminal
    # voltage) and add OLTC tap parameters (±9, 1.25 % step, HV side).
    #
    # Special case — bus 20 (0-idx 19):  Gen 3 at bus 34 (0-idx 33) is connected
    # via TWO series trafos:  bus 19 (0-idx 18) → bus 20 (0-idx 19) → bus 34
    # (0-idx 33).  We replace both with a single trafo from bus 19 (0-idx 18)
    # directly to bus 34 (0-idx 33).
    machine_trafo_indices: List[int] = []
    machine_trafo_gen_map: List[int] = []
    gen_grid_bus_indices: List[int] = []

    # Identify all generator buses (for two-trafo chain detection)
    gen_bus_set_initial = set(gen_bus_indices)

    for g in gen_indices:
        gen_bus = int(net.gen.at[g, "bus"])

        # Find transformers directly connected to this generator bus
        mask = (net.trafo["lv_bus"] == gen_bus) | (net.trafo["hv_bus"] == gen_bus)
        trafo_idx = net.trafo.index[mask].tolist()

        if not trafo_idx:
            # Generator is directly connected to the 345 kV grid (no step-up trafo).
            # Create a new machine transformer: grid_bus (345 kV) -> new terminal bus (10.5 kV).
            grid_bus = gen_bus
            gen_grid_bus_indices.append(grid_bus)
            grid_vn = float(net.bus.at[grid_bus, "vn_kv"])
            gen_terminal_kv = 10.5
            p_mw = float(net.gen.at[g, "p_mw"])
            sn_mva = max(p_mw * 1.2, 100.0)

            # Create terminal bus
            term_bus = pp.create_bus(
                net, vn_kv=gen_terminal_kv,
                name=f"GEN_TERM|gen{g}_bus{grid_bus}",
                type="b", subnet="GEN_TERM",
            )
            # Move generator to terminal bus
            net.gen.at[g, "bus"] = int(term_bus)

            # Create 2W machine transformer
            tidx = pp.create_transformer_from_parameters(
                net,
                hv_bus=grid_bus, lv_bus=int(term_bus),
                sn_mva=sn_mva,
                vn_hv_kv=grid_vn, vn_lv_kv=gen_terminal_kv,
                vk_percent=12.0, vkr_percent=0.3,
                pfe_kw=0.0, i0_percent=0.0,
                tap_side="hv", tap_neutral=0,
                tap_min=-9, tap_max=9, tap_pos=0,
                tap_step_percent=1.25, shift_degree=0.0,
                tap_changer_type="Ratio",
                name=f"MachineTrf|gen{g}_bus{grid_bus}",
            )
            machine_trafo_indices.append(int(tidx))
            machine_trafo_gen_map.append(g)
            continue

        # Pick the trafo whose OTHER side is the grid bus
        tidx = trafo_idx[0]
        if int(net.trafo.at[tidx, "lv_bus"]) == gen_bus:
            grid_bus = int(net.trafo.at[tidx, "hv_bus"])
        else:
            grid_bus = int(net.trafo.at[tidx, "lv_bus"])

        # ── Two-trafo chain detection ────────────────────────────────────
        # If the "grid bus" itself is connected to the backbone only via
        # another trafo (no lines), we have a two-trafo chain.
        # Replace both trafos with a single one from the true grid bus to
        # the generator bus.  Any load at the intermediate bus is moved
        # to the true grid bus.
        #
        # Example in case39: bus 19 (0-idx 18) --trafo5--> bus 20 (0-idx 19)
        #   --trafo7--> bus 34 (0-idx 33, gen).  Replace with single trafo
        #   from bus 18 to bus 33.
        mask2 = ((net.trafo["lv_bus"] == grid_bus) | (net.trafo["hv_bus"] == grid_bus))
        trafos_at_grid_bus = net.trafo.index[mask2].tolist()
        trafos_at_grid_bus = [t for t in trafos_at_grid_bus if t != tidx]
        has_lines = ((net.line["from_bus"] == grid_bus) | (net.line["to_bus"] == grid_bus)).any()
        has_other_gen = grid_bus in (gen_bus_set_initial - {gen_bus})

        if len(trafos_at_grid_bus) == 1 and not has_lines and not has_other_gen:
            # Two-trafo chain: intermediate bus = grid_bus
            # The second trafo connects grid_bus to the true backbone bus
            t2 = trafos_at_grid_bus[0]
            if int(net.trafo.at[t2, "lv_bus"]) == grid_bus:
                true_grid_bus = int(net.trafo.at[t2, "hv_bus"])
            else:
                true_grid_bus = int(net.trafo.at[t2, "lv_bus"])

            # Remove the second (intermediate) trafo
            net.trafo.drop(t2, inplace=True)

            # Rewire the gen-side trafo: true_grid_bus -> gen_bus
            if int(net.trafo.at[tidx, "hv_bus"]) == grid_bus:
                net.trafo.at[tidx, "hv_bus"] = true_grid_bus
            else:
                net.trafo.at[tidx, "lv_bus"] = true_grid_bus

            # Move any loads from the intermediate bus to the true grid bus
            for li in net.load.index:
                if int(net.load.at[li, "bus"]) == grid_bus:
                    net.load.at[li, "bus"] = true_grid_bus

            # Remove the intermediate bus
            net.bus.drop(index=[grid_bus], inplace=True)

            grid_bus = true_grid_bus

        gen_grid_bus_indices.append(grid_bus)

        # ── Set secondary side to 10.5 kV (generator terminal) ──────────
        lv_bus = int(net.trafo.at[tidx, "lv_bus"])
        net.trafo.at[tidx, "vn_lv_kv"] = 10.5
        net.bus.at[lv_bus, "vn_kv"] = 10.5

        # ── Set OLTC tap parameters ─────────────────────────────────────
        net.trafo.at[tidx, "tap_min"] = -9
        net.trafo.at[tidx, "tap_max"] = 9
        net.trafo.at[tidx, "tap_step_percent"] = 1.25
        net.trafo.at[tidx, "tap_pos"] = 0
        net.trafo.at[tidx, "tap_side"] = "hv"
        net.trafo.at[tidx, "tap_neutral"] = 0

        machine_trafo_indices.append(int(tidx))
        machine_trafo_gen_map.append(g)

    # ── Add controllable OLTCs for the network transformers at Bus 12 ───────
    #
    # Bus 12 (1-indexed = bus 11, 0-indexed) is connected to the backbone via
    # two transformers: 11-10 (Bus 12-Bus 11) and 11-12 (Bus 12-Bus 13).
    # Even though these are not machine step-up transformers, we make them
    # controllable OLTCs to give the TSO more reactive-power control.
    bus12_0idx = 11
    mask_b12 = (net.trafo["lv_bus"] == bus12_0idx) | (net.trafo["hv_bus"] == bus12_0idx)
    b12_trafos = net.trafo.index[mask_b12].tolist()

    for tidx in b12_trafos:
        if tidx in machine_trafo_indices:
            continue
        # Set OLTC tap parameters (standard ±9, 1.25 % step)
        net.trafo.at[tidx, "tap_min"] = -9
        net.trafo.at[tidx, "tap_max"] = 9
        net.trafo.at[tidx, "tap_step_percent"] = 1.25
        net.trafo.at[tidx, "tap_pos"] = 0
        net.trafo.at[tidx, "tap_side"] = "hv"
        net.trafo.at[tidx, "tap_neutral"] = 0

        machine_trafo_indices.append(int(tidx))
        machine_trafo_gen_map.append(-1)  # -1 indicates a network (non-machine) OLTC

    # Refresh gen_bus_indices after potential bus reassignments
    gen_bus_indices = [int(net.gen.at[g, "bus"]) for g in gen_indices]

    # ── Add TSO DER sgens at PQ load buses ───────────────────────────────────
    #
    # TSO DERs must sit at PQ (load) buses so that the Q sensitivity
    # dV/dQ_DER can be computed via the reduced Jacobian.  Generator buses
    # are PV buses -- their voltage is fixed by the AVR and dV/dQ_DER = 0
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
        # Profiles for TN-DER units (analogous to HV network DERs)
        tn_der_profiles = ["WP10", "WP7", "PV3"]
        
        # Collect unique PQ load buses (not PV or slack)
        load_buses_pq: List[int] = sorted(
            {int(net.load.at[li, "bus"]) for li in net.load.index
             if int(net.load.at[li, "bus"]) not in pv_and_slack_buses}
        )
        for i, bus in enumerate(load_buses_pq):
            # Default DER rating: 200 MVA. Scale with p_mw fraction (default 0.3)
            # for a small active component (solar / wind).
            sn_mva = 200.0
            p_mw   = sn_mva * der_p_mw_fraction
            
            # Select profile from rotation
            prof = tn_der_profiles[i % len(tn_der_profiles)]
            
            idx = pp.create_sgen(
                net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=0.0,
                sn_mva=sn_mva,
                name=f"TN_DER|load_bus{bus}",
                subnet='TN',
                op_diagram='VDE-AR-N-4120-v2',
            )
            net.sgen.at[idx, "profile"] = prof
            
            tso_der_indices.append(int(idx))
            tso_der_buses.append(bus)

    # ── Run initial power flow ────────────────────────────────────────────────
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    meta = IEEE39NetworkMeta(
        tn_bus_indices   = tuple(tn_buses),
        tn_line_indices  = tuple(tn_lines),
        gen_indices      = tuple(gen_indices),
        gen_bus_indices  = tuple(gen_bus_indices),
        gen_grid_bus_indices = tuple(gen_grid_bus_indices),
        machine_trafo_indices = tuple(machine_trafo_indices),
        machine_trafo_gen_map = tuple(machine_trafo_gen_map),
        tso_der_indices  = tuple(tso_der_indices),
        tso_der_buses    = tuple(tso_der_buses),
        # DSO fields are empty until add_dso_feeders() is called
    )
    return net, meta


# ---------------------------------------------------------------------------
#  110 kV HV sub-network attachment (TUDA topology copies)
# ---------------------------------------------------------------------------

@dataclass
class HVNetworkInfo:
    """Tracking information for one attached 110 kV sub-network.

    Each HV sub-network replicates the TUDA 110 kV distribution topology
    (10 buses, 11 lines) and is coupled to the IEEE 39-bus 345 kV
    transmission network via 3-winding 345/110/20 kV transformers at three
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
    """3-winding transformer indices (``net.trafo3w``) coupling
    345 kV TN to 110 kV HV with a 20 kV tertiary winding."""

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
    # dict(net_id="DSO_4", zone=3,
    #      ieee_1idx=(24, 21, 23), hv_buses=(3, 0, 8), scale=2.00, gen="pv"),
    # dict(net_id="DSO_5", zone=1,
    #      ieee_1idx=(27, 26, 25), hv_buses=(3, 0, 8), scale=3.00, gen="wind"),
]


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


def _reduce_loads_at_bus(
    net: pp.pandapowerNet,
    bus: int,
    p_remove_mw: float,
    q_remove_mvar: float,
) -> None:
    """Reduce load at *bus* by (p_remove, q_remove), keeping the remainder.

    If the removal equals or exceeds the existing load, the load is deleted
    entirely.  Otherwise the load's P and Q are reduced in place.
    """
    mask = net.load["bus"] == bus
    if not mask.any():
        return
    existing_p = float(net.load.loc[mask, "p_mw"].sum())
    existing_q = float(net.load.loc[mask, "q_mvar"].sum())

    remaining_p = existing_p - p_remove_mw
    remaining_q = existing_q - q_remove_mvar

    if remaining_p <= 0.01 and remaining_q <= 0.01:
        # Nothing left — remove entirely
        net.load.drop(index=net.load.index[mask], inplace=True)
    else:
        # Scale down existing loads proportionally
        idx = net.load.index[mask]
        if len(idx) == 1:
            net.load.at[idx[0], "p_mw"] = max(remaining_p, 0.0)
            net.load.at[idx[0], "q_mvar"] = max(remaining_q, 0.0)
        else:
            # Multiple loads at same bus: scale all proportionally
            scale_p = max(remaining_p, 0.0) / max(existing_p, 1e-6)
            scale_q = max(remaining_q, 0.0) / max(existing_q, 1e-6)
            for i in idx:
                net.load.at[i, "p_mw"] *= scale_p
                net.load.at[i, "q_mvar"] *= scale_q


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
    ``_reduce_loads_at_bus``.

    For buses without a load, the average of the buses that DO have loads
    in the same 3-bus set is used.  DSO_3 is special: it uses the average
    of DSO_1 and DSO_2 totals (since its coupling buses have no IEEE load).

    Returns
    -------
    dict : net_id -> (total_p_mw, total_q_mvar)
    """
    max_s_mva = n_couplers * coupler_sn_mva          # 900 MVA default
    n_nets = len(_SUBNET_DEFS)

    # ── Step 1: Pool all real loads at ALL coupling buses ───────────────
    pool_p = 0.0
    pool_q = 0.0
    for sdef in _SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            p, q = _get_load_at_bus(net, b1 - 1)
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
    for sdef in _SUBNET_DEFS:
        ref[sdef["net_id"]] = (share_p, share_q)

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
    # 2. Reduce IEEE loads at coupling buses by the amount moved to DN
    # =====================================================================
    # The total DN load (pooled across all DSOs) was distributed equally
    # among the HV networks.  We now reduce the original TN loads at the
    # coupling buses by a total of (n_nets × share) = pool.  The reduction
    # is distributed proportionally across all loaded coupling buses.
    all_coupling_buses_0idx = set()

    # Collect all coupling buses and their original loads
    _original_bus_loads: Dict[int, Tuple[float, float]] = {}
    for sdef in _SUBNET_DEFS:
        for b1 in sdef["ieee_1idx"]:
            b0 = b1 - 1
            all_coupling_buses_0idx.add(b0)
            if b0 not in _original_bus_loads:
                _original_bus_loads[b0] = _get_load_at_bus(net, b0)

    # Total load at all coupling buses (= pool) and total DN load
    pool_p = sum(p for p, q in _original_bus_loads.values())
    pool_q = sum(q for p, q in _original_bus_loads.values())
    n_nets = len(_SUBNET_DEFS)
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
        _reduce_loads_at_bus(net, b, remove_p, remove_q)

    if verbose:
        print("[add_hv_networks] Reduced IEEE loads at coupling buses "
              f"(0-idx): {sorted(all_coupling_buses_0idx)}")
        for b in sorted(all_coupling_buses_0idx):
            orig_p, orig_q = _original_bus_loads[b]
            now_p, now_q = _get_load_at_bus(net, b)
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
    # 4. Wire EHV profiles to remaining IEEE 39-bus loads
    # =====================================================================
    _wire_ehv_profiles(net)

    # =====================================================================
    # 5. Power flow
    # =====================================================================

    # Verify no stale trafo references
    #print(net.trafo[["hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv"]])
    pp.runpp(net, run_control=False, calculate_voltage_angles=True, init='auto', max_iteration=50)

    # =====================================================================
    # 6. Debug output
    # =====================================================================
    if verbose:
        _print_hv_summary(hv_nets, net)

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


def remove_generators(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    gen_indices_to_remove: List[int],
) -> IEEE39NetworkMeta:
    """
    Remove synchronous generators (net.gen) by pandapower index.

    When machine transformers are present, also removes the associated 2W
    machine trafo and the 10.5 kV terminal bus from the network. Failing to
    do so would leave a floating PQ bus (P=0, Q=0) connected to a stiff
    transformer, which degrades Newton-Raphson conditioning.

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to modify in-place.
    meta : IEEE39NetworkMeta
        Current metadata catalogue.
    gen_indices_to_remove : List[int]
        Pandapower net.gen row indices to delete. Every entry must exist
        in net.gen.index; missing indices raise KeyError immediately.

    Returns
    -------
    meta : IEEE39NetworkMeta
        Updated metadata with all gen-related fields pruned consistently.

    Raises
    ------
    KeyError
        If any index in gen_indices_to_remove is not present in net.gen.
    ValueError
        If gen_indices_to_remove is empty.
    """
    if not gen_indices_to_remove:
        raise ValueError("gen_indices_to_remove must contain at least one index.")

    existing  = set(int(i) for i in net.gen.index)
    requested = set(int(i) for i in gen_indices_to_remove)
    missing   = requested - existing
    if missing:
        raise KeyError(
            f"The following gen indices do not exist in net.gen: {sorted(missing)}"
        )

    # -- Remove associated machine trafos and terminal buses (if present) ----
    #
    # With machine trafos, each generator sits on a 10.5 kV terminal bus
    # connected to the grid via a 2W transformer. Simply removing net.gen[g]
    # without removing the terminal bus leaves a floating PQ bus (P=0, Q=0)
    # at the transformer's LV side. This causes ill-conditioning of the NR
    # Jacobian and must be avoided.
    machine_trafos_to_remove: List[int] = []
    terminal_buses_to_remove: List[int] = []

    for t_idx, g_idx in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map):
        if g_idx in requested:
            machine_trafos_to_remove.append(int(t_idx))
            terminal_buses_to_remove.append(int(net.trafo.at[t_idx, "lv_bus"]))

    if machine_trafos_to_remove:
        net.trafo.drop(index=machine_trafos_to_remove, inplace=True)
    if terminal_buses_to_remove:
        net.bus.drop(index=terminal_buses_to_remove, inplace=True)

    # -- Drop the generator rows from net.gen ---------------------------------
    net.gen.drop(index=sorted(requested), inplace=True)

    # -- Rebuild all gen-related meta fields ----------------------------------
    #
    # All four fields (gen_indices, gen_bus_indices, gen_grid_bus_indices,
    # machine_trafo_*) must be pruned in parallel so they remain consistent.
    # Passing only gen_indices + gen_bus_indices (the old bug) left the other
    # three fields at their dataclass defaults of (), causing the zone-lookup
    # fallback in run_multi_tso_dso to use 10.5 kV terminal buses and assign
    # zero generators to every zone.
    new_gen_indices = [
        g for g in meta.gen_indices if g not in requested
    ]
    new_gen_bus_indices = [
        b for g, b in zip(meta.gen_indices, meta.gen_bus_indices)
        if g not in requested
    ]
    new_gen_grid_bus_indices = [
        b for g, b in zip(meta.gen_indices, meta.gen_grid_bus_indices)
        if g not in requested
    ] if meta.gen_grid_bus_indices else []

    new_machine_trafo_indices = [
        t for t, g in zip(meta.machine_trafo_indices, meta.machine_trafo_gen_map)
        if g not in requested
    ]
    new_machine_trafo_gen_map = [
        g for g in meta.machine_trafo_gen_map
        if g not in requested
    ]

    return IEEE39NetworkMeta(
        tn_bus_indices        = meta.tn_bus_indices,
        tn_line_indices       = meta.tn_line_indices,
        gen_indices           = tuple(new_gen_indices),
        gen_bus_indices       = tuple(new_gen_bus_indices),
        gen_grid_bus_indices  = tuple(new_gen_grid_bus_indices),   # ← was missing
        machine_trafo_indices = tuple(new_machine_trafo_indices),  # ← was missing
        machine_trafo_gen_map = tuple(new_machine_trafo_gen_map),  # ← was missing
        tso_der_indices       = meta.tso_der_indices,
        tso_der_buses         = meta.tso_der_buses,
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