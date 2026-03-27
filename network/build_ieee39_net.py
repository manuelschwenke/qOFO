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
            sn_mva = 50.0
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
