"""
network/ieee39/build.py
=======================
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

Scenarios
---------
* ``"base"``: all 9 PV generators active, no modifications.
* ``"reduced_gen_z2"``: removes the ex-slack generator at bus 30 from Zone 2.
* ``"wind_replace"``: replaces selected generators with STATCOM-capable wind
  parks (all Zone 2 at half P_mw, 2 each in Zones 1 and 3 at same P_mw).

Usage
-----
    from network.ieee39.build import build_ieee39_net

    # Step 1: base network (no DSO feeders)
    net, meta = build_ieee39_net()

Author: Manuel Schwenke / Claude Code
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from network.ieee39.constants import (
    GEN_NAMEPLATE,
    LOAD_CONST_FRACTION,
    LOAD_PEAK_BOOST,
    LOAD_VAR_FRACTION,
    NAMEPLATE_FACTOR,
    PROFILE_MAX,
    PROFILE_MEAN,
    ZONE3_BUSES_0IDX,
)
from network.ieee39.meta import IEEE39NetworkMeta
from network.ieee39.helpers import fix_line_lengths, swap_slack_to_bus38
from network.ieee39.scenarios import SCENARIO_REGISTRY


# ---------------------------------------------------------------------------
#  Main build function
# ---------------------------------------------------------------------------

def build_ieee39_net(
    *,
    ext_grid_vm_pu: float = 1.03,
    scenario: str = "base",
    verbose: bool = False,
) -> Tuple[pp.pandapowerNet, IEEE39NetworkMeta]:
    """
    Build the IEEE 39-bus New England test network.

    Parameters
    ----------
    ext_grid_vm_pu : float
        Voltage setpoint of the slack external grid [p.u.].
    scenario : str
        Network scenario.  ``"base"`` leaves all generators active.
        ``"reduced_gen_z2"`` removes the generator at bus 30 (Zone 2),
        leaving only Gen 1 (bus 31) in that zone.
        ``"wind_replace"`` replaces generators with STATCOM-capable wind
        parks: all Zone 2 gens at half P_mw; G1 + G8 in Zone 1 and
        G4 + G5 in Zone 3 at the same P_mw as the original generators.

    Returns
    -------
    net : pp.pandapowerNet
        Fully initialised pandapower network ready for power flow.
    meta : IEEE39NetworkMeta
        Immutable index catalogue.  DSO feeder fields are empty tuples;
        call :func:`add_dso_feeders` to populate them.
    """
    # -- Load standard IEEE 39-bus case ----------------------------------------
    # pandapower.networks.case39() returns the standard New England test system.
    # All buses are at 345 kV; generators at buses 30, 31, ..., 38, 39 (1-indexed).
    net = pn.case39()

    # -- Fix line lengths (preserve total impedance) ---------------------------
    fix_line_lengths(net)

    # -- Move slack to bus 38 (IEEE standard location) -------------------------
    # pandapower case39() places the ext_grid at bus 30 (0-indexed = bus 31,
    # 1-indexed), but the IEEE standard has the slack at bus 39 (1-indexed).
    _new_gen_bus30_idx = swap_slack_to_bus38(net, ext_grid_vm_pu)

    # Set all PV generators to the same voltage setpoint as the slack so
    # the reactive power burden is distributed evenly.  The case39 defaults
    # range from 0.984 to 1.064 which concentrates Q at the slack.
    net.gen["vm_pu"] = ext_grid_vm_pu

    # -- Metadata: label all original buses as TN (transmission network) -------
    #
    # We add a "subnet" column matching the convention in build_tuda_net.py so
    # that helper functions that filter by subnet work unchanged.
    net.bus["subnet"] = "TN"
    net.bus["vn_kv"]  # already set by case39()

    # -- Tag lines with subnet for later filtering -----------------------------
    # case39 has no trafo3w; all connections between TN buses are net.line or
    # net.trafo (there are a few step-up trafos in some versions).
    for li in net.line.index:
        net.line.at[li, "subnet"] = "TN"

    # -- Generators ------------------------------------------------------------
    # case39() puts the slack at ext_grid (bus 38 in 0-indexed = bus 39 in
    # 1-indexed).  All remaining generators are in net.gen as PV buses.
    # We collect their indices and bus indices.
    gen_indices: List[int] = sorted(int(g) for g in net.gen.index)
    gen_bus_indices: List[int] = [int(net.gen.at[g, "bus"]) for g in gen_indices]

    # -- Explicit nameplate / fuel type / Q envelope on every machine ---------
    # Single source of truth: ``GEN_NAMEPLATE`` (constants.py) keyed by the
    # gen's terminal bus 0-idx.  After ``swap_slack_to_bus38`` every machine
    # sits at one of buses {29..38} and the table covers all ten.  The Q
    # envelope is set to ``±0.5·sn_mva`` (typical synchronous-machine Q-circle)
    # so downstream capability plots and actuator bounds scale consistently.
    for gi in net.gen.index:
        term_bus = int(net.gen.at[gi, "bus"])
        if term_bus not in GEN_NAMEPLATE:
            raise KeyError(
                f"Gen {gi} sits on bus {term_bus} with no entry in "
                f"GEN_NAMEPLATE; expected one of {sorted(GEN_NAMEPLATE)}"
            )
        label, sn_mva, gen_type = GEN_NAMEPLATE[term_bus]
        net.gen.at[gi, "sn_mva"]     = sn_mva
        net.gen.at[gi, "max_p_mw"]   = sn_mva
        net.gen.at[gi, "min_p_mw"]   = 0.0
        net.gen.at[gi, "type"]       = gen_type
        net.gen.at[gi, "max_q_mvar"] =  0.5 * sn_mva
        net.gen.at[gi, "min_q_mvar"] = -0.5 * sn_mva
        net.gen.at[gi, "name"]       = f"{label}_bus{term_bus}"

    # -- Distributed-slack weights (approximate primary frequency response) ----
    # Every synchronous machine in ``net.gen`` participates in the
    # distributed slack with a weight proportional to its rated capacity
    # (``sn_mva``).  The ``slack=True`` gen at bus 38 additionally
    # anchors the voltage-angle reference (``swap_slack_to_bus38``
    # already converted the former ``ext_grid`` into such a gen), so the
    # former ``if ei in net.ext_grid`` branch is intentionally gone.
    for gi in net.gen.index:
        sn = float(net.gen.at[gi, "sn_mva"])
        assert sn > 0 and not pd.isna(sn), (
            f"gen {gi} missing sn_mva; nameplate loop should have set it"
        )
        net.gen.at[gi, "slack_weight"] = sn

    # -- TN buses and lines ----------------------------------------------------
    tn_buses: List[int] = sorted(int(b) for b in net.bus.index
                                  if str(net.bus.at[b, "subnet"]) == "TN")
    tn_lines: List[int] = sorted(int(li) for li in net.line.index
                                  if str(net.line.at[li, "subnet"]) == "TN")

    # -- Update original machine transformers (gen step-up trafos) -------------
    #
    # Strategy: keep existing case39 trafos that connect generator buses to the
    # 345 kV grid, but change their LV winding to 10.5 kV (generator terminal
    # voltage) and add OLTC tap parameters (+/-9, 1.25 % step, HV side).
    #
    # Special case -- bus 20 (0-idx 19):  Gen 3 at bus 34 (0-idx 33) is connected
    # via TWO series trafos:  bus 19 (0-idx 18) -> bus 20 (0-idx 19) -> bus 34
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
            # Size the machine trafo at the generator nameplate (already set
            # by the nameplate loop above).  Keeping trafo sn_mva ≥ gen sn_mva
            # avoids bottlenecking the machine through its step-up.
            sn_mva = float(net.gen.at[g, "sn_mva"])

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

        # -- Two-trafo chain detection ----------------------------------------
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

        # -- Set secondary side to 10.5 kV (generator terminal) ---------------
        lv_bus = int(net.trafo.at[tidx, "lv_bus"])
        net.trafo.at[tidx, "vn_lv_kv"] = 10.5
        net.bus.at[lv_bus, "vn_kv"] = 10.5

        # -- Set OLTC tap parameters ------------------------------------------
        net.trafo.at[tidx, "tap_min"] = -9
        net.trafo.at[tidx, "tap_max"] = 9
        net.trafo.at[tidx, "tap_step_percent"] = 1.25
        net.trafo.at[tidx, "tap_pos"] = 0
        net.trafo.at[tidx, "tap_side"] = "hv"
        net.trafo.at[tidx, "tap_neutral"] = 0
        net.trafo.at[tidx, "name"] = f"MachineTrf|gen{g}_bus{grid_bus}"

        machine_trafo_indices.append(int(tidx))
        machine_trafo_gen_map.append(g)

    # -- Add controllable OLTCs for the network transformers at Bus 12 ---------
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
        # Set OLTC tap parameters (standard +/-9, 1.25 % step)
        net.trafo.at[tidx, "tap_min"] = -9
        net.trafo.at[tidx, "tap_max"] = 9
        net.trafo.at[tidx, "tap_step_percent"] = 1.25
        net.trafo.at[tidx, "tap_pos"] = 0
        net.trafo.at[tidx, "tap_side"] = "hv"
        net.trafo.at[tidx, "tap_neutral"] = 0
        other_bus = (int(net.trafo.at[tidx, "hv_bus"])
                     if int(net.trafo.at[tidx, "lv_bus"]) == bus12_0idx
                     else int(net.trafo.at[tidx, "lv_bus"]))
        net.trafo.at[tidx, "name"] = (
            f"NetworkOLTC|bus{bus12_0idx}_bus{other_bus}"
        )

        machine_trafo_indices.append(int(tidx))
        machine_trafo_gen_map.append(-1)  # -1 indicates a network (non-machine) OLTC

    # Refresh gen_bus_indices after potential bus reassignments
    gen_bus_indices = [int(net.gen.at[g, "bus"]) for g in gen_indices]

    # -- Split every 345 kV load into half-constant + half-profile -----------
    # For every IEEE load whose bus is a TN (345 kV) bus, halve its p/q in
    # place (the remaining row becomes the constant half, with no profile
    # columns set) and add a second load at the same bus with ``base_p_mw``
    # / ``base_q_mvar`` scaled by ``0.5 / mean(profile)`` so that the *time
    # mean* of the two halves sums to the original nominal load.  The
    # profile-driven half carries the HS4 or HS5 simbench profile depending
    # on zone (HS5 for zone-3 buses per ``ZONE3_BUSES_0IDX``, HS4 otherwise).
    _split_tn_loads(net, tn_buses=tn_buses)

    # -- Run initial power flow ------------------------------------------------
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)

    meta = IEEE39NetworkMeta(
        tn_bus_indices   = tuple(tn_buses),
        tn_line_indices  = tuple(tn_lines),
        gen_indices      = tuple(gen_indices),
        gen_bus_indices  = tuple(gen_bus_indices),
        gen_grid_bus_indices = tuple(gen_grid_bus_indices),
        machine_trafo_indices = tuple(machine_trafo_indices),
        machine_trafo_gen_map = tuple(machine_trafo_gen_map),
        tso_der_indices  = (),
        tso_der_buses    = (),
        # DSO fields are empty until add_dso_feeders() is called
    )

    # -- Apply scenario --------------------------------------------------------
    if scenario not in SCENARIO_REGISTRY:
        raise ValueError(
            f"Unknown scenario: {scenario!r}. "
            f"Valid: {sorted(SCENARIO_REGISTRY)}"
        )
    apply_fn = SCENARIO_REGISTRY[scenario]
    net, meta = apply_fn(net, meta, ext_grid_vm_pu=ext_grid_vm_pu,
                         new_gen_bus30_idx=_new_gen_bus30_idx)

    if verbose:
        p_load_const = float(
            net.load.loc[net.load["profile_p"].isna(), "p_mw"].sum()
        )
        p_load_profile = float(
            net.load.loc[net.load["profile_p"].notna(), "p_mw"].sum()
        )
        sgen_p = float(net.sgen["p_mw"].sum()) if len(net.sgen) else 0.0
        gen_p_base = float(net.gen["p_mw"].sum())
        gen_sn = float(net.gen["sn_mva"].sum())
        eg_sn = (
            float(net.ext_grid["sn_mva"].sum())
            if (not net.ext_grid.empty and "sn_mva" in net.ext_grid.columns)
            else 0.0
        )
        n_gen, n_sgen = len(net.gen), len(net.sgen)
        print(
            f"[build_ieee39_net scenario={scenario!r}] "
            f"load P (const+profile@mean) = {p_load_const:.0f} + "
            f"{p_load_profile:.0f} MW = {p_load_const + p_load_profile:.0f} MW | "
            f"sgen P = {sgen_p:.0f} MW | "
            f"gen P (base-case, n={n_gen}) = {gen_p_base:.0f} MW | "
            f"nameplate sum sn_mva = {gen_sn:.0f} (gen) + {eg_sn:.0f} (ext) "
            f"= {gen_sn + eg_sn:.0f} MVA | "
            f"sgen count = {n_sgen}"
        )

    return net, meta


# ---------------------------------------------------------------------------
#  50 / 50 constant + profile load split for TN buses
# ---------------------------------------------------------------------------

def _split_tn_loads(net: pp.pandapowerNet, *, tn_buses: List[int]) -> None:
    """Split every IEEE load at a TN (345 kV) bus into two rows.

    * Existing row is reduced in place to ``LOAD_CONST_FRACTION * orig``
      and keeps empty ``profile_p`` / ``profile_q`` (constant load).
    * A sibling row is created at the same bus with
      ``base_p/q = LOAD_VAR_FRACTION * orig / max(profile)`` and profile
      columns set to HS4 or HS5 depending on zone membership (Zone 3 ->
      HS5, else HS4).

    With ``c = LOAD_CONST_FRACTION``, ``v = LOAD_VAR_FRACTION`` and
    ``profile_max = PROFILE_MAX[prof]`` the per-bus total at time t is

        p_total(t) = p_orig * (c + v * profile[t] / profile_max)

    so peak load = ``(c + v) * p_orig`` and trough = ``c * p_orig``.  With
    the default ``c = 0.30, v = 0.70`` this caps peak at the IEEE 39 base
    and lets load drop to 30 % during low-profile hours.
    """
    if "profile_p" not in net.load.columns:
        net.load["profile_p"] = None
    if "profile_q" not in net.load.columns:
        net.load["profile_q"] = None
    if "subnet" not in net.load.columns:
        net.load["subnet"] = None
    if "base_p_mw" not in net.load.columns:
        net.load["base_p_mw"] = net.load["p_mw"].astype(float)
        net.load["base_q_mvar"] = net.load["q_mvar"].astype(float)

    c = LOAD_CONST_FRACTION
    v = LOAD_VAR_FRACTION

    tn_set = set(int(b) for b in tn_buses)
    for li in list(net.load.index):
        bus = int(net.load.at[li, "bus"])
        if bus not in tn_set:
            continue

        p_orig = float(net.load.at[li, "p_mw"])
        q_orig = float(net.load.at[li, "q_mvar"])

        # Constant share of the load — pin base to ``c * orig`` so
        # apply_profiles() leaves it alone (no profile columns set).
        net.load.at[li, "p_mw"] = c * p_orig
        net.load.at[li, "q_mvar"] = c * q_orig
        net.load.at[li, "base_p_mw"] = c * p_orig
        net.load.at[li, "base_q_mvar"] = c * q_orig
        net.load.at[li, "profile_p"] = None
        net.load.at[li, "profile_q"] = None
        net.load.at[li, "subnet"] = "TN"

        if bus in ZONE3_BUSES_0IDX:
            prof_p, prof_q = "HS5_pload", "HS5_qload"
        else:
            prof_p, prof_q = "HS4_pload", "HS4_qload"

        base_p_profile = LOAD_PEAK_BOOST * v * p_orig / PROFILE_MAX[prof_p]
        base_q_profile = LOAD_PEAK_BOOST * v * q_orig / PROFILE_MAX[prof_q]
        orig_name = net.load.at[li, "name"] if "name" in net.load.columns else None
        new_name = f"{orig_name}_profile" if orig_name else f"Load_bus{bus}_profile"

        # Initial ``p_mw`` / ``q_mvar`` set so the per-bus total at startup
        # (constant row + this variable row) equals ``p_orig`` / ``q_orig``
        # -- the IEEE 39 base case dispatch.  apply_profiles() will overwrite
        # these at every step.  This avoids a large startup surplus that
        # would force the slack to absorb gigawatts before the simulation
        # even begins.
        pp.create_load(
            net,
            bus=bus,
            p_mw=v * p_orig,
            q_mvar=v * q_orig,
            name=new_name,
        )
        new_idx = net.load.index[-1]
        net.load.at[new_idx, "profile_p"] = prof_p
        net.load.at[new_idx, "profile_q"] = prof_q
        net.load.at[new_idx, "subnet"] = "TN"
        net.load.at[new_idx, "base_p_mw"] = base_p_profile
        net.load.at[new_idx, "base_q_mvar"] = base_q_profile


# ---------------------------------------------------------------------------
#  q_mode tagging (refactor_v2, Soleimani §III-B)
# ---------------------------------------------------------------------------

def tag_der_q_modes(
    net: pp.pandapowerNet,
    meta: IEEE39NetworkMeta,
    *,
    tso_q_mode: str = "qv",
    dso_q_mode: str = "qv",
    q_mode_overrides: Optional[Dict[int, str]] = None,
    tso_qv_slope_pu: float = 0.07,
    dso_qv_slope_pu: float = 0.07,
    qv_slope_pu_overrides: Optional[Dict[int, float]] = None,
    tso_qv_vref_pu: float = 1.00,
    dso_qv_vref_pu: float = 1.00,
    qv_vref_pu_overrides: Optional[Dict[int, float]] = None,
    tso_qv_deadband_pu: float = 0.005,
    dso_qv_deadband_pu: float = 0.005,
    qv_deadband_pu_overrides: Optional[Dict[int, float]] = None,
    tso_cosphi: float = 1.0,
    dso_cosphi: float = 1.0,
    cosphi_overrides: Optional[Dict[int, float]] = None,
    tso_cosphi_sign: int = -1,
    dso_cosphi_sign: int = -1,
    cosphi_sign_overrides: Optional[Dict[int, int]] = None,
    verbose: bool = False,
) -> IEEE39NetworkMeta:
    """Tag every TSO and DSO DER sgen with its q_mode and parameters.

    Adds the following columns to ``net.sgen`` (creating them if absent):

    * ``q_mode``             — ``"qv"`` (Q(V) droop) or ``"cosphi"`` (fixed PF).
    * ``qv_slope_pu``        — droop slope (pu_q/pu_v).  Read only when ``q_mode=="qv"``.
    * ``qv_vref_pu``         — nominal droop centre voltage; used as cold-start
                                fallback when the anchor column below is NaN.
    * ``qv_deadband_pu``     — half-width of the symmetric deadband around the
                                reanchored V_ref.  Default 0.005 p.u.
    * ``cosphi``             — power factor magnitude in [0, 1].  Read only when
                                ``q_mode=="cosphi"``.
    * ``cosphi_sign``        — ``+1`` over-excited (Q injection) / ``-1`` under-
                                excited (Q absorption).
    * ``q_set_mvar``         — controller-commanded Q setpoint at the reanchored
                                V_ref (Mvar).  Initialised to 0.0; the OFO writes
                                here every step in w-shift mode.
    * ``qv_vref_anchor_pu``  — V_ref reanchored to the most recent measured bus
                                voltage by the apply step.  Initialised to NaN
                                so the local loop falls back to ``qv_vref_pu``
                                until the first apply.

    Hierarchy: per-DER override > level (TSO/DSO) default.  Keys in the
    ``*_overrides`` maps are pandapower sgen indices.

    This function is **additive**.  It does not modify ``meta``; it only
    tags ``net.sgen`` columns that the Q(V) local loop reads at run time.

    Parameters
    ----------
    net, meta
        Network and metadata catalogue.  ``meta.tso_der_indices`` and
        ``meta.dso_der_indices`` define the populations being tagged.
    tso_q_mode, dso_q_mode
        Default q_mode for the level.  Must be ``"qv"`` or ``"cosphi"``.
    q_mode_overrides
        Per-sgen-index override of the level default.
    tso_qv_*_pu, dso_qv_*_pu
        Default qv parameters per level.
    qv_*_pu_overrides
        Per-sgen-index overrides of the qv parameters.
    tso_cosphi, dso_cosphi, tso_cosphi_sign, dso_cosphi_sign
        Default cosphi parameters per level.  Sign convention: ``+1``
        over-excited, ``-1`` under-excited (DE LV grid-code default).
    cosphi_*_overrides
        Per-sgen-index overrides of the cosphi parameters.
    verbose
        Print a one-line summary of the tagged populations.

    Returns
    -------
    IEEE39NetworkMeta
        ``meta`` returned unchanged for caller chaining.

    Notes
    -----
    The OFO controllers exclude ``q_mode=="cosphi"`` DERs from their
    action vectors (these DERs are not actuators); the OFO writes
    ``q_set_mvar`` only for ``q_mode=="qv"`` DERs and reanchors
    ``qv_vref_anchor_pu`` to the most recent measured bus voltage.
    The local Q(V) characteristic then reads
    ``Q = q_set - R · (V - V_anchor)`` with deadband ``qv_deadband_pu``.
    """
    valid_modes = {"qv", "cosphi"}
    for nm, mode in (("tso_q_mode", tso_q_mode), ("dso_q_mode", dso_q_mode)):
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid {nm} = {mode!r}; expected one of {sorted(valid_modes)}"
            )

    q_mode_overrides = q_mode_overrides or {}
    qv_slope_pu_overrides = qv_slope_pu_overrides or {}
    qv_vref_pu_overrides = qv_vref_pu_overrides or {}
    qv_deadband_pu_overrides = qv_deadband_pu_overrides or {}
    cosphi_overrides = cosphi_overrides or {}
    cosphi_sign_overrides = cosphi_sign_overrides or {}

    # Validate any override values up-front so the failure mode is clear.
    for s, m in q_mode_overrides.items():
        if m not in valid_modes:
            raise ValueError(
                f"q_mode_overrides[{s}] = {m!r}; expected one of "
                f"{sorted(valid_modes)}"
            )
    for s, sg in cosphi_sign_overrides.items():
        if sg not in (-1, +1):
            raise ValueError(
                f"cosphi_sign_overrides[{s}] = {sg!r}; expected +1 or -1"
            )

    # Ensure all columns exist with a sensible default for sgens not in
    # tso_/dso_der_indices (e.g. equivalent units).  Numerics default to
    # NaN so an unintended read shows up clearly; q_mode defaults to
    # "qv" and q_set_mvar to 0.0 (a safe "no shift" starting point).
    # qv_vref_anchor_pu is NaN until the first apply step populates it
    # from the bus voltage of the just-completed PF.
    if "q_mode" not in net.sgen.columns:
        net.sgen["q_mode"] = "qv"
    if "qv_slope_pu" not in net.sgen.columns:
        net.sgen["qv_slope_pu"] = float("nan")
    if "qv_vref_pu" not in net.sgen.columns:
        net.sgen["qv_vref_pu"] = float("nan")
    if "qv_deadband_pu" not in net.sgen.columns:
        net.sgen["qv_deadband_pu"] = float("nan")
    if "cosphi" not in net.sgen.columns:
        net.sgen["cosphi"] = float("nan")
    if "cosphi_sign" not in net.sgen.columns:
        net.sgen["cosphi_sign"] = 0
    if "q_set_mvar" not in net.sgen.columns:
        net.sgen["q_set_mvar"] = 0.0
    if "qv_vref_anchor_pu" not in net.sgen.columns:
        net.sgen["qv_vref_anchor_pu"] = float("nan")

    def _tag(s: int, level_default_mode: str,
             level_slope: float, level_vref: float, level_db: float,
             level_cosphi: float, level_sign: int) -> None:
        if s not in net.sgen.index:
            return
        mode = q_mode_overrides.get(s, level_default_mode)
        net.sgen.at[s, "q_mode"] = mode
        net.sgen.at[s, "qv_slope_pu"] = float(
            qv_slope_pu_overrides.get(s, level_slope)
        )
        net.sgen.at[s, "qv_vref_pu"] = float(
            qv_vref_pu_overrides.get(s, level_vref)
        )
        net.sgen.at[s, "qv_deadband_pu"] = float(
            qv_deadband_pu_overrides.get(s, level_db)
        )
        net.sgen.at[s, "cosphi"] = float(
            cosphi_overrides.get(s, level_cosphi)
        )
        net.sgen.at[s, "cosphi_sign"] = int(
            cosphi_sign_overrides.get(s, level_sign)
        )
        # q_set_mvar and qv_vref_anchor_pu are runtime state, not config;
        # the OFO / apply step overwrites them every step.  Preserve any
        # existing values so that re-calling tag_der_q_modes mid-run does
        # not stomp on the live state.
        if pd.isna(net.sgen.at[s, "q_set_mvar"]):
            net.sgen.at[s, "q_set_mvar"] = 0.0
        # qv_vref_anchor_pu intentionally left NaN at tag time; the local
        # Q(V) loop falls back to qv_vref_pu (cold start) until the first
        # apply step writes the measured bus voltage.

    for s in meta.tso_der_indices:
        _tag(int(s), tso_q_mode,
             tso_qv_slope_pu, tso_qv_vref_pu, tso_qv_deadband_pu,
             tso_cosphi, tso_cosphi_sign)
    for s in meta.dso_der_indices:
        _tag(int(s), dso_q_mode,
             dso_qv_slope_pu, dso_qv_vref_pu, dso_qv_deadband_pu,
             dso_cosphi, dso_cosphi_sign)

    if verbose:
        n_tso = len(meta.tso_der_indices)
        n_dso = len(meta.dso_der_indices)
        n_cosphi = int((net.sgen["q_mode"] == "cosphi").sum())
        n_qv = int((net.sgen["q_mode"] == "qv").sum())
        print(
            f"[tag_der_q_modes] tagged {n_tso} TSO + {n_dso} DSO DERs; "
            f"net.sgen now has {n_qv} qv-mode and {n_cosphi} cosphi-mode rows"
        )

    return meta
