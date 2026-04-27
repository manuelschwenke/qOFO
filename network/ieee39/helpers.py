"""
network.ieee39.helpers
======================
Package-internal helper functions for building and modifying the IEEE 39-bus
New England test network.

These were originally private helpers (prefixed with ``_``) inside the
monolithic ``build_ieee39_net.py``.  They are now module-level functions
consumed by ``network.ieee39.builder`` and, in the case of
``remove_generators``, by run scripts directly.
"""

from __future__ import annotations

from typing import List, Tuple

import pandapower as pp

from network.ieee39.constants import LINE_LENGTHS_KM
from network.ieee39.meta import IEEE39NetworkMeta


# ---------------------------------------------------------------------------
#  Line-length correction
# ---------------------------------------------------------------------------

def fix_line_lengths(net: pp.pandapowerNet) -> None:
    """Replace default 1-km lengths with real distances, preserving total Z.

    pandapower case39() stores total impedance as per-km values with
    length_km = 1.0.  This function sets the real length and rescales the
    per-km values so that ``per_km * length_km`` (the total impedance seen
    by the power flow) is exactly preserved.

    Lines without a matching entry in ``LINE_LENGTHS_KM`` (e.g. line 29
    which is a generator step-up connection) are left unchanged.
    """
    for li in net.line.index:
        fb_1 = int(net.line.at[li, "from_bus"]) + 1
        tb_1 = int(net.line.at[li, "to_bus"]) + 1
        key = (fb_1, tb_1)
        key_rev = (tb_1, fb_1)

        new_len = LINE_LENGTHS_KM.get(key) or LINE_LENGTHS_KM.get(key_rev)
        if new_len is None:
            continue

        old_len = float(net.line.at[li, "length_km"])
        scale = old_len / new_len   # = 1.0 / new_len

        net.line.at[li, "r_ohm_per_km"] = float(net.line.at[li, "r_ohm_per_km"]) * scale
        net.line.at[li, "x_ohm_per_km"] = float(net.line.at[li, "x_ohm_per_km"]) * scale
        net.line.at[li, "c_nf_per_km"]  = float(net.line.at[li, "c_nf_per_km"]) * scale
        net.line.at[li, "g_us_per_km"]  = float(net.line.at[li, "g_us_per_km"]) * scale
        net.line.at[li, "length_km"]    = new_len


# ---------------------------------------------------------------------------
#  Slack relocation
# ---------------------------------------------------------------------------

def swap_slack_to_bus38(
    net: pp.pandapowerNet,
    ext_grid_vm_pu: float,
) -> int:
    """Replace the case39 ext_grid slack with a distributed-slack-ready gen.

    pandapower case39() places the slack at an ``ext_grid`` on bus 30
    (0-indexed = bus 31 1-indexed).  The IEEE 39-bus standard has the
    slack / equivalent generator at bus 39 (1-indexed) = bus 38 (0-indexed).

    This function converts the network to a fully-distributed-slack layout:

      1. Removes the PV generator at bus 38 (the case39 IEEE G9 slot).
      2. Creates a new PV generator at bus 30 with limits taken from the
         original ext_grid entry.
      3. Drops the ``ext_grid`` and creates a real :class:`gen` at bus 38
         with ``slack=True``.  This gen anchors the voltage-angle
         reference AND participates in the distributed active-power
         slack with a positive ``slack_weight`` (assigned by
         ``build_ieee39_net``).  It has finite Q capability just like
         every other synchronous machine — Q imbalance is no longer
         absorbed silently by an ideal-voltage ext_grid.

    Returns the pandapower index of the newly created generator at bus 30.
    """
    eg_idx = net.ext_grid.index[0]
    old_bus = int(net.ext_grid.at[eg_idx, "bus"])      # 30

    # Save original ext_grid limits for the replacement gen at bus 30
    eg_max_p = float(net.ext_grid.at[eg_idx, "max_p_mw"])
    eg_min_p = float(net.ext_grid.at[eg_idx, "min_p_mw"])
    eg_max_q = float(net.ext_grid.at[eg_idx, "max_q_mvar"])
    eg_min_q = float(net.ext_grid.at[eg_idx, "min_q_mvar"])

    # -- 1. Remove gen at bus 38 -----------------------------------------------
    gen_at_38 = net.gen.index[net.gen["bus"] == 38].tolist()
    if gen_at_38:
        net.gen.drop(index=gen_at_38, inplace=True)

    # -- 2. Fix trafo 1 (hv=5, lv=30) impedance --------------------------------
    # In case39 this trafo has vk=45 % because bus 30 represents the
    # equivalent NY external system behind a high-impedance connection.
    # When we place a real generator at bus 30, the machine-trafo loop
    # will convert this to a 345/10.5 kV step-up.  With vk=45 % that
    # would be unrealistically high (typical gen step-up: 12-15 %).
    # Reset to realistic values before the machine-trafo loop runs.
    trafo_mask = (net.trafo["lv_bus"] == old_bus) | (net.trafo["hv_bus"] == old_bus)
    for ti in net.trafo.index[trafo_mask]:
        net.trafo.at[ti, "vk_percent"] = 12.0
        net.trafo.at[ti, "vkr_percent"] = 0.3
        net.trafo.at[ti, "sn_mva"] = max(eg_max_p * 1.2, 100.0)
        net.trafo.at[ti, "pfe_kw"] = 0.0
        net.trafo.at[ti, "i0_percent"] = 0.0

    # -- 3. Create new PV generator at bus 30 -----------------------------------
    new_gen_idx = pp.create_gen(
        net,
        bus=old_bus,
        p_mw=500.0,
        vm_pu=ext_grid_vm_pu,
        max_p_mw=eg_max_p,
        min_p_mw=eg_min_p,
        max_q_mvar=eg_max_q,
        min_q_mvar=eg_min_q,
        in_service=True,
        name="Gen_bus30 (ex-slack)",
    )

    # -- 4. Replace ext_grid with a slack-enabled gen at bus 38 ----------------
    # The new gen inherits a realistic IEEE G10 nameplate (1000 MW / 500
    # Mvar envelope).  ``slack=True`` anchors θ=0 at bus 38; the nameplate
    # loop in build_ieee39_net assigns its sn_mva and P limits via
    # NAMEPLATE_FACTOR; its ``slack_weight`` is set downstream to its
    # nameplate so it participates in the distributed-slack P allocation
    # like every other machine.
    pp.create_gen(
        net,
        bus=38,
        p_mw=1000.0,
        vm_pu=ext_grid_vm_pu,
        max_p_mw=1000.0, min_p_mw=0.0,
        max_q_mvar=500.0, min_q_mvar=-500.0,
        slack=True,
        in_service=True,
        name="Gen_bus38 (slack anchor, ex-G10)",
    )
    net.ext_grid.drop(index=eg_idx, inplace=True)

    return int(new_gen_idx)


# ---------------------------------------------------------------------------
#  Load helpers
# ---------------------------------------------------------------------------

def get_load_at_bus(net: pp.pandapowerNet, bus: int) -> Tuple[float, float]:
    """Return total (p_mw, q_mvar) at a bus, or (0, 0) if no load."""
    mask = net.load["bus"] == bus
    if not mask.any():
        return 0.0, 0.0
    return (
        float(net.load.loc[mask, "p_mw"].sum()),
        float(net.load.loc[mask, "q_mvar"].sum()),
    )


def delete_loads_at_bus(net: pp.pandapowerNet, bus: int) -> None:
    """Remove all loads connected to a bus."""
    mask = net.load["bus"] == bus
    if mask.any():
        net.load.drop(index=net.load.index[mask], inplace=True)


def reduce_loads_at_bus(
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


# ---------------------------------------------------------------------------
#  Generator removal
# ---------------------------------------------------------------------------

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

    # -- Remove associated machine trafos and terminal buses (if present) ------
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

    # -- Drop the generator rows from net.gen -----------------------------------
    net.gen.drop(index=sorted(requested), inplace=True)

    # -- Rebuild all gen-related meta fields ------------------------------------
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
        gen_grid_bus_indices  = tuple(new_gen_grid_bus_indices),
        machine_trafo_indices = tuple(new_machine_trafo_indices),
        machine_trafo_gen_map = tuple(new_machine_trafo_gen_map),
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
