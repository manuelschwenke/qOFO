#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/contingency.py
==================
Contingency application helper for the cascaded TSO-DSO OFO simulation.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List

import numpy as np
import pandapower as pp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .records import ContingencyEvent


# ---------------------------------------------------------------------------
#  Pre-create dormant loads for load-contingency events
# ---------------------------------------------------------------------------

def prepare_load_contingencies(
    net: pp.pandapowerNet,
    contingencies: List[ContingencyEvent],
    verbose: int = 0,
) -> None:
    """
    Pre-create pandapower loads for ``element_type="load"`` contingency events.

    Each unique ``(bus, p_mw, q_mvar)`` triple among ``"connect"`` events
    produces one load row in ``net.load`` with ``in_service=False``.  The
    assigned load index is written back into ``element_index`` on **all**
    matching events (both ``"connect"`` and ``"trip"``).

    The loads are created without ``profile_p`` / ``profile_q`` so that
    :func:`apply_profiles` never scales them — they behave as constant loads.
    ``base_p_mw`` and ``base_q_mvar`` are set to **0** so that
    :func:`compute_zonal_gen_dispatch` (which is pre-computed) does not
    account for them.  After connection the extra demand is absorbed by
    the distributed slack, analogous to any other contingency disturbance.

    Must be called **after** :func:`snapshot_base_values`.
    """
    load_events = [
        ev for ev in contingencies if ev.element_type == "load"
    ]
    if not load_events:
        return

    # ── Validate and group connect events by (bus, p_mw, q_mvar) ──────────
    connect_events = [ev for ev in load_events if ev.action == "connect"]
    key_to_index: dict[tuple[int, float, float], int] = {}

    for ev in connect_events:
        if ev.bus is None:
            raise ValueError(
                f"Load contingency at minute {ev.minute} has bus=None"
            )
        if ev.bus not in net.bus.index:
            raise ValueError(
                f"Load contingency at minute {ev.minute}: "
                f"bus {ev.bus} not in net.bus.index"
            )
        if math.isnan(ev.p_mw) or math.isnan(ev.q_mvar):
            raise ValueError(
                f"Load contingency at minute {ev.minute}: "
                f"p_mw and q_mvar must not be NaN"
            )

        key = (ev.bus, ev.p_mw, ev.q_mvar)
        if key not in key_to_index:
            # Create the dormant load
            lidx = pp.create_load(
                net,
                bus=ev.bus,
                p_mw=ev.p_mw,
                q_mvar=ev.q_mvar,
                in_service=False,
                name=f"CONTINGENCY_LOAD|bus{ev.bus}",
            )
            # Set base values to 0 so pre-computed gen dispatch is unaffected
            if "base_p_mw" in net.load.columns:
                net.load.at[lidx, "base_p_mw"] = 0.0
            if "base_q_mvar" in net.load.columns:
                net.load.at[lidx, "base_q_mvar"] = 0.0
            key_to_index[key] = lidx
            if verbose > 0:
                print(f"  Created dormant contingency load idx={lidx} "
                      f"at bus {ev.bus}: {ev.p_mw:.1f} MW, "
                      f"{ev.q_mvar:.1f} Mvar")

        ev.element_index = key_to_index[key]

    # ── Assign element_index to trip/restore events ───────────────────────
    for ev in load_events:
        if ev.action == "connect":
            continue  # already handled
        key = (ev.bus, ev.p_mw, ev.q_mvar)
        if key not in key_to_index:
            raise ValueError(
                f"Load contingency at minute {ev.minute}: "
                f"action='{ev.action}' for (bus={ev.bus}, "
                f"p_mw={ev.p_mw}, q_mvar={ev.q_mvar}) but no "
                f"matching 'connect' event was defined"
            )
        ev.element_index = key_to_index[key]


# ---------------------------------------------------------------------------
#  Apply a single contingency event
# ---------------------------------------------------------------------------

def _apply_contingency(
    net,
    ev: ContingencyEvent,
    verbose: int,
    gen_trafo_map: dict[int, int] | None = None,
) -> tuple:
    """
    Apply a single contingency event to the pandapower network.

    When a generator trips/restores and ``gen_trafo_map`` is provided,
    the associated machine transformer is also tripped/restored (realistic
    behaviour: breaker opens on both sides of the unit transformer).

    Parameters
    ----------
    gen_trafo_map : dict[int, int] | None
        Mapping from ``net.gen`` index → ``net.trafo`` index of the
        machine transformer.  Built from ``meta.machine_trafo_gen_map``
        and ``meta.machine_trafo_indices``.

    Returns a (description, short_label) tuple for logging.
    """
    trip = ev.action == "trip"
    in_service = not trip
    tag = "TRIP" if trip else "RESTORE"
    short_tag = "[T]" if trip else "[R]"

    if ev.element_type == "line":
        net.line.at[ev.element_index, "in_service"] = in_service
        name = net.line.at[ev.element_index, "name"]
        from_bus = int(net.line.at[ev.element_index, "from_bus"])
        to_bus = int(net.line.at[ev.element_index, "to_bus"])
        desc = f"{tag} line {ev.element_index} ({name})"
        short_label = f"{short_tag} Line {from_bus}-{to_bus}"
    elif ev.element_type == "gen":
        if ev.element_index not in net.gen.index:
            if verbose > 0:
                print(f"  WARNING: gen {ev.element_index} not in network "
                      f"(removed by scenario?) — skipping contingency")
            return
        net.gen.at[ev.element_index, "in_service"] = in_service
        name = net.gen.at[ev.element_index, "name"]
        bus = int(net.gen.at[ev.element_index, "bus"])
        desc = f"{tag} gen {ev.element_index} ({name})"
        short_label = f"{short_tag} Gen {bus}"

        # Also trip/restore the associated machine transformer
        if gen_trafo_map and ev.element_index in gen_trafo_map:
            t_idx = gen_trafo_map[ev.element_index]
            net.trafo.at[t_idx, "in_service"] = in_service
            desc += f" + machine trafo {t_idx}"
            if verbose > 0:
                print(f"    → Also {'tripping' if trip else 'restoring'} "
                      f"machine transformer {t_idx}")
    elif ev.element_type == "ext_grid":
        old_setpoint = net.ext_grid.at[ev.element_index, "vm_pu"]
        net.ext_grid.at[ev.element_index, "vm_pu"] = ev.new_setpoint
        name = net.ext_grid.at[ev.element_index, "name"]
        bus = int(net.ext_grid.at[ev.element_index, "bus"])
        desc = f"SETPOINT-CHANGE ext_grid {ev.element_index} ({name}): old {old_setpoint} -> new {ev.new_setpoint}"
        short_label = f"[S] ExtGrid {bus}"
    elif ev.element_type == "load":
        # "connect" is functionally identical to "restore" (in_service=True)
        connect = ev.action == "connect"
        in_service = connect or (ev.action == "restore")
        net.load.at[ev.element_index, "in_service"] = in_service
        name = net.load.at[ev.element_index, "name"]
        bus = int(net.load.at[ev.element_index, "bus"])
        p = float(net.load.at[ev.element_index, "p_mw"])
        q = float(net.load.at[ev.element_index, "q_mvar"])
        if connect:
            tag_l, short_tag_l = "CONNECT", "[C]"
        elif ev.action == "restore":
            tag_l, short_tag_l = "RESTORE", "[R]"
        else:
            tag_l, short_tag_l = "SHED", "[S]"
        desc = (f"{tag_l} load {ev.element_index} ({name}, "
                f"{p:.1f} MW + {q:.1f} Mvar @ bus {bus})")
        short_label = f"{short_tag_l} Load bus {bus}"
    else:
        raise ValueError(
            f"Unknown element_type '{ev.element_type}' "
            f"(expected 'line', 'gen', 'ext_grid', or 'load')"
        )

    if verbose > 0:
        if ev.time_s is not None:
            t_label = f"t={ev.time_s:.0f}s"
        else:
            t_label = f"min {ev.minute}"
        print(f"\n  {'=' * 60}")
        print(f"  *** CONTINGENCY {t_label}: {desc}")
        print(f"  {'=' * 60}\n")
    return desc, short_label
