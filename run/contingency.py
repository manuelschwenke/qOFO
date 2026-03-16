#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/contingency.py
==================
Contingency application helper for the cascaded TSO-DSO OFO simulation.
"""

from .records import ContingencyEvent


def _apply_contingency(
    net,
    ev: ContingencyEvent,
    verbose: int,
) -> tuple:
    """
    Apply a single contingency event to the pandapower network.

    Returns a human-readable description string for logging.
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
        net.gen.at[ev.element_index, "in_service"] = in_service
        name = net.gen.at[ev.element_index, "name"]
        bus = int(net.gen.at[ev.element_index, "bus"])
        desc = f"{tag} gen {ev.element_index} ({name})"
        short_label = f"{short_tag} Gen {bus}"
    elif ev.element_type == "ext_grid":
        old_setpoint = net.ext_grid.at[ev.element_index, "vm_pu"]
        net.ext_grid.at[ev.element_index, "vm_pu"] = ev.new_setpoint
        name = net.ext_grid.at[ev.element_index, "name"]
        bus = int(net.ext_grid.at[ev.element_index, "bus"])
        desc = f"SETPOINT-CHANGE ext_grid {ev.element_index} ({name}): old {old_setpoint} -> new {ev.new_setpoint}"
        short_label = f"[S] ExtGrid {bus}"
    else:
        raise ValueError(
            f"Unknown element_type '{ev.element_type}' (expected 'line' or 'gen')"
        )

    if verbose > 0:
        print(f"\n  {'=' * 60}")
        print(f"  *** CONTINGENCY min {ev.minute}: {desc}")
        print(f"  {'=' * 60}\n")
    return desc, short_label
