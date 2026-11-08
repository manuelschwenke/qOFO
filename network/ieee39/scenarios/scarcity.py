"""
Scenario family: scarcity redesign candidates (2026-11-08)
==========================================================

The transmission-side wind replacement of ``wind_replace`` with optional
345 kV lines not built.  Candidate ``*_no939``: line 9-39 (IEEE numbering) is
absent, so zone 2 is no longer tied directly to the network equivalent G1 at
IEEE bus 39; its reactive shortfalls then have to be met inside the zone or
over the longer paths through zones 1 and 3 (author 2026-11-08: screen before
adoption; the E2 reference scenario uses the same line as an outage event).

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""
from __future__ import annotations

from typing import Iterable, Tuple

import pandapower as pp

from network.ieee39.scenarios.wind_replace import apply_wind_replace


def apply_scarcity(net, meta, *, ext_grid_vm_pu=1.03, ts_wind_layout="replace",
                   lines_not_built: Iterable[Tuple[int, int]] = (), **kwargs):
    """``apply_wind_replace``, then take the listed lines (IEEE bus pairs) out
    of service.  Indices are kept, so every other element index is unchanged."""
    net, meta = apply_wind_replace(net, meta, ext_grid_vm_pu=ext_grid_vm_pu,
                                   ts_wind_layout=ts_wind_layout, **kwargs)
    pairs = {tuple(sorted(p)) for p in lines_not_built}
    found = set()
    for li in net.line.index:
        p = tuple(sorted((int(net.line.at[li, "from_bus"]) + 1,
                          int(net.line.at[li, "to_bus"]) + 1)))
        if p in pairs and str(net.line.at[li, "subnet"]) == "TN":
            net.line.at[li, "in_service"] = False
            found.add(p)
    missing = pairs - found
    if missing:
        raise ValueError(f"lines not found: {sorted(missing)}")
    pp.runpp(net, run_control=False, calculate_voltage_angles=True)
    net["lines_not_built"] = sorted(found)
    return net, meta
