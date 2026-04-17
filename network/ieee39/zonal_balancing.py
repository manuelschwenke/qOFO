"""
network/ieee39/zonal_balancing.py
=================================
Pre-compute per-zone generator active-power dispatch from residual load
(total load minus non-controllable sgen) for the IEEE 39-bus network, and
apply the dispatched P to ``net.gen`` at each simulation timestep.

Moved here from ``core/profiles.py`` during the April 2026 refactor because
both functions are IEEE 39-bus specific (they assume an ``ext_grid`` slack
and a ``net.gen`` table keyed by the zone partition).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandapower as pp


def compute_zonal_gen_dispatch(
    net: pp.pandapowerNet,
    profiles: pd.DataFrame,
    zone_map: Dict[int, List[int]],
    *,
    gen_p_min_mw: Union[float, Dict[int, float]] = 0.0,
) -> pd.DataFrame:
    """Pre-compute generator active-power dispatch from per-zone residual load.

    For every timestep in *profiles* the function computes each zone's
    residual (``sum(loads) - sum(sgens)``) and distributes it among the
    zone's generators proportionally to their rated capacity (``sn_mva``).
    An iterative clipping pass redistributes any shortfall from generators
    hitting their limits.

    The ``ext_grid`` slack is excluded -- it absorbs only network losses
    plus residual zone imbalance that flows via tie lines.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network after :func:`network.ieee39.build.build_ieee39_net` and
        :func:`network.ieee39.hv_networks.add_hv_networks`.  Must already
        have ``base_p_mw`` columns (call
        :func:`core.profiles.snapshot_base_values` first).
    profiles : pd.DataFrame
        Output of :func:`core.profiles.load_profiles`.
    zone_map : dict[int, list[int]]
        Zone-id -> list of bus indices (from
        :func:`network.zone_partition.fixed_zone_partition_ieee39`).
    gen_p_min_mw : float or dict[int, float]
        Minimum generator output [MW].  If a float, applies uniformly.
        If a dict, maps ``net.gen`` index -> P_min [MW]; gens not in the
        dict default to 0.0.

    Returns
    -------
    pd.DataFrame
        Index = profile timestamps, columns = generator indices,
        values = dispatched P_mw.
    """
    if "base_p_mw" not in net.load.columns:
        raise RuntimeError(
            "Call snapshot_base_values(net) before compute_zonal_gen_dispatch."
        )

    if isinstance(gen_p_min_mw, dict):
        _p_min_map: Dict[int, float] = gen_p_min_mw
    else:
        _p_min_map = {}
    _p_min_scalar: float = gen_p_min_mw if isinstance(gen_p_min_mw, (int, float)) else 0.0

    def _get_p_min(gi: int) -> float:
        return _p_min_map.get(gi, _p_min_scalar)

    bus_zone: Dict[int, int] = {}
    for z, buses in zone_map.items():
        for b in buses:
            bus_zone[b] = z

    ZoneLoadInfo = List[Tuple[float, Optional[str]]]
    zone_loads: Dict[int, ZoneLoadInfo] = {z: [] for z in zone_map}

    for li in net.load.index:
        bus = int(net.load.at[li, "bus"])
        z = bus_zone.get(bus)
        if z is None:
            continue
        base_p = float(net.load.at[li, "base_p_mw"])
        prof = net.load.at[li, "profile_p"] if "profile_p" in net.load.columns else None
        if pd.isna(prof):
            prof = None
        zone_loads[z].append((base_p, prof))

    ZoneSgenInfo = List[Tuple[float, Optional[str]]]
    zone_sgens: Dict[int, ZoneSgenInfo] = {z: [] for z in zone_map}

    for si in net.sgen.index:
        name = str(net.sgen.at[si, "name"])
        if name.startswith("BOUND_"):
            continue
        bus = int(net.sgen.at[si, "bus"])
        z = bus_zone.get(bus)
        if z is None:
            continue
        base_p = float(net.sgen.at[si, "base_p_mw"])
        prof = net.sgen.at[si, "profile"] if "profile" in net.sgen.columns else None
        if pd.isna(prof):
            prof = None
        zone_sgens[z].append((base_p, prof))

    ext_grid_buses = set(int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index)
    zone_gens: Dict[int, List[int]] = {z: [] for z in zone_map}

    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        if bus in ext_grid_buses:
            continue
        z = bus_zone.get(bus)
        if z is None:
            continue
        zone_gens[z].append(int(gi))

    gen_p_max: Dict[int, float] = {}
    for gi in net.gen.index:
        sn = net.gen.at[gi, "sn_mva"] if "sn_mva" in net.gen.columns else np.nan
        if pd.isna(sn):
            sn = net.gen.at[gi, "max_p_mw"] if "max_p_mw" in net.gen.columns else np.nan
        if pd.isna(sn):
            sn = float(net.gen.at[gi, "p_mw"]) * 2.0
        gen_p_max[int(gi)] = float(sn)

    all_gen_indices = sorted(
        gi for gens in zone_gens.values() for gi in gens
    )
    dispatch = pd.DataFrame(
        index=profiles.index,
        columns=all_gen_indices,
        dtype=np.float64,
    )

    zone_gen_cap: Dict[int, float] = {
        z: sum(gen_p_max.get(gi, 0.0) for gi in zone_gens[z])
        for z in zone_map
    }

    for idx in profiles.index:
        row = profiles.loc[idx]

        for z in zone_map:
            zone_load_total = 0.0
            for base_p, prof in zone_loads[z]:
                scale = float(row[prof]) if (prof and prof in row.index and not np.isnan(row[prof])) else 1.0
                zone_load_total += base_p * scale

            zone_sgen_total = 0.0
            for base_p, prof in zone_sgens[z]:
                scale = float(row[prof]) if (prof and prof in row.index and not np.isnan(row[prof])) else 1.0
                zone_sgen_total += base_p * scale

            zone_residual = zone_load_total - zone_sgen_total

            if not zone_gens[z]:
                continue

            zone_p_min_total = sum(_get_p_min(gi) for gi in zone_gens[z])
            if zone_residual < zone_p_min_total and idx == profiles.index[0]:
                import warnings
                warnings.warn(
                    f"Zone {z}: residual load {zone_residual:.1f} MW < "
                    f"total P_min {zone_p_min_total:.1f} MW at first "
                    f"timestep -- generators will be dispatched at P_min, "
                    f"excess absorbed by slack.",
                    stacklevel=2,
                )

            cap = zone_gen_cap[z]

            if cap > 0.0 and zone_residual > 0.0:
                for gi in zone_gens[z]:
                    share = gen_p_max.get(gi, 0.0) / cap
                    dispatch.at[idx, gi] = np.clip(
                        zone_residual * share,
                        _get_p_min(gi), gen_p_max.get(gi, 9999.0),
                    )
            else:
                for gi in zone_gens[z]:
                    dispatch.at[idx, gi] = _get_p_min(gi)

            for _ in range(5):
                total_dispatched = sum(dispatch.at[idx, gi] for gi in zone_gens[z])
                shortfall = zone_residual - total_dispatched
                if abs(shortfall) < 0.01:
                    break

                if shortfall > 0:
                    free = [gi for gi in zone_gens[z]
                            if dispatch.at[idx, gi] < gen_p_max.get(gi, 9999.0) - 0.01]
                else:
                    free = [gi for gi in zone_gens[z]
                            if dispatch.at[idx, gi] > _get_p_min(gi) + 0.01]

                if not free:
                    break
                free_cap = sum(gen_p_max.get(gi, 0.0) for gi in free)
                if free_cap < 0.01:
                    break

                for gi in free:
                    new_p = dispatch.at[idx, gi] + shortfall * gen_p_max.get(gi, 0.0) / free_cap
                    dispatch.at[idx, gi] = np.clip(
                        new_p, _get_p_min(gi), gen_p_max.get(gi, 9999.0),
                    )

    print()
    print("=" * 72)
    print("  Zonal Generator Dispatch (per-zone residual-load balancing)")
    print("=" * 72)
    for z in sorted(zone_map.keys()):
        n_loads = len(zone_loads[z])
        n_sgens = len(zone_sgens[z])
        n_gens = len(zone_gens[z])
        if zone_gens[z]:
            gen_cols = zone_gens[z]
            p_min_val = dispatch[gen_cols].min().min()
            p_max_val = dispatch[gen_cols].max().max()
            p_mean = dispatch[gen_cols].mean().mean()
            print(f"  Zone {z}: {n_loads} loads, {n_sgens} sgens, {n_gens} gens"
                  f"  ->  P_gen range [{p_min_val:.1f}, {p_max_val:.1f}] MW, mean {p_mean:.1f} MW")
        else:
            print(f"  Zone {z}: {n_loads} loads, {n_sgens} sgens, {n_gens} gens"
                  f" (no dispatch -- slack covers via tie lines)")
    print("-" * 72)
    total_disp = dispatch[all_gen_indices].sum(axis=1)
    total_cap = sum(gen_p_max.get(gi, 0) for gi in all_gen_indices)
    print(f"  System gen dispatch: [{total_disp.min():.1f}, {total_disp.max():.1f}] MW, "
          f"mean {total_disp.mean():.1f} MW")
    print(f"  Generators: {len(all_gen_indices)}, total Pmax: {total_cap:.1f} MW")
    print("  (Slack absorbs losses + unresolved zone imbalances)")
    print("=" * 72)
    print()

    return dispatch


def apply_gen_dispatch(
    net: pp.pandapowerNet,
    dispatch: pd.DataFrame,
    t: datetime,
) -> None:
    """Write pre-computed generator P dispatch for time *t* into ``net.gen``.

    Parameters
    ----------
    net : pp.pandapowerNet
    dispatch : pd.DataFrame
        Output of :func:`compute_zonal_gen_dispatch`.
    t : datetime
        Current simulation time.
    """
    idx = dispatch.index.get_indexer([t], method="nearest")[0]
    row = dispatch.iloc[idx]

    for gi in dispatch.columns:
        if gi in net.gen.index:
            net.gen.at[gi, "p_mw"] = float(row[gi])
