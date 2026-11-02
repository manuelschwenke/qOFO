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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pandapower as pp


def compute_zonal_gen_dispatch(
    net: pp.pandapowerNet,
    profiles: pd.DataFrame,
    zone_map: Dict[int, List[int]],
    *,
    gen_p_min_mw: Union[float, Dict[int, float]] = 0.0,
    gen_p_cap_factor: float = 0.85,
) -> pd.DataFrame:
    """Pre-compute generator active-power dispatch from per-zone residual load.

    For every timestep in *profiles* the function computes each zone's
    residual (``sum(loads) - sum(sgens)``) and distributes it among the
    zone's generators proportionally to their rated capacity (``sn_mva``).
    An iterative clipping pass redistributes any shortfall from generators
    hitting their limits.

    To keep every gen inside the synchronous-machine capability curve
    (Milano §12.2.1) with usable Q range, the per-gen P upper bound used
    for dispatch is ``gen_p_cap_factor * sn_mva`` (default 0.85, leaving a
    stator-circle Q margin of ≈ 0.53 · sn_mva at the cap).  When a zone's
    residual exceeds the sum of those caps inside the zone, a cross-zone
    *spill* pass redistributes the excess to gens in other zones
    proportionally to their remaining headroom — instead of letting the
    slack at bus 38 absorb it.

    The ``ext_grid`` slack still absorbs network losses and any residue
    that exceeds the system-wide soft-capped capacity.

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
    gen_p_cap_factor : float
        α ∈ (0, 1].  Per-gen dispatch upper bound is α · sn_mva.  Default
        0.85 corresponds to a stator-limited cos φ = 0.85 operating point.
        Lower α → more Q headroom, more cross-zone spill.

    Returns
    -------
    pd.DataFrame
        Index = profile timestamps, columns = generator indices,
        values = dispatched P_mw.
    """
    if not (0.0 < gen_p_cap_factor <= 1.0):
        raise ValueError(
            f"gen_p_cap_factor must lie in (0, 1]; got {gen_p_cap_factor}"
        )
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

    # Gens carrying a ``profile`` column (exogenous wind/PV P) are
    # treated as exogenous P injectors (sgen-equivalent) for zone
    # residual computation — they MUST NOT take part in zonal dispatch
    # because their P is the wind/PV resource, not a dispatchable
    # quantity.  Synchronous-machine gens have no ``profile`` and fall
    # through to the dispatched-gen list below.
    if "profile" in net.gen.columns and "base_p_mw" in net.gen.columns:
        for gi in net.gen.index:
            prof = net.gen.at[gi, "profile"]
            if pd.isna(prof):
                continue
            bus = int(net.gen.at[gi, "bus"])
            z = bus_zone.get(bus)
            if z is None:
                continue
            base_p = float(net.gen.at[gi, "base_p_mw"])
            zone_sgens[z].append((base_p, str(prof)))

    ext_grid_buses = set(int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index)
    zone_gens: Dict[int, List[int]] = {z: [] for z in zone_map}

    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        if bus in ext_grid_buses:
            continue
        # Skip promoted DER gens — they are wind/PV injectors, not
        # dispatchable.  Their P is set by apply_profiles.
        if "profile" in net.gen.columns:
            prof = net.gen.at[gi, "profile"]
            if not pd.isna(prof):
                continue
        z = bus_zone.get(bus)
        if z is None:
            continue
        zone_gens[z].append(int(gi))

    gen_p_max: Dict[int, float] = {}
    for gi in net.gen.index:
        sn = net.gen.at[gi, "sn_mva"]
        assert not pd.isna(sn) and sn > 0, (
            f"gen {gi} has invalid sn_mva={sn}; nameplate must be set in "
            f"build_ieee39_net (see constants.NAMEPLATE_FACTOR)"
        )
        gen_p_max[int(gi)] = float(sn) * gen_p_cap_factor

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

    n_steps = len(profiles.index)
    zone_residual_t: Dict[int, np.ndarray] = {
        z: np.zeros(n_steps, dtype=np.float64) for z in zone_map
    }
    spill_in_t: Dict[int, np.ndarray] = {
        z: np.zeros(n_steps, dtype=np.float64) for z in zone_map
    }
    spill_out_t: Dict[int, np.ndarray] = {
        z: np.zeros(n_steps, dtype=np.float64) for z in zone_map
    }

    for t_i, idx in enumerate(profiles.index):
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
            zone_residual_t[z][t_i] = zone_residual

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

        # ── Cross-zone P spill ────────────────────────────────────────────
        # If any zone's residual could not be covered within its own gens
        # (because each gen is soft-capped at α · sn_mva to preserve Q room),
        # redistribute the unmet portion to gens system-wide proportionally
        # to their remaining headroom.  Hard cap per receiving gen is the
        # same soft cap (α · sn_mva), so the capability-curve invariant
        # holds for every gen — slack still absorbs anything above the
        # system-wide soft-capped capacity.
        deficit_zones = {
            z: zone_residual_t[z][t_i] - sum(dispatch.at[idx, gi] for gi in zone_gens[z])
            for z in zone_map
            if zone_gens[z]
        }
        total_deficit = sum(d for d in deficit_zones.values() if d > 0.01)
        if total_deficit > 0.01:
            for _ in range(5):
                free_gens: List[Tuple[int, float]] = []
                for z in zone_map:
                    for gi in zone_gens[z]:
                        head = gen_p_max[gi] - dispatch.at[idx, gi]
                        if head > 0.01:
                            free_gens.append((gi, head))
                free_total = sum(h for _, h in free_gens)
                if free_total < 0.01:
                    break
                add_total = min(total_deficit, free_total)
                for gi, head in free_gens:
                    add = add_total * head / free_total
                    if add > head:
                        add = head
                    dispatch.at[idx, gi] = min(
                        gen_p_max[gi], dispatch.at[idx, gi] + add
                    )
                # Remaining deficit must be measured SYSTEM-WIDE, not as a sum of
                # per-zone (residual_z - dispatch_z) terms.  The spill above places
                # generation in OTHER zones (chiefly the high-capacity slack gen) to
                # cover a deficit zone; a per-zone measure never registers that
                # coverage, so it re-adds the same deficit on every iteration and the
                # slack/zone-1 gen ends up dispatched several times the real residual
                # (the root cause of high-load over-dispatch -> 200%+ line overloads
                # and PF collapse under distributed_slack).
                total_dispatched_all = sum(
                    dispatch.at[idx, gi]
                    for zz in zone_map for gi in zone_gens[zz]
                )
                total_residual_all = sum(
                    zone_residual_t[zz][t_i] for zz in zone_map
                )
                total_deficit = max(0.0, total_residual_all - total_dispatched_all)
                if total_deficit < 0.01:
                    break

        # Bookkeeping: spill_in = zone exported P (D > R), spill_out = zone
        # residual unmet by spill (slack residue, D < R).
        for z in zone_map:
            if not zone_gens[z]:
                continue
            d = sum(dispatch.at[idx, gi] for gi in zone_gens[z])
            r = zone_residual_t[z][t_i]
            if d > r + 0.01:
                spill_in_t[z][t_i] = d - r
            elif r > d + 0.01:
                spill_out_t[z][t_i] = r - d

    print()
    print("=" * 72)
    print("  Zonal Generator Dispatch (per-zone residual-load balancing)")
    print(f"  Per-gen P cap: alpha * sn_mva   with alpha = {gen_p_cap_factor:.2f}")
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
            si_mean = float(spill_in_t[z].mean())
            si_peak = float(spill_in_t[z].max())
            so_mean = float(spill_out_t[z].mean())
            so_peak = float(spill_out_t[z].max())
            print(f"  Zone {z}: {n_loads} loads, {n_sgens} sgens, {n_gens} gens"
                  f"  ->  P_gen range [{p_min_val:.1f}, {p_max_val:.1f}] MW, mean {p_mean:.1f} MW")
            print(f"          spill_in : mean {si_mean:6.1f} MW, peak {si_peak:6.1f} MW   "
                  f"(zone exports to cover other zones)")
            print(f"          spill_out: mean {so_mean:6.1f} MW, peak {so_peak:6.1f} MW   "
                  f"(unmet residual absorbed by slack)")
        else:
            print(f"  Zone {z}: {n_loads} loads, {n_sgens} sgens, {n_gens} gens"
                  f" (no dispatch -- slack covers via tie lines)")
    print("-" * 72)
    total_disp = dispatch[all_gen_indices].sum(axis=1)
    total_cap = sum(gen_p_max.get(gi, 0) for gi in all_gen_indices)
    sys_spill_per_t = sum(spill_in_t[z] for z in zone_map)
    if sys_spill_per_t.max() > 0.01:
        worst_i = int(np.argmax(sys_spill_per_t))
        worst_t = profiles.index[worst_i]
        print(f"  System gen dispatch: [{total_disp.min():.1f}, {total_disp.max():.1f}] MW, "
              f"mean {total_disp.mean():.1f} MW")
        print(f"  Generators: {len(all_gen_indices)}, total Pmax (soft): {total_cap:.1f} MW")
        print(f"  Worst-case spill timestep: {worst_t}  total cross-zone spill = "
              f"{sys_spill_per_t[worst_i]:.1f} MW")
    else:
        print(f"  System gen dispatch: [{total_disp.min():.1f}, {total_disp.max():.1f}] MW, "
              f"mean {total_disp.mean():.1f} MW")
        print(f"  Generators: {len(all_gen_indices)}, total Pmax (soft): {total_cap:.1f} MW")
        print("  No cross-zone spill triggered.")
    print("  (Slack absorbs losses + unresolved zone imbalances)")
    print("=" * 72)
    print()

    return dispatch


def _zone_residual_series(
    net: pp.pandapowerNet,
    profiles: pd.DataFrame,
    zone_map: Dict[int, List[int]],
) -> Dict[int, np.ndarray]:
    """Per-zone residual load ``sum(loads) - sum(sgens)`` [MW] at every
    timestep of *profiles* -- the same accounting as
    :func:`compute_zonal_gen_dispatch` (``BOUND_`` sgens skipped, profile gens
    counted as exogenous injectors)."""
    bus_zone = {b: z for z, buses in zone_map.items() for b in buses}
    terms: Dict[int, List[Tuple[float, Optional[str], float]]] = {z: [] for z in zone_map}

    def _add(bus, base_p, prof, sign):
        z = bus_zone.get(int(bus))
        if z is None:
            return
        terms[z].append((float(base_p), None if pd.isna(prof) else str(prof), sign))

    for li in net.load.index:
        _add(net.load.at[li, "bus"], net.load.at[li, "base_p_mw"],
             net.load.at[li, "profile_p"] if "profile_p" in net.load.columns else None, +1.0)
    for si in net.sgen.index:
        if str(net.sgen.at[si, "name"]).startswith("BOUND_"):
            continue
        _add(net.sgen.at[si, "bus"], net.sgen.at[si, "base_p_mw"],
             net.sgen.at[si, "profile"] if "profile" in net.sgen.columns else None, -1.0)
    if "profile" in net.gen.columns and "base_p_mw" in net.gen.columns:
        for gi in net.gen.index:
            if not pd.isna(net.gen.at[gi, "profile"]):
                _add(net.gen.at[gi, "bus"], net.gen.at[gi, "base_p_mw"],
                     net.gen.at[gi, "profile"], -1.0)

    out = {z: np.zeros(len(profiles.index)) for z in zone_map}
    for z, items in terms.items():
        for base_p, prof, sign in items:
            if prof and prof in profiles.columns:
                scale = profiles[prof].to_numpy(dtype=float)
                scale = np.where(np.isnan(scale), 1.0, scale)
            else:
                scale = 1.0
            out[z] += sign * base_p * scale
    return out


def compute_merit_order_dispatch(
    net: pp.pandapowerNet,
    profiles: pd.DataFrame,
    zone_map: Dict[int, List[int]],
    *,
    min_stable_frac: Mapping[str, float],
    merit_order: Sequence[str],
    gen_p_cap_factor: float = 0.85,
) -> Tuple[pd.DataFrame, Dict[int, bool]]:
    """Merit-order dispatch with unit commitment over the window of *profiles*.

    Per zone, the dispatchable machines (``net.gen`` without a ``profile`` and
    without the ``slack`` flag) are taken in *merit_order* of their ``type``.
    A machine is committed while the zone's peak residual over the window,
    less the caps (``gen_p_cap_factor * sn_mva``) of the machines committed
    before it, is at least its technical minimum
    ``min_stable_frac[type] * sn_mva``; otherwise it is skipped and the next
    one is tried.  The commitment holds for the whole window (unit commitment
    acts on hourly to daily time scales, longer than one episode).

    At every timestep the committed machines of a zone are loaded in merit
    order between their minimum and their cap.  A zone deficit goes to the
    headroom of committed machines elsewhere, again in merit order; whatever
    remains -- positive or negative -- is the import of the network
    equivalent (the slack machine, always committed).

    Returns
    -------
    dispatch : pd.DataFrame
        Index = profile timestamps, columns = every ``net.gen`` index that
        :func:`compute_zonal_gen_dispatch` would dispatch (decommitted machines
        at 0 MW), values = P [MW].
    committed : dict[int, bool]
        Commitment of every dispatchable machine and of the equivalent.
    """
    rank = {t: i for i, t in enumerate(merit_order)}
    bus_zone = {b: z for z, buses in zone_map.items() for b in buses}
    ext_grid_buses = {int(net.ext_grid.at[e, "bus"]) for e in net.ext_grid.index}

    slack: List[int] = []
    units: Dict[int, List[int]] = {z: [] for z in zone_map}
    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        if bus in ext_grid_buses or bus_zone.get(bus) is None:
            continue
        if "profile" in net.gen.columns and not pd.isna(net.gen.at[gi, "profile"]):
            continue
        if bool(net.gen.at[gi, "slack"]) if "slack" in net.gen.columns else False:
            slack.append(int(gi))
            continue
        gtype = str(net.gen.at[gi, "type"])
        if gtype not in rank or gtype not in min_stable_frac:
            raise ValueError(f"gen {gi} ({net.gen.at[gi, 'name']}) has type {gtype!r} "
                             f"without merit rank or technical minimum")
        units[bus_zone[bus]].append(int(gi))
    if len(slack) != 1:
        raise ValueError(f"expected one network-equivalent (slack) machine, found {slack}")
    eq = slack[0]

    sn = {int(g): float(net.gen.at[g, "sn_mva"]) for g in net.gen.index}
    cap = {g: gen_p_cap_factor * sn[g] for g in sn}
    pmin = {g: float(min_stable_frac[str(net.gen.at[g, "type"])]) * sn[g]
            for z in units for g in units[z]}
    for z in units:
        units[z].sort(key=lambda g: (rank[str(net.gen.at[g, "type"])], g))

    residual = _zone_residual_series(net, profiles, zone_map)

    committed: Dict[int, bool] = {eq: True}
    on: Dict[int, List[int]] = {}
    for z, gens in units.items():
        need = float(np.max(residual[z])) if len(residual[z]) else 0.0
        on[z] = []
        for g in gens:
            ok = need >= pmin[g]
            committed[g] = ok
            if ok:
                on[z].append(g)
                need -= cap[g]
    order_all = sorted((g for z in on for g in on[z]),
                       key=lambda g: (rank[str(net.gen.at[g, "type"])], g))

    cols = sorted([g for z in units for g in units[z]] + [eq])
    dispatch = pd.DataFrame(0.0, index=profiles.index, columns=cols, dtype=np.float64)
    for t_i, idx in enumerate(profiles.index):
        p: Dict[int, float] = {}
        deficit = 0.0
        for z in zone_map:
            # Every committed machine runs at least at its technical minimum;
            # only the residual above the minima is loaded in merit order.
            # (Until 2026-10-31 ~09:00 the first machine was loaded to its cap
            # before the later machines' minima were honoured, so a falling
            # residual produced a zone surplus while hydro stayed at its cap.)
            rest = float(residual[z][t_i])
            for g in on.get(z, []):
                p[g] = pmin[g]
                rest -= pmin[g]
            for g in on.get(z, []):
                add = min(max(rest, 0.0), cap[g] - p[g])
                p[g] += add
                rest -= add
            deficit += rest
        if deficit > 0.0:
            for g in order_all:
                add = min(deficit, cap[g] - p[g])
                if add > 0.0:
                    p[g] += add
                    deficit -= add
        p[eq] = deficit
        for g, val in p.items():
            dispatch.at[idx, g] = val

    print()
    print("=" * 72)
    print("  Merit-order dispatch with unit commitment (window of the run)")
    print(f"  order {list(merit_order)} -> equivalent; P cap {gen_p_cap_factor:.2f} S_n")
    for z in sorted(zone_map):
        if not units[z]:
            continue
        desc = ", ".join(
            f"{net.gen.at[g, 'name']} ({net.gen.at[g, 'type']}, "
            f"{'ON' if committed[g] else 'off'})" for g in units[z])
        print(f"  Zone {z}: peak residual {np.max(residual[z]):7.1f} MW | {desc}")
    imp = dispatch[eq]
    print(f"  Equivalent {net.gen.at[eq, 'name']}: import [{imp.min():.1f}, {imp.max():.1f}] MW")
    print("=" * 72)
    print()
    return dispatch, committed


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
