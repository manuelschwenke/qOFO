"""
Time-series profile loading, interpolation, and application to pandapower networks.

Profiles CSV format: semicolon-delimited, columns ``time;PV3;WP7;WP10;mv_rural_qload;mv_rural_pload``
with 15-minute resolution and German date format ``dd.mm.yyyy HH:MM``.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

DEFAULT_PROFILES_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "profiles.csv"
)


def load_profiles(
    csv_path: str,
    timestep_min: int = 15,
    timestep_s: float | None = None,
) -> pd.DataFrame:
    """Load profiles CSV and interpolate to the requested resolution.

    Parameters
    ----------
    csv_path : str
        Path to the semicolon-delimited profiles CSV.
    timestep_min : int
        Target resolution in minutes (default 15).
        Ignored when *timestep_s* is provided.
    timestep_s : float, optional
        Target resolution in seconds.  Overrides *timestep_min* when set.
        Allows sub-minute interpolation (e.g. ``timestep_s=10``).

    Returns a DataFrame indexed by ``DatetimeIndex`` with one column per
    profile (e.g. ``PV3``, ``WP7``, ``mv_rural_pload``, …).
    """
    df = pd.read_csv(csv_path, delimiter=";")
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M", dayfirst=True)
    for col in df.columns.drop("time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Determine effective resolution in seconds
    eff_s = timestep_s if timestep_s is not None else timestep_min * 60.0
    native_s = 15 * 60  # CSV is 15-minute resolution

    if eff_s < native_s:
        if eff_s >= 60.0 and eff_s % 60 == 0:
            df = df.resample(f"{int(eff_s // 60)}min").interpolate(method="linear")
        else:
            df = df.resample(f"{int(eff_s)}s").interpolate(method="linear")
    return df


def snapshot_base_values(net: pp.pandapowerNet) -> None:
    """Store original P/Q so profiles always scale from base values."""
    if "base_p_mw" not in net.load.columns:
        net.load["base_p_mw"] = net.load["p_mw"].copy()
        net.load["base_q_mvar"] = net.load["q_mvar"].copy()
    if "base_p_mw" not in net.sgen.columns:
        net.sgen["base_p_mw"] = net.sgen["p_mw"].copy()


def apply_profiles(net: pp.pandapowerNet, profiles: pd.DataFrame, t: datetime) -> None:
    """Apply profile scaling factors for time *t* to loads and sgens.

    Each load uses the profile columns specified in its ``profile_p`` and
    ``profile_q`` columns (e.g. ``mv_rural_pload``, ``HS4_pload``).
    Each sgen uses the profile column specified in its ``profile`` column.
    """
    idx = profiles.index.get_indexer([t], method="nearest")[0]
    row = profiles.iloc[idx]

    # Loads: scale P and Q by per-load profile columns
    if "profile_p" in net.load.columns:
        for prof_name in net.load["profile_p"].dropna().unique():
            if prof_name in row.index and not np.isnan(row[prof_name]):
                mask = net.load["profile_p"] == prof_name
                net.load.loc[mask, "p_mw"] = (
                    net.load.loc[mask, "base_p_mw"] * float(row[prof_name])
                )
    if "profile_q" in net.load.columns:
        for prof_name in net.load["profile_q"].dropna().unique():
            if prof_name in row.index and not np.isnan(row[prof_name]):
                mask = net.load["profile_q"] == prof_name
                net.load.loc[mask, "q_mvar"] = (
                    net.load.loc[mask, "base_q_mvar"] * float(row[prof_name])
                )

    # Sgen P: scale by individual profile column (PV3 / WP7 / WP10 / …)
    if "profile" in net.sgen.columns:
        for prof_name in net.sgen["profile"].dropna().unique():
            if prof_name in row.index and not np.isnan(row[prof_name]):
                mask = (net.sgen["profile"] == prof_name) & (
                    ~net.sgen["name"].astype(str).str.startswith("BOUND_")
                )
                net.sgen.loc[mask, "p_mw"] = (
                    net.sgen.loc[mask, "base_p_mw"] * float(row[prof_name])
                )


# ---------------------------------------------------------------------------
#  Zonal residual-load balancing (generator dispatch)
# ---------------------------------------------------------------------------

def compute_zonal_gen_dispatch(
    net: pp.pandapowerNet,
    profiles: pd.DataFrame,
    zone_map: Dict[int, List[int]],
    *,
    gen_p_min_mw: float = 0.0,
) -> pd.DataFrame:
    """Pre-compute generator active-power dispatch from zonal residual load.

    For every timestep in *profiles* and every zone the function computes:

        residual_load = sum(load_P) - sum(sgen_P)

    and distributes the residual equally among all ``net.gen`` units in that
    zone.  The slack generator (``net.ext_grid``) is excluded — it will absorb
    any remaining system-wide mismatch during the power flow.

    The result is a DataFrame indexed by the profile timestamps with one column
    per generator index, containing the dispatched P [MW] at each timestep.
    Use :func:`apply_gen_dispatch` to write a single row into ``net.gen.p_mw``
    before each power-flow solve.

    Parameters
    ----------
    net : pp.pandapowerNet
        Network after ``build_ieee39_net()`` and ``add_hv_networks()``.
        Must already have ``base_p_mw`` columns (call ``snapshot_base_values``
        first).
    profiles : pd.DataFrame
        Output of :func:`load_profiles`.
    zone_map : dict[int, list[int]]
        Zone-id → list of bus indices (from ``zone_partition``).
    gen_p_min_mw : float
        Minimum generator output [MW].  Generators are clipped to
        ``[gen_p_min_mw, p_max]`` after equal distribution.

    Returns
    -------
    pd.DataFrame
        Index = profile timestamps, columns = generator indices (int),
        values = dispatched P_mw.
    """
    if "base_p_mw" not in net.load.columns:
        raise RuntimeError(
            "Call snapshot_base_values(net) before compute_zonal_gen_dispatch."
        )

    # ── Map each bus to its zone ────────────────────────────────────────────
    bus_zone: Dict[int, int] = {}
    for z, buses in zone_map.items():
        for b in buses:
            bus_zone[b] = z

    # ── Classify loads by zone and profile ──────────────────────────────────
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

    # ── Classify sgens by zone and profile ──────────────────────────────────
    ZoneSgenInfo = List[Tuple[float, Optional[str]]]
    zone_sgens: Dict[int, ZoneSgenInfo] = {z: [] for z in zone_map}

    for si in net.sgen.index:
        name = str(net.sgen.at[si, "name"])
        if name.startswith("BOUND_") or name.startswith("TN_DER"):
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

    # ── Classify generators by zone ─────────────────────────────────────────
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

    # ── Pre-compute dispatch for every timestep ─────────────────────────────
    all_gen_indices = sorted(
        gi for gens in zone_gens.values() for gi in gens
    )
    dispatch = pd.DataFrame(
        index=profiles.index,
        columns=all_gen_indices,
        dtype=np.float64,
    )

    for idx in profiles.index:
        row = profiles.loc[idx]

        for z in zone_map:
            # Total zone load at this timestep
            total_load_p = 0.0
            for base_p, prof in zone_loads[z]:
                scale = float(row[prof]) if (prof and prof in row.index and not np.isnan(row[prof])) else 1.0
                total_load_p += base_p * scale

            # Total zone sgen at this timestep
            total_sgen_p = 0.0
            for base_p, prof in zone_sgens[z]:
                scale = float(row[prof]) if (prof and prof in row.index and not np.isnan(row[prof])) else 1.0
                total_sgen_p += base_p * scale

            residual = total_load_p - total_sgen_p

            # Distribute equally among generators in this zone
            gens_in_zone = zone_gens[z]
            if not gens_in_zone:
                continue

            per_gen = residual / len(gens_in_zone)
            for gi in gens_in_zone:
                p_clipped = np.clip(per_gen, gen_p_min_mw, gen_p_max.get(gi, 9999.0))
                dispatch.at[idx, gi] = p_clipped

    # Print summary
    print()
    print("=" * 72)
    print("  Zonal Generator Dispatch (residual load balancing)")
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
            print(f"  Zone {z}: {n_loads} loads, {n_sgens} sgens, {n_gens} gens (no dispatch)")
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

