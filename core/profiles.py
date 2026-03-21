"""
Time-series profile loading, interpolation, and application to pandapower networks.

Profiles CSV format: semicolon-delimited, columns ``time;PV3;WP7;WP10;mv_rural_qload;mv_rural_pload``
with 15-minute resolution and German date format ``dd.mm.yyyy HH:MM``.
"""

from __future__ import annotations
import os
from datetime import datetime
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

