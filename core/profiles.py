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


def load_profiles(csv_path: str, timestep_min: int = 15) -> pd.DataFrame:
    """Load profiles CSV and interpolate to *timestep_min* resolution.

    Returns a DataFrame indexed by ``DatetimeIndex`` with one column per
    profile (e.g. ``PV3``, ``WP7``, ``mv_rural_pload``, …).
    """
    df = pd.read_csv(csv_path, delimiter=";")
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M", dayfirst=True)
    for col in df.columns.drop("time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if timestep_min < 15:
        df = df.resample(f"{timestep_min}min").interpolate(method="linear")
    return df


def snapshot_base_values(net: pp.pandapowerNet) -> None:
    """Store original P/Q so profiles always scale from base values."""
    if "base_p_mw" not in net.load.columns:
        net.load["base_p_mw"] = net.load["p_mw"].copy()
        net.load["base_q_mvar"] = net.load["q_mvar"].copy()
    if "base_p_mw" not in net.sgen.columns:
        net.sgen["base_p_mw"] = net.sgen["p_mw"].copy()


def apply_profiles(net: pp.pandapowerNet, profiles: pd.DataFrame, t: datetime) -> None:
    """Apply profile scaling factors for time *t* to loads and sgens."""
    idx = profiles.index.get_indexer([t], method="nearest")[0]
    row = profiles.iloc[idx]

    # Loads: P *= mv_rural_pload, Q *= mv_rural_qload
    if "mv_rural_pload" in row.index and "mv_rural_qload" in row.index:
        fp, fq = float(row["mv_rural_pload"]), float(row["mv_rural_qload"])
        net.load["p_mw"] = net.load["base_p_mw"] * fp
        net.load["q_mvar"] = net.load["base_q_mvar"] * fq

    # Sgen P: scale by individual profile column (PV3 / WP7 / WP10 / …)
    if "profile" in net.sgen.columns:
        for prof_name in net.sgen["profile"].dropna().unique():
            if prof_name in row.index:
                mask = (net.sgen["profile"] == prof_name) & (
                    ~net.sgen["name"].astype(str).str.startswith("BOUND_")
                )
                net.sgen.loc[mask, "p_mw"] = (
                    net.sgen.loc[mask, "base_p_mw"] * float(row[prof_name])
                )

