"""
Diagnostic: compare realised peak load (P and Q) of the wind_replace
network to the IEEE 39 base case, across the 5 h experiment window and
the full year.

Hypothesis (H2) under test
--------------------------
The TN load split in ``network/ieee39/build.py:459-460`` uses

    base_p_profile = 0.5 * p_orig / PROFILE_MEAN[prof_p]

so that the *time mean* of the aggregated bus load equals ``p_orig``.
The peak however is bounded only by ``max(profile)``:

    peak_load = 0.5*p_orig + (0.5*p_orig / PROFILE_MEAN[prof]) * max(profile)

For example, with HS4_pload (PROFILE_MEAN=0.4436) and max(profile)=1.0:
    peak = 0.5 + 0.5/0.4436 ~ 1.63 * p_orig    (63 % overshoot)

The user expects "maximums in the range of original IEEE 39, but
variable values well below the maximum" -- i.e. peak <= p_orig.
This script confirms or refutes the overshoot empirically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Make the repo root importable when the script is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pandapower.networks as pn

from core.profiles import DEFAULT_PROFILES_CSV, load_profiles
from network.ieee39.build import build_ieee39_net
from network.ieee39.constants import PROFILE_MEAN
from network.ieee39.hv_networks import add_hv_networks


WINDOW_START = datetime(2016, 4, 15, 8, 0)
WINDOW_END   = WINDOW_START + timedelta(hours=5)


def _profile_stats(profiles: pd.DataFrame, name: str) -> tuple[float, float, float, float]:
    """Return (mean_year, max_year, mean_window, max_window) for a column."""
    if name not in profiles.columns:
        return (np.nan, np.nan, np.nan, np.nan)
    full = profiles[name].dropna()
    win  = profiles.loc[WINDOW_START:WINDOW_END, name].dropna()
    return (
        float(full.mean()),
        float(full.max()),
        float(win.mean()) if len(win) else np.nan,
        float(win.max()) if len(win) else np.nan,
    )


def main() -> None:
    print("=" * 78)
    print(f"Load-peak diagnostic: window {WINDOW_START} -> {WINDOW_END}")
    print("=" * 78)

    profiles = load_profiles(DEFAULT_PROFILES_CSV, timestep_min=15)
    print(f"Loaded {len(profiles)} samples from {DEFAULT_PROFILES_CSV}")
    print(f"Profile date range: {profiles.index.min()} .. {profiles.index.max()}")

    print()
    print("-" * 78)
    print("Profile statistics (full year vs 5 h window)")
    print(f"{'profile':<18} {'PROFILE_MEAN':>13} {'mean_year':>11} "
          f"{'max_year':>10} {'mean_win':>10} {'max_win':>10}")
    profile_cols = ["HS4_pload", "HS4_qload", "HS5_pload", "HS5_qload",
                    "mv_rural_pload", "mv_rural_qload", "WP10", "PV3"]
    stats: dict[str, tuple[float, float, float, float]] = {}
    for col in profile_cols:
        st = _profile_stats(profiles, col)
        stats[col] = st
        pm = PROFILE_MEAN.get(col, np.nan)
        print(f"{col:<18} {pm:>13.4f} {st[0]:>11.4f} "
              f"{st[1]:>10.4f} {st[2]:>10.4f} {st[3]:>10.4f}")

    print()
    print("-" * 78)
    print("Calibration drift: PROFILE_MEAN constant vs realised mean (full year)")
    for col, st in stats.items():
        pm = PROFILE_MEAN.get(col, np.nan)
        if not np.isnan(pm):
            drift_pct = 100.0 * (st[0] - pm) / pm if pm != 0 else np.nan
            print(f"  {col:<18}: drift = {drift_pct:+.2f} %")
    print("(positive = realised mean above constant; load is over-scaled)")

    # Build the network the same way the experiment does.
    net, meta = build_ieee39_net(
        ext_grid_vm_pu=1.03,
        scenario="wind_replace",
        verbose=False,
    )
    meta = add_hv_networks(
        net, meta,
        install_tso_tertiary_shunts=False,
        verbose=False,
    )

    # Snapshot base values (same convention as core.profiles.snapshot_base_values).
    if "base_p_mw" not in net.load.columns:
        net.load["base_p_mw"] = net.load["p_mw"].copy()
        net.load["base_q_mvar"] = net.load["q_mvar"].copy()

    # Reference: original IEEE 39 base loads (no modifications).
    ieee = pn.case39()
    p_orig_total = float(ieee.load["p_mw"].sum())
    q_orig_total = float(ieee.load["q_mvar"].sum())

    # Constant load total (rows without a profile)
    const_mask = net.load["profile_p"].isna()
    p_const_total = float(net.load.loc[const_mask, "p_mw"].sum())
    q_const_total = float(net.load.loc[const_mask, "q_mvar"].sum())

    # Variable load: total at profile-peak.  Each variable row contributes
    #   p_mw_at_peak = base_p_mw * max(profile_p)
    var_mask = ~const_mask
    p_var_peak_year = 0.0
    q_var_peak_year = 0.0
    p_var_peak_win  = 0.0
    q_var_peak_win  = 0.0
    p_var_mean_year = 0.0
    q_var_mean_year = 0.0
    for li in net.load.index[var_mask]:
        prof_p = net.load.at[li, "profile_p"]
        prof_q = net.load.at[li, "profile_q"]
        bp = float(net.load.at[li, "base_p_mw"])
        bq = float(net.load.at[li, "base_q_mvar"])
        if isinstance(prof_p, str) and prof_p in stats:
            mean_y, max_y, _mean_w, max_w = stats[prof_p]
            p_var_peak_year += bp * max_y
            p_var_peak_win  += bp * (max_w if not np.isnan(max_w) else max_y)
            p_var_mean_year += bp * mean_y
        else:
            p_var_peak_year += bp
            p_var_peak_win  += bp
            p_var_mean_year += bp
        if isinstance(prof_q, str) and prof_q in stats:
            mean_y, max_y, _mean_w, max_w = stats[prof_q]
            q_var_peak_year += bq * max_y
            q_var_peak_win  += bq * (max_w if not np.isnan(max_w) else max_y)
            q_var_mean_year += bq * mean_y
        else:
            q_var_peak_year += bq
            q_var_peak_win  += bq
            q_var_mean_year += bq

    # Aggregate totals
    p_total_peak_year = p_const_total + p_var_peak_year
    q_total_peak_year = q_const_total + q_var_peak_year
    p_total_peak_win  = p_const_total + p_var_peak_win
    q_total_peak_win  = q_const_total + q_var_peak_win
    p_total_mean_year = p_const_total + p_var_mean_year
    q_total_mean_year = q_const_total + q_var_mean_year

    print()
    print("-" * 78)
    print("System aggregate load (MW / Mvar)")
    print(f"{'metric':<32} {'P [MW]':>14} {'Q [Mvar]':>14}")
    print(f"{'IEEE 39 base (orig)':<32} {p_orig_total:>14.1f} {q_orig_total:>14.1f}")
    print(f"{'Constant-half (built)':<32} {p_const_total:>14.1f} {q_const_total:>14.1f}")
    print(f"{'Mean (full year)':<32} {p_total_mean_year:>14.1f} {q_total_mean_year:>14.1f}")
    print(f"{'Peak (full year)':<32} {p_total_peak_year:>14.1f} {q_total_peak_year:>14.1f}")
    print(f"{'Peak (5 h window)':<32} {p_total_peak_win:>14.1f} {q_total_peak_win:>14.1f}")

    print()
    print("Overshoot vs IEEE 39 base (peak / orig)")
    print(f"  full year: P = {100*(p_total_peak_year/p_orig_total - 1):+6.1f} %, "
          f"Q = {100*(q_total_peak_year/q_orig_total - 1):+6.1f} %")
    print(f"  5 h window: P = {100*(p_total_peak_win /p_orig_total - 1):+6.1f} %, "
          f"Q = {100*(q_total_peak_win /q_orig_total - 1):+6.1f} %")

    print()
    print("-" * 78)
    print("Per-bus breakdown for variable TN loads (top 10 by P-overshoot)")
    print(f"{'bus':>4} {'profile':<12} {'p_orig':>9} "
          f"{'p_const':>9} {'p_var_pk':>9} {'p_total_pk':>11} {'overshoot':>10}")
    rows = []
    for bus in sorted(set(net.load["bus"].unique()) & set(ieee.load["bus"].unique())):
        # skip non-TN buses: the IEEE-39 case has 0..38; HV DSO buses are above.
        if bus > 38:
            continue
        ieee_at_bus = ieee.load[ieee.load["bus"] == bus]
        if ieee_at_bus.empty:
            continue
        p_orig = float(ieee_at_bus["p_mw"].sum())
        if p_orig <= 0:
            continue

        rows_at_bus = net.load[net.load["bus"] == bus]
        p_const_b = float(rows_at_bus.loc[rows_at_bus["profile_p"].isna(), "p_mw"].sum())
        p_var_peak_b = 0.0
        prof_p_used = ""
        for li in rows_at_bus.index[rows_at_bus["profile_p"].notna()]:
            prof_p = net.load.at[li, "profile_p"]
            bp = float(net.load.at[li, "base_p_mw"])
            if prof_p in stats:
                p_var_peak_b += bp * stats[prof_p][1]   # max_year
                prof_p_used = prof_p
        p_total_b = p_const_b + p_var_peak_b
        overshoot_pct = 100.0 * (p_total_b / p_orig - 1.0)
        rows.append((bus, prof_p_used, p_orig, p_const_b, p_var_peak_b,
                     p_total_b, overshoot_pct))

    rows.sort(key=lambda r: -r[6])
    for r in rows[:10]:
        print(f"{r[0]:>4} {r[1]:<12} {r[2]:>9.1f} "
              f"{r[3]:>9.1f} {r[4]:>9.1f} {r[5]:>11.1f} {r[6]:>+9.1f} %")


if __name__ == "__main__":
    main()
