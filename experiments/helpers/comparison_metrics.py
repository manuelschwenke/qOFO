#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/helpers/comparison_metrics.py
=========================================
Aggregation helpers for cross-scenario comparison of
:class:`MultiTSOIterationRecord` logs produced by ``run_multi_tso_dso``.

All functions accept ``records: List[MultiTSOIterationRecord]`` and return
either a dict of NumPy arrays (one entry per simulation step) or a pandas
DataFrame summary.  The downstream plotter
(``visualisation.plot_compare_scenarios.plot_scenario_comparison``) consumes
these directly.

Voltage metrics are split into **transmission system (TS)** and
**distribution system (DS)** variants because the two have very different
operating ranges, control authorities, and physical meaning of bound
violations:

* TS — `zone_v_*` aggregates over EHV buses inside each TSO control zone.
  This is the headline metric for the multi-TSO OFO controller.
* DS — `dso_group_v_*_pu` aggregates over the HV (110 kV) buses in each
  HV sub-network.

The system-wide envelope (TS + DS combined) is still available via
:func:`voltage_envelope_all` for backward compatibility.

In addition to min/mean/max envelopes, we report a **voltage RMSD to
setpoint** per timestep — the spatial root-mean-square deviation of zone
(or DSO-group) mean voltages from the configured setpoint (default 1.03
pu).  Time-averaged RMSDs are reported as scalar columns in the summary
table.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.helpers.records import MultiTSOIterationRecord


# Default voltage setpoint (matches MultiTSOConfig.v_setpoint_pu).
V_SET_DEFAULT: float = 1.03


# ---------------------------------------------------------------------------
#  Time axis
# ---------------------------------------------------------------------------


def _times_min(records: List[MultiTSOIterationRecord]) -> NDArray[np.float64]:
    """Time-axis in minutes, one entry per record."""
    return np.array([r.time_s / 60.0 for r in records], dtype=np.float64)


# ---------------------------------------------------------------------------
#  Voltage envelope (TS / DS / combined)
# ---------------------------------------------------------------------------


def _envelope_from_dicts(records, mins_attr: str, maxs_attr: str, means_attr: str
                         ) -> Dict[str, NDArray[np.float64]]:
    """Generic per-step (min, mean, max) envelope from three record dicts."""
    n = len(records)
    t = _times_min(records)
    v_min = np.full(n, np.nan, dtype=np.float64)
    v_max = np.full(n, np.nan, dtype=np.float64)
    v_mean = np.full(n, np.nan, dtype=np.float64)

    for i, r in enumerate(records):
        mins = list(getattr(r, mins_attr).values())
        maxs = list(getattr(r, maxs_attr).values())
        means = list(getattr(r, means_attr).values())
        if mins:
            v_min[i] = float(min(v for v in mins if np.isfinite(v)) if any(np.isfinite(v) for v in mins) else np.nan)
        if maxs:
            v_max[i] = float(max(v for v in maxs if np.isfinite(v)) if any(np.isfinite(v) for v in maxs) else np.nan)
        if means:
            finite = [v for v in means if np.isfinite(v)]
            if finite:
                v_mean[i] = float(np.mean(finite))

    return {"t_min": t, "v_min": v_min, "v_max": v_max, "v_mean": v_mean}


def voltage_envelope_ts(records: List[MultiTSOIterationRecord]
                        ) -> Dict[str, NDArray[np.float64]]:
    """Per-step TRANSMISSION-SYSTEM voltage envelope (across all TSO zones).

    Uses ``zone_v_min/max/mean`` populated from EHV bus voltages in
    ``zd.v_bus_indices`` for each :class:`ZoneDefinition`.
    """
    return _envelope_from_dicts(records, "zone_v_min", "zone_v_max", "zone_v_mean")


def voltage_envelope_ds(records: List[MultiTSOIterationRecord]
                        ) -> Dict[str, NDArray[np.float64]]:
    """Per-step DISTRIBUTION-SYSTEM voltage envelope (across all HV groups)."""
    return _envelope_from_dicts(records, "dso_group_v_min_pu",
                                "dso_group_v_max_pu", "dso_group_v_mean_pu")


def voltage_envelope_all(records: List[MultiTSOIterationRecord]
                         ) -> Dict[str, NDArray[np.float64]]:
    """Combined TS + DS envelope (legacy; retained for backward compatibility)."""
    n = len(records)
    t = _times_min(records)
    v_min = np.full(n, np.nan, dtype=np.float64)
    v_max = np.full(n, np.nan, dtype=np.float64)
    v_mean = np.full(n, np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        mins = list(r.zone_v_min.values()) + list(r.dso_group_v_min_pu.values())
        maxs = list(r.zone_v_max.values()) + list(r.dso_group_v_max_pu.values())
        means = list(r.zone_v_mean.values()) + list(r.dso_group_v_mean_pu.values())
        finite_mins = [v for v in mins if np.isfinite(v)]
        finite_maxs = [v for v in maxs if np.isfinite(v)]
        finite_means = [v for v in means if np.isfinite(v)]
        if finite_mins:
            v_min[i] = float(min(finite_mins))
        if finite_maxs:
            v_max[i] = float(max(finite_maxs))
        if finite_means:
            v_mean[i] = float(np.mean(finite_means))
    return {"t_min": t, "v_min": v_min, "v_max": v_max, "v_mean": v_mean}


# ---------------------------------------------------------------------------
#  Voltage RMSD to setpoint (TS / DS)
# ---------------------------------------------------------------------------


def _rmsd_from_means(records, means_attr: str, v_set: float
                     ) -> NDArray[np.float64]:
    """Per-step root-mean-square deviation of per-zone (or per-group) mean
    voltages from ``v_set``.  RMSD over the spatial dimension at each step.
    """
    n = len(records)
    out = np.full(n, np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        means = [v for v in getattr(r, means_attr).values() if np.isfinite(v)]
        if means:
            arr = np.asarray(means, dtype=np.float64) - v_set
            out[i] = float(np.sqrt(np.mean(arr ** 2)))
    return out


def voltage_rmsd_ts(records: List[MultiTSOIterationRecord],
                    v_set: float = V_SET_DEFAULT
                    ) -> Dict[str, NDArray[np.float64]]:
    """Per-step TS voltage RMSD: sqrt(mean over zones of (V_mean - v_set)^2)."""
    return {"t_min": _times_min(records),
            "rmsd_pu": _rmsd_from_means(records, "zone_v_mean", v_set)}


def voltage_rmsd_ds(records: List[MultiTSOIterationRecord],
                    v_set: float = V_SET_DEFAULT
                    ) -> Dict[str, NDArray[np.float64]]:
    """Per-step DS voltage RMSD: sqrt(mean over DSO groups of
    (V_mean - v_set)^2)."""
    return {"t_min": _times_min(records),
            "rmsd_pu": _rmsd_from_means(records, "dso_group_v_mean_pu", v_set)}


# ---------------------------------------------------------------------------
#  Voltage violation counts (TS / DS)
# ---------------------------------------------------------------------------


def _violation_counts(records, mins_attr: str, maxs_attr: str,
                      low: float, high: float
                      ) -> Dict[str, NDArray[np.float64]]:
    n = len(records)
    t = _times_min(records)
    n_low = np.zeros(n, dtype=np.float64)
    n_high = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(records):
        mins = getattr(r, mins_attr).values()
        maxs = getattr(r, maxs_attr).values()
        n_low[i] = float(sum(1 for v in mins if np.isfinite(v) and v < low))
        n_high[i] = float(sum(1 for v in maxs if np.isfinite(v) and v > high))
    return {"t_min": t, "n_low": n_low, "n_high": n_high}


def voltage_violation_counts_ts(records: List[MultiTSOIterationRecord],
                                low: float = 0.95, high: float = 1.05
                                ) -> Dict[str, NDArray[np.float64]]:
    """Per-step TS violation count (zones with v_min < low or v_max > high)."""
    return _violation_counts(records, "zone_v_min", "zone_v_max", low, high)


def voltage_violation_counts_ds(records: List[MultiTSOIterationRecord],
                                low: float = 0.95, high: float = 1.05
                                ) -> Dict[str, NDArray[np.float64]]:
    """Per-step DS violation count (HV groups with v_min < low or v_max > high)."""
    return _violation_counts(records, "dso_group_v_min_pu",
                             "dso_group_v_max_pu", low, high)


# Backward-compatibility shims (combined TS + DS).
def voltage_envelope(records):
    return voltage_envelope_all(records)


def voltage_violation_counts(records, low=0.95, high=1.05):
    """Combined TS + DS counts (legacy)."""
    a = voltage_violation_counts_ts(records, low, high)
    b = voltage_violation_counts_ds(records, low, high)
    return {"t_min": a["t_min"],
            "n_low": a["n_low"] + b["n_low"],
            "n_high": a["n_high"] + b["n_high"]}


# ---------------------------------------------------------------------------
#  Loss + gen-Q-headroom (unchanged)
# ---------------------------------------------------------------------------


def loss_series(records: List[MultiTSOIterationRecord]
                ) -> Dict[str, NDArray[np.float64]]:
    """Per-step total active-power network losses (lines + 2W + 3W trafos)."""
    return {
        "t_min": _times_min(records),
        "losses_mw": np.array([r.total_losses_mw for r in records], dtype=np.float64),
    }


def gen_q_headroom_series(records: List[MultiTSOIterationRecord]
                          ) -> Dict[str, NDArray[np.float64]]:
    """Per-step minimum sync-gen Q headroom across ALL zones."""
    n = len(records)
    t = _times_min(records)
    q_min = np.full(n, np.nan, dtype=np.float64)
    q_mean = np.full(n, np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        if not r.gen_q_headroom_mvar:
            continue
        all_h = np.concatenate([np.asarray(arr, dtype=np.float64)
                                for arr in r.gen_q_headroom_mvar.values()
                                if arr is not None and len(arr) > 0])
        if all_h.size:
            q_min[i] = float(all_h.min())
            q_mean[i] = float(all_h.mean())
    return {"t_min": t, "q_headroom_min_mvar": q_min,
            "q_headroom_mean_mvar": q_mean}


def q_tie_deviation_series(records: List[MultiTSOIterationRecord],
                           q_tie_setpoint_mvar: float = 0.0,
                           ) -> Dict[str, NDArray[np.float64]]:
    """Per-step summed absolute deviation of inter-zone tie-line Q flows
    from their setpoint.

    ``records[k].zone_tie_q_mvar`` is a ``Dict[Tuple[int, int], float]``
    keyed by ordered zone pairs ``(zi, zj)`` with ``zi < zj``; each value
    is the aggregate Q [Mvar] flowing from zone ``zi`` into zone ``zj``,
    summed over all physical boundary lines between the two zones (sign
    convention: positive = Q leaves zi).

    The returned ``sum_abs_dev_mvar`` is, at each step,
    ``sum_pair |Q_tie_pair - q_tie_setpoint_mvar|`` — the total absolute
    inter-zone reactive exchange relative to the schedule.

    Parameters
    ----------
    records
        Per-step iteration records from ``run_multi_tso_dso``.
    q_tie_setpoint_mvar
        Common setpoint applied to every zone pair.  Default 0.0 reflects
        the Phase B "no inter-zone Q exchange" target.

    Returns
    -------
    dict with keys
        ``t_min``           -- time axis [min].
        ``sum_abs_dev_mvar`` -- summed absolute deviation across all
                                zone pairs at each step.
        ``max_abs_dev_mvar`` -- worst-pair absolute deviation at each
                                step (which pair carries the most |Q|).
        ``n_pairs_per_step`` -- number of zone pairs reported each step
                                (constant across the run, but useful as
                                a sanity check).
    """
    n = len(records)
    t = _times_min(records)
    sum_abs = np.full(n, np.nan, dtype=np.float64)
    max_abs = np.full(n, np.nan, dtype=np.float64)
    n_pairs = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(records):
        pair_q = r.zone_tie_q_mvar
        if not pair_q:
            continue
        devs = np.array(
            [abs(q - q_tie_setpoint_mvar) for q in pair_q.values()],
            dtype=np.float64,
        )
        sum_abs[i] = float(devs.sum())
        max_abs[i] = float(devs.max())
        n_pairs[i] = len(devs)
    return {
        "t_min": t,
        "sum_abs_dev_mvar": sum_abs,
        "max_abs_dev_mvar": max_abs,
        "n_pairs_per_step": n_pairs,
    }


# ---------------------------------------------------------------------------
#  Scalar summary table — TS / DS split
# ---------------------------------------------------------------------------


def summary_table(logs: Dict[str, List[MultiTSOIterationRecord]],
                  v_set: float = V_SET_DEFAULT,
                  low: float = 0.95, high: float = 1.05,
                  ) -> pd.DataFrame:
    """One row per scenario with headline scalars, TS / DS split.

    Columns
    -------
    n_steps                 -- number of records (0 if diverged)
    converged               -- True iff at least one record
    ts_v_min, ts_v_max      -- TS extremes across all steps
    ts_v_mean               -- mean of per-step TS v_mean
    ts_v_rmsd_pu            -- time-mean of per-step TS RMSD to setpoint
    ts_n_low, ts_n_high     -- TS step-aggregate violation counts
    ds_v_min, ds_v_max      -- DS extremes
    ds_v_mean               -- mean of per-step DS v_mean
    ds_v_rmsd_pu            -- time-mean of per-step DS RMSD to setpoint
    ds_n_low, ds_n_high     -- DS step-aggregate violation counts
    losses_mean_mw, losses_max_mw
    q_headroom_min, q_headroom_mean
    """
    rows = []
    for name, recs in logs.items():
        if not recs:
            rows.append({
                "scenario": name, "n_steps": 0, "converged": False,
                "ts_v_min": np.nan, "ts_v_max": np.nan, "ts_v_mean": np.nan,
                "ts_v_rmsd_pu": np.nan,
                "ts_n_low": 0, "ts_n_high": 0,
                "ds_v_min": np.nan, "ds_v_max": np.nan, "ds_v_mean": np.nan,
                "ds_v_rmsd_pu": np.nan,
                "ds_n_low": 0, "ds_n_high": 0,
                "losses_mean_mw": np.nan, "losses_max_mw": np.nan,
                "q_headroom_min": np.nan, "q_headroom_mean": np.nan,
            })
            continue

        ts = voltage_envelope_ts(recs)
        ds = voltage_envelope_ds(recs)
        ts_rmsd = voltage_rmsd_ts(recs, v_set)["rmsd_pu"]
        ds_rmsd = voltage_rmsd_ds(recs, v_set)["rmsd_pu"]
        ts_vv = voltage_violation_counts_ts(recs, low, high)
        ds_vv = voltage_violation_counts_ds(recs, low, high)
        ls = loss_series(recs)
        qh = gen_q_headroom_series(recs)

        rows.append({
            "scenario":         name,
            "n_steps":          len(recs),
            "converged":        True,
            "ts_v_min":         float(np.nanmin(ts["v_min"])),
            "ts_v_max":         float(np.nanmax(ts["v_max"])),
            "ts_v_mean":        float(np.nanmean(ts["v_mean"])),
            "ts_v_rmsd_pu":     float(np.nanmean(ts_rmsd)),
            "ts_n_low":         int(np.nansum(ts_vv["n_low"])),
            "ts_n_high":        int(np.nansum(ts_vv["n_high"])),
            "ds_v_min":         float(np.nanmin(ds["v_min"])),
            "ds_v_max":         float(np.nanmax(ds["v_max"])),
            "ds_v_mean":        float(np.nanmean(ds["v_mean"])),
            "ds_v_rmsd_pu":     float(np.nanmean(ds_rmsd)),
            "ds_n_low":         int(np.nansum(ds_vv["n_low"])),
            "ds_n_high":        int(np.nansum(ds_vv["n_high"])),
            "losses_mean_mw":   float(np.nanmean(ls["losses_mw"])),
            "losses_max_mw":    float(np.nanmax(ls["losses_mw"])),
            "q_headroom_min":   float(np.nanmin(qh["q_headroom_min_mvar"])),
            "q_headroom_mean":  float(np.nanmean(qh["q_headroom_min_mvar"])),
        })

    return pd.DataFrame(rows).set_index("scenario")
