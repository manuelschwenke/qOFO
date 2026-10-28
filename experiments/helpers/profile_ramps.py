"""
experiments/helpers/profile_ramps.py
====================================

Held profiles and load ramps on the interpolated profile frame, for the step
tests of Case 2 (thesis ``ch:case3:step``, plan
``docs/architecture/2026-10-28_case2_step_mc_plan.md``).

Both act on the frame the runner has already interpolated to ``dt_s`` and
clipped to the horizon, so they are exact at the step resolution, reach the
plant through the ordinary ``apply_profiles`` path, and are inert unless the
config asks for them.

* :func:`hold_profile_frame` -- every row equals the row at ``start_time``:
  loads, infeed and (through the zonal dispatch computed from the frame) the
  generator dispatch are frozen at the operating point, so the test
  disturbance is the only exogenous change.
* :func:`apply_profile_ramps` -- ramp specifications (dicts, from
  ``MultiTSOConfig.profile_ramps``):

  ``{"kind": "bus", "bus": b, "q_mvar": Q, "p_mw": P, "t0_s": t0,
  "ramp_s": T, "hold_s": H, "down": True}``
      an additional load at bus ``b`` (pandapower index) whose P and Q follow
      ``shape(t) * (P, Q)``;
  ``{"kind": "ts_loads", "rel": r, "exclude_buses": [...], "t0_s": t0,
  "ramp_s": T, "hold_s": H, "down": False}``
      every load outside ``exclude_buses`` (the sub-transmission buses) is
      scaled by ``1 + r * shape(t)`` in P and Q, through its own copy of its
      profile columns so loads that share a column with an STS are untouched.

  ``shape`` rises linearly from 0 at ``t0_s`` to 1 at ``t0_s + ramp_s``,
  holds for ``hold_s`` and, if ``down``, returns to 0 at the same rate.

Author: Manuel Schwenke / Claude Code (2026-10-28)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pandapower as pp


def hold_profile_frame(profiles: pd.DataFrame, start_time: datetime) -> pd.DataFrame:
    """The frame with every row replaced by the row nearest ``start_time``."""
    i0 = int(profiles.index.get_indexer([start_time], method="nearest")[0])
    row = profiles.iloc[i0].to_numpy()
    return pd.DataFrame(np.tile(row, (len(profiles), 1)), index=profiles.index,
                        columns=profiles.columns)


def ramp_shape(index: pd.DatetimeIndex, start_time: datetime,
               spec: Dict[str, Any]) -> np.ndarray:
    """0 -> 1 linear from ``t0_s`` over ``ramp_s``, hold ``hold_s``, and back
    to 0 over ``ramp_s`` if ``down``."""
    t = np.asarray((index - pd.Timestamp(start_time)).total_seconds(), dtype=float)
    t0, T = float(spec["t0_s"]), float(spec["ramp_s"])
    if T <= 0.0:
        raise ValueError("ramp_s must be positive")
    up = np.clip((t - t0) / T, 0.0, 1.0)
    if not spec.get("down", False):
        return up
    t1 = t0 + T + float(spec.get("hold_s", 0.0))
    return up - np.clip((t - t1) / T, 0.0, 1.0)


def apply_profile_ramps(net: pp.pandapowerNet, profiles: pd.DataFrame,
                        start_time: datetime, specs: Iterable[Dict[str, Any]],
                        verbose: int = 0) -> pd.DataFrame:
    """Add the ramps to ``profiles`` (returns the new frame) and wire the
    loads of ``net`` to the new columns.  Requires the base columns of
    :func:`core.profiles.snapshot_base_values`."""
    specs = list(specs or ())
    if not specs:
        return profiles
    profiles = profiles.copy()
    for k, spec in enumerate(specs):
        shape = ramp_shape(profiles.index, start_time, spec)
        kind = spec.get("kind")
        if kind == "bus":
            col = f"RAMP{k}_BUS_{int(spec['bus'])}"
            profiles[col] = shape
            q, p = float(spec.get("q_mvar", 0.0)), float(spec.get("p_mw", 0.0))
            row = pp.create_load(net, bus=int(spec["bus"]), p_mw=0.0, q_mvar=0.0,
                                 name=f"STEP_RAMP_{k}")
            net.load.at[row, "base_p_mw"] = p
            net.load.at[row, "base_q_mvar"] = q
            net.load.at[row, "profile_p"] = col if p != 0.0 else np.nan
            net.load.at[row, "profile_q"] = col
            if verbose:
                print(f"  [ramp {k}] bus {int(spec['bus'])}: P {p:+g} MW, Q {q:+g} Mvar,"
                      f" t0 {spec['t0_s']:g} s, ramp {spec['ramp_s']:g} s,"
                      f" hold {spec.get('hold_s', 0):g} s, down={bool(spec.get('down'))}")
        elif kind == "ts_loads":
            rel = float(spec["rel"])
            excl = {int(b) for b in spec.get("exclude_buses", ())}
            rows = [int(i) for i in net.load.index
                    if int(net.load.at[i, "bus"]) not in excl
                    and not str(net.load.at[i, "name"]).startswith("STEP_RAMP_")]
            factor = 1.0 + rel * shape
            made: Dict[str, str] = {}
            for i in rows:
                for pcol in ("profile_p", "profile_q"):
                    if pcol not in net.load.columns:
                        continue
                    src = net.load.at[i, pcol]
                    if not isinstance(src, str) or src not in profiles.columns:
                        continue
                    if src not in made:
                        made[src] = f"RAMP{k}_TS_{src}"
                        profiles[made[src]] = profiles[src].to_numpy() * factor
                    net.load.at[i, pcol] = made[src]
            if verbose:
                print(f"  [ramp {k}] {len(rows)} TS loads x (1 + {rel:g} shape), "
                      f"{len(made)} profile columns, t0 {spec['t0_s']:g} s, "
                      f"ramp {spec['ramp_s']:g} s, down={bool(spec.get('down'))}")
        else:
            raise ValueError(f"unknown ramp kind {kind!r} (bus | ts_loads)")
    return profiles


def sts_bus_set(hv_networks: Sequence[Any]) -> List[int]:
    """Every bus of every sub-transmission system (to exclude from TS ramps)."""
    return sorted({int(b) for hv in hv_networks for b in hv.bus_indices})
