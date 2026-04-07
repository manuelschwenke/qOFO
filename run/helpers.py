#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/helpers.py
==============
Miscellaneous helper functions used by the cascade simulation loop.

Moved here from the root ``run_S_TSO_M_DSO.py`` as part of the ``run/`` package
refactor.  The root ``run_S_TSO_M_DSO.py`` is now a backward-compatibility shim.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List

import numpy as np
import pandapower as pp

from core.network_state import NetworkState

from .records import IterationRecord


def _network_state(net: pp.pandapowerNet, source: str = "COMBINED") -> NetworkState:
    """Snapshot NetworkState from a converged combined network."""
    buses = np.array(net.bus.index, dtype=np.int64)
    vm = net.res_bus.loc[buses, "vm_pu"].values.astype(np.float64)
    va = np.deg2rad(net.res_bus.loc[buses, "va_degree"].values.astype(np.float64))
    slack = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
    pv = np.array(
        [
            int(net.gen.at[g, "bus"])
            for g in net.gen.index
            if net.gen.at[g, "in_service"]
        ],
        dtype=np.int64,
    )
    pq = np.array(
        [int(b) for b in buses if int(b) != slack and int(b) not in pv], dtype=np.int64
    )
    return NetworkState(
        bus_indices=buses,
        voltage_magnitudes_pu=vm,
        voltage_angles_rad=va,
        slack_bus_index=slack,
        pv_bus_indices=pv,
        pq_bus_indices=pq,
        transformer_indices=np.array([], dtype=np.int64),
        tap_positions=np.array([], dtype=np.float64),
        source_case=source,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cached_at_iteration=0,
    )


def _sgen_at_bus(net, bus: int, *, exclude_bound: bool = True):
    """Return first non-boundary sgen index at a bus."""
    for s in net.sgen.index[net.sgen["bus"] == bus]:
        if exclude_bound and str(net.sgen.at[s, "name"]).startswith("BOUND_"):
            continue
        return s
    raise ValueError(f"No sgen at bus {bus}")


def _build_Gw(diag_vec, n_pre_oltc, n_oltc, cross_weight):
    """Return 1-D diag vector or full 2-D G_w matrix."""
    if cross_weight == 0.0 or n_oltc <= 1:
        return diag_vec  # no coupling needed
    n = len(diag_vec)
    G = np.diag(diag_vec.copy())
    i0 = n_pre_oltc
    i1 = n_pre_oltc + n_oltc
    # Add g_cross · 𝟏𝟏ᵀ to the OLTC sub-block (all elements)
    G[i0:i1, i0:i1] += cross_weight
    return G


def print_summary(v_set: float, log: List[IterationRecord]):
    final = log[-1]
    print()
    print("=" * 72)
    print(f"  SUMMARY  --  V_set = {v_set:.3f} p.u.")
    print("=" * 72)

    if final.plant_tn_voltages_pu is not None:
        v = final.plant_tn_voltages_pu
        print(
            f"  Final TN V: min={np.min(v):.4f}  mean={np.mean(v):.4f}  max={np.max(v):.4f}"
        )
        print(f"  Max |V - V_set| = {np.max(np.abs(v - v_set)):.4f} p.u.")

    n_tso = sum(1 for r in log if r.tso_active)
    n_dso = sum(1 for r in log if r.dso_active)
    print(f"  TSO steps: {n_tso},  DSO steps: {n_dso}")

    tso_recs = [r for r in log if r.tso_active and r.plant_tn_voltages_pu is not None]
    if tso_recs:
        print()
        print(
            f"  {'min':>5s}  {'V_min':>8s}  {'V_mean':>8s}  {'V_max':>8s}  "
            f"{'|err|_max':>10s}  {'obj':>12s}  {'status':>10s}"
        )
        print(
            f"  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 12}  {'-' * 10}"
        )
        for r in tso_recs:
            v = r.plant_tn_voltages_pu
            err = np.max(np.abs(v - v_set))
            print(
                f"  {r.minute:5d}  {np.min(v):8.4f}  {np.mean(v):8.4f}  "
                f"{np.max(v):8.4f}  {err:10.4f}  "
                f"{r.tso_objective or 0:12.4e}  {r.tso_solver_status or '':>10s}"
            )

    print("=" * 72)
