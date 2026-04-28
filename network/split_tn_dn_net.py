#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Splitting — Separate TN and DN Models
==============================================

Splits the combined 380/110/20 kV benchmark network produced by
:func:`build_tuda_net` into two independent pandapower networks:

* **TN network** — Transmission grid (380 kV) with boundary static
  generators that represent the aggregate effect of each DSO area.
  The 3-winding coupler transformers are removed; the power they
  exchanged is injected directly at the coupler HV (380 kV) buses.

* **DN network** — Distribution grid (110 kV) with the 3-winding
  coupler transformers retained.  One coupler HV bus becomes the
  slack bus; the remaining coupler HV buses receive boundary static
  generators representing the upstream transmission feed.

Public API
----------
``split_network(combined_net, meta, dn_slack_coupler_index)``
    → ``SplitResult``

``validate_split(combined_net, meta, result)``
    → ``bool``

``CouplerPowerFlow``
    Converged power flow at one coupler transformer (dataclass).

``SplitResult``
    Container for the two split networks and associated metadata.

Author: Manuel Schwenke
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandapower as pp
import pandapower.toolbox as tb

from network.build_tuda_net import NetworkMetadata


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CouplerPowerFlow:
    """Converged power-flow quantities at one 3-winding coupler transformer.

    All powers follow the pandapower sign convention (positive = into the
    transformer winding from the bus side).

    Attributes
    ----------
    trafo3w_idx : int
        Index of the 3-winding transformer in the combined network.
    hv_bus : int
        380 kV bus index.
    mv_bus : int
        110 kV bus index.
    lv_bus : int
        20 kV tertiary bus index.
    p_hv_mw, q_hv_mvar : float
        Active and reactive power at the HV winding [MW / Mvar].
    p_mv_mw, q_mv_mvar : float
        Active and reactive power at the MV winding [MW / Mvar].
    p_lv_mw, q_lv_mvar : float
        Active and reactive power at the LV winding [MW / Mvar].
    vm_hv_pu : float
        Voltage magnitude at the HV bus [p.u.].
    va_hv_deg : float
        Voltage angle at the HV bus [degrees].
    """

    trafo3w_idx: int
    hv_bus: int
    mv_bus: int
    lv_bus: int
    p_hv_mw: float
    q_hv_mvar: float
    p_mv_mw: float
    q_mv_mvar: float
    p_lv_mw: float
    q_lv_mvar: float
    vm_hv_pu: float
    va_hv_deg: float


@dataclass(frozen=True)
class SplitResult:
    """Container for the outcome of a network split.

    Attributes
    ----------
    tn_net : pp.pandapowerNet
        Transmission-only network (converged).
    dn_net : pp.pandapowerNet
        Distribution-only network (converged).
    coupler_flows : list[CouplerPowerFlow]
        Coupler power-flow data extracted from the *combined* network,
        in the same order as ``meta.coupler_trafo3w_indices``.
    tn_boundary_sgen_indices : list[int]
        Indices of boundary sgens in ``tn_net.sgen``.
    dn_boundary_sgen_indices : list[int]
        Indices of boundary sgens in ``dn_net.sgen``.
    dn_slack_ext_grid_index : int
        Index of the slack ext_grid in ``dn_net.ext_grid``.
    dn_slack_coupler_index : int
        Which coupler was chosen as the DN slack (0-based).
    """

    tn_net: pp.pandapowerNet
    dn_net: pp.pandapowerNet
    coupler_flows: List[CouplerPowerFlow]
    tn_boundary_sgen_indices: List[int]
    dn_boundary_sgen_indices: List[int]
    dn_slack_ext_grid_index: int
    dn_slack_coupler_index: int


# ═══════════════════════════════════════════════════════════════════════════════
#  COUPLER POWER-FLOW EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_coupler_flows(
    net: pp.pandapowerNet,
    meta: NetworkMetadata,
) -> List[CouplerPowerFlow]:
    """Read converged coupler quantities from the combined network.

    Raises
    ------
    RuntimeError
        If the combined network has not converged (``res_trafo3w`` missing).
    """
    if not hasattr(net, "res_trafo3w") or net.res_trafo3w.empty:
        raise RuntimeError(
            "Combined network has no converged res_trafo3w.  "
            "Run pp.runpp() before splitting."
        )

    flows: List[CouplerPowerFlow] = []
    for i, tidx in enumerate(meta.coupler_trafo3w_indices):
        hv = meta.coupler_hv_buses[i]
        mv = meta.coupler_mv_buses[i]
        lv = meta.coupler_lv_buses[i]
        flows.append(CouplerPowerFlow(
            trafo3w_idx=tidx,
            hv_bus=hv, mv_bus=mv, lv_bus=lv,
            p_hv_mw=float(net.res_trafo3w.at[tidx, "p_hv_mw"]),
            q_hv_mvar=float(net.res_trafo3w.at[tidx, "q_hv_mvar"]),
            p_mv_mw=float(net.res_trafo3w.at[tidx, "p_mv_mw"]),
            q_mv_mvar=float(net.res_trafo3w.at[tidx, "q_mv_mvar"]),
            p_lv_mw=float(net.res_trafo3w.at[tidx, "p_lv_mw"]),
            q_lv_mvar=float(net.res_trafo3w.at[tidx, "q_lv_mvar"]),
            vm_hv_pu=float(net.res_bus.at[hv, "vm_pu"]),
            va_hv_deg=float(net.res_bus.at[hv, "va_degree"]),
        ))
    return flows


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: BUS SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _buses_by_subnet(net: pp.pandapowerNet, subnet: str) -> List[int]:
    """Return bus indices whose ``subnet`` column equals *subnet*."""
    if "subnet" not in net.bus.columns:
        raise RuntimeError("Network buses have no 'subnet' column.")
    mask = net.bus["subnet"].astype(str) == subnet
    return [int(b) for b in net.bus.index[mask]]


# ═══════════════════════════════════════════════════════════════════════════════
#  NETWORK SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

def split_network(
    combined_net: pp.pandapowerNet,
    meta: NetworkMetadata,
    *,
    dn_slack_coupler_index: int = 0,
) -> SplitResult:
    """Split the combined network into separate TN and DN models.

    Both output networks are returned in a *converged* state.

    Parameters
    ----------
    combined_net : pp.pandapowerNet
        Converged combined TSO-DSO network from :func:`build_tuda_net`.
    meta : NetworkMetadata
        Metadata produced by :func:`build_tuda_net`.
    dn_slack_coupler_index : int
        Which coupler (0-based) shall serve as the DN slack bus.

    Returns
    -------
    SplitResult
        Container with both networks, coupler flows, and boundary-element
        indices.

    Raises
    ------
    ValueError
        If *dn_slack_coupler_index* is out of range.
    RuntimeError
        If the combined network is not converged.
    """
    n_couplers = len(meta.coupler_trafo3w_indices)
    if not (0 <= dn_slack_coupler_index < n_couplers):
        raise ValueError(
            f"dn_slack_coupler_index={dn_slack_coupler_index} out of range "
            f"[0, {n_couplers})."
        )

    coupler_flows = _extract_coupler_flows(combined_net, meta)

    # -----------------------------------------------------------------
    #  TN network
    # -----------------------------------------------------------------
    tn_net = copy.deepcopy(combined_net)

    # Boundary sgens: inject the power the coupler drew at the HV side
    # (negated, because removing the coupler removes that load/source).
    tn_boundary: List[int] = []
    for i, cf in enumerate(coupler_flows):
        sidx = pp.create_sgen(
            tn_net, bus=cf.hv_bus,
            p_mw=-cf.p_hv_mw, q_mvar=-cf.q_hv_mvar,
            name=f"BOUND_DN|Coupler{i}", in_service=True,
        )
        tn_boundary.append(int(sidx))

    # Remove 3W transformers
    tn_net.trafo3w.drop(
        index=[i for i in meta.coupler_trafo3w_indices
               if i in tn_net.trafo3w.index],
        inplace=True,
    )

    # Keep only TN buses, generator-terminal buses, and coupler HV buses
    tn_keep = (
        set(_buses_by_subnet(tn_net, "TN"))
        | set(_buses_by_subnet(tn_net, "GEN_TERM"))
        | set(meta.coupler_hv_buses)
    )
    tn_drop = [int(b) for b in tn_net.bus.index if int(b) not in tn_keep]
    if tn_drop:
        tb.drop_buses(tn_net, tn_drop, drop_elements=True)

    # Run power flow on TN
    pp.runpp(tn_net, run_control=True, calculate_voltage_angles=True)

    # -----------------------------------------------------------------
    #  DN network
    # -----------------------------------------------------------------
    dn_net = copy.deepcopy(combined_net)

    # Keep DN buses + coupler HV/MV/LV buses
    dn_keep = (
        set(_buses_by_subnet(dn_net, "DN"))
        | set(meta.coupler_hv_buses)
        | set(meta.coupler_mv_buses)
        | set(meta.coupler_lv_buses)
    )
    dn_drop = [int(b) for b in dn_net.bus.index if int(b) not in dn_keep]
    if dn_drop:
        tb.drop_buses(dn_net, dn_drop, drop_elements=True)

    # Remove any surviving ext_grid elements (they belonged to TN)
    if not dn_net.ext_grid.empty:
        dn_net.ext_grid.drop(index=dn_net.ext_grid.index.tolist(), inplace=True)

    # Boundary sgens at each coupler HV bus (TN feed)
    dn_boundary: List[int] = []
    for i, cf in enumerate(coupler_flows):
        sidx = pp.create_sgen(
            dn_net, bus=cf.hv_bus,
            p_mw=cf.p_hv_mw, q_mvar=cf.q_hv_mvar,
            name=f"BOUND_TN|Coupler{i}", in_service=True,
        )
        dn_boundary.append(int(sidx))

    # DN slack bus at the selected coupler HV bus
    slack_cf = coupler_flows[dn_slack_coupler_index]
    dn_slack_idx = int(pp.create_ext_grid(
        dn_net,
        bus=slack_cf.hv_bus,
        vm_pu=slack_cf.vm_hv_pu,
        va_degree=slack_cf.va_hv_deg,
        name=f"DN_SLACK|Coupler{dn_slack_coupler_index}",
    ))

    # Run power flow on DN
    pp.runpp(dn_net, run_control=True, calculate_voltage_angles=True)

    return SplitResult(
        tn_net=tn_net,
        dn_net=dn_net,
        coupler_flows=coupler_flows,
        tn_boundary_sgen_indices=tn_boundary,
        dn_boundary_sgen_indices=dn_boundary,
        dn_slack_ext_grid_index=dn_slack_idx,
        dn_slack_coupler_index=dn_slack_coupler_index,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compare(
    label: str,
    a: np.ndarray,
    b: np.ndarray,
    atol: float,
) -> bool:
    """Element-wise comparison of two arrays (NaN-safe).  Prints a summary."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        print(f"  {label}: SKIP (no finite entries)")
        return True  # nothing to compare — not a failure

    diff = np.abs(a[finite] - b[finite])
    max_err = float(np.max(diff))
    ok = bool(np.allclose(a[finite], b[finite], atol=atol, rtol=0.0))
    status = "OK" if ok else "FAIL"
    print(f"  {label}: {status}  (max |Δ| = {max_err:.3e}, atol = {atol:.0e})")
    return ok


def _compare_scalar(label: str, a: float, b: float, atol: float) -> bool:
    """Compare two scalars.  Prints a summary line."""
    ok = abs(a - b) <= atol
    status = "OK" if ok else "FAIL"
    print(f"  {label}: {status}  (a={a:.6f}, b={b:.6f}, |Δ|={abs(a-b):.3e})")
    return ok


def validate_split(
    combined_net: pp.pandapowerNet,
    meta: NetworkMetadata,
    result: SplitResult,
    *,
    v_atol: float = 5e-4,
    va_atol_deg: float = 5e-3,
    loading_atol_pct: float = 1e-2,
    pq_atol: float = 1e-6,
) -> bool:
    """Verify that the split networks reproduce the combined operating point.

    The following checks are performed:

    1. **TN bus voltages** match the combined network within *v_atol*.
    2. **TN bus angles** match within *va_atol_deg*.
    3. **TN line loadings** match within *loading_atol_pct*.
    4. **TN boundary sgen setpoints** are consistent with coupler flows.
    5. **DN boundary sgen setpoints** are consistent with coupler flows.
    6. **DN slack bus voltage** equals the combined operating point.
    7. **DN voltage range** is within [0.85, 1.15] p.u.

    Parameters
    ----------
    combined_net : pp.pandapowerNet
        The original converged combined network.
    meta : NetworkMetadata
        Build metadata.
    result : SplitResult
        Output of :func:`split_network`.
    v_atol, va_atol_deg, loading_atol_pct, pq_atol : float
        Tolerances for the respective comparison.

    Returns
    -------
    bool
        ``True`` if **all** checks pass.
    """
    ok = True
    tn = result.tn_net
    dn = result.dn_net
    cfs = result.coupler_flows

    print("=" * 60)
    print("SPLIT VALIDATION")
    print("=" * 60)

    # --- 1 & 2: TN bus voltages and angles ---
    tn_buses = sorted(set(tn.bus.index) & set(combined_net.res_bus.index))
    if tn_buses:
        ok &= _compare(
            "TN vm_pu",
            combined_net.res_bus.loc[tn_buses, "vm_pu"].values,
            tn.res_bus.loc[tn_buses, "vm_pu"].values,
            atol=v_atol,
        )
        ok &= _compare(
            "TN va_degree",
            combined_net.res_bus.loc[tn_buses, "va_degree"].values,
            tn.res_bus.loc[tn_buses, "va_degree"].values,
            atol=va_atol_deg,
        )
    else:
        print("  TN buses: FAIL (no overlapping buses)")
        ok = False

    # --- 3: TN line loadings ---
    tn_lines = sorted(
        set(tn.res_line.index) & set(combined_net.res_line.index)
    )
    if tn_lines:
        ok &= _compare(
            "TN line loading_%",
            combined_net.res_line.loc[tn_lines, "loading_percent"].values,
            tn.res_line.loc[tn_lines, "loading_percent"].values,
            atol=loading_atol_pct,
        )

    # --- 4: TN boundary sgen setpoints ---
    print("  ── TN boundary sgens ──")
    for i, cf in enumerate(cfs):
        sidx = result.tn_boundary_sgen_indices[i]
        ok &= _compare_scalar(
            f"  TN sgen[{sidx}] P_mw",
            float(tn.sgen.at[sidx, "p_mw"]), -cf.p_hv_mw, atol=pq_atol,
        )
        ok &= _compare_scalar(
            f"  TN sgen[{sidx}] Q_mvar",
            float(tn.sgen.at[sidx, "q_mvar"]), -cf.q_hv_mvar, atol=pq_atol,
        )

    # --- 5: DN boundary sgen setpoints ---
    print("  ── DN boundary sgens ──")
    for i, cf in enumerate(cfs):
        sidx = result.dn_boundary_sgen_indices[i]
        ok &= _compare_scalar(
            f"  DN sgen[{sidx}] P_mw",
            float(dn.sgen.at[sidx, "p_mw"]), cf.p_hv_mw, atol=pq_atol,
        )
        ok &= _compare_scalar(
            f"  DN sgen[{sidx}] Q_mvar",
            float(dn.sgen.at[sidx, "q_mvar"]), cf.q_hv_mvar, atol=pq_atol,
        )

    # --- 6: DN slack bus voltage ---
    slack_cf = cfs[result.dn_slack_coupler_index]
    ok &= _compare_scalar(
        "DN slack vm_pu",
        float(dn.res_bus.at[slack_cf.hv_bus, "vm_pu"]),
        slack_cf.vm_hv_pu, atol=1e-6,
    )
    ok &= _compare_scalar(
        "DN slack va_deg",
        float(dn.res_bus.at[slack_cf.hv_bus, "va_degree"]),
        slack_cf.va_hv_deg, atol=1e-6,
    )

    # --- 7: DN voltage range ---
    dn_vm = dn.res_bus["vm_pu"].values
    vm_min, vm_max = float(np.nanmin(dn_vm)), float(np.nanmax(dn_vm))
    range_ok = 0.85 <= vm_min and vm_max <= 1.15
    status = "OK" if range_ok else "FAIL"
    print(f"  DN vm range: {status}  ({vm_min:.4f} … {vm_max:.4f} p.u.)")
    ok &= range_ok

    # --- Summary ---
    print("─" * 60)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return ok
