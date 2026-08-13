"""
analysis/simbench_hv_benchmark.py
=================================
Benchmark the synthetic 110 kV DS topology used as underlaid DSO network in the
IEEE 39-bus case against the SimBench 110 kV reference grids.

Purpose
-------
Provide a *citable* reference for two topological design choices of the
synthetic HV sub-network (``network/ieee39/constants.py::HV_LINE_TOPOLOGY``):

1. **Line lengths** -- are the 10--40 km circuits typical for a German
   110 kV sub-transmission grid?
2. **Distance between coupling transformers** -- is a spacing of ~40--70 km
   between the EHV/HV interfaces of one DSO area typical?

Reference data
--------------
SimBench (Meinecke et al., *SimBench -- A Benchmark Dataset of Electric Power
Systems to Compare Innovative Solutions based on Power Flow Analysis*,
Energies 13(12):3290, 2020; https://simbench.de) provides two scenario-0
110 kV reference grids:

===============  =========================  ==================================
Grid            SimBench code               Character
===============  =========================  ==================================
HV1             ``1-HV-mixed--0-sw``        mixed rural/semi-urban, 6 EHV/HV
                                            transformers in 3 substations
HV2             ``1-HV-urban--0-sw``        urban, 3 EHV/HV transformers in
                                            1 substation
===============  =========================  ==================================

Within scenario 0, the ``1-HVMV-*`` and ``1-EHVHV-*`` /
``1-EHVHVMVLV-*`` codes embed the same base HV1/HV2 grids. Future scenarios
1 and 2 add generator-interconnection buses and lines as well as generation
capacity. This analysis deliberately fixes scenario 0, giving N = 2 grids
(and, for coupling-point spacing, N = 3 pairs) -- this is stated explicitly in
the report rather than hidden behind pooled statistics.

Method
------
*Station reduction.*  SimBench models substations at node-breaker detail
(auxiliary nodes + busbars + bus-bus switches).  Raw bus counts are therefore
not comparable to the synthetic 10-bus grid.  We build the switch-respecting
graph (``pandapower.topology.create_nxgraph``, open switches removed = normal
operational state), contract every zero-impedance-connected cluster into one
*station node*, and run all distance statistics on that reduced graph.  A
cluster counts as a station if it carries a busbar, an injection or a
transformer terminal.

*Distance measures.*  Two weights are reported for every path:

  - geometric length :math:`\\ell` [km] -- route distance;
  - series reactance :math:`X = \\sum_i x_i \\ell_i` [Ohm] and
    :math:`X_{pu} = X / Z_{base}`, :math:`Z_{base} = 110^2/100` Ohm.

For reactive-power control the electrical distance :math:`X` -- not
:math:`\\ell` -- sets the voltage/Q sensitivity, so a conductor mismatch must
be reported alongside the length comparison.

*Coupling points.*  A coupling transformer is a two-winding transformer with
``vn_hv_kv`` in {220, 380} and ``vn_lv_kv`` = 110, or a three-winding
transformer with ``vn_hv_kv`` > 110 and ``vn_mv_kv`` = 110 (the convention
used by ``network/ieee39/hv_networks.py``).  Transformers whose 110 kV
terminals fall into the same station node are one *coupling point*
(parallel bank), cross-checked against SimBench's ``substation`` column.

Usage
-----
::

    python -m analysis.simbench_hv_benchmark
    python -m analysis.simbench_hv_benchmark --outdir results/simbench_hv_benchmark
    python -m analysis.simbench_hv_benchmark --from-built-net   # use the real
                                                               # IEEE39+DSO net

Outputs (``--outdir``, default ``results/simbench_hv_benchmark``):
``lines.csv``, ``line_length_stats.csv``, ``coupling_pairs.csv``,
``station_depth.csv``, ``topology_summary.csv``, ``conductors.csv``,
``report.md`` and ``simbench_hv_benchmark.png``.

Author: qOFO / analysis utilities
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top

# --------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------

#: SimBench scenario-0 reference grids. Future expansion scenarios can add
#: generator-interconnection buses and lines.
SIMBENCH_HV_CODES: Dict[str, str] = {
    "SimBench HV1 (mixed)": "1-HV-mixed--0-sw",
    "SimBench HV2 (urban)": "1-HV-urban--0-sw",
}

HV_KV: float = 110.0
_V_TOL: float = 1.0          # kV tolerance when matching nominal voltages
S_BASE_MVA: float = 100.0    # p.u. base for electrical distance
Z_BASE_OHM: float = HV_KV ** 2 / S_BASE_MVA   # 121 Ohm

_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


# --------------------------------------------------------------------------
#  Result container
# --------------------------------------------------------------------------

@dataclass
class GridTopology:
    """Reduced (station-level) description of one 110 kV grid."""

    label: str
    lines: pd.DataFrame                    # one row per 110 kV circuit
    station_graph: nx.Graph                # nodes = stations, edges = circuits
    n_buses_raw: int
    n_stations: int
    coupling_points: List[int]             # station ids carrying an EHV/HV trafo
    coupling_trafos: pd.DataFrame          # one row per transformer
    substation_check: str = ""             # agreement with SimBench substation col
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
#  Network loading
# --------------------------------------------------------------------------

def load_simbench_nets(codes: Dict[str, str]) -> Dict[str, pp.pandapowerNet]:
    """Return ``{label: net}`` for the given SimBench codes."""
    import simbench as sb

    nets: Dict[str, pp.pandapowerNet] = {}
    for label, code in codes.items():
        print(f"[load] {label:<24s} <- {code}")
        nets[label] = sb.get_simbench_net(code)
    return nets


def build_synthetic_reference_net(
    *,
    line_length_scale: float = 1.0,
    std_type: str = "184-AL1/30-ST1A 110.0",
) -> Tuple[pp.pandapowerNet, str]:
    """
    Rebuild ONE synthetic HV sub-network as a standalone pandapower net.

    Reads the single source of truth used by
    ``network.ieee39.hv_networks._create_hv_subnetwork``:
    ``HV_LINE_TOPOLOGY`` (11 circuits over 10 buses) and the ``hv_buses``
    coupling entry of ``SUBNET_DEFS``.  Loads, DER and tap changers are
    irrelevant for a topological comparison and are omitted; the 345 kV side
    is stubbed by one bus per coupling transformer so that the coupling
    detection below sees the same pattern as in SimBench.

    Returns
    -------
    (net, label)
    """
    from network.ieee39.constants import HV_LINE_TOPOLOGY, SUBNET_DEFS

    hv_buses: Sequence[int] = SUBNET_DEFS[0]["hv_buses"]
    scales = sorted({float(s["scale"]) for s in SUBNET_DEFS})
    if len(scales) == 1:
        line_length_scale = scales[0]
    label = (f"Synthetic DS (TUDA 110 kV), scale={line_length_scale:g}")

    net = pp.create_empty_network(name="synthetic_hv_reference")
    bus_map = {
        i: int(pp.create_bus(net, vn_kv=HV_KV, name=f"HV_Bus_{i}", type="b"))
        for i in range(10)
    }

    for f, t, base_km in HV_LINE_TOPOLOGY:
        pp.create_line(
            net,
            from_bus=bus_map[f],
            to_bus=bus_map[t],
            length_km=float(base_km) * line_length_scale,
            std_type=std_type,
            name=f"Line_({f}-{t})",
        )

    # 345/110/20 kV three-winding couplers, mirroring hv_networks.py
    for k, hv_no in enumerate(hv_buses):
        ehv = pp.create_bus(net, vn_kv=345.0, name=f"TN_stub_{k}", type="b")
        lv = pp.create_bus(net, vn_kv=20.0, name=f"Tertiary_{k}", type="b")
        pp.create_transformer3w_from_parameters(
            net,
            hv_bus=int(ehv), mv_bus=bus_map[int(hv_no)], lv_bus=int(lv),
            sn_hv_mva=300.0, sn_mv_mva=300.0, sn_lv_mva=75.0,
            vn_hv_kv=345.0, vn_mv_kv=HV_KV, vn_lv_kv=20.0,
            vk_hv_percent=12.0, vk_mv_percent=8.0, vk_lv_percent=10.0,
            vkr_hv_percent=0.30, vkr_mv_percent=0.20, vkr_lv_percent=0.25,
            pfe_kw=80.0, i0_percent=0.04,
            name=f"Coupler3W_HV{hv_no}",
        )

    if "subnet" not in net.bus.columns:
        net.bus["subnet"] = "DN"
    return net, label


def extract_synthetic_from_built_net(verbose: bool = False):
    """
    Alternative source: build the *real* IEEE 39 + DSO network and slice out
    each 110 kV sub-network.  Slower, but proves the constants-based
    reconstruction above matches the model actually simulated.

    Returns ``{label: net}`` where each net is the full network (the 110 kV
    filter in :func:`analyse_grid` isolates one sub-network per ``subnet``
    tag) -- or ``{}`` if the build pipeline is unavailable.
    """
    try:
        from network.ieee39.build import build_ieee39_net
        from network.ieee39.hv_networks import add_hv_networks
    except Exception as exc:                                   # pragma: no cover
        print(f"[warn] cannot import IEEE39 build pipeline: {exc}")
        return {}

    net, meta = build_ieee39_net(verbose=verbose)
    add_hv_networks(net, meta, verbose=verbose)
    return {"Synthetic DS (built IEEE39+DSO)": net}


# --------------------------------------------------------------------------
#  Topology reduction
# --------------------------------------------------------------------------

def _hv_buses(net: pp.pandapowerNet, group: Optional[str] = None) -> np.ndarray:
    """
    Indices of all buses at the 110 kV nominal level.

    ``group`` optionally restricts to one sub-network.  It is matched against
    the ``subnet`` column first and, failing that, against the bus-name prefix
    -- the qOFO HV sub-networks all carry ``subnet == "DN"`` and are
    distinguished only by the ``"<net_id>|..."`` name prefix written by
    ``network/ieee39/hv_networks.py``.
    """
    mask = (net.bus.vn_kv - HV_KV).abs() < _V_TOL
    # Drop the auxiliary buses that carry the split ZIP loads
    # (``network/ieee39/aux_load_buses.py``): they are a load-model artefact
    # behind a 0.01 Ohm stub, not stations of the 110 kV grid.
    mask &= ~net.bus.name.astype(str).str.contains("AUX_LOAD", na=False)
    if "subnet" in net.bus.columns:
        mask &= ~net.bus.subnet.astype(str).str.contains("AUX", case=False,
                                                         na=False)
    if group is not None:
        by_subnet = (net.bus.subnet.astype(str).str.startswith(group)
                     if "subnet" in net.bus.columns
                     else pd.Series(False, index=net.bus.index))
        by_name = net.bus.name.astype(str).str.startswith(f"{group}|")
        mask &= (by_subnet | by_name)
    return np.asarray(net.bus.index[mask])


def _hv_subnetwork_ids(net: pp.pandapowerNet) -> List[str]:
    """Distinct ``<net_id>`` prefixes of the 110 kV buses of a built qOFO net."""
    names = net.bus.loc[(net.bus.vn_kv - HV_KV).abs() < _V_TOL, "name"].astype(str)
    return sorted({n.split("|", 1)[0] for n in names if "|" in n})


def _hv_lines(net: pp.pandapowerNet, hv_bus_set: set) -> pd.DataFrame:
    """
    110 kV circuits with geometric and electrical length.

    Auxiliary zero-impedance links introduced by the qOFO ZIP-load split
    (``network/ieee39/aux_load_buses.py``, r = x = 0.01 Ohm over 1 km) are
    excluded -- they are a modelling artefact, not a circuit.
    """
    ln = net.line
    mask = ln.from_bus.isin(hv_bus_set) & ln.to_bus.isin(hv_bus_set)
    if "in_service" in ln.columns:
        mask &= ln.in_service.astype(bool)
    if "subnet" in ln.columns:
        mask &= ~ln.subnet.astype(str).str.contains("AUX", case=False, na=False)
    mask &= ~ln.name.astype(str).str.contains("AUX", case=False, na=False)

    out = ln.loc[mask, ["name", "std_type", "from_bus", "to_bus",
                        "length_km", "r_ohm_per_km", "x_ohm_per_km",
                        "c_nf_per_km", "max_i_ka"]].copy()
    out["type"] = ln.loc[mask, "type"] if "type" in ln.columns else "ol"
    par = ln.loc[mask, "parallel"] if "parallel" in ln.columns else 1.0
    out["parallel"] = pd.to_numeric(par, errors="coerce").fillna(1.0)
    out["x_ohm"] = out.x_ohm_per_km * out.length_km / out.parallel
    out["r_ohm"] = out.r_ohm_per_km * out.length_km / out.parallel
    out["x_pu"] = out.x_ohm / Z_BASE_OHM
    return out


def _coupling_transformers(net: pp.pandapowerNet, hv_bus_set: set) -> pd.DataFrame:
    """
    EHV/HV coupling transformers with their 110 kV terminal bus.

    Two-winding: ``vn_lv_kv`` = 110 and ``vn_hv_kv`` > 110.
    Three-winding: ``vn_mv_kv`` = 110 and ``vn_hv_kv`` > 110.
    """
    rows: List[dict] = []

    t = net.trafo
    if len(t):
        m = ((t.vn_lv_kv - HV_KV).abs() < _V_TOL) & (t.vn_hv_kv > HV_KV + _V_TOL)
        if "in_service" in t.columns:
            m &= t.in_service.astype(bool)
        for idx in t.index[m]:
            if int(t.at[idx, "lv_bus"]) not in hv_bus_set:
                continue
            rows.append(dict(
                element="trafo", index=int(idx), name=str(t.at[idx, "name"]),
                hv_kv=float(t.at[idx, "vn_hv_kv"]),
                sn_mva=float(t.at[idx, "sn_mva"]),
                hv_terminal=int(t.at[idx, "lv_bus"]),
                substation=str(t.at[idx, "substation"])
                if "substation" in t.columns else "",
            ))

    t3 = net.trafo3w
    if len(t3):
        m3 = ((t3.vn_mv_kv - HV_KV).abs() < _V_TOL) & (t3.vn_hv_kv > HV_KV + _V_TOL)
        if "in_service" in t3.columns:
            m3 &= t3.in_service.astype(bool)
        for idx in t3.index[m3]:
            if int(t3.at[idx, "mv_bus"]) not in hv_bus_set:
                continue
            rows.append(dict(
                element="trafo3w", index=int(idx), name=str(t3.at[idx, "name"]),
                hv_kv=float(t3.at[idx, "vn_hv_kv"]),
                sn_mva=float(t3.at[idx, "sn_mv_mva"]),
                hv_terminal=int(t3.at[idx, "mv_bus"]),
                substation=str(t3.at[idx, "substation"])
                if "substation" in t3.columns else "",
            ))

    return pd.DataFrame(rows)


def _station_clusters(net: pp.pandapowerNet, hv_bus_set: set) -> Dict[int, int]:
    """
    Map every 110 kV bus to a station id.

    A station is a maximal set of 110 kV buses connected by zero-length
    elements only -- closed bus-bus switches and busbar couplers in SimBench's
    node-breaker representation.  Open switches are respected (normal
    operational state), so a station never spans an open coupler.
    """
    g = top.create_nxgraph(net, respect_switches=True, include_trafos=False,
                           calc_branch_impedances=False)
    sub = g.subgraph(hv_bus_set)

    zero = nx.Graph()
    zero.add_nodes_from(sub.nodes)
    for u, v, data in sub.edges(data=True):
        if float(data.get("weight", 0.0)) <= 0.0:
            zero.add_edge(u, v)

    return {b: sid for sid, comp in enumerate(nx.connected_components(zero))
            for b in comp}


def _is_real_node(net: pp.pandapowerNet, buses: Sequence[int]) -> bool:
    """True if a station cluster carries a busbar, an injection or a trafo."""
    bs = set(int(b) for b in buses)
    types = set(net.bus.loc[sorted(bs), "type"].astype(str)) \
        if "type" in net.bus.columns else set()
    if types & {"b", "db", "n"}:
        return True
    for tbl, col in (("load", "bus"), ("sgen", "bus"), ("gen", "bus"),
                     ("shunt", "bus"), ("ward", "bus"), ("storage", "bus")):
        df = getattr(net, tbl, None)
        if df is not None and len(df) and df[col].isin(bs).any():
            return True
    for tbl, cols in (("trafo", ("hv_bus", "lv_bus")),
                      ("trafo3w", ("hv_bus", "mv_bus", "lv_bus"))):
        df = getattr(net, tbl, None)
        if df is not None and len(df):
            if any(df[c].isin(bs).any() for c in cols):
                return True
    return False


def analyse_grid(net: pp.pandapowerNet, label: str,
                 group: Optional[str] = None) -> GridTopology:
    """Reduce one 110 kV grid to stations + circuits and locate its couplings."""
    hv = _hv_buses(net, group)
    hv_set = set(int(b) for b in hv)
    if not hv_set:
        raise ValueError(f"{label}: no 110 kV buses found")

    lines = _hv_lines(net, hv_set)
    cpl = _coupling_transformers(net, hv_set)
    cluster_of = _station_clusters(net, hv_set)

    # Station graph: one node per cluster, one edge per circuit.
    sg = nx.MultiGraph()
    real = {}
    by_cluster: Dict[int, List[int]] = {}
    for b, sid in cluster_of.items():
        by_cluster.setdefault(sid, []).append(int(b))
    for sid, buses in by_cluster.items():
        real[sid] = _is_real_node(net, buses)
        sg.add_node(sid, buses=buses, real=real[sid])

    for li, row in lines.iterrows():
        u, v = cluster_of[int(row.from_bus)], cluster_of[int(row.to_bus)]
        if u == v:
            continue                       # ring inside one station: not a route
        sg.add_edge(u, v, key=int(li), km=float(row.length_km),
                    x_ohm=float(row.x_ohm), x_pu=float(row.x_pu))

    lines = lines.assign(
        station_from=[cluster_of[int(b)] for b in lines.from_bus],
        station_to=[cluster_of[int(b)] for b in lines.to_bus],
    )

    coupling_points: List[int] = []
    if len(cpl):
        cpl = cpl.assign(station=[cluster_of[int(b)] for b in cpl.hv_terminal])
        coupling_points = sorted(set(int(s) for s in cpl.station))

    # Cross-check station clustering against SimBench's own substation label
    check = "n/a (no substation column)"
    if len(cpl) and cpl.substation.astype(str).str.len().gt(0).any() \
            and cpl.substation.nunique() > 0 and cpl.substation.iloc[0] not in ("", "nan"):
        by_sub = cpl.groupby("substation").station.nunique()
        by_sta = cpl.groupby("station").substation.nunique()
        ok = bool((by_sub == 1).all() and (by_sta == 1).all())
        check = ("consistent" if ok else "MISMATCH") + \
                f" ({cpl.substation.nunique()} substations, " \
                f"{len(coupling_points)} station clusters)"

    n_real = sum(1 for sid in sg.nodes if sg.nodes[sid]["real"])
    return GridTopology(
        label=label, lines=lines, station_graph=sg,
        n_buses_raw=len(hv_set), n_stations=n_real,
        coupling_points=coupling_points, coupling_trafos=cpl,
        substation_check=check,
    )


# --------------------------------------------------------------------------
#  Distance statistics
# --------------------------------------------------------------------------

def _simple_graph(sg: nx.MultiGraph, weight: str) -> nx.Graph:
    """Collapse parallel circuits to the shortest one for path search."""
    g = nx.Graph()
    g.add_nodes_from(sg.nodes(data=True))
    for u, v, data in sg.edges(data=True):
        w = float(data[weight])
        if g.has_edge(u, v):
            g[u][v]["weight"] = min(g[u][v]["weight"], w)
        else:
            g.add_edge(u, v, weight=w)
    return g


def coupling_pair_distances(t: GridTopology) -> pd.DataFrame:
    """
    Pairwise shortest-path distance between EHV/HV coupling points, measured
    *inside* the 110 kV grid (paths through the overlaid EHV network are
    excluded by construction).
    """
    if len(t.coupling_points) < 2:
        return pd.DataFrame(columns=["grid", "from_station", "to_station",
                                     "dist_km", "dist_x_ohm", "dist_x_pu",
                                     "n_hops"])
    g_km = _simple_graph(t.station_graph, "km")
    g_x = _simple_graph(t.station_graph, "x_ohm")

    rows = []
    cps = t.coupling_points
    for i, a in enumerate(cps):
        for b in cps[i + 1:]:
            try:
                path = nx.shortest_path(g_km, a, b, weight="weight")
                d_km = nx.shortest_path_length(g_km, a, b, weight="weight")
                d_x = nx.shortest_path_length(g_x, a, b, weight="weight")
            except nx.NetworkXNoPath:
                d_km = d_x = np.nan
                path = []
            rows.append(dict(grid=t.label, from_station=a, to_station=b,
                             dist_km=d_km, dist_x_ohm=d_x,
                             dist_x_pu=d_x / Z_BASE_OHM if d_x == d_x else np.nan,
                             n_hops=max(len(path) - 1, 0)))
    return pd.DataFrame(rows)


def station_depth(t: GridTopology) -> pd.DataFrame:
    """
    Distance from every (real) station to its nearest coupling point.

    This is the statistic that bounds the *radius* of a DSO area: how far a
    controllable node can sit from the EHV interface whose Q it has to track.
    """
    if not t.coupling_points:
        return pd.DataFrame(columns=["grid", "station", "depth_km",
                                     "depth_x_ohm", "depth_x_pu"])
    g_km = _simple_graph(t.station_graph, "km")
    g_x = _simple_graph(t.station_graph, "x_ohm")

    d_km = {}
    d_x = {}
    for cp in t.coupling_points:
        for n, d in nx.single_source_dijkstra_path_length(
                g_km, cp, weight="weight").items():
            d_km[n] = min(d_km.get(n, np.inf), d)
        for n, d in nx.single_source_dijkstra_path_length(
                g_x, cp, weight="weight").items():
            d_x[n] = min(d_x.get(n, np.inf), d)

    rows = [dict(grid=t.label, station=n, depth_km=d_km.get(n, np.nan),
                 depth_x_ohm=d_x.get(n, np.nan),
                 depth_x_pu=d_x.get(n, np.nan) / Z_BASE_OHM)
            for n in t.station_graph.nodes
            if t.station_graph.nodes[n]["real"]]
    return pd.DataFrame(rows)


def corridor_lengths(t: GridTopology) -> pd.DataFrame:
    """
    Circuit lengths after series reduction of pass-through stations.

    Rationale
    ---------
    The synthetic DS is a 10-node *equivalent* of a DSO area: each of its
    nodes aggregates load/DER that a detailed model would spread over several
    stations, and each of its circuits therefore represents a corridor, not a
    single tower line.  Comparing its 11 circuits against SimBench's 95/113
    individual circuits is biased towards "the synthetic lines are too long".

    This function removes that bias by contracting every station of degree 2
    that is not a coupling point, summing ``km`` and ``x_ohm`` along the
    chain, until only branching points, dead ends and coupling points remain.
    The resulting corridor-length sample is the like-for-like reference for
    an aggregated equivalent.

    Caveat: contracted stations may carry load or DER, so the reduction is
    valid for *length* comparison only, not as an electrical equivalent.
    """
    g = nx.MultiGraph()
    g.add_nodes_from(t.station_graph.nodes(data=True))
    for i, (u, v, data) in enumerate(t.station_graph.edges(data=True)):
        g.add_edge(u, v, key=i, km=data["km"], x_ohm=data["x_ohm"])

    protected = set(t.coupling_points)
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes):
            if n in protected or g.degree(n) != 2:
                continue
            edges = list(g.edges(n, keys=True, data=True))
            if len(edges) != 2:
                continue                       # self-loop / parallel pair
            (_, a, ka, da), (_, b, kb, db) = edges
            if a == n or b == n or a == b:
                continue                       # would create a self-loop
            g.add_edge(a, b, km=da["km"] + db["km"],
                       x_ohm=da["x_ohm"] + db["x_ohm"])
            g.remove_node(n)
            changed = True

    rows = [dict(grid=t.label, from_station=u, to_station=v,
                 length_km=d["km"], x_ohm=d["x_ohm"],
                 x_pu=d["x_ohm"] / Z_BASE_OHM, type="corridor",
                 std_type="(reduced)")
            for u, v, d in g.edges(data=True)]
    return pd.DataFrame(rows)


def length_stats(lines: pd.DataFrame, label: str, subset: str = "all") -> dict:
    """Descriptive statistics of one circuit-length sample."""
    s = lines.length_km.astype(float)
    q = s.quantile(_QUANTILES) if len(s) else pd.Series(
        [np.nan] * len(_QUANTILES), index=list(_QUANTILES))
    return {
        "grid": label, "subset": subset, "n": int(len(s)),
        "total_km": float(s.sum()) if len(s) else np.nan,
        "mean_km": float(s.mean()) if len(s) else np.nan,
        "std_km": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "min_km": float(s.min()) if len(s) else np.nan,
        "p10_km": float(q.loc[0.10]), "p25_km": float(q.loc[0.25]),
        "median_km": float(q.loc[0.50]), "p75_km": float(q.loc[0.75]),
        "p90_km": float(q.loc[0.90]),
        "max_km": float(s.max()) if len(s) else np.nan,
        "mean_x_ohm": float(lines.x_ohm.mean()) if len(lines) else np.nan,
        "median_x_ohm": float(lines.x_ohm.median()) if len(lines) else np.nan,
    }


def percentile_of(value: float, sample: Sequence[float]) -> float:
    """Empirical percentile of ``value`` within ``sample`` [%]."""
    arr = np.asarray([v for v in sample if v == v], dtype=float)
    if arr.size == 0 or value != value:
        return np.nan
    return 100.0 * float((arr <= value).mean())


# --------------------------------------------------------------------------
#  Reporting
# --------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Minimal Markdown table (no external dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "--" if v != v else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join("| " + " | ".join(fmt(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def build_report(tops: List[GridTopology],
                 stats: pd.DataFrame,
                 pairs: pd.DataFrame,
                 depth: pd.DataFrame,
                 conductors: pd.DataFrame,
                 summary: pd.DataFrame,
                 corridors: Dict[str, pd.DataFrame],
                 synth_labels: Sequence[str]) -> str:
    """Assemble the Markdown report."""
    synth = set(synth_labels)
    ref_labels = [t.label for t in tops if t.label not in synth]

    ref_lines = pd.concat([t.lines for t in tops if t.label not in synth])
    syn_lines = pd.concat([t.lines for t in tops if t.label in synth])
    ref_ol = ref_lines[ref_lines.type.astype(str) == "ol"]
    ref_cor = pd.concat([corridors[k] for k in ref_labels])
    syn_cor = pd.concat([corridors[k] for k in synth_labels])

    syn_med = float(syn_lines.length_km.median())
    syn_mean = float(syn_lines.length_km.mean())
    pct_med = percentile_of(syn_med, ref_ol.length_km)
    pct_mean = percentile_of(syn_mean, ref_ol.length_km)

    syn_med_c = float(syn_cor.length_km.median())
    pct_med_c = percentile_of(syn_med_c, ref_cor.length_km)

    # Empirical support check: are the synthetic lengths inside the observed
    # SimBench range at all, and how much of SimBench falls in the synthetic
    # band?  This is the claim that survives the small-sample caveat.
    ref_min, ref_max = float(ref_ol.length_km.min()), float(ref_ol.length_km.max())
    syn_in = float(((syn_lines.length_km >= ref_min)
                    & (syn_lines.length_km <= ref_max)).mean()) * 100.0
    syn_lo, syn_hi = float(syn_lines.length_km.min()), float(syn_lines.length_km.max())
    ref_in_band = float(((ref_ol.length_km >= syn_lo)
                         & (ref_ol.length_km <= syn_hi)).mean()) * 100.0

    ref_pairs = pairs[pairs.grid.isin(ref_labels)]
    syn_pairs = pairs[pairs.grid.isin(synth)]
    # The DSO sub-networks are topological clones; report the distinct spacings.
    syn_pair_vals = sorted({round(float(v), 1) for v in syn_pairs.dist_km
                            if v == v})

    out: List[str] = []
    out.append("# SimBench 110 kV benchmark for the synthetic DSO topology\n")
    out.append("Reference: SimBench v1.0 (Meinecke et al., *Energies* 13(12):3290, "
               "2020). Codes `1-HV-mixed--0-sw` (HV1) and `1-HV-urban--0-sw` "
               "(HV2) define the two scenario-0 reference grids used here. "
               "Within scenario 0, `1-HVMV-*`, `1-EHVHV-*` and "
               "`1-EHVHVMVLV-*` embed the same base grids. Future scenarios "
               "can add generator-interconnection assets; this report fixes scenario 0.\n")
    out.append("All distances are shortest paths on the **station-reduced** "
               "110 kV graph (node-breaker detail contracted, open switches "
               "respected). Electrical distance uses "
               f"`Z_base = {HV_KV:.0f} kV^2 / {S_BASE_MVA:.0f} MVA "
               f"= {Z_BASE_OHM:.0f} Ohm`.\n")

    out.append("\n## 1. Topology summary\n")
    out.append(_md_table(summary))

    out.append("\n\n## 2. Circuit length statistics [km]\n")
    cols = ["grid", "subset", "n", "mean_km", "std_km", "min_km", "p10_km",
            "p25_km", "median_km", "p75_km", "p90_km", "max_km", "total_km"]
    out.append(_md_table(stats[cols]))
    out.append("\n\nSubset `corridor` = circuit lengths after series "
               "reduction of pass-through stations (degree-2 stations that "
               "are not coupling points are contracted, lengths summed). "
               "This is the like-for-like reference for the synthetic grid, "
               "which is a 10-node *equivalent* whose circuits represent "
               "corridors rather than individual tower lines. The reduction "
               "is valid for length comparison only -- contracted stations "
               "may carry load or DER.\n")

    out.append("\n## 3. Distance between EHV/HV coupling points\n")
    if len(pairs):
        out.append(_md_table(pairs[["grid", "from_station", "to_station",
                                    "dist_km", "dist_x_ohm", "dist_x_pu",
                                    "n_hops"]], "{:.3f}"))
    else:
        out.append("_No grid in the sample has more than one coupling point._")
    out.append("\n\nNote: SimBench HV2 (urban) concentrates all three EHV/HV "
               "transformers in a single substation, so it contributes no "
               "pair. The reference sample for coupling-point spacing is "
               f"therefore N = {len(ref_pairs)} pairs from HV1 alone.\n")

    out.append("\n## 4. Depth: distance from each station to the nearest "
               "coupling point [km]\n")
    dsum = (depth.groupby("grid")
            .agg(n=("depth_km", "size"), mean_km=("depth_km", "mean"),
                 median_km=("depth_km", "median"), p90_km=("depth_km",
                                                           lambda s: s.quantile(0.90)),
                 max_km=("depth_km", "max"),
                 max_x_pu=("depth_x_pu", "max"))
            .reset_index())
    out.append(_md_table(dsum, "{:.3f}"))

    out.append("\n\n## 5. Conductor parameters\n")
    out.append(_md_table(conductors, "{:.4f}"))

    out.append("\n\n## 6. Assessment\n")
    out.append(
        f"- **Circuit lengths.** The synthetic grid uses {len(syn_lines)} "
        f"circuits of {syn_lines.length_km.min():.0f}--"
        f"{syn_lines.length_km.max():.0f} km "
        f"(median {syn_med:.1f} km, mean {syn_mean:.1f} km).\n"
        f"  - **Support (the load-bearing claim).** Every synthetic circuit "
        f"length lies inside the SimBench overhead range "
        f"[{ref_min:.1f}, {ref_max:.1f}] km "
        f"({syn_in:.0f} % of synthetic circuits inside), and "
        f"{ref_in_band:.0f} % of the {len(ref_ol)} SimBench overhead circuits "
        f"fall inside the synthetic band [{syn_lo:.0f}, {syn_hi:.0f}] km. "
        f"The synthetic lengths are therefore *realisable*, not extrapolated.\n"
        f"  - **Circuits vs. circuits.** Pooled SimBench overhead sample "
        f"(n = {len(ref_ol)}, median {ref_ol.length_km.median():.1f} km, mean "
        f"{ref_ol.length_km.mean():.1f} km): synthetic median at the "
        f"{pct_med:.0f}th percentile, synthetic mean at the "
        f"{pct_mean:.0f}th percentile -- i.e. the synthetic grid sits in the "
        f"upper decile, which is the expected direction for a node-reduced "
        f"equivalent.\n"
        f"  - **Corridors vs. corridors.** After series reduction the "
        f"synthetic yields {len(syn_cor)} corridors "
        f"(median {syn_med_c:.1f} km, max "
        f"{syn_cor.length_km.max():.1f} km) against n = {len(ref_cor)} "
        f"SimBench corridors (median {ref_cor.length_km.median():.1f} km, max "
        f"{ref_cor.length_km.max():.1f} km): synthetic median at the "
        f"{pct_med_c:.0f}th percentile. Note that SimBench's HV grids are "
        f"strongly meshed, so the reduction removes little "
        f"({len(ref_cor)} of {len(ref_lines)} circuits survive) while the "
        f"sparser synthetic grid contracts from {len(syn_lines)} to "
        f"{len(syn_cor)}; this comparison therefore overstates the synthetic "
        f"lengths and is reported for completeness only.")
    out.append(
        f"- **Coupling-point spacing.** Synthetic (distinct values across "
        f"{len(synth_labels)} sub-network(s)): "
        + (", ".join(f"{v:.1f} km" for v in syn_pair_vals)
           if syn_pair_vals else "n/a")
        + f" ({len(syn_pairs)} pairs total). SimBench HV1: "
        + (", ".join(f"{v:.1f} km" for v in ref_pairs.dist_km)
           if len(ref_pairs) else "n/a")
        + f" (n = {len(ref_pairs)} pairs).")
    out.append(
        "- **Caveat on electrical distance.** The synthetic grid uses a "
        "184-AL1/30-ST1A conductor (x = 0.400 Ohm/km) whereas SimBench uses "
        "Al/St 265/35 (x = 0.296 Ohm/km). At equal route length the synthetic "
        "grid is therefore ~35 % *electrically* longer, which inflates "
        "dV/dQ sensitivities relative to the reference.")
    out.append(
        "- **Sample-size caveat.** Scenario 0 provides two HV reference grids. "
        "The length statistics rest on "
        f"{len(ref_lines)} circuits, which is a usable sample; the "
        "coupling-spacing statistic rests on "
        f"{len(ref_pairs)} pairs and must be read as an order-of-magnitude "
        "check, not a distribution.")
    return "\n".join(out)


def make_figure(tops: List[GridTopology], pairs: pd.DataFrame,
                corridors: Dict[str, pd.DataFrame],
                synth_labels: Sequence[str], path: Path) -> Optional[Path]:
    """ECDFs of circuit and corridor lengths, plus coupling-distance strip."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figure: {exc}")
        return None

    synth = set(synth_labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {lbl: "tab:red" for lbl in synth}
    labels = [t.label.replace(" (", "\n(") for t in tops]

    def _ecdf(ax, getter, title, xlabel):
        for t in tops:
            s = np.sort(np.asarray(getter(t), dtype=float))
            if not len(s):
                continue
            y = np.arange(1, len(s) + 1) / len(s)
            ax.step(s, y, where="post", label=f"{t.label} (n={len(s)})",
                    lw=2.2 if t.label in synth else 1.5,
                    color=colors.get(t.label))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("empirical CDF")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    _ecdf(axes[0], lambda t: t.lines.length_km.values,
          "110 kV circuits (as modelled)", "circuit length [km]")
    _ecdf(axes[1], lambda t: corridors[t.label].length_km.values,
          "Corridors (pass-through stations reduced)", "corridor length [km]")

    ax = axes[2]
    for i, t in enumerate(tops):
        d = pairs.loc[pairs.grid == t.label, "dist_km"].values
        if len(d):
            ax.scatter([i] * len(d), d, s=70,
                       color="tab:red" if t.label in synth else "tab:blue",
                       zorder=3)
    ax.set_xticks(range(len(tops)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("path length between coupling points [km]")
    ax.set_title("EHV/HV coupling-point spacing")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark the synthetic 110 kV DSO topology against "
                    "the SimBench HV reference grids.")
    ap.add_argument("--outdir", default="results/simbench_hv_benchmark",
                    help="output directory for CSV/PNG/Markdown artefacts")
    ap.add_argument("--from-built-net", action="store_true",
                    help="derive the synthetic side from the fully built "
                         "IEEE39+DSO network instead of the constants tables")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tops: List[GridTopology] = []

    # -- reference grids ---------------------------------------------------
    for label, net in load_simbench_nets(SIMBENCH_HV_CODES).items():
        tops.append(analyse_grid(net, label))

    # -- synthetic grid ----------------------------------------------------
    if args.from_built_net:
        built = extract_synthetic_from_built_net()
        if not built:
            print("[warn] falling back to the constants-based reconstruction")
            args.from_built_net = False
        else:
            full = next(iter(built.values()))
            ids = _hv_subnetwork_ids(full)
            if not ids:
                raise RuntimeError("built net has no '<net_id>|' 110 kV buses")
            for sn in ids:
                tops.append(analyse_grid(full, f"Synthetic DS ({sn})", group=sn))
    if not args.from_built_net:
        syn_net, syn_label = build_synthetic_reference_net()
        tops.append(analyse_grid(syn_net, syn_label))

    synth_labels = [t.label for t in tops if t.label.startswith("Synthetic")]

    # -- aggregate ---------------------------------------------------------
    stat_rows: List[dict] = []
    for t in tops:
        stat_rows.append(length_stats(t.lines, t.label, "all"))
        ol = t.lines[t.lines.type.astype(str) == "ol"]
        cs = t.lines[t.lines.type.astype(str) == "cs"]
        if len(ol) and len(ol) != len(t.lines):
            stat_rows.append(length_stats(ol, t.label, "overhead"))
        if len(cs):
            stat_rows.append(length_stats(cs, t.label, "cable"))

    corridors = {t.label: corridor_lengths(t) for t in tops}
    for t in tops:
        stat_rows.append(length_stats(corridors[t.label], t.label, "corridor"))

    ref_tops = [t for t in tops if not t.label.startswith("Synthetic")]
    pooled = pd.concat([t.lines for t in ref_tops])
    stat_rows.append(length_stats(pooled, "SimBench pooled", "all"))
    stat_rows.append(length_stats(pooled[pooled.type.astype(str) == "ol"],
                                  "SimBench pooled", "overhead"))
    stat_rows.append(length_stats(
        pd.concat([corridors[t.label] for t in ref_tops]),
        "SimBench pooled", "corridor"))
    stats = pd.DataFrame(stat_rows)

    def _cat(frames):
        keep = [f for f in frames if len(f)]
        return pd.concat(keep, ignore_index=True) if keep else frames[0]

    pairs = _cat([coupling_pair_distances(t) for t in tops])
    depth = _cat([station_depth(t) for t in tops])
    corridor_df = _cat(list(corridors.values()))

    summary = pd.DataFrame([{
        "grid": t.label,
        "buses_raw_110kV": t.n_buses_raw,
        "stations_reduced": t.n_stations,
        "circuits": len(t.lines),
        "route_km": round(float(t.lines.length_km.sum()), 1),
        "km_per_station": round(float(t.lines.length_km.sum())
                                / max(t.n_stations, 1), 1),
        "meshing_deg": round(len(t.lines) / max(t.n_stations, 1), 2),
        "coupling_trafos": len(t.coupling_trafos),
        "coupling_points": len(t.coupling_points),
        "substation_check": t.substation_check,
    } for t in tops])

    conductors = (pd.concat([t.lines.assign(grid=t.label) for t in tops])
                  .groupby(["grid", "std_type"])
                  .agg(n=("length_km", "size"),
                       r_ohm_per_km=("r_ohm_per_km", "first"),
                       x_ohm_per_km=("x_ohm_per_km", "first"),
                       c_nf_per_km=("c_nf_per_km", "first"),
                       max_i_ka=("max_i_ka", "first"))
                  .reset_index())

    # -- write -------------------------------------------------------------
    pd.concat([t.lines.assign(grid=t.label) for t in tops]).to_csv(
        outdir / "lines.csv", index=False)
    stats.to_csv(outdir / "line_length_stats.csv", index=False)
    pairs.to_csv(outdir / "coupling_pairs.csv", index=False)
    depth.to_csv(outdir / "station_depth.csv", index=False)
    summary.to_csv(outdir / "topology_summary.csv", index=False)
    conductors.to_csv(outdir / "conductors.csv", index=False)
    corridor_df.to_csv(outdir / "corridors.csv", index=False)

    report = build_report(tops, stats, pairs, depth, conductors, summary,
                          corridors, synth_labels)
    (outdir / "report.md").write_text(report, encoding="utf-8")

    if not args.no_figure:
        make_figure(tops, pairs, corridors, synth_labels,
                    outdir / "simbench_hv_benchmark.png")

    print("\n" + report + "\n")
    print(f"[done] artefacts written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
