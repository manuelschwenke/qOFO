"""
analysis/dso_ts_coupling_geometry.py
====================================
Geometric consistency check between the 345 kV transmission network and the
synthetic 110 kV DSO sub-networks attached to it.

Question
--------
Each DSO in ``SUBNET_DEFS`` couples to the IEEE 39-bus TN at three buses
(``ieee_1idx``) via three transformers landing on three HV buses
(``hv_buses``), paired positionally::

    ieee_1idx=(7, 8, 5)  ->  hv_buses=(3, 0, 8)
    i.e. TN bus 7 <-> HV bus 3,  TN bus 8 <-> HV bus 0,  TN bus 5 <-> HV bus 8

A 380/110 kV (here 345/110 kV) coupling transformer sits *inside* the EHV
substation, so HV bus 3 is geographically at the site of TN bus 7, and so on.
The three HV coupling buses therefore span the same geography as the three TN
buses.  If the HV route between two coupling points is 60 km while the TN
route between the corresponding TN buses is 240 km, the sub-network is
geometrically inconsistent with the transmission grid it hangs off.

Consistency criterion
---------------------
Let :math:`D_{ij}` be the TN route distance between coupling buses *i*, *j*
and :math:`d_{ij}` the HV route distance between the corresponding HV buses at
``scale = 1``.  Define the *detour factor*

.. math::  r_{ij} = d_{ij} / D_{ij}

Physically :math:`r \\gtrsim 1`: EHV lines are built as direct corridors while
a 110 kV path between the same two points follows a meshed regional structure,
so the HV route is normally equal to or somewhat longer than the EHV route.
A plausible band is :math:`r \\in [1.0, 1.5]`.  This is a modelling
convention, not a measured law -- it is stated as an assumption, not a fact.

The recommended per-DSO scale factor for a target detour :math:`\\kappa` is

.. math::  s^\\star = \\kappa \\,/\\, \\mathrm{geomean}_{ij}(r_{ij})

The geometric mean is used because ``scale`` acts multiplicatively; it is the
least-squares estimator in log space and is not dominated by the single
largest pair the way an arithmetic fit is.

Two failure modes are reported separately
-----------------------------------------
1. **Magnitude** -- the whole sub-network is too small/large relative to its
   TN footprint.  Fixed by ``scale``.
2. **Ordering** -- the *permutation* is wrong: the TN pair that is farthest
   apart is mapped onto the HV pair that is closest together.  ``scale``
   cannot fix this; only reassigning ``hv_buses`` can.  The script searches
   all 6 assignments and reports the best one.

Usage
-----
::

    python -m analysis.dso_ts_coupling_geometry
    python -m analysis.dso_ts_coupling_geometry --kappa 1.25
    python -m analysis.dso_ts_coupling_geometry --outdir results/dso_ts_geometry

The script is read-only: it prints a recommendation and a ready-to-paste
``SUBNET_DEFS`` block but never edits ``network/ieee39/constants.py``.

Author: qOFO / analysis utilities
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

TN_KV: float = 345.0
HV_KV: float = 110.0
S_BASE_MVA: float = 100.0
Z_BASE_TN: float = TN_KV ** 2 / S_BASE_MVA     # 1190.25 Ohm
Z_BASE_HV: float = HV_KV ** 2 / S_BASE_MVA     # 121.0 Ohm

#: x/km of the HV conductor used by ``_create_hv_subnetwork``
#: (``184-AL1/30-ST1A 110.0``).
HV_X_OHM_PER_KM: float = 0.400

#: Longest 110 kV overhead circuit in the SimBench reference grids [km],
#: from ``analysis/simbench_hv_benchmark.py`` (HV2, `1-HV-urban--0-sw`).
#: Used as an upper admissibility bound on ``scale``: beyond it the synthetic
#: grid contains individual circuits longer than anything SimBench observes.
SIMBENCH_MAX_OVERHEAD_KM: float = 51.64


# --------------------------------------------------------------------------
#  Graphs
# --------------------------------------------------------------------------

def build_tn_graph(verbose: bool = False) -> Tuple[nx.Graph, nx.Graph,
                                                   Dict[int, List[int]]]:
    """
    Graphs of the 345 kV transmission network, 1-indexed IEEE bus labels.

    Returns ``(g_geo, g_elec, proxy)``:

    ``g_geo``
        **Lines only**, weighted by the realistic distances installed by
        ``fix_line_lengths`` (``LINE_LENGTHS_KM``).  Transformers are
        deliberately excluded from the geometric metric: a transformer has no
        geographic extent, so admitting it as a zero-length edge creates
        spurious shortcuts.  Concretely, IEEE 39-bus feeds bus 12 through two
        transformers from buses 11 and 13, which would make the route 11-12-13
        cost 0 km although 11 and 13 are two substations 57.6 km apart via
        bus 10.
    ``g_elec``
        Lines **and** transformers, weighted by series reactance -- the
        electrically correct graph, where the transformer branches are real
        impedance and must be traversable.
    ``proxy``
        For every bus with no incident line (bus 12, and the generator
        terminals 30--38), the line-connected buses it hangs off.  Geometric
        distances to such a bus are taken as the minimum over its proxies,
        i.e. it is treated as co-sited with its parent substation(s).
    """
    from network.ieee39.build import build_ieee39_net

    net, _meta = build_ieee39_net(verbose=False)

    g_geo = nx.Graph()
    g_elec = nx.Graph()
    for b in net.bus.index:
        g_geo.add_node(int(b) + 1)
        g_elec.add_node(int(b) + 1)

    stale: List[str] = []
    for li in net.line.index:
        f = int(net.line.at[li, "from_bus"]) + 1
        t = int(net.line.at[li, "to_bus"]) + 1
        km = float(net.line.at[li, "length_km"])
        x = float(net.line.at[li, "x_ohm_per_km"]) * km
        if abs(km - 1.0) < 1e-9:
            stale.append(f"{f}-{t}")       # never matched LINE_LENGTHS_KM
        for g, w in ((g_geo, km), (g_elec, x)):
            if g.has_edge(f, t):
                g[f][t]["km"] = min(g[f][t]["km"], km)
                g[f][t]["x_ohm"] = min(g[f][t]["x_ohm"], x)
            else:
                g.add_edge(f, t, km=km, x_ohm=x, kind="line")

    for ti in net.trafo.index:
        f = int(net.trafo.at[ti, "hv_bus"]) + 1
        t = int(net.trafo.at[ti, "lv_bus"]) + 1
        vk = float(net.trafo.at[ti, "vk_percent"])
        sn = float(net.trafo.at[ti, "sn_mva"])
        vn = float(net.trafo.at[ti, "vn_hv_kv"])
        x = (vk / 100.0) * (vn ** 2) / max(sn, 1e-6)
        if not g_elec.has_edge(f, t):
            g_elec.add_edge(f, t, km=0.0, x_ohm=x, kind="trafo")

    # Buses with no line: attach them to their transformer parents.
    proxy: Dict[int, List[int]] = {}
    lineless = [n for n in g_geo.nodes if g_geo.degree(n) == 0]
    for n in lineless:
        parents = [m for m in g_elec.neighbors(n) if g_geo.degree(m) > 0]
        if parents:
            proxy[n] = sorted(parents)

    if verbose:
        if stale:
            print(f"[warn] {len(stale)} TN line(s) kept the case39 default "
                  f"length_km = 1.0 (no LINE_LENGTHS_KM entry): "
                  f"{', '.join(stale)} -- these are generator step-up "
                  f"connections and do not lie on any coupling-bus path.")
        if proxy:
            print("[info] lineless buses proxied to their line-connected "
                  "parents: "
                  + ", ".join(f"{b}->{p}" for b, p in sorted(proxy.items())))
    return g_geo, g_elec, proxy


def build_hv_graph(scale: float = 1.0) -> nx.Graph:
    """Weighted graph of one synthetic HV sub-network at the given scale."""
    from network.ieee39.constants import HV_LINE_TOPOLOGY

    g = nx.Graph()
    g.add_nodes_from(range(10))
    for f, t, base_km in HV_LINE_TOPOLOGY:
        km = float(base_km) * scale
        g.add_edge(int(f), int(t), km=km, x_ohm=km * HV_X_OHM_PER_KM)
    return g


def _dist(g: nx.Graph, a: int, b: int, weight: str) -> float:
    if a == b:
        return 0.0
    try:
        return float(nx.shortest_path_length(g, a, b, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("nan")


def _geo_dist(g_geo: nx.Graph, proxy: Dict[int, List[int]],
              a: int, b: int) -> float:
    """Geometric TN distance, resolving lineless buses via their proxies."""
    aa = proxy.get(a, [a])
    bb = proxy.get(b, [b])
    vals = [_dist(g_geo, u, v, "km") for u in aa for v in bb]
    vals = [v for v in vals if v == v]
    return min(vals) if vals else float("nan")


# --------------------------------------------------------------------------
#  Per-DSO comparison
# --------------------------------------------------------------------------

def compare_dso(g_geo: nx.Graph, g_elec: nx.Graph, proxy: Dict[int, List[int]],
                hv: nx.Graph, sdef: dict) -> pd.DataFrame:
    """Pairwise TN vs HV distances for one DSO under its current assignment."""
    tn_buses: Sequence[int] = sdef["ieee_1idx"]
    hv_buses: Sequence[int] = sdef["hv_buses"]

    rows = []
    for (i, j) in itertools.combinations(range(len(tn_buses)), 2):
        a_tn, b_tn = int(tn_buses[i]), int(tn_buses[j])
        a_hv, b_hv = int(hv_buses[i]), int(hv_buses[j])
        d_tn = _geo_dist(g_geo, proxy, a_tn, b_tn)
        x_tn = _dist(g_elec, a_tn, b_tn, "x_ohm")
        d_hv = _dist(hv, a_hv, b_hv, "km")
        x_hv = _dist(hv, a_hv, b_hv, "x_ohm")
        rows.append(dict(
            dso=sdef["net_id"], zone=sdef["zone"],
            tn_pair=f"{a_tn}-{b_tn}", hv_pair=f"{a_hv}-{b_hv}",
            tn_km=d_tn, hv_km=d_hv, detour=d_hv / d_tn if d_tn else np.nan,
            tn_x_pu=x_tn / Z_BASE_TN, hv_x_pu=x_hv / Z_BASE_HV,
        ))
    return pd.DataFrame(rows)


def geomean(vals: Sequence[float]) -> float:
    a = np.asarray([v for v in vals if v == v and v > 0], dtype=float)
    return float(np.exp(np.log(a).mean())) if a.size else float("nan")


def rank_agreement(tn_km: Sequence[float], hv_km: Sequence[float]) -> str:
    """Do TN and HV order the three pairs the same way?"""
    t = np.argsort(np.argsort(np.asarray(tn_km, dtype=float)))
    h = np.argsort(np.argsort(np.asarray(hv_km, dtype=float)))
    if np.array_equal(t, h):
        return "match"
    if np.array_equal(t, h[::-1]) and len(t) == 3:
        return "INVERTED"
    return "partial"


def best_assignment(g_geo: nx.Graph, proxy: Dict[int, List[int]],
                    hv: nx.Graph, sdef: dict) -> dict:
    """
    Search all permutations of ``hv_buses`` against the fixed ``ieee_1idx``.

    Score = spread of the per-pair detour factors, measured as the standard
    deviation of ``log(detour)``.  A permutation with a low spread can be
    corrected by a single ``scale``; a high spread means no scalar fixes it.
    """
    tn_buses = [int(b) for b in sdef["ieee_1idx"]]
    hv_buses = [int(b) for b in sdef["hv_buses"]]

    results = []
    for perm in itertools.permutations(hv_buses):
        ratios, tn_km, hv_km = [], [], []
        for (i, j) in itertools.combinations(range(len(tn_buses)), 2):
            d_tn = _geo_dist(g_geo, proxy, tn_buses[i], tn_buses[j])
            d_hv = _dist(hv, perm[i], perm[j], "km")
            tn_km.append(d_tn)
            hv_km.append(d_hv)
            ratios.append(d_hv / d_tn if d_tn else np.nan)
        valid = [r for r in ratios if r == r and r > 0]
        spread = float(np.std(np.log(valid), ddof=0)) if len(valid) > 1 else np.nan
        results.append(dict(perm=perm, geomean_detour=geomean(ratios),
                            log_spread=spread,
                            rank=rank_agreement(tn_km, hv_km)))
    results.sort(key=lambda r: (r["log_spread"]
                                if r["log_spread"] == r["log_spread"] else 1e9))
    current = next(r for r in results if list(r["perm"]) == hv_buses)
    return dict(current=current, best=results[0], all=results)


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check geometric consistency between the 345 kV TN and "
                    "the synthetic 110 kV DSO sub-networks.")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="target detour factor d_HV / D_TN (default 1.0)")
    ap.add_argument("--outdir", default="results/dso_ts_geometry")
    ap.add_argument("--simbench-max-km", type=float,
                    default=SIMBENCH_MAX_OVERHEAD_KM,
                    help="longest admissible single circuit [km]; the "
                         "SimBench observed maximum by default")
    args = ap.parse_args(argv)

    from network.ieee39.constants import HV_LINE_TOPOLOGY, SUBNET_DEFS

    longest_base_km = max(float(k) for _f, _t, k in HV_LINE_TOPOLOGY)
    scale_cap = args.simbench_max_km / longest_base_km

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[build] IEEE 39-bus TN graph ...")
    g_geo, g_elec, proxy = build_tn_graph(verbose=True)
    hv1 = build_hv_graph(scale=1.0)

    frames, recs = [], []
    for sdef in SUBNET_DEFS:
        df = compare_dso(g_geo, g_elec, proxy, hv1, sdef)
        frames.append(df)

        gm = geomean(df.detour)
        s_star = args.kappa / gm if gm == gm and gm > 0 else np.nan
        assign = best_assignment(g_geo, proxy, hv1, sdef)
        recs.append(dict(
            dso=sdef["net_id"], zone=sdef["zone"],
            tn_buses=str(tuple(sdef["ieee_1idx"])),
            hv_buses_current=str(tuple(sdef["hv_buses"])),
            scale_current=float(sdef["scale"]),
            tn_km_min=float(df.tn_km.min()), tn_km_max=float(df.tn_km.max()),
            hv_km_min=float(df.hv_km.min()), hv_km_max=float(df.hv_km.max()),
            detour_geomean=gm,
            detour_min=float(df.detour.min()), detour_max=float(df.detour.max()),
            rank=rank_agreement(df.tn_km, df.hv_km),
            scale_recommended=s_star,
            # Residual at the scale currently configured in SUBNET_DEFS.
            # Equals kappa when the recommendation is already applied.
            detour_at_configured=gm * float(sdef["scale"]),
            log_spread_current=assign["current"]["log_spread"],
            hv_buses_best=str(assign["best"]["perm"]),
            log_spread_best=assign["best"]["log_spread"],
            scale_if_reassigned=(args.kappa / assign["best"]["geomean_detour"]
                                 if assign["best"]["geomean_detour"] > 0
                                 else np.nan),
            scale_admissible=min(s_star, scale_cap) if s_star == s_star else np.nan,
            longest_circuit_km=(s_star * longest_base_km
                                if s_star == s_star else np.nan),
            simbench_ok=bool(s_star == s_star and s_star <= scale_cap),
        ))

    pairs = pd.concat(frames, ignore_index=True)
    rec = pd.DataFrame(recs)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print("\n" + "=" * 96)
    print("PAIRWISE DISTANCES  (TN route vs HV route at scale = 1.0)")
    print("=" * 96)
    print(pairs.round(3).to_string(index=False))

    print("\n" + "=" * 96)
    print(f"PER-DSO SUMMARY  (target detour kappa = {args.kappa:g})")
    print("=" * 96)
    cols = ["dso", "tn_buses", "hv_buses_current", "tn_km_min", "tn_km_max",
            "hv_km_min", "hv_km_max", "detour_geomean", "detour_min",
            "detour_max", "rank", "scale_current", "scale_recommended",
            "detour_at_configured"]
    print(rec[cols].round(3).to_string(index=False))

    print("\n" + "=" * 96)
    print("ORDERING CHECK  (can a scalar `scale` fix this DSO?)")
    print("=" * 96)
    cols2 = ["dso", "rank", "log_spread_current", "hv_buses_best",
             "log_spread_best", "scale_if_reassigned"]
    print(rec[cols2].round(3).to_string(index=False))
    print("\nlog_spread = std of log(detour) across the 3 pairs. 0 means one "
          "scale factor makes every pair match exactly; larger means the "
          "shape of the HV grid does not match the TN footprint and no "
          "scalar can fix it.")

    print("\n" + "=" * 96)
    print("CROSS-CHECK vs SimBench circuit lengths")
    print("=" * 96)
    print(f"Longest base circuit in HV_LINE_TOPOLOGY: {longest_base_km:.0f} km. "
          f"Admissible scale cap = {args.simbench_max_km:.1f} / "
          f"{longest_base_km:.0f} = {scale_cap:.2f}\n"
          f"(above it the sub-network holds circuits longer than any of the "
          f"{args.simbench_max_km:.1f} km SimBench maximum).")
    cols3 = ["dso", "scale_recommended", "longest_circuit_km", "simbench_ok",
             "scale_admissible"]
    print(rec[cols3].round(3).to_string(index=False))

    print("\n" + "=" * 96)
    print("PROPOSED SUBNET_DEFS  (scale only, hv_buses unchanged)")
    print("=" * 96)
    for sdef, r in zip(SUBNET_DEFS, recs):
        s = r["scale_recommended"]
        print(f'    dict(net_id="{sdef["net_id"]}", zone={sdef["zone"]},')
        print(f'         ieee_1idx={tuple(sdef["ieee_1idx"])}, '
              f'hv_buses={tuple(sdef["hv_buses"])}, '
              f'scale={s:.2f}, gen="{sdef["gen"]}"),'
              f'  # was {sdef["scale"]:.2f}')

    pairs.to_csv(outdir / "pairwise_distances.csv", index=False)
    rec.to_csv(outdir / "recommendation.csv", index=False)
    print(f"\n[done] artefacts written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
