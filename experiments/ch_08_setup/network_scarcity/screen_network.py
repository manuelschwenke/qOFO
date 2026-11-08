#!/usr/bin/env python3
"""
experiments/ch_08_setup/network_scarcity/screen_network.py
==========================================================

Static screen of the network redesign for reactive scarcity (author
2026-11-08).  Decides, before any tuning or simulation campaign, whether a
candidate network operates well over the profile year: initialisation,
N-1 security, voltages, thermal loading, and how scarce the transmission areas
are of their own reactive reserve while the STSs have capability.

Candidates (``network.ieee39.scenarios``):

====  =======================  ===============================================
key   scenario                 change against the base
====  =======================  ===============================================
B     rural_700                none (the network of every result so far)
R     rural_700_res            coupling-bus parks with reserved capability
N     rural_700_north          W1, W2, W4 replaced by one park at IEEE bus 26
NR    rural_700_north_res      both
RT    rural_700_res_no939      R, and line 9-39 not built
NRT   rural_700_north_res_no9  NR, and line 9-39 not built
====  =======================  ===============================================

``--north-mw`` overrides ``NORTH_WIND_MW`` (1500) for a size sweep; the
scenario key then carries the size (``N1200``).

Operating points: the start of every drawn episode of the Case-2 MC prescreen
(the MC's own draw rule and unit commitment, 180-min window), the design
windows W1--W6, the step-test OPs, and the extreme wind and load hours of the
year.  Each is initialised by the runner itself (LOC configuration, no tuned
weight involved): ``run_multi_tso_dso`` up to its pre-loop hook.

Per snapshot:

* init: converged or the exception;
* voltages (TS per area, HV per STS), 345 kV line loading against the case39
  ratings, coupler loading;
* per area: machine Q and own reserve (machines without G1, plus TS IBGs),
  TS-IBG capability, reactive losses, tie flows; the equivalent G1;
* per STS: interface P/Q, IBG Q capability under the 4120 v2 diagram and under
  reserved capability;
* N-1 (every admissible 345 kV line, every committed non-slack machine), in
  solve mode ``--mode`` (below);
* own-reserve ladder per area 2 and 3: the area's reactive demand is raised in
  steps, spread evenly over its 345 kV load buses (constant Q), while only the
  area's own transmission resources respond -- machines under AVR with their
  Q box, TS IBGs as voltage sources inside their capability -- and every STS
  is frozen.  Reported: the step at which the own resources are exhausted,
  at which the area's lowest voltage leaves 0.95 pu, at which the solve needs
  a machine beyond its Q box, and at which it fails.  This is the scarcity
  measure: the reactive demand the area can meet without STS help.

Solve modes: ``frozen`` -- tap relays and every IBG Q(V) loop held at their
pre-disturbance state, slack Q limit active, machine Q limits enforced (the
instant after a disturbance; fast); ``loops`` -- as the runner after a
disturbance (relays held, Q(V) loops and slack Q limit iterated,
``max_iter=300``; 5-9 s per solve).  The own-reserve ladder always uses its
own setting above.

One JSON per (operating point, candidate) in ``<run>/ops/``; a rerun with
``--resume <run>`` computes only the missing ones.  ``--report`` writes the
tables.

Run:
    python -m experiments.ch_08_setup.network_scarcity.screen_network \
        --jobs 12 [--candidates B,R,N,NR] [--limit-mc 400] [--tag ...]

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import contextlib
import copy
import dataclasses
import datetime as _dt
import io
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.thesis_runs import add_run_arguments, start_thesis_run  # noqa: E402

TARGET = ("ch:setup:multi:windreplace",)
CANDIDATES = {"B": "rural_700", "R": "rural_700_res",
              "N": "rural_700_north", "NR": "rural_700_north_res",
              "RT": "rural_700_res_no939", "NRT": "rural_700_north_res_no939",
              "BG600": "rural_700", "RG400": "rural_700_res",
              "NRG600": "rural_700_north_res", "NRG400": "rural_700_north_res"}
#: Config overrides per candidate (the G1 limit is a config value, 2026-11-08).
OVERRIDES = {"BG600": {"slack_q_limit_mvar": 600.0},
             "RG400": {"slack_q_limit_mvar": 400.0},
             "NRG600": {"slack_q_limit_mvar": 600.0},
             "NRG400": {"slack_q_limit_mvar": 400.0}}
HORIZON_MIN = 180                       # the MC window, also the UC window
MC_RUN = Path(r"F:\qOFO_res\results\ch_11_case2_multisystem\mc"
              r"\0002_2026-11-07_183649_run_mc_newbase")
#: Design windows of Ch 8 (tuning.scenarios_mc_v2.CONDITIONS) and step OPs.
FIXED_OPS = {
    "W1": "2016-01-05T08:00", "W2": "2016-02-22T13:00", "W3": "2016-01-04T08:00",
    "W4": "2016-11-26T10:00", "W5": "2016-02-03T20:00", "W6": "2016-07-18T14:00",
    "OP-H": "2016-04-13T09:30", "OP-M": "2016-11-01T19:30",
}
#: Own-reserve ladder: areas and total added reactive demand per step [Mvar].
LADDER_ZONES = (2, 3)
LADDER_MVAR = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0,
               800.0, 1000.0, 1250.0, 1500.0)
V_LOW = 0.95
#: Lines whose loading is reported individually (IEEE numbering).
CORRIDOR = ("25-26", "26-27", "26-28", "26-29", "28-29", "17-27", "2-3",
            "1-2", "2-25", "1-39", "9-39", "8-9", "3-4", "3-18", "14-15",
            "16-17", "17-18", "4-14", "5-8", "6-11")


# ---------------------------------------------------------------------------
#  Operating points
# ---------------------------------------------------------------------------
def extreme_ops(n_each: int = 6) -> Dict[str, str]:
    """Extreme hours of the year for the zone-1 park (profile WP10) and load."""
    from core.profiles import load_profiles
    prof = load_profiles(str(_ROOT / "data" / "profiles.csv"), timestep_min=15)
    load = prof[[c for c in ("HS4_pload", "HS5_pload", "mv_rural_pload")
                 if c in prof.columns]].mean(axis=1)
    wind = prof["WP10"]
    pv = prof["PV3"] if "PV3" in prof.columns else wind * 0.0
    df = pd.DataFrame({"wind": wind, "load": load, "pv": pv})
    df = df[(df.index.minute == 0) & (df.index <= df.index.max()
                                      - pd.Timedelta(hours=4))]
    out: Dict[str, str] = {}

    def pick(tag: str, sub: pd.DataFrame, by: str, asc: bool) -> None:
        days = set()
        for ts, _row in sub.sort_values(by, ascending=asc).iterrows():
            if ts.date() in days:
                continue
            days.add(ts.date())
            out[f"{tag}{len(days)}"] = ts.strftime("%Y-%m-%dT%H:%M")
            if len(days) >= n_each:
                break

    hi = df[df.wind >= df.wind.quantile(0.98)]
    pick("XWL", hi, "load", True)         # full wind, light load (export)
    pick("XWH", hi, "load", False)        # full wind, heavy load (transit)
    lo = df[df.wind <= df.wind.quantile(0.05)]
    pick("XCH", lo, "load", False)        # calm, heavy load (upward need)
    pick("XPL", df[df.pv >= df.pv.quantile(0.98)], "load", True)  # PV, light
    return out


def operating_points(limit_mc: Optional[int]) -> List[Tuple[str, str, List[int]]]:
    """``(op id, start ISO, offline machines from the MC prescreen or [])``."""
    ops: List[Tuple[str, str, List[int]]] = []
    files = sorted((MC_RUN / "prescreen").glob("*.json"))
    if limit_mc is not None:
        files = files[:limit_mc]
    for f in files:
        d = json.loads(f.read_text())
        ops.append((f"MC{d['seed']}", d["start_time"][:16],
                    list(d.get("uc_offline") or [])))
    for k, v in {**FIXED_OPS, **extreme_ops()}.items():
        ops.append((k, v, []))
    return ops


# ---------------------------------------------------------------------------
#  One snapshot
# ---------------------------------------------------------------------------
def _ieee(net, li: int) -> str:
    return (f"{int(net.line.at[li, 'from_bus']) + 1}-"
            f"{int(net.line.at[li, 'to_bus']) + 1}")


def _solve(net, cfg, mode: str = "loops") -> str:
    """Plant solve after a disturbance (see the module docstring for the
    modes), machine Q limits enforced; if that fails, the runner's fallback
    without them.  Returns ``"qlim"`` or ``"noqlim"`` (= solvable only with
    some machine beyond its Q box, i.e. AVR saturation would follow)."""
    from core.plant import runpp_relays_held
    from controller.der_qv_local_loop import QVLocalLoop
    kw = dict(calculate_voltage_angles=True, max_iteration=50, max_iter=300,
              init="results", distributed_slack=bool(cfg.distributed_slack))
    held = []
    if mode == "frozen" and len(net.controller):
        held = [i for i in net.controller.index
                if isinstance(net.controller.at[i, "object"], QVLocalLoop)
                and bool(net.controller.at[i, "in_service"])]
        net.controller.loc[held, "in_service"] = False
    try:
        try:
            runpp_relays_held(net, enforce_q_lims=bool(cfg.enforce_q_lims_plant), **kw)
            return "qlim"
        except Exception:                         # noqa: BLE001
            runpp_relays_held(net, enforce_q_lims=False, **kw)
            return "noqlim"
    finally:
        if held:
            net.controller.loc[held, "in_service"] = True


def _capability(sn: float, op: str, p: float) -> Tuple[float, float]:
    from controller.der_qv_local_loop import _qv_capability
    return _qv_capability(float(sn), str(op), float(p))


class _Topo:
    """Zones, ties and element sets of one initialised network."""

    def __init__(self, net, meta, zone_defs):
        self.bus_zone: Dict[int, int] = {}
        for z, zd in zone_defs.items():
            for b in zd.bus_indices:
                self.bus_zone[int(b)] = int(z)
        for g in net.gen.index:                     # machine terminals
            gb = int(net.gen.at[g, "bus"])
            for t in net.trafo.index:
                hv, lv = int(net.trafo.at[t, "hv_bus"]), int(net.trafo.at[t, "lv_bus"])
                if lv == gb and hv in self.bus_zone:
                    self.bus_zone.setdefault(gb, self.bus_zone[hv])
        self.zones = sorted(zone_defs)
        tn = {int(b) for b in net.bus.index if float(net.bus.at[b, "vn_kv"]) >= 200}
        self.tn_buses = tn
        self.tn_lines = [int(li) for li in net.line.index
                         if int(net.line.at[li, "from_bus"]) in tn
                         and int(net.line.at[li, "to_bus"]) in tn]
        # aux buses (TN_AUX) inherit the zone of their parent
        for li in self.tn_lines:
            fb, tb = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
            if fb in self.bus_zone and tb not in self.bus_zone:
                self.bus_zone[tb] = self.bus_zone[fb]
            elif tb in self.bus_zone and fb not in self.bus_zone:
                self.bus_zone[fb] = self.bus_zone[tb]
        self.rated = {li for li in self.tn_lines
                      if str(net.line.at[li, "subnet"]) == "TN"
                      and bool(net.line.at[li, "in_service"])}
        self.slack = {int(g) for g in net.gen.index if bool(net.gen.at[g, "slack"])}
        self.gen_zone = {int(g): self.bus_zone.get(int(net.gen.at[g, "bus"]), -1)
                         for g in net.gen.index}
        self.ts_ibg = [int(s) for s in net.sgen.index
                       if str(net.sgen.at[s, "subnet"]) == "TN"]
        self.hv = list(meta.hv_networks)
        self.n1_lines = self._admissible(net)

    def _admissible(self, net) -> List[int]:
        import networkx as nx
        import pandapower.topology as top
        g_all = top.create_nxgraph(net, multi=True, include_trafo3ws=False)
        g_tn = g_all.subgraph(self.tn_buses).copy()
        out = []
        for li in sorted(self.rated):
            g = g_tn.copy()
            fb, tb = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
            for key in [k for k in g[fb][tb] if k == ("line", li)]:
                g.remove_edge(fb, tb, key)
            if nx.number_connected_components(g) == 1:
                out.append(li)
        return out


def _voltages(net, T: _Topo) -> Dict[str, float]:
    v = net.res_bus.vm_pu
    out: Dict[str, float] = {}
    for z in T.zones:
        bz = [b for b, zz in T.bus_zone.items() if zz == z and b in T.tn_buses]
        out[f"v_min_{z}"] = float(v[bz].min())
        out[f"v_max_{z}"] = float(v[bz].max())
    hvb = [int(b) for hv in T.hv for b in hv.bus_indices]
    out["v_min_hv"] = float(v[hvb].min())
    out["v_max_hv"] = float(v[hvb].max())
    return out


def _loading(net, T: _Topo) -> Dict[str, Any]:
    ld = net.res_line.loading_percent
    rated = sorted(T.rated)
    s = ld[rated]
    li = int(s.idxmax())
    out: Dict[str, Any] = {"line_max": float(s.max()), "line_max_at": _ieee(net, li),
                           "n_line_over100": int((s > 100.0).sum())}
    lab = {_ieee(net, i): float(ld[i]) for i in rated}
    for c in CORRIDOR:
        if c in lab:
            out[f"ld_{c}"] = lab[c]
    if len(net.trafo3w):
        out["coupler_max"] = float(net.res_trafo3w.loading_percent.max())
    return out


def _area(net, T: _Topo) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rg = net.res_gen
    for z in T.zones:
        gens = [g for g, zz in T.gen_zone.items() if zz == z and g not in T.slack
                and bool(net.gen.at[g, "in_service"])]
        q = float(rg.q_mvar[gens].sum()) if gens else 0.0
        up = float(sum(net.gen.at[g, "max_q_mvar"] - rg.at[g, "q_mvar"] for g in gens))
        dn = float(sum(rg.at[g, "q_mvar"] - net.gen.at[g, "min_q_mvar"] for g in gens))
        ibg = [s for s in T.ts_ibg if T.bus_zone.get(int(net.sgen.at[s, "bus"])) == z]
        iq = ip = iup = idn = 0.0
        for s in ibg:
            if not bool(net.sgen.at[s, "in_service"]):
                continue
            p = float(net.res_sgen.at[s, "p_mw"])
            qq = float(net.res_sgen.at[s, "q_mvar"])
            lo, hi = _capability(net.sgen.at[s, "sn_mva"], net.sgen.at[s, "op_diagram"], p)
            iq += qq; ip += p; iup += hi - qq; idn += qq - lo
        out.update({f"sg_n_{z}": len(gens), f"sg_q_{z}": q, f"sg_up_{z}": up,
                    f"sg_dn_{z}": dn, f"ibg_p_{z}": ip, f"ibg_q_{z}": iq,
                    f"ibg_up_{z}": iup, f"ibg_dn_{z}": idn,
                    f"own_up_{z}": up + iup, f"own_dn_{z}": dn + idn,
                    f"sg_p_{z}": float(rg.p_mw[gens].sum()) if gens else 0.0})
        # reactive losses of the area's 345 kV lines (ties counted half)
        ql = 0.0
        for li in T.rated:
            fz = T.bus_zone.get(int(net.line.at[li, "from_bus"]))
            tz = T.bus_zone.get(int(net.line.at[li, "to_bus"]))
            w = (1.0 if fz == tz == z else 0.5 if z in (fz, tz) else 0.0)
            ql += w * float(net.res_line.at[li, "ql_mvar"])
        out[f"qloss_{z}"] = ql
        loads = [i for i in net.load.index
                 if T.bus_zone.get(int(net.load.at[i, "bus"])) == z]
        out[f"load_p_{z}"] = float(net.res_load.p_mw[loads].sum())
        out[f"load_q_{z}"] = float(net.res_load.q_mvar[loads].sum())
    # ties: P and Q leaving the lower-numbered area at the from end
    for li in T.rated:
        fz = T.bus_zone.get(int(net.line.at[li, "from_bus"]))
        tz = T.bus_zone.get(int(net.line.at[li, "to_bus"]))
        if fz is not None and tz is not None and fz != tz:
            end = "from" if fz < tz else "to"
            out[f"tie_p_{_ieee(net, li)}"] = float(net.res_line.at[li, f"p_{end}_mw"])
            out[f"tie_q_{_ieee(net, li)}"] = float(net.res_line.at[li, f"q_{end}_mvar"])
    g1 = sorted(T.slack)[0]
    out["g1_p"] = float(rg.at[g1, "p_mw"])
    out["g1_q"] = float(rg.at[g1, "q_mvar"])
    return out


def _sts(net, T: _Topo) -> Dict[str, Any]:
    from core.actuator_bounds import RESERVED_OP_DIAGRAM
    out: Dict[str, Any] = {}
    for hv in T.hv:
        k = hv.net_id
        cp = list(hv.coupling_trafo_indices)
        out[f"if_p_{k}"] = float(net.res_trafo3w.p_hv_mw[cp].sum())
        out[f"if_q_{k}"] = float(net.res_trafo3w.q_hv_mvar[cp].sum())
        up4 = dn4 = upr = dnr = q = p = 0.0
        for s in hv.sgen_indices:
            if not bool(net.sgen.at[s, "in_service"]):
                continue
            ps = float(net.res_sgen.at[s, "p_mw"])
            sn = float(net.sgen.at[s, "sn_mva"])
            op = str(net.sgen.at[s, "op_diagram"])
            coupling = "WP_STATCOM_HV" in str(net.sgen.at[s, "name"])
            lo4, hi4 = _capability(sn, "VDE-AR-N-4120-v2", ps)
            lor, hir = _capability(sn, RESERVED_OP_DIAGRAM if coupling else op, ps)
            up4 += hi4; dn4 += -lo4; upr += hir; dnr += -lor
            q += float(net.res_sgen.at[s, "q_mvar"]); p += ps
        out.update({f"ibg_p_{k}": p, f"ibg_q_{k}": q, f"cap_up4_{k}": up4,
                    f"cap_dn4_{k}": dn4, f"cap_upR_{k}": upr, f"cap_dnR_{k}": dnr,
                    f"zone_{k}": int(hv.zone)})
    return out


def _probe(net, cfg, T: _Topo, mode: str) -> Dict[str, Any]:
    """Converged?, V range, max loading of an already modified copy."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            how = _solve(net, cfg, mode)
    except Exception as exc:                      # noqa: BLE001
        return {"ok": False, "err": type(exc).__name__}
    v = net.res_bus.vm_pu[sorted(T.tn_buses)]
    ld = net.res_line.loading_percent[sorted(T.rated)]
    hv = net.res_bus.vm_pu[[int(b) for h in T.hv for b in h.bus_indices]]
    return {"ok": True, "how": how, "v_min": float(v.min()),
            "v_max": float(v.max()), "v_min_hv": float(hv.min()),
            "v_max_hv": float(hv.max()), "ld_max": float(ld.max()),
            "ld_at": _ieee(net, int(ld.idxmax()))}


def _n1(net, cfg, T: _Topo, mode: str) -> List[Dict[str, Any]]:
    rows = []
    for li in T.n1_lines:
        n = copy.deepcopy(net)
        n.line.at[li, "in_service"] = False
        rows.append({"kind": "line", "el": _ieee(net, li), **_probe(n, cfg, T, mode)})
    for g in sorted(T.gen_zone):
        if g in T.slack or not bool(net.gen.at[g, "in_service"]):
            continue
        n = copy.deepcopy(net)
        n.gen.at[g, "in_service"] = False
        rows.append({"kind": "gen", "el": str(net.gen.at[g, "name"]),
                     **_probe(n, cfg, T, mode)})
    return rows


def _own_resources_only(n, T: _Topo, zone: int) -> List[int]:
    """Make the zone's TS IBGs voltage sources inside their capability (as a
    TS controller would use them) and hold every IBG Q(V) loop; returns the
    temporary machine indices."""
    import pandapower as pp
    from controller.der_qv_local_loop import QVLocalLoop
    tmp = []
    for s_ in T.ts_ibg:
        b = int(n.sgen.at[s_, "bus"])
        if T.bus_zone.get(b) != zone or not bool(n.sgen.at[s_, "in_service"]):
            continue
        p = float(n.res_sgen.at[s_, "p_mw"])
        lo, hi = _capability(n.sgen.at[s_, "sn_mva"], n.sgen.at[s_, "op_diagram"], p)
        if hi - lo < 1e-6:
            continue                              # no capability: stays a PQ source
        n.sgen.at[s_, "in_service"] = False
        tmp.append(int(pp.create_gen(n, bus=b, p_mw=p,
                                     vm_pu=float(n.res_bus.at[b, "vm_pu"]),
                                     min_q_mvar=lo, max_q_mvar=hi,
                                     slack_weight=0.0, name="_OWN_IBG")))
    loops = [i for i in n.controller.index
             if isinstance(n.controller.at[i, "object"], QVLocalLoop)]
    n.controller.loc[loops, "in_service"] = False
    return tmp


def _ladder(net, cfg, T: _Topo, zone: int) -> Dict[str, Any]:
    """Own-reserve ladder of ``zone`` (module docstring)."""
    import pandapower as pp
    from core.plant import runpp_relays_held
    gens = [g for g, zz in T.gen_zone.items() if zz == zone and g not in T.slack
            and bool(net.gen.at[g, "in_service"])]
    bz = [b for b, zz in T.bus_zone.items() if zz == zone and b in T.tn_buses]
    lbuses = sorted({int(net.load.at[i, "bus"]) for i in net.load.index
                     if T.bus_zone.get(int(net.load.at[i, "bus"])) == zone
                     and str(net.load.at[i, "subnet"]).startswith("TN")})
    n = copy.deepcopy(net)
    tmp = _own_resources_only(n, T, zone)
    own = gens + tmp
    lds = [int(pp.create_load(n, bus=b, p_mw=0.0, q_mvar=0.0, name="_LADDER"))
           for b in lbuses]
    for c in [c for c in n.load.columns if c.startswith("const_")]:
        n.load.loc[lds, c] = 0.0                  # constant-Q stress load
    cps = [t for hv in T.hv if int(hv.zone) == zone
           for t in hv.coupling_trafo_indices]
    ties = [li for li in T.rated
            if (T.bus_zone.get(int(net.line.at[li, "from_bus"])) == zone)
            != (T.bus_zone.get(int(net.line.at[li, "to_bus"])) == zone)]
    res: Dict[str, Any] = {"n_load_buses": len(lbuses), "own_lim": None,
                           "v_low": None, "qlim_fail": None, "pf_fail": None,
                           "steps": [], "cols": ["dq", "vmin", "own_head",
                                                 "g1_q", "sts_q_in", "tie_q_in"]}
    kw = dict(calculate_voltage_angles=True, max_iteration=50, max_iter=300,
              init="results", distributed_slack=bool(cfg.distributed_slack))
    for dq in LADDER_MVAR:
        n.load.loc[lds, "q_mvar"] = dq / max(len(lds), 1)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    runpp_relays_held(n, enforce_q_lims=True, **kw)
                except Exception:                 # noqa: BLE001
                    runpp_relays_held(n, enforce_q_lims=False, **kw)
                    if res["qlim_fail"] is None:
                        res["qlim_fail"] = dq
        except Exception:                         # noqa: BLE001
            res["pf_fail"] = dq
            break
        vmin = float(n.res_bus.vm_pu[bz].min())
        head = float(sum(n.gen.at[g, "max_q_mvar"] - n.res_gen.at[g, "q_mvar"]
                         for g in own)) if own else 0.0
        sts_q = -float(n.res_trafo3w.q_hv_mvar[cps].sum()) if cps else 0.0
        tie_q = 0.0                               # Q flowing into the zone
        for li in ties:
            fz = T.bus_zone.get(int(net.line.at[li, "from_bus"]))
            tie_q += (float(n.res_line.at[li, "q_from_mvar"]) if fz != zone
                      else float(n.res_line.at[li, "q_to_mvar"]))
        res["steps"].append([dq, vmin, head,
                             float(n.res_gen.q_mvar[sorted(T.slack)].sum()),
                             sts_q, tie_q])
        if res["own_lim"] is None and head < 1.0:
            res["own_lim"] = dq
        if res["v_low"] is None and vmin < V_LOW:
            res["v_low"] = dq
    return res


def snapshot(op_id: str, start: str, offline_mc: List[int], cand: str,
             north_mw: Optional[float], out_dir: str,
             mode: str = "frozen") -> Dict[str, Any]:
    """Initialise one candidate at one OP and measure it; writes the JSON."""
    t0 = time.perf_counter()
    key = cand if north_mw is None else f"{cand}{int(north_mw)}"
    path = Path(out_dir) / f"{op_id}__{key}{'' if mode == 'frozen' else '__' + mode}.json"
    rec: Dict[str, Any] = {"op": op_id, "start": start, "cand": key,
                           "scenario": CANDIDATES[cand], "uc_offline_mc": offline_mc}
    try:
        import network.ieee39.scenarios.wind_replace as wr
        if north_mw is not None:
            wr.NORTH_WIND_MW = float(north_mw)
        from experiments.ch_11_case2_multisystem import variants as case2
        from experiments.runners.multi_tso_dso import run_multi_tso_dso
        cfg = case2.resolved("L2", HORIZON_MIN, [], None)
        cfg = dataclasses.replace(
            cfg, scenario=CANDIDATES[cand],
            start_time=_dt.datetime.fromisoformat(start), verbose=0,
            result_dir=str(Path(out_dir).parent / "_scratch"),
            **OVERRIDES.get(cand, {}))
        rec["overrides"] = OVERRIDES.get(cand, {})
        cap: Dict[str, Any] = {}

        def hook(state):
            cap.update(state)
            return True

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                run_multi_tso_dso(cfg, pre_loop_hook=hook)
        except Exception as exc:                  # noqa: BLE001
            rec.update(init_ok=False, err=f"{type(exc).__name__}: {exc}"[:400])
            raise _Done()
        net, meta, zd = cap["net"], cap["meta"], cap["zone_defs"]
        rec["init_ok"] = True
        rec["seed_note"] = [l for l in buf.getvalue().splitlines()
                            if "build-time seed" in l or "relays held" in l][:5]
        T = _Topo(net, meta, zd)
        rec["uc_offline"] = [str(net.gen.at[g, "name"]) for g in net.gen.index
                             if not bool(net.gen.at[g, "in_service"])]
        # the plant as the runner leaves it, re-solved with the stress settings
        base = _probe(net, cfg, T, "loops")
        rec["base_ok"] = base["ok"]
        if not base["ok"]:
            raise _Done()
        rec.update(_voltages(net, T))
        rec.update(_loading(net, T))
        rec.update(_area(net, T))
        rec.update(_sts(net, T))
        rec["ts_ibg_init"] = [[str(net.sgen.at[s, "name"]),
                               float(net.res_sgen.at[s, "p_mw"]),
                               float(net.res_sgen.at[s, "q_mvar"]),
                               *_capability(net.sgen.at[s, "sn_mva"],
                                            net.sgen.at[s, "op_diagram"],
                                            float(net.res_sgen.at[s, "p_mw"]))]
                              for s in T.ts_ibg]
        rec["n1_mode"] = mode
        rec["n1"] = _n1(net, cfg, T, mode)
        rec["ladder"] = {str(z): _ladder(net, cfg, T, z) for z in LADDER_ZONES}
    except _Done:
        pass
    except Exception as exc:                      # noqa: BLE001
        rec.update(screen_error=f"{type(exc).__name__}: {exc}",
                   tb=traceback.format_exc()[-1500:])
    rec["wall_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(rec, default=float))
    return {"op": op_id, "cand": key, "init_ok": rec.get("init_ok"),
            "wall_s": rec["wall_s"]}


class _Done(Exception):
    pass


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
def _pool(jobs: List[Tuple], n: int, log) -> None:
    pending = list(jobs)
    for attempt in range(4):
        if not pending:
            return
        done = set()
        try:
            with ProcessPoolExecutor(max_workers=n) as ex:
                futs = {ex.submit(snapshot, *j): i for i, j in enumerate(pending)}
                for k, f in enumerate(as_completed(futs), 1):
                    i = futs[f]
                    try:
                        r = f.result()
                        log(f"{k}/{len(pending)} {r}")
                    except Exception as exc:          # noqa: BLE001
                        log(f"{k}/{len(pending)} {pending[i][0]} {pending[i][3]} "
                            f"worker error {exc!r}")
                    done.add(i)
        except BrokenProcessPool as exc:
            log(f"pool broken ({exc!r}); restarting with the missing tasks")
        pending = [j for i, j in enumerate(pending) if i not in done]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--candidates", default="B,R,N,NR")
    ap.add_argument("--north-mw", type=float, default=None)
    ap.add_argument("--limit-mc", type=int, default=None)
    ap.add_argument("--ops", default=None,
                    help="comma list of op ids to restrict to (e.g. W1,XWL1)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--mode", choices=("frozen", "loops"), default="frozen",
                    help="solve mode of the N-1 probes (module docstring)")
    ap.add_argument("--ops-file", default=None,
                    help="ops_list.json of an earlier run (same operating points "
                         "on a host without the MC prescreen)")
    add_run_arguments(ap)
    args = ap.parse_args()

    ops = (json.loads(Path(args.ops_file).read_text()) if args.ops_file
           else operating_points(args.limit_mc))
    if args.ops:
        keep = set(args.ops.split(","))
        ops = [o for o in ops if o[0] in keep]
    cands = [c for c in args.candidates.split(",") if c]
    run = start_thesis_run(
        __file__, target=TARGET, feeds=(),
        parameters={"candidates": {c: [CANDIDATES[c], OVERRIDES.get(c, {})]
                                   for c in cands},
                    "north_mw": args.north_mw or 1500.0,
                    "n_ops": len(ops), "mc_prescreen": str(MC_RUN),
                    "horizon_min": HORIZON_MIN, "ladder_zones": LADDER_ZONES,
                    "ladder_mvar": LADDER_MVAR, "v_low": V_LOW,
                    "n1_mode": args.mode,
                    "variant": "L2 (LOC), runner initialisation only"},
        time_series=f"{len(ops)} snapshots: MC prescreen starts, W1-W6, OP-H/M, "
                    f"extreme wind/load hours",
        note=args.note, tag=args.tag, resume=args.resume)
    out = run.path("ops")
    out.mkdir(parents=True, exist_ok=True)
    run.path("_scratch").mkdir(exist_ok=True)
    run.path("ops_list.json").write_text(json.dumps(ops, indent=1))
    logf = open(run.path("screen.log"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{_dt.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    jobs = []
    for op_id, start, off in ops:
        for c in cands:
            key = c if args.north_mw is None else f"{c}{int(args.north_mw)}"
            suffix = "" if args.mode == "frozen" else "__" + args.mode
            if (out / f"{op_id}__{key}{suffix}.json").exists():
                continue
            jobs.append((op_id, start, off, c, args.north_mw, str(out), args.mode))
    log(f"run {run.root}: {len(jobs)} snapshots to compute "
        f"({len(ops)} OPs x {cands})")
    _pool(jobs, args.jobs, log)
    run.describe_output("ops/", "one JSON per (operating point, candidate)")
    run.describe_output("ops_list.json", "the operating points")
    run.describe_output("screen.log", "driver log")
    run.finish()
    log("done")


if __name__ == "__main__":
    main()
