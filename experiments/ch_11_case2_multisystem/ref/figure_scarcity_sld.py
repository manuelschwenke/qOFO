"""
experiments/ch_11_case2_multisystem/ref/figure_scarcity_sld.py
==============================================================

Reactive-reserve map of a Case-2 reference run on the single-line diagram of
``fig:ieee39_wind_replace`` (thesis Fig 7.2), figure stem
``case3_scarcity_sld``.  Three panels:

* (a) the start of the period (first record of L2): the state the unit
  commitment creates, before any dispatch;
* (b) O2 (OFO, no STS access) at the instant t*;
* (c) O3 (OFO+, STS access) at the same instant.

**Rule for t*** (declared before drawing, 2026-11-02): the first record at
which the area-2 own upward reserve under O2 reaches its minimum over the run.

**Encoding.**  One ring per reactive source: ring area proportional to its
reactive capability Q_max - Q_min; the dark arc, clockwise from 12 o'clock, is
the upward reserve Q_max - Q as a share of the ring.

* Synchronous machines: Q, Q_min, Q_max as recorded.  A machine out of service
  at the first record is drawn dashed (decommitted by the merit order); one
  that goes out later is drawn crossed (tripped by an event).  G1 (the network
  equivalent, slack) has no ring and is not counted.
* TS wind parks: VDE-AR-N 4120 v2 box at the recorded P, S_n = P_n of the
  replaced machine (``network/ieee39/scenarios/wind_replace.py``).  The sizes
  are checked against the recorded normalised reserve ``tso_der_q_reserve``.
* STS: the DER inside the STS, sum of their Q, Q_min, Q_max -- the reactive
  resource of the STS, not the interface capability (the coupler and the STS
  voltage band transfer less).  The interface injection and, for O3, the
  capability interval reported by layer 2 are written to the CSV.
* Areas: tint by the own upward reserve R+_z (machines without G1, plus wind
  parks), alpha = 0.4 * max(0, 1 - R+_z / 500 Mvar).

**Layout.**  Bus coordinates, line routing, STS couplings and labels are
those of ``graphics/Ch7/ieee39_wind_replace_body.tex`` (thesis) as corrected
on 2026-11-02 to the code's network: the machine at IEEE bus 30 is the
synchronous hydro unit G10, there are four wind parks W1-W4, and W2 (ex-G2,
IEEE bus 31) is coupled at bus 6.

Each output is written as PDF and PNG to ``<run>/figures/`` and its data as
CSV to ``<run>/figures/data/``.  With ``--thesis <latex dir>`` the PDF is also
copied to ``graphics/Ch10/`` and the CSV to
``graphics/Ch10/data/<run date>_case3_scarcity_sld/`` with a README naming the
run.

Run:
    python -m experiments.ch_11_case2_multisystem.ref.figure_scarcity_sld
        <run dir> [--thesis <latex dir>]

Author: Manuel Schwenke / Claude Code (2026-11-02)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from experiments.ch_11_case2_multisystem.ref.figures_ref import CM, _setup

STEM = "case3_scarcity_sld"
WIDTH = 18.5 * CM          # sidewaysfigure, like fig:ieee39_wind_replace

C_RES = "#004E8A"          # upward reserve (TU blue, O3 in the Case-2 figures)
C_TRACK = "#cfcfcf"
C_LINE = "#a0a0a0"
C_TINT = "#CC4C03"         # area tint, the orange of Fig 7.2
R_TINT = 500.0             # Mvar: no tint at or above this own reserve
R0 = 0.80                  # ring radius [diagram units] at 1000 Mvar capability
RING_W = 0.17

# ---------------------------------------------------------------------------
#  Geometry of Fig 7.2 (thesis graphics/Ch7/ieee39_wind_replace_body.tex)
# ---------------------------------------------------------------------------
#: bus -> (x, y, half width of the bar)
BUS = {
    "B02": (1.0, 10.5, .75), "B01": (1.0, 9.0, .75), "B39": (1.0, 7.5, .75),
    "B25": (4.0, 10.5, .6), "B26": (8.0, 10.5, 1.0), "B27": (13.0, 9.2, .6),
    "B28": (12.0, 10.5, .7), "B29": (14.5, 10.5, .8), "B38": (14.5, 12.0, .45),
    "B03": (3.5, 8.0, .75), "B04": (6.0, 8.0, .75), "B14": (8.5, 8.0, 1.0),
    "B05": (2.4, 5.0, 1.0), "B06": (5.1, 5.0, .75), "B07": (5.2, 1.5, .75),
    "B08": (2.8, 1.5, .8), "B09": (1.0, 0.5, .75), "B12": (7.4, 5.5, .85),
    "B11": (6.4, 4.2, .85), "B13": (8.4, 4.2, .5), "B10": (7.4, 1.5, .6),
    "B32": (7.4, 0.3, .45),
    "B15": (11.0, 4.8, .45), "B16": (13.0, 6.0, 1.0), "B17": (16.0, 7.5, .6),
    "B18": (13.0, 7.5, .6), "B19": (11.2, 1.5, .45), "B21": (13.4, 1.5, .45),
    "B22": (13.4, 0.3, .45), "B23": (16.0, 1.5, .75), "B24": (16.0, 6.0, .75),
    "B33": (11.2, 0.3, .45), "B36": (16.0, 0.3, .45),
}
#: \IEEEvhvo{a}{dxa}{b}{dxb}{y}: vertical - horizontal at y - vertical
VHV = [
    ("B39", -.2, "B09", -.2, 4.0), ("B02", .3, "B25", -.4, 11.2),
    ("B25", .4, "B26", -.8, 10.0), ("B26", .2, "B27", 0.0, 9.5),
    ("B26", .6, "B28", -.5, 10.0), ("B28", .5, "B29", -.6, 10.0),
    ("B26", -.4, "B29", -.6, 11.2), ("B02", .3, "B03", -.4, 9.8),
    ("B03", .5, "B04", -.5, 8.4), ("B04", .5, "B14", -.5, 8.4),
    ("B04", -.5, "B05", .4, 7.2), ("B05", .8, "B06", -.6, 5.5),
    ("B06", .3, "B07", .2, 3.0), ("B07", -.5, "B08", .5, 1.2),
    ("B08", -.6, "B09", .5, 1.0), ("B14", .3, "B13", .4, 6.2),
    ("B14", .7, "B15", -.3, 7.6), ("B17", .4, "B27", .35, 8.8),
    ("B17", -.4, "B18", .4, 8.0), ("B17", .4, "B16", -.4, 6.7),
    ("B15", .2, "B16", -.8, 5.4), ("B16", -.4, "B19", .3, 3.5),
    ("B16", .4, "B21", 0.0, 3.0), ("B16", .8, "B24", -.6, 5.5),
    ("B22", .3, "B23", -.5, 0.9), ("B24", .6, "B23", .6, 3.5),
]
#: explicit polylines (straight lines, the 3-18 detour, B12 transformers,
#: machine transformers, slack connection)
POLY = [
    [(.8, 10.5), (.8, 9.0)], [(.8, 9.0), (.8, 7.5)], [(2.2, 5.0), (2.2, 1.5)],
    [(13.1, 1.5), (13.1, .3)], [(5.7, 5.0), (5.7, 4.2)],
    [(5.7, 4.2), (5.7, 1.8), (7.0, 1.8), (7.0, 1.5)],
    [(3.7, 8.0), (3.7, 8.7), (11.1, 8.7), (11.1, 8.0), (12.6, 8.0), (12.6, 7.5)],
    [(6.7, 5.5), (6.7, 4.2)], [(8.1, 5.5), (8.1, 4.2)],
    [(7.4, 1.5), (7.4, .3)], [(11.2, 1.5), (11.2, .3)], [(16.0, 1.5), (16.0, .3)],
    [(14.5, 10.5), (14.5, 12.0)], [(1.0, 7.5), (.4, 7.5), (.4, 7.1)],
]
#: zone separators of Fig 7.2 closed into polygons
ZONE = {
    1: [(-.9, 13.6), (17.7, 13.6), (17.7, 8.4), (11.5, 8.4), (11.5, 9.2),
        (2.4, 9.2), (2.4, 6.3), (-.9, 6.3)],
    2: [(-.9, 6.3), (2.4, 6.3), (2.4, 9.2), (11.5, 9.2), (11.5, 6.0),
        (9.9, 6.0), (9.9, -2.1), (-.9, -2.1)],
    3: [(11.5, 8.4), (17.7, 8.4), (17.7, -2.1), (9.9, -2.1), (9.9, 6.0),
        (11.5, 6.0)],
}
#: pandapower gen index -> (label, symbol position, connection, label side)
#: (index map read 2026-11-02 from the logs' zone_gen_idx and
#: network/ieee39/constants.py GEN_NAMEPLATE)
GEN = {
    0: ("G10", (0.75, 11.75), [(.75, 11.75), (.75, 10.5)], "r"),
    7: ("G9", (14.5, 12.85), [(14.5, 12.85), (14.5, 12.0)], "r"),
    9: ("G1", (0.4, 6.8), None, "l"),
    1: ("G3", (7.4, -0.65), [(7.4, -.65), (7.4, .3)], "l"),
    2: ("G4", (11.2, -0.65), [(11.2, -.65), (11.2, .3)], "l"),
    5: ("G7", (16.0, -0.65), [(16.0, -.65), (16.0, .3)], "r"),
}
ZONE_GEN = {1: [0, 7, 9], 2: [1], 3: [2, 5]}
#: TS wind parks per area, in the order of zone_q_der: (label, S_n [MVA],
#: position, connection, label side).  Labels as in Fig 7.2.
WIND = {
    1: [("W1", 540.0, (4.0, 11.6), [(4.0, 11.6), (4.0, 10.5)], "r")],
    2: [("W2", 500.0, (4.75, 4.15), [(4.75, 4.15), (4.75, 5.0)], "l")],
    3: [("W3", 508.0, (10.75, 2.65), [(10.75, 2.65), (10.75, 1.5)], "l"),
        ("W4", 650.0, (13.1, -0.45), [(13.1, -.45), (13.1, .3)], "r")],
}
#: STS: (label, position, coupling buses, label side)
STS = {
    "DSO_1": ("STS 1", (3.6, 3.0), ("B05", "B08", "B07"), "l"),
    "DSO_2": ("STS 2", (7.4, 6.75), ("B04", "B14", "B06"), "r"),
    "DSO_3": ("STS 3", (7.25, 2.9), ("B11", "B13", "B10"), "l"),
    "DSO_4": ("STS 4", (15.0, 3.4), ("B24", "B21", "B23"), "l"),
}
STS_AREA = {"DSO_1": 2, "DSO_2": 2, "DSO_3": 2, "DSO_4": 3}
XLIM, YLIM = (-0.9, 17.7), (-2.1, 13.6)


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def vde_q_bounds(p_mw: float, s_n: float) -> Tuple[float, float]:
    """VDE-AR-N 4120 v2 box of core/actuator_bounds.compute_der_q_bounds."""
    x = p_mw / s_n
    if x < 0.1:
        return 0.0, 0.0
    if x >= 0.2:
        return -0.33 * s_n, 0.41 * s_n
    a = (x - 0.1) / 0.1
    return -(0.10 + 0.23 * a) * s_n, (0.10 + 0.31 * a) * s_n


def _load(run: Path, var: str) -> list:
    with open(run / var / "log.pkl", "rb") as fh:
        log = pickle.load(fh)
    return [r for r in log if 1 in getattr(r, "zone_q_gen", {})]


def _own_reserve(r, z: int) -> float:
    q = np.asarray(r.zone_q_gen[z], float)
    hi = np.asarray(r.gen_q_max_mvar[z], float)
    keep = (np.asarray(r.zone_gen_in_service[z], bool)
            & ~np.asarray(r.zone_gen_is_slack[z], bool))
    res = float(np.sum((hi - q)[keep]))
    for k, (_lab, sn, *_rest) in enumerate(WIND[z]):
        res += vde_q_bounds(r.zone_tso_der_p_mw[z][k], sn)[1] - r.zone_q_der[z][k]
    return res


def _check(r) -> None:
    """Gen index map and wind-park sizes against the record."""
    for z, idx in ZONE_GEN.items():
        if list(map(int, r.zone_gen_idx[z])) != idx:
            raise ValueError(f"area {z}: gen indices {r.zone_gen_idx[z]} != {idx}")
    for z, parks in WIND.items():
        for k, (lab, sn, *_rest) in enumerate(parks):
            lo, hi = vde_q_bounds(r.zone_tso_der_p_mw[z][k], sn)
            q = r.zone_q_der[z][k]
            obs = r.tso_der_q_reserve[z][k]
            if hi - lo > 1e-9 and np.isfinite(obs):
                pred = min(hi - q, q - lo) / (hi - lo)
                if abs(pred - obs) > 1e-6:
                    raise ValueError(f"{lab}: S_n {sn} gives reserve {pred:.6f}, "
                                     f"recorded {obs:.6f}")


def panel_state(r, out_at_start: Dict[int, bool]) -> Dict:
    """Per-source and per-area values of one record."""
    _check(r)
    rows = []
    for z, idx in ZONE_GEN.items():
        ins = np.asarray(r.zone_gen_in_service[z], bool)
        slack = np.asarray(r.zone_gen_is_slack[z], bool)
        for k, g in enumerate(idx):
            if slack[k]:
                status = "equivalent"
            elif ins[k]:
                status = "on"
            else:
                status = "decommitted" if out_at_start[g] else "tripped"
            q = float(r.zone_q_gen[z][k]) if ins[k] else 0.0
            lo = float(r.gen_q_min_mvar[z][k])
            hi = float(r.gen_q_max_mvar[z][k])
            rows.append(dict(kind="machine", key=g, label=GEN[g][0], area=z,
                             status=status, q=q, q_min=lo, q_max=hi,
                             reserve=hi - q if status == "on" else 0.0))
        for k, (lab, sn, *_rest) in enumerate(WIND[z]):
            p = float(r.zone_tso_der_p_mw[z][k])
            lo, hi = vde_q_bounds(p, sn)
            q = float(r.zone_q_der[z][k])
            rows.append(dict(kind="wind", key=lab, label=lab, area=z, status="on",
                             q=q, q_min=lo, q_max=hi, reserve=hi - q, p=p))
    for net, (lab, *_rest) in STS.items():
        q = float(np.sum(r.dso_der_q_mvar[net]))
        lo = float(np.sum(r.dso_der_q_min_mvar[net]))
        hi = float(np.sum(r.dso_der_q_max_mvar[net]))
        row = dict(kind="sts", key=net, label=lab, area=STS_AREA[net], status="on",
                   q=q, q_min=lo, q_max=hi, reserve=hi - q,
                   iface_inj=-float(r.dso_q_actual_mvar[net]))
        if r.dso_trafo_q_cap_min_mvar:
            cap_lo = [v for k, v in r.dso_trafo_q_cap_min_mvar.items()
                      if r.dso_trafo_group[k] == net]
            cap_hi = [v for k, v in r.dso_trafo_q_cap_max_mvar.items()
                      if r.dso_trafo_group[k] == net]
            row["iface_inj_max_reported"] = -float(np.sum(cap_lo))
            row["iface_inj_min_reported"] = -float(np.sum(cap_hi))
        rows.append(row)
    areas = {z: dict(own_reserve=_own_reserve(r, z), v_min=float(r.zone_v_min[z]))
             for z in (1, 2, 3)}
    return dict(t_min=r.time_s / 60.0, rows=rows, areas=areas)


def build_panels(run: Path) -> Tuple[List[Dict], float]:
    l2, o2, o3 = (_load(run, v) for v in ("L2", "O2", "O3"))
    first = l2[0]
    out_at_start = {}
    for z, idx in ZONE_GEN.items():
        ins = np.asarray(first.zone_gen_in_service[z], bool)
        for k, g in enumerate(idx):
            out_at_start[g] = not ins[k]
    r2 = np.array([_own_reserve(r, 2) for r in o2])
    k = int(np.argmin(r2))                       # first record of the minimum
    t_star = o2[k].time_s
    rec3 = min(o3, key=lambda r: abs(r.time_s - t_star))
    if abs(rec3.time_s - t_star) > 1e-6:
        raise ValueError(f"O3 has no record at t* = {t_star} s")
    panels = []
    for tag, var, rec in (("a", "L2", first), ("b", "O2", o2[k]), ("c", "O3", rec3)):
        st = panel_state(rec, out_at_start)
        st.update(tag=tag, variant=var)
        panels.append(st)
    return panels, t_star / 60.0


# ---------------------------------------------------------------------------
#  Drawing
# ---------------------------------------------------------------------------

def _radius(row) -> float:
    return R0 * np.sqrt(max(row["q_max"] - row["q_min"], 1.0) / 1000.0)


def _ring(ax, xy, row, style: str = "on") -> float:
    r = _radius(row)
    if style == "decommitted":
        ax.add_patch(Circle(xy, r - RING_W / 2, fill=False, ec="#8a8a8a",
                            lw=0.6, ls=(0, (2, 1.5)), zorder=4))
        return r
    ax.add_patch(Wedge(xy, r, 0, 360, width=RING_W, fc=C_TRACK, ec="none", zorder=4))
    if style == "tripped":
        d = 0.75 * r
        ax.plot([xy[0] - d, xy[0] + d], [xy[1] - d, xy[1] + d], color="#555555",
                lw=0.7, zorder=6)
        ax.plot([xy[0] - d, xy[0] + d], [xy[1] + d, xy[1] - d], color="#555555",
                lw=0.7, zorder=6)
        return r
    rng = row["q_max"] - row["q_min"]
    f = float(np.clip(row["reserve"] / rng, 0.0, 1.0)) if rng > 1e-9 else 0.0
    if f > 1e-3:
        ax.add_patch(Wedge(xy, r, 90 - 360 * f, 90, width=RING_W, fc=C_RES,
                           ec="none", zorder=5))
    return r


def _label(ax, xy, r, text, side, **kw):
    dx = r + 0.18
    x = xy[0] + (dx if side == "r" else -dx)
    ax.text(x, xy[1], text, ha="left" if side == "r" else "right", va="center",
            fontsize=6, zorder=7, **kw)


def draw_panel(ax, st: Dict, title: str) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    for z, poly in ZONE.items():
        res = st["areas"][z]["own_reserve"]
        alpha = 0.4 * max(0.0, 1.0 - res / R_TINT)
        ax.add_patch(Polygon(poly, closed=True, fc=C_TINT, alpha=alpha, ec="none",
                             zorder=0))
        ax.add_patch(Polygon(poly, closed=True, fill=False, ec="#8a8a8a", lw=0.5,
                             ls=(0, (0.8, 1.6)), zorder=1))
    kw = dict(color=C_LINE, lw=0.45, zorder=2, solid_capstyle="butt")
    for a, da, b, db, y in VHV:
        xa, ya, _ = BUS[a]
        xb, yb, _ = BUS[b]
        ax.plot([xa + da, xa + da, xb + db, xb + db], [ya, y, y, yb], **kw)
    for pl in POLY:
        ax.plot(*zip(*pl), **kw)
    for name, (x, y, w) in BUS.items():
        ax.plot([x - w, x + w], [y, y], color="black", lw=1.3, zorder=3,
                solid_capstyle="butt")
    rows = {(row["kind"], row["key"]): row for row in st["rows"]}
    for g, (lab, xy, conn, side) in GEN.items():
        row = rows[("machine", g)]
        if conn:
            ax.plot(*zip(*conn), **kw)
        if row["status"] == "equivalent":
            ax.add_patch(Rectangle((xy[0] - .22, xy[1] - .22), .44, .44, fc="white",
                                   ec="black", lw=0.6, zorder=5))
            _label(ax, xy, 0.22, "G1", side, color="#555555")
            continue
        r = _ring(ax, xy, row, row["status"] if row["status"] != "on" else "on")
        ax.add_patch(Circle(xy, 0.13, fc="black" if row["status"] == "on" else "white",
                            ec="black" if row["status"] == "on" else "#8a8a8a",
                            lw=0.5, zorder=6))
        _label(ax, xy, r, lab, side,
               color="black" if row["status"] == "on" else "#777777")
    for z, parks in WIND.items():
        for lab, _sn, xy, conn, side in parks:
            row = rows[("wind", lab)]
            ax.plot(*zip(*conn), **kw)
            r = _ring(ax, xy, row)
            ax.add_patch(Circle(xy, 0.13, fc="white", ec="black", lw=0.6, zorder=6))
            _label(ax, xy, r, lab, side)
    for net, (lab, xy, buses, side) in STS.items():
        row = rows[("sts", net)]
        for b in buses:
            ax.plot([xy[0], BUS[b][0]], [xy[1], BUS[b][1]], color=C_LINE, lw=0.45,
                    ls=(0, (2, 1.4)), zorder=2)
        r = _ring(ax, xy, row)
        ax.add_patch(Rectangle((xy[0] - .17, xy[1] - .17), .34, .34, fc="white",
                               ec="black", lw=0.6, zorder=6))
        _label(ax, xy, r, lab, side)
    for z, (x, y, ha) in {1: (7.0, 12.95, "center"), 2: (-0.7, -1.8, "left"),
                          3: (17.5, -1.8, "right")}.items():
        ax.text(x, y, f"Area {z}", ha=ha, va="center", fontsize=6.5,
                color="#444444", zorder=7)
    a = st["areas"]
    inj = sum(row["iface_inj"] for row in st["rows"]
              if row["kind"] == "sts" and row["area"] == 2)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.text(0.5, -0.02,
            "own reserve $R^+$: "
            + ", ".join(f"A{z} {a[z]['own_reserve']:.0f}" for z in (1, 2, 3))
            + " Mvar\n"
            + f"area 2: $V_\\min$ {a[2]['v_min']:.3f} pu, "
            + f"STS 1–3 inject {inj:.0f} Mvar",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            linespacing=1.3)


def draw_legend(ax) -> None:
    """Glyphs in diagram units, so they match the panels (equal aspect)."""
    w = XLIM[1] - XLIM[0]                  # one panel's width in diagram units
    pos = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    span = 3 * w + 0.4                     # three panels and their gaps
    ax.set_xlim(0, span)
    ax.set_ylim(0, span * (pos.height * fig_h) / (pos.width * fig_w))
    ax.set_aspect("equal")
    ax.axis("off")
    y = 0.5 * ax.get_ylim()[1]
    r = 0.62
    tk = dict(va="center", fontsize=6.5)

    def ring(x, style, f=0.35):
        if style == "dec":
            ax.add_patch(Circle((x, y), r - RING_W / 2, fill=False, ec="#8a8a8a",
                                lw=0.6, ls=(0, (2, 1.5))))
            return
        ax.add_patch(Wedge((x, y), r, 0, 360, width=RING_W, fc=C_TRACK, ec="none"))
        if style == "trip":
            d = 0.75 * r
            ax.plot([x - d, x + d], [y - d, y + d], color="#555555", lw=0.7)
            ax.plot([x - d, x + d], [y + d, y - d], color="#555555", lw=0.7)
            return
        ax.add_patch(Wedge((x, y), r, 90 - 360 * f, 90, width=RING_W, fc=C_RES,
                           ec="none"))

    ring(0.8, "on")
    ax.text(1.7, y, "circle area: capability $Q_\\max-Q_\\min$\n"
            "dark arc: upward reserve $Q_\\max-Q$", **tk)
    ring(15.4, "dec")
    ax.text(16.3, y, "machine decommitted\n(merit order)", **tk)
    ring(24.2, "trip")
    ax.text(25.1, y, "machine tripped\n(event)", **tk)
    ax.add_patch(Rectangle((31.4, y - 0.55), 1.3, 1.1, fc=C_TINT, alpha=0.4,
                           ec="none"))
    ax.text(33.1, y, f"area tint: own reserve $R^+$\n(none at $\\geq${R_TINT:.0f} Mvar)",
            **tk)
    x0, dy = 43.2, 0.62
    ax.add_patch(Circle((x0, y + dy), 0.13, fc="black", ec="black", lw=0.5))
    ax.text(x0 + 0.45, y + dy, "machine", **tk)
    ax.add_patch(Circle((x0, y - dy), 0.13, fc="white", ec="black", lw=0.6))
    ax.text(x0 + 0.45, y - dy, "wind park", **tk)
    x1 = x0 + 4.8
    ax.add_patch(Rectangle((x1 - .17, y + dy - .17), .34, .34, fc="white", ec="black",
                           lw=0.6))
    ax.text(x1 + 0.45, y + dy, "STS (DER inside)", **tk)
    ax.add_patch(Rectangle((x1 - .22, y - dy - .22), .44, .44, fc="white", ec="black",
                           lw=0.6))
    ax.text(x1 + 0.45, y - dy, "G1, equivalent", **tk)


def make_figure(run: Path, thesis: Path | None = None) -> Path:
    _setup()
    panels, t_star = build_panels(run)
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    scen = man["parameters"]["scenario"]
    titles = {
        "a": "(a) start of the period, before any dispatch",
        "b": f"(b) OFO, no STS access, $t^*$ = {t_star:.0f} min",
        "c": f"(c) OFO+, STS access, $t^*$ = {t_star:.0f} min",
    }
    h_panel = (WIDTH / 3) * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
    fig = plt.figure(figsize=(WIDTH, h_panel + 2.3 * CM))
    gs = fig.add_gridspec(2, 3, height_ratios=[h_panel, 1.0 * CM],
                          wspace=0.04, hspace=0.38,
                          left=0.0, right=1.0, top=0.95, bottom=0.02)
    for k, st in enumerate(panels):
        draw_panel(fig.add_subplot(gs[0, k]), st, titles[st["tag"]])
    draw_legend(fig.add_subplot(gs[1, :]))

    out = run / "figures"
    (out / "data").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)

    recs = []
    for st in panels:
        for row in st["rows"]:
            recs.append(dict(panel=st["tag"], variant=st["variant"],
                             t_min=st["t_min"], **row))
        for z, a in st["areas"].items():
            recs.append(dict(panel=st["tag"], variant=st["variant"],
                             t_min=st["t_min"], kind="area", key=z,
                             label=f"Area {z}", area=z,
                             reserve=a["own_reserve"], v_min=a["v_min"]))
    pd.DataFrame(recs).to_csv(out / "data" / f"{STEM}.csv", index=False,
                              float_format="%.4f")
    print(f"scenario {scen}: t* = {t_star:.2f} min; written {out / STEM}.pdf")
    if thesis is not None:
        _to_thesis(run, out, Path(thesis), t_star)
    return out / f"{STEM}.pdf"


def _to_thesis(run: Path, out: Path, thesis: Path, t_star: float) -> None:
    """PDF to graphics/Ch10/, CSV to graphics/Ch10/data/<date>_case3_scarcity_sld/."""
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    date = run.name.split("_")[1]
    gdir = thesis / "graphics" / "Ch10"
    ddir = gdir / "data" / f"{date}_{STEM}"
    ddir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / f"{STEM}.pdf", gdir / f"{STEM}.pdf")
    shutil.copy2(out / "data" / f"{STEM}.csv", ddir / f"{STEM}.csv")
    code = (man.get("invocations") or [{}])[0].get("git", {})
    (ddir / "README.md").write_text(
        f"# Data of the reactive-reserve map (case3_scarcity_sld)\n\n"
        f"- Run: `qOFO/results/ch_11_case2_multisystem/ref/{run.name}/` "
        f"(scenario {man['parameters']['scenario']})\n"
        f"- Run code: qOFO {str(code.get('commit', ''))[:7]} on "
        f"{code.get('branch', '')}; paramset "
        f"{man.get('paramset', {}).get('name', man.get('paramset', ''))}\n"
        f"- Figure: `graphics/Ch10/{STEM}.pdf`, made by "
        f"`experiments/ch_11_case2_multisystem/ref/figure_scarcity_sld.py`\n"
        f"- Panels: (a) L2 first record, (b) O2 and (c) O3 at t* = "
        f"{t_star:.2f} min, the first record of the minimum area-2 own "
        f"upward reserve under O2 (rule declared 2026-11-02).\n"
        f"- File: one row per source (machine, wind park, STS DER) and per "
        f"area and panel; Q in Mvar (generator convention; STS interface "
        f"injection into the TS), voltages in pu.\n",
        encoding="utf-8")
    try:
        from experiments.thesis_runs import ThesisRun
        tr = ThesisRun.load(run)
        tr.record_export(f"graphics/Ch10/{STEM}.pdf", __file__)
        tr.record_export(f"graphics/Ch10/data/{ddir.name}/", __file__)
        tr.write()
    except Exception as exc:           # a copied folder without a manifest
        print(f"(exports not recorded: {exc})")
    print(f"thesis: {STEM}.pdf -> {gdir}, data -> {ddir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run_ref run folder")
    ap.add_argument("--thesis", default=None,
                    help="LaTeX root: copy the PDF and data into graphics/Ch10")
    a = ap.parse_args()
    make_figure(Path(a.run), a.thesis and Path(a.thesis))


if __name__ == "__main__":
    main()
