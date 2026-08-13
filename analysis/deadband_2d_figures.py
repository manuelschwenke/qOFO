#!/usr/bin/env python3
r"""Figures for the 2D dead-band study (minimal style, no titles).

Reads ``results/deadband_2d/deadband_2d_metrics.csv`` (written by
``analysis.deadband_2d``) and draws, per step amplitude:

    fig_2d_ifq_<amp>      interface-Q error over the delta_TS x delta_DS grid
    fig_2d_dv_<amp>       max |dV| at the TS and at the DS parks, side by side
    fig_2d_pareto_<amp>   interface Q against max |dV|, Pareto front marked

The heat maps carry printed cell values: a 4x4 grid is small enough that the
numbers are the point, and a colour bar alone would make the reader estimate
them.  Titles are omitted deliberately -- captions belong in the document.

Usage::

    python -m analysis.deadband_2d_figures

Author: Manuel Schwenke / Claude Code (2026-08-02)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RES = PROJECT_ROOT / "results" / "deadband_2d"
FIGS = RES / "figures"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (5.2, 3.4),
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200)
    print(f"  wrote {name}.pdf / .png")


def _grid(df: pd.DataFrame, col: str):
    """Pivot to a delta_TS x delta_DS matrix."""
    ts = sorted(df["delta_ts_pu"].unique())
    ds = sorted(df["delta_ds_pu"].unique())
    m = np.full((len(ts), len(ds)), np.nan)
    for i, a in enumerate(ts):
        for j, b in enumerate(ds):
            sel = df[(df["delta_ts_pu"] == a) & (df["delta_ds_pu"] == b)]
            if len(sel):
                m[i, j] = float(sel[col].iloc[0])
    return ts, ds, m


def _heat(plt, ax, df, col, fmt):
    ts, ds, m = _grid(df, col)
    im = ax.imshow(m, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ds)), [f"{d:g}" for d in ds])
    ax.set_yticks(range(len(ts)), [f"{d:g}" for d in ts])
    ax.set_xlabel(r"$\delta_{\mathrm{DS}}$ [p.u.]")
    ax.set_ylabel(r"$\delta_{\mathrm{TS}}$ [p.u.]")
    ax.grid(False)
    finite = m[np.isfinite(m)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0.0
    for i in range(len(ts)):
        for j in range(len(ds)):
            if not np.isfinite(m[i, j]):
                continue
            ax.text(j, i, format(m[i, j], fmt), ha="center", va="center",
                    fontsize=7,
                    color="white" if m[i, j] < mid else "black")
    return im


def fig_heatmaps(plt, df: pd.DataFrame, amp: float) -> None:
    g = df[df["step_mw"] == amp]
    if g.empty:
        return
    fig, ax = plt.subplots()
    im = _heat(plt, ax, g, "ifq_post_mvar", ".2f")
    fig.colorbar(im, ax=ax, label="interface-Q error [Mvar]")
    save(fig, f"fig_2d_ifq_{int(amp)}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, col, lab in ((axes[0], "dv_max_ts_pu", "TS parks"),
                         (axes[1], "dv_max_ds_pu", "DS parks")):
        im = _heat(plt, ax, g, col, ".4f")
        ax.set_title(lab, fontsize=8)      # panel label, not a figure title
        fig.colorbar(im, ax=ax)
    save(fig, f"fig_2d_dv_{int(amp)}")
    plt.close(fig)


def fig_pareto(plt, df: pd.DataFrame, amp: float,
               cost="ifq_post_mvar", design="dv_max_pu") -> None:
    g = df[(df["step_mw"] == amp) & np.isfinite(df[cost])
           & np.isfinite(df[design])]
    if g.empty:
        return
    front = []
    for _, r in g.iterrows():
        if not any((o[cost] <= r[cost] and o[design] <= r[design])
                   and (o[cost] < r[cost] or o[design] < r[design])
                   for _, o in g.iterrows()):
            front.append(r)
    fig, ax = plt.subplots()
    ax.plot(g[design], g[cost], "o", color="0.7", label="all cells")
    if front:
        f = pd.DataFrame(front).sort_values(design)
        ax.plot(f[design], f[cost], "s-", color="#c0392b",
                label="Pareto front")
        for _, r in f.iterrows():
            ax.annotate(f"({r['delta_ts_pu']:g}, {r['delta_ds_pu']:g})",
                        (r[design], r[cost]), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"max $|\Delta V|$ from twin [p.u.]")
    ax.set_ylabel("interface-Q error [Mvar]")
    ax.legend()
    save(fig, f"fig_2d_pareto_{int(amp)}")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path,
                    default=RES / "deadband_2d_metrics.csv")
    args = ap.parse_args(argv)
    if not args.metrics.exists():
        print(f"missing {args.metrics} -- run analysis.deadband_2d first")
        return 1
    df = pd.read_csv(args.metrics)
    if df.empty:
        print("metrics file is empty")
        return 1
    plt = style()
    for amp in sorted(df["step_mw"].unique()):
        fig_heatmaps(plt, df, amp)
        fig_pareto(plt, df, amp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
