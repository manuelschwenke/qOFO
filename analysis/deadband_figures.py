#!/usr/bin/env python3
r"""Publication figures for the dead-band study (minimal style, no titles).

Titles are deliberately omitted -- captions belong in the document. Styling is
kept plain so it can be swapped for a house style later: no chart junk, thin
spines, light grid, small legends.

Figures written to ``results/deadband_selection/figures/`` as .pdf and .png:

    fig_threshold      response vs delta, one curve per excitation amplitude,
                       with each amplitude's open-loop |dV| marked. This is the
                       figure the design rule reads off.
    fig_chatter        direction reversals vs delta, one curve per window
    fig_tracking       interface-Q error vs delta, one curve per window
    fig_dsvoltage      DS group voltage deviation vs delta, one curve per window
    fig_tradeoff       DS voltage against interface Q, delta as the path
                       parameter -- the Pareto view

Usage::

    python -m analysis.deadband_figures

Author: Manuel Schwenke / Claude Code (2026-08-02)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RES = PROJECT_ROOT / "results" / "deadband_selection"
FIGS = RES / "figures"

#: Net infeed [MW] -- the physical axis the windows span.
NET: Dict[str, float] = {
    "2016-02-22T13:00": -117,
    "2016-01-05T08:00": 409,
    "2016-01-15T03:00": 805,
    "2016-12-18T14:00": 1367,
    "2016-05-01T16:00": 2200,
}
DEGENERATE = {"2016-07-15T03:00"}


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


def wlabel(w: str) -> str:
    return f"{w[:10]}  {NET.get(w, 0):+d} MW"


def fig_threshold(plt) -> None:
    """Response vs delta at fixed excitation -- the design-rule figure."""
    df = pd.read_csv(RES / "deadband_threshold.csv")
    df = df[df.factor.isin([1.10, 1.25])]
    if df.empty:
        print("  [skip] fig_threshold: no threshold data")
        return
    fig, ax = plt.subplots()
    marks = ["o-", "s--"]
    for i, (f, g) in enumerate(sorted(df.groupby("factor"))):
        g = g.sort_values("delta_pu")
        # open-loop excitation: measured where the droop is inactive (widest
        # dead bands), because a tight band suppresses its own input
        wide = g[g.delta_pu >= 0.015]["dv_settled_pu"]
        dv = float(wide.mean()) if len(wide) else float(g["dv_settled_pu"].max())
        ax.plot(g["delta_pu"], g["traverse_per_park_interval"],
                marks[i % 2], color=f"C{i}",
                label=rf"step $\times${f:g},  $|\Delta V|$ = {dv:.4f} pu")
        ax.axvline(dv, color=f"C{i}", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"dead-zone half-width $\delta$  [pu]")
    ax.set_ylabel("DER traverse after step\n[Mvar / park / interval]")
    ax.legend()
    save(fig, "fig_threshold")
    plt.close(fig)


def fig_per_window(plt, csv: str, col: str, ylabel: str, name: str,
                   logy: bool = False) -> None:
    df = pd.read_csv(RES / csv)
    df = df[~df.window.isin(DEGENERATE)]
    if df.empty:
        print(f"  [skip] {name}")
        return
    fig, ax = plt.subplots()
    marks = ["o-", "s--", "^:", "v-.", "D-"]
    for i, w in enumerate(sorted(df.window.unique(), key=lambda x: NET.get(x, 0))):
        g = df[df.window == w].sort_values("delta_pu")
        ax.plot(g["delta_pu"], g[col], marks[i % len(marks)],
                color=f"C{i}", label=wlabel(w))
    ax.set_xlabel(r"dead-zone half-width $\delta$  [pu]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    save(fig, name)
    plt.close(fig)


def fig_tradeoff(plt) -> None:
    df = pd.read_csv(RES / "deadband_metrics.csv")
    df = df[~df.window.isin(DEGENERATE)]
    fig, ax = plt.subplots()
    marks = ["o-", "s--", "^:", "v-.", "D-"]
    for i, w in enumerate(sorted(df.window.unique(), key=lambda x: NET.get(x, 0))):
        g = df[df.window == w].sort_values("delta_pu")
        ax.plot(g["ifq_mean_abs_err_mvar"], g["ds_v_rms_dev_pu"],
                marks[i % len(marks)], color=f"C{i}", label=wlabel(w))
        for d in (0.0, 0.005, 0.03):
            r = g[np.isclose(g.delta_pu, d)]
            if len(r):
                ax.annotate(rf"$\delta$={d:g}",
                            (float(r["ifq_mean_abs_err_mvar"].iloc[0]),
                             float(r["ds_v_rms_dev_pu"].iloc[0])),
                            fontsize=6, xytext=(3, 3),
                            textcoords="offset points", color=f"C{i}")
    ax.set_xlabel("interface-$Q$ tracking error  [Mvar]")
    ax.set_ylabel("DS group voltage deviation  [pu]")
    ax.legend()
    save(fig, "fig_tradeoff")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    plt = style()
    print("figures ->", FIGS)
    fig_threshold(plt)
    fig_per_window(plt, "deadband_activity.csv",
                   "reversals_per_park_interval",
                   "direction reversals\n[per park / interval]",
                   "fig_chatter")
    fig_per_window(plt, "deadband_metrics.csv", "ifq_mean_abs_err_mvar",
                   "interface-$Q$ tracking error  [Mvar]", "fig_tracking")
    fig_per_window(plt, "deadband_metrics.csv", "ds_v_rms_dev_pu",
                   "DS group voltage deviation  [pu]", "fig_dsvoltage")
    fig_tradeoff(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
