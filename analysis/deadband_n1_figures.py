#!/usr/bin/env python3
r"""Figures for the N-1 dead-band study (minimal style, no titles).

The design intent is that the Q(V) layer rejects EVENTS while leaving
PROFILE-driven voltage shifts to the OFO.  The figures are therefore built
around the separation of two distributions rather than around an optimum:

    fig_n1_detector    profile-drift CDF per level with the dead-band ladder
                       and the N-1 excursions marked -- the design-rule figure:
                       an admissible delta lies right of the drift distribution
                       and left of the event it must catch
    fig_n1_admissible  false-activation rate and compensation effectiveness
                       against delta, per level, with the admissible band shaded
    fig_n1_rejection   post-trip peak and residual |dV| against delta, per gen

Reads ``results/deadband_n1/deadband_n1_metrics.csv`` (written by
``analysis.deadband_n1``).  Titles are omitted deliberately -- captions belong
in the document.

Usage::

    python -m analysis.deadband_n1_figures

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

RES = PROJECT_ROOT / "results" / "deadband_n1"
FIGS = RES / "figures"

LEVELS = (("ts", "TS parks", "#2171b5"), ("ds", "DS parks", "#08856b"))


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


def wtag(window: str) -> str:
    """Filename-safe window tag, e.g. 2016-01-05T08:00 -> 20160105_0800.

    Every figure is per-window: drift and event severity are both properties of
    the operating point, so overlaying windows in one panel would compare curves
    the same plant never produced together.
    """
    return (str(window).replace("-", "").replace(":", "")
            .replace("T", "_"))


def fig_detector(plt, df: pd.DataFrame, window: str) -> None:
    """Where the dead band sits between profile drift and event excursion."""
    df = df[df["window"] == window]
    if df.empty:
        return
    fig, ax = plt.subplots()
    deltas = sorted(df["delta_pu"].unique())
    for lvl, lab, col in LEVELS:
        # the open-loop drift distribution is summarised per row; use the
        # false-activation rate itself, which IS the exceedance curve
        sub = df.drop_duplicates("delta_pu").sort_values("delta_pu")
        y = sub.get(f"faopen_{lvl}")
        if y is None or not np.isfinite(y).any():
            continue
        ax.plot(sub["delta_pu"], y, "o-", color=col, label=f"{lab}: P(drift > δ)")
    for gen, mark in zip(sorted(df["gen"].unique()), ("--", ":")):
        g = df[df["gen"] == gen]
        for lvl, lab, col in LEVELS:
            peak = g.get(f"peak_dv_{lvl}_pu")
            if peak is None or not np.isfinite(peak).any():
                continue
            ax.axvline(float(np.nanmax(peak)), ls=mark, lw=0.9, color=col,
                       alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"dead band $\delta$ [p.u.]")
    ax.set_ylabel("false-activation rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    save(fig, f"fig_n1_detector_{wtag(window)}")
    plt.close(fig)


def fig_admissible(plt, df: pd.DataFrame, window: str) -> None:
    """False activation against compensation effectiveness, per level."""
    df = df[df["window"] == window]
    for gen in sorted(df["gen"].unique()):
        g = df[df["gen"] == gen].sort_values("delta_pu")
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        for lvl, lab, col in LEVELS:
            fa = g.get(f"faopen_{lvl}")
            ce = g.get(f"comp_eff_{lvl}")
            if fa is not None and np.isfinite(fa).any():
                ax.plot(g["delta_pu"], fa, "o-", color=col,
                        label=f"{lab}: false activation")
            if ce is not None and np.isfinite(ce).any():
                ax2.plot(g["delta_pu"], ce, "s--", color=col, alpha=0.65,
                         label=f"{lab}: compensation")
        ax.set_xscale("log")
        ax.set_xlabel(r"dead band $\delta$ [p.u.]")
        ax.set_ylabel("false-activation rate")
        ax2.set_ylabel("compensation effectiveness")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center left")
        save(fig, f"fig_n1_admissible_gen{int(gen)}_{wtag(window)}")
        plt.close(fig)


def fig_rejection(plt, df: pd.DataFrame, window: str) -> None:
    """Post-trip peak and residual deviation against delta."""
    df = df[df["window"] == window]
    for gen in sorted(df["gen"].unique()):
        g = df[df["gen"] == gen].sort_values("delta_pu")
        fig, ax = plt.subplots()
        for lvl, lab, col in LEVELS:
            pk = g.get(f"peak_dv_{lvl}_pu")
            rs = g.get(f"resid_dv_{lvl}_pu")
            if pk is not None and np.isfinite(pk).any():
                ax.plot(g["delta_pu"], pk, "o-", color=col, label=f"{lab}: peak")
            if rs is not None and np.isfinite(rs).any():
                ax.plot(g["delta_pu"], rs, "s--", color=col, alpha=0.65,
                        label=f"{lab}: at next TSO dispatch")
        ax.set_xscale("log")
        ax.set_xlabel(r"dead band $\delta$ [p.u.]")
        ax.set_ylabel(r"$|\Delta V|$ from twin [p.u.]")
        ax.legend()
        save(fig, f"fig_n1_rejection_gen{int(gen)}_{wtag(window)}")
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path,
                    default=RES / "deadband_n1_metrics.csv")
    args = ap.parse_args(argv)
    if not args.metrics.exists():
        print(f"missing {args.metrics} -- run analysis.deadband_n1 first")
        return 1
    df = pd.read_csv(args.metrics)
    if df.empty:
        print("metrics file is empty")
        return 1
    if "window" not in df.columns:
        print("metrics file predates per-window keying -- re-run "
              "analysis.deadband_n1")
        return 1
    plt = style()
    for window in sorted(df["window"].unique()):
        print(f"  [{window}]")
        fig_detector(plt, df, window)
        fig_admissible(plt, df, window)
        fig_rejection(plt, df, window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
