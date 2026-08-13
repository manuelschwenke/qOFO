#!/usr/bin/env python3
r"""Cross-window figures for the N-1 dead-band study (minimal, no titles).

The per-window figures (``analysis.deadband_n1_figures``) show one operating
point each.  The thesis argument is a REPLICATION argument -- a dead band is
only defensible if its properties hold across operating points -- so the
figures that carry it put every window on one axis:

    fig_n1x_false_activation   the headline: false-activation rate vs delta,
                               one line per window, TS and DS panels.  This is
                               what rules out delta = 0.005 (13.3 % of TS
                               windows at the import point) and justifies 0.01.
    fig_n1x_compensation       compensation vs delta, one line per window, one
                               panel per tripped machine -- shows the flat
                               region and where it decays.
    fig_n1x_peak               post-trip peak |dV| vs delta, with each window's
                               no-droop level (delta = 0.5) as a dashed
                               reference.

Windows are labelled by NET INFEED, which is the physically meaningful ordering
(-117 / +409 / +1367 MW), not by date.

Only the ladder common to all windows is drawn: the reference window carries
three extra dead bands (0.001, 0.075, 0.15) that the others never ran, and
plotting them would show phantom gaps in the other two series.

Usage::

    python -m analysis.deadband_n1_figures_multiwindow

Author: Manuel Schwenke / Claude Code (2026-08-04)
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

from analysis.deadband_n1_figures import save, style  # noqa: E402

RES = PROJECT_ROOT / "results" / "deadband_n1"

#: Window -> (net infeed [MW], colour, marker).  Ordered by net infeed.
WINDOWS = {
    "2016-02-22T13:00": (-117, "#c0392b", "o"),
    "2016-01-05T08:00": (+409, "#2171b5", "s"),
    "2016-12-18T14:00": (+1367, "#08856b", "^"),
}

#: The dead band the study recommends; drawn as a guide line.
DELTA_STAR = 0.01

#: Drawn only where every window has a point (see module docstring).
def common_ladder(df: pd.DataFrame) -> np.ndarray:
    sets = [set(df[df["window"] == w]["delta_pu"]) for w in df["window"].unique()]
    return np.array(sorted(set.intersection(*sets))) if sets else np.array([])


def _series(df, window, gen, col, ladder):
    """(delta, value) for one window/gen, restricted to the common ladder."""
    g = df[(df["window"] == window) & (df["delta_pu"].isin(ladder))]
    if gen is not None:
        g = g[g["gen"] == gen]
    g = g.drop_duplicates("delta_pu").sort_values("delta_pu")
    return g["delta_pu"].to_numpy(float), g[col].to_numpy(float)


def _decorate(ax, ylabel):
    ax.axvline(DELTA_STAR, color="0.35", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"dead band $\delta$ [p.u.]")
    ax.set_ylabel(ylabel)


def fig_false_activation(plt, df, ladder) -> None:
    """The headline figure: does the dead band stay silent on profiles?"""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, lvl, lab in ((axes[0], "ts", "TS parks"), (axes[1], "ds", "DS parks")):
        for w, (net, col, mk) in WINDOWS.items():
            if w not in set(df["window"]):
                continue
            x, y = _series(df, w, None, f"faopen_{lvl}", ladder)
            if not np.isfinite(y).any():
                continue
            ax.plot(x, y, mk + "-", color=col, label=f"{net:+d} MW")
        _decorate(ax, "false-activation rate" if lvl == "ts" else "")
        ax.set_ylim(-0.03, 1.03)
        ax.text(0.03, 0.94, lab, transform=ax.transAxes, fontsize=8,
                va="top")
    axes[0].legend(title="net infeed")
    save(fig, "fig_n1x_false_activation")
    plt.close(fig)


def fig_compensation(plt, df, ladder) -> None:
    gens = sorted(df["gen"].unique())
    fig, axes = plt.subplots(1, len(gens), figsize=(3.9 * len(gens), 3.2),
                             sharey=True, squeeze=False)
    for ax, gen in zip(axes[0], gens):
        for w, (net, col, mk) in WINDOWS.items():
            if w not in set(df["window"]):
                continue
            x, y = _series(df, w, gen, "comp_eff_ts", ladder)
            if not np.isfinite(y).any():
                continue
            m = np.isfinite(y)
            ax.plot(x[m], y[m], mk + "-", color=col, label=f"{net:+d} MW")
        _decorate(ax, "compensation effectiveness"
                  if gen == gens[0] else "")
        ax.set_ylim(-0.05, 0.85)
        ax.text(0.03, 0.06, f"trip gen {int(gen)}", transform=ax.transAxes,
                fontsize=8)
    axes[0][0].legend(title="net infeed")
    save(fig, "fig_n1x_compensation")
    plt.close(fig)


def fig_peak(plt, df, ladder) -> None:
    gens = sorted(df["gen"].unique())
    fig, axes = plt.subplots(1, len(gens), figsize=(3.9 * len(gens), 3.2),
                             sharey=True, squeeze=False)
    for ax, gen in zip(axes[0], gens):
        for w, (net, col, mk) in WINDOWS.items():
            if w not in set(df["window"]):
                continue
            x, y = _series(df, w, gen, "peak_dv_ts_pu", ladder)
            m = np.isfinite(y)
            if not m.any():
                continue
            ax.plot(x[m], y[m], mk + "-", color=col, label=f"{net:+d} MW")
            # the no-droop level: what the slow layer would face unaided
            nod = df[(df["window"] == w) & (df["gen"] == gen)
                     & (df["delta_pu"] == df["delta_pu"].max())]
            if len(nod) and np.isfinite(nod["peak_dv_ts_pu"].iloc[0]):
                ax.axhline(float(nod["peak_dv_ts_pu"].iloc[0]), color=col,
                           lw=0.7, ls="--", alpha=0.55)
        _decorate(ax, r"post-trip peak $|\Delta V|$ [p.u.]"
                  if gen == gens[0] else "")
        ax.text(0.03, 0.94, f"trip gen {int(gen)}", transform=ax.transAxes,
                fontsize=8, va="top")
    axes[0][0].legend(title="net infeed", loc="center left")
    save(fig, "fig_n1x_peak")
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
    if "window" not in df.columns or df.empty:
        print("metrics file empty or predates per-window keying")
        return 1
    unknown = set(df["window"]) - set(WINDOWS)
    if unknown:
        print(f"[warn] windows absent from the label map, not plotted: "
              f"{sorted(unknown)} -- add them to WINDOWS")
    ladder = common_ladder(df)
    print(f"  windows: {sorted(set(df['window']) & set(WINDOWS))}")
    print(f"  common ladder: {[f'{d:g}' for d in ladder]}")
    plt = style()
    fig_false_activation(plt, df, ladder)
    fig_compensation(plt, df, ladder)
    fig_peak(plt, df, ladder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
