"""
experiments/ch_10_case1_svr/figures_case1.py
============================================

Thesis figures of Case 1 (``ch:case_svr``) from a Case-1 run directory.

* ``case_svr_equivalence.pdf``  (fig:case_svr:equivalence)
  (a) pilot voltage of area 1 over the reference run, S1^- and O0 overlaid;
  (b) area-1 RMS voltage error around the tie-line trip at 210 min for the
      integral hierarchy (S1^- = O0), S1 (integral + safeguard), O1 and O2,
      with the
      transmission dispatch instants.
* ``case_svr_pareto.pdf``  (fig:case_svr:pareto)
  pilot error and all-bus error against summed AVR set-point travel, one
  front per scheme.
* ``case_svr_data.csv``  the aggregates behind the second figure.

The thesis text width is 341.4 pt (12.0 cm); figures are drawn at that width
in a Palatino face at 8 pt so they need no scaling in LaTeX.

Run
---
    python -m experiments.ch_10_case1_svr.figures_case1 [--run-dir <dir>] --out <dir>

Without ``--run-dir`` the newest complete run of ``run_case1`` is used.  The
exported files are listed in that run's README.

Author: Manuel Schwenke / Claude Code (2026-10-07)
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.thesis_runs import annotate_run, latest_run  # noqa: E402

#: The producer whose runs this script draws (not imported: it pulls the plant).
_RUN_CASE1 = Path(__file__).with_name("run_case1.py")

CM = 1 / 2.54
WIDTH = 12.0 * CM
PILOT_AREA1 = 25
EVENTS = [30, 120, 180, 210, 300, 330]

#: TU Darmstadt corporate colours, as defined in the thesis preamble
#: (ClassicThesis.tex: tured, tuamber, tuteal, ...; qOFO configs/color_config.py).
#: The variant -> colour mapping is the validated Part III palette of
#: experiments/ch_11_case2_multisystem/ladder/figures.PALETTE (S1 red, O1 amber,
#: O2 teal, O3 dark blue), so a variant has one colour in every chapter.
#: S1^- = O0 is not in that palette and takes TU purple (11c).
TU = {
    "red": "#B90F22",      # 9c  tured
    "amber": "#D28700",    # 7c  tuamber
    "teal": "#008877",     # 3c  tuteal
    "darkblue": "#004E8A", # 1c  tudarkblue
    "purple": "#611C73",   # 11c
}
COLOR = {"S1": TU["red"], "O1": TU["amber"], "O2": TU["teal"],
         "S1m": TU["purple"]}

STYLE = {
    "S1m": dict(color=COLOR["S1m"], lw=1.6, label=r"S1$^{-}$ = O0 (integral)"),
    "S1": dict(color=COLOR["S1"], lw=1.2, label="S1 (integral + safeguard)"),
    "O1": dict(color=COLOR["O1"], lw=1.2, label="O1"),
    "O2": dict(color=COLOR["O2"], lw=1.2, ls="--", label="O2"),
}


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _load(run: Path, name: str):
    p = run / name / "log.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f) or None


def fig_equivalence(run: Path, out: Path) -> None:
    s1m, o0 = _load(run, "S1m"), _load(run, "O0")
    t = np.array([r.time_s for r in s1m]) / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(WIDTH, 9.6 * CM),
                           gridspec_kw=dict(height_ratios=[1.0, 1.15]))
    vp = lambda log: np.array([r.bus_vm_pu[PILOT_AREA1] for r in log])
    for e in EVENTS:
        ax[0].axvline(e, color="0.88", lw=0.6, zorder=0)
    # Same hue for the pair that coincides: S1^- as a light, wide band, O0 as
    # a thin dashed line on top of it.
    ax[0].plot(t, vp(s1m), color=COLOR["S1m"], alpha=0.30, lw=2.6,
               label=r"S1$^{-}$")
    ax[0].plot(t, vp(o0), color=COLOR["S1m"], lw=0.8, ls="--", label="O0")
    ax[0].set_xlim(0, 360)
    ax[0].set_xlabel("time [min]")
    ax[0].set_ylabel("pilot voltage [pu]")
    ax[0].legend(loc="lower right", frameon=False, ncol=2)
    ax[0].set_title("(a) pilot node of area 1, whole reference run", loc="left")

    win_lo, win_hi = 205.0, 225.0
    for k in range(int(win_lo * 60 // 180), int(win_hi * 60 // 180) + 1):
        ax[1].axvline(k * 3.0, color="0.9", lw=0.5, zorder=0)
    ax[1].axvline(210, color="k", lw=0.7)
    for name in ("S1m", "S1", "O1", "O2"):
        log = _load(run, name)
        if log is None:
            continue
        tt = np.array([r.time_s for r in log]) / 60.0
        e = np.array([r.zone_v_rms_err_pu.get(1, np.nan) for r in log]) * 1e3
        m = (tt >= win_lo) & (tt <= win_hi)
        ax[1].plot(tt[m], e[m], **STYLE[name])
    ax[1].set_xlim(win_lo, win_hi)
    ax[1].set_xlabel("time [min]   (thin lines: transmission dispatch instants)")
    ax[1].set_ylabel("RMS voltage error [mpu]")
    # Below the axes: inside, the legend covers the S1 decay (2026-10-07).
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                 frameon=False, handlelength=1.8, columnspacing=1.0)
    ax[1].set_title("(b) area 1 after the trip of tie line 17–27 at 210 min", loc="left")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out / "case_svr_equivalence.pdf")
    plt.close(fig)


def fig_pareto(run: Path, out: Path) -> None:
    m = pd.read_csv(run / "metrics.csv")
    agg = m.groupby("variant").agg(
        e_v=("e_v", lambda x: np.sqrt(np.mean(np.square(x))) * 1e3),
        e_vp=("e_vp", lambda x: np.sqrt(np.mean(np.square(x))) * 1e3),
        avr=("avr_trav", "sum"))
    agg.to_csv(out / "case_svr_data.csv", float_format="%.6g")
    series = [
        ("S1", ["S1_T112", "S1_T128", "S1", "S1_T600"], COLOR["S1"], "o", "S1"),
        ("O1", ["O1_gw0p01", "O1_gw0p1", "O1"], COLOR["O1"], "s", "O1"),
        ("O2", ["O2_gw0p01", "O2_gw0p1", "O2"], COLOR["O2"], "^", "O2"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 6.2 * CM))
    for j, (col, lab, title) in enumerate((
            ("e_vp", "pilot error [mpu]", "(a) pilot nodes"),
            ("e_v", "error over $\\mathcal{R}_a$ [mpu]", "(b) whole reference set"))):
        for name, vs, c, mk, deployed in series:
            vs = [v for v in vs if v in agg.index]
            sub = agg.loc[vs]
            ax[j].plot(sub["avr"], sub[col], color=c, marker=mk, ms=3.5, lw=0.9,
                       label=name)
            if deployed in agg.index:
                ax[j].plot(agg.loc[deployed, "avr"], agg.loc[deployed, col],
                           marker="o", ms=8, mfc="none", mec=c, mew=0.8, ls="none")
        if "S1m" in agg.index:
            ax[j].plot(agg.loc["S1m", "avr"], agg.loc["S1m", col], marker="x",
                       color=COLOR["S1m"], ms=5, ls="none", label=r"S1$^{-}$ = O0")
        ax[j].set_xscale("log")
        ax[j].set_xlabel("AVR set-point travel [pu]")
        ax[j].set_ylabel(lab)
        ax[j].set_title(title, loc="left")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2,
               handlelength=1.6)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.09, 1, 1))
    fig.savefig(out / "case_svr_pareto.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="a run_case1 run folder (default: the newest complete one)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = (Path(args.run_dir) if args.run_dir
           else latest_run(_RUN_CASE1, producer="run_case1"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run}")
    _setup()
    fig_equivalence(run, out)
    fig_pareto(run, out)
    print("written:", sorted(os.listdir(out)))
    annotate_run(run, by=__file__, exports=[
        (out / name).resolve() for name in
        ("case_svr_equivalence.pdf", "case_svr_pareto.pdf", "case_svr_data.csv")
        if (out / name).exists()])


if __name__ == "__main__":
    main()
