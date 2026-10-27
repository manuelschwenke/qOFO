"""
experiments/ch_11_case2_multisystem/ref/figures_ref.py
======================================================

Figures of the Case-2 reference run (thesis ``ch:case3:nominal``) from a
``run_ref`` run folder:

* ``fig:case3:vtrack``  -- voltage-tracking error e_v,a(t) of each TS area,
  every variant, events marked;
* ``fig:case3:qtrack``  -- interface flow of each STS (sum over its couplers)
  against the dispatched reference, O3 and S2, with O2's flow for comparison;
* ``fig:case3:tieflow`` -- reactive flow on the three inter-area corridors,
  every variant (measured, not controlled).

Each figure is written as PDF and PNG to ``<run>/figures/`` and its data as CSV
to ``<run>/figures/data/`` (the files to copy into the thesis
``graphics/Ch11/data/<date>_*``), and listed in the run's README.

Palette: the validated four hues of the earlier Part-III figures
(``ladder/figures.PALETTE``): S1 red, O2 teal, O3 dark blue; S2 takes amber,
the slot of O1, which Case 2 does not have.  L2 and M stay grey/black and are
separated by dash pattern -- the floor and the bound are context, not
candidates.

Run:
    python -m experiments.ch_11_case2_multisystem.ref.figures_ref <run dir>

Author: Manuel Schwenke / Claude Code (2026-10-27)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ch_11_case2_multisystem.ref import scenarios as sc

PALETTE = {"L2": "#767676", "S1": "#B90F22", "S2": "#D28700",
           "O2": "#008877", "O3": "#004E8A", "M": "#1a1a1a"}
STYLE = {"L2": dict(ls=(0, (1.5, 1.5)), lw=1.0), "S1": dict(ls="-", lw=1.2),
         "S2": dict(ls="-", lw=1.2), "O2": dict(ls="-", lw=1.2),
         "O3": dict(ls="-", lw=1.6), "M": dict(ls=(0, (4, 1.5)), lw=1.0)}
ORDER = ["L2", "S1", "S2", "O2", "O3", "M"]


def _load(run: Path) -> Dict[str, list]:
    logs = {}
    for name in ORDER:
        p = run / name / "log.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                log = pickle.load(fh)
            if log:
                logs[name] = log
    return logs


def _event_times(run: Path) -> List[tuple]:
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    evs = sc.events(man["parameters"]["scenario"])
    return [(e.effective_time_s / 60.0, f"{e.element_type} {e.action}")
            for e in sorted(evs, key=lambda e: e.effective_time_s)]


def _mark(ax, events) -> None:
    for k, (t, _what) in enumerate(events, start=1):
        ax.axvline(t, color="#9a9a9a", lw=0.7, ls=":", zorder=0)
        ax.text(t, 1.0, str(k), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="#555555")


def _save(fig, out: Path, stem: str) -> List[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("pdf", "png"):
        f = out / f"{stem}.{ext}"
        fig.savefig(f, dpi=200, bbox_inches="tight")
        files.append(f.name)
    plt.close(fig)
    return files


def fig_vtrack(logs, events, out: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True)
    data = {}
    for k, z in enumerate((1, 2, 3)):
        ax = axes[k]
        for name in ORDER:
            if name not in logs:
                continue
            t = np.array([r.time_s / 60.0 for r in logs[name]])
            e = np.array([r.zone_v_rms_err_pu.get(z, np.nan) for r in logs[name]]) * 1e3
            ax.plot(t, e, color=PALETTE[name], label=name, **STYLE[name])
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"e_v_area{z}_{name}_mpu"] = pd.Series(e)
        _mark(ax, events)
        ax.set_ylabel(f"$e_{{v,{z}}}$ [mpu]")
        ax.grid(alpha=0.25, lw=0.5)
    axes[-1].set_xlabel("time [min]")
    axes[0].legend(ncol=6, fontsize=8, loc="lower center",
                   bbox_to_anchor=(0.5, 1.12), frameon=False)
    _save(fig, out, "case3_vtrack")
    return pd.DataFrame(data)


def _sts_series(log, kind: str) -> Dict[str, np.ndarray]:
    """Per STS, the sum over its couplers of the actual flow or the set-point."""
    field = "dso_trafo_q_actual_mvar" if kind == "act" else "dso_trafo_q_set_mvar"
    groups: Dict[str, List[float]] = {}
    for r in log:
        sums: Dict[str, float] = {}
        for k, q in getattr(r, field).items():
            g = r.dso_trafo_group.get(k, k.split("|")[0])
            if q is not None and np.isfinite(q):
                sums[g] = sums.get(g, 0.0) + float(q)
        for g in set(groups) | set(sums):
            groups.setdefault(g, []).append(sums.get(g, np.nan))
    return {g: np.array(v) for g, v in groups.items()}


def fig_qtrack(logs, events, out: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    data = {}
    sts = sorted(_sts_series(logs.get("O3") or next(iter(logs.values())), "act"))
    for ax, g in zip(axes.flat, sts):
        for name, ls in (("O3", "-"), ("S2", "--")):
            if name not in logs:
                continue
            t = np.array([r.time_s / 60.0 for r in logs[name]])
            act, ref = _sts_series(logs[name], "act"), _sts_series(logs[name], "set")
            ax.plot(t, act.get(g, np.full(t.shape, np.nan)), color=PALETTE[name],
                    ls=ls, lw=1.2, label=f"{name} flow")
            if g in ref:
                ax.step(t, ref[g], where="post", color=PALETTE[name], ls=ls,
                        lw=0.8, alpha=0.6, label=f"{name} reference")
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"{g}_{name}_act_mvar"] = pd.Series(act.get(g, np.full(t.shape, np.nan)))
            if g in ref:
                data[f"{g}_{name}_ref_mvar"] = pd.Series(ref[g])
        if "O2" in logs:
            t = np.array([r.time_s / 60.0 for r in logs["O2"]])
            act = _sts_series(logs["O2"], "act")
            ax.plot(t, act.get(g, np.full(t.shape, np.nan)), color=PALETTE["O2"],
                    lw=0.9, alpha=0.8, label="O2 flow (no access)")
            data["t_min_O2"] = pd.Series(t)
            data[f"{g}_O2_act_mvar"] = pd.Series(act.get(g, np.full(t.shape, np.nan)))
        _mark(ax, events)
        ax.text(0.02, 0.95, g, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        ax.set_ylabel("$Q$ at HV port [Mvar]")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("time [min]")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    _save(fig, out, "case3_qtrack")
    return pd.DataFrame(data)


def fig_tieflow(logs, events, out: Path) -> pd.DataFrame:
    pairs = sorted({k for log in logs.values() for r in log[:1]
                    for k in r.zone_tie_q_mvar})
    fig, axes = plt.subplots(len(pairs), 1, figsize=(7.0, 2.0 * len(pairs)),
                             sharex=True, squeeze=False)
    data = {}
    for ax, pair in zip(axes[:, 0], pairs):
        for name in ORDER:
            if name not in logs:
                continue
            t = np.array([r.time_s / 60.0 for r in logs[name]])
            q = np.array([r.zone_tie_q_mvar.get(pair, np.nan) for r in logs[name]])
            ax.plot(t, q, color=PALETTE[name], label=name, **STYLE[name])
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"tie_{pair[0]}_{pair[1]}_{name}_mvar"] = pd.Series(q)
        _mark(ax, events)
        ax.set_ylabel(f"$Q_{{{pair[0]}\\to{pair[1]}}}$ [Mvar]")
        ax.grid(alpha=0.25, lw=0.5)
    axes[-1, 0].set_xlabel("time [min]")
    axes[0, 0].legend(ncol=6, fontsize=8, loc="lower center",
                      bbox_to_anchor=(0.5, 1.12), frameon=False)
    _save(fig, out, "case3_tieflow")
    return pd.DataFrame(data)


def make_figures(run: Path) -> List[str]:
    logs = _load(run)
    if not logs:
        raise SystemExit(f"no variant log in {run}")
    events = _event_times(run)
    out = run / "figures"
    written = []
    for stem, fn in (("case3_vtrack", fig_vtrack), ("case3_qtrack", fig_qtrack),
                     ("case3_tieflow", fig_tieflow)):
        df = fn(logs, events, out)
        (out / "data").mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "data" / f"{stem}.csv", index=False)
        written.append(stem)
    try:
        from experiments.thesis_runs import ThesisRun
        tr = ThesisRun.load(run)
        tr.describe_output("figures/", "case3_vtrack/qtrack/tieflow (.pdf/.png) "
                           "and figures/data/*.csv")
        tr.write()
    except Exception as exc:           # a copied folder without a manifest
        print(f"(README not updated: {exc})")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run_ref run folder")
    a = ap.parse_args()
    print("written:", make_figures(Path(a.run)))


if __name__ == "__main__":
    main()
