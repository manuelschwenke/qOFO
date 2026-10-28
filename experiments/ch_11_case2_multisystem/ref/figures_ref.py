"""
experiments/ch_11_case2_multisystem/ref/figures_ref.py
======================================================

Figures of the Case-2 reference run (thesis ``ch:case3:nominal``) from a
``run_ref`` run folder, in the thesis style of the Case-1 figures (12 cm text
width, Palatino, 7-8 pt):

* ``case3_vtrack``  -- ``fig:case3:vtrack``: voltage-tracking error e_v,a(t) of
  each TS area and the aggregate over the areas (RMS), every variant;
* ``case3_stsv``    -- ``fig:case3:stsv``: voltage deviation of each STS from
  its set-point (spatial RMS over the STS buses), every variant;
* ``case3_qtrack``  -- ``fig:case3:qtrack``: interface flow of each STS (sum
  over its couplers) against the dispatched reference, O3 and S2, with O2's
  flow for comparison;
* ``case3_event``   -- ``fig:case3:event``: the load step in area 2 at the
  20-s resolution -- area error and the change of the area-2 interface flows
  until the next dispatches (the withdrawal of eq:case3:epost);
* ``case3_summary`` -- ``fig:case3:summary``: time-RMS errors per area and
  aggregate (all-bus and pilot), and the duration curve of the aggregate
  error;
* ``case3_tieflow`` -- ``fig:case3:tieflow``: reactive flow on the three
  inter-area corridors (measured, not controlled).

Each figure is written as PDF and PNG to ``<run>/figures/`` and its data as CSV
to ``<run>/figures/data/``.  With ``--thesis <latex dir>`` the PDFs are also
copied to ``graphics/Ch11/`` and the CSVs to
``graphics/Ch11/data/<run date>_case3_ref/`` with a README naming the run.

Palette: the validated four hues of the earlier Part-III figures: S1 red,
O2 teal, O3 dark blue; S2 takes amber.  L2 and M stay grey/black and are
separated by dash pattern -- the floor and the centralised reference are
context, not candidates.

Run:
    python -m experiments.ch_11_case2_multisystem.ref.figures_ref <run dir>
        [--thesis <latex dir>]

Author: Manuel Schwenke / Claude Code (2026-10-27, thesis style 2026-10-28)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ch_11_case2_multisystem.ref import scenarios as sc

PALETTE = {"L2": "#767676", "S1": "#B90F22", "S2": "#D28700",
           "O2": "#008877", "O3": "#004E8A", "M": "#1a1a1a"}
STYLE = {"L2": dict(ls=(0, (1.2, 1.2)), lw=0.9), "S1": dict(ls="-", lw=0.9),
         "S2": dict(ls="-", lw=0.9), "O2": dict(ls="-", lw=0.9),
         "O3": dict(ls="-", lw=1.3), "M": dict(ls=(0, (3.5, 1.5)), lw=0.9)}
ORDER = ["L2", "S1", "S2", "O2", "O3", "M"]
AREAS = (1, 2, 3)
#: TS area of each STS (``hv.zone`` of the network builder, read 2026-10-28).
STS_AREA = {"DSO_1": 2, "DSO_2": 2, "DSO_3": 2, "DSO_4": 3}

CM = 1 / 2.54
WIDTH = 12.0 * CM


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


def _mark(ax, events, labels: bool = True) -> None:
    for k, (t, _what) in enumerate(events, start=1):
        ax.axvline(t, color="#9a9a9a", lw=0.6, ls=":", zorder=0)
        if labels:
            ax.text(t, 1.0, str(k), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, color="#555555")


def _grid(ax) -> None:
    ax.grid(alpha=0.25, lw=0.4)


def _save(fig, out: Path, stem: str) -> List[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("pdf", "png"):
        f = out / f"{stem}.{ext}"
        fig.savefig(f, dpi=300)
        files.append(f.name)
    plt.close(fig)
    return files


def _t(log) -> np.ndarray:
    return np.array([r.time_s / 60.0 for r in log])


def _area_err(log, z) -> np.ndarray:
    return np.array([r.zone_v_rms_err_pu.get(z, np.nan) for r in log]) * 1e3


def _agg_err(log) -> np.ndarray:
    """Instantaneous aggregate: RMS over the areas (its time-RMS is the
    reported aggregate e_v, the RMS over the areas of the area values)."""
    e = np.vstack([_area_err(log, z) for z in AREAS])
    return np.sqrt(np.mean(e ** 2, axis=0))


def _legend_top(fig, axes0, ncol=6) -> None:
    h, lab = axes0.get_legend_handles_labels()
    fig.legend(h, lab, ncol=ncol, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

def fig_vtrack(logs, events, out: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 13.0 * CM), sharex=True)
    data = {}
    for k, z in enumerate(AREAS + ("all",)):
        ax = axes[k]
        for name in ORDER:
            if name not in logs:
                continue
            t = _t(logs[name])
            e = _agg_err(logs[name]) if z == "all" else _area_err(logs[name], z)
            ax.plot(t, e, color=PALETTE[name], label=name, **STYLE[name])
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"e_v_{'all' if z == 'all' else f'area{z}'}_{name}_mpu"] = pd.Series(e)
        _mark(ax, events, labels=(k == 0))
        ax.set_ylabel(r"$e_{\mathrm{v}}$ [mpu]" if z == "all"
                      else rf"$e_{{\mathrm{{v}},{z}}}$ [mpu]")
        ax.text(0.995, 0.93, "all areas (RMS)" if z == "all" else f"area {z}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        ax.set_ylim(bottom=0)
        _grid(ax)
    axes[-1].set_xlabel("time [min]")
    axes[-1].set_xlim(0, 360)
    axes[-1].set_xticks(range(0, 361, 60))
    fig.tight_layout(h_pad=0.4)
    _legend_top(fig, axes[0])
    _save(fig, out, "case3_vtrack")
    return pd.DataFrame(data)


def fig_stsv(logs, events, out: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 8.0 * CM), sharex=True,
                             sharey=True)
    data = {}
    for ax, g in zip(axes.flat, sorted(STS_AREA)):
        for name in ORDER:
            if name not in logs:
                continue
            t = _t(logs[name])
            f = np.array([r.dso_group_v_rms_dev_pu.get(g, np.nan)
                          for r in logs[name]]) * 1e3
            ax.plot(t, f, color=PALETTE[name], label=name, **STYLE[name])
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"f_v_{g}_{name}_mpu"] = pd.Series(f)
        _mark(ax, events)
        ax.text(0.03, 0.93, f"STS {g.split('_')[-1]} (area {STS_AREA[g]})",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        _grid(ax)
    top = max(np.nanmax(v) for k, v in data.items() if k.startswith("f_v_"))
    axes.flat[0].set_ylim(0, 1.05 * top)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$f_{\mathrm{V,STS}}$ [mpu]")
    for ax in axes[-1]:
        ax.set_xlabel("time [min]")
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 60))
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    _legend_top(fig, axes.flat[0])
    _save(fig, out, "case3_stsv")
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
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 8.4 * CM), sharex=True)
    data = {}
    for ax, g in zip(axes.flat, sorted(STS_AREA)):
        if "O2" in logs:
            t = _t(logs["O2"])
            act = _sts_series(logs["O2"], "act")
            ax.plot(t, act.get(g, np.full(t.shape, np.nan)), color=PALETTE["O2"],
                    lw=0.8, alpha=0.9, label="O2 flow (no access)")
            data["t_min_O2"] = pd.Series(t)
            data[f"{g}_O2_act_mvar"] = pd.Series(act.get(g, np.full(t.shape, np.nan)))
        for name, ls in (("S2", (0, (3, 1.5))), ("O3", "-")):
            if name not in logs:
                continue
            t = _t(logs[name])
            act, ref = _sts_series(logs[name], "act"), _sts_series(logs[name], "set")
            if g in ref:
                ax.step(t, ref[g], where="post", color=PALETTE[name], ls="-",
                        lw=2.4, alpha=0.25, label=f"{name} reference")
            ax.plot(t, act.get(g, np.full(t.shape, np.nan)), color=PALETTE[name],
                    ls=ls, lw=1.0, label=f"{name} flow")
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"{g}_{name}_act_mvar"] = pd.Series(act.get(g, np.full(t.shape, np.nan)))
            if g in ref:
                data[f"{g}_{name}_ref_mvar"] = pd.Series(ref[g])
        _mark(ax, events)
        ax.text(0.03, 0.06, f"STS {g.split('_')[-1]} (area {STS_AREA[g]})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        _grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Q$ at HV port [Mvar]")
    for ax in axes[-1]:
        ax.set_xlabel("time [min]")
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 60))
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    _legend_top(fig, axes.flat[0], ncol=5)
    _save(fig, out, "case3_qtrack")
    return pd.DataFrame(data)


def fig_event(logs, events, out: Path, t_ts_min: float = 3.0) -> pd.DataFrame:
    """The load step (event 2) in area 2: e_v,2 and the change of the summed
    interface flow of the area-2 STSs since the step's pre-event state."""
    t_e = events[1][0]
    lo, hi = t_e - 2.0, t_e + 10.0
    names = [n for n in ("S1", "S2", "O2", "O3", "M") if n in logs]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.4 * CM))
    data = {}
    area2 = [g for g, a in STS_AREA.items() if a == 2]
    for name in names:
        log = logs[name]
        t = _t(log)
        sel = (t >= lo) & (t <= hi)
        e = _area_err(log, 2)
        axes[0].plot(t[sel], e[sel], color=PALETTE[name], label=name,
                     drawstyle="steps-post", **STYLE[name])
        i_e = int(np.argmin(np.abs(t - t_e)))
        pre = getattr(log[i_e], "pre_control_zone_v_rms_err_pu", {}) or {}
        if 2 in pre:
            axes[0].plot([t_e], [pre[2] * 1e3], "o", ms=3.2, mfc="white",
                         mec=PALETTE[name], mew=0.9, zorder=5)
            data[f"e_v_area2_pre_control_{name}_mpu"] = pd.Series([pre[2] * 1e3])
        data[f"t_min_{name}"] = pd.Series(t[sel])
        data[f"e_v_area2_{name}_mpu"] = pd.Series(e[sel])
        if name in ("S2", "O2", "O3"):
            act = _sts_series(log, "act")
            q = np.nansum(np.vstack([act[g] for g in area2 if g in act]), axis=0)
            i0 = int(np.searchsorted(t, t_e - 1e-6)) - 1      # last pre-event record
            dq = q - q[i0]
            axes[1].plot(t[sel], dq[sel], color=PALETTE[name], label=name,
                         drawstyle="steps-post", **STYLE[name])
            rec = log[i0 + 1]
            q_pre = getattr(rec, "pre_control_trafo_q_mvar", {}) or {}
            cpl = {k.split("|")[-1]: rec.dso_trafo_group.get(k, k.split("|")[0])
                   for k in rec.dso_trafo_q_actual_mvar}
            pre_sum = sum(float(v) for c, v in q_pre.items()
                          if STS_AREA.get(cpl.get(c, ""), 0) == 2)
            if q_pre:
                axes[1].plot([t_e], [pre_sum - q[i0]], "o", ms=3.2, mfc="white",
                             mec=PALETTE[name], mew=0.9, zorder=5)
                data[f"dq_iface_area2_pre_control_{name}_mvar"] = pd.Series([pre_sum - q[i0]])
            data[f"dq_iface_area2_{name}_mvar"] = pd.Series(dq[sel])
    for ax in axes:
        k0 = int(np.ceil(lo / t_ts_min))
        for k in range(k0, int(hi / t_ts_min) + 1):
            ax.axvline(k * t_ts_min, color="#c8c8c8", lw=0.5, zorder=0)
        ax.axvline(t_e, color="#555555", lw=0.7, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("time [min]")
        _grid(ax)
    axes[0].set_ylabel(r"$e_{\mathrm{v},2}$ [mpu]")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel(r"$\Delta Q$ area-2 interfaces [Mvar]")
    axes[1].axhline(0.0, color="#555555", lw=0.5)
    axes[0].set_title("(a) area-2 error", loc="left", fontsize=7.5)
    axes[1].set_title("(b) interface flow change, negative = support",
                      loc="left", fontsize=7.5)
    fig.tight_layout(w_pad=1.0)
    _legend_top(fig, axes[0], ncol=5)
    _save(fig, out, "case3_event")
    return pd.DataFrame(data)


def fig_summary(logs, metrics: Dict, out: Path) -> pd.DataFrame:
    names = [n for n in ORDER if n in metrics and isinstance(metrics[n], dict)
             and "areas" in metrics[n]]
    groups = [str(z) for z in AREAS] + ["all"]
    rows = []
    for n in names:
        m = metrics[n]
        for key, agg in (("e_v", "e_v_rms_areas"), ("e_vp", "e_vp_rms_areas")):
            vals = [m["areas"][z][key] * 1e3 for z in groups[:-1]] + [m[agg] * 1e3]
            rows += [dict(variant=n, metric=key, group=g, value_mpu=v)
                     for g, v in zip(groups, vals)]
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(WIDTH, 9.6 * CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    width = 0.8 / len(names)
    x = np.arange(len(groups))
    for ax, key, title in ((ax_a, "e_v", r"(a) $\bar{e}_{\mathrm{v},a}$, all buses"),
                           (ax_b, "e_vp", r"(b) $\bar{e}_{\mathrm{v},p,a}$, pilot")):
        for i, n in enumerate(names):
            v = df[(df.variant == n) & (df.metric == key)].set_index("group").loc[groups, "value_mpu"]
            ax.bar(x - 0.4 + (i + 0.5) * width, v.values, width * 0.92,
                   color=PALETTE[n], label=n,
                   alpha=0.55 if n in ("L2", "M") else 1.0,
                   hatch="////" if n == "M" else None, edgecolor="white", lw=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(["area 1", "area 2", "area 3", "all"])
        ax.set_ylabel("[mpu]")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
    data = {}
    for n in ORDER:
        if n not in logs:
            continue
        e = np.sort(_agg_err(logs[n]))[::-1]
        share = (np.arange(1, e.size + 1) / e.size) * 100.0
        ax_c.plot(share, e, color=PALETTE[n], label=n, **STYLE[n])
        data[f"share_pct_{n}"] = pd.Series(share)
        data[f"e_v_all_sorted_{n}_mpu"] = pd.Series(e)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel("share of time the aggregate error is exceeded [%]")
    ax_c.set_ylabel(r"$e_{\mathrm{v}}$ [mpu]")
    ax_c.set_title("(c) duration curve of the aggregate error", loc="left",
                   fontsize=7.5)
    _grid(ax_c)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _legend_top(fig, ax_a)
    _save(fig, out, "case3_summary")
    df.to_csv(out / "data" / "case3_summary_bars.csv", index=False)
    return pd.DataFrame(data)


def fig_tieflow(logs, events, out: Path) -> pd.DataFrame:
    pairs = sorted({k for log in logs.values() for r in log[:1]
                    for k in r.zone_tie_q_mvar})
    fig, axes = plt.subplots(len(pairs), 1, figsize=(WIDTH, 3.1 * CM * len(pairs)),
                             sharex=True, squeeze=False)
    data = {}
    for ax, pair in zip(axes[:, 0], pairs):
        for name in ORDER:
            if name not in logs:
                continue
            t = _t(logs[name])
            q = np.array([r.zone_tie_q_mvar.get(pair, np.nan) for r in logs[name]])
            ax.plot(t, q, color=PALETTE[name], label=name, **STYLE[name])
            data[f"t_min_{name}"] = pd.Series(t)
            data[f"tie_{pair[0]}_{pair[1]}_{name}_mvar"] = pd.Series(q)
        _mark(ax, events, labels=(pair == pairs[0]))
        ax.set_ylabel(rf"$Q_{{{pair[0]}\to{pair[1]}}}$ [Mvar]")
        _grid(ax)
    axes[-1, 0].set_xlabel("time [min]")
    axes[-1, 0].set_xlim(0, 360)
    axes[-1, 0].set_xticks(range(0, 361, 60))
    fig.tight_layout(h_pad=0.4)
    _legend_top(fig, axes[0, 0])
    _save(fig, out, "case3_tieflow")
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

STEMS = ("case3_vtrack", "case3_stsv", "case3_qtrack", "case3_event",
         "case3_summary", "case3_tieflow")


def make_figures(run: Path, thesis: Optional[Path] = None) -> List[str]:
    _setup()
    logs = _load(run)
    if not logs:
        raise SystemExit(f"no variant log in {run}")
    events = _event_times(run)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    out = run / "figures"
    (out / "data").mkdir(parents=True, exist_ok=True)
    makers = {
        "case3_vtrack": lambda: fig_vtrack(logs, events, out),
        "case3_stsv": lambda: fig_stsv(logs, events, out),
        "case3_qtrack": lambda: fig_qtrack(logs, events, out),
        "case3_event": lambda: fig_event(logs, events, out),
        "case3_summary": lambda: fig_summary(logs, metrics, out),
        "case3_tieflow": lambda: fig_tieflow(logs, events, out),
    }
    for stem in STEMS:
        makers[stem]().to_csv(out / "data" / f"{stem}.csv", index=False)
    try:
        from experiments.thesis_runs import ThesisRun
        tr = ThesisRun.load(run)
        tr.describe_output("figures/", ", ".join(STEMS) + " (.pdf/.png) and "
                           "figures/data/*.csv")
        tr.write()
    except Exception as exc:           # a copied folder without a manifest
        print(f"(README not updated: {exc})")
    if thesis is not None:
        _to_thesis(run, out, Path(thesis))
    return list(STEMS)


def _to_thesis(run: Path, out: Path, thesis: Path) -> None:
    """PDFs to graphics/Ch11/, CSVs to graphics/Ch11/data/<date>_case3_ref/."""
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    date = run.name.split("_")[1]
    gdir = thesis / "graphics" / "Ch11"
    ddir = gdir / "data" / f"{date}_case3_ref"
    ddir.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        shutil.copy2(out / f"{stem}.pdf", gdir / f"{stem}.pdf")
    for f in sorted((out / "data").glob("*.csv")):
        shutil.copy2(f, ddir / f.name)
    code = (man.get("invocations") or [{}])[0].get("git", {})
    (ddir / "README.md").write_text(
        f"# Data of the Case 2 reference-run figures (ch:case3:nominal)\n\n"
        f"- Run: `qOFO/results/ch_11_case2_multisystem/ref/{run.name}/`\n"
        f"- Code: qOFO {str(code.get('commit', ''))[:7]} on "
        f"{code.get('branch', '')}\n"
        f"- Figures: `graphics/Ch11/{{{','.join(STEMS)}}}.pdf`, made by "
        f"`experiments/ch_11_case2_multisystem/ref/figures_ref.py`\n"
        f"- Files: one CSV per figure (time in min, errors in mpu, flows in "
        f"Mvar); `case3_summary_bars.csv` holds the bar values.\n",
        encoding="utf-8")
    try:
        from experiments.thesis_runs import ThesisRun
        tr = ThesisRun.load(run)
        for stem in STEMS:
            tr.record_export(f"graphics/Ch11/{stem}.pdf", __file__)
        tr.record_export(f"graphics/Ch11/data/{ddir.name}/", __file__)
        tr.write()
    except Exception as exc:           # a copied folder without a manifest
        print(f"(exports not recorded: {exc})")
    print(f"thesis: {len(STEMS)} PDFs -> {gdir}, data -> {ddir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run_ref run folder")
    ap.add_argument("--thesis", default=None,
                    help="LaTeX root: copy PDFs and data into graphics/Ch11")
    a = ap.parse_args()
    print("written:", make_figures(Path(a.run), a.thesis and Path(a.thesis)))


if __name__ == "__main__":
    main()
