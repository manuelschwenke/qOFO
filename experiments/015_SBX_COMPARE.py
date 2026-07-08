#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/015_SBX_COMPARE.py
==============================
Small paired comparison ``none`` vs ``sbx`` on one 013 scenario:
voltage-tracking quality, tie-line (corridor) flows, and the reactive
infeed of generators, TSO-DER and the DSOs — as small-multiple figures
with the two arms overlaid.

Data source: the per-arm pickles of the 013 campaign
(``results/013_SBX_LADDER/<scenario>/arm_{none,sbx}.pkl`` — 360 min,
calibrated tier-1 band).  Nothing is re-run by default; ``--run``
executes the two missing arms through the 013 machinery if the pickles
are absent.

Figures (``results/015_SBX_COMPARE/<scenario>/``):

* ``F1_voltage_tracking.png`` — per zone: the recorded voltage-tracking
  RMS error (left) and the zone voltage envelope [min, max] with the
  zone's hard bounds (right).
* ``F2_corridor_flows.png``   — per corridor: the recorded inter-zone
  reactive flow of both arms, with the sbx arm's schedule staircase for
  context.  (Convention note: ``rec.zone_tie_q_mvar`` negates the
  from-end flow on ties oriented from the higher zone — finding F7 in
  STATUS_SBX.md — which is identical in both arms, so the COMPARISON is
  unaffected; absolute levels on corridors (1,2)/(1,3) differ from the
  SBX reference-end convention by the line charging.)
* ``F3_reactive_infeed.png``  — 3 zones x 3 sources: synchronous
  generators, TSO-DER, and the DSO interface exchange (Q out of the
  transmission level), both arms overlaid.

Colour convention (fixed per arm across every panel): none = neutral
grey, sbx = blue.

Run:
    python experiments/015_SBX_COMPARE.py --scenario asym_z3
    python experiments/015_SBX_COMPARE.py --scenario asym_z2 --run

Author: Manuel Schwenke / Claude Code
Date: 2026-07-08 (SBX Phase 7 follow-up)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sbx.fail import rep1  # noqa: E402

_013 = importlib.import_module("experiments.013_SBX_LADDER")

RESULT_DIR = REPO / "results" / "015_SBX_COMPARE"

#: Fixed arm colours (identity never repainted; grey/blue is CVD-safe).
C_NONE = "#888888"
C_SBX = "#4477aa"
C_SCHED = "#117733"
C_BOUND = "#cc3311"
C_STRESS = "#f0efec"

ZONES = (1, 2, 3)


# ───────────────────────────────────────────────────────────────────────
#  Data access
# ───────────────────────────────────────────────────────────────────────


def load_arm(scenario: str, arm: str) -> dict:
    pkl = REPO / "results" / "013_SBX_LADDER" / scenario / f"arm_{arm}.pkl"
    if not pkl.exists():
        rep1("013 arm pickle missing — run 015 with --run or the 013 "
             "campaign first", scenario=scenario, arm=arm, path=str(pkl))
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def ensure_arms(scenario: str, minutes: float) -> None:
    """Run the two arms through the 013 machinery if pickles are absent
    (band for the sbx arm = the 014 calibrated defaults)."""
    sdir = REPO / "results" / "013_SBX_LADDER" / scenario
    sdir.mkdir(parents=True, exist_ok=True)
    _014 = importlib.import_module("experiments.014_SBX_SINGLE_DEMO")
    for arm in ("none", "sbx"):
        pkl = sdir / f"arm_{arm}.pkl"
        if pkl.exists():
            continue
        band = (_014.CALIBRATED_BAND_MVAR[scenario]
                if arm == "sbx" else None)
        _cfg, recs, ext = _013.run_arm(scenario, arm, minutes, band)
        with open(pkl, "wb") as fh:
            pickle.dump({"scenario": scenario, "arm": arm,
                         "minutes": minutes, "records": recs,
                         "sbx": ext, "q_band_mvar": band}, fh)


def series(recs, getter) -> tuple:
    """(t_min, values) for a per-record getter; NaN where absent."""
    t, v = [], []
    for r in recs:
        t.append(r.time_s / 60.0)
        try:
            v.append(getter(r))
        except (KeyError, AttributeError):
            v.append(float("nan"))
    return np.asarray(t), np.asarray(v, dtype=np.float64)


# ───────────────────────────────────────────────────────────────────────
#  Figures
# ───────────────────────────────────────────────────────────────────────


def _decorate(ax, ylab: str, title: str) -> None:
    ax.axvspan(_013.STRESS_ON_MIN, _013.STRESS_OFF_MIN,
               color=C_STRESS, zorder=0)
    ax.axvline(_013.WARMUP_MIN, color="0.75", lw=0.7, ls=":")
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=8)


def fig_voltage_tracking(scenario, recs_none, recs_sbx, spec, cfg_ref,
                         out_dir: Path) -> None:
    fig, axes = plt.subplots(len(ZONES), 2, figsize=(11, 7.2),
                             sharex=True)
    for i, z in enumerate(ZONES):
        # Left: voltage-tracking RMS error (the quality metric proper).
        ax = axes[i, 0]
        for recs, col, lab in ((recs_none, C_NONE, "none"),
                               (recs_sbx, C_SBX, "sbx")):
            t, v = series(recs, lambda r: r.zone_v_rms_err_pu[z])
            ax.plot(t, 1e3 * v, color=col, lw=1.2, label=lab)
        _decorate(ax, "RMS err / mpu",
                  f"zone {z}: voltage-tracking RMS error")
        # Right: zone voltage envelope with the zone's hard bounds.
        ax = axes[i, 1]
        v_lo = spec["zone_v_min"].get(z, cfg_ref.v_min_pu)
        v_hi = spec["zone_v_max"].get(z, cfg_ref.v_max_pu)
        for recs, col, lab in ((recs_none, C_NONE, "none"),
                               (recs_sbx, C_SBX, "sbx")):
            t, vmin = series(recs, lambda r: r.zone_v_min[z])
            _, vmax = series(recs, lambda r: r.zone_v_max[z])
            ax.fill_between(t, vmin, vmax, color=col, alpha=0.22, lw=0)
            ax.plot(t, vmin, color=col, lw=1.0, label=f"{lab} [min, max]")
        for b in (v_lo, v_hi):
            if cfg_ref.v_min_pu - 0.02 < b < cfg_ref.v_max_pu + 0.02:
                ax.axhline(b, color=C_BOUND, lw=0.9, ls="--")
        _decorate(ax, "V / pu", f"zone {z}: voltage envelope and bounds")
    for ax in axes[-1, :]:
        ax.set_xlabel("time / min")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{scenario}: voltage-tracking quality — none vs sbx",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "F1_voltage_tracking.png", dpi=160)
    plt.close(fig)


def fig_corridor_flows(scenario, recs_none, recs_sbx, sbx_ext,
                       out_dir: Path) -> None:
    keys = sorted({k for r in recs_sbx for k in r.zone_tie_q_mvar})
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.6 * len(keys)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for recs, col, lab in ((recs_none, C_NONE, "none"),
                               (recs_sbx, C_SBX, "sbx")):
            t, q = series(recs, lambda r: r.zone_tie_q_mvar[key])
            ax.plot(t, q, color=col, lw=1.2, label=lab)
        # sbx schedule staircase for context (reference-end convention;
        # see the module docstring's F7 note on the level offset).
        if sbx_ext is not None and key in sbx_ext["records"]:
            rl = sbx_ext["records"][key]
            cfg = sbx_ext["config"]
            tt = [_013.WARMUP_MIN + (rl[0].cycle - 1) * cfg.t_cycle_min]
            qq = []
            for r in rl:
                tt.append(_013.WARMUP_MIN + r.cycle * cfg.t_cycle_min)
                qq.append(r.q_sched_mvar)
            ax.step(tt, [qq[0]] + qq, where="pre", color=C_SCHED,
                    lw=1.0, ls="--", label="sbx q_sched (ref-end conv.)")
            deals = [(_013.WARMUP_MIN + r.cycle * cfg.t_cycle_min,
                      r.q_sched_mvar) for r in rl
                     if r.deal.dq_deal_mvar != 0.0]
            if deals:
                ax.scatter(*zip(*deals), marker="v", s=24,
                           color=C_SCHED, zorder=5, label="deal")
        _decorate(ax, "Q / Mvar",
                  f"corridor ({key[0]},{key[1]}): inter-zone reactive "
                  f"flow (+ = leaves zone {key[0]})")
    axes[0].legend(fontsize=8, frameon=False, ncol=4)
    axes[-1].set_xlabel("time / min")
    fig.suptitle(f"{scenario}: corridor flows — none vs sbx",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "F2_corridor_flows.png", dpi=160)
    plt.close(fig)


def fig_reactive_infeed(scenario, recs_none, recs_sbx,
                        out_dir: Path) -> None:
    sources = (
        ("synchronous generators",
         lambda r, z: r.zone_balance_gen_q_mvar[z]),
        ("TSO-DER",
         lambda r, z: r.zone_balance_der_q_mvar[z]),
        ("DSO interface (Q out of TN)",
         lambda r, z: r.zone_balance_tso_dso_q_out_mvar[z]),
    )
    fig, axes = plt.subplots(len(ZONES), len(sources),
                             figsize=(12.5, 7.2), sharex=True)
    for i, z in enumerate(ZONES):
        for j, (name, getter) in enumerate(sources):
            ax = axes[i, j]
            drew = False
            for recs, col, lab in ((recs_none, C_NONE, "none"),
                                   (recs_sbx, C_SBX, "sbx")):
                t, q = series(recs, lambda r: getter(r, z))
                if np.all(np.isnan(q)):
                    continue
                ax.plot(t, q, color=col, lw=1.1, label=lab)
                drew = True
            if not drew:
                ax.text(0.5, 0.5, "not recorded", ha="center",
                        va="center", transform=ax.transAxes,
                        fontsize=8, color="0.5")
            _decorate(ax, "Q / Mvar" if j == 0 else "",
                      f"zone {z}: {name}")
    for ax in axes[-1, :]:
        ax.set_xlabel("time / min")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{scenario}: reactive infeed by source — none vs sbx",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "F3_reactive_infeed.png", dpi=160)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
#  Driver
# ───────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired none-vs-sbx comparison figures from the "
                    "013 pickles.")
    ap.add_argument("--scenario", type=str, default="compl_z1z3",
                    choices=sorted(_013.SCENARIOS.keys()))
    ap.add_argument("--run", action="store_true",
                    help="run missing arms (360 min each) instead of "
                         "failing on absent pickles")
    ap.add_argument("--minutes", type=float, default=_013.DEFAULT_MINUTES)
    args = ap.parse_args()

    scenario = args.scenario
    if args.run:
        ensure_arms(scenario, args.minutes)
    d_none = load_arm(scenario, "none")
    d_sbx = load_arm(scenario, "sbx")
    recs_none, recs_sbx = d_none["records"], d_sbx["records"]
    sbx_ext = d_sbx["sbx"]
    spec = _013.SCENARIOS[scenario]
    cfg_ref = _013.make_config(scenario, "none", args.minutes)

    out_dir = RESULT_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_voltage_tracking(scenario, recs_none, recs_sbx, spec, cfg_ref,
                         out_dir)
    fig_corridor_flows(scenario, recs_none, recs_sbx, sbx_ext, out_dir)
    fig_reactive_infeed(scenario, recs_none, recs_sbx, out_dir)

    # Compact numeric summary alongside the figures.
    lines = [f"# 015 comparison — {scenario} (none vs sbx)", ""]
    for z in ZONES:
        _, e_none = series(recs_none, lambda r: r.zone_v_rms_err_pu[z])
        _, e_sbx = series(recs_sbx, lambda r: r.zone_v_rms_err_pu[z])
        lines.append(
            f"- zone {z}: mean tracking RMS error "
            f"none {1e3 * np.nanmean(e_none):.2f} mpu, "
            f"sbx {1e3 * np.nanmean(e_sbx):.2f} mpu"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print(f"figures + summary written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
