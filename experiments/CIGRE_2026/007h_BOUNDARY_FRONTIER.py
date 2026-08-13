#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007h_BOUNDARY_FRONTIER.py
================================================
Compare the two boundary models on their TUNING FRONTIERS, not at single
points -- and find out why a change to how a TS represents ANOTHER TS shows up
in the TS-DS interface tracking error.

Why a frontier
--------------
``007g`` compared single (boundary, gain) points and the comparison kept
collapsing into the tuning confound: a truer H implies a different loop gain,
so any single-point result mixes model quality with detuning.  The clean
comparison sweeps the SAME gain ladder on both boundaries and asks which
resulting curve dominates in the (voltage tracking, interface tracking) plane.
A model that is genuinely better must produce a frontier that lies inside the
other's -- better voltage tracking at equal interface error, or vice versa.
That question has no tuning confound left in it.

The interface-tracking puzzle
-----------------------------
``rms_e_sts`` is ``q_actual - q_set`` at the TSO-DSO interface -- the DSO's
error against the setpoint the TSO dispatches.  Changing how a TS area
represents a NEIGHBOURING TS area should not touch it.  In 007g it moved
monotonically with how aggressive the TS loop was, across both boundaries:

    th 1.95 (n_sw 1)  <  pq 1.94  <  th_gw 2.19  <  th_gwz 2.26  <  pq_gwz 2.55

Hypothesis: it is not a boundary effect at all.  ``Q_DS,set`` is one of the
TSO's own actuators.  A more aggressive TS loop moves that setpoint further and
faster each update, so the DSO -- which is supposed to settle within one parent
period -- is chasing a faster-moving target and its residual error grows.  The
cascade's timescale separation is being eroded from above.

The test: measure how much the TSO actually moves the setpoint (RMS of the
per-update change in ``Q_DS,set``) and check whether ``rms_e_sts`` is a single
function of THAT, independent of which boundary produced it.  If one curve
fits both boundaries, the interface degradation is a loop-gain consequence and
the boundary model is exonerated.

Usage
-----
    python experiments/CIGRE_2026/007h_BOUNDARY_FRONTIER.py
    python experiments/CIGRE_2026/007h_BOUNDARY_FRONTIER.py --replot

Each arm is pickled, so an interrupted sweep resumes.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.helpers.comparison_metrics import (  # noqa: E402
    cigre_summary_table, gen_s_rated_by_zone, q_iface_err_per_group,
    voltage_rms_err_all, voltage_rms_err_per_zone,
)
from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

OUT_DIR = _ROOT / "results" / "007_tie_boundary" / "frontier"
K_TH = 1.5
_TSO_GW_KEYS = ("g_w_der", "g_w_gen", "g_w_pcc", "g_w_tso_oltc")

#: Gain ladder, applied identically to both boundaries.  1.0 is the shipped
#: tuning; smaller = more aggressive TS loop (step ~ H / g_w).
KAPPAS: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.2)
BOUNDARIES: Tuple[str, ...] = ("pq", "thevenin")


def _arm_name(boundary: str, kap: float) -> str:
    return f"{'pq' if boundary == 'pq' else 'th'}_k{kap:g}"


def _build_cfg(boundary: str, kap: float, hours: float):
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.n_total_s = 3600.0 * hours
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    cfg.tie_boundary_equivalent = boundary
    if boundary == "thevenin":
        cfg.tie_thevenin_k = K_TH
    for gk in _TSO_GW_KEYS:
        setattr(cfg, gk, getattr(cfg, gk) * float(kap))
    return cfg


def _setpoint_movement(recs) -> float:
    """RMS per-update change in the dispatched interface setpoint [Mvar].

    Summed over DSO groups, so it measures how hard the TSO is driving the
    cascade's downward channel -- the quantity the DSO has to chase.  Zero
    steps (between TSO updates the setpoint is held) are excluded, otherwise
    the figure would just report the TSO/DSO period ratio.
    """
    d = q_iface_err_per_group(recs)
    per_group: List[float] = []
    for g in d["groups"]:
        s = np.asarray(d["set_mvar"][g], dtype=float)
        s = s[np.isfinite(s)]
        if s.size < 2:
            continue
        ds = np.diff(s)
        ds = ds[np.abs(ds) > 1e-9]
        if ds.size:
            per_group.append(float(np.sqrt(np.mean(ds ** 2))))
    return float(np.mean(per_group)) if per_group else float("nan")


def _rms_iface_err(recs) -> float:
    d = q_iface_err_per_group(recs)
    vals: List[float] = []
    for g in d["groups"]:
        e = np.asarray(d["err_mvar"][g], dtype=float)
        e = e[np.isfinite(e)]
        if e.size:
            vals.append(float(np.sqrt(np.mean(e ** 2))))
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print(f"BOUNDARY TUNING FRONTIER -- {args.hours:g} h QSS, V4 cascade")
    print(f"  boundaries {BOUNDARIES}   gain ladder kappa {KAPPAS}")
    print("=" * 82)

    logs: Dict[str, List[Any]] = {}
    for boundary in BOUNDARIES:
        for kap in KAPPAS:
            arm = _arm_name(boundary, kap)
            pkl = OUT_DIR / f"{arm}_{args.hours:g}h.pkl"
            if pkl.exists():
                with open(pkl, "rb") as fh:
                    logs[arm] = pickle.load(fh)
                print(f"[{arm}] loaded {len(logs[arm])} records from disk")
                continue
            if args.replot:
                print(f"[{arm}] no pickle and --replot set -- skipped")
                continue
            print(f"\n[{arm}] boundary={boundary} kappa={kap:g} ...", flush=True)
            t0 = time.time()
            try:
                recs = run_multi_tso_dso(_build_cfg(boundary, kap, args.hours))
            except Exception as exc:
                print(f"[{arm}] FAILED ({type(exc).__name__}: {exc})")
                continue
            if not recs:
                print(f"[{arm}] no records")
                continue
            print(f"[{arm}] {len(recs)} records in {time.time() - t0:.0f} s")
            logs[arm] = recs
            with open(pkl, "wb") as fh:
                pickle.dump(recs, fh)

    if not logs:
        print("\nnothing to analyse")
        return 1

    v_set = 1.03
    try:
        rated = gen_s_rated_by_zone("base_410")
    except Exception:
        rated = None

    rows: List[Dict[str, Any]] = []
    for boundary in BOUNDARIES:
        for kap in KAPPAS:
            arm = _arm_name(boundary, kap)
            if arm not in logs:
                continue
            recs = logs[arm]
            va = voltage_rms_err_all(recs, v_set)
            e = np.asarray(va["rms_err_pu"], dtype=float)
            pz = voltage_rms_err_per_zone(recs, v_set)
            per_zone = {}
            for z in pz.get("zones", []):
                ez = np.asarray(pz["rms_err_pu"][z], dtype=float)
                per_zone[z] = (float(np.nanmean(ez))
                               if np.any(np.isfinite(ez)) else np.nan)
            rows.append(dict(
                arm=arm, boundary=boundary, kappa=kap,
                rms_v=float(np.nanmean(e)) if np.any(np.isfinite(e)) else np.nan,
                rms_e_sts=_rms_iface_err(recs),
                set_move=_setpoint_movement(recs),
                per_zone=per_zone,
            ))

    try:
        df = cigre_summary_table(logs, v_set=v_set, gen_s_rated_mva=rated)
        df.to_csv(OUT_DIR / f"summary_{args.hours:g}h.csv")
        nsw = {ix: int(df.loc[ix, "n_sw"]) for ix in df.index}
        tie = {ix: float(df.loc[ix, "rms_q_tie_mvar"]) for ix in df.index}
    except Exception as exc:
        print(f"\n(summary table unavailable: {type(exc).__name__}: {exc})")
        nsw, tie = {}, {}

    print("\n" + "=" * 82)
    print("FRONTIER")
    print("=" * 82)
    hdr = (f"{'arm':>10} {'kappa':>6} {'rms_v_ts':>10} {'rms_e_sts':>10} "
           f"{'set_move':>10} {'n_sw':>5} {'q_tie':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>10} {r['kappa']:>6.2f} {r['rms_v']:>10.5f} "
              f"{r['rms_e_sts']:>10.4f} {r['set_move']:>10.4f} "
              f"{nsw.get(r['arm'], -1):>5} {tie.get(r['arm'], float('nan')):>8.2f}")

    # ── Frontier dominance: at comparable interface error, who tracks better? ──
    print("\nPaired at equal kappa (same gain ladder, so the only difference "
          "is the boundary):")
    hdr2 = (f"{'kappa':>6} {'rms_v pq':>10} {'rms_v th':>10} {'th vs pq':>10} "
            f"{'e_sts pq':>10} {'e_sts th':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    for kap in KAPPAS:
        a = next((r for r in rows if r["boundary"] == "pq" and r["kappa"] == kap), None)
        b = next((r for r in rows if r["boundary"] == "thevenin" and r["kappa"] == kap), None)
        if not a or not b:
            continue
        rel = ((b["rms_v"] - a["rms_v"]) / a["rms_v"] * 100.0
               if np.isfinite(a["rms_v"]) and a["rms_v"] > 0 else np.nan)
        print(f"{kap:>6.2f} {a['rms_v']:>10.5f} {b['rms_v']:>10.5f} "
              f"{rel:>9.1f}% {a['rms_e_sts']:>10.4f} {b['rms_e_sts']:>10.4f}")

    # ── The interface question: is e_sts one function of setpoint movement? ──
    print("\nInterface error vs how hard the TSO drives the setpoint.")
    print("If ONE relationship fits both boundaries, the interface degradation")
    print("is a loop-gain effect and not a boundary effect.")
    hdr3 = f"{'boundary':>10} {'kappa':>6} {'set_move':>10} {'rms_e_sts':>10} {'ratio':>8}"
    print(hdr3)
    print("-" * len(hdr3))
    for boundary in BOUNDARIES:
        for kap in KAPPAS:
            r = next((x for x in rows if x["boundary"] == boundary
                      and x["kappa"] == kap), None)
            if not r:
                continue
            ratio = (r["rms_e_sts"] / r["set_move"]
                     if np.isfinite(r["set_move"]) and r["set_move"] > 0
                     else np.nan)
            print(f"{boundary:>10} {kap:>6.2f} {r['set_move']:>10.4f} "
                  f"{r['rms_e_sts']:>10.4f} {ratio:>8.3f}")
    fin = [r for r in rows
           if np.isfinite(r.get("set_move", np.nan))
           and np.isfinite(r.get("rms_e_sts", np.nan)) and r["set_move"] > 0]
    if len(fin) >= 3:
        x = np.array([r["set_move"] for r in fin])
        y = np.array([r["rms_e_sts"] for r in fin])
        c = float(np.corrcoef(x, y)[0, 1])
        print(f"\ncorrelation(setpoint movement, interface error) over all "
              f"{len(fin)} arms: r = {c:.3f}")
        ratios = y / x
        print(f"ratio e_sts/set_move: mean {ratios.mean():.3f}, "
              f"spread [{ratios.min():.3f}, {ratios.max():.3f}]")

    print("\nPer-zone voltage RMS:")
    zs = sorted({z for r in rows for z in r["per_zone"]})
    hdr4 = f"{'arm':>10}" + "".join(f"{'zone ' + str(z):>11}" for z in zs)
    print(hdr4)
    print("-" * len(hdr4))
    for r in rows:
        print(f"{r['arm']:>10}" + "".join(
            f"{r['per_zone'].get(z, np.nan):>11.5f}" for z in zs))

    with open(OUT_DIR / f"frontier_{args.hours:g}h.csv", "w",
              encoding="utf-8") as fh:
        fh.write("arm,boundary,kappa,rms_v_ts,rms_e_sts,set_move,n_sw,q_tie\n")
        for r in rows:
            fh.write(f"{r['arm']},{r['boundary']},{r['kappa']},{r['rms_v']},"
                     f"{r['rms_e_sts']},{r['set_move']},"
                     f"{nsw.get(r['arm'], '')},{tie.get(r['arm'], '')}\n")
    print(f"\nwritten: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
