#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_voltage.py
===========================
Analyse the e_v SAWTOOTH the user sees in Fig3a (V3 after 60 min).
Computes system-wide RMS voltage tracking error e_v(t) per variant from
``record.zone_v_rms_err_pu``, overlays TSO-step times, measures the V3 teeth
period/amplitude in [60,120] min, and saves a zoom PNG (V3 vs V4, 50-130 min)
with TSO-step markers. Read-only on results/005_cigre.
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join("results", "005_cigre")
DIAG = os.path.join(OUT_ROOT, "_zigzag")


def load(name):
    with open(os.path.join(OUT_ROOT, name, "log.pkl"), "rb") as f:
        return pickle.load(f)


def ev_series(log):
    """System-wide RMS tracking error (mp.u.) + time(min) + tso_active mask."""
    t, e, act = [], [], []
    for r in log:
        d = getattr(r, "zone_v_rms_err_pu", {}) or {}
        vals = [v for v in d.values() if v is not None and np.isfinite(v)]
        if not vals:
            continue
        t.append(getattr(r, "time_s", 0.0) / 60.0)
        e.append(np.sqrt(np.mean(np.square(vals))) * 1e3)  # mp.u.
        act.append(bool(getattr(r, "tso_active", False)))
    return np.array(t), np.array(e), np.array(act, bool)


def teeth(t, e, t0, t1):
    """Local maxima in (t0,t1): count, mean spacing, mean peak-to-trough."""
    m = (t >= t0) & (t <= t1)
    tt, ee = t[m], e[m]
    if ee.size < 5:
        return {}
    # local maxima / minima
    pk = [i for i in range(1, len(ee) - 1) if ee[i] > ee[i-1] and ee[i] >= ee[i+1]]
    tr = [i for i in range(1, len(ee) - 1) if ee[i] < ee[i-1] and ee[i] <= ee[i+1]]
    spacing = np.diff(tt[pk]) if len(pk) > 1 else np.array([])
    # peak-to-trough amplitude (pair each peak with following trough)
    amps = []
    for i in pk:
        later = [j for j in tr if j > i]
        if later:
            amps.append(ee[i] - ee[later[0]])
    return dict(n_peaks=len(pk),
                mean_spacing_min=float(np.mean(spacing)) if spacing.size else np.nan,
                mean_p2t=float(np.mean(amps)) if amps else np.nan,
                emin=float(ee.min()), emax=float(ee.max()))


def main():
    os.makedirs(DIAG, exist_ok=True)
    variants = ["V1", "V2", "V3", "V4", "V5"]
    data = {}
    for v in variants:
        try:
            data[v] = ev_series(load(v))
        except FileNotFoundError:
            print(f"  {v}: missing")
    # TSO step cadence for V3
    t3, e3, a3 = data["V3"]
    tso_t = t3[a3]
    tso_dt = np.diff(tso_t)
    print(f"V3: {len(t3)} records, TSO-active {a3.sum()}  "
          f"median TSO spacing {np.median(tso_dt):.0f} min "
          f"(range {tso_dt.min():.0f}-{tso_dt.max():.0f})")

    for v in variants:
        if v not in data:
            continue
        t, e, a = data[v]
        th = teeth(t, e, 60, 120)
        print(f"\n{v}: e_v[60-120] peaks={th.get('n_peaks','?')} "
              f"spacing={th.get('mean_spacing_min',float('nan')):.1f}min "
              f"peak-to-trough={th.get('mean_p2t',float('nan')):.2f} mp.u. "
              f"(min {th.get('emin',float('nan')):.1f}, max {th.get('emax',float('nan')):.1f})")

    # Print V3 raw e_v 55-130 min with TSO markers
    print("\nV3 e_v(t) 55-130 min  (*=TSO step):")
    m = (t3 >= 55) & (t3 <= 130)
    for tt, ee, aa in zip(t3[m], e3[m], a3[m]):
        bar = "#" * int(round(ee))
        print(f"  {tt:5.0f}min {'*' if aa else ' '} {ee:5.1f} {bar}")

    # Zoom plot V3 vs V4, 50-130 min, with TSO markers
    fig, ax = plt.subplots(figsize=(9, 4))
    for v, col in (("V3", "tab:blue"), ("V4", "tab:green")):
        t, e, a = data[v]
        m = (t >= 50) & (t <= 130)
        ax.plot(t[m], e[m], color=col, label=v, lw=1.3)
        ax.plot(t[m][a[m]], e[m][a[m]], "o", color=col, ms=3)
    for x in (60, 120):
        ax.axvline(x, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("time / min"); ax.set_ylabel("e_v / mp.u.")
    ax.set_title("V3 vs V4 e_v zoom (dots = TSO steps); 60=gen trip, 120=load")
    ax.legend()
    png = os.path.join(DIAG, "v3_ev_zoom.png")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
