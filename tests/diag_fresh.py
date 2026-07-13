#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/diag_fresh.py
=========================
RE-RUN at the TRUE current config (Claude Code, 2026-06-24).
Earlier analysis used stale pickles (TSO@360s, dt=60, 600-min V4/V5).  This uses
make_cigre_config() UNMODIFIED: TSO 180 s, DSO 20 s, dt 20 s, 300 min.

Variants:
  Vdbg  TS-OFO + STS cos-phi=1  (Q(V) OFF at HV)  <- debug: isolates Q(V)
  V3    TS-OFO + STS local Q(V)  (one-sided)
  V4    cascaded TS-OFO + STS-OFO (proposed)
  V5    central OFO

Compares the e_v sawtooth across all four; if Vdbg (no HV Q(V)) still sawtooths,
the teeth are TSO-cadence inter-sample drift, not a Q(V) effect.  Writes
results/_diag_fresh/.
"""
from __future__ import annotations
import os, sys, time, pickle, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
_spec = importlib.util.spec_from_file_location(
    "cigre005", os.path.join(os.path.dirname(HERE), "experiments", "005_CIGRE_MULTI.py"))
c5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c5)

from experiments.runners import run_multi_tso_dso
from tests.diag_voltage import ev_series, teeth

OUT = os.path.join("results", "_diag_fresh")

VDBG = dict(
    tso_mode="ofo",
    dso_mode="local", local_der_mode="cos_phi_1",
    tso_q_mode="qv", dso_q_mode="cosphi",
    tso_qv_vref_pu=1.03, tso_qv_slope_pu=0.06, tso_qv_deadband_pu=0.01,
    g_w_pcc=1.0e10,
)

RUNS = [
    ("Vdbg", VDBG),
    ("V3", c5.VARIANTS["V3"]),
    ("V4", c5.VARIANTS["V4"]),
    ("V5", c5.VARIANTS["V5"]),
]


def run_true(tag, overrides):
    cfg = c5.make_cigre_config()          # TRUE config; NO dt/period override
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.verbose = 0
    d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
    cfg.result_dir = d
    t0 = time.time()
    try:
        log = run_multi_tso_dso(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] FAILED {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        log = []
    with open(os.path.join(d, "log.pkl"), "wb") as f:
        pickle.dump(log, f)
    print(f"[{tag}] {len(log)} records in {time.time()-t0:.0f}s "
          f"(dt={cfg.dt_s}s, tso={cfg.tso_period_s}s, dso={cfg.dso_period_s}s)",
          flush=True)
    return log


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for tag, ov in RUNS:
        print("\n" + "=" * 72 + f"\n  RUN {tag}\n" + "=" * 72, flush=True)
        log = run_true(tag, ov)
        if log:
            data[tag] = ev_series(log)

    print("\n" + "#" * 72 + "\n  e_v teeth (TRUE config) — window [65,175] min\n" + "#" * 72)
    print(f"{'run':<6}{'#TSOsteps':>10}{'TSOspacing[min]':>16}{'peaks':>7}"
          f"{'spacing[min]':>13}{'p2t[mp.u.]':>12}")
    for tag, _ in RUNS:
        if tag not in data:
            print(f"{tag:<6}  (empty)"); continue
        t, e, a = data[tag]
        tso_dt = np.diff(t[a]) if a.sum() > 1 else np.array([np.nan])
        th = teeth(t, e, 65, 175)
        print(f"{tag:<6}{int(a.sum()):>10}{np.median(tso_dt):>16.1f}"
              f"{th.get('n_peaks',0):>7}{th.get('mean_spacing_min',np.nan):>13.1f}"
              f"{th.get('mean_p2t',np.nan):>12.2f}")

    # full-horizon + zoom plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    for tag, col in (("Vdbg", "tab:orange"), ("V3", "tab:blue"),
                     ("V4", "tab:green"), ("V5", "k")):
        if tag not in data:
            continue
        t, e, a = data[tag]
        axs[0].plot(t, e, color=col, lw=1.0, label=tag)
        m = (t >= 55) & (t <= 130)
        axs[1].plot(t[m], e[m], color=col, lw=1.2, label=tag)
        axs[1].plot(t[m][a[m]], e[m][a[m]], "o", color=col, ms=2.5)
    for ax in axs:
        for x in (60, 120, 180, 260):
            ax.axvline(x, color="0.7", lw=0.7, ls="--")
        ax.set_ylabel("e_v / mp.u."); ax.legend(ncol=4, fontsize=8)
    axs[0].set_title("e_v full horizon (TRUE config: TSO 180s, DSO 20s, dt 20s)")
    axs[1].set_title("zoom 55-130 min (dots = TSO steps)")
    axs[1].set_xlabel("time / min")
    png = os.path.join(OUT, "ev_compare_truecfg.png")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
