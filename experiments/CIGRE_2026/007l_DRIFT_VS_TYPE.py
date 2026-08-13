#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007l_DRIFT_VS_TYPE.py
============================================
Which error matters more: the boundary TYPE, or the fact that the boundary is
never refreshed?

The reduced model is built once at initialisation and frozen (verified: the
only rebuild path is guarded on a shunt switch, and the case study installs no
shunts).  So its equivalents carry two independent errors:

* **type error**   -- the DERIVATIVE is wrong.  PQ over-states the corridor
  gain ~2.8x even at a perfectly anchored operating point.  Refreshing does
  not fix this.
* **drift error**  -- the ANCHOR is stale.  Every type reproduces the operating
  point exactly at build time by construction and diverges from it as load
  profiles move.  Changing the type does not fix this.

Complementary, not substitutes.  Everything measured so far addressed only the
first.  This sizes the second against it, in the same units, with the same
estimator:

    truth        = numerical H on the full plant at t1
    frozen  pq   = reduced net built from the t0 plant, PQ boundary
    frozen  th   = reduced net built from the t0 plant, Thevenin boundary
    refreshed pq = reduced net built from the t1 plant, PQ boundary
    refreshed th = reduced net built from the t1 plant, Thevenin boundary

A refresh needs no new information: it re-reads corridor flows, own terminal
voltages and coupler flows -- quantities the TSO already measures -- at the
current instant instead of at t0.  The ownership premise is untouched.

Caveat: the t1 plant here is the profile-driven state at t1 from a fresh
initialisation, not the state a 5 h controlled run would have reached.  It
therefore isolates operating-point drift from the profiles, and excludes
control-induced drift.  Profile drift is the dominant driver over this horizon,
but the number is a lower bound on total drift.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402
from sensitivity.network_reduction import (  # noqa: E402
    THEVENIN_K_PER_CORRIDOR, build_tso_local_net,
)
from sensitivity.numerical_h import compute_numerical_h_tso  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).with_name(fname)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_CIGRE = _load("_cigre005", "005_CIGRE_MULTI.py")
_D = _load("_cigre007d", "007d_THEVENIN_SWEEP.py")  # _freeze, _zone_args

OUT_DIR = _ROOT / "results" / "007_tie_boundary"


def _capture_at(t: datetime) -> Dict[str, Any]:
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.start_time = t
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    st: Dict[str, Any] = {}
    run_multi_tso_dso(cfg, pre_loop_hook=lambda s: (st.update(s), True)[1])
    return st


def _corridor_rows(ctrl, zd) -> List[int]:
    vb = list(ctrl.config.voltage_bus_indices)
    return sorted({vb.index(int(b)) for b in (zd.tie_line_endpoint_buses or [])
                   if int(b) in vb})


def _rel(a, b, rows=None) -> float:
    if rows is not None:
        a, b = a[rows, :], b[rows, :]
    d = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / d) if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0,
                    help="drift horizon t1 - t0")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = datetime(2016, 1, 5, 8, 0)
    t1 = t0 + timedelta(hours=args.hours)
    print("=" * 92)
    print(f"DRIFT vs TYPE -- boundary models built at {t0:%H:%M}, "
          f"scored against the plant at {t1:%H:%M}")
    print("=" * 92)

    print(f"\n[setup] t0 = {t0:%Y-%m-%d %H:%M} ...", flush=True)
    st0 = _capture_at(t0)
    print(f"[setup] t1 = {t1:%Y-%m-%d %H:%M} ...", flush=True)
    st1 = _capture_at(t1)
    plant0, plant1 = st0["net"], st1["net"]

    dv = float(np.max(np.abs(plant1.res_bus["vm_pu"].values
                             - plant0.res_bus["vm_pu"].values)))
    print(f"\n[check] operating point moved between t0 and t1: "
          f"max |dV| = {dv:.4f} pu")
    if dv < 1e-4:
        print("  !! the two states are essentially identical -- the profiles "
              "may not be advancing; drift cannot be measured this way")
        return 1

    variants = [("pq", dict(tie_boundary="pq")),
                ("th", dict(tie_boundary="thevenin",
                            tie_thevenin_k=THEVENIN_K_PER_CORRIDOR))]

    rows: List[Dict[str, Any]] = []
    for z in sorted(st1["tso_controllers"].keys()):
        ctrl = st1["tso_controllers"][z]
        zd = st1["zone_defs"][z]
        cr = _corridor_rows(ctrl, zd)

        print("\n" + "-" * 92)
        print(f"ZONE {z}")
        print("-" * 92)
        print("  truth H on the full plant at t1 ...", flush=True)
        truth = compute_numerical_h_tso(_D._freeze(plant1), ctrl,
                                        closed_loop=False)

        args_t1 = _D._zone_args(st1, z)
        args_t0 = dict(args_t1)
        args_t0["net"] = plant0  # same index sets, older operating point

        hdr = (f"{'model':>16} {'relF all':>10} {'relF corridor':>14}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for lb, kw in variants:
            for age, base in (("frozen@t0", args_t0), ("refreshed@t1", args_t1)):
                try:
                    red = build_tso_local_net(**base, **kw).net
                    h = compute_numerical_h_tso(_D._freeze(red), ctrl,
                                                closed_loop=False)
                except Exception as exc:
                    print(f"{lb + ' ' + age:>16}  FAILED "
                          f"({type(exc).__name__}: {exc})")
                    continue
                e_all = _rel(h, truth)
                e_cor = _rel(h, truth, cr) if cr else float("nan")
                rows.append(dict(zone=z, boundary=lb, age=age,
                                 relF_all=e_all, relF_corr=e_cor))
                print(f"{lb + ' ' + age:>16} {e_all:>10.4f} {e_cor:>14.4f}")

    # ── Which axis dominates? ─────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("VERDICT -- mean over zones, corridor rows")
    print("=" * 92)

    def mean_of(b, a, key="relF_corr"):
        v = [r[key] for r in rows if r["boundary"] == b and r["age"] == a
             and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    grid = {(b, a): mean_of(b, a) for b in ("pq", "th")
            for a in ("frozen@t0", "refreshed@t1")}
    hdr = f"{'':>14}" + "".join(f"{a:>16}" for a in ("frozen@t0", "refreshed@t1"))
    print(hdr)
    print("-" * len(hdr))
    for b in ("pq", "th"):
        print(f"{b:>14}" + "".join(
            f"{grid[(b, a)]:>16.4f}" for a in ("frozen@t0", "refreshed@t1")))

    d_type_frozen = grid[("pq", "frozen@t0")] - grid[("th", "frozen@t0")]
    d_drift_pq = grid[("pq", "frozen@t0")] - grid[("pq", "refreshed@t1")]
    d_drift_th = grid[("th", "frozen@t0")] - grid[("th", "refreshed@t1")]
    print(f"\n  error removed by fixing the TYPE  (pq -> th, both frozen): "
          f"{d_type_frozen:+.4f}")
    print(f"  error removed by REFRESHING pq                            : "
          f"{d_drift_pq:+.4f}")
    print(f"  error removed by REFRESHING th                            : "
          f"{d_drift_th:+.4f}")
    if np.isfinite(d_type_frozen) and np.isfinite(d_drift_pq):
        if abs(d_drift_pq) > abs(d_type_frozen):
            print("\n  -> DRIFT dominates: refreshing the existing PQ model "
                  "buys more than switching type.")
        else:
            print("\n  -> TYPE dominates: the boundary derivative matters more "
                  "than the staleness of its anchor.")

    out = OUT_DIR / "drift_vs_type.csv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("zone,boundary,age,relF_all,relF_corr\n")
        for r in rows:
            fh.write(f"{r['zone']},{r['boundary']},{r['age']},"
                     f"{r['relF_all']},{r['relF_corr']}\n")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
