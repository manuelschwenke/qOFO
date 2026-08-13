#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007j_CORRIDOR_VS_INTERIOR.py
===================================================
Why does the LESS faithful boundary model control better?

Standing puzzle: Thevenin's H is 12-70x more faithful than PQ's by every row
block (007i), yet it tracks 15-22 % worse in closed loop at every gain setting
(007h).  Two candidate mechanisms have already been refuted -- degraded Q_PCC
rows, and over-stated gain compensating an under-tuned loop.

Third hypothesis, and the one this script tests: **shared-boundary
under-provision.**

Both neighbouring areas monitor and track their OWN terminal of the same
corridor, both toward the same schedule.  A Thevenin boundary correctly tells
area i that the neighbour is stiff and partly holds that bus -- so the corridor
term contributes less to area i's gradient and it invests less effort there,
reallocating inward.  But the neighbour's model says the same thing about area
i.  Both correctly recognise the other's contribution, both correctly reduce
their own, and the shared bus ends up under-served.  PQ's "error" -- believing
the far end floats freely, so the corridor terminal is mine alone to fix --
makes each area take full responsibility, and since both push the same way the
efforts add.

If that is what is happening, the degradation must be LOCALISED AT THE CORRIDOR
TERMINALS.  Interior buses have no such competition and should be unaffected or
better under Thevenin, whose H is more faithful there too.

Test: split each zone's monitored EHV buses into corridor terminals (the
in-zone endpoints of its tie lines) and interior buses, and compare the RMS
voltage error of each group across the 007h frontier arms.

Prediction if the hypothesis holds
----------------------------------
    corridor:  Thevenin clearly worse than PQ
    interior:  Thevenin equal or better

Prediction if it fails: the degradation is spread evenly, and the mechanism is
something else again.

Reads the ``007h`` pickles; runs the production setup once only to recover the
zone definitions (which are not stored in the records).

Author: Manuel Schwenke / Claude Code
Date: 2026-08-12
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]

FRONTIER_DIR = _ROOT / "results" / "007_tie_boundary" / "frontier"
V_SET = 1.03


def _zone_bus_split() -> Dict[int, Dict[str, Set[int]]]:
    """{zone: {"corridor": {...}, "interior": {...}}} over monitored EHV buses.

    Corridor terminals are the IN-ZONE endpoints of the zone's tie lines --
    the buses each area tracks and which its neighbour also tracks from the
    other side.
    """
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    st: Dict[str, Any] = {}
    run_multi_tso_dso(cfg, pre_loop_hook=lambda s: (st.update(s), True)[1])

    out: Dict[int, Dict[str, Set[int]]] = {}
    for z, zd in st["zone_defs"].items():
        monitored = {int(b) for b in zd.v_bus_indices}
        corridor = {int(b) for b in (zd.tie_line_endpoint_buses or [])} & monitored
        out[int(z)] = {"corridor": corridor, "interior": monitored - corridor}
    return out


def _rms_group(recs, buses: Set[int]) -> float:
    """Time-mean of the spatial RMS voltage error over *buses*."""
    if not buses:
        return float("nan")
    per_step: List[float] = []
    for r in recs:
        vm = getattr(r, "bus_vm_pu", None) or {}
        vals = [float(vm[b]) for b in buses
                if b in vm and np.isfinite(vm[b])]
        if vals:
            d = np.asarray(vals) - V_SET
            per_step.append(float(np.sqrt(np.mean(d ** 2))))
    return float(np.mean(per_step)) if per_step else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0)
    args = ap.parse_args()

    print("=" * 84)
    print("CORRIDOR TERMINALS vs INTERIOR BUSES -- is the Thevenin loss "
          "localised at the shared boundary?")
    print("=" * 84)

    split = _zone_bus_split()
    for z in sorted(split):
        print(f"  zone {z}: {len(split[z]['corridor'])} corridor terminal(s) "
              f"{sorted(split[z]['corridor'])}, "
              f"{len(split[z]['interior'])} interior")

    arms = [f"{b}_k{k:g}" for b in ("pq", "th") for k in (1.0, 0.5, 0.3, 0.2)]
    logs: Dict[str, Any] = {}
    for arm in arms:
        p = FRONTIER_DIR / f"{arm}_{args.hours:g}h.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                logs[arm] = pickle.load(fh)
    if not logs:
        print("\nno frontier pickles found -- run 007h first")
        return 1

    all_corr = set().union(*(split[z]["corridor"] for z in split))
    all_int = set().union(*(split[z]["interior"] for z in split))

    print("\nSystem-wide, over all zones:")
    hdr = f"{'arm':>9} {'corridor':>10} {'interior':>10} {'corr/int':>10}"
    print(hdr)
    print("-" * len(hdr))
    res: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        if arm not in logs:
            continue
        c = _rms_group(logs[arm], all_corr)
        i = _rms_group(logs[arm], all_int)
        res[arm] = {"corridor": c, "interior": i}
        print(f"{arm:>9} {c:>10.5f} {i:>10.5f} {c / i:>10.3f}")

    print("\nThevenin vs PQ at equal kappa:")
    hdr2 = f"{'kappa':>6} {'corridor':>12} {'interior':>12}"
    print(hdr2)
    print("-" * len(hdr2))
    for k in (1.0, 0.5, 0.3, 0.2):
        a, b = f"pq_k{k:g}", f"th_k{k:g}"
        if a not in res or b not in res:
            continue
        dc = (res[b]["corridor"] - res[a]["corridor"]) / res[a]["corridor"] * 100
        di = (res[b]["interior"] - res[a]["interior"]) / res[a]["interior"] * 100
        print(f"{k:>6.2f} {dc:>11.1f}% {di:>11.1f}%")

    print("\nPer zone, Thevenin vs PQ at equal kappa "
          "(corridor % / interior %):")
    hdr3 = f"{'kappa':>6}" + "".join(f"{'zone ' + str(z):>20}" for z in sorted(split))
    print(hdr3)
    print("-" * len(hdr3))
    for k in (1.0, 0.5, 0.3, 0.2):
        a, b = f"pq_k{k:g}", f"th_k{k:g}"
        if a not in logs or b not in logs:
            continue
        cells = []
        for z in sorted(split):
            ca = _rms_group(logs[a], split[z]["corridor"])
            cb = _rms_group(logs[b], split[z]["corridor"])
            ia = _rms_group(logs[a], split[z]["interior"])
            ib = _rms_group(logs[b], split[z]["interior"])
            dc = (cb - ca) / ca * 100 if np.isfinite(ca) and ca > 0 else np.nan
            di = (ib - ia) / ia * 100 if np.isfinite(ia) and ia > 0 else np.nan
            cells.append(f"{dc:>9.1f}% /{di:>7.1f}%")
        print(f"{k:>6.2f}" + "".join(f"{c:>20}" for c in cells))

    print("\nVERDICT")
    ks = [k for k in (1.0, 0.5, 0.3, 0.2)
          if f"pq_k{k:g}" in res and f"th_k{k:g}" in res]
    if ks:
        dcs = [(res[f'th_k{k:g}']['corridor'] - res[f'pq_k{k:g}']['corridor'])
               / res[f'pq_k{k:g}']['corridor'] * 100 for k in ks]
        dis = [(res[f'th_k{k:g}']['interior'] - res[f'pq_k{k:g}']['interior'])
               / res[f'pq_k{k:g}']['interior'] * 100 for k in ks]
        print(f"  mean Thevenin penalty at corridor terminals: "
              f"{np.mean(dcs):+.1f} %")
        print(f"  mean Thevenin penalty at interior buses    : "
              f"{np.mean(dis):+.1f} %")
        if np.mean(dcs) > np.mean(dis) + 3.0:
            print("  -> LOCALISED at the shared boundary: consistent with "
                  "shared-boundary under-provision.")
        elif abs(np.mean(dcs) - np.mean(dis)) <= 3.0:
            print("  -> SPREAD EVENLY: hypothesis NOT supported; the loss is "
                  "not specific to the shared boundary.")
        else:
            print("  -> WORSE IN THE INTERIOR: hypothesis refuted, and the "
                  "opposite of what it predicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
