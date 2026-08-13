#!/usr/bin/env python3
r"""Pareto analysis of the dead-band grid: voltage quality vs interface-Q tracking.

The dead-band study establishes that the two controlled quantities select
opposite ends of the swept range -- DS voltage is best at delta = 0 in every
live window, interface Q is worst there. There is therefore no single optimal
delta, and the honest object is the Pareto front over the two objectives plus a
stated compromise rule.

Objectives (all minimised, all measured on the same runs):

    ifq   mean |Q_act - Q_set| over the TS-DSO interfaces        [Mvar]
    dsv   RMS deviation of each DSO group's mean V from V_ref    [pu]
    tsv   per-zone RMS voltage error                             [pu]

Two views are reported per window:

  * The non-dominated (Pareto) set. A dead band is dominated when another is at
    least as good in every objective and strictly better in one.
  * A compromise choice. Each objective is normalised to its own best value in
    that window, r_i(delta) = f_i(delta) / min_delta f_i(delta) >= 1, so the
    metrics become dimensionless and comparable. Two standard rules are then
    applied:
        Chebyshev  argmin_delta  max_i r_i(delta)   -- best worst-case ratio
        sum        argmin_delta  sum_i r_i(delta)   -- best average ratio
    Chebyshev is the more defensible default: it reports the dead band whose
    WORST relative sacrifice across the objectives is smallest, and it does not
    let one objective's large relative range dominate the sum.

Windows in which delta has no effect (see analysis/deadband_selection.py,
``_flat``) are excluded: every delta is trivially non-dominated there.

Usage::

    python -m analysis.deadband_pareto
    python -m analysis.deadband_pareto --objectives ifq dsv

Author: Manuel Schwenke / Claude Code (2026-07-31)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV = (PROJECT_ROOT / "results" / "deadband_selection"
               / "deadband_metrics.csv")

COLS = {"ifq": "ifq_mean_abs_err_mvar",
        "tsv": "ts_v_rms_err_pu",
        "dsv": "ds_v_rms_dev_pu"}
LABEL = {"ifq": "interface Q [Mvar]",
         "tsv": "TS V [pu]",
         "dsv": "DS V [pu]"}


def load(path: Path) -> Dict[str, Dict[float, Dict[str, float]]]:
    out: Dict[str, Dict[float, Dict[str, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            w = row["window"]
            d = float(row["delta_pu"])
            out.setdefault(w, {})[d] = {
                k: float(row[c]) for k, c in COLS.items()
            }
            out[w][d]["run"] = row["run"]
    return out


def dominated(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if b dominates a (b <= a everywhere, < somewhere)."""
    return all(y <= x for x, y in zip(a, b)) and any(y < x for x, y in zip(a, b))


def pareto(points: Dict[float, Sequence[float]]) -> List[float]:
    return sorted(d for d, p in points.items()
                  if not any(dominated(p, q) for e, q in points.items()
                             if e != d))


def analyse(window: str, rows: Dict[float, Dict[str, float]],
            objs: Sequence[str]) -> dict:
    deltas = sorted(rows)
    vals = {d: [rows[d][o] for o in objs] for d in deltas}

    # A window where nothing varies cannot be traded off.
    flat = all(
        max(rows[d][o] for d in deltas) - min(rows[d][o] for d in deltas)
        <= 1e-9 * max(1.0, max(abs(rows[d][o]) for d in deltas))
        for o in objs
    )
    if flat:
        return {"window": window, "flat": True}

    best = {o: min(rows[d][o] for d in deltas) for o in objs}
    ratios = {d: {o: (rows[d][o] / best[o] if best[o] > 0 else 1.0)
                  for o in objs} for d in deltas}
    front = pareto(vals)
    cheb = min(deltas, key=lambda d: max(ratios[d].values()))
    ssum = min(deltas, key=lambda d: sum(ratios[d].values()))
    return {"window": window, "flat": False, "deltas": deltas, "rows": rows,
            "ratios": ratios, "front": front, "cheb": cheb, "sum": ssum,
            "best": best}


def report(res: dict, objs: Sequence[str]) -> None:
    w = res["window"]
    print("\n" + "=" * 72)
    print(f"{w}")
    if res["flat"]:
        print("  delta has no effect in this window -- no trade-off exists; "
              "excluded.")
        return
    hdr = "  " + f"{'delta':>7} {'run':>5} " + "".join(
        f"{LABEL[o]:>20}" for o in objs) + f"{'max ratio':>11}  front"
    print(hdr)
    for d in res["deltas"]:
        r = res["rows"][d]
        cells = "".join(f"{r[o]:20.5f}" for o in objs)
        mark = "  *" if d in res["front"] else "   "
        note = ""
        if d == res["cheb"]:
            note += "  <- Chebyshev"
        if d == res["sum"]:
            note += "  <- sum"
        print(f"  {d:7.4f} {r['run']:>5} {cells}"
              f"{max(res['ratios'][d].values()):11.3f}{mark}{note}")
    print(f"\n  Pareto set: {', '.join(f'{d:g}' for d in res['front'])}")
    dom = [d for d in res["deltas"] if d not in res["front"]]
    print(f"  dominated : {', '.join(f'{d:g}' for d in dom) or 'none'}")
    c = res["cheb"]
    print(f"  compromise (Chebyshev): delta = {c:g}  "
          f"-- worst objective is {max(res['ratios'][c].values()):.2f}x its best")
    for o in objs:
        print(f"      {LABEL[o]:<20} {res['rows'][c][o]:.5f}  "
              f"({res['ratios'][c][o]:.2f}x best)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--objectives", nargs="+", default=["ifq", "dsv"],
                    choices=sorted(COLS), help="objectives to trade off "
                                               "(default: ifq dsv)")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"no metrics CSV at {args.csv}; run analysis.deadband_selection "
              "first.")
        return 1
    data = load(args.csv)
    objs = list(args.objectives)
    print(f"objectives: {', '.join(LABEL[o] for o in objs)}")
    print(f"source    : {args.csv}")

    results = [analyse(w, rows, objs) for w, rows in sorted(data.items())]
    for res in results:
        report(res, objs)

    live = [r for r in results if not r["flat"]]
    if len(live) > 1:
        print("\n" + "=" * 72)
        print("ACROSS WINDOWS")
        common = set(live[0]["front"])
        for r in live[1:]:
            common &= set(r["front"])
        print(f"  Pareto in EVERY live window: "
              f"{', '.join(f'{d:g}' for d in sorted(common)) or 'none'}")
        print(f"  Chebyshev compromise per window: "
              + ", ".join(f"{r['window'][:10]}={r['cheb']:g}" for r in live))
        print("\n  A dead band that is dominated in any window is a poor")
        print("  single fixed choice, since a better one exists there for both")
        print("  objectives at once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
