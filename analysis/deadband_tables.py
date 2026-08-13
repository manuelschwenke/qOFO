#!/usr/bin/env python3
"""Cross-tabulate the dead-band study: delta x window, one table per metric.

Reads ``results/deadband_selection/deadband_metrics.csv`` (written by
``analysis.deadband_selection``) and prints:

    1. interface-Q tracking error       [Mvar]
    2. TS zone voltage error            [pu]
    3. DS group voltage deviation       [pu]
    4. the combined Pareto evaluation

Windows are ordered by net infeed, which is the physical axis the study varies;
the old-topology screening figure is deliberately not used for ordering.
Per-window minima are marked ``*``, and an argmin sitting at either end of the
swept range is marked ``e`` because it is a bound rather than a measured
optimum.

Usage::

    python -m analysis.deadband_tables
    python -m analysis.deadband_tables --markdown

Author: Manuel Schwenke / Claude Code (2026-08-02)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV = (PROJECT_ROOT / "results" / "deadband_selection"
               / "deadband_metrics.csv")

#: Net infeed [MW] measured on the current topology -- the ordering axis.
NET_INFEED: Dict[str, float] = {
    "2016-02-22T13:00": -117.4,
    "2016-01-05T08:00": 408.6,
    "2016-01-15T03:00": 805.0,
    "2016-12-18T14:00": 1367.1,
    "2016-05-01T16:00": 2200.0,
    "2016-07-15T03:00": -1026.1,   # degenerate: zero DER capability
}
DEGENERATE = {"2016-07-15T03:00"}

METRICS = (
    ("ifq_mean_abs_err_mvar", "interface-Q tracking error", "Mvar", "{:8.3f}"),
    ("ts_v_rms_err_pu", "TS zone voltage error", "pu", "{:8.5f}"),
    ("ds_v_rms_dev_pu", "DS group voltage deviation", "pu", "{:8.5f}"),
)


def load(path: Path):
    data: Dict[str, Dict[float, Dict[str, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            w, d = row["window"], float(row["delta_pu"])
            data.setdefault(w, {})[d] = {
                k: float(row[k]) for k, *_ in METRICS
            }
    return data


def windows_sorted(data) -> List[str]:
    live = [w for w in data if w not in DEGENERATE]
    return sorted(live, key=lambda w: NET_INFEED.get(w, 9e9))


def deltas_sorted(data, wins) -> List[float]:
    ds = set()
    for w in wins:
        ds.update(data[w])
    return sorted(ds)


def table(data, wins, deltas, key, title, unit, fmt, markdown: bool) -> None:
    print(f"\n### {title}  [{unit}]\n")
    hdr = ["delta"] + [f"{w[:10]} {NET_INFEED.get(w, 0):+.0f}" for w in wins]
    if markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
    else:
        print("  " + "".join(f"{h:>22}" if i else f"{h:>8}"
                             for i, h in enumerate(hdr)))

    best = {}
    for w in wins:
        vals = {d: data[w][d][key] for d in data[w]}
        if vals:
            best[w] = min(vals, key=vals.get)

    for d in deltas:
        cells = []
        for w in wins:
            if d not in data[w]:
                cells.append("--")
                continue
            v = data[w][d][key]
            mark = ""
            if best.get(w) == d:
                ds_w = sorted(data[w])
                mark = "*e" if d in (ds_w[0], ds_w[-1]) else "* "
            cells.append(fmt.format(v) + mark)
        if markdown:
            print(f"| {d:g} | " + " | ".join(c.strip() for c in cells) + " |")
        else:
            print(f"  {d:8g}" + "".join(f"{c:>22}" for c in cells))
    if not markdown:
        print("    * = per-window minimum   e = at the edge of the swept "
              "range (a bound, not an optimum)")


def dominated(a, b) -> bool:
    return all(y <= x for x, y in zip(a, b)) and any(y < x for x, y in zip(a, b))


def pareto_table(data, wins, deltas, markdown: bool) -> None:
    objs = ["ifq_mean_abs_err_mvar", "ds_v_rms_dev_pu"]
    print(f"\n### Combined evaluation: Pareto over "
          f"(interface Q, DS voltage)\n")
    front: Dict[str, set] = {}
    cheb: Dict[str, Tuple[float, float]] = {}
    for w in wins:
        pts = {d: [data[w][d][o] for o in objs] for d in data[w]}
        front[w] = {d for d, p in pts.items()
                    if not any(dominated(p, q) for e, q in pts.items() if e != d)}
        bestv = [min(data[w][d][o] for d in data[w]) for o in objs]
        ratios = {d: max(data[w][d][o] / b if b > 0 else 1.0
                         for o, b in zip(objs, bestv)) for d in data[w]}
        c = min(ratios, key=ratios.get)
        cheb[w] = (c, ratios[c])

    hdr = ["delta"] + [w[:10] for w in wins] + ["in all?"]
    if markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
    for d in deltas:
        cells = []
        for w in wins:
            if d not in data[w]:
                cells.append("--")
            else:
                cells.append("PARETO" if d in front[w] else "dominated")
        every = all(d in front[w] for w in wins if d in data[w]) and \
            all(d in data[w] for w in wins)
        row = f"| {d:g} | " + " | ".join(cells) + f" | {'YES' if every else ''} |"
        print(row if markdown else "  " + row)

    print("\n  Chebyshev compromise per window "
          "(each objective normalised to its own best):")
    for w in wins:
        c, r = cheb[w]
        print(f"    {w}  net {NET_INFEED.get(w, 0):+7.0f} MW  "
              f"delta = {c:<7g} worst objective {r:.2f}x its best")
    common = set.intersection(*front.values()) if front else set()
    print(f"\n  Pareto-optimal in EVERY window: "
          f"{', '.join(f'{d:g}' for d in sorted(common)) or 'NONE'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"no metrics CSV at {args.csv}")
        return 1
    data = load(args.csv)
    wins = windows_sorted(data)
    deltas = deltas_sorted(data, wins)
    print(f"source: {args.csv}")
    print(f"windows ordered by net infeed; {len(wins)} live, "
          f"{len(DEGENERATE & set(data))} degenerate (excluded)")
    for key, title, unit, fmt in METRICS:
        table(data, wins, deltas, key, title, unit, fmt, args.markdown)
    pareto_table(data, wins, deltas, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
