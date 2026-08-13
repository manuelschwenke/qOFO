#!/usr/bin/env python3
r"""List the (delta, gen) cells of a dead-band sweep that still need running.

A resume script must not re-run work that is already on disk -- at ~22 min and
~180 MB per run that is expensive in both time and, as of 2026-08-04, disk.

A cell counts as DONE only if its trace file exists AND is plausibly complete.
Size is the test: every good trace in this study is 25.7-27.6 MB, so a file
below ``--min-mb`` is a truncated write and the cell is reported as missing so
it gets redone. This matters because a partially written CSV still parses --
it would silently shorten a run's time series rather than failing loudly.

Cells are matched on (window, delta, gen) at a fixed droop and horizon, which
is exactly how ``analysis/deadband_n1.py`` keys them, so "missing here" means
"absent from the analysis" and nothing else.

Prints one ``<delta> <gen>`` pair per line for the requested window, where
gen -1 is the undisturbed twin -- a format a PowerShell caller can consume
directly.

Usage::

    python -m tools.missing_cells --droop 0.1 --window "2016-01-05 08:00"
    python -m tools.missing_cells --droop 0.1 --window ... --summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "rms_phase6_replay")

DEFAULT_DELTAS = ["0.0025", "0.005", "0.0075", "0.01",
                  "0.025", "0.05", "0.1", "0.5"]
DEFAULT_GENS = [-1, 1, 5]


def completed(droop: float, horizon: float, min_mb: float):
    """Set of (window, delta, gen) whose trace file is present and complete."""
    done = set()
    for d in glob.glob(os.path.join(RESULTS, "0*")):
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8",
                                 errors="ignore"))["runner_static"]
        except Exception:
            continue
        if not cfg:
            continue
        try:
            if abs(float(cfg.get("n_total_s", 0) or 0) - horizon) > 1e-9:
                continue
            slope = cfg.get("tso_qv_slope_pu")
            if slope is None or abs(float(slope) - droop) > 1e-9:
                continue
        except (TypeError, ValueError):
            continue
        csv = os.path.join(d, "csv", "rms_der_raw.csv")
        if not os.path.exists(csv):
            continue
        if os.path.getsize(csv) / 1e6 < min_mb:
            continue                      # truncated write -> treat as missing
        ctg = cfg.get("contingencies") or []
        gen = int(ctg[0]["element_index"]) if ctg else -1
        try:
            delta = float(cfg.get("tso_qv_deadband_pu"))
        except (TypeError, ValueError):
            continue
        done.add((str(cfg.get("start_time", ""))[:16], round(delta, 8), gen))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--droop", type=float, required=True)
    ap.add_argument("--window", required=True,
                    help="'YYYY-MM-DD HH:MM' or the ISO form used in configs")
    ap.add_argument("--deltas", nargs="*", default=DEFAULT_DELTAS)
    ap.add_argument("--gens", nargs="*", type=int, default=DEFAULT_GENS)
    ap.add_argument("--horizon", type=float, default=600.0)
    ap.add_argument("--min-mb", type=float, default=20.0)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    win = args.window.replace(" ", "T")[:16]
    done = completed(args.droop, args.horizon, args.min_mb)

    missing = [(d, g) for g in args.gens for d in args.deltas
               if (win, round(float(d), 8), g) not in done]

    if args.summary:
        total = len(args.deltas) * len(args.gens)
        print(f"window {win}  droop {args.droop:g}: "
              f"{total - len(missing)}/{total} done, {len(missing)} missing",
              file=sys.stderr)
    for d, g in missing:
        print(f"{d} {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
