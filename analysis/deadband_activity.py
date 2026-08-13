#!/usr/bin/env python3
r"""Designs A and B: measure the mechanism, and normalise delta by what it filters.

The profiled-operation sweep cannot define delta* because the interface-Q metric
is not a well-defined function of delta at stressed operating points -- the
cascade settles into different equilibria and the argmin scatters. Two analyses
of the SAME stored runs address that without new simulation:

**A -- measure the mechanism directly.** The dead zone governs how much the
local droop moves. Actuator traverse and direction reversals count motion rather
than reading a possibly-bistable end state, so they are monotone in delta by
construction and immune to the equilibrium ambiguity that defeats the tracking
metric.

    traverse   sum over parks of sum_t |dQ|                    [Mvar]
    reversals  sign changes of dQ, i.e. chatter                [-]
    both also reported per park per dispatch interval

**B -- normalise delta by the voltage variability it filters.** A dead zone is
an amplitude-selective filter, so the meaningful abscissa is plausibly
delta / sigma_V rather than delta in pu. If delta* is constant in those units,
the scatter in absolute delta resolves; if it is not, that hypothesis dies
cheaply. sigma_V is measured per run from the DER terminal voltages.

Reads the per-park record ``csv/rms_der_raw.csv`` written by every run, and
admits only runs the dead-band study admits (see
``analysis.deadband_selection.ADMIT``), so rev-1 runs cannot enter.

Usage::

    python -m analysis.deadband_activity
    python -m analysis.deadband_activity --no-figures

Author: Manuel Schwenke / Claude Code (2026-08-02)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.deadband_selection import _admit, DEFAULT_RESULTS_ROOT  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "results" / "deadband_selection"

#: Net infeed [MW] on the current topology -- the physical ordering axis.
NET_INFEED: Dict[str, float] = {
    "2016-02-22T13:00": -117.4,
    "2016-01-05T08:00": 408.6,
    "2016-01-15T03:00": 805.0,
    "2016-12-18T14:00": 1367.1,
    "2016-05-01T16:00": 2200.0,
    "2016-07-15T03:00": -1026.1,
}
DEGENERATE = {"2016-07-15T03:00"}


def measure(run_dir: Path, dt_s: float = 20.0) -> Optional[dict]:
    """Actuator traverse, reversals and voltage variability for one run."""
    path = run_dir / "csv" / "rms_der_raw.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["signal", "time_s", "value"])
    except Exception:
        return None
    if df.empty:
        return None

    q = df[df["signal"].str.startswith("qDER_")]
    v = df[df["signal"].str.startswith("uDER_")]
    if q.empty:
        return None

    traverse = 0.0
    reversals = 0
    n_parks = 0
    for _sig, grp in q.groupby("signal", sort=False):
        s = grp.sort_values("time_s")["value"].to_numpy(dtype=float)
        if s.size < 3:
            continue
        d = np.diff(s)
        traverse += float(np.abs(d).sum())
        # Count sign changes, ignoring numerical dither below 1 kvar.
        sig = np.sign(d)
        sig[np.abs(d) < 1e-3] = 0.0
        nz = sig[sig != 0.0]
        if nz.size > 1:
            reversals += int((np.diff(nz) != 0).sum())
        n_parks += 1

    horizon = float(q["time_s"].max() - q["time_s"].min()) or 1.0
    n_int = max(horizon / dt_s, 1.0)

    sigma_v = np.nan
    p2p_v = np.nan
    if not v.empty:
        per = v.groupby("signal")["value"]
        sigma_v = float(per.std().mean())
        p2p_v = float((per.max() - per.min()).mean())

    return {
        "traverse_mvar": traverse,
        "traverse_per_park_interval": traverse / max(n_parks, 1) / n_int,
        "reversals": reversals,
        "reversals_per_park_interval": reversals / max(n_parks, 1) / n_int,
        "n_parks": n_parks,
        "sigma_v_pu": sigma_v,
        "p2p_v_pu": p2p_v,
    }


def collect(results_root: Path, scenario: str) -> List[dict]:
    rows: List[dict] = []
    for run_dir in sorted(results_root.glob("0*")):
        if not (run_dir / "rms_records.pkl").exists():
            continue
        adm = _admit(run_dir, scenario)
        if adm is None:
            continue
        window, delta = adm
        m = measure(run_dir)
        if m is None:
            continue
        m.update({"window": window, "delta_pu": delta,
                  "run": run_dir.name[:4],
                  "net_infeed_mw": NET_INFEED.get(window, float("nan"))})
        rows.append(m)
    return rows


def report(rows: List[dict]) -> None:
    live = [r for r in rows if r["window"] not in DEGENERATE]
    wins = sorted({r["window"] for r in live},
                  key=lambda w: NET_INFEED.get(w, 9e9))
    deltas = sorted({r["delta_pu"] for r in live})
    idx = {(r["window"], r["delta_pu"]): r for r in live}

    print("\n" + "=" * 78)
    print("A. ACTUATOR TRAVERSE  [Mvar per park per dispatch interval]")
    print("   the motion the dead zone exists to suppress")
    print("=" * 78)
    print(f"{'delta':>8}" + "".join(f"{w[:10]:>13}" for w in wins))
    print(f"{'':>8}" + "".join(f"{NET_INFEED[w]:>+13.0f}" for w in wins))
    for d in deltas:
        cells = []
        for w in wins:
            r = idx.get((w, d))
            cells.append(f"{r['traverse_per_park_interval']:13.4f}" if r
                         else f"{'--':>13}")
        print(f"{d:8g}" + "".join(cells))

    print("\n" + "=" * 78)
    print("A. DIRECTION REVERSALS  [per park per dispatch interval]")
    print("=" * 78)
    print(f"{'delta':>8}" + "".join(f"{w[:10]:>13}" for w in wins))
    for d in deltas:
        cells = []
        for w in wins:
            r = idx.get((w, d))
            cells.append(f"{r['reversals_per_park_interval']:13.3f}" if r
                         else f"{'--':>13}")
        print(f"{d:8g}" + "".join(cells))

    # monotonicity check -- the property the tracking metric lacks
    print("\n  monotone decreasing in delta?")
    for w in wins:
        seq = [idx[(w, d)]["traverse_per_park_interval"]
               for d in deltas if (w, d) in idx]
        mono = all(b <= a * 1.02 for a, b in zip(seq, seq[1:]))
        print(f"    {w}  {'YES' if mono else 'no '}   "
              f"{seq[0]:.4f} -> {seq[-1]:.4f}  "
              f"({seq[-1] / seq[0]:.2f}x)" if seq else "")

    print("\n" + "=" * 78)
    print("B. VOLTAGE VARIABILITY AND THE NORMALISED DEAD BAND")
    print("=" * 78)
    print(f"{'window':>18} {'net MW':>8} {'sigma_V [pu]':>13} "
          f"{'p2p_V [pu]':>11}")
    sig = {}
    for w in wins:
        vals = [idx[(w, d)]["sigma_v_pu"] for d in deltas if (w, d) in idx]
        p2p = [idx[(w, d)]["p2p_v_pu"] for d in deltas if (w, d) in idx]
        sig[w] = float(np.nanmean(vals)) if vals else np.nan
        print(f"{w:>18} {NET_INFEED[w]:>8.0f} {sig[w]:>13.5f} "
              f"{float(np.nanmean(p2p)) if p2p else np.nan:>11.5f}")
    return sig, wins, deltas, idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="rural_700")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    rows = collect(args.results_root, args.scenario)
    print(f"[activity] {len(rows)} run(s) measured")
    if not rows:
        print("no admitted runs with per-park records")
        return 1
    report(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "deadband_activity.csv"
    cols = ["window", "net_infeed_mw", "delta_pu", "run", "traverse_mvar",
            "traverse_per_park_interval", "reversals",
            "reversals_per_park_interval", "n_parks", "sigma_v_pu", "p2p_v_pu"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["net_infeed_mw"], r["delta_pu"])):
            w.writerow(r)
    print(f"\n[csv] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
