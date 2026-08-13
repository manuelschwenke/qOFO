#!/usr/bin/env python3
r"""Design E readout: locate the dead-zone threshold from an amplitude sweep.

The dead zone is an amplitude threshold on the controller's input: the droop is
silent while |V - V_anchor| < delta and engages beyond it. Every earlier
analysis measured the *consequence* of that threshold through 300 s of
uncontrolled multi-frequency excitation and a cascade with multiple equilibria,
which is why the interface-Q argmin scatters (CV 0.715 over five windows).

This measures the threshold directly. At one operating point, a load step of
controlled amplitude is applied and the local droop's response recorded. The
predicted signature is a knee: negligible DER motion while the induced |dV|
stays inside the dead band, rising response once it does not.

**The abscissa is measured, not assumed.** A load step is an INDIRECT voltage
excitation -- the knob is the load factor, but the physical input to the dead
zone is the deviation seen at the DER terminals. Plotting against the step
factor would therefore confound the excitation with the network's response to
it, so |dV| is extracted per run:

    V_pre    mean terminal voltage over one dispatch interval before the step
    dV_peak  max |V(t) - V_pre| over the post-step window, per park

Response metrics are the Design A ones restricted to the post-step window --
they count actuator motion, so unlike a tracking error they cannot be
confounded by which equilibrium the run settles into:

    traverse   sum_t |dQ| per park            [Mvar]
    reversals  sign changes of dQ per park    [-]

The test is whether the knee position tracks delta with slope ~1. It does not
merely fit a curve: if the knee sits at |dV| ~ k*delta with k != 1 there is a
systematic offset to explain, and if there is no knee at all the local droop is
not the dominant path at this operating point.

Usage::

    python -m analysis.deadband_threshold
    python -m analysis.deadband_threshold --no-figures

Author: Manuel Schwenke / Claude Code (2026-08-02)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.deadband_selection import (  # noqa: E402
    undisturbed_topology, uniform_deadband,
)

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rms_phase6_replay"
DEFAULT_OUT = PROJECT_ROOT / "results" / "deadband_selection"

#: Post-step window used for both the excitation and the response [s].
POST_S = 60.0
DT_S = 20.0


def _cfg(run_dir: Path) -> Optional[dict]:
    try:
        raw = json.loads((run_dir / "config.json").read_text(
            encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return raw.get("runner_static") or None


def _terminal_voltages(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, usecols=["signal", "time_s", "value"])
    except Exception:
        return None
    v = df[df["signal"].str.startswith("uDER_")]
    return v if not v.empty else None


def measure(run_dir: Path, step_t: float,
            twin_dir: Optional[Path] = None) -> Optional[dict]:
    """Induced |dV| and post-step actuator response for one stepped run.

    The excitation is referenced against the UNDISTURBED TWIN at the same
    instant, not against this run's own pre-step voltage. Two reasons, both
    measured:

    * the profile keeps evolving after the step, so a pre-step reference
      accumulates ordinary drift -- at x1.01 that drift (0.00239 pu) is an order
      of magnitude larger than the step's own effect (0.00026 pu), which would
      invert the abscissa entirely;
    * the sub-second switching transient is not the excitation the dead zone
      responds to in steady operation. At x1.5 the peak deviation is 0.071 pu
      reached 0.5 s after the step, against a settled value of 0.0099 pu -- a
      factor of 7. Both are reported; the settled figure is the meaningful
      abscissa and the transient is kept for reference.
    """
    path = run_dir / "csv" / "rms_der_raw.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["signal", "time_s", "value"])
    except Exception:
        return None
    if df.empty:
        return None

    v = df[df["signal"].str.startswith("uDER_")]
    q = df[df["signal"].str.startswith("qDER_")]
    if v.empty or q.empty:
        return None

    # ---- excitation, referenced to the undisturbed twin ------------------
    dv_peak: List[float] = []
    dv_settled: List[float] = []
    twin_v = _terminal_voltages(twin_dir / "csv" / "rms_der_raw.csv") \
        if twin_dir else None
    twin_map = ({s: g.sort_values("time_s") for s, g in
                 twin_v.groupby("signal", sort=False)} if twin_v is not None
                else {})

    for sig, g in v.groupby("signal", sort=False):
        g = g.sort_values("time_s")
        t = g["time_s"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)
        post = (t >= step_t) & (t <= step_t + POST_S)
        if not post.any():
            continue
        tw = twin_map.get(sig)
        if tw is not None:
            ref = np.interp(t, tw["time_s"].to_numpy(dtype=float),
                            tw["value"].to_numpy(dtype=float))
            dev = np.abs(y - ref)
        else:                       # no twin: fall back to a pre-step baseline
            pre = (t >= step_t - DT_S) & (t < step_t)
            if not pre.any():
                continue
            dev = np.abs(y - float(np.nanmean(y[pre])))
        dv_peak.append(float(np.nanmax(dev[post])))
        # settled: the second half of the post-step window, past the transient
        late = (t >= step_t + POST_S / 2) & (t <= step_t + POST_S)
        if late.any():
            dv_settled.append(float(np.nanmean(dev[late])))
    if not dv_peak:
        return None

    # ---- response: actuator motion after the step ------------------------
    traverse = 0.0
    reversals = 0
    n_parks = 0
    for _sig, g in q.groupby("signal", sort=False):
        g = g.sort_values("time_s")
        t = g["time_s"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)
        m = (t >= step_t) & (t <= step_t + POST_S)
        s = y[m]
        if s.size < 3:
            continue
        d = np.diff(s)
        traverse += float(np.abs(d).sum())
        sig = np.sign(d)
        sig[np.abs(d) < 1e-3] = 0.0
        nz = sig[sig != 0.0]
        if nz.size > 1:
            reversals += int((np.diff(nz) != 0).sum())
        n_parks += 1

    n_int = max(POST_S / DT_S, 1.0)
    return {
        "dv_settled_pu": float(np.mean(dv_settled)) if dv_settled else np.nan,
        "dv_peak_mean_pu": float(np.mean(dv_peak)),
        "dv_peak_max_pu": float(np.max(dv_peak)),
        "traverse_mvar": traverse,
        "traverse_per_park_interval": traverse / max(n_parks, 1) / n_int,
        "reversals_per_park_interval": reversals / max(n_parks, 1) / n_int,
        "n_parks": n_parks,
    }


_TWIN_CACHE: Dict[tuple, Optional[Path]] = {}


def _twin_for(root: Path, scenario: str, window: str,
              delta: float) -> Optional[Path]:
    """The undisturbed run at the same window and dead band, if one exists."""
    key = (scenario, window, round(delta, 8))
    if key in _TWIN_CACHE:
        return _TWIN_CACHE[key]
    found = None
    for d in sorted(root.glob("0*")):
        if not (d / "csv" / "rms_der_raw.csv").exists():
            continue
        c = _cfg(d)
        if not c or c.get("load_step_time_s") is not None:
            continue
        if str(c.get("scenario", "")) != scenario:
            continue
        if str(c.get("start_time", ""))[:16] != window:
            continue
        if int(c.get("sensitivity_reduction_rev", 0) or 0) != 2:
            continue
        if not uniform_deadband(c) or not undisturbed_topology(c):
            continue        # off-diagonal or N-1: not this run's twin
        try:
            if abs(float(c.get("tso_qv_deadband_pu")) - delta) < 1e-9:
                found = d
        except (TypeError, ValueError):
            continue
    _TWIN_CACHE[key] = found
    return found


def collect(root: Path, scenario: str, window: Optional[str]) -> List[dict]:
    rows: List[dict] = []
    for run_dir in sorted(root.glob("0*")):
        if not (run_dir / "rms_records.pkl").exists():
            continue
        cfg = _cfg(run_dir)
        if not cfg or str(cfg.get("scenario", "")) != scenario:
            continue
        step_t = cfg.get("load_step_time_s")
        if step_t is None:
            continue                       # undisturbed run
        if cfg.get("der_q_capability_override_pu") is not None:
            continue                       # not the physical capability
        if int(cfg.get("sensitivity_reduction_rev", 0) or 0) != 2:
            continue                       # rev-1 sensitivities
        if not uniform_deadband(cfg):
            continue                       # off-diagonal: the 2D study's
        if not undisturbed_topology(cfg):
            continue                       # N-1: the contingency study's
        w = str(cfg.get("start_time", ""))[:16]
        if window and w != window:
            continue
        delta = float(cfg.get("tso_qv_deadband_pu"))
        m = measure(run_dir, float(step_t),
                    twin_dir=_twin_for(root, scenario, w, delta))
        if m is None:
            continue
        m.update({
            "window": w,
            "delta_pu": float(cfg.get("tso_qv_deadband_pu")),
            "factor": float(cfg.get("load_step_factor", 1.0)),
            "run": run_dir.name[:4],
        })
        rows.append(m)
    return rows


def report(rows: List[dict]) -> None:
    deltas = sorted({r["delta_pu"] for r in rows})
    factors = sorted({r["factor"] for r in rows})
    idx = {(r["delta_pu"], r["factor"]): r for r in rows}

    print("\n" + "=" * 78)
    print("EXCITATION: induced |dV| at the DER terminals  [pu]")
    print("  (does the amplitude range bracket the dead-band set?)")
    print("=" * 78)
    print(f"{'factor':>8}" + "".join(f"{f'd={d:g}':>12}" for d in deltas))
    for f in factors:
        cells = []
        for d in deltas:
            r = idx.get((d, f))
            cells.append(f"{r['dv_peak_mean_pu']:12.5f}" if r else f"{'--':>12}")
        print(f"{f:8.3f}" + "".join(cells))

    print("\n" + "=" * 78)
    print("RESPONSE: post-step DER traverse  [Mvar per park per interval]")
    print("=" * 78)
    print(f"{'factor':>8}" + "".join(f"{f'd={d:g}':>12}" for d in deltas))
    for f in factors:
        cells = []
        for d in deltas:
            r = idx.get((d, f))
            cells.append(f"{r['traverse_per_park_interval']:12.4f}" if r
                         else f"{'--':>12}")
        print(f"{f:8.3f}" + "".join(cells))

    print("\n" + "=" * 78)
    print("THRESHOLD: knee in response vs induced |dV|, per dead band")
    print("=" * 78)
    print(f"{'delta':>9} {'knee |dV|':>11} {'knee/delta':>11} "
          f"{'resp below':>11} {'resp above':>11}")
    for d in deltas:
        pts = sorted(((idx[(d, f)]["dv_peak_mean_pu"],
                       idx[(d, f)]["traverse_per_park_interval"])
                      for f in factors if (d, f) in idx))
        if len(pts) < 3:
            print(f"{d:9g}  too few points")
            continue
        dv = np.array([p[0] for p in pts])
        tr = np.array([p[1] for p in pts])
        # Knee: first |dV| at which the response exceeds 20% of its span above
        # the quiet floor (median of the two smallest excitations).
        floor = float(np.median(tr[:2]))
        span = float(tr.max() - floor)
        if span <= 0:
            print(f"{d:9g}  no rise -- response independent of amplitude")
            continue
        above = np.where(tr >= floor + 0.20 * span)[0]
        knee = float(dv[above[0]]) if above.size else float("nan")
        below_i = above[0] - 1 if above.size and above[0] > 0 else 0
        print(f"{d:9g} {knee:11.5f} {knee / d if d else np.nan:11.2f} "
              f"{tr[below_i]:11.4f} {tr[above[0]] if above.size else np.nan:11.4f}")
    print("\n  knee/delta ~ 1 => the dead zone behaves as specified.")
    print("  a consistent k != 1 => systematic offset (anchor lag, filtering).")
    print("  no knee => the local droop is not the dominant path here.")


def figure(rows: List[dict], out_dir: Path) -> Optional[Path]:
    if len(rows) < 4:
        print("[figure] too few runs; skipped")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = sorted({r["delta_pu"] for r in rows})
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    styles = ["o-", "s--", "^:", "v-.", "D-"]
    for i, d in enumerate(deltas):
        pts = sorted(((r["dv_peak_mean_pu"], r["traverse_per_park_interval"])
                      for r in rows if r["delta_pu"] == d))
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                styles[i % len(styles)], label=rf"$\delta$ = {d:g} pu")
        ax.axvline(d, color=f"C{i}", ls=":", lw=.8, alpha=.5)
    ax.set_xlabel(r"induced $|\Delta V|$ at the DER terminals  [pu]")
    ax.set_ylabel("post-step DER traverse  [Mvar / park / interval]")
    ax.set_title("Dead-zone threshold: response versus excitation amplitude\n"
                 "(dotted verticals mark each dead band)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    png = out_dir / "deadband_threshold.png"
    fig.savefig(png, dpi=160)
    fig.savefig(out_dir / "deadband_threshold.pdf")
    plt.close(fig)
    print(f"[figure] wrote {png}")
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="rural_700")
    ap.add_argument("--window", default="2016-01-05T08:00")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    rows = collect(args.results_root, args.scenario, args.window or None)
    print(f"[threshold] {len(rows)} stepped run(s) at {args.window}")
    if not rows:
        print("none yet -- run experiments/run_deadband_threshold.ps1")
        return 1
    report(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "deadband_threshold.csv"
    cols = ["window", "delta_pu", "factor", "run", "dv_settled_pu",
            "dv_peak_mean_pu", "dv_peak_max_pu", "traverse_mvar",
            "traverse_per_park_interval", "reversals_per_park_interval",
            "n_parks"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["delta_pu"], r["factor"])):
            w.writerow(r)
    print(f"\n[csv] wrote {path}")
    if not args.no_figures:
        figure(rows, args.out / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
