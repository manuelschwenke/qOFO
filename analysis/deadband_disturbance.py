#!/usr/bin/env python3
r"""Disturbance rejection: how well each dead band rejects an exogenous load step.

Companion to :mod:`analysis.deadband_selection`, which measures the same three
controlled quantities under undisturbed profiled operation. The two studies
share a results root and are separated by their admission filters: this one
admits ONLY runs carrying a load step, that one only runs without.

The expectation under test is the opposite of the profiled-operation result: a
NARROWER dead band should reject a step better, because the local Q(V) droop
begins responding at a smaller voltage deviation and therefore acts sooner. In
profiled operation a WIDER band tracked the interface better at stressed
operating points, because the droop stopped competing with the outer
optimisation between dispatches. If both hold, delta also sets how much of a
disturbance the local layer absorbs before the slower cascade responds.

Method
------
Every stepped run is paired with the undisturbed run at the SAME window and the
SAME dead band. Pairing rather than differencing pre- and post-step windows
within one run matters here: only 4 samples precede the step, and the profile is
still evolving underneath it, so the undisturbed twin is the honest baseline.

Reported per (window, delta, factor):

    ifq_post        mean |Q_act - Q_set| over post-step samples      [Mvar]
    ifq_peak        max  |Q_act - Q_set| over post-step samples      [Mvar]
    ifq_excess      ifq_post minus the undisturbed twin's ifq_post   [Mvar]
                    -- the degradation attributable to the step
    v_excess        the same for TS and DS voltage                   [pu]
    recovery        samples until |Q_err| falls back within the
                    undisturbed twin's post-step mean (None = never)

``ifq_excess`` is the headline: it isolates the disturbance response from the
operating point's own tracking difficulty, which varies by an order of magnitude
across windows.

Usage::

    python -m analysis.deadband_disturbance
    python -m analysis.deadband_disturbance --no-figures

Author: Manuel Schwenke / Claude Code (2026-08-01)
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.deadband_selection import (  # noqa: E402
    ADMIT, V_REF, undisturbed_topology, uniform_deadband,
)

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rms_phase6_replay"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "deadband_selection"

#: Keys shared with the undisturbed study. The load-step keys are handled
#: separately: here a step is REQUIRED, there it must be absent.
_SHARED = {k: v for k, v in ADMIT.items() if k != "load_step_time_s"}


def _cfg(run_dir: Path) -> Optional[dict]:
    try:
        raw = json.loads((run_dir / "config.json").read_text(
            encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return raw.get("runner_static") or None


def _matches_shared(cfg: dict, scenario: str) -> bool:
    if str(cfg.get("scenario", "")) != scenario:
        return False
    # One dead band for both levels (see deadband_selection.uniform_deadband).
    if not uniform_deadband(cfg):
        return False
    # This study's disturbance is a load step on an intact network. An N-1 run
    # carries a load step of None and would otherwise enter as a baseline.
    if not undisturbed_topology(cfg):
        return False
    # This study's disturbance is the UNIFORM multiplicative step, keyed by
    # load_step_factor. The 2D study steps a single bus additively and leaves
    # the factor at its 1.0 default, so its runs would enter here as "factor
    # 1.0" cells -- a disturbance-rejection curve built from a different
    # disturbance.
    if cfg.get("load_step_bus") is not None:
        return False
    for key, want in _SHARED.items():
        got = cfg.get(key)
        if want is None:
            if got is not None:
                return False
        elif isinstance(want, dict):
            if not isinstance(got, dict):
                return False
            try:
                if {str(k): float(v) for k, v in got.items()} != want:
                    return False
            except (TypeError, ValueError):
                return False
        elif isinstance(want, float):
            try:
                if abs(float(got) - want) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif bool(got) != bool(want):
            return False
    return True


def _series(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Per-sample time series of the three controlled quantities."""
    pkl = run_dir / "rms_records.pkl"
    if not pkl.exists():
        return None
    with pkl.open("rb") as fh:
        recs = pickle.load(fh)
    if not recs:
        return None
    t, q, tsv, dsv = [], [], [], []
    for r in recs:
        t.append(float(getattr(r, "time_s", np.nan)))
        want = dict(r.dso_trafo_q_set_mvar)
        got = dict(r.dso_trafo_q_actual_mvar)
        errs = [abs(got[k] - want[k]) for k in want if k in got]
        q.append(float(np.mean(errs)) if errs else np.nan)
        zone = getattr(r, "zone_v_rms_err_pu", None)
        tsv.append(float(np.mean([abs(v) for v in zone.values()]))
                   if isinstance(zone, dict) and zone else np.nan)
        grp = dict(r.dso_group_v_mean_pu)
        dsv.append(float(np.sqrt(np.mean([(v - V_REF) ** 2
                                          for v in grp.values()])))
                   if grp else np.nan)
    return {"t": np.asarray(t), "ifq": np.asarray(q),
            "tsv": np.asarray(tsv), "dsv": np.asarray(dsv)}


def collect(results_root: Path, scenario: str, verbose: bool = True):
    """Return ``(stepped, baseline)`` keyed for pairing."""
    stepped: Dict[Tuple[str, float, float], dict] = {}
    baseline: Dict[Tuple[str, float], dict] = {}
    n_skip = 0
    for run_dir in sorted(results_root.glob("0*")):
        cfg = _cfg(run_dir)
        if not cfg or not _matches_shared(cfg, scenario):
            n_skip += 1
            continue
        try:
            delta = float(cfg.get("tso_qv_deadband_pu"))
        except (TypeError, ValueError):
            n_skip += 1
            continue
        window = str(cfg.get("start_time", ""))[:16]
        s = _series(run_dir)
        if s is None:
            continue
        step_t = cfg.get("load_step_time_s")
        rec = {"run": run_dir.name[:4], "series": s,
               "step_t": None if step_t is None else float(step_t)}
        if step_t is None:
            baseline[(window, delta)] = rec
        else:
            factor = float(cfg.get("load_step_factor", 1.0))
            stepped[(window, delta, factor)] = rec
    if verbose:
        print(f"[admit] scenario={scenario!r}: {len(stepped)} stepped run(s), "
              f"{len(baseline)} undisturbed twin(s), {n_skip} skipped")
    return stepped, baseline


def evaluate(stepped, baseline) -> List[dict]:
    rows = []
    for (window, delta, factor), rec in sorted(stepped.items()):
        s = rec["series"]
        post = s["t"] >= rec["step_t"]
        if not post.any():
            continue
        twin = baseline.get((window, delta))
        row = {
            "window": window, "delta_pu": delta, "factor": factor,
            "run": rec["run"],
            "twin": twin["run"] if twin else "",
            "ifq_post": float(np.nanmean(s["ifq"][post])),
            "ifq_peak": float(np.nanmax(s["ifq"][post])),
            "tsv_post": float(np.nanmean(s["tsv"][post])),
            "dsv_post": float(np.nanmean(s["dsv"][post])),
        }
        if twin is not None:
            b = twin["series"]
            bpost = b["t"] >= rec["step_t"]
            base_ifq = float(np.nanmean(b["ifq"][bpost]))
            row["ifq_excess"] = row["ifq_post"] - base_ifq
            row["tsv_excess"] = row["tsv_post"] - float(np.nanmean(b["tsv"][bpost]))
            row["dsv_excess"] = row["dsv_post"] - float(np.nanmean(b["dsv"][bpost]))

            # PEAK excess is the primary rejection measure. The mean-based
            # `ifq_excess` above conflates two different things: the transient
            # the step provokes, and the fact that the post-step operating
            # point is genuinely different (more load), where the steady-state
            # tracking error may legitimately be lower. It comes out NEGATIVE
            # at delta = 0, i.e. the disturbance appears to improve tracking --
            # consistent with the delta = 0 pathology being chatter that the
            # step displaces the system out of, not a loss of authority.
            #
            # Taking the per-sample difference first, then its maximum, removes
            # the shared baseline: it is how much worse this run is than the
            # same run without the step, at the worst instant.
            n = min(post.sum(), bpost.sum())
            if n:
                d = s["ifq"][post][:n] - b["ifq"][bpost][:n]
                row["ifq_peak_excess"] = float(np.nanmax(d))
                row["ifq_peak_ratio"] = (
                    float(np.nanmax(s["ifq"][post][:n]))
                    / float(np.nanmax(b["ifq"][bpost][:n]))
                    if np.nanmax(b["ifq"][bpost][:n]) > 0 else float("nan"))
            else:
                row["ifq_peak_excess"] = float("nan")
                row["ifq_peak_ratio"] = float("nan")

            # First post-step sample at or below the undisturbed level.
            back = np.where((s["t"] >= rec["step_t"]) & (s["ifq"] <= base_ifq))[0]
            row["recovery_samples"] = (int(back[0] - np.argmax(post))
                                       if back.size else None)
        else:
            for k in ("ifq_excess", "tsv_excess", "dsv_excess",
                      "ifq_peak_excess", "ifq_peak_ratio"):
                row[k] = float("nan")
            row["recovery_samples"] = None
        rows.append(row)
    return rows


def report(rows: List[dict]) -> None:
    if not rows:
        print("\nNo stepped runs yet. Run the disturbance phase first.")
        return
    by_wf: Dict[Tuple[str, float], List[dict]] = {}
    for r in rows:
        by_wf.setdefault((r["window"], r["factor"]), []).append(r)

    for (window, factor), group in sorted(by_wf.items()):
        print("\n" + "=" * 76)
        print(f"{window}   load step x{factor:g}")
        print(f"{'delta':>8} {'run':>5} {'twin':>5} {'ifQ peak':>10} "
              f"{'peak excess':>12} {'peak ratio':>11} {'mean excess':>12} "
              f"{'recov':>6}")
        for r in sorted(group, key=lambda x: x["delta_pu"]):
            rec = r["recovery_samples"]
            print(f"{r['delta_pu']:8.4f} {r['run']:>5} {r['twin']:>5} "
                  f"{r['ifq_peak']:10.3f} {r['ifq_peak_excess']:12.3f} "
                  f"{r['ifq_peak_ratio']:11.3f} {r['ifq_excess']:12.3f} "
                  f"{('-' if rec is None else rec):>6}")
        live = [r for r in group if np.isfinite(r.get("ifq_peak_excess",
                                                      np.nan))]
        if len(live) >= 2:
            best = min(live, key=lambda r: r["ifq_peak_excess"])
            worst = max(live, key=lambda r: r["ifq_peak_excess"])
            print(f"  best rejection (peak excess) : delta="
                  f"{best['delta_pu']:g}  ({best['ifq_peak_excess']:.3f} Mvar)")
            print(f"  worst rejection              : delta="
                  f"{worst['delta_pu']:g}  ({worst['ifq_peak_excess']:.3f} Mvar)")
            by_delta = sorted(live, key=lambda r: r["delta_pu"])
            vals = [r["ifq_peak_excess"] for r in by_delta]
            monotone = all(b >= a for a, b in zip(vals, vals[1:]))
            print(f"  peak excess rises monotonically with delta: {monotone}"
                  + ("  <- tighter band rejects better, as expected"
                     if monotone else "  <- expectation NOT met"))


def write_csv(rows: List[dict], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "deadband_disturbance.csv"
    cols = ["window", "factor", "delta_pu", "run", "twin", "ifq_post",
            "ifq_peak", "ifq_excess", "tsv_post", "tsv_excess", "dsv_post",
            "dsv_excess", "recovery_samples"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["window"], x["factor"],
                                             x["delta_pu"])):
            w.writerow(r)
    print(f"\n[csv] wrote {path}")
    return path


def figure(rows: List[dict], out_dir: Path) -> Optional[Path]:
    live = [r for r in rows if np.isfinite(r.get("ifq_excess", np.nan))]
    if len(live) < 2:
        print("[figure] need >=2 paired runs; skipped")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_wf: Dict[Tuple[str, float], List[dict]] = {}
    for r in live:
        by_wf.setdefault((r["window"], r["factor"]), []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    styles = ["o-", "s--", "^:", "v-."]
    for i, ((window, factor), group) in enumerate(sorted(by_wf.items())):
        g = sorted(group, key=lambda r: r["delta_pu"])
        ax.plot([r["delta_pu"] for r in g], [r["ifq_excess"] for r in g],
                styles[i % len(styles)],
                label=f"{window}  x{factor:g}")
    ax.axhline(0.0, color="0.5", lw=.8)
    ax.set_xlabel(r"dead-zone half-width $\delta$  [pu]")
    ax.set_ylabel("interface-Q excess over undisturbed twin  [Mvar]")
    ax.set_title("Load-step rejection versus dead band")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    png = out_dir / "deadband_step.png"
    fig.savefig(png, dpi=160)
    fig.savefig(out_dir / "deadband_step.pdf")
    plt.close(fig)
    print(f"[figure] wrote {png}")
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="rural_700")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    stepped, baseline = collect(args.results_root, args.scenario)
    rows = evaluate(stepped, baseline)
    report(rows)
    if rows:
        write_csv(rows, args.out)
        if not args.no_figures:
            figure(rows, args.out / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
