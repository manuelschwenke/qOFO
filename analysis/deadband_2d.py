#!/usr/bin/env python3
r"""2D dead-band study: delta_TS x delta_DS under a localised TS load step.

Answers the question the 1D study could not: do the TS-connected and the
DS-connected DER populations want the SAME Q(V) dead-zone half-width?  The 1D
sweep varied one number applied to every park at both voltage levels, so a
difference between the levels was not representable -- and, before the
2026-08-02 plumbing fix, not even expressible in the RMS plant, whose Q(V)
pre-controllers were anchored from the exported snapshot's default dead band.

Two controlled quantities span the Pareto plane, both measured on the SAME
stepped runs over the post-step horizon and both referenced to the
same-``(delta_TS, delta_DS)`` undisturbed twin::

    interface Q     mean |Q_act - Q_set| over the TS-DSO interfaces   [Mvar]
    max |dV|        largest DER-terminal voltage deviation from the
                    twin, over parks and over post-step samples       [pu]

The second is the DESIGN PARAMETER: a limit on it ("no more than 0.01 pu") is
the specification, and the useful dead band is the one that buys the best
tracking subject to that limit.  It is reported for the TS and DS populations
separately, because a single aggregate would hide exactly the asymmetry this
study exists to measure.

Referencing to the twin, rather than to V_ref, is what isolates the step's
effect: the post-step operating point carries more load and its voltages are
genuinely lower, which is not a control failure.

Outputs (into ``results/deadband_2d/``)::

    deadband_2d_metrics.csv    one row per stepped run
    deadband_2d_pareto.csv     the non-dominated cells per amplitude

Usage::

    python -m analysis.deadband_2d
    python -m analysis.deadband_2d --scenario rural_700

Author: Manuel Schwenke / Claude Code (2026-08-02)
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rms_phase6_replay"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "deadband_2d"

#: Configuration a run must match to enter the study, compared against the
#: ``runner_static`` block of its own ``config.json``.  ``None`` means the key
#: must be absent or null.
ADMIT: Dict[str, object] = {
    "der_q_capability_override_pu": None,
    "use_profiles": True,
    "dso_qv_slope_pu": 0.06,
    "seed_der_anchor_to_local_v": False,
    "disable_qv_seed": False,
    "dso_der_scale": {"DSO_3": 2.0},
    "dso_load_p_scale": {"DSO_3": 2.0},
    "sensitivity_reduction_rev": 2.0,
    # The blanket scalar override pins EVERY park to one dead band and takes
    # precedence over the per-level values in pf.plant.  A run carrying it
    # cannot have delta_TS != delta_DS in the RMS plant no matter what its
    # tso_/dso_qv_deadband_pu fields say, so admitting one would put a
    # mislabelled diagonal cell into the matrix.  Every 1D-study run sets it.
    "der_qv_deadband_override_pu": None,
    # Simulation horizon.  Every metric here is a mean or a maximum over the
    # post-step window, so a run with a different horizon is not comparable --
    # and, worse, a short validation run would happily pair with a full-length
    # twin and produce a cell built from two different post-step lengths.  The
    # plumbing smoke runs (0277/0278, 160 s) are excluded by this.
    "n_total_s": 300.0,
}

#: Signals whose park is TS-connected.  The RMS trace names carry the level.
TS_MARK = "_WP_TSO_"


def _cfg(run_dir: Path) -> Optional[dict]:
    try:
        raw = json.loads((run_dir / "config.json").read_text(
            encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    cfg = raw.get("runner_static")
    return cfg if cfg else None


def _matches(cfg: dict, scenario: str) -> bool:
    if str(cfg.get("scenario", "")) != scenario:
        return False
    for key, want in ADMIT.items():
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


def _gate_e(run_dir: Path) -> str:
    """The run's Gate E verdict, recorded but NOT used to filter.

    Gate E validates QSS/RMS equivalence.  A chattering plant legitimately
    breaks that equivalence, so the whole ``delta = 0`` row and column are
    expected to read FAIL_SETTLING irrespective of the disturbance (measured:
    the pre-step DER traverse at ``delta_DS = 0`` is 313 Mvar/park/interval
    against 1.07 at ``delta_DS = 0.02``, i.e. the chatter is present before the
    step and the step reduces it).  Those cells remain valid *within* the RMS
    comparison this study makes, so the verdict is carried alongside the metrics
    rather than used to reject runs -- but a column of FAILs must not be read as
    a broken sweep.
    """
    p = run_dir / "gate_e_summary.md"
    if not p.exists():
        return ""
    m = re.search(r"verdict: \*\*([A-Z_]+)\*\*",
                  p.read_text(encoding="utf-8", errors="ignore"))
    return m.group(1) if m else ""


def _ifq_series(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Per-sample mean interface-Q tracking error [Mvar]."""
    pkl = run_dir / "rms_records.pkl"
    if not pkl.exists():
        return None
    with pkl.open("rb") as fh:
        recs = pickle.load(fh)
    if not recs:
        return None
    t, q = [], []
    for r in recs:
        t.append(float(getattr(r, "time_s", np.nan)))
        want = dict(r.dso_trafo_q_set_mvar)
        got = dict(r.dso_trafo_q_actual_mvar)
        errs = [abs(got[k] - want[k]) for k in want if k in got]
        q.append(float(np.mean(errs)) if errs else np.nan)
    return {"t": np.asarray(t), "ifq": np.asarray(q)}


def _der_voltages(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """DER terminal voltage traces, keyed by signal name."""
    path = run_dir / "csv" / "rms_der_raw.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["signal", "time_s", "value"])
    df = df[df["signal"].str.startswith("uDER_")]
    return {s: g.sort_values("time_s") for s, g in df.groupby("signal")}


def _max_dv(run_dir: Path, twin_dir: Path, step_t: float, onset: float
            ) -> Dict[str, float]:
    """Largest |V - V_twin| at the TS and at the DS parks, post- and pre-step.

    Maximum over parks AND over samples: the design limit is a limit on the
    worst park at the worst instant, not on an average that a single badly
    placed park can hide beneath.

    The PRE window is a validity check, not a result.  A stepped run and its
    twin are identical before the disturbance, so any difference there is
    chatter the two runs do not share; where it approaches the post-step figure,
    the post-step number is noise rather than step response.

    That window ends at ``onset``, NOT at ``step_t``.  The disturbance is a
    RAMP: the profile frame has ``dt_s`` rows and the RMS ElmFile playback
    interpolates between them, so the load starts moving one dispatch interval
    early (measured 2026-08-02: first activity at t = 80.5 s for a step
    configured at t = 100 s).  Ending the quiet window at ``step_t`` captures
    the ramp and reports the disturbance itself as a chatter floor -- which is
    exactly how a quiet run was misread as chattering at 300 Mvar per park per
    interval.
    """
    run_v = _der_voltages(run_dir)
    twin_v = _der_voltages(twin_dir)
    worst = {"TS": 0.0, "DS": 0.0, "TS_pre": 0.0, "DS_pre": 0.0}
    seen = {"TS": False, "DS": False}
    for sig, g in run_v.items():
        tw = twin_v.get(sig)
        if tw is None:
            continue
        t = g["time_s"].to_numpy(float)
        post = t >= step_t
        if not post.any():
            continue
        ref = np.interp(t, tw["time_s"].to_numpy(float),
                        tw["value"].to_numpy(float))
        d = np.abs(g["value"].to_numpy(float) - ref)
        if not np.isfinite(d).any():
            continue
        grp = "TS" if TS_MARK in sig else "DS"
        seen[grp] = True
        worst[grp] = max(worst[grp], float(np.nanmax(d[post])))
        pre = t < onset
        if pre.any() and np.isfinite(d[pre]).any():
            worst[grp + "_pre"] = max(worst[grp + "_pre"],
                                      float(np.nanmax(d[pre])))
    return {
        "dv_max_ts_pu": worst["TS"] if seen["TS"] else float("nan"),
        "dv_max_ds_pu": worst["DS"] if seen["DS"] else float("nan"),
        "dv_pre_ts_pu": worst["TS_pre"] if seen["TS"] else float("nan"),
        "dv_pre_ds_pu": worst["DS_pre"] if seen["DS"] else float("nan"),
    }


def collect(results_root: Path, scenario: str, verbose: bool = True):
    """Return ``(stepped, twins)`` keyed on the dead-band pair."""
    stepped: Dict[Tuple[float, float, float], dict] = {}
    twins: Dict[Tuple[float, float], dict] = {}
    n_skip = 0
    for run_dir in sorted(results_root.glob("0*")):
        cfg = _cfg(run_dir)
        if not cfg or not _matches(cfg, scenario):
            n_skip += 1
            continue
        try:
            d_ts = float(cfg.get("tso_qv_deadband_pu"))
            d_ds = float(cfg.get("dso_qv_deadband_pu"))
        except (TypeError, ValueError):
            n_skip += 1
            continue
        s = _ifq_series(run_dir)
        if s is None:
            n_skip += 1
            continue
        rec = {"run": run_dir.name[:4], "dir": run_dir, "series": s,
               "gate_e": _gate_e(run_dir),
               "window": str(cfg.get("start_time", ""))[:16]}
        step_t = cfg.get("load_step_time_s")
        if step_t is None:
            twins[(d_ts, d_ds)] = rec
        else:
            rec["step_t"] = float(step_t)
            rec["dt_s"] = float(cfg.get("dt_s", 20.0) or 20.0)
            rec["bus"] = cfg.get("load_step_bus")
            amp = float(cfg.get("load_step_delta_mw", 0.0) or 0.0)
            stepped[(d_ts, d_ds, amp)] = rec
    if verbose:
        print(f"[admit] scenario={scenario!r}: {len(stepped)} stepped, "
              f"{len(twins)} twin(s), {n_skip} skipped")
    return stepped, twins


def evaluate(stepped, twins, verbose: bool = True) -> List[dict]:
    rows: List[dict] = []
    for (d_ts, d_ds, amp), rec in sorted(stepped.items()):
        twin = twins.get((d_ts, d_ds))
        if twin is None:
            if verbose:
                print(f"[pair] no twin for dTS={d_ts:g} dDS={d_ds:g} "
                      f"(run {rec['run']}) -- cell dropped")
            continue
        s, b = rec["series"], twin["series"]
        step_t = rec["step_t"]
        post, bpost = s["t"] >= step_t, b["t"] >= step_t
        if not post.any() or not bpost.any():
            continue
        n = min(int(post.sum()), int(bpost.sum()))
        dv = _max_dv(rec["dir"], twin["dir"], step_t,
                     step_t - float(rec.get("dt_s", 20.0)))
        row = {
            "delta_ts_pu": d_ts, "delta_ds_pu": d_ds, "step_mw": amp,
            "bus": rec.get("bus"), "window": rec["window"],
            "run": rec["run"], "twin": twin["run"],
            "gate_e": rec.get("gate_e", ""),
            "gate_e_twin": twin.get("gate_e", ""),
            "ifq_post_mvar": float(np.nanmean(s["ifq"][post])),
            "ifq_twin_mvar": float(np.nanmean(b["ifq"][bpost])),
            # Per-sample difference first, then the maximum: this removes the
            # shared baseline, so it is how much worse the stepped run is than
            # the same configuration without the step, at the worst instant.
            "ifq_peak_excess_mvar": float(np.nanmax(
                s["ifq"][post][:n] - b["ifq"][bpost][:n])) if n else np.nan,
        }
        row.update(dv)
        row["dv_max_pu"] = float(np.nanmax([dv["dv_max_ts_pu"],
                                            dv["dv_max_ds_pu"]]))
        row["dv_pre_max_pu"] = float(np.nanmax([dv["dv_pre_ts_pu"],
                                                dv["dv_pre_ds_pu"]]))
        # Signal-to-chatter: below ~2 the post-step deviation is not
        # distinguishable from the noise the two runs do not share.
        row["dv_snr"] = (row["dv_max_pu"] / row["dv_pre_max_pu"]
                         if row["dv_pre_max_pu"] > 0 else float("inf"))
        rows.append(row)
    return rows


def pareto(rows: List[dict], cost: str, design: str) -> List[dict]:
    """Non-dominated cells: minimise ``cost`` and ``design`` jointly.

    Reported per step amplitude -- a front mixing amplitudes would compare
    cells that were never subjected to the same disturbance.
    """
    out: List[dict] = []
    for amp in sorted({r["step_mw"] for r in rows}):
        grp = [r for r in rows if r["step_mw"] == amp
               and np.isfinite(r[cost]) and np.isfinite(r[design])]
        for r in grp:
            dominated = any(
                (o[cost] <= r[cost] and o[design] <= r[design])
                and (o[cost] < r[cost] or o[design] < r[design])
                for o in grp if o is not r)
            if not dominated:
                out.append(dict(r, step_mw=amp))
    return out


def write_csvs(rows, front, out_root: Path) -> Tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    cols = ["delta_ts_pu", "delta_ds_pu", "step_mw", "bus", "window", "run",
            "twin", "ifq_post_mvar", "ifq_twin_mvar", "ifq_peak_excess_mvar",
            "dv_max_ts_pu", "dv_max_ds_pu", "dv_max_pu",
            "dv_pre_ts_pu", "dv_pre_ds_pu", "dv_pre_max_pu", "dv_snr",
            "gate_e", "gate_e_twin"]
    m = out_root / "deadband_2d_metrics.csv"
    p = out_root / "deadband_2d_pareto.csv"
    for path, data in ((m, rows), (p, front)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(data, key=lambda x: (x["step_mw"],
                                                 x["delta_ts_pu"],
                                                 x["delta_ds_pu"])):
                w.writerow(r)
    return m, p


def print_tables(rows: List[dict]) -> None:
    """delta_TS x delta_DS cross-tabs, one per amplitude and metric."""
    for amp in sorted({r["step_mw"] for r in rows}):
        grp = [r for r in rows if r["step_mw"] == amp]
        ts = sorted({r["delta_ts_pu"] for r in grp})
        ds = sorted({r["delta_ds_pu"] for r in grp})
        for key, label, fmt in (
                ("ifq_post_mvar", "interface-Q error [Mvar]", "8.3f"),
                ("dv_max_ts_pu", "max |dV| at TS parks [pu]", "8.5f"),
                ("dv_max_ds_pu", "max |dV| at DS parks [pu]", "8.5f"),
        ):
            print(f"\n=== step +{amp:g} MW -- {label} "
                  f"(rows delta_TS, cols delta_DS) ===")
            print(f"{'':>8}" + "".join(f"{d:>9g}" for d in ds))
            for a in ts:
                cells = []
                for b in ds:
                    hit = [r for r in grp if r["delta_ts_pu"] == a
                           and r["delta_ds_pu"] == b]
                    cells.append(format(hit[0][key], fmt) if hit
                                 else f"{'--':>8}")
                print(f"{a:>8g}" + "".join(f" {c}" for c in cells))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--scenario", default="rural_700")
    ap.add_argument("--cost", default="ifq_post_mvar")
    ap.add_argument("--design", default="dv_max_pu")
    args = ap.parse_args(argv)

    stepped, twins = collect(args.results_root, args.scenario)
    rows = evaluate(stepped, twins)
    if not rows:
        print("no admitted (stepped, twin) pairs -- nothing to report")
        return 1
    front = pareto(rows, args.cost, args.design)
    print_tables(rows)
    print(f"\n=== Pareto front ({args.cost} vs {args.design}) ===")
    for r in sorted(front, key=lambda x: (x["step_mw"], x[args.cost])):
        print(f"  +{r['step_mw']:>6g} MW  dTS={r['delta_ts_pu']:<7g} "
              f"dDS={r['delta_ds_pu']:<7g}  {args.cost}={r[args.cost]:.4f}  "
              f"{args.design}={r[args.design]:.5f}  run {r['run']}")
    m, p = write_csvs(rows, front, args.out)
    print(f"\nwrote {m}\n      {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
