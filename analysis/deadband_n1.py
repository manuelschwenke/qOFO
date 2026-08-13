#!/usr/bin/env python3
r"""Dead-band selection under an N-1 outage: delta as a DETECTOR THRESHOLD.

The droop compares ``|V - V_anchor|`` against ``delta``, and ``V_anchor`` is
re-anchored every time the OFO writes that park's setpoint
(``core/plant.py``).  So ``delta`` does not discriminate voltage levels -- it
discriminates **drift since the last dispatch**, which makes it a detector
threshold separating ordinary operation from a real event.

That framing replaces the argmin the 1D study looked for and never found
(delta* CV 0.715 across windows).  There is no optimum because the two
populations overlap:

    normal drift   TS median 0.00087, max 0.0031   (over 180 s TSO windows)
                   DS median 0.00051, max 0.0111   (over  20 s DSO windows)
    N-1 excursion  gen 7 0.0104 ... gen 2 0.1025

``0.0111 > 0.0104``: the DS drift tail already exceeds the mildest credible
event, so no ``delta`` is simultaneously always-quiet and always-responsive.
The design is therefore a quantile trade-off, and this module reports the two
axes of it per voltage level:

    false activation   fraction of inter-dispatch windows in which the
                       UNDISTURBED twin's drift exceeds delta -- the droop
                       firing on ordinary operation
    missed detection   post-trip peak |dV| and the residual still present at
                       the next TSO dispatch -- what the slow layer inherits

plus the cost side (interface-Q tracking, actuator traverse).

Every deviation is referenced to the same-``delta`` undisturbed twin.  At the
mild outage the event excursion is the same order as ordinary profile drift, so
a within-run reference would count drift as rejection.

Usage::

    python -m analysis.deadband_n1
    python -m analysis.deadband_n1 --scenario rural_700

Author: Manuel Schwenke / Claude Code (2026-08-02)
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "rms_phase6_replay"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "deadband_n1"

TS_MARK = "_WP_TSO_"

#: Configuration a run must match, compared against ``runner_static``.
ADMIT: Dict[str, object] = {
    "der_q_capability_override_pu": None,
    "use_profiles": True,
    "dso_qv_slope_pu": 0.06,
    "seed_der_anchor_to_local_v": False,
    "disable_qv_seed": False,
    "dso_der_scale": {"DSO_3": 2.0},
    "dso_load_p_scale": {"DSO_3": 2.0},
    "sensitivity_reduction_rev": 2.0,
    "load_step_time_s": None,        # the disturbance here is a trip
    # The blanket scalar overrides every park and takes precedence in pf.plant,
    # so a run carrying it cannot express per-level dead bands.  This study
    # drives both levels through --tso-deadband/--dso-deadband so that stage 1
    # and stage 2 share one code path (the per-sgen map).
    "der_qv_deadband_override_pu": None,
    # Horizon: recovery is governed by the 180 s TSO period, so the metrics are
    # not comparable across horizons.  600 s gives two post-trip dispatches.
    "n_total_s": 600.0,
}


def _cfg(run_dir: Path) -> Optional[dict]:
    try:
        raw = json.loads((run_dir / "config.json").read_text(
            encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return raw.get("runner_static") or None


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


def _trip_of(cfg: dict) -> Tuple[Optional[int], Optional[float]]:
    """``(gen index, trip time)`` of the single generator outage, if any."""
    for e in (cfg.get("contingencies") or []):
        if str(e.get("element_type")) == "gen" and str(e.get("action")) == "trip":
            t = e.get("time_s")
            if t is None:
                t = float(e.get("minute", 0)) * 60.0
            return int(e.get("element_index")), float(t)
    return None, None


def _der_traces(run_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    path = run_dir / "csv" / "rms_der_raw.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["signal", "time_s", "value"])
    df = df[df["signal"].str.startswith(prefix)]
    return {s: g.sort_values("time_s") for s, g in df.groupby("signal")}


def _ifq(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
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


def drift_samples(twin_dir: Path, period_ts: float, period_ds: float,
                  t_start: float) -> Dict[str, np.ndarray]:
    """Max |V - V_anchor| within each inter-dispatch window, per level.

    The anchor is the park's voltage at the start of its own level's dispatch
    window, which is what the apply step writes.  ``t_start`` discards the
    initialisation transient.
    """
    out = {"TS": [], "DS": []}
    for sig, g in _der_traces(twin_dir, "uDER_").items():
        grp = "TS" if TS_MARK in sig else "DS"
        period = period_ts if grp == "TS" else period_ds
        t = g["time_s"].to_numpy(float)
        v = g["value"].to_numpy(float)
        m = t >= t_start
        t, v = t[m], v[m]
        if t.size < 3:
            continue
        w = np.floor(t / period).astype(int)
        for wi in np.unique(w):
            sel = w == wi
            if sel.sum() < 2:
                continue
            seg = v[sel]
            if not np.isfinite(seg).all():
                continue
            out[grp].append(float(np.nanmax(np.abs(seg - seg[0]))))
    return {g: np.asarray(v, dtype=float) for g, v in out.items()}


def false_activation(samples: Dict[str, np.ndarray], delta: float,
                     prefix: str) -> Dict[str, float]:
    """Fraction of inter-dispatch windows whose drift exceeds ``delta``.

    The rate at which the droop fires on ordinary operation -- the detector's
    false-alarm rate.

    Reported twice, because the two readings answer different questions and
    disagree by construction:

    ``fa_``       measured on this cell's OWN twin, i.e. CLOSED loop.  At narrow
                  delta the droop is active and suppresses the very drift being
                  measured, so this is the self-consistent rate that actually
                  occurs at that setting.
    ``faopen_``   measured on the WIDEST-delta twin, where the droop is silent,
                  so the drift is open loop.  This is the detector view: given
                  the natural drift distribution, how often would a threshold
                  delta fire?  It is the quantity a design rule is stated
                  against, since it does not depend on the setting being chosen.
    """
    res: Dict[str, float] = {}
    for grp in ("TS", "DS"):
        a = samples.get(grp)
        lo = grp.lower()
        if a is not None and a.size:
            res[f"{prefix}{lo}"] = float(np.mean(a > delta))
            res[f"{prefix}p90_{lo}_pu"] = float(np.percentile(a, 90))
            res[f"{prefix}max_{lo}_pu"] = float(a.max())
            res[f"{prefix}n_{lo}"] = int(a.size)
        else:
            res[f"{prefix}{lo}"] = float("nan")
            res[f"{prefix}p90_{lo}_pu"] = float("nan")
            res[f"{prefix}max_{lo}_pu"] = float("nan")
            res[f"{prefix}n_{lo}"] = 0
    return res


def rejection(run_dir: Path, twin_dir: Path, trip_t: float,
              next_dispatch_t: float, onset_t: float) -> Dict[str, float]:
    """Post-trip peak |dV| and the residual at the next TSO dispatch [pu].

    Referenced to the same-delta twin so ordinary profile drift is not counted
    as an event -- at the mild outage the two are the same order.

    The window opens at ``onset_t = trip_t - dt_s``, NOT at ``trip_t``.  In the
    RMS leg the ``EvtOutage`` is armed at ``plant.t + EPS``, and at the dispatch
    step labelled ``trip_t`` the plant has simulated only to ``trip_t - dt_s``;
    the outage therefore fires ~0.5 s into that interval (measured: 180.5 s for
    a trip configured at 200 s) so that it lands in the step's own
    measurements.  The electromechanical transient -- and hence the PEAK, which
    is the design parameter -- is over long before ``trip_t``.  Opening the
    window at ``trip_t`` would silently report the post-transient value as the
    peak.
    """
    run_v = _der_traces(run_dir, "uDER_")
    twin_v = _der_traces(twin_dir, "uDER_")
    peak = {"TS": [], "DS": []}
    resid = {"TS": [], "DS": []}
    for sig, g in run_v.items():
        tw = twin_v.get(sig)
        if tw is None:
            continue
        grp = "TS" if TS_MARK in sig else "DS"
        t = g["time_s"].to_numpy(float)
        ref = np.interp(t, tw["time_s"].to_numpy(float),
                        tw["value"].to_numpy(float))
        d = np.abs(g["value"].to_numpy(float) - ref)
        post = t >= onset_t
        if not post.any() or not np.isfinite(d[post]).any():
            continue
        peak[grp].append(float(np.nanmax(d[post])))
        # residual the slow layer inherits: the deviation still standing when
        # the next TSO dispatch finally acts
        j = np.searchsorted(t, next_dispatch_t)
        if 0 <= j < d.size and np.isfinite(d[j]):
            resid[grp].append(float(d[j]))
    out: Dict[str, float] = {}
    for grp in ("TS", "DS"):
        out[f"peak_dv_{grp.lower()}_pu"] = (
            float(np.max(peak[grp])) if peak[grp] else float("nan"))
        out[f"resid_dv_{grp.lower()}_pu"] = (
            float(np.max(resid[grp])) if resid[grp] else float("nan"))
    return out


def activity(run_dir: Path, t_lo: float, t_hi: float,
             dt_s: float) -> Dict[str, float]:
    """Actuator traverse per park per dispatch interval [Mvar]."""
    acc = {"TS": [], "DS": []}
    for sig, g in _der_traces(run_dir, "qDER_").items():
        grp = "TS" if TS_MARK in sig else "DS"
        t = g["time_s"].to_numpy(float)
        y = g["value"].to_numpy(float)
        m = (t >= t_lo) & (t <= t_hi)
        s = y[m]
        if s.size < 3:
            continue
        acc[grp].append(float(np.abs(np.diff(s)).sum()))
    span = max((t_hi - t_lo) / max(dt_s, 1e-9), 1.0)
    return {f"traverse_{g.lower()}": (float(np.mean(v)) / span if v
                                      else float("nan"))
            for g, v in acc.items()}


def collect(results_root: Path, scenario: str, verbose: bool = True):
    """Return ``(twins, trips)`` keyed by ``(window, delta[, gen])``.

    The window MUST be part of the key.  Keying on ``delta`` alone silently
    overwrites: a δ = 0.005 twin from 2016-12-18 would replace the one from
    2016-01-05 and every deviation at the first window would then be referenced
    against a run from a different operating point.  Drift and event severity
    are both window-dependent, so the whole study is per-window.
    """
    twins: Dict[Tuple[str, float], dict] = {}
    trips: Dict[Tuple[str, float, int], dict] = {}
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
        if abs(d_ts - d_ds) > 1e-12:
            n_skip += 1          # stage 2 (off-diagonal) is analysed separately
            continue
        gen, trip_t = _trip_of(cfg)
        win = str(cfg.get("start_time", ""))[:16]
        rec = {"run": run_dir.name[:4], "dir": run_dir, "delta": d_ts,
               "cfg": cfg, "gen": gen, "trip_t": trip_t, "window": win}
        key = (win, d_ts) if gen is None else (win, d_ts, gen)
        store = twins if gen is None else trips
        if key in store and verbose:
            # Last-writer-wins is fine for a genuine re-run but must never be
            # silent: this is how a run from a different configuration would
            # replace a study run with no trace in the output.
            print(f"[admit] duplicate cell {key}: run {rec['run']} supersedes "
                  f"{store[key]['run']}")
        store[key] = rec
    if verbose:
        wins = sorted({k[0] for k in list(twins) + list(trips)})
        print(f"[admit] scenario={scenario!r}: {len(twins)} twin(s), "
              f"{len(trips)} trip run(s), {n_skip} skipped; "
              f"{len(wins)} window(s): {', '.join(wins)}")
    return twins, trips


def open_loop_reference(twins, window: str, verbose: bool = True
                        ) -> Dict[str, np.ndarray]:
    """Pooled open-loop drift distribution for ONE operating window.

    Per window, never pooled across windows: profile-driven drift is a property
    of the operating point, so mixing windows would score every delta against a
    distribution none of them actually saw.
    #
    # Pooled over every twin that is PROVABLY open loop -- delta above that
    # twin's own observed drift maximum, so the dead zone was never crossed and
    # the droop never engaged.  Using only the widest twin would leave the TS
    # sample at 12 windows (4 parks x 3 TSO windows in 600 s), i.e. a
    # false-activation rate resolvable only to 1/12; pooling the wide twins
    # multiplies that by the number of them at no extra simulation cost.
    #
    # Admission is against the WIDEST twin's drift maximum, never against a
    # twin's own.  Comparing a twin's delta with its own observed drift is
    # self-fulfilling: a droop that successfully suppresses drift below its
    # threshold then looks inactive.  Measured 2026-08-03 -- the twins at
    # delta = 0.05/0.075/0.15 all report a drift max of 0.01101, while the
    # delta = 0.01 twin reports 0.00375; the latter is a SUPPRESSED residual,
    # yet 0.01 > 0.00375 would have admitted it and diluted the reference.
    """
    open_samples: Dict[str, np.ndarray] = {"TS": np.array([]),
                                           "DS": np.array([])}
    used: List[str] = []
    cache: Dict[float, Dict[str, np.ndarray]] = {}
    deltas = sorted(d for (w_, d) in twins if w_ == window)
    for d_val in sorted(deltas, reverse=True):
        w = twins[(window, d_val)]
        c = w["cfg"]
        cache[d_val] = drift_samples(
            w["dir"], float(c.get("tso_period_s", 180.0)),
            float(c.get("dso_period_s", 20.0)),
            t_start=float(c.get("tso_period_s", 180.0)))
    if cache:
        widest = max(cache)
        ref_max = max((a.max() for a in cache[widest].values() if a.size),
                      default=np.inf)
        for d_val in sorted(cache, reverse=True):
            if not np.isfinite(ref_max) or d_val <= ref_max:
                continue          # dead zone is crossed here: droop engages
            for g in ("TS", "DS"):
                if cache[d_val][g].size:
                    open_samples[g] = np.concatenate(
                        [open_samples[g], cache[d_val][g]])
            used.append(f"{twins[(window, d_val)]['run']}(d={d_val:g})")
        if verbose:
            print(f"[drift] {window}: admission threshold from widest twin "
                  f"(delta={widest:g}): drift max {ref_max:.5f} pu")
    if verbose:
        print(f"[drift] {window}: open-loop reference pooled over "
              f"{len(used)} twin(s) {' '.join(used) if used else '-- NONE'}: "
              f"TS n={open_samples['TS'].size} DS n={open_samples['DS'].size}")
        if not used:
            print(f"[drift] !! {window}: no twin is provably open loop; the "
                  "droop engaged even at the widest dead band, so no "
                  "false-activation rate can be stated")
    return open_samples


def evaluate(twins, trips, verbose: bool = True) -> List[dict]:
    rows: List[dict] = []
    windows = sorted({k[0] for k in trips})
    open_by_window = {w: open_loop_reference(twins, w, verbose)
                      for w in windows}
    for (window, delta, gen), rec in sorted(trips.items()):
        twin = twins.get((window, delta))
        if twin is None:
            if verbose:
                print(f"[pair] {window}: no twin for delta={delta:g} "
                      f"(run {rec['run']}) -- cell dropped")
            continue
        cfg = rec["cfg"]
        ts_p = float(cfg.get("tso_period_s", 180.0))
        ds_p = float(cfg.get("dso_period_s", 20.0))
        dt_s = float(cfg.get("dt_s", 20.0))
        n_tot = float(cfg.get("n_total_s", 600.0))
        trip_t = float(rec["trip_t"])
        # the first TSO dispatch strictly after the trip
        nxt = ts_p * (np.floor(trip_t / ts_p) + 1.0)

        row = {"window": window, "delta_pu": delta, "gen": gen,
               "run": rec["run"], "twin": twin["run"], "trip_t_s": trip_t,
               "next_tso_dispatch_s": float(nxt)}
        row.update(false_activation(
            drift_samples(twin["dir"], ts_p, ds_p, t_start=ts_p),
            delta, prefix="fa_"))
        row.update(false_activation(open_by_window[window], delta,
                                    prefix="faopen_"))
        # The RMS event fires inside the interval ENDING at trip_t (see
        # rejection()), so every raw-trace window opens one interval early.
        onset = trip_t - dt_s
        row["onset_t_s"] = float(onset)
        row.update(rejection(rec["dir"], twin["dir"], trip_t, nxt, onset))
        row.update(activity(rec["dir"], onset, n_tot, dt_s))

        s, b = _ifq(rec["dir"]), _ifq(twin["dir"])
        if s is not None and b is not None:
            post, bpost = s["t"] >= trip_t, b["t"] >= trip_t
            if post.any() and bpost.any():
                row["ifq_post_mvar"] = float(np.nanmean(s["ifq"][post]))
                row["ifq_twin_mvar"] = float(np.nanmean(b["ifq"][bpost]))
                n = min(int(post.sum()), int(bpost.sum()))
                row["ifq_peak_excess_mvar"] = float(np.nanmax(
                    s["ifq"][post][:n] - b["ifq"][bpost][:n])) if n else np.nan
        rows.append(row)

    # ---- the design criterion, stated directly -------------------------
    # The Q(V) layer exists to reject EVENTS while leaving PROFILE-driven
    # shifts to the OFO.  So a dead band is admissible when
    #   (a) profile drift stays INSIDE it            -> faopen_* ~ 0
    #   (b) the event excursion falls OUTSIDE it     -> detected
    #   (c) and the droop actually shrinks that excursion -> comp_eff > 0
    # (c) is only meaningful against a no-local-control reference, which is
    # the widest dead band in the sweep: there the droop never engages, so its
    # peak is the excursion the slow layer would have to absorb alone.
    # Grouped by (window, gen): the no-droop reference must come from the SAME
    # window, since both the drift and the event severity are window-dependent.
    for window, gen in {(r["window"], r["gen"]) for r in rows}:
        grp = [r for r in rows
               if r["window"] == window and r["gen"] == gen]
        ref = max(grp, key=lambda r: r["delta_pu"])
        for lvl in ("ts", "ds"):
            base = ref.get(f"peak_dv_{lvl}_pu", float("nan"))
            for r in grp:
                p = r.get(f"peak_dv_{lvl}_pu", float("nan"))
                r[f"comp_eff_{lvl}"] = (
                    1.0 - p / base
                    if np.isfinite(p) and np.isfinite(base) and base > 0
                    else float("nan"))
                # Detection uses the OPEN-LOOP excursion (the reference cell),
                # not this cell's suppressed one: whether an event is inside or
                # outside the dead band is a property of the event, and using
                # the suppressed value would let a working droop argue itself
                # out of having detected anything.
                r[f"detected_{lvl}"] = (
                    bool(base > r["delta_pu"]) if np.isfinite(base) else False)
        ref_run = ref["run"]
        for r in grp:
            r["nolocal_ref_run"] = ref_run
            r["nolocal_ref_delta_pu"] = ref["delta_pu"]
    return rows


COLS = ["window", "delta_pu", "gen", "run", "twin", "trip_t_s", "onset_t_s",
        "next_tso_dispatch_s",
        "fa_ts", "fa_ds", "fa_p90_ts_pu", "fa_p90_ds_pu",
        "fa_max_ts_pu", "fa_max_ds_pu", "fa_n_ts", "fa_n_ds",
        "faopen_ts", "faopen_ds", "faopen_p90_ts_pu", "faopen_p90_ds_pu",
        "faopen_max_ts_pu", "faopen_max_ds_pu", "faopen_n_ts", "faopen_n_ds",
        "peak_dv_ts_pu", "peak_dv_ds_pu", "resid_dv_ts_pu", "resid_dv_ds_pu",
        "traverse_ts", "traverse_ds",
        "comp_eff_ts", "comp_eff_ds", "detected_ts", "detected_ds",
        "nolocal_ref_run", "nolocal_ref_delta_pu",
        "ifq_post_mvar", "ifq_twin_mvar", "ifq_peak_excess_mvar"]


def write_csv(rows, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / "deadband_n1_metrics.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["window"], x["gen"], x["delta_pu"])):
            w.writerow(r)
    return p


def print_tables(rows: List[dict]) -> None:
    for window, gen in sorted({(r["window"], r["gen"]) for r in rows}):
        g = [r for r in rows
             if r["window"] == window and r["gen"] == gen]
        print(f"\n=== {window}   trip gen {gen} "
              f"(rows: dead band; twin-referenced) ===")
        print(f"{'delta':>7} | {'FAopTS':>6} {'FAopDS':>6} | "
              f"{'peakTS':>8} {'peakDS':>8} | {'resTS':>8} {'resDS':>8} | "
              f"{'ifq':>7} {'travTS':>8}")
        for r in sorted(g, key=lambda x: x["delta_pu"]):
            print(f"{r['delta_pu']:7g} | "
                  f"{r.get('faopen_ts', float('nan')):6.2f} "
                  f"{r.get('faopen_ds', float('nan')):6.2f} | "
                  f"{r.get('peak_dv_ts_pu', float('nan')):8.5f} "
                  f"{r.get('peak_dv_ds_pu', float('nan')):8.5f} | "
                  f"{r.get('resid_dv_ts_pu', float('nan')):8.5f} "
                  f"{r.get('resid_dv_ds_pu', float('nan')):8.5f} | "
                  f"{r.get('ifq_post_mvar', float('nan')):7.3f} "
                  f"{r.get('traverse_ts', float('nan')):8.3f}")
        print("  FAop = fraction of inter-dispatch windows whose OPEN-LOOP "
              "drift exceeds delta (false-activation rate);")
        print("  peak/res are twin-referenced |dV| at the worst park "
              "(missed-detection depth).")
        print(f"{'delta':>7} | {'detTS':>5} {'detDS':>5} | "
              f"{'compTS':>7} {'compDS':>7}   <- event detected & compensated")
        for r in sorted(g, key=lambda x: x["delta_pu"]):
            print(f"{r['delta_pu']:7g} | "
                  f"{str(r.get('detected_ts', '')):>5} "
                  f"{str(r.get('detected_ds', '')):>5} | "
                  f"{r.get('comp_eff_ts', float('nan')):7.2f} "
                  f"{r.get('comp_eff_ds', float('nan')):7.2f}")


def print_verdict(rows: List[dict], fa_tol: float = 0.0) -> None:
    """Which dead bands satisfy the design intent, stated as the intent itself.

    The Q(V) layer is a disturbance-rejection mechanism: profile-driven voltage
    shifts belong to the OFO and must stay INSIDE the dead band, while
    event-driven excursions must fall outside it and be compensated.  A dead
    band is therefore admissible for a given event when

        false activation <= fa_tol   (profile drift stays inside)
        detected                     (the event's open-loop excursion exceeds it)
        comp_eff > 0                 (the droop actually shrinks the excursion)

    An EMPTY admissible set is a result, not a failure: it means the profile
    and event distributions overlap at that level, so no threshold separates
    them and the architecture -- not the tuning -- has to change.
    """
    print("\n=== admissible dead bands (profile inside, event compensated) ===")
    for window, gen in sorted({(r["window"], r["gen"]) for r in rows}):
        g = [r for r in rows
             if r["window"] == window and r["gen"] == gen]
        for lvl in ("ts", "ds"):
            ok = [r for r in sorted(g, key=lambda x: x["delta_pu"])
                  if np.isfinite(r.get(f"faopen_{lvl}", np.nan))
                  and r[f"faopen_{lvl}"] <= fa_tol
                  and r.get(f"detected_{lvl}", False)
                  and np.isfinite(r.get(f"comp_eff_{lvl}", np.nan))
                  and r[f"comp_eff_{lvl}"] > 0.0]
            lo = f"{ok[0]['delta_pu']:g}" if ok else "--"
            hi = f"{ok[-1]['delta_pu']:g}" if ok else "--"
            note = "" if ok else "  (EMPTY: profile and event overlap here)"
            print(f"  {window}  gen {gen}  {lvl.upper()}: delta in [{lo}, {hi}]"
                  f"  from {len(ok)} admissible cell(s){note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--scenario", default="rural_700")
    ap.add_argument("--fa-tol", type=float, default=0.0,
                    help="tolerated false-activation rate: the fraction of "
                         "inter-dispatch windows in which the droop may fire "
                         "on ordinary profile drift. 0 = the droop must be "
                         "completely silent on profiles, which is the stated "
                         "design intent")
    args = ap.parse_args(argv)

    twins, trips = collect(args.results_root, args.scenario)
    rows = evaluate(twins, trips)
    if not rows:
        print("no admitted (trip, twin) pairs -- nothing to report")
        return 1
    print_tables(rows)
    print_verdict(rows, fa_tol=args.fa_tol)
    p = write_csv(rows, args.out)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
