#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py
============================================================
Stage II-a of the Chapter-9.3 weight campaign: the screening sweep.

What it does
------------
1. Draws ``--n`` weight vectors from a scrambled Sobol sequence in log space,
   over a box centred on the Stage-I design point (``--paramset``).
2. Puts every draw through the **analytic** contraction screen
   (:mod:`tuning.stage1_screen`), which needs no simulation. Measured
   2026-09-10, this rejects about three quarters of the box in seconds.
3. Simulates only the survivors, on the short bank -- the six operating
   conditions of thesis Table ``tab:timescales:conditions`` at ``--hours``.
4. Writes one JSON per candidate, keyed by a fingerprint of its weights, so an
   interrupted sweep resumes instead of restarting.

Why the ordering matters: a candidate costs ~1.7 worker-hours on the 3 h bank
and ~5.6 on the 12 h bank (5.6 min wall per simulated hour per worker, measured
on this server). Screening analytically first is what turns a week of compute
into a night.

What this stage does NOT decide
-------------------------------
* **Tap wear.** One tap is 0.33/h on a 3 h window against 0.083/h at 12 h, so a
  per-day switching budget cannot be read off the short bank. The budget is
  applied in Stage II-b/II-c on the 12 h bank; here the limits are checked with
  ``wear_is_meaningful=False``.
* **The final point.** This is a screening sweep. Its outputs are a feasibility
  map, a per-coordinate sensitivity ranking, and a shortlist.

Usage::

    python experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py \
        --paramset stage1_start --n 512 --hours 3 --workers 28
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _pin_threads() -> None:
    """One BLAS thread per worker.

    Without this each worker spawns as many threads as there are cores and the
    pool spends its time in contention rather than in power flows.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _abort_hook(job: Dict[str, Any], events_s) -> Any:
    """Per-step hook that stops a window once its candidate is lost anyway.

    Once per TS iteration it counts this window's unexplained tap and bank
    reversals (the filter's own ``hunting_summary`` / ``shunt_summary`` on the
    records so far), writes them to ``<abort_dir>/<key>__<window>.json`` and sums
    the files of all the candidate's windows on this host.  At ``abort_tap`` tap
    or ``abort_bank`` bank reversals it raises ``EarlyStop``.  A reversal is
    final once its second move is recorded, so the partial count never exceeds
    the full one and an aborted candidate is rejected by the filter regardless.
    """
    from experiments.runners.multi_tso_dso import EarlyStop
    from tuning.stage2_search import hunting_summary, shunt_summary

    adir = Path(job["abort_dir"])
    adir.mkdir(parents=True, exist_ok=True)
    mine = adir / f"{job['key']}__{job['window']}.json"
    last = {"n": None}

    def hook(step: int, log: list) -> None:
        if step % 9:                      # once per TS iteration (T_TS / T_STS)
            return
        n = (int(hunting_summary(log, events_s)["n_hunting"]),
             int(shunt_summary(log, events_s)["n_shunt_hunting"]))
        if n != last["n"]:
            mine.write_text(json.dumps({"tap": n[0], "bank": n[1]}), encoding="utf-8")
            last["n"] = n
        tap = bank = 0
        for f in adir.glob(f"{job['key']}__*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # a sibling is writing it right now
                continue
            tap += int(d.get("tap", 0))
            bank += int(d.get("bank", 0))
        if ((job["abort_tap"] and tap >= job["abort_tap"])
                or (job["abort_bank"] and bank >= job["abort_bank"])):
            raise EarlyStop(f"candidate {job['key']}: {tap} tap and {bank} bank "
                            f"reversal(s) over its windows")

    return hook


def _run_one_window(job: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ONE (candidate, window) pair.  Module-level, so picklable.

    The work unit is a single window rather than a whole candidate.  On the
    12 h bank a candidate is ~3.4 h of wall clock, and losing that to one
    worker death -- which, through ``BrokenProcessPool``, takes every other
    pending future with it -- cost the first Stage II-b attempt its entire
    1.8 h run with nothing written.  A window is ~34 min and is persisted the
    moment it finishes, so a death costs one window and the sweep resumes from
    disk.
    """
    _pin_threads()
    import warnings
    warnings.filterwarnings("ignore")

    from configs.paramsets import apply_paramset
    from experiments.run_multi_system_ofo import make_config
    from tuning.metrics import MetricScales
    from tuning.parameters import apply_to_config
    from tuning.runner import run_scenario_with_records
    from tuning.stage2_search import (avr_step_metrics, hunting_summary,
                                  shunt_summary)
    from tuning.scenarios_mc_v2 import condition_set, condition_set_evented

    out_path = Path(job["out"])
    if out_path.exists() and not job.get("force"):
        return {"key": job["key"], "window": job["window"], "status": "cached"}

    base = apply_to_config(
        apply_paramset(make_config(), job["paramset"], verbose=False),
        job["weights"])
    if job.get("frozen_network"):
        # The batch network, applied to the SIMULATED window as well as to the
        # analytic screen.  ScenarioSpec.overlay_on does not touch the per-DSO
        # multipliers, so without this the window runs symmetric while the
        # screen that admitted the candidate ran doubled.
        import dataclasses as _dc

        from experiments.ch_09_parameter_selection._ch9_selected_design import (
            FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE,
        )
        base = _dc.replace(
            base,
            dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
            dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
    bank = (condition_set_evented if job.get("evented") else condition_set)
    sc = next(s for s in bank(hours=job["hours"]) if s.name == job["window"])

    import time as _t
    import experiments.runners.multi_tso_dso as _runner
    use_abort = ((job.get("abort_tap") or job.get("abort_bank"))
                 and not job.get("is_baseline"))
    _runner.STEP_HOOK = _abort_hook(job, sc.event_times_s) if use_abort else None
    t0 = _t.perf_counter()
    try:
        try:
            result, records = run_scenario_with_records(sc, base, MetricScales())
        finally:
            _runner.STEP_HOOK = None      # a pool worker runs many units
        m = result.metrics
        early = (result.failure_reason or "").startswith("EarlyStop")
        # Ranking voltage metric (2026-10-16, protocol D1).  'areamean' is the
        # RMS over areas of (area-mean V - v_set), the definition of every
        # campaign before 2026-10-16; 'bus' is the per-bus spatial RMS
        # (v_rms_ts / v_rms_ds), which is what the thesis text describes and
        # what the controllers minimise.  Both are always stored; e_v_ts /
        # e_v_ds carry the one the campaign ranks on.
        _areamean = (float(getattr(m, "rmsd_v_ts", float("nan"))),
                     float(getattr(m, "rmsd_v_ds", float("nan"))))
        _bus = (float(getattr(m, "v_rms_ts", float("nan"))),
                float(getattr(m, "v_rms_ds", float("nan"))))
        v_metric = job.get("v_metric", "areamean")
        if v_metric not in ("areamean", "bus"):
            raise ValueError(f"v_metric must be 'areamean' or 'bus', got {v_metric!r}")
        _rank = _bus if v_metric == "bus" else _areamean
        row = {
            "e_v_ts": _rank[0],
            "e_v_ds": _rank[1],
            "v_metric": v_metric,
            "e_v_ts_areamean": _areamean[0],
            "e_v_ds_areamean": _areamean[1],
            "e_v_ts_bus": _bus[0],
            "e_v_ds_bus": _bus[1],
            "e_q": float(getattr(m, "itae_q_pcc", float("nan"))),
            "v_band_excess_ts": float(getattr(m, "v_band_excess_ts", 0.0) or 0.0),
            "v_band_excess_ds": float(getattr(m, "v_band_excess_ds", 0.0) or 0.0),
            "n_viol_v_ts": float(getattr(m, "n_viol_v_ts", 0) or 0),
            "n_viol_v_ds": float(getattr(m, "n_viol_v_ds", 0) or 0),
            "v_worst_ts": float(getattr(m, "v_worst_ts", 0.0) or 0.0),
            "v_worst_ds": float(getattr(m, "v_worst_ds", 0.0) or 0.0),
            "tap_ops_per_h_tso": float(getattr(m, "tap_ops_per_h_tso", 0.0) or 0.0),
            "tap_ops_per_h_dso": float(getattr(m, "tap_ops_per_h_dso", 0.0) or 0.0),
            "tap_rev_per_h_tso": float(getattr(m, "tap_reversals_per_h_tso", 0.0) or 0.0),
            "tap_rev_per_h_dso": float(getattr(m, "tap_reversals_per_h_dso", 0.0) or 0.0),
            "pf_failures": float(getattr(m, "pf_failures", 0) or 0),
            "wall_s": float(result.wall_time_s),
            # An early stop is not a failed simulation: its record holds the
            # hunting that rejects the candidate (procedure v3).
            "failed": 1.0 if (result.failure_reason and not early) else 0.0,
            "failure_reason": result.failure_reason or "",
            "early_abort": 1.0 if early else 0.0,
        }
        row.update(avr_step_metrics(records))
        # Hunting, and the raw move list it is derived from.  Persisting the
        # moves is what lets a revised wear definition be applied without
        # re-simulating; the first Stage II-b dropped its records and could
        # not be re-scored when the definition changed.
        row.update(hunting_summary(records, sc.event_times_s))
        row.update(shunt_summary(records, sc.event_times_s))
        # Per hour, like the tap rates, so aggregate() and check_limits()
        # treat bank wear exactly as they treat tap wear (2026-10-22).
        row["shunt_ops_per_h_bank"] = (float(row["shunt_ops_max_bank"])
                                       / float(job["hours"]))
        del records                       # the log is the large object; drop it
    except Exception as exc:              # noqa: BLE001 - recorded, not raised
        import traceback
        row = {"failed": 1.0,
               "failure_reason": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(),
               "wall_s": _t.perf_counter() - t0}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"key": job["key"], "window": job["window"], "hours": job["hours"],
         "weights": job["weights"], "lambda_max_B": job["lambda_max_B"],
         "is_baseline": bool(job.get("is_baseline")), "row": row},
        indent=1, default=float), encoding="utf-8")
    return {"key": job["key"], "window": job["window"],
            "status": "error" if row.get("failed") else "ok",
            "e_v_ts": row.get("e_v_ts"), "wall_s": row.get("wall_s", 0.0)}


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ch_9_3_stage2_screen")
    p.add_argument("--paramset", default="stage1_start",
                   help="Stage-I design point; the search box is centred here.")
    p.add_argument("--frozen-network", action="store_true",
                   help="Apply the Chapter-9 per-DSO multipliers (DSO_3 x2 in "
                        "installed DER and active load) to BOTH the analytic "
                        "screen and every simulated window. The batch is "
                        "declared on that network; make_config() is symmetric, "
                        "so the Stage-I/II campaign of 2026-09-10/11 ran "
                        "symmetric throughout and its 12 h metrics do not "
                        "transfer to the batch network.")
    p.add_argument("--n", type=int, default=512,
                   help="Sobol draws BEFORE the analytic screen (power of two).")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Cap the number of survivors actually simulated.")
    p.add_argument("--ops-dir", type=Path, required=True,
                   help="Cached Stage-I operating points (op_*.npz).")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "THESIS_ch9_3_stage2" / "screen")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-restarts", type=int, default=5,
                   help="Pool rebuilds allowed after a worker death.")
    p.add_argument("--dry-run", action="store_true",
                   help="Screen and report, simulate nothing.")
    p.add_argument("--candidates-json", type=Path, default=None,
                   help="Evaluate an EXPLICIT list of weight vectors "
                        "instead of drawing a Sobol design. Used by "
                        "Stage II-b to re-run a shortlist on the 12 h "
                        "bank, where the per-day switching budget is "
                        "readable and the short-bank ranking is not.")
    p.add_argument("--evented", action="store_true",
                   help="Use the condition bank that carries one disturbance "
                        "per window. Required for the hunting metric to be "
                        "able to distinguish an event-explained reversal from "
                        "a limit cycle.")
    p.add_argument("--no-wear-limits", action="store_true",
                   help="Force wear limits off (implied for --hours < 6).")
    p.add_argument("--v-metric", choices=("areamean", "bus"), default="areamean",
                   help="Voltage metric the candidates are ranked on. "
                        "'areamean' (default, every campaign before 2026-10-16): "
                        "RMS over areas of the area-mean deviation. 'bus': "
                        "per-bus spatial RMS (v_rms_ts / v_rms_ds), as the "
                        "thesis text reads; required by the 2026-10-16 "
                        "protocol (D1). Both are stored either way.")
    p.add_argument("--windows", default="",
                   help="Comma-separated subset of the bank's windows to "
                        "SIMULATE (default: all). The analytic screen always "
                        "uses every condition. Added 2026-10-25 to drop W5, "
                        "which enters neither the filter nor the ranking.")
    p.add_argument("--op-horizon-h", type=float, default=None,
                   help="Run length of the operating-point capture [h], i.e. "
                        "the window the unit commitment is computed over "
                        "(tuning.stage1_screen.load_or_build_operating_points). "
                        "Default: 60 s, as before 2026-10-31.")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Do not prepend the Stage-I point. For a shard of a "
                        "campaign split over hosts, whose baseline another "
                        "shard evaluates (added 2026-10-31).")
    p.add_argument("--abort-tap-reversals", type=int, default=0, metavar="N",
                   help="Stop every window of a candidate once its windows together "
                        "count N unexplained tap reversals (hunting tolerance + 1); "
                        "0 = never (default). The Stage-I point is never stopped. "
                        "Procedure v3, 2026-11-03.")
    p.add_argument("--abort-bank-reversals", type=int, default=0, metavar="N",
                   help="As --abort-tap-reversals, for bank hunting reversals.")
    p.add_argument("--span", action="append", default=[], metavar="NAME=DECADES",
                   help="Override one log-span of the search box (repeatable), "
                        "e.g. --span g_w_pcc=1.5. Unlisted coordinates keep "
                        "tuning.stage2_search.DEFAULT_SPANS. The resolved spans "
                        "are written to run_meta.json.")
    args = p.parse_args(argv)

    from configs.paramsets import load_paramset
    from tuning.parameters import SEARCHED_FIELDS, weight_fingerprint
    from tuning.stage1_screen import screen_weights, select_ceiling, worst
    from tuning.stage2_search import Box, DEFAULT_SPANS, sobol_candidates

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / stamp
    cand_dir = args.out_dir / "candidates"
    win_dir = args.out_dir / "windows"
    run_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    win_dir.mkdir(parents=True, exist_ok=True)

    centre = dict(load_paramset(args.paramset)["fields"])

    # Search-box spans: defaults, overridden per coordinate by --span.
    spans = dict(DEFAULT_SPANS)
    for item in args.span:
        name, sep, value = item.partition("=")
        if not sep or name not in SEARCHED_FIELDS:
            raise SystemExit(f"--span {item!r}: expected NAME=DECADES with NAME "
                             f"in {list(SEARCHED_FIELDS)}")
        spans[name] = float(value)
        if not spans[name] > 0.0:
            raise SystemExit(f"--span {item!r}: span must be > 0")

    # Operating points, selected BY NETWORK.  A bare ``glob("op_*.npz")`` was
    # correct only while one network existed on disk: since 2026-09-15 the
    # cache is keyed on the per-DSO multipliers as well as the benchmark name,
    # so the same directory holds the symmetric and the doubled-DSO_3 points
    # and the glob would have screened a 12-point MIXTURE of two networks.
    # Going through load_or_build_operating_points reuses the keying rather
    # than restating it, and builds the points if they are absent.
    from experiments.run_multi_system_ofo import make_config
    from tuning.parameters import apply_to_config as _apply_w
    from tuning.stage1_screen import load_or_build_operating_points
    from tuning.scenarios_mc_v2 import CONDITIONS

    _base = _apply_w(make_config(), centre)
    if args.frozen_network:
        from experiments.ch_09_parameter_selection._ch9_selected_design import (
            FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE,
        )
        _base = dataclasses.replace(
            _base,
            dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
            dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))
    ops = load_or_build_operating_points(
        _base, [(r[0], r[1]) for r in CONDITIONS], Path(args.ops_dir),
        horizon_s=60.0 if args.op_horizon_h is None else 3600.0 * args.op_horizon_h)
    if not ops:
        raise SystemExit(f"no cached operating points in {args.ops_dir}")
    print(f"[stage2] {len(ops)} operating points on the "
          f"{'DOUBLED-DSO_3' if args.frozen_network else 'symmetric'} network")

    ceiling = select_ceiling()
    kept: List[Dict[str, Any]] = []
    lam_all: List[float] = []

    if args.candidates_json:
        explicit = json.loads(args.candidates_json.read_text(encoding='utf-8'))
        for w in explicit:
            lam = worst(screen_weights(ops, w)).lambda_max
            lam_all.append(lam)
            kept.append({"weights": w, "lambda_max_B": lam})
        print(f"[stage2] {len(kept)} explicit candidates from "
              f"{args.candidates_json.name}")
    else:
        box = Box(centre=centre, spans=spans)
        cands = sobol_candidates(box, args.n, seed=args.seed)
        for w in cands:
            lam = worst(screen_weights(ops, w)).lambda_max
            lam_all.append(lam)
            if lam <= ceiling:
                kept.append({"weights": w, "lambda_max_B": lam})

    if not args.candidates_json:
        print(f"[stage2a] {args.n} Sobol draws, seed {args.seed}")
        print(f"[stage2a] analytic screen at {ceiling:.4f}: "
              f"{len(kept)} survivors ({100 * len(kept) / args.n:.1f} %)")

    # The Stage-I point itself is always evaluated, first.  Two reasons: every
    # improvement has to be quoted against a NAMED baseline, and the relative
    # interface-Q bound needs a reference that is a stated design point rather
    # than whichever candidate happened to score best.
    if not args.skip_baseline:
        centre_lambda = worst(screen_weights(ops, centre)).lambda_max
        kept.insert(0, {"weights": dict(centre), "lambda_max_B": centre_lambda,
                        "is_baseline": True})
        print(f"[stage2a] baseline (Stage-I point) prepended, "
              f"lambda_max(B) = {centre_lambda:.4f}")
    if args.max_candidates:
        kept = kept[: args.max_candidates]
        print(f"[stage2a] capped to {len(kept)} candidates")

    est_h = len(kept) * args.hours * len(ops) * (5.6 / 60.0) / max(args.workers, 1)
    print(f"[stage2a] bank {len(ops)} x {args.hours:g} h, {args.workers} workers"
          f" -> estimated {est_h:.1f} h wall")

    meta = {
        "stamp": stamp, "paramset": args.paramset, "n_draws": args.n,
        "frozen_network": bool(args.frozen_network),
        "network": ("DSO_3 x2 (Chapter-9 batch)" if args.frozen_network
                    else "symmetric"),
        "seed": args.seed, "hours": args.hours, "workers": args.workers,
        "screen_ceiling": ceiling, "n_survivors": len(kept),
        "box_centre": centre, "box_spans": spans,
        "v_metric": args.v_metric,
        "searched": list(SEARCHED_FIELDS),
        "conditions": [o.condition for o in ops],
        "simulated_windows": args.windows or "all",
        "op_horizon_h": args.op_horizon_h,
        "skip_baseline": bool(args.skip_baseline),
        "lambda_max_all": lam_all,
        "estimated_wall_h": est_h,
        "argv": sys.argv,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=1, default=float), encoding="utf-8")

    if args.dry_run:
        print(f"[stage2a] dry run; wrote {run_dir / 'run_meta.json'}")
        return 0

    # Deduplicate by fingerprint before submitting.  Two jobs with the same
    # weights write the same result file, and concurrent writes to one path can
    # tear it -- which would corrupt exactly the baseline record every
    # comparison is quoted against.  An explicit shortlist that already
    # contains the baseline is the normal way this happens.
    window_names = [o.condition for o in ops]
    if args.windows:
        wanted = [w.strip() for w in args.windows.split(",") if w.strip()]
        unknown = sorted(set(wanted) - set(window_names))
        if unknown:
            raise SystemExit(f"--windows: unknown window(s) {unknown}; the bank "
                             f"has {window_names}")
        window_names = [w for w in window_names if w in wanted]
    jobs: List[Dict[str, Any]] = []
    seen_keys: set = set()
    cand_meta: Dict[str, Dict[str, Any]] = {}
    for c in kept:
        key = weight_fingerprint(c["weights"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cand_meta[key] = c
        for wname in window_names:
            jobs.append({
                "key": key, "window": wname, "weights": c["weights"],
                "lambda_max_B": c["lambda_max_B"],
                "is_baseline": bool(c.get("is_baseline")),
                "paramset": args.paramset, "hours": args.hours,
                "evented": bool(args.evented),
                "frozen_network": bool(args.frozen_network),
                "v_metric": args.v_metric,
                "out": str(win_dir / f"{key}__{wname}.json"),
                "force": args.force,
                "abort_tap": int(args.abort_tap_reversals),
                "abort_bank": int(args.abort_bank_reversals),
                "abort_dir": str(win_dir.parent / "_abort"),
            })
    todo = [j for j in jobs if args.force or not Path(j["out"]).exists()]
    print(f"[stage2] {len(seen_keys)} candidates x {len(window_names)} windows "
          f"= {len(jobs)} units, {len(todo)} to run")

    t0 = time.perf_counter()
    ok = failed = 0
    done = 0
    # BrokenProcessPool marks EVERY pending future as failed when one worker
    # dies, so the pool is rebuilt and the remaining units retried from disk
    # rather than the run being abandoned.  The first Stage II-b attempt lost
    # 1.8 h and 20 whole candidates to exactly this.
    attempt = 0
    while todo and attempt < args.max_restarts + 1:
        attempt += 1
        if attempt > 1:
            print(f"[stage2] pool restart {attempt - 1}, {len(todo)} units left",
                  flush=True)
        broken = False
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_run_one_window, j): j for j in todo}
                for fut in as_completed(futs):
                    done += 1
                    try:
                        r = fut.result()
                    except BrokenProcessPool:
                        broken = True
                        break
                    except Exception as exc:      # noqa: BLE001
                        failed += 1
                        print(f"[stage2] {done}/{len(jobs)} unit failed: {exc}",
                              flush=True)
                        continue
                    if r["status"] == "ok":
                        ok += 1
                        print(f"[stage2] {done}/{len(jobs)} {r['key']} "
                              f"{r['window']} e_v_ts={r['e_v_ts']:.5f} "
                              f"({r['wall_s'] / 60:.1f} min)", flush=True)
                    elif r["status"] == "error":
                        failed += 1
                        print(f"[stage2] {done}/{len(jobs)} {r['key']} "
                              f"{r['window']} ERROR", flush=True)
        except BrokenProcessPool:
            broken = True
        todo = [j for j in jobs if not Path(j["out"]).exists()]
        if not broken:
            break
        print(f"[stage2] pool broke; {len(todo)} units remain", flush=True)

    # ---- assemble per-window results into candidate records --------------
    from tuning.stage2_search import aggregate
    from tuning.scenarios_mc_v2 import ZERO_CAPABILITY

    n_written = 0
    for key, c in cand_meta.items():
        per_window: Dict[str, Any] = {}
        for wname in window_names:
            f = win_dir / f"{key}__{wname}.json"
            if f.exists():
                per_window[wname] = json.loads(f.read_text(encoding="utf-8"))["row"]
        if len(per_window) != len(window_names):
            print(f"[stage2] {key}: only {len(per_window)}/{len(window_names)} "
                  f"windows present -- candidate record NOT written")
            continue
        rec = {
            "key": key, "weights": c["weights"],
            "lambda_max_B": c["lambda_max_B"],
            "is_baseline": bool(c.get("is_baseline")),
            "hours": args.hours, "per_window": per_window,
            "aggregate": aggregate(per_window, exclude=ZERO_CAPABILITY),
            "aggregate_with_zero_capability": aggregate(per_window),
            "wall_s": sum(float(r.get("wall_s", 0.0)) for r in per_window.values()),
        }
        (cand_dir / f"cand_{key}.json").write_text(
            json.dumps(rec, indent=1, default=float), encoding="utf-8")
        n_written += 1

    print(f"[stage2] finished in {(time.perf_counter() - t0) / 3600:.2f} h: "
          f"{ok} units ok, {failed} failed, {n_written} candidate records")
    print(f"[stage2] candidates in {cand_dir}")
    print(f"[stage2] meta in {run_dir / 'run_meta.json'}")
    return 0 if (failed == 0 and n_written == len(cand_meta)) else 2


if __name__ == "__main__":
    sys.exit(main())
