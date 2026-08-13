"""
tuning/scripts/run_holdout.py
=============================
Stage 5 — score candidate points on the **holdout** scenario set, once.

Why not ``tuning/validate.py``
-----------------------------
Two reasons it cannot serve:

1. It builds scenarios with ``validation_set(seed, n)``, which draws
   ``start_time`` uniformly over 2016 with **no ISO-week parity constraint**.
   The v2 tune set is restricted to *odd* ISO weeks
   (``scenarios._HOLDOUT_WEEK_PARITY = 0``), so ``validation_set`` samples tune
   weeks too.  Given SimBench's strong within-day autocorrelation that leaks the
   tune set into the "validation" score.  ``holdout_set_v2`` enforces the parity.
2. It evaluates a single ``--params`` point.  Stage 5 compares several points on
   *identical* scenarios.

Evaluate once
-------------
The holdout is spent the first time it is read.  If tune-to-holdout degradation
is large, that is an **overfitting result to report**, not a licence to re-tune —
re-tuning on it consumes the only independent evidence the campaign has.  This
script therefore refuses to overwrite an existing output file without
``--force``, so a second invocation cannot quietly replace the first.

Scores
------
Per point, aggregated across the holdout scenarios:

* ``v_rms_ts`` / ``v_rms_ds``  — spatial-RMS voltage deviation [pu]
* ``q_pcc_rms_mvar`` / ``q_tie_rms_mvar`` — interface-Q tracking **RMS**
  [Mvar], computed here (see :mod:`tuning.holdout_metrics`); the recorded
  metrics carry only ITAE, a different statistic
* ``itae_q_pcc`` / ``itae_q_tie`` — reported alongside for continuity with the
  tuning objective
* ``tap_ops_per_h_tso`` / ``_dso`` and the reversal rates — **worst** across
  scenarios, matching how constraint g5a/g5b aggregate

Usage::

    python -m tuning.scripts.run_holdout \\
        --bo-params configs/tuned_params_reparam.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_BASELINE = _SCRIPT_DIR / "configs" / "baseline_ieee39.yaml"
DEFAULT_HANDTUNED = _SCRIPT_DIR / "configs" / "baseline_ieee39_handtuned.yaml"


def _eval_one_scenario(scenario, cfg, scales) -> tuple[dict[str, Any], Any]:
    """Run one scenario and reduce it to a row plus its ``RunResult``.

    Module-level (not a closure) so :mod:`joblib` can pickle it.  The RMS is
    computed **here**, inside the worker, so only the small row travels back --
    returning the raw record list would ship an entire trajectory per scenario
    across the process boundary.
    """
    from tuning.holdout_metrics import (
        q_pcc_error_series,
        q_tie_error_series,
        rms_of_series,
    )
    from tuning.objectives_v2 import _run_scenario

    res, records = _run_scenario(scenario, cfg, scales)
    m = res.metrics
    _t, e_pcc, has_pcc = q_pcc_error_series(records)
    _t, e_tie, has_tie = q_tie_error_series(records)
    row = {
        "scenario": scenario.name,
        "feasible": bool(m.feasible),
        "infeasible_reason": m.infeasible_reason,
        "v_rms_ts": float(m.v_rms_ts),
        "v_rms_ds": float(m.v_rms_ds),
        "q_pcc_rms_mvar": rms_of_series(e_pcc, has_pcc),
        "q_tie_rms_mvar": rms_of_series(e_tie, has_tie),
        "itae_q_pcc": float(m.itae_q_pcc),
        "itae_q_tie": float(m.itae_q_tie),
        "tap_ops_per_h_tso": float(m.tap_ops_per_h_tso),
        "tap_ops_per_h_dso": float(m.tap_ops_per_h_dso),
        "tap_reversals_per_h_tso": float(m.tap_reversals_per_h_tso),
        "tap_reversals_per_h_dso": float(m.tap_reversals_per_h_dso),
        "cost_J": float(m.cost_J),
    }
    return row, res


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    """Median / p90 / worst over the finite entries."""
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"median": float("nan"), "p90": float("nan"),
                "worst": float("nan"), "n": 0}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "worst": float(np.max(finite)),
        "n": len(finite),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.run_holdout")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                   help="Config supplying the gauge and the reference point.")
    p.add_argument("--handtuned", type=Path, default=DEFAULT_HANDTUNED,
                   help="Pre-calibration snapshot, scored as a separate point "
                        "when it differs from --baseline.")
    p.add_argument("--bo-params", type=Path, default=None,
                   help="Reparam coords YAML from the BO run "
                        "(configs/tuned_params_reparam.yaml).")
    p.add_argument("--legacy-params", type=Path, default=None,
                   help="Optional legacy 8-dim optimum; skipped if absent.")
    p.add_argument("--analytic-lambda", type=float, default=0.9,
                   help="lambda for the analytic Tier-1+2 point.  OFO is stable "
                        "for eig(M) in (0,2); gw_precondition.py:22 calls it "
                        "well-damped for lambda_max(M) <~ 1, and the BO_DIMS_V2 "
                        "comment names 0.9 specifically.  0.9 chosen 2026-08-04: "
                        "it sits inside the well-damped region rather than on "
                        "its boundary, so the analytic point is not judged at a "
                        "threshold it only marginally satisfies.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-scenarios", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=6,
                   help="Scenarios evaluated concurrently.  Serial would be "
                        "~11 h for 3 points at the measured ~5.5 min/simulation "
                        "(the handover's ~4 h assumed 175 s).  Each job is one "
                        "core once BLAS is pinned, so keep n_jobs within the "
                        "machine's core budget: "
                        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                        "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1.")
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "results" / "tuning" / "holdout_scores.json")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing --out.  The holdout is spent on "
                        "first read; only pass this if the earlier run is known "
                        "to be invalid (e.g. it crashed).")
    p.add_argument("--lambda-sweep", type=str, default="",
                   help="Comma-separated lambda values to score at tau=1, "
                        "priority=1 (the analytic family), e.g. "
                        "'0.5,0.7,0.9,1.1'.  Values already covered by another "
                        "point are skipped rather than duplicated.")
    p.add_argument("--include-production", action="store_true",
                   help="Also score --baseline exactly as it stands, WITHOUT "
                        "apply_reparam_to_config.  Needed because that function "
                        "forces precondition_g_w=True, so every coordinate point "
                        "runs with the preconditioner on while production has it "
                        "off -- and with it off there is no lambda, so production "
                        "cannot be written as a coordinate.  Without this flag the "
                        "comparison is against a preconditioned reference, not "
                        "against the controller actually in use.")
    p.add_argument("--dry-run", action="store_true",
                   help="Assemble the points and the scenario set, print them, "
                        "and exit without simulating.  Validates the config "
                        "load, gauge, coordinate extraction and holdout draw at "
                        "zero CPU cost -- and does NOT consume the holdout.")
    args = p.parse_args(argv)

    if args.out.exists() and not args.force and not args.dry_run:
        print(f"[holdout] REFUSING to run: {args.out} already exists.\n"
              f"          The holdout must be evaluated exactly once -- a "
              f"second evaluation, especially after seeing the first, is no "
              f"longer independent evidence.\n"
              f"          If the previous run is genuinely invalid, pass "
              f"--force (and say so in the daily log).", file=sys.stderr)
        return 2

    from tuning._io import load_config_yaml, load_tuned_params
    from tuning.metrics import MetricScales
    from tuning.objectives_v2 import feasibility_constraints
    from tuning.parameters import FIXED_OVERRIDES
    from tuning.reparam import (
        BO_DIMS_V2,
        Gauge,
        apply_reparam_to_config,
        coords_from_config,
    )
    from tuning.scenarios import holdout_set_v2

    cfg0 = load_config_yaml(args.baseline)
    cfg0 = dataclasses.replace(cfg0, **FIXED_OVERRIDES)
    gauge = Gauge.from_config(cfg0)
    scales = MetricScales()
    coord_names = [d.name for d in BO_DIMS_V2]

    # ── Assemble the candidate points ────────────────────────────────────────
    points: dict[str, dict[str, float]] = {}
    points["reference"] = coords_from_config(cfg0, gauge)

    if args.handtuned.exists():
        cfg_h = dataclasses.replace(load_config_yaml(args.handtuned),
                                    **FIXED_OVERRIDES)
        coords_h = coords_from_config(cfg_h, gauge)
        if any(abs(coords_h[k] - points["reference"][k]) > 1e-12
               for k in coord_names):
            points["handtuned_precalib"] = coords_h
        else:
            print("[holdout] --handtuned matches --baseline in reparam "
                  "coordinates; scoring it once, as 'reference'.")

    points["analytic_tier12"] = {
        "tso_lambda": float(args.analytic_lambda),
        "dso_lambda": float(args.analytic_lambda),
        "tau_der_pcc": 1.0,
        "dso_v_priority": 1.0,
    }

    # ── Optional lambda sweep ────────────────────────────────────────────────
    # A pure loop-gain sweep at tau=1, priority=1, i.e. the analytic family.
    # Needed because the seed-42 draw suggested tracking was monotone in lambda
    # and the seed-43 draw refuted it (best at 0.9, worse at 1.1975) -- three
    # points on one draw could not settle the shape.  This measures the curve on
    # the GOAL metric (v_rms_ts) with paired statistics.
    for lam in [float(s) for s in (args.lambda_sweep or "").split(",") if s.strip()]:
        name = f"lambda_{lam:g}".replace(".", "p")
        if any(abs(c["tso_lambda"] - lam) < 1e-12 and c["tau_der_pcc"] == 1.0
               and c["dso_v_priority"] == 1.0 for c in points.values()):
            print(f"[holdout] lambda={lam:g} already covered by an existing "
                  f"point; not duplicating.")
            continue
        points[name] = {"tso_lambda": lam, "dso_lambda": lam,
                        "tau_der_pcc": 1.0, "dso_v_priority": 1.0}

    if args.bo_params is not None:
        bo_coords, bo_meta = load_tuned_params(args.bo_params)
        missing = set(coord_names) - set(bo_coords)
        if missing:
            print(f"[holdout] --bo-params lacks reparam coords {sorted(missing)}; "
                  f"it has {sorted(bo_coords)}.  Not scoring it -- a legacy "
                  f"8-dim file belongs in --legacy-params.", file=sys.stderr)
        else:
            points["bo_optimum"] = {k: float(bo_coords[k]) for k in coord_names}
            print(f"[holdout] BO optimum from trial "
                  f"{bo_meta.get('best_trial_number', '?')} "
                  f"value={bo_meta.get('best_value', '?')}")

    if args.legacy_params is not None and args.legacy_params.exists():
        print("[holdout] --legacy-params is an 8-dim point and is not "
              "expressible in the 4-dim reparam space; scoring it needs the "
              "legacy apply_to_config path.  Skipped -- see the daily log.",
              file=sys.stderr)

    # ── The production point, which is NOT a coordinate ─────────────────────
    # ``apply_reparam_to_config`` forces ``precondition_g_w=True``, so every
    # coordinate point above runs with the curvature preconditioner ON.  The
    # production controller has it OFF, and with it off there is no lambda at all
    # (``coords_from_config``: lambda is "a property of the cached H"), so the
    # production config cannot be expressed as a coordinate.  It is therefore
    # carried as a raw config and evaluated directly -- the only way to compare
    # the tuned point against what is actually run.
    raw_points: dict[str, Any] = {}
    if args.include_production:
        raw_points["production_precond_off"] = cfg0
        print("[holdout] scoring the production config directly "
              "(precondition_g_w=%s) -- not a reparam coordinate."
              % getattr(cfg0, "precondition_g_w", None))

    scenarios = holdout_set_v2(args.seed, args.n_scenarios)
    # Per-simulation wall time measured at ~6-way concurrency over 720 samples
    # (study v5_reparam_v2, 2026-08-04): median 160 s, mean 220 s, p90 232 s.
    #
    # Use the CONCURRENT figure directly and divide by n_jobs -- do not also
    # apply a parallel speed-up factor.  An earlier version did both (330 s,
    # itself measured under load, then /2) and over-estimated by ~5x, printing
    # 5.5 h for a job that runs in ~1 h.
    n_sims = (len(points) + len(raw_points)) * len(scenarios)
    per_sim_s = 200.0
    est_h = n_sims * per_sim_s / 3600.0 / max(args.n_jobs, 1)
    print(f"[holdout] {len(points) + len(raw_points)} points x "
          f"{len(scenarios)} holdout scenarios "
          f"= {n_sims} simulations (seed={args.seed}, n_jobs={args.n_jobs}); "
          f"~{est_h:.1f} h")
    for name, c in points.items():
        print(f"    {name:22s} " + "  ".join(f"{k}={c[k]:.5g}" for k in coord_names))
    for name in raw_points:
        print(f"    {name:22s} (raw config, preconditioner off -- no coordinates)")

    if args.dry_run:
        weeks = sorted({sc.start_time.isocalendar().week for sc in scenarios})
        bad = [w for w in weeks if w % 2 != 0]
        print(f"\n[holdout] dry run -- nothing simulated, holdout not consumed.")
        print(f"    ISO weeks drawn: {weeks}")
        print(f"    odd (tune-set) weeks present: {bad or 'none -- split is clean'}")
        durations = sorted({float(sc.overlay_on(cfg0).n_total_s)
                            for sc in scenarios})
        print(f"    distinct durations [s]: {durations} "
              f"({'fixed -- no T^2 ITAE bias' if len(durations) == 1 else 'VARYING'})")
        networks = sorted({sc.scenario for sc in scenarios})
        print(f"    networks: {networks}")
        return 0

    # ── Evaluate ─────────────────────────────────────────────────────────────
    out: dict[str, Any] = {
        "seed": args.seed,
        "n_scenarios": len(scenarios),
        "analytic_lambda": args.analytic_lambda,
        "baseline": str(args.baseline),
        "points": {},
    }

    # Coordinate points go through the reparam map; raw points are used as-is.
    evaluation_plan = [(n, apply_reparam_to_config(cfg0, c, gauge,
                                                   fixed_overrides=FIXED_OVERRIDES), c)
                       for n, c in points.items()]
    evaluation_plan += [(n, c, None) for n, c in raw_points.items()]

    for name, cfg, coords in evaluation_plan:
        print(f"\n[holdout] === {name} ===", flush=True)
        if args.n_jobs <= 1:
            pairs = [_eval_one_scenario(sc, cfg, scales) for sc in scenarios]
        else:
            from joblib import Parallel, delayed
            pairs = list(Parallel(n_jobs=args.n_jobs, prefer="processes")(
                delayed(_eval_one_scenario)(sc, cfg, scales) for sc in scenarios
            ))
        per_scenario = [row for row, _res in pairs]
        results = [res for _row, res in pairs]
        for row in per_scenario:
            print(f"  {row['scenario']:26s} feas={str(row['feasible']):5s} "
                  f"v_rms_ts={row['v_rms_ts']:.5f} "
                  f"q_pcc_rms={row['q_pcc_rms_mvar']:8.2f} "
                  f"tapTS/h={row['tap_ops_per_h_tso']:6.3f}", flush=True)

        n_feas = sum(1 for r in per_scenario if r["feasible"])
        summary = {
            key: _aggregate([r[key] for r in per_scenario])
            for key in ("v_rms_ts", "v_rms_ds", "q_pcc_rms_mvar",
                        "q_tie_rms_mvar", "itae_q_pcc", "itae_q_tie",
                        "tap_ops_per_h_tso", "tap_ops_per_h_dso",
                        "tap_reversals_per_h_tso", "tap_reversals_per_h_dso",
                        "cost_J")
        }
        out["points"][name] = {
            "coords": coords,   # None for raw (non-coordinate) points
            "precondition_g_w": bool(getattr(cfg, "precondition_g_w", False)),
            "n_feasible": n_feas,
            "n_scenarios": len(scenarios),
            "constraints": dict(zip(
                [f"g{i}" for i in range(1, 7)],
                [float(v) for v in feasibility_constraints(results, cfg)],
            )),
            "summary": summary,
            "per_scenario": per_scenario,
        }
        print(f"  -> feasible {n_feas}/{len(scenarios)}; "
              f"v_rms_ts median={summary['v_rms_ts']['median']:.5f}; "
              f"q_pcc_rms median={summary['q_pcc_rms_mvar']['median']:.2f} Mvar; "
              f"worst tapTS/h={summary['tap_ops_per_h_tso']['worst']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[holdout] wrote {args.out}")

    # ── Comparison table ─────────────────────────────────────────────────────
    print(f"\n{'point':22s} {'feas':>7s} {'v_rms_ts':>10s} {'q_pcc_rms':>11s} "
          f"{'q_tie_rms':>11s} {'tapTS/h':>9s} {'revTS/h':>9s}")
    print("-" * 84)
    for name, d in out["points"].items():
        s = d["summary"]
        print(f"{name:22s} {d['n_feasible']:3d}/{d['n_scenarios']:<3d} "
              f"{s['v_rms_ts']['median']:10.5f} "
              f"{s['q_pcc_rms_mvar']['median']:11.2f} "
              f"{s['q_tie_rms_mvar']['median']:11.2f} "
              f"{s['tap_ops_per_h_tso']['worst']:9.3f} "
              f"{s['tap_reversals_per_h_tso']['worst']:9.3f}")
    print("\n(medians across scenarios; tap columns are the WORST scenario, "
          "matching how g5a/g5b aggregate)")
    print("Acceptance: if the BO optimum does not beat the reference here, keep "
          "the reference -- the deliverable is then the methodological evidence "
          "for a setting already in hand.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
