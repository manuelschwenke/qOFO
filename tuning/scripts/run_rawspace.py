"""tuning/scripts/run_rawspace.py — run a raw-weight study (preconditioner off).

Three modes:

``--measure-reference``
    Evaluate the baseline itself (all ratios = 1, ``precondition_g_w=False``)
    across the scenario set and print its performance scalar and the metrics the
    constraints are built from.  Run this **before** the study: it says what the
    hand-tuned controller scores under the same objective as the reparameterised
    study, and whether the limits calibrated on the *preconditioned* reference
    admit it.

``--worker N``
    One worker process: create/load the study and run N trials against it.

(default)
    Launch ``--workers`` worker processes against one SQLite study, the same
    multi-process pattern as :mod:`tuning.scripts.run_tuning_parallel` (the RDB
    serialises trial handout).  Throughput on this class of machine peaks at 6
    workers; BLAS threads are pinned to 1 each.

Usage::

    python -m tuning.scripts.run_rawspace --measure-reference
    python -m tuning.scripts.run_rawspace --n-trials 50 --workers 6
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_BASELINE = (_SCRIPT_DIR / "configs" / "baseline_ieee39_thevenin.yaml")
_MAX_USEFUL_WORKERS = 6


def _load_limits(path: Path | None):
    from tuning.objectives_v2 import ConstraintLimits
    if path is None:
        return ConstraintLimits()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(ConstraintLimits)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"[raw] Unknown limit fields: {sorted(unknown)}")
    return ConstraintLimits(**data)


def _build(args):
    """Shared setup: baseline, reference, scenarios, weights, limits."""
    from tuning._io import load_config_yaml
    from tuning.objectives_v2 import PERF_WEIGHT_PROFILES
    from tuning.parameters import FIXED_OVERRIDES
    from tuning.rawspace import RawReference, raw_space_fingerprint
    from tuning.scenarios import tune_set_v2

    cfg = load_config_yaml(Path(args.baseline))
    ref = RawReference.from_config(cfg)
    if args.perf_weights not in PERF_WEIGHT_PROFILES:
        raise SystemExit(f"[raw] Unknown --perf-weights {args.perf_weights!r}")
    weights = PERF_WEIGHT_PROFILES[args.perf_weights]
    limits = _load_limits(args.limits)
    scenarios = tune_set_v2()
    return cfg, ref, scenarios, weights, limits, FIXED_OVERRIDES, \
        raw_space_fingerprint(ref)


def _print_reference_header(ref, weights, limits, fingerprint):
    print(f"[raw] Reference weights (ratio 1.0): "
          f"{ {k: float(v) for k, v in ref.values.items()} }", flush=True)
    print(f"[raw] Gauge (pinned): { {k: float(v) for k, v in ref.gauge.items()} }",
          flush=True)
    print(f"[raw] precondition_g_w = False on every trial", flush=True)
    print(f"[raw] Objective weights: {dataclasses.asdict(weights)}", flush=True)
    print(f"[raw] Constraint limits: {dataclasses.asdict(limits)}", flush=True)
    print(f"[raw] Space fingerprint: {fingerprint}", flush=True)


def measure_reference(args) -> int:
    """Score the baseline itself under the study's objective."""
    from tuning.metrics import MetricScales
    from tuning.objectives_v2 import (
        CONSTRAINT_NAMES, _run_scenario, _worst_settling_s,
        feasibility_constraints, performance_scalar,
    )
    from tuning.objective import cvar_aggregate
    from tuning.rawspace import apply_raw_to_config

    cfg0, ref, scenarios, weights, limits, fixed, fp = _build(args)
    _print_reference_header(ref, weights, limits, fp)

    coords = {name: 1.0 for name in
              ("g_w_der_ratio", "g_w_pcc_ratio", "g_w_dso_der_ratio",
               "dso_v_priority", "shunt_int_gain")}
    cfg = apply_raw_to_config(cfg0, coords, ref, fixed_overrides=fixed)
    scales = MetricScales()

    results, settling, perf = [], [], {}
    for sc in scenarios:
        t0 = time.perf_counter()
        res, records = _run_scenario(sc, cfg, scales)
        results.append(res)
        settling.append(_worst_settling_s(records, sc.event_times_s))
        total, parts = performance_scalar(res.metrics, weights, scales)
        perf[sc.name] = total
        m = res.metrics
        print(f"[raw] {sc.name:24s} perf={total:8.4f}  "
              f"rho_p95={m.rho_emp_p95:6.4f}  "
              f"ops/h TSO={m.tap_ops_per_h_tso:5.3f} DSO={m.tap_ops_per_h_dso:5.3f}  "
              f"rev/h TSO={m.tap_reversals_per_h_tso:5.3f} "
              f"DSO={m.tap_reversals_per_h_dso:5.3f}  "
              f"feasible={m.feasible}  ({time.perf_counter() - t0:.0f} s)"
              + (f"  ERROR {res.failure_reason[:200]}"
                 if res.failure_reason else ""), flush=True)

    g = feasibility_constraints(results, cfg, limits,
                                settling_s_by_scenario=settling)
    agg = cvar_aggregate(list(perf.values()), pct=args.cvar_pct)
    print(f"\n[raw] REFERENCE CVaR-{args.cvar_pct:g}: {agg:.6f}", flush=True)
    print(f"[raw] constraints (<=0 feasible):", flush=True)
    for name, v in zip(CONSTRAINT_NAMES, g):
        print(f"         {name:22s} {v:12.6f}   {'OK' if v <= 0 else 'VIOLATED'}",
              flush=True)
    out = Path(args.reference_out) if args.reference_out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "cvar": float(agg), "cvar_pct": float(args.cvar_pct),
            "perf": {k: float(v) for k, v in perf.items()},
            "constraints": dict(zip(CONSTRAINT_NAMES, map(float, g))),
            "limits": dataclasses.asdict(limits),
            "reference_weights": {k: float(v) for k, v in ref.values.items()},
            "gauge": {k: float(v) for k, v in ref.gauge.items()},
            "fingerprint": fp,
        }, indent=1), encoding="utf-8")
        print(f"[raw] wrote {out}", flush=True)
    return 0


def run_worker(args) -> int:
    import optuna
    from tuning.objectives_v2 import constraints_func
    from tuning.rawspace import RAW_DIMS, make_raw_objective

    cfg0, ref, scenarios, weights, limits, fixed, fp = _build(args)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.n_startup_trials,
        multivariate=True, group=True, constraints_func=constraints_func,
    )
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="minimize", sampler=sampler, load_if_exists=True,
    )
    # Identity guards: a resumed study must be the same space, the same
    # objective and the same feasible set, or its trials are not comparable.
    for key, value in (("raw_space_fingerprint", fp),
                       ("perf_weights_profile", args.perf_weights),
                       ("constraint_limits", dataclasses.asdict(limits)),
                       ("cvar_pct", float(args.cvar_pct))):
        stored = study.user_attrs.get(key)
        if stored is None:
            study.set_user_attr(key, value)
        elif stored != value:
            raise SystemExit(
                f"[raw] Refusing to resume {args.study_name!r}: {key} differs "
                f"(stored {stored!r}, requested {value!r})."
            )

    if args.worker == 0:
        _print_reference_header(ref, weights, limits, fp)
        print(f"[raw] Raw space ({len(RAW_DIMS)} dims): "
              f"{[p.name for p in RAW_DIMS]}", flush=True)
    if not args.no_warm_start_baseline and len(study.trials) == 0:
        study.enqueue_trial({p.name: 1.0 for p in RAW_DIMS})
        print("[raw] Warm-start: enqueued the reference point as trial 0",
              flush=True)

    objective = make_raw_objective(
        baseline_cfg=cfg0, ref=ref, scenarios=scenarios,
        fixed_overrides=fixed, limits=limits, weights=weights,
        cvar_pct=args.cvar_pct,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    from tuning.objectives_v2 import (
        best_feasible_trial, constraint_violation_report,
    )
    print(f"\n[raw] Constraint violations over {len(study.trials)} trials:",
          flush=True)
    for name, count in constraint_violation_report(study).items():
        n = max(len([t for t in study.trials if t.state.name == "COMPLETE"]), 1)
        print(f"         {name:22s} {count:5d}  ({100.0 * count / n:5.1f} %)",
              flush=True)
    try:
        best = best_feasible_trial(study)
        print(f"[raw] Best CVaR-{args.cvar_pct:g}: {best.value:.6f}", flush=True)
        print(f"[raw] Best coords: {best.params}", flush=True)
        print(f"[raw] Best weights: "
              f"{ {k[3:]: v for k, v in best.user_attrs.items() if k.startswith('w__')} }",
              flush=True)
    except RuntimeError as exc:
        print(f"[raw] {exc}", flush=True)
    return 0


def launch(args) -> int:
    per_worker = [args.n_trials // args.workers] * args.workers
    for i in range(args.n_trials % args.workers):
        per_worker[i] += 1

    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    print(f"[raw] study={args.study_name!r} storage={args.storage}", flush=True)
    print(f"[raw] {args.n_trials} trials over {args.workers} workers "
          f"{per_worker}", flush=True)

    log_dir = _REPO_ROOT / "results" / "tuning" / "rawspace"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for w, n in enumerate(per_worker):
        if n == 0:
            continue
        cmd = [
            sys.executable, "-m", "tuning.scripts.run_rawspace",
            "--worker", str(w), "--n-trials", str(n),
            "--baseline", str(args.baseline),
            "--study-name", args.study_name, "--storage", args.storage,
            "--perf-weights", args.perf_weights,
            "--cvar-pct", str(args.cvar_pct),
            "--n-startup-trials", str(args.n_startup_trials),
            "--seed", str(2000 + w),
        ]
        if args.limits:
            cmd += ["--limits", str(args.limits)]
        if w > 0:
            cmd.append("--no-warm-start-baseline")
        log = log_dir / f"worker_{w}.log"
        fh = log.open("w", encoding="utf-8")
        print(f"[raw] worker {w}: {n} trials -> {log}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env,
                                      stdout=fh, stderr=subprocess.STDOUT))
        time.sleep(20.0)

    t0 = time.perf_counter()
    codes = [p.wait() for p in procs]
    print(f"[raw] all workers finished in {(time.perf_counter() - t0)/3600:.2f} h; "
          f"exit codes {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.run_rawspace")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--study-name", type=str, default="rawspace_thevenin_2026-08-14")
    p.add_argument("--storage", type=str,
                   default="sqlite:///F:/qofo_tuning/rawspace_thevenin_2026-08-14.db")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--n-startup-trials", type=int, default=12)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--cvar-pct", type=float, default=100.0)
    p.add_argument("--perf-weights", type=str, default="ts_voltage_primary")
    p.add_argument("--limits", type=Path, default=None)
    p.add_argument("--worker", type=int, default=None)
    p.add_argument("--no-warm-start-baseline", action="store_true")
    p.add_argument("--measure-reference", action="store_true")
    p.add_argument("--reference-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.workers > _MAX_USEFUL_WORKERS:
        print(f"[raw] WARNING: throughput peaks at {_MAX_USEFUL_WORKERS} workers.")
    if args.measure_reference:
        return measure_reference(args)
    if args.worker is not None:
        return run_worker(args)
    return launch(args)


if __name__ == "__main__":
    sys.exit(main())
