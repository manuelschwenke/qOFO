"""tuning/scripts/run_tuning_parallel.py — run one study across N processes.

Runbook step 4.  A `--reparam` trial costs ~15 min (5 scenarios x ~175 s), so 80
trials is ~19 h serial.  This launches N worker processes that all call
``study.optimize`` against the **same** SQLite study, which is Optuna's
documented multi-process pattern: the RDB serialises trial handout, so the
workers need no coordination and the objective needs no changes.

Why process-level and not the per-scenario ``--n-jobs`` path
------------------------------------------------------------
``tuning/objective.py`` parallelises the *scenarios within one trial* via
joblib/loky, and its own docstring warns that this "has been seen to
occasionally interfere with pandapower's solver setup".  Trial-level
parallelism avoids that entirely — each worker is a plain single-threaded run of
exactly the code path validated serially.

Expected speed-up, measured not assumed
---------------------------------------
The bottleneck is memory bandwidth on the per-step sparse Newton power flow, not
CPU.  The 2026-06-02 Monte-Carlo campaign measured scaling on this class of
machine: **K=2 -> 1.5x, K=6 -> 2.14x (peak), K=8 -> 2.02x, K=10 -> regression**.
So the default here is 5 workers and there is no point going past ~6; expect
roughly **2x**, i.e. ~9-10 h for 80 trials, not 80/N h.

BLAS threads are pinned to 1 per worker for the same reason the campaign pins
them: N workers each spawning a thread pool oversubscribes the memory bus and
makes total throughput *worse*.

Usage::

    python -m tuning.scripts.run_tuning_parallel --n-trials 80 --workers 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
DEFAULT_BASELINE = _SCRIPT_DIR / "configs" / "baseline_ieee39.yaml"

#: Beyond this the 2026-06-02 measurement shows throughput regressing.
_MAX_USEFUL_WORKERS = 6


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuning.scripts.run_tuning_parallel")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--n-trials", type=int, default=80,
                   help="TOTAL trials across all workers.")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--n-startup-trials", type=int, default=12)
    p.add_argument("--study-name", type=str, default="v5_reparam")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--output", type=Path,
                   default=_REPO_ROOT / "configs" / "tuned_params_reparam.yaml")
    p.add_argument("--cvar-pct", type=float, default=25.0,
                   help="Forwarded to tuning.tune.  Over a 3-4 scenario set "
                        "pct=25 IS the maximum; pass 100 for the mean.")
    p.add_argument("--perf-exclude", type=str, default="",
                   help="Forwarded to tuning.tune: scenarios left out of the "
                        "performance aggregate but still run and constrained.")
    p.add_argument("--scenario-set", type=str, default="tune_v2",
                   choices=("design", "tune_v2"))
    p.add_argument("--perf-weights", type=str, default="calibrated_2026_08",
                   help="Forwarded to tuning.tune: named objective weight "
                        "profile. Every worker must pass the same one; the "
                        "study records it and refuses a mismatched resume.")
    p.add_argument("--limits", type=Path, default=None,
                   help="Forwarded to tuning.tune: JSON of ConstraintLimits "
                        "fields. Omit for the 2026-08-04 defaults.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.workers > _MAX_USEFUL_WORKERS:
        print(f"[par] WARNING: {args.workers} workers requested; measured "
              f"throughput peaks at 6 and regresses past 8 (memory-bandwidth "
              f"bound, see docs/daily_log/2026-06-02_006_cigre_montecarlo.md).")

    storage = args.storage or (
        "sqlite:///" + str(_REPO_ROOT / "results" / "tuning" / "studies.db"))
    if storage.startswith("sqlite:///"):
        Path(storage[len("sqlite:///"):]).parent.mkdir(parents=True,
                                                       exist_ok=True)

    # Trials are split evenly; the RDB hands them out, so an uneven finish just
    # means one worker stops slightly earlier.
    per_worker = [args.n_trials // args.workers] * args.workers
    for i in range(args.n_trials % args.workers):
        per_worker[i] += 1

    env = dict(os.environ)
    # One BLAS thread per worker: N workers x a thread pool each oversubscribes
    # the memory bus and reduces total throughput.
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    })

    print(f"[par] study={args.study_name!r} storage={storage}")
    print(f"[par] {args.n_trials} trials over {args.workers} workers "
          f"{per_worker}; expect ~2x speed-up, not {args.workers}x")

    procs: list[subprocess.Popen] = []
    for w, n in enumerate(per_worker):
        if n == 0:
            continue
        cmd = [
            sys.executable, "-m", "tuning.tune",
            "--baseline", str(args.baseline),
            "--reparam",
            "--scenario-set", args.scenario_set,
            "--n-trials", str(n),
            "--n-startup-trials", str(args.n_startup_trials),
            "--study-name", args.study_name,
            "--storage", storage,
            "--output", str(args.output),
            "--no-progress-bar",
            # Distinct sampler seeds, or every worker proposes the same startup
            # points and the Sobol-like coverage collapses.
            "--seed", str(1000 + w),
            "--cvar-pct", str(args.cvar_pct),
            "--perf-weights", args.perf_weights,
        ]
        if args.perf_exclude:
            cmd += ["--perf-exclude", args.perf_exclude]
        if args.limits:
            cmd += ["--limits", str(args.limits)]
        if w > 0:
            # Only worker 0 enqueues the reference; the others would duplicate
            # it and waste ~15 min each on an identical evaluation.
            cmd.append("--no-warm-start-baseline")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        log = _REPO_ROOT / "results" / "tuning" / f"worker_{w}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", encoding="utf-8")
        print(f"[par] worker {w}: {n} trials -> {log}")
        procs.append(subprocess.Popen(
            cmd, cwd=_REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT))
        # Stagger: all workers computing the LMI ceilings and building the
        # network simultaneously is the worst moment for the memory bus, and
        # concurrent first writes to a fresh SQLite file can collide.
        time.sleep(20.0)

    if args.dry_run:
        return 0

    t0 = time.perf_counter()
    codes = [pr.wait() for pr in procs]
    print(f"[par] all workers finished in {(time.perf_counter() - t0) / 3600:.2f} h; "
          f"exit codes {codes}")

    bad = [i for i, c in enumerate(codes) if c != 0]
    if bad:
        print(f"[par] workers {bad} exited non-zero — inspect their logs before "
              f"trusting the study. A worker that died early simply contributed "
              f"fewer trials; one that died on a *systematic* error may have "
              f"contributed none.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
