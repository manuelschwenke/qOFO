"""
tuning/validate.py
==================
CLI to evaluate tuned params on the randomised validation set.

Usage::

    python -m tuning.validate \\
        --params configs/tuned_params.yaml \\
        --baseline configs/baseline.yaml \\
        --n-scenarios 200 \\
        --seed 42 \\
        --report results/tuning/validation_v1.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from tuning._io import load_config_yaml, load_tuned_params
from tuning.metrics import CostWeights, NoiseFloors
from tuning.runner import RunResult, run_one
from tuning.scenarios import validation_set


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tuning.validate",
        description="Validate tuned controller params on a randomised "
                    "scenario set.",
    )
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--n-scenarios", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument(
        "--report", type=Path,
        default=Path("results/tuning/validation_report.html"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    print(f"[validate] Loading baseline {args.baseline} ...", flush=True)
    baseline_cfg = load_config_yaml(args.baseline)

    print(f"[validate] Loading tuned params {args.params} ...", flush=True)
    params, meta = load_tuned_params(args.params)
    print(
        f"[validate] Tuned at trial {meta.get('best_trial_number', '?')} "
        f"with best_value={meta.get('best_value', '?')}",
        flush=True,
    )

    scenarios = validation_set(args.seed, args.n_scenarios)
    print(
        f"[validate] Running {len(scenarios)} scenarios "
        f"with n_jobs={args.n_jobs} ...",
        flush=True,
    )

    cost_weights = CostWeights()
    noise_floors = NoiseFloors()

    t0 = time.perf_counter()
    if args.n_jobs <= 1:
        results: List[RunResult] = [
            run_one(params, sc, baseline_cfg, cost_weights, noise_floors)
            for sc in scenarios
        ]
    else:
        from joblib import Parallel, delayed
        results = list(Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(run_one)(
                params, sc, baseline_cfg, cost_weights, noise_floors,
            )
            for sc in scenarios
        ))
    wall = time.perf_counter() - t0
    print(f"[validate] Total wall time: {wall:.1f} s", flush=True)

    n_pf_fail = sum(1 for r in results if r.metrics.pf_failures > 0)
    Js = [float(r.metrics.cost_J) for r in results]
    print(
        f"[validate] PF failures: {n_pf_fail}/{len(results)}",
        flush=True,
    )
    if Js:
        print(
            f"[validate] Cost J: median={np.median(Js):.4f}  "
            f"mean={np.mean(Js):.4f}  p95={np.percentile(Js, 95):.4f}  "
            f"max={np.max(Js):.4f}",
            flush=True,
        )

    from tuning.reports.validation_report import write_validation_report
    write_validation_report(results, params, meta, args.report)
    print(f"[validate] Wrote report -> {args.report}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
