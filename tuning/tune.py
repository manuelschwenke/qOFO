"""
tuning/tune.py
==============
CLI entry point for the BO tuning loop.

Usage::

    python -m tuning.tune \\
        --baseline configs/baseline.yaml \\
        --n-trials 80 \\
        --study-name v1_wind_replace \\
        --storage sqlite:///results/tuning/studies.db \\
        --n-jobs 1 \\
        --output configs/tuned_params.yaml

Resumability: re-running with the same ``--study-name`` and
``--storage`` continues the existing study (Optuna handles persistence).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import optuna

from tuning._io import load_config_yaml, save_tuned_params
from tuning.ceilings import compute_ceilings
from tuning.metrics import CostWeights, NoiseFloors
from tuning.objective import make_objective
from tuning.scenarios import design_set


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tuning.tune",
        description="Run Bayesian-optimisation controller-weight tuning.",
    )
    p.add_argument(
        "--baseline", type=Path, required=True,
        help="Path to baseline MultiTSOConfig YAML.",
    )
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument(
        "--n-startup-trials", type=int, default=15,
        help="Sobol-style initial trials before TPE kicks in.",
    )
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument(
        "--storage", type=str,
        default="sqlite:///results/tuning/studies.db",
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel scenarios per trial (1 is safest).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path,
        default=Path("configs/tuned_params.yaml"),
    )
    p.add_argument(
        "--report", type=Path,
        default=Path("results/tuning/tuning_report.html"),
    )
    p.add_argument("--no-cache-ceilings", action="store_true")
    p.add_argument("--cvar-pct", type=float, default=25.0)
    p.add_argument(
        "--no-progress-bar", action="store_true",
        help="Suppress the Optuna progress bar (useful for tests / CI).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(
            parents=True, exist_ok=True,
        )

    print(f"[tune] Loading baseline from {args.baseline} ...", flush=True)
    baseline_cfg = load_config_yaml(args.baseline)

    print("[tune] Computing LMI ceilings (may take ~30 s on first run) ...",
          flush=True)
    ceilings = compute_ceilings(
        baseline_cfg, use_cache=not args.no_cache_ceilings,
    )
    print(f"[tune] Ceilings: {ceilings.as_dict()}", flush=True)

    scenarios = design_set()
    print(
        f"[tune] Design set: {len(scenarios)} scenarios "
        f"({[s.name for s in scenarios]})",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    objective = make_objective(
        baseline_cfg=baseline_cfg,
        ceilings=ceilings,
        design_scenarios=scenarios,
        cost_weights=CostWeights(),
        noise_floors=NoiseFloors(),
        n_jobs=args.n_jobs,
        cvar_pct=args.cvar_pct,
    )

    print(f"[tune] Running {args.n_trials} trials ...", flush=True)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=not args.no_progress_bar,
    )

    best = study.best_trial
    print(
        f"\n[tune] Best CVaR-{args.cvar_pct:.0f}: {best.value:.6f}",
        flush=True,
    )
    print(f"[tune] Best params: {best.params}", flush=True)

    meta = {
        "study_name":        args.study_name,
        "n_trials":          len(study.trials),
        "best_value":        float(best.value) if best.value is not None else None,
        "best_trial_number": int(best.number),
        "ceilings":          ceilings.as_dict(),
        "ceilings_notes":    ceilings.notes,
        "cvar_pct":          float(args.cvar_pct),
        "timestamp":         datetime.now().isoformat(),
        "baseline_path":     str(args.baseline),
    }
    save_tuned_params(best.params, meta, args.output)
    print(f"[tune] Wrote tuned params -> {args.output}", flush=True)

    from tuning.reports.tuning_report import write_tuning_report
    write_tuning_report(study, ceilings, args.report)
    print(f"[tune] Wrote report -> {args.report}", flush=True)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
