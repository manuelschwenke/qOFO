"""Bayesian optimisation over continuous G_w classes with per-trial LMIs.

Run from the project root:

    python -m tuning.stability_certificate.bo --n-trials 16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna

from configs.config import MultiTSOConfig

from .candidate import CandidateEvaluation, evaluate_candidate
from .hierarchy import (
    DEFAULT_CONFIG_FACTORY,
    certificate_from_result,
    load_config_factory,
    write_json as write_certificate_json,
)
from .report import write_markdown as write_certificate_markdown
from .snapshot import (
    CONTINUOUS_BO_FIELDS,
    CachedCurvatureSnapshot,
    load_or_extract_snapshot,
    rebuild_stability_result,
)


SEARCH_FACTORS: dict[str, tuple[float, float]] = {
    "g_w_der": (0.10, 2.0),
    "g_w_pcc": (0.10, 2.0),
    "g_w_dso_der": (0.25, 2.0),
}


@dataclass(frozen=True)
class StabilityBOResult:
    baseline: CandidateEvaluation
    best: CandidateEvaluation
    trials: tuple[CandidateEvaluation, ...]
    fixed_parameters: dict[str, float]
    search_bounds: dict[str, tuple[float, float]]
    n_trials: int
    seed: int
    snapshot_cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _search_bounds(config: MultiTSOConfig) -> dict[str, tuple[float, float]]:
    return {
        name: (
            float(getattr(config, name)) * SEARCH_FACTORS[name][0],
            float(getattr(config, name)) * SEARCH_FACTORS[name][1],
        )
        for name in CONTINUOUS_BO_FIELDS
    }


def run_stability_bo(
    snapshot: CachedCurvatureSnapshot,
    baseline: MultiTSOConfig,
    *,
    n_trials: int = 16,
    seed: int = 7,
    closeness_weight: float = 0.05,
) -> StabilityBOResult:
    """Run an in-memory Optuna study over continuous class weights."""

    if n_trials < 1:
        raise ValueError("n_trials must be at least one")
    bounds = _search_bounds(baseline)
    evaluations: dict[int, CandidateEvaluation] = {}

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(8, max(2, n_trials // 3)),
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    baseline_params = {
        name: float(getattr(baseline, name))
        for name in CONTINUOUS_BO_FIELDS
    }
    study.enqueue_trial(baseline_params)

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_float(
                name,
                bounds[name][0],
                bounds[name][1],
                log=True,
            )
            for name in CONTINUOUS_BO_FIELDS
        }
        evaluation = evaluate_candidate(
            snapshot,
            baseline,
            params,
            closeness_weight=closeness_weight,
        )
        evaluations[trial.number] = evaluation
        trial.set_user_attr("g_w_gen_fixed", evaluation.fixed_g_w_gen)
        trial.set_user_attr("coupled_active_rho", evaluation.coupled_active_rho)
        trial.set_user_attr("c3_gamma", evaluation.c3_gamma)
        trial.set_user_attr(
            "all_candidate_lmis_certified",
            evaluation.all_candidate_lmis_certified,
        )
        for name, rho in evaluation.local_rho.items():
            trial.set_user_attr(f"rho__{name}", rho)
        return evaluation.objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    ordered = tuple(evaluations[index] for index in sorted(evaluations))
    if not ordered:
        raise RuntimeError("The stability BO did not evaluate any candidate.")
    baseline_evaluation = evaluations[0]
    best_evaluation = evaluations[study.best_trial.number]

    return StabilityBOResult(
        baseline=baseline_evaluation,
        best=best_evaluation,
        trials=ordered,
        fixed_parameters={
            "g_w_gen": float(baseline.g_w_gen),
            "g_w_tso_oltc": float(baseline.g_w_tso_oltc),
            "g_w_tso_shunt": float(baseline.g_w_tso_shunt),
            "g_w_dso_oltc": float(baseline.g_w_dso_oltc),
            "shunt_int_g_w": float(baseline.shunt_int_g_w),
        },
        search_bounds=bounds,
        n_trials=len(ordered),
        seed=seed,
        snapshot_cache_key=snapshot.cache_key,
    )


def _number(value: float) -> str:
    return f"{value:.7g}"


def bo_report_markdown(result: StabilityBOResult) -> str:
    baseline = result.baseline
    best = result.best
    lines = [
        "# Cached-curvature LMI Bayesian optimisation",
        "",
        "## Verdict",
        "",
        (
            "- Every candidate rebuilt the preconditioned cached curvature and "
            "reran local sector LMIs, the coupled active-mode Lyapunov LMI, "
            "and the exact discrete C3 spectral-radius test."
        ),
        f"- Trials: {result.n_trials}; seed: {result.seed}.",
        f"- Baseline objective: {_number(baseline.objective)}.",
        f"- Best objective: {_number(best.objective)}.",
        f"- Baseline coupled active rho: {_number(baseline.coupled_active_rho)}.",
        f"- Best coupled active rho: {_number(best.coupled_active_rho)}.",
        f"- C3 baseline/best gamma: {_number(baseline.c3_gamma)} / {_number(best.c3_gamma)}.",
        "",
        "## Parameters",
        "",
        "| Parameter | Baseline | Best | Ratio | Search interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CONTINUOUS_BO_FIELDS:
        baseline_value = baseline.params[name]
        best_value = best.params[name]
        low, high = result.search_bounds[name]
        lines.append(
            f"| {name} | {_number(baseline_value)} | {_number(best_value)} | "
            f"{_number(best_value / baseline_value)} | "
            f"[{_number(low)}, {_number(high)}] |"
        )
    lines.extend(
        [
            "",
            "## Fixed parameters",
            "",
        ]
    )
    lines.extend(
        f"- {name} = {_number(value)}" for name, value in result.fixed_parameters.items()
    )
    lines.extend(
        [
            "",
            "The generator weight was asserted unchanged in every candidate. "
            "Discrete weights were fixed because this study optimises only "
            "continuous projected-gradient rates.",
            "",
            "## Local rates",
            "",
            "| Controller | Baseline rho | Best rho | Best LMI |",
            "|---|---:|---:|---|",
        ]
    )
    for name, baseline_rho in baseline.local_rho.items():
        lines.append(
            f"| {name} | {_number(baseline_rho)} | "
            f"{_number(best.local_rho[name])} | "
            f"{best.local_lmi_certified[name]} |"
        )
    lines.extend(
        [
            "",
            "## Certificate interpretation",
            "",
            (
                f"- Candidate continuous LMIs: {best.all_candidate_lmis_certified}. "
                f"Neutral coupled modes: {best.n_coupled_neutral}."
            ),
            (
                f"- Exact modeled MIQP small-gain condition: rho(Gamma) = "
                f"{_number(best.c3_gamma)} < 1 is {best.c3_certified}."
            ),
            (
                "- The C3 result is a good certificate for the discrete "
                "interconnection represented in Gamma. It does not cover the "
                "separately dispatched hysteretic shunt integrator."
            ),
            (
                "- Failure of a row-sum or individual G_w-margin sufficient "
                "test does not invalidate the exact rho(Gamma) < 1 result."
            ),
            "",
            "## All trials",
            "",
            "| Trial | Objective | g_w_der | g_w_pcc | g_w_dso_der | "
            "Coupled rho | C3 gamma | LMIs |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for index, trial in enumerate(result.trials):
        lines.append(
            f"| {index} | {_number(trial.objective)} | "
            f"{_number(trial.params['g_w_der'])} | "
            f"{_number(trial.params['g_w_pcc'])} | "
            f"{_number(trial.params['g_w_dso_der'])} | "
            f"{_number(trial.coupled_active_rho)} | "
            f"{_number(trial.c3_gamma)} | "
            f"{trial.all_candidate_lmis_certified} |"
        )
    return "\n".join(lines) + "\n"


def write_bo_outputs(
    result: StabilityBOResult,
    *,
    markdown_path: Path,
    json_path: Path,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(bo_report_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-factory", default=DEFAULT_CONFIG_FACTORY)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--closeness-weight", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stability_certificate/bo"),
    )
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config_factory(args.config_factory)
    snapshot = load_or_extract_snapshot(config, use_cache=not args.no_cache)
    result = run_stability_bo(
        snapshot,
        config,
        n_trials=args.n_trials,
        seed=args.seed,
        closeness_weight=args.closeness_weight,
    )
    write_bo_outputs(
        result,
        markdown_path=args.output_dir / "stability_bo_report.md",
        json_path=args.output_dir / "stability_bo_report.json",
    )

    best_config = dataclasses.replace(config, **result.best.params)
    best_result = rebuild_stability_result(snapshot, best_config)
    best_certificate = certificate_from_result(
        best_result,
        best_config,
        config_factory=args.config_factory + " + stability-BO overrides",
    )
    write_certificate_markdown(
        best_certificate,
        args.output_dir / "best_candidate_certificate.md",
    )
    write_certificate_json(
        best_certificate,
        args.output_dir / "best_candidate_certificate.json",
    )
    (args.output_dir / "best_candidate_params.json").write_text(
        json.dumps(
            {
                "continuous_overrides": result.best.params,
                "fixed_g_w_gen": result.best.fixed_g_w_gen,
                "objective": result.best.objective,
                "c3_gamma": result.best.c3_gamma,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Trials: {result.n_trials}")
    print(f"Best objective: {result.best.objective:.7g}")
    print(f"Best params: {result.best.params}")
    print(f"Fixed g_w_gen: {result.best.fixed_g_w_gen:.7g}")
    print(f"C3 gamma: {result.best.c3_gamma:.7g}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
