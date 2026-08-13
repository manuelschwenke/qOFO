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
import dataclasses
import math
import sys
from datetime import datetime
from pathlib import Path

import optuna

from tuning._io import load_config_yaml, save_tuned_params
from tuning.ceilings import compute_ceilings
from tuning.metrics import CostWeights, NoiseFloors
from tuning.objective import make_objective
from tuning.parameters import (
    BO_DIMS,
    SEARCH_SPACE_VERSION,
    out_of_box_params,
    params_from_config,
    resolve_high,
    search_space_fingerprint,
)
from tuning.scenarios import design_set


#: ``study.user_attrs`` key holding the search-space digest.
_FINGERPRINT_KEY = "search_space_fingerprint"

#: ``study.user_attrs`` key holding the objective's weight-profile name.
_PERF_PROFILE_KEY = "perf_weights_profile"


def _guard_perf_profile(study: optuna.Study, profile: str) -> None:
    """Refuse to resume a study under a different objective weight profile.

    The weights are preferences, not measurements: two profiles define two
    different functions, so trials scored under one are not comparable with
    trials scored under the other.  Optuna would mix them silently, and the
    resulting "best trial" would be the argmin of neither.
    """
    recorded = study.user_attrs.get(_PERF_PROFILE_KEY)
    if recorded is None:
        if study.trials:
            raise SystemExit(
                f"[tune] REFUSING TO RESUME: study {study.study_name!r} has "
                f"{len(study.trials)} trials but no recorded objective weight "
                f"profile, so it cannot be shown to have been scored with "
                f"{profile!r}. Use a new --study-name."
            )
        study.set_user_attr(_PERF_PROFILE_KEY, profile)
        return
    if recorded != profile:
        raise SystemExit(
            f"[tune] REFUSING TO RESUME: study {study.study_name!r} was scored "
            f"with objective weight profile {recorded!r}; this invocation asks "
            f"for {profile!r}. Those are different objectives — their values "
            f"are not comparable. Use a new --study-name."
        )


def _guard_search_space(study: optuna.Study, allow_drift: bool) -> None:
    """Refuse to resume a study whose search space has since changed.

    Optuna will happily continue a study across a redefinition of the decision
    space, silently mixing trials that were never comparable.  That is not
    hypothetical here: every persisted IEEE-39 study carries a 9th parameter,
    ``tso_g_q_tie``, that no longer exists in ``BO_DIMS`` -- or on
    ``MultiTSOConfig``.  Resuming one of those would blend trials scored under a
    different space *and* a different cost function.
    """
    current = search_space_fingerprint()
    recorded = study.user_attrs.get(_FINGERPRINT_KEY)

    if recorded is None:
        if study.trials:
            msg = (
                f"Study {study.study_name!r} has {len(study.trials)} trials but "
                f"no recorded search-space fingerprint, so it predates "
                f"fingerprinting and cannot be shown to be comparable with the "
                f"current space (v{SEARCH_SPACE_VERSION}, {current}). "
                f"Use a new --study-name, or pass --allow-schema-drift if you "
                f"have verified the spaces match."
            )
            if not allow_drift:
                raise SystemExit(f"[tune] REFUSING TO RESUME: {msg}")
            print(f"[tune] WARNING: {msg}", flush=True)
        study.set_user_attr(_FINGERPRINT_KEY, current)
        study.set_user_attr("search_space_version", SEARCH_SPACE_VERSION)
        return

    if recorded != current:
        msg = (
            f"Study {study.study_name!r} was created with search-space "
            f"fingerprint {recorded!r}; the current space is {current!r} "
            f"(v{SEARCH_SPACE_VERSION}).  Resuming would mix incomparable "
            f"trials.  Use a new --study-name."
        )
        if not allow_drift:
            raise SystemExit(f"[tune] REFUSING TO RESUME: {msg}")
        print(f"[tune] WARNING: {msg}", flush=True)


def _resolve_solver_name() -> str:
    """Which MIQP solver cvxpy will actually pick, plus its version.

    ``optimisation.miqp_solver.MIQP_SOLVERS`` is a preference list with a
    silent fallback, and the module's own comments record that SCIP returned
    ``optimal_inaccurate`` on 54/60 DSO solves -- so two studies run with
    different solvers are not comparable.  Record it.
    """
    try:
        import cvxpy as cp

        from optimisation.miqp_solver import MIQPSolver

        installed = set(cp.installed_solvers())
        for name in MIQPSolver.MIQP_SOLVERS:
            if name in installed:
                try:
                    mod = __import__(name.lower())
                    ver = getattr(mod, "__version__", "?")
                except Exception:
                    ver = "?"
                return f"{name} {ver}"
        return "none-of:" + ",".join(MIQPSolver.MIQP_SOLVERS)
    except Exception as exc:  # pragma: no cover - provenance must never break a run
        return f"unresolved ({type(exc).__name__})"


def _warn_on_bound_hugging(study: optuna.Study, ceilings) -> None:
    """Flag a best-trial coordinate pinned against its search bound.

    A boundary optimum usually means the bound is binding rather than the
    optimum being interior -- i.e. the reported value is an artefact of where
    the box was drawn.  The last two studies had ``g_w_der`` at 3 % of its
    log-range and ``dso_g_v`` at 98 %, which nothing in the pipeline surfaced.
    """
    try:
        best = study.best_trial
    except ValueError:
        return
    flags: list[str] = []
    for p in BO_DIMS:
        v = best.params.get(p.name)
        if v is None or v <= 0:
            continue
        hi = resolve_high(p, ceilings)
        if hi <= p.low:
            continue
        pos = ((math.log10(v) - math.log10(p.low))
               / (math.log10(hi) - math.log10(p.low)))
        if pos < 0.05 or pos > 0.95:
            edge = "LOWER" if pos < 0.05 else "UPPER"
            flags.append(
                f"    {p.name}={v:g} at {pos * 100:.0f} % of "
                f"[{p.low:g}, {hi:g}] ({edge} bound)"
            )
    if flags:
        print(
            "[tune] WARNING: best trial sits against a search bound -- the\n"
            "       optimum is likely an artefact of the box, not interior:\n"
            + "\n".join(flags),
            flush=True,
        )


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
    p.add_argument(
        "--n-ei-candidates", type=int, default=50,
        help="Number of EI candidates per TPE proposal "
             "(Optuna default: 24; bumped here because the search "
             "space is 9-dimensional).",
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
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--output", type=Path,
        default=Path("configs/tuned_params.yaml"),
    )
    p.add_argument(
        "--report", type=Path,
        default=Path("results/tuning/tuning_report.html"),
    )
    p.add_argument("--no-cache-ceilings", action="store_true")
    p.add_argument("--cvar-pct", type=float, default=25.0,
                   help="Aggregate the worst PCT %% of scenario scalars.  Note "
                        "that over a 3-4 scenario set pct=25 IS the maximum, so "
                        "the largest-magnitude scenario becomes the whole "
                        "objective; pass 100 for the mean.")
    p.add_argument("--perf-exclude", type=str, default="",
                   help="Comma-separated scenario names left OUT of the "
                        "performance aggregate.  They still run and still enter "
                        "the constraint vector, so a candidate must survive "
                        "them.  Use for scenarios that cannot discriminate "
                        "candidates -- e.g. v2_undervoltage_ramp, a winter "
                        "18:00 case where TS-DER reactive capability is zero, "
                        "making tau_der_pcc structurally inert.")
    p.add_argument(
        "--no-progress-bar", action="store_true",
        help="Suppress the Optuna progress bar (useful for tests / CI).",
    )
    p.add_argument(
        "--no-warm-start-baseline", action="store_true",
        help=(
            "Disable enqueueing the baseline-config parameters as the "
            "first trial. By default, when a fresh study is created the "
            "baseline is evaluated as trial 0 to give TPE a known-good "
            "anchor. Skipped automatically when resuming a study that "
            "already has trials."
        ),
    )
    p.add_argument(
        "--reparam", action="store_true",
        help=(
            "Use the gauge-fixed reparameterised space (tuning.reparam) with "
            "the constrained-scalar objective (tuning.objectives_v2) instead of "
            "the legacy 8-dim raw-weight space. Feasibility becomes an Optuna "
            "constraint rather than a term in the cost, and g_v/g_q/g_w_gen are "
            "pinned at the baseline as the gauge."
        ),
    )
    p.add_argument(
        "--perf-weights", type=str, default="calibrated_2026_08",
        help=(
            "Named objective weight profile (--reparam only). "
            "'calibrated_2026_08' reproduces the 2026-08 campaign; "
            "'ts_voltage_primary' makes TS voltage tracking ~66 %% of the "
            "scalar and demotes interface-Q to the coupling term it is. "
            "Recorded on the study; resuming under a different profile is "
            "refused, because the two are different objectives."
        ),
    )
    p.add_argument(
        "--scenario-set", type=str, default=None,
        choices=("design", "tune_v2"),
        help=(
            "Which scenario set to tune on. Defaults to 'tune_v2' under "
            "--reparam and 'design' otherwise. 'design' is the legacy set, "
            "which does not excite the OLTCs."
        ),
    )
    p.add_argument(
        "--allow-schema-drift", action="store_true",
        help=(
            "Downgrade the search-space-fingerprint mismatch from a hard stop "
            "to a warning. Only use when you have verified that the recorded "
            "trials are comparable with the current decision space."
        ),
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

    # --reparam defaults to the excitation-gated v2 set.  The legacy design set
    # cannot identify either OLTC weight: taps were frozen in 77 % of clean runs
    # and `nominal_quiet` produces 1 TSO tap and 0 DSO taps, so those weights
    # have no leverage on any objective built from it.
    which = args.scenario_set or ("tune_v2" if args.reparam else "design")
    if which == "tune_v2":
        from tuning.scenarios import tune_set_v2
        scenarios = tune_set_v2()
    else:
        scenarios = design_set()
    print(
        f"[tune] Scenario set '{which}': {len(scenarios)} scenarios "
        f"({[s.name for s in scenarios]})",
        flush=True,
    )
    if which == "design" and args.reparam:
        print(
            "[tune] WARNING: the legacy design set does not excite the OLTCs "
            "(1 TSO tap, 0 DSO taps on nominal_quiet), so g_w_*_oltc cannot be "
            "identified from it. Run tuning.scripts.audit_design_set first.",
            flush=True,
        )

    constraints_fn = None
    if args.reparam:
        from tuning.objectives_v2 import (
            constraints_func as _constraints_func,
        )
        constraints_fn = _constraints_func

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=args.n_ei_candidates,
        multivariate=True,
        group=True,
        # Constrained TPE partitions trials by feasibility *before* fitting its
        # good/bad densities, so an infeasible trial's objective value never
        # attracts the sampler.  A penalty term cannot achieve that -- which is
        # how divergence came to be a profitable search direction.
        constraints_func=constraints_fn,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    _guard_search_space(study, allow_drift=args.allow_schema_drift)

    if args.reparam:
        from tuning.objectives_v2 import (
            PERF_WEIGHT_PROFILES,
            make_constrained_objective,
        )
        from tuning.parameters import FIXED_OVERRIDES
        from tuning.reparam import (
            BO_DIMS_V2,
            Gauge,
            coords_from_config,
            priority_report,
        )

        if args.perf_weights not in PERF_WEIGHT_PROFILES:
            raise SystemExit(
                f"[tune] Unknown --perf-weights {args.perf_weights!r}; "
                f"available: {sorted(PERF_WEIGHT_PROFILES)}"
            )
        perf_weights = PERF_WEIGHT_PROFILES[args.perf_weights]
        _guard_perf_profile(study, args.perf_weights)

        gauge = Gauge.from_config(baseline_cfg)
        print(
            f"[tune] Reparameterised space ({len(BO_DIMS_V2)} dims): "
            f"{[p.name for p in BO_DIMS_V2]}",
            flush=True,
        )
        print(
            f"[tune] Objective weights '{args.perf_weights}': "
            f"{dataclasses.asdict(perf_weights)}",
            flush=True,
        )
        print(
            f"[tune] Gauge (pinned): g_v={gauge.g_v:g} g_q={gauge.g_q:g} "
            f"g_w_gen={gauge.g_w_gen:g} tso_g_q_pcc={gauge.tso_g_q_pcc:g}",
            flush=True,
        )
        print(
            f"[tune] Reference priorities: "
            f"{ {k: round(v, 4) for k, v in priority_report(baseline_cfg).items()} }",
            flush=True,
        )
        perf_exclude = frozenset(
            s.strip() for s in (args.perf_exclude or "").split(",") if s.strip()
        )
        if perf_exclude:
            print(f"[tune] Excluded from the performance aggregate (still run, "
                  f"still constrained): {sorted(perf_exclude)}", flush=True)
        objective = make_constrained_objective(
            baseline_cfg=baseline_cfg,
            gauge=gauge,
            scenarios=scenarios,
            fixed_overrides=FIXED_OVERRIDES,
            weights=perf_weights,
            cvar_pct=args.cvar_pct,
            perf_exclude=perf_exclude,
        )
        if not args.no_warm_start_baseline and len(study.trials) == 0:
            # Representable by construction: every ratio coordinate is defined
            # relative to this reference, so it sits at 1.0.
            study.enqueue_trial(coords_from_config(baseline_cfg, gauge))
            print("[tune] Warm-start: enqueued the reference point as trial 0",
                  flush=True)
    else:
        objective = make_objective(
            baseline_cfg=baseline_cfg,
            ceilings=ceilings,
            design_scenarios=scenarios,
            cost_weights=CostWeights(),
            noise_floors=NoiseFloors(),
            n_jobs=args.n_jobs,
            cvar_pct=args.cvar_pct,
        )

    # Warm-start: enqueue baseline as trial 0 in fresh studies.  Skipped
    # for resumed studies (already have trials) and when the user passes
    # --no-warm-start-baseline.  All baseline values must lie in
    # [low, high] of their BOParam, otherwise enqueue_trial raises.
    if (not args.reparam and not args.no_warm_start_baseline
            and len(study.trials) == 0):
        outside = out_of_box_params(baseline_cfg, ceilings)
        if outside:
            detail = "\n".join(
                f"    {k}={v:g} not in [{lo:g}, {hi:g}]"
                for k, (v, lo, hi) in sorted(outside.items())
            )
            raise SystemExit(
                "[tune] The baseline lies OUTSIDE the search space, so it can "
                "neither warm-start the study nor ever be proposed by a "
                "trial:\n" + detail + "\n"
                "       A search space that cannot express the operating point "
                "you are benchmarking against is a defect in the space, not in "
                "the baseline. Widen the bounds or re-express the space "
                "relative to this point. (Passing --no-warm-start-baseline "
                "silences this, but the optimum remains unreachable.)"
            )
        warm_params = params_from_config(baseline_cfg)
        study.enqueue_trial(warm_params)
        print(
            f"[tune] Warm-start: enqueued baseline params as trial 0 "
            f"({warm_params})",
            flush=True,
        )

    print(f"[tune] Running {args.n_trials} trials ...", flush=True)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=not args.no_progress_bar,
    )

    if args.reparam:
        from tuning.objectives_v2 import (
            best_feasible_trial,
            constraint_violation_report,
        )

        violations = constraint_violation_report(study)
        n_complete = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        print(f"\n[tune] Constraint violations over {n_complete} trials:",
              flush=True)
        for name, count in violations.items():
            print(f"         {name:22s} {count:4d}"
                  f"  ({count / max(n_complete, 1) * 100:5.1f} %)", flush=True)
        try:
            # NOT study.best_trial: it ignores constraints in a single-objective
            # study and would happily report an infeasible point as the answer.
            best = best_feasible_trial(study)
        except RuntimeError as exc:
            raise SystemExit(
                f"[tune] {exc}\n"
                f"       The per-constraint counts above say which limit is "
                f"binding. Relax the one that is actually responsible rather "
                f"than widening all of them."
            )
    else:
        best = study.best_trial

    print(
        f"\n[tune] Best CVaR-{args.cvar_pct:.0f}: {best.value:.6f}",
        flush=True,
    )
    print(f"[tune] Best params: {best.params}", flush=True)

    if not args.reparam:
        # Bound-hugging is diagnosed against BO_DIMS; in the reparameterised
        # space the reference sits at the centre of every ratio coordinate by
        # construction, so the failure mode this warns about cannot arise there.
        _warn_on_bound_hugging(study, ceilings)

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
        # Provenance.  `MIQP_SOLVERS` falls back from GUROBI to SCIP silently,
        # and miqp_solver.py itself records that SCIP returned
        # `optimal_inaccurate` on 54/60 DSO solves -- so results from the two
        # are not comparable and the solver must be part of the record.
        "search_space_fingerprint": search_space_fingerprint(),
        "search_space_version":     SEARCH_SPACE_VERSION,
        "solver":                   _resolve_solver_name(),
        "space":                    "reparam" if args.reparam else "raw_weights",
    }
    if args.reparam:
        from tuning.objectives_v2 import CONSTRAINT_NAMES, constraints_func

        meta["constraints"] = dict(
            zip(CONSTRAINT_NAMES,
                [float(v) for v in constraints_func(best)])
        )
        # The weights are the preference the scalar encodes; without them the
        # recorded objective value cannot be interpreted, let alone reproduced.
        meta["perf_weights_profile"] = args.perf_weights
        meta["perf_weights"] = dataclasses.asdict(perf_weights)
        meta["perf_exclude"] = sorted(perf_exclude)
        meta["scenario_set"] = which
    save_tuned_params(best.params, meta, args.output)
    print(f"[tune] Wrote tuned params -> {args.output}", flush=True)

    if args.reparam:
        # The report writer is built around the legacy certificate-ratio table,
        # which has no meaning in the reparameterised space (the coordinates are
        # ratios, not raw g_w).  Skip rather than emit a misleading table.
        print("[tune] Report skipped in --reparam mode (certificate-ratio "
              "table does not apply to ratio coordinates).", flush=True)
        return 0

    from tuning.reports.tuning_report import write_tuning_report
    write_tuning_report(study, ceilings, args.report)
    print(f"[tune] Wrote report -> {args.report}", flush=True)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
