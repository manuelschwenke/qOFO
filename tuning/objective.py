"""
tuning/objective.py
===================
Optuna objective function: one trial = one evaluation across the design
set, aggregated as CVaR-25 (mean of the worst 25 % of scenario costs).

Why CVaR-25
-----------
* Mean is too optimistic: one bad scenario averages out.
* Max is too noisy: one outlier dominates.
* The 25th-percentile worst-case is the standard robust-optimisation
  choice and matches the "safe across the operating envelope" goal.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import optuna

from configs.multi_tso_config import MultiTSOConfig
from tuning._types import Ceilings
from tuning.metrics import CostWeights, NoiseFloors
from tuning.parameters import BO_DIMS, resolve_high
from tuning.runner import RunResult, run_one
from tuning.scenarios import ScenarioSpec


def cvar_aggregate(values: List[float], pct: float = 25.0) -> float:
    """Mean of the worst (largest) ``pct`` percent of ``values``.

    For minimisation, "worst" = largest.  For ``pct=25`` this returns
    the mean of the largest 25 % of values, i.e. ``CVaR_{0.75}``.
    Returns ``inf`` for empty input so a vacuous trial is treated as
    pessimal.
    """
    if not values:
        return float("inf")
    arr = np.asarray(values, dtype=float)
    k = max(1, int(np.ceil(arr.size * pct / 100.0)))
    return float(np.mean(np.sort(arr)[-k:]))


def sample_params(
    trial: optuna.Trial,
    ceilings: Ceilings | None,
) -> dict[str, float]:
    """Sample one BO-param dict from an Optuna ``trial``.

    Reads :data:`BO_DIMS` for bounds and log/linear, resolving
    ``"ceil"`` placeholders against ``ceilings``.
    """
    params: dict[str, float] = {}
    for p in BO_DIMS:
        high = resolve_high(p, ceilings)
        if p.log:
            params[p.name] = trial.suggest_float(p.name, p.low, high, log=True)
        else:
            params[p.name] = trial.suggest_float(p.name, p.low, high)
    return params


def make_objective(
    baseline_cfg: MultiTSOConfig,
    ceilings: Ceilings | None,
    design_scenarios: List[ScenarioSpec],
    cost_weights: CostWeights | None = None,
    noise_floors: NoiseFloors | None = None,
    n_jobs: int = 1,
    cvar_pct: float = 25.0,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function closing over the baseline
    config and the design scenarios.

    Returns a callable ``objective(trial) -> float`` that:

    1. Samples params from :data:`BO_DIMS`.
    2. Runs :func:`run_one` over every scenario in ``design_scenarios``,
       in parallel if ``n_jobs > 1`` (process-based, since
       ``run_multi_tso_dso`` does not release the GIL).
    3. Records per-scenario diagnostics on the trial via
       :meth:`optuna.Trial.set_user_attr`.
    4. Returns the CVaR aggregate of ``cost_J`` across scenarios.

    Notes
    -----
    ``n_jobs > 1`` requires :mod:`joblib`'s loky backend and has been
    seen to occasionally interfere with pandapower's solver setup; keep
    ``n_jobs=1`` until smoke-tested in your specific environment.
    """
    cost_weights = cost_weights or CostWeights()
    noise_floors = noise_floors or NoiseFloors()

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, ceilings)

        if n_jobs <= 1:
            results: List[RunResult] = [
                run_one(params, sc, baseline_cfg, cost_weights, noise_floors)
                for sc in design_scenarios
            ]
        else:
            from joblib import Parallel, delayed
            results = list(Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(run_one)(
                    params, sc, baseline_cfg, cost_weights, noise_floors,
                )
                for sc in design_scenarios
            ))

        # per-scenario diagnostics
        for r in results:
            trial.set_user_attr(f"J__{r.scenario_name}",
                                float(r.metrics.cost_J))
            trial.set_user_attr(f"wall_s__{r.scenario_name}",
                                float(r.wall_time_s))
            trial.set_user_attr(f"pf_fail__{r.scenario_name}",
                                int(r.metrics.pf_failures))
            if r.failure_reason:
                trial.set_user_attr(f"err__{r.scenario_name}",
                                    r.failure_reason[:500])

        # aggregate scenario costs
        Js = [float(r.metrics.cost_J) for r in results]
        agg = cvar_aggregate(Js, pct=cvar_pct)
        trial.set_user_attr("cvar_J", agg)
        trial.set_user_attr("mean_J", float(np.mean(Js)) if Js else float("inf"))
        trial.set_user_attr("max_J",  float(np.max(Js))  if Js else float("inf"))
        return agg

    return objective
