"""
tuning
======
Bayesian-optimization tuning system for the multi-TSO/DSO OFO controller.

Public modules:
    parameters -- BO search-space declaration and Config overlay.
    ceilings   -- One-shot LMI ceiling extraction from
                  ``analyse_multi_zone_stability``.
    scenarios  -- Declarative design / validation scenario sets.
    metrics    -- Trajectory metric extraction and composite cost.
    runner     -- One-trial wrapper around ``run_multi_tso_dso``.

Task 1: parameters + ceilings.
Task 2: scenarios + metrics + runner.
Task 3 (later): objective, CLI, BO library integration.
"""

from tuning._types import BOParam, Ceilings
from tuning.metrics import (
    CostWeights,
    NoiseFloors,
    TrajectoryMetrics,
    extract_metrics,
)
from tuning.parameters import (
    BO_DIMS,
    FIXED_OVERRIDES,
    apply_to_config,
    params_from_config,
    resolve_high,
    search_space,
)
from tuning.runner import RunResult, run_one
from tuning.scenarios import ScenarioSpec, design_set, validation_set

__all__ = [
    "BOParam",
    "Ceilings",
    "BO_DIMS",
    "FIXED_OVERRIDES",
    "CostWeights",
    "NoiseFloors",
    "RunResult",
    "ScenarioSpec",
    "TrajectoryMetrics",
    "apply_to_config",
    "design_set",
    "extract_metrics",
    "params_from_config",
    "resolve_high",
    "run_one",
    "search_space",
    "validation_set",
]
