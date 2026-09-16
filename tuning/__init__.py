"""
tuning
======
Weight-campaign support for the cascaded MIQP-OFO controller.

Public modules
--------------
``parameters``      the canonical nine-weight vector and its config overlay
``stage1_screen``   Stage I: the analytic contraction screen (no simulation)
``scenarios``       declarative scenario / window specifications
``metrics``         trajectory metric extraction
``runner``          one (weights, window) evaluation
``bisect_switching``  OLTC weight calibrated against a taps/day budget

The Bayesian-optimisation stack (``tune``, ``objective``, ``objectives_v2``,
``reparam``, ``rawspace``, ``ceilings``, ``stability_certificate``) and the
``tuning_mc`` staged campaign were retired on 2026-09-10 with the weight reset;
they remain in git history.
"""

from tuning.metrics import (
    CostWeights,
    NoiseFloors,
    TrajectoryMetrics,
    extract_metrics,
)
from tuning.parameters import (
    GAUGE_FIELD,
    SEARCHED_FIELDS,
    TIMING_FIELDS,
    WEIGHT_FIELDS,
    apply_to_config,
    params_from_config,
    weight_fingerprint,
)
from tuning.runner import RunResult, run_one, run_scenario_with_records
from tuning.scenarios import ScenarioSpec

__all__ = [
    "CostWeights",
    "GAUGE_FIELD",
    "NoiseFloors",
    "RunResult",
    "SEARCHED_FIELDS",
    "ScenarioSpec",
    "TIMING_FIELDS",
    "TrajectoryMetrics",
    "WEIGHT_FIELDS",
    "apply_to_config",
    "extract_metrics",
    "params_from_config",
    "run_one",
    "run_scenario_with_records",
    "weight_fingerprint",
]
