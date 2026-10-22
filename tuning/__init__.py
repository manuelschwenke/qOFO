"""
tuning
======
Weight-campaign support for the cascaded MIQP-OFO controller.

Public modules
--------------
``parameters``      the canonical nine-weight vector and its config overlay
``stage1_screen``   Stage I: the analytic contraction screen (no simulation)
``scenarios``       declarative scenario / window specifications
``scenarios_mc_v2`` the W1-W6 condition bank and the tier-1 design windows
``metrics``         trajectory metric extraction
``runner``          one (weights, window) evaluation
``bisect_switching``  OLTC weight calibrated against a taps/day budget
``coupled_spectral_radius``  contraction of the assembled coupled operator

The Bayesian-optimisation stack (``tune``, ``objective``, ``objectives_v2``,
``reparam``, ``rawspace``, ``ceilings``, ``stability_certificate``) and the
staged ``tuning_mc`` campaign (``stage_0_preconditioning``,
``stage_0_coupling_decomposition``, ``stage_1_search``, ``select_windows_v2``,
``scenarios_mc``) were retired on 2026-09-10 with the weight reset; they remain
in git history.

The ``tuning_mc`` package itself was removed on 2026-10-22.  Its two surviving
modules -- the scenario bank ``scenarios_mc_v2`` and the diagnostic
``coupled_spectral_radius`` -- are the ones listed above: both already imported
``tuning.scenarios`` and ``tuning._sim_loader``, while ``tuning.stage1_screen``
imported the bank back out of ``tuning_mc``, so the two packages were mutually
dependent and only the function-local placement of that import kept the cycle
from biting.
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
