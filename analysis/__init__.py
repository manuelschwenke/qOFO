"""
Analysis package for the cascaded OFO controller.

Provides:
    stability_analysis  — Three-condition stability framework (Theorem 3.3).
    auto_tune           — Parameter tuning for C1/C2/C3 conditions.
"""

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    analyse_dso_stability,
    analyse_continuous_stability,
    analyse_discrete_small_gain,
    recommend_gw_min,
    MultiZoneStabilityResult,
    DSOStabilityResult,
    ContinuousStabilityResult,
    DiscreteSmallGainResult,
    ZoneStabilityResult,
)
from analysis.auto_tune import (
    auto_tune,
    tune_continuous_gw,
    tune_discrete_gw,
    tune_dso_gw,
    filter_stability_inputs,
    expand_gw_with_excluded,
    TuningResult,
    DSOTuneInput,
)

__all__ = [
    'analyse_multi_zone_stability',
    'analyse_dso_stability',
    'analyse_continuous_stability',
    'analyse_discrete_small_gain',
    'recommend_gw_min',
    'MultiZoneStabilityResult',
    'DSOStabilityResult',
    'ContinuousStabilityResult',
    'DiscreteSmallGainResult',
    'ZoneStabilityResult',
    'auto_tune',
    'tune_continuous_gw',
    'tune_discrete_gw',
    'tune_dso_gw',
    'filter_stability_inputs',
    'expand_gw_with_excluded',
    'TuningResult',
    'DSOTuneInput',
]
