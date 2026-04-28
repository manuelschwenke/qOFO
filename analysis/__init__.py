"""
Analysis package for the cascaded OFO controller.

Provides:
    stability_analysis  -- Three-condition stability framework (Theorem 3.3).
"""

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    analyse_dso_stability,
    analyse_continuous_stability,
    analyse_discrete_small_gain,
    MultiZoneStabilityResult,
    DSOStabilityResult,
    ContinuousStabilityResult,
    DiscreteSmallGainResult,
    ZoneStabilityResult,
)

__all__ = [
    'analyse_multi_zone_stability',
    'analyse_dso_stability',
    'analyse_continuous_stability',
    'analyse_discrete_small_gain',
    'MultiZoneStabilityResult',
    'DSOStabilityResult',
    'ContinuousStabilityResult',
    'DiscreteSmallGainResult',
    'ZoneStabilityResult',
]
