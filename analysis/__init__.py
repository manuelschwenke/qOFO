"""
Analysis package for the cascaded OFO controller.

Currently provides:
    stability_analysis  — Lipschitz / SVD-based stability bounds.
"""

from analysis.stability_analysis import (
    analyse_stability,
    recommend_gw_min,
    CascadeStabilityResult,
    ControllerStabilityResult,
)

__all__ = [
    'analyse_stability',
    'recommend_gw_min',
    'CascadeStabilityResult',
    'ControllerStabilityResult',
]
