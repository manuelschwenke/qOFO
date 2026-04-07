"""
Analysis package for the cascaded OFO controller.

Provides:
    stability_analysis    — Lipschitz / SVD-based stability bounds.
    (run_stability_tuda removed — not present in this branch)
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
