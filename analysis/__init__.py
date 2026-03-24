"""
Analysis package for the cascaded OFO controller.

Provides:
    stability_analysis    — Lipschitz / SVD-based stability bounds.
    run_stability_tuda    — Integration script for the TUDa network.
"""

from analysis.stability_analysis import (
    analyse_stability,
    recommend_gw_min,
    CascadeStabilityResult,
    ControllerStabilityResult,
)
from analysis.run_stability_tuda import run_stability_analysis

__all__ = [
    'analyse_stability',
    'recommend_gw_min',
    'run_stability_analysis',
    'CascadeStabilityResult',
    'ControllerStabilityResult',
]
