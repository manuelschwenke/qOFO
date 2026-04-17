"""
configs package
===============
Central configuration dataclasses for experiment runners.

Exports
-------
MultiTSOConfig  — multi-zone TSO / multi-DSO experiment (experiments/000_M_TSO_M_DSO.py).
CascadeConfig   — single-TSO single-DSO cascade experiment (experiments/001_S_TSO_S_DSO.py).
"""

from configs.multi_tso_config import MultiTSOConfig
from configs.cascade_config import CascadeConfig

__all__ = [
    "MultiTSOConfig",
    "CascadeConfig",
]
