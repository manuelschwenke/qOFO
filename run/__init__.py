# -*- coding: utf-8 -*-
"""
run/__init__.py
===============
Public API for the ``run`` package.

Re-exports everything that was previously importable from the root
``run_cascade`` module so that ``from run_cascade import X`` continues to
work via the backward-compatibility shim at the project root.
"""

from .contingency import _apply_contingency
from .helpers import _build_Gw, _network_state, _sgen_at_bus, print_summary
from .plant_io import _apply_dso, _apply_tso
from .records import (
    A2S,
    A3S,
    CascadeResult,
    ContingencyEvent,
    IterationRecord,
)
from .run_cascade import main, run_cascade

__all__ = [
    # lambdas
    "A2S",
    "A3S",
    # data-classes
    "ContingencyEvent",
    "IterationRecord",
    "CascadeResult",
    # contingency helper (keep underscore for backward compat)
    "_apply_contingency",
    # plant I/O helpers
    "_apply_tso",
    "_apply_dso",
    # network helpers
    "_network_state",
    "_sgen_at_bus",
    "_build_Gw",
    # summary / entry-point
    "print_summary",
    "run_cascade",
    "main",
]
