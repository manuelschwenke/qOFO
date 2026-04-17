# -*- coding: utf-8 -*-
"""
experiments/__init__.py
=======================
Public API for the ``experiments`` package.

Experiment entry-point scripts (``000_M_TSO_M_DSO.py``, ``001_S_TSO_S_DSO.py``)
start with a digit and therefore cannot be imported as Python modules --
they are invoked directly, e.g. ``python experiments/000_M_TSO_M_DSO.py``.
All re-usable symbols (data-classes, contingency helpers, plant I/O) live in
the sibling sub-modules below and are re-exported here.
"""

from .contingency import _apply_contingency, prepare_load_contingencies
from .helpers import _build_Gw, _network_state, _sgen_at_bus, build_der_mapping, print_summary
from .plant_io import _apply_dso, _apply_tso
from .records import (
    A2S,
    A3S,
    CascadeResult,
    ContingencyEvent,
    IterationRecord,
    MultiTSOIterationRecord,
)

__all__ = [
    # lambdas
    "A2S",
    "A3S",
    # data-classes
    "ContingencyEvent",
    "IterationRecord",
    "CascadeResult",
    "MultiTSOIterationRecord",
    # contingency helpers
    "_apply_contingency",
    "prepare_load_contingencies",
    # plant I/O helpers
    "_apply_tso",
    "_apply_dso",
    # network helpers
    "_network_state",
    "_sgen_at_bus",
    "_build_Gw",
    "build_der_mapping",
    # summary
    "print_summary",
]
