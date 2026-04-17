# -*- coding: utf-8 -*-
"""
experiments/__init__.py
=======================
Public API for the ``experiments`` package.

Experiment entry-point scripts (``000_M_TSO_M_DSO.py``,
``001_S_TSO_S_DSO.py``) start with a digit and therefore cannot be imported
as Python modules -- they are invoked directly, e.g.
``python experiments/000_M_TSO_M_DSO.py``.  All re-usable symbols
(data-classes, contingency helpers, plant I/O, utilities) live in the
:mod:`experiments.helpers` sub-package and are re-exported here so
external callers (visualisation, configs, core.results_storage) can keep
using ``from experiments import <name>`` unchanged.
"""

from experiments.helpers import (
    A2S,
    A3S,
    CascadeResult,
    ContingencyEvent,
    IterationRecord,
    MultiTSOIterationRecord,
    _apply_contingency,
    _apply_dso,
    _apply_tso,
    _build_Gw,
    _network_state,
    _sgen_at_bus,
    apply_cos_phi_one_local_control,
    apply_dso_controls,
    apply_qv_local_control,
    apply_zone_tso_controls,
    build_der_mapping,
    prepare_load_contingencies,
    print_summary,
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
    # plant I/O helpers (cascade)
    "_apply_tso",
    "_apply_dso",
    # plant I/O helpers (multi-zone)
    "apply_zone_tso_controls",
    "apply_dso_controls",
    "apply_qv_local_control",
    "apply_cos_phi_one_local_control",
    # network helpers
    "_network_state",
    "_sgen_at_bus",
    "_build_Gw",
    "build_der_mapping",
    # summary
    "print_summary",
]
