# -*- coding: utf-8 -*-
"""
experiments/helpers/
====================
Shared helper modules for the experiment entry-point scripts.

Submodules
----------
* :mod:`contingency` -- scheduled network-contingency injection.
* :mod:`plant_io`    -- write controller outputs back into the pandapower
  network (single-TSO cascade + multi-zone variants).
* :mod:`records`     -- data-classes that hold per-iteration simulation
  results (``IterationRecord``, ``MultiTSOIterationRecord``,
  ``CascadeResult``, ``ContingencyEvent``).
* :mod:`utils`       -- small utility functions (``_network_state``,
  ``_build_Gw``, ``_sgen_at_bus``, ``build_der_mapping``,
  ``print_summary``).

All public symbols are re-exported here so callers can do
``from experiments.helpers import <name>`` without knowing which
submodule owns a given class or function.
"""

from .contingency import _apply_contingency, prepare_load_contingencies
from .plant_io import (
    _apply_dso,
    _apply_tso,
    apply_central_controls,
    apply_cos_phi_one_local_control,
    apply_dso_controls,
    apply_qv_local_control,
    apply_zone_tso_controls,
    install_cos_phi_one,
    install_qv_characteristic_controllers,
)
from .records import (
    A2S,
    A3S,
    CascadeResult,
    ContingencyEvent,
    IterationRecord,
    MultiTSOIterationRecord,
)
from .utils import (
    _build_Gw,
    _network_state,
    _sgen_at_bus,
    build_der_mapping,
    print_summary,
)

__all__ = [
    # contingency
    "_apply_contingency",
    "prepare_load_contingencies",
    # plant I/O (cascade)
    "_apply_tso",
    "_apply_dso",
    # plant I/O (multi-zone)
    "apply_zone_tso_controls",
    "apply_dso_controls",
    "apply_central_controls",
    "apply_qv_local_control",
    "apply_cos_phi_one_local_control",
    "install_qv_characteristic_controllers",
    "install_cos_phi_one",
    # records
    "A2S",
    "A3S",
    "CascadeResult",
    "ContingencyEvent",
    "IterationRecord",
    "MultiTSOIterationRecord",
    # utils
    "_build_Gw",
    "_network_state",
    "_sgen_at_bus",
    "build_der_mapping",
    "print_summary",
]
