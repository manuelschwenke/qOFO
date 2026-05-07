# -*- coding: utf-8 -*-
"""
experiments/runners/
====================
Shared simulation entry-point functions for the experiment scripts.

The experiment scripts under :mod:`experiments` (000_M_TSO_M_DSO,
001_S_TSO_S_DSO, 002_M_TSO_M_DSO_COMPARE, 003_M_DSO_CIGRE_2026) all start
with a digit, which prevents them from being imported as Python modules.
The reusable simulation logic therefore lives here so any script can
``from experiments.runners import run_multi_tso_dso, run_cascade`` cleanly.

Runners
-------
* :func:`run_multi_tso_dso` -- multi-TSO / multi-DSO OFO on the IEEE 39-bus
  network with N=3 zones and per-zone HV sub-networks.  Used by 000, 002,
  and 003.
* :func:`run_cascade` -- cascaded single-TSO / single-DSO OFO on the
  combined TUDA benchmark network.  Used by 001.
"""

from experiments.runners.multi_tso_dso import run_multi_tso_dso
from experiments.runners.cascade import run_cascade

__all__ = ["run_multi_tso_dso", "run_cascade"]
