#!/usr/bin/env python3
"""DEPRECATED shim -- use ``run_comparison_rms_cosim_qss.py``.

Renamed 2026-07-31.  The old name was a misnomer: the script never replayed
anything.  It runs ``run_multi_tso_dso`` twice as two *independent closed
loops* (one per plant) and compares the results, i.e. a co-simulation
comparison.  The genuine replay is ``run_openloop_qss_to_rms.py``, which
applies the quasi-static run's recorded actuator timeline to the RMS plant.

Kept so existing shell history and notes keep working; delegates verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_comparison_rms_cosim_qss import main  # noqa: E402,F401

if __name__ == "__main__":
    print("[deprecated] run_rms_phase6_replay.py -> "
          "run_comparison_rms_cosim_qss.py (RMS-only: run_rms_cosim.py)\n")
    raise SystemExit(main())
