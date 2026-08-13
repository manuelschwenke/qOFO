r"""
pf/timescale_study.py
=====================
**Moved 2026-08-06.** The timescale settling battery now lives with the other
Ch. 9 parameter-selection entry points:

    experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py

This module is a forwarding shim so the command in
``docs/handover_timescale_study.md`` and in the Chapter 9 source comment keeps
working. It holds no logic of its own -- there is one implementation, and it
is the one above. New work goes there; this file may be deleted once the
thesis text no longer names it.

The PowerFactory driver infrastructure the study uses (``ScreeningContext``,
the step catalogues, ``settling_metrics``) stays in ``pf/screening.py``: it is
shared with the RMS build and the dead-band study.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ch_9_parameter_selection.ch_9_1_timescale_seperation import (  # noqa: E402,F401
    BAND_Q_MVAR,
    BAND_VOLTAGE_PU,
    TABLE_ROWS,
    T_DS_S,
    T_TS_S,
    build_table,
    derive,
    main as _main,
    preflight,
    run_case,
    write_outputs,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print("[timescale] NOTE: pf/timescale_study.py has moved to "
          "experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py; "
          "this shim forwards to it.")
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
