"""
pf/setup_study_cases.py
=======================
Create the co-simulation study cases in IEEE39_qOFO as copies of the
pristine template case ``1. Power Flow`` (copies inherit the grid
activation configuration; active variations are stored per study case,
which is the whole point of separating them -- see docs/pf_api_notes.md §3).

Created (idempotent -- existing cases are left untouched):

* ``01_LDF_Parity``  -- pf_sync / pf_parity target (Gates A-C).
* ``02_RMS_CoSim``   -- RMS initialisation + OFO-in-the-loop (Phases 5-6).

The template case remains the active one afterwards; nothing else in the
project is modified.

Run on the PF machine::

    python pf\\setup_study_cases.py

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    list_study_cases,
)

TEMPLATE_CASE = "1. Power Flow"
NEW_CASES = ("01_LDF_Parity", "02_RMS_CoSim")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = connect(DEFAULT_PROJECT_PATH)

    folder = app.GetProjectFolder("study")
    cases = {c.loc_name: c for c in list_study_cases(app)}
    if TEMPLATE_CASE not in cases:
        raise PFSessionError(
            f"Template study case {TEMPLATE_CASE!r} not found; available: "
            f"{sorted(cases)}"
        )
    template = cases[TEMPLATE_CASE]

    for name in NEW_CASES:
        if name in cases:
            print(f"[setup] study case {name!r} already exists -- skipped")
            continue
        copy = folder.AddCopy(template, name)
        if copy is None:
            raise PFSessionError(
                f"AddCopy({TEMPLATE_CASE!r} -> {name!r}) returned None"
            )
        # AddCopy may or may not apply the requested name depending on
        # release; enforce it explicitly.
        if copy.loc_name != name:
            copy.loc_name = name
        print(f"[setup] created study case {name!r} "
              f"(copy of {TEMPLATE_CASE!r})")

    # Leave the template case active (AddCopy does not change activation,
    # but assert the invariant rather than assuming it).
    active = app.GetActiveStudyCase()
    active_name = active.loc_name if active is not None else None
    if active_name != TEMPLATE_CASE:
        raise PFSessionError(
            f"Active study case is {active_name!r} after setup; expected "
            f"{TEMPLATE_CASE!r} to remain active"
        )

    print("[setup] final study-case inventory:")
    for c in list_study_cases(app):
        print(f"    - {c.loc_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
