"""
pf/hello_pf.py
==============
Phase-1 manual smoke test for the PowerFactory Python API
(docs/RMS_IEEE39_PowerFactory_Build_Plan.md, Gate 1).

Run this ON THE PF MACHINE (see docs/pf_api_notes.md for setup)::

    python pf\hello_pf.py                      # default project qOFO\IEEE39_qOFO
    python pf\hello_pf.py "qOFO\IEEE39_qOFO"   # explicit project path

It prints, in order:
  1. Interpreter + powerfactory module provenance.
  2. Active project and the list of study cases (so the co-sim study case
     layout can be created/checked).
  3. All synchronous machines (ElmSym) with their loc_names -- needed to
     verify pf/naming.py::TEMPLATE_MACHINE_NAMES.
  4. Load-flow (ComLdf) execution on the active study case and the first
     few terminal voltages.
  5. The ComLdf load-voltage-dependency flag state (parity-relevant).

Paste the full console output back into the repo discussion; on success,
record the Python version pin in docs/pf_api_notes.md and set
TEMPLATE_NAMES_VERIFIED = True in pf/naming.py once the machine names are
confirmed.

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo importable when the script is started from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    get_all,
    get_attr,
    list_study_cases,
    run_ldf,
)


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    project = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROJECT_PATH

    print("=" * 70)
    print("qOFO PowerFactory hello-world (Phase 1, Gate 1)")
    print("=" * 70)
    print(f"[1] Python: {sys.version}")

    app = connect(project)
    try:
        pf_version = app.GetAttribute("e:version")
    except Exception:
        pf_version = "<read via Help->About and record manually>"
    print(f"[1] PowerFactory version attribute: {pf_version}")

    prj = app.GetActiveProject()
    print(f"[2] Active project: {prj.GetFullName()}")
    cases = list_study_cases(app)
    print(f"[2] Study cases ({len(cases)}):")
    for c in cases:
        print(f"      - {c.loc_name}")

    machines = get_all(app, "ElmSym")
    print(f"[3] Synchronous machines ({len(machines)}) -- verify against "
          f"pf/naming.py TEMPLATE_MACHINE_NAMES:")
    for m in sorted(machines, key=lambda o: o.loc_name):
        print(f"      - {m.loc_name!r}  (outserv={get_attr(m, 'outserv')})")

    print("[4] Executing ComLdf on the active study case ...")
    ldf = run_ldf(app)
    terms = get_all(app, "ElmTerm")
    print(f"[4] Load flow converged; {len(terms)} terminals. First five:")
    for t in sorted(terms, key=lambda o: o.loc_name)[:5]:
        try:
            u = get_attr(t, "m:u")
        except PFSessionError:
            u = float("nan")
        print(f"      - {t.loc_name}: u = {u:.5f} pu")

    # Parity-relevant ComLdf option: voltage dependency of loads.  The
    # attribute name differs between releases; probe the known candidates
    # and report whichever exists.
    print("[5] ComLdf load-voltage-dependency flag:")
    for attr in ("iopt_pq", "i_power"):
        try:
            print(f"      - ComLdf.{attr} = {ldf.GetAttribute(attr)!r}")
        except Exception:
            print(f"      - ComLdf.{attr}: <not present in this release>")

    print("=" * 70)
    print("SUCCESS -- record the Python pin + machine names, then Gate 1 "
          "is complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PFSessionError as exc:
        print(f"\nPFSessionError: {exc}", file=sys.stderr)
        sys.exit(2)
