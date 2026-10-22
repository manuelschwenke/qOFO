"""
pf/wecc_introspect.py
=====================
Dump the full structure and parameters of a WECC composite model, and probe
its RMS initialisation + Q-reference response.

Run this once you (the user) have built and verified ONE WECC composite in
the GUI on a wind park.  It captures everything I need to replicate the
composite across the other parks via the API (copy + re-point + rescale):

* the composite frame and its ``pelm`` slot fillers,
* every DSL block (REGC / REEC / REPC), its block-definition type and all
  its settable parameters,
* candidate plant Q-reference signals (the OFO write handle that replaces
  the load-flow-only ``qsetp``).

Usage (PF machine, after the GUI build)::

    python pf\\wecc_introspect.py                 # auto-find WECC composites
    python pf\\wecc_introspect.py --name WECC_WP_TSO_s0_b18

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    connect,
    deactivate_variations_except,
    get_all,
    set_variation_active,
)

RMS_STUDY_CASE = "02_RMS_CoSim"

#: Attribute-name fragments that flag a reactive-power reference / setpoint.
_QREF_HINTS = ("qref", "qext", "vref", "qset", "refq", "q_ref", "pfref")


def _activate_full(app) -> None:
    deactivate_variations_except(app, keep=None)
    set_variation_active(app, "wind_replace", True)
    set_variation_active(app, "full", True)
    nd = app.GetProjectFolder("netdat")
    for g in nd.GetContents("*.ElmNet"):
        if g.loc_name.startswith("DSO_") and not g.IsCalcRelevant():
            g.Activate()


def _dump_dsl(dsl, indent: str = "    ") -> None:
    typ = dsl.GetAttribute("typ_id")
    print(f"{indent}ElmDsl {dsl.loc_name!r}  block={typ.loc_name if typ else None}")
    # Scalar parameter names belong to the block definition.  ``dsl.params``
    # is the numeric value vector and must not be interpreted as names.
    names: List[str] = []
    for owner in (typ, dsl):
        if owner is None:
            continue
        try:
            raw = owner.GetAttribute("sParams")
        except Exception:
            continue
        if not raw:
            continue
        entries = [raw] if isinstance(raw, str) else list(raw)
        for entry in entries:
            if not isinstance(entry, str):
                continue
            names.extend(
                token.strip()
                for token in entry.split(",")
                if token.strip()
            )
        if names:
            break
    for p in names:
        try:
            print(f"{indent}    {p} = {dsl.GetAttribute(p)!r}")
        except Exception:
            pass
    # Q-reference candidates by attribute-name hint.
    for p in names:
        if any(h in p.lower() for h in _QREF_HINTS):
            print(f"{indent}    >>> Q-ref candidate: {dsl.loc_name}.{p}")


def introspect(app, comp) -> None:
    print("=" * 70)
    print(f"Composite model: {comp.loc_name!r}")
    frame = comp.GetAttribute("typ_id")
    print(f"  frame: {frame.loc_name if frame else None}")
    slots = [o for o in frame.GetContents() if o.GetClassName() == "BlkSlot"] \
        if frame else []
    pelm = comp.GetAttribute("pelm")
    print(f"  slots / fillers ({len(pelm)}):")
    for i, filler in enumerate(pelm):
        slot_name = slots[i].loc_name if i < len(slots) else f"slot{i}"
        cls = filler.GetClassName() if filler else "-"
        nm = filler.loc_name if filler else None
        print(f"    [{i}] {slot_name:22s} = {nm} ({cls})")
    print("  DSL blocks:")
    for dsl in [o for o in comp.GetContents() if o.GetClassName() == "ElmDsl"]:
        _dump_dsl(dsl)
    # The generator this composite drives.
    gen = pelm[0] if pelm else None
    if gen is not None:
        print(f"  generator: {gen.loc_name} sgn="
              f"{gen.GetAttribute('sgn')} MVA av_mode={gen.GetAttribute('av_mode')}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dump a WECC composite spec.")
    parser.add_argument("--name", default=None,
                        help="composite loc_name (default: all WECC composites)")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    args = parser.parse_args(argv)

    app = connect(args.project, study_case=RMS_STUDY_CASE)
    _activate_full(app)

    comps = get_all(app, "ElmComp")
    if args.name:
        comps = [c for c in comps if c.loc_name == args.name]
    else:
        comps = [c for c in comps
                 if re.search(r"wecc|regc|reec|repc|wp|der", c.loc_name, re.I)
                 and c.loc_name not in ("Power Plant 03",)]
        # exclude the template machine plants
        comps = [c for c in comps if not c.loc_name.startswith("Power Plant")
                 and c.loc_name not in ("Rest of U.S.A. / Canada",)]
    if not comps:
        print("No WECC composite found. Build one in the GUI first, or pass "
              "--name. Existing ElmComp: "
              + ", ".join(c.loc_name for c in get_all(app, "ElmComp")))
        return 1
    for c in comps:
        introspect(app, c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
