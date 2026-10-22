#!/usr/bin/env python3
r"""Inspect what a machine AVR composite looks like, so one can be built for G 01.

G 01 (the 10 GVA 'Rest of U.S.A. / Canada' equivalent) is a bare ``ElmSym``
with no controller blocks: in RMS it runs at constant excitation, so its
terminal voltage floats, while the static plant models it as the pandapower
SLACK with ``vm_pu`` pinned at 1.03.  That asymmetry lifts zone 1 and skews the
whole reactive balance (run 0047: bus 38 drifts 1.0292 -> 1.0343 pu).

This is READ-ONLY: it dumps the frame, slot order and block inventory of an
existing machine composite (the template to copy) plus G 01's own handles.
Run it only when no closed-loop run is in flight -- a second PF session kills
the running one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.screening import RMS_STUDY_CASE  # noqa: E402
from pf.session import connect, get_all  # noqa: E402


def full(o) -> str:
    try:
        return o.GetFullName()
    except Exception:  # noqa: BLE001
        return repr(o)


def main() -> int:
    app = connect(study_case=RMS_STUDY_CASE)

    # ---- G 01 itself -------------------------------------------------
    g01 = next((m for m in get_all(app, "ElmSym")
                if m.loc_name.strip() in ("G 01", "G01")), None)
    print("=== G 01 ===")
    if g01 is None:
        print("  NOT FOUND")
        return 1
    print(f"  {full(g01)}")
    for attr in ("outserv", "ngnum", "usetp", "pgini", "qgini", "c_pmod"):
        try:
            v = g01.GetAttribute(attr)
            print(f"    {attr:8s} = {full(v) if hasattr(v, 'GetFullName') else v}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {attr:8s} : {exc}")
    cub = g01.GetAttribute("bus1")
    term = cub.cterm if cub is not None else None
    print(f"    cubicle  = {full(cub) if cub else None}")
    print(f"    terminal = {full(term) if term else None}")

    # G 01's OWN composite: it exists ("Rest of U.S.A. / Canada") but carries
    # no AVR -- dump its frame/slots/pelm so the empty slot can be identified.
    own = g01.GetAttribute("c_pmod")
    if own is not None:
        print(f"\n=== G 01 composite: {full(own)} ===")
        frame = own.GetAttribute("typ_id")
        print(f"  frame : {full(frame) if frame else None}")
        if frame is not None:
            try:
                slots = [s.loc_name for s in frame.GetContents()
                         if s.GetClassName() == "BlkSlot"]
                print(f"  slots : {slots}")
            except Exception as exc:  # noqa: BLE001
                print(f"  slots : (read failed: {exc})")
        pelm = own.GetAttribute("pelm") or []
        print(f"  pelm  : {[(e.loc_name if e is not None else '<EMPTY>') for e in pelm]}")
        print("  contents:")
        for d in own.GetContents():
            print(f"    {d.GetClassName():12s} {d.loc_name}")

    # ---- a template composite (a machine that HAS an AVR) ------------
    print("\n=== machine composites (templates) ===")
    for comp in get_all(app, "ElmComp"):
        pelm = comp.GetAttribute("pelm") or []
        sym = next((e for e in pelm if e is not None
                    and e.GetClassName() == "ElmSym"), None)
        if sym is None:
            continue
        frame = comp.GetAttribute("typ_id")
        print(f"\n  composite : {full(comp)}")
        print(f"    machine : {sym.loc_name}")
        print(f"    frame   : {full(frame) if frame else None}")
        # slot order == pelm order; print both so the mapping is unambiguous
        slots = []
        if frame is not None:
            try:
                slots = [s.loc_name for s in frame.GetContents()
                         if s.GetClassName() == "BlkSlot"]
            except Exception as exc:  # noqa: BLE001
                print(f"    (slot read failed: {exc})")
        print(f"    slots   : {slots}")
        print(f"    pelm    : {[ (e.loc_name if e is not None else None) for e in pelm ]}")
        print("    contents:")
        for d in comp.GetContents():
            cls = d.GetClassName()
            extra = ""
            if cls == "ElmDsl":
                t = d.GetAttribute("typ_id")
                extra = f"  typ={full(t) if t else None}"
            print(f"      {cls:12s} {d.loc_name}{extra}")

    print("\n=== AVR DSL detail (first found) ===")
    for comp in get_all(app, "ElmComp"):
        avr = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl"
                    and "avr" in d.loc_name.lower()), None)
        if avr is None:
            continue
        print(f"  {full(avr)}")
        t = avr.GetAttribute("typ_id")
        print(f"    typ_id  = {full(t) if t else None}")
        for attr in ("params", "sInput", "sOutput"):
            try:
                print(f"    {attr:8s} = {avr.GetAttribute(attr)}")
            except Exception as exc:  # noqa: BLE001
                print(f"    {attr:8s} : {exc}")
        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
