#!/usr/bin/env python3
r"""H2: give G 01 an AVR in the RMS model (voltage-regulation parity).

Why
---
``G 01`` is the 10 GVA 'Rest of U.S.A. / Canada' equivalent.  The two Gate-E
plants model it incompatibly:

* static (pandapower): ``gen[9]`` @ bus 40 with ``slack=True``/``vm_pu=1.03``
  -- an ideal stiff slack: terminal voltage pinned at 1.03, unlimited Q;
* RMS (PowerFactory): its composite ``Rest of U.S.A. / Canada`` uses the same
  ``SYM Frame_no droop`` frame as every other plant, but **every controller
  slot is empty** -- constant excitation, so the terminal voltage floats.

Measured in run 0047: its HV bus drifts 1.0292 -> 1.0343 pu (+0.0043), which
matches the +0.0044 pu static-vs-RMS gap at that bus and the +0.0043 pu TS
zone-1 gap.  An aggregate of many real machines would regulate voltage, so
filling the AVR slot is both the physical and the parity-restoring fix.

What this does
--------------
Copies a working ``avr_IEEET1`` DSL from a reference plant into G 01's own
composite, names it ``AVR 01`` and binds it to the ``Avr Slot``.  The Gov/Pss
slots are deliberately left EMPTY so this change isolates the *voltage*
asymmetry; the governor/frequency counterpart is a separate hypothesis.

Copying an existing, initialised DSL (rather than creating one from the
BlkDef) avoids the dead-parameter-table failure mode seen during the QVPRE
rollout.  Slots are bound BY NAME through ``ElmComp.pblk`` -- ``pelm`` is
ordered by ``pblk``, not by the frame's ``GetContents``, and an index-based
write can silently evict the generator.

Usage::

    python -X utf8 -m pf.add_g01_avr --verify     # report only
    python -X utf8 -m pf.add_g01_avr --apply
    python -X utf8 -m pf.add_g01_avr --apply --smoke 60
    python -X utf8 -m pf.add_g01_avr --revert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.screening import RMS_STUDY_CASE, ScreeningContext  # noqa: E402
from pf.session import PFSessionError, connect, get_all  # noqa: E402

G01_NAMES = ("G 01", "G01")
AVR_NAME = "AVR 01"
AVR_SLOT = "Avr Slot"
SYM_SLOT = "Sym Slot"
REF_AVR_HINT = "AVR 02"          # template to copy (any working avr_IEEET1)


def _full(o) -> str:
    try:
        return o.GetFullName()
    except Exception:  # noqa: BLE001
        return repr(o)


def _find_g01(app):
    g = next((m for m in get_all(app, "ElmSym")
              if m.loc_name.strip() in G01_NAMES), None)
    if g is None:
        raise PFSessionError("G 01 ElmSym not found")
    comp = g.GetAttribute("c_pmod")
    if comp is None:
        raise PFSessionError("G 01 has no composite (c_pmod)")
    return g, comp


def _find_template_avr(app):
    """A working avr_IEEET1 ElmDsl from another plant."""
    best = None
    for comp in get_all(app, "ElmComp"):
        for d in comp.GetContents():
            if d.GetClassName() != "ElmDsl":
                continue
            if "avr" not in d.loc_name.lower():
                continue
            if REF_AVR_HINT in d.loc_name:
                return d
            best = best or d
    if best is None:
        raise PFSessionError("no template AVR ElmDsl found")
    return best


def _slot_report(comp):
    pblk = comp.GetAttribute("pblk") or []
    pelm = comp.GetAttribute("pelm") or []
    return [(s.loc_name if s is not None else "?",
             e.loc_name if e is not None else None)
            for s, e in zip(pblk, pelm)]


def _existing_avr(comp):
    return next((d for d in comp.GetContents()
                 if d.GetClassName() == "ElmDsl"
                 and d.loc_name.strip() == AVR_NAME), None)


def _bind_slots(comp, sym, avr):
    """Bind Sym/Avr slots by NAME via pblk; every other slot stays empty."""
    pblk = comp.GetAttribute("pblk") or []
    pelm = []
    for slot in pblk:
        nm = slot.loc_name if slot is not None else None
        if nm == SYM_SLOT:
            pelm.append(sym)
        elif nm == AVR_SLOT:
            pelm.append(avr)
        else:
            pelm.append(None)          # Gov/Pss/Uel/Oel/MeasBus1 stay empty
    comp.SetAttribute("pelm", pelm)
    return pelm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--smoke", type=float, default=0.0,
                    help="after applying, run a flat RMS of N s and report "
                         "whether G 01's bus holds voltage")
    args = ap.parse_args(argv)
    if not (args.apply or args.revert or args.verify):
        args.verify = True

    app = connect(study_case=RMS_STUDY_CASE)
    g01, comp = _find_g01(app)
    print(f"G 01      : {_full(g01)}")
    print(f"composite : {_full(comp)}")
    print(f"slots now : {_slot_report(comp)}")

    if args.revert:
        avr = _existing_avr(comp)
        if avr is None:
            print("revert: no AVR 01 present -- nothing to do")
        else:
            _bind_slots(comp, g01, None)
            avr.Delete()
            print("revert: AVR 01 deleted and Avr Slot cleared")
            print(f"slots now : {_slot_report(comp)}")
        return 0

    if args.verify and not args.apply:
        avr = _existing_avr(comp)
        print(f"verify    : AVR 01 present = {avr is not None}")
        if avr is not None:
            print(f"            {_full(avr)}")
            print(f"            typ_id = {_full(avr.GetAttribute('typ_id'))}")
        return 0

    # ---- apply -------------------------------------------------------
    avr = _existing_avr(comp)
    if avr is not None:
        print("apply     : AVR 01 already present (idempotent) -- rebinding")
    else:
        tpl = _find_template_avr(app)
        print(f"template  : {_full(tpl)}")
        copied = comp.AddCopy(tpl)
        if copied is None:
            raise PFSessionError("AddCopy of the template AVR failed")
        # AddCopy may return a list-like on some PF builds
        avr = copied[0] if isinstance(copied, (list, tuple)) else copied
        avr.loc_name = AVR_NAME
        print(f"created   : {_full(avr)}")

    bound = _bind_slots(comp, g01, avr)
    print(f"bound     : {[ (b.loc_name if b is not None else None) for b in bound ]}")
    print(f"slots now : {_slot_report(comp)}")

    # ---- initialise + verify the block is live -----------------------
    ctx = ScreeningContext(app, verbose=False)
    ctx.purge_events()
    ctx.set_monitors([])
    if ctx.inc.Execute():
        try:
            win = app.GetOutputWindow()
            print("ComInc FAILED; output window:")
            print(win.GetContent() if hasattr(win, "GetContent") else win)
        except Exception:  # noqa: BLE001
            pass
        raise PFSessionError("ComInc failed after wiring AVR 01")
    print("ComInc    : OK")
    for sig in ("s:usetp", "s:uerrs", "c:Ka"):
        try:
            print(f"  {sig:10s} = {avr.GetAttribute(sig)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sig:10s} : {exc}")

    if args.smoke > 0:
        term = None
        for t in get_all(app, "ElmTerm"):
            if t.loc_name == "TN_bus38":
                term = t
                break
        if term is None:
            print("smoke: TN_bus38 not found")
            return 0
        ctx.set_monitors([(term, "m:u", "u_TN_bus38"),
                          (g01, "m:Q:bus1", "q_G01")])
        if ctx.inc.Execute():
            raise PFSessionError("ComInc failed before smoke")
        ctx.simulate(float(args.smoke))
        t_, u_ = ctx.read(term, "m:u", stride=50)
        _, q_ = ctx.read(g01, "m:Q:bus1", stride=50)
        print(f"\nsmoke {args.smoke:g}s: TN_bus38 u {u_[0]:.5f} -> {u_[-1]:.5f} "
              f"(drift {u_[-1]-u_[0]:+.5f})   G 01 Q {q_[0]:.2f} -> {q_[-1]:.2f} Mvar")
        print("  (with the AVR the drift should be ~0; without it run 0047 "
              "drifted +0.0043 pu over 900 s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
