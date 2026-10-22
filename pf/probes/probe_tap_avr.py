"""
pf/probes/probe_tap_avr.py
===================
Phase-5 probe: discover + RMS-verify the two remaining actuator write
handles that the DER-Q step battery does not cover
(docs/daily_log/07_2026/2026-07-20_rms_phase5_screening_build.md, open item 1):

1. **Machine AVR V-ref.** ``EvtParam`` on ``ElmSym.usetp`` gave a zero RMS
   response (2026-07-20, run 083416) -- the reference lives inside the AVR
   DSL block.  Dump every retained machine's composite (DSL blocks, block
   definitions, parameters), then step the AVR's ``usetp`` input signal via
   ``EvtParam`` (the mechanism proven on ``REEC_D.Qext``) and check the
   machine terminal voltage actually moves.

2. **OLTC tap.** ``EvtParam`` on the tap-position attribute also gave a
   zero response (taps are read at init; the Y-matrix is not rebuilt on a
   parameter event).  Create an ``EvtTap`` object, dump its attributes,
   then fire a +1 tap on one NC3W coupler and check the MV-side PCC
   voltage moves.

3. **MSC/MSR shunt** (bonus, same mechanism): ``EvtTap`` on an ``ElmShnt``
   step, check the local voltage moves.

Each verification is a short RMS run (event at t=1 s, 6 s horizon) on the
full_t0 WECC model in ``02_RMS_CoSim``.  Pure diagnosis -- nothing is
persisted; all events are purged afterwards.

Usage (PF machine)::

    python pf\\probes\\probe_tap_avr.py

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.screening import ScreeningContext  # noqa: E402
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    get_all,
)

RMS_STUDY_CASE = "02_RMS_CoSim"

#: Candidate EvtTap attributes (GUI dialog fields, PF 2025); probed one by
#: one because the API has no reliable attribute enumeration.
EVTTAP_CANDIDATES = [
    "time", "p_target", "hrtime", "mtime",
    "iopt_seq", "iopt_ev", "iTap", "itap", "ntap", "ntapabs", "iTapCh",
    "i_tapdir", "iNeutral", "winding", "iwinding", "iopt_winding", "n_side",
]


def _machine_composites(app) -> List[Tuple[Any, Any]]:
    """(ElmComp, ElmSym) for every in-service synchronous machine."""
    out = []
    for comp in get_all(app, "ElmComp"):
        pelm = comp.GetAttribute("pelm") or []
        for el in pelm:
            if el is not None and el.GetClassName() == "ElmSym" \
                    and el.GetAttribute("outserv") == 0:
                out.append((comp, el))
    return out


def _dump_composite(comp, sym) -> Optional[Any]:
    """Print the composite structure; return the AVR ElmDsl (or None)."""
    frame = comp.GetAttribute("typ_id")
    print(f"\ncomposite {comp.loc_name!r}  machine={sym.loc_name!r}  "
          f"frame={frame.loc_name if frame else None!r}")
    avr = None
    for dsl in [o for o in comp.GetContents() if o.GetClassName() == "ElmDsl"]:
        typ = dsl.GetAttribute("typ_id")
        tnm = typ.loc_name if typ else "?"
        print(f"    ElmDsl {dsl.loc_name!r:24s} block={tnm!r}")
        names: List[str] = []
        for attr in ("sParams", "params"):
            try:
                v = dsl.GetAttribute(attr)
                if v:
                    names = list(v) if not isinstance(v, str) else [v]
                    break
            except Exception:
                continue
        if not names and typ is not None:
            try:
                names = list(typ.GetAttribute("sParams") or [])
            except Exception:
                names = []
        if names:
            vals = []
            for p in names[:40]:
                try:
                    vals.append(f"{p}={dsl.GetAttribute(p)!r}")
                except Exception:
                    vals.append(f"{p}=<unreadable>")
            print(f"        params: {', '.join(vals)}")
        if "avr" in (dsl.loc_name + tnm).lower() or "vco" in tnm.lower():
            avr = dsl
    return avr


def _probe_object_attrs(obj, candidates: List[str]) -> None:
    """Try GetAttributeNames() if the release has it, else candidates."""
    try:
        names = obj.GetAttributeNames()
        print(f"    GetAttributeNames(): {sorted(names)}")
        return
    except Exception:
        pass
    for a in candidates:
        try:
            print(f"    {a} = {obj.GetAttribute(a)!r}")
        except Exception:
            pass


def _run_and_report(ctx: ScreeningContext, label: str,
                    monitors: List[Tuple[Any, str, str]],
                    make_event, horizon: float = 7.0) -> None:
    """Init, place the event via ``make_event(ctx)``, run, print deltas."""
    ctx.set_monitors(monitors)
    ctx.purge_events()
    ctx.initialise()
    make_event(ctx)
    ctx.simulate(horizon)
    print(f"  [{label}] responses (t=0 vs t={horizon:.0f}s, and min/max):")
    for obj, var, mlab in monitors:
        t, y = ctx.read(obj, var, stride=2)
        print(f"    {mlab:28s} init={y[0]:.6f} final={y[-1]:.6f} "
              f"min={min(y):.6f} max={max(y):.6f} "
              f"|delta|={abs(y[-1] - y[0]):.3e}")


def main() -> int:
    app = connect(DEFAULT_PROJECT_PATH, study_case=RMS_STUDY_CASE)
    ctx = ScreeningContext(app)

    # ------------------------------------------------------------------
    # 1) Machine composites: structure + AVR block discovery
    # ------------------------------------------------------------------
    print("=" * 72)
    print("1) machine composites (retained fleet)")
    comps = _machine_composites(app)
    avr_of = {}
    for comp, sym in comps:
        avr = _dump_composite(comp, sym)
        if avr is not None:
            avr_of[sym.loc_name] = avr
    print(f"\nAVR blocks found: { {k: v.loc_name for k, v in avr_of.items()} }")

    # ------------------------------------------------------------------
    # 2) EvtTap attribute discovery (create one, probe, delete)
    # ------------------------------------------------------------------
    print("=" * 72)
    print("2) EvtTap attribute probe")
    ev = ctx.evt_folder.CreateObject("EvtTap", "probe_tap")
    if ev is None:
        print("    !! CreateObject('EvtTap') returned None -- class name wrong?")
    else:
        _probe_object_attrs(ev, EVTTAP_CANDIDATES)
        ev.Delete()

    # ------------------------------------------------------------------
    # Shared monitor set for the verification runs
    # ------------------------------------------------------------------
    tr3 = next(t for t in sorted(get_all(app, "ElmTr3"),
                                 key=lambda o: o.loc_name)
               if t.loc_name.startswith("NC3W_"))
    mv_term = tr3.GetAttribute("busmv").cterm
    hv_term = tr3.GetAttribute("bushv").cterm
    shunts = sorted(get_all(app, "ElmShnt"), key=lambda o: o.loc_name)
    print("=" * 72)
    print(f"coupler under test: {tr3.loc_name!r}  MV bus {mv_term.loc_name!r}")
    print("shunt inventory:")
    for sh in shunts:
        try:
            print(f"    {sh.loc_name!r:28s} ncapx={sh.GetAttribute('ncapx')!r} "
                  f"ncapa={sh.GetAttribute('ncapa')!r} "
                  f"outserv={sh.GetAttribute('outserv')!r} "
                  f"bus={sh.GetAttribute('bus1').cterm.loc_name!r}")
        except Exception as exc:
            print(f"    {sh.loc_name!r}: <{exc}>")

    # ------------------------------------------------------------------
    # 3) Verification A -- AVR usetp step on one machine
    # ------------------------------------------------------------------
    print("=" * 72)
    print("3) AVR V-ref step verification (+0.02 pu on one machine)")
    sym_name, avr = next(iter(sorted(avr_of.items())))
    sym = next(s for s in get_all(app, "ElmSym") if s.loc_name == sym_name)
    gterm = sym.GetAttribute("bus1").cterm
    mons_avr = [(gterm, "m:u", f"u_{gterm.loc_name}"),
                (hv_term, "m:u", f"u_{hv_term.loc_name}"),
                (sym, "s:xspeed", f"spd_{sym_name}")]

    def _evt_avr(c: ScreeningContext) -> None:
        # Read the initialised reference, then step it -- same recipe that
        # was verified on REEC_D.Qext.
        cur = None
        for a in ("usetp", "s:usetp", "c:usetp"):
            try:
                cur = float(avr.GetAttribute(a))
                print(f"    {avr.loc_name}.{a} initialised at {cur:.6f}")
                break
            except Exception:
                continue
        if cur is None:
            raise PFSessionError(f"cannot read usetp on {avr.loc_name!r}")
        c.add_param_event(avr, "usetp", cur + 0.02, 1.0)

    try:
        _run_and_report(ctx, f"AVR {sym_name} usetp+0.02", mons_avr, _evt_avr)
    except PFSessionError as exc:
        print(f"  !! AVR verification failed: {exc}")

    # ------------------------------------------------------------------
    # 4) Verification B -- EvtTap +1 on the coupler 3W
    # ------------------------------------------------------------------
    print("=" * 72)
    print("4) coupler-3W tap event verification (+1 tap)")
    for a in ("n3tap_h", "n3tap_m", "n3tap_l", "nntap"):
        try:
            print(f"    {tr3.loc_name}.{a} = {tr3.GetAttribute(a)!r}")
        except Exception:
            pass
    mons_tap = [(mv_term, "m:u", f"u_{mv_term.loc_name}"),
                (hv_term, "m:u", f"u_{hv_term.loc_name}"),
                (tr3, "m:Q:bushv", f"qSTS_{tr3.loc_name}")]

    def _evt_tap(c: ScreeningContext) -> None:
        ev = c.evt_folder.CreateObject("EvtTap", "probe_tap_step")
        if ev is None:
            raise PFSessionError("EvtTap creation failed")
        ev.SetAttribute("p_target", tr3)
        ev.SetAttribute("time", 1.0)
        # Direction/position attribute: first writable candidate wins.
        wrote = False
        for attr in ("iTap", "itap", "iTapCh", "i_tapdir", "ntap"):
            try:
                ev.SetAttribute(attr, 1)
                print(f"    EvtTap.{attr} := 1 (accepted)")
                wrote = True
                break
            except Exception:
                continue
        if not wrote:
            raise PFSessionError("no writable tap-direction attribute found "
                                 "-- read the attribute probe output above")

    try:
        _run_and_report(ctx, f"EvtTap +1 on {tr3.loc_name}", mons_tap, _evt_tap)
    except PFSessionError as exc:
        print(f"  !! tap verification failed: {exc}")

    # ------------------------------------------------------------------
    # 5) Verification C -- EvtTap on one MSC/MSR shunt
    # ------------------------------------------------------------------
    print("=" * 72)
    print("5) shunt step verification (EvtTap on ElmShnt)")
    sh = next((s for s in shunts if s.GetAttribute("outserv") == 0), None)
    if sh is None:
        print("    no in-service shunt -- skipped")
    else:
        sterm = sh.GetAttribute("bus1").cterm
        mons_sh = [(sterm, "m:u", f"u_{sterm.loc_name}"),
                   (hv_term, "m:u", f"u_{hv_term.loc_name}")]

        def _evt_shunt(c: ScreeningContext) -> None:
            ev = c.evt_folder.CreateObject("EvtTap", "probe_shunt_step")
            if ev is None:
                raise PFSessionError("EvtTap creation failed")
            ev.SetAttribute("p_target", sh)
            ev.SetAttribute("time", 1.0)
            for attr in ("iTap", "itap", "iTapCh", "i_tapdir", "ntap"):
                try:
                    ev.SetAttribute(attr, 1)
                    print(f"    EvtTap.{attr} := 1 (accepted)")
                    return
                except Exception:
                    continue
            raise PFSessionError("no writable tap attribute on shunt EvtTap")

        try:
            _run_and_report(ctx, f"EvtTap +1 on {sh.loc_name}", mons_sh,
                            _evt_shunt)
        except PFSessionError as exc:
            print(f"  !! shunt verification failed: {exc}")

    ctx.purge_events()
    print("=" * 72)
    print("probe done (events purged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
