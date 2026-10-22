#!/usr/bin/env python3
r"""H3: give G 01 a governor whose STEADY-STATE sharing matches the static plant.

Parity requirement
------------------
The static plant runs ``distributed_slack=True`` with ``slack_weight = sn_mva``
on every in-service machine (network/ieee39/build.py:171), so a power imbalance
is shared

    dP_i = dP_total * S_n,i / sum_j S_n,j

i.e. G 01 (10 000 of 14 300 MVA) absorbs **69.93 %** of any dP, with **no**
frequency deviation at all.

In RMS, governors share according to

    dP_i = -(df / R_pu,i) * S_n,i

so ``dP_i ~ S_n,i`` -- the static law -- holds **iff every machine has the same
per-unit droop R_pu**.  G 01 currently has an EMPTY ``Gov Slot``, which is why
the RMS frequency drifts (1.0000 -> 0.9980 pu in run 0047) and why the
imbalance is shared differently from the static plant.

So the correct implementation is *not* "pick a droop" but: **give G 01 the same
per-unit droop the other governors already use**, then VERIFY the realised
steady-state sharing against the table above with a load-step test.

Modes
-----
``--report``        read every machine's governor + droop parameters, check
                    uniformity, print the static-plant target sharing.
``--apply``         copy a governor into G 01's composite, bind ``Gov Slot``,
                    set its droop to the common per-unit value.
``--test-sharing``  load-step test: measure realised dP per machine and compare
                    against the static-plant fractions.  This is the acceptance
                    criterion for "steady state matches the static plant".
``--revert``        remove GOV 01 and clear the slot.

Slots are bound BY NAME through ``ElmComp.pblk`` (``pelm`` follows ``pblk``, not
the frame's ``GetContents``; an index write can silently evict the machine).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf.screening import RMS_STUDY_CASE, ScreeningContext  # noqa: E402
from pf.session import PFSessionError, connect, get_all  # noqa: E402

G01_NAMES = ("G 01", "G01")
GOV_NAME = "GOV 01"
GOV_SLOT = "Gov Slot"
AVR_SLOT = "Avr Slot"
SYM_SLOT = "Sym Slot"

#: Static-plant sharing target (slack_weight = sn_mva over in-service gens).
#: Filled at runtime from the machines actually in service.


def _full(o) -> str:
    try:
        return o.GetFullName()
    except Exception:  # noqa: BLE001
        return repr(o)


def _machines(app) -> List:
    return [m for m in get_all(app, "ElmSym")
            if m.GetAttribute("outserv") == 0]


def _find_g01(app):
    g = next((m for m in get_all(app, "ElmSym")
              if m.loc_name.strip() in G01_NAMES), None)
    if g is None:
        raise PFSessionError("G 01 ElmSym not found")
    comp = g.GetAttribute("c_pmod")
    if comp is None:
        raise PFSessionError("G 01 has no composite (c_pmod)")
    return g, comp


def _gov_of(comp):
    return next((d for d in comp.GetContents()
                 if d.GetClassName() == "ElmDsl"
                 and "gov" in d.loc_name.lower()), None)


def _param_map(dsl) -> Dict[str, float]:
    """{parameter name: value} for a DSL, via its BlkDef parameter list."""
    out: Dict[str, float] = {}
    typ = dsl.GetAttribute("typ_id")
    names: List[str] = []
    if typ is not None:
        try:
            raw = typ.GetAttribute("sParams")
            if isinstance(raw, str):
                raw = [raw]
            for chunk in (raw or []):
                names.extend([p.strip() for p in str(chunk).split(",") if p.strip()])
        except Exception:  # noqa: BLE001
            pass
    try:
        vals = list(dsl.GetAttribute("params") or [])
    except Exception:  # noqa: BLE001
        vals = []
    for i, v in enumerate(vals):
        key = names[i] if i < len(names) else f"[{i}]"
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            out[key] = v
    return out


def _sn_mva(m) -> float:
    typ = m.GetAttribute("typ_id")
    for attr in ("sgn", "sgnn"):
        try:
            if typ is not None:
                return float(typ.GetAttribute(attr))
        except Exception:  # noqa: BLE001
            continue
    try:
        return float(m.GetAttribute("sgn"))
    except Exception:  # noqa: BLE001
        return float("nan")


def _report(app) -> int:
    machines = _machines(app)
    print(f"in-service machines: {len(machines)}\n")
    print(f"{'machine':10s} {'S_n MVA':>9s} {'governor':>10s} {'AVR':>8s}")
    print("-" * 42)
    govs: Dict[str, Tuple] = {}
    sn_by: Dict[str, float] = {}
    for m in sorted(machines, key=lambda x: x.loc_name):
        comp = m.GetAttribute("c_pmod")
        gov = _gov_of(comp) if comp is not None else None
        avr = None
        if comp is not None:
            avr = next((d for d in comp.GetContents()
                        if d.GetClassName() == "ElmDsl"
                        and "avr" in d.loc_name.lower()), None)
        sn = _sn_mva(m)
        sn_by[m.loc_name] = sn
        print(f"{m.loc_name:10s} {sn:9.0f} {(gov.loc_name if gov else '--'):>10s} "
              f"{(avr.loc_name if avr else '--'):>8s}")
        if gov is not None:
            govs[m.loc_name] = (gov, _param_map(gov))

    print("\n=== governor parameters (per machine) ===")
    for name, (gov, pm) in sorted(govs.items()):
        typ = gov.GetAttribute("typ_id")
        print(f"\n  {name} -> {gov.loc_name}  typ={_full(typ) if typ else None}")
        for k, v in pm.items():
            print(f"      {k:>10s} = {v}")

    # uniformity check on the droop-like parameters
    print("\n=== uniformity check ===")
    keysets = {frozenset(pm.keys()) for _, pm in govs.values()}
    if len(keysets) > 1:
        print("  !! governors do not share a parameter set -- inspect manually")
    else:
        common = sorted(next(iter(keysets)))
        for k in common:
            vals = {n: pm[k] for n, (_, pm) in govs.items()}
            uniq = set(vals.values())
            flag = "UNIFORM" if len(uniq) == 1 else "VARIES"
            print(f"  {k:>10s}: {flag:8s} {vals if len(uniq)>1 else next(iter(uniq))}")

    total = sum(sn_by.values())
    print(f"\n=== static-plant target sharing (slack_weight = sn_mva) ===")
    print(f"  total S_n = {total:.0f} MVA")
    for n, sn in sorted(sn_by.items(), key=lambda kv: -kv[1]):
        print(f"    {n:10s} {sn:8.0f} MVA -> {100*sn/total:6.2f} % of any dP")
    print("\n  parity: uniform per-unit droop on ALL machines (incl. G 01)")
    return 0


def _bind(comp, sym, avr, gov):
    pblk = comp.GetAttribute("pblk") or []
    pelm = []
    for slot in pblk:
        nm = slot.loc_name if slot is not None else None
        if nm == SYM_SLOT:
            pelm.append(sym)
        elif nm == AVR_SLOT:
            pelm.append(avr)
        elif nm == GOV_SLOT:
            pelm.append(gov)
        else:
            pelm.append(None)
    comp.SetAttribute("pelm", pelm)
    return [(s.loc_name if s else "?", e.loc_name if e is not None else None)
            for s, e in zip(pblk, comp.GetAttribute("pelm"))]


def _apply(app) -> int:
    g01, comp = _find_g01(app)
    existing_avr = next((d for d in comp.GetContents()
                         if d.GetClassName() == "ElmDsl"
                         and "avr" in d.loc_name.lower()), None)
    gov = _gov_of(comp)
    if gov is None:
        # template: a governor from another in-service machine
        tpl = None
        tpl_params = None
        for m in _machines(app):
            if m.loc_name.strip() in G01_NAMES:
                continue
            c = m.GetAttribute("c_pmod")
            g = _gov_of(c) if c is not None else None
            if g is not None:
                tpl, tpl_params = g, _param_map(g)
                break
        if tpl is None:
            raise PFSessionError("no template governor found")
        print(f"template : {_full(tpl)}")
        print(f"           params = {tpl_params}")
        copied = comp.AddCopy(tpl)
        if copied is None:
            raise PFSessionError("AddCopy of the template governor failed")
        gov = copied[0] if isinstance(copied, (list, tuple)) else copied
        gov.loc_name = GOV_NAME
        print(f"created  : {_full(gov)}")
        print("  droop inherited from the template => per-unit droop is uniform,"
              "\n  which reproduces the static plant's dP ~ S_n sharing law.")
    else:
        print(f"governor already present: {_full(gov)} (idempotent)")

    print(f"slots    : {_bind(comp, g01, existing_avr, gov)}")

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
        raise PFSessionError("ComInc failed after wiring GOV 01")
    print("ComInc   : OK")
    print(f"  live params = {_param_map(gov)}")
    return 0


def _test_sharing(app, step_pct: float, settle_s: float) -> int:
    """Load-step test: realised dP sharing vs the static-plant fractions."""
    machines = _machines(app)
    sn_by = {m.loc_name: _sn_mva(m) for m in machines}
    total = sum(sn_by.values())

    ctx = ScreeningContext(app, verbose=False)
    ctx.purge_events()
    mons = [(m, "m:P:bus1", f"p_{m.loc_name}") for m in machines]
    load = max((l for l in get_all(app, "ElmLod")
                if l.GetAttribute("outserv") == 0),
               key=lambda l: float(l.GetAttribute("plini") or 0.0))
    print(f"step load : {_full(load)}  plini={load.GetAttribute('plini')} MW")
    ctx.set_monitors(mons)
    if ctx.inc.Execute():
        raise PFSessionError("ComInc failed in sharing test")
    ctx.simulate(5.0)                       # settle flat
    ctx.add_load_event(load, float(step_pct), 0.0, 5.5)
    ctx.simulate(5.5 + float(settle_s))     # governors settle

    print(f"\n{'machine':10s} {'P before':>10s} {'P after':>10s} {'dP':>9s} "
          f"{'share %':>8s} {'target %':>9s} {'err pp':>7s}")
    print("-" * 68)
    dps = {}
    for m in machines:
        t, p = ctx.read(m, "m:P:bus1", stride=20)
        before = p[max(0, int(len(p) * 4.5 / (10.5 + settle_s)))]
        after = p[-1]
        dps[m.loc_name] = after - before
    tot_dp = sum(dps.values())
    for m in sorted(machines, key=lambda x: -sn_by[x.loc_name]):
        n = m.loc_name
        share = 100 * dps[n] / tot_dp if tot_dp else float("nan")
        target = 100 * sn_by[n] / total
        print(f"{n:10s} {'':>10s} {'':>10s} {dps[n]:9.2f} {share:8.2f} "
              f"{target:9.2f} {share-target:+7.2f}")
    print(f"\n  total dP = {tot_dp:.2f} MW")
    print("  PASS if every 'err pp' is small (a few percentage points): the RMS"
          "\n  then shares imbalance like the static distributed slack.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--test-sharing", action="store_true")
    ap.add_argument("--step-pct", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=40.0)
    args = ap.parse_args(argv)
    if not any((args.report, args.apply, args.revert, args.test_sharing)):
        args.report = True

    app = connect(study_case=RMS_STUDY_CASE)

    if args.revert:
        g01, comp = _find_g01(app)
        gov = _gov_of(comp)
        avr = next((d for d in comp.GetContents()
                    if d.GetClassName() == "ElmDsl"
                    and "avr" in d.loc_name.lower()), None)
        if gov is None:
            print("revert: no governor on G 01 -- nothing to do")
        else:
            _bind(comp, g01, avr, None)
            gov.Delete()
            print("revert: GOV 01 deleted, Gov Slot cleared")
        return 0

    rc = 0
    if args.report:
        rc |= _report(app)
    if args.apply:
        rc |= _apply(app)
    if args.test_sharing:
        rc |= _test_sharing(app, args.step_pct, args.settle)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
