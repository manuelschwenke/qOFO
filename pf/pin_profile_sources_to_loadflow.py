r"""Pin every ``ElmFile`` profile source to its target's load-flow value.

**The problem.** An open-loop settling battery needs the plant to sit still: a
flat run must not drift, or the drift is added to every settling time. The
study case drives loads and DER-P from a recorded profile whose multipliers
belong to whatever operating point that recording was made at, while the static
model (and therefore the load flow) carries values from a different sync. Each
driven element is then pulled from its load-flow value toward
``base * multiplier``, the machines take up the difference through their
governors, and the plant settles somewhere else.

Measured 2026-08-19 on `02_RMS_CoSim`: with the profile frozen at t0 the DER
parks hold exactly (``sum|dP| = 0.000`` MW) but the loads move ``5.97`` MW --
three DSO_3 loads start at 29.805 / 29.832 / 29.845 MW and all converge on
30.062 MW, the single value ``base * multiplier`` gives them. Preflight drift
``1.36e-02`` pu against a ``1e-4`` tolerance.

**The fix.** ``afac`` is per source and the profile's first channel is a
constant 1.0 (``UNITY_COLUMN``). Pointing a source at that channel with
``afac`` set to its own target's load-flow value makes it drive exactly that
value -- per element, exactly, with no dependence on which recording the
multipliers came from. A shared multiplier channel cannot do this: the three
DSO_3 loads above need three different values from one channel.

The result is a **static operating point by construction**, which is what the
battery is specified on. It is not a substitute for a correctly re-exported
profile in a closed-loop replay -- it deliberately removes the time variation.

Reversible: prior ``icol``/``afac``/``f_name`` are written to a restore file
and the result is verified, restoring automatically if it does not hold.

Usage::

    python pf\pin_profile_sources_to_loadflow.py --dry-run
    python pf\pin_profile_sources_to_loadflow.py
    python pf\pin_profile_sources_to_loadflow.py --restore <file.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROJECT = r"qOFO\IEEE39_qOFO"
DEFAULT_STUDY_CASE = "02_RMS_CoSim"
RESTORE_DIR = REPO_ROOT / "results" / "pf_elmfile_restore"

#: 1-based ElmFile column of the constant-1.0 channel that
#: ``profile_playback.build_profile_schedule`` always writes first.
UNITY_CHANNEL = 1


def _owner_of(app, source) -> Optional[Any]:
    """The element this source drives, found through its composite.

    Load sources sit in a ``qOFO RMS Profile Load N`` composite alongside their
    ``ElmLod``; DER sources sit in the park's ``WECC_*`` composite alongside
    its ``ElmGenstat``. Both are located by scanning composites for one that
    references the source, rather than by name arithmetic -- the DER naming
    index is a build index and does not track the machine.
    """
    from pf.session import get_all

    for comp in get_all(app, "ElmComp"):
        try:
            slots = comp.GetAttribute("pelm") or []
        except Exception:
            continue
        if not any(o is not None and o.GetFullName() == source.GetFullName()
                   for o in slots):
            continue
        for o in slots:
            if o is None:
                continue
            if o.GetClassName() in ("ElmLod", "ElmGenstat"):
                return o
    return None


def _static_pq(obj) -> Optional[tuple]:
    """``(P, Q)`` the load flow used for this element, in MW / Mvar."""
    cls = obj.GetClassName()
    try:
        if cls == "ElmLod":
            return float(obj.GetAttribute("plini")), float(obj.GetAttribute("qlini"))
        if cls == "ElmGenstat":
            return float(obj.GetAttribute("pgini")), float(obj.GetAttribute("qgini"))
    except Exception:
        return None
    return None


def _check(app) -> Dict[str, int]:
    ldf = app.GetFromStudyCase("ComLdf")
    inc = app.GetFromStudyCase("ComInc")
    e1 = int(ldf.Execute())
    e2 = int(inc.Execute()) if e1 == 0 else -1
    return {"ldf": e1, "inc": e2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--study-case", default=DEFAULT_STUDY_CASE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", type=Path, default=None)
    ap.add_argument("--loads-only", action="store_true",
                    help="pin only load sources; DER-P sources already hold "
                         "exactly once the profile is frozen")
    a = ap.parse_args(argv)

    from pf.session import connect, get_all

    app = connect(a.project, study_case=a.study_case)
    print(f"[pin] project {a.project}, case {a.study_case}")

    sources = list(get_all(app, "ElmFile"))
    by_name = {o.GetFullName(): o for o in sources}
    print(f"[pin] {len(sources)} ElmFile source(s)")

    if a.restore is not None:
        saved = json.loads(a.restore.read_text(encoding="utf-8"))
        n = 0
        for rec in saved["sources"]:
            obj = by_name.get(rec["full_name"])
            if obj is None:
                print(f"  [warn] missing: {rec['full_name']}")
                continue
            obj.SetAttribute("icol", list(rec["icol"]))
            obj.SetAttribute("afac", list(rec["afac"]))
            if rec.get("f_name"):
                obj.SetAttribute("f_name", str(rec["f_name"]))
            n += 1
        print(f"[pin] restored icol/afac on {n} source(s)")
        print(f"[pin] check: {_check(app)}")
        return 0

    ldf = app.GetFromStudyCase("ComLdf")
    if ldf.Execute():
        print("[abort] load flow did not converge; nothing to pin to")
        return 1

    plan: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    for src in sources:
        owner = _owner_of(app, src)
        if owner is None:
            unresolved.append(src.loc_name)
            continue
        if a.loads_only and owner.GetClassName() != "ElmLod":
            continue
        pq = _static_pq(owner)
        if pq is None:
            unresolved.append(src.loc_name)
            continue
        try:
            icol = list(src.GetAttribute("icol"))
            afac = list(src.GetAttribute("afac"))
        except Exception:
            unresolved.append(src.loc_name)
            continue
        n = len(icol)
        new_icol = [UNITY_CHANNEL] * n
        new_afac = [float(pq[0]), float(pq[1])] + [1.0] * (n - 2)
        plan.append({"full_name": src.GetFullName(), "loc_name": src.loc_name,
                     "owner": owner.loc_name, "owner_class": owner.GetClassName(),
                     "icol": icol, "afac": afac,
                     "f_name": str(src.GetAttribute("f_name")),
                     "new_icol": new_icol, "new_afac": new_afac,
                     "p": pq[0], "q": pq[1], "_obj": src})

    print(f"[pin] resolved {len(plan)} source(s) to a target; "
          f"{len(unresolved)} unresolved")
    for nm in unresolved[:5]:
        print(f"    unresolved: {nm}")
    if not plan:
        print("[abort] nothing to pin")
        return 1

    kinds = {}
    for r in plan:
        kinds[r["owner_class"]] = kinds.get(r["owner_class"], 0) + 1
    print(f"[pin] by target class: {kinds}")
    for r in plan[:5]:
        print(f"    {r['loc_name']:34s} -> {r['owner']:24s} "
              f"P={r['p']:9.4f} Q={r['q']:9.4f}   "
              f"icol {r['icol'][:2]} -> {r['new_icol'][:2]}   "
              f"afac {[round(x,4) for x in r['afac'][:2]]} -> "
              f"{[round(x,4) for x in r['new_afac'][:2]]}")

    if a.dry_run:
        print("[pin] --dry-run: nothing modified")
        return 0

    before = _check(app)
    print(f"[pin] check BEFORE: {before}")

    RESTORE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    restore_file = RESTORE_DIR / f"elmfile_pin_{stamp}.json"
    restore_file.write_text(json.dumps({
        "timestamp": datetime.now().astimezone().isoformat(),
        "project": a.project, "study_case": a.study_case,
        "reason": ("pinned ElmFile sources to their targets' load-flow values "
                   "so the open-loop settling battery sees a static operating "
                   "point"),
        "check_before": before,
        "sources": [{k: r[k] for k in
                     ("full_name", "loc_name", "owner", "owner_class",
                      "icol", "afac", "f_name")} for r in plan],
    }, indent=2), encoding="utf-8")
    print(f"[pin] prior state -> {restore_file}")

    for r in plan:
        r["_obj"].SetAttribute("icol", r["new_icol"])
        r["_obj"].SetAttribute("afac", r["new_afac"])
    print(f"[pin] pinned {len(plan)} source(s) to the unity channel")

    after = _check(app)
    print(f"[pin] check AFTER: {after}")
    if after["ldf"] != 0 or after["inc"] != 0:
        print("[pin] initialisation broke -- RESTORING")
        for r in plan:
            r["_obj"].SetAttribute("icol", r["icol"])
            r["_obj"].SetAttribute("afac", r["afac"])
        print(f"[pin] check after restore: {_check(app)}")
        return 2

    print("[pin] load flow and ComInc succeed")
    print(f"[pin] to undo:  python pf\\pin_profile_sources_to_loadflow.py "
          f"--restore {restore_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
