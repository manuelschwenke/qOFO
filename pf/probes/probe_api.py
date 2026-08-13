"""
pf/probes/probe_api.py
===============
Read-only attribute probe of the IEEE39_qOFO template (Phase-2 preparation).

Prints, for one representative object of every class the sync script will
touch, which candidate attributes exist and their current values.  The
pf_sync field maps are written against this output rather than against
assumed spellings (per-unit and attribute-name guesses are the classic
Gate-A time sink).

Run on the PF machine::

    python pf\\probes\\probe_api.py

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.session import DEFAULT_PROJECT_PATH, connect, get_all  # noqa: E402

#: Candidate attributes per class -- superset of what pf_sync will need.
CANDIDATES = {
    "ElmNet": ("loc_name", "frnom"),
    "ElmTerm": ("loc_name", "uknom", "iUsage", "phtech", "systype", "outserv"),
    "ElmLne": ("loc_name", "dline", "nlnum", "outserv"),
    "TypLne": ("loc_name", "uline", "sline", "InomAir", "rline", "xline",
               "bline", "cline", "gline", "tline", "mlei", "nlnph", "frnom",
               "cohl_"),
    "ElmTr2": ("loc_name", "nntap", "ntrcn", "outserv", "i_auto"),
    "TypTr2": ("loc_name", "strn", "utrn_h", "utrn_l", "uktr", "uktrr",
               "pcutr", "pfe", "curmg", "tap_side", "dutap", "phitr",
               "nntap0", "ntpmn", "ntpmx", "vecgrp", "tr2cn_h", "tr2cn_l",
               "nt2ag"),
    "ElmLod": ("loc_name", "plini", "qlini", "slini", "coslini", "scale0",
               "u0", "outserv", "i_sym", "mode_inp"),
    "TypLod": ("loc_name", "systp", "loddy", "aP", "bP", "cP", "kpu0",
               "kpu1", "kpu", "aQ", "bQ", "cQ", "kqu0", "kqu1", "kqu",
               "i_sym", "udmax", "udmin", "iLoad"),
    "ElmSym": ("loc_name", "pgini", "qgini", "usetp", "ip_ctrl", "av_mode",
               "iv_mode", "ngnum", "outserv", "q_min", "q_max", "Pmin_uc",
               "Pmax_uc", "cQ_min", "cQ_max", "iqtype"),
    "TypSym": ("loc_name", "sgn", "ugn", "cosn", "h", "xd", "xq", "xds",
               "iturbo", "rstr"),
    "StaCubic": ("loc_name", "obj_id", "obj_bus", "it2p1"),
}

#: Representative template objects (loc_name, class).
SAMPLES = (
    ("Bus 01", "ElmTerm"),
    ("Bus 31", "ElmTerm"),        # 16.5 kV machine terminal
    ("G 01", "ElmSym"),
    ("G 05", "ElmSym"),
)


def probe(obj, cls: str) -> None:
    print(f"--- {cls}: {getattr(obj, 'loc_name', '?')!r} "
          f"(class {obj.GetClassName()}) ---")
    fold = None
    try:
        fold = obj.fold_id
    except Exception:
        pass
    if fold is not None:
        print(f"    parent: {fold.loc_name!r} ({fold.GetClassName()})")
    for attr in CANDIDATES.get(cls, ()):
        try:
            print(f"    {attr} = {obj.GetAttribute(attr)!r}")
        except Exception:
            print(f"    {attr}: <absent>")


def main() -> int:
    app = connect(DEFAULT_PROJECT_PATH, study_case="01_LDF_Parity")

    # Grid folder(s)
    for grid in get_all(app, "ElmNet"):
        probe(grid, "ElmNet")

    # Named samples
    for name, cls in SAMPLES:
        objs = [o for o in get_all(app, cls) if o.loc_name == name]
        if objs:
            probe(objs[0], cls)
            typ = None
            try:
                typ = objs[0].typ_id
            except Exception:
                pass
            if typ is not None:
                probe(typ, typ.GetClassName())

    # First line / first 2W trafo / first load, with their types + cubicles
    for cls in ("ElmLne", "ElmTr2", "ElmLod"):
        objs = get_all(app, cls)
        print(f"\n### {cls}: {len(objs)} objects; first three names: "
              f"{[o.loc_name for o in objs[:3]]}")
        if not objs:
            continue
        obj = objs[0]
        probe(obj, cls)
        try:
            typ = obj.typ_id
            if typ is not None:
                probe(typ, typ.GetClassName())
            else:
                print("    typ_id = None")
        except Exception:
            print("    typ_id: <absent>")
        # connection cubicles
        for con in ("bus1", "bus2", "bushv", "buslv"):
            try:
                cub = obj.GetAttribute(con)
            except Exception:
                continue
            if cub is None:
                print(f"    {con} = None")
                continue
            term = None
            try:
                term = cub.cterm
            except Exception:
                pass
            print(f"    {con} -> cubicle {cub.loc_name!r} in terminal "
                  f"{term.loc_name if term is not None else '?'!r} "
                  f"(cub class {cub.GetClassName()})")

    # Machine connection cubicle (for the G1 reconnection later)
    g01 = [o for o in get_all(app, "ElmSym") if o.loc_name == "G 01"]
    if g01:
        cub = g01[0].GetAttribute("bus1")
        print(f"\nG 01 bus1 cubicle: {cub.loc_name!r} in "
              f"{cub.cterm.loc_name!r}")

    # Type library folder
    for key in ("equip", "netdat", "study"):
        try:
            f = app.GetProjectFolder(key)
            print(f"GetProjectFolder({key!r}) -> "
                  f"{f.loc_name!r} ({f.GetClassName()})" if f is not None
                  else f"GetProjectFolder({key!r}) -> None")
        except Exception as exc:
            print(f"GetProjectFolder({key!r}) raised {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
