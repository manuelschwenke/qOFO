"""
pf/probes/probe_modal_residue.py
================================
Read-only probe: can the modal screen be turned into an *observability*
statement about the **controlled outputs**?

Motivation
----------
``pf/screening.py modal`` reports eigenvalues, damping and a per-mode 2 %-band
settling ``T_s = 4/|Re lambda|``.  The full-model table
(``results/screening/full_t0_wecc/20260720-125758/modal.md``) contains a
**107.3 s** non-oscillatory mode, a 35.3 s mode, and nine modes above 10 s,
while the Gate-D step battery measures the worst controlled-output settling at
**13.2 s**.  Read literally the modal table forbids ``T_DS = 20 s``; the
battery says 20 s carries 6.8 s of margin.

Both can be true, because a per-mode settling time is a property of a *mode*,
not of an *output*.  Mode ``l`` reaches output ``y_i`` from actuator ``u_j``
only through its residue ``R_ijl``, with the transfer contribution
``R_ijl / (s - lambda_l)``.  A mode with tiny ``|R_ijl|`` in the interface Q
and the constrained bus voltages does not appear in the quantity the timescale
premise is about.  Closing that gap is what converts the modal screen from
"slow modes exist somewhere in the model" into a *defensible* bound.

The residue needs the output row ``c_i^T = dy_i/dx``.  For a **state**
(machine speed, flux, controller state) PF supplies this.  Every quantity the
OFO actually controls -- the 12 coupler interface Q flows and the TN/PCC bus
voltages -- is **algebraic**, i.e. part of ``z``, not ``x``.  Whether PF hands
back an output matrix for those is *the* open question, and it decides the
method:

* **Path A (exact).** ``ComMod`` exports the linearised system matrices
  (A, B, C, D) for the monitored variable set.  Residues are then computed
  directly and are exact at the linearisation point.
* **Path B (empirical, fallback).** No usable C.  Modal content of the
  controlled outputs is instead fitted from the recorded step-battery
  trajectories (Prony / matrix pencil, ``analysis/modal_residue.py``).  This
  needs no undocumented PF internals and measures the right quantity, but it
  is per-operating-point and per-step rather than analytic.

This probe decides between them.  It mutates nothing.

Run on the PF machine::

    python pf\\probes\\probe_modal_residue.py

Author: Manuel Schwenke / Claude Code (2026-08-03)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    get_all,
)

RMS_STUDY_CASE = "02_RMS_CoSim"

#: ComMod option attributes to interrogate.  Spellings differ across PF
#: versions and none of this is in the scripting reference, so the probe
#: tries a superset and reports which exist.  The matrix-export group is the
#: one that decides Path A -- it is listed first for that reason.
COMMOD_CANDIDATES: Tuple[str, ...] = (
    # --- matrix export (Path A lives or dies here) ---
    "iExpMat", "iMatExp", "iExport", "iExportMat", "iWriteMat",
    "dirMat", "dirMatrix", "sExpPath", "fileMat", "iMatlab", "iFormat",
    # --- eigenvector / participation options ---
    "iLeftEV", "iRightEV", "iLeft", "iRight", "iEigenvec", "iEigVec",
    "iPart", "iPartFac", "iParticipation",
    # --- observability / controllability, if exposed at all ---
    "iObserv", "iObservability", "iContr", "iControllability", "iResidue",
    # --- solver / selection ---
    "iCalcType", "iAlgorithm", "nMode", "iSelect", "iSelMode",
    "repMode", "cinitMode", "iShowAll", "iFreqRange", "fmin", "fmax",
)

#: Attribute names a modal ElmRes may use for its variable/column set.
RES_COL_CANDIDATES: Tuple[str, ...] = (
    "cn_name", "variable", "sVars", "cVars", "obj_id",
)


def _try(obj: Any, attr: str) -> Tuple[bool, Any]:
    try:
        return True, obj.GetAttribute(attr)
    except Exception:
        return False, None


def _report_attrs(obj: Any, names: Tuple[str, ...], indent: str = "    ") -> Dict[str, Any]:
    """Print which of ``names`` exist on ``obj``; return the ones that do."""
    found: Dict[str, Any] = {}
    for name in names:
        ok, val = _try(obj, name)
        if ok:
            found[name] = val
            print(f"{indent}{name:<20} = {val!r}")
    missing = [n for n in names if n not in found]
    if missing:
        print(f"{indent}(absent: {', '.join(missing)})")
    return found


def probe_commod(app) -> Dict[str, Any]:
    """Interrogate the ComMod command object without executing it."""
    print("=" * 72)
    print("ComMod (modal analysis command) -- available options")
    print("=" * 72)
    mod = app.GetFromStudyCase("ComMod")
    if mod is None:
        raise PFSessionError("no ComMod in the active study case")
    found = _report_attrs(mod, COMMOD_CANDIDATES)

    export_attrs = [k for k in found
                    if k.lower().startswith(("iexp", "imat", "iwrite",
                                             "dirmat", "dirmatrix",
                                             "sexp", "filemat", "iformat",
                                             "imatlab"))]
    print()
    if export_attrs:
        print(f"  -> matrix-export attributes PRESENT: {', '.join(export_attrs)}")
        print("     Path A is worth attempting: enable the export, run ComMod,")
        print("     and read A/B/C/D from the export directory.")
    else:
        print("  -> NO matrix-export attribute found on this ComMod.")
        print("     Path A is unavailable through the command object; the")
        print("     probe continues to check whether the result object")
        print("     carries eigenvectors anyway.")
    return found


def probe_modal_results(app) -> None:
    """Run ComMod and describe every modal result object it produces.

    The run itself is a calculation, not a model change: nothing is written
    back to the network.  The initial conditions are computed first because
    ComMod linearises about the RMS operating point.
    """
    print()
    print("=" * 72)
    print("Executing ComInc + ComMod (calculation only, no model mutation)")
    print("=" * 72)
    inc = app.GetFromStudyCase("ComInc")
    if inc.Execute():
        raise PFSessionError("ComInc (RMS init) failed -- cannot linearise")
    mod = app.GetFromStudyCase("ComMod")
    if mod.Execute():
        raise PFSessionError("ComMod failed")

    results = [r for r in get_all(app, "ElmRes") if "Modal" in r.loc_name]
    if not results:
        print("  !! no ElmRes with 'Modal' in loc_name; nothing to describe")
        return
    for res in results:
        print()
        print(f"--- ElmRes {res.loc_name!r} ---")
        try:
            res.Load()
        except Exception as exc:
            print(f"    Load() failed: {exc}")
            continue
        try:
            n_rows = res.GetNumberOfRows()
        except Exception:
            n_rows = -1
        try:
            n_cols = res.GetNumberOfColumns()
        except Exception:
            n_cols = -1
        print(f"    rows = {n_rows}, columns = {n_cols}")

        # Describe the first columns: name, object, variable.  A modal result
        # holding only 2 columns is eigenvalues (Re, Im) and nothing else --
        # that is the current screening assumption and it is what makes the
        # observability question unanswerable from this object alone.
        for c in range(min(n_cols if n_cols > 0 else 0, 12)):
            desc: List[str] = []
            for meth, label in (("GetColumnName", "name"),
                                ("GetObject", "obj"),
                                ("GetVariable", "var")):
                try:
                    val = getattr(res, meth)(c)
                    if meth == "GetObject" and val is not None:
                        val = getattr(val, "loc_name", val)
                    desc.append(f"{label}={val!r}")
                except Exception:
                    continue
            sample = None
            if n_rows > 0:
                try:
                    sample = res.GetValue(0, c)
                except Exception:
                    sample = None
            print(f"      col {c:>3}: {'; '.join(desc) or '(no descriptors)'}"
                  f"  row0={sample!r}")
        if n_cols > 12:
            print(f"      ... {n_cols - 12} further columns")


def probe_output_map(app) -> None:
    """State plainly which controlled outputs are algebraic.

    This is the crux: if the modal result exposes eigenvectors only over the
    state vector, then the interface Q flows and bus voltages -- every
    quantity in ``monitored_outputs`` that the Gate-D verdict is computed on
    -- have no row in it, and no residue can be formed without the algebraic
    output map.
    """
    print()
    print("=" * 72)
    print("Controlled outputs and their variable kind")
    print("=" * 72)
    n_q = sum(1 for tr in get_all(app, "ElmTr3")
              if tr.loc_name.startswith("NC3W_"))
    n_v = sum(1 for t in get_all(app, "ElmTerm")
              if t.loc_name.startswith("TN_bus"))
    n_spd = sum(1 for m in get_all(app, "ElmSym")
                if m.GetAttribute("outserv") == 0)
    print(f"  coupler interface Q (m:Q:bushv)   : {n_q:>3}  ALGEBRAIC")
    print(f"  TN bus voltages     (m:u)         : {n_v:>3}  ALGEBRAIC")
    print(f"  machine speeds      (s:xspeed)    : {n_spd:>3}  STATE")
    print()
    print("  The Gate-D verdict is computed on the two ALGEBRAIC groups.")
    print("  Machine speed is the diagnostic 'ring' channel only.  A residue")
    print("  path that reaches states but not algebraic variables therefore")
    print("  answers the wrong question and must not be used to claim")
    print("  observability in the controlled outputs.")


def verdict(commod_attrs: Dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("VERDICT -- record this in docs/pf_api_notes.md")
    print("=" * 72)
    print("""
Answer these three from the output above, then choose the path:

  1. Does ComMod expose a matrix-export option (A/B/C/D to file)?
  2. Does the modal ElmRes carry more than the (Re, Im) eigenvalue pair --
     specifically right/left eigenvectors or participation factors?
  3. If eigenvectors exist, are their rows STATES only, or do they include
     algebraic variables?

  Path A (exact residues) requires 1 = yes, OR (2 = yes AND 3 = includes
  algebraic).  Then extend cmd_modal to form R_ijl = c_i^T v_l w_l^T b_j and
  report residue-weighted OUTPUT settling.

  Path B (Prony fit of step-battery trajectories) is required otherwise, and
  is implemented in analysis/modal_residue.py.  It needs the battery re-run
  with --save-trajectories, because the Gate-D CSVs persisted only summary
  statistics (y_init, y_final, step, t_settle, overshoot) and NOT the time
  series -- the modal content of those runs is unrecoverable.

  Either way the headline number in ch:param:timescales becomes an OUTPUT
  settling time, never a per-mode one.
""".rstrip())


def main(argv: List[str] | None = None) -> int:
    project = argv[0] if argv else DEFAULT_PROJECT_PATH
    app = connect(project, study_case=RMS_STUDY_CASE)
    print(f"[probe] project {project!r}, study case {RMS_STUDY_CASE!r}")
    commod_attrs = probe_commod(app)
    probe_output_map(app)
    probe_modal_results(app)
    verdict(commod_attrs)
    print()
    print("probe done (read-only; no events created, no model change).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
