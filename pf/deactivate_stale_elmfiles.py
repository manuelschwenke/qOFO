r"""Set ``ElmFile`` profile sources out of service when their data file is gone.

**Why this exists.** ``pf/profile_playback.py`` writes an *absolute* path into
each ``ElmFile.f_name`` of the shared study case, so every RMS replay run
repoints ``02_RMS_CoSim`` at its own results snapshot. ``results/`` is
gitignored and pruned, so the study case breaks as soon as that snapshot is
deleted: every source fails to open, the load flow does not converge, and
``ComInc`` cannot initialise from a non-converged point. Diagnosed 2026-08-19,
when it blocked the Ch. 9.1 settling battery after ~30 min inside ``ComInc``
(`docs/daily_log/08_2026/2026-08-19_ch9_settling_table_emitter_rework.md`).

**What this does.** Sets the affected sources out of service, having first
written every prior value to a restore file, then verifies that a load flow and
``ComInc`` now succeed. If they do not, the prior state is restored
automatically and the script reports failure: a study case that does not
initialise is no worse than before, but one silently left half-modified is.

**Scope.** Only sources whose ``f_name`` does not resolve to an existing file
are touched. A source with a live profile is left alone, so a case that is
genuinely driving profiles is not quietly disarmed.

Usage::

    python pf\deactivate_stale_elmfiles.py --dry-run
    python pf\deactivate_stale_elmfiles.py
    python pf\deactivate_stale_elmfiles.py --restore <restore_file.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROJECT = r"qOFO\IEEE39_qOFO"
DEFAULT_STUDY_CASE = "02_RMS_CoSim"
RESTORE_DIR = REPO_ROOT / "results" / "pf_elmfile_restore"


def _survey(app) -> List[Dict[str, Any]]:
    """Every ``ElmFile``, with its path and whether that path resolves."""
    from pf.session import get_all

    out: List[Dict[str, Any]] = []
    for src in get_all(app, "ElmFile"):
        try:
            f_name = str(src.GetAttribute("f_name"))
        except Exception:
            f_name = ""
        try:
            outserv = int(src.GetAttribute("outserv"))
        except Exception:
            outserv = 0
        exists = bool(f_name) and Path(f_name).exists()
        out.append({"loc_name": src.loc_name, "full_name": src.GetFullName(),
                    "f_name": f_name, "outserv": outserv, "exists": exists,
                    "_obj": src})
    return out


def _check(app) -> Dict[str, int]:
    """Load flow and RMS initialisation return codes (0 = success)."""
    ldf = app.GetFromStudyCase("ComLdf")
    inc = app.GetFromStudyCase("ComInc")
    ierr_ldf = int(ldf.Execute())
    ierr_inc = int(inc.Execute()) if ierr_ldf == 0 else -1
    return {"ldf": ierr_ldf, "inc": ierr_inc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--study-case", default=DEFAULT_STUDY_CASE)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; modify nothing")
    ap.add_argument("--restore", type=Path, default=None,
                    help="restore outserv/f_name from a previous restore file")
    a = ap.parse_args(argv)

    from pf.session import connect

    app = connect(a.project, study_case=a.study_case)
    print(f"[elmfile] project {a.project}, study case {a.study_case}")

    rows = _survey(app)
    by_name = {r["full_name"]: r["_obj"] for r in rows}
    print(f"[elmfile] {len(rows)} ElmFile source(s); "
          f"{sum(1 for r in rows if not r['exists'])} with a missing data file")

    # ---- restore mode ----------------------------------------------------
    if a.restore is not None:
        saved = json.loads(a.restore.read_text(encoding="utf-8"))
        n = 0
        for rec in saved["sources"]:
            obj = by_name.get(rec["full_name"])
            if obj is None:
                print(f"  [warn] no longer present: {rec['full_name']}")
                continue
            obj.SetAttribute("outserv", int(rec["outserv"]))
            n += 1
        print(f"[elmfile] restored outserv on {n} source(s) from {a.restore}")
        print(f"[elmfile] check after restore: {_check(app)}")
        return 0

    stale = [r for r in rows if not r["exists"]]
    if not stale:
        print("[elmfile] every data file resolves; nothing to do")
        print(f"[elmfile] check: {_check(app)}")
        return 0

    missing_paths = sorted({r["f_name"] for r in stale})
    print(f"[elmfile] missing path(s): {len(missing_paths)}")
    for pth in missing_paths[:3]:
        print(f"    {pth}")
    already_out = sum(1 for r in stale if r["outserv"] == 1)
    print(f"[elmfile] of the {len(stale)} stale sources, {already_out} are "
          f"already out of service")

    if a.dry_run:
        print("[elmfile] --dry-run: nothing modified")
        return 0

    before = _check(app)
    print(f"[elmfile] check BEFORE: {before}")

    RESTORE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    restore_file = RESTORE_DIR / f"elmfile_outserv_{stamp}.json"
    restore_file.write_text(json.dumps({
        "timestamp": datetime.now().astimezone().isoformat(),
        "project": a.project, "study_case": a.study_case,
        "reason": ("stale ElmFile data files blocked ComInc; sources set out "
                   "of service for the Ch. 9.1 open-loop settling battery, "
                   "which does not use profile playback"),
        "check_before": before,
        "sources": [{k: r[k] for k in
                     ("loc_name", "full_name", "f_name", "outserv", "exists")}
                    for r in stale],
    }, indent=2), encoding="utf-8")
    print(f"[elmfile] prior state written to {restore_file}")

    n = 0
    for r in stale:
        try:
            r["_obj"].SetAttribute("outserv", 1)
            n += 1
        except Exception as exc:
            print(f"  [warn] {r['loc_name']}: {type(exc).__name__}: {exc}")
    print(f"[elmfile] set {n} source(s) out of service")

    after = _check(app)
    print(f"[elmfile] check AFTER: {after}")

    if after["ldf"] != 0 or after["inc"] != 0:
        print("[elmfile] initialisation still fails -- RESTORING prior state "
              "rather than leaving the case half-modified")
        for r in stale:
            try:
                r["_obj"].SetAttribute("outserv", int(r["outserv"]))
            except Exception:
                pass
        print(f"[elmfile] check after restore: {_check(app)}")
        return 2

    print("[elmfile] load flow and ComInc now succeed")
    print(f"[elmfile] to undo:  python pf\\deactivate_stale_elmfiles.py "
          f"--restore {restore_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
