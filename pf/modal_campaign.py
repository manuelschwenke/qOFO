"""
pf/modal_campaign.py
====================
W1-W6 campaign of the plant-time-constant modal analysis (handoff 2026-09-22,
M5/M6) plus the residue analysis (R1 option 1, 2026-09-23).

Per window, each PF step in its OWN process (``GetApplication*`` is once per
process):

1. ``pf.pf_sync <snap> --phase full --study-case 02_RMS_CoSim``   -> sync.log
2. the same with ``--dry-run``                                     -> dryrun.log
   STOP if it reports any created/renamed/updated/deleted object;
3. ``pf.modal_window``: STOP on error (rc 2), failed restore (rc 3), any
   unstable mode (stop rule 1) or a failed db_zero acceptance check (§5);
4. ``analysis.modal_observability`` on the window folder (offline).

Always at the end (M6): re-sync W1 and dry-run it.

Usage::

    python -m pf.modal_campaign W2 W3 W4 W5 W6 [--reanalyse W1]

Author: Manuel Schwenke / Claude Code (2026-09-23)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "results" / "screening" / "modal_campaign"
PY = sys.executable
SNAPSHOTS = {
    "W1": "full_t0_20160105-0800_dso3x2.json",
    "W2": "full_at20160222-1300_20160222-1300_dso3x2.json",
    "W3": "full_at20160104-0800_20160104-0800_dso3x2.json",
    "W4": "full_at20161126-1000_20161126-1000_dso3x2.json",
    "W5": "full_at20160203-2000_20160203-2000_dso3x2.json",
    "W6": "full_at20160718-1400_20160718-1400_dso3x2.json",
}


def _run(args: List[str], log: Path) -> int:
    print(f"[campaign] {' '.join(args[2:])}  -> {log.name}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.run([PY, "-X", "utf8"] + args, cwd=REPO, stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def sync_counts(log: Path) -> Dict[str, int]:
    out = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s+(created|renamed|updated|deleted|unchanged): (\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def sync(window: str, folder: Path, tag: str = "") -> Dict[str, Dict[str, int]]:
    snap = str(Path("export") / "snapshots" / SNAPSHOTS[window])
    base = ["-m", "pf.pf_sync", snap, "--phase", "full", "--study-case", "02_RMS_CoSim"]
    s_log, d_log = folder / f"sync{tag}.log", folder / f"dryrun{tag}.log"
    if _run(base, s_log):
        raise RuntimeError(f"{window}: sync failed, see {s_log}")
    if _run(base + ["--dry-run"], d_log):
        raise RuntimeError(f"{window}: dry-run failed, see {d_log}")
    res = {"sync": sync_counts(s_log), "dryrun": sync_counts(d_log)}
    moved = {k: v for k, v in res["dryrun"].items() if k != "unchanged" and v}
    if moved or not res["dryrun"]:
        raise RuntimeError(f"{window}: dry-run after sync is not clean: {res['dryrun']}")
    return res


def latest(folder: Path) -> Path:
    return sorted(p for p in folder.iterdir() if p.is_dir())[-1]


def check_window(wdir: Path) -> List[str]:
    """Stop conditions read from the window's own outputs."""
    import pandas as pd
    state = json.loads((wdir / "window_state.json").read_text(encoding="utf-8"))
    stops = []
    if state.get("error"):
        stops.append(f"error: {state['error']}")
    if not state.get("restore", {}).get("ok") or \
            not state.get("restore", {}).get("commod_options_unchanged", True):
        stops.append(f"restore not verified: {state.get('restore')}")
    for case, info in state.get("cases", {}).items():
        m = pd.read_csv(Path(info["run_dir"]) / "modal_modes.csv")
        n_unst = int((m["re"] > 1e-6).sum())
        if n_unst:
            stops.append(f"stop rule 1: {n_unst} unstable mode(s) in {case}")
    acc = state.get("acceptance", {})
    if acc and not acc.get("passed", False):
        stops.append(f"db_zero acceptance failed: {acc}")
    return stops


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Modal campaign over windows.")
    ap.add_argument("windows", nargs="*")
    ap.add_argument("--reanalyse", nargs="*", default=[],
                    help="windows whose latest folder only needs the offline analysis")
    args = ap.parse_args(argv)
    record = {"started": datetime.now().isoformat(timespec="seconds"), "windows": {}}
    rc = 0
    try:
        for w in args.reanalyse:
            wdir = latest(CAMPAIGN / w)
            code = _run(["-m", "analysis.modal_observability", str(wdir)],
                        wdir / "observability_console.log")
            record["windows"][w] = {"window_dir": str(wdir), "observability_rc": code}
            if code:
                raise RuntimeError(f"{w}: observability failed")
        for w in args.windows:
            folder = CAMPAIGN / w
            folder.mkdir(parents=True, exist_ok=True)
            counts = sync(w, folder)
            print(f"[campaign] {w} sync {counts}", flush=True)
            snap = str(Path("export") / "snapshots" / SNAPSHOTS[w])
            code = _run(["-m", "pf.modal_window", "--window", w, "--snapshot", snap,
                         "--sync-log", str(folder / "sync.log"),
                         "--dryrun-log", str(folder / "dryrun.log")],
                        folder / "modal_window_console.log")
            wdir = latest(folder)
            stops = check_window(wdir) + ([f"modal_window rc {code}"] if code else [])
            record["windows"][w] = {"window_dir": str(wdir), "sync": counts,
                                    "modal_window_rc": code, "stops": stops}
            if stops:
                print(f"[campaign] STOP at {w}: {stops}", flush=True)
                rc = 1
                break
            code = _run(["-m", "analysis.modal_observability", str(wdir)],
                        wdir / "observability_console.log")
            record["windows"][w]["observability_rc"] = code
            print(f"[campaign] {w} done (observability rc {code})", flush=True)
            if code:
                rc = 1
                break
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[campaign] ERROR: {record['error']}", flush=True)
        rc = 2
    finally:
        # M6: leave the project as found -- W1 synced and verified.
        if args.windows:
            try:
                record["final_W1"] = sync("W1", CAMPAIGN / "W1", tag="_final")
                print(f"[campaign] final W1 re-sync {record['final_W1']}", flush=True)
            except Exception as exc:
                record["final_W1_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[campaign] FINAL W1 RE-SYNC FAILED: {exc}", flush=True)
                rc = 3
        record["finished"] = datetime.now().isoformat(timespec="seconds")
        out = CAMPAIGN / f"campaign_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"[campaign] record -> {out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
