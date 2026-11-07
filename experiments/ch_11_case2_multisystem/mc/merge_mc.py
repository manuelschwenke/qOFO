#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_11_case2_multisystem/mc/merge_mc.py
==================================================
Merge the runs of a campaign share simulated elsewhere (``run_mc_remote``)
into the campaign of record, so that ``run_mc --report`` / a resume and
``evaluate_mc.py`` see one folder (plan
``docs/architecture/2026-11-07_case2_mc_new_base_plan.md``).

* Prescreens: the draw and the L2 base case are deterministic, so a seed
  screened in both folders must agree on ``valid``, ``start_time``, the events
  and the commitment.  A disagreement is listed and the destination's record
  is kept; the source's runs on such a seed are NOT copied.
* Runs: ``runs/<seed>_<variant>.{json,npz,log.pkl}`` of the source are copied
  when the destination has no JSON for that (seed, variant); nothing is
  overwritten.  A ``task_crash`` is not copied (rerun it).
* The merge is recorded in the destination's README (``annotate_run``) and in
  ``merged_from_<source>.json``.

    python -m experiments.ch_11_case2_multisystem.mc.merge_mc SRC DST [--dry-run]

Author: Manuel Schwenke / Claude Code (2026-11-07)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Prescreen fields that must agree between the two folders.
SAME = ("valid", "start_time", "events", "uc_offline")


def merge(src: Path, dst: Path, dry_run: bool = False) -> Dict[str, Any]:
    sp, dp = src / "prescreen", dst / "prescreen"
    mismatch: List[int] = []
    pre_copied = 0
    for p in sorted(sp.glob("*.json")):
        a = json.loads(p.read_text(encoding="utf-8"))
        q = dp / p.name
        if q.exists():
            b = json.loads(q.read_text(encoding="utf-8"))
            if any(a.get(k) != b.get(k) for k in SAME):
                mismatch.append(int(a["seed"]))
        else:
            pre_copied += 1
            if not dry_run:
                dp.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
    copied, skipped, crashes = [], [], []
    for p in sorted((src / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        key = (int(r["seed"]), str(r["variant"]))
        if r["outcome"] == "task_crash":
            crashes.append(key)
            continue
        if key[0] in mismatch or (dst / "runs" / p.name).exists():
            skipped.append(key)
            continue
        copied.append(key)
        if not dry_run:
            (dst / "runs").mkdir(parents=True, exist_ok=True)
            stem = p.name[:-len(".json")]
            for f in (src / "runs").glob(stem + ".*"):
                shutil.copy2(f, dst / "runs" / f.name)
    res = {"source": str(src), "destination": str(dst),
           "when": f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}",
           "prescreen_copied": pre_copied, "prescreen_mismatch": mismatch,
           "runs_copied": len(copied), "runs_skipped": [list(k) for k in skipped],
           "task_crash_not_copied": [list(k) for k in crashes],
           "variants_copied": sorted({v for _, v in copied})}
    if not dry_run:
        (dst / f"merged_from_{src.name}.json").write_text(json.dumps(res, indent=1),
                                                         encoding="utf-8")
        host = src / "remote_host.json"
        if host.exists():
            shutil.copy2(host, dst / f"remote_host_{src.name}.json")
        try:
            from experiments.thesis_runs import annotate_run
            annotate_run(dst, by=Path(__file__), outputs={
                f"merged_from_{src.name}.json":
                    f"{len(copied)} runs of {src.name} merged in"})
        except Exception as exc:                   # noqa: BLE001 -- the record exists
            print(f"  [merge] README not annotated: {exc}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    res = merge(Path(a.src), Path(a.dst), a.dry_run)
    print(json.dumps({k: v for k, v in res.items() if k != "runs_skipped"}, indent=1))
    if res["prescreen_mismatch"]:
        print(f"  [merge] WARNING: prescreens differ on {res['prescreen_mismatch']}; "
              f"their runs were not copied -- rerun them here")


if __name__ == "__main__":
    main()
