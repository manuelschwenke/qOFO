#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_16_ref_after_v4.py
==============================================================
The Ch 10 REF runs with the procedure-v4 weights, queued behind the v4 campaign
(author 2026-11-04: "queue the REF runs with new weights"; GA and GE excluded,
"some of the GE and GA scenarios didn't converge anyway").  Nothing starts unless
the campaign ends with ``GO.json`` and, by default, the held-out W5 report counts
no tap or bank reversal (``--no-w5-gate`` drops that condition; run 0006 was GO
with 31 W5 reversals, and the author kept its follow-ups paused).

1. **Record.**  The v4 result gets what its selection ran with and what the REF
   variants read, listed under ``provenance.record_additions``:

   * every ``--config-set`` field of the campaign as a parameter-set field
     (``sts_coupler_hold_after_bank = 1``), so the REF controllers are the
     selected ones (``apply_paramset`` applies any MultiTSOConfig field);
   * ``provenance.lambda_max_B`` = the v4 lambda without the bank columns (author
     B1b: banks excluded "as in variant M"), M's loop-gain ceiling in
     ``variants.selected_lambda_max``.  The v3/v4 results record only
     ``design_lambda_max_B``, so M would stop with a KeyError.

2. **Switch.**  SELECTED_PARAMSET -> the v4 result in ``_ch9_selected_design.py``;
   the design file and the parameter set are committed.
3. **REF** here (e5wimi, 20-core mask): scenarios A, B, D, E, each with the six
   variants, 360 min, ``--jobs 5`` (20 processes), launched 90 s apart so the run
   folders are numbered one at a time.  A start that dies before its variants
   finish (the llvmlite start failure of this host) is completed with
   ``--run-dir`` for the missing variants, twice at most; a variant that fails in
   the model is a result and is not repeated.

Usage (from the repository root, on e5wimi; waits for the campaign)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_16_ref_after_v4.py \
        --run results/ch_09_parameter_selection/0008_2026-11-04_133847_ch_9_14_procedure_v4

Author: Manuel Schwenke / Claude Code (2026-11-04)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_12_followups_v3 as F  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_14_procedure_v4 as V4  # noqa: E402

RESULT = V4.RESULT
SCENARIOS = ("A", "B", "D", "E")
#: Longest first (S1, M; REF runs of 2026-10-31), so the sixth waits for a short one.
VARIANTS = "S1,M,L2,O2,O3,S2"
REF_DIR = ROOT / F.REF_REL
AFFINITY_MASK = 0x5555555555          # e5wimi: 20 cores, one logical CPU each (author 2026-11-04)
LAUNCH_GAP_S = 90.0
RETRIES = 2
OUT: Path = ROOT                      # set in main()


def w5_clean(doc: dict) -> Tuple[bool, str]:
    """No counted tap and no counted bank reversal in the W5 report."""
    w5 = doc.get("provenance", {}).get("w5_report") or {}
    n, m = w5.get("n_hunting"), w5.get("n_shunt_hunting")
    return n == 0 and m == 0, f"W5 report: {n} tap and {m} bank reversal(s) counted"


def record_additions(doc: dict) -> Dict[str, Any]:
    """What :func:`record` adds to the v4 result; empty once it has been added."""
    from configs.config import MultiTSOConfig
    prov = doc["provenance"]
    if "record_additions" in prov:
        return {}
    v4 = prov["v4"]
    known = {f.name for f in dataclasses.fields(MultiTSOConfig)}
    fields: Dict[str, Any] = {}
    for item in v4.get("config_set", []):         # parsed as ch_9_3_stage2_screen does
        name, sep, raw = item.partition("=")
        if not sep or name not in known:
            raise RuntimeError(f"config_set {item!r}: expected FIELD=VALUE of MultiTSOConfig")
        try:
            fields[name] = json.loads(raw)
        except json.JSONDecodeError:
            fields[name] = raw
        if name in doc["fields"] and doc["fields"][name] != fields[name]:
            raise RuntimeError(f"{name}: the result has {doc['fields'][name]!r}, "
                               f"the campaign ran with {fields[name]!r}")
    add: Dict[str, Any] = {"fields": fields}
    if prov.get("lambda_max_B") is None:
        add["provenance"] = {"lambda_max_B": float(v4["lambda_max_B_v4"])}
    return add


def record(path: Path, dry_run: bool) -> None:
    doc = json.loads(path.read_text("utf-8"))
    add = record_additions(doc)
    if add:
        doc["fields"].update(add["fields"])
        doc["provenance"].update(add.get("provenance", {}))
        doc["provenance"]["record_additions"] = dict(
            add, by=P.rel(Path(__file__)), on=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            reason="the selection ran with these config fields (--config-set); "
                   "lambda_max_B is M's loop-gain ceiling in the REF variants "
                   "(author B1b: banks excluded, as in variant M)")
        if not dry_run:
            path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    P.log(f"record {path.name}: "
          + (f"added {json.dumps(add)}" + (" (dry run, not written)" if dry_run else "")
             if add else "additions present already"))


def switch_selected(key: str, dry_run: bool) -> None:
    text = F.DESIGN.read_text("utf-8")
    new, n = re.subn(r'^SELECTED_PARAMSET = "[^"]+"', f'SELECTED_PARAMSET = "{RESULT}"',
                     text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"SELECTED_PARAMSET line not found in {F.DESIGN}")
    if dry_run:
        P.log(f"dry run: SELECTED_PARAMSET would become {RESULT}, committed")
        return
    from configs.paramsets import PARAMSET_DIR
    if new != text:
        F.DESIGN.write_text(new, "utf-8")
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    files = [P.rel(F.DESIGN), P.rel(PARAMSET_DIR / f"{RESULT}.json")]
    subprocess.check_call(git + ["add", *files])
    if subprocess.call(git + ["diff", "--cached", "--quiet"]) != 0:
        subprocess.check_call(git + ["commit", "-q", "-m",
                                     f"SELECTED_PARAMSET -> {RESULT} (procedure v4 GO, {key})\n\n"
                                     "The parameter set carries the campaign's config fields "
                                     "and lambda_max_B (provenance.record_additions); REF "
                                     "queued by the author 2026-11-04.\n\n"
                                     "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"])
    P.log(f"SELECTED_PARAMSET = {RESULT} (commit {P.local_head()[:7]})")


def _folder(tag: str) -> Optional[Path]:
    hits = sorted(REF_DIR.glob(f"*_run_ref_{tag}"))
    return hits[-1] if hits else None


def _missing(folder: Path, names: List[str]) -> List[str]:
    return [n for n in names if not (folder / n / "status.json").exists()]


def ref_one(i: int, s: str, key: str, jobs: int, dry_run: bool) -> Dict[str, Any]:
    """One scenario: launch, then complete the variants a start failure left out."""
    if not dry_run:
        time.sleep(LAUNCH_GAP_S * i)
    names = VARIANTS.split(",")
    tag = f"v4_{s}"
    note = (f"Ch 10 REF, scenario {s}, with the procedure-v4 weights {RESULT} ({key}); "
            "queued behind the v4 campaign (author 2026-11-04)")
    folder: Optional[Path] = None
    for attempt in range(1 + RETRIES):
        miss = names if folder is None else _missing(folder, names)
        if not miss:
            break
        argv = ["-m", "experiments.ch_11_case2_multisystem.ref.run_ref",
                "--variants", ",".join(miss), "--jobs", str(min(jobs, len(miss)))]
        argv += (["--scenario", s, "--minutes", "360", "--tag", tag, "--note", note]
                 if folder is None else ["--run-dir", P.rel(folder)])
        if dry_run:
            P.log("dry run: " + " ".join(argv))
            return {"scenario": s, "folder": None, "status": {}}
        rc = P.run(argv, OUT / f"ref_{s}.log")
        folder = _folder(tag)
        left = names if folder is None else _missing(folder, names)
        if left:
            P.log(f"REF {s}: attempt {attempt + 1} ended with rc {rc}; "
                  + ("no run folder" if folder is None else f"missing {left}"))
    status = {} if folder is None else {
        n: json.loads((folder / n / "status.json").read_text("utf-8"))["status"]
        for n in names if (folder / n / "status.json").exists()}
    P.log(f"REF {s}: {P.rel(folder) if folder else 'no run folder'}; {status}")
    return {"scenario": s, "folder": P.rel(folder) if folder else None, "status": status}


def main(argv: Optional[List[str]] = None) -> int:
    global OUT
    ap = argparse.ArgumentParser(prog="ch_9_16_ref_after_v4")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--no-w5-gate", action="store_true",
                    help="start the REF runs on GO even if the W5 report counts reversals")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--affinity-mask", type=lambda x: int(x, 0), default=AFFINITY_MASK)
    ap.add_argument("--dry-run", action="store_true",
                    help="gates, record and commands only: nothing written, committed or run")
    ap.add_argument("--paramset-dir", type=Path, default=None,
                    help="read the result from this folder (tests)")
    a = ap.parse_args(argv)
    V4._pin(a.affinity_mask)
    run = (ROOT / a.run).resolve() if not a.run.is_absolute() else a.run
    OUT = run / ("ref_v4_dry" if a.dry_run else "ref_v4")
    OUT.mkdir(parents=True, exist_ok=True)
    P.RUN = OUT                                   # P.log writes to ref_v4/driver.log
    scen = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    P.log(f"REF queue armed for {run.name}: scenarios {scen}, variants {VARIANTS}, "
          f"--jobs {a.jobs}, mask {a.affinity_mask:#x}, W5 gate {not a.no_w5_gate}"
          + (" (dry run)" if a.dry_run else ""))
    key = F.wait_for_campaign(run)
    if key is None:
        (OUT / "SKIPPED.txt").write_text("campaign not GO\n", "utf-8")
        return 0
    from configs.paramsets import PARAMSET_DIR
    path = (a.paramset_dir or PARAMSET_DIR) / f"{RESULT}.json"
    ok, why = w5_clean(json.loads(path.read_text("utf-8")))
    P.log(why)
    if not ok and not a.no_w5_gate:
        P.log("W5 gate: REF not started; the author decides")
        (OUT / "SKIPPED.txt").write_text(f"W5 gate: {why}\n", "utf-8")
        return 0
    record(path, a.dry_run)
    switch_selected(key, a.dry_run)
    with ThreadPoolExecutor(max_workers=len(scen)) as ex:
        res = list(ex.map(lambda t: ref_one(t[0], t[1], key, a.jobs, a.dry_run),
                          enumerate(scen)))
    if not a.dry_run:
        (OUT / "REF_DONE.json").write_text(json.dumps(
            {"selected": key, "paramset": RESULT, "runs": res,
             "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=1), "utf-8")
    P.log("REF queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
