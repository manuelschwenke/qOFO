#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_18_followups_v5.py
=============================================================
Follow-ups of the procedure-v5 campaign (author 2026-11-08: "start all
experiments (B, C, D1, D2)"; 30 workers on e5power, 20 on e5wimi).  Waits for the
campaign; nothing starts unless it ends with ``GO.json``.

1. **Record.**  The v5 result gets the campaign's config fields
   (``sts_coupler_hold_after_bank = 1``) and ``provenance.lambda_max_B``
   (``ch_9_16_ref_after_v4.record``).
2. **Switch.**  SELECTED_PARAMSET -> ``ch9_selected_nr_2026_11_v5``, committed on
   this checkout's branch.
3. **C:** the settling figure (``ch_9_12_followups_v3.settling_figure``: the
   Stage-I verification against the winner's confirmation).
4. **D1 + D2** (``ch_9_4_deadband``) on e5power with 30 workers
   (``ch_9_16_ref_after_v4.deadband``; code and parameter set copied there and
   hash-checked), and at the same time
5. **G3-off report** here with 20 workers: the winner in the G3-off episodes of
   the mini-MC (e5power run 0007 of 2026-11-08, ``rural_700_north_res``, 180 min),
   variants O2 (OFO) and O3 (OFO+) (procedure v5, change 4; reported, not ranked).

``FOLLOWUPS_DONE.json`` lists what ran; REF is not part of this queue.

Usage (from the repository root, on e5wimi with the author's logon)::

    pythonw -X utf8 experiments/ch_09_parameter_selection/ch_9_18_followups_v5.py \
        --run results/ch_09_parameter_selection/<v5 run>

Author: Manuel Schwenke / Claude Code (2026-11-08)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_12_followups_v3 as F  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_16_ref_after_v4 as Q  # noqa: E402
from experiments.ch_09_parameter_selection import ch_9_17_procedure_v5 as V5  # noqa: E402

RESULT = V5.RESULT
#: The mini-MC of the network decision: its prescreen holds each seed's episode and
#: unit commitment on rural_700_north_res; the G3-off seeds are the report's episodes.
MINI_MC = Path(r"F:\qOFO_net\results\ch_08_setup\network_scarcity\_remote_e5power"
               r"\0007_2026-11-08_160221_mini_mc_north_res")
#: The 21 seeds of run 0007 whose rural_700_north_res commitment keeps G3 off (its
#: README; the prescreen's ``uc_offline`` is the base network's and does not say
#: this).  Confirmed by this session's 3-h captures for all but 20261110, whose
#: prescreen is "valid": false (all variants failed in MC 0002); run 0007 ran it.
G3_OFF_SEEDS = (20261010, 20261014, 20261024, 20261038, 20261057, 20261063, 20261069,
                20261070, 20261074, 20261077, 20261086, 20261094, 20261100, 20261110,
                20261113, 20261122, 20261128, 20261132, 20261134, 20261137, 20261138)
REPORT_VARIANTS = "O2,O3"
LOCAL_JOBS = 20


def switch_selected(key: str) -> None:
    from configs.paramsets import PARAMSET_DIR
    text = F.DESIGN.read_text("utf-8")
    new, n = re.subn(r'^SELECTED_PARAMSET = "[^"]+"', f'SELECTED_PARAMSET = "{RESULT}"',
                     text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"SELECTED_PARAMSET line not found in {F.DESIGN}")
    if new != text:
        F.DESIGN.write_text(new, "utf-8")
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    subprocess.check_call(git + ["add", P.rel(F.DESIGN), P.rel(PARAMSET_DIR / f"{RESULT}.json")])
    if subprocess.call(git + ["diff", "--cached", "--quiet"]) != 0:
        subprocess.check_call(git + ["commit", "-q", "-m",
                                     f"SELECTED_PARAMSET -> {RESULT} (procedure v5 GO, {key})\n\n"
                                     "The parameter set carries the campaign's config fields "
                                     "and lambda_max_B (provenance.record_additions); C, D1/D2 "
                                     "and the G3-off report follow (author 2026-11-08).\n\n"
                                     "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"])
    P.log(f"SELECTED_PARAMSET = {RESULT} (commit {P.local_head()[:7]})")


def g3_off_seeds() -> List[int]:
    missing = [s for s in G3_OFF_SEEDS if not (MINI_MC / "prescreen" / f"{s}.json").exists()]
    if missing:
        raise RuntimeError(f"prescreen of {missing} not in {MINI_MC}")
    return list(G3_OFF_SEEDS)


def g3_off_report(key: str, out: Path) -> Dict[str, Any]:
    seeds = g3_off_seeds()
    inp = out / "g3off_input"
    (inp / "prescreen").mkdir(parents=True, exist_ok=True)
    for s in seeds:
        shutil.copy2(MINI_MC / "prescreen" / f"{s}.json", inp / "prescreen" / f"{s}.json")
    (inp / "seeds.json").write_text(json.dumps(seeds), "utf-8")
    argv = ["-m", "experiments.ch_08_setup.network_scarcity.mini_mc",
            "--seeds", P.rel(inp / "seeds.json"), "--prescreen", P.rel(inp / "prescreen"),
            "--variants", REPORT_VARIANTS, "--jobs", str(LOCAL_JOBS), "--tag", "v5_g3off",
            "--note", f"G3-off report of procedure v5 ({RESULT}, {key}): {len(seeds)} G3-off "
                      f"seeds of mini-MC 0007, OFO and OFO+ (reported, not ranked)"]
    P.log(f"G3-off report: {len(seeds)} seeds, variants {REPORT_VARIANTS}, {LOCAL_JOBS} jobs")
    rc = P.run(argv, out / "g3off_report.log")
    base = ROOT / "results" / "ch_08_setup" / "network_scarcity"
    runs = sorted(base.glob("*_mini_mc_v5_g3off"), key=lambda d: d.stat().st_mtime)
    return {"rc": rc, "seeds": seeds, "variants": REPORT_VARIANTS,
            "folder": P.rel(runs[-1]) if runs else None}


def deadband(key: str, out: Path) -> Dict[str, Any]:
    from configs.paramsets import PARAMSET_DIR
    Q.push_inputs(PARAMSET_DIR / f"{RESULT}.json")
    folder = Q.deadband(key, dry_run=False, note=(
        f"D1/D2 at the procedure-v5 weights {RESULT} ({key}) on {P.REMOTE}; queued behind "
        "the v5 campaign (author 2026-11-08: B, C, D1, D2)"))
    if folder is None:
        return {"folder": None, "width_gate": None}
    ok, why = Q.width_gate(folder)
    return {"folder": P.rel(folder), "width_gate": {"ok": ok, "why": why}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="ch_9_18_followups_v5")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--affinity-mask", type=lambda x: int(x, 0), default=V5.AFFINITY_MASK)
    a = ap.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    from experiments.ch_09_parameter_selection import ch_9_14_procedure_v4 as V4
    V4._pin(a.affinity_mask)
    run = (ROOT / a.run).resolve()
    out = run / "followups_v5"
    out.mkdir(parents=True, exist_ok=True)
    P.RUN = out
    P.REMOTE_ROOT = V5.REMOTE_ROOT
    Q.RESULT = RESULT
    Q.OUT = out
    P.log(f"queue armed for {run.name}: record, switch, C, D1 + D2 on {P.REMOTE} "
          f"({Q.REMOTE_WORKERS} workers, {P.REMOTE_ROOT}), G3-off report here ({LOCAL_JOBS})")
    key = F.wait_for_campaign(run)
    if key is None:
        (out / "SKIPPED.txt").write_text("campaign not GO\n", "utf-8")
        return 0
    from configs.paramsets import PARAMSET_DIR
    Q.record(PARAMSET_DIR / f"{RESULT}.json", dry_run=False)
    switch_selected(key)
    done: Dict[str, Any] = {"key": key, "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    try:
        F.settling_figure(run, key)
        done["C"] = "settling figure regenerated"
    except Exception as exc:                      # noqa: BLE001 - the others still run
        done["C"] = f"not regenerated: {type(exc).__name__}: {exc}"
        P.log(f"settling figure not regenerated: {type(exc).__name__}: {exc}")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fd = ex.submit(deadband, key, out)
        fg = ex.submit(g3_off_report, key, out)
        for name, fut in (("D1+D2", fd), ("G3-off report", fg)):
            try:
                done[name] = fut.result()
            except Exception as exc:              # noqa: BLE001
                done[name] = f"{type(exc).__name__}: {exc}"
                P.log(f"{name} failed: {type(exc).__name__}: {exc}")
    done["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    (out / "FOLLOWUPS_DONE.json").write_text(json.dumps(done, indent=1, default=str), "utf-8")
    P.log(f"follow-ups done: {json.dumps(done, default=str)[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
