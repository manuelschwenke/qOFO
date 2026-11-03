#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_12_followups_v3.py
=============================================================
The follow-ups of the procedure-v3 campaign (author, 2026-11-03), started only
when the campaign ends with ``GO.json`` (go criteria in
``docs/tuning/2026-11-03_weight_selection_procedure_v3.md``):

1. **SELECTED_PARAMSET** -> ``ch9_selected_uc_2026_11_v3`` in
   ``_ch9_selected_design.py``, committed -- only with ``--switch-selected``
   (the author decides the parameter set of record).  D1/D2 and the REF runs
   read it, so without the switch they are not started.
2. The code and the v3 parameter set to e5power (hash-checked copy).
3. **Settling figure** ``fig:param:timescales:settling_cdf``: the thesis
   generator ``graphics/Ch8/data/2026-10-28_settling/make_settling_cdf.py``,
   unchanged, copied to ``graphics/Ch8/data/<date>_settling/`` and pointed at
   the campaign (start: the Stage-I verification; final: the settling
   confirmation of the selected set).  It rewrites ``graphics/Ch8/settling_cdf.tex``
   in the thesis clone; nothing is committed there.
4. **D1 + D2** (``ch_9_4_deadband``) here, with ``--d-workers`` workers.
5. **Ch 10 REF**: scenarios A, B, D, E, GA, GE, each with L2, S1, S2, O2, O3, M,
   360 min, on e5power (one ``run_ref`` per scenario, 6 jobs each), launched
   90 s apart so the run folders are numbered one at a time; pulled back here
   when they end.

Usage (from the repository root; waits for the campaign)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_12_followups_v3.py \
        --run results/ch_09_parameter_selection/0005_2026-11-03_164755_ch_9_11_procedure_v3 \
        --switch-selected

Author: Manuel Schwenke / Claude Code (2026-11-03)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P  # noqa: E402

RESULT = "ch9_selected_uc_2026_11_v3"
DESIGN = ROOT / "experiments" / "ch_09_parameter_selection" / "_ch9_selected_design.py"
GEN = Path(r"Z:\latex_diss_clone\graphics\Ch8\data\2026-10-28_settling\make_settling_cdf.py")
SCENARIOS = ("A", "B", "D", "E", "GA", "GE")
VARIANTS = "L2,S1,S2,O2,O3,M"
REF_REL = Path("results") / "ch_11_case2_multisystem" / "ref"


def latest(d: Path) -> Path:
    return d / (d / "_latest.txt").read_text("utf-8").strip()


def wait_for_campaign(run: Path) -> Optional[str]:
    """The selected key on GO, ``None`` on NOGO or failure."""
    while not ((run / "DONE.txt").exists() or (run / "FAILED.txt").exists()):
        time.sleep(300)
    if (run / "FAILED.txt").exists():
        P.log("campaign FAILED; no follow-ups")
        return None
    if not (run / "GO.json").exists():
        P.log(f"campaign ended with NOGO ({(run / 'NOGO.json').read_text('utf-8')[:300]}); "
              "no follow-ups")
        return None
    key = (run / "DONE.txt").read_text("utf-8").split()[-1]
    P.log(f"campaign GO, selected {key}")
    return key


def switch_selected() -> None:
    text = DESIGN.read_text("utf-8")
    new, n = re.subn(r'^SELECTED_PARAMSET = "[^"]+"', f'SELECTED_PARAMSET = "{RESULT}"',
                     text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"SELECTED_PARAMSET line not found in {DESIGN}")
    if new != text:
        DESIGN.write_text(new, "utf-8")
        git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
        subprocess.check_call(git + ["add", str(DESIGN.relative_to(ROOT))])
        subprocess.check_call(git + ["commit", "-q", "-m",
                                     f"SELECTED_PARAMSET -> {RESULT} (procedure v3 GO; "
                                     "author 2026-11-03)\n\n"
                                     "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"])
    P.log(f"SELECTED_PARAMSET = {RESULT}")


def settling_figure(run: Path, key: str) -> None:
    dst = GEN.parent.parent / f"{datetime.now():%Y-%m-%d}_settling"
    dst.mkdir(parents=True, exist_ok=True)
    script = dst / GEN.name
    if not script.exists():
        shutil.copy2(GEN, script)
    spec = importlib.util.spec_from_file_location("make_settling_cdf", script)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    paths = {"start": latest(run / "stage1" / "verification" / "v2_pq10"),
             "final": latest(run / "expC" / f"cand_{key}")}
    gen.BUNDLE = str(run)

    def load():
        out = {}
        for k, p in paths.items():
            n = gen.pd.read_csv(p / "steps.csv").n_settle_final_1mvar.to_numpy()
            assert len(n) == 120 and not gen.np.isnan(n).any(), p
            assert n.max() <= gen.N_MAX, (p, n.max())
            out[k] = (str(p), n)
        return out

    gen.load = load
    argv, sys.argv = sys.argv, [str(script)]
    try:
        gen.main()
    finally:
        sys.argv = argv
    P.log(f"settling figure regenerated from {paths['start']} and {paths['final']} "
          f"(generator copy {script})")


def deadband(workers: int, out: Path) -> int:
    with (out / "deadband.log").open("a", encoding="utf-8") as fh:
        rc = subprocess.call([P.PY, "-X", "utf8", "-m",
                              "experiments.ch_09_parameter_selection.ch_9_4_deadband",
                              "--workers", str(workers)], cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    P.log(f"D1 + D2 exit {rc}")
    return rc


def ref_runs(out: Path, key: str) -> int:
    t_launch = datetime.now()

    def one(i: int, s: str) -> int:
        time.sleep(90.0 * i)
        argv = ["-m", "experiments.ch_11_case2_multisystem.ref.run_ref", "--scenario", s,
                "--variants", VARIANTS, "--minutes", "360", "--jobs", "6",
                "--note", f"Ch 10 REF, scenario {s}, with the procedure-v3 weights "
                          f"{RESULT} ({key}); follow-up of the v3 campaign"]
        return P.run_remote(f"ref_{s}", argv, out, [])

    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as ex:
        rcs = list(ex.map(one, range(len(SCENARIOS)), SCENARIOS))
    P.log(f"REF runs ended with {dict(zip(SCENARIOS, rcs))}")
    listing = P.ps(f"Invoke-Command -ComputerName {P.REMOTE} -ScriptBlock {{ "
                   f"Get-ChildItem {P._q(P.REMOTE_ROOT + chr(92) + str(REF_REL))} -Directory | "
                   f"Where-Object {{ $_.CreationTime -gt [datetime]'{t_launch:%Y-%m-%d %H:%M:%S}' }} | "
                   f"ForEach-Object {{ $_.Name }} }}").split()
    for name in listing:
        n = P.pull_dir(P.REMOTE_ROOT + "\\" + str(REF_REL / name), ROOT / REF_REL / name)
        P.log(f"pulled REF run {name} ({n} files)")
    return 0 if not any(rcs) else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="ch_9_12_followups_v3")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--switch-selected", action="store_true",
                    help="set SELECTED_PARAMSET to the v3 result on GO (author decision); "
                         "D1/D2 and REF run only with it")
    ap.add_argument("--d-workers", type=int, default=30)
    a = ap.parse_args(argv)
    run = (ROOT / a.run).resolve() if not a.run.is_absolute() else a.run
    out = run / "followups"
    out.mkdir(parents=True, exist_ok=True)
    P.RUN = out                                   # P.log writes to followups/driver.log
    P.log(f"follow-ups armed for {run.name} (switch SELECTED_PARAMSET: {a.switch_selected})")
    key = wait_for_campaign(run)
    if key is None:
        (out / "SKIPPED.txt").write_text("campaign not GO\n", "utf-8")
        return 0
    try:
        settling_figure(run, key)
    except Exception as exc:                      # noqa: BLE001 - the others still run
        P.log(f"settling figure not regenerated: {type(exc).__name__}: {exc}")
    if not a.switch_selected:
        P.log("SELECTED_PARAMSET not switched (no --switch-selected): D1/D2 and REF not started")
        (out / "FOLLOWUPS_DONE.txt").write_text("figure only\n", "utf-8")
        return 0
    switch_selected()
    from configs.paramsets import PARAMSET_DIR
    from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3
    P.push(PARAMSET_DIR / f"{RESULT}.json", P.REMOTE_ROOT + rf"\configs\paramsets\{RESULT}.json")
    V3.sync_remote_code()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_d = ex.submit(deadband, a.d_workers, out)
        f_r = ex.submit(ref_runs, out, key)
        rcs = (f_d.result(), f_r.result())
    (out / "FOLLOWUPS_DONE.txt").write_text(json.dumps(
        {"deadband_rc": rcs[0], "ref_rc": rcs[1], "selected": key,
         "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=1), "utf-8")
    P.log(f"follow-ups finished: D1+D2 rc {rcs[0]}, REF rc {rcs[1]}")
    return 0 if not any(rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
