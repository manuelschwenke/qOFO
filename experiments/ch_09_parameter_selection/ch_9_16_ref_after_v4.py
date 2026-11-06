#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_16_ref_after_v4.py
==============================================================
D1 + D2 and then the Ch 10 REF runs with the procedure-v4 weights, queued behind
the v4 campaign (author 2026-11-04: "queue the REF runs with new weights"; GA and
GE excluded, "some of the GE and GA scenarios didn't converge anyway"; "It should
wait for W5, and do D1 and D2 before"; e5wimi at 10 workers after 08:00).

Nothing starts unless the campaign ends with ``GO.json`` and the held-out W5
report counts no tap or bank reversal (``--no-w5-gate`` drops the W5 condition;
run 0006 was GO with 31 W5 reversals, and the author kept its follow-ups paused).

1. **Record.**  The v4 result gets what its selection ran with and what the
   downstream runs read, listed under ``provenance.record_additions``:

   * every ``--config-set`` field of the campaign as a parameter-set field
     (``sts_coupler_hold_after_bank = 1``), so D1/D2 and REF run the selected
     controllers (``apply_paramset`` applies any MultiTSOConfig field);
   * ``provenance.lambda_max_B`` = the v4 lambda without the bank columns (author
     B1b: banks excluded "as in variant M"), M's loop-gain ceiling in
     ``variants.selected_lambda_max``.  The v3/v4 results record only
     ``design_lambda_max_B``, so M would stop with a KeyError.

2. **Switch.**  SELECTED_PARAMSET -> the v4 result in ``_ch9_selected_design.py``;
   the design file and the parameter set are committed.
3. **D1 + D2** (``ch_9_4_deadband``) on e5power with 30 workers (author
   2026-11-04: "put d1 and d2 there"), idle once the campaign has ended: the
   parameter set and this checkout's code are copied there (hash-checked, as
   for the v3 follow-ups), the run is started through a detached shard
   (``ch_9_6_full_procedure.run_remote``) and its folder is pulled back here.
   thesis_runs numbers a run max + 1 over its own host's folders, so before the
   start e5power gets an empty folder of this host's highest number if its own
   are lower; the run then keeps its number here.
4. **Width gate.**  REF starts only if D1's design width equals the droop
   dead band the REF variants run with (``tso_qv_deadband_pu``,
   ``dso_qv_deadband_pu``); a different width is the author's decision.
5. **REF** here: scenarios A, B, D, E, each with the six variants, 360 min, 12
   processes in all (``--jobs`` 3 each; author 2026-11-05: "we can use 12 workers
   here"), launched 90 s apart so the run folders are numbered one at a time.  A
   start that dies before its variants finish (the llvmlite start failure of this
   host; a start failure exits within seconds and is repeated up to 12 times) is
   completed with ``--run-dir`` for the missing variants; a later failure is
   retried twice; a variant that fails in the model is a result and is not
   repeated.  D1 + D2 is retried the same way: a start that leaves no run
   folder is repeated, a run folder without ``d1/selection.json`` is completed
   with ``--reduce`` (its cached runs are kept) at most four times.

``--parallel-ref`` (author 2026-11-05: "Yes, REF in parallel") starts REF at the
same time as D1 + D2 instead of after the width gate.  The gate is then a check on
the finished REF runs, not a precondition: its verdict goes into ``REF_DONE.json``,
and a failed gate also writes ``WIDTH_GATE_FAILED.txt`` (the author decides).

The queue and everything it starts here are pinned to the 12-core mask.

Usage (from the repository root, on e5wimi with the author's logon, which the
remoting needs; waits for the campaign)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_16_ref_after_v4.py \
        --run results/ch_09_parameter_selection/0008_2026-11-04_133847_ch_9_14_procedure_v4

``--remote-smoke`` checks step 3 at once: ``ch_9_4_deadband --smoke`` on e5power
with the parameter set selected there, pulled back here; nothing else.

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
#: Scenario -> run_ref --jobs; 12 processes in all (e5wimi, author 2026-11-05).
SCENARIO_JOBS = {"A": 3, "B": 3, "D": 3, "E": 3}
#: Longest first (S1, M; REF runs of 2026-10-31), so the waiting ones are short.
VARIANTS = "S1,M,L2,O2,O3,S2"
REF_DIR = ROOT / F.REF_REL
DEADBAND_REL = Path("results") / "ch_09_parameter_selection"
DEADBAND_DIR = ROOT / DEADBAND_REL
DEADBAND = "ch_9_4_deadband"
REMOTE_WORKERS = 30                   # D1 + D2 on e5power (author 2026-11-04)
REMOTE_REF_JOBS = 6                   # --ref-remote: 4 x 6 = 24 of e5power's 30 workers
SMOKE_WORKERS = 5                     # --remote-smoke: W1, five runs in stage 2
AFFINITY_MASK = 0x555555              # e5wimi: 12 cores, one logical CPU each
LAUNCH_GAP_S = 90.0
#: An exit within START_FAIL_S is a start failure (llvmlite.dll, "[WinError 6]": 2 of
#: 3 starts of this queue on 2026-11-04 20:12) and is repeated up to MAX_STARTS times;
#: a later failure is retried RETRIES times.
START_FAIL_S = 120.0
MAX_STARTS = 12
RETRIES = 2
#: D1 + D2 completions with --reduce: a worker death breaks the whole pool (the
#: e5power smoke, 2026-11-04 21:31, one of 7 runs), and the cached runs are kept.
D_RETRIES = 4
OUT: Path = ROOT                      # set in main()
PARALLEL = False                      # --parallel-ref, set in main()


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
                                     "and lambda_max_B (provenance.record_additions); D1/D2 "
                                     "and REF queued by the author 2026-11-04.\n\n"
                                     "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"])
    P.log(f"SELECTED_PARAMSET = {RESULT} (commit {P.local_head()[:7]})")


# -- D1 + D2 and the width gate -----------------------------------------------------------

def _remote(script: str) -> str:
    return P.ps(f"Invoke-Command -ComputerName {P.REMOTE} -ScriptBlock {{ {script} }}")


def _rq(rel: Path) -> str:
    """A path below the repository root on e5power, quoted for PowerShell."""
    return P._q(P.REMOTE_ROOT + "\\" + str(rel))


def _counter(name: str) -> int:
    from experiments.thesis_runs import RUN_RE
    m = RUN_RE.match(name)
    return int(m["counter"]) if m else 0


def local_names(rel: Path = DEADBAND_REL) -> List[str]:
    """Run folders of the parent ``rel`` here and in its archive (thesis_runs counts both)."""
    return [d.name for base in (ROOT / rel, ROOT / "_local_archive" / rel)
            if base.is_dir() for d in base.iterdir() if d.is_dir()]


def remote_names(rel: Path = DEADBAND_REL) -> List[str]:
    """The same on e5power."""
    return _remote("; ".join(
        f"if (Test-Path {_rq(p)}) {{ Get-ChildItem {_rq(p)} -Directory | "
        f"ForEach-Object {{ $_.Name }} }}"
        for p in (rel, Path("_local_archive") / rel))).split()


def reserve_number(rel: Path = DEADBAND_REL) -> int:
    """The number e5power will give the run, made at least this host's next one."""
    loc = [n for n in local_names(rel) if _counter(n)]
    n_loc = max(map(_counter, loc), default=0) + 1
    n_rem = max(map(_counter, remote_names(rel)), default=0) + 1
    if n_rem < n_loc:
        top = max(loc, key=_counter)
        _remote(f"New-Item -ItemType Directory -Force -Path {_rq(rel / top)} | Out-Null")
        P.log(f"{P.REMOTE}: empty {top} created (the number is taken here)")
    return max(n_loc, n_rem)


def push_inputs(paramset: Path) -> None:
    """The v4 parameter set and this checkout's code, to e5power."""
    from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3
    P.push(paramset, P.REMOTE_ROOT + rf"\configs\paramsets\{RESULT}.json")
    V3.sync_remote_code()


def _pull(name: str, rc: Optional[int], synced: bool, rel: Path = DEADBAND_REL,
          what: str = "D1 + D2", need: Optional[str] = "d1/selection.json") -> Optional[Path]:
    clash = [n for n in local_names(rel) if _counter(n) == _counter(name) and n != name]
    if clash:
        P.log(f"{what}: number {_counter(name):04d} is taken here by {clash} as well")
    local = ROOT / rel / name
    n = P.pull_dir(P.REMOTE_ROOT + "\\" + str(rel / name), local, True)
    P.log(f"{what}: {P.rel(local)} pulled from {P.REMOTE} ({n} files), rc {rc}")
    from experiments.thesis_runs import annotate_run
    host = {"host": P.REMOTE, "remote_root": P.REMOTE_ROOT, "remote_head": P.remote_head(),
            "code": (f"this checkout's commit {P.local_head()}, copied over the remote "
                     "commit and hash-checked (ch_9_11_procedure_v3.sync_remote_code)"
                     if synced else "the remote checkout as it was (not copied)"),
            "pulled": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "by": P.rel(Path(__file__))}
    (local / "remote_host.json").write_text(json.dumps(host, indent=1), "utf-8")
    annotate_run(local, by=Path(__file__), outputs={
        "remote_host.json": f"simulated on {P.REMOTE}, pulled back to this host"})
    return local if need is None or (local / need).exists() else None


def deadband(key: str, dry_run: bool, smoke: bool = False) -> Optional[Path]:
    """D1 + D2 on e5power at the weights selected there; the pulled run folder, or None."""
    note = (f"D1/D2 at the procedure-v4 weights {RESULT} ({key}) on {P.REMOTE}; queued "
            "behind the v4 campaign (author 2026-11-04)")
    if smoke:
        note = (f"Smoke of D1/D2 on {P.REMOTE} (the path of {Path(__file__).name}), "
                f"at the parameter set selected there")
    name: Optional[str] = None
    starts = fails = 0
    while starts < MAX_STARTS and fails <= D_RETRIES:
        argv = ["-m", f"experiments.ch_09_parameter_selection.{DEADBAND}",
                "--workers", str(SMOKE_WORKERS if smoke else REMOTE_WORKERS)]
        argv += ["--smoke"] if smoke else []
        argv += ["--note", note] if name is None else ["--reduce", str(DEADBAND_REL / name)]
        if dry_run:
            P.log(f"dry run: on {P.REMOTE}: " + " ".join(argv))
            return None
        if name is None:
            P.log(f"D1 + D2: {P.REMOTE} numbers the run {reserve_number():04d}")
        before = set(remote_names())
        starts += 1
        try:
            rc: Optional[int] = P.run_remote(f"deadband_{time.strftime('%m%d_%H%M%S')}",
                                             argv, OUT, [])
        except RuntimeError as exc:              # the shard died 3 times, or no remoting
            rc = None
            P.log(f"D1 + D2: start {starts}: {exc}")
        new = sorted(n for n in set(remote_names()) - before if f"_{DEADBAND}" in n)
        name = name or (new[-1] if new else None)
        # A start failure leaves no folder (start anew); a broken pool leaves one
        # whose cached runs --reduce keeps (done.json) and whose missing ones it runs.
        fails += name is not None or rc is None
        if name is not None and _remote(
                f"Test-Path {_rq(DEADBAND_REL / name / 'd1' / 'selection.json')}").strip() == "True":
            return _pull(name, rc, synced=not smoke)
        P.log(f"D1 + D2: start {starts} ended with rc {rc}; "
              + ("no run folder" if name is None else "no d1/selection.json"))
    return None


def ref_width() -> Dict[str, float]:
    """The droop dead band the REF variants run with (identical in all of them)."""
    from experiments.ch_11_case2_multisystem import variants as v
    w = {}
    for n in VARIANTS.split(","):
        c = v.resolved(n, 10)
        w[n] = (float(c.tso_qv_deadband_pu), float(c.dso_qv_deadband_pu))
    if len(set(w.values())) != 1:
        raise RuntimeError(f"REF variants differ in the dead band: {w}")
    tso, dso = next(iter(w.values()))
    return {"tso_qv_deadband_pu": tso, "dso_qv_deadband_pu": dso}


def width_gate(folder: Path) -> Tuple[bool, str]:
    sel = json.loads((folder / "d1" / "selection.json").read_text("utf-8"))
    d = float(sel["delta_design_pu"])
    ref = ref_width()
    ok = all(abs(d - x) < 1e-9 for x in ref.values())
    return ok, (f"D1 design width {d:g} pu (rule {sel['delta_rule_pu']:g} pu); REF runs "
                f"with {ref}")


# -- REF ------------------------------------------------------------------------------------

def _folder(tag: str) -> Optional[Path]:
    hits = sorted(REF_DIR.glob(f"*_run_ref_{tag}"))
    return hits[-1] if hits else None


def _missing(folder: Path, names: List[str]) -> List[str]:
    return [n for n in names if not (folder / n / "status.json").exists()]


def ref_one(i: int, s: str, key: str, dry_run: bool) -> Dict[str, Any]:
    """One scenario: launch, then complete the variants a start failure left out."""
    if not dry_run:
        time.sleep(LAUNCH_GAP_S * i)
    names = VARIANTS.split(",")
    tag = f"v4_{s}"
    note = (f"Ch 10 REF, scenario {s}, with the procedure-v4 weights {RESULT} ({key}); "
            "queued behind the v4 campaign (author 2026-11-04)")
    if PARALLEL:
        note += ("; started with D1/D2, so the width gate is checked after this run "
                 "(ref_v4/REF_DONE.json; author 2026-11-05)")
    folder: Optional[Path] = None
    starts = fails = 0
    while starts < MAX_STARTS and fails <= RETRIES:
        miss = names if folder is None else _missing(folder, names)
        if not miss:
            break
        argv = ["-m", "experiments.ch_11_case2_multisystem.ref.run_ref",
                "--variants", ",".join(miss), "--jobs", str(min(SCENARIO_JOBS[s], len(miss)))]
        argv += (["--scenario", s, "--minutes", "360", "--tag", tag, "--note", note]
                 if folder is None else ["--run-dir", P.rel(folder)])
        if dry_run:
            P.log("dry run: " + " ".join(argv))
            return {"scenario": s, "folder": None, "status": {}}
        t0 = time.time()
        rc = P.run(argv, OUT / f"ref_{s}.log")
        starts += 1
        fails += time.time() - t0 >= START_FAIL_S
        folder = _folder(tag)
        left = names if folder is None else _missing(folder, names)
        if left:
            P.log(f"REF {s}: start {starts} ended with rc {rc}; "
                  + ("no run folder" if folder is None else f"missing {left}"))
    status = {} if folder is None else {
        n: json.loads((folder / n / "status.json").read_text("utf-8"))["status"]
        for n in names if (folder / n / "status.json").exists()}
    P.log(f"REF {s}: {P.rel(folder) if folder else 'no run folder'}; {status}")
    return {"scenario": s, "folder": P.rel(folder) if folder else None, "status": status}


def ref_remote_one(i: int, s: str, key: str, dry_run: bool) -> Dict[str, Any]:
    """One scenario on e5power (``--ref-remote``): launch, complete, pull back here."""
    if not dry_run:
        time.sleep(LAUNCH_GAP_S * i)
    rel, names, tag = F.REF_REL, VARIANTS.split(","), f"v4_{s}"
    note = (f"Ch 10 REF, scenario {s}, with the procedure-v4 weights {RESULT} ({key}); "
            f"on {P.REMOTE} with --jobs {REMOTE_REF_JOBS} (author 2026-11-05: \"Yes, REF to "
            "e5power\"); width gate checked on the D1 + D2 run (ref_v4/REF_DONE.json)")
    name: Optional[str] = None
    local: Optional[Path] = None
    starts = fails = 0
    while starts < MAX_STARTS and fails <= RETRIES:
        miss = names if local is None else _missing(local, names)
        if not miss:
            break
        argv = ["-m", "experiments.ch_11_case2_multisystem.ref.run_ref",
                "--variants", ",".join(miss), "--jobs", str(min(REMOTE_REF_JOBS, len(miss)))]
        argv += (["--scenario", s, "--minutes", "360", "--tag", tag, "--note", note]
                 if name is None else ["--run-dir", str(rel / name)])
        if dry_run:
            P.log(f"dry run: on {P.REMOTE}: " + " ".join(argv))
            return {"scenario": s, "folder": None, "status": {}}
        if name is None:
            P.log(f"REF {s}: {P.REMOTE} numbers the run {reserve_number(rel):04d}")
        before = set(remote_names(rel))
        starts += 1
        t0 = time.time()
        try:
            rc: Optional[int] = P.run_remote(f"ref_{s}_{time.strftime('%m%d_%H%M%S')}",
                                             argv, OUT, [])
        except RuntimeError as exc:              # the shard died 3 times, or no remoting
            rc = None
            P.log(f"REF {s}: start {starts}: {exc}")
        fails += rc is None or time.time() - t0 >= START_FAIL_S
        new = sorted(n for n in set(remote_names(rel)) - before if n.endswith(f"_run_ref_{tag}"))
        name = name or (new[-1] if new else None)
        if name is not None:
            local = _pull(name, rc, synced=True, rel=rel, what=f"REF {s}", need=None)
        left = names if local is None else _missing(local, names)
        if left:
            P.log(f"REF {s}: start {starts} ended with rc {rc}; "
                  + ("no run folder" if name is None else f"missing {left}"))
    status = {} if local is None else {
        n: json.loads((local / n / "status.json").read_text("utf-8"))["status"]
        for n in names if (local / n / "status.json").exists()}
    P.log(f"REF {s}: {P.rel(local) if local else 'no run folder'}; {status}")
    return {"scenario": s, "folder": P.rel(local) if local else None, "status": status,
            "host": P.REMOTE}


def ref_remote(key: str, scen: List[str], folder: Path, dry_run: bool) -> int:
    """REF on e5power, all scenarios at once; the width gate of the given D1 + D2 run."""
    ok, why = width_gate(folder)
    P.log(f"width gate ({P.rel(folder)}): {why}")
    with ThreadPoolExecutor(max_workers=len(scen)) as ex:
        res = list(ex.map(lambda t: ref_remote_one(t[0], t[1], key, dry_run), enumerate(scen)))
    if dry_run:
        P.log("queue finished (dry run)")
        return 0
    if not ok:
        (OUT / "WIDTH_GATE_FAILED.txt").write_text(why + "\n", "utf-8")
    (OUT / "REF_DONE.json").write_text(json.dumps(
        {"selected": key, "paramset": RESULT, "remote": P.REMOTE, "deadband": P.rel(folder),
         "width_gate": {"ok": ok, "why": why}, "runs": res,
         "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=1), "utf-8")
    P.log("queue finished" + ("" if ok else "; width gate failed, the author decides"))
    return 0


def _stop(why: str) -> int:
    P.log(f"{why}; stopped, the author decides")
    (OUT / "SKIPPED.txt").write_text(why + "\n", "utf-8")
    return 0


def parallel(key: str, scen: List[str], dry_run: bool) -> int:
    """D1 + D2 on e5power and REF here at once; the width gate then judges REF."""
    with ThreadPoolExecutor(max_workers=1 + len(scen)) as ex:
        fut = ex.submit(deadband, key, dry_run)
        res = list(ex.map(lambda t: ref_one(t[0], t[1], key, dry_run), enumerate(scen)))
        folder = fut.result()
    ok, why = (width_gate(folder) if folder is not None
               else (False, "D1 + D2 gave no d1/selection.json"))
    P.log(f"width gate on the parallel REF runs: {why}")
    if dry_run:
        P.log("queue finished (dry run)")
        return 0
    if not ok:
        (OUT / "WIDTH_GATE_FAILED.txt").write_text(why + "\n", "utf-8")
    (OUT / "REF_DONE.json").write_text(json.dumps(
        {"selected": key, "paramset": RESULT, "parallel": True,
         "deadband": P.rel(folder) if folder else None,
         "width_gate": {"ok": ok, "why": why}, "runs": res,
         "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=1), "utf-8")
    P.log("queue finished" + ("" if ok else "; width gate failed, the author decides"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    global OUT, PARALLEL
    ap = argparse.ArgumentParser(prog="ch_9_16_ref_after_v4")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--no-w5-gate", action="store_true",
                    help="go on after GO even if the W5 report counts reversals")
    ap.add_argument("--scenarios", default=",".join(SCENARIO_JOBS))
    ap.add_argument("--affinity-mask", type=lambda x: int(x, 0), default=AFFINITY_MASK)
    ap.add_argument("--dry-run", action="store_true",
                    help="gates, record and commands only: nothing written, committed or run")
    ap.add_argument("--paramset-dir", type=Path, default=None,
                    help="read the result from this folder (tests)")
    ap.add_argument("--remote-smoke", action="store_true",
                    help=f"only {DEADBAND} --smoke on {P.REMOTE}, pulled back here, at once")
    ap.add_argument("--parallel-ref", action="store_true",
                    help="REF here at the same time as D1 + D2; the width gate judges it after")
    ap.add_argument("--ref-remote", type=Path, default=None, metavar="DEADBAND_RUN",
                    help="REF on e5power after the given D1 + D2 run (no new D1 + D2)")
    a = ap.parse_args(argv)
    PARALLEL = a.parallel_ref
    V4._pin(a.affinity_mask)
    run = (ROOT / a.run).resolve() if not a.run.is_absolute() else a.run
    OUT = run / ("ref_v4_dry" if a.dry_run else "ref_v4_smoke" if a.remote_smoke else "ref_v4")
    OUT.mkdir(parents=True, exist_ok=True)
    P.RUN = OUT                                   # P.log writes to ref_v4/driver.log
    if not a.dry_run:                             # the remoting works from this process
        P.log(f"{P.REMOTE} at {P.remote_head()[:7]}")
    if a.remote_smoke:
        folder = deadband("remote smoke", dry_run=False, smoke=True)
        P.log(f"remote smoke: {P.rel(folder) if folder else 'no d1/selection.json'}")
        return 0 if folder else 1
    scen = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    P.log(f"queue armed for {run.name}: D1 + D2 ({P.REMOTE}, {REMOTE_WORKERS} workers), "
          f"{'with' if a.parallel_ref else 'then'} REF here {scen} "
          f"(jobs {[SCENARIO_JOBS[s] for s in scen]}), variants "
          f"{VARIANTS}, mask {a.affinity_mask:#x}, W5 gate {not a.no_w5_gate}"
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
        return _stop(f"W5 gate: {why}")
    record(path, a.dry_run)
    switch_selected(key, a.dry_run)
    if not a.dry_run:
        push_inputs(path)
    if a.ref_remote is not None:
        folder = a.ref_remote if a.ref_remote.is_absolute() else ROOT / a.ref_remote
        return ref_remote(key, scen, folder, a.dry_run)
    if a.parallel_ref:
        return parallel(key, scen, a.dry_run)
    folder = deadband(key, a.dry_run)
    if folder is not None:
        ok, why = width_gate(folder)
        P.log(why)
        if not ok:
            return _stop(f"width gate: {why}")
    elif not a.dry_run:
        return _stop("D1 + D2 gave no d1/selection.json")
    with ThreadPoolExecutor(max_workers=len(scen)) as ex:
        res = list(ex.map(lambda t: ref_one(t[0], t[1], key, a.dry_run), enumerate(scen)))
    if not a.dry_run:
        (OUT / "REF_DONE.json").write_text(json.dumps(
            {"selected": key, "paramset": RESULT, "deadband": P.rel(folder), "runs": res,
             "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=1), "utf-8")
    P.log("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
