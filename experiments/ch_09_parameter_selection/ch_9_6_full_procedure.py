#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py
==============================================================
The complete controller weight selection on the unit-commitment base of
2026-10-31 (author decision 2026-10-31, after the re-validation run 0010 did
not confirm c08b1d3d33ab).  Declared before the run in
``docs/analysis/2026-10-31_ch9_full_procedure_uc_declaration.md``.

The procedure is ``docs/tuning/2026-10-28_weight_selection_procedure.md``
(frozen), unchanged; this driver follows run 0001's ``driver.py`` step by step
and adds the engagement bound of amendment 1 to every selection, as the frozen
text requires:

  1. Stage I: ``tuning.stage1_screen --design`` on operating points captured
     with the commitment of the 12 h window (``--op-horizon-h 12``), written as
     ``configs/paramsets/stage1_start_uc_2026_10.json``.
  2. Stage II-a: 512 Sobol draws around it, screened at the select ceiling,
     survivors and the Stage-I point simulated 12 h in W1-W4, W6.
  3. Stage II-b: rounds of 64 draws around the current best (selection rule
     without settling), f halved when the best is unchanged, at most 4 rounds.
  4. Settling requirement in rank order (Experiment C), four candidates at a
     time as in run 0001.
  5. W5 for the result and the Stage-I point, for the report only.
  6. Result written as ``configs/paramsets/ch9_selected_uc_2026_10.json``.

Each stage's units are split over two hosts: this one (the master) and
e5power, reached through PowerShell remoting.  A shard is an unchanged
``ch_9_3_stage2_screen.py`` (or Experiment C) invocation on part of the
candidates; the window files come back to the master, which assembles the
candidate records with one more invocation that finds every unit on disk.

Usage (from the repository root; the master needs the user's own logon for the
remoting, so start it on this host, not through a remote session)::

    python -X utf8 experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py --note "..."
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py --resume <run dir>
    python -X utf8 experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py shard <spec.json>

Author: Manuel Schwenke / Claude Code (2026-10-31)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = sys.executable
START = "stage1_start_uc_2026_10"
RESULT = "ch9_selected_uc_2026_10"
SEED = 20260910
N_2A = 512
N_LOCAL, F0, F_MIN, MAX_ROUNDS = 64, 0.5, 0.125, 4
WINDOWS = "W1,W2,W3,W4,W6"               # W5 enters neither filter nor ranking
HOURS = 12.0
BAND = (0.92, 1.08)
BANK_BUDGET = 12.0
N_INNER = 9
C_PARALLEL = 4                           # settling checks at a time, as run 0001
#: Engagement bound on theta_STS,min [pu]: amendment 2 (author, 2026-10-31,
#: docs/tuning/2026-10-31_amendment2_sts_engagement_0p02.md), declared before
#: the first selection of this run; amendment 1 had V_max,band - V_ref = 0.05.
THETA_MAX = 0.02
D_ENGAGE = {"ts": 0.0075, "sts": 0.015, "shunt": 0.010}
SIM_HOURS = HOURS                        # simulated length; HOURS fixes the capture
EXPC_EXTRA: List[str] = []
SMOKE = False

REMOTE = "e5power"
REMOTE_ROOT = r"F:\qOFO_runs\qOFO_res"
REMOTE_PY = r"F:\python_environments\qOFO_clean\python.exe"
PS = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
POLL_S = 300.0

_LOG_LOCK = threading.Lock()
_PULL_LOCK = threading.Lock()            # shard-log pulls share their destination
RUN: Path = ROOT                         # set in main()


# -- helpers ---------------------------------------------------------------------------

def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with _LOG_LOCK, (RUN / "driver.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def rel(p: Path) -> str:
    return os.path.relpath(Path(p).resolve(), ROOT)


def remote_path(p: Path) -> str:
    return REMOTE_ROOT + "\\" + rel(p)


def run(args: List[str], logfile: Path) -> int:
    log(f"run: {' '.join(args)}  > {logfile.name}")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-X", "utf8", *args], cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    log(f"exit {rc}: {logfile.name}")
    return rc


def ps(script: str, timeout: float = 1800.0) -> str:
    """Run a PowerShell script on this host (it reaches e5power by remoting)."""
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    for attempt in range(3):
        r = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
        # The tail: stderr opens with CLIXML progress records, the error comes last.
        log(f"powershell attempt {attempt + 1} failed: {r.stderr.strip()[-1500:]}")
        time.sleep(30.0)
    raise RuntimeError(f"powershell failed three times: {r.stderr.strip()[-1500:]}")


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def push(local: Path, remote: str) -> None:
    """Copy one file to e5power (parent created)."""
    parent = remote.rsplit("\\", 1)[0]
    ps(f"$s = New-PSSession -ComputerName {REMOTE}; "
       f"Invoke-Command -Session $s -ScriptBlock {{ New-Item -ItemType Directory -Force "
       f"-Path {_q(parent)} | Out-Null }}; "
       f"Copy-Item -ToSession $s -Path {_q(str(local))} -Destination {_q(remote)} -Force; "
       f"Remove-PSSession $s")


def pull_dir(remote_dir: str, local_dir: Path, overwrite: bool = False) -> int:
    """Bring a directory back from e5power (zipped in transit); files already
    present here are kept unless ``overwrite``.  Returns the number copied."""
    stamp = f"{os.getpid()}_{time.time_ns()}"
    rzip = REMOTE_ROOT + rf"\results\_transfer\pull_{stamp}.zip"
    stage = RUN / "_transfer" / f"pull_{stamp}"
    stage.mkdir(parents=True, exist_ok=True)
    lzip = stage.with_suffix(".zip")
    ps(f"$s = New-PSSession -ComputerName {REMOTE}; "
       f"Invoke-Command -Session $s -ScriptBlock {{ "
       f"New-Item -ItemType Directory -Force -Path {_q(REMOTE_ROOT + chr(92) + 'results' + chr(92) + '_transfer')} | Out-Null; "
       f"Compress-Archive -Path {_q(remote_dir + chr(92) + '*')} -DestinationPath {_q(rzip)} -Force }}; "
       f"Copy-Item -FromSession $s -Path {_q(rzip)} -Destination {_q(str(lzip))} -Force; "
       f"Invoke-Command -Session $s -ScriptBlock {{ Remove-Item {_q(rzip)} -Force }}; "
       f"Remove-PSSession $s; "
       f"Expand-Archive -Path {_q(str(lzip))} -DestinationPath {_q(str(stage))} -Force")
    n = 0
    for f in stage.rglob("*"):
        if f.is_file():
            dst = local_dir / f.relative_to(stage)
            if dst.exists() and not overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    shutil.rmtree(stage, ignore_errors=True)
    lzip.unlink(missing_ok=True)
    return n


def remote_head() -> str:
    return ps(f"Invoke-Command -ComputerName {REMOTE} -ScriptBlock {{ "
              f"git -c safe.directory=* -C {_q(REMOTE_ROOT)} rev-parse HEAD }}").strip()


def local_head() -> str:
    return subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(ROOT),
                                    "rev-parse", "HEAD"], text=True).strip()


# -- shards ----------------------------------------------------------------------------

def shard_main(spec_path: Path) -> int:
    """Worker-host side: run one invocation, then write the done marker."""
    spec = json.loads(spec_path.read_text("utf-8"))
    logf = ROOT / spec["log"]
    logf.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / spec["pid"]).write_text(str(os.getpid()), "utf-8")
    with logf.open("a", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-X", "utf8", *spec["argv"]], cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    (ROOT / spec["done"]).write_text(json.dumps(
        {"rc": rc, "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}), "utf-8")
    return rc


def run_remote(tag: str, argv: List[str], where: Path,
               pull: List[Tuple[Path, bool]], max_launches: int = 3) -> int:
    """Run ``argv`` on e5power through a detached shard process; wait for its
    done marker; pull the listed directories (``(dir, overwrite)``) back."""
    rdir = where / "_remote"
    rdir.mkdir(parents=True, exist_ok=True)
    spec = {"argv": argv, "log": rel(rdir / f"{tag}.log"),
            "done": rel(rdir / f"{tag}.done.json"), "pid": rel(rdir / f"{tag}.pid")}
    sp = rdir / f"{tag}.spec.json"
    sp.write_text(json.dumps(spec, indent=1), "utf-8")
    push(sp, remote_path(sp))
    done_r = REMOTE_ROOT + "\\" + spec["done"]
    script_r = REMOTE_ROOT + r"\experiments\ch_09_parameter_selection\ch_9_6_full_procedure.py"

    def status() -> Tuple[Optional[dict], bool]:
        out = ps(f"Invoke-Command -ComputerName {REMOTE} -ScriptBlock {{ "
                 f"if (Test-Path {_q(done_r)}) {{ 'DONE ' + (Get-Content {_q(done_r)} -Raw) }} "
                 f"else {{ $p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 f"Where-Object {{ $_.CommandLine -like {_q('*' + tag + '.spec.json*')} }}; "
                 f"if ($p) {{ 'ALIVE' }} else {{ 'DEAD' }} }} }}").strip()
        if out.startswith("DONE "):
            return json.loads(out[5:]), False
        return None, out == "ALIVE"

    def status_tolerant() -> Tuple[Optional[dict], bool]:
        # A remoting hiccup is not a dead shard: retry for up to an hour.
        for k in range(12):
            try:
                return status()
            except Exception as exc:                          # noqa: BLE001
                log(f"remote {tag}: status unavailable ({exc}); retry {k + 1}")
                time.sleep(POLL_S)
        raise RuntimeError(f"remote {tag}: {REMOTE} unreachable for an hour")

    done, alive = status_tolerant()
    launches = 0
    while done is None:
        if not alive:
            if launches >= max_launches:
                raise RuntimeError(f"remote shard {tag} died {launches} times")
            launches += 1
            cmd = (f'"{REMOTE_PY}" -u -X utf8 "{script_r}" shard '
                   f'"{remote_path(sp)}"')
            out = ps(f"$r = Invoke-CimMethod -ComputerName {REMOTE} -ClassName Win32_Process "
                     f"-MethodName Create -Arguments @{{ CommandLine = {_q(cmd)}; "
                     f"CurrentDirectory = {_q(REMOTE_ROOT)} }}; "
                     f"\"$($r.ReturnValue) $($r.ProcessId)\"").strip()
            log(f"remote {tag}: launch {launches} on {REMOTE}: {out}")
            time.sleep(60.0)
        else:
            time.sleep(POLL_S)
        done, alive = status_tolerant()
    log(f"remote {tag}: done, rc {done['rc']} ({done['finished']})")
    for d, overwrite in pull:
        n = pull_dir(remote_path(d), d, overwrite=overwrite)
        log(f"remote {tag}: pulled {n} file(s) into {rel(d)}")
    # The shard logs are a record, not a result: a failed copy must not end the
    # run (2026-11-01 19:12, while another shard was still writing its log there;
    # 2026-11-02 08:57, WinError 32 when three shards that finished together copied
    # the same files at once).  One such pull at a time.
    try:
        with _PULL_LOCK:
            pull_dir(remote_path(rdir), rdir, overwrite=True)
    except (RuntimeError, OSError) as exc:
        log(f"remote {tag}: shard logs not pulled ({exc})")
    return int(done["rc"])


def split(items: List[Any], w_local: int, w_remote: int) -> Tuple[List[Any], List[Any]]:
    """Interleave so each host's share of units follows its share of workers."""
    loc: List[Any] = []
    rem: List[Any] = []
    for it in items:
        if (len(loc) + 1) / w_local <= (len(rem) + 1) / w_remote:
            loc.append(it)
        else:
            rem.append(it)
    return loc, rem


def kept_split(cands: List[Dict[str, float]], cj_loc: Path, cj_rem: Path
               ) -> Optional[Tuple[List[Any], List[Any]]]:
    """The split an earlier invocation wrote, if it covers exactly ``cands``.

    A resume with other worker counts must not move a candidate between the hosts
    while a shard may still be working on it: the remote shard keeps its own list,
    and the units it has finished are only here after its pull (2026-11-02).
    """
    from tuning.parameters import weight_fingerprint
    if not (cj_loc.exists() and cj_rem.exists()):
        return None
    loc = json.loads(cj_loc.read_text("utf-8"))
    rem = json.loads(cj_rem.read_text("utf-8"))
    if sorted(map(weight_fingerprint, loc + rem)) != sorted(map(weight_fingerprint, cands)):
        return None
    return loc, rem


# -- configuration and operating points ------------------------------------------------

def base_config():
    import dataclasses

    from experiments.ch_09_parameter_selection._ch9_selected_design import (
        FROZEN_DSO_DER_SCALE, FROZEN_DSO_LOAD_P_SCALE)
    from experiments.run_multi_system_ofo import make_config
    return dataclasses.replace(make_config(),
                               dso_der_scale=dict(FROZEN_DSO_DER_SCALE),
                               dso_load_p_scale=dict(FROZEN_DSO_LOAD_P_SCALE))


def check_config() -> Dict[str, Any]:
    cfg = base_config()
    got = {"dso_band": (cfg.dso_v_min_pu, cfg.dso_v_max_pu),
           "shunt_dispatch": cfg.shunt_dispatch,
           "unit_commitment": bool(cfg.use_zonal_gen_dispatch and cfg.unit_commitment),
           "slack_q_limit_mvar": cfg.slack_q_limit_mvar,
           "gen_vm_max_pu": cfg.gen_vm_max_pu,
           "tso_int_cooldown": cfg.tso_int_cooldown}
    if got["dso_band"] != BAND:
        raise RuntimeError(f"DSO band {got['dso_band']}, the procedure requires {BAND}")
    if got["shunt_dispatch"] != "miqp" or not got["unit_commitment"]:
        raise RuntimeError(f"configuration is not the commitment base with banks: {got}")
    log(f"configuration: {got}")
    return got


def ops_dir() -> Path:
    return RUN / "operating_points"


def load_ops():
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    from tuning.parameters import apply_to_config
    from tuning.scenarios_mc_v2 import CONDITIONS
    start = {k: float(v) for k, v in load_paramset(START)["fields"].items()}
    return s1.load_or_build_operating_points(
        apply_to_config(base_config(), start), [(r[0], r[1]) for r in CONDITIONS],
        ops_dir(), verbose=False, horizon_s=3600.0 * HOURS)


# -- 1. Stage I --------------------------------------------------------------------------

def stage1() -> None:
    from configs.paramsets import PARAMSET_DIR
    path = PARAMSET_DIR / f"{START}.json"
    if path.exists():
        doc = json.loads(path.read_text("utf-8"))
        if doc.get("provenance", {}).get("operating_point_horizon_s") != 3600.0 * HOURS:
            raise RuntimeError(f"{path} exists but was not designed on the 12 h capture")
        log(f"stage I: {START} exists (resume); operating points reused")
    else:
        rc = run(["-m", "tuning.stage1_screen", "--design", "--frozen-network",
                  "--d-engage-ts", str(D_ENGAGE["ts"]),
                  "--d-engage-sts", str(D_ENGAGE["sts"]),
                  "--d-engage-shunt", str(D_ENGAGE["shunt"]),
                  "--op-horizon-h", f"{HOURS:g}", "--cache-dir", rel(ops_dir()),
                  "--out", rel(RUN / "stage1" / "stage1_design.json"),
                  "--write-paramset", START], RUN / "stage1" / "stage1.log")
        if rc != 0 or not path.exists():
            raise RuntimeError(f"stage I exit {rc}; see stage1/stage1.log")
    fields = json.loads(path.read_text("utf-8"))["fields"]
    log(f"stage I: {', '.join(f'{k} {v:.6g}' for k, v in fields.items())}")


def push_inputs() -> None:
    """The Stage-I parameter set and the captured operating points, to e5power."""
    from configs.paramsets import PARAMSET_DIR
    push(PARAMSET_DIR / f"{START}.json",
         REMOTE_ROOT + rf"\configs\paramsets\{START}.json")
    for f in sorted(ops_dir().glob("op_*.npz")):
        push(f, remote_path(f))
    log(f"pushed {START} and {len(list(ops_dir().glob('op_*.npz')))} operating points "
        f"to {REMOTE}")


# -- 2./3. Stage II units on both hosts --------------------------------------------------

def screen_argv(out: Path, cj: Path, workers: int, windows: str = WINDOWS,
                skip_baseline: bool = False) -> List[str]:
    a = ["experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py",
         "--paramset", START, "--frozen-network", "--hours", f"{SIM_HOURS:g}",
         "--evented", "--v-metric", "bus", "--windows", windows,
         "--workers", str(workers), "--ops-dir", rel(ops_dir()),
         "--op-horizon-h", f"{HOURS:g}", "--candidates-json", rel(cj),
         "--out-dir", rel(out)]
    return a + (["--skip-baseline"] if skip_baseline else [])


def simulate(out: Path, cands: List[Dict[str, float]], w_local: int,
             w_remote: int, split_local: Optional[float] = None) -> None:
    """Simulate the Stage-I point and ``cands`` in W1-W4, W6, split over the two
    hosts, then assemble every candidate record here.

    ``split_local`` is the local host's weight in the split, in remote workers
    (default ``w_local``); a host whose workers are slower gets a smaller share."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "windows").mkdir(exist_ok=True)
    cj_all, cj_loc, cj_rem = (out / "all_candidates.json", out / "shard_local.json",
                              out / "shard_remote.json")
    kept = kept_split(cands, cj_loc, cj_rem) if w_remote > 0 else None
    if kept is not None:
        loc, rem = kept
        log(f"{rel(out)}: split of the earlier invocation kept")
    else:
        w_split = float(split_local) if split_local is not None else float(w_local)
        loc, rem = split(cands, w_split, w_remote) if w_remote > 0 else (cands, [])
    cj_all.write_text(json.dumps(cands, indent=1), "utf-8")
    cj_loc.write_text(json.dumps(loc, indent=1), "utf-8")
    cj_rem.write_text(json.dumps(rem, indent=1), "utf-8")
    log(f"{rel(out)}: {len(cands)} candidates + Stage-I point; "
        f"{len(loc)} here ({w_local} workers), {len(rem)} on {REMOTE} ({w_remote})")

    def local() -> int:
        for attempt in range(3):           # a rerun resumes from the unit files
            if run(screen_argv(out, cj_loc, w_local), out / "shard_local.log") == 0:
                return 0
            log(f"{rel(out)}: local shard incomplete (attempt {attempt + 1})")
        return 2

    def remote() -> int:
        if not rem:
            return 0
        push(cj_rem, remote_path(cj_rem))
        rc = 2
        for attempt in range(3):
            rc = run_remote(f"{out.name}_{attempt}", screen_argv(
                out, cj_rem, w_remote, skip_baseline=True), out,
                [(out / "windows", False)])
            if rc == 0:
                return 0
            log(f"{rel(out)}: remote shard incomplete (attempt {attempt + 1})")
        return rc

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_loc, f_rem = ex.submit(local), ex.submit(remote)
        rcs = (f_loc.result(), f_rem.result())
    if any(rcs):
        raise RuntimeError(f"{rel(out)}: shards ended with {rcs}")
    # Assembly: every unit is on disk now, so this runs nothing (a unit lost in
    # transit would be simulated here, which only costs time).
    if run(screen_argv(out, cj_all, max(w_local, 1)), out / "assemble.log") != 0:
        raise RuntimeError(f"{rel(out)}: assembly did not complete")


# -- selection (frozen rule + engagement bound) -------------------------------------------

def select_union(dirs: List[Path], tag: str, ops) -> dict:
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import theta_sts_min
    sel = RUN / "selection"
    union = sel / f"union_{tag}"
    union.mkdir(parents=True, exist_ok=True)
    thetas: Dict[str, float] = {}
    base_key = None
    for d in dirs:
        for f in sorted((d / "candidates").glob("cand_*.json")):
            c = json.loads(f.read_text("utf-8"))
            th = theta_sts_min(c["weights"], ops)
            thetas[c["key"]] = th
            if c.get("is_baseline"):
                base_key = c["key"]
            if th <= THETA_MAX:
                shutil.copy2(f, union / f.name)
    if base_key is None or thetas[base_key] > THETA_MAX:
        raise RuntimeError(f"Stage-I point missing or outside the engagement bound "
                           f"({base_key}: {thetas.get(base_key)})")
    (sel / f"theta_{tag}.json").write_text(json.dumps(
        {"theta_max_pu": THETA_MAX, "theta_sts_min_pu": thetas,
         "removed": sorted(k for k, v in thetas.items() if v > THETA_MAX)},
        indent=1), "utf-8")
    out = sel / f"selection_{tag}.json"
    run(["experiments/ch_09_parameter_selection/ch_9_3_select.py",
         "--candidates", rel(union), "--hours", f"{HOURS:g}",
         "--shunt-ops-per-day", f"{BANK_BUDGET:g}", "--out", rel(out)],
        sel / f"select_{tag}.log")
    if not out.exists():
        raise RuntimeError(f"selection {tag}: no report (no feasible candidate?)")
    rep = json.loads(out.read_text("utf-8"))
    n_adm = sum(1 for v in thetas.values() if v <= THETA_MAX)
    log(f"selection {tag}: {n_adm} / {len(thetas)} inside the engagement bound "
        f"{THETA_MAX:g} pu, "
        f"{rep['n_feasible']} / {rep['n_total']} feasible, ranked first "
        f"{[r['key'] for r in rep['ranked'][:4]]}")
    rep["_theta"] = thetas
    return rep


# -- the stages ------------------------------------------------------------------------

def stage2a(ops, w_local: int, w_remote: int) -> dict:
    from configs.paramsets import load_paramset
    from tuning import stage1_screen as s1
    from tuning.stage2_search import DEFAULT_SPANS, Box, sobol_candidates
    out = RUN / "stage2a"
    centre = {k: float(v) for k, v in load_paramset(START)["fields"].items()}
    draws = sobol_candidates(Box(centre=centre, spans=dict(DEFAULT_SPANS)), N_2A,
                             seed=SEED)
    lam = [s1.worst(s1.screen_weights(ops, w)).lambda_max for w in draws]
    keep = [w for w, l in zip(draws, lam) if l <= s1.select_ceiling()]
    out.mkdir(parents=True, exist_ok=True)
    (out / "draws.json").write_text(json.dumps(
        {"seed": SEED, "n_draws": N_2A, "centre": centre, "spans": dict(DEFAULT_SPANS),
         "select_ceiling": s1.select_ceiling(), "lambda_max_all": lam,
         "n_survivors": len(keep)}, indent=1), "utf-8")
    log(f"stage II-a: {len(keep)} / {N_2A} draws pass the screen at "
        f"{s1.select_ceiling():.4f}")
    simulate(out, keep, w_local, w_remote)
    return select_union([out], "2a", ops)


def stage2b(rep: dict, ops, w_local: int, w_remote: int) -> Tuple[dict, List[Path]]:
    from tuning import stage1_screen as s1
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    from tuning.stage2_search import DEFAULT_SPANS, Box, sobol_candidates
    s2a = RUN / "stage2a"
    dirs = [s2a]
    inc = rep["ranked"][0]
    evaluated = {p.stem[5:] for p in (s2a / "candidates").glob("cand_*.json")}
    base = next(json.loads(p.read_text("utf-8"))
                for p in (s2a / "candidates").glob("cand_*.json")
                if json.loads(p.read_text("utf-8")).get("is_baseline"))
    f, r = F0, 0
    while r < MAX_ROUNDS:
        r += 1
        rd = RUN / "stage2b" / f"round{r}"
        centre = {k: float(inc["weights"][k]) for k in WEIGHT_FIELDS}
        spans = {k: f * v for k, v in DEFAULT_SPANS.items()}
        draws = sobol_candidates(Box(centre=centre, spans=spans), N_LOCAL, seed=SEED + r)
        keep = [w for w in draws
                if s1.worst(s1.screen_weights(ops, w)).lambda_max <= s1.select_ceiling()
                and weight_fingerprint(w) not in evaluated]
        (rd / "windows").mkdir(parents=True, exist_ok=True)
        (rd / "draws.json").write_text(json.dumps(
            {"round": r, "seed": SEED + r, "f": f, "centre": centre, "spans": spans,
             "n_draws": N_LOCAL, "n_survivors": len(keep)}, indent=1), "utf-8")
        log(f"round {r}: f {f:g}, centre {weight_fingerprint(centre)}, "
            f"{len(keep)} / {N_LOCAL} draws pass the screen")
        # The Stage-I point's windows are the stored ones (same key, same flags).
        for f_ in (s2a / "windows").glob(f"{base['key']}__*.json"):
            shutil.copy2(f_, rd / "windows" / f_.name)
        simulate(rd, keep, w_local, w_remote)
        dirs.append(rd)
        evaluated |= {p.stem[5:] for p in (rd / "candidates").glob("cand_*.json")}
        rep = select_union(dirs, f"2b_round{r}", ops)
        new = rep["ranked"][0]
        if new["key"] != inc["key"]:
            log(f"round {r}: new best {new['key']} (e_v_ts "
                f"{new['aggregate']['e_v_ts']:.5f}, e_v_ds "
                f"{new['aggregate']['e_v_ds']:.5f}); recentre, f stays {f:g}")
            inc = new
        else:
            f /= 2
            log(f"round {r}: best unchanged ({inc['key']}); f -> {f:g}")
            if f < F_MIN:
                break
    log(f"stage II-b ends after {r} round(s); best {inc['key']}")
    return rep, dirs


def exp_c_argv(label: str, wf: Path, workers: int) -> List[str]:
    return ["experiments/ch_09_parameter_selection/ch_9_1_ninner_isolated_sts.py",
            "--paramset", START, "--weights-json", rel(wf), "--label", label,
            "--out", rel(RUN / "expC"), "--workers", str(workers),
            "--max-step-mvar", "10", "--save-trajectories", *EXPC_EXTRA]


def settle(rep: dict, c_workers: int) -> Tuple[Optional[dict], dict]:
    """Run 0001's settle(), with the batch split over the two hosts."""
    from experiments.ch_09_parameter_selection.ch_9_3_amendment1 import _criterion
    from tuning.parameters import WEIGHT_FIELDS
    from tuning.stage2_search import rank_lexicographic
    expc = RUN / "expC"
    expc.mkdir(parents=True, exist_ok=True)
    feasible = {r["key"]: r for r in rep.get("ranked", [])}
    path = expc / "criterion_results.json"
    results = json.loads(path.read_text("utf-8")) if path.exists() else {}
    while True:
        remaining = [r for k, r in feasible.items()
                     if not (k in results and not results[k]["pass"])]
        if not remaining:
            return None, results
        order = rank_lexicographic(remaining, indifference=0.02)
        if order[0]["key"] in results:
            return order[0], results
        batch = [r for r in order if r["key"] not in results][:C_PARALLEL]
        log(f"settling check on {[r['key'] for r in batch]}")

        def one(i_r: Tuple[int, dict]) -> dict:
            i, r = i_r
            label = f"cand_{r['key']}"
            wf = expc / f"{label}_weights.json"
            wf.write_text(json.dumps({k: float(r["weights"][k]) for k in WEIGHT_FIELDS},
                                     indent=1), "utf-8")
            argv = exp_c_argv(label, wf, c_workers)
            if i % 2 == 0:
                run(argv, expc / f"{label}.log")
            else:
                push(wf, remote_path(wf))
                run_remote(label, argv, expc, [(expc / label, True)])
            return _criterion(expc, label)

        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            outs = list(ex.map(one, enumerate(batch)))
        for r, o in zip(batch, outs):
            results[r["key"]] = o
            log(f"  {r['key']}: {'PASS' if o['pass'] else 'FAIL'} p95 "
                f"{o.get('p95')} median residual {o.get('median_residual_mvar')} "
                f"{o.get('error', '')}")
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")


def report_w5(win: dict) -> dict:
    out = RUN / "report_w5"
    out.mkdir(parents=True, exist_ok=True)
    cj = out / "result.json"
    cj.write_text(json.dumps([win["weights"]], indent=1), encoding="utf-8")
    run(screen_argv(out, cj, 2, windows="W5"), out / "report_w5.log")
    f = out / "windows" / f"{win['key']}__W5.json"
    return json.loads(f.read_text("utf-8"))["row"] if f.exists() else {}


def write_result(win: dict, rep: dict, c_res: dict, rounds: int, w5: dict) -> Path:
    from configs.paramsets import PARAMSET_DIR
    from tuning.parameters import WEIGHT_FIELDS, weight_fingerprint
    path = PARAMSET_DIR / f"{RESULT}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a parameter set is never overwritten")
    fields = {k: float(win["weights"][k]) for k in WEIGHT_FIELDS}
    doc = {
        "name": RESULT,
        "title": "Controller weights selected by the Ch 8 procedure (Experiment B) "
                 "on the unit-commitment base of 2026-10-31",
        "description": [
            "Selected by docs/tuning/2026-10-28_weight_selection_procedure.md "
            "(frozen), applied on the unit-commitment base (merit order, "
            "technical minima, equivalent Q limited to +-1200 Mvar):",
            "Stage I analytic point designed on operating points with the "
            "commitment of the 12 h window; Stage II-a 512 Sobol draws; Stage "
            "II-b local rounds; filter with the engagement bound "
            "theta_STS,min <= 0.02 pu (amendment 2); ranking on the per-bus TS voltage "
            "deviation with a 2 % band on the STS deviation; settling "
            "requirement in rank order.",
            "Declared in docs/analysis/2026-10-31_ch9_full_procedure_uc_declaration.md.",
        ],
        "provenance": {
            "producer": rel(Path(__file__)),
            "procedure": "docs/tuning/2026-10-28_weight_selection_procedure.md",
            "declaration": "docs/analysis/2026-10-31_ch9_full_procedure_uc_declaration.md",
            "run_dir": rel(RUN),
            "starting_paramset": START,
            "dso_voltage_band_pu": list(BAND),
            "simulated_windows": WINDOWS,
            "w5_report": w5,
            "selected_key": win["key"],
            "stage2b_rounds": rounds,
            "n_candidates": rep.get("n_total"),
            "n_feasible": rep.get("n_feasible"),
            "theta_sts_min_pu": rep["_theta"].get(win["key"]),
            "theta_sts_max_pu": THETA_MAX,
            "amendment2": "docs/tuning/2026-10-31_amendment2_sts_engagement_0p02.md",
            "failed_settling": sorted(k for k, v in c_res.items() if not v["pass"]),
            "lambda_max_B": win.get("lambda_max_B"),
            "aggregate_ranked_windows_per_bus": win["aggregate"],
            "baseline_aggregate": rep.get("baseline", {}).get("aggregate"),
            "settling": c_res.get(win["key"]),
            "weight_fingerprint": weight_fingerprint(fields),
        },
        "fields": fields,
    }
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    log(f"wrote {path}")
    return path


# -- main ------------------------------------------------------------------------------

def procedure(args) -> int:
    check_config()
    if args.w_remote > 0:
        lh, rh = local_head(), remote_head()
        if lh != rh:
            raise RuntimeError(f"code differs: here {lh[:7]}, {REMOTE} {rh[:7]}")
        log(f"code {lh[:7]} on both hosts")
    log("== 1. Stage I")
    stage1()
    ops = load_ops()
    if args.w_remote > 0:
        push_inputs()
    if args.stop_after == "stage1":
        return 0
    log("== 2. Stage II-a")
    rep = stage2a(ops, args.w_local_2a, args.w_remote)
    log("== 3. Stage II-b")
    rep, dirs = stage2b(rep, ops, args.w_local, args.w_remote)
    log("== 4. selection over all evaluated candidates, settling check")
    win, c_res = settle(rep, args.c_workers)
    if win is None:
        log("no feasible candidate passes the settling requirement")
        (RUN / "FAILED.txt").write_text("no candidate passes\n", "utf-8")
        return 2
    log(f"SELECTED {win['key']}")
    log("== 5. W5 for the report")
    w5 = report_w5(win)
    log(f"  W5: e_v_ts {w5.get('e_v_ts')}, e_v_ds {w5.get('e_v_ds')}")
    if SMOKE:
        (RUN / "SMOKE_DONE.txt").write_text(f"selected {win['key']}\n", "utf-8")
        return 0
    write_result(win, rep, c_res, len(dirs) - 1, w5)
    (RUN / "DONE.txt").write_text(f"selected {win['key']}\n", "utf-8")
    return 0


def smoke(args) -> int:
    """The plumbing on both hosts at toy size: 16 draws, one round of 8, two
    windows of 6 min, settling on one DSO of W1, no parameter set written.
    Stage I is not repeated: the capture of ``--ops-from`` is copied."""
    global RUN, N_2A, N_LOCAL, MAX_ROUNDS, WINDOWS, SIM_HOURS, EXPC_EXTRA, C_PARALLEL, SMOKE
    global POLL_S
    N_2A, N_LOCAL, MAX_ROUNDS, WINDOWS, SIM_HOURS = 16, 8, 1, "W1,W2", 0.1
    EXPC_EXTRA, C_PARALLEL, SMOKE = ["--windows", "W1", "--dsos", "DSO_1"], 2, True
    POLL_S = 30.0
    RUN = ROOT / "results" / "_smoke_ch9_6" / f"{datetime.now():%Y%m%d-%H%M%S}"
    RUN.mkdir(parents=True)
    (RUN / "operating_points").mkdir()
    for f in Path(args.ops_from).glob("op_*.npz"):
        shutil.copy2(f, RUN / "operating_points" / f.name)
    log(f"SMOKE in {RUN}")
    try:
        return procedure(args)
    except Exception as exc:                                      # noqa: BLE001
        import traceback
        log(f"smoke error: {type(exc).__name__}: {exc}")
        (RUN / "FAILED.txt").write_text(traceback.format_exc(), "utf-8")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["shard"]:
        return shard_main(Path(argv[1]))
    p = argparse.ArgumentParser(prog="ch_9_6_full_procedure")
    p.add_argument("--w-local-2a", type=int, default=14,
                   help="workers here in Stage II-a (REF runs share this host)")
    p.add_argument("--w-local", type=int, default=20, help="workers here from II-b on")
    p.add_argument("--w-remote", type=int, default=20, help=f"workers on {REMOTE}")
    p.add_argument("--c-workers", type=int, default=10,
                   help="workers per settling check (two per host at a time)")
    p.add_argument("--stop-after", choices=("stage1",), default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--note", default="")
    p.add_argument("--smoke", action="store_true", help="toy-size plumbing test")
    p.add_argument("--ops-from", type=Path, default=None,
                   help="with --smoke: the operating_points folder of a run")
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    if args.smoke:
        return smoke(args)

    global RUN
    from experiments.thesis_runs import start_thesis_run
    with start_thesis_run(
            __file__, target=["ch:timescales:weights"],
            feeds=["tab:param:weights:resolved", "fig:param:timescales:settling_cdf"],
            parameters={"procedure": "docs/tuning/2026-10-28_weight_selection_procedure.md",
                        "declaration": "docs/analysis/2026-10-31_ch9_full_procedure_uc_declaration.md",
                        "stage1_paramset": START, "result_paramset": RESULT,
                        "sobol_seed": SEED, "stage2a_draws": N_2A,
                        "stage2b_rounds_max": MAX_ROUNDS, "op_horizon_h": HOURS,
                        "theta_sts_max_pu": THETA_MAX,
                        "hosts": f"e5wimi {args.w_local_2a}/{args.w_local}, "
                                 f"{REMOTE} {args.w_remote} workers"},
            time_series=("W1-W4, W6 of tuning.scenarios_mc_v2.CONDITIONS (2016 "
                         "profiles), 12 h each, one disturbance each, evented, frozen "
                         "network; T_TS 180 s, T_STS 20 s, dt 20 s; W5 report only. "
                         "Experiment C: 120 isolated interface steps in the same windows"),
            note=args.note or "Complete Ch 9 weight selection on the unit-commitment base",
            resume=args.resume) as trun:
        RUN = trun.root
        log(f"run folder {RUN}")
        t0 = time.time()
        try:
            rc = procedure(args)
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            log(f"driver error: {type(exc).__name__}: {exc}")
            (RUN / "FAILED.txt").write_text(traceback.format_exc(), "utf-8")
            rc = 1
        log(f"driver finished rc={rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


if __name__ == "__main__":
    sys.exit(main())
