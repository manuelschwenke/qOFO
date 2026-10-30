"""Weekend chains of 2026-10-30 (harmonised device limits), one per host.

e5wimi:  Ch 9 re-validation (20 workers) -> DECISION -> push DECISION_OK/STOP to
         e5power -> STEP (14) and Case 1 K1-ALL (6) in parallel.
e5power: REF A, B, D (3 x 6 workers, speculatively on c08b1d3d33ab) -> wait for
         DECISION_OK -> MC (20 workers).

Usage:  python -X utf8 _weekend/chain.py e5wimi|e5power
"""
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
HOST = sys.argv[1]
LOGDIR = (Path(r"F:\qOFO_runs\launch_logs") if HOST == "e5power"
          else ROOT / "results" / "weekend_logs")      # results/ is git-ignored
LOGDIR.mkdir(parents=True, exist_ok=True)
GATE = Path(r"F:\qOFO_runs")       # e5power: decision files, outside the repo
CHAIN_LOG = LOGDIR / f"20261030_weekend_chain_{HOST}.log"
NOTE = ("Weekend 2026-10-30: harmonised limits (AVR [0.95, 1.05] pu, TS taps "
        "15 min, couplers 120 s, local TS tap relays 1.03 +/- 0.01 pu and local "
        "bank rule in LOC/SVR/SVR+), weights c08b1d3d33ab")
ENV = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT),
           # the run records read git; the worktree's git dir is on a share
           GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    with CHAIN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def start(tag, args):
    out = open(LOGDIR / f"20261030_weekend_{tag}.log", "w", encoding="utf-8")
    log(f"start {tag}: {' '.join(args)}")
    return subprocess.Popen([PY, "-u", "-X", "utf8", *args], cwd=ROOT, env=ENV,
                            stdout=out, stderr=subprocess.STDOUT)


def wait_all(procs):
    for tag, p in procs:
        rc = p.wait()
        log(f"end {tag}: exit {rc}")


def push_to_e5power(name, text):
    ps = (f"Invoke-Command -ComputerName e5power -ScriptBlock {{ "
          f"Set-Content -Path 'F:\\qOFO_runs\\weekend_{name}' -Value '{text}' }}")
    rc = subprocess.call([r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                          "-NoProfile", "-Command", ps])
    log(f"pushed {name} to e5power: exit {rc}")


def e5wimi():
    rc = start("revalidation", [
        "experiments/ch_09_parameter_selection/ch_9_5_avr_revalidation.py",
        "--workers", "20"]).wait()
    log(f"re-validation exit {rc}")
    if rc != 0:
        (LOGDIR / "DECISION_STOP.txt").write_text(
            f"re-validation exit {rc}; see its DECISION.json", "utf-8")
        push_to_e5power("DECISION_STOP.txt", f"re-validation exit {rc}")
        return
    push_to_e5power("DECISION_OK.txt", "c08b1d3d33ab unchanged")
    wait_all([
        ("step", start("step", ["-m", "experiments.ch_11_case2_multisystem.step.run_step",
                                "--jobs", "14", "--note", NOTE])),
        ("case1", start("case1", ["-m", "experiments.ch_10_case1_svr.run_case1",
                                  "--only", "K1-ALL", "--jobs", "6", "--note", NOTE])),
    ])


def e5power():
    wait_all([(f"ref{s}", start(f"ref{s}", [
        "-m", "experiments.ch_11_case2_multisystem.ref.run_ref", "--scenario", s,
        "--jobs", "6", "--note", NOTE + (f"; declared scenario {s}" if s != "A" else "")]))
        for s in ("A", "B", "D")])
    while True:
        if (GATE / "weekend_DECISION_STOP.txt").exists():
            log("DECISION_STOP: MC not started")
            return
        if (GATE / "weekend_DECISION_OK.txt").exists():
            break
        time.sleep(300)
    wait_all([("mc", start("mc", ["-m", "experiments.ch_11_case2_multisystem.mc.run_mc",
                                  "--new", "--jobs", "20", "--note", NOTE]))])


if __name__ == "__main__":
    log(f"chain {HOST} started in {ROOT}")
    {"e5wimi": e5wimi, "e5power": e5power}[HOST]()
    log(f"chain {HOST} done")
