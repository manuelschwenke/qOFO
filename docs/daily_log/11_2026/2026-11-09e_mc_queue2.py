"""Night queue v2 for the Ch 10 MC on v5 / rural_700_north_res: the full declared N = 100
tonight (author 2026-11-09 ~21:40: "If it is faster then expected, we can run 50 additional MC
runs overnight in queue, mainly on e5power").  Replaces mc_queue.py (same log), whose e5wimi
share had not started.

The second 50 complete the sample size fixed in advance (N = 100); the decision rests on
throughput only, not on any result of the first 50.  Split by variant (the declared protocol
splits a campaign by variant over two hosts and merges with merge_mc.py):

* e5power (tag v5nr_e5power): after the running first call (S1, O2, --n 50) has ended and
  been pulled back, a second call on the same campaign with S1, L2, O2 and --n 100 --buffer 6
  (S1, O2 on the seeds 57-106; L2 on all 106, moved here from e5wimi).
* e5wimi (tag v5nr, new campaign): once REF 0004-0006 are complete, M, S2, O3 with
  --n 100 --n-o3 100 --buffer 6 (O3's further 200 episodes later); 30 CPUs at night, 10 from
  08:00, 20 from 18:00 (e5wimi rule), own process tree only.
Detached (pythonw); log mc_queue.log next to this file; pid in mc_queue2.pid.
"""
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
LOG = HERE / "mc_queue.log"
PY = r"F:\python_environments\qOFO_clean\python.exe"
ROOT = Path(r"F:\qOFO_ref")
REF = ROOT / "results" / "ch_11_case2_multisystem" / "ref"
MC = ROOT / "results" / "ch_11_case2_multisystem" / "mc"
REF_RUNS = ("0004_2026-11-09_185815_run_ref_v5_E2", "0005_2026-11-09_185843_run_ref_v5_P2R",
            "0006_2026-11-09_185912_run_ref_v5_P2R_B03B05")
REF_LAUNCHERS = ("refR2_E2", "refR3_P2R", "refR3_B")
FIRST_WRAPPER_PID = 103212          # mc_remote_wrapper.py 50 --variants S1,O2 (mc_queue.py)
N, BUFFER, JOBS = 100, 6, 30
NOTE = ("Ch 10 Monte-Carlo on rural_700_north_res with the v5 weights (ch9_selected_nr_2026_11_v5), "
        "declared protocol of 2026-11-07 (rule (b), void rule), G3-off episodes as drawn (author "
        "2026-11-09: '(a)'); the declared N = 100 in one night (author 2026-11-09 ~21:40: '50 "
        "additional MC runs overnight ... mainly on e5power'), decided on throughput only; "
        "variants re-split: e5wimi M, S2, O3; e5power S1, L2, O2")
TAG_LOCAL, TAG_REMOTE = "v5nr", "v5nr_e5power"
CPUS_NIGHT = list(range(0, 40, 2)) + list(range(1, 20, 2))     # 30 logical CPUs
CPUS_DAY = list(range(0, 20, 2))                               # 10, the e5wimi day rule
CPUS_EVENING = list(range(0, 40, 2))                           # 20 after 18:00
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
           FOR_DISABLE_CONSOLE_CTRL_HANDLER="1", MPLBACKEND="Agg")
_lock = threading.Lock()


def log(msg):
    with _lock, open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [v2] {msg}\n")


def alive(name):
    p = HERE / f"{name}.pid"
    try:
        return psutil.pid_exists(int(p.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return False


def first_wrapper_running():
    try:
        p = psutil.Process(FIRST_WRAPPER_PID)
        return "mc_remote_wrapper" in " ".join(p.cmdline())
    except psutil.Error:
        return False


def ref_done():
    n = sum(1 for r in REF_RUNS for _ in (REF / r).glob("*/status.json"))
    return n == 18 and not any(alive(x) for x in REF_LAUNCHERS), n


def pin_tree(pid, cpus):
    try:
        root = psutil.Process(pid)
        for p in [root] + root.children(recursive=True):
            try:
                p.cpu_affinity(cpus)
            except psutil.Error:
                pass
        return True
    except psutil.NoSuchProcess:
        return False


def remote_share():
    while first_wrapper_running():
        time.sleep(120)
    log("first e5power call (S1, O2, --n 50) has ended -> second call S1, L2, O2 --n 100")
    args = [PY, "-X", "utf8", "-u", str(HERE / "mc_remote_wrapper.py"), str(N),
            "--variants", "S1,L2,O2", "--jobs", str(JOBS), "--buffer", str(BUFFER),
            "--tag", TAG_REMOTE, "--note", NOTE + "; e5power share (S1, L2, O2)"]
    with open(HERE / "mc_remote_console.log", "a", encoding="utf-8") as fh:
        rc = subprocess.call(args, cwd=str(ROOT), env=ENV, stdin=subprocess.DEVNULL, stdout=fh,
                             stderr=subprocess.STDOUT, creationflags=0x08000000)
    log(f"e5power MC share (second call) ended, rc {rc} (pulled back by run_mc_remote)")


def local_share():
    while True:
        ok, n = ref_done()
        if ok:
            break
        time.sleep(120)
    log("REF runs 0004-0006 complete (18 status files) -> e5wimi MC share (M, S2, O3)")
    for attempt in range(12):
        started = any(MC.glob(f"*_run_mc_{TAG_LOCAL}"))
        args = [PY, "-X", "utf8", "-u", "-m", "experiments.ch_11_case2_multisystem.mc.run_mc",
                "--variants", "M,S2,O3", "--n", str(N), "--n-o3", str(N), "--buffer", str(BUFFER),
                "--jobs", str(JOBS), "--tag", TAG_LOCAL, "--note", NOTE + "; e5wimi share (M, S2, O3)"]
        if not started:
            args.insert(args.index("--variants"), "--new")
        t0 = time.time()
        with open(HERE / "mc_local_console.log", "a", encoding="utf-8") as fh:
            proc = subprocess.Popen(args, cwd=str(ROOT), env=ENV, stdin=subprocess.DEVNULL, stdout=fh,
                                    stderr=subprocess.STDOUT, creationflags=0x08000000)
            log(f"local run_mc pid {proc.pid} (attempt {attempt + 1}, new={not started})")
            mode = None
            while proc.poll() is None:
                h = time.localtime().tm_hour
                want = "day" if 8 <= h < 18 else ("evening" if h >= 18 else "night")
                if want != mode:
                    cpus = {"day": CPUS_DAY, "evening": CPUS_EVENING, "night": CPUS_NIGHT}[want]
                    pin_tree(proc.pid, cpus)
                    log(f"local MC tree pinned to {len(cpus)} CPUs ({want})")
                    mode = want
                elif int(time.time()) % 600 < 60:
                    pin_tree(proc.pid, {"day": CPUS_DAY, "evening": CPUS_EVENING,
                                        "night": CPUS_NIGHT}[mode])   # new workers
                time.sleep(60)
        rc = proc.returncode
        log(f"local run_mc exit {rc} after {time.time() - t0:.0f} s")
        if rc == 0 or time.time() - t0 > 120.0:
            break


(HERE / "mc_queue2.pid").write_text(str(os.getpid()), encoding="utf-8")
log("queue v2 armed: N = 100 tonight; e5power S1,L2,O2 after the first call; "
    "e5wimi M,S2,O3 after REF 0004-0006")
threads = [threading.Thread(target=remote_share), threading.Thread(target=local_share)]
for t in threads:
    t.start()
for t in threads:
    t.join()
log("queue v2 done")
