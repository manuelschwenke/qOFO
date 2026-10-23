# Part III Monte-Carlo ensemble: running it on the server

Driver: `experiments/ch_11_case2_multisystem/mc/run_mc.py` (2026-10-07; moved
from `experiments/part3_mc/` on 2026-10-23). It runs the first slice:
L2, S1 and O2 on N = 100 paired episodes. These are the "no STS access" row of
Case 2 and the statistics of Case 1 (`tab:case_svr:stats`). S2, O3 and M are
added later on the same seeds.

## Declared before any run (do not change once results exist)

| Constant | Value |
|---|---|
| master seed | 20261007 (episode k = master + k) |
| valid episodes N | 100 (the first 100 seeds that pass the prescreen) |
| horizon | 180 min (540 steps of 20 s, 60 TS dispatches) |
| start time | uniform over 2016 on a 15-min grid; the days of W1–W6 are excluded (W1's day = reference period) |
| events | slots every 30 min; each fires with p = 0.25; kind ∈ {machine out, line out, reactive load 100–400 Mvar}; every event is reversed after 90 min |
| discard rule | a draw is discarded ONLY if L2 without events fails (prescreen); a failing variant is an outcome |
| δ (non-inferiority) | 1 mpu per area, time-RMS pilot error, O2 − S1 |
| normal operating range | [0.95, 1.05] pu (time outside), hard band [0.90, 1.10] pu |
| configuration | `run_case1.resolved()`: make_config + DSO_3 ×2 + paramset `ch9_selected` |
| S1 | integral RVR + RPR, safeguard on, T_RVR = 300 s (author 2026-10-07; NOT the earlier PI form). Check: `resolved("S1").svr_k_p_rvr == 0.0` |

## 1. Prerequisites on the server

* The conda env `qOFO_clean` (Python 3.12; on the laptop: pandapower 3.4.0, numpy 2.3.5, scipy 1.16.3). Install the same versions, otherwise the runs are not comparable with K1.
* A **Gurobi licence on that machine**. The driver checks this and stops if Gurobi is missing; without it O2's MIQP would fall back to SCIP and would not be the controller of the reference runs. Academic licences are tied to one machine, so get a licence for the server (`grbgetkey`).
* Enough RAM: plan for about 1 GB per worker.

## 2. Copy the code to a LOCAL disk on the server

Runs on a network drive are about 3× slower. Copy the code from the share once, for example:

```bat
robocopy \\e5server\homefolders$\mschwenke\Python_Projekte\qOFO D:\qOFO /E /XD results _local_archive .git\objects __pycache__ /XF *.pyc
```

Keep `.git` (without `objects` if space is short) or note the commit: every call records the commit, the dirty flag and `git diff HEAD` in the campaign's `run_manifest.json` / `README.md` and `code_state*.diff` (`experiments/thesis_runs.py`). Without git these entries say "unavailable"; the run is still valid, but then copy the code state (zip) next to the results.

## 3. Smoke test (about 3 min)

```bat
cd /d D:\qOFO
conda activate qOFO_clean
python -m experiments.ch_11_case2_multisystem.mc.run_mc --smoke --jobs 4
```

It writes `results\ch_11_case2_multisystem\mc\<NNNN>_<stamp>_run_mc_smoke\report.md`: 2 episodes × 10 min, three variants, all outcomes `ok`.

## 4. The campaign

```bat
python -m experiments.ch_11_case2_multisystem.mc.run_mc --jobs 30 --note "first slice, server" > results\run_mc_console.log 2>&1
```

* Use `--jobs` = physical cores − 2 (each worker is single-threaded; BLAS threads are pinned to 1).
* **Resume** = the same command: it continues the newest untagged campaign folder. Prescreens and runs that already have a JSON are skipped. A `task_crash` (a Python error in the driver, not in the simulation) is rerun. Each call is listed under *Invocations* in the campaign's README. `--new` starts a fresh campaign folder instead.
* Tables only: `python -m experiments.ch_11_case2_multisystem.mc.run_mc --report`.
* The S1 runs are submitted first, because they are the longest.

**Expected cost.** These are laptop single-core figures; the server should be similar per core.

| Variant | per 180-min episode |
|---|---|
| S1 | ~75 min (incl. a ~2.5 min calibration) |
| O2 | ~10–20 min |
| L2 | ~6 min |
| prescreen | < 1 min |

That is about 160 CPU-hours in total: roughly 5–6 h on 30 workers, 10–12 h on 16.

## 5. Output (`results\ch_11_case2_multisystem\mc\<NNNN>_<stamp>_run_mc\`)

| Path | Content |
|---|---|
| `README.md`, `run_manifest.json`, `latex_header.tex` | the run record and the thesis provenance block (`experiments/thesis_runs.py`) |
| `elements.json` | the drawable machines, lines and load buses |
| `prescreen\<seed>.json` | start time, events, class, valid/invalid |
| `runs\<seed>_<variant>.json` | outcome, diagnostics, wall time, per-area metrics |
| `runs\<seed>_<variant>.npz` | per-step series (area error, pilot voltage, area v_min/v_max) |
| `logs\*.log` | runner output per task (only for debugging; may be deleted) |
| `metrics_per_episode.csv`, `outcomes_per_episode.csv`, `report.md` | the tables |
| `code_state*.diff` | `git diff HEAD` of each launch whose tree was dirty |

**Outcomes.** `ok`, `fail_unreachable` (reachability guard), `fail_abort`, `fail_pf` (a step without a converged power flow), `fail_v_end` (outside 0.90/1.10 pu at the end) and `fail_osc` (a tap reverses direction ≥ 2× in the last 30 min). An episode that fails under every variant is reported as infeasible.

## 6. What to bring back

The whole campaign run folder, without `logs\` and `_scratch\` if space is short. Copy it under the same relative path to `\\e5server\...\qOFO\results\ch_11_case2_multisystem\mc\`. Its README keeps the server's save location and host, so the copy stays traceable.

The 2026-10-07 slice was run from `qOFO_2027` and lies in `qOFO_2027\results\THESIS_part3_mc` (old layout, no README); see `results\ch_11_case2_multisystem\README.md`.

If the code changes between this slice and the later S2/O3/M slice, rerun three S1/O2 seeds after the change and check that their JSONs are identical. Otherwise the later slice is not paired with this one.
