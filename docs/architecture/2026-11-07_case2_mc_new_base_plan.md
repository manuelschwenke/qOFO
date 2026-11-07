# 2026-11-07 – Case-2 Monte-Carlo on the new base: plan

Status: **agreed with the author on 2026-11-07, ~16:15**:
- split 30 workers here + 30 on e5power;
- current OFO+;
- void off-line outages;
- variant split as below;
- exclusion days unchanged.

**Not launched.** The launch waits for the preconditions of §7 and the author's go.

Thesis section `ch:case3:montecarlo` (Chapter 10). The plan replaces the ensemble of record, qOFO mc/0003 of 2026-10-29: old base, paramset `ch9_selected_band_2026_10_a1`, no unit commitment.

Variants (code key in brackets, used only in commands): LOC (L2), SVR (S1), SVR+ (S2), OFO (O2), OFO+ (O3), CEN (M).

## 1. What stays as declared

These are the constants of 2026-10-07, with the changes of 2026-10-28 and 2026-11-01. They are in `run_mc.py` and the provenance block of `ch:case3:montecarlo`.
- **Seeds:** master seed 20261007; episode k has seed 20261007 + k.
- **Episodes:** 100 valid episodes of 180 min with all six variants; OFO+ on the first 300 valid.
- **Start times:** uniform over 2016 on a 15-min grid, without the W1–W6 days.
- **Events:**
  - slots every 30 min, each fires with p = 0.25;
  - kinds: machine outage, line outage, reactive load step of 100–400 Mvar;
  - every event is reverted after 90 min;
  - each event and its reversal fall at a phase drawn from {20, …, 160} s after a TS dispatch.
- **Noise:** class "minimum" for every controller and the droop, seed = episode seed, shared by all variants.
- **Discard rule:** a draw is discarded only if LOC fails its base case (prescreen: no events, 180-s step).
- **Comparison window, rule (b) of 2026-11-01:**
  - the first 100 valid episodes in draw order on which all six variants complete;
  - an episode on which any variant fails is replaced by the next one, and still counts for the failure bounds;
  - stop at the first 150 valid episodes.
- **Failures:** `fail_pf`, `fail_abort`, `fail_unreachable`, `fail_osc`. An end voltage outside [0.90, 1.10] pu is an observation, not a failure.
- **Decision family (unchanged):**
  - the pairs OFO−SVR, OFO+−SVR, OFO+−SVR+, OFO+−OFO and SVR+−SVR;
  - non-inferiority on the pilot error with δ = 1 mpu per area (one-sided 95 % bootstrap bound of the median);
  - superiority on e_v, worst bus and t_out (one-sided Wilcoxon signed-rank test, Holm over the family);
  - normal operating range [0.95, 1.05] pu.

## 2. The new base (checked on the resolved configurations, 2026-11-07)

| item | where | checked |
|---|---|---|
| paramset `ch9_selected_uc_2026_11_v4`, λ_max(B) = 1.4048 | `_ch9_selected_design.SELECTED_PARAMSET`, `variants.PARAMSET` | yes |
| unit commitment on, window = the 180-min episode | `unit_commitment=True`, `use_zonal_gen_dispatch=True`, `unit_commitment_window_s=None` for all six | yes |
| machine taps held in LOC, SVR, SVR+; dispatched in OFO, OFO+, CEN | `fix_machine_taps` True / False | yes |
| CEN = M at λ 1.405 (author 2026-11-07); no λ 0.90 variant in the campaign | `central_lambda_target = 1.4048` | yes |
| OLTC rate-limiter fix and SVR calibration fix | c643d83, an ancestor of HEAD | yes |
| relays-held re-solve: un-freezes the STS q(v) droops after a disturbance in LOC, SVR, OFO | 47e0668, an ancestor of HEAD; the limiter is off in OFO+ and CEN, and SVR+ has no relays | yes |
| noise on SVR and on the droop's voltage reading | `apply_to_svr`, `apply_to_der_voltage` | yes |

## 3. Protocol changes, declared before the run

**D1. Outages of off-line machines are void (author 2026-11-07).**
- The catalogue draws machine outages from the five non-slack machines, pandapower gens 0, 1, 2, 5, 7. On the commitment base, at least one of them is off-line in 41 % of the 3-h windows.
- Without a rule, a drawn trip of an off-line machine does nothing. Its restore 90 min later starts the machine, as an unplanned synchronous condenser.
- **Rule:** such an outage is void together with its restore. The draw is unchanged, so start times and all other events are those of mc/0003.
- The prescreen records:
  - the commitment (`uc_offline`);
  - the void events (`events_void`);
  - the episode class counted on the events that apply (`class`); `class_drawn` keeps the declared draw's class.
- The runner applies the same rule and reports `uc_offline_gens` and `uc_void_events` in its first record. `run_mc` checks per run that the runner's commitment equals the prescreen's (`diag.uc_mismatch_prescreen`).
- **Size:** in the first 300 draws, 109 episodes contain a machine trip (117 trips, 46 with the restore inside the horizon). At about 17 % off-line machine-windows, expect about 20 void trips (about 6 % of the episodes) and about 8 avoided machine starts. The exact count comes from the prescreen.

**D2. Excluded days stay W1–W6 (author 2026-11-07).**
- App. B separates design days from validation days. The two reference periods (5 Jan, already excluded as W1; 6 Nov 2016) and the step-test OPs (13 Apr, 21 Jun, 16 Dec) are validation conditions, not design conditions.
- This keeps the draw identical to mc/0003.

**D3. OFO+ as it is (author 2026-11-07).**
- The step tests' remedy rule fired: Δe_post of OFO+ was 15–35 mpu after scarce steps. The remedy (event-triggered TS dispatch) is not decided.
- The MC runs the current OFO+. If a remedy is adopted later, OFO and OFO+ are rerun (about 110 CPU-h). LOC, SVR, SVR+ and CEN stay valid.

**D4. Scheduling only. Neither the window nor any count changes.**
- **`--buffer 12`:** the variants of each host also run, from the start, on the next 12 valid episodes after the first 100.
  - Then the replacements that rule (b) needs run in the first wave, instead of a second wave of slow runs in the morning.
  - mc/0003 had 8 of its first 100 episodes not completed by all six.
  - Runs outside the window are listed, not counted (as `--also-seeds` before).
- **Split by variant over two hosts:**
  - each host runs its own share in its own campaign folder;
  - in the morning, `merge_mc.py` copies e5power's runs into the campaign of record here. A seed whose prescreen differs between the hosts is not copied; it is rerun here;
  - a final invocation with all six variants then completes the window if more than 12 episodes were replaced.

## 4. Code changes (qOFO_res, `res/commitment-base-2026-10-31`, to be committed before the launch)

| file | change |
|---|---|
| `experiments/runners/multi_tso_dso.py` | D1 guard after the commitment, in place on the event list; `uc_offline_gens` / `uc_void_events` on record 1 |
| `experiments/helpers/records.py` | the two fields (defaults keep older logs loadable) |
| `experiments/ch_11_case2_multisystem/mc/run_mc.py` | `void_events`, `applied_events`; prescreen fields; per-run commitment check; `--buffer`; parameters `buffer` and `uc_void_rule` in the manifest; docstring |
| `experiments/ch_11_case2_multisystem/mc/evaluate_mc.py` | failure timing against the applied events only |
| `experiments/ch_11_case2_multisystem/mc/run_mc_remote.py` (new) | one campaign share on e5power: hash-checked code sync, run number, detached shard, pull back |
| `experiments/ch_11_case2_multisystem/mc/merge_mc.py` (new) | the merge of D4, recorded in `merged_from_<src>.json` and the README |
| `tests/experiments/test_mc_uc_void.py`, `tests/experiments/test_mc_merge.py`, `tests/test_uc_void_events.py` (new) | void rule, buffer, merge. The runner test runs at 2016-12-16 18:45 (G3, G7, G9 off-line): the trip and restore of an off-line machine are voided and the machine stays off |

## 5. Cost and split

**Per-episode cost (180 min, one core, no SMT contention).** Basis:
- the old ensemble's per-run `wall_s` (e5power, old base): medians LOC 29, SVR 56, OFO 22, SVR+ 12, OFO+ 8.5, CEN 7.8 min;
- the new-base reference runs, 360 min, per variant `status.json` wall_s, after c643d83: E2 0025 on e5power and P2R 0028 on e5wimi;
- +8 % to +10 % for LOC, SVR and OFO from the relays-held fix (E2 test).

| variant | min per episode, mid (range) | episodes | CPU-h, mid (range) | host |
|---|---|---|---|---|
| SVR | 72 (60–85) | 112 | 134 (112–159) | e5power |
| CEN | 32 (22–43) | 112 | 60 (41–80) | e5power |
| OFO | 33 (30–40) | 112 | 62 (56–75) | e5power |
| prescreen for 112 valid (≈141 draws) | 5.5 (4.5–7) | 141 | 13 (10–16) | e5power |
| LOC | 45 (30–60) | 112 | 84 (56–112) | e5wimi |
| SVR+ | 18 (14–25) | 112 | 34 (26–47) | e5wimi |
| OFO+ | 9 (8–11) | 300 | 45 (40–55) | e5wimi |
| prescreen for 300 valid (≈376 draws) | 5.5 (4.5–7) | 376 | 34 (28–44) | e5wimi |
| **total** | | | **≈ 466 (≈ 360–590)** | |

**Hosts.**
- **e5wimi** (here): EPYC 9475F, 20 cores / 40 threads, 256 GB.
  - At 15:45 another user's job ran 8 Python workers at one core each (since 11:05), so about 12 cores (31 threads) were free.
  - 30 workers plus those 8 make 38 threads. Each thread then runs at about 0.65 of a core, about 19.5 core-equivalents in all.
  - **SVR, the longest variant, therefore goes to e5power,** where there is no hyperthreading.
- **e5power:** 2× Xeon Gold 6154, 36 cores, no SMT, 191 GB.
  - At 16:05, another user's 30 processes (started 05:19–05:42) used about 10 cores, and the E2 / Case 1 reruns used 12 (until about 19:00).
  - Afterwards, 30 workers give about 27 core-equivalents. If the other user's job rises to full load, about 18.
- Per-thread speed: the reference runs suggest e5power ≈ e5wimi.

**Split** (author: e5power SVR, CEN, OFO; e5wimi LOC, SVR+, OFO+ and the full prescreen):

| host | CPU-h mid (range) | capacity | wall time | ETA from 19:00 |
|---|---|---|---|---|
| e5power | 269 (219–330) | ≈ 27 | ≈ 10 h (8–12) | ≈ 05:00 (03:00–07:15) |
| e5wimi | 197 (150–258) | ≈ 19.5 (22.6 without the other job) | ≈ 10 h (7.7–13) | ≈ 05:00 (02:45–08:15) |

**Ordering.** The submission order is the `--variants` order, longest first. Within a variant it is draw order, so the paired episodes come before OFO+'s further ones. Each host ends on its shortest runs (OFO resp. OFO+), so its tail is short.
- On e5wimi the prescreen of all 376 draws runs first (about 1.8 h). It is needed only for OFO+'s further episodes, and it is throughput-neutral. Even at the high estimate, LOC and SVR+ on the paired episodes finish by about 05:30.

**Resumability.** Every (seed, variant) writes its JSON when it ends; the same command resumes and skips what exists. On e5power a dead shard is relaunched up to 3 times by `run_remote`, with the same call (no `--new`).

**Watching.** Per-run `wall_s` of the first SVR runs (e5power) and LOC runs (here) is in `runs/*.json` after about 1.5 h. The ETA is re-estimated from those, not from the README.

**Amendment A1 (2026-11-07 ~18:05, author, before the launch): CEN moves to e5wimi.**
- Load check at 18:00:
  - e5power: another user's 30 processes on 29.8 of 36 cores;
  - e5wimi: the other user's job ended, about 4 threads busy.
- The agreed split would have taken about 15 h on e5power.
- New split:

| host | variants | CPU-h | wall time |
|---|---|---|---|
| e5power | SVR, OFO (`--variants S1,O2`) | ≈ 209 | ≈ 11.5 h at ≈ 18 core-equivalents |
| e5wimi | LOC, CEN, SVR+, OFO+ and the prescreen (`--variants L2,M,S2,O3`) | ≈ 257 | ≈ 11.4 h at ≈ 22.6 core-equivalents |

- ETA about 06:30–07:30 on 8 Nov.
- E2 0029 on e5power (the fix over a full run) had the wall times SVR 141, LOC 91, OFO 61, CEN 44, OFO+ 25, SVR+ 23 min for 360 min. They are consistent with the per-episode estimates of the table above.

## 6. Commands (after the author's go)

The numbering:
- the smoke took mc/0001;
- the campaign of record here becomes mc/0002;
- e5power's share becomes 0003 (`reserve_number`).

**Launch the local share first.** Both run detached from any session: a `.cmd` launcher started as a WMI process.

`F:\qOFO_res\results\ch_11_case2_multisystem\mc\_launch\launch_newbase_e5wimi.cmd`:

```bat
cd /d F:\qOFO_res
F:\python_environments\qOFO_clean\python.exe -u -X utf8 -m experiments.ch_11_case2_multisystem.mc.run_mc --jobs 30 --variants L2,S2,O3 --buffer 12 --tag newbase --note "Case-2 MC on the new base (plan docs/architecture/2026-11-07_case2_mc_new_base_plan.md): e5wimi share LOC, SVR+, OFO+ and the full prescreen; SVR, CEN, OFO on e5power (tag newbase_e5power), merged with merge_mc.py" > results\ch_11_case2_multisystem\mc\_launch\newbase_e5wimi.log 2>&1
```

`F:\qOFO_res\results\ch_11_case2_multisystem\mc\_launch\launch_newbase_e5power.cmd`: the driver runs here and polls e5power every 5 min.

```bat
cd /d F:\qOFO_res
F:\python_environments\qOFO_clean\python.exe -u -X utf8 -m experiments.ch_11_case2_multisystem.mc.run_mc_remote --variants S1,M,O2 --jobs 30 --buffer 12 --tag newbase_e5power --note "Case-2 MC on the new base (plan docs/architecture/2026-11-07_case2_mc_new_base_plan.md): e5power share SVR, CEN, OFO; merged into the newbase campaign of e5wimi" > results\ch_11_case2_multisystem\mc\_launch\newbase_e5power_driver.log 2>&1
```

Start both, the local one first and the remote one a minute later:

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd /c F:\qOFO_res\results\ch_11_case2_multisystem\mc\_launch\launch_newbase_e5wimi.cmd'; CurrentDirectory = 'F:\qOFO_res' }
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd /c F:\qOFO_res\results\ch_11_case2_multisystem\mc\_launch\launch_newbase_e5power.cmd'; CurrentDirectory = 'F:\qOFO_res' }
```

**In the morning:**

1. Both logs end: `newbase_e5wimi.log` with the report; the driver log with `MC newbase_e5power: … pulled …, rc 0`.
2. Merge. Run a dry run first, then the merge:
   `python -m experiments.ch_11_case2_multisystem.mc.merge_mc <mc/0003_…_run_mc_newbase_e5power> <mc/0002_…_run_mc_newbase> [--dry-run]`
3. Final invocation here with all six variants. It runs only what is missing: replacements beyond the buffer, seeds whose prescreen differed, task crashes.
   `python -X utf8 -m experiments.ch_11_case2_multisystem.mc.run_mc --jobs 30 --variants S1,L2,M,O2,S2,O3 --buffer 12 --tag newbase --note "Final invocation after the merge of the e5power share"`
4. `run_mc --report --tag newbase`, then `evaluate_mc.py <campaign> --jobs 8`.

## 7. Preconditions before the launch

1. The E2 rerun `ref/0029` (all six) and the Case 1 rerun `ch_10_case1_svr/0004` on e5power are finished and ok, ETA 18:30–19:00. Their STS Q follows the q(v) law, so the fix holds over full runs. This also frees e5power's 12 cores.
2. The full test suite passes on the tree to be committed. **Done 17:30:**
   - 1041 passed, 15 skipped, 3 failed (76 min);
   - the three failures are the known order-dependent ones in `tests/experiments/test_ch9_procedure_v2.py` (daily logs 2026-11-06 and 2026-11-07b); the file passes alone (4/4);
   - the new tests pass (void rule, buffer, merge, and the runner test at OP-R).
3. The `run_mc` smoke passes: 2 episodes × 10 min, all six variants, `--buffer 1`. **Done 16:32:**
   - run `mc/0001_2026-11-07_162307_run_mc_smoke`: 18 / 18 ok (3 episodes × 6 variants; the buffer episode 20261009 ran too);
   - the runner's commitment equals the prescreen's in every run;
   - the uncommitted tree on 66c5618, only the docstring of `run_mc` differs from the tree to be committed.
   - The rates of the first 10 min are higher than the per-episode estimate, because they include the start transient. The ETA is checked on the first full runs.
     - Day episode 20261009 (24 Aug 10:45): LOC and OFO 2.7 min, SVR 8.0, SVR+ 6.6, OFO+ 2.3, CEN 1.9 min per 10 simulated minutes, init and calibration included.
     - Night episodes: LOC and OFO 0.4 min, SVR 2.4 (mostly calibration).
4. Commit in qOFO_res (`git -c safe.directory=*`), only the files of §4 and this plan. `sync_remote_code` then copies to e5power with a hash check.
5. Load check right before the launch: free cores on both hosts. If the other user's job on e5power is at full load, launch anyway (ETA about 15 h) or with fewer workers there; the author decides then.

## 8. Evaluation

- `evaluate_mc.py` on the merged campaign (rule (b), n = 100, cap 150). It writes `evaluation/`: `episodes.csv`, `events.csv`, `failures.csv`, `paired.csv`, `recruited.csv`, `summary.json`.
- Copies go to `graphics/Ch10/data/2026-11-08_case3_mc/` (new folder; the 2026-10-29 folder stays for the record). Then:
  - `make_tikz_tables.py` (taken over from the 2026-10-29 folder) writes `tikz/*.dat`;
  - `graphics/Ch10/case3_mc.tex` and `case3_mc_detail.tex` are pointed at the new folder (pgfplots, styles of `graphics/common/plotstyles.tex`).
- Text and tables: `tab:case3:mc`, `tab:case3:mc:tests`, `tab:case3:resources`, the numbers of `ch:case3:montecarlo` and `ch:case3:adverse`, `ch:case3:robustness:n1`, and the discussion. Also:
  - the provenance block (`latex_header.tex` of the run plus the merge);
  - the void rule D1 and the class counts on applied events;
  - the old todo on the three guard aborts, rechecked on the new base.
- Also reported: the number of void outages, the commitment over the episodes, the replaced episodes with causes, and old vs new per episode (same draws) as a check.
