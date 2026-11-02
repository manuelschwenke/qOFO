# G1 limit by the operability rule, dispatch fix (technical minima first), REF period with G3 offline

Timestamp: 2026-10-31, ~07:20–10:00 Europe/Berlin. Branch `res/commitment-base-2026-10-31`,
worktree `F:\qOFO_res`. Side session of the Case-2 main session, on the author's decisions of
the morning:
- G1 limit set by the operability rule (~07:20);
- stop the old-base STEP, and start a new tuning campaign autonomously if the check requires
  it (~07:55);
- keep ±1200 Mvar, since it is effectively non-binding (~08:30);
- declare and queue a REF period with G3 offline (~08:50).

## What changed (key structures)

| commit | file | change |
|---|---|---|
| 2091fdf, b79d4a6 | `docs/analysis/2026-10-31_g1_operability_rule.md` | rule declared before the tests; result; amendment A1 |
| 114b0ac | `configs/config.py` | `slack_q_limit_mvar` 500 → **1200** Mvar (rule A1) |
| 35b447e | `network/ieee39/zonal_balancing.py` | **bug fix** in `compute_merit_order_dispatch` (details below) |
| 0c3f5c2 | `docs/analysis/2026-10-31_ref_period_G3_offline_declaration.md` | rule for the REF period with G3 offline, declared before the search |

**Dispatch fix (35b447e).**
- **Before:** in each zone, the first committed machine was loaded up to its cap
  (`clip(rest, pmin, cap)` in merit order) before the later machines' technical minima were
  honoured. When the residual fell inside the window, the zone over-generated while hydro
  stayed at its cap, and the surplus went to the network equivalent.
- **Now:** every committed machine first gets its technical minimum. Only the residual above
  the sum of the minima is loaded in merit order up to the caps.
- **Unchanged:** commitment, inter-zone deficit handling, the equivalent as balance.
- **Effect** (`disp_cmp.py`):

| window | max change in machine P | G1 export peak, before → after |
|---|---|---|
| REF period, 6 h | 83 MW (13:45) | — |
| W1, 12 h | 395 MW | −1577 → −1273 MW |
| W6, 12 h | 280 MW | unchanged |
| RA period | 0 MW | — |

## Findings

**G1 limit**

| case | result |
|---|---|
| W1, LOC, G1 unlimited | fails at ≈19:48 after drawing 1181 Mvar from G1, so the declared rule has no solution |
| W6, LOC | needs 1062 Mvar |
| W2–W4, LOC | need < 400 Mvar |
| old-base MC (808 runs, G1 unlimited) | never more than 1000 Mvar; above 500 in 62 runs |
| new-base REF period incl. post-event states | ≤ 381 Mvar |

- Amendment A1 gives **±1200 Mvar**.
- The limit does not bind in any operable state found so far. The new base therefore differs
  from the old one through commitment, not through G1.

**W1 evening collapse**
- The old base runs W1 under LOC with G1 at about 0 Mvar.
- On the new base, the 12-h window commits all five machines for the midday peak (residual
  sum 2686 MW). At 19:45 the residual is 773 MW, and area 2 alone has a surplus of 574 MW.
- With the machines at their minima, the system must export more than 1.2 GW through G1, and
  LOC loses voltage stability.
- Under OFO, before the fix, G1 unlimited passes with 1386 Mvar from G1, but ±1200 fails at
  19:51.
- **Cause:** commitment is fixed per run window from the window's peak, so a 12-h window with
  a large residual swing over-commits in the evening. A 30-min window at 19:00 commits only
  G10 and G4 and is healthy (V_min ≈ 1.0).
- Re-test on the fixed dispatch: see the table at the end.

## Runs

**Stopped**
- **Old-base STEP** at 07:58, on the author's instruction: 404 / 1350 units, partial results
  in `F:\qOFO_weekend\results\ch_11_case2_multisystem\step\0001_...`. The weekend chain then
  logged the end of Case 1, and the queued old-base E started at 07:58.
- **`chain_res.py` waiter** (07:28): its note carried ±500 Mvar. Replaced by `chain_res2.py`.

**New-base REF**
- First start at 08:03 (runs 0001–0003). Stopped at 08:53 because of the dispatch fix.
- Restarted at 08:54 at 35b447e: A, E, B in parallel (3 × 6 workers), then D.

**e5power:** re-validation and MC not deployed. They wait for the W1 re-test on the fixed
dispatch, because W1 is a re-validation window under OFO.

## Open

- **Commitment window length** is a modelling choice with a visible effect (W1 evening). Fixed
  per window from the peak; an hourly schedule with minimum up and down times would remove it.
  That would be an architectural change and needs the author.

## Results after the fix (09:00–10:00)

**W1 and W6 re-test, ±1200**

| window | LOC | OFO |
|---|---|---|
| W1 | ok, G1 max 927 | ok, G1 max 913 |
| W6 | ok, G1 max 1087 | — |

- The re-validation windows are operable without events.
- Before the fix: W1 under OFO at ±1200 failed at 19:51. Coordination did not hold a binding
  ±500 either (W1 OFO fails ≈17:33; W6 OFO and OFO+ fail after 75 and 87 min).

**G3-offline REF period**
- Selected: **08 Jan 2016 06:00** (G10, G9, G4, G7 committed), by the declared rule.
- Search table: `docs/analysis/2026-10-31_ref_period_G3_offline_declaration.md`.
- Scenarios **GA** and **GE** in `ref/scenarios.py` (be66584). Commit 01117ea carries only the
  document; its registry edit did not apply and followed in be66584.

**Launches**

| what | host | when | details |
|---|---|---|---|
| re-validation | e5power | 09:38 | run 0010, 70 units, 20 workers; chain `chain_res2.py e5power`: MC follows if c08b is confirmed |
| GA, GE | e5wimi | queued | `chain_res3.py`: GA when the second of A/E/B ends, GE at the third (6 workers each) |

## Status 13:15

**e5wimi REF (new base, weights c08b)**

| run | scenario | state |
|---|---|---|
| 0004 | B | done 12:56, all six variants ok (1080 records each) |
| 0006 | A | done 12:56, all six ok |
| 0005 | E | five ok, SVR (S1) still running |
| 0007 | D | started 12:56 |
| 0008 | GA | started 12:57 |
| — | GE | starts when E ends |

**e5power re-validation, run 0010: c08b is not confirmed on the new base**
- **Units:** all 70 units ok. 0 of 70 reproduce the stored e_v(TS) bit-exactly, which is
  expected because the base changed.
- **Select:** 5 of 14 feasible, ranked 301c285ee52b, 9e48320e8a01, e95cb58c9043, 34a9f0cf1444,
  4ccfa217c2b5.
- **Rejected:** c08b, the baseline 3cb66225e1a7 and 7 others, all for tap "hunting" (a reversal
  within < 2 h dwell that no event explains).
- **The rejections share one reversal.** For c08b, the baseline, 67ad, ff3f, 0efa, c9a5 and
  7935 it is the same one:
  - window **W6**, transformer `1[0]`;
  - first move at the first dispatch (t = 180 s, or 360 s for the baseline), reversed at
    t ≈ 4,700–5,600 s;
  - dwell 1.25–1.5 h, before W6's first event at 19,440 s.
- 57f0 and 8323 additionally reverse twice in W3 after an event.
- **Reading (hypothesis, not tested):** this looks like an initialisation transient of the new
  base in W6 (the window with the largest import, G1 up to 1087 Mvar under LOC), not hunting
  caused by the weights. The frozen filter counts it nonetheless.
- **Settle:** 301c's settling result from run 0008 is a fail, so Experiment C runs on
  **9e48320e8a01** (8 workers): 40 / 120 cases at 13:10, ETA ≈16:30.
- **Consequence:** DECISION.json will report `unchanged: false`, the script exits 3, and the
  chain logs "MC NOT started". By the weekend plan §5.1, the settling winner then becomes the
  new paramset and the campaigns rerun on it. Which path to take (§5.1 winner, the full Ch 9
  procedure, or the author's judgement of the W6 reversal) is put to the author.

## Complete Ch 9 procedure on the new base (author decision (b), ~13:30)

**Declaration.** `docs/analysis/2026-10-31_ch9_full_procedure_uc_declaration.md`, committed
with the code in 4d02199 before any unit ran.

**Code (4d02199)**

| file | change | why |
|---|---|---|
| `tuning/stage1_screen.py` | `load_or_build_operating_points(horizon_s=)`, CLI `--op-horizon-h`; `_network_key(cfg, horizon_s)` adds commitment, horizon and slack limit when commitment is on; paramset provenance records them | the capture ran 60 s, so with commitment it would commit from the first minute, not from the 12-h window the controller runs in; without the key change a cache captured with all machines in service would be loaded silently |
| `experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py` | `--op-horizon-h`, `--skip-baseline` | the same capture for the λ screen; shards of a split run |
| `experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py` (new) | driver: run 0001's steps, the 2026-10-28 procedure with the engagement bound in every selection; master on e5wimi, e5power reached through PowerShell remoting (detached shard processes, done markers, zipped pull of window files); `--smoke` | run 0001's driver is not re-runnable in place, and one host alone would need ≈55 h |

- **Key structure.**
  - `simulate()` splits the explicit candidate list by worker share and runs the unchanged
    stage-2 screen on each host.
  - It then pulls the remote `windows/` and re-invokes the screen with the whole list, which
    finds every unit on disk and assembles the records.
  - `settle()` is run 0001's loop, with batch members alternating between the hosts.
- **Checked:**
  - `tests/tuning/test_stage1_banks.py`: 6 pass.
  - The 12-h and 60-s keys differ: `netf8c419bf` and `neta68582a1`.
  - Both differ from the key run 0001 loaded, `net82afed5b`.
- **Deployed** to e5power by bundle at 13:47. The run-0010 settling was still running there.
  Its code path is unaffected (only new flags and a new file).

**Run** `results/ch_09_parameter_selection/0001_2026-10-31_135000_ch_9_6_full_procedure`
(e5wimi)

**13:50–13:59 Stage I**
- Written as `configs/paramsets/stage1_start_uc_2026_10.json`.
- The 12-h captures commit per window. Worst λ_max(B) = 1.000 at W6, STS DSO_4 and TS area 2.
- Against `stage1_start_10_2026_banks`, the weights change by at most 4 %:

| weight | old | new |
|---|---|---|
| g_w_der, g_w_pcc | 0.0931 | 0.0897 |
| g_w_tso_oltc | 21.37 | 21.31 |
| g_w_tso_shunt | 27.21 | 26.98 |
| g_w_dso_der | 0.5017 | 0.5038 |
| g_w_dso_oltc | 16.64 | 16.26 |

- g_v, g_q, dso_g_v and g_w_gen are unchanged.

**14:00–14:18 smoke test** (`results/_smoke_ch9_6/20261031-140016`, code 1f5932e). Every path
ran on both hosts:
- II-a: 5 survivors, 3 here and 2 on e5power; the windows were pulled back and the assembly
  ran 0 units with 6 records;
- selection with the engagement bound;
- one II-b round;
- settling with one candidate here and one on e5power, whose result was pulled and read by
  the criterion.
- The toy outcome ("no candidate passes", tap counts per 12 h from 6-min windows) carries no
  meaning.

**14:19 campaign launched** (`--resume`, code 1f5932e, master pid 130452 on e5wimi, log
`results/res_logs/20261031_ch9_6_run.log`)
- Stage II-a: 115 / 512 draws pass the screen; run 0001 had 120.
- Split: 47 candidates here (14 workers) and 68 on e5power (20 workers).
- ETA: II-a ≈03:00 Sun, the 4 rounds ≈18:00 Sun, settling ≈Sun 23:00 – Mon 01:00.

**15:00 restart with 30 workers on e5power** (author: "yes please", after the note that
e5power has 36 physical cores without hyperthreading and the campaign used 20 of them)
- **Stopped:** the master and both shards. When the remote shard was killed first, the master
  relaunched it once (launch 2, 14:59); that relaunch was killed as well. No done marker was
  written.
- **Kept:** the 21 units finished before the restart, 6 here and 15 there, now copied to both
  hosts.
- **Code:** e5power synced to e83c75c. The untracked copy of the Stage-I paramset was removed so
  the fast-forward could write the committed one.
- **15:01 relaunched** (`--resume`, `--w-remote 30`):
  - split 36 here (14 workers, 175 units to run) and 79 on e5power (30 workers, 384 to run);
  - II-a ETA ≈00:00–01:00.
- **Engagement bound, analytic, no simulation result used.** θ_STS,min of the 115 II-a
  survivors on the 12-h captures:

| statistic or threshold | value |
|---|---|
| median | 0.068 pu |
| admitted at θ ≤ 0.05 | 47 |
| admitted at θ ≤ 0.025 | 31 |
| admitted at θ ≤ 0.02 | 26 |
| admitted at θ ≤ 0.015 | 15 |
| Stage-I point | 0.0150 |
| c08b | 0.0195 |
| fde2 | 0.0816 |

**~15:20 amendment 2** (author: "Declare 0.02 now")
- The engagement bound becomes θ_STS,min ≤ 0.02 pu, the "far" level of the author's intent
  check in run 0008. It was 0.05 pu, at which a coupler commits only at the band edge.
- Declared in `docs/tuning/2026-10-31_amendment2_sts_engagement_0p02.md`, with a pointer in the
  campaign declaration.
- The declaration states what was known: analytic θ values only, and c08b at 0.0195 just
  below the bound.
- **Driver:** a module constant `THETA_MAX = 0.02` replaces amendment 1's import; it is
  recorded in the selection log line, the run parameters and the result provenance.
- **Restart:** the master is restarted during Stage II-a so the bound acts from the first
  selection. The remote shard (30 workers) keeps running, and the local shard resumes from its
  unit files, losing its ≈14 units in progress.
- **15:33 relaunched** on 3d98d31, identical on both hosts:
  - remote shard found alive (pid 4332, since 15:03) and not relaunched;
  - local shard resumed with 175 units to run;
  - 37 remote units on disk at 15:40.
- The first commit attempt failed on PowerShell 5.1 argument quoting: a message containing
  double quotes was split. Committed with `git commit -F`.

**Status 15:40 of the other runs**
- **c08b REF on the new base:**
  - E done 15:32 (SVR 397 min);
  - D running, S1 and M left;
  - GA running, O3, S1 and M left; O3 has taken > 2.5 h, against 27 min in A;
  - GE started 15:33.
- **Old base:** REF E of the main session (`F:\qOFO_scarcity`) done 15:19, all six ok.
- **Re-validation run 0010:** settling e95c, 40 / 120 cases.

**Status 23:50**
- **Ch 9 rerun, Stage II-a:**
  - the remote shard ended at 22:55 (rc 0): 395 units in 7.9 h, ≈36 min per unit at 30 workers;
    its window files are pulled;
  - the local shard is at 144 / 185 at 23:45. It is the slower half: 14 workers on a host shared
    with the c08b REF runs until 22:21 and with other sessions' processes (CPU 100 %);
  - e5power has been idle since 22:55.
- **Re-validation run 0010 finished by itself at 17:26** (the author had allowed stopping it;
  there was nothing left to stop):
  - 9e48 failed settling (p95 34, median residual 1.81 Mvar);
  - e95c passed (p95 23, residual 0.35 Mvar);
  - DECISION.json: winner e95cb58c9043, unchanged False.
  - It is a record only; the full rerun replaces it.
- **c08b REF on the new base, all finished:**
  - D (0007) at 17:55, all six ok;
  - GA (0008) at 22:21, all six ok; S1 e_v 31.0 mpu;
  - GE (0009) at 19:43: **five of six variants end with `LoadflowNotConverged`** (Newton–Raphson,
    50 iterations). Only CEN runs the full 360 min. The other five fail in this order, all
    after the area-2 load connection at 212 min:
    - SVR at 212 min;
    - LOC at 313 min;
    - SVR+ at 335 min;
    - OFO at 342 min;
    - OFO+ at 354 min.
  - In GE, area 2 has no synchronous machine (G3 offline). Its time outside the band is
    26–50 %, and 58 % for CEN.
  - Reported as found. Whether the non-convergence is a voltage collapse or a numerical
    failure is open.
