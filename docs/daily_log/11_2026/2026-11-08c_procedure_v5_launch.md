# 2026-11-08c — Procedure v5 (B) and its follow-ups (C, D1, D2): design, dry run, launch

Session "STS capabilities tuning analysis". Branch **`res/tuning-v5-2026-11-08`** (from
`res/network-north-2026-11-08` @ e1bceec). Worktrees: `F:\qOFO_v5` (e5wimi) and
`F:\qOFO_runs\qOFO_v5` (e5power, from bundle). Diagnosis before this: `2026-11-08_sts_underuse_diagnosis.md`.

## Author requests and decisions (2026-11-08)
- "Adapt tuning to have good STS support"; the network session's hand-off: `rural_700_north_res`
  is the study network (62c4944).
- "Please start all experiments (B, C, D1, D2) … 30 workers on e5power, 20 workers on e5wimi."
- Asked before coding (two decisions, both answered with the recommended option):
  - STS rule: **"Bound g_w_pcc"** (declared bounds from G3-off zone 2);
  - G3-off states: **"Lambda bank + report"** (λ screen with both G3 states; winner simulated in
    G3-off MC episodes as a report, not ranked).

## Pre-check on the new network (v4 weights; `2026-11-08c_nr_capture.py`, `_nr_gpcc.py/.out`)
35 captures: W1–W6 (12 h, all G3 on) and the 29 valid mini-MC seeds (3 h; 20 G3 off).
- Zone 2 has 9 PCC, 1 machine (G3), 3 taps, 6 banks, **no TS IBG**; zone 3 has W3 + 2 machines.
- λ (banks excluded), v4 weights: TS_1 worst 1.47/1.50 (G3 on/off), TS_2 0.24/0.28, TS_3 ≈ 0.56;
  STS_DSO_4 worst 1.585 > 1.546 at MC20261084 (G3 on): the v4 STS design does not carry over.
- G3 off, zone 2 PCC-only loop gain at v4: median 0.156 (≈ 14 TS periods for 90 %).
- First-step voltage-effect share of the STS, σ_V = Σ_pcc (g_v/g_w)c² / Σ_cont (g_v/g_w)c²:

| g_w_pcc | σ_V zone 2 (G3 on) | σ_V zone 3 | λ TS_2, G3 off |
|---|---|---|---|
| 0.873 (v4) | 0.47 | 0.12 | 0.28 |
| 0.19 | 0.80 | 0.38 | 1.02 |
| 0.12 | 0.87 | 0.49 | 1.60 |

- Not used: an AVR Mvar equivalent from H's Q_gen row. It gives dQ_G3/dV_set = 22 Mvar/pu at
  W1, which is implausibly small (open point; the row feeds only the g_z_q_gen soft constraint).

## Code (commit 7910d84, + e17263f notes)
- `docs/tuning/2026-11-08_weight_selection_procedure_v5.md`: FROZEN before the campaign.
- `experiments/ch_09_parameter_selection/ch_9_17_procedure_v5.py`: v4 + patches (`configure`):
  - `load_ops`: W1–W6 captured on rural_700_north_res into the run (v4 copied run 0001's);
  - `prepare_lambda_bank`: + `ch_9_13` grid (96 starts) at 12 h and 3 h, parallel capture;
    both G3 states;
  - `compute_bounds`: κ_p = λ_max(g_v H_V,pccᵀH_V,pcc) of TS_2 on the G3-off points;
    g_w_pcc ∈ [max κ / 1.546, min κ / 0.5];
  - `passes_screen` = responsiveness ∧ v4 screen; `stage2a` with the bound; `spans_v2` sets the
    g_w_pcc span to the interval half-width; `design` = v3 Stage I with g_w_pcc at the centre;
  - result `ch9_selected_nr_2026_11_v5` (provenance `v5`), Stage-I sets `stage1_v5_nr_2026_11_*`,
    `P.REMOTE_ROOT` = `F:\qOFO_runs\qOFO_v5`; `--reuse-ops` copies captures of a dry run.
- `ch_9_18_followups_v5.py`: wait for GO → `ch_9_16.record` → SELECTED_PARAMSET switch (commit)
  → C (`ch_9_12.settling_figure`) → in parallel D1+D2 on e5power (30, `ch_9_16.deadband`, run
  note v5) and the G3-off report here (`mini_mc`, 21 seeds, O2/O3, 20 jobs).
  - The G3-off seeds are pinned to the network session's list: the prescreen's `uc_offline` is
    the base network's (it marks 28 seeds) and does not identify G3-off on rural_700_north_res.
- `ch_9_16_ref_after_v4.deadband`: optional `note` (default unchanged).
- `tests/test_procedure_v5.py`: 5 tests (bounds, responsiveness, span, isolation, grid); with
  `tests/test_procedure_v4.py` 10 passed.

## Dry run (`results/_dry_ch9_17/20261108-183806`, 12 min)
- Bank: 186 / 192 captured; 6 failures = 3 starts at both horizons (F0408_14, F0622_14,
  F0822_14), `LoadflowNotConverged` at initialisation (listed in `lambda_bank_failures.json`).
- 16 G3-off points; κ 0.1423 … 0.1694 → **g_w_pcc ∈ [0.1096, 0.2846]**, centre 0.1766 (±0.207 decade).
- Stage-I point (θ_STS 0.01): g_w_pcc 0.1766, g_w_der 0.0897, g_w_tso_oltc 21.3, g_w_tso_shunt
  14.92, g_w_dso_der 1.528, g_w_dso_oltc 11.34, g_w_gen 5e6; λ on the bank 1.409 (W1–W6 1.400),
  λ_STS 0.806, θ_TS 0.0076. II-a: 238 / 512 draws pass (v4: 176).

## Launch
- e5wimi was at ≈ 52–64 % CPU before the launch (42 python processes of another logon), so the
  host split uses `--local-speed 0.6`.
- Driver (detached, pythonw, no console): pid 15524, 18:5x,
  `--reuse-ops results\_dry_ch9_17\20261108-183806 --config-set sts_coupler_hold_after_bank=1
  --local-workers 20 --remote-workers 30 --verify-workers 30 --confirm-workers 30 --local-speed 0.6`.
- Run folder `results/ch_09_parameter_selection/0001_2026-11-08_185224_ch_9_17_procedure_v5`
  (the v5 checkout numbers its own runs, so 0001 here is not run 0001 of `F:\qOFO_res`).
  It reused the dry run's 192 captures and re-tried the 6 failed ones.
- Follow-up queue `ch_9_18_followups_v5.py --run <that folder>` (detached, pythonw): pid 101548,
  log `<run>/followups_v5/driver.log`; it waits for DONE/FAILED and acts only on GO.json.

### Restart as run 0002 (18:55)
- Run 0001 stopped at 18:55 before Stage I: `V3.sync_remote_code` built an empty file list, because
  e5power was at this checkout's commit, and Get-FileHash failed on an empty path.
  - Fix b8f30f0: return early when no file differs; test added.
  - e5power fast-forwarded to b8f30f0 by bundle.
  - Run 0001 has `ABANDONED.txt`; its queue logged "campaign FAILED" and exited (SKIPPED.txt).
- **Run `0002_2026-11-08_185554_ch_9_17_procedure_v5`**: driver pid 159568, same arguments, with
  `--reuse-ops` of run 0001.
  - Same bounds as the dry run: [0.1096, 0.2846], centre 0.1766.
  - "e5power: no Python file differs from b8f30f0".
  - Stage I `stage1_v5_nr_2026_11_pq10`: ρ 2.5e4, κ 0.262, θ_STS 0.0100, θ_TS 0.0076,
    λ_max(B) on W1–W6 1.400.
  - Verification shard on e5power (30 workers) started 18:57.
- Queue for run 0002: pid 49828, armed 18:57:50.
- **19:25 Stage-I verification PASS at the first rung** (θ_STS 0.01): p95 21.05 ≤ 25, median
  residual 0.21 Mvar ≤ 0.8.
- **19:36 Stage II-a started.**
  - 238 / 512 draws pass the screen. The λ ceiling on the 192-point bank leaves 443, θ_TS ≤ 0.009
    leaves 295, and λ_STS ≥ 0.5 holds for all 512 (the g_w_pcc box is the interval itself).
  - Split: 68 candidates here (20 workers), 170 on e5power (30; remote pid 26772).

### 21:00 progress check (no code change)
- Driver (159568) and queue (49828) alive. e5wimi at 100 % CPU (other logon plus the 20 workers).
- II-a windows done: 72 / 276 here, 163 / 680 on e5power. Rates 0.86 and 1.9 windows/min, so II-a
  should end at about 01:00–01:30. v4 run 0008 needed 20.5 h in total, of which 3.8 h for II-a.
- Of 244 window rows: 0 failed simulations. 212 rows were early-stopped.
  - 53 of the 63 candidates touched so far were lost to the early stop.
  - That rate is biased high, because stopped windows finish first. v4 II-a lost 166 of 177.
- Stage-I point e5d47f3be21e: 0 tap and 0 bank reversals on W1/W3/W4/W6, mean e_v_ts 0.00590.
  v4's Stage-I point had 3 tap and 1 bank reversals.
- Losing a candidate does not correlate with g_w_pcc (Spearman +0.01). In v4 II-a, the lost
  candidates had a median g_w_pcc of 0.173, against 0.322 for the feasible ones.
- 6 candidates are feasible so far (tap 0, bank 0). This is a provisional view: mean e_v_ts over
  the four windows, not the procedure's engagement-level selection.
  - The leader is 6b1aacf04e8b: e_v_ts 0.00533, g_w_pcc 0.196, g_w_der 0.0178, ratio 11 (v4: 61).
- σ_V, the STS voltage-effect share, uses the definition of `2026-11-08c_nr_gpcc.py` on the
  35 pre-check captures. The last column is λ_pcc at the smallest κ.

| set | g_w_pcc/g_w_der | σ_V z2 (G3 on) | σ_V z3 | λ_pcc z2 (G3 off) |
|---|---|---|---|---|
| v4 061d | 61 | 0.47 | 0.12 | 0.16 |
| 6b1aacf04e8b (leader) | 11 | 0.61 | 0.27 | 0.73 |
| 09690e55fa9f | 2.1 | 0.75 | 0.46 | 1.08 |
| 498bb86f10fe | 1.1 | 0.66 | 0.38 | 0.83 |
| e5d47f3be21e (Stage I) | 2.0 | 0.93 | 0.73 | 0.81 |
| 53e4bbdb03b5 | 0.2 | 0.68 | 0.41 | 0.78 |
| 33416077ca8e | 0.2 | 0.99 | 0.96 | 0.92 |

- Open point: v5 bounds g_w_pcc absolutely, but not relative to g_w_der. The ratio to the TS IBG,
  which matters in zone 3, is therefore left to the e_v_ts ranking.

### 2026-11-09 01:57 Stage II-a selection
- The local shard finished at 23:50 and e5power at 01:52; 680 files were pulled. No driver error.
- Selection 2a:
  - 239 / 239 candidates lie inside the θ bounds; 86 reach engagement level 'sts+bank';
  - **40 are feasible at hunting tolerance 0** (v4 run 0008 II-a: 3).
- Ranking: e_v_ts, 2 % band, then e_v_ds as the tie-break.

| key | e_v_ts | e_v_ds | g_w_pcc | g_w_der | ratio | σ_V z2 (G3 on) | σ_V z3 | λ_pcc z2 (G3 off) |
|---|---|---|---|---|---|---|---|---|
| **6b1aacf04e8b** (1st) | 0.00556 | 0.00672 | 0.196 | 0.0178 | 11.0 | 0.61 | 0.27 | 0.73 |
| 1504773ceeba | 0.00555 | 0.00681 | 0.142 | 0.0164 | 8.7 | 0.72 | 0.36 | 1.00 |
| 56d48f30f188 | 0.00556 | 0.00688 | 0.117 | 0.0229 | 5.1 | 0.66 | 0.35 | 1.22 |
| 6f8b5f488e13 | 0.00566 | 0.00676 | 0.155 | 0.0305 | 5.1 | 0.75 | 0.42 | 0.92 |
| Stage-I point e5d47f3be21e | 0.00611 | 0.00742 | 0.177 | 0.0897 | 2.0 | 0.93 | 0.73 | 0.81 |

- Observation: the e_v_ds tie-break again picks the band member with the largest g_w_pcc and the lowest
  σ_V, the same direction as in v4. The v5 bound now limits it to g_w_pcc ≤ 0.285.
- 01:57 Stage II-b round 1: f 0.5, centred on 6b1aacf04e8b; 30 / 64 draws pass; 8 run here, 22 on e5power.
- Pending, decision by the author: H sn_mva slip (fixes A/B, `2026-11-08d_qgen_row_check.md` on
  `res/qgen-row-check-2026-11-08`). v5 continues on the as-built H.

### 2026-11-09 12:19 Stage II-b result (4 rounds), start of II-c
- Rounds: 1 (f 0.5) → 158fdfc8e812; 2 unchanged → f 0.25; 3 → 74ef403199e5; 4 → **55a1878c7730**.
  II-b stops at MAX_ROUNDS = 4. No driver error.
- From 08:30 (round 3 onward) the local shards ran on 12 of 40 logical CPUs (affinity 0x555555, Normal
  priority; colleagues work on e5wimi until 18:00). Round 4 here: 08:26–12:18; e5power: 08:27–10:51.
- Selection 2b round 4:
  - 372 / 372 inside the θ bounds; 190 at engagement level 'sts+bank';
  - **78 feasible at hunting tolerance 0**.

| key | e_v_ts | e_v_ds | g_w_pcc | g_w_der | ratio | g_w_tso_shunt | g_w_tso_oltc | g_w_gen |
|---|---|---|---|---|---|---|---|---|
| **55a1878c7730** (1st) | 0.00552 | 0.00647 | 0.203 | 0.0158 | 12.9 | 19.9 | 12.7 | 3.7e5 |
| ae17a8cfdcce | 0.00549 | 0.00654 | 0.218 | 0.0225 | 9.7 | 20.1 | 8.9 | 4.0e5 |
| 10cd507c196a | 0.00551 | 0.00658 | 0.186 | 0.0153 | 12.2 | 18.1 | 12.3 | 5.7e5 |
| 74ef403199e5 (round-3 best) | 0.00550 | 0.00659 | 0.198 | 0.0181 | 10.9 | 18.1 | 11.1 | 3.8e5 |
| 158fdfc8e812 (round-1 best) | 0.00547 | 0.00665 | 0.181 | 0.0275 | 6.6 | 20.0 | 10.0 | 3.7e5 |
| Stage-I point e5d47f3be21e | 0.00611 | 0.00742 | 0.177 | 0.0897 | 2.0 | | | |

DSO weights are equal in all rows (g_w_dso_der 1.528, g_w_dso_oltc 11.34).

- **e_v_ts is flat across the band:** 0.00547–0.00552, a spread of 0.9 % inside the 2 % indifference
  band. Rounds 2–4 moved only the tie-break e_v_ds, from 0.00665 to 0.00647 (−2.7 %).
- **Against the Stage-I point:** e_v_ts −9.7 %, e_v_ds −12.8 %.
- **The tie-break again prefers a large g_w_pcc / g_w_der ratio** (12.9 for the leader). This is the
  direction already noted at II-a. g_w_pcc stays inside the v5 bound (≤ 0.285).
- **Stage II-c** (12:19):
  - leader 55a1878c7730, g_w_tso_shunt 19.93;
  - grid of 7 prices [4.72, 5.80, 7.12, 8.75, 10.75, 13.21, 16.23], all 7 pass the screen;
  - 2 run here (8 window runs, 12 CPUs) and 5 on e5power (20 window runs); no early stop.
- **ETA** from v4 run 0008's stage durations (II-c 1.1 h, completion 2.9 h, settling 1.2 h, W5 + GO
  0.4 h): II-c ends ≈ 13:30–14:15, completion ≈ 17:00–17:45, settling ≈ 18:15–19:00 (20 CPUs
  here from 18:00), GO / NOGO ≈ 18:45–19:30.

### 2026-11-09 13:31 Stage II-c result (bank price alone), completion on W2 started
- Leader 55a1878c7730 had g_w_tso_shunt 19.93. Seven lower prices were run on W1, W3, W4 and W6
  without the early stop. e5power finished at 13:18, the local shard at 13:30.

| price | key | e_v_ts | hunting | bank hunting | bank after events | status |
|---|---|---|---|---|---|---|
| 4.72 | f44f0f26a03e | 0.00541 | 2 | 1 | 4 windows | rejected |
| 5.80 | 3ed9c17d101b | 0.00540 | 1 | 0 | 4 | rejected |
| 7.12 | d6639aa6fa69 | 0.00542 | 2 | 0 | 4 | rejected |
| **8.75** | **45b169134c8c** | 0.00546 | 0 | 0 | 4 | **feasible, pick** |
| 10.75 | 586e637a1c6f | 0.00548 | 0 | 0 | 4 | feasible |
| 13.21 | 86ef9f91328e | 0.00550 | 0 | 0 | 3 | feasible |
| 16.23 | 22ca384cc720 | 0.00551 | 1 | 0 | 3 | rejected |
| 19.93 | 55a1878c7730 | 0.00552 | 0 | 0 | 3 | feasible (leader) |

- **Pick:** 45b169134c8c, the lowest feasible price. Its e_v_ts lies inside the leader's 2 % band.
  It joins the completion band.
- **Observation 1: the pick sits one grid step from the hunting region.** 7.12 and every lower
  price hunt.
- **Observation 2: feasibility is not monotone in the price.** 16.23 hunts once, while 13.21 and
  19.93 do not. A single hunting event decides feasibility at tolerance 0, so the label is
  sensitive to individual events.
- **Completion (13:31):** 21 band candidates plus the Stage-I point on W2; 6 run here, 15 on
  e5power.
- **ETA:** completion ≈ 15:30–16:30, settling ≈ 17:00–17:45, GO/NOGO ≈ 17:30–18:30.
- The trial paramset `trial_55a1_v5_run0002` (branch `res/trial-v5-2026-11-09`, author's manual
  check) carries the II-b leader's price 19.93, not the II-c pick.

### 2026-11-09 15:34 Completion result (21 band candidates + Stage-I point, ranking on W1–W4, W6)
- e5power finished at 15:30 (75 window runs: its checkout reran all 5 windows per candidate). The
  local shard finished at 14:09 (7 W2 runs). Selection `selection/selection_complete_sts+bank.json`.
- 22/22 inside the theta bounds, 22 feasible at hunting tolerance 0. Engagement level `sts+bank`
  holds 21 candidates; only the Stage-I point is outside it.
- Band: best e_v_ts 0.00546 (f54bd61bcbbc), 2 % → ≤ 0.00557. Ranks 1–13 lie inside the band and
  are ordered by e_v_ds. Ranks 14 and up lie outside it and are ordered by e_v_ts.

| rank | key | e_v_ts | e_v_ds | g_w_pcc | g_w_der | ratio | g_w_tso_shunt | bank commits | AVR max step |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **ae17a8cfdcce** | 0.00551 | **0.00675** | 0.218 | 0.0225 | 9.7 | 20.09 | 6 | 0.00426 |
| 2 | 45b169134c8c (II-c pick) | 0.00553 | 0.00681 | 0.203 | 0.0158 | 12.9 | 8.75 | 20 | 0.00468 |
| 3 | 74ef403199e5 | 0.00556 | 0.00687 | 0.198 | 0.0181 | 10.9 | 18.08 | – | 0.00450 |
| 4 | f54bd61bcbbc | **0.00546** | 0.00695 | 0.169 | 0.0359 | 4.7 | 18.83 | – | 0.00477 |
| 15 | 55a1878c7730 (II-b leader, trial) | 0.00558 | 0.00677 | 0.203 | 0.0158 | 12.9 | 19.93 | 6 | 0.00468 |
| 22 | e5d47f3be21e (Stage-I point) | 0.00617 | 0.00751 | 0.177 | 0.0897 | 2.0 | 14.92 | 12 | 0.00035 |

- **Selected: ae17a8cfdcce**, λ_max(B) 1.530. The Pareto set on (e_v_ts, e_v_ds) is
  {f54bd61bcbbc, ae17a8cfdcce}.
  - Weights: g_w_der 0.02253, g_w_pcc 0.21849, g_w_gen 404118, g_w_tso_oltc 8.930,
    g_w_tso_shunt 20.09. The rest is as in Stage I (g_v 4e4, g_q 0.4, dso_g_v 1e4,
    g_w_dso_der 1.528, g_w_dso_oltc 11.34).
- **The II-b leader 55a1878c7730 falls out of the band** (0.00558 / 0.00546 = 1.022). The W2 runs
  shifted the aggregate: on W2 it reaches 0.00582, f54bd61bcbbc 0.00543.
- **The II-c price is not adopted.** The II-c pick (8.75, the leader's weights) ranks 2nd on e_v_ds.
  The winner is another candidate, at its own price 20.09. This follows the procedure as written:
  II-c adds a candidate to the band and does not set the price of the eventual winner. At the lower
  price, the bank switches 20 times instead of 6.
- e_v_ts across the whole band varies by only 2.7 % (0.00546–0.00561). The pick is decided by
  e_v_ds, which differs by 0.9 % between ranks 1 and 2.
- **Settling confirmation** of ae17a8cfdcce started 15:34 on e5power (30 workers). In v4 it took
  72 min and W5 + go criteria 24 min, so GO/NOGO ≈ 17:10.

### 2026-11-09 16:37 Settling confirmation PASS, W5 report, **GO**: ch9_selected_nr_2026_11_v5 = ae17a8cfdcce
- **Settling confirmation** (e5power, 30 workers, 15:34–16:09). 120 periods: 5 windows × 4 DSOs
  × 3 interfaces × up/down, 10-Mvar steps. W5 is excluded (VDE dead zone).
  - **p95 N_inner 20.05** (limit 27); max 24; 0 periods above the limit; 0 censored.
  - p95 per DSO: DSO_1 17.6, DSO_2 12.0, DSO_3 22.6, DSO_4 10.0.
  - Median residual 0.25 Mvar.
  - `expC/criterion_results.json`.
- **W5 (report only, not ranked):**

  | key | e_v_ts | e_v_ds | hunting | bank commits | AVR max step | V violations |
  |---|---|---|---|---|---|---|
  | ae17a8cfdcce | 0.00493 | 0.00630 | 0 | 8 | **0.00764 pu** | 0 / 0 |
  | Stage-I point e5d47f3be21e | 0.00554 | 0.00636 | 2 | 10 | 0.00058 pu | 0 / 0 |

  - **The AVR step of the winner exceeds the 0.005 pu limit in W5.** The limit is checked on the
    ranked windows only, where the worst step was 0.00426. W5 is the window in which the IBG
    reactive capability is zero (VDE dead zone), so the AVR setpoints carry more of the
    adjustment.
- **Go criteria: GO.**
  - e_v_ts 0.00551 ≤ 0.01.
  - STS couplers move in every search window: W1 5, W3 3, W4 9, W6 20 moves.
  - Banks act after events in 3 windows (≥ 2 required).
  - All limits are met.
  - Engagement level sts+bank.
- **Written:** `configs/paramsets/ch9_selected_nr_2026_11_v5.json`. Its fields equal the trial
  `trial_ae17_v5_run0002` on `res/trial-v5-2026-11-09`.
- The driver finished rc 0 after 21.68 h. The follow-up queue (record, switch, C, D1+D2,
  G3-off report) polls every 5 min.
