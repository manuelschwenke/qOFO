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
