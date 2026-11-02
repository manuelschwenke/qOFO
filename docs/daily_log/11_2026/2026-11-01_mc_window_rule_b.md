# MC amendment: comparison window rule (b), voltage at the horizon end no failure

Timestamp: 2026-11-01, ~10:50–11:30 Europe/Berlin. Branch `res/commitment-base-2026-10-31`,
worktree `F:\qOFO_res`.

Author decisions:
- ~00:30 GE: leave as recorded, no replay of the failing steps for now.
- ~10:45 MC on the new weights: "count them as failures and run another run for each failed
  one such that we have 100 successful in total". Asked to choose between the readings, the
  author took (b) with "a separate report of failures and causes of failures".
- Also: a voltage out of band at the end should not be a failure.

## What changed (key structures)

| file | change |
|---|---|
| `experiments/ch_11_case2_multisystem/mc/run_mc.py` | window rules by date: `WINDOW_RULES`, default `"2026-11-01"`; `comparison_window(valid, outcome, n, rule)` (any failure replaces an episode once all six ran on it); `N_VALID_MAX = 150` stop condition (`--max-valid`); `classify` no longer returns `fail_v_end` and flags `v_end_outside_hard_band` instead; `effective_outcome` re-reads older `fail_v_end` records as `fail_osc`/`ok`; `failure_row` and `failure_report` write `failures.md` and `failures_per_run.csv`; `report` limits the distributions and paired tests to the window and notes incomplete window episodes; run JSON stores `dt_s`, `t_ts_s`; campaign parameters record the rule; CLI `--rule`, `--max-valid` (`--no-top-up` = `--rule 2026-10-07`) |
| `experiments/ch_11_case2_multisystem/mc/evaluate_mc.py` | same rule and outcome definition; `--rule`, `--max-valid`; `summary.json["window"]` has `rule`, `replaced` |
| `tests/experiments/test_mc_window.py` | rewritten: rule 2026-11-01, the earlier rules, `effective_outcome`, `failure_row`, `failure_report` (15 tests) |
| `docs/architecture/2026-11-01_mc_amendment_window_and_failures.md` | the amendment: decision, stop condition, statistical meaning, check |
| `docs/architecture/2026-10-28_case2_step_mc_plan.md` | pointer to the amendment in §5 |

## Reason

- The paired comparisons should sit on episodes where all six variants complete.
- Failures, and why they happen, are a result of their own and get their own report.
- **Stated to the author before the choice:** under (b) the comparison metrics are conditional
  on all six completing. The failure counts stay unconditional, because every variant runs on
  every valid episode up to the last one used.

## Checks

- `pytest tests/experiments/test_mc_window.py tests/experiments/test_mc_draw.py`: 17 passed.
- **Re-report** of the old-base campaign `qOFO_weekend/.../mc/0001_2026-10-30_184316_run_mc`,
  from a copy of its JSONs in the scratchpad; the campaign folder is untouched:
  - rule 2026-10-30: 4 replaced, all infeasible;
  - rule 2026-11-01: 7 replaced (+3 tap-hunting episodes: L2 twice, S1 once);
  - 3 window episodes, where only O3 ran, are flagged as not yet run by all six;
  - no run ended outside the hard band.
- **Driver smoke** (`--smoke --jobs 2 --out <scratch>`, 2 episodes × 10 min):
  - 13 runs (6 × 2 plus O3 on a third episode), all `ok`;
  - window 2 / 2, rule 2026-11-01, 0 replaced;
  - `report.md` and `failures.md` written;
  - run JSON carries `dt_s` 20, `t_ts_s` 180 and `v_end_outside_hard_band`.

## Note on the running Ch 9 campaign

- The commit changes only MC files. The Ch 9 driver compares git HEADs at start only.
- Stage-II invocations started after this commit record the new HEAD in their run metadata.
  Their code (`ch_9_3_stage2_screen.py`, `tuning/`) is unchanged.
- e5power stays on 3d98d31 until the next deploy.
