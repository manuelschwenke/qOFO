# Complete Ch 9 weight selection on the unit-commitment base: declaration

**Author decision.** 2026-10-31, ~13:30, option (b): "Full Ch 9 campaign now".

**Context.**
- The re-validation (run 0010 on e5power, weekend plan §5) did not confirm c08b1d3d33ab on
  the new base.
- c08b, the Stage-I point and seven others are rejected for the same single tap reversal: W6,
  transformer `1[0]`, first move at the first dispatch, reversed after 1.25–1.5 h (daily log
  2026-10-31b, status 13:15).

**Declared** by Claude Code before any unit of this campaign runs. Commits:
- this document;
- the driver `experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py`;
- the two flags it needs in `tuning/stage1_screen.py` and `ch_9_3_stage2_screen.py`.

> **Amended 2026-10-31 ~15:20**, before the first selection of the run, by author decision.
> The engagement bound is **θ_STS,min ≤ 0.02 pu** instead of 0.05 pu.
> See `docs/tuning/2026-10-31_amendment2_sts_engagement_0p02.md`.
> Everything else in this declaration stands.

## 1. What is applied

**The procedure.** `docs/tuning/2026-10-28_weight_selection_procedure.md` (frozen), **unchanged**:
- Stage I analytic point;
- Stage II-a: 512 Sobol draws, seed 20260910, spans as stated there, analytic screen at 1.546;
- Stage II-b: 64 draws per round, seed 20260910 + r, f = 0.5 halved on no change, end at
  f < 0.125 or after 4 rounds;
- **the filter including the engagement bound θ_STS,min ≤ 0.05 pu**, in every selection,
  i.e. after II-a, after each round, and the final one;
- ranking on e_v(TS) with the 2 % band on e_v(DS);
- the settling requirement (p95 ≤ 27, median residual ≤ 1 Mvar) in rank order.

**The base**, as of commit 35b447e plus the commits of 31 Oct:
- merit-order unit commitment (technical minima nuclear 0.5, coal 0.4, hydro 0.1 S_n), fixed
  per run window from its peak residual;
- dispatch with the technical minima first;
- equivalent Q limited to ±1200 Mvar;
- AVR range [0.95, 1.05] pu;
- TS tap lock 15 min;
- DSO band 0.92–1.08 pu;
- banks in the TS-MIQP;
- DSO_3 ×2.

**Windows.** W1–W4 and W6 of `tuning.scenarios_mc_v2.CONDITIONS`, 12 h, evented, frozen
network. W5 is simulated for the report only.

## 2. What the new base requires (and nothing else changes)

1. **Stage I is recomputed.** Its operating points are captured again, because the committed
   machines change H and B.
   - The capture uses the **commitment of the 12-h window** it screens: `--op-horizon-h 12`.
     That is the commitment the Stage-II simulation of the same window runs with.
   - Until today the capture ran 60 s. With commitment, that would commit from the first minute
     only.
   - Same design inputs as `stage1_start_10_2026_banks`:
     - Δv_eng = 0.0075 (TS taps), 0.015 (STS taps), 0.010 pu (banks);
     - π = 1, τ = 1, g_w,gen = 10⁸, λ targets 1.
   - Written as `configs/paramsets/stage1_start_uc_2026_10.json`.
2. **The operating-point cache key** includes the commitment, its horizon and the slack
   limit (`tuning.stage1_screen._network_key`). A capture with all machines in service is
   never loaded for the commitment base.
   - The captures of this campaign live in its run folder (`operating_points/`). The engagement
     bound is evaluated on them.
3. **Experiment C** keeps the call of run 0001, with the new Stage-I set as `--paramset`. Its
   step cases commit over their own run length.
   - Stated assumption: the isolated-STS bank freezes the TS parent after its single solve at
     t = 0, so the commitment enters only through that one operating point.

## 3. Execution (does not change any decision)

**Hosts.**
- e5wimi (master) and e5power. Each stage's candidates are split in proportion to the
  workers.
- Each host runs the unchanged `ch_9_3_stage2_screen.py` on its share, through the new
  `--skip-baseline` flag. Only the master's share includes the Stage-I point.
- The window files are brought to the master, which assembles the candidate records with
  one more invocation of the same script.

**Workers.**
- Stage II-a: 14 on e5wimi, because REF D/GA/GE share that host until ≈17:30, and 20 on
  e5power.
- From II-b on: 20 + 20.
- Settling: four candidates at a time, as in run 0001, with 10 workers each, two per host.

**Plumbing test.** A toy-size smoke test (`--smoke`, folder `results/_smoke_ch9_6/`)
precedes the run. It uses 16 draws, one round of 8, and 6-min windows; nothing from it enters
the result.

**Expected cost.**
- About 1,000 worker-h, as in run 0001: 243 candidates × 5 windows × ≈43 min, plus settling.
- That is ≈30–35 h wall time on 34–40 workers.

## 4. Outcome

- The first candidate in rank order that passes settling is the result. It is written as
  `configs/paramsets/ch9_selected_uc_2026_10.json`.
- Switching `SELECTED_PARAMSET` and the campaigns (REF, MC) to it is a separate step, after
  the author has seen the result.
- The REF runs on c08b of 31 Oct (new base) are kept as a weights-sensitivity record.

## 5. Stated in advance

- If the W6 reversal at the first dispatch is a start-up property of the new base, it also
  rejects candidates here, because the filter is unchanged. That is part of the procedure's
  outcome, not a reason to amend it during the run.
- If no candidate passes the filter and settling, the run ends with `FAILED.txt`, and the
  author decides.
