# 2026-11-12b — MC rerun of SVR+ (S2) and OFO+ (O3) with the coupler dip hold: handover for the 36-core machine

**Status:** ready to run (author 2026-11-12: "i will do the rerun of MC on another machine with 36 cores").
**Why:** the coupler dip hold (20 mpu per STS step, 27 STS steps, early release at the STS band or at < 5 % upward
IBG headroom; `docs/daily_log/11_2026/2026-11-12_coupler_dip_hold.md`) is now part of the common Case-2 base. It
acts only where an STS-OFO exists (S2, O3). The MC campaign of record ran without it, so S2 and O3 must be re-run.
L2, S1, O2 and M are taken over unchanged: the hold has no effect there (R2 check: O2 identical to 7e-10 pu).

## Code

- Branch **`res/mc-dip27-2026-11-12`** (pushed to `origin`; also in the Z: repo `Z:\Python_Projekte\qOFO`).
  It is `res/coupler-dip-hold-2026-11-12` (5c2d8b0 + dip hold + R3 scenarios), merged with
  `res/rerun-base-2026-11-10` (a00fe71: CEN rule and the plant's warm power-flow re-solve, i.e. the code of the
  campaign's closing pass), plus `merge_mc.py --variants`.
- **Not** included: the OEL branch `res/oel-2026-11-11`. It changes the plant for every variant. In the campaign of
  record it was used only for S1 on 20261073 and 20261089. Re-running S2/O3 with it would break the pairing.
- Campaign of record: `Z:\Python_Projekte\qOFO\results\ch_11_case2_multisystem\mc\0001_2026-11-10_155634_run_mc_v5rb`
  (original on e5wimi `F:\qOFO_rerun\results\...`).

## Steps (Windows, conda env `qOFO_clean`, Gurobi licence on the machine)

1. Code on a **local** disk (network drive is ~3x slower), either
   `git -C Z:\Python_Projekte\qOFO worktree add D:\qOFO_mc_dip27 res/mc-dip27-2026-11-12`, or
   `git clone https://github.com/manuelschwenke/qOFO.git D:\qOFO_mc_dip27` and
   `git -C D:\qOFO_mc_dip27 checkout res/mc-dip27-2026-11-12`.
2. Check (in `D:\qOFO_mc_dip27`, `set PYTHONPATH=%CD%`):
   `python -m pytest -q tests/test_sts_coupler_dip_hold.py tests/test_merge_mc_variants.py tests/experiments/test_case2_variants.py`
   (expected: all pass) and the smoke run
   `python -m experiments.ch_11_case2_multisystem.mc.run_mc --smoke --new --variants S2,O3 --jobs 4`.
3. New campaign, S2 and O3 only (same seeds; the prescreen is recomputed and is deterministic):

   ```bat
   python -m experiments.ch_11_case2_multisystem.mc.run_mc --new --variants S2,O3 --n 100 --n-o3 100 --buffer 6 --jobs 34 --tag v5rb_dip27 --note "Ch 10 MC, S2 and O3 re-run with the coupler dip hold (20 mpu, 27 STS steps; author 2026-11-12), res/mc-dip27-2026-11-12" > results\run_mc_dip27_console.log 2>&1
   ```

   This runs both variants on the first 106 valid episodes (100 + buffer 6). Cost from the campaign of record:
   S2 median 12 min, O3 median 9 min per episode, about 42 CPU-hours, so roughly 1.5–2 h on 34 workers.
4. Merge L2, S1, O2 and M from the campaign of record (first with `--dry-run`, `prescreen_mismatch` must be empty):

   ```bat
   python -m experiments.ch_11_case2_multisystem.mc.merge_mc <campaign of record> results\ch_11_case2_multisystem\mc\<new folder> --variants L2,S1,O2,M
   ```

   `--variants` keeps the old S2/O3 runs out; they must never fill a gap of the new campaign.
5. Closing pass with all six variants (resumes the newest `v5rb_dip27` campaign; runs only what is missing):

   ```bat
   python -m experiments.ch_11_case2_multisystem.mc.run_mc --variants M,S1,L2,O2,S2,O3 --n 100 --n-o3 100 --buffer 6 --jobs 34 --tag v5rb_dip27 --note "closing pass: all six variants, window rule 2026-11-01"
   ```

   The window rule replaces an episode on which any variant failed. If S2/O3 fail on the same episodes as before,
   this pass runs S2 and O3 on 20261117 and 20261118 (the two top-ups of the campaign of record) and nothing else.
   If the window moves further, the other variants also run on the new episodes on this code (no OEL).
6. Tables and evaluation: `python -m experiments.ch_11_case2_multisystem.mc.run_mc --report --tag v5rb_dip27` and
   `python -m experiments.ch_11_case2_multisystem.mc.evaluate_mc results\ch_11_case2_multisystem\mc\<new folder> --jobs 34`.
7. Copy the whole campaign folder to `Z:\Python_Projekte\qOFO\results\ch_11_case2_multisystem\mc\` (same folder name).
   Then the thesis update (tab:case3:mc, fig:case3:mc, MC sentences in Ch 10).

## What to expect (screen of the campaign of record, logged voltages)

- The hold fires only at disturbances (load connections at bus 4/5, trips of G1, G2, G5). The largest dip at a
  non-event step was 18.1 mpu (O3) and 24.7 mpu (S2, logged; the measured value is about 25 % smaller).
- Episodes with a logged trigger (these are expected to change):
  - O3 (13): 20261008, 20261010, 20261015, 20261024, 20261039, 20261045, 20261064, 20261068, 20261089, 20261096,
    20261105, 20261107, 20261116.
  - S2 (16): 20261008, 20261010, 20261015, 20261024, 20261038, 20261039, 20261045, 20261064, 20261068, 20261080,
    20261089, 20261096, 20261100, 20261105, 20261107, 20261116.
- All other S2/O3 episodes should reproduce the old metrics (check `metrics_per_episode.csv`). A difference there
  means the measured dip with noise crossed 20 mpu or a code difference, so look at `dso_coupler_dip_hold` in the log.
- Of the eight replaced episodes, only 20261038 and 20261100 contain a trigger (both S2). Both stay replaced
  whatever S2 does, because S1 and O3 failed there. The window therefore changes only if S2 or O3 now fail on a
  triggered episode; step 5 handles that.
