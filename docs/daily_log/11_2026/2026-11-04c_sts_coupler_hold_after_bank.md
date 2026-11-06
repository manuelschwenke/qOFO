# 2026-11-04 — STS coupler hold after a TS bank commit (diagnostic switch + test campaign)

**Reason.** Author question (2026-11-04): tuning cannot get banks to engage after events
without coupler OLTCs reacting and reversing after the switch. Analysis of run 0006
Stage II-c (worktree `F:/qOFO_res`, only `g_w_tso_shunt` varied around the leader
8ad82442269b) showed:

- Every counted tap reversal of the rejected prices 14.1 (58bcdc27c248) and 17.9
  (29580fc8a748) sits at the **same instant** as a bank commit in the same STS (W3 at
  20880 s, W4 at 19440 s), and the reversing coupler is the **third** coupler of that STS
  (DSO_4|trafo_11, DSO_1|trafo_2), which hosts no bank.
- W3, 20880 s: the coupler moves identically at all three prices; it reverses after
  exactly 120 s (the coupler lock) only where z3[0] is released in the same instant.
- Runner order at a TS instant: TS writes → STS reads the plant **without** a re-solve →
  STS writes → power flow (`multi_tso_dso.py` TSO step, DSO step, `plant.advance`). The
  STS therefore decides on a measurement that predates the bank switch, and both layers
  answer the same deviation.
- Bank → STS mapping checked on the build: z2[0]/z2[1] DSO_1 (MSC on trafo 1, MSR on 0),
  z2[2]/[3] DSO_2 (4/3), z2[4]/[5] DSO_3 (7/6), z3[0]/[1] DSO_4 (10/9).

Author decision: "Yes, test the hold".

## Changes (≈ 10:00–10:20, shared checkout, branch minimal/thesis-core, not committed)

- `controller/base_controller.py`: new `BaseOFOController.hold_oltcs(n_iterations)`.
  Pins every OLTC integer (`_oltc_int_indices`) for the next `n` calls of `step()` through
  the existing iteration lock `_int_lock_until` (keeps a longer lock, extends a shorter
  one). Shunt integers and continuous actuators are untouched.
- `configs/config.py`: new field `sts_coupler_hold_after_bank: int = 0`, validated in
  `validate_integrator_mode()` (int ≥ 0). 0 = off = every run before today.
- `experiments/runners/multi_tso_dso.py`: after `receive_disturbance_message(msg)` of a
  TS-MIQP bank commit, the hosting STS calls `hold_oltcs(n)` when the field is > 0. The
  STS acts on a message it received, not on the plant. Local bank relays send no message
  and trigger no hold. Central mode (M) is not affected.
- `tests/test_sts_coupler_hold.py` (new, 11 tests): initialisation guard, OLTC-only lock,
  no-op for n ≤ 0, longer lock kept / shorter extended, n = 1 pins exactly the next step
  (bounds captured at the MIQP builder), config default and validation.
- `experiments/ch_09_parameter_selection/ch_9_13_coupler_hold.py` (new driver, thesis
  run via `start_thesis_run`, target `ch:timescales:weights`): three legs through
  `ch_9_3_stage2_screen.py` with run 0006's II-c flags, Stage-I paramset
  (`stage1_v3_uc_2026_11_pq10`, imported untracked) and operating points:
  - `h0` hold 0, 58bcdc27c248 on W3/W4: must reproduce run 0006;
  - `h1` hold 1 and `h6` hold 6 (= the 120 s coupler lock), the three II-c candidates
    on W1/W3/W4/W6.
  - This checkout's screen has no `--startup-exempt-s` (v3 amendment 2 lives on
    `res/commitment-base`). The driver re-scores every row from its persisted
    `moves`/`shunt_moves`/`events_s` with a copy of v3's `_split_reversals`
    (start-up 900 s). Checked: the re-scoring gives run 0006's stored counts for all
    12 windows; e_v,TS is aggregated as the quadratic mean like `aggregate()`
    (0.00864 / 0.00865 / 0.00869, as in the II-c log).
  - Verdict per leg: II-c pick rule as far as it is recorded per window (≤ 1 counted
    reversal, no bank reversal, e_v,TS within 2 % of the leader in the same leg).

## Verification

- `pytest tests/test_sts_coupler_hold.py tests/test_oltc_cooldown.py
  tests/test_msc_msr_banks.py tests/test_controller.py tests/experiments/test_case2_variants.py`:
  all pass except `test_m_is_ready_and_has_the_cascade_timescale`. That failure is not
  caused by this change: the test still expects the TS lock of `int_cooldown` 5 (18 min),
  while `tso_int_cooldown` is 4 (15 min) since 2026-10-30 (900 ≠ 1080 s).
- Screen dry run with `--config-set sts_coupler_hold_after_bank=1`: accepted and recorded
  in `run_meta.json`; operating points found cached (net key f8c419bf).

## Campaign

- Run folder `results/ch_09_parameter_selection/0014_2026-11-04_102104_ch_9_13_coupler_hold`,
  started 10:21 on e5wimi, 10 workers (h0 1, h1 5, h6 4), 26 units of 12 h; expected
  ≈ 2.5–3 h.
- The workers import the code of this shared checkout at start; edits to the controllers
  or the runner by other sessions during the run would enter later units.

## Result (run 0014, driver finished 13:13, rc 0, 2.87 h)

`compare.md` / `compare.json` in the run folder. Checks: re-scoring port reproduces run
0006's counts (True); h0 reproduces run 0006 on W3 and W4 exactly (metrics, tap and bank
move lists).

II-c pick rule on W1+W3+W4+W6 (counted reversals with the 900 s start-up exemption, bank
reversals, quadratic-mean e_v,TS, within 2 % of the leader in the same leg):

| bank price | run 0006 | hold 1 | hold 6 | e_v,TS (all legs) | bank commits |
|---|---|---|---|---:|---:|
| 14.11 (58bcdc27c248) | 2 → infeasible | 2 (W6) → infeasible | 3 (W6) → infeasible | 0.00864 | 16 |
| 17.92 (29580fc8a748) | 2 → infeasible | **0 → feasible** | **0 → feasible** | 0.00865 | 11 |
| 22.76 leader (8ad82442269b) | 1 → feasible | 0 → feasible | 0 → feasible | 0.00869 | 6 |

- Every coupler move at the instant of a bank commit in the hosting STS is gone at both
  holds (run 0006: 8 such moves over the three candidates in W3/W4). Bank commit instants
  and directions are identical to run 0006 except 17.9 W6 (z2[0] 5040 → 4860 s at both
  holds, z2[2]/z2[4] 6120 → 5940 s at hold 1; one TS period earlier, downstream of the
  changed start-up).
  No bank reversal anywhere.
- e_v,TS unchanged to 5 digits per window except W6 (±0.00002).
- Tap moves (TS + STS, 4 windows): 14.1 65 → 56/52, 17.9 58 → 48/48, leader 55 → 48/47.
- e_v,DS (quadratic mean over the 4 windows): 14.1 0.00603 → 0.00607/0.00600; 17.9
  0.00599 → 0.00613/0.00606; leader 0.00593 → 0.00615/0.00617. The rise is mostly W4
  (17.9 0.00488 → 0.00542, leader 0.00471 → 0.00554): there the coupler move that later
  reversed had lowered the DS voltage error in between.
- 14.1 fails only in W6, by the slow DSO_1|trafo_2 oscillation described below, not at a
  bank instant.

Interpretation:
- Fact: the hold removes the same-instant double response and its reversals, without
  changing bank dispatch or e_v,TS.
- Under the II-c rule with the hold, 17.9 becomes the lowest feasible price of the three
  tested, i.e. more bank commits (11 vs 6) at the same e_v,TS. Prices below 14.1 of the
  II-c grid were not re-tested with the hold.
- Not adopted: adoption changes STS behaviour in every TS-MIQP-bank variant and reopens
  procedure v3 Stage II-c; a matter for the v4 requirements (author).

## First TS dispatch (≈ 12:50–13:00)

Author question answered 2026-11-04: "Should the hold also apply at the first TS dispatch?"
→ **yes**. The runner already triggers the hold on every TS-MIQP bank commit, step 1
included; no behaviour change. Docstring of `sts_coupler_hold_after_bank` and the runner
comment now say so.

Evidence, probe `results/.../0014_.../probe_first_dispatch/` (screen window built exactly
as `_run_one_window`, stopped after 200 s by raising from the record constructor; the
hold-0 runs reproduce run 0006's recorded start-up moves, e.g. 58bc W6 at 60 s and 140 s):

- At step 1 (t = 20 s) the TS-MIQP engages MSCs in **all 12** probed (candidate, window)
  pairs: one MSC per STS (4) in W1/W4/W6 for 14.1 and 17.9, and in W4/W6 for the leader;
  only DSO_4's MSC in W3; only DSO_1's in the leader's W1.
- Without the hold, the STS steps of that instant move couplers in W4 (DSO_1|trafo_2,
  DSO_2|trafo_5, DSO_3|trafo_8, DSO_4|trafo_11, each −1) and W6 (DSO_1|trafo_2 −1).
  Several are reversed within 160 s: W4 DSO_1|trafo_2 and DSO_4|trafo_11 at 180 s, W6
  DSO_1|trafo_2 at 140 s. Same stale-measurement double response as in W3/W4 at the
  events.
- **Neither the step-1 bank commits nor the step-1 coupler moves are visible to the
  scoring**: `tap_moves` and `shunt_summary` (`tuning/stage2_search.py`) take the first
  record, written after step 1, as the reference. Part of the "start-up moves" that the
  v3 900 s exemption covers are the second halves of these hidden reversals.
- With hold 1 the hold fires at iteration 0 on all four STS (58bc W4, W6). W4: no coupler
  moves in the first 200 s. W6: the step-1 move of DSO_1|trafo_2 and its reversal at 140 s
  are gone, and DSO_1|trafo_1's move at 60 s too.
- Consequence for 58bc W6 under the hold: DSO_1|trafo_2's first recorded move falls at
  4880 s (after the bank commit z2[0] at 4860 s, with DSO_1|trafo_1), outside the start-up
  exemption; its later moves −1 at 7020 s and +1 at 7820 s (no event before 19440 s) then
  count as 2 reversals (3 at hold 6). This slow oscillation is not at a bank instant; the
  hold does not address it.
- DSO_2|trafo_5 is at −1 in the first record of W6 with and without the hold, so it is
  most likely its initial position, not a step-1 move (not checked further).

## Open

- Held-out W5 of the leader (run 0006: 31 tap reversals, 22 at a bank switch) not yet
  tested with the hold.
- Proposal, not implemented (changes the frozen v3/v4 scoring): score taps and banks
  against the state BEFORE the first dispatch, so step-1 commits and moves count.
- If the hold is adopted, it changes STS behaviour in every TS-MIQP-bank variant and
  reopens procedure v3's Stage II-c (and possibly the settling confirmation); it would
  need a declared amendment, coordinated with the tuning session.
