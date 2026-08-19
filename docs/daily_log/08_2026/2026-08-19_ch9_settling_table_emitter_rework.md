# 2026-08-19 — Ch 9.1 Table 9.1: the emitter and the thesis table are made the same object again

**Author:** Manuel Schwenke / Claude Code
**Timestamp:** 2026-08-19, Europe/Berlin
**Reason:** Task A of the Ch 9 §9.1 handoff
(`00_daily_log/2026-08-19_handoff_ch9_experiments_A_B.md`). The open-loop
settling battery emitted a *different* set of rows from the one printed in the
thesis, so the table was being filled by hand — and a hand-transcription error
had already got in. Fixing the number alone would leave the mechanism that
produced it intact, so the emitter was reworked instead.

## The defect

`TABLE_ROWS` emitted 8 rows; the thesis table prints 7 *different* ones (AVR
split by machine, a machine-transformer OLTC row added, the two-step
instrument row dropped). Because the two sets never matched, nobody could
paste — and the coupler-tap row in the thesis reads

| | thesis | measured, `20260807-085455` |
|---|---|---|
| OLTC coupling transformer, one step | `11.13` s, location "STS 1 B00" | **`16.28`** s at `u_DSO_1_bus43` |

`11.13` appears nowhere under `results/timescale/`; `B00` is an unfilled
placeholder. Consequences carried into the chapter: the binding row is the
**coupler tap at 16.28 s**, not the machine transformer at 15.13 s; the margin
at `T_STS = 20 s` is **3.72 s**, not the printed 4.87 s; and the ordering
sentence inverts.

## What changed

| change | where |
|---|---|
| `TABLE_ROWS` reworked to the 11 thesis rows, one emitted line each, in thesis order | `experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py:173` |
| AVR rows split by machine **and** magnitude (`+0.02` / `+0.001` pu, G 09 / G 10) | same, and `pf/screening.py::AVR_VREF_MAGNITUDES` |
| machine-transformer tap given its own table row (`tap_+1_MT`) | same |
| two-step tap rows dropped from the table, **kept in the battery** | same |
| new catalogue case: `avr_vref_+0.001` on both machines | `pf/screening.py::default_catalogue` |
| new catalogue case: `tap_+2seq_MT_*`, instrument only | `pf/screening.py::default_catalogue` |
| tap split `T_mech`/`T_elec` computed **per tap class** | new `_tap_split()`, used by `derive()` |
| summary reports both splits in their own section | `write_outputs()` |
| `--self-test` extended; pytest file extended 12 → 17 tests | `tests/experiments/test_ch9_timescale_seperation.py` |

The location column is emitted from the measured worst signal and is never
typed. Matching stays **literal substring, never regex** — `StepDef` names
contain `+`, which a regex reads as a quantifier.

## The one real trap in this change

Adding `tap_+2seq_MT_*` gave the catalogue a *second* two-step case. `derive()`
paired taps with an unqualified `"tap_+2seq" in case`, which would then have
matched whichever class the catalogue emitted first — pairing a coupler
one-step against a machine-transformer two-step and producing a `T_mech`
belonging to neither transformer. **No symptom:** the arithmetic still closes.
The split is now class-qualified (`tap_+1_NC3W`/`tap_+2seq_NC3W` against
`tap_+1_MT`/`tap_+2seq_MT`) and the synthetic fixtures give the two classes
deliberately different splits so a leak is detectable.

## QVPRE — option (a), and why

The battery steps DER Q through `QVPRE.qset`, so the local re-anchored Q(V)
droop is inside every measured settling time. Taken: **(a) keep the physics,
reword the caption** to "no secondary dispatch; primary control (AVR,
governors, local Q(V)) active".

The handoff expected the battery to differ from closed loop because `Vanchor`
is not re-anchored during it. Reading the code, it does not:

- closed loop writes **both** `qset` and `Vanchor` at every dispatch
  (`pf/plant.py:727-733`), re-anchoring the droop to the voltage measured at
  the dispatch instant, so `veff = x1 - Vanchor` starts at ~0;
- the battery anchors at the load-flow point (`pf/wecc_apply.py:318-352`) and
  the preflight holds that point to `1.4e-10` pu drift, so `veff` is also ~0
  at the step instant.

The battery therefore reproduces the closed-loop anchoring condition rather
than degrading it, which makes (a) the *faithful* option and not merely the
cheap one. Neutralising `QVPRE` would measure a plant that never operates.
Deadband is `0.01` pu; whether the droop engages at all is now checkable from
`--save-trajectories` and is reported with the run.

## Status

Seat-free checks pass on the reworked code (`--self-test` all-pass; `--dry-run`
prints `gen[1] -> G 03`, `gen[7] -> G 09`). Battery launched from this commit.

## Unresolved

- `tests/pf/test_screening_event_pool.py::test_persistent_pool_grows_admits_and_retires_events`
  fails **at HEAD**, independently of this change (`created == 2`, expected 3,
  while `pending_admission == 3` passes). Not touched here; flagged separately.
