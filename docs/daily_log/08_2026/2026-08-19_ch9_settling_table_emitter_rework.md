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

---

# Addendum, 2026-08-19 ~15:45 — the battery run FAILED, and why

Launched from clean commit `705b017` at 13:09. Aborted after ~30 min:

```
File "pf/screening.py", line 769, in initialise
    raise PFSessionError("ComInc (RMS init) failed")
```

**Not a defect in the battery, and not a licence problem.** A read-only
diagnostic (`ComLdf` then `ComInc`, modifying nothing) localised it:

```
[diag] connect + activate: 23.7 s
[diag] ComLdf.Execute() -> 1  (1.4 s)   DID NOT CONVERGE
[diag] ComInc.Execute() -> 2  (0.0 s)   FAILED
```

The PowerFactory output window gives the cause — every profile source in the
study case cannot open its data file:

```
err - Grid\qOFO RMS Profile DER Source 0.ElmFile:
      Cannot open measurement-file
      "...\results\rms_phase6_replay\0543_2026-08-07_142902\snapshot\rms_profiles_elmfile.txt"
```

repeated for every `ElmFile` (DER sources 0-3, load sources 21, 22, ...).

**The referenced run does not exist.** `results/rms_phase6_replay/` now begins
at `0559_2026-08-07_195513`; run `0543` has been deleted. With every profile
source dead the load flow does not converge, and `ComInc` cannot initialise
from a non-converged operating point.

## Why it worked on 2026-08-07 and not now

The run of record is `20260807-085455` — 08:54. Replay run `0543` is from
14:29 the **same day**, i.e. *after* it. `pf/profile_playback.py:365` writes an
absolute path into each `ElmFile`:

```python
source.SetAttribute("f_name", str(Path(file_path).resolve()))
```

So every replay run repoints the shared `02_RMS_CoSim` study case at its own
results snapshot, and the study case then breaks as soon as that snapshot is
cleaned up. The battery did not change; the study case was silently left
pointing at a transient directory by a later, unrelated run.

**This is a latent fragility, not a one-off.** Any future use of
`02_RMS_CoSim` breaks the same way after the next results prune.

## Options (author decision — the PF project was not modified)

1. Repoint the `ElmFile` sources at a surviving snapshot (earliest is `0559`)
   — only valid if that profile matches the intended operating point.
2. Regenerate the snapshot for the battery's operating point.
3. Give the battery its own study case, or have it re-point the sources itself,
   so a replay run cannot leave it broken.
4. Deactivate the profile sources for the open-loop battery: it holds a fixed
   operating point (preflight drift `1.4e-10` pu in the run of record), so
   time-varying playback may not be needed at all — but the run of record did
   have them active, so this changes what is measured.

**Nothing in the PF project was modified by this session.** The `--self-test`
and `--dry-run` paths both pass on the reworked code, so Task A's code changes
are verified as far as they can be without a seat-side model that initialises.

---

# Addendum 2 — the ElmFile repair was necessary but not sufficient

After the `ComInc` fix (above), the battery reaches its preflight and is
refused there. Three ElmFile configurations were tried:

| `ElmFile` state | `ComLdf` | `ComInc` | preflight drift |
|---|---|---|---|
| pointing at deleted `0543` | 1 (fail) | 2 (fail) | — never reached |
| out of service | 0 | 0 | `1.36e-02` pu |
| repointed at `0566` (a trajectory) | 0 | 0 | `1.42e-02` pu |
| repointed at a **frozen t0** profile | 0 | 0 | `1.36e-02` pu |

Run of record, 2026-08-07: `1.41e-10` pu. Tolerance: `1e-4`.

**The profile is not the cause.** Frozen and out-of-service give the *same*
drift to three digits, and the trajectory differs by only 4 %.

## What the drift actually is

Established by read-only diagnostics (all in the session scratchpad, none
promoted):

1. **`ComInc` initialises exactly on the load flow.** P and Q compared across
   all 163 injectors and loads: `sum|dP| = 0.000` MW, `sum|dQ| = 0.000` Mvar.
   The initial condition is not the problem.
2. **Q(V) anchoring is consistent.** All 44 `QVPRE` blocks carry
   `Vanchor == V_LF` and `qset == Q_LF/S_n` to machine precision; `|dV|` max
   `0.000e+00`. Also `db = 0.5` pu, so the droop cannot engage at all.
3. **No discrete action.** No `ElmTr2`/`ElmTr3`/`ElmShnt` tap or step moves
   during the flat run.
4. **The trajectory is a step, not a ramp.** The worst signal moves
   `1.02194 -> 1.03528` within the first sample and then sits flat
   (`1.03528 -> 1.03580` over the remaining 54 s).
5. **Active power redistributes.** Over 60 s the four TSO parks each shed
   0.43-0.56 MW (`sum|dP| = 5.2` MW over 44 parks) while the synchronous
   machines gain `7.4` MW, G 01 alone `+5.3` MW, and machine Q *falls*
   (G 01 `107.4 -> 103.9` Mvar). The voltage-dependent `*_var_*` loads then
   follow. That is a governor response to a power imbalance, not a controller
   hunting.

So the plant is released at the load-flow point, discovers a real-power
imbalance, and relaxes to a different equilibrium `1.4e-2` pu away.

## Leading hypothesis, NOT verified

The frozen profile freezes *`0566`'s* channel values, which encode that run's
park dispatch — not the dispatch the current static model (and hence the load
flow) carries. The parks are then driven to a P that the load flow did not
solve for, and the governors close the gap.

Testing it needs the per-source channel scaling: 97 `ElmFile` sources share a
single 10-channel file, so the channels are normalised shapes and each source
applies its own scale. Confirming the hypothesis means resolving that mapping;
fixing it means **re-exporting a profile from the current model state** rather
than reusing a recorded one.

**That is a model-data operation and it was deliberately not attempted
unattended.** It changes what the benchmark *is*, which is an author decision.

## Where the study case was left

Pointing at `pf/profiles/rms_profile_t0_frozen.txt`, all 97 sources in
service. `ComLdf` and `ComInc` both succeed, so the case is usable — it is
strictly better than the dead `0543` path it had, and the target is under
`pf/`, which is not pruned. Three restore files record every prior state, in
order:

| file | state it restores |
|---|---|
| `elmfile_outserv_20260819-185102.json` | original: in service, pointing at the deleted `0543` |
| `elmfile_outserv_20260819-185508.json` | out of service, still pointing at `0543` |
| `elmfile_outserv_20260819-191003.json` | in service, pointing at `0566` |

They live under `results/`, which is gitignored and pruned — **copy them
somewhere durable if the original state matters.**

## Status of Task A

- Code: **done and verified offline.** `--self-test` all-pass (27 checks),
  `--dry-run` resolves `gen[1] -> G 03` and `gen[7] -> G 09`, 17 pytest cases.
- Run: **blocked**, and correctly so. Every settling time measured from this
  model state would carry a `1.4e-2` pu relaxation, which is 14x the `1e-3` pu
  voltage band the table is measured against. The preflight is the only reason
  that did not become a table of plausible-looking wrong numbers.
- The 2026-08-07 numbers in `20260807-085455` remain the best available, with
  the defect the handoff identified: the thesis prints `11.13` s for the
  coupler tap against a measured `16.28` s.
