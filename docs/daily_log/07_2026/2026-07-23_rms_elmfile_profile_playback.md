# Production RMS profile playback through ElmFile

**Timestamp:** 2026-07-23 15:56 CEST  
**Reason:** Replace the correctness- and initialization-cost-heavy
per-element profile events with trajectories known before `ComInc`, while
preserving the measurement-dependent OFO command path.

## Architectural boundary

The controller implementation was not changed.  TSO and DSO controllers still
receive only plant measurements and cached sensitivities.  Their controlled
outputs remain TS/DS interface reactive-power flows and nodal voltages.  Their
actuators remain DER Q references, synchronous-machine AVR references, OLTCs,
and shunts.

Only known exogenous inputs moved to file-backed playback:

- profile-driven load P/Q:
  `ElmFile.y1/y2 -> ElmLod.Pext/Qext`;
- profile-driven DER P:
  `ElmFile.y1 -> WTGWGO_A.Pref_in`.

Online `qset`/`Vanchor`, AVR, tap, and shunt commands remain pre-created
one-shot events because their values depend on the preceding RMS measurement.
The rejected live-IntMat/DSL mailbox alternatives are documented in
`2026-07-23_rms_profile_mailbox_probes.md`.

## Code changes

- Added `pf/profile_playback.py`.
  - Generates one shared plain PowerFactory measurement file from the runner's
    already-clipped profile DataFrame.
  - Reproduces the former event timing: initial row at RMS `t=0`, then wall
    profile step `k` at `(k-1)*dt + 0.51 s`.  The 0.51 s transition is the
    former 0.5 s event offset plus the observed 10 ms RMS firing row.
  - Creates 53 load sources/composites and 44 DER-P sources.
  - Copies the existing project-local WECC frame, adds one file slot, and
    routes `y1` to the weak-grid block's `Pref_in`.
  - Restores/removes only objects carrying the `qOFO RMS Profile` ownership
    prefix before an idempotent reinstall.
- Extended `PowerFactoryReplayFactory` with selectable
  `profile_delivery={"elmfile","events"}` and a pre-construction profile
  configuration hook.
- Extended the multi-TSO/DSO runner to pass its exact clipped profile table to
  a plant factory before plant construction.
- `PowerFactoryPlant.apply_exogenous` now updates only its pandapower mirror
  under ElmFile playback; the physical PowerFactory model follows the
  preloaded RMS-time trajectory.
- `ScreeningContext.prepare_persistent_event_pool` can discard selected
  qOFO-owned keys.  ElmFile mode deletes only profile-driven `EvtLod` slots
  and `Pref_in` parameter slots; controller and discrete-actuator slots remain.
- The replay CLI defaults profiles-on runs to `--profile-delivery elmfile`;
  `--profile-delivery events` retains the old path for comparison.

## Verification

Python verification, forced UTF-8:

- syntax compilation of the six changed production modules: PASS;
- new schedule/file/pruning tests: **5 passed**;
- wider focused suite: **17 passed, 2 unrelated pre-existing failures**:
  one stale event-admission test signature and the existing Gate-E test fixture
  omitting the temporary `der_q_capability_override_pu=1.0`.

PowerFactory 2025 SP4 smoke:

```text
run_rms_phase6_replay.py --duration 40 --profiles
  --profile-delivery elmfile --profile-settle 1 --no-pdf --verbose 1
```

Result:
`results/rms_phase6_replay/0048_2026-07-23_154718`.

- `ComInc` accepted both generated composite frames.
- The 40 s RMS closed loop completed and Gate E reported PASS.
- Event folder: 4,661 objects, down from 9,511 before migration.
- Event classes: 4,650 `EvtParam`, 11 `EvtTap`, **0 `EvtLod`**.
- Remaining `Pref_in` event slots: **0**.
- Active file-backed objects: 53 load sources, 53 load composites, 44 DER
  sources, one load frame, one extended WECC frame.
- Controller events still fired: parameter pool usage 173/4,650.
- Generated file has 10 channels and three 40 s smoke rows at
  `t={0, 0.51, 20.51}` s.

## Assumptions and constraints

- PowerFactory 2025 SP4, balanced RMS, 10 ms integration/output grid.
- Profile values are finite and at most 23 distinct profile columns are
  referenced (one of ElmFile's 24 channels is reserved for unity).
- The profile file remains unchanged until `ResetCalculation`; it is not an
  online mailbox.
- The smoke retained the temporary P-independent DER Q capability override
  of +/-1.0 pu already present in the Gate-E configuration.

## Risks and unresolved points

- A long run still pre-creates one-shot online controller events.  Its
  initialization cost therefore scales with dispatch intervals, but only with
  the controller-event set rather than controller plus 97 profile events per
  interval.
- With physical P-dependent DER capability diagrams restored, Q limits must
  also evolve with DER P.  This was already unresolved in the legacy
  profile-event path; the current +/-1.0 pu diagnostic override masks it.
- The smoke proves DER frame acceptance, zero `Pref_in` events, and successful
  closed-loop operation.  A dedicated high-amplitude DER-P trace comparison
  would provide a stronger quantitative test of the routed `Pref_in` signal
  than the slowly varying 40 s profile window.
