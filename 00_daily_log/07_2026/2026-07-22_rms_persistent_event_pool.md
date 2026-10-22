# RMS persistent high-water event pool

**Timestamp:** 2026-07-22 18:25 CEST  
**Reason:** PowerFactory 2025 SP4 admitted only a bounded number of event
objects created during an active RMS calculation per `ComSim.Execute`. With
profiles enabled, the approximately 177 events per 20 s interval exceeded
that rate and caused progressively late or missing plant commands.

## Established live-probe facts

1. An unused event created before `ComInc` can be moved from an inert future
   time into the active horizon and fires on-grid.
2. A fired object cannot be re-armed in the same active calculation.
3. Alternating two objects does not change that rule: A/B fired at 1/3 s,
   while reuse of A/B at 5/7 s failed for both `EvtParam` and `EvtLod`.
4. The exact same fired objects are reusable after
   `ResetCalculation -> ComInc`. The cross-ComInc probe produced exact Q
   responses in both runs and load steps at 1.005 s in both runs.
5. Two `ComSim.Execute(tstop=current_time)` calls admitted a dynamically
   created 200-event batch without advancing simulated time. The last-created
   Q sentinel and all 199 load events fired at 1.005 s.

## Implementation

- `pf/screening.py`
  - Added named persistent `EvtParam`, `EvtLod`, and `EvtTap` pools.
  - Pool preparation resets the calculation, discovers retained qOFO-owned
    objects, sets every slot to `time=1e9`, resets cursors, and deletes only
    unmanaged stale events.
  - Missing slots grow on demand. Newly created live slots are admitted using
    zero-time `ComSim` calls with a conservative 64-event batch bound.
  - Fired slots are moved back to the inert timestamp after a full RMS-step
    safety margin. They remain unavailable for the current calculation but
    become reusable after the next `ComInc`.
  - Added pool provenance: allocated/used parameter, load and tap counts;
    discovered/created/pending/admission/retired counts.
- `pf/plant.py`
  - Replaced destructive startup purge and full-horizon allocation with
    persistent discovery and minimum-capacity assurance.
  - `advance()` now performs zero-time admission before the real interval
    advance, without changing the plant clock or the shared controller loop.
- `experiments/run_rms_phase6_replay.py` and
  `experiments/run_rms_openloop_uy.py`
  - Removed `ceil(duration/dt)+1` startup allocation; each key now starts with
    one minimum slot and grows only when its historical high-water mark is
    exceeded.
- `tests/pf/test_screening_event_pool.py`
  - Added persistent growth, admission, retirement, discovery, unmanaged-event
    hygiene, and tap-pool coverage.
- Added live diagnostic probes for cyclic reuse, cross-ComInc reuse, and
  admission barriers under `pf/probe_event_*.py`.

## Verification

- Focused persistent-pool tests: **7 passed**.
- Broader PF/RMS replay regression: **28 passed, 1 pre-existing failure**.
  The unrelated failure is the temporary `der_q_capability_override_pu=1.0`
  diagnostic override differing from the reference test's expected `None`.
- Headless profiles-on 40 s smoke run, run
  `0041_2026-07-22_182057`, with `g_w_dso_oltc=150`:
  - Gate E: PASS.
  - Folder: 373 managed events.
  - Pool: 261/261 parameter, 106/106 load, 6/6 tap.
  - `created=373`, `admission_executes=4`, `pending_admission=0`,
    `retired=373`.
- A fresh Python/PowerFactory process then reported:
  `discovered=373`, `created=0`, all used cursors zero, folder still 373.
  This establishes persistence across process boundaries.

## Assumptions and controlled scope

- Controller implementation is unchanged. The TSO/DSO MIQPs still operate
  only on measurements and cached sensitivities; the change is confined to
  delivery of actuator/profile events to the RMS plant.
- Controlled outputs remain zone/interface reactive-power flows and bus
  voltages. Actuators remain DER Q, AVR references, OLTCs, and shunts; profiles
  use `EvtLod` and WECC `Pref_in` events.
- The current comparison uses profiles, `rms_profile_settle_s=0`, and the
  temporary non-physical DER capability override of +/-1 pu of park rating.

## Risks / unresolved points

- First-time work still scales with the longest continuous horizon because an
  event cannot be reused inside that calculation. A first 18,000 s run may
  grow to roughly 160,000-170,000 stored events; later equal/shorter runs reuse
  that high-water pool.
- PowerFactory event-folder lookup, project size, and result export may become
  the dominant cost. At 10 ms, 18,000 s also produces about 1.8 million result
  rows before trajectory export.
- A killed process may leave a small number of non-inert scheduled slots. The
  next pool preparation resets every retained slot before `ComInc`, preserving
  correctness at the cost of a full pool scan.
- Native PowerFactory time characteristics or a PF-side signal bus remain the
  preferred later optimization for reducing the absolute event count.
