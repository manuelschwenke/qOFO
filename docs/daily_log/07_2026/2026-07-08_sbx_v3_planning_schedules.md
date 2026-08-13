# 2026-07-08 — SBX v3: planning-anchored contract-voltage schedules

**Task:** Manuel approved the v3 amendment: `v_std` becomes an hourly
schedule from a planning power flow (DACF/IDCF emulation) instead of
the settled-state snapshot.

**Changed:**

- `sbx/contract.py`: optional `CorridorContract.v_std_schedule`
  (ordered `(t_from_s, v_std_a, v_std_b)` intervals, total coverage
  from t = 0, constant fields = first interval), `v_std_at(t_s)`
  lookup, `q_std_mvar(..., time_s=)` (REQUIRED for schedule-bearing
  contracts — no silent constant fallback),
  `build_default_contract(..., v_std_schedule=)` (planning replaces the
  snapshot; the snapshot branch is untouched).
- `sbx/scheduler.py`: per-corridor ACTIVE contract voltages
  (`_CorridorState.v_std_[ab]_act`) resolved at every boundary
  (Step 1) and used for q_std, capability sensitivities, Step-4
  references/dv/invariant; `run_cycle(..., time_s=)` and
  `initial_references(time_s=)` (time required when any contract has a
  schedule). Surplus/unwind/settlement semantics untouched.
- `sbx/adapter.py`: `freeze_time_s` (scenario time of the construction
  tick, cycle-clock origin) and `v_std_schedules` parameters; per-
  boundary scenario time passed to the scheduler.
- `configs/multi_tso_config.py`: `sbx_v_std_schedule_path`; runner
  loads the JSON and passes schedules + freeze time to the adapter.
- `experiments/017_SBX_PLANNING.py` (new): hourly planning pre-pass on
  the same net/profiles/zonal dispatch, WITHOUT contingencies and
  closed-loop controllers (gens at the zone voltage schedule, taps and
  shunts at build defaults — deliberately a simplified planning model);
  modes perfect | persistence (t − 24 h) | noise (σ, seeded); outputs
  schedule JSON + per-corridor plot. Validated: 7 hourly PFs converge,
  schedule written.
- `experiments/014_SBX_SINGLE_DEMO.py`: `--schedule <json>` flag.
- `tests/sbx/test_contract.py`:
  `test_v3_schedule_lookup_and_time_requirement` (lookup, q_std jump at
  an interval boundary, time requirement, validation guards).
- Test suite repairs after Manuel's SBXConfig default changes
  (k_sched 5→2, quantum rate 10→30, n_need 5→1): the tests asserting
  absolute Mvar·h/EUR/persistence numbers now pin their reference
  config EXPLICITLY (plan-v2 §5 values) instead of relying on defaults.

**Test state:** `pytest tests/sbx` → 58 passed. Closed-loop validation
(014 + perfect schedule, 150 min, spanning an hourly v_std switch)
launched.

**Design note:** with a planning schedule the standing
`q_meas − q_std` offset becomes the PLANNING ERROR (crude planning
model + closed-loop reality) — a feature, not a calibration artefact;
the tier-1 band should now be sized against the forecast quality (017's
persistence/noise modes produce exactly that data).

## Addendum (same day): F10 fix — scheduled operating point in the plan

`plan_operating_point()` added to 017 (STATCOM temp-PV-gen + machine
2W OLTC at the zone schedule, coupler 3W OLTC at the DSO target, per
hour, taps persistent across hours); `--no-oltc-schedule` keeps the
crude plan for comparison (`_crude` tag). Closed-loop verification
(asym_z3, 90 min, contracts from t = 0): plan-reality gap on (1,2)
165 → ≈ 13 Mvar, on (1,3) sub-Mvar; the plant visibly converges onto
the plan and returns toward it after the stress. Also this session:
`--schedule` in 014 now sets `sbx_warmup_s = 0` (the warmup existed
only for the snapshot; with a schedule the mechanism arms at the first
TSO tick — fixes the lingering Figure-6 placeholder Manuel reported).
ORPF discussion recorded in STATUS (future `--mode orpf`).
