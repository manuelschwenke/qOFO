# 2026-07-20 — Phase 6: Plant abstraction + PowerFactoryPlant

**What.** Phase 6 steps 1–3 of the RMS build plan implemented: the
`Plant` protocol, the bit-for-bit static reference plant (Gate E,
first checkbox), and the RMS co-simulation plant.

## 6a — `core/plant.py` + regression (Gate E checkbox 1 GREEN)

- **`Plant` protocol**: `apply_u(writes)` / `advance(T)` / `read_y()`.
  `read_y()` returns a *pandapower measurement image* (a net whose
  `res_*` tables hold the plant state at t_k⁻), so `core.measurement`
  and every controller stay byte-for-byte unaware which plant they face —
  consistent with the project rule that controllers only ever see
  measurements and cached sensitivities, never the plant model.
- **`ActuatorWrites`**: one dispatch, keyed by pandapower indices
  (`der_q_set_mvar`, `gen_v_pu`, `tap_2w`, `tap_3w`, `shunt_step`) —
  the shared namespace both plants understand (PF `loc_name` embeds the
  same indices, docs/pf_naming.md).
- **`PandapowerStaticPlant`**: today's behaviour behind the interface.
  `advance` = the exact main-loop power flow of
  `experiments/runners/multi_tso_dso.py` (`run_control=True`,
  `calculate_voltage_angles=True`, `max_iteration=50`, `max_iter=300`,
  `distributed_slack`, `enforce_q_lims`); DER writes use the w-shift
  recipe (`write_der_q_set`: reanchor `qv_vref_anchor_pu` from the last
  `res_bus`, then set `q_set_mvar`) copied from
  `experiments/helpers/plant_io.py`.
- **Regression test `tests/test_plant_static.py` (5 passed, 18.7 s)**:
  a dispatch touching every actuator class applied once through the
  legacy `plant_io` helpers + runner `pp.runpp`, once through
  `ActuatorWrites`/`apply_u`/`advance` — element tables and every
  result table compared with `check_exact=True`. Bit-identical.

## 6b — `pf/plant.py` (`PowerFactoryPlant`)

- **apply_u** (paused-state event scheduling, all handles verified by
  `pf/probe_tap_avr.py` the same day):
  - DER Q → `EvtParam` on `REEC_D.Qext` (pu of park S_n),
  - machine V-ref → `EvtParam` on the AVR DSL's `usetp` signal
    (`avr_IEEET1`; **G 01 has no AVR** — `on_missing_avr="raise"|"skip"`),
  - OLTC taps → `EvtTap` (relative `ntap`), one ±1 event per step at
    `t + 5 s, t + 10 s, …` (mechanical delay, sequential multi-tap per
    plan Phase 5 step 3),
  - MSC/MSR steps → `EvtTap` at t+ε (breaker, no delay).
  - Discrete state (taps, shunt steps) lives in a **shadow store**
    initialised from PF input data (= snapshot, per Gate C): simulation
    events do not update input attributes.
- **advance(T)** → `ComSim.tstop = t + T; Execute()`.
- **read_y** → paused-state attribute reads into the mirror net, using
  the exact parity-validated mapping (pf/pf_parity.py): `m:u`/`m:phiu` →
  `res_bus`, `m:I:bus1` → `res_line.i_from_ka`, `m:P/Q:bushv` →
  `res_trafo3w`, `m:P/Q:bus1` → `res_sgen`/`res_gen` (machine results
  are plant totals). Mirror input columns (`tap_pos`, `step`, `vm_pu`)
  are kept in sync by `apply_u`.
- **harvest_trajectories(since_s)** → per-dispatch-chunk `{label: (t, y)}`
  from the shared `ElmRes` (monitors = the screening controlled-output
  set).
- **Event hygiene**: constructor purge with the fixed
  `ScreeningContext.purge_events()` (ResetCalculation + verified-empty,
  see the event-accumulation log); `advance` does **not** delete
  mid-run (deletion silently fails while the sim is active, and past
  events cannot re-fire within one continuous run).
- Smoke test: `pf/plant_smoke.py` — **5/5 PASS** on the live model
  (50 s chunked co-sim, 20+20+10 s advances): t0 measurement image vs
  snapshot solution max |dV| = 1.545e-5 pu (= the Gate-C parity level);
  +20 Mvar DER dispatch lands exactly (+20.00 Mvar at the park);
  +1 coupler tap moves q_STS 21.70 → 13.93 Mvar with the shadow tap
  advancing −1 → 0; hold-interval drift 6.7e-6 pu; trajectory harvest
  60 signals / 201 samples over the last dispatch chunk.

## Documented modelling divergences (static vs RMS plant)

1. Between dispatches the RMS DERs hold constant Q (REEC_D Q-control,
   no droop); the static plant's `QVLocalLoop` re-droops around the
   reanchored V_ref every power flow. The RMS side has no autonomous
   Q(V) response between OFO dispatches.
2. `G 01` (10 GVA equivalent) V-ref is not actuatable in RMS (template
   has no AVR/GOV blocks on it) — closed-loop replays must either skip
   its V commands (`on_missing_avr="skip"`, recorded in
   `skipped_writes`) or exclude it from the TSO's V actuator set.

## 6c — Runner refactor: `run_multi_tso_dso(plant_factory=...)` (same day)

The main loop of `experiments/runners/multi_tso_dso.py` now acts on the
plant exclusively through the `Plant` interface (user-approved option (a);
STS cadence confirmed **20 s**, `dso_period_s = 20` — plan doc updated,
the 10 s wording was stale):

- **Seam**: after the init phases the runner constructs
  `PandapowerStaticPlant(net, ...)` (default) or calls the new
  `plant_factory(net)`. A non-static plant keeps `net` as its
  mirror/measurement image (`PowerFactoryPlant` gained `mirror_net=` for
  this), so the runner's direct `net` *reads* (records, sensitivities,
  diagnostics) stay valid for both — only writes, plant response, and
  measurement refresh were rerouted.
- **Rerouted sites**: post-profile PF -> `plant.advance(dt_s)` +
  `read_y()`; end-of-step PF -> `plant.advance(0.0)` + `read_y()`;
  `measure_zone_tso/dso`, `measure_central` read `plant.read_y()`;
  TSO/DSO/central/shunt-commit writes -> `plant.apply_u(...)` via new
  u-vector adapters in core/plant.py (`writes_from_zone_tso`,
  `writes_from_dso`, `writes_from_central`, `shunt_steps_for_buses` —
  exact plant_io slicing; plant_io helpers no longer called by this
  runner).
- **Kept direct (static-only recovery paths)**: post-contingency retry
  ladder, OLTC tap-rate-limiter re-runs, end-of-step flat-start retry.
  A pre-loop guard raises `NotImplementedError` when a non-static plant
  is combined with contingencies, the tap-rate limiter, or time-series
  profiles (PF-side load events not wired yet).
- **Regression (Gate E checkbox 1, full-runner form): PASS.**
  `tests/runner_refactor_regression.py`: an 18-step default-config run
  (360 s, two 180 s TSO firings, noise on, seeded) frozen with the
  pre-refactor code, re-run post-refactor — every
  `MultiTSOIterationRecord` bit-identical (only wall-clock solve-time
  fields excluded; a pre-refactor self-check confirmed the run is
  deterministic, so the comparison is meaningful).

## Remaining for Gate E

- PF-side exogenous-disturbance events (profiles per dispatch step) for
  the RMS plant, then the closed-loop replay driver (plan step 4): one
  CIGRE scenario window, 20 s STS / 180 s TS, overlay y(t_k⁻) on the
  quasi-static trajectory.
- Settling statistics per dispatch interval (plan step 5).

## Unrelated pre-existing red (flagged, not today's scope)

~50 unit tests fail with one shared root cause:
`sensitivity/jacobian.py:220` — `_ppc['internal']['J'] is None` after the
(uncommitted, 2026-07-17) warm-start guard; the 1e-8 kick fails to force a
stored Jacobian on *small* test networks (full-network runs are fine,
which is why experiments and the regression above pass). Task chip filed.

*Files*: `core/plant.py`, `tests/test_plant_static.py`, `pf/plant.py`,
`pf/plant_smoke.py`, `pf/probe_tap_avr.py`; catalogue + event fixes in
`pf/screening.py`.
