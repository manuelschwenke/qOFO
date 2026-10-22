# 2026-07-20 — Phase 6 Gate E baseline replay and Q(V) blocker

**Timestamp:** 2026-07-20 17:37 CEST
**Reason:** complete Phase 6 steps 4–5 after the plant/runner seam, using the
authoritative `experiments/run_multi_system_ofo.py` configuration including
the tertiary shunts, and produce traceable results and figures.

## Implementation

- Added `pf/replay.py`: controller-free snapshot export/synchronisation,
  solver-option application, variation activation and a one-shot
  `PowerFactoryReplayFactory` that supplies the runner's external plant.
- Added `experiments/helpers/rms_replay.py`: static endpoint extraction,
  RMS trajectory normalisation, 12-interface-Q and three-zone-voltage
  comparison, bounded per-interval settling analysis, and PNG/PDF overlays.
- Added `experiments/run_rms_phase6_replay.py`: reproducible static and RMS
  orchestration, CSV/JSON/Markdown provenance, figures and non-zero exit on
  a failed or scientifically blocked Gate E verdict.
- The Gate E configuration is obtained from
  `experiments.run_multi_system_ofo.make_config`; only the experimental
  envelope is changed (900 s, 20 s STS, 180 s TSO, fixed injections, no
  contingencies/noise/live plots, no quasi-static reachability guard).
  Consequently SBX-H coordination, `install_tso_tertiary_shunts=True`,
  `shunt_dispatch="integrator"`, MSC/MSR levels/steps and all controller
  weights remain authoritative.
- The runner now advances external RMS time even on intervals with no new
  MIQP action and propagates external-plant exceptions instead of entering
  pandapower recovery paths. `PowerFactoryPlant.read_y()` refreshes both
  line ends, P/Q, losses and loading so no stale pandapower tie-line values
  enter a controller measurement.
- `PowerFactoryPlant.der_qv_local_control_equivalent=False` is explicit.
  Reports separate the observed settling verdict from Gate E validation and
  mark current artifacts `BLOCKED_DER_QV_MISMATCH`.
- `pf/wecc_introspect.py` now reads scalar parameter names from the DSL block
  definition rather than treating the numeric `params` vector as names.
  Future WECC rebuilds explicitly set `QFlag=0` as well as `PfFlag=0`;
  `VFlag` is immaterial in constant-Q mode.

## Assumptions and constraints

- Exogenous load and active-power injections are fixed over 900 s; the
  experiment isolates plant dynamics and the closed-loop hierarchy.
- Measurement noise is disabled. Each controller run sees only its own
  plant's measurements and its cached sensitivity/model; neither controller
  receives PowerFactory equations or internal states.
- Actuators: DER reactive-power references, synchronous-machine AVR
  references, 2W/3W OLTC taps, and tertiary MSC/MSR steps dispatched by the
  integrator.
- Controlled outputs: all 12 TS–STS interface reactive-power flows and mean
  TN-PQ-bus voltage in each of the three TSO zones. The settling band is 2%
  of the interval step with absolute floors of 1 Mvar and 0.001 pu.
- The quasi-static reachability guard is disabled because it would solve a
  second plant behind the RMS adapter. Profiles/contingencies are not yet
  supported by the external-plant seam and are disabled explicitly.

## 900 s diagnostic baseline

Run directory:
`results/rms_phase6_replay/0005_2026-07-20_163715`

- 45 static records, 45 RMS records, PowerFactory final time 900 s.
- Interface Q settling: 540/540 windows settled; maximum 6.3817 s,
  95th percentile 2.4767 s.
- Zone-voltage settling: 132/135 windows settled; three failures occur only
  in the 20–40 s window (one per zone). Maximum is the censored 20 s;
  95th percentile 11.7417 s.
- Static-vs-RMS endpoint errors: interface Q RMSE 16.2185 Mvar,
  MAE 11.3153 Mvar, maximum 49.6185 Mvar; zone voltage RMSE 0.006348 pu,
  MAE 0.004629 pu, maximum 0.020068 pu.
- Six G 01 voltage-reference writes were recorded and skipped because the
  adopted 10 GVA external-equivalent template has no AVR.
- TSO tertiary shunt and tap decisions were identical between static and RMS
  controller runs at every TSO firing; the divergence is not caused by a
  different discrete branch.

Artifacts include:

- `gate_e_summary.md` / `gate_e_summary.json`;
- complete endpoint, trajectory and settling CSV tables;
- `figures/interface_q_static_vs_rms.{png,pdf}`;
- `figures/zone_voltage_static_vs_rms.{png,pdf}`;
- static/RMS record pickles and snapshot/synchronisation reports.

## Diagnosis: the baseline does not compare equivalent actuator laws

This is an established model fact, not a timescale hypothesis:

1. Initial controller input measurements match to approximately `1e-6 pu`
   across all three zone-voltage means, so snapshot synchronisation is not
   the source of the closed-loop divergence.
2. At the first DSO dispatch the aggregate raw `q_set` commands were
   72.1372, 45.1205, 46.4804 and 55.3973 Mvar. PowerFactory reached
   72.1356, 45.1161, 46.4808 and 55.3975 Mvar: `REEC_D.Qext` tracks the
   requested constant-Q command correctly.
3. The static plant's re-anchored Q(V) fixed point instead realised
   17.2917, -9.3163, 3.5418 and 15.3041 Mvar. Its law is

   `Q = clip(q_set - (S_n / slope) * deadband(V - V_anchor), capability)`

   with `V_anchor` rewritten to the measured DER-bus voltage at each OFO
   dispatch. PowerFactory currently has no corresponding plant-side layer.
4. The PowerFactory 2025 SP4 Station Controller technical reference states
   that `ElmStactrl` is used in load-flow calculations. Its Q(V)
   characteristic therefore cannot supply the missing continuous RMS law.
   The installed WECC reference and EPRI generic-model guide confirm that
   `(PfFlag,QFlag)=(0,0)` is constant local Q; REEC_D's native coordinated
   Q/V mode is a dynamic cascaded controller and is not, without an exact
   equivalence proof, the same re-anchored algebraic characteristic.

Therefore the generated overlays and endpoint errors are useful diagnostic
evidence but are not the dissertation headline validation figure. The
artifact summary now states this explicitly.

## Proposed layered revision (not yet implemented)

Keep three responsibilities separate:

1. **OFO/controller layer:** unchanged `q_set_mvar` command and cached
   sensitivities; it remains plant-agnostic.
2. **Plant-side Q(V) pre-controller:** one small, testable block per DER that
   stores `q_set`, the dispatch-time `V_anchor`, slope/deadband and capability
   limits and computes the exact reference law above from local measured V.
3. **Dynamic converter layer:** the existing REGC_C + constant-Q REEC_D
   composite tracks the pre-controller's `Qref` with its physical dynamics.

This matches the fallback already anticipated in Phase 5 and avoids
embedding controller intelligence in the plant adapter. Because adding and
wiring the DSL/pre-controller is a material dynamic-model architecture
change, implementation waits for explicit review/approval.

## Verification and open risks

- Replay-analysis tests: **8 passed**.
- Focused Jacobian/Q(V) tests: **14 passed**.
- Broader relevant Phase 1–6 regression: **77 passed, 6 skipped**.
- The 900 s replay completed technically and produced all artifacts, but its
  Gate E validation verdict is **blocked**, not passed or failed solely on
  the three voltage settling windows.
- Open: exact DSL signal wiring into the WECC composite, RMS event updates of
  `q_set` and `V_anchor`, and capability clipping must be proven on one park
  before full rebuild/rollout.
- Open: decide whether G 01 should remain an explicitly skipped actuator or
  be excluded from the TSO actuator vector for RMS studies.
