# 2026-07-19 — RMS Phase 4: full-model synchronizer started

- **Timestamp:** 2026-07-19 01:03 CEST
> **Status update 2026-07-19 19:58 CEST:** the layered architecture was
> approved and implemented; the incomplete full artefact was deleted and
> rebuilt; both Gate-C snapshots now pass. This start log is superseded by
> `docs/daily_log/07_2026/2026-07-19_rms_phase4_gate_c_closed.md`.
- **Reason:** continue the RMS build plan with the four 110 kV DSO underlays
  after closing Gate B.

## Code changed

`pf/pf_sync.py` was extended with:

- deterministic routing of buses, lines, loads, static generators,
  controllers, shunts, and coupling transformers to one `ElmNet` per
  `DSO_1` through `DSO_4`;
- line rated voltage derived and validated from both endpoint buses (345 kV
  for TN/TN_AUX and 110 kV for DSO lines);
- generic static-generator creation plus one constant-Q `ElmStactrl` write
  handle per TSO/DSO park;
- `ElmTr3`/`TypTr3` creation for all twelve 345/110/20 kV couplers, including
  pairwise positive-sequence impedances, phase shifts, HV OLTC parameters,
  and snapshot tap positions;
- `ElmShnt` creation for the four MSC and four MSR banks;
- direct recursive Network Data discovery so disconnected objects and empty
  DSO grids remain discoverable after an interrupted sync.

The direct-discovery change was prompted by a live fail-fast/resume test:
`GetCalcRelevantObjects` omits an empty or disconnected new DSO and is not a
suitable idempotency lookup for construction code.

## Live PowerFactory facts established

The installed PF 2025 SP4 technical references and disposable API probes
established the following mappings:

- `TypTr3.r1pu_h/x1pu_h`: HV-MV pair;
- `TypTr3.r1pu_m/x1pu_m`: MV-LV pair;
- `TypTr3.r1pu_l/x1pu_l`: LV-HV pair;
- HV OLTC fields: `n3tmn_h`, `n3tmx_h`, `n3tp0_h`, `du3tp_h`, with actual
  position `ElmTr3.n3tap_h`;
- `TypTr3` has no `frnom` attribute; the guarded assignment was removed;
- `ElmShnt.shtype=2`: pure capacitor; `shtype=1`: reactor;
- shunt `mode_inp='Q'` exposes `qcapn`/`qrean` as rated per-step Mvar;
- `ElmStactrl.i_ctrl=1`, `qu_char=0`: constant-Q station control.

## Architecture finding (decision pending)

The first implementation treated `wind_replace` and `full` as alternative,
self-contained variations. Live PF showed that object names remain reserved
across inactive variations. Recreating `AUX_TN_*`, `WP_TSO_*`, and their
controllers in `full` therefore produces decorated names and makes the
deterministic naming invariant false.

The proposed correction is a layered variation state:

1. base project;
2. activate `wind_replace`;
3. activate `full` on top of it.

Under this design `full` contains only the four DSO underlays and the full
snapshot's dispatch, tap, and Q-setpoint overrides. Deactivating `full` while
leaving `wind_replace` active restores the validated wind state.

This is a material variation-architecture change and is awaiting user
confirmation before implementation, as required by the project instructions.

## Cleanup and current project state

An initial full sync stopped on the unsupported `TypTr3.frnom` schema guard.
A resume attempt exposed the name-reservation issue. All verified second-run
duplicates (decorated DSO grids/elements, base duplicates, external cubicles,
and duplicate types) were removed using exact grid-name and class-count
assertions. They are reproducible from the snapshot but are not recoverable by
PowerFactory undo across the closed process.

The incomplete `full` variation is now inactive. The active project was
restored to `wind_replace` t0 and revalidated:

- max `|d vm| = 3.118e-8 pu`;
- max `|d va| = 5.407e-6 deg`;
- Gate B: PASS.

## Assumptions, constraints, actuators, and controlled outputs

- **Assumption:** full-model parity should reuse the exact Gate-B wind plant
  rather than create a second logically identical set of TN wind objects.
- **Constraint:** deterministic `loc_name` values must remain unique and
  stable across all active variation combinations; build operations must be
  interruption-safe and idempotent.
- **Actuators prepared:** 12 HV-side 3W OLTCs, 8 tertiary MSC/MSR banks, 44
  static-generator Q controllers, and the retained synchronous-machine AVR
  setpoints. Automatic actions remain off for Gate C.
- **Controlled outputs for Gate C:** every physical TN/DSO bus voltage and
  angle, every TN/DSO line flow, all three winding flows of every coupler,
  interface P/Q, and source/load injections.

## Risks / unresolved points

- The layered-variation design must be approved and then implemented in the
  sync and parity activation logic before another full write.
- Gate C has not been run. Three-winding impedance/base conversion and phase
  convention remain hypotheses until full t0 and peak-residual parity pass.
- The current `sync_full` implementation still assumes the superseded
  self-contained state. The `--phase full` CLI branch is therefore explicitly
  fail-closed until the layered-variation decision is applied.
