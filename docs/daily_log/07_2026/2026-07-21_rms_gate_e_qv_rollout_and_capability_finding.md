# 2026-07-21 — Gate E: Q(V) rollout completed; endpoint gap traced to a Q-capability mismatch

## What was changed

1. **Q(V) layer rolled out to all 44 parks** (`pf/wecc_apply.py`), verified
   live after the replay: 44/44 WECC composites on `Frame WECC PV qOFO`,
   each with a `QVPRE` `ElmDsl` bound into the Plant Control slot.
2. **`pf/plant.py`** — `der_qv_local_control_equivalent` is now set per
   instance in `_resolve_handles` (`der_q_mode == "qv_precontroller"`).
   It was only ever a **class attribute defaulting to `False`**, so the Gate-E
   verdict read `False` no matter what the model did and always reported
   `BLOCKED_DER_QV_MISMATCH`. The class default is kept as the conservative
   fallback.
3. **`experiments/run_rms_phase6_replay.py`** — the "RMS DERs hold
   `REEC_D.Qext` constant" validity warning was emitted unconditionally; it is
   now selected by `qv_equivalent` so the report cannot contradict the model.
4. **`pf/screening.py` / `pf/plant.py`** — `monitored_outputs(app,
   include_der=True)` additionally records each park's `m:Q:bus1` and `m:u`.
   Rationale: an interface-Q error alone cannot distinguish a capability clip
   from droop amplification of a voltage difference. The Gate-E plant opts in.

## Replay result (run `0008_2026-07-21_103859`, 900 s)

| metric | 0005 (no Q(V)) | 0008 (Q(V)) |
|---|---|---|
| zone-voltage intervals unsettled | 3 | **0** |
| max settling time | 20 s | **8.78 s** |
| interface_q RMSE / max_abs | 16.22 / 49.62 | 19.07 / 53.30 |
| zone_voltage RMSE / max_abs | 0.00635 / 0.02007 | 0.01112 / 0.02675 |

Settling improved; **endpoint errors got worse**. Error-vs-time is not a
constant offset: mean \|err\| on `interface_q` starts at 11.2 Mvar (RMS init
transient), decays to 0.57 Mvar by t=120 s, then grows monotonically to
21.3 Mvar at t=900 s. The early decay is *deeper* than in 0005 (0.57 vs
0.72), i.e. the droop law itself is right.

## Root cause (established, not hypothesis)

`controller/der_qv_local_loop.py::_qv_capability` clips the static plant's
park Q to the operating diagram. For `op_diagram == "STATCOM"` it uses the
circular characteristic `q_pu = sqrt(1 − (P/S_n)²)`. In this scenario the
STATCOM-diagram parks are dimensioned with **`p_mw == sn_mva` exactly**, so

    q_min = q_max = 0.0   →   np.clip(q_target, 0.0, 0.0) == 0.0

**16 of 44 parks** (4 `TSO-WP` + 12 `DSO-COUPLING-WP`) therefore contribute
**exactly zero** reactive power in the static plant, whatever the OFO commands.
The remaining 28 (`VDE-AR-N-4120-v2`) clip at −0.33/+0.41·S_n.

On the RMS side `REEC_D` declares **no `Qmax`/`Qmin` at all** — the only limit
is `Imax = 1.3` pu with `PqFlag = 0` (Q priority). Those 16 parks are
effectively unconstrained in Q.

This explains every observation:

- **Why the correct law made endpoints worse.** Before the rollout the RMS
  parks held `Qext` at their load-flow value (≈0), which *accidentally*
  matched the static plant's hard clip to 0. Giving them the true droop let
  them move, while their static counterparts still cannot move at all.
- **The one-sided sign** of the interface errors (RMS consistently less
  reactive absorption at `qSTS_t0/t3/t6`).
- **The monotonic growth**: the droop keeps pushing the free RMS parks as the
  open-loop voltages drift apart; with `Kdroop = 1/0.06 ≈ 16.67`, the observed
  0.0267 pu zone-voltage error maps to ≈0.445 pu·S_n ≈ 49 Mvar — the observed
  interface-error magnitude.
- **The static run's own saturation** (`DSO_3: Q_set=−75.16, Q_act=+9.85,
  |err|=85.01 Mvar`): the DSO controller is commanding parks that physically
  cannot deliver.

## Open question for the supervisor (architectural — not changed unilaterally)

An actuator labelled `STATCOM` that supplies zero reactive power is
self-contradictory: a STATCOM is a pure-Q device. Three candidate resolutions,
each with different reach:

1. **Oversize the converter** (`sn_mva = k · p_mw`, e.g. k = 1.1) — keeps the
   circular diagram, restores headroom, changes park ratings.
2. **Give `STATCOM` a P-independent range** (±S_n) — matches the device name
   and the intent of a `STATCOM` label, changes `_qv_capability`.
3. **Relabel** the full-P wind parks to a diagram that has headroom at rated P.

Note the blast radius: `_qv_capability` also backs
`core.actuator_bounds.ActuatorBounds._compute_single_der_q_capability`, so any
change moves the **OFO's own bound computation**, and therefore every published
CIGRE/SBX result — not just Gate E. Do not treat this as a Gate-E-local fix.

## Resolution (same day, user decision)

Supervisor chose **option 3 (relabel)** over option 2, after being shown that
relabelling *tightens* capability at every operating point below
P/S_n ≈ 0.93 (circle ±0.866·S_n at P/S_n = 0.5 vs the box −0.33/+0.41) and
therefore shifts profile-on CIGRE/SBX results, whereas the ±S_n fix would have
matched the builders' documented intent and moved only the P≈S_n corner.
Recorded because the code comments still read as if the circle were intended.

Implemented:

1. **Relabel** — `op_diagram` `'STATCOM' → 'VDE-AR-N-4120-v2'` for the 4 TSO
   wind parks (`network/ieee39/scenarios/wind_replace.py`) and the 12 HV
   coupling parks (`network/ieee39/hv_networks.py`), with the stale
   "full Q-circle from S_n" / "Q headroom from profile < 1" comments replaced.
2. **RMS-side limits mirrored** — `QVPRE` gained `qmin`/`qmax` params and an
   output limiter, so both plants clip with the same box.  `REEC_D` declares
   no Qmax/Qmin, so without this the RMS park was bounded only by
   `Imax = 1.3` pu.  DSL has no min/max operator; both limits use the same
   arithmetic identity as the deadband (`max(a,b) = (a+b+|a-b|)/2`).
3. **`ensure_qvpre_blockdef`** added to `pf/wecc_apply.py`: the BlkDef had been
   authored by a throwaway script and existed only inside the PF project.  It
   is now versioned in the repo and reproducible.  All 44 elements were
   recreated (an `ElmDsl` predating its final BlkDef keeps a dead param table).
4. **Role classification decoupled from the capability model**
   (`export/dynamic_snapshot.py`).  `_sgen_role` keyed `DSO-COUPLING-WP` off
   `op_diagram == "STATCOM"`, so the relabel silently reclassified all 12
   coupling parks and renamed their PF objects `WPC_* → DER_*`, which broke
   the plant's handle resolution outright
   (`sgen[11] 'DER_DSO_1_s11_b46': WECC composite missing`).  Now keyed off the
   builder's deliberate name marker `WP_STATCOM_HV`, with the old diagram test
   retained as a fallback so pre-2026-07-21 snapshots still classify.

## Result (run `0011_2026-07-21_114549`, 900 s)

| metric | 0005 no Q(V) | 0008 Q(V), no limits | **0011 Q(V) + limits** |
|---|---|---|---|
| interface_q RMSE / max_abs | 16.22 / 49.62 | 19.07 / 53.30 | **8.93 / 29.70** |
| zone_voltage RMSE / max_abs | 0.00635 / 0.02007 | 0.01112 / 0.02675 | 0.01518 / 0.03531 |
| unsettled intervals (q / V) | — / 3 | 0 / 0 | 0 / 0 |
| max settling | 20 s | 8.78 s | 10.48 s |

The **runaway divergence is gone**: mean \|err\| on `interface_q` now goes
10.36 → 5.21 (t=100 s) → 9.54 Mvar (t=900 s), i.e. it plateaus, where 0008
climbed monotonically 11.2 → 0.57 → 21.3.  The static plant's DSO saturation
also eased at short horizon (`DSO_3` |err| 85.01 → 2.49 Mvar over 120 s).

**Caveat on the table:** 0008 and 0011 are not the same experiment.  The
relabel changed the *static plant's own* capability, hence the OFO's bounds and
the whole reference trajectory.  Only the within-run static-vs-RMS agreement is
a like-for-like quantity; the columns show how the setup evolved, not a
controlled comparison.

**The verdict string is weaker than it looks.** `gate_e_validation_verdict`
is `gate_ok and qv_equivalent` — settling inside the window AND both plants
running the same actuator law.  It does **not** test endpoint agreement.
"PASS" therefore means *the comparison is now valid*, not *the trajectories
match*.  A residual gap remains: interface-Q RMSE 8.93 Mvar and zone-voltage
RMSE 0.0152 pu (max 0.0353), and the zone-voltage error is larger than in 0008.
At 900 s the static DSO Q-tracking is still heavily saturated
(`DSO_2: Q_set=-119.73, Q_act=-21.03`), so both plants operate near their
limits, where small differences change which parks clip.

## Instrumentation bug found while attributing the residual

`monitored_outputs(app, include_der=True)` was registered in
`PowerFactoryPlant.__init__`, but `harvest_trajectories` **re-derived the list
with default arguments**, so the `qDER_`/`uDER_` labels were never in the
iteration list and the export came back empty.  The monitor list is now stored
on the instance and reused.  Also, `m:u` does not exist on an `ElmGenstat` in
RMS (`FindColumn` → −1, while `m:Q:bus1` → a real column); the park voltage is
now read from its `ElmTerm`.  **`FindColumn` only returns valid indices after
`res.Load()`** — calling it earlier raises `RuntimeError: method call failed`.

## Phantom generation: 12 duplicate ElmGenstat objects (found + fixed)

The run that failed on the role rename had already *synced* before it raised,
creating 12 `DER_DSO_*` static generators alongside the correct `WPC_DSO_*`
ones -- same buses, all in service, **480 MW / 78 Mvar of generation
pandapower does not have**.  `_sync_static_generators` created missing
genstats but never deleted stale ones; terminals, lines, trafos and loads all
prune, static generators were the gap.

Fix: `delete_stale_sgens` in `pf/pf_sync.py`, called from both
`sync_wind_replace` and `sync_full`.  The rule is deliberately narrow -- a
genstat is stale only if it is absent from the snapshot **and shares a
terminal with one that is present**.  The obvious rule ("any genstat missing
from the snapshot") would have deleted all 40 DSO parks during the
wind_replace phase, because `_network_all` walks inactive islands too.
Dry-run confirmed exactly 12 targets, 0 creations; Gate C parity after
deletion 1.475e-05.

## Initialisation: verified correct, hypothesis refuted

A flat 60 s RMS run with a **purged event folder**:

| stage | max abs dV vs pandapower |
|---|---|
| parity load flow | 0.0000148 |
| ComInc, t = 0 | 0.0000148 |
| 60 s flat, no dispatch | 0.0000148 |

The RMS initialises exactly on the pandapower operating point and holds it.
The proposed settle-and-feed-back handshake is therefore unnecessary.

**A first attempt at this measurement reported 0.032 pu drift and was wrong**:
the killed 600 s replay had left **1122 stale events** in `p_event`, which PF
replays on every subsequent `ComInc`, so the "flat" run was executing the dead
run's dispatch sequence (`c:qset` read 0.1319 against an anchored 0.0).  Ad-hoc
probe scripts do not call `purge_events()`; the production path does.  Two
process rules follow: **always purge before probing**, and **never connect to
PF while a run is in flight** (doing so killed the run with "User session has
been terminated").

## What the apparent initial offset actually was

The static run's first *record* is at t = 20 s, after its first dispatch; the
RMS trace starts at t = 0.  The figure therefore compared a pre-dispatch RMS
state against a post-dispatch static one.  Both genuinely start identically.

## Discrete actuator divergence is the dominant residual

`analysis/gate_e_diagnostics.py` (new) plots every tap/shunt trajectory and
per-bus voltages, and writes `csv/actuator_divergence.csv`.

Run 0012 (with orphans): **15/27** actuators ended different, DSO_1 couplers
up to 9 steps apart, RMS taps marching monotonically to -8 without converging.
DSO_1 had both the worst tap divergence and the worst voltage gap (~0.04 pu).

Run 0015 (clean, 600 s): **7/27**, all DSO couplers, max 2 steps, no runaway.

| metric | 0012 (orphans, t<=600) | 0015 (clean) |
|---|---|---|
| interface_q RMSE / max | 7.297 / 24.83 | **5.718** / 30.08 |
| zone_voltage RMSE / max | 0.01738 / 0.03531 | **0.00271 / 0.00766** |

Zone-voltage RMSE improved 6.4x and max 4.6x: the phantom 480 MW *was*
contaminating runs 0011/0012, and the figures drawn from them.

Tap sync itself was verified correct and is **not** the cause: sync writes
`nntap`/`n3tap_h` = `tap_pos` with matching neutrals, `_init_shadow_state`
reads the same attributes, and `PARITY_LDF_SETTINGS` has `iopt_at = 0` so the
load flow cannot move taps.  The divergence is created by the first dispatch,
i.e. the two independent closed loops decide differently.

## G 01: actuator-set asymmetry removed (supervisor chose option b)

`gen[9] G1_bus38` is the 10 GVA 'Rest of USA/Canada' equivalent, `slack=True`,
`vm_pu=1.03`.  pandapower let the OFO dispatch its setpoint (6 writes in run
0012); the PF RMS model has no AVR block for it, so every write was skipped --
the static plant was strictly more capable than the RMS one.  Beyond the
setpoint, the RMS machine runs at constant excitation so its terminal voltage
*floats*, while pandapower pins it under all conditions.

Resolved by withdrawing the actuator from **both** plants, not adding an AVR:

- `MultiTSOConfig.dispatch_slack_gen_v_ref = False` (new, default off).
- `TSOControllerConfig.non_dispatchable_gen_indices` -- V-ref bounds pinned to
  the present setpoint and the `H` column zeroed.  Deliberately weaker than the
  OOS mask, which additionally zeroes the Q_gen row, gradient and reserve
  terms; here the machine stays fully observed and only loses its actuator.
- Runner derives the set from `net.gen.slack`.

Verified: `[actuators] AVR V-ref withheld from slack/equivalent gen(s): [9]`
and zone 1's third setpoint holds at exactly 1.030000 across dispatches while
its neighbours move.  13 controller tests pass.

## Instrumentation added (needs a re-run to take effect)

- `Record.bus_vm_pu`: per-bus voltages for every zone and DSO bus, static side.
- `monitored_outputs(..., include_der=True)` now also monitors DSO feeder
  buses; the replay harvest filter widened from `u_TN_bus` to `u_`.

Both close documented gaps in the figures: until a re-run, the zone plot is
RMS-per-bus against a static *envelope*, and the DSO plot is envelope vs
envelope.

## Status

Gate E comparison **valid**, settling **PASS**, and on the clean model the
endpoint agreement is now good: zone-voltage RMSE 0.0027 pu (max 0.0077),
interface-Q RMSE 5.72 Mvar.  The remaining residual is attributable to 7 DSO
coupler taps differing by 1-2 steps between two independent closed loops --
not to plant fidelity.  Proving that last step still needs the open-loop
`u -> y` test (apply the static run's recorded `u` to the RMS and compare `y`),
which remains the missing instrument.
