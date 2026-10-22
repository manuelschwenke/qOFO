# 2026-08-06 — Q(V) switch-on at the contingency, and a settable RMS step

**Timestamp:** 2026-08-06
**Scope:** `pf/plant.py`, `pf/replay.py`, `pf/screening.py`,
`experiments/run_comparison_rms_cosim_qss.py`. Two new CLI options; both
default to the previous behaviour, so no existing run path changes.
**Follows:** `2026-08-05_second_excursion_is_missing_primary_control.md`.

---

## 1. `--qv-switch-at-contingency`

**Reason.** In the dead-band × droop comparison the two legs do not start from
the same operating point. Measured on the post-fix twins (0414 δ = 0.5 vs 0415
δ = 0.01), the droop leg sits **+0.0062 pu** higher at the worst TS park and
its fleet injects **+61.9 Mvar** more before the outage. So the measured
difference conflates two effects:

* the droop's **steady-state voltage support** (the offset above), and
* its **disturbance rejection** (the fleet injects a further +64 Mvar within
  0.1 s of the outage, against +35 → −13.7 Mvar without it).

Both are genuine benefits, but the figure is captioned as if it showed the
second alone.

**Change.** With the flag, the RMS parks run with the Q(V) dead band held at
`QV_DISABLED_DEADBAND_PU = 0.5` (the study's own encoding of "droop off")
through the whole run-up; at the instant of the contingency an `EvtParam` on
each `QVPRE.db` installs the configured half-width. Both legs then share a
bit-identical run-up and the comparison isolates rejection.

**Method.** `_anchor_qv_precontrollers` stashes the intended dead band per park
in `_qv_target_db` and writes the disabled value instead; `apply_contingency`
emits the `db` events at the same `t_evt` the outage uses.

**Landmine handled.** The persistent event pool preallocates `qset` and
`Vanchor` per QVPRE but **not** `db`. 44 events created after ComInc would hit
PF's mid-run admission limit — the failure documented on 2026-07-23, where a
1-slot default froze every actuator at t ≈ 41 s. `_preallocate_event_slots`
now reserves one `db` slot per park (seeded from `params:3`).

**Deliberately RMS-only.** The static plant keeps Q(V) active throughout, so
**Gate E cannot certify a run made with this flag** — the two plants run
different local control during the run-up. This is scoped to this one
experiment; everywhere else both plants run Q(V) permanently (user decision,
2026-08-06).

## 2. `--rms-step-ms`, `--rms-step-max-ms`, `--adaptive-step`

**Reason.** The RMS step was the module constant `RMS_STEP_MS = 10.0`, written
straight into `ComInc.dtgrd`, with `iopt_adapt = 0`. But the converter models
carry dynamics far faster than 10 ms — `REGC_C` `Te = 0.1 ms`, PLL
`Kipll = 1400` — so the inner loops are integrated at ~100× their time
constant, and the post-switching trajectory shows zig-zag that is plausibly
numerical rather than physical.

**Change.** `ScreeningContext` gained `rms_step_ms` (the step, or the
*smallest* step when adapting), `rms_step_max_ms` and `adaptive_step`; it sets
`dtgrd`, `iopt_adapt` and `dtgrd_max` accordingly and prints what it used.
Plumbed through `PowerFactoryPlant` and the replay factory to the CLI.
Defaults reproduce every run before today.

Intended setting for the re-run: `--adaptive-step --rms-step-ms 1
--rms-step-max-ms 10`, i.e. down to 1 ms where the error tolerance demands it,
10 ms elsewhere.

**Validity caveat.** This buys *numerical* resolution, not physics. An RMS
phasor model assumes fundamental-frequency phasors, so below ~10 ms — half a
cycle at 50 Hz — its output should not be read as a physical claim, however
finely integrated.

## 3. What the re-run will be

180 s run-up, outage at 180.6 s, **240 s total** (≈ 59 s after the outage).

* the run-up cannot be shortened: the TS period **is** 180 s, so the first TS
  dispatch happens at t = 180 (visible as `zone_v_rms_err` 0.00984 → 0.00810).
  An outage at t = 40 would land in a system whose upper layer has never
  dispatched, dissolving the "drift since the last dispatch" framing; the DSO
  layer is also 3× less converged there (σ = 2.27 vs 0.73 at t = 160).
* 240 s costs the metric nothing: max |ΔV| over [outage + 0.5 s, end] is
  **0.01502 / 0.00753, ratio 2.00** for end = 240, 270, 300, 330 and 360 s
  alike, because both peaks land at t ≈ 181.8 s.
* **one twin**, not two: with the flag a twin never reaches a contingency, so
  its droop stays off for the whole run and the δ = 0.5 and δ = 0.01 twins
  would be identical. One shared baseline (user decision, 2026-08-06).

## 4. Open

* The metric window itself still has to be fixed and applied to every cell.
  It currently depends on `--stride`, an implementation detail: the 100 ms
  export reports 0.03458 / 0.01126 (ratio **3.07**) where any consistent 10 ms
  window gives **1.6–2.1**. The dead band's measured benefit is inflated ~50 %
  by sampling alone. Recommended: outage + 500 ms, 10 ms sampling.
* Interface Q differs between the twins by rms 4.3 Mvar and, unlike zone
  voltage, does **not** decay with dispatch count (4.95 → 4.03 → 4.32 at
  t = 60 / 359 / 580 s). Zone voltage does decay (0.0030 → 0.0011), which
  supports treating the voltage offset as incomplete convergence; the Q
  difference needs its own explanation.
