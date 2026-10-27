# D1 result (run 0003): Δ\* = 0.020 pu, set by discrete events; start-up fix and rerun of stage 2

Timestamp: 2026-10-27, ~18:00–18:20 Europe/Berlin.

**Runs:**
- `results/ch_09_parameter_selection/0003_2026-10-27_171554_ch_9_4_deadband`
  (17:15–17:59): first campaign. Stage 1 is valid; stage 2 is superseded.
- `0005_2026-10-27_181444_ch_9_4_deadband`: stage 2 rerun with stage 1
  reused (`--stage1-from`).

## D1 reduction (suppressed traces, W1–W6, 2 h, dt 10 s, no noise)

- **dt check:** W1 routine d_max is 0.0016355 pu at dt 20 s and 0.0016355 pu
  at dt 10 s. The excursion does not depend on the grid, so dt = 10 s is
  sufficient.
- **Start-up phase (first 30 min):** d_max = 0.046 pu. This is the
  cold-start anchor (`qv_vref_pu` = 1.03) before each converter's first
  dispatch, followed by the controllers' initial moves.
- **Routine operation:**
  - Overall: d_max = 0.0158 pu (W2, STS), so Δ\* = 0.020 pu (narrower
    0.015) under the thesis rule (eq:param:deadband:detector, ε_U = 0.001).
  - By window and level (STS / TS): W1 0.0016 / 0.0010, W2 0.0158 / 0.0025,
    W3 0.0014 / 0.0018, W4 0.0137 / 0.0034, W5 0.0136 / 0.0048,
    W6 0.0011 / 0.0005.
  - Distribution over 65 784 converter intervals: median 0.0001,
    99th percentile 0.0010, 99.9th percentile 0.0047 pu.
  - 57 intervals exceed 0.005 pu. **All 57 coincide with a discrete event:**
    - **27, the DER capability step (W2, W4).** A group of STS converters
      with one profile crosses P/S_n = 0.1. In the modelled VDE-AR-N 4120
      capability, `_qv_capability` drops from ±0.1·S_n to 0 there. Their
      delivered Q falls from q_set to 0, by up to 10 Mvar each, and
      neighbouring converter buses step by up to 0.0158 pu. This happens at
      a profile solve, without any dispatch.
    - **30, TS bank and tap actions at the TS dispatch (W5).** The STS
      anchors are re-set from the voltage before that dispatch, so the bank
      step itself (two MSC levels in zone 2, +0.0136 pu at STS buses)
      appears as an excursion of the new interval.
  - Without these events, routine drift stays ≤ 0.00497 pu.
- **Thesis definition:** the thesis counts "ordinary OLTC and shunt
  actions" and profile movement as normal operation, so the rule as written
  gives 0.020 pu. The decomposition above is the result to report; how to
  read it is the author's decision.

## Flaw in run 0003's stage 2, and the fix

- **Flaw:** the D1 confirmation runs and the D2 legs used the tested width
  from t = 0. The start-up excursions (cold-start anchors) activated the
  droop: 22–87 activations per window at Δ\*, **all** in the start-up
  phase, and zero after it in every window at both widths. The runs
  therefore left start-up in a different state from the suppressed
  reference.
  - D1 "identical" was False everywhere except W5.
  - The D2 legs had no common pre-event state, which the thesis requires.
- **Fix** (7bf0a9c):
  - `DroopProbePlant(width_after=(t_s, width))` switches
    `net.sgen.qv_deadband_pu` of the Q(V) converters at `WARMUP_S`. Every
    droop-active run starts suppressed.
  - The D2 summary checks the pre-event converter-bus voltages of each leg
    against the droop-off leg.
  - Confirmation activations are split into start-up and routine.
- **Smoke run** (`_local_archive/.../0004_…_smoke`, start-up 300 s):
  - start-up bit-identical to the reference (ΔV = 0 over 1 936 solves);
  - Δ\* identical over the whole run;
  - onset of activation at the narrower width;
  - D2 legs identical before the event (ΔV = 0).
- **Thesis wording:** the text says "with the tested width active from the
  start of each run". The fix activates it at the end of start-up, 30 min
  before the event, so that sentence needs to be adapted (author).

## Also

- Archiving the two smoke runs failed half-way, twice: the session's working
  directory was inside the run folder. Each time the copy was complete
  (45 files) and the source was empty. The empty source folder was removed
  and the manifest marked `superseded` by hand. No data was lost.

## Run 0005 (stage 2 rerun, 18:14–18:44) and reduction fix

- **Run of record:**
  `results/ch_09_parameter_selection/0005_2026-10-27_181444_ch_9_4_deadband`
  (3 invocations: the campaign and two reduction reruns). Run 0003 is
  archived, superseded by 0005.
- **Reduction fix (`--reduce`, no re-simulation):** droop support is now
  `q − clip(q_set, q_min, q_max)` where the dead zone is active, and 0
  elsewhere. Before, `q − q_set` also counted commands outside the
  capability band, and the Q(V) loop's convergence residual (≤ 0.1 Mvar)
  where the dead zone was inactive. With that version, the droop-off legs
  showed a spurious "hand-over".
- **D1 confirmation:**
  - At Δ\* = 0.020: 0 activations, identical to the reference in W1–W6
    (ΔU = 0, interface ΔQ = 0, same tap and bank operations).
  - At 0.015: onset in W2 only, at the capability step at 7170 s
    (4 activations, max ΔU 0.0006 pu). The other windows are identical.
- **D2 (72/72 events verified; legs identical before the event):**

  | Event | ΔU at event bus, droop off | Δ\* = 0.020: support TS + STS, mean (range) | ΔU, Δ\* | 2Δ\* = 0.040: support | ΔU, 2Δ\* |
  |---|---|---|---|---|---|
  | +200 Mvar at bus 7 (inductive) | −0.0349 | 7.7 + 18.5 Mvar (W1–W3, W6: 33–37; W4: 20; W5: 0) | −0.0309 | 0 | −0.0349 |
  | −200 Mvar at bus 7 (capacitive) | +0.0356 | −8.7 − 19.1 Mvar | +0.0316 | 0 | +0.0356 |
  | +200 Mvar at bus 11 | −0.0635 | 0 + 15.8 Mvar (W4: 10; W5: 0) | −0.0596 | 7.4 Mvar | −0.0613 |
  | −200 Mvar at bus 11 | +0.0642 | 0 − 16.3 Mvar | +0.0604 | −7.6 Mvar | +0.0622 |

  - The droop support returns to zero within 360 s at Δ\* (200 s at 2Δ\*),
    i.e. by the second TS dispatch after the event.
  - **Observation:** after the inductive events, the first TS dispatch
    (3780 s) commands one bank level in three STSs plus TS taps, the same
    in every leg. With the droop off this lifts an STS bus to 1.114 pu (W2)
    or 1.109 pu (W6) for one 20-s interval, until the STS controllers
    correct it. At Δ\* the droop absorbs the step and the maximum stays at
    1.065 pu.
