# BME Phase 6b — first full ladder (360 min) + D6 calibration

**Date:** 2026-07-03
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Phase 6 item (2): D6 (ε_switch, c_switch) calibration from the
360-min bme rung's switching ledger, plus the first full four-rung ladder
run (`experiments/011_BME_LADDER.py --rung all --minutes 360`).

## What was run

The complete ladder on the shared 005 case-study scenario (gen-2 trip
@60′/restore @180′, 200 MW + 100 Mvar load step @90′–360′, tie-line-25
trip @260′/restore @360′), hygiene at ε = 0 (pre-calibration). Last-hour
means:

| rung | Φ [MW] | losses [MW] | OLTC switches | V range [pu] |
|---|---|---|---|---|
| none | 37.47 | 47.10 | 5 | [0.978, 1.055] |
| vref | 37.48 | 46.99 | 4 | [0.981, 1.054] |
| bme | 36.42 | 45.64 (−3.1 %) | 26 | [0.982, 1.063] |
| bme_loss | 30.98 | 40.80 (−13.4 %) | 33 | [0.986, 1.170] |

Key readings: vref neutral on the uniform-schedule scenario (documented
behaviour); bme −3.1 % sustained losses with the voltage envelope
essentially held; **bme_loss's −13.4 % is inadmissible — V reaches
1.170 pu** over the 6-h horizon (the 60-min calibration's 1.059 was
deceptive), the strongest empirical vindication of the D2 band hinge yet.

## D6 calibration (delegated decision, executed per the recorded rule)

From the bme rung's 57-entry ledger (scaled Φ̂ units): anchor
(b) = median per-step |ΔΦ| on no-commit steps = 1039;
**ε_switch = 5×(b) = 5193 ≈ 5.2e3**, agreeing with the independent
sanity cap 0.5·median|ΔΦ̂_proposal| = 5256; **c_oltc = 1.0e3,
c_shunt = 5.2e3** (breaker vs tap wear, consistent with the shunt
integrator's stricter dwell/budget treatment). Premise data (§3.10.2):
sign agreement predicted/realised 0.78 across a four-contingency window
(1.00 on the clean 60-min run). Constants wired into
`011_BME_LADDER.py` (both bme rungs; ablation isolates w_band only);
analysis script: session scratchpad `d6_calibration.py` (method
documented in BME_STATUS.md §6b).

## Structure of the change

* `experiments/011_BME_LADDER.py`: `BME_EPSILON_SWITCH`,
  `BME_SWITCH_COST_OLTC`, `BME_SWITCH_COST_SHUNT` constants (with
  derivation comment) + wiring in `make_ladder_config`.
* `docs/BME_STATUS.md`: §6b section (ladder table, D6 derivation,
  bme_loss voltage finding, premise statistics).
* Post-ε re-run of both bme rungs appended to §6b on completion
  (expected effect: switch count drops toward the baseline while Φ is
  minimally affected — spec headline claim 4).

## Reason

Spec §5 Phase 6 (ladder rungs one command each; D6 calibration item) and
DECISION D6 (delegated 2026-07-02: ε ≈ 5× median per-step continuous
improvement, c_switch from device-wear reasoning, rationale documented at
calibration time — this log and BME_STATUS.md §6b are that rationale).
