# D1/D2 dead-band experiments on the QSS plant: producer, probe, selected set switched

Timestamp: 2026-10-27, ~15:00–16:00 Europe/Berlin.

**Reason:** the author asked for D1/D2 to be run on the quasi-steady-state
plant (Ch 9 `\todo`s), and decided on the plan
`docs/tuning/2026-10-27_D1_D2_deadband_plan.md`: decisions 1–3 by the author,
4–6 delegated. The decisions are recorded in §6 of the plan.

## What changed

| File | Change |
|---|---|
| `experiments/ch_09_parameter_selection/_ch9_selected_design.py` | `SELECTED_PARAMSET = "ch9_selected_band_2026_10"` (author decision 1). Was `ch9_selected_final_2026_10`. |
| `experiments/ch_09_parameter_selection/_droop_probe.py` (new) | Described in detail below. |
| `experiments/ch_09_parameter_selection/ch_9_4_deadband.py` (new) | The D1/D2 producer, writing a documented run (`start_thesis_run`, target `ch:timescales:deadband`). Its stages are listed below. |
| `experiments/ch_09_parameter_selection/README.md` | New row for the producer. |
| `docs/tuning/2026-10-27_D1_D2_deadband_plan.md` | Status DECIDED; §6 with the decisions. |

**The probe.** `DroopProbePlant` is a subclass of `PandapowerStaticPlant`,
passed as `plant_factory`; `DroopProbeFactory` mirrors the runner's default
plant flags.
- It records, per Q(V) converter and after every `advance`: time, phase
  (`profile` = pre-dispatch with held anchors; `end` = after a dispatch or
  event), anchor, bus voltage, q_set, delivered Q, P, capability, dead band,
  slope, the unsaturated droop increment (the law of
  `QVLocalLoop._compute_target`) and a dead-zone flag.
- It also records load Q and voltage at the watched event buses.
- **No runner edit:** every `isinstance(plant, PandapowerStaticPlant)`
  check treats the subclass as the default plant. The parallel `case2/ref`
  branch changes the runner, so this avoids conflicts.

**The producer's stages.**
1. **D1 reference:** W1–W6, 2 h, droop suppressed (dead band 1 pu), no
   measurement noise, dt 10 s.
2. **dt check:** one W1 trace at dt 20 s.
3. **Reduction:** excursions d_ik per converter and anchor interval, with
   intervals that start in the first 30 min reported as start-up.
4. **Selection:** Δ\* = min{d ∈ 𝒟 : d ≥ max d_ik + ε_U}, with
   𝒟 = {0.005 … 0.020} pu and ε_U = 0.001 pu.
5. **D1 confirmation:** runs at Δ\* and at the adjacent narrower width,
   compared with the reference at the same solves.
6. **D2:** 6 windows × 4 events (±200 Mvar at pandapower buses 7 and 11, a
   dormant load switched in at 3610 s) × 3 legs (Δ\*, 2Δ\*, off). Each cell is
   checked for the event reaching the plant; the producer records support,
   voltage deviation, extrema and a hand-over series.

## Verified

- **Smoke run** (W1, 15-min traces, one event, every stage), archived to
  `_local_archive/results/ch_09_parameter_selection/0002_…_smoke`:
  - The event reached the plant (+200.0 Mvar), and bus 7 dropped 0.035 pu.
  - At the selected width the trajectories are identical to the suppressed
    reference (Δv = 0); at the narrower width there were 4 activations.
  - The run ended `complete`.
- **Smoke excursion of 0.041 pu:** this is the start-up interval before the
  first STS dispatch. The converters still carry the cold-start anchor
  (`qv_vref_pu` = 1.03) at a 1.07 pu bus. From the second interval on the
  excursions are about 0.0002–0.003 pu. The campaign excludes the first
  30 min as start-up; the smoke run had no start-up exclusion.
- **Full test suite** at cb91dd1 + the band switch: 889 passed, 15 skipped,
  0 failed (61 min).

## Open

- **W5:** the author asked whether W5 can be dropped. It is one of the six
  Stage-I conditions and is in the λ screen; per the Ch 9 source comment it
  binds the TS scaling (λ = 1.000 at TS_2, W5). Dropping it changes the
  starting weights, so Experiment B would have to be re-applied. See the
  reply of 2026-10-27; the decision is the author's.
