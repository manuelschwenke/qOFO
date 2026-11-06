# 2026-11-04 — Step-like interface-Q offsets in the STS window: profile ramp, not an actuator

**Question (author, ≈ 09:40).** With `TRIAL_PARAMSET = "trial_8ad8_v3_run0006"` (run 0020,
`make_config()`, frozen doubled-DSO_3 network, γ_OLTC,Q = 0), the STS live window shows sudden,
held steps in the metered interface Q of DSO_2 and DSO_3 (e.g. DSO_2 T5) while the dashed
setpoints stay almost constant. No coupler tap, TS tap or MSC/MSR moves at those times.

**No code changed.** Diagnosis only (10:04).

## Method
- Run 0020 lives in a PyCharm console and has no `records.pkl`. Re-ran its config headless for
  130 min (scratch driver, live plots off, otherwise identical): 390 steps, 604 s.
- Compared with the stored runs 0001–0016 of `run_multi_system_ofo` (same period,
  2016-01-05 08:00).

## Findings (established from the records)
1. **No discrete change between 30 and 120 min.** TS OLTCs, MSC/MSR levels, coupler taps and
   units in service are constant from the G4 trip (30 min) to the load event (120 min).
2. **The disturbance is the STS load profile.** `data/profiles.csv` has 15-min values,
   linearly interpolated to the 20-s step (`core/profiles.py`). `mv_rural_qload` and
   `mv_rural_pload` change slope at 09:00 and 09:30. DSO_2 load Q per 20-s step:

   | window | ΔQ_load per step |
   |:--|--:|
   | 48–60 min | +0.075 Mvar |
   | 64–90 min | +0.871 Mvar (−33 → +45 Mvar; P 177 → 233 MW) |
   | 95–105 min | −0.274 Mvar |
   | 108–120 min | +0.062 Mvar |

   The IBGs follow the ramp: the DSO_2 IBG sum goes from −16 to +63 Mvar between 60 and 90 min.
3. **The plotted offset has two parts, both proportional to the ramp rate r.**
   - *Sampling instant.* The STS window plots the pre-control metered sample y⁻_k, taken after
     the profile increment and before the DSO acts. Summed over the couplers,
     y⁻_k − y⁺_(k−1) = 0.84 Mvar (DSO_2) and 0.91 Mvar (DSO_3), against ΔQ_load = 0.87 Mvar.
   - *Ramp-following lag of the integrating OFO.* The post-control plant-truth error e⁺ also
     grows during the ramp. For DSO_2 T5 it goes from 0.13 to 0.65 Mvar. Over the three couplers
     it grows by 0.51 Mvar.
   - Plotted error e⁻ of DSO_2 T5: +0.20 → +1.10 → −0.48 → −0.07 Mvar over the four windows.
4. **Not caused by the trial weights.** Run 0016 (`ch9_selected_uc_2026_11_v2`, 2026-11-03)
   differs by less than 0.05 Mvar in every quantity above. All stored runs 0001 (2026-10-22) to
   0016 show the same signature at 60 and 90 min on DSO_2 T5.

## Interpretation (hypothesis-level where marked)
- Scalar fit (hypothesis, aggregate over the multivariable loop): e⁻ = r/λ and
  e⁺ = r(1−λ)/λ. From the DSO_2 sums, Δr = 0.78 and Δe⁻ = 1.29 give λ ≈ 0.60 per step.
  The predicted Δe⁺ = 0.51 Mvar matches the measured value.
- A slope discontinuity in the profile therefore moves the steady lag within a few DSO steps.
  This looks like a step response with no setpoint step.
- Why it looks new (hypothesis): until 3f1577e (2026-11-02) the cascade window plotted
  `dso_q_actual_mvar`, the post-control plant truth summed per DSO. The STS window plots
  per-coupler pre-control metered samples on a y-range of about 5 Mvar.

## Open (for decision, not implemented)
- Show `dso_trafo_q_actual_mvar` (post-control) next to the metered sample in the STS window,
  or say in the tile title that the metered sample is pre-control.
- The lag itself is inherent to the integral-type OFO under ramps. Options would be a
  disturbance-rate term, or smoother profile interpolation (which changes the scenario).
  Both are architectural or scenario changes and need the author's decision.
- In the screenshot of run 0020, the actuator tiles (IBG Q, coupler taps) show the 30-min event
  marker at ≈ 37 % of the width, the output tiles at 25 %. They displayed different time
  windows, although `_render_all` sets the same x-limits on all axes. Not investigated.
