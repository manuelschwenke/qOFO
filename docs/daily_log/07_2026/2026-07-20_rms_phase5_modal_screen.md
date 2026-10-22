# 2026-07-20 — Phase 5 kickoff: RMS foundation + modal screen (preliminary 10 s verdict)

**Context.** Phase 5 start. Confirmed the RMS foundation, then ran a modal
screen first (cheapest read on the 10 s timescale-separation assumption)
before building the full screening battery.

## Foundation (all confirmed live, `02_RMS_CoSim`, full model)

- Dynamic models present: 10 `ElmComp` frames, 27 `ElmDsl` blocks
  (9 AVR + 9 governor + 9 PSS); every retained machine has a plant model
  (G 01 = "Rest of U.S.A./Canada" equivalent).
- `ElmStactrl` natively supports a **Q(V) characteristic with deadband**
  (`qu_char` Q(V) option, `udeadblow`/`udeadbup`) — the plan's biggest risk
  (needing a DSL droop frame) is eliminated.
- `ComInc` (RMS init) + a 2 s `ComSim` flat run execute cleanly (return 0).
- The `02_RMS_CoSim` `ComLdf` was aligned to the ZIP parity settings
  (`iopt_pq=1`, no auto taps/shunts/limits) so the RMS init starts from the
  validated operating point.

## Modal analysis (`ComMod`, full model at t0, PSS disabled)

110 eigenvalues / 94 distinct modes. **System is stable (0 modes with
Re > 0).** But damping of the electromechanical band is poor:

| ζ | f [Hz] | T_s = 4/|Re| [s] | kind |
|---|---|---|---|
| 0.037 | 0.776 | 22.4 | inter-area/local |
| 0.042 | 1.081 | 13.9 | local |
| 0.047 | 1.086 | 12.4 | local |
| 0.055 | 1.173 | 9.8 | local |
| 0.056 | 1.342 | 8.4 | local |

Plus slow **non-oscillatory** modes at T_s ≈ 105 s and 34 s (governor /
frequency-regulation band — monotonic, do not "ring"). 9 of 94 modes have
T_s > 10 s; 34 have T_s > 2 s.

**PSS contingency tested and rejected (as the plan foresaw).** Enabling the
9 template PSS blocks made damping *worse* (min ζ 0.037 → 0.025, slowest
oscillatory T_s 24.7 → 31.6 s): the IEEE task-force PSS parameters do not
fit the Type-1 AVRs. PSS left disabled (documented state). Proper PSS
*tuning* is a contingency task, not a toggle.

## Interpretation (facts vs open question)

- **Fact:** the wind_replace+full plant, PSS-off, has several
  electromechanical modes (0.78–1.34 Hz, ζ < 0.06) whose modal settling
  time (8–22 s) exceeds the 10 s STS dispatch window. Removing 4 machines
  and replacing them with STATCOM wind parks reduces system damping.
- **Open question (the crux):** modal settling time is a *per-mode*
  property, not the settling time of the *controlled outputs* (EHV–HV
  interface Q, nodal voltages) to an *OFO dispatch step*. The OFO commands
  DER Q-setpoints and OLTC taps, which primarily excite AVR-driven
  voltage/Q dynamics rather than rotor-angle oscillations. Whether a
  Q/tap dispatch step actually excites the poorly-damped rotor modes enough
  to matter for the controlled outputs is undetermined by the modal screen
  alone.

**→ The step-response battery is now clearly necessary and well-motivated:**
it measures the 2 %-band settling of the actual controlled outputs to the
largest single-dispatch actuator steps, which is the real 10 s test.

## Next

Build `pf/screening.py` (`flat` / `modal` / `steps`) and run the step
battery at t0 + peakres → Gate D verdict. Step catalogue (per-actuator
magnitudes) needs definition — see report.
