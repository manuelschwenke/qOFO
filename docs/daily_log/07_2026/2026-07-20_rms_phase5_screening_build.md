# 2026-07-20 — Phase 5 screening COMPLETE: Gate D verdict (20 s window)

> **⚠ SUPERSEDED (2026-07-20, later the same day).** The step battery this
> log reports was contaminated by an event-accumulation bug: `Delete()` on
> simulation events silently no-ops while a calculation is active, so every
> battery entry after the first replayed all previous events.  Of the numbers
> below only the **+60 Mvar TSO-park step (13.2 s)** is valid; the
> "+10 Mvar DSO park, 17.4 s" row was in reality "+60 Mvar TSO **and**
> +10 Mvar DSO together" — the clean DSO-park response never leaves the
> settled band (T_s = 0.0 s).  See
> `2026-07-20_rms_phase5_event_accumulation_fix.md` for the diagnosis, the
> fix, and the corrected Gate-D table (verdict unchanged: PASS at 20 s, with
> much larger margins).

**Headline (window = 20 s, confirmed by the user).** The open-loop
timescale-separation assumption **holds for the DER reactive actuators**:
after a DER Q dispatch the controlled bus voltages settle in **13–17 s**
(to a 1e-3 pu band), inside the 20 s window. Gate D: **PASS** for DER Q,
with a thin margin on the small DSO park (17.4 s → 2.6 s spare). The
response is strongly underdamped (~200 % overshoot, electromechanical
ringing, PSS-off), so PSS tuning would add margin but is not required at
20 s. *(Against the originally-assumed 10 s window this was a FAIL; the
step responses themselves are unchanged — 13–17 s.)*

**Open risk for the 20 s window:** the OLTC tap steps (5 s mechanical delay
+ the electromechanical ring) are not yet tested and could approach or
exceed 20 s — these are now the item to check, not the DER Q actuators.

## The three screening tools (`pf/screening.py`, all working)

- **flat** — 60 s no-event run, drift 6.8e-12 pu (genuine equilibrium).
- **modal** — full model *with* WECC DER dynamics: 837 modes, 0 unstable,
  **9 slower than 10 s**, min damping 0.038. The 743 new converter modes are
  fast/well-damped; the slow modes are still the machine electromechanical
  band (0.78–1.34 Hz, ζ ≈ 0.04). The WECC converters neither help nor hurt
  the electromechanical damping.
- **steps** — DER Q dispatch via the WECC handle `REEC_D.Qext`; 2 %-band /
  absolute-floor settling of the controlled outputs (voltage 1e-3 pu,
  interface Q 1 Mvar), machine-speed ring reported as diagnostic.

## Step-battery result (`full_t0_wecc`)

| dispatch | worst controlled output | T_s [s] | overshoot | Δ |
|---|---|--:|--:|--:|
| +60 Mvar on WP_TSO_s0_b18 (508 MVA) | u_TN_bus18 | **13.2** | 2.08 | 0.0059 pu |
| +10 Mvar on DER_DSO_1_s10_b50 (20 MVA) | u_TN_bus18 | **17.4** | 2.15 | 0.0061 pu |

- The DER Q step produces a real ~0.6 % voltage change at the park's own bus,
  which **overshoots ~200 %** and rings at the electromechanical frequency,
  settling to 1e-3 pu only after 13–17 s.
- The **DSO interface Q flows barely move** (≈ 0.02 Mvar) for a TSO-park step
  — the wind park is electrically distant from the couplers — so they settle
  within the 1 Mvar band immediately.

## Interpretation (facts vs caveats)

- **Fact:** local controlled voltages exhibit large overshoot and 13–17 s
  electromechanical ringing after a DER Q dispatch — consistent with the
  modal T_s (8–22 s) of the ζ ≈ 0.04 modes. Cause = low electromechanical
  damping with PSS disabled (enabling the untuned template PSSs made damping
  *worse*, see the modal daily log).
- **Caveat (band sensitivity):** settling is measured to a 1e-3 pu band; at
  the OFO's looser operational voltage tolerance (~5e-3 pu) the settling
  would be shorter. The band-independent red flag is the **~200 % overshoot**
  — an unambiguously underdamped response.
- **Scope:** the battery covers the DER Q actuators (the OFO's primary
  reactive actuators). Machine AVR V-ref and OLTC/shunt tap steps still need
  their RMS control-input signals wired; they would only *add* slower
  responses (taps carry a 5 s mechanical delay), so they cannot rescue the
  verdict.

## Go/no-go for the dissertation (20 s window)

The DER Q actuators (the OFO's primary reactive actuators) settle within the
20 s window, so open-loop timescale separation holds for them. Standing
items:
1. **Test the OLTC tap / machine steps** — with the 5 s tap delay these are
   the only responses that could exceed 20 s; wire their RMS control inputs
   and confirm.
2. **PSS tuning** is now optional (adds margin to the 17.4 s DSO case and to
   the tap steps) rather than required.
3. The OFO should still sample `y(t_k⁻)` before each dispatch (good practice
   given the underdamped ring).

## Artefacts / fixes this session

- `pf/wecc_apply.py` — WECC rollout (44 DER, ComInc green, Gate C intact).
- `pf/screening.py` — retargeted DER Q steps at `REEC_D.Qext`; down-sampled
  trajectory read (stride 5) + reduced monitor set (fixes the full-model
  read timeout); absolute-band settling metric; controlled-output-based
  Gate-D verdict with machine-speed ring as diagnostic.
- Results in `results/screening/full_t0_wecc/`.

## Remaining (Phase 6 + polish)

- Machine AVR V-ref RMS signal; OLTC/shunt RMS tap events (completeness).
- `ComRes` CSV export if larger monitor sets are wanted.
- Phase 6: `PowerFactoryPlant.apply_u` writes `REEC_D.Qext` / AVR usetp /
  tap events; closed-loop replay + settling statistics.
