# Prompt for the PowerFactory machine — what fires ~7 s after the N-1?

Paste the section below to that session. It is self-contained.

---

## Task

In the dead-band × droop study, every N-1 run shows **two** voltage excursions,
not one. Find out what causes the second. It decides whether a number in the
dissertation stands.

## What is observed

The outage is delivered once, at **t ≈ 180.6 s** (configured for 200 s; armed at
the start of the enclosing solver step). Then, at the worst TS park:

1. an immediate dip of 0.018–0.031 pu, which **decays** — the envelope over one
   swing period (~3.2 s) falls by a factor 3–7 over the next several seconds;
2. a **second, abrupt excursion**, five to ten times larger, rising within about
   one second. This is the peak the study's upper bound is read from.

| cell (no droop, δ = 0.5) | run | twin | 2nd excursion | peak |
|---|---|---|---|---|
| 650 MW @ −117 MW, m = 0.10 | 0453 | 0440 | +6.4 s | 0.0995 pu @ 189.7 s |
| 650 MW @ +409 MW, m = 0.10 | 0405 | 0391 | +7.2 s | 0.0825 pu @ 190.4 s |
| 560 MW @ −117 MW, m = 0.10 | 0466 | 0440 | +9.2 s | 0.0731 pu @ 192.2 s |
| 560 MW @ +409 MW, m = 0.10 | 0426 | 0391 | +11.2 s | 0.0412 pu @ 194.1 s |

Delays are from the outage. **They order by the size of the tripped machine**,
not by the size of the initial transient (which is 0.031 / 0.023 / 0.018 /
0.028 pu — not monotone). That points at something integrating a *system-wide*
deficit, not a local voltage error.

Setup common to all: `rural_700`, DSO_3 ×2, rev-2 sensitivities, 600 s horizon,
`tso_qv_deadband_pu = dso_qv_deadband_pu = 0.5` (so the local Q(V) droop never
engages — this is the no-droop reference leg), trip `gen[1]` or `gen[5]`.
Windows: `2016-02-22T13:00` (−117 MW) and `2016-01-05T08:00` (+409 MW).

## Already excluded — do not redo these

- **Late/duplicated delivery of the outage.** 91 of the 94 monitor signals move
  together at 180.6 s; the DS bus voltages swing ~10 % at once.
- **An external control action.** TS OLTC taps (`zone_oltc_taps`) and MSC/MSR
  banks (`zone_tso_shunt_states`) are unchanged until t = 360 s. The one DS tap
  that moves (`dso_trafo_tap_pos`) does so at t = 220 s, i.e. *after* the second
  excursion. Checked in `rms_records.pkl` against the twin.
- **A growing oscillation.** The envelope decays monotonically right up to the
  jump; an exponential fit over the decaying stretch has the wrong sign
  (R² = 0.13 in the clearest cell).

Scripts and outputs: `results/deadband_droop/check_discrete_actuators.*` and
`check_second_excursion_envelope.*`.

## Hypotheses, in the order the evidence favours them

1. **Frequency threshold.** With weak or absent primary control the frequency
   ramps at a rate proportional to the lost infeed, so a fixed threshold is
   crossed at a time inversely proportional to it. Explains the ordering by
   machine size mechanically. (650/560 ≈ 1.16 predicts a delay ratio ~1.16;
   observed 1.44–1.56 — same direction, and the two windows differ in loading.)
2. **Inverse-time limiter or protection** on the units picking up the deficit —
   a field-current/over-excitation limiter or an overload element. Both act
   sooner for a larger loss and both give exactly this shape.
3. **Latching logic in the converter models or in the custom `QVPRE` DSL block.**
   The parks are driven by dispatch-time `qset`/`Vanchor` *parameter events*, so
   a mis-scheduled event would look like this and would be our own bug.

## Plan, cheapest first — stop as soon as one answers it

1. **Read the simulator's output window** after `ComSim` for messages timestamped
   186–192 s. Any relay operation, limiter activation or breaker action prints
   there. This alone probably ends the investigation.
2. **Enumerate scheduled events**: `app.GetFromStudyCase('IntEvt').GetContents()`.
   Confirm exactly one outage at ~180.5 s. Then check the `EvtParam` objects the
   co-simulation adapter writes for `qset`/`Vanchor` — **if one lands near 187 s
   that is the answer, and it is a bug rather than physics.**
3. **Re-run one cell with proper monitoring.** 650 MW @ −117 MW, δ = 0.5,
   simulate 175–200 s only. Monitor per `ElmSym`: `s:xspeed`, `s:firel`, field
   current and voltage, `m:Q:bus1`; the AVR/OEL composite's limiter output and
   integrator state; per `ElmLne`: `c:loading`; every `ElmRelay` trip signal; and
   breaker `on_off`. Find the first variable that steps at ~187 s.
4. **Plot COI frequency 180–195 s.** If it ramps and the excursion coincides with
   a round threshold (49.8 / 49.5 Hz), it is hypothesis 1. If frequency is flat,
   hypothesis 1 is dead.
5. **Knock-outs, one at a time, same cell**: all protection `outserv = 1`; then
   OEL limits raised out of range; then a **standalone run** of the same trip
   with no external controller writes — if the second excursion vanishes there,
   the adapter is implicated.

## Practical notes

- The study's run directories were **moved** on 2026-08-05 to
  `results/deadband_droop/runs/` (manifest: `MANIFEST_runs.csv`, 95 usable of
  102). `analysis/deadband_n1.py` still defaults to `results/rms_phase6_replay`,
  which now holds only the older m = 0.06 study.
- **`exit=1` is normal** for every N-1 run: the entry point returns Gate E's
  verdict, and Gate E validates QSS/RMS *equivalence*, which a topology change
  legitimately breaks. Never use the exit code as the failure test.
- Per-run logs are **UTF-16**; an ASCII grep silently matches nothing.
- Environment on that machine: `F:\python_environments\qOFO_clean\python.exe`,
  run from `Z:\Python_Projekte\qOFO_GH`, `-X utf8`. `powershell.exe` is at
  `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`.

## What to report back

The first variable that moves at the second excursion, with its timestamp, and
whether it is (a) plant physics, (b) protection or a limiter, or (c) an artefact
of the adapter or solver.

**Why it matters:** the study's *upper* bound on the dead-band half-width is read
off that second peak. If it is an artefact, the upper bound moves and the study
needs re-running. The *lower* bound is unaffected — it is measured on undisturbed
twins, where none of this happens — and so is the ordering of half-widths, which
is monotone in the dead band at every series.

Also worth fixing while you are there: re-run the void cell **0480**
(m = 0.05, −117 MW, δ = 0.5, 560 MW trip). It diverges from its three siblings
from t = 0.6 s, i.e. before the disturbance, and is currently excluded.
