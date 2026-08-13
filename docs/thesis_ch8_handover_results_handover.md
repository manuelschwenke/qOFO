> **SUPERSEDED 2026-07-29 — DO NOT USE.**
> Replaced by `docs/thesis_ch8_deadband_selection_handover.md`.
> This document framed the result as a droop-vs-OFO "share". That is not a
> meaningful quantity under the re-anchoring/offset law: at every dispatch the
> OFO absorbs whatever the droop delivered into the new qset, so the
> steady-state split is ~100 % OFO by construction and any "share" is an
> artefact of where in the dispatch cycle it is measured. The tables below
> (29 %, 53 %, 78 %, … and the outage row) must not be written into the thesis.

# Handover: dead-band handover results → thesis Ch. 8 §2

**To:** session with access to both `Z:\Python_Projekte\qOFO_GH` and the thesis
repository (`latex_diss_ms`)
**From:** qOFO_GH code session, 2026-07-29
**Target:** new Chapter 8 *"Parameterisation of the Control Hierarchy"*, §2
(dead-band selection / handover between the primary inverter droop and the OFO).
Chapter structure is defined in `docs/thesis_restructure_handover.md` — read that
first if the restructuring has not been applied yet.

---

## 1. What to write

**Claim.** The DER Q(V) dead-zone half-width δ sets the division of labour between
the autonomous local droop (Ch. 5) and the OFO (Ch. 4/6). The split is monotonic in
δ for moderate-to-large disturbances, and δ ≈ 0.005 pu is preferred.

**Framing constraint (important).** The author confirmed δ is a free design
parameter within regulatory limits, with 0.005 and 0.01 both clearly admissible.
Justify the selection on **control performance** — tracking error, actuator effort,
and this handover behaviour. Do **not** justify it by quasi-steady-state validity;
that appears only as a *consequence* in Ch. 11. Choosing a plant parameter to suit
the simulation method would be circular.

---

## 2. Results

**These are ladder-v2, baseline-corrected numbers (runs 0014–0018 + control).
Earlier tables circulated on 2026-07-28 from runs 0008–0012 are superseded and
must not be used** — they used a wrong revert formula (base drifted +75 % across
the ladder) and were uncorrected for baseline drift.

### 2.1 Droop share of the disturbance response

Fraction of the reactive-power response delivered by the local droop rather than
the OFO. Between dispatch instants `qset` is constant, so any change in park Q is
the droop; the step at a dispatch instant is the OFO. Both components are corrected
by subtracting the same quantity measured in a **no-disturbance control run**.

| disturbance | δ=0.0025 | δ=0.005 | δ=0.0075 | δ=0.01 | δ=0.015 |
|---|---|---|---|---|---|
| +5 %   | *15 %* | *55 %* | *97 %* | *74 %* | *56 %* |
| +15 %  | 37 % | 12 % | 0 %  | 0 %  | 0 %  |
| **+40 %**  | **81 %** | **66 %** | **46 %** | **33 %** | **25 %** |
| **+100 %** | **84 %** | **78 %** | **71 %** | **74 %** | **63 %** |

*The +5 % row is italicised because it is NOT usable — see §5.2.*

### 2.2 Net response magnitude [Mvar] (baseline-corrected)

Shows the ladder is properly resolved: response scales with disturbance size.

| disturbance | δ=0.0025 | δ=0.005 | δ=0.0075 | δ=0.01 | δ=0.015 |
|---|---|---|---|---|---|
| +5 %   | 5.5  | 5.5  | 3.8  | 3.1  | *21.0* |
| +15 %  | 8.9  | 8.8  | 8.8  | 8.8  | 7.3  |
| +40 %  | 23.2 | 25.9 | 25.0 | 24.3 | 30.6 |
| +100 % | 49.7 | 53.1 | 57.5 | 59.3 | 48.7 |

### 2.3 The statements to make

1. **The handover is tunable and monotonic in δ.** For +40 % the droop share falls
   81 → 66 → 46 → 33 → 25 %; for +100 % 84 → 78 → 71 → 74 → 63 %; +15 % supports the
   same direction (37 → 12 → 0 → 0 → 0 %). A narrow dead zone lets the fast local
   layer absorb the disturbance; a wide one defers it to the OFO. **Base the claim on
   +40 % and +100 %**, which have well-separated net responses.
2. **Disturbance magnitude must be swept, not maximised.** A machine outage tried as
   an anchor moved DSO_4's group Q by only 0.2–1.6 Mvar net (`gen[0]` is electrically
   remote from that group), so its apparent "≈50 % share" was the ratio of two
   opposing near-cancelling terms and carried no information. A worst-case-only design
   would have returned a null result and the false conclusion that δ is irrelevant.
   This is a methodological point worth one sentence.
3. **A no-disturbance control is essential on this cascade.** The TSO layer dispatches
   on a 180 s cadence, which beats against the event spacing, so some analysis windows
   contain a TSO dispatch and others do not. Measured in the control: baseline net ΔQ
   of **8.07 Mvar** in the first window and **4.55 Mvar** in the third, against ~0.3 in
   the other two. Uncorrected, this inverted the low end of the ladder entirely.

### 2.4 Supporting result from an earlier run (already in the code session)

Interface-Q tracking and actuator effort vs δ, physical VDE capability, RMS plant,
15 dispatch intervals — **independent of the handover measurement above**:

| δ | mean \|e\| [Mvar] | max \|e\| [Mvar] | DER-Q travel [Mvar] | tap switches |
|---|---|---|---|---|
| 0      | 2.846 | 6.337  | 100.8 | 0 |
| 0.0025 | 1.242 | 6.187  | 104.7 | 0 |
| **0.005** | **1.104** | 6.250 | **100.1** | 0 |
| 0.0075 | 1.496 | 6.815  | 110.6 | 0 |
| 0.01   | 1.652 | 10.179 | 126.4 | 1 |
| 0.015  | 3.013 | 16.826 | 156.2 | 1 |
| 0.02   | 2.912 | 16.828 | 145.2 | 1 |

δ = 0.005 minimises tracking error (−33 % vs the nominal 0.01) at unchanged actuator
effort. **Note for the text:** δ = 0 is *not* the optimum — it gives the worst
tracking (2.846) because without a dead zone all 44 parks respond to arbitrarily
small deviations, including each other's, and chatter. That is a useful point for
Ch. 5: the dead zone has a control-theoretic purpose beyond avoiding wear.

Two independent measurements — tracking and handover — therefore both favour 0.005.

---

## 3. Files (local paths)

**Figure — ready to include:**
```
Z:\Python_Projekte\qOFO_GH\results\handover_study\figures\handover_deadband.pdf
Z:\Python_Projekte\qOFO_GH\results\handover_study\figures\handover_deadband.png
```
Panel (a) droop share vs δ, panel (b) containment vs δ, decision range 0.005–0.01
shaded. Use the **PDF** in LaTeX.

**Run directories** (each: `config.json`, `rms_records.pkl`, `csv/rms_full.csv`):
```
Z:\Python_Projekte\qOFO_GH\results\handover_study\0008_2026-07-28_160739   delta=0.0025
Z:\Python_Projekte\qOFO_GH\results\handover_study\0009_2026-07-28_163709   delta=0.005
Z:\Python_Projekte\qOFO_GH\results\handover_study\0010_2026-07-28_170557   delta=0.0075
Z:\Python_Projekte\qOFO_GH\results\handover_study\0011_2026-07-28_173402   delta=0.01
Z:\Python_Projekte\qOFO_GH\results\handover_study\0012_2026-07-28_180314   delta=0.015
```
Runs 0001–0007 in that folder are smoke/verification runs — **do not cite them**.

**Scripts** (in the session scratchpad; copy into the repo if they should be archived):
```
scratchpad\handover_study.py     the driver (disturbance ladder + injection)
scratchpad\handover_sweep2.sh    the 5-run sweep
scratchpad\handover_analyse.py   regenerates both tables from csv/rms_full.csv
scratchpad\handover_fig.py       regenerates the figure
scratchpad\calib_v2.py           disturbance-magnitude calibration (PF-free)
```

**Method record:** `docs/daily_log/07_2026/2026-07-26_rms_outage_event_and_handover_study_design.md`

---

## 4. Method details needed for the text

- **Benchmark:** IEEE 39 `wind_replace`, 3 TSO zones + 4 DSO underlays, physical
  VDE-AR-N-4120 capability (the ±1.0 pu diagnostic override is **off**),
  `g_w_dso_oltc = 200`.
- **Excitation:** profiles **off** deliberately, so the injected disturbances are the
  only excitation and the treatment is isolated.
- **Disturbance ladder:** area-wide load steps on all 20 DSO_4 loads at +10 / +25 /
  +50 / +100 %, each reverted 40 s later, then a 250 MW machine outage (`gen[0]`,
  `G 10`) as the N-1 anchor. Events are mid-interval (dispatch boundary + 5 s) so the
  droop gets ~15 s of response before the OFO can intervene.
- **Fixed ladder across all runs**, only δ varies — so the same event is 2× the dead
  band at δ=0.005 and 1× at δ=0.01, which is the sweep in ΔV/δ that locates the
  handover.
- **Window:** 40 s (2 dispatch intervals) per disturbance.

---

## 5. Caveats that MUST appear

### 5.1 Method caveats

1. **Baseline correction is applied.** All droop/OFO components have the
   corresponding quantity from a no-disturbance control run subtracted. The control
   was run at **δ = 0.01 only**, so it is assumed δ-independent — a rigorous
   treatment would need one control per δ (5 further runs). State the assumption.
2. **Single operating point**, profiles off, DSO_4 only. Generalisation across
   operating points is not established here.
3. Ladder magnitudes are area-wide load steps on all 20 DSO_4 loads, reverted by
   exactly −X % (see §4). Residual base drift across the whole ladder is +1.2 %.

### 5.2 The +5 % row is NOT usable — do not cite it

Its droop and OFO components are each only 1–3 Mvar after baseline correction, so
the ratio is unstable (15 / 55 / 97 / 74 / 56 % across δ, non-monotonic), and at
δ = 0.015 its net response is anomalous (21.0 Mvar against 7.3 for the larger
+15 % step). **The sub-dead-band regime is therefore NOT characterised by this
study.** That is a real gap: it is exactly the regime in which the droop should
correctly stay dormant. Two ladder designs failed to resolve it. Report it as an
open item rather than interpolating.

### 5.3 Superseded data — do not use

Runs **0001–0007** are smoke/verification runs, several containing events that never
fired. Runs **0008–0012** are ladder v1: they used a revert of −X/(1+X) % on the
mistaken assumption that `EvtLod` percentages are relative to the present value. They
are additive on the original value, so the base load drifted **+75 %** across the
ladder and the rungs were not independent. Any table quoting +10 / +25 / +50 / +100 %
rungs or an outage row comes from that superseded set.

---

## 6. Do NOT

- Do not present δ = 0 as a candidate. It is not deployable and it measures as the
  worst tracking configuration; it appears only as a limiting reference.
- Do not justify δ = 0.005 by quasi-steady-state validity (see §1).
- Do not cite runs 0001–0007.
- Do not describe the ladder as chosen for coverage without noting that the outage
  anchor is a single fixed magnitude.
