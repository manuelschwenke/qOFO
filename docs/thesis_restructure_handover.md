# Handover: thesis restructuring (Part II/III)

**To:** session with access to `latex_diss_ms`
**From:** qOFO_GH code session, 2026-07-26
**Status of decisions:** agreed with the author unless marked OPEN.

## Why

Two contributions are currently under-signalled, and one evaluation chapter is
a grab-bag:

1. The **local Q(V) re-anchoring / offsetting** concept answers **Gap 1**
   (per §1.5: "the local-layer co-design of Gap 1") but sits inside
   §4.6 *"Actuator-Level Mechanisms Beyond the Joint MIQP Step"* — a title that
   reads as leftovers.
2. Controller **parameterisation** decisions (tuning, sampling rates, dead
   band, shunt engagement) are scattered across §4.7, §5.4 and §9.2 with no
   chapter that states them and their evidence together.
3. Old Ch. 5 (Single-System OFO, 8 pp, ending in "From Single System to
   Cascade") is a bridge, not a chapter.

## Target structure (12 chapters, unchanged count)

| new | chapter | action |
|-----|---------|--------|
| 4 | OFO: Method and Actuator Mechanisms | **Q(V) content removed** (→ new Ch. 5). **Keeps** OLTC handling, **shunt integrator**, switch notices, stability screening. §4.7 (Bayesian tuning) **moves out** → new Ch. 8. |
| **5** | **Adaptive Inverter Droop as Primary Voltage Control Layer** | **NEW.** Built from the Q(V) material extracted from §4.6. Gap-1 contribution. |
| 6 | Multi-System OFO for Hierarchical and Multi-Area Voltage Control | **Absorbs old Ch. 5.** Chapter now *opens* with the single-system DSO-OFO description (old §5.1–5.4), then vertical (BRC-V), then horizontal (BRC-H). Old §5.5 "From Single System to Cascade" becomes the hinge into the vertical section. |
| 7 | Benchmark Networks, Reference Models, and Evaluation Methodology | **+ dynamic IEEE 39 model** (move old §11.2 here — a benchmark's dynamic model belongs with the benchmarks). |
| **8** | **Parameterisation of the Control Hierarchy** | **NEW.** See section list below. |
| 9 | Case Study I | was Ch. 8 |
| 10 | Case Study II | was Ch. 9; **absorbs old Ch. 10** (remuneration / balancing) as a section |
| 11 | RMS-Dynamic Verification of the Quasi-Steady-State Premise | unchanged position; §11.2 moved out to Ch. 7; §11.7 reframed (below) |
| 12 | Conclusions and Outlook | — |

### Splitting principle for Ch. 4 vs Ch. 5 (use this if a case is ambiguous)

- **Ch. 5** = a distinct control **layer**: continuous-time, autonomous, acts
  *between* dispatch instants (the Q(V) droop).
- **Ch. 4** = **mechanisms of the OFO dispatch step itself**: how the OFO
  handles actuators that don't fit the joint MIQP (shunt integrator, OLTC
  rate/cooldown handling, switch notices). No existence between dispatches.

### New Ch. 8 — section list

1. Timescale selection: analytic settling bound from the modal analysis,
   validated per actuator class in RMS → fixes `T_DS`, `T_TS`.
2. Dead-band selection: RMS event study → handover between the primary
   inverter droop (Ch. 5) and the OFO (Ch. 4/6).
3. Shunt engagement: persistent-need threshold for the shunt integrator.
4. Empirical tuning via offline Bayesian optimisation (**moved from §4.7**,
   existing content, ~8 pp).

Only item 4 exists today. Items 1–3 are being produced in the code session;
**leave them as stubs with section headings** and do not invent numbers.

## Content notes

- **§11.7 reframe.** Currently "A *Deliberate* Violation: When Timescale
  Separation Fails". Measurements show the premise fails **at nominal
  parameters** on some operating points (same network, same settings:
  0.69 Mvar on a January morning vs 8.28 Mvar on a July night). An
  *inadvertent* violation at design settings is the stronger finding. Suggest
  "When Timescale Separation Fails" or "…: An Inadvertent Violation".
- **Ch. 5 ↔ Ch. 11 cross-reference.** The QSS validity condition is governed
  by how far voltage travels per dispatch interval, which the dead band
  controls. Ch. 11's design implications are therefore a parameterisation
  guideline for Ch. 5's own control law — link both ways.
- **Ch. 5 needs Part III machinery** (its value is invisible in QSS: in a
  quasi-static model everything settles instantly at each dispatch, so a
  droop whose job is bridging the inter-dispatch interval has nothing to do).
  Keep the *concept* in Ch. 5 with the dead band as a free parameter and
  forward-reference the study in Ch. 8.

## Mechanical work

- Renumber chapters 8→9, 9→10, 10→(section of 10), and all `\label`/`\ref`
  (`ch:case1`, `ch:case2`, `ch:case3`, …).
- Update §1.5 "Thesis Structure" — it currently enumerates chapters 4–12 in
  prose and will be wrong in several places.
- Update the Part II title if needed: currently *"OFO-Based Voltage and
  Reactive-Power Control: From Method to Multi-System Architecture"* — still
  accurate with the droop chapter inserted.
- Check §1.4.3/§1.4.4 forward references to Chapter 11 (they are correct and
  should stay — §1.4.3 already states the QSS premise is "tested rather than
  assumed", which is the reader-confidence hook).

## Do NOT

- Do **not** move Ch. 11 earlier. Its §11.4 consumes worst cases from the case
  studies, and Ch. 12 synthesises it into RQ3 design rules.
- Do **not** write results for the new Ch. 8 sections 1–3; they are pending.

## RESOLVED (author decision, 2026-07-26)

- **Ch. 8 title: "Parameterisation of the Control Hierarchy".** Use this.
- **The dead band IS a free design parameter**, within regulatory limits;
  δ = 0.005 and δ = 0.01 both lie clearly inside those limits. Ch. 8 §2 is
  therefore a **selection study**, not a verification. Frame the selection on
  **control-performance grounds** (tracking, actuator effort, handover with
  the OFO) — *not* on quasi-steady-state validity, which must appear only as
  a consequence in Ch. 11. Selecting a plant parameter to suit the simulation
  method would be circular and is to be avoided in the wording.
