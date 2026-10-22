# 2026-07-26 — RMS generator-outage events + dead-band handover study

## Context / reason
Thesis restructuring (see `docs/thesis_restructure_handover.md`) introduces a new
Chapter 8 "Parameterisation of the Control Hierarchy", whose §2 selects the DER Q(V)
dead band from the **handover between the primary inverter droop and the OFO**. That
requires disturbance events inside a closed-loop RMS co-simulation. The author
confirmed the dead band is a free design parameter within regulatory limits, with
δ = 0.005 and δ = 0.01 both clearly admissible, so §2 is a **selection study** framed
on control performance — not on QSS validity (that stays a consequence, in Ch. 11).

## Blocker found
`experiments/runners/multi_tso_dso.py:2710-2726` raises
`NotImplementedError: non-static plant does not support: contingency events`, and the
plant's event inventory was `EvtParam` / `EvtLod` / `EvtTap` only — no switching or
outage event. So generator outages were not deliverable into an RMS run at all.

## What was done

### Probe: is an outage deliverable? (`scratchpad/outage_probe.py`)
All four candidate classes can be created (`EvtOutage`, `EvtSwitch`, `EvtGen`,
`EvtShc`). `EvtOutage` on `ElmSym` **fires and the system responds**:

| quantity | before | after |
|---|---|---|
| target `G 10` `m:P:bus1` | 131.9 MW | *(attribute gone)* |
| witness `gen[1]` `m:P:bus1` | 555.5 MW | 574.9 MW (+19.4, governor pickup) |
| park terminal `m:u` | 1.01927 pu | 1.01379 pu (−0.0055) |

PF output window logs `evt - (t=05:000 s) Grid\G 10.ElmSym:`.

The two-part verdict (target removed **and** system reacted) was deliberate: "the
result attribute vanished" alone would also be consistent with PF applying a *static*
out-of-service flag with no dynamic event, which would silently produce an outage
study containing no outages.

### Landmines recorded
* **`p_target` must be set** or PF silently does nothing — the same trap that made
  `EvtLod` look permanently broken in 2026-07-20.
* **`outserv` is NOT updated by the event** (still reads 0 afterwards), exactly as tap
  positions are not updated by `EvtTap`. Detect the outage from the **disappearance of
  the `m:` result variables**, never from `outserv`. Any readout of a possibly-outaged
  element must be defensive (`GetAttribute` raises).
* Events created inside an active calculation need
  `_new_events_pending_admission += 1` (admission barriers), as for `EvtParam`.

### Implementation
`pf/screening.py`: new `ScreeningContext.add_outage_event(target, t_event, seq=0)`,
placed after `add_tap_event`, with the landmines in the docstring. No persistent pool
— outages are rare one-shot disturbances, not per-dispatch commands, so the
create-fresh + admission-barrier path is used. Compiles clean.

The runner's contingency guard was **left untouched**: the study driver injects events
through the plant directly, so no production control path changes for an experiment.
Lifting that guard is a separate task if §9.7 (N-1) later needs it.

## Study design (not yet run)
Question: where should the handover between the autonomous local droop and the OFO
sit? Too narrow a dead band and the fast local law pre-empts deviations the OFO would
have handled optimally (measured: δ=0 gives the best voltage containment but the
**worst** interface tracking, 2.85 vs 1.10 Mvar); too wide and nothing acts between
dispatches.

* Disturbances must be **swept in magnitude**, not maximised. A generator outage moves
  voltage ~0.02–0.05 pu, which swamps the 0.005-vs-0.01 difference (<20 % of the
  driving signal) and would show no dead-band sensitivity. The discriminating regime is
  deviations comparable to δ (at 0.015 pu, δ=0.005 leaves 0.010 pu of effective driving
  vs 0.005 for δ=0.01 — a 2× difference in local response).
* δ grid: dense over the decision range 0.005–0.01, with 0.015/0.02 as failure-side
  anchors and δ=0 as a limiting reference (not a candidate).
* Metrics per event: **containment** (max |ΔV| before the next dispatch), **tracking**
  (interface-Q error at the next dispatch), **effort** (DER-Q travel), and the **split**
  between droop-delivered and OFO-commanded Q — the handover itself.

## 2026-07-28 — EvtLod unavailable; outage is the only usable disturbance class

`EvtLod` does **not** execute in this configuration. Established by direct
measurement of the load's own `m:P:bus1` (not inferred from bus voltage, which
the droop and OFO can mask), across three probe variants:

| variant | result |
|---|---|
| mid-calculation, no preallocated slots | 22.307 MW → 22.307 MW |
| mid-calculation, slots preallocated pre-ComInc | 22.307 MW → 22.307 MW |
| armed a full dispatch interval ahead | 22.307 MW → 22.307 MW |

PF logs nothing in any case. The third variant matters: arming one interval
ahead is exactly what made `EvtOutage` work (it had failed for that reason), so
the timing defect was ruled out rather than assumed. Note the profile path
verified `EvtLod` on 2026-07-21, so the mechanism is not broken in general —
but it is not usable through this call path, and the cause is unidentified.

**Consequence for the study:** the disturbance ladder rests on machine outages
alone — 3 magnitudes (0.0079 / 0.0142 / 0.0253 pu, gens 1 / 7 / 0), and they are
not revertible, so one disturbance per run. This is coarser than designed: no
sub-dead-band case (~0.0025 pu) is reachable, so the study cannot show the
regime in which the droop *correctly* stays dormant. State this as a constraint
of the disturbance class in Ch. 8 rather than presenting the ladder as chosen.

**Also fixed (production):** `pf/plant.py::read_y` could not survive an
out-of-service element — it read `m:P:bus1` unguarded for every machine, so the
run crashed the moment an outage landed (this cost 5 sweep runs, ~90 min,
because the sweep was launched without checking run 1). Now tolerant via
`_read_m`, deliberately narrow: an element that has never read successfully
re-raises (a handle/attribute defect cannot be an outage), afterwards it is
recorded, warned once, read as 0.0, and cleared again if it returns. Exposed as
`plant.out_of_service`. Needed for thesis §9.7 N-1 independently of this study.

**Also fixed:** events must be armed a full dispatch interval before they are
due; arming inside the advance that already contains the event time leaves the
admission barriers no room and the event lands late or not at all. The sweep
driver now arms at `t_event - dt_s` and confirms after the event time.

## 2026-07-28 — RESULTS: the handover is tunable and monotonic in the dead band

Root cause of the event failures: **the first two persistent-pool slots per target
are dead** — leftovers discovered from earlier runs in the persisted PF project,
already fired, which PF refuses to re-arm (`warn - Modification of events that have
already been executed is not allowed`). Diagnosed by ORDINAL, not time: run A failed
at t=65/105 and fired at 145/185; run B failed at *145/185* and fired at 225+. Fixed
in the driver by arming two zero-magnitude sacrificial events first. **Proper fix,
not yet done and affecting all callers incl. Gate E: `prepare_persistent_event_pool`
should skip or reset already-fired discovered slots instead of handing them out.**

Sweep: 5 runs, delta in {0.0025, 0.005, 0.0075, 0.01, 0.015}, each with the full
ladder (+10/25/50/100 % area steps on DSO_4, reverted) plus a 250 MW outage anchor.
Droop share = |dQ| accumulated between dispatches / total |dQ| (qset is constant
between dispatches, so that change is the autonomous droop; the step at a dispatch
instant is the OFO).

| event | 0.0025 | 0.005 | 0.0075 | 0.01 | 0.015 |
|---|---|---|---|---|---|
| +10 % | 29 % | 33 % | 46 % | 50 % | 47 % |
| +25 % | 53 % | 33 % | 22 % | 5 % | 0 % |
| +50 % | 78 % | 65 % | 52 % | 41 % | 27 % |
| +100 % | 80 % | 76 % | 68 % | 62 % | 54 % |
| outage | 51 % | 50 % | 48 % | 47 % | 46 % |

1. **The handover is tunable and monotonic** for moderate-to-large disturbances
   (+50 %: 78 → 27 % as the dead zone widens). Narrow dead zone ⇒ the fast local
   layer does the work; wide ⇒ it defers to the OFO.
2. **Large disturbances do not discriminate.** The outage row is flat at 46–51 %
   across every dead band — it clears all of them immediately. The originally
   proposed "biggest possible events" design would therefore have produced a NULL
   result and the false conclusion that the dead band is irrelevant. This vindicates
   sweeping magnitude instead.
3. **Containment degrades with delta**, sharply at the wide end: the +10 % event goes
   0.0091 → 0.0275 pu between delta=0.0025 and 0.015.

Together with the earlier tracking measurement (delta=0.005 gives -33 % interface-Q
error at unchanged actuator effort), two independent measurements favour 0.005.

Caveats: reverts do not exactly restore (ZIP voltage dependence + incremental-% of a
drifting present value), so later rungs sit on a higher base — each event is measured
against its own pre-event state but absolute magnitudes creep. The +10 % row is
non-monotonic (29→33→46→50→47 %), most likely noise at that size; do not build an
argument on it. One operating point, profiles off, DSO_4 only.

## Risks / open
- Load steps (`EvtLod`) and outages (`EvtOutage`) both available; outage magnitude is
  not freely tunable, so the magnitude sweep will rest mainly on load steps with
  outages as the large-disturbance anchor.
- The study measures the RMS plant, which is the right reference for a design decision;
  the QSS validity consequence belongs in Ch. 11 and must not be used as justification.
