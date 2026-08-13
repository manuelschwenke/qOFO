# Handover — writing the dead-band results into the thesis

> **UPDATED 2026-08-04 — the recommendation changed.** This document was written
> when only ONE operating window had been measured and it recommends
> δ = 0.005 pu throughout. Two further windows **refuted** that: at the
> net-import window δ = 0.005 false-activates on 13.3 % of TS inter-dispatch
> windows, against 0 % at the two export windows. **The recommendation is now
> δ ≈ 0.01 pu.**
>
> The *reasoning* below (detector framing, keep/compress/drop of the older
> sections, pitfalls) is unchanged and still correct. The *numbers* in §3 are
> single-window. For current data, figures and numbers use
> **`docs/handover_deadband_n1_data.md`**.

**Task.** Write the DER Q(V) dead-band chapter using **only the most recent
approach** (the N-1 detector-threshold experiment) as the basis. The existing
`docs/ch8_deadband_selection.tex` contains several older experiments; they are
*motivation*, not results.

**Status 2026-08-03.** One operating window measured. A second window
(2016-12-18 14:00) is planned and will be run overnight; the values below are
provisional, the reasoning is settled.

---

## 1. The one narrative to write

> The Q(V) layer is a **disturbance-rejection** mechanism. Profile-driven
> voltage shifts belong to the OFO, which re-optimises every 180 s (TSO) /
> 20 s (DSO) and re-anchors each park's V_ref when it writes a setpoint.
> Event-driven excursions belong to the droop.
> Therefore δ is a **detector threshold**, not a quantity with an optimum:
>
> max(profile drift) ≲ δ ≲ min(event excursion)

Everything else in the chapter serves this. **δ has no optimum — it has an
admissible interval**, and the design choice inside that interval is a quantile
decision.

This also *explains* the earlier failure: the profiled sweep scattered with
δ\* CV = 0.715 across windows because an argmin was being fitted to a curve
whose flat region is the answer. Write that failure as **evidence for the
reframing**, in one short paragraph — not as an experiment with its own results.

---

## 2. What to keep, compress, drop

| Section in `ch8_deadband_selection.tex` | Action |
|---|---|
| §1 parameter & question, plant/actuators/outputs | **Keep**, trim |
| §Excitation | **Rewrite** — contingencies ARE now used; the old text says "no contingency is applied anywhere" |
| §2 `sec:rev2` sensitivity-reduction correction | **Keep in full.** Correctness precondition for every number; without it the DSO reductions sat on the wrong P–V branch |
| §3 `sec:windows` | **Compress** to the window actually used + keep `sec:degenerate` (parks at P=0 have zero Q capability) as a window-selection caveat |
| §4 `sec:sweep` profiled sweep | **Compress to one paragraph** — the CV = 0.715 failure, as motivation |
| §5 `sec:activity`, §6 `sec:threshold`, §7 `sec:rule` | **Drop or appendix.** Superseded. If kept, note the load-step design reached δ = 0.005 independently — a corroboration worth one sentence |
| §8 `sec:n1` | **This is the chapter.** Expand |
| §9 Threats to validity | **Keep**, already updated |

---

## 3. Numbers (quote exactly; do not round up)

Setup: window 2016-01-05 08:00 (+409 MW net infeed), `rural_700`, DSO_3 ×2,
physical VDE-AR-N-4120 capability, rev-2 sensitivities, RMS co-simulation,
600 s horizon, trip at t = 200 s (20 s *after* a TSO dispatch, maximising
droop-only exposure to 160 s). 8 dead bands × {twin, gen 7, gen 1} = 24 runs.

**Profile drift** (open-loop pool: 4 twins with δ ≥ 0.025; TS n = 48, DS n = 3360)

| | p90 | max |
|---|---|---|
| TS | 0.00285 | 0.00304 |
| DS | 0.00108 | 0.01101 |

**False activation** (windows in which the droop fires on ordinary drift)

| δ | TS | DS |
|---|---|---|
| 0.005 | 0/48 | 20/3360 (0.6 %) |
| 0.01 | 0/48 | 4/3360 (0.12 %) |
| 0.025 | 0/48 | 0/3360 |

**Rejection** — compensation `1 − peak(δ)/peak(δ=0.15)`, and min absolute V

| δ | comp gen 1 | comp gen 7 | V_min gen 1 | V_min gen 7 |
|---|---|---|---|---|
| 0 | 0.78 | 0.55 | 1.006 | 0.935 |
| 0.005 | 0.73 | 0.54 | 1.003 | 0.934 |
| 0.025 | 0.49 | 0.57 | 0.972 | 0.939 |
| 0.05 | 0.28 | 0.52 | 0.966 | 0.916 |
| 0.075 | 0.08 | 0.42 | 0.951 | **0.880** |
| 0.15 | 0.00 | 0.00 | 0.937 | **0.816** |

**Answer**

```
strict (zero false activation)   : δ ∈ [0.025, 0.05]
with 1 % false-activation budget : δ ∈ [0.005, 0.05]
recommended                      : δ = 0.005 pu
```

Justification for 0.005, in three parts:
1. **before the knee** — peak rises +20 % from 0.005→0.01, then +55 % to 0.025;
2. **retains 94 % (gen 1) / 98 % (gen 7)** of the compensation achievable at any
   dead band (ceilings 0.78 / 0.55 at δ = 0);
3. **effectively silent on profiles** — 0/48 TS, 0.6 % of DS windows.

Summary sentence:

> At δ = 0.005 pu the local layer ignores 99.4 % of profile-driven
> inter-dispatch variation, retains 94–98 % of achievable event compensation,
> and holds the worst credible N-1 at 0.934 pu.

**One dead band, not two.** TS admissible set [0.005, 0.075] *contains* the DS
set [0.025, 0.075] — DS binds, TS has margin. So δ_TS ≠ δ_DS is not required by
this evidence. It is a **null result at one window**, not a proof; the per-level
machinery exists and the 2D sweep is the natural follow-up.

---

## 4. The methodological result worth its own subsection

### 4a. The excursion is a TRANSIENT — state this explicitly

gen 7, worst park, droop disabled (twin flat at 1.0441–1.0466 throughout):

| t [s] | 179 | 190 | 195 | 220 | 260 | 300 | 600 |
|---|---|---|---|---|---|---|---|
| V [pu] | 1.0441 | 0.9132 | 0.8949 | 0.9024 | 1.0052 | 1.0389 | 1.0559 |
| \|ΔV\| | 0.0000 | 0.1315 | 0.1497 | 0.1424 | 0.0397 | 0.0060 | 0.0093 |

Peak 0.2240 at **10.5 s after the trip**; above 0.05 pu for **90 s**, above
0.01 pu for 118 s; **settled below 0.01 pu**. It is a temporal excursion the
control stack clears, *not* a sustained offset — and the settled value is
*smaller* than the QSS number, not larger.

**This strengthens the argument rather than weakening it:** the TSO re-optimises
every 180 s, so the whole excursion opens and closes *inside one TSO interval*.
Whatever is done about it there is done by the DSO layer and the local droop —
precisely the inter-OFO-step role the dead band governs.

### 4b. Quasi-static screening cannot select the severe case

| worst-park \|ΔV\| [pu] | static scan<br>(open loop, settled) | RMS, droop disabled<br>(closed loop, peak) |
|---|---|---|
| trip gen 1 (650 MW) | 0.0854 | 0.1039 |
| trip gen 7 (830 MW) | **0.0122** | **0.2240** |

**Do not present these as a ratio** — they are different quantities (an
unregulated post-outage power flow versus the peak of a closed-loop transient).
An earlier draft claimed "18× the predicted excursion"; that was wrong and has
been removed from the tex.

The claim that survives is about **ordering**: statically the larger machine
looks milder, because a distributed slack absorbs its 830 MW instantly;
dynamically it produces the deeper swing and is the only case reaching an
undervoltage violation. A severe case selected on the quasi-static plant would
have been the wrong machine.

General point: controllers may reason on a reduced or quasi-static model, but
the **evidence** must come from the plant that has the dynamics.

---

## 5. Assets

| | path |
|---|---|
| metrics | `results/deadband_n1/deadband_n1_metrics.csv` |
| figures | `results/deadband_n1/figures/fig_n1_{detector,admissible_gen{1,7},rejection_gen{1,7}}_<WINDOWTAG>.{pdf,png}` — one set **per window**, tag e.g. `20160105_0800` |
| analysis | `analysis/deadband_n1.py` (`--fa-tol` sets the tolerated false-activation rate) |
| figures src | `analysis/deadband_n1_figures.py` |
| sweep | `experiments/run_deadband_n1.ps1` |
| draft prose | `docs/ch8_deadband_selection.tex` §`sec:n1` — reuse, don't rewrite |
| design record | `docs/daily_log/08_2026/2026-08-02_deadband_n1_experiment_design.md` |

`\graphicspath` in the tex already includes `../results/deadband_n1/figures/`.

Regenerate everything:
```bash
python -m analysis.deadband_n1 && python -m analysis.deadband_n1_figures
```

Figures currently cited (both for window `20160105_0800`):
`fig_n1_rejection_gen7_*` (the knee) and `fig_n1_admissible_gen1_*` (the two
competing rates). `fig_n1_detector_*` is drawn but not yet cited — it is the
natural lead figure once several windows exist.

**Everything is keyed per window.** Metrics rows carry a `window` column, twins
and trip runs are keyed `(window, δ[, gen])`, and the open-loop drift reference
is computed per window — drift and event severity are both properties of the
operating point, so pooling them across windows would score every δ against a
distribution the plant never produced. The figure filenames carry the window tag
for the same reason. If a metrics CSV without a `window` column is encountered,
it predates this and must be regenerated.

---

## 6. Pitfalls — each of these would be caught in review

1. **Do not write "no false activation" for δ = 0.005.** It is 0/48 on TS but
   0.6 % on DS. Write "≤ 0.6 %".
2. **Do not quote the drift *maximum* as the bound.** A maximum grows with the
   observation horizon: the same window gives DS max 0.0014 over 300 s and
   0.0110 over 600 s. Use a quantile; δ = 0.005 is the 99.4th percentile.
3. **Do not reuse the static outage ranking** for anything (see §4).
4. **`exit=1` and "Gate E FAIL" are expected** for every N-1 run. The entry
   point returns Gate E's verdict, and Gate E validates QSS/RMS *equivalence*,
   which a topology change legitimately breaks. All 24 runs produced complete
   data. A real failure = no `rms_records.pkl`.
5. **Single window.** Do not generalise. Mark provisional until the second
   window lands.
6. **No measurement noise** anywhere. A dead zone's textbook justification is
   noise rejection, and that mechanism is absent — so every δ here is a *lower*
   bound for a noisy plant. Argued, not measured.
7. **The DS bound rests on 20 windows out of 3360.** Sensitive to which windows
   are drawn; this is the strongest argument for replication.
8. **Only 2 of 6 machines tripped** (gen 9 diverges in the static scan and is
   excluded). No N-1 completeness claim. Line outages are not deliverable by the
   RMS adapter yet.

---

## 7. When the second window lands

Re-run the two commands in §5 — the analysis pools windows automatically once
the runs exist. Then:

- replace every "provisional" marker with the two-window statement;
- report whether δ = 0.005 still sits below the knee and above the drift
  distribution at the new window — **that is the headline test**;
- if the DS drift tail at 2016-12-18 exceeds 0.005 appreciably, the
  false-activation figure rises and the recommendation may move to 0.01. Say so
  plainly rather than defending 0.005.
