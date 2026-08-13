# 2026-08-02 — Measuring the mechanism (A), refuting the normalisation (B), and designing the threshold experiment (E)

**Timestamp:** 2026-08-02
**Scope:** re-analysis of stored rev-2 runs; one new experiment queued.
No simulation was re-run for A or B.

---

## 0. Why: the profiled sweep cannot define δ\*

The rev-2 matrix (45 undisturbed runs, 5 windows × 9 dead bands, corrected
sensitivities) does **not** yield a dead-band optimum:

* no δ is Pareto-optimal in every window — the intersection is **empty**;
* the interface-Q argmin scatters over 0.0025 … 0.03 with no relation to
  loading (**CV 0.715**);
* the stressed windows' curves are non-monotone with 20–25 % scatter.

The scatter is *structural*, not statistical: the degenerate window
(2016-07-15 03:00, zero DER capability) reproduces two different δ to identical
digits, so the pipeline is deterministic. Different δ therefore reach different
equilibria — the dead-zone multi-equilibrium behaviour of
`docs/qss_rms_divergence_analysis.tex`, now dominating rather than appearing as
isolated anomalies.

Root causes in the design: one realisation per cell; a δ-dependent initial
condition (the Q(V) seeding uses the dead band); a metric that mixes transient
with steady state; and no measurement of the mechanism the dead band actually
governs.

## 1. Design A — measure the mechanism, not the end state

`analysis/deadband_activity.py` (new) reads each run's per-park record
`csv/rms_der_raw.csv` and computes actuator **traverse** (Σ|ΔQ|) and
**direction reversals**, per park per dispatch interval. These count motion, so
they cannot be confounded by which equilibrium a run settles into.

Direction reversals per park per interval:

| δ | −117 | +409 | +805 | +1367 | +2200 |
|---|---|---|---|---|---|
| 0 | 7.274 | 2.847 | 9.176 | 12.559 | 16.783 |
| 0.0025 | 4.944 | 1.873 | 3.698 | 5.685 | 8.952 |
| 0.005 | 4.639 | 1.517 | 2.718 | 4.552 | 6.995 |
| 0.0075 | 4.409 | 1.303 | 2.321 | 3.935 | 6.750 |
| 0.01 | **4.167** | **1.209** | 1.914 | 3.352 | **6.420** |
| 0.02 | 4.224 | 1.242 | **1.447** | 3.030 | 6.823 |
| 0.03 | 4.430 | 1.245 | 1.485 | **2.998** | 8.044 |

Monotone with a clear knee in **every** window — smooth, no oscillation. Defining
the knee as the smallest δ within 10 % of that window's floor:

| quantity | mean | CV |
|---|---|---|
| chatter knee | 0.0100 pu | **0.418** |
| interface-Q argmin | 0.016 pu | 0.715 |

**A works.** The chatter knee is ~2× better determined than the tracking argmin,
and it is the quantity the dead band directly controls.

## 2. Design B — refuted

Hypothesis: a dead zone is an amplitude threshold, so the meaningful abscissa is
δ/σ_V, and normalising should collapse the per-window curves. σ_V measured per
run from the DER terminal voltages:

| window | net MW | σ_V [pu] |
|---|---|---|
| 2016-01-05 08:00 | +409 | 0.00128 |
| 2016-02-22 13:00 | −117 | 0.00230 |
| 2016-01-15 03:00 | +805 | 0.00301 |
| 2016-12-18 14:00 | +1367 | 0.00347 |
| 2016-05-01 16:00 | +2200 | 0.00481 |

| quantity | CV in pu | CV in σ_V |
|---|---|---|
| chatter knee | 0.418 | **0.427** |
| interface-Q argmin | 0.715 | **0.747** |

Normalising makes both *slightly worse*. **δ is not simply a threshold in units
of local voltage variability.** Cost of the test: one re-analysis, no runs.

## 3. Design E — threshold characterisation (queued)

`experiments/run_deadband_threshold.ps1` (new). A dead zone is a property of the
*characteristic*, not of an operating point, so it should be measured by
controlled excitation rather than inferred from annual windows.

One fixed operating point (`2016-01-05 08:00` — the only window whose profiled
response is unambiguous, and the lowest σ_V so the step dominates the
background). For each δ ∈ {0.0025, 0.005, 0.01, 0.02}, sweep the exogenous
load-step amplitude over eight geometrically spaced factors (1.01 … 1.25) and
measure the DER response to the step. 32 runs, ~7 h.

Expected signature: negligible DER motion while the induced |ΔV| < δ, rising
response above it. Fitting that knee per δ gives the transfer characteristic
directly — and unlike an argmin over a bistable metric, it is a quantity the
dead band determines.

The instrument is the load-step mechanism added 2026-07-31, which perturbs the
interpolated profile frame and therefore reaches both plants through supported
paths.

## 4. Risks / unresolved points

1. E assumes the step's induced |ΔV| spans the δ set. If the smallest factors
   produce ΔV well above 0.02 pu, every dead band is exceeded and the threshold
   is not located — the amplitude range would need extending downward.
2. E measures one operating point by construction. Whether the threshold it
   finds transfers across operating points is a second question, and the
   profiled windows become *validation* rather than the means of selection.
3. The multi-equilibrium behaviour itself remains uncharacterised. Design D
   (ensembles over perturbed initial conditions per cell) would address it
   directly and is the natural follow-up if E's threshold proves
   operating-point dependent.
