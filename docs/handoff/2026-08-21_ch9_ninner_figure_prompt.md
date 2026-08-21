# Handoff prompt — the `N_inner` figure for Ch 9 §9.1 (TikZ/pgfplots)

Written 2026-08-21 on the server session that produced the data. Everything
between the rules is the prompt for the laptop session that owns the thesis
repo. It needs the three data files from
`docs/data/2026-08-21_ch9_ninner/` — copy that directory across first.

---

You are my research assistant on a PhD project on hierarchical multi-zone
reactive-power control. This session owns the dissertation LaTeX repo
(`...\12_Dissertation\latex_diss_ms`). **Your job is one figure** for §9.1
("Timescale Separation of the Control Layers"), plus the caption and the
paragraph of body text that reads it.

**Do not invent numbers.** Every value you need is in the data directory or in
this prompt. If something you want is not there, say so rather than filling it
in — the chapter has its own guard against unmeasured values reaching the text,
and this figure exists because that guard caught two of them.

## The data

Copy `docs/data/2026-08-21_ch9_ninner/` from the qOFO repo. Four files:

| file | contents |
|---|---|
| `README.md` | the numbers and their provenance — **read this first** |
| `ninner_cdf.dat` | empirical CDF, columns `n all_free unconstrained vconstrained`, `n = 0…46` |
| `ninner_hist.dat` | binned density, columns `lo hi mid all_free unconstrained vconstrained` |
| `ninner_steps.csv` | one row per measured step (240), for recomputation |

Both `.dat` files are whitespace-separated with a header row, ready for
`\addplot table`. The CDF columns **asymptote below 1.0**, by exactly the
never-flat fraction of each population. That is intended and the figure must not
hide it — see "the honest bits" below.

## What was measured

Two control layers, both MIQP online-feedback-optimisation:

- **TS-OFO** (transmission, EHV), period `T_TS`, issues an interface-Q setpoint
  `q_set` down to each subordinate network.
- **STS-OFO** (110 kV), period `T_STS = 20 s`, tracks that setpoint with local
  OLTCs and HV DER while also regulating its own voltages.

`T_TS = N_inner · T_STS`, and `N_inner = 9` was an *assumed* value used to
calibrate the weights `G_w`. Experiment B1 measures `N_inner` with those weights
in place, on the isolated subordinate loop with the supervisory parent frozen
(it solves once at `t = 0` and holds), stepping the interface-Q setpoint to the
edge of the reported capability band in both directions, over 10 design windows
× 4 subordinate networks × 3 interfaces × 2 directions = 240 steps.

**Criterion — this is the point of the figure.** `N_inner` is the first
subordinate iteration from which `|q_pcc(k) − q_pcc(k−1)|` stays below **1.0
Mvar**. That is *flatness*: the loop has stopped moving. It is **not** arrival at
`q_set`. Observation horizon is 900 s / 20 s = 45 iterations.

## The three populations

Read off `README.md`; reproduced here so you can check the plot against them.

| population | n | median | p90 | p95 | never flat | P(N ≤ 9) |
|:--|--:|--:|--:|--:|--:|--:|
| capability-free (all) | 207 | 4 | 18.7 | 25.0 | 1.4 % | 0.69 |
| …voltage-**un**constrained | 114 | 2 | 9.8 | 14.4 | 0.9 % | **0.89** |
| …voltage-**constrained** | 93 | 11 | 25.0 | 30.0 | 2.2 % | 0.44 |

A capability filter removed 33 of the 240 steps — those where the subordinate
layer's reported headroom in the commanded direction was under 5 Mvar, or where
the residual exceeded that headroom. They converge trivially fast (median 0),
because a step into a rail stops moving at once, so **excluding them raises the
median rather than lowering it**. Say that in the caption; the naive expectation
is the opposite.

"Voltage-constrained" means the step ran with a non-zero soft-constraint slack or
a voltage-bound violation at some iteration.

## The figure

One figure, `pgfplots`, matching the chapter's existing style (find a comparable
figure in the repo and copy its axis setup, fonts, and colour macros — do not
introduce a new palette).

- **Main axis:** empirical CDF of `N_inner` from `ninner_cdf.dat`. Three curves:
  `unconstrained`, `vconstrained`, and `all_free` (the pooled one lighter or
  dashed — it is context, the split is the message). `x` from 0 to 45,
  `y` from 0 to 1.
- **Mark `N_inner = 9`** with a vertical rule, labelled with the assumed value.
  The reader's eye should land on where that rule crosses each curve: **0.89**
  for the unconstrained population and **0.44** for the constrained one.
- **Mark the observation horizon at 45** so the right-hand end is visibly a
  measurement limit and not a plateau of the plant.
- Annotate each curve's asymptote with its never-flat fraction, or state both in
  the caption. A CDF that stops at 0.978 must not look like a drawing error.
- `ninner_hist.dat` is there if you prefer a small inset or a stacked companion
  panel. Use it only if it earns the space; the CDF carries the argument.

## What the surrounding text has to say

Three claims, in this order. The first two are the result; the third is the one
that is easy to get wrong and I want stated precisely.

1. **`N_inner = 9` is supported as a typical value, not as a coverage bound.**
   Median 4 over all capability-free steps, and 9 covers 89 % of steps where the
   subordinate loop is unconstrained. It is not a bound: p90 ≈ 19 over all free
   steps.
2. **Where 9 fails is a voltage result, not a timescale result.** Voltage-
   constrained steps go flat about five times more slowly (median 11 vs 2, p95 30
   vs 14.4). This connects §9.1 to the per-area voltage relief in §9.3 rather
   than arguing for a larger `T_TS`.
3. **Do not write that voltage-constrained steps "cannot converge".** They do:
   only 2.2 % never go flat, so a larger `N_inner` would capture them. What is
   genuinely unreachable regardless of `N_inner` is *arrival at `q_set`* — and
   that is unreachable for **both** populations almost equally (87.1 % vs 92.1 %
   censored, with the same ~11.5 Mvar steady-state offset). Voltage constraints
   do not explain it. The subordinate layer is multi-objective and its optimum
   sits at a non-zero interface-Q offset by design.

On (3): the supervisory parent was **frozen** for this run, so `q_set` is
constant throughout. The ~11.5 Mvar offset therefore cannot be a moving target or
integrator drift — it is the subordinate optimum itself. That is the cleanest
available evidence for why the flatness criterion is the right one, and it is
worth a sentence.

Related, and settled, so state it as a modelling choice rather than an open
question: the supervisory controller does **not** re-anchor its stored interface
request to the achieved value. The mechanism exists (`apply_avt_reset`,
"Achieved-Value Tracking", gain `k_t_avt`, 0 = no reset, 1 = full reset) and the
author has decided it stays off for this work. The loop still closes through the
measurement — each supervisory correction is computed from the measured interface
flow — so what is not re-anchored is the bookkeeping of the commanded value, not
the feedback. This is why "arrival at `q_set`" is not a meaningful convergence
test here and flatness is.

## The circularity — must be stated, not smoothed over

`G_w` was calibrated with `N_inner = 9` **assumed**, and this measures `N_inner`
with those weights in place. It is a fixed-point argument evaluated at one
iteration: the guess is used, the weights follow, and this tests whether the
guess survives its own consequence. Write it that way — as a check on the guess,
not as an independent measurement. The chapter should not claim more.

## The honest bits, none of which may be dropped

- The capability filter is built on the subordinate layer's own **reported**
  headroom, which is the quantity known to over-report (the capability message is
  voltage-blind). "Not capability-limited" therefore rests on a number that
  over-states available reactive power. One sentence in the caption.
- p90/p95 are resolved, not horizon-bound: only 1.9 % of free steps sit near the
  45-iteration horizon at this 1 Mvar tolerance. Worth stating, because at a
  tighter tolerance they would not be.
- The 1 Mvar tolerance is a **choice**, and it is the choice that makes the
  answer. It is the same band the open-loop settling battery uses as its absolute
  floor on interface flows, which is why it was picked — internal consistency
  within §9.1. At 0.1 Mvar the median would be 16–17 instead of 4. If the caption
  has room for one methodological sentence, make it this one.
- One operating point, one benchmark (IEEE 39-bus with four 110 kV sub-networks,
  `rural_700`), 10 of 12 design windows — two were excluded because DER reactive
  capability is structurally zero in them (VDE dead zone), so there is no
  capability band to traverse and no `N_inner` to measure. Excluded, not failed.

## Provenance for the caption

- Experiment B1, `ch_9_1_ninner_isolated_sts.py`, run `20260820-140643`.
- Commit `e5d1602` on `main`, working tree clean.
- Weights: campaign `stage1`, candidate `fe010aa3ead1`, rebuilt through the
  campaign's own recipe and asserted equal to the archived values.
- Parent-silent variant `frozen_ofo`; settle 600 s, observe 900 s, step 0.95 of
  the reported capability band; `T_STS = 20 s`.

## Working rules

1. Distinguish measured facts, hypotheses and open questions. Label anything
   projected or carried over. No invented numbers.
2. If something contradicts this brief, trust the data files and say so.
3. Answer as: short answer; assumptions; details; risks / open points.
4. The figure is the deliverable — TikZ source, caption, and the body paragraph
   that reads it. Do not restructure §9.1 around it.
