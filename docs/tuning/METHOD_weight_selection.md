# Weight selection: the shortest path from theory to empirical values

Status: derived from the 0814 and 0815 campaigns
(`docs/daily_log/08_2026/2026-08-14_lambda_calibration_run_and_thesis_9_3.md`,
`docs/daily_log/08_2026/2026-08-15_campaign_0815_tiered_bank_and_recalibration.md`).
Every number quoted below is measured on `rural_700` with the Thevenin boundary
model; the *procedure* is plant-independent, the *values* are not.

> **Corrected 2026-08-18.** §1, §3 and §7 rule 6 previously described the
> empirical work as *three one-dimensional scans*, with `tau` among the
> coordinates a scan settles. That is wrong: `tau` rotates `g_w_der` against
> `g_w_pcc` and therefore **moves `lambda_max(B)`**, so it lies on the
> constraint surface together with `engage_tso_pu` and `lambda_tso`, and the
> three must be moved together on `rho = lambda_bar`. Only `lambda_dso` and the
> DSO pair are genuine one-dimensional scans, and that is verified rather than
> assumed. See `docs/handoff/2026-08-17_thesis_writeup_handoff.md` §6. The
> thesis (Appendix D, `ch:app_bo:space`) carries the corrected reading; do not
> re-derive it from an older copy of this file.

---

## 1 — The idea in one paragraph

The controller weights are never searched. They are **generated** by an analytic
design rule (Stage 0) from **six numbers that have engineering meaning**. Of
those six, three are statements in physical units that are chosen once and do
not move; two are loop gains fixed by measurement against a stated criterion;
one is an objective trade-off. So the empirical work is not a free optimisation
in six dimensions — it is **a joint move of three coordinates along one
constraint surface, plus two one-dimensional scans off it**, and which
coordinate belongs in which group is forced by measured properties of the
plant, not by convenience.

The split is the whole method, so state it plainly:

| group | coordinates | why | how settled |
|---|---|---|---|
| **on the surface** | `engage_tso_pu`, `lambda_tso`, `tau` | each moves `lambda_max(B)` | moved **together** along `rho = lambda_bar`; a compass search cannot find the optimum here |
| **off the surface** | `lambda_dso` | leaves `lambda_max(B)` bit-identical | one scan on `f_q` |
| **off the surface** | `engage_dso_pu`, `r_v` | leave `lambda_max(B)` bit-identical | scanned as a pair, Chebyshev choice on the three costs |

`tau` is the one that moves between groups against an earlier reading of this
document: it is not a pure allocation knob. Rotating `g_w_der` against
`g_w_pcc` changes the layer gain and not only its split.

---

## 2 — What theory fixes, and what has to be measured

Stage 0's curvature rule sets every weight from the cached sensitivity `H` once
the six knobs are given. It is *nearly* right and predictably wrong in one place:

| quantity | analytic | measured | gap |
|---|---|---|---|
| contraction floor (integer columns alone) | 1.1101 | 1.0929 | **1.6 %** |
| contraction slope in λ_TSO | 1.3221 | 1.5446 | **14 % low** |

**The rule knows the floor and understates the gain.** That is the entire
justification for measuring rather than computing λ: the intercept — what the
tap changers, shunts and the excluded AVR column impose on their own — comes out
of the model almost exactly, but how much contraction the continuous actuators
then add does not. No amount of analysis fixes a 14 % slope error; one scan
does.

Everything else Stage 0 emits is used as designed.

---

## 3 — Why the scans can be done one at a time, and in this order

Three measured facts. Each was checked more than once, on different banks.

**(i) `rho_emp_p95` is a function of λ_TSO alone.** Measured exactly constant —
to four decimals — across a **12.7x** move in `g_w_dso_der`, on three separate
experiments at three different λ_TSO values. So λ_TSO can be fixed first,
without knowing λ_DSO.

**(ii) λ_DSO is invisible to the supervisory criterion and identifiable from the
subordinate one.** Over its whole admissible range `f_ts` moves 0.29 % while
`f_q` moves by a factor of 2.4. So λ_DSO is fixed second, on `f_q`, and cannot
disturb step (i).

**(iii) A pattern search moves only one coordinate — and that is a warning, not
a licence.** Across two campaigns and a full compass search, `tau`,
`engage_tso_pu` and `engage_dso_pu` were never accepted as a move; only
`dso_g_v_ratio` was relocated, to the same value both times.

**Do not read that as "the others are already optimal". Measured 2026-08-17, it
is false.** A compass search polls ±δ in one coordinate at a time, so it cannot
find a direction whose gain needs two coordinates to move together:

* raising `engage_tso_pu` alone worsens `f_ts` → rejected;
* raising `lambda_tso` alone breaks the contraction ceiling → rejected;
* raising **both** improves `f_ts` *and* `f_q` and quadruples the stability
  margin → never polled.

The search rejected each half of a move that pays as a whole. **The coordinates
a local search leaves alone are exactly the ones that need a dedicated sweep**,
and "the search did not move it" is evidence about the search, not about the
coordinate.

`tau` is the sharpest case of this, and an earlier version of this section drew
the wrong conclusion from it. A dedicated sweep over {0.25 … 4} does put its
`f_ts` optimum at the analytic value of 1 — but that does not make `tau` a
settled one-dimensional coordinate, because the sweep moves `lambda_max(B)`
underneath it. `tau` rotates `g_w_der` against `g_w_pcc`, which changes the
layer gain, so a `tau` sweep at fixed `engage_tso_pu` and `lambda_tso` walks off
the constraint surface rather than along it. **Sweep it on the surface, jointly
with the other two, or the optimum you find is the optimum of a different
problem.** That the answer came out at 1 anyway is a fact about this plant, not
a licence to scan it alone next time.

---

## 4 — The procedure

Six steps. Roughly **20 simulated candidates** in total.

### Step 0 — analytic design (no simulation, ~60 s)

```bash
python -m tuning_mc.stage_0_preconditioning --scenario none --per-area
```

Set the three engineering knobs and leave them:

| knob | value | what it is a statement about |
|---|---|---|
| `tau` | 1.0 | DER vs PCC allocation; 1.0 = no preference |
| `engage_tso_pu` | 0.015 | the voltage error at which a TSO tap should commit |
| `engage_dso_pu` | 0.025 | the same, DSO |

These are operator statements in pu. They should be argued from the plant, not
tuned — and empirically they survive tuning untouched.

### Step 1 — λ_TSO, from the contraction criterion (~6 candidates)

```bash
python -m tuning_mc.stage_1_search --phase scan --scan-knob lambda_tso \
  --scan-values 0.10,0.15,0.20,0.25,0.40 --fix lambda_dso=1.0 \
  --scenario-set tier1 --limits tuning_mc/configs/limits_mc_v2_tier1.json
```

Fit `rho = floor + slope * lambda` and take the largest λ meeting the ceiling
**with the margin of §5.1**. Report the fit, not just the answer: the floor is
the physically meaningful half of it.

Measured 0815: `rho = 1.0929 + 1.5446 λ`, max residual 0.0100.

### Step 2 — λ_DSO, from the subordinate criterion (~6 candidates)

```bash
--scan-knob lambda_dso --scan-values 0.6,0.9,1.2,1.4,1.6,1.8 --fix lambda_tso=<step 1>
```

Take the interior minimum of `f_q`. **Two checks, both non-negotiable:**

* `rho` must come out *exactly* constant across the scan. If it does not, the
  layer separation is broken and the sequential procedure is invalid — stop.
* The optimum must be *interior*. If the best point is at either end, extend the
  grid. On 0815 the eight-point grid put the optimum at its own edge; extending
  to the bound turned it into a genuine minimum at 1.6.

### Step 3 — `dso_g_v_ratio`, from the trade-off (~3 candidates)

```bash
--scan-knob dso_g_v_ratio --scan-values 0.25,0.5,2.0 --fix lambda_tso=<1>,lambda_dso=<2>
```

(1.0 is already measured as the step-2 optimum, so three new points complete the
frontier.) This coordinate has no plant-derived value — it is the price of
interface-Q against the DSO's own voltage schedule — so report the frontier and
choose on a stated rule rather than presenting one point as optimal.

Measured 0815 at (0.25, 1.6):

| ratio | f_ts | f_q |
|---|---|---|
| 0.25 | 1.25357 | **0.06590** |
| **0.50** | **1.24946** | 0.06886 |
| 1.00 | 1.25502 | 0.08106 |
| 2.00 | 1.26621 | 0.09142 (infeasible) |

### Step 3b — one round trip (optional, ~3 candidates)

`f_q`'s λ_DSO optimum depends on `dso_g_v_ratio` — measured on both campaigns.
Re-running step 2 at the chosen ratio closes that loop. If it does not move
λ_DSO, say so; that is a result.

### Step 4 — confirmation (2 candidates, disjoint bank)

Evaluate the chosen point **and** the step-0/1/2 analytic baseline on a bank
drawn from the same distribution but disjoint calendar weeks.

**The rule: an in-sample improvement that does not survive is not carried
forward.** Report per capability stratum, never in aggregate (§5.3).

### Step 5 — wear audit (2 candidates, 12-h windows)

The only place a per-day switching budget may be evaluated. Report ops/h and
reversals/h **per transformer**, and split quiet windows from evented ones
(§5.2).

---

## 5 — Three corrections without which the procedure gives wrong answers

### 5.1 λ\* needs a margin for the sampling variability of a maximum

`rho_emp_p95` is reported as the **worst window** in the bank. A maximum over a
small sample is downward-biased as an estimate of the same quantity on a fresh
draw — so a λ\* calibrated to sit just under the ceiling in-sample will sit over
it out of sample.

Measured directly on 0815: the calibration margin was **1.7 %**
(`rho = 1.4743` against a ceiling of 1.5), and the worst-window figure moved
**3.1 %** between two banks drawn from the same distribution
(`1.4743 -> 1.5201`). The chosen point was **infeasible on the confirmation
set**, and so was the analytic baseline — it is the calibration that fails, not
the controller.

**Fix, either:** reduce the ceiling by the measured between-bank spread of
`rho_worst` before selecting λ\* (on this plant ~3 %, which returns λ\* to
≈0.20); **or** calibrate against a statistic that is stable on a small bank —
the p95 over windows, or the mean of the worst two — instead of the max.

Whichever is chosen, **state the margin as a margin.** The 1.5 ceiling is
already a declared 25 % margin below the OFO bound of 2; this is a second,
statistical margin and it serves a different purpose.

### 5.2 Wear and hunting need two limit tiers, because taps are integers

One tap in a window is the quantisation step: **0.667/h at 90 min, 0.083/h at
12 h.** A limit that falls between adjacent steps is not a constraint, it is a
coin toss.

| | 90-min window | 12-h window |
|---|---|---|
| step | 0.667 /h | 0.083 /h |
| 30 taps/day = 1.25/h | between 1 and 2 taps — **unusable** | 15 steps — fine |
| hunting 0.25/h | below 1 reversal — **unusable** | 3 steps — fine |

So the search bank carries a loose **chatter screen**
(`limits_mc_v2_tier1.json`, 4.0 / 2.0 — both ≥3 steps above zero) and the
**budget** (`limits_mc_v2.json`, 1.25 / 0.25) is applied only on 12-h windows.

Applying the budget to the search bank does not make the search stricter; it
makes it *stop*. Every candidate becomes infeasible, and a filter that rejects
infeasible candidates outright converges without moving.

**And a day is not a window.** 94 % of measured tap activity fell in the two
audit windows carrying an injected event. Quote "taps/day" from quiet
profile-driven windows, and report evented windows separately as a stress
figure.

### 5.3 Every stated controlled output must be in the filter

The filter compares candidates on `f_ts` (supervisory voltage) and `f_q`
(interface-Q). `f_ds` — the **DSO's own voltage cost** — is computed on every
candidate and classed as "a reported diagnostic, never optimised directly".

That is unsound. DSO nodal voltage is a *stated controlled output* of the
subordinate layer, and **a criterion outside the filter is one the search is
free to spend.** It did:

| `dso_g_v_ratio` | f_ts | f_q | **f_ds (DSO V)** | DSO taps/h |
|---|---|---|---|---|
| 0.25 | 1.2536 | **0.0659** | 0.4515 | 0.669 |
| 0.50 (searched) | 1.2495 | 0.0689 | 0.4306 | 1.338 |
| 1.00 (analytic) | 1.2550 | 0.0811 | 0.2925 | 2.007 |
| 2.00 | 1.2662 | 0.0921 | **0.1539** | 5.353 |

Moving `dso_g_v_ratio` 1.0 → 0.5 improves `f_q` 15 % and **degrades DSO voltage
regulation 47 %**. The "reduced tap wear" that step 5 reports as a benefit is
the same fact from the flattering side: the taps act less because the layer was
told to value its own voltages less.

**Two further consequences.**

*The trade-off knobs are redundant.* `engage_dso_pu` moves the identical axis
through the tap price instead of the objective weight. `dso_g_v_ratio = 0.5012`
and `engage_dso_pu = 0.05` land on the same operating point to within 0.3 % on
every criterion, both at 1.338 DSO taps/h. The space is effectively
five-dimensional here, and a search moves whichever it polls first.

*A wear screen inherits the baseline's bias.* The Tier-1 chatter screen was set
at "2× what the previous campaign did" — but that campaign had already given DSO
voltage away, so the screen rejects `dso_g_v_ratio = 2.0` on tap activity. **A
screen calibrated from a degenerate baseline forbids the region that fixes the
degeneracy.**

**Fix:** put `f_ds` in the filter as a third criterion, or as a constraint with
a stated bound, before running step 3. Otherwise step 3 optimises one controlled
output by spending another.

**And measure the trade-off on Tier 2, not Tier 1.** On the 12-h audit windows
the same comparison is **9 %**, not 47 % — the event-dense design bank
exaggerates this coordinate roughly fivefold. Only the 12-h figures are
quotable:

| `dso_g_v_ratio` | `engage_dso_pu` | f_ds | taps/day | reversals/day |
|---|---|---|---|---|
| 0.50 (searched) | 0.025 | 0.2377 | 26 | 2 |
| **1.00 (analytic)** | **0.025** | **0.2181** | **30** | 2 |
| 2.00 | 0.050 | 0.2187 | 30 | 2 |
| 4.00 | 0.050 | 0.1177 | 38 | 6 |
| 4.00 | 0.025 | 0.0377 | 52 | 32 |

Three conclusions, and the third is the useful one.

*The frontier inside the budget is flat.* `(1.0, 0.025)` and `(2.0, 0.050)`
give 0.2181 and 0.2187 at the same 30 taps/day. Nothing to gain by tuning inside
the budget.

*The searched value is dominated.* At `ratio = 0.50` the layer runs 4 taps/day
inside a budget that gives no reward for restraint, and pays 9 % of DSO voltage
for it. **Revert it: use the analytic `dso_g_v_ratio = 1.0`.**

*The wear budget, not the weights, is what limits DSO voltage regulation.*
Every materially better `f_ds` requires exceeding 30 taps/day, at a measured
exchange rate of **~5 % of `f_ds` per extra tap/day** over 30–52. The DSO taps
little because the switching budget permits little. Better voltage is bought
from the maintenance budget, and that is the asset owner's decision, not the
control designer's.

*The two levers are not interchangeable.* Raising `dso_g_v` and cheapening the
taps reach the same `f_ds` by different routes, but at `engage_dso_pu = 0.0125`
the reversal rate reaches 18–25/h on Tier 1 against 0.7–2.0/h at 0.05. **Recover
voltage through `dso_g_v`, never by lowering the commit threshold.**

### 5.4 Report per DER-capability stratum, never only in aggregate

Below `P/Sn = 0.1` the VDE curve gives **exactly zero** reactive capability, and
that is **18.6 %** of the profile year. In those windows `tau`, `lambda_dso` and
`dso_g_v_ratio` have nothing to allocate — and, measured on 0815, neither does
`lambda_tso`: the zero-capability windows returned **bit-identical `f_ts` and
`f_q` across a 9x move in λ_TSO**.

They contributed **35 % of the aggregate `f_ts` as a pure constant**. Excluding
them, the measured λ_TSO effect grows from 13.9 % to 21.9 % — **the aggregate
dilutes every effect by ~1.6x.**

Keep them in the bank (they are a real 19 % of the year) at their true share,
record each window's stratum, and report `f_ts` / `f_q` per stratum alongside
the aggregate.

---

## 6 — What this replaces, and what it costs

| | candidates | wall time at 12–16 workers |
|---|---|---|
| Steps 1–3 (three scans) | ~15 | ~1.5 h |
| Steps 4–5 (confirm + audit) | 4 | ~2.5 h |
| **Total** | **~20** | **~4 h** |
| Phase A probe + Phase B compass search (0815) | ~67 | ~6 h |

The pattern search is **optional**, and on 0815 it earned one move —
`dso_g_v_ratio` 1.0 → 0.50119 — which the three-point scan of step 3 finds
directly. Five of its six coordinates did not move at all.

Run the search when the aim is to *demonstrate* that the analytic construction
is not improvable, which is a real and publishable claim. Do not run it to
*find* the values.

At ~3.2 min wall per simulated hour per worker (measured on this server, rising
to 4.1 at λ = 0.9), one Tier-1 candidate is 18 simulated hours ≈ 58 min and one
Tier-2 candidate is 48 ≈ 155 min.

---

## 7 — Reporting rules

1. Quote λ_DSO **only with the λ_TSO and `dso_g_v_ratio` it was measured at.**
   Its optimum moves with both.
2. Quote `rho` as the fitted relation `floor + slope·λ`, not as a single number.
   The floor is what no continuous weight can reach.
3. Never convert a 90-min tap rate to a daily budget.
4. `f_q` is a time-weighted integral and scales ~quadratically with window
   length. It is comparable **within** a tier only — 0.089 on 90-min windows and
   9.6 on 12-h windows are the same controller.
5. Report the aggregate and the per-stratum split together.
6. State which of the six knobs moved and which did not — **and say which
   search was asked.** A coordinate that stays put under a compass search has
   not been shown to be optimal; it has been shown that no single-coordinate
   step improved it (§3). Only a coordinate held still by a sweep *on the
   constraint surface* is a result.
7. Name the baseline every improvement is quoted against. In the 0815 campaign
   it is the analytic point at `engage_tso_pu` 0.015 / `lambda_tso` 0.25 /
   `r_v` 1.0, **not** the weight set in service — that set was never simulated,
   and no candidate in the campaign carries its `g_w_der` = 20.
