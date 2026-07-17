# SBX-H v6 — architecture candidates for TSO–TSO reactive-power coordination

**Status: DISCUSSION DRAFT (2026-07-12). Nothing here is implemented.**
This document is self-contained so it can be given to external reviewers
(human or AI) without access to the codebase. Please critique freely;
review questions are listed at the end.

---

## 1. Context: the system and what has been measured

Setting: a multi-area transmission system (study system: IEEE 39, three
zones, CIGRE HV distribution networks at the EHV–HV interfaces). Each
zone/TSO runs an independent Online-Feedback-Optimisation (OFO) MIQP
controller every 3 min that dispatches AVR setpoints, OLTC taps, shunt
states, transmission-connected DER, and reactive-power setpoints to
underlying DSOs. Controllers never see the plant model — only their own
cached sensitivities and measurements. Controlled outputs: nodal
voltages within bounds and reactive flows at zone boundaries. Inter-area
ties are stiff: 40–75 Mvar of corridor flow change per mpu (10⁻³ pu) of
terminal-voltage difference.

The mechanism under evaluation, **SBX-H (Scheduled Boundary Exchange,
horizontal)**, has two layers:

1. **Contract layer**: agreed per-corridor boundary-voltage schedules
   v_std (from a planning power-flow pre-pass or a settled-state
   snapshot) that each zone tracks with high priority (20× the normal
   voltage-tracking weight), implying scheduled corridor flows q_std.
2. **Deal layer**: a runtime request/grant protocol — a zone with a
   persistent voltage-bound violation requests corridor-flow quanta
   (12 Mvar per 6-min cycle) from neighbours, matched against their
   LP-computed capability offers, executed as a terminal-voltage shift
   dv on the supporter's side, settled at fixed contract prices.

**Measured results** (controlled experiment "015": constructed
deficit × supporter-headroom matrix, arms = no-coordination /
contract-only / full mechanism; metric = violation exposure of the
stressed zone in pu·steps):

| scenario | no coordination | contract only | full (deals) |
|---|---|---|---|
| deep deficit (900 Mvar sink), supporters strong | 5.207 | 1.444 | 1.444 |
| moderate deficit (500 Mvar) | 1.458 | 0.119 | 0.119 |
| benign | 0.000 | 0.000 | 0.000 |

Findings that motivate this document:

- **G2:** the contract layer carries essentially ALL the value. Two
  causal channels: (i) priority-tracking the agreed boundary voltages
  redirects the stressed zone's OWN resources to the violated region
  (its default objective — uniform voltage tracking across all buses —
  had misdirected them); (ii) the neighbour's side of each tie becomes
  a firm, obligated voltage anchor, so support is deliberate and
  bounded rather than passive drift.
- **G3:** deal delivery is unverifiable at quantum scale: under deep
  stress the natural (physics-driven) corridor-flow shift is 100–300
  Mvar, so a commanded 12–48 Mvar schedule change cannot be confirmed
  from flow measurements, regardless of deadband tuning. A commanded-
  quantity market at this scale founders on ATTRIBUTION.
- **G4:** with verification disabled, stacked deals deliver a real but
  small acceleration (+0.166 pu·step ≈ 11 % on top of the contract
  layer) at a delivery ratio of 0.16 — i.e. the settlement would price
  96 Mvar of schedule for ~11 Mvar of realised flow change.
- **G7:** with an honest exhaustion test (arm requests only when the
  requester demonstrably cannot recover), the deal layer never arms in
  this system class: even the 900-Mvar case self-recovers under the
  contract layer alone within ~30 min (violation depth 20 → 0.6 mpu in
  five 6-min cycles). The deal layer is at best an acceleration option,
  never a feasibility necessity.
- Passive/obligatory support is LARGE and currently FREE: in the deep-
  deficit case, 184 Mvar (contract arms) to 310 Mvar (uncoordinated)
  of additional reactive power flowed toward the stressed zone across
  the ties — involuntary, unremunerated.

**The architectural question:** if commanded runtime deals are (a)
unverifiable, (b) physically marginal, and (c) never armed by an honest
trigger — what should the coordination-plus-remuneration architecture
be instead?

---

## 2. The design space

Two axes organise the candidates:

- **WHEN is coordination information exchanged?**
  planning time (day-ahead/hourly) · slow loop (re-planning on
  indicator) · control cadence (minutes) · never (physics only).
- **WHAT is priced?**
  nothing · sold capacity (ex ante) · attributed delivered support
  (ex post) · commanded quanta (the current deal layer) · marginal
  prices (dual signals).

The current SBX-H v5 = {planning + dormant control-cadence requests} ×
{commanded quanta}. The measured pathology sits precisely in the
"commanded quanta at control cadence" cell.

---

## 3. Candidate architectures

### A1 — Scheduled references + ex-post attributed remuneration ("SRS+EPR")

*The proposal under primary discussion (Manuel's suggestion, refined).*

**Control plane (unchanged from the proven contract layer, NO runtime
communication):** hourly v_std schedules per corridor terminal from a
shared planning pre-pass; every TSO priority-tracks its own terminals;
nothing else.

**Commercial plane (ex post, per 15-min metering window):** decompose
the realised corridor-flow deviation Δq = q_meas − q_std with the
per-line first-order decomposition already validated in the SBX-H
settlement engine:

    Δq ≈ C_A + C_B + C_P + residual
    C_A = Σ_lines s_A,l · (v_meas_A,l − v_std_A,l)     (A-side voltage state)
    C_B = Σ_lines s_B,l · (v_meas_B,l − v_std_B,l)     (B-side voltage state)
    C_P = Σ_lines s_P,l · (p_meas_l − p_sched_l)       (active-power transfer)

with sensitivities s from the contracted π-line model at the scheduled
operating point. Settlement rules (sketch):

- **Causer identification:** the side whose terminal-state term C_X is
  dominant and beyond a deadband is the CAUSER of the deviation
  (typically the sagging, stressed side).
- **Support remuneration:** the OTHER side is paid
  p_support · (attributable delivered Mvar·h) for the support it
  provided while the causer's disturbance persisted — where "provided"
  means: held its terminals at/above schedule (obligation tier, small
  or zero price) or actively raised them beyond schedule
  (over-performance tier, higher price). Payment flows causer →
  supporter.
- **Non-performance penalty (symmetric):** a supporter whose own
  terminal state sagged below schedule during the neighbour's stress
  (C_B in the aggravating direction without being a causer itself)
  pays instead of earns.

**Optional autonomy hook:** each TSO MAY add an economic term to its
own OFO objective — revenue for over-holding its boundary voltages
while its tie EXPORT exceeds q_std (the tie flow itself signals
neighbour stress; the physics is the communication channel). Operators
with spare resources then earn by regulation effort, exactly as
proposed. No message exchange is required for this; the price schedule
is static contract data.

**Why this dissolves the G3 attribution problem:** nothing is
commanded, so nothing must be verified against a counterfactual. The
question changes from "did my 12-Mvar order arrive through a 300-Mvar
storm?" (unanswerable) to "how does the realised deviation decompose
into each side's measured terminal state?" (answerable per line from
telemetry both sides already exchange; the decomposition closed with
zero unattributed cycles in the validation campaigns).

**Known risks / gaming vectors:**
- *Baseline inflation:* a party could bias the agreed v_std low so its
  normal operation looks like paid over-performance. Mitigation: v_std
  comes from a JOINT planning power flow (both parties compute and
  cross-check it — the machinery exists), plus deadbands sized from
  forecast-error ensembles.
- *Voltage creep:* an always-on over-performance price rewards running
  the boundary high. Mitigation: pay over-performance only in windows
  where a causer-side deviation exists (conditional price), and cap at
  the planning voltage limits.
- *Small money:* in the study system the marginal physical headroom
  between obligation and over-performance is small (G4: ~11 Mvar
  realised). The larger flows (the 184–310 Mvar) are obligation-tier;
  whether the obligation tier is priced (fairness for asymmetric
  burdens; degraded-neighbour compensation) or free (mutual-assistance
  convention, as between today's synchronous-area TSOs) is a POLICY
  choice, not a control question. Both variants are implementable.

**Regulatory resonance:** this is structurally the German
Leitfaden-Blindleistung construction (measured Q exchange, four-quadrant
metering, band + ex-post pricing with causer/deliverer roles) applied
HORIZONTALLY to tie-lines instead of vertically to TSO–DSO interfaces —
and the author's vertical mechanism (SBX-V) already implements exactly
that philosophy (no gating, priced dispatch, ex-post settlement,
persistent-exceedance indicator as feedback into planning). A1 unifies
the horizontal and vertical designs into one family.

### A2 — Capacity market at planning time + automatic activation ("options")

Ex ante (day-ahead/hourly): areas SELL boundary-support capacity —
mpu-bands around v_std (equivalently Mvar-bands around q_std, converted
by the contracted stiffness) — sized from the same forecast-error
ensemble machinery that already exists for band sizing. Ex post:
activation is automatic physics (the neighbour's sag pulls on the held
voltage anchor); remuneration = capacity price (Leistungspreis) for the
sold band + energy price for attributed activation (metering as in A1).
No runtime negotiation; the "deal" moved entirely into planning, where
verification is trivial (the band either was held or not).

- Pro: gives the INVESTMENT/availability signal A1 lacks; matches the
  Vorhalteleistung planning-product logic of E VDE-AR-N 4141-4.
- Con: requires the planning/forecast plane (deliberately out of the
  thesis scope); adds a market-clearing problem.
- Natural role: the "full product" sketch in a discussion chapter, with
  A1 as the operable core the thesis demonstrates.

### A3 — Runtime price signals (dual/marginal-price exchange)

Each area publishes a boundary marginal price (dual of its voltage
constraints) every control cycle; neighbours add it to their
objectives. This is the previously implemented BME mechanism. Evidence
from its own Monte-Carlo campaign: benefits appear under stress,
mechanism is fragile to design details (an "oracle" variant inverted on
random scenarios; zero-delay coupling unstable), and it prices
INTENTIONS rather than deliveries. Listed for completeness as the third
pole of the design space; not recommended for the thesis (already
descoped).

### A4 — Contract layer + re-planning escalation (replace deals with plan updates)

Keep the contract layer untouched. Remove the runtime deal protocol
entirely. Escalation path: when the persistent-exceedance /
persistent-violation indicator fires (it exists and is validated), the
CONTRACT ITSELF is re-negotiated at slow cadence — re-run the planning
pre-pass from the current system state and issue a new v_std interval
(the hourly-schedule machinery already supports piecewise contracts).
The stressed epoch is then handled by a new agreed operating point that
shares the burden, rather than by trading quanta against the old one.

- Pro: consistent with EVERYTHING measured (the plan being wrong is the
  actual root cause in the constructed deep-stress cases); zero
  unverifiable commands; communication at minutes-to-hour cadence where
  verification is easy; the v_std-schedule plumbing already exists.
- Con: slower than a deal could theoretically be (bounded by the
  re-planning latency); needs a governance rule for "who may trigger a
  re-plan and how the burden is shared" — which is a governance
  question anyway.
- Combines naturally with A1: A4 fixes the reference when it is wrong;
  A1 prices the deviations around a correct reference.

### A6 — Boundary voltage-droop characteristic: buying STRENGTH (added 2026-07-13)

*Motivated by Manuel's question: is "raising the boundary voltage"
really the right service, or should the buyable product be the
STRENGTH of the supporting grid?*

**Definition.** The supporter sells, per corridor terminal and window,
a **boundary characteristic** with three transparent numbers:

    v_term(Q_extra) ≥ v0 − Q_extra / k_s     for 0 ≤ Q_extra ≤ Q_cap

* ``v0``   — the intercept (level): the scheduled terminal voltage,
  optionally raised (= the existing planned-support product);
* ``k_s``  — the slope (STRENGTH) in Mvar/mpu: how firmly the terminal
  is held while the neighbour draws on it — a voltage-droop guarantee;
* ``Q_cap`` — the capacity limit up to which the characteristic holds.

Level and strength are complementary, not substitutes: the level moves
the operating point (efficient when the stress is FORECAST — a
scheduled, energy-like product); the strength bounds the response to
the UNSCHEDULED (an option-like product that delivers automatically,
in proportion to the actual disturbance, with no forecast needed).
The v6 planned-support product is the special case k_s unspecified —
which the measurements below show is a real contract gap.

**Empirical motivation (015/D2, deep-deficit cell, measured from the
existing per-sample interface telemetry):** during the stress window
the two supporters delivered against their (identical-form) contracts
with very different anchor quality —

| supporter side | extra flow delivered | own-terminal sag below promise | effective stiffness |
|---|---|---|---|
| corridor (2,3), zone 2 | +150 Mvar | 2.6 mpu | ≈ 17 Mvar/mpu |
| corridor (1,3), zone 1 | +54 Mvar  | 8.0 mpu | ≈ 3.6 Mvar/mpu |

Under the raised-voltage (planned-support) arm, zone 1's sag grew to
10.2 mpu — part of its raised promise "evaporates" into sag, while
zone 2 delivers most of it.  Today both are treated identically: the
contract specifies the level but not the firmness, so a weak anchor
and a strong anchor sell the same product.

**Why NOT contract the objective-function weight.**  In the
implementation the realised strength emerges from the tracking weight
(w_track × g_v) competing with every other objective term, times the
available actuator reserves.  That weight is internal, dimensionless,
non-portable between TSOs and non-monotone across systems — exactly
the kind of quantity a contract must not reference.  The droop curve
is the OBSERVABLE consequence at the interface: verifiable per sample
(is v_term on/above the promised line for the measured Q_extra?),
with no excitation problem (quiet windows are trivially compliant and
only the capacity fee applies — the Vorhalteleistung logic).  Each
TSO meets the promised curve by whatever internal means it likes
(weight tuning, reserve allocation); the contract prices outcomes,
not controller internals — the same principle that resolved the
delivery-verification failure of the deal layer.

**Pricing (maps onto the SBX-V structure):** capacity fee for
(k_s, Q_cap) per window = the Vorhalteleistung analogue and the
investment/availability signal; the attributed Mvar·h actually drawn
under the characteristic = the Arbeitspreis analogue (the existing
C_B attribution measures it).  Regulatory resonance: Q(U)
characteristics at network connection points are established German
TSO practice; k_s is dimensionally the familiar Mvar-per-voltage
stiffness operators already reason with (the study ties themselves
are 40–75 Mvar/mpu).

**Implementation sketch (NOT implemented; pending review):**
1. Verification-first: settlement checks the droop line per sample
   from existing telemetry (small settlement extension; the contract
   dataclass gains k_s/Q_cap per side).
2. Optional controller support: tracking a droop is tracking one
   synthetic linear output V_term + Q_tie/k_s — a linear combination
   of two existing output rows (small, contained controller
   extension).
3. Planning-time sizing: what (k_s, Q_cap) can a supporter honestly
   sell?  The archived capability-LP machinery is exactly this
   computation, moved from runtime protocol to planning tool.

**Additional review questions:**
7. Is the pointwise droop-line compliance test robust against fast
   transients (should compliance be on cycle averages, quantile-based,
   or sample-wise with a dwell)?
8. The measured terminal characteristic conflates the supporter's
   control effort with the passive network stiffness behind the
   terminal.  For the BUYER this is irrelevant (the total interface
   characteristic is what supports it) — but should the seller be
   allowed to sell passive stiffness it does not actively provide?
9. How should the planning pre-pass split a support need between
   intercept (v0 raise) and slope (k_s) — is there a natural
   optimum, e.g. intercept for the forecast component and slope for
   the forecast-error band?

### A5 — Penalty-only discipline (degenerate baseline)

No payments for support; only penalties for violating one's OWN
boundary schedule (causer pays). Support is unpriced obligation.
Simplest possible commercial layer; loses the effort/investment
incentive. Useful as the null-hypothesis arm in any economic
evaluation.

---

## 4. Recommendation for discussion

**A1 + A4 as the SBX-H v6 core** (with A2 sketched as the product
extension):

- Control: scheduled boundary voltages, tracked with the ordinary voltage
  weight by default. Higher priority is an explicit sensitivity only. (Proven:
  +3.76 / +1.34 pu·step vs no coordination; exactly 0 cost when idle.)
- Escalation: plan renegotiation on the persistent-exceedance
  indicator (A4), not runtime quantum trading.
- Commerce: ex-post attributed settlement (A1) on the existing
  C_A/C_B/C_P decomposition — obligation tier and conditional
  over-performance tier; policy switch for whether the obligation tier
  is priced.
- Deleted relative to today: the request/grant/matching/offer
  machinery and its capability LPs at control cadence (retained only
  as an appendix/negative result: "commanded-quantity exchange is
  unverifiable at quantum scale on stiff AC ties").

Evaluation plan (the existing 015 harness carries over): arms = none /
contract-only / A1(+autonomy hook) / A1+A4 on the same deficit matrix;
metrics = violation exposure (physics must stay ≥ contract-only),
payment flows, attribution shares, gaming probes (biased-baseline arm),
and a re-planning-latency sweep for A4.

---

## 5. Questions for reviewers (human or AI)

1. A1's causer/supporter identification rests on a first-order per-line
   decomposition around the scheduled operating point. Under what
   conditions (deep voltage excursions, P-flow reversals, topology
   changes) does the linearisation mis-attribute, and what safeguards
   (re-linearisation cadence, residual thresholds, UNATTRIBUTED-cycle
   handling) would you require before money flows on it?
2. Is pricing the OBLIGATION tier (the large passive support flows)
   fair compensation for asymmetric burdens, or does it create perverse
   incentives to under-invest in one's own reactive resources and
   "lean" on neighbours? Which side does European practice (mutual
   emergency assistance vs priced redispatch) suggest?
3. The conditional over-performance price pays B only while A is a
   measured causer. Does this create an incentive for B to DELAY its
   support until the deviation is deep enough to be classified (a
   threshold-gaming problem), and how would you shape the price/deadband
   to avoid it?
4. A4 replaces runtime deals with re-planning. What is the right
   trigger/governance rule so that a party cannot force re-plans
   strategically (e.g. to reset an unfavourable schedule), and should
   the re-planned schedule be burden-sharing-optimal (ORPF) or
   minimal-deviation from the old one?
5. Is there a fourth architecture we are missing — in particular, any
   scheme that keeps runtime quantity-commands but solves the
   attribution problem at 10-Mvar scale against 100–300-Mvar natural
   flow shifts on stiff ties (measurement-based counterfactuals,
   probing signals, PMU-based state attribution)?
6. Voltage-stability angle: paying for boundary-voltage over-holding
   pushes operation toward higher voltage profiles. Under which
   contingency assumptions is that unambiguously beneficial, and where
   would it need a cap or a reactive-reserve-margin condition?

---

*Provenance: derived from the 015_SBX_COMPARE campaign (findings
G1–G7), the SBX-H v4→v5 redesign, and the SBX-V (vertical) design
decisions; full quantitative record in STATUS_SBX.md and
results/015_SBX_COMPARE/ of the qOFO repository.*
