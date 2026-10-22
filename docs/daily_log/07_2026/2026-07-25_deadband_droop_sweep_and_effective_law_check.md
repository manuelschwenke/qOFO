# 2026-07-25 — Deadband/droop parametric sweep + effective-law equivalence check

## Context
User questions on the DSO_4 static-vs-RMS divergence: (1) what makes the deadband special —
could a plain droop also have two equilibria? (2) what makes the first profile step special?
(3) try steeper/shallower droop and ±0.005 deadband. Then: did I verify the *effective*
steady-state RMS law (filtered/remote/compensated voltage, bias, mode switch), not the label?

## Theory (Q1/Q2)
- A smooth, monotone droop `Q=qset−R(V−Vanchor)` with monotone network `V=V0+S_VQ·Q`
  (R,S_VQ>0) has a UNIQUE equilibrium: `Q(1+R·S_VQ)=const`, coefficient never zero. Steeper
  droop only raises the loop gain `R·S_VQ`; if >1 the fixed-point ITERATION oscillates but the
  equilibrium stays unique. So a plain droop does NOT get two equilibria from gain alone
  (needs a network fold — voltage collapse — which isn't the case).
- The static map here is also single-valued (seed experiment: run_control seed-independent).
  So the divergence is NOT two equilibria of one map — it is the TWO INDEPENDENT CLOSED LOOPS
  (static-OFO, RMS-OFO) diverging in **qset**, with the deadband's non-smoothness (a gain
  discontinuity: dQ/dqset≈1 in-band, droop-on out-of-band) letting two loops at slightly
  different operating points be amplified differently. db=0 smooths it → they track.
- Q2: the first step is only where divergence BEGINS (plants identical at t=0). The t20 kick
  partially recovers (DSO_4 −4.5→−1.6), then a slow growth takes over — DSO_4's voltage sweeps
  1.02→1.05 (3× the deadband), so it spends the run near the kink; DSO_1/2/3 self-correct.

## Parametric sweep (co-sim, db=0.01 unless noted, oltc=200, 300 s, ±1.0 override)
CLI flags added: `--der-slope` (both plants via config.tso/dso_qv_slope_pu → tag_der_q_modes →
net.sgen.qv_slope_pu → static QVLocalLoop AND RMS snapshot/QVPRE; no override needed).

**Deadband width — DSO_4 peak static-vs-RMS DER-Q gap:**
| deadband | 0 | 0.005 | 0.01 |
|---|---|---|---|
| peak\|gap\| | 0.6 | 2.6 | 9.9 |
Super-linear in width; sign flips between 0.005 and 0.01. Runs: 0058 / 0061 / 0059.

**Loop gain (slope) at db=0.01 — DSO_4 peak gap (and DSO_1/2/3):**
| slope | 0.03 (2× gain) | 0.06 (1×) | 0.12 (½×) |
|---|---|---|---|
| DSO_4 | 17.2 | 9.9 | 4.1 |
| DSO_1/2/3 | 40/37/32 | 8/8/6 | 6/5/4 |
~Linear in loop gain R=Sn/slope; steep slope blows up ALL DSOs (loss of self-correction).
Runs: 0064 / 0059 / 0065. **Conclusion: divergence = loop-gain amplification of a seed,
gated by the deadband non-smoothness. Both width and gain drive it, monotonically.**

## Effective-law equivalence — VERIFIED from code (user's suspects)
QVPRE DSL (pf/wecc_apply.py): `x1.=(u−x1)/Tf; veff=x1−Vanchor; qcorr=db(veff); Qext=clip(qset−Kdroop·qcorr)`.
- **Node:** RMS reads `gen.bus1.cterm.m:u` (DER's own terminal); static reads `res_bus[sgen.bus]`;
  parity 1.5e-5 ⇒ SAME node. Not remote/pilot. (plant.py:346-347, 409-414; der_qv_local_loop.py:239,282)
- **Positive-sequence:** both. **Filter:** Tf=0.02 s, 3 orders below the 20 s window ⇒ per-interval
  x1=u exactly (no steady-state lag). **Gain:** Kdroop=1/slope = static R=Sn/slope in pu.
  **Deadband:** same param. **Compensation:** none (veff has no current term). **Washout/tracker/
  V-Q mode switch:** none (REEC_D PfFlag=0/QFlag=0, Qext is the Q reference). **Bias:** none.
- ⇒ the effective steady-state droop laws MATCH. The user's suspects are ruled out at the law level.

## The one unverified residual → clean u→y test (RUNNING)
The earlier `u→y` open-loop test (`experiments/run_rms_openloop_uy.py`) found a ~2.94 Mvar
"plant floor" (identical u, different y) — but measured UNDER the event-starvation bug (not clean).
NOT the droop law (verified). Candidate sources: REEC_D Qext→actual-Q tracking, or the RMS not
settling within 20 s (electromechanical ring 13-22 s > 20 s ⇒ RMS reads mid-transient u while the
static reads a fully-settled algebraic V). **Clean u→y re-run launched (db=0.01, 300 s, profiles)
to isolate the seed: δ≈0 ⇒ laws+plant match, divergence is pure closed-loop amplification; δ≠0 ⇒
effective mismatch localised.**

### RESULT (run `results/rms_openloop_uy/0004_2026-07-25_092021`) — δ ≈ 0
Plant-only residual under IDENTICAL u: **interface_q RMSE 0.710 / MAE 0.411 / max 4.42 Mvar;
zone_voltage RMSE 0.00203 / MAE 0.00142 / max 0.00513 pu** (n=180/45).
**The old 2.94 Mvar "plant floor" was an artifact of the event-starvation bug — the true floor
is 0.71 Mvar RMSE (4× better).**
⇒ No hidden law difference (remote node / compensated / filtered voltage / bias / mode switch)
and no material Qext→Q tracking or 20 s-settling mismatch: all would show here as a large δ.
The user's entire suspect list is closed out at BOTH the law level (code) and the response level
(experiment).

## CONCLUSION — the divergence is CLOSED-LOOP, not plant/law
- plant-only (same u): **0.71 Mvar** RMSE
- closed-loop db=0.01: DSO_4 gap **9.9** (300 s) → **18.3** (600 s) → **48** (3 h)
Two independent OFO loops amplify a sub-Mvar plant residual by 1–2 ORDERS OF MAGNITUDE, with
the measured amplification law: ∝ loop gain (4.1/9.9/17.2 for slope .12/.06/.03) and
super-linear in deadband width (0.6/2.6/9.9 for db 0/.005/.01) — the dead-zone kink is what
lets two loops at slightly different points be amplified differently.
**Thesis framing: P2 (QSS substitution) HOLDS at the plant level to 0.71 Mvar / 0.002 pu. What
fails is CLOSED-LOOP COMPARABILITY (P6): two independent controllers, each seeing only its own
plant through a non-smooth actuator law, need not converge to the same operating point even
when the plants are equivalent.** This supersedes the earlier "solver-vs-solver fixed-point
difference" framing — that described the static/RMS Q(V) settling, but the DOMINANT effect is
controller-loop amplification.
Caveat: the u→y run's own tracking errors (DSO |err| 2.5–5.3 Mvar) exceed the closed-loop runs'
— expected (fixed u ⇒ no feedback correction); does not affect the δ conclusion.

Probes: `scratchpad/ab_compare.py`; runs 0058-0065.

## P2 VALIDITY MAP (built 2026-07-25) — and a CORRECTION to the conclusion above

Figure: `results/p2_validity_map/p2_validity_map.{png,pdf}` (script `scratchpad/p2_validity_map.py`).
10 PF runs: 5 closed-loop co-sims + 4 open-loop u→y plant floors + slope sweep. All metrics are
**interface-Q RMSE (`dso_trafo_q_actual_mvar`, 12 flows) over a COMMON 15-interval / 300 s
horizon (n=180)** so every point is comparable. New CLI: `--der-deadband`/`--der-slope` added to
`run_rms_openloop_uy.py` so the PLANT floor can be measured per dead-zone width.

| db [pu] | plant floor (identical u) | closed loop | amplification |
|---|---|---|---|
| 0     | 0.106 | 0.12 | 1.1× |
| 0.005 | 0.246 | 0.65 | 2.6× |
| 0.01  | 0.710 | 2.11 | 3.0× |
| 0.02  | **8.906** | 4.33 | **0.5×** |

Loop gain at db=0.01 (closed loop): slope 0.12 → 1.26, 0.06 → 2.11, 0.03 → 9.04 Mvar.

**CORRECTION — the earlier claim "P2 holds at plant level to 0.71 Mvar; what fails is only
closed-loop comparability (P6)" was WRONG because it generalised a floor measured at ONE
dead-zone width.** Measuring the floor per db shows:
1. **The dead zone is the DOMINANT driver of plant-level (P2) divergence: the floor scales
   ~84× (0.106 → 8.906 Mvar) from db=0 to db=0.02**, far more than the closed-loop term.
2. **The closed-loop (P6) term is NON-MONOTONIC and flips sign**: the outer OFO feedback
   AMPLIFIES the plant residual at narrow dead zones (1.1× → 2.6× → 3.0×) but SUPPRESSES it
   at db=0.02 (0.5×) — the curves cross at db≈0.0125.
3. **Mechanism:** a wide dead zone switches OFF the local Q(V) regulation (most parks sit
   inside it, Q≈qset), so under IDENTICAL u the two plants have no local voltage feedback to
   hold their trajectories together and drift apart (8.91 Mvar). In closed loop the OFO
   controllers observe that drift and correct it (4.33 Mvar). At narrow dead zones the plants
   already agree (0.1-0.7 Mvar) and the dominant effect is the two independent loops diverging.
4. **db=0 is the clean control: floor 0.106, closed loop 0.12, amplification 1.1×** — the
   machinery (plants, controllers, co-simulation, event delivery) is sound; the dead zone is
   what breaks agreement.

Integrity check on the surprising db=0.02 floor: all four floor runs are structurally identical
(15 profile / 66 apply_u / 30 advance calls), deadband override confirmed applied, no
non-convergence, no skipped writes, MAE 8.28 ≈ RMSE 8.91 (a systematic offset, not noise).

## RE-ANCHOR + OFFSET DELIVERY — VERIFIED LIVE (probe `scratchpad/reanchor_probe.py`)
User question: does the RMS re-anchoring actually work — we must re-anchor V *and* offset Q; is
the offset in? Risk being probed: each dispatch writes TWO EvtParams per park (`qset`, `Vanchor`);
if only one is admitted/fires, the law is silently corrupted (new offset vs stale anchor ⇒ a
spurious `Kdroop·(V−V_anchor_old)` term; at Kdroop=16.7 a 0.01 pu stale anchor ⇒ ~17 %-of-rating
Q error). `Vanchor` is NOT in the monitored set (only `s:qset`), so no existing run could rule it out.
Probe issues known dispatches and reads `c:qset`/`c:Vanchor`/`c:Kdroop`/`c:db` back **from the DSL**:
| dispatch | max\|Δqset\| | max\|ΔVanchor\| | max\|ΔQ vs law\| |
|---|---|---|---|
| 1 (+0.10 pu) | 0.00e+00 | 0.00e+00 | 0.007 Mvar |
| 2 (−0.15 pu) | 0.00e+00 | 0.00e+00 | 0.129 Mvar |
| 3 (+0.05 pu) | 0.00e+00 | 0.00e+00 | 0.212 Mvar |
**VERDICT: both events land on all 44 parks, every dispatch; qset and Vanchor bit-exact (Δ=0);
delivered Q matches the law with PF's own held params to ≤0.21 Mvar (~0.04 % of a 500 MVA park).**
The small residual is expected: the check uses instantaneous `u` while the DSL integrates the
filtered `x1` (Tf=0.02 s) and the plant is still settling inside the 20 s advance (rings 13-22 s);
it grows with dispatch index for that reason. Semantics confirmed correct on both sides: at a
dispatch `Vanchor:=V_now` zeroes the droop term so `Q:=qset` (the absolute OFO command), and the
droop then acts on deviations from that point; static does the same (`core/plant.py:122-128`).
⇒ **The last mechanism-level suspect is closed: the divergence is NOT a broken re-anchor.**

## QSS → RMS NO-DISTURBANCE HOLD — stationarity CONFIRMED (probe `scratchpad/qss_to_rms_hold.py`)
Closes the last gap (raised via a ChatGPT review): we had verified settling only on the
CONTROLLED OUTPUTS (2 % band) from the BASE point over 60 s, while electromechanical modes ring
13-22 s. New test: (1) take the static plant to a PROFILED operating point (t=+300 s) and converge
the CLOSED-LOOP QSS problem there (`run_control=True`, so the Q(V) droop is at its fixed point);
(2) sync PF to that exact state + ComInc; (3) advance 300 s with **NO events at all**; (4) measure
drift vs t=0 in bus V, park Q, machine Q and machine SPEED (governor/frequency proxy).
Run at the NOMINAL deadband 0.01 — i.e. the configuration where the closed loop diverges.
Result: **worst bus-V drift 6.17e-11 pu over 300 s, FLAT** (6.11e-11 already at t=20 s, 6.15e-11
at t=300 s — noise, not creep); park Q, machine Q, machine-speed drift all ≈0 (≤4e-12 speed).
⇒ **The QSS point IS an equilibrium of the RMS DAE.** Pre-committed threshold was <1e-4 pu.

### THE DECISIVE SYNTHESIS
The QSS equilibrium holds exactly in RMS, yet the RMS reaches a DIFFERENT point when it evolves
dynamically along the co-simulation path. **Both are valid solutions of the same RMS model ⇒
MULTIPLE EQUILIBRIA WITH PATH DEPENDENCE, demonstrated positively (not by elimination).**
Full decision tree closed with measurements:
| candidate cause | evidence | verdict |
|---|---|---|
| models differ subtly | identical-u residual 0.106 Mvar @db=0; law verified (same node/gain/deadband, no PI, no compensation, no washout); qset+Vanchor bit-exact | EXCLUDED |
| RMS not actually stationary | 300 s hold, drift 6.17e-11 pu, flat | EXCLUDED |
| more than one equilibrium | QSS point holds in RMS **and** dynamic path lands elsewhere; residual scales 0.106→8.906 Mvar over db 0→0.02 | **CONFIRMED** |
Also: the reviewer's own key diagnostic `r_Q = Q_RMS − clip(Q*(V_RMS))` measured at ≤0.043 Mvar
⇒ the RMS DOES implement the assumed law, which by that decision tree forces this branch.
**Amendment to the reviewer's framing:** it lists the deadband as a *hidden difference between*
the models. Ours is IDENTICAL on both sides — and that is precisely the point: a correctly
implemented, SHARED deadband is by itself sufficient to create multiple equilibria, because
inside the dead zone the DER is a constant-Q source and the control equation stops constraining
V; at the edge both branches are simultaneously self-consistent, across 44 parks.

**Thesis framing (revised):** P2 (`y_rms(∞;u)=y_qss(u)`) holds tightly ONLY when the DER dead
zone is narrow or absent; its validity degrades ~84× across a realistic dead-zone range, because
the dead zone removes the local voltage regulation that keeps the QSS and RMS trajectories
together. Closed-loop comparability (P6) adds a secondary, non-monotonic term (≤3× amplification
at narrow dead zones, suppression at wide ones). Caveat: single operating point, 300 s horizon,
±1.0 pu capability override still active.
