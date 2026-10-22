# 2026-08-05 — The second N-1 excursion is the frequency nadir of an island with no primary control

**Timestamp:** 2026-08-05
**Scope:** diagnosis only. No production code and no PowerFactory object was
changed; every PF call in this investigation was a read of attributes.
**Answers:** `docs/PROMPT_pf_second_excursion.md`.

---

## 1. Question

Every N-1 run in the dead-band × droop study shows two voltage excursions. The
first is the outage transient at t ≈ 180.6 s and it decays. A second, five to
ten times larger, arrives 6–11 s later and is the peak from which the study's
**upper** bound on the dead-band half-width is read. Its delay orders by the
size of the tripped machine, not by the size of the initial transient.

## 2. Answer

There is **no discrete trigger**. No relay, no limiter, no protection element,
no scheduled event and no adapter write occurs at the second excursion. The
excursion is the reactive-power consequence of a **frequency ramp that the
model has almost nothing to arrest**, and it happens at the frequency nadir.

Classification asked for in the prompt: **(a) plant physics** — but the physics
of a mis-parameterised primary-control layer, so with respect to the *study* it
is an artefact. It is not (b) protection or a limiter, and not (c) the adapter
or the solver.

### 2.1 The frequency ramp

Machine speeds are in the full ComRes export (`s:xspeed`), so this needed no
re-simulation. Excluding the tripped machine, run 0453 (650 MW @ −117 MW):

| t [s] | 180 | 182 | 184 | 186 | 188 | 189 | 190 | 195 | 300 | 599 |
|---|---|---|---|---|---|---|---|---|---|---|
| f [Hz] | 49.93 | 49.59 | 49.21 | 48.91 | 48.67 | 48.64 | 48.67 | 48.68 | 49.05 | 49.10 |

**The island falls 1.3 Hz** and is still 0.9 Hz low after 600 s.

### 2.2 Why: 70 % of the capacity has no governor

Read out of the model (`pf/add_g01_gov.py --report`, and the composite `pblk`/
`pelm` slots):

| machine | S_n [MVA] | Gov slot | governor | Oel slot | Uel slot |
|---|---|---|---|---|---|
| **G 01** | **10 000** | **EMPTY** | **none** | EMPTY | EMPTY |
| G 03 | 800 | GOV 03 | gov_IEEEG1, K = 5 | EMPTY | EMPTY |
| G 04 | 800 | GOV 04 | gov_IEEEG1, K = 5 | EMPTY | EMPTY |
| G 07 | 700 | GOV 07 | gov_IEEEG1, K = 5 | EMPTY | EMPTY |
| G 09 | 1000 | GOV 09 | gov_IEEEG1, K = 5 | EMPTY | EMPTY |
| G 10 | 1000 | GOV 10 | gov_IEEEG3, σ = 0.04, T_r = 10 s | EMPTY | EMPTY |

* **G 01, the 10 GVA "Rest of U.S.A./Canada" equivalent — 69.9 % of installed
  capacity — has an empty Gov slot.** Its mechanical power is fixed. Its
  *electrical* output still rises (217 → 598 MW in run 0453, from
  `zone_p_gen`), but only by decelerating, which is exactly why the frequency
  keeps falling.
* The four governed machines carry **K = 5, i.e. a 20 % droop** — four to five
  times slacker than the 4–5 % that is normal — and G 10 is a hydro governor
  with T_r = 10 s, i.e. deliberately slow. Their valve limits are ±0.3 pu
  (±0.1 for G 10).

The responding capacity is therefore ≈ 3500 MVA at a 20 % droop against a
650 MW loss. A 1.3 Hz excursion follows directly; with a correctly
parameterised layer the same loss would give some 0.2–0.3 Hz.

### 2.3 Why the delay orders by machine size, and why it looks like a threshold

The ramp rate is proportional to the deficit, so a given frequency is reached
at a time inversely proportional to it. Onset determined as the departure of
total machine Q from its 183–187 s trend:

| cell | tripped | df/dt [Hz/s] | t_onset [s] | **f_onset [Hz]** | f_nadir [Hz] | ΣQ_gen at onset → max [Mvar] |
|---|---|---|---|---|---|---|
| 650 MW @ −117 MW | G 03 | −0.1553 | 188.25 | **48.645** | 48.630 | 390 → 1026 |
| 650 MW @ +409 MW | G 03 | −0.1545 | 188.99 | **48.650** | 48.633 | 387 → 1062 |
| 560 MW @ −117 MW | G 07 | −0.1299 | 191.08 | **48.653** | 48.652 | 313 → 783 |
| 560 MW @ +409 MW | G 07 | −0.1278 | 193.35 | **48.664** | 48.664 | 304 → 535 |

The onset times span 5.1 s; the onset **frequencies span 0.019 Hz**. The
excursion is locked to a frequency, not to a time — and that frequency is the
**nadir**, where the governed machines have run into their valve limits. The
same limits in every cell are why the nadir is the same in every cell despite
different deficits, which is what made this look like a fixed relay setting.

Note the residual ordering: the faster the ramp, the lower the onset frequency
(48.645 for −0.155 Hz/s, 48.664 for −0.128 Hz/s). That is a measurement/filter
lag on a common limit, not four independent thresholds.

### 2.4 The excursion itself is continuous

At 10 ms resolution through 186–191 s there is **no discontinuity anywhere**.
Total machine Q runs 310 → 315 → 323 → 335 → 350 → 368 → 390 → 414 → 441 →
470 → 502 → 535 Mvar in successive 50 ms samples — smooth and accelerating.
The largest single-sample step in the whole 185–192 s window is 2.9 Mvar on
G 01's Q, which is a fast continuous change, not a step.

The peak magnitudes order with the ramp rate as well (0.0995 / 0.0825 /
0.0731 / 0.0412 pu against −0.1553 / −0.1545 / −0.1299 / −0.1278 Hz/s), i.e.
they measure *how hard* the network is driven past its reactive limit. A
discrete relay would give a similar-sized event irrespective of ramp rate.

## 3. What was excluded, and how

| hypothesis | verdict | evidence |
|---|---|---|
| Over-excitation limiter / field-current limit | **dead** | Every machine's `Oel` **and** `Uel` slot is empty — no limiter exists to act. And machine Q *triples* through the event (315 → 1026 Mvar) instead of being clamped, so no ceiling is reached. |
| Frequency relay / protection trip | **dead** | The WECC `Protection` block is present on all 44 parks (`Protection V_f_ROCOF`) but carries **60 Hz template thresholds**: `f_min = 57.5`, `f_max = 61.5`, with `Prot_f = 1`, `Prot_rocof = 0`. In a 50 Hz system these are unreachable, and no park disconnects in run 0453 (0 of 44). See §5 — the thresholds are nonsense as they stand and should be fixed regardless. |
| Adapter parameter events (`qset` / `Vanchor`) | **dead** | All 44 `s:qset` signals change **once**, at t = 180.50 s, the scheduled dispatch tick. Nothing moves at 186–192 s. |
| AVR setpoint change | **dead** | All six `s:usetp` constant over 170–200 s. |
| Discrete actuators | **dead** | `c:nntap`, `c:n3tap_h`, `c:ncapa` all constant over 170–200 s. Confirms the earlier `check_discrete_actuators` result. |
| Growing oscillation | **dead** (already) | Envelope decays right up to the jump. |

### 2.5 This was predicted, and filed as dormant

From the RMS co-sim notes, 2026-07-21:

> Primary control is architecturally different but currently DORMANT:
> pandapower uses `distributed_slack=True` with `slack_weight=sn_mva`
> (algebraic, no frequency state); RMS has governors (uniform K = 5) and G 01
> as reference machine with NO governor. Steady-state participation CAN match
> (uniform R_pu ⇒ ΔP ∝ S_n = the distributed-slack law) but the transient
> cannot. Irrelevant to current Gate E (fixed injections, no
> profiles/contingencies ⇒ no P imbalance); **becomes decisive for any
> generator-outage study.**

The dead-band × droop study *is* a generator-outage study. The asymmetry was
correctly identified and correctly scoped a fortnight before this campaign was
launched; nothing carried the warning forward into the N-1 design. Worth a
pre-flight check on the two-plant asymmetries whenever a study changes class
(fixed injections → contingencies).

Note also the second half of that entry: the **static** plant shares ΔP by
`slack_weight = sn_mva`, i.e. G 01 takes 69.9 % of any deficit *instantly and
without a frequency excursion*. So the static and RMS plants do not merely
differ in transient — under an outage they sit at different operating points
entirely, which is a separate reason Gate E cannot certify these runs.

## 4. Consequence for the study

* The **upper** bound on the dead-band half-width is read off a peak produced
  by a frequency excursion that the model cannot arrest because the dominant
  machine has no governor. It measures the missing primary-control layer, not
  the dead band. **It does not stand as written.**
* The **lower** bound is unaffected: it is measured on undisturbed twins, where
  frequency stays within 0.14 Hz of nominal and none of this happens.
* The **ordering** of half-widths is unaffected — it is monotone in the dead
  band at every series, and the mechanism above is common to all cells of a
  series.

The decisive confirmation is a re-run of one cell with a governor on G 01
(`pf/add_g01_gov.py --apply` exists for exactly this). Expected: the frequency
excursion drops to a few tenths of a Hz and the second excursion disappears.
**Not done — it modifies the PowerFactory model and needs the user's go-ahead.**

## 5. Two defects found in passing

1. **The WECC protection thresholds are 60 Hz values in a 50 Hz system**
   (`f_min = 57.5`, `f_max = 61.5`, `Prot_f = 1`), inherited from the template
   `WECC Large-scale PV Plant 110MVA 60Hz`. They did not fire here, but the
   parks are effectively running without frequency protection, and any future
   study that *needs* it would be silently wrong.
2. **The outage lands ~19.4 s early** — configured for t = 200 s, delivered at
   t ≈ 180.6 s, one whole dispatch interval ahead, coincident with the 180.50 s
   dispatch tick. Harmless for this diagnosis, but it means "trip at 200 s" is
   not what the runs did.

## 6. Open

* The chain "governor valve limits → transfer peaks → reactive demand exceeds
  what the AVRs can hold at the weak TS park" is inferred from `zone_p_gen`
  at 20 s dispatch resolution plus the Q and V traces. Machine P is not in the
  RMS export at full resolution, so the link is well-supported but not directly
  measured. Adding `m:P:bus1` and the governor output to the monitor set would
  close it.
* During the excursion the worst park's Q *falls* with V (+24.6 → −31 Mvar)
  although the Q(V) droop is disabled by the 0.5 pu dead band. Candidates:
  REEC_D current limiting (`Imax = 1.3`, `PqFlag = 0`, so Q headroom shrinks
  as V drops) and the REGC_C converter reactance (`xe = 0.1` pu) making
  terminal Q differ from injected Q. Either way the parks *withdraw* support as
  V falls, which is what makes the excursion abrupt rather than gradual. Not
  resolved.

## 7. Void cell 0480 — re-run, and it is NOT a transient failure

Re-ran the cell (m = 0.05, −117 MW, δ = 0.5, 560 MW / gen 5 trip) with the
study's own invocation. It completed cleanly — 600 s, 30/30 records, 25 MB
trace — and **reproduced the defect**:

| run | max \|dV\| vs twin, t < 180 s | first deviation > 1e-4 pu |
|---|---|---|
| twin 0454 | 0.000000 | — |
| 650 MW sibling 0474 | 0.000000 | 180.55 s (correct) |
| **560 MW original 0480** | **+0.009109** | **0.59 s** |
| **560 MW re-run (0413_2026-08-05_195630)** | **−0.011378** | **0.59 s** |

Same instant, **opposite sign, different magnitude** — reproducible in timing,
non-deterministic in value. Re-running does not recover this cell; it should
stay VOID, and the manifest note should say so rather than implying a one-off.

An audit of the whole study puts this in context: of **64** N-1 runs measured
against the twin of their own (droop, window, dead-band) group, **63 are
bit-identical to their twin until t = 180.55 s** and **0480 is the only
exception**. So the campaign's data integrity is otherwise exact, and whatever
0480 hits is specific to that one parameter combination — note its 650 MW
sibling 0474, identical in every setting except which machine is armed to trip
at t = 180.6 s, is clean. Not diagnosed; a candidate worth checking is the
Q(V) multi-equilibrium at the plant seam (`seed_qv_equilibrium` reports
`max |Q* − Q_c| = 108.9 Mvar` at t = 0).

Bookkeeping hazard: the re-run was allocated counter **0413** in
`results/rms_phase6_replay`, because moving the study's runs to
`results/deadband_droop/runs` left the allocator free to reuse numbers the
manifest already holds (`0413` there is 0.05 / +409 MW / δ = 0.1 / 650 MW
trip). Different directories, so nothing was overwritten, but re-running
`build_manifest.py` across both trees would collide.

## 8. G 01 governor applied (user-approved) — and a bug in the apply script

`pf/add_g01_gov.py --apply` bound a governor into G 01's empty Gov slot;
ComInc green. **But `_apply` takes the first machine it finds with a governor
as its template, and on this model that is `GOV 10` — a *hydro* governor**
(`gov_IEEEG3`: σ = 0.04, Δ = 0.2, T_r = 10 s, T_w = 0.75 s). Wrong twice over
for a 10 GVA thermal equivalent:

1. It defeats the script's own printed rationale ("droop inherited from the
   template ⇒ per-unit droop is uniform"). σ = 0.04 is a **4 % droop**; the
   steam sets G 03/04/07/09 run `gov_IEEEG1` with **K = 5, a 20 % droop**.
2. `T_w = 0.75 s` is water hammer — the initial power response of a hydro
   turbine goes the *wrong way*. On 70 % of the system that distorts exactly
   the first seconds after a trip.

Measured with `--test-sharing` (10 % step on the 428.7 MW load, ΔP = 37.48 MW):

| machine | ΔP [MW] | share % | target % (S_n) | err pp | droop |
|---|---|---|---|---|---|
| **G 01** | 31.13 | **83.06** | 69.93 | **+13.13** | 4 % |
| G 10 | 3.11 | 8.29 | 6.99 | +1.29 | 4 % |
| G 09 | 0.93 | 2.48 | 6.99 | −4.51 | 20 % |
| G 03 | 0.75 | 2.01 | 5.59 | −3.59 | 20 % |
| G 04 | 0.83 | 2.22 | 5.59 | −3.37 | 20 % |
| G 07 | 0.73 | 1.95 | 4.90 | −2.95 | 20 % |

The error is exactly droop-sorted: both 4 % machines over-contribute, all four
20 % machines under-contribute. GOV 01 was therefore rebuilt from **GOV 03**
(`gov_IEEEG1`, K = 5) instead. `--revert` still removes it cleanly.

**Note the pre-existing defect this exposes:** the model never had uniform
per-unit droop — `--report`'s own uniformity check prints "governors do not
share a parameter set". Even with GOV 01 correct, G 10 remains at 4 % against
everyone else's 20 %, so `ΔP ∝ S_n` cannot hold exactly.

**Decision taken (user, 2026-08-05): uniform 4 %.** Applied — `K = 5 → 25` on
all five `gov_IEEEG1` sets (GOV 01/03/04/07/09); G 10 keeps `σ = 0.04`, which
is the value the others were matched *to*. ComInc green, live `K = 25.0`
confirmed on all five.

| | 650 MW loss ⇒ Δf | change |
|---|---|---|
| before | ≈ 1.3 Hz (measured) | G 01 no governor, mixed 4 %/20 % |
| uniform 20 % | ≈ 0.48 Hz | rejected |
| **uniform 4 %** | **≈ 0.10 Hz** | **applied** |

(Δf ≈ R · ΔP / ΣS_n, ΣS_n = 13 500 MVA post-trip, before load relief.) K = 5
was unusually slack for a steam governor — typical IEEEG1 gain is 20–25 — so
those values were themselves suspect. Under uniform droop no governor limit
binds (G 01's required share is 454 MW = 0.045 pu against a ±0.1 pu gate
limit), which also removes the valve-limit-set nadir of §2.3.

### 8.1 Does the QSS side need the same droop? No — it has none

pandapower runs `distributed_slack=True` with `slack_weight = sn_mva`
(snapshot values exactly 10000/1000/1000/800/800/700), i.e. it closes the
power balance **algebraically** in proportion to S_n with **no frequency
state**. There is no droop parameter on that side to match.

With *uniform* R the RMS steady-state share is ΔP_i = (Δf/R)·S_n,i, so the
**shares are ∝ S_n whatever R is** — R cancels. Therefore:

* **Uniformity, not the droop value, is the parity condition.** The 20 % → 4 %
  change does *not* alter static/RMS sharing agreement; that was already
  fixed by making the droops uniform.
* R alone sets Δf = R·ΔP/ΣS_n, which the QSS plant does not represent. The QSS
  plant is effectively an R → 0 machine set: exact ∝S_n sharing, zero
  frequency deviation, instantaneous.

**Irreducible asymmetry.** Under a contingency the RMS will always show a
frequency dip and a transient the QSS structurally lacks, for *any* droop
choice. This is the mechanism behind the campaign's empirical rule that
"`exit=1` is normal for every N-1 run": Gate E validates QSS/RMS equivalence,
and a topology change breaks it by construction. Removing this would mean
giving the QSS side its own Δf state and distributing ΔP by S_n/R — a real
modelling change, judged not worth it; one sentence in the thesis instead,
because a reader will ask why the QSS N-1 shows no frequency effect.

## 9. Confirmed by re-running the Ch. 9 trajectory figure post-fix

The four cells behind `graphics/Ch9/deadband_trajectory.tex` were re-run on the
fixed model (2026-08-05/06). Same signal, same window, same everything except
the primary-control layer:

| | pre-fix (0474/0464/0454/0443) | post-fix (0416/0417/0414/0415) |
|---|---|---|
| outage onset | 180.55 s | 180.55 s |
| no droop, deepest | **0.9272 pu @ 189.65 s** | **0.9855 pu @ 180.65 s** |
| no droop, peak dev vs twin | **0.0995 pu** | **0.0346 pu** |
| δ = 0.01, deepest | 1.0037 pu @ 204.13 s | 1.0172 pu @ 181.85 s |
| δ = 0.01, peak dev vs twin | 0.0244 pu | 0.0113 pu |
| δ = 0.01 leaves dead zone | 189.35 s | **never** |
| Gate E verdict | FAIL (topology change) | **PASS / VALID, all four** |

**The delayed excursion is gone.** The deepest point moves from 9.1 s after the
outage to 0.1 s after it — it is now the immediate transient, nothing else.
Peak deviation on the no-droop leg fell 2.9×. This is independent confirmation
of §2: the mechanism was the frequency ramp, and removing the ramp removes the
excursion.

**Gate E now PASSES on the N-1 runs.** The campaign's rule that "`exit=1` is
normal for every N-1 run" no longer holds — with G 01 governed and the droop
uniform, the RMS post-contingency operating point agrees with the static
plant's distributed-slack point closely enough for the gate. That is a stronger
statement than §8.1 anticipated: the *residual* frequency transient remains
unrepresentable in QSS, but it is no longer large enough to break equivalence.

### 9.1 Two things the new figure needs before it goes in the thesis

1. **The δ = 0.01 leg never leaves its dead zone**, so the plotted park's droop
   never engages — yet the two legs still differ (0.0113 vs 0.0346 pu). The old
   caption's mechanism ("the local layer acts only from t = 189.4 s") is dead.
   δ *also* sets the TSO/DSO dispatch dead band (`--tso-deadband`/
   `--dso-deadband`) and every DS park's zone, which re-anchors every 20 s.
   **Which of those carries the difference is NOT established.** Do not assert
   a mechanism in the caption until it is.
2. **The undisturbed reference is no longer safe to draw as one line.** It is
   the mean of the two legs' twins. Pre-fix they differed by 1.55e-3 pu on an
   axis spanning 0.126 pu — under a point. Post-fix they differ by **5.26e-3 pu
   on an axis spanning 0.062 pu, ~8 % of the axis height**. Either draw both
   twins or state the spread in the caption.

Generators: `results/deadband_droop/make_trajectory_figure.py` (pre-fix,
unmodified) and `make_trajectory_figure_postfix.py` (new). Outputs
`deadband_trajectory_prefix.tex` and `deadband_trajectory_postfix.tex` in the
same folder; both need copying into `graphics/Ch9/` on the workstation, since
the author's LaTeX tree is not reachable from the server.

## 10. Method

Scripts are in the session scratchpad, not the repo (diagnosis only):
`extract_window.py`, `first_mover.py`, `fine_trace.py`, `cross_cell.py`,
`full_horizon.py`, `p_pickup.py`, `der_trip_check.py`, `dump_protection.py`,
`slots.py`, `reec.py`.

Data actually used: `csv/rms_comres_full.csv` (237 columns at 10 ms — it
already carries `s:xspeed`, `s:qset`, `s:usetp` and all tap/shunt states, which
is why no re-simulation was needed), `csv/rms_der_raw.csv`, `rms_records.pkl`.
The ComRes export is `;`-separated with a **decimal comma** and two header
rows; duplicate element/variable labels must be de-duplicated or a naive
`endswith` match silently sees 1 of 44 columns.
