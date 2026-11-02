# Weekend plan 30 Oct – 2 Nov 2026: harmonised device limits, Ch 9 re-validation, REF / STEP / MC, Case 1

Date: 2026-10-30, 14:00–15:30. Author of the plan: Claude Code (side session), for M. Schwenke.
Status: **approved and implemented 30 Oct afternoon; see §8**, which supersedes §§3–6 where they differ.
Companion note: `2026-10-30_sts_access_gain.md` (analysis, proposals, decisions D1–D5).

Naming: thesis variant names throughout (LOC, SVR, SVR+, OFO, OFO+, CEN; code L2, S1, S2, O2, O3, M).
G3 is area 2's only synchronous machine (pandapower `gen` 1, 800 MVA; called "gen 1" in the
companion note).

## 1. Author decisions of 30 Oct (afternoon) and what follows from them

| # | decision | consequence for the weekend |
|---|---|---|
| D2 | AVR set-point range harmonised to [0.95, 1.05] pu for every variant | code change (§4, item 1); Ch 9 re-validation (§5) before any campaign; REF, STEP, MC rerun |
| D2 scope | Case 1 included | K1-REF and K1-PAR rerun |
| D3 | declared re-validation now; the full Ch 9 procedure later, when capacity is free | §5; the thesis wording must match what was run (§5.4) |
| new | minimum holding time harmonised for every OLTC, local-rule ones included | code change (§4, item 2); §3 gives the current state |
| D4 | open: the author did not know "scenario B" | §2.3 explains it and proposes B and D |
| D1, D5 | not answered | not needed for the weekend |

## 2. Can the TS be short of reactive power while P stays feasible?

### 2.1 Short answer

- **Yes, locally.**
  - P feasibility needs only two things: P balance, here held by the distributed slack over all
    machines weighted by S_n, and a solvable voltage problem.
  - Q scarcity shows as deep voltage deviations and machines at their Q limits (PV → PQ). Only at
    the extreme does the power flow stop converging (voltage collapse).
  - Between "scarce" and "infeasible" there is a wide working range.
- **Scarcity is a local property.**
  - Reactive power does not travel: series losses scale with I²X, and dV/dQ decays with
    electrical distance.
  - The system-wide reserve (MC median ≈ 2,800 Mvar) does not protect area 2. Only the resources
    with a high dV/dQ at the disturbed bus count.

### 2.2 Model facts (code of record 9c0864d)

- **The TS is rich in Q by construction.**
  - Machine nameplates are 2 × the case39 base-case P, so the machines run at about 50 % of
    rating (`network/ieee39/constants.py`, nameplate comment).
  - The plant's machine Q box is **static**: ±0.5 S_n (`network/ieee39/build.py:167–168`). It is
    enforced in every plant PF (`enforce_q_lims_plant = True`).
- **The plant box does not depend on P; the controller's capability model does.**
  - The controller uses the P-dependent capability law of `core/actuator_bounds.py`, with stator,
    rotor and UEL limits.
  - Raising a machine's P therefore shrinks only the *controller's* Q range, not the plant's. At
    760 MW, G3's plant box allows 400 Mvar; the stator limit is about 250 Mvar.
  - The runner carries this as an open ToDo (`experiments/runners/multi_tso_dso.py:1154`).
  - **Consequence:** "G3 near rating" is not a valid scarcity mechanism in the present plant. It
    would create a model–plant mismatch, not physical scarcity. Making it valid needs a plant
    correction (P-dependent Q box in the plant), which is a design note, not this weekend.
- **A machine outage is a valid mechanism without any model change.**
  - Taking G3 out removes area 2's only machine.
  - The distributed slack replaces its P, and the PF converges.

### 2.3 Measured, open loop (plant model only, every control frozen, t = 0 of the reference period, `scarcity_openloop.py`)

| event(s) | area-2 RMS deviation from 1.03 pu [mpu] | lowest area-2 bus | G3 Q [Mvar] |
|---|---|---|---|
| none | 5.7 | 1.025 | −3 |
| 250 Mvar at IEEE 12 (A's area-2 event) | 32.8 | 0.946 (IEEE 12) | 130 |
| 250 Mvar at IEEE 8 | 27.1 | 0.987 (IEEE 8) | 98 |
| 250 Mvar at IEEE 4 (rule bus, §3.10 of the note) | 23.5 | 0.991 (IEEE 4) | 91 |
| G3 out | 5.5 | 1.024 | – |
| **B**: G3 out, line 4–14 out, 250 Mvar at IEEE 8 | **48.7** | 0.965 (IEEE 8) | – |
| G3 out, 250 Mvar at IEEE 4 | 41.3 | 0.975 (IEEE 4) | – |
| **D** (proposed): G3 out, line 4–14 out, 250 Mvar at IEEE 4 | **43.0** | 0.965 (IEEE 4) | – |
| G3 at 0.95 S_n (controller's op-diagram box set in the plant), 250 Mvar at IEEE 4 | 32.7 | 0.980 (IEEE 4) | 187 |

Reading (fact):
- At the 08:00 operating point G3 carries Q ≈ 0. Its outage alone changes nothing; scarcity
  appears only when Q is demanded.
- With G3 in service it takes 90–130 Mvar of a 250-Mvar step at once. Without it, the same step
  at IEEE 4 deviates the area by 76 % more (23.5 → 41.3 mpu).

**What "scenario B" is.**
- In my sketch "B" meant "a second, declared stress scenario".
- The code already has a scenario **B**, defined and screened on 27 Oct before this analysis
  (`experiments/ch_11_case2_multisystem/ref/scenarios.py`):
  - "area-2 stress": G3 out at 30:40, line 4–14 out at 121:40, 250 Mvar at IEEE 8 at 212:20;
  - the same structure and offsets as A.
- The 27-Oct screening called B "mild" (LOC outside [0.95, 1.05] pu 4.5 % of the time). That
  metric counts band violations.
- B's sag is deep but stays inside the band (0.965 pu open loop). On the thesis metric (deviation
  from 1.03 pu) it is the **hardest** of the candidates: 48.7 against 32.8 mpu for A's area-2
  event.

### 2.4 Proposal for D4 (declare before running; plant-only criterion, no controller result used)

- Run **both**, and report both whatever the outcome:
  - **B** as defined on 27 Oct: pre-registered before any OFO+ − OFO result existed;
  - **D**, a new scenario: B with the load moved from IEEE 8 to IEEE 4 by the location rule
    (§3.10 of the note).
- What D removes:
  - no MSC at the load bus (IEEE 8 has one);
  - no TS tap at the load bus (not the 12–11 / 12–13 transformers);
  - area 2's machine, by the outage.
  - These are the three substitutes the author named.
- **Scarcity criterion.**
  - The open-loop area-2 deviation after the load event must be at least that of A's area-2
    event (32.8 mpu). Fulfilled: B 48.7, D 43.0.
  - It is computed on the plant model only, so the choice does not depend on any compared
    variant.
- **Expected sign (hypothesis, written before the run).**
  - OFO+ − OFO < 0 in area 2 after the load event.
  - Its magnitude larger than in A's corresponding window (A, c08b: −0.34 mpu after 212 min).
  - SVR+ − SVR < 0 likewise.
  - **Why:** without G3 neither OFO nor SVR has an AVR lever in area 2. OFO is left with the
    TS-DER at IEEE 6, three MSCs and the 12–11 / 12–13 taps. OFO+ adds STS 1–3.
- **Cost:** 12 REF runs, about 4.5 h on 12 workers.
- **Not this weekend:** the "G3 near rating" variant (needs the plant correction of §2.2).

## 3. Current device rate limits (code of record), and the harmonisation

### 3.1 OLTC minimum time between two taps of the same transformer

| OLTC class | LOC | SVR | SVR+ | OFO | OFO+ | CEN |
|---|---|---|---|---|---|---|
| TS 2-winding (machine transformers, 12–11, 12–13) | held at commissioned tap | held | held | MIQP lock (1+5)·180 s = **18 min** | 18 min | 54 central steps = 18 min |
| STS coupler (3-winding) | local relay, **60 s** | local relay, 60 s | STS-OFO lock (1+5)·20 s = **120 s** | local relay, 60 s | 120 s | central lock (1+5)·20 s = 120 s |

Sources:
- `configs/config.py:495` (`int_cooldown` = 5) and `:544` (`oltc_cooldown_s_nc` = 60 s);
- `experiments/ch_11_case2_multisystem/variants.py:161–176`;
- `experiments/runners/multi_tso_dso.py:3082–3122`: no TS relay under SVR or `tso_oltc_mode="local"`;
  the machine taps are held.

- The TS taps have no local rule in any thesis variant, so nothing needs harmonising there. The
  18-min lock is already common to OFO, OFO+ and CEN.
- The couplers differ: 60 s under the local relay against 120 s under every MIQP.

**Proposal:** `oltc_cooldown_s_nc` 60 → **120 s**, i.e. (1 + `int_cooldown`)·T_STS.
- **A priori:** it is the lock of the selected design (author decision 2026-10-14, thesis Ch 3),
  not a tuned value.
- **Scope:**
  - it changes only the relay variants: LOC, SVR and OFO in Case 2, and every Case 1 variant;
  - OFO+, SVR+ and CEN are untouched;
  - the Ch 9 procedure is untouched (no relays in its cascade).
- **Side benefit:** the access steps SVR → SVR+ and OFO → OFO+ no longer also change the coupler
  rate limit.
- **Alternative, not recommended:** make the MIQP lock 60 s (`int_cooldown` = 2). That changes
  the controller of record and the Ch 9 basis.

### 3.2 Shunt banks under LOC and SVR

- There is **no local rule**. The banks are not even installed in LOC, SVR or SVR+
  (`_NO_SHUNTS`, `experiments/ch_10_case1_svr/run_case1.py:100–103`). Recorded reason: "a bank
  is only dispatched where the TSO MIQP runs".
- **Consequence (fact):**
  - The law step SVR → OFO bundles three changes: the law, the in-loop TS taps and the banks
    (`STEP_ALLOWED[("S1","O2")]`, declared).
  - The access steps SVR → SVR+ and OFO → OFO+ are not affected.
  - Case 1 isolates the device effect under OFO (OFO without and with taps and banks).
- **Option:** a classical local bank rule for LOC, SVR and SVR+:
  - a voltage relay with dead band;
  - the same 30-min holding time and MSC/MSR exclusion as the MIQP.
  - It is a new mechanism with its own thresholds, so it needs a design note. **Not this
    weekend**; report the asymmetry as declared.

### 3.3 AVR set-point range

| | LOC | SVR / SVR+ | OFO / OFO+ | CEN |
|---|---|---|---|---|
| now | pinned at v_set | [0.95, 1.05] (`svr_controller.py:230`) | [0.95, 1.07] (`tso_controller.py:143–144`, class default; the runner does not set it) | inherits 1.07 (`CentralControllerConfig(TSOControllerConfig)`) |
| after D2 | unchanged | unchanged | [0.95, 1.05] | [0.95, 1.05] |

## 4. Implementation (before anything runs; ≈3–4 h including smoke tests)

On a new branch from `minimal/thesis-core`. Both hosts must run the same commit (e5power pulls
it).

1. **AVR range, one source of truth.**
   - `MultiTSOConfig.gen_vm_min_pu = 0.95`, `gen_vm_max_pu = 1.05`.
   - Pass them into `TSOControllerConfig` (runner, at `multi_tso_dso.py:1260`) and into the central
     controller's config. Build `SVRConfig.v_set_min/max_pu` from the same fields.
   - Set the class default 1.07 → 1.05 as well, so a direct use cannot fall back to 1.07.
   - The `STEP_ALLOWED` guard then keeps the range equal across variants automatically, because
     the field is in no step's allowed set.
2. **Coupler relay holding time:** `oltc_cooldown_s_nc` 60 → 120 s, with a docstring citing this
   decision.
3. **Scenario D:** a new entry in `ref/scenarios.py`:
   - `_gen(1, _at(30, 40))`, `_line(7, _at(120, 100))`, `_load(3, 250.0, _at(210, 140))`;
   - the same offsets as B; `--scenario D` works with the existing `run_ref`.
4. **Declaration** `docs/analysis/2026-10-30_scenarios_B_D_declaration.md`, committed before the
   first B/D run. Content: §2.4 (rule, criterion with the measured open-loop values, expected sign,
   "both reported").
5. **Re-validation producer** `experiments/ch_09_parameter_selection/ch_9_5_avr_revalidation.py`:
   - reuses the unit re-simulation of `ch_9_3_amendment1.py`;
   - `--keys` so that the set can be split over two hosts;
   - writes `DECISION.json` (§5).
6. **Chain scripts per host** (detached via WMI `Win32_Process.Create`, below-normal priority, at
   most 15 workers each):
   - e5wimi: re-validation → decision → STEP → Case 1;
   - e5power: REF A/B/D → wait for `DECISION.json` → MC.
7. **Smoke tests:**
   - every Case 2 variant 10 min on A, and every Case 1 variant 10 min;
   - one Ch 9 unit for 30 min;
   - assert in the OFO smoke log that no AVR set-point exceeds 1.05;
   - assert in the LOC smoke log that no coupler taps twice within 120 s.
8. Every run of record goes through `thesis_runs.start_thesis_run`.

## 5. Ch 9 re-validation under the corrected AVR range (declared now, before it runs)

### 5.1 Rule

- **Set.** Every candidate admitted by the run-0008 filter whose e_v(TS) is within **10 %** of
  the best:
  - 13 candidates: 301c285ee52b, c08b1d3d33ab, 67ad1cf843ac, 9e48320e8a01, ff3f747d3f96,
    c9a5a0bb139e, 0efa83f8046a, 57f0162a5166, 4ccfa217c2b5, e95cb58c9043, 34a9f0cf1444,
    7935cc303372, 8323fe3f0c0b;
  - each on W1–W4, W6, i.e. 65 units of 12 h each.
- **Re-simulate** these 65 units with the new code, otherwise exactly as in run 0008: same network,
  windows, noise seed and bank settings.
- **Re-apply the frozen selection** of `docs/tuning/2026-10-28_weight_selection_procedure.md`
  (filter, ranking with the 2 % indifference band, settling requirement in rank order) to the
  re-simulated set.
  - **Settling results are reused** from runs 0001/0008: the isolated-STS step bank contains no
    TS-OFO, so the AVR range does not enter it. To be confirmed in item 5 of §4 before launch.
- **Outcome.**
  - **c08b remains the result** if it passes the filter and is the first candidate in rank order
    that passes the settling requirement.
  - Otherwise the new first candidate is the result. It is written as a new paramset; the
    speculative REF runs on e5power are stopped, and every campaign starts on the new paramset.
- **Reported in addition:**
  - how many of the 65 units reproduce the stored result bit-exactly, i.e. where the 1.05 bound
    was inactive throughout;
  - the old and new rank order of the 13.

### 5.2 Why 10 %, and what it does not cover (assumption, stated in the thesis)

- In the old ranking:
  - the margin from c08b to rank 3 is 2.1 %, and to rank 14 it is 10 %;
  - 5 candidates lie within 5 %, 13 within 10 % and 28 within 20 %.
- A candidate outside the set could overtake c08b only if a *tightened* bound improved its e_v(TS)
  by more than ~10 % relative to c08b. That is not impossible in a nonlinear closed loop, but it
  is implausible for a constraint that removes voltage-raising authority. It is the stated
  assumption of the re-validation.
- Candidates rejected by the filter (mostly tap or bank hunting) are not re-simulated. The AVR
  range could in principle change hunting; same assumption.

### 5.3 Cost

- 65 units at ≈43 min each (stored wall time 2,586 s for one W4 unit) ≈ 47 worker-h.
- That is ≈3.5–4 h on the 15 workers of one host, or ≈2.5 h if split over both.

### 5.4 Thesis wording (author decision, 30 Oct afternoon)

- The thesis describes the **complete selection procedure** (Stages I, II-a, II-b, selection) as
  the method.
- The complete procedure will be run under the harmonised limits as soon as computing time is
  free, and before the text is final.
- The weekend re-validation is the interim check that the campaigns of this weekend rest on the
  weights that procedure is expected to select.
- **Consequence to keep in mind:** if the complete run later selects different weights, the
  Case 1 and Case 2 runs of this weekend have to be repeated with them.

## 6. Schedule (Fri 30 Oct, 14:00, to Mon 2 Nov)

Measured costs:
- STEP ≈280 worker-h (≈19 h on 15);
- MC ≈280 worker-h (≈19 h on 15);
- REF ≈4.5 h per batch of six (SVR is the long pole);
- Case 1 K1-REF + K1-PAR: 13 runs, ≈4.5 h.

| when | e5wimi (≤15 workers) | e5power (≤15 workers) |
|---|---|---|
| Fri 14:30–18:30 | implementation and smoke tests (§4); STEP at 1.07 finishes (60 of 1,350 runs left at 14:09, all in part E; 20 workers, already running) | MC rerun at 1.07 finishes |
| Fri ≈19:00–23:00 | **Ch 9 re-validation** (65 units) → `DECISION.json` | **REF A, B, D**, 6 variants each, 18 runs (≈5–6 h), on c08b *speculatively* |
| Fri 23:00 – Sat ≈18:00 | if c08b stays: **STEP**, all parts (≈19 h) | REF ends ≈01:00; then **MC** (≈19 h, ends Sat ≈20:00) |
| Sat ≈18:00–23:00 | **Case 1** K1-REF + K1-PAR (≈4.5 h) | MC (continued) |
| Sun | buffer for resumes; evaluation (REF report, STEP metrics, MC evaluation) | buffer; evaluation |
| Sun night (optional) | the full Ch 9 procedure, if its driver can split across the two hosts (≈1,100 worker-h ≈ 37–40 h) | likewise |
| Mon | results into ch:case_svr and ch:case3; the Ch 9 sentence per §5.4 | |

- **Fallback** (re-validation selects another candidate): stop the speculative REF, switch the
  paramset, restart REF. STEP, MC and Case 1 start on the new paramset. Loss ≈5 h; it still ends
  by Sunday morning.
- **Cut-off for the 1.07 runs:** if STEP or the MC rerun at 1.07 is not finished by 19:00, stop
  it. The completed parts remain usable as the 1.07 record.
- **Kept, not deleted:** REF 0010, STEP 0001 (e5wimi), MC 0003 and its rerun. With the new runs
  they give a measured AVR-range sensitivity (1.07 against 1.05) for every campaign. That also
  tests the prediction of the companion note (§4, proposal 3): the access gain should grow and the
  law gain SVR → OFO shrink.

## 7. Risks and open points

- **Time.** Implementation must be done by ≈18:30 for the schedule to hold. Every hour later
  shifts everything; the buffer is Sunday.
- **Unattended chain.** A crash on Saturday stays unnoticed until someone looks. A check on
  Saturday morning and evening is advisable (author or a session).
- **The re-validation assumption** (§5.2) must be stated in the thesis.
- **Both 1.07 records show the access gain** (see the companion note). The weekend will show
  whether it grows under 1.05, as predicted. If it does not, the prediction was wrong; it is
  reported as such.
- **B and D are new evidence**, not a replacement for A. A stays the reference scenario (author
  decision 2026-10-27).
- **Plant correction (P-dependent machine Q box)** and the **local bank rule** for the classical
  variants are design-note topics for next week. Both would change results in every chapter that
  uses the plant.
- **e5power** must run the same commit; its clone path and environment follow
  `experiments/ch_11_case2_multisystem/mc/README_server.md`.

## 8. Implementation and launch (30 Oct afternoon; supersedes §§3–6 where different)

**Author decisions on top of §1:**
- **TS taps in the classical variants** (LOC, SVR, SVR+; Case 1 wherever the TS taps are outside
  the MIQP): a local relay at **1.03 ± 0.01 pu on the HV side**, like the coupler relays, instead
  of being held.
- **Minimum time between two taps:** **120 s for STS couplers, 15 min for TS taps**, in every
  variant.
  - The TS-OFO lock therefore changes 18 → 15 min (`tso_int_cooldown` = 4), and CEN changes with it
    (44 central steps).
  - This is the minimum-time-between-taps reading of "15 min", consistent with the 120 s for
    couplers.
- **Banks in the classical variants:** a local rule.
  - The MSC engages below 1.01 pu, the MSR above 1.05 pu, on the primary side of the hosting
    coupler.
  - Release at the opposite threshold (Claude's completion of the rule, so that a bank cannot
    stand engaged against the voltage).
  - 30-min holding time and MSC/MSR exclusion as in the MIQP.
- **Workers:** 20 per host.

**Code:**
- Branch `weekend/harmonised-limits-2026-10-30` (worktree `F:\qOFO_weekend`).
- Daily log `docs/daily_log/10_2026/2026-10-30_weekend_harmonised_limits.md` on that branch.
- Declaration of B and D: `docs/analysis/2026-10-30_scenarios_B_D_declaration.md` on that branch.

**Chains** (`_weekend/chain.py`):
- **e5wimi:** Ch 9 re-validation (14 candidates × 5 windows, 20 workers) → `DECISION.json`.
  - If unchanged: `DECISION_OK` is pushed to e5power, then STEP (14 workers) and Case 1 K1-ALL
    (13 variants, 6 workers) start.
  - If changed: `DECISION_STOP`.
- **e5power** (`F:\qOFO_runs\qOFO_weekend`): REF A, B, D (3 × 6 workers, speculatively on c08b)
  → MC (20 workers) after `DECISION_OK`.
- Results stay on each host's F: until copied back.
