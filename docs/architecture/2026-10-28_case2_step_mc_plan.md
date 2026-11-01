# 2026-10-28: Plan of the Case 2 step tests (STEP) and Monte-Carlo (MC)

Status: **proposal for the author**. Nothing is implemented yet.

Context:
- Reference run of record: `0009_2026-10-28_093739_run_ref` (scenario A, six variants).
- Author decisions applied here (2026-10-27/28):
  - events lie between two TS dispatches;
  - "minimum" metering for every controller and the droop;
  - STEP complements the MC and includes load ramps;
  - O3 is reported as is; its remedies are decided after STEP has measured the effects;
  - one STEP test is a load ramp, in which the voltage tracking during the ramp is assessed.

## 0. Common setup

- **Variants:** `experiments/ch_11_case2_multisystem/variants.py`: L2, S1, S2, O2, O3, M, with
  `assert_case2_controlled` before every campaign.
- **Paramset and timing:** `ch9_selected_band_2026_10`; T_TS = 180 s, T_STS = 20 s; v_ref = 1.03 pu.
- **Noise:** "minimum" class, one noise seed per episode or case, shared by all variants.
- **Provenance:** every campaign is one `thesis_runs` folder (README, manifest, LaTeX header),
  runs on e5power (detached WMI launch, process pool), keeps the full `log.pkl` per run, and is
  pulled back to the same relative path.
- **Cost basis:** measured in 0009 (6 parallel processes on e5power). Wall minutes per simulated
  hour:

| L2 | S1 | S2 | O2 | O3 | M | all six |
|---|---|---|---|---|---|---|
| 11.5 | 37.0 | 2.8 | 7.5 | 3.2 | 2.1 | 64.1 |

  S1 alone is 58 % of the total, because of its second power flow per 20 s step (known since
  2026-10-07). How the cost scales beyond 6 parallel processes is **unmeasured**; a scaling test
  (O2, 10 min, at 8/16/24/30 parallel) comes first.

## 1. Operating points for STEP (validation OPs, disjoint from W1–W6)

STEP starts from a **settled state with held profiles**: loads, infeed and the generator dispatch
are frozen at the OP's timestamp. The test disturbance is then the only exogenous change.

Three OPs, picked by a script from the 2016 profile year without the W1–W6 days, each passing
the L2 base-case check:

| OP | Criterion | Purpose |
|---|---|---|
| OP-H | TS load ≥ P90, TS+STS wind ≤ P25 | high support demand, low voltages |
| OP-L | TS load ≤ P25, STS export ≥ P90 | high voltages, STS upper band active (CAIR issue) |
| OP-M | TS load and STS exchange near the median | typical |

The author approves the three timestamps before any STEP run.

## 2. STEP-R: load ramps (voltage tracking during a ramp)

**Question.** How closely does each variant hold the voltages while a load ramps? How does the
error scale with the ramp rate?

**Hypothesis, to be tested.** In the linear regime the ramp-following error of the TS-only loops
(S1, O2) is ∝ ramp rate / loop gain per dispatch, and shows the 180-s sawtooth seen in 0009. The
access variants (S2, O3) should track much more closely, because the STS-OFOs act every 20 s.
Two ramp rates test the linearity.

**Cases (per OP):**
- **R-a (local ramp, one per area):** purely reactive +300 Mvar at one TS load bus per area:
  - area 2: IEEE bus 12, as in REF;
  - areas 1 and 3: the load bus with the largest |∂V/∂Q|, chosen by the script.

  Two rates: 30 min (10 Mvar/min) and 10 min (30 Mvar/min). Profile:

  | Phase | Duration |
  |---|---|
  | settle | 30 min |
  | ramp up | T_R |
  | hold | 30 min |
  | ramp down, same rate | T_R |
  | hold | 30 min |

  The ramp starts 100 s after a dispatch.
- **R-b (system-wide ramp):** all TS loads +10 % P and Q over 60 min, then hold 30 min. A
  "morning pick-up" case.
- **Runs:** 3 OPs × (3 areas × 2 rates + 1) = 21 cases × 6 variants = **126 runs**, about
  **56 core-h**.

**Metrics:**

| Metric | Window | Definition |
|---|---|---|
| Ramp-following error Δe_ramp,a | ramp window | e_v,a(t) − settled pre-ramp e_v,a, as time-RMS and as peak; for e_vp likewise |
| Aggregate | ramp window | RMS over areas |
| Sawtooth amplitude | ramp window | peak-to-peak within each 180-s period (mean, max) |
| Settling after the ramp | after the ramp | time until e_v,a stays within 0.5 mpu of its final value |
| Effort | ramp window | AVR travel, TS-IBG Q moved, taps, banks, STS DER Q moved |
| Recruitment | ramp window | O3 − O2 and S2 − S1 interface ΔQ |
| STS side | ramp window | interface gap e_STS, STS band margin, time above the band by > 0.5 mpu |

**Code needed:**
- a `hold_profiles` flag: the profile frame is held at `start_time` (runner, about 10 lines plus
  a test);
- a **ramp action** for dormant loads in `experiments/helpers/contingency.py`: q(t) = Q_R ·
  clip((t − t0)/T_R, 0, 1), applied every step, for ramp up and ramp down;
- R-b ramps the profile multipliers of the TS load columns on the interpolated frame. That is
  exact at the 20 s resolution, like the existing `load_step_*` mechanism.

## 3. STEP-E: event battery (response and recovery per event)

**Run profile:**

| Time | Action |
|---|---|
| 0–30 min | settle |
| 30 min + 100 s | the event (between two dispatches) |
| 30 min later | reversal, at the same phase |
| 90 min | end |

**Catalogue:** the Ch 7 N−1 catalogue:
- 5 machines;
- 33 lines (the 4 islanding lines excluded);
- a +250 Mvar reactive load at each of the 20 load buses (the REF size).

**Metrics per event and per reversal:**
- impact e(t_e+), measured before any controller acts;
- peak;
- Δe_post and W (the withdrawal);
- n_rec: dispatches until within 1 mpu of the final value;
- final error;
- contraction per dispatch ρ̂, from a log-linear fit of |e(k) − e_final| at dispatch instants,
  set against the loop-gain prediction from the controllers' curvature spectra;
- recruited ΔQ;
- effort;
- STS margin and e_STS.

**Size (author decision):**

| Option | Cases | Runs | Core-h |
|---|---|---|---|
| (a) full catalogue | 58 events × 3 OPs = 174 | 1044 | ≈ 280 |
| (b) severity-stratified | all 5 machines + the 10 lines and 9 load buses (3 per area) with the largest static L2 impact, from a one-power-flow screen per event and OP: 24 events × 3 OPs = 72 | 432 | ≈ 115 |

Recommendation: **(a)**. It is affordable, has no selection step, and gives per-event-type
statistics.

## 4. STEP-P: event-phase sweep (withdrawal, O3 remedy decision)

- **Events (5):**
  - the REF load step (bus 12, 250 Mvar);
  - the G4 trip;
  - a 250 Mvar load step at the R-a buses of areas 1 and 3;
  - the tie 17–27 trip.
- **Phases:** {20, 60, 100, 140, 160} s after a dispatch.
- **Grid:** 3 OPs. Variants O2 (control), O3, S2.
- **Runs:** 75 cases × 3 = **225 runs**, about **25 core-h**.
- **Readout:** Δe_post and W against the remaining time to the next dispatch (180 s − phase).
- **Decision rule, for the author before the run:** the interface-droop remedy is designed if the
  worst-case Δe_post of O3 exceeds an agreed threshold (proposal: 5 mpu, or any excursion out of
  [0.95, 1.05] pu caused by the withdrawal).

## 5. MC: the declared ensemble with the 2026-10-27/28 changes

**Declared 2026-10-07 (unchanged):**
- MASTER_SEED 20261007; episode k has seed MASTER_SEED + k.
- N = 100 valid episodes for every variant, and 300 for O3 (failure bound).
- 180 min per episode; start time uniform over 2016 without the W1–W6 days.
- Event slots every 30 min, each firing with p = 0.25; kinds gen / line / reactive load
  U[100, 400] Mvar; every disturbance reverted after 90 min.
- Discard only if the L2 base-case prescreen fails.
- δ = 1 mpu per area on e_vp; operating range [0.95, 1.05] pu.

**Changes to `mc/run_mc.py`, all following author decisions:**

| | Change |
|---|---|
| a | Configuration from `variants.resolved` (Case-2 registry, paramset band, noise incl. SVR and droop) instead of `run_case1.resolved` |
| b | **Event phase:** drawn per event, uniform on {20, 40, …, 160} s after the slot's dispatch instant, from an independent child stream `default_rng([seed, 1])`. The declared draw (start times, event lists) stays identical; the reversal keeps the phase |
| c | Noise seed per episode = episode seed, shared by all variants |
| d | Metrics from `metrics_case2` (area, u_q, R±, STS, e_STS, Δe_post/W per event, recruited flow) |
| e | Full `log.pkl` kept (≈ 2–3 GB in total); partial log on abort |
| f | All six variants on the same episodes |

**Statistics, declared before the run as one fixed family:**
- **Non-inferiority, e_vp per area, δ = 1 mpu:**
  - S1 vs O3 and S2 vs O3;
  - the Case-1 slice, S1 vs O2.
- **Superiority, δ = 0**, on e_v, worst-bus deviation and t_out per area:
  - O3 vs O2 (access);
  - O2 vs S1 (law);
  - S2 vs S1 (access under the classical law);
  - O3 vs S2 (law with access).
- **Tests:** one-sided Wilcoxon signed-rank plus a bootstrap 95 % CI of the median paired
  difference; Holm correction within the family.
- **Failures:** per class, with Clopper–Pearson bounds.
- **M** is not tested. The O3 − M gap is reported as a distribution.
- **Episode classes** (undisturbed, single, multiple): median and P95 per class.

**Cost:** 100 × 6 runs + 200 extra O3 runs ≈ **353 core-h**.

**Amended 2026-11-01 (author):** comparison window over the first 100 valid episodes on which all six variants complete, an episode with any failure replaced; failures and causes reported separately; a voltage outside [0.90, 1.10] pu at the horizon end is no failure. See `2026-11-01_mc_amendment_window_and_failures.md`.

## 6. Order, total cost

| Step | Content | Core-h |
|---|---|---|
| 1 | Code (worktree, tests): hold_profiles, ramp action, OP picker, STEP producer, run_mc migration, ramp/ρ̂/settling metrics | — |
| 2 | e5power scaling test, smoke runs of STEP and MC (10 min) | small |
| 3 | STEP-R (answers the ramp question first) | 56 |
| 4 | STEP-P | 25 |
| 5 | STEP-E | 115–280 |
| 6 | MC (overnight) | 353 |

Total ≈ 550–715 core-h. That is about 35–45 h of wall time at 16 workers, and less if the scaling
test allows more.

## 7. Evaluations (figures and tables)

**REF (illustration; few figures):** per-area error and the aggregate, the STS voltage deviation,
interface tracking, tie flows, a compact scorecard (error against effort), and one zoom on the
load step that shows the withdrawal.

**STEP:**
- Δe_ramp against ramp rate per variant (log-log; slope ≈ 1 in the linear regime);
- ramp-window traces for one case;
- Δe_post against phase (O2, O3, S2);
- ρ̂ against the predicted contraction;
- the n_rec distribution per event type;
- recruited ΔQ against STS headroom.

**MC (statistical):**
- paired-difference distributions per area with the δ line (NI/superiority at a glance);
- ECDFs of episode e_v per variant;
- per-class medians and P95;
- failures with Clopper–Pearson bars;
- the error–effort plane per episode (Pareto view: what each variant pays for its error);
- the O3 − M gap ("cost of decomposition") against the number of events;
- Δe_post against the drawn phase over all events (H-NMP);
- the STS band and e_STS distributions (the CAIR voltage blind spot).

## 8. Open decisions for the author

1. OP criteria (§1); then approve the three picked timestamps.
2. Ramp design: purely reactive vs P+Q, +300 Mvar, the two rates, the bus choice in areas 1 and 3,
   and R-b at +10 %.
3. STEP-E size: full (a) or stratified (b).
4. STEP-P decision threshold for the interface-droop remedy.
5. MC phase per event (proposed) vs per episode (the wording of 2026-10-27).
6. Thesis label for STEP: a new subsection `ch:case3:step`, or within `ch:case3:robustness:n1`.
7. Whether R1 (γ) and R4 (noise levels) stay in this campaign. They need the γ hooks (not built)
   and would run on the STEP-P/STEP-E subset.
