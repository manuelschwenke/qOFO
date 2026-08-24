# 2026-08-21 - Corrected Experiment B (`N_inner`) at 10 Mvar

Timestamp: 2026-08-21 14:56 CEST

## Reason

Reassess Chapter 9 Experiment B with supervisory interface-Q steps representative of the TSO controller, reduce the design from 240 to 120 cases without dropping operating-point coverage, audit unexpectedly slow spring cases, and persist a dissertation-grade record.

## Established findings before rerun

- The historical capability-edge design used much larger commands than normal supervisory operation: median 49.865 Mvar, p90 111.117 Mvar, maximum 145.820 Mvar; 204/240 steps exceeded 20 Mvar.
- In the archived TSO `T_TS = 180 s` run and the ten admissible Experiment-B windows, 3,597/3,600 command increments (99.9167%) were at most 10 Mvar. Only three exceeded 10 Mvar and two exceeded 20 Mvar.
- Therefore 10 Mvar was selected as the single primary maximum. This is an empirical envelope, not a formal controller bound.

## Correctness defect found

`ch_9_1_ninner_isolated_sts.py` lexically sorted record transformer keys. For DSO 4 this produced `[trafo_10, trafo_11, trafo_9]`, which did not match the controller's numeric coupling-transformer order `[9, 10, 11]`. The commanded setpoint could be applied to a sibling interface while the intended focus interface was measured.

Consequence: all DSO-4 rows from pre-correction runs are invalid. The initial 10 Mvar run (`results/ch9_ninner/stepcap_10mvar/20260821-121619`) and the stopped 20 Mvar run must not be used as aggregate dissertation evidence.

## Code changes

Modified `experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py`:

- numerically order transformer keys by global index
- add a post-run invariant verifying every delivered setpoint against the intended vector to `1e-6 Mvar`
- add `--max-step-mvar` while preserving the raw capability-edge target
- add `--case-design balanced_half` with one direction per triple, balanced to 60 up / 60 down and five of each direction per interface
- record raw/applied step, cap status, delivery error, and per-case wall time
- add targeted `--focus-trafo` and `--direction` diagnostics
- add optional signed trajectory persistence with `--save-trajectories`
- save a script snapshot and SHA-256 digest with each run
- correct the written definition of `N_inner` to the persistent per-iteration-flow-change criterion

## Verification

- offline script self-tests passed, including numeric ordering, step capping, balanced-half allocation, trajectory signs, and output schema
- Python compilation passed in the `qOFO_clean` environment
- Git whitespace/diff check passed (apart from unrelated line-ending warnings elsewhere in the dirty tree)
- corrected DSO-4 ordering probe delivered exact 20 Mvar up/down separation on `trafo_9`, `trafo_10`, and `trafo_11`
- primary run: zero case failures and maximum absolute command-delivery error 0 Mvar

## Primary run

Command:

```text
python experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py --workers 16 --label stepcap_10mvar_balanced_half_corrected --max-step-mvar 10 --flat-mvar 1.0 --case-design balanced_half
```

Original output: `results/ch9_ninner/stepcap_10mvar_balanced_half_corrected/20260821-134412`

Persistent copy: `docs/data/2026-08-21_ch9_ninner_10mvar_corrected/main_run/`

Design and result:

- 120 cases = 10 windows x 4 DSOs x 3 interfaces x one balanced direction
- 113/120 targets capped at 10 Mvar
- all cases: median 4, p80 9, p90 32.5, p95 46, maximum 46; 12/120 right-censored
- non-censored primary population: n = 108, median 4, p80 6, p90 9, p95 13.3, maximum 31
- 98/108 = 90.74% of the non-censored cases satisfy `N_inner <= 9`
- all 12 censored cases occurred in DSO 4

## Exclusion decision

For the present `N_inner` investigation, exclude individual cases with `censored == True` from the percentile estimate. Retain those rows in `nonconvergent_cases.csv` and state the 10% censoring rate next to the conditional percentile. Do not remove complete windows: doing so is a post-hoc operating-point exclusion and would lower p90 from 9 to 4.

The label is intentionally `non-convergent/censored`, not universally `oscillatory`. Censoring is not itself an oscillation detector.

## Slow-case audit

The slow behavior is not a generic spring effect. `d_ramp_down_spring` had p90 = 1 and no censoring. The difficult groups were concentrated in DSO 4, particularly generator-trip, quiet, reversal, and some ramp-up/ramp-down operating points. Non-spring DSO-4 cases also censored.

Across all 12 censored rows:

- no capability-limited cases
- no recorded voltage-limit violations
- zero OLTC moves
- exact command delivery
- retained capability headroom 26-146 Mvar
- final tracking residual magnitude 0.58-2.98 Mvar

This rules out an unreachable setpoint, voltage clipping, tap motion, and command mapping as direct explanations. Persistent continuous-loop motion remains the leading mechanism.

## Confirmed period-2 diagnostic

A targeted signed-trajectory run was executed for `d_reversal_spring / DSO_4 / trafo_10 / up`:

`results/ch9_ninner/stepcap_10mvar_reversal_spring_dso4_t10_up_trace/20260821-143528`

Persistent copy: `docs/data/2026-08-21_ch9_ninner_10mvar_corrected/confirmed_period2_trace/`

The actual interface Q alternated around the 4.141 Mvar setpoint with growing amplitude. All 44 evaluated signed increments changed sign; lag-1 correlation was -0.987 and lag-2 correlation +0.995. The last six values were approximately `[6.886, 1.319, 6.908, 1.298, 6.931, 1.275] Mvar`. This is a confirmed slightly divergent period-2 response for that case.

## Assumptions and controlled system

- quasi-steady-state plant simulation
- subordinate controller period 20 s
- parent/supervisory setpoint frozen during the response
- 600 s pre-step settling and 900 s observation
- 1 Mvar persistent flatness threshold
- controlled outputs: EHV-HV interface reactive-power flow and constrained DSO voltages
- controller acts through configured DSO resources using cached sensitivities; observed OLTC movement was zero
- selected Stage-1 candidate `fe010aa3ead1`; its weights were calibrated with `N_inner = 9` assumed

## Decision and unresolved risks

No OFO gain/weight change was made in this investigation. Reducing aggressiveness changes the selected controller and should be a separate stability-sensitivity experiment with its own tracking-performance comparison.

Remaining risks/open questions:

- the conditional p90 = 9 does not establish convergence for the censored 10%
- only one censored response has a persisted signed trajectory proving period-2 behavior
- the balanced-half design cannot estimate paired direction effects at every operating point
- a formal TSO command maximum still requires an explicit slew constraint or external operational requirement
- run metadata reports a dirty working tree; exact scripts are archived, but full bitwise reproduction depends on contemporaneous imported modules and the simulation environment
## Addendum - 15:58 CEST: paired DSO-4 relief factor 10 sensitivity

### Motivation

All 12 censored rows in the corrected factor-20 run were DSO 4. Manuel proposed that `dso_v_authority = 20` may be too strong and that 10 may suffice.

### Reproducibility decision

The live working tree was not used because concurrent uncommitted configuration/refactor changes failed the selected-weight guard (`g_w_dso_der` rebuilt as 552.53 instead of 549.88 and `g_w_dso_oltc` as 1241.65 instead of 392.64). Running there would not be an A/B comparison.

An isolated Git archive of commit `5d8e0f31c2aa5b4256782f6709ce4516142a2bdf` was created and overlaid with the exact corrected Experiment-B script snapshot plus the archived selected-candidate JSON files. The clean builder reproduced all five selected weights exactly.

A diagnostic-only CLI override was applied inside that temporary snapshot. It first removed the generated factor-20 per-area pair, then re-applied factor 10 to DSO 2/4. This is necessary because the historical `apply_dso_v_relief` implementation is not idempotent on an already-relieved `dso_g_w_class`. Assertions verified:

- `dso_g_v_per_area = 841395.1416451951` on DSO 2/4
- per-area `g_w_dso_oltc = 3926.443639860221`
- the paired 30-case DSO-4 grid exactly matched the factor-20 reference

### Command

```text
python ch_9_1_ninner_isolated_sts.py --workers 16 --label dso4_relief10_paired --max-step-mvar 10 --flat-mvar 1.0 --case-design balanced_half --dsos DSO_4 --dso-v-authority-override 10
```

### Result

| DSO-4 metric | Factor 20 | Factor 10 |
|---|---:|---:|
| cases | 30 | 30 |
| censored | 12 (40.0%) | 7 (23.3%) |
| non-censored median | 2 | 2 |
| non-censored p90 | 10.0 | 25.6 |
| non-censored p95 | 25.05 | 29.6 |
| cases `N_inner <= 9` | 16 | 16 |

Paired transitions: five rescued, seven still censored, zero newly censored. The five rescued cases settled at 18, 24, 26, 30 and 44 iterations. Exploratory paired exact test: one-sided `p = 0.03125`, two-sided `p = 0.0625`; this is post-hoc and not confirmatory.

Persistent failures: all three `d_gen_trip_spring`, all three `d_ramp_down_summer`, and `d_quiet_spring / DSO_4 / trafo_10 / up`.

Both variants had zero tap moves, zero voltage violations, zero slack, and exact command delivery. The maximum recorded DSO voltage rose slightly from 1.05895 to 1.06149 pu at factor 10.

Replacing only DSO-4 rows in the full population gives 7/120 censored instead of 12/120. Among non-censored cases p90 changes 9 -> 12 and p95 13.3 -> 16.2; the unconditional number at `N_inner <= 9` remains 98/120.

### Interpretation and decision

Established: factor 20 contributes to five DSO-4 failures, but is not their sole cause. DSO 2 used the same factor without censoring, and seven DSO-4 cases barely respond to halving it.

Factor 10 is therefore not enough for either universal convergence or the nine-iteration requirement. It is a better stability point in the narrow sense of censoring, but it is not adopted as the selected controller: full-bank voltage headroom and objective performance have not been revalidated.

Hypothesis for the seven persistent cases: an interface-Q/cached-sensitivity mode in the continuous DSO-4 DER loop, rather than the voltage-relief term. A separate targeted sensitivity should increase the DSO-4 `dso_der` step penalty or reduce its effective OFO step size. This was not implemented because it changes another selected controller parameter.

Artifacts: `docs/data/2026-08-21_ch9_ninner_10mvar_corrected/relief_factor_10_diagnostic/`.
