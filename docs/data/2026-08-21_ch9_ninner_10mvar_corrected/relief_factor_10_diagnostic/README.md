# Diagnostic: DSO-4 voltage-relief factor 20 versus 10

Date: 2026-08-21

## Question

Do the 12 DSO-4 right-censored cases in the corrected 10 Mvar `N_inner` experiment result from the selected `dso_v_authority = 20`, and is factor 10 sufficient?

## Short answer

Factor 20 contributes to the DSO-4 non-convergence, but it is not the sole cause. Reducing the factor to 10 rescued five of the twelve censored cases and introduced no new censored cases. Seven cases remained censored. Moreover, the number of DSO-4 cases satisfying `N_inner <= 9` remained exactly 16/30.

Therefore factor 10 improves the stability margin, but is not sufficient if the requirement is either (a) convergence of every DSO-4 case or (b) settlement within nine subordinate iterations.

## Provenance and paired design

The live working tree was not used: concurrent uncommitted changes no longer reproduced the archived selected weights. The diagnostic was executed in an isolated export of commit `5d8e0f31c2aa5b4256782f6709ce4516142a2bdf`, overlaid with the exact corrected Experiment-B script snapshot. The isolated builder reproduced all five archived selected weights exactly.

The factor-10 grid contains the same 30 `(window, DSO_4, interface, direction)` cells as the factor-20 reference. A diagnostic-only runtime override stripped the generated factor-20 per-area pair and re-derived factor 10 from the selected base scalars. This avoided compounding the historical non-idempotent relief implementation.

Command:

```text
python ch_9_1_ninner_isolated_sts.py --workers 16 --label dso4_relief10_paired --max-step-mvar 10 --flat-mvar 1.0 --case-design balanced_half --dsos DSO_4 --dso-v-authority-override 10
```

## What the factor changes

| Quantity on DSO 2/4 | Factor 20 | Factor 10 |
|---|---:|---:|
| `dso_g_v` | 1,682,790.28 | 841,395.14 |
| `g_w_dso_oltc` | 7,852.89 | 3,926.44 |
| `dso_g_v / g_w_dso_oltc` | unchanged | unchanged |
| `dso_g_v / g_w_dso_der` | 3,060.28 | 1,530.14 |

The OLTC voltage-loop ratio is preserved by construction. The continuous DER regularization `g_w_dso_der = 549.882` is not scaled, so factor 10 halves the voltage-gradient authority acting on the continuous DER block. The run used `dso_gamma_oltc_q = 0` and no per-area `g_q` scaling. Recorded OLTC moves were zero in both variants.

## Paired result

| Scope and metric | Factor 20 | Factor 10 |
|---|---:|---:|
| DSO-4 cases | 30 | 30 |
| Right-censored | 12 (40.0%) | 7 (23.3%) |
| Non-censored | 18 | 23 |
| Non-censored median | 2 | 2 |
| Non-censored p90 | 10.0 | 25.6 |
| Non-censored p95 | 25.05 | 29.6 |
| Non-censored maximum | 31 | 44 |
| All cases with `N_inner <= 9` | 16/30 | 16/30 |
| Voltage violations | 0 | 0 |
| OLTC moves | 0 | 0 |
| Maximum DSO voltage | 1.05895 pu | 1.06149 pu |

Paired censoring transitions:

- five factor-20 censored cases converged at factor 10
- seven remained censored
- no factor-20 converged case became censored

The five rescued cases were the two `d_gen_trip_summer` cells, one `d_quiet_spring` cell, and two `d_reversal_spring` cells. They settled late at 18, 24, 26, 30, and 44 iterations.

The seven persistent cases were all three `d_gen_trip_spring` cells, all three `d_ramp_down_summer` cells, and `d_quiet_spring / trafo_10 / up`.

The paired directional exact test gives one-sided `p = 0.03125` and two-sided `p = 0.0625` for five rescues versus zero new failures. This is exploratory: factor 10 was proposed after observing the factor-20 failures, and the sample is small.

## Consequence for the full 120-case population

Replacing only the 30 DSO-4 rows by factor-10 results while leaving DSO 1-3 unchanged gives:

| Metric among non-censored cases | Factor 20 | Factor 10 |
|---|---:|---:|
| Non-censored cases | 108 | 113 |
| Censoring fraction | 10.0% | 5.83% |
| Median | 4 | 4 |
| p90 | 9 | 12 |
| p95 | 13.3 | 16.2 |
| Cases satisfying `N_inner <= 9` (unconditional) | 98/120 | 98/120 |

The conditional p90 rises because factor 10 converts five censored responses into measurable but slow responses. This is a selection effect, not evidence that the factor-10 controller is generally slower on cases that already converged; the ramp-up-spring tail improved from 24/31 to 14/15 iterations.

## Interpretation

Established:

- Every censored case in the original corrected experiment was in DSO 4.
- DSO 2 received the same factor 20 but had no censored cases, so the factor cannot explain failure without the DSO-specific plant/sensitivity interaction.
- Halving the relief rescues five cases with no newly censored case, so factor 20 contributes causally to part of the DSO-4 instability.
- Seven cases are almost insensitive to the change: their final per-iteration motion remains close to the factor-20 value. Those modes are not primarily controlled by the voltage-relief factor.
- No tap, voltage-limit, capability, slack, or command-delivery mechanism explains either variant's censoring.

Hypotheses:

- The rescued cases contain a voltage-authority-driven mode close to the period-2 boundary; halving the voltage term moves it just inside the boundary, hence settlement at 18-44 iterations.
- The persistent generator-trip-spring and ramp-down-summer modes are dominated by the interface-Q loop and cached-sensitivity/model mismatch of the continuous DSO-4 DER block.

Open questions and risks:

- The isolated 10 Mvar experiment does not establish that factor 10 provides enough voltage headroom over the full scenario bank or a 24 h run. Maximum voltage increased slightly, though it remained feasible here.
- Factor 20 was part of the selected Stage-1 trade-off and improved voltage margin. Replacing it with 10 changes the selected controller and requires objective/headroom re-evaluation.
- Lowering the factor further may rescue additional voltage-driven cases but is unlikely to fix the seven cases that barely changed.
- The next targeted lever should damp the DSO-4 continuous DER update, for example a per-area increase of `g_w_dso_der` or a DSO-specific step-size reduction. That is a separate controller sensitivity and was not mixed into this diagnostic.

## File inventory

- `steps.csv`, `cases.csv`, `n_inner_summary.csv`, `run_meta.json`, `summary.md`: exact factor-10 raw output
- `script_snapshot.py`: exact executed diagnostic script
- `selected_design_builder.py`: clean selected-design builder used by the run
- `relief_override.diff`, `subset_index.diff`: the two explicit diagnostic changes
- `diagnostic_context.json`: source commit, hashes, command, and paired-test context
- `paired_factor20_factor10.csv`: complete 30-row paired table
- `rescued_cases.csv`: five factor-20 failures that converge at factor 10
- `still_censored_cases.csv`: seven persistent failures
- `window_comparison.csv`: paired results by disturbance window
- `aggregate_comparison.csv`: DSO-4 and combined-120 summaries
