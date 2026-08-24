# Experiment B rerun: `N_inner` under 10 Mvar supervisory steps

Date: 2026-08-21

## Decision and headline result

The primary dissertation estimate excludes individual right-censored cases (`censored == True`) but retains every experimental window and reports the excluded population separately. This is a conditional estimate:

- converged/non-censored cases: 108 of 120
- median: 4 subordinate iterations
- p80: 6
- p90: **9**
- p95: 13.3
- maximum: 31
- 98/108 = 90.74% of the retained cases satisfy `N_inner <= 9`

Thus, for this controller parameterization and a maximum 10 Mvar interface-Q command step, `N_inner = 9` covers approximately 90% of the cases that converge within the observation horizon. It is not evidence that all operating points are stable: 12/120 cases (10%) were right-censored and remain a separately reported result.

## Why 10 Mvar was selected

The earlier capability-edge experiment did not represent normal supervisory commands. Its 240 commanded steps had median 49.865 Mvar, p90 111.117 Mvar, and maximum 145.820 Mvar; 204/240 exceeded 20 Mvar.

In the archived TSO run at `T_TS = 180 s`, the empirical absolute interface setpoint increments for the ten Experiment-B-admissible windows were:

- 3,600 commands in total
- median 0.253 Mvar, p90 1.468 Mvar, p95 2.153 Mvar, p99 3.733 Mvar
- 3,597/3,600 = 99.9167% were at most 10 Mvar
- 3 exceeded 10 Mvar and 2 exceeded 20 Mvar

Consequently, 10 Mvar is used as an empirical operational envelope. A 20 Mvar cap would include only one additional observed command in this sample while approximately doubling the imposed stress for most cases. This is empirical support, not a formal upper bound: the present supervisory controller has no explicit PCC-Q slew constraint.

The exact source and statistics are in `tso_command_step_envelope.csv`.

## Experimental design

The corrected primary run is archived in `main_run/` and was launched as:

```text
python experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py --workers 16 --label stepcap_10mvar_balanced_half_corrected --max-step-mvar 10 --flat-mvar 1.0 --case-design balanced_half
```

Design properties:

- 10 admissible disturbance windows
- 4 DSOs
- 3 EHV-HV interfaces per DSO
- one direction per `(window, DSO, interface)` triple, for 120 cases
- checkerboard allocation yielding 60 upward and 60 downward steps
- each interface receives five upward and five downward steps across the ten windows
- 113/120 raw capability-edge targets were capped; applied step magnitudes ranged from 6.038 to 10 Mvar

The half-fraction design preserves coverage of all windows, DSOs, interfaces, and directions. It does not preserve a paired up/down comparison at every operating point; a full design would be needed for that interaction.

## Assumptions, constraints, actuators, and outputs

- Quasi-steady-state simulation with `T_STS = 20 s`, 600 s pre-step settling, and 900 s post-step observation.
- The supervisory/parent command is frozen during each response, isolating the subordinate DSO controller.
- The imposed setpoint change is capped at 10 Mvar but remains inside the reported interface capability direction.
- `N_inner` is the first subordinate iteration from which the per-iteration interface-flow change remains at or below 1 Mvar for the rest of the observation horizon. It is a flatness criterion, not an absolute tracking-error criterion.
- Controlled outputs are interface reactive-power flow and constrained DSO voltages.
- The configured local controller can use its DSO actuators and cached sensitivities; it does not observe the plant model directly. In this run, the recorded OLTC movement count was zero.
- The final selected Stage-1 controller weights were used (candidate `fe010aa3ead1`). These weights were calibrated assuming `N_inner = 9`; this rerun checks that assumption but is not statistically independent of it.

## Exclusion rule and oscillation audit

The analysis excludes individual rows only when `censored == True`. It does **not** remove every row belonging to a window that contains one censored case. Removing whole windows would discard valid observations post hoc and changes the p90 from 9 to 4.

All 12 censored cases occurred in DSO 4. They were not capability-limited, had no recorded voltage-limit violation, had zero OLTC moves, and had exact command delivery. Their retained capability headroom was 26-146 Mvar, while final tracking residuals were 0.58-2.98 Mvar. This points to persistent continuous-loop motion rather than an unreachable target, voltage constraint, or tap action.

A targeted rerun of `d_reversal_spring / DSO_4 / trafo_10 / up` confirmed a slightly growing period-2 response. The interface flow alternated about the 4.141 Mvar setpoint; the last values were approximately 6.886, 1.319, 6.908, 1.298, 6.931, and 1.275 Mvar. The signed increments flipped sign on all 44 evaluated transitions, with lag-1 correlation -0.987 and lag-2 correlation +0.995. Its exact trajectory is archived in `confirmed_period2_trace/trajectories.csv`.

`censored == True` is nevertheless a non-convergence flag, not a mathematical oscillation detector. The representative case establishes oscillation for that case only. Confirming the temporal mode of every excluded row would require trajectory exports for all 12.

## Correctness fix

The prior script lexically sorted transformer keys and therefore ordered DSO 4 as `trafo_10, trafo_11, trafo_9` instead of the controller's numeric order `trafo_9, trafo_10, trafo_11`. In pre-correction runs, a DSO-4 command could be delivered to a sibling while another interface was measured. All pre-correction DSO-4 rows and aggregates containing them are invalid for inference.

The persistent script now:

- orders transformer keys by numeric global index
- verifies every delivered interface setpoint against the intended command vector to within `1e-6 Mvar`
- records command-delivery error, raw and applied steps, cap status, and case wall time
- supports `--max-step-mvar`, `--case-design balanced_half`, targeted interface/direction filters, and optional signed trajectory export
- saves the exact executed script and its SHA-256 digest with every run

The corrected 120-case run had zero delivery errors and zero execution failures.

## Population results

| Population | n | Median | p80 | p90 | p95 | Max | `N_inner > 9` | Censored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All cases | 120 | 4 | 9 | 32.5 | 46 | 46 | 22 | 12 |
| Non-censored only (primary) | 108 | 4 | 6 | **9** | 13.3 | 31 | 10 | 0 |
| Non-censored DSO 1-3 | 90 | 4 | 6 | **9** | 12 | 22 | 8 | 0 |
| Windows without any censoring (diagnostic only) | 60 | 1 | 2 | 4 | 8.05 | 31 | 2 | 0 |

The all-case p90 is dominated by the right-censoring boundary and must not be interpreted as a measured settling-time quantile.

## Interpretation and unresolved points

- Established: the originally imposed steps were far larger than nearly all observed supervisory commands.
- Established: conditional on non-censoring, p90 is 9 iterations for the corrected 10 Mvar balanced-half experiment.
- Established: DSO 4 causes all censoring; one representative hard case is a period-2 oscillation.
- Hypothesis: DSO-4 sensitivity/model mismatch or controller gain coupling at particular operating points produces the oscillatory mode.
- Open: whether a modest reduction in OFO aggressiveness removes the DSO-4 mode without degrading tracking or the selected Stage-1 objective.
- Open: a formal maximum TSO command step requires either a stated operational requirement or an explicit constraint such as `|Delta Q_PCC,set| <= Delta Q_max`; the empirical distribution alone cannot establish a hard bound.
- Risk: tuning OFO aggressiveness would define a different controller and therefore a separate sensitivity experiment. It was intentionally not mixed into this `N_inner` rerun.
- Risk: the recorded Git tree was dirty. The exact experiment script and metadata are archived, but full bitwise reproduction also depends on the contemporaneous imported project modules and simulation environment.

## File inventory

- `main_run/`: exact 120-case raw outputs, run metadata, summary, and script snapshot
- `confirmed_period2_trace/`: exact single-case diagnostic outputs and signed trajectory
- `converged_only_steps.csv`: the 108-row primary analysis population
- `nonconvergent_cases.csv`: the 12 excluded rows retained for audit
- `population_summary.csv`: unconditional and conditional percentile tables
- `converged_by_dso_summary.csv`: conditional results by DSO
- `window_summary.csv`: results and diagnostic fields by disturbance window
- `tso_command_step_envelope.csv`: empirical TSO command-step justification
## Addendum: diagnostic voltage-relief factor 10

A paired 30-case DSO-4 sensitivity was run at `dso_v_authority = 10` in an isolated clean snapshot. It used the same 10 Mvar steps and the same `(window, interface, direction)` cells as the factor-20 reference.

| DSO-4 result | Factor 20 | Factor 10 |
|---|---:|---:|
| Censored | 12/30 | 7/30 |
| Non-censored p90 | 10.0 | 25.6 |
| `N_inner <= 9` over all 30 | 16 | 16 |

Five censored cases were rescued and none newly failed, so factor 20 contributes to the DSO-4 instability. Factor 10 is nevertheless not sufficient: seven cases remain censored and the number satisfying the nine-iteration requirement is unchanged. Across the combined 120-case population, censoring would fall from 10.0% to 5.83%, while the conditional p90 rises from 9 to 12 because the rescued cases settle at 18-44 iterations.

See `relief_factor_10_diagnostic/README.md` and its paired/raw tables. This is a diagnostic controller variant, not a replacement of the selected Stage-1 controller.
