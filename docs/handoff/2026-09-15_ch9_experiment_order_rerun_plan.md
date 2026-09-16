# Chapter 9: experiment order and rerun plan

Date: 2026-09-15. Status: design agreed for restructuring; simulations not launched.
This plan separates proposed runs from verified existing evidence. Reordering the chapter does not establish a historical execution sequence.

## 1. Order and names

| New | Purpose | Previous role |
|---|---|---|
| A | Plant settling -> T_STS | A, retained |
| B | Analytical starting weights | Analytical part of old D |
| C | Initial subordinate response -> provisional N_inner and T_TS | New initial-weight version of old B |
| D | Final weight selection at that cadence | Performance-selection part of old D |
| E | Subordinate-period confirmation at final weights | Old B, if configuration matches |
| F1 | Smallest deadband with no routine droop activation | Redesigned old C1 |
| F2 | Disturbance-response demonstration at that deadband | Redesigned old C2; earlier appendix demonstration as a method reference |

Dependencies: A -> B -> C -> D -> E -> F1 -> F2.
If E rejects the provisional cadence, revisit C/D and confirm again before F1.
No changes to old filenames, source archives, run IDs or measurements are implied by these names.

## 2. Reuse versus rerun

| Study | Decision | Reason / closure condition |
|---|---|---|
| A | Retain | Corrected plant-settling evidence supports T_STS=20 s. Reordering and changes to OFO weights do not invalidate this open-loop experiment. Changes to the physical dynamic model, primary settings or actuator timing need targeted checks. |
| B | Reuse or regenerate analytical export | A starting point exists as stage1_start. Verify/rebuild its local screen on the common frozen network and intended inactive-droop model. No RMS campaign is required. Archive the screen bound actually used. |
| C | New closed-loop QSS campaign | The chapter has no verified initial-response run at these analytical starting weights. An old final-weight test cannot be relabelled as this result. |
| D | Audit and reuse existing search where applicable | Do not repeat the entire search just to change the order. Confirm its cadence, network, sensitivity construction, local droop regime and auxiliary settings. If C changes cadence or provenance differs, first run selected point + analytical baseline on the corrected configuration; reopen the search only if needed. |
| E | Conditional reuse | The archived 20260909-190213 response run confirms the older frozen weights only. Retain it if those exact weights and relevant settings are chosen. If adopting the current ch9_selected point, repeat E. |
| F1 | New RMS campaign | Old analysis used 20 s anchors at both levels, removed switching intervals and allowed 0.1% crossings; it cannot establish the revised claim. |
| F2 | Correct delivery/export path, then new RMS campaign | The previous complete exports are identical across nominal events and widths. They are not valid disturbance-response evidence. |

**The user's proposed minimum is correct only if final weights remain those of the archived E run:** retain A and E, supply C at initial weights, then new F1 and F2, with B/D provenance checks.
**With the newer working weight selection, the required simulation set is C + E + F1 + F2**, plus targeted D validation if its network/settings differ.

The initial C run is required to present an empirically established initial-period selection. Without it, an honest alternative is to call N_inner=9 an initial assumption and E its subsequent confirmation. A reconstructed C run must be described as a retrospective verification, not as the historical source of the already-run D campaign.

## 3. Configuration freeze before runs

Create separate complete resolved manifests for the analytical and final controllers. They must differ in the declared weights only, except for explicitly derived time-dependent settings.

Record:
- executable source snapshot and revision, resolved parameters and fingerprints;
- interconnected rural_700 benchmark, topology, device catalogue and DSO_3 DER/active-load multipliers of 2, with reactive-load scaling recorded separately;
- T_STS=20 s, provisional N_inner=9 and T_TS=180 s;
- all per-area voltage, Q and movement weights; shunt integrator gain, hysteresis, delay and dwell rules; actual actuator bounds;
- nominal droop regime used by analytical screening/tuning, gain law S_n/m, deadband, capability saturation, measurement filtering and anchor updates;
- voltage/current limits, profile source and exact timestamps, measurement noise, solver and export settings;
- controller cached sensitivities and their update policy, with no extra plant information supplied to controllers.

Known evidence conflicts that must not be hidden:
1. Dissertation September batch: final weight candidate fe010aa3ead1, global subordinate DER/OLTC weights 549.8818105 / 392.6443640, with per-area corrections.
2. Current qOFO_2027 working selection: ch9_selected identifies 399e30d3b60b. This is a different weight vector; E cannot be transferred.
3. _ch9_selected_design.py still defaults to stage1_start despite ch9_selected being present.
4. The analytical factory used by the newer search is make_config(); the response factory additionally applies the frozen DSO_3 multipliers. The search's network must be checked against its actual run records before transfer; identical scenario names are insufficient.
5. shunt_int_g_w=0.03 was added to parameter-set files after prior runs in which shunts were inert. Check each accepted run's actual value and switching activity.
6. The old 1.5 spectral screen and newer 1.6 screen belong to different campaign records. Do not combine their numbers or silently change a requirement.

## 4. Preparation tasks (implementation work, not performed in this edit)

### P1. Make controller identity explicit
- Add an explicit initial/final parameter-set argument to the subordinate-response entry point, passed into parent and worker configurations.
- Update stale reads of design['campaign'], ['candidate_key'] and ['verified'] to the new provenance schema.
- Persist resolved configuration in every run; do not rely on the helper's current default.
- Use one structural configuration for screening, response tests, tuning and RMS, with only documented experimental overrides.

### P2. Repair and instrument RMS delivery
- Remove the legacy --dso-oltc-switch-cost 200 override from new campaigns; use the chosen resolved per-area weights.
- Add/export actual anchor values, effective anchor-update times, filtered dead-zone input, unsaturated incremental droop command, saturated reference and delivered Q separately.
- For a single diagnostic event, verify realised event time, load Q before/after, target identity and installed deadband from the plant itself.
- Check that fresh full-resolution outputs belong to their run. Identical hashes are an investigation trigger; differing hashes alone are not proof that an event occurred.
- Pass one diagnostic event before launching the full F2 bank.

These are necessary experiment-runner preparations; no controller architecture change is proposed here.

## 5. Experiment C and conditional E

Model: QSS subordinate OFO with the supervisor held after initialization. Physical capability, voltages/current limits, continuous DER and discrete taps remain enabled.

Run the same request bank twice only when E cannot be reused:
- C: analytical weights from B.
- E: final weights from D.

Use the exact archived request-bank timestamps and interfaces for comparability. The existing evidence archive contains 288 planned cases / 240 measured responses, including no-band records; do not describe that as a newly completed six-window bank without reconciling its actual timestamp list. Include W1-W6 explicitly in the new manifest if that is the intended scope.

Protocol:
- 600 s before request step, 900 s after; T_STS=20 s.
- Both directions, up to 10 Mvar at each interface, respecting real capability.
- Flatness: first iteration after which |Q(k)-Q(k-1)| <= 1 Mvar remains true through iteration 45.
- Also report distance to final response, request residual, constraint activation, delivered request amplitude and discrete moves.
- Report every unavailable/clipped/censored case and coverage per STS/direction, not only a pooled percentile.
- Keep the declared engineering coverage rule. The old n=9 result is 227/240=94.6%; it is not 100% coverage or a strict >=95% result.
- If C supports n=9, existing D at 180 s need not be rerun solely for chronology. If it does not, evaluate the period/weight pair before proceeding.
- If E passes under the same requirement, fix the cadence and proceed. If not, revisit weights or ratio.

Deliverables: resolved config, exact case manifest, per-interface trajectories, flatness CDF, per-group coverage, no-band/censoring table, and selected ratio with its scope.

## 6. Experiment F1: select and confirm routine inactivity

Scope: the whole OFO hierarchy in RMS, all usual actuators and constraints, no external contingency. Ordinary taps and shunt operations are retained.

Recommended bank:
- Six uninterrupted 2 h traces beginning at the W1-W6 initial timestamps. This includes the later W2/W6 hours instead of cold-starting there.
- Declare the startup exclusion before reduction; use the same rule in every leg and report startup separately. Do not infer convergence solely from 180 s having elapsed.
- A run with no normal switching cannot establish switching non-interference. Extend duration or add a declared ordinary operating scenario if necessary, rather than silently removing the actuator.

F1-reference: six droop-suppressed traces.
- OFO remains active; use the same nominal cached-model construction and update policy intended for inactive droop.
- Measure max |U_filtered-U_anchor| between each device's actual effective anchor updates.
- Nominal TS interval is 180 s and STS interval 20 s; actual logged timing governs.
- Include all routine continuous and discrete movements, including delayed contacts.
- Choose a common half-width from the maximum across all devices and windows plus a declared numerical/measurement margin.
- Start with the former candidate grid as a comparison, but extend/refine it around the measured maximum as needed. Do not preselect 0.01 pu.
- Freeze the margin and pass tolerances from signal/export/solver accuracy before comparing candidates.

F1-confirmation: six matched traces with the selected width active from initialization.
- Verify zero dead-zone activation and zero incremental droop command within the declared numerical tolerances.
- Check matched voltage and interface-Q trajectories, actuator commands and discrete switching.
- Account for converter tracking/filter dynamics when interpreting Q_actual-Q_set; it is not itself a pure droop signal.
- Report zero-headroom devices separately: saturation must not hide an active dead-zone command.

Recommended additional comparison: six traces with the adjacent narrower candidate, to demonstrate the onset of interaction.

Size: **12 mandatory RMS traces; 18 with the narrower-width comparison.**
No multi-slope campaign is needed for the inactive reference. A separate noise robustness bank can follow; zero observed activations is not a universal probabilistic guarantee.

## 7. Experiment F2: disturbance demonstration

Baseline design:
- W1-W6.
- Three declared reactive disturbances inherited from the earlier plan: +200 Mvar at TS bus 7; +200 and -200 Mvar at TS bus 11.
- Three legs: F1-selected width, one wider width, droop disabled.
- Fixed slope and converter ratings; use S_n/m with separate capability saturation.
- **6 x 3 x 3 = 54 RMS cells**, including all nine W5 boundary-case cells and their references.
- Choose event time by a stated phase relative to actual dispatch/re-anchoring, such as 0.5 s after a supervisory update following a common established run-up. Record realised time.
- Activate the candidate width throughout the operational run. If an event-time switch is used for isolation, label it as a separate instrument and retain a continuous-operation confirmation.
- A negative step that would make a nominal load capacitive must be checked and described physically; do not silently clip or replace it.

Metrics:
- Actual dead-zone activation delay, unsaturated/saturated incremental command, delivered incremental Q, and available directional headroom.
- Time-weighted |U-U_pre| through the next subordinate dispatch; use identical fixed measurement locations or a declared common aggregation rule.
- Unblanked voltage minimum/maximum and time outside the voltage corridor.
- If post-switching blanking is useful, report that metric separately; keep the raw extrema.
- Ratios relative to no droop only when the denominator is meaningful; always retain absolute values.
- Continue the three legs of two selected representative cases for at least three supervisory periods after the event, showing OFO handover. These six extended cells are part of the 54 if planned before execution.

Presentation:
- Summary by every operating window, including W5 as zero-capability limitation.
- Two representative detailed trajectory figures (selected by declared operating regime, not largest improvement); other traces in appendix.
- No universal 25% recovery condition is imposed for this demonstration. If a separate mandatory contingency requirement is adopted later, declare it explicitly and evaluate feasibility without retrospectively changing F1.

## 8. Execution gates and stopping rule

1. Resolve analytical and final manifests, then repair configuration/provenance interfaces.
2. Verify/rebuild B on the frozen configuration; run C.
3. Reconcile existing D at the resulting cadence; perform only necessary targeted confirmation or retuning.
4. Reuse matching E or run E with the final controller.
5. Complete F1 reference, width selection and live-width confirmation.
6. Verify one F2 event and fresh export; then execute the 54-cell bank.
7. Update dissertation tables and claims from complete manifests; keep old evidence explicitly historical.

Do not launch F1/F2 while weights, cadence, local-model construction or shunt settings are still changing.
No full repetition of A, automatic restart of the entire tuning search, or new universal contingency guarantee is required by this plan.
