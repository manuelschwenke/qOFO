# Joint OFO shunt dispatch with separate slow memories

Date: 20 October 2026. Status: experimental, explicitly enabled.

This implements the requested design: a regular shunt decision in the TS MIQP,
plus separate per-bank memories for persistent STS tracking error and reserve
restoration near capability limits. The existing selected parameter sets and
study defaults are unchanged. The trial gains below are not tuned replacements.

## Implemented behaviour

The TS MIQP jointly chooses ordinary actuator changes and integer bank steps.
The additional bank columns cover the observed voltages, interface Q, line
currents and synchronous-generator Q. They use cached controller models and
measured-voltage rescaling. The controller does not inspect plant truth.

`g_w_tso_shunt` prices a squared physical bank-level change in this mode. The
regular bank gradient is the shared output-objective gradient projected through
its complete output column. Strong present demand can therefore cause a switch
without waiting for slow confirmation. With other effects held fixed, a
one-step move reduces the local objective when its signed gradient benefit
exceeds the movement cost. Constraints and simultaneous actuator changes affect
the actual joint decision.

`controller/shunt_hybrid.py` wraps the existing solver. It appends the bank
variables to the optimisation problem, then returns the original controller
control slice and the selected bank's additional output prediction. This keeps
the existing actuator interface intact while including the bank in all joint
output constraints and in the public prediction.

Each bank also has two independent signed accumulators:

- Tracking pressure follows persistent interface-Q error outside the configured
  deadband. A whole bank step must reduce the tracking cost.
- Reserve pressure follows measured interface utilisation outside the central
  part of the reported capability interval. It is zero inside that central
  band. A reserve commit also requires tracking within tolerance and a reduced
  delegated request within the reported interval.

The costs and their gradients are defined in `controller/shunt_support.py`.
The reserve memory is observed independently of the tracking memory. An
appearing tracking error does not erase existing reserve pressure, but it
prevents a reserve commit while the tracking error remains outside tolerance.

For each cost separately, the nameplate-Mvar accumulator updates once per TS
observation as

    a_next = clip(a - gradient / (2 shunt_int_g_w), -q_step, q_step).

A proposal needs both accumulated magnitude at least `q_step/2 + delta` and
the configured minimum persistence. Sign reversal, disappearing need, stale
feedback or an excessive observation gap clears that memory. A physical commit
clears all slow memories in the zone because it changes their shared operating
point. Persistence concerns sampled observations, not an unobserved continuous
signal.

The regular MIQP has first choice. If it holds all banks, the supervisor tests
eligible slow proposals by fixing one bank step and reoptimising the ordinary
controls. Tracking proposals have priority over reserve proposals. Within one
reason, larger relative support-cost reduction has priority, followed by the
joint objective and bank order.

Every path shares the same physical level, dwell clock and switching budget.
At most one bank step per zone and TS instant is allowed. A selected bank step
must satisfy the hard predicted TS voltage corridor. Slow proposals must also
avoid increasing any predicted output-limit violation, and may increase each
observed bus's voltage tracking error by at most
`shunt_hybrid_voltage_error_allowance_pu`, default 0.001 pu, relative to the
regular solution. These are local linear checks. They do not guarantee the
nonlinear multi-zone plant response.

If the hard voltage corridor makes the augmented problem infeasible, the
original MIQP runs with the banks held. This preserves the existing recovery
action for ordinary actuators.

## STS reference allocation

For a slow proposal, let `delta_q` be the predicted bank effect across all
interfaces of the affected STS. The reoptimised problem fixes the delegated
requests to

    delegated_candidate = delegated_regular_solution - delta_q.
    offset_new = offset_old + delta_q.

The total reference sent to the STS is therefore the same as in the regular
solution. Ordinary TS controls can move to accommodate this allocation. The
runner applies the bank once and updates the offset once. It does not repeat
the delegated-request correction after the solve.

A regular objective-driven bank move instead follows the existing feedforward
convention. Its total STS reference may change along with the jointly selected
delegated request and bank contribution.

## Assumptions and scope

The implemented path requires cascaded OFO controllers, all STS controllers
active, `coordination_mode="none"` and input-side PCC capability bounds.
Incompatible configurations fail early. This work tests the quasi-static AC
plant. It does not validate RMS co-simulation or horizontal/vertical coordination.

The TS controls DER Q, delegated STS interface-Q requests, AVR voltage
references, OLTCs and the existing tertiary MSC/MSR banks. The normal TS
objective is unchanged. The STS controls its DER and OLTCs to track interface Q
while respecting its voltage and current limits.

The reserve signal uses the reported interface capability bands. These are
DER-based envelopes, not the complete feasible set of the STS network and not
individual DER headroom measurements. Successful reserve restoration assumes
that the STS can subsequently track its reduced delegated contribution. The
implementation does not establish that assumption for every operating point.

The existing daily budget uses reset windows rather than a rolling 24-hour
counter. That behaviour is retained.

## Relation to the CIGRE paper

The supplied *A Dynamic Adjustment Method of Voltage Reference Value and Dead
Band Width for Advanced Voltage Control in Power Systems*, CIGRE CSE 026,
November 2022, describes integration of voltage deviation outside a deadband
in Section 2.6. Figure 6(a), PDF page 8, resets the integral state after a switch.
This supports accumulation and reset as switching mechanisms. It does not
validate the proposed STS reserve cost or the joint MIQP architecture.

Source supplied by the author:
`C:/Users/Manuel Schwenke/Downloads/CSE_26-a-dynamic-adjustment-method-of-voltage-reference-value-and-dead-band-width-for-advanced-voltage-control-in-power-systems.pdf`.

## Tests

The focused suite passed **198 tests**, with one pre-existing skip, in 65.94 s.
The skipped controller test already lacks a no-slack infeasibility fixture.
The deliberate unreachable-corridor test produces one expected Gurobi
infeasible-or-unbounded warning and verifies the ordinary-MIQP fallback.

The tests cover regular engagement thresholds, independent memories, minimum
confirmation, disappearing and stale need, observation gaps, current-command
checks, shared dwell and budget, release steps, competing banks, joint voltage
compensation, output prediction and reference transfer with `alpha=1` and 0.2.
Explicit support-candidate tests pass with both Gurobi and SCIP.

Reduced closed-loop tests demonstrate:

- An unreachable 40 Mvar reference with an STS contribution limited to 20 Mvar
  is met after one 25 Mvar bank step. The STS contribution settles at 15 Mvar.
- A tracked 45 Mvar reference with a contribution range of ±50 Mvar triggers
  one near-limit reserve step. The contribution settles at 20 Mvar and further
  switching stops.

Four additional AC tests exercise MSC/MSR engagement and release through the
joint solver. A constructed persistent request is used while the subordinate
contribution is held. All four steps reduce actual interface error and preserve
the commanded total reference. These isolate physical direction and allocation.
They do not represent a spontaneous tracking failure of the complete STS MIQP.

Reproduction with the project's `qOFO_clean` Python:

```text
python -B -X utf8 -m pytest -q -p no:cacheprovider
  tests/test_shunt_hybrid.py tests/test_shunt_hybrid_ac.py
  tests/test_miqp_solver.py tests/test_controller.py
  tests/test_shunt_support.py tests/test_shunt_support_ac.py
  tests/test_shunt_integrator.py tests/test_shunt_integrator_integration.py
  tests/test_shunt_voltage_rescaling.py tests/test_tso_tertiary_shunt.py
  tests/test_tso_output_gradient.py
```

## One-hour AC comparisons

The diagnostic scenario uses `ch9_selected_pcc27_gqceil_comp_q_off`, the symmetric
benchmark, real profiles, minimum measurement noise with seed 20260717 and
TS/STS periods of 180/20 s. A 150 Mvar TS load connects at minute 10 and is
removed at minute 30. Bank steps are 25 Mvar with two levels and 1800 s dwell.
Horizontal coordination is disabled.

The comparison uses a regular bank movement weight of 5 per squared level,
slow memory weight 0.01, persistence 360 s, tracking weight 10, reserve weight
100, a 5 Mvar tracking deadband, 25 Mvar normalisation, reserve threshold 0.8
and response wait 40 s. The regular-only variant sets both slow cost weights
to zero. The slow-only diagnostic sets the regular bank movement weight to
1e6. All values are declared diagnostic choices.

### Recorded results

| Variant | Bank commits | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] | Aggregate DER utilisation p95 | TS mean spatial V RMS [pu] | STS mean spatial V RMS [pu] |
|---|---:|---:|---:|---:|---:|---:|
| `legacy` | 0 | 4.36584 | 11.72540 | 0.39897 | 0.0059942 | 0.0067532 |
| `regular` | 10, all objective | 8.06344 | 19.27176 | 0.49759 | 0.0058640 | 0.0076895 |
| `hybrid` | 10, all objective | 8.06344 | 19.27176 | 0.49759 | 0.0058640 | 0.0076895 |
| `slow` | 0 | 4.36584 | 11.72540 | 0.39897 | 0.0059942 | 0.0067532 |

Every variant records 180 TS records, 63 optimal TS solves and 720 optimal STS
solves, with no recovery, infeasibility or hybrid fallback entry. TS voltages
remain within 0.97794–1.05080 pu and STS voltages within 0.98334–1.05543 pu.
The joint formulation raises the TS solve time p95 from 0.0020 s to
0.0373–0.0476 s, with a recorded maximum of 0.0578 s. That remains three orders
of magnitude below the 180 s TS period, so the added integer columns do not
threaten the Layer-1 timescale in this problem size.

Four observations follow from these runs.

**The trial movement weight is far too permissive.** At `g_w_tso_shunt = 5` per
squared level the regular joint decision commits ten bank steps within one hour,
with 1800 s dwell and six banks available. Interface-Q RMSE rises from 4.36584
to 8.06344 Mvar and the absolute-error p95 from 11.72540 to 19.27176 Mvar. The
regular path therefore reaches its switching threshold at a demand level that
the continuous actuators were still tracking better on their own.

The recorded commit sequence shows that this is a ratchet rather than a
regulating action. All ten commits have `direction = +1`; no bank is released
within the hour. They occur at 20, 180, 360, 540, 720 and 900 s — one per TS
instant, consecutively, as fast as the one-step-per-zone constraint permits —
then two at 1800 s and two at 1980 s. The first commit precedes the load step by
eight minutes, so the sequence is not disturbance-driven. In zone 2 the solver
engaged every available bank, MSC and MSR alike (`shunt_idx` 1, 0, 5, 4, 2, 3),
whose net interface effects largely cancel. With `g_v = 40000` dominating the
shared objective, a bank step is priced as a local voltage action with a large
interface-Q side effect, and at a movement cost of 5 every such step is
profitable. Whether the simultaneous MSC/MSR engagement is the sibling cycle
already recorded in `docs/daily_log/10_2026/2026-10-14_shunt_gw_and_msc_msr_sibling_cycle.md`
is not yet established.

**The bank steps did not relieve subordinate reserve here.** Aggregate DER
utilisation p95 rises from 0.39897 to 0.49759. In this scenario the committed
steps increased DER loading rather than restoring headroom, which is the
opposite of the intended reserve effect and is consistent with the degraded
interface tracking.

**The reported voltage benefit is small and not uniform across layers.** TS mean
spatial voltage RMS improves from 0.0059942 to 0.0058640 pu while STS mean
spatial RMS worsens from 0.0067532 to 0.0076895 pu. The shared objective prices
only the TS part of this exchange.

**The slow memories decided nothing in either direction.** With the regular path
active, `hybrid` reproduces `regular` to every recorded digit, including the same
ten objective commits: the tracking memory was non-zero at nine samples and
ready at six, but the instantaneous decision had already moved the banks. With
the regular path priced out at `g_w_tso_shunt = 1e6`, the `slow` variant
reproduces `legacy` instead, having proposed five candidates and had all five
refused at the final check. Those five are consecutive TS instants from
2880 s to 3600 s, all in zone 3 on bank `shunt_idx = 6` (MSC), with the
tracking memory saturated at its 25 Mvar clip and a steady tracking gradient
near −0.22. Each instant records two candidate solves, so the proposal passed
the delegated-bound test and was refused only after the joint reoptimisation.

The four runs above predate the per-candidate rejection diagnostics, so they
carry the refusal count but not its reason. The two instrumented re-runs below
supply it.

### Refusal reason and the voltage allowance

Re-running `slow` with the instrumented solver reproduces the earlier result
exactly and attributes all five refusals to the voltage-tracking test. The
recorded worst-bus predicted tracking-error increase is 0.00540, 0.00549,
0.00544, 0.00577 and 0.00570 pu, that is 5.4 to 5.8 times the 0.001 pu
allowance, with `max_output_violation_increase = 0.0` in every case. The
proposals were therefore feasible and were refused solely because a full 25 Mvar
bank step moves the worst observed bus by about the size of the TS mean spatial
tracking error itself.

Repeating the case with a deliberately wide 0.02 pu allowance admits the first
proposal and nothing further:

| `slow` variant | Allowance [pu] | Candidates | Commits | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] | DER utilisation p95 |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.001 | 5, all refused | 0 | 4.36584 | 11.72540 | 0.39897 |
| relaxed | 0.020 | 1, accepted | 1 `sts_tracking` | 4.22566 | 10.75032 | 0.40347 |

The single commit occurs at 2880 s on zone 3 bank `shunt_idx = 6`, engaging the
MSC. Interface-Q RMSE improves by 3.2% and the absolute-error p95 by 8.3%.
Aggregate DER utilisation p95 rises marginally, by 1.1%. The recorded TS
voltage range, 0.97887–1.05006 pu, and the STS range, 0.98435–1.05462 pu, are
identical to the refusing run to all printed digits. After the commit the
demand is satisfied and no further candidate appears, so persistence and reset
behave as designed.

Two conclusions follow. First, the slow path is functional and in this scenario
beneficial for its own controlled output; it was gated off rather than idle, and
the acceptance threshold lies between 0.001 and 0.0054 pu for a 25 Mvar bank.
Second, the guard's currency is questionable: it refuses on an instantaneous
linearised worst-bus prediction, while the realised hourly voltage range did not
change. Pricing the candidate against the shared objective, where `g_v` and
`g_q_tso` already express this exchange, would replace an arbitrary pu
allowance with the weights the controller is tuned on.

Taken together, the runs establish that the regular path at this weight is a
ratchet that both degrades tracking and pre-empts the slow path, that the
movement weight must be derived from the objective scale rather than chosen,
and that the slow path's single commit improved its controlled output. They do
not establish a net benefit of hybrid dispatch. They are one scenario of one
hour with one commit, and they are not the campaign's ranking metrics. The
`sts_reserve` memory never became non-zero in any run, while aggregate DER
utilisation p95 reached 0.498 in the regular variant, so the reported-capability
proxy with `r0 = 0.8` is insensitive here and a reported DER-utilisation signal
should be considered instead.

The instrumented runs are stored under
`results/shunt_hybrid_diag_20261020/allow_default` and `.../allow_wide`.

Run the experiment with:

```text
python -B -X utf8 experiments/shunt_hybrid_check.py
  --cases legacy regular hybrid slow --seconds 3600
  --output results/hybrid_shunt_check
```

Apply the hybrid overrides after applying a parameter set, since the existing
parameter workflow fixes the legacy dispatch structure. The experiment script
does this explicitly. Each case stores its full configuration, source hashes,
log, records and summary. Existing result directories are not overwritten.

Local evidence is in
`C:/Users/Manuel Schwenke/Desktop/Daten/01_Forschung/12_Dissertation/latex_diss_ms/tmp/shunt_hybrid_20261020/results/`.
The additional aggregate diagnostics are in `comparison_analysis.json` in the
parent folder. Source changes during these runs were restricted to
configuration descriptions/error wording, tests and the Stage-I rejection guard.
They do not alter the simulated control law.

## Revision: shared-objective support gate, regular path parked

Decided on 20 October after the weight verification. Two changes follow from it.

**The regular MIQP bank path is parked.** `g_w_tso_shunt` keeps its
configuration default of 3600, and `experiments/shunt_hybrid_check.py` now
defaults `--movement-weight` to the same value. Admission requires a per-level
gradient above the weight, and the recorded distribution tops out at 34.4, so
the regular path is inactive unless it is un-parked explicitly. The reason is
recorded in `docs/tuning/2026-10-20_hybrid_shunt_weight_calibration.md`: the
path improves only the heavily weighted TS voltage term and pays for it in
interface-Q tracking, STS voltage quality and STS DER utilisation, and no
movement weight alters that exchange. Re-activating it requires pricing the
bank's interface-Q and subordinate-layer effects in the objective that selects
it, which is a change to the shared TS objective and remains an open design
question, not a tuning step.

**The slow support gate can be priced in the shared objective.**
`shunt_hybrid_support_gate` selects between the original
`'voltage_allowance'`, still the default so the recorded October runs remain
reproducible, and `'shared_objective'`. Under the latter a confirmed candidate
is admitted iff

    ∇_y f ᵀ (y_candidate − y_regular) ≤ shunt_hybrid_support_exchange · ΔJ_support,

with `ΔJ_support = cost_before − cost_after` of the tracking or reserve cost and
`y` the predicted outputs of the two solves of the same augmented problem. The
candidate fixes its bank level and pins the delegated STS requests to
`regular − delta_q`, so its predicted outputs already contain the bank effect
and the ordinary actuator motion that accommodates the reference re-allocation.

The measure is the **first-order change of the shared objective between the two
predicted operating points**, not a difference of MIQP objective values. The
first attempt used the objective values and was wrong by about ten orders of
magnitude: the quadratic `G_w` terms regularise the OFO step rather than
express `f`, and a candidate is forced to move its delegated request by a whole
bank step, so the comparison was dominated by that step-size price —
`5.5·10¹⁰` against support reductions of order 0.5 in the one-hour check. The
bank's own movement weight entered the same way. The output-space form avoids
both by construction; a test pins its invariance to the movement weight.

`∇_y f` carries the output part of the objective — voltages, interface Q, line
currents and generator Q — and by construction not the DER-reserve term, which
acts on a control variable. The slow support costs concern exactly those
outputs, but the gate therefore does not price a candidate's effect on the DER
reserve term.

The hard predicted voltage corridor and the refusal of any increased predicted
output-limit violation apply under both gates. What the new gate removes is the
per-unit bound on the worst bus's predicted voltage tracking error, which had no
relation to the weights the controller is tuned on and refused ten of ten
candidates at roughly 5.5 times its value.

`shunt_hybrid_support_exchange` is **not calibrated** and defaults to 1.0. Each
candidate now records `shared_objective_increase` and `support_cost_reduction`,
so the weight can be derived from a run by the same admission-threshold
reasoning used for `g_w_tso_shunt`, with `tuning/shunt_weight_calibration.py` as
the model for the procedure.

The focused suite passes 203 tests with the one pre-existing skip. The new
cases cover admission when the benefit outweighs the cost, refusal with reason
`shared_objective` when it does not, invariance to the step-size prices, and
fail-fast validation of an unknown gate and a negative exchange.

### Verification of the gate

One hour with the regular path parked and the new gate at the default exchange
of 1.0, otherwise the same scenario, in
`results/shunt_hybrid_shared_gate_v2_20261020`:

| Gate | Candidates | Commits | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] | DER utilisation p95 |
|---|---:|---:|---:|---:|---:|
| none (`legacy`) | – | 0 | 4.36584 | 11.72540 | 0.39897 |
| `voltage_allowance`, 0.001 pu | 5 | 0 | 4.36584 | 11.72540 | 0.39897 |
| `shared_objective`, exchange 1.0 | 1 | 1 `sts_tracking` | 4.22566 | 10.75032 | 0.40347 |

A single candidate appears, at 2880 s on zone 3 bank 6, and is admitted:

    ΔJ_shared = −1.3450,    ΔJ_support = 0.3910.

**The refused candidate was never a trade-off.** Its first-order effect on the
shared objective is *negative*, so the step improves the objective the
controller is minimising and would be admitted by any non-negative exchange,
including zero. The 0.001 pu bound had been refusing, five times in a row, an
action that the controller's own objective wants: it inspected the worst
observed bus and voltage alone, whereas `∇_y f` weighs every observed output
with its tuned weight, and the 0.0054 pu worst-bus increase is outweighed by
the interface-Q improvement and by the other buses.

Interface-Q RMSE improves by 3.2% and the absolute-error p95 by 8.3%, the
hourly TS and STS voltage ranges are unchanged to all printed digits, and
aggregate DER utilisation p95 rises by 1.1%. These are the same figures as the
earlier 0.02 pu diagnostic, which admitted the same single step; the difference
is that the gate now admits it for a reason expressed in the controller's own
weights and without an arbitrary bound.

Consequently `shunt_hybrid_support_exchange` is **not exercised** by this hour.
One candidate with a negative cost carries no information about its value, so
the weight was uncalibrated and, so far, unneeded.

**Its default was therefore set to 0.0.** Support may act only when it does not
worsen the shared objective to first order. This needs no calibration, it
admits the commit above unchanged, and it keeps an underivable number off the
critical path. A positive value remains available and is a deliberate
willingness to pay TS objective for subordinate support; it must be derived
from candidates whose recorded `shared_objective_increase` is positive, of which
none has yet been observed. A test covers admission at the zero default of a
step that improves the objective while exceeding the pu bound the previous gate
would have applied.

## Consequences for the tuning campaign

The completed October selection remains a result for the original controller.
Adding an inactive experimental option requires no rerun. Adopting hybrid
dispatch for the thesis requires a revised tuning protocol and new selection.

This differs from the earlier separate-supervisor prototype: hybrid dispatch
adds integer columns and their movement weight to the TS optimisation problem.
The previous statement that the MIQP is unchanged therefore does not apply.

| Component | Consequence of adopting hybrid dispatch |
|---|---|
| Physical operating points and cached sensitivities | Potentially reusable after an identity check. |
| Stage-I curvature capture and analytical screen | Extend to include the added bank columns and `g_w_tso_shunt`. Recompute the augmented screen and candidate eligibility. |
| Regular shunt calibration | Choose and calibrate a per-level MIQP movement weight. The old integrator-gain rule does not provide it. |
| Slow support calibration | Calibrate tracking/reserve cost ratios, memory gain, persistence, deadbands and voltage allowance. |
| Stage-II closed-loop evaluation | Repeat the baseline and all candidates admitted by the revised protocol, with full controller provenance. |
| Isolated STS settling | Reuse only for unchanged STS settings and identical held-bank initialisation. Repeat for newly selected settings. |
| Subsequent studies | Repeat affected runs with active hybrid banks or changed selected weights. |

The present Stage-I capture reads only the base controller matrix and would
omit the appended bank columns. Its cache identity also does not distinguish
this change. Both fresh capture and cached loading now explicitly reject hybrid
mode until those extensions exist. The old screen is a result for the old
model, not a certificate for this controller.

The existing tuning parameter registry deliberately excludes
`g_w_tso_shunt` because shunts previously sat outside the MIQP. Before a
production campaign it must carry the hybrid movement weight, mode, support
parameters and complete provenance. Common scaling of the regular objective
must include the regular bank weight. Slow cost weights and the memory weight
must retain their intended ratios.

The 16 October protocol, with Amendments 1–3, evaluated 118 analytical
survivors plus a baseline on six 12-hour windows. That count cannot simply be
assumed for a changed analytical screen. Keep the original Sobol samples if
the search coordinates remain suitable, but recompute eligibility and then
evaluate all admitted candidates under the declared selection rule. Use a
new result directory, since existing workers can reuse completed result files.
No full tuning campaign was launched in this task.
