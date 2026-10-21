# Shared-objective gate for slow shunt support, regular path parked

Timestamp: 2026-10-20, 17:35 CEST.

Reason: the verification at `g_w_tso_shunt = 20` showed that the regular MIQP
bank path improves the TS voltage term by 0.6% while degrading interface-Q RMSE
by 54%, STS voltage RMS by 7.6% and DER utilisation p95 by 21%. No movement
weight repairs that exchange. The author decided to park the regular path and
to re-price the slow support gate against the shared objective instead of the
per-unit voltage allowance, which had refused ten of ten candidates at roughly
5.5 times its bound.

Changes:

- `controller/shunt_hybrid.py`: `HybridShuntSolver` takes `support_gate`
  (`'voltage_allowance'`, the previous behaviour and still the default, or
  `'shared_objective'`) and `support_exchange`. Under the new gate a confirmed
  candidate is admitted iff

      grad_y f @ (y_candidate − y_regular) ≤ support_exchange · (support cost removed),

  that is, the first-order change of the shared objective between the predicted
  operating points of the two solves of the same augmented problem. The hard
  predicted voltage corridor and the no-increase-in-output-violation check apply
  under both gates.
- Corrected the measure after the first verification run. The initial version
  compared MIQP objective values and was wrong by about ten orders of
  magnitude: the quadratic `G_w` terms regularise the OFO step rather than
  express `f`, and a candidate is forced to move its delegated request by a
  whole bank step, so the comparison was dominated by that step-size price. The
  run recorded increases of `5.5·10¹⁰` against support reductions of order 0.5,
  and refused all five candidates for the wrong reason. Subtracting the bank's
  own movement weight, as the first version did, removed only one of the two
  step-size terms. The output-space form avoids both by construction and is
  invariant to the parked movement weight, which a test pins.
- `grad_y f` covers voltages, interface Q, line currents and generator Q, and
  by construction not the DER-reserve term, which acts on a control variable.
  The gate therefore does not price a candidate's effect on that term.
- Each candidate now records `shared_objective_increase` and
  `support_cost_reduction`, and the new refusal reason is `shared_objective`.
- `configs/config.py`: added `shunt_hybrid_support_gate` and
  `shunt_hybrid_support_exchange` with fail-fast validation; defaults preserve
  the recorded October behaviour. Wired through `experiments/runners/multi_tso_dso.py`.
- `experiments/shunt_hybrid_check.py`: `--support-gate` and `--support-exchange`
  added, and `--movement-weight` now defaults to 3600, the configuration
  default, which parks the regular path. Admission requires a per-level
  gradient above that, whereas the recorded distribution tops out at 34.4, so
  the regular path must now be un-parked explicitly.

Method: the gate compares two solves of the same augmented problem — the
regular solution and the candidate with its bank level fixed and the delegated
STS requests reduced by the predicted bank effect — so the comparison includes
the cost of the reference re-allocation and of the ordinary actuator motion
needed to accommodate it. The exchange weight converts support-cost units into
shared-objective units; it replaces a pu bound that had no relation to the
weights the controller is tuned on.

Verification: 203 tests pass with the one pre-existing skip, in 61.00 s; the
focused hybrid file passes 32. Three new tests cover admission when the benefit
outweighs the cost, refusal with the reason `shared_objective` when it does
not, and invariance of the comparison to the parked movement weight. Two new
configuration cases cover an unknown gate and a negative exchange.

## Verification, 18:20 CEST

One hour with the parked regular path and the new gate at the default exchange
of 1.0, stored in `results/shunt_hybrid_shared_gate_v2_20261020`.

- One candidate, at 2880 s on zone 3 bank 6, admitted and committed as
  `sts_tracking`. `shared_objective_increase = −1.3450`,
  `support_cost_reduction = 0.3910`.
- The refused candidate was never a trade-off: its first-order effect on the
  shared objective is negative, so the step improves the objective the
  controller minimises and would be admitted by any non-negative exchange,
  including zero. The 0.001 pu bound had refused, five times, an action the
  objective wants — it inspected the worst observed bus and voltage alone,
  while `grad_y f` weighs every observed output with its tuned weight.
- Interface-Q RMSE 4.36584 → 4.22566 Mvar, absolute-error p95
  11.72540 → 10.75032 Mvar, aggregate DER utilisation p95 0.39897 → 0.40347,
  hourly TS and STS voltage ranges unchanged to all printed digits. The same
  figures as the earlier 0.02 pu diagnostic, which admitted the same step, but
  now for a reason expressed in the controller's own weights.

## Zero exchange as the default, 18:45 CEST

`shunt_hybrid_support_exchange` was not exercised by the verification hour: one
candidate with a negative cost carries no information about its value. The
author therefore set its default to **0.0**, so support may act only when it
does not worsen the shared objective to first order. This needs no calibration,
admits the one recorded commit unchanged, and keeps an underivable number off
the critical path. The constructor, configuration and experiment-script
defaults agree, and a test covers admission at zero of a step that improves the
objective while exceeding the pu bound the previous gate applied.

A positive exchange remains available as a deliberate willingness to pay TS
objective for subordinate support. It must be derived from candidates whose
recorded `shared_objective_increase` is positive, and none has been observed in
any run so far.

Verification: 204 tests pass with the one pre-existing skip, in 76.38 s; the
focused hybrid file passes 33. One hour at the new default, stored in
`results/shunt_hybrid_exchange0_20261020`, reproduces the exchange-1.0 run to
every recorded digit: one `sts_tracking` commit at 2880 s on zone 3 bank 6,
interface-Q RMSE 4.22566158 Mvar, absolute-error p95 10.75032271 Mvar, 43
candidate solves. Its stored configuration confirms the default travelled
through the runner — `shunt_hybrid_support_exchange = 0.0`,
`shunt_hybrid_support_gate = 'shared_objective'`, `g_w_tso_shunt = 3600.0` —
and the admitted candidate records `shared_objective_increase = −1.3450`
against a worst-bus voltage error increase of 0.005399 pu, that is 5.4 times
the now inactive `shunt_hybrid_voltage_error_allowance_pu`.

Open: the `sts_reserve` memory has still never become non-zero, and no bank
release has been observed in any run. Both the parking decision and the gate
verification rest on one diagnostic hour with one disturbance pair.
