# Switched-shunt banks: one-step-per-zone and hard-voltage-row restrictions removed

Timestamp: 2026-10-22, Europe/Berlin.

Reason: author decision while revising the thesis section `ch:ofo:shunt`
(Ch 4). A bank should be an ordinary integer input of the TS-MIQP apart from
its level range and minimum holding time: its effect on the outputs enters
through its column of `H`, and the output constraints stay soft as for every
other actuator. The two extra restrictions introduced with the hybrid in-MIQP
bank dispatch (`b2da40f`) were dropped from the text and are now dropped from
the code, so text and code agree again.

## What the code did before

- **One step per zone.** `TSOController._solver_restrictions` passed the bank
  block as `MIQPProblem.one_step_indices`; the solver added
  `cp.norm1(w[banks]) <= 1`, so at most one bank of a zone stepped per solve.
- **Hard voltage rows.** The same method passed all voltage rows as
  `MIQPProblem.hard_output_indices` (no slack) and tightened them to the bank
  guard band `[shunt_v_min_pu, shunt_v_max_pu] = [0.90, 1.10]`. This applied to
  every solve of a zone with banks, whether or not a bank step was selected.
- **Fallback.** If that was infeasible, `BaseController._hold_restricted`
  re-solved with every bank pinned and the voltage rows soft again.
- The per-bank bounds from `ShuntBank.available_directions` already limited a
  single bank to ±1 level, so the "no two-level step of one bank" part of the
  one-step restriction was redundant.

## Changes

- `optimisation/miqp_solver.py`: removed `MIQPProblem.one_step_indices` and
  `hard_output_indices`, their validation in `__post_init__`, and the two
  constraint blocks in both solve paths. Unused `Tuple` import dropped.
- `controller/base_controller.py`: `step()` calls the solver once on the
  problem as built; `_solver_restrictions` and `_hold_restricted` removed
  (unused `replace` / `MIQPProblem` imports dropped).
- `controller/tso_controller.py`: `set_shunt_bank_state(lower, upper)` no
  longer takes a voltage guard; `_shunt_voltage_guard` and the
  `_solver_restrictions` override removed. The level bounds still come from
  `available_directions` (level limits + holding time), and a missing push
  still fails fast.
- `controller/shunt_banks.py`: `ShuntBankConfig.y_h_min/y_h_max` and
  `ShuntBank._feasible` (the predicted-voltage guard) removed; docstrings
  updated.
- `configs/config.py`: `shunt_v_min_pu` / `shunt_v_max_pu` and their check in
  `validate_integrator_mode` removed.
- `configs/paramsets/__init__.py`: `shunt_v_min_pu`, `shunt_v_max_pu` and the
  older `shunt_int_v_min_pu`, `shunt_int_v_max_pu` moved to `RETIRED_FIELDS`
  (the `shunt_int_v_*` rename entries dropped), so stored parameter sets that
  name them still load and the loader reports them as ignored.
- `experiments/runners/multi_tso_dso.py`: bank configs built without the band,
  `set_shunt_bank_state(lower_b, upper_b)`. Commit-loop comment corrected:
  several banks of a zone can now commit in one instant. The loop already
  handled a list of changed banks; every feedforward is evaluated on the
  pre-switch models, i.e. superposed to first order, which is how the MIQP
  priced the joint step.
- `experiments/CIGRE_2026/005_CIGRE_MULTI.py`, `006_CIGRE_MONTECARLO.py`:
  removed the `shunt_v_min_pu/shunt_v_max_pu` keyword arguments (both scripts
  run with `shunt_dispatch="off"`, so their results are unaffected).
- Tests: `test_controller.py` — `test_bank_restrictions_reach_the_solver`
  replaced by `test_bank_is_an_ordinary_bounded_integer` (one solve, bank
  bounds as pushed, voltage band = controller's own `v_min_pu/v_max_pu`);
  guard-band validation case removed. `test_shunt_banks.py` — band fixture,
  band validation and the two `_feasible` tests removed.
  `test_shunt_banks_integration.py` — the guard-from-a-real-column test
  removed.

## Verification

- `tests/test_shunt_banks.py tests/test_controller.py tests/test_miqp_solver.py
  tests/test_shunt_banks_integration.py`: 99 passed, 1 skipped.
- Full suite `pytest tests`: 858 passed, 15 skipped, 7 warnings (pre-existing
  deprecation / dispatch warnings), 40 min.
- Edited runner and CIGRE scripts compile; a probe parameter set naming
  `shunt_int_v_min_pu` and `shunt_v_max_pu` loads with both reported as
  retired.
- Not verified: closed-loop behaviour. No simulation was re-run.

## Consequences and open points

- **Results before this change used the restricted law.** Every run with
  `shunt_dispatch="miqp"` before this entry had hard, band-tightened voltage
  rows in every solve of a zone with banks, plus the one-step cap. They are
  not reproducible with the current code.
- **Hypothesis to check in simulation:** without the hard rows a 25–50 Mvar
  bank step can be selected although the linear prediction shows a voltage
  band violation, at the price of slack penalty; several banks of one zone
  (e.g. the MSC and the MSR of one DSO, on different couplers) can step in the
  same iteration, and their linearisation errors add up. The D1 shunt case
  should be re-run to see whether either occurs.
- Thesis: Ch 4 `ch:ofo:shunt` already states the unrestricted formulation
  (`eq:shunt:bounds`); its `% CODE STATUS` comment was updated to this entry.

## Closed-loop check, restricted vs unrestricted law

Note: the "D1 shunt case" named in the session is not a runnable experiment
(thesis Ch 8 D1 is still [TBD]). The run used here is the existing AC check
`experiments/shunt_dispatch_check.py` (docstring updated to the new law).

Setup, identical for all runs: paramset `ch9_selected_pcc27_gqceil`, 6 h from
2016-01-05 08:00 (W1), `coordination_mode="none"`, T_TS = 180 s, T_STS = 20 s,
`g_w_tso_shunt = 10`, holding time 1800 s, MSC/MSR 2 × 25 Mvar per bank,
+150 Mvar load at bus 11 connected at min 10 and tripped at min 30. Both law
variants ran from local `git archive` copies of HEAD `fa94411`; the
unrestricted copy differs only by this entry's change. The restricted run
counted `_hold_restricted` calls. Stored in
`results/2026-10-22_unrestricted_banks/` (`restricted_head/`, `unrestricted/`,
`banks_off/`, `comparison.json`, the driver `run_check.py` and
`analyse_runs.py`).

| | off | restricted (HEAD) | unrestricted |
|---|---|---|---|
| bank commits | 0 | 3 | 8 |
| instants with a commit | 0 | 3 | 4 |
| instants with > 1 bank in a zone | 0 | 0 | 2 |
| releases | 0 | 0 | 0 |
| records outside 0.90–1.10 pu (TS / STS) | 0 / 0 | 0 / 0 | 0 / 0 |
| TS V range [pu] | 0.9790–1.0502 | 0.9790–1.0502 | 0.9790–1.0507 |
| STS V max [pu] | 1.0561 | 1.0615 | 1.0615 |
| mean TS V RMS error [pu] | 0.004778 | 0.004697 | 0.004580 |
| interface-Q error RMSE [Mvar] | 0.581 | 0.572 | 0.640 |
| interface-Q abs. error p95 [Mvar] | 0.517 | 0.492 | 0.556 |
| aggregate DER utilisation p95 | 0.282 | 0.289 | 0.277 |
| fallback solves (pinned banks) | – | 0 | – |

Findings:

- **Reproducibility.** The restricted run at HEAD reproduces
  `results/2026-10-22_post_repair/miqp` exactly (same three commits, same
  metrics), and `off` reproduces `post_repair/off`; the rename commits after
  that run were behaviour-neutral.
- **The hard voltage rows never bound in this scenario.** The guard band
  equalled the controller band (0.90/1.10), so the tightening was a no-op, the
  plant stayed within 0.979–1.062 pu, and the fallback was never called. All
  differences come from removing the one-step cap.
- **Simultaneous steps occur.** At 720 s (load connected) zone 2 engages the
  MSCs of DSO_1, DSO_2 and DSO_3 together (75 Mvar) instead of DSO_1 alone; at
  1800 s (load tripped) it engages the three MSRs of the same DSOs instead of
  DSO_2's alone. A further MSC level at DSO_4 (zone 3) follows at 1980 s.
  Zone-2 voltage change across the commit instant: +0.0009 / −0.0016 pu
  (restricted +0.0006 / −0.0011); largest STS change −0.012 pu (restricted
  −0.005).
- **Opposing pairs remain engaged.** At 1800 s the MSCs engaged at 720 s are
  still inside their holding time (until 2520 s), so the MIQP reduces injection
  by engaging the MSRs of the same STSs. Afterwards every one of DSO_1–3 carries
  one engaged MSC and one engaged MSR on different couplers for the remaining
  5 h; net bank Q in zone 2 is zero, as in the restricted run (DSO_1 MSC vs
  DSO_2 MSR), but with six instead of two switched devices. No release
  occurs later because releasing either member of a pair alone is opposed by
  the gradient and its movement weight. (Mechanism inferred from the commit
  times and the holding-time bounds; not instrumented.)
- **Trade-off.** Against the restricted law: TS voltage RMS error −2.5 %,
  interface-Q RMSE +11.8 % and p95 +13 %, DER utilisation p95 −4 %. Against
  banks off, interface-Q RMSE is +10 %.

Open points:

- The standing MSC/MSR pairs are the real issue, and the one-step cap only made
  them smaller. A remedy is a design decision: e.g. a cost on the standing
  level, mutual exclusion of opposing banks in one STS, or allowing a release
  when the opposite device would otherwise be engaged.
- One scenario and one window; the voltage band was far from binding, so the
  run says nothing about the case the hard rows were meant for.
