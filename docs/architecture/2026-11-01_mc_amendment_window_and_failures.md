# 2026-11-01: MC amendment: comparison window (rule b) and failure criteria

Amends §5 of `2026-10-28_case2_step_mc_plan.md` and the window rule of 2026-10-30
(`run_mc.comparison_window`, top-up of infeasible episodes). Declared before any MC run on the
unit-commitment base or on the weights of the Ch 9 rerun.

## Decision (author, 2026-11-01)

1. **Comparison window, rule (b).**
   - The paired comparisons run over the first N = 100 valid episodes in draw order on which
     **all six variants complete** (`ok`).
   - An episode on which **any** variant fails is replaced by the next valid episode, once all
     six have run on it.
   - Failures and their causes are reported **separately**.
2. **A voltage outside [0.90, 1.10] pu at the horizon end is not a failure.** It is kept as an
   observation (flag `v_end_outside_hard_band` in the run's `diag`; table in `failures.md`).

The remaining failure criteria are unchanged:
- `fail_abort`: exception or short log, including a power flow that does not converge and stops
  the run;
- `fail_unreachable`: reachability guard;
- `fail_pf`: steps without a converged power flow, run continued;
- `fail_osc`: ≥ 2 tap reversals in the last 30 min.

## Added with the implementation (Claude, stated to the author before the decision)

- **Stop condition `N_VALID_MAX = 150`.** The window never uses an episode beyond the first
  150 valid ones. If fewer than 100 complete episodes are found by then, the driver stops and
  the decision returns to the author.
- The author did not object, but did not confirm the value either. It can be changed before
  the first run (`--max-valid`).

## What the rule means statistically

- **The comparison metrics are conditional** on all six variants completing. They are biased
  towards episodes every variant can handle; the hardest episodes leave the comparison.
- **The failure counts stay unconditional.** Every variant runs on every valid episode up to the
  last one used (window plus replaced). So the counts and Clopper–Pearson bounds per variant
  are taken over all of these.
  - As before, an episode on which all six fail (infeasible) is kept out of the bounds.
  - It is listed in the failure report.
- **The failure report** (`failures.md`, `failures_per_run.csv`) gives, for every failed run:
  - the cause;
  - the stop time and the time since the last event;
  - whether the run stopped in an event's own step (`at_event`: the post-event state has no
    power-flow solution before any controller acts) or on the TS dispatch grid;
  - the replaced episodes variant by variant.
- **O3's further episodes** (300 for its failure bound) are listed in the failure report, not
  counted in the window statistics, as before.

## Implementation

- `run_mc.py`:
  - `WINDOW_RULES` / `WINDOW_RULE = "2026-11-01"`, `N_VALID_MAX`;
  - `comparison_window(valid, outcome, n, rule)`;
  - `effective_outcome` (a `fail_v_end` of older code becomes `fail_osc` or `ok` from its tap
    reversals);
  - `failure_row`, `failure_report`;
  - `report(out, n, rule, max_valid)` restricts the distributions and paired tests to the window;
  - CLI `--rule`, `--max-valid`; `--no-top-up` is the same as `--rule 2026-10-07`;
  - the run JSON stores `dt_s` and `t_ts_s`;
  - the campaign parameters record `window_rule`, `n_valid_max` and `v_end_is_failure: false`.
- `evaluate_mc.py`: same rule and outcome definition (`--rule`, `--max-valid`);
  `summary.json["window"]` has `rule` and `replaced`.
- The earlier rules stay available for re-reporting earlier campaigns.

## Check on an earlier campaign (after the decision)

The old-base campaign `qOFO_weekend/.../mc/0001_2026-10-30_184316_run_mc` was re-reported from
a copy of its JSON files.
- **Rule 2026-10-30:** 4 episodes replaced, all of them infeasible.
- **Rule 2026-11-01:** 7 episodes replaced. The 4 infeasible ones are joined by 3 tap-hunting
  episodes (L2 twice, S1 once).
- No run of that campaign ended outside the hard band.
- Of the 4 infeasible episodes:
  - 2 stop in the event's own step for every variant;
  - 1 stops 4–6 min after a line trip;
  - 1 stops partly at the event and partly 0.3–1 min after it.
