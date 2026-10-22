# Bank step price becomes a searched weight; Stage I and the selection see the banks

Timestamp: 2026-10-22, ~20:15 Europe/Berlin.

Reason: Phase 0 of the Ch 8 Experiment B rerun. Since the switched-shunt banks
were folded into the TS-MIQP (`2e06a1b`, unrestricted and with STS mutual
exclusion in `ac8b52f`), their step price `g_w_tso_shunt` is an integer step
price like the tap prices. Author decision 2026-10-22, option (a): Stage I
places it from a stated engagement deviation by the tap rule, and Stage II
searches it at ±0.5 decade like the taps. Before this change the campaign code
could not run on the current controller at all: `tuning/stage1_screen.py`
refused `shunt_dispatch='miqp'`.

## What changed

| File | Change |
|---|---|
| `configs/config.py` | `g_w_tso_shunt: Optional[float] = None` (was `10.0`) and added to `REQUIRED_WEIGHTS`, so `require_weights()` raises without it; the error names the reason when it is the missing one. Stale `shunt_dispatch` docstring (one bank per zone, hard voltage rows) corrected to the law of `ac8b52f`. |
| `tuning/parameters.py` | `g_w_tso_shunt` moved into `WEIGHT_FIELDS` (ten weights, nine searched). `GAUGE_COUPLED_FIELDS` removed — its only member is now a weight and scales with the gauge by construction. Docstrings updated. |
| `tuning/stage1_screen.py` | Guard `_check_shunt_screen_support` removed. `build_operating_point` raises if `shunt_dispatch='miqp'` and no TS layer carries a bank column. `_network_key` includes the bank layout (dispatch, MSC/MSR coupler positions, step sizes) under `'miqp'`, so captures made without bank columns are never reloaded. `design_stage1(d_engage_shunt=...)` prices class `shunt` with the existing `design_integer_weights` (`g_w = 2 g_V |Σ_j H_V[j,b]| Δv_eng`, maximum over bank columns, areas, conditions) and raises if banks are captured but no deviation is given — no default. CLI: `--d-engage-shunt` replaces `--g-w-tso-shunt` (hand-set, default 10); `--write-paramset` writes the designed price and refuses a design that does not price every weight. |
| `tuning/stage2_search.py` | `DEFAULT_SPANS["g_w_tso_shunt"] = 0.5`. `shunt_summary(records, events_s)` adds `shunt_ops_max_bank` (busiest single bank) and event-attributed `n_shunt_hunting` / `n_shunt_event_explained` via `hunting_reversals` (tap definition); the raw unattributed `shunt_reversals` stays as a diagnostic. `aggregate` carries `has_shunt_wear`, `worst_shunt_ops_per_day_bank`, `n_shunt_hunting`, `n_shunt_event_explained`, `shunt_commits`. `Limits.shunt_ops_per_day` (no default) and `Limits.max_shunt_hunting = 0`; `check_limits` judges both as counts, like the taps. The 2026-09-15 "shunts deliberately not bounded" scope comment replaced by its history and the reason it no longer holds. |
| `experiments/ch_9_parameter_selection/ch_9_3_stage2_screen.py` | Window row: `shunt_summary` with the window's event times, and `shunt_ops_per_h_bank`. |
| `experiments/ch_9_parameter_selection/ch_9_3_select.py` | `--shunt-ops-per-day` (no default; refuses to judge a bank with shunt wear records without it) and `--max-shunt-hunting` (0). Ranked table shows bank commits/day and bank hunting; the report records the new limits. |
| `tuning/shunt_weight_calibration.py` | Docstring: the one-bank-per-zone statement removed. |
| Tests | `tests/tuning/test_stage1_banks.py` (new, 5): bank price follows the rule; design refuses without `d_engage_shunt`; designed price round-trips to the stated deviation at the strongest bank; no bank price without bank columns; cache key separates captures with/without banks. `tests/tuning/test_stage2_shunt_wear.py` (new, 7): per-bank count, unexplained vs event-explained reversal, aggregate per-day rate, budget judged as a count, bank hunting rejects, old records pass. `test_parameters.py`: nine searched, bank price searched and required, every searched field has a span, missing bank price names the reason. Fixtures in `tests/tuning/conftest.py`, `test_parameters.py`, `test_dso_v_relief_pairing.py` carry `g_w_tso_shunt`. |

## Consequences

- **Every parameter set written before today lacks `g_w_tso_shunt` and no
  longer runs under `shunt_dispatch='miqp'`** — including `SELECTED_PARAMSET`
  = `ch9_selected_10_2026`, which therefore makes every Ch 9 entry point that
  uses the default raise until the rerun writes a new selection. Until today
  all of them silently ran at the config default 10. This is intended (fail
  fast, author decision N7); the files are not rewritten.
- Fingerprints (`weight_fingerprint`) of new candidates cover ten weights and
  are not comparable with earlier keys.
- Stage-II campaigns before today stay re-selectable: their records carry no
  shunt wear, `has_shunt_wear` is 0 and the new limits pass them unchanged.
  Re-writing a parameter set from them with `--write-paramset` fails, since
  their weights lack `g_w_tso_shunt`.
- The W5 question (N5) is untouched: the selection filter still uses the
  ranking aggregate, which excludes W5, for bank wear as for tap wear.

## Verification

- New and changed fast tests: 42 passed, 2 skipped.
- Full suite `pytest tests`: 867 passed, 15 skipped, 7 warnings (pre-existing
  deprecation / dispatch warnings), 61 min.
- Stage-I capture on the real network: six operating points rebuilt on the
  frozen network (DSO_3 ×2) with banks in the MIQP, cache key `net41a2af08`
  in `results/stage1/operating_points/`; seven layers each, TS zone 2 carries
  six bank columns (MSC and MSR of DSO_1–3), zone 3 two (DSO_4). The
  bank-column check passed for every condition.

## Δv_eng,sh table (decision input, nothing frozen)

`results/2026-10-22_ch8_shunt_engage_table/` (`engage_table.py`,
`box_shift.py`, `one_column_per_sts.py`, logs and JSON). Systematic reading,
g_V = 4·10⁴, taps at D7 (0.0075 / 0.015).

- One bank level moves its zone's mean TS voltage by 0.0020–0.0028 pu
  (|Σ H_V| 0.022–0.034 pu per level; strongest MSR@DSO_2, W5). At the
  `g_w_tso_shunt = 10` used since 2026-10-21 the banks engage at a zone-wide
  offset of 0.0037–0.0057 pu, i.e. earlier than the TS taps (0.0075); at 20,
  0.0073–0.0113 pu.
- **The bank columns raise the TS integer floor.** With every bank column in
  λ_max(B): Δv_eng,sh = 0.005 is infeasible (TS_2 floor 1.36 > target 1.0);
  0.0075 → `g_w_tso_shunt` 20.4, floor 0.94, κ_TS 2.92 (TS DER/PCC prices
  +1.74 decades over the October start); 0.01 → 27.2, 0.72, κ 0.199 (+0.58);
  0.015 → 40.8, 0.50 (tap floor), κ 0.103 (+0.29). Only `g_w_der` and
  `g_w_pcc` move; every other Stage-I weight is unchanged.
- **The October selection does not pass the screen once the banks are
  counted.** `ch9_selected_10_2026` at the bank price 10 it has silently run
  with since the fold: worst λ_max(B) 2.99 (TS_2, W5), above the OFO bound 2;
  it reaches the selection ceiling 1.546 only from about 40.
- **The full-column screen counts moves the controller cannot make.** With the
  MSC/MSR exclusion only one bank per STS can move in a given direction.
  Keeping the stronger column per STS: 0.0075 → 20.4, floor 0.59, κ 0.126;
  0.01 → 27.2, floor 0.50 (taps), κ 0.093; October selection 2.18 at 10, 1.62
  at 20, 1.47 at 27.2.
- Open, author decision before freezing: (i) whether the screen counts all
  bank columns or one per STS; (ii) Δv_eng,sh. Indicative cross-check only
  (other weights, older law): the 2026-10-20 recorded gradient distribution
  had its event mode at 25–34, which is where Δv_eng,sh = 0.01–0.0125 places
  the price.
