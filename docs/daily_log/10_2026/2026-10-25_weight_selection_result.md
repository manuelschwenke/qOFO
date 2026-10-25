# Weight selection procedure: result

Timestamp: 2026-10-25, ~14:15 Europe/Berlin.

Reason: completion of the single application of
`docs/tuning/2026-10-24_weight_selection_procedure.md` (driver
`results/THESIS_ch8_weight_selection_20261024/driver.py`, 24.1 h).

## What changed

| File | Change |
|---|---|
| `configs/paramsets/ch9_selected_final_2026_10.json` | New, written by the driver: c08b1d3d33ab with provenance. Not yet adopted as `SELECTED_PARAMSET`. |
| `docs/tuning/2026-10-24_weight_selection_procedure.md` | Execution record appended; the procedure text is unchanged. |

No code changed.

## Result

- Reproduction check passed exactly; Stage II-a reused.
- Stage II-b: four rounds, each found a new best, so the stage ended at the
  4-round cap with f = 0.5 throughout.
- 242 candidates evaluated, 84 feasible. Ranking winner 301c285ee52b failed
  the settling requirement; the recomputed ranking selects **c08b1d3d33ab**,
  which passes (p95 22.05, median residual 0.25 Mvar).
- Against the Stage-I point: e_v(TS) 0.00449 vs 0.00547 (−18 %), e_v(DS)
  0.00997 vs 0.00749 (+33 %), e_q 0.13 ×. λ_max(B) 1.414; taps ≤ 4/day,
  bank commits ≤ 4/day per bank, no hunting in the ranked windows.
- Notable weights: g_w_gen 9.44·10⁵ (2.0 spans below the Stage-I value: the
  local rounds moved the AVR step price far outside the II-a box), and
  g_w_dso_oltc 4.63 (−1.1 spans).

## Open, for the author

- Adopt `ch9_selected_final_2026_10` as `SELECTED_PARAMSET` and the
  `run_multi_system_ofo.py` default.
- The ranking puts e_v(TS) first: the selected set trades a 33 % higher
  e_v(DS) than the Stage-I point for the 18 % lower e_v(TS). That is the
  stated rule, reported here because it is large.
- Stage II-b stopped at its round cap while still improving; the procedure
  does not claim a converged optimum.
