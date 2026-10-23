# Experiment B with the banks in the TS-MIQP: result of the overnight run

Timestamp: 2026-10-23, ~19:00 Europe/Berlin.

Reason: completion of the campaign launched 2026-10-22 22:30 under
`docs/tuning/2026-10-22_protocol_experiment_B_banks.md` (driver
`results/THESIS_ch8_expB_banks_20261022/driver.py`). Full record in the
protocol, §11.

## What changed

| File | Change |
|---|---|
| `configs/paramsets/ch9_selected_10_2026_banks.json` | New, written by the driver: winner faf088d2570e with provenance (protocol, campaign folder, Experiment C result, aggregates). **Not** adopted as `SELECTED_PARAMSET`. |
| `docs/tuning/2026-10-22_protocol_experiment_B_banks.md` | §11 execution record filled in. |

No code changed.

## Result

- Stage II: 720 / 720 units, 16.85 h; 41 / 120 candidates feasible; the bank
  budget (4–48 commits per bank and day) changes neither the feasible count
  above 8 nor the top five.
- Selected **faf088d2570e**: e_v(TS) 0.00507 (−7.4 % vs. the starting
  weights), e_v(DS) 0.00844 (+12.6 %), e_q 0.11 ×, λ_max(B) 1.487;
  Experiment C pass (p95 21.05, median residual 0.30 Mvar). Ranks 2–4,
  including the starting weights, fail Experiment C.
- `g_w_tso_shunt` 46.32 (+0.92 of its span; Δv_eng,sh ≈ 0.017 pu): the
  winner never commits a bank in the ranked windows (11 commits in W5).
  `g_w_gen` also at −0.92 of its span.

## Open, for the author

- The declared orientation run — the October weights with the Stage-I bank
  price — is feasible and dominates the winner (e_v(TS) −0.6 %, inside the
  2 % band; e_v(DS) −17 %). It lies outside the search box (g_w_der −1.17 of
  the span). Accept the protocol result, or amend the box (an amendment made
  after seeing this, to be recorded as such).
- Confirmation of the decisions taken under delegation (Δv_eng,sh 0.010, bank
  budget 12, n 512, W5 handling, result name), then whether to adopt the
  result as `SELECTED_PARAMSET`.
