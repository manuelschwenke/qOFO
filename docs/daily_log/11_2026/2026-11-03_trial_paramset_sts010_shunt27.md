# 2026-11-03 — Trial weight set sts010 + shunt27 in `run_multi_system_ofo.main()`

**Reason.** Author request: make the diagnostic combination "sts010 + shunt27" runnable from
`experiments/run_multi_system_ofo.py`, to try it by hand. It is a diagnostic, not a selection;
SELECTED_PARAMSET is unchanged.

## Changes (15:2x, shared checkout, branch minimal/thesis-core, not committed)

- **New parameter set `configs/paramsets/trial_1b94_sts010_shunt27.json`.**
  - Base: `ch9_selected_uc_2026_11_v2`, the procedure-v2 winner 1b945ad40b42 (run 0004).
  - g_w_dso_oltc 32.518 → 10.839: θ_STS,min 0.03 → 0.01 pu, outside the frozen bounds
    [0.02, 0.04].
  - g_w_tso_shunt 88.197 → 26.982: the Stage-I value of `stage1_v2_uc_2026_11_pq10`.
  - The other eight weights are the winner's.
- **`experiments/run_multi_system_ofo.py`.**
  - New module constants `TRIAL_PARAMSET` (default `"trial_1b94_sts010_shunt27"`; `None`
    runs SELECTED_PARAMSET) and `TRIAL_FROZEN_NETWORK` (default `True`).
  - `main()` applies `TRIAL_PARAMSET or SELECTED_PARAMSET`. With `TRIAL_FROZEN_NETWORK` it
    sets the doubled-DSO_3 multipliers (`_ch9_selected_design.FROZEN_DSO_*`), the network
    the Chapter-9 weights are designed and evaluated on; `make_config()` is symmetric.
  - The `TEST` line in `main()` that set `dso_gamma_oltc_q=1.0` (couplers also track
    interface Q) now sets 0.0 (author, 2026-11-03: "Yes, set it to 0"), as in the
    diagnostics; a comment records the former value. `make_config()` itself already has
    0.0.
  - `make_config()` is unchanged.
- Checked: the config `main()` builds carries the ten weights above, DSO_3 ×2,
  `shunt_ff_ledger` True, DSO band 0.92–1.08 pu, v_set 1.03 pu; the file parses.

## 15:3x — the tuning winner as the default trial (author: "set the tuning winner up")

- Copied `configs/paramsets/ch9_selected_uc_2026_11_v2.json` from the worktree F:/qOFO_res
  (byte-identical; untracked here like `stage1_v2_uc_2026_11_pq10.json`).
- `TRIAL_PARAMSET` now defaults to `"ch9_selected_uc_2026_11_v2"`; the comment lists
  `"trial_1b94_sts010_shunt27"` as the alternative. SELECTED_PARAMSET is unchanged.
- Checked: `main()` builds the winner's ten weights, DSO_3 ×2, `dso_gamma_oltc_q` 0.0,
  `shunt_ff_ledger` True.

## What is known about this combination (worktree F:/qOFO_res, run-0004 code)

- Diagnostics on W1–W4 and W6 (`results/diagnostics/2026-11-03_winner_variants_0004`):
  - sts010 alone passes the frozen filter: e_v,DS 0.00632, max DS voltage 1.0673 pu, STS
    taps active in every window, no hunting.
  - shunt27 alone and shunt27 + sts020 fail on one unexplained TS-tap reversal (W6,
    transformer 1[0], 87 min).
  - sts010 + shunt27 itself has not been simulated.
- Settling tests of sts020 and sts010 were running on e5power when this was written.

## Differences from those diagnostics when run from here

- **`shunt_ff_ledger = True`** (default on this branch since ac69918). Run 0004 and the
  diagnostics ran without the ledger, so bank releases behave differently.
- `main()` runs `make_config()`'s period and events, not the 12-h windows W1–W6.
