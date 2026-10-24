# Experiment B with banks: Amendment 1-alt selects the October weights at the Stage-I bank price

Timestamp: 2026-10-24, ~11:15 Europe/Berlin.

Reason: run 1 of `docs/tuning/2026-10-22_protocol_experiment_B_banks.md`
selected faf088d2570e, but the declared orientation run — the October
selection `ch9_selected_10_2026` at the Stage-I bank price — dominated it on
both voltage criteria and lay outside the search box. The author asked for a
widening amendment with its cost, then chose the union alternative after the
analysis showed the widening would not sample the October region. Decided
after seeing run 1, recorded as such in the protocol (§10).

## What changed

| File | Change |
|---|---|
| `docs/tuning/2026-10-22_protocol_experiment_B_banks.md` | §10: Amendment 1 (widening) drafted with cost and coverage finding, not adopted; Amendment 1-alt (union with cffa2ca108c5) adopted and frozen before its run. §11: execution record. |
| `configs/paramsets/ch9_selected_10_2026_banks_a1.json` | New: result of Amendment 1-alt, provenance names the amendment. Not adopted as `SELECTED_PARAMSET`. |
| `results/THESIS_ch8_expB_banks_20261022/amendment1_options.py`, `amendment1alt.py` | Analysis of the widening options (analytic, no simulation) and the amendment run, reusing the driver's Experiment C, criterion and repeated-selection functions. Untracked (results/ is ignored). |

No code changed.

## Findings

- **Widening does not sample the October region.** With `g_w_der` / `g_w_pcc`
  spans 1.25 or 1.5 the October point is inside the box, but the nearest
  screen survivor is 0.66 / 0.58 decades away and none lies within
  0.5 decade; doubling n changes nothing, since scipy's scrambled Sobol design
  of 1024 points begins with the 512-point design (checked). Cost would have
  been ~14.5–15.8 h (n 512) or ~27–30 h (n 1024) plus ~3 h.
- **Amendment 1-alt result:** cffa2ca108c5 passes Experiment C (p95 23.05,
  median residual 0.40 Mvar) and ranks first over the union (e_v(TS) 0.00504,
  faf088d2570e 0.00507 inside the 2 % band; e_v(DS) 0.00697 vs. 0.00844). The
  selected weights are the October selection's nine weights plus the Stage-I
  bank price 27.21.

## Open, for the author

- Adopt `ch9_selected_10_2026_banks_a1` as `SELECTED_PARAMSET` (this unblocks
  every Ch 9/10 entry point that uses the default), and confirm the decisions
  taken under delegation (Δv_eng,sh 0.010, bank budget 12, n 512, W5
  handling, result names).
- Thesis Ch 8: the selection now rests on an amendment decided after the
  outcome; the text has to say so, and that the search of run 1 did not
  sample the region of the selected point.
