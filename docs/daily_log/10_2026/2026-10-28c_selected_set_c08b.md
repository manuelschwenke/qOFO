# Selected set after procedure 2026-10-28: c08b1d3d33ab (301c failed settling)

Timestamp: 2026-10-28, ~11:50–15:00 Europe/Berlin.

## Author decisions

- **Thesis framing.** The thesis presents
  `docs/tuning/2026-10-28_weight_selection_procedure.md` (the 2026-10-25
  procedure plus the STS coupler engagement bound) as the procedure,
  applied once in run 0008. Run 0001 is a development campaign whose
  Stage-II units are reused (identical decision path, bit-exact
  simulations). The 2026-10-25 text is marked superseded.
- **Choice of 301c285ee52b.** The author chose 301c285ee52b (rank 1) and
  asked for it as the selected set for a test run before its settling check
  had finished. Per the procedure, c08b1d3d33ab follows if 301c fails.

## Changes

| time | commit | change |
|---|---|---|
| ~11:55 | 5975ab6 | `docs/tuning/2026-10-28_weight_selection_procedure.md` (consolidated text); 10-25 marked superseded |
| ~12:00 | 750b5a7 | provisional `configs/paramsets/ch9_candidate_301c285ee52b.json` (301c weights, settling pending) and `SELECTED_PARAMSET` pointed to it, for the author's test run |
| 14:43 | — | Experiment C of 301c: **FAIL** (details below) |
| 14:43 | — | the producer crashed at the result write |
| 14:44 | c012ea5 | crash fixed and `--resume` added |
| 14:45 | — | run 0008 resumed (no simulation repeated); it wrote `configs/paramsets/ch9_selected_band_2026_10_a1.json` = c08b1d3d33ab; run 0008 status complete |
| after | this commit | `SELECTED_PARAMSET = "ch9_selected_band_2026_10_a1"` (c08b1d3d33ab); the provisional 301c set is withdrawn but kept on disk, since test runs may name it |

**301c settling result.** p95 31.0 > 27 (per DSO 30.55 / 27.55 / 32.0 /
25.55), 29/120 steps > 27, max 46, median residual 0.78 Mvar. The fail is
clear, not marginal.

**The producer crash.** `os.path.relpath(Path(__file__), REPO_ROOT)` failed
between the Z: drive and the UNC path. `_rel` now resolves both sides; with
`--resume`, the settling stage continues from the run's own
`criterion_results.json`.

## c08b1d3d33ab against the old set fde2e81af6b1 (W1–W4, W6)

| quantity | c08b | fde2 |
|---|---|---|
| e_v(TS) | 0.00449 (+2.2 %) | 0.00440 |
| e_v(DS) | 0.00991 (−24 %) | 0.01305 |
| e_q | 138 244 (−54 %) | 299 762 |
| λ_max(B) | 1.414 | 1.467 |
| θ_STS,min | 0.0191 pu | 0.0797 pu |
| STS operations | 8 | 1 |
| min with STS median dev ≥ 0.02 pu (of 240 area-hours) | 16 | 782 |

c08b also has one 14-min episode (DSO_3, W4) without a tap operation.

## Stale

All of the following were made with fde2:
- D1/D2 runs 0006/0007 and the Ch 9 text quoting them;
- the outputs of the `ch_9_1_*` producers;
- the settling CDF: c08b's Experiment C exists (run 0001 expC), so the CDF
  can be regenerated without simulation;
- `tab:param:weights:resolved`;
- the Case-2 result runs.

The author's test runs with the provisional 301c set are not results.
