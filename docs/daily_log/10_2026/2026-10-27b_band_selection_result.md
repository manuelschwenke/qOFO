# Weight selection with the DSO band: result fde2e81af6b1; Experiment C reference; settling CDF

Timestamp: 2026-10-27, ~12:50 Europe/Berlin.

**Reason:** completion of the procedure re-application with the DSO operating
band. Procedure: `docs/tuning/2026-10-25_weight_selection_procedure.md`,
whose execution record is now filled in. Campaign folder:
`results/THESIS_ch8_weight_selection_band_20261025/`.

**Code:** no repository code changed. New file:
`configs/paramsets/ch9_selected_band_2026_10.json`, written by the driver.
`SELECTED_PARAMSET` is **not** switched yet; it is still
`ch9_selected_final_2026_10`, pending the author's decision after the STS
margin check.

## Result

- **Selected:** fde2e81af6b1, the Stage II-b round-4 best.
- **Candidate pool:** 242 evaluated, 94 feasible (84 in the no-band
  application).
- **Settling check (rank order):**
  - fde2e81af6b1 PASS: p95 25.0, 4 / 120 steps > 27, max 33, median
    residual 0.35 Mvar.
  - c08b1d3d33ab PASS: 22.05.
  - e27ce5e4f01e PASS: 22.05.
  - 301c285ee52b aborted; see `2026-10-27_settling_abort_301c_and_margin_check_queued.md`.
- **Driver total:** 38.1 h.

| Over W1–W4, W6 | fde2e81af6b1 | Stage-I point | c08b1d3d33ab (no-band result) |
|---|---|---|---|
| e_v(TS) | 0.00440 | 0.00547 | 0.00449 |
| e_v(DS) | 0.01305 | 0.00749 | 0.00991 (with band) |
| e_q | 299 762 | 1 039 596 | — |
| λ_max(B) | 1.467 | 1.000 | 1.414 |
| taps / transformer / day (TS, STS) | 4, 2 | 4, 4 | — |
| bank commits / bank / day | 4 | 4 | — |
| AVR max step [pu] | 0.0035 | 0.00003 | — |

**Weights:**

| Weight | fde2e81af6b1 | Stage-I point | Ratio |
|---|---|---|---|
| g_q | 0.1738 | 0.04 | ×4.3 |
| dso_g_v | 659.9 | 10⁴ | ×0.066 |
| g_w_der | 0.03493 | — | — |
| g_w_pcc | 0.06768 | — | — |
| g_w_gen | 7.28·10⁵ | — | — |
| g_w_tso_oltc | 25.59 | — | — |
| g_w_tso_shunt | 23.81 | — | — |
| g_w_dso_der | 0.708 | — | — |
| g_w_dso_oltc | 5.834 | — | — |

**Interpretation:** the STS voltage priority is low, so the band, not
dso_g_v, keeps the STS voltages in check. This is consistent with
fde2e81af6b1 failing the hard corridor in W1 without the band.

**W5 (report only, in neither filter nor ranking):**
- e_v(TS) 0.00779 vs 0.01097 for the Stage-I point; e_v(DS) 0.01172 vs
  0.00973.
- The AVR max step of 0.0109 pu and 2 bank hunting events would violate the
  filter limits if W5 were filtered.
- The Stage-I point has 12 tap hunting events in W5.

## Experiment C reference at the Stage-I weights

- **Run:** `expC/stage1_reference/20261027-120335`, 12:03–12:42.
- **Result:** p95 31.1 (per DSO 31.0 / 29.55 / 33.0 / 30.1), 36 / 120 steps
  > 27, max 34, median residual 2.26 Mvar.
- **Verdict:** under the selection criterion it would FAIL. The starting
  weights do not meet the settling requirement; the selected weights do.

## Settling CDF regenerated (thesis clone, uncommitted)

- **What ran:** `followup_settling_cdf.py` ran the unchanged generator
  `make_settling_cdf.py` on the reference and on fde2e81af6b1.
- **Files rewritten:** `graphics/Ch9/data/2026-09-30_settling/settling_cdf.dat`
  and `graphics/Ch9/settling_cdf.tex` in `Z:\latex_diss_clone`. Nothing is
  committed there.
- **What changed in the figure:**
  - F(27) is 0.700 (start) and 0.967 (final), previously 0.700 and 0.975.
  - F ≥ 0.95 is first reached at n = 31 (start) and 25 (final), previously
    32 and 23.
  - The in-figure label now reads 96.7 %.
- **Manual follow-up:** the README in that folder still names the September
  sources. Chapter text that quotes 97.5 %, 23 or 32 must be updated by hand.

## Still running

`followup_margin.py` started at 12:43. It runs fde2e81af6b1 and the Stage-I
point in W1–W4, W6 with full records. Output:
`margin_check/margin_report.txt`, then `MARGIN_DONE.txt`, ~13:30.
