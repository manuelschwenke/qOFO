# ladder/ — the seven-rung control ladder (superseded)

The pre-2026-10-06 design of Case 2: one deterministic 360-min run of the
seven rungs L1, L2, S1, O1, O2, O3, M, its Monte-Carlo extension, and the S1
PI-gain screens. The thesis replaced it on 2026-10-06 with the 2x2 variant
design (L2/S1/S2/O2/O3/M), and since 2026-10-07 S1 uses the integral laws
only. **Its numbers are not reported and must not be cited.**

Kept runnable, not deleted, because its machinery is the template of
`../mc/run_mc.py` and its palette is the one every Part III figure uses.

| File | Was (until 2026-10-23) |
|---|---|
| `variant_ladder.py` | `experiments/ch_10_case_study/ch_10_1_variant_ladder.py` |
| `figures.py` | `ch_10_2_figures.py` |
| `preview.py` | `ch_10_2_preview.py` |
| `export_tikz_data.py` | `ch_10_3_export_tikz_data.py` |
| `s1_pi_screen.py` | `ch_10_4_pi_screen.py` |
| `s1_pi_candidates.py` | `ch_10_5_pi_candidates.py` |
| `ladder_mc.py` | `ch_10_6_ladder_mc.py` |

Two behaviours changed with the move:

- Every producer writes a documented run folder under
  `results/ch_11_case2_multisystem/ladder/` (`experiments/thesis_runs.py`).
- `figures.py` and `export_tikz_data.py` write into the run's own `figures/`
  and `tikz_data/` by default. They used to write into the thesis's
  `graphics/Ch10`, which no longer belongs to this design. Pass `--out` to
  export.

Old results: `qOFO_GH/results/THESIS_ch10_variants_single_run`, `ch10_ladder*`,
`ch10_s1_*` (see `results/ch_11_case2_multisystem/README.md`).
