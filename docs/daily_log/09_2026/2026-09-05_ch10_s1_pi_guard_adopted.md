# 2026-09-05 - ch10 S1 adopted as PI + guard; screen and candidates run

## What was run

1. `ch_10_4_pi_screen --minutes 90` -- 4 runs (k_p in {0, 0.2} x guard on/off),
   270 records each, no failures. `results/ch10_s1_pi_screen/screen.json`.
2. `ch_10_5_pi_candidates --minutes 360` (new file) -- both PI candidates,
   1080 records each, no failures. `results/ch10_s1_candidates/`.

## What was changed

* **NEW** `experiments/ch_10_case_study/ch_10_5_pi_candidates.py` -- runs the
  360-min S1 PI candidates. Copies `ch_10_4_pi_screen.main()`'s config pattern
  (`make_thesis_config` -> `VARIANTS["S1"]` -> swept fields) so no weight is
  restated. `--only <tag>` runs one corner; `candidates.json` MERGES so the
  sweep can be split across invocations without losing earlier entries.
* **EDIT** `ch_10_1_variant_ladder.py`, `VARIANTS["S1"]`: added
  `svr_k_p_rvr=0.2`, `svr_k_p_rpr=0.2`. `assert_ladder_controlled()` still
  passes (both fields were already in `STEP_ALLOWED` for `L2->S1` and
  `S1->O1`). Gains are INHERITED from the 42-min screen, not tuned.
* **DATA** `results/THESIS_ch10_variants_single_run/g_w_gen_1e9/S1/log.pkl`
  replaced with the PI+guard run. See `S1_PROVENANCE_2026-09-05.md` in that
  folder. Old runs archived under `_archive_S1_classical/`.

## Why

The stored S1 was the classical I-only guard-off run while the code said
guard-on -- found by noticing `peak_gen_trip = 0.031424` matched the screen's
`kp0_noguard` corner, then confirmed by md5 against the archive. Guard-off S1
loses to plain local control on both transient peaks, which made the `L2->S1`
rung read as hierarchical coordination *degrading* transients. With PI+guard,
S1 beats L1/L2 on peaks and the guard cuts the 210-min line-trip peak 34 %.

## Findings worth keeping

* Determinism confirmed twice: `peak_gen_trip` matched to 18 digits between the
  90-min screen and the 360-min candidate of the same config, and across
  parallel vs solo execution.
* Parallel contention on this server measured at **1.07x** for two concurrent
  sims, not the assumed ~3x. Apparent slowdowns were OLTC tap-rate-limiter
  re-runs (a second power flow per step), which rise with horizon and are
  6-10x more frequent with the guard OFF.

## Open

* `g_w_gen_1e8/S1` still classical -- see the provenance note.
* The 360-min **I-only + guard** corner does not exist, so the P term and the
  guard cannot yet be attributed separately at full horizon.
* `README.md` of the results folder is stale w.r.t. S1.
