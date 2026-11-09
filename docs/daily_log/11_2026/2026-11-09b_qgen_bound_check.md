# 2026-11-09b — Do machines reach their Q capability bound in v5's windows?

Session "STS capabilities tuning analysis". Follow-up to `2026-11-09_h_sn_mva_fix.md`. Author: "Yes,
check it" (whether fix A matters for procedure v5).

## Question
Fix A (Q_gen rows per V_set / tap ×100) changes only the TSO soft constraint on machine Q
(`rho_q_gen`), which acts when a machine is predicted outside the capability band
[Q_min, Q_max] that the controller draws from the PQ curve each iteration. Does any machine come
near that band, or near the plant's Q limit, in the v5 windows?

## Method (`2026-11-09b_qgen_bound_check.py`, output `.out`)
- **Code:** as-built v5 code, from a detached worktree `F:\qOFO_v5chk` at 583c24c. Same code as
  run 0002 and e5power; the campaign checkout was not used.
- **Candidate and windows:** 158fdfc8e812 (II-b round-1 best at 08:00), paramset
  `stage1_v5_nr_2026_11_pq10`, `sts_coupler_hold_after_bank=1`, frozen network. Evented 12-h
  windows W1–W6, set up exactly as `ch_9_3_stage2_screen._run_one_window` does.
- **Logged** via `STEP_HOOK`, once per TS iteration (240 per window). The controllers are not
  changed. For each machine:
  - measured Q;
  - the controller's band (`_get_output_limits`, Q_gen rows);
  - the plant limits (`net.gen.min/max_q_mvar`).
- **Runtime:** 6 processes, 57–144 min per window. The CPU cap (10, later 12, cores) applied from
  08:30.

## Result (established, for this candidate and these windows)
- **No sample outside the controller band, none within 10 Mvar of it, none at a plant limit.**
  This holds for all 6 windows and all in-service machines (gens 0, 1 = G3, 2, 5, 7, 9 = G1).
- **Smallest margin to the band:** 121 Mvar (gen 5, zone 3, W5). Machine Q stays between −122 and
  +428 Mvar; the +428 is G1 in W6, whose plant limit is ±1200.
- **Consequence:** fix A would not have changed a single MIQP step in these windows. v5's results
  are unaffected by the Q_gen slip, as far as the 12-h windows go.

## Side observation (model–plant mismatch, open)
On the over-excited side the controller's band is wider than the plant's table limit:

| machine | window | controller band (median upper) | plant limit |
|---|---|---|---|
| G3 | W1 | 596 Mvar | 400 Mvar |
| gen 7 | W3 | 695 Mvar | 500 Mvar |
| G1 | all | ≈ 9550 Mvar | 1200 Mvar |

At a machine's over-excitation limit the plant saturates before the soft constraint acts. This
does not matter at the operating points above.

## Not covered
- G3-off states: G3 is in service in all six 12-h windows. Those states occur in the 3-h MC
  episodes (G3-off report after GO), where the remaining machines carry zone 2's Q.
- Other candidates. The machines' Q is set mainly by the operating point; the weights move it by
  tens of Mvar, against margins above 120 Mvar.
