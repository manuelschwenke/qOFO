# 2026-11-01 (evening): Ch 9 rerun, settling failures of the four leaders, diagnosis

Run `results/ch_09_parameter_selection/0001_2026-10-31_135000_ch_9_6_full_procedure`. No code was
changed. Analysis scripts are in the session scratchpad (`settle_cmp.py`, `c08b_probe/theta_c08b.py`).

## Status

- Stage II-b ended at 13:08 after the 4-round cap. Round 4 found a new best (068ee81ca7ef), so the
  local search stopped at the cap, not by convergence.
- Settling check, batch 1 (068e, dd14, 1ac0, 71ea): **all FAIL** at 16:16.
  - p95 of n_settle: 30–33 (limit 27).
  - Median residual |q_final − q_set|: 2.3–3.1 Mvar (limit 1 Mvar).
- Batch 2 (e45c, 378f, c63c, f74f) started at 16:17. Expected ≈19:30.

## Findings (from the saved settling outputs; no new simulation)

1. **The residual is not a capability limit.**
   - 0 of 120 steps per candidate are capability-limited; the median headroom is 48 Mvar
     against 10-Mvar steps.
   - No tap moves.
   - The residual is a steady offset after settling, larger on "down" steps (median
     3.6–5.0 Mvar) than on "up" steps (1.2–2.1 Mvar).
2. **The base alone does not explain it.**
   - In the old-base settling check, c08b/e27c/fde2 had their STS voltages *further* from
     1.03 (median v_max 1.046–1.053 vs 1.037 now), but residuals of only 0.24–0.35 Mvar.
3. **The STS weights differ by two orders of magnitude.**

   | Weight set | g_q | dso_g_v | dso_g_v/g_q | g_w_dso_der/g_q |
   |:--|--:|--:|--:|--:|
   | c08b (old base) | 0.635 | 2186 | 3.4e3 | 3.3 |
   | dd14 | 0.027 | 8393 | 3.1e5 | 9.0 |
   | 068e | 0.019 | 9731 | 5.2e5 | 13.8 |
   | Stage-I point (all bases) | 0.04 | 1e4 | 2.5e5 | 12.6 |
   | 85 feasible candidates, min / median | – | – | 9.7e4 / 2.8e5 | 5.4 / 11.1 |

   - Hypothesis (H-a): with a voltage weight that large relative to the Q-tracking weight,
     the STS optimum stops short of q_set wherever its voltages deviate from 1.03.
   - A larger step price relative to g_q slows convergence, which would explain the p95
     failures.
   - Across the four, residual ∝ (dso_g_v/g_q)^0.54. Extrapolated to c08b's ratio, this
     gives 0.21 Mvar; c08b measured 0.25 on the old base. This is consistent, but rests on
     4 points.
4. **The e_v,DS tie-break favours exactly these weights.**
   - Spearman(e_v,DS, dso_g_v/g_q) is −0.57 over the feasible set and −0.55 in the tie group.
   - The settling check walks the rank order, so it starts with the candidates with the
     largest residuals.
5. **The c08b-like region was sampled but filtered out.**
   - 43 evaluated candidates have dso_g_v/g_q < 5e4. Of these, 42 were removed by
     amendment 2 (θ > 0.02); 1 (29479acd2f5e, θ 0.018) was rejected for 3 hunting reversals.
   - No feasible candidate has a ratio below 9.7e4. The extrapolation gives ≈1.3 Mvar
     there, so it is likely that **no candidate of this run passes**.
6. **c08b's own θ on the commitment-base operating points is 0.0195.** It would pass
   amendment 2, by a margin of 0.0005.
   - Check of the method: dd14 is reproduced at 0.0127.
   - c08b is still infeasible on this base (run 0010: one tap reversal, W6).

## Diagnostic probe (author OK 2026-11-01, "Yes, do it")

- c08b weights through the same settling check on the commitment base:
  `ch_9_1_ninner_isolated_sts.py --paramset stage1_start_uc_2026_10`, 10 workers.
- Run on e5power at commit 3d98d31 (the commit of the candidates' checks), launched 18:42.
- Output: `F:\qOFO_runs\qOFO_res\results\diagnostics\2026-11-01_c08b_settle_uc`, kept apart
  from the run folder.
- Expected ≈21:15–21:30.
- Reading the result:
  - c08b passes → settling-capable STS weights exist on this base; the failures come from
    the weight region the search ranks first (H-a).
  - c08b fails → the base also degrades settling.

## Update 19:30: batch 2, driver stop, what-if on θ

- **Driver stopped at 19:12 (rc 1).**
  - After 378f finished on e5power (18:49), the pull of `expC\_remote` failed three times in
    `pull_dir`. The log truncates the stderr at 400 characters, so the cause is not visible.
    It may be a race with the second shard, which was still writing its log in the same
    folder.
  - All four batch-2 checks completed; their outputs are on disk.
  - `criterion_results.json` holds batch 1 only.
- **Batch 2, evaluated read-only with the driver's `_criterion`:** all FAIL.
  - e45c: p95 30, residual 1.96 Mvar; 378f: p95 28, 1.64; c63c: p95 33, 3.54;
    f74f: p95 33, 2.75.
- **Residual against the weights over all eight:** Spearman(residual, dso_g_v/g_q) = 0.93;
  Spearman(residual, g_q) = −0.92.
- **What-if: the frozen selection rule re-run offline on the 244 simulated candidates**
  (scratchpad `whatif/`), with different θ bounds:

  | θ rule | Feasible | Winner | Min dso_g_v/g_q among feasible |
  |:--|--:|:--|--:|
  | ≤ 0.02 (amendment 2) | 85 | 068e | 9.7e4 |
  | ≤ 0.03 | 116 | 068e | 3.1e4 |
  | ≤ 0.05 (amendment 1) | 129 | 068e | 2.7e4 |
  | [0.02, 0.03] | 31 | aa95103d3f53 (dso_g_v/g_q 3.5e5) | 3.1e4 |

  Relaxing θ does not change the outcome.
- **The high-g_q region is infeasible on this base.**
  - 21 of the 22 candidates with dso_g_v/g_q < 2e4 are rejected, almost all for tap
    hunting reversals (< 2 h dwell, unexplained by an event).
  - The only feasible one, 1ad7f229713e, has e_v,DS 0.0184 and θ 0.11.
  - c08b fails by 1 reversal (run 0010); the Stage-I point fails by 2.

## Update 19:45: hunting diagnosis (author: "Yes, do it"); what-ifs

Source: the persisted tap moves and `hunting_detail` of all 244 candidates. No simulation.

- **Where the hunting is.**
  - 1111 hunting events in total; 842 (76 %) are on TS machine transformers: zone 1 [1] 480,
    zone 3 [1] 235, zone 1 [0] 127. The rest are on STS couplers.
  - Most dwells are 10–20 min (559 events), which fits the ≈21-min cooldown limit cycles
    documented on 2026-09-11.
  - Only 34 events start at the first TS dispatch. It is **not** a start-up artefact.
- **What the hunting follows.** It does **not** follow g_q: Spearman(n_hunting, g_q) = −0.12;
  (n_hunting, dso_g_v/g_q) = −0.05. It follows the TS tap price g_w_tso_oltc (−0.67) and the
  DER step prices (g_w_der +0.54, g_w_dso_der +0.47).

  | g_w_tso_oltc | n | hunting-free |
  |:--|--:|--:|
  | < 15 | 38 | 0 % |
  | 15–20 | 16 | 6 % |
  | 20–25 | 16 | 62 % |
  | ≥ 25 | 174 | 88–95 % |

  - The high-g_q candidates hunted because 13 of the 22 were drawn with g_w_tso_oltc < 16.
  - The earlier reading "high g_q is infeasible" is **withdrawn**.
  - c08b has g_w_tso_oltc 16.7, consistent with its single reversal on this base.
- **What-if: up to 1 hunting reversal allowed (`--max-hunting 1`).** The winner is unchanged
  for θ ≤ 0.02, θ ≤ 0.03 and θ ∈ [0.02, 0.03]; only 2 more candidates become feasible.
- **Voltage criteria against the STS weight ratio** (hunting-free candidates):
  - e_v,TS is indifferent: Spearman(e_v,TS, dso_g_v/g_q) = −0.03 over all, +0.08 over the
    Sobol sample.
  - e_v,DS improves strongly with the ratio (−0.66 / −0.69).
  - So it is the **e_v,DS tie-break** that pulls the selection towards weak Q tracking.
- **Empirical settling relation on this base:** over the 8 measured candidates, residual ∝
  (dso_g_v/g_q)^0.81, so 1 Mvar is reached at a ratio of ≈1.1e5.
- **Candidates already simulated** that meet the frozen limits with θ ≤ 0.05 and a ratio
  ≤ 1.1e5: 9. The most relevant:
  - **d162c73ca1fc**: θ 0.0208, e_v,TS 0.00846 (+0.3 %), e_v,DS 0.00833, ratio 3.1e4,
    g_q 0.29, g_w_tso_oltc 36.9.
  - e63aed2d04cb: θ 0.0279, +1.2 %, ratio 1.09e5.
  - 1553167154de: θ 0.0142, +1.1 %, ratio 9.7e4.

## Diagnostic settling checks, launched 19:40

- d162 and e63a on e5wimi; 1553 on e5power. 10 workers each.
- Same test and same code as the candidates (no change in `experiments/ch_09…`, `tuning/`
  since 3d98d31).
- Output: `results\diagnostics\2026-11-01_settle_ratio_candidates` (local; on e5power under
  `F:\qOFO_runs\qOFO_res\…`).
- Expected ≈22:15–22:45.

## Amendment 3 implemented (author 20:00: "Do the proposed revision")

- **Note:** `docs/tuning/2026-11-01_amendment3_theta_band.md`.
  - Engagement band 0.02 ≤ θ_STS,min ≤ 0.03 pu, declared after run 0001's results.
  - Hunting tolerance stays 0; one accepted event was checked and changes nothing.
  - "Settling before the tie-break" turned out to be the frozen loop's semantics already
    (failures leave, then re-rank). So the only change is the band.
- **New driver** `experiments/ch_09_parameter_selection/ch_9_7_amendment3.py`. It mirrors
  `ch_9_3_amendment1.py`:
  - selection from the source run's 244 candidate records with the band; the Stage-I point
    stays as the e_q reference, never ranked;
  - the unchanged `ch_9_3_select.py`;
  - the settling loop of `ch_9_6.settle` with a configurable host split (`--local-slots`,
    `--remote-slots`; `slots.json` in the run folder overrides them from the next batch on);
  - take-over of finished checks of the same test from `--reuse` folders (same arguments as
    the source run's checks, same weights; checked against `run_meta.json`);
  - the W5 report;
  - result `ch9_selected_uc_2026_10_a3`. SELECTED_PARAMSET is not switched.
  - Guard: the files an e5power shard runs must not differ from e5power's HEAD
    (shard-neutral paths: docs/, tests/, MC code, `ch_9_6`, `ch_9_7`).
- **Transport fix in `ch_9_6_full_procedure.py`:**
  - `ps()` logs the stderr tail (1500 characters) instead of the head (400);
  - the pull of the shard logs (`_remote`) is no longer fatal (the 19:12 crash).
  - Master side only; the shard code is unchanged.
- **Tests:** `tests/experiments/test_ch9_amendment3.py` (5: band ends, host split,
  same-test check against the source run's dd14 check). 20 passed, together with the MC
  window tests.
- **Dry run** (`--dry-run`, `results/_dry_ch9_7`): 36 / 244 inside the band, 31 feasible,
  ranked first aa95, ffca, 5891, 55cf. This matches the offline what-if.
- **Diagnostic 1553 (θ 0.0142, outside the band) stopped at 20:18** on e5power, to free its
  workers. Its result was no longer needed.
- **Committed 8b02929.** Run started 20:19:
  `results/ch_09_parameter_selection/0002_2026-11-01_201949_ch_9_7_amendment3` (driver
  pid 115256, detached).
  - Batch 1: aa95 and ffca on e5power.
  - A watcher (scratchpad `slots_watch.ps1`) widens `slots.json` to 2 local once d162/e63a
    have finished (≈22:45), and to 3 remote once the c08b probe has finished.
  - d162 and e63a are then taken over.
  - Expected: about 4 batches in all, result Monday ≈08:00–11:00.
- **20:27: band widened to 0.02–0.04 pu (author).**
  - Run 0002 stopped (driver plus 24 processes on e5power) and marked `SUPERSEDED.txt`;
    watcher stopped.
  - Commit fe30478.
  - Dry run: 50 / 244 inside, 38 feasible, 23 in the tie group, d162 last.
  - Launch held: the author asked whether a new full campaign is the better path.
  - Claude recommended the re-selection. A full campaign under the same rules would probably
    repeat the failure, because the II-b centring does not see settling. A full campaign is
    worth it only with a settling-aware search, which needs a design note first.
- **20:38: re-selection started (author: "Yes. Run re-selection", 30 cores per host).**
  - Run `0003_2026-11-01_203826_ch_9_7_amendment3` (driver pid 129848).
  - Batch 1: aa8e here; aa95 and ffca on e5power.
  - Watcher `slots_watch.ps1` (rewritten after a sed escaping error; the broken version
    failed at parse and wrote nothing): 1 + 2 now, 3 here once d162/e63a have finished,
    3 on e5power once the c08b probe has finished.
  - Expected: result Monday ≈11:00–13:00.
- **23:40: batch 1 done; first verdicts under the band.**
  - Run 0003, batch 1: aa8e, aa95 and ffca all FAIL. e63a (diagnostic, taken over) also FAILS.
  - Batch 2 started at 23:17: 5891 here; 55cf, efee and 77ef on e5power.
  - The watcher switched to 3 + 3 at 23:35, so batch 3 onwards runs six checks.
  - The `_remote` shard-log pull for aa95 failed at 23:01. The cause: `Expand-Archive`
    found no zip, probably because ffca was still writing its log in the same folder. The
    failure is now non-fatal (fix in 8b02929), and aa95's log arrived with ffca's pull at
    23:06. Nothing is lost.
  - Diagnostics (criterion of run 0001, `_criterion`):

    | candidate | source | θ (pu) | dso_g_v/g_q | g_w_tso_oltc | p95 n_settle | median residual (Mvar) | verdict |
    |:--|:--|--:|--:|--:|--:|--:|:--|
    | c08b1d3d33ab | probe, e5power | 0.0195 | 3.4e3 | 16.7 | 21.0 | 0.27 | pass |
    | d162c73ca1fc | diagnostic | 0.0208 | 3.1e4 | 36.9 | 26.0 | 0.71 | pass |
    | e63aed2d04cb | diagnostic, taken over | 0.0279 | 1.09e5 | 27.1 | 30.0 | 1.46 | FAIL |

  - **d162 passes, but narrowly:** p95 is 26 against a limit of 27, 2 of 120 steps lie
    above 27, and DSO_3's own p95 is 27.55. Its residual of 0.71 Mvar is twice the 0.35
    predicted at 19:45.
  - **c08b settles on the new base** (residual 0.27 Mvar), but it remains inadmissible: θ
    0.0195 is below the band, and it fails the hunting limit (g_w_tso_oltc 16.7).
  - **Updated empirical relation, 14 checked candidates:** residual ≈ 0.0043 ·
    (dso_g_v/g_q)^0.50, Spearman 0.98. It reaches 1 Mvar at ratio ≈ 5.4e4, not 1.1e5 as
    estimated at 19:45 from 8 candidates.
    - a1de (1.29e5) and 8054 (1.12e5) are therefore now predicted to FAIL (≈1.5 Mvar).
    - In the tie group, only d162 has a ratio below 5.4e4.
  - **Hypothesis, not a result:** every candidate ranked ahead of d162 fails, and d162 is
    selected after about four more batches.
    - It would be selected only after every candidate ranked above it has been checked,
      including band-shift entrants (f34d, 1bcd, 6655 are possible).
    - The procedure stays as declared. The prediction is not used to skip checks.
- **2026-11-02 07:55: batches 2 and 3 done.**
  - Batch 2 (5891, 55cf, efee, 77ef) and batch 3 (96d0, d21f, 95b8, 2028, e3bd, 794f) all
    FAIL, with median residuals of 1.50–3.09 Mvar.
    - All ten were predicted to FAIL by the 23:40 relation (ratio ≥ 1.1e5). They were not
      part of the fit.
  - d162 was taken over at 01:59: PASS (p95 26, 0.71 Mvar).
  - Batch 4 started at 05:52: band entrants 1bcd, 6655 and f34d, plus a1de, c38b and 8054.
    Predicted residuals are 1.4–2.2 Mvar.
  - If batch 4 fails entirely, the rebuilt order (`rank_lexicographic`, computed offline)
    puts e738, 6cb6, d088, 17cd and f7ce ahead of d162. After those, d162 heads the order
    with no new entrant ahead of it.
  - **`slots.json` set to 2 + 3 for batch 5.**
    - A 3 + 3 batch would add 58b5, which ranks below d162. Its verdict cannot change the
      selection.
    - With three concurrent checks here (40 logical CPUs), a check takes 3 h 10 min to
      3 h 55 min, against about 2 h 40 min with fewer.
    - The selection depends only on the verdicts in rank order, so the outcome is the same.
  - **Expected end:**
    - batch 4 ≈09:05–09:50;
    - if batch 5 is needed, it runs to ≈12:00–12:50;
    - then W5 (≈30 min), so DONE ≈12:30–13:20.
    - If a batch-4 candidate passes: DONE ≈09:40–10:20.
- **08:39: no batch 5 (author).**
  - The author wants to discuss the results after batch 4, revise the procedure so that it
    can be expected to produce a winner, and then run a complete new campaign.
  - `slots.json` set to 0 + 0. After batch 4, the driver records the verdicts:
    - if a batch-4 candidate passes, it is selected (W5, result);
    - otherwise the next batch is empty, `ThreadPoolExecutor(max_workers=0)` raises, and
      the driver writes `FAILED.txt`. Nothing new is launched.
- **Settling-predictor validation** (`tuning/settling_predictor.py`, a linear STS model on
  cached sensitivities; script and outputs in
  `results/diagnostics/2026-11-02_settling_predictor_uc/`):
  - n = 28 measured checks. Spearman 0.986 for the residual, 0.945 for p95.
  - Calibration: measured residual ≈ 1.7 · predicted^0.60; measured p95 − predicted p95
    ∈ [−1, +5].
- **Old search box.** In run 0001's II-a draws, only 3 of 512 meet θ band ∧ predicted
  settling, and 1 of 512 with g_w_tso_oltc ≥ 25.
- **Proposal written:** `docs/tuning/2026-11-02_procedure_v2_proposal.md`.
  - Options: A, the minimal adaptation; B, the cascade design (STS first, verified once;
    TS-only search), which is recommended.
  - Not implemented; the author decides.
- **08:57: run 0003 ended, by a crash rather than the zero-slot stop.**
  - All six batch-4 checks had finished.
  - Error: `run_remote` raised PermissionError [WinError 32] in the shard-log pull
    (`_remote`). The three remote checks finished within one minute, so their pulls copied
    the same log files onto the same local files at once. The guard of 8b02929 covered
    only RuntimeError.
  - The exception surfaced when the last local check (c38b) ended, so the driver recorded
    no batch-4 verdict.
  - Verdicts computed read-only with `_criterion` into
    `expC/batch4_verdicts_readonly.json`: 1bcd (1.44 Mvar), 6655 (1.52), f34d (2.47),
    a1de (1.07, p95 25), c38b (2.00), 8054 (1.31). All FAIL.
  - Run note `STOPPED_BY_AUTHOR.txt`. No batch 5 was launched, and both hosts are free.
  - On the commitment base: 29 checks, one pass (d162).
  - **Fix** (`ch_9_6_full_procedure.run_remote`):
    - a module lock `_PULL_LOCK` serialises the shard-log pulls;
    - `OSError` is non-fatal as well as `RuntimeError`.
    - Tests: `tests/experiments/test_ch9_remote_transport.py` (2 tests: a locked file
      does not end the run; three shards finishing together pull one at a time).
      7 passed, with the amendment-3 tests.
  - Validation re-run with batch 4 (n = 30): Spearman 0.985 (residual), 0.952 (p95);
    calibration unchanged (1.69 · predicted^0.60).
- **Author decisions on v2:** option B; DSO band lower edge stays 0.92; 12 workers on
  e5wimi (first 15) and 36 on e5power; the TS tap engagement is left to Claude, but the "OLTCs should
  not be just inert".
- **TS-tap analysis** (run 0001, 248 candidates; TS machine transformers only):
  - Hunting-free operation needs a tap price of about 25 or more. Above that, TS taps move
    almost only in W6 (952 of 964 moves) and are inert in W1–W4.
  - 14 hunting-free candidates have θ_TS,min ≤ 0.009; 8 of them act in at least 2 of W1–W4.
  - Proposal (§ 6a of the v2 note):
    - Δv_eng,TS unchanged at 0.0075;
    - new bound θ_TS,min ≤ 0.009 pu ("not inert"), applied analytically to the draws and
      in the filter;
    - hunting tolerance 0;
    - N_2A 512;
    - residual margin measured on three STS variants (≤ 0.8 Mvar, p95 ≤ 25) rather than
      predicted.
  - Awaiting the go to freeze and implement.
- **Further author constraints:**
  - e5wimi first 15, then 12, finally **8 workers**; e5power **36**.
  - Hunting: "you can accept one hunting event maybe, but of course we would prefer 0".
    Implemented as a fallback: tolerance 0; tolerance 1 (`--max-hunting 1`) only if
    nothing is feasible at 0. That yields the same winner as ranking zero-hunting
    candidates first.
  - The author changed only the cores after the proposal, so the remaining v2 values are
    taken as accepted.
- **Procedure v2 frozen and implemented (commit f51dffc):**
  - Frozen text: `docs/tuning/2026-11-02_weight_selection_procedure_v2.md`.
  - Driver: `experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py`. It reuses the
    `ch_9_6` transport, simulation units and the unchanged `ch_9_3_select.py`. New parts:
    - Stage I with three STS variants;
    - a settling verification with a measured margin;
    - spans that hold the STS weights;
    - a screen and filter θ_TS,min ≤ 0.009;
    - the hunting fallback;
    - a settling confirmation, one check at a time on e5power.
  - Tests: `tests/experiments/test_ch9_procedure_v2.py`, 4 tests. 11 passed in total with
    the transport and amendment-3 tests.
    - The Stage-I test confirms θ_STS 0.030, θ_TS 0.0075, λ_STS 1.400, and that the TS part
      is identical to `stage1_start_uc_2026_10`.
  - Dry run: 288 / 512 II-a draws pass λ, 297 pass θ_TS, 159 pass both.
- **09:19: campaign started**, run `0004_2026-11-02_091930_ch_9_8_procedure_v2` (driver pid
  76304).
  - e5power at 3d98d31; the 17 differing files are shard-neutral.
  - Variant parameter sets `stage1_v2_uc_2026_11_pq10/20/30` written.
  - Verification shards launched on e5power at 09:19:48, 3 × 12 workers.
  - Expected: verification ≈12:00–13:00; II-a ≈14.5 h; II-b ≈9 h; confirmation and W5
    ≈2 h. The result is expected Tuesday afternoon or evening (corrected; "Wednesday morning" was an addition error).
- **09:47: author raised e5wimi to 10 workers.**
  - Driver 76304 stopped during the verification wait; it had no local work. The three
    shards on e5power (24900, 12036, 14420) kept running.
  - Resumed with `--resume ... --local-workers 10` (pid 97020). `run_remote` finds the
    shards ALIVE and waits; nothing was relaunched, and the paramsets were reused.
  - Revised estimate:
    - verification ≈11:30–12:00;
    - II-a ≈14 h (10 + 36 workers);
    - II-b 3–4 rounds, ≈9–14 h;
    - confirmation and W5 ≈2 h.
    - Result Tuesday ≈13:00–18:00.
- **Thesis text (author: "start to fix the tuning process description").**
  - `Z:\latex_diss_clone\Chapters\Chapter08.tex`, method of Experiment B revised to
    procedure v2: STS first, five steps, the new requirements of items 2 and 3, the
    weights table restructured.
  - Results are left as red `\todo` until run 0004 finishes.
  - Log: `Z:\latex_diss_clone\00_daily_log\2026-11-02_ch8_procedure_v2_text.md`.
  - Not committed: the clone holds other uncommitted edits (Ch 7, Ch 10).
  - Ch 7 (author "Yes, do it here"):
    - the active-power schedule now describes the unit commitment (new
      `eq:setup:commitment`, `eq:setup:dispatch` rewritten) and the ±1200 Mvar envelope of
      the equivalent;
    - test build without errors;
    - the reference-period numbers (old rule) are still to be recomputed.
  - Commitment in the tuning windows (12 h; `compute_merit_order_dispatch` via the
    runner's step [9]): all five machines in W1–W5; G9 off in W6; G3 committed in every
    tuning window.

- **11:06: STS verification of run 0004, all three variants PASS** (120 steps each; criterion
  of run 0001):

  | π_q | ρ | p95 n_settle | per-DSO p95 | max | steps > 27 | median residual (Mvar) | predicted (calibrated) residual / p95 bound |
  |--:|--:|--:|:--|--:|--:|--:|:--|
  | 10 | 2.5·10⁴ | 24.0 | 18.55 / 13.2 / 24.55 / 16.65 | 27 | 0 | 0.61 | 0.60 / ≤ 23 |
  | 20 | 1.25·10⁴ | 23.0 | 17.55 / 13.1 / 24.1 / 16.1 | 27 | 0 | 0.41 | 0.44 / ≤ 22 |
  | 30 | 8.3·10³ | 23.0 | 17.55 / 16.2 / 23.55 / 15.75 | 27 | 0 | 0.31 | 0.36 / ≤ 21 |

  - **Choice by the declared rule:** π_q 10 (largest ρ with p95 ≤ 25 and residual ≤ 0.8);
    Stage-I point `stage1_v2_uc_2026_11_pq10`.
  - For comparison, the v1 candidates had 10–59 steps above 27, and d162 had 2.
  - The calibrated predictor hit the residual within 0.05 Mvar. p95 came out 1–2
    iterations above "predicted + 5", still inside the margin.
- **II-a started at 11:06:** 159 candidates plus the Stage-I point; 34 here (10 workers), 125
  on e5power (36).
  - The first local attempt failed at start: `OSError: Could not find/load ... llvmlite.dll`
    while 13 worker processes were starting. A DLL load race is suspected; memory was not
    short (151 GB free).
  - `simulate()` retried at once; attempt 2 is running. Remote shard launched at 11:06:51.
  - For about 5 min, 13 workers ran here: 10 for the campaign plus 3 for the W6 preview
    started at 10:44 (`results/diagnostics/2026-11-02_v2_stage1_preview_W6`).

- **W6 preview of the three Stage-I variants** (12 h; 3 local workers 10:44–11:10;
  `results/diagnostics/2026-11-02_v2_stage1_preview_W6`; compared with run 0001 records on
  the same base and window):

  | W6 | e_v(TS) | e_v(DS) | e_q | worst STS dev. p95 | TS / STS tap moves | hunting |
  |:--|--:|--:|--:|--:|:--|:--|
  | v2 π_q 10 / 20 / 30 | 0.01525 | 0.0183 / 0.0190 / 0.0192 | 7.7 / 7.0 / 6.8 e5 | 0.052 | 9 / 18–21 | 1 (TS 1[0]: 180 s −1, 5400 s +1) |
  | v1 Stage I | 0.0153 | 0.0111 | 1.80e6 | 0.043 | 9 / 28 | 2 |
  | v1 d162 | 0.0155 | 0.0097 | 2.04e5 | 0.045 | 4 / 6 | 0 |
  | v1 068e | 0.0154 | 0.0061 | 9.64e5 | 0.020 | 7 / 12 | 0 |

  - **STS side as designed:**
    - interface-Q error 2.3–2.6× below v1's Stage-I point;
    - STS taps active (18–21 moves in 12 h);
    - no hard-limit violation.
  - **Cost:** the STS voltages deviate more from 1.03 (e_v(DS) +65–73 %; worst-bus p95
    0.052).
    - Open: the record does not say whether this is above 1.08 (soft band) or below.
  - **TS side:** with the v1 TS weights it is unchanged. e_v(TS) is the same, and TS taps are
    active, but the start-up reversal of 1[0] counts as one hunting event. This is c08b's
    failure pattern; the TS search has to avoid it, or the tolerance-1 fallback applies.

- **15:05: II-a progress** 55 / 175 here, 224 / 625 on e5power. Expected end ≈23:45 (local
  is slower); result Tuesday ≈09:00–12:00.
- **Peek** (unit files copied read-only to the scratchpad; 51 candidates with all five
  windows; means over W1–W4, W6 of the per-window values, NOT the selection's aggregate):

  | | e_v(TS) | e_v(DS) | e_q | hunting | TS moves (windows of W1–W4) |
  |:--|--:|--:|--:|--:|:--|
  | v2 Stage I (π_q 10) | 0.00745 | 0.0127 | 5.0e5 | 1 (W6 start-up) | 17 (4) |
  | v2 e8c9 (best hunting-free so far) | 0.00732 | 0.0077 | 1.5e5 | 0 | 22 (4) |
  | v1 Stage I | 0.00753 | 0.0084 | 1.3e6 | 2 | — |
  | v1 d162 | 0.00739 | 0.0082 | 1.8e5 | 0 | — |
  | v1 068e / aa8e | 0.00761 / 0.00764 | 0.0047 / 0.0051 | 6.8e5 / 1.1e6 | 0 | — |

  - Hunting-free (taps and banks): 4 / 51, all with TS moves in at least 2 of W1–W4.
    Another 10 have exactly one hunting event.
  - The other filters (wear, AVR step, e_q bound) are not yet applied, and the official
    aggregate may rank them differently.

- **Author: 20 workers on e5wimi for anything started after 18:30, until tomorrow morning.**
  - Problem: a resume recomputed the host split from the worker counts. Candidates could
    move between hosts while the remote shard still holds its old list, so units would be
    simulated twice.
  - Fix: `ch_9_6.kept_split` (commit ef594bd) reuses `shard_local/shard_remote.json` when
    they cover the same candidates. Test added; run 0004's stage2a split (34 / 125) is
    reused.
  - Plan for 18:30 (timer set):
    1. stop driver 97020 and the local stage2a shard tree (in-flight local units are lost);
    2. resume with `--local-workers 20`. The remote shard `stage2a_0` keeps running and is
       waited for; II-b rounds then use 20 here.
    3. If a local shard is still running in the morning, restart with 10 (safe now).
- **18:31: restart done.**
  - Stopped driver 97020 and the local stage2a shard tree (pid 121196). At that time 106 /
    175 local units were done.
  - Resumed as pid 72044 with `--local-workers 20`. The log shows "split of the earlier
    invocation kept", 34 here (20 workers) and 125 on e5power (36).
  - Remote shard `stage2a_0` (pid 12456) kept running; there was no new launch.
  - At 20:01: local 138 / 175, local shard pid 78672 with 20 spawn workers; remote 523 /
    625. II-a ends ≈21:40–22:00.
- **22:07: II-a done (local 21:27, remote 22:05); selection 2a:**
  - 160 / 160 inside the engagement bounds; **12 feasible with hunting tolerance 0**, so the
    fallback is not needed.
  - Official aggregate:

    | | e_v(TS) | e_v(DS) | e_q | hunting | g_w_tso_oltc |
    |:--|--:|--:|--:|--:|--:|
    | Stage-I point (π_q 10) | 0.00845 | 0.0130 | 5.5e5 | 1 | 21.3 |
    | 1. e8c95a175235 | 0.00836 | 0.00826 | 1.68e5 | 0 | 17.0 |
    | 2. 436000bda618 | 0.00835 | 0.00838 | 1.90e5 | 0 | 16.8 |
    | 3. a7aa0bf5bd28 | 0.00846 | 0.00857 | 2.21e5 | 0 | 21.3 |
    | 4. ead3ee4f5638 | 0.00841 | 0.01085 | 4.59e5 | 0 | 20.7 |

  - For comparison, v1's leaders on the same aggregate were 0.00857–0.00860 (068e, dd14),
    and the best of the amendment-3 band was 0.00844.
  - II-b round 1 started at 22:07: centre e8c9, f 0.5, 54 / 64 draws pass the screen; 19
    here (20 workers), 35 on e5power (36).
  - Local throughput is ≈1.2 units per worker-hour against ≈1.67 remote, so the rounds are
    local-bound, ≈4 h each. Revised result: Tuesday ≈11:00–14:00, depending on 3 or 4
    rounds and the morning cut to 10 workers.
- **22:35: rebalance from round 2 (author: "You can rebalance from round 2 on").**
  - Commit 0903e07: `ch_9_6.simulate(split_local=...)` and `ch_9_8 --local-speed`, i.e. a host
    split weighted by throughput. Test added (13 pass).
  - 0.75 = e5wimi with 20 workers at ≈1.2 units per worker-hour against e5power at ≈1.6. A
    54-draw round goes from 19 / 35 to 16 / 38.
  - Restart: stopped driver 72044 and the round-1 local shard (pid 125964; 5 of 95 units
    were done, and ≈28 min of 20 workers were lost).
  - Resumed as pid 6216 with `--local-speed 0.75`. The resume re-checked stage2a (nothing
    re-run, 0 files pulled), produced an identical selection 2a, and kept the round-1 split
    (19 / 35). e5power `round1_0` was not relaunched.
  - Estimate: round 1 ends ≈02:30 (local-bound). Rounds 2–4 ≈3.3 h each, balanced; slower
    once e5wimi is cut to 10 workers in the morning. Result Tuesday ≈11:00 with 3 rounds,
    ≈13:00–14:00 with 4.
- **22:46: e5power check** (author: "E5power has 36 workers max"). Our processes there are
  36 spawn workers of the round-1 screen, the screen parent and the shard launcher. The
  other 39 python processes on e5power are not ours.
- **22:50, author: "No restart anymore, keep like this."** No further restarts, so the
  planned morning cut of e5wimi to 10 workers is dropped: 20 here and 36 on e5power until
  the end. Revised estimate without that cut: result Tuesday ≈10:00 with 3 rounds,
  ≈13:00 with 4.
- **2026-11-03 07:34, status:** II-b round 3 of 4 running.
  - Rounds 1 and 2 each gave a new best:

    | Round | Leader | e_v,TS | e_v,DS | Feasible (hunting 0) |
    |---|---|---|---|---|
    | 1 | d15a97b38160 | 0.00824 | 0.00865 | 31 / 214 |
    | 2 | 5b98cb042d56 | 0.00824 | 0.00836 | 45 / 254 |

  - Round 2 changed the leader only through the e_v,DS tie-break inside the 2 % band; e_v,TS
    has stayed at 0.00824 since round 1.
  - The leaders at g_w_tso_oltc ≈ 17 have no hunting with the pq10 STS design. The < 20
    threshold of 2026-11-01 19:45 was found with v1 STS weights.
  - The STS taps of the leaders move little: worst window 0–4 operations per day (5b98: 2;
    TS taps 22 per day). Raise this with the author when the result is reported.
  - `v_worst_ds` (0.052) is the p95 over time of the worst-bus deviation from v_set, not an
    absolute voltage. `n_viol_v_ds` counts against `cfg.v_min_pu`/`v_max_pu` (default
    0.90/1.10), so the 1.08 question stays open; check it on the winner's records.
- **07:59, author: "For round 4, you should use 12 workers max on e5wimi."**
  - The worker count is fixed per invocation, so this needs a resume. It is timed so that
    nothing is lost:
    - driver 6216 stopped at 07:59:22;
    - the round-3 local shard (pid 71456, 60 / 70 units done) left running with its 20
      workers.
  - A detached watcher (scratchpad `resume_r4_12w.ps1`, pid 32824) waits for that shard to
    exit, then resumes with `--local-workers 12 --local-speed 0.9`.
    - 0.9 is the per-worker throughput expected at 12 workers, estimated from the 10-worker
      phase.
  - Expected behaviour of the resume:
    - it re-checks the finished stages and keeps round 3's split (13 / 33);
    - it waits for e5power's `round3_0` without relaunching it;
    - round 4 is split with weight 10.8 against 36, i.e. ≈23 % of the draws here.
  - Estimate: round 3 ends ≈09:00 (e5power-bound), round 4 ≈3.2 h, confirmation and W5
    report ≈1.5 h. Result ≈13:45–14:45 if the first settling check passes.
- **08:22: resumed as planned.**
  - The round-3 local shard exited at 08:22:38; the watcher started driver 136736 at
    08:22:40.
  - The resume re-checked II-a and rounds 1–2 (nothing re-run, 0 files pulled, identical
    selections) and kept round 3's split; e5power's `round3_0` was not relaunched.
- **08:51: round 3 done.**
  - 65 / 300 feasible at hunting 0.
  - New best 1b945ad40b42: e_v,TS 0.00827, e_v,DS 0.00828; TS taps 18 per day, STS taps 4
    per day; g_w_tso_oltc 17.7.
  - It leads by the e_v,DS tie-break inside the 2 % band (the lowest feasible e_v,TS is
    0.00812, so the band reaches 0.00828). The leader's e_v,TS has not improved since
    round 1.
  - Round 4: centre 1b94, f 0.5, 37 / 64 draws pass the screen; 8 here with 12 workers
    (verified: 12 pool processes), 29 on e5power with 36.
  - Estimate: round 4 ends ≈11:50 (e5power-bound, 145 units). Result ≈13:15–13:45 if the
    first settling check passes.
- **12:01: round 4 done; stage II-b ends after 4 rounds.**
  - e5wimi shard exited 11:32, e5power shard 11:56 (pulled 12:00).
  - 80 / 337 feasible at hunting 0; best unchanged 1b945ad40b42, so f → 0.25.
  - Runner-up a531b32f7109 (round 4): e_v,TS 0.008247, e_v,DS 0.008281, same tap counts as
    1b94. The two differ by 0.04 % in the tie-break: the optimum is flat.
- **13:00: settling confirmation of 1b94 PASSES** (e5power, 36 workers, 12:01–13:00).
  - p95 22 (≤ 27), median residual 0.42 Mvar (≤ 1), max 27, 0 censored.
  - It differs from the pq10 verification (p95 24, 0.61 Mvar) although the STS weights are
    identical: the `frozen_ofo` variant lets the supervisory OFO solve once at t = 0 with the
    candidate's TS weights (`ch_9_1_ninner_isolated_sts.py:757`), so the TS weights set the
    starting operating point of the steps. DSO_3 is unchanged (p95 24.55).
- **13:32: run 0004 DONE** (driver rc 0 after 5.16 h of the last invocation).
  - Result `configs/paramsets/ch9_selected_uc_2026_11_v2.json` (key 1b945ad40b42, θ_TS,min
    0.0062 pu, θ_STS,min 0.03 pu, λ_max(B) 1.40).
  - SELECTED_PARAMSET not switched; the author decides.
- **Winner against the Stage-I point** (W1–W4, W6; worst window per wear metric):

  | Metric | 1b94 | Stage-I point | 
  |---|---|---|
  | e_v,TS | 0.00827 | 0.00845 |
  | e_v,DS | 0.00828 | 0.01302 |
  | e_q (interface Q error) | 1.87·10⁵ | 5.49·10⁵ |
  | TS taps / reversals per day | 18 / 2 | 18 / 4 |
  | STS taps per day | 4 | 14 |
  | bank commits (sum) | 0 | 2 |
  | hunting events | 0 | 1 |

- **W5 report (held out, not used for selection):**

  | Metric | 1b94 | Stage-I point |
  |---|---|---|
  | e_v,TS | 0.00933 | 0.00795 |
  | e_v,DS | 0.01276 | 0.01175 |
  | e_q | 0.99·10⁵ | 1.01·10⁵ |
  | TS taps per hour | 1.25 | 0.50 |
  | bank commits | 0 | 21 |
  | hunting events | 2 (TS trafo 3[1], dwell 33 and 63 min) | 1 |

  - W5 is the window with structurally zero DER reactive capability (VDE dead zone).
    Hypothesis: there the banks are the main remaining Q resource, and the winner's bank
    price (g_w_tso_shunt 88.2 against 27.0 at the Stage-I point) keeps them uncommitted.
  - Reported only; no re-selection on W5 (no tuning on the test data).
- **Against v1 (run 0003, selection_a3):** leaders e_v,TS 0.00859–0.00860 (e_v,DS
  0.0053–0.0067), all failing settling; d162, the only pass, e_v,TS 0.00846, e_v,DS 0.00833,
  p95 26, 0.71 Mvar. v1 ran before the DSO band 0.92–1.08 was set, so the e_v,DS values are
  not strictly comparable.
- **Actuator engagement of the winner, per window:**
  - TS taps: 1–2 moves in each of W1–W4, 9 in W6 (12 h each).
  - STS taps: 0 in W1–W4, 4 in W6 (at most 2 per transformer). Banks: 0 commits in every
    window, W5 included.
- **DS voltage against 1.08 pu: not answerable from the stored rows.**
  - The screen stores no time series; `v_worst_ds` is the p95 over time of the worst-bus
    |v − 1.03|. The winner's worst window W4 has 0.0517, i.e. the worst bus lies outside
    [0.978, 1.082] for 5 % of the time; which side is not stored.
  - The Stage-I point sits at 0.0518–0.0519 in four windows. The repeated value suggests a
    bus held at the soft DSO band edge with ≈0.002 pu excess (a hypothesis).
  - A record-saving simulation of the winner is needed to settle it.
- **13:42: record-keeping re-run launched on e5power (author: "yes, you can run it").**
  - New script `experiments/ch_09_parameter_selection/ch_9_10_winner_records.py`: sets up
    each (candidate, window) exactly as `ch_9_3_stage2_screen._run_one_window`, recomputes
    the screen's row (to check the re-run against the stored window), and keeps a DSO-band
    summary per HV group, the voltage envelopes (npz) and the record list (gzip pickle).
  - Candidates: the winner 1b94 and the Stage-I point b46b, W1–W6, 12 workers; launched
    through the `ch_9_6` shard mechanism (pid 25400). Output
    `results/diagnostics/2026-11-03_winner_records_0004/out`.
- **Author question: why no bank commits and almost no STS taps; can the weights be lowered
  with every filter still met?** Analysis of all 337 evaluated candidates (read-only):
  - **Banks.**
    - g_w_tso_shunt < 25: 62 candidates, all commit banks (median 11–32 commits in 60 h),
      none feasible.
    - 184 candidates commit banks, 30 of them feasible. Rejection counts among the others:
      unexplained tap hunting 148, bank hunting 51, tap operations 15 (several per
      candidate).
    - All 10 feasible candidates inside the 2 % band commit no bank. The best feasible bank
      user, 1528ece0d32a (1 commit), has e_v,TS 0.00831, 0.4 % outside the band (0.00828).
      Feasible bank users commit 1–4 banks in 60 h.
    - The bank users with the lowest e_v,TS (0.00797–0.00806, below the winner) are
      rejected for tap hunting. 307 of their 1310 unexplained reversals start within 30 min
      of a bank move (median distance 66 min), so a causal link is not established.
    - The ledger check of the shunt feedforward on release (other session, running since
      12:01) tests exactly the candidates in the band with bank releases.
  - **STS taps.**
    - g_w_dso_oltc is fixed in Stage I at θ_STS,min = 0.03 pu.
    - The winner's per-bus RMS DS deviation is 0.0043–0.0114 pu per window, so the tap
      commit condition (an h-weighted mean deviation ≥ θ) is rarely met.
    - Even the filter's lower bound θ 0.02 is about twice the largest window RMS; engaging
      the taps in W1–W4 would need θ ≈ 0.005–0.01, outside the frozen bounds [0.02, 0.04].
  - Any change of these weights now is an amendment declared after the results are known.
- **13:52: four diagnostic variants of the winner launched on e5wimi, 4 workers (author:
  "you can run the diagnostic runs also here").**
  - They vary the winner only; they are not a re-selection.
    - `1b94_shunt27`: g_w_tso_shunt at the Stage-I value 26.98.
    - `1b94_sts020`: g_w_dso_oltc 21.68 (θ_STS,min 0.02 pu).
    - `1b94_sts010`: g_w_dso_oltc 10.84 (θ_STS,min 0.01 pu, outside the frozen bounds).
    - `1b94_shunt27_sts020`: both changes.
  - Windows W1–W4 and W6 only. W5 is left out so that the held-out window does not inform
    a possible amendment.
  - 20 units, pid 151840, output `results/diagnostics/2026-11-03_winner_variants_0004/out`;
    ≈3.5–4 h at 4 workers.
  - **13:57: moved to e5power (author: "then use e5power with more workers").**
    - The local run was stopped before any unit finished; its start log is kept as
      `aborted_local_13-52.log`.
    - Relaunched on e5power at 13:58 with 20 workers (shard pid 15196), alongside the
      record run's 12 (32 of 36). The script hash matches the committed version.
    - All 20 units run at once, so the run ends with the slowest W6, ≈15:00–15:10.
- **14:39: record run done (0.94 h); JSON and npz pulled, W4 record pickles pulled.**
  - **Reproduction:** every recomputed row equals the stored run-0004 window to ≤ 3.4·10⁻¹²
    relative (e_v,TS); tap, hunting and bank counts identical. The re-run is the same
    simulation.
  - **DSO band 0.92–1.08 pu, true HV-group voltages (20 s steps):**

    | Window | 1b94: max V, share > 1.08, longest | Stage-I point: max V, share > 1.08, longest |
    |---|---|---|
    | W1 | 1.0700, 0, – | 1.0846, 27.9 %, 159 min |
    | W2 | 1.0517, 0, – | 1.0783, 0, – |
    | W3 | 1.0533, 0, – | 1.0821, 23.0 %, 105 min |
    | W4 | **1.0821, 8.1 % (DSO_4), 58 min** | 1.0821, 18.3 %, 88 min |
    | W5 | 1.0715, 0, – | 1.0758, 0, – |
    | W6 | **1.0805, 0.7 % (DSO_1), 0.7 min** | 1.0909, 27.8 %, 105 min |

    - The lower edge is never approached: the lowest DS voltage of the winner is 0.992 pu (W5).
  - **W4, DSO_4 (winner), during the 174 steps above 1.08:**
    - the controller's measured group maximum is 1.0798 pu (median), the true one 1.0821;
    - its band slack stays 0 (max 7·10⁻⁷);
    - DSO_4's IBG Q is not saturated (median +16.7 Mvar within [−49.5, 61.5]);
    - the three couplers do not move; the interface Q setpoints ask for −15, −41 and
      −36 Mvar.
    - So the DSO controller holds the band on what it measures. Both maxima are taken over the
      configured `voltage_bus_indices` (`_multi_tso_helpers.py:540` and `:679`), so the
      0.0023 pu offset is not a different bus set by construction.
    - Zero-mean noise would bias a measured maximum upward, not downward. Open hypotheses:
      (a) the measurement packet lacks the highest bus (NaN entries are dropped), or (b) a
      systematic measurement offset. Not checked yet: the records keep no per-bus DS
      voltages.
- **14:57: winner variants done (e5power, 0.99 h).** Judged with the frozen filter
  (`stage2_search.aggregate` / `check_limits`, the limits of run 0004, e_q reference = the
  Stage-I point; λ_max(B) taken as 1.40, since only integer prices changed):

  | Variant | Filter | e_v,TS | e_v,DS | e_q | STS tap moves W1/W2/W3/W4/W6 | bank commits | max DS V (W4) |
  |---|---|---|---|---|---|---|---|
  | 1b94 (winner) | pass | 0.008265 | 0.00828 | 1.87·10⁵ | 0/0/0/0/4 | 0 | 1.0821 |
  | shunt27 | **fail**: 1 hunting reversal (W6, TS trafo 1[0], 87 min) | 0.008219 | 0.00912 | 1.60·10⁵ | 0/0/0/0/0 | 4 (W1 1, W6 3) | 1.0822 |
  | sts020 (θ_STS 0.02) | pass | 0.008270 | **0.00780** | 1.92·10⁵ | 0/0/0/2/6 | 0 | 1.0796 |
  | sts010 (θ_STS 0.01) | pass (θ_STS outside [0.02, 0.04]) | 0.008281 | **0.00632** | **1.57·10⁵** | 6/4/4/12/18 | 0 | **1.0673** |
  | shunt27 + sts020 | **fail**: same reversal | 0.008219 | 0.00753 | 1.61·10⁵ | 0/0/0/2/5 | 4 | 1.0738 |

  - All five lie inside the 2 % band (≤ 0.008284). On e_v,DS both STS variants would rank
    above 1b94.
  - sts010: STS taps at most 8 operations per transformer and day, no hunting, no DS voltage
    above 1.0673 pu in any window. sts020 removes the W4 exceedance (1.0796) and leaves W6 at
    1.0801 for 0.1 % of the time.
  - Cheaper banks: banks commit (4 in 60 h), but one unexplained TS-tap reversal on
    trafo 1[0] in W6 rejects the variant. Consistent with the 337-candidate pattern above.
- **15:00: settling tests of sts020 and sts010 launched on e5power** (18 workers each, shard
  pids 23272 and 26628, `results/diagnostics/2026-11-03_winner_variants_0004/expC`). Same
  argv as the run-0004 confirmation apart from the worker count.
- **These are diagnostics, not a selection.** Adopting sts010 changes the frozen STS design
  (θ_STS bounds); adopting sts020 changes a weight fixed in Stage I. Either is an amendment
  declared after the results are known; the author decides.
- **Correction (15:28): λ_max(B) does depend on the STS coupler price.** The filter table
  above assumed 1.40 for all variants. With the run-0004 screen (`stage1_screen.screen_weights`,
  worst of the six operating points):
  - winner and shunt27 1.4000;
  - sts020 and shunt27 + sts020 1.4095;
  - sts010 1.4487.
  - All stay below the screen limit 1.546 and the filter limit 1.6, so no verdict changes.
- **Author question: why are the AVRs so inert; earlier tunings moved them more.**
  - g_w_gen (the AVR setpoint step weight) is 3.67·10⁷ at the winner. The AVRs move
    0.0025–0.0087 pu per 12 h window, with steps of at most 1·10⁻⁴ pu per TS iteration.
  - They are not held by a limit. In W4 the setpoints stay at 1.028–1.033 pu, far from
    V_max 1.05, and generator Q reserves are 0.30–0.50. The weight alone makes them slow.
  - Earlier selections:
    - October sets (`ch9_selected_final_2026_10`, `_band_2026_10`, `_band_2026_10_a1` = c08b)
      had g_w_gen 7.3·10⁵–1.0·10⁶, about 40× lower;
    - September sets (`ch9_selected*`) had 1.55·10⁷.
    - On the commitment base, v1 (runs 0001, 0003) ranked leaders at 1.4–2.4·10⁸.
  - In run 0004 the II-a box was one decade around the Stage-I value 10⁸, i.e.
    1.0·10⁷–9.5·10⁸, and no candidate below 1.04·10⁷ was simulated. The II-b centres had
    3.3–4.2·10⁷.
  - Lower g_w_gen did not cost feasibility (about 30 % feasible in each band down to 10⁷).
    Among feasible candidates, lower g_w_gen goes with lower e_v,TS (correlation +0.33)
    and lower e_v,DS (+0.18).
  - The best feasible e_v,TS of the campaign, 2ef56ba78008, has g_w_gen 1.07·10⁷, at the
    box edge.
  - Hypothesis: the optimum lies below the explored range; the box, not the plant, keeps
    the AVRs inert.
  - Screen for sts010 with lower g_w_gen: λ_max(B) stays 1.4487 down to 3·10⁵ and jumps to
    1.84 (> 1.546) at 10⁵.
- **15:30: g_w_gen variants prepared (author: "you can test them").**
  - sts010 with g_w_gen 10⁷, 3·10⁶ and 10⁶; 15 units, W1–W4 and W6, 15 workers.
  - Files pushed to e5power (`results/diagnostics/2026-11-03_winner_ggen_0004`).
  - Launch as soon as the two settling tests have freed e5power, which is at its 36-worker
    limit now.
- **Author decisions 15:5x (for the procedure-v3 proposal Claude drafts once the settling,
  g_w_gen and ledger results are in):**
  - g_w_gen not too low: about 5·10⁵ is the low end. The queued run gained a fourth variant,
    g_w_gen 5·10⁵ (20 units, 20 workers).
  - **No reserve term** in the objective for the standard case study. Banks should still
    commit, e.g. after load steps and outages; their weights may need to be lower. A
    reserve term may be investigated later, outside the standard case.
  - STS taps unused at the winner is not the intended behaviour; the design rules are to be
    corrected.
- **Speed-up analysis (run 0004, 336 candidates, 1040 worker-h):**
  - **Early abort at the first unexplained reversal** is exact under hunting tolerance 0.
    - Stopping all of a candidate's windows at its first reversal would have saved 67 %;
      stopping each window at its own first reversal, 30 %.
    - The median first reversal comes at 11 % of the window.
  - **Window subset first:** W1 + W6 cost 44 % of the compute and catch 99 % of the
    rejections. No feasible candidate is lost, all 10 band members and the winner are kept,
    and Spearman of e_v,TS among feasible candidates is 0.97. Completing W2–W4 for survivors
    only saves ≈ 42 %, but it is approximate, based on one campaign.
  - Reversals by window: W6 574, W4 524, W3 421, W1 167, W2 93. Share of the compute:
    W6 30 %, W3 22 %, W4 18 %, W2 16 %, W1 13 %.
- **Hunting forensics (first pass):**
  - Of 1780 unexplained reversals in the 251 candidates rejected for tap hunting, 235 (13 %)
    begin with a move in the first 15 min, i.e. the start-up dispatch from the captured
    operating point; 118 (7 %) bracket a bank move.
  - 30 candidates were rejected only for reversals that start in the first 15 min.
  - The shunt27 variant's single reversal is of this kind: trafo 1[0] −1 at 180 s, bank
    z2[0] commits at 5040 s, trafo 1[0] +1 at 5400 s.
  - Median start of a reversal 302 min, near the events at 40–45 % of the window.
- **Early abort, recomputed with a wall-time model** (windows of a candidate run in parallel at
  their own speeds; a reversal at simulated time t is seen after wall_w · t / T):
  - **Tolerance 0:** abort at the first reversal saves 62 % per candidate (corrects the 67 %
    above, which assumed equal simulation speed in every window), or 30 % per window.
  - **Tolerance 1** (author question; abort at the second reversal): 39 % per candidate,
    18 % per window.
  - **What tolerance 1 does to the run-0004 selection:** 116 feasible instead of 80;
    best e_v,TS 0.00811; 27 in the band instead of 10, 3 of them bank users. The leader
    stays 1b945ad40b42, because the bank users in the band have e_v,DS 0.0088–0.0143 and
    lose the tie-break.
- **Author decision: hunting tolerance 1 for v3.** Author proposal: search on W1, W4 and W6
  only, all windows later (settling test, final checks).
- **Replay of run 0004 at tolerance 1** (search on a subset, then the subset's band confirmed
  on all five windows; `scratchpad/subset_tol1.py`):

  | Search windows | Compute (all-window = 100 %) | with abort at 2nd reversal | Subset-feasible / of these infeasible on all | All-window leader at subset rank | All-window band (27) inside subset band | Confirming the subset band on all windows gives |
  |---|---|---|---|---|---|---|
  | W1+W4+W6 | 62 % | 44 % | 163 / 47 | 57 | 17 | 70937b3e3417 (different) |
  | **W1+W3+W6** | 66 % | **41 %** | **116 / 0** | **4** | **27** | **1b945ad40b42 (same)** |
  | W1+W6 | 44 % | 31 % | 163 / 47 | 63 | 25 | a531b32f7109 (different) |
  | W3+W4+W6 | 70 % | 43 % | 116 / 0 | 8 | 27 | 1b945ad40b42 (same) |

  - Under tolerance 1 a candidate's second reversal often comes from W3: 189 candidates
    reverse there, against 52 in W4.
  - Without W3 the search admits 47 candidates that fail on all windows, and its band
    misses the all-window leader.
  - All windows with abort: 63 %.
  - **Author question: W1+W3+W4+W6.**
    - Compute 84 %, or 53 % with the abort. The feasible set is exact (116 / 0).
    - The all-window leader is at subset rank 3; the subset band has 33 members with all 27
      inside. Spearman e_v,TS 0.988. Confirmation gives the same leader.
    - W1+W2+W3+W6 is similar: 52 %, rank 3, band 35.
    - With confirmation of the band on the missing windows, the total is ≈ 55 % of an
      all-window search, against ≈ 45 % for W1+W3+W6.
    - W4 is the window built to exhaust the continuous reactive range and hand authority to
      the tap changers (`scenarios_mc_v2._condition_events`): where v3's engagement rules
      act.
  - **Author decision: v3 searches on W1+W3+W4+W6** ("Yes, fix it").
    - Early abort at the second unexplained tap reversal or the first bank reversal.
    - The subset's 2 % band is completed on W2 and ranked on all five windows.
    - Settling test and W5 report as in v2.
- **Author decisions 16:2x for the v3 campaign:**
  - II-a 512 draws; II-b rounds under discussion (at least 3; Claude recommends 4, as v2).
  - Hosts: 30 workers on e5power and 30 on e5wimi; simulations from 17:00 on.
  - **Go criteria for the follow-ups** (settling figure regeneration, D1+D2, Ch 10 REF):
    - e_v,TS ≤ 1.10 × v2 (0.00827, i.e. ≤ 0.00910);
    - STS couplers act in every search window;
    - banks commit after the events in at least two windows;
    - all wear limits met.
- **Feasibility check of the bank criterion on run 0004** (search windows W1, W3, W4, W6;
  "after an event" = a commit within 2 h of an event time):
  - 54 of 337 candidates have such a commit in ≥ 2 windows; their g_w_tso_shunt is ≈ 9–21.
  - Only 1 of these 54 is feasible at hunting tolerance 1; the others have 3–45
    unexplained TS-tap reversals. Their e_v,TS is 0.00797–0.00812, better than the winner.
  - None of them moves STS couplers in every window, since the STS price was fixed at
    θ 0.03.
  - shunt27 commits in 2 windows but never within 2 h after an event (W1 at 11.75 h, W6 at
    1.4 h and 1.85 h).
  - **So on the v2 code (ledger off), event-responsive banks and hunting tolerance 1 are
    nearly incompatible.** The ledger verdict (≈ 19–20 h) is the gate for the bank part of
    v3.
- **Author 16:3x:** launch v3 only after the ledger verdict, with **36 workers on e5power and
  30 on e5wimi**.
- **Expected schedule (estimates from run-0004 timings):**

  | Step | Expected done |
  |---|---|
  | Settling tests sts020 / sts010 | Tue ≈ 17:00 |
  | g_w_gen variants (e5power, 20 workers) | Tue ≈ 18:05 |
  | v3 document and driver changes (with tests) | Tue ≈ 19:30 |
  | Ledger verdict (other session) | Tue ≈ 19:00–19:30 |
  | Decisions, freeze, code sync, launch | Tue ≈ 20:15 |
  | Stage I verification | Tue ≈ 21:20 |
  | II-a (512 draws) | Wed ≈ 02:00–04:30 |
  | II-b, 4 rounds | Wed ≈ 08:30–14:30 |
  | Completion of the band on W2 | Wed ≈ 09:00–15:00 |
  | Settling confirmation (+1.1 h per failed check) | Wed ≈ 10:00–16:00 |
  | W5 report, v3 result | Wed ≈ 10:30–16:30 |
  | Settling figure, D1+D2 (if go) | Wed ≈ 12:00–18:00 |
  | Ch 10 REF (if go) | Wed ≈ 15:00–21:00 |
- **16:33: bank-release fix brought into this branch (author: "Yes, bring it in").**
  - Cherry-pick of ac69918 (minimal/thesis-core) as a8d5e23; adds config flag
    `shunt_ff_ledger` (default True), so v3 runs with the ledger on.
  - Conflicts resolved by hand:
    - `experiments/runners/_multi_tso_helpers.py` keeps this branch's file and adds the
      new helper `_ledger_feedforward`, unchanged;
    - the daily log `2026-11-02d_ts_pcc_setpoint_offset.md`, deleted on this branch,
      stays deleted. Its account of the fix is on minimal/thesis-core.
  - The added and removed lines are identical to ac69918 (91 lines).
    `tests/test_shunt_ff_ledger.py` and `tests/test_shunt_banks.py`: 19 passed.
  - Not synced to e5power yet; the queued g_w_gen diagnostic keeps the run-0004 code.
  - Reproduction check launched: the winner (no bank commit, so the ledger cannot act) on
    W1 and W2 with the new code, e5wimi, 2 workers, pid 83544,
    `results/diagnostics/2026-11-03_ledger_repro`. Expected: identical rows.
- **16:3x–16:5x, author: start the campaign by 17:30 at the latest**, i.e. before the
  ledger verdict, and shorten the preparation.
  - The g_w_gen diagnostic is dropped: v3's II-a box covers 5·10⁵–5·10⁷.
  - The running settling tests no longer gate anything; v3 verifies its own Stage-I point.
- **Code changes for v3 (16:40–16:55):**
  - `experiments/runners/multi_tso_dso.py`:
    - module hook `STEP_HOOK(step, log)`, called after each recorded step, `None` by
      default (no run changes);
    - exception `EarlyStop`, which carries the records as `partial_log`.
  - `tuning/runner.py`: an `EarlyStop` keeps its partial records for the metrics; any
    other failure still gives none.
  - `ch_9_3_stage2_screen.py`:
    - options `--abort-tap-reversals N` and `--abort-bank-reversals N`;
    - `_abort_hook` counts the window's unexplained reversals once per TS iteration with
      the filter's own `hunting_summary` / `shunt_summary` and sums all the candidate's
      windows through `<out>/_abort/` files, then raises `EarlyStop`;
    - the Stage-I point is never stopped;
    - rows get `early_abort`, and an early stop is not a failed window.
  - New driver `ch_9_11_procedure_v3.py`, which reuses v2's code (`V2`, `P`) and overrides:
    - Stage I: STS at θ 0.01, bank price from Δv 0.005, g_w_gen 5·10⁶; g_w_der, g_w_pcc
      and g_w_tso_oltc from v2's Stage I. Re-running the TS design with the cheaper bank
      would have put g_w_der at 0.49, where v2 had 0 of 40 feasible;
    - search windows W1, W3, W4, W6 with the early stop at 2 tap or 1 bank reversals;
    - selection at tolerance 1 with the engagement levels (couplers + banks, couplers,
      none);
    - completion of the band on W2;
    - go criteria to GO.json / NOGO.json;
    - copy of every changed Python file to e5power with a hash comparison
      (`sync_remote_code`).
  - Tests:
    - smoke run on 12a46c9001b6, W4 + W6: W4 stopped after 1.6 min at its 2nd reversal,
      W6 on the candidate total, both rows `early_abort` with n_hunting 2 + 0;
    - `tests/test_procedure_v3.py`: 2 passed;
    - ledger and bank tests: 19 passed.
  - Dry run: Stage-I point λ_max(B) 1.447, θ_STS 0.0100, θ_TS 0.0075; 100 of 512 II-a
    draws pass the screen (v2: 159).
  - Procedure text: `docs/tuning/2026-11-03_weight_selection_procedure_v3.md`.
- **16:47: v3 campaign launched**, run `0005_2026-11-03_164755_ch_9_11_procedure_v3`, driver
  pid 154356, code 065c3e9.
  - e5power: 13 Python files copied over 3d98d31, hashes equal.
  - Stage-I verification started on 36 workers.
- **Follow-up launcher (author: "Yes, automate")**: `ch_9_12_followups_v3.py`.
  - It waits for DONE/FAILED of run 0005 and acts only on GO.json.
  - On GO:
    1. SELECTED_PARAMSET → `ch9_selected_uc_2026_11_v3`, committed. This happens only with
       `--switch-selected`, the author's decision; D1/D2 and REF read it.
    2. Code and the v3 parameter set to e5power.
    3. Settling figure: the thesis generator `graphics/Ch8/data/2026-10-28_settling/
       make_settling_cdf.py`, unchanged, copied to `<date>_settling` and pointed at
       run 0005 (start: Stage-I verification; final: the confirmation). It rewrites
       `graphics/Ch8/settling_cdf.tex` in `Z:\latex_diss_clone`, with no commit there.
    4. D1 + D2 (`ch_9_4_deadband --workers 30`) on e5wimi.
    5. Ch 10 REF on e5power: scenarios A, B, D, E, GA, GE × L2, S1, S2, O2, O3, M,
       360 min, 6 jobs each, launched 90 s apart; pulled back to
       `results/ch_11_case2_multisystem/ref/`.
  - Tests:
    - the figure step on run 0004's data in a scratch copy of the thesis tree (p95 24
      start, 22 final; .dat and .tex written there);
    - the SELECTED_PARAMSET line pattern matches;
    - `variants.NOT_READY` is empty.
- **17:00: settling tests of the winner variants (diagnostic, e5power 2 × 18 workers),
  scored with `ch_9_3_amendment1._criterion`:**

  | Variant | p95 | median residual | max | over 27 / censored | per-DSO p95 (worst) | steps with a coupler move |
  |---|---|---|---|---|---|---|
  | sts020 (θ 0.02) | 22.0 | 0.42 Mvar | 27 | 0 / 0 | DSO_3 24.55 | 0 of 120 |
  | sts010 (θ 0.01) | **25.1** | 0.34 Mvar | **46** | **5 / 1** | **DSO_3 29.1** | 8 of 120 |

  - Both pass the confirmation criterion (p95 ≤ 27, residual ≤ 1 Mvar).
  - sts020 equals the winner exactly: no coupler moves during the steps.
  - sts010 misses v3's Stage-I verification margin (p95 ≤ 25) by 0.1 at the winner's
    TS weights. Coupler moves lengthen some steps: max 46, one step does not settle.
  - Run 0005's Stage-I verification at θ 0.01 (with other TS weights) may therefore fail
    its margin, and under the frozen text the campaign would then stop.
- **17:07: reproduction check passed.** The winner on W1 and W2 with the current code:
  - code = ledger fix a8d5e23 plus the v3 hook changes, as present at 16:34;
  - every compared field (e_v,TS, e_v,DS, e_q, v_worst_ds, tap rates, hunting, bank
    commits) equals the stored run-0004 rows exactly (difference 0).
  - So the new code changes nothing for a bank-free trajectory.
- **17:15–17:30: v3 amendment 1, author: "we could do that" (the θ ladder) and "use 4 periods
  instead of 3 for the settling".**
  - `ch_9_11_procedure_v3.py`:
    - `criterion()` wraps `ch_9_3_amendment1._criterion` with the budget
      `N_PERIODS × N_inner` = 36 and is patched into that module, so V2's verify and
      confirm use it;
    - `V2.VERIFY_P95` = 34;
    - `THETA_LADDER` = (0.01, 0.015, 0.02), each rung with its own parameter set
      (`stage1_v3_uc_2026_11_pq10`, `_th015`, `_th020`) and verification label (`v2_pq10`,
      the label of the shard already running, then `v3_th015`, `v3_th020`);
    - results go to `stage1/verification/verification_v3.json`, and the choice to
      `stage1/chosen.json`.
  - `ch_9_12_followups_v3.py`:
    - the figure's start data comes from `chosen.json`;
    - the generator's `n.max() <= 36` check is dropped, because with coupler moves a step
      can exceed the axis (sts010: max 46).
  - Checks:
    - sts010 under the new criterion: p95 25.1 ≤ 36, 1 step over 36, pass;
    - dry run: the three rungs give g_w_dso_oltc 10.84, 16.26 and 21.68, λ_max(B) 1.447,
      100 / 512 II-a draws each;
    - tests: 2 passed.
  - Procedure text: amendment 1 added before the verification result.
  - Run 0005 is restarted by `--resume`; its running verification shard (`v2_pq10`) is
    kept.
  - The follow-up launcher is re-armed with the new code, still figure only.
- **17:30: run 0005 resumed** (driver pid 152520, code 00978eb); the `v2_pq10` shard
  is still running on e5power. Follow-up launcher re-armed (pid 151716).
- **Author 17:31 on the 4 periods:** "only that we do not filter too hard; hopefully the best
  candidate nonetheless settles faster, and maybe fulfils the 3 periods limit". So the
  result report also states the 3-period check (p95 ≤ 27) of the selected set. It is
  computed from the confirmation's steps.csv when the run ends; the running driver is not
  changed for it.
- **17:33, author: "Yes, switch on GO".** The follow-up launcher is re-armed with
  `--switch-selected` (pid 34232). On GO it sets SELECTED_PARAMSET =
  `ch9_selected_uc_2026_11_v3` in `_ch9_selected_design.py`, commits it, and then starts
  the settling figure, D1+D2 and the Ch 10 REF runs. On NOGO nothing is switched.
- **17:51: Stage-I verification at θ_STS 0.01 PASSES** (first rung of the ladder):
  - p95 24.0, which meets the 3-period limit 27 as well as the margin 34;
  - median residual 0.37 Mvar; max 35; 0 steps over 36; 0 censored;
  - per-DSO p95: DSO_1 25.65, DSO_2 21.1, DSO_3 24.55, DSO_4 19.1;
  - couplers move in 7 of 120 steps.
  - The v3 Stage-I TS weights settle better than the winner's at the same θ (sts010:
    p95 25.1, max 46).
- **17:51: II-a launched:** 100 candidates + Stage-I point; 37 on e5wimi (30 workers),
  63 on e5power (36); windows W1, W3, W4, W6 with the early stop at 2 tap and 1 bank
  reversals.
- **19:46: II-a done** (1 h 55 min; forecast was 22:00–23:30).
  - Early stop: 94 of 101 records. Median wall time is 8.7 min for a stopped window and
    39.7 min for a full window.
  - θ bounds: 101 of 101 admitted.
  - Engagement level `sts+bank` leaves no ranked candidate: only the Stage-I point is
    eligible, and it is infeasible (9 tap reversals, 1 bank reversal).
  - The driver falls back to `sts`, which admits 91 candidates; 6 are feasible at hunting
    tolerance 1.
  - Leader f6a6c81e99aa:
    - search windows: e_v,TS 0.00882, e_v,DS 0.00624;
    - v2 winner 1b945ad40b42 on the same windows: 0.00896 and 0.00900 (scratchpad
      `v2_on_search.py`), so the leader is −1.6 % on e_v,TS and −31 % on e_v,DS.
  - **Couplers:** every ranked candidate moves them in every search window: 3–26 moves per
    window (W3 fewest, W6 most).
  - **Banks:**
    - No ranked candidate commits within 2 h after an event in any search window. Their
      only commits are in W6 at 1.3–1.9 h, before the load shedding at 5.4 h.
    - All six have g_w_tso_shunt 25–43, the upper half of the box (4.3–42.7).
    - The five records with a post-event commit have prices 13–22 (incl. the Stage-I point
      at 13.5). All five reach the hunting thresholds.
    - In this box, event-responsive banks therefore come with tap or bank reversals: the
      risk stated in the procedure.
  - Reporting artefact: stopped records show tap rates such as "270 per 12 h". This is
    extrapolated from their truncated logs; they are rejected on hunting anyway.
- **19:46: II-b round 1** centred on f6a6c81e99aa (round-1 box for the bank price
  ±0.25 decade, i.e. ≈ 21–68). 55 of 64 draws pass the screen; 20 run here, 35 on
  e5power.
- **Consequence for the go criteria:** the bank criterion (commits after events in
  ≥ 2 search windows) is unlikely to be met by the final band. A NOGO starts none of the
  follow-ups. Whether the follow-ups should still run when only the bank criterion fails
  is put to the author; nothing changed.
- **≈ 20:15: diagnosis of the bank result** (II-a records of run 0005).
  - **Start-up reversal.** 71 of 101 candidates reverse TSO transformer `1[0]` in W6:
    first move at the first dispatch (t = 3 min), back 1.2–1.5 h later.
    - It is the only start-up reversal type.
    - Without it, 55 of the 94 stopped candidates would have had ≤ 1 counted reversal at
      their stop.
  - **Censoring by the early stop.** Of the 188 W1/W3 records of stopped candidates, only 2
    reach the outage + 2 h. The bank criterion was therefore judged on 7 records.
    This is a design error of v3: the early stop and the engagement measure share the
    windows.
  - **Remaining interaction.** In W6, cheap banks commit 2–3 zone-2 banks within ≈ 0.5 h.
    The coupler `DSO_3|trafo_8` reverses 0.7–1.5 h later.
    - Hypothesis: the bank's voltage step is not fed forward, while its Q step is
      (`q_itf_sh_offset`).
    - Tolerance 1 allows one such reversal.
  - **Checked and kept:** a coupler move at an event that reverses 2 min later counts as
    hunting. This is deliberate (`hunting_reversals` docstring, 2026-09-11).
- **20:43, author: "Yes. Rerun with a and b"** (Amendment 2 in the v3 document).
  - Run 0005 stopped at 20:43: driver 152520, local shard, e5power shard 13376 (tree) and
    follow-up launcher 34232.
  - **Code:**
    - `tuning/stage2_search.py`:
      - new `_split_reversals(moves, events, dwell, tol, startup_s)` → (hunting,
        event-explained, start-up);
      - `hunting_reversals(..., startup_s=0.0)` keeps its signature;
      - `hunting_summary` / `shunt_summary` take `startup_s` and report
        `n_startup_exempt` / `startup_detail` (banks: `n_shunt_startup_exempt`,
        `shunt_startup_detail`).
      - Only the pair formed by a device's first two moves can be exempt, and only if
        the first lies before `startup_s`; an event-explained pair stays event-explained.
    - `ch_9_3_stage2_screen.py`: option `--startup-exempt-s S` (default 0), carried in
      the job and passed to the early-stop hook and to the window row.
    - `ch_9_11_procedure_v3.py`:
      - `STARTUP_EXEMPT_S = 900`, added by `screen_argv` to every unit;
      - stage `bank` = search windows without the early stop;
      - `stage_select(..., levels=)`;
      - `bank_grid`, `pick_lowest_price` and `stage2c` (grid 4.27 → leader's price,
        8 values, the leader's own records reused; the pick joins the completion band
        through `complete_band(..., extra=)`);
      - level fallback after a failed settling confirmation;
      - `--reuse-stage1 <run>` copies `stage1/verification` (Stage I is unchanged);
      - provenance `amendment2` in the result.
  - **Tests** `tests/test_procedure_v3.py`, three new: screen arguments per stage,
    start-up exemption, II-c grid and pick. 24 pass with the bank tests.
  - **Validation on run 0005's stored moves:**
    - with the exemption off, the recomputed counts equal the stored ones in every
      window (0 mismatches);
    - with it on, counted tap reversals go from 185 to 113 (72 exempt).
  - Dry run: same 100 / 512 II-a draws.
- **21:20: run 0006 launched** (`0006_2026-11-03_211954_ch_9_11_procedure_v3`, commit
  932b4a9).
  - Driver pid 29448; e5power shard 23364.
  - Stage-I verification taken over from 0005; II-a on 37 + 63 candidates with
    `--startup-exempt-s 900`.
  - Follow-up launcher re-armed for 0006 with `--switch-selected` (pid 34968).
- **23:33: run 0006, II-a done** (remote shard ended 23:28, about 30 min later than
  estimated).
  - Engagement level `sts+bank`: 5 candidates incl. the Stage-I point, 0 feasible.
    - Stage-I point d6aa9ec0ad10 (bank price 13.49): banks after events in all 4 search
      windows, but 8 counted tap reversals and 1 bank reversal.
    - The others (prices 14.7–23.0) stop at 2–3 tap reversals; 6991782d98c8 also has
      16 TSO tap operations per 12 h (limit 15).
  - Level `sts` applies: 95 candidates, 7 feasible at tolerance 1, bank prices 25.2–42.5.
    - Leader b9dcc64d8b49: e_v,TS 0.00875, e_v,DS 0.00606 on W1+W3+W4+W6; price 29.07;
      g_w_gen 1.67·10⁶; 1 counted reversal.
  - Reading (hypothesis): exemption (a) moved the stops later but did not make cheap banks
    feasible. The stops now come mostly from a W6 episode at 1.3–3 h (bank commit and
    release within ≈ 1 h; several couplers and `1[0]` reverse together).
  - II-b round 1 started 23:33 (54 candidates; 20 local, 34 on e5power).
- **2026-11-04 01:38: II-b round 1 done** (2 h 05 min; fewer early stops than in II-a).
  - Level `sts+bank` again 0 feasible: 14 of 155 candidates have banks after events in
    ≥ 2 windows, none passes the hunting limits.
  - Level `sts`: 20 feasible, bank prices 24.6–49.6.
  - New leader fc7fd1df5d51: e_v,TS 0.00875, e_v,DS 0.00556 (tie-break within 2 %);
    bank price 40.1; g_w_gen 8.8·10⁵; g_w_tso_oltc 24.3.
  - Round 2: 27 candidates (10 local, 17 on e5power), started 01:38.
- **02:58: II-b round 2 done.** First feasible candidate at level `sts+bank`:
  77930f9b1f53.
  - Weights: bank price 23.3, g_w_gen 1.15·10⁶, g_w_tso_oltc 22.6, g_w_der 0.016,
    g_w_pcc 0.45.
  - Search windows: e_v,TS 0.00868, e_v,DS 0.00597; 1 counted reversal;
    0 bank reversals; 0 start-up exemptions.
  - Bank commits, none released:
    - W1: z2[0] at 11.65 h, not within 2 h of the outage;
    - W3: z3[0] at 4.8 h, the outage;
    - W4: z2[0] at 5.4 h, the start of the load ramp;
    - W6: z2[0] at 1.45 h, then z2[2] and z2[4] at 1.9 h, in the 1.3–3 h episode, but
      held.
  - The counted reversal: DSO_1|trafo_2[0] in W4 at 5.4 → 5.73 h. This is a move at
    the ramp start reversed within the merged ramp event; it counts by design.
  - Round 3 is centred on it: 34 candidates (12 local, 22 on e5power), started 02:58.
- **04:45: II-b round 3 done** (1 h 47 min).
  - Level `sts+bank`: 2 feasible of 23 bank-engaging candidates.
    - 77930f9b1f53 stays the leader.
    - Second: ccc91db1de92, e_v,TS 0.00877 (inside the 2 % band), e_v,DS 0.00599;
      bank price 19.7, g_w_gen 3.3·10⁶, g_w_tso_oltc 16.1; 0 counted reversals.
  - f → 0.25. Round 4 (last): 46 candidates (17 local, 29 on e5power), started 04:45.
  - Full-window wall time of the leader: W1 26, W3 42, W4 31, W6 56 min. II-c
    therefore needs about 1 h (one wave, W6 sets the pace).
- **06:57: II-b done after 4 rounds** (round 4: 2 h 12 min).
  - Level `sts+bank`: 6 feasible of 43 bank-engaging candidates.
  - II-b leader 8ad82442269b: e_v,TS 0.00869, e_v,DS 0.00593 (77930f9b1f53 has 0.00868
    and 0.00597; the tie-break inside the 2 % band decides). Its weights:
    g_w_tso_shunt 22.76, g_w_gen 1.29·10⁶, g_w_tso_oltc 19.6, g_w_der 0.011,
    g_w_pcc 0.41.
- **06:57: Stage II-c started.** Only 2 of 7 grid prices pass the screen, because the
  contraction bound binds. Screen values at the leader's other weights (recomputed
  here with `V2.passes_screen`'s functions):

  | g_w_tso_shunt | 4.27 | 5.42 | 6.88 | 8.74 | 11.1 | 14.1 | 17.9 | 22.8 (leader) |
  |---|---|---|---|---|---|---|---|---|
  | λ_max(B) | 3.41 | 2.83 | 2.37 | 2.01 | 1.73 | 1.51 | 1.40 | 1.40 |

  - Ceiling 1.546, so the bank price can drop only to about 13–14.
  - θ_TS,min is 0.0069 pu throughout.
  - Hence II-c simulates 14.1 and 17.9 only.
  - Model fact: λ_max(B) grows as the bank price falls, because the bank step enters
    the linearised loop gain. Above about 17, the λ of 1.40 is set by another
    actuator.
- **07:54: Stage II-c done.** Pick: the leader itself, 8ad82442269b (bank price 22.76).

  | g_w_tso_shunt | e_v,TS | counted reversals | bank reversals | windows with banks after events | verdict |
  |---|---|---|---|---|---|
  | 14.1 (58bcdc27c248) | 0.00864 | 2 | 0 | 4 | rejected |
  | 17.9 (29580fc8a748) | 0.00865 | 2 | 0 | 3 | rejected |
  | 22.76 (leader) | 0.00869 | 1 | 0 | 2 | feasible |

  - All three share one counted reversal: DSO_1|trafo_2[0] in W4, a move at the ramp
    start (5.4 h) reversed within the merged ramp event (5.53–5.74 h).
  - The cheaper two add a second: DSO_4|trafo_11[0] in W3, a move at the restoration
    (5.8 h) reversed 2 min later (5.83 h).
  - At these prices bank z3[0] commits at the outage and is released at the restoration
    in W3. The leader's bank stays committed.
  - Both counted reversals are a move at an event reversed within minutes. That is the
    case Amendment 2 considered and kept as hunting. Not changed; noted for the thesis
    discussion of the hunting rule.
- **07:54: completion on W2 started:** 6 band candidates + the Stage-I point (2 local,
  4 on e5power).
- **08:05–08:21: e5wimi capped at 10 cores** (author: "reduce load at e5wimi to 10 cores
  max", without aborting anything). Nothing was stopped and no code was changed.
  - Method: Windows CPU affinity on the campaign's process trees:
    - driver 29448 and its children (screen 83192, workers 72568, 149404, 151400);
    - follow-up launcher 34968.
  - New child processes inherit the mask, so later local steps stay inside it: W5 with
    2 workers; on GO, D1+D2 with `--d-workers 30`, which then shares 10 processors. The
    settling confirmation and the REF runs run on e5power and are unaffected.
  - Final mask 0x55555 (logical processors 0, 2, …, 18). e5wimi has 20 cores and 40
    logical processors (AMD EPYC 9475F).
    - Masks 30–39, even 20–38 and 10–19 gave the 3 workers only 33–60 % of a core each.
      An unpinned worker reached 99 % at the same time.
    - With 0x55555 all three run at 95–96 %.
    - Likely cause (hypothesis): contention with the other users' 25 unpinned python
      processes and hyper-threading siblings.
  - Effect on the timeline: about 10 min at half speed for the 3 local W2 jobs. The
    estimate stays at ≈ 08:50–09:00.
- **08:52: completion on W2 done.** All 6 band candidates are feasible at level `sts+bank`
  on all five windows (W1–W4, W6).
  - 8ad82442269b ranks first: e_v,TS 0.00799, e_v,DS 0.00571 (go threshold 0.00910;
    v2 winner 0.00827). It makes no bank move in W2.
  - Next: 77930f9b1f53 (0.00798 / 0.00575) and d890586ee0f9 (0.00803 / 0.00573).
  - Settling confirmation of 8ad82442269b on e5power started 08:52.
- **≈ 09:00:** at the author's request, 8ad82442269b was written as the trial set
  `trial_8ad8_v3_run0006` in the shared checkout (`run_multi_system_ofo.TRIAL_PARAMSET`);
  see that checkout's `docs/daily_log/11_2026/2026-11-04_trial_paramset_v3_leader.md`.
- **09:55: settling confirmation PASS, SELECTED 8ad82442269b.**
  - 120 isolated interface steps (W1–W4, W6) on e5power.
  - Pooled n_settle (1 Mvar): median 10.5, p95 23.0, max 46; 1 censored step (DSO_3).
    The 4-period budget (p95 ≤ 36) and v2's 3-period limit (p95 ≤ 27) are both met.
  - Median residual 0.33 Mvar (limit 1).
  - Per DSO p95: DSO_1 22.1, DSO_2 16.5, DSO_3 30.2, DSO_4 14.3. DSO_3 alone is above 27;
    the criterion is pooled.
  - Per window p95: W1 22.4, W2 18.7, W3 26.4, W4 26.9, W6 17.0.
  - Share of steps ≤ 27: 97.5 %; ≤ 36: 99.2 %.
  - W5 report started 09:55 (local, inside the 10-core cap).
- **10:23: run 0006 done, GO** (13.06 h). It wrote
  `configs/paramsets/ch9_selected_uc_2026_11_v3.json` (8ad82442269b).
  - Go checks: e_v,TS 0.00799 ≤ 0.00910; couplers in every search window (4, 6, 16,
    19 moves); banks after events in 2 windows; all limits met.
- **W5 report (held out, not part of the go criteria): 31 counted tap reversals**, against
  0–1 per window in W1–W4 and W6.
  - W5 is 3 Feb 20:00 + 12 h (night; outage 4.8 h, restoration 5.8 h); e_v,TS 0.00612,
    e_v,DS 0.00810; no voltage violations.
  - Coupler activity: 1.33 DSO tap operations per hour, against 0.08–0.33 in the other
    windows. 13 bank operations; z2[0], z2[2], z2[4] and z3[0] switch together at the
    outage (4.8 h), the restoration (5.8 h) and the morning ramp (11.25–11.35 h).
  - Classification of the 31 reversals:

    | first move | dwell < 1 h | dwell 1–2 h |
    |---|---|---|
    | within 10 min of a bank switching | 14 | 8 |
    | no bank switching | 4 | 5 |

    - 0 bank reversals; 5 reversals explained by an event; 0 start-up exemptions.
  - Reading (hypothesis): the bank–coupler coupling under discussion in the parallel
    session dominates W5. 22 of 31 reversals start at a bank switching. The search windows
    (all day-time starts) did not expose it; W5 is the only night window.
- **10:25:04: follow-up launcher 34968 suspended** (NtSuspendProcess), before it started
  anything (its log still shows only "armed").
  - Reason: the W5 result, and the author's statement at 10:1x that a tap hold at bank
    switching is being evaluated and a re-tuning may follow. The author had not yet
    answered whether to hold the follow-ups.
  - SELECTED_PARAMSET unchanged; no REF run and no D1+D2 started.
  - Resume with NtResumeProcess on pid 34968; it then continues as armed.
- **≈ 10:40: author: "Keep paused".** The launcher stays suspended (checked: 89 threads
  suspended).
- **≈ 10:40: first author decisions for the next selection (v4)**, recorded in
  `docs/tuning/2026-11-04_procedure_v4_requirements_draft.md`:
  - hunting tolerance 0 (taps and banks);
  - settling p95 ≤ 27 (3 TS periods);
  - go e_v,TS ≤ 0.01 (absolute);
  - λ ceiling: the author asked "1.6 actually, correct?". The code (`stage1_screen`)
    screens at 1.6 and selects at 1.6 / 1.035 = 1.546, with ε_op 3.5 % for operating
    points outside the screened bank. Open.
  - Run 0006 re-scored at zero tolerance (exact for the evaluated candidates): 11 of 264
    feasible, only ccc91db1de92 with banks after events.
  - Bank price floor at 1.546 / 1.6: 13.5 / 12.7 at 8ad82442269b's weights; 10.9 / 10.4
    at ccc91db1de92's.
- **Author: "Keep 1.546"** (λ ceiling unchanged; recorded in the v4 draft).
- **≈ 11:05: re-measurement of ε_op** (author: "can you remeasure it? … we have to wait
  … anyway").
  - Rule declared in the v4 draft before any result.
  - New script `experiments/ch_09_parameter_selection/ch_9_13_eps_op_remeasure.py`, a
    thesis run with target `ch:param:experiment:d`.
    - It captures 96 fresh operating points (2016; 12 months × days 8, 22 × hours 02, 08,
      14, 20) with the design bank's convention, in a Pool of 10 workers pinned by
      `--affinity-mask`.
    - It computes λ_max per candidate and operating point for all 264 candidates of run
      0006: overall, TS-only, STS-only.
    - It reports s(w) = λ_max(D ∪ F)/λ_max(D) − 1, ε_op,new by the rule, the setting
      conditions and layers, and a resampling of 6-point banks (2000 per candidate,
      seed 20261104).
  - Checks before launch:
    - a fresh capture of W1 reproduces the stored W1 λ to 10 digits; one capture takes
      29 s;
    - a smoke test of the λ part takes ≈ 20 s for 264 × 102.
  - Observation from the smoke test: the STS layer (DSO_4, designed to λ_STS 1.4 on the
    same 6 points) often sets λ_max. The TS-only and STS-only shifts are therefore
    reported separately.
- **11:12: run 0007 launched** (commit 006a2d0). Two problems:
  - **The in-script affinity call failed silently:** `SetProcessAffinityMask` had no
    ctypes argtypes.
  - **Mis-pinning of another session's processes:** while correcting this from outside,
    I pinned the wrong process tree. pid 105740 and its 13 children are the parallel
    session's coupler-hold evaluation (`…_ch_9_13_coupler_hold`, h0/h1/h6), not run 0007.
    The tree was found by a command-line filter that also matched `spawn_main`.
    - That tree ran on mask 0x55555 for about 4–5 min and was restored to 0xFFFFFFFFFF
      at 11:21:27.
    - It was not interrupted, only slowed.
  - **Oversubscription:** the capture workers ran numerical-library threads, up to ≈ 4.6
    cores per worker. Next to the other campaign, 4 of 96 captures finished in 9 min.
- **11:24: run 0007 stopped and resumed** (commit 8d04d51): one thread per worker
  (`OMP/MKL/OPENBLAS/NUMEXPR/NUMBA_NUM_THREADS=1`), affinity with argtypes, `--resume`.
  - The 4 finished captures are kept by the cache.
  - Resumed 11:27 as pid 66120, with 10 workers on 0x55555.
  - Note: the parallel session's processes run unpinned (about 11 workers), so e5wimi's
    total load exceeds the author's 10-core cap. Not touched; reported to the author.
- **11:31: run 0007 done** (93 of 96 captures in 203 s).
  - 3 captures failed: F0408_14, F0622_14, F0822_14, LoadflowNotConverged at the start
    state (summer and spring middays).
  - **Declared rule: ε_op,new = 2.88.**
    - All 264 candidates count as "near the ceiling", because the STS layer sits at 1.4
      ≥ 1.391 for every candidate. The near-ceiling filter does not discriminate; a flaw
      of my rule.
    - λ_max(D ∪ F) is set by F0822_08 (TS_2) for 263 of 264 candidates.
  - **Post-hoc analysis (not part of the rule), `analysis_commitment_split.json`:** the
    large shifts are a commitment-state effect, not sampling noise.
    - In 10 of 93 fresh points the zone-2 generator G3 (nuclear, bus 31, the only machine
      in zone 2) is not committed by the merit order. Checked at F0822_08 against F0108_14
      (`net.gen.in_service`).
    - All six design windows have G3 committed.
    - With G3 off, the voltage curvature of the zone-2 actuators roughly triples (IBG
      1.3e-7 → 5e-7, banks 1.3e-4 → 4.4e-4), and so does λ.
      - TS λ over the 264 candidates: median 3.53, max 5.93; only 16 below 2.
      - 8ad82442269b: 4.16 with G3 off, against 1.40 on D.
    - With G3 committed (83 fresh points), shifts near the ceiling: median 0.3 %, p95 59 %,
      max 106 %. The large ones come from a zone-3 machine (`TS_3:gen6`) not committed:
      8ad82442269b reaches 1.85 at F1022_14 (TS_3).
  - Reading:
    - Established, in the screen's model: in G3-off states the selected weights violate
      the contraction condition λ < 2 of the linearised TS iteration in zone 2.
    - Hypothesis: limit cycles or hunting in zone 2 whenever G3 is off. This would be
      consistent with the Case-2 scenarios GA and GE (both G3 off; GE ended in
      LoadflowNotConverged for 5 of 6 variants with c08b). Not verified by simulation.
  - Consequence for v4 (author decides): ε_op cannot cover this, because it is a gap in
    the coverage of commitment states, not a sampling margin. The screen bank needs the
    commitment states (at least G3 off and the zone-3 machine off), or the weights must
    depend on the commitment state. The latter is an architectural change.
- **≈ 11:40, author:** GA (G3 off) with the trial set 8ad82442269b "does not work well …
  a lot of oscillations". The predicted non-contraction in zone 2 is thus observed in one
  simulation (the author's run, not inspected here).
  - Frequency of G3 off, from the merit-order table of 2026-11-03 (6-h windows): 20 % of
    hourly starts; December 58 %; 8–24 % in the other months; at all hours. Among the 93
    fresh 12-h captures: 11 %.
  - Author's question: exclude the G3-off cases, or re-tune? Options discussed: a robust
    single set (screen bank with the commitment states), commitment-scheduled weights,
    or exclusion as a declared scope limit.
- **≈ 11:45: B1 check** (author: "I would prefer B1. you can do the check … we should not
  investigate in too many new directions"). Screen only, on the enlarged bank (99 points:
  W1–W6 and the 93 fresh points, G3-off included); no simulation.
  - 3 of the 264 run-0006 candidates already pass (λ ≤ 1.546): 779a03b9bda4, 4b9bb14c0313,
    466260e30090.
    - Search-window e_v,TS 0.00857, 0.00831, 0.00955 (leader 8ad82442269b: 0.00869).
    - Counted reversals 3, 2, 1 (truncated at the early stop); no bank commits after events.
    - Common to them: higher g_w_pcc (0.80–1.46) and bank prices (38–49).
  - 8ad82442269b on the enlarged bank: λ 4.16 (F0822_08, TS_2).
    - No single weight suffices, even ×1000.
    - g_w_der, g_w_pcc and g_w_tso_shunt together ×2.72 (bank price 22.8 → 62).
    - θ_TS,min stays 0.0069 pu, so the TS taps stay active.
  - **Bank price floor under B1:**
    - 29.0 with IBG and PCC frozen (the absolute floor, set by F0822_08 TS_2);
    - 33.8 / 39.9 with g_w_der and g_w_pcc at the upper edge of the v3 box;
    - none with g_w_pcc at its Stage-I value;
    - the box edge is 42.7.
  - **Run 0006 evidence:** no candidate with a bank price ≥ 25 commits a bank after an event
    in any window (107 candidates at 25–50). At 20–25: 24 of 61 in ≥ 2 windows.
    - Hence B1 with banks in the λ check is incompatible with "banks commit after events".
  - **Variant with the λ check over the continuous actuators only** (banks excluded; the
    code has this as `central_lambda_rule = "continuous"` for variant M):
    - 8ad82442269b needs only g_w_der and g_w_pcc ×1.45 (0.016 and 0.59); the bank price
      can stay at 22.8.
    - Its λ with banks excluded is 2.24 on the enlarged bank.
  - Options put to the author:
    - B1a: banks in the λ check, and the bank-engagement requirement dropped (reported
      only).
    - B1b: λ over the continuous actuators only, plus a G3-off window among the search
      windows, so that the zero-tolerance hunting filter judges the banks there.
- **≈ 11:50, author: "B1b, banks engage".** Recorded in the v4 draft; no code changed yet.
- **≈ 11:55, author accepts W7:** 8 Oct 2016 20:00 + 12 h, G3 off, with W4's load ramp. The operability check follows.
- **≈ 12:00, author: "can you then just replace the old W4 by the new window?"** Run 0006
  records (264 candidates, truncated at the early stop):

  | window | candidates with ≥ 1 counted reversal | reversals only in this window | bank commit ≤ 2 h after an event |
  |---|---|---|---|
  | W1 (outage, 5 Jan 08:00) | 7 | 0 | 15 |
  | W3 (outage, 4 Jan 08:00) | 19 | 2 | 48 |
  | W4 (load ramp, 26 Nov 10:00) | 54 | 15 | 50 |
  | W6 (load shedding, 18 Jul 14:00) | 229 | 180 | 1 |

  - W4 is the second most discriminating window for hunting, and with W3 the main source
    of the bank-engagement credit.
  - W1 is the most redundant: it has the same event type as W3, one day apart, and decided
    no rejection on its own.
  - Proposal to the author: replace W1 rather than W4. W1 moves to the completion stage
    with W2, so all windows still enter the final ranking.
- Author question (≈ 12:00): why is bank engagement a problem, when at 0.03 pu deviation
  even an expensive bank should switch in?
  - Answer recorded in the v4 draft: the commit threshold is Δv_eng = 0.005 pu ×
    price / 13.49. So at 0.03 pu uniform deviation a bank commits up to a price of about
    80.
  - In the tuning windows the residual deviation after an event stays below about
    0.01 pu: the leader's per-window e_v,TS is 0.004–0.006, W6 0.015. The continuous
    actuators correct first, and without a reserve term nothing rewards a bank for taking
    over.
  - This explains why no bank commits above a price of 25. Reading, not separately
    verified per event.
- W7 operability check (S2, S1, S1e under LOC) launched, `scratchpad/w7_operability.py`.
- **12:37: W7 = 8 Oct 20:00 fails.**
  - S2 passes (V_TS 1.009–1.041 after the ramp).
  - S1 (LOC, no events, 12 h, dt 180 s) and S1e (with the ramp) end in
    LoadflowNotConverged, each after ≈ 19.5 min wall.
  - G3 is the only machine off-line besides the usual.
- **12:38: replacement rule declared in the v4 draft, then all 8 remaining G3-off candidates
  screened in parallel.** Night starts first: 22 Jun 20:00, 22 Oct 02:00. Then day starts:
  22 Jun 08:00, 22 Aug 08:00, 8 Oct 14:00, 22 Oct 08:00, 8 Dec 08:00, 8 Dec 14:00.
  - `scratchpad/w7op_driver.py`, 8 single-threaded processes on mask 0x55555.
- **≈ 12:42, author: "Replace W1".** v4 search windows W3, W4, W6, W7; W1 moves to the completion stage with W2.
- **≈ 12:45, author proposal: G3 off out of the tuning, into an additional analysis**; asks
  whether G3 must then be must-run. Screen on the 89 G3-on points:
  - B1b: 220 of 264 pass, 8ad82442269b 1.532;
  - banks in λ: 178 pass, 8ad82442269b 1.851 (F1022_14, zone-3 machine off).
  - REF G3 status: A–E on; RA, RE, GA, GE off.
  - Recommendation recorded in the v4 draft: scope restriction (MC G3-on filter, REF A–E
    main, RA, RE, GA, GE additional) instead of must-run.
- **≈ 12:50, author: "Yes, scope filter".**
  - v4 search windows back to W1, W3, W4, W6; W7 dropped.
  - Screen bank W1–W6 plus the 89 G3-on points; λ over the continuous actuators.
  - MC filter G3-on (to be implemented); RA, RE, GA, GE as the additional analysis.
  - The running G3-off window screen continues, for the additional analysis only.
- **≈ 12:53, author:** slow reversals (dwell 1–2 h) do not count as hunting. The v4 dwell threshold is therefore 1 h.
  - Launch v4 after the parallel session's tap-hold result, with 30 workers on e5power and 10 on e5wimi.
  - Clock times of my entries between 11:31 and 12:50 were corrected at 12:55: they had been estimated and ran up to 1 h 20 min ahead. The anchors are file times (W7 check finished 12:37:42, the 8-window screen started 12:39).
- **12:55–13:15: procedure v4 built** (code; not launched). Waiting for the tap-hold result.
  - `tuning/stage1_screen.py`: `lambda_max_B` and `screen_weights` take `exclude_classes`.
    - The columns of those classes are dropped from B, after the MSC/MSR group reduction.
    - Default unchanged. For 8ad82442269b on the 99-point bank: 2.2377 without banks, against 2.2378 with the bank weight at 1e7.
  - `tuning/stage2_search.py`: `shunt_summary(..., dwell_s=HUNT_DWELL_S)`. Bank hunting now takes the dwell like the taps; `shunt_reversals` keeps 2 h.
  - `ch_9_3_stage2_screen.py`:
    - `--hunt-dwell-s` (job field `dwell_s`, filter and early stop; recorded as `hunt_dwell_s` in each row);
    - `--lambda-exclude CLASSES` and `--lambda-extra-ops DIR` (points that enter λ only);
    - `--config-set FIELD=VALUE`, ported by hand from f1f4a6c (minimal/thesis-core);
    - all recorded in `run_meta.json`.
  - New driver `ch_9_14_procedure_v4.py`: v3 with `configure()` patches, not applied at import.
    - Tolerance 0, early stop at 1, dwell 3600 s.
    - Settling 3 periods (verification 25); go e_v,TS ≤ 0.01.
    - λ without banks on W1–W6 plus the 83 G3-on points of run 0007 (copied into the run, pushed to e5power).
    - Stage-I verification re-judged at the v4 margin; `--config-set`; 10 local workers pinned to 0x55555; 30 remote.
  - Tests: `tests/test_procedure_v4.py` (5) and v3 (5) pass together; 57 tests of the procedure, shunt and ledger modules pass.
  - Dry run 13:11:
    - Stage-I λ v4 1.405 (v3 rule 1.447);
    - II-a: 176 / 512 draws pass (v3 rule 100);
    - the screen arguments are as intended.
  - Frozen note `docs/tuning/2026-11-04_weight_selection_procedure_v4.md`; the only open item is the tap hold.
  - The hold code is uncommitted in the shared checkout (`configs/config.py`, `controller/base_controller.py`, `experiments/runners/multi_tso_dso.py`; field `sts_coupler_hold_after_bank`). It must be ported here if adopted.
- **13:13: the parallel hold evaluation finished** (run 0014; note `2026-11-04c_sts_coupler_hold_after_bank.md` in the shared checkout).
  - Hold n = 1 and n = 6 both remove the coupler moves at bank instants.
  - II-c candidates: 17.9 goes 2 → 0 reversals, the leader 1 → 0, 14.1 stays at 2 / 3.
  - e_v,TS unchanged.
  - **13:20: hold ported here** (commit 3b4aab0; runner hunk placed by hand); 26 tests pass.
  - Smoke run of 8ad82442269b, W4, 12 min, hold 1 with the v4 options: ok.
  - Launch waits for the author's choice of n.
- **13:2x: G3-off window screen done (additional analysis only).** S2, S1 and S1e under LOC, with W4's ramp:
  - pass: 22 Jun 20:00, 22 Oct 02:00, 8 Oct 14:00, 22 Oct 08:00, 8 Dec 08:00, 8 Dec 14:00;
  - fail: 22 Jun 08:00 (ControllerNotConverged), 22 Aug 08:00 (LoadflowNotConverged);
  - earlier: 8 Oct 20:00 failed.
  - By the declared order, the first passing window is 22 Jun 20:00 (G3 and G9 off). It is a candidate period for the "low synchronous generation" analysis; it is not used in the tuning.
- **≈ 13:30, author: "Hold n = 1".** Entered in the frozen v4 note; v4 launch follows.
- **13:38: run 0008 launched** (`0008_2026-11-04_133847_ch_9_14_procedure_v4`, commit a707786, driver pid 106732).
  - Started detached with stdout and stderr in `scratchpad/v4_driver.out/.err`.
  - Options: `--reuse-stage1` run 0005; `--config-set sts_coupler_hold_after_bank=1`; 10 local workers on mask 0x55555 (checked on the driver and the shard); 30 on e5power.
  - e5power: 19 Python files copied over 3d98d31, hashes equal; the Stage-I paramset, 6 operating points and 83 λ-bank points pushed.
  - Stage-I verification re-judged: PASS at the v4 margin (p95 24.0 ≤ 25, residual 0.37 ≤ 0.8).
    - The logged λ 1.4471 is v2's `describe` value (banks included, 6 points); the v4 λ of the Stage-I point is 1.405.
  - II-a: 176 / 512 draws pass (λ: 314, θ_TS: 297); 34 candidates here, 143 on e5power.
    - The local shard logs `lambda over 89 operating points, without shunt`.
  - Estimated end ≈ 2026-11-05 06:30–08:00.
- **≈ 15:10: run 0008 status, II-a (no code change, no intervention).**
  - Progress: here 60 / 136 windows, e5power 232 / 572; no failed rows.
    - All 11 local processes are on mask 0x55555; e5power has 32 python processes.
    - Projected II-a end ≈ 16:30–17:30, earlier than estimated (early stop at the first reversal).
  - Finished candidates: 79 of 177.
    - 55 were rejected for taps and 22 for banks only.
    - 2 are complete and clean on W1, W3, W4, W6:
      - f988258641a4: sts+bank, bank price 22.0, e_v,TS mean 0.0076, W6 0.0153;
      - b9dcc64d8b49: sts only, price 29.1.
    - Clean candidates run full windows (≈ 60 min) while rejected ones stop early, so the clean share of the finished set is biased low.
  - Counted tap reversals in finished rows: 62.
    - By window: W6 51, W4 9, W1 2.
    - By type: STS coupler 37, TS OLTC 25.
  - **Finding (hold n = 1).** 27 coupler reversals lie within 60 s of a bank move; in 26 of them the coupler moves +20 s after the commit, i.e. at the first STS step after the hold.
    - On this wider candidate set, n = 1 shifts the coupler reaction by one step rather than removing it. In run 0014 (II-c candidates) it removed it.
    - Not acted on: the procedure is frozen. The zero-tolerance filter rejects these candidates, as declared.
  - **Boundary (wording).** `_split_reversals` counts a reversal when its dwell is ≤ `dwell_s` (`(t1 - t0) > dwell_s` skips it); the frozen note says "< 1 h".
    - Three finished reversals have a dwell of exactly 3600 s.
    - Only d5c278b98a71's rejection rests on one of them (W6, its single counted reversal; early-stopped there).
    - v3 used the same ≤ convention at 7200 s. Left to the author.
- **15:20: erratum in the v4 note** (author: "yes, add erratum"). The dwell rule reads ≤ 1 h, as the code applies it. Wording only; no code or run change.
- **≈ 15:25: why W6 is hard (analysis of the II-a rows, 388 windows; read-only).**
  - Profile (`data/profiles.csv`, 18 Jul 14:00 + 12 h):
    - wind WP7/WP10 rises 0.71/0.61 → 0.90/0.91 at 16:00 (+2 h), then falls to 0.40 by 19:00–20:00;
    - PV falls 0.39 → 0 by 19:00;
    - of the six windows, W6 has the largest wind range (0.79/0.81) and the steepest 1-h wind change (0.30).
    - Injection therefore peaks 2 h into the window.
  - Measured (facts):
    - the banks step +1 during 0.5–2.0 h and −1 during 2.0–3.0 h;
    - 45 of 63 counted W6 tap reversals start at 1.7–2.4 h; all 45 start with a +1 move, return within 0.3–1.0 h, and 34 of them have a dwell ≥ 1800 s;
    - 36 of 63 lie within 60 s of a bank move (see the hold finding above);
    - full-length W6 rows: e_v,TS 0.0151–0.0153 across candidates, against 0.004–0.006 in W1, W3 and W4;
    - area-mean error in W6: 0.0033–0.0037; p95 worst bus 0.065 pu; mean excess beyond ±0.02 band 0.025 pu;
    - λ_max(B) without banks in W6 is not elevated on the TS layers (TS_1 lower than elsewhere). STS_DSO_4 is at 1.40, its highest, but DSO_4 has only 2 of the 63 reversals.
  - Interpretation (hypothesis):
    - most W6 reversals follow the turn of the injection profile, not a loop oscillation. A tap that moves just before the 16:00 peak and returns after it has a dwell under 1 h, which the zero-tolerance rule counts.
    - The W6 error level is a spatial spread across buses that holds for all candidates, i.e. a floor of the window, not of the weights.
- **≈ 15:35, author asks whether W6 reversals could be accepted.**
  - Claude advised against a window exception. Of 69 W6 reversals, 37 are bank-coupled; 24 follow the profile turn; 8 are other.
  - Instead Claude proposed a cause-based profile-turn exemption as a fallback. Author: "can you plan this already?"
- **≈ 15:45: plan written**, `docs/tuning/2026-11-04_amendment_profile_turn_draft.md` (DRAFT, not applied; run 0008 unchanged).
  - Rule:
    - turn of the area's residual load (prominence 5 % of the area's IBG capacity within ±1 h);
    - reversal exempt if a turn lies in [t0 − 900 s, t1], no bank move within ±60 s (taps), one per device per turn.
  - Dry evaluation on the II-a rows of 109 candidates (`analysis/profile_turn_dry/`):
    - the rule finds the W6 turn at 2.0 h in every area;
    - taps only: 24 W6 reversals exempt, 24 candidates released (they would need re-simulation);
    - taps and banks: 45 released.
  - Open for the author: D1 banks, D2 trigger (fallback vs adopt), D3 v4.1 full vs re-entry.
  - Implementation must go into a separate worktree while run 0008 is running: later local shards import this checkout's code.
- **≈ 15:50, author: "Wait for run 0008".** The exemption is not implemented now. D1–D3 are decided at the result.
- **≈ 15:55: Claude advised against aborting 0008. Author: "Keep 0008, prep fallback".**
- **16:00–16:45: fallback implemented, switched off by default, NOT merged.**
  - Branch `res/profile-turn-fallback-2026-11-04`, worktree `F:\qOFO_res_pt`.
    - It is separate because run 0008's later local stages import `F:\qOFO_res` code.
  - New: `tuning/profile_turns.py`; `ch_9_15_procedure_v4_1.py` (`--banks yes|no` required); `tests/test_profile_turns.py`.
  - Changed:
    - `tuning/stage2_search.py`: `hunting_summary` / `shunt_summary` take `turns=None`, and None means exactly v4;
    - `ch_9_3_stage2_screen.py`: `--write-profile-turns`, `--profile-turns`, `--profile-turn-banks`; the window configuration moved unchanged into `_window_config`.
  - Tests: 10 new tests pass, including the off-path regression over the stored rows of run 0008; the v3, v4 and hold tests pass.
    - The v2 procedure tests fail when run after the v3/v4 tests, on the unchanged `F:\qOFO_res` code too. This is test-state pollution that predates the change.
  - Smoke launch failures: `import llvmlite` fails intermittently ("[WinError 6] The handle is invalid").
    - It is not caused by stdin, PATH or affinity; foreground runs pass. The same error hit run 0004 on 2 Nov at 11:06, and the driver's shard retry absorbed it.
    - The smoke was finally started via `Win32_Process.Create` with retries; one attempt in three failed.
    - Risk for run 0008: its stage transitions start fresh processes; each shard has 3 attempts.
  - **Smoke 16:20–16:45** (one worker, mask 0x55555): candidate 0e54bf50664d, W6 at 12 h, exemption on.
    - Its first 23 moves are identical to run 0008's row.
    - The 1[0] reversal at the turn (7020 → 8640 s) is exempt, and the window continues.
    - It stops at a later counted coupler reversal (DSO_3|trafo_8, 8280 → 11180 s; the turn is 1080 s before t0, more than L = 900 s). The candidate stays rejected.
  - Commits on the fallback branch: 21b8f0c (code, tests, note), then the smoke note.
- **16:45: run 0008 II-a near its end**: here 134/136, e5power 559/572.
- **17:29: run 0008 II-a done** (local shard exit 0 at 17:08; e5power done 17:28, 572 files pulled).
  - Selection 2a: 177/177 inside the θ bounds; engagement level **sts+bank** (9 candidates); **3 feasible at tolerance 0**.
    - 82c1839cd2c0: e_v,TS 0.00862, e_v,DS 0.00587, bank price 23.0, 7 bank commits.
    - f988258641a4: e_v,TS 0.00883, bank price 22.0, 8 commits.
    - b9cff7864fd0: e_v,TS 0.00919, bank price 13.4, 28 commits.
  - 17:29:55: II-b round 1 started, centred on 82c1839cd2c0: 46/64 draws pass the screen; 8 here, 38 on e5power.
    - The local shard started without the llvmlite failure.
- **19:22: e5wimi core cap lifted to 20** (author: "From now on, you can also use 20 cores on e5wimi again").
  - Host: 40 logical CPUs on 20 physical cores. The new mask is 0x5555555555: one logical CPU on each physical core, the same convention as the 10-core mask 0x55555.
  - Run 0008's own processes were re-pinned from 0x55555 to 0x5555555555: the driver (pid 106732), the round-1 local shard (82420) and its 10 pool workers. Nothing was aborted, and no other session's process was touched.
  - Reason: at 19:19 the host was about 11 % loaded. CPUs 1, 3, 5 and 7 (the HT siblings of the shard's cores) carried other processes, and CPUs 20–39 were idle. With the wider mask the scheduler can move the workers onto idle physical cores.
  - The driver's `--local-workers 10` is fixed in memory. Later stages therefore still run 10 workers here and keep the 10 : 30 split (local speed factor 0.7).
  - 20 workers here would need `--resume <run 0008> --local-workers 20 --affinity-mask 0x5555555555`, best restarted during an assembly step, when no shard is running.
    - Estimated gain: about 15 % shorter split stages (II-b rounds 2–4, II-c, completion), roughly 1–1.5 h. Confirmation, verification and W5 run on e5power only.
    - Not done; the author decides.
  - Round 1 at 19:19: here 30/36 window jobs done.
- **19:50: REF runs queued behind run 0008** (author: "Maybe we can already queue the REF runs with new weights"; later: "some of the GE and GA scenarios didn't converge anyway so they can be excluded").
  - New `experiments/ch_09_parameter_selection/ch_9_16_ref_after_v4.py` (commit a57a04e) and `tests/test_ref_after_v4.py` (4 tests pass).
  - Gate: GO.json, and the held-out W5 report counts no tap and no bank reversal (`--no-w5-gate` drops the W5 condition). Reason: run 0006 was GO with 31 W5 reversals, and the author kept its follow-ups paused.
  - Record step, on the v4 result `ch9_selected_uc_2026_11_v4`, listed under `provenance.record_additions`:
    - the campaign's `--config-set` fields become parameter-set fields (`sts_coupler_hold_after_bank = 1`). Otherwise the REF controllers would run without the hold the weights were selected with (config default 0).
    - `provenance.lambda_max_B` = `provenance.v4.lambda_max_B_v4` (banks excluded, author B1b "as in variant M"). `variants.selected_lambda_max` reads this key for M's loop-gain ceiling; the v3/v4 results record only `design_lambda_max_B`, so M would stop with a KeyError.
  - Then SELECTED_PARAMSET → v4 in `_ch9_selected_design.py`, committed together with the parameter set.
  - REF: scenarios A, B, D, E (GA, GE excluded), variants S1, M, L2, O2, O3, S2 (longest first), 360 min, `--jobs 5` each, so 20 processes here on mask 0x5555555555; e5power is not used.
    - Launched 90 s apart, tag `v4_<scenario>`.
    - A start that dies before its variants finish is completed with `--run-dir` for the missing variants, at most twice.
  - Checks on a stand-in (the v3 result plus a fake v4 block, run 0006 as the campaign):
    - the W5 gate skips (31 reversals);
    - with `--no-w5-gate`, the record additions and the four run_ref commands are as intended;
    - the record step is idempotent;
    - `variants.resolved("M")` carries the hold and the λ ceiling, and the Case-2 pair guard passes.
  - Armed 19:49:42: launcher pid 107764 (scratchpad `launch_ref_queue.py`, which repeats a start that dies within 2 min), queue pid 120700. Logs in `<run 0008>/ref_v4/`.
  - Not queued: the settling figure and D1+D2.
  - The suspended v3 follow-up launcher (pid 34968) would switch SELECTED_PARAMSET to v3 if it were resumed.
- **19:53: run 0008 II-b round 1 done** (here exit 0 at 19:37; e5power done 19:48, pulled 19:52).
  - Selection 2b_round1: 223/223 inside the θ bounds; engagement level sts+bank (20 candidates); 6 feasible at tolerance 0; ranked first 82c1839cd2c0, b3ca74277466, 0eccdcf9916b, c8c2f9f2cf70.
  - The best is unchanged (82c1839cd2c0), so f → 0.25.
  - Round 2 started 19:53, centred on 82c1839cd2c0: 64/64 draws pass the screen; 12 here, 52 on e5power. That is 65 candidates, against 47 in round 1.
  - Stop rule: if round 2 also leaves the best unchanged, f becomes 0.125 and round 3 runs. If round 3 leaves it unchanged as well, f becomes 0.0625 < F_MIN and II-b ends after round 3.
- **20:16: queue revised** (author: "Yes. It should wait for W5, and do D1 and D2 before. And after 8 in the morning, please use only 10 workers on e5wimi again"). Commits ed09b29 and fafb0fb; 7 tests pass.
  - Order: GO → W5 clean → record and switch → D1 + D2 (`ch_9_4_deadband`, 10 workers) → width gate → REF.
  - Width gate: REF starts only if D1's design width equals the dead band the REF variants run with (0.01 pu TSO and DSO, read from `variants.resolved`). Otherwise the queue stops for the author.
  - REF runs 10 processes in all (`--jobs` 2, 2, 3, 3 for A, B, D, E). The wall time is still set by E's S1 (about 6.6 h).
  - The queue pins itself to 0x55555.
  - `repin_0800.ps1` (pid 112520, scratchpad) re-pins run 0008's tree and the queue launcher's tree to 0x55555 at 2026-11-05 08:00. A dry run found run 0008's 12 Python processes.
    - A root counts only if its command line is the expected one, and a descendant only if it was created after its parent, so a reused pid is skipped.
  - Smoke of D1/D2 on the commitment base (`0009_2026-11-04_195947_ch_9_4_deadband_smoke`; old weights a1, W1 only, 0.25 h): complete, no failed run, the D2 event reached the plant.
    - Runs with the droop active take about 5–6 s per 10-s step (D1 confirmation 505 s, D2 legs 553–567 s; suppressed 86 s). The full D1 + D2 therefore needs about 6 h on 10 workers (estimate).
    - The smoke's design width is 0.02 pu (rule 0.015 pu), not 0.01 pu. It is only indicative (start-up-dominated 0.25 h, old weights), but the width gate may stop REF.
    - The smoke took counter 0009 of `results/ch_09_parameter_selection`.
  - The llvmlite start failure hit the queue's own start 3 times in 4 (20:12–20:16).
    - Start failures (exit within 120 s) are now repeated up to 12 times, in the queue and the launcher.
    - Armed 20:16:21 as queue pid 53396 (launcher 9312, `ref_v4/launcher.pid`).
- **21:28: D1 + D2 moved to e5power** (author: "Yes, put d1 and d2 there. In e5power"). Commit 479035a; 9 tests pass; the dry run on the stand-in gives the commands.
  - After the switch commit, the v4 parameter set is pushed to e5power, and `ch_9_11_procedure_v3.sync_remote_code` copies this checkout's code there and checks the hashes.
  - `ch_9_4_deadband` then runs there with 30 workers, through a detached shard (`run_remote`, logs in `ref_v4/_remote/`).
  - Its run folder is pulled back into `results/ch_09_parameter_selection/`, with `remote_host.json` (host, remote commit, local commit) recorded in its README.
  - Numbering: thesis_runs numbers a run max + 1 over the folders on its own host.
    - If e5power's numbers are lower than this host's, e5power first gets an empty folder named after this host's highest run, so the pulled run keeps its number here.
    - Today e5power is ahead: it holds older folders 0009 and 0010 (`ch_9_5_avr_revalidation`, 2026-10-30/31) that are not here. So the remote runs get 0011 and 0012, and 0010 stays unused here.
  - Retries: a start that leaves no run folder is repeated (up to 12 times); a folder without `d1/selection.json` is completed with `--reduce` at most twice.
  - REF stays on e5wimi with 10 processes.
  - Read-only hash check before the change: e5power's code equals this checkout, except for files the remote runs do not import (`ch_9_16_ref_after_v4.py`, `analysis/profile_turn_dry/*`). So no copy was needed for the smoke.
  - Smoke on e5power (`--remote-smoke`; W1, 0.25 h, 5 workers, the parameter set selected there, a1): `0011_2026-11-04_212919_ch_9_4_deadband_smoke`, pulled back here at 21:49 (47 files), status complete.
    - First start (21:29): one worker died abruptly (W1 suppressed, right after the Gurobi licence line), so the pool broke (`BrokenProcessPool`), rc 1.
      - Nothing in the Windows event log; RAM was free (140 of 191 GB).
      - No pool broke in six days of logs on either host before this.
    - The `--reduce` restart (21:36) kept the finished dt check and completed the rest by 21:47, rc 0. The manifest records both invocations (failed, complete).
    - Result equal to the local smoke 0009 (same code and weights): design width 0.02 pu, rule 0.015 pu, d_max agreeing to about 1e-12; D2 event reached in all three legs.
    - Wall times on e5power, under run 0008's load: droop active 425–468 s, suppressed 85–90 s. That is about 15 % faster than e5wimi.
    - So the full D1 + D2 with 30 workers takes about 2–2.5 h (estimate).
    - `remote_host.json` first said the code was copied; the smoke copied nothing. Corrected by hand at 21:51, and the code now says which applies.
  - Commit 2c303f8: D1 + D2 is completed with `--reduce` up to 4 times (`D_RETRIES`; the full run has 91 runs, unattended).
  - Re-armed 21:53:24: queue pid 69232, launcher 148152, both on mask 0x55555. One quick start failure first.
    - The queue's first remoting call works from this detached process ("e5power at 3d98d31").
    - The old queue (53396) and launcher (9312) were stopped first; both were only waiting for GO.
  - Run 0008 at 21:54, round 2: 31/52 units here and 95/208 on e5power. Round 2 therefore ends about 00:15, later than the 23:10 estimated at 20:16.
- **22:47: v3 follow-up launcher ended** (author: "Yes", to "Should I end the suspended v3 follow-up launcher (pid 34968)?").
  - What it was: `ch_9_12_followups_v3.py --run <0006> --switch-selected`, armed 2026-11-03 21:21:54, suspended since (all 89 threads), with no child besides its console host.
  - It had done nothing beyond arming: `SELECTED_PARAMSET` is still `ch9_selected_band_2026_10_a1`, and no follow-up run was started.
  - Note left in `<0006>/followups/STOPPED.txt`.
  - Run 0008 (106732), the queue (69232, launcher 148152) and the re-pin (112520) are running, untouched.
- **2026-11-05 00:05: run 0008 II-b round 2 done** (4 h 11 min; here exit 0 at 23:43; e5power done 00:01, pulled 00:04).
  - Selection 2b_round2: 287/287 inside the θ bounds; engagement level sts+bank (48 candidates); 22 feasible at tolerance 0; ranked first ca2eda42e753, 4d9eabd5bb55, 6cf395038c79, d88cce6949d1.
  - New best ca2eda42e753, so II-b recentres and f stays 0.25.
    - Against the round-1 best 82c1839cd2c0, now ranked 5th: e_v_ts 0.00864 against 0.00862, inside the 0.02 indifference; e_v_ds 0.00577 against 0.00587.
    - Its worst TSO tap rate is 18.0 ops/day, against 10.0 for 82c1839cd2c0. Hunting and V violations are 0 for both.
  - Round 3 started 00:05, centred on ca2eda42e753: 45/64 draws pass the screen; 8 here, 37 on e5power (shard pid 11368). Round 1 had the same size and took 2 h 23 min, so round 3 should end about 02:30.
  - Stop rule (F_MIN 0.125, MAX_ROUNDS 4): with f at 0.25, round 3 cannot end II-b. An unchanged best gives f = 0.125, which is not below F_MIN. So round 4 runs in either case, and the 4-round ETA applies: GO/NOGO about 10:00–10:30. This supersedes the 3-round branch noted at 19:53.
- **02:57: run 0008 II-b round 3 done** (2 h 53 min, not the 2 h 23 min estimated; here exit 0 at 01:43; e5power done 02:54, pulled 02:56).
  - Selection 2b_round3: 332/332 inside the θ bounds; engagement level sts+bank (63 candidates); 28 feasible at tolerance 0; ranked first 061dadc05755, ca2eda42e753, 8f8e0d8fadf4, 4d9eabd5bb55.
  - New best 061dadc05755: e_v_ts 0.00872, e_v_ds 0.00556, e_q 1.45e5, worst taps 18.0/day TSO and 8.0/day DSO, 4 bank ops/day. Hunting and V violations are 0.
    - Against the round-2 best ca2eda42e753: e_v_ts 0.00872 against 0.00864, inside the 0.02 indifference; e_v_ds 0.00556 against 0.00577; e_q 1.45e5 against 1.91e5.
    - Observation: e_v_ts of the incumbent has risen each round (0.00862, 0.00864, 0.00872), while e_v_ds falls. The lowest e_v_ts in the union is 0.00861 (4d9eabd5bb55), so the new best is 1.3 % above it, still inside the indifference.
  - Round 4 (the last, MAX_ROUNDS 4) started 02:57, centred on 061dadc05755: 47/64 draws pass the screen; 9 here, 38 on e5power (shard pid 19492). By round 3's rate it ends about 05:55, and II-b ends with it. GO/NOGO about 10:00–10:30.
- **04:28: run 0008 round 4, e5power part done early** (rc 0 at 04:25, 1 h 28 min against 2 h 49 min in round 3; 152 files pulled). Local shard at 35/40 units.
  - Cause: early aborts, not failures. Round 4 has 174/191 windows with `early_abort`, against 120/184 in round 3; `failed` is 0 in both.
  - Only 3 of 48 round-4 candidates are clean in all four windows (59b9c16c9bc9, 772dfdc0707a, d6aa9ec0ad10), against 15 of 46 in round 3.
  - Each of the 44 aborted candidates has at least one window with a reversal. Of the reversing windows: 45 have TSO and DSO tap reversals, 9 TSO only, 7 DSO only, 4 TSO + shunt, 3 shunt only. The other aborted windows show no reversal.
  - Hypothesis (not tested): the incumbent 061dadc05755 lies near the edge of the hunting-free region in weight space; most draws within f = 0.25 of it hunt. That makes the held-out W5 gate more likely to fail.
- **04:35: run 0008 II-b done after 4 rounds; II-c started.** Local round-4 shard exit 0 at 04:34.
  - Selection 2b_round4: 379/379 inside the θ bounds; engagement level sts+bank (74 candidates); 31 feasible at tolerance 0; ranked first 061dadc05755, ca2eda42e753, 8f8e0d8fadf4, 772dfdc0707a.
  - Best unchanged, so f → 0.125; MAX_ROUNDS reached. II-b leader 061dadc05755:
    - g_v 4.0e4, g_q 0.4, dso_g_v 1.0e4, g_w_der 0.014, g_w_pcc 0.873, g_w_gen 1.70e6, g_w_tso_oltc 15.61, g_w_tso_shunt 19.02, g_w_dso_der 1.50, g_w_dso_oltc 10.84.
    - Aggregate: e_v_ts 0.00872, e_v_ds 0.00556, e_q 1.45e5, worst taps 18.0/day TSO and 8.0/day DSO, 4 bank ops/day.
  - II-c (bank price on its own, W1/W3/W4/W6 without the early stop): g_w_tso_shunt grid 4.27–15.36 (7 points), 7/7 pass the screen; 1 here, 6 on e5power (pid 27272).
- **05:39: run 0008 II-c done; the leader is kept.** Here exit 0 at 05:31; e5power done 05:35, pulled 05:39.
  - All 7 lower bank prices are rejected for hunting. The leader is the only feasible point (selection 2c: 1/9 feasible at tolerance 0).

    | g_w_tso_shunt | tap hunting | bank hunting | bank after events | e_v_ts |
    |---|---|---|---|---|
    | 4.27 | 9 | 15 | 4 windows | 0.00870 |
    | 5.28 | 14 | 9 | 4 | 0.00869 |
    | 6.54 | 10 | 4 | 4 | 0.00869 |
    | 8.10 | 6 | 1 | 4 | 0.00868 |
    | 10.02 | 7 | 0 | 4 | 0.00869 |
    | 12.41 | 8 | 0 | 4 | 0.00869 |
    | 15.36 | 2 | 0 | 3 | 0.00869 |
    | **19.02 (leader)** | 0 | 0 | 2 | 0.00872 |

  - Observation: bank hunting falls to 0 at a price of about 10. Tap hunting persists up to 15.36, i.e. 19 % below the leader's price. Consistent with the round-4 hypothesis that the leader sits at the edge of the hunting-free region; here along the bank-price axis.
  - e_v_ts hardly changes across the grid (0.00868–0.00872), so the price is set by hunting, not by voltage error.
  - Stage 5 started 05:39: completion of the band on W2 (27 band candidates and the Stage-I point); 5 here, 22 on e5power (pid 17764).
- **07:08: Stage 5 still running on e5power.** Here exit 0 at 06:10 (6 units, 0.51 h). e5power at 60/110 units.
  - e5power simulates all five windows (W1–W4, W6) for its 22 candidates, i.e. 110 units, not only the 22 W2 windows. Here only the missing windows ran (6 units).
  - Cause (reading of `simulate()`, not tested): the stored windows are copied into `complete/windows` on this host only; the remote shard does not get them. The pull uses `overwrite=False`, so the copies here are kept and the duplicate windows are discarded. Results are therefore unaffected; only time is lost.
  - The same happened in run 0006 (v3), but there 4 candidates × 5 windows fit in one wave (56 min). Here it costs about 2 h.
  - Not changed mid-run. ETA: e5power done about 08:20–08:30; settling confirmation (about 1 h in v3), W5 (about 0.5 h), GO/NOGO about 10:00–10:30.
- **07:25, author: "Can you already do the settling check for the leader on e5wimi or is it only clear after w2 on e5power?"** Answer: not worth it; nothing changed.
  - Stage 6 checks the first of the five-window ranking, so formally only after W2.
  - A provisional ranking on 21/27 candidates with W2 kept the leader first. The 6 missing ones could only displace it with W2 e_v_ts ≤ 0.00372; the observed range was 0.00404–0.00422.
  - `V2.confirm` would reuse a result found in `expC/criterion_results.json`. But on e5wimi (10 workers from 08:00) the 120 steps take about 3 h, i.e. later than e5power from about 08:25. Unfinished, it would also collide with the driver's pull into `expC/cand_<key>`.
- **08:00:11: re-pin done** — 3 Python processes (driver 106732, queue 69232, launcher 148152) on 0x55555; the re-pin script (112520) has exited.
- **08:33: run 0008 Stage 5 done; settling confirmation started.** e5power done 08:28 (110 units), pulled 08:32.
  - Selection complete: 28/28 inside the θ bounds; engagement level sts+bank (27 candidates); 27 feasible at tolerance 0; ranked first 061dadc05755, 8f8e0d8fadf4, ca2eda42e753, d88cce6949d1. Same order as the provisional ranking at 07:25.
  - Leader on five windows: e_v_ts 0.008023, e_v_ds 0.005457. The lowest e_v_ts is 0.007873, so the indifference limit is 0.008030: the leader is inside the band by 0.09 %.
  - Settling confirmation of 061dadc05755 on e5power (30 workers, pid 25240). By v3's time (63 min) it ends about 09:40; W5 and GO/NOGO about 10:10.
- **09:45: settling confirmation PASS; SELECTED 061dadc05755.** e5power done 09:43 (1 h 10 min), pulled 09:45.
  - Criterion (`expC/criterion_results.json`): p95 23.05 ≤ 27 over 120 steps; 3 steps over the limit; max 32; censored 0; median residual 0.344 Mvar.
  - The ninner pooled statistics: median 12, p95 20.0. The criterion's p95 is computed differently (23.05); the criterion decides.
  - Per DSO interface p95: DSO_1 24.3, DSO_2 16.0, DSO_3 30.2, DSO_4 13.75. DSO_3 alone is above 27. The criterion is pooled by design, so this does not fail it.
  - Stage 7 started 09:45: W5 for the report (2 workers here, child pid 100760), then the result parameter set and the go criteria. About 30 min.
- **10:00: trial parameter set for 061dadc05755 in the shared checkout** (author: "can you already set 061dadc05755 as a trial paramset such i can run it in run_multi_system_ofo").
  - New, untracked: `Z:/Python_Projekte/qOFO/configs/paramsets/trial_061d_v4_run0008.json`. It holds the 10 weights at full precision from `selection_complete_sts+bank.json` (ranked[0], equal to `expC/cand_061dadc05755_weights.json`) plus `sts_coupler_hold_after_bank: 1`, the v4 config set. Its form follows `trial_8ad8_v3_run0006.json`.
  - `Z:/.../experiments/run_multi_system_ofo.py`: `TRIAL_PARAMSET` set to `"trial_061d_v4_run0008"` (was `"trial_8ad8_v3_run0006"`), plus one comment entry. Nothing committed there; the file already had the author's uncommitted changes, which are untouched.
  - Its name differs from the driver's output `ch9_selected_uc_2026_11_v4`, which the driver refuses to overwrite.
  - Check: `apply_paramset(make_config(), TRIAL_PARAMSET)` loads in Z. With `main()`'s frozen network and `dso_gamma_oltc_q = 0`, the config equals the settling run's `resolved_config.json` in all 226 fields, except output settings (verbose, live plots, result_dir) and `sts_coupler_hold_after_bank`.
  - Finding: the v4 settling check ran with `sts_coupler_hold_after_bank = 0`. `CONFIG_SET` is applied only in `screen_argv` (12-h screen windows), not in `settle_argv`. Whether the flag matters for isolated interface steps is open (it acts only after a bank switch).
- **10:09: run 0008 done, GO; selected 061dadc05755** (20.51 h). The driver wrote `configs/paramsets/ch9_selected_uc_2026_11_v4.json`.
  - Go checks all true: e_v_ts 0.00802 ≤ 0.01; STS couplers move in every search window (W1 4, W3 5, W4 10, W6 16); banks after events in 2 windows; all limits met.
  - **W5 report (held out): 16 tap and 3 bank reversals counted** (`n_hunting` 16, `n_shunt_hunting` 3). Also e_v_ts 0.00617, e_v_ds 0.00616, no V violation, 23 bank commits (12 up, 11 down), 84 tap moves. v3 (8ad8) had 31 tap and 0 bank reversals there.
  - Structure: bank z2[0] switches in and out three times — 4140/6480 s, 14220/16740 s, 40140/43020 s; dwell 2340–2880 s, under the 1-h hunting dwell. These are the 3 bank reversals.
  - 14 of the 16 tap reversals lie inside these three in/out cycles; they start up to about 13 min before or after the bank comes in. They are on the STS couplers of all four DSOs together (DSO_1 trafo_2, DSO_3 trafo_8, DSO_4 trafo_11 in each cycle; DSO_2 trafo_4/5, DSO_4 trafo_10 and TS trafo 3[1] in the third). The other 2 (DSO_1 trafo_1/trafo_2) start at 17280 s, an event time with a bank switch (z2[2], z2[5], z3[0]).
  - Hypothesis (not tested): in W5 the IBG reactive capability is zero (VDE dead band), so the TS controller has only banks and taps. The z2[0] in/out cycle and the common-mode coupler reversals are one oscillation. `sts_coupler_hold_after_bank = 1` (one period, 180 s) is much shorter than the 2300–2900 s cycle.
  - The queue's W5 gate (0 tap and 0 bank reversals) therefore fails: D1 + D2 and REF do not start, and the record additions and the switch of `SELECTED_PARAMSET` are not made. The author decides.

## Open
  - Settling tests only for variants that pass the frozen filter, on e5power.
  - `ch_9_10` change for this run:
    - jobs are window-major in the order given (`--windows W6,W3,W4,W2,W1`), so the pool
      does not end on a lone W6;
    - the row also carries n_viol_v_ts and the tap reversal rates.
    - The e5power record run (launched 13:42) uses the earlier file; its rows lack these
      three fields.

## Open

- On the commitment base, g_q is squeezed from both sides (superseded at 19:45: hunting is
  set by g_w_tso_oltc, not g_q; the conflict is e_v,DS against Q tracking): settling needs it larger, the
  zero-tolerance hunting filter needs it smaller.
- Next diagnostic step proposed to the author: where and when the high-g_q candidates hunt
  (start-up at the first dispatch, or a genuine TS–STS interaction).
- If the driver is resumed:
  - e5power must first be synced to the local HEAD;
  - batch 2 would be re-simulated unless its verdicts are entered;
  - `pull_dir` should log the full stderr.
- Any amendment now would be declared after the results are known.
