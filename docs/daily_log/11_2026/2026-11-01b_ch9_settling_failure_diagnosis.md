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
