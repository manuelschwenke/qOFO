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
