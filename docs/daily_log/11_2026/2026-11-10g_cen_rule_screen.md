# 2026-11-10g -- CEN (M): weight rule agreed with the author, R2 screen, gamma selection

Session cen-tuning-2 (side session of "REF R2/R3 results analysis"). Branch **`res/cen-rule-2026-11-10`**, worktree
`F:\qOFO_cen`, on the common rerun base `res/rerun-base-2026-11-10` (5c2d8b0). Screens, not runs of record.

## 1. Author decisions (2026-11-10, in this order)

- Starting point (author): "slightly manually tune the discrete weights to have an only 'slightly' tuned central
  reference that beats all other variants, and state that in the dissertation."
- Discussed: discrete weights act on switching, not on the e_v gap. Established on R2 (27f0651 records, 2026-11-10d
  section 6): Mblk's TS term of CEN's own objective is 150.2 vs O3's 136.2 (ratio 1.10), and (10.60/10.06)^2 = 1.11,
  i.e. the e_v gap is the TS share of CEN's objective; CEN trades it for STS voltage quality (39.6 vs 75.1) and beats
  O3 on its own objective by 10 %. The lever for e_v is the TS : STS weight ratio.
- Author: "should we use C2, and maybe manually tune (C3) a little to have less weight on the STS voltages".
- Author: "beats all" means **e_v only**.
- Author approved the rule: gamma on the STS weight -> block-wise loop gain -> C2 on the MT taps and banks; grid
  gamma in {1, 0.5, 0.25}; selection = the largest gamma at which M's e_v on R2 lies below every other variant's.
- C2 owner curvature (after the mask finding of section 2): **definition B** (owner's zone rows on M's full model;
  no C1). O3 comparison rerun by this session on the same code.
- No R3 screen: the reruns of record test the frozen gamma on R3 / R3b out of sample.
- **"we should have a uniform 1.452 now"**: M's cap is the lambda_max(B) of the selected weights at the 1.00 pu
  start of the rerun base (gate 2026-11-10f), not the selection-time 1.530; the paramset file stays unchanged.

## 2. Findings before the screen (init only, no loop; E2, new base, 6-h commitment window)

Scripts: scratchpad `diag_masks.py`, `diag_gamma.py` (copied below as needed); output quoted here.

- **Why the cascade never moves trafo 2 and 6.** The zone-local TS model promotes the zone's largest in-service
  machine to the local slack; its machine-transformer tap then has no sensitivity column and the runner flags it
  out of service on the TS-OFO (`sensitivity/network_reduction.py` step 9, `multi_tso_dso.py`
  `_apply_promoted_oos_oltc`). Zone 2: G3 (gen 1, trafo 2); zone 3: G4 (gen 2, trafo 6). M's full model has both
  columns. This is a limitation of the cascade's reduced model, not an outage, so C1 (hold them in M) would hand the
  reference the cascade's model error; the author chose definition B instead.
- **Correction of 2026-11-10d section 6.** `a14` ran a 10-min init whose unit commitment covers only 10 min: G3 is then
  decommitted (the E2 G3 trip/restore are void), M's raw continuous lambda is 4.01 and kappa 2.62 on the new base
  (~2.5 on 27f0651). The curvature ratios there and "trafo 2: generator out of service at the start" are therefore
  not the as-run state. With `unit_commitment_window_s=21600` (as in the 6-h run): only G9 is off-line, raw lambda
  2.343, kappa 1.531 (27f0651: 2.22 / 1.451). Diagnostics of M at the run start need the 6-h window.
- Per-switch curvature ratio c_CEN / c_owner at the as-run start (owner's own zone-local model): tap 0 1.89, taps 3/4
  0.76 / 0.73, tap 11 1.10, banks 1.56-3.08; trafo 2 and 6 infinite (owner curvature 0).
- C2 factor g_w,M / g_w,owner under definition B, and the block rule (TS block -> 1.530, STS -> 1.40, common factor):

| gamma | raw lambda TS / STS / joint | block g_w factor TS / STS | tap 0 | tap 2 | taps 3/4 | tap 6 | tap 11 | banks |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.044 / 0.426 / 2.343 | 2.321 / 0.528 | 1.85 | 1.99 | 1.35 / 1.32 | 1.76 | 1.46 | 2.39-3.25 |
| 0.5 | 1.866 / 0.293 / 2.072 | 2.130 / 0.365 | 1.74 | 1.61 | 1.17 / 1.16 | 1.60 | 1.31 | 1.78-2.35 |
| 0.25 | 1.780 / 0.228 / 1.944 | 2.047 / 0.287 | 1.68 | 1.42 | 1.09 / 1.08 | 1.52 | 1.23 | 1.48-1.91 |

  Under B every switch costs M at least as much as its owner (M's rows contain the owner's). Previously (kappa
  scope "all") every tap and bank had factor kappa = 1.531.

## 3. Code (branch res/cen-rule-2026-11-10)

| Commit | File | Change |
|---|---|---|
| 0332a82 (= 3826639 before the rebase onto 5c2d8b0) | `configs/config.py` | `central_kappa_scope` gains `"blocks"`; new `central_lambda_target_sts`, `central_switch_g_w_rule` (`"kappa"` / `"owner"`) |
| | `experiments/runners/multi_tso_dso.py` | `_central_column_groups`, `_block_lambda`, `_central_block_g_w` (block rule; taps, banks, couplers x kappa_all = max(lambda_joint / target, 1)), `_central_owner_switch_g_w` (C2-B: g_w,j = g_w,own,j * h_j / h_own,j, h over M's rows / the owner zone's TS rows on M's full H; h_own = 0 keeps g_w); kappa block: scope "blocks" always applied, owner rule after it; printed `[central] blocks: ...` and `[central] switch g_w from the owner's curvature ...` |
| | `experiments/ch_11_case2_multisystem/variants.py` | M: `central_kappa_scope="blocks"`, `central_switch_g_w_rule="owner"`, `central_lambda_target_sts = selected_lambda_max_sts()` (provenance `design_lambda_max_B_sts` 1.40), `central_dso_g_v = M_STS_G_V_FACTOR * dso_g_v`; O3 -> M step allows the three new fields; module docstring |
| a45e22a | `variants.py` | `LAMBDA_MAX_B_GATED = {"ch9_selected_nr_2026_11_v5": 1.4523628862957012}`; `selected_lambda_max()` returns it (raises for any other paramset) |
| | `tests/experiments/test_case2_variants.py`, `tests/test_central_cen_rule.py` | M's rule fields, gated cap vs provenance 1.5301, block rule (targets, common factor, kappa_all on discrete), owner rule (ratio >= 1, h_own = 0) |

Checks: init of M on E2 reproduces the diagnostic exactly (TS x2.321, STS x0.5283, common 1.737; C2 factors as in the
table; joint lambda 1.530 at the 1.530 cap); `assert_case2_controlled` passes; 19 tests pass
(`tests/experiments/test_case2_variants.py`, `tests/test_central_cen_rule.py`, `tests/test_central_updater_wrapper.py`).

## 4. Screens

Driver `F:\qOFO_screens\2026-11-10_cen_rule_gamma\cen_rule_screen.py` (README there; gamma via `central_dso_g_v`, no
hook), scenario E2, paramset ch9_selected_nr_2026_11_v5, noise seed 20261027.
- **S1, e5wimi, 15:43**, code 3826639 (cap 1.530, the paramset provenance): M_g100, M_g050, M_g025, O3; 4 workers
  pinned to CPUs 32/34/36/38 from 15:48 (core split with the main session).
- **S2, e5power, 16:07**, code a45e22a (cap 1.4524, the final rule), bundle clone `F:\qOFO_runs\qOFO_cen`, driver
  `cen_rule_screen_e5power.py` (copy in the screen folder), output `F:\qOFO_runs\qOFO_cen_screen`: M145_g100,
  M145_g050, M145_g025; pinned to CPUs 32-35. O3 does not depend on M's cap, so S1's O3 is the comparison.
- gamma is selected on S2 (the final rule); S1 is the 5 % loop-gain sensitivity.
- Evaluation: `2026-11-10g_a15_cen_rule_eval.py` (run_ref.report, i.e. Table 10.5 metrics; stage means; objective
  split; sign reversals; switching per device). Dry run on the 27f0651 records reproduces 2026-11-10d (objective
  211.2 / 229.1 / 189.8, IBG reversals 7 % / 8 %).

## 5. Results (S2, the final rule; evaluated 17:55)

S2 finished 17:4x on e5power (all ok, pulled by `pull_e5power.ps1`); O3 of S1 finished 16:38 (ok, 3231 s). The S1 M
runs at the 1.530 cap ran at ~13 s per step on e5wimi (contention) and were still running at 17:52 (t = 162 / 234 /
306 min); they are the loop-gain sensitivity and do not enter the selection.

R2, new base, Table 10.5 metrics (`run_ref.report`; mpu unless noted):

| R2 | e_v | e_vp | V_min,TS | t_out | U_Q^STS | f_V^STS | V_min,STS | AVR travel pu | MT taps | banks | coupler/dev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| O3 (new base) | 9.84 | 5.83 | 0.922 | 0.7 % | 0.615 | 14.3 | 0.932 | 0.13 | 10 | 8 | 6.25 |
| M, gamma 1 | 11.23 | 7.26 | 0.964 | 16.9 % | 0.536 | 10.2 | 0.991 | 0.38 | 26 | 7 | 0.83 |
| M, gamma 0.5 | 10.26 | 5.64 | 0.965 | 10.5 % | 0.554 | 13.2 | 0.991 | 0.36 | 26 | 13 | 0.17 |
| **M, gamma 0.25** | **9.37** | **4.37** | **0.969** | **0.6 %** | 0.589 | 17.4 | 0.988 | 0.28 | 17 | 6 | 0.00 |
| for reference, 27f0651: O3 / M / Mblk | 10.06 / 11.31 / 10.60 | 6.01 / 6.30 / 5.51 | 0.922 / 0.957 / 0.961 | 6.0 / 16.5 / 4.3 % | | 13.7 / 12.2 / 10.0 | | | 10 / 47 / 57 | 7 / 29 / 32 | |

Selection (rule fixed before the screen): M's e_v is below O3's only at gamma = 0.25 (9.37 vs 9.84, margin 0.47 mpu
= 4.8 %), so **M_STS_G_V_FACTOR = 0.25** (`central_dso_g_v` = 2500). On the 27f0651 records O3 was the best other
variant by 2.5 mpu (S2 12.55); the other variants are not yet rerun on the new base, so "below every other variant"
is checked against O3 only.

Stage means of e_v (stages 0-30.7 / -91.7 / -152.3 / -241 / -360 min) and per-area e_v:
- O3: 6.58 / 5.35 / 13.52 / 8.55 / 9.68; areas 6.24 / 14.64 / 6.09.
- M gamma 0.25: 5.02 / 5.27 / 10.98 / 9.09 / 10.98; areas 6.91 / 13.50 / 5.76.
- M leads in stages 1-3 (load step: -2.5 mpu) and in areas 2 and 3; it trails in stages 4 (G3 back) and 5 (line 3-4
  out) and in area 1.

CEN's objective at the STS weight 1e4 (time-mean weighted squared deviation, approximation of 2026-11-10d):
O3 130.2 + 82.1 = 212.2; gamma 1 167.2 + 41.6 = 208.8; gamma 0.5 139.6 + 69.4 = 209.0; gamma 0.25 116.1 + 121.4 =
237.5. At its own weights (STS term / 4) gamma 0.25 gives 146.5 vs O3's 150.7. With gamma 0.25, M buys 11 % lower TS
deviation with 48 % higher STS deviation than O3; V_min,STS stays above O3's (0.988 vs 0.932).

Period-2 check (share of sign reversals of successive increments): V_gen 8 / 10 / 10 % (gamma 1 / 0.5 / 0.25; O3
15 %), Q_gen 18 / 19 / 21 % (O3 29 %), STS IBG Q 16 / 14 / 11 % (O3 10 %). No period-2 cycle.

Switching per device (moves / reversals; median gap to the next move of the same device):
- O3: taps 10 / 4 (34.5 min), banks 8 / 1 (60 min); tap 0 3/1, taps 3, 4 2/1 each, tap 11 3/1.
- M gamma 0.25: taps 17 / 6 (60.7 min), banks 6 / 0 (61 min); tap 0 4/1, tap 2 3/1, taps 3, 4 2/1 each, tap 6 3/1,
  tap 11 3/1.
- M gamma 1 / 0.5: taps 26 / 8 each (60.7 min), banks 7 / 0 and 13 / 3.
- Against M on 27f0651 (47 / 15 taps, 29 / 14 banks, lock-limited at 15.3 min), the owner weights (C2) cut M's tap
  moves by 45-64 % and its bank operations by 55-79 %; M's discrete devices are no longer lock-limited. Of M's 7
  extra tap moves over O3 (gamma 0.25), 6 are on trafo 2 and 6 (3 each), the taps the cascade's zone models cannot
  move, and 1 on tap 0.
- M does not move a coupler at gamma 0.25 (0.83 / 0.17 per device at gamma 1 / 0.5; O3 6.25): the couplers keep
  kappa_all and their curvature falls with gamma.

Established on this one scenario: the agreed rule gives a CEN that beats OFO+ on e_v (-0.47 mpu), e_vp (-1.46),
V_min,TS (+0.047 pu), t_out and V_min,STS, with 7 more MT tap moves and 2 fewer bank operations, at the cost of a higher
STS voltage deviation f_V^STS (+3.1 mpu) and no coupler operations.

## 6. Proposed dissertation wording (to be approved by the author)

> CEN uses the selected weights with two construction rules and one tuned factor. (i) Its continuous actuators are
> scaled block-wise: the TS block (AVR set-points, TS-connected IBGs) and the STS block (STS IBGs) are brought to the
> loop-gain ceilings of the two cascade layers, $\lambda_{\max}(B) = 1.45$ and $1.40$, and then by one common factor
> to a joint $1.45$. (ii) Each transmission tap and switched shunt carries the switching weight of its owning TS
> controller, multiplied by the ratio of its effect on all voltages CEN weights to its effect on the owner zone's
> voltages, so that no switch is cheaper for CEN than in the cascade. (iii) CEN's STS voltage weight is a quarter of
> the STS controllers' weight. This factor was chosen on scenario R2 as the largest of $\{1, 0.5, 0.25\}$ for which CEN
> attains a lower $e_V$ than every other variant, and it is held fixed in all other scenarios and the Monte Carlo
> ensemble. The ratio of TS to STS voltage weight has no counterpart in the cascade, where the two weights are
> selected in different layers. CEN is therefore a tuned centralised reference, not a bound.

## 7. Code of the selection

`M_STS_G_V_FACTOR = 0.25` in `variants.py` (commit after a45e22a, section "Branch"). 19 tests pass; resolved M:
`central_dso_g_v` 2500, `central_lambda_target` 1.4524, `central_lambda_target_sts` 1.40, scope "blocks", switch
rule "owner".

## Open points

- **Author:** gamma = 0.25 is the edge of the agreed grid and a factor of 4, not "a little"; the margin over O3 on R2 is
  0.47 mpu. The rule gives it; whether the dissertation wording "tuned" is sufficient is the author's call.
- R3 / R3b: out of sample in the reruns of record (on 27f0651 M at gamma 1 without blocks/C2 trailed O3 by 0.08 on R3
  and led by 0.14 on R3b). If M loses there, report it; do not re-tune.
- S1 (1.530 cap) sensitivity: append when the e5wimi runs end (~18:30-20:00).
- The REF / MC reruns of M (runs of record) after the branch is merged into the base: by the main session or the
  author, via thesis_runs.
- f_V^STS of M at gamma 0.25 is above O3's: the reference is better on TS voltages, not on STS voltage deviation.
