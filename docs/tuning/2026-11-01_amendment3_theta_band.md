# Amendment 3 to the weight-selection procedure: STS coupler engagement band 0.02–0.04 pu

**Status:** author decision 2026-11-01 ("Do the proposed revision").

**Widened at 20:27 the same day** ("Please widen to 0.02 to 0.04"), from 0.02–0.03 to
0.02–0.04 pu. This was before any settling result under the narrower band. Run
`0002_2026-11-01_201949_ch_9_7_amendment3` was stopped after 7 min (`SUPERSEDED.txt`).
Wherever this note says 0.03 below, it now reads 0.04, and "1.05–1.06" reads 1.05–1.07.
Under 0.02–0.04: 50 / 244 inside, 38 feasible, 23 in the tie group. d162 is last in the tie
group. Seven feasible candidates are new, four of them in the tie group: aa8e (rank 1,
ratio 7.8e5), 8054 (ratio 1.12e5, borderline), d088, 17cd.

**Declared AFTER the results** of run
`results/ch_09_parameter_selection/0001_2026-10-31_135000_ch_9_6_full_procedure` (amendment 2).
That run reached the settling step and found no candidate passing it. This amendment is a
reaction to that outcome and is reported as such.

**Applies to:** the re-selection on the unit-commitment base,
`experiments/ch_09_parameter_selection/ch_9_7_amendment3.py`, result
`configs/paramsets/ch9_selected_uc_2026_10_a3.json`. Run 0001 stays as recorded.

## Rule

The engagement bound of amendment 2 becomes a band:

$$
0.02\ \mathrm{pu} \;\le\; \theta_{\mathrm{STS,min}}
= \min_{c} \min_{i} \frac{g_{w,\mathrm{dso\_oltc}}}{2\, g_{V,\mathrm{DS}}\, \big|\sum_j H_V^{(c)}[j,i]\big|}
\;\le\; 0.03\ \mathrm{pu}
\qquad (\text{amendment 2: } \theta_{\mathrm{STS,min}} \le 0.02).
$$

- **Meaning.** Under a uniform deviation of the STS voltages from 1.03 pu, the voltage
  gradient alone commits one step of the strongest coupler between 1.05 and 1.06 pu, neither
  earlier nor later.
- **The author's intent, stated 2026-11-01:** STS voltages inside the 1.03 ± 0.05 soft band,
  and taps engaging when all voltages approach the upper edge, "1.05, 1.06".
- **The lower end is new.** Amendment 2 admitted θ down to 0. The leaders of run 0001
  (θ 0.011–0.013) would commit already at about 1.041–1.043 pu.
- θ is computed exactly as before (`ch_9_3_amendment1.theta_sts_min`, on the run's
  operating points).

## Unchanged

- Limits (`ch_9_3_select.py`, unchanged), including **hunting tolerance 0**.
  - The author offered to accept one hunting event. That was checked and **not adopted**,
    because it changes nothing: the winner is the same, and only 2 more candidates become
    feasible.
- Ranking: e_v(TS), with the tie-break on e_v(DS) within 2 %.
- The settling requirement: p95 n_settle ≤ 3·N_inner = 27, and median residual ≤ 1 Mvar.
- **The settling loop, also unchanged** (`ch_9_6_full_procedure.settle`):
  - checks run in rank order;
  - a failed candidate leaves the set, and the order, including the 2 % band, is rebuilt
    from the remaining candidates before the e_v(DS) tie-break applies again.
  - Claude first described "settling before the e_v(DS) tie-break" as part of this
    amendment. In fact it is already the semantics of the frozen loop, so it is not a change.
- The Stage-I point stays in the bank as the e_q reference of the limits. Its θ (0.0150) is
  outside the band, so it is never ranked; it is infeasible anyway (2 hunting reversals).

## No new search

- The candidates are the 244 that run 0001 simulated:
  - Stage II-a: the Sobol survivors;
  - four II-b rounds, which were centred on amendment-2 winners.
- A II-b search under the band could find other points. That is not done here; it would be
  a further decision.
- 36 candidates fall inside the band and 31 are feasible. 19 of them lie in the e_v(TS) tie
  group.

## Why (findings of 2026-11-01, daily log `2026-11-01b_ch9_settling_failure_diagnosis.md`)

1. **Run 0001 failed settling.** All eight candidates it checked fail, with median
   residuals of 1.6–3.5 Mvar per 10-Mvar step.
   - The residual is a steady offset, not a capability limit: 0 of 120 steps are
     capability-limited.
   - It grows with dso_g_v/g_q: Spearman 0.93 over the eight.
2. **The e_v(DS) tie-break drives it.** e_v(DS) improves with dso_g_v/g_q (Spearman −0.66),
   and e_v(TS) is indifferent to it (−0.03). So the tie-break favours STS weights that
   track the TSO's Q setpoint weakly.
3. **Hunting is not the obstacle to stronger Q tracking.**
   - 76 % of the hunting events are on TS machine transformers.
   - They are set by the TS tap price: none of 38 candidates with g_w_tso_oltc < 15 is
     hunting-free; 88–95 % of those with ≥ 25 are.
   - Hunting does not follow g_q (Spearman −0.12).
4. **Relaxing only the upper bound** (θ ≤ 0.03 or ≤ 0.05) gives the same winner as
   amendment 2, which fails settling. The band takes out the candidates whose taps engage
   too early, and these are also the ones with the largest dso_g_v.

## Expected cost and outcome (stated before the settling checks under the band)

- d162c73ca1fc (θ 0.0208, ratio 3.1e4) is predicted to pass (≈0.35 Mvar from the
  empirical relation residual ∝ ratio^0.81). But it ranks last in the tie group (e_v(DS)
  0.00833).
- It is selected only if the 18 candidates ranked above it fail, plus any that enter the
  band when its base moves.
- Most of those are predicted to fail (ratio ≥ 1.3e5). e63aed2d04cb (1.09e5) and
  a1de75d1fc4e (1.29e5) are borderline.
- Two checks here and three on e5power at a time: about 10–15 h.
- Finished checks of the same test (diagnostics of 2026-11-01, same code, same arguments,
  same weights) are taken over.

## Reporting

- The thesis states the amendment, its timing (after run 0001's result) and its reason.
- It also reports what run 0001 produced: no settling-capable winner under amendment 2.
