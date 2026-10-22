# Calibration of the hybrid bank movement weight

Timestamp: 2026-10-20, 16:40 CEST.

Reason: the hybrid regular path ratcheted through every available bank at the
trial weight `g_w_tso_shunt = 5`, which had no derivation. The author chose to
derive the weight from the recorded bank-gradient distribution before changing
anything else.

Changes:

- Added `tuning/shunt_weight_calibration.py`: reads
  `ShuntRecord.zone_tso_shunt_g_h` from one or more run directories, converts it
  to the per-level gradient with the bank step from each run's `config.json`,
  filters to the available direction for held-bank baselines, and reports
  quantiles, the bank-instants each candidate weight admits and the value at a
  declared quantile. Analysis only; no control law or default is touched.
- Added `docs/tuning/2026-10-20_hybrid_shunt_weight_calibration.md` with the
  derivation, the distribution, the proposed value and the limits.

Method: the MIQP objective is `wᵀ G_w w + ∇f ᵀ w` with no factor ½ and
`G_w[bank, bank] = g_w_tso_shunt`, so for `w_bank ∈ {−1, 0, +1}` a one-level
step is admitted iff `|∇f_bank| > g_w_tso_shunt`. The weight is thus an
admission threshold in objective units per level, not an accumulation rate, and
the protocol D2b rule for `shunt_int_g_w` does not transfer to it. The recorded
`g_h` is per Mvar, so the per-level gradient is `g_h · q_step_mvar`.

Verification: reconstructing the admitted set at `g_w_tso_shunt = 5` for the
recorded `regular` run, with levels tracked from the commit list and dwell and
level limits applied, reproduces all ten commits with none unexplained. The two
admitted-but-not-committed instants are blocked by the 1800 s dwell on a bank
that had committed at 20 s and commits again at 1980 s.

Findings:

- Over the held-bank baseline, 85 engagement-demanding bank-instants give
  per-level |∇f| of p50 2.218, p75 4.971, p90 12.339, p95 28.455, p99 33.702,
  max 34.370. The `legacy` run gives the same distribution and the switching
  `regular` run differs only marginally, so it is not sensitive to the switching
  it causes in this scenario.
- The distribution is bimodal, not long-tailed: a routine mode from 2 to 12 and
  an event mode from 25 to 34, with an empty interval between p90 and p95.
- Proposed `g_w_tso_shunt = 20`, inside that gap. It admits seven bank-instants,
  all at the load connection at minute 10 and its removal at minute 30, and
  none in the 36 minutes afterwards. Any value from about 17 to 24 admits the
  same seven, so the choice inside the gap is insensitive. The p99, 33.7,
  admits three.
- The trial value 5 sits at roughly the p76 of the distribution, which is the
  quantitative reason for the ratchet: routine operating pressure was already
  profitable.

## Verification, 17:05 CEST

Two one-hour runs at `--movement-weight 20`, `regular` and `hybrid`, against the
recorded `legacy` baseline, stored in `results/shunt_hybrid_gw20_20261020`.

- The prediction is confirmed exactly: three commits, at 20 s on zone 3 bank 6,
  720 s on zone 2 bank 0 and 1800 s on zone 2 bank 1, and nothing after 2160 s.
  The calibration predicts the switching prospectively. All three are
  engagements; no bank is released.
- It does not make the regular path useful. Interface-Q RMSE improves only from
  8.06344 to 6.73692 Mvar against a baseline of 4.36584, and the absolute-error
  p95 does not improve at all (19.33 against 19.27). Three locked-in 25 Mvar
  steps cause most of the error, so the switching rate was not the binding
  problem.
- The exchange rate relative to `legacy` is −0.6% TS mean spatial voltage RMS
  against +54.3% interface-Q RMSE, +64.9% absolute-error p95, +7.6% STS mean
  spatial voltage RMS and +21.1% aggregate DER utilisation p95. The bank step
  improves the quantity the shared objective weights heavily and degrades those
  it weights lightly or not at all. This is structural, not a weight choice:
  values from about 34.4 upward switch the regular path off and restore the
  baseline, and anything lower buys TS voltage with interface-Q and STS reserve.
- The same mechanism explains the absent release. Once a bank is engaged, the
  release pressure stays below even a threshold of 5, because the engaged bank
  keeps improving the TS voltage term it was selected by.
- In the `hybrid` variant ten slow candidates appeared, twice the earlier five,
  and all ten were refused as `voltage_tracking` against the unchanged 0.001 pu
  allowance.

Open: the regular MIQP bank decision is viable only if the bank's interface-Q
and subordinate-layer effects are priced in the objective that selects it. That
is a change to the shared TS objective or to the bank's own selection cost, and
therefore an architectural decision to discuss rather than a tuning step. The
calibration itself should be repeated over representative campaign windows,
where the routine and event modes may merge.
