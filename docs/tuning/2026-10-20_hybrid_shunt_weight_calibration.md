# Calibration of the hybrid bank movement weight `g_w_tso_shunt`

Date: 20 October 2026. Status: provisional value from one diagnostic hour.
No default, parameter set or study configuration was changed.

## What the weight is

Hybrid dispatch adds one integer bank-level variable per tertiary MSC/MSR bank
to the TS MIQP. The solver minimises

    wᵀ G_w w + ∇f ᵀ w,    G_w[bank, bank] = g_w_tso_shunt,

over bank increments `w_bank ∈ {−1, 0, +1}`, with no factor ½
(`optimisation/miqp_solver.py:445`). For a one-level step on bank *j*, holding
every other actuator fixed, the objective changes by
`g_w_tso_shunt − |∇f_j|`. Hence

    a one-level step is admitted iff  |∇f_j| > g_w_tso_shunt,
    with  ∇f_j = (∂y/∂level_j)ᵀ ∇_y f.

`g_w_tso_shunt` is therefore an **admission threshold in objective units per
level**, not an accumulation rate. The protocol D2b rule for `shunt_int_g_w`
(p90 boundary pressure mapped to a nominal 60-minute accumulation) does not
transfer to it, which is why the trial value 5 had no derivation.

`ShuntRecord.zone_tso_shunt_g_h` stores this gradient **per Mvar** of nameplate
coordinate at both the hybrid and the integrator recording site
(`experiments/runners/multi_tso_dso.py:4219` and `:4398`), so the per-level
gradient is `g_h · q_step_mvar`.

## Identity checked against the observed commits

Reconstructing the admitted set at `g_w_tso_shunt = 5` for the recorded
`regular` run, with bank levels tracked from the commit list and the dwell and
level limits applied, reproduces the switching exactly: **all ten commits lie in
the admitted set, none is unexplained.** The two instants with a non-empty
admitted set and no commit (360 s and 720 s in zone 3) are blocked by the
1800 s dwell after the 20 s commit on the same bank, and that bank commits again
at 1980 s. The identity therefore describes the realised decision, subject to
dwell, level limits and the one-step-per-zone constraint.

## Distribution

Source: the held-bank baseline `results/shunt_hybrid_diag_20261020/allow_default/slow`
— hybrid mode with the regular path priced out, so the banks never move and the
gradients are those of the undisturbed trajectory. 21 TS instants, 8 banks,
168 bank-instants, of which 85 demand engagement, the only available direction
while every bank sits at level 0. The same distribution is obtained from the
`legacy` run, and the switching `regular` run differs only marginally
(p95 28.39 against 28.41), so in this scenario the distribution is not
sensitive to the switching it produces.

Per-level |∇f| over the 85 engagement-demanding samples:

| p50 | p75 | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|
| 2.218 | 4.971 | 12.339 | 28.455 | 33.702 | 34.370 |

Admitted bank-instants per hour, by candidate weight:

| `g_w_tso_shunt` | 5 | 10 | 15 | 20 | 25 | 30 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| admitted bank-instants | 21 | 11 | 8 | 7 | 6 | 3 | 0 |

## Proposed value

The distribution is **bimodal rather than long-tailed**: a routine mode between
2 and 12 and an event mode between 25 and 34, with an empty interval between
p90 = 12.3 and p95 = 28.5. The threshold belongs in that gap, and the choice
inside it is insensitive — any value from about 17 to 24 admits the same seven
bank-instants.

**Proposed: `g_w_tso_shunt = 20`.** It admits seven bank-instants in this hour:
one at 20 s, three at 720 s and three at 1800 s, that is the load connection at
minute 10 and its removal at minute 30, and nothing in the 36 minutes after the
disturbance, where no bank-instant exceeds even 5. With the one-step-per-zone
constraint and the 1800 s dwell this corresponds to at most four commits. A
more conservative reading of "very high present demand" is the p99, 33.7,
which admits three.

The trial value 5 lies inside the routine mode, at roughly the p76 of the
distribution. That is the quantitative statement of why the regular path
ratcheted: at that price, ordinary operating pressure is already profitable.

## Limits

* Admitted is not beneficial. All ten commits admitted at `g_w_tso_shunt = 5`
  degraded interface-Q tracking. A higher threshold removes routine switching;
  it does not establish that the remaining event-triggered steps help. With
  `g_v = 40000` dominating the shared objective, a high-gradient bank step is a
  voltage-motivated action whose interface-Q cost is not priced proportionally.
* The gap is a property of this scenario: 85 samples, one hour, one load step
  up and down. Over the campaign windows the two modes may merge, in which case
  the declared quantile governs. The procedure, not the number, is the reusable
  artefact.
* The value is specific to the objective weights that produced the record. Any
  change of the common objective scale requires recalibration, as
  `docs/tuning/2026-10-20_hybrid_shunt_design_and_tests.md` already notes for
  the parameter registry.
* The identity neglects the joint solve: the MIQP selects bank steps together
  with the continuous actuators and under the output constraints. It predicts
  the admitted set, not which member of it is chosen — at 180 s and 360 s the
  selected bank was not the one with the largest gradient.
* Equal MSC and MSR steps are assumed, since the recorded gradient vector
  carries no device label per entry. The calibration module refuses unequal
  steps unless the step is passed explicitly.

## Reproduction

```text
python -B -X utf8 tuning/shunt_weight_calibration.py
  --runs results/shunt_hybrid_diag_20261020/allow_default/slow
  --quantile 0.99 --json results/shunt_hybrid_diag_20261020/gw_calibration.json
```

Several run directories may be pooled with one `--runs` list, which is how this
should be repeated over a representative set of campaign windows before any
selection claim.

## Verification at `g_w_tso_shunt = 20`

Two further one-hour runs, `regular` and `hybrid`, were executed at
`--movement-weight 20` with every other setting unchanged, and compared with the
recorded `legacy` baseline. `legacy` was not repeated: it is deterministic, and
the earlier re-run of `slow` reproduced its staged metrics to every digit, which
establishes that the deployed repository and the staging copy behave
identically. Results are in `results/shunt_hybrid_gw20_20261020`.

| Case | Commits | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] | DER utilisation p95 | TS V RMS [pu] | STS V RMS [pu] |
|---|---:|---:|---:|---:|---:|---:|
| `legacy` | 0 | 4.36584 | 11.72540 | 0.39897 | 0.005994 | 0.006753 |
| `regular`, `g_w = 5` | 10 | 8.06344 | 19.27176 | 0.49759 | 0.005864 | 0.007689 |
| `regular`, `g_w = 20` | 3 | 6.73692 | 19.33423 | 0.48303 | 0.005959 | 0.007263 |
| `hybrid`, `g_w = 20` | 3 | 6.73692 | 19.33423 | 0.48303 | 0.005959 | 0.007263 |

**The admitted-set prediction is confirmed exactly.** The commits occur at
20 s on zone 3 bank 6, at 720 s on zone 2 bank 0 and at 1800 s on zone 2
bank 1 — precisely the three instants the distribution predicted, with nothing
after 2160 s, where no bank-instant exceeds even a threshold of 5. The
calibration therefore predicts the switching behaviour prospectively, not only
in reconstruction. All three commits are engagements; no bank is released.

**The calibration does not make the regular path useful.** Reducing the commits
from ten to three recovers about half of the interface-Q RMSE degradation,
4.36584 → 6.73692 Mvar instead of 8.06344, and does not improve the
absolute-error p95 at all, 19.33 against 19.27. Three locked-in 25 Mvar steps
are sufficient to cause most of the error. The switching *rate* was therefore
not the binding problem.

The exchange rate the TS objective is accepting, relative to `legacy`, is:

| Quantity | Change | Measured by the TS objective? |
|---|---:|---|
| TS mean spatial voltage RMS | −0.6% | yes |
| Interface-Q RMSE | +54.3% | yes, but at a far smaller weight |
| Interface absolute error p95 | +64.9% | no |
| STS mean spatial voltage RMS | +7.6% | no |
| Aggregate DER utilisation p95 | +21.1% | no |

A hybrid bank step improves the one quantity the shared objective weights
heavily, the TS nodal voltages, by 0.6%, and degrades every quantity it weights
lightly or not at all. This is a structural property of the objective, not of
the weight: `g_w_tso_shunt` governs only how often the exchange is made. In
this scenario the regular path has no useful operating point — values from
about 34.4 upward switch it off and restore the baseline, and anything lower
buys TS voltage with interface-Q and STS reserve.

It also explains why no bank is ever released. After the disturbance clears, the
release pressure on an engaged bank falls below even a threshold of 5, because
the engaged bank continues to improve the TS voltage term it was selected by.
The objective is self-consistent: engage for voltage, retain for voltage, and
let the unweighted quantities pay.

In the `hybrid` variant ten slow candidates appeared — twice the five seen at
`g_w = 5`, since the regular path now holds the banks for most of the hour —
and all ten were refused as `voltage_tracking`. The slow guard is unchanged and
still gated by the 0.001 pu allowance.

Consequence for the design: the regular MIQP bank decision is viable only if
the bank's interface-Q and subordinate-layer effects are priced in the
objective that selects it. That is a change to the shared TS objective or to
the bank's own selection cost, and it is an architectural decision rather than
a tuning step.
