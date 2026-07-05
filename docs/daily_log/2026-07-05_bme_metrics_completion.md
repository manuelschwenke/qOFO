# 2026-07-05 — BME Phase 6 item (5): metrics completion (gap-to-oracle, Phulpin fairness, oscillation indicator)

**Context.** Phase 6 items 1–4 were closed (w_Φ = 1e5; D6 ε/c; D2
(1.01, 1.05) × 1e4; oracle rung 6d). Spec §6 requires three more
uniform metrics: gap to oracle (Φ integral and terminal), per-TSO
normalised overcost (Phulpin's fairness metric), and an oscillation
indicator on boundary voltages (dominant AR pole). This work package
implements all three plus the recording infrastructure they need.

## What was changed

### 1. Runner: METRIC-objective split (`experiments/runners/multi_tso_dso.py`)

Problem found before coding: the recorded `bme_phi_zone_mw` came from
the CONTROL-layer `CommonObjective`, so the oracle rung
(`single_zone_partition=True`) recorded a single-zone Φ_i — unusable
for fairness — and the `bme_loss` rung recorded a w_band = 0 metric —
a *different functional* than every other rung. This is the same
failure class as the 6d dispatch-map bug (control partition leaking
into a scenario/metric object).

Fix: a separate `bme_metric_obj` used ONLY for recording:

- partition: ALWAYS the fixed 3-area map (`fixed_zone_partition_ieee39`)
  even under `single_zone_partition` (Φ_global is partition-invariant,
  so `bme_phi_mw` is unchanged; Φ_i attribution becomes uniform);
- band weight: new config field `bme_metric_w_band`
  (`configs/multi_tso_config.py`, default `None` = follow `bme_w_band`)
  so the losses-only ablation zeroes only the CONTROL gradient while
  the recorded metric keeps the D2 band;
- new per-step signal `rec.bme_v_boundary` (`experiments/helpers/records.py`
  — the field existed uncommitted from the previous session, now
  wired): vm_pu at the 9-bus fixed registry B, on every rung — the §6
  oscillation-indicator input.

The control-layer `bme_obj` (gradients, hygiene gate ledger ΔΦ) is
untouched.

### 2. Ladder derived metrics (`experiments/011_BME_LADDER.py`)

New section "Derived cross-rung metrics", computed from the on-disk
pickles only (regenerated with the figures / `--plot`), written to
`results/011_BME_LADDER/metrics_derived.csv` + printed:

- **Gap to oracle** (spec §6): terminal = last-hour Φ means, integral
  = full-horizon Φ means, plus gap closure
  100·(Φ_none − Φ_r)/(Φ_none − Φ_oracle) — the quantifier for headline
  claim (2).
- **Phulpin normalised overcost** per zone i:
  100·(Φ_i^rung − Φ_i^none)/|Φ_i^none| on last-hour means; a positive
  maximum identifies a net loser (claim 3). Also added as a second
  panel to `fig5_zone_phi`. Requires matching 3-area zone sets (old
  single-zone oracle pickles drop out gracefully).
- **Oscillation indicator** (spec §6): dominant pole of an AR(2) fit
  per boundary-voltage series over the last-hour window, max modulus
  over the registry reported with period and signal label. Estimator:
  least-squares covariance method (regress y_k on y_{k−1}, y_{k−2}) after
  mean removal + linear detrend — Yule-Walker's biased autocovariances
  were measurably wrong on short windows (0.85 for a sustained
  sinusoid on 60 samples; the covariance method solves the damped-
  cosine recursion exactly). Fallback for pickles predating
  `bme_v_boundary`: inter-zone tie-Q series, labelled `tie_q_proxy`.
- **Fix**: `_band_violation_fraction` used the obsolete (0.97, 1.03)
  band → every rung reported 1.0; now defaults to the D2 edges
  (1.01, 1.05).
- Docstring updates (oracle rung description was still "NOT YET
  WIRED"; metric uniformity note).

### 3. Tests + validation

- `tests/test_bme_ladder_metrics.py` (8 green): AR(2) pole recovery on
  damped/sustained sinusoids (modulus, complex pair, period), constant/
  ramp/white-noise non-oscillation guards, gap + closure + overcost
  arithmetic on synthetic three-rung data, boundary-v vs tie-Q-proxy
  signal selection, D2 band-violation fix.
- `tests/diag_metric_partition.py` (smoke, ~3 min): 10-min oracle run →
  3-zone Φ_i, Σ_i Φ_i == Φ_global every step (<1e-6), 9-bus
  `bme_v_boundary`; 10-min bme_loss run → metric partition + invariant
  hold with control w_band = 0 and metric w_band = 1e4.

### 4. Consistent 120-min ladder re-run

The on-disk pickles mixed horizons (360-min rungs from 6b with
pre-6c calibration vs the 120-min oracle) — all five rungs re-run at
120 min with the final calibration so every derived metric compares
like with like. Regression evidence: the re-run oracle's last-hour
losses are BITWISE identical to the 6d run (57.379570332617654 MW) —
the metric split changed recording only.

**Results** (full table: `BME_STATUS.md` §6e,
`results/011_BME_LADDER/metrics_derived.csv`): on losses, bme ≈ oracle
to 2 kW (57.339 vs 57.337 MW, both ≈ −2.95 % vs none 59.085; vref
−0.02 %). On the now-uniform Φ metric the last-hour ranking INVERTS
(none 51.59 < vref 51.62 < bme 51.99 < oracle 52.57): the coordinated
rungs ride the upper D2 edge and the realised hinge cost exceeds the
in-scope loss gain — property of per-step greedy Φ descent under this
calibration (the exact-gradient oracle shows it too), and partly of
the metric's construction (the baseline tracks the band centre). The
gap-closure metric therefore got a sign guard (emitted only when the
oracle improves on none). bme_loss: +0.19 MW extra loss gain for
≈ +35 MW realised band penalty. Fairness: zone 2 is a net loser under
bme (+3.96 % Φ_i) and MORE so under the central optimum (+9.15 %).
Oscillation: no complex AR pair on any rung (no sustained boundary
oscillation this scenario); dominant real pole falls with coordination
strength (0.968 → 0.951 → 0.905). Calibration-philosophy question
(present divergence as finding vs revisit (w_Φ × w_band)) left to
Manuel — deliberately not retuned.

Also added: `fig0_concepts` (per-rung coordination-concept schematic,
static, regenerates with `--plot`).

### 5. Leftover uncommitted diffs folded in

`experiments/005_CIGRE_MULTI.py` (flat 1.03 pu zone schedule — the
operating point the D2 band was calibrated around), 007/009 docstring
renames (2026-07-03 consolidation), `records.py` field docstring —
all load-bearing for reproducing the calibrated results, previously
uncommitted.

## Why

Spec §5 Phase 6 / §6 metrics module; headline claims (1)–(3) need
these numbers. The metric-objective split enforces the spec-§6 rule
that rungs share ONE metric definition — the third instance of the
"same scenario/metric" assumption breaking through an indirect
coupling (after the Φ-metric definition in 6c and the dispatch
partition in 6d); all three are now guarded.

## Remaining (Phase 6)

Item (6): MC campaign (load scenarios × d ∈ {0,1,2,5} × H error × β ×
ε_switch; parquet + summary). Then Phase 7 analysis artefacts.
