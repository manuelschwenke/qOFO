# Tap engagement at the selected weights: analysis and decision note

Timestamp: 2026-10-28, ~09:45–10:15 Europe/Berlin.

**Reason:** the author saw that the STS coupler taps barely move at
`ch9_selected_band_2026_10` (running `experiments/run_multi_system_ofo.py`),
although DSO_4 sits at about 1.05–1.075 pu, and suspected the engagement
rule. This was a side session of the Ch 9 D1/D2 session. It is analysis
only: no weights, parameter sets, procedure or producers changed, and no
simulation was run.

**Decision note:** `docs/tuning/2026-10-28_tap_engagement_at_selected_weights.md`.

## What was added

- `analysis/tap_engagement/` (new, read-only post-processing):
  - `thresholds.py`: per-column commit threshold
    θ_i = g_w,i / (2 g_V |Σ_j H_V[j,i]|) via
    `tuning.stage1_screen.engage_deviation` on the cached operating points
    (`op_rural_700_net82afed5b_W1..W6`), Stage-I point vs selected set. It
    first reproduces `stage1_start_10_2026_banks` with `design_stage1`
    (exact).
  - `usage.py`: tap, bank and coupler operations and STS/TS deviations from
    1.03 pu, read from the `margin_check` pickles of run 0001.
  - `candidates.py`: θ_STS,min, feasibility, hunting attribution (STS
    vs TS) and STS operations for the 242 Stage-II candidates.
  - `capture_H.py`: one controller build at W1 (no time loop). It reads the
    voltage columns of every tap and bank, their sign structure and the
    device names.
  - `followups.py`: four checks:
    - the necessary condition max|e| ≥ θ at each recorded move;
    - λ_max(B) under a tap-price-only change;
    - the variance share of dso_g_v in θ_STS;
    - offline re-selection under bounded-threshold rules.
- Outputs (gitignored): `results/analysis/2026-10-28_tap_engagement/`.

## Key results

- **STS thresholds.** The selected set multiplies the STS coupler threshold
  by 5.31 (g_w ×0.35, dso_g_v ×0.066): 0.0150 → **0.0797 pu** for the
  strongest coupler and up to 0.148 pu for the weakest. This exceeds the band
  headroom of 1.08 − 1.03 = 0.05 pu.
- **No interface-Q term.** `dso_gamma_oltc_q = 0`, so the STS taps carry no
  interface-Q gradient.
- **STS operations.** They fall from 19 to 1 per 5 × 12 h. The single
  remaining move (DSO_4, W1) occurred at v_max 1.0818 pu, above the band,
  with max|e| below θ. It is constraint-driven.
- **TS thresholds.** They rose by only 20 % (0.0075 → 0.0090 pu). Low TS
  tap use comes from:
  - a well-regulated TS voltage;
  - one class price set by the strongest column;
  - two TS network OLTCs at 0.04–0.06 pu;
  - two machine transformers (gen1, gen2) with a zero voltage column in the
    TS model (open question).
- **Cause in the procedure.** Stage II spans dso_g_v 3× wider than the tap
  price, so dso_g_v carries 76 % of var(log θ_STS). The filter bounds θ only
  from below: STS hunting appears only at θ ≤ 0.0169 pu. The e_v(TS) ranking
  walked dso_g_v from 10⁴ to 660. This is the same mechanism as on
  2026-08-20.
- **Re-selection.** Over the existing candidates, any bound (θ_STS ≤
  0.02/0.03/0.05 pu, or e_v(DS) ≤ 1.25×/1.5× Stage I) selects
  **301c285ee52b**, whose settling run was aborted (not measured). The next
  candidate, c08b1d3d33ab, has settling PASS.

## Recommendation and open decision

The recommendation is option (c1): amend the filter with θ_STS,min ≤ 0.05 pu,
re-select offline, and run Experiment C for 301c285ee52b, then D1/D2. The
author's alternative is (a): accept, and report the implied thresholds as a
finding. The note gives the costs and consequences of (a)–(d).
