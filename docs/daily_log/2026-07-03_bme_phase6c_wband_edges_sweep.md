# BME Phase 6c — D2 (w_band × edges) sweep + fairness groundwork

**Date:** 2026-07-03
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Phase 6 item (3): calibrate the φ_band pairing (soft edges ×
w_band) of the bme ladder rung; plus per-zone Φ_i recording (groundwork
for the Phulpin fairness metric, item 5).

## What was done

1. Nine-point sweep at the **120-min horizon** (per Manuel's directive
   that calibration sweeps need only short horizons — saved as a
   standing preference). Grid: spec-default edges (0.97, 1.03), wide
   corridor (0.95, 1.05), and Manuel's operating-point-centred
   (1.01, 1.05) / (1.00, 1.06), each over w_band ∈ {1e2 … 1e4} subsets.
   Full table in `docs/BME_STATUS.md` §6c.
2. **Decision (Manuel): edges (1.01, 1.05), w_band = 1e4** — dominates
   the Pareto: −5.4 % last-hour losses (6b pairing: −3.9 %), V ∈
   [1.002, 1.062] with the best post-trip voltage support (lower hinge
   lifts the dip from the baseline's 0.978 to 1.002 pu) and the
   healthiest solve times (410 s vs 800–1300 s for the other families).
   Wired into `experiments/011_BME_LADDER.py` (`BME_W_BAND`,
   `BME_V_SOFT_MIN/MAX`).
3. **Uniform-Φ-metric fix**: the ladder now sets the D2 band definition
   on EVERY rung so the recorded Φ metric is the identical functional
   across none/vref/bme; the bme_loss control ablation zeroes w_band for
   its gradient only (its Φ metric differs by definition — compare on
   losses).
4. **Per-zone Φ_i recording**: `MultiTSOIterationRecord.bme_phi_zone_mw`
   + runner fill via `CommonObjective.phi_zone` — Phulpin fairness
   premise data; partition invariant Σ_i Φ_i = Φ_global verified live
   (1e-6) on the runner net.
5. Ablation-backstop question (6b) resolved by rationale: bme_loss keeps
   NO voltage backstop — the voltage escape is the ablation's finding.

## Findings

- Soft hinges do not CAP voltages, they slow excursions; strict
  containment requires either a stiff tight band (−2.2 % only, 9×
  runtime) or hard constraints.
- MIQP solve time is a useful calibration diagnostic: it spikes when the
  hinge fights the loss gradient at the operating point or when the band
  is too weak to settle the voltage profile.
- Manuel's edge-centring intuition (band centred on the operating
  schedule, hinge inactive at nominal) is confirmed empirically on all
  three axes (losses, voltage support, solve health).

## Reason

DECISION D2 (2026-07-02) deferred w_band magnitude and edge placement to
the Phase 6 calibration; this log and BME_STATUS.md §6c record the sweep,
the decision and its rationale.
