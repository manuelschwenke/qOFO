# 2026-07-09 — SBX-V E2 results (band-width Pareto)

**What:** Completed the E2 campaign (`experiments/019_SBXV_E2.py`): 3 Monte-Carlo seeds
(012/006 harness — random profile-year start + random contingency schedule) × arms
{none, ±0, ±25, ±50, ±75, ±100 Mvar, `ar41414_default`} × 120 min. All 21 cells green.
Outputs: `results/019_SBXV_E2/e2_sweep.csv`, `e2_pareto.png`, per-cell JSON journals.

**Findings** (details STATUS_SBXV.md §5.5):

1. Clean monotone Pareto: payments 14 468 → 0 € and no-grant exceedance 198 → 0 Mvarh from
   ±0 to ±100 Mvar; knee between ±25 and ±50. TS voltage quality and reserve margin are
   IDENTICAL across all arms — the commercial layer redistributes cost without touching the
   physics (V-D1 in closed loop).
2. Persistent-exceedance indicator behaves per the LF Präambel: full-horizon persistence at
   ±0 (8 consecutive windows), isolated single windows at the knee.
3. `ar41414_default` passed the Anhang C spread assertion (contracted P from Σ sn_hv_mva)
   and prices ≈ ±50 symmetric, but is scenario-sensitive (0 € in seed 1, 2 622 € in seed 2)
   → band sizing is a quantile question (E4).
4. Grant pipeline dormant in all cells (0 requests): the random draws never violate the TS
   voltage band, so condition B never fires — the zones hold voltage by paying the
   Grenzpreis instead (±0 arm: 14.5 k€/2 h and still zero violation energy). Closed-loop
   grant exercise needs E3's targeted in-feed trip.

**Also:** post-V-D9-fix E1 re-run launched (replaces the stale §5.3 numbers).

**Why:** Plan §9 Phase 5 E2 acceptance (sweep runnable end-to-end, tidy CSV + figure,
findings note in STATUS).
