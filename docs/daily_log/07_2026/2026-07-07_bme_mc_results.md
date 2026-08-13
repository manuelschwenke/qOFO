# 2026-07-07 — BME MC campaign results (Phase 6 item 6 complete → Phase 6 ✅)

Campaign of `experiments/012_BME_MONTECARLO.py` (launched 2026-07-06,
`--jobs 4`) completed: **75/75 converged runs** — 10 accepted paired
scenarios × {none, bme, oracle} (11 attempts; seed 20260710 dropped —
its random contingency schedule NaN'd the private MIQP gradient on all
three arms including `none`, a scenario-generator pathology handled by
the drop-and-replace design) + all 45 one-factor sweep runs on seeds
20260705–07. Artefacts: `results/012_BME_MC/` (MC_SUMMARY.md,
runs.parquet, ledger.parquet with 2 011 entries, schedules.csv).
Full readings table: `docs/BME_STATUS.md` §6f. Headlines:

1. **bme vs none**: paired loss reduction +1.06 % ± 2.41 — slightly
   negative on light scenarios (−0.7…−2.0 %), clearly positive on
   heavy ones (+0.2…+5.1 %); on the common Φ, bme improves 8/10.
2. **Oracle inverts out-of-distribution**: the greedy centralised
   per-step Φ-MIQP is worse than `none` on 9/10 random scenarios and
   catastrophic on Φ (132 vs 45 MW, hinge-dominated; 34.3 taps).
   Distributed bme beats it on 10/10 (−2.09 ± 0.65 MW). The 6d/6e
   "≈100 % closure" was 005-scenario-specific. Hypothesis for the
   chapter: delay + filter + slotting are stabilisers, not
   concessions.
3. **Delay is a stability boundary, not graceful decay**: d=0 unstable
   2/3 seeds, d=2 unstable 1/3, d=1 and d=5 stable — synchronous
   exchange is the pathological case.
4. **H-error graceful** (+0.37 MW worst at σ=30 %); **drops robust**
   to p=0.2; selfish ablation nearly free on the (light) sweep seeds —
   heavy-scenario sweeps not probed (coverage caveat recorded).
5. **ε-sweep**: monotone switching control (14→7 taps over
   ε = 0→2.6e4) at ≤0.03 MW loss spread.
6. **§3.10.2 premise weak as measured** (P = 0.45, sign 0.47 pooled) —
   realised ΔΦ confounded by exogenous drift on random scenarios
   (clean-window reference was 1.00); counterfactual frozen-integer
   measurement = future work, recorded honestly.

No code changes today; interim `--summarize` invocations only.
Phase 6 is complete. Phase 7 (analysis artefacts) next; the
oracle-inversion and d=0 findings should shape its figure set
(per-scenario paired panels; (d, β) stability map candidate).
Working tree (H-error axis + D2 edges + 012 + docs) still uncommitted
— Manuel's call on commit timing.
