# 2026-07-09 — SBX-V Phase 1: band, directions, MIQP tier cost layer

**What:** Implemented Phase 1 of the SBX-V build plan. New package modules
`sbxv/config.py`, `sbxv/directions.py`, `sbxv/band.py`, `sbxv/miqp_cost.py` plus
`tests/sbxv/` (37 tests). No existing module modified (hard rule 5); SBX-H and MIQP-solver
suites re-run green (86 tests).

**Gates cleared beforehand (same day):** DP2 approved (€→objective constant
`obj_per_eur_per_step = 250/15 ≈ 16.7`, anchored so one quantum at Grenzpreis per step costs the
objective as much as a 5-mpu voltage violation under `g_v`); DP3 confirmed (positive netted
`q_hv` = LOWERING, negative = RAISING); DP5 confirmed (netted AggregationArea, option b);
B1 resolved (both regulatory PDFs now under `docs/regulatory/`).

**Key structure:**

- `sbxv/directions.py` — `Direction` enum + THE single (direction, magnitude) ↔ signed
  boundary-Q mapping (hard rule 8, DP3 executable).
- `sbxv/band.py` — `NormalBand` per AggregationArea; presets `fixed` (±50 v1) and
  `ar41414_default` (5 %/10 % of contracted P, spread ≥ 70 Mvar asserted, AR Anhang C).
- `sbxv/config.py` — frozen `SBXVConfig`, all plan-§8 keys explicit, Phase-0-resolved values;
  DP2 conversion helpers (€/Mvarh → objective units per Mvar per step).
- `sbxv/miqp_cost.py` — exact convex piecewise-linear tier pricing of the netted boundary Q,
  integrated via a **solver-instance proxy** (`PricingSolver`): neutral spec ⇒ untouched
  pass-through (R1 byte-identity by construction at the seam); active spec ⇒ problem
  augmentation (one tier row per area/direction on the netted quantity, one continuous variable
  per segment with pure linear cost), result stripped back to the original shape,
  `TierDecomposition` retained and §5 reconstruction asserted per solve.

**Design decisions documented in STATUS_SBXV.md §1.2:** open-tail final segment at the
Grenzpreis (V-D1 never-tighten dominates the plan's literal `q_ug` bound); tier-row slack bound
`slope/(2·g_z_tier)` asserted (an exactly hard row is impossible under the solver's shared-slack
encoding); V-D9 opposite-edge re-anchoring encodes exact Leitfaden marginal prices (worked-example
geometry unit-tested); simultaneous both-direction grants under the Leitfaden model `rep1`
(undefined in the plan).

**Why:** Plan §9 Phase 1; commercial layer strictly separated from physics (V-D1), no
modification of protected modules.

**Next:** Phase 2 (metering + settlement from the now-available regulatory documents) or
Phase 3 (request pipeline) per plan order; closed-loop R1 re-run lands with the Phase-4 wiring.
