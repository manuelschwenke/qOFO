# 2026-06-25 — TSO–TSO coordination: marginal (reserve) extension

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Extend the two-loop ΔV_ref coordinator so the outer loop can negotiate
a **non-zero** exchange for **cost-based reserve sharing** — the user's original
"charming dual" (`μ_i + μ_j`) idea, scoped safely to retargeting the bounded
setpoint anchor (never a competing gradient). Opt-in via `tie_econ_gamma`
(default 0 = unchanged subsidiarity).

## Mechanism

The subsidiarity anchor target shifts from 0 to the economic exchange
```
μ_i = per-zone reactive-reserve scarcity ∈ [0,1]   (0 abundant, 1 saturated)
ΔV_ref^econ = clip( γ·(μ_j − μ_i), ±dvref_max )     # route abundant → scarce
ΔV_ref ← (1−relax)ΔV_ref + relax·ΔV_realized − anchor·DB_Δ(ΔV_ref − ΔV_ref^econ)
```
`μ` aggregates the existing reserve signal — mean ``|Q − Q_mid|/Q_half`` over the
zone's synchronous machines + TS-DER (the band used by ``g_res_sg``/``g_res_der``).
When reserves are symmetric (`μ_i = μ_j`) the target is 0 ⇒ today's behaviour.

## What changed

- `controller/tso_controller.py`: `report_reserve_scarcity(measurement) → [0,1]`.
- `controller/tie_coordinator.py`: `TieCoordinatorConfig.econ_gamma`; `update`
  now takes `reserve={zone_id: μ}` and computes/anchors toward `ΔV_ref^econ`;
  `econ_target` exposed in `state()`.
- `experiments/runners/multi_tso_dso.py`: collects per-zone scarcity each round
  (`report_reserve_scarcity`), passes to `update`, records `tie_econ_target` and
  `zone_reserve_scarcity`.
- `configs/multi_tso_config.py`: `tie_econ_gamma` (default 0).
- `experiments/helpers/records.py`: `tie_econ_target`, `zone_reserve_scarcity`.
- `visualisation/plot_tie_coordination.py`: econ target dotted in the ΔV tile.
- Tests: 6 new reserve tests in `tests/test_tie_coordinator.py` (24 total green);
  `test_tie_coordination_hooks.py` (8) + `test_tso_output_gradient.py` green.

## Demonstration (`007 --reserve`) — honest-negative / inconclusive

Gen-4 trip @20 min, uniform schedule, OFF / COORD-sub / COORD-econ (γ=0.05):
```
steady zone reserve scarcity:
  OFF        : Z1=0.220  Z2=0.133  Z3=0.067
  COORD-sub  : Z1=0.220  Z2=0.133  Z3=0.067   (== OFF, subsidiarity neutral)
  COORD-econ : Z1=0.235  Z2=0.133  Z3=0.081   (mildly WORSE)
```
- The mechanism is **active** (the figure's economic-target panel shows non-zero
  `ΔV_ref^econ` per tie, −0.001…−0.008 pu, routing toward the more-scarce zone).
- **But there is no real scarcity to share:** max μ ≈ 0.235 ≪ 1 — the CIGRE base
  is reactively well-provisioned and the single gen trip does not saturate any
  zone. With nothing to share, the econ routing slightly perturbs the (already
  well-tracked) operating point and marginally *increases* the scarcity spread
  and shuffles tie flows — no benefit.
## Update — SG-only reserve + targeted gen trips (the decisive finding)

Reserve signal narrowed to **synchronous generators only** (TS-DER excluded).
Two trips:

```
trip G9 @bus37 (zone 1):   OFF Z1=0.402 Z2=0.059 Z3=0.071 ; econ ~same (Z3 0.071->0.105 worse)
trip G4 @bus32 (zone 3):   OFF Z1=0.422 Z2=0.039 Z3=0.268 ; econ ~same (no help to Z3)
```

The bus-32 trip **does** strain zone 3 (μ_3 = 0.27). But the extension still does
not help — because the **reserve signal is physically inverted for zone 1**:

- Zone 1 reads as the *most* scarce (μ_1 ≈ 0.42) even though it is the *richest*
  zone, because the SG-only count sees only zone 1's small controllable machines
  (G10 ±500, G8 ±350) and **excludes G1 (the ±5000 Mvar slack / system
  equivalent, idle)** — G1 is the slack, sits in `zone None`, and is not in any
  zone's `gen_indices`.
- So the economic target routes support *toward* zone 1 (highest μ, wrong) and
  *away* from the truly-strained zone 3 — the scarcity bars are unchanged across
  OFF/sub/econ.

**Mechanism is correct; the signal definition is the crux.** "SG reserve over the
zone's controllable gens" misidentifies the neediest zone whenever a zone hosts a
slack/equivalent machine. Also note the bus-38 machine (G1) is the slack ⇒ not
trippable, and its presence makes zone 1 structurally un-strainable.

**Fix (needs a decision before implementing):** the reserve signal must reflect
*true physical* reserve, i.e. **include the equivalent/slack machine's headroom
in its electrical zone** (G1 → zone 1, via TN bus 38). Then zone 1 reads abundant
(G1 idle), the routing inverts to 1→3, and zone 3 should actually be relieved.
Options: (a) add the slack gen to its electrical zone's reserve aggregate;
(b) capability-weight the per-gen scarcity (so G1's huge idle headroom dominates);
(c) exclude slack-hosting zones from being recipients.

## Update 2 — fix (a)+(b) done, exposes the deeper metric problem

Implemented: reserve signal is now **capability-weighted** (big machines'
headroom dominates) and the runner **folds equivalent/slack machines into their
electrical zone** (`report_reserve_capability` + the `tie_extra_gens` map: G1 →
zone 1 via TN bus 38, G7 → zone 3). Re-ran the G4@bus32 trip:

```
  OFF/sub    Z1=0.399  Z2=0.039  Z3=0.268
  COORD-econ Z1=0.399  Z2=0.037  Z3=0.270   (still no help)
```

Zone 1 barely moved (0.402→0.399). **Reason: G1 is not idle** — as the *slack*
it carries ~40 % of its ±5000 Mvar (~2000 Mvar) balancing the system, so its
*normalised* scarcity is ~0.40, the same as zone 1's small machines. Folding it
in changes nothing.

**Root cause is the *metric*, not the inclusion.** Normalised utilisation
`|Q−Q_mid|/Q_half` answers "what fraction is used", not "how much reserve is
left". Zone 1 has ~3000 Mvar of *absolute* headroom (the right provider) yet
reads needy (0.40); zone 3 has ~765 Mvar yet reads 0.27. The correct signal for
reserve sharing is **absolute remaining headroom [Mvar]** (μ_z = −headroom_z, or
a decreasing function of it), routing from high- to low-headroom zones.

**Also:** IEEE-39's single ±5000 Mvar slack machine in zone 1 is a structurally
awkward testbed for inter-zone reserve sharing — zone 1's reactive state is
dominated by the slack's balancing duty, not by reserve scarcity. A network with
distributed per-zone generation (no dominant slack) would demonstrate it cleanly.

**Status:** mechanism correct + unit-tested; signal improved (capability-weighted
+ slack-inclusive) but still normalised; clean demonstration blocked on (i) an
absolute-headroom metric AND (ii) a better-suited network. Kept the capability/
fold changes (physically more correct, opt-in, harmless when γ=0).

## Risks / open

- The extension is correct, active, and safe (opt-in, default off), but its
  **value is still unproven** — now blocked on the absolute-headroom *metric*
  and the slack-dominated test network, not on the mechanism.
- `μ` = aggregate normalised reserve is the simple first cut; a fuller
  marginal-cost `∂J_i/∂Q` would be more rigorous but heavier.
- When there is no asymmetry the extension should be left off (γ=0); with γ>0 in
  a non-scarce / mis-signalled system it is a mild, unnecessary perturbation.
