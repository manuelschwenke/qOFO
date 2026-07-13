# 2026-06-29 — Schedule→plant coherence for divergent zone voltage schedules

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Fix a confound that invalidated every divergent-schedule run (007
divergent default + tie_grad_eps sweep, and the gen-trip "reserve" demo, which
inherits a divergent schedule from `make_cigre_config`). The per-zone voltage
schedule (`zone_v_setpoints_pu`) only set the OFO *tracking reference*; it never
reached the autonomous voltage layer, so the plant never realised the schedule.

## Root cause (confirmed in code)

`zone_v_setpoints_pu` → `ZoneDefinition.v_setpoint_pu` (the OFO's voltage-tracking
target) only. The plant-side voltage anchors were left at a uniform 1.03:

- **AVR setpoints frozen at 1.03.** `network/ieee39/build.py:115` sets *all*
  `gen.vm_pu = ext_grid_vm_pu` (=1.03); `g_w_gen ≈ 5e9` (CIGRE config) then
  penalises AVR-setpoint *changes* so heavily that the OFO never moves them. So a
  zone referenced to 1.05 is held at ~1.03 by its own machines — the OFO chases
  1.05 with OLTC/shunt/DER-Q against a 1.03 anchor.
- **DER Q(V) droops re-anchor on measured V.** `experiments/helpers/plant_io.py`
  writes `qv_vref_anchor_pu = res_bus.vm_pu` each apply step, so the droop centre
  follows the (pinned ~1.03) measured voltage; the nominal fallback is also 1.03.

Net effect: the divergent schedule was never on the plant. The ~11 mpu baseline
V-RMS in the divergent runs was largely this systematic reference–plant offset,
**not** genuine tracking error, so the apparent coordination "benefit" (−12%/−8%
at grad_eps=0) was reducing an artifact. (Manu identified this.)

## Fix

`experiments/runners/multi_tso_dso.py` — after the per-zone gen/DER partition,
gated on an explicit `zone_v_setpoints_pu`, anchor the autonomous layer at the
schedule (uniform/`None` leaves build values intact — 000 unchanged):

1. `gen.vm_pu` ← zone schedule for every synchronous machine in the zone
   (incl. the slack gen, which is zone-mapped); `ext_grid.vm_pu` ← its zone's
   schedule. `g_w_gen` now holds each zone *at* its schedule instead of at 1.03.
2. TSO-DER `qv_vref_pu` (nominal droop centre) ← zone schedule, for a consistent
   cold-start / `seed_qv_equilibrium`; the runtime re-anchor then tracks measured
   V around the schedule.

Tie anchor was already correct (`TieLink.v_nom_i/j = zone_defs[z].v_setpoint_pu`,
runner ~L747), so the coordinator negotiates around the right per-side levels.

## Validation (divergent {1.05,1.03,1.02}, coordination OFF, 20-min)

```
  zone mean voltage  (target -> realised):
    Z1: 1.050 -> 1.0444   (-5.6 mpu)
    Z2: 1.030 -> 1.0267   (-3.3 mpu)
    Z3: 1.020 -> 1.0173   (-2.7 mpu)
```

Zones now hold their divergent schedules (pre-fix: all ~1.03). The few-mpu
residual is physical load-driven sag below the gen terminals; the remaining
per-zone V-RMS (~9–16 mpu) is legitimate intra-zone bus spread, not the old
reference fight.

## Coherent coordination results (007, gradient-exchange)

**Divergent default** {1.05,1.03,1.02}, `tie_grad_eps=1e-3`, OFF→COORD:
Σ|Q_tie| 235.1→154.3 Mvar (**−34%**), mean zone V-RMS 12.97→11.06 mpu (**−15%**) —
a genuine win-win (per-tie −16…−78%; L14 now small, ~4 Mvar in OFF).

**`tie_grad_eps` Pareto** (mutual-aid budget; OFF Σ|Q|=235.1, V-RMS=12.97):

| grad_eps | Σ\|Q\| Mvar | ΔΣ\|Q\| | V-RMS mpu | ΔV |
|---|---|---|---|---|
| 0    | 190.5 | −19% | 11.24 | −13% |
| 1e-4 | 159.1 | −32% | 11.38 | −12% |
| 3e-4 | 151.8 | −35% | 11.06 | −15% |
| 1e-3 | 154.3 | −34% | 11.06 | −15% |
| 3e-3 | 156.0 | −34% | 11.17 | −14% |
| 1e-2 | 156.0 | −34% | 11.16 | −14% |

- Monotone-saturating: more budget → more flow reduction + equal/better voltage,
  plateauing at ≈ −34/−35% & −15% by grad_eps ≈ 3e-4–1e-3.
- `grad_eps=0` (strict Pareto/subsidiarity) captures only the uncontested gain
  (−19%) — too conservative on a real divergence; the knee (≈1e-3) unlocks the
  full win-win; beyond it saturated (safeguard rarely binds).
- **Decision: keep default `tie_grad_eps=1e-3`** (at the efficiency knee). This
  REVERSES the confounded sweep (grad_eps=0 looked best there) — artifact gone.
- 007 `--sweep` made resilient (per-point retry/skip) after repeated transient
  `Z:` share drops aborted runs (incl. a `profiles.csv` FileNotFoundError).

## Ancillary-support demo (007 `--reserve`, rewritten) — honest negative

Rewrote `main_reserve` to a clean test: UNIFORM 1.03 schedule (coherent; isolates
reserve stress from voltage divergence), three-way OFF / COORD-Pareto (grad_eps=0)
/ COORD-aid (grad_eps=1e-2), voltage-framed relief, resilient runs. Trip idx 1 =
zone-2's only synchronous machine (bus 31, 800 MVA, confirmed by net inspection).

```
                 Z1 V-RMS  Z2 V-RMS  Z3 V-RMS   μ_Z2
  OFF            6.25      4.37      6.80       0.230
  COORD-Pareto   4.72      5.10      7.21       0.599
  COORD-aid      4.45      5.73     10.67       0.595
```

**Negative, and diagnostic:** the gen trip creates a **reserve-headroom** stress
(μ_Z2 0.02→0.23), NOT a voltage stress — zone 2's 31 DER hold its voltage (V-RMS
stays ~4 mpu). The coordinator descends the **boundary-voltage** gradient (γ ≈
2·g_v·err_b dominates; g_res weak), so it sees no signal that zone 2 needs help;
instead it pulls support toward the boundary-rich zone 1 (Z1 improves) and aid mode
over-perturbs zone 3 (+3.87 mpu). **Conclusion: the mechanism is a voltage-profile
coordination tool (demonstrated by the divergent −34%/−15% win-win), not a
reactive-reserve-sharing tool.** Reserve sharing is a distinct ancillary product
the voltage lever doesn't deliver on IEEE-39 (single dominant slack, DER-rich zones
self-hold voltage) — consistent with the 06-25 reserve-extension finding.

**Voltage-stress retry (compound: Z2 gen trip + 400 Mvar reactive load step at
bus 9, uniform 1.03):** now genuinely sags Z2 (OFF V-RMS Z2=11.64 mpu vs 4.4 with
trip alone; μ_Z2=0.46). Detection correctly picks Z2.

```
                 Z1 V-RMS  Z2 V-RMS  Z3 V-RMS
  OFF            7.00      11.64     7.55
  COORD-Pareto   6.24      10.97 (-0.67)  8.48
  COORD-aid      3.70      11.34 (-0.30)  7.75
```

**Still negative for *preferential* rescue, and now precisely diagnosed:** more
aid budget does NOT help the stressed Z2 more — its relief *shrinks* (−0.67→−0.30)
while zone 1's improvement *grows* (7.00→3.70). Reason: Z2's stress is a deep
*local* deficit (400 Mvar pinned at the tie bus), so raising the L14 boundary moves
bus-9 voltage little per unit of Q zone 1 must inject — a poor cost/benefit. The
per-zone safeguard *correctly refuses* the large sacrifice a rescue needs, and the
joint optimiser banks the cheap zone-1 gains instead.

**Characterisation (final for this testbed):** the gradient-exchange coordinator is
a *bounded joint voltage-profile optimiser*, NOT a preferential ancillary-rescue
mechanism. Ancillary service = cheap, mutually-beneficial, bounded boundary-voltage
support; an expensive rescue is deliberately out of scope (the safeguard protects
the provider). True preferential aid would need *asymmetric weighting* (scale γ by
scarcity μ, or bias κ toward the needier side) — a deliberate design change that
trades away subsidiarity symmetry; discuss before building.

**Latent bug fixed:** `connect` load-contingency events are mutated in-place by
`prepare_load_contingencies` (element_index assignment); sharing them across runs
raised "row already exists". `main_reserve` now builds fresh events per run.

## Consequences / open

- **All prior divergent empirical results are void** (007 divergent default,
  tie_grad_eps Pareto, the gen-trip reserve sweep). Re-running 007 default +
  sweep on the coherent plant; will re-decide the `tie_grad_eps` default then.
- **005_CIGRE_MULTI behaviour changes** (it schedules {1.04,1.02,1.00}); now those
  are physically realised. Expected/intended, but its results will shift.
- **DSO-level DER droops** (HV/MV, `dso_qv_vref_pu`=1.03) are *not* touched here;
  the TSO/EHV layer is what matters for tie coordination, but a fully coherent
  cascade would propagate the HV setpoint to the DSO droop centres too — deferred.
- **Part B (coordinator ΔV_ref → boundary AVR/droop) intentionally not done.** The
  AVRs are frozen (g_w_gen) and the droops re-anchor, so the coordinator's small
  boundary moves are tracked by the OFO's other actuators; revisit only if the
  coordinator proves under-powered on the coherent plant.
