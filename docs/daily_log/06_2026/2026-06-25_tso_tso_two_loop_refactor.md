# 2026-06-25 — TSO–TSO coordination: two-loop ΔV_ref refactor

**Author:** Manuel Schwenke (with Claude Code)
**Scope:** Replace the failed inner-price/dual tie coordinator with a two-loop
**ΔV_ref negotiation**: each zone tracks an agreed per-side boundary voltage
setpoint via its *primary* `g_v` (no price); a slow outer loop negotiates the
agreed difference `ΔV_ref` (relax toward the realised difference + subsidiarity
anchor toward 0). Motivated by the finding that a price competing with `g_v` is
inert when weak and PF-diverging when strong.

## Why

The price design added `±λ·V_boundary` to the *primary* objective. Empirically:
inert at `λ≪g_v`, and at `λ~g_v` it overdrove the boundary voltages and the
power flow diverged. Root cause: a secondary objective must not compete in the
primary's gradient. The fix moves coordination to a **setpoint** (bounded, fed
into `g_v` tracking) negotiated by a slow outer loop — it cannot destabilise the
PF, and an unreachable setpoint is just a bounded tracking error.

## Mechanism

Per tie `e=(i,j)`, schedules `V_nom_i/j`, anchor `V_anchor=½(V_nom_i+V_nom_j)`:
```
inner:  each zone tracks  V_ref_i = V_anchor + κ·ΔV_ref ,  V_ref_j = V_anchor − (1−κ)·ΔV_ref   via g_v
outer:  ΔV_ref ← Π_±max{ (1−relax)·ΔV_ref + relax·(V_i−V_j) − anchor·DB_Δ(ΔV_ref) }
marginal (free, envelope thm):  μ_i = −2g_v·(V_i − V_ref_i) = boundary tracking error
```
`relax = α·g_v` folded so gains are O(0.1–1) (no parameter tracks `g_v`).
Controllable tie ⇒ zones track ⇒ relax term →0 ⇒ anchor drives `ΔV_ref→0`
(decouple). Stiff tie (bus 39) ⇒ large tracking error ⇒ relax holds `ΔV_ref` at
the structural value (left alone), no instability.

## What changed

- **`controller/tie_coordinator.py`** — rewritten. State = `dvref` per tie (+
  `dv_realized`, `marginal` diagnostics). `TieLink` gains `v_nom_i/v_nom_j`
  (+ `v_anchor`). `TieCoordinatorConfig` = `relax, anchor, deadband_v_pu, kappa,
  dvref_max` (dropped `alpha_lambda, lambda_max, v_min/max, ff_smoothing`).
  `generate_messages` emits per-side `V_ref` (no price).
- **`core/message.py`** — `TieCoordinationMessage` drops the `price` field.
- **`controller/tso_controller.py`** — removed `_tie_price_by_vrow` and the
  price injections in `_compute_output_gradient` / `_compute_objective_gradient`;
  `receive_tie_coordination` now only redirects boundary `v_setpoints_pu`
  (no price). Gradient invariant `∇_u f == ∇_y f @ H` restored to baseline.
- **`experiments/helpers/records.py`** — tie fields now `tie_dvref`,
  `tie_dv_realized`, `tie_marginal` (+ kept `tie_v_i/j`, `tie_q_mvar`).
- **`experiments/runners/_multi_tso_helpers.py`** — per-line tie Q recorded
  unconditionally post-PF into `tie_q_mvar` (so any config has per-line flows).
- **`configs/multi_tso_config.py`** — knobs `tie_relax, tie_anchor,
  tie_deadband_v_pu, tie_kappa, tie_dvref_max` (+ kept `tie_q_band_mvar`).
  `enable_tie_coordination` default **False** (was flipped True earlier; reset
  given it is neutral in the base case and needs per-scenario validation).
- **`experiments/runners/multi_tso_dso.py`** — `TieLink` per-side schedules,
  new coordinator config, round records `dvref/dv_realized/marginal`, plotter
  args updated.
- **`visualisation/plot_tie_coordination.py`** — tiles: ΔV_ref vs realised Δ |
  Q_tie+band | V_i/V_j | marginal `m_e`.
- Tests `tests/test_tie_coordinator.py` (18) + `tests/test_tie_coordination_hooks.py`
  (8) rewritten for the new API; `tests/test_tso_output_gradient.py` green.

## Verification (CIGRE base case, 70-min, last 30 min)

```
setting        L2    L5   L14*   L18    L25   V_rms  Vmin   Vmax
OFF          28.2  10.0  17.8  10.6   53.3   7.33  1.004  1.047
COORD a=0.3  29.6  10.7  19.9  13.8   54.3   7.31  1.004  1.048
COORD a=0.8  29.4  10.5  19.3  12.9   54.0   7.32  1.004  1.048
ΔV_ref(a=0.8): L14 −0.0068 (anchored from structural ≈−0.014 toward 0); others |·|<0.005
```

**Verdict:**
- ✅ **Stable** — both runs completed, no PF divergence (the price design's
  failure mode is structurally gone).
- ✅ **Voltage-safe** — `V_rms` unchanged (7.33→7.31/7.32), band respected.
- ✅ **Negotiation behaves as designed** — anchor pulls `ΔV_ref→0`; stronger
  anchor (0.8) pulls L14 closer to 0 (−0.0068 vs −0.0101) but the structural
  difference resists full collapse.
- ❌ **No flow reduction in this base case** — flows ~unchanged (slightly up).
  Confirms the decentralised-sweep finding: with a uniform 1.03 schedule the
  boundaries are already near-consensus and the tie flows are **structural**
  (active-dispatch/load driven), so there is no reducible voltage divergence for
  the coordinator to remove.

## Divergence-scenario validation (the decisive test) — POSITIVE

Added a per-zone schedule hook (`MultiTSOConfig.zone_v_setpoints_pu`, applied at
`ZoneDefinition` construction) and ran zones at **1.05 / 1.03 / 1.01** so the
boundaries genuinely diverge.  CIGRE, 70-min, last 30 min:

```
setting        L2    L5   L14*   L18    L25   V_rms  Vmin   Vmax
OFF (div)    88.1  40.6  13.8  38.3   95.8  14.47  0.994  1.061
COORD a=0.5  75.2  33.1  16.0  34.0   90.2  13.28  0.997  1.060
COORD a=1.5  74.4  32.0  15.3  32.8   89.8  13.23  0.997  1.059
Σ|Q_tie|: −10% (a=0.5), −12% (a=1.5)
```

**Verdict — the concept works when there is divergence to act on:**
- **Tie flows reduced** on every controllable tie (L5 −21%, L2 −15%, L18 −14%,
  L25 −6%); Σ|Q_tie| −10…−12%.
- **Voltage tracking IMPROVED** (V_rms 14.47→13.23 mpu, −8%) and the band
  tightened (Vmax 1.061→1.059, Vmin 0.994→0.997) — the large inter-zone reactive
  transfer caused by divergent schedules was *stressing* voltage; decoupling it
  helped both objectives at once (no trade-off here, a win-win).
- **Stiff L14 (bus 39) correctly resists**: its agreed `ΔV_ref` is pulled toward
  0 by the anchor (−0.0139 → −0.0079 → −0.0050) but the *realised* difference and
  flow don't follow (−0.0137, flow even +11%) — the marginal/residual persists,
  exactly the designed "leave the pinned tie alone" behaviour.
- **Stable** throughout; diminishing returns past `anchor ≈ 0.5` (the boundaries
  hit the zones' control limits).

So the base-case neutrality was a *scenario* artifact (uniform schedule ⇒ nothing
to decouple), not a design failure: given genuine inter-zone voltage divergence,
the two-loop `ΔV_ref` coordination decouples the controllable ties, improves
voltage, and leaves the structurally-pinned tie alone — stably.

Reproduce: `make_cigre_config()` + `zone_v_setpoints_pu={1:1.05,2:1.03,3:1.01}`,
`enable_tie_coordination=True`, `tie_relax=0.5`, `tie_anchor=0.5`.

**Permanent experiment:** `experiments/007_TIE_COORDINATION.py` runs OFF vs COORD
on the divergence scenario and saves a 4-panel figure
(`results/007_tie_coordination/tie_coordination_divergence.png`): per-tie flow
bars, Σ|Q_tie| and V-RMS time series, agreed-vs-realised ΔV per tie (the L14 gap
marks the pinned tie).  `--live` runs COORD with the live plots instead.
Also added `MultiTSOConfig.zone_v_setpoints_pu` (per-zone schedule override,
keyed by zone id 1/2/3) — usable from any runner entry point incl. `000`.

## Open / next
- User scratch `experiments/diag_tie_coord_sweep.py` references the **old** knobs
  (`tie_alpha_lambda/lambda_max/ff_smoothing`, `rec.tie_lambda`) and needs
  updating to `tie_relax/tie_anchor`, `rec.tie_dvref/tie_marginal`.
- Two-timescale stability (two neighbours both relaxing `ΔV_ref`) still wants a
  contraction argument before any formal non-oscillation claim.
