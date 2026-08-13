# Heterogeneous per-zone TSO strategies — tie-coordination demo

**Date:** 2026-07-01
**Author:** Manuel Schwenke (with research assistant)
**Scope:** New experiment telling a different story than 007/008's "divergent
voltage references": three TSO zones each running a genuinely different
*operational strategy* (voltage tracking / loss-min bounds-only / pure
reserve optimisation), to test whether the horizontal gradient-exchange tie
coordinator still finds a jointly better operating point when zones don't
even share the same objective *type*. Separately, the user reported that on
long simulations the coordinator sometimes *degrades* performance (both for
same- and different-reference zones) — that tuning/algorithm problem is
**not** addressed here; this notebook was designed so it could also *expose*
that behaviour if present in this scenario (it did not — see Results).

## What was added

### 1. Per-zone objective-weight plumbing (small, additive, backward-compatible)

Until now, only the voltage-tracking *reference* (`zone_v_setpoints_pu`) was
per-zone configurable in `MultiTSOConfig`; the voltage-tracking *weight*
(`g_v`), the hard voltage bounds (`v_min_pu`/`v_max_pu`), and the
reserve/loss weights (`tso_g_res_sg`, `tso_g_res_der`, `tso_g_loss`) were all
global scalars applied uniformly to every zone. Added six new
`Optional[Dict[int, float]]` override fields mirroring the exact
`zone_v_setpoints_pu` idiom (falls back to the existing global scalar when a
zone isn't listed, `None` reproduces legacy behaviour byte-for-byte):
`zone_g_v`, `zone_v_min_pu`, `zone_v_max_pu`, `zone_tso_g_res_sg`,
`zone_tso_g_res_der`, `zone_tso_g_loss`. Also added global `v_min_pu`/
`v_max_pu` fields (defaults 0.90/1.10) since the runner previously never
passed these to `TSOControllerConfig` at all, silently relying on its own
dataclass default.

**No changes were needed to `controller/tie_coordinator.py`.** The
coordinator already treats each zone's boundary gradient as a black box, so
whichever objective terms are active per zone automatically flow into the
negotiation.

### Key method / structure of change

Added a single local helper in `run_multi_tso_dso()`:

```python
def _zone_scalar(zone_dict, zone_id, fallback):
    return float(zone_dict.get(zone_id, fallback)) if zone_dict is not None else float(fallback)
```

and used it at the existing per-zone construction sites: `ZoneDefinition.g_v`
and, inside the per-zone `TSOControllerConfig(...)` construction, `v_min_pu`,
`v_max_pu`, `g_res_sg`, `g_res_der`, `g_loss`. Added a `verbose>=1` print of
each zone's constructed weights (`[zone z] g_v=... v_min/max_pu=...
g_res_sg=... g_res_der=... g_loss=...`) — the only way to confirm the wiring
took effect, since `run_multi_tso_dso` returns only the record log, not the
controller objects.

### 2. New notebook: `experiments/010_TSO_HETEROGENEOUS_STRATEGIES_DEMO.ipynb`

IEEE-39, fixed 3-area partition, `enable_tie_coordination` OFF vs COORD
(gradient-exchange), clean 360-minute SimBench-profile-driven horizon (no
discrete contingencies — deliberately overrides `make_cigre_config()`'s
own baked-in gen-trip/load-step/line-trip schedule). Per-zone strategy,
editable via a single `ZONE_STRATEGIES` dict at the top of the notebook:

| Zone | Strategy | g_v | v bounds | g_loss | g_res_sg / g_res_der |
|---|---|---|---|---|---|
| 1 | track, v_ref=1.02 p.u. | 1e7 (= base `g_v`) | [0.90, 1.10] | 0 | 0 |
| 2 (DSO-rich) | bounds-only, loss-focus | 0 | [1.00, 1.06] | 3e5 | 0 |
| 3 | pure reserve optimisation | 0 | [0.90, 1.10] | 0 | 1e4 / 1e4 |

Reuses `005_CIGRE_MULTI.make_cigre_config()` as the base config (same
pattern as 007/008) and 007's validated `TIE_KW` tuning constants
(`tie_grad_step=0.5, tie_anchor=0.5, tie_deadband_v_pu=0.002,
tie_dvref_max=0.08`), inlined rather than imported to avoid a redundant
second import of 005. Adds a **windowed-trend diagnostic** beyond what
007/008 do: six 60-minute bins across the horizon, OFF vs COORD, for zone-1
V-RMS error, `Σ|Q_tie|`, whole-network `total_losses_mw`, and zone-3 reserve
scarcity — specifically to make a slow degradation trend visible if present,
plus a reference system-load trace per window to distinguish genuine drift
from profile-driven variation.

## Results (first full run, this scenario/tuning only)

`results/010_tso_heterogeneous_strategies_demo/`: `summary.csv`,
`fig1_per_zone_story.png`, `fig2_coordination_mechanism.png`,
`fig3_windowed_trend.png`, `log_{OFF,COORD}.pkl` (1080 records each, ~380s /
~330s wall-clock).

Steady-window (last 120 min) scalars, OFF → COORD:

- **Zone-1 V-RMS tracking error: 10.03 → 5.65 mpu (−44%).** Clear
  improvement — coordination lets zone 1 draw on zones 2/3's spare reactive
  capacity (neither tracks voltage) to hold 1.02 p.u.
- **Σ|Q_tie|: 562 → 574 Mvar (+2%).** Small increase, not a reduction —
  unlike 007/008's divergent-reference scenario where coordination reduced
  *both* tracking error and tie flow, here the mixed per-tie pattern (some
  ties carry more, some less; see `fig2`, panel 1) shows zone 1 pulling more
  reactive support through the ties as the price of better tracking. Net
  effect is a small increase, not the clean win-win of the earlier scenario.
- **`total_losses_mw`: 32.82 → 33.19 MW (+1.1%).** Essentially a wash;
  zone 2's `g_loss=3e5` does not show a clear loss-reducing effect at this
  weight (see Open points below).
- **Zone-3 reserve scarcity μ: 0.0584 → 0.0575 (small improvement).**

**Windowed-trend finding (directly relevant to the separately-reported
degradation issue):** in this scenario/tuning, COORD does **not** reproduce
long-run degradation. Zone-1 V-RMS error under OFF *rises* monotonically
across the 6 windows (6.3 → 10.1 mpu) as the SimBench daily profile
progresses, while COORD stays essentially flat (5.2–6.1 mpu) — coordination
is the *more* stable of the two here. None of the other three windowed
metrics show a widening OFF/COORD gap either; `Σ|Q_tie|` and
`total_losses_mw` both rise across windows for OFF and COORD in lockstep
(profile-driven, not a coordination artefact — confirmed against the
per-window reference load trace). This suggests the degradation reported
for the divergent-voltage-reference scenario is not a universal property of
the gradient-exchange mechanism, at least not in this scenario/horizon/
tuning — worth comparing directly against that scenario's own windowed
behaviour as a follow-up.

**Unexpected finding — zone 2's soft voltage corridor does not hold.**
Zone 2's realised voltage envelope reaches up to ≈1.116 p.u. (OFF) /
≈1.113 p.u. (COORD) during the steady window — well above its intended
[1.00, 1.06] "bounds-only" corridor (see `fig1`, row 2). Root cause: the
corridor is enforced only as a *soft* slack penalty at weight
`g_z_voltage=1e-12` (the repo-wide default), which is negligible — so the
corridor is nominal, not actually binding, in this configuration. The
"bounds-only" story is therefore not fully realised as intended; zone 2's
voltage is effectively unconstrained in practice, only weakly nudged.

## Assumptions / model facts

- Cascaded OFO (`control_scope="cascaded"`), `local_sensitivities_tso/dso=True`
  (per-zone/per-DSO reduced Jacobians, same as `make_cigre_config()`'s
  default) — does not affect the tie coordinator's own mechanism (message
  passing on each zone's own boundary gradient, not on cross-zone Jacobian
  blocks).
- `zone_v_rms_err_pu` is computed against a single *global* reference
  (`cfg.v_setpoint_pu`, pinned here to zone 1's 1.02) for every zone, so it
  is only a meaningful "tracking error" for zone 1; zones 2/3 have no
  tracking objective and are correctly excluded from that panel.
- `total_losses_mw` is whole-network, not zone-2-only — there is no
  per-zone MW-loss field logged in `MultiTSOIterationRecord`.
- The tie coordinator's step size (`grad_alpha=tie_grad_step/(2·g_v)`,
  `grad_eps=tie_grad_eps·g_v`) is derived from the single *global* `g_v`,
  not any per-zone value — calibrated for zone 1's side of a tie, not
  necessarily for zone 2/3's side (curvature there comes from `g_loss` /
  `g_res_sg` / `g_res_der` instead, different units entirely).

## Risks / unresolved points

1. **`g_loss=3e5` and `g_res_sg=g_res_der=1e4` are unvalidated starting
   guesses** — no prior numeric precedent for `g_res_sg`/`g_res_der`
   anywhere in this repo. The loss panel in particular shows only a ~1%
   effect; a `g_loss` ablation (rerun with it at `0.0`) would confirm
   whether that's a genuinely weak effect or the weight simply isn't biting
   yet. Left for a follow-up run, not done here.
2. **Zone 2's voltage corridor is not actually binding** (see Results) —
   would need a per-zone `g_z_voltage` override (not implemented; `g_z_voltage`
   remains a single global scalar) to make the "bounds-only" story hold as a
   real constraint rather than a nominal one.
3. **L14 (zone 1 ↔ zone 2, at the slack bus) remains structurally pinned**,
   consistent with 007/008 — visible in `fig2`'s agreed-vs-realised ΔV_ref
   panel as the largest agreed/realised gap of the five ties.
4. Whether this scenario's specific numeric tuning (weights, `TIE_KW`)
   generalises to other zone/strategy assignments is untested.

## Files changed

- `configs/multi_tso_config.py` — 8 new fields (`v_min_pu`, `v_max_pu`,
  `zone_v_min_pu`, `zone_v_max_pu`, `zone_g_v`, `zone_tso_g_res_sg`,
  `zone_tso_g_res_der`, `zone_tso_g_loss`), all additive/backward-compatible.
- `experiments/runners/multi_tso_dso.py` — `_zone_scalar()` helper; wired at
  the `ZoneDefinition` and per-zone `TSOControllerConfig` construction sites;
  added a `verbose>=1` diagnostic print of each zone's constructed weights.
- `experiments/010_TSO_HETEROGENEOUS_STRATEGIES_DEMO.ipynb` — new notebook
  (19 cells).

Verified: static config-sanity assertions pass; smoke run (2 TSO periods,
both OFF/COORD, `verbose=1`) confirms the printed per-zone weights match
`ZONE_STRATEGIES` exactly, no exceptions; full 360-min OFF/COORD run
completed cleanly (1080 records each) and produced the figures/summary
above.

## Next note to update in Obsidian

`[[todo]]` — mark "heterogeneous per-zone strategies demo" done; add
follow-ups: (a) `g_loss`/`g_res_sg`/`g_res_der` ablation sweep to confirm
weight sensitivity; (b) per-zone `g_z_voltage` override if a real (binding)
bounds-only corridor is wanted for zone 2; (c) compare this scenario's
windowed-trend behaviour directly against the divergent-voltage-reference
scenario (007/008) to check whether the separately-reported long-run
degradation is scenario-specific.

## Second follow-up (same day): extended to 16h — degradation reproduced here too

Manu extended the notebook's default horizon to 16h (`HORIZON_MIN=16*60`,
`FORCE_RERUN=True`) directly in the saved `.ipynb`, deliberately, to look for
long-run drift in this scenario specifically. Builder script + notebook
synced to match (`FORCE_RERUN` reset to `False` afterward so the cache is
reused going forward); `results/010_tso_heterogeneous_strategies_demo/`
(`log_{OFF,COORD}.pkl`, all 4 figures, `summary.csv`) now reflect the 16h run,
superseding the 360-min numbers above.

**Result: the degradation reproduces in this scenario too, as an oscillation, not a monotonic decline.**
Steady-window (last 5h): zone-1 V-RMS **5.40 → 5.63 mpu (+4.3%, COORD now
WORSE)** — reversed from the 360-min result where COORD was clearly better
(−2.8%). `Σ|Q_tie|` stays persistently 10-30 Mvar higher under COORD for most
of the run. Zone-2 loss and zone-3 reserve both show clean, monotonic,
profile-driven declines over the day (not coordination artefacts) — the
signal is specifically in zone-1 tracking.

The full 16-hour trace (`fig1`/`fig2`, zone-1 V-RMS panel) shows COORD ahead
for the first ~4h, then OFF pulls ahead for an extended stretch (~300-700
min, OFF dropping to its best value of the run, ~4.2 mpu, around t=550-650),
then two more crossings before the run ends. This lines up closely with zone
3's reserve headroom (`fig1`, bottom-right) collapsing from ~1650 to ~800
Mvar starting around t=450 — right when zone-1's tracking gets most erratic.

**Connects directly to the same-day gradient-magnitude finding (see
`docs/daily_log/07_2026/2026-07-01_tso_tso_long_run_degradation_investigation.md`
and memory `tso_tso_tie_coordination`):** zone 3's boundary gradient is
10,000-30,000× smaller than its neighbours' on 3 of its 5 ties, i.e. the
tie-coordination negotiation is effectively a zone1-vs-zone2 conversation
with zone 3 rubber-stamped out almost everywhere. As zone 3's condition
shifts (reserve margin collapsing), the negotiation has no effective channel
to reflect that change — plausible link to why the previously-clean
same/divergent-reference scenarios stayed robust over 24h (all zones there
share comparable-scale objectives) while THIS heterogeneous scenario, where
the imbalance is large by design, does not. Not yet a proven causal test —
see the open points below.

**Open follow-up, not yet done:** re-run this same 16h scenario with a
gradient-normalization fix applied (see the 3 candidate designs discussed
with Manu — quick per-zone weight division, adaptive/EMA, or the principled
curvature-projection extension of `controller/gw_precondition.py`) to test
whether restoring zone 3's effective voice removes or reduces this
oscillation. This is the most direct test connecting the two findings.

## Follow-up (same day): zone-2 corridor fix + real per-zone loss/reserve plots

Manu reviewed `fig1` and caught that zone 2's voltage was reaching ≈1.116
p.u. — the [1.00, 1.06] corridor from item (b) above was not actually
binding — and separately asked for the per-zone loss/reserve *objectives*
to be evaluated and plotted directly (fig1 had only shown whole-network
`total_losses_mw` and the abstract `zone_reserve_scarcity` index).

### What was added

1. **`configs/multi_tso_config.py`**: `zone_g_z_voltage:
   Optional[Dict[int, float]]` — per-zone override of `g_z_voltage` (the
   voltage-slack weight), same fallback idiom as the other `zone_*` fields.
   Root cause of the corridor not binding: `g_z_voltage` defaults to `1E-12`
   repo-wide — an intentionally near-inert placeholder that normally relies
   on `g_v` tracking to keep voltage inside bounds. Zone 2 runs `g_v=0`, so
   nothing was actually pulling it back inside `[1.00, 1.06]`; the corridor
   was nominal, not enforced.
2. **`experiments/runners/multi_tso_dso.py`**: wired via `_zone_scalar` at
   the `gz_diag_target` construction (same per-zone loop as the other
   overrides); added `g_z_voltage` to the `verbose>=1` diagnostic print.
3. **`experiments/helpers/records.py`**: new `zone_losses_mw: Dict[int,
   float]` — ground-truth active-power line loss (`net.res_line.pl_mw`)
   summed over exactly each zone's own `line_indices` (the same EHV lines
   the zone's `g_loss` objective targets), computed in the runner's existing
   per-zone per-step loop. Mirrors `total_losses_mw`'s role at the
   whole-network level but scoped correctly to one zone.
4. **No new field needed for reserve** — `zone_reserve_headroom_cap_mvar` /
   `zone_reserve_headroom_ind_mvar` (directional Mvar headroom across each
   zone's SG + TS-DER fleet, from `TSOController.report_reserve_headroom`)
   were already logged unconditionally per zone; the notebook now plots
   these directly instead of the derived `[0,1]` scarcity index.

### Finding the corridor weight

Swept `g_z_voltage` candidates for zone 2 on progressively longer smoke-scale
runs (a 6-TSO-period / 18-min run showed **no effect at all** — zone 2's
voltage hadn't yet climbed near 1.06 at that horizon; had to extend to
150 min to see the bound actually challenged):

| `g_z_voltage` | V_max @150min | frac. time >1.06 | zone-2 loss @150min |
|---|---|---|---|
| 1e-12 / 1e4 / 1e6 | 1.088 (indistinguishable) | 0.52 | 4.86 MW |
| 1e8 | 1.068 | — | 4.68 MW |
| 1e9 | 1.061 | 0.39 | 4.68 MW |
| 3e9 | 1.060 | 0.16 | 4.68 MW |
| **1e10** | **1.060** | **0.06** | 4.67 MW |

Picked **1e10** (diminishing returns past ~1e9; loss barely changes further,
so this is close to the cheapest weight that holds the line). Re-verified on
the full 360-min run: `v_max_z2_pu` = 1.0605 (OFF) / 1.0604 (COORD) — the
corridor now holds almost exactly at 1.06 p.u., down from 1.116 in the
buggy version. It remains a genuinely *soft* bound by design (~6% of steps
still marginally over at the smoke-scale check), not a hard MIQP constraint.

### Corrected results — supersedes the numbers in the section above

Steady-window (last 120 min), OFF → COORD, re-run with the fix:

- Zone-1 V-RMS: 6.22 → 6.05 mpu (**−2.8%**) — a real but much smaller
  improvement than the −44% reported before the fix.
- Σ|Q_tie|: 347 → 372 Mvar (**+7.2%**) — smaller absolute scale than before
  (562→574) but a proportionally larger relative increase.
- Zone-2's own line loss (`zone_losses_mw[2]`, NOT whole-network): 4.30 →
  4.41 MW (+2.6%) — COORD is very slightly worse here, not better.
- Zone-3 reactive-reserve headroom: capacitive 1655→1641 Mvar (−14),
  inductive 1680→1694 Mvar (+14) — a small, roughly offsetting shift out of
  a ~1650-1700 Mvar pool.

**Important correction to the earlier windowed-trend claim.** The original
run's windowed-trend figure showed OFF's zone-1 tracking error rising
sharply over the day (6.3→10.1 mpu) while COORD stayed flat, and this was
reported as evidence that the long-run degradation Manu separately observed
did *not* reproduce here. That was **wrong, or at least confounded**: it was
the corridor bug, not a genuine coordination effect — zone 2's unconstrained
voltage climbing past 1.11 p.u. was distorting the network in a way that
made OFF's zone-1 tracking specifically get worse over time. With the
corridor properly enforced, OFF is now fairly stable across all six windows
(6.1–6.7 mpu, no clear trend), and COORD is modestly better for most of the
run but crosses slightly *above* OFF in the final window (300–360 min:
COORD 6.2 vs OFF 6.1 mpu) — visible directly in `fig2`'s zone-1 tracking
trace as a late-run crossover. This is a small, single-run effect (not
necessarily robust — would need repeated seeds to confirm), but it means the
"no degradation in this scenario" conclusion should be treated as
**retracted pending a clean re-check**, not as confirmed. The comparison
against 007/008's own windowed behaviour (follow-up (c) above) is now more
important, not less.

Outputs regenerated in place under
`results/010_tso_heterogeneous_strategies_demo/` (same filenames).
