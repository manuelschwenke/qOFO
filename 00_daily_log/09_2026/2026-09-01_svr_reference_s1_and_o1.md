# 2026-09-01 — S1 (classical pilot-node SVR) and O1 (TS-OFO, continuous only)

**New files**
- `controller/svr_controller.py`
- `experiments/CIGRE_2026/008_PILOT_BUS_SELECT.py`
- `experiments/CIGRE_2026/009_S1_VS_O1.py`

**Modified**
- `configs/config.py` — `tso_mode='svr'`, `tso_oltc_mode`, five `svr_*` fields
- `experiments/runners/multi_tso_dso.py` — six surgical edits (below)
- `experiments/CIGRE_2026/005_CIGRE_MULTI.py` — `S1`, `O1`, `PILOT_BUSES`, `DISPLAY_NAMES`

## Why

The thesis needed a controlled comparison against the deployed classical scheme.
Two rungs were missing from the V1–V5 ladder:

- **S1** — classical pilot-node SVR at the TS layer, droop beneath, TS taps
  under the rule-based logic. The state-of-practice reference.
- **O1** — TS-OFO on the **continuous actuators only**, taps under that *same*
  rule-based logic.

`O1` is the load-bearing one. Without it, going from the classical scheme to
the proposed one changes the control law **and** the discrete treatment at once.
With it, `S1 → O1` changes the control law and nothing else.

## O1: no new solve path needed

`TSOController._block_sizes` computes `n_integer = n_oltc + n_shunt` and
`integer_indices = range(n_continuous, n_continuous + n_integer)`. Withholding
the taps leaves both empty and the MIQP degenerates to a QP **with no change to
the formulation**. So `tso_oltc_mode='local'` simply skips populating
`zone_oltc_trafos`, and reuses the existing `DiscreteTapControl` install path so
the withheld taps are *rule-based* rather than frozen.

Note the base CIGRE config has `install_tso_tertiary_shunts=False` and
`shunt_dispatch='off'`, so OLTCs are the only TS discrete class — nothing else
had to be withheld.

## S1: design

The controller emits its dispatch in the **same per-zone `u` layout the OFO
uses** — `[Q_DER | Q_PCC_set | V_gen_set | s_OLTC | s_shunt]`, see
`core.plant.writes_from_zone_tso` — so it goes through the identical apply,
record and score path. That is what makes `S1 → O1` a comparison of control laws
rather than of harnesses.

- Synchronous machines: RPR drives `V_gen_set` (classical path).
- Converter TS-DER: direct Q via `Q_DER` (RPR assumed converged).
- `Q_PCC_set` held at zero — the classical scheme dispatches no interface
  setpoint and has no port for a capability interval. **Interface metrics are
  not reported for S1**, and no `SVR-above-OFO-below` configuration exists.
- Discrete blocks echo the measured position back (no-op, preserves offsets).

**Two timescales, kept separate.** The RVR integrates only on TSO ticks
(`tso_period_s = 180 s`, per-dispatch gain `180/300 = 0.6`); the RPR runs every
step (`dt_s = 20 s`, gain `20/60 = 0.33`). Collapsing both onto the dispatch
grid gives the RPR a per-step gain of **3.0** and it overshoots — an artefact of
the harness, not a property of the scheme.

**Gain normalisation.** `k_i = 1 / (T_RVR · |s_p,a|)` is not free. `s_p,a` is
estimated once by perturbation on `copy.deepcopy(net)` (so the live plant is
never disturbed), then frozen — classical SVR computes participation offline.
The controllers borrow the OFO's own `actuator_bounds`, so both schemes see the
same capability curves.

## Pilot buses

`008_PILOT_BUS_SELECT.py`, criterion `argmax_p Σ_{j∈R_a} |dV_j/dQ_p|` over the
reduced Jacobian. PQ restriction falls out of the Jacobian (PV/slack have no
row), which also guarantees the pilot lies inside the OFO's reference set.

| Zone | Pilot bus | \|R_a\| | s_p,a [pu/level] | Margin over runner-up |
|------|-----------|---------|------------------|----------------------|
| 1    | 25        | 8       | +0.736           | 4.7 % |
| 2    | 6         | 12      | +0.089           | **0.7 %** |
| 3    | 20        | 10      | +0.159           | **0.2 %** |

Two things to carry into the write-up:

1. The criterion **barely discriminates** in zones 2 and 3. Those pilots are
   effectively arbitrary among near-equivalents — a known property of pilot-node
   schemes on meshed areas. Put no weight on their identity.
2. `s_p,a` spans a **factor of eight**. Under a fixed `k_i`, zone 1 would be
   driven eight times harder than zone 2, and any difference measured against
   the proposed controller would be partly about detuning. The normalisation is
   what removes that.

## Two bugs found and fixed during bring-up

**(1) Zero-initialised integrator collapses the system.** The reactive level is
an area-wide *utilisation fraction*, so `q_a = 0` commands **zero reactive
output**. On the loaded benchmark the machines dumped their Q in one step and
the pilot voltages fell to **0.88 pu** before the RVR could integrate back, then
overshot to 1.09. Classical SVR seeds the integrator from the dispatch already
in service. Fixed with an aggregate-ratio bumpless start
(`_initialise_level`), so the area's *total* reactive output is preserved across
the handover rather than merely its average utilisation.

This is ordinary anti-bump initialisation of an integral controller. It is **not**
a repair of the scheme's absences — no inequality constraints, no discrete
actuators, no inter-area exchange all stay exactly as they are.

**(2) Sign error in that fix.** `_headroom(upper=False)` already returns
magnitudes; negating the lower branch as well double-flipped the sign, and zone 1
(`Q_area = −134.3 Mvar`, absorbing) was handed a **positive** level of `+0.027`
— the exact bump the initialisation exists to prevent. Both branches now take a
positive denominator and the sign comes from `q_tot` alone.

## (3) The one that mattered: two controllers on one transformer

First full run: **both S1 and O1 diverged** with `LoadflowNotConverged`, about a
minute after the gen-2 trip at minute 60. Not the `res_gen` NaN I expected.

The machine-transformer taps were winding up. Across steps 181–186 the relay
*wanted* `trafo#0` at +1, +2, +4, +6, +9 while the rate limiter dragged it back
to −1 each step — monotonic windup, then divergence.

Cause: **S1/O1 is the first configuration where `DiscreteTapControl` on a
machine transformer coexists with a *moving* generator voltage setpoint.**

| variant | MT taps | gen `vm_pu` | outcome |
|---------|---------|-------------|---------|
| L1/L2 | relay | **pinned** at 1.03 | stable — relay has a static target |
| O2/O3/M | OFO-owned | OFO-moved | stable — one controller per device |
| S1 | relay | **RPR-moved** every 20 s | **diverges** |
| O1 | relay | **OFO-moved** | **diverges** |

A relay regulating the HV side while an AVR regulates the LV side of the same
transformer is two integral controllers on one device. It does not settle.

**Resolution (author's decision, 2026-09-01): hold the MT taps at their
planning position in S1 and O1.** No relay installed.

This is the faithful classical treatment, not a workaround. In the classical
hierarchy the MT tap is a planning/commissioning setting, adjusted by the
operator far above the secondary timescale — which is exactly the contrast Ch 2
draws, and the reason in-loop discrete co-optimisation is a *differentiator*
rather than something both schemes share. No plant runs a relay against its own
AVR.

Consequences:
- `O1 → O2` now reads "held at planning position vs co-optimised in the loop",
  which is **cleaner** than "local relay vs optimiser" and matches the Ch 2
  claim directly.
- `L2 → S1` now also changes the tap treatment (relay → held). Documented as
  footnote b of `tab:case3:steps`. That step is read for the value of
  hierarchical coordination; the two rungs carrying the chapter's argument
  (`S1→O1`, `O2→O3`) are unaffected.
- L1/L2 keep the relay, so existing V1/V2 results stay valid.

## (4) The CIGRE drivers were on the wrong scenario

Flagged by the author: the study uses `rural_700`, but `005_CIGRE_MULTI.py` and
`006_CIGRE_MONTECARLO.py` both set `cfg.scenario = "base_410"`.

**That line was never chosen.** `git log -L` shows commit `5266850` mechanically
rewrote the deprecated `"wind_replace"` to the string its own deprecation shim
maps to (`build.py:378-381`, *"use 'base_410'"*). So the drivers have been
running the DEFAULT installed-capacity scenario ever since that rename.

The difference is not marginal — 700 MW installed DER per DSO against 410:

| | wind | PV |
|---|------|-----|
| `base_410` | 270 MW (150 internal + 3×40 coupling) | 140 MW |
| `rural_700` | 460 MW (250 internal + 3×70 coupling) | 240 MW |

**Independent confirmation:** `tuning_mc/scenarios_mc.py:108` — *"Five 90-min
`rural_700` scenarios"*. The controller weights (campaign_0815, reported in the
thesis as `ch:timescales:weights`) were tuned on **rural_700** while the case
study ran on **base_410**. The two were tuned and evaluated on different
networks. Correcting the driver removes a real inconsistency; it is not a
preference.

Corrected in `005` (both the config line and the figure-label argument), `006`,
and `008`. Propagation verified: `build_ieee39_net` stamps
`net["ieee39_scenario"]` (build.py:390) and `add_hv_networks` reads it when
`dso_generation_scenario is None` (hv_networks.py:704-706), so setting
`cfg.scenario` is sufficient — the runner passes nothing further.

### Consequences

**Pilot buses moved, and zone 2 flipped.**

| Zone | base_410 | rural_700 | margin |
|------|----------|-----------|--------|
| 1 | 25 | 25 | 2.4 % |
| 2 | **6** | **7** | **0.5 %** |
| 3 | 20 | 20 | 1.2 % |

Bus 7 scores 0.17914 against bus 6's 0.17832. The scenario change was enough to
flip it. This upgrades the earlier caution to a demonstrated result: **the
pilot-node criterion does not identify a unique node in zone 2**, and nothing
should be read into that pilot's identity.

**It was masking a second defect.** `008` used a bare `pp.runpp`, which
converged under base_410 but does NOT under rural_700 — at that DER penetration
the base case is only well posed once the autonomous voltage layer is present.
`008` now runs the runner's chain (`tag_der_q_modes` → `install_der_q_loops` →
`PandapowerStaticPlant` with distributed slack and machine Q limits) and sources
every parameter from `make_cigre_config()`, so it drifts only if the runner's
setup SEQUENCE changes, not when a parameter value does. It aborts outright if
its scenario disagrees with 005's.

**All V1–V5 logs on disk are from base_410** and are inconsistent with the
corrected drivers. Regenerating the full seven-variant ladder is why the
overnight run exists.

**Still open:** `006_CIGRE_MONTECARLO.py:652` calls
`_gen_info_with_k("base_410")`. Untouched — it may be a generator-rating lookup
rather than a scenario selector. Read it before changing it. Affects C7, not C1.

## (5) Headless backend and plotter palette

Two fixes needed before the sweep could run unattended:

- `005:54-60` claimed *"headless rendering for the batch sweep"* while
  unconditionally forcing **Qt5Agg**, an interactive backend needing a display.
  Now honours `MPLBACKEND` exactly as `visualisation/style.py` already does, so
  `MPLBACKEND=Agg` renders headless while interactive runs keep their live
  plots.
- `CIGRE_PALETTE` was hardcoded V1–V5, so every figure would have raised
  `KeyError` once S1/O1 carried logs — a guaranteed crash at the END of a
  multi-hour sweep. Both added.

## Controlled-pair guard

`009_S1_VS_O1.py` now diffs the two variants' overrides before every run and
aborts unless the only differences are `tso_mode` and the `svr_*` keys (inert
under `tso_mode='ofo'`). Any override added to one and not the other otherwise
turns a control-law measurement into a comparison of two unrelated
configurations, and nothing else in the pipeline would notice.

## Runner edits (six, all guarded)

1. `zone_oltc_trafos` left empty under `tso_oltc_mode='local'`.
2. `_svr_tso` added; `_local_tso` widened to `in ("local", "svr")` so SVR
   inherits the local-mode TSO plumbing.
3. Tap-control install narrowed to `_install_local_taps = _local_tso and not
   _tso_oltc_local and not _svr_tso` — the relay goes only where the AVR
   setpoints stand still. The AVR pin stays on `_local_tso` alone (under SVR it
   is only the RPR's starting point).
4. `_oltc_local_active` left as `(_local_dso or _local_tso)`: with no MT relay
   installed there is nothing to rate-limit on the TS side, and the DSO coupler
   taps reach the limiter through `_local_dso` anyway.
5. New SVR dispatch block, placed **before** the OFO block and deliberately not
   threaded through it: that block is OFO-specific throughout (sensitivity
   refresh, capability messaging, coupling diagnostics, SBX hooks) and
   conditionals through it would leave the reference scheme at the mercy of
   edits meant for the proposed controller. The ~25-line apply/record tail is
   duplicated instead — the smaller risk.
6. `_miqp_acted` includes SVR, ungated on `run_tso` since the RPR acts every step.

Existing V1–V5 behaviour is unchanged: every edit is behind `tso_mode == 'svr'`
or `tso_oltc_mode == 'local'`, both defaulting off.

## Naming

Code keys stay `V1..V5` so existing `results/005_cigre/` directories remain
valid. `DISPLAY_NAMES` maps them to the thesis names
(`V1→L1, V2→L2, V3→O2, V4→O3, V5→M`); `S1` and `O1` are native.

## Status

**No result numbers exist yet. Do not quote any.**

Four full runs of the S1/O1 pair completed today and ALL FOUR were discarded:

1. machine-transformer tap windup → divergence at the gen trip;
2. zero-initialised SVR integrator → voltage collapse at t=0;
3. slack machine driven by the RPR (and inflating zone 1's `s_p` by 2.3×);
4. wrong installed-capacity scenario (`base_410` instead of `rural_700`).

Each was found before its numbers were used. The fifth attempt is the overnight
seven-variant sweep on `rural_700`, launched with `MPLBACKEND=Agg`; results go
to `results/005_cigre/<variant>/log.pkl` and the S1→O1 comparison is regenerated
with `python -m experiments.CIGRE_2026.009_S1_VS_O1 --report`.

## Open

- `_headroom` reads `res_gen`, which may be NaN under a generator trip. The
  smoke run had contingencies disabled; the full campaign does not. Untested.
- Calibration is slow over the network drive (deepcopy + ~2·(n_gen+n_der) power
  flows per zone). Fine as a one-off; would need caching for Monte Carlo.
- `tab:case3:inventory` in the thesis disagrees with the harness for zone 1
  (table: 2 machines + 2 DER; harness: 3 + 1). Zones 2 and 3 agree.
