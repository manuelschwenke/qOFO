# 2026-08-21 — the RMS plant ran open-loop from t = 60 s, and the config collapse

**What changed:** `pf/screening.py`, `pf/plant.py`, `configs/config.py`,
`configs/paramsets/` (new), `experiments/run_multi_system_ofo.py`,
`experiments/runners/multi_tso_dso.py`, `tuning_mc/stage_0_preconditioning.py`,
`tests/pf/test_screening_event_pool.py`, `tests/test_dso_v_relief_pairing.py`.

**Why:** `run_comparison_rms_cosim_qss.py` stopped producing matching
static/RMS results. Run
`results/rms_phase6_replay/0567_2026-08-21_104022` reports Gate E **PASS**
while `csv/actuator_divergence.csv` shows `DSO_4|trafo_11` at tap **13** and
`DSO_2|trafo_5` at **12** on the RMS leg against **0** on the static leg.

---

## 1. Short answer

Three defects, one of them the cause.

**(D1) Every PowerFactory parameter event after t = 60 s was written into the
past and silently dropped.** From t = 60 s onward the RMS plant received *no*
dispatch at all — no DER `qset`/`Vanchor`, no AVR `usetp`, no OLTC `ntapcmd`.
It coasted open-loop on its ElmFile profiles for 98 % of the run while the
controllers integrated against a plant that could not respond.

Introduced by commit `6126898` (2026-08-20), which added the 60 s
event-window fold to `ScreeningContext.add_param_event`:

```python
window = EVENT_WINDOW_S * math.floor(getattr(self, "_sim_time", 0.0) / EVENT_WINDOW_S)
ev.SetAttribute("time", float(t_event) - window)
```

**(D2)** The `TAPCTRL` DSL composites persist in the PF project between runs,
and `inc(x1) = ntapcmd` initialises the tap from whatever the *last* run left
in `params:0`. Four couplers carried a stale `ntapcmd = -1` and PF pulled them
to −1 at t = 0.01 s while the plant's shadow store — and therefore every
controller — read 0.

**(D3)** Nothing verified that a commanded tap actually moved.
`_verify_shunt_steps` has done this for MSC/MSR banks since 2026-07-31; the
OLTCs had no equivalent, which is why D1 presented as a plausible-looking
closed-loop trajectory instead of an exception.

### Why the taps in particular ran to the rail

`dso_gamma_oltc_q = 0`, so the DSO OLTC is driven by voltage alone, and
DSO_2/DSO_4 carry the ×20 voltage relief. They are the only actuators in the
loop with enough authority to keep pushing against an unresponsive plant. Each
interval the controller saw the same voltage error, committed one more tap
(rate-limited by `local_oltc_max_step_per_dt = 1` and
`oltc_cooldown_s_nc = 60 s`), wrote the new position into its own shadow, and
saw no response. That is integrator wind-up against a dead actuator, and it
stopped exactly where the mechanics stop: `tap_max = 13`.

The static leg is unaffected — its taps never moved because its voltages never
left the band. The comparison is therefore not "the RMS model taps too much";
it is "the RMS model received no setpoints and the only actuator still
reacting wound up".

---

## 2. Assumptions used

* Both legs are built by the same `make_cosim_config(duration)`; the archived
  `config.json` confirms `runner_static == runner_rms` field-for-field, so no
  configuration difference is in play.
* `dso_gamma_oltc_q = 0.0` and `dso_v_relief_factors = {DSO_2: 20, DSO_4: 20}`
  (archived: `dso_g_v_per_area = {DSO_2: 2e6, DSO_4: 2e6}`).
* DSO coupler taps: `tap_min/tap_max = ∓13`, `tap_step_percent = 1.25`,
  `tap_side = "hv"`, on 345/110 kV three-winding couplers.
* Measurement noise off, no contingencies, profiles on over 3600 s.
* The `c:` tap attributes (`c:nntap`, `c:n3tap_h`) are **live** during RMS and
  track the DSL output; the input attributes (`nntap`, `n3tap_h`) are not and
  read their pre-ComInc value throughout. Measured, see §3.

---

## 3. Evidence

### 3.1 The setpoints stopped

From `csv/rms_comres_full.csv`, the last change of every monitored setpoint
over the 3600 s run:

| signal | columns | first change | **last change** | n changes |
| --- | --- | --- | --- | --- |
| DER `s:qset` | 44 | 2.0 s | **42.0 s** | ≤ 3 |
| AVR `s:usetp` | 6 | 2.0 s | **2.0 s** | ≤ 1 |

At a 20 s dispatch cadence over 3600 s that is 180 dispatches, of which
**three** landed (t = 0.5, 20.5, 40.5 s). Everything from t = 60.5 s onward was
lost. The DSO voltage envelopes show it directly: the static traces are
staircases, the RMS traces are smooth.

### 3.2 The taps did nothing

Interface Q and DSO_4 bus voltages at the instants `DSO_4|trafo_11` was
commanded (t = 180, 320, 460, 600 s …):

| t [s] | `NC3W_DSO_4_t11` q_hv [Mvar] | `NC3W_DSO_4_t9` q_hv (**not** tapped) | DSO_4 bus 103 [pu] |
| --- | --- | --- | --- |
| 179 | −12.060 | −6.525 | 1.02870 |
| 181 | −12.232 | −6.588 | 1.02880 |
| 319 | −13.117 | −6.965 | 1.02910 |
| 321 | −13.209 | −7.020 | 1.02920 |

The commanded transformer and an untapped neighbour move by the same relative
amount at the same instants: that is profile drift, not a tap. A real ±1 tap on
this coupler is ~9.5 Mvar and ~0.005 pu (§3.3).

### 3.3 Reproduced and bisected on the plant

Eight consecutive +1 taps on `NC3W_DSO_4_t11`, 20 s apart, from
`export/snapshots/full_t0_20160105-0800.json`:

| commanded tap | t [s] | q_hv [Mvar] **with** the fold | q_hv [Mvar] **without** the fold |
| --- | --- | --- | --- |
| 1 | 40 | −19.131 | −19.131 |
| 2 | 60 | −28.648 | −28.648 |
| 3 | 80 | −28.844 | −38.377 |
| 4 | 100 | −28.848 | −48.158 |
| 5 | 120 | −28.848 | −57.614 |
| 6 | 140 | −28.848 | −62.029 |
| 7 | 160 | −28.848 | −66.270 |
| 8 | 180 | −28.848 | −70.980 |

Exactly the first two land — the two armed at a calculation clock below 60 s,
where the fold is the identity. `event_pool_stats()` reported
`param_used: 8, retired: 8`: the Python side believed all eight fired.

### 3.4 Which tap attribute is live

Same ladder, reading every candidate observable after each advance:

| t [s] | cmd | `c:n3tap_h` | `n3tap_h` | DSL `s:nntapin` | `c:nntap` (2W) |
| --- | --- | --- | --- | --- | --- |
| 40 | 1 | 0.9595 | 0.0 | 0.9595 | 0.9798 |
| 60 | 2 | 1.9790 | 0.0 | 1.9790 | 1.9794 |
| 80 | 3 | 2.9794 | 0.0 | 2.9794 | 2.9794 |
| 140 | 6 | 5.9794 | 0.0 | 5.9794 | 5.9794 |

`c:nntap` / `c:n3tap_h` follow the DSL output with the block's `Tmech = 5 s`
lag; after a full 20 s interval the residual is ≤ 0.041 taps. The input
attributes never move in RMS — reading them is how the earlier reading of the
ComRes export (`c:n3tap_h` "frozen at −1") was misinterpreted as a dead
wire in the 3W frame. The frames are fine; both were dumped and both carry
their `IntGrfnet`, with the 3W `Transformer` slot correctly declaring
`nntapin_h`.

---

## 3.5 Verification — run 0569 (3600 s, after the fix)

`results/rms_phase6_replay/0569_2026-08-21_145243`, same configuration and
horizon as 0567.

**Dispatch liveness.** 49 of the 50 monitored setpoint channels now change
**180 times** each — exactly one per 20 s dispatch interval — with the last
change at t = 3581 s. In 0567 they stopped at t = 42 s after at most three
changes. The 50th is the AVR on the slack/equivalent machine, which the runner
withholds by design (`AVR V-ref withheld from slack/equivalent gen(s): [9]`).

**Actuators.** All 27 rows of `csv/actuator_divergence.csv` have
`delta = 0.0` — the two runaway couplers included.

| actuator | 0567 static -> rms | 0569 static -> rms |
| --- | --- | --- |
| `DSO_4\|trafo_11` | 0 -> **13** | 0 -> 0 |
| `DSO_2\|trafo_5` | 0 -> **12** | 0 -> 0 |
| `z1[1]` | -1 -> -2 | -1 -> -1 |
| non-zero deltas | 3 | **0** |

**Endpoint errors.**

| signal | 0567 rmse | 0569 rmse | 0567 max_abs | 0569 max_abs |
| --- | --- | --- | --- | --- |
| interface_q [Mvar] | 4.1268 | **0.2809** | 13.967 | **0.6178** |
| zone_voltage [pu] | 2.472e-3 | **5.731e-4** | 4.280e-3 | **3.921e-3** |

Interface-Q tracking is ~15x better than 0567 and better than any earlier
co-simulation run, because the taps and the DER references now both track
instead of the RMS leg coasting on its profiles.

`_verify_taps` did not fire at any interval of the 3600 s run, so the guard's
tolerance is not producing false positives at the 20 s dispatch cadence.

---

## 4. What was changed

### 4.1 `pf/screening.py` — fold only off the persistent pool

The 60 s modulo is a property of **how the event object reached the running
calculation**, not of `EvtParam`:

* objects created or taken mid-run (fresh `CreateObject`, non-persistent pool)
  are read against the current window base and **must** be folded — this is the
  Ch. 9.1 battery, which pre-settles to 900 s and then arms, and which is what
  `6126898` correctly fixed;
* **pre-created persistent slots** exist before `ComInc`, so the calculation
  knows them from the start and reads their times **absolutely**.

`add_param_event` now branches on which slot it got. `EVENT_WINDOW_S`'s own
note already says the modulo was established on `EvtTap`/`EvtSwitch`
(`ElmTr2`/`ElmTr3`/`ElmShnt`), never on a pre-created `EvtParam`.

Regression test:
`tests/pf/test_screening_event_pool.py::test_param_event_window_fold_applies_only_off_the_persistent_pool`.

### 4.2 `pf/plant.py` — seed the DSL, and verify the taps

* `_preallocate_event_slots` now re-seeds each `TAPCTRL` composite's own
  `ntapcmd` (`params:0`) to the transformer's current tap, preserving that
  composite's `Tmech`. Previously only the *event slots* were seeded, so
  `inc(x1) = ntapcmd` initialised from the previous run's leftovers (D2).
* New `_verify_taps()`, called from `advance()` next to
  `_verify_shunt_steps()`. Compares `c:nntap` / `c:n3tap_h` against the shadow
  store with `_TAP_VERIFY_TOL = 0.25` taps — an order of magnitude above the
  0.041 mechanical-lag residual measured in §3.4, and still tight enough to
  catch a single lost tap. Under D1 this raises on the first interval after
  t = 60 s instead of producing a fictitious trajectory (D3).

### 4.3 Config collapse (separate request, same session)

`experiments/run_multi_system_ofo.py` carried four config factories. Three were
the same configuration with a different **weight set**, each restating the
whole block, and each applying the DSO voltage relief itself — which made the
call *order* load-bearing, because `apply_dso_v_relief` read an existing
per-area `dso_oltc` entry as its base and so squared the factor on a second
call. Two of the factories carried hand-written work-arounds for exactly that.

* **One factory.** `make_config()` now sets only the horizon, the OFO cadence,
  the objective weights, the shunt dispatch, the boundary equivalent, the
  diagnostics and the profile/contingency schedule. No command-line switches.
* **Everything else moved into `configs/config.py` defaults**, overwriting the
  previous values: `coordination_mode`, `sbx_config`, `tie_thevenin_k`, the
  shunt bank geometry and integrator settings, the OLTC cooldowns, the
  preconditioner block, `verbose`, `g_w_tso_shunt`. `sbx_config` and
  `tie_thevenin_k` use lazy `default_factory` so `configs/config.py` keeps its
  pandapower-free and `sbx_h`-free import graph.
* **The relief is a field.** `dso_v_relief_factors` is applied by a new
  `MultiTSOConfig.__post_init__`. The derivation is a pure function of
  `dso_g_v` and `g_w_dso_oltc`, so it is **idempotent** — a
  `dataclasses.replace` re-derives instead of compounding, and editing either
  base weight moves the relieved pair with it. `apply_dso_v_relief` is now a
  thin wrapper that writes the field, kept because
  `tuning_mc.stage_1_search` applies a *searched* factor after its overlay.

  **This broke Stage-1 campaign reproducibility and needed a second fix.**
  The field carries a non-empty default, and `__post_init__` installs it on
  *every* construction — including the tuning baselines, which reach
  `MultiTSOConfig` through `tuning._io.load_config_yaml`. Every Stage-1
  campaign silently gained a relief it had never had.
  `tests/tuning/test_stage1_rerun_guards.py` caught it
  (`test_campaign_default_reproduces_earlier_campaigns` plus the two
  `dso_v_authority = 1.0` cases). `tuning_mc.stage_1_search.build_config` now
  writes its own `dso_v_relief_factors` into the overlay **and** strips the one
  the baseline arrived with, via a new shared
  `configs.config.strip_dso_v_relief`. That helper removes only what the relief
  itself would have written for the areas named in the config's own
  `dso_v_relief_factors` — the per-area `dso_g_v` entry and the `dso_oltc`
  class weight — so a genuine per-area design survives. `__post_init__` cannot
  do this on its own: by the time it runs, the object already holds the NEW
  factor set and cannot know which entries the OLD one installed.

  The lesson generalises beyond this field: **a default that `__post_init__`
  materialises reaches every config in the project, including ones loaded from
  archived YAML.** A setting that is a statement about a particular plant does
  not belong to every `MultiTSOConfig` unless every consumer has been checked.
* **`scale_q` removed.** The relief scales voltage only, as originally
  specified. `dso_gamma_oltc_q_per_area` remains the instrument for the tap's
  interface-Q channel; it multiplies only the OLTC columns and so leaves every
  DER column bit-for-bit alone.
* **Alternative weight sets are data.** `configs/paramsets/{tuned,per_area}.json`
  plus a loader, holding only the fields that differ from `make_config()` plus
  their provenance. `make_config_dso_oltc_active` was **deleted rather than
  archived**: its Stage-1 candidate `ac8941a46134` *is* the runner's weight
  set, to the rounding of its full-precision literals.

Equivalence check: the rebuilt `make_cosim_config(3600)` was diffed
field-by-field against run 0567's archived `config.json`. All 191 fields agree
except the six the comparison script's own CLI sets afterwards
(`use_zonal_gen_dispatch`, the two `dso_*_scale` maps, the three deadbands) and
three serialisation artefacts. **The refactor changes no numbers.**

---

## 5. Risks and unresolved points

* **Every RMS co-simulation run between 2026-08-20 13:54 and this fix is
  invalid past t = 60 s**, including run 0567 and its Gate E PASS. The verdict
  was structurally unable to fail: the settling metric is computed on the RMS
  trajectory alone, and a plant holding constant setpoints settles beautifully.
  Re-run anything from that window.
* **Gate E did not catch an open-loop plant.** The settling check rewards a
  frozen plant. It needs a liveness assertion — e.g. that each dispatch
  interval contains at least one commanded actuator change that the plant
  executed. `_verify_taps` covers the OLTCs; DER `qset` and AVR `usetp` have no
  equivalent yet.
* **The Ch. 9.1 battery is untouched but unverified against this change.** It
  uses the non-persistent path, which keeps the fold, so it should be
  unaffected — this was reasoned from the call sites
  (`ScreeningContext(app, …)` with `persistent_event_pool` defaulting to
  `False`), not re-measured.
* **The defaults moved globally.** Every `MultiTSOConfig()` now carries
  `coordination_mode="sbx_h"`, the runner's `sbx_config`, the measured Thevenin
  table, `verbose=1` and the relief. One consumer was already caught this way
  (Stage 1, §4.3) and fixed; the remaining exposure is anything that builds a
  config by field-name overlay onto a bare `MultiTSOConfig()` and assumed the
  old values. The test evidence below is the check that was actually run, not
  proof of absence.
* **Test baseline.** The full suite minus `tests/pf` gives 785 passed /
  31 failed / 6 errors. The same seven files were then run in a clean
  `git worktree` at HEAD (5d8e0f3): **27 of those failures reproduce on
  unmodified HEAD** and are unrelated to this work — all of
  `test_tso_tertiary_shunt` (13), `test_tso_loss_objective` (4),
  `test_zip_load_model` (2), `test_tso_output_gradient` (1), `test_hierarchy`
  (1: `assert 14.4 == 50`, a stale pre-2026-08 weight expectation) and the six
  `test_dynamic_snapshot_roundtrip[base]` errors. The remaining four were the
  Stage-1 relief regression above and now pass.
  `tests/pf/test_screening_event_pool.py::test_persistent_pool_grows_admits_and_retires_events`
  also fails on `created == 2` vs `3` both before and after.
* D2's stale `ntapcmd` was inferred from the run trace plus the post-run DSL
  parameters (`params:0 = -1.0` on exactly the four couplers that read −1 at
  t = 0.01 s). The seeding fix is verified — all twelve now seed to 0.0 — but
  the original corruption's provenance is not established.
