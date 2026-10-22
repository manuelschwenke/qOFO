# Chapter 9: one structural configuration, one named parameter set

Timestamp: 2026-09-15T13:40:00+02:00
Branch: `weights-reset-2026-09-10`
Reason: the author required that every Chapter-9 experiment resolve its controller as
`apply_paramset(make_config(), "ch9_selected")`. Auditing that requirement showed the
Chapter-9.1 response experiments were building a *different* configuration object, and
had been unable to start at all since the 2026-09-10 weight reset.

No simulation was launched. This entry records code and provenance changes only.

## 1. What was actually wrong

### 1.1 Two config lineages, fourteen differing fields

`_ch9_selected_design.build_selected_config()` built on a bare `MultiTSOConfig()`.
Everything else in the chapter — `ch_9_3_stage2_screen.py`, `ch_9_3_compare_runs.py`,
`experiments/helpers/rms_cosim_config.py`, `run_multi_system_ofo.main()` — built on
`make_config()`. Measured diff of the two resolved configs at the same paramset:

| field | `make_config()` | bare `MultiTSOConfig()` |
|---|---|---|
| `install_tso_tertiary_shunts` | `True` | `False` |
| `shunt_dispatch` | `integrator` | `off` |
| `tie_boundary_equivalent` | `thevenin` | `pq` |
| `local_sensitivities_tso` / `_dso` | `True` | `False` |
| `zone_v_setpoints_pu` | 1.03 in zones 1–3 | `None` |
| `dso_der_scale` / `dso_load_p_scale` | `{}` (symmetric) | `DSO_3: 2.0` |

(plus `n_total_s`, `start_time`, `use_profiles`, `contingencies` and two `live_plot_*`
flags, which `ScenarioSpec.overlay_on` or the co-sim envelope overwrite anyway.)

Consequence: Experiments C/E and the `T_TS` sweep ran with the MSC/MSR banks **absent**
— so `shunt_int_g_w = 0.03` was inert there regardless of the parameter set — on a PQ
rather than Thevenin boundary equivalent, with no local sensitivity refresh, no zone
voltage setpoints, and on the doubled-DSO_3 network that the Stage-II weight search
never saw. `N_inner` measured under those conditions is not `N_inner` of the controller
Chapter 9.3 reports.

### 1.2 The Chapter-9.1 entry points could not run

`ch_9_1_ninner_isolated_sts.py:1295` and `ch_9_1_ts_period_sweep.py:952` print
`design['campaign']`, `design['candidate_key']` and `design['verified']`. The
2026-09-10 rewrite of `_ch9_selected_design.py` replaced that provenance schema with
`{paramset, title, fields, paramset_provenance, frozen_network}` and did not update the
readers, so both scripts raised `KeyError: 'campaign'` before submitting a single job.

### 1.3 The selected parameter set was not the one being loaded

`SELECTED_PARAMSET` still said `"stage1_start"` — the Stage-I *analytic* point — while
`ch9_selected` had existed since 2026-09-11. `run_multi_system_ofo.main()` worked around
this with a hand-edit, `apply_paramset(make_config(), 'ch9_selected')#SELECTED_PARAMSET)`,
so the runner and the Chapter-9.1 scripts silently disagreed about which controller the
chapter runs.

### 1.4 The RMS co-simulation defaulted to a configuration that cannot run

`--paramset` defaulted to `None`. Since the weight reset `make_config()` is
weight-incomplete and `MultiTSOConfig.require_weights()` raises in the runner, so every
RMS run without the flag aborts — and **none** of the eight archived `.ps1` launchers
pass it. Its help text advertised a parameter set named `ch9_frozen` that has never
existed, and quoted weights from the pre-reset gauge.

`--dso-oltc-switch-cost 200`, which those eight launchers *do* pass, overrides
`g_w_dso_oltc` after the parameter set — 200 against the selected 15.7164, a factor of
12.7 — and because it is a plain assignment rather than `dataclasses.replace`,
`__post_init__` does not re-run and the derived per-DSO voltage relief is left on the
parameter set's base while the weight itself has moved.

## 2. Changes

| file | change |
|---|---|
| `experiments/ch_9_parameter_selection/_ch9_selected_design.py` | rewritten. Base is now `apply_paramset(make_config(), paramset)`. `SELECTED_PARAMSET = "ch9_selected"`; new `INITIAL_PARAMSET = "stage1_start"`. Provenance regained `campaign` / `candidate_key` / `verified`, derived from the parameter set rather than restated. New `resolved_manifest()` returns the complete resolved config. |
| `ch_9_1_ninner_isolated_sts.py` | new `--paramset` (default `ch9_selected`), `--bank` and `--symmetric-network`; all three travel in the job dict so each worker resolves the controller and the window bank explicitly instead of defaulting. Design banner fixed and extended with the structural settings and the network. Writes `resolved_config.json` per run; `summary.md` now states the network mismatch in words. |
| `ch_9_1_ts_period_sweep.py` | same treatment, including the `--reaggregate` path. |
| `experiments/run_multi_system_ofo.py` | `main()` uses `SELECTED_PARAMSET` again; the hard-coded literal is gone. `make_config_ch9_frozen` docstring corrected: it is now `make_config()` + the weights + the frozen network, and says which of those the Stage-I/II campaign did not share. |
| `experiments/helpers/rms_cosim_config.py` | new `DEFAULT_PARAMSET = "ch9_selected"`, used as the `--paramset` default; stale `ch9_frozen` help replaced. `--dso-oltc-switch-cost` now prints what it overrides and that the relief is not re-derived. |

Method: the controller identity is passed as data (a paramset name in the job dict),
never inherited from a module constant at the point of use. The structural configuration
has exactly one definition, `make_config()`, and a parameter set states only what differs
from it.

### 2.1 Two experiment-design decisions, taken by the author 2026-09-15

**Network: doubled DSO_3.** `build_selected_config(apply_frozen_network=True)` is the
default; `--symmetric-network` opts out. The Chapter-9 batch is declared on that network
and the PowerFactory project stores it.

This leaves **one deliberate mismatch**, which the chapter must state rather than hide:
the Stage-I/II weight campaign ran on the *symmetric* model, because `make_config()` sets
no DSO multipliers. The response experiments therefore evaluate the selected weights on a
network the search never saw. The run banner, `resolved_config.json` and the `summary.md`
provenance block all record which network was used and say so in words. The other five
physical differences of §1.1 are gone.

**Window bank: W1–W6.** New `--bank {conditions,design12}`, defaulting to `conditions` —
the six thesis conditions, which is also what F1 and F2 are specified on. W5 is
structurally zero-capability and is excluded by `DEAD_ZONE_STRATA`, so **five windows are
measured**, matching what the chapter already says about Experiment B. The grid is
5 × 4 DSO × 3 interfaces × 2 directions = **120 cases**, against 240 in the
`design12` archive. That is a scope change and must be declared; in particular the
archived 227/240 = 94.6 % coverage figure may not be quoted alongside a 120-case result.
`--bank design12` reproduces the old bank as a coverage check.

## 3. Verification

- `ch_9_1_ninner_isolated_sts.py --self-test` — 32/32 PASS (re-run after the bank change).
- `ch_9_1_ts_period_sweep.py --self-test` — 22/22 PASS (likewise).
- `pytest tests/tuning/test_parameters.py tests/test_dso_v_relief_pairing.py
  tests/tuning/test_io.py tests/tuning/test_runner.py` — **34 passed, 2 skipped**, plus
  **3 passed** in `test_runner.py`, which simulates and took 53 min.
- `apply_paramset(make_config(), "ch9_selected")` resolves weight-complete at
  `T_TS = 180 s`, `T_STS = dt = 20 s`, `rural_700`, `shunt_int_g_w = 0.03`.
- Both Chapter-9.1 entry points reach job submission on every combination:

  ```
  [design] paramset ch9_selected candidate 399e30d3b60b from .../ch_9_3_select.py
  [design] base ...make_config; shunts=True/integrator boundary=thevenin network=DSO_3 x2
  [design]   -- the Stage-I/II weight campaign ran SYMMETRIC; this run does not share its network.
  [ninner] bank conditions: 6 window(s)
  [ninner] excluded 1 window(s) ... ['W5']
  ```

  and with `--paramset stage1_start`, `--bank design12`, `--symmetric-network` the banner
  changes accordingly.
- **End-to-end simulation**, one period-sweep case on the unified configuration
  (`d_quiet_spring @ T_TS = 180 s`, `--paramset ch9_selected`): 270 records, 372 intervals,
  583 s wall; ρ median 0.012, ρ p95 0.015, n_k median 0, censoring 0.00, CAIR width
  193.1 Mvar. One window is not a result — this verifies the machinery, and fixes the cost
  model at **2.16 s/QSS step**, from which C and E are ~30 min each at 16 workers.

## 4. Evidence found while auditing (not produced by this change)

### 4.1 `ch9_selected` is `399e30d3b60b`, and the file is faithful to the report

`results/THESIS_ch9_3_stage2/refine_evented/selection_report.json` selects
`399e30d3b60b`; its nine weights match `configs/paramsets/ch9_selected.json` exactly.
The rival `a9555cdf78ba` named in `ch9_alt_lowwear.json`'s description was **rejected**
in the evented bank for 5 hunting reversals. That description is stale: it was written
against the non-evented `refine` bank, and `ch9_alt_lowwear.json` now carries a `fields`
block byte-identical to `ch9_selected.json`.

### 4.2 The selection bank had the shunts inert; a later bank did not

No window record under `refine_evented/` carries a `shunt_*` key (0 of 162):
`shunt_summary()` was added to `ch_9_3_stage2_screen.py` after that campaign ran. So the
campaign that selected `ch9_selected` has **no recorded shunt activity at all**.

`results/THESIS_ch9_3_shunt_live/` (2026-09-11 20:12, `--paramset ch9_selected
--evented --hours 12`) is a complete six-window re-run with `shunt_int_g_w = 0.03` live.
Re-scoring it through `tuning.stage2_search.aggregate` + `check_limits` with the same
`Limits()`:

| bank | e_v(TS) | e_v(DS) | worst taps/day | n_hunting | feasible |
|---|---|---|---|---|---|
| `refine_evented` (shunts inert) | 0.0016515 | 0.0048507 | 4.00 | 0 | yes |
| `shunt_live` (`shunt_int_g_w=0.03`) | 0.0013994 | 0.0050374 | 4.00 | 0 | yes |

So **the selection verdict survives turning the shunts on**, and e_v(TS) improves 15.2 %
at 3.8 % worse e_v(DS).

But the shunt activity itself is unscreened — `aggregate()` carries no shunt field and
`Limits` has no shunt bound — and it is not small (12 h counts, `399e30d3b60b`):

| window | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| commits | 8 | 12 | 17 | **49** | **112** | 15 |
| reversals (<2 h dwell) | 2 | 6 | 10 | **32** | **80** | 9 |

W4 is a *ranked* window: 98 commits/day and 64 reversals/day on a bulk MSC/MSR bank,
against a 30 ops/day budget for a tap changer. `shunt_summary`'s own docstring calls a
bank reversing inside two hours undesirable.

### 4.3 W5's exclusion rationale no longer holds

`W5` is excluded from the ranking aggregate as zero-capability, on the stated grounds
that "with nothing to allocate, every weight coordinate is structurally inert". With the
shunt integrator live that is no longer true: W5 is where the shunts are the *only*
reactive actuator, and it shows 112 commits / 80 reversals plus 24 tap-hunting
reversals and ~50 taps/day. The exclusion is now hiding the window the shunt gain
most affects.

## 5. Scope and assumptions

Controlled outputs unchanged: nodal voltages and TS–STS interface Q. Actuators unchanged:
AVR references, DER Q, OLTCs, MSC/MSR. Controllers still see the plant only through their
cached sensitivities; nothing here changes what is cached or when it refreshes — it
changes only whether `local_sensitivities_*` is on, and it now is, in every Chapter-9
experiment rather than in some of them.

The network and the window bank are **decisions**, not neutral refactors, and both were
put to the author and answered on 2026-09-15 (§2.1). The doubled-DSO_3 network is kept, so
the response experiments and the weight-selection campaign do *not* share a network; that
has to be stated in the chapter rather than assumed away.

## 6. Risks / unresolved

1. **The Chapter-9.1 archive is not transferable.** Every archived `ninner` and
   `ts_period_sweep` result was measured on the old lineage — shunts off, PQ boundary
   equivalent, local sensitivities off. The network now matches that archive again, but
   the other four differences do not, so C and E must be re-run; nothing can be
   relabelled. The bank also changed from 12 windows to W1–W6.
2. **D and C/E are measured on different networks, by decision.** D's aggregate is
   symmetric, C/E are doubled-DSO_3. They may be reported together only with that
   difference stated. `--symmetric-network` produces the like-for-like comparison for
   ~30 min of compute per leg if the chapter needs it.
3. **Shunt wear is measured but not governed.** Either a shunt limit joins `Limits` and
   the selection is re-scored on `shunt_live`, or the thesis states explicitly that bulk
   shunt switching is out of the selection's scope. It currently does neither.
4. **W5's exclusion needs restating** for the shunt-live configuration (§4.3).
5. **`ch9_alt_lowwear.json` is a duplicate** of `ch9_selected.json` with a contradicting
   description. It should be corrected or deleted before it is cited.
6. **The eight archived `.ps1` launchers remain uncorrected** — they pass
   `--dso-oltc-switch-cost 200` and no `--paramset`. They were deliberately left alone as
   the record of past campaigns; new campaigns must not reuse them. See
   `experiments/ch_9_parameter_selection/RUNBOOK_ch9.md`.
7. **F1's core measurement is not yet instrumented.** `csv/rms_der_raw.csv` carries only
   `qDER_*` and `uDER_*`; the Q(V) anchor (`Vanchor` on the PF pre-controller,
   `qv_vref_anchor_pu` in the static plant), the filtered dead-zone input and the
   unsaturated droop command are not exported. `|U_filtered − U_anchor|` between *actual*
   anchor updates cannot be measured from current runs.
