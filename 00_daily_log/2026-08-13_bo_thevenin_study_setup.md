# 2026-08-13 — BO tuning study for the Thevenin TS–TS boundary: setup

**What:** configured and launched a Bayesian-optimisation controller-tuning
study against the **Thevenin** tie-boundary equivalent, following
`00_daily_log/2026-08-13_BO_STUDY_SETUP_PROMPT.md`. Four code changes were
needed before the study could be trusted; all are recorded below with their
reason.

Study: `thevenin_tsv_2026-08-13`, reparameterised (gauge-fixed) space,
scenario set `tune_v2`, 6 workers.

---

## 1. `tuning/_io.py` — YAML round-trip stringified mapping keys (blocking)

`save_config_yaml` routes the config through `jsonable`, which stringifies
**every** dict key (`str(k)`). A save/load round-trip therefore turned

| field | before | after |
|---|---|---|
| `tie_thevenin_k` | `{(14, 8): 2.73, …}` | `{"(14, 8)": 2.73, …}` |
| `zone_g_w_scale` | `{1: 0.3, …}` | `{"1": 0.3, …}` |
| `zone_v_setpoints_pu` | `{1: 1.03, …}` | `{"1": 1.03, …}` |

Every consumer looks these up with the original key type and **falls back
without raising**: `build_tso_local_net._k_for` substitutes
`THEVENIN_K_DEFAULT` for each missing corridor, and the runner's
`zone_g_w_scale.get(int(z), 1.0)` drops the override. Since `tune.py` takes its
baseline as a **YAML path**, a Thevenin study would have declared the ten
measured per-corridor impedances and run with none of them — i.e. tuned a
different boundary model than the one reported.

Fixed by restoring the key types on load (`_INT_KEY_FIELDS`,
`_TUPLE_INT_KEY_FIELDS`, `_restore_dict_keys`), with a regression test
(`tests/tuning/test_io.py::test_roundtrip_preserves_non_string_mapping_keys`)
and a content check in `save_baseline.py`'s verify step.

**Scope note:** this also means every earlier YAML-baseline study silently ran
with `zone_g_w_scale` inactive. Nothing in the 2026-08 campaign declared one,
so no recorded result changes; it would have mattered from here on.

## 2. `make_config()` — uniform `zone_g_w_scale` folded into the weights

`zone_g_w_scale = {1: 0.3, 2: 0.3, 3: 0.3}` multiplied the whole TSO
`params.g_w` vector **after** controller construction. Two consequences:

* a uniform factor *f* on `g_w` scales `lambda_max(M)` by `1/f`
  (`M_sym = sum_i (1/g_w_i) a_i a_i^T`), which is exactly the direction the BO
  coordinate `tso_lambda` controls through the curvature preconditioner. With
  the scale in place, a search over `tso_lambda ∈ [0.05, 1.20]` would in fact
  explore an effective `[0.167, 4.0]` — past the hard OFO bound of 2 — and the
  reported coordinate would not be the realised loop gain;
* the scale is applied after preconditioning, so it re-scales precisely what
  the preconditioner had just set.

Folded ×0.3 into the TSO block and set the knob to `None`. **Algebraically
identical** in the cascaded path, so this is a re-parameterisation, not a
re-tuning:

| field | before | after |
|---|---|---|
| `g_w_der` | 50 | 15 |
| `g_w_pcc` | 150 | 45 |
| `g_w_gen` | 5e9 | 1.5e9 |
| `g_w_tso_oltc` | 5000 | 1500 |
| `g_w_tso_shunt` | 12000 | 3600 (inert in integrator mode) |
| `zone_g_w_scale` | `{1: .3, 2: .3, 3: .3}` | `None` |

DSO weights are untouched: the runner applies the zone scale to TSO
controllers only. `precondition_g_w` stays `False` in `make_config` (the
production path); `apply_reparam_to_config` switches it on per trial, which is
where the tuned λ lives.

Reinstate `zone_g_w_scale` only for a genuinely **per-zone** (non-uniform)
re-gain — that is what the field is for.

## 3. Shunts: first excluded, then tuned (`shunt_int_gain`, 5th coordinate)

Initially the baseline was built with `--shunts off`, on the grounds that the
MSC/MSR banks are dispatched by `controller.shunt_integrator` outside the MIQP
and are therefore not a MIQP tuning degree of freedom. Revised the same day:
the integrator **gain** is tunable even though the dispatch is not, so
`shunt_int_g_w` is now searched.

**Trap found and kept documented:** `shunt_dispatch='off'` alone does *not*
remove the shunts. The runner (`multi_tso_dso.py`, "Resolve the effective
switched-shunt dispatch mode") reinterprets `'off'` as the legacy `'miqp'` mode
whenever `install_tso_tertiary_shunts=True` — which puts shunt **integers back
into the MIQP**, the opposite of the intent. `save_baseline.py --shunts off`
sets both fields together; the shipped baseline uses the default
`--shunts as-configured`, i.e. `integrator` with the banks installed.

The new coordinate is a **ratio** to the reference `shunt_int_g_w`
(±1.5 decades, log), like the other non-gain coordinates. The integrator step
is `delta = g_H / (2 g_w)`, so smaller commits sooner and in larger increments.
`shunt_int_g_w` is a member of the exact-scaling group, but that statement is
about the *common factor* only: its ratio to the rest of the weights is a
genuine, identifiable degree of freedom — it decides how much of a persistent
reactive imbalance is absorbed by the discrete banks rather than by continuous
DER/PCC authority. `g_w_tso_shunt` remains inert (MIQP path only).

**Risk to check in the results:** if the banks never commit across the tune
set, the coordinate carries no signal — the same structural inertness
`tau_der_pcc` had on `v2_undervoltage_ramp`. `TrajectoryMetrics` has no shunt
activity field, so this must be read from `zone_tso_shunt_states` in the
records, not from the objective.

## 4. Objective weight profile — TS voltage primary

Stated design intent (2026-08-13): TS voltage tracking **is** the objective;
interface-Q tracking is the means by which the cascade delivers it; DS voltage
matters with a looser tolerance; OLTC wear is a switching limit, not a cost
term.

`tuning/objectives_v2.py` gained named `PERF_WEIGHT_PROFILES`, selected with
`tune.py --perf-weights` and stamped on the study (resuming under a different
profile is refused — they are different objectives and their values are not
comparable).

| term | `calibrated_2026_08` | share | `ts_voltage_primary` | share |
|---|---|---|---|---|
| `v_rms_ts` | 1.0 | | 2.0 | |
| `v_worst_ts` | 0.5 | | 1.0 | |
| `v_band_ts` | 1.0 | **61.0 %** | 2.0 | **66.7 %** |
| `v_rms_ds` | 0.3 | 7.3 % | 1.2 | 16.0 % |
| `q_pcc` | 1.0 | 24.4 % | 1.0 | 13.3 % |
| `pcc_underutil` | 0.3 | 7.3 % | 0.3 | 4.0 % |

Interface-Q falls from 24 % to 13 % by being out-weighted, not suppressed: it
is the only term that scores the inter-layer coupling at all, and zeroing it
would leave the DSO layer's tracking job unmeasured.

OLTC behaviour stays where it belongs — constraints `g5a` (tap operations per
hour, per transformer) and `g5b` (reversals per hour, the hunting mode).
Folding wear back into the scalar is the failure mode `objectives_v2` exists to
avoid.

## 5. Constraint limits re-anchored on this plant

The `ConstraintLimits` defaults were calibrated 2026-08-04 against the
PQ-boundary plant. The reference was therefore re-measured here on all four
`tune_v2` scenarios (`--limits` was added to `tune.py`, with a resume guard —
two limit sets define two different feasible sets, hence two different
studies).

Reference at the analytic point (λ_tso = λ_dso = 0.9, τ = 1, all ratios 1),
all four scenarios feasible:

| scenario | `rho_emp_p95` | ops/h TSO | ops/h DSO | rev/h TSO | rev/h DSO | excess/step |
|---|---|---|---|---|---|---|
| `v2_quiet_spring` | 1.0132 | 0.000 | 3.214 | 0.000 | 0.000 | 0 |
| `v2_gen_trip` | 1.0132 | 0.000 | 4.018 | 0.000 | 0.804 | 8.2e-6 |
| `v2_undervoltage_ramp` | 1.0436 | 2.411 | 2.411 | 0.000 | 0.000 | 0 |
| `v2_overvoltage_rural` | 1.0093 | 0.804 | 0.000 | 0.000 | 0.000 | 0 |

Limits at margin 1.5 (`tuning/scripts/configs/limits_thevenin_2026-08-13.json`):

| limit | 2026-08-04 default | reference worst | new |
|---|---|---|---|
| `corridor_excess_pu` | 1e-4 | 8.2e-6 | 1e-4 (floor) |
| `rho_emp_p95` | 1.0 | 1.0436 | **1.5654** |
| `tap_ops_per_h` | 9.643 | 4.018 | **6.027** |
| `tap_reversals_per_h` | 1.2054 | 0.804 | 1.2054 |
| `settling_s` | 1500 | inactive | 1500 |

Two notes:

* `rho_emp_p95` had to move. Its docstring keeps it at 1.0 "regardless"
  because "the reference passes it on its own merits (measured 0.929)" — that
  premise is false on this plant, where the reference measures 1.0436. The
  quantity is the coordinator's `alpha*(lambda_max(M_ii) + sum_j ||M_ij||)`,
  which adds a rank-1 term per integer OLTC column and so over-counts a
  per-tick effect the tap cooldown bounds; gating it *relative to* the
  known-good point is the honest reading. Keeping 1.0 would have rejected the
  reference and probably emptied the box — the same defect that made
  `tap_ops_per_h = 6.0` reject 100 % of draws in the last campaign.
* Switching gets **tighter**, not looser: 9.64 → 6.03 ops/h. That matches the
  stated requirement that OLTCs not switch excessively. The reversal limit is
  unchanged at 1.205/h and the reference is nearly reversal-free (0.804/h
  worst, DSO, on `v2_gen_trip`).

A separate measurement worth recording: an earlier configuration (no tertiary
shunts, `g_w_tso_oltc = 1500`) measured `rho_emp_p95 = 2.95` on
`v2_quiet_spring`. Installing the shunts and pricing the OLTCs at 5000 brings
it to ~1.01. The contraction diagnostic is strongly sensitive to the integer
weights and to the shunt presence, and **not at all** to `tso_lambda` at init
(measured identical at λ = 0.05 and 0.10) — it only separates during the run.

## 6. All four scenarios stay in the performance aggregate

The 2026-08 campaign excluded `v2_undervoltage_ramp` from the aggregate
because, under CVaR-25 over four scenarios (which *is* the maximum), its
scalar was ~85x the others and it became the entire objective. Re-measured
here under the recalibrated p90 scales and the `ts_voltage_primary` weights:

| scenario | perf scalar | share of the mean |
|---|---|---|
| `v2_quiet_spring` | 1.4614 | 12.5 % |
| `v2_gen_trip` | 1.4374 | 12.3 % |
| `v2_overvoltage_rural` | 1.2138 | 10.4 % |
| `v2_undervoltage_ramp` | 7.5444 | **64.7 %** |

The ratio to the next scenario is **5.2x, not 85x**, and with `cvar_pct=100`
(the mean) the degenerate max-aggregator is gone. Since the ramp is the only
case in the set where TS voltage is genuinely stressed — `v_band_ts` 2.59 and
`v_rms_ts` 2.31 of its 7.54, against 0.52 and 0.001 on the quiet case — and TS
voltage tracking is the stated objective, excluding it would remove the very
condition being optimised for. It is therefore **retained**.

Carried risk, unchanged from the earlier finding: PV-based TS-DER have zero
reactive capability at that winter-evening start, so `tau_der_pcc` is
structurally inert *within* that scenario and now gets ~35 % of the aggregate
signal instead of 100 %. Check `tau_der_pcc`'s marginal identifiability
per scenario (`tuning.scripts.identifiability`, plus the
`perf__<scenario>__<term>` trial attributes) before reading its posterior; if
it comes out unidentified, re-run with `--perf-exclude v2_undervoltage_ramp`.

---

## Carried-over caveats (not changed here)

* `tso_g_q_pcc = 0` in the reference, so the TSO objective has a single active
  term and there is **no TSO-side objective trade-off in the search**. This is
  consistent with "TS voltage is the objective" and was left off deliberately.
* `FIXED_OVERRIDES` pins `int_cooldown = 1` during tuning while `make_config`
  ships 6. The g5a/g5b limits were calibrated under the same override, so the
  study is internally consistent, but the tuned point ships into a slower
  integer cadence.
* `sensitivity_update_interval = 1E6`: `H` is cached once, by design. λ is a
  loop gain **with respect to the cached model**, not the true Jacobian.
