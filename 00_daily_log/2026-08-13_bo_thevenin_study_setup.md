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

## 3. Shunts excluded from the tuned plant

The MSC/MSR banks are dispatched by `controller.shunt_integrator`, outside the
MIQP, on a 1800 s dwell; they are not a tuning degree of freedom. Baseline
therefore built with `--shunts off`.

**Trap:** `shunt_dispatch='off'` alone is *not* enough. The runner
(`multi_tso_dso.py`, "Resolve the effective switched-shunt dispatch mode")
reinterprets `'off'` as the legacy `'miqp'` mode whenever
`install_tso_tertiary_shunts=True` — which would put shunt **integers back into
the MIQP**, the opposite of the intent. `save_baseline.py --shunts off` sets
both fields together and documents why.

Consequence to carry into the report: the tuned point is optimal for a plant
**without** tertiary shunts. Shunt engagement is inherited from the reference
when it ships, and `shunt_int_g_w` (currently being hand-tuned) is not
identified by this study.

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
