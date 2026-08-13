# 2026-08-03 — Starting the BO re-tuning campaign on the always-on machine: three blockers, two of them silent

**Reason.** Executing [`docs/tuning/HANDOVER_bo_retuning.md`](../../tuning/HANDOVER_bo_retuning.md)
on the server (`Z:\Python_Projekte\qOFO_GH`, env
`F:\python_environments\qOFO_clean\python.exe`, 3.12.13). The handover states the
work is "implemented, unit-tested and smoke-run" and that what remains is
compute. Three defects surfaced before any hour-scale stage produced a number.

Environment confirmed first: GUROBI resolves (academic licence, expires
2027-07-21), so no SCIP fallback and results stay comparable to the recorded
runs. 20 physical cores / 275 GB RAM; the campaign is pinned to 12 cores at the
user's instruction.

## 1. `np.trapz` compat shim raised the error it was written to prevent

`tuning/metrics.py:316` read:

```python
trapz_fn = getattr(np, "trapezoid", np.trapz)
```

The default argument of `getattr` is evaluated **eagerly**, and NumPy 2.x
*removed* `np.trapz` rather than deprecating it — the in-code comment asserts it
"raises a DeprecationWarning", which is wrong for 2.x. So on this env
(numpy 2.4.6) the fallback expression itself raised
`AttributeError: module 'numpy' has no attribute 'trapz'`.

Effect: **9 failures** in `tests/tuning/test_metrics.py`, including
`test_diverged_log_is_infeasible_not_cheap` — i.e. the regression guard for the
historical defect where a diverged trajectory scored *better* than 35–43 % of
converged ones and divergence became a rewarded search direction. The ITAE path
is used by every objective, so this would have failed inside the first trial of
any stage that computes a cost.

Fixed with a lazily-evaluated branch:

```python
trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
```

## 2. Stage 3 could not run at all, and the tests could not have caught it

`python -m tuning.scripts.calibrate_switching` died on its first probe:

```
KeyError: "Missing BO params: ['dso_g_v', 'g_q', 'g_v', 'g_w_der',
           'g_w_dso_der', 'g_w_dso_oltc', 'g_w_pcc', 'g_w_tso_oltc']"
```

`tuning/bisect_switching.py:153` called `run(params or {}, ...)`. `run_one`
overlays parameters through `parameters.apply_to_config`, which by contract
requires *exactly* the 8 `BO_DIMS` keys and raises `KeyError` on a partial dict.
An empty dict therefore fails on all 8.

**Why no test caught it:** all five bisection tests in
`tests/tuning/test_reparam.py` inject `runner=_fake_runner(...)`, whose stub
reads `cfg.g_w_dso_oltc` directly and never touches `params`. The bisection
*logic* is well covered; the single line that invokes the real simulator was
never executed. This is what "smoke-run" did not cover.

**The non-obvious half.** The naive repair — seed the dict from
`params_from_config(baseline_cfg)` — is wrong and fails *silently*. The swept
weight is itself one of the 8 BO dimensions, so the overlay would overwrite the
value set by `dataclasses.replace` and every rung of the ladder would run at the
baseline weight. The measured rate would then be constant in `g_w`, and the
bisection would report a confident `plateau_high` / `plateau_low` with no error
raised — precisely the status the handover §5 warns must not be misread as a
tuned value. Structure of the fix:

```python
cfg = dataclasses.replace(baseline_cfg, **{field: float(g_w)})
probe_params = params_from_config(cfg)   # seeded from the *replaced* cfg
if params:
    probe_params.update(params)
probe_params[field] = float(g_w)         # the swept field always wins
```

Regression test added
(`test_probe_hands_the_runner_a_complete_param_set_with_the_swept_value`): it
calls the real `apply_to_config` inside the runner stub to pin completeness, and
asserts the probed weights actually reach the runner and differ across rungs, so
an inert ladder fails loudly instead of returning a plausible status.

Verified against the real runner (1 scenario, `--max-iter 1`) — the ladder now
responds to `g_w`:

```
g_w_tso_oltc:  1 -> 9.643    1e5 -> 0.000    316.2 -> 9.643
g_w_dso_oltc:  1 -> 90.000   1e5 -> 0.000    316.2 -> 0.000
```

## 3. Non-atomic ceilings cache (needed for concurrent stages)

Stages 2 and 3 are prescribed serially (~6 h + ~5 h) but are single-process and
read only `--baseline`; neither consumes the other's output. Running them
concurrently is faithful to the prescribed semantics — in the serial order Stage
2 also sees the pre-Stage-3 baseline — and halves wall clock.

The one coupling is `tuning/ceilings.py`. Its cache key deliberately excludes all
`g_w_*`, so both stages derive the *same* keys, and the write was a plain
`open("w")` + `json.dump`. Two processes missing the same key could leave a
reader seeing a half-written JSON. Now writes to a per-PID temp file and
`os.replace`s it; the payload is deterministic, so whichever replace lands last
is equivalent.

## 4. Units: `ops_per_day` is per *hour*

`calibrate_switching`'s `--target-tso` / `--target-dso` help said "Tap
operations/day", contradicting handover §5. The consumed metric is
`tap_ops_per_h_tso` (`bisect_switching.py:75-78`) and `metrics.py:470-473`
computes `np.max(ops) / (duration_s / 3600.0)` — per hour, worst transformer in
class. The handover's `--target 6` is correct; the `ops_per_day` naming
throughout `bisect_switching.py` is vestigial from the metric version that made
the reference's ordinary 4 taps/h read as a pathological "96/day". Help strings
corrected; the field rename is deferred (pure rename, separate change).

## Stage 1 result — gate passed, and it reproduces exactly

```
scenario                network      recs  tapTS  tapDS  minQres  peak|dV|   verdict
v2_quiet_spring         base_410      225      2      4    0.240    0.0241     ADMIT
v2_gen_trip             base_410      225      3      8    0.237    0.0241     ADMIT
v2_undervoltage_ramp    base_410      225     19      9    0.080    0.0568     ADMIT
v2_overvoltage_rural    rural_700     225      3      1    0.180    0.0226     ADMIT
```

Every column matches the handover's 2026-08-03 table exactly, on a different
machine, Python patch version and NumPy — a stronger reproducibility statement
than the table alone. Both OLTC classes move in 4/4 scenarios, so neither weight
is unidentifiable in the way that wasted every previous campaign.

Test suite after the fixes: **121 passed, 1 failed**. The failure is the
pre-existing, unrelated
`tests/tuning/stability_certificate/test_hierarchy.py::test_default_factory_reads_run_multi_system_ofo_parameters`
(asserts `g_w_pcc == 200`, working tree sets 80). Note the handover claims
`-m "not slow"` excludes it; it does not, it merely fails.

## Change

- `tuning/metrics.py` — lazy `np.trapezoid` / `np.trapz` selection.
- `tuning/bisect_switching.py` — `_probe` now hands `run_one` a complete BO
  param set seeded from the replaced config, with the swept field forced last;
  imports `params_from_config`.
- `tuning/ceilings.py` — atomic cache write (temp + `os.replace`).
- `tuning/scripts/calibrate_switching.py` — `--target-*` help corrected to
  per-hour, with the ~19× extrapolation trap named.
- `tests/tuning/test_reparam.py` — regression test for the `_probe` contract.

## Deviation from the handover

`--n-draws 64` instead of 40 for Stage 2 (user decision). At n=40 scipy warns
that Sobol balance properties require a power of two; this sample sets
`MetricScales` **and** `ConstraintLimits` for the whole campaign and is computed
once, so the imbalance was not worth carrying. Cost ~9.5 h instead of ~6 h, and
it overlaps Stage 3.

## Open

- **Stage 3 resolution.** Both smoke ladders look step-like (≈9.6 ops/h flat,
  then 0) rather than smoothly monotone. If that survives the full run
  (`--max-iter 8`, 4 scenarios, median) then no weight achieves 6 ops/h,
  `within_tolerance` stays False, and the §5 framing of this weight as a
  *stated operational requirement* does not hold — the rate would not be
  continuously controllable by `g_w`. To be judged from the full ladder, and
  reported rather than tuned around.
- **Stage 5 has no driver.** Must be written before the holdout, which may be
  evaluated only once.
- **QSS only.** `tuning/` never passes a `plant_factory`, so nothing here is
  validated against RMS dynamics — most consequential for exactly the two OLTC
  weights Stage 3 calibrates, given
  `2026-07-30_rms_oltc_taps_never_fire_midrun.md`.
- `TaskStop` on a `nohup`'d stage kills the shell but orphans the Python child;
  one orphaned Stage 2 process had to be killed by PID. Later stages are
  launched without `nohup`.
