# 2026-08-03 — `prepare_load_contingencies` mutates its input, so every reused scenario set silently loses its load-contingency scenarios from the second run onward

**Reason.** Stage 3 of the BO re-tuning campaign
([handover](../../tuning/HANDOVER_bo_retuning.md)) reported
`status=bracketed` for both OLTC weights, but the JSON carried `n_failed: 2` of
4 scenarios on **17 of 18 probes**. Stage 2 showed the same shape:
`draw 0: 3/3 feasible`, then every subsequent draw `2/3`. Stage 1's excitation
gate had admitted all four scenarios minutes earlier.

## Measurement

`infeasible_reason` is only ever `pf_failure`, `non_finite_metric` or
`empty_log` (`metrics.py:1008-1012`), so these were simulation breakdowns, not
constraint violations. Running the *same* reference config and the *same*
params twice in one process:

```
=== pass 1 ===
  v2_quiet_spring        feasible=True   recs=225
  v2_gen_trip            feasible=True   recs=225
  v2_undervoltage_ramp   feasible=True   recs=225
  v2_overvoltage_rural   feasible=True   recs=225

=== pass 2 ===  (identical config, identical params)
  v2_quiet_spring        feasible=True   recs=225
  v2_gen_trip            feasible=True   recs=225
  v2_undervoltage_ramp   feasible=False  recs=0   empty_log
  v2_overvoltage_rural   feasible=False  recs=0   empty_log
```

The error was invisible because `run_one` swallows every exception into
`RunResult.failure_reason` and redirects stdout/stderr into `StringIO` buffers
that it then discards (`tuning/runner.py:106-117`). Printing `failure_reason`:

```
ValueError: Load contingency at minute 20: action='connect' with explicit
element_index=122 is a contradiction — the row already exists.
  experiments/helpers/contingency.py:71
  <- experiments/runners/multi_tso_dso.py:1746
       prepare_load_contingencies(net, contingencies, verbose=verbose)
```

## Mechanism

`prepare_load_contingencies` *resolves* events by writing the resolved index
back into them — `ev.element_index = key_to_index[key]`
(`contingency.py:161`, also `:109`, `:176`). The runner obtained its list with

```python
contingencies = list(config.contingencies) if config.contingencies else []
```

which copies the **list** but shares the `ContingencyEvent` objects (a plain
non-frozen dataclass) with the caller's config. Therefore:

1. Run 1 — event has no `element_index` and `action='connect'`, so it takes the
   mode-3 legacy dormant-load path, creates load row 122, and writes
   `ev.element_index = 122` **into the caller's object**.
2. Run 2 — the same `ScenarioSpec` is reused, so the event now presents an
   explicit index *together with* `action='connect'`, and the contradiction
   guard at `contingency.py:69-76` correctly rejects it.

The guard is right; the defect is that the resolver is **not idempotent** and
mutates data it does not own. Only scenarios using mode-3 `connect` load events
are affected — hence exactly `v2_undervoltage_ramp` (load ramp) and
`v2_overvoltage_rural`; `v2_quiet_spring` has no events and `v2_gen_trip` uses
`element_type='gen'`.

## Change

`experiments/runners/multi_tso_dso.py:1647` — deep-copy the events at the
ownership boundary, so resolution happens on the runner's private copies:

```python
contingencies = (
    [copy.deepcopy(ev) for ev in config.contingencies]
    if config.contingencies else []
)
```

Verified: three identical passes, 12/12 runs feasible at 225 records each.
Test suite 122 passed, 1 pre-existing unrelated failure
(`test_hierarchy.py`, asserts `g_w_pcc == 200`).

## Why the excitation gate could not catch this

`tuning/scripts/audit_design_set.py` runs each `ScenarioSpec` **exactly once**,
so the mutation never fires. The gate is structurally blind to the defect, which
is why it passed 4/4 (reproducing the handover's table exactly) while the two
hour-scale stages were losing half their design set. Any gate that validates
scenarios one-shot cannot certify a driver that reuses them.

## What it invalidated

- **Stage 3** — discarded. 17 of 18 probes took the median over 2 surviving
  scenarios, which is also why both weight classes returned the identical
  `achieved = 4.821 ops/h`.
- **Stage 2** — killed at ~20/65 draws. Every non-reference draw dropped
  `v2_undervoltage_ramp`, which per Stage 1 is the **only** scenario with
  material tap excitation (19 TS / 9 DS taps against 2–3 elsewhere). Its
  dead-term audit would therefore have declared the tap/switching cost terms
  dead, and handover §4 instructs acting on that by removing them. A bug-induced
  false positive would have deleted legitimate cost structure.
- **Stage 1** — unaffected, see above.

## Hypothesis worth testing: this may be the real cause of the historical unidentifiability

[`2026-07-31_bo_tuning_audit.md`](../07_2026/2026-07-31_bo_tuning_audit.md)
concluded that the OLTC weights were unidentifiable because the *search space /
parameterisation* was wrong, evidenced by taps being frozen in **77 %** of runs.
This defect produces exactly that signature by a different mechanism: a BO
driver builds its scenario set once, so trial 1 sees the full set and every
later trial loses the tap-carrying scenarios. The weights then have no leverage
on any objective — not because the parameterisation is wrong, but because the
plant silently stopped being excited.

Open, not asserted. Confirming it requires checking whether `design_set()`'s
tap-active scenarios also use mode-3 `connect` events, and whether the 77 %
figure matches the fraction of runs that were not first-in-process. If it does,
the conclusion of that audit — and the rationale for the whole reparameterisation
— needs revisiting.

## Separately: the switching target is not on the achievable grid

Tap rates are quantised: one operation over the 1.2444 h window (224 records ×
20 s) = 0.8036 ops/h. Observed ladder values are exact multiples —
`9.643 = 12 ops`, `4.821 = 6 ops`, `2.411 = 3 ops`, and half-integers such as
`3.616 = 4.5 ops` arise because the median of an even-sized scenario set averages
two order statistics.

A target of 6 ops/h corresponds to **7.47 operations**, which no trajectory can
produce. The nearest achievable points are 7 ops = 5.625 ops/h (6.3 % off) and
8 ops = 6.429 ops/h (7.1 % off). With `tol_rel = 0.2` the bisection accepted
6 ops = 4.821 ops/h — 19.65 % off, i.e. inside the band by 0.35 pp — and
reported `within_tolerance: true`. That is an artifact of a wide band, not
convergence. The re-run uses `--tol-rel 0.1`, whose band `[5.4, 6.6]` admits both
grid neighbours but excludes 4.821.

Also worth noting for interpretation: the hand-tuned `g_w_tso_oltc = 5000` yields
~1.2 ops/h, so a 6 ops/h target *loosens* TSO switching roughly 4×. The
maintenance budget does not bind at the reference; §5's framing of it as a
constraint to respect does not match the measurement.

## Open

- **Audit the other drivers.** `tuning/validate.py`, `run_tuning.py`,
  `run_tuning_parallel.py` and the Monte-Carlo paths all reuse a scenario set.
  The fix is at the runner boundary so it should cover them, but this needs
  confirming, plus a regression test that calls `run_multi_tso_dso` twice with
  mode-3 `connect` events and asserts both logs are non-empty.
- **`run_one` hides failures.** Swallowing the traceback into `failure_reason`
  and discarding stdout/stderr is why a fatal, 100 %-reproducible `ValueError`
  presented as a quiet `2/3 feasible` for hours. Consider surfacing
  `failure_reason` in the drivers' per-draw output.
- **§5 vs §0.3.** §5 says write the calibrated weights into `make_config()`;
  §0.3 designates `make_config()` as the fixed benchmark and warm start. Doing
  both means Stage 5's gauge is no longer the hand-tuned point. Undecided.
