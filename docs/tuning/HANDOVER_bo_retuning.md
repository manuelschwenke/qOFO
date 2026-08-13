# Handover — BO controller re-tuning (run on the always-on machine)

**Written 2026-08-03.** Everything below is implemented, unit-tested and
smoke-run. What remains is compute: roughly **25 h serial / 15 h with the
parallel driver**, in four stages, each gating the next.

Background and evidence: [`docs/daily_log/07_2026/2026-07-31_bo_tuning_audit.md`](../daily_log/07_2026/2026-07-31_bo_tuning_audit.md).

---

## 0. Read this first — three things that will bite

### 0.1 This is quasi-static (QSS) only

Every command below runs the **`PandapowerStaticPlant`** quasi-static plant.
`run_multi_tso_dso` accepts a `plant_factory` for the PowerFactory RMS plant
(`pf/plant.py`, used by `experiments/run_rms_phase6_replay.py`), but the whole
`tuning/` package never passes one — verified by grep, there is no `plant_factory`
reference anywhere in it.

**Consequence:** the tuned weights are optimal *for the QSS model*. They are not
validated against RMS dynamics, and the recent Gate-E RMS work
(`2026-07-31_rms_tap_control_gate_e_result.md`,
`2026-07-30_rms_oltc_taps_never_fire_midrun.md`) suggests tap behaviour in
particular differs between the two. Treat an RMS replay of the tuned point as a
required follow-up, not an optional one — especially for the two OLTC weights,
whose whole calibration is a switching-rate target.

### 0.2 The old studies are not resumable, by design

`tuning/tune.py` refuses to resume a study whose search-space fingerprint differs
from the current one. All 19 historical studies predate fingerprinting and every
IEEE-39 one carries a 9th parameter (`tso_g_q_tie`) that is not a
`MultiTSOConfig` field any more. **Use a fresh `--study-name`.** Do not reach for
`--allow-schema-drift` to get around this; it exists for the case where you have
verified the spaces match.

### 0.3 The acceptance criterion is beating your hand-tuned point

`experiments/run_multi_system_ofo.py::make_config()` is the reference. It is the
gauge, the warm start, and the benchmark. **If the BO optimum does not beat it on
the holdout, keep the hand-tuned point** — the deliverable is then the
methodological evidence for a setting you already had, which is still worth
having and is the honest thing to report.

---

## 1. Environment

```
python:  C:\Users\Manuel Schwenke\.conda\envs\qOFO_clean\python.exe   (3.12.12)
cwd:     the repo root (scripts resolve their own paths, but cwd must be the repo)
solver:  GUROBI must resolve.  If it falls back to SCIP the results are NOT
         comparable to anything else — miqp_solver.py records SCIP returning
         `optimal_inaccurate` on 54/60 DSO solves.  The solver name is stamped
         into the output metadata; check it.
```

Sanity check before committing an overnight run:

```bash
python -m pytest tests/tuning/ tests/test_gw_precondition.py -q -m "not slow"
```

Expect **~108 passed**. One pre-existing failure lives in
`tests/tuning/stability_certificate/test_hierarchy.py` (it asserts
`g_w_pcc == 200` against a working-tree edit that sets 80) — unrelated to this
work, and excluded above.

---

## 2. Stage 0 — regenerate the baseline (seconds)

Only needed if `make_config()` has changed since 2026-08-03.

```bash
python -m tuning.scripts.save_baseline
```

Writes `tuning/scripts/configs/baseline_ieee39.yaml` and **verifies the YAML
round-trip**. That check is not ceremonial: `save_config_yaml` →
`load_config_yaml` used to silently turn `sbx_config` and `measurement_noise`
into plain `dict`s, so `cfg.sbx_config.k_sched` would raise `AttributeError` deep
inside a run. It was latent only because no baseline had ever carried
`coordination_mode="sbx_h"`.

---

## 3. Stage 1 — excitation gate (~12 min)

```bash
python -m tuning.scripts.audit_design_set --set tune_v2 \
    --csv results/tuning/excitation_audit.csv
```

**Exit code 0 is required before proceeding.** This is the check whose absence
made both OLTC weights unidentifiable in every previous campaign: taps were
frozen in 77 % of runs, so those two weights had no leverage on any objective and
every trial spent on them was wasted.

Expected (measured 2026-08-03, 4 scenarios):

| scenario | tap TS | tap DS | min Q-reserve | peak \|dV\| |
|---|---|---|---|---|
| `v2_quiet_spring` | 2 | 4 | 0.240 | 0.0241 |
| `v2_gen_trip` | 3 | 8 | 0.237 | 0.0241 |
| `v2_undervoltage_ramp` | **19** | **9** | **0.080** | 0.0568 |
| `v2_overvoltage_rural` | 3 | 1 | 0.180 | 0.0226 |

Excitation is judged at **set level** — a weight is identifiable if *some*
scenario moves its actuator, not if every one does. The sustained one-way ramp is
what does the work; the quiescent case is kept deliberately, so tuning cannot
drift toward a controller that only behaves under stress.

If it fails, read *which* set-level criterion is empty. A per-scenario REJECT
means only that the reference could not complete that scenario, so it cannot
discriminate between candidates and should be withdrawn.

---

## 4. Stage 2 — metric calibration and dead-term audit (~6 h)

```bash
python -m tuning.scripts.calibrate_metrics --n-draws 40 --n-scenarios 3
```

Sobol sample from the *prior*, not from optimiser output — that is what breaks
the circularity behind the six successive cost revisions in the study history.

Three outputs, written to `results/tuning/metric_calibration.json`:

1. **Suggested `MetricScales`** (medians). Paste into
   `tuning/metrics.py::MetricScales` if the ratios to the current values are far
   from 1.
2. **Suggested `ConstraintLimits`** from the reference with 1.5× margin. Apply
   these — the current defaults were partly hand-set and the reference failed
   three of six on the first attempt.
3. **The dead-term audit.** *Act on this.* Any term zero in > 90 % of runs
   contributes nothing; the script exits 1 and names it. Two terms in the legacy
   cost failed this way and nothing caught it (the oscillation term was zero in
   **100 %** of 1555 scenario-runs).

Also check the constraint table: a constraint violated in 0 % of draws carries no
information; one violated in 100 % means the box is empty.

---

## 5. Stage 3 — OLTC switching calibration (~5 h)

```bash
python -m tuning.scripts.calibrate_switching --target-tso 6 --target-dso 6
```

Targets are **tap operations per hour per transformer**, worst transformer in the
class — *not* per day. The design scenarios are event-dense 75-min windows and a
real day is mostly quiet, so extrapolating inflates the figure ~19×: the
reference's perfectly ordinary 4 taps/hour read as a pathological "96/day" under
the first version of this metric and tripped a limit it had no business tripping.

Check `status` in `results/tuning/switching_calibration.json`:

- `bracketed` — usable. Take `g_w` and put it in the baseline.
- `plateau_high` — the budget is **slack**; this weight is not what limits
  switching. **Do not read the returned `g_w` as "the tuned value".**
- `plateau_low` — the budget is unreachable in the bracket; the binding
  constraint is elsewhere (cooldowns, scenario severity, loop gain).

Publish the `ladder` — it is the evidence that the response really is monotone
over the bracket, and a thesis figure on its own.

Then write both bracketed values into `make_config()` and re-run Stage 0.

---

## 6. Stage 4 — the tuning run (~9–10 h)

Smoke it first:

```bash
python -m tuning.scripts.run_tuning_parallel --n-trials 10 --workers 3 \
    --study-name v5_smoke
```

I have only **dry-run** the parallel driver. It splits trials across N processes
against one SQLite study (Optuna's documented multi-process pattern), pins BLAS
to one thread per worker, staggers startup by 20 s, and gives each worker a
distinct sampler seed so they do not all propose the same startup points. Only
worker 0 enqueues the reference, so the others do not waste 15 min duplicating it.

Then the real run:

```bash
python -m tuning.scripts.run_tuning_parallel \
    --n-trials 80 --workers 5 --study-name v5_reparam \
    --output configs/tuned_params_reparam.yaml
```

**Expect ~2×, not 5×.** The bottleneck is memory bandwidth on the per-step sparse
Newton power flow. Your own Monte-Carlo campaign measured K=2 → 1.5×,
K=6 → 2.14× (peak), K=8 → 2.02×, K=10 → regression
(`2026-06-02_006_cigre_montecarlo.md`). There is no point past 6 workers.

Per-worker logs land in `results/tuning/worker_*.log`.

The run prints a **per-constraint violation table** at the end. If no feasible
trial exists, that table says which limit is binding — relax the one actually
responsible, not all of them.

---

## 7. Stage 5 — holdout, once (~4 h)

No driver written yet. Evaluate `holdout_set_v2(seed=42, n=40)` on **four**
points: the hand-tuned reference, the BO optimum, the analytic Tier-1+2 point
(`tau=1`, λ from the preconditioner), and — if you want the comparison — the
legacy 8-dim optimum.

**Evaluate once.** Holdout weeks are the even ISO weeks and the tune set uses
only odd ones (SimBench profiles are strongly autocorrelated within a day, so a
random day split leaks). If tune-to-holdout degradation is large, that is an
overfitting *result to report* — re-tuning on it consumes the holdout and leaves
no independent evidence at all.

Score on `rms_v_ts_pu`, interface-Q RMS, and taps/hour per transformer.

---

## 8. Stage 6 — the closing checks (minutes)

**Identifiability re-test.** Repeat the analysis that condemned the old setup, on
the new study:

- random-forest out-of-fold R² from the 4 coordinates to the constrained scalar
  — **success is materially above 0.09**;
- log-spread of the top-10 trials per coordinate — **success is well under one
  decade** (it was 1.1–3.8 decades before).

If those fail, the parameterization is still wrong and no further budget should
be spent. The queries are in the daily log.

**Where the optimum sits relative to the reference.** The legacy optimum was
10²–10⁵ away in every ratio; anything similar is a red flag, not a discovery.

Then revise `docs/tuning/tuning_strategy.md` — its §4.2 still describes scenario
durations that no longer exist, and §3 needs the corrected reading of the LMI
bound direction.

---

## 9. Open items (do not let these pass silently)

| item | status |
|---|---|
| **g4 settling is inactive.** Its limit sits above the window width, so it contributes nothing. The reference is *censored* at exactly 1200 s — some signal never enters the 2 % band. Diagnose which, then re-enable. | open |
| **No both-layers-stressed case on `rural_700`.** The candidate diverged at the reference weights at two severities and was withdrawn rather than softened further. Needs the divergence understood first. | open |
| **`lambda_scope='preconditioned'`** makes λ identifiable but no longer bounds the *true* worst-case contraction (a probe reached λ_all = 19 while λ_cont was on target). Constraint g3 is what catches that — keep it enabled. | by design, watch it |
| **RMS validation** of the tuned point (§0.1). | required follow-up |
| `tests/tuning/stability_certificate/test_hierarchy.py` fails pre-existing. | unrelated |

---

## 10. What NOT to do

- **Do not re-tune the cost weights against BO output.** That loop produced six
  incomparable revisions. Weights and scales come from the prior sample.
- **Do not relax a constraint because no trial is feasible** without reading the
  violation table first.
- **Do not soften a scenario because it yields an unflattering number.** Softening
  because the *reference cannot complete it* is legitimate; the other is fitting
  the test set.
- **Do not evaluate the holdout more than once.**
