# 2026-06-02 — New Monte-Carlo CIGRE driver (`006_CIGRE_MONTECARLO.py`)

**Timestamp:** 2026-06-02 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
Reviewer comment: the CIGRE Table-3 ranking V1→V5 rested on a **single
deterministic 300-min run with one fixed contingency schedule**, so the ordering
could be scenario-specific. To make the robustness claim, repeat the case study
over many randomized scenarios and report the metrics as **distributions**
(mean ± spread), showing the ordering is stable across scenarios.

## What was added
New file **`experiments/006_CIGRE_MONTECARLO.py`** (clone of
`005_CIGRE_MULTI.py`; 005 untouched). No changes to existing modules.

### Key structure
- **Paired Monte-Carlo loop.** Each scenario draws a random `start_time`
  (anywhere in the 2016 SimBench profile year, 15-min grid, leaving the 300-min
  tail) and a random contingency schedule; **all five variants V1–V5 run on the
  identical scenario** (same `start_time` + same schedule, `deepcopy`'d per
  variant because `prepare_load_contingencies` mutates `element_index`).
- **Random schedule generator** (`build_random_schedule`): candidate slots every
  30 min (30…270); each fires with 25 % probability (~2–3 events/run, verified:
  mean 2.55 over 40 seeds). Event type ∈ {line trip, gen trip, load connect},
  chosen uniformly among currently-admissible types.
  - *Lightly constrained* for feasibility: ≤1 line out and ≤1 gen out at a time,
    slack machine excluded, load step fixed at the proven-stable 200 MW/100 Mvar.
    `MAX_LOADS_CONNECTED = 1` (two concurrent 200/100 = 400/200 collapses the V1
    cos φ=1 baseline — known from 005 notes — and would needlessly reject runs).
  - Tripped gen reconnects at `trip+180 min` (omitted if > horizon); line
    outages and load connections persist to the end.
- **Drop-and-replace.** A scenario is *accepted* only if **all five** variants
  converge (full 300-step log; a diverged run re-raises `LoadflowNotConverged`
  inside `run_multi_tso_dso`, caught → empty log). On any divergence the whole
  scenario is discarded (short-circuit: stop after the first failing variant)
  and the next seed is drawn, until `--runs` paired-valid scenarios are
  collected. Safety cap `MAX_ATTEMPT_FACTOR*N+25` attempts.
- **Element enumeration** (`enumerate_elements`): rebuilds the combined net like
  the runner (`build_ieee39_net` → `fixed_zone_partition_ieee39` →
  `add_hv_networks`) → gens `[1,2,5,7]` (machine gens minus slack), 35 TN lines
  (HV feeders excluded), 20 TN load-host buses.
- **Compact per-run time series** persisted to `timeseries/run_XXXX.npz`
  (voltage-RMS error per variant via `voltage_rms_err_all`; selected-gen P/Q via
  `zone_p_gen`/`zone_q_gen`) — heavy logs are **not** pickled. Enables `--replot`
  and resume across sessions.
- **Reuse:** `cigre_summary_table` (Table-3 metrics), `voltage_rms_err_all`,
  `ContingencyEvent`, `run_multi_tso_dso`, and plot_cigre helpers
  (`_gen_info_with_k`, `_select_gens`, `CIGRE_PALETTE`, `apply_cigre_style`,
  `_save_pdf`, `compute_generator_q_limits`). `make_cigre_config`/`VARIANTS`
  copied verbatim from 005 (keep in sync if 005 retunes).

### Outputs (`results/006_cigre_mc/`)
`contingency_schedules.csv`, `metrics_per_run.csv`, `failed_attempts.csv`,
`timeseries/*.npz`, `table3_distribution.csv` (per-variant mean/std/median/IQR —
the populated Table 3), `ranking_stability.csv` (fraction of runs V4 is best
among V1–V4 + mean ranks), and three figures: `Fig_mc_voltage_band.pdf`
(voltage tracking error, mean + min-max band, one panel/variant),
`Fig_mc_capability.pdf` (gen P-Q clouds pooled over all runs vs Milano
envelope; excludes V1/V5 like 005), `Fig_mc_table3_box.pdf` (box plots).

### Parallelism (added same day)
**Scenario-level process parallelism** (`--jobs P`), since each scenario is
independent. Restructured around **per-seed staging**: `run_one_scenario(seed)`
(module-level, picklable) runs all 5 variants and writes
`_runs/scenario_<seed>.json` (+ `ts_<seed>.npz` when accepted, `log_<seed>.txt`
when run in a worker). A **collector** (`collect_and_finalize`) reads the staging,
keeps the first N accepted scenarios *by ascending seed* (deterministic
regardless of worker completion order → `run_id = 0..N-1`), and rebuilds the
master CSVs + canonical `timeseries/run_XXXX.npz` + tables + figures.
- Serial path (`--jobs 1`) unchanged in behaviour; parallel path uses
  `ProcessPoolExecutor` in waves of P seeds. **Process-based, not threads**
  (runner monkey-patches module globals; Gurobi/matplotlib not thread-safe;
  spawn re-imports cleanly under the `__main__` guard).
- BLAS/OpenMP pinned to 1 thread/process at module top (before numpy import) so
  P workers don't oversubscribe cores. Per-worker `result_dir = _scratch/pid<pid>`
  avoids file collisions. **Named-user `gurobi.lic`** (confirmed) → no local
  concurrent-process cap, so P is bounded only by cores/RAM.
- `--resume` now keys off the `_runs/` staging (skips already-tried seeds);
  `--replot` = run the collector only.

### Base-case pre-screen (added after first 50-run attempt showed heavy rejects)
Diagnosis from `_runs/log_*.txt`: most rejections were **V1 (cos φ=1) failing the
base power flow at winter-peak `start_time`s, *before* any contingency** (e.g.
2016-02-20 17:45, ~8.8 GW of 11.3 GW; no "Post-contingency PF" line). A second,
rarer cause is the inner Q(V)+OLTC `run_control` loop hitting the 301-iter cap
even at low load (numerical, e.g. V2 at a summer night). Uniform full-year
`start_time` kept sampling extreme peaks where cos φ=1 simply can't operate, so
the "all-5-converge" gate biased the accepted set toward mild conditions.

**Fix (user chose "pre-screen base case, else resample"):** `run_one_scenario`
first runs `_prescreen_base_case(start_time)` — V1 (cos φ=1) with **no
contingencies** at a coarse `PRESCREEN_DT_S=900 s` step (20 probes over the
window, spans the load peak; V1 has no MIQP so it's cheap). If it diverges the
scenario is recorded `failing_variant="base_infeasible"` and the five full runs
are skipped (resample). Verified: winter-peak → INFEASIBLE in ~6 s; April
low-load → feasible in ~50 s. The OFO variants add reactive support, so a
V1-feasible base implies the others are feasible at base; **contingency**-induced
divergence is still caught by the full run → drop-and-replace. Toggle with
`--no-prescreen`; threaded through `run_monte_carlo`/workers as an arg (spawned
children don't see parent globals).

**Methodological scope/caveat:** pre-screen + resample restricts the study to
operating conditions where the cos φ=1 baseline can hold the *base* case. This
deliberately excludes peak-load conditions where V1 collapses outright — i.e. the
reported ranking is **conservative** for V4 (it omits exactly the conditions
where V4's advantage is largest). If we later want to showcase that fragility,
switch to the "report survival rate" design (run all 5, no gate).

### CLI
`--runs N` (default 50), `--seed S` (default 20260602; scenario seed = S+attempt),
`--jobs P` (parallel over scenarios), `--no-prescreen` (disable base-case
pre-screen), `--resume` (skip already-tried seeds via `_runs/`),
`--replot` (collector only). Headless `Agg` backend.

## Validation done (2026-06-02)
- Pure-logic (qOFO_clean env): element enumeration, schedule generation
  (mean 2.55 events/run), schedule CSV round-trip, gen selection — OK.
- Aggregation + all three figures on synthetic 5-run data — OK (Table 3 +
  ranking CSV + PDFs/PNGs render; NaN-only `rms_e_sts` handled).
- Live run path: drop-and-replace confirmed (seed 20260602 → V1 diverged →
  rejected → advanced to next seed). **Real serial `--runs 1` completed (exit 0):**
  seed 20260604 accepted (3 attempts, 2 rejected), Table 3 + ranking + 3 figures
  written.
- Parallel refactor: orchestrator + collector + `--resume` validated on a fake
  worker (drop-and-replace, ascending-seed run_id assignment, resume skips tried
  seeds, collector rebuilds CSVs/timeseries/figures) — OK.
- **Real `--jobs 3 --runs 1` completed (exit 0):** 3 worker processes spawned and
  imported cleanly; wave ran seeds 602/603/604 concurrently (rejected V1, V2;
  accepted 604 = run_id 0). Metrics **identical** to the serial run → parallel is
  deterministic and equivalent. `ProcessPoolExecutor` confirmed working with the
  digit-named main module under Windows spawn.

## Revisions (2026-06-02, later) — severity, figures, performance
- **Severe contingencies:** connected-load reactive power is now random
  `U[100, 400] Mvar` per event (`LOAD_Q_MIN_MVAR`/`LOAD_Q_MAX_MVAR`), drawn from
  the seeded rng in `build_random_schedule`; active part fixed 200 MW. Verified
  min 100 / max 399 / mean 262 over 200 seeds.
- **Voltage-band figure (`Fig_mc_voltage_band`):** shaded band is now
  **mean ± 1σ** (std across runs per timestep), lower bound clipped at 0 (RMS
  error ≥ 0) — was min–max.
- **Box plots (`Fig_mc_table3_box`):** now a **single row** of a selectable
  subset; `--box-metrics a,b,c` (validated vs the 6 names, honoured by
  `--replot`); default `rms_v_ts_pu,res_util,rms_q_tie_mvar,n_sw`; `METRIC_LABELS`
  for readable axes.

### Performance investigation (why the overnight `--jobs 10` was ~serial)
Measured with throwaway probes (deleted):
- **Not Gurobi:** a full V4 run = 222 s but total MIQP solve time = **0.6 s**
  (300 tiny per-zone CVXPY→Gurobi solves, ~2 ms each). Backend confirmed
  `MIQP_SOLVERS=['GUROBI']` but it's irrelevant to runtime.
- **Not BLAS:** threads pinned to 1 (`openblas=1`, confirmed via threadpoolctl).
- **Not network I/O:** parallel with profiles copied to a local disk was the
  same speed as reading the CSV from the network share (56.0 vs 56.7 s/run).
- **It's memory bandwidth.** The cost is the per-step sparse Newton power flow +
  `run_control` Q(V)/OLTC loop. Machine = 8 physical / 16 logical cores, 32 GB.
  Throughput sweep (30-min runs): K=2 → 1.5×, K=4 → ~1.6–2.3× (noisy),
  **K=6 → 2.14× (peak)**, K=8 → 2.02×. Per-run slowdown grows 1.06×→3.34× as K
  rises. So throughput plateaus ~2–2.3× and **regresses past ~6–8**; `--jobs 10`
  oversubscribed the 8 physical cores → the ~2× overnight.
- **joblib/loky would NOT help:** same N OS processes → identical bandwidth
  contention; both spawn on Windows (no fork COW); tasks are coarse (~200 s) with
  tiny args/returns so dispatch overhead is negligible; worker reuse + BLAS
  capping already in place. The bottleneck is hardware, not the pool library.

### Fix applied: continuous-feed scheduler (replaces the wave barrier)
The old parallel loop submitted a wave of `jobs` seeds then **waited for the
whole wave** (`as_completed` per batch). Because task durations span ~100×
(base_infeasible pre-screen reject ~6 s · single-variant reject ~200 s · full
accept ~1000 s), a barrier left workers idle waiting for the slowest in each
batch. Replaced with a **continuously-fed pool** (`wait(FIRST_COMPLETED)` +
submit-on-completion): keeps exactly `jobs` scenarios in flight, refills the
instant any finishes, stops refilling once the target is met and lets the
remainder drain (collector keeps the first `n_runs` accepts by ascending seed).
Validated with a real `ProcessPoolExecutor` + fast dummy worker: max in-flight
never exceeded `jobs`, stops at target, drains, no deadlock. This is
framework-independent (the win is the scheduling pattern, not loky).
`--jobs` help text updated to document the memory-bandwidth ceiling.

### Beyond one machine
Per-seed staging (`_runs/scenario_<seed>.json`) already enables **multi-node**
scaling: run the driver on several machines against the same `_runs/` dir (or
disjoint seed ranges) and let the collector merge — aggregate memory bandwidth
scales with nodes, which no single-node pool (incl. loky) can provide.

### Root cause of the `base_infeasible` rejections (2026-06-02, later)
Investigated why so many random start_times were `base_infeasible`. **It is NOT
cos φ=1.** At a winter-peak start (2016-02-20 17:45) **all five variants** fail
the base power flow identically — including V2/V4/V5 which give the DSO DER full
Q(V)/OFO reactive support. So reactive support is not the issue.

Drilling in (raw `pp.runpp` at the operating point):
- **Single-slack PF converges fine** at every window step (Vmin ≈ 1.01–1.03,
  max line ≈ 97–106 %, balanced gen ≈ load).
- With **`distributed_slack=True`** (the inherited config value) the SAME points
  give Vmin 0.89–0.91, **line loading 217–224 %**, and outright divergence at the
  higher-load steps.
- The zonal dispatch assigns the **slack machine `G1_bus38` a ~6.4 GW P
  setpoint** while load is ~4.7 GW (total dispatched ≈ 9 GW → **+4.3 GW
  surplus**). A single slack ignores that setpoint and just balances; but
  `distributed_slack=True` *honours* it, pushing ~4 GW of phantom surplus through
  the lines → overloads/voltage collapse/non-convergence. `run_control` is
  irrelevant (identical result with it off).

**Fix applied (006-local):** `make_cigre_config()` now sets
`cfg.distributed_slack = False`. Verified: the winter-peak base case that
diverged now **converges for V1 and V4** over the full coarse window. This is a
006-only copy of the config, so 002/003/005 are untouched. Single slack is also
consistent with the dispatch's own design ("slack absorbs … imbalances").

**Implications:**
- Far fewer (likely near-zero) `base_infeasible` rejections now → the
  Monte-Carlo can sample high-load winter conditions, and the earlier
  "conservative scope" caveat is largely lifted.
- The base-case **pre-screen is now mostly redundant** (it added ~47 s/feasible
  scenario). Recommend running with `--no-prescreen` post-fix; kept available for
  safety. (Did not flip the default.)
- **Latent shared-code bug:** `compute_zonal_gen_dispatch`
  (`network/ieee39/zonal_balancing.py`) over-assigns the slack gen's P at high
  load. Harmless under single slack, but wrong for any experiment using
  `distributed_slack=True`. Flagged as a separate task.

### Proper fix (2026-06-02, later still): zonal-dispatch spill bug + distributed slack restored
Per the user's intent (all machines should share the active-power imbalance
after a load step / gen trip), `distributed_slack=False` was the wrong fix —
verified that with single slack a gen trip is absorbed entirely by `G1_bus38`
(only its P moves), whereas `distributed_slack=True` shares it by `slack_weight`
(all gens move). So the real fix is to make `distributed_slack=True` stable.

**Root cause (shared code):** `compute_zonal_gen_dispatch` in
`network/ieee39/zonal_balancing.py`. Measured: the dispatch's *load view*
(4689 MW) matched the applied load (4696 MW) — it does NOT over-count load. The
over-dispatch came from the **cross-zone spill loop**: it places a deficit
zone's shortfall on OTHER zones' gens (mainly the high-capacity slack gen9), but
then recomputed the remaining deficit **per-zone-locally**
(`residual_z − dispatch_in_zone_z`). Covering zone 2/3's deficit via zone-1 gen9
never reduced that per-zone measure, so the same ~1 GW deficit was re-added every
iteration (×5) → gen9 dispatched to 6.4 GW (≈2× the real residual). Single slack
hid it (slack setpoint ignored); `distributed_slack=True` honoured it →
200 %+ line overloads / collapse at high load.

**Fix:** recompute the spill's remaining deficit **system-wide**
(`max(0, Σ residual − Σ dispatched)`) instead of summing per-zone terms. ~6-line
change in the spill loop. Now total dispatched ≈ system residual, so under
distributed slack only the (small) loss mismatch is shared out.

**Verified:** winter 17:45 dispatch 4667 MW vs load 4696 MW (gen9 slack 2465 MW,
was 6371); winter-peak base now **converges** for V1 and V4 with
`distributed_slack=True`; a gen5 trip (−133 MW) is shared gen1/2/7/9 =
+8.5/+8.5/+10.7/+106.5 MW (slack-weighted). `006` reverted to
`cfg.distributed_slack = True`.

**Caveats:** the dispatch fix is **shared code** (002/003/005 also use it). Their
gen-P dispatch (hence results/figures) will change slightly and they should be
re-run/re-validated. With this fix the winter base-case collapse is gone, so the
**pre-screen is now largely unnecessary** — recommend `--no-prescreen` (default
left ON for safety). The separately-flagged dispatch task is now resolved here
(the chip can be dismissed).

## Open / next
- **Re-validate 002/003/005** after the shared dispatch fix (gen-P dispatch changed).
- Re-run 006 with `distributed_slack=True` (now default in 006) and likely
  `--no-prescreen`: expect high acceptance even at winter peaks. `--jobs 6`.
- Fix `compute_zonal_gen_dispatch` slack over-assignment (separate task) if
  `distributed_slack=True` is wanted elsewhere.
- Runtime: 50 runs × 5 variants × 300 steps is long (hours) — run in background
  with `--resume`.
- If acceptance rate is low (winter high-load + gen/line combos diverge),
  consider disallowing concurrent line+gen outages (currently allowed per the
  user's "≤1 line + ≤1 gen at a time" choice) or shrinking the load step.
- `rms_e_sts_mvar` is NaN for V1/V2/V5 (no dispatched interface setpoint) — its
  distribution/ranking is meaningful only for V3/V4 (matches paper footnote).
- Obsidian MCP offline → this log lives in the repo `docs/daily_log/`.
