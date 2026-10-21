# Handover: 12-hour confirm stage for the TS fleet-relief screen

Written 21 October 2026 on the workstation session, for a fresh Claude Code
session on the many-core host. Same account, same checkout over the share.

**Your job:** run the 12-hour confirmation runs for the variants listed in
§4, collect the metrics in §5, and write back the table in §7. **Do not change
any source file, do not run git commands, and do not change any default.** If
something looks like it needs a code change, stop and report it instead; the
change belongs on the workstation session so the source hashes stay comparable.

## 1. What this is

The TS (Layer-1) MIQP controller can dispatch tertiary MSC/MSR banks as genuine
integer variables ("hybrid" shunt dispatch). Two findings from 20–21 October set
up this stage:

* The **regular** bank path is **parked** (`g_w_tso_shunt` at its configuration
  default of 3600, above anything the recorded gradient distribution reaches).
  It was parked because it improved the TS voltage term by 0.6% while degrading
  subordinate interface-Q tracking by 54%, STS voltage RMS by 7.6% and DER
  utilisation p95 by 21%.
* That exchange was **unpriced**: the selected parameter set has `g_q = 0`,
  `tso_g_res_sg = 0` and `tso_g_res_der = 0`, so a bank step was priced on its
  voltage effect alone. This stage tests whether giving the TS objective a
  reserve term for its **own** fleet — synchronous generators and TS-connected
  DER — makes bank dispatch worthwhile, with `g_q` non-zero so the subordinate
  tracking cost is no longer free.

No new control code is involved: `Q_gen` is already a TS output, `∇_y f` already
carries `2·g_res_sg·(Q_gen − Q_mid)/Q_half²`, and the hybrid bank column already
includes `∂Q_gen/∂Q_shunt`. This is a configuration experiment.

Background, in reading order:

1. `docs/tuning/2026-10-21_ts_fleet_relief_option.md` — the option, the weight
   audit and the screen design. **Read this one.**
2. `docs/tuning/2026-10-20_hybrid_shunt_design_and_tests.md` — the dispatcher,
   the shared-objective support gate, the campaign consequences.
3. `docs/tuning/2026-10-20_hybrid_shunt_weight_calibration.md` — why
   `g_w_tso_shunt` is an admission threshold and how it is derived.
4. `docs/tuning/2026-10-21_reserve_trigger_margin_derivation.md` — the
   subordinate-reserve path, not used here, and the reference-convention trace.

## 2. Three checks before you start

**(a) Gurobi licence — critical.** The licence is node-locked and has bitten
this project before (`GurobiError: HostID mismatch` on the PowerFactory host,
2026-07-17). `optimisation/miqp_solver.py` keeps SCIP as a fallback but states
that SCIP was measured as inaccurate enough to change *integer* decisions, so
**a SCIP run is not comparable to a Gurobi run.** Check first:

```bash
python -c "import cvxpy as cp; print([s for s in cp.installed_solvers() if s in ('GUROBI','SCIP')])"
```

```bash
python -c "import gurobipy as g; m=g.Model(); print('gurobi ok', g.gurobi.version())"
```

If Gurobi is unavailable, do **not** mix: either get it licensed, or run every
variant *including the baseline* on that host with SCIP and label the whole
table as SCIP. Never compare across solvers.

**(b) Python.** On the workstation the interpreter is
`C:\Users\Manuel Schwenke\.conda\envs\qOFO_clean\python.exe` (Python 3.12.12,
env `qOFO_clean`). Find the equivalent env on your host; do not create a new one.

**(c) Repo.** Work in the shared checkout
`\\130.83.232.108\homefolders$\mschwenke\Python_Projekte\qOFO_2027`
(also reachable as `\\e5server.e5.e-technik.tu-darmstadt.de\homefolders$\...`;
it is mapped to `Z:` on the workstation, so your drive letter may differ — use
the UNC path). Confirm the branch is `weights-reset-2026-09-10` and **leave it
alone**; it carries uncommitted work.

## 3. Why not the Stage-II harness

`experiments/ch_9_parameter_selection/ch_9_3_stage2_screen.py` is the parallel
campaign harness and takes `--workers`, but it requires `--ops-dir`, the
Stage-I operating points, and **Stage-I capture and cache loading reject hybrid
mode outright** because the analytical screen omits the bank columns. Extending
it is open work. So this stage uses `experiments/shunt_hybrid_check.py`, which
is sequential per run — the cores are used by running the variants as
**concurrent processes**, not by threading one run.

## 4. The runs

`--cases regular` is deliberate: it sets both slow-support weights to zero, so
only the regular objective path can commit and the feedforward offset-only
convention applies. That is the convention fleet relief needs — the offset
exists "so the DSO does not counteract a committed switch". Do not enable the
slow support weights here; the paired transfer would make the subordinate DER
fight the bank.

`--seconds 43200` is 12 simulated hours. The 1-hour runs took 220–390 s wall
clock, so budget **45–80 min per run**.

> **TO FILL by the workstation session once screen pass 2 completes:** the
> surviving variants and their per-variant `--movement-weight`, derived with
> `tuning/shunt_weight_calibration.py` from the parked-bank pass. Until this
> table is filled, do not start — an un-derived movement weight reproduces the
> ratchet.
>
> | variant | `--movement-weight` | extra flags |
> |---|---|---|
> | `v1_selected` | 3600 (parked baseline) | *(none)* |
> | `…` | `…` | `…` |

Command pattern, one per variant, each with its **own** output directory:

```bash
python -B -X utf8 experiments/shunt_hybrid_check.py --cases regular --seconds 43200 --movement-weight <DERIVED> --g-q 0.0125 --g-res-sg <X> --g-res-der <Y> --output results/fleet_relief_confirm_20261022/<VARIANT>
```

The baseline keeps the selected weights and the parked bank path:

```bash
python -B -X utf8 experiments/shunt_hybrid_check.py --cases regular --seconds 43200 --output results/fleet_relief_confirm_20261022/v1_selected
```

Concurrency: start **three** processes, watch memory and the Gurobi licence
(academic licences often cap concurrent sessions), then raise if it is healthy.
`--output` uses `mkdir(exist_ok=False)`, so a clash refuses rather than
overwrites — that is a feature, do not work around it.

## 5. Metrics to collect

Each run writes `config.json`, `source_hashes.json`, `run.log`, `summary.json`
and `records.pkl` into its directory. From `summary.json`:

* `n_switches`, `switch_reasons`, and the `commits` list with times and banks
* `interface_error_rmse_mvar`, `interface_error_abs_p95_mvar` — the cost side
* `aggregate_der_utilisation_p95` — subordinate DER, distance from band centre
* `ts_voltage_min_max_pu`, `sts_voltage_min_max_pu`, `dso_solver_status_counts`

The point of the exercise is **TS fleet relief**, and those metrics are not in
`summary.json`. They are in `experiments/helpers/comparison_metrics.py`:
`m_bar_mvar`, `m_bar_pu` and `res_util = mean_g |Q|/(|Q| + headroom)`, built on
`gen_q_headroom_series`. Compute them per run from `records.pkl` and report them
— they are the primary metrics here.

Also report, per run: wall time, any `ERROR`/`recovery`/`infeasible` line in
`run.log`, and whether `source_hashes.json` is identical across all runs (it
must be; a difference means the source changed mid-stage and the comparison is
void).

## 6. Hazards

* **Same filesystem.** The workstation session is also writing under
  `results/`. Use only the new `fleet_relief_confirm_20261022/` tree.
* **No git.** Do not commit, stage, pull, switch branches or stash.
* **No source edits.** `source_hashes.json` is the provenance of the whole
  stage.
* **Solver provenance.** Record which solver actually ran. See §2(a).
* Each process loads its own pandapower networks; expect 1–2 GB per process.

## 7. What to write back

A table with one row per variant: solver, wall time, commits, `m_bar_pu`,
`res_util`, `interface_error_rmse_mvar`, `interface_error_abs_p95_mvar`,
`aggregate_der_utilisation_p95`, TS and STS voltage ranges, and any log
anomaly. Plus one sentence per variant on whether fleet reserve improved and
what it cost in subordinate tracking.

Then stop. The interpretation, the choice of `g_q` within its bracket
(0.003 / 0.0125 / 0.08), and any decision about the campaign proper belong to
the workstation session, which holds the design history.
