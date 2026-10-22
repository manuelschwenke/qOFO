# 2026-06-08 — Voltage-stability / nose-curve reachability guard

**Timestamp:** 2026-06-08 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
The multi-TSO/DSO simulation is **quasi-static**: at every step it resolves only
the algebraic load flow (`pp.runpp`). Newton-Raphson can converge to operating
points that lie **on or beyond the saddle-node (nose)** of the P-V / Q-V curve —
i.e. on the lower (unstable) voltage branch. These are valid algebraic
solutions, but the physical dynamic system could never reach or hold them, so
any controller result recorded there is unphysical. We add a guard that, at
every reported equilibrium, verifies the point is on the **stable upper branch**
and aborts (Fail-Fast) at the first violation, whilst recording the stability
margin at every step.

## Criterion (modal voltage-stability analysis)
For the converged NR solution the per-unit power-flow Jacobian (polar
coordinates, slack excluded, PV buses contribute angle states only) is

```
J = [[ dP/dtheta, dP/dV ],
     [ dQ/dtheta, dQ/dV ]]
```

1. **Singularity guard.** `sigma_min(J) -> 0` at the nose (J becomes singular).
   Abort when `sigma_min(J) < tau_sigma`. Detects only the *immediate vicinity*
   of the tip; it does **not** by itself separate the upper from the lower
   branch (verified empirically — see below).
2. **Reduced Q-V Jacobian (Gao/Morison/Kundur, IEEE Trans. PWRS 1992).** Schur
   complement
   ```
   J_R = dQ/dV - (dQ/dtheta)(dP/dtheta)^{-1}(dP/dV)
   ```
   **Sign convention: all eigenvalues of J_R having positive real part <=> stable
   upper branch.** Abort when `min(real(eig(J_R))) <= tau_eig`. The critical bus
   = largest participation (squared right-eigenvector entries) in the
   minimum-eigenvalue mode.

Defaults `tau_sigma = tau_eig = 1e-6` (proximity thresholds, exposed as config).

## Method / key implementation points
- **Backend.** pandapower. The internal NR Jacobian is read directly from
  `net._ppc["internal"]["J"]` with the `pq`/`pv` partition from
  `net._ppc["internal"]`; the Jacobian is **not** rebuilt from Ybus.
- **Distributed-slack subtlety (important).** The main loop solves with
  `distributed_slack=True`, which **augments** the internal Jacobian by one
  row/column for the slack-distribution variable (shape `n_pv+2 n_pq+1` instead
  of the canonical `n_pv+2 n_pq`). Empirically the augmented matrix is **not** a
  plain principal sub-block of the canonical one (leading-block diff ~24 on
  case9), so it cannot be sliced. When `check_reachability` sees this
  non-canonical shape it **re-converges a deep copy with
  `distributed_slack=False, run_control=False`** — the same device
  `JacobianSensitivities.__init__` already uses — to obtain the canonical
  single-slack Jacobian. This lets pandapower's own NR produce the Jacobian (not
  a hand rebuild) and leaves the caller's `net` and its recorded `res_*` tables
  untouched (deep copy). On case9 the canonicalised result is **identical** to a
  native single-slack solve (lambda_min, sigma_min, critical bus all match).
- **Fail-Fast.** Missing/None net, unconverged PF, missing/NaN Jacobian, empty
  PQ set, singular `dP/dtheta`, or an unexpected Jacobian shape each raise a
  descriptive exception. No silent defaults, no swallowed errors.
- **Integration point.** Inserted in `run_multi_tso_dso` immediately after the
  end-of-step power flow converges and **before** any metric logging, **outside**
  the PF `try/except` so the `ReachabilityViolation` propagates out of the runner
  rather than being swallowed by the "Power flow failed" handler. Steps whose PF
  fails `continue` before the check (no equilibrium to test).

## What was changed

### `analysis/reachability.py` (new)
- `ReachabilityResult` dataclass: `on_stable_branch, sigma_min_J, cond_J,
  lambda_min_JR, critical_bus, step_index`.
- `ReachabilityViolation(RuntimeError)`.
- `check_reachability(net, step_index=None, tau_sigma=1e-6, tau_eig=1e-6,
  *, ensure_standard_structure=True) -> ReachabilityResult` — pure-numpy/scipy
  (`scipy.linalg.svd`, `scipy.linalg.solve`, `scipy.linalg.eig`); sparse-aware.
- `ReachabilityMonitor` — time-series wrapper: records the margin every step
  into `self.records` (exposed via `to_dataframe()`) and raises
  `ReachabilityViolation` at the first lower-branch step with step, time,
  critical bus and the offending margins in the message.

### `configs/multi_tso_config.py`
- `MultiTSOConfig`: `enable_reachability_guard: bool = True`,
  `reach_tau_sigma: float = 1e-6`, `reach_tau_eig: float = 1e-6`.

### `experiments/helpers/records.py`
- `MultiTSOIterationRecord`: three defaulted fields `reach_sigma_min_J`,
  `reach_lambda_min_JR`, `reach_critical_bus` (older pickles still load).

### `experiments/runners/multi_tso_dso.py`
- Import `ReachabilityMonitor, ReachabilityViolation`.
- Instantiate one `ReachabilityMonitor` before the main loop (when
  `enable_reachability_guard`).
- Per-step `monitor.check_step(net, step_index=step, time_s=time_s)` at the
  post-PF recording point; margins copied into the step record.

### `tests/test_reachability.py` (new) and `tests/smoke_reachability.py` (new)

## Validation
- **Numerical (2-bus nose ramp, predominantly reactive line).** Upper branch
  (flat start): `lambda_min(J_R)` decreases 5.40 -> 1.40 and `sigma_min(J)`
  5.01 -> 0.93 as the load ramps toward the nose, both staying positive; beyond
  the nose NR diverges. Lower branch (low-voltage `init_vm_pu`): converges to
  v≈0.48 with `lambda_min(J_R) = -1.6` while `sigma_min(J)=0.79` is **not**
  near zero — confirming the modal eigenvalue, not `sigma_min`, discriminates
  the branch.
- **Unit tests** `pytest tests/test_reachability.py` → **7 passed** (145 s on
  the network share). Covers: upper-branch stable; margin monotonically shrinks
  toward the nose; lower-branch flagged; monitor passes the upper ramp then
  raises at the post-nose point and records the full trajectory; Fail-Fast on
  `None` and on an unconverged net; distributed-slack canonicalisation matches
  the single-slack result.
- **Integration smoke** `tests/smoke_reachability.py` (3-step `wind_replace`
  run, guard ON): runner returns **without** a `ReachabilityViolation`; every
  step on the stable upper branch with `sigma_min(J) ≈ 0.268`,
  `lambda_min(J_R) ≈ 0.92–0.94`, critical bus 59 (recorded into each record).
  Per-step cost is dominated by the MIQP solves (steady-state ≈ 7–10 s/step;
  the one-off 74 s step-1 is Gurobi-license + DER-seed overhead, not the guard);
  the guard's deep-copy + single-slack `pp.runpp` is marginal next to it.

## Thresholds used
`tau_sigma = 1e-6`, `tau_eig = 1e-6` (defaults; `MultiTSOConfig.reach_tau_sigma`
/ `reach_tau_eig`).

## Implementation status

| Component | Status | Notes |
|---|---|---|
| `analysis/reachability.py` | ✅ Implemented | `check_reachability` + `ReachabilityMonitor`; sigma_min(J) guard + modal J_R eigenvalue criterion; participation-factor critical bus |
| Fail-Fast input validation | ✅ Implemented | None/unconverged/missing-J/NaN/empty-pq/singular-dP-dtheta/bad-shape all raise |
| Distributed-slack canonicalisation | ✅ Implemented | deep-copy + single-slack re-converge; verified identical to native single-slack on case9 |
| Runner integration | ✅ Implemented | per-step check before metric logging; violation propagates (Fail-Fast) |
| Margin recording | ✅ Implemented | `MultiTSOIterationRecord.reach_*` + `ReachabilityMonitor.to_dataframe()` |
| Config flags | ✅ Implemented | `enable_reachability_guard` (default True), `reach_tau_sigma`, `reach_tau_eig` |
| Unit test | ✅ 7 passed | `tests/test_reachability.py` |
| Integration smoke | ✅ Passed | `tests/smoke_reachability.py`; 3-step wind_replace, guard ON, no abort, all upper branch (λ_min(J_R)≈0.93, σ_min≈0.27, crit bus 59) |

## Open / next
- **Default is ON.** `enable_reachability_guard=True` means every existing
  experiment now runs the guard and will **abort** if it ever passes a
  lower-branch point. That is the intended behaviour, but it is a behaviour
  change for in-flight runs (005/006). Set the flag `False` to disable per run.
- **Per-step cost.** The distributed-slack canonicalisation does one extra
  deep-copy + single-slack `pp.runpp` per step. If this proves too costly on
  long horizons, options: (a) cache/avoid the deep copy by snapshotting and
  restoring `net._ppc`/`res_*` around an in-place single-slack solve; (b) gate
  the guard to fire only on TSO/DSO/contingency steps. Cost measured in the
  smoke run — see above.
- **Recording on the violating step.** `check_step` records the margin into the
  monitor's own trajectory *before* raising, so `monitor.to_dataframe()` always
  contains the violating step; the per-step `rec.reach_*` fields are populated
  only on the non-violating path (the run aborts before that step's record is
  appended to `log`).

## Addendum (2026-06-08, later) — 005/006 honour the guard but tag it distinctly

Both `005_CIGRE_MULTI` and `006_CIGRE_MONTECARLO` build `MultiTSOConfig` and call
`run_multi_tso_dso`, so the guard is **active on every fresh re-simulation**
(default ON; not exercised under `--replot`). Both previously wrapped the runner
in a broad `except Exception`, which would have funnelled a `ReachabilityViolation`
into the same bucket as a power-flow divergence. Per request, a nose-curve
rejection is now **tagged distinctly and its margin trajectory persisted**:

- **`ReachabilityViolation` enriched** (`analysis/reachability.py`): carries
  `.result` (violating `ReachabilityResult`) and `.margins` (full per-step
  trajectory incl. the violating step). The runner attaches `.partial_log` (the
  records computed before the violation) on its way out.
- **005** (`run_one`): dedicated `except ReachabilityViolation` → prints
  `VOLTAGE-UNSTABLE (nose)`, keeps `partial_log`, dumps the margin trajectory to
  `results/005_cigre/<V>/reach_margins.csv`.
- **006**: `run_variant` now returns `(log, conv, reason, margins)` with
  `reason ∈ {"voltage_unstable","diverged",None}`; `_prescreen_base_case` returns
  `(ok, reason)`. `run_one_scenario` records a new `failing_reason` field in
  `scenario_<seed>.json` and the report/`FAILED_CSV`, and writes
  `RUNS_DIR/reach_margins_<seed>_<V>.csv` for a voltage-unstable rejection. The
  console reject tag now reads e.g. `reject (V3/voltage_unstable)`. This keeps
  the MC feasibility statistics separating physically-distinct voltage-instability
  rejections from genuine divergences. `failing_reason` is additive/optional, so
  resumed runs with old JSONs still load (`.get` → None).

Validation: `py_compile` clean on all four files; the enriched exception payload
verified on the 2-bus lower-branch case (`.result`, `.margins`, catch-as-Exception
fallback ordering, `.partial_log` default).

## Suggested Obsidian note to update
`[[voltage-stability-guard]]` — record the modal Q-V criterion, the
distributed-slack Jacobian augmentation gotcha (use single-slack re-converge,
do **not** slice the augmented J), and the empirical finding that `sigma_min(J)`
alone cannot separate the branches; link from `[[todo]]`.
