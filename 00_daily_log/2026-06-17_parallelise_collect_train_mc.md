# 2026-06-17 — Parallelise MC training-data collection (joblib/loky)

## Reason
`experiments/_collect_train_mc.py` ran the full Monte-Carlo collection
(`N_OP` operating points × `K_PERTURB` walk steps, each a `pp.runpp` +
`compute_numerical_h_dso`) sequentially. Each op-point walk is fully
independent — within-walk ΔH / residual pairs never cross op-point
boundaries — so the loop is embarrassingly parallel. Requested: parallelise
with joblib loky.

Also found a pre-existing blocker: the script called
`exp003.make_base_config()`, which does **not** exist in 003 (only
`make_config`; `make_base_config` lives in 002). The script could not have
run as-is.

## What changed (full restructure of `experiments/_collect_train_mc.py`)
- **`make_base_config()` → `make_config()`** (the real 003 entry point; it
  also now yields the `[Q|V]`-only H from the `dso_monitor_currents=False`
  change, so `n_y = n_q_tr + n_v`, `n_i = 0`).
- **All top-level execution moved into `main()` under
  `if __name__ == "__main__":`** — required for loky on Windows (spawn
  re-imports the module in every worker; heavy top-level code would re-run).
- **Extracted functions** (all module-level, so workers can import them):
  - `_extract_y(net, q_idx, v_idx, i_idx)`, `_build_u(q_cor, taps, tap_neutral)`,
    `_seed_qv(net)` — now take explicit args instead of module globals.
  - `_build_worker_state(quiet)` — runs the 1-step init, decouples DSO_2 behind
    pinned slacks, derives all index/bound parameters + `Q_MAX_FRAC`, loads
    profiles; returns a state dict `S`. Sets `verbose=0`, live plots off.
  - `_walk_one_op(S, t_mc, seed)` — one operating-point random walk; returns
    `(records, dh_within, residuals, skipped)`. Per-op `rng = default_rng(seed)`.
    **OLTC change (see below):** only the continuous DER `q_set` is perturbed;
    taps move via DiscreteTapControl.
  - `_process_chunk(ts_chunk, seed_chunk)` — **one init per chunk**, then walks
    its timestamps. Chunking (not per-op tasks) bounds inits to
    `min(n_jobs, N_OP)` and keeps each task self-contained (robust under
    cloudpickle, no cross-task worker caching).
  - `_sample_timestamps(rng)`.
- **Parallel map:** `np.array_split` the `N_OP` indices into `n_eff =
  min(effective_n_jobs(N_JOBS), N_OP)` chunks; `Parallel(n_jobs=N_JOBS,
  backend="loky", verbose=10)`. Results merge in input order (op-point order
  preserved). `N_JOBS == 1` takes a sequential path (no joblib).
- **New env knob `MC_N_JOBS`** (default `-1` = all cores; `1` = sequential).
- **RNG:** per-op independent streams via `SeedSequence(SEED).spawn(N_OP+1)`
  (child 0 → timestamps, children 1.. → walks). Reproducible given SEED but
  **not bit-identical** to the old sequential RNG order (inherent to
  parallelising). Statistics are equivalent (MC samples are exchangeable).
- Q/R estimation, shard mode (`MC_PARTIAL_OUT`), training-data save, and ANN
  training are unchanged (run in the parent after the merge).

## OLTC: random ±1 → DiscreteTapControl (per user request)
The old walk perturbed the coupler 3W taps with a random `±1` draw
(`rng.choice([-1,0,0,0,1])`). Replaced with the AVR-style
`pandapower.control.DiscreteTapControl` — the same controller the runner
installs in its Phase-2 init (`multi_tso_dso.py`, `side="mv"`,
`element="trafo3w"`, band = `oltc_init_v_target_pu ± dso_oltc_init_tol_pu`).
- In `ofo` mode the runner DROPS those coupler controllers after init (the OFO
  owns the taps), so `_build_worker_state` re-installs them on
  `cfg.oltc_trafo_indices` and records their `net.controller` indices
  (`tap_ctrl_idx`).
- During the walk only `q_set` is perturbed; `pp.runpp(run_control=True)` lets
  the DiscreteTapControl move the taps to hold the MV-side voltage band. The
  settled `tap_pos` is read back into the actuator vector `u`.
- **Critical:** `compute_numerical_h_dso(closed_loop=True)` perturbs `tap_pos`
  by ±1 and re-solves with `run_control=True` (numerical_h.py:533-538). An
  active DiscreteTapControl would move the tap straight back and zero the
  `∂y/∂tap` column. So the tap controllers are toggled `in_service=False`
  around the H call (in a `try/finally`); the deepcopy inside the function
  inherits the disabled state, giving a clean tap finite difference. The DER
  QVLocalLoops stay active (closed-loop secant).

Consequence: tap motion is now the realistic voltage-driven response correlated
with `q_set`, not an independent random probe. The `∂y/∂tap` column of H is
still identified (clean FD), but the tap values seen in `u` span a narrower,
operating-point-driven range than the old random walk. Worth checking that the
ANN/KF still see enough tap variation to learn the OLTC column; if not, widen
the band or add occasional explicit tap probes.

## u_scale now saved in kalman_matrices.npz (consistency with ‖Δu‖² Σ_q)
The MC path previously saved only `Q`/`R`, so `_KalmanHPredictor._load_matrices`
fell back to `u_scale = ones` — inconsistent with the `s² = ‖Δu/u_scale‖²`
process-noise scaling added earlier today. Now:
- `_walk_one_op` accumulates every within-walk `Δu` (`du_within`, all pairs incl.
  small ones), threaded through `_process_chunk` → `main` (functions now return a
  5-tuple).
- `main` computes `u_scale = sqrt(mean(Δu², axis=0))` with the same zero-channel
  guard (`→ 1.0`) as `generate_kalman_matrices`, and saves it:
  `np.savez(kalman_path, Q=Q, R=R, u_scale=u_scale)`.
- Shard mode also dumps the raw `_du_within` so a (future) merger can compute a
  global `u_scale`.
Note: OLTC channels under DiscreteTapControl move rarely, so their `u_scale` may
hit the 1.0 guard if a tap never moves in the dataset — expected, not a bug.

## Smoke test (MC_N_OP=4, K_PERTURB=3, N_JOBS=2) — verified 2026-06-17
- py_compile OK on all edited files (multi_tso_config.py, multi_tso_dso.py,
  003_S_DSO_CIGRE_2026.py, _collect_train_mc.py).
- Ran end-to-end (exit 0): parallel collection across 2 loky workers, 0
  convergence failures, 12 samples; DiscreteTapControl installed on couplers
  [3,4,5]; H shape (13,13) (no current rows — dso_monitor_currents=False);
  kalman_matrices.npz saved with Q (169×169), R (13×13), u_scale (13); ANN
  trained and saved.
- Fixed a Windows console crash: a `Δ` (U+0394) in a print is outside cp1252
  (colorama path raised UnicodeEncodeError). Replaced with ASCII `du`; also
  changed an added `±` to `+-` to match the runner's convention. NB the save
  preceded the bad print, so kalman_matrices.npz was written even on the failed
  first run — but the ANN step was skipped.

### DECISION (with M. Schwenke): accept realistic no-tap-motion
The smoke test showed the coupler taps **never moved** — constant [-3,-3,-4]
across all 12 samples, so OLTC `u_scale = [1,1,1]` (zero-motion guard). On the
slack-pinned stiff island the MV voltage stays near 1.03 and the ±0.01 band is
never tripped by the q_set perturbations. Decision: **keep as-is** — this is the
honest DiscreteTapControl behaviour and is consistent with the paper's
near-constant / over-actuated sensitivity narrative. Consequence (intended, not
a bug): the OLTC actuator dimension is unexcited in the training data, so the
ANN/KF cannot learn ∂y/∂tap from data; the ∂y/∂tap column is still present in
the target H (computed analytically by compute_numerical_h_dso). Do NOT
"fix" the constant tap / u_scale_OLTC=1.0.

### TODO before using the data
The smoke run **overwrote** training_data.npz and kalman_matrices.npz with the
degenerate 12-sample (4-op) output. Regenerate with the full run before any real
KF/ANN evaluation:
`python experiments/_collect_train_mc.py`  (defaults: N_OP=60, K_PERTURB=15,
N_JOBS=all cores).

## Verification
- `py_compile`: **NOT YET RUN** — command-execution safety classifier was
  temporarily unavailable at edit time. Run before use:
  `python -m py_compile experiments/_collect_train_mc.py`
- joblib confirmed already a repo dependency (used in `tuning/objective.py`,
  `tuning/validate.py`), so no new install expected. Confirm version:
  `python -c "import joblib; print(joblib.__version__)"`

## Impact / TODO
- Invoke as `python experiments/_collect_train_mc.py` (or `-m
  experiments._collect_train_mc`). Smoke-test first with a small fast run, e.g.
  `MC_N_OP=4 MC_K_PERTURB=3 MC_N_JOBS=2 python experiments/_collect_train_mc.py`.
- Existing repo note (`tuning/objective.py`) warns loky can occasionally
  interfere with pandapower's solver setup on some machines — if so, set
  `MC_N_JOBS=1`.
- Each worker runs its own 1-step init (the controller is not pickled — too
  risky given the embedded MIQP solver), so wall-time ≈ one init + (serial
  walk time)/n_jobs.
