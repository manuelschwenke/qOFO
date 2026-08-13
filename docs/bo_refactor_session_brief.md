# Session brief: continue the BO refactoring plan

**Written:** 2026-07-31, from the session running the dead-band sweep.
**Purpose:** start a fresh session on the BO tuning rework without re-deriving context.

---

## The plan

`docs/daily_log/07_2026/2026-07-31_bo_tuning_audit.md`. The plan proper is §3
"Status and next steps" (Phases 1–4 plus the acceptance criterion). §1 is the
evidence base; §2 records what has already been changed.

## §3 is stale — read this before planning work

§3 lists Phase 1 as open, but later sections of the same document show more is
done than §3 admits:

| Section | State | What it established |
|---|---|---|
| §2d Phase-1 curvature attribution | **done** | AVR hypothesis **refuted** — `gen` is only 0.6–2.3 % of TSO curvature. `g_w_gen` stays pinned; the `avr_band` coordinate is dropped. |
| §2e Phase-1 invariance test | **done** | Invariance confirmed bit-identically (~4e-10). The breaker is the **shunt integrator** (`controller/shunt_integrator.py:313`), not `g_w_gen` or `g_z_*`. Remedy is **gauge-fixing** (dimensionless ratios about a reference point), not a wider box. |
| §2f Phase-3 `objective_curvature_inputs` | **done** | **Reverses §2d(d):** `g_w_dso_der` is the *dominant* DSO knob (77–79 %), not negligible. Also: the DSO controllers have **never been preconditioned at all**. |

§4's first open question — "whether the TSO curvature is AVR-dominated (Phase 1
decides)" — is answered by §2d(a). It is refuted.

## Genuinely open

1. **Phase 0e** — regenerate the baseline. Check first whether §2c already did
   this: `tuning/scripts/save_baseline.py` now writes
   `tuning/scripts/configs/baseline_ieee39.yaml`.
2. **Phase 2** — constrained-scalar objective. Feasibility moves into Optuna
   constraints; the scalar keeps tracking and utilisation only.
3. **Phase 3 proper** — reparameterization into per-layer loop gain, gauge-fixed
   per-class relative damping, objective priority ratios, OLTC weights by 1-D
   bisection against a taps/day target.
4. **Phase 4** — design set with an excitation gate, then the run.

**First task:** verify the true state against the code rather than trusting §3
or this summary, then propose which phase to do next and confirm before
implementing (per `.claude/CLAUDE.md`: discuss major architectural changes first).

## Key code

`tuning/` (`metrics.py`, `scenarios.py`, `parameters.py`, `ceilings.py`,
`tune.py`, `_types.py`, `_io.py`, `compat.py`, `scripts/run_tuning.py`,
`scripts/save_baseline.py`), `controller/gw_precondition.py`,
`optimisation/miqp_solver.py`, and
`experiments/run_multi_system_ofo.py::make_config` — the hand-tuned reference the
BO must beat.

## Hard constraint — do not touch PowerFactory

A 15-run PowerFactory RMS sweep runs until roughly **16:20 on 2026-07-31**.
PowerFactory permits **one** session: any second `connect()` terminates the
running one (exit 114) and destroys ~3 h of results.

Do not run anything importing `pf.session` / `pf.replay`, and do not start
`experiments/run_deadband_sweep.ps1`, `experiments/run_comparison_rms_cosim_qss.py`
or `experiments/run_rms_cosim.py`. Tuning work uses the pandapower static plant
and is safe. If unsure, ask.

## Environment

- Python: `F:\python_environments\qOFO_clean\python.exe`
  (the workstation path in `.claude/CLAUDE.md` is **not** valid on this server).
- Project root: `Z:\Python_Projekte\qOFO_GH`.
- Several sessions edit this tree concurrently — re-check files before assuming
  their contents.

## One gotcha

`docs/daily_log/07_2026/2026-07-31_remove_dso_q_integral_and_pf_probes_subpackage.md`
removed `dso_g_qi` / `dso_lambda_qi` / `dso_q_integral_max_mvar`, changing
`search_space_fingerprint()`. `tuning.tune.main` will therefore **refuse to
resume any existing persisted study**. The refusal is a false positive — the
removed entries pinned the integrator off, which every trial already ran — but
resuming needs a deliberate decision: bump `SEARCH_SPACE_VERSION` and re-stamp,
or accept a fresh study.
