# Archived: Phase-3 observer post-processing (`analyse_observer_run.py`)

**Archived:** 2026-07-31 (was `analysis/observer/analyse_observer_run.py`).

## Why

The script had been broken at import for some time. It loaded the experiment
module by path:

```python
spec = importlib.util.spec_from_file_location(
    "experiment_000", ROOT / "experiments" / "000_M_TSO_M_DSO.py",
)
```

`experiments/000_M_TSO_M_DSO.py` no longer exists — that experiment became
`experiments/run_multi_system_ofo.py`, with the loop itself moved into
`experiments/runners/multi_tso_dso.py`.

## Why it was not simply repointed

The script captures the observer instance by monkey-patching
`mod.attach_observer` and then calling `mod.run_multi_tso_dso(CFG)`. The
current runner, `experiments/runners/multi_tso_dso.py`, does **not** import
or call `attach_observer` — that factory lives in
`analysis/observer/stability_integration_ieee39.py` and is today invoked
directly by callers such as `analysis/observer/demo_wind_replace_observer.py`.

Patching an attribute the runner never reads is a no-op, so a repointed
script would fall straight through to its own guard at line 109:

```python
raise RuntimeError("observer was never instantiated; check integration")
```

Reviving the analysis therefore needs the observer hook re-integrated into the
current runner (a design decision about where the attach point belongs), not a
path fix. Archived rather than half-fixed.

## What it produced

1. Slack-ratio table `g_w_current / g_w_spectral_gap_p95` per zone × block,
   classified as spectral-gap-certified / box-regularised / strongly
   box-regularised.
2. `||M||_op` distribution per zone (mean, p95, max) from
   `observer.trajectories`.
3. Empirical contraction rate `rho_k` per zone from
   `MultiTSOIterationRecord`.
4. JSON summary + PNG plot.

The live entry point for observer-equipped runs is
`analysis/observer/demo_wind_replace_observer.py`, which calls
`attach_observer` directly.

## Note

The config literal in this script was edited on 2026-07-31 to drop
`dso_g_qi` / `dso_lambda_qi` / `dso_q_integral_max_mvar` when the DSO
integral Q-tracking term was removed, so it is at least constructor-valid
against the current `MultiTSOConfig`. See
`docs/daily_log/07_2026/2026-07-31_remove_dso_q_integral_and_pf_probes_subpackage.md`.
