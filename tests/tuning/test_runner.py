"""Unit tests for ``tuning/runner.py``."""

from __future__ import annotations

import math

import pytest

from configs.multi_tso_config import MultiTSOConfig
from tuning.metrics import TrajectoryMetrics
from tuning.parameters import params_from_config
from tuning.runner import RunResult, run_one
from tuning.scenarios import design_set


# ---------------------------------------------------------------------------
# 1. End-to-end smoke: nominal_quiet runs cleanly
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_one_smoke(baseline_cfg: MultiTSOConfig) -> None:
    """One-trial end-to-end test — the Task-2 acceptance gate.

    Uses the ``nominal_quiet`` design scenario (1 h, no contingencies)
    and the baseline-config-derived params.  Asserts the run completes
    without raising and that all :class:`TrajectoryMetrics` fields are
    finite.
    """
    params = params_from_config(baseline_cfg)
    scenario = next(s for s in design_set() if s.name == "nominal_quiet")

    result = run_one(params, scenario, baseline_cfg)

    assert isinstance(result, RunResult)
    assert result.scenario_name == "nominal_quiet"
    assert result.failure_reason == "", (
        f"Smoke test caught a failure: {result.failure_reason[:500]}"
    )
    assert result.metrics.pf_failures == 0
    assert result.metrics.n_records > 0
    assert result.wall_time_s > 0.0

    # Every numeric metric must be finite (sentinel paths are NOT taken
    # on a successful run).
    m: TrajectoryMetrics = result.metrics
    for fld in (
        "itae_v_ts", "itae_v_ds", "rmsd_v_ts", "rmsd_v_ds", "itae_q_pcc",
        "rho_emp_p95", "losses_mean_mw", "cost_J",
    ):
        v = getattr(m, fld)
        assert math.isfinite(v), f"{fld} not finite: {v}"


# ---------------------------------------------------------------------------
# 2. Bad params do not crash run_one
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_one_handles_bad_params_gracefully(
    baseline_cfg: MultiTSOConfig,
) -> None:
    """A pathologically tiny ``g_w_der`` may diverge or recover; either
    way :func:`run_one` must not raise."""
    params = params_from_config(baseline_cfg)
    params["g_w_der"] = 1e-6
    scenario = next(s for s in design_set() if s.name == "nominal_quiet")

    result = run_one(params, scenario, baseline_cfg)

    assert isinstance(result, RunResult)
    # Either the run completes or the failure is captured -- never raised.
    if result.failure_reason:
        assert result.metrics.pf_failures >= 1
        assert result.metrics.cost_J >= 1000.0


# ---------------------------------------------------------------------------
# 3. apply_to_config validation errors propagate
# ---------------------------------------------------------------------------

def test_run_one_invalid_param_keys_raises(
    baseline_cfg: MultiTSOConfig,
) -> None:
    """Validation errors from :func:`apply_to_config` are intentionally
    NOT swallowed by :func:`run_one` — only simulator-time exceptions
    are caught."""
    bad = params_from_config(baseline_cfg)
    bad["not_a_real_param"] = 1.0
    scenario = next(s for s in design_set() if s.name == "nominal_quiet")

    with pytest.raises(ValueError, match="Unknown BO params"):
        run_one(bad, scenario, baseline_cfg)
