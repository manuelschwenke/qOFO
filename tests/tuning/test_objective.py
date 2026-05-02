"""Unit tests for ``tuning/objective.py``."""

from __future__ import annotations

import math

import numpy as np
import optuna
import pytest

from tuning.metrics import CostWeights, TrajectoryMetrics, cost_components
from tuning.objective import cvar_aggregate, sample_params
from tuning.parameters import BO_DIMS, resolve_high


# ---------------------------------------------------------------------------
# 1. cvar_aggregate basic case
# ---------------------------------------------------------------------------

def test_cvar_aggregate_basic() -> None:
    # 25 % of 8 values = 2 → mean of [7, 8] = 7.5
    assert cvar_aggregate([1, 2, 3, 4, 5, 6, 7, 8], pct=25.0) == pytest.approx(7.5)

    # pct=100 → mean of all values
    assert cvar_aggregate([1, 2, 3, 4], pct=100.0) == pytest.approx(2.5)

    # pct=0 still returns at least 1 element (via ceil(0)=0 → max(1,…))
    assert cvar_aggregate([1, 2, 3], pct=0.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 2. cvar_aggregate empty
# ---------------------------------------------------------------------------

def test_cvar_aggregate_empty_returns_inf() -> None:
    assert cvar_aggregate([], pct=25.0) == math.inf


# ---------------------------------------------------------------------------
# 3. sample_params returns 8 finite keys
# ---------------------------------------------------------------------------

def test_sample_params_returns_all_keys() -> None:
    fixed = {p.name: max(p.low * 2.0, 1e-1) for p in BO_DIMS}
    trial = optuna.trial.FixedTrial(fixed)
    out = sample_params(trial, ceilings=None)

    assert set(out.keys()) == {p.name for p in BO_DIMS}
    assert len(out) == len(BO_DIMS)
    for v in out.values():
        assert math.isfinite(v)
        assert v > 0.0


# ---------------------------------------------------------------------------
# 4. sample_params spans the log-range under uniform sampling
# ---------------------------------------------------------------------------

def test_sample_params_respects_log_uniform() -> None:
    """With ``ceilings=None`` so ``fallback_high`` applies, drive a
    ``RandomSampler`` over many trials and confirm the geometric mean of
    a representative log-scale param sits within ±0.5 decades of the
    midpoint of its log-range."""
    sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    def cheap(t: optuna.Trial) -> float:
        sample_params(t, ceilings=None)
        return 0.0

    study.optimize(cheap, n_trials=200, show_progress_bar=False)

    # Pick g_w_der: log-midpoint of the resolved range should match the
    # sample geometric mean.  Use ``resolve_high(target, None)`` rather
    # than ``target.fallback_high`` directly, because numeric ``high``
    # values (e.g. when the LMI ceiling is bypassed) bypass the
    # fallback path entirely.
    target = next(p for p in BO_DIMS if p.name == "g_w_der")
    samples = [t.params[target.name] for t in study.trials]
    log_mean = float(np.mean(np.log10(samples)))
    log_low = math.log10(target.low)
    log_high = math.log10(resolve_high(target, None))
    log_mid = 0.5 * (log_low + log_high)
    assert abs(log_mean - log_mid) < 0.5, (
        f"log-mean {log_mean:.2f} drifted >0.5 decades from midpoint "
        f"{log_mid:.2f} (range [{log_low:.2f}, {log_high:.2f}])"
    )


# ---------------------------------------------------------------------------
# 5. cost_components includes the new pcc_underutil keys
# ---------------------------------------------------------------------------

def _make_traj(
    *,
    itae_pcc_underutil: float,
    cost_J: float = 0.0,
) -> TrajectoryMetrics:
    """Build a minimal :class:`TrajectoryMetrics` for ``cost_components``
    regression tests — only the fields read by ``cost_components`` need
    sensible values."""
    return TrajectoryMetrics(
        itae_v_ts=0.0, itae_v_ds=0.0,
        rmsd_v_ts=0.0, rmsd_v_ds=0.0,
        itae_q_pcc=0.0, itae_q_tie=0.0,
        itae_pcc_underutil=itae_pcc_underutil,
        n_viol_v_ts=0, n_viol_v_ds=0,
        voltage_excess_pu=0.0,
        n_osc_der=0, n_osc_pcc=0, n_osc_v_gen=0,
        n_tap_switches_tso=0, n_tap_switches_dso=0,
        osc_rate=0.0,
        rho_emp_p95=0.0, pf_failures=0,
        losses_mean_mw=0.0,
        cost_J=cost_J,
        n_records=10, n_tso_active=10, n_dso_active=10,
    )


def test_cost_components_includes_pcc_underutil_keys() -> None:
    """``cost_components`` exposes both ``norm_pcc_underutil`` and
    ``contrib_pcc_underutil`` for the new conditional DSO-utilisation
    term, regardless of whether it is active."""
    m = _make_traj(itae_pcc_underutil=0.0)
    out = cost_components(m, weights=CostWeights())
    assert "norm_pcc_underutil" in out
    assert "contrib_pcc_underutil" in out


def test_cost_components_pcc_underutil_contributes_when_active() -> None:
    """When ``itae_pcc_underutil > 0``, both the normalised value and
    the weighted contribution are positive and follow the expected
    formula ``contrib = w_pcc_underutil × itae / 1400``."""
    m = _make_traj(itae_pcc_underutil=1400.0)   # norm should be 1.0
    weights = CostWeights()
    out = cost_components(m, weights=weights)
    assert out["norm_pcc_underutil"] == pytest.approx(1.0)
    assert out["contrib_pcc_underutil"] == pytest.approx(weights.w_pcc_underutil)


def test_cost_components_zero_metrics_yield_zero_contributions() -> None:
    """All-zero ``TrajectoryMetrics`` yield all-zero ``contrib_*``
    entries, including the new term."""
    m = _make_traj(itae_pcc_underutil=0.0)
    out = cost_components(m, weights=CostWeights())
    contrib_keys = [k for k in out if k.startswith("contrib_")]
    assert contrib_keys, "cost_components returned no contrib_* keys"
    for k in contrib_keys:
        assert out[k] == pytest.approx(0.0), f"{k}={out[k]}"
