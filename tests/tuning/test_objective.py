"""Unit tests for ``tuning/objective.py``."""

from __future__ import annotations

import math

import numpy as np
import optuna
import pytest

from tuning.objective import cvar_aggregate, sample_params
from tuning.parameters import BO_DIMS


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

    # Pick g_w_der: low=0.1, fallback_high=1e4 → midpoint at sqrt(0.1*1e4)=10
    target = next(p for p in BO_DIMS if p.name == "g_w_der")
    samples = [t.params[target.name] for t in study.trials]
    log_mean = float(np.mean(np.log10(samples)))
    log_low = math.log10(target.low)
    log_high = math.log10(float(target.fallback_high))
    log_mid = 0.5 * (log_low + log_high)
    assert abs(log_mean - log_mid) < 0.5, (
        f"log-mean {log_mean:.2f} drifted >0.5 decades from midpoint "
        f"{log_mid:.2f} (range [{log_low:.2f}, {log_high:.2f}])"
    )
