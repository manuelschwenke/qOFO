from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from configs.config import MultiTSOConfig
from tuning.stability_certificate.bo import run_stability_bo
from tuning.stability_certificate.candidate import evaluate_candidate
from tuning.stability_certificate.snapshot import (
    CACHE_VERSION,
    CachedCurvatureSnapshot,
    candidate_gw_lists,
)


def _snapshot() -> CachedCurvatureSnapshot:
    return CachedCurvatureSnapshot(
        cache_version=CACHE_VERSION,
        cache_key="synthetic",
        generated_at="2026-07-15T00:00:00+00:00",
        zone_ids=(1,),
        H_blocks={(1, 1): np.array([[1.0, 0.2, 0.001]])},
        Q_obj_list=(np.array([1.0]),),
        baseline_gw_list=(np.array([1.0, 2.0, 100.0]),),
        tso_gw_fields=(("g_w_der", "g_w_pcc", "g_w_gen"),),
        actuator_counts=(
            {
                "n_der": 1,
                "n_pcc": 1,
                "n_gen": 1,
                "n_oltc": 0,
                "n_shunt": 0,
            },
        ),
        dso_models=(),
        baseline_gw_parameters={
            "g_w_der": 1.0,
            "g_w_pcc": 2.0,
            "g_w_gen": 100.0,
        },
        tso_period_s=180.0,
        dso_period_s=20.0,
        baseline_c3_gamma=0.0,
        baseline_c3_certified=True,
    )


def _config() -> MultiTSOConfig:
    return MultiTSOConfig(
        g_w_der=1.0,
        g_w_pcc=2.0,
        g_w_gen=100.0,
    )


def test_candidate_reweights_classes_but_preserves_generator() -> None:
    snapshot = _snapshot()
    candidate = dataclasses.replace(
        _config(),
        g_w_der=0.5,
        g_w_pcc=4.0,
    )

    tso, dso = candidate_gw_lists(snapshot, candidate)

    np.testing.assert_allclose(tso[0], [0.5, 4.0, 100.0])
    assert dso == ()


def test_candidate_rejects_changed_generator_weight() -> None:
    snapshot = _snapshot()
    candidate = dataclasses.replace(_config(), g_w_gen=99.0)

    with pytest.raises(ValueError, match="g_w_gen is fixed"):
        candidate_gw_lists(snapshot, candidate)


def test_candidate_rebuilds_curvature_and_reruns_lmis() -> None:
    evaluation = evaluate_candidate(
        _snapshot(),
        _config(),
        {
            "g_w_der": 1.0,
            "g_w_pcc": 2.0,
            "g_w_dso_der": 10.0,
        },
    )

    assert evaluation.fixed_g_w_gen == 100.0
    assert evaluation.n_coupled_neutral == 2
    assert evaluation.local_lmi_certified["TSO zone 1"]
    assert evaluation.coupled_active_lmi_certified
    assert evaluation.c3_gamma == pytest.approx(0.0)
    assert evaluation.c3_certified
    assert evaluation.all_candidate_lmis_certified

def test_short_bo_preserves_generator_and_evaluates_baseline() -> None:
    result = run_stability_bo(
        _snapshot(),
        _config(),
        n_trials=3,
        seed=3,
    )

    assert result.n_trials == 3
    assert result.baseline.params == {
        "g_w_der": 1.0,
        "g_w_pcc": 2.0,
        "g_w_dso_der": 10.0,
    }
    assert all(trial.fixed_g_w_gen == 100.0 for trial in result.trials)
    assert result.best.all_candidate_lmis_certified