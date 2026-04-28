"""Unit tests for ``tuning/metrics.py``."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers.records import MultiTSOIterationRecord
from tuning.metrics import (
    CostWeights,
    NoiseFloors,
    TrajectoryMetrics,
    _count_oscillations,
    _count_tap_switches,
    _itae,
    _normalise,
    extract_metrics,
)


# ---------------------------------------------------------------------------
# 1. Empty-log sentinel
# ---------------------------------------------------------------------------

def test_extract_metrics_empty_log_returns_sentinel(
    baseline_cfg: MultiTSOConfig,
) -> None:
    m = extract_metrics([], baseline_cfg)
    assert isinstance(m, TrajectoryMetrics)
    assert m.pf_failures == 1
    assert m.cost_J >= 1000.0
    assert m.n_records == 0
    assert m.n_tso_active == 0
    assert m.n_dso_active == 0


# ---------------------------------------------------------------------------
# 2. Below-floor noise → zero oscillations
# ---------------------------------------------------------------------------

def test_count_oscillations_below_threshold() -> None:
    rng = np.random.default_rng(0)
    # 50 steps × 1 actuator, all values within ±0.05, far below floor 1.0
    u = rng.uniform(-0.05, 0.05, size=(50, 1))
    assert _count_oscillations(u, noise_floor=1.0) == 0


# ---------------------------------------------------------------------------
# 3. Clear sign-flipping → ≥ 3 oscillations
# ---------------------------------------------------------------------------

def test_count_oscillations_above_threshold() -> None:
    # Δu sequence: +10, -20, +20, -20  → all flips above floor=1.0
    u = np.array([[0.0], [10.0], [-10.0], [10.0], [-10.0]])
    n = _count_oscillations(u, noise_floor=1.0)
    assert n >= 3


# ---------------------------------------------------------------------------
# 4. Tap-switch counting
# ---------------------------------------------------------------------------

def test_count_tap_switches() -> None:
    seq = np.array([[0.0], [1.0], [1.0], [2.0], [1.0]])
    # |Δ| = [1, 0, 1, 1] → sum = 3
    assert _count_tap_switches(seq) == 3


# ---------------------------------------------------------------------------
# 5. ITAE on zero error
# ---------------------------------------------------------------------------

def test_itae_zero_error() -> None:
    t = np.linspace(0.0, 10.0, 11)
    e = np.zeros_like(t)
    assert _itae(t, e) == 0.0


# ---------------------------------------------------------------------------
# 6. _normalise defensive behaviour
# ---------------------------------------------------------------------------

def test_normalize_with_zero_scale_returns_zero() -> None:
    assert _normalise(5.0, scale=0.0) == 0.0
    assert _normalise(5.0, scale=-1.0) == 0.0
    assert _normalise(float("nan"), scale=1.0) == 1.0
    assert _normalise(float("inf"), scale=1.0) == 1.0
    assert _normalise(2.0, scale=4.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. End-to-end synthetic log → all fields finite
# ---------------------------------------------------------------------------

def _make_record(
    step: int,
    *,
    v_mean: float = 1.03,
    der_q: float = 100.0,
    pcc_q: float = 50.0,
    v_gen: float = 1.03,
    oltc: int = 0,
    dso_tap: int = 0,
    dso_q_set: float = 5.0,
    dso_q_act: float = 5.0,
) -> MultiTSOIterationRecord:
    return MultiTSOIterationRecord(
        step=step,
        time_s=float(step) * 60.0,
        tso_active=True,
        dso_active=True,
        zone_q_der={1: np.array([der_q]),
                    2: np.array([der_q + 1.0])},
        zone_q_pcc_set={1: np.array([pcc_q]),
                        2: np.array([pcc_q])},
        zone_v_gen={1: np.array([v_gen]),
                    2: np.array([v_gen])},
        zone_oltc_taps={1: np.array([oltc]),
                        2: np.array([oltc])},
        zone_v_min={1: v_mean - 0.01, 2: v_mean - 0.01},
        zone_v_max={1: v_mean + 0.01, 2: v_mean + 0.01},
        zone_v_mean={1: v_mean,        2: v_mean},
        zone_contraction_lhs={1: 0.5, 2: 0.6},
        dso_trafo_q_set_mvar={"DSO_1": dso_q_set},
        dso_trafo_q_actual_mvar={"DSO_1": dso_q_act},
        dso_trafo_tap_pos={"DSO_1": dso_tap},
        dso_group_v_min_pu={"DSO_1": v_mean - 0.005},
        dso_group_v_max_pu={"DSO_1": v_mean + 0.005},
        dso_group_v_mean_pu={"DSO_1": v_mean},
        total_losses_mw=12.5,
    )


def test_extract_metrics_synthetic_log_runs(baseline_cfg: MultiTSOConfig) -> None:
    log: List[MultiTSOIterationRecord] = [
        _make_record(0, der_q=100.0, oltc=0),
        _make_record(1, der_q=120.0, oltc=0),
        _make_record(2, der_q= 80.0, oltc=1),       # DER flip + TSO tap step
        _make_record(3, der_q=130.0, oltc=1, dso_tap=1),
        _make_record(4, der_q= 70.0, oltc=2, dso_tap=2),
    ]
    m = extract_metrics(log, baseline_cfg)
    assert isinstance(m, TrajectoryMetrics)

    # All numeric fields finite
    for fld in (
        "itae_v_ts", "itae_v_ds", "rmsd_v_ts", "rmsd_v_ds", "itae_q_pcc",
        "rho_emp_p95", "losses_mean_mw", "cost_J",
    ):
        v = getattr(m, fld)
        assert math.isfinite(v), f"{fld} not finite: {v}"

    assert m.n_records == 5
    assert m.n_tso_active == 5
    assert m.n_dso_active == 5
    assert m.pf_failures == 0
    assert m.n_tap_switches_tso > 0  # OLTC steps in synthetic data
    assert m.n_tap_switches_dso > 0


# ---------------------------------------------------------------------------
# 8. Larger w_osc → larger J when oscillations are present
# ---------------------------------------------------------------------------

def test_cost_weights_increase_J(baseline_cfg: MultiTSOConfig) -> None:
    log: List[MultiTSOIterationRecord] = [
        _make_record(0, der_q=100.0),
        _make_record(1, der_q=200.0),
        _make_record(2, der_q=  0.0),
        _make_record(3, der_q=200.0),
        _make_record(4, der_q=  0.0),
    ]
    light = CostWeights(w_osc=1.0)
    heavy = CostWeights(w_osc=100.0)
    m_light = extract_metrics(log, baseline_cfg, weights=light)
    m_heavy = extract_metrics(log, baseline_cfg, weights=heavy)
    assert m_light.n_osc_der > 0
    assert m_heavy.cost_J > m_light.cost_J
