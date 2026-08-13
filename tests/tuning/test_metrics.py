"""Unit tests for ``tuning/metrics.py``."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from configs.config import MultiTSOConfig
from experiments.helpers.records import MultiTSOIterationRecord
from tuning.metrics import (
    INFEASIBLE_SENTINEL,
    CostWeights,
    NoiseFloors,
    TrajectoryMetrics,
    _count_oscillations,
    _count_tap_switches,
    _decimate_to_ticks,
    _itae,
    _itae_pcc_underutilization,
    _normalise,
    _tap_wear,
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
    # An empty log means nothing was simulated, so it must be inadmissible —
    # not merely expensive.  It previously scored ``w_pf`` (=100), which was
    # *cheaper* than 35–43 % of converged runs and so made divergence a
    # profitable direction for the sampler.
    assert m.cost_J == pytest.approx(INFEASIBLE_SENTINEL)
    assert not m.feasible
    assert m.infeasible_reason == "empty_log"
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
    assert _normalise(2.0, scale=4.0) == pytest.approx(0.5)


def test_normalize_maps_non_finite_to_inf() -> None:
    """A metric that could not be computed must dominate the cost.

    This previously returned ``1.0`` ("nominal-bad").  Combined with
    ``_itae`` returning ``0.0`` for an all-NaN trajectory, that made a
    diverged run cost exactly ``w_pf`` — which sat at only the ~60th
    percentile of *converged* costs, so divergence was a rewarded search
    direction.  Regression guard: keep this mapping at ``inf``.
    """
    assert _normalise(float("nan"), scale=1.0) == float("inf")
    assert _normalise(float("inf"), scale=1.0) == float("inf")


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
        "itae_v_ts", "itae_v_ds", "rmsd_v_ts", "rmsd_v_ds",
        "itae_q_pcc", "itae_q_tie", "itae_pcc_underutil",
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


# ---------------------------------------------------------------------------
# 9. Conditional DSO under-utilisation metric
# ---------------------------------------------------------------------------

def _make_underutil_log(
    n: int,
    *,
    v_offset: float,
    dso_q_act: float,
    v_set: float = 1.0,
) -> tuple[List[MultiTSOIterationRecord], np.ndarray, np.ndarray]:
    """Build (records, v_mean_ts, t_min) with constant per-step values.

    `v_offset` is the (signed) deviation of v_mean_ts from `v_set`;
    `dso_q_act` is the per-DSO actual PCC reactive injection (Mvar).
    The time grid spans 0..75 minutes with `n` evenly spaced samples.
    """
    log = [_make_record(i, dso_q_act=dso_q_act) for i in range(n)]
    v_mean = np.full(n, v_set + v_offset)
    t_min = np.linspace(0.0, 75.0, n)
    return log, v_mean, t_min


def test_itae_pcc_underutilization_zero_in_deadband() -> None:
    """Inside the voltage deadband (|v_err| ≤ deadband_v), the metric
    is zero regardless of DSO PCC injection."""
    log, v_mean, t_min = _make_underutil_log(
        n=20, v_offset=0.003, dso_q_act=0.0,    # |v_err|=3 mpu < 5 mpu
    )
    out = _itae_pcc_underutilization(
        log, v_mean, t_min, v_set=1.0, deadband_v=0.005, q_ref_mvar=100.0,
    )
    assert out == pytest.approx(0.0)


def test_itae_pcc_underutilization_zero_when_dso_at_ref() -> None:
    """When the per-DSO mean ``|Q_PCC_actual|`` is at or above
    ``q_ref_mvar``, the metric is zero regardless of TSO voltage
    stress."""
    log, v_mean, t_min = _make_underutil_log(
        n=20, v_offset=0.05, dso_q_act=120.0,   # 5% v error, but |Q|>ref
    )
    out = _itae_pcc_underutilization(
        log, v_mean, t_min, v_set=1.0, deadband_v=0.005, q_ref_mvar=100.0,
    )
    assert out == pytest.approx(0.0)


def test_itae_pcc_underutilization_known_value() -> None:
    """Constant stress (s = |v_err|−deadband = 5 mpu) and constant
    inactivity (q = q_ref − |Q_actual| = 50 Mvar) over 75 min give

        ITAE = ∫₀⁷⁵ t·(s·q) dt = 0.005 × 50 × 75²/2 = 703.125 .

    Trapezoidal integration over the 76-sample evenly spaced grid
    matches the analytic value to within numerical noise.
    """
    log, v_mean, t_min = _make_underutil_log(
        n=76,                # 1 sample per minute, inclusive of 0 and 75
        v_offset=0.01,       # |v_err|=10 mpu, deadband 5 mpu → s=5 mpu
        dso_q_act=50.0,      # |Q_actual|=50, q_ref=100 → q=50 Mvar
    )
    out = _itae_pcc_underutilization(
        log, v_mean, t_min, v_set=1.0, deadband_v=0.005, q_ref_mvar=100.0,
    )
    assert out == pytest.approx(703.125, rel=1e-6)


def test_itae_pcc_underutilization_negative_voltage_error_symmetric() -> None:
    """Voltage error sign does not matter — only its magnitude past
    the deadband."""
    log_pos, v_pos, t_min = _make_underutil_log(
        n=40, v_offset=+0.02, dso_q_act=20.0,
    )
    log_neg, v_neg, _ = _make_underutil_log(
        n=40, v_offset=-0.02, dso_q_act=20.0,
    )
    out_pos = _itae_pcc_underutilization(
        log_pos, v_pos, t_min, v_set=1.0, deadband_v=0.005, q_ref_mvar=100.0,
    )
    out_neg = _itae_pcc_underutilization(
        log_neg, v_neg, t_min, v_set=1.0, deadband_v=0.005, q_ref_mvar=100.0,
    )
    assert out_pos == pytest.approx(out_neg)
    assert out_pos > 0.0


# ---------------------------------------------------------------------------
# Regression guards for the four defects found in the 2026-07-31 BO audit.
# Each of these encodes a bug that made a tuning parameter unidentifiable or
# made divergence a profitable search direction.  See
# docs/daily_log/07_2026/2026-07-31_bo_tuning_audit.md.
# ---------------------------------------------------------------------------


def test_oscillation_survives_held_commands() -> None:
    """Defect 2: the oscillation term was identically zero in 100 % of runs.

    Commands are logged once per plant step but only change on controller
    ticks, so with ``dt_s=20`` / ``tso_period_s=180`` each command is held for
    9 records and ``Delta u`` becomes ``[0, ..., 0, D, 0, ..., 0, D, ...]``.
    ``_count_oscillations`` needs two *adjacent* above-floor deltas, so on the
    held sequence it can never fire.  Decimating to the tick grid first is what
    restores the intended semantics — this test fails if that is removed.
    """
    held = np.repeat(
        np.array([[0.0], [50.0], [-50.0], [60.0], [-60.0], [70.0]]), 9, axis=0,
    )
    active = np.zeros(held.shape[0], dtype=bool)
    active[::9] = True                       # one controller tick per hold

    assert _count_oscillations(held, 20.0) == 0, "held sequence must be blind"
    assert _count_oscillations(_decimate_to_ticks(held, active), 20.0) == 4


def test_decimate_is_a_noop_without_tick_flags() -> None:
    """Older logs carry no ``*_active`` flags; degrade gracefully, never empty."""
    seq = np.array([[1.0], [2.0], [3.0]])
    out = _decimate_to_ticks(seq, np.zeros(3, dtype=bool))
    assert out.shape == seq.shape


def test_tap_wear_is_per_transformer_not_fleet_sum() -> None:
    """Defect 3: wear limits are per transformer, so the fleet sum hides them.

    ``norm_tap = (n_tso + n_dso)/5`` also meant the two OLTC weights entered
    the cost only through their *sum* — they could not be told apart no matter
    how many trials were spent.  Here one changer hunts while the other is
    quiet; the metric must report the worst one, not the average.
    """
    taps = np.zeros((41, 2))
    taps[:, 0] = np.tile([0.0, 1.0], 21)[:41]                    # hunts
    taps[:, 1] = np.concatenate([np.zeros(20), np.ones(21)])     # one move

    ops_per_h, reversals_per_h = _tap_wear(taps, duration_s=86400.0)
    # 40 moves over 24 h on the worst changer.  Reported per HOUR, not per day:
    # the design scenarios are event-dense 75-min windows and a real day is
    # mostly quiet, so extrapolating one to the other inflates the rate ~19x.
    # The hand-tuned reference makes 5.0 taps in its window -- an unremarkable
    # 4.0/hour that read as a pathological "96/day" under the first version of
    # this metric, and tripped a wear limit it had no business tripping.
    assert ops_per_h == pytest.approx(40.0 / 24.0)   # worst changer, not mean
    assert reversals_per_h > 1.0


def test_absent_quantity_is_not_a_divergence(
    baseline_cfg: MultiTSOConfig,
) -> None:
    """A metric with nothing to measure scores 0, not ``nan``.

    Guards the distinction between "this network has no tie pairs / no PCC
    interfaces" (fine, nothing to get wrong) and "the values are non-finite"
    (divergence).  Collapsing the two marks healthy runs inadmissible.
    """
    log = [_make_record(i) for i in range(5)]
    for r in log:
        r.zone_tie_q_mvar.clear()             # no inter-zone ties at all
    m = extract_metrics(log, baseline_cfg)
    assert m.itae_q_tie == 0.0
    assert m.feasible, m.infeasible_reason


def test_diverged_log_is_infeasible_not_cheap(
    baseline_cfg: MultiTSOConfig,
) -> None:
    """Defect 1, end to end: divergence must never undercut a converged run.

    A NaN-voltage trajectory used to produce ``itae = 0`` (perfect tracking!)
    and a total cost of ``w_pf`` = 100 — cheaper than 35-43 % of *converged*
    runs, so the optimiser was rewarded for finding divergent regions.
    """
    log = [_make_record(i) for i in range(5)]
    for r in log:
        for z in r.zone_v_mean:
            r.zone_v_mean[z] = float("nan")
            r.zone_v_min[z] = float("nan")
            r.zone_v_max[z] = float("nan")

    m = extract_metrics(log, baseline_cfg)
    assert not m.feasible
    assert m.infeasible_reason == "pf_failure"
    assert m.cost_J == pytest.approx(INFEASIBLE_SENTINEL)
    assert m.cost_J > CostWeights().w_pf * 100
