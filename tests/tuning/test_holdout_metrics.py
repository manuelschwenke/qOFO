"""Tests for ``tuning/holdout_metrics.py``.

The property that matters: the interface-Q error series used for the holdout RMS
must be the *same* series ``metrics._itae_q_pcc`` / ``_itae_q_tie`` consume.  The
RMS lives in a separate module (so that adding it cannot perturb a running tuning
job), which creates the risk of the two drifting apart.  The consistency tests
below close that gap by reconstructing the ITAE from the exported series and
comparing it against the metrics module's own value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tuning.holdout_metrics import (
    q_pcc_error_series,
    q_tie_error_series,
    rms_of_series,
)
from tuning.metrics import _itae, _itae_q_pcc, _itae_q_tie


class _Rec:
    """Minimal stand-in for ``MultiTSOIterationRecord``."""

    def __init__(self, time_s, q_set=None, q_act=None, tie=None, tie_set=None):
        self.time_s = time_s
        self.dso_trafo_q_set_mvar = q_set if q_set is not None else {}
        self.dso_trafo_q_actual_mvar = q_act if q_act is not None else {}
        self.zone_tie_q_mvar = tie if tie is not None else {}
        if tie_set is not None:
            self.zone_tie_q_set_mvar = tie_set


def _pcc_records():
    return [
        _Rec(0.0,   {0: 10.0, 1: 20.0}, {0: 12.0, 1: 17.0}),   # errs 2, 3 -> 2.5
        _Rec(60.0,  {0: 10.0, 1: 20.0}, {0: 10.0, 1: 24.0}),   # errs 0, 4 -> 2.0
        _Rec(120.0, {0: 10.0, 1: 20.0}, {0: 16.0, 1: 20.0}),   # errs 6, 0 -> 3.0
    ]


def test_pcc_series_is_the_per_step_mean_absolute_error() -> None:
    t_min, err, has = q_pcc_error_series(_pcc_records())
    assert has
    assert t_min == pytest.approx([0.0, 1.0, 2.0])
    assert err == pytest.approx([2.5, 2.0, 3.0])


def test_pcc_series_reproduces_the_metrics_itae() -> None:
    """Same signal, verified through the metrics module's own ITAE."""
    recs = _pcc_records()
    t_min, err, _has = q_pcc_error_series(recs)
    assert _itae(t_min, err) == pytest.approx(_itae_q_pcc(recs))


def test_tie_series_reproduces_the_metrics_itae() -> None:
    recs = [
        _Rec(0.0,   tie={("a", "b"): 5.0, ("b", "c"): -3.0}),
        _Rec(60.0,  tie={("a", "b"): 1.0, ("b", "c"): 7.0}),
        _Rec(120.0, tie={("a", "b"): 0.0, ("b", "c"): 0.0}),
    ]
    t_min, err, has = q_tie_error_series(recs)
    assert has
    # Default setpoint is 0 Mvar, so the error is |q|.
    assert err == pytest.approx([4.0, 4.0, 0.0])
    assert _itae(t_min, err) == pytest.approx(_itae_q_tie(recs))


def test_tie_series_uses_per_pair_setpoints_when_present() -> None:
    recs = [
        _Rec(0.0,  tie={("a", "b"): 5.0}, tie_set={("a", "b"): 4.0}),
        _Rec(60.0, tie={("a", "b"): 5.0}, tie_set={("a", "b"): 9.0}),
    ]
    _t, err, _has = q_tie_error_series(recs)
    assert err == pytest.approx([1.0, 4.0])


def test_rms_is_the_root_mean_square_not_the_mean() -> None:
    err = np.array([0.0, 4.0])          # mean 2.0, rms sqrt(8)=2.828...
    assert rms_of_series(err, True) == pytest.approx(math.sqrt(8.0))


def test_rms_ignores_nan_steps() -> None:
    err = np.array([3.0, np.nan, 4.0])
    assert rms_of_series(err, True) == pytest.approx(math.sqrt(12.5))


def test_no_interfaces_scores_zero_not_nan() -> None:
    """A network with nothing to track has not got anything wrong.

    Collapsing this into ``nan`` would mark an interface-free run inadmissible,
    the same distinction ``metrics._itae_q_pcc`` draws.
    """
    recs = [_Rec(0.0), _Rec(60.0)]
    _t, err, has = q_pcc_error_series(recs)
    assert has is False
    assert rms_of_series(err, has) == 0.0


def test_diverged_series_is_nan_not_zero() -> None:
    """Interfaces present but non-finite must not read as perfect tracking.

    Returning 0.0 here is the defect that once made divergence a *rewarded*
    search direction (see ``metrics._itae``); nan propagates to inf instead.
    """
    err = np.array([np.nan, np.nan, np.nan])
    assert math.isnan(rms_of_series(err, True))


def test_single_finite_sample_is_nan_to_match_itae() -> None:
    """One surviving sample is nan, as ITAE's trapezoid rule also requires two.

    Without this floor a near-total divergence would score a finite RMS while
    ITAE reported nan, so the two statistics would disagree on admissibility.
    """
    err = np.array([np.nan, 5.0, np.nan])
    assert math.isnan(rms_of_series(err, True))


def test_empty_records_do_not_raise() -> None:
    t_min, err, has = q_pcc_error_series([])
    assert has is False
    assert t_min.size == 0 and err.size == 0
    assert rms_of_series(err, has) == 0.0
