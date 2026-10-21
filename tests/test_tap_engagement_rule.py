"""
Tests for the tap engagement rule of the MIQP step (thesis eq:ofo_tap_engagement).

The step objective ``w^T G_w w + c^T w + z^T G_z z`` has a diagonal ``G_w``,
so with no output limit active it splits into one term per actuator and a free
integer column commits one step iff ``|c_i| > g_w_i``; its step is the relaxed
step ``-c_i / (2 g_w_i)`` rounded.  ``tuning.stage1_screen`` uses this rule to
price the tap classes; until 2026-10-14 it carried an extra ``||a_i||^2``.

Verifies that:
1. the controllers' ``MIQPSolver`` commits a tap exactly when ``|c| > g_w``,
   independently of the continuous actuator next to it;
2. ``engage_deviation`` and ``g_w_for_engage`` are inverse to each other and
   match ``g_w = 2 g_V |h| dv``, with no ``||a_i||^2`` term.

Author: Manuel Schwenke
Date: 2026-10-14
"""

import numpy as np
import pytest

from optimisation.miqp_solver import MIQPSolver, build_miqp_problem
from tuning.stage1_screen import (
    LayerCurvature,
    engage_deviation,
    g_w_for_engage,
)


def _solve_one_tap(c_tap: float, c_cont: float, g_w_tap: float = 1.0):
    """One continuous actuator (0) and one tap (1); output limits far away."""
    prob = build_miqp_problem(
        alpha=1.0,
        u_current=np.array([0.0, 0.0]),
        y_current=np.array([1.0]),
        H=np.array([[0.5, 2.0]]),
        grad_f=np.array([c_cont, c_tap]),
        u_lower=np.array([-10.0, -5.0]),
        u_upper=np.array([10.0, 5.0]),
        y_lower=np.array([-100.0]),
        y_upper=np.array([100.0]),
        g_w=np.array([1.0, g_w_tap]),
        g_u=0.0,
        g_z=1e6,
        integer_indices=[1],
    )
    res = MIQPSolver().solve(prob)
    assert res.is_feasible
    return float(np.atleast_1d(res.w_continuous)[0]), int(np.atleast_1d(res.w_integer)[0])


@pytest.mark.parametrize("c_tap, expected", [
    (0.80, 0), (0.95, 0), (1.05, -1), (-1.05, 1), (2.5, -1), (4.5, -2), (5.5, -3),
])
@pytest.mark.parametrize("c_cont", [0.0, 3.0])
def test_miqp_commits_iff_gradient_exceeds_weight(c_tap, expected, c_cont):
    w_cont, w_tap = _solve_one_tap(c_tap, c_cont)
    # Tap step = relaxed step -c/(2 g_w) rounded; moves iff |c| > g_w = 1.
    assert w_tap == expected
    assert w_tap == int(np.round(-c_tap / 2.0))
    # The continuous actuator does not depend on the tap decision.
    assert w_cont == pytest.approx(-c_cont / 2.0, abs=1e-6)


def _layer(h_v: float, a_sq: float) -> LayerCurvature:
    """One continuous and one tap column; S_V carries a large tap self-term."""
    S_V = np.diag([1.0, a_sq])
    return LayerCurvature(
        key="TS_1", layer="TS", area="1",
        S_V=S_V, S_Q=np.zeros((2, 2)),
        colsum_V=np.array([1.0, h_v]), maxabs_V=np.array([1.0, h_v]),
        classes=("der", "oltc"), n_v=1, n_q=0,
    )


def test_engage_rule_has_no_column_norm_term():
    g_v, d_eng, h = 50.0, 0.01, 0.12
    lc = _layer(h_v=h, a_sq=4.0)         # ||a||^2 = g_v * 4 would have mattered
    g_w = g_w_for_engage(lc, g_v, 0.0, d_eng)
    assert np.isnan(g_w[0])               # continuous column
    assert g_w[1] == pytest.approx(2.0 * g_v * h * d_eng)
    dev = engage_deviation(lc, g_v, 0.0, np.array([1.0, g_w[1]]))
    assert dev[1] == pytest.approx(d_eng)


def test_column_without_voltage_effect_is_not_achievable():
    lc = _layer(h_v=0.0, a_sq=1.0)
    assert g_w_for_engage(lc, 50.0, 0.0, 0.01)[1] == 0.0
    assert np.isnan(engage_deviation(lc, 50.0, 0.0, np.array([1.0, 1.0]))[1])
