"""Procedure v5 (ch_9_17): the g_w_pcc bounds and the module patches."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ch_09_parameter_selection import ch_9_6_full_procedure as P
from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V2
from experiments.ch_09_parameter_selection import ch_9_11_procedure_v3 as V3
from experiments.ch_09_parameter_selection import ch_9_14_procedure_v4 as V4
from experiments.ch_09_parameter_selection import ch_9_17_procedure_v5 as V5


def _op(name, s_pcc, g3_on):
    """A TS_2 layer with two PCC columns (Gram ``s_pcc``) and the G3 column."""
    S = np.zeros((3, 3))
    S[:2, :2] = s_pcc
    S[2, 2] = 1.0 if g3_on else 0.0
    lc = SimpleNamespace(key="TS_2", classes=("pcc", "pcc", "gen"), S_V=S)
    return SimpleNamespace(condition=name, layers=[lc])


@pytest.fixture
def v5():
    prev = V5.configure(["sts_coupler_hold_after_bank=1"])
    yield V5
    V5.restore(prev)


def test_configure_and_restore_leave_v2_v3_v4_unchanged():
    before = (V2.passes_screen, V2.stage2a, V2.spans_v2, V2.load_ops, V2.RESULT, V2.OPS_SOURCE,
              V3.design, V3.write_result, V3.RESULT, V3.VARIANT_NAME, V4.RESULT, P.REMOTE_ROOT)
    prev = V5.configure([])
    assert V2.passes_screen is V5.passes_screen and V3.design is V5.design
    assert V3.RESULT == V2.RESULT == V4.RESULT == V5.RESULT
    assert P.REMOTE_ROOT == V5.REMOTE_ROOT
    V5.restore(prev)
    after = (V2.passes_screen, V2.stage2a, V2.spans_v2, V2.load_ops, V2.RESULT, V2.OPS_SOURCE,
             V3.design, V3.write_result, V3.RESULT, V3.VARIANT_NAME, V4.RESULT, P.REMOTE_ROOT)
    assert before == after


def test_bounds_from_the_g3_off_points_only(v5):
    g_v = 4.0e4
    off_a = _op("a", np.diag([2e-6, 1e-6]), g3_on=False)     # kappa = g_v * 2e-6 = 0.08
    off_b = _op("b", np.diag([4e-6, 1e-6]), g3_on=False)     # kappa = 0.16
    on = _op("c", np.diag([1e-4, 1e-4]), g3_on=True)          # ignored by the bounds
    V4.LAM_OPS[:] = [off_a, off_b, on]
    b = v5.compute_bounds(g_v)
    assert b["n_scarce_points"] == 2
    assert b["kappa_min"] == pytest.approx(0.08) and b["kappa_max"] == pytest.approx(0.16)
    assert b["g_w_pcc_hi"] == pytest.approx(0.08 / v5.LAMBDA_STS_MIN)
    assert b["g_w_pcc_lo"] == pytest.approx(0.16 / b["ceiling"])
    assert b["g_w_pcc_centre"] == pytest.approx(np.sqrt(b["g_w_pcc_lo"] * b["g_w_pcc_hi"]))
    assert b["span_decades"] == pytest.approx(0.5 * np.log10(b["g_w_pcc_hi"] / b["g_w_pcc_lo"]))


def test_responsiveness_bound_and_span(v5):
    V4.LAM_OPS[:] = [_op("a", np.diag([2e-6, 1e-6]), g3_on=False)]
    b = v5.compute_bounds(4.0e4)
    w = {"g_v": 4.0e4, "g_w_pcc": b["g_w_pcc_hi"]}
    assert v5.lambda_sts(w) == pytest.approx(v5.LAMBDA_STS_MIN)
    assert v5.responsive(w)
    assert not v5.responsive(dict(w, g_w_pcc=1.01 * b["g_w_pcc_hi"]))
    assert V2.spans_v2()["g_w_pcc"] == pytest.approx(b["span_decades"])
    assert V2.spans_v2()["g_w_der"] == V5._ORIG["V2.spans_v2"]()["g_w_der"]


def test_no_g3_off_point_is_an_error(v5):
    V4.LAM_OPS[:] = [_op("c", np.diag([1e-4, 1e-4]), g3_on=True)]
    with pytest.raises(RuntimeError, match="no G3-off point"):
        v5.compute_bounds(4.0e4)


def test_bank_grid_has_both_horizons():
    names = [n for n, _, _ in V5.bank_conditions()]
    assert len(names) == 192 and len(set(names)) == 192
    assert sum(n.endswith("_12h") for n in names) == sum(n.endswith("_3h") for n in names) == 96
