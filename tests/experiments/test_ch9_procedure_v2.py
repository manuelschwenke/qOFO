"""Procedure v2 (ch_9_8_procedure_v2): STS weights held in the search, engagement
bounds, the choice among the verified STS variants, and the Stage-I design."""
import pickle

import pytest

from experiments.ch_09_parameter_selection import ch_9_8_procedure_v2 as V


def test_the_search_holds_the_sts_weights_and_keeps_the_v1_ts_spans():
    from tuning.stage2_search import DEFAULT_SPANS
    spans = V.spans_v2()
    assert all(spans[k] == 0.0 for k in V.STS_FIELDS)
    assert all(spans[k] == DEFAULT_SPANS[k] for k in DEFAULT_SPANS if k not in V.STS_FIELDS)
    assert "g_v" not in spans


def test_engagement_bounds():
    assert V.admitted(0.03, 0.0075) and V.admitted(0.02, 0.009) and V.admitted(0.04, 0.001)
    assert not V.admitted(0.03, 0.0091)
    assert not V.admitted(0.0199, 0.005) and not V.admitted(0.0401, 0.005)


def test_the_largest_rho_within_the_margins_is_chosen():
    ok = {"p95": 24.0, "median_residual_mvar": 0.5, "pass": True}
    assert V.choose_variant({10.0: {"p95": 26.0, "median_residual_mvar": 0.6, "pass": True},
                             20.0: ok, 30.0: ok}) == 20.0
    assert V.choose_variant({10.0: {"p95": 25.0, "median_residual_mvar": 0.8, "pass": True},
                             20.0: ok}) == 10.0
    assert V.choose_variant({10.0: {"p95": 24.0, "median_residual_mvar": 0.81, "pass": True},
                             30.0: {"pass": False, "error": "no run"}}) is None


def _ops():
    files = sorted(V.OPS_SOURCE.glob("op_*.npz"))
    if len(files) != 6:
        pytest.skip("run-0001 operating points not available here")
    return [pickle.load(f.open("rb")) for f in files]


def test_stage1_designs_the_sts_and_reproduces_the_v1_ts_part():
    ops = _ops()
    designs = V.design_variants(ops)                 # raises if the TS part differs
    for pi_q, (w, _) in designs.items():
        d = V.describe(w, ops)
        assert d["rho"] == pytest.approx(2.5e5 / pi_q, rel=1e-9)
        assert d["theta_sts_min"] == pytest.approx(V.D_ENGAGE["sts"], rel=1e-6)
        assert d["theta_ts_min"] == pytest.approx(V.D_ENGAGE["ts"], rel=1e-6)
        assert d["lambda_max_B_sts"] == pytest.approx(V.LAMBDA_TARGET["sts"], rel=1e-6)
        assert V.admitted(d["theta_sts_min"], d["theta_ts_min"])
