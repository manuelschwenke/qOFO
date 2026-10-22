"""Stage I with the MSC/MSR banks as integer columns of the TS-MIQP.

Since 2026-10-22 the banks are ordinary integer inputs of the TS-MIQP, so the
analytic screen captures their columns and prices ``g_w_tso_shunt`` from a
stated engagement deviation, by the same commit rule as the taps.  These tests
pin the three things that make that true on synthetic curvature: the price
follows the rule, the design refuses to invent a bank deviation, and the
operating-point cache cannot hand back a capture made without the banks.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from tuning.stage1_screen import (
    LayerCurvature,
    OperatingPoint,
    _grams,
    _network_key,
    design_integer_weights,
    design_stage1,
    engage_deviation,
    lambda_max_B,
)

G_V = 4.0e4          # = 1 / 0.005^2, the Stage-I gauge at pi_v_ts = 1
D_ENG_SHUNT = 0.01


def _layer(key, layer, area, H_V, H_Q, classes) -> LayerCurvature:
    H_V = np.asarray(H_V, dtype=float)
    H_Q = (np.asarray(H_Q, dtype=float) if H_Q is not None
           else np.zeros((0, H_V.shape[1])))
    S_V, S_Q, colsum, maxabs = _grams(H_V, H_Q)
    return LayerCurvature(key=key, layer=layer, area=area, S_V=S_V, S_Q=S_Q,
                          colsum_V=colsum, maxabs_V=maxabs,
                          classes=tuple(classes), n_v=H_V.shape[0],
                          n_q=H_Q.shape[0])


def _ts_layer(area: str, shunt_cols) -> LayerCurvature:
    # Columns: der, pcc, gen, oltc, then one column per bank.
    base = np.array([
        [1e-3, 1e-3, 1e-4, 0.01],
        [1e-3, 1e-3, 1e-4, 0.01],
        [1e-3, 1e-3, 1e-4, 0.01],
        [1e-3, 1e-3, 1e-4, 0.01],
    ])
    H_V = base
    if len(shunt_cols):
        H_V = np.hstack([base, np.asarray(shunt_cols, dtype=float).T])
    classes = ["der", "pcc", "gen", "oltc"] + ["shunt"] * len(shunt_cols)
    return _layer(f"TS_{area}", "TS", area, H_V, None, classes)


def _sts_layer(area: str) -> LayerCurvature:
    H_V = [[1e-3, 0.01], [1e-3, 0.01], [1e-3, 0.01]]
    H_Q = [[1.0, 0.5]]
    return _layer(f"STS_{area}", "STS", area, H_V, H_Q, ["dso_der", "dso_oltc"])


def _ops() -> list[OperatingPoint]:
    op = OperatingPoint(condition="c1")
    op.layers = [
        _ts_layer("1", [[0.002, 0.004, 0.001, 0.0]]),          # colsum 0.007
        _ts_layer("2", [[0.001, 0.001, 0.001, 0.001],          # colsum 0.004
                        [0.003, 0.003, 0.002, 0.001]]),        # colsum 0.009
        _sts_layer("DSO_1"),
    ]
    return [op]


def test_bank_price_follows_the_commit_rule() -> None:
    """``g_w = 2 g_V |sum_j H_V[j,b]| d_engage``, maximised over bank columns,
    so the strongest bank commits exactly at the stated deviation."""
    lcs = [lc for op in _ops() for lc in op.layers]
    out = design_integer_weights(
        lcs, {"g_v": G_V, "g_q": 1.0, "dso_g_v": 1e4},
        {"oltc": 0.0075, "dso_oltc": 0.015, "shunt": D_ENG_SHUNT})
    assert out["shunt"] == pytest.approx(2.0 * G_V * 0.009 * D_ENG_SHUNT)


def test_design_requires_a_bank_deviation_when_banks_are_captured() -> None:
    with pytest.raises(ValueError, match="d_engage_shunt"):
        design_stage1(_ops(), d_engage_ts=0.01, d_engage_sts=0.02)


def test_design_prices_the_bank_and_round_trips_its_deviation() -> None:
    weights, report = design_stage1(_ops(), d_engage_ts=0.01,
                                    d_engage_sts=0.02,
                                    d_engage_shunt=D_ENG_SHUNT)
    assert weights["g_w_tso_shunt"] == pytest.approx(
        2.0 * weights["g_v"] * 0.009 * D_ENG_SHUNT)
    assert report["d_engage"]["shunt"] == D_ENG_SHUNT

    # The strongest bank column engages at exactly the stated deviation, the
    # weaker ones later -- the same reading the taps get.
    worst = 0.0
    for lc in (lc for op in _ops() for lc in op.layers if lc.layer == "TS"):
        dev = engage_deviation(lc, weights["g_v"], 0.0, lc.g_w_vector(weights))
        bank = [dev[i] for i, c in enumerate(lc.classes) if c == "shunt"]
        assert min(bank) >= D_ENG_SHUNT * (1 - 1e-12)
        worst = max(worst, max(bank))
    assert worst > D_ENG_SHUNT


def test_design_without_banks_prices_no_bank() -> None:
    op = OperatingPoint(condition="c1")
    op.layers = [_ts_layer("1", []), _sts_layer("DSO_1")]
    weights, _ = design_stage1([op], d_engage_ts=0.01, d_engage_sts=0.02)
    assert "g_w_tso_shunt" not in weights


def test_exclusive_group_counts_only_the_stronger_bank() -> None:
    """MSC and MSR of one STS cannot move in the same direction together, so
    the screen sees the stronger of the two, not their sum."""
    strong, weak = [0.003, 0.003, 0.002, 0.001], [0.001, 0.001, 0.001, 0.001]
    both = _ts_layer("2", [strong, weak])
    alone = _ts_layer("2", [strong])
    grouped = dataclasses.replace(both, exclusive_groups=((4, 5),))
    w = {"g_v": G_V, "g_w_der": 1.0, "g_w_pcc": 1.0, "g_w_gen": 1e8,
         "g_w_tso_oltc": 1e12, "g_w_tso_shunt": 20.0}

    def lam(lc):
        return lambda_max_B(lc, G_V, 0.0, lc.g_w_vector(w))

    assert lam(grouped) == pytest.approx(lam(alone), rel=1e-12)
    assert lam(both) > lam(grouped)


def _cfg(dispatch: str, msc_pos: int = 1, exclusion: bool = True,
         **extra) -> SimpleNamespace:
    return SimpleNamespace(
        shunt_dispatch=dispatch, shunt_sts_mutual_exclusion=exclusion,
        tso_shunt_msc_coupling_pos=msc_pos, tso_shunt_msr_coupling_pos=0,
        tso_shunt_msc_q_step_mvar=25.0, tso_shunt_msr_q_step_mvar=25.0,
        dso_der_scale=extra.get("der"), dso_load_p_scale=extra.get("load_p"),
        dso_load_q_profile_base_mvar=None)


def test_cache_key_separates_captures_with_and_without_banks() -> None:
    """A capture made before the banks entered the MIQP has fewer columns and
    must never be loaded for a controller that has them."""
    assert _network_key(_cfg("off")) == ""
    with_banks = _network_key(_cfg("miqp"))
    assert with_banks != ""
    assert with_banks != _network_key(_cfg("miqp", msc_pos=0))
    assert with_banks != _network_key(_cfg("miqp", exclusion=False))
    doubled = {"DSO_3": 2.0}
    assert (_network_key(_cfg("off", der=doubled, load_p=doubled))
            != _network_key(_cfg("miqp", der=doubled, load_p=doubled)))
