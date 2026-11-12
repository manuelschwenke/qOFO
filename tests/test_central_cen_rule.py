"""CEN (M) weight rule of 2026-11-10: block-wise loop gain and the owner's per-switch g_w
(``experiments.runners.multi_tso_dso._central_block_g_w`` / ``_central_owner_switch_g_w``)."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.runners.multi_tso_dso import (
    _block_lambda, _central_block_g_w, _central_column_groups, _central_owner_switch_g_w)

# Layout [Q_DER (2 TS-IBG, 2 STS-IBG) | V_gen (1) | tap (1) | bank (1) | coupler (1)]
CFG = SimpleNamespace(der_indices=[10, 11, 20, 21], gen_indices=[0], oltc_trafo_indices=[5],
                      shunt_bus_indices=[7], oltc_trafo3w_indices=[3],
                      voltage_bus_indices=[100, 101, 102, 200, 201])
META = SimpleNamespace(tso_der_indices=[10, 11])
RNG = np.random.default_rng(3)
H = RNG.normal(size=(5, 8))
G_V = np.array([4.0, 4.0, 4.0, 1.0, 1.0])
G_W0 = np.array([0.5, 0.6, 2.0, 2.5, 3.0, 9.0, 20.0, 11.0])


def test_column_groups():
    g = _central_column_groups(CFG, META)
    assert g == {"ts": [0, 1, 4], "sts": [2, 3], "switch": [5, 6], "coupler": [7]}


def test_block_rule_hits_the_targets_and_caps_the_joint_gain():
    g = _central_column_groups(CFG, META)
    gw, info = _central_block_g_w(H, G_V, G_W0, g, 1.45, 1.40)
    cont = g["ts"] + g["sts"]
    assert _block_lambda(H, G_V, gw, cont) <= 1.45 + 1e-9
    # one common factor after the block targets: their ratio is kept
    assert _block_lambda(H, G_V, gw, g["ts"]) / _block_lambda(H, G_V, gw, g["sts"]) \
        == pytest.approx(1.45 / 1.40)
    kappa_all = max(_block_lambda(H, G_V, G_W0, cont) / 1.45, 1.0)
    assert info["kappa_all"] == pytest.approx(kappa_all)
    assert np.allclose(gw[5:], G_W0[5:] * kappa_all)


def test_owner_rule_prices_a_switch_relative_to_the_owners_rows():
    zones = {1: SimpleNamespace(v_bus_indices=[100, 101], oltc_trafo_indices=[5], shunt_bus_indices=[7])}
    gw, fac = _central_owner_switch_g_w(H, G_V, G_W0, CFG, zones, 8.0, 20.0)
    for kind, j, own in (("tap 5", 5, 8.0), ("bank 7", 6, 20.0)):
        ratio = np.sum(G_V * H[:, j] ** 2) / np.sum(G_V[:2] * H[:2, j] ** 2)
        assert fac[kind] == pytest.approx(ratio) and ratio >= 1.0
        assert gw[j] == pytest.approx(own * ratio)
    assert np.allclose(gw[:5], G_W0[:5]) and gw[7] == G_W0[7]


def test_owner_rule_keeps_a_device_without_owner_curvature():
    Hz = H.copy()
    Hz[:, 5] = 0.0
    zones = {1: SimpleNamespace(v_bus_indices=[100, 101], oltc_trafo_indices=[5], shunt_bus_indices=[7])}
    gw, fac = _central_owner_switch_g_w(Hz, G_V, G_W0, CFG, zones, 8.0, 20.0)
    assert "tap 5" not in fac and gw[5] == G_W0[5]
