"""The stage-1 STS curvature applies gamma_oltc_q to the Q-per-tap columns.

With the sn_mva fix (2026-11-09) the interface-Q x OLTC entries are 100x their
old size.  The DSO step sees them only through gamma_oltc_q (default 0), so the
screen must weight them the same way, or it would report a loop gain the
controller never has.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tuning import stage1_screen as s1


def _ctrl(H, gamma, n_der=2, n_oltc=1, n_shunt=0, n_if=2, n_v=3):
    cfg = SimpleNamespace(
        interface_trafo_indices=list(range(n_if)), voltage_bus_indices=list(range(n_v)),
        der_mapping=None, der_indices=list(range(n_der)),
        oltc_trafo_indices=list(range(n_oltc)), shunt_bus_indices=list(range(n_shunt)),
        gamma_oltc_q=gamma)
    return SimpleNamespace(config=cfg, _build_sensitivity_matrix=lambda: H,
                           _expand_H_to_der_level=lambda h: h)


@pytest.fixture
def H():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(5, 3))       # rows [Q_if (2) | V (3)], cols [der der oltc]
    H[:2, 2] = 6.5                     # Q per tap step, Mvar
    return H


def test_gamma_zero_removes_the_q_part_of_the_tap_column(H):
    lc = s1._dso_curvature("X", _ctrl(H, 0.0))
    assert np.all(lc.S_Q[2, :] == 0.0) and np.all(lc.S_Q[:, 2] == 0.0)
    np.testing.assert_allclose(lc.S_Q[:2, :2], H[:2, :2].T @ H[:2, :2])
    np.testing.assert_allclose(lc.S_V, H[2:].T @ H[2:])   # voltage rows untouched


def test_gamma_one_keeps_the_full_gram(H):
    lc = s1._dso_curvature("X", _ctrl(H, 1.0))
    np.testing.assert_allclose(lc.S_Q, H[:2].T @ H[:2])


def test_gamma_scales_quadratically(H):
    lc = s1._dso_curvature("X", _ctrl(H, 0.5))
    full = H[:2].T @ H[:2]
    np.testing.assert_allclose(lc.S_Q[2, 2], 0.25 * full[2, 2])
    np.testing.assert_allclose(lc.S_Q[0, 2], 0.5 * full[0, 2])
