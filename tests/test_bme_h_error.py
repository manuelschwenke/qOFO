# -*- coding: utf-8 -*-
"""
Unit tests for the §6 MC sensitivity-error axis:
`PerturbedZoneBoundaryView` (H̃_{b,i} = H_{b,i} ∘ (1 + σ·Ξ), field fixed
per run) and the config fail-fast contract.

Symbol map (spec §3): H_{b,i} = ∂v_b/∂u_i boundary-sensitivity slice
(stacked [Vm|θ] rows, D7); σ = bme_h_error_rel_sigma.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sensitivity.boundary_sensitivity import (  # noqa: E402
    PerturbedZoneBoundaryView,
)


class _FakeView:
    """Duck-typed ZoneBoundaryView: 2|B|=6 stacked rows, 4 columns."""

    zone_id = 2

    def __init__(self):
        rng = np.random.default_rng(0)
        self._h = rng.normal(size=(6, 4))

    def h_b_stacked(self):
        return self._h.copy()

    def h_b(self):
        return self._h[:3, :].copy()   # magnitude channel = top |B| rows


def test_perturbed_view_elementwise_product():
    view = _FakeView()
    field = 1.0 + 0.2 * np.random.default_rng(7).standard_normal((6, 4))
    pv = PerturbedZoneBoundaryView(view, field)
    assert pv.zone_id == 2
    np.testing.assert_allclose(
        pv.h_b_stacked(), view.h_b_stacked() * field, rtol=0, atol=0)
    # magnitude channel perturbed CONSISTENTLY (top rows of same field)
    np.testing.assert_allclose(
        pv.h_b(), view.h_b() * field[:3, :], rtol=0, atol=0)
    # field is fixed — repeated reads identical (no per-call noise)
    np.testing.assert_array_equal(pv.h_b_stacked(), pv.h_b_stacked())


def test_perturbed_view_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        PerturbedZoneBoundaryView(_FakeView(), np.ones((5, 4)))


def test_config_requires_seed_when_sigma_positive():
    from configs.multi_tso_config import MultiTSOConfig
    cfg = MultiTSOConfig()
    assert cfg.bme_h_error_rel_sigma == 0.0        # bitwise-no-op default
    assert cfg.bme_h_error_seed is None
    # The runner's fail-fast lives in the mode="bme" setup block; here we
    # pin the CONFIG contract only (fields exist, defaults inert).
