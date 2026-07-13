"""
SBX-V Phase 1 tests — directions, band, configuration.

Covers hard rule 8 (single sign mapping, DP3 as confirmed 2026-07-09),
V-D2 (band presets, AR Anhang C spread assertion) and the plan §8
configuration assertions.
"""

from __future__ import annotations

import pytest

from sbx_h.fail import SBXError
from sbx_v.band import (MIN_PRESET_SPREAD_MVAR, NormalBand, band_from_config)
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction, signed_q_hv_mvar, split_signed_q_hv


# ----------------------------------------------------------------------
#  Directions (DP3)
# ----------------------------------------------------------------------

class TestDirections:
    def test_dp3_signs(self):
        # Positive netted q_hv (TS -> DS) = DS under-excited = LOWERING;
        # negative = DS injects into the TS = RAISING.
        assert signed_q_hv_mvar(Direction.LOWERING, 30.0) == +30.0
        assert signed_q_hv_mvar(Direction.RAISING, 30.0) == -30.0

    def test_round_trip(self):
        for d in Direction:
            q = signed_q_hv_mvar(d, 12.5)
            d2, m2 = split_signed_q_hv(q)
            assert d2 is d
            assert m2 == 12.5

    def test_zero_resolves_to_lowering(self):
        d, m = split_signed_q_hv(0.0)
        assert d is Direction.LOWERING
        assert m == 0.0

    def test_negative_magnitude_rejected(self):
        with pytest.raises(SBXError):
            signed_q_hv_mvar(Direction.RAISING, -1.0)

    def test_non_finite_rejected(self):
        with pytest.raises(SBXError):
            split_signed_q_hv(float("nan"))

    def test_opposite(self):
        assert Direction.RAISING.opposite is Direction.LOWERING
        assert Direction.LOWERING.opposite is Direction.RAISING


# ----------------------------------------------------------------------
#  Band (V-D2)
# ----------------------------------------------------------------------

class TestBand:
    def test_ar41414_is_the_default_preset(self):
        # V-D2 as revised 2026-07-10: the operative Normalbereich is
        # the AR 4141-4 preset; it requires a contracted P.
        assert SBXVConfig().band_preset == "ar41414_default"
        band = band_from_config(SBXVConfig(), "area_1",
                                contracted_p_mw=1000.0)
        assert band.q_raise_mvar == pytest.approx(50.0)
        assert band.q_lower_mvar == pytest.approx(100.0)

    def test_fixed_preset_explicit(self):
        band = band_from_config(SBXVConfig(band_preset="fixed"), "area_1")
        assert band.q_raise_mvar == 50.0
        assert band.q_lower_mvar == 50.0
        assert band.spread_mvar == 100.0

    def test_signed_edges(self):
        band = NormalBand("a", 40.0, 60.0)
        assert band.signed_edge_mvar(Direction.RAISING) == -40.0
        assert band.signed_edge_mvar(Direction.LOWERING) == +60.0

    def test_zero_band_allowed_when_explicit(self):
        # E2 sweeps band = 0; explicit values carry no spread assertion.
        cfg = SBXVConfig(band_preset="fixed",
                         band_q_raise_mvar=0.0, band_q_lower_mvar=0.0)
        band = band_from_config(cfg, "area_1")
        assert band.spread_mvar == 0.0

    def test_ar41414_preset(self):
        cfg = SBXVConfig(band_preset="ar41414_default")
        band = band_from_config(cfg, "area_1", contracted_p_mw=1000.0)
        assert band.q_raise_mvar == pytest.approx(50.0)   # 5 %
        assert band.q_lower_mvar == pytest.approx(100.0)  # 10 %
        assert band.spread_mvar >= MIN_PRESET_SPREAD_MVAR

    def test_ar41414_spread_assertion(self):
        cfg = SBXVConfig(band_preset="ar41414_default")
        # 0.15 * 400 = 60 Mvar spread < 70 -> AR Anhang C violation.
        with pytest.raises(SBXError):
            band_from_config(cfg, "area_1", contracted_p_mw=400.0)

    def test_ar41414_needs_contracted_p(self):
        cfg = SBXVConfig(band_preset="ar41414_default")
        with pytest.raises(SBXError):
            band_from_config(cfg, "area_1")

    def test_negative_edge_rejected(self):
        with pytest.raises(SBXError):
            NormalBand("a", -1.0, 50.0)


# ----------------------------------------------------------------------
#  Configuration (§8)
# ----------------------------------------------------------------------

class TestConfig:
    def test_v1_defaults_valid(self):
        cfg = SBXVConfig()
        assert cfg.k_window == 5           # 900 s / 180 s
        assert cfg.n_persist == 1          # 180 s / 180 s
        assert cfg.dq_grant_mvar == 30.0

    def test_dp2_conversion(self):
        cfg = SBXVConfig()
        # Anchor arithmetic: 10 EUR/Mvarh * 0.05 h * (250/15) per EUR.
        assert cfg.c_ug_obj_per_mvar_step == pytest.approx(
            10.0 * 0.05 * 250.0 / 15.0)
        assert cfg.c_vh_obj_per_mvar_step == pytest.approx(
            5.0 * 0.05 * 250.0 / 15.0)
        assert cfg.c_ug_obj_per_mvar_step > cfg.c_vh_obj_per_mvar_step

    def test_grenz_must_exceed_avg(self):
        with pytest.raises(SBXError):
            SBXVConfig(price_arb_grenz_eur_per_mvarh=5.0)  # == avg
        with pytest.raises(SBXError):
            SBXVConfig(price_lp_grenz_eur_per_mvar_day=25.0)

    def test_window_must_be_iteration_multiple(self):
        with pytest.raises(SBXError):
            SBXVConfig(window_s=1000.0)  # not a multiple of 180 s

    def test_persistence_must_be_iteration_multiple(self):
        with pytest.raises(SBXError):
            SBXVConfig(t_persist_s=200.0)

    def test_unknown_cost_model_rejected(self):
        with pytest.raises(SBXError):
            SBXVConfig(miqp_cost_model="free_lunch")

    def test_unknown_preset_rejected(self):
        with pytest.raises(SBXError):
            SBXVConfig(band_preset="whatever")
