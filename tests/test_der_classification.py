"""
Tests for the DER classification module (core/der_classification.py).
"""

from __future__ import annotations

import pytest

from core.der_classification import DERClassification, DERMode


class TestDERMode:
    def test_value_strings(self):
        assert DERMode.GRID_FORMING.value == "grid_forming"
        assert DERMode.GRID_FOLLOWING.value == "grid_following"


class TestDERClassificationDefaults:
    def test_unknown_der_id_is_grid_following(self):
        c = DERClassification()
        assert c.is_grid_following(99)
        assert not c.is_grid_forming(99)

    def test_empty_grid_forming_list(self):
        c = DERClassification()
        assert c.grid_forming_der_ids() == []
        assert c.grid_following_der_ids() == []

    def test_qv_slope_default_fallback(self):
        c = DERClassification()
        assert c.qv_slope(7, default=0.07) == pytest.approx(0.07)

    def test_qv_v_ref_default_fallback(self):
        c = DERClassification()
        assert c.qv_v_ref_init(7, default=1.03) == pytest.approx(1.03)


class TestFromDefault:
    def test_tso_indices_become_grid_forming(self):
        c = DERClassification.from_default(
            tso_der_indices=(10, 11),
            dso_der_indices=(20, 21, 22),
        )
        assert c.is_grid_forming(10)
        assert c.is_grid_forming(11)
        assert c.is_grid_following(20)
        assert c.is_grid_following(21)
        assert c.is_grid_following(22)

    def test_grid_forming_list(self):
        c = DERClassification.from_default(
            tso_der_indices=(10, 11),
            dso_der_indices=(20, 21),
        )
        assert sorted(c.grid_forming_der_ids()) == [10, 11]
        assert sorted(c.grid_following_der_ids()) == [20, 21]

    def test_int_coercion(self):
        # Pass numpy ints / strings? Numpy ints come from pandas indexes.
        import numpy as np
        c = DERClassification.from_default(
            tso_der_indices=(np.int64(10),),
            dso_der_indices=(np.int64(20),),
        )
        assert c.is_grid_forming(10)
        assert c.is_grid_following(20)


class TestOverrides:
    def test_override_flips_tso_to_grid_following(self):
        c = DERClassification.from_default(
            tso_der_indices=(10, 11),
            dso_der_indices=(20, 21),
        )
        c2 = c.with_overrides({10: "grid_following"})
        assert c2.is_grid_following(10)
        assert c2.is_grid_forming(11)
        # Original is unchanged (immutability of the operation)
        assert c.is_grid_forming(10)

    def test_override_flips_dso_to_grid_forming(self):
        c = DERClassification.from_default(
            tso_der_indices=(),
            dso_der_indices=(20,),
        )
        c2 = c.with_overrides({20: "grid_forming"})
        assert c2.is_grid_forming(20)

    def test_override_adds_unknown_der(self):
        c = DERClassification.from_default()
        c2 = c.with_overrides({99: "grid_forming"})
        assert c2.is_grid_forming(99)

    def test_override_invalid_value_raises(self):
        c = DERClassification.from_default()
        with pytest.raises(ValueError, match="Invalid DER mode"):
            c.with_overrides({1: "rogue_mode"})


class TestPromotionRecord:
    def test_record_promotion_then_lookup(self):
        c = DERClassification.from_default(tso_der_indices=(5,))
        c.record_promotion(der_id=5, gen_idx=2)
        assert c.gen_idx(5) == 2

    def test_record_promotion_for_grid_following_raises(self):
        c = DERClassification.from_default(dso_der_indices=(5,))
        with pytest.raises(ValueError, match="not classified grid-forming"):
            c.record_promotion(der_id=5, gen_idx=2)

    def test_gen_idx_before_promotion_raises(self):
        c = DERClassification.from_default(tso_der_indices=(5,))
        with pytest.raises(KeyError, match="has no recorded gen_idx"):
            c.gen_idx(5)

    def test_gen_idx_for_grid_following_raises(self):
        c = DERClassification.from_default(dso_der_indices=(5,))
        with pytest.raises(KeyError, match="not classified grid-forming"):
            c.gen_idx(5)


class TestQVOverrides:
    def test_per_der_slope_override(self):
        c = DERClassification(
            mode={20: DERMode.GRID_FOLLOWING},
            qv_slope_pu={20: 0.04},
        )
        assert c.qv_slope(20, default=0.07) == pytest.approx(0.04)
        # Unrelated DER falls back
        assert c.qv_slope(21, default=0.07) == pytest.approx(0.07)

    def test_per_der_vref_override(self):
        c = DERClassification(
            mode={20: DERMode.GRID_FOLLOWING},
            qv_v_ref_init_pu={20: 1.00},
        )
        assert c.qv_v_ref_init(20, default=1.03) == pytest.approx(1.00)
        assert c.qv_v_ref_init(21, default=1.03) == pytest.approx(1.03)
