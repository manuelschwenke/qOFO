"""
Unit tests for ``network.ieee39.build.tag_der_q_modes`` (refactor_v2 commit 2).

The function is additive: it only writes columns onto ``net.sgen`` and
does not modify the meta dataclass.  These tests verify that the
hierarchy resolution (per-DER override > level default) and column
defaults are correct.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from network.ieee39.build import build_ieee39_net, tag_der_q_modes
from network.ieee39.hv_networks import add_hv_networks


@pytest.fixture()
def base_net_meta():
    """Fresh wind_replace IEEE39 + HV sub-networks per test."""
    net, meta = build_ieee39_net(scenario="wind_replace", verbose=False)
    meta = add_hv_networks(net, meta, verbose=False)
    return net, meta


# ---------------------------------------------------------------------------
#  Default behaviour
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_adds_required_columns(self, base_net_meta):
        net, meta = base_net_meta
        tag_der_q_modes(net, meta)
        for col in [
            "q_mode", "qv_slope_pu", "qv_vref_pu", "qv_deadband_pu",
            "cosphi", "cosphi_sign", "q_set_mvar", "qv_vref_anchor_pu",
        ]:
            assert col in net.sgen.columns

    def test_all_qv_by_default(self, base_net_meta):
        net, meta = base_net_meta
        tag_der_q_modes(net, meta)
        for s in list(meta.tso_der_indices) + list(meta.dso_der_indices):
            assert net.sgen.at[s, "q_mode"] == "qv"

    def test_default_qv_parameters(self, base_net_meta):
        import math
        net, meta = base_net_meta
        tag_der_q_modes(net, meta)
        for s in list(meta.tso_der_indices) + list(meta.dso_der_indices):
            assert net.sgen.at[s, "qv_slope_pu"] == pytest.approx(0.07)
            assert net.sgen.at[s, "qv_vref_pu"] == pytest.approx(1.00)
            # Deadband default is 0.005 p.u. (w-shift configuration).
            assert net.sgen.at[s, "qv_deadband_pu"] == pytest.approx(0.005)
            assert net.sgen.at[s, "cosphi"] == pytest.approx(1.0)
            assert net.sgen.at[s, "cosphi_sign"] == -1
            assert net.sgen.at[s, "q_set_mvar"] == pytest.approx(0.0)
            # qv_vref_anchor_pu is NaN until the first apply step.
            assert math.isnan(float(net.sgen.at[s, "qv_vref_anchor_pu"]))


# ---------------------------------------------------------------------------
#  Level defaults (TSO vs DSO) and per-DER overrides
# ---------------------------------------------------------------------------

class TestLevelDefaults:
    def test_tso_dso_independent(self, base_net_meta):
        net, meta = base_net_meta
        tag_der_q_modes(
            net, meta,
            tso_q_mode="cosphi",
            dso_q_mode="qv",
            tso_cosphi=0.92,
            dso_qv_slope_pu=0.04,
        )
        for s in meta.tso_der_indices:
            assert net.sgen.at[s, "q_mode"] == "cosphi"
            assert net.sgen.at[s, "cosphi"] == pytest.approx(0.92)
        for s in meta.dso_der_indices:
            assert net.sgen.at[s, "q_mode"] == "qv"
            assert net.sgen.at[s, "qv_slope_pu"] == pytest.approx(0.04)


class TestOverrides:
    def test_per_der_q_mode_override(self, base_net_meta):
        net, meta = base_net_meta
        first_dso = int(meta.dso_der_indices[0])
        tag_der_q_modes(
            net, meta,
            dso_q_mode="qv",
            q_mode_overrides={first_dso: "cosphi"},
            cosphi_overrides={first_dso: 0.85},
        )
        # Override hits the first DSO DER
        assert net.sgen.at[first_dso, "q_mode"] == "cosphi"
        assert net.sgen.at[first_dso, "cosphi"] == pytest.approx(0.85)
        # Sibling DSO DER stays on the level default
        second_dso = int(meta.dso_der_indices[1])
        assert net.sgen.at[second_dso, "q_mode"] == "qv"

    def test_qv_parameter_overrides(self, base_net_meta):
        net, meta = base_net_meta
        first_tso = int(meta.tso_der_indices[0])
        tag_der_q_modes(
            net, meta,
            qv_slope_pu_overrides={first_tso: 0.10},
            qv_vref_pu_overrides={first_tso: 1.02},
            qv_deadband_pu_overrides={first_tso: 0.03},
        )
        assert net.sgen.at[first_tso, "qv_slope_pu"] == pytest.approx(0.10)
        assert net.sgen.at[first_tso, "qv_vref_pu"] == pytest.approx(1.02)
        assert net.sgen.at[first_tso, "qv_deadband_pu"] == pytest.approx(0.03)
        # Other TSO DERs keep the level default
        second_tso = int(meta.tso_der_indices[1])
        assert net.sgen.at[second_tso, "qv_slope_pu"] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
#  Re-invocation safety
# ---------------------------------------------------------------------------

class TestReInvocation:
    def test_q_set_mvar_preserved_across_calls(self, base_net_meta):
        net, meta = base_net_meta
        tag_der_q_modes(net, meta)
        # Simulate a controller writing a q_set command.
        first_tso = int(meta.tso_der_indices[0])
        net.sgen.at[first_tso, "q_set_mvar"] = 5.5
        # Re-tag with different parameters.
        tag_der_q_modes(net, meta, tso_qv_slope_pu=0.04)
        # q_set_mvar must NOT be reset to 0 — it's runtime state.
        assert net.sgen.at[first_tso, "q_set_mvar"] == pytest.approx(5.5)
        # But qv_slope_pu picks up the new level default.
        assert net.sgen.at[first_tso, "qv_slope_pu"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_bad_level_mode(self, base_net_meta):
        net, meta = base_net_meta
        with pytest.raises(ValueError, match="Invalid tso_q_mode"):
            tag_der_q_modes(net, meta, tso_q_mode="bogus")
        with pytest.raises(ValueError, match="Invalid dso_q_mode"):
            tag_der_q_modes(net, meta, dso_q_mode="bogus")

    def test_bad_override_mode(self, base_net_meta):
        net, meta = base_net_meta
        first_tso = int(meta.tso_der_indices[0])
        with pytest.raises(ValueError, match="q_mode_overrides"):
            tag_der_q_modes(
                net, meta, q_mode_overrides={first_tso: "bogus"},
            )

    def test_bad_cosphi_sign(self, base_net_meta):
        net, meta = base_net_meta
        first_tso = int(meta.tso_der_indices[0])
        with pytest.raises(ValueError, match="cosphi_sign_overrides"):
            tag_der_q_modes(
                net, meta, cosphi_sign_overrides={first_tso: 0},
            )


# ---------------------------------------------------------------------------
#  Coexistence with apply_der_classification (transition guarantee)
# ---------------------------------------------------------------------------

class TestCoexistenceWithApply:
    """tag_der_q_modes is additive; it must not interfere with the legacy
    apply_der_classification path.  The two will coexist through commits
    2-6 and tag_der_q_modes will become canonical in commit 7."""

    def test_tag_then_apply(self, base_net_meta):
        net, meta = base_net_meta
        from network.ieee39.build import apply_der_classification
        tag_der_q_modes(net, meta)
        # The legacy promotion still runs cleanly (drops sgens, makes gens)
        meta2 = apply_der_classification(net, meta, overrides=None)
        # After promotion, surviving sgens still carry the q_mode columns
        if len(meta2.dso_der_indices) > 0:
            s = int(meta2.dso_der_indices[0])
            assert s in net.sgen.index
            assert net.sgen.at[s, "q_mode"] == "qv"
            assert net.sgen.at[s, "qv_slope_pu"] == pytest.approx(0.07)

    def test_apply_then_tag(self, base_net_meta):
        net, meta = base_net_meta
        from network.ieee39.build import apply_der_classification
        meta2 = apply_der_classification(net, meta, overrides=None)
        # tag_der_q_modes after promotion only sees surviving sgens
        tag_der_q_modes(net, meta2)
        # Surviving DSO DERs are tagged
        if len(meta2.dso_der_indices) > 0:
            s = int(meta2.dso_der_indices[0])
            assert net.sgen.at[s, "q_mode"] == "qv"
        # Promoted gens are NOT in net.sgen anymore — tag silently skips them
        # (that's the per-DER guard in _tag).  Test passes implicitly.
