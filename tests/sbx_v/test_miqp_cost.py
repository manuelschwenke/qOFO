"""
SBX-V Phase 1 tests — MIQP tier cost layer (plan §5, V-D1, V-D9).

Covers:

* side-spec construction for both V-D9 incentive models;
* problem augmentation shapes with and without integer variables;
* the NEUTRAL bypass (regression R1 at the solver seam: byte-identical
  results when no spec is active);
* the plan §5 reconstruction invariant;
* the Phase-1 acceptance behaviour test: a synthetic two-step problem
  where the dispatch prefers the free band under nonzero prices and
  crosses it only when the pull exceeds the tier price;
* ordered filling of the granted tier before the ungesichert tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimisation.miqp_solver import MIQPSolver, build_miqp_problem
from sbx_h.fail import SBXError
from sbx_v.band import NormalBand
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction
from sbx_v.miqp_cost import (AreaTierSpec, OPEN_TAIL_MVAR, PricingSolver,
                            SideSpec, TierSegment, area_tier_spec,
                            augment_problem, build_side_spec, strip_result)


# ----------------------------------------------------------------------
#  Synthetic problem helpers
# ----------------------------------------------------------------------

def _one_pcc_problem(*, y0: float, pull: float, g_w: float = 0.005,
                     alpha: float = 1.0):
    """One continuous control, one Q_PCC output row with dq/du = 1.

    Baseline optimum: min g_w·w² − pull·w  →  w* = pull / (2 g_w).
    """
    return build_miqp_problem(
        alpha=alpha,
        u_current=np.array([0.0]),
        y_current=np.array([y0]),
        H=np.array([[1.0]]),
        grad_f=np.array([-pull]),
        u_lower=np.array([-1000.0]),
        u_upper=np.array([+1000.0]),
        y_lower=np.array([-1e6]),
        y_upper=np.array([+1e6]),
        g_w=g_w,
        g_u=0.0,
        g_z=1e-2,   # soft, mirrors the repo's g_z_q_pcc
        integer_indices=[],
    )


def _spec(rows, *, anchor_low: float, segments_low, anchor_rai: float = 50.0,
          segments_rai=((OPEN_TAIL_MVAR, 100.0),)) -> AreaTierSpec:
    """Manual spec: LOWERING side under test, RAISING side priced high."""
    return AreaTierSpec(
        area_id="area_1",
        pcc_output_rows=tuple(rows),
        raising=SideSpec(
            direction=Direction.RAISING,
            anchor_mvar=anchor_rai,
            segments=tuple(TierSegment(w, s) for w, s in segments_rai),
        ),
        lowering=SideSpec(
            direction=Direction.LOWERING,
            anchor_mvar=anchor_low,
            segments=tuple(TierSegment(w, s) for w, s in segments_low),
        ),
    )


# ----------------------------------------------------------------------
#  Side-spec construction (V-D9)
# ----------------------------------------------------------------------

class TestBuildSideSpec:
    def test_no_grant_both_models_identical(self):
        band = NormalBand("a", 50.0, 50.0)
        for model in ("nearest_edge", "leitfaden_exact_when_granted"):
            cfg = SBXVConfig(miqp_cost_model=model)
            side = build_side_spec(direction=Direction.LOWERING, band=band,
                                   grant_mvar=0.0, config=cfg)
            assert side.anchor_mvar == 50.0
            assert len(side.segments) == 1
            assert side.segments[0].slope_obj_per_mvar == pytest.approx(
                cfg.c_ug_obj_per_mvar_step)
            assert side.segments[0].width_mvar >= OPEN_TAIL_MVAR

    def test_nearest_edge_with_grant(self):
        band = NormalBand("a", 50.0, 50.0)
        cfg = SBXVConfig(miqp_cost_model="nearest_edge")
        side = build_side_spec(direction=Direction.LOWERING, band=band,
                               grant_mvar=30.0, config=cfg)
        assert side.anchor_mvar == 50.0
        assert side.segments[0].width_mvar == 30.0
        assert side.segments[0].slope_obj_per_mvar == pytest.approx(
            cfg.c_vh_obj_per_mvar_step)
        assert side.segments[1].slope_obj_per_mvar == pytest.approx(
            cfg.c_ug_obj_per_mvar_step)

    def test_leitfaden_opposite_edge_anchor(self):
        # Worked-example geometry [LF §8.2]: band ±50, grant 100 beyond
        # the upper edge → Durchschnittspreis segment spans 200 Mvar from
        # the opposite edge (the Vorhalteleistung of the worked example).
        band = NormalBand("a", 50.0, 50.0)
        cfg = SBXVConfig()  # default model = leitfaden_exact_when_granted
        side = build_side_spec(direction=Direction.LOWERING, band=band,
                               grant_mvar=100.0, config=cfg)
        assert side.anchor_mvar == -50.0
        assert side.segments[0].width_mvar == pytest.approx(200.0)
        assert side.segments[0].slope_obj_per_mvar == pytest.approx(
            cfg.c_vh_obj_per_mvar_step)

    def test_double_grant_blocked_under_leitfaden(self):
        band = NormalBand("a", 50.0, 50.0)
        cfg = SBXVConfig()
        with pytest.raises(SBXError):
            area_tier_spec(area_id="a", pcc_output_rows=(0,), band=band,
                           grant_raise_mvar=30.0, grant_lower_mvar=30.0,
                           config=cfg)

    def test_convexity_asserted(self):
        with pytest.raises(SBXError):
            SideSpec(direction=Direction.LOWERING, anchor_mvar=50.0,
                     segments=(TierSegment(10.0, 2.0),
                               TierSegment(OPEN_TAIL_MVAR, 1.0)))

    def test_open_tail_required(self):
        with pytest.raises(SBXError):
            SideSpec(direction=Direction.LOWERING, anchor_mvar=50.0,
                     segments=(TierSegment(10.0, 1.0),))


# ----------------------------------------------------------------------
#  Neutral bypass (R1 at the solver seam)
# ----------------------------------------------------------------------

class TestNeutralBypass:
    def test_bypass_is_byte_identical(self):
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        inner = MIQPSolver()
        baseline = inner.solve(problem)
        proxy = PricingSolver(MIQPSolver(), lambda: None, g_z_tier=1e4)
        proxied = proxy.solve(problem)
        assert baseline.status == proxied.status
        assert np.array_equal(baseline.w_continuous, proxied.w_continuous)
        assert np.array_equal(baseline.w_integer, proxied.w_integer)
        assert np.array_equal(baseline.z, proxied.z)
        assert baseline.objective_value == proxied.objective_value
        assert proxy.last_decompositions is None

    def test_zero_price_augmentation_matches_baseline(self):
        # Not byte-identical (different problem), but the argmin must
        # coincide when every slope is zero and anchors are wide.
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        baseline = MIQPSolver().solve(problem)
        spec = _spec([0], anchor_low=1e6,
                     segments_low=((OPEN_TAIL_MVAR, 0.0),),
                     anchor_rai=1e6,
                     segments_rai=((OPEN_TAIL_MVAR, 0.0),))
        proxy = PricingSolver(MIQPSolver(), lambda: [spec], g_z_tier=1e4)
        priced = proxy.solve(problem)
        assert priced.is_feasible
        np.testing.assert_allclose(
            priced.w_continuous, baseline.w_continuous, atol=1e-6)


# ----------------------------------------------------------------------
#  Phase-1 acceptance: band preference under nonzero prices
# ----------------------------------------------------------------------

class TestBandPreference:
    def test_price_above_pull_pins_dispatch_at_the_edge(self):
        # pull = 1.0, tier slope 2.0 > pull → the optimum sits exactly at
        # the band edge (free tier preferred, plan Phase 1 acceptance).
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        spec = _spec([0], anchor_low=50.0,
                     segments_low=((OPEN_TAIL_MVAR, 2.0),))
        proxy = PricingSolver(MIQPSolver(), lambda: [spec], g_z_tier=1e4)
        res = proxy.solve(problem)
        assert res.is_feasible
        q_new = 40.0 + res.w_continuous[0]
        assert q_new == pytest.approx(50.0, abs=1e-3)
        (dec_low,) = [d for d in proxy.last_decompositions
                      if d.direction is Direction.LOWERING]
        assert dec_low.excess_mvar == pytest.approx(0.0, abs=1e-3)
        assert dec_low.cost_obj == pytest.approx(0.0, abs=1e-2)

    def test_price_below_pull_crosses_at_the_priced_optimum(self):
        # pull = 1.0, slope 0.5 → beyond the edge the stationary point is
        # 2·g_w·w + slope = pull → w* = 50, q = 90, excess 40 at cost
        # 0.5 · 40 = 20 objective units.
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        spec = _spec([0], anchor_low=50.0,
                     segments_low=((OPEN_TAIL_MVAR, 0.5),))
        proxy = PricingSolver(MIQPSolver(), lambda: [spec], g_z_tier=1e4)
        res = proxy.solve(problem)
        assert res.is_feasible
        q_new = 40.0 + res.w_continuous[0]
        assert q_new == pytest.approx(90.0, abs=1e-2)
        (dec_low,) = [d for d in proxy.last_decompositions
                      if d.direction is Direction.LOWERING]
        assert dec_low.excess_mvar == pytest.approx(40.0, abs=1e-2)
        assert dec_low.cost_obj == pytest.approx(20.0, rel=1e-2)

    def test_granted_tier_fills_before_ungesichert(self):
        # Tier 2 (20 Mvar at 0.3) fills completely before tier 3 (0.5):
        # stationary point 0.01·(10+e) + 0.5 = 1 → e = 40 → x = (20, 20).
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        spec = _spec([0], anchor_low=50.0,
                     segments_low=((20.0, 0.3), (OPEN_TAIL_MVAR, 0.5)))
        proxy = PricingSolver(MIQPSolver(), lambda: [spec], g_z_tier=1e4)
        res = proxy.solve(problem)
        assert res.is_feasible
        (dec_low,) = [d for d in proxy.last_decompositions
                      if d.direction is Direction.LOWERING]
        assert dec_low.excess_mvar == pytest.approx(40.0, abs=1e-2)
        assert dec_low.x_mvar[0] == pytest.approx(20.0, abs=1e-2)
        assert dec_low.x_mvar[1] == pytest.approx(20.0, abs=1e-2)
        assert dec_low.cost_obj == pytest.approx(
            20.0 * 0.3 + 20.0 * 0.5, rel=1e-2)

    def test_raising_side(self):
        # Mirror case: pull the boundary Q negative; RAISING side priced
        # above the pull → pinned at the signed edge −50.
        problem = _one_pcc_problem(y0=-40.0, pull=-1.0)
        spec = AreaTierSpec(
            area_id="area_1",
            pcc_output_rows=(0,),
            raising=SideSpec(direction=Direction.RAISING, anchor_mvar=50.0,
                             segments=(TierSegment(OPEN_TAIL_MVAR, 2.0),)),
            lowering=SideSpec(direction=Direction.LOWERING, anchor_mvar=50.0,
                              segments=(TierSegment(OPEN_TAIL_MVAR, 2.0),)),
        )
        proxy = PricingSolver(MIQPSolver(), lambda: [spec], g_z_tier=1e4)
        res = proxy.solve(problem)
        assert res.is_feasible
        q_new = -40.0 + res.w_continuous[0]
        assert q_new == pytest.approx(-50.0, abs=1e-3)
        (dec_rai,) = [d for d in proxy.last_decompositions
                      if d.direction is Direction.RAISING]
        assert dec_rai.q_netted_signed_mvar == pytest.approx(-50.0, abs=1e-3)
        assert dec_rai.excess_mvar == pytest.approx(0.0, abs=1e-3)


# ----------------------------------------------------------------------
#  Augmentation mechanics (shapes, netting, integers, reconstruction)
# ----------------------------------------------------------------------

class TestAugmentation:
    def test_shapes_and_layout(self):
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        spec = _spec([0], anchor_low=50.0,
                     segments_low=((20.0, 0.3), (OPEN_TAIL_MVAR, 0.5)))
        aug, amap = augment_problem(problem, [spec], g_z_tier=1e4)
        # 1 original control + (1 raising + 2 lowering) aux variables.
        assert aug.n_continuous == 1 + 3
        assert aug.n_outputs == 1 + 2
        assert amap.n_aux == 3
        # Aux columns are zero in the original rows.
        assert np.all(aug.H_tilde[0, 1:4] == 0.0)
        # Tier rows carry −1 at their own aux columns only.
        rai, low = amap.sides
        assert aug.H_tilde[rai.row_index, rai.aux_start] == -1.0
        assert aug.H_tilde[low.row_index, low.aux_start] == -1.0
        assert aug.H_tilde[low.row_index, low.aux_start + 1] == -1.0
        # Anchors land in y_upper; wide finite lower bounds.
        assert aug.y_upper[low.row_index] == 50.0
        assert np.all(np.isfinite(aug.y_lower))

    def test_multi_pcc_netting(self):
        # DP5: three PCC rows netted into one area quantity.  Push each
        # row via its own control; the tier row must see the SUM.
        n = 3
        problem = build_miqp_problem(
            alpha=1.0,
            u_current=np.zeros(n),
            y_current=np.array([10.0, 20.0, 10.0]),   # netted 40
            H=np.eye(n),
            grad_f=np.array([-1.0, 0.0, 0.0]),
            u_lower=np.full(n, -1000.0),
            u_upper=np.full(n, +1000.0),
            y_lower=np.full(n, -1e6),
            y_upper=np.full(n, +1e6),
            g_w=0.005,
            g_u=0.0,
            g_z=1e-2,
            integer_indices=[],
        )
        spec = _spec([0, 1, 2], anchor_low=50.0,
                     segments_low=((OPEN_TAIL_MVAR, 2.0),))
        proxy = PricingSolver(MIQPSolver(), lambda: [spec], g_z_tier=1e4)
        res = proxy.solve(problem)
        assert res.is_feasible
        (dec_low,) = [d for d in proxy.last_decompositions
                      if d.direction is Direction.LOWERING]
        # Netted total pinned at the area edge, not any per-NVP value.
        assert dec_low.q_netted_signed_mvar == pytest.approx(50.0, abs=1e-3)

    def test_integer_block_shift(self):
        # One continuous + one integer control (dq/dstep = 5 Mvar); the
        # integer block must shift cleanly and strip back unchanged.
        problem = build_miqp_problem(
            alpha=1.0,
            u_current=np.array([0.0, 0.0]),
            y_current=np.array([40.0]),
            H=np.array([[1.0, 5.0]]),
            grad_f=np.array([-1.0, 0.0]),
            u_lower=np.array([-1000.0, -2.0]),
            u_upper=np.array([+1000.0, +2.0]),
            y_lower=np.array([-1e6]),
            y_upper=np.array([+1e6]),
            g_w=np.array([0.005, 0.5]),
            g_u=0.0,
            g_z=1e-2,
            integer_indices=[1],
        )
        spec = _spec([0], anchor_low=50.0,
                     segments_low=((OPEN_TAIL_MVAR, 2.0),))
        aug, amap = augment_problem(problem, [spec], g_z_tier=1e4)
        assert aug.integer_indices == [aug.n_continuous]
        res = MIQPSolver().solve(aug)
        assert res.is_feasible
        stripped, decomp = strip_result(res, amap)
        assert len(stripped.w_continuous) == 1
        assert len(stripped.w_integer) == 1
        assert len(stripped.z) == 1
        # Reconstruction ran inside strip_result without raising; the
        # predicted netted Q must respect the priced edge.
        (dec_low,) = [d for d in decomp
                      if d.direction is Direction.LOWERING]
        assert dec_low.q_netted_signed_mvar <= 50.0 + 1e-2

    def test_row_claimed_twice_rejected(self):
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        s1 = _spec([0], anchor_low=50.0,
                   segments_low=((OPEN_TAIL_MVAR, 0.5),))
        s2 = AreaTierSpec(
            area_id="area_2", pcc_output_rows=(0,),
            raising=s1.raising, lowering=s1.lowering,
        )
        with pytest.raises(SBXError):
            augment_problem(problem, [s1, s2], g_z_tier=1e4)

    def test_empty_specs_rejected(self):
        problem = _one_pcc_problem(y0=40.0, pull=1.0)
        with pytest.raises(SBXError):
            augment_problem(problem, [], g_z_tier=1e4)
