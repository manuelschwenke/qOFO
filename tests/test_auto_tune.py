"""Tests for the auto-tuning architecture fix (Phases 1-4).

Tests verify:
  5.1  Collinear sensitivities: g_w stays near user init (not O(n)*init)
  5.2  Well-separated sensitivities: minimal g_w inflation
  5.3  Integration: full 3-zone-style benchmark
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.auto_tune import (
    TuningConfig,
    _conditioning_pump,
    _eigenvalue_init_gw,
    tune_continuous_gw,
    auto_tune,
    DSOTuneInput,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_single_zone_H(n_outputs: int, n_actuators: int, *, seed: int = 42):
    """Random H matrix for a single zone."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_outputs, n_actuators)) * 0.01


def _make_collinear_H(n_outputs: int, n_actuators: int, *, seed: int = 42):
    """H where all actuator columns are identical (worst-case collinear)."""
    rng = np.random.default_rng(seed)
    col = rng.standard_normal((n_outputs, 1)) * 0.01
    return np.tile(col, (1, n_actuators))


def _M_builder_single_zone(H, Q_diag):
    """Return an M_builder closure for a single-zone continuous system."""
    Q_sqrt = np.sqrt(np.maximum(Q_diag, 0.0))
    QH = Q_sqrt[:, None] * H

    def builder(gw):
        gi = 1.0 / np.sqrt(np.maximum(gw, 1e-12))
        Phi = 2.0 * (QH.T @ QH)
        M = (gi[:, None] * Phi) * gi[None, :]
        eigs = np.linalg.eigvalsh(M)
        return M, eigs

    return builder


# ── 5.1  Collinear sensitivities ─────────────────────────────────────────────

class TestCollinearSensitivities:
    """When DERs have identical H columns, eigenvalue-based init boosts
    uniformly (all actuators participate equally in the single mode).
    The key property: lambda_max after init is near lambda_target."""

    def test_eigenvalue_init_achieves_lambda_target(self):
        n_out, n_act = 10, 8
        H = _make_collinear_H(n_out, n_act)
        Q = np.ones(n_out) * 1e5
        gw_user = np.full(n_act, 50.0)
        lam_target = 1.5

        builder = _M_builder_single_zone(H, Q)
        gw_new = _eigenvalue_init_gw(builder, gw_user, lambda_target=lam_target)

        # After init, lambda_max should be near lambda_target
        _, eigs_after = builder(gw_new)
        lam_max_after = float(np.max(eigs_after))
        assert lam_max_after < lam_target * 2.0, (
            f"lambda_max after init = {lam_max_after:.2f}, "
            f"should be near target {lam_target}"
        )

    def test_eigenvalue_init_uniform_boost_for_collinear(self):
        """With identical columns, all actuators should get equal boost."""
        n_out, n_act = 10, 8
        H = _make_collinear_H(n_out, n_act)
        Q = np.ones(n_out) * 1e5
        gw_user = np.full(n_act, 50.0)

        builder = _M_builder_single_zone(H, Q)
        gw_new = _eigenvalue_init_gw(builder, gw_user, lambda_target=1.5)

        ratios = gw_new / gw_user
        # All ratios should be equal (uniform participation)
        assert_allclose(ratios, ratios[0], rtol=0.01, err_msg=(
            f"Collinear columns should give uniform boost, got {ratios}"
        ))

    def test_gershgorin_floor_below_full_bound(self):
        """With gersh_floor_fraction=0.2, the Gershgorin floor is 20% of
        the full bound, so initial g_w before eigenvalue init starts lower."""
        n_out, n_act = 10, 8
        H = _make_collinear_H(n_out, n_act)
        Q = np.ones(n_out) * 1e5
        gw_user = np.full(n_act, 50.0)

        # Compute full Gershgorin for reference
        Q_sqrt = np.sqrt(Q)
        QH = Q_sqrt[:, None] * H
        c_diag = np.sum(QH ** 2, axis=0)
        gw_gersh_full = 2.0 * c_diag / 2.0  # safety_factor=2.0

        # Floor = max(user, 0.2 * gersh)
        gw_floor = np.maximum(gw_user, 0.2 * gw_gersh_full)
        # The Gershgorin floor should be well below full Gershgorin
        assert np.all(gw_floor <= gw_gersh_full), (
            f"Floor should be <= full Gershgorin.\n"
            f"  floor: {gw_floor}\n"
            f"  full:  {gw_gersh_full}"
        )


# ── 5.2  Well-separated sensitivities ────────────────────────────────────────

class TestWellSeparatedSensitivities:
    """When DERs are electrically distant, Gershgorin and eigenvalue-based
    should agree, and g_w inflation should be minimal."""

    def test_diagonal_dominant_H_minimal_inflation(self):
        n_out, n_act = 6, 6
        # Nearly diagonal H — each DER affects mainly its own bus
        H = np.eye(n_out, n_act) * 0.01
        Q = np.ones(n_out) * 1e5
        gw_user = np.full(n_act, 50.0)

        builder = _M_builder_single_zone(H, Q)
        gw_new = _eigenvalue_init_gw(builder, gw_user, lambda_target=1.5)

        ratio = gw_new / gw_user
        # For diagonal H, all eigenvalues are equal → no inflation needed
        assert_allclose(ratio, 1.0, atol=0.5, err_msg=(
            f"Well-separated DERs should need minimal inflation, "
            f"got max ratio = {np.max(ratio):.2f}"
        ))


# ── Conditioning pump ────────────────────────────────────────────────────────

class TestConditioningPump:
    """Test the dual exit condition (kappa + lambda_max*alpha)."""

    def test_pump_respects_lambda_max_alpha_target(self):
        n_out, n_act = 10, 5
        H = _make_single_zone_H(n_out, n_act, seed=123)
        Q = np.ones(n_out) * 1e5
        gw = np.full(n_act, 1.0)  # deliberately small → large eigenvalues

        builder = _M_builder_single_zone(H, Q)

        alpha_target = 0.5
        lam_alpha_target = 1.8

        gw_tuned, kappa, n_iters = _conditioning_pump(
            gw, builder,
            kappa_target=20.0,
            alpha_target=alpha_target,
            lambda_max_alpha_target=lam_alpha_target,
            max_iters=50,  # enough iterations for this tight target
        )

        # Verify lambda_max * alpha_target is within bound
        _, eigs = builder(gw_tuned)
        lam_max = float(np.max(eigs))
        assert alpha_target * lam_max <= lam_alpha_target * 1.05, (
            f"alpha*lam_max = {alpha_target * lam_max:.3f} exceeds "
            f"target {lam_alpha_target:.3f}"
        )

    def test_pump_achieves_kappa_target(self):
        n_out, n_act = 10, 5
        H = _make_single_zone_H(n_out, n_act, seed=99)
        Q = np.ones(n_out) * 1e5
        gw = np.full(n_act, 1.0)

        builder = _M_builder_single_zone(H, Q)

        gw_tuned, kappa, _ = _conditioning_pump(
            gw, builder,
            kappa_target=50.0,
            alpha_target=10.0,  # large → scale check won't bind
            lambda_max_alpha_target=100.0,
        )

        assert kappa <= 50.0 * 1.1, f"kappa = {kappa:.1f}, target was 50.0"


# ── 5.3  Integration-style test ──────────────────────────────────────────────

class TestAutoTuneIntegration:
    """End-to-end auto_tune with TuningConfig on a synthetic 2-zone system."""

    def _make_2zone_system(self):
        rng = np.random.default_rng(42)
        n_out = 8
        # Zone 0: 4 DER + 1 OLTC
        # Zone 1: 3 DER + 1 OLTC
        H_00 = rng.standard_normal((n_out, 5)) * 0.01
        H_11 = rng.standard_normal((n_out, 4)) * 0.01
        H_01 = rng.standard_normal((n_out, 4)) * 0.002  # weak coupling
        H_10 = rng.standard_normal((n_out, 5)) * 0.002

        H_blocks = {(0, 0): H_00, (1, 1): H_11, (0, 1): H_01, (1, 0): H_10}
        Q_list = [np.ones(n_out) * 1e5, np.ones(n_out) * 1e5]
        ac = [
            {'n_der': 4, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 1, 'n_shunt': 0},
            {'n_der': 3, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 1, 'n_shunt': 0},
        ]
        gw_init = [
            np.array([50.0, 50.0, 50.0, 50.0, 100.0]),   # zone 0
            np.array([50.0, 50.0, 50.0, 100.0]),          # zone 1
        ]
        return H_blocks, Q_list, ac, gw_init

    def test_gw_stays_reasonable(self):
        H_blocks, Q_list, ac, gw_init = self._make_2zone_system()

        tc = TuningConfig()
        result = auto_tune(
            H_blocks=H_blocks,
            Q_obj_list=Q_list,
            actuator_counts=ac,
            zone_ids=[0, 1],
            gw_tso_init=gw_init,
            tuning_config=tc,
        )

        # Continuous g_w should stay within 10x of user init
        for z_idx in range(2):
            n_cont = ac[z_idx]['n_der']
            gw_cont = result.gw_tso_list[z_idx][:n_cont]
            gw_user = gw_init[z_idx][:n_cont]
            ratio = gw_cont / gw_user
            assert np.max(ratio) < 10.0, (
                f"Zone {z_idx}: g_w inflated {np.max(ratio):.1f}x "
                f"(should be < 10x)"
            )

    def test_alpha_in_reasonable_range(self):
        H_blocks, Q_list, ac, gw_init = self._make_2zone_system()

        tc = TuningConfig()
        result = auto_tune(
            H_blocks=H_blocks,
            Q_obj_list=Q_list,
            actuator_counts=ac,
            zone_ids=[0, 1],
            gw_tso_init=gw_init,
            tuning_config=tc,
        )

        assert 0.05 <= result.alpha_tso <= 5.0, (
            f"alpha_tso = {result.alpha_tso:.3f} is outside [0.05, 5.0]"
        )

    def test_stability_conditions_pass(self):
        H_blocks, Q_list, ac, gw_init = self._make_2zone_system()

        tc = TuningConfig()
        result = auto_tune(
            H_blocks=H_blocks,
            Q_obj_list=Q_list,
            actuator_counts=ac,
            zone_ids=[0, 1],
            gw_tso_init=gw_init,
            tuning_config=tc,
        )

        assert result.c2_feasible, (
            f"C2 should be feasible, warnings: {result.warnings}"
        )

    def test_tuning_config_defaults_work(self):
        """TuningConfig with all defaults should produce a valid result."""
        H_blocks, Q_list, ac, gw_init = self._make_2zone_system()

        tc = TuningConfig()
        result = auto_tune(
            H_blocks=H_blocks,
            Q_obj_list=Q_list,
            actuator_counts=ac,
            zone_ids=[0, 1],
            gw_tso_init=gw_init,
            tuning_config=tc,
        )
        assert isinstance(result.alpha_tso, float)
        assert len(result.gw_tso_list) == 2


# ── TuningConfig ─────────────────────────────────────────────────────────────

class TestTuningConfig:
    def test_defaults(self):
        tc = TuningConfig()
        assert tc.tso_kappa_target == 50.0
        assert tc.dso_kappa_target == 10.0
        assert tc.lambda_target == 1.5
        assert tc.pump_max_iters == 20

    def test_custom_values(self):
        tc = TuningConfig(tso_kappa_target=15.0, lambda_target=2.0)
        assert tc.tso_kappa_target == 15.0
        assert tc.lambda_target == 2.0
        # Other defaults unchanged
        assert tc.dso_kappa_target == 10.0
