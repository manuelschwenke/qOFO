"""
Tests for tune_ofo_params.py
============================

Validates per-actuator g_w computation (Phase 1), multi-zone joint
tuning (Phase 2), and DSO cascade tuning against known analytical
results on synthetic sensitivity matrices.
"""

import numpy as np
import pytest

from core.cascade_config import CascadeConfig
from analysis.tune_ofo_params import (
    ZoneTuningResult,
    TuningResult,
    compute_optimal_gw,
    tune_multi_zone,
    tune_dso,
    _compute_curvature_diagonal,
    _compute_curvature_matrix,
    _apply_gw_floors,
    _build_actuator_labels,
    _effective_eigenspectrum,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: build synthetic H blocks
# ═══════════════════════════════════════════════════════════════════════════════

def _identity_system(n: int, q_weight: float = 1.0):
    """Identity H with uniform q_obj -- known analytics."""
    H = np.eye(n)
    q_obj = np.full(n, q_weight)
    return H, q_obj


def _diagonal_system(h_diag: np.ndarray, q_weight: float = 1.0):
    """Diagonal H with specified entries."""
    H = np.diag(h_diag)
    q_obj = np.full(len(h_diag), q_weight)
    return H, q_obj


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests: _compute_curvature_diagonal
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurvatureDiagonal:
    def test_identity(self):
        """C = I^T I = I  →  diag(C) = [1, 1, ..., 1]."""
        H, q = _identity_system(5)
        c_diag = _compute_curvature_diagonal(H, q)
        np.testing.assert_allclose(c_diag, np.ones(5))

    def test_diagonal(self):
        """diag(H^T Q H) = q * h_ii^2 for diagonal H."""
        h = np.array([2.0, 3.0, 5.0])
        H, q = _diagonal_system(h, q_weight=4.0)
        c_diag = _compute_curvature_diagonal(H, q)
        expected = 4.0 * h ** 2
        np.testing.assert_allclose(c_diag, expected)

    def test_rectangular(self):
        """Fat matrix: more outputs than controls."""
        rng = np.random.default_rng(42)
        H = rng.standard_normal((10, 3))
        q = np.ones(10)
        c_diag = _compute_curvature_diagonal(H, q)
        C_full = H.T @ H
        np.testing.assert_allclose(c_diag, np.diag(C_full), atol=1e-12)

    def test_matches_full_matrix(self):
        """Diagonal of efficient method matches full C = H^T Q H."""
        rng = np.random.default_rng(99)
        H = rng.standard_normal((8, 5))
        q = rng.uniform(0.1, 10.0, size=8)
        c_diag = _compute_curvature_diagonal(H, q)
        C = _compute_curvature_matrix(H, q)
        np.testing.assert_allclose(c_diag, np.diag(C), atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests: _apply_gw_floors
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyGwFloors:
    def test_generator_override(self):
        """Generator weights are hard-overridden, not floored."""
        g_w = np.array([0.5, 0.3, 100.0, 20.0])  # DER, PCC, Gen, OLTC
        counts = {'n_der': 1, 'n_pcc': 1, 'n_gen': 1, 'n_oltc': 1}
        result = _apply_gw_floors(g_w, counts, min_gw=0.01,
                                  min_gw_discrete=40.0, gen_gw_override=1e7)
        assert result[2] == 1e7  # gen override
        assert result[3] == 40.0  # OLTC floor applied

    def test_continuous_floors(self):
        """DER and PCC below min_gw are floored up."""
        g_w = np.array([0.001, 0.005, 1e7, 50.0])
        counts = {'n_der': 1, 'n_pcc': 1, 'n_gen': 1, 'n_oltc': 1}
        result = _apply_gw_floors(g_w, counts, min_gw=0.01,
                                  min_gw_discrete=40.0, gen_gw_override=1e7)
        assert result[0] == 0.01
        assert result[1] == 0.01

    def test_does_not_mutate_input(self):
        """_apply_gw_floors returns a copy."""
        g_w = np.array([0.5, 0.3, 100.0, 20.0])
        original = g_w.copy()
        counts = {'n_der': 1, 'n_pcc': 1, 'n_gen': 1, 'n_oltc': 1}
        _apply_gw_floors(g_w, counts, 0.01, 40.0, 1e7)
        np.testing.assert_array_equal(g_w, original)

    def test_multiple_actuators_per_type(self):
        """Works with multiple DERs, PCCs, etc."""
        g_w = np.array([0.5, 0.8, 0.3, 0.4, 1e7, 1e7, 20.0])
        counts = {'n_der': 2, 'n_pcc': 2, 'n_gen': 2, 'n_oltc': 1}
        result = _apply_gw_floors(g_w, counts, min_gw=0.01,
                                  min_gw_discrete=40.0, gen_gw_override=5e6)
        assert result[4] == 5e6
        assert result[5] == 5e6
        assert result[6] == 40.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests: _build_actuator_labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildActuatorLabels:
    def test_basic(self):
        counts = {'n_der': 2, 'n_pcc': 1, 'n_gen': 1, 'n_oltc': 1}
        labels = _build_actuator_labels(counts)
        assert labels == ['Q_DER_0', 'Q_DER_1', 'Q_PCC_0',
                          'V_gen_0', 'OLTC_0']

    def test_empty_type(self):
        counts = {'n_der': 1, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0}
        labels = _build_actuator_labels(counts)
        assert labels == ['Q_DER_0']


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests: compute_optimal_gw  (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeOptimalGw:
    def test_identity_single_zone(self):
        """H = I, Q = I  →  C_diag = [1,...,1]
        g_w = safety * alpha * 1 / 2 = 2 * 1 * 1 / 2 = 1.0"""
        n = 4
        H = np.eye(n)
        q = np.ones(n)
        config = CascadeConfig(alpha=1.0, gw_tso_v_gen=1e7)
        # 2 DER, 1 PCC, 1 gen, 0 OLTC
        counts = [{'n_der': 2, 'n_pcc': 1, 'n_gen': 1, 'n_oltc': 0}]
        result = compute_optimal_gw(
            config, {(0, 0): H}, [q], counts,
            safety_factor=2.0, min_gw=0.01, min_gw_discrete=40.0,
        )
        assert len(result) == 1
        g_w = result[0]
        assert len(g_w) == 4
        # DER and PCC get safety * alpha * 1 / 2 = 1.0
        np.testing.assert_allclose(g_w[0], 1.0)  # DER_0
        np.testing.assert_allclose(g_w[1], 1.0)  # DER_1
        np.testing.assert_allclose(g_w[2], 1.0)  # PCC_0
        # Gen is overridden
        assert g_w[3] == 1e7

    def test_scaled_curvature(self):
        """Diagonal H with different sensitivities → proportional g_w."""
        H = np.diag([1.0, 10.0, 0.1])  # 3 actuators
        q = np.ones(3)
        config = CascadeConfig(alpha=0.5, gw_tso_v_gen=1e7)
        counts = [{'n_der': 3, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0}]
        result = compute_optimal_gw(
            config, {(0, 0): H}, [q], counts,
            safety_factor=2.0, min_gw=0.01,
        )
        g_w = result[0]
        # g_w[k] = 2 * 0.5 * h_k^2 / 2 = 0.5 * h_k^2
        np.testing.assert_allclose(g_w[0], 0.5 * 1.0)
        np.testing.assert_allclose(g_w[1], 0.5 * 100.0)
        np.testing.assert_allclose(g_w[2], max(0.5 * 0.01, 0.01))

    def test_two_zones(self):
        """Two zones produce two g_w vectors."""
        H1 = np.eye(2)
        H2 = 2.0 * np.eye(3)
        q1 = np.ones(2)
        q2 = np.ones(3)
        config = CascadeConfig(alpha=1.0, gw_tso_v_gen=1e7)
        counts = [
            {'n_der': 2, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0},
            {'n_der': 3, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0},
        ]
        result = compute_optimal_gw(
            config, {(1, 1): H1, (2, 2): H2}, [q1, q2], counts,
        )
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 3
        # Zone 2 has 4x curvature → 4x g_w
        np.testing.assert_allclose(result[1][0] / result[0][0], 4.0)

    def test_discrete_floor(self):
        """OLTC actuators respect min_gw_discrete."""
        H = np.diag([0.01])  # tiny sensitivity
        q = np.ones(1)
        config = CascadeConfig(alpha=1.0, gw_tso_v_gen=1e7)
        counts = [{'n_der': 0, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 1}]
        result = compute_optimal_gw(
            config, {(0, 0): H}, [q], counts,
            min_gw_discrete=40.0,
        )
        assert result[0][0] == 40.0  # floor applied


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests: tune_multi_zone  (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTuneMultiZone:
    def test_uncoupled_converges_immediately(self):
        """Two uncoupled zones (no off-diagonal H) → gamma < 1 at iter 1."""
        H1 = np.diag([1.0, 2.0])
        H2 = np.diag([3.0, 1.5])
        q1 = np.ones(2)
        q2 = np.ones(2)

        result = tune_multi_zone(
            H_blocks={(1, 1): H1, (2, 2): H2},
            Q_obj_list=[q1, q2],
            actuator_counts=[
                {'n_der': 2, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0},
                {'n_der': 2, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0},
            ],
            gen_gw=1e7,
            gamma_target=0.8,
            verbose=False,
        )
        assert result.converged
        assert result.small_gain_gamma < 0.8
        assert len(result.zones) == 2

    def test_coupled_system_converges(self):
        """Two coupled zones with diagonally dominant H → converges.

        Uses structured (diagonally dominant) H matrices that are
        representative of real power system sensitivities, where
        dV_i/dQ_i >> dV_i/dQ_j for buses i and j far apart.
        """
        rng = np.random.default_rng(123)
        n1, n2 = 4, 3
        n_y1, n_y2 = 4, 3

        # Diagonally dominant local blocks (realistic V-Q sensitivity)
        H11 = np.diag([0.8, 0.6, 0.5, 0.7]) + rng.standard_normal((n_y1, n1)) * 0.05
        H22 = np.diag([0.9, 0.7, 0.5]) + rng.standard_normal((n_y2, n2)) * 0.05
        # Weak cross-zone coupling
        H12 = rng.standard_normal((n_y1, n2)) * 0.03
        H21 = rng.standard_normal((n_y2, n1)) * 0.03

        result = tune_multi_zone(
            H_blocks={
                (0, 0): H11, (0, 1): H12,
                (1, 0): H21, (1, 1): H22,
            },
            Q_obj_list=[np.ones(n_y1), np.ones(n_y2)],
            actuator_counts=[
                {'n_der': n1, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0},
                {'n_der': n2, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0},
            ],
            gen_gw=1e7,
            gamma_target=0.8,
            verbose=False,
        )
        assert result.converged
        assert result.small_gain_gamma < 0.8

    def test_gw_vectors_method(self):
        """TuningResult.gw_vectors() returns list of arrays."""
        H = np.eye(3)
        q = np.ones(3)
        result = tune_multi_zone(
            H_blocks={(0, 0): H},
            Q_obj_list=[q],
            actuator_counts=[{'n_der': 3, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0}],
            gen_gw=1e7,
            verbose=False,
        )
        vecs = result.gw_vectors()
        assert len(vecs) == 1
        assert len(vecs[0]) == 3

    def test_alpha_dict_method(self):
        """TuningResult.alpha_dict() returns {zone_id: alpha}."""
        H = np.eye(3)
        q = np.ones(3)
        result = tune_multi_zone(
            H_blocks={(5, 5): H},
            Q_obj_list=[q],
            actuator_counts=[{'n_der': 3, 'n_pcc': 0, 'n_gen': 0, 'n_oltc': 0}],
            zone_ids=[5],
            gen_gw=1e7,
            verbose=False,
        )
        d = result.alpha_dict()
        assert 5 in d
        assert d[5] > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests: tune_dso
# ═══════════════════════════════════════════════════════════════════════════════

class TestTuneDso:
    def test_identity_dso(self):
        """Identity H_dso with unit q_obj → known cascade margin."""
        n = 3
        H = np.eye(n)
        q = np.ones(n)
        g_w, rho, margin = tune_dso(
            H, q, n_der=3, n_oltc=0, n_shunt=0,
            alpha=1.0, safety_factor=2.0,
            tso_period_s=180.0, dso_period_s=60.0,
        )
        assert len(g_w) == 3
        assert 0.0 <= rho < 1.0
        assert margin > 0.0

    def test_cascade_margin_met(self):
        """DSO tuner achieves cascade_margin_target."""
        rng = np.random.default_rng(77)
        H = rng.standard_normal((6, 4))
        q = np.ones(6)
        g_w, rho, margin = tune_dso(
            H, q, n_der=2, n_oltc=1, n_shunt=1,
            alpha=0.5, safety_factor=2.0,
            cascade_margin_target=0.3,
        )
        assert margin >= 0.3 or rho == 0.0

    def test_discrete_floors(self):
        """OLTC and shunt actuators respect min_gw_discrete."""
        H = np.diag([0.01, 0.01, 0.01])  # tiny sensitivities
        q = np.ones(3)
        g_w, _, _ = tune_dso(
            H, q, n_der=1, n_oltc=1, n_shunt=1,
            min_gw_discrete=40.0,
        )
        assert g_w[1] >= 40.0  # OLTC
        assert g_w[2] >= 40.0  # shunt
