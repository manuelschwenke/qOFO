"""
Tests for DERMapping and per-DER H matrix expansion.
"""

import numpy as np
import pytest

from core.der_mapping import DERMapping


class TestDERMapping:
    """Tests for the DERMapping dataclass."""

    def _make_simple_mapping(self) -> DERMapping:
        """Two DERs at bus 10, one DER at bus 20."""
        return DERMapping(
            sgen_indices=(0, 1, 2),
            bus_indices=(10, 10, 20),
            s_rated_mva=np.array([50.0, 30.0, 100.0]),
            p_max_mw=np.array([40.0, 25.0, 80.0]),
            weights=np.ones(3),
        )

    def test_n_der(self):
        m = self._make_simple_mapping()
        assert m.n_der == 3

    def test_unique_bus_indices(self):
        m = self._make_simple_mapping()
        assert m.unique_bus_indices == [10, 20]

    def test_n_unique_bus(self):
        m = self._make_simple_mapping()
        assert m.n_unique_bus == 2

    def test_E_matrix_shape(self):
        m = self._make_simple_mapping()
        E = m.E
        assert E.shape == (2, 3)

    def test_E_matrix_values(self):
        m = self._make_simple_mapping()
        E = m.E
        # Bus 10 maps to DER 0 and DER 1
        np.testing.assert_array_equal(E[0, :], [1.0, 1.0, 0.0])
        # Bus 20 maps to DER 2
        np.testing.assert_array_equal(E[1, :], [0.0, 0.0, 1.0])

    def test_aggregate_to_bus(self):
        m = self._make_simple_mapping()
        u_der = np.array([10.0, 5.0, 20.0])
        u_bus = m.aggregate_to_bus(u_der)
        np.testing.assert_array_almost_equal(u_bus, [15.0, 20.0])

    def test_disaggregate_capacity_weighted(self):
        m = self._make_simple_mapping()
        u_bus = np.array([80.0, 100.0])  # bus 10: 80, bus 20: 100
        u_der = m.disaggregate_to_der(u_bus, method="capacity_weighted")
        # Bus 10: 80 split by capacity 50/(50+30) and 30/(50+30)
        assert u_der[0] == pytest.approx(80.0 * 50.0 / 80.0)
        assert u_der[1] == pytest.approx(80.0 * 30.0 / 80.0)
        # Bus 20: single DER gets all
        assert u_der[2] == pytest.approx(100.0)

    def test_disaggregate_equal(self):
        m = self._make_simple_mapping()
        u_bus = np.array([80.0, 100.0])
        u_der = m.disaggregate_to_der(u_bus, method="equal")
        assert u_der[0] == pytest.approx(40.0)
        assert u_der[1] == pytest.approx(40.0)
        assert u_der[2] == pytest.approx(100.0)

    def test_bus_to_der_indices(self):
        m = self._make_simple_mapping()
        assert m.bus_to_der_indices(10) == [0, 1]
        assert m.bus_to_der_indices(20) == [2]
        assert m.bus_to_der_indices(99) == []

    def test_get_bus_aggregated_s_rated(self):
        m = self._make_simple_mapping()
        s_bus = m.get_bus_aggregated_s_rated()
        np.testing.assert_array_almost_equal(s_bus, [80.0, 100.0])

    def test_validation_mismatch_lengths(self):
        with pytest.raises(ValueError, match="bus_indices"):
            DERMapping(
                sgen_indices=(0, 1),
                bus_indices=(10,),  # wrong length
                s_rated_mva=np.array([50.0, 30.0]),
                p_max_mw=np.array([40.0, 25.0]),
                weights=np.ones(2),
            )

    def test_single_der_per_bus(self):
        """No co-located DERs: E should be identity."""
        m = DERMapping(
            sgen_indices=(0, 1),
            bus_indices=(10, 20),
            s_rated_mva=np.array([50.0, 100.0]),
            p_max_mw=np.array([40.0, 80.0]),
            weights=np.ones(2),
        )
        E = m.E
        np.testing.assert_array_equal(E, np.eye(2))


class TestHExpansion:
    """Test that H_bus @ E produces the correct H_der."""

    def test_expand_duplicates_columns(self):
        """Two DERs at the same bus should produce identical H columns."""
        m = DERMapping(
            sgen_indices=(0, 1, 2),
            bus_indices=(10, 10, 20),
            s_rated_mva=np.array([50.0, 30.0, 100.0]),
            p_max_mw=np.array([40.0, 25.0, 80.0]),
            weights=np.ones(3),
        )
        E = m.E

        # Simulate a bus-level H with 2 outputs and 2 bus-DER columns
        H_bus = np.array([
            [0.1, 0.3],  # dV1/dQ_bus10, dV1/dQ_bus20
            [0.2, 0.4],  # dV2/dQ_bus10, dV2/dQ_bus20
        ])

        H_der = H_bus @ E  # (2, 3)
        assert H_der.shape == (2, 3)

        # Columns 0 and 1 (both at bus 10) should be identical
        np.testing.assert_array_equal(H_der[:, 0], H_der[:, 1])
        # Column 2 (bus 20) should match H_bus column 1
        np.testing.assert_array_equal(H_der[:, 2], H_bus[:, 1])

    def test_rank_deficiency(self):
        """H_der should be rank-deficient when DERs share a bus."""
        m = DERMapping(
            sgen_indices=(0, 1, 2),
            bus_indices=(10, 10, 20),
            s_rated_mva=np.array([50.0, 30.0, 100.0]),
            p_max_mw=np.array([40.0, 25.0, 80.0]),
            weights=np.ones(3),
        )
        H_bus = np.array([
            [0.1, 0.3],
            [0.2, 0.4],
        ])
        H_der = H_bus @ m.E
        assert np.linalg.matrix_rank(H_bus) == 2  # full rank
        assert np.linalg.matrix_rank(H_der) == 2  # rank = n_unique_bus, not n_der

    def test_no_expansion_without_duplicates(self):
        """When each DER is on its own bus, H_der == H_bus."""
        m = DERMapping(
            sgen_indices=(0, 1),
            bus_indices=(10, 20),
            s_rated_mva=np.array([50.0, 100.0]),
            p_max_mw=np.array([40.0, 80.0]),
            weights=np.ones(2),
        )
        H_bus = np.array([
            [0.1, 0.3],
            [0.2, 0.4],
        ])
        H_der = H_bus @ m.E
        np.testing.assert_array_equal(H_der, H_bus)


class TestMIQPPerDERWeights:
    """Test that per-DER weight vectors work in build_miqp_problem."""

    def test_per_variable_weights(self):
        from optimisation.miqp_solver import build_miqp_problem

        n_total = 3
        n_outputs = 2
        H = np.random.randn(n_outputs, n_total)
        u_current = np.zeros(n_total)
        y_current = np.zeros(n_outputs)
        grad_f = np.zeros(n_total)
        u_lower = -np.ones(n_total)
        u_upper = np.ones(n_total)
        y_lower = -np.ones(n_outputs)
        y_upper = np.ones(n_outputs)
        alpha = 0.1

        # Scalar weights
        problem_scalar = build_miqp_problem(
            alpha=alpha,
            u_current=u_current,
            y_current=y_current,
            H=H, grad_f=grad_f,
            u_lower=u_lower, u_upper=u_upper,
            y_lower=y_lower, y_upper=y_upper,
            g_w=1.0, g_u=0.0, g_z=100.0,
            g_w_vector=None,
        )

        # Vector weights (should match scalar when uniform)
        problem_vector = build_miqp_problem(
            alpha=alpha,
            u_current=u_current,
            y_current=y_current,
            H=H, grad_f=grad_f,
            u_lower=u_lower, u_upper=u_upper,
            y_lower=y_lower, y_upper=y_upper,
            g_w=1.0, g_u=0.0, g_z=100.0,
            g_w_vector=np.ones(n_total),
        )

        np.testing.assert_array_almost_equal(
            problem_scalar.G_w, problem_vector.G_w
        )

    def test_nonuniform_weights_differ(self):
        from optimisation.miqp_solver import build_miqp_problem

        n_total = 3
        n_outputs = 2
        H = np.random.randn(n_outputs, n_total)

        problem = build_miqp_problem(
            alpha=0.1,
            u_current=np.zeros(n_total),
            y_current=np.zeros(n_outputs),
            H=H, grad_f=np.zeros(n_total),
            u_lower=-np.ones(n_total), u_upper=np.ones(n_total),
            y_lower=-np.ones(n_outputs), y_upper=np.ones(n_outputs),
            g_w=1.0, g_u=0.0, g_z=100.0,
            g_w_vector=np.array([1.0, 2.0, 0.5]),
        )

        # G_w should be diagonal with different values
        assert problem.G_w[0, 0] == pytest.approx(1.0)
        assert problem.G_w[1, 1] == pytest.approx(2.0)
        assert problem.G_w[2, 2] == pytest.approx(0.5)
