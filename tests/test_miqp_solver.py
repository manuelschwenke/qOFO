"""
Unit Tests for MIQP Solver
==========================

Tests for the Mixed-Integer Quadratic Programme solver.

Author: Manuel Schwenke
Date: 2025-02-05
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from optimisation.miqp_solver import (
    MIQPSolver,
    MIQPProblem,
    MIQPResult,
    build_miqp_problem,
)


class TestMIQPProblem:
    """Tests for MIQPProblem data class validation."""
    
    def test_valid_problem_construction(self) -> None:
        """Test that a valid problem can be constructed."""
        n_continuous = 3
        n_integer = 2
        n_outputs = 4
        n_total = n_continuous + n_integer
        
        problem = MIQPProblem(
            n_continuous=n_continuous,
            n_integer=n_integer,
            n_outputs=n_outputs,
            alpha=0.5,
            G_w=np.eye(n_total),
            G_z=np.eye(n_outputs),
            G_s=np.eye(n_integer),
            grad_f=np.zeros(n_total),
            H_tilde=np.ones((n_outputs, n_total)),
            u_current=np.zeros(n_total),
            u_lower=-np.ones(n_total),
            u_upper=np.ones(n_total),
            y_current=np.zeros(n_outputs),
            y_lower=-np.ones(n_outputs),
            y_upper=np.ones(n_outputs),
            integer_indices=list(range(n_continuous, n_total)),
        )
        
        assert problem.n_total == n_total
        assert problem.n_continuous == n_continuous
        assert problem.n_integer == n_integer
    
    def test_invalid_G_w_shape_raises(self) -> None:
        """Test that incorrect G_w shape raises ValueError."""
        with pytest.raises(ValueError, match="G_w shape"):
            MIQPProblem(
                n_continuous=2,
                n_integer=0,
                n_outputs=2,
                alpha=0.5,
                G_w=np.eye(3),  # Wrong shape
                G_z=np.eye(2),
                G_s=np.zeros((0, 0)),
                grad_f=np.zeros(2),
                H_tilde=np.ones((2, 2)),
                u_current=np.zeros(2),
                u_lower=-np.ones(2),
                u_upper=np.ones(2),
                y_current=np.zeros(2),
                y_lower=-np.ones(2),
                y_upper=np.ones(2),
                integer_indices=[],
            )
    
    def test_invalid_alpha_raises(self) -> None:
        """Test that non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            MIQPProblem(
                n_continuous=2,
                n_integer=0,
                n_outputs=2,
                alpha=-0.5,  # Invalid
                G_w=np.eye(2),
                G_z=np.eye(2),
                G_s=np.zeros((0, 0)),
                grad_f=np.zeros(2),
                H_tilde=np.ones((2, 2)),
                u_current=np.zeros(2),
                u_lower=-np.ones(2),
                u_upper=np.ones(2),
                y_current=np.zeros(2),
                y_lower=-np.ones(2),
                y_upper=np.ones(2),
                integer_indices=[],
            )
    
    def test_invalid_bounds_ordering_raises(self) -> None:
        """Test that u_lower > u_upper raises ValueError."""
        with pytest.raises(ValueError, match="u_lower"):
            MIQPProblem(
                n_continuous=2,
                n_integer=0,
                n_outputs=2,
                alpha=0.5,
                G_w=np.eye(2),
                G_z=np.eye(2),
                G_s=np.zeros((0, 0)),
                grad_f=np.zeros(2),
                H_tilde=np.ones((2, 2)),
                u_current=np.zeros(2),
                u_lower=np.array([1.0, -1.0]),  # First element invalid
                u_upper=np.array([0.5, 1.0]),
                y_current=np.zeros(2),
                y_lower=-np.ones(2),
                y_upper=np.ones(2),
                integer_indices=[],
            )


class TestMIQPResult:
    """Tests for MIQPResult data class."""
    
    def test_is_optimal_true(self) -> None:
        """Test is_optimal property when status is optimal."""
        result = MIQPResult(
            w_continuous=np.array([0.1, 0.2]),
            w_integer=np.array([], dtype=np.int64),
            z=np.array([0.0]),
            objective_value=0.5,
            status='optimal',
            solve_time_s=0.01,
        )
        assert result.is_optimal is True
    
    def test_is_optimal_false(self) -> None:
        """Test is_optimal property when status is not optimal."""
        result = MIQPResult(
            w_continuous=np.zeros(2),
            w_integer=np.array([], dtype=np.int64),
            z=np.zeros(1),
            objective_value=np.inf,
            status='infeasible',
            solve_time_s=0.01,
        )
        assert result.is_optimal is False


class TestBuildMIQPProblem:
    """Tests for build_miqp_problem convenience function."""
    
    def test_build_continuous_only_problem(self) -> None:
        """Test building a problem with only continuous variables."""
        n_u = 3
        n_y = 2
        
        problem = build_miqp_problem(
            alpha=0.5,
            u_current=np.zeros(n_u),
            y_current=np.zeros(n_y),
            H=np.random.randn(n_y, n_u),
            grad_f=np.zeros(n_u),
            u_lower=-10 * np.ones(n_u),
            u_upper=10 * np.ones(n_u),
            y_lower=-1 * np.ones(n_y),
            y_upper=1 * np.ones(n_y),
            g_w=1.0,
            g_u=0.1,
            g_z=100.0,
            g_s=10.0,
            integer_indices=None,
        )
        
        assert problem.n_continuous == n_u
        assert problem.n_integer == 0
        assert problem.n_outputs == n_y
    
    def test_build_mixed_integer_problem(self) -> None:
        """Test building a problem with mixed integer and continuous variables."""
        n_continuous = 3
        n_integer = 2
        n_total = n_continuous + n_integer
        n_y = 4
        
        problem = build_miqp_problem(
            alpha=0.5,
            u_current=np.zeros(n_total),
            y_current=np.zeros(n_y),
            H=np.random.randn(n_y, n_total),
            grad_f=np.zeros(n_total),
            u_lower=-10 * np.ones(n_total),
            u_upper=10 * np.ones(n_total),
            y_lower=-1 * np.ones(n_y),
            y_upper=1 * np.ones(n_y),
            g_w=1.0,
            g_u=0.1,
            g_z=100.0,
            g_s=10.0,
            integer_indices=[3, 4],  # Last two are integers
        )
        
        assert problem.n_continuous == n_continuous
        assert problem.n_integer == n_integer
        assert problem.n_outputs == n_y
    
    def test_build_invalid_H_shape_raises(self) -> None:
        """Test that mismatched H shape raises ValueError."""
        with pytest.raises(ValueError, match="H shape"):
            build_miqp_problem(
                alpha=0.5,
                u_current=np.zeros(3),
                y_current=np.zeros(2),
                H=np.random.randn(3, 3),  # Wrong shape
                grad_f=np.zeros(3),
                u_lower=-np.ones(3),
                u_upper=np.ones(3),
                y_lower=-np.ones(2),
                y_upper=np.ones(2),
                g_w=1.0,
                g_u=0.1,
                g_z=100.0,
                g_s=10.0,
            )


class TestMIQPSolver:
    """Tests for MIQPSolver class."""
    
    def test_solver_initialisation(self) -> None:
        """Test that solver can be initialised with default parameters."""
        solver = MIQPSolver()
        assert solver.verbose is False
        assert solver.time_limit_s == 60.0
        assert solver.mip_gap == 1e-4
    
    def test_solver_initialisation_with_params(self) -> None:
        """Test that solver can be initialised with custom parameters."""
        solver = MIQPSolver(
            solver='OSQP',
            verbose=True,
            time_limit_s=30.0,
            mip_gap=1e-3,
        )
        assert solver.solver == 'OSQP'
        assert solver.verbose is True
        assert solver.time_limit_s == 30.0
        assert solver.mip_gap == 1e-3
    
    @pytest.mark.parametrize("n_u,n_y", [(2, 2), (5, 3), (3, 5)])
    def test_solve_simple_qp(self, n_u: int, n_y: int) -> None:
        """Test solving a simple continuous QP problem."""
        # Create a simple QP: minimise ||w||^2 with bounds
        problem = build_miqp_problem(
            alpha=1.0,
            u_current=np.zeros(n_u),
            y_current=np.zeros(n_y),
            H=np.eye(n_y, n_u),  # Identity or truncated identity
            grad_f=np.ones(n_u),  # Gradient pushes towards negative
            u_lower=-10 * np.ones(n_u),
            u_upper=10 * np.ones(n_u),
            y_lower=-5 * np.ones(n_y),
            y_upper=5 * np.ones(n_y),
            g_w=1.0,
            g_u=0.0,
            g_z=1000.0,
            g_s=0.0,
        )
        
        solver = MIQPSolver(verbose=False)
        result = solver.solve(problem)
        
        # Check that we got a solution
        assert result.status in ('optimal', 'optimal_inaccurate')
        assert result.is_feasible
        assert len(result.w_continuous) == n_u
        assert result.objective_value < np.inf
    
    def test_solve_bounded_qp(self) -> None:
        """Test that QP solution respects bounds."""
        n_u = 3
        n_y = 2
        
        # Current state at upper bound
        u_current = np.array([9.0, 9.0, 9.0])
        
        problem = build_miqp_problem(
            alpha=1.0,
            u_current=u_current,
            y_current=np.zeros(n_y),
            H=np.eye(n_y, n_u),
            grad_f=-np.ones(n_u),  # Gradient pushes towards positive (increase)
            u_lower=-10 * np.ones(n_u),
            u_upper=10 * np.ones(n_u),  # Upper bound at 10
            y_lower=-100 * np.ones(n_y),
            y_upper=100 * np.ones(n_y),
            g_w=1.0,
            g_u=0.0,
            g_z=1000.0,
            g_s=0.0,
        )
        
        solver = MIQPSolver(verbose=False)
        result = solver.solve(problem)
        
        assert result.is_feasible
        
        # Check that w respects bounds: u + α*w <= u_upper
        # => w <= (u_upper - u_current) / alpha = 1
        u_new = u_current + result.w_continuous
        assert np.all(u_new <= 10.0 + 1e-6)
        assert np.all(u_new >= -10.0 - 1e-6)
    
    def test_solve_with_active_output_constraints(self) -> None:
        """Test QP with output constraints that become active."""
        n_u = 2
        n_y = 1
        
        # H maps u directly to y: y = H @ u
        H = np.array([[1.0, 1.0]])  # y = u[0] + u[1]
        
        problem = build_miqp_problem(
            alpha=1.0,
            u_current=np.array([0.0, 0.0]),
            y_current=np.array([0.0]),  # Current y = 0
            H=H,
            grad_f=np.array([-1.0, -1.0]),  # Push both towards positive
            u_lower=-10 * np.ones(n_u),
            u_upper=10 * np.ones(n_u),
            y_lower=np.array([-100.0]),
            y_upper=np.array([1.0]),  # y limited to <= 1
            g_w=0.1,
            g_u=0.0,
            g_z=100.0,  # High penalty for slack
            g_s=0.0,
        )
        
        solver = MIQPSolver(verbose=False)
        result = solver.solve(problem)
        
        assert result.is_feasible
        
        # Check that output constraint is approximately satisfied
        # y_new = y_current + alpha * H @ w = 0 + 1 * H @ w
        y_new = H @ result.w_continuous
        
        # With high slack penalty, y should be close to upper bound
        # Allow some slack due to soft constraint
        assert y_new[0] <= 1.0 + result.z[0] + 1e-6


class TestMIQPSolverIntegration:
    """Integration tests for MIQP solver with realistic OFO scenarios."""
    
    def test_voltage_regulation_scenario(self) -> None:
        """Test a voltage regulation scenario with DER Q control."""
        # 3 DERs controlling voltage at 2 buses
        n_der = 3
        n_bus = 2
        
        # Sensitivity matrix (∂V/∂Q)
        H = np.array([
            [0.02, 0.01, 0.005],  # Bus 1 sensitivities
            [0.01, 0.02, 0.01],   # Bus 2 sensitivities
        ])
        
        # Current state: voltages too low
        y_current = np.array([0.95, 0.94])  # p.u.
        y_lower = np.array([0.95, 0.95])    # Lower voltage limit
        y_upper = np.array([1.05, 1.05])    # Upper voltage limit
        
        # DER Q setpoints
        u_current = np.zeros(n_der)
        u_lower = np.array([-5.0, -5.0, -5.0])  # Mvar
        u_upper = np.array([5.0, 5.0, 5.0])     # Mvar
        
        problem = build_miqp_problem(
            alpha=0.5,
            u_current=u_current,
            y_current=y_current,
            H=H,
            grad_f=np.zeros(n_der),  # No preference for Q direction
            u_lower=u_lower,
            u_upper=u_upper,
            y_lower=y_lower,
            y_upper=y_upper,
            g_w=1.0,
            g_u=0.01,
            g_z=1000.0,
            g_s=0.0,
        )
        
        solver = MIQPSolver(verbose=False)
        result = solver.solve(problem)
        
        assert result.is_feasible
        
        # With low voltage, solver should increase Q (capacitive)
        # Check that predicted voltage improves
        y_predicted = y_current + 0.5 * H @ result.w_continuous
        
        # Voltage should increase towards lower limit
        assert np.all(y_predicted >= y_current - 1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
