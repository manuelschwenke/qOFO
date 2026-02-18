"""
MIQP Solver Module
==================

This module provides the Mixed-Integer Quadratic Programme (MIQP) solver
interface for the OFO controllers.

The solver uses CVXPY with SCIP as the backend for mixed-integer problems.

The MIQP problem solved at each OFO iteration is:

    σ^k = argmin  g(w, z, Δs)
           w,z,Δs

where:
    g = w^T G_w w + ∇f^T H̃ w + z^T G_z z

subject to:
    αw ∈ [u_LL - u^k, u_UL - u^k]             (input constraints)
    α∇H w ∈ [y_LL - y^k - z, y_UL - y^k + z]  (output constraints with slack)
    z ≥ 0                                      (slack non-negativity)
    w_i ∈ ℤ                                    (integer variables)

References
----------
[1] Schwenke et al., PSCC 2026, Section III (Optimisation Formulation)
[2] CVXPY Documentation: https://www.cvxpy.org/

Author: Manuel Schwenke
Date: 2025-02-05
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

try:
    import cvxpy as cp
except ImportError as e:
    raise ImportError(
        "CVXPY is required for the MIQP solver. "
        "Install it with: pip install cvxpy"
    ) from e


@dataclass(frozen=True)
class MIQPResult:
    """
    Result of the MIQP optimisation.
    
    This immutable dataclass contains the solution and solver status.
    
    Attributes
    ----------
    w_continuous : NDArray[np.float64]
        Optimal change in continuous control variables (DER Q).
    w_integer : NDArray[np.int64]
        Optimal change in integer control variables (OLTC taps, shunt states).
    z : NDArray[np.float64]
        Optimal slack variables for soft constraints.
    objective_value : float
        Optimal objective function value.
    status : str
        CVXPY solver status (e.g., 'optimal', 'infeasible').
    solve_time_s : float
        Solver computation time in seconds.
    """
    w_continuous: NDArray[np.float64]
    w_integer: NDArray[np.int64]
    z: NDArray[np.float64]
    objective_value: float
    status: str
    solve_time_s: float
    
    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status == cp.OPTIMAL
    
    @property
    def is_feasible(self) -> bool:
        """Check if a feasible solution was found."""
        return self.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@dataclass
class MIQPProblem:
    """
    MIQP problem formulation for a single OFO iteration.
    
    This class encapsulates all data required to formulate the MIQP problem
    at iteration k. The problem structure follows Equations (23)-(27) from
    the PSCC 2026 paper.
    
    Attributes
    ----------
    n_continuous : int
        Number of continuous control variables (DER Q setpoints).
    n_integer : int
        Number of integer control variables (OLTC taps + shunt states).
    n_outputs : int
        Number of output constraints (Q_trafo, V_bus, I_branch).
    alpha : float
        Controller gain (step size) for the OFO update.
    G_w : NDArray[np.float64]
        Quadratic weighting matrix for control changes (n_total x n_total).
    G_z : NDArray[np.float64]
        Quadratic weighting matrix for slack variables (n_outputs x n_outputs).
    grad_f : NDArray[np.float64]
        Objective gradient vector (n_total,).
    H_tilde : NDArray[np.float64]
        Modified sensitivity matrix (n_outputs x n_total).
    u_current : NDArray[np.float64]
        Current control variable values (n_total,).
    u_lower : NDArray[np.float64]
        Lower bounds on control variables (n_total,).
    u_upper : NDArray[np.float64]
        Upper bounds on control variables (n_total,).
    y_current : NDArray[np.float64]
        Current output measurements (n_outputs,).
    y_lower : NDArray[np.float64]
        Lower bounds on outputs (n_outputs,).
    y_upper : NDArray[np.float64]
        Upper bounds on outputs (n_outputs,).
    integer_indices : List[int]
        Indices of integer variables within the control vector.
    """
    n_continuous: int
    n_integer: int
    n_outputs: int
    alpha: float
    G_w: NDArray[np.float64]
    G_z: NDArray[np.float64]
    grad_f: NDArray[np.float64]
    H_tilde: NDArray[np.float64]
    u_current: NDArray[np.float64]
    u_lower: NDArray[np.float64]
    u_upper: NDArray[np.float64]
    y_current: NDArray[np.float64]
    y_lower: NDArray[np.float64]
    y_upper: NDArray[np.float64]
    integer_indices: List[int]
    
    def __post_init__(self) -> None:
        """Validate problem dimensions after initialisation."""
        n_total = self.n_continuous + self.n_integer
        
        if self.G_w.shape != (n_total, n_total):
            raise ValueError(
                f"G_w shape {self.G_w.shape} does not match "
                f"expected ({n_total}, {n_total})"
            )
        
        if self.G_z.shape != (self.n_outputs, self.n_outputs):
            raise ValueError(
                f"G_z shape {self.G_z.shape} does not match "
                f"expected ({self.n_outputs}, {self.n_outputs})"
            )
        
        if len(self.grad_f) != n_total:
            raise ValueError(
                f"grad_f length {len(self.grad_f)} does not match "
                f"n_total {n_total}"
            )
        
        if self.H_tilde.shape != (self.n_outputs, n_total):
            raise ValueError(
                f"H_tilde shape {self.H_tilde.shape} does not match "
                f"expected ({self.n_outputs}, {n_total})"
            )
        
        if len(self.u_current) != n_total:
            raise ValueError(
                f"u_current length {len(self.u_current)} does not match "
                f"n_total {n_total}"
            )
        
        if len(self.u_lower) != n_total:
            raise ValueError(
                f"u_lower length {len(self.u_lower)} does not match "
                f"n_total {n_total}"
            )
        
        if len(self.u_upper) != n_total:
            raise ValueError(
                f"u_upper length {len(self.u_upper)} does not match "
                f"n_total {n_total}"
            )
        
        if len(self.y_current) != self.n_outputs:
            raise ValueError(
                f"y_current length {len(self.y_current)} does not match "
                f"n_outputs {self.n_outputs}"
            )
        
        if len(self.y_lower) != self.n_outputs:
            raise ValueError(
                f"y_lower length {len(self.y_lower)} does not match "
                f"n_outputs {self.n_outputs}"
            )
        
        if len(self.y_upper) != self.n_outputs:
            raise ValueError(
                f"y_upper length {len(self.y_upper)} does not match "
                f"n_outputs {self.n_outputs}"
            )
        
        if len(self.integer_indices) != self.n_integer:
            raise ValueError(
                f"integer_indices length {len(self.integer_indices)} does not "
                f"match n_integer {self.n_integer}"
            )
        
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        
        # Check bounds ordering
        for i in range(n_total):
            if self.u_lower[i] > self.u_upper[i]:
                raise ValueError(
                    f"u_lower[{i}] = {self.u_lower[i]} > "
                    f"u_upper[{i}] = {self.u_upper[i]}"
                )
        
        for i in range(self.n_outputs):
            if self.y_lower[i] > self.y_upper[i]:
                raise ValueError(
                    f"y_lower[{i}] = {self.y_lower[i]} > "
                    f"y_upper[{i}] = {self.y_upper[i]}"
                )
    
    @property
    def n_total(self) -> int:
        """Total number of control variables."""
        return self.n_continuous + self.n_integer


class MIQPSolver:
    """
    Mixed-Integer Quadratic Programme solver for OFO controllers.
    
    This class provides the interface for solving the MIQP optimisation
    problem at each OFO iteration. It uses CVXPY for problem formulation
    and supports multiple solver backends.
    
    The solver handles both continuous variables (DER reactive power) and
    integer variables (OLTC tap positions, shunt states).
    
    Attributes
    ----------
    solver : str
        Name of the solver backend to use.
    verbose : bool
        Whether to print solver output.
    time_limit_s : float
        Maximum solver time in seconds.
    mip_gap : float
        Relative MIP gap tolerance for integer problems.
    
    Notes
    -----
    Recommended solvers for MIQP:
    - SCIP: Open-source, good for mixed-integer problems
    - GUROBI: Commercial, very fast (requires licence)
    - MOSEK: Commercial, good for convex problems (requires licence)
    - ECOS_BB: Open-source, branch-and-bound for small MIPs
    
    For continuous QP (no integer variables), OSQP or ECOS are efficient.
    """
    
    # Solver preference order for MIQP problems
    MIQP_SOLVERS = ['MOSEK', 'GUROBI', 'SCIP', 'ECOS_BB']
    
    # Solver preference order for QP problems (continuous only)
    QP_SOLVERS = ['OSQP', 'ECOS', 'SCS', 'CVXOPT']
    
    def __init__(
        self,
        solver: Optional[str] = None,
        verbose: bool = False,
        time_limit_s: float = 60.0,
        mip_gap: float = 1e-4,
    ) -> None:
        """
        Initialise the MIQP solver.
        
        Parameters
        ----------
        solver : str, optional
            Solver backend to use. If None, automatically selects based on
            problem type and available solvers.
        verbose : bool, optional
            Whether to print solver output (default: False).
        time_limit_s : float, optional
            Maximum solver time in seconds (default: 60.0).
        mip_gap : float, optional
            Relative MIP gap tolerance (default: 1e-4).
        """
        self.solver = solver
        self.verbose = verbose
        self.time_limit_s = time_limit_s
        self.mip_gap = mip_gap
    
    def solve(self, problem: MIQPProblem) -> MIQPResult:
        """
        Solve the MIQP problem.
        
        Parameters
        ----------
        problem : MIQPProblem
            The MIQP problem to solve.
        
        Returns
        -------
        MIQPResult
            The solution result containing optimal values and status.
        
        Raises
        ------
        ValueError
            If the problem is malformed or the solver fails unexpectedly.
        """
        # Determine if this is a pure QP or MIQP
        has_integers = problem.n_integer > 0
        
        # Select solver if not specified
        solver_name = self._select_solver(has_integers)
        
        # Build and solve the CVXPY problem
        if has_integers:
            return self._solve_miqp(problem, solver_name)
        else:
            return self._solve_qp(problem, solver_name)
    
    def _select_solver(self, has_integers: bool) -> str:
        """Select an appropriate solver based on problem type."""
        if self.solver is not None:
            return self.solver
        
        solver_list = self.MIQP_SOLVERS if has_integers else self.QP_SOLVERS
        
        for solver_name in solver_list:
            try:
                if solver_name in cp.installed_solvers():
                    return solver_name
            except Exception:
                continue
        
        # Fallback to default CVXPY solver
        return None
    
    def _solve_qp(
        self,
        problem: MIQPProblem,
        solver_name: Optional[str],
    ) -> MIQPResult:
        """
        Solve a continuous QP problem (no integer variables).
        
        Parameters
        ----------
        problem : MIQPProblem
            The QP problem to solve.
        solver_name : str or None
            Name of the solver to use.
        
        Returns
        -------
        MIQPResult
            The solution result.
        """
        n_total = problem.n_total
        n_outputs = problem.n_outputs
        alpha = problem.alpha
        
        # Decision variables
        w = cp.Variable(n_total, name='w')
        z = cp.Variable(n_outputs, name='z', nonneg=True)
        
        # Objective function (Equation 27 from PSCC paper)
        # g = w^T G_w w + ∇f^T H̃ w + z^T G_z z
        objective = (
            cp.quad_form(w, problem.G_w) +
            problem.grad_f @ w +
            cp.quad_form(z, problem.G_z)
        )
        
        # Constraints
        constraints = []
        
        # Input constraints (Equation 24): αw ∈ [u_LL - u^k, u_UL - u^k]
        w_lower = (problem.u_lower - problem.u_current) / alpha
        w_upper = (problem.u_upper - problem.u_current) / alpha
        constraints.append(w >= w_lower)
        constraints.append(w <= w_upper)
        
        # Output constraints (Equation 25): α∇H w ∈ [y_LL - y^k - z, y_UL - y^k + z]
        # Reformulated as two inequalities:
        #   α H̃ w >= y_LL - y^k - z
        #   α H̃ w <= y_UL - y^k + z
        Hw = alpha * (problem.H_tilde @ w)
        y_error_lower = problem.y_lower - problem.y_current
        y_error_upper = problem.y_upper - problem.y_current
        
        constraints.append(Hw >= y_error_lower - z)
        constraints.append(Hw <= y_error_upper + z)
        
        # Formulate and solve
        cvxpy_problem = cp.Problem(cp.Minimize(objective), constraints)
        
        solver_kwargs = {
            'verbose': self.verbose,
        }
        
        if solver_name is not None:
            solver_kwargs['solver'] = solver_name
        
        try:
            cvxpy_problem.solve(**solver_kwargs)
        except cp.SolverError as e:
            return MIQPResult(
                w_continuous=np.zeros(n_total),
                w_integer=np.array([], dtype=np.int64),
                z=np.zeros(n_outputs),
                objective_value=np.inf,
                status=str(e),
                solve_time_s=0.0,
            )
        
        # Extract solution
        if cvxpy_problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            w_opt = np.array(w.value, dtype=np.float64)
            z_opt = np.array(z.value, dtype=np.float64)
            obj_val = float(cvxpy_problem.value)
        else:
            w_opt = np.zeros(n_total)
            z_opt = np.zeros(n_outputs)
            obj_val = np.inf
        
        solve_time = cvxpy_problem.solver_stats.solve_time or 0.0
        
        return MIQPResult(
            w_continuous=w_opt,
            w_integer=np.array([], dtype=np.int64),
            z=z_opt,
            objective_value=obj_val,
            status=cvxpy_problem.status,
            solve_time_s=solve_time,
        )
    
    def _solve_miqp(
        self,
        problem: MIQPProblem,
        solver_name: Optional[str],
    ) -> MIQPResult:
        """
        Solve a mixed-integer QP problem.
        
        Parameters
        ----------
        problem : MIQPProblem
            The MIQP problem to solve.
        solver_name : str or None
            Name of the solver to use.
        
        Returns
        -------
        MIQPResult
            The solution result.
        """
        n_continuous = problem.n_continuous
        n_integer = problem.n_integer
        n_total = problem.n_total
        n_outputs = problem.n_outputs
        alpha = problem.alpha
        
        # Decision variables
        # w_c: continuous changes (DER Q)
        # w_i: integer changes (OLTC taps, shunt states)
        w_c = cp.Variable(n_continuous, name='w_continuous')
        w_i = cp.Variable(n_integer, name='w_integer', integer=True)
        z = cp.Variable(n_outputs, name='z', nonneg=True)
        
        # Combine w = [w_c; w_i] for matrix operations
        w = cp.hstack([w_c, w_i])
        
        # Objective function: g = w^T G_w w + ∇f^T H̃ w + z^T G_z z
        objective = (
            cp.quad_form(w, problem.G_w) +
            problem.grad_f @ w +
            cp.quad_form(z, problem.G_z)
        )
        
        # Constraints
        constraints = []
        
        # Input constraints for continuous variables
        w_c_lower = (problem.u_lower[:n_continuous] - 
                     problem.u_current[:n_continuous]) / alpha
        w_c_upper = (problem.u_upper[:n_continuous] - 
                     problem.u_current[:n_continuous]) / alpha
        constraints.append(w_c >= w_c_lower)
        constraints.append(w_c <= w_c_upper)
        
        # Input constraints for integer variables
        # Integer variables are changes in tap position or shunt state
        # Typically limited to small changes per iteration (e.g., ±1, ±2)
        w_i_lower = (problem.u_lower[n_continuous:] - 
                     problem.u_current[n_continuous:]) / alpha
        w_i_upper = (problem.u_upper[n_continuous:] - 
                     problem.u_current[n_continuous:]) / alpha
        
        # Round bounds to integers for integer variables
        w_i_lower_int = np.floor(w_i_lower).astype(np.int64)
        w_i_upper_int = np.ceil(w_i_upper).astype(np.int64)
        
        constraints.append(w_i >= w_i_lower_int)
        constraints.append(w_i <= w_i_upper_int)
        
        # Output constraints with slack variables
        Hw = alpha * (problem.H_tilde @ w)
        y_error_lower = problem.y_lower - problem.y_current
        y_error_upper = problem.y_upper - problem.y_current
        
        constraints.append(Hw >= y_error_lower - z)
        constraints.append(Hw <= y_error_upper + z)
        
        # Formulate and solve
        cvxpy_problem = cp.Problem(cp.Minimize(objective), constraints)
        
        solver_kwargs = {
            'verbose': self.verbose,
        }
        
        if solver_name is not None:
            solver_kwargs['solver'] = solver_name
            
            # Add solver-specific options
            if solver_name == 'SCIP':
                solver_kwargs['scip_params'] = {
                    'limits/time': self.time_limit_s,
                    'limits/gap': self.mip_gap,
                }
            elif solver_name == 'GUROBI':
                solver_kwargs['TimeLimit'] = self.time_limit_s
                solver_kwargs['MIPGap'] = self.mip_gap
            elif solver_name == 'MOSEK':
                solver_kwargs['mosek_params'] = {
                    'MSK_DPAR_OPTIMIZER_MAX_TIME': self.time_limit_s,
                    'MSK_DPAR_MIO_TOL_REL_GAP': self.mip_gap,
                }
        
        try:
            cvxpy_problem.solve(**solver_kwargs)
        except cp.SolverError as e:
            return MIQPResult(
                w_continuous=np.zeros(n_continuous),
                w_integer=np.zeros(n_integer, dtype=np.int64),
                z=np.zeros(n_outputs),
                objective_value=np.inf,
                status=str(e),
                solve_time_s=0.0,
            )
        
        # Extract solution
        if cvxpy_problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            w_c_opt = np.array(w_c.value, dtype=np.float64)
            w_i_opt = np.round(w_i.value).astype(np.int64)
            z_opt = np.array(z.value, dtype=np.float64)
            obj_val = float(cvxpy_problem.value)
        else:
            w_c_opt = np.zeros(n_continuous)
            w_i_opt = np.zeros(n_integer, dtype=np.int64)
            z_opt = np.zeros(n_outputs)
            obj_val = np.inf
        
        solve_time = cvxpy_problem.solver_stats.solve_time or 0.0
        
        return MIQPResult(
            w_continuous=w_c_opt,
            w_integer=w_i_opt,
            z=z_opt,
            objective_value=obj_val,
            status=cvxpy_problem.status,
            solve_time_s=solve_time,
        )


def build_miqp_problem(
    alpha: float,
    u_current: NDArray[np.float64],
    y_current: NDArray[np.float64],
    H: NDArray[np.float64],
    grad_f: NDArray[np.float64],
    u_lower: NDArray[np.float64],
    u_upper: NDArray[np.float64],
    y_lower: NDArray[np.float64],
    y_upper: NDArray[np.float64],
    g_w: Union[float, NDArray[np.float64]],
    g_u: Union[float, NDArray[np.float64]],
    g_z: float,
    integer_indices: Optional[List[int]] = None,
) -> MIQPProblem:
    """
    Build an MIQP problem from OFO controller data.

    This is a convenience function that constructs the weight matrices
    and problem structure from scalar weights and sensitivity data.

    Parameters
    ----------
    alpha : float
        Controller gain (step size).
    u_current : NDArray[np.float64]
        Current control variable values.
    y_current : NDArray[np.float64]
        Current output measurements.
    H : NDArray[np.float64]
        Sensitivity matrix (∇H) from Jacobian calculations.
    grad_f : NDArray[np.float64]
        Objective gradient vector.
    u_lower : NDArray[np.float64]
        Lower bounds on control variables.
    u_upper : NDArray[np.float64]
        Upper bounds on control variables.
    y_lower : NDArray[np.float64]
        Lower bounds on outputs.
    y_upper : NDArray[np.float64]
        Upper bounds on outputs.
    g_w : float or NDArray[np.float64]
        Weight for control variable changes. Either a scalar (applied
        uniformly to all variables) or an array of length n_total with
        per-variable weights for the diagonal of G_w.
    g_u : float or NDArray[np.float64]
        Weight for control variable usage (regularisation).  Either a
        scalar (uniform for all variables) or a per-variable array of
        length n_total.  Set entries to 0 for actuators that should
        not be regularised towards zero (e.g. OLTC taps, shunts).
    g_z : float
        Scalar weight for slack variables (constraint violations).
    integer_indices : List[int], optional
        Indices of integer variables within u_current. If None, all
        variables are treated as continuous.

    Returns
    -------
    MIQPProblem
        The constructed MIQP problem.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.
    """
    n_total = len(u_current)
    n_outputs = len(y_current)

    if integer_indices is None:
        integer_indices = []

    n_integer = len(integer_indices)
    n_continuous = n_total - n_integer

    # Validate dimensions
    if H.shape != (n_outputs, n_total):
        raise ValueError(
            f"H shape {H.shape} does not match expected "
            f"({n_outputs}, {n_total})"
        )

    if len(grad_f) != n_total:
        raise ValueError(
            f"grad_f length {len(grad_f)} does not match n_total {n_total}"
        )

    if len(u_lower) != n_total:
        raise ValueError(
            f"u_lower length {len(u_lower)} does not match n_total {n_total}"
        )

    if len(u_upper) != n_total:
        raise ValueError(
            f"u_upper length {len(u_upper)} does not match n_total {n_total}"
        )

    if len(y_lower) != n_outputs:
        raise ValueError(
            f"y_lower length {len(y_lower)} does not match n_outputs {n_outputs}"
        )

    if len(y_upper) != n_outputs:
        raise ValueError(
            f"y_upper length {len(y_upper)} does not match n_outputs {n_outputs}"
        )

    # Build weight matrices
    # G_w combines the change weight and usage regularisation weight:
    #   G_w = diag(g_w) + α² · diag(g_u)
    # The g_u term adds a quadratic penalty on the absolute level of u,
    # which penalises actuator *usage* (deviation from zero).  When g_u
    # is a per-variable vector, only selected actuators are regularised.
    g_w_vec = np.broadcast_to(np.asarray(g_w, dtype=np.float64), (n_total,)).copy()
    g_u_vec = np.broadcast_to(np.asarray(g_u, dtype=np.float64), (n_total,)).copy()
    G_w = np.diag(g_w_vec + alpha**2 * g_u_vec)

    # G_z is the slack variable weight
    G_z = g_z * np.eye(n_outputs)

    # Modified gradient includes the linear part of the usage regularisation:
    #   grad_f_mod = grad_f + 2 · α · g_u · u_current
    grad_f_mod = grad_f + 2.0 * alpha * g_u_vec * u_current
    
    # Reorder u and H to put continuous variables first, then integer
    continuous_indices = [i for i in range(n_total) if i not in integer_indices]
    reorder_indices = continuous_indices + list(integer_indices)
    
    u_current_reordered = u_current[reorder_indices]
    u_lower_reordered = u_lower[reorder_indices]
    u_upper_reordered = u_upper[reorder_indices]
    grad_f_reordered = grad_f_mod[reorder_indices]
    H_reordered = H[:, reorder_indices]
    G_w_reordered = G_w[np.ix_(reorder_indices, reorder_indices)]
    
    # Integer indices in the reordered vector are at the end
    integer_indices_reordered = list(range(n_continuous, n_total))
    
    return MIQPProblem(
        n_continuous=n_continuous,
        n_integer=n_integer,
        n_outputs=n_outputs,
        alpha=alpha,
        G_w=G_w_reordered,
        G_z=G_z,
        grad_f=grad_f_reordered,
        H_tilde=H_reordered,
        u_current=u_current_reordered,
        u_lower=u_lower_reordered,
        u_upper=u_upper_reordered,
        y_current=y_current,
        y_lower=y_lower,
        y_upper=y_upper,
        integer_indices=integer_indices_reordered,
    )
