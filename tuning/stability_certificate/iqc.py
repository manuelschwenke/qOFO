"""Small IQC/LMI rate certificates used by the tuning diagnostics.

The LMIs are the repeated-scalar reduction of the sector IQC in
Lessard, Recht, and Packard (2016).  The controller convention is
``alpha = 1``: step amplitude is represented by the diagonal ``G_w``
metric before these routines are called.
"""

from __future__ import annotations

from collections.abc import Iterable

import cvxpy as cp
import numpy as np

from .models import CertificateStatus, LMIResult


_SOLVERS = ("CLARABEL", "SCS")
_VALIDATION_TOL = 2e-5


def _solve_problem(problem: cp.Problem) -> str | None:
    """Solve with the first available conic solver and return its name."""

    installed = set(cp.installed_solvers())
    for solver in _SOLVERS:
        if solver not in installed:
            continue
        kwargs: dict[str, object] = {"solver": solver, "verbose": False}
        if solver == "SCS":
            kwargs.update(eps=1e-7, max_iters=50_000)
        try:
            problem.solve(**kwargs)
        except cp.error.SolverError:
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return solver
    return None


def _sector_q(m: float, L: float) -> np.ndarray:
    return np.array([[-2.0 * m * L, m + L], [m + L, -2.0]])


def _fixed_rho_sector(
    m: float,
    L: float,
    rho: float,
    *,
    delta: float,
) -> tuple[bool, str | None, float | None, dict[str, float]]:
    """Check the sector LMI at a fixed rate.

    ``delta`` bounds multiplicative gradient/model error as
    ``||u - w|| <= delta ||w||``.  For ``delta=0`` the exact nominal
    two-by-two LMI is used.
    """

    q_gradient = _sector_q(m, L)
    if delta == 0.0:
        lam_f = cp.Variable(nonneg=True)
        base = np.array([[1.0 - rho**2, -1.0], [-1.0, 1.0]])
        matrix = base + lam_f * q_gradient
        problem = cp.Problem(cp.Minimize(0.0), [matrix << 0.0])
        solver = _solve_problem(problem)
        if solver is None or lam_f.value is None:
            return False, solver, None, {}
        numeric = base + float(lam_f.value) * q_gradient
        residual = float(np.max(np.linalg.eigvalsh(numeric)))
        return (
            residual <= _VALIDATION_TOL,
            solver,
            residual,
            {"lambda_gradient": float(lam_f.value)},
        )

    # Coordinates are [x, exact_gradient w, inexact_gradient u].
    lam_f = cp.Variable(nonneg=True)
    lam_delta = cp.Variable(nonneg=True)
    base = np.array(
        [
            [1.0 - rho**2, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    q_f_embed = np.zeros((3, 3))
    q_f_embed[:2, :2] = q_gradient
    q_delta_embed = np.zeros((3, 3))
    q_delta_embed[1:, 1:] = np.array(
        [[delta**2 - 1.0, 1.0], [1.0, -1.0]]
    )
    matrix = base + lam_f * q_f_embed + lam_delta * q_delta_embed
    problem = cp.Problem(cp.Minimize(0.0), [matrix << 0.0])
    solver = _solve_problem(problem)
    if solver is None or lam_f.value is None or lam_delta.value is None:
        return False, solver, None, {}
    numeric = (
        base
        + float(lam_f.value) * q_f_embed
        + float(lam_delta.value) * q_delta_embed
    )
    residual = float(np.max(np.linalg.eigvalsh(numeric)))
    return (
        residual <= _VALIDATION_TOL,
        solver,
        residual,
        {
            "lambda_gradient": float(lam_f.value),
            "lambda_error": float(lam_delta.value),
        },
    )


def sector_rate_certificate(
    m: float,
    L: float,
    *,
    delta: float = 0.0,
    bisection_steps: int = 24,
) -> LMIResult:
    """Find the smallest certified rate ``rho < 1`` by bisection.

    The result also applies to projected gradient descent when the
    feasible set is fixed and convex in the ``G_w``-preconditioned
    coordinates, because metric projection is nonexpansive.
    """

    if not (np.isfinite(m) and np.isfinite(L) and 0.0 < m <= L):
        return LMIResult(
            status=CertificateStatus.NOT_APPLICABLE,
            rho=None,
            note="The sector IQC requires finite 0 < m <= L.",
        )
    if not (np.isfinite(delta) and 0.0 <= delta < 1.0):
        raise ValueError("delta must satisfy 0 <= delta < 1")

    feasible, solver, residual, multipliers = _fixed_rho_sector(
        m, L, 1.0 - 1e-7, delta=delta
    )
    if not feasible:
        return LMIResult(
            status=CertificateStatus.NOT_CERTIFIED,
            rho=None,
            solver=solver,
            max_lmi_eigenvalue=residual,
            note=f"No contraction rate below one was feasible for delta={delta:g}.",
        )

    low, high = 0.0, 1.0 - 1e-7
    best = (solver, residual, multipliers)
    for _ in range(bisection_steps):
        mid = 0.5 * (low + high)
        ok, mid_solver, mid_residual, mid_multipliers = _fixed_rho_sector(
            m, L, mid, delta=delta
        )
        if ok:
            high = mid
            best = (mid_solver, mid_residual, mid_multipliers)
        else:
            low = mid
    solver, residual, multipliers = best
    return LMIResult(
        status=CertificateStatus.CERTIFIED,
        rho=float(high),
        solver=solver,
        max_lmi_eigenvalue=residual,
        multipliers=multipliers,
        note=(
            "Full-state certificate." if m > 0.0 else "Active-mode diagnostic only."
        ),
    )


def robust_rate_sweep(
    m: float,
    L: float,
    deltas: Iterable[float],
) -> dict[str, LMIResult]:
    """Evaluate explicit relative gradient/model-error assumptions."""

    return {
        f"{float(delta):g}": sector_rate_certificate(m, L, delta=float(delta))
        for delta in deltas
        if float(delta) > 0.0
    }
