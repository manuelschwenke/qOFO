"""Lyapunov and projection-IQC certificates for a frozen linear map."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.linalg import schur

from .iqc import _VALIDATION_TOL, _solve_problem
from .models import CertificateStatus, LMIResult


def _fixed_rho_linear(
    A: np.ndarray,
    rho: float,
) -> tuple[bool, str | None, float | None, float | None]:
    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    margin = 1e-8
    lmi = A.T @ P @ A - rho**2 * P
    problem = cp.Problem(
        cp.Minimize(cp.trace(P)),
        [P >> np.eye(n), lmi << -margin * np.eye(n)],
    )
    solver = _solve_problem(problem)
    if solver is None or P.value is None:
        return False, solver, None, None
    p_value = 0.5 * (P.value + P.value.T)
    numeric = A.T @ p_value @ A - rho**2 * p_value
    residual = float(np.max(np.linalg.eigvalsh(0.5 * (numeric + numeric.T))))
    p_eigs = np.linalg.eigvalsh(p_value)
    condition = float(p_eigs[-1] / max(p_eigs[0], 1e-15))
    valid = p_eigs[0] >= 1.0 - _VALIDATION_TOL and residual <= _VALIDATION_TOL
    return valid, solver, residual, condition


def linear_rate_certificate(
    A: np.ndarray,
    *,
    bisection_steps: int = 18,
) -> LMIResult:
    """Certify ``A.T P A - rho^2 P <= 0`` for an exact linear map."""

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if A.size == 0:
        return LMIResult(
            status=CertificateStatus.NOT_APPLICABLE,
            rho=None,
            note="The map has no states.",
        )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if spectral_radius >= 1.0 - 1e-9:
        return LMIResult(
            status=CertificateStatus.NOT_CERTIFIED,
            rho=None,
            spectral_radius=spectral_radius,
            note="The exact linear map has spectral radius greater than or equal to one.",
        )

    low = max(0.0, spectral_radius * (1.0 + 1e-8))
    high = 1.0 - 1e-7
    ok, solver, residual, condition = _fixed_rho_linear(A, high)
    if not ok:
        return LMIResult(
            status=CertificateStatus.SOLVER_ERROR,
            rho=None,
            spectral_radius=spectral_radius,
            solver=solver,
            max_lmi_eigenvalue=residual,
            p_condition=condition,
            note="A stable map was detected spectrally, but the SDP was not validated.",
        )
    best = (solver, residual, condition)
    for _ in range(bisection_steps):
        mid = 0.5 * (low + high)
        feasible, mid_solver, mid_residual, mid_condition = _fixed_rho_linear(A, mid)
        if feasible:
            high = mid
            best = (mid_solver, mid_residual, mid_condition)
        else:
            low = mid
    solver, residual, condition = best
    return LMIResult(
        status=CertificateStatus.CERTIFIED,
        rho=float(high),
        spectral_radius=spectral_radius,
        solver=solver,
        max_lmi_eigenvalue=residual,
        p_condition=condition,
        note="Exact frozen linear-map certificate (constraints locally inactive).",
    )


def _fixed_rho_projection(
    A: np.ndarray,
    rho: float,
) -> tuple[bool, str | None, float | None, float | None, float | None]:
    """Projection-IQC feasibility for x+ = projection(A x)."""

    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable(nonneg=True)
    zero = np.zeros((n, n))
    base = cp.bmat([[-rho**2 * P, zero], [zero, P]])
    projection_iqc = np.block([[zero, A.T], [A, -2.0 * np.eye(n)]])
    matrix = base + lam * projection_iqc
    problem = cp.Problem(
        cp.Minimize(cp.trace(P)),
        [P >> np.eye(n), matrix << 0.0],
    )
    solver = _solve_problem(problem)
    if solver is None or P.value is None or lam.value is None:
        return False, solver, None, None, None
    p_value = 0.5 * (P.value + P.value.T)
    base_value = np.block(
        [
            [-rho**2 * p_value, zero],
            [zero, p_value],
        ]
    )
    numeric = base_value + float(lam.value) * projection_iqc
    residual = float(np.max(np.linalg.eigvalsh(0.5 * (numeric + numeric.T))))
    p_eigs = np.linalg.eigvalsh(p_value)
    condition = float(p_eigs[-1] / max(p_eigs[0], 1e-15))
    valid = p_eigs[0] >= 1.0 - _VALIDATION_TOL and residual <= _VALIDATION_TOL
    return valid, solver, residual, condition, float(lam.value)


def projected_linear_rate_certificate(
    A: np.ndarray,
    *,
    bisection_steps: int = 18,
) -> LMIResult:
    """Certify a fixed-metric projection following the frozen affine map.

    The incremental projection IQC is ``2 p.T (y-p) >= 0``.  The result
    is only applicable when the controller really is a projection onto
    one fixed convex feasible set in the preconditioned coordinates.
    """

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if A.size == 0:
        return LMIResult(CertificateStatus.NOT_APPLICABLE, None)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    ok, solver, residual, condition, multiplier = _fixed_rho_projection(
        A, 1.0 - 1e-7
    )
    if not ok:
        return LMIResult(
            status=CertificateStatus.NOT_CERTIFIED,
            rho=None,
            spectral_radius=spectral_radius,
            solver=solver,
            max_lmi_eigenvalue=residual,
            p_condition=condition,
            note="No full-state projected-map contraction below one was certified.",
        )
    low, high = 0.0, 1.0 - 1e-7
    best = (solver, residual, condition, multiplier)
    for _ in range(bisection_steps):
        mid = 0.5 * (low + high)
        feasible, mid_solver, mid_residual, mid_condition, mid_multiplier = (
            _fixed_rho_projection(A, mid)
        )
        if feasible:
            high = mid
            best = (mid_solver, mid_residual, mid_condition, mid_multiplier)
        else:
            low = mid
    solver, residual, condition, multiplier = best
    return LMIResult(
        status=CertificateStatus.CERTIFIED,
        rho=float(high),
        spectral_radius=spectral_radius,
        solver=solver,
        max_lmi_eigenvalue=residual,
        p_condition=condition,
        multipliers={"lambda_projection": float(multiplier or 0.0)},
        note="Full-state frozen projected-map IQC certificate.",
    )


def active_invariant_block(
    A: np.ndarray,
    *,
    relative_tolerance: float = 1e-8,
) -> tuple[np.ndarray, int, float]:
    """Return the real-Schur invariant block away from neutral eigenvalue one."""

    A = np.asarray(A, dtype=float)
    M = np.eye(A.shape[0]) - A
    curvature_radius = float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0
    tolerance = max(1e-10, relative_tolerance * max(curvature_radius, 1.0))

    def select(real: float, imag: float) -> bool:
        return abs(complex(real, imag) - 1.0) > tolerance

    T, _, n_active = schur(A, output="real", sort=select)
    return T[:n_active, :n_active], A.shape[0] - n_active, tolerance
