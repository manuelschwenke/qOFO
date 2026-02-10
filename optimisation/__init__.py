"""
Optimisation Module
===================

This module provides the MIQP solver interface for the OFO controllers.

Classes
-------
MIQPSolver
    Mixed-Integer Quadratic Programme solver using CVXPY.
MIQPProblem
    Data class for MIQP problem formulation.
MIQPResult
    Data class for MIQP solution results.

Functions
---------
build_miqp_problem
    Convenience function to construct MIQP problems from OFO data.
"""

from .miqp_solver import (
    MIQPSolver,
    MIQPProblem,
    MIQPResult,
    build_miqp_problem,
)

__all__ = [
    "MIQPSolver",
    "MIQPProblem",
    "MIQPResult",
    "build_miqp_problem",
]
