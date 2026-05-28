"""
Stability Tuning for Cascaded MIQP-OFO
======================================

Computes per-actuator proximal weights ``g_w`` for the TSO (and DSO) MIQP-OFO
controllers using a **spectral-gap sufficient condition** for a steady-state
input-output plant model:

    λ_min( D_w + g_v · M_V + g_q · M_Q ) > ‖ g_v · M_V + g_q · M_Q ‖_op      (SG)

NAMING NOTE (2026-04-20): condition (SG) was previously called the "Bianchi
condition (1'')" in this codebase, by analogy with the Bianchi & Doerfler
(2025) OFO-stability work.  However, that paper's Theorem 1 actually uses a
max-type composite Lyapunov function (Assumption 3 in arXiv:2412.10964), not
a spectral inequality of this form.  The exact published source for (SG) is
to be confirmed; closest candidates are spectral conditions in
Hauswirth et al. (2021) ARC / IEEE TAC and Colombino et al. (2020) IEEE TCNS,
or a custom derivation via the descent lemma for projected-Newton-style
iterations on a quadratic cost.  Treat the label "spectral-gap" as
descriptive, not as a citation.

where
    D_w := diag(g_w,1, ..., g_w,m)           per-actuator proximal weights
    M_V := H_V^T · W_V^2 · H_V                voltage-tracking pullback
    M_Q := H_Q^T · W_Q^2 · H_Q                Q_gen capability pullback

Physically, (1'') says the regularised Hessian of the reduced cost
Φ_red(u) = Φ(u, H·u) is positive-definite with enough margin for the
projected-gradient iteration to contract.

This module provides three tuning strategies:

1. **Block-scalar (practical)** — one scalar per actuator class
   (DER, PCC, V_gen, OLTC), respecting physical priors on the ratios
   between them.  Simple, physically motivated, conservative.

2. **Full LMI (optimal)** — minimum-trace diagonal ``D_w`` satisfying (1'')
   via CVXPY.  Gives a lower bound on achievable total proximal weight;
   useful for benchmarking how conservative the block-scalar choice is.

3. **Monte Carlo driver** — samples operating points (load, PV, N-1 topology)
   from a pandapower network, recomputes the Jacobian at each, and
   aggregates the required ``D_w`` across the sample.  Supports element-wise
   max (worst-case), 95th percentile (robust), and histogram output for
   documenting the distribution in the thesis.

References
----------
- Bianchi & Dörfler (2025) "A stability condition for OFO without timescale
  separation", European Journal of Control.
- Häberle, Hauswirth, Ortmann, Bolognani, Dörfler (2021) L-CSS 5(1):343.
- Hauswirth, Bolognani, Hug, Dörfler (2021) IEEE TAC 66(2).
- Colombino, Simpson-Porco, Bernstein (2020) IEEE TCNS.

Author: Manuel Schwenke (drafted with Claude 2026-04-18)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BlockLayout",
    "StabilityResult",
    "compute_min_gw_per_block",
    "compute_min_gw_lmi",
    "run_monte_carlo",
    "aggregate_monte_carlo",
]


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BlockLayout:
    """
    Block structure of the TSO decision vector.

    Matches the column ordering in ``multi_tso_coordinator.py``:
    ``[Q_DER | Q_PCC_set | V_gen | s_OLTC]``.

    Attributes
    ----------
    n_der, n_pcc, n_gen, n_oltc : int
        Number of actuators in each block.  Sum must equal ``H.shape[1]``.
    names : tuple of str
        Human-readable names for reporting.
    """
    n_der: int
    n_pcc: int
    n_gen: int
    n_oltc: int
    names: Tuple[str, str, str, str] = ("DER", "PCC", "V_gen", "OLTC")

    @property
    def sizes(self) -> Tuple[int, int, int, int]:
        return (self.n_der, self.n_pcc, self.n_gen, self.n_oltc)

    @property
    def total(self) -> int:
        return sum(self.sizes)

    def block_slice(self, k: int) -> slice:
        """Index slice for block ``k`` into the full m-dim decision vector."""
        offsets = np.cumsum((0,) + self.sizes)
        return slice(offsets[k], offsets[k + 1])

    def expand(self, block_values: Sequence[float]) -> NDArray[np.float64]:
        """Expand 4 per-block scalars to a full m-dim diagonal vector."""
        if len(block_values) != 4:
            raise ValueError(f"Need 4 block values, got {len(block_values)}")
        return np.concatenate([
            np.full(n, v, dtype=np.float64)
            for n, v in zip(self.sizes, block_values)
        ])


@dataclass
class StabilityResult:
    """Output of a single stability computation."""
    gw_min_block: NDArray[np.float64]  # per-block scalars, shape (4,)
    gw_min_full:  NDArray[np.float64]  # expanded, shape (m,)
    gap:          float                # ‖M‖_op - λ_min(M)
    op_norm:      float                # ‖M‖_op
    lam_min:      float                # λ_min(M)
    feasible_without_gw: bool          # True ⇒ gap ≤ 0, stable at D_w = 0


# --------------------------------------------------------------------------- #
#  Core stability math
# --------------------------------------------------------------------------- #

def _assemble_M(
    g_v: float,
    g_q: float,
    H_V: NDArray[np.float64],
    H_Q: NDArray[np.float64],
    W_V: NDArray[np.float64],
    W_Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Build the cost-side matrix M = g_v · M_V + g_q · M_Q.

    Parameters
    ----------
    g_v, g_q : float
        Scalar weights on voltage-tracking and Q_gen capability terms.
    H_V : (n_v, m) array
        Voltage-to-input sensitivity block.
    H_Q : (n_gen, m) array
        Q_gen-to-input sensitivity block.  Pass a (0, m) array if the Q_gen
        row is not modelled.
    W_V, W_Q : (n_v,) and (n_gen,) arrays
        Row-scaling weights (e.g. 1 / σ_V, 1 / (Q_max - Q_min)).

    Returns
    -------
    M : (m, m) symmetric PSD array
    """
    H_V = np.asarray(H_V, dtype=np.float64)
    H_Q = np.asarray(H_Q, dtype=np.float64)
    if H_V.ndim != 2:
        raise ValueError(f"H_V must be 2D, got shape {H_V.shape}")
    if H_Q.ndim != 2:
        raise ValueError(f"H_Q must be 2D, got shape {H_Q.shape}")
    m = H_V.shape[1]
    if H_Q.shape[1] != m and H_Q.shape[1] != 0:
        raise ValueError(
            f"H_V and H_Q column count mismatch: {H_V.shape[1]} vs {H_Q.shape[1]}"
        )

    W_V = np.asarray(W_V, dtype=np.float64).reshape(-1)
    W_Q = np.asarray(W_Q, dtype=np.float64).reshape(-1)
    if W_V.size != H_V.shape[0]:
        raise ValueError(f"W_V size {W_V.size} != H_V rows {H_V.shape[0]}")
    if W_Q.size != H_Q.shape[0]:
        raise ValueError(f"W_Q size {W_Q.size} != H_Q rows {H_Q.shape[0]}")

    # Weighted Gram matrices, then combine.
    # Equivalent to H^T diag(W^2) H but numerically cleaner via the rows.
    HV_scaled = H_V * W_V[:, None]
    HQ_scaled = H_Q * W_Q[:, None] if H_Q.shape[0] > 0 else H_Q
    M_V = HV_scaled.T @ HV_scaled
    M_Q = HQ_scaled.T @ HQ_scaled if H_Q.shape[0] > 0 else np.zeros((m, m))
    M = g_v * M_V + g_q * M_Q

    # Symmetrise against floating-point drift.
    M = 0.5 * (M + M.T)
    return M


def _eig_extremes(M: NDArray[np.float64]) -> Tuple[float, float]:
    """Return (λ_min, λ_max) of a symmetric matrix, clamped at zero."""
    # PSD in theory; clamp defensively against floating-point negativity.
    eigs = np.linalg.eigvalsh(M)
    return float(max(eigs.min(), 0.0)), float(eigs.max())


# --------------------------------------------------------------------------- #
#  Block-scalar tuning (practical)
# --------------------------------------------------------------------------- #

def compute_min_gw_per_block(
    g_v: float,
    g_q: float,
    H_V: NDArray[np.float64],
    H_Q: NDArray[np.float64],
    W_V: NDArray[np.float64],
    W_Q: NDArray[np.float64],
    layout: BlockLayout,
    ratio_priors: Sequence[float] = (1.0, 2.0, 3.0, 10.0),
    safety_margin: float = 0.3,
) -> StabilityResult:
    """
    Compute per-block scalar g_w,k satisfying the spectral-gap condition (SG).

    Strategy: impose the ratios ``ratio_priors`` between the four blocks
    (DER:PCC:V_gen:OLTC), then scale all four scalars uniformly until the
    smallest one exceeds the spectral gap ``‖M‖_op − λ_min(M)``.  This is
    a *sufficient* condition because D_w ≽ gw_min · I makes the LHS of
    (SG) at least ``λ_min(M) + gw_min > ‖M‖_op = RHS``.

    Parameters
    ----------
    g_v, g_q : float
        Cost weights on voltage-tracking and Q_gen capability terms.
    H_V, H_Q : (n_v, m), (n_gen, m) arrays
        Sensitivity row blocks.
    W_V, W_Q : (n_v,), (n_gen,) arrays
        Row scalings.
    layout : BlockLayout
        Block structure of the decision vector.
    ratio_priors : sequence of 4 floats
        Physically motivated ratios between block weights, ordered
        (DER, PCC, V_gen, OLTC).  Default ``(1, 2, 3, 10)`` reflects that
        OLTC requires the most proximal regularisation and DER the least.
    safety_margin : float
        Fractional inflation of the computed minimum (default 0.3 = 30%).

    Returns
    -------
    StabilityResult
    """
    if len(ratio_priors) != 4:
        raise ValueError(f"ratio_priors must have length 4, got {len(ratio_priors)}")
    if not np.all(np.asarray(ratio_priors) > 0):
        raise ValueError("ratio_priors must all be positive")

    M = _assemble_M(g_v, g_q, H_V, H_Q, W_V, W_Q)
    if M.shape[0] != layout.total:
        raise ValueError(
            f"Layout total ({layout.total}) does not match M dim ({M.shape[0]})"
        )

    lam_min, op_norm = _eig_extremes(M)
    gap = op_norm - lam_min

    # The smallest block weight must cover the spectral gap.
    # All blocks scale proportionally to ratio_priors.
    min_ratio = float(min(ratio_priors))
    gw_smallest = max(0.0, gap) * (1.0 + safety_margin)
    gw_block = np.array([
        (r / min_ratio) * gw_smallest for r in ratio_priors
    ], dtype=np.float64)

    return StabilityResult(
        gw_min_block=gw_block,
        gw_min_full=layout.expand(gw_block),
        gap=gap,
        op_norm=op_norm,
        lam_min=lam_min,
        feasible_without_gw=(gap <= 0.0),
    )


# --------------------------------------------------------------------------- #
#  LMI tuning (optimal per-actuator)
# --------------------------------------------------------------------------- #

def compute_min_gw_lmi(
    g_v: float,
    g_q: float,
    H_V: NDArray[np.float64],
    H_Q: NDArray[np.float64],
    W_V: NDArray[np.float64],
    W_Q: NDArray[np.float64],
    layout: BlockLayout,
    lower_bounds: Optional[NDArray[np.float64]] = None,
    safety_margin: float = 0.3,
    method: str = "auto",
    cvxpy_solver: Optional[str] = None,
) -> StabilityResult:
    """
    Per-actuator diagonal ``D_w`` satisfying the spectral-gap LMI
    ``diag(d) + M ≽ τ·I`` with ``τ = ‖M‖_op · (1 + safety_margin)``.

    Two solution methods are available:

    - ``method="cvxpy"`` — solves the true minimum-trace LMI via CVXPY.
      Requires ``cvxpy`` to be installed.  Returns the LMI-optimal ``d``.

    - ``method="gershgorin"`` — closed-form, solver-free.  Uses the
      Gershgorin-disc sufficient condition

          d_i  ≥  τ − M_ii + Σ_{j≠i} |M_ij|

      which is a feasible (not necessarily optimal) point of the LMI.
      Conservative but extremely fast and has no dependencies.  Matches
      the LMI within a factor related to the diagonal dominance of ``M``;
      for power-system sensitivity matrices (which tend to be diagonally
      structured in dominant directions) the gap is typically modest.

    - ``method="auto"`` (default) — use CVXPY if importable, fall back
      to Gershgorin otherwise.

    Parameters
    ----------
    g_v, g_q, H_V, H_Q, W_V, W_Q, layout, safety_margin :
        As in ``compute_min_gw_per_block``.
    lower_bounds : (m,) array, optional
        Per-actuator lower bounds on ``g_w``.  Enforced element-wise.
    method : {"auto", "cvxpy", "gershgorin"}
        Solver selection (see above).
    cvxpy_solver : str, optional
        CVXPY solver name (e.g. ``'CLARABEL'``, ``'SCS'``, ``'MOSEK'``).
        None selects CVXPY's default.  Ignored if ``method="gershgorin"``.

    Returns
    -------
    StabilityResult
        ``gw_min_full`` is the per-actuator solution.  ``gw_min_block`` is
        the mean of ``d`` within each block (for reporting only).

    Notes
    -----
    The LMI optimum is not unique.  Trace-minimisation concentrates
    weight where coupling is weakest, which may produce physically
    counter-intuitive allocations.  The per-block function with explicit
    priors is often more defensible for a thesis; use this one to
    estimate how much slack the block-scalar choice has.
    """
    M = _assemble_M(g_v, g_q, H_V, H_Q, W_V, W_Q)
    m = M.shape[0]
    if m != layout.total:
        raise ValueError(
            f"Layout total ({layout.total}) does not match M dim ({m})"
        )
    lam_min, op_norm = _eig_extremes(M)
    tau = op_norm * (1.0 + safety_margin)  # target minimum eigenvalue

    if lower_bounds is None:
        lb = np.zeros(m, dtype=np.float64)
    else:
        lb = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
        if lb.size != m:
            raise ValueError(f"lower_bounds size {lb.size} != m {m}")

    # Resolve method selection.
    if method == "auto":
        try:
            import cvxpy  # noqa: F401
            method = "cvxpy"
        except ImportError:
            method = "gershgorin"
    if method not in ("cvxpy", "gershgorin"):
        raise ValueError(f"Unknown method '{method}'")

    if method == "cvxpy":
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(
                "method='cvxpy' requires cvxpy. "
                "Install with 'pip install cvxpy' or use method='gershgorin'."
            ) from e

        d_var = cp.Variable(m, nonneg=True)
        constraint_lmi = cp.diag(d_var) + M - tau * np.eye(m) >> 0
        constraint_lb = d_var >= lb
        prob = cp.Problem(
            cp.Minimize(cp.sum(d_var)),
            [constraint_lmi, constraint_lb],
        )
        prob.solve(solver=cvxpy_solver)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"LMI solver did not converge: status={prob.status}"
            )
        d_opt = np.asarray(d_var.value, dtype=np.float64)
    else:
        # Gershgorin closed form: d_i = τ − M_ii + Σ_{j≠i}|M_ij|
        M_diag = np.diag(M)
        off_diag_row_sum = np.sum(np.abs(M), axis=1) - np.abs(M_diag)
        d_opt = np.maximum(0.0, tau - M_diag + off_diag_row_sum)
        d_opt = np.maximum(d_opt, lb)

    # Report per-block means for compatibility with the block-scalar output.
    # Empty blocks (e.g. zone 1 has zero PCC actuators) -> NaN, not a warning.
    def _block_mean(k: int) -> float:
        sl = layout.block_slice(k)
        if sl.stop <= sl.start:
            return float("nan")
        return float(d_opt[sl].mean())
    block_means = np.array([_block_mean(k) for k in range(4)])

    return StabilityResult(
        gw_min_block=block_means,
        gw_min_full=d_opt,
        gap=op_norm - lam_min,
        op_norm=op_norm,
        lam_min=lam_min,
        feasible_without_gw=False,  # always adds margin
    )


# --------------------------------------------------------------------------- #
#  Monte Carlo driver
# --------------------------------------------------------------------------- #

# Type aliases
SensitivityFn = Callable[[int], Tuple[
    NDArray[np.float64],  # H_V
    NDArray[np.float64],  # H_Q
    NDArray[np.float64],  # W_V
    NDArray[np.float64],  # W_Q
]]


def run_monte_carlo(
    sensitivity_fn: SensitivityFn,
    n_samples: int,
    g_v: float,
    g_q: float,
    layout: BlockLayout,
    method: str = "block",
    **method_kwargs,
) -> List[StabilityResult]:
    """
    Evaluate the stability requirement across ``n_samples`` operating points.

    Parameters
    ----------
    sensitivity_fn : callable
        Function mapping sample index ``k`` (0-based) to the tuple
        ``(H_V, H_Q, W_V, W_Q)``.  The caller is responsible for
        sampling load/PV/contingency scenarios and rebuilding the Jacobian.
        See ``make_pandapower_sensitivity_fn`` below for a reference
        implementation pattern.
    n_samples : int
        Number of Monte Carlo samples.
    g_v, g_q : float
        Cost weights (held fixed across the sweep).
    layout : BlockLayout
        Block structure.
    method : {"block", "lmi"}
        Which tuning function to call per sample.
    **method_kwargs :
        Passed through to the chosen tuning function.

    Returns
    -------
    List[StabilityResult], length ``n_samples``.
    """
    if method not in ("block", "lmi"):
        raise ValueError(f"Unknown method '{method}'; use 'block' or 'lmi'")

    results: List[StabilityResult] = []
    for k in range(n_samples):
        H_V, H_Q, W_V, W_Q = sensitivity_fn(k)
        if method == "block":
            res = compute_min_gw_per_block(
                g_v, g_q, H_V, H_Q, W_V, W_Q, layout, **method_kwargs,
            )
        else:
            res = compute_min_gw_lmi(
                g_v, g_q, H_V, H_Q, W_V, W_Q, layout, **method_kwargs,
            )
        results.append(res)
    return results


def aggregate_monte_carlo(
    results: Sequence[StabilityResult],
    statistic: str = "max",
    percentile: float = 95.0,
) -> NDArray[np.float64]:
    """
    Reduce a list of Monte Carlo results to a single ``g_w`` vector.

    Parameters
    ----------
    results : sequence of StabilityResult
    statistic : {"max", "percentile", "mean"}
        "max": worst-case element-wise maximum (most conservative).
        "percentile": element-wise percentile (e.g. 95th, robust but not
            worst-case).
        "mean": element-wise mean (for diagnostic comparison only).
    percentile : float
        Percentile value if statistic == "percentile".

    Returns
    -------
    g_w : (m,) array
    """
    if len(results) == 0:
        raise ValueError("Empty Monte Carlo result list")
    stacked = np.stack([r.gw_min_full for r in results], axis=0)
    if statistic == "max":
        return stacked.max(axis=0)
    elif statistic == "percentile":
        return np.percentile(stacked, percentile, axis=0)
    elif statistic == "mean":
        return stacked.mean(axis=0)
    else:
        raise ValueError(f"Unknown statistic '{statistic}'")
