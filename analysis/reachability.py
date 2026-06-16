"""
Voltage-Stability / Nose-Curve Reachability Guard
=================================================

Quasi-static (steady-state power-flow) time-series simulations resolve only
the algebraic load-flow equations at each step.  The Newton-Raphson iteration
can therefore converge to operating points that lie *on* or *beyond* the
saddle-node (nose) of the P-V / Q-V curve -- i.e. on the lower (unstable)
voltage branch.  Such points are valid solutions of the algebraic equations
but the physical dynamic system could never reach or hold them.  This module
provides a guard that, for a *converged* power-flow solution, decides whether
the operating point lies on the stable (upper) voltage branch.

Criterion
---------
For the converged solution the power-flow Jacobian (per unit, polar
coordinates) is partitioned by bus type (slack excluded; PV buses contribute
angle states only):

    J = [[ dP/dtheta, dP/dV ],
         [ dQ/dtheta, dQ/dV ]]

**(1) Singularity guard on the full Jacobian.**  The nose is exactly where
``J`` becomes singular, hence the smallest singular value ``sigma_min(J) -> 0``
signals proximity to the saddle-node.  We abort when ``sigma_min(J) <
tau_sigma``.  Note that ``sigma_min`` only detects the *immediate vicinity* of
the nose; it does not by itself distinguish the upper from the lower branch
(both branches recover a non-zero ``sigma_min`` away from the tip).

**(2) Reduced Q-V Jacobian -- modal criterion** (Gao, Morison & Kundur,
"Voltage Stability Evaluation Using Modal Analysis", IEEE Trans. Power
Systems, vol. 7, no. 4, pp. 1529-1542, 1992).  Eliminating the active-power /
angle sub-system via the Schur complement yields the reduced reactive
power-voltage Jacobian

    J_R = dQ/dV - (dQ/dtheta) (dP/dtheta)^{-1} (dP/dV)

Sign convention: **all eigenvalues of J_R having positive real part <=> the
operating point is on the stable upper branch.**  A zero or negative
eigenvalue indicates the point is at or beyond the nose (lower branch).  We
abort when ``min(real(eig(J_R))) <= tau_eig``.

For diagnostics the *critical bus* is reported as the bus with the largest
participation in the minimum-eigenvalue mode, where the bus participation is
taken as the squared entries of the corresponding right eigenvector.

Power-flow backend
------------------
The plant simulator is pandapower.  After ``pp.runpp`` the internal
Newton-Raphson Jacobian is available at ``net._ppc["internal"]["J"]`` in per
unit on the internal ppc ordering, together with the bus-type partition
(``pv`` / ``pq`` / ``ref`` index arrays) in ``net._ppc["internal"]``.  This
guard uses that Jacobian directly; it does **not** rebuild it from the bus
admittance matrix.

A subtlety specific to this repository: the time-series loop solves with
``distributed_slack=True``, which augments the internal Jacobian by one
row/column for the slack-distribution variable (shape ``n_pv + 2 n_pq + 1``
instead of the canonical ``n_pv + 2 n_pq``).  The augmented matrix is *not* a
plain principal sub-block of the canonical one, so it cannot be sliced.  When
``check_reachability`` detects this non-canonical shape it re-converges a deep
copy of the network with ``distributed_slack=False, run_control=False`` -- the
same device already used by :class:`sensitivity.jacobian.JacobianSensitivities`
-- to obtain the canonical single-slack Jacobian.  This lets pandapower's own
Newton-Raphson produce the Jacobian and is therefore *not* a hand rebuild from
Ybus.  The deep copy guarantees the caller's network (and its recorded
results) are left untouched.

Author: Manuel Schwenke / Claude Code
Date: 2026-06-08
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

import pandapower as pp

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


# ---------------------------------------------------------------------------
#  Exceptions
# ---------------------------------------------------------------------------


class ReachabilityViolation(RuntimeError):
    """Raised at the first time-series step whose equilibrium is *not* on the
    stable upper voltage branch.

    The message carries the step index, simulation time, critical bus and the
    offending stability margins so the run aborts with full diagnostics.

    Diagnostic payload (set by :meth:`ReachabilityMonitor.check_step`):

    * ``result`` -- the :class:`ReachabilityResult` of the violating step.
    * ``margins`` -- the full per-step margin trajectory up to and including the
      violating step (list of dicts; the same rows as
      :meth:`ReachabilityMonitor.to_dataframe`).  Lets callers persist the
      margin history even though the run aborts.

    The runner additionally attaches ``partial_log`` -- the iteration records
    accumulated before the violation -- so experiment drivers can keep what was
    already computed.
    """

    result: "Optional[ReachabilityResult]" = None
    margins: "Optional[list[dict]]" = None
    partial_log: "Optional[list]" = None


# ---------------------------------------------------------------------------
#  Result container
# ---------------------------------------------------------------------------


@dataclass
class ReachabilityResult:
    """Outcome of a single reachability check.

    Attributes
    ----------
    on_stable_branch
        ``True`` iff the operating point passes *both* the singularity guard
        (``sigma_min_J >= tau_sigma``) and the modal criterion
        (``lambda_min_JR > tau_eig``).
    sigma_min_J
        Smallest singular value of the full power-flow Jacobian ``J``.
    cond_J
        2-norm condition number of ``J`` (``sigma_max / sigma_min``;
        ``inf`` if ``J`` is exactly singular).
    lambda_min_JR
        Minimum real part over the eigenvalues of the reduced Q-V Jacobian
        ``J_R``.  Positive on the stable upper branch, non-positive at/beyond
        the nose.
    critical_bus
        Pandapower bus index with the largest participation in the
        minimum-eigenvalue mode (squared right-eigenvector entries).  Falls
        back to the internal ppc bus index when the critical bus is an
        auxiliary bus with no pandapower counterpart (e.g. a 3-winding
        transformer star point).
    step_index
        Time-series step the result belongs to (``None`` for a standalone
        check).
    """

    on_stable_branch: bool
    sigma_min_J: float
    cond_J: float
    lambda_min_JR: float
    critical_bus: int
    step_index: Optional[int] = None


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _ppc_bus_to_pp_bus(net: "pp.pandapowerNet", ppc_bus_idx: int) -> int:
    """Map an internal ppc bus index back to its pandapower bus index.

    Inverts ``net._pd2ppc_lookups['bus']`` (which maps pandapower -> ppc).
    Returns the pandapower bus index of the first matching entry.  Auxiliary
    buses created internally (e.g. 3-winding transformer star points) have no
    pandapower counterpart; for those the ppc index itself is returned so the
    diagnostic still identifies *a* bus.
    """
    lookups = getattr(net, "_pd2ppc_lookups", None)
    if not lookups or lookups.get("bus") is None:
        raise ValueError(
            "Cannot map ppc bus to pandapower bus: _pd2ppc_lookups['bus'] "
            "missing (run a power flow first)."
        )
    bus_lookup = np.asarray(lookups["bus"])
    matches = np.where(bus_lookup == ppc_bus_idx)[0]
    if matches.size == 0:
        # Auxiliary / star-point bus with no pandapower index.
        return int(ppc_bus_idx)
    return int(matches[0])


def _extract_dense_jacobian(net: "pp.pandapowerNet") -> np.ndarray:
    """Return the internal Newton-Raphson Jacobian as a dense float array.

    Raises a descriptive exception when the converged solution or the internal
    Jacobian is missing (fail-fast; no silent reconstruction).
    """
    if not hasattr(net, "_ppc") or net._ppc is None:
        raise ValueError(
            "Network has no _ppc data; a power flow must be solved before "
            "the reachability check (net._ppc is missing)."
        )
    internal = net._ppc.get("internal")
    if internal is None:
        raise ValueError(
            "Network _ppc has no 'internal' block; the internal Newton-Raphson "
            "Jacobian is unavailable (was pp.runpp executed with the standard "
            "NR algorithm?)."
        )
    if "J" not in internal or internal["J"] is None:
        raise ValueError(
            "Internal Jacobian net._ppc['internal']['J'] is missing; cannot "
            "perform the voltage-stability reachability check."
        )
    J = internal["J"]
    J_dense = J.toarray() if sp.issparse(J) else np.asarray(J)
    J_dense = np.asarray(J_dense, dtype=np.float64)
    if not np.all(np.isfinite(J_dense)):
        raise ValueError(
            "Internal Jacobian contains non-finite entries (NaN/Inf); the "
            "upstream power flow did not produce a valid solution."
        )
    return J_dense


# ---------------------------------------------------------------------------
#  Per-step reachability check
# ---------------------------------------------------------------------------


def check_reachability(
    net: "pp.pandapowerNet",
    step_index: Optional[int] = None,
    tau_sigma: float = 1e-6,
    tau_eig: float = 1e-6,
    *,
    ensure_standard_structure: bool = True,
) -> ReachabilityResult:
    """Decide whether a *converged* power-flow solution lies on the stable
    (upper) voltage branch.

    Implements the singularity guard on the full Jacobian and the modal
    reduced-Q-V criterion described in the module docstring.  All-positive
    real eigenvalues of ``J_R`` (and a non-singular ``J``) imply the stable
    upper branch.

    Parameters
    ----------
    net
        Pandapower network with a converged Newton-Raphson power flow.
    step_index
        Optional time-series step index, stamped on the result.
    tau_sigma
        Proximity threshold on the smallest singular value of ``J``.  The
        point is rejected when ``sigma_min(J) < tau_sigma``.
    tau_eig
        Proximity threshold on the minimum real eigenvalue of ``J_R``.  The
        point is rejected when ``min(real(eig(J_R))) <= tau_eig``.
    ensure_standard_structure
        When ``True`` (default) and the internal Jacobian is *not* in the
        canonical ``n_pv + 2 n_pq`` structure (e.g. because the solve used
        ``distributed_slack=True``), a deep copy of the network is
        re-converged with ``distributed_slack=False, run_control=False`` to
        obtain the canonical single-slack Jacobian.  When ``False`` a
        non-canonical Jacobian raises immediately.

    Returns
    -------
    ReachabilityResult

    Raises
    ------
    ValueError
        On any missing/invalid input: no converged solution, missing internal
        Jacobian, non-finite Jacobian entries, an empty PQ set, an unexpected
        Jacobian shape, or a singular active-power sub-Jacobian.
    """
    if net is None:
        raise ValueError("check_reachability received net=None.")
    if not getattr(net, "converged", False):
        raise ValueError(
            "Power flow has not converged (net.converged is False); the "
            "reachability check requires a converged equilibrium."
        )

    work_net = net
    J = _extract_dense_jacobian(work_net)
    pq = np.asarray(work_net._ppc["internal"]["pq"], dtype=np.int64)
    pv = np.asarray(work_net._ppc["internal"]["pv"], dtype=np.int64)
    n_pq = int(pq.size)
    n_pv = int(pv.size)
    m_std = n_pv + 2 * n_pq

    # Detect and repair the distributed-slack augmentation by re-converging a
    # deep copy with single slack (see module docstring).  This is the same
    # canonicalisation already relied upon by JacobianSensitivities.
    if J.shape[0] != m_std:
        if not ensure_standard_structure:
            raise ValueError(
                f"Internal Jacobian has non-canonical shape {J.shape}; "
                f"expected ({m_std}, {m_std}) for n_pv={n_pv}, n_pq={n_pq}. "
                "This typically means the solve used distributed_slack=True. "
                "Re-converge with distributed_slack=False, run_control=False "
                "or call with ensure_standard_structure=True."
            )
        work_net = copy.deepcopy(net)
        try:
            # A flat (default) start is used deliberately rather than
            # init="results": when Newton-Raphson is warm-started exactly at
            # the solution it can converge at iteration 0 without ever forming
            # the Jacobian, so net._ppc["internal"]["J"] would be absent.  A
            # flat start always performs at least one iteration and therefore
            # always stores J.  The operating point is unchanged (single slack
            # only redistributes the slack power).
            pp.runpp(
                work_net,
                run_control=False,
                calculate_voltage_angles=True,
                distributed_slack=False,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise with context
            raise ValueError(
                "Failed to re-converge a single-slack copy for the canonical "
                f"power-flow Jacobian: {exc}"
            ) from exc
        J = _extract_dense_jacobian(work_net)
        pq = np.asarray(work_net._ppc["internal"]["pq"], dtype=np.int64)
        pv = np.asarray(work_net._ppc["internal"]["pv"], dtype=np.int64)
        n_pq = int(pq.size)
        n_pv = int(pv.size)
        m_std = n_pv + 2 * n_pq
        if J.shape[0] != m_std:
            raise ValueError(
                "Canonical single-slack re-convergence still yielded a "
                f"Jacobian of shape {J.shape} (expected ({m_std}, {m_std})); "
                "the internal NR structure is unexpected."
            )

    if n_pq == 0:
        raise ValueError(
            "Reduced Q-V Jacobian is undefined: the network has no PQ buses "
            "(empty pq set), so no voltage-stability mode can be formed."
        )

    # ── (1) Singularity guard on the full Jacobian ──────────────────────────
    sigma = sla.svd(J, compute_uv=False)
    sigma_min_J = float(sigma[-1])
    sigma_max_J = float(sigma[0])
    cond_J = float(sigma_max_J / sigma_min_J) if sigma_min_J > 0.0 else float("inf")

    # ── (2) Reduced Q-V Jacobian via the Schur complement ───────────────────
    # State / equation ordering of the canonical NR Jacobian:
    #   states:    [theta(pv + pq), V(pq)]
    #   equations: [P(pv + pq),     Q(pq)]
    n = n_pv + n_pq
    J_Ptheta = J[:n, :n]
    J_PV = J[:n, n:]
    J_Qtheta = J[n:, :n]
    J_QV = J[n:, n:]

    try:
        # J_R = J_QV - J_Qtheta @ J_Ptheta^{-1} @ J_PV
        schur_term = J_Qtheta @ sla.solve(J_Ptheta, J_PV)
    except (sla.LinAlgError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "Active-power sub-Jacobian (dP/dtheta) is singular; the operating "
            "point coincides with the saddle-node and the reduced Q-V "
            f"Jacobian cannot be formed: {exc}"
        ) from exc
    J_R = J_QV - schur_term

    eigvals, eigvecs = sla.eig(J_R)
    real_parts = eigvals.real
    mode = int(np.argmin(real_parts))
    lambda_min_JR = float(real_parts[mode])

    # Bus participation in the critical mode = squared right-eigenvector entries.
    participation = np.abs(eigvecs[:, mode]) ** 2
    crit_local = int(np.argmax(participation))
    crit_ppc_bus = int(pq[crit_local])
    critical_bus = _ppc_bus_to_pp_bus(work_net, crit_ppc_bus)

    on_stable_branch = (sigma_min_J >= tau_sigma) and (lambda_min_JR > tau_eig)

    return ReachabilityResult(
        on_stable_branch=on_stable_branch,
        sigma_min_J=sigma_min_J,
        cond_J=cond_J,
        lambda_min_JR=lambda_min_JR,
        critical_bus=critical_bus,
        step_index=step_index,
    )


# ---------------------------------------------------------------------------
#  Time-series wrapper
# ---------------------------------------------------------------------------


class ReachabilityMonitor:
    """Accumulate the per-step stability margin and abort on the first
    violation (fail-fast).

    The monitor records ``(sigma_min_J, cond_J, lambda_min_JR, critical_bus,
    on_stable_branch)`` for *every* step so the full margin trajectory is
    available even when no violation occurs, and raises
    :class:`ReachabilityViolation` at the first step whose equilibrium is not
    on the stable upper branch.

    Parameters
    ----------
    tau_sigma, tau_eig
        Proximity thresholds forwarded to :func:`check_reachability`.
    """

    def __init__(self, tau_sigma: float = 1e-6, tau_eig: float = 1e-6) -> None:
        self.tau_sigma = float(tau_sigma)
        self.tau_eig = float(tau_eig)
        self.records: list[dict] = []

    def check_step(
        self,
        net: "pp.pandapowerNet",
        step_index: Optional[int] = None,
        time_s: Optional[float] = None,
    ) -> ReachabilityResult:
        """Check one converged equilibrium, record its margin, and raise on
        the first stable-branch violation.

        Parameters
        ----------
        net
            Pandapower network with a converged power flow for this step.
        step_index, time_s
            Step index and simulation time, recorded with the margin and
            included in the violation message.

        Returns
        -------
        ReachabilityResult

        Raises
        ------
        ReachabilityViolation
            When the equilibrium is not on the stable upper branch.
        """
        result = check_reachability(
            net,
            step_index=step_index,
            tau_sigma=self.tau_sigma,
            tau_eig=self.tau_eig,
        )
        self.records.append(
            {
                "step_index": step_index,
                "time_s": time_s,
                "sigma_min_J": result.sigma_min_J,
                "cond_J": result.cond_J,
                "lambda_min_JR": result.lambda_min_JR,
                "critical_bus": result.critical_bus,
                "on_stable_branch": result.on_stable_branch,
            }
        )
        if not result.on_stable_branch:
            violation = ReachabilityViolation(
                f"Voltage-stability reachability violated at step "
                f"{step_index} (t={time_s} s): operating point is on/beyond "
                f"the nose (lower branch). "
                f"sigma_min(J)={result.sigma_min_J:.3e} "
                f"(tau_sigma={self.tau_sigma:.1e}), "
                f"lambda_min(J_R)={result.lambda_min_JR:.3e} "
                f"(tau_eig={self.tau_eig:.1e}), "
                f"critical bus={result.critical_bus}."
            )
            # Attach the violating result and the full margin trajectory (which
            # already includes this step, appended above) so callers can keep
            # the margin history even though the run aborts.
            violation.result = result
            violation.margins = list(self.records)
            raise violation
        return result

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the recorded margin trajectory as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.records)
