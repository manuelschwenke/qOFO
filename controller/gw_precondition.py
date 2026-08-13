"""
Curvature-based g_w preconditioning (Tier-2 OFO gain selection)
===============================================================

This module turns the **manual** V5 curvature probe
(``experiments.runners.multi_tso_dso._dump_central_curvature``) into a
**reusable, automatic** rule for choosing the proximal weights ``g_w`` of
*any* OFO controller from its cached sensitivities — with no Bayesian
optimisation and no closed-loop simulation.

Background — the per-tick voltage-error map
-------------------------------------------
One unconstrained OFO tick (slack/usage/bounds dropped) is

    sigma* = -G_w^{-1} H_V^T diag(g_v) (V - V*),

so the voltage error evolves as ``e_{k+1} = (I - M) e_k`` with the
**closed-loop curvature**

    M = H_V G_w^{-1} H_V^T diag(g_v).                                  (1)

OFO is stable iff ``eig(M) ⊂ (0, 2)`` and well-damped for
``lambda_max(M) ≲ 1`` (see ``docs/daily_log/06_2026/2026-06-22_v5_central_tuning.md``).
Because (1) is similar to the symmetric PSD matrix

    M_sym = D_v^{1/2} H_V G_w^{-1} H_V^T D_v^{1/2},   D_v = diag(g_v),  (2)

its spectrum is real and non-negative; we compute eigenvalues from (2).

The two roles of g_w — why this is *not* a job for BO
-----------------------------------------------------
Writing ``A = D_v^{1/2} H_V`` (the *output-weighted* sensitivity), (2)
becomes a sum of rank-1 terms

    M_sym = Σ_i (1 / g_w_i) a_i a_i^T,     a_i = i-th column of A,     (3)

so column ``i`` contributes with weight ``||a_i||^2 / g_w_i``.  Two facts
follow, and they are the whole point of separating *what* the loop
optimises from *how fast* it gets there:

* **Conditioning (shape of M).**  Setting ``g_w_i ∝ ||a_i||^2`` equalises
  the per-actuator rank-1 contributions in (3).  This is exactly the
  diagonal scaling-matrix ``S`` of Zagorowska et al. (IFAC WC 2026,
  Eq. 16) in this project's convention ``g_w_i ∝ 1/S_i`` — a
  *preconditioner*, not a preference.

* **Gain (scale of M).**  Since ``M ∝ G_w^{-1}``, multiplying the whole
  ``g_w`` block by a scalar ``kappa`` scales ``lambda_max(M)`` by
  ``1/kappa``.  So a single ``kappa`` places ``lambda_max(M)`` at any
  target — the same one-scalar cooling that was hand-applied to V5
  (``KAPPA_V5 = 1.25``), here solved for automatically.

Public API
----------
* :func:`curvature_spectrum` — eigenvalues / ``lambda_max`` / ``cond`` of
  (2) for a given ``(H_V, g_v, g_w)``.  Shared by the V5 probe and the
  preconditioner so both read one implementation.
* :func:`precondition_g_w` — returns a new per-variable ``g_w`` vector
  obtained by (i) column-norm preconditioning the selected (continuous)
  actuator classes and (ii) solving for the global ``kappa`` that drives
  ``lambda_max(M)`` to a target.  Integer classes are left untouched by
  design (their cost is switching frequency, not curvature — a different
  tuning primitive; see ``docs/daily_log/06_2026/2026-06-22_shunt_integrator.md``).

Scope of this prototype
-----------------------
The curvature used here is the **voltage-tracking** block only (rows
``H_V``, weight ``g_v``), matching the validated V5 probe.  When the
interface-Q / reserve objective weights (``g_q``, ``g_res_*``) are
non-negligible the governing curvature is the *full* output-weighted
``M`` over all weighted rows; extending :func:`precondition_g_w` to a
general ``(H_y, g_y)`` is a drop-in change (the math in (3) is unchanged)
and is the documented next step.

References
----------
[1] Zagorowska, Ortmann, Belgioioso, Imsland. "Adaptive Tuning of Online
    Feedback Optimization for Process Control Applications." IFAC WC 2026,
    arXiv:2604.12863, Eqs. (15)-(16).
[2] docs/daily_log/06_2026/2026-06-22_v5_central_tuning.md (M, kappa derivation).

Author: Manuel Schwenke (with Claude Code)
Date: 2026-06-23
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "CurvatureSpectrum",
    "PreconditionResult",
    "curvature_spectrum",
    "precondition_g_w",
]


# ---------------------------------------------------------------------------
# Spectrum of the closed-loop curvature M
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurvatureSpectrum:
    """Eigen-summary of ``M = H_V G_w^{-1} H_V^T diag(g_v)``.

    Attributes
    ----------
    eigenvalues
        Ascending eigenvalues of the symmetric form (2) (real, ``>= 0``).
    lambda_max
        Largest eigenvalue — the OFO contraction driver.  Stable iff
        ``< 2``; well-damped if ``<~ 1``.
    lambda_min_pos
        Smallest eigenvalue above a relative null-space tolerance
        (directions the actuators *can* move).  ``0`` if none.
    cond
        ``lambda_max / lambda_min_pos`` (``inf`` if no positive mode) —
        a structural conditioning number of the actuator set; large
        values flag near-uncontrollable output directions that no ``g_w``
        choice can fix.
    """

    eigenvalues: NDArray[np.float64]
    lambda_max: float
    lambda_min_pos: float
    cond: float


def curvature_spectrum(
    H_v: NDArray[np.float64],
    g_v: NDArray[np.float64],
    g_w: NDArray[np.float64],
) -> CurvatureSpectrum:
    """Eigen-summary of the closed-loop curvature ``M`` (Eqs. 1-2).

    Parameters
    ----------
    H_v
        Voltage rows of the (expanded) sensitivity matrix, shape
        ``(n_v, n_u)`` — ``∂V/∂u`` in the controller's control ordering.
    g_v
        Per-voltage-bus tracking weights, shape ``(n_v,)``.  The bare
        ``g_v`` (the factor 2 of the gradient is absorbed by the QP's
        ``1/2``), matching :func:`controller's M definition`.
    g_w
        Per-variable proximal weights, shape ``(n_u,)``, strictly
        positive.

    Returns
    -------
    CurvatureSpectrum
    """
    H_v = np.asarray(H_v, dtype=np.float64)
    g_v = np.asarray(g_v, dtype=np.float64)
    g_w = np.asarray(g_w, dtype=np.float64)

    if H_v.ndim != 2:
        raise ValueError(f"H_v must be 2-D, got shape {H_v.shape}")
    n_v, n_u = H_v.shape
    if g_v.shape != (n_v,):
        raise ValueError(f"g_v shape {g_v.shape} != ({n_v},)")
    if g_w.shape != (n_u,):
        raise ValueError(f"g_w shape {g_w.shape} != ({n_u},)")
    if np.any(g_w <= 0.0):
        raise ValueError("g_w must be strictly positive")
    if np.any(g_v < 0.0):
        raise ValueError("g_v must be non-negative")

    if n_v == 0 or n_u == 0:
        empty = np.zeros(0, dtype=np.float64)
        return CurvatureSpectrum(empty, 0.0, 0.0, float("inf"))

    sqrt_gv = np.sqrt(g_v)
    HW = H_v * (1.0 / g_w)[np.newaxis, :]                  # H_V G_w^{-1}
    M_sym = (sqrt_gv[:, None] * HW) @ (H_v.T * sqrt_gv[None, :])
    # Symmetrise to kill floating-point asymmetry before eigvalsh.
    M_sym = 0.5 * (M_sym + M_sym.T)
    eig = np.linalg.eigvalsh(M_sym)
    eig = np.clip(eig, 0.0, None)                          # PSD: no negatives
    lam_max = float(eig[-1]) if eig.size else 0.0

    tol = 1e-12 * max(lam_max, 1.0)
    pos = eig[eig > tol]
    lam_min = float(pos.min()) if pos.size else 0.0
    cond = (lam_max / lam_min) if lam_min > 0.0 else float("inf")
    return CurvatureSpectrum(eig, lam_max, lam_min, cond)


# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreconditionResult:
    """Outcome of :func:`precondition_g_w`.

    Attributes
    ----------
    g_w_new
        New per-variable ``g_w`` vector (same length as the input).
        Entries of non-preconditioned (integer) classes are unchanged.
    lambda_max_before, lambda_max_after
        ``lambda_max(M)`` with the original and the new ``g_w``.
    lambda_floor
        ``lambda_max(M)`` contributed by the *fixed* (non-preconditioned)
        columns alone — the lower bound that scaling the preconditioned
        columns cannot beat.
    status
        Outcome.  Under ``mode='cap'`` (default) preconditioning may only
        *add* damping, never raise a loop's ``lambda_max``; under
        ``mode='set'`` it moves in both directions.

        * ``"reduced"`` — the loop was hotter than the target
          (``lambda_max_before > lambda_target > lambda_floor``); ``g_w``
          was reshaped and scaled down in gain to ``lambda_max_after ==
          lambda_target``.
        * ``"raised"`` — ``mode='set'`` only: the loop was *cooler* than the
          target and was scaled up to meet it.
        * ``"within_margin"`` — ``mode='cap'`` only: the loop was already at
          or below the target (``lambda_max_before <= lambda_target``); left
          at its config ``g_w`` (no-op), since acting could only make it more
          aggressive.
        * ``"integer_dominated"`` — the fixed (integer/OLTC) curvature
          already exceeds the target (``lambda_floor >= lambda_target``);
          no choice of continuous ``g_w`` helps, so left at config and
          flagged (the binding constraint is OLTC switching cadence, a
          Tier-2' concern, not ``g_w``).
        * ``"no_class"`` — no preconditionable class present.
    applied
        ``True`` when ``g_w`` actually changed, i.e. ``status`` is
        ``"reduced"`` or ``"raised"``.  (``"raised"`` only occurs in
        ``mode='set'``; ``mode='cap'`` never increases ``lambda_max``.)
    kappa
        Global scalar applied to the preconditioned columns so that
        ``lambda_max_after == lambda_target`` (when ``applied``); ``1.0``
        for every no-op outcome.
    class_scales
        Per preconditioned class: the mean assigned ``g_w`` (the
        interpretable per-class number, directly comparable to a
        BO-tuned ``g_w_<class>``).
    preconditioned_classes
        Names of the classes that were actually rescaled.
    spectrum_before, spectrum_after
        Full :class:`CurvatureSpectrum` before/after (for reporting
        ``cond``, etc.).
    """

    g_w_new: NDArray[np.float64]
    lambda_max_before: float
    lambda_max_after: float
    lambda_floor: float
    status: str
    applied: bool
    kappa: float
    class_scales: Dict[str, float]
    preconditioned_classes: Tuple[str, ...]
    spectrum_before: CurvatureSpectrum
    spectrum_after: CurvatureSpectrum


# Global clamp on the gain scalar so a degenerate problem can never
# inflate g_w to absurd values (= freezing the actuators) or collapse it
# to zero (= infinite gain).
_KAPPA_MIN = 1e-8
_KAPPA_MAX = 1e12


def _lambda_max_with(
    H_v: NDArray[np.float64],
    g_v: NDArray[np.float64],
    g_w: NDArray[np.float64],
    keep_cols: Optional[NDArray[np.int64]],
) -> float:
    """``lambda_max(M)`` optionally restricted to a subset of columns.

    When ``keep_cols`` is given, all other columns are removed from ``M``
    (``g_w -> inf`` ⇒ zero rank-1 contribution), so the result is the
    curvature produced by ``keep_cols`` alone.
    """
    if keep_cols is None:
        return curvature_spectrum(H_v, g_v, g_w).lambda_max
    g = np.full_like(g_w, np.inf)
    g[keep_cols] = g_w[keep_cols]
    return curvature_spectrum(H_v, g_v, g).lambda_max


def _solve_kappa(
    H_v: NDArray[np.float64],
    g_v: NDArray[np.float64],
    g_w_base: NDArray[np.float64],
    pre_idx: NDArray[np.int64],
    prov: NDArray[np.float64],
    lambda_target: float,
    keep_cols: Optional[NDArray[np.int64]],
    max_iter: int = 80,
) -> float:
    """Solve for the global scalar ``kappa`` on the preconditioned columns
    that makes ``lambda_max == lambda_target``.

    ``g_w(kappa)`` equals ``g_w_base`` everywhere except at ``pre_idx``,
    where it is ``prov * kappa``.  The matched ``lambda_max`` is taken over
    ``keep_cols`` (``None`` = all columns).  Because the preconditioned
    columns scale as ``M ∝ 1/kappa``, ``lambda_max`` is monotone decreasing
    in ``kappa``, so a log-space bisection converges robustly.  The result
    is clamped to ``[_KAPPA_MIN, _KAPPA_MAX]``.
    """
    def lam_at(kappa: float) -> float:
        g = g_w_base.copy()
        g[pre_idx] = prov[pre_idx] * kappa
        return _lambda_max_with(H_v, g_v, g, keep_cols)

    lam1 = lam_at(1.0)
    if lam1 <= 0.0 or not np.isfinite(lam1):
        return 1.0
    guess = float(np.clip(lam1 / lambda_target, _KAPPA_MIN, _KAPPA_MAX))

    # Bracket [lo, hi] with lam(lo) >= target >= lam(hi) (monotone ↓).
    lo = hi = guess
    for _ in range(max_iter):
        if lam_at(lo) >= lambda_target or lo <= _KAPPA_MIN:
            break
        lo = max(lo * 0.5, _KAPPA_MIN)
    for _ in range(max_iter):
        if lam_at(hi) <= lambda_target or hi >= _KAPPA_MAX:
            break
        hi = min(hi * 2.0, _KAPPA_MAX)

    # Bisection in log-space.
    for _ in range(max_iter):
        mid = float(np.sqrt(lo * hi))
        lam_mid = lam_at(mid)
        if lam_mid > lambda_target:
            lo = mid
        else:
            hi = mid
        if abs(lam_mid - lambda_target) <= 1e-6 * lambda_target:
            break
    return float(np.clip(np.sqrt(lo * hi), _KAPPA_MIN, _KAPPA_MAX))


def precondition_g_w(
    H_v: NDArray[np.float64],
    g_v: NDArray[np.float64],
    g_w_current: NDArray[np.float64],
    class_index_map: Mapping[str, NDArray[np.int64]],
    preconditionable_classes: Sequence[str],
    lambda_target: float = 0.9,
    granularity: str = "class",
    floor_frac: float = 1e-6,
    mode: str = "cap",
    class_scale_overrides: Optional[Mapping[str, float]] = None,
    lambda_scope: str = "all",
) -> PreconditionResult:
    """Derive a curvature-preconditioned ``g_w`` vector from cached
    sensitivities.

    Two-step rule (see module docstring):

    1. **Precondition (shape).**  For each class in
       ``preconditionable_classes`` set the provisional weight from the
       output-weighted column norm ``||a_i||^2`` (Eq. 3):

       * ``granularity='class'`` — one shared value per class
         (``mean_i ||a_i||^2``); keeps the interpretable per-class scalar
         structure and is directly A/B-comparable to BO's ``g_w_<class>``.
       * ``granularity='column'`` — per-variable ``||a_i||^2`` (full
         Zagorowska-``S`` diagonal); best conditioning.

    2. **Scale (gain).**  Solve one global ``kappa`` on the
       preconditioned columns so ``lambda_max(M) == lambda_target``.

    Integer / non-listed classes keep their incoming ``g_w_current``
    (their tuning primitive is switching frequency, not curvature).

    Parameters
    ----------
    H_v, g_v
        Voltage rows and weights (see :func:`curvature_spectrum`).
    g_w_current
        Incoming per-variable ``g_w`` (e.g. the BO/config diagonal),
        shape ``(n_u,)``.  Used verbatim for non-preconditioned classes.
    class_index_map
        ``{class_name: indices_into_u}`` (from the controller's
        ``_actuator_class_indices``).
    preconditionable_classes
        Which classes to rescale (typically the continuous ones).
        Names absent from ``class_index_map`` or empty are skipped.
    lambda_target
        Desired ``lambda_max(M)`` (``~0.9`` = well-damped, ``<2`` stable).
    granularity
        ``'class'`` (default) or ``'column'``.
    floor_frac
        Columns with ``||a_i||^2`` below ``floor_frac * max_j ||a_j||^2``
        are floored (near-uncontrollable directions) so their ``g_w``
        never collapses toward zero (which would mean infinite gain).
    mode
        ``'cap'`` (default) — the historical **cap-only** rule: act only when
        the loop is *hotter* than ``lambda_target``, never raise
        ``lambda_max``.  Correct for production, where preconditioning is a
        safety net that must not make a stable loop more aggressive.

        ``'set'`` — solve ``kappa`` in both directions so ``lambda_max``
        *equals* ``lambda_target`` whether the loop starts hotter or cooler.
        Required when ``lambda_target`` is a **tuning coordinate**: under
        ``'cap'`` every target above the current ``lambda_max`` collapses to
        the same no-op, so the coordinate would be flat over half its range
        and the optimiser could learn nothing from it.
    class_scale_overrides
        Optional ``{class_name: factor}`` multiplying that class's provisional
        weight before ``kappa`` is solved.  This is the *shape* knob: with
        ``kappa`` fixing the overall gain, the ratios between classes are the
        only remaining freedom, so a gauge-fixed set of factors (geometric mean
        1) spans exactly the actuator-preference directions.  Unknown class
        names are ignored; the analytic preconditioner corresponds to all
        factors equal to 1.
    lambda_scope
        Which columns ``lambda_target`` refers to.

        ``'all'`` (default) — the full ``lambda_max(M)``, i.e. the real
        worst-case contraction mode.  Correct for a production safety cap.

        ``'preconditioned'`` — only the columns actually being scaled.  ``M``
        treats the integer OLTC columns as continuous per-tick moves, but they
        step at most one tap per wall-clock cooldown, so their rank-1 term
        ``||a||^2 / g_w`` is an *upper bound* on their real effect rather than
        the effect itself.  Measured 2026-07-31 at the hand-tuned operating
        point, that bound alone puts TSO zone 1 at ``lambda_floor = 1.085``
        (and so ``integer_dominated``) while its continuous loop sits at 0.021
        — and the controller demonstrably works.  Use this scope whenever
        ``lambda_target`` is a tuning coordinate, so the target refers to
        something the tuned weights can actually move.

    Returns
    -------
    PreconditionResult
    """
    H_v = np.asarray(H_v, dtype=np.float64)
    g_v = np.asarray(g_v, dtype=np.float64)
    g_w_current = np.asarray(g_w_current, dtype=np.float64).copy()

    if granularity not in ("class", "column"):
        raise ValueError(
            f"granularity must be 'class' or 'column', got {granularity!r}"
        )
    if mode not in ("cap", "set"):
        raise ValueError(f"mode must be 'cap' or 'set', got {mode!r}")
    if lambda_scope not in ("all", "preconditioned"):
        raise ValueError(
            f"lambda_scope must be 'all' or 'preconditioned', "
            f"got {lambda_scope!r}"
        )
    if not (0.0 < lambda_target < 2.0):
        raise ValueError(
            f"lambda_target must be in (0, 2) for OFO stability, "
            f"got {lambda_target}"
        )
    n_v, n_u = H_v.shape
    if g_w_current.shape != (n_u,):
        raise ValueError(
            f"g_w_current shape {g_w_current.shape} != ({n_u},)"
        )

    spec_before = curvature_spectrum(H_v, g_v, g_w_current)

    # Output-weighted column energies ||a_i||^2 with a_i = D_v^{1/2} H_V[:,i].
    sqrt_gv = np.sqrt(g_v)
    A = sqrt_gv[:, None] * H_v
    col_sq = np.einsum("ij,ij->j", A, A)                  # (n_u,)
    col_sq_max = float(col_sq.max()) if col_sq.size else 1.0
    floor = floor_frac * col_sq_max if col_sq_max > 0.0 else floor_frac

    prov = g_w_current.copy()
    used: list[str] = []
    pre_idx_parts: list[NDArray[np.int64]] = []
    for cls in preconditionable_classes:
        idx = class_index_map.get(cls)
        if idx is None:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        c = col_sq[idx]
        if granularity == "class":
            rep = float(np.mean(c))
            prov[idx] = max(rep, floor)
        else:  # column
            prov[idx] = np.maximum(c, floor)
        # Shape override: with kappa fixing the overall gain, the per-class
        # factors are the only remaining freedom and span exactly the
        # actuator-preference directions.  Applied before kappa is solved so
        # the resulting lambda_max still lands on the target.
        if class_scale_overrides:
            factor = class_scale_overrides.get(cls)
            if factor is not None:
                factor = float(factor)
                if not (math.isfinite(factor) and factor > 0.0):
                    raise ValueError(
                        f"class_scale_overrides[{cls!r}] must be finite and "
                        f"positive, got {factor}"
                    )
                prov[idx] = prov[idx] * factor
        used.append(cls)
        pre_idx_parts.append(idx)

    def _result(g_w_new, kappa, status, lambda_floor):
        spec_after = curvature_spectrum(H_v, g_v, g_w_new)
        class_scales = {
            cls: float(np.mean(
                g_w_new[np.asarray(class_index_map[cls], dtype=np.int64)]
            ))
            for cls in used
        }
        return PreconditionResult(
            g_w_new=g_w_new,
            lambda_max_before=spec_before.lambda_max,
            lambda_max_after=spec_after.lambda_max,
            lambda_floor=float(lambda_floor),
            status=status,
            applied=(status in ("reduced", "raised")),
            kappa=float(kappa),
            class_scales=class_scales,
            preconditioned_classes=tuple(used),
            spectrum_before=spec_before,
            spectrum_after=spec_after,
        )

    if not used:
        # No preconditionable class present.
        return _result(g_w_current, 1.0, "no_class", spec_before.lambda_max)

    pre_idx = np.concatenate(pre_idx_parts)

    # Which columns the target refers to.  Under 'preconditioned' the fixed
    # columns are out of scope entirely, so there is no floor to be blocked by.
    keep_cols = pre_idx if lambda_scope == "preconditioned" else None

    # Curvature floor: what the FIXED (non-preconditioned) columns produce
    # on their own — the lower bound scaling the others cannot beat.
    g_inf = g_w_current.copy()
    g_inf[pre_idx] = np.inf
    lambda_floor = (
        0.0 if lambda_scope == "preconditioned"
        else curvature_spectrum(H_v, g_v, g_inf).lambda_max
    )

    # lambda as currently seen *in scope* — this is what the target is compared
    # against, and what the cap-only rule tests.
    lambda_in_scope = _lambda_max_with(H_v, g_v, g_w_current, keep_cols)

    # ── Cap-only rule: never raise a loop's lambda_max ──────────────────
    # Skipped in mode='set', where lambda_target is a tuning coordinate and
    # must bite in both directions -- otherwise every target above the current
    # lambda_max collapses to the same no-op and the coordinate is flat over
    # half its range.
    if mode == "cap" and lambda_in_scope <= lambda_target:
        # Already at/below the cap: acting could only make it hotter.
        # Leave at config (the conditioning gain is not worth the risk of
        # increasing the worst-case contraction mode).
        return _result(g_w_current, 1.0, "within_margin", lambda_floor)

    if lambda_floor >= lambda_target:
        # The fixed (integer/OLTC) curvature alone exceeds the target;
        # continuous g_w cannot pull lambda_max below the floor.  Binding
        # constraint is OLTC switching cadence (Tier-2'), not g_w.
        #
        # Caveat measured 2026-07-31: M treats the integer OLTC columns as
        # continuous per-tick moves, but they step at most one tap per
        # wall-clock cooldown, so their rank-1 term is an UPPER BOUND on the
        # real effect.  An `integer_dominated` verdict is therefore weak
        # evidence -- at the hand-tuned operating point TSO zone 1 reports
        # lambda_floor = 1.085 from the OLTC columns alone while its continuous
        # loop sits at 0.021, and the controller demonstrably works.  Prefer
        # passing a continuous-only class map when lambda is a tuning
        # coordinate.
        return _result(g_w_current, 1.0, "integer_dominated", lambda_floor)

    # Reachable: reshape + solve kappa so lambda_max(M) == lambda_target.  In
    # mode='cap' this only ever scales the gain DOWN (the branch above has
    # already returned otherwise); in mode='set' it moves in both directions.
    kappa = _solve_kappa(
        H_v, g_v, g_w_current, pre_idx, prov, lambda_target,
        keep_cols=keep_cols,
    )
    g_w_new = g_w_current.copy()
    g_w_new[pre_idx] = prov[pre_idx] * kappa
    status = "reduced" if lambda_in_scope > lambda_target else "raised"
    return _result(g_w_new, kappa, status, lambda_floor)
