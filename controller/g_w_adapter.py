"""
Adaptive g_w Module
===================

Online, sign-only adaptation of the per-variable input-change weight ``g_w``
following Zagorowska et al. (IFAC WC 2026, arXiv:2604.12863), Eq. 16
("diagonal formulation"), translated from the paper's scaling-matrix S to
this project's MIQP cost convention where ``g_w_i ∝ 1/S_i``.

Update rule (per variable ``i`` whose ``adapt_mask[i]`` is True):

    s_i = -grad_f[i] * w[i]            # sign estimate of ∂Φ_{k+1}/∂g_{w,i}

    if  s_i >  tol:  g_w[i] *= (1 - β1)   # descent regime  → shrink g_w (= grow S)
    if  s_i < -tol:  g_w[i] *= (1 + β2)   # anti-descent    → grow g_w   (= shrink S)
    else:            (no change)
    g_w[i] = clip(g_w[i], t_min, t_max)

The sign convention follows from the unconstrained one-step QP minimum
``w_i* = -grad_f_i / (2 g_w_i)`` and the linearisation
``Φ_{k+1} ≈ Φ_k + grad_f^T w_k``, which together give
``∂Φ_{k+1}/∂g_{w,i} ≈ -grad_f_i · w_i / g_{w,i}``.  Since ``g_{w,i} > 0``,
``sign(∂Φ/∂g_{w,i}) = sign(-grad_f_i · w_i) = sign(s_i)``.

Indices outside ``adapt_mask`` are left untouched and the corresponding
``t_min`` / ``t_max`` entries are unused, so they may be wider than the
non-adapting ``g_w_init`` value without raising.

References
----------
[1] Zagorowska, Ortmann, Belgioioso, Imsland.  "Adaptive Tuning of Online
    Feedback Optimization for Process Control Applications."
    IFAC World Congress 2026, arXiv:2604.12863, Eqs. (15)-(16).

Author: Manuel Schwenke
Date: 2026-04-29
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np
from numpy.typing import NDArray


__all__ = ["GwAdaptMeta", "GwAdapter", "make_adapter_from_class_indices"]


@dataclass(frozen=True)
class GwAdaptMeta:
    """Per-class meta-parameters for adaptive ``g_w``.

    Attributes
    ----------
    beta1
        Multiplicative shrink rate of ``g_w`` in the descent regime
        (``∂Φ/∂g_w > 0``).  Must be in ``[0, 1)``; 0 disables shrinking.
        Maps to paper Eq. (16) β₁ in the S-space convention (where it is
        the *grow* rate of S).
    beta2
        Multiplicative grow rate of ``g_w`` in the anti-descent regime
        (``∂Φ/∂g_w < 0``).  Must be ``≥ 0``.  Maps to paper Eq. (16) β₂.
    t_min
        Absolute floor on ``g_w_i`` after clipping.  Must satisfy
        ``t_min ≤ g_w_init ≤ t_max`` for adapted entries.
    t_max
        Absolute ceiling on ``g_w_i`` after clipping.
    deadband_rel
        Relative tolerance on ``|s_i|`` below which no update is applied,
        scaled by ``max(||grad_f|| · ||w||, 1.0)``.  Defaults to ``1e-6``.
    """
    beta1: float = 0.05
    beta2: float = 0.10
    t_min: float = 1e-2
    t_max: float = 1e6
    deadband_rel: float = 1e-6

    def __post_init__(self) -> None:
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(
                f"beta1 must be in [0, 1), got {self.beta1}"
            )
        if self.beta2 < 0.0:
            raise ValueError(f"beta2 must be >= 0, got {self.beta2}")
        if self.t_min <= 0.0:
            raise ValueError(
                f"t_min must be > 0 (g_w lives in (0, ∞)), got {self.t_min}"
            )
        if self.t_max < self.t_min:
            raise ValueError(
                f"t_max ({self.t_max}) must be >= t_min ({self.t_min})"
            )
        if self.deadband_rel < 0.0:
            raise ValueError(
                f"deadband_rel must be >= 0, got {self.deadband_rel}"
            )


class GwAdapter:
    """Online sign-rule adaptation of per-variable ``g_w`` (paper Eq. 16).

    Holds a mutable per-variable ``g_w_live`` vector and updates it each
    OFO step using the gradient ``grad_f`` and just-solved MIQP step
    ``w_k``.  Indices not in ``adapt_mask`` are never touched.

    The class is deliberately controller-agnostic: it operates on flat
    ``(n_total,)`` arrays and expects the caller to know which entries
    belong to which actuator class.  Use
    :func:`make_adapter_from_class_indices` to build an instance from
    per-class metadata.
    """

    def __init__(
        self,
        g_w_init: NDArray[np.float64],
        adapt_mask: NDArray[np.bool_],
        beta1: NDArray[np.float64],
        beta2: NDArray[np.float64],
        t_min: NDArray[np.float64],
        t_max: NDArray[np.float64],
        deadband_rel: float = 1e-6,
    ) -> None:
        n = int(g_w_init.shape[0])
        for arr, name in (
            (adapt_mask, "adapt_mask"),
            (beta1, "beta1"),
            (beta2, "beta2"),
            (t_min, "t_min"),
            (t_max, "t_max"),
        ):
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} shape {arr.shape} does not match "
                    f"g_w_init shape ({n},)"
                )

        if np.any((beta1 < 0.0) | (beta1 >= 1.0)):
            raise ValueError("beta1 entries must be in [0, 1)")
        if np.any(beta2 < 0.0):
            raise ValueError("beta2 entries must be >= 0")
        if np.any(t_min <= 0.0):
            raise ValueError("t_min entries must be > 0")
        if np.any(t_max < t_min):
            raise ValueError("t_max must be >= t_min element-wise")
        if deadband_rel < 0.0:
            raise ValueError("deadband_rel must be >= 0")

        # Only validate g_w_init bounds where adaptation is enabled — the
        # unused slots can carry the controller's static g_w value, which
        # may legitimately fall outside any per-class clip box.
        adapt_bool = adapt_mask.astype(bool)
        if np.any(adapt_bool):
            below = adapt_bool & (g_w_init < t_min - 1e-12)
            above = adapt_bool & (g_w_init > t_max + 1e-12)
            if np.any(below) or np.any(above):
                bad = np.flatnonzero(below | above)
                raise ValueError(
                    "g_w_init outside [t_min, t_max] at adapted indices "
                    f"{bad.tolist()}"
                )

        self._g_w_live: NDArray[np.float64] = g_w_init.astype(np.float64).copy()
        self._adapt_mask: NDArray[np.bool_] = adapt_bool.copy()
        self._beta1: NDArray[np.float64] = beta1.astype(np.float64).copy()
        self._beta2: NDArray[np.float64] = beta2.astype(np.float64).copy()
        self._t_min: NDArray[np.float64] = t_min.astype(np.float64).copy()
        self._t_max: NDArray[np.float64] = t_max.astype(np.float64).copy()
        self._deadband_rel = float(deadband_rel)
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def g_w_live(self) -> NDArray[np.float64]:
        """Read-only view of the current adapted ``g_w`` vector."""
        out = self._g_w_live.view()
        out.flags.writeable = False
        return out

    @property
    def adapt_mask(self) -> NDArray[np.bool_]:
        out = self._adapt_mask.view()
        out.flags.writeable = False
        return out

    @property
    def n_updates(self) -> int:
        return self._n_updates

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(
        self,
        grad_f: NDArray[np.float64],
        w: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply Eq. 16 sign rule using the current step's ``grad_f`` and
        ``w_k``.

        Parameters
        ----------
        grad_f
            OFO gradient ``G_k = H(u_k)^T ∇Φ^T(u_k, y_k)`` of shape
            ``(n_total,)`` in the same variable order as ``g_w_live``.
            This is the same ``grad_f`` already passed to
            :func:`build_miqp_problem` — the chain rule is assumed to
            have been applied upstream.
        w
            Just-solved MIQP step ``w_k`` of shape ``(n_total,)`` in
            original variable order (continuous block then integer
            block, in the controller's native layout).  The adapter
            uses only the sign of ``w``, so the α scaling on continuous
            entries is irrelevant.

        Returns
        -------
        NDArray[np.float64]
            Read-only view of the updated ``g_w_live``.
        """
        n = self._g_w_live.shape[0]
        if grad_f.shape != (n,):
            raise ValueError(
                f"grad_f shape {grad_f.shape} != ({n},)"
            )
        if w.shape != (n,):
            raise ValueError(f"w shape {w.shape} != ({n},)")

        s = -grad_f * w  # sign(∂Φ/∂g_w_i) for an unconstrained one-step QP

        scale = float(np.linalg.norm(grad_f) * np.linalg.norm(w))
        tol = self._deadband_rel * max(scale, 1.0)

        idx = self._adapt_mask
        shrink = idx & (s > tol)    # descent regime → shrink g_w
        grow = idx & (s < -tol)     # anti-descent  → grow g_w

        if np.any(shrink):
            self._g_w_live[shrink] *= (1.0 - self._beta1[shrink])
        if np.any(grow):
            self._g_w_live[grow] *= (1.0 + self._beta2[grow])

        np.clip(
            self._g_w_live,
            self._t_min,
            self._t_max,
            out=self._g_w_live,
        )

        self._n_updates += 1
        return self.g_w_live


def make_adapter_from_class_indices(
    g_w_init: NDArray[np.float64],
    class_indices: Mapping[str, NDArray[np.int64]],
    adapt_flags: Mapping[str, bool],
    metas: Union[GwAdaptMeta, Mapping[str, GwAdaptMeta]],
) -> GwAdapter:
    """Build a :class:`GwAdapter` from per-actuator-class metadata.

    Classes whose ``adapt_flags[name]`` is missing or ``False`` are not
    adapted; their indices retain the corresponding entries of
    ``g_w_init`` and are never modified by :meth:`GwAdapter.update`.

    Parameters
    ----------
    g_w_init
        Per-variable initial weights of shape ``(n_total,)``.  Used both
        as the initial ``g_w_live`` and as the (unused) values for
        non-adapted slots.
    class_indices
        Map ``actuator_class_name -> indices_into_g_w`` (1-D int arrays).
        Classes do not need to partition ``[0, n_total)``; only the
        adapted classes affect the result.  Overlapping classes raise.
    adapt_flags
        Map ``actuator_class_name -> bool``.  Missing keys default to
        ``False``.
    metas
        Either a single :class:`GwAdaptMeta` applied to all adapted
        classes, or a per-class mapping.  When a mapping, classes
        adapted but missing from the mapping fall back to ``GwAdaptMeta()``.
    """
    n = int(g_w_init.shape[0])
    adapt_mask = np.zeros(n, dtype=bool)
    beta1 = np.zeros(n, dtype=np.float64)
    beta2 = np.zeros(n, dtype=np.float64)
    # Sentinel-wide bounds for non-adapted entries so the constructor's
    # "g_w_init within [t_min, t_max]" check is vacuous for them.
    t_min = np.full(n, np.finfo(np.float64).tiny, dtype=np.float64)
    t_max = np.full(n, np.finfo(np.float64).max, dtype=np.float64)

    seen: NDArray[np.bool_] = np.zeros(n, dtype=bool)
    for cls, idx_arr in class_indices.items():
        if not adapt_flags.get(cls, False):
            continue
        idx_arr = np.asarray(idx_arr, dtype=np.int64)
        if np.any(seen[idx_arr]):
            overlap = idx_arr[seen[idx_arr]]
            raise ValueError(
                f"Class {cls!r} indices overlap an earlier adapted class "
                f"at positions {overlap.tolist()}"
            )
        seen[idx_arr] = True

        if isinstance(metas, GwAdaptMeta):
            m = metas
        else:
            m = metas.get(cls, GwAdaptMeta())

        adapt_mask[idx_arr] = True
        beta1[idx_arr] = m.beta1
        beta2[idx_arr] = m.beta2
        t_min[idx_arr] = m.t_min
        t_max[idx_arr] = m.t_max

    # ``deadband_rel`` is a scalar property of the adapter as a whole
    # (it gates updates against numerical noise in ``s_i = -grad·w``,
    # which has no per-class meaning).  When ``metas`` is a per-class
    # mapping we take the value from the first class — callers that
    # build the mapping from a single shared scalar (e.g.
    # :meth:`MultiTSOConfig.make_g_w_adapt_meta`) get exactly that
    # scalar back; mixed values across classes are not supported and
    # the first one wins by iteration order.
    if isinstance(metas, GwAdaptMeta):
        deadband_rel_used = metas.deadband_rel
    elif metas:
        deadband_rel_used = next(iter(metas.values())).deadband_rel
    else:
        deadband_rel_used = GwAdaptMeta().deadband_rel

    return GwAdapter(
        g_w_init=g_w_init.astype(np.float64),
        adapt_mask=adapt_mask,
        beta1=beta1,
        beta2=beta2,
        t_min=t_min,
        t_max=t_max,
        deadband_rel=deadband_rel_used,
    )
