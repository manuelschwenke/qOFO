"""
tuning/reparam.py
=================
Identifiable coordinates for the cascaded MIQP-OFO controller.

Why the raw weights cannot be searched directly
-----------------------------------------------
Two facts, both measured rather than assumed (2026-07-31):

1. **The weight space carries an exact redundant direction.**  The MIQP feasible
   set contains no weight, so scaling every objective weight, every ``g_w``,
   every ``g_z`` *and* the shunt integrator's gain by a common ``lambda``
   reproduces the trajectory to ~4e-10 — including the integer OLTC tap
   sequence.  Searching raw weights therefore spends budget moving along a
   direction that provably changes nothing.  Established group::

       {g_v, tso_g_q_pcc, g_q, dso_g_v} u {g_w_*} u {g_z_*} u {shunt_int_g_w}

2. **The box excluded the only known-good point.**  The hand-tuned
   configuration has ``g_v = 1e7`` against a box of ``[1e2, 1e5]`` and
   ``g_w_pcc = 80`` against a ceiling of ``30`` — unreachable by any trial.

Both are fixed by the same move: **fix the gauge at a reference operating point
and search dimensionless ratios about it.**  The reference is representable by
construction (it is the origin), and the redundant direction is quotiented out
rather than sampled.

Coordinates
-----------
Writing the curvature as in ``gw_precondition`` Eq. (3),
``M_sym = sum_i (1/g_w_i) a_i a_i^T`` with ``a_i`` the i-th column of
``D_y^{1/2} H_y``, the weights factor into

* a **gain** per layer — one scalar ``kappa`` placing ``lambda_max(M)`` on a
  target, and
* a **shape** — the ratios between actuator classes, which are what "prefer
  DER over PCC" actually means.

so the coordinates are ``(lambda per layer, per-class shape ratios, objective
priority ratios)``.  Each maps back to :class:`MultiTSOConfig` through
:func:`apply_reparam_to_config`.

Two measured facts shape the definitions:

* ``lambda`` is computed over the **continuous columns only**.  ``M`` treats the
  integer OLTC columns as continuous per-tick moves, but they step at most one
  tap per wall-clock cooldown, so their rank-1 term is an upper bound.  Including
  them makes TSO zone 1 read as ``integer_dominated`` (``lambda_floor = 1.085``)
  while its continuous loop sits at 0.021.
* The OLTC weights are **not** coordinates here.  They price integer switching,
  are monotone in switching rate, and are calibrated by 1-D bisection against an
  operational taps/day target — see :mod:`tuning.bisect_switching`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from configs.config import MultiTSOConfig
from tuning._types import BOParam

__all__ = [
    "BO_DIMS_V2",
    "Gauge",
    "PriorityScales",
    "apply_reparam_to_config",
    "coords_from_config",
    "reparam_search_space",
]


@dataclass(frozen=True)
class PriorityScales:
    """Engineering tolerances turning a weight into a dimensionless priority.

    A raw weight is not comparable across outputs measured in different units.
    Writing ``pi = g * sigma^2`` with ``sigma`` the tolerance on that output
    makes priorities commensurable, so "voltage matters more than interface-Q"
    becomes a statement about ``pi`` ratios rather than about raw magnitudes.

    Applied to the hand-tuned baseline this immediately shows something the raw
    weights hide: ``pi_q = g_q sigma_q^2 = 200 * 5^2 = 5000`` against
    ``pi_v_ds = dso_g_v sigma_v_ds^2 = 1e5 * 0.01^2 = 10``, i.e. the DSO weights
    interface-Q tracking ~500x above its voltage schedule.
    """

    sigma_v_ts: float = 0.005    # pu
    sigma_v_ds: float = 0.010    # pu
    sigma_q: float = 5.0         # Mvar


@dataclass(frozen=True)
class Gauge:
    """Reference operating point that fixes the gauge.

    Every coordinate is a ratio *relative to* these values, so the reference is
    the origin of the search space and is always representable — the defect that
    made the hand-tuned point unreachable cannot recur.

    Build with :meth:`from_config` from the baseline YAML, which should be the
    hand-tuned ``run_multi_system_ofo.make_config()``.
    """

    g_v: float
    g_q: float
    tso_g_q_pcc: float
    dso_g_v: float
    g_w_gen: float
    shunt_int_g_w: float
    scales: PriorityScales = dataclasses.field(default_factory=PriorityScales)

    @classmethod
    def from_config(
        cls,
        cfg: MultiTSOConfig,
        scales: PriorityScales | None = None,
    ) -> "Gauge":
        return cls(
            g_v=float(cfg.g_v),
            g_q=float(cfg.g_q),
            tso_g_q_pcc=float(getattr(cfg, "tso_g_q_pcc", 0.0) or 0.0),
            dso_g_v=float(cfg.dso_g_v),
            g_w_gen=float(cfg.g_w_gen),
            shunt_int_g_w=float(getattr(cfg, "shunt_int_g_w", 1.0) or 1.0),
            scales=scales or PriorityScales(),
        )


#: The reparameterised decision space.
#:
#: ``g_w_der`` / ``g_w_pcc`` / ``g_w_dso_der`` do **not** appear: they are
#: *outputs* of ``(lambda, tau)`` via the curvature preconditioner, and are
#: reported per class through ``PreconditionResult.class_scales`` so they stay
#: directly comparable with the hand-tuned values.
#:
#: ``g_w_gen`` does not appear either.  Measured 2026-07-31, the AVR contributes
#: 0.6-2.3 % of TSO curvature, so pinning it high (as specified) is correct and
#: tuning it would waste a dimension.
#: Half-width, in decades, of the multiplicative window around the reference
#: for every ratio coordinate.  Bounds are expressed *relative to the reference*
#: so the reference sits at the centre by construction — the search space cannot
#: fail to contain the operating point it is benchmarked against.  Defining
#: bounds as absolute round numbers is exactly how the previous space came to
#: exclude the hand-tuned point in two coordinates.
_WINDOW_DECADES: float = 1.5

BO_DIMS_V2: tuple[BOParam, ...] = (
    # Per-layer loop gain: target lambda_max(M) over the CONTINUOUS columns.
    # Uniform (not log) because the physically meaningful comparisons are
    # additive in lambda -- 0.9 is "well-damped", 2.0 is the hard OFO bound.
    BOParam("tso_lambda", log=False, low=0.05, high=1.20),
    BOParam("dso_lambda", log=False, low=0.05, high=1.20),

    # Shape: relative damping of DER vs PCC within the TSO block, gauge-fixed
    # to geometric mean 1 so it is orthogonal to the gain.  1.0 = the analytic
    # column-norm preconditioner; deviations express an actuator *preference*.
    BOParam("tau_der_pcc", log=True, low=1.0 / 64.0, high=64.0),

    # DSO objective trade-off: voltage schedule vs interface-Q tracking,
    # as a multiple of the reference ``dso_g_v``.  1.0 = the reference.
    BOParam("dso_v_priority", log=True,
            low=10.0 ** -_WINDOW_DECADES, high=10.0 ** _WINDOW_DECADES),
)

# Deliberately absent, with reasons:
#
#   g_v, g_q        the gauge.  Pinning them is what quotients out the exact
#                   scaling redundancy measured on 2026-07-31.
#   g_w_gen         the AVR contributes 0.6-2.3 % of TSO curvature, so tuning
#                   it would spend a dimension on nothing.  Stays pinned high.
#   tso_g_q_pcc     zero in the reference configuration, i.e. the TSO
#                   interface-Q objective is switched off.  A log coordinate
#                   cannot represent zero, and a coordinate whose reference sits
#                   on its own lower bound is the defect this module exists to
#                   avoid.  With it off, the TSO objective has a single active
#                   term whose scale *is* the gauge — so there is no TSO-side
#                   objective trade-off left to tune.  Re-add a relative
#                   coordinate here if a non-zero reference is ever adopted.
#   g_w_*_oltc      priced integer switching; monotone, with exactly-flat tails.
#                   Calibrated by 1-D bisection against an operational taps/day
#                   budget -- see :mod:`tuning.bisect_switching`.


def reparam_search_space() -> dict[str, tuple[float, float, bool]]:
    """``{name: (low, high, log)}`` for the reparameterised space."""
    return {
        p.name: (float(p.low), float(p.high), bool(p.log))
        for p in BO_DIMS_V2
    }


def apply_reparam_to_config(
    cfg: MultiTSOConfig,
    coords: dict[str, float],
    gauge: Gauge,
    *,
    fixed_overrides: dict[str, Any] | None = None,
) -> MultiTSOConfig:
    """Overlay reparameterised coordinates onto a config.

    ``g_v`` and ``g_q`` are held at their gauge values: they are the numeraire,
    not decision variables.  Only ratios move.

    ``lambda`` cannot be written into a scalar field — ``kappa`` depends on the
    cached sensitivity ``H`` — so it is routed through the runner's existing
    preconditioning hook via ``precondition_lambda_target_tso`` / ``_dso``,
    with ``precondition_mode='set'`` so the target binds in both directions.
    """
    expected = {p.name for p in BO_DIMS_V2}
    given = set(coords)
    if given - expected:
        raise ValueError(f"Unknown reparam coords: {sorted(given - expected)}")
    if expected - given:
        raise KeyError(f"Missing reparam coords: {sorted(expected - given)}")
    for k, v in coords.items():
        f = float(v)
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(f"Coordinate {k} must be finite and positive: {v}")

    tau = float(coords["tau_der_pcc"])

    overlay: dict[str, Any] = {
        # Gauge: pinned, never searched.
        "g_v": gauge.g_v,
        "g_q": gauge.g_q,
        "g_w_gen": gauge.g_w_gen,
        "tso_g_q_pcc": gauge.tso_g_q_pcc,

        # DSO objective trade-off, as a multiple of the reference.
        "dso_g_v": float(coords["dso_v_priority"]) * gauge.dso_g_v,

        # Loop gain + shape, applied by the preconditioner at controller init.
        "precondition_g_w": True,
        "precondition_mode": "set",
        # lambda refers to the columns actually being scaled.  Under the
        # default 'all' scope the integer OLTC columns -- whose rank-1 term is
        # an upper bound, not a real per-tick effect -- can block the target
        # outright (TSO zone 1: integer_dominated at 1.085 while its continuous
        # loop sits at 0.021), which would leave the coordinate inert.
        "precondition_lambda_scope": "preconditioned",
        "precondition_granularity": "column",
        "precondition_lambda_target_tso": float(coords["tso_lambda"]),
        "precondition_lambda_target_dso": float(coords["dso_lambda"]),
        # Gauge-fixed: geometric mean of the two factors is 1, so this moves
        # only the DER/PCC ratio and leaves the gain to lambda.
        "precondition_class_scales": {
            "der": math.sqrt(tau),
            "pcc": 1.0 / math.sqrt(tau),
        },
    }

    if fixed_overrides:
        overlay.update(fixed_overrides)
    return dataclasses.replace(cfg, **overlay)


def coords_from_config(cfg: MultiTSOConfig, gauge: Gauge) -> dict[str, float]:
    """Inverse of :func:`apply_reparam_to_config`, for warm-starting.

    ``tso_lambda`` / ``dso_lambda`` cannot be recovered from a config alone —
    they are properties of the cached ``H`` — so they fall back to the config's
    stated target, or the midpoint of the range.  The ratio coordinates are
    recovered exactly, and are 1.0 when ``cfg`` *is* the reference.
    """
    def _clamp(name: str, value: float) -> float:
        p = next(p for p in BO_DIMS_V2 if p.name == name)
        return float(min(max(value, float(p.low)), float(p.high)))

    lam_tso = getattr(cfg, "precondition_lambda_target_tso", None)
    lam_dso = getattr(cfg, "precondition_lambda_target_dso", None)
    shared = float(getattr(cfg, "precondition_lambda_target", 0.9) or 0.9)

    v_priority = (
        float(cfg.dso_g_v) / gauge.dso_g_v if gauge.dso_g_v > 0 else 1.0
    )

    return {
        "tso_lambda": _clamp("tso_lambda", float(lam_tso or shared)),
        "dso_lambda": _clamp("dso_lambda", float(lam_dso or shared)),
        "tau_der_pcc": 1.0,          # analytic preconditioner = no preference
        "dso_v_priority": _clamp("dso_v_priority", v_priority),
    }


def priority_report(cfg: MultiTSOConfig,
                    scales: PriorityScales | None = None) -> dict[str, float]:
    """Objective weights as dimensionless priorities ``pi = g * sigma^2``.

    Diagnostic only — the coordinates above are *relative* factors, which keeps
    the reference representable.  This view is for reading a configuration:
    raw weights across different output units are not comparable, and the pi
    form exposes imbalances the raw numbers hide.  At the hand-tuned baseline it
    shows ``pi_q / pi_v_ds ~ 500``, i.e. the DSO prioritises interface-Q
    tracking far above its voltage schedule — the opposite of what the raw
    ``g_q = 200`` vs ``dso_g_v = 1e5`` suggests at a glance.
    """
    s = scales or PriorityScales()
    return {
        "pi_v_ts":  float(cfg.g_v) * s.sigma_v_ts ** 2,
        "pi_q_tso": float(getattr(cfg, "tso_g_q_pcc", 0.0) or 0.0) * s.sigma_q ** 2,
        "pi_q_dso": float(cfg.g_q) * s.sigma_q ** 2,
        "pi_v_ds":  float(cfg.dso_g_v) * s.sigma_v_ds ** 2,
    }
