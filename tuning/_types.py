"""
tuning/_types.py
================
Pure data classes shared across the tuning package.

No logic, no imports from the rest of ``tuning/``.  Kept separate so that
``parameters.py`` and ``ceilings.py`` can both depend on these types
without a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class BOParam:
    """One Bayesian-optimization decision variable.

    Attributes
    ----------
    name : str
        Field name on ``MultiTSOConfig`` (e.g. ``"g_w_der"``).
    log : bool
        Whether to sample in log-space.
    low : float
        Lower bound.  Used directly.
    high : float | str
        Upper bound.  The literal string ``"ceil"`` defers to the LMI
        ceiling for this parameter (looked up in the ceilings dict at
        sample time).
    fallback_high : float
        Fallback used if the ceiling lookup returns ``None``, a value
        below ``low``, or a non-finite value (e.g. when the LMI condition
        cannot be evaluated).
    """

    name: str
    log: bool
    low: float
    high: float | str
    fallback_high: float = 1e6


@dataclass(frozen=True)
class Ceilings:
    """Per-actuator-class LMI thresholds extracted from
    :class:`analysis.stability_analysis.MultiZoneStabilityResult`.

    .. warning::
       **The fields do not all point the same way.**  For every ``g_w_*`` the
       value is a stability *floor* -- the smallest weight at which the
       contraction certificate holds -- whereas for ``g_v`` it is a genuine
       *ceiling*.  See :attr:`DIRECTION`.

       Both are nevertheless used as BO *upper* bounds, which for the ``g_w_*``
       fields is deliberate: above the floor the loop is sufficient-but-sluggish,
       so the budget is spent below it, in the region where the certificate is
       silent but the controller is more responsive.  The consequence is that
       **every sampled point is non-certified**, and the empirical contraction
       ``rho_emp_p95 < 1`` is then the only stability evidence the procedure
       has.  See ``tuning/ceilings.py`` and ``docs/tuning/tuning_strategy.md``
       Sec. 1 (the conservatism gap).

    Values are ``np.inf`` when the condition cannot be evaluated (e.g.
    a zone has no actuators of that class, or the analytical bound is
    not implemented).
    """

    g_w_der: float
    g_w_pcc: float
    g_w_tso_oltc: float
    g_w_tso_shunt: float
    g_w_dso_der: float
    g_w_dso_oltc: float
    g_v: float
    notes: str = ""

    #: Which side of the stability boundary each field sits on.  ``"floor"``
    #: means "certificate holds at or above this value"; ``"ceiling"`` means
    #: "certificate holds at or below it".  Consult this before using any
    #: field as a search bound.
    DIRECTION: ClassVar[dict[str, str]] = {
        "g_w_der":       "floor",
        "g_w_pcc":       "floor",
        "g_w_tso_oltc":  "floor",
        "g_w_tso_shunt": "floor",
        "g_w_dso_der":   "floor",
        "g_w_dso_oltc":  "floor",
        "g_v":           "ceiling",
    }

    def as_dict(self) -> dict[str, float]:
        return {
            "g_w_der":       self.g_w_der,
            "g_w_pcc":       self.g_w_pcc,
            "g_w_tso_oltc":  self.g_w_tso_oltc,
            "g_w_tso_shunt": self.g_w_tso_shunt,
            "g_w_dso_der":   self.g_w_dso_der,
            "g_w_dso_oltc":  self.g_w_dso_oltc,
            "g_v":           self.g_v,
        }
