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
    """Per-actuator-class LMI ceilings extracted from
    :class:`analysis.stability_analysis.MultiZoneStabilityResult`.

    Each field is the smallest ``g_w`` (or, for ``g_v``, the largest
    weight) that satisfies the corresponding stability condition.  BO
    uses these as upper bounds: above the ceiling the system is
    sufficient-but-sluggish (for ``g_w_*``) or unstable per the
    certificate (for ``g_v``); the practically interesting search region
    lies below.

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
