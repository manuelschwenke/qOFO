"""Optional STS assistance for the slow switched-shunt dispatcher.

Only communicated measurements, received references and reported capability
bounds enter this module. All Q values use the same HV-port convention.
The supervisor has two explicit costs, separate from the normal TS objective:
tracking error outside a tolerance, then requested STS utilisation outside a
central capability band. Tracking assistance has priority over reserve support.

An assistance commit transfers the predicted bank contribution out of the
delegated STS request. Together with the usual shunt offset update, this holds
the total interface reference fixed. Reserve predictions assume that the STS
can subsequently track the reduced delegated request. Reported capability
intervals are necessary checks, not a certificate of network feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class STSShuntFeedback:
    interface_indices: NDArray[np.int64]
    q_measured: NDArray[np.float64]
    q_requested: NDArray[np.float64]
    q_min: NDArray[np.float64]
    q_max: NDArray[np.float64]
    reference_age_s: NDArray[np.float64]
    observed_at_s: float

    def __post_init__(self) -> None:
        names = ("q_measured", "q_requested", "q_min", "q_max", "reference_age_s")
        n = len(self.interface_indices)
        if n == 0 or len(set(self.interface_indices)) != n:
            raise ValueError("Feedback needs distinct interface indices")
        for name in names:
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (n,) or not np.all(np.isfinite(a)):
                raise ValueError(f"Invalid feedback {name}")
        if np.any(self.q_max < self.q_min) or np.any(self.reference_age_s < 0):
            raise ValueError("Invalid capability interval or reference age")
        if not math.isfinite(self.observed_at_s):
            raise ValueError("Feedback timestamp must be finite")


class STSFeedbackTracker:
    """Record reference ages while the faster STS loop produces feedback."""

    def __init__(self, change_tolerance_mvar: float = 1.0) -> None:
        if not math.isfinite(change_tolerance_mvar) or change_tolerance_mvar < 0:
            raise ValueError("change_tolerance_mvar must be finite and non-negative")
        self.change_tolerance_mvar = change_tolerance_mvar
        self._reference: Optional[NDArray[np.float64]] = None
        self._changed_at: Optional[NDArray[np.float64]] = None
        self._indices: Optional[NDArray[np.int64]] = None
        self._last_t: Optional[float] = None

    def observe(self, *, interface_indices, q_measured, q_requested,
                delta_min, delta_max, t_now: float) -> STSShuntFeedback:
        indices = np.asarray(interface_indices, dtype=np.int64).copy()
        measured = np.asarray(q_measured, dtype=float).copy()
        requested = np.asarray(q_requested, dtype=float).copy()
        if self._last_t is not None and t_now < self._last_t:
            raise ValueError("STS feedback times must be non-decreasing")
        if self._indices is not None and not np.array_equal(indices, self._indices):
            raise ValueError("STS interface ordering changed")
        if self._reference is None:
            reference = requested.copy()
            changed_at = np.full(len(indices), t_now, dtype=float)
        else:
            reference = self._reference.copy()
            changed_at = self._changed_at.copy()
            changed = np.abs(requested - reference) > self.change_tolerance_mvar
            reference[changed] = requested[changed]
            changed_at[changed] = t_now
        feedback = STSShuntFeedback(
            indices, measured, requested,
            measured + np.asarray(delta_min, dtype=float),
            measured + np.asarray(delta_max, dtype=float),
            t_now - changed_at, float(t_now),
        )
        self._indices, self._reference = indices, reference
        self._changed_at, self._last_t = changed_at, float(t_now)
        return feedback


@dataclass(frozen=True)
class STSShuntSupportConfig:
    tracking_weight: float = 1.0
    reserve_weight: float = 1.0
    tracking_deadband_mvar: float = 5.0
    tracking_scale_mvar: float = 25.0
    reserve_start: float = 0.8
    response_wait_s: float = 40.0
    max_feedback_age_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("tracking_weight", "reserve_weight", "tracking_deadband_mvar",
                     "response_wait_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.reserve_start) or not 0 < self.reserve_start < 1:
            raise ValueError("reserve_start must lie strictly between zero and one")
        for name in ("tracking_scale_mvar", "max_feedback_age_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class STSShuntRequest:
    kind: str
    gradient: float
    predicted_delta_q: NDArray[np.float64]
    cost_before: float
    cost_after: float


def _outside_band(value, width):
    return np.sign(value) * np.maximum(np.abs(value) - width, 0.0)


@dataclass(frozen=True)
class FleetMargins:
    """Directional reactive margins of the TS zone's own reactive resources.

    One entry per unit: the first ``n_gen`` are synchronous machines, whose
    reactive power is a TS *output*; the rest are TS-connected DER, whose
    reactive power is a TS *control*. The distinction matters for prediction,
    not for the margins themselves.

    Margins are **directional and in Mvar**, not a normalised position in a
    band. The normalised form conflates two different conditions: a unit with
    no capability at all (a VDE-AR-N 4120 park below ``P/S_n = 0.1``) is
    excluded as ineligible, whereas a unit with capability but no room left is
    what the reserve question is about. A margin of zero states the second
    unambiguously, and the asymmetry of the capability curve (-0.33 against
    +0.41 S_n) is carried rather than averaged away.
    """
    q_mvar: NDArray[np.float64]
    q_min: NDArray[np.float64]
    q_max: NDArray[np.float64]
    eligible: NDArray[np.bool_]
    n_gen: int

    def __post_init__(self) -> None:
        n = len(self.q_mvar)
        if not 0 <= self.n_gen <= n:
            raise ValueError("n_gen must index into the unit vector")
        for name in ("q_min", "q_max", "eligible"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Fleet margin {name} must have length {n}")
        for name in ("q_mvar", "q_min", "q_max"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name), dtype=float))):
                raise ValueError(f"Fleet margin {name} must be finite")
        ok = np.asarray(self.eligible, dtype=bool)
        if np.any(np.asarray(self.q_max)[ok] < np.asarray(self.q_min)[ok]):
            raise ValueError("Eligible units need q_max >= q_min")

    def directional(self, q_mvar=None):
        """Aggregate room to raise and to lower fleet reactive power [Mvar]."""
        q = np.asarray(self.q_mvar if q_mvar is None else q_mvar, dtype=float)
        if q.shape != self.q_mvar.shape:
            raise ValueError("Replacement reactive power must match the fleet")
        ok = np.asarray(self.eligible, dtype=bool)
        up = float(np.sum(np.maximum(np.asarray(self.q_max, dtype=float)[ok]-q[ok], 0.)))
        down = float(np.sum(np.maximum(q[ok]-np.asarray(self.q_min, dtype=float)[ok], 0.)))
        return up, down

    def substituted(self, q_gen=None, q_der=None):
        """Unit reactive power with machine and/or DER values replaced."""
        q = np.asarray(self.q_mvar, dtype=float).copy()
        if q_gen is not None:
            q[:self.n_gen] = np.asarray(q_gen, dtype=float)
        if q_der is not None:
            q[self.n_gen:] = np.asarray(q_der, dtype=float)
        return q


def fleet_margin_cost(margins: FleetMargins, *, weight: float,
                      margin_mvar: float, q_gen=None, q_der=None) -> float:
    """Shortfall of either directional margin below ``margin_mvar``.

        J = weight · [ s(R_up)² + s(R_down)² ] / margin_mvar²,
        s(R) = max(margin_mvar − R, 0)

    so the cost is zero while both directions keep at least ``margin_mvar`` of
    room, and the weight is dimensionless. The caller expresses
    ``margin_mvar`` as a multiple of the bank step, which gives the threshold
    the actuator's own scale instead of an arbitrary fraction of a band.

    ``q_gen`` and ``q_der`` substitute the machine and DER reactive power, which
    is how a complete candidate is evaluated: the machines from the predicted
    outputs, the DER from the candidate's chosen controls.
    """
    if margin_mvar <= 0:
        raise ValueError("margin_mvar must be positive")
    up, down = margins.directional(margins.substituted(q_gen, q_der))
    shortfall = max(margin_mvar-up, 0.)**2 + max(margin_mvar-down, 0.)**2
    return weight * shortfall / margin_mvar**2


def fleet_reserve_direction(margins: Optional[FleetMargins], *, margin_mvar: float,
                            injects: int, level: int, n_levels: int) -> int:
    """Bank direction that relieves the short side, or 0 if none applies.

    ``injects`` is +1 for a bank whose engagement injects reactive power (MSC)
    and −1 for one that absorbs it (MSR). A short ``up`` margin means the fleet
    is near its upper rail and the network must be supplied from elsewhere, so
    an injecting bank engages and an absorbing bank releases; a short ``down``
    margin is the mirror case. Checked against the 21 October capacitive
    scenario, where the machines sat at their absorbing rail and the action
    taken was an MSR engagement.

    Both sides short cannot be relieved by one bank, so no request is made.
    """
    if margins is None or margin_mvar <= 0:
        return 0
    if injects not in (-1, 1):
        raise ValueError("injects must be +1 (MSC) or -1 (MSR)")
    up, down = margins.directional()
    short_up, short_down = up < margin_mvar, down < margin_mvar
    if short_up == short_down:
        return 0
    direction = 1 if injects == (1 if short_up else -1) else -1
    return direction if 0 <= level + direction <= n_levels else 0


def support_request(feedback: Optional[STSShuntFeedback], *, h_q,
                    q_step_mvar: float, level: int, n_levels: int,
                    t_now: float, config: STSShuntSupportConfig,
                    current_q_requested=None,
                    ) -> Optional[STSShuntRequest]:
    """Return a directional request only when a full physical step helps.

    ``h_q`` covers every STS interface in feedback order, per positive Mvar of
    the bank's nameplate coordinate. The device sign is already in ``h_q``.
    A tracking cost uses the immediate interface response. The reserve cost
    uses the reduced STS request after the bank contribution is transferred.
    """
    if feedback is None:
        return None
    h = np.asarray(h_q, dtype=float)
    if h.shape != feedback.q_measured.shape or not np.all(np.isfinite(h)):
        raise ValueError("h_q must be finite and match STS interface ordering")
    if not math.isfinite(q_step_mvar) or q_step_mvar <= 0:
        raise ValueError("q_step_mvar must be finite and positive")
    if not math.isfinite(t_now) or not 0 <= level <= n_levels:
        raise ValueError("Invalid time or bank level")
    age = t_now - feedback.observed_at_s
    if (age < 0 or age > config.max_feedback_age_s
            or np.any(feedback.reference_age_s < config.response_wait_s)):
        return None

    error = feedback.q_requested - feedback.q_measured
    residual = _outside_band(error, config.tracking_deadband_mvar)
    if np.any(residual):
        # Reserve support never overrides an unresolved tracking error.
        weight = config.tracking_weight / config.tracking_scale_mvar**2
        gradient = -2.0 * weight * float(residual @ h)
        before = weight * float(residual @ residual)
        kind = "sts_tracking"
    else:
        half = (feedback.q_max - feedback.q_min) / 2.0
        mid = (feedback.q_max + feedback.q_min) / 2.0
        valid = half > 1e-6
        normalised = np.zeros_like(half)
        normalised[valid] = (feedback.q_requested[valid] - mid[valid]) / half[valid]
        excess = _outside_band(normalised, config.reserve_start)
        gradient = -2.0 * config.reserve_weight * float(
            excess[valid] @ (h[valid] / half[valid]))
        before = config.reserve_weight * float(excess @ excess)
        kind = "sts_reserve"
    if gradient == 0.0:
        return None
    direction = -1 if gradient > 0 else +1
    if not 0 <= level + direction <= n_levels:
        return None
    dq = h * (direction * q_step_mvar)
    if kind == "sts_tracking":
        after_residual = _outside_band(error - dq, config.tracking_deadband_mvar)
        after = weight * float(after_residual @ after_residual)
    else:
        # The reduced delegated target must remain inside the reported band.
        target = feedback.q_requested - dq
        if np.any(target < feedback.q_min) or np.any(target > feedback.q_max):
            return None
        after_normalised = np.zeros_like(half)
        after_normalised[valid] = (target[valid] - mid[valid]) / half[valid]
        after_excess = _outside_band(after_normalised, config.reserve_start)
        after = config.reserve_weight * float(after_excess @ after_excess)
    if after >= before - 1e-12:
        return None
    if current_q_requested is not None:
        # The TSO may have changed its command since this feedback was sampled.
        # Old unmet demand may accumulate only while a full step also helps the
        # current command. This check does not claim the STS has responded to it.
        current = support_request(
            replace(feedback, q_requested=np.asarray(current_q_requested, dtype=float)),
            h_q=h, q_step_mvar=q_step_mvar, level=level, n_levels=n_levels,
            t_now=t_now, config=config,
        )
        if (current is None or current.kind != kind
                or np.sign(current.gradient) != np.sign(gradient)):
            return None
    return STSShuntRequest(kind, gradient, dq.copy(), before, after)
