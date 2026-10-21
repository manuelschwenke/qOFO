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
class TSFleetReserveState:
    """Reactive-reserve state of the TS zone's own synchronous machines.

    Every field is a quantity the TS measures or derives itself, so no
    subordinate report and no assumption about a subordinate response enters
    this cost. ``q_mid`` and ``q_half`` must come from the same capability
    source as the controller's own reserve term, which
    ``TSOController.fleet_reserve_state`` guarantees.
    """
    q_mvar: NDArray[np.float64]
    q_mid: NDArray[np.float64]
    q_half: NDArray[np.float64]
    eligible: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n = len(self.q_mvar)
        for name in ("q_mid", "q_half", "eligible"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Fleet reserve {name} must have length {n}")
        for name in ("q_mvar", "q_mid", "q_half"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name), dtype=float))):
                raise ValueError(f"Fleet reserve {name} must be finite")
        if np.any(np.asarray(self.q_half)[np.asarray(self.eligible)] <= 0.0):
            raise ValueError("Eligible machines need a positive capability half-width")

    def normalised(self) -> NDArray[np.float64]:
        """Signed distance from the band centre, zero where not eligible."""
        out = np.zeros(len(self.q_mvar), dtype=float)
        ok = np.asarray(self.eligible, dtype=bool)
        out[ok] = ((np.asarray(self.q_mvar, dtype=float)[ok]
                    - np.asarray(self.q_mid, dtype=float)[ok])
                   / np.asarray(self.q_half, dtype=float)[ok])
        return out


def fleet_cost(state: TSFleetReserveState, q_gen, *, weight: float,
               start: float) -> float:
    """``J_fleet`` evaluated at an arbitrary machine reactive-power vector.

    Used to measure the relief of a *complete* candidate solution, in which the
    ordinary controls have been re-optimised around the forced bank step and may
    have cancelled part of the bank-only effect.
    """
    moved = replace(state, q_mvar=np.asarray(q_gen, dtype=float))
    excess = _outside_band(moved.normalised(), start)
    return weight * float(excess @ excess)


def ts_fleet_reserve_request(state: Optional[TSFleetReserveState], *, h_gen,
                             q_step_mvar: float, level: int, n_levels: int,
                             weight: float, start: float,
                             ) -> Optional[STSShuntRequest]:
    """Request a bank step when the TS's own machines are near their rails.

    ``h_gen`` is the machine reactive-power change per positive bank level, the
    ``Q_gen`` block of the bank's output column. The cost is

        J = weight · || D_start(r) ||²,   r = (Q − Q_mid)/Q_half,

    with ``D_start`` the componentwise outside-band operator, so the cost is
    zero while every machine stays inside the central part of its band and no
    centring pressure is exerted there. A whole physical step must reduce it.

    This is a *supervisory* cost: it is evaluated only for the bank decision and
    is deliberately absent from the shared TS objective, so the continuous
    actuators keep their voltage task. Admission of the resulting candidate is
    decided separately, against the shared objective.
    """
    if state is None or weight <= 0.0:
        return None
    h = np.asarray(h_gen, dtype=float)
    if h.shape != state.q_mvar.shape or not np.all(np.isfinite(h)):
        raise ValueError("h_gen must be finite and match the machine ordering")
    if not math.isfinite(q_step_mvar) or q_step_mvar <= 0:
        raise ValueError("q_step_mvar must be finite and positive")
    if not 0 <= level <= n_levels:
        raise ValueError("Invalid bank level")
    if not 0.0 < start < 1.0:
        raise ValueError("start must lie strictly between zero and one")

    ok = np.asarray(state.eligible, dtype=bool)
    if not np.any(ok):
        return None
    half = np.asarray(state.q_half, dtype=float)
    excess = _outside_band(state.normalised(), start)
    if not np.any(excess):
        return None
    gradient = 2.0 * weight * float(excess[ok] @ (h[ok] / half[ok]))
    if gradient == 0.0:
        return None
    direction = -1 if gradient > 0 else +1
    if not 0 <= level + direction <= n_levels:
        return None
    before = weight * float(excess @ excess)
    moved = np.asarray(state.q_mvar, dtype=float) + direction*h
    after_state = replace(state, q_mvar=moved)
    after_excess = _outside_band(after_state.normalised(), start)
    after = weight * float(after_excess @ after_excess)
    if after >= before - 1e-12:
        return None
    return STSShuntRequest("ts_reserve", gradient, (direction*h).copy(), before, after)


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
