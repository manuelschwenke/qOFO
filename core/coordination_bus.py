"""
core/coordination_bus.py
========================
Horizontal TSO–TSO coordination bus and boundary-marginal signals for the
Boundary Marginal Exchange (BME) scheme — spec §3.4, §3.8, §3.9 and §4
(see ``docs/BME_STATUS.md``).

Symbol map (code ↔ spec)
------------------------
* ``MarginalSignal.mu``    ↔ μ_j(k) = dΦ_j/dv_b ∈ R^{|B|} (§3.4), in the
                             global boundary registry order, sparse by
                             exact zeros outside zone j's adjacent set.
* ``MarginalSignal.v_b_meas`` ↔ the v_b^meas(k) snapshot μ was computed
                             at (diagnostics only, §4).
* ``SwitchNotice.dv_b_pred`` ↔ H_{b,i}^d · Δu_i_d — the predicted boundary
                             voltage step of a committed discrete move
                             (§3.8.1; consumed in Phase 5).
* ``delay_steps``          ↔ d, the communication delay in control steps
                             (D4, default 1; d = 0 = same-step exchange,
                             required by the Phase 4 identity test).
* ``drop_probability``     ↔ the optional message-loss simulation (§3.9).
* ``MarginalReceiver``     ↔ the receiver-side first-order low-pass
                             μ_j^filt(k) = (1 − β)·μ_j^filt(k−1) + β·μ_j(k−d)
                             (§3.4, D3 β = 0.3) plus the §3.8 cold-start /
                             missing-signal policy.

Locality and scope (§3.9)
-------------------------
The bus carries ONLY μ vectors and switch notices — no models, no internal
measurements, no objectives. The SELF-marginal μ_i never touches the bus:
under Convention A it enters zone i's own price term undelayed and
unfiltered (BME_STATUS.md §0.2 revision); the receiver therefore sums
NEIGHBOUR marginals only (``mu_neighbour_sum``) and the controller adds
its own μ_i locally (Phase 4).

Explicit policies (§3.8 — documented, not silent defaults)
----------------------------------------------------------
* Cold start: for the first d steps after ``start_step`` no signal can
  have arrived; the receiver reports ``coordinated=False`` and logs one
  ``cold_start`` event per step. Exactly d steps, then warm.
* After warm-up, a MISSING expected signal raises (protocol violation)
  unless drop simulation is enabled on the bus, in which case the policy
  is hold-last-FILTERED-value, logged per occurrence (``hold_last``).
* If a sender's very first signal is dropped there is no filtered value
  to hold: that sender contributes exactly zero (a neutral price) until
  its first signal arrives, logged per occurrence (``extended_cold``).
* Filter initialisation: the first received signal initialises the
  filter state (equivalent to β = 1 for the first sample) — starting the
  recursion from zero would bias the price towards zero for ~1/β steps.

Determinism: drop decisions are drawn from a bus-owned
``numpy.random.default_rng(seed)`` AT PUBLISH TIME, one draw per
(message, receiver) in ascending receiver order — the pattern depends
only on the publish sequence, never on query order or repetition. A
non-zero drop probability without a seed raises.

Fail-fast throughout: unknown zones, wrong vector lengths, non-finite
entries, duplicate publishes, non-consecutive receiver steps and missing
expected signals (drops disabled) all raise with precise messages.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-02 (BME Phase 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _validated_vector(name: str, value, owner: str) -> NDArray[np.float64]:
    """1-D finite float array, copied and frozen against mutation."""
    arr = np.array(value, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(
            f"{owner}: {name} must be a 1-D vector, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{owner}: {name} contains non-finite entries — a corrupt "
            "signal must never enter the bus."
        )
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class MarginalSignal:
    """μ_zone(step) in global boundary registry order (§3.4, §4)."""

    zone_id: int
    step: int
    mu: NDArray[np.float64]
    v_b_meas: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mu",
            _validated_vector("mu", self.mu, f"MarginalSignal(zone {self.zone_id})"),
        )
        object.__setattr__(
            self, "v_b_meas",
            _validated_vector(
                "v_b_meas", self.v_b_meas,
                f"MarginalSignal(zone {self.zone_id})",
            ),
        )
        if self.mu.shape != self.v_b_meas.shape:
            raise ValueError(
                f"MarginalSignal(zone {self.zone_id}): mu {self.mu.shape} "
                f"and v_b_meas {self.v_b_meas.shape} must share the "
                "boundary registry length."
            )


@dataclass(frozen=True)
class SwitchNotice:
    """Horizontal feedforward notice of a committed discrete move
    (§3.8.1; emitted and consumed in Phase 5)."""

    zone_id: int
    step: int
    dv_b_pred: NDArray[np.float64]
    devices: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dv_b_pred",
            _validated_vector(
                "dv_b_pred", self.dv_b_pred,
                f"SwitchNotice(zone {self.zone_id})",
            ),
        )
        if not self.devices:
            raise ValueError(
                f"SwitchNotice(zone {self.zone_id}): devices must name at "
                "least one moved device."
            )
        if not all(isinstance(d, str) and d for d in self.devices):
            raise ValueError(
                f"SwitchNotice(zone {self.zone_id}): devices must be "
                "non-empty strings."
            )


@dataclass(frozen=True)
class CoordinationEvent:
    """Structured log entry (asserted by tests; §3.8 'documented policy,
    not a silent default').

    kind ∈ {'cold_start', 'hold_last', 'extended_cold', 'drop'};
    ``sender`` is None for receiver-global events (cold_start).
    """

    kind: str
    step: int
    receiver: int
    sender: Optional[int] = None
    channel: str = "marginal"


class CoordinationBus:
    """In-process pub/sub with integer step delay d and optional drop
    probability (spec §3.9, §4). No hidden global state: zones interact
    with this object and the plant only.

    Parameters
    ----------
    zone_ids :
        Participating zones (≥ 2).
    n_boundary :
        |B|, the boundary registry length every μ / dv_b vector must have.
    delay_steps :
        d ≥ 0 (D4: default 1; 0 = same-step, for the identity test).
    drop_probability :
        Per-(message, receiver) loss probability in [0, 1]; > 0 requires
        ``seed``.
    seed :
        Seed for the bus-owned RNG (determinism requirement, spec §8).
    """

    def __init__(
        self,
        zone_ids: Sequence[int],
        n_boundary: int,
        *,
        delay_steps: int = 1,
        drop_probability: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        zones = [int(z) for z in zone_ids]
        if len(zones) < 2:
            raise ValueError(
                f"CoordinationBus needs at least two zones, got {zones}"
            )
        if len(set(zones)) != len(zones):
            raise ValueError(f"duplicate zone ids: {zones}")
        if int(n_boundary) < 1:
            raise ValueError(f"n_boundary must be ≥ 1, got {n_boundary}")
        if int(delay_steps) < 0 or delay_steps != int(delay_steps):
            raise ValueError(
                f"delay_steps must be a non-negative integer, got "
                f"{delay_steps}"
            )
        if not (0.0 <= float(drop_probability) <= 1.0):
            raise ValueError(
                f"drop_probability must lie in [0, 1], got {drop_probability}"
            )
        if float(drop_probability) > 0.0 and seed is None:
            raise ValueError(
                "drop_probability > 0 requires an explicit seed — drop "
                "patterns must be reproducible (spec §8 determinism)."
            )
        self.zone_ids: List[int] = sorted(zones)
        self.n_boundary = int(n_boundary)
        self.delay_steps = int(delay_steps)
        self.drop_probability = float(drop_probability)
        self._rng = (
            np.random.default_rng(seed) if seed is not None else None
        )

        self._marginals: Dict[Tuple[int, int], MarginalSignal] = {}
        self._notices: Dict[Tuple[int, int], List[SwitchNotice]] = {}
        # (sender, receiver, step, channel) → dropped?  Drawn at publish.
        self._dropped: Dict[Tuple[int, int, int, str], bool] = {}
        self.drop_log: List[CoordinationEvent] = []

    # ------------------------------------------------------------------
    #  Publish side
    # ------------------------------------------------------------------

    def _require_zone(self, zone: int) -> None:
        if zone not in self.zone_ids:
            raise ValueError(
                f"zone {zone} is not registered on this bus "
                f"(zones: {self.zone_ids})"
            )

    def _draw_drops(self, sender: int, step: int, channel: str) -> None:
        """One Bernoulli draw per receiver, ascending receiver order —
        deterministic given the publish sequence."""
        for receiver in self.zone_ids:
            if receiver == sender:
                continue
            key = (sender, receiver, int(step), channel)
            if self.drop_probability <= 0.0:
                self._dropped[key] = False
                continue
            dropped = bool(self._rng.random() < self.drop_probability)
            self._dropped[key] = dropped
            if dropped:
                self.drop_log.append(CoordinationEvent(
                    kind="drop", step=int(step), receiver=receiver,
                    sender=sender, channel=channel,
                ))
                logger.info(
                    "CoordinationBus: dropped %s %s→%s published at "
                    "step %d", channel, sender, receiver, step,
                )

    def publish_marginal(self, signal: MarginalSignal) -> None:
        """Publish μ_zone(step). One signal per (zone, step)."""
        self._require_zone(signal.zone_id)
        if signal.mu.shape != (self.n_boundary,):
            raise ValueError(
                f"zone {signal.zone_id}, step {signal.step}: mu has shape "
                f"{signal.mu.shape}; the boundary registry length is "
                f"{self.n_boundary}."
            )
        key = (signal.zone_id, int(signal.step))
        if key in self._marginals:
            raise ValueError(
                f"zone {signal.zone_id} already published a marginal for "
                f"step {signal.step} — one signal per zone per step."
            )
        self._marginals[key] = signal
        self._draw_drops(signal.zone_id, signal.step, "marginal")

    def publish_notice(self, notice: SwitchNotice) -> None:
        """Publish a switch notice (multiple per step are allowed — one
        per committed discrete move)."""
        self._require_zone(notice.zone_id)
        if notice.dv_b_pred.shape != (self.n_boundary,):
            raise ValueError(
                f"zone {notice.zone_id}, step {notice.step}: dv_b_pred has "
                f"shape {notice.dv_b_pred.shape}; the boundary registry "
                f"length is {self.n_boundary}."
            )
        key = (notice.zone_id, int(notice.step))
        self._notices.setdefault(key, []).append(notice)
        self._draw_drops(notice.zone_id, notice.step, "notice")

    # ------------------------------------------------------------------
    #  Receive side (raw — the filter/policy lives in MarginalReceiver)
    # ------------------------------------------------------------------

    def marginals_visible(
        self, receiver_zone: int, step: int
    ) -> Dict[int, MarginalSignal]:
        """Signals visible to ``receiver_zone`` at ``step``: exactly the
        neighbours' publishes of step − d that were not dropped. A
        message published at k is visible at k + d and not earlier
        (spec §5 Phase 3)."""
        self._require_zone(receiver_zone)
        send_step = int(step) - self.delay_steps
        out: Dict[int, MarginalSignal] = {}
        for sender in self.zone_ids:
            if sender == receiver_zone:
                continue
            sig = self._marginals.get((sender, send_step))
            if sig is None:
                continue
            if self._dropped[(sender, receiver_zone, send_step, "marginal")]:
                continue
            out[sender] = sig
        return out

    def notices_visible(
        self, receiver_zone: int, step: int
    ) -> List[SwitchNotice]:
        """Switch notices visible to ``receiver_zone`` at ``step`` (same
        delay semantics; a dropped notice is LOST, not held — it is an
        event, not a state)."""
        self._require_zone(receiver_zone)
        send_step = int(step) - self.delay_steps
        out: List[SwitchNotice] = []
        for sender in self.zone_ids:
            if sender == receiver_zone:
                continue
            for notice in self._notices.get((sender, send_step), []):
                if self._dropped[
                    (sender, receiver_zone, send_step, "notice")
                ]:
                    continue
                out.append(notice)
        return out


@dataclass(frozen=True)
class ReceivedMarginals:
    """Result of one receiver step.

    ``coordinated`` is False exactly during the §3.8 cold-start window
    (the first d steps); ``mu_neighbour_sum`` is Σ_{j ≠ i} μ_j^filt in
    registry order (None while not coordinated). The SELF-marginal is
    NOT included — Convention A adds it locally, undelayed and
    unfiltered.
    """

    step: int
    coordinated: bool
    mu_neighbour_sum: Optional[NDArray[np.float64]]


class MarginalReceiver:
    """Receiver-side low-pass filter and §3.8 signal policy for one zone.

    Must be stepped once per control step, consecutively (a skipped step
    would silently corrupt the filter cadence — it raises instead).

    Parameters
    ----------
    zone_id :
        The receiving zone.
    bus :
        The coordination bus (provides d and the visible signals).
    beta :
        Filter constant β ∈ (0, 1] (D3, default 0.3); β = 1 disables
        smoothing (identity-test configuration).
    start_step :
        First control step of the run (cold-start window is
        [start_step, start_step + d)).
    expected_senders :
        Zones whose μ this receiver requires each warm step. Default:
        every other zone on the bus (H_{b,i} spans ALL of B, so every
        zone's marginal is relevant — sparsity is in the vectors, not
        the routing).
    """

    def __init__(
        self,
        zone_id: int,
        bus: CoordinationBus,
        *,
        beta: float = 0.3,
        start_step: int = 0,
        expected_senders: Optional[Sequence[int]] = None,
    ) -> None:
        if zone_id not in bus.zone_ids:
            raise ValueError(
                f"zone {zone_id} is not registered on the bus "
                f"(zones: {bus.zone_ids})"
            )
        if not (0.0 < float(beta) <= 1.0):
            raise ValueError(f"beta must lie in (0, 1], got {beta}")
        self.zone_id = int(zone_id)
        self._bus = bus
        self.beta = float(beta)
        self.start_step = int(start_step)
        if expected_senders is None:
            self.expected_senders: List[int] = [
                z for z in bus.zone_ids if z != self.zone_id
            ]
        else:
            senders = sorted(int(z) for z in expected_senders)
            unknown = [z for z in senders if z not in bus.zone_ids]
            if unknown:
                raise ValueError(
                    f"expected_senders contain unknown zones {unknown} "
                    f"(bus zones: {bus.zone_ids})"
                )
            if self.zone_id in senders:
                raise ValueError(
                    f"zone {self.zone_id} cannot expect its own marginal "
                    "from the bus — the self term is local (Convention A)."
                )
            if not senders:
                raise ValueError("expected_senders must not be empty")
            self.expected_senders = senders

        self._filt: Dict[int, NDArray[np.float64]] = {}
        self._last_step: Optional[int] = None
        self.events: List[CoordinationEvent] = []

    def mu_filtered(self, sender: int) -> Optional[NDArray[np.float64]]:
        """Current filter state for a sender (None before its first
        received signal) — diagnostics and tests."""
        state = self._filt.get(int(sender))
        return None if state is None else state.copy()

    def update(self, step: int) -> ReceivedMarginals:
        """Advance the receiver by one control step and return the
        filtered neighbour sum (§3.4) or the cold-start marker (§3.8)."""
        step = int(step)
        if self._last_step is None:
            if step != self.start_step:
                raise ValueError(
                    f"zone {self.zone_id}: first update must be at "
                    f"start_step {self.start_step}, got {step}."
                )
        elif step != self._last_step + 1:
            raise ValueError(
                f"zone {self.zone_id}: receiver must be stepped "
                f"consecutively (last {self._last_step}, got {step}) — "
                "a skipped step would corrupt the filter cadence."
            )
        self._last_step = step

        # ── Cold start: the first d steps after start_step (§3.8) ──
        if step < self.start_step + self._bus.delay_steps:
            self.events.append(CoordinationEvent(
                kind="cold_start", step=step, receiver=self.zone_id,
            ))
            logger.info(
                "zone %d step %d: cold start — running uncoordinated "
                "(no marginal can have arrived yet, d=%d).",
                self.zone_id, step, self._bus.delay_steps,
            )
            return ReceivedMarginals(
                step=step, coordinated=False, mu_neighbour_sum=None,
            )

        visible = self._bus.marginals_visible(self.zone_id, step)
        total = np.zeros(self._bus.n_boundary, dtype=np.float64)
        for sender in self.expected_senders:
            sig = visible.get(sender)
            if sig is not None:
                prev = self._filt.get(sender)
                if prev is None:
                    # First sample initialises the filter (β = 1 once).
                    self._filt[sender] = sig.mu.copy()
                else:
                    self._filt[sender] = (
                        (1.0 - self.beta) * prev + self.beta * sig.mu
                    )
            else:
                if self._bus.drop_probability <= 0.0:
                    raise RuntimeError(
                        f"zone {self.zone_id} step {step}: expected "
                        f"marginal from zone {sender} (published at step "
                        f"{step - self._bus.delay_steps}) is missing and "
                        "drop simulation is disabled — protocol violation "
                        "(spec §3.8)."
                    )
                if sender in self._filt:
                    self.events.append(CoordinationEvent(
                        kind="hold_last", step=step,
                        receiver=self.zone_id, sender=sender,
                    ))
                    logger.info(
                        "zone %d step %d: marginal from zone %d dropped "
                        "— holding last filtered value.",
                        self.zone_id, step, sender,
                    )
                else:
                    self.events.append(CoordinationEvent(
                        kind="extended_cold", step=step,
                        receiver=self.zone_id, sender=sender,
                    ))
                    logger.info(
                        "zone %d step %d: no marginal from zone %d has "
                        "ever arrived — its contribution stays zero.",
                        self.zone_id, step, sender,
                    )
            state = self._filt.get(sender)
            if state is not None:
                total += state

        return ReceivedMarginals(
            step=step, coordinated=True, mu_neighbour_sum=total,
        )
