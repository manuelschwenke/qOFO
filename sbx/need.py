"""
sbx/need.py
===========
Need flag for SBX (plan v2 §2.3) — deliberately simple.

Per area: the flag is set iff the area's own tracked voltage constraints
show a violation deeper than ``v_viol_threshold_pu`` persisting for at
least ``n_need`` consecutive OFO iterations.  Direction:

* undervoltage → request Q **import** (corridor-flow change towards the
  own area),
* overvoltage  → request Q **export**.

The violation is an area-wide condition; the per-corridor request SIGN in
corridor-flow space (positive = export from the reference end A) follows
from the area's end of that corridor:

    request_sign(corridor) = direction · (−1 if own end is A else +1)

with ``direction = +1`` for import-need (undervoltage) and ``−1`` for
export-need (overvoltage) — importing more means a more negative q_corr
when the area is end A, a more positive q_corr when it is end B.

If both bounds are violated simultaneously (pathological), the deeper
violation decides; an exact tie resolves to undervoltage/import
(deterministic, documented — not silent: the decision carries both
depths).

The relieving-sign SANITY ASSERT (plan §2.3: an assert, not a condition)
lives in :func:`assert_relieving_sign`: the local sensitivity of the
worst-violated bus voltage to the requested corridor-flow change must
have the relieving sign; ``rep1`` with diagnostics otherwise.  The
scheduler computes that scalar from its cached H and the corridor row.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sbx.config import SBXConfig
from sbx.fail import rep1


@dataclass(frozen=True)
class NeedDecision:
    """Outcome of one need-flag update (one area, one OFO iteration).

    ``direction`` ∈ {+1, −1, 0}: +1 = import need (undervoltage),
    −1 = export need (overvoltage), 0 = no violation this iteration.
    ``flag`` is True only once the violation has persisted for
    ``n_need`` consecutive iterations.
    """

    iteration: int
    flag: bool
    direction: int
    depth_under_pu: float
    depth_over_pu: float
    worst_bus: Optional[int]
    consecutive: int

    def request_sign(self, own_end: str) -> int:
        """Per-corridor request sign in corridor-flow space (§2.3).

        ``own_end`` ∈ {"a", "b"} — the area's end of the corridor.
        Raises unless the flag is set (a sign without a need is a
        protocol error, not a neutral value).
        """
        if own_end not in ("a", "b"):
            rep1("own_end must be 'a' or 'b'", own_end=own_end)
        if not self.flag or self.direction == 0:
            rep1("request_sign queried without a set need flag",
                 iteration=self.iteration, flag=self.flag,
                 direction=self.direction)
        return self.direction * (-1 if own_end == "a" else +1)


class NeedTracker:
    """Consecutive-violation tracker for one area (plan §2.3).

    Feed one measurement per OFO iteration via :meth:`update`; the
    tracker enforces consecutive iteration indices (a gap resets — a
    skipped evaluation cannot count as persistence).
    """

    def __init__(self, config: SBXConfig, area_id: int) -> None:
        self._threshold = config.v_viol_threshold_pu
        self._n_need = config.n_need
        self.area_id = int(area_id)
        self._consecutive = 0
        self._direction = 0
        self._last_iteration: Optional[int] = None

    def update(
        self,
        iteration: int,
        bus_indices: Sequence[int],
        v_meas_pu: Sequence[float],
        v_min_pu: Sequence[float],
        v_max_pu: Sequence[float],
    ) -> NeedDecision:
        """Evaluate the area's tracked voltage constraints at one iteration.

        All sequences are aligned per tracked bus.  Depths are the
        maximum bound violations (0 when inside bounds); the flag needs
        depth > threshold on the SAME direction for ``n_need``
        consecutive iterations — a direction change restarts the count.
        """
        n = len(bus_indices)
        if not (len(v_meas_pu) == len(v_min_pu) == len(v_max_pu) == n):
            rep1("need-flag inputs must align per tracked bus",
                 area=self.area_id, n_bus=n, n_v=len(v_meas_pu),
                 n_min=len(v_min_pu), n_max=len(v_max_pu))
        if n == 0:
            rep1("need flag needs at least one tracked voltage bus",
                 area=self.area_id)
        v = np.asarray(v_meas_pu, dtype=np.float64)
        lo = np.asarray(v_min_pu, dtype=np.float64)
        hi = np.asarray(v_max_pu, dtype=np.float64)
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(lo))
                and np.all(np.isfinite(hi))):
            rep1("need-flag inputs contain non-finite entries",
                 area=self.area_id, iteration=iteration)
        if self._last_iteration is not None and \
                iteration != self._last_iteration + 1:
            # A gap in the iteration sequence cannot count as
            # persistence; restart rather than raise so a controller
            # restart does not poison the tracker.
            self._consecutive = 0
            self._direction = 0
        self._last_iteration = int(iteration)

        under = lo - v          # > 0 where undervoltage
        over = v - hi           # > 0 where overvoltage
        depth_under = float(np.max(under))
        depth_over = float(np.max(over))

        if max(depth_under, depth_over) <= self._threshold:
            direction = 0
            worst_bus = None
        elif depth_under >= depth_over:
            # Exact tie resolves to import need (deterministic).
            direction = +1
            worst_bus = int(bus_indices[int(np.argmax(under))])
        else:
            direction = -1
            worst_bus = int(bus_indices[int(np.argmax(over))])

        if direction != 0 and direction == self._direction:
            self._consecutive += 1
        elif direction != 0:
            self._direction = direction
            self._consecutive = 1
        else:
            self._direction = 0
            self._consecutive = 0

        return NeedDecision(
            iteration=int(iteration),
            flag=(self._consecutive >= self._n_need),
            direction=self._direction,
            depth_under_pu=max(depth_under, 0.0),
            depth_over_pu=max(depth_over, 0.0),
            worst_bus=worst_bus,
            consecutive=self._consecutive,
        )


def assert_relieving_sign(
    decision: NeedDecision,
    dv_worst_per_dq_request: float,
) -> None:
    """Sanity assert (plan §2.3): the request must relieve the violation.

    ``dv_worst_per_dq_request`` — local-model sensitivity of the worst-
    violated bus voltage PER UNIT of the SIGNED requested corridor-flow
    change (the scheduler computes it from its cached H and the corridor
    row).  For an import need (undervoltage, ``direction = +1``) the
    request must RAISE that voltage (positive sensitivity); for an export
    need (``direction = −1``) it must LOWER it (negative sensitivity).
    Wrong sign is a model inconsistency → ``rep1`` with diagnostics, per
    the plan an assert and not a request condition.
    """
    if not decision.flag:
        rep1("relieving-sign assert queried without a set need flag",
             iteration=decision.iteration)
    if not math.isfinite(dv_worst_per_dq_request):
        rep1("relieving-sign sensitivity is non-finite",
             iteration=decision.iteration,
             dv_worst_per_dq_request=dv_worst_per_dq_request)
    if decision.direction * dv_worst_per_dq_request <= 0.0:
        rep1("local Jacobian column does not have the relieving sign for "
             "the violated bus (plan §2.3 sanity assert)",
             iteration=decision.iteration, direction=decision.direction,
             worst_bus=decision.worst_bus,
             depth_under_pu=decision.depth_under_pu,
             depth_over_pu=decision.depth_over_pu,
             dv_worst_per_dq_request=dv_worst_per_dq_request)
