"""
sbx_h/need.py
===========
Violation indicator for SBX-H v6 (originally the plan-v2 §2.3 need flag).
Since v6 it drives ONLY the escalation indicator and reporting; the
request machinery it used to arm was removed with the deal layer.

Per area: the flag is set iff the area's own tracked voltage constraints
show a violation deeper than ``v_viol_threshold_pu`` persisting for at
least ``n_need`` consecutive OFO iterations; ``direction`` = +1 for
undervoltage, −1 for overvoltage.  Once latched, the flag clears only
below ``release_threshold_pu`` (hysteresis / preventive release).

If both bounds are violated simultaneously (pathological), the deeper
violation decides; an exact tie resolves deterministically to
undervoltage (the decision carries both depths).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-12 (v6 indicator; original need flag 2026-07-07)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sbx_h.config import SBXConfig
from sbx_h.fail import rep1


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



class NeedTracker:
    """Consecutive-violation tracker for one area (plan §2.3).

    Feed one measurement per OFO iteration via :meth:`update`; the
    tracker enforces consecutive iteration indices (a gap resets — a
    skipped evaluation cannot count as persistence).
    """

    def __init__(self, config: SBXConfig, area_id: int) -> None:
        self._threshold = config.v_viol_threshold_pu
        # v5 preventive release (hysteresis): once latched, the flag
        # clears only when the depth falls BELOW the release threshold
        # (≤ the set threshold; None = equal, the v4 behaviour).
        self._release = (config.release_threshold_pu
                         if config.release_threshold_pu is not None
                         else config.v_viol_threshold_pu)
        self._n_need = config.n_need
        self.area_id = int(area_id)
        self._consecutive = 0
        self._direction = 0
        self._latched = False
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
            self._latched = False
        self._last_iteration = int(iteration)

        under = lo - v          # > 0 where undervoltage
        over = v - hi           # > 0 where overvoltage
        depth_under = float(np.max(under))
        depth_over = float(np.max(over))

        # Direction candidate against the SET threshold (v4 rule).
        if max(depth_under, depth_over) <= self._threshold:
            cand = 0
        elif depth_under >= depth_over:
            # Exact tie resolves to import need (deterministic).
            cand = +1
        else:
            cand = -1

        # v5 hysteresis: a LATCHED flag persists while the depth in the
        # latched direction exceeds the release threshold; it clears
        # below the release threshold.  A direction FLIP (the candidate
        # is the opposite bound) falls through to the v4 restart.  With
        # release == set threshold (the default) this block reproduces
        # the v4 behaviour exactly.
        if self._latched and cand in (0, self._direction):
            depth_lat = depth_under if self._direction > 0 else depth_over
            if depth_lat > self._release:
                self._consecutive += 1
                arr = under if self._direction > 0 else over
                return NeedDecision(
                    iteration=int(iteration),
                    flag=True,
                    direction=self._direction,
                    depth_under_pu=max(depth_under, 0.0),
                    depth_over_pu=max(depth_over, 0.0),
                    worst_bus=int(bus_indices[int(np.argmax(arr))]),
                    consecutive=self._consecutive,
                )
            self._latched = False
            self._consecutive = 0
            self._direction = 0
            return NeedDecision(
                iteration=int(iteration),
                flag=False,
                direction=0,
                depth_under_pu=max(depth_under, 0.0),
                depth_over_pu=max(depth_over, 0.0),
                worst_bus=None,
                consecutive=0,
            )

        direction = cand
        if direction > 0:
            worst_bus: Optional[int] = int(bus_indices[int(np.argmax(under))])
        elif direction < 0:
            worst_bus = int(bus_indices[int(np.argmax(over))])
        else:
            worst_bus = None

        if direction != 0 and direction == self._direction:
            self._consecutive += 1
        elif direction != 0:
            self._direction = direction
            self._consecutive = 1
        else:
            self._direction = 0
            self._consecutive = 0

        flag = self._consecutive >= self._n_need
        self._latched = flag
        return NeedDecision(
            iteration=int(iteration),
            flag=flag,
            direction=self._direction,
            depth_under_pu=max(depth_under, 0.0),
            depth_over_pu=max(depth_over, 0.0),
            worst_bus=worst_bus,
            consecutive=self._consecutive,
        )
