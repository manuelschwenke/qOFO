"""
sbx_v/need_flag.py
=================
Deterministic vertical request trigger (build plan §6) — no shadow
prices, SBX-H v2 philosophy:

* **Condition A**: the netted PCC dispatch is saturated at the
  currently free-or-granted segment edge for that (area, direction)
  for ≥ ``t_persist_s``;
* **Condition B**: persistent deviation of the monitored transmission
  buses beyond the Sollspannungs band in the CORRESPONDING direction
  for ≥ ``t_persist_s`` [AR §5.3 vocabulary] — undervoltage calls for
  RAISING (the DS injects Q), overvoltage for LOWERING;
* flag = A ∧ B, with hysteresis on clearing (both conditions must stay
  clear for ``n_clear`` consecutive iterations before the flag drops).

Persistence is counted in TSO iterations, mirroring
``sbx_h.need.NeedTracker`` (consecutive counting; an iteration gap
resets); ``t_persist_s`` must be an exact multiple of ``tso_period_s``.

Request sizing (plan §6): the smallest ``n`` with
``n · dq_grant_mvar ≥ shortfall``, capped by the day-ahead posted
potential beyond band + existing grants.  The v1 shortfall estimate is
the caller-provided persistent estimate when available (e.g. the
unclipped MIQP desire) and one quantum otherwise — one request per
(area, direction, window); RE-ISSUE in later windows rather than
renegotiating [LF §6.9 spirit].

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from sbx_h.fail import rep1
from sbx_v.directions import Direction
from sbx_v.messages import PotentialMessage


@dataclass(frozen=True)
class VerticalNeedDecision:
    """Outcome of one iteration for one (area, direction)."""

    iteration: int
    flag: bool
    cond_a_count: int
    cond_b_count: int
    shortfall_mvar: float


class VerticalNeedTracker:
    """A ∧ B persistence tracker for ONE (area, direction) pair."""

    def __init__(
        self,
        area_id: str,
        direction: Direction,
        *,
        n_persist: int,
        n_clear: int,
        sat_tol_mvar: float,
        v_dev_threshold_pu: float,
    ) -> None:
        if n_persist < 1 or n_clear < 1:
            rep1("persistence counts must be positive",
                 n_persist=n_persist, n_clear=n_clear)
        if sat_tol_mvar < 0.0 or v_dev_threshold_pu <= 0.0:
            rep1("tolerances must be non-negative / positive",
                 sat_tol_mvar=sat_tol_mvar,
                 v_dev_threshold_pu=v_dev_threshold_pu)
        self.area_id = area_id
        self.direction = direction
        self._n_persist = int(n_persist)
        self._n_clear = int(n_clear)
        self._sat_tol = float(sat_tol_mvar)
        self._v_thr = float(v_dev_threshold_pu)
        self._a_count = 0
        self._b_count = 0
        self._clear_count = 0
        self._flag = False
        self._last_it: Optional[int] = None
        self._shortfall = 0.0

    def update(
        self,
        iteration: int,
        q_pcc_netted_mvar: float,
        segment_edge_mvar: float,
        v_dev_pu: float,
        *,
        shortfall_estimate_mvar: Optional[float] = None,
    ) -> VerticalNeedDecision:
        """One TSO iteration.

        ``q_pcc_netted_mvar`` — signed netted PCC flow (DP3 convention);
        ``segment_edge_mvar`` — the CURRENT free-or-granted magnitude
        edge of this direction (band + active grant);
        ``v_dev_pu`` — the worst transmission-bus deviation beyond the
        Sollspannungs band IN THIS DIRECTION (≥ 0; the caller maps
        under-/overvoltage to RAISING/LOWERING).
        """
        if not (math.isfinite(q_pcc_netted_mvar)
                and math.isfinite(segment_edge_mvar)
                and math.isfinite(v_dev_pu) and v_dev_pu >= 0.0
                and segment_edge_mvar >= 0.0):
            rep1("need-flag inputs invalid", area_id=self.area_id,
                 direction=self.direction.value,
                 q_pcc_netted_mvar=q_pcc_netted_mvar,
                 segment_edge_mvar=segment_edge_mvar, v_dev_pu=v_dev_pu)
        if self._last_it is not None and iteration != self._last_it + 1:
            # A gap cannot count as persistence (SBX-H pattern).
            self._a_count = self._b_count = 0
            self._clear_count = 0
            self._flag = False
        self._last_it = int(iteration)

        # Condition A: dispatch magnitude in THIS direction saturates
        # the free-or-granted edge.
        q_dir = self.direction.q_hv_sign * q_pcc_netted_mvar
        cond_a = q_dir >= segment_edge_mvar - self._sat_tol \
            and segment_edge_mvar > 0.0
        # Condition B: persistent voltage deviation in this direction.
        cond_b = v_dev_pu > self._v_thr

        self._a_count = self._a_count + 1 if cond_a else 0
        self._b_count = self._b_count + 1 if cond_b else 0

        if self._a_count >= self._n_persist \
                and self._b_count >= self._n_persist:
            self._flag = True
            self._clear_count = 0
            self._shortfall = (float(shortfall_estimate_mvar)
                               if shortfall_estimate_mvar is not None
                               else self._shortfall)
        elif self._flag:
            # Hysteresis on clearing: BOTH conditions must stay clear.
            if not cond_a and not cond_b:
                self._clear_count += 1
                if self._clear_count >= self._n_clear:
                    self._flag = False
                    self._shortfall = 0.0
            else:
                self._clear_count = 0
        if shortfall_estimate_mvar is not None:
            if not math.isfinite(shortfall_estimate_mvar) \
                    or shortfall_estimate_mvar < 0.0:
                rep1("shortfall estimate must be a non-negative "
                     "magnitude", area_id=self.area_id,
                     shortfall_estimate_mvar=shortfall_estimate_mvar)
            self._shortfall = float(shortfall_estimate_mvar)

        return VerticalNeedDecision(
            iteration=int(iteration), flag=self._flag,
            cond_a_count=self._a_count, cond_b_count=self._b_count,
            shortfall_mvar=self._shortfall,
        )


def size_request_quanta(
    shortfall_mvar: float,
    dq_grant_mvar: float,
    forecast_potential: PotentialMessage,
    band_edge_mvar: float,
    granted_mvar: float,
) -> int:
    """Plan §6 sizing: smallest covering ``n``, capped by the day-ahead
    posted potential beyond band + existing grants.  Returns 0 when the
    cap leaves no room (the request is then not emitted)."""
    if dq_grant_mvar <= 0.0:
        rep1("dq_grant_mvar must be positive",
             dq_grant_mvar=dq_grant_mvar)
    want = max(1, int(math.ceil(
        max(shortfall_mvar, 0.0) / dq_grant_mvar - 1e-12)) or 1)
    headroom = max(0.0, forecast_potential.q_pot_mvar
                   - band_edge_mvar - granted_mvar)
    cap = int(math.floor(headroom / dq_grant_mvar + 1e-12))
    return max(0, min(want, cap))
