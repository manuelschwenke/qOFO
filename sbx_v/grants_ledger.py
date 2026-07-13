"""
sbx_v/grants_ledger.py
=====================
Active Vorhalteleistung grants per (AggregationArea, direction, window)
— build plan §3.

Invariants (asserted, plan §3): grants never overlap in
(area, direction, window) beyond SUMMATION of quanta; the total granted
per window never exceeds the last accepted feasibility answer for that
window; grants activate and expire only at window boundaries (they are
defined as half-open window ranges, so this holds by construction and
is asserted at query time).

The ledger is the single source of the ``grant(a, d, window)`` bound of
the MIQP cost layer (plan §5) and exports
:class:`sbx_v.settlement.GrantRecord`s for the settlement plane.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sbx_h.fail import rep1
from sbx_v.directions import Direction
from sbx_v.messages import FeasibilityReply, GrantConfirmation
from sbx_v.settlement import DSO_DELIVERS, TSO_DELIVERS, GrantRecord


@dataclass(frozen=True)
class LedgerEntry:
    """One confirmed grant: ``q_grant_mvar`` beyond the band over the
    half-open window range ``[window_first, window_end)``."""

    order_id: str
    aggregation_area_id: str
    direction: Direction
    q_grant_mvar: float
    window_first: int
    window_end: int

    def __post_init__(self) -> None:
        if self.q_grant_mvar <= 0.0 or not math.isfinite(self.q_grant_mvar):
            rep1("ledger entry magnitude must be positive",
                 order_id=self.order_id, q_grant_mvar=self.q_grant_mvar)
        if self.window_end <= self.window_first:
            rep1("ledger entry window range must be non-empty",
                 order_id=self.order_id,
                 window_first=self.window_first,
                 window_end=self.window_end)


class GrantsLedger:
    """Confirmed grants with the plan-§3 invariants."""

    def __init__(self, dq_grant_mvar: float) -> None:
        if dq_grant_mvar <= 0.0 or not math.isfinite(dq_grant_mvar):
            rep1("dq_grant_mvar must be positive",
                 dq_grant_mvar=dq_grant_mvar)
        self.dq_grant_mvar = float(dq_grant_mvar)
        self._entries: List[LedgerEntry] = []
        #: Last ACCEPTED feasibility headroom [quanta] per
        #: (area, direction, window_index) — the confirmation cap.
        self._feasibility_quanta: Dict[Tuple[str, str, int], int] = {}

    # ------------------------------------------------------------------
    def note_feasibility(
        self,
        area_id: str,
        direction: Direction,
        window_index: int,
        reply: FeasibilityReply,
    ) -> None:
        """Record the feasibility answer governing later confirmations."""
        self._feasibility_quanta[(area_id, direction.value,
                                  window_index)] = reply.n_quanta_offered

    def confirm(self, confirmation: GrantConfirmation) -> LedgerEntry:
        """Write one :class:`GrantConfirmation` (plan §3: total granted
        per window must stay within the last accepted feasibility)."""
        area = confirmation.aggregation_area_id
        d = confirmation.direction
        w = confirmation.window
        q_new = confirmation.n_quanta * self.dq_grant_mvar
        key = (area, d.value, w.index)
        if key not in self._feasibility_quanta:
            rep1("confirmation without a recorded feasibility answer",
                 order_id=confirmation.order_id, area_id=area,
                 direction=d.value, window_index=w.index)
        offered = self._feasibility_quanta[key]
        already = self.granted_mvar(area, d, w.index)
        if already + q_new > offered * self.dq_grant_mvar + 1e-9:
            rep1("total granted quanta would exceed the last accepted "
                 "feasibility answer (plan §3 invariant)",
                 order_id=confirmation.order_id, area_id=area,
                 direction=d.value, window_index=w.index,
                 already_mvar=already, new_mvar=q_new,
                 offered_quanta=offered)
        entry = LedgerEntry(
            order_id=confirmation.order_id,
            aggregation_area_id=area,
            direction=d,
            q_grant_mvar=q_new,
            window_first=w.index,
            window_end=w.index + 1,
        )
        self._entries.append(entry)
        return entry

    # ------------------------------------------------------------------
    def granted_mvar(self, area_id: str, direction: Direction,
                     window_index: int) -> float:
        """Active granted magnitude beyond the band (summed quanta)."""
        return sum(
            e.q_grant_mvar for e in self._entries
            if e.aggregation_area_id == area_id
            and e.direction is direction
            and e.window_first <= window_index < e.window_end
        )

    def entries(self) -> Tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def assert_invariants(self, window_index: int) -> None:
        """Plan-§3 window invariant sweep (call once per boundary)."""
        seen: Dict[Tuple[str, str], float] = {}
        for e in self._entries:
            if not (e.window_first <= window_index < e.window_end):
                continue
            k = (e.aggregation_area_id, e.direction.value)
            seen[k] = seen.get(k, 0.0) + e.q_grant_mvar
        for (area, d), total in seen.items():
            key = (area, d, window_index)
            offered = self._feasibility_quanta.get(key)
            if offered is None:
                rep1("active grant without a feasibility record",
                     area_id=area, direction=d,
                     window_index=window_index)
            if total > offered * self.dq_grant_mvar + 1e-9:
                rep1("active grants exceed the accepted feasibility",
                     area_id=area, direction=d,
                     window_index=window_index, total_mvar=total,
                     offered_quanta=offered)

    # ------------------------------------------------------------------
    def to_grant_records(
        self, *, delivering_party: str = DSO_DELIVERS
    ) -> Tuple[GrantRecord, ...]:
        """Settlement export (v1: the DSO is the delivering party —
        the pipeline's requester is the TSO, plan §6)."""
        if delivering_party not in (DSO_DELIVERS, TSO_DELIVERS):
            rep1("delivering_party must be dso_delivers or tso_delivers",
                 delivering_party=delivering_party)
        return tuple(
            GrantRecord(
                area_id=e.aggregation_area_id,
                direction=e.direction,
                q_grant_mvar=e.q_grant_mvar,
                delivering_party=delivering_party,
                window_first=e.window_first,
                window_end=e.window_end,
            )
            for e in self._entries
        )
