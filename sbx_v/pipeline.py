"""
sbx_v/pipeline.py
================
The SBX-V request pipeline — request → Machbarkeitsprüfung → binding
order → confirmation (build plan §4, scheduling plane steps 1–3;
[LF §6.3–6.7]).

Deterministic by construction: no randomness, no wall clock — the event
log of two identical scenarios is byte-identical (Phase-3 acceptance).
One request per (area, direction, window); re-issue happens in LATER
windows [LF §6.9 spirit].  The pipeline is transport-free (in-memory
messages, plan §12): the same object acts for the requesting TSO and
the answering DSO side, but every step goes through the §3 message
dataclasses so the information accounting stays explicit.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 3)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from sbx_h.fail import rep1
from sbx_v.band import NormalBand
from sbx_v.directions import Direction
from sbx_v.feasibility import check_feasibility
from sbx_v.grants_ledger import GrantsLedger
from sbx_v.messages import (
    VERDICT_REJECT,
    BindingOrder,
    FeasibilityReply,
    GrantConfirmation,
    PotentialMessage,
    ReserveRequest,
    Window,
)
from sbx_v.need_flag import VerticalNeedDecision, size_request_quanta
from sbx_v.potentials import band_edge_mvar


class RequestPipeline:
    """Deterministic per-window state machine (plan §4 steps 1–3).

    ``forecast_provider(area_id, direction, window)`` returns the
    day-ahead :class:`PotentialMessage` for the window or ``None`` —
    ``None`` must already have been substituted by the caller
    (:func:`sbx_v.potentials.substitute_potential`); the pipeline
    fails fast on it (the substitute decision is the potentials
    plane's job, hard rule 1)."""

    def __init__(
        self,
        band_by_area: Dict[str, NormalBand],
        ledger: GrantsLedger,
        *,
        reserve_margin_mvar: float = 0.0,
        all_or_nothing: bool = False,
    ) -> None:
        if not band_by_area:
            rep1("pipeline needs at least one AggregationArea band")
        self._bands = dict(band_by_area)
        self.ledger = ledger
        self._margin = float(reserve_margin_mvar)
        self._all_or_nothing = bool(all_or_nothing)
        #: Replay-comparable event log: tuples of primitives only.
        self.log: List[Tuple] = []
        self._requested: set = set()

    # ------------------------------------------------------------------
    def run_window(
        self,
        window: Window,
        need_decisions: Dict[Tuple[str, Direction],
                             VerticalNeedDecision],
        forecast_provider: Callable[
            [str, Direction, Window], Optional[PotentialMessage]],
    ) -> List[GrantConfirmation]:
        """Steps 1–3 for one upcoming window.

        ``need_decisions`` are the CURRENT flags per (area, direction);
        a set flag emits one :class:`ReserveRequest` for ``window``
        (sized per plan §6), which is checked, ordered and confirmed in
        the same pass — the Leitfaden pipeline has no negotiation loop.
        """
        confirmations: List[GrantConfirmation] = []
        for (area_id, direction) in sorted(
                need_decisions.keys(),
                key=lambda k: (k[0], k[1].value)):
            decision = need_decisions[(area_id, direction)]
            if not decision.flag:
                continue
            if area_id not in self._bands:
                rep1("need decision for an unknown AggregationArea",
                     area_id=area_id, known=sorted(self._bands))
            key = (area_id, direction.value, window.index)
            if key in self._requested:
                # One request per (area, direction, window) — plan §6.
                continue
            band = self._bands[area_id]
            edge = band_edge_mvar(band, direction)

            forecast = forecast_provider(area_id, direction, window)
            if forecast is None:
                rep1("forecast_provider returned None — the missing-"
                     "message substitute must be applied by the "
                     "potentials plane (hard rule 1)",
                     area_id=area_id, direction=direction.value,
                     window_index=window.index)
            granted = self.ledger.granted_mvar(area_id, direction,
                                               window.index)
            n_quanta = size_request_quanta(
                decision.shortfall_mvar, self.ledger.dq_grant_mvar,
                forecast, edge, granted,
            )
            if n_quanta < 1:
                self.log.append(("no_headroom", window.index, area_id,
                                 direction.value,
                                 decision.shortfall_mvar,
                                 forecast.q_pot_mvar,
                                 forecast.is_substitute))
                continue

            request = ReserveRequest(
                aggregation_area_id=area_id,
                direction=direction,
                n_quanta=n_quanta,
                window=window,
                all_or_nothing=self._all_or_nothing,
            )
            self._requested.add(key)
            self.log.append(("request", window.index, area_id,
                             direction.value, n_quanta,
                             request.all_or_nothing))

            reply = check_feasibility(
                request, forecast, edge, self.ledger,
                reserve_margin_mvar=self._margin,
            )
            self.log.append(("reply", window.index, area_id,
                             direction.value, reply.verdict,
                             reply.n_quanta_offered))
            if reply.verdict == VERDICT_REJECT:
                continue

            order = BindingOrder(request_id=request.request_id,
                                 n_quanta=reply.n_quanta_offered)
            confirmation = GrantConfirmation(
                order_id=order.order_id,
                aggregation_area_id=area_id,
                direction=direction,
                n_quanta=order.n_quanta,
                window=window,
            )
            self.ledger.confirm(confirmation)
            confirmations.append(confirmation)
            self.log.append(("grant", window.index, area_id,
                             direction.value, order.n_quanta))

        self.ledger.assert_invariants(window.index)
        return confirmations
