"""
sbx_v/feasibility.py
===================
Machbarkeitsprüfung [LF §6.4] — the deterministic feasibility answer of
the delivering side (build plan §4, scheduling plane step 2).

The answer derives from the POSTED potential for the request's window
(the day-ahead ``PotentialMessage``, or the codified substitute) minus
the band, minus already-granted Vorhalteleistung, minus the
Reserve-Observer margin: accept / partial / reject — no negotiation
loop; a rejection leaves only the ungesichert best-effort tier
[LF §6.7].  ``all_or_nothing`` requests (the single v1 condition,
plan §3) turn a partial answer into a rejection.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 3)
"""

from __future__ import annotations

import math

from sbx_h.fail import rep1
from sbx_v.grants_ledger import GrantsLedger
from sbx_v.messages import (
    VERDICT_ACCEPT,
    VERDICT_PARTIAL,
    VERDICT_REJECT,
    FeasibilityReply,
    PotentialMessage,
    ReserveRequest,
)


def check_feasibility(
    request: ReserveRequest,
    potential: PotentialMessage,
    band_edge_mvar: float,
    ledger: GrantsLedger,
    *,
    reserve_margin_mvar: float = 0.0,
) -> FeasibilityReply:
    """Deterministic Machbarkeitsprüfung for one request [LF §6.4]."""
    if potential.aggregation_area_id != request.aggregation_area_id \
            or potential.direction is not request.direction \
            or potential.window.index != request.window.index:
        rep1("potential message does not match the request",
             request_id=request.request_id,
             potential_area=potential.aggregation_area_id,
             potential_direction=potential.direction.value,
             potential_window=potential.window.index)
    if not math.isfinite(band_edge_mvar) or band_edge_mvar < 0.0:
        rep1("band edge must be a non-negative magnitude",
             band_edge_mvar=band_edge_mvar)
    if reserve_margin_mvar < 0.0 or not math.isfinite(reserve_margin_mvar):
        rep1("reserve margin must be a non-negative magnitude",
             reserve_margin_mvar=reserve_margin_mvar)

    granted = ledger.granted_mvar(request.aggregation_area_id,
                                  request.direction,
                                  request.window.index)
    headroom = max(0.0, potential.q_pot_mvar - band_edge_mvar
                   - granted - reserve_margin_mvar)
    n_possible = int(math.floor(headroom / ledger.dq_grant_mvar + 1e-12))
    n_offered = min(request.n_quanta, n_possible)

    if n_offered <= 0:
        verdict, n_offered = VERDICT_REJECT, 0
    elif n_offered == request.n_quanta:
        verdict = VERDICT_ACCEPT
    elif request.all_or_nothing:
        # The single v1 request condition: partial coverage is refused
        # as a whole [plan §3].
        verdict, n_offered = VERDICT_REJECT, 0
    else:
        verdict = VERDICT_PARTIAL

    reply = FeasibilityReply(request_id=request.request_id,
                             verdict=verdict,
                             n_quanta_offered=n_offered)
    ledger.note_feasibility(request.aggregation_area_id,
                            request.direction, request.window.index,
                            reply)
    return reply
