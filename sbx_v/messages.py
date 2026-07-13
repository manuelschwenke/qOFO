"""
sbx_v/messages.py
================
SBX-V domain model (build plan §3) — the message dataclasses of the
vertical band-and-request pipeline.

All dataclasses are frozen; magnitudes are NON-NEGATIVE floats in Mvar
plus a :class:`sbx_v.directions.Direction` (hard rule 8); windows carry
both controller-step and wall-clock coordinates.  Identifiers are
DETERMINISTIC strings derived from (area, direction, window) — the
Phase-3 acceptance requires that identical scenarios replay to
byte-identical request/grant logs, so no randomness may enter an id.

Vocabulary mapping (plan §0):

* ``ReserveRequest``            ↔ unverbindliche Anforderung [LF §6.3]
* ``FeasibilityReply``          ↔ Machbarkeitsprüfung result [LF §6.4]
* ``BindingOrder``              ↔ verbindliche Anforderung [LF §6.6]
* ``GrantConfirmation``         ↔ Bestätigung [LF §6.7]
* ``PotentialMessage``          ↔ Potenzialmeldung [AR §6.3, §6.4.3]
* ``IncapabilityDeclaration``   ↔ Nichtverfügbarkeits-Meldung
                                  [AR §6.4.3; LF §6.9]

``EmergencyCall`` (Notfall-Abruf) is Phase 4 and deliberately absent.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from sbx_h.fail import rep1
from sbx_v.directions import Direction

#: Feasibility verdicts [LF §6.4]: full acceptance, partial quanta, or
#: rejection (leaving only the ungesichert best-effort tier, [LF §6.7]).
VERDICT_ACCEPT = "ACCEPT"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_REJECT = "REJECT"
_VERDICTS = (VERDICT_ACCEPT, VERDICT_PARTIAL, VERDICT_REJECT)


@dataclass(frozen=True)
class Window:
    """One SBX-V scheduling window (15 min), half-open ``[k_start,
    k_end)`` in TSO controller steps with the matching wall-clock span.

    ``index`` is the window counter used by the ledger, the metering
    registers and the settlement (one integer per 15-min window).
    """

    index: int
    k_start: int
    k_end: int
    t_start_s: float
    t_end_s: float

    def __post_init__(self) -> None:
        if self.index < 0:
            rep1("window index must be non-negative", index=self.index)
        if self.k_end <= self.k_start:
            rep1("window step range must be non-empty",
                 k_start=self.k_start, k_end=self.k_end)
        if not (math.isfinite(self.t_start_s)
                and math.isfinite(self.t_end_s)
                and self.t_end_s > self.t_start_s):
            rep1("window wall-clock span must be finite and non-empty",
                 t_start_s=self.t_start_s, t_end_s=self.t_end_s)


def _require_area(area_id: str) -> None:
    if not area_id:
        rep1("aggregation_area_id must be non-empty")


def _require_magnitude(name: str, value: float, *,
                       allow_zero: bool = True) -> None:
    if not math.isfinite(value) or value < 0.0 or \
            (value == 0.0 and not allow_zero):
        rep1(f"{name} must be a non-negative finite magnitude "
             f"(direction-separated, hard rule 8)", **{name: value})


@dataclass(frozen=True)
class PotentialMessage:
    """Potenzialmeldung — the CAIR capability posted per (area,
    direction, window) [AR §6.3.2, §6.4.3].

    ``q_pot_mvar`` is the total postable magnitude beyond zero in this
    direction (absolute, netted per AggregationArea, band included);
    ``q_vh_flagged_mvar`` marks the gesichert share reserved for active
    grants; ``is_forecast`` distinguishes the day-ahead plane.  Posting
    implies consent to Abruf (V-D4, [LF §6]).
    """

    aggregation_area_id: str
    direction: Direction
    q_pot_mvar: float
    q_vh_flagged_mvar: float
    window: Window
    is_forecast: bool = False
    is_substitute: bool = False
    """True when this message is the CODIFIED missing-message
    substitute (potential := 0 beyond the band) of hard rule 1
    [AR §6.3.2, Schritt 2] — constructed loudly by
    :func:`sbx_v.potentials.substitute_potential`, never silently."""

    def __post_init__(self) -> None:
        _require_area(self.aggregation_area_id)
        _require_magnitude("q_pot_mvar", self.q_pot_mvar)
        _require_magnitude("q_vh_flagged_mvar", self.q_vh_flagged_mvar)
        if self.q_vh_flagged_mvar > self.q_pot_mvar + 1e-9:
            rep1("gesichert share cannot exceed the posted potential",
                 area_id=self.aggregation_area_id,
                 q_pot_mvar=self.q_pot_mvar,
                 q_vh_flagged_mvar=self.q_vh_flagged_mvar)


@dataclass(frozen=True)
class ReserveRequest:
    """Unverbindliche Anforderung [LF §6.3] — non-binding, quantised.

    ``n_quanta`` × ``dq_grant_mvar`` is the requested Vorhalteleistung
    beyond the band for ``window``; ``all_or_nothing`` is the only
    supported condition in v1 (plan §3).
    """

    aggregation_area_id: str
    direction: Direction
    n_quanta: int
    window: Window
    all_or_nothing: bool = False

    def __post_init__(self) -> None:
        _require_area(self.aggregation_area_id)
        if self.n_quanta < 1:
            rep1("a request must ask for at least one quantum",
                 area_id=self.aggregation_area_id,
                 n_quanta=self.n_quanta)

    @property
    def request_id(self) -> str:
        """Deterministic identifier (replay-stable, plan Phase 3)."""
        return (f"req:{self.aggregation_area_id}:{self.direction.value}"
                f":w{self.window.index}")


@dataclass(frozen=True)
class FeasibilityReply:
    """Machbarkeitsprüfung result [LF §6.4] — deterministic, no
    negotiation loop; ``n_quanta_offered`` ≤ requested."""

    request_id: str
    verdict: str
    n_quanta_offered: int

    def __post_init__(self) -> None:
        if not self.request_id:
            rep1("request_id must be non-empty")
        if self.verdict not in _VERDICTS:
            rep1("verdict must be ACCEPT, PARTIAL or REJECT",
                 verdict=self.verdict)
        if self.verdict == VERDICT_REJECT and self.n_quanta_offered != 0:
            rep1("REJECT must offer zero quanta",
                 request_id=self.request_id,
                 n_quanta_offered=self.n_quanta_offered)
        if self.verdict != VERDICT_REJECT and self.n_quanta_offered < 1:
            rep1("ACCEPT/PARTIAL must offer at least one quantum",
                 request_id=self.request_id, verdict=self.verdict,
                 n_quanta_offered=self.n_quanta_offered)


@dataclass(frozen=True)
class BindingOrder:
    """Verbindliche Anforderung [LF §6.6]; ``n_quanta`` ≤ offered."""

    request_id: str
    n_quanta: int

    def __post_init__(self) -> None:
        if not self.request_id:
            rep1("request_id must be non-empty")
        if self.n_quanta < 1:
            rep1("a binding order must order at least one quantum",
                 request_id=self.request_id, n_quanta=self.n_quanta)

    @property
    def order_id(self) -> str:
        return f"ord:{self.request_id}"


@dataclass(frozen=True)
class GrantConfirmation:
    """Bestätigung [LF §6.7] — writes to the :class:`GrantsLedger`."""

    order_id: str
    aggregation_area_id: str
    direction: Direction
    n_quanta: int
    window: Window

    def __post_init__(self) -> None:
        if not self.order_id:
            rep1("order_id must be non-empty")
        _require_area(self.aggregation_area_id)
        if self.n_quanta < 1:
            rep1("a confirmation must confirm at least one quantum",
                 order_id=self.order_id, n_quanta=self.n_quanta)


@dataclass(frozen=True)
class IncapabilityDeclaration:
    """Real-time de-registration of gesichert potential [AR §6.4.3;
    LF §6.9] — consumed by settlement (Tabelle 8.1 case 3a via
    :class:`sbx_v.settlement.IncapabilityRecord`) and logged as a
    Reserve-Observer event."""

    aggregation_area_id: str
    direction: Direction
    q_shortfall_mvar: float
    window: Window

    def __post_init__(self) -> None:
        _require_area(self.aggregation_area_id)
        _require_magnitude("q_shortfall_mvar", self.q_shortfall_mvar,
                           allow_zero=False)
