"""
sbx_v/emergency.py
=================
Optional Notfall-Abruf (plan §3/§9 Phase 4; [AR §6.4.6]) — strictly
flag-gated and event-scripted only.

An :class:`EmergencyCall` is the ONE sanctioned exception to the
commit-instant rule: it activates an immediate best-effort extension of
the priced segment structure (at the Grenzpreis — no higher tier
exists) for the REMAINDER of the current window, outside the normal
request pipeline.  It creates no ledger entry and no Vorhalteleistung —
settlement sees the resulting exceedance through the ordinary no-grant
paths (STATUS_SBXV.md §2.2).  Everything is logged loudly.

With ``emergency_call_enabled = False`` (the default) any call fails
fast — the feature must be switched on per scenario, never silently.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 4)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sbx_h.fail import rep1
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmergencyCall:
    """Notfall-Abruf [AR §6.4.6] — event-scripted only (plan §3)."""

    aggregation_area_id: str
    direction: Direction
    q_req_mvar: float

    def __post_init__(self) -> None:
        if not self.aggregation_area_id:
            rep1("aggregation_area_id must be non-empty")
        if not math.isfinite(self.q_req_mvar) or self.q_req_mvar <= 0.0:
            rep1("an emergency call must request a positive magnitude",
                 q_req_mvar=self.q_req_mvar)


class EmergencyHandler:
    """Flag-gated collector of active emergency extensions.

    ``call`` registers the extension effective IMMEDIATELY (iteration
    ``k_now``) until the end of the current window (``k_window_end``);
    the commit scheduler consults :meth:`active_extra_mvar` when
    building the priced segment structure.  The extension widens the
    Grenzpreis-priced envelope used for bookkeeping — feasibility is
    never touched (V-D1: the capability box stays the only physical
    constraint).
    """

    def __init__(self, config: SBXVConfig) -> None:
        self._enabled = bool(config.emergency_call_enabled)
        #: (area, direction) → (q_extra_mvar, k_from, k_until_excl)
        self._active: Dict[Tuple[str, Direction],
                           Tuple[float, int, int]] = {}
        #: Replay-comparable event log (primitives only).
        self.log: List[Tuple] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def call(self, call: EmergencyCall, *, k_now: int,
             k_window_end: int) -> None:
        """Register one Notfall-Abruf (fails fast when the feature is
        disabled — flag-gated by requirement, plan §9 Phase 4)."""
        if not self._enabled:
            rep1("EmergencyCall received but emergency_call_enabled is "
                 "False — the Notfall-Abruf path is flag-gated",
                 area_id=call.aggregation_area_id,
                 direction=call.direction.value,
                 q_req_mvar=call.q_req_mvar)
        if k_window_end <= k_now:
            rep1("emergency call outside its window",
                 k_now=k_now, k_window_end=k_window_end)
        key = (call.aggregation_area_id, call.direction)
        prev = self._active.get(key)
        q = call.q_req_mvar if prev is None else max(prev[0],
                                                     call.q_req_mvar)
        self._active[key] = (q, k_now, k_window_end)
        logger.warning(
            "SBX-V EMERGENCY: Notfall-Abruf area %s direction %s "
            "%.1f Mvar, effective iterations [%d, %d) — immediate "
            "activation outside the commit instant [AR §6.4.6].",
            call.aggregation_area_id, call.direction.value,
            call.q_req_mvar, k_now, k_window_end,
        )
        self.log.append(("emergency", call.aggregation_area_id,
                         call.direction.value, call.q_req_mvar,
                         k_now, k_window_end))

    def active_extra_mvar(self, area_id: str, direction: Direction,
                          k: int) -> float:
        """Active emergency extension magnitude at iteration ``k``."""
        entry = self._active.get((area_id, direction))
        if entry is None:
            return 0.0
        q, k_from, k_until = entry
        return q if k_from <= k < k_until else 0.0
