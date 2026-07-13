"""
sbx_v/potentials.py
==================
Potenzialmeldung construction (build plan §3/§4; DP1 sanctioned wrapper).

The operative vertical CAIR message (`core.message.CapabilityMessage`)
carries per-interface DELTA bounds from the current operating point
(load convention at the HV port) and knows neither windows, directions,
nor gesichert flags.  This module wraps it — the original class is
never modified (hard rule 5):

* absolute netted box = Σ_NVP q_meas + [Σ q_min, Σ q_max] (DP5: netting
  per AggregationArea; per-NVP values never receive a reference point);
* direction split per DP3 (`Direction.q_hv_sign`): the LOWERING
  potential is the reachable positive netted ``q_hv`` extreme, the
  RAISING potential the reachable negative extreme (as a magnitude);
* the gesichert share ``q_vh_flagged_mvar`` marks active grants
  [AR §6.4.3];
* the day-ahead plane produces the same message with
  ``is_forecast=True`` [AR §6.3].

The ONE codified fail-fast exception (hard rule 1): a MISSING message
for a window is replaced by *potential := 0 beyond the band*
[AR §6.3.2, Schritt 2] via :func:`substitute_potential` — explicit,
flagged on the message, and LOUDLY logged.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 3)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from core.message import CapabilityMessage
from sbx_h.fail import rep1
from sbx_v.band import NormalBand
from sbx_v.directions import Direction
from sbx_v.grants_ledger import GrantsLedger
from sbx_v.messages import PotentialMessage, Window

logger = logging.getLogger(__name__)


def band_edge_mvar(band: NormalBand, direction: Direction) -> float:
    """Band magnitude of one side (RAISING ↔ q_raise, LOWERING ↔
    q_lower) — the free-tier edge [LF §5.2]."""
    return (band.q_raise_mvar if direction is Direction.RAISING
            else band.q_lower_mvar)


def build_potential_message(
    area_id: str,
    direction: Direction,
    capability: Optional[CapabilityMessage],
    q_meas_netted_mvar: float,
    window: Window,
    band: NormalBand,
    ledger: GrantsLedger,
    *,
    is_forecast: bool = False,
) -> PotentialMessage:
    """One Potenzialmeldung per (area, direction, window) from the
    posted CAIR capability; ``capability is None`` triggers the codified
    substitute (hard rule 1)."""
    if capability is None:
        return substitute_potential(area_id, direction, window, band,
                                    is_forecast=is_forecast)
    if not math.isfinite(q_meas_netted_mvar):
        rep1("netted PCC measurement must be finite",
             area_id=area_id, q_meas_netted_mvar=q_meas_netted_mvar)
    q_min = float(np.sum(np.asarray(capability.q_min_mvar,
                                    dtype=np.float64)))
    q_max = float(np.sum(np.asarray(capability.q_max_mvar,
                                    dtype=np.float64)))
    if not (math.isfinite(q_min) and math.isfinite(q_max)
            and q_min <= q_max):
        rep1("netted capability delta box is invalid",
             area_id=area_id, q_min=q_min, q_max=q_max)
    lo_abs = q_meas_netted_mvar + q_min
    hi_abs = q_meas_netted_mvar + q_max
    # DP3: LOWERING operates at positive netted q_hv, RAISING at
    # negative — the direction's potential is the reachable extreme of
    # the absolute box on that side, as a magnitude (≥ 0).
    if direction is Direction.LOWERING:
        q_pot = max(0.0, hi_abs)
    else:
        q_pot = max(0.0, -lo_abs)
    granted = ledger.granted_mvar(area_id, direction, window.index)
    q_vh_flagged = min(granted, max(0.0, q_pot))
    return PotentialMessage(
        aggregation_area_id=area_id,
        direction=direction,
        q_pot_mvar=q_pot,
        q_vh_flagged_mvar=q_vh_flagged,
        window=window,
        is_forecast=is_forecast,
    )


def substitute_potential(
    area_id: str,
    direction: Direction,
    window: Window,
    band: NormalBand,
    *,
    is_forecast: bool = False,
) -> PotentialMessage:
    """The CODIFIED missing-message substitute [AR §6.3.2, Schritt 2]:
    potential := 0 beyond the band, i.e. the posted potential collapses
    to the band edge.  Loud by requirement (hard rule 1)."""
    edge = band_edge_mvar(band, direction)
    logger.warning(
        "SBX-V: MISSING PotentialMessage for area %s, direction %s, "
        "window %d (%s plane) — applying the codified substitute "
        "'potential := 0 beyond the band' (q_pot = band edge = %.1f "
        "Mvar) per AR §6.3.2 Schritt 2.",
        area_id, direction.value, window.index,
        "forecast" if is_forecast else "operational", edge,
    )
    return PotentialMessage(
        aggregation_area_id=area_id,
        direction=direction,
        q_pot_mvar=edge,
        q_vh_flagged_mvar=0.0,
        window=window,
        is_forecast=is_forecast,
        is_substitute=True,
    )
