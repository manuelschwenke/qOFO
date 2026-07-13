"""
sbx_v/directions.py
==================
Direction semantics for SBX-V (build plan §0 vocabulary, hard rule 8).

Regulatory vocabulary [LF §4.4; AR §5.4]:

* ``Direction.RAISING``  — *spannungshebend* / übererregt: the
  distribution system injects reactive power into the transmission
  system, raising transmission-side voltages.
* ``Direction.LOWERING`` — *spannungssenkend* / untererregt: the
  distribution system absorbs reactive power from the transmission
  system, lowering transmission-side voltages.

Executable sign convention (DP3 — CONFIRMED by Manuel, 2026-07-09)
------------------------------------------------------------------
The physical boundary quantity of one AggregationArea is its netted
``q_hv_mvar``: the sum over the area's PCC interface transformers of
``net.res_trafo3w.q_hv_mvar`` — pandapower load convention at the EHV
port, positive = reactive power flowing from the EHV bus into the
transformer (TS → DS).  Hence:

    positive netted q_hv  →  DS under-excited (absorbs Q)  →  LOWERING
    negative netted q_hv  →  DS injects Q into the TS      →  RAISING

Hard rule 8: this module holds THE single mapping between a
(direction, magnitude) pair and the signed pandapower boundary Q.
Every other SBX-V module passes non-negative magnitudes plus a
:class:`Direction` across its boundaries and converts only here.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 1)
"""

from __future__ import annotations

import enum
import math
from typing import Tuple

from sbx_h.fail import rep1


class Direction(enum.Enum):
    """Direction of a vertical reactive-power product [LF §4.4]."""

    RAISING = "raising"    # spannungshebend (DS delivers Q to the TS)
    LOWERING = "lowering"  # spannungssenkend (DS absorbs Q from the TS)

    @property
    def q_hv_sign(self) -> float:
        """Sign of the netted ``q_hv_mvar`` on this side (DP3).

        ``LOWERING`` operates at positive ``q_hv`` (TS → DS),
        ``RAISING`` at negative ``q_hv``.
        """
        return +1.0 if self is Direction.LOWERING else -1.0

    @property
    def opposite(self) -> "Direction":
        """The other direction (used for opposite-edge references, V-D8)."""
        return (Direction.RAISING if self is Direction.LOWERING
                else Direction.LOWERING)


def signed_q_hv_mvar(direction: Direction, magnitude_mvar: float) -> float:
    """THE single (direction, magnitude) → signed boundary-Q mapping.

    ``magnitude_mvar`` must be a finite non-negative Mvar value; the
    result is the signed netted ``q_hv_mvar`` of the AggregationArea in
    the pandapower load convention at the EHV port.
    """
    if not isinstance(direction, Direction):
        rep1("direction must be an sbxv Direction", direction=direction)
    if not math.isfinite(magnitude_mvar) or magnitude_mvar < 0.0:
        rep1("magnitude must be a finite non-negative Mvar value",
             direction=direction, magnitude_mvar=magnitude_mvar)
    return direction.q_hv_sign * magnitude_mvar


def split_signed_q_hv(q_hv_mvar: float) -> Tuple[Direction, float]:
    """Signed netted boundary Q → (direction, non-negative magnitude).

    Exactly zero is direction-free; it resolves deterministically to
    ``(LOWERING, 0.0)`` (documented convention — a zero magnitude has no
    register, price, or settlement effect in any downstream module).
    """
    if not math.isfinite(q_hv_mvar):
        rep1("signed boundary Q must be finite", q_hv_mvar=q_hv_mvar)
    if q_hv_mvar < 0.0:
        return Direction.RAISING, -q_hv_mvar
    return Direction.LOWERING, q_hv_mvar
