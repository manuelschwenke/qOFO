"""
sbx_v/band.py
============
Standard band (*Normalbereich*) per AggregationArea (plan §3, V-D2).

v1: static symmetric box, ±50 Mvar about 0 Mvar, constant for the whole
scenario.  The preset ``ar41414_default`` (5 % raising / 10 % lowering
of contracted P [AR §5.2.1]) is prepared but not the v1 default; a
spread ≥ 70 Mvar is asserted whenever that preset COMPUTES the band
[AR Anhang C].  Explicit fixed values (including the E2 sweep points
down to 0 Mvar) are user data, not preset computations, and carry no
spread assertion.

Sign handling follows hard rule 8: the band stores non-negative edge
magnitudes per direction; signed edges come from
:func:`sbx_v.directions.signed_q_hv_mvar` only.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from sbx_h.fail import rep1
from sbx_v.config import BAND_PRESET_AR41414, BAND_PRESET_FIXED, SBXVConfig
from sbx_v.directions import Direction, signed_q_hv_mvar

#: Minimum band spread asserted for COMPUTED presets [AR Anhang C].
MIN_PRESET_SPREAD_MVAR = 70.0

#: Preset fractions of contracted P [AR §5.2.1] (V-D2).
AR41414_RAISE_FRAC = 0.05
AR41414_LOWER_FRAC = 0.10


@dataclass(frozen=True)
class NormalBand:
    """Free tier (*Normalbereich*) of one AggregationArea.

    Edge magnitudes are non-negative Mvar per direction; the free region
    of the netted signed boundary Q is
    ``[-q_raise_mvar, +q_lower_mvar]`` (DP3 sign convention).
    """

    area_id: str
    q_raise_mvar: float
    q_lower_mvar: float

    def __post_init__(self) -> None:
        if not self.area_id:
            rep1("area_id must be a non-empty string", area_id=self.area_id)
        for name in ("q_raise_mvar", "q_lower_mvar"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                rep1(f"band edge {name} must be finite and non-negative",
                     area_id=self.area_id, **{name: v})

    @property
    def spread_mvar(self) -> float:
        """Band spread [Mvar] = raising edge + lowering edge."""
        return self.q_raise_mvar + self.q_lower_mvar

    def edge_mvar(self, direction: Direction) -> float:
        """Edge MAGNITUDE [Mvar] on the given side."""
        if direction is Direction.RAISING:
            return self.q_raise_mvar
        if direction is Direction.LOWERING:
            return self.q_lower_mvar
        rep1("direction must be an sbxv Direction", direction=direction)

    def signed_edge_mvar(self, direction: Direction) -> float:
        """SIGNED band edge in boundary-Q space (via the single mapping)."""
        return signed_q_hv_mvar(direction, self.edge_mvar(direction))


def band_from_config(
    config: SBXVConfig,
    area_id: str,
    contracted_p_mw: Optional[float] = None,
) -> NormalBand:
    """Construct the area band from the configured preset (V-D2).

    ``fixed``: the explicit config edge magnitudes (no spread assertion —
    they are user data, e.g. the E2 sweep includes 0 Mvar).
    ``ar41414_default``: 5 % raising / 10 % lowering of ``contracted_p_mw``
    [AR §5.2.1]; requires ``contracted_p_mw`` and asserts a spread of at
    least 70 Mvar [AR Anhang C].
    """
    if config.band_preset == BAND_PRESET_FIXED:
        return NormalBand(
            area_id=area_id,
            q_raise_mvar=config.band_q_raise_mvar,
            q_lower_mvar=config.band_q_lower_mvar,
        )
    if config.band_preset == BAND_PRESET_AR41414:
        if contracted_p_mw is None or not math.isfinite(contracted_p_mw) \
                or contracted_p_mw <= 0.0:
            rep1("preset 'ar41414_default' needs a positive contracted P "
                 "[AR §5.2.1]", area_id=area_id,
                 contracted_p_mw=contracted_p_mw)
        band = NormalBand(
            area_id=area_id,
            q_raise_mvar=AR41414_RAISE_FRAC * contracted_p_mw,
            q_lower_mvar=AR41414_LOWER_FRAC * contracted_p_mw,
        )
        if band.spread_mvar < MIN_PRESET_SPREAD_MVAR:
            rep1("computed band spread below the AR Anhang C minimum "
                 "(70 Mvar) — the preset may not be used for this area",
                 area_id=area_id, spread_mvar=band.spread_mvar,
                 contracted_p_mw=contracted_p_mw)
        return band
    rep1("unknown band preset", band_preset=config.band_preset)
