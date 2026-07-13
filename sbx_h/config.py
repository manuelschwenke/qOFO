"""
sbx_h/config.py
===============
Frozen configuration for SBX-H v6 (contract layer + attributed
settlement + escalation indicator).

v6 (2026-07-12) removed every deal-layer knob — quanta, caps, dust
thresholds, capability modes, delivery gates, arming, request sizing,
release dwell — together with the mechanisms they parameterised (015
campaign, findings G1–G7; v5 archive: ``_archive/sbx_h_v5/``).

Symbol map
----------
* ``k_sched``            ↔ cycle length in TSO OFO iterations (metering
                           / settlement cadence; schedule lookups happen
                           at cycle boundaries).
* ``q_band_mvar``        ↔ tier-1 band half-width around ``q_std``
                           (in-band deviations are netted, not priced).
* ``p_dev_eur_per_mvarh``↔ price basis of the attributed beyond-band
                           deviation tier (causer-pays at
                           ``kappa_penalty ×`` this price).
* ``v_viol_threshold_pu``/``n_need``/``release_threshold_pu``
                         ↔ per-area violation indicator (set threshold,
                           persistence, release hysteresis) — feeds the
                           ESCALATION flag, no runtime action.
* ``escalation_cycles``  ↔ consecutive flagged/beyond-band cycle
                           boundaries after which the re-planning
                           escalation (candidate A4) is signalled.
* ``w_track``/``w_track_factor``
                         ↔ priority weight of the corridor-terminal
                           voltage references in each zone's existing
                           tracked-output mechanism.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-12 (SBX-H v6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sbx_h.fail import rep1


@dataclass(frozen=True)
class SBXConfig:
    """Frozen SBX-H v6 configuration."""

    k_sched: int = 2
    """Cycle length in TSO OFO iterations (metering and settlement
    cadence; 2 × 180 s = 6-min cycles on the shared 005 scenario)."""

    tso_period_s: float = 180.0
    """TSO OFO iteration period [s]; must match the experiment's
    ``MultiTSOConfig.tso_period_s``."""

    q_band_mvar: float = 5.0
    """Tier-1 band half-width [Mvar] around the standard flow (constant
    default; hourly planning bands enter via the contract's
    ``q_band_schedule``)."""

    p_dev_eur_per_mvarh: float = 5.0  # PLACEHOLDER — calibrate
    """Price basis [EUR/Mvar·h] of the attributed deviation tier."""

    kappa_penalty: float = 2.0
    """Beyond-band deviations attributed to a voltage side are charged
    ``kappa_penalty × p_dev_eur_per_mvarh`` (causer-pays)."""

    v_viol_threshold_pu: float = 0.005
    n_need: int = 3
    """Violation indicator: set when the area's worst monitored-bus
    bound violation exceeds ``v_viol_threshold_pu`` for ``n_need``
    consecutive TSO iterations (direction-consistent)."""

    release_threshold_pu: Optional[float] = None
    """Hysteresis of the violation indicator: once set, it clears only
    when the depth falls below THIS threshold (must be ≤
    ``v_viol_threshold_pu``; ``None`` = equal, no hysteresis)."""

    escalation_cycles: int = 4
    """A4 escalation indicator: an area whose violation flag — or a
    corridor whose beyond-band exceedance — persists for MORE than
    this many consecutive cycle boundaries is flagged for
    RE-PLANNING (the schedule, not the control, is the object that
    needs updating).  Recorded and reported; no runtime action."""

    w_track: Optional[float] = None
    """ABSOLUTE boundary-voltage tracking weight for the corridor
    terminals; ``None`` (default) resolves to ``w_track_factor × the
    zone's g_v``."""

    w_track_factor: float = 20.0
    """Corridor-terminal contract references are tracked with
    ``w_track_factor × g_v`` — priority over the ordinary schedule
    tracking (v4 measurement: with a uniform weight the zone MIQP
    delivered only a fraction of the commanded boundary state)."""

    delta_max_rad: float = 0.6
    """δ bracket of the contracted π-line P-equation."""

    attribution_residual_abs_mvar: float = 1.0
    attribution_residual_rel: float = 0.20
    """Deviation-tier attribution: decomposition residual above
    ``max(abs, rel × excess)`` → ``UNATTRIBUTED``, no charge."""

    n_settle_cycles: int = 1
    """Settlement averaging window (> 1 = rolling mean of the last n
    cycle observations; one settlement per cycle)."""

    def __post_init__(self) -> None:
        if self.k_sched < 1:
            rep1("k_sched must be a positive iteration count",
                 k_sched=self.k_sched)
        if self.tso_period_s <= 0.0:
            rep1("tso_period_s must be positive",
                 tso_period_s=self.tso_period_s)
        for name in ("q_band_mvar", "p_dev_eur_per_mvarh"):
            if getattr(self, name) <= 0.0:
                rep1(f"{name} must be positive",
                     **{name: getattr(self, name)})
        if self.kappa_penalty < 1.0:
            rep1("kappa_penalty must be >= 1 (the deviation tier prices "
                 "a multiple of the base price)",
                 kappa_penalty=self.kappa_penalty)
        if self.v_viol_threshold_pu <= 0.0:
            rep1("v_viol_threshold_pu must be positive",
                 v_viol_threshold_pu=self.v_viol_threshold_pu)
        if self.n_need < 1:
            rep1("n_need must be a positive count", n_need=self.n_need)
        if self.release_threshold_pu is not None and not (
                0.0 < self.release_threshold_pu
                <= self.v_viol_threshold_pu):
            rep1("release_threshold_pu must lie in "
                 "(0, v_viol_threshold_pu]",
                 release_threshold_pu=self.release_threshold_pu,
                 v_viol_threshold_pu=self.v_viol_threshold_pu)
        if self.escalation_cycles < 1:
            rep1("escalation_cycles must be >= 1",
                 escalation_cycles=self.escalation_cycles)
        if self.w_track is not None and self.w_track <= 0.0:
            rep1("w_track, when set, must be positive",
                 w_track=self.w_track)
        if self.w_track_factor <= 0.0:
            rep1("w_track_factor must be positive",
                 w_track_factor=self.w_track_factor)
        if self.delta_max_rad <= 0.0:
            rep1("delta_max_rad must be positive",
                 delta_max_rad=self.delta_max_rad)
        if self.attribution_residual_abs_mvar <= 0.0 or \
                not (0.0 < self.attribution_residual_rel < 1.0):
            rep1("attribution residual thresholds out of range",
                 abs_mvar=self.attribution_residual_abs_mvar,
                 rel=self.attribution_residual_rel)
        if self.n_settle_cycles < 1:
            rep1("n_settle_cycles must be >= 1",
                 n_settle_cycles=self.n_settle_cycles)

    @property
    def t_cycle_min(self) -> float:
        """Cycle length [min] = k_sched · tso_period_s / 60."""
        return self.k_sched * self.tso_period_s / 60.0
