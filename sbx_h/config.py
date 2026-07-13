"""
sbx_h/config.py
===============
Frozen configuration for the minimal SBX-H v6 mechanism: scheduled
terminal-voltage references, hold/sag support-energy settlement, and a
slow re-planning escalation indicator.

The v5 runtime deal layer remains archived under '_archive/sbx_h_v5/'.
The active path has no requests, grants, capability offers, delivery
gates, or runtime setpoint changes beyond applying the agreed schedule.

Symbol map
----------
* k_sched: metering / settlement cycle length in TSO OFO iterations.
* q_band_mvar: reactive-flow deadband around the baseline implied by
  scheduled voltages at measured active transfer.
* p_support_eur_per_mvarh: energy price paid by the sagging side to the
  side that remains inside its holding band.
* v_hold_tolerance_pu / v_sag_threshold_pu: absolute role thresholds
  relative to the scheduled corridor-terminal voltage.
* v_viol_threshold_pu / n_need / release_threshold_pu: independent
  area-level violation indicator feeding slow re-planning.
* w_track / w_track_factor: relative weight of scheduled terminal references
  in the local TSO controller.

Author: Manuel Schwenke / Claude Code / OpenAI Codex
Date: 2026-07-13 (minimal SBX-H v6 settlement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sbx_h.fail import rep1


@dataclass(frozen=True)
class SBXConfig:
    """Frozen SBX-H v6 configuration."""

    k_sched: int = 2
    """Cycle length in TSO OFO iterations (2 × 180 s = 6 min by
    default)."""

    tso_period_s: float = 180.0
    """TSO OFO iteration period [s]."""

    q_band_mvar: float = 5.0
    """Reactive-flow settlement deadband [Mvar]. The flow baseline is
    recomputed at scheduled terminal voltages and measured active
    transfer. Hourly values may enter through q_band_schedule."""

    p_support_eur_per_mvarh: float = 5.0  # PLACEHOLDER — calibrate
    """Support-energy price [EUR/Mvar·h]."""

    q_support_cap_mvar: Optional[float] = None
    """Optional paid-support cap [Mvar] per corridor and window.
    None leaves delivered support uncapped. This is a commercial
    exposure cap, not a sold strength product."""

    v_hold_tolerance_pu: float = 0.0025
    """A side HOLDS when every own corridor terminal is no more than
    this amount below its active scheduled voltage."""

    v_sag_threshold_pu: float = 0.005
    """A side SAGS when at least one own corridor terminal lies more
    than this amount below its active scheduled voltage."""

    v_viol_threshold_pu: float = 0.005
    n_need: int = 3
    """Independent area-level violation indicator used only for the
    slow re-planning escalation signal."""

    release_threshold_pu: Optional[float] = None
    """Indicator release hysteresis; None uses the set threshold."""

    escalation_cycles: int = 4
    """Consecutive flagged/beyond-band cycle boundaries after which
    re-planning is signalled. No runtime control action is taken."""

    w_track: Optional[float] = None
    """Absolute boundary-voltage tracking weight; None resolves to
    w_track_factor times the zone's ordinary voltage-tracking weight."""

    # Neutral by default: scheduled terminals do not receive an implicit
    # priority over the other monitored voltage buses. Values above one
    # remain available for explicit sensitivity studies.
    w_track_factor: float = 1.0

    delta_max_rad: float = 0.6
    """Delta bracket of the contracted pi-line P-equation."""

    n_settle_cycles: int = 1
    """Rolling averaging window; one settlement is emitted per cycle."""

    def __post_init__(self) -> None:
        if self.k_sched < 1:
            rep1("k_sched must be a positive iteration count",
                 k_sched=self.k_sched)
        if self.tso_period_s <= 0.0:
            rep1("tso_period_s must be positive",
                 tso_period_s=self.tso_period_s)
        for name in ("q_band_mvar", "p_support_eur_per_mvarh"):
            if getattr(self, name) <= 0.0:
                rep1(f"{name} must be positive",
                     **{name: getattr(self, name)})
        if self.q_support_cap_mvar is not None and \
                self.q_support_cap_mvar <= 0.0:
            rep1("q_support_cap_mvar must be positive when set",
                 q_support_cap_mvar=self.q_support_cap_mvar)
        if not (0.0 <= self.v_hold_tolerance_pu
                < self.v_sag_threshold_pu):
            rep1("hold/sag thresholds must satisfy "
                 "0 <= v_hold_tolerance_pu < v_sag_threshold_pu",
                 v_hold_tolerance_pu=self.v_hold_tolerance_pu,
                 v_sag_threshold_pu=self.v_sag_threshold_pu)
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
        if self.n_settle_cycles < 1:
            rep1("n_settle_cycles must be >= 1",
                 n_settle_cycles=self.n_settle_cycles)

    @property
    def t_cycle_min(self) -> float:
        """Cycle length [min] = k_sched * tso_period_s / 60."""
        return self.k_sched * self.tso_period_s / 60.0
