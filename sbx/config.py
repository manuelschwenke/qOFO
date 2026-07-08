"""
sbx/config.py
=============
Frozen configuration for SBX Minimal (plan v2 §5; v2.2 amendment adds no
new fields).

Symbol map (code ↔ plan)
------------------------
* ``k_sched``                       ↔ cycle length in TSO OFO iterations
                                      (§2.1 t_cycle; Phase 0: 5 iterations
                                      = 15 min at the 3-min TSO period).
* ``dq_quant_rate_mvar_per_15min``  ↔ schedule-change quantum, defined as a
                                      rate; per-cycle quantum
                                      dq_quant = rate · (t_cycle / 15 min).
* ``dq_min_deal_mvar``              ↔ dust threshold (§2.2 Step 3).
* ``dq_contract_max_mvar``          ↔ cap on |q_sched − q_std| (§2.1).
* ``q_band_mvar``                   ↔ standard-range half-width (tier 1).
* ``p_surplus_eur_per_mvarh``       ↔ fixed surplus price (tier 2).
* ``kappa_penalty``                 ↔ tier-3 multiple.
* ``v_viol_threshold_pu``/``n_need``↔ need flag (§2.3).
* ``m_release``                     ↔ unwind dwell in cycles (§2.2 Step 5).
* ``w_track``                       ↔ boundary-V soft-tracking weight; None
                                      resolves to the zone's existing
                                      ``g_v`` (STATUS_SBX.md A4/G5 — per-bus
                                      weights would require touching the
                                      MIQP assembly, which is forbidden).
* ``v_search_range_pu``             ↔ bracket for ``v_sched_for_q``.
* ``dv_search_range_pu``            ↔ bracket for the corridor ``dv`` solve
                                      (§2.2 Step 4).
* ``voltage_margin_pu``             ↔ capability-LP margin; per v2.2 item 3
                                      also absorbs within-cycle terminal
                                      shifts from parallel corridor deals.
* ``delta_max_rad``                 ↔ δ bracket of the tie-line P-equation.
* ``attribution_residual_*``        ↔ tier-3 UNATTRIBUTED flag threshold:
                                      max(abs, rel · |excess|).
* ``n_settle_cycles``               ↔ settlement averaging window; > 1 only
                                      in the short-cycle ablation (§ Phase 7).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sbx.fail import rep1


@dataclass(frozen=True)
class SBXConfig:
    """Frozen SBX configuration (plan v2 §5 defaults)."""

    k_sched: int = 2
    """Cycle length in TSO OFO iterations.  Phase 0: the shared 005/011/012
    scenario runs the TSO every 180 s, so 5 iterations ≙ 15 min."""

    tso_period_s: float = 180.0
    """TSO OFO iteration period [s] (Phase 0 step-time semantics).  Used
    only to convert ``k_sched`` into the cycle length ``t_cycle``; must
    match the experiment's ``MultiTSOConfig.tso_period_s``."""

    dq_quant_rate_mvar_per_15min: float = 30.0
    dq_min_deal_mvar: float = 1.0
    dq_contract_max_mvar: float = 150.0
    q_band_mvar: float = 5.0
    p_surplus_eur_per_mvarh: float = 5.0  # PLACEHOLDER — calibrate
    kappa_penalty: float = 2.0
    v_viol_threshold_pu: float = 0.005
    n_need: int = 1
    m_release: int = 3

    w_track: Optional[float] = None
    """Boundary-V soft-tracking weight.  ``None`` (default) resolves to the
    zone's existing ``g_v`` at scheduler construction (A4/G5)."""

    v_search_range_pu: Tuple[float, float] = (0.90, 1.10)
    dv_search_range_pu: Tuple[float, float] = (-0.05, +0.05)

    voltage_margin_pu: float = 0.01
    """Capability-LP voltage margin; per v2.2 item 3 it also absorbs
    within-cycle terminal shifts from parallel deals.  Plan §5 default
    was 0.005; RAISED to 0.01 by the Phase 5 smoke calibration
    (2026-07-07): the worst observed within-cycle supporter-side
    terminal shift over deal cycles was 6.4 mpu — quantum tracking plus
    the neighbour's recovery transient, which no observational split can
    exclude (plan §7: defaults revisited only with evidence)."""
    delta_max_rad: float = 0.6
    attribution_residual_abs_mvar: float = 1.0
    attribution_residual_rel: float = 0.20
    n_settle_cycles: int = 1

    def __post_init__(self) -> None:
        if self.k_sched < 1:
            rep1("k_sched must be a positive iteration count",
                 k_sched=self.k_sched)
        if self.tso_period_s <= 0.0:
            rep1("tso_period_s must be positive", tso_period_s=self.tso_period_s)
        for name in ("dq_quant_rate_mvar_per_15min", "dq_contract_max_mvar",
                     "q_band_mvar", "p_surplus_eur_per_mvarh"):
            if getattr(self, name) <= 0.0:
                rep1(f"{name} must be positive", **{name: getattr(self, name)})
        if self.dq_min_deal_mvar < 0.0:
            rep1("dq_min_deal_mvar must be non-negative",
                 dq_min_deal_mvar=self.dq_min_deal_mvar)
        if self.kappa_penalty < 1.0:
            rep1("kappa_penalty must be >= 1 (tier 3 charges a multiple of "
                 "the surplus price)", kappa_penalty=self.kappa_penalty)
        if self.v_viol_threshold_pu <= 0.0:
            rep1("v_viol_threshold_pu must be positive",
                 v_viol_threshold_pu=self.v_viol_threshold_pu)
        if self.n_need < 1 or self.m_release < 1:
            rep1("n_need and m_release must be positive counts",
                 n_need=self.n_need, m_release=self.m_release)
        if self.w_track is not None and self.w_track <= 0.0:
            rep1("w_track, when set, must be positive", w_track=self.w_track)
        lo, hi = self.v_search_range_pu
        if not (0.0 < lo < hi):
            rep1("v_search_range_pu must be an increasing positive interval",
                 v_search_range_pu=self.v_search_range_pu)
        dlo, dhi = self.dv_search_range_pu
        if not (dlo < 0.0 < dhi):
            rep1("dv_search_range_pu must straddle zero",
                 dv_search_range_pu=self.dv_search_range_pu)
        if self.voltage_margin_pu <= 0.0:
            rep1("voltage_margin_pu must be positive",
                 voltage_margin_pu=self.voltage_margin_pu)
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

    @property
    def dq_quant_mvar(self) -> float:
        """Per-cycle schedule-change quantum [Mvar] (rate-scaled, §2.1)."""
        return self.dq_quant_rate_mvar_per_15min * (self.t_cycle_min / 15.0)
