"""
sbx_h/config.py
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

from sbx_h.fail import rep1


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

    dq_quant_rate_mvar_per_15min: float = 10.0
    dq_min_deal_mvar: float = 1.0
    dq_contract_max_mvar: float = 150.0
    q_band_mvar: float = 5.0
    p_surplus_eur_per_mvarh: float = 5.0  # PLACEHOLDER — calibrate
    kappa_penalty: float = 2.0
    v_viol_threshold_pu: float = 0.005
    n_need: int = 3
    m_release: int = 3

    w_track: Optional[float] = None
    """ABSOLUTE boundary-V tracking weight for the corridor terminals.
    ``None`` (default) resolves to ``w_track_factor × the zone's g_v``
    (v4).  Setting it overrides the factor with one absolute value for
    every zone."""

    w_track_factor: float = 20.0
    """v4 'deliverable SBX' (2026-07-09): corridor-terminal contract
    references are tracked with ``w_track_factor × g_v`` — PRIORITY over
    the ordinary schedule tracking.  With the pre-v4 uniform weight the
    zone MIQP traded the contract reference off against every other
    monitored bus and delivered only a fraction of the commanded
    boundary shift (measured: ~250 Mvar schedule/measurement divergence
    under a 150-Mvar ratchet).  ``1.0`` restores the legacy A4/G5
    behaviour."""

    delivery_gate: bool = True
    """v4: delivery-conditioned requesting.  While a corridor's last
    settled cycle is classified ``sign_mismatch`` or ``magnitude_off``
    (the scheduled surplus is demonstrably not realised), NEW requests
    on that corridor are suppressed, and the non-delivery counts like a
    cleared need flag towards the unwind dwell — undelivered scheduled
    support is wound back instead of billed indefinitely.  After a full
    unwind the evidence resets (``no_surplus``) and the mechanism
    PROBES again: under persistent non-delivery the result is a bounded
    deal/mismatch/unwind retry loop (surplus ≤ ~1 quantum) rather than
    the pre-v4 ratchet to the contract cap."""

    capability_mode: str = "per_corridor"
    """v4: ``"joint_box"`` = v2.2 D13 (any-combination guarantee; known
    to collapse to zero offers for areas with collinear corridor
    couplings — finding F2); ``"per_corridor"`` = one 2-vertex LP per
    corridor (own corridor guaranteed, cross-corridor side effects
    tolerated and priced by the tier-3 attribution); ``"auto"`` =
    joint box first, per-corridor fallback on collapse.  DEFAULT
    ``per_corridor`` (Manuel, 2026-07-09): the outer approximation is
    the same philosophy the vertical CAIR uses per interface, and for
    small quanta the over-promise is bounded (≈ quantum/|s_corr| per
    corridor in terminal-voltage terms) and absorbed by the voltage
    margin + delivery gate + tier-3 attribution.  The scheduler warns
    when Σ_c quantum_c/|s_corr,c| exceeds the margin budget (the
    quantifiable 'small quantum' condition)."""

    record_modal_capability: bool = False
    """v4 evidence recording (Manuel's modal-offer proposal,
    2026-07-09): for two-corridor areas, compute the modal bounds
    (a_plus, a_minus) — |z±| ≤ a± with z± = (Δq_1 ± Δq_2)/2 — once per
    cycle from two support-function LPs and record them in
    ``SBXScheduler.modal_records``.  Diagnostic only (no protocol
    change).  Default OFF since v5 (2026-07-10): two extra LPs per
    cycle for a diagnostic; enable explicitly when the modal evidence
    is wanted."""

    # ── v5 "evidence-based SBX" (2026-07-10, Manuel-approved redesign
    # after the 015 helpfulness campaign; findings G1–G6 in
    # STATUS_SBX.md / results/015_SBX_COMPARE/REPORT.md) ────────────────
    require_exhaustion_to_request: bool = True
    """v5 Move 1 — C1 arming.  A need flag alone does not emit a
    request; the area must ALSO be unable to help itself: the
    OPTIMISTIC cached-model bound on the area's own reachable lift at
    the worst-violated bus (Σ_j |H[worst, j]| · relieving headroom_j,
    voltage constraints elsewhere ignored) falls below
    ``c1_arming_factor × depth``.  Measured basis (015): in the
    misdirected regime the pinned control clears the violation with
    the area's OWN reserves and deals merely execute-and-unwind;
    arming on exhaustion removes that dead activity.  ``False``
    restores the v4 trigger (need flag alone)."""

    c1_arming_factor: float = 1.0
    """Arming margin for the model-bound part of
    :attr:`require_exhaustion_to_request`: the bound arms iff
    ``self_help_lift < factor × violation_depth``.  1.0 = arm exactly
    when even the optimistic bound cannot reach the violated limit.
    NOTE the bound uses SETPOINT headroom (e.g. AVR voltage-setpoint
    boxes) which can vastly overstate the physically deliverable Q of
    saturated machines — the measured-stall clause below is the
    robust arming path on real controllers (first D2S1 v5 run,
    2026-07-10: the bound alone never armed the exhausted zone)."""

    c1_stall_cycles: int = 2
    """Measured-stall arming (the OR-clause of C1): a need flag that
    has persisted for MORE than this many consecutive cycle
    boundaries..."""

    c1_stall_improvement: float = 0.3
    """...while the violation depth has recovered by LESS than this
    fraction of its onset value counts as exhausted — the area is
    demonstrably not helping itself, whatever the model bound says.
    Behavioural, model-free, and immune to the setpoint-vs-physical
    headroom mismatch."""

    delivery_check: str = "voltage"
    """v5 Move 2 — what the delivery gate verifies (active only with
    ``delivery_gate = True``):

    * ``"voltage"`` (v5 default): the ACTING side's corridor-terminal
      voltages tracked their shifted references at the elapsed cycle's
      LAST sample within ``v_delivery_tol_pu`` — verifies what the
      supporter actually controls and measures locally, and is immune
      to the stress-driven natural flow shift that blinds the
      magnitude test (015 finding G3: band-independent suppression).
    * ``"magnitude"`` (v4 legacy): the consistency classification
      (sign_mismatch / magnitude_off) on the realised corridor flow —
      kept for comparison arms."""

    v_delivery_tol_pu: float = 0.0025
    """Voltage-delivery tolerance [pu] for
    ``delivery_check = "voltage"``: worst per-line acting-terminal
    tracking error at the elapsed cycle's LAST sample (settled view —
    the first in-cycle sample still shows the pre-shift plant).  The
    v4 priority tracking realises dv steps to ~1 mpu within one TSO
    iteration; 2.5 mpu flags genuine non-delivery (saturated or
    non-tracking supporter, F11) without false positives."""

    release_threshold_pu: Optional[float] = None
    """v5 Move 3 — preventive release (hysteresis on the need flag).
    Once set, the flag clears only when the violation depth falls
    below THIS threshold (must be ≤ ``v_viol_threshold_pu``).
    ``None`` (default) = equal to ``v_viol_threshold_pu``, the v4
    behaviour whose stopping rule stalls relief 'just below flag
    depth' (F9).  Set lower (e.g. 0.001) to keep requesting until a
    comfort margin is restored."""

    request_sizing: str = "gap"
    """v5 Move 3 — request magnitude:

    * ``"gap"`` (default): ``n = ceil((depth − release) /
      (|dV_worst/dQ_import| · dq_quant))`` quanta, clipped to
      [1, ``k_max_quanta_per_request``] — sized to the remaining
      violation via the area's own cached corridor sensitivity;
    * ``"single"`` (v4 legacy): exactly one quantum per cycle."""

    k_max_quanta_per_request: int = 4
    """Cap on the sized request AND on the per-corridor offer scale
    (``a_c = min(t, k_max) · dq_quant``) in quanta per cycle.  Bounds
    the within-cycle terminal shift a supporter may be asked for; the
    scheduler's margin warning accounts for it."""

    tier2_requires_delivery: bool = True
    """v5 settlement rule: the paid surplus is billed (tier 2) only
    for cycles whose delivery verification succeeded — no payment for
    undelivered support (015/D2S1_nogate measured 96 Mvar scheduled
    per ~11 Mvar delivered under v4 billing).  ``False`` restores
    unconditional v4 billing."""

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
        if self.w_track_factor <= 0.0:
            rep1("w_track_factor must be positive",
                 w_track_factor=self.w_track_factor)
        if self.capability_mode not in ("auto", "joint_box",
                                        "per_corridor"):
            rep1("capability_mode must be 'auto', 'joint_box' or "
                 "'per_corridor'", capability_mode=self.capability_mode)
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
        # v5 knobs.
        if self.c1_arming_factor <= 0.0:
            rep1("c1_arming_factor must be positive",
                 c1_arming_factor=self.c1_arming_factor)
        if self.c1_stall_cycles < 1:
            rep1("c1_stall_cycles must be >= 1",
                 c1_stall_cycles=self.c1_stall_cycles)
        if not (0.0 < self.c1_stall_improvement < 1.0):
            rep1("c1_stall_improvement must lie in (0, 1)",
                 c1_stall_improvement=self.c1_stall_improvement)
        if self.delivery_check not in ("voltage", "magnitude"):
            rep1("delivery_check must be 'voltage' or 'magnitude'",
                 delivery_check=self.delivery_check)
        if self.v_delivery_tol_pu <= 0.0:
            rep1("v_delivery_tol_pu must be positive",
                 v_delivery_tol_pu=self.v_delivery_tol_pu)
        if self.release_threshold_pu is not None and not (
                0.0 < self.release_threshold_pu
                <= self.v_viol_threshold_pu):
            rep1("release_threshold_pu must lie in "
                 "(0, v_viol_threshold_pu]",
                 release_threshold_pu=self.release_threshold_pu,
                 v_viol_threshold_pu=self.v_viol_threshold_pu)
        if self.request_sizing not in ("gap", "single"):
            rep1("request_sizing must be 'gap' or 'single'",
                 request_sizing=self.request_sizing)
        if self.k_max_quanta_per_request < 1:
            rep1("k_max_quanta_per_request must be >= 1",
                 k_max_quanta_per_request=self.k_max_quanta_per_request)

    @property
    def t_cycle_min(self) -> float:
        """Cycle length [min] = k_sched · tso_period_s / 60."""
        return self.k_sched * self.tso_period_s / 60.0

    @property
    def dq_quant_mvar(self) -> float:
        """Per-cycle schedule-change quantum [Mvar] (rate-scaled, §2.1)."""
        return self.dq_quant_rate_mvar_per_15min * (self.t_cycle_min / 15.0)
