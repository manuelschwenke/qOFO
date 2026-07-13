"""
sbx_v/config.py
==============
Frozen configuration for SBX-V (build plan §8 — all keys explicit, no
defaults hidden in code; the values below ARE the plan §8 v1 defaults
and the Phase-0-resolved ``<report>`` values, STATUS_SBXV.md §0.2).

Prices are exogenous constants (administratively derived pass-through
Durchschnitts-/Grenzpreise in the sense of §12h EnWG / BK6-23-072; no
market between Netzbetreiber, no profit [LF §1, §4.1]).  One
Verrechnungsperiode per scenario.

DP2 (€ → objective conversion, APPROVED by Manuel 2026-07-09)
-------------------------------------------------------------
The TSO MIQP objective carries no currency dimension (weighted squared
errors; ``g_v = 1e7`` per pu², ``alpha = 1``).  ``obj_per_eur_per_step``
is the single explicit exchange rate: objective units per euro of
per-step commercial cost.  Anchor: one quantum (30 Mvar) at Grenzpreis
for one TSO step (10 €/Mvarh · 30 Mvar · 0.05 h = 15 €) costs the
objective exactly as much as one bus sitting ``v_viol_threshold_pu``
(5 mpu, the SBX need-flag threshold) outside its voltage band under
``g_v`` (1e7 · 0.005² = 250 objective units) → 250 / 15 ≈ 16.7.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sbx_h.fail import rep1

#: Sanctioned band presets (V-D2).
BAND_PRESET_FIXED = "fixed"
BAND_PRESET_AR41414 = "ar41414_default"

#: Sanctioned MIQP incentive models (V-D9).
COST_MODEL_NEAREST_EDGE = "nearest_edge"
COST_MODEL_LEITFADEN = "leitfaden_exact_when_granted"


@dataclass(frozen=True)
class SBXVConfig:
    """Frozen SBX-V configuration (plan §8)."""

    # --- Band (V-D2, REVISED by Manuel 2026-07-10) ---
    band_q_raise_mvar: float = 50.0
    band_q_lower_mvar: float = 50.0
    """Explicit edge magnitudes [Mvar]; used ONLY under
    ``band_preset="fixed"`` (e.g. the E2 sweep arms)."""

    band_preset: str = BAND_PRESET_AR41414
    """V-D2 as revised 2026-07-10: the operative Normalbereich is the
    E VDE-AR-N 4141-4 default — 5 % raising / 10 % lowering of the
    contracted P per AggregationArea [AR §5.2.1], spread ≥ 70 Mvar
    asserted [AR Anhang C].  The adapter derives the contracted P from
    the rated interface capacity (Σ ``sn_hv_mva``); constructing the
    band without a contracted P fails fast.  ``"fixed"`` selects the
    explicit edges above (sweep arms, unit tests)."""

    # --- Scheduling plane cadence (V-D3, DP4 §0.3) ---
    window_s: float = 900.0
    tso_period_s: float = 180.0
    """TSO OFO iteration period [s]; must match the experiment's
    ``MultiTSOConfig.tso_period_s`` (asserted at wiring time, Phase 4).
    ``window_s`` must be an exact integer multiple of it — SBX-V keeps
    its OWN window counter and does not reuse the SBX-H 6-min cycle
    (STATUS_SBXV.md §0.3)."""

    dq_grant_mvar: float = 30.0
    """Grant quantum [Mvar] per 15-min window — the SBX-H quantum rate
    (30 Mvar per 15 min) evaluated on one window (Phase 0 §0.2)."""

    t_persist_s: float = 180.0
    """Need-flag persistence [s]; must be an exact integer multiple of
    ``tso_period_s`` (persistence is counted in TSO iterations, mirroring
    ``sbx_h.need.NeedTracker``)."""

    n_clear: int = 2
    """Need-flag clearing hysteresis in TSO iterations: BOTH conditions
    must stay clear this many consecutive iterations before the flag
    drops (plan §6; Phase-4 wiring of the Phase-3 constructor knob)."""

    sat_tol_mvar: float = 1.0
    """Condition-A saturation tolerance [Mvar]: the netted PCC dispatch
    counts as saturated within this distance of the free-or-granted
    segment edge."""

    v_dev_threshold_pu: float = 0.005
    """Condition-B threshold [pu]: worst transmission-bus deviation
    beyond the Sollspannungs band in the corresponding direction (the
    SBX-H need-flag voltage threshold, 5 mpu)."""

    reserve_margin_mvar: float = 0.0
    """Reserve-Observer margin [Mvar] subtracted on the DSO side of the
    Machbarkeitsprüfung (feasibility answers only — request sizing is
    TSO-side and never sees it)."""

    ramp_steps: int = 3
    """Grant-expiry ramp length in TSO iterations (§4 scheduling plane);
    proposed value, no repo anchor (STATUS_SBXV.md §0.2)."""

    tolerance_frac: float = 0.10
    """±10 % tolerance on the Vorhalteleistung magnitude [LF §4.7]."""

    # --- Exogenous prices (V-D4/V-D8; placeholders per STATUS §0.2) ---
    price_arb_avg_eur_per_mvarh: float = 5.0
    price_arb_grenz_eur_per_mvarh: float = 10.0
    price_lp_avg_eur_per_mvar_day: float = 25.0
    price_lp_grenz_eur_per_mvar_day: float = 50.0

    # --- MIQP incentive model (V-D9) ---
    miqp_cost_model: str = COST_MODEL_LEITFADEN

    miqp_pricing_enabled: bool = True
    """When True (default), the tier prices act on the TSO MIQP from
    window 0 onward — beyond-band dispatch costs the Grenzpreis even
    without a grant (V-D9: '0 in band, Grenzpreis beyond, when no grant
    is active').  ``False`` is the NEUTRAL/R1 configuration: the
    PricingSolver passes every problem through untouched and the
    dispatch is byte-identical to the CAIR baseline; metering,
    need flags, pipeline and settlement still run (reporting only)."""

    obj_per_eur_per_step: float = 250.0 / 15.0
    """DP2 conversion constant (approved 2026-07-09): objective units
    per euro of per-step commercial cost.  See module docstring."""

    g_z_tier: float = 1.0e4
    """Slack weight of the tier boundary rows added to the TSO MIQP by
    :mod:`sbx_v.miqp_cost`.  The shared-slack encoding of the solver
    makes an exactly hard tier row impossible; with quadratic slack the
    incentive distortion is bounded by ``c/(2·g_z_tier)`` Mvar per side
    (≈ 4e-4 Mvar at the default prices) — asserted post-solve."""

    # --- Emergency (Phase 4, flag-gated) ---
    emergency_call_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("band_q_raise_mvar", "band_q_lower_mvar"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                rep1(f"{name} must be finite and non-negative",
                     **{name: v})
        if self.band_preset not in (BAND_PRESET_FIXED, BAND_PRESET_AR41414):
            rep1("band_preset must be 'fixed' or 'ar41414_default'",
                 band_preset=self.band_preset)
        if self.tso_period_s <= 0.0 or not math.isfinite(self.tso_period_s):
            rep1("tso_period_s must be positive",
                 tso_period_s=self.tso_period_s)
        if self.window_s <= 0.0 or not math.isfinite(self.window_s):
            rep1("window_s must be positive", window_s=self.window_s)
        k = self.window_s / self.tso_period_s
        if abs(k - round(k)) > 1e-9 or round(k) < 1:
            rep1("window_s must be a positive integer multiple of "
                 "tso_period_s (SBX-V window counter, STATUS §0.3)",
                 window_s=self.window_s, tso_period_s=self.tso_period_s)
        n = self.t_persist_s / self.tso_period_s
        if abs(n - round(n)) > 1e-9 or round(n) < 1:
            rep1("t_persist_s must be a positive integer multiple of "
                 "tso_period_s (persistence is counted in iterations)",
                 t_persist_s=self.t_persist_s,
                 tso_period_s=self.tso_period_s)
        if self.dq_grant_mvar <= 0.0:
            rep1("dq_grant_mvar must be positive",
                 dq_grant_mvar=self.dq_grant_mvar)
        if self.n_clear < 1:
            rep1("n_clear must be a positive iteration count",
                 n_clear=self.n_clear)
        if not math.isfinite(self.sat_tol_mvar) or self.sat_tol_mvar < 0.0:
            rep1("sat_tol_mvar must be finite and non-negative",
                 sat_tol_mvar=self.sat_tol_mvar)
        if not math.isfinite(self.v_dev_threshold_pu) or \
                self.v_dev_threshold_pu <= 0.0:
            rep1("v_dev_threshold_pu must be positive",
                 v_dev_threshold_pu=self.v_dev_threshold_pu)
        if not math.isfinite(self.reserve_margin_mvar) or \
                self.reserve_margin_mvar < 0.0:
            rep1("reserve_margin_mvar must be finite and non-negative",
                 reserve_margin_mvar=self.reserve_margin_mvar)
        if self.ramp_steps < 1:
            rep1("ramp_steps must be a positive iteration count",
                 ramp_steps=self.ramp_steps)
        if not (0.0 < self.tolerance_frac < 1.0):
            rep1("tolerance_frac must lie in (0, 1)",
                 tolerance_frac=self.tolerance_frac)
        for name in ("price_arb_avg_eur_per_mvarh",
                     "price_arb_grenz_eur_per_mvarh",
                     "price_lp_avg_eur_per_mvar_day",
                     "price_lp_grenz_eur_per_mvar_day"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0.0:
                rep1(f"{name} must be a positive finite constant",
                     **{name: v})
        if not (self.price_arb_grenz_eur_per_mvarh
                > self.price_arb_avg_eur_per_mvarh):
            rep1("Arbeits-Grenzpreis must exceed the Durchschnittspreis "
                 "(plan §8 assertion)",
                 avg=self.price_arb_avg_eur_per_mvarh,
                 grenz=self.price_arb_grenz_eur_per_mvarh)
        if not (self.price_lp_grenz_eur_per_mvar_day
                > self.price_lp_avg_eur_per_mvar_day):
            rep1("Leistungs-Grenzpreis must exceed the Durchschnittspreis",
                 avg=self.price_lp_avg_eur_per_mvar_day,
                 grenz=self.price_lp_grenz_eur_per_mvar_day)
        if self.miqp_cost_model not in (COST_MODEL_NEAREST_EDGE,
                                        COST_MODEL_LEITFADEN):
            rep1("miqp_cost_model must be 'nearest_edge' or "
                 "'leitfaden_exact_when_granted' (V-D9)",
                 miqp_cost_model=self.miqp_cost_model)
        if not math.isfinite(self.obj_per_eur_per_step) or \
                self.obj_per_eur_per_step <= 0.0:
            rep1("obj_per_eur_per_step must be a positive finite constant "
                 "(DP2)", obj_per_eur_per_step=self.obj_per_eur_per_step)
        if not math.isfinite(self.g_z_tier) or self.g_z_tier <= 0.0:
            rep1("g_z_tier must be positive", g_z_tier=self.g_z_tier)

    # ------------------------------------------------------------------
    #  Derived quantities
    # ------------------------------------------------------------------

    @property
    def k_window(self) -> int:
        """Window length in TSO iterations (integer, asserted above)."""
        return int(round(self.window_s / self.tso_period_s))

    @property
    def n_persist(self) -> int:
        """Need-flag persistence in TSO iterations."""
        return int(round(self.t_persist_s / self.tso_period_s))

    def arb_price_obj_per_mvar_step(
        self, price_eur_per_mvarh: float,
    ) -> float:
        """€/Mvarh → objective units per Mvar per TSO step (DP2).

        ``c = price · (tso_period_s / 3600 h) · obj_per_eur_per_step``.
        """
        if not math.isfinite(price_eur_per_mvarh) or \
                price_eur_per_mvarh < 0.0:
            rep1("price must be finite and non-negative",
                 price_eur_per_mvarh=price_eur_per_mvarh)
        return (price_eur_per_mvarh * (self.tso_period_s / 3600.0)
                * self.obj_per_eur_per_step)

    @property
    def c_vh_obj_per_mvar_step(self) -> float:
        """Tier-2 (Vorhalteleistung) slope in objective units/Mvar/step."""
        return self.arb_price_obj_per_mvar_step(
            self.price_arb_avg_eur_per_mvarh)

    @property
    def c_ug_obj_per_mvar_step(self) -> float:
        """Tier-3 (ungesichert) slope in objective units/Mvar/step."""
        return self.arb_price_obj_per_mvar_step(
            self.price_arb_grenz_eur_per_mvarh)
