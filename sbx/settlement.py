"""
sbx/settlement.py
=================
Three-tier settlement for SBX Minimal (plan v2 §2.5) — per elapsed
cycle, cycle-averaged measurements only, at FIXED contract prices.

All tiers are relative to the schedule that was ACTIVE during the
elapsed cycle (``q_sched = q_std + s``, references including the
acting-side ``dv``):

* **Tier 1 (standard range).**  ``|q_meas − q_sched| ≤ q_band`` → free;
  the signed in-band deviation [Mvar·h] goes to an UNMONETISED netting
  ledger per corridor (plan §7 item 3: logged only in v2).  When the
  deviation exceeds the band, the in-band portion (clipped, signed) is
  still logged and only the EXCESS enters tier 3.
* **Tier 2 (surplus transfer).**  The scheduler decomposes the surplus
  into paid / unpaid accumulators (unilateral deals → paid, mutual →
  unpaid; unwind reduces paid first — all upstream in
  ``sbx.scheduler``).  The PAID portion is billed
  ``p_surplus × |paid| × t_cycle`` per cycle; **payer = the importing
  (non-acting) side** (§2.5 verbatim).  Note: for an export-need
  requester the acting side IS the requester, so the importer/payer is
  then the supporter — the plan's rule is price-per-delivered-Mvar·h,
  not requester-pays; recorded as an open point in STATUS_SBX.md.
* **Tier 3 (unsolicited excess).**  ``e = |q_meas − q_sched| − q_band
  > 0`` is charged ``κ · p_surplus`` per Mvar·h, attributed by the
  DOMINANT term of the per-line first-order decomposition summed per
  side (references include the acting-side dv):

      Δq ≈ Σ_ℓ s_a,ℓ·(v_a,ℓ^meas − v_a,ℓ^sched)
         + Σ_ℓ s_b,ℓ·(v_b,ℓ^meas − v_b,ℓ^sched)
         + Σ_ℓ s_p,ℓ·(p_ℓ^meas − p_ℓ^sched)
         =  C_A + C_B + C_P.

  ΔP-driven components are settlement-neutral (dominant C_P → no
  charge).  Decomposition residual ``|Δq − (C_A + C_B + C_P)| >
  max(attribution_residual_abs_mvar, attribution_residual_rel · e)`` →
  flag ``UNATTRIBUTED``, no charge.

Payments are per-corridor bilateral transfers (payer negative, payee
positive) and must sum to zero per corridor and cycle — asserted on
every emitted settlement (fail-fast, plan hard rule 1).

``n_settle_cycles > 1`` (short-cycle ablation, §4 Phase 7): the tier
evaluation runs on the ROLLING MEAN of the last n cycle observations
(measurements AND schedule quantities), one settlement per cycle.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 6)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from sbx.config import SBXConfig
from sbx.contract import CorridorContract
from sbx.corridor import Corridor, corridor_sensitivities
from sbx.fail import rep1

#: Tier-3 attribution outcomes.
ATTRIB_NONE = "none"                  # no excess this cycle
ATTRIB_SIDE_A = "a"                   # side A's voltage deviation dominates
ATTRIB_SIDE_B = "b"                   # side B's voltage deviation dominates
ATTRIB_DP_NEUTRAL = "dp_neutral"      # ΔP dominates → settlement-neutral
ATTRIB_UNATTRIBUTED = "UNATTRIBUTED"  # residual above tolerance → no charge


@dataclass(frozen=True)
class CycleObservation:
    """Cycle-averaged inputs of one corridor's elapsed cycle (§2.5).

    Per-line tuples follow the corridor line order.  ``v_sched_*``
    are the references that were ACTIVE during the elapsed cycle
    (contract voltages plus the acting-side ``dv``); ``acting_end`` is
    ``"a"``/``"b"`` or ``None`` at zero surplus.
    """

    cycle: int
    q_meas_mvar: float
    q_std_mvar: float
    surplus_mvar: float
    surplus_paid_mvar: float
    surplus_unpaid_mvar: float
    acting_end: Optional[str]
    v_meas_a_pu: Tuple[float, ...]
    v_meas_b_pu: Tuple[float, ...]
    v_sched_a_pu: Tuple[float, ...]
    v_sched_b_pu: Tuple[float, ...]
    p_meas_mw: Tuple[float, ...]
    p_sched_mw: Tuple[float, ...]

    @property
    def q_sched_mvar(self) -> float:
        return self.q_std_mvar + self.surplus_mvar


@dataclass(frozen=True)
class CycleSettlement:
    """Outcome of settling one corridor cycle (window means at n > 1)."""

    corridor: Tuple[int, int]
    cycle: int
    t_cycle_h: float
    q_meas_mvar: float
    q_sched_mvar: float
    # Tier 1
    band_dev_mvar: float          # signed in-band deviation (clipped)
    netting_mvarh: float          # signed ledger increment
    # Tier 2
    paid_mvarh: float             # |paid surplus| × t_cycle
    tier2_eur: float
    tier2_payer: Optional[int]    # area id (importing, non-acting side)
    # Tier 3
    excess_mvar: float            # beyond-band magnitude (≥ 0)
    contrib_a_mvar: float
    contrib_b_mvar: float
    contrib_p_mvar: float
    residual_mvar: float
    attribution: str
    tier3_eur: float
    tier3_payer: Optional[int]
    # Bilateral transfers (payer negative, payee positive); sum == 0.
    payments_eur: Dict[int, float]


@dataclass
class _CorridorLedger:
    """Cumulative per-corridor ledger state."""

    netting_mvarh: float = 0.0
    payments_eur: Dict[int, float] = field(default_factory=dict)
    n_unattributed: int = 0
    n_tier3_charged: int = 0


class SettlementEngine:
    """Per-corridor three-tier settlement (one instance per corridor)."""

    def __init__(
        self,
        corridor: Corridor,
        contract: CorridorContract,
        config: SBXConfig,
    ) -> None:
        contract.assert_matches(corridor)
        self.corridor = corridor
        self.contract = contract
        self.config = config
        self.key = (corridor.area_a, corridor.area_b)
        self._window: Deque[CycleObservation] = deque(
            maxlen=config.n_settle_cycles,
        )
        self.ledger = _CorridorLedger(
            payments_eur={corridor.area_a: 0.0, corridor.area_b: 0.0},
        )
        self.settlements: List[CycleSettlement] = []

    # ------------------------------------------------------------------

    def observe(self, obs: CycleObservation) -> CycleSettlement:
        """Settle one elapsed cycle (window mean when n_settle_cycles > 1)."""
        n_lines = self.corridor.n_lines
        for name, seq in (("v_meas_a_pu", obs.v_meas_a_pu),
                          ("v_meas_b_pu", obs.v_meas_b_pu),
                          ("v_sched_a_pu", obs.v_sched_a_pu),
                          ("v_sched_b_pu", obs.v_sched_b_pu),
                          ("p_meas_mw", obs.p_meas_mw),
                          ("p_sched_mw", obs.p_sched_mw)):
            if len(seq) != n_lines:
                rep1("observation arity mismatch", corridor=self.key,
                     field=name, got=len(seq), n_lines=n_lines)
            if not all(math.isfinite(x) for x in seq):
                rep1("observation contains non-finite entries",
                     corridor=self.key, field=name)
        if obs.acting_end not in (None, "a", "b"):
            rep1("acting_end must be None, 'a' or 'b'",
                 corridor=self.key, acting_end=obs.acting_end)
        if abs(obs.surplus_paid_mvar + obs.surplus_unpaid_mvar
               - obs.surplus_mvar) > 1e-9:
            rep1("paid + unpaid surplus must equal the surplus",
                 corridor=self.key, paid=obs.surplus_paid_mvar,
                 unpaid=obs.surplus_unpaid_mvar, s=obs.surplus_mvar)

        self._window.append(obs)
        w = list(self._window)

        def _mean(get) -> float:
            return float(np.mean([get(o) for o in w]))

        def _mean_tuple(get) -> Tuple[float, ...]:
            return tuple(float(x)
                         for x in np.mean([get(o) for o in w], axis=0))

        q_meas = _mean(lambda o: o.q_meas_mvar)
        q_sched = _mean(lambda o: o.q_sched_mvar)
        paid = _mean(lambda o: o.surplus_paid_mvar)
        v_meas_a = _mean_tuple(lambda o: o.v_meas_a_pu)
        v_meas_b = _mean_tuple(lambda o: o.v_meas_b_pu)
        v_sched_a = _mean_tuple(lambda o: o.v_sched_a_pu)
        v_sched_b = _mean_tuple(lambda o: o.v_sched_b_pu)
        p_meas = _mean_tuple(lambda o: o.p_meas_mw)
        p_sched = _mean_tuple(lambda o: o.p_sched_mw)

        t_h = self.config.t_cycle_min / 60.0
        band = self.contract.q_band_mvar
        a_id, b_id = self.key
        payments = {a_id: 0.0, b_id: 0.0}

        # ── Tier 1: in-band deviation → unmonetised netting ledger ─────
        dev = q_meas - q_sched
        band_dev = max(-band, min(band, dev))
        netting = band_dev * t_h

        # ── Tier 2: paid surplus billed; payer = importing side ────────
        paid_mvarh = abs(paid) * t_h
        tier2_eur = self.config.p_surplus_eur_per_mvarh * paid_mvarh
        tier2_payer: Optional[int] = None
        if tier2_eur > 0.0:
            # §2.5: the acting side is sign(q_sched − q_std) = sign(s).
            # The billed quantity is the WINDOWED mean paid surplus, so
            # the payer direction must come from the same window: with
            # n_settle_cycles > 1 the LATEST cycle may already be fully
            # unwound (obs.acting_end None) while the window still
            # carries paid history (bug found in the 016 short-cycle
            # ablation, 2026-07-08). Knife-edge fallback: a zero windowed
            # surplus with nonzero windowed paid takes the direction of
            # the paid component itself (the transfer actually billed).
            s_mean = _mean(lambda o: o.surplus_mvar)
            ref = s_mean if s_mean != 0.0 else paid
            acting_w = "a" if ref > 0.0 else "b"
            # Acting side exports the surplus; the importer pays.
            tier2_payer = b_id if acting_w == "a" else a_id
            payee = a_id if tier2_payer == b_id else b_id
            payments[tier2_payer] -= tier2_eur
            payments[payee] += tier2_eur

        # ── Tier 3: beyond-band excess, attributed or flagged ──────────
        excess = abs(dev) - band
        contrib_a = contrib_b = contrib_p = residual = 0.0
        attribution = ATTRIB_NONE
        tier3_eur = 0.0
        tier3_payer: Optional[int] = None
        if excess > 0.0:
            per_line, _, _ = corridor_sensitivities(
                self.corridor, list(v_sched_a), list(v_sched_b),
                list(p_sched), delta_max_rad=self.config.delta_max_rad,
            )
            for k in range(n_lines):
                s_a, s_b, s_p = per_line[k]
                contrib_a += s_a * (v_meas_a[k] - v_sched_a[k])
                contrib_b += s_b * (v_meas_b[k] - v_sched_b[k])
                contrib_p += s_p * (p_meas[k] - p_sched[k])
            residual = dev - (contrib_a + contrib_b + contrib_p)
            tol = max(self.config.attribution_residual_abs_mvar,
                      self.config.attribution_residual_rel * excess)
            if abs(residual) > tol:
                attribution = ATTRIB_UNATTRIBUTED
                self.ledger.n_unattributed += 1
            else:
                dominant = max(
                    (abs(contrib_a), ATTRIB_SIDE_A),
                    (abs(contrib_b), ATTRIB_SIDE_B),
                    (abs(contrib_p), ATTRIB_DP_NEUTRAL),
                )[1]
                attribution = dominant
                if dominant in (ATTRIB_SIDE_A, ATTRIB_SIDE_B):
                    tier3_payer = a_id if dominant == ATTRIB_SIDE_A \
                        else b_id
                    payee = b_id if tier3_payer == a_id else a_id
                    tier3_eur = (self.config.kappa_penalty
                                 * self.config.p_surplus_eur_per_mvarh
                                 * excess * t_h)
                    payments[tier3_payer] -= tier3_eur
                    payments[payee] += tier3_eur
                    self.ledger.n_tier3_charged += 1

        # ── Conservation assert (per corridor and cycle) ───────────────
        if abs(sum(payments.values())) > 1e-9:
            rep1("settlement payments do not sum to zero",
                 corridor=self.key, cycle=obs.cycle, payments=payments)

        settlement = CycleSettlement(
            corridor=self.key,
            cycle=obs.cycle,
            t_cycle_h=t_h,
            q_meas_mvar=q_meas,
            q_sched_mvar=q_sched,
            band_dev_mvar=band_dev,
            netting_mvarh=netting,
            paid_mvarh=paid_mvarh,
            tier2_eur=tier2_eur,
            tier2_payer=tier2_payer,
            excess_mvar=max(excess, 0.0),
            contrib_a_mvar=contrib_a,
            contrib_b_mvar=contrib_b,
            contrib_p_mvar=contrib_p,
            residual_mvar=residual,
            attribution=attribution,
            tier3_eur=tier3_eur,
            tier3_payer=tier3_payer,
            payments_eur=payments,
        )
        self.settlements.append(settlement)
        self.ledger.netting_mvarh += netting
        for z, x in payments.items():
            self.ledger.payments_eur[z] += x
        return settlement


# ----------------------------------------------------------------------
#  Outputs (plan §2.5: ledger CSV + Markdown summary per experiment)
# ----------------------------------------------------------------------

_CSV_COLUMNS = (
    "corridor", "cycle", "t_cycle_h", "q_meas_mvar", "q_sched_mvar",
    "band_dev_mvar", "netting_mvarh", "paid_mvarh", "tier2_eur",
    "tier2_payer", "excess_mvar", "contrib_a_mvar", "contrib_b_mvar",
    "contrib_p_mvar", "residual_mvar", "attribution", "tier3_eur",
    "tier3_payer",
)


def write_settlement_outputs(
    engines: Dict[Tuple[int, int], SettlementEngine],
    result_dir,
    experiment_name: str,
) -> Tuple[str, str]:
    """Write the per-cycle ledger CSV and the Markdown summary.

    Returns the two file paths.  ``result_dir`` follows the repo's
    ``results/<experiment>/`` convention (created if missing).
    """
    import csv
    from pathlib import Path

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"{experiment_name}_sbx_settlement_ledger.csv"
    md_path = result_dir / f"{experiment_name}_sbx_settlement_summary.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        for key in sorted(engines):
            for s in engines[key].settlements:
                writer.writerow([
                    f"{key[0]}-{key[1]}", s.cycle, f"{s.t_cycle_h:.6f}",
                    f"{s.q_meas_mvar:.4f}", f"{s.q_sched_mvar:.4f}",
                    f"{s.band_dev_mvar:.4f}", f"{s.netting_mvarh:.4f}",
                    f"{s.paid_mvarh:.4f}", f"{s.tier2_eur:.4f}",
                    "" if s.tier2_payer is None else s.tier2_payer,
                    f"{s.excess_mvar:.4f}", f"{s.contrib_a_mvar:.4f}",
                    f"{s.contrib_b_mvar:.4f}", f"{s.contrib_p_mvar:.4f}",
                    f"{s.residual_mvar:.4f}", s.attribution,
                    f"{s.tier3_eur:.4f}",
                    "" if s.tier3_payer is None else s.tier3_payer,
                ])

    lines = [
        f"# SBX settlement summary — {experiment_name}",
        "",
        "Per-corridor totals (fixed contract prices, plan §2.5).",
        "",
        "| Corridor | Cycles | Netting [Mvar·h] | Tier-2 [EUR] | "
        "Tier-3 charged | UNATTRIBUTED | Net payments per area [EUR] |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(engines):
        eng = engines[key]
        tier2_total = sum(s.tier2_eur for s in eng.settlements)
        pay = ", ".join(
            f"z{z}: {x:+.2f}"
            for z, x in sorted(eng.ledger.payments_eur.items())
        )
        lines.append(
            f"| ({key[0]},{key[1]}) | {len(eng.settlements)} | "
            f"{eng.ledger.netting_mvarh:+.3f} | {tier2_total:.2f} | "
            f"{eng.ledger.n_tier3_charged} | "
            f"{eng.ledger.n_unattributed} | {pay} |"
        )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return str(csv_path), str(md_path)
