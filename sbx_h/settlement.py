"""
sbx_h/settlement.py
===================
SBX-H v6 settlement: per elapsed cycle, cycle-averaged measurements,
fixed contract prices.  Two tiers relative to the ACTIVE schedule
(``q_std`` from the contracted voltages incl. planned-support
intervals):

* **Tier 1 (standard range).**  ``|q_meas − q_std| ≤ q_band`` → free;
  the signed in-band deviation [Mvar·h] goes to an UNMONETISED netting
  ledger per corridor.  When the deviation exceeds the band, the
  in-band portion (clipped, signed) is still logged and only the
  EXCESS enters the deviation tier.
* **Deviation tier (attributed, causer-pays).**  The beyond-band
  excess ``e = |q_meas − q_std| − q_band > 0`` is decomposed per line
  to first order around the scheduled operating point,

      Δq ≈ Σ_ℓ s_a,ℓ·(v_a,ℓ^meas − v_a,ℓ^sched)
         + Σ_ℓ s_b,ℓ·(v_b,ℓ^meas − v_b,ℓ^sched)
         + Σ_ℓ s_p,ℓ·(p_ℓ^meas − p_ℓ^sched)
         =  C_A + C_B + C_P,

  and the DOMINANT voltage side pays ``κ · p_dev`` per Mvar·h of
  excess.  ΔP-dominant cycles are settlement-neutral; a decomposition
  residual above ``max(attribution_residual_abs_mvar,
  attribution_residual_rel · e)`` flags ``UNATTRIBUTED`` (no charge).

This decomposition is the measurement kernel of architecture candidate
A1 (``docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md``): the ex-post
REMUNERATION of a supporter's over-performance would hook in exactly
here (the C-side term of the non-causer, priced positively) once the
A1 review closes — deliberately NOT implemented yet.

Payments are per-corridor bilateral transfers (payer negative, payee
positive) and must sum to zero per corridor and cycle — asserted on
every emitted settlement.

``n_settle_cycles > 1``: tier evaluation on the ROLLING MEAN of the
last n cycle observations, one settlement per cycle.

v6 (2026-07-12) removed the deal-coupled tier 2 (paid-surplus billing)
and the delivery-verification plumbing together with the deal layer —
archive: ``_archive/sbx_h_v5/``.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-12 (SBX-H v6; original three-tier version 2026-07-07)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from sbx_h.config import SBXConfig
from sbx_h.contract import CorridorContract
from sbx_h.corridor import Corridor, corridor_sensitivities
from sbx_h.fail import rep1

#: Deviation-tier attribution outcomes.
ATTRIB_NONE = "none"                  # no excess this cycle
ATTRIB_SIDE_A = "a"                   # side A's voltage deviation dominates
ATTRIB_SIDE_B = "b"                   # side B's voltage deviation dominates
ATTRIB_DP_NEUTRAL = "dp_neutral"      # ΔP dominates → settlement-neutral
ATTRIB_UNATTRIBUTED = "UNATTRIBUTED"  # residual above tolerance → no charge


@dataclass(frozen=True)
class CycleObservation:
    """Cycle-averaged inputs of one corridor's elapsed cycle.

    Per-line tuples follow the corridor line order.  ``v_sched_*`` are
    the contract voltages that were ACTIVE during the elapsed cycle
    (including planned-support intervals)."""

    cycle: int
    q_meas_mvar: float
    q_std_mvar: float
    v_meas_a_pu: Tuple[float, ...]
    v_meas_b_pu: Tuple[float, ...]
    v_sched_a_pu: Tuple[float, ...]
    v_sched_b_pu: Tuple[float, ...]
    p_meas_mw: Tuple[float, ...]
    p_sched_mw: Tuple[float, ...]
    q_band_mvar: float
    """Tier-1 band ACTIVE during the observed cycle (hourly band
    schedules make this a per-cycle quantity)."""


@dataclass(frozen=True)
class CycleSettlement:
    """Outcome of settling one corridor cycle (window means at n > 1)."""

    corridor: Tuple[int, int]
    cycle: int
    t_cycle_h: float
    q_meas_mvar: float
    q_std_mvar: float
    # Tier 1
    band_dev_mvar: float          # signed in-band deviation (clipped)
    netting_mvarh: float          # signed ledger increment
    # Deviation tier (attributed, causer-pays)
    excess_mvar: float            # beyond-band magnitude (≥ 0)
    contrib_a_mvar: float
    contrib_b_mvar: float
    contrib_p_mvar: float
    residual_mvar: float
    attribution: str
    dev_eur: float
    dev_payer: Optional[int]
    # Bilateral transfers (payer negative, payee positive); sum == 0.
    payments_eur: Dict[int, float]


@dataclass
class _CorridorLedger:
    """Cumulative per-corridor ledger state."""

    netting_mvarh: float = 0.0
    payments_eur: Dict[int, float] = field(default_factory=dict)
    n_unattributed: int = 0
    n_dev_charged: int = 0


class SettlementEngine:
    """Per-corridor v6 settlement (one instance per corridor)."""

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
        if not (math.isfinite(obs.q_band_mvar) and obs.q_band_mvar > 0.0):
            rep1("observation q_band_mvar must be finite and positive",
                 corridor=self.key, cycle=obs.cycle,
                 q_band_mvar=obs.q_band_mvar)

        self._window.append(obs)
        w = list(self._window)

        def _mean(get) -> float:
            return float(np.mean([get(o) for o in w]))

        def _mean_tuple(get) -> Tuple[float, ...]:
            return tuple(float(x)
                         for x in np.mean([get(o) for o in w], axis=0))

        q_meas = _mean(lambda o: o.q_meas_mvar)
        q_std = _mean(lambda o: o.q_std_mvar)
        v_meas_a = _mean_tuple(lambda o: o.v_meas_a_pu)
        v_meas_b = _mean_tuple(lambda o: o.v_meas_b_pu)
        v_sched_a = _mean_tuple(lambda o: o.v_sched_a_pu)
        v_sched_b = _mean_tuple(lambda o: o.v_sched_b_pu)
        p_meas = _mean_tuple(lambda o: o.p_meas_mw)
        p_sched = _mean_tuple(lambda o: o.p_sched_mw)

        t_h = self.config.t_cycle_min / 60.0
        band = _mean(lambda o: o.q_band_mvar)
        a_id, b_id = self.key
        payments = {a_id: 0.0, b_id: 0.0}

        # ── Tier 1: in-band deviation → unmonetised netting ledger ─────
        dev = q_meas - q_std
        band_dev = max(-band, min(band, dev))
        netting = band_dev * t_h

        # ── Deviation tier: beyond-band excess, attributed or flagged ──
        excess = abs(dev) - band
        contrib_a = contrib_b = contrib_p = residual = 0.0
        attribution = ATTRIB_NONE
        dev_eur = 0.0
        dev_payer: Optional[int] = None
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
                    dev_payer = a_id if dominant == ATTRIB_SIDE_A \
                        else b_id
                    payee = b_id if dev_payer == a_id else a_id
                    dev_eur = (self.config.kappa_penalty
                               * self.config.p_dev_eur_per_mvarh
                               * excess * t_h)
                    payments[dev_payer] -= dev_eur
                    payments[payee] += dev_eur
                    self.ledger.n_dev_charged += 1

        # ── Conservation assert (per corridor and cycle) ───────────────
        if abs(sum(payments.values())) > 1e-9:
            rep1("settlement payments do not sum to zero",
                 corridor=self.key, cycle=obs.cycle, payments=payments)

        settlement = CycleSettlement(
            corridor=self.key,
            cycle=obs.cycle,
            t_cycle_h=t_h,
            q_meas_mvar=q_meas,
            q_std_mvar=q_std,
            band_dev_mvar=band_dev,
            netting_mvarh=netting,
            excess_mvar=max(excess, 0.0),
            contrib_a_mvar=contrib_a,
            contrib_b_mvar=contrib_b,
            contrib_p_mvar=contrib_p,
            residual_mvar=residual,
            attribution=attribution,
            dev_eur=dev_eur,
            dev_payer=dev_payer,
            payments_eur=payments,
        )
        self.settlements.append(settlement)
        self.ledger.netting_mvarh += netting
        for z, x in payments.items():
            self.ledger.payments_eur[z] += x
        return settlement


# ----------------------------------------------------------------------
#  Outputs (ledger CSV + Markdown summary per experiment)
# ----------------------------------------------------------------------

_CSV_COLUMNS = (
    "corridor", "cycle", "t_cycle_h", "q_meas_mvar", "q_std_mvar",
    "band_dev_mvar", "netting_mvarh", "excess_mvar", "contrib_a_mvar",
    "contrib_b_mvar", "contrib_p_mvar", "residual_mvar", "attribution",
    "dev_eur", "dev_payer",
)


def write_settlement_outputs(
    engines: Dict[Tuple[int, int], SettlementEngine],
    result_dir,
    experiment_name: str,
) -> Tuple[str, str]:
    """Write the per-cycle ledger CSV and the Markdown summary.

    Returns the two file paths.  ``result_dir`` follows the repo's
    ``results/<experiment>/`` convention (created if missing)."""
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
                    f"{s.q_meas_mvar:.4f}", f"{s.q_std_mvar:.4f}",
                    f"{s.band_dev_mvar:.4f}", f"{s.netting_mvarh:.4f}",
                    f"{s.excess_mvar:.4f}", f"{s.contrib_a_mvar:.4f}",
                    f"{s.contrib_b_mvar:.4f}", f"{s.contrib_p_mvar:.4f}",
                    f"{s.residual_mvar:.4f}", s.attribution,
                    f"{s.dev_eur:.4f}",
                    "" if s.dev_payer is None else s.dev_payer,
                ])

    lines = [
        f"# SBX-H settlement summary — {experiment_name}",
        "",
        "Per-corridor totals (v6: in-band netting + attributed "
        "deviation tier, causer-pays; fixed contract prices).",
        "",
        "| Corridor | Cycles | Netting [Mvar·h] | Deviation charged | "
        "UNATTRIBUTED | Net payments per area [EUR] |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(engines):
        eng = engines[key]
        pay = ", ".join(
            f"z{z}: {x:+.2f}"
            for z, x in sorted(eng.ledger.payments_eur.items())
        )
        lines.append(
            f"| ({key[0]},{key[1]}) | {len(eng.settlements)} | "
            f"{eng.ledger.netting_mvarh:+.3f} | "
            f"{eng.ledger.n_dev_charged} | "
            f"{eng.ledger.n_unattributed} | {pay} |"
        )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return str(csv_path), str(md_path)
