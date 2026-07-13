"""
Minimal SBX-H v6 settlement.

The active bilateral rule uses only scheduled terminal voltages and
cycle-averaged corridor measurements:

1. The reactive-flow baseline is recomputed at the ACTIVE scheduled
   terminal voltages and the MEASURED active-power transfer.
2. A side SAGS if at least one terminal is below its schedule by more
   than the sag threshold.
3. A side HOLDS if all of its terminals remain within the holding
   tolerance below schedule.
4. Payment occurs only when exactly one side sags, the other side
   holds, and the beyond-band reactive flow points from the holding
   side toward the sagging side.

Positive corridor Q is export from area A to area B. Therefore A
supporting sagging B requires positive deviation; B supporting sagging
A requires negative deviation. Payments are bilateral transfers and
sum to zero in every settlement window.

No actuator-level causality or network-strength product is claimed.
observed_strength_mvar_per_mpu is an optional ex-post diagnostic ratio
of delivered support to the holding side's voltage drop.

Author: Manuel Schwenke / OpenAI Codex
Date: 2026-07-13
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from sbx_h.config import SBXConfig
from sbx_h.contract import CorridorContract
from sbx_h.corridor import Corridor, corridor_q_flow
from sbx_h.fail import rep1


SUPPORT_NONE = "none"
SUPPORT_BOTH_SAG = "both_sag"
SUPPORT_A_SAGS_B_NOT_HOLDING = "a_sags_b_not_holding"
SUPPORT_B_SAGS_A_NOT_HOLDING = "b_sags_a_not_holding"
SUPPORT_A_SAGS_B_HOLDS = "a_sags_b_holds"
SUPPORT_B_SAGS_A_HOLDS = "b_sags_a_holds"

DIRECTION_B_TO_A = "b_to_a"
DIRECTION_A_TO_B = "a_to_b"


@dataclass(frozen=True)
class CycleObservation:
    """Cycle-averaged measurements and active schedule of one corridor."""

    cycle: int
    q_meas_mvar: float
    v_meas_a_pu: Tuple[float, ...]
    v_meas_b_pu: Tuple[float, ...]
    v_sched_a_pu: Tuple[float, ...]
    v_sched_b_pu: Tuple[float, ...]
    p_meas_mw: Tuple[float, ...]
    q_band_mvar: float


@dataclass(frozen=True)
class CycleSettlement:
    """Result of one hold/sag support-energy settlement window."""

    corridor: Tuple[int, int]
    cycle: int
    t_cycle_h: float
    q_meas_mvar: float
    q_baseline_mvar: float
    deviation_mvar: float
    q_band_mvar: float
    min_v_error_a_pu: float
    min_v_error_b_pu: float
    a_sags: bool
    b_sags: bool
    a_holds: bool
    b_holds: bool
    support_state: str
    support_direction: Optional[str]
    uncapped_support_mvar: float
    support_mvar: float
    support_mvarh: float
    support_eur: float
    support_payer: Optional[int]
    support_payee: Optional[int]
    observed_strength_mvar_per_mpu: Optional[float]
    payments_eur: Dict[int, float]


@dataclass
class _CorridorLedger:
    """Cumulative support-energy and bilateral payment totals."""

    support_mvarh: float = 0.0
    payments_eur: Dict[int, float] = field(default_factory=dict)
    n_paid: int = 0
    n_a_sags_b_holds: int = 0
    n_b_sags_a_holds: int = 0
    n_both_sag: int = 0


class SettlementEngine:
    """Per-corridor minimal v6 settlement engine."""

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

    def observe(self, obs: CycleObservation) -> CycleSettlement:
        """Settle an elapsed cycle, using a rolling mean if configured."""
        n_lines = self.corridor.n_lines
        for name, seq in (
            ("v_meas_a_pu", obs.v_meas_a_pu),
            ("v_meas_b_pu", obs.v_meas_b_pu),
            ("v_sched_a_pu", obs.v_sched_a_pu),
            ("v_sched_b_pu", obs.v_sched_b_pu),
            ("p_meas_mw", obs.p_meas_mw),
        ):
            if len(seq) != n_lines:
                rep1(
                    "observation arity mismatch",
                    corridor=self.key,
                    field=name,
                    got=len(seq),
                    n_lines=n_lines,
                )
            if not all(math.isfinite(x) for x in seq):
                rep1(
                    "observation contains non-finite entries",
                    corridor=self.key,
                    field=name,
                )
        if not math.isfinite(obs.q_meas_mvar):
            rep1(
                "observation q_meas_mvar must be finite",
                corridor=self.key,
                cycle=obs.cycle,
                q_meas_mvar=obs.q_meas_mvar,
            )
        if not (math.isfinite(obs.q_band_mvar) and obs.q_band_mvar > 0.0):
            rep1(
                "observation q_band_mvar must be finite and positive",
                corridor=self.key,
                cycle=obs.cycle,
                q_band_mvar=obs.q_band_mvar,
            )

        self._window.append(obs)
        window = list(self._window)

        def _mean(get) -> float:
            return float(np.mean([get(item) for item in window]))

        def _mean_tuple(get) -> Tuple[float, ...]:
            return tuple(
                float(value)
                for value in np.mean([get(item) for item in window], axis=0)
            )

        q_meas = _mean(lambda item: item.q_meas_mvar)
        v_meas_a = _mean_tuple(lambda item: item.v_meas_a_pu)
        v_meas_b = _mean_tuple(lambda item: item.v_meas_b_pu)
        v_sched_a = _mean_tuple(lambda item: item.v_sched_a_pu)
        v_sched_b = _mean_tuple(lambda item: item.v_sched_b_pu)
        p_meas = _mean_tuple(lambda item: item.p_meas_mw)
        q_band = _mean(lambda item: item.q_band_mvar)

        q_baseline = corridor_q_flow(
            self.corridor,
            list(v_sched_a),
            list(v_sched_b),
            list(p_meas),
            delta_max_rad=self.config.delta_max_rad,
        )
        deviation = q_meas - q_baseline

        errors_a = tuple(
            measured - scheduled
            for measured, scheduled in zip(v_meas_a, v_sched_a)
        )
        errors_b = tuple(
            measured - scheduled
            for measured, scheduled in zip(v_meas_b, v_sched_b)
        )
        min_error_a = min(errors_a)
        min_error_b = min(errors_b)

        a_sags = min_error_a < -self.contract.v_sag_threshold_pu
        b_sags = min_error_b < -self.contract.v_sag_threshold_pu
        a_holds = min_error_a >= -self.contract.v_hold_tolerance_pu
        b_holds = min_error_b >= -self.contract.v_hold_tolerance_pu

        state = SUPPORT_NONE
        direction: Optional[str] = None
        uncapped_support = 0.0
        candidate_payer: Optional[int] = None
        candidate_payee: Optional[int] = None
        holder_drop_mpu: Optional[float] = None
        a_id, b_id = self.key

        if a_sags and b_sags:
            state = SUPPORT_BOTH_SAG
        elif a_sags:
            if b_holds:
                state = SUPPORT_A_SAGS_B_HOLDS
                direction = DIRECTION_B_TO_A
                uncapped_support = max(0.0, -deviation - q_band)
                candidate_payer = a_id
                candidate_payee = b_id
                holder_drop_mpu = max(0.0, -min_error_b) * 1000.0
            else:
                state = SUPPORT_A_SAGS_B_NOT_HOLDING
        elif b_sags:
            if a_holds:
                state = SUPPORT_B_SAGS_A_HOLDS
                direction = DIRECTION_A_TO_B
                uncapped_support = max(0.0, deviation - q_band)
                candidate_payer = b_id
                candidate_payee = a_id
                holder_drop_mpu = max(0.0, -min_error_a) * 1000.0
            else:
                state = SUPPORT_B_SAGS_A_NOT_HOLDING

        support = uncapped_support
        if self.contract.q_support_cap_mvar is not None:
            support = min(support, self.contract.q_support_cap_mvar)

        t_cycle_h = self.contract.t_cycle_min / 60.0
        support_mvarh = support * t_cycle_h
        support_eur = (
            support_mvarh * self.contract.p_support_eur_per_mvarh
        )

        payments = {a_id: 0.0, b_id: 0.0}
        payer: Optional[int] = None
        payee: Optional[int] = None
        if support_eur > 0.0:
            payer = candidate_payer
            payee = candidate_payee
            if payer is None or payee is None:
                rep1(
                    "positive support lacks bilateral payment roles",
                    corridor=self.key,
                    cycle=obs.cycle,
                    state=state,
                )
            payments[payer] -= support_eur
            payments[payee] += support_eur

        observed_strength: Optional[float] = None
        if (
            support > 0.0
            and holder_drop_mpu is not None
            and holder_drop_mpu > 1e-9
        ):
            observed_strength = support / holder_drop_mpu

        if abs(sum(payments.values())) > 1e-9:
            rep1(
                "settlement payments do not sum to zero",
                corridor=self.key,
                cycle=obs.cycle,
                payments=payments,
            )

        settlement = CycleSettlement(
            corridor=self.key,
            cycle=obs.cycle,
            t_cycle_h=t_cycle_h,
            q_meas_mvar=q_meas,
            q_baseline_mvar=q_baseline,
            deviation_mvar=deviation,
            q_band_mvar=q_band,
            min_v_error_a_pu=min_error_a,
            min_v_error_b_pu=min_error_b,
            a_sags=a_sags,
            b_sags=b_sags,
            a_holds=a_holds,
            b_holds=b_holds,
            support_state=state,
            support_direction=direction,
            uncapped_support_mvar=uncapped_support,
            support_mvar=support,
            support_mvarh=support_mvarh,
            support_eur=support_eur,
            support_payer=payer,
            support_payee=payee,
            observed_strength_mvar_per_mpu=observed_strength,
            payments_eur=payments,
        )
        self.settlements.append(settlement)
        self.ledger.support_mvarh += support_mvarh
        for area, amount in payments.items():
            self.ledger.payments_eur[area] += amount
        if support_eur > 0.0:
            self.ledger.n_paid += 1
        if state == SUPPORT_A_SAGS_B_HOLDS:
            self.ledger.n_a_sags_b_holds += 1
        elif state == SUPPORT_B_SAGS_A_HOLDS:
            self.ledger.n_b_sags_a_holds += 1
        elif state == SUPPORT_BOTH_SAG:
            self.ledger.n_both_sag += 1
        return settlement


_CSV_COLUMNS = (
    "corridor",
    "cycle",
    "t_cycle_h",
    "q_meas_mvar",
    "q_baseline_mvar",
    "deviation_mvar",
    "q_band_mvar",
    "min_v_error_a_pu",
    "min_v_error_b_pu",
    "a_sags",
    "b_sags",
    "a_holds",
    "b_holds",
    "support_state",
    "support_direction",
    "uncapped_support_mvar",
    "support_mvar",
    "support_mvarh",
    "support_eur",
    "support_payer",
    "support_payee",
    "observed_strength_mvar_per_mpu",
)


def write_settlement_outputs(
    engines: Dict[Tuple[int, int], SettlementEngine],
    result_dir,
    experiment_name: str,
) -> Tuple[str, str]:
    """Write the per-cycle ledger CSV and Markdown summary."""
    import csv
    from pathlib import Path

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"{experiment_name}_sbx_settlement_ledger.csv"
    md_path = result_dir / f"{experiment_name}_sbx_settlement_summary.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for key in sorted(engines):
            for item in engines[key].settlements:
                writer.writerow([
                    f"{key[0]}-{key[1]}",
                    item.cycle,
                    f"{item.t_cycle_h:.6f}",
                    f"{item.q_meas_mvar:.4f}",
                    f"{item.q_baseline_mvar:.4f}",
                    f"{item.deviation_mvar:.4f}",
                    f"{item.q_band_mvar:.4f}",
                    f"{item.min_v_error_a_pu:.6f}",
                    f"{item.min_v_error_b_pu:.6f}",
                    item.a_sags,
                    item.b_sags,
                    item.a_holds,
                    item.b_holds,
                    item.support_state,
                    item.support_direction or "",
                    f"{item.uncapped_support_mvar:.4f}",
                    f"{item.support_mvar:.4f}",
                    f"{item.support_mvarh:.4f}",
                    f"{item.support_eur:.4f}",
                    "" if item.support_payer is None
                    else item.support_payer,
                    "" if item.support_payee is None
                    else item.support_payee,
                    "" if item.observed_strength_mvar_per_mpu is None
                    else f"{item.observed_strength_mvar_per_mpu:.6f}",
                ])

    lines = [
        f"# SBX-H settlement summary - {experiment_name}",
        "",
        "Minimal v6 rule: scheduled terminal references plus "
        "directional support-energy payment when one side sags and "
        "the other holds.",
        "",
        "| Corridor | Cycles | Paid windows | Support [Mvar h] | "
        "A sags/B holds | B sags/A holds | Both sag | "
        "Net payments per area [EUR] |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in sorted(engines):
        engine = engines[key]
        payments = ", ".join(
            f"z{area}: {amount:+.2f}"
            for area, amount in sorted(
                engine.ledger.payments_eur.items()
            )
        )
        lines.append(
            f"| ({key[0]},{key[1]}) | "
            f"{len(engine.settlements)} | "
            f"{engine.ledger.n_paid} | "
            f"{engine.ledger.support_mvarh:.3f} | "
            f"{engine.ledger.n_a_sags_b_holds} | "
            f"{engine.ledger.n_b_sags_a_holds} | "
            f"{engine.ledger.n_both_sag} | {payments} |"
        )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return str(csv_path), str(md_path)
