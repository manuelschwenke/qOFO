"""
Minimal SBX-H v6 settlement.

The active bilateral rule uses only scheduled terminal voltages and
cycle-averaged corridor measurements:

1. The reactive-flow baseline is recomputed at the ACTIVE scheduled
   terminal voltages and the MEASURED active-power transfer.
2. A side VIOLATES its schedule if at least one terminal is below or
   above the schedule by more than the voltage-violation threshold.
3. A side HOLDS if every terminal remains inside the symmetric holding
   band around its schedule.
4. Payment occurs only when exactly one side violates, the other side
   holds, and the beyond-band reactive flow has the relieving sign.

Positive corridor Q is export from area A to area B. An undervoltage
needs Q toward the violating side; an overvoltage needs Q away from it.
Payments are bilateral transfers and sum to zero in every window.

The historical fields ``a_sags`` / ``b_sags`` are retained for output
compatibility, but now mean symmetric scheduled-voltage violation.

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

VIOLATION_UNDER = "under"
VIOLATION_OVER = "over"
VIOLATION_MIXED = "mixed"


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
    max_v_error_a_pu: float
    max_v_error_b_pu: float
    violation_kind_a: Optional[str]
    violation_kind_b: Optional[str]
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
        min_error_a, max_error_a = min(errors_a), max(errors_a)
        min_error_b, max_error_b = min(errors_b), max(errors_b)

        def _voltage_role(errors):
            under = min(errors) < -self.contract.v_sag_threshold_pu
            over = max(errors) > self.contract.v_sag_threshold_pu
            holds = max(abs(error) for error in errors) \
                <= self.contract.v_hold_tolerance_pu
            if under and over:
                kind = VIOLATION_MIXED
            elif under:
                kind = VIOLATION_UNDER
            elif over:
                kind = VIOLATION_OVER
            else:
                kind = None
            return under or over, holds, kind

        a_sags, a_holds, violation_kind_a = _voltage_role(errors_a)
        b_sags, b_holds, violation_kind_b = _voltage_role(errors_b)

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
                if violation_kind_a == VIOLATION_UNDER:
                    direction = DIRECTION_B_TO_A
                    uncapped_support = max(0.0, -deviation - q_band)
                elif violation_kind_a == VIOLATION_OVER:
                    direction = DIRECTION_A_TO_B
                    uncapped_support = max(0.0, deviation - q_band)
                if direction is not None:
                    candidate_payer = a_id
                    candidate_payee = b_id
                    holder_drop_mpu = max(
                        abs(error) for error in errors_b
                    ) * 1000.0
            else:
                state = SUPPORT_A_SAGS_B_NOT_HOLDING
        elif b_sags:
            if a_holds:
                state = SUPPORT_B_SAGS_A_HOLDS
                if violation_kind_b == VIOLATION_UNDER:
                    direction = DIRECTION_A_TO_B
                    uncapped_support = max(0.0, deviation - q_band)
                elif violation_kind_b == VIOLATION_OVER:
                    direction = DIRECTION_B_TO_A
                    uncapped_support = max(0.0, -deviation - q_band)
                if direction is not None:
                    candidate_payer = b_id
                    candidate_payee = a_id
                    holder_drop_mpu = max(
                        abs(error) for error in errors_a
                    ) * 1000.0
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
            max_v_error_a_pu=max_error_a,
            max_v_error_b_pu=max_error_b,
            violation_kind_a=violation_kind_a,
            violation_kind_b=violation_kind_b,
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
    "max_v_error_a_pu",
    "max_v_error_b_pu",
    "violation_kind_a",
    "violation_kind_b",
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
                    f"{item.max_v_error_a_pu:.6f}",
                    f"{item.max_v_error_b_pu:.6f}",
                    item.violation_kind_a or "",
                    item.violation_kind_b or "",
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
        "directional support-energy payment when one side violates "
        "its symmetric voltage band and the other holds.",
        "",
        "| Corridor | Cycles | Paid windows | Support [Mvar h] | "
        "A violates/B holds | B violates/A holds | Both violate | "
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
