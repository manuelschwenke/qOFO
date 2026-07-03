"""
controller/discrete_hygiene.py
==============================
Discrete hygiene for the Boundary Marginal Exchange (BME) scheme — spec
§3.8 (see ``docs/BME_STATUS.md``).

Symbol map (code ↔ spec)
------------------------
* ``SlottingSchedule``   ↔ §3.8.2 discrete slotting (DECISION D5: round
                           robin, slot length 1 TSO step): area i may
                           COMMIT discrete moves only when
                           ``(tick // slot_length) mod N_A`` selects it;
                           continuous moves every step. Deterministically
                           prevents two areas from counter-switching on
                           the same stale marginals in the same step.
* ``epsilon_accepts``    ↔ §3.8.3 ε-improvement acceptance: commit the
                           discrete part only if
                           Φ̂(MIQP) ≤ Φ̂(QP_frozen) − ε_switch
                           − c_switchᵀ·|Δu_d|, where Φ̂ is the LOCAL
                           quadratic model prediction (the MIQP/QP
                           objective values of the per-step problem) and
                           c_switch a per-device-class switching cost
                           (DECISION D6 — magnitudes calibrated in
                           Phase 6). Scope per Q5: MIQP integers only;
                           MSC/MSR integrator banks keep their own commit
                           rule but emit notices and ledger entries.
* ``SwitchingLedger``    ↔ §3.8.3 switching ledger (append-only):
                           predicted ΔΦ̂, realised ΔΦ (filled one step
                           later from measurements — Φ_global is the
                           simulation oracle's privilege), accepted flag,
                           device class, slot info. This is the empirical
                           premise data for the finite-switching argument
                           (§3.10.2).

Fail-fast: invalid schedules, unknown zones, double realised-ΔΦ fills
and out-of-range indices raise. The ledger is append-only — entries are
never mutated except for the single deferred ``realised_dphi`` fill.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-03 (BME Phase 5)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class SlottingSchedule:
    """Round-robin discrete-commit schedule (§3.8.2, D5).

    Parameters
    ----------
    zone_ids :
        Participating zones; the rotation order is ascending zone id
        (deterministic, no configuration surface).
    slot_length :
        Slot length in TSO steps (D5 default 1).
    """

    def __init__(self, zone_ids: Sequence[int], slot_length: int = 1) -> None:
        zones = sorted(int(z) for z in zone_ids)
        if len(zones) < 1:
            raise ValueError("SlottingSchedule needs at least one zone")
        if len(set(zones)) != len(zones):
            raise ValueError(f"duplicate zone ids: {zones}")
        if int(slot_length) < 1:
            raise ValueError(
                f"slot_length must be ≥ 1, got {slot_length}"
            )
        self.zone_order: List[int] = zones
        self.slot_length = int(slot_length)

    def slot_owner(self, tick: int) -> int:
        """The zone allowed to commit discrete moves at this TSO tick."""
        if int(tick) < 0:
            raise ValueError(f"tick must be ≥ 0, got {tick}")
        pos = (int(tick) // self.slot_length) % len(self.zone_order)
        return self.zone_order[pos]

    def may_commit(self, zone: int, tick: int) -> bool:
        if int(zone) not in self.zone_order:
            raise ValueError(
                f"unknown zone {zone}; schedule covers {self.zone_order}"
            )
        return self.slot_owner(tick) == int(zone)


def epsilon_accepts(
    obj_miqp: float,
    obj_frozen: float,
    delta_int_abs: np.ndarray,
    switch_costs: np.ndarray,
    epsilon_switch: float,
) -> Tuple[bool, float, float]:
    """§3.8.3 acceptance test.

    Returns ``(accepted, predicted_dphi, total_switch_cost)`` where
    ``predicted_dphi = Φ̂(MIQP) − Φ̂(QP_frozen)`` (negative = predicted
    improvement of the local quadratic model) and the rule is
    ``predicted_dphi ≤ −(ε_switch + c_switchᵀ|Δu_d|)``.
    """
    delta_int_abs = np.asarray(delta_int_abs, dtype=np.float64)
    switch_costs = np.asarray(switch_costs, dtype=np.float64)
    if delta_int_abs.shape != switch_costs.shape:
        raise ValueError(
            f"delta_int_abs {delta_int_abs.shape} and switch_costs "
            f"{switch_costs.shape} must be aligned."
        )
    if np.any(delta_int_abs < 0.0) or np.any(switch_costs < 0.0):
        raise ValueError("magnitudes and costs must be non-negative")
    if float(epsilon_switch) < 0.0:
        raise ValueError(
            f"epsilon_switch must be ≥ 0, got {epsilon_switch}"
        )
    predicted_dphi = float(obj_miqp) - float(obj_frozen)
    total_cost = float(switch_costs @ delta_int_abs)
    accepted = predicted_dphi <= -(float(epsilon_switch) + total_cost)
    return accepted, predicted_dphi, total_cost


@dataclass
class LedgerEntry:
    """One discrete-commit decision (§3.8.3). Append-only except for the
    single deferred ``realised_dphi`` fill at the next step."""

    step: int
    zone: int
    devices: Tuple[str, ...]
    delta_int: Tuple[int, ...]
    predicted_dphi: float
    accepted: bool
    reason: str                 # 'accepted' | 'epsilon_reject' | 'slot_blocked' | 'integrator_commit'
    slot_owner: int
    epsilon_switch: float
    switch_cost: float
    realised_dphi: Optional[float] = None

    def __post_init__(self) -> None:
        allowed = {
            "accepted", "epsilon_reject", "slot_blocked",
            "integrator_commit",
        }
        if self.reason not in allowed:
            raise ValueError(
                f"reason '{self.reason}' not in {sorted(allowed)}"
            )
        if len(self.devices) != len(self.delta_int):
            raise ValueError(
                f"devices ({len(self.devices)}) and delta_int "
                f"({len(self.delta_int)}) must be aligned."
            )


class SwitchingLedger:
    """Append-only switching ledger (§3.8.3, premise data for §3.10.2)."""

    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> int:
        """Append and return the entry index (for the deferred fill)."""
        if not isinstance(entry, LedgerEntry):
            raise TypeError("append expects a LedgerEntry")
        self._entries.append(entry)
        return len(self._entries) - 1

    def fill_realised(self, index: int, realised_dphi: float) -> None:
        """One-time deferred fill of the realised ΔΦ (evaluated one step
        later from measurements)."""
        if not (0 <= int(index) < len(self._entries)):
            raise IndexError(
                f"ledger index {index} out of range "
                f"(len={len(self._entries)})"
            )
        e = self._entries[int(index)]
        if e.realised_dphi is not None:
            raise ValueError(
                f"ledger entry {index} already carries a realised ΔΦ — "
                "entries are append-only."
            )
        if not np.isfinite(float(realised_dphi)):
            raise ValueError("realised_dphi must be finite")
        e.realised_dphi = float(realised_dphi)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def to_records(self) -> List[Dict]:
        """Schema round-trip surface (list of plain dicts — feeds the
        Phase 6 parquet export)."""
        return [asdict(e) for e in self._entries]

    @classmethod
    def from_records(cls, records: List[Dict]) -> "SwitchingLedger":
        led = cls()
        for r in records:
            r = dict(r)
            r["devices"] = tuple(r["devices"])
            r["delta_int"] = tuple(int(d) for d in r["delta_int"])
            realised = r.pop("realised_dphi", None)
            entry = LedgerEntry(**r)
            led.append(entry)
            if realised is not None:
                led.fill_realised(len(led) - 1, float(realised))
        return led
