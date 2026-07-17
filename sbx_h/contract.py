"""
sbx_h/contract.py
=================
Agreed contract data per corridor (SBX-H v6).

The contract is the WHOLE mechanism's normative object: agreed terminal
voltages per tie line (controller intent, an explicit planning schedule,
or planned SUPPORT intervals), the implied standard flows, the
reactive-flow deadband, and the symmetric voltage-violation support-energy settlement terms.  v6 removed the deal-layer fields
(quantum rate, contract cap, dust threshold — archive:
``_archive/sbx_h_v5/``).

Planned support (v6, Manuel 2026-07-12): support agreed IN ADVANCE is a
schedule product — an interval during which one side holds a
deliberately RAISED (or lowered) boundary voltage for the neighbour,
e.g. "+2 mpu on B's terminals from minute 60 to 120".  Constructed with
:func:`with_planned_support`; the settlement automatically references
the raised schedule (the supporter is measured against its promise, and
the implied ``q_std`` carries the scheduled support flow).

Symbol map
----------
* ``v_std_a_pu`` / ``v_std_b_pu`` ↔ agreed terminal-voltage pair per tie
  line; bilateral data, neither side moves it unilaterally.
* ``q_band_mvar``                 ↔ support-flow deadband.
* ``p_support_eur_per_mvarh``     ↔ delivered support-energy price.
* ``v_hold_tolerance_pu`` /
  ``v_sag_threshold_pu``          ↔ absolute terminal-voltage roles.
* ``q_support_cap_mvar``          ↔ optional payment-exposure cap.
* ``k_sched`` / ``t_cycle_min``   ↔ metering cycle length.
* ``q_std_mvar()``                ↔ q_std = Σ_ℓ q_flow(v_std[ℓ,A],
  v_std[ℓ,B], p_sched[ℓ]) — a pure function both sides evaluate
  identically each cycle.

Contract immutability: frozen dataclass with scalar/tuple fields only.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-12 (SBX-H v6; v2 original 2026-07-07)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import pandapower as pp

from sbx_h.config import SBXConfig
from sbx_h.corridor import Corridor, corridor_q_flow
from sbx_h.fail import rep1

#: Decimal places of the default contract voltages.  Decision trail
#: (STATUS_SBX.md §1.3, Manuel 2026-07-07): the plan's 1e−3 pu shifted
#: q_std by up to 4 Mvar (|b| ≈ 40–75 pu → 4–7.5 Mvar per mpu of
#: voltage-difference error); 1e−4 pu (tried first) failed golden test 4
#: marginally on corridor (1,3); 1e−5 pu passes every corridor with
#: ≥ 85 % margin (worst deviation 0.067 Mvar).
V_STD_DECIMALS = 5


@dataclass(frozen=True)
class CorridorContract:
    """Frozen contract data of one corridor (v6).

    Per-line tuples are in the corridor's line order (ascending
    ``line_idx``); ``line_indices`` pins that order so any misalignment
    with a :class:`~sbx_h.corridor.Corridor` fails fast.
    """

    area_a: int
    area_b: int
    line_indices: Tuple[int, ...]
    v_std_a_pu: Tuple[float, ...]
    v_std_b_pu: Tuple[float, ...]
    q_band_mvar: float
    p_support_eur_per_mvarh: float
    q_support_cap_mvar: Optional[float]
    v_hold_tolerance_pu: float
    v_sag_threshold_pu: float
    k_sched: int
    t_cycle_min: float
    v_std_schedule: Optional[
        Tuple[Tuple[float, Tuple[float, ...], Tuple[float, ...]], ...]
    ] = None
    """Optional contract-voltage SCHEDULE: ordered ``(t_from_s,
    v_std_a_pu, v_std_b_pu)`` intervals in scenario time — the planning
    pre-pass output (v3) and/or planned-support intervals (v6,
    :func:`with_planned_support`).  The first interval must start at
    0 s; the constant ``v_std_*_pu`` fields must equal the first
    interval (the t = 0 view).  ``None`` keeps the constant-snapshot
    semantics."""

    q_band_schedule: Optional[Tuple[Tuple[float, float], ...]] = None
    """Optional planning-derived tier-1 band schedule: ordered
    ``(t_from_s, q_band_mvar)`` intervals (same rules as
    ``v_std_schedule``); the constant ``q_band_mvar`` must equal the
    first interval.  ``None`` keeps the constant band."""

    def __post_init__(self) -> None:
        if self.area_a >= self.area_b:
            rep1("CorridorContract requires area_a < area_b",
                 area_a=self.area_a, area_b=self.area_b)
        n = len(self.line_indices)
        if n == 0:
            rep1("CorridorContract must cover at least one tie line",
                 corridor=(self.area_a, self.area_b))
        if len(self.v_std_a_pu) != n or len(self.v_std_b_pu) != n:
            rep1("contract voltage tuples must align with line_indices",
                 corridor=(self.area_a, self.area_b),
                 n_lines=n, n_v_a=len(self.v_std_a_pu),
                 n_v_b=len(self.v_std_b_pu))
        for name, values in (("v_std_a_pu", self.v_std_a_pu),
                             ("v_std_b_pu", self.v_std_b_pu)):
            for v in values:
                if not (math.isfinite(v) and v > 0.0):
                    rep1("contract voltages must be finite and positive",
                         corridor=(self.area_a, self.area_b),
                         field=name, values=values)
        for name in ("q_band_mvar", "p_support_eur_per_mvarh",
                     "t_cycle_min"):
            if getattr(self, name) <= 0.0:
                rep1(f"contract field {name} must be positive",
                     corridor=(self.area_a, self.area_b),
                     **{name: getattr(self, name)})
        if self.q_support_cap_mvar is not None and \
                self.q_support_cap_mvar <= 0.0:
            rep1("q_support_cap_mvar must be positive when set",
                 corridor=(self.area_a, self.area_b),
                 q_support_cap_mvar=self.q_support_cap_mvar)
        if not (0.0 <= self.v_hold_tolerance_pu
                < self.v_sag_threshold_pu):
            rep1("contract hold/sag thresholds must satisfy "
                 "0 <= hold < sag",
                 corridor=(self.area_a, self.area_b),
                 v_hold_tolerance_pu=self.v_hold_tolerance_pu,
                 v_sag_threshold_pu=self.v_sag_threshold_pu)
        if self.k_sched < 1:
            rep1("k_sched must be a positive iteration count",
                 corridor=(self.area_a, self.area_b), k_sched=self.k_sched)

        if self.v_std_schedule is not None:
            n = len(self.line_indices)
            if len(self.v_std_schedule) == 0:
                rep1("v_std_schedule must not be empty (use None for the "
                     "constant-snapshot contract)",
                     corridor=(self.area_a, self.area_b))
            prev_t = None
            for entry in self.v_std_schedule:
                if len(entry) != 3:
                    rep1("v_std_schedule entries must be "
                         "(t_from_s, v_std_a, v_std_b)",
                         corridor=(self.area_a, self.area_b), entry=entry)
                t_from, va, vb = entry
                if not math.isfinite(t_from):
                    rep1("v_std_schedule t_from_s must be finite",
                         corridor=(self.area_a, self.area_b), entry=entry)
                if prev_t is None and t_from != 0.0:
                    rep1("v_std_schedule must start at t = 0 s (total "
                         "coverage of the scenario)",
                         corridor=(self.area_a, self.area_b),
                         first_t_from_s=t_from)
                if prev_t is not None and t_from <= prev_t:
                    rep1("v_std_schedule intervals must be strictly "
                         "ascending", corridor=(self.area_a, self.area_b),
                         t_from_s=t_from, previous=prev_t)
                prev_t = t_from
                if len(va) != n or len(vb) != n:
                    rep1("v_std_schedule entry arity must match the "
                         "corridor's line list",
                         corridor=(self.area_a, self.area_b),
                         t_from_s=t_from, n_lines=n,
                         n_a=len(va), n_b=len(vb))
                for v in tuple(va) + tuple(vb):
                    if not (math.isfinite(v) and v > 0.0):
                        rep1("v_std_schedule voltages must be finite and "
                             "positive",
                             corridor=(self.area_a, self.area_b),
                             t_from_s=t_from)
            t0, va0, vb0 = self.v_std_schedule[0]
            if (tuple(va0) != tuple(self.v_std_a_pu)
                    or tuple(vb0) != tuple(self.v_std_b_pu)):
                rep1("the constant v_std fields must equal the schedule's "
                     "first interval (the t = 0 view)",
                     corridor=(self.area_a, self.area_b),
                     v_std_a_pu=self.v_std_a_pu, schedule_t0_a=va0)
        if self.q_band_schedule is not None:
            if len(self.q_band_schedule) == 0:
                rep1("q_band_schedule must not be empty (use None for "
                     "the constant band)",
                     corridor=(self.area_a, self.area_b))
            prev_t = None
            for entry in self.q_band_schedule:
                if len(entry) != 2:
                    rep1("q_band_schedule entries must be "
                         "(t_from_s, q_band_mvar)",
                         corridor=(self.area_a, self.area_b), entry=entry)
                t_from, band = entry
                if not (math.isfinite(t_from) and math.isfinite(band)
                        and band > 0.0):
                    rep1("q_band_schedule needs finite times and positive "
                         "bands", corridor=(self.area_a, self.area_b),
                         entry=entry)
                if prev_t is None and t_from != 0.0:
                    rep1("q_band_schedule must start at t = 0 s",
                         corridor=(self.area_a, self.area_b),
                         first_t_from_s=t_from)
                if prev_t is not None and t_from <= prev_t:
                    rep1("q_band_schedule intervals must be strictly "
                         "ascending",
                         corridor=(self.area_a, self.area_b),
                         t_from_s=t_from, previous=prev_t)
                prev_t = t_from
            if self.q_band_schedule[0][1] != self.q_band_mvar:
                rep1("the constant q_band_mvar must equal the band "
                     "schedule's first interval (the t = 0 view)",
                     corridor=(self.area_a, self.area_b),
                     q_band_mvar=self.q_band_mvar,
                     schedule_t0=self.q_band_schedule[0][1])

    @property
    def n_lines(self) -> int:
        return len(self.line_indices)

    def v_std_at(
        self, time_s: float
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Active contract-voltage pair at scenario time ``time_s``.

        Piecewise-constant lookup over ``v_std_schedule``; without a
        schedule the constant fields apply at every time.  Negative
        times are a caller error.
        """
        if not math.isfinite(time_s) or time_s < 0.0:
            rep1("v_std_at needs a finite non-negative scenario time",
                 corridor=(self.area_a, self.area_b), time_s=time_s)
        if self.v_std_schedule is None:
            return self.v_std_a_pu, self.v_std_b_pu
        active = self.v_std_schedule[0]
        for entry in self.v_std_schedule:
            if entry[0] <= time_s:
                active = entry
            else:
                break
        return tuple(active[1]), tuple(active[2])

    def q_band_at(self, time_s: float) -> float:
        """Active tier-1 band half-width at scenario time ``time_s``."""
        if not math.isfinite(time_s) or time_s < 0.0:
            rep1("q_band_at needs a finite non-negative scenario time",
                 corridor=(self.area_a, self.area_b), time_s=time_s)
        if self.q_band_schedule is None:
            return self.q_band_mvar
        band = self.q_band_schedule[0][1]
        for t_from, b in self.q_band_schedule:
            if t_from <= time_s:
                band = b
            else:
                break
        return float(band)

    def assert_matches(self, corridor: Corridor) -> None:
        """Fail fast unless the contract covers exactly this corridor."""
        if (corridor.area_a, corridor.area_b) != (self.area_a, self.area_b):
            rep1("contract and corridor cover different area pairs",
                 contract=(self.area_a, self.area_b),
                 corridor=(corridor.area_a, corridor.area_b))
        corr_lines = tuple(ln.line_idx for ln in corridor.lines)
        if corr_lines != self.line_indices:
            rep1("contract and corridor line lists differ",
                 contract_lines=self.line_indices,
                 corridor_lines=corr_lines)


def build_default_contract(
    corridor: Corridor,
    net: pp.pandapowerNet,
    config: SBXConfig,
    *,
    v_std_schedule: Optional[Sequence] = None,
    q_band_schedule: Optional[Sequence] = None,
) -> CorridorContract:
    """Low-level builder from an explicit schedule or snapshot.

    The active runner adapter always supplies controller intent or an
    explicit planning schedule. The snapshot fallback remains only for
    isolated analytical tests and explicit low-level callers; it is not
    the active control-reference default.
    """
    if v_std_schedule is not None:
        sched = tuple(
            (float(t), tuple(round(float(v), V_STD_DECIMALS) for v in va),
             tuple(round(float(v), V_STD_DECIMALS) for v in vb))
            for t, va, vb in v_std_schedule
        )
        v_std_a = list(sched[0][1])
        v_std_b = list(sched[0][2])
    else:
        sched = None
        if not hasattr(net, "res_bus") or len(net.res_bus) != len(net.bus):
            rep1("build_default_contract needs a converged power flow on "
                 "the base-case net (net.res_bus incomplete)",
                 corridor=(corridor.area_a, corridor.area_b),
                 n_res_bus=len(getattr(net, "res_bus", ())),
                 n_bus=len(net.bus))
        v_std_a, v_std_b = [], []
        for ln in corridor.lines:
            v_a = float(net.res_bus.at[ln.bus_a, "vm_pu"])
            v_b = float(net.res_bus.at[ln.bus_b, "vm_pu"])
            if not (math.isfinite(v_a) and math.isfinite(v_b)):
                rep1("base-case terminal voltage is non-finite",
                     line_idx=ln.line_idx, v_a=v_a, v_b=v_b)
            v_std_a.append(round(v_a, V_STD_DECIMALS))
            v_std_b.append(round(v_b, V_STD_DECIMALS))

    band_sched = None
    q_band = config.q_band_mvar
    if q_band_schedule is not None:
        band_sched = tuple(
            (float(t), float(b)) for t, b in q_band_schedule
        )
        q_band = band_sched[0][1]

    return CorridorContract(
        v_std_schedule=sched,
        q_band_schedule=band_sched,
        area_a=corridor.area_a,
        area_b=corridor.area_b,
        line_indices=tuple(ln.line_idx for ln in corridor.lines),
        v_std_a_pu=tuple(v_std_a),
        v_std_b_pu=tuple(v_std_b),
        q_band_mvar=q_band,
        p_support_eur_per_mvarh=config.p_support_eur_per_mvarh,
        q_support_cap_mvar=config.q_support_cap_mvar,
        v_hold_tolerance_pu=config.v_hold_tolerance_pu,
        v_sag_threshold_pu=config.v_sag_threshold_pu,
        k_sched=config.k_sched,
        t_cycle_min=config.t_cycle_min,
    )


def with_planned_support(
    contract: CorridorContract,
    t_from_s: float,
    t_to_s: float,
    *,
    dv_a_pu: float = 0.0,
    dv_b_pu: float = 0.0,
    horizon_s: Optional[float] = None,
) -> CorridorContract:
    """Planned support agreed IN ADVANCE (v6, Manuel 2026-07-12).

    Returns a new contract whose voltage schedule holds side A's / B's
    terminals SHIFTED by ``dv_a_pu`` / ``dv_b_pu`` during
    ``[t_from_s, t_to_s)`` — e.g. "the neighbour holds +2 mpu from
    minute 60 to 120".  Outside the interval the previous schedule (or
    the constant contract) applies unchanged; after ``t_to_s`` the
    pre-interval values resume.

    The shift composes with an existing schedule interval-wise (every
    base interval overlapping the support window is split as needed),
    so planning schedules and support windows stack.  Settlement
    consequences are automatic: ``q_std`` carries the scheduled support
    flow, and the supporter is measured against its RAISED promise.
    """
    if not (math.isfinite(t_from_s) and math.isfinite(t_to_s)
            and 0.0 <= t_from_s < t_to_s):
        rep1("planned support needs 0 <= t_from_s < t_to_s",
             corridor=(contract.area_a, contract.area_b),
             t_from_s=t_from_s, t_to_s=t_to_s)
    if not (math.isfinite(dv_a_pu) and math.isfinite(dv_b_pu)):
        rep1("planned-support shifts must be finite",
             dv_a_pu=dv_a_pu, dv_b_pu=dv_b_pu)
    if dv_a_pu == 0.0 and dv_b_pu == 0.0:
        rep1("planned support needs a nonzero shift on at least one side",
             corridor=(contract.area_a, contract.area_b))
    if horizon_s is not None and t_to_s > horizon_s:
        rep1("planned-support window exceeds the scenario horizon",
             t_to_s=t_to_s, horizon_s=horizon_s)

    base = contract.v_std_schedule
    if base is None:
        base = ((0.0, contract.v_std_a_pu, contract.v_std_b_pu),)

    def _shift(va, vb):
        return (tuple(round(v + dv_a_pu, V_STD_DECIMALS) for v in va),
                tuple(round(v + dv_b_pu, V_STD_DECIMALS) for v in vb))

    # Piecewise composition: walk the base intervals; inside the
    # support window emit shifted values, splitting at the window edges.
    events = sorted({t for t, _, _ in base} | {t_from_s, t_to_s})
    out = []
    for t in events:
        # Base values active at t.
        va, vb = base[0][1], base[0][2]
        for tb, a, b in base:
            if tb <= t:
                va, vb = a, b
            else:
                break
        if t_from_s <= t < t_to_s:
            va, vb = _shift(va, vb)
        if out and out[-1][1] == tuple(va) and out[-1][2] == tuple(vb):
            continue                       # merge identical neighbours
        out.append((float(t), tuple(va), tuple(vb)))
    if out[0][0] != 0.0:
        rep1("composed support schedule must start at t = 0",
             first=out[0][0])

    return replace(
        contract,
        v_std_schedule=tuple(out),
        v_std_a_pu=out[0][1],
        v_std_b_pu=out[0][2],
    )


def q_std_mvar(
    contract: CorridorContract,
    corridor: Corridor,
    p_sched_mw: Sequence[float],
    *,
    time_s: Optional[float] = None,
    delta_max_rad: float = 0.6,
) -> float:
    """Standard flow [Mvar] at the reference end A.

    ``q_std = Σ_ℓ q_flow(v_std[ℓ,A], v_std[ℓ,B], p_sched[ℓ])`` — a pure
    function of the frozen contract and the cycle's ``p_sched``; both
    sides evaluate it identically (computed, never communicated).
    With a schedule on the contract, ``time_s`` is REQUIRED and selects
    the active contract voltages; there is no silent constant fallback.
    """
    contract.assert_matches(corridor)
    if contract.v_std_schedule is not None and time_s is None:
        rep1("q_std_mvar needs the scenario time for a schedule-bearing "
             "contract — no silent constant fallback",
             corridor=(contract.area_a, contract.area_b))
    v_a, v_b = contract.v_std_at(time_s if time_s is not None else 0.0)
    return corridor_q_flow(
        corridor,
        list(v_a),
        list(v_b),
        list(p_sched_mw),
        delta_max_rad=delta_max_rad,
    )
