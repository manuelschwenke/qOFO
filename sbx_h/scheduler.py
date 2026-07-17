"""
sbx_h/scheduler.py
==================
SBX-H v6 cycle scheduler: metering, schedule application, hold/sag
support-energy settlement, escalation indicator.  NO runtime negotiation.

Per cycle boundary (every ``k_sched`` TSO iterations):

1. **Metering / settlement of the ELAPSED cycle** — cycle-averaged
   per-line P and terminal voltages, cycle-averaged corridor flow
   ``q_meas`` at the reference end A; one
   :class:`~sbx_h.settlement.CycleObservation` per corridor into the
   per-corridor :class:`~sbx_h.settlement.SettlementEngine`.
   Payment occurs only for correctly directed beyond-band reactive
   support when exactly one side violates and the other holds.
2. **Schedule application** — resolve the ACTIVE contract voltages and
   band at the boundary's scenario time (constant contract, hourly
   planning schedule, or planned-support interval — all through
   ``contract.v_std_at``/``q_band_at``) and return them as the frozen
   per-area terminal references for the STARTING cycle.  ``q_std`` is
   re-evaluated from the persistence ``p_sched`` (pure function; both
   sides compute it identically).
3. **Escalation indicator (candidate A4)** — an area whose violation
   flag, or a corridor whose beyond-band exceedance, persists for more
   than ``escalation_cycles`` consecutive boundaries is flagged for
   RE-PLANNING.  Recorded and reported; no runtime action is taken —
   updating the schedule is a planning-plane responsibility.

``record_step`` is called every TSO iteration (violation indicator +
cycle averaging).  Iteration 0 initialises the references at the
contract voltages.

The v5 deal layer that used to occupy Steps 2–5 (capability offers,
requests, matching, dv execution, unwind, delivery gate) was removed
on 2026-07-12 — evidence and rationale in STATUS_SBX.md (G1–G7) and
``docs/SBX_H_V6_ARCHITECTURE_CANDIDATES.md``; code archive in
``_archive/sbx_h_v5/``.

Author: Manuel Schwenke / Claude Code / OpenAI Codex
Date: 2026-07-13 (minimal SBX-H v6 settlement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sbx_h.config import SBXConfig
from sbx_h.contract import CorridorContract, q_std_mvar
from sbx_h.corridor import Corridor
from sbx_h.fail import rep1
from sbx_h.need import NeedDecision, NeedTracker
from sbx_h.settlement import CycleObservation, CycleSettlement, \
    SettlementEngine


@dataclass(frozen=True)
class AreaStepInput:
    """Per-TSO-step data of one area (feeds the violation indicator)."""

    bus_indices: Tuple[int, ...]
    v_meas_pu: Tuple[float, ...]
    v_min_pu: Tuple[float, ...]
    v_max_pu: Tuple[float, ...]


@dataclass
class CorridorCycleRecord:
    """Diagnostic record of one corridor's cycle (plots / evaluation)."""

    cycle: int
    iteration: int
    p_sched_mw: Tuple[float, ...]
    q_meas_mvar: float
    q_std_mvar: float
    q_band_mvar: float
    deviation_mvar: float          # q_meas − q_std of the ELAPSED cycle
    beyond_band: bool              # |deviation| > band (elapsed cycle)
    exceedance_run: int            # consecutive beyond-band boundaries
    escalation: bool               # exceedance_run > escalation_cycles
    need_a: bool
    need_b: bool
    support_state: str
    support_direction: Optional[str]
    a_sags: bool
    b_sags: bool
    violation_kind_a: Optional[str]
    violation_kind_b: Optional[str]
    a_holds: bool
    b_holds: bool
    support_mvar: float
    support_eur: float
    support_payer: Optional[int]
    support_payee: Optional[int]


@dataclass
class _CorridorState:
    """Mutable per-corridor state."""

    q_std_mvar: float = 0.0
    q_meas_mvar: float = 0.0
    p_sched_mw: Tuple[float, ...] = ()
    #: Active contract voltages / band of the RUNNING cycle (resolved
    #: from the schedule at each boundary; constants never change).
    v_std_a_act: Tuple[float, ...] = ()
    v_std_b_act: Tuple[float, ...] = ()
    q_band_act: float = 0.0
    refs_a: Dict[int, float] = field(default_factory=dict)
    refs_b: Dict[int, float] = field(default_factory=dict)
    exceedance_run: int = 0


class SBXScheduler:
    """Orchestrates metering, schedule application and settlement.

    Plant-agnostic: the runner feeds measurements via
    :meth:`record_step` and calls :meth:`run_cycle` at boundaries; the
    returned frozen references go into each zone's existing
    voltage-tracking mechanism.
    """

    def __init__(
        self,
        config: SBXConfig,
        corridors: Mapping[Tuple[int, int], Corridor],
        contracts: Mapping[Tuple[int, int], CorridorContract],
    ) -> None:
        if set(corridors.keys()) != set(contracts.keys()):
            rep1("corridors and contracts must cover the same keys",
                 corridors=sorted(corridors.keys()),
                 contracts=sorted(contracts.keys()))
        self.config = config
        self.corridors = dict(corridors)
        self.contracts = dict(contracts)
        for key, contract in self.contracts.items():
            contract.assert_matches(self.corridors[key])

        self.area_ids: List[int] = sorted(
            {a for key in corridors for a in key}
        )
        self.corridors_of_area: Dict[int, List[Tuple[int, int]]] = {
            z: sorted(k for k in corridors if z in k) for z in self.area_ids
        }
        self._need = {z: NeedTracker(config, z) for z in self.area_ids}
        self._need_last: Dict[int, Optional[NeedDecision]] = {
            z: None for z in self.area_ids
        }
        #: Consecutive FLAGGED cycle boundaries per area (A4 indicator).
        self._flag_run: Dict[int, int] = {z: 0 for z in self.area_ids}
        #: Areas whose violation flag persisted beyond escalation_cycles
        #: — ``[(cycle, area), ...]``.
        self.escalations: List[Tuple[int, int]] = []

        self._state = {key: _CorridorState() for key in corridors}
        self._p_samples: Dict[Tuple[int, int], List[Tuple[float, ...]]] = {
            key: [] for key in corridors
        }
        self._q_samples: Dict[Tuple[int, int], List[float]] = {
            key: [] for key in corridors
        }
        self._va_samples: Dict[Tuple[int, int], List[Tuple[float, ...]]] = {
            key: [] for key in corridors
        }
        self._vb_samples: Dict[Tuple[int, int], List[Tuple[float, ...]]] = {
            key: [] for key in corridors
        }
        self.settlement_engines: Dict[Tuple[int, int], SettlementEngine] = {
            key: SettlementEngine(corridors[key], contracts[key], config)
            for key in corridors
        }
        self.settlements: Dict[Tuple[int, int], List[CycleSettlement]] = {
            key: self.settlement_engines[key].settlements
            for key in corridors
        }
        self._last_iteration: Optional[int] = None
        self.records: Dict[Tuple[int, int], List[CorridorCycleRecord]] = {
            key: [] for key in corridors
        }
        self._initialised = False

    # ------------------------------------------------------------------
    #  Per-TSO-step feed
    # ------------------------------------------------------------------

    def is_cycle_boundary(self, iteration: int) -> bool:
        return iteration > 0 and iteration % self.config.k_sched == 0

    def initial_references(
        self, time_s: float = 0.0
    ) -> Dict[int, Dict[int, float]]:
        """Iteration-0 references: every corridor terminal at the
        schedule interval active at ``time_s``."""
        refs: Dict[int, Dict[int, float]] = {z: {} for z in self.area_ids}
        for key, corr in self.corridors.items():
            contract = self.contracts[key]
            st = self._state[key]
            va, vb = contract.v_std_at(time_s)
            st.v_std_a_act, st.v_std_b_act = va, vb
            st.q_band_act = contract.q_band_at(time_s)
            st.refs_a = {ln.bus_a: va[k]
                         for k, ln in enumerate(corr.lines)}
            st.refs_b = {ln.bus_b: vb[k]
                         for k, ln in enumerate(corr.lines)}
            refs[corr.area_a].update(st.refs_a)
            refs[corr.area_b].update(st.refs_b)
        self._initialised = True
        return refs

    def record_step(
        self,
        iteration: int,
        area_inputs: Mapping[int, AreaStepInput],
        tie_p_mw: Mapping[Tuple[int, int], Sequence[float]],
        tie_q_mvar: Mapping[Tuple[int, int], Sequence[float]],
        tie_v_a_pu: Mapping[Tuple[int, int], Sequence[float]],
        tie_v_b_pu: Mapping[Tuple[int, int], Sequence[float]],
    ) -> None:
        """Feed one TSO iteration: violation indicator + cycle averaging.

        ``tie_p_mw`` / ``tie_q_mvar`` are per-corridor, per-line values
        measured at the REFERENCE END A (export-positive), in corridor
        line order.  ``tie_v_a_pu`` / ``tie_v_b_pu`` are the measured
        terminal voltage magnitudes per line at the A / B end (the
        scheduled-reference settlement inputs)."""
        if not self._initialised:
            rep1("record_step before initial_references()",
                 iteration=iteration)
        if self._last_iteration is not None and \
                iteration != self._last_iteration + 1:
            rep1("record_step iterations must be consecutive",
                 iteration=iteration, last=self._last_iteration)
        self._last_iteration = int(iteration)

        if set(area_inputs.keys()) != set(self.area_ids):
            rep1("area_inputs must cover every SBX area",
                 expected=self.area_ids, got=sorted(area_inputs.keys()))
        for z, inp in area_inputs.items():
            self._need_last[z] = self._need[z].update(
                iteration, inp.bus_indices, inp.v_meas_pu,
                inp.v_min_pu, inp.v_max_pu,
            )
        for key, corr in self.corridors.items():
            p = tuple(float(x) for x in tie_p_mw[key])
            q = [float(x) for x in tie_q_mvar[key]]
            va = tuple(float(x) for x in tie_v_a_pu[key])
            vb = tuple(float(x) for x in tie_v_b_pu[key])
            if len(p) != corr.n_lines or len(q) != corr.n_lines or \
                    len(va) != corr.n_lines or len(vb) != corr.n_lines:
                rep1("tie measurement arity mismatch", corridor=key,
                     n_lines=corr.n_lines, n_p=len(p), n_q=len(q),
                     n_va=len(va), n_vb=len(vb))
            self._p_samples[key].append(p)
            self._q_samples[key].append(sum(q))
            self._va_samples[key].append(va)
            self._vb_samples[key].append(vb)

    # ------------------------------------------------------------------
    #  Cycle boundary
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        iteration: int,
        time_s: Optional[float] = None,
    ) -> Dict[int, Dict[int, float]]:
        """Settle the elapsed cycle, apply the active schedule, update
        the escalation indicator.

        Returns the frozen per-area references ``{area: {bus: v_ref}}``
        covering ALL corridor terminal buses.  ``time_s`` (scenario
        time of this boundary) is REQUIRED when any contract carries a
        schedule; it selects the active interval for the STARTING
        cycle."""
        if not self.is_cycle_boundary(iteration):
            rep1("run_cycle called off the cycle boundary",
                 iteration=iteration, k_sched=self.config.k_sched)
        if time_s is None and any(
                c.v_std_schedule is not None
                for c in self.contracts.values()):
            rep1("run_cycle needs the scenario time for schedule-bearing "
                 "contracts", iteration=iteration)
        t_act = float(time_s) if time_s is not None else 0.0
        cycle = iteration // self.config.k_sched

        # A4 indicator: advance per-area flagged runs at the boundary.
        for z in self.area_ids:
            need = self._need_last[z]
            if need is not None and need.flag:
                self._flag_run[z] += 1
                if self._flag_run[z] > self.config.escalation_cycles:
                    self.escalations.append((cycle, z))
            else:
                self._flag_run[z] = 0

        refs: Dict[int, Dict[int, float]] = {z: {} for z in self.area_ids}
        for key, corr in self.corridors.items():
            samples = self._p_samples[key]
            if len(samples) < self.config.k_sched:
                rep1("cycle boundary reached with too few P samples",
                     corridor=key, got=len(samples),
                     expected=self.config.k_sched)
            arr = np.asarray(samples[-self.config.k_sched:],
                             dtype=np.float64)
            st = self._state[key]
            first_boundary = (st.p_sched_mw == ())
            q_band_elapsed = st.q_band_act
            refs_a_elapsed = dict(st.refs_a)
            refs_b_elapsed = dict(st.refs_b)
            st.p_sched_mw = tuple(float(x) for x in arr.mean(axis=0))
            st.q_meas_mvar = float(np.mean(
                self._q_samples[key][-self.config.k_sched:]
            ))

            # Settle the ELAPSED cycle. The first boundary only creates
            # the persistence P estimate used by the starting cycle.
            settlement: Optional[CycleSettlement] = None
            if not first_boundary:
                va_avg = tuple(float(x) for x in np.mean(
                    self._va_samples[key][-self.config.k_sched:], axis=0))
                vb_avg = tuple(float(x) for x in np.mean(
                    self._vb_samples[key][-self.config.k_sched:], axis=0))
                settlement = self.settlement_engines[key].observe(
                    CycleObservation(
                        cycle=cycle - 1,
                        q_meas_mvar=st.q_meas_mvar,
                        v_meas_a_pu=va_avg,
                        v_meas_b_pu=vb_avg,
                        v_sched_a_pu=tuple(
                            refs_a_elapsed[ln.bus_a]
                            for ln in corr.lines
                        ),
                        v_sched_b_pu=tuple(
                            refs_b_elapsed[ln.bus_b]
                            for ln in corr.lines
                        ),
                        p_meas_mw=st.p_sched_mw,
                        q_band_mvar=q_band_elapsed,
                    )
                )

            deviation = (
                settlement.deviation_mvar
                if settlement is not None
                else float("nan")
            )
            beyond = (
                settlement is not None
                and abs(deviation) > q_band_elapsed
            )
            st.exceedance_run = st.exceedance_run + 1 if beyond else 0
            escalation = st.exceedance_run > self.config.escalation_cycles

            # ── Apply the ACTIVE schedule for the starting cycle. ──────
            st.v_std_a_act, st.v_std_b_act = \
                self.contracts[key].v_std_at(t_act)
            st.q_band_act = self.contracts[key].q_band_at(t_act)
            st.q_std_mvar = q_std_mvar(
                self.contracts[key], corr, st.p_sched_mw,
                time_s=t_act,
                delta_max_rad=self.config.delta_max_rad,
            )
            st.refs_a = {ln.bus_a: st.v_std_a_act[k]
                         for k, ln in enumerate(corr.lines)}
            st.refs_b = {ln.bus_b: st.v_std_b_act[k]
                         for k, ln in enumerate(corr.lines)}
            refs[corr.area_a].update(st.refs_a)
            refs[corr.area_b].update(st.refs_b)

            need_a = self._need_last[corr.area_a]
            need_b = self._need_last[corr.area_b]
            self.records[key].append(CorridorCycleRecord(
                cycle=cycle,
                iteration=iteration,
                p_sched_mw=st.p_sched_mw,
                q_meas_mvar=st.q_meas_mvar,
                q_std_mvar=(
                    settlement.q_baseline_mvar
                    if settlement is not None
                    else st.q_std_mvar
                ),
                q_band_mvar=(
                    q_band_elapsed
                    if settlement is not None
                    else st.q_band_act
                ),
                deviation_mvar=deviation,
                beyond_band=beyond,
                exceedance_run=st.exceedance_run,
                escalation=escalation,
                need_a=bool(need_a is not None and need_a.flag),
                need_b=bool(need_b is not None and need_b.flag),
                support_state=(
                    settlement.support_state
                    if settlement is not None
                    else "not_settled"
                ),
                support_direction=(
                    settlement.support_direction
                    if settlement is not None
                    else None
                ),
                a_sags=bool(
                    settlement is not None and settlement.a_sags
                ),
                b_sags=bool(
                    settlement is not None and settlement.b_sags
                ),
                violation_kind_a=(
                    settlement.violation_kind_a
                    if settlement is not None
                    else None
                ),
                violation_kind_b=(
                    settlement.violation_kind_b
                    if settlement is not None
                    else None
                ),
                a_holds=bool(
                    settlement is not None and settlement.a_holds
                ),
                b_holds=bool(
                    settlement is not None and settlement.b_holds
                ),
                support_mvar=(
                    settlement.support_mvar
                    if settlement is not None
                    else 0.0
                ),
                support_eur=(
                    settlement.support_eur
                    if settlement is not None
                    else 0.0
                ),
                support_payer=(
                    settlement.support_payer
                    if settlement is not None
                    else None
                ),
                support_payee=(
                    settlement.support_payee
                    if settlement is not None
                    else None
                ),
            ))

        # Averaging buffers restart for the next cycle.
        for key in self.corridors:
            self._p_samples[key].clear()
            self._q_samples[key].clear()
            self._va_samples[key].clear()
            self._vb_samples[key].clear()
        return refs

    # ------------------------------------------------------------------
    #  Views
    # ------------------------------------------------------------------

    def corridor_state(self, key: Tuple[int, int]) -> _CorridorState:
        if key not in self._state:
            rep1("unknown corridor key", key=key,
                 known=sorted(self._state.keys()))
        return self._state[key]

    def last_need(self, area: int) -> Optional[NeedDecision]:
        """Most recent violation-indicator decision of one area."""
        if area not in self._need_last:
            rep1("unknown SBX area", area=area, known=self.area_ids)
        return self._need_last[area]
