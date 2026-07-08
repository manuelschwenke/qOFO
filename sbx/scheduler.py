"""
sbx/scheduler.py
================
Six-step SBX cycle scheduler (plan v2 §2.2, amended v2.2: ALL corridors
execute Steps 1–5 in parallel every cycle; joint feasibility is
guaranteed upstream in ``sbx.capability``).

Schedule representation (documented interpretation of §2.1/§2.2)
----------------------------------------------------------------
The persistent per-corridor state is the SURPLUS ``s = q_sched − q_std``
(the accumulated, deliberate deal balance).  ``q_std`` is re-evaluated
every cycle from the persistence forecast ``p_sched`` (Step 1), so the
absolute schedule is ``q_sched(cycle) = q_std(cycle) + s``.  Every plan
formula holds verbatim under this reading (``q_sched += dq_deal`` ⇔
``s += dq_deal`` since ``q_std`` is a common term); the deviation
settlement (tiers 1/3) and the acting side (Step 4) see exactly the
values the plan prescribes, while pure P drift never masquerades as a
deal.

Cycle timing: one cycle = ``k_sched`` TSO OFO iterations (Phase 0: 5 ×
3 min = 15 min).  ``record_step`` is called every TSO iteration (need
trackers + cycle averaging); ``run_cycle`` fires at iterations
``c · k_sched`` (c ≥ 1) and consumes the elapsed cycle's averages.
Iteration 0 initialises the references at the contract voltages.

Step map (code ↔ plan §2.2)
---------------------------
* Step 1 ↔ ``p_sched`` = per-line cycle-averaged measured P at the
  reference end (previous cycle, persistence); ``q_std`` via
  ``sbx.contract.q_std_mvar``.
* Step 2 ↔ ``PeerCairMessage`` per area (offer from the v2.2 joint-box
  LP, request iff the need flag is set, ``p_sched`` from end A).
* Step 3 ↔ ``sbx.matching.match`` + deal-record checksum comparison of
  two independent evaluations (both sides compute; in-process
  simulation of the exchange).
* Step 4 ↔ acting side = sign(s); common ``dv`` from
  ``corridor_solve_dv``; far end holds ``v_std``; INVARIANT asserted
  every cycle: at most one deviating end per corridor.
* Step 5 ↔ unwind one quantum towards ``q_std`` (clamped at zero, paid
  surplus first) when every requester-of-record's need flag has been
  clear for ``m_release`` consecutive cycles; accumulators reset at
  surplus zero-crossings.  A cycle executes AT MOST ONE schedule update
  per corridor (deal XOR unwind).
* Step 6 ↔ the returned references are frozen until the next cycle
  (the runner writes them once per cycle into each area's existing
  tracked-output mechanism, weight ``w_track`` ≡ the zone's ``g_v``).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from optimisation.miqp_solver import MIQPSolver
from sbx.capability import CapabilityResult, CorridorCoupling, \
    joint_box_capability
from sbx.config import SBXConfig
from sbx.contract import CorridorContract, q_std_mvar
from sbx.corridor import Corridor, corridor_sensitivities, corridor_solve_dv
from sbx.fail import rep1
from sbx.matching import DealRecord, KIND_SCARCITY, match
from sbx.messages import SBX_MESSAGE_VERSION, PeerCairMessage, \
    assert_checksums_match
from sbx.need import NeedDecision, NeedTracker, assert_relieving_sign
from sbx.settlement import CycleObservation, CycleSettlement, \
    SettlementEngine

#: |dv| below this is "no deviation" for the Step-4 invariant assert.
DV_ZERO_TOL_PU = 1e-9

#: Contract-consistency classification of the ELAPSED cycle (proposal
#: recorded 2026-07-07): does the measured corridor response carry the
#: expected sign/magnitude relative to the schedule that was active?
#: Classified, never aborted — the standing tracking-residual offset on
#: stiff ties (mpu-scale voltage errors × |b| ≈ 40–75 Mvar/mpu) makes
#: hard sign asserts false-positive-prone; the classification protects
#: against tie-line-model sign errors and terminal-reference mix-ups by
#: making them VISIBLE (smoke test / Phase 7 report on the counts).
CONSISTENCY_NA = "n/a"                    # first boundary: no schedule yet
CONSISTENCY_NO_SURPLUS = "no_surplus"     # |s| ≤ dust: nothing to verify
CONSISTENCY_DEADBAND = "deadband"         # |q_meas − q_std| ≤ q_band: noise
CONSISTENCY_CONSISTENT = "consistent"     # sign matches, magnitude sane
CONSISTENCY_SIGN_MISMATCH = "sign_mismatch"
CONSISTENCY_MAGNITUDE_OFF = "magnitude_off"

#: |dev| / |surplus| band accepted as "approximate magnitude" for the
#: consistency classification (outside → magnitude_off).
CONSISTENCY_MAG_BAND = (0.25, 4.0)


@dataclass(frozen=True)
class AreaStepInput:
    """Per-TSO-step data of one area (feeds the need tracker)."""

    bus_indices: Tuple[int, ...]
    v_meas_pu: Tuple[float, ...]
    v_min_pu: Tuple[float, ...]
    v_max_pu: Tuple[float, ...]


@dataclass(frozen=True)
class AreaCycleData:
    """Per-cycle LOCAL data of one area for the joint-box capability LP.

    ``h_loc`` rows align with ``v_bus_indices``; ``terminal_h_rows`` maps
    a corridor terminal bus to its cached ∂v_bus/∂u row (used to compose
    the corridor control rows).  ``dv_dq_import_by_corridor`` is the
    local-model sensitivity of the area's WORST-violated bus voltage per
    Mvar of Q imported through that corridor (plan §2.3 sanity assert);
    the adapter computes it from the cached reduced Jacobian.
    """

    u_now: NDArray[np.float64]
    u_min: NDArray[np.float64]
    u_max: NDArray[np.float64]
    v_bus_indices: Tuple[int, ...]
    v_meas_pu: NDArray[np.float64]
    v_min_pu: NDArray[np.float64]
    v_max_pu: NDArray[np.float64]
    h_loc: NDArray[np.float64]
    terminal_h_rows: Mapping[int, NDArray[np.float64]]
    dv_dq_import_by_corridor: Mapping[Tuple[int, int], float]


@dataclass
class CorridorCycleRecord:
    """Diagnostic record of one corridor's cycle (Phase 7 plotting)."""

    cycle: int
    iteration: int
    p_sched_mw: Tuple[float, ...]
    q_meas_mvar: float
    q_std_mvar: float
    q_sched_mvar: float
    consistency: str
    surplus_mvar: float
    surplus_paid_mvar: float
    surplus_unpaid_mvar: float
    deal: DealRecord
    unwound_mvar: float
    acting_area: Optional[int]
    dv_pu: float
    need_a: bool
    need_b: bool
    offer_a: Tuple[float, float]
    offer_b: Tuple[float, float]
    t_a: float
    t_b: float


@dataclass
class _CorridorState:
    """Mutable per-corridor protocol state."""

    surplus_mvar: float = 0.0
    surplus_paid_mvar: float = 0.0
    surplus_unpaid_mvar: float = 0.0
    areas_of_record: set = field(default_factory=set)
    release_counter: int = 0
    q_std_mvar: float = 0.0
    q_meas_mvar: float = 0.0
    consistency_last: str = CONSISTENCY_NA
    p_sched_mw: Tuple[float, ...] = ()
    #: Active contract voltages of the running cycle (v3: resolved from
    #: the schedule at each boundary; constant contracts never change).
    v_std_a_act: Tuple[float, ...] = ()
    v_std_b_act: Tuple[float, ...] = ()
    refs_a: Dict[int, float] = field(default_factory=dict)
    refs_b: Dict[int, float] = field(default_factory=dict)


class SBXScheduler:
    """Orchestrates the SBX cycle over all corridors of the partition.

    The scheduler is plant-agnostic: the runner feeds measurements via
    :meth:`record_step` and per-area cycle data via :meth:`run_cycle`,
    and writes the returned frozen references into each zone's existing
    voltage-tracking mechanism.  Both areas of a corridor are simulated
    in-process (STATUS_SBX.md A9); the two-sided determinism of Step 3
    is still exercised through independent double evaluation plus the
    checksum comparison.
    """

    def __init__(
        self,
        config: SBXConfig,
        corridors: Mapping[Tuple[int, int], Corridor],
        contracts: Mapping[Tuple[int, int], CorridorContract],
        solver: Optional[MIQPSolver] = None,
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
        self.solver = solver if solver is not None else MIQPSolver()

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
        self._state = {key: _CorridorState() for key in corridors}
        # Per-corridor per-step P/Q/terminal-V samples of the RUNNING
        # cycle (reference-end measurements, corridor line order).
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
        # §2.5 settlement: one engine per corridor, fed the elapsed
        # cycle's averages at every boundary (Phase 6).
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
        self.scarcity_events: List[DealRecord] = []
        self._initialised = False

    # ------------------------------------------------------------------
    #  Per-TSO-step feed
    # ------------------------------------------------------------------

    def is_cycle_boundary(self, iteration: int) -> bool:
        return iteration > 0 and iteration % self.config.k_sched == 0

    def initial_references(
        self, time_s: float = 0.0
    ) -> Dict[int, Dict[int, float]]:
        """Iteration-0 references: every corridor terminal at v_std.

        v3: ``time_s`` (scenario time of the contract-freeze tick)
        selects the active interval of schedule-bearing contracts.
        """
        refs: Dict[int, Dict[int, float]] = {z: {} for z in self.area_ids}
        for key, corr in self.corridors.items():
            contract = self.contracts[key]
            st = self._state[key]
            va, vb = contract.v_std_at(time_s)
            st.v_std_a_act, st.v_std_b_act = va, vb
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
        """Feed one TSO iteration: need trackers + cycle averaging.

        ``tie_p_mw`` / ``tie_q_mvar`` are per-corridor, per-line values
        measured at the REFERENCE END A (export-positive), in corridor
        line order.  ``tie_v_a_pu`` / ``tie_v_b_pu`` are the measured
        terminal voltage magnitudes per line at the A / B end (the §2.5
        settlement's attribution inputs).
        """
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
    #  Cycle protocol
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        iteration: int,
        cycle_data: Mapping[int, AreaCycleData],
        time_s: Optional[float] = None,
    ) -> Dict[int, Dict[int, float]]:
        """Execute Steps 1–6 for every corridor in parallel (v2.2).

        Returns the frozen per-area references ``{area: {bus: v_ref}}``
        covering ALL corridor terminal buses; the runner writes them into
        the zones' tracked-output mechanisms and leaves them untouched
        until the next boundary (Step 6).

        v3: ``time_s`` (scenario time of this boundary) is REQUIRED when
        any contract carries a planning schedule; it selects the active
        contract voltages for the STARTING cycle.
        """
        if not self.is_cycle_boundary(iteration):
            rep1("run_cycle called off the cycle boundary",
                 iteration=iteration, k_sched=self.config.k_sched)
        if set(cycle_data.keys()) != set(self.area_ids):
            rep1("cycle_data must cover every SBX area",
                 expected=self.area_ids, got=sorted(cycle_data.keys()))
        if time_s is None and any(
                c.v_std_schedule is not None
                for c in self.contracts.values()):
            rep1("run_cycle needs the scenario time for schedule-bearing "
                 "contracts (v3)", iteration=iteration)
        t_act = float(time_s) if time_s is not None else 0.0
        cycle = iteration // self.config.k_sched

        # ── Step 1: p_sched (persistence) and q_std per corridor ────────
        for key, corr in self.corridors.items():
            samples = self._p_samples[key]
            if len(samples) < self.config.k_sched:
                rep1("cycle boundary reached with too few P samples",
                     corridor=key, got=len(samples),
                     expected=self.config.k_sched)
            arr = np.asarray(samples[-self.config.k_sched:],
                             dtype=np.float64)
            st = self._state[key]
            # Elapsed cycle's active schedule BEFORE it is re-evaluated
            # (first boundary: no schedule existed yet — p_sched empty).
            first_boundary = (st.p_sched_mw == ())
            q_std_elapsed = st.q_std_mvar
            surplus_elapsed = st.surplus_mvar
            p_sched_elapsed = st.p_sched_mw
            refs_a_elapsed = dict(st.refs_a)
            refs_b_elapsed = dict(st.refs_b)
            st.p_sched_mw = tuple(float(x) for x in arr.mean(axis=0))
            # Cycle-averaged measured corridor Q (reference end A) of the
            # ELAPSED cycle — the settlement quantity (§2.5) and the
            # q_sched-vs-q_meas plot series (Phase 7).
            st.q_meas_mvar = float(np.mean(
                self._q_samples[key][-self.config.k_sched:]
            ))
            st.consistency_last = self._classify_consistency(
                self.contracts[key], first_boundary,
                st.q_meas_mvar, q_std_elapsed, surplus_elapsed,
            )
            # ── §2.5 settlement of the ELAPSED cycle (Phase 6): the
            # schedule, references (incl. the acting-side dv) and
            # paid/unpaid split that were ACTIVE during the elapsed
            # cycle, against its cycle-averaged measurements.  First
            # boundary: no schedule existed yet — nothing to settle.
            if not first_boundary:
                va_avg = tuple(float(x) for x in np.mean(
                    self._va_samples[key][-self.config.k_sched:], axis=0))
                vb_avg = tuple(float(x) for x in np.mean(
                    self._vb_samples[key][-self.config.k_sched:], axis=0))
                acting_end: Optional[str] = None
                if surplus_elapsed > 0.0:
                    acting_end = "a"
                elif surplus_elapsed < 0.0:
                    acting_end = "b"
                self.settlement_engines[key].observe(CycleObservation(
                    cycle=cycle - 1,
                    q_meas_mvar=st.q_meas_mvar,
                    q_std_mvar=q_std_elapsed,
                    surplus_mvar=surplus_elapsed,
                    surplus_paid_mvar=st.surplus_paid_mvar,
                    surplus_unpaid_mvar=st.surplus_unpaid_mvar,
                    acting_end=acting_end,
                    v_meas_a_pu=va_avg,
                    v_meas_b_pu=vb_avg,
                    v_sched_a_pu=tuple(refs_a_elapsed[ln.bus_a]
                                       for ln in corr.lines),
                    v_sched_b_pu=tuple(refs_b_elapsed[ln.bus_b]
                                       for ln in corr.lines),
                    p_meas_mw=st.p_sched_mw,
                    p_sched_mw=p_sched_elapsed,
                ))
            # v3: resolve the ACTIVE contract voltages of the starting
            # cycle (piecewise-constant schedule lookup; constant
            # contracts return their fixed pair).
            st.v_std_a_act, st.v_std_b_act = \
                self.contracts[key].v_std_at(t_act)
            st.q_std_mvar = q_std_mvar(
                self.contracts[key], corr, st.p_sched_mw,
                time_s=t_act,
                delta_max_rad=self.config.delta_max_rad,
            )
        # Averaging buffers restart for the next cycle.
        for key in self.corridors:
            self._p_samples[key].clear()
            self._q_samples[key].clear()
            self._va_samples[key].clear()
            self._vb_samples[key].clear()

        # ── Step 2: capability per area, then one message per direction ─
        capability: Dict[int, CapabilityResult] = {
            z: self._area_capability(z, cycle_data[z])
            for z in self.area_ids
        }

        refs: Dict[int, Dict[int, float]] = {z: {} for z in self.area_ids}
        for key, corr in self.corridors.items():
            self._run_corridor_cycle(
                key, corr, cycle, iteration, capability, cycle_data, refs,
            )
        return refs

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_consistency(
        contract: CorridorContract,
        first_boundary: bool,
        q_meas: float,
        q_std_elapsed: float,
        surplus_elapsed: float,
    ) -> str:
        """Contract-consistency classification of the elapsed cycle.

        Verifies sign(q_meas − q_std) == sign(q_sched − q_std) and an
        approximate magnitude whenever a surplus was actually scheduled;
        tiny deviations are classified as deadband/noise, never aborted
        (see the CONSISTENCY_* docstring).
        """
        if first_boundary:
            return CONSISTENCY_NA
        if abs(surplus_elapsed) <= contract.dq_min_deal_mvar:
            return CONSISTENCY_NO_SURPLUS
        dev = q_meas - q_std_elapsed
        if abs(dev) <= contract.q_band_mvar:
            return CONSISTENCY_DEADBAND
        if dev * surplus_elapsed < 0.0:
            return CONSISTENCY_SIGN_MISMATCH
        ratio = abs(dev) / abs(surplus_elapsed)
        lo, hi = CONSISTENCY_MAG_BAND
        if not (lo <= ratio <= hi):
            return CONSISTENCY_MAGNITUDE_OFF
        return CONSISTENCY_CONSISTENT

    def _area_capability(
        self, area: int, data: AreaCycleData
    ) -> CapabilityResult:
        """v2.2 D13: one joint-box LP per area over its corridor set."""
        couplings: List[CorridorCoupling] = []
        for key in self.corridors_of_area[area]:
            corr = self.corridors[key]
            contract = self.contracts[key]
            own_end = "a" if area == corr.area_a else "b"
            per_line, _sa, _sb = corridor_sensitivities(
                corr,
                list(self._state[key].v_std_a_act),
                list(self._state[key].v_std_b_act),
                list(self._state[key].p_sched_mw),
                delta_max_rad=self.config.delta_max_rad,
            )
            n_u = data.u_now.size
            row = np.zeros(n_u, dtype=np.float64)
            for k, ln in enumerate(corr.lines):
                bus = ln.bus_a if own_end == "a" else ln.bus_b
                if bus not in data.terminal_h_rows:
                    rep1("terminal_h_rows lacks a corridor terminal bus",
                         area=area, corridor=key, bus=bus)
                h_row = np.asarray(data.terminal_h_rows[bus],
                                   dtype=np.float64)
                if h_row.size != n_u:
                    rep1("terminal H row length mismatch", area=area,
                         bus=bus, got=h_row.size, n_u=n_u)
                s_own = per_line[k][0] if own_end == "a" else per_line[k][1]
                row += s_own * h_row
            couplings.append(CorridorCoupling(
                key=key, control_row=row,
                dq_quant_mvar=contract.dq_quant_mvar,
            ))
        return joint_box_capability(
            data.u_now, data.u_min, data.u_max,
            data.v_meas_pu, data.v_min_pu, data.v_max_pu,
            data.h_loc, couplings, self.solver,
            voltage_margin_pu=self.config.voltage_margin_pu,
        )

    def _build_message(
        self,
        key: Tuple[int, int],
        corr: Corridor,
        cycle: int,
        sender: int,
        capability: CapabilityResult,
        data: AreaCycleData,
    ) -> PeerCairMessage:
        contract = self.contracts[key]
        own_end = "a" if sender == corr.area_a else "b"
        receiver = corr.area_b if sender == corr.area_a else corr.area_a
        need = self._need_last[sender]

        request: Optional[float] = None
        if need is not None and need.flag:
            sign = need.request_sign(own_end)
            # §2.3 sanity assert (not a request condition): the local
            # Jacobian column must have the relieving sign.
            if key not in data.dv_dq_import_by_corridor:
                rep1("dv_dq_import_by_corridor lacks this corridor",
                     area=sender, corridor=key)
            g = float(data.dv_dq_import_by_corridor[key])
            assert_relieving_sign(need, need.direction * g)
            request = sign * contract.dq_quant_mvar

        return PeerCairMessage(
            version=SBX_MESSAGE_VERSION,
            sender_area=sender,
            receiver_area=receiver,
            corridor=key,
            cycle=cycle,
            offer_range_mvar=capability.offers_mvar[key],
            request_mvar=request,
            p_sched_mw=(self._state[key].p_sched_mw
                        if own_end == "a" else None),
        )

    def _apply_schedule_delta(
        self, st: _CorridorState, dq: float, paid: bool
    ) -> None:
        """Apply one schedule update; reset accumulators at zero-crossings."""
        s_old = st.surplus_mvar
        s_new = s_old + dq
        if s_old != 0.0 and (s_new == 0.0 or (s_new > 0.0) != (s_old > 0.0)):
            # Surplus zero-crossing (or exact landing on zero): the
            # paid/unpaid history of the OLD direction is settled.
            st.surplus_paid_mvar = 0.0
            st.surplus_unpaid_mvar = 0.0
            st.areas_of_record.clear()
            st.release_counter = 0
            residual = s_new
            if residual != 0.0:
                if paid:
                    st.surplus_paid_mvar = residual
                else:
                    st.surplus_unpaid_mvar = residual
        else:
            if paid:
                st.surplus_paid_mvar += dq
            else:
                st.surplus_unpaid_mvar += dq
        st.surplus_mvar = s_new

    def _unwind_step(self, st: _CorridorState, quantum: float) -> float:
        """Step 5: move one quantum towards q_std, paid surplus first.

        Returns the SIGNED schedule change applied (0 if nothing moved).
        Clamped at zero — the surplus never overshoots through unwind,
        so acting-role flips only happen via subsequent deals from zero.
        """
        s = st.surplus_mvar
        if s == 0.0:
            return 0.0
        step = min(quantum, abs(s))
        direction = -math.copysign(1.0, s)
        remaining = step
        # Paid surplus is reduced first (§2.2 Step 5: the requester stops
        # paying as soon as the need is gone), then unpaid.
        for attr in ("surplus_paid_mvar", "surplus_unpaid_mvar"):
            comp = getattr(st, attr)
            if remaining <= 0.0 or comp == 0.0:
                continue
            take = min(abs(comp), remaining)
            setattr(st, attr, comp + direction * take)
            remaining -= take
        st.surplus_mvar = st.surplus_paid_mvar + st.surplus_unpaid_mvar
        if abs(st.surplus_mvar) < 1e-12:
            st.surplus_mvar = 0.0
            st.surplus_paid_mvar = 0.0
            st.surplus_unpaid_mvar = 0.0
            st.areas_of_record.clear()
            st.release_counter = 0
        return direction * step

    def _run_corridor_cycle(
        self,
        key: Tuple[int, int],
        corr: Corridor,
        cycle: int,
        iteration: int,
        capability: Mapping[int, CapabilityResult],
        cycle_data: Mapping[int, AreaCycleData],
        refs: Dict[int, Dict[int, float]],
    ) -> None:
        contract = self.contracts[key]
        st = self._state[key]
        a, b = corr.area_a, corr.area_b

        # ── Step 2/3: messages, deterministic matching, checksums ───────
        msg_a = self._build_message(key, corr, cycle, a, capability[a],
                                    cycle_data[a])
        msg_b = self._build_message(key, corr, cycle, b, capability[b],
                                    cycle_data[b])
        q_sched_before = st.q_std_mvar + st.surplus_mvar
        deal = match(msg_a, msg_b, contract, q_sched_before, st.q_std_mvar)
        # Both sides evaluate independently; the in-process simulation
        # still exercises the protocol via double evaluation + checksum.
        deal_check = match(msg_a, msg_b, contract, q_sched_before,
                           st.q_std_mvar)
        assert_checksums_match(deal.checksum(), deal_check.checksum(),
                               corridor=key, cycle=cycle,
                               what="deal records")

        if deal.kind == KIND_SCARCITY:
            self.scarcity_events.append(deal)

        executed = deal.dq_deal_mvar != 0.0
        if executed:
            self._apply_schedule_delta(st, deal.dq_deal_mvar, deal.paid)
            if deal.requester is not None:
                st.areas_of_record.add(deal.requester)
            else:
                st.areas_of_record.update((a, b))
            st.release_counter = 0

        # ── Step 5: unwind (at most one schedule update per cycle) ──────
        unwound = 0.0
        if not executed and st.surplus_mvar != 0.0:
            flags_clear = all(
                not (self._need_last[z] is not None
                     and self._need_last[z].flag)
                for z in (st.areas_of_record or {a, b})
            )
            if flags_clear:
                st.release_counter += 1
            else:
                st.release_counter = 0
            if st.release_counter >= self.config.m_release:
                unwound = self._unwind_step(st, contract.dq_quant_mvar)

        # ── Step 4: setpoints (invariant: ≤ 1 deviating end) ────────────
        q_sched = st.q_std_mvar + st.surplus_mvar
        dv = 0.0
        acting_area: Optional[int] = None
        v_ref_a = {ln.bus_a: st.v_std_a_act[k]
                   for k, ln in enumerate(corr.lines)}
        v_ref_b = {ln.bus_b: st.v_std_b_act[k]
                   for k, ln in enumerate(corr.lines)}
        if st.surplus_mvar != 0.0:
            acting_end = "a" if st.surplus_mvar > 0.0 else "b"
            acting_area = a if acting_end == "a" else b
            dv = corridor_solve_dv(
                corr,
                list(st.v_std_a_act),
                list(st.v_std_b_act),
                list(st.p_sched_mw),
                q_sched,
                acting_end,
                dv_search_range_pu=self.config.dv_search_range_pu,
                delta_max_rad=self.config.delta_max_rad,
            )
            if acting_end == "a":
                v_ref_a = {bus: v + dv for bus, v in v_ref_a.items()}
            else:
                v_ref_b = {bus: v + dv for bus, v in v_ref_b.items()}

        # Invariant assert (plan Step 4 / Phase 5): at most one end's
        # references deviate from the contract voltages.
        dev_a = any(
            abs(v_ref_a[ln.bus_a] - st.v_std_a_act[k]) > DV_ZERO_TOL_PU
            for k, ln in enumerate(corr.lines)
        )
        dev_b = any(
            abs(v_ref_b[ln.bus_b] - st.v_std_b_act[k]) > DV_ZERO_TOL_PU
            for k, ln in enumerate(corr.lines)
        )
        if dev_a and dev_b:
            rep1("Step-4 invariant violated: both corridor ends deviate "
                 "from the contract voltages",
                 corridor=key, cycle=cycle, dv=dv,
                 surplus_mvar=st.surplus_mvar)
        if st.surplus_mvar == 0.0 and (dev_a or dev_b):
            rep1("Step-4 invariant violated: zero surplus but a deviating "
                 "end", corridor=key, cycle=cycle)

        st.refs_a, st.refs_b = v_ref_a, v_ref_b
        refs[a].update(v_ref_a)
        refs[b].update(v_ref_b)

        need_a = self._need_last[a]
        need_b = self._need_last[b]
        self.records[key].append(CorridorCycleRecord(
            cycle=cycle,
            iteration=iteration,
            p_sched_mw=st.p_sched_mw,
            q_meas_mvar=st.q_meas_mvar,
            q_std_mvar=st.q_std_mvar,
            q_sched_mvar=q_sched,
            consistency=st.consistency_last,
            surplus_mvar=st.surplus_mvar,
            surplus_paid_mvar=st.surplus_paid_mvar,
            surplus_unpaid_mvar=st.surplus_unpaid_mvar,
            deal=deal,
            unwound_mvar=unwound,
            acting_area=acting_area,
            dv_pu=dv,
            need_a=bool(need_a is not None and need_a.flag),
            need_b=bool(need_b is not None and need_b.flag),
            offer_a=capability[a].offers_mvar[key],
            offer_b=capability[b].offers_mvar[key],
            t_a=capability[a].t,
            t_b=capability[b].t,
        ))

    # ------------------------------------------------------------------
    #  Views
    # ------------------------------------------------------------------

    def corridor_state(self, key: Tuple[int, int]) -> _CorridorState:
        if key not in self._state:
            rep1("unknown corridor key", key=key,
                 known=sorted(self._state.keys()))
        return self._state[key]

    def last_need(self, area: int) -> Optional[NeedDecision]:
        """Most recent need decision of one area (None before any step).

        Exposed for the runner adapter, which must compute the
        relieving-sign scalar (``AreaCycleData.dv_dq_import_by_corridor``)
        for the SAME worst-violated bus this scheduler will assert
        against in the coming ``run_cycle``.
        """
        if area not in self._need_last:
            rep1("unknown SBX area", area=area, known=self.area_ids)
        return self._need_last[area]
