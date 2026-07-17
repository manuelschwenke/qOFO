"""
sbx_h/adapter.py
================
Runner-side adapter for SBX-H v6.

Maps the runner's live objects — per-zone :class:`~core.measurement.
Measurement` and :class:`~controller.tso_controller.TSOController` —
onto the plant-agnostic :class:`~sbx_h.scheduler.SBXScheduler`, and
writes the schedule's corridor-terminal references into each zone's
EXISTING voltage-tracking mechanism (``TSOControllerConfig.
v_setpoints_pu`` via ``update_voltage_setpoints``) with
weight ``w_track_factor × g_v`` (ordinary voltage weight by default).

Data-flow per TSO tick (consecutive ``iteration`` = 0, 1, 2, …):

1. On a cycle boundary: settle the elapsed cycle and apply the ACTIVE
   contract schedule (controller intent / hourly planning /
   planned-support
   interval) — :meth:`SBXScheduler.run_cycle`; write the returned
   references.  Iteration-0 references are written at construction.
2. Every tick: feed :meth:`SBXScheduler.record_step` with the areas'
   monitored voltages/bounds (violation indicator) and the per-corridor
   tie P/Q/terminal-V measured at the reference end A (cycle
   averaging + scheduled-reference settlement inputs).

v6 (2026-07-12): the capability-LP composition, the cached-Jacobian
relieving-sign machinery and every other deal-layer input were removed
with the deal layer itself (archive ``_archive/sbx_h_v5/``) — the
adapter no longer reads ANY controller-internal model object; it
consumes measurements and controller CONFIG only.

Planned support: ``support_intervals`` maps a corridor key to
``(t_from_s, t_to_s, dv_a_pu, dv_b_pu)`` windows applied to the built
contracts via :func:`sbx_h.contract.with_planned_support` — "the
neighbour holds a raised boundary voltage from minute X to Y, agreed
in advance".

Author: Manuel Schwenke / Claude Code
Date: 2026-07-12 (SBX-H v6)
"""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandapower as pp
from numpy.typing import NDArray

from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract, with_planned_support
from sbx_h.corridor import Corridor, build_corridor_registry
from sbx_h.fail import rep1
from sbx_h.metrics import VoltageTrackingEquity, voltage_tracking_equity
from sbx_h.scheduler import AreaStepInput, SBXScheduler


def controller_intent_schedule(
    corridor: Corridor,
    tso_controllers: Mapping[int, object],
) -> Tuple[Tuple[float, Tuple[float, ...], Tuple[float, ...]], ...]:
    """Constant t=0 schedule from each area's intended bus setpoints.

    Measured plant voltages are deliberately not consulted: a realised
    operating point is evidence for feasibility, not a control promise.
    """

    def _intent(area: int, bus: int) -> float:
        if area not in tso_controllers:
            rep1("SBX area has no TSO controller", area=area,
                 controllers=sorted(tso_controllers))
        cfg = tso_controllers[area].config
        if cfg.v_setpoints_pu is None:
            rep1("SBX requires intended voltage setpoints",
                 area=area, bus=bus)
        buses = np.asarray(cfg.voltage_bus_indices, dtype=np.int64)
        refs = np.asarray(cfg.v_setpoints_pu, dtype=np.float64)
        if refs.ndim != 1 or refs.size != buses.size:
            rep1("TSO voltage setpoints do not align with monitored buses",
                 area=area, n_buses=int(buses.size),
                 setpoint_shape=tuple(refs.shape))
        hits = np.flatnonzero(buses == int(bus))
        if hits.size != 1:
            rep1("corridor terminal is not uniquely monitored by its TSO",
                 area=area, bus=bus, hits=hits.tolist())
        value = float(refs[int(hits[0])])
        if not (math.isfinite(value) and value > 0.0):
            rep1("intended terminal voltage must be finite and positive",
                 area=area, bus=bus, v_ref_pu=value)
        return value

    v_a = tuple(
        _intent(corridor.area_a, line.bus_a) for line in corridor.lines
    )
    v_b = tuple(
        _intent(corridor.area_b, line.bus_b) for line in corridor.lines
    )
    return ((0.0, v_a, v_b),)


class SBXRunnerAdapter:
    """Wires the v6 scheduler into the multi-TSO runner loop.

    By default the contract schedule is each TSO controller's intended
    terminal voltage. An explicit planning schedule overrides that
    intent. The converged plant state is used only for the initial
    feasibility diagnostic; the iteration-0 references are written
    immediately.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        area_map: Mapping[int, List[int]],
        tso_controllers: Mapping[int, object],
        config: SBXConfig,
        *,
        v_std_schedules: Optional[Mapping[Tuple[int, int], object]] = None,
        q_band_schedules: Optional[Mapping[Tuple[int, int], object]] = None,
        support_intervals: Optional[Mapping[
            Tuple[int, int], Sequence[Tuple[float, float, float, float]]
        ]] = None,
        freeze_time_s: float = 0.0,
    ) -> None:
        """``support_intervals``: per corridor key, planned-support
        windows ``(t_from_s, t_to_s, dv_a_pu, dv_b_pu)`` composed onto
        the built contract's voltage schedule.  ``freeze_time_s`` is
        the scenario time of this construction tick — the lookup time
        of the initial references."""
        self.config = config
        self.freeze_time_s = float(freeze_time_s)
        if not (math.isfinite(self.freeze_time_s)
                and self.freeze_time_s >= 0.0):
            rep1("freeze_time_s must be finite and non-negative",
                 freeze_time_s=freeze_time_s)
        self.registry = build_corridor_registry(net, area_map)
        self.schedule_source = (
            "planning" if v_std_schedules is not None else "controller_intent"
        )
        if v_std_schedules is not None:
            missing = set(self.registry) - set(v_std_schedules)
            extra = set(v_std_schedules) - set(self.registry)
            if missing or extra:
                rep1("v_std_schedules must cover exactly the corridor "
                     "registry", missing=sorted(missing),
                     extra=sorted(extra))
        if q_band_schedules is not None:
            if v_std_schedules is None:
                rep1("q_band_schedules require v_std_schedules (both "
                     "come from the same planning pre-pass)")
            missing = set(self.registry) - set(q_band_schedules)
            extra = set(q_band_schedules) - set(self.registry)
            if missing or extra:
                rep1("q_band_schedules must cover exactly the corridor "
                     "registry", missing=sorted(missing),
                     extra=sorted(extra))
        if support_intervals is not None:
            unknown = set(support_intervals) - set(self.registry)
            if unknown:
                rep1("support_intervals name unknown corridors",
                     unknown=sorted(unknown),
                     known=sorted(self.registry))
        self.contracts = {}
        for key, corr in self.registry.items():
            schedule = (
                v_std_schedules[key] if v_std_schedules is not None
                else controller_intent_schedule(corr, tso_controllers)
            )
            contract = build_default_contract(
                corr, net, config,
                v_std_schedule=schedule,
                q_band_schedule=(q_band_schedules[key]
                                 if q_band_schedules is not None
                                 else None),
            )
            if support_intervals is not None and key in support_intervals:
                for t_from, t_to, dv_a, dv_b in support_intervals[key]:
                    contract = with_planned_support(
                        contract, float(t_from), float(t_to),
                        dv_a_pu=float(dv_a), dv_b_pu=float(dv_b),
                    )
            self.contracts[key] = contract
        self.scheduler = SBXScheduler(config, self.registry,
                                      self.contracts)
        self.initial_schedule_diagnostics = (
            self._initial_schedule_diagnostics(net)
        )

        # ── Per-area wiring checks + terminal-bus positions ────────────
        # Terminal buses must be monitored voltage buses of their zone;
        # their references are tracked through the existing g_v term
        # with the configured SBX tracking weight.
        self._vpos: Dict[int, Dict[int, int]] = {}
        self._terminals: Dict[int, List[int]] = {}
        for z in self.scheduler.area_ids:
            if z not in tso_controllers:
                rep1("SBX area has no TSO controller", area=z,
                     controllers=sorted(tso_controllers.keys()))
            ctrl = tso_controllers[z]
            if ctrl.config.v_setpoints_pu is None:
                rep1("SBX requires active voltage tracking "
                     "(v_setpoints_pu is None)", area=z)
            if ctrl.config.g_v <= 0.0:
                rep1("SBX requires a positive voltage-tracking weight "
                     "g_v", area=z, g_v=ctrl.config.g_v)
            v_bus = list(ctrl.config.voltage_bus_indices)
            pos = {int(b): k for k, b in enumerate(v_bus)}
            terminals = sorted({
                (ln.bus_a if z == corr.area_a else ln.bus_b)
                for key in self.scheduler.corridors_of_area[z]
                for corr in (self.registry[key],)
                for ln in corr.lines
            })
            missing = [b for b in terminals if b not in pos]
            if missing:
                rep1("corridor terminal buses are not monitored voltage "
                     "buses of their zone", area=z, missing=missing,
                     monitored=v_bus)
            self._vpos[z] = pos
            self._terminals[z] = terminals
            w_term = (float(config.w_track) if config.w_track is not None
                      else config.w_track_factor * float(ctrl.config.g_v))
            ctrl.update_voltage_tracking_weights(
                np.asarray(terminals, dtype=np.int64), w_term,
            )

        # Per-corridor positions of the corridor's lines within area A's
        # tie measurement arrays (lazily validated on first measurement).
        self._tie_pos: Optional[Dict[Tuple[int, int], List[int]]] = None

        # Per-TSO-tick corridor-terminal snapshot
        # ``(iteration, {bus: v_meas_pu}, {bus: v_ref_pu})`` for plots
        # and evaluation.
        self.terminal_history: List[
            Tuple[int, Dict[int, float], Dict[int, float]]
        ] = []

        # Rolling ex-post voltage-tracking equity over one SBX cycle.
        # It uses all monitored TSO voltage outputs and never feeds the
        # controller or settlement decision.
        self.tracking_equity_history: List[
            Tuple[int, VoltageTrackingEquity]
        ] = []
        self._tracking_error_window: Dict[
            int, Deque[NDArray[np.float64]]
        ] = {
            area: deque(maxlen=self.config.k_sched)
            for area in self.scheduler.area_ids
        }

        # The adapter may be constructed mid-run: the first on_tso_step
        # call defines the internal iteration origin.
        self._it_offset: Optional[int] = None

        # Border-actuator diagnostic (controllable AVR generators / TSO
        # DERs directly at a corridor terminal bus, hop 0, or one
        # transformer winding away, hop 1) — diagnostic only.
        self.border_actuators = self._border_actuator_diagnostic(
            net, tso_controllers,
        )

        # Iteration-0: every corridor terminal at the schedule interval
        # active at the freeze tick.
        self._write_references(
            self.scheduler.initial_references(time_s=self.freeze_time_s),
            tso_controllers,
        )

    # ------------------------------------------------------------------
    #  Per-TSO-tick entry point
    # ------------------------------------------------------------------

    def on_tso_step(
        self,
        iteration: int,
        measurements: Mapping[int, object],
        tso_controllers: Mapping[int, object],
    ) -> None:
        """One TSO tick: (cycle boundary, then) record_step.

        Ordering matches the scheduler contract: the boundary consumes
        the ELAPSED cycle's samples before the tick's sample is
        recorded.  ``iteration`` is the runner's consecutive TSO tick
        counter; internally rebased to 0 at the first call."""
        if self._it_offset is None:
            self._it_offset = int(iteration)
        iteration = int(iteration) - self._it_offset
        for z in self.scheduler.area_ids:
            if z not in measurements:
                rep1("measurement missing for SBX area", area=z,
                     got=sorted(measurements.keys()))

        if self.scheduler.is_cycle_boundary(iteration):
            time_s = self.freeze_time_s \
                + iteration * self.config.tso_period_s
            refs = self.scheduler.run_cycle(iteration, time_s=time_s)
            self._write_references(refs, tso_controllers)

        area_inputs = {
            z: self._area_step_input(tso_controllers[z], measurements[z])
            for z in self.scheduler.area_ids
        }
        self._record_tracking_equity(
            iteration,
            area_inputs,
            tso_controllers,
        )
        tie_p, tie_q = self._corridor_tie_measurements(measurements)
        tie_va, tie_vb = self._corridor_terminal_voltages(measurements)
        self.scheduler.record_step(iteration, area_inputs, tie_p, tie_q,
                                   tie_va, tie_vb)

        term_v: Dict[int, float] = {}
        for z in self.scheduler.area_ids:
            v_bus, v = self._monitored_voltages(
                tso_controllers[z], measurements[z],
            )
            pos = self._vpos[z]
            for bus in self._terminals[z]:
                term_v[bus] = float(v[pos[bus]])
        term_ref: Dict[int, float] = {}
        for key in self.registry:
            st = self.scheduler.corridor_state(key)
            term_ref.update(st.refs_a)
            term_ref.update(st.refs_b)
        self.terminal_history.append((int(iteration), term_v, term_ref))

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _record_tracking_equity(
        self,
        iteration: int,
        area_inputs: Mapping[int, AreaStepInput],
        tso_controllers: Mapping[int, object],
    ) -> None:
        """Record rolling all-bus tracking burden and cross-area Gini.

        The rolling window contains the latest ``k_sched`` TSO samples,
        i.e. one contractual metering cycle.  Errors are stored after
        subtracting the active reference, so a schedule change is handled
        correctly within a window.
        """
        for area in self.scheduler.area_ids:
            measured = np.asarray(
                area_inputs[area].v_meas_pu,
                dtype=np.float64,
            )
            scheduled = np.asarray(
                tso_controllers[area].config.v_setpoints_pu,
                dtype=np.float64,
            )
            if measured.shape != scheduled.shape:
                rep1(
                    "voltage-tracking equity measurement/reference "
                    "shape mismatch",
                    area=area,
                    measured_shape=measured.shape,
                    scheduled_shape=scheduled.shape,
                )
            self._tracking_error_window[area].append(
                measured - scheduled
            )

        errors_by_area = {
            area: tuple(
                float(value)
                for sample in self._tracking_error_window[area]
                for value in sample
            )
            for area in self.scheduler.area_ids
        }
        metric = voltage_tracking_equity(errors_by_area)
        self.tracking_equity_history.append((int(iteration), metric))

    def _initial_schedule_diagnostics(
        self, net: pp.pandapowerNet,
    ) -> List[Dict[str, object]]:
        """Compare intended/planned promises with the converged snapshot.

        This is a pre-check, not a capability certificate: it reports
        whether each terminal is initially within the contractual hold
        tolerance without redefining the schedule from the measurement.
        """
        if not hasattr(net, "res_bus") or len(net.res_bus) != len(net.bus):
            rep1("SBX initial schedule diagnostic needs a converged plant",
                 n_res_bus=len(getattr(net, "res_bus", ())),
                 n_bus=len(net.bus))
        rows: List[Dict[str, object]] = []
        tolerance = float(self.config.v_hold_tolerance_pu)
        for key, corridor in self.registry.items():
            v_a, v_b = self.contracts[key].v_std_at(self.freeze_time_s)
            for line, scheduled_a, scheduled_b in zip(
                corridor.lines, v_a, v_b
            ):
                for side, area, bus, scheduled in (
                    ("a", corridor.area_a, line.bus_a, scheduled_a),
                    ("b", corridor.area_b, line.bus_b, scheduled_b),
                ):
                    measured = float(net.res_bus.at[bus, "vm_pu"])
                    error = measured - float(scheduled)
                    margin = tolerance - abs(error)
                    rows.append({
                        "corridor": key,
                        "line_idx": int(line.line_idx),
                        "side": side,
                        "area": int(area),
                        "bus": int(bus),
                        "v_meas_pu": measured,
                        "v_sched_pu": float(scheduled),
                        "v_error_pu": error,
                        "hold_margin_pu": margin,
                        "initially_holds": bool(margin >= 0.0),
                        "schedule_source": self.schedule_source,
                    })
        return rows

    def _border_actuator_diagnostic(
        self, net: pp.pandapowerNet, tso_controllers: Mapping[int, object],
    ) -> List[Dict[str, object]]:
        """Detect controllable actuators at/next to corridor terminals.

        One entry per (corridor, area, terminal, element) hit with
        ``element`` ∈ {"gen", "der"} and ``hop`` 0 (at the terminal) or
        1 (one transformer winding away).  Diagnostic only."""
        adjacent: Dict[int, set] = {}

        def _link(buses: List[int]) -> None:
            for b in buses:
                adjacent.setdefault(b, set()).update(
                    x for x in buses if x != b
                )

        if hasattr(net, "trafo") and len(net.trafo) > 0:
            for _, row in net.trafo.iterrows():
                _link([int(row["hv_bus"]), int(row["lv_bus"])])
        if hasattr(net, "trafo3w") and len(net.trafo3w) > 0:
            for _, row in net.trafo3w.iterrows():
                _link([int(row["hv_bus"]), int(row["mv_bus"]),
                       int(row["lv_bus"])])

        hits: List[Dict[str, object]] = []
        for z in self.scheduler.area_ids:
            cfg = tso_controllers[z].config
            controllable: List[Tuple[str, int, int]] = [
                ("gen", int(g), int(b))
                for g, b in zip(cfg.gen_indices, cfg.gen_bus_indices)
            ] + [
                ("der", int(s), int(net.sgen.at[s, "bus"]))
                for s in cfg.der_indices
            ]
            for key in self.scheduler.corridors_of_area[z]:
                corr = self.registry[key]
                own_terms = {
                    (ln.bus_a if z == corr.area_a else ln.bus_b)
                    for ln in corr.lines
                }
                for kind, idx, bus in controllable:
                    if bus in own_terms:
                        hop = 0
                    elif own_terms & adjacent.get(bus, set()):
                        hop = 1
                    else:
                        continue
                    term = (bus if hop == 0 else sorted(
                        own_terms & adjacent[bus])[0])
                    hits.append({
                        "corridor": key, "area": z,
                        "terminal_bus": int(term), "element": kind,
                        "index": idx, "bus": bus, "hop": hop,
                    })
        return hits

    @staticmethod
    def _monitored_voltages(
        ctrl, meas
    ) -> Tuple[List[int], NDArray[np.float64]]:
        """Measured voltages at the zone's monitored buses."""
        v_bus = [int(b) for b in ctrl.config.voltage_bus_indices]
        bus_arr = np.asarray(meas.bus_indices)
        idx = np.searchsorted(bus_arr, v_bus)
        if np.any(idx >= bus_arr.size) or \
                not np.array_equal(bus_arr[idx], np.asarray(v_bus)):
            rep1("monitored voltage buses missing from the measurement",
                 v_bus=v_bus)
        return v_bus, meas.voltage_magnitudes_pu[idx].astype(np.float64)

    def _area_step_input(self, ctrl, meas) -> AreaStepInput:
        v_bus, v = self._monitored_voltages(ctrl, meas)
        n = len(v_bus)
        return AreaStepInput(
            bus_indices=tuple(v_bus),
            v_meas_pu=tuple(float(x) for x in v),
            v_min_pu=(float(ctrl.config.v_min_pu),) * n,
            v_max_pu=(float(ctrl.config.v_max_pu),) * n,
        )

    def _corridor_tie_measurements(
        self, measurements: Mapping[int, object]
    ) -> Tuple[Dict[Tuple[int, int], List[float]],
               Dict[Tuple[int, int], List[float]]]:
        """Per-corridor tie P/Q at the reference end A, corridor line
        order (load convention at the in-zone endpoint = export-from-A
        positive; no sign flip)."""
        if self._tie_pos is None:
            tie_pos: Dict[Tuple[int, int], List[int]] = {}
            for key, corr in self.registry.items():
                m = measurements[corr.area_a]
                positions: List[int] = []
                for ln in corr.lines:
                    hits = np.where(
                        np.asarray(m.tie_line_indices) == ln.line_idx
                    )[0]
                    if hits.size != 1:
                        rep1("tie line not uniquely present in area A's "
                             "measurement", corridor=key,
                             line_idx=ln.line_idx, hits=hits.tolist())
                    p = int(hits[0])
                    endp = int(np.asarray(m.tie_line_endpoint_buses)[p])
                    if endp != ln.bus_a:
                        rep1("area A's tie measurement endpoint is not "
                             "the corridor's reference-end bus",
                             corridor=key, line_idx=ln.line_idx,
                             endpoint=endp, bus_a=ln.bus_a)
                    positions.append(p)
                tie_pos[key] = positions
            self._tie_pos = tie_pos

        tie_p: Dict[Tuple[int, int], List[float]] = {}
        tie_q: Dict[Tuple[int, int], List[float]] = {}
        for key, corr in self.registry.items():
            m = measurements[corr.area_a]
            if len(m.tie_line_p_mw) != len(m.tie_line_q_mvar):
                rep1("tie_line_p_mw not populated (measure_zone_tso "
                     "predates the G2 extension?)", corridor=key,
                     n_p=len(m.tie_line_p_mw), n_q=len(m.tie_line_q_mvar))
            pos = self._tie_pos[key]
            tie_p[key] = [float(m.tie_line_p_mw[p]) for p in pos]
            tie_q[key] = [float(m.tie_line_q_mvar[p]) for p in pos]
        return tie_p, tie_q

    def _corridor_terminal_voltages(
        self, measurements: Mapping[int, object]
    ) -> Tuple[Dict[Tuple[int, int], List[float]],
               Dict[Tuple[int, int], List[float]]]:
        """Per-corridor, per-line measured terminal voltages (A / B
        end), each side read from ITS OWN area's measurement (the
        scheduled-reference settlement inputs)."""

        def _v_at(meas, bus: int) -> float:
            bus_arr = np.asarray(meas.bus_indices)
            idx = int(np.searchsorted(bus_arr, bus))
            if idx >= bus_arr.size or int(bus_arr[idx]) != bus:
                rep1("terminal bus missing from the measurement", bus=bus)
            return float(meas.voltage_magnitudes_pu[idx])

        tie_va: Dict[Tuple[int, int], List[float]] = {}
        tie_vb: Dict[Tuple[int, int], List[float]] = {}
        for key, corr in self.registry.items():
            m_a = measurements[corr.area_a]
            m_b = measurements[corr.area_b]
            tie_va[key] = [_v_at(m_a, ln.bus_a) for ln in corr.lines]
            tie_vb[key] = [_v_at(m_b, ln.bus_b) for ln in corr.lines]
        return tie_va, tie_vb

    def _write_references(
        self,
        refs: Mapping[int, Dict[int, float]],
        tso_controllers: Mapping[int, object],
    ) -> None:
        """Write the frozen corridor-terminal references into each
        zone's tracked-output mechanism; non-terminal setpoints are
        left untouched."""
        for z, bus_refs in refs.items():
            ctrl = tso_controllers[z]
            vec = np.array(
                ctrl.config.v_setpoints_pu, dtype=np.float64, copy=True,
            )
            for bus, v_ref in bus_refs.items():
                vec[self._vpos[z][bus]] = float(v_ref)
            ctrl.update_voltage_setpoints(vec)
