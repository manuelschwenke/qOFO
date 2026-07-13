"""
sbx_h/adapter.py
==============
Runner-side adapter for SBX (Phase 5 control integration).

Maps the existing runner objects — per-zone :class:`~core.measurement.
Measurement` and :class:`~controller.tso_controller.TSOController` — onto
the plant-agnostic :class:`~sbx_h.scheduler.SBXScheduler` interface, and
writes the frozen corridor-terminal references back into each zone's
EXISTING voltage-tracking mechanism (``TSOControllerConfig.
v_setpoints_pu`` via ``update_voltage_setpoints``, weight ``g_v`` —
STATUS_SBX.md A4/G5).  No BME module, no vertical CAIR path, no MIQP
assembly and no solver wrapper is touched (plan hard rule 5).

Data-flow per TSO tick (consecutive ``iteration`` = 0, 1, 2, …):

1. On a cycle boundary (``iteration % k_sched == 0``, iteration > 0):
   build :class:`~sbx_h.scheduler.AreaCycleData` per area from the
   controller's CACHED local model (never the plant), run the six-step
   cycle, write the returned references.  Iteration 0 references
   (contract voltages) are written at construction.
2. Every tick: feed :meth:`SBXScheduler.record_step` with the areas'
   monitored voltages/bounds (need flag) and the per-corridor tie P/Q
   measured at the reference end A (cycle averaging).

Local-model composition of ``AreaCycleData`` (all cached, plan hard
rule "controllers never see the plant"):

* ``h_loc``            — voltage rows of the zone's expanded sensitivity
                         matrix (output layout ``[V_bus | …]``).
* ``terminal_h_rows``  — the same rows at the corridor terminal buses.
* actuator box         — the controller's own operating-point bounds
                         (``_compute_input_bounds``) with two capability-
                         specific adjustments documented below.
* ``dv_dq_import_by_corridor`` — relieving-sign scalar (plan §2.3):
                         mean ∂V(worst bus)/∂Q injection over the own-end
                         terminal buses from the zone's cached Jacobian
                         sensitivities (import ≙ +Q injection at the own
                         terminals); computed only while the need flag is
                         set.

Capability-box adjustments (documented deviations, conservative):

* Integer actuators (OLTC taps, MSC/MSR states) are FROZEN in the
  capability LP (``Δu = 0``): a discrete move is not guaranteed within a
  cycle (cooldowns, MIQP gating), so offers are backed by continuous
  actuators only.
* Under ``pcc_capability_on_output=True`` the controller's input bounds
  on Q_PCC,set are a wide engineering band (the DSO capability is a soft
  OUTPUT constraint there).  The capability LP has no such output row, so
  the adapter anchors the Q_PCC,set box at the measured interface Q plus
  the vertically reported DSO capability interval instead — the same
  quantity the legacy hard-bound mode uses.
* The box is widened to contain the current ``u`` (operating-point drift
  between the bound evaluation and the measurement must not make
  ``Δu = 0`` infeasible).

Author: Manuel Schwenke / Claude Code
Date: 2026-07-07 (SBX Phase 5)
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandapower as pp
from numpy.typing import NDArray

from optimisation.miqp_solver import MIQPSolver
from sbx_h.config import SBXConfig
from sbx_h.contract import build_default_contract
from sbx_h.corridor import build_corridor_registry
from sbx_h.fail import rep1
from sbx_h.scheduler import AreaCycleData, AreaStepInput, SBXScheduler


class SBXRunnerAdapter:
    """Wires the SBX scheduler into the multi-TSO runner loop.

    Construction requires the CONVERGED experiment base case (contract
    defaults, STATUS_SBX.md A7) and the live zone controllers; the
    iteration-0 references (contract voltages) are written immediately.
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
        freeze_time_s: float = 0.0,
    ) -> None:
        """v3 additions: ``v_std_schedules`` maps corridor keys to the
        planning pre-pass intervals ``(t_from_s, v_std_a, v_std_b)``
        (planning replaces the snapshot); ``freeze_time_s`` is the
        scenario time of this construction tick — the origin of the
        adapter's cycle clock and the lookup time of the initial
        references."""
        self.config = config
        self.freeze_time_s = float(freeze_time_s)
        if not (math.isfinite(self.freeze_time_s)
                and self.freeze_time_s >= 0.0):
            rep1("freeze_time_s must be finite and non-negative",
                 freeze_time_s=freeze_time_s)
        self.registry = build_corridor_registry(net, area_map)
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
        self.contracts = {
            key: build_default_contract(
                corr, net, config,
                v_std_schedule=(v_std_schedules[key]
                                if v_std_schedules is not None else None),
                q_band_schedule=(q_band_schedules[key]
                                 if q_band_schedules is not None
                                 else None),
            )
            for key, corr in self.registry.items()
        }
        # Joint-box LP backend: the wrapper's continuous default (OSQP,
        # first-order) stalls at 'user_limit' on the LP's mixed column
        # scales (Δu in Mvar for DER/PCC vs pu for V_gen); HiGHS (an LP
        # simplex/IPM specialist) solves it robustly.  Explicit backend
        # choice through the EXISTING wrapper — no solver modification.
        self.scheduler = SBXScheduler(
            config, self.registry, self.contracts,
            solver=MIQPSolver(solver="HIGHS"),
        )

        # ── Per-area wiring checks + terminal-bus positions ────────────
        # Terminal buses must be monitored voltage buses of their zone
        # (Phase 0: same precondition the vref path asserts) — their
        # references are tracked through the existing g_v term.
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
            # v4 'deliverable SBX' (2026-07-09): the corridor-terminal
            # contract references get PRIORITY over the ordinary
            # schedule tracking — with a uniform weight the MIQP
            # delivered only a fraction of the commanded boundary shift
            # (A4/G5 lifted; per-bus weights now exist in the
            # controller).  w_track (absolute) overrides the factor.
            w_term = (float(config.w_track) if config.w_track is not None
                      else config.w_track_factor * float(ctrl.config.g_v))
            ctrl.update_voltage_tracking_weights(
                np.asarray(terminals, dtype=np.int64), w_term,
            )

        # Per-corridor positions of the corridor's lines within area A's
        # tie measurement arrays (lazily validated on first measurement).
        self._tie_pos: Optional[Dict[Tuple[int, int], List[int]]] = None

        # Per-TSO-tick corridor-terminal snapshot
        # ``(iteration, {bus: v_meas_pu}, {bus: v_ref_pu})`` — the v2.2
        # item-3 margin check (worst within-cycle quantum-induced
        # terminal shift vs ``voltage_margin_pu``), the terminal
        # reference-tracking error report, and Phase 7 plots.
        self.terminal_history: List[
            Tuple[int, Dict[int, float], Dict[int, float]]
        ] = []

        # The adapter may be constructed mid-run (contracts frozen after
        # the warmup window): the first on_tso_step call defines the
        # internal iteration origin, so cycle boundaries count from the
        # contract-freeze tick regardless of the runner's tick numbering.
        self._it_offset: Optional[int] = None

        # Border-actuator diagnostic (Phase 5 amendment, 2026-07-07):
        # controllable AVR generators / TSO DERs sitting DIRECTLY at a
        # corridor terminal bus (hop 0) or one transformer away (hop 1).
        # Border-bus PV controllers can produce decentralised
        # coordination artefacts (MAVR thesis finding) — such corridors
        # are logged so acting-side behaviour is interpreted with that
        # in mind; the Step-4 invariant assert itself runs every cycle
        # regardless.
        self.border_actuators = self._border_actuator_diagnostic(
            net, tso_controllers,
        )

        # Iteration-0 Step 6: every corridor terminal at v_std (v3: the
        # schedule interval active at the freeze tick).
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
        """One TSO tick: (cycle protocol on boundaries, then) record_step.

        Ordering matches the scheduler contract: the boundary consumes
        the ELAPSED cycle's samples (iterations ``k − k_sched … k − 1``)
        before the tick ``k`` sample is recorded.  ``iteration`` is the
        runner's consecutive TSO tick counter; internally it is rebased
        to 0 at the first call (contract-freeze tick).
        """
        if self._it_offset is None:
            self._it_offset = int(iteration)
        iteration = int(iteration) - self._it_offset
        for z in self.scheduler.area_ids:
            if z not in measurements:
                rep1("measurement missing for SBX area", area=z,
                     got=sorted(measurements.keys()))

        if self.scheduler.is_cycle_boundary(iteration):
            cycle_data = {
                z: self._area_cycle_data(
                    z, tso_controllers[z], measurements[z],
                )
                for z in self.scheduler.area_ids
            }
            # v3: scenario time of this boundary (the adapter's cycle
            # clock starts at the contract-freeze tick).
            time_s = self.freeze_time_s \
                + iteration * self.config.tso_period_s
            refs = self.scheduler.run_cycle(iteration, cycle_data,
                                            time_s=time_s)
            self._write_references(refs, tso_controllers)

        area_inputs = {
            z: self._area_step_input(tso_controllers[z], measurements[z])
            for z in self.scheduler.area_ids
        }
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

    def _border_actuator_diagnostic(
        self, net: pp.pandapowerNet, tso_controllers: Mapping[int, object],
    ) -> List[Dict[str, object]]:
        """Detect controllable actuators at/next to corridor terminals.

        Returns one entry per (corridor, area, terminal, element) hit:
        ``{"corridor", "area", "terminal_bus", "element", "index",
        "bus", "hop"}`` with ``element`` ∈ {"gen", "der"} and ``hop`` 0
        (directly at the terminal) or 1 (one transformer winding away).
        Diagnostic only — nothing is disabled or reweighted.
        """
        # Buses one transformer away from any bus (2W and 3W tables).
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
        """Per-corridor tie P/Q at the reference end A, corridor line order.

        Area A's zone measurement reads the tie flow at the in-zone
        endpoint in LOAD convention (positive = into the line = leaving
        the zone), which IS export-from-A positive — no sign flip.
        """
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
        """Per-corridor, per-line measured terminal voltages (A / B end),
        each side read from ITS OWN area's measurement (§2.5 settlement
        attribution inputs)."""

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

    def _area_cycle_data(self, z: int, ctrl, meas) -> AreaCycleData:
        """Compose one area's joint-box LP inputs from its CACHED model."""
        if ctrl._u_current is None:
            rep1("controller has no current actuator vector yet "
                 "(cycle boundary before the first TSO solve?)", area=z)
        u_now = np.asarray(ctrl._u_current, dtype=np.float64).copy()

        v_bus, v_meas = self._monitored_voltages(ctrl, meas)
        n_v = len(v_bus)

        # Cached local sensitivity, voltage block (output layout
        # [V_bus | Q_PCC | I_line | Q_gen | Q_tie], DER-level columns).
        h_full = ctrl._expand_H_to_der_level(
            ctrl._build_sensitivity_matrix()
        )
        h_loc = np.ascontiguousarray(h_full[:n_v, :], dtype=np.float64)
        if h_loc.shape[1] != u_now.size:
            rep1("sensitivity column count does not match the actuator "
                 "vector", area=z, n_cols=h_loc.shape[1], n_u=u_now.size)

        u_lo, u_hi = ctrl._compute_input_bounds(
            np.asarray(meas.interface_q_hv_side_mvar, dtype=np.float64),
            np.asarray(meas.der_p_mw, dtype=np.float64),
        )
        u_lo = np.asarray(u_lo, dtype=np.float64).copy()
        u_hi = np.asarray(u_hi, dtype=np.float64).copy()

        # Q_PCC,set: anchor at measured interface Q + DSO capability
        # interval (see module docstring) when the controller's own input
        # bound is the wide engineering band.
        mapping = ctrl.config.der_mapping
        n_der = mapping.n_der if mapping is not None \
            else len(ctrl.config.der_indices)
        n_pcc = len(ctrl.config.pcc_trafo_indices)
        if n_pcc > 0 and ctrl.config.pcc_capability_on_output:
            q_iface = np.asarray(
                meas.interface_q_hv_side_mvar, dtype=np.float64,
            )
            u_lo[n_der:n_der + n_pcc] = q_iface + np.asarray(
                ctrl.pcc_capability_min_mvar, dtype=np.float64)
            u_hi[n_der:n_der + n_pcc] = q_iface + np.asarray(
                ctrl.pcc_capability_max_mvar, dtype=np.float64)

        # Freeze integer actuators (conservative capability).
        _, _, int_idx = ctrl._get_control_structure()
        if int_idx:
            u_lo[int_idx] = u_now[int_idx]
            u_hi[int_idx] = u_now[int_idx]

        # Δu = 0 must stay feasible under operating-point drift.
        u_lo = np.minimum(u_lo, u_now)
        u_hi = np.maximum(u_hi, u_now)

        terminal_h_rows = {
            bus: h_loc[self._vpos[z][bus], :]
            for bus in self._terminals[z]
        }

        # Relieving-sign scalar (§2.3): only while the need flag is set;
        # mean cached ∂V(worst)/∂Q injection over the own-end terminals.
        dv_dq: Dict[Tuple[int, int], float] = {}
        need = self.scheduler.last_need(z)
        if need is not None and need.flag:
            if need.worst_bus is None:
                rep1("need flag set without a worst bus", area=z,
                     iteration=need.iteration)
            for key in self.scheduler.corridors_of_area[z]:
                corr = self.registry[key]
                own_buses = [
                    ln.bus_a if z == corr.area_a else ln.bus_b
                    for ln in corr.lines
                ]
                try:
                    mat, obs_map, inj_map = \
                        ctrl.sensitivities.compute_dV_dQ_der(
                            own_buses, [int(need.worst_bus)],
                        )
                except ValueError as exc:
                    rep1("cached dV/dQ unavailable for the relieving-"
                         "sign assert", area=z, corridor=key,
                         worst_bus=need.worst_bus, error=str(exc))
                if not obs_map or not inj_map:
                    rep1("cached dV/dQ returned no usable rows/columns",
                         area=z, corridor=key, worst_bus=need.worst_bus,
                         obs=obs_map, inj=inj_map)
                dv_dq[key] = float(np.mean(mat[0, :]))

        return AreaCycleData(
            u_now=u_now,
            u_min=u_lo,
            u_max=u_hi,
            v_bus_indices=tuple(v_bus),
            v_meas_pu=v_meas,
            v_min_pu=np.full(n_v, float(ctrl.config.v_min_pu)),
            v_max_pu=np.full(n_v, float(ctrl.config.v_max_pu)),
            h_loc=h_loc,
            terminal_h_rows=terminal_h_rows,
            dv_dq_import_by_corridor=dv_dq,
        )

    def _write_references(
        self,
        refs: Mapping[int, Dict[int, float]],
        tso_controllers: Mapping[int, object],
    ) -> None:
        """Step 6: write the frozen corridor-terminal references into each
        zone's tracked-output mechanism; non-terminal setpoints are left
        untouched."""
        for z, bus_refs in refs.items():
            ctrl = tso_controllers[z]
            vec = np.array(
                ctrl.config.v_setpoints_pu, dtype=np.float64, copy=True,
            )
            for bus, v_ref in bus_refs.items():
                vec[self._vpos[z][bus]] = float(v_ref)
            ctrl.update_voltage_setpoints(vec)
