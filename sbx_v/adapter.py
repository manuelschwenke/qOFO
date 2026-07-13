"""
sbx_v/adapter.py
===============
Runner-side adapter for SBX-V (build plan §9 Phase 5 wiring; mirrors the
``sbx_h.adapter.SBXRunnerAdapter`` pattern).

Responsibilities (all OUTSIDE the protected modules, hard rule 5):

* install a :class:`sbx_v.miqp_cost.PricingSolver` proxy on each zone TSO
  controller's ``solver`` attribute (the Phase-1 seam) — the proxy's
  spec provider reads this adapter's per-zone :class:`CommitScheduler`
  at the CURRENT iteration; in a neutral scenario every solve bypasses
  untouched (closed-loop R1);
* per TSO tick, BEFORE the zones solve: feed the need trackers from the
  zone measurements (netted PCC flow per AggregationArea, per-direction
  transmission-voltage deviations) and let the commit scheduler run the
  request pipeline for the next window;
* per TSO tick, AFTER the zones solve: capture the dispatched netted
  PCC-Q reference per area (the Abruf is the logged dispatch, plan §0)
  into per-window accumulators for the settlement plane;
* per plant step: four-quadrant metering of the per-NVP boundary Q
  (:class:`sbx_v.metering.AreaMeter`, interval ``[t−dt, t)`` with the
  post-power-flow state at ``t``);
* at the end of the run (:meth:`finalise`): assemble the settlement
  inputs and run the Phase-2 :class:`sbx_v.settlement.SettlementEngine`.

AggregationAreas are the DS areas (DP5): one per DSO id, netting that
DSO's interface transformers.  Areas, PCC output rows and bands are
derived from the zone controllers' configurations — the controllers
themselves are never modified.

Feedforward note (plan §4, STATUS §4.2): the scheduled-envelope lead is
available from the commit scheduler, but the v1 wiring applies NO
setpoint offset — unlike an MSR/MSC switch there is no plant-side jump
the DSO could counteract; the reference itself moves through the priced
MIQP in quantum-rate micro-steps.  E1 evaluates whether commit-instant
transients warrant the lead.

Author: Manuel Schwenke / Claude Code
Date: 2026-07-09 (SBX-V Phase 5)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from core.message import CapabilityMessage
from sbx_h.fail import rep1
from sbx_v.band import band_from_config
from sbx_v.commit import AreaIterationInput, CommitScheduler
from sbx_v.config import SBXVConfig
from sbx_v.directions import Direction
from sbx_v.messages import PotentialMessage, Window
from sbx_v.metering import AreaMeter
from sbx_v.miqp_cost import PricingSolver
from sbx_v.potentials import build_potential_message
from sbx_v.settlement import (DSO_DELIVERS, SettlementEngine,
                             SettlementResult, WindowObservation)

logger = logging.getLogger(__name__)


class _Area:
    """One AggregationArea: a DSO's netted PCC interfaces (DP5)."""

    def __init__(self, dso_id: str, zone: int,
                 trafo_indices: Tuple[int, ...],
                 pcc_positions: Tuple[int, ...],
                 pcc_output_rows: Tuple[int, ...]) -> None:
        self.dso_id = dso_id
        self.zone = zone
        self.trafo_indices = trafo_indices    # pandapower trafo3w indices
        self.pcc_positions = pcc_positions    # positions in the zone's
        #                                       pcc_trafo_indices list
        self.pcc_output_rows = pcc_output_rows


class SBXVRunnerAdapter:
    """Wires the SBX-V planes into the multi-TSO/DSO runner."""

    def __init__(
        self,
        config: SBXVConfig,
        tso_controllers: Mapping[int, object],
        *,
        tso_period_s: float,
        net=None,
    ) -> None:
        if abs(config.tso_period_s - float(tso_period_s)) > 1e-9:
            rep1("SBXVConfig.tso_period_s must match the runner's "
                 "tso_period_s (window counter in TSO iterations)",
                 sbxv=config.tso_period_s, runner=tso_period_s)
        self.config = config
        self._net_for_preset = net
        self._k_now = -1        # no priced solve before the first tick

        # ── Derive AggregationAreas from the zone controller configs ──
        self.areas: Dict[str, _Area] = {}
        areas_by_zone: Dict[int, List[str]] = {}
        for z, ctrl in sorted(tso_controllers.items()):
            trafos = list(ctrl.config.pcc_trafo_indices)
            dso_ids = list(ctrl.config.pcc_dso_controller_ids)
            n_v = len(ctrl.config.voltage_bus_indices)
            by_dso: Dict[str, List[int]] = {}
            for pos, dso_id in enumerate(dso_ids):
                by_dso.setdefault(dso_id, []).append(pos)
            for dso_id, positions in sorted(by_dso.items()):
                if dso_id in self.areas:
                    rep1("a DSO area appears in two zones — the "
                         "AggregationArea derivation is inconsistent",
                         dso_id=dso_id)
                self.areas[dso_id] = _Area(
                    dso_id=dso_id, zone=z,
                    trafo_indices=tuple(int(trafos[p])
                                        for p in positions),
                    pcc_positions=tuple(positions),
                    pcc_output_rows=tuple(n_v + p for p in positions),
                )
                areas_by_zone.setdefault(z, []).append(dso_id)

        # ── Bands, per-zone commit schedulers, solver proxies ──
        # Preset 'ar41414_default' derives the contracted P of an area
        # from the rated interface capacity Σ sn_hv_mva of its coupling
        # transformers [AR §5.2.1 proxy]; requires the plant net at
        # construction.  The AR Anhang C spread assertion (≥ 70 Mvar)
        # applies inside band_from_config and fails fast per area.
        self.bands = {}
        for a, area in self.areas.items():
            contracted = None
            if config.band_preset == "ar41414_default":
                if net is None:
                    rep1("preset 'ar41414_default' needs the plant net "
                         "at adapter construction (contracted P from "
                         "the rated interface capacity)", area_id=a)
                contracted = float(sum(
                    net.trafo3w.at[t, "sn_hv_mva"]
                    for t in area.trafo_indices))
            self.bands[a] = band_from_config(config, a,
                                             contracted_p_mw=contracted)
        self.schedulers: Dict[int, CommitScheduler] = {}
        for z, dso_ids in sorted(areas_by_zone.items()):
            self.schedulers[z] = CommitScheduler(
                config,
                {a: self.bands[a] for a in dso_ids},
                {a: self.areas[a].pcc_output_rows for a in dso_ids},
            )
        self._proxies: Dict[int, PricingSolver] = {}
        for z, ctrl in sorted(tso_controllers.items()):
            if z not in self.schedulers:
                continue
            sched = self.schedulers[z]
            proxy = PricingSolver(
                ctrl.solver,
                (lambda s=sched: s.specs_for(self._k_now)
                 if self._k_now >= 0 else None),
                g_z_tier=config.g_z_tier,
            )
            ctrl.solver = proxy
            self._proxies[z] = proxy

        # ── Metering and reference accumulators ──
        self.meters: Dict[str, AreaMeter] = {
            a: AreaMeter(a, n_nvp=len(area.trafo_indices),
                         window_s=config.window_s)
            for a, area in self.areas.items()
        }
        #: (area, window) → list of netted dispatched references [Mvar].
        self._q_set_acc: Dict[Tuple[str, int], List[float]] = {}
        if not self.areas:
            rep1("SBX-V adapter found no AggregationAreas — the zone "
                 "controllers report no PCC interfaces")
        logger.info(
            "SBX-V adapter: %d AggregationArea(s) across %d zone(s); "
            "band preset '%s'; k_window=%d.",
            len(self.areas), len(self.schedulers), config.band_preset,
            config.k_window,
        )

    # ------------------------------------------------------------------
    #  TSO-tick hooks
    # ------------------------------------------------------------------

    def before_solve(
        self,
        k: int,
        measurements: Mapping[int, object],
        tso_controllers: Mapping[int, object],
    ) -> None:
        """Need trackers + request pipeline, then arm the spec provider
        for iteration ``k`` (the zones solve right after this)."""
        self._k_now = int(k)
        for z, sched in self.schedulers.items():
            ctrl = tso_controllers[z]
            meas = measurements[z]
            q_per_pcc = np.asarray(
                ctrl._extract_trafo_reactive_power(meas),
                dtype=np.float64)
            v_dev_r, v_dev_l = self._zone_voltage_deviations(ctrl, meas)
            inputs = {}
            for a in sched.bands:
                area = self.areas[a]
                q_net = float(np.sum(q_per_pcc[list(area.pcc_positions)]))
                inputs[a] = AreaIterationInput(
                    q_pcc_netted_mvar=q_net,
                    v_dev_raising_pu=v_dev_r,
                    v_dev_lowering_pu=v_dev_l,
                )
            sched.step(
                k, inputs,
                lambda a, d, w, _z=z, _c=ctrl, _m=meas:
                    self._forecast(a, d, w, _c, _m),
            )

    def after_solve(
        self,
        k: int,
        tso_controllers: Mapping[int, object],
    ) -> None:
        """Capture the dispatched netted PCC-Q reference per area into
        the window accumulator (the logged Abruf, plan §0)."""
        w = k // self.config.k_window
        for z, sched in self.schedulers.items():
            ctrl = tso_controllers[z]
            u = ctrl._u_current
            if u is None:
                rep1("zone controller has no control vector after the "
                     "solve", zone=z, iteration=k)
            mapping = ctrl.config.der_mapping
            n_der = (mapping.n_der if mapping is not None
                     else len(ctrl.config.der_indices))
            for a in sched.bands:
                area = self.areas[a]
                q_ref = float(sum(u[n_der + p]
                                  for p in area.pcc_positions))
                self._q_set_acc.setdefault((a, w), []).append(q_ref)

    # ------------------------------------------------------------------
    #  Plant-step hook (metering)
    # ------------------------------------------------------------------

    def on_plant_step(self, time_s: float, dt_s: float, net) -> None:
        """Record the interval ``[time_s − dt_s, time_s)`` with the
        post-power-flow boundary Q at ``time_s`` (right-continuous
        integration — documented in :mod:`sbx_v.metering`)."""
        for a, area in self.areas.items():
            q = [float(net.res_trafo3w.at[t, "q_hv_mvar"])
                 for t in area.trafo_indices]
            self.meters[a].record_step(time_s - dt_s, dt_s, q)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _zone_voltage_deviations(ctrl, meas) -> Tuple[float, float]:
        """Worst zone-bus deviation beyond the Sollspannungs band per
        direction: (raising = undervoltage depth, lowering =
        overvoltage depth), both ≥ 0 [pu]."""
        v_dev_r = 0.0
        v_dev_l = 0.0
        v_min = float(ctrl.config.v_min_pu)
        v_max = float(ctrl.config.v_max_pu)
        for bus in ctrl.config.voltage_bus_indices:
            idx = np.where(meas.bus_indices == bus)[0]
            if len(idx) == 0:
                rep1("monitored bus missing from the zone measurement",
                     bus=int(bus))
            v = float(meas.voltage_magnitudes_pu[idx[0]])
            v_dev_r = max(v_dev_r, v_min - v)
            v_dev_l = max(v_dev_l, v - v_max)
        return v_dev_r, v_dev_l

    def _forecast(
        self,
        area_id: str,
        direction: Direction,
        window: Window,
        ctrl,
        meas,
    ) -> PotentialMessage:
        """Day-ahead plane, v1 = persistence: the CURRENT posted CAIR
        capability of the area, wrapped per DP1 with
        ``is_forecast=True`` [AR §6.3]."""
        area = self.areas[area_id]
        pos = list(area.pcc_positions)
        q_per_pcc = np.asarray(
            ctrl._extract_trafo_reactive_power(meas), dtype=np.float64)
        capability = CapabilityMessage(
            source_controller_id=area_id,
            target_controller_id=f"tso_zone_{area.zone}",
            iteration=self._k_now,
            interface_transformer_indices=np.asarray(
                area.trafo_indices, dtype=np.int64),
            q_min_mvar=np.asarray(
                ctrl.pcc_capability_min_mvar[pos], dtype=np.float64),
            q_max_mvar=np.asarray(
                ctrl.pcc_capability_max_mvar[pos], dtype=np.float64),
        )
        sched = self.schedulers[area.zone]
        return build_potential_message(
            area_id, direction, capability,
            float(np.sum(q_per_pcc[pos])), window,
            self.bands[area_id], sched.ledger,
            is_forecast=True,
        )

    # ------------------------------------------------------------------
    #  Finalisation (settlement plane, offline)
    # ------------------------------------------------------------------

    def finalise(self) -> Dict[str, object]:
        """Assemble the settlement inputs and settle the scenario (one
        Verrechnungsperiode, plan §2).

        Returns a dict with the ``SettlementResult``, the window
        observations, the grant records, per-zone pipeline logs, and any
        grants dropped because their window lies beyond the metered
        horizon (loud, never silent)."""
        observations: List[WindowObservation] = []
        metered_windows: Dict[str, set] = {}
        for a, meter in self.meters.items():
            regs = meter.finalise()
            metered_windows[a] = {r.window_index for r in regs}
            if meter.incomplete_tail_s > 0.0:
                logger.warning(
                    "SBX-V: area %s has %.0f s of unmetered tail — the "
                    "partial window is not settled.", a,
                    meter.incomplete_tail_s)
            for r in regs:
                acc = self._q_set_acc.get((a, r.window_index))
                q_set = (float(np.mean(acc)) if acc else None)
                observations.append(WindowObservation(
                    area_id=a, window_index=r.window_index,
                    t_start_s=r.t_start_s, q_meas_mvar=r.q_mean_mvar,
                    q_set_mvar=q_set,
                ))
        grant_records = []
        dropped = []
        for z, sched in self.schedulers.items():
            for g in sched.ledger.to_grant_records(
                    delivering_party=DSO_DELIVERS):
                if all(w in metered_windows.get(g.area_id, set())
                       for w in range(g.window_first, g.window_end)):
                    grant_records.append(g)
                else:
                    dropped.append(g)
                    logger.warning(
                        "SBX-V: grant for area %s window [%d, %d) lies "
                        "beyond the metered horizon — excluded from "
                        "settlement (extend n_total_s to settle it).",
                        g.area_id, g.window_first, g.window_end)
        engine = SettlementEngine(self.config, self.bands)
        result: Optional[SettlementResult] = None
        if observations:
            result = engine.settle(observations, grant_records)
        return {
            "settlement": result,
            "bands": {a: (b.q_raise_mvar, b.q_lower_mvar)
                      for a, b in self.bands.items()},
            "observations": observations,
            "grant_records": grant_records,
            "dropped_grants": dropped,
            "pipeline_logs": {z: list(s.pipeline.log)
                              for z, s in self.schedulers.items()},
            "scheduler_logs": {z: list(s.log)
                               for z, s in self.schedulers.items()},
            "decompositions": {z: p.last_decompositions
                               for z, p in self._proxies.items()},
        }
