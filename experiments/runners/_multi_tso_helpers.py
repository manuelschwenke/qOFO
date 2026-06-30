# -*- coding: utf-8 -*-
"""
experiments/runners/_multi_tso_helpers.py
==========================================
Private helpers for :func:`experiments.runners.multi_tso_dso.run_multi_tso_dso`.

Extracted from ``experiments/000_M_TSO_M_DSO.py`` so the main runner module
stays focused on the simulation loop itself.  Kept as a private (leading
underscore) module because the helpers are an implementation detail of the
runner -- callers should use :func:`run_multi_tso_dso` directly.

Note for monkey-patching: ``run_multi_tso_dso`` resolves
``_run_delayed_stability_analysis`` via its own module globals (it imports
the symbol at module top), so :mod:`tuning.ceilings` can still rebind the
name on the loaded ``multi_tso_dso`` module to install a capturing wrapper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from numpy.typing import NDArray

from analysis.stability_analysis import (
    analyse_multi_zone_stability,
    MultiZoneStabilityResult,
)
from controller.dso_controller import DSOController
from controller.multi_tso_coordinator import MultiTSOCoordinator, ZoneDefinition
from core.reporting import (
    write_stability_analysis_markdown,
    write_tuned_params_json,
)
from network.ieee39 import HVNetworkInfo
from configs.multi_tso_config import MultiTSOConfig
from experiments.helpers import ContingencyEvent, MultiTSOIterationRecord


def _collect_contingency_watch_buses(
    net: pp.pandapowerNet,
    events: List["ContingencyEvent"],
    gen_trafo_map: Dict[int, int],
) -> List[int]:
    """Grid-bus + first-order line neighbours for every fired gen-trip event."""
    watch: set[int] = set()
    for ev in events:
        if ev.element_type == "gen" and ev.action == "trip" \
                and ev.element_index in gen_trafo_map:
            t_idx = gen_trafo_map[ev.element_index]
            if t_idx in net.trafo.index:
                grid_bus = int(net.trafo.at[t_idx, "hv_bus"])
                watch.add(grid_bus)
                line_mask = (
                    (net.line["from_bus"] == grid_bus)
                    | (net.line["to_bus"] == grid_bus)
                )
                for li in net.line.index[line_mask]:
                    watch.add(int(net.line.at[li, "from_bus"]))
                    watch.add(int(net.line.at[li, "to_bus"]))
    return sorted(watch)


def _controlled_oltc_keys(obj) -> List[Tuple[str, int]]:
    """Return ``[(table, trafo_index), ...]`` controlled by a
    ``DiscreteTapControl``.

    pandapower stores the controlled index as ``element_index`` (a scalar in
    single-index mode, or an ``ndarray`` in 'loc' mode), **not** ``tid``.
    Reading ``tid`` returned -1 and silently made the whole rate-limiter a
    no-op (machine 2W and coupler 3W OLTCs were never clamped).  Falls back to
    ``tid`` for any legacy pandapower that still exposes it."""
    table = str(getattr(obj, "element", "trafo"))
    ei = getattr(obj, "element_index", None)
    if ei is None:
        ei = getattr(obj, "tid", None)
    if ei is None:
        return []
    keys: List[Tuple[str, int]] = []
    for x in np.atleast_1d(ei):
        try:
            keys.append((table, int(x)))
        except (TypeError, ValueError):
            continue
    return keys


def _snapshot_oltc_taps(
    net: pp.pandapowerNet,
) -> Dict[Tuple[str, int], int]:
    """Snapshot ``tap_pos`` of every ``DiscreteTapControl`` in ``net.controller``.

    Keyed by ``(element_table, element_idx)`` with ``element_table`` in
    ``{'trafo', 'trafo3w'}``.  Returns an empty dict if no controllers
    are installed.  Used by the local-mode OLTC rate-limit clamp to
    bound per-step tap movement (see
    :attr:`configs.multi_tso_config.MultiTSOConfig.local_oltc_max_step_per_dt`).
    """
    from pandapower.control import DiscreteTapControl
    snap: Dict[Tuple[str, int], int] = {}
    if not hasattr(net, "controller") or len(net.controller) == 0:
        return snap
    for cid in net.controller.index:
        obj = net.controller.at[cid, "object"]
        if not isinstance(obj, DiscreteTapControl):
            continue
        for table, tid in _controlled_oltc_keys(obj):
            if table not in net or tid not in net[table].index:
                continue
            snap[(table, tid)] = int(net[table].at[tid, "tap_pos"])
    return snap


def _clamp_oltc_taps(
    net: pp.pandapowerNet,
    snapshot: Dict[Tuple[str, int], int],
    max_step: int,
) -> List[Tuple[str, int, int, int]]:
    """Clamp each ``DiscreteTapControl``'s tap_pos to ``±max_step`` of snapshot.

    Returns a list of ``(table, tid, prev_tap, clamped_tap)`` for every
    OLTC whose tap_pos was clamped.  An empty list means no clamping
    happened; the caller can skip the re-run PF.  Modifies
    ``net[table].tap_pos`` in place.
    """
    from pandapower.control import DiscreteTapControl
    moved: List[Tuple[str, int, int, int]] = []
    if not snapshot or not hasattr(net, "controller") or len(net.controller) == 0:
        return moved
    if max_step < 0:
        return moved
    for cid in net.controller.index:
        obj = net.controller.at[cid, "object"]
        if not isinstance(obj, DiscreteTapControl):
            continue
        for table, tid in _controlled_oltc_keys(obj):
            key = (table, tid)
            if key not in snapshot:
                continue
            prev = snapshot[key]
            curr = int(net[table].at[tid, "tap_pos"])
            delta = curr - prev
            if abs(delta) <= max_step:
                continue
            clamped = prev + (max_step if delta > 0 else -max_step)
            net[table].at[tid, "tap_pos"] = int(clamped)
            moved.append((table, tid, prev, int(clamped)))
    return moved


class _OLTCRateLimiter:
    """Per-step ``DiscreteTapControl`` rate limiter for local-mode OLTCs.

    Wraps :func:`_snapshot_oltc_taps` / :func:`_clamp_oltc_taps` with a
    persistent ``last_change_time_s`` map so each OLTC can be locked for
    ``cooldown_s`` simulation seconds after every actual tap movement,
    in addition to the existing per-step ``±max_step`` clamp.

    Usage pattern (once per outer simulation step, around every plant
    PF):

     limiter = _OLTCRateLimiter(max_step=1, cooldown_s=30.0)
     # at the top of every step:
     limiter.snapshot(net)
     # after every pp.runpp(run_control=True):
     moved = limiter.clamp(net, time_s)
     if moved:
    ...     pp.runpp(net, run_control=False, ...)
    """

    def __init__(self, max_step: int, cooldown_s: float,
                 cooldown_by_table: Optional[Dict[str, float]] = None) -> None:
        self.max_step: int = int(max_step)
        self.cooldown_s: float = float(cooldown_s)
        #: Per-element-table cooldown override (e.g. ``{"trafo": 180.0,
        #: "trafo3w": 60.0}`` to rate-limit machine 2W (MT) OLTCs and coupler
        #: 3W (NC) OLTCs at different wall-clock rates).  Tables absent from
        #: this map fall back to the scalar ``cooldown_s``.
        self._cooldown_by_table: Dict[str, float] = {
            str(k): float(v) for k, v in (cooldown_by_table or {}).items()
        }
        self._snapshot: Dict[Tuple[str, int], int] = {}
        self._last_change_time_s: Dict[Tuple[str, int], float] = {}

    def _cooldown_for(self, table: str) -> float:
        """Resolve the cooldown (s) for an OLTC on ``table`` (per-table
        override if present, else the scalar ``cooldown_s``)."""
        return self._cooldown_by_table.get(table, self.cooldown_s)

    @property
    def active(self) -> bool:
        """``True`` when the limiter will do anything (either rate-cap
        or cooldown).  ``max_step < 0`` AND every cooldown ``<= 0`` means
        the limiter is a no-op and callers can skip the snapshot/clamp
        round-trip entirely."""
        return (self.max_step >= 0
                or self.cooldown_s > 0.0
                or any(v > 0.0 for v in self._cooldown_by_table.values()))

    def snapshot(self, net: pp.pandapowerNet) -> None:
        """Record every ``DiscreteTapControl`` tap_pos at the start of a
        step.  Subsequent :meth:`clamp` calls within the same step
        reference this snapshot."""
        self._snapshot = _snapshot_oltc_taps(net)

    def clamp(
        self,
        net: pp.pandapowerNet,
        time_s: float,
    ) -> List[Tuple[str, int, int, int]]:
        """Enforce the per-step ``±max_step`` clamp AND the wall-clock
        ``cooldown_s`` lock against this step's snapshot.

        Per OLTC, with ``delta = curr - snapshot``:
          * ``delta == 0``  → no-op.
          * ``delta != 0`` AND ``time_s − last_change_time_s < cooldown_s``
            → revert ``tap_pos`` to the snapshot (cooldown not yet
            elapsed); recorded as ``(table, tid, curr, prev)`` in the
            returned list so the caller can log the rate-limit revert.
          * Otherwise → apply the existing ``±max_step`` clamp; if the
            tap actually moved, update ``last_change_time_s[(table,
            tid)] = time_s``.

        Returns the list of OLTCs whose ``tap_pos`` was modified by this
        call (whether by clamp or by cooldown revert), so the caller
        can decide whether to re-run the PF with ``run_control=False``.
        """
        from pandapower.control import DiscreteTapControl
        moved: List[Tuple[str, int, int, int]] = []
        if not self._snapshot:
            return moved
        if not hasattr(net, "controller") or len(net.controller) == 0:
            return moved
        for cid in net.controller.index:
            obj = net.controller.at[cid, "object"]
            if not isinstance(obj, DiscreteTapControl):
                continue
            for table, tid in _controlled_oltc_keys(obj):
                key = (table, tid)
                if key not in self._snapshot:
                    continue
                prev = self._snapshot[key]
                curr = int(net[table].at[tid, "tap_pos"])
                delta = curr - prev
                if delta == 0:
                    continue
                last_t = self._last_change_time_s.get(key, float("-inf"))
                cd = self._cooldown_for(table)
                if cd > 0.0 and (time_s - last_t) < cd:
                    # Cooldown not elapsed: revert any tap movement.
                    net[table].at[tid, "tap_pos"] = int(prev)
                    moved.append((table, tid, curr, prev))
                    continue
                # Apply per-step ±max_step clamp on top of the cooldown gate
                # (max_step < 0 disables this stage).
                if self.max_step >= 0 and abs(delta) > self.max_step:
                    clamped = prev + (self.max_step if delta > 0 else -self.max_step)
                    net[table].at[tid, "tap_pos"] = int(clamped)
                    if clamped != prev:
                        self._last_change_time_s[key] = time_s
                    moved.append((table, tid, prev, int(clamped)))
                else:
                    # The natural move (within ±max_step) is allowed.
                    self._last_change_time_s[key] = time_s
        return moved


def _dump_contingency_diagnostics(
    net: pp.pandapowerNet,
    label: str,
    watch_bus_0idx: Optional[List[int]] = None,
) -> None:
    """Print gen P/Q utilisation, ext_grid load, and watched-bus voltages.

    Used around contingency events to identify whether PF divergence is a
    reactive-power / Q-limit issue vs. a slack redistribution issue.
    """
    print(f"\n  -- Contingency diagnostics: {label} --")

    # Generators: P, Q, slack weight, in_service, Q limits
    if not net.res_gen.empty:
        gen_df = pd.DataFrame(index=net.gen.index)
        gen_df["bus"] = net.gen["bus"]
        gen_df["in_srv"] = net.gen["in_service"]
        gen_df["slack_w"] = net.gen.get("slack_weight", np.nan)
        gen_df["p_mw"] = net.res_gen["p_mw"]
        gen_df["q_mvar"] = net.res_gen["q_mvar"]
        if "min_q_mvar" in net.gen.columns:
            gen_df["q_min"] = net.gen["min_q_mvar"]
        if "max_q_mvar" in net.gen.columns:
            gen_df["q_max"] = net.gen["max_q_mvar"]
        if "vm_pu" in net.gen.columns:
            gen_df["vm_set"] = net.gen["vm_pu"]
        print("  Generators:")
        print(gen_df.to_string(float_format="%.2f"))

    # ext_grid
    if not net.res_ext_grid.empty:
        eg_df = pd.DataFrame(index=net.ext_grid.index)
        eg_df["bus"] = net.ext_grid["bus"]
        eg_df["in_srv"] = net.ext_grid["in_service"]
        eg_df["slack_w"] = net.ext_grid.get("slack_weight", np.nan)
        eg_df["p_mw"] = net.res_ext_grid["p_mw"]
        eg_df["q_mvar"] = net.res_ext_grid["q_mvar"]
        if "vm_pu" in net.ext_grid.columns:
            eg_df["vm_set"] = net.ext_grid["vm_pu"]
        print("  ext_grid:")
        print(eg_df.to_string(float_format="%.2f"))

    # Aggregate: slack weight sum of in-service participants
    if "slack_weight" in net.gen.columns:
        in_srv_mask = net.gen["in_service"]
        sw_sum = float(net.gen.loc[in_srv_mask, "slack_weight"].sum())
        total_sw = float(net.gen["slack_weight"].sum())
        print(f"  slack_weight sum (in-service gens): {sw_sum:.3f} "
              f"/ total configured: {total_sw:.3f}")

    # Watched bus voltages
    if watch_bus_0idx and not net.res_bus.empty:
        buses_in = [b for b in watch_bus_0idx if b in net.res_bus.index]
        if buses_in:
            v_df = net.res_bus.loc[buses_in, ["vm_pu", "va_degree"]]
            print("  Watched bus voltages:")
            print(v_df.to_string(float_format="%.4f"))

    # Aggregate load + gen balance
    p_load = float(net.res_load["p_mw"].sum()) if not net.res_load.empty else 0.0
    q_load = float(net.res_load["q_mvar"].sum()) if not net.res_load.empty else 0.0
    p_gen = float(net.res_gen["p_mw"].sum()) if not net.res_gen.empty else 0.0
    q_gen = float(net.res_gen["q_mvar"].sum()) if not net.res_gen.empty else 0.0
    p_eg = float(net.res_ext_grid["p_mw"].sum()) if not net.res_ext_grid.empty else 0.0
    q_eg = float(net.res_ext_grid["q_mvar"].sum()) if not net.res_ext_grid.empty else 0.0
    p_sgen = float(net.res_sgen["p_mw"].sum()) if not net.res_sgen.empty else 0.0
    q_sgen = float(net.res_sgen["q_mvar"].sum()) if not net.res_sgen.empty else 0.0
    print(f"  Balance: P load={p_load:.1f}  gen={p_gen:.1f}  "
          f"ext_grid={p_eg:.1f}  sgen={p_sgen:.1f}  "
          f"Δ={p_gen + p_eg + p_sgen - p_load:.1f} MW")
    print(f"  Balance: Q load={q_load:.1f}  gen={q_gen:.1f}  "
          f"ext_grid={q_eg:.1f}  sgen={q_sgen:.1f}  "
          f"Δ={q_gen + q_eg + q_sgen - q_load:.1f} Mvar")
    print(f"  -- end diagnostics: {label} --\n")


def _record_dso_group_and_transformer_data(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    dso_ids: List[str],
    dsocontrollers: Dict[str, DSOController],
    dso_group_map: Dict[str, str],
    last_dso_q_set_mvar: Dict[str, Optional[NDArray]],
    hv_info_map: Dict[str, HVNetworkInfo],
) -> None:
    """
    Write DSO transformer- and network-group-level observables into rec.

    Each DSO may have multiple interface transformers (3 per HV sub-network).
    Per-trafo data is keyed by ``"{dso_id}|trafo_{trafo_idx}"``.
    """
    group_q_der: Dict[str, List[float]] = {}
    group_q_der_min: Dict[str, List[float]] = {}
    group_q_der_max: Dict[str, List[float]] = {}
    group_v_min: Dict[str, List[float]] = {}
    group_v_mean: Dict[str, List[float]] = {}
    group_v_max: Dict[str, List[float]] = {}

    for dso_id in dso_ids:
        if dso_id not in dsocontrollers:
            raise KeyError(f"Missing DSO controller '{dso_id}'.")
        if dso_id not in dso_group_map:
            raise KeyError(f"Missing network-group mapping for DSO '{dso_id}'.")

        dso_ctrl = dsocontrollers[dso_id]
        dsocfg = dso_ctrl.config
        group_id = dso_group_map[dso_id]

        # Retrieve per-trafo Q setpoints (vector or None)
        q_set_vec = last_dso_q_set_mvar.get(dso_id)

        # Last absolute capability envelope reported upward by this DSO
        # (set by :meth:`DSOController.generate_capability_message`).  May
        # be ``None`` for the very first record before any capability
        # message has been generated.
        cap_min_vec = dso_ctrl._last_capability_q_iface_min_mvar
        cap_max_vec = dso_ctrl._last_capability_q_iface_max_mvar

        rec.dso_controller_group[dso_id] = group_id

        # Per-trafo recording
        for k, trafo_idx in enumerate(dsocfg.interface_trafo_indices):
            trafo_idx = int(trafo_idx)
            trafo_key = f"{dso_id}|trafo_{trafo_idx}"

            rec.dso_trafo_group[trafo_key] = group_id

            if q_set_vec is not None and k < len(q_set_vec):
                rec.dso_trafo_q_set_mvar[trafo_key] = float(q_set_vec[k])
            elif q_set_vec is not None:
                rec.dso_trafo_q_set_mvar[trafo_key] = float(q_set_vec[0])

            if cap_min_vec is not None and k < len(cap_min_vec):
                rec.dso_trafo_q_cap_min_mvar[trafo_key] = float(cap_min_vec[k])
                rec.dso_trafo_q_cap_max_mvar[trafo_key] = float(cap_max_vec[k])

            if trafo_idx in net.res_trafo3w.index:
                rec.dso_trafo_q_actual_mvar[trafo_key] = float(
                    net.res_trafo3w.at[trafo_idx, "q_hv_mvar"]
                )
                rec.dso_trafo_p_actual_mw[trafo_key] = float(
                    net.res_trafo3w.at[trafo_idx, "p_hv_mw"]
                )
            if trafo_idx in net.trafo3w.index:
                rec.dso_trafo_tap_pos[trafo_key] = int(
                    net.trafo3w.at[trafo_idx, "tap_pos"]
                )

        # DER and voltage group data
        q_der_total = float(net.res_sgen.loc[dsocfg.der_indices, "q_mvar"].sum())
        vm_pu = net.res_bus.loc[dsocfg.voltage_bus_indices, "vm_pu"].to_numpy(dtype=float)

        # Per-DSO DER reactive power capability (sum over the DSO's DERs).
        # Used by the live plotter to draw a shaded headroom band around the
        # DER Q line. Bounds come from the VDE-AR-N 4120 capability curve in
        # actuator_bounds; they depend on the current active power dispatch.
        der_p = net.res_sgen.loc[dsocfg.der_indices, "p_mw"].to_numpy(dtype=float)
        q_min_arr, q_max_arr = dso_ctrl.actuator_bounds.compute_der_q_bounds(der_p)

        group_q_der.setdefault(group_id, []).append(q_der_total)
        group_q_der_min.setdefault(group_id, []).append(float(q_min_arr.sum()))
        group_q_der_max.setdefault(group_id, []).append(float(q_max_arr.sum()))
        if vm_pu.size > 0:
            group_v_min.setdefault(group_id, []).append(float(np.min(vm_pu)))
            group_v_mean.setdefault(group_id, []).append(float(np.mean(vm_pu)))
            group_v_max.setdefault(group_id, []).append(float(np.max(vm_pu)))

    for group_id, values in group_q_der.items():
        rec.dso_group_q_der_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_q_der_min.items():
        rec.dso_group_q_der_min_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_q_der_max.items():
        rec.dso_group_q_der_max_mvar[group_id] = float(np.sum(values))
    for group_id, values in group_v_min.items():
        rec.dso_group_v_min_pu[group_id] = float(np.min(values))
    for group_id, values in group_v_mean.items():
        rec.dso_group_v_mean_pu[group_id] = float(np.mean(values))
    for group_id, values in group_v_max.items():
        rec.dso_group_v_max_pu[group_id] = float(np.max(values))

    # ── HV-group live-plot aggregates (line loading %, DER P, load P/Q) ─────
    _record_hv_group_observables(rec, net, hv_info_map)


def _record_hv_group_observables(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    hv_info_map: Dict[str, HVNetworkInfo],
) -> None:
    """Populate per-HV-group line-loading %, DER P, and load P/Q on the record.

    Works independently of controller state so it can be called from both
    the OFO and local-DSO paths.
    """
    for group_id, hv in hv_info_map.items():
        valid_lines = [li for li in hv.line_indices if li in net.res_line.index]
        if valid_lines:
            loadings = net.res_line.loc[valid_lines, "loading_percent"].to_numpy(dtype=float)
            rec.dso_group_i_max_pct[group_id]  = float(np.nanmax(loadings))
            rec.dso_group_i_mean_pct[group_id] = float(np.nanmean(loadings))
            rec.dso_group_i_min_pct[group_id]  = float(np.nanmin(loadings))
        if hv.sgen_indices:
            sgens = [s for s in hv.sgen_indices if s in net.res_sgen.index]
            if sgens:
                rec.dso_group_der_p_mw[group_id] = float(
                    net.res_sgen.loc[sgens, "p_mw"].sum()
                )
        if hv.load_indices:
            loads = [l for l in hv.load_indices if l in net.res_load.index]
            if loads:
                rec.dso_group_load_p_mw[group_id]    = float(
                    net.res_load.loc[loads, "p_mw"].sum()
                )
                rec.dso_group_load_q_mvar[group_id]  = float(
                    net.res_load.loc[loads, "q_mvar"].sum()
                )


def _record_local_dso_trafo_data(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    hv_info_map: Dict[str, HVNetworkInfo],
) -> None:
    """Populate per-trafo Q/P actuals and tap positions in local-DSO mode."""
    for group_id, hv in hv_info_map.items():
        for k, trafo_idx in enumerate(hv.coupling_trafo_indices):
            t = int(trafo_idx)
            trafo_key = f"{group_id}|trafo_{t}"
            rec.dso_trafo_group[trafo_key] = group_id
            if t in net.res_trafo3w.index:
                rec.dso_trafo_q_actual_mvar[trafo_key] = float(
                    net.res_trafo3w.at[t, "q_hv_mvar"]
                )
                rec.dso_trafo_p_actual_mw[trafo_key] = float(
                    net.res_trafo3w.at[t, "p_hv_mw"]
                )
            if t in net.trafo3w.index:
                rec.dso_trafo_tap_pos[trafo_key] = int(
                    net.trafo3w.at[t, "tap_pos"]
                )


def _record_zone_live_plot_observables(
    rec: MultiTSOIterationRecord,
    net: pp.pandapowerNet,
    zone_defs: Dict[int, ZoneDefinition],
    tn_zone_map: Dict[int, List[int]],
    tie_line_map: Dict[Tuple[int, int], List[int]],
) -> None:
    """Populate per-zone line loadings, balance aggregates, tie-line Q, shunts.

    Called every step (after PF, regardless of run_tso/run_dso) to keep the
    live plotters fed with plant measurements.
    """
    # Per-zone line loadings + zone balance aggregates
    for z, zd in zone_defs.items():
        valid_lines = [li for li in zd.line_indices if li in net.res_line.index]
        if valid_lines:
            loadings = net.res_line.loc[valid_lines, "loading_percent"].to_numpy(dtype=float)
            rec.zone_line_loading_max_pct[z]  = float(np.nanmax(loadings))
            rec.zone_line_loading_mean_pct[z] = float(np.nanmean(loadings))
            rec.zone_line_loading_min_pct[z]  = float(np.nanmin(loadings))

        if zd.tso_der_indices:
            ders = [s for s in zd.tso_der_indices if s in net.res_sgen.index]
            if ders:
                p_arr = net.res_sgen.loc[ders, "p_mw"].to_numpy(dtype=float)
                q_arr = net.res_sgen.loc[ders, "q_mvar"].to_numpy(dtype=float)
                rec.zone_tso_der_p_mw[z]        = p_arr
                rec.zone_balance_der_p_mw[z]    = float(p_arr.sum())
                rec.zone_balance_der_q_mvar[z]  = float(q_arr.sum())

        if zd.gen_indices:
            gens = [g for g in zd.gen_indices if g in net.res_gen.index]
            if gens:
                rec.zone_balance_gen_p_mw[z]   = float(net.res_gen.loc[gens, "p_mw"].sum())
                rec.zone_balance_gen_q_mvar[z] = float(net.res_gen.loc[gens, "q_mvar"].sum())

        tn_bus_set = set(tn_zone_map.get(z, []))
        if tn_bus_set and len(net.load.index) > 0:
            zone_loads = net.load.index[net.load["bus"].isin(tn_bus_set)].tolist()
            zone_loads = [l for l in zone_loads if l in net.res_load.index]
            if zone_loads:
                rec.zone_balance_load_p_mw[z]   = float(
                    net.res_load.loc[zone_loads, "p_mw"].sum()
                )
                rec.zone_balance_load_q_mvar[z] = float(
                    net.res_load.loc[zone_loads, "q_mvar"].sum()
                )

        if zd.pcc_trafo_indices:
            pccs = [t for t in zd.pcc_trafo_indices if t in net.res_trafo3w.index]
            if pccs:
                rec.zone_balance_tso_dso_p_out_mw[z]   = float(
                    net.res_trafo3w.loc[pccs, "p_hv_mw"].sum()
                )
                rec.zone_balance_tso_dso_q_out_mvar[z] = float(
                    net.res_trafo3w.loc[pccs, "q_hv_mvar"].sum()
                )

        # Shunt states — ZoneDefinition.shunt_bus_indices holds the bus
        # indices where TSO-owned shunts are connected.  Map each bus to
        # its row in net.shunt to read the current step.  Order is
        # preserved so the plot tile stays aligned with the zone's
        # actuator vector.
        if getattr(zd, "shunt_bus_indices", None):
            steps: List[int] = []
            for sb in zd.shunt_bus_indices:
                mask = net.shunt["bus"] == sb
                if mask.any():
                    sh_idx = net.shunt.index[mask][0]
                    steps.append(int(net.shunt.at[sh_idx, "step"]))
            rec.zone_tso_shunt_states[z] = np.asarray(steps, dtype=np.int64)
        else:
            rec.zone_tso_shunt_states[z] = np.array([], dtype=np.int64)

    # Inter-zone tie-line Q flow (positive = Q leaves zi toward zj)
    for (zi, zj), line_ids in tie_line_map.items():
        total = 0.0
        any_val = False
        bus_set_i = set(tn_zone_map.get(zi, []))
        for li in line_ids:
            if li not in net.res_line.index:
                continue
            q_from = float(net.res_line.at[li, "q_from_mvar"])
            fb = int(net.line.at[li, "from_bus"])
            signed = q_from if fb in bus_set_i else -q_from
            total += signed
            any_val = True
            # Per-line flow (signed, leaving the lower-id zone zi), recorded
            # unconditionally post-PF so individual ties (e.g. L14) can be
            # isolated regardless of whether the coordinator is active.
            rec.tie_q_mvar[int(li)] = signed
        if any_val:
            rec.zone_tie_q_mvar[(zi, zj)] = total


# =============================================================================
#  Delayed stability analysis helpers
# =============================================================================

def _run_delayed_stability_analysis(
    *,
    config: "MultiTSOConfig",
    time_s: float,
    net,
    coordinator: "MultiTSOCoordinator",
    zone_defs,
    tso_controllers,
    dso_controllers,
    hv_info_map: Dict[str, "HVNetworkInfo"],
    verbose: int,
) -> "MultiZoneStabilityResult":
    """Refresh cross-sensitivities at the current operating point, run
    the multi-zone stability analysis, and save the results (per-zone +
    per-DSO g_w / alpha tables) to a markdown file under
    ``config.result_dir``.  Returns the stability result.
    """
    import os

    if verbose >= 1:
        print()
        print(f"  [stability] Running multi-zone stability analysis at t = {time_s/60.0:.1f} min ...")

    # Refresh the coordinator's cross-sensitivity blocks so the analysis
    # reflects the current operating point (profiles + controller state).
    coordinator.compute_cross_sensitivities()
    coordinator.compute_M_blocks()

    zone_ids_sorted = sorted(zone_defs.keys())
    H_blocks_stab = {k: coordinator.get_H_block(*k)
                     for k in coordinator._H_blocks}
    Q_obj_list = [zone_defs[z].q_obj_diagonal() for z in zone_ids_sorted]
    G_w_list   = [tso_controllers[z].params.g_w for z in zone_ids_sorted]
    actuator_counts = [
        {
            'n_der':  len(zone_defs[z].tso_der_indices),
            'n_pcc':  len(zone_defs[z].pcc_trafo_indices),
            'n_gen':  len(zone_defs[z].gen_indices),
            'n_oltc': len(zone_defs[z].oltc_trafo_indices),
            'n_shunt': len(zone_defs[z].shunt_bus_indices),
        }
        for z in zone_ids_sorted
    ]

    # Build DSO data for C1 analysis
    dso_data_list = []
    for dso_id_key, dso_ctrl in dso_controllers.items():
        dso_cfg_local = dso_ctrl.config
        n_interfaces = len(dso_cfg_local.interface_trafo_indices)
        n_voltage    = len(dso_cfg_local.voltage_bus_indices)
        n_current    = len(dso_cfg_local.current_line_indices)
        q_obj_dso = np.zeros(n_interfaces + n_voltage + n_current)
        q_obj_dso[:n_interfaces] = float(config.g_q)
        if dso_cfg_local.v_setpoints_pu is not None and n_voltage > 0:
            q_obj_dso[n_interfaces:n_interfaces + n_voltage] = float(config.dso_g_v)
        try:
            H_bus_dso = dso_ctrl._build_sensitivity_matrix()
            H_dso = dso_ctrl._expand_H_to_der_level(H_bus_dso)
        except Exception:
            continue
        dso_data_list.append({
            'H': H_dso, 'Q': q_obj_dso,
            'G_w': np.asarray(dso_ctrl.params.g_w).ravel(),
            'id': dso_id_key,
            'alpha': float(dso_ctrl.params.alpha),
            'actuator_counts': {
                'n_der': len(dso_cfg_local.der_indices),
                'n_oltc': len(dso_cfg_local.interface_trafo_indices),
                'n_shunt': len(dso_cfg_local.shunt_bus_indices),
            },
        })

    alpha_tso = tso_controllers[zone_ids_sorted[0]].params.alpha
    stab_result = analyse_multi_zone_stability(
        H_blocks=H_blocks_stab,
        Q_obj_list=Q_obj_list,
        G_w_list=G_w_list,
        zone_ids=zone_ids_sorted,
        zone_names=[f"Zone {z}" for z in zone_ids_sorted],
        actuator_counts=actuator_counts,
        alpha=alpha_tso,
        verbose=(verbose >= 1),
        dso_data=dso_data_list,
        tso_period_s=config.tso_period_s,
        dso_period_s=config.dso_period_s,
    )

    # Write markdown report + machine-readable JSON snapshot
    minutes = int(round(time_s / 60.0))
    md_path = os.path.join(config.result_dir,
                           f"stability_analysis_t{minutes}min.md")
    json_path = os.path.join(config.result_dir,
                             f"tuned_params_t{minutes}min.json")
    try:
        write_stability_analysis_markdown(
            md_path,
            time_s=time_s,
            config=config,
            net=net,
            zone_ids_sorted=zone_ids_sorted,
            zone_defs=zone_defs,
            tso_controllers=tso_controllers,
            dso_controllers=dso_controllers,
            hv_info_map=hv_info_map,
            stab_result=stab_result,
        )
        if verbose >= 1:
            print(f"  Stability report written to: {md_path}")
    except Exception as exc:
        if verbose >= 1:
            print(f"  WARNING: failed to write stability report to {md_path}: {exc}")

    try:
        write_tuned_params_json(
            json_path,
            time_s=time_s,
            zone_ids_sorted=zone_ids_sorted,
            zone_defs=zone_defs,
            tso_controllers=tso_controllers,
            dso_controllers=dso_controllers,
            hv_info_map=hv_info_map,
            stab_result=stab_result,
            pump_result=getattr(coordinator, "_last_pump_result", None),
        )
        if verbose >= 1:
            print(f"  Tuned params snapshot:       {json_path}")
            print(f"  (set config.load_tuned_params_path to this file "
                  f"to skip auto-tune next run)")
            print()
    except Exception as exc:
        if verbose >= 1:
            print(f"  WARNING: failed to write tuned params JSON to {json_path}: {exc}")

    return stab_result
