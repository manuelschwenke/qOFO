"""
sensitivity/network_reduction.py
================================
Build reduced pandapower networks for **per-controller local sensitivity**
computation.

The default operating mode of ``run_multi_tso_dso`` is to give every
TSO and DSO controller the *same* :class:`sensitivity.jacobian.JacobianSensitivities`
instance, built from the full IEEE 39-bus + HV sub-networks plant net.
Each controller's H matrix is then a sub-block of one global Jacobian.

The functions in this module produce an alternative: a Ward-style
*reduced* network per controller that contains only the buses/elements
the controller can actually see and act on, with the rest of the system
condensed into equivalent boundary representations from the cached
operating point.

Two boundary conventions are used (chosen to match the user prompt of
2026-05-27):

* **TSO zone (``build_tso_local_net``):** every boundary is a *PQ load* at
  the boundary bus; the slack lives on a synchronous generator inside the
  zone (the original IEEE 39 slack-gen if it is in the zone, otherwise
  the largest gen is promoted).  Boundaries are:

  - Tie-line far-end bus  → PQ load representing the rest-of-system draw.
  - 3W-trafo primary (HV/TS) bus of every DSO whose sub-network attaches
    in this zone → PQ load = cached ``(p_hv_mw, q_hv_mvar)`` of that
    coupling 3W trafo; the trafo itself and the HV sub-network are
    dropped.

  TSO-owned bipolar shunts originally sit on the LV (20 kV tertiary) side
  of the 3W coupler.  Under reduction the tertiary is gone, so the TSO
  controller's sensitivity Jacobian instead sees a *synthetic shunt at
  the 3W primary bus* with the same ``q_mvar`` per step and the same
  cached step value.  The mapping ``synthetic_shunt_map`` (returned
  alongside the net) tells the runner how to translate between the
  plant tertiary shunt bus and the local synthetic primary bus.

* **DSO sub-network (``build_dso_local_net``):** the boundary is a
  *virtual slack-gen* at the 3W primary bus pinned to ``V_cached``.  No
  explicit PQ load is added there — the slack auto-dispatches the cached
  HV flow at the cached operating point (a separate PQ load at the slack
  bus would only double-count the same injection).  Inside the kept
  region (HV sub-network + 3W trafos + tertiary buses + TSO-owned
  tertiary shunts) every element is preserved unchanged.

The returned reduced nets are converged by ``pp.runpp`` so the caller can
hand them straight to :class:`sensitivity.jacobian.JacobianSensitivities`.

Notes
-----
* All bus indices in the reduced net match those in the original plant
  net (pandapower preserves explicit row labels through deepcopy +
  selective drop), so the controllers' existing index-based lookups
  (``self.config.der_indices`` → ``self.sensitivities.net.sgen.at[i, ...]``)
  keep working without any controller-side change.
* The reduced nets do **not** keep ``distributed_slack=True`` — the
  reduced TSO zone has too few gens (3-4) for the dispatch to make sense
  numerically, and the reduced DSO has none.  The Jacobian we extract
  later runs ``run_control=False, distributed_slack=False`` internally
  inside :class:`JacobianSensitivities.__init__`.
* Synthetic shunts are placed at the 3W primary bus with the same
  ``q_mvar`` per step as the original tertiary shunt.  The 3W coupler's
  series impedance is low enough that the susceptance effect on TN
  voltages is approximately the same magnitude as if the shunt were
  placed at the tertiary, so this approximation preserves the *sign*
  and *order of magnitude* of the TSO MIQP's shunt actuator column.

Author: Manuel Schwenke / Claude Code
Date: 2026-05-27
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandapower as pp

from network.ieee39.meta import HVNetworkInfo, IEEE39NetworkMeta


# ---------------------------------------------------------------------------
#  Result containers
# ---------------------------------------------------------------------------

@dataclass
class TSOLocalNetResult:
    """Return container for :func:`build_tso_local_net`.

    Attributes
    ----------
    net : pp.pandapowerNet
        Reduced pandapower net, converged at the cached operating point.
    synthetic_shunt_map : Dict[int, int]
        Maps each TSO-owned tertiary shunt bus (plant index) to the
        synthetic shunt bus in the reduced net (always the matching 3W
        primary bus).  Empty when the zone has no TSO-owned shunts.
    slack_gen_idx : Optional[int]
        ``net.gen`` index of the slack-gen used in the reduced net.
        ``None`` if no gen was needed as slack (degenerate zone).
    promoted_slack_oltc_indices : Tuple[int, ...]
        ``net.trafo`` indices of machine OLTCs that the controller must
        mark out-of-service in this zone because one of their endpoints
        (typically the LV gen-terminal bus) became the slack-reference
        bus in the reduced net — :meth:`JacobianSensitivities.compute_dV_ds_2w`
        cannot produce a sensitivity column for a trafo touching the
        slack bus.  Empty when the original plant slack-gen is in the
        zone (no promotion needed).
    """
    net: pp.pandapowerNet
    synthetic_shunt_map: Dict[int, int] = field(default_factory=dict)
    slack_gen_idx: Optional[int] = None
    promoted_slack_oltc_indices: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class DSOLocalNetResult:
    """Return container for :func:`build_dso_local_net`."""
    net: pp.pandapowerNet
    virtual_slack_gen_indices: Tuple[int, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
#  TSO reduction
# ---------------------------------------------------------------------------

def build_tso_local_net(
    net: pp.pandapowerNet,
    zone_bus_indices: Iterable[int],
    gen_indices_in_zone: Iterable[int],
    machine_trafo_indices_in_zone: Iterable[int],
    tie_line_indices: Iterable[int],
    tie_line_endpoint_buses: Iterable[int],
    hv_networks_in_zone: Iterable[HVNetworkInfo],
    tso_shunt_buses_in_zone: Iterable[int],
    tso_shunt_q_steps_mvar_in_zone: Iterable[float],
    *,
    verbose: int = 0,
) -> TSOLocalNetResult:
    """Build the reduced TSO network for one zone.

    Parameters mirror the index sets already gathered by the runner.  The
    returned net contains:

    * every TN bus in the zone (``zone_bus_indices``),
    * every generator + its machine 2W trafo + LV terminal bus,
    * every tie line (the in-zone endpoint stays, the far-end bus is
      kept as a "stub" PQ-load bus),
    * every 3W primary bus (HV/TS side) of every DSO in the zone, as a
      PQ-load stub (the 3W trafo and HV sub-network are dropped),
    * one synthetic shunt per TSO-owned tertiary shunt, placed on the
      corresponding 3W primary bus.

    Everything else is deleted.  The slack is the existing IEEE 39 slack-
    gen if it lives in the zone, otherwise the largest gen in the zone is
    promoted to slack.

    Parameters
    ----------
    net : pp.pandapowerNet
        Plant network at the cached operating point.  Must be converged
        (``net.res_*`` tables populated).
    zone_bus_indices : Iterable[int]
        TN bus indices that belong to this zone (TN-only — gen terminal
        buses are added below from ``machine_trafo_indices_in_zone``).
    gen_indices_in_zone : Iterable[int]
        ``net.gen`` indices in the zone.
    machine_trafo_indices_in_zone : Iterable[int]
        ``net.trafo`` indices for the machine 2W trafos of the zone's
        gens.  Their LV (terminal) buses are added to the keep set.
    tie_line_indices : Iterable[int]
        ``net.line`` indices of tie lines monitored by this zone.
    tie_line_endpoint_buses : Iterable[int]
        IN-ZONE endpoint of each tie line (parallel to
        ``tie_line_indices``).
    hv_networks_in_zone : Iterable[HVNetworkInfo]
        HV sub-network metadata objects whose ``zone`` matches this zone.
        Their coupling 3W trafos are *dropped*; their primary buses are
        kept as PQ-load stubs.
    tso_shunt_buses_in_zone : Iterable[int]
        Plant tertiary bus index of each TSO-owned shunt in this zone.
    tso_shunt_q_steps_mvar_in_zone : Iterable[float]
        Per-shunt rated Mvar per step (same order).

    Returns
    -------
    TSOLocalNetResult
    """
    sub = copy.deepcopy(net)

    zone_bus_set: set = set(int(b) for b in zone_bus_indices)
    gen_set: set = set(int(g) for g in gen_indices_in_zone)
    machine_trafos_in_zone: List[int] = [int(t) for t in machine_trafo_indices_in_zone]
    tie_lines: List[int] = [int(li) for li in tie_line_indices]
    tie_in_endpoints: List[int] = [int(b) for b in tie_line_endpoint_buses]
    hv_list: List[HVNetworkInfo] = list(hv_networks_in_zone)
    shunt_buses: List[int] = [int(b) for b in tso_shunt_buses_in_zone]
    shunt_q_steps: List[float] = [float(q) for q in tso_shunt_q_steps_mvar_in_zone]

    # ── 1. Compute keep-bus set ───────────────────────────────────────────
    keep_buses: set = set(zone_bus_set)

    # Add LV terminal buses of in-zone machine trafos (gen terminals)
    for t in machine_trafos_in_zone:
        keep_buses.add(int(sub.trafo.at[t, "lv_bus"]))
        keep_buses.add(int(sub.trafo.at[t, "hv_bus"]))

    # Add tie-line far-end buses
    far_end_buses: List[Tuple[int, int]] = []  # (line_idx, far_bus)
    for li, in_bus in zip(tie_lines, tie_in_endpoints):
        if li not in sub.line.index:
            continue
        from_bus = int(sub.line.at[li, "from_bus"])
        to_bus = int(sub.line.at[li, "to_bus"])
        if in_bus == from_bus:
            far = to_bus
        elif in_bus == to_bus:
            far = from_bus
        else:
            # in_bus doesn't match either endpoint — skip
            continue
        keep_buses.add(far)
        far_end_buses.append((li, far))

    # Add 3W primary buses for DSOs in this zone
    primary_bus_for_3w: Dict[int, int] = {}   # 3w_idx → primary bus
    for hv in hv_list:
        for t3w in hv.coupling_trafo_indices:
            if t3w not in sub.trafo3w.index:
                continue
            primary = int(sub.trafo3w.at[t3w, "hv_bus"])
            keep_buses.add(primary)
            primary_bus_for_3w[int(t3w)] = primary

    # ── 2. Capture cached boundary flows BEFORE editing tables ────────────
    # Tie-line far-end PQ-load values: net injection from "rest of system"
    # into b_far at cached state = +p_xxx_mw_at_far (pandapower's
    # res_line.p_xxx is power INTO the line at side xxx, so power into
    # bus from rest-of-system = +p_xxx_mw_at_far).  Load draws this much
    # ⇒ load.p_mw = -p_xxx_mw_at_far.
    tie_load_specs: List[Tuple[int, float, float]] = []  # (bus, p, q)
    for li, far in far_end_buses:
        if li not in sub.res_line.index:
            continue
        from_bus = int(sub.line.at[li, "from_bus"])
        if far == from_bus:
            p_far = float(sub.res_line.at[li, "p_from_mw"])
            q_far = float(sub.res_line.at[li, "q_from_mvar"])
        else:
            p_far = float(sub.res_line.at[li, "p_to_mw"])
            q_far = float(sub.res_line.at[li, "q_to_mvar"])
        tie_load_specs.append((far, -p_far, -q_far))

    # 3W primary PQ-load values: trafo was drawing (p_hv_mw, q_hv_mvar)
    # from the TN at cached state; after we delete the trafo, replace by a
    # load that draws the same.
    primary_load_specs: List[Tuple[int, float, float, int]] = []
    # (primary_bus, p_mw, q_mvar, trafo3w_idx)
    for t3w, primary in primary_bus_for_3w.items():
        if t3w not in sub.res_trafo3w.index:
            continue
        p_hv = float(sub.res_trafo3w.at[t3w, "p_hv_mw"])
        q_hv = float(sub.res_trafo3w.at[t3w, "q_hv_mvar"])
        primary_load_specs.append((primary, p_hv, q_hv, t3w))

    # Cached voltage at every primary bus (for synthetic-shunt q_mvar
    # scaling, if we choose to scale; currently we use a 1:1 mapping).
    primary_v_cached: Dict[int, float] = {}
    for _, primary in primary_bus_for_3w.items():
        primary_v_cached[primary] = float(sub.res_bus.at[primary, "vm_pu"])

    # ── 3. Keep the in-zone PCC 3W trafos (with their primary/MV/LV) ──
    # The user prompt asked for "primary-side PQ injection" but the TSO
    # controller's :meth:`_build_sensitivity_matrix` requires a *live*
    # 3W coupler row in ``net.trafo3w`` so that
    # ``compute_dQtrafo3w_hv_*`` can populate the Q_PCC output rows AND
    # ``compute_dV_dQ_der`` at the primary bus can populate the
    # Q_PCC,set actuator columns.  Without a live trafo, those blocks
    # come out as zeros or NaN and the OFO sees no V-tracking leverage
    # from PCC dispatch (observed symptom: TSO commands a constant
    # Q_PCC,set forever).
    #
    # We therefore keep every PCC 3W trafo + its MV bus + LV (tertiary)
    # bus alive, plus the primary bus.  The Ward equivalent moves *one
    # bus deeper* than the user's literal spec: the PQ load lands on
    # the MV-side bus (the boundary between the trafo and the dropped
    # HV sub-network).  Semantically the user's intent is preserved —
    # the entire HV sub-network behind the trafo is replaced by a
    # constant PQ injection — only the injection bus moves from the
    # primary (HV side, TN) to the MV side.  TSO-owned tertiary shunts
    # (on the LV bus) stay where they are.
    pcc_trafo3w_in_zone = set(int(t) for t in primary_bus_for_3w.keys())
    pcc_t3w_mv_buses: List[Tuple[int, int]] = []   # (mv_bus, trafo_idx)
    pcc_t3w_lv_buses: List[int] = []
    for hv in hv_list:
        for t, mv_bus, lv_bus in zip(
            hv.coupling_trafo_indices,
            hv.coupling_hv_bus_indices,
            hv.coupling_lv_bus_indices,
        ):
            if int(t) in pcc_trafo3w_in_zone:
                pcc_t3w_mv_buses.append((int(mv_bus), int(t)))
                pcc_t3w_lv_buses.append(int(lv_bus))
                keep_buses.add(int(mv_bus))
                keep_buses.add(int(lv_bus))

    # Cached MV-side flow (Ward injection value) — read BEFORE we touch
    # the trafo3w table or the surrounding net.  Sign: pandapower's
    # ``q_mv_mvar`` is the Q flowing INTO the trafo at the MV bus
    # (load convention from the bus's perspective).  After we drop the
    # HV sub-network, the MV bus loses its downstream load that used
    # to draw this Q; the new ``pp.create_load`` substitutes for it.
    mv_load_specs: List[Tuple[int, float, float]] = []  # (mv_bus, p_mw, q_mvar)
    for mv_bus, t in pcc_t3w_mv_buses:
        if t not in sub.res_trafo3w.index:
            continue
        # Power flowing INTO the trafo at the MV bus = power leaving the
        # bus through the trafo. Power that the bus consumed from the
        # rest of the HV sub-network = -q_mv_mvar (= what flows into the
        # bus from the rest of the sub-net to be sent through the trafo).
        # When we delete the HV sub-network, we must replace that draw
        # by a new load at the MV bus with the SAME consumed power.
        # Cached ``q_mv_mvar`` IS positive when the bus loses Q to the
        # trafo, so the load that USED to supply it is +q_mv_mvar in
        # magnitude, modelled here as a (negative-q) generator or a
        # load with q_mvar = -q_mv_mvar.  Equivalently we add a load
        # equal to (-p_mv_mw, -q_mv_mvar) so the bus net injection
        # stays at the cached operating point.
        p_mv = float(sub.res_trafo3w.at[t, "p_mv_mw"])
        q_mv = float(sub.res_trafo3w.at[t, "q_mv_mvar"])
        mv_load_specs.append((mv_bus, -p_mv, -q_mv))

    # ── 4. Drop HV sub-network elements (buses, lines, sgens, loads, …) ─
    hv_buses_to_drop: set = set()
    for hv in hv_list:
        for b in hv.bus_indices:
            hv_buses_to_drop.add(int(b))
        # Tertiary buses are explicit in coupling_lv_bus_indices
        for b in hv.coupling_lv_bus_indices:
            hv_buses_to_drop.add(int(b))
        # MV-side (110 kV) coupling bus indices
        for b in hv.coupling_hv_bus_indices:
            hv_buses_to_drop.add(int(b))
    if hv_buses_to_drop:
        # Use pp.drop_buses to cascade-delete attached elements cleanly.
        # Exclude any bus we still want to keep (primary + MV + LV of
        # the PCC trafos in this zone, plus the original TN buses).
        hv_buses_to_drop -= keep_buses
        if hv_buses_to_drop:
            pp.drop_buses(sub, list(hv_buses_to_drop))

    # Strip every element attached to the surviving MV/LV stubs except
    # the trafo itself (and any TSO-owned tertiary shunt on the LV bus).
    # Specifically: drop the original load that used to absorb the MV
    # flow downstream — the new ``mv_load_specs`` adds a fresh load
    # with the cached Ward injection.
    for b in {bus for (bus, _t) in pcc_t3w_mv_buses}:
        # Drop loads/sgens/gens at b (downstream HV sub-net loads).
        for tbl in ("load", "sgen", "gen"):
            df = getattr(sub, tbl)
            if not df.empty:
                mask = df["bus"] == b
                if mask.any():
                    df.drop(index=df.index[mask], inplace=True)
        # Drop lines attached to b (HV sub-net lines going downstream).
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            sub.line.drop(index=li, inplace=True)
    for b in pcc_t3w_lv_buses:
        # Drop loads/sgens/gens at the tertiary (but keep TSO shunts).
        for tbl in ("load", "sgen", "gen"):
            df = getattr(sub, tbl)
            if not df.empty:
                mask = df["bus"] == b
                if mask.any():
                    df.drop(index=df.index[mask], inplace=True)
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            sub.line.drop(index=li, inplace=True)

    # ── 5. Drop every bus not in keep_buses ───────────────────────────────
    remaining_buses = set(int(b) for b in sub.bus.index)
    extra_drop = remaining_buses - keep_buses
    if extra_drop:
        pp.drop_buses(sub, list(extra_drop))

    # ── 6. Strip every element attached to far-end "stub" buses ──────────
    # After pp.drop_buses, the keep-buses are still alive but the tie-line
    # far-end buses sit outside the zone — they exist in the reduced net
    # only as anchors for the tie line.  We strip every attached element
    # (loads, sgens, gens, shunts, other lines) so the far-end becomes a
    # pure PQ stub, then add a fresh Ward-equivalent load in step 7.
    #
    # 3W-primary buses are *also* regular zone-TN buses with legitimate
    # in-zone line / load / gen attachments; we already removed the
    # offending 3W trafo (and its HV sub-network) in step 3-4.  Their
    # remaining TN attachments must stay — otherwise the zone's TN
    # backbone is severed.  We only add the Ward-equivalent load (step 7)
    # to substitute for the dropped 3W's HV-side draw.
    far_end_set = set(b for _, b in far_end_buses)

    for b in far_end_set:
        # Drop loads at b
        mask = sub.load["bus"] == b
        if mask.any():
            sub.load.drop(index=sub.load.index[mask], inplace=True)
        # Drop sgens at b
        mask = sub.sgen["bus"] == b
        if mask.any():
            sub.sgen.drop(index=sub.sgen.index[mask], inplace=True)
        # Drop gens at b
        mask = sub.gen["bus"] == b
        if mask.any():
            sub.gen.drop(index=sub.gen.index[mask], inplace=True)
        # Drop shunts at b
        if not sub.shunt.empty:
            mask = sub.shunt["bus"] == b
            if mask.any():
                sub.shunt.drop(index=sub.shunt.index[mask], inplace=True)
        # Drop every line attached EXCEPT the tie line itself
        keep_lines = {li for li, fb in far_end_buses if fb == b}
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            if int(li) not in keep_lines:
                sub.line.drop(index=li, inplace=True)
        # Drop every 2W trafo with a leg at b (the far-end has no machine
        # gen by construction, but defensive).
        mask_tr = (sub.trafo["hv_bus"] == b) | (sub.trafo["lv_bus"] == b)
        for t in sub.trafo.index[mask_tr]:
            sub.trafo.drop(index=t, inplace=True)

    # ── 7. Add equivalent PQ loads at boundary stubs ──────────────────────
    # Tie-line far-end stubs get a Ward load representing the rest-of-
    # system net injection.  The 3W coupler's HV sub-network (downstream
    # of the MV bus) is represented by a load at the MV bus — the
    # trafo itself stays alive and the primary bus remains a normal
    # zone-TN bus with its existing TN connectivity intact.
    for far, p_load, q_load in tie_load_specs:
        if far in sub.bus.index:
            pp.create_load(sub, bus=int(far), p_mw=p_load, q_mvar=q_load,
                           name="WARD_TIE")
    for mv_bus, p_load, q_load in mv_load_specs:
        if mv_bus in sub.bus.index:
            pp.create_load(sub, bus=int(mv_bus), p_mw=p_load, q_mvar=q_load,
                           name="WARD_3W_MV")

    # ── 8. Synthetic shunts at 3W primary buses ───────────────────────────
    synthetic_shunt_map: Dict[int, int] = {}
    if shunt_buses:
        # For each TSO-owned tertiary shunt, look up which 3W trafo it
        # belongs to (via the plant net's res_trafo3w), then map to the
        # corresponding primary bus we've kept.  hv_list already restricts
        # to this zone's DSOs.
        plant_3w_lv: Dict[int, int] = {}
        for hv in hv_list:
            for t3w, lv_bus, hv_bus in zip(
                hv.coupling_trafo_indices,
                hv.coupling_lv_bus_indices,
                hv.coupling_ieee_buses,
            ):
                plant_3w_lv[int(lv_bus)] = int(hv_bus)
        for tert_bus, q_step in zip(shunt_buses, shunt_q_steps):
            primary = plant_3w_lv.get(int(tert_bus))
            if primary is None or primary not in sub.bus.index:
                continue
            # Read the cached step from the plant net so the synthetic
            # susceptance matches the plant susceptance at the cached
            # operating point.
            plant_shunt_mask = net.shunt["bus"] == int(tert_bus)
            cached_step = 0
            if plant_shunt_mask.any():
                cached_step = int(
                    net.shunt.at[net.shunt.index[plant_shunt_mask][0], "step"]
                )
            pp.create_shunt(
                sub, bus=int(primary), q_mvar=q_step, p_mw=0.0,
                step=cached_step, max_step=10,
                name="SYNTH_TSO_TERTIARY_SHUNT",
            )
            synthetic_shunt_map[int(tert_bus)] = int(primary)

    # ── 9. Slack handling ─────────────────────────────────────────────────
    # Pick a slack gen inside the zone (preserve existing slack-gen if
    # it's in the zone; otherwise promote the largest gen).  Track
    # whether the slack-gen was newly promoted — in that case its
    # machine OLTC sits on the slack-reference bus and the Jacobian's
    # ``compute_dV_ds_2w`` cannot produce a sensitivity column for it.
    # The caller (runner) flags that trafo as out-of-service on the
    # controller via ``promoted_slack_oltc_indices``.
    slack_gen_idx: Optional[int] = None
    slack_promoted: bool = False
    if not sub.gen.empty:
        zone_gen_mask = sub.gen.index.isin([int(g) for g in gen_set])
        zone_gens_all = sub.gen.index[zone_gen_mask].tolist()
        # Filter to in-service gens — a tripped gen can't anchor a slack
        # reference (pp.runpp emits 'No reference bus is available').
        in_service_mask = (
            sub.gen.loc[zone_gens_all, "in_service"].astype(bool)
            if "in_service" in sub.gen.columns
            else None
        )
        if in_service_mask is not None:
            zone_gens = [
                int(g) for g, ok in zip(zone_gens_all, in_service_mask) if bool(ok)
            ]
        else:
            zone_gens = [int(g) for g in zone_gens_all]
        if zone_gens:
            # Identify existing in-service slack-gens
            if "slack" in sub.gen.columns:
                existing_slacks = [
                    int(g) for g in zone_gens
                    if bool(sub.gen.at[g, "slack"])
                ]
            else:
                existing_slacks = []
            if existing_slacks:
                # Keep the first existing slack; clear any others
                slack_gen_idx = existing_slacks[0]
                for g in sub.gen.index:
                    if "slack" in sub.gen.columns:
                        sub.gen.at[g, "slack"] = (int(g) == slack_gen_idx)
            else:
                # Promote the largest in-service gen in the zone to slack
                sn_series = (
                    sub.gen.loc[zone_gens, "sn_mva"]
                    if "sn_mva" in sub.gen.columns else None
                )
                if sn_series is not None and not sn_series.empty:
                    slack_gen_idx = int(sn_series.idxmax())
                else:
                    slack_gen_idx = int(zone_gens[0])
                if "slack" not in sub.gen.columns:
                    sub.gen["slack"] = False
                sub.gen["slack"] = False
                sub.gen.at[slack_gen_idx, "slack"] = True
                slack_promoted = True

    # Machine OLTC(s) attached to the promoted slack gen's terminal bus
    # — these need to be flagged OOS on the controller's mask.  Look up
    # via the trafo's LV bus (machine trafos have lv_bus == gen terminal).
    promoted_slack_oltc_indices: List[int] = []
    if slack_promoted and slack_gen_idx is not None:
        slack_bus = int(sub.gen.at[slack_gen_idx, "bus"])
        for t in machine_trafos_in_zone:
            if t not in sub.trafo.index:
                continue
            if int(sub.trafo.at[t, "lv_bus"]) == slack_bus:
                promoted_slack_oltc_indices.append(int(t))
            elif int(sub.trafo.at[t, "hv_bus"]) == slack_bus:
                promoted_slack_oltc_indices.append(int(t))

    # Clear any external grid (the original network has none on IEEE 39
    # since the slack-gen replaced the ext_grid in build_ieee39_net, but
    # we guard defensively).
    if not sub.ext_grid.empty:
        # Drop external grids that are inside the zone (they conflict with
        # the slack-gen).  Keep none — we use the gen-side slack.
        sub.ext_grid.drop(index=sub.ext_grid.index, inplace=True)

    if slack_gen_idx is None and sub.ext_grid.empty:
        # No slack — return un-converged net; caller will fail at
        # JacobianSensitivities.
        if verbose >= 1:
            print("  [build_tso_local_net] no slack candidate; returning empty result")
        return TSOLocalNetResult(net=sub, synthetic_shunt_map=synthetic_shunt_map,
                                  slack_gen_idx=None)

    # ── 10. Converge the reduced net ──────────────────────────────────────
    # Try ``init='results'`` first to warm-start NR from the cached plant
    # state (preserved through deepcopy).  This is necessary now that the
    # reduced net keeps the 3W coupler trafos with their MV/LV stubs —
    # a flat start often fails to converge under multiple coupler trafos
    # at off-nominal tap positions.  Fall back to flat if results-start
    # diverges.  Either way ``net._ppc['internal']['J']`` is populated
    # because NR runs at least one iteration to verify the warm start
    # (results-start mismatches enough to require a Jacobian build).
    # Apply a small numerical kick so NR runs ≥ 1 iteration (otherwise
    # results-init can converge in 0 steps and leave J unpopulated).
    if not sub.bus.empty:
        first_bus = int(sub.bus.index[0])
        if first_bus in sub.res_bus.index:
            sub.res_bus.at[first_bus, "vm_pu"] = float(
                sub.res_bus.at[first_bus, "vm_pu"]
            ) + 1e-8
    try:
        pp.runpp(
            sub,
            run_control=False,
            distributed_slack=False,
            calculate_voltage_angles=True,
            enforce_q_lims=False,
            max_iteration=50,
            init="results",
        )
    except Exception:
        pp.runpp(
            sub,
            run_control=False,
            distributed_slack=False,
            calculate_voltage_angles=True,
            enforce_q_lims=False,
            max_iteration=200,
            init="flat",
        )

    if verbose >= 2:
        print(f"  [build_tso_local_net] reduced net: "
              f"{len(sub.bus)} buses, {len(sub.line)} lines, "
              f"{len(sub.gen)} gens, {len(sub.load)} loads, "
              f"{len(sub.shunt)} shunts, slack_gen={slack_gen_idx}")

    return TSOLocalNetResult(
        net=sub,
        synthetic_shunt_map=synthetic_shunt_map,
        slack_gen_idx=slack_gen_idx,
        promoted_slack_oltc_indices=tuple(promoted_slack_oltc_indices),
    )


# ---------------------------------------------------------------------------
#  DSO reduction
# ---------------------------------------------------------------------------

def build_dso_local_net(
    net: pp.pandapowerNet,
    hv_info: HVNetworkInfo,
    *,
    verbose: int = 0,
) -> DSOLocalNetResult:
    """Build the reduced DSO network for one HV sub-network.

    Kept elements:

    * Every 110 kV bus in ``hv_info.bus_indices``.
    * Every 20 kV tertiary bus in ``hv_info.coupling_lv_bus_indices``.
    * Every MV-side coupling bus in ``hv_info.coupling_hv_bus_indices``
      (these are typically the same as the HV sub-network's 110 kV
      buses, but listed separately).
    * Every coupling 3W trafo (controlled by the DSO via OLTC).
    * Every 3W primary bus (HV/TS side) — kept as a *virtual slack-gen*
      pinned to V_cached.
    * Every line, load, sgen, shunt internal to the sub-network or
      attached to the kept buses (including the TSO-owned tertiary shunt
      so the DSO can see its disturbance effect).

    Dropped: every bus, line, trafo, etc., that is not in the keep set
    (i.e., the entire TN backbone, all other zones, all other DSOs).

    Parameters
    ----------
    net : pp.pandapowerNet
        Plant network at the cached operating point.
    hv_info : HVNetworkInfo
        Metadata for the HV sub-network whose local Jacobian we build.

    Returns
    -------
    DSOLocalNetResult
    """
    sub = copy.deepcopy(net)

    # ── 1. Build the keep-bus set ─────────────────────────────────────────
    keep_buses: set = set(int(b) for b in hv_info.bus_indices)
    keep_buses.update(int(b) for b in hv_info.coupling_lv_bus_indices)
    keep_buses.update(int(b) for b in hv_info.coupling_hv_bus_indices)
    primary_buses: List[int] = [int(b) for b in hv_info.coupling_ieee_buses]
    keep_buses.update(primary_buses)

    # ── 2. Cached V at primary buses ─────────────────────────────────────
    primary_v_cached: Dict[int, float] = {}
    for b in primary_buses:
        if b in sub.res_bus.index:
            primary_v_cached[b] = float(sub.res_bus.at[b, "vm_pu"])
        else:
            primary_v_cached[b] = 1.0  # fallback

    # ── 3. Drop everything not in keep_buses ──────────────────────────────
    # First drop other trafo3w (other DSOs' couplers).  Then pp.drop_buses
    # on the remaining out-of-set TN/other-DSO buses cascades to lines,
    # loads, sgens, gens, shunts.
    other_t3w = [
        int(t) for t in sub.trafo3w.index
        if int(t) not in [int(x) for x in hv_info.coupling_trafo_indices]
    ]
    if other_t3w:
        sub.trafo3w.drop(index=other_t3w, inplace=True)

    remaining_buses = set(int(b) for b in sub.bus.index)
    extra_drop = remaining_buses - keep_buses
    if extra_drop:
        pp.drop_buses(sub, list(extra_drop))

    # ── 4. Strip every element at the primary buses EXCEPT 3W trafos ─────
    # The primary buses become slack-gen anchors; everything else there
    # (loads from upstream zone, gens, other 2W trafos, lines, sgens) goes
    # away.
    for b in primary_buses:
        if b not in sub.bus.index:
            continue
        # Loads
        mask = sub.load["bus"] == b
        if mask.any():
            sub.load.drop(index=sub.load.index[mask], inplace=True)
        # Sgens
        mask = sub.sgen["bus"] == b
        if mask.any():
            sub.sgen.drop(index=sub.sgen.index[mask], inplace=True)
        # Gens (including the original IEEE 39 slack-gen if one was here)
        mask = sub.gen["bus"] == b
        if mask.any():
            sub.gen.drop(index=sub.gen.index[mask], inplace=True)
        # Shunts attached to the primary bus
        if not sub.shunt.empty:
            mask = sub.shunt["bus"] == b
            if mask.any():
                sub.shunt.drop(index=sub.shunt.index[mask], inplace=True)
        # Lines attached to the primary bus (any remaining TN line stub)
        mask_line = (sub.line["from_bus"] == b) | (sub.line["to_bus"] == b)
        for li in sub.line.index[mask_line]:
            sub.line.drop(index=li, inplace=True)
        # 2W trafos with a leg at the primary bus
        mask_tr = (sub.trafo["hv_bus"] == b) | (sub.trafo["lv_bus"] == b)
        for t in sub.trafo.index[mask_tr]:
            sub.trafo.drop(index=t, inplace=True)

    # ── 5. Add virtual slack-gens at each primary bus ─────────────────────
    virtual_slacks: List[int] = []
    if "slack" not in sub.gen.columns:
        sub.gen["slack"] = False
    else:
        # Clear any stray slack flag inherited from the original net.
        sub.gen["slack"] = False
    if not sub.ext_grid.empty:
        sub.ext_grid.drop(index=sub.ext_grid.index, inplace=True)
    for k, b in enumerate(primary_buses):
        if b not in sub.bus.index:
            continue
        v_cached = primary_v_cached.get(b, 1.0)
        # Only the first primary bus becomes the true slack; any
        # additional primary buses (multi-trafo DSOs) get pinned via PV
        # gens at the same V_cached.  pandapower allows only one slack.
        is_slack = (k == 0) or all(
            int(g) not in sub.gen.index or not bool(sub.gen.at[g, "slack"])
            for g in sub.gen.index
        )
        is_slack = (k == 0)
        gi = pp.create_gen(
            sub, bus=int(b), p_mw=0.0, vm_pu=float(v_cached),
            slack=is_slack,
            min_p_mw=-1e6, max_p_mw=1e6,
            min_q_mvar=-1e6, max_q_mvar=1e6,
            name=f"WARD_DSO_BOUNDARY_{k}",
        )
        virtual_slacks.append(int(gi))

    # ── 6. Converge ───────────────────────────────────────────────────────
    # Use init='flat' so NR actually runs (and the Jacobian gets stored —
    # see comment in :func:`build_tso_local_net`).
    pp.runpp(
        sub,
        run_control=False,
        distributed_slack=False,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        max_iteration=100,
        init="flat",
    )

    if verbose >= 2:
        print(f"  [build_dso_local_net {hv_info.net_id}] reduced net: "
              f"{len(sub.bus)} buses, {len(sub.line)} lines, "
              f"{len(sub.gen)} gens, {len(sub.load)} loads, "
              f"{len(sub.shunt)} shunts, "
              f"virtual_slacks={virtual_slacks}")

    return DSOLocalNetResult(
        net=sub,
        virtual_slack_gen_indices=tuple(virtual_slacks),
    )
