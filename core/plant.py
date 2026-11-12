#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/plant.py
=============
Plant abstraction for OFO-in-the-loop simulation (RMS build plan, Phase 6).

The OFO cascade never sees the plant model -- it acts through actuator
setpoints and observes measurements.  This module makes that boundary an
explicit interface so the same controller stack can run against

* :class:`PandapowerStaticPlant` -- the quasi-static pandapower plant used
  by every experiment today (power flow = the plant settles instantly
  within one dispatch interval), and
* ``pf.plant.PowerFactoryPlant`` -- the DIgSILENT RMS plant (Phase 6b),
  which advances a phasor simulation between dispatches.

Interface contract
------------------
``apply_u(writes)``
    Write actuator setpoints (:class:`ActuatorWrites`, keyed by pandapower
    element indices -- the shared namespace both plants understand; PF
    object ``loc_name`` embeds the same indices, see docs/pf_naming.md).
``advance(duration_s)``
    Let the plant respond for one dispatch interval.  Static plant: one
    converged power flow (duration ignored -- the quasi-static assumption).
    RMS plant: continue the simulation by ``duration_s``.
``read_y()``
    Return the *measurement image*: a ``pandapowerNet`` whose ``res_*``
    tables hold the plant state at the current instant (t_k^-).  The
    static plant returns its own net; the RMS plant harvests paused-state
    values into a mirror net.  Downstream, ``core.measurement.measure_*``
    and the controllers consume this image unchanged -- they cannot tell
    which plant produced it.

The DER convention follows the w-shift actuator mode used by every runner
(experiments/helpers/plant_io.py): a DER Q command is ``q_set_mvar`` at a
V_ref reanchored to the latest measured bus voltage; the plant-side
``QVLocalLoop`` resolves the droop response inside ``advance``.

Author: Manuel Schwenke / Claude Code (2026-07-20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, runtime_checkable

import pandapower as pp
from pandapower.control import DiscreteTapControl
from pandapower.auxiliary import ControllerNotConverged, LoadflowNotConverged

from core.measurement_noise import measured_der_voltage

#: Plant solves in which the local tap relays had to be held (see
#: :func:`runpp_holding_taps`).
HELD_TAP_SOLVES = 0


def runpp_holding_taps(net: pp.pandapowerNet, **kwargs) -> None:
    """``pp.runpp``; if the plant-side control loop does not converge, re-solve
    once with the local tap relays (``DiscreteTapControl``) held.

    With few synchronous machines on line (unit commitment, 2026-10-31) a
    coupler relay can hunt inside one control loop against the DER Q(V)
    loops: a tap step moves the voltage across the band, the Q(V) loops move
    it back.  Each loop converges alone.  A real relay acts with a time delay
    and does not hunt within one dispatch interval, so the relays keep their
    positions for this solve; the per-step tap-rate limiter acts as before.
    Solves that converge are untouched (bit-identical to plain ``pp.runpp``).
    """
    global HELD_TAP_SOLVES
    try:
        pp.runpp(net, **kwargs)
        return
    except ControllerNotConverged:
        if not kwargs.get("run_control") or not len(net.controller):
            raise
        if not _tap_relays(net):
            raise
    runpp_relays_held(net, **kwargs)
    HELD_TAP_SOLVES += 1
    print(f"  [plant] tap relays held for one solve (hunting against the "
          f"Q(V) loops); {HELD_TAP_SOLVES} so far")


def _tap_relays(net: pp.pandapowerNet) -> list:
    """Indices of the in-service ``DiscreteTapControl`` relays of ``net``."""
    if not hasattr(net, "controller") or not len(net.controller):
        return []
    return [i for i in net.controller.index
            if isinstance(net.controller.at[i, "object"], DiscreteTapControl)
            and bool(net.controller.at[i, "in_service"])]


def runpp_relays_held(net: pp.pandapowerNet, **kwargs) -> None:
    """``pp.runpp(run_control=True)`` with the tap relays held at their
    current positions and every other plant-side controller active.

    The re-solve after the OLTC rate limiter has moved relay taps back
    (FIX 2026-11-07).  It used to be ``run_control=False``, which kept the DER
    Q(V) loops (and the slack Q box) at the state they had reached with the
    relays free: the relays had brought the voltage into their band, which is
    also the droop's dead band, so the STS droops sat at Q ~ 0 while the
    voltage fell back to 0.94-0.98 pu (no-access variants, E2 load step;
    docs/daily_log/11_2026/2026-11-07b_sts_qv_frozen_by_oltc_limiter_resolve.md).
    """
    kwargs["run_control"] = True
    relays = _tap_relays(net)
    if relays:
        net.controller.loc[relays, "in_service"] = False
    try:
        pp.runpp(net, **kwargs)
    finally:
        if relays:
            net.controller.loc[relays, "in_service"] = True


def snapshot_results(net: pp.pandapowerNet) -> dict:
    """Copies of every result table (``res_*``) and of the DER outputs, the warm
    start of :func:`runpp_warm_staged`.  All tables, not only ``res_bus``: a warm
    start also reads the star-point voltages of the three-winding transformers
    from ``res_trafo3w``, and a diverged solve clears them (2026-11-10)."""
    snap = {k: net[k].copy() for k in list(net.keys())
            if k.startswith("res_") and hasattr(net[k], "copy")}
    snap["_sgen_q_mvar"] = net.sgen["q_mvar"].copy()
    return snap


def _restore_results(net: pp.pandapowerNet, snap: dict) -> None:
    for k, v in snap.items():
        if k == "_sgen_q_mvar":
            net.sgen["q_mvar"] = v.copy()
        else:
            net[k] = v.copy()


def runpp_warm_staged(net: pp.pandapowerNet, snap: dict, **kwargs) -> str:
    """Warm, staged re-solve after a diverged end-of-step power flow, tried before
    the flat-start fallback (FIX 2026-11-10).

    Case 1 SVR, scenario A, at the 250 Mvar connection at bus 11: the end-of-step
    PF with Q limits and every local loop active diverged, and so did the
    flat-start retry without Q limits, although the plant has a solution there
    (G3 at its 400 Mvar limit; daily log 2026-11-10h).  Starting from the last
    converged state ``snap`` (:func:`snapshot_results`):

    1. relays held, Q(V) loops active, no Q limits -- the post-contingency
       re-solve that converged in that step;
    2. from that solution, the Q limits as requested, Q(V) loops active;
    3. if 2 fails: the Q limits with the Q(V) loops held at their stage-1 output
       for this one solve.

    Returns the stage that gave the result; raises if stage 1 fails.  Only the
    failure path calls it, so every solve that converges stays bit-identical.
    """
    q_lims = bool(kwargs.pop("enforce_q_lims", True))
    for k in ("init", "run_control"):
        kwargs.pop(k, None)
    fails = (LoadflowNotConverged, ControllerNotConverged)
    # A stressed point needs more Newton iterations than the plant's normal budget
    # (50): the step-366 state converges with 100, not with 50 (2026-11-10).
    kwargs["max_iteration"] = max(int(kwargs.get("max_iteration", 50) or 50), 200)
    _restore_results(net, snap)
    if not q_lims:
        runpp_relays_held(net, init="results", enforce_q_lims=False, **kwargs)
        return "warm, no Q limits"
    # (a) directly with the Q limits.  With a machine just above its limit the
    # unconstrained Q(V) iteration may not settle while the constrained one does
    # (Case 1 SVR, step 366, 2026-11-10: fix v4).
    try:
        runpp_relays_held(net, init="results", enforce_q_lims=True, **kwargs)
        return "warm, Q limits, Q(V) loops active"
    except fails:
        _restore_results(net, snap)
    # (b) through the unconstrained solve (stages 1 and 2 of the docstring)
    try:
        runpp_relays_held(net, init="results", enforce_q_lims=False, **kwargs)
        snap1 = snapshot_results(net)
        try:
            runpp_relays_held(net, init="results", enforce_q_lims=True, **kwargs)
            return "warm via no Q limits, Q limits, Q(V) loops active"
        except fails:
            _restore_results(net, snap1)
            pp.runpp(net, init="results", enforce_q_lims=True, run_control=False, **kwargs)
            return "warm, Q limits, Q(V) loops at their stage-1 output"
    except fails:
        _restore_results(net, snap)
    # (c) the Q limits with the Q(V) loops held at the snapshot's DER outputs
    pp.runpp(net, init="results", enforce_q_lims=True, run_control=False, **kwargs)
    return "warm, Q limits, Q(V) loops held at the last converged output"


@dataclass(frozen=True)
class ActuatorWrites:
    """One dispatch's actuator setpoints, keyed by pandapower indices.

    Only the entries present are written; every field defaults to empty so
    a TSO-only or DSO-only dispatch constructs naturally.
    """

    #: sgen index -> OFO Q command [Mvar] (w-shift: written to
    #: ``net.sgen.q_set_mvar`` after reanchoring ``qv_vref_anchor_pu``).
    der_q_set_mvar: Dict[int, float] = field(default_factory=dict)
    #: gen index -> AVR voltage setpoint [pu] (``net.gen.vm_pu``).
    gen_v_pu: Dict[int, float] = field(default_factory=dict)
    #: trafo index -> tap position (machine / network 2W OLTC).
    tap_2w: Dict[int, int] = field(default_factory=dict)
    #: trafo3w index -> tap position (TS-STS coupler OLTC).
    tap_3w: Dict[int, int] = field(default_factory=dict)
    #: shunt index -> step (MSC/MSR; explicit shunt index, not bus, so a
    #: tertiary hosting both an MSC and an MSR stays unambiguous).
    shunt_step: Dict[int, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.der_q_set_mvar or self.gen_v_pu or self.tap_2w
                    or self.tap_3w or self.shunt_step)


@runtime_checkable
class Plant(Protocol):
    """Actuate / advance / measure -- the only surface the OFO loop sees."""

    def apply_u(self, writes: ActuatorWrites) -> None: ...

    def advance(self, duration_s: float) -> None: ...

    def read_y(self) -> pp.pandapowerNet: ...

    def apply_exogenous(self, profiles, t) -> None:
        """Push the time-series operating point for wall-clock ``t``.

        Exogenous means "not commanded by the OFO": profile-driven load P/Q
        and DER active power.  Routing this through the plant rather than
        writing ``net`` directly is what lets a non-static plant follow the
        profiles at all -- the RMS plant must translate them into simulation
        events, because PowerFactory reads element input attributes only at
        initialisation.
        """
        ...

    def apply_contingency(self, event, gen_trafo_map=None) -> None:
        """Deliver a scheduled contingency to the plant.

        The runner applies every event to ``net`` first (see
        ``experiments.helpers._apply_contingency``).  For the quasi-static
        plant ``net`` *is* the plant, so nothing further is needed.  A
        non-static plant holds its own state and must translate the event:
        PowerFactory reads element input attributes only at initialisation, so
        a topology change during an RMS calculation has to become a simulation
        event.

        Without this hook the runner would mutate the mirror and leave the
        simulator on the pre-contingency topology, so every measurement after
        the event would compare two different networks.
        """
        ...


def write_der_q_set(net: pp.pandapowerNet, sgen_idx: int,
                    q_mvar: float) -> None:
    """w-shift DER Q write: reanchor V_ref, then set ``q_set_mvar``.

    Byte-identical to the per-sgen block in
    ``experiments/helpers/plant_io.py`` (apply_zone_tso_controls /
    apply_dso_controls / apply_central_controls): defensive column
    creation, V_ref reanchored to the latest ``res_bus.vm_pu`` when one
    exists, and the OFO command written to ``q_set_mvar`` (the plant-side
    ``QVLocalLoop`` reads both columns on the next power flow).
    """
    if "q_set_mvar" not in net.sgen.columns:
        net.sgen["q_set_mvar"] = 0.0
    if "qv_vref_anchor_pu" not in net.sgen.columns:
        net.sgen["qv_vref_anchor_pu"] = float("nan")
    has_res_bus = (
        hasattr(net, "res_bus")
        and net.res_bus is not None
        and not net.res_bus.empty
        and "vm_pu" in net.res_bus.columns
    )
    if has_res_bus:
        bus = int(net.sgen.at[sgen_idx, "bus"])
        if bus in net.res_bus.index:
            net.sgen.at[sgen_idx, "qv_vref_anchor_pu"] = measured_der_voltage(
                net, sgen_idx, bus
            )
    net.sgen.at[sgen_idx, "q_set_mvar"] = float(q_mvar)


# ---------------------------------------------------------------------------
#  Control-vector adapters (u -> ActuatorWrites)
# ---------------------------------------------------------------------------
# These mirror the slicing of experiments/helpers/plant_io.py exactly; the
# configs are duck-typed (attribute access only) so core.plant does not
# import the controller stack.  Proven equivalent in
# tests/test_plant_static.py and end-to-end by
# tests/runner_refactor_regression.py.

def shunt_index_for_bus(net: pp.pandapowerNet, bus: int):
    """First ``net.shunt`` row at ``bus`` (plant_io convention), or None."""
    mask = net.shunt["bus"] == int(bus)
    return int(net.shunt.index[mask][0]) if mask.any() else None


def shunt_steps_for_buses(net: pp.pandapowerNet, buses) -> list:
    """Current step per shunt bus (0 when no shunt exists at the bus) --
    the ``prev_shunt_steps`` contract of ``apply_zone_tso_controls``."""
    out = []
    for sb in buses:
        idx = shunt_index_for_bus(net, int(sb))
        out.append(int(net.shunt.at[idx, "step"]) if idx is not None else 0)
    return out


def writes_from_zone_tso(net: pp.pandapowerNet, zone_def,
                         u) -> ActuatorWrites:
    """``u = [Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]`` per zone.

    The PCC block is skipped -- interface setpoints are messaged to the
    DSOs, never written to the plant (apply_zone_tso_controls parity).
    """
    n_der = len(zone_def.tso_der_indices)
    n_pcc = len(zone_def.pcc_trafo_indices)
    off = 0
    der = {int(s): float(u[off + k])
           for k, s in enumerate(zone_def.tso_der_indices)}
    off += n_der + n_pcc
    gen = {int(g): float(u[off + k])
           for k, g in enumerate(zone_def.gen_indices)}
    off += len(zone_def.gen_indices)
    t2w = {int(t): int(round(u[off + k]))
           for k, t in enumerate(zone_def.oltc_trafo_indices)}
    off += len(zone_def.oltc_trafo_indices)
    sh: Dict[int, int] = {}
    for k, sb in enumerate(zone_def.shunt_bus_indices):
        idx = shunt_index_for_bus(net, int(sb))
        if idx is not None:
            sh[idx] = int(round(u[off + k]))
    return ActuatorWrites(der_q_set_mvar=der, gen_v_pu=gen, tap_2w=t2w,
                          shunt_step=sh)


def writes_from_dso(dso_cfg, u) -> ActuatorWrites:
    """``u = [Q_DER | s_OLTC3w]``; shunt switching intentionally skipped
    (apply_dso_controls parity -- multi-zone shunts are TSO-owned)."""
    n_der = len(dso_cfg.der_indices)
    der = {int(s): float(u[k]) for k, s in enumerate(dso_cfg.der_indices)}
    t3w = {int(t): int(round(u[n_der + k]))
           for k, t in enumerate(dso_cfg.oltc_trafo_indices)}
    return ActuatorWrites(der_q_set_mvar=der, tap_3w=t3w)


def writes_from_central(net: pp.pandapowerNet, central_cfg,
                        u) -> ActuatorWrites:
    """``u = [Q_DER | Q_PCC(0) | V_gen | s_OLTC2w | s_shunt | s_OLTC3w]``
    (apply_central_controls parity)."""
    der_idx = list(central_cfg.der_indices)
    gen_idx = list(central_cfg.gen_indices)
    o2w = list(central_cfg.oltc_trafo_indices)
    sh_buses = list(central_cfg.shunt_bus_indices)
    o3w = list(getattr(central_cfg, "oltc_trafo3w_indices", []) or [])
    off = 0
    der = {int(s): float(u[off + k]) for k, s in enumerate(der_idx)}
    off += len(der_idx) + len(central_cfg.pcc_trafo_indices)
    gen = {int(g): float(u[off + k]) for k, g in enumerate(gen_idx)}
    off += len(gen_idx)
    t2w = {int(t): int(round(u[off + k])) for k, t in enumerate(o2w)}
    off += len(o2w)
    sh: Dict[int, int] = {}
    for k, sb in enumerate(sh_buses):
        idx = shunt_index_for_bus(net, int(sb))
        if idx is not None:
            sh[idx] = int(round(u[off + k]))
    off += len(sh_buses)
    t3w = {int(t): int(round(u[off + k])) for k, t in enumerate(o3w)}
    return ActuatorWrites(der_q_set_mvar=der, gen_v_pu=gen, tap_2w=t2w,
                          tap_3w=t3w, shunt_step=sh)


class PandapowerStaticPlant:
    """Quasi-static pandapower plant -- today's behaviour behind the
    :class:`Plant` interface.

    ``advance`` runs exactly the main-loop power flow of
    ``experiments/runners/multi_tso_dso.py`` (run_control=True so the
    plant-side Q(V) loops and any discrete tap controllers iterate,
    voltage angles, distributed slack, machine Q limits); the flags are
    constructor parameters so a config change stays a single edit.
    """

    def __init__(self, net: pp.pandapowerNet, *,
                 run_control: bool = True,
                 calculate_voltage_angles: bool = True,
                 max_iteration: int = 50,
                 max_iter: int = 300,
                 distributed_slack: bool = True,
                 enforce_q_lims: bool = True):
        self.net = net
        self.run_control = run_control
        self.calculate_voltage_angles = calculate_voltage_angles
        self.max_iteration = max_iteration
        self.max_iter = max_iter
        self.distributed_slack = distributed_slack
        self.enforce_q_lims = enforce_q_lims
        self.warm_resolves = 0          # diverged solves rescued by runpp_warm_staged

    # ── Plant protocol ───────────────────────────────────────────────────
    def apply_u(self, writes: ActuatorWrites) -> None:
        net = self.net
        for s_idx, q in writes.der_q_set_mvar.items():
            write_der_q_set(net, int(s_idx), float(q))
        for g_idx, v in writes.gen_v_pu.items():
            net.gen.at[int(g_idx), "vm_pu"] = float(v)
        for t_idx, tap in writes.tap_2w.items():
            net.trafo.at[int(t_idx), "tap_pos"] = int(round(tap))
        for t_idx, tap in writes.tap_3w.items():
            net.trafo3w.at[int(t_idx), "tap_pos"] = int(round(tap))
        for sh_idx, step in writes.shunt_step.items():
            if int(sh_idx) not in net.shunt.index:
                raise ValueError(
                    f"ActuatorWrites.shunt_step: index {sh_idx} not in "
                    f"net.shunt"
                )
            net.shunt.at[int(sh_idx), "step"] = int(round(step))

    def advance(self, duration_s: float = 0.0) -> None:
        """One converged power flow; ``duration_s`` is ignored (the
        quasi-static plant settles instantly within a dispatch interval).

        If the solve diverges, a warm staged re-solve from the state before it
        (:func:`runpp_warm_staged`, FIX 2026-11-10): pandapower starts every
        solve from a DC estimate ("auto" init), which fails on the stressed
        point after the 250 Mvar connection of Case 1 SVR although the plant
        has a solution next to the previous one.  A solve that converges is
        untouched; if the re-solve fails too, the original exception is raised.
        """
        snap = snapshot_results(self.net)
        kw = dict(calculate_voltage_angles=self.calculate_voltage_angles,
                  max_iteration=self.max_iteration, max_iter=self.max_iter,
                  distributed_slack=self.distributed_slack,
                  enforce_q_lims=self.enforce_q_lims)
        try:
            runpp_holding_taps(self.net, run_control=self.run_control, **kw)
        except LoadflowNotConverged as exc:
            if not self.run_control:
                raise
            try:
                how = runpp_warm_staged(self.net, snap, **kw)
            except (LoadflowNotConverged, ControllerNotConverged):
                _restore_results(self.net, snap)
                raise exc
            self.warm_resolves += 1
            print(f"  [plant] power flow diverged; warm re-solve converged ({how}); "
                  f"{self.warm_resolves} so far")

    def read_y(self) -> pp.pandapowerNet:
        return self.net

    def apply_exogenous(self, profiles, t) -> None:
        """Scale ``net`` loads/sgens to the profile row for ``t``."""
        from core.profiles import apply_profiles
        apply_profiles(self.net, profiles, t)

    def apply_contingency(self, event, gen_trafo_map=None) -> None:
        """No-op: ``net`` is the plant, and the runner has already mutated it."""
        return None
