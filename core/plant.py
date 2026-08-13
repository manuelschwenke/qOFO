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
            net.sgen.at[sgen_idx, "qv_vref_anchor_pu"] = float(
                net.res_bus.at[bus, "vm_pu"]
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
        quasi-static plant settles instantly within a dispatch interval)."""
        pp.runpp(
            self.net,
            run_control=self.run_control,
            calculate_voltage_angles=self.calculate_voltage_angles,
            max_iteration=self.max_iteration,
            max_iter=self.max_iter,
            distributed_slack=self.distributed_slack,
            enforce_q_lims=self.enforce_q_lims,
        )

    def read_y(self) -> pp.pandapowerNet:
        return self.net

    def apply_exogenous(self, profiles, t) -> None:
        """Scale ``net`` loads/sgens to the profile row for ``t``."""
        from core.profiles import apply_profiles
        apply_profiles(self.net, profiles, t)

    def apply_contingency(self, event, gen_trafo_map=None) -> None:
        """No-op: ``net`` is the plant, and the runner has already mutated it."""
        return None
