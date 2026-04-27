#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/plant_io.py
=======================
Functions that write controller outputs back into the pandapower network.

Two naming conventions live here:

* ``_apply_tso`` / ``_apply_dso`` -- single-TSO single-DSO cascade (used by
  ``001_S_TSO_S_DSO.py``).
* ``apply_zone_tso_controls`` / ``apply_dso_controls`` /
  ``apply_qv_local_control`` -- multi-zone IEEE 39-bus runner
  (``000_M_TSO_M_DSO.py``), per-zone ``ZoneDefinition``.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Sequence, TYPE_CHECKING

import numpy as np
import pandapower as pp
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net
from pandapower.control import CharacteristicControl
from pandapower.control.util.characteristic import Characteristic

from controller.base_controller import ControllerOutput
from controller.dso_controller import DSOControllerConfig
from controller.tso_controller import TSOControllerConfig

if TYPE_CHECKING:
    from controller.multi_tso_coordinator import ZoneDefinition
    from network.ieee39.hv_networks import HVNetworkInfo


class DampedCharacteristicControl(CharacteristicControl):
    """:class:`CharacteristicControl` with first-order damping on the output.

    Pandapower's stock controller writes ``Q_new = char(V)`` directly each
    iteration.  For Q(V) droops on large STATCOM-class units (Sn ~600 Mvar)
    with a tight slope (0.07 pu), the open-loop gain ``q_max / slope`` is so
    large that the un-damped controller oscillates between ±q_max and never
    converges within :func:`pp.control.run_control`'s default 30 iterations.

    This subclass replaces the write step with a damped update

        Q_{k+1} = Q_k + alpha * (target - Q_k),  0 < alpha <= 1

    while keeping the convergence test against the *un-damped* target so that
    the loop still terminates when the implicit equation ``Q = char(V(Q))`` is
    satisfied within ``tol``.

    Parameters
    ----------
    damping : float, default 0.5
        Damping factor alpha.  Lower = more damped (slower but more stable).
    """

    def __init__(self, net, *args, damping: float = 0.5, **kwargs):
        super().__init__(net, *args, **kwargs)
        self.damping = float(damping)

    def is_converged(self, net) -> bool:
        input_values = read_from_net(
            net, self.input_element, self.input_element_index,
            self.input_variable, self.read_flag,
        )
        target = net.characteristic.object.at[self.characteristic_index](input_values)
        output_values = read_from_net(
            net, self.output_element, self.output_element_index,
            self.output_variable, self.write_flag,
        )
        damped = output_values + self.damping * (target - output_values)
        write_to_net(
            net, self.output_element, self.output_element_index,
            self.output_variable, damped, self.write_flag,
        )
        # Convergence is judged on |target - current|, not on the damped step.
        return self.applied and np.all(np.abs(target - output_values) < self.tol)


def _apply_tso(net: pp.pandapowerNet, out: ControllerOutput, cfg: TSOControllerConfig):
    """Apply TSO controls to combined network. PCC Q setpoints are NOT applied here."""
    u = out.u_new
    n_der = len(cfg.der_indices)
    n_pcc = len(cfg.pcc_trafo_indices)
    n_gen = len(cfg.gen_indices)
    n_oltc = len(cfg.oltc_trafo_indices)

    # TS-DER Q
    for k, s in enumerate(cfg.der_indices):
        net.sgen.at[s, "q_mvar"] = float(u[k])

    # AVR setpoints
    off = n_der + n_pcc
    for k, g in enumerate(cfg.gen_indices):
        net.gen.at[g, "vm_pu"] = float(u[off + k])

    # 2W machine trafo OLTCs
    off += n_gen
    for k, t in enumerate(cfg.oltc_trafo_indices):
        net.trafo.at[t, "tap_pos"] = int(np.round(u[off + k]))

    # 380 kV shunts
    off += n_oltc
    for k, sb in enumerate(cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = int(np.round(u[off + k]))


def _apply_dso(net: pp.pandapowerNet, out: ControllerOutput, cfg: DSOControllerConfig):
    """Apply DSO controls to combined network."""
    u = out.u_new
    n_der = len(cfg.der_indices)
    n_oltc = len(cfg.oltc_trafo_indices)

    # DN DER Q
    for k, s in enumerate(cfg.der_indices):
        net.sgen.at[s, "q_mvar"] = float(u[k])

    # 3W coupler OLTCs
    for k, t in enumerate(cfg.oltc_trafo_indices):
        net.trafo3w.at[t, "tap_pos"] = int(np.round(u[n_der + k]))

    # Tertiary shunts
    for k, sb in enumerate(cfg.shunt_bus_indices):
        mask = net.shunt["bus"] == sb
        if mask.any():
            net.shunt.at[net.shunt.index[mask][0], "step"] = int(
                np.round(u[n_der + n_oltc + k])
            )


# ---------------------------------------------------------------------------
#  Multi-zone IEEE 39-bus apply helpers (used by experiments/000_M_TSO_M_DSO.py)
# ---------------------------------------------------------------------------


def apply_zone_tso_controls(
    net: pp.pandapowerNet,
    zone_def: "ZoneDefinition",
    tso_out,
) -> None:
    """
    Write TSO control output for one zone back to the pandapower plant network.

    Control variable ordering in ``u`` (must match ``TSOControllerConfig``):
        ``u = [Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]``

    PCC Q setpoints are *not* applied here; they are communicated to the DSO
    via ``TSOController.generate_setpoint_messages()``.
    """
    u = tso_out.u_new
    n_der = len(zone_def.tso_der_indices)
    n_pcc = len(zone_def.pcc_trafo_indices)
    n_gen = len(zone_def.gen_indices)
    off = 0

    for k, s_idx in enumerate(zone_def.tso_der_indices):
        net.sgen.at[s_idx, "q_mvar"] = float(u[off + k])
    off += n_der

    # PCC Q setpoints are forwarded to DSO controllers, not written to net.
    off += n_pcc

    for k, g_idx in enumerate(zone_def.gen_indices):
        net.gen.at[g_idx, "vm_pu"] = float(u[off + k])
    off += n_gen

    n_oltc = len(zone_def.oltc_trafo_indices)
    for k, t_idx in enumerate(zone_def.oltc_trafo_indices):
        net.trafo.at[t_idx, "tap_pos"] = int(round(u[off + k]))
    off += n_oltc
    # Shunt switching omitted for IEEE 39-bus (no TN shunts in base setup).


def apply_dso_controls(
    net: pp.pandapowerNet,
    dso_cfg: DSOControllerConfig,
    dso_out,
) -> None:
    """
    Write multi-zone DSO control output to the pandapower plant network.

    DSO ``u = [Q_DER | s_OLTC | s_shunt]``.  3-winding coupling trafo taps
    live in ``net.trafo3w``; shunt switching is intentionally skipped (shunts
    are initialised separately in the multi-zone runner).
    """
    u = dso_out.u_new
    n_der = len(dso_cfg.der_indices)
    n_oltc = len(dso_cfg.oltc_trafo_indices)
    off = 0

    for k, s_idx in enumerate(dso_cfg.der_indices):
        net.sgen.at[s_idx, "q_mvar"] = float(u[off + k])
    off += n_der

    for k, t_idx in enumerate(dso_cfg.oltc_trafo_indices):
        net.trafo3w.at[t_idx, "tap_pos"] = int(round(u[off + k]))
    off += n_oltc
    # Shunt switching skipped for multi-zone setup.


def _sgen_q_capability(net: pp.pandapowerNet, s_idx: int) -> tuple[float, float]:
    """Return (q_min, q_max) for sgen ``s_idx`` per its op_diagram label.

    STATCOMs have full ±S_n; everything else follows VDE-AR-N 4120 v2
    (-0.33 S_n to +0.41 S_n).
    """
    sn = float(net.sgen.at[s_idx, "sn_mva"])
    od = (net.sgen.at[s_idx, "op_diagram"]
          if "op_diagram" in net.sgen.columns else None)
    if str(od) == "STATCOM":
        return -sn, sn
    return -0.33 * sn, 0.41 * sn


def install_qv_characteristic_controllers(
    net: pp.pandapowerNet,
    sgen_indices: Sequence[int],
    v_set: float,
    slope: float,
    name_prefix: str = "qv",
    tol_mvar: float = 2.0,
    damping: float = 0.2,
) -> List[int]:
    """Install a pandapower :class:`CharacteristicControl` per sgen index.

    For each sgen, build a piecewise-linear Q(V) characteristic with
    breakpoints

        (0,             q_max),
        (v_set - slope, q_max),
        (v_set,           0.0),
        (v_set + slope, q_min),
        (2.0,           q_min),

    where ``(q_min, q_max)`` come from :func:`_sgen_q_capability`.  Note
    pandapower's ``Characteristic.__call__`` uses the breakpoints as a
    monotone interpolation in *x*; we extend the characteristic to
    ``x ∈ [0, 2]`` with constant Q outside the linear region so that
    voltages outside the droop band saturate Q rather than extrapolate.

    On every ``pp.runpp(net, run_control=True)`` the controller reads the
    sgen's bus voltage from ``net.res_bus.vm_pu`` and writes the new Q
    to ``net.sgen.q_mvar`` until convergence (within ``tol_mvar``).

    Implementation note: uses :class:`DampedCharacteristicControl` (not the
    stock :class:`CharacteristicControl`) because Q(V) on STATCOM-class
    windparks (Sn ~600 Mvar) with slope 0.07 pu has open-loop gain
    ~9 GVar/pu — the un-damped controller oscillates between rails and
    fails ``run_control``.  A damping factor of 0.5 (default) typically
    converges in 4-6 inner iterations.

    The function also stores the characteristic under
    ``net['characteristics'][f'{name_prefix}_{s_idx}']`` and (if the sgen
    has a name) ``net['sgen_characteristics'][sgen_name]`` for traceability,
    matching the ad-hoc convention used elsewhere in this codebase.

    Returns the list of registered controller indices.
    """
    controller_indices: List[int] = []

    if "characteristics" not in net:
        net["characteristics"] = {}
    if "sgen_characteristics" not in net:
        net["sgen_characteristics"] = {}
    if "qctrl" not in net.sgen.columns:
        net.sgen["qctrl"] = ""

    for s_idx in sgen_indices:
        bus = int(net.sgen.at[s_idx, "bus"])
        q_min, q_max = _sgen_q_capability(net, s_idx)

        v_pts = [0.0, v_set - slope, v_set, v_set + slope, 2.0]
        q_pts = [q_max, q_max, 0.0, q_min, q_min]

        char = Characteristic(net, x_values=v_pts, y_values=q_pts)

        char_key = f"{name_prefix}_{s_idx}"
        net["characteristics"][char_key] = {
            "type": "Q(V)", "x": list(v_pts), "y": list(q_pts),
        }
        sgen_name = str(net.sgen.at[s_idx, "name"])
        if sgen_name and sgen_name != "nan":
            net["sgen_characteristics"][sgen_name] = char

        net.sgen.at[s_idx, "qctrl"] = "Q(V)"

        cc = DampedCharacteristicControl(
            net,
            output_element="sgen",
            output_variable="q_mvar",
            output_element_index=int(s_idx),
            input_element="res_bus",
            input_variable="vm_pu",
            input_element_index=bus,
            characteristic_index=int(char.index),
            tol=tol_mvar,
            level=0,
            order=0,
            damping=damping,
        )
        controller_indices.append(int(cc.index))

    return controller_indices


def install_cos_phi_one(
    net: pp.pandapowerNet,
    sgen_indices: Sequence[int],
) -> None:
    """Force unity-power-factor on the listed sgens.

    Sets ``net.sgen.q_mvar = 0`` and tags ``net.sgen.qctrl = 'cphi_1'``.
    No controller is registered: Q stays at 0 across PFs unless overwritten
    by a profile or another control.
    """
    if "qctrl" not in net.sgen.columns:
        net.sgen["qctrl"] = ""
    for s_idx in sgen_indices:
        net.sgen.at[s_idx, "q_mvar"] = 0.0
        net.sgen.at[s_idx, "qctrl"] = "cphi_1"


def _hv_sgen_indices(hv_networks: List["HVNetworkInfo"]) -> List[int]:
    """Flatten ``[hv.sgen_indices for hv in hv_networks]`` into a single list."""
    return [int(s) for hv in hv_networks for s in hv.sgen_indices]


def apply_qv_local_control(
    net: pp.pandapowerNet,
    hv_networks: List["HVNetworkInfo"],
    v_set: float,
    slope: float,
) -> None:
    """
    Apply a linear Q(V) droop to all DSO HV-connected DER (back-compat wrapper).

    Replaces the previous manual iteration loop.  Each sgen receives a
    pandapower :class:`CharacteristicControl` (built by
    :func:`install_qv_characteristic_controllers`); the next
    ``pp.runpp(net, run_control=True)`` then iterates Q(V) automatically
    until convergence.

    Idempotent: re-calling this function does NOT re-install duplicate
    controllers — it only does so on the first call.  Subsequent calls
    are no-ops (the existing CharacteristicControl objects keep iterating
    on every PF).
    """
    sgen_indices = _hv_sgen_indices(hv_networks)
    if not sgen_indices:
        return
    if _qv_controllers_already_installed(net, sgen_indices):
        return
    install_qv_characteristic_controllers(
        net, sgen_indices, v_set=v_set, slope=slope, name_prefix="qv_dso",
    )


def apply_cos_phi_one_local_control(
    net: pp.pandapowerNet,
    hv_networks: List["HVNetworkInfo"],
) -> None:
    """Force HV-connected DER to operate at cos phi = 1 (Q = 0 Mvar).

    Back-compat wrapper around :func:`install_cos_phi_one`.  Overwrites
    any Q set by time-series profiles.
    """
    install_cos_phi_one(net, _hv_sgen_indices(hv_networks))


def _qv_controllers_already_installed(
    net: pp.pandapowerNet, sgen_indices: Sequence[int],
) -> bool:
    """True if every sgen in ``sgen_indices`` already has a CharacteristicControl
    (or :class:`DampedCharacteristicControl`) writing to its q_mvar.  Used to
    make installation idempotent."""
    if not hasattr(net, "controller") or len(net.controller) == 0:
        return False
    sgen_set = set(int(s) for s in sgen_indices)
    seen: set = set()
    for _, row in net.controller.iterrows():
        obj = row["object"]
        if not isinstance(obj, CharacteristicControl):
            continue  # DampedCharacteristicControl is a subclass; isinstance covers both.
        if (getattr(obj, "output_element", None) == "sgen"
                and getattr(obj, "output_variable", None) == "q_mvar"):
            for idx in np.atleast_1d(getattr(obj, "output_element_index", [])):
                seen.add(int(idx))
    return sgen_set.issubset(seen)
