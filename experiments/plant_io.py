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

from typing import List, TYPE_CHECKING

import numpy as np
import pandapower as pp

from controller.base_controller import ControllerOutput
from controller.dso_controller import DSOControllerConfig
from controller.tso_controller import TSOControllerConfig

if TYPE_CHECKING:
    from controller.multi_tso_coordinator import ZoneDefinition
    from network.ieee39.hv_networks import HVNetworkInfo


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


def apply_qv_local_control(
    net: pp.pandapowerNet,
    hv_networks: List["HVNetworkInfo"],
    v_set: float,
    slope: float,
) -> None:
    """
    Apply a linear Q(V) droop to all DSO HV-connected DER.

    Reads bus voltage from the last converged power flow and sets the sgen
    Q from a Q(V) characteristic crossing zero at ``v_set``:

        frac = clamp((V - v_set) / slope, -1, 1)
        Q    = Q_min * frac   if frac > 0   (inductive, V too high)
             = Q_max * |frac| if frac < 0   (capacitive, V too low)

    Capability bounds follow VDE-AR-N 4120 v2 (+/- S_n for STATCOMs).
    """
    for hv in hv_networks:
        for s_idx in hv.sgen_indices:
            bus = int(net.sgen.at[s_idx, "bus"])
            v_pu = float(net.res_bus.at[bus, "vm_pu"])
            sn = float(net.sgen.at[s_idx, "sn_mva"])

            od = (net.sgen.at[s_idx, "op_diagram"]
                  if "op_diagram" in net.sgen.columns else None)
            if str(od) == "STATCOM":
                q_min, q_max = -sn, sn
            else:
                q_min, q_max = -0.33 * sn, 0.41 * sn

            dv = v_pu - v_set
            if slope > 0:
                frac = max(min(dv / slope, 1.0), -1.0)
            else:
                frac = 1.0 if dv > 0 else (-1.0 if dv < 0 else 0.0)

            q = q_min * frac if frac > 0 else q_max * (-frac)
            net.sgen.at[s_idx, "q_mvar"] = float(q)
