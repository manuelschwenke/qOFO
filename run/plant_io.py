#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run/plant_io.py
===============
Functions that write controller outputs back into the pandapower network.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandapower as pp

from controller.base_controller import ControllerOutput
from controller.dso_controller import DSOControllerConfig
from controller.tso_controller import TSOControllerConfig


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
