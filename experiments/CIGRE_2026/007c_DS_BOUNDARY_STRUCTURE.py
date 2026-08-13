#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007c_DS_BOUNDARY_STRUCTURE.py
====================================================
Does the DS-side boundary convention cost the DSO controller any
sensitivity it structurally cannot recover?

``build_dso_local_net`` pins every coupling-transformer primary bus with a
voltage source: the first is the angle reference (slack), the rest are PV.
Both are structural zeros in a power-flow Jacobian --

* at a slack bus the voltage is fixed by definition and the bus is
  eliminated from the reduced Jacobian entirely;
* at a PV bus ``d|V|/dQ = 0`` by construction.

So the DSO can have NO voltage sensitivity at its own primary-side
terminals, and any actuator whose branch touches the slack bus loses its
sensitivity column -- which is exactly the failure that forced
``promoted_slack_oltc_indices`` on the TSO side. The DSO's coupling
transformers ARE its OLTC actuators and they terminate on those buses, so
this is the place to look.

This script reports, per DSO:

1.  which primary buses are slack / PV in the reduced net;
2.  whether any of them is in ``voltage_bus_indices`` (i.e. whether the
    controller thinks it is monitoring a bus whose sensitivity is
    structurally zero);
3.  the column norms of the controller's own H matrix, split by actuator
    block, flagging any all-zero column;
4.  a direct probe of dV/dQ at each primary bus in the reduced net.

No modelling change is made here -- this is a diagnosis of the CURRENT
model.

Author: Manuel Schwenke / Claude Code
Date: 2026-08-11
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.runners.multi_tso_dso import run_multi_tso_dso  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_cigre005", str(Path(__file__).with_name("005_CIGRE_MULTI.py"))
)
_CIGRE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CIGRE)  # type: ignore[union-attr]


def _capture() -> Dict[str, Any]:
    cfg = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(cfg, k, v)
    cfg.verbose = 0
    cfg.run_stability_analysis = False
    got: Dict[str, Any] = {}

    def _hook(state):
        got.update(state)
        return True

    run_multi_tso_dso(cfg, pre_loop_hook=_hook)
    return got


def main() -> int:
    st = _capture()
    meta = st["meta"]
    hv_by_id = {hv.net_id: hv for hv in meta.hv_networks}

    print("=" * 78)
    print("DS LOCAL MODEL -- STRUCTURAL SENSITIVITY AT THE PRIMARY-SIDE BUSES")
    print("=" * 78)

    for dso_id, ctrl in sorted(st["dso_controllers"].items()):
        red = ctrl.sensitivities.net
        hv = hv_by_id.get(dso_id)
        primaries = [int(b) for b in getattr(hv, "coupling_ieee_buses", ())]
        cfg = ctrl.config

        slack_buses, pv_buses = [], []
        for g in red.gen.index:
            b = int(red.gen.at[g, "bus"])
            if b not in primaries:
                continue
            is_slack = "slack" in red.gen.columns and bool(red.gen.at[g, "slack"])
            (slack_buses if is_slack else pv_buses).append(b)

        print("\n" + "-" * 78)
        print(f"DSO {dso_id}:  primary buses {primaries}")
        print(f"  slack (V-theta) at {slack_buses};  PV at {pv_buses}")

        monitored = [int(b) for b in cfg.voltage_bus_indices]
        overlap = [b for b in primaries if b in monitored]
        print(f"  primary buses inside voltage_bus_indices: "
              f"{overlap if overlap else 'none'}")

        # ── direct probe: dV/dQ at each primary bus in the reduced net ──
        for b in primaries:
            try:
                s = ctrl.sensitivities.compute_dV_dQ_der(
                    der_bus_indices=[b], observation_bus_indices=[b],
                )
                val = np.asarray(s[0] if isinstance(s, tuple) else s,
                                 dtype=float).ravel()
                got = float(val[0]) if val.size else float("nan")
            except Exception as exc:  # structural absence shows up as a raise
                got = float("nan")
                print(f"    dV/dQ at primary {b}: unavailable ({type(exc).__name__})")
                continue
            print(f"    dV/dQ at primary {b}: {got:+.3e} pu/Mvar"
                  f"{'   <-- structurally zero' if abs(got) < 1e-12 else ''}")

        # ── H column norms by actuator block ──
        try:
            H = ctrl.build_sensitivity_matrix()
        except Exception:
            H = getattr(ctrl, "_H_cache", None)
        if H is None:
            print("  H unavailable")
            continue
        H = np.asarray(H, dtype=float)
        n_der = len(cfg.der_indices)
        n_oltc = len(getattr(cfg, "oltc_trafo_indices", []) or [])
        blocks = [("DER", 0, n_der), ("OLTC", n_der, n_der + n_oltc)]
        print(f"  H shape {H.shape}")
        for name, a, b_ in blocks:
            if b_ <= a or b_ > H.shape[1]:
                continue
            norms = np.linalg.norm(H[:, a:b_], axis=0)
            dead = int(np.sum(norms < 1e-14))
            print(f"    {name:>5} columns: {b_ - a:>2}  "
                  f"norms [{norms.min():.3e}, {norms.max():.3e}]  "
                  f"all-zero: {dead}"
                  f"{'   <-- LOST ACTUATOR(S)' if dead else ''}")
            if name == "OLTC":
                for k, t in enumerate(cfg.oltc_trafo_indices):
                    tag = ""
                    if t in red.trafo3w.index:
                        legs = [int(red.trafo3w.at[t, c])
                                for c in ("hv_bus", "mv_bus", "lv_bus")]
                    elif t in red.trafo.index:
                        legs = [int(red.trafo.at[t, c])
                                for c in ("hv_bus", "lv_bus")]
                    else:
                        legs = []
                    if set(legs) & set(slack_buses):
                        tag = "  [touches SLACK bus]"
                    elif set(legs) & set(pv_buses):
                        tag = "  [touches PV bus]"
                    print(f"      trafo {t:>3} legs {legs}  "
                          f"|col| = {norms[k]:.3e}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
