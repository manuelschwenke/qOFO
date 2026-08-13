#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/CIGRE_2026/007b_TIE_BOUNDARY_DIAG.py
================================================
Structural diagnostics behind 007_TIE_BOUNDARY_COMPARE.

Answers three questions the H-fidelity numbers raise:

1.  How many tie-line far-end stubs actually receive a constant-admittance
    equivalent under ``tie_boundary="z"``, and how many fall back to PQ
    because the equivalent is a net source?  (Zone 3's ``z`` result was
    bit-identical to ``pq``, which would mean a full fallback.)
2.  Which zones had to PROMOTE an in-zone machine to slack, and which
    OLTC column that cost them?  A zone holding the system slack already
    has a stiff in-area voltage anchor, so the boundary condition should
    matter less there.
3.  Sign and magnitude of each corridor flow at the linearisation point.

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


def _cfg(variant: str):
    c = _CIGRE.make_cigre_config()
    for k, v in _CIGRE.VARIANTS["V4"].items():
        setattr(c, k, v)
    c.tie_boundary_equivalent = variant
    c.verbose = 0
    c.run_stability_analysis = False
    return c


def _capture(cfg) -> Dict[str, Any]:
    got: Dict[str, Any] = {}

    def _hook(state):
        got.update(state)
        return True

    run_multi_tso_dso(cfg, pre_loop_hook=_hook)
    return got


def main() -> int:
    st = _capture(_cfg("z"))
    plant = st["net"]
    zone_defs = st["zone_defs"]

    plant_slack_gens = {
        int(g) for g in plant.gen.index
        if "slack" in plant.gen.columns and bool(plant.gen.at[g, "slack"])
    }
    print(f"plant slack gen(s): {sorted(plant_slack_gens)} "
          f"at bus(es) {[int(plant.gen.at[g, 'bus']) for g in sorted(plant_slack_gens)]}")

    for z in sorted(st["tso_controllers"].keys()):
        ctrl = st["tso_controllers"][z]
        zd = zone_defs[z]
        red = ctrl.sensitivities.net

        n_z = int((red.shunt["name"] == "WARD_TIE_Z").sum()) if not red.shunt.empty else 0
        n_pq = int((red.load["name"] == "WARD_TIE").sum()) if not red.load.empty else 0

        red_slack = [
            int(g) for g in red.gen.index
            if "slack" in red.gen.columns and bool(red.gen.at[g, "slack"])
        ]
        promoted = bool(red_slack) and not set(red_slack) & plant_slack_gens
        oos = np.asarray(getattr(ctrl, "_oos_oltc_mask", []), dtype=bool)

        print("\n" + "-" * 70)
        print(f"ZONE {z}")
        print(f"  tie stubs: {n_z} constant-Z, {n_pq} fell back to PQ")
        print(f"  reduced-net slack gen {red_slack} -> "
              f"{'PROMOTED (no system slack in zone)' if promoted else 'system slack is in-zone'}")
        print(f"  OLTC columns: {len(oos)} total, {int(oos.sum())} masked out")

        for li, b_in in zip(zd.tie_line_indices, zd.tie_line_endpoint_buses):
            li, b_in = int(li), int(b_in)
            if li not in plant.res_line.index:
                continue
            fb = int(plant.line.at[li, "from_bus"])
            far = int(plant.line.at[li, "to_bus"]) if b_in == fb else fb
            if far == fb:
                p, q = (float(plant.res_line.at[li, "p_from_mw"]),
                        float(plant.res_line.at[li, "q_from_mvar"]))
            else:
                p, q = (float(plant.res_line.at[li, "p_to_mw"]),
                        float(plant.res_line.at[li, "q_to_mvar"]))
            v = float(plant.res_bus.at[far, "vm_pu"])
            kind = "absorbs" if (-p >= 0 and -q >= 0) else "net source -> PQ fallback"
            print(f"    line {li:>3}: far bus {far:>3}  V={v:.4f}  "
                  f"injection into far bus P={p:+8.2f} MW  Q={q:+8.2f} Mvar   [{kind}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
