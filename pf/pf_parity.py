"""
pf/pf_parity.py
===============
Gate A/B/C parity check: run the PowerFactory load flow in the parity study
case and compare every bus voltage/angle and every branch flow against the
``solution`` section of a dynamic snapshot (the pandapower oracle).

ComLdf option set (applied on every run)
----------------------------------------
AC balanced, load voltage dependency ON (the anchored-ZIP convention shared
by both models), no automatic taps/shunts, no reactive/active power limits
(no machine is near a Q limit in any reference snapshot; the oracle's
interior solution is then reproduced exactly).

Angle convention: both models anchor 0 deg at the slack machine's terminal
(pandapower ``slack=True`` gen; PF reference machine G 01 set by pf_sync).
The comparison nevertheless re-aligns angles on that bus defensively.

Usage (PF machine)::

    python pf\\pf_parity.py export\\snapshots\\base_t0_20160105-0800.json
    python pf\\pf_parity.py <snapshot> --tol-vm 1e-4 --tol-va 0.01

Exit code 0 iff the vm/va gate passes.

Author: Manuel Schwenke / Claude Code (2026-07-17)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.dynamic_snapshot import load_snapshot  # noqa: E402
from pf.naming import build_name_map  # noqa: E402
from pf.session import (  # noqa: E402
    DEFAULT_PROJECT_PATH,
    PFSessionError,
    connect,
    deactivate_variations_except,
    get_all,
    run_ldf,
    set_variation_active,
)

PARITY_STUDY_CASE = "01_LDF_Parity"

#: Snapshot phase -> ordered variation stack to activate.
_PHASE_VARIATIONS = {
    "base": (),
    "wind_replace": ("wind_replace",),
    "full": ("wind_replace", "full"),
}

#: The authoritative ComLdf parity option set (docs/pf_api_notes.md §4).
PARITY_LDF_SETTINGS = {
    "iopt_net": 0,     # AC balanced
    "iopt_pq": 1,      # load voltage dependency ON (anchored ZIP)
    "iopt_at": 0,      # no automatic transformer taps
    "iopt_asht": 0,    # no automatic shunt adjustment
    "iopt_lim": 0,     # no reactive power limits (interior oracle solution)
    "iopt_plim": 0,    # no active power limits
}


@dataclass
class Deviation:
    quantity: str      # e.g. 'bus.vm_pu'
    element: str       # loc_name
    stored: float
    pf: float

    @property
    def dev(self) -> float:
        return abs(self.pf - self.stored)


def _collect(app, doc) -> Tuple[List[Deviation], Dict[str, float]]:
    names = build_name_map(doc)
    model = doc["model"]
    solution = doc["solution"]

    by_name: Dict[str, Any] = {}
    for cls in ("ElmTerm", "ElmLne", "ElmTr2", "ElmTr3", "ElmLod",
                "ElmSym", "ElmGenstat", "ElmShnt"):
        for o in get_all(app, cls):
            by_name[o.loc_name] = o

    devs: List[Deviation] = []

    # ── Angle reference: the slack machine's terminal on both sides ──────
    slack_keys = [k for k, r in model["gen"].items() if r["slack"]]
    if len(slack_keys) != 1:
        raise PFSessionError(f"Expected one slack gen, found {slack_keys}")
    slack_bus_idx = int(model["gen"][slack_keys[0]]["bus"])
    slack_bus_name = names[("bus", slack_bus_idx)]
    va_ref_snap = float(solution["bus"][str(slack_bus_idx)]["va_degree"])
    va_ref_pf = float(by_name[slack_bus_name].GetAttribute("m:phiu"))

    # ── Buses: vm / aligned va ───────────────────────────────────────────
    for key, rec in solution["bus"].items():
        name = names[("bus", int(key))]
        term = by_name.get(name)
        if term is None:
            raise PFSessionError(f"ElmTerm {name!r} missing in PF model")
        vm_pf = float(term.GetAttribute("m:u"))
        va_pf = float(term.GetAttribute("m:phiu")) - va_ref_pf
        va_sn = float(rec["va_degree"]) - va_ref_snap
        devs.append(Deviation("bus.vm_pu", name, float(rec["vm_pu"]), vm_pf))
        devs.append(Deviation("bus.va_deg", name, va_sn, va_pf))

    # ── Branch flows ─────────────────────────────────────────────────────
    flow_map = (
        ("line", ("p_from_mw", "m:P:bus1"), ("q_from_mvar", "m:Q:bus1"),
         ("p_to_mw", "m:P:bus2"), ("q_to_mvar", "m:Q:bus2")),
        ("trafo", ("p_hv_mw", "m:P:bushv"), ("q_hv_mvar", "m:Q:bushv"),
         ("p_lv_mw", "m:P:buslv"), ("q_lv_mvar", "m:Q:buslv")),
        ("trafo3w", ("p_hv_mw", "m:P:bushv"),
         ("q_hv_mvar", "m:Q:bushv"),
         ("p_mv_mw", "m:P:busmv"), ("q_mv_mvar", "m:Q:busmv"),
         ("p_lv_mw", "m:P:buslv"), ("q_lv_mvar", "m:Q:buslv")),
    )
    for table, *pairs in flow_map:
        for key, rec in solution.get(table, {}).items():
            name = names[(table, int(key))]
            obj = by_name.get(name)
            if obj is None:
                raise PFSessionError(f"{table} {name!r} missing in PF model")
            for snap_field, pf_attr in pairs:
                devs.append(Deviation(
                    f"{table}.{snap_field}", name,
                    float(rec[snap_field]),
                    float(obj.GetAttribute(pf_attr)),
                ))

    # ── Loads (served ZIP power) ─────────────────────────────────────────
    for key, rec in solution["load"].items():
        name = names[("load", int(key))]
        obj = by_name.get(name)
        if obj is None:
            raise PFSessionError(f"ElmLod {name!r} missing in PF model")
        devs.append(Deviation("load.p_mw", name, float(rec["p_mw"]),
                              float(obj.GetAttribute("m:P:bus1"))))
        devs.append(Deviation("load.q_mvar", name, float(rec["q_mvar"]),
                              float(obj.GetAttribute("m:Q:bus1"))))

    # ── Static generators / wind parks (P and converged Q) ───────────────
    for key, rec in solution.get("sgen", {}).items():
        name = names[("sgen", int(key))]
        obj = by_name.get(name)
        if obj is None:
            raise PFSessionError(f"ElmGenstat {name!r} missing in PF model")
        devs.append(Deviation("sgen.p_mw", name, float(rec["p_mw"]),
                              float(obj.GetAttribute("m:P:bus1"))))
        devs.append(Deviation("sgen.q_mvar", name, float(rec["q_mvar"]),
                              float(obj.GetAttribute("m:Q:bus1"))))

    # ── Machines (plant totals; PF results are per element = whole
    #    parallel group, so no ngnum scaling is applied here -- verified
    #    against G 05 with ngnum = 2) ──────────────────────────────────────
    # Shunts use the same load-reference sign as pandapower: capacitive
    # injection is negative Q, reactor consumption is positive Q.
    for key, rec in solution.get("shunt", {}).items():
        name = names[("shunt", int(key))]
        obj = by_name.get(name)
        if obj is None:
            raise PFSessionError(f"ElmShnt {name!r} missing in PF model")
        devs.append(Deviation("shunt.p_mw", name, float(rec["p_mw"]),
                              float(obj.GetAttribute("m:P:bus1"))))
        devs.append(Deviation("shunt.q_mvar", name, float(rec["q_mvar"]),
                              float(obj.GetAttribute("m:Q:bus1"))))

    from pf.naming import machine_template_name
    for key, rec in model["gen"].items():
        tpl = machine_template_name(rec)
        mach = by_name.get(tpl)
        if mach is None:
            raise PFSessionError(f"ElmSym {tpl!r} missing")
        sol = solution["gen"][key]
        devs.append(Deviation("gen.p_mw", tpl, float(sol["p_mw"]),
                              float(mach.GetAttribute("m:P:bus1"))))
        devs.append(Deviation("gen.q_mvar", tpl, float(sol["q_mvar"]),
                              float(mach.GetAttribute("m:Q:bus1"))))

    max_dev = {}
    for d in devs:
        family = d.quantity.split(".")[0] + "." + d.quantity.split(".")[1]
        max_dev[family] = max(max_dev.get(family, 0.0), d.dev)
    return devs, max_dev


def _print_interfaces(devs: List[Deviation]) -> None:
    """Print HV/MV P/Q parity for all 3W coupling interfaces."""
    fields = ("p_hv_mw", "q_hv_mvar", "p_mv_mw", "q_mv_mvar")
    selected = [
        d for d in devs
        if d.quantity.startswith("trafo3w.")
        and d.quantity.split(".", 1)[1] in fields
    ]
    elements = sorted({d.element for d in selected})
    print("\nInterface winding-flow parity (HV/MV):")
    print(f"  {'element':24s} {'quantity':13s} {'snapshot':>14s} "
          f"{'powerfactory':>14s} {'|dev|':>10s}")
    for element in elements:
        indexed = {
            d.quantity.split(".", 1)[1]: d
            for d in selected if d.element == element
        }
        missing = [field for field in fields if field not in indexed]
        if missing:
            raise PFSessionError(
                f"Interface {element!r} missing parity fields {missing}"
            )
        for field in fields:
            d = indexed[field]
            print(f"  {element:24s} {field:13s} {d.stored:14.6f} "
                  f"{d.pf:14.6f} {d.dev:10.3e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PowerFactory-vs-snapshot load-flow parity check.")
    parser.add_argument("snapshot")
    parser.add_argument("--tol-vm", type=float, default=1e-4)
    parser.add_argument("--tol-va", type=float, default=0.01)
    parser.add_argument("--tol-mw", type=float, default=1.0,
                        help="Flow/injection gate tolerance [MW/Mvar]")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--study-case", default=PARITY_STUDY_CASE)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--interfaces", action="store_true",
        help="Print all 12 coupler HV/MV winding P/Q comparisons",
    )
    args = parser.parse_args(argv)

    doc = load_snapshot(args.snapshot)
    phase = doc["provenance"].get("phase")
    if phase not in _PHASE_VARIATIONS:
        raise SystemExit(f"Unsupported snapshot phase {phase!r}")

    app = connect(args.project, study_case=args.study_case)

    # Recreate the exact ordered state: base -> wind_replace -> full.
    deactivate_variations_except(app, keep=None)
    for variation in _PHASE_VARIATIONS[phase]:
        set_variation_active(app, variation, True)

    run_ldf(app, PARITY_LDF_SETTINGS)

    devs, max_dev = _collect(app, doc)
    devs.sort(key=lambda d: d.dev, reverse=True)

    label = doc["provenance"].get("label", args.snapshot)
    print("=" * 72)
    print(f"Parity report: {label}")
    print("=" * 72)
    for family in sorted(max_dev):
        print(f"  max |d {family}| = {max_dev[family]:.3e}")
    print(f"\nWorst {args.top} deviations:")
    print(f"  {'quantity':18s} {'element':24s} {'snapshot':>14s} "
          f"{'powerfactory':>14s} {'|dev|':>10s}")
    for d in devs[:args.top]:
        print(f"  {d.quantity:18s} {d.element:24s} {d.stored:14.6f} "
              f"{d.pf:14.6f} {d.dev:10.3e}")

    if args.interfaces:
        _print_interfaces(devs)

    vm_ok = max_dev.get("bus.vm_pu", 0.0) <= args.tol_vm
    va_ok = max_dev.get("bus.va_deg", 0.0) <= args.tol_va
    flow_families = [
        family for family in max_dev
        if family.startswith((
            "line.", "trafo.", "trafo3w.", "gen.", "load.", "sgen.",
            "shunt.",
        ))
    ]
    flows_ok = all(max_dev[f] <= args.tol_mw for f in flow_families)

    print("\nGate verdict:")
    print(f"  vm  <= {args.tol_vm:g} pu : {'PASS' if vm_ok else 'FAIL'} "
          f"({max_dev.get('bus.vm_pu', 0.0):.3e})")
    print(f"  va  <= {args.tol_va:g} deg: {'PASS' if va_ok else 'FAIL'} "
          f"({max_dev.get('bus.va_deg', 0.0):.3e})")
    print(f"  flows <= {args.tol_mw:g} MW/Mvar: "
          f"{'PASS' if flows_ok else 'FAIL'}")
    return 0 if (vm_ok and va_ok and flows_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
