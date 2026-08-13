"""Targeted diagnostics for the Phase-3 ``wind_replace`` parity gate.

The ordinary parity report shows terminal powers, which makes a shifted
operating point look like many unrelated branch-flow errors.  This helper
instead compares branch losses (the sum of both terminal powers) and system
power balances.  It is intentionally read-only with respect to network
objects; only the requested variation state and the load-flow calculation
state are changed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.dynamic_snapshot import load_snapshot  # noqa: E402
from pf.naming import build_name_map  # noqa: E402
from pf.pf_parity import PARITY_LDF_SETTINGS  # noqa: E402
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
WIND_REPLACE_VARIATION = "wind_replace"


@dataclass(frozen=True)
class LossComparison:
    table: str
    element: str
    p_snapshot: float
    p_powerfactory: float
    q_snapshot: float
    q_powerfactory: float

    @property
    def p_delta(self) -> float:
        return self.p_powerfactory - self.p_snapshot

    @property
    def q_delta(self) -> float:
        return self.q_powerfactory - self.q_snapshot


def _objects_by_name(app, class_names: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for class_name in class_names:
        for obj in get_all(app, class_name):
            if obj.loc_name in result:
                raise PFSessionError(
                    f"Duplicate calculation-relevant loc_name {obj.loc_name!r}"
                )
            result[obj.loc_name] = obj
    return result


def _branch_losses(app, doc: Mapping[str, Any]) -> list[LossComparison]:
    names = build_name_map(doc)
    solution = doc["solution"]
    objects = _objects_by_name(app, ("ElmLne", "ElmTr2"))
    result: list[LossComparison] = []

    specifications = (
        (
            "line",
            "p_from_mw",
            "p_to_mw",
            "q_from_mvar",
            "q_to_mvar",
            "m:P:bus1",
            "m:P:bus2",
            "m:Q:bus1",
            "m:Q:bus2",
        ),
        (
            "trafo",
            "p_hv_mw",
            "p_lv_mw",
            "q_hv_mvar",
            "q_lv_mvar",
            "m:P:bushv",
            "m:P:buslv",
            "m:Q:bushv",
            "m:Q:buslv",
        ),
    )
    for (table, p1_field, p2_field, q1_field, q2_field,
         p1_attr, p2_attr, q1_attr, q2_attr) in specifications:
        for key, record in solution[table].items():
            name = names[(table, int(key))]
            obj = objects.get(name)
            if obj is None:
                raise PFSessionError(f"{table} {name!r} missing in PF model")
            result.append(
                LossComparison(
                    table=table,
                    element=name,
                    p_snapshot=float(record[p1_field]) + float(record[p2_field]),
                    p_powerfactory=(
                        float(obj.GetAttribute(p1_attr))
                        + float(obj.GetAttribute(p2_attr))
                    ),
                    q_snapshot=float(record[q1_field]) + float(record[q2_field]),
                    q_powerfactory=(
                        float(obj.GetAttribute(q1_attr))
                        + float(obj.GetAttribute(q2_attr))
                    ),
                )
            )
    return result


def _sum_snapshot_solution(
    table: Mapping[str, Mapping[str, Any]], field: str
) -> float:
    return sum(float(record[field]) for record in table.values())


def _print_balance(app, doc: Mapping[str, Any]) -> None:
    solution = doc["solution"]
    pf_objects = {
        class_name: get_all(app, class_name)
        for class_name in ("ElmSym", "ElmGenstat", "ElmLod")
    }

    snapshot_generation = (
        _sum_snapshot_solution(solution["gen"], "p_mw")
        + _sum_snapshot_solution(solution.get("sgen", {}), "p_mw")
    )
    snapshot_load = _sum_snapshot_solution(solution["load"], "p_mw")

    pf_generation = sum(
        float(obj.GetAttribute("m:P:bus1"))
        for class_name in ("ElmSym", "ElmGenstat")
        for obj in pf_objects[class_name]
        if not int(obj.GetAttribute("outserv"))
    )
    pf_load = sum(
        float(obj.GetAttribute("m:P:bus1"))
        for obj in pf_objects["ElmLod"]
        if not int(obj.GetAttribute("outserv"))
    )

    print("System active-power balance [MW]")
    print("  source            generation          load       gen-load")
    print(
        f"  snapshot      {snapshot_generation:14.6f}"
        f" {snapshot_load:13.6f} {snapshot_generation - snapshot_load:14.6f}"
    )
    print(
        f"  PowerFactory  {pf_generation:14.6f}"
        f" {pf_load:13.6f} {pf_generation - pf_load:14.6f}"
    )
    print(
        f"  PF - snapshot {pf_generation - snapshot_generation:14.6f}"
        f" {pf_load - snapshot_load:13.6f}"
        f" {(pf_generation - pf_load) - (snapshot_generation - snapshot_load):14.6f}"
    )


def _print_losses(rows: Sequence[LossComparison], top: int) -> None:
    print("\nBranch-loss totals [MW / Mvar]")
    print("  family       P snapshot          P PF        dP"
          "    Q snapshot          Q PF        dQ")
    for family in ("line", "trafo", "all"):
        selected = rows if family == "all" else [r for r in rows if r.table == family]
        p_snapshot = sum(r.p_snapshot for r in selected)
        p_pf = sum(r.p_powerfactory for r in selected)
        q_snapshot = sum(r.q_snapshot for r in selected)
        q_pf = sum(r.q_powerfactory for r in selected)
        print(
            f"  {family:7s} {p_snapshot:14.6f} {p_pf:13.6f}"
            f" {p_pf - p_snapshot:9.6f} {q_snapshot:13.6f}"
            f" {q_pf:13.6f} {q_pf - q_snapshot:9.6f}"
        )

    ordered = sorted(rows, key=lambda row: abs(row.p_delta), reverse=True)
    print(f"\nLargest {min(top, len(ordered))} active-loss deviations [MW]")
    print(
        f"  {'family':7s} {'element':24s} {'snapshot':>12s}"
        f" {'PowerFactory':>13s} {'PF-snapshot':>13s}"
        f" {'dQ [Mvar]':>12s}"
    )
    for row in ordered[:top]:
        print(
            f"  {row.table:7s} {row.element:24s} {row.p_snapshot:12.6f}"
            f" {row.p_powerfactory:13.6f} {row.p_delta:13.6f}"
            f" {row.q_delta:12.6f}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose wind_replace branch-loss parity."
    )
    parser.add_argument("snapshot")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--study-case", default=PARITY_STUDY_CASE)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    doc = load_snapshot(args.snapshot)
    if doc["provenance"].get("phase") != "wind_replace":
        raise SystemExit("Gate-B diagnostics require a wind_replace snapshot")
    if args.top <= 0:
        raise SystemExit("--top must be positive")

    app = connect(args.project, study_case=args.study_case)
    deactivate_variations_except(app, keep=WIND_REPLACE_VARIATION)
    set_variation_active(app, WIND_REPLACE_VARIATION, True)
    run_ldf(app, PARITY_LDF_SETTINGS)

    print(f"Gate-B loss diagnostic: {doc['provenance'].get('label')}")
    _print_balance(app, doc)
    _print_losses(_branch_losses(app, doc), args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
