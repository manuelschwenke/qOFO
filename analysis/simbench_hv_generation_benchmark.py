"""
Generation benchmark for the synthetic IEEE39-underlaid 110 kV DS models.

This complements :mod:`analysis.simbench_hv_benchmark`.  It parses installed
generation from all three SimBench expansion scenarios for the two HV grid
families. Future scenarios add generator-interconnection assets, not only
capacity. The script compares scenario 0 (the case used by the topology
benchmark) with each fully built synthetic DSO.

Two SimBench scopes are kept separate:

* ``direct_110kV``: plants represented as connected directly to the HV grid;
* ``equivalent_below_110kV``: aggregated generation in equivalent downstream
  MV grids, represented by an injection at a 110 kV station.

This distinction is essential for comparison with qOFO: every synthetic DSO
generator is a physical ``sgen`` on a 110 kV bus, although some or all of
those plants could be *interpreted* as aggregate DER equivalents.

Outputs
-------
``generation_units.csv``
    One row per non-zero generator element.
``generation_by_scope_carrier.csv``
    Capacity and unit counts grouped by grid, scenario, connection scope and
    carrier.
``generation_summary.csv``
    Total/direct/equivalent capacity, reference demand and topology-normalised
    generation densities.
``scenario0_comparison.csv``
    Synthetic-to-SimBench ratios for the scenario-0 reference.
``report.md`` and ``simbench_hv_generation_benchmark.png``
    Human-readable assessment and compact visual summary.

Run
---
``python -m analysis.simbench_hv_generation_benchmark``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from analysis.simbench_hv_benchmark import (
    GridTopology,
    _hv_buses,
    _hv_subnetwork_ids,
    _md_table,
    analyse_grid,
    extract_synthetic_from_built_net,
)


SIMBENCH_GENERATION_CODES: Dict[str, Tuple[str, int, str]] = {
    "SimBench HV1 (mixed), scenario 0": (
        "HV1", 0, "1-HV-mixed--0-sw"
    ),
    "SimBench HV1 (mixed), scenario 1": (
        "HV1", 1, "1-HV-mixed--1-sw"
    ),
    "SimBench HV1 (mixed), scenario 2": (
        "HV1", 2, "1-HV-mixed--2-sw"
    ),
    "SimBench HV2 (urban), scenario 0": (
        "HV2", 0, "1-HV-urban--0-sw"
    ),
    "SimBench HV2 (urban), scenario 1": (
        "HV2", 1, "1-HV-urban--1-sw"
    ),
    "SimBench HV2 (urban), scenario 2": (
        "HV2", 2, "1-HV-urban--2-sw"
    ),
}

_P_TOL_MW = 1e-9


def _case_buses(
    net: pp.pandapowerNet,
    *,
    group: Optional[str],
    include_auxiliary: bool,
) -> set:
    """Return 110 kV buses belonging to one parsed case."""
    if group is None:
        return set(int(b) for b in _hv_buses(net))

    mask = (net.bus.vn_kv.astype(float) - 110.0).abs() < 1.0
    mask &= net.bus.name.astype(str).str.startswith(f"{group}|")
    if not include_auxiliary:
        mask &= ~net.bus.name.astype(str).str.contains("AUX_LOAD", na=False)
    return set(int(b) for b in net.bus.index[mask])


def _carrier(element: str, row: pd.Series, scope: str) -> str:
    """Map heterogeneous pandapower/SimBench labels to a small carrier set."""
    if scope == "equivalent_below_110kV":
        return "aggregated_RES_unspecified"
    if element == "storage":
        return "storage"

    labels = " ".join(
        str(row.get(col, "")) for col in ("type", "phys_type", "profile", "name")
    ).lower()
    if "wind" in labels or str(row.get("type", "")).upper() == "WP":
        return "wind"
    if "pv" in labels or "solar" in labels:
        return "solar_pv"
    if "hydro" in labels:
        return "hydro"
    if "biomass" in labels:
        return "biomass"
    return "other"


def parse_generation(
    net: pp.pandapowerNet,
    *,
    label: str,
    topology: str,
    scenario: str,
    group: Optional[str] = None,
    simbench_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Parse all non-zero generation represented on the selected 110 kV grid.

    ``p_mw`` is used as installed active power because this is the capacity
    field populated by SimBench and its sum reproduces published SimBench
    installed-generation totals. ``sn_mva`` is retained separately.
    ``ext_grid`` is deliberately excluded: it is the upstream system, not
    generation installed inside the studied HV grid. ``storage`` is also
    excluded because its signed ``p_mw`` is a dispatch point (negative while
    charging in the SimBench future scenarios), not installed generation.
    """
    buses = _case_buses(net, group=group, include_auxiliary=False)
    rows: List[dict] = []
    zero_excluded = 0

    for element in ("gen", "sgen"):
        table = getattr(net, element, None)
        if table is None or not len(table) or "bus" not in table.columns:
            continue
        selected = table.loc[table.bus.isin(buses)].copy()
        if "in_service" in selected.columns:
            selected = selected.loc[selected.in_service.astype(bool)]

        for idx, row in selected.iterrows():
            p_mw = float(pd.to_numeric(row.get("p_mw", 0.0), errors="coerce"))
            sn_raw = pd.to_numeric(row.get("sn_mva", np.nan), errors="coerce")
            sn_mva = float(sn_raw) if pd.notna(sn_raw) else np.nan
            if abs(p_mw) <= _P_TOL_MW and (
                not np.isfinite(sn_mva) or abs(sn_mva) <= _P_TOL_MW
            ):
                zero_excluded += 1
                continue

            subnet = str(row.get("subnet", ""))
            scope = (
                "equivalent_below_110kV"
                if simbench_root is not None and subnet != simbench_root
                else "direct_110kV"
            )
            rows.append({
                "grid": label,
                "topology": topology,
                "scenario": scenario,
                "element": element,
                "index": int(idx),
                "name": str(row.get("name", "")),
                "bus": int(row.bus),
                "bus_name": str(net.bus.at[int(row.bus), "name"]),
                "scope": scope,
                "carrier": _carrier(element, row, scope),
                "type_raw": str(row.get("type", "")),
                "profile": str(row.get("profile", "")),
                "subnet": subnet,
                "p_inst_mw": p_mw,
                "sn_mva": sn_mva,
            })

    columns = [
        "grid", "topology", "scenario", "element", "index", "name", "bus",
        "bus_name", "scope", "carrier", "type_raw", "profile", "subnet",
        "p_inst_mw", "sn_mva",
    ]
    return pd.DataFrame(rows, columns=columns), zero_excluded


def reference_load_p_mw(
    net: pp.pandapowerNet,
    *,
    group: Optional[str] = None,
) -> float:
    """
    Sum the represented active load on the selected HV system.

    Synthetic auxiliary ZIP-load buses are included because they remain part
    of the same DSO electrically, despite being excluded from topology counts.
    """
    buses = _case_buses(net, group=group, include_auxiliary=True)
    if not len(net.load):
        return 0.0
    load = net.load.loc[net.load.bus.isin(buses)].copy()
    if "in_service" in load.columns:
        load = load.loc[load.in_service.astype(bool)]
    return float(pd.to_numeric(load.p_mw, errors="coerce").fillna(0.0).sum())


def summarize_case(
    *,
    units: pd.DataFrame,
    topology: GridTopology,
    topology_name: str,
    scenario: str,
    load_p_mw: float,
    zero_excluded: int,
) -> dict:
    """Aggregate one case without losing the direct/equivalent distinction."""
    direct = units.loc[units.scope == "direct_110kV"]
    equiv = units.loc[units.scope == "equivalent_below_110kV"]
    route_km = float(topology.lines.length_km.sum())
    p_total = float(units.p_inst_mw.sum())

    def p_carrier(frame: pd.DataFrame, carrier: str) -> float:
        return float(frame.loc[frame.carrier == carrier, "p_inst_mw"].sum())

    return {
        "grid": topology.label,
        "topology": topology_name,
        "scenario": scenario,
        "n_units": len(units),
        "n_zero_placeholders_excluded": zero_excluded,
        "p_inst_total_mw": p_total,
        "sn_total_mva": float(units.sn_mva.fillna(0.0).sum()),
        "p_direct_110kv_mw": float(direct.p_inst_mw.sum()),
        "p_equiv_below_110kv_mw": float(equiv.p_inst_mw.sum()),
        "p_wind_direct_mw": p_carrier(direct, "wind"),
        "p_pv_direct_mw": p_carrier(direct, "solar_pv"),
        "p_other_direct_mw": float(
            direct.loc[
                ~direct.carrier.isin(["wind", "solar_pv"]), "p_inst_mw"
            ].sum()
        ),
        "load_p_reference_mw": load_p_mw,
        "generation_to_load_ratio": (
            p_total / load_p_mw if load_p_mw > _P_TOL_MW else np.nan
        ),
        "stations_reduced": topology.n_stations,
        "route_km": route_km,
        "mw_per_station": p_total / max(topology.n_stations, 1),
        "mw_per_route_km": p_total / route_km if route_km > 0 else np.nan,
    }


def build_scenario0_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Synthetic/reference ratios for the two scenario-0 SimBench grids."""
    syn = summary.loc[summary.scenario == "synthetic"]
    ref = summary.loc[summary.scenario == "0"]
    rows: List[dict] = []
    for _, srow in syn.iterrows():
        for _, rrow in ref.iterrows():
            rows.append({
                "synthetic_grid": srow.grid,
                "reference_grid": rrow.grid,
                "total_capacity_ratio": (
                    srow.p_inst_total_mw / rrow.p_inst_total_mw
                ),
                "direct_capacity_ratio": (
                    srow.p_direct_110kv_mw / rrow.p_direct_110kv_mw
                ),
                "mw_per_station_ratio": (
                    srow.mw_per_station / rrow.mw_per_station
                ),
                "mw_per_route_km_ratio": (
                    srow.mw_per_route_km / rrow.mw_per_route_km
                ),
                "generation_to_load_ratio_ratio": (
                    srow.generation_to_load_ratio
                    / rrow.generation_to_load_ratio
                ),
            })
    return pd.DataFrame(rows)


def build_report(
    summary: pd.DataFrame,
    by_scope: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    """Build the concise, thesis-oriented Markdown report."""
    scenario0 = summary.loc[
        summary.scenario.isin(["0", "synthetic"])
    ].copy()
    scenario0_cols = [
        "grid", "n_units", "p_inst_total_mw", "sn_total_mva",
        "p_direct_110kv_mw", "p_equiv_below_110kv_mw",
        "p_wind_direct_mw", "p_pv_direct_mw",
        "load_p_reference_mw", "generation_to_load_ratio",
    ]
    density_cols = [
        "grid", "stations_reduced", "route_km", "p_inst_total_mw",
        "mw_per_station", "mw_per_route_km",
    ]
    scenarios = summary.loc[
        summary.scenario.isin(["0", "1", "2"])
    ][[
        "grid", "scenario", "n_units", "p_inst_total_mw",
        "p_direct_110kv_mw", "p_equiv_below_110kv_mw",
        "p_wind_direct_mw", "p_pv_direct_mw",
    ]]

    syn = summary.loc[summary.scenario == "synthetic"].copy()
    hv1 = summary.loc[
        (summary.topology == "HV1") & (summary.scenario == "0")
    ].iloc[0]
    hv2 = summary.loc[
        (summary.topology == "HV2") & (summary.scenario == "0")
    ].iloc[0]
    syn_ratio = float(syn.generation_to_load_ratio.iloc[0])
    direct_syn = float(syn.p_direct_110kv_mw.iloc[0])
    wind_syn = float(syn.p_wind_direct_mw.iloc[0])
    pv_syn = float(syn.p_pv_direct_mw.iloc[0])

    out: List[str] = []
    out.append(
        "# SimBench generation benchmark for the synthetic 110 kV DS\n"
    )
    out.append(
        "Reference grids are the two SimBench HV grid families. "
        "Scenario 0 is the primary comparison because it is the scenario used "
        "by `analysis/simbench_hv_benchmark.py`; scenarios 1 and 2 are retained "
        "as expansion-scenario sensitivity cases. They also add generator-"
        "interconnection buses and lines, so their topology counts differ.\n"
    )
    out.append(
        "Installed active generation is `sum(p_mw)` over in-service, non-zero "
        "`gen`/`sgen` rows represented on the 110 kV component. `ext_grid` "
        "and storage are excluded; storage `p_mw` is a signed dispatch point, "
        "not generation nameplate. `sum(sn_mva)` is reported separately. "
        "For SimBench, downstream MV equivalents are not silently treated as "
        "direct 110 kV plants.\n"
    )

    out.append("\n## 1. Scenario-0 and synthetic totals\n")
    out.append(_md_table(scenario0[scenario0_cols], "{:.3f}"))

    out.append("\n\n## 2. Normalised generation density\n")
    out.append(_md_table(scenario0[density_cols], "{:.3f}"))
    out.append(
        "\n\n`MW/route-km` is a geometric density indicator, not an electrical "
        "loading measure. It changes across DSO_1--DSO_4 because the same "
        "410 MW generation portfolio is retained while line-length scale "
        "changes from 0.52 to 2.44."
    )

    out.append("\n\n## 3. SimBench expansion-scenario sensitivity\n")
    out.append(_md_table(scenarios, "{:.3f}"))

    out.append("\n\n## 4. Scenario-0 synthetic/reference ratios\n")
    out.append(_md_table(comparison, "{:.3f}"))

    out.append("\n\n## 5. Assessment\n")
    out.append(
        f"- **Synthetic model fact.** Every DSO has {direct_syn:.0f} MW "
        f"installed directly on its represented 110 kV buses: "
        f"{wind_syn:.0f} MW wind and {pv_syn:.0f} MW PV in 10 controllable "
        f"`sgen` elements. With {syn.load_p_reference_mw.iloc[0]:.1f} MW "
        f"reference active demand, installed generation / reference demand "
        f"is {syn_ratio:.2f}."
    )
    out.append(
        f"- **SimBench scenario-0 fact.** HV1 contains "
        f"{hv1.p_inst_total_mw:.1f} MW in total, of which "
        f"{hv1.p_direct_110kv_mw:.1f} MW is directly connected at 110 kV and "
        f"{hv1.p_equiv_below_110kv_mw:.1f} MW represents generation below "
        f"110 kV. HV2 contains {hv2.p_inst_total_mw:.1f} MW in total "
        f"({hv2.p_direct_110kv_mw:.1f} MW direct, "
        f"{hv2.p_equiv_below_110kv_mw:.1f} MW equivalent downstream)."
    )
    out.append(
        f"- **Load-relative result.** The synthetic ratio {syn_ratio:.2f} lies "
        f"between SimBench HV1 ({hv1.generation_to_load_ratio:.2f}) and HV2 "
        f"({hv2.generation_to_load_ratio:.2f}) for scenario 0. This supports "
        f"the order of magnitude of the aggregate generation, but does not "
        f"validate the exact spatial placement or reactive capability."
    )
    out.append(
        "- **Density result.** At 41.0 MW per reduced station, each synthetic "
        "DS is more generation-dense than scenario-0 HV1 and HV2. Per route-km "
        "the conclusion depends strongly on the newly calibrated topology: "
        "DSO_2 is close to HV1, DSO_4 is close to HV2, while DSO_1 and "
        "especially DSO_3 are substantially denser."
    )
    out.append(
        "- **Carrier result.** The synthetic direct-HV mix is 65.9 % wind / "
        "34.1 % PV. SimBench HV1 scenario 0 has only wind among its direct-HV "
        "plants; HV2 is predominantly wind with 10 MW direct-HV PV. The "
        "synthetic model therefore deliberately carries a larger HV-level PV "
        "share than either scenario-0 reference."
    )

    out.append("\n\n## 6. Constraints and unresolved interpretation\n")
    out.append(
        "- SimBench equivalent-MV injections use aggregate `mv_*` profiles, so "
        "their carrier composition cannot be recovered from the HV-only net. "
        "Use an HVMV model if a carrier-resolved downstream comparison is "
        "required."
    )
    out.append(
        "- The SimBench element count and installed MVA are not a proxy for "
        "available DSO control actuators. The synthetic controller dispatches "
        "all 10 `sgen` Q setpoints using its cached sensitivities and "
        "VDE-AR-N-4120 operating diagrams; SimBench rows are parsed here only "
        "as an asset benchmark."
    )
    out.append(
        "- The generation-to-load ratio compares SimBench static reference "
        "load with the synthetic constructed reference load. Time-series "
        "coincidence (wind/PV/load profiles) is outside this installed-capacity "
        "comparison."
    )
    out.append(
        "- SimBench scenarios 1 and 2 contain storage. It is intentionally "
        "excluded from installed generation; a storage-capacity comparison "
        "would require separate MW/MWh metrics."
    )
    out.append(
        "- Open modelling question: should the 410 MW portfolio remain equal "
        "in all four DSOs after their footprints were differentiated, or "
        "should installed generation scale with area, demand, or coupling "
        "capacity? No network/controller parameter is changed by this analysis."
    )
    return "\n".join(out)


def make_figure(summary: pd.DataFrame, path: Path) -> Optional[Path]:
    """Plot scenario-0 capacity split and topology-normalised densities."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figure: {exc}")
        return None

    data = summary.loc[summary.scenario.isin(["0", "synthetic"])].copy()
    short = data.grid.str.replace(
        r"SimBench (HV[12]).*", r"SB \1", regex=True
    ).str.replace("Synthetic DS ", "", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(data))
    axes[0].bar(
        x, data.p_direct_110kv_mw, label="direct 110 kV", color="tab:blue"
    )
    axes[0].bar(
        x,
        data.p_equiv_below_110kv_mw,
        bottom=data.p_direct_110kv_mw,
        label="equivalent below 110 kV",
        color="tab:orange",
    )
    axes[0].set_ylabel("installed active generation [MW]")
    axes[0].set_title("Capacity represented at HV")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, data.mw_per_station, color="tab:green")
    axes[1].set_ylabel("MW per reduced station")
    axes[1].set_title("Station-normalised density")

    axes[2].bar(x, data.mw_per_route_km, color="tab:purple")
    axes[2].set_ylabel("MW per route-km")
    axes[2].set_title("Route-normalised density")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare installed SimBench HV generation with the "
        "synthetic IEEE39-underlaid DSO networks."
    )
    ap.add_argument(
        "--outdir",
        default="results/simbench_hv_generation_benchmark",
        help="output directory for CSV/PNG/Markdown artefacts",
    )
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import simbench as sb

    unit_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    # SimBench: both HV grid families and all three expansion scenarios.
    for label, (root, scenario, code) in SIMBENCH_GENERATION_CODES.items():
        print(f"[load] {label:<40s} <- {code}")
        net = sb.get_simbench_net(code)
        topology = analyse_grid(net, label)
        units, n_zero = parse_generation(
            net,
            label=label,
            topology=root,
            scenario=str(scenario),
            simbench_root=root,
        )
        unit_frames.append(units)
        summary_rows.append(summarize_case(
            units=units,
            topology=topology,
            topology_name=root,
            scenario=str(scenario),
            load_p_mw=reference_load_p_mw(net),
            zero_excluded=n_zero,
        ))

    # Synthetic: build the actual IEEE39 + all configured DSOs, then parse
    # each sub-network by its name prefix. This follows the same code path as
    # the controller experiments and captures all coupling wind parks.
    built = extract_synthetic_from_built_net(verbose=False)
    if not built:
        raise RuntimeError("could not build IEEE39 + synthetic DSO network")
    full = next(iter(built.values()))
    for group in _hv_subnetwork_ids(full):
        label = f"Synthetic DS ({group})"
        topology = analyse_grid(full, label, group=group)
        units, n_zero = parse_generation(
            full,
            label=label,
            topology=group,
            scenario="synthetic",
            group=group,
        )
        unit_frames.append(units)
        summary_rows.append(summarize_case(
            units=units,
            topology=topology,
            topology_name=group,
            scenario="synthetic",
            load_p_mw=reference_load_p_mw(full, group=group),
            zero_excluded=n_zero,
        ))

    units = pd.concat(unit_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    by_scope = (
        units.groupby(
            ["grid", "topology", "scenario", "scope", "carrier"],
            dropna=False,
        )
        .agg(
            n_units=("p_inst_mw", "size"),
            p_inst_mw=("p_inst_mw", "sum"),
            sn_mva=("sn_mva", "sum"),
        )
        .reset_index()
    )
    comparison = build_scenario0_comparison(summary)

    units.to_csv(outdir / "generation_units.csv", index=False)
    by_scope.to_csv(outdir / "generation_by_scope_carrier.csv", index=False)
    summary.to_csv(outdir / "generation_summary.csv", index=False)
    comparison.to_csv(outdir / "scenario0_comparison.csv", index=False)

    report = build_report(summary, by_scope, comparison)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    if not args.no_figure:
        make_figure(
            summary, outdir / "simbench_hv_generation_benchmark.png"
        )

    print("\n" + report + "\n")
    print(f"[done] artefacts written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
