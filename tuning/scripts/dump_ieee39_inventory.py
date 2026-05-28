"""
Dump a human-readable inventory of the IEEE 39 + HV-attached network
(scenario = ``wind_replace``) to ``docs/network_inventory.md``.

Tables produced
---------------
1. Synchronous generators (incl. slack), with terminal/grid bus, zone,
   nameplate S_n, max P, max |Q|, and fuel type.
2. TSO DERs — sgens with ``subnet == "TN"`` (wind park STATCOMs).
3. DSO DERs — sgens with ``subnet == "DN"``, grouped per DSO sub-network.
4. TN loads — split into constant and profile rows by ``_split_tn_loads``.
5. DSO HV loads — 10 buses x 2 rows (const + var) per DSO.
6. Distribution systems — one row per ``meta.hv_networks`` entry, with
   coupling buses, 3W trafo sn_hv_mva, total reference P/Q, and counts.

Re-generate:
    python tuning/scripts/dump_ieee39_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Make the repo root importable when the script is launched directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from network.ieee39 import add_hv_networks, build_ieee39_net
from network.ieee39.constants import GEN_NAMEPLATE
from network.zone_partition import fixed_zone_partition_ieee39


OUTPUT_PATH = _REPO_ROOT / "docs" / "network_inventory.md"


# ---------------------------------------------------------------------------
#  Formatting helpers
# ---------------------------------------------------------------------------


def _esc(cell: str) -> str:
    """Escape characters that break GFM tables (``|`` is the cell separator)."""
    return cell.replace("|", "\\|")


def _md_table(headers: List[str], rows: List[List[str]],
              align: Optional[List[str]] = None) -> str:
    """Render a GFM table; ``align`` is one of {'l', 'c', 'r'} per column."""
    if not rows:
        return f"| {' | '.join(headers)} |\n| {' | '.join('---' for _ in headers)} |\n| _(empty)_ |\n"
    if align is None:
        align = ["l"] * len(headers)
    sep_for = {"l": ":---", "c": ":---:", "r": "---:"}
    sep = [sep_for[a] for a in align]
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_esc(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _fmt_num(x: float, prec: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{prec}f}"


def _bus_zone_label(b: int, bus_zone: dict) -> str:
    z = bus_zone.get(int(b))
    return str(z) if z is not None else "—"


def _parse_dso_id(name: str) -> Optional[str]:
    """Extract ``DSO_x`` from a name like ``"DSO_1|..."``; None if no prefix."""
    if not isinstance(name, str):
        return None
    head, sep, _ = name.partition("|")
    if sep and head.startswith("DSO_"):
        return head
    return None


# ---------------------------------------------------------------------------
#  Section builders
# ---------------------------------------------------------------------------


def _gen_nameplate_fuel(name: str) -> str:
    """Recover the GEN_NAMEPLATE fuel via the bus encoded in the gen name.

    The naming convention from ``build_ieee39_net`` is ``f"{label}_bus{N}"``
    where ``N`` is the original IEEE bus 0-idx, which is a stable
    GEN_NAMEPLATE key even when the two-trafo-chain refactor in
    ``build.py`` reassigns the terminal bus to a new LV node.
    """
    if not isinstance(name, str) or "_bus" not in name:
        return "—"
    try:
        original_bus = int(name.rsplit("_bus", 1)[1])
    except ValueError:
        return "—"
    entry = GEN_NAMEPLATE.get(original_bus)
    return entry[2] if entry is not None else "—"


def _section_synchronous_gens(net, meta, bus_zone) -> str:
    grid_bus_map = {
        int(g): int(gb)
        for g, gb in zip(meta.gen_indices, meta.gen_grid_bus_indices)
    }
    headers = ["gen_idx", "name", "term bus (0-idx)", "grid bus (0-idx)",
               "zone", "slack", "S_n [MVA]", "P_max [MW]",
               "Q_max [Mvar]", "fuel"]
    align = ["r", "l", "r", "r", "c", "c", "r", "r", "r", "l"]
    rows: List[List[str]] = []
    for g in net.gen.index:
        term_bus = int(net.gen.at[g, "bus"])
        grid_bus = grid_bus_map.get(int(g), term_bus)
        slack = bool(net.gen.at[g, "slack"]) if "slack" in net.gen.columns else False
        sn = float(net.gen.at[g, "sn_mva"])
        p_max = float(net.gen.at[g, "max_p_mw"])
        q_max = float(net.gen.at[g, "max_q_mvar"])
        name = str(net.gen.at[g, "name"])
        rows.append([
            str(int(g)),
            name,
            str(term_bus),
            str(grid_bus),
            _bus_zone_label(grid_bus, bus_zone),
            "yes" if slack else "no",
            _fmt_num(sn, 0),
            _fmt_num(p_max, 0),
            _fmt_num(q_max, 0),
            _gen_nameplate_fuel(name),
        ])
    return _md_table(headers, rows, align)


def _section_tso_ders(net, bus_zone) -> str:
    headers = ["sgen_idx", "name", "bus (0-idx)", "zone",
               "P_rated [MW]", "S_n [MVA]", "op_diagram", "profile"]
    align = ["r", "l", "r", "c", "r", "r", "l", "l"]
    rows: List[List[str]] = []
    mask = net.sgen["subnet"].astype(str) == "TN"
    for s in net.sgen.index[mask]:
        bus = int(net.sgen.at[s, "bus"])
        rows.append([
            str(int(s)),
            str(net.sgen.at[s, "name"]),
            str(bus),
            _bus_zone_label(bus, bus_zone),
            _fmt_num(float(net.sgen.at[s, "p_mw"]), 0),
            _fmt_num(float(net.sgen.at[s, "sn_mva"]), 0),
            str(net.sgen.at[s, "op_diagram"]) if "op_diagram" in net.sgen.columns else "—",
            str(net.sgen.at[s, "profile"]) if "profile" in net.sgen.columns else "—",
        ])
    return _md_table(headers, rows, align)


def _section_dso_ders(net, meta) -> str:
    """One sub-table per DSO sub-network, with a totals footer per group."""
    headers = ["sgen_idx", "HV bus (0-idx)", "name",
               "P_rated [MW]", "S_n [MVA]", "op_diagram", "profile"]
    align = ["r", "r", "l", "r", "r", "l", "l"]
    out_chunks: List[str] = []
    for hv in meta.hv_networks:
        rows: List[List[str]] = []
        sum_p = 0.0
        sum_sn = 0.0
        for s in hv.sgen_indices:
            bus = int(net.sgen.at[s, "bus"])
            p = float(net.sgen.at[s, "p_mw"])
            sn = float(net.sgen.at[s, "sn_mva"])
            sum_p += p
            sum_sn += sn
            rows.append([
                str(int(s)),
                str(bus),
                str(net.sgen.at[s, "name"]),
                _fmt_num(p, 1),
                _fmt_num(sn, 1),
                str(net.sgen.at[s, "op_diagram"]) if "op_diagram" in net.sgen.columns else "—",
                str(net.sgen.at[s, "profile"]) if "profile" in net.sgen.columns else "—",
            ])
        rows.append([
            "**total**", "", "",
            f"**{_fmt_num(sum_p, 1)}**",
            f"**{_fmt_num(sum_sn, 1)}**",
            "", "",
        ])
        out_chunks.append(f"#### {hv.net_id} (zone {hv.zone})\n\n"
                          + _md_table(headers, rows, align))
    return "\n".join(out_chunks)


def _section_tn_loads(net, bus_zone) -> str:
    headers = ["load_idx", "bus (0-idx)", "zone", "name",
               "P [MW]", "Q [Mvar]", "P_base [MW]", "Q_base [Mvar]",
               "profile_p", "profile_q", "role"]
    align = ["r", "r", "c", "l", "r", "r", "r", "r", "l", "l", "l"]
    rows: List[List[str]] = []
    mask = net.load["subnet"].astype(str) == "TN"
    for li in net.load.index[mask]:
        bus = int(net.load.at[li, "bus"])
        prof_p = net.load.at[li, "profile_p"]
        prof_q = net.load.at[li, "profile_q"]
        role = "const" if (prof_p is None or str(prof_p) in ("", "nan", "None", "NaN")) else "profile"
        base_p = float(net.load.at[li, "base_p_mw"]) if "base_p_mw" in net.load.columns else float("nan")
        base_q = float(net.load.at[li, "base_q_mvar"]) if "base_q_mvar" in net.load.columns else float("nan")
        rows.append([
            str(int(li)),
            str(bus),
            _bus_zone_label(bus, bus_zone),
            str(net.load.at[li, "name"]),
            _fmt_num(float(net.load.at[li, "p_mw"]), 2),
            _fmt_num(float(net.load.at[li, "q_mvar"]), 2),
            _fmt_num(base_p, 2),
            _fmt_num(base_q, 2),
            str(prof_p) if prof_p is not None else "—",
            str(prof_q) if prof_q is not None else "—",
            role,
        ])
    return _md_table(headers, rows, align)


def _section_dso_loads(net, meta) -> str:
    """One sub-table per DSO sub-network."""
    headers = ["load_idx", "HV bus (0-idx)", "name",
               "P [MW]", "Q [Mvar]", "P_base [MW]", "Q_base [Mvar]",
               "profile_p", "profile_q", "role"]
    align = ["r", "r", "l", "r", "r", "r", "r", "l", "l", "l"]
    out_chunks: List[str] = []
    for hv in meta.hv_networks:
        rows: List[List[str]] = []
        sum_p = 0.0
        sum_q = 0.0
        for li in hv.load_indices:
            bus = int(net.load.at[li, "bus"])
            prof_p = net.load.at[li, "profile_p"]
            prof_q = net.load.at[li, "profile_q"]
            nm = str(net.load.at[li, "name"])
            role = "var" if nm.endswith("_var") else "const"
            base_p = float(net.load.at[li, "base_p_mw"]) if "base_p_mw" in net.load.columns else float("nan")
            base_q = float(net.load.at[li, "base_q_mvar"]) if "base_q_mvar" in net.load.columns else float("nan")
            p = float(net.load.at[li, "p_mw"])
            q = float(net.load.at[li, "q_mvar"])
            sum_p += p
            sum_q += q
            rows.append([
                str(int(li)),
                str(bus),
                nm,
                _fmt_num(p, 2),
                _fmt_num(q, 2),
                _fmt_num(base_p, 2),
                _fmt_num(base_q, 2),
                str(prof_p) if prof_p is not None else "—",
                str(prof_q) if prof_q is not None else "—",
                role,
            ])
        rows.append([
            "**total**", "", "",
            f"**{_fmt_num(sum_p, 1)}**",
            f"**{_fmt_num(sum_q, 1)}**",
            "", "", "", "", "",
        ])
        out_chunks.append(f"#### {hv.net_id} (zone {hv.zone})\n\n"
                          + _md_table(headers, rows, align))
    return "\n".join(out_chunks)


def _section_distribution_systems(net, meta) -> str:
    headers = ["net_id", "zone", "coupling IEEE bus (0-idx, 1-idx)",
               "HV bus (0-idx)", "3W trafo idx", "S_hv [MVA]",
               "ref P [MW]", "ref Q [Mvar]",
               "n_buses", "n_lines", "n_sgens", "n_loads"]
    align = ["l", "c", "l", "l", "l", "r", "r", "r", "r", "r", "r", "r"]
    rows: List[List[str]] = []
    for hv in meta.hv_networks:
        ieee_pairs = ", ".join(f"{b} ({b + 1})" for b in hv.coupling_ieee_buses)
        hv_buses_str = ", ".join(str(b) for b in hv.coupling_hv_bus_indices)
        trafo_str = ", ".join(str(int(t)) for t in hv.coupling_trafo_indices)
        sn_hv = []
        for t in hv.coupling_trafo_indices:
            sn_hv.append(_fmt_num(float(net.trafo3w.at[int(t), "sn_hv_mva"]), 0))
        rows.append([
            hv.net_id,
            str(int(hv.zone)),
            ieee_pairs,
            hv_buses_str,
            trafo_str,
            ", ".join(sn_hv),
            _fmt_num(hv.total_ref_p_mw, 1),
            _fmt_num(hv.total_ref_q_mvar, 1),
            str(len(hv.bus_indices)),
            str(len(hv.line_indices)),
            str(len(hv.sgen_indices)),
            str(len(hv.load_indices)),
        ])
    return _md_table(headers, rows, align)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    net, meta = build_ieee39_net(scenario="wind_replace", verbose=False)
    meta = add_hv_networks(
        net, meta, install_tso_tertiary_shunts=False, verbose=False,
    )
    _zone_map, bus_zone = fixed_zone_partition_ieee39(net, verbose=False)

    preamble = (
        "# IEEE 39-bus + HV Sub-Networks — Inventory\n\n"
        "Scenario: `wind_replace` (see\n"
        "[`network/ieee39/scenarios/wind_replace.md`](../network/ieee39/scenarios/wind_replace.md)).\n"
        "Auto-generated by `tuning/scripts/dump_ieee39_inventory.py`.\n\n"
        "Buses are 0-indexed (pandapower convention).  The fixed 3-area zone\n"
        "partition (see `network/zone_partition.fixed_zone_partition_ieee39`)\n"
        "is used throughout: Zone 1 = NE around buses 0–1, 25–29, 36–38; "
        "Zone 2 = central around 2–13, 30–31; Zone 3 = SW around 14–23, 32–35.\n"
    )

    sections = [
        ("## 1. Synchronous generators", _section_synchronous_gens(net, meta, bus_zone)),
        ("## 2. TSO DERs (wind-park STATCOMs)", _section_tso_ders(net, bus_zone)),
        ("## 3. DSO DERs (per sub-network)", _section_dso_ders(net, meta)),
        ("## 4. TN loads", _section_tn_loads(net, bus_zone)),
        ("## 5. DSO HV loads", _section_dso_loads(net, meta)),
        ("## 6. Distribution systems", _section_distribution_systems(net, meta)),
    ]

    body_chunks: List[str] = [preamble]
    for header, body in sections:
        body_chunks.append("\n" + header + "\n\n" + body)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(body_chunks), encoding="utf-8")

    print(f"Wrote inventory to {OUTPUT_PATH}")
    print(f"  net.gen     : {len(net.gen)} rows")
    print(f"  net.sgen TN : {int((net.sgen['subnet'].astype(str) == 'TN').sum())} rows")
    print(f"  net.sgen DN : {int((net.sgen['subnet'].astype(str) == 'DN').sum())} rows")
    print(f"  net.load TN : {int((net.load['subnet'].astype(str) == 'TN').sum())} rows")
    print(f"  net.load DN : {int((net.load['subnet'].astype(str) == 'DN').sum())} rows")
    print(f"  hv_networks : {len(meta.hv_networks)} DSOs")


if __name__ == "__main__":
    main()
