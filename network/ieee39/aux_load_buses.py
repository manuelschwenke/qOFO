"""Internal load-separation nodes for pandapower ZIP consistency.

pandapower stores one ZIP composition per ppc bus and applies that factor to
the aggregate bus demand.  A fixed-power ``sgen`` sharing a bus with a ZIP
load is therefore inadvertently voltage-scaled inside Newton--Raphson even
though ``res_sgen`` reports the unscaled setpoint.  This module creates a
numerically near-coincident load node so load and injection occupy distinct
ppc buses in both pandapower and PowerFactory.

A closed zero-impedance bus-bus switch is unsuitable: pandapower fuses its
endpoints into one ppc node.  The explicit 0.01 + j0.01 ohm link was selected
by the Phase-3 t0/peak sweep; it is small enough to be physically negligible
but avoids the ill-conditioning observed for a 1e-5 ohm link.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandapower as pp


AUX_LOAD_LINK_LENGTH_KM: float = 1.0
AUX_LOAD_LINK_R_OHM: float = 0.01
AUX_LOAD_LINK_X_OHM: float = 0.01
AUX_LOAD_LINK_MAX_I_KA: float = 10.0


def separate_colocated_zip_loads(
    net: pp.pandapowerNet,
    *,
    load_indices: Sequence[int],
    injection_sgen_indices: Sequence[int],
    aux_subnet: str,
    name_prefix: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Move selected loads off buses carrying selected fixed-PQ injections.

    Returns parallel lists ``(aux_buses, parent_buses, aux_lines)``.  The
    injection remains at the physical parent bus; all selected load rows at
    that parent move together to one internal auxiliary node.
    """
    selected_loads = [int(i) for i in load_indices]
    selected_sgens = [int(i) for i in injection_sgen_indices]
    missing_loads = sorted(set(selected_loads) - set(int(i) for i in net.load.index))
    missing_sgens = sorted(set(selected_sgens) - set(int(i) for i in net.sgen.index))
    if missing_loads or missing_sgens:
        raise ValueError(
            "Auxiliary-load separation received unknown indices: "
            f"loads={missing_loads}, sgens={missing_sgens}"
        )

    injection_buses = {
        int(net.sgen.at[sidx, "bus"]) for sidx in selected_sgens
        if bool(net.sgen.at[sidx, "in_service"])
    }
    aux_buses: List[int] = []
    parent_buses: List[int] = []
    aux_lines: List[int] = []

    for parent in sorted(injection_buses):
        at_parent = [
            idx for idx in selected_loads
            if bool(net.load.at[idx, "in_service"])
            and int(net.load.at[idx, "bus"]) == parent
        ]
        if not at_parent:
            continue

        if name_prefix:
            bus_name = f"{name_prefix}|AUX_LOAD|parent_bus{parent}"
            line_name = f"{name_prefix}|AUX_LOAD_LINK|parent_bus{parent}"
        else:
            # Preserve the established Phase-3 wind-replacement names.
            bus_name = f"AUX_LOAD|grid_bus{parent}"
            line_name = f"AUX_LOAD_LINK|grid_bus{parent}"

        aux_bus = int(pp.create_bus(
            net,
            vn_kv=float(net.bus.at[parent, "vn_kv"]),
            name=bus_name,
            type="b",
            in_service=True,
            subnet=aux_subnet,
        ))
        aux_line = int(pp.create_line_from_parameters(
            net,
            from_bus=parent,
            to_bus=aux_bus,
            length_km=AUX_LOAD_LINK_LENGTH_KM,
            r_ohm_per_km=AUX_LOAD_LINK_R_OHM / AUX_LOAD_LINK_LENGTH_KM,
            x_ohm_per_km=AUX_LOAD_LINK_X_OHM / AUX_LOAD_LINK_LENGTH_KM,
            c_nf_per_km=0.0,
            g_us_per_km=0.0,
            max_i_ka=AUX_LOAD_LINK_MAX_I_KA,
            df=1.0,
            parallel=1,
            name=line_name,
            type="ol",
            in_service=True,
            subnet=aux_subnet,
        ))
        net.load.loc[at_parent, "bus"] = aux_bus
        aux_buses.append(aux_bus)
        parent_buses.append(parent)
        aux_lines.append(aux_line)

    return aux_buses, parent_buses, aux_lines
