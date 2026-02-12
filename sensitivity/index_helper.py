"""
Index-Helper Module
===============

This module provides PPC or PANDAPOWER indices, or converts them between each other.

Author: Manuel Schwenke
Date: 2025-02-05
"""

# imports
from typing import Tuple, List, Optional, Dict
import pandapower as pp
import numpy as np


def get_jacobian_indices(net: pp.pandapowerNet, bus_idx: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Get Jacobian indices for a bus.

    The Jacobian is structured with PV buses first, then PQ buses.
    Voltage magnitude indices are only available for PQ buses.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with solved power flow (contains _ppc internal data).
    bus_idx : int
        Pandapower bus index.

    Returns
    -------
    theta_idx : int or None
        Index in the Jacobian for voltage angle of this bus.
        None if bus is slack.
    v_idx : int or None
        Index in the Jacobian for voltage magnitude of this bus.
        None if bus is PV or slack (voltage is fixed).

    Raises
    ------
    ValueError
        If the network does not have the required internal power flow data.
    """
    if not hasattr(net, '_ppc') or 'internal' not in net._ppc:
        raise ValueError("Network must have converged power flow with internal data.")

    pq_buses = net._ppc['internal']['pq']
    pv_buses = net._ppc['internal']['pv']

    if bus_idx in pq_buses:     # Check if bus is PQ (both angle and magnitude in state vector)
        pq_idx = list(pq_buses).index(bus_idx)
        n_pv = len(pv_buses)
        theta_idx = n_pv + pq_idx
        v_idx = pq_idx
        return theta_idx, v_idx
    elif bus_idx in pv_buses:   # Check if bus is PV (only angle in state vector)
        pv_idx = list(pv_buses).index(bus_idx)
        theta_idx = pv_idx
        return theta_idx, None
    else:                       # Bus is slack (not in state vector)
        return None, None


def get_ppc_trafo_index(net: pp.pandapowerNet, trafo_idx: int) -> Optional[int]:
    """
    Get the pypower branch index for a pandapower transformer.

    In pandapower's internal pypower representation, branches are ordered as:
    [lines, trafos, trafo3w, impedances, ...]. This function finds the
    correct index for a given transformer.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with solved power flow.
    trafo_idx : int
        Pandapower transformer index.

    Returns
    -------
    ppc_idx : int or None
        Index in _ppc['branch'] array, or None if not found.

    Raises
    ------
    ValueError
        If the network does not have the required internal lookup data.
    """
    if not hasattr(net, '_ppc'):
        raise ValueError("Network must have converged power flow with _ppc data.")

    # Try using lookup tables first (if available)
    if hasattr(net, '_pd2ppc_lookups') and net._pd2ppc_lookups is not None:
        try:
            trafo_start = net._pd2ppc_lookups['branch']['trafo'][0]
            ppc_idx = trafo_start + trafo_idx
            if ppc_idx >= len(net._ppc['branch']):
                return None
            return ppc_idx
        except (KeyError, IndexError, TypeError):
            pass

    # Fallback: In ppc['branch'] transformers follow lines in branch array -> take n_lines from net.line
    n_lines = len(net.line) if 'line' in net else 0
    ppc_idx = n_lines + trafo_idx
    if ppc_idx >= len(net._ppc['branch']):
        return None
    return ppc_idx


def pp_bus_to_ppc_bus(net: pp.pandapowerNet, pp_bus_idx: int) -> int:
    """
    Map a pandapower bus index to its internal pypower (_ppc) bus index.

    In pandapower, bus ordering may differ between the user-facing DataFrame
    and the internal pypower representation.  This is particularly relevant
    when auxiliary buses are created for 3-winding transformer star points.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with solved power flow.
    pp_bus_idx : int
        Pandapower bus index (as in ``net.bus.index``).

    Returns
    -------
    ppc_bus_idx : int
        Corresponding index in ``net._ppc['bus']``.

    Raises
    ------
    ValueError
        If the mapping cannot be established.
    """
    if not hasattr(net, '_pd2ppc_lookups') or net._pd2ppc_lookups is None:
        raise ValueError(
            "Network must have _pd2ppc_lookups (run power flow first)."
        )

    bus_lookup = net._pd2ppc_lookups.get('bus')
    if bus_lookup is None:
        raise ValueError("Bus lookup table not available in _pd2ppc_lookups.")

    if pp_bus_idx < 0 or pp_bus_idx >= len(bus_lookup):
        raise ValueError(
            f"Pandapower bus index {pp_bus_idx} out of range for "
            f"bus lookup of length {len(bus_lookup)}."
        )

    ppc_bus_idx = int(bus_lookup[pp_bus_idx])
    return ppc_bus_idx


def get_jacobian_indices_ppc(
    net: pp.pandapowerNet,
    ppc_bus_idx: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Get Jacobian matrix indices for a bus specified by its _ppc bus index.
    It is required for auxiliary buses (e.g. 3-winding transformer star points)
    which have no pandapower bus index but do appear in the Jacobian.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with solved power flow.
    ppc_bus_idx : int
        Internal pypower bus index.

    Returns
    -------
    theta_idx : int or None
        Row/column index for the voltage angle of this bus in the Jacobian.
        ``None`` if the bus is the slack bus.
    v_idx : int or None
        Row/column index for the voltage magnitude of this bus in the
        reduced Jacobian (``dV_dQ_reduced``).  ``None`` if the bus is PV
        or slack.
    """
    if not hasattr(net, '_ppc') or 'internal' not in net._ppc:
        raise ValueError("Network must have converged power flow with internal data.")

    pq_buses = net._ppc['internal']['pq']
    pv_buses = net._ppc['internal']['pv']

    pq_match = np.where(pq_buses == ppc_bus_idx)[0]     # Check PQ buses (both angle and magnitude in state vector)
    if len(pq_match) > 0:
        pq_pos = int(pq_match[0])
        n_pv = len(pv_buses)
        theta_idx = n_pv + pq_pos
        v_idx = pq_pos
        return theta_idx, v_idx

    pv_match = np.where(pv_buses == ppc_bus_idx)[0]     # Check PV buses (only angle in state vector)
    if len(pv_match) > 0:
        pv_pos = int(pv_match[0])
        theta_idx = pv_pos
        return theta_idx, None

    return None, None                                   # Bus is slack or reference (not in state vector)


def get_ppc_trafo3w_branch_indices(
    net: pp.pandapowerNet,
    trafo3w_idx: int,
) -> Tuple[int, int, int, int]:
    """
    Get the three internal pypower branch indices for a 3-winding transformer.
    Pandapower decomposes each 3-winding transformer into three 2-winding
    equivalent branches connected through an auxiliary star-point bus:

        HV bus ─── [HV branch] ─── star bus ─── [MV branch] ─── MV bus
                                      │
                                 [LV branch]
                                      │
                                   LV bus

    This function identifies which ``_ppc['branch']`` entries correspond
    to the HV, MV, and LV windings by matching their terminal buses.

    Parameters
    ----------
    net : pp.pandapowerNet
        Pandapower network with solved power flow.
    trafo3w_idx : int
        Index into ``net.trafo3w``.

    Returns
    -------
    hv_branch_idx : int
        ``_ppc['branch']`` index for the HV winding (HV bus ↔ star bus).
    mv_branch_idx : int
        ``_ppc['branch']`` index for the MV winding (MV bus ↔ star bus).
    lv_branch_idx : int
        ``_ppc['branch']`` index for the LV winding (LV bus ↔ star bus).
    aux_bus_ppc : int
        ``_ppc['bus']`` index for the auxiliary star-point bus.

    Raises
    ------
    ValueError
        If the trafo3w index is not found, the lookup is unavailable,
        or the internal branch structure cannot be resolved.
    """
    if trafo3w_idx not in net.trafo3w.index:    # trafo id in net.trafo3w table?
        raise ValueError(f"Three-winding transformer {trafo3w_idx} not found in network.")
    if not hasattr(net, '_pd2ppc_lookups') or net._pd2ppc_lookups is None:  # pd2pcc lookup exists?
        raise ValueError("Network must have _pd2ppc_lookups (run power flow first).")

    # Locate the branch range for trafo3w elements in _ppc
    branch_lookup = net._pd2ppc_lookups.get('branch')
    if branch_lookup is None or 'trafo3w' not in branch_lookup:
        raise ValueError("Branch lookup for trafo3w not available in _pd2ppc_lookups.")

    t3w_range = branch_lookup['trafo3w']
    t3w_start = t3w_range[0]

    # Position of this trafo3w element in the DataFrame (handles non-contiguous indices after row deletions)
    trafo3w_positions = list(net.trafo3w.index)
    if trafo3w_idx not in trafo3w_positions:
        raise ValueError(f"trafo3w index {trafo3w_idx} not in net.trafo3w.index.")
    pos = trafo3w_positions.index(trafo3w_idx)

    # Pandapower orders 3W branches by winding type first, then by element:
    #     [all HV branches, all MV branches, all LV branches]
    # i.e. for N trafo3w elements the layout is:
    #     HV[0..N-1], MV[0..N-1], LV[0..N-1]
    N = len(net.trafo3w)    # ToDo: Is this true?
    candidate_indices = [
        t3w_start + pos,           # HV branch
        t3w_start + N + pos,       # MV branch
        t3w_start + 2 * N + pos,   # LV branch
    ]

    n_branches = len(net._ppc['branch'])
    for ci in candidate_indices:
        if ci >= n_branches:
            raise ValueError(
                f"Computed branch index {ci} out of range "
                f"(n_branches={n_branches}).  Check trafo3w lookup."
            )

    # Map pandapower HV/MV/LV buses to _ppc bus indices
    hv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'hv_bus'])
    mv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'mv_bus'])
    lv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'lv_bus'])
    hv_bus_ppc = pp_bus_to_ppc_bus(net, hv_bus_pp)
    mv_bus_ppc = pp_bus_to_ppc_bus(net, mv_bus_pp)
    lv_bus_ppc = pp_bus_to_ppc_bus(net, lv_bus_pp)
    known_buses = {hv_bus_ppc, mv_bus_ppc, lv_bus_ppc}

    # Identify which branch belongs to which winding
    hv_branch_idx: Optional[int] = None
    mv_branch_idx: Optional[int] = None
    lv_branch_idx: Optional[int] = None
    aux_bus_ppc: Optional[int] = None

    for br_idx in candidate_indices:
        from_bus = int(net._ppc['branch'][br_idx, 0])
        to_bus = int(net._ppc['branch'][br_idx, 1])

        # One end is a known winding bus, the other is the auxiliary star bus
        if from_bus in known_buses:
            winding_bus = from_bus
            star_bus = to_bus
        elif to_bus in known_buses:
            winding_bus = to_bus
            star_bus = from_bus
        else:
            raise ValueError(
                f"Branch {br_idx} (from={from_bus}, to={to_bus}) does not "
                f"connect to any known winding bus of trafo3w {trafo3w_idx}."
            )

        # Record auxiliary bus (must be consistent across all 3 branches)
        if aux_bus_ppc is None:
            aux_bus_ppc = star_bus
        elif aux_bus_ppc != star_bus:
            raise ValueError(
                f"Inconsistent auxiliary bus: expected {aux_bus_ppc}, "
                f"got {star_bus} for branch {br_idx}."
            )

        # Assign branch to the correct winding
        if winding_bus == hv_bus_ppc:
            hv_branch_idx = br_idx
        elif winding_bus == mv_bus_ppc:
            mv_branch_idx = br_idx
        elif winding_bus == lv_bus_ppc:
            lv_branch_idx = br_idx

    # Validate that all three branches were found
    if hv_branch_idx is None:
        raise ValueError(f"Could not identify HV branch for trafo3w {trafo3w_idx}.")
    if mv_branch_idx is None:
        raise ValueError(f"Could not identify MV branch for trafo3w {trafo3w_idx}.")
    if lv_branch_idx is None:
        raise ValueError(f"Could not identify LV branch for trafo3w {trafo3w_idx}.")
    if aux_bus_ppc is None:
        raise ValueError(f"Could not identify auxiliary star bus for trafo3w {trafo3w_idx}.")

    return hv_branch_idx, mv_branch_idx, lv_branch_idx, aux_bus_ppc


def _get_trafo3w_hv_branch_data(
    net,
    trafo3w_idx: int,
) -> Dict:
    """
    Extract all data needed for HV-side sensitivity computations
    of a 3-winding transformer.

    Returns a dictionary with keys:
        hv_branch_idx, aux_bus_ppc, hv_bus_ppc, aux_bus_ppc,
        U_hv, U_aux, theta_hv, theta_aux, tau, delta_tau,
        g, b, r_pu, x_pu,
        theta_hv_jac, v_hv_jac, theta_aux_jac, v_aux_jac

    Raises
    ------
    ValueError
        If the transformer or its buses cannot be processed.
    """
    hv_br, _, _, aux_bus_ppc = get_ppc_trafo3w_branch_indices(
        net, trafo3w_idx
    )

    # Pandapower and _ppc bus indices
    hv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'hv_bus'])
    hv_bus_ppc = pp_bus_to_ppc_bus(net, hv_bus_pp)

    # Voltage states from _ppc bus results
    U_hv = float(net._ppc['bus'][hv_bus_ppc, 7])     # VM column
    U_aux = float(net._ppc['bus'][aux_bus_ppc, 7])
    theta_hv = float(np.deg2rad(net._ppc['bus'][hv_bus_ppc, 8]))   # VA column
    theta_aux = float(np.deg2rad(net._ppc['bus'][aux_bus_ppc, 8]))

    # Tap ratio from _ppc branch data
    #tap_ppc = float(self.net._ppc['branch'][hv_br, 8])

    # Pandapower tap parameters
    s0 = float(net.trafo3w.at[trafo3w_idx, 'tap_pos'])
    delta_tau = float(net.trafo3w.at[trafo3w_idx, 'tap_step_percent']) / 100.0
    tau = (1.0 + s0 * delta_tau)

    # Branch impedance
    r_pu = float(net._ppc['branch'][hv_br, 2])
    x_pu = float(net._ppc['branch'][hv_br, 3])
    y_pu = 1.0 / complex(r_pu, x_pu)
    g = y_pu.real
    b = y_pu.imag

    # Jacobian indices (using _ppc bus indices)
    theta_hv_jac, v_hv_jac = get_jacobian_indices_ppc(
        net, hv_bus_ppc
    )
    theta_aux_jac, v_aux_jac = get_jacobian_indices_ppc(
        net, aux_bus_ppc
    )

    return {
        'hv_branch_idx': hv_br,
        'hv_bus_ppc': hv_bus_ppc,
        'aux_bus_ppc': aux_bus_ppc,
        'U_hv': U_hv,
        'U_aux': U_aux,
        'theta_hv': theta_hv,
        'theta_aux': theta_aux,
        'tau': tau,
        'delta_tau': delta_tau,
        'g': g,
        'b': b,
        'r_pu': r_pu,
        'x_pu': x_pu,
        'theta_hv_jac': theta_hv_jac,
        'v_hv_jac': v_hv_jac,
        'theta_aux_jac': theta_aux_jac,
        'v_aux_jac': v_aux_jac,
    }