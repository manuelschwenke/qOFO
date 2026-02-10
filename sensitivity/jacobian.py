"""
Jacobian Module
===============

This module provides Jacobian-based sensitivity calculations derived from
the power flow equations.

All sensitivities are computed from a cached NetworkState and are valid
for small deviations around that operating point.

Mathematical Background
-----------------------
The power flow equations in compact form:

    g(x) = [P(θ,V) - P_G - P_D] = 0
          [Q(θ,V) - Q_G - Q_D]

where x = [θ, V]^T is the state vector.

The Jacobian matrix:

    J = [∂P/∂θ  ∂P/∂V] = [J_Pθ  J_PV]
        [∂Q/∂θ  ∂Q/∂V]   [J_Qθ  J_QV]

The reduced Jacobian (neglecting active power changes):

    J_r^{-1} = (J_QV - J_Qθ J_Pθ^{-1} J_PV)^{-1} = ∂V/∂Q

The following Sensitivities are implemented:

∂V/∂XXX
-------------
1)  ∂V/∂Q_n = J_r^{-1}                                          -> compute_dV_dQ_der()
2)  ∂V/∂s_i (2W) = Δτ·[D₂₁ D₂₂]·(∂g/∂τ_i)                     -> compute_dV_ds()
                                                                -> compute_dV_ds_matrix()
2b) ∂V/∂s_i (3W) = same formulation for HV winding of 3W trafo  -> compute_dV_ds_trafo3w()
                                                                -> compute_dV_ds_trafo3w_matrix()
3)  ∂V/∂Q_shunt: uses equations from 1)                         -> compute_dV_dQ_shunt()

∂Q_tr/∂XXX (2-winding transformers)
------------------------------------
4)  ∂Q_tr/∂Q_n = (∂Q_ij/∂V_i·∂V_i/∂Q_n + ∂Q_ij/∂V_j·∂V_j/∂Q_n) -> compute_dQtrafo_dQ_der()
                                                                -> compute_dQtrafo_dQ_der_matrix()
5)  ∂Q_tr/∂s_i = Δτ·(∂Q_ij/∂x·J⁻¹·∂g/∂τ_i + ∂Q_ij/∂τ_i)       -> compute_dQtrafo_ds()
                                                                -> compute_dQtrafo_ds_matrix()

∂Q_HV/∂XXX (3-winding transformers, HV-side reactive power)
-------------------------------------------------------------
6)  ∂Q_HV/∂Q_n: chain rule via HV-star internal branch          -> compute_dQtrafo3w_hv_dQ_der()
                                                                -> compute_dQtrafo3w_hv_dQ_der_matrix()
7)  ∂Q_HV/∂s_i (3W): indirect + direct effects (Eq. 17)         -> compute_dQtrafo3w_hv_ds()
                                                                -> compute_dQtrafo3w_hv_ds_matrix()
8)  ∂Q_HV/∂Q_shunt: delegates to 6) with scaling                -> compute_dQtrafo3w_hv_dQ_shunt()

∂I/∂XXX
-------------
9)  ∂I_ij/∂Q_n = Y_ij·(∂V_i/∂Q_n - ∂V_j/∂Q_n)                  -> compute_dI_dQ_der()
    and ∂|I_ij|/∂Q_n = (1/|I_ij|) Re{I_ij*·∂I_ij/∂Q_n}         -> compute_dI_dQ_der_matrix()
10) ∂I_ij/∂s_i:                                                 -> ToDo: Not implemented yet!
11) ∂I_ij/∂Q_shunt: uses equations from 9)                      -> compute_dI_dQ_shunt()

Author: Manuel Schwenke
Date: 2025-02-05

References
----------
[1] Schwenke et al., PSCC 2026, Section II (Linearised Power System Model)
[2] Milano, F. "Power System Modelling and Scripting", Springer 2010
[3] Kundur & Malik, "Power System Stability and Control", 2nd ed.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Dict
import pandapower as pp
import copy


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
    
    # Check if bus is PQ (both angle and magnitude in state vector)
    if bus_idx in pq_buses:
        pq_idx = list(pq_buses).index(bus_idx)
        n_pv = len(pv_buses)
        theta_idx = n_pv + pq_idx
        v_idx = pq_idx
        return theta_idx, v_idx
    
    # Check if bus is PV (only angle in state vector)
    elif bus_idx in pv_buses:
        pv_idx = list(pv_buses).index(bus_idx)
        theta_idx = pv_idx
        return theta_idx, None
    
    # Bus is slack (not in state vector)
    else:
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

    # Fallback: In standard pandapower, transformers follow lines in branch array
    # Branch order: lines first, then transformers
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

    This is the internal-index counterpart of :func:`get_jacobian_indices`.
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
        raise ValueError(
            "Network must have converged power flow with internal data."
        )

    pq_buses = net._ppc['internal']['pq']
    pv_buses = net._ppc['internal']['pv']

    # Check PQ buses (both angle and magnitude in state vector)
    pq_match = np.where(pq_buses == ppc_bus_idx)[0]
    if len(pq_match) > 0:
        pq_pos = int(pq_match[0])
        n_pv = len(pv_buses)
        theta_idx = n_pv + pq_pos
        v_idx = pq_pos
        return theta_idx, v_idx

    # Check PV buses (only angle in state vector)
    pv_match = np.where(pv_buses == ppc_bus_idx)[0]
    if len(pv_match) > 0:
        pv_pos = int(pv_match[0])
        theta_idx = pv_pos
        return theta_idx, None

    # Bus is slack or reference (not in state vector)
    return None, None


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
    if trafo3w_idx not in net.trafo3w.index:
        raise ValueError(
            f"Three-winding transformer {trafo3w_idx} not found in network."
        )

    if not hasattr(net, '_pd2ppc_lookups') or net._pd2ppc_lookups is None:
        raise ValueError(
            "Network must have _pd2ppc_lookups (run power flow first)."
        )

    # --- Locate the branch range for trafo3w elements in _ppc ---
    branch_lookup = net._pd2ppc_lookups.get('branch')
    if branch_lookup is None or 'trafo3w' not in branch_lookup:
        raise ValueError(
            "Branch lookup for trafo3w not available in _pd2ppc_lookups."
        )

    t3w_range = branch_lookup['trafo3w']
    t3w_start = t3w_range[0]

    # Position of this trafo3w element in the DataFrame (handles
    # non-contiguous indices after row deletions)
    trafo3w_positions = list(net.trafo3w.index)
    if trafo3w_idx not in trafo3w_positions:
        raise ValueError(
            f"trafo3w index {trafo3w_idx} not in net.trafo3w.index."
        )
    pos = trafo3w_positions.index(trafo3w_idx)

    # Pandapower orders 3W branches by winding type first, then by element:
    #   [all HV branches, all MV branches, all LV branches]
    # i.e. for N trafo3w elements the layout is:
    #   HV[0..N-1], MV[0..N-1], LV[0..N-1]
    N = len(net.trafo3w)
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

    # --- Map pandapower HV/MV/LV buses to _ppc bus indices ---
    hv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'hv_bus'])
    mv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'mv_bus'])
    lv_bus_pp = int(net.trafo3w.at[trafo3w_idx, 'lv_bus'])

    hv_bus_ppc = pp_bus_to_ppc_bus(net, hv_bus_pp)
    mv_bus_ppc = pp_bus_to_ppc_bus(net, mv_bus_pp)
    lv_bus_ppc = pp_bus_to_ppc_bus(net, lv_bus_pp)

    known_buses = {hv_bus_ppc, mv_bus_ppc, lv_bus_ppc}

    # --- Identify which branch belongs to which winding ---
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
        raise ValueError(
            f"Could not identify HV branch for trafo3w {trafo3w_idx}."
        )
    if mv_branch_idx is None:
        raise ValueError(
            f"Could not identify MV branch for trafo3w {trafo3w_idx}."
        )
    if lv_branch_idx is None:
        raise ValueError(
            f"Could not identify LV branch for trafo3w {trafo3w_idx}."
        )
    if aux_bus_ppc is None:
        raise ValueError(
            f"Could not identify auxiliary star bus for trafo3w {trafo3w_idx}."
        )

    return hv_branch_idx, mv_branch_idx, lv_branch_idx, aux_bus_ppc


class JacobianSensitivities:
    """
    Jacobian-based sensitivity calculator for OFO controllers.
    
    This class computes all required sensitivities from a cached network state.
    The sensitivities are used to construct the input-output relationship
    matrix ∇H for the OFO optimisation problem.
    
    The class is designed to be stateless after initialisation: all methods
    compute sensitivities from the stored network state without modifying it.
    
    Attributes
    ----------
    net : pp.pandapowerNet
        Deep copy of the network at the cached operating point.
    J : NDArray[np.float64]
        Full Jacobian matrix from power flow.
    J_inv : NDArray[np.float64]
        Inverse of the full Jacobian matrix.
    pq_buses : NDArray[np.int64]
        Indices of PQ buses in pypower ordering.
    pv_buses : NDArray[np.int64]
        Indices of PV buses in pypower ordering.
    n_theta : int
        Number of voltage angle state variables (n_pv + n_pq).
    n_v : int
        Number of voltage magnitude state variables (n_pq).
    x_size : int
        Total state vector size (n_theta + n_v).
    dV_dQ_reduced : NDArray[np.float64]
        Reduced Jacobian inverse (∂V/∂Q) computed via Schur complement.
    
    Notes
    -----
    The network must have a converged power flow solution before initialisation.
    The class creates a deep copy of the network to ensure the cached state
    is not modified by external operations.
    """
    
    def __init__(self, net: pp.pandapowerNet) -> None:
        """
        Initialise the sensitivity calculator from a converged network.
        
        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network with converged power flow solution.
        
        Raises
        ------
        ValueError
            If the network has not converged or lacks required internal data.
        """
        if not net.converged:
            raise ValueError("Network power flow must have converged.")
        
        if not hasattr(net, '_ppc') or 'internal' not in net._ppc:
            raise ValueError("Network must have internal power flow data (_ppc).")
        
        if 'J' not in net._ppc['internal']:
            raise ValueError("Network must have Jacobian matrix in internal data.")
        
        # Store deep copy of network state
        self.net = copy.deepcopy(net)
        
        # Extract Jacobian from power flow solution
        self.J = np.array(self.net._ppc['internal']['J'].todense())
        
        # Extract bus type information
        self.pq_buses = self.net._ppc['internal']['pq']
        self.pv_buses = self.net._ppc['internal']['pv']
        
        # Compute dimensions
        self.n_theta = len(self.pq_buses) + len(self.pv_buses)
        self.n_v = len(self.pq_buses)
        self.x_size = self.n_theta + self.n_v
        
        # Compute Jacobian inverse
        self.J_inv = self._compute_jacobian_inverse()
        
        # Compute reduced Jacobian (∂V/∂Q) via Schur complement
        self.dV_dQ_reduced = self._compute_reduced_jacobian_inverse()
    
    def _compute_jacobian_inverse(self) -> NDArray[np.float64]:
        """
        Compute the inverse of the full Jacobian matrix.
        
        Returns
        -------
        J_inv : NDArray[np.float64]
            Inverse of the Jacobian matrix.
        
        Raises
        ------
        ValueError
            If the Jacobian is singular and cannot be inverted.
        """
        try:
            return np.linalg.inv(self.J)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Jacobian matrix is singular and cannot be inverted: {e}")
    
    def _compute_reduced_jacobian_inverse(self) -> NDArray[np.float64]:
        """
        Compute the reduced Jacobian inverse using the Schur complement.
        
        This computes J_r^{-1} = (J_QV - J_Qθ J_Pθ^{-1} J_PV)^{-1} = ∂V/∂Q
        
        This is Equation (9) from the PSCC 2026 paper.
        
        Returns
        -------
        dV_dQ : NDArray[np.float64]
            Matrix of shape (n_pq, n_pq) representing ∂V/∂Q.
        
        Raises
        ------
        ValueError
            If the Schur complement is singular.
        """
        n_pq_pv = len(self.pq_buses) + len(self.pv_buses)
        
        # Extract Jacobian submatrices
        # J = [J_Pθ  J_PV]
        #     [J_Qθ  J_QV]
        J_Ptheta = self.J[:n_pq_pv, :n_pq_pv]
        J_PV = self.J[:n_pq_pv, n_pq_pv:]
        J_Qtheta = self.J[n_pq_pv:, :n_pq_pv]
        J_QV = self.J[n_pq_pv:, n_pq_pv:]
        
        if J_Ptheta.size == 0 or J_QV.size == 0:
            raise ValueError("Invalid Jacobian structure: empty submatrices.")
        
        try:
            J_Ptheta_inv = np.linalg.inv(J_Ptheta)
            schur_complement = J_QV - J_Qtheta @ J_Ptheta_inv @ J_PV
            dV_dQ = np.linalg.inv(schur_complement)
            return dV_dQ
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Schur complement is singular: {e}")
    
    # =========================================================================
    # A. Bus Voltage to DER Reactive Power Infeed Sensitivity (Eq. 9, 10 PSCC 2026)
    # =========================================================================
    
    def compute_dV_dQ_der(
        self,
        der_bus_indices: List[int],
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute bus voltage sensitivity to DER reactive power injection.
        
        This implements Equation (10) from the PSCC 2026 paper:
            ΔV = J_r^{-1} ΔQ
        
        Parameters
        ----------
        der_bus_indices : List[int]
            Pandapower bus indices where DERs are connected.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.
        
        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape (n_obs, n_der) where entry (i,j) is ∂V_i/∂Q_j.
            Units: [p.u. / Mvar] (depends on network base power).
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.
        der_bus_mapping : List[int]
            Ordered list of DER bus indices actually included.
        
        Raises
        ------
        ValueError
            If no valid buses are found in either list.
        """
        # Map observation buses to Jacobian row indices
        obs_jacobian_rows = []
        obs_bus_mapping = []
        
        for bus_idx in observation_bus_indices:
            _, v_idx = get_jacobian_indices(self.net, bus_idx)
            if v_idx is not None and v_idx < self.dV_dQ_reduced.shape[0]:
                obs_jacobian_rows.append(v_idx)
                obs_bus_mapping.append(bus_idx)
        
        if not obs_jacobian_rows:
            raise ValueError("No valid observation buses found (all may be PV or slack).")
        
        # Map DER buses to Jacobian column indices
        der_jacobian_cols = []
        der_bus_mapping = []
        
        for bus_idx in der_bus_indices:
            _, v_idx = get_jacobian_indices(self.net, bus_idx)
            if v_idx is not None and v_idx < self.dV_dQ_reduced.shape[1]:
                der_jacobian_cols.append(v_idx)
                der_bus_mapping.append(bus_idx)
        
        if not der_jacobian_cols:
            raise ValueError("No valid DER buses found (all may be PV or slack).")
        
        # Extract submatrix from reduced Jacobian
        sensitivity_matrix = self.dV_dQ_reduced[np.ix_(obs_jacobian_rows, der_jacobian_cols)]
        
        return sensitivity_matrix, obs_bus_mapping, der_bus_mapping
    
    # =========================================================================
    # B. Bus Voltage to OLTC Position Sensitivity (Eq. 11 PSCC 2026)
    # =========================================================================
    
    def compute_dV_ds(
        self,
        trafo_idx: int,
        observation_bus_indices: List[int],
        delta_s: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute bus voltage sensitivity to transformer tap position change.
        
        This implements Equation (11) from the PSCC 2026 paper:
            ΔV = Δτ_i [D_21  D_22] (∂g/∂τ_i) Δs_i
        
        where D = J^{-1} and D_21, D_22 are the lower submatrices.
        
        Parameters
        ----------
        trafo_idx : int
            Pandapower transformer index.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.
        delta_s : float, optional
            Tap step size (default: 1.0 tap step).
        
        Returns
        -------
        dV_ds : NDArray[np.float64]
            Sensitivity vector of shape (n_obs,) representing ∂V/∂s.
            Units: [p.u. per tap step].
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.
        
        Raises
        ------
        ValueError
            If transformer is not found or no valid observation buses exist.
        """
        if trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Transformer {trafo_idx} not found in network.")
        
        ppc_br_idx = get_ppc_trafo_index(self.net, trafo_idx)
        if ppc_br_idx is None:
            raise ValueError(f"Could not find pypower branch index for transformer {trafo_idx}.")
        
        # Get transformer buses
        hv_bus = self.net.trafo.at[trafo_idx, 'hv_bus']
        lv_bus = self.net.trafo.at[trafo_idx, 'lv_bus']
        
        # Get voltage and angle at transformer terminals
        U_i = self.net.res_bus.at[hv_bus, 'vm_pu']
        U_j = self.net.res_bus.at[lv_bus, 'vm_pu']
        theta_i = np.deg2rad(self.net.res_bus.at[hv_bus, 'va_degree'])
        theta_j = np.deg2rad(self.net.res_bus.at[lv_bus, 'va_degree'])
        
        # Get tap parameters
        s0 = self.net.trafo.at[trafo_idx, 'tap_pos']
        delta_tau = self.net.trafo.at[trafo_idx, 'tap_step_percent'] / 100.0
        tau = 1.0 + s0 * delta_tau
        
        # Get transformer impedance from pypower data
        r_pu = self.net._ppc['branch'][ppc_br_idx, 2]
        x_pu = self.net._ppc['branch'][ppc_br_idx, 3]
        y_pu = 1.0 / complex(r_pu, x_pu)
        g = y_pu.real
        b = y_pu.imag
        
        # Get Jacobian indices for transformer buses
        theta_i_idx, v_i_idx = get_jacobian_indices(self.net, hv_bus)
        theta_j_idx, v_j_idx = get_jacobian_indices(self.net, lv_bus)
        
        if theta_i_idx is None or theta_j_idx is None:
            raise ValueError("Could not find Jacobian indices for transformer buses.")
        
        # Compute ∂g/∂τ (power injection derivatives with respect to tap ratio)
        # These are the derivatives of the power balance equations
        dg_dtau = np.zeros(self.x_size)
        
        # Active power derivatives at HV and LV buses
        dP_dtau_i = (-2.0 * g * U_i**2 / tau**3 + 
                     U_i * U_j * (g * np.cos(theta_i - theta_j) + b * np.sin(theta_i - theta_j)) / tau**2)
        dP_dtau_j = (U_i * U_j * (g * np.cos(theta_i - theta_j) + b * np.sin(theta_i - theta_j)) / tau**2)
        
        # Reactive power derivatives at HV and LV buses
        dQ_dtau_i = (2.0 * b * U_i**2 / tau**3 + 
                     U_i * U_j * (g * np.sin(theta_i - theta_j) - b * np.cos(theta_i - theta_j)) / tau**2)
        dQ_dtau_j = (U_i * U_j * (g * np.sin(theta_i - theta_j) - b * np.cos(theta_i - theta_j)) / tau**2)
        
        # Accumulate derivatives into dg_dtau vector
        if theta_i_idx is not None:
            dg_dtau[theta_i_idx] += dP_dtau_i
        if theta_j_idx is not None:
            dg_dtau[theta_j_idx] += dP_dtau_j
        if v_i_idx is not None:
            dg_dtau[self.n_theta + v_i_idx] += dQ_dtau_i
        if v_j_idx is not None and (self.n_theta + v_j_idx) < self.x_size:
            dg_dtau[self.n_theta + v_j_idx] += dQ_dtau_j
        
        # Compute full state sensitivity: dx/ds = -J^{-1} (∂g/∂τ) Δτ
        # Note: The minus sign comes from implicit function theorem
        dx_ds = -self.J_inv @ dg_dtau * delta_tau
        
        # Extract voltage magnitude sensitivities (lower half of state vector)
        dV_ds_full = dx_ds[self.n_theta:]
        
        # Map observation buses to output
        obs_jacobian_rows = []
        obs_bus_mapping = []
        
        for bus_idx in observation_bus_indices:
            _, v_idx = get_jacobian_indices(self.net, bus_idx)
            if v_idx is not None and v_idx < len(dV_ds_full):
                obs_jacobian_rows.append(v_idx)
                obs_bus_mapping.append(bus_idx)
        
        if not obs_jacobian_rows:
            raise ValueError("No valid observation buses found.")
        
        dV_ds = dV_ds_full[obs_jacobian_rows]
        
        return dV_ds, obs_bus_mapping
    
    def compute_dV_ds_matrix(
        self,
        trafo_indices: List[int],
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute voltage sensitivity matrix for multiple transformers.
        
        Parameters
        ----------
        trafo_indices : List[int]
            Pandapower transformer indices.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.
        
        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape (n_obs, n_trafo) where entry (i,j) is ∂V_i/∂s_j.
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.
        trafo_mapping : List[int]
            Ordered list of transformer indices actually included.
        """
        n_obs = len(observation_bus_indices)
        n_trafo = len(trafo_indices)
        
        if n_obs == 0 or n_trafo == 0:
            return np.zeros((n_obs, n_trafo)), [], []
        
        sensitivity_matrix = np.zeros((n_obs, n_trafo))
        trafo_mapping = []
        obs_bus_mapping = None
        
        for j, trafo_idx in enumerate(trafo_indices):
            try:
                dV_ds_col, obs_mapping = self.compute_dV_ds(
                    trafo_idx=trafo_idx,
                    observation_bus_indices=observation_bus_indices,
                    delta_s=1.0,
                )
                sensitivity_matrix[:len(dV_ds_col), j] = dV_ds_col
                trafo_mapping.append(trafo_idx)
                if obs_bus_mapping is None:
                    obs_bus_mapping = obs_mapping
            except ValueError:
                # Skip invalid transformers
                continue
        
        if obs_bus_mapping is None:
            obs_bus_mapping = []
        
        return sensitivity_matrix[:len(obs_bus_mapping), :len(trafo_mapping)], obs_bus_mapping, trafo_mapping
    
    # =========================================================================
    # C. Transformer Q to DER Reactive Power Infeed Sensitivity (Eq. 12-14 PSCC 2026)
    # =========================================================================
    
    def compute_dQtrafo_dQ_der(
        self,
        trafo_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute transformer reactive power flow sensitivity to DER reactive power.
        
        This implements Equation (14) from the PSCC 2026 paper:
            ΔQ_ij = (∂Q_ij/∂V_i · ∂V_i/∂Q_n + ∂Q_ij/∂V_j · ∂V_j/∂Q_n) ΔQ_n
        
        Parameters
        ----------
        trafo_idx : int
            Pandapower transformer index (measurement location).
        der_bus_idx : int
            Pandapower bus index where the DER is connected.
        
        Returns
        -------
        sensitivity : float
            Sensitivity ∂Q_trafo/∂Q_DER in [Mvar/Mvar] (dimensionless).
        
        Raises
        ------
        ValueError
            If transformer or DER bus is not found or cannot be processed.
        """
        if trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Transformer {trafo_idx} not found in network.")
        
        ppc_br_idx = get_ppc_trafo_index(self.net, trafo_idx)
        if ppc_br_idx is None:
            raise ValueError(f"Could not find pypower branch index for transformer {trafo_idx}.")
        
        # Get transformer buses
        hv_bus = self.net.trafo.at[trafo_idx, 'hv_bus']
        lv_bus = self.net.trafo.at[trafo_idx, 'lv_bus']
        
        # Get voltage states
        U_i = self.net.res_bus.at[hv_bus, 'vm_pu']
        U_j = self.net.res_bus.at[lv_bus, 'vm_pu']
        theta_i = np.deg2rad(self.net.res_bus.at[hv_bus, 'va_degree'])
        theta_j = np.deg2rad(self.net.res_bus.at[lv_bus, 'va_degree'])
        
        # Get transformer parameters
        r_pu = self.net._ppc['branch'][ppc_br_idx, 2]
        x_pu = self.net._ppc['branch'][ppc_br_idx, 3]
        y_pu = 1.0 / complex(r_pu, x_pu)
        g = y_pu.real
        b = y_pu.imag
        
        # Get tap ratio
        delta_tau = self.net.trafo.at[trafo_idx, 'tap_step_percent'] / 100.0
        tau = 1.0 + self.net.trafo.at[trafo_idx, 'tap_pos'] * delta_tau
        
        # Get Jacobian indices
        _, v_i_idx = get_jacobian_indices(self.net, hv_bus)
        _, v_j_idx = get_jacobian_indices(self.net, lv_bus)
        _, v_der_idx = get_jacobian_indices(self.net, der_bus_idx)
        
        if v_i_idx is None or v_j_idx is None or v_der_idx is None:
            raise ValueError("Could not find Jacobian indices for buses.")
        
        # Compute ∂Q_ij/∂V_i and ∂Q_ij/∂V_j (Equation 12)
        # Q_ij = -b * U_i^2 / τ^2 + U_i * U_j * (b * cos(θ_i - θ_j) - g * sin(θ_i - θ_j)) / τ
        dQ_ij_dV_i = (b * (-2.0 * U_i / tau**2 + U_j * np.cos(theta_i - theta_j) / tau) - 
                      g * U_j * np.sin(theta_i - theta_j) / tau)
        dQ_ij_dV_j = (b * U_i * np.cos(theta_i - theta_j) / tau - 
                      g * U_i * np.sin(theta_i - theta_j) / tau)
        
        # Get voltage sensitivities from reduced Jacobian (Equation 13)
        if v_i_idx >= self.dV_dQ_reduced.shape[0] or v_der_idx >= self.dV_dQ_reduced.shape[1]:
            raise ValueError("Jacobian index out of bounds.")
        if v_j_idx >= self.dV_dQ_reduced.shape[0]:
            raise ValueError("Jacobian index out of bounds.")
        
        dV_i_dQ_n = self.dV_dQ_reduced[v_i_idx, v_der_idx]
        dV_j_dQ_n = self.dV_dQ_reduced[v_j_idx, v_der_idx]
        
        # Apply chain rule (Equation 14)
        sensitivity = dQ_ij_dV_i * dV_i_dQ_n + dQ_ij_dV_j * dV_j_dQ_n
        
        return sensitivity
    
    def compute_dQtrafo_dQ_der_matrix(
        self,
        trafo_indices: List[int],
        der_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute transformer Q sensitivity matrix to DER reactive power.
        
        Parameters
        ----------
        trafo_indices : List[int]
            Pandapower transformer indices.
        der_bus_indices : List[int]
            Pandapower bus indices where DERs are connected.
        
        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape (n_trafo, n_der) where entry (i,j) is ∂Q_i/∂Q_j.
        trafo_mapping : List[int]
            Ordered list of transformer indices actually included.
        der_bus_mapping : List[int]
            Ordered list of DER bus indices actually included.
        """
        n_trafo = len(trafo_indices)
        n_der = len(der_bus_indices)
        
        if n_trafo == 0 or n_der == 0:
            return np.zeros((n_trafo, n_der)), [], []
        
        sensitivity_matrix = np.zeros((n_trafo, n_der))
        trafo_mapping = []
        der_bus_mapping = []
        
        for j, der_bus_idx in enumerate(der_bus_indices):
            try:
                _, v_der_idx = get_jacobian_indices(self.net, der_bus_idx)
                if v_der_idx is None:
                    continue
                if j == 0 or der_bus_idx not in der_bus_mapping:
                    der_bus_mapping.append(der_bus_idx)
            except ValueError:
                continue
        
        for i, trafo_idx in enumerate(trafo_indices):
            try:
                trafo_mapping.append(trafo_idx)
                for j, der_bus_idx in enumerate(der_bus_indices):
                    if der_bus_idx not in der_bus_mapping:
                        continue
                    j_mapped = der_bus_mapping.index(der_bus_idx)
                    sensitivity_matrix[i, j_mapped] = self.compute_dQtrafo_dQ_der(trafo_idx, der_bus_idx)
            except ValueError:
                continue
        
        return sensitivity_matrix[:len(trafo_mapping), :len(der_bus_mapping)], trafo_mapping, der_bus_mapping
    
    # =========================================================================
    # D. Transformer Q to OLTC Position Sensitivity (Eq. 15-17)
    # =========================================================================
    
    def compute_dQtrafo_ds(
        self,
        meas_trafo_idx: int,
        chg_trafo_idx: int,
        delta_s: float = 1.0,
    ) -> float:
        """
        Compute transformer reactive power flow sensitivity to tap position change.
        
        This implements Equation (17) from the PSCC 2026 paper:
            ΔQ_ij = Δτ_i (∂Q_ij/∂x J^{-1} ∂g/∂τ_i + ∂Q_ij/∂τ_i) Δs_i
        
        The sensitivity has two components:
            A: Indirect effect via system-wide voltage/angle changes
            B: Direct effect from tap change on the measurement transformer
        
        Parameters
        ----------
        meas_trafo_idx : int
            Pandapower transformer index where Q is measured.
        chg_trafo_idx : int
            Pandapower transformer index where tap is changed.
        delta_s : float, optional
            Tap step size (default: 1.0 tap step).
        
        Returns
        -------
        sensitivity : float
            Sensitivity ∂Q_meas/∂s_chg in [Mvar per tap step].
        
        Raises
        ------
        ValueError
            If transformers are not found or cannot be processed.
        """
        if meas_trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Measurement transformer {meas_trafo_idx} not found.")
        if chg_trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Change transformer {chg_trafo_idx} not found.")
        
        # === Part 1: Compute ∂g/∂τ for the change transformer ===
        ppc_br_idx_chg = get_ppc_trafo_index(self.net, chg_trafo_idx)
        if ppc_br_idx_chg is None:
            raise ValueError(f"Could not find pypower index for transformer {chg_trafo_idx}.")
        
        hv_bus_chg = self.net.trafo.at[chg_trafo_idx, 'hv_bus']
        lv_bus_chg = self.net.trafo.at[chg_trafo_idx, 'lv_bus']
        
        U_i_chg = self.net.res_bus.at[hv_bus_chg, 'vm_pu']
        U_j_chg = self.net.res_bus.at[lv_bus_chg, 'vm_pu']
        theta_i_chg = np.deg2rad(self.net.res_bus.at[hv_bus_chg, 'va_degree'])
        theta_j_chg = np.deg2rad(self.net.res_bus.at[lv_bus_chg, 'va_degree'])
        
        s0_chg = self.net.trafo.at[chg_trafo_idx, 'tap_pos']
        delta_tau_chg = self.net.trafo.at[chg_trafo_idx, 'tap_step_percent'] / 100.0
        tau_chg = 1.0 + s0_chg * delta_tau_chg
        
        r_pu_chg = self.net._ppc['branch'][ppc_br_idx_chg, 2]
        x_pu_chg = self.net._ppc['branch'][ppc_br_idx_chg, 3]
        y_pu_chg = 1.0 / complex(r_pu_chg, x_pu_chg)
        g_chg = y_pu_chg.real
        b_chg = y_pu_chg.imag
        
        theta_i_idx_chg, v_i_idx_chg = get_jacobian_indices(self.net, hv_bus_chg)
        theta_j_idx_chg, v_j_idx_chg = get_jacobian_indices(self.net, lv_bus_chg)
        
        if theta_i_idx_chg is None or theta_j_idx_chg is None:
            raise ValueError("Could not find Jacobian indices for change transformer buses.")
        
        # Compute ∂g/∂τ
        dg_dtau = np.zeros(self.x_size)
        
        dP_dtau_i = (-2.0 * g_chg * U_i_chg**2 / tau_chg**3 + 
                     U_i_chg * U_j_chg * (g_chg * np.cos(theta_i_chg - theta_j_chg) + 
                                          b_chg * np.sin(theta_i_chg - theta_j_chg)) / tau_chg**2)
        dP_dtau_j = (U_i_chg * U_j_chg * (g_chg * np.cos(theta_i_chg - theta_j_chg) + 
                                           b_chg * np.sin(theta_i_chg - theta_j_chg)) / tau_chg**2)
        dQ_dtau_i = (2.0 * b_chg * U_i_chg**2 / tau_chg**3 + 
                     U_i_chg * U_j_chg * (g_chg * np.sin(theta_i_chg - theta_j_chg) - 
                                          b_chg * np.cos(theta_i_chg - theta_j_chg)) / tau_chg**2)
        dQ_dtau_j = (U_i_chg * U_j_chg * (g_chg * np.sin(theta_i_chg - theta_j_chg) - 
                                           b_chg * np.cos(theta_i_chg - theta_j_chg)) / tau_chg**2)
        
        if theta_i_idx_chg is not None:
            dg_dtau[theta_i_idx_chg] += dP_dtau_i
        if theta_j_idx_chg is not None:
            dg_dtau[theta_j_idx_chg] += dP_dtau_j
        if v_i_idx_chg is not None:
            dg_dtau[self.n_theta + v_i_idx_chg] += dQ_dtau_i
        if v_j_idx_chg is not None and (self.n_theta + v_j_idx_chg) < self.x_size:
            dg_dtau[self.n_theta + v_j_idx_chg] += dQ_dtau_j
        
        # === Part 2: Compute ∂Q_ij/∂x for the measurement transformer ===
        ppc_br_idx_meas = get_ppc_trafo_index(self.net, meas_trafo_idx)
        if ppc_br_idx_meas is None:
            raise ValueError(f"Could not find pypower index for transformer {meas_trafo_idx}.")
        
        hv_bus_meas = self.net.trafo.at[meas_trafo_idx, 'hv_bus']
        lv_bus_meas = self.net.trafo.at[meas_trafo_idx, 'lv_bus']
        
        U_i_meas = self.net.res_bus.at[hv_bus_meas, 'vm_pu']
        U_j_meas = self.net.res_bus.at[lv_bus_meas, 'vm_pu']
        theta_i_meas = np.deg2rad(self.net.res_bus.at[hv_bus_meas, 'va_degree'])
        theta_j_meas = np.deg2rad(self.net.res_bus.at[lv_bus_meas, 'va_degree'])
        
        s0_meas = self.net.trafo.at[meas_trafo_idx, 'tap_pos']
        delta_tau_meas = self.net.trafo.at[meas_trafo_idx, 'tap_step_percent'] / 100.0
        tau_meas = 1.0 + s0_meas * delta_tau_meas
        
        r_pu_meas = self.net._ppc['branch'][ppc_br_idx_meas, 2]
        x_pu_meas = self.net._ppc['branch'][ppc_br_idx_meas, 3]
        y_pu_meas = 1.0 / complex(r_pu_meas, x_pu_meas)
        g_meas = y_pu_meas.real
        b_meas = y_pu_meas.imag
        
        theta_i_idx_meas, v_i_idx_meas = get_jacobian_indices(self.net, hv_bus_meas)
        theta_j_idx_meas, v_j_idx_meas = get_jacobian_indices(self.net, lv_bus_meas)
        
        if theta_i_idx_meas is None or theta_j_idx_meas is None:
            raise ValueError("Could not find Jacobian indices for measurement transformer buses.")
        
        # Compute ∂Q_ij/∂x
        dQ_dx = np.zeros(self.x_size)
        
        dQ_dtheta_i = (-b_meas * U_i_meas * U_j_meas * np.sin(theta_i_meas - theta_j_meas) / tau_meas - 
                       g_meas * U_i_meas * U_j_meas * np.cos(theta_i_meas - theta_j_meas) / tau_meas)
        dQ_dtheta_j = (b_meas * U_i_meas * U_j_meas * np.sin(theta_i_meas - theta_j_meas) / tau_meas + 
                       g_meas * U_i_meas * U_j_meas * np.cos(theta_i_meas - theta_j_meas) / tau_meas)
        dQ_dU_i = (b_meas * (-2.0 * U_i_meas / tau_meas**2 + U_j_meas * np.cos(theta_i_meas - theta_j_meas) / tau_meas) - 
                   g_meas * U_j_meas * np.sin(theta_i_meas - theta_j_meas) / tau_meas)
        dQ_dU_j = (b_meas * U_i_meas * np.cos(theta_i_meas - theta_j_meas) / tau_meas - 
                   g_meas * U_i_meas * np.sin(theta_i_meas - theta_j_meas) / tau_meas)
        
        if theta_i_idx_meas is not None:
            dQ_dx[theta_i_idx_meas] += dQ_dtheta_i
        if theta_j_idx_meas is not None:
            dQ_dx[theta_j_idx_meas] += dQ_dtheta_j
        if v_i_idx_meas is not None:
            dQ_dx[self.n_theta + v_i_idx_meas] += dQ_dU_i
        if v_j_idx_meas is not None and (self.n_theta + v_j_idx_meas) < self.x_size:
            dQ_dx[self.n_theta + v_j_idx_meas] += dQ_dU_j
        
        # === Part 3: Compute indirect effect (A) ===
        # A = ∂Q_ij/∂x · J^{-1} · ∂g/∂τ
        indirect_effect = -dQ_dx @ self.J_inv @ dg_dtau * delta_tau_chg

        # === Part 4: Compute direct effect (B) ===
        # Only applies if the measurement transformer is the same as change transformer
        direct_effect = 0.0
        if meas_trafo_idx == chg_trafo_idx:
            # ∂Q_ij/∂τ (direct derivative of Q flow with respect to tap ratio)
            dQ_dtau_direct = (2.0 * b_meas * U_i_meas**2 / tau_meas**3 -
                              b_meas * U_i_meas * U_j_meas * np.cos(theta_i_meas - theta_j_meas) / tau_meas**2 +
                              g_meas * U_i_meas * U_j_meas * np.sin(theta_i_meas - theta_j_meas) / tau_meas**2)
            direct_effect = dQ_dtau_direct * delta_tau_meas

        # Total sensitivity (Equation 17), scaled from per-unit to Mvar
        sensitivity = (indirect_effect + direct_effect) * self.net.sn_mva

        return sensitivity
    
    def compute_dQtrafo_ds_matrix(
        self,
        trafo_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute transformer Q sensitivity matrix to tap positions.
        
        Parameters
        ----------
        trafo_indices : List[int]
            Pandapower transformer indices.
        
        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape (n_trafo, n_trafo) where entry (i,j) is ∂Q_i/∂s_j.
        trafo_mapping : List[int]
            Ordered list of transformer indices actually included.
        """
        n_trafo = len(trafo_indices)
        
        if n_trafo == 0:
            return np.zeros((0, 0)), []
        
        sensitivity_matrix = np.zeros((n_trafo, n_trafo))
        trafo_mapping = list(trafo_indices)
        
        for i, meas_idx in enumerate(trafo_indices):
            for j, chg_idx in enumerate(trafo_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dQtrafo_ds(meas_idx, chg_idx, delta_s=1.0)
                except ValueError:
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, trafo_mapping
    
    # =========================================================================
    # E. Branch Current to DER Reactive Power Infeed Sensitivity (Eq. 18-20)
    # =========================================================================
    
    def compute_dI_dQ_der(
        self,
        line_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute branch current magnitude sensitivity to DER reactive power.
        
        This implements Equations (19) and (20) from the PSCC 2026 paper:
            ∂I_ij/∂Q_n = Y_ij (∂V_i/∂Q_n - ∂V_j/∂Q_n)
            ∂|I_ij|/∂Q_n = (1/|I_ij|) Re{I_ij* · ∂I_ij/∂Q_n}
        
        Parameters
        ----------
        line_idx : int
            Pandapower line index.
        der_bus_idx : int
            Pandapower bus index where the DER is connected.
        
        Returns
        -------
        sensitivity : float
            Sensitivity ∂|I|/∂Q in [kA/Mvar] (or per-unit equivalent).
        
        Raises
        ------
        ValueError
            If line or DER bus is not found or cannot be processed.
        """
        if line_idx not in self.net.line.index:
            raise ValueError(f"Line {line_idx} not found in network.")
        
        # Get line buses
        from_bus = self.net.line.at[line_idx, 'from_bus']
        to_bus = self.net.line.at[line_idx, 'to_bus']
        
        # Get voltage states
        U_from = self.net.res_bus.at[from_bus, 'vm_pu']
        U_to = self.net.res_bus.at[to_bus, 'vm_pu']
        theta_from = np.deg2rad(self.net.res_bus.at[from_bus, 'va_degree'])
        theta_to = np.deg2rad(self.net.res_bus.at[to_bus, 'va_degree'])
        
        # Compute complex voltages
        V_from = U_from * np.exp(1j * theta_from)
        V_to = U_to * np.exp(1j * theta_to)
        
        # Get line impedance in per-unit
        base_voltage_kv = self.net.bus.at[from_bus, 'vn_kv']
        base_impedance = base_voltage_kv**2 / self.net.sn_mva
        
        r_ohm = self.net.line.at[line_idx, 'r_ohm_per_km'] * self.net.line.at[line_idx, 'length_km']
        x_ohm = self.net.line.at[line_idx, 'x_ohm_per_km'] * self.net.line.at[line_idx, 'length_km']
        
        r_pu = r_ohm / base_impedance
        x_pu = x_ohm / base_impedance
        
        Y_line = 1.0 / complex(r_pu, x_pu)
        
        # Compute current phasor (Equation 18)
        I_ij = Y_line * (V_from - V_to)
        I_mag = np.abs(I_ij)
        
        if I_mag < 1e-10:
            # Avoid division by zero for very small currents
            return 0.0
        
        # Get Jacobian indices
        _, v_from_idx = get_jacobian_indices(self.net, from_bus)
        _, v_to_idx = get_jacobian_indices(self.net, to_bus)
        _, v_der_idx = get_jacobian_indices(self.net, der_bus_idx)
        
        if v_from_idx is None or v_to_idx is None or v_der_idx is None:
            raise ValueError("Could not find Jacobian indices for buses.")
        
        # Get voltage sensitivities
        if (v_from_idx >= self.dV_dQ_reduced.shape[0] or 
            v_to_idx >= self.dV_dQ_reduced.shape[0] or 
            v_der_idx >= self.dV_dQ_reduced.shape[1]):
            raise ValueError("Jacobian index out of bounds.")
        
        dV_from_dQ = self.dV_dQ_reduced[v_from_idx, v_der_idx]
        dV_to_dQ = self.dV_dQ_reduced[v_to_idx, v_der_idx]
        
        # Compute current phasor derivative (Equation 19)
        # Simplified: assuming small angle changes, dV/dQ primarily affects magnitude
        dI_ij_dQ = Y_line * (dV_from_dQ * np.exp(1j * theta_from) - 
                             dV_to_dQ * np.exp(1j * theta_to))
        
        # Compute magnitude derivative (Equation 20)
        sensitivity = (1.0 / I_mag) * np.real(np.conj(I_ij) * dI_ij_dQ)
        
        return sensitivity
    
    def compute_dI_dQ_der_matrix(
        self,
        line_indices: List[int],
        der_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute branch current sensitivity matrix to DER reactive power.
        
        Parameters
        ----------
        line_indices : List[int]
            Pandapower line indices.
        der_bus_indices : List[int]
            Pandapower bus indices where DERs are connected.
        
        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape (n_line, n_der) where entry (i,j) is ∂|I_i|/∂Q_j.
        line_mapping : List[int]
            Ordered list of line indices actually included.
        der_bus_mapping : List[int]
            Ordered list of DER bus indices actually included.
        """
        n_lines = len(line_indices)
        n_der = len(der_bus_indices)
        
        if n_lines == 0 or n_der == 0:
            return np.zeros((n_lines, n_der)), [], []
        
        sensitivity_matrix = np.zeros((n_lines, n_der))
        line_mapping = list(line_indices)
        der_bus_mapping = list(der_bus_indices)
        
        for i, line_idx in enumerate(line_indices):
            for j, der_bus_idx in enumerate(der_bus_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dI_dQ_der(line_idx, der_bus_idx)
                except ValueError:
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, line_mapping, der_bus_mapping
    
    # =========================================================================
    # F. Shunt Reactive Power Sensitivity (for switchable shunts)
    # =========================================================================

    def compute_dV_dQ_shunt(
        self,
        shunt_bus_idx: int,
        observation_bus_indices: List[int],
        q_step_mvar: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute voltage sensitivity to shunt reactive power injection.

        Shunts are modelled as discrete Q injections. This computes dV/dQ_shunt
        which can be multiplied by the shunt step size to get dV per state change.

        Parameters
        ----------
        shunt_bus_idx : int
            Pandapower bus index where the shunt is connected.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.
        q_step_mvar : float, optional
            Reactive power step per shunt state change (default: 1.0 Mvar).

        Returns
        -------
        dV_dQ_shunt : NDArray[np.float64]
            Sensitivity vector of shape (n_obs,) representing dV/dQ_shunt.
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.

        Raises
        ------
        ValueError
            If shunt bus is not valid or no observation buses are found.
        """
        # Shunt sensitivity delegates to DER Q sensitivity at the same bus,
        # but with a sign flip: pandapower shunts use load convention
        # (positive q_mvar = absorbing), whereas the Jacobian uses generator
        # convention (positive Q = injecting).  Therefore dQ_inj = -dQ_shunt.
        dV_dQ, obs_map, der_map = self.compute_dV_dQ_der(
            der_bus_indices=[shunt_bus_idx],
            observation_bus_indices=observation_bus_indices,
        )

        if len(der_map) == 0:
            raise ValueError(f"Shunt bus {shunt_bus_idx} is not valid for Q injection.")

        # Negate to account for load-convention sign, then scale by step size
        dV_dQ_shunt = -dV_dQ[:, 0] * q_step_mvar

        return dV_dQ_shunt, obs_map

    def compute_dI_dQ_shunt(
        self,
        shunt_bus_idx: int,
        line_indices: List[int],
        q_step_mvar: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute branch current sensitivity to shunt reactive power injection.

        Parameters
        ----------
        shunt_bus_idx : int
            Pandapower bus index where the shunt is connected.
        line_indices : List[int]
            Pandapower line indices for current measurements.
        q_step_mvar : float, optional
            Reactive power step per shunt state change (default: 1.0 Mvar).

        Returns
        -------
        dI_dQ_shunt : NDArray[np.float64]
            Sensitivity vector of shape (n_lines,) representing d|I|/dQ_shunt.
        line_mapping : List[int]
            Ordered list of line indices actually included.
        """
        dI_dQ, line_map, der_map = self.compute_dI_dQ_der_matrix(
            line_indices=line_indices,
            der_bus_indices=[shunt_bus_idx],
        )

        if len(der_map) == 0:
            raise ValueError(f"Shunt bus {shunt_bus_idx} is not valid for Q injection.")

        # Negate to account for load-convention sign, then scale by step size
        dI_dQ_shunt = -dI_dQ[:, 0] * q_step_mvar

        return dI_dQ_shunt, line_map

    # =========================================================================
    # G. Three-Winding Transformer Sensitivities
    # =========================================================================

    def _get_trafo3w_hv_branch_data(
        self,
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
            self.net, trafo3w_idx
        )

        # Pandapower and _ppc bus indices
        hv_bus_pp = int(self.net.trafo3w.at[trafo3w_idx, 'hv_bus'])
        hv_bus_ppc = pp_bus_to_ppc_bus(self.net, hv_bus_pp)

        # Voltage states from _ppc bus results
        U_hv = float(self.net._ppc['bus'][hv_bus_ppc, 7])     # VM column
        U_aux = float(self.net._ppc['bus'][aux_bus_ppc, 7])
        theta_hv = float(np.deg2rad(self.net._ppc['bus'][hv_bus_ppc, 8]))   # VA column
        theta_aux = float(np.deg2rad(self.net._ppc['bus'][aux_bus_ppc, 8]))

        # Tap ratio from _ppc branch data
        tap_ppc = float(self.net._ppc['branch'][hv_br, 8])
        # Pandapower tap parameters
        s0 = float(self.net.trafo3w.at[trafo3w_idx, 'tap_pos'])
        delta_tau = float(
            self.net.trafo3w.at[trafo3w_idx, 'tap_step_percent']
        ) / 100.0
        # Effective tap ratio: pandapower stores the final tap in _ppc.
        # For sensitivity, we need delta_tau for the derivative d(tau)/ds.
        tau = tap_ppc if abs(tap_ppc) > 1e-12 else (1.0 + s0 * delta_tau)

        # Branch impedance
        r_pu = float(self.net._ppc['branch'][hv_br, 2])
        x_pu = float(self.net._ppc['branch'][hv_br, 3])
        y_pu = 1.0 / complex(r_pu, x_pu)
        g = y_pu.real
        b = y_pu.imag

        # Jacobian indices (using _ppc bus indices)
        theta_hv_jac, v_hv_jac = get_jacobian_indices_ppc(
            self.net, hv_bus_ppc
        )
        theta_aux_jac, v_aux_jac = get_jacobian_indices_ppc(
            self.net, aux_bus_ppc
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

    def _compute_dg_dtau_3w(
        self,
        d: Dict,
    ) -> NDArray[np.float64]:
        """
        Compute the partial derivative dg/dtau for the HV winding
        of a 3-winding transformer.

        This is the sensitivity of the power-balance equations g(x) = 0
        with respect to the effective tap ratio tau of the HV-star branch.
        The structure is identical to the 2-winding case (Eq. 11, PSCC 2026)
        but applied to the HV bus and the auxiliary star-point bus.

        Parameters
        ----------
        d : dict
            Data dictionary from :meth:`_get_trafo3w_hv_branch_data`.

        Returns
        -------
        dg_dtau : NDArray[np.float64]
            Vector of length ``x_size`` (state-vector dimension).
        """
        U_i = d['U_hv']
        U_j = d['U_aux']
        theta_i = d['theta_hv']
        theta_j = d['theta_aux']
        tau = d['tau']
        g = d['g']
        b = d['b']

        dg_dtau = np.zeros(self.x_size)

        # Active power derivatives at HV bus (terminal i) and star bus (terminal j)
        dP_dtau_i = (
            -2.0 * g * U_i**2 / tau**3
            + U_i * U_j * (
                g * np.cos(theta_i - theta_j)
                + b * np.sin(theta_i - theta_j)
            ) / tau**2
        )
        dP_dtau_j = (
            U_i * U_j * (
                g * np.cos(theta_i - theta_j)
                + b * np.sin(theta_i - theta_j)
            ) / tau**2
        )

        # Reactive power derivatives
        dQ_dtau_i = (
            2.0 * b * U_i**2 / tau**3
            + U_i * U_j * (
                g * np.sin(theta_i - theta_j)
                - b * np.cos(theta_i - theta_j)
            ) / tau**2
        )
        dQ_dtau_j = (
            U_i * U_j * (
                g * np.sin(theta_i - theta_j)
                - b * np.cos(theta_i - theta_j)
            ) / tau**2
        )

        # Place into the state-vector-sized dg/dtau
        if d['theta_hv_jac'] is not None:
            dg_dtau[d['theta_hv_jac']] += dP_dtau_i
        if d['theta_aux_jac'] is not None:
            dg_dtau[d['theta_aux_jac']] += dP_dtau_j
        if d['v_hv_jac'] is not None:
            dg_dtau[self.n_theta + d['v_hv_jac']] += dQ_dtau_i
        if d['v_aux_jac'] is not None:
            idx_v = self.n_theta + d['v_aux_jac']
            if idx_v < self.x_size:
                dg_dtau[idx_v] += dQ_dtau_j

        return dg_dtau

    def _compute_dQhv_dx_3w(
        self,
        d: Dict,
    ) -> NDArray[np.float64]:
        """
        Compute partial derivative of Q_{HV} with respect to the state vector.

        Q_{HV} is the reactive power flow at the HV side of the HV-star branch,
        i.e. the reactive power leaving the HV bus towards the star point.
        This is the quantity measured at the primary side of the 3W transformer.

        Uses Eq. (5) from the PSCC 2026 paper applied to the HV-star branch.

        Parameters
        ----------
        d : dict
            Data dictionary from :meth:`_get_trafo3w_hv_branch_data`.

        Returns
        -------
        dQ_dx : NDArray[np.float64]
            Vector of length ``x_size``.
        """
        U_i = d['U_hv']
        U_j = d['U_aux']
        theta_i = d['theta_hv']
        theta_j = d['theta_aux']
        tau = d['tau']
        g_val = d['g']
        b_val = d['b']

        dQ_dx = np.zeros(self.x_size)

        # ∂Q_ij/∂θ_i
        dQ_dtheta_i = (
            -b_val * U_i * U_j * np.sin(theta_i - theta_j) / tau
            - g_val * U_i * U_j * np.cos(theta_i - theta_j) / tau
        )
        # ∂Q_ij/∂θ_j
        dQ_dtheta_j = (
            b_val * U_i * U_j * np.sin(theta_i - theta_j) / tau
            + g_val * U_i * U_j * np.cos(theta_i - theta_j) / tau
        )
        # ∂Q_ij/∂V_i (HV bus)
        dQ_dU_i = (
            b_val * (-2.0 * U_i / tau**2
                     + U_j * np.cos(theta_i - theta_j) / tau)
            - g_val * U_j * np.sin(theta_i - theta_j) / tau
        )
        # ∂Q_ij/∂V_j (star bus)
        dQ_dU_j = (
            b_val * U_i * np.cos(theta_i - theta_j) / tau
            - g_val * U_i * np.sin(theta_i - theta_j) / tau
        )

        if d['theta_hv_jac'] is not None:
            dQ_dx[d['theta_hv_jac']] += dQ_dtheta_i
        if d['theta_aux_jac'] is not None:
            dQ_dx[d['theta_aux_jac']] += dQ_dtheta_j
        if d['v_hv_jac'] is not None:
            dQ_dx[self.n_theta + d['v_hv_jac']] += dQ_dU_i
        if d['v_aux_jac'] is not None:
            idx_v = self.n_theta + d['v_aux_jac']
            if idx_v < self.x_size:
                dQ_dx[idx_v] += dQ_dU_j

        return dQ_dx

    def compute_dV_ds_trafo3w(
        self,
        trafo3w_idx: int,
        observation_bus_indices: List[int],
        delta_s: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute bus voltage sensitivity to a 3W transformer OLTC tap change.

        Implements Eq. (11) from the PSCC 2026 paper applied to the HV winding
        of a 3-winding transformer:

            ΔV = Δτ · [D₂₁  D₂₂] · (∂g/∂τ) · Δs

        where the OLTC is on the HV winding and D = J⁻¹.

        Parameters
        ----------
        trafo3w_idx : int
            Pandapower ``net.trafo3w`` index.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.
        delta_s : float, optional
            Tap step size (default: 1.0 tap step).

        Returns
        -------
        dV_ds : NDArray[np.float64]
            Sensitivity vector of shape ``(n_obs,)`` in [p.u. per tap step].
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.

        Raises
        ------
        ValueError
            If the transformer is not found or observation buses are invalid.
        """
        d = self._get_trafo3w_hv_branch_data(trafo3w_idx)
        dg_dtau = self._compute_dg_dtau_3w(d)

        # Full state sensitivity:  Δx = -J⁻¹ · (∂g/∂τ) · Δτ
        dx_ds = -self.J_inv @ dg_dtau * d['delta_tau']

        # Extract voltage magnitude part (lower half of state vector)
        dV_ds_full = dx_ds[self.n_theta:]

        # Map observation buses to Jacobian V indices
        obs_rows: List[int] = []
        obs_map: List[int] = []
        for bus_idx in observation_bus_indices:
            ppc_bus = pp_bus_to_ppc_bus(self.net, bus_idx)
            _, v_idx = get_jacobian_indices_ppc(self.net, ppc_bus)
            if v_idx is not None and v_idx < len(dV_ds_full):
                obs_rows.append(v_idx)
                obs_map.append(bus_idx)

        if not obs_rows:
            raise ValueError(
                "No valid PQ observation buses found for 3W OLTC sensitivity."
            )

        return dV_ds_full[obs_rows], obs_map

    def compute_dV_ds_trafo3w_matrix(
        self,
        trafo3w_indices: List[int],
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute voltage sensitivity matrix for multiple 3W transformer OLTCs.

        Parameters
        ----------
        trafo3w_indices : List[int]
            Pandapower ``net.trafo3w`` indices.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.

        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape ``(n_obs, n_trafo3w)``.
        observation_bus_mapping : List[int]
        trafo3w_mapping : List[int]
        """
        n_obs = len(observation_bus_indices)
        n_t3w = len(trafo3w_indices)
        if n_obs == 0 or n_t3w == 0:
            return np.zeros((n_obs, n_t3w)), [], []

        matrix = np.zeros((n_obs, n_t3w))
        t3w_map: List[int] = []
        obs_map: Optional[List[int]] = None

        for j, t3w_idx in enumerate(trafo3w_indices):
            col, mapping = self.compute_dV_ds_trafo3w(
                trafo3w_idx=t3w_idx,
                observation_bus_indices=observation_bus_indices,
            )
            matrix[:len(col), j] = col
            t3w_map.append(t3w_idx)
            if obs_map is None:
                obs_map = mapping

        if obs_map is None:
            obs_map = []

        return matrix[:len(obs_map), :len(t3w_map)], obs_map, t3w_map

    def compute_dQtrafo3w_hv_dQ_der(
        self,
        trafo3w_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute HV-side reactive power sensitivity of a 3W transformer
        to DER reactive power injection.

        Implements Eq. (14) from the PSCC 2026 paper applied to the
        HV-star internal branch:

            ΔQ_HV = (∂Q_HV/∂V_hv · ∂V_hv/∂Q_n + ∂Q_HV/∂V_star · ∂V_star/∂Q_n) · ΔQ_n

        Parameters
        ----------
        trafo3w_idx : int
            Pandapower ``net.trafo3w`` index.
        der_bus_idx : int
            Pandapower bus index where the DER is connected.

        Returns
        -------
        sensitivity : float
            ∂Q_HV/∂Q_DER [Mvar/Mvar] (dimensionless).

        Raises
        ------
        ValueError
            If transformer or DER bus cannot be processed.
        """
        d = self._get_trafo3w_hv_branch_data(trafo3w_idx)

        U_i = d['U_hv']
        U_j = d['U_aux']
        theta_i = d['theta_hv']
        theta_j = d['theta_aux']
        tau = d['tau']
        g_val = d['g']
        b_val = d['b']

        # ∂Q_HV/∂V_hv and ∂Q_HV/∂V_star
        dQ_dV_hv = (
            b_val * (-2.0 * U_i / tau**2
                     + U_j * np.cos(theta_i - theta_j) / tau)
            - g_val * U_j * np.sin(theta_i - theta_j) / tau
        )
        dQ_dV_star = (
            b_val * U_i * np.cos(theta_i - theta_j) / tau
            - g_val * U_i * np.sin(theta_i - theta_j) / tau
        )

        # Voltage sensitivities from reduced Jacobian: ∂V/∂Q_DER
        v_hv_jac = d['v_hv_jac']
        v_aux_jac = d['v_aux_jac']

        # DER bus mapping
        ppc_der = pp_bus_to_ppc_bus(self.net, der_bus_idx)
        _, v_der_jac = get_jacobian_indices_ppc(self.net, ppc_der)
        if v_der_jac is None:
            raise ValueError(
                f"DER bus {der_bus_idx} (ppc={ppc_der}) is not a PQ bus; "
                f"cannot compute ∂V/∂Q sensitivity."
            )

        # Bounds check
        n_rows, n_cols = self.dV_dQ_reduced.shape
        if v_der_jac >= n_cols:
            raise ValueError(
                f"DER bus Jacobian index {v_der_jac} out of bounds "
                f"(reduced Jacobian has {n_cols} columns)."
            )

        sensitivity = 0.0

        if v_hv_jac is not None and v_hv_jac < n_rows:
            dV_hv_dQ = self.dV_dQ_reduced[v_hv_jac, v_der_jac]
            sensitivity += dQ_dV_hv * dV_hv_dQ

        if v_aux_jac is not None and v_aux_jac < n_rows:
            dV_star_dQ = self.dV_dQ_reduced[v_aux_jac, v_der_jac]
            sensitivity += dQ_dV_star * dV_star_dQ

        return sensitivity

    def compute_dQtrafo3w_hv_dQ_der_matrix(
        self,
        trafo3w_indices: List[int],
        der_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute HV-side Q sensitivity matrix for multiple 3W transformers
        and DER buses.

        Parameters
        ----------
        trafo3w_indices : List[int]
        der_bus_indices : List[int]

        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Shape ``(n_trafo3w, n_der)``.
        trafo3w_mapping : List[int]
        der_bus_mapping : List[int]
        """
        n_t3w = len(trafo3w_indices)
        n_der = len(der_bus_indices)
        if n_t3w == 0 or n_der == 0:
            return np.zeros((n_t3w, n_der)), [], []

        matrix = np.zeros((n_t3w, n_der))
        t3w_map = list(trafo3w_indices)
        der_map = list(der_bus_indices)

        for i, t3w_idx in enumerate(trafo3w_indices):
            for j, der_bus in enumerate(der_bus_indices):
                matrix[i, j] = self.compute_dQtrafo3w_hv_dQ_der(
                    t3w_idx, der_bus
                )

        return matrix, t3w_map, der_map

    def compute_dQtrafo3w_hv_ds(
        self,
        meas_trafo3w_idx: int,
        chg_trafo3w_idx: int,
        delta_s: float = 1.0,
    ) -> float:
        """
        Compute HV-side Q sensitivity of a 3W transformer to a 3W OLTC
        tap position change.

        Implements Eq. (17) from the PSCC 2026 paper for 3W transformers:

            ΔQ_HV = Δτ · (∂Q_HV/∂x · J⁻¹ · ∂g/∂τ  +  ∂Q_HV/∂τ) · Δs

        Component A (indirect): System-wide voltage/angle changes.
        Component B (direct):   Immediate Q change from tap ratio change
                                (only when measurement and change transformer
                                are the same).

        Parameters
        ----------
        meas_trafo3w_idx : int
            ``net.trafo3w`` index where Q is measured (HV side).
        chg_trafo3w_idx : int
            ``net.trafo3w`` index where the OLTC tap is changed.
        delta_s : float
            Tap step size (default: 1.0).

        Returns
        -------
        sensitivity : float
            ∂Q_HV_meas / ∂s_chg [Mvar per tap step].
        """
        # Data for the change transformer (for ∂g/∂τ)
        d_chg = self._get_trafo3w_hv_branch_data(chg_trafo3w_idx)
        dg_dtau = self._compute_dg_dtau_3w(d_chg)

        # Data for the measurement transformer (for ∂Q_HV/∂x)
        d_meas = self._get_trafo3w_hv_branch_data(meas_trafo3w_idx)
        dQ_dx = self._compute_dQhv_dx_3w(d_meas)

        # Component A (indirect effect):
        #   A = ∂Q_HV/∂x · (-J⁻¹) · ∂g/∂τ · Δτ
        indirect = -dQ_dx @ self.J_inv @ dg_dtau * d_chg['delta_tau']

        # Component B (direct effect, only if same transformer):
        direct = 0.0
        if meas_trafo3w_idx == chg_trafo3w_idx:
            U_i = d_meas['U_hv']
            U_j = d_meas['U_aux']
            theta_i = d_meas['theta_hv']
            theta_j = d_meas['theta_aux']
            tau = d_meas['tau']
            g_val = d_meas['g']
            b_val = d_meas['b']

            # ∂Q_ij/∂τ  (direct derivative of HV-side Q w.r.t. tap ratio)
            dQ_dtau_direct = (
                2.0 * b_val * U_i**2 / tau**3
                - b_val * U_i * U_j
                * np.cos(theta_i - theta_j) / tau**2
                + g_val * U_i * U_j
                * np.sin(theta_i - theta_j) / tau**2
            )
            direct = dQ_dtau_direct * d_meas['delta_tau']

        # Scale from per-unit to Mvar
        return (indirect + direct) * self.net.sn_mva

    def compute_dQtrafo3w_hv_ds_matrix(
        self,
        meas_trafo3w_indices: List[int],
        chg_trafo3w_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute Q_HV sensitivity matrix for multiple 3W transformers
        with respect to multiple 3W OLTC tap positions.

        Parameters
        ----------
        meas_trafo3w_indices : List[int]
            Indices of 3W transformers where Q_HV is measured (rows).
        chg_trafo3w_indices : List[int]
            Indices of 3W transformers whose OLTC is changed (columns).

        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Shape ``(n_meas, n_chg)``.
        meas_mapping : List[int]
        chg_mapping : List[int]
        """
        n_meas = len(meas_trafo3w_indices)
        n_chg = len(chg_trafo3w_indices)
        if n_meas == 0 or n_chg == 0:
            return np.zeros((n_meas, n_chg)), [], []

        matrix = np.zeros((n_meas, n_chg))
        for i, m_idx in enumerate(meas_trafo3w_indices):
            for j, c_idx in enumerate(chg_trafo3w_indices):
                matrix[i, j] = self.compute_dQtrafo3w_hv_ds(m_idx, c_idx)

        return matrix, list(meas_trafo3w_indices), list(chg_trafo3w_indices)

    def compute_dQtrafo3w_hv_dQ_shunt(
        self,
        trafo3w_idx: int,
        shunt_bus_idx: int,
        q_step_mvar: float = 1.0,
    ) -> float:
        """
        Compute HV-side Q sensitivity of a 3W transformer to a shunt
        reactive power injection.

        Internally delegates to :meth:`compute_dQtrafo3w_hv_dQ_der` and
        scales by the shunt step size, since a shunt is modelled as a
        discrete Q injection at the bus.

        Parameters
        ----------
        trafo3w_idx : int
            ``net.trafo3w`` index.
        shunt_bus_idx : int
            Pandapower bus index of the shunt.
        q_step_mvar : float
            Reactive power per shunt state change [Mvar].

        Returns
        -------
        sensitivity : float
            ∂Q_HV / ∂(shunt state) [Mvar per state step].
        """
        # Negate to account for load-convention sign of shunt.q_mvar
        return (
            -self.compute_dQtrafo3w_hv_dQ_der(trafo3w_idx, shunt_bus_idx)
            * q_step_mvar
        )

    # =========================================================================
    # Combined Sensitivity Matrix Construction
    # =========================================================================

    def build_sensitivity_matrix_H(
        self,
        der_bus_indices: List[int],
        observation_bus_indices: List[int],
        line_indices: List[int],
        trafo_indices: Optional[List[int]] = None,
        trafo3w_indices: Optional[List[int]] = None,
        oltc_trafo_indices: Optional[List[int]] = None,
        oltc_trafo3w_indices: Optional[List[int]] = None,
        shunt_bus_indices: Optional[List[int]] = None,
        shunt_q_steps_mvar: Optional[List[float]] = None,
    ) -> Tuple[NDArray[np.float64], dict]:
        """
        Build the combined input-output sensitivity matrix H for OFO.

        This constructs the matrix used in the MIQP optimisation problem.
        The input (column) vector u consists of:

            u = [Q_DER | s_OLTC_2w | s_OLTC_3w | state_shunt]^T

        The output (row) vector y consists of:

            y = [Q_trafo_2w | Q_trafo3w_hv | V_bus | I_branch]^T

        The sensitivity matrix H relates changes in inputs to changes
        in outputs:  dy = H @ du

        Parameters
        ----------
        der_bus_indices : List[int]
            Pandapower bus indices where DERs are connected (continuous Q).
        observation_bus_indices : List[int]
            Pandapower bus indices for voltage measurements (outputs).
        line_indices : List[int]
            Pandapower line indices for current measurements (outputs).
        trafo_indices : List[int], optional
            ``net.trafo`` indices for Q measurements (2W transformers).
        trafo3w_indices : List[int], optional
            ``net.trafo3w`` indices for HV-side Q measurements
            (3-winding transformers).
        oltc_trafo_indices : List[int], optional
            ``net.trafo`` indices with OLTC (2W, integer inputs).
        oltc_trafo3w_indices : List[int], optional
            ``net.trafo3w`` indices with OLTC (3W, integer inputs).
        shunt_bus_indices : List[int], optional
            Pandapower bus indices where switchable shunts are connected.
        shunt_q_steps_mvar : List[float], optional
            Reactive power step per state change for each shunt [Mvar].
            Must match length of ``shunt_bus_indices``.

        Returns
        -------
        H : NDArray[np.float64]
            Combined sensitivity matrix of shape ``(n_outputs, n_inputs)``.

            * Columns: ``[DER Q | OLTC_2w taps | OLTC_3w taps | shunt states]``
            * Rows:    ``[Q_trafo_2w | Q_trafo3w_hv | V_bus | I_branch]``

        mappings : dict
            Dictionary containing the index mappings:

            * ``'der_buses'``: DER bus indices (continuous inputs)
            * ``'oltc_trafos'``: 2W OLTC transformer indices (integer)
            * ``'oltc_trafo3w'``: 3W OLTC transformer indices (integer)
            * ``'shunt_buses'``: shunt bus indices (integer)
            * ``'trafos'``: 2W transformer indices (Q outputs)
            * ``'trafo3w'``: 3W transformer indices (Q_HV outputs)
            * ``'obs_buses'``: observation bus indices (V outputs)
            * ``'lines'``: line indices (I outputs)
            * ``'input_types'``: list of ``'continuous'`` / ``'integer'``

        Raises
        ------
        ValueError
            If no valid inputs or outputs are found, or if shunt
            parameter lists have mismatched lengths.
        """
        if trafo_indices is None:
            trafo_indices = []
        if trafo3w_indices is None:
            trafo3w_indices = []
        if oltc_trafo_indices is None:
            oltc_trafo_indices = []
        if oltc_trafo3w_indices is None:
            oltc_trafo3w_indices = []
        if shunt_bus_indices is None:
            shunt_bus_indices = []
        if shunt_q_steps_mvar is None:
            shunt_q_steps_mvar = [1.0] * len(shunt_bus_indices)

        if len(shunt_bus_indices) != len(shunt_q_steps_mvar):
            raise ValueError(
                "shunt_bus_indices and shunt_q_steps_mvar must have same length."
            )

        # =================================================================
        # Part 1: DER Q sensitivities (continuous inputs)
        # =================================================================

        # --- 2W trafo Q rows ---
        if trafo_indices:
            dQtr2w_dQder, trafo_map, _ = self.compute_dQtrafo_dQ_der_matrix(
                trafo_indices, der_bus_indices
            )
        else:
            trafo_map = []
            dQtr2w_dQder = np.zeros((0, len(der_bus_indices)))

        # --- 3W trafo Q_HV rows ---
        if trafo3w_indices:
            dQtr3w_dQder, t3w_map, _ = self.compute_dQtrafo3w_hv_dQ_der_matrix(
                trafo3w_indices, der_bus_indices
            )
        else:
            t3w_map = []
            dQtr3w_dQder = np.zeros((0, len(der_bus_indices)))

        # --- Voltage rows ---
        dV_dQder, obs_bus_map, _ = self.compute_dV_dQ_der(
            der_bus_indices, observation_bus_indices
        )

        # --- Current rows ---
        dI_dQder, line_map, _ = self.compute_dI_dQ_der_matrix(
            line_indices, der_bus_indices
        )

        n_der = len(der_bus_indices)
        n_trafo2w_out = dQtr2w_dQder.shape[0]
        n_trafo3w_out = dQtr3w_dQder.shape[0]
        n_bus_out = dV_dQder.shape[0]
        n_line_out = dI_dQder.shape[0]
        n_outputs = n_trafo2w_out + n_trafo3w_out + n_bus_out + n_line_out

        # Row offsets for assembling H
        row_q2w = 0
        row_q3w = n_trafo2w_out
        row_v = n_trafo2w_out + n_trafo3w_out
        row_i = row_v + n_bus_out

        # =================================================================
        # Part 2: 2W OLTC tap sensitivities (integer inputs)
        # =================================================================
        dV_ds2w_list: List[NDArray] = []
        dQtr2w_ds2w_list: List[NDArray] = []
        dQtr3w_ds2w_list: List[NDArray] = []
        dI_ds2w_list: List[NDArray] = []
        oltc2w_map: List[int] = []

        for oltc_idx in oltc_trafo_indices:
            try:
                dV_col, _ = self.compute_dV_ds(
                    trafo_idx=oltc_idx,
                    observation_bus_indices=observation_bus_indices,
                )
                # 2W trafo Q w.r.t. 2W OLTC
                dQtr2w_col = np.zeros(n_trafo2w_out)
                for i, mt in enumerate(trafo_map):
                    try:
                        dQtr2w_col[i] = self.compute_dQtrafo_ds(
                            meas_trafo_idx=mt, chg_trafo_idx=oltc_idx
                        )
                    except ValueError:
                        dQtr2w_col[i] = 0.0
                # 3W trafo Q_HV w.r.t. 2W OLTC  (cross-type, indirect only)
                # Not applicable: a 2W OLTC does not directly appear in a
                # 3W branch.  Set to zero (coupling is via the network-wide
                # Jacobian, which is negligible for separated TSO/DSO).
                dQtr3w_col = np.zeros(n_trafo3w_out)
                # Branch current (TODO: implement dI/ds)
                dI_col = np.zeros(n_line_out)

                dV_ds2w_list.append(dV_col)
                dQtr2w_ds2w_list.append(dQtr2w_col)
                dQtr3w_ds2w_list.append(dQtr3w_col)
                dI_ds2w_list.append(dI_col)
                oltc2w_map.append(oltc_idx)
            except ValueError:
                continue

        # =================================================================
        # Part 3: 3W OLTC tap sensitivities (integer inputs)
        # =================================================================
        dV_ds3w_list: List[NDArray] = []
        dQtr2w_ds3w_list: List[NDArray] = []
        dQtr3w_ds3w_list: List[NDArray] = []
        dI_ds3w_list: List[NDArray] = []
        oltc3w_map: List[int] = []

        for oltc3w_idx in oltc_trafo3w_indices:
            try:
                dV_col, _ = self.compute_dV_ds_trafo3w(
                    trafo3w_idx=oltc3w_idx,
                    observation_bus_indices=observation_bus_indices,
                )
                # 2W trafo Q w.r.t. 3W OLTC  (cross-type, zero for same reason)
                dQtr2w_col = np.zeros(n_trafo2w_out)
                # 3W trafo Q_HV w.r.t. 3W OLTC
                dQtr3w_col = np.zeros(n_trafo3w_out)
                for i, mt3w in enumerate(t3w_map):
                    try:
                        dQtr3w_col[i] = self.compute_dQtrafo3w_hv_ds(
                            meas_trafo3w_idx=mt3w, chg_trafo3w_idx=oltc3w_idx
                        )
                    except ValueError:
                        dQtr3w_col[i] = 0.0
                # Branch current (TODO: implement dI/ds)
                dI_col = np.zeros(n_line_out)

                dV_ds3w_list.append(dV_col)
                dQtr2w_ds3w_list.append(dQtr2w_col)
                dQtr3w_ds3w_list.append(dQtr3w_col)
                dI_ds3w_list.append(dI_col)
                oltc3w_map.append(oltc3w_idx)
            except ValueError:
                continue

        # =================================================================
        # Part 4: Shunt sensitivities (integer inputs)
        # =================================================================
        dV_dshunt_list: List[NDArray] = []
        dQtr2w_dshunt_list: List[NDArray] = []
        dQtr3w_dshunt_list: List[NDArray] = []
        dI_dshunt_list: List[NDArray] = []
        shunt_map: List[int] = []

        for shunt_pos, shunt_bus in enumerate(shunt_bus_indices):
            q_step = shunt_q_steps_mvar[shunt_pos]
            try:
                dV_col, _ = self.compute_dV_dQ_shunt(
                    shunt_bus_idx=shunt_bus,
                    observation_bus_indices=observation_bus_indices,
                    q_step_mvar=q_step,
                )
                # 2W trafo Q w.r.t. shunt
                dQtr2w_col = np.zeros(n_trafo2w_out)
                for i, mt in enumerate(trafo_map):
                    try:
                        # Negate for shunt load-convention sign
                        dQtr2w_col[i] = -self.compute_dQtrafo_dQ_der(
                            trafo_idx=mt, der_bus_idx=shunt_bus
                        ) * q_step
                    except ValueError:
                        dQtr2w_col[i] = 0.0
                # 3W trafo Q_HV w.r.t. shunt
                dQtr3w_col = np.zeros(n_trafo3w_out)
                for i, mt3w in enumerate(t3w_map):
                    try:
                        dQtr3w_col[i] = self.compute_dQtrafo3w_hv_dQ_shunt(
                            trafo3w_idx=mt3w,
                            shunt_bus_idx=shunt_bus,
                            q_step_mvar=q_step,
                        )
                    except ValueError:
                        dQtr3w_col[i] = 0.0
                # Branch current w.r.t. shunt
                dI_col, _ = self.compute_dI_dQ_shunt(
                    shunt_bus_idx=shunt_bus,
                    line_indices=line_indices,
                    q_step_mvar=q_step,
                )

                dV_dshunt_list.append(dV_col)
                dQtr2w_dshunt_list.append(dQtr2w_col)
                dQtr3w_dshunt_list.append(dQtr3w_col)
                dI_dshunt_list.append(dI_col)
                shunt_map.append(shunt_bus)
            except ValueError:
                continue

        # =================================================================
        # Part 5: Assemble the full H matrix
        # =================================================================
        n_oltc2w = len(oltc2w_map)
        n_oltc3w = len(oltc3w_map)
        n_shunt_actual = len(shunt_map)
        n_inputs = n_der + n_oltc2w + n_oltc3w + n_shunt_actual

        if n_inputs == 0:
            raise ValueError("No valid inputs found for H matrix.")
        if n_outputs == 0:
            raise ValueError("No valid outputs found for H matrix.")

        H = np.zeros((n_outputs, n_inputs))

        # Column offsets
        col_der = 0
        col_oltc2w = n_der
        col_oltc3w = n_der + n_oltc2w
        col_shunt = n_der + n_oltc2w + n_oltc3w

        # --- DER Q columns ---
        H[row_q2w:row_q2w + n_trafo2w_out, col_der:col_der + n_der] = (
            dQtr2w_dQder[:, :n_der]
        )
        H[row_q3w:row_q3w + n_trafo3w_out, col_der:col_der + n_der] = (
            dQtr3w_dQder[:, :n_der]
        )
        H[row_v:row_v + n_bus_out, col_der:col_der + n_der] = (
            dV_dQder[:, :n_der]
        )
        H[row_i:row_i + n_line_out, col_der:col_der + n_der] = (
            dI_dQder[:, :n_der]
        )

        # --- 2W OLTC columns ---
        for j, (dQtr2w, dQtr3w, dV, dI) in enumerate(zip(
            dQtr2w_ds2w_list, dQtr3w_ds2w_list,
            dV_ds2w_list, dI_ds2w_list,
        )):
            c = col_oltc2w + j
            H[row_q2w:row_q2w + n_trafo2w_out, c] = dQtr2w
            H[row_q3w:row_q3w + n_trafo3w_out, c] = dQtr3w
            H[row_v:row_v + n_bus_out, c] = dV
            H[row_i:row_i + n_line_out, c] = dI

        # --- 3W OLTC columns ---
        for j, (dQtr2w, dQtr3w, dV, dI) in enumerate(zip(
            dQtr2w_ds3w_list, dQtr3w_ds3w_list,
            dV_ds3w_list, dI_ds3w_list,
        )):
            c = col_oltc3w + j
            H[row_q2w:row_q2w + n_trafo2w_out, c] = dQtr2w
            H[row_q3w:row_q3w + n_trafo3w_out, c] = dQtr3w
            H[row_v:row_v + n_bus_out, c] = dV
            H[row_i:row_i + n_line_out, c] = dI

        # --- Shunt columns ---
        for j, (dQtr2w, dQtr3w, dV, dI) in enumerate(zip(
            dQtr2w_dshunt_list, dQtr3w_dshunt_list,
            dV_dshunt_list, dI_dshunt_list,
        )):
            c = col_shunt + j
            H[row_q2w:row_q2w + n_trafo2w_out, c] = dQtr2w
            H[row_q3w:row_q3w + n_trafo3w_out, c] = dQtr3w
            H[row_v:row_v + n_bus_out, c] = dV
            H[row_i:row_i + n_line_out, c] = dI

        # Build input type list
        input_types = (
            ['continuous'] * n_der
            + ['integer'] * n_oltc2w
            + ['integer'] * n_oltc3w
            + ['integer'] * n_shunt_actual
        )

        mappings = {
            'der_buses': der_bus_indices,
            'oltc_trafos': oltc2w_map,
            'oltc_trafo3w': oltc3w_map,
            'shunt_buses': shunt_map,
            'trafos': trafo_map,
            'trafo3w': t3w_map,
            'obs_buses': obs_bus_map,
            'lines': line_map,
            'input_types': input_types,
        }

        return H, mappings
