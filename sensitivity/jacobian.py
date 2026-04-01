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
10) ∂I_ij/∂s_i (2W):                                               -> compute_dI_ds_2w()
                                                                   -> compute_dI_ds_2w_matrix()
10b)∂I_ij/∂s_i (3W):                                               -> compute_dI_ds_3w()
                                                                   -> compute_dI_ds_3w_matrix()
11) ∂I_ij/∂Q_shunt: uses equations from 9)                      -> compute_dI_dQ_shunt()

Author: Manuel Schwenke
Date: 2025-02-05

References
----------
[1] Schwenke et al., PSCC 2026, Section II (Linearised Power System Model)
[2] Milano, F. "Power System Modelling and Scripting", Springer 2010
[3] Kundur & Malik, "Power System Stability and Control", 2nd ed.
"""
from multiprocessing.managers import Value
from optparse import Values

# imports
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Dict
import pandapower as pp
import copy
from sensitivity.index_helper import (get_jacobian_indices, get_ppc_trafo_index, pp_bus_to_ppc_bus,
                                      get_jacobian_indices_ppc, get_ppc_trafo3w_branch_indices,
                                      _get_trafo3w_hv_branch_data)

def _apply_offdiagonal_scaling(
    submatrix: NDArray[np.float64],
    scale_factor: float,
) -> NDArray[np.float64]:
    """Scale all off-diagonal elements of a square submatrix by a given factor.

    This is used to reduce the cross-coupling terms in the dQtr3w_ds block of
    the H matrix, e.g. to account for modelling uncertainty in inter-transformer
    reactive power sensitivities.

    Parameters
    ----------
    submatrix : NDArray[np.float64]
        Square matrix of shape (n, n).
    scale_factor : float
        Multiplicative factor applied to all off-diagonal elements.
        Must be in (0, 1] for physical plausibility; values outside this
        range are not blocked but should be used with caution.

    Returns
    -------
    NDArray[np.float64]
        Modified matrix with scaled off-diagonal elements.

    Raises
    ------
    ValueError
        If the submatrix is not square.
    """
    n_rows, n_cols = submatrix.shape
    if n_rows != n_cols:
        raise ValueError(
            f"Off-diagonal scaling requires a square submatrix, "
            f"got shape ({n_rows}, {n_cols})."
        )

    result = submatrix.copy()
    # Identify off-diagonal positions and apply scaling
    off_diag_mask = ~np.eye(n_rows, dtype=bool)
    result[off_diag_mask] *= scale_factor
    return result


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
        # error handling
        if not net.converged:
            raise ValueError("Network power flow must have converged.")
        if not hasattr(net, '_ppc') or 'internal' not in net._ppc:
            raise ValueError("Network must have internal power flow data (_ppc).")
        if 'J' not in net._ppc['internal']:
            raise ValueError("Network must have Jacobian matrix in internal data.")
        
        # Store deep copy of network state -> this is the network state all sensitivites are based on
        self.net = copy.deepcopy(net)
        
        # Extract Jacobian from power flow solution
        self.J = np.array(self.net._ppc['internal']['J'].todense()) # ToDo: Dense or Sparse?
        
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
            n_obs = len(obs_jacobian_rows) if obs_jacobian_rows else len(observation_bus_indices)
            return np.zeros((n_obs, 0)), obs_bus_mapping, []
        
        # Extract submatrix from reduced Jacobian
        #            column
        #  row      dV_1/dQ_1   dV_1/dQ_2   dV_1/dQ_m
        #           dV_2/dQ_1   dV_2/dQ_2   dV_2/dQ_m
        #           dV_n/dQ_1   dV_n/dQ_2   dV_n/dQ_m
        sensitivity_matrix = self.dV_dQ_reduced[np.ix_(obs_jacobian_rows, der_jacobian_cols)]

        return sensitivity_matrix, obs_bus_mapping, der_bus_mapping

    # =========================================================================
    # A2. Bus Voltage to PV Generator Voltage Setpoint Sensitivity
    # =========================================================================

    def compute_dV_dVgen(
        self,
        gen_bus_ppc: int,
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute observation-bus voltage sensitivity to a PV-bus voltage change.

        For a PV generator bus *k*, this computes ∂V_obs / ∂V_gen by
        constructing the column ∂g/∂V_k (the mismatch sensitivity to the
        PV-bus voltage magnitude) and propagating it through J⁻¹:

            Δx = -J⁻¹ · (∂g/∂V_k) · ΔV_k

        Parameters
        ----------
        gen_bus_ppc : int
            Pypower (ppc) bus index of the PV generator bus.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.

        Returns
        -------
        dV_dVgen : NDArray[np.float64]
            Sensitivity vector of shape ``(n_obs,)`` in [p.u. / p.u.].
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.
        """
        bus_data = self.net._ppc['bus']
        Ybus = np.array(self.net._ppc['internal']['Ybus'].todense())

        n_bus = bus_data.shape[0]
        V_complex = bus_data[:, 7] * np.exp(1j * np.deg2rad(bus_data[:, 8]))

        k = gen_bus_ppc
        V_k = V_complex[k]
        Vm_k = np.abs(V_k)

        # Build ∂g/∂V_k  (state-sized vector: [Δθ_PV, Δθ_PQ, ΔV_PQ])
        # The power injections at every bus i depend on V_k if Y_ik ≠ 0.
        # S_i = V_i · (Σ_j Y_ij V_j)*
        # ∂S_i/∂Vm_k = V_i · conj(Y_ik) · exp(jθ_k)     for i ≠ k
        # ∂S_k/∂Vm_k = (2 Vm_k conj(Y_kk) + Σ_{j≠k} V_j conj(Y_kj)) · exp(jθ_k)
        #            ... but separating P and Q is cleaner via real-form.

        # ∂P_i/∂Vm_k and ∂Q_i/∂Vm_k for all buses
        dP_dVk = np.zeros(n_bus)
        dQ_dVk = np.zeros(n_bus)
        theta = np.angle(V_complex)
        Vm = np.abs(V_complex)
        G = np.real(Ybus)
        B = np.imag(Ybus)

        for i in range(n_bus):
            cos_ik = np.cos(theta[i] - theta[k])
            sin_ik = np.sin(theta[i] - theta[k])
            if i == k:
                # ∂P_k/∂Vm_k = 2 Vm_k G_kk + Σ_{j≠k} Vm_j (G_kj cos + B_kj sin)
                dP_dVk[i] = 2 * Vm[k] * G[k, k]
                dQ_dVk[i] = -2 * Vm[k] * B[k, k]
                for j in range(n_bus):
                    if j != k:
                        cos_kj = np.cos(theta[k] - theta[j])
                        sin_kj = np.sin(theta[k] - theta[j])
                        dP_dVk[i] += Vm[j] * (G[k, j] * cos_kj + B[k, j] * sin_kj)
                        dQ_dVk[i] += Vm[j] * (G[k, j] * sin_kj - B[k, j] * cos_kj)
            else:
                dP_dVk[i] = Vm[i] * (G[i, k] * cos_ik + B[i, k] * sin_ik)
                dQ_dVk[i] = Vm[i] * (G[i, k] * sin_ik - B[i, k] * cos_ik)

        # Map to Jacobian ordering: [P_PV, P_PQ, Q_PQ]
        pv_list = list(self.pv_buses)
        pq_list = list(self.pq_buses)

        dg_dVk = np.zeros(self.x_size)
        for idx_pv, bus_pv in enumerate(pv_list):
            dg_dVk[idx_pv] = dP_dVk[bus_pv]
        n_pv = len(pv_list)
        for idx_pq, bus_pq in enumerate(pq_list):
            dg_dVk[n_pv + idx_pq] = dP_dVk[bus_pq]
            dg_dVk[self.n_theta + idx_pq] = dQ_dVk[bus_pq]

        # Δx = -J⁻¹ · ∂g/∂V_k
        dx_dVk = -self.J_inv @ dg_dVk

        # Extract ΔV_PQ at observation buses
        dV_full = dx_dVk[self.n_theta:]  # V_PQ part

        obs_rows: List[int] = []
        obs_map: List[int] = []
        for bus_idx in observation_bus_indices:
            ppc_bus = pp_bus_to_ppc_bus(self.net, bus_idx)
            if ppc_bus == k:
                # The PV bus itself: ∂V_k/∂V_k = 1
                obs_rows.append(-1)  # sentinel
                obs_map.append(bus_idx)
            else:
                _, v_idx = get_jacobian_indices_ppc(self.net, ppc_bus)
                if v_idx is not None and v_idx < len(dV_full):
                    obs_rows.append(v_idx)
                    obs_map.append(bus_idx)

        dV_dVgen = np.zeros(len(obs_map))
        for i, r in enumerate(obs_rows):
            if r == -1:
                dV_dVgen[i] = 1.0
            else:
                dV_dVgen[i] = dV_full[r]

        return dV_dVgen, obs_map

    def compute_dV_dVgen_matrix(
        self,
        gen_bus_indices_pp: List[int],
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute observation-bus voltage sensitivities for multiple PV generators.

        Parameters
        ----------
        gen_bus_indices_pp : List[int]
            Pandapower bus indices of PV generator buses.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.

        Returns
        -------
        dV_dVgen : NDArray[np.float64]
            Matrix of shape ``(n_obs, n_gen)`` [p.u. / p.u.].
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.
        gen_bus_mapping : List[int]
            Ordered list of generator bus indices actually included.
        """
        # Use first generator to establish observation bus mapping
        first_ppc = pp_bus_to_ppc_bus(self.net, gen_bus_indices_pp[0])
        _, obs_map = self.compute_dV_dVgen(first_ppc, observation_bus_indices)

        n_obs = len(obs_map)
        n_gen = len(gen_bus_indices_pp)
        result = np.zeros((n_obs, n_gen))
        gen_map: List[int] = []

        for j, gen_bus_pp in enumerate(gen_bus_indices_pp):
            gen_bus_ppc = pp_bus_to_ppc_bus(self.net, gen_bus_pp)
            dV, om = self.compute_dV_dVgen(gen_bus_ppc, observation_bus_indices)
            # om should match obs_map (same observation buses)
            result[:, j] = dV
            gen_map.append(gen_bus_pp)

        return result, obs_map, gen_map

    # =========================================================================
    # B. Bus Voltage to OLTC Position Sensitivity (Eq. 11 PSCC 2026)
    # =========================================================================
    
    def compute_dV_ds_2w(
        self,
        trafo_idx: int,
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute bus voltage sensitivity to transformer tap position change for two-winding trafos.
        
        This implements Equation (11) from the PSCC 2026 paper:
            ΔV = Δτ_i [D_21  D_22] (∂g/∂τ_i) Δs_i
        
        where D = J^{-1} and D_21, D_22 are the lower submatrices.
        
        Parameters
        ----------
        trafo_idx : int
            Pandapower transformer index.
        observation_bus_indices : List[int]
            Pandapower bus indices where voltages are observed.
        
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
        V_i = self.net.res_bus.at[hv_bus, 'vm_pu']
        V_j = self.net.res_bus.at[lv_bus, 'vm_pu']
        theta_i = np.deg2rad(self.net.res_bus.at[hv_bus, 'va_degree'])
        theta_j = np.deg2rad(self.net.res_bus.at[lv_bus, 'va_degree'])
        theta = theta_i - theta_j
        
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
        dg_dtau = np.zeros(self.x_size)     # self.x_size = self.n_theta + self.n_v (number of equations)
        
        # Active power derivatives at HV and LV buses
        dPi_dtau = (V_i * V_j * (g * np.cos(theta) + b * np.sin(theta)) / tau**2        # ij
                     - 2 * g * V_i**2 / tau**3)                                         # ii
        dPj_dtau = (V_j * V_i * (g * np.cos(theta) - b * np.sin(theta)) / tau**2)       # ij
        
        # Reactive power derivatives at HV and LV buses
        dQi_dtau = (V_i * V_j * (g * np.sin(theta) - b * np.cos(theta)) / (tau**2) +    # ij
                    + 2 * b * V_i ** 2 / (tau ** 3))                                    # ii
        dQj_dtau = (V_j * V_i * (- g * np.sin(theta) - b * np.cos(theta)) / tau**2)       # ij
        
        # Accumulate derivatives into dg_dtau vector
        if theta_i_idx is not None:
            dg_dtau[theta_i_idx] += dPi_dtau
        if theta_j_idx is not None:
            dg_dtau[theta_j_idx] += dPj_dtau
        if v_i_idx is not None:
            dg_dtau[self.n_theta + v_i_idx] += dQi_dtau
        if v_j_idx is not None and (self.n_theta + v_j_idx) < self.x_size:
            dg_dtau[self.n_theta + v_j_idx] += dQj_dtau
        
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

    def compute_dV_ds_2w_matrix(
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
                dV_ds_col, obs_mapping = self.compute_dV_ds_2w(
                    trafo_idx=trafo_idx,
                    observation_bus_indices=observation_bus_indices,
                )
                sensitivity_matrix[:len(dV_ds_col), j] = dV_ds_col
                trafo_mapping.append(trafo_idx)
                if obs_bus_mapping is None:
                    obs_bus_mapping = obs_mapping
            except ValueError:
                print(ValueError)
                continue
        
        if obs_bus_mapping is None:
            obs_bus_mapping = []
        
        return sensitivity_matrix[:len(obs_bus_mapping), :len(trafo_mapping)], obs_bus_mapping, trafo_mapping
    
    # =========================================================================
    # C. Transformer Q to DER Reactive Power Infeed Sensitivity (Eq. 12-14 PSCC 2026)
    # =========================================================================
    
    def compute_dQtrafo_dQder_2w(
        self,
        trafo_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute two winding transformer reactive power flow sensitivity to DER reactive power.
        
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
        # error handling
        if trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Transformer {trafo_idx} not found in network.")
        ppc_br_idx = get_ppc_trafo_index(self.net, trafo_idx)
        if ppc_br_idx is None:
            raise ValueError(f"Could not find pypower branch index for transformer {trafo_idx}.")
        
        # Get transformer buses
        hv_bus = self.net.trafo.at[trafo_idx, 'hv_bus']
        lv_bus = self.net.trafo.at[trafo_idx, 'lv_bus']
        
        # Get voltage states
        V_i = self.net.res_bus.at[hv_bus, 'vm_pu']
        V_j = self.net.res_bus.at[lv_bus, 'vm_pu']
        theta_i = np.deg2rad(self.net.res_bus.at[hv_bus, 'va_degree'])
        theta_j = np.deg2rad(self.net.res_bus.at[lv_bus, 'va_degree'])
        theta = theta_i - theta_j
        
        # Get transformer parameters
        r_pu = self.net._ppc['branch'][ppc_br_idx, 2]
        x_pu = self.net._ppc['branch'][ppc_br_idx, 3]
        y_pu = 1.0 / complex(r_pu, x_pu)
        g = y_pu.real
        b = y_pu.imag
        
        # Get tap ratio
        s0 = self.net.trafo.at[trafo_idx, 'tap_pos']
        delta_tau = self.net.trafo.at[trafo_idx, 'tap_step_percent'] / 100.0
        tau = 1.0 +  s0 * delta_tau
        
        # Get Jacobian indices
        _, v_i_idx = get_jacobian_indices(self.net, hv_bus) # index for Q-equation @ hv bus
        _, v_j_idx = get_jacobian_indices(self.net, lv_bus) # index for Q-equation @ lv bus
        _, v_der_idx = get_jacobian_indices(self.net, der_bus_idx) # index for Q-equation @ der bus
        if v_i_idx is None or v_j_idx is None or v_der_idx is None:
            raise ValueError("Could not find Jacobian indices for buses.")
        
        # Compute ∂Q_ij/∂V_i and ∂Q_ij/∂V_j (Equation 12)
        # Q_ij = -b * U_i^2 / τ^2 + U_i * U_j * (b * cos(θ_i - θ_j) - g * sin(θ_i - θ_j)) / τ
        dQ_ij_dV_i = (b * (-2.0 * V_i / (tau**2) + V_j * np.cos(theta) / tau) -
                      g * V_j * np.sin(theta) / tau)
        dQ_ij_dV_j = (b * V_i * np.cos(theta) / tau -
                      g * V_i * np.sin(theta) / tau)
        
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
    
    def compute_dQtrafo_dQder_2w_matrix(
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
                print(ValueError)
                continue
        
        for i, trafo_idx in enumerate(trafo_indices):
            try:
                trafo_mapping.append(trafo_idx)
                for j, der_bus_idx in enumerate(der_bus_indices):
                    if der_bus_idx not in der_bus_mapping:
                        continue
                    j_mapped = der_bus_mapping.index(der_bus_idx)
                    sensitivity_matrix[i, j_mapped] = self.compute_dQtrafo_dQder_2w(trafo_idx, der_bus_idx)
            except ValueError:
                continue
        
        return sensitivity_matrix[:len(trafo_mapping), :len(der_bus_mapping)], trafo_mapping, der_bus_mapping
    
    # =========================================================================
    # D. Transformer Q to OLTC Position Sensitivity (Eq. 15-17)
    # =========================================================================
    
    def compute_dQtrafo_2w_ds(
        self,
        meas_trafo_idx: int,
        chg_trafo_idx: int,
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
        
        Returns
        -------
        sensitivity : float
            Sensitivity ∂Q_meas/∂s_chg in [Mvar per tap step].
        
        Raises
        ------
        ValueError
            If transformers are not found or cannot be processed.
        """
        # error handling
        if meas_trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Measurement transformer {meas_trafo_idx} not found.")
        if chg_trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Change transformer {chg_trafo_idx} not found.")
        
        # === Part 1: Compute ∂g/∂τ for the change transformer ===
        # ========================================================
        ppc_br_idx_chg = get_ppc_trafo_index(self.net, chg_trafo_idx)
        if ppc_br_idx_chg is None:
            raise ValueError(f"Could not find pypower index for transformer {chg_trafo_idx}.")
        
        hv_bus_chg = self.net.trafo.at[chg_trafo_idx, 'hv_bus']
        lv_bus_chg = self.net.trafo.at[chg_trafo_idx, 'lv_bus']
        
        V_i_chg = self.net.res_bus.at[hv_bus_chg, 'vm_pu']
        V_j_chg = self.net.res_bus.at[lv_bus_chg, 'vm_pu']
        theta_i_chg = np.deg2rad(self.net.res_bus.at[hv_bus_chg, 'va_degree'])
        theta_j_chg = np.deg2rad(self.net.res_bus.at[lv_bus_chg, 'va_degree'])
        theta_chg = theta_i_chg - theta_j_chg
        
        s0_chg = self.net.trafo.at[chg_trafo_idx, 'tap_pos']
        delta_tau_chg = self.net.trafo.at[chg_trafo_idx, 'tap_step_percent'] / 100.0
        tau_chg = 1.0 + s0_chg * delta_tau_chg
        
        r_pu_chg = self.net._ppc['branch'][ppc_br_idx_chg, 2]
        x_pu_chg = self.net._ppc['branch'][ppc_br_idx_chg, 3]
        y_pu_chg = 1.0 / complex(r_pu_chg, x_pu_chg)
        g_chg = y_pu_chg.real
        b_chg = y_pu_chg.imag

        # Get jacobian indices
        theta_i_idx_chg, v_i_idx_chg = get_jacobian_indices(self.net, hv_bus_chg) # P_i, Q_i indices
        theta_j_idx_chg, v_j_idx_chg = get_jacobian_indices(self.net, lv_bus_chg) # P_j, Q_j indices
        if theta_i_idx_chg is None or theta_j_idx_chg is None:
            raise ValueError("Could not find Jacobian indices for change transformer buses.")
        
        # Compute ∂g/∂τ
        dg_dtau = np.zeros(self.x_size)

        # Active power derivatives at HV and LV buses
        dPi_dtau = (V_i_chg * V_j_chg * (g_chg * np.cos(theta_chg) + b_chg * np.sin(theta_chg)) / tau_chg ** 2
                    - 2 * g_chg * V_i_chg ** 2 / tau_chg ** 3)  # ii
        dPj_dtau = (V_j_chg * V_i_chg * (g_chg * np.cos(theta_chg) - b_chg * np.sin(theta_chg)) / tau_chg ** 2)

        # Reactive power derivatives at HV and LV buses
        dQi_dtau = (V_i_chg * V_j_chg * (g_chg * np.sin(theta_chg) - b_chg * np.cos(theta_chg)) / (tau_chg ** 2) +
                    + 2 * b_chg * V_i_chg ** 2 / (tau_chg ** 3))  # ii
        dQj_dtau = (V_j_chg * V_i_chg * (-g_chg * np.sin(theta_chg) - b_chg * np.cos(theta_chg)) / tau_chg ** 2)

        if theta_i_idx_chg is not None:
            dg_dtau[theta_i_idx_chg] += dPi_dtau
        if theta_j_idx_chg is not None:
            dg_dtau[theta_j_idx_chg] += dPj_dtau
        if v_i_idx_chg is not None:
            dg_dtau[self.n_theta + v_i_idx_chg] += dQi_dtau
        if v_j_idx_chg is not None and (self.n_theta + v_j_idx_chg) < self.x_size:
            dg_dtau[self.n_theta + v_j_idx_chg] += dQj_dtau
        
        # === Part 2: Compute ∂Q_ij/∂x for the measurement transformer ===
        # ================================================================
        ppc_br_idx_meas = get_ppc_trafo_index(self.net, meas_trafo_idx)
        if ppc_br_idx_meas is None:
            raise ValueError(f"Could not find pypower index for transformer {meas_trafo_idx}.")
        
        hv_bus_meas = self.net.trafo.at[meas_trafo_idx, 'hv_bus']
        lv_bus_meas = self.net.trafo.at[meas_trafo_idx, 'lv_bus']
        
        V_i_meas = self.net.res_bus.at[hv_bus_meas, 'vm_pu']
        V_j_meas = self.net.res_bus.at[lv_bus_meas, 'vm_pu']
        theta_i_meas = np.deg2rad(self.net.res_bus.at[hv_bus_meas, 'va_degree'])
        theta_j_meas = np.deg2rad(self.net.res_bus.at[lv_bus_meas, 'va_degree'])
        theta_meas = theta_i_meas - theta_j_meas

        s0_meas = self.net.trafo.at[meas_trafo_idx, 'tap_pos']
        delta_tau_meas = self.net.trafo.at[meas_trafo_idx, 'tap_step_percent'] / 100.0
        tau_meas = 1.0 + s0_meas * delta_tau_meas
        
        r_pu_meas = self.net._ppc['branch'][ppc_br_idx_meas, 2]
        x_pu_meas = self.net._ppc['branch'][ppc_br_idx_meas, 3]
        y_pu_meas = 1.0 / complex(r_pu_meas, x_pu_meas)
        g_meas = y_pu_meas.real
        b_meas = y_pu_meas.imag

        # Get jacbian indices
        theta_i_idx_meas, v_i_idx_meas = get_jacobian_indices(self.net, hv_bus_meas)
        theta_j_idx_meas, v_j_idx_meas = get_jacobian_indices(self.net, lv_bus_meas)
        if theta_i_idx_meas is None or theta_j_idx_meas is None:
            raise ValueError("Could not find Jacobian indices for measurement transformer buses.")
        
        # Compute ∂Q_ij/∂x
        dQ_dx = np.zeros(self.x_size)
        
        dQ_ij_dtheta_i = V_i_meas * V_j_meas / tau_meas  * (-b_meas * np.sin(theta_meas)
                                                            - g_meas * np.cos(theta_meas))
        dQ_ij_dtheta_j = V_i_meas * V_j_meas / tau_meas * (b_meas * np.sin(theta_meas)
                                                            + g_meas * np.cos(theta_meas))
        dQ_ij_dV_i = (b_meas * (-2.0 * V_i_meas / (tau_meas**2) + V_j_meas * np.cos(theta_meas) / tau_meas) -
                      g_meas * V_j_meas * np.sin(theta_meas) / tau_meas)
        dQ_ij_dV_j = V_i_meas / tau_meas * (b_meas * np.cos(theta_meas)
                                            - g_meas * np.sin(theta_meas))

        if theta_i_idx_meas is not None:
            dQ_dx[theta_i_idx_meas] += dQ_ij_dtheta_i
        if theta_j_idx_meas is not None:
            dQ_dx[theta_j_idx_meas] += dQ_ij_dtheta_j
        if v_i_idx_meas is not None:
            dQ_dx[self.n_theta + v_i_idx_meas] += dQ_ij_dV_i
        if v_j_idx_meas is not None and (self.n_theta + v_j_idx_meas) < self.x_size:
            dQ_dx[self.n_theta + v_j_idx_meas] += dQ_ij_dV_j
        
        # === Part 3: Compute indirect effect (A) ===
        # A = ∂Q_ij/∂x · J^{-1} · ∂g/∂τ
        indirect_effect = -dQ_dx @ self.J_inv @ dg_dtau * delta_tau_chg

        # === Part 4: Compute direct effect (B) ===
        # Only applies if the measurement transformer is the same as change transformer
        direct_effect = 0.0
        if meas_trafo_idx == chg_trafo_idx:
            # ∂Q_ij/∂τ (direct derivative of Q flow with respect to tap ratio)
            dQ_dtau_direct = (2.0 * b_meas * V_i_meas**2 / tau_meas**3 -
                              b_meas * V_i_meas * V_j_meas * np.cos(theta_i_meas - theta_j_meas) / tau_meas**2 +
                              g_meas * V_i_meas * V_j_meas * np.sin(theta_i_meas - theta_j_meas) / tau_meas**2)
            direct_effect = dQ_dtau_direct * delta_tau_chg

        # Total sensitivity (Equation 17), scaled from per-unit to Mvar
        sensitivity = (indirect_effect + direct_effect)

        return sensitivity

    def compute_dQtrafo_ds_2w_matrix(
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
                    sensitivity_matrix[i, j] = self.compute_dQtrafo_2w_ds(meas_idx, chg_idx)
                except ValueError:
                    print(ValueError)
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

        # Get line's ppc branch index
        line_start = self.net._pd2ppc_lookups['branch']['line'][0]
        pos = list(self.net.line.index).index(line_idx)
        ppc_line_idx = line_start + pos

        # Get series admittance AND shunt susceptance
        r_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 2]))
        x_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 3]))
        b_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 4]))

        Y_line = 1.0 / complex(r_pu, x_pu)
        Y_shunt = 1j * b_pu / 2.0  # Shunt admittance at the from-bus

        # Compute current phasor with the Pi-model shunt included (Equation 18)
        I_ij = Y_line * (V_from - V_to) + Y_shunt * V_from
        I_mag = np.abs(I_ij)

        if I_mag < 1e-10:
            return 0.0

        # Get Jacobian indices (theta_idx, v_idx) for each bus
        theta_from_idx, v_from_idx = get_jacobian_indices(self.net, from_bus)
        theta_to_idx, v_to_idx = get_jacobian_indices(self.net, to_bus)
        _, v_der_idx = get_jacobian_indices(self.net, der_bus_idx)

        if v_der_idx is None:
            raise ValueError("DER bus must be a PQ bus to compute Q injection sensitivity.")

        if v_der_idx >= self.dV_dQ_reduced.shape[1]:
            raise ValueError("Jacobian index out of bounds.")

        if (v_from_idx is not None and v_from_idx >= self.dV_dQ_reduced.shape[0]) or \
                (v_to_idx is not None and v_to_idx >= self.dV_dQ_reduced.shape[0]):
            raise ValueError("Jacobian index out of bounds.")

        # Column in J_inv for a Q perturbation at der_bus
        col = self.n_theta + v_der_idx

        # ∂|V|/∂Q_pu — PV buses have constant voltage magnitude, so derivative is 0.0
        dVmag_from_dQpu = self.dV_dQ_reduced[v_from_idx, v_der_idx] if v_from_idx is not None else 0.0
        dVmag_to_dQpu = self.dV_dQ_reduced[v_to_idx, v_der_idx] if v_to_idx is not None else 0.0

        # ∂θ/∂Q_pu — from upper-right block of full J_inv
        dTheta_from_dQpu = self.J_inv[theta_from_idx, col] if theta_from_idx is not None else 0.0
        dTheta_to_dQpu = self.J_inv[theta_to_idx, col] if theta_to_idx is not None else 0.0

        # Full complex voltage sensitivity ∂V_i/∂Q_pu
        dV_from_dQmvar = (dVmag_from_dQpu + 1j * U_from * dTheta_from_dQpu) \
                         * np.exp(1j * theta_from) / self.net.sn_mva
        dV_to_dQmvar = (dVmag_to_dQpu + 1j * U_to * dTheta_to_dQpu) \
                       * np.exp(1j * theta_to) / self.net.sn_mva

        # ∂I_ij/∂Q_Mvar — Include the derivative of the shunt term!
        dI_ij_dQmvar = Y_line * (dV_from_dQmvar - dV_to_dQmvar) + Y_shunt * dV_from_dQmvar

        # ∂|I_ij|/∂Q_Mvar (Equation 20)
        dI_mag_dQmvar = (1.0 / I_mag) * np.real(np.conj(I_ij) * dI_ij_dQmvar)

        # Convert p.u. current → kA via current base
        base_voltage_kv = self.net.bus.at[from_bus, 'vn_kv']
        I_base_kA = self.net.sn_mva / (np.sqrt(3) * base_voltage_kv)
        sensitivity = dI_mag_dQmvar * I_base_kA  # [kA/Mvar]

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

    def compute_dI_ds_2w(
            self,
            line_idx: int,
            trafo_idx: int,
    ) -> float:
        """
        Compute branch current magnitude sensitivity to two-winding transformer tap position.
        """
        if line_idx not in self.net.line.index:
            raise ValueError(f"Line {line_idx} not found in network.")
        if trafo_idx not in self.net.trafo.index:
            raise ValueError(f"Transformer {trafo_idx} not found in network.")

        # --- 1. Compute dx/ds for the transformer ---
        ppc_br_idx = get_ppc_trafo_index(self.net, trafo_idx)
        if ppc_br_idx is None:
            raise ValueError(f"Could not find pypower index for transformer {trafo_idx}.")

        hv_bus = self.net.trafo.at[trafo_idx, 'hv_bus']
        lv_bus = self.net.trafo.at[trafo_idx, 'lv_bus']

        V_i = self.net.res_bus.at[hv_bus, 'vm_pu']
        V_j = self.net.res_bus.at[lv_bus, 'vm_pu']
        theta_i = np.deg2rad(self.net.res_bus.at[hv_bus, 'va_degree'])
        theta_j = np.deg2rad(self.net.res_bus.at[lv_bus, 'va_degree'])
        theta = theta_i - theta_j

        s0 = self.net.trafo.at[trafo_idx, 'tap_pos']
        delta_tau = self.net.trafo.at[trafo_idx, 'tap_step_percent'] / 100.0
        tau = 1.0 + s0 * delta_tau

        r_pu_tr = self.net._ppc['branch'][ppc_br_idx, 2]
        x_pu_tr = self.net._ppc['branch'][ppc_br_idx, 3]
        y_pu_tr = 1.0 / complex(r_pu_tr, x_pu_tr)
        g = y_pu_tr.real
        b = y_pu_tr.imag

        theta_i_idx, v_i_idx = get_jacobian_indices(self.net, hv_bus)
        theta_j_idx, v_j_idx = get_jacobian_indices(self.net, lv_bus)

        dg_dtau = np.zeros(self.x_size)

        dPi_dtau = (V_i * V_j * (g * np.cos(theta) + b * np.sin(theta)) / tau ** 2 - 2 * g * V_i ** 2 / tau ** 3)
        dPj_dtau = (V_j * V_i * (g * np.cos(theta) - b * np.sin(theta)) / tau ** 2)
        dQi_dtau = (V_i * V_j * (g * np.sin(theta) - b * np.cos(theta)) / tau ** 2 + 2 * b * V_i ** 2 / tau ** 3)
        dQj_dtau = (V_j * V_i * (-g * np.sin(theta) - b * np.cos(theta)) / tau ** 2)

        if theta_i_idx is not None: dg_dtau[theta_i_idx] += dPi_dtau
        if theta_j_idx is not None: dg_dtau[theta_j_idx] += dPj_dtau
        if v_i_idx is not None: dg_dtau[self.n_theta + v_i_idx] += dQi_dtau
        if v_j_idx is not None and (self.n_theta + v_j_idx) < self.x_size:
            dg_dtau[self.n_theta + v_j_idx] += dQj_dtau

        dx_ds = -self.J_inv @ dg_dtau * delta_tau

        # --- 2. Compute branch current sensitivity ---
        from_bus = self.net.line.at[line_idx, 'from_bus']
        to_bus = self.net.line.at[line_idx, 'to_bus']

        U_from = self.net.res_bus.at[from_bus, 'vm_pu']
        U_to = self.net.res_bus.at[to_bus, 'vm_pu']
        theta_from = np.deg2rad(self.net.res_bus.at[from_bus, 'va_degree'])
        theta_to = np.deg2rad(self.net.res_bus.at[to_bus, 'va_degree'])

        V_from = U_from * np.exp(1j * theta_from)
        V_to = U_to * np.exp(1j * theta_to)

        line_start = self.net._pd2ppc_lookups['branch']['line'][0]
        pos = list(self.net.line.index).index(line_idx)
        ppc_line_idx = line_start + pos

        r_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 2]))
        x_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 3]))
        b_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 4]))

        Y_line = 1.0 / complex(r_pu, x_pu)
        Y_shunt = 1j * b_pu / 2.0

        I_ij = Y_line * (V_from - V_to) + Y_shunt * V_from
        I_mag = np.abs(I_ij)
        if I_mag < 1e-10:
            return 0.0

        theta_from_idx, v_from_idx = get_jacobian_indices(self.net, from_bus)
        theta_to_idx, v_to_idx = get_jacobian_indices(self.net, to_bus)

        dTheta_from_ds = dx_ds[theta_from_idx] if theta_from_idx is not None else 0.0
        dTheta_to_ds = dx_ds[theta_to_idx] if theta_to_idx is not None else 0.0
        dVmag_from_ds = dx_ds[self.n_theta + v_from_idx] if v_from_idx is not None else 0.0
        dVmag_to_ds = dx_ds[self.n_theta + v_to_idx] if v_to_idx is not None else 0.0

        dV_from_ds = (dVmag_from_ds + 1j * U_from * dTheta_from_ds) * np.exp(1j * theta_from)
        dV_to_ds = (dVmag_to_ds + 1j * U_to * dTheta_to_ds) * np.exp(1j * theta_to)

        dI_ij_ds = Y_line * (dV_from_ds - dV_to_ds) + Y_shunt * dV_from_ds
        dI_mag_ds = (1.0 / I_mag) * np.real(np.conj(I_ij) * dI_ij_ds)

        base_voltage_kv = self.net.bus.at[from_bus, 'vn_kv']
        I_base_kA = self.net.sn_mva / (np.sqrt(3) * base_voltage_kv)

        return dI_mag_ds * I_base_kA

    def compute_dI_ds_2w_matrix(
            self,
            line_indices: List[int],
            trafo_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        n_lines = len(line_indices)
        n_trafos = len(trafo_indices)
        if n_lines == 0 or n_trafos == 0:
            return np.zeros((n_lines, n_trafos)), [], []

        sensitivity_matrix = np.zeros((n_lines, n_trafos))
        line_mapping = list(line_indices)
        trafo_mapping = list(trafo_indices)

        for i, line_idx in enumerate(line_indices):
            for j, trafo_idx in enumerate(trafo_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dI_ds_2w(line_idx, trafo_idx)
                except ValueError:
                    sensitivity_matrix[i, j] = 0.0

        return sensitivity_matrix, line_mapping, trafo_mapping

    def compute_dI_ds_3w(
            self,
            line_idx: int,
            trafo3w_idx: int,
    ) -> float:
        """
        Compute branch current magnitude sensitivity to 3W transformer HV OLTC tap.

        Uses the implicit function theorem applied to the HV-star branch of a
        3-winding transformer, then chains through the pi-model current formula.

        Parameters
        ----------
        line_idx : int
            Pandapower line index for the current measurement.
        trafo3w_idx : int
            Pandapower ``net.trafo3w`` index.

        Returns
        -------
        sensitivity : float
            d|I_line|/ds  [kA per tap step].

        Raises
        ------
        ValueError
            If the line or transformer is not found.
        """
        if line_idx not in self.net.line.index:
            raise ValueError(f"Line {line_idx} not found in network.")
        if trafo3w_idx not in self.net.trafo3w.index:
            raise ValueError(f"Transformer 3W {trafo3w_idx} not found in network.")

        # --- 1. Compute dx/ds for the 3W transformer ---
        d = _get_trafo3w_hv_branch_data(self.net, trafo3w_idx)
        dg_dtau = self._compute_dg_dtau_3w(d)

        # Full state sensitivity:  dx/ds = -J^{-1} (dg/dtau) * delta_tau
        dx_ds = -self.J_inv @ dg_dtau * d['delta_tau']

        # --- 2. Compute branch current sensitivity ---
        from_bus = self.net.line.at[line_idx, 'from_bus']
        to_bus = self.net.line.at[line_idx, 'to_bus']

        U_from = self.net.res_bus.at[from_bus, 'vm_pu']
        U_to = self.net.res_bus.at[to_bus, 'vm_pu']
        theta_from = np.deg2rad(self.net.res_bus.at[from_bus, 'va_degree'])
        theta_to = np.deg2rad(self.net.res_bus.at[to_bus, 'va_degree'])

        V_from = U_from * np.exp(1j * theta_from)
        V_to = U_to * np.exp(1j * theta_to)

        line_start = self.net._pd2ppc_lookups['branch']['line'][0]
        pos = list(self.net.line.index).index(line_idx)
        ppc_line_idx = line_start + pos

        r_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 2]))
        x_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 3]))
        b_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 4]))

        Y_line = 1.0 / complex(r_pu, x_pu)
        Y_shunt = 1j * b_pu / 2.0

        I_ij = Y_line * (V_from - V_to) + Y_shunt * V_from
        I_mag = np.abs(I_ij)
        if I_mag < 1e-10:
            return 0.0

        theta_from_idx, v_from_idx = get_jacobian_indices(self.net, from_bus)
        theta_to_idx, v_to_idx = get_jacobian_indices(self.net, to_bus)

        dTheta_from_ds = dx_ds[theta_from_idx] if theta_from_idx is not None else 0.0
        dTheta_to_ds = dx_ds[theta_to_idx] if theta_to_idx is not None else 0.0
        dVmag_from_ds = dx_ds[self.n_theta + v_from_idx] if v_from_idx is not None else 0.0
        dVmag_to_ds = dx_ds[self.n_theta + v_to_idx] if v_to_idx is not None else 0.0

        dV_from_ds = (dVmag_from_ds + 1j * U_from * dTheta_from_ds) * np.exp(1j * theta_from)
        dV_to_ds = (dVmag_to_ds + 1j * U_to * dTheta_to_ds) * np.exp(1j * theta_to)

        dI_ij_ds = Y_line * (dV_from_ds - dV_to_ds) + Y_shunt * dV_from_ds
        dI_mag_ds = (1.0 / I_mag) * np.real(np.conj(I_ij) * dI_ij_ds)

        base_voltage_kv = self.net.bus.at[from_bus, 'vn_kv']
        I_base_kA = self.net.sn_mva / (np.sqrt(3) * base_voltage_kv)

        return dI_mag_ds * I_base_kA

    def compute_dI_ds_3w_matrix(
            self,
            line_indices: List[int],
            trafo3w_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute branch current sensitivity matrix for multiple 3W OLTC taps.

        Parameters
        ----------
        line_indices : List[int]
            Pandapower line indices.
        trafo3w_indices : List[int]
            Pandapower ``net.trafo3w`` indices with OLTC.

        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Matrix of shape ``(n_lines, n_trafo3w)``.
        line_mapping : List[int]
        trafo3w_mapping : List[int]
        """
        n_lines = len(line_indices)
        n_trafos = len(trafo3w_indices)
        if n_lines == 0 or n_trafos == 0:
            return np.zeros((n_lines, n_trafos)), [], []

        sensitivity_matrix = np.zeros((n_lines, n_trafos))
        line_mapping = list(line_indices)
        trafo_mapping = list(trafo3w_indices)

        for i, line_idx in enumerate(line_indices):
            for j, trafo3w_idx in enumerate(trafo3w_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dI_ds_3w(line_idx, trafo3w_idx)
                except ValueError:
                    sensitivity_matrix[i, j] = 0.0

        return sensitivity_matrix, line_mapping, trafo_mapping

    def compute_dI_dVgen(
            self,
            line_idx: int,
            gen_bus_ppc: int,
    ) -> float:
        """
        Compute branch current magnitude sensitivity to PV generator voltage setpoint.
        """
        if line_idx not in self.net.line.index:
            raise ValueError(f"Line {line_idx} not found in network.")

        # --- 1. Compute dx/dVk for the generator ---
        bus_data = self.net._ppc['bus']
        Ybus = np.array(self.net._ppc['internal']['Ybus'].todense())
        n_bus = bus_data.shape[0]
        V_complex = bus_data[:, 7] * np.exp(1j * np.deg2rad(bus_data[:, 8]))

        k = gen_bus_ppc
        Vm = np.abs(V_complex)
        theta = np.angle(V_complex)
        G = np.real(Ybus)
        B = np.imag(Ybus)

        dP_dVk = np.zeros(n_bus)
        dQ_dVk = np.zeros(n_bus)
        for i in range(n_bus):
            cos_ik = np.cos(theta[i] - theta[k])
            sin_ik = np.sin(theta[i] - theta[k])
            if i == k:
                dP_dVk[i] = 2 * Vm[k] * G[k, k]
                dQ_dVk[i] = -2 * Vm[k] * B[k, k]
                for j_bus in range(n_bus):
                    if j_bus != k:
                        cos_kj = np.cos(theta[k] - theta[j_bus])
                        sin_kj = np.sin(theta[k] - theta[j_bus])
                        dP_dVk[i] += Vm[j_bus] * (G[k, j_bus] * cos_kj + B[k, j_bus] * sin_kj)
                        dQ_dVk[i] += Vm[j_bus] * (G[k, j_bus] * sin_kj - B[k, j_bus] * cos_kj)
            else:
                dP_dVk[i] = Vm[i] * (G[i, k] * cos_ik + B[i, k] * sin_ik)
                dQ_dVk[i] = Vm[i] * (G[i, k] * sin_ik - B[i, k] * cos_ik)

        pv_list = list(self.pv_buses)
        pq_list = list(self.pq_buses)
        dg_dVk = np.zeros(self.x_size)

        for idx_pv, bus_pv in enumerate(pv_list):
            dg_dVk[idx_pv] = dP_dVk[bus_pv]
        n_pv = len(pv_list)
        for idx_pq, bus_pq in enumerate(pq_list):
            dg_dVk[n_pv + idx_pq] = dP_dVk[bus_pq]
            dg_dVk[self.n_theta + idx_pq] = dQ_dVk[bus_pq]

        dx_dVk = -self.J_inv @ dg_dVk

        # --- 2. Compute branch current sensitivity ---
        from_bus = self.net.line.at[line_idx, 'from_bus']
        to_bus = self.net.line.at[line_idx, 'to_bus']

        U_from = self.net.res_bus.at[from_bus, 'vm_pu']
        U_to = self.net.res_bus.at[to_bus, 'vm_pu']
        theta_from = np.deg2rad(self.net.res_bus.at[from_bus, 'va_degree'])
        theta_to = np.deg2rad(self.net.res_bus.at[to_bus, 'va_degree'])

        V_from = U_from * np.exp(1j * theta_from)
        V_to = U_to * np.exp(1j * theta_to)

        line_start = self.net._pd2ppc_lookups['branch']['line'][0]
        pos = list(self.net.line.index).index(line_idx)
        ppc_line_idx = line_start + pos

        r_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 2]))
        x_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 3]))
        b_pu = float(np.real(self.net._ppc['branch'][ppc_line_idx, 4]))

        Y_line = 1.0 / complex(r_pu, x_pu)
        Y_shunt = 1j * b_pu / 2.0

        I_ij = Y_line * (V_from - V_to) + Y_shunt * V_from
        I_mag = np.abs(I_ij)
        if I_mag < 1e-10:
            return 0.0

        theta_from_idx, v_from_idx = get_jacobian_indices(self.net, from_bus)
        theta_to_idx, v_to_idx = get_jacobian_indices(self.net, to_bus)

        dTheta_from_dVk = dx_dVk[theta_from_idx] if theta_from_idx is not None else 0.0
        dTheta_to_dVk = dx_dVk[theta_to_idx] if theta_to_idx is not None else 0.0

        # For magnitudes, if a bus is the PV generator bus, its dVmag/dVk = 1.0
        from_ppc = pp_bus_to_ppc_bus(self.net, from_bus)
        to_ppc = pp_bus_to_ppc_bus(self.net, to_bus)

        dVmag_from_dVk = 1.0 if from_ppc == gen_bus_ppc else (
            dx_dVk[self.n_theta + v_from_idx] if v_from_idx is not None else 0.0)
        dVmag_to_dVk = 1.0 if to_ppc == gen_bus_ppc else (
            dx_dVk[self.n_theta + v_to_idx] if v_to_idx is not None else 0.0)

        dV_from_dVk = (dVmag_from_dVk + 1j * U_from * dTheta_from_dVk) * np.exp(1j * theta_from)
        dV_to_dVk = (dVmag_to_dVk + 1j * U_to * dTheta_to_dVk) * np.exp(1j * theta_to)

        dI_ij_dVk = Y_line * (dV_from_dVk - dV_to_dVk) + Y_shunt * dV_from_dVk
        dI_mag_dVk = (1.0 / I_mag) * np.real(np.conj(I_ij) * dI_ij_dVk)

        base_voltage_kv = self.net.bus.at[from_bus, 'vn_kv']
        I_base_kA = self.net.sn_mva / (np.sqrt(3) * base_voltage_kv)

        return dI_mag_dVk * I_base_kA

    def compute_dI_dVgen_matrix(
            self,
            line_indices: List[int],
            gen_bus_indices_pp: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        n_lines = len(line_indices)
        n_gens = len(gen_bus_indices_pp)
        if n_lines == 0 or n_gens == 0:
            return np.zeros((n_lines, n_gens)), [], []

        sensitivity_matrix = np.zeros((n_lines, n_gens))
        line_mapping = list(line_indices)
        gen_mapping = list(gen_bus_indices_pp)

        for j, gen_bus_pp in enumerate(gen_bus_indices_pp):
            gen_bus_ppc = pp_bus_to_ppc_bus(self.net, gen_bus_pp)
            for i, line_idx in enumerate(line_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dI_dVgen(line_idx, gen_bus_ppc)
                except ValueError:
                    sensitivity_matrix[i, j] = 0.0

        return sensitivity_matrix, line_mapping, gen_mapping


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

        # Negate to account for load-convention sign, then scale by step size.
        # Shunts are constant-susceptance devices: Q_shunt = B * V².
        # The q_step_mvar is the rated step at V=1 pu, so the actual
        # reactive power per step at the current operating point is
        # q_step_mvar * V_pu².
        V_pu = self.net.res_bus.at[shunt_bus_idx, 'vm_pu']
        dV_dQ_shunt = -dV_dQ[:, 0] * q_step_mvar * V_pu ** 2

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

        # Negate to account for load-convention sign, then scale by step size.
        # Shunts are constant-susceptance: Q_shunt = B * V².
        V_pu = self.net.res_bus.at[shunt_bus_idx, 'vm_pu']
        dI_dQ_shunt = -dI_dQ[:, 0] * q_step_mvar * V_pu ** 2

        return dI_dQ_shunt, line_map

    # =========================================================================
    # G. Three-Winding Transformer Sensitivities
    # =========================================================================

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
                - b * np.sin(theta_i - theta_j)
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
                - g * np.sin(theta_i - theta_j)
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
        d = _get_trafo3w_hv_branch_data(self.net, trafo3w_idx)
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
        d = _get_trafo3w_hv_branch_data(self.net, trafo3w_idx)

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

        Returns
        -------
        sensitivity : float
            ∂Q_HV_meas / ∂s_chg [Mvar per tap step].
        """
        # Data for the change transformer (for ∂g/∂τ)
        d_chg = _get_trafo3w_hv_branch_data(self.net, chg_trafo3w_idx)
        dg_dtau = self._compute_dg_dtau_3w(d_chg)

        # Data for the measurement transformer (for ∂Q_HV/∂x)
        d_meas = _get_trafo3w_hv_branch_data(self.net, meas_trafo3w_idx)
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

        return (indirect + direct)

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
        # Negate to account for load-convention sign of shunt.q_mvar.
        # Shunts are constant-susceptance: Q_shunt = B * V².
        V_pu = self.net.res_bus.at[shunt_bus_idx, 'vm_pu']
        return (
            -self.compute_dQtrafo3w_hv_dQ_der(trafo3w_idx, shunt_bus_idx)
            * q_step_mvar
            * V_pu ** 2
        )

    def compute_dQtrafo3w_hv_dVgen(
        self,
        trafo3w_idx: int,
        gen_bus_ppc: int,
        observation_bus_indices_for_gen: Optional[List[int]] = None,
    ) -> float:
        """
        Compute HV-side Q sensitivity of a 3W transformer to a PV
        generator voltage setpoint change.

        Uses the chain rule:

            ∂Q_HV/∂V_gen = ∂Q_HV/∂V_hv · ∂V_hv/∂V_gen
                          + ∂Q_HV/∂V_star · ∂V_star/∂V_gen

        where ``∂Q_HV/∂V_hv`` and ``∂Q_HV/∂V_star`` are the branch-
        level partial derivatives (same as in ``compute_dQtrafo3w_hv_dQ_der``)
        and ``∂V_bus/∂V_gen`` comes from ``compute_dV_dVgen``.

        Parameters
        ----------
        trafo3w_idx : int
            ``net.trafo3w`` index.
        gen_bus_ppc : int
            Pypower (ppc) bus index of the PV generator bus.

        Returns
        -------
        sensitivity : float
            ∂Q_HV / ∂V_gen [Mvar / p.u.].
        """
        d = _get_trafo3w_hv_branch_data(self.net, trafo3w_idx)

        U_i = d['U_hv']
        U_j = d['U_aux']
        theta_i = d['theta_hv']
        theta_j = d['theta_aux']
        tau = d['tau']
        g_val = d['g']
        b_val = d['b']

        # ∂Q_HV/∂V_hv and ∂Q_HV/∂V_star (identical to compute_dQtrafo3w_hv_dQ_der)
        dQ_dV_hv = (
            b_val * (-2.0 * U_i / tau**2
                     + U_j * np.cos(theta_i - theta_j) / tau)
            - g_val * U_j * np.sin(theta_i - theta_j) / tau
        )
        dQ_dV_star = (
            b_val * U_i * np.cos(theta_i - theta_j) / tau
            - g_val * U_i * np.sin(theta_i - theta_j) / tau
        )

        # ∂V/∂V_gen at hv_bus and aux_bus
        hv_bus_pp = int(self.net.trafo3w.at[trafo3w_idx, 'hv_bus'])
        # The aux bus is internal to pandapower (not in net.bus), so we
        # need to use the ppc bus index directly and extract from dx/dVk.
        # Re-use the full dV_dVgen computation and request both buses.
        #
        # For the HV bus: use compute_dV_dVgen with [hv_bus_pp]
        # For the aux bus: extract from the state vector directly
        v_hv_jac = d['v_hv_jac']
        v_aux_jac = d['v_aux_jac']

        # Run full sensitivity computation for this generator
        bus_data = self.net._ppc['bus']
        Ybus = np.array(self.net._ppc['internal']['Ybus'].todense())
        n_bus = bus_data.shape[0]
        V_complex = bus_data[:, 7] * np.exp(1j * np.deg2rad(bus_data[:, 8]))

        k = gen_bus_ppc
        V_k = V_complex[k]
        Vm = np.abs(V_complex)
        theta = np.angle(V_complex)
        G = np.real(Ybus)
        B = np.imag(Ybus)

        dP_dVk = np.zeros(n_bus)
        dQ_dVk = np.zeros(n_bus)
        for i in range(n_bus):
            cos_ik = np.cos(theta[i] - theta[k])
            sin_ik = np.sin(theta[i] - theta[k])
            if i == k:
                dP_dVk[i] = 2 * Vm[k] * G[k, k]
                dQ_dVk[i] = -2 * Vm[k] * B[k, k]
                for j_bus in range(n_bus):
                    if j_bus != k:
                        cos_kj = np.cos(theta[k] - theta[j_bus])
                        sin_kj = np.sin(theta[k] - theta[j_bus])
                        dP_dVk[i] += Vm[j_bus] * (G[k, j_bus] * cos_kj + B[k, j_bus] * sin_kj)
                        dQ_dVk[i] += Vm[j_bus] * (G[k, j_bus] * sin_kj - B[k, j_bus] * cos_kj)
            else:
                dP_dVk[i] = Vm[i] * (G[i, k] * cos_ik + B[i, k] * sin_ik)
                dQ_dVk[i] = Vm[i] * (G[i, k] * sin_ik - B[i, k] * cos_ik)

        pv_list = list(self.pv_buses)
        pq_list = list(self.pq_buses)
        dg_dVk = np.zeros(self.x_size)
        for idx_pv, bus_pv in enumerate(pv_list):
            dg_dVk[idx_pv] = dP_dVk[bus_pv]
        n_pv = len(pv_list)
        for idx_pq, bus_pq in enumerate(pq_list):
            dg_dVk[n_pv + idx_pq] = dP_dVk[bus_pq]
            dg_dVk[self.n_theta + idx_pq] = dQ_dVk[bus_pq]

        dx_dVk = -self.J_inv @ dg_dVk
        dV_full = dx_dVk[self.n_theta:]   # V_PQ part

        sensitivity = 0.0

        if v_hv_jac is not None and v_hv_jac < len(dV_full):
            dV_hv_dVgen = dV_full[v_hv_jac]
            sensitivity += dQ_dV_hv * dV_hv_dVgen
        else:
            # HV bus is a PV bus (generator bus itself) — dV/dVgen = 1
            # if it IS the generator, else check if slack
            if d['hv_bus_ppc'] == gen_bus_ppc:
                sensitivity += dQ_dV_hv * 1.0

        if v_aux_jac is not None and v_aux_jac < len(dV_full):
            dV_star_dVgen = dV_full[v_aux_jac]
            sensitivity += dQ_dV_star * dV_star_dVgen

        return sensitivity

    def compute_dQtrafo3w_hv_dVgen_matrix(
        self,
        trafo3w_indices: List[int],
        gen_bus_indices_pp: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute HV-side Q sensitivity matrix for multiple 3W transformers
        with respect to multiple PV generator voltage setpoints.

        Parameters
        ----------
        trafo3w_indices : List[int]
            ``net.trafo3w`` indices for measurement transformers.
        gen_bus_indices_pp : List[int]
            Pandapower bus indices of PV generator buses.

        Returns
        -------
        sensitivity_matrix : NDArray[np.float64]
            Shape ``(n_trafo3w, n_gen)`` [Mvar / p.u.].
        trafo3w_mapping : List[int]
        gen_bus_mapping : List[int]
        """
        n_t3w = len(trafo3w_indices)
        n_gen = len(gen_bus_indices_pp)
        if n_t3w == 0 or n_gen == 0:
            return np.zeros((n_t3w, n_gen)), [], []

        matrix = np.zeros((n_t3w, n_gen))
        for j, gen_bus_pp in enumerate(gen_bus_indices_pp):
            gen_bus_ppc = pp_bus_to_ppc_bus(self.net, gen_bus_pp)
            for i, t3w_idx in enumerate(trafo3w_indices):
                matrix[i, j] = self.compute_dQtrafo3w_hv_dVgen(
                    t3w_idx, gen_bus_ppc,
                )

        return matrix, list(trafo3w_indices), list(gen_bus_indices_pp)

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
            * ``'shunt_cached_v_pu'``: cached V_pu at shunt buses (for updater)

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
            dQtr2w_dQder, trafo_map, _ = self.compute_dQtrafo_dQder_2w_matrix(
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
                dV_col, _ = self.compute_dV_ds_2w(
                    trafo_idx=oltc_idx,
                    observation_bus_indices=observation_bus_indices,
                )
                # 2W trafo Q w.r.t. 2W OLTC
                dQtr2w_col = np.zeros(n_trafo2w_out)
                for i, mt in enumerate(trafo_map):
                    try:
                        dQtr2w_col[i] = self.compute_dQtrafo_2w_ds(
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
                #dI_col = np.zeros(n_line_out)
                dI_col = np.zeros(n_line_out)
                for i_line, line_idx in enumerate(line_indices):
                    try:
                        dI_col[i_line] = self.compute_dI_ds_2w(line_idx, oltc_idx)
                    except ValueError:
                        dI_col[i_line] = 0.0

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
                # Branch current w.r.t. 3W OLTC tap
                dI_col = np.zeros(n_line_out)
                for i_line, line_idx in enumerate(line_indices):
                    try:
                        dI_col[i_line] = self.compute_dI_ds_3w(line_idx, oltc3w_idx)
                    except ValueError:
                        dI_col[i_line] = 0.0

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
                # Shunts are constant-susceptance: Q_shunt = B * V².
                V_pu_shunt = self.net.res_bus.at[shunt_bus, 'vm_pu']
                dQtr2w_col = np.zeros(n_trafo2w_out)
                for i, mt in enumerate(trafo_map):
                    try:
                        # Negate for shunt load-convention sign
                        dQtr2w_col[i] = -self.compute_dQtrafo_dQder_2w(
                            trafo_idx=mt, der_bus_idx=shunt_bus
                        ) * q_step * V_pu_shunt ** 2
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
                # Branch current w.r.t. shunt.
                # compute_dI_dQ_shunt raises ValueError when line_indices is
                # empty (no lines → der_map empty).  Skip gracefully: the I
                # rows are zero-length anyway when there are no lines.
                if line_indices:
                    dI_col, _ = self.compute_dI_dQ_shunt(
                        shunt_bus_idx=shunt_bus,
                        line_indices=line_indices,
                        q_step_mvar=q_step,
                    )
                else:
                    dI_col = np.zeros(0, dtype=np.float64)

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

        # --- Off-diagonal scaling of dQtr3w_ds submatrix ---
        # The cross-transformer coupling terms (off-diagonal) are attenuated by
        # a tunable factor to account for modelling uncertainty.
        # Only applied when the block is square (nmeas == nchg == ntrafo3w).
        # _OFFDIAG_SCALE_DQTR3W_DS: float = 0.1  # Tuning parameter
        #
        # if n_oltc3w > 0 and n_trafo3w_out == n_oltc3w:
        #     block = H[row_q3w: row_q3w + n_trafo3w_out, col_oltc3w: col_oltc3w + n_oltc3w]
        #     H[row_q3w: row_q3w + n_trafo3w_out, col_oltc3w: col_oltc3w + n_oltc3w] = (
        #         _apply_offdiagonal_scaling(block, _OFFDIAG_SCALE_DQTR3W_DS)
        #     )

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

        # Cached voltages at shunt buses for SensitivityUpdater V² rescaling
        shunt_cached_v_pu = np.array([
            self.net.res_bus.at[bus, 'vm_pu'] for bus in shunt_map
        ]) if shunt_map else np.array([])

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
            'shunt_cached_v_pu': shunt_cached_v_pu,
        }

        return H, mappings
