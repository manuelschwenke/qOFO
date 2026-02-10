#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Sensitivity Module
============================

This module provides numerical sensitivity calculations using finite differences
and repeated power flow solutions. It is intended as a reference implementation
to validate the analytical Jacobian-based sensitivities and as a debugging tool.

All sensitivities are computed using the perturbation method:
    dY/dX ≈ (Y(X + δX) - Y(X)) / δX

where δX is a small perturbation applied to the input X, and power flow is
solved for both the perturbed and baseline states.

Author: Manuel Schwenke
Date: 2026-02-10

Notes
-----
- This is computationally expensive (requires N power flows for N inputs)
- Accuracy depends on perturbation size (too small → numerical errors,
  too large → linearisation errors)
- Recommended for validation and debugging only, not for production use
- Retains same interface as JacobianSensitivities for drop-in replacement
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Dict
import pandapower as pp
import copy
import warnings


class NumericalSensitivities:
    """
    Numerical sensitivity calculator using finite differences and power flow.
    
    This class computes all required sensitivities via the perturbation method.
    It provides the same interface as JacobianSensitivities and can be used
    as a drop-in replacement for debugging and validation.
    
    The numerical approach repeatedly solves power flow with small perturbations
    to compute sensitivities. This is robust and does not rely on Jacobian
    matrix structure, making it useful for identifying issues in the analytical
    implementation.
    
    Attributes
    ----------
    net_baseline : pp.pandapowerNet
        Deep copy of the network at the cached operating point.
    delta_q_mvar : float
        Perturbation size for reactive power injections [Mvar].
    delta_s_tap : float
        Perturbation size for OLTC tap positions [taps].
    delta_shunt_state : int
        Perturbation size for shunt state changes [states].
    
    Notes
    -----
    The network must have a converged power flow solution before initialisation.
    
    Typical perturbation sizes:
        - Q: 0.1 Mvar (0.1% of 100 MVA base)
        - s: 0.5 taps (small enough for linearisation, large enough vs. numerical noise)
        - shunt: 1 state
    """
    
    def __init__(
        self,
        net: pp.pandapowerNet,
        delta_q_mvar: float = 0.1,
        delta_s_tap: float = 0.5,
        delta_shunt_state: int = 1,
    ) -> None:
        """
        Initialise the numerical sensitivity calculator from a converged network.
        
        Parameters
        ----------
        net : pp.pandapowerNet
            Pandapower network with converged power flow solution.
        delta_q_mvar : float, optional
            Perturbation size for Q injections [Mvar] (default: 0.1).
        delta_s_tap : float, optional
            Perturbation size for tap positions [taps] (default: 0.5).
        delta_shunt_state : int, optional
            Perturbation size for shunt states (default: 1).
        
        Raises
        ------
        ValueError
            If the network has not converged.
        """
        if not net.converged:
            raise ValueError(
                "Network power flow must have converged before initialising "
                "numerical sensitivities."
            )
        
        # Store deep copy of baseline network state
        self.net_baseline = copy.deepcopy(net)
        
        # Perturbation sizes
        self.delta_q_mvar = delta_q_mvar
        self.delta_s_tap = delta_s_tap
        self.delta_shunt_state = delta_shunt_state
        
        # Network base power (for per-unit conversions)
        self.sn_mva = self.net_baseline.sn_mva
    
    def _run_power_flow(self, net: pp.pandapowerNet) -> bool:
        """
        Run power flow on a network copy.
        
        Parameters
        ----------
        net : pp.pandapowerNet
            Network to solve.
        
        Returns
        -------
        converged : bool
            True if power flow converged, False otherwise.
        """
        try:
            pp.runpp(net, run_control=False, calculate_voltage_angles=True,
                     numba=False, lightsim2grid=False)
            return net.converged
        except Exception as e:
            warnings.warn(
                f"Power flow failed during numerical sensitivity computation: {e}"
            )
            return False
    
    # =========================================================================
    # A. Bus Voltage to DER Reactive Power Infeed Sensitivity
    # =========================================================================
    
    def compute_dV_dQ_der(
        self,
        der_bus_indices: List[int],
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute bus voltage sensitivity to DER reactive power injection.
        
        Uses finite differences: perturb Q at each DER bus, solve power flow,
        measure voltage changes at observation buses.
        
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
            Units: [p.u. / Mvar].
        observation_bus_mapping : List[int]
            Ordered list of observation bus indices actually included.
        der_bus_mapping : List[int]
            Ordered list of DER bus indices actually included.
        
        Raises
        ------
        ValueError
            If no valid buses are found in either list.
        """
        n_obs = len(observation_bus_indices)
        n_der = len(der_bus_indices)
        
        if n_obs == 0:
            raise ValueError("No observation buses provided.")
        if n_der == 0:
            raise ValueError("No DER buses provided.")
        
        sensitivity_matrix = np.zeros((n_obs, n_der))
        
        # Baseline voltages
        v_baseline = np.array([
            self.net_baseline.res_bus.at[b, 'vm_pu']
            for b in observation_bus_indices
        ])
        
        # Perturb each DER bus and measure voltage changes
        for j, der_bus in enumerate(der_bus_indices):
            # Create perturbed network
            net_pert = copy.deepcopy(self.net_baseline)
            
            # Find all sgens at this bus and perturb their Q
            sgen_mask = net_pert.sgen['bus'] == der_bus
            n_sgens = sgen_mask.sum()
            
            if n_sgens == 0:
                # No sgens at this bus - set sensitivity to zero
                sensitivity_matrix[:, j] = 0.0
                continue
            
            # Distribute perturbation equally across all sgens at this bus
            delta_q_per_sgen = self.delta_q_mvar / n_sgens
            for sidx in net_pert.sgen.index[sgen_mask]:
                net_pert.sgen.at[sidx, 'q_mvar'] += delta_q_per_sgen
            
            # Solve perturbed power flow
            converged = self._run_power_flow(net_pert)
            
            if not converged:
                warnings.warn(
                    f"Power flow did not converge for DER Q perturbation at bus {der_bus}. "
                    f"Setting sensitivity to zero."
                )
                sensitivity_matrix[:, j] = 0.0
                continue
            
            # Measure voltage changes
            v_pert = np.array([
                net_pert.res_bus.at[b, 'vm_pu']
                for b in observation_bus_indices
            ])
            
            # Compute numerical derivative
            sensitivity_matrix[:, j] = (v_pert - v_baseline) / self.delta_q_mvar
        
        return sensitivity_matrix, observation_bus_indices, der_bus_indices
    
    # =========================================================================
    # B. Bus Voltage to OLTC Position Sensitivity (2W transformers)
    # =========================================================================
    
    def compute_dV_ds(
        self,
        trafo_idx: int,
        observation_bus_indices: List[int],
        delta_s: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute bus voltage sensitivity to 2W transformer tap position change.
        
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
        if trafo_idx not in self.net_baseline.trafo.index:
            raise ValueError(f"Transformer {trafo_idx} not found in network.")
        
        n_obs = len(observation_bus_indices)
        if n_obs == 0:
            raise ValueError("No observation buses provided.")
        
        # Baseline voltages
        v_baseline = np.array([
            self.net_baseline.res_bus.at[b, 'vm_pu']
            for b in observation_bus_indices
        ])
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb tap position
        s_orig = net_pert.trafo.at[trafo_idx, 'tap_pos']
        net_pert.trafo.at[trafo_idx, 'tap_pos'] = s_orig + self.delta_s_tap
        
        # Check tap limits
        tap_min = net_pert.trafo.at[trafo_idx, 'tap_min']
        tap_max = net_pert.trafo.at[trafo_idx, 'tap_max']
        if net_pert.trafo.at[trafo_idx, 'tap_pos'] > tap_max:
            net_pert.trafo.at[trafo_idx, 'tap_pos'] = tap_max
            actual_delta_s = tap_max - s_orig
        elif net_pert.trafo.at[trafo_idx, 'tap_pos'] < tap_min:
            net_pert.trafo.at[trafo_idx, 'tap_pos'] = tap_min
            actual_delta_s = tap_min - s_orig
        else:
            actual_delta_s = self.delta_s_tap
        
        if abs(actual_delta_s) < 1e-6:
            # No perturbation possible
            return np.zeros(n_obs), observation_bus_indices
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for OLTC perturbation at trafo {trafo_idx}. "
                f"Setting sensitivity to zero."
            )
            return np.zeros(n_obs), observation_bus_indices
        
        # Measure voltage changes
        v_pert = np.array([
            net_pert.res_bus.at[b, 'vm_pu']
            for b in observation_bus_indices
        ])
        
        # Compute numerical derivative
        dV_ds = (v_pert - v_baseline) / actual_delta_s
        
        return dV_ds, observation_bus_indices
    
    def compute_dV_ds_matrix(
        self,
        trafo_indices: List[int],
        observation_bus_indices: List[int],
    ) -> Tuple[NDArray[np.float64], List[int], List[int]]:
        """
        Compute voltage sensitivity matrix for multiple 2W transformers.
        
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
            return np.zeros((n_obs, n_trafo)), observation_bus_indices, trafo_indices
        
        sensitivity_matrix = np.zeros((n_obs, n_trafo))
        
        for j, trafo_idx in enumerate(trafo_indices):
            try:
                dV_ds_col, _ = self.compute_dV_ds(
                    trafo_idx=trafo_idx,
                    observation_bus_indices=observation_bus_indices,
                    delta_s=1.0,
                )
                sensitivity_matrix[:, j] = dV_ds_col
            except ValueError as e:
                warnings.warn(f"Failed to compute dV/ds for trafo {trafo_idx}: {e}")
                sensitivity_matrix[:, j] = 0.0
        
        return sensitivity_matrix, observation_bus_indices, trafo_indices
    
    # =========================================================================
    # C. Transformer Q to DER Reactive Power Infeed Sensitivity
    # =========================================================================
    
    def compute_dQtrafo_dQ_der(
        self,
        trafo_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute 2W transformer reactive power flow sensitivity to DER reactive power.
        
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
        if trafo_idx not in self.net_baseline.trafo.index:
            raise ValueError(f"Transformer {trafo_idx} not found in network.")
        
        # Baseline Q flow (HV side)
        q_baseline = self.net_baseline.res_trafo.at[trafo_idx, 'q_hv_mvar']
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb DER Q
        sgen_mask = net_pert.sgen['bus'] == der_bus_idx
        n_sgens = sgen_mask.sum()
        
        if n_sgens == 0:
            raise ValueError(f"No sgens found at DER bus {der_bus_idx}.")
        
        delta_q_per_sgen = self.delta_q_mvar / n_sgens
        for sidx in net_pert.sgen.index[sgen_mask]:
            net_pert.sgen.at[sidx, 'q_mvar'] += delta_q_per_sgen
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for dQtrafo/dQder computation "
                f"(trafo {trafo_idx}, DER bus {der_bus_idx}). Setting to zero."
            )
            return 0.0
        
        # Measure Q flow change
        q_pert = net_pert.res_trafo.at[trafo_idx, 'q_hv_mvar']
        
        # Compute numerical derivative
        sensitivity = (q_pert - q_baseline) / self.delta_q_mvar
        
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
            return np.zeros((n_trafo, n_der)), trafo_indices, der_bus_indices
        
        sensitivity_matrix = np.zeros((n_trafo, n_der))
        
        for i, trafo_idx in enumerate(trafo_indices):
            for j, der_bus_idx in enumerate(der_bus_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dQtrafo_dQ_der(
                        trafo_idx, der_bus_idx
                    )
                except ValueError as e:
                    warnings.warn(
                        f"Failed dQtrafo/dQder for trafo {trafo_idx}, "
                        f"DER bus {der_bus_idx}: {e}"
                    )
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, trafo_indices, der_bus_indices
    
    # =========================================================================
    # D. Transformer Q to OLTC Position Sensitivity
    # =========================================================================
    
    def compute_dQtrafo_ds(
        self,
        meas_trafo_idx: int,
        chg_trafo_idx: int,
        delta_s: float = 1.0,
    ) -> float:
        """
        Compute 2W transformer reactive power flow sensitivity to tap position change.
        
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
        if meas_trafo_idx not in self.net_baseline.trafo.index:
            raise ValueError(f"Measurement transformer {meas_trafo_idx} not found.")
        if chg_trafo_idx not in self.net_baseline.trafo.index:
            raise ValueError(f"Change transformer {chg_trafo_idx} not found.")
        
        # Baseline Q flow
        q_baseline = self.net_baseline.res_trafo.at[meas_trafo_idx, 'q_hv_mvar']
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb tap position
        s_orig = net_pert.trafo.at[chg_trafo_idx, 'tap_pos']
        net_pert.trafo.at[chg_trafo_idx, 'tap_pos'] = s_orig + self.delta_s_tap
        
        # Check tap limits
        tap_min = net_pert.trafo.at[chg_trafo_idx, 'tap_min']
        tap_max = net_pert.trafo.at[chg_trafo_idx, 'tap_max']
        if net_pert.trafo.at[chg_trafo_idx, 'tap_pos'] > tap_max:
            net_pert.trafo.at[chg_trafo_idx, 'tap_pos'] = tap_max
            actual_delta_s = tap_max - s_orig
        elif net_pert.trafo.at[chg_trafo_idx, 'tap_pos'] < tap_min:
            net_pert.trafo.at[chg_trafo_idx, 'tap_pos'] = tap_min
            actual_delta_s = tap_min - s_orig
        else:
            actual_delta_s = self.delta_s_tap
        
        if abs(actual_delta_s) < 1e-6:
            return 0.0
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for dQtrafo/ds computation "
                f"(meas trafo {meas_trafo_idx}, chg trafo {chg_trafo_idx}). "
                f"Setting to zero."
            )
            return 0.0
        
        # Measure Q flow change
        q_pert = net_pert.res_trafo.at[meas_trafo_idx, 'q_hv_mvar']
        
        # Compute numerical derivative
        sensitivity = (q_pert - q_baseline) / actual_delta_s
        
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
        
        for i, meas_idx in enumerate(trafo_indices):
            for j, chg_idx in enumerate(trafo_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dQtrafo_ds(
                        meas_idx, chg_idx, delta_s=1.0
                    )
                except ValueError as e:
                    warnings.warn(
                        f"Failed dQtrafo/ds for meas trafo {meas_idx}, "
                        f"chg trafo {chg_idx}: {e}"
                    )
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, trafo_indices
    
    # =========================================================================
    # E. Branch Current to DER Reactive Power Infeed Sensitivity
    # =========================================================================
    
    def compute_dI_dQ_der(
        self,
        line_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute branch current magnitude sensitivity to DER reactive power.
        
        Parameters
        ----------
        line_idx : int
            Pandapower line index.
        der_bus_idx : int
            Pandapower bus index where the DER is connected.
        
        Returns
        -------
        sensitivity : float
            Sensitivity ∂|I|/∂Q in [kA/Mvar].
        
        Raises
        ------
        ValueError
            If line or DER bus is not found or cannot be processed.
        """
        if line_idx not in self.net_baseline.line.index:
            raise ValueError(f"Line {line_idx} not found in network.")
        
        # Baseline current magnitude
        i_baseline = self.net_baseline.res_line.at[line_idx, 'i_from_ka']
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb DER Q
        sgen_mask = net_pert.sgen['bus'] == der_bus_idx
        n_sgens = sgen_mask.sum()
        
        if n_sgens == 0:
            raise ValueError(f"No sgens found at DER bus {der_bus_idx}.")
        
        delta_q_per_sgen = self.delta_q_mvar / n_sgens
        for sidx in net_pert.sgen.index[sgen_mask]:
            net_pert.sgen.at[sidx, 'q_mvar'] += delta_q_per_sgen
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for dI/dQder computation "
                f"(line {line_idx}, DER bus {der_bus_idx}). Setting to zero."
            )
            return 0.0
        
        # Measure current change
        i_pert = net_pert.res_line.at[line_idx, 'i_from_ka']
        
        # Compute numerical derivative
        sensitivity = (i_pert - i_baseline) / self.delta_q_mvar
        
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
            return np.zeros((n_lines, n_der)), line_indices, der_bus_indices
        
        sensitivity_matrix = np.zeros((n_lines, n_der))
        
        for i, line_idx in enumerate(line_indices):
            for j, der_bus_idx in enumerate(der_bus_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dI_dQ_der(
                        line_idx, der_bus_idx
                    )
                except ValueError as e:
                    warnings.warn(
                        f"Failed dI/dQder for line {line_idx}, "
                        f"DER bus {der_bus_idx}: {e}"
                    )
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, line_indices, der_bus_indices
    
    # =========================================================================
    # F. Shunt Reactive Power Sensitivity
    # =========================================================================
    
    def compute_dV_dQ_shunt(
        self,
        shunt_bus_idx: int,
        observation_bus_indices: List[int],
        q_step_mvar: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute voltage sensitivity to shunt reactive power injection.
        
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
        # Shunt sensitivity is computed via DER Q sensitivity
        # (shunt is modelled as Q injection)
        dV_dQ, obs_map, _ = self.compute_dV_dQ_der(
            der_bus_indices=[shunt_bus_idx],
            observation_bus_indices=observation_bus_indices,
        )
        
        # Negate to account for load-convention sign
        # (pandapower shunts: positive q_mvar = absorbing)
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
        dI_dQ, line_map, _ = self.compute_dI_dQ_der_matrix(
            line_indices=line_indices,
            der_bus_indices=[shunt_bus_idx],
        )
        
        # Negate to account for load-convention sign
        dI_dQ_shunt = -dI_dQ[:, 0] * q_step_mvar
        
        return dI_dQ_shunt, line_map
    
    # =========================================================================
    # G. Three-Winding Transformer Sensitivities
    # =========================================================================
    
    def compute_dV_ds_trafo3w(
        self,
        trafo3w_idx: int,
        observation_bus_indices: List[int],
        delta_s: float = 1.0,
    ) -> Tuple[NDArray[np.float64], List[int]]:
        """
        Compute bus voltage sensitivity to a 3W transformer OLTC tap change.
        
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
        if trafo3w_idx not in self.net_baseline.trafo3w.index:
            raise ValueError(f"3W transformer {trafo3w_idx} not found in network.")
        
        n_obs = len(observation_bus_indices)
        if n_obs == 0:
            raise ValueError("No observation buses provided.")
        
        # Baseline voltages
        v_baseline = np.array([
            self.net_baseline.res_bus.at[b, 'vm_pu']
            for b in observation_bus_indices
        ])
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb tap position
        s_orig = net_pert.trafo3w.at[trafo3w_idx, 'tap_pos']
        net_pert.trafo3w.at[trafo3w_idx, 'tap_pos'] = s_orig + self.delta_s_tap
        
        # Check tap limits
        tap_min = net_pert.trafo3w.at[trafo3w_idx, 'tap_min']
        tap_max = net_pert.trafo3w.at[trafo3w_idx, 'tap_max']
        if net_pert.trafo3w.at[trafo3w_idx, 'tap_pos'] > tap_max:
            net_pert.trafo3w.at[trafo3w_idx, 'tap_pos'] = tap_max
            actual_delta_s = tap_max - s_orig
        elif net_pert.trafo3w.at[trafo3w_idx, 'tap_pos'] < tap_min:
            net_pert.trafo3w.at[trafo3w_idx, 'tap_pos'] = tap_min
            actual_delta_s = tap_min - s_orig
        else:
            actual_delta_s = self.delta_s_tap
        
        if abs(actual_delta_s) < 1e-6:
            return np.zeros(n_obs), observation_bus_indices
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for 3W OLTC perturbation "
                f"at trafo3w {trafo3w_idx}. Setting sensitivity to zero."
            )
            return np.zeros(n_obs), observation_bus_indices
        
        # Measure voltage changes
        v_pert = np.array([
            net_pert.res_bus.at[b, 'vm_pu']
            for b in observation_bus_indices
        ])
        
        # Compute numerical derivative
        dV_ds = (v_pert - v_baseline) / actual_delta_s
        
        return dV_ds, observation_bus_indices
    
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
            return np.zeros((n_obs, n_t3w)), observation_bus_indices, trafo3w_indices
        
        sensitivity_matrix = np.zeros((n_obs, n_t3w))
        
        for j, t3w_idx in enumerate(trafo3w_indices):
            try:
                dV_ds_col, _ = self.compute_dV_ds_trafo3w(
                    trafo3w_idx=t3w_idx,
                    observation_bus_indices=observation_bus_indices,
                )
                sensitivity_matrix[:, j] = dV_ds_col
            except ValueError as e:
                warnings.warn(
                    f"Failed dV/ds for 3W trafo {t3w_idx}: {e}"
                )
                sensitivity_matrix[:, j] = 0.0
        
        return sensitivity_matrix, observation_bus_indices, trafo3w_indices
    
    def compute_dQtrafo3w_hv_dQ_der(
        self,
        trafo3w_idx: int,
        der_bus_idx: int,
    ) -> float:
        """
        Compute HV-side reactive power sensitivity of a 3W transformer
        to DER reactive power injection.
        
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
        if trafo3w_idx not in self.net_baseline.trafo3w.index:
            raise ValueError(f"3W transformer {trafo3w_idx} not found in network.")
        
        # Baseline Q flow (HV side)
        q_baseline = self.net_baseline.res_trafo3w.at[trafo3w_idx, 'q_hv_mvar']
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb DER Q
        sgen_mask = net_pert.sgen['bus'] == der_bus_idx
        n_sgens = sgen_mask.sum()
        
        if n_sgens == 0:
            raise ValueError(f"No sgens found at DER bus {der_bus_idx}.")
        
        delta_q_per_sgen = self.delta_q_mvar / n_sgens
        for sidx in net_pert.sgen.index[sgen_mask]:
            net_pert.sgen.at[sidx, 'q_mvar'] += delta_q_per_sgen
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for dQtrafo3w_hv/dQder computation "
                f"(trafo3w {trafo3w_idx}, DER bus {der_bus_idx}). Setting to zero."
            )
            return 0.0
        
        # Measure Q flow change
        q_pert = net_pert.res_trafo3w.at[trafo3w_idx, 'q_hv_mvar']
        
        # Compute numerical derivative
        sensitivity = (q_pert - q_baseline) / self.delta_q_mvar
        
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
            return np.zeros((n_t3w, n_der)), trafo3w_indices, der_bus_indices
        
        sensitivity_matrix = np.zeros((n_t3w, n_der))
        
        for i, t3w_idx in enumerate(trafo3w_indices):
            for j, der_bus in enumerate(der_bus_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dQtrafo3w_hv_dQ_der(
                        t3w_idx, der_bus
                    )
                except ValueError as e:
                    warnings.warn(
                        f"Failed dQtrafo3w_hv/dQder for trafo3w {t3w_idx}, "
                        f"DER bus {der_bus}: {e}"
                    )
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, trafo3w_indices, der_bus_indices
    
    def compute_dQtrafo3w_hv_ds(
        self,
        meas_trafo3w_idx: int,
        chg_trafo3w_idx: int,
        delta_s: float = 1.0,
    ) -> float:
        """
        Compute HV-side Q sensitivity of a 3W transformer to a 3W OLTC
        tap position change.
        
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
        if meas_trafo3w_idx not in self.net_baseline.trafo3w.index:
            raise ValueError(
                f"Measurement 3W transformer {meas_trafo3w_idx} not found."
            )
        if chg_trafo3w_idx not in self.net_baseline.trafo3w.index:
            raise ValueError(
                f"Change 3W transformer {chg_trafo3w_idx} not found."
            )
        
        # Baseline Q flow
        q_baseline = self.net_baseline.res_trafo3w.at[meas_trafo3w_idx, 'q_hv_mvar']
        
        # Create perturbed network
        net_pert = copy.deepcopy(self.net_baseline)
        
        # Perturb tap position
        s_orig = net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_pos']
        net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_pos'] = s_orig + self.delta_s_tap
        
        # Check tap limits
        tap_min = net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_min']
        tap_max = net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_max']
        if net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_pos'] > tap_max:
            net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_pos'] = tap_max
            actual_delta_s = tap_max - s_orig
        elif net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_pos'] < tap_min:
            net_pert.trafo3w.at[chg_trafo3w_idx, 'tap_pos'] = tap_min
            actual_delta_s = tap_min - s_orig
        else:
            actual_delta_s = self.delta_s_tap
        
        if abs(actual_delta_s) < 1e-6:
            return 0.0
        
        # Solve perturbed power flow
        converged = self._run_power_flow(net_pert)
        
        if not converged:
            warnings.warn(
                f"Power flow did not converge for dQtrafo3w_hv/ds computation "
                f"(meas trafo3w {meas_trafo3w_idx}, chg trafo3w {chg_trafo3w_idx}). "
                f"Setting to zero."
            )
            return 0.0
        
        # Measure Q flow change
        q_pert = net_pert.res_trafo3w.at[meas_trafo3w_idx, 'q_hv_mvar']
        
        # Compute numerical derivative
        sensitivity = (q_pert - q_baseline) / actual_delta_s
        
        return sensitivity
    
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
            return np.zeros((n_meas, n_chg)), meas_trafo3w_indices, chg_trafo3w_indices
        
        sensitivity_matrix = np.zeros((n_meas, n_chg))
        
        for i, m_idx in enumerate(meas_trafo3w_indices):
            for j, c_idx in enumerate(chg_trafo3w_indices):
                try:
                    sensitivity_matrix[i, j] = self.compute_dQtrafo3w_hv_ds(
                        m_idx, c_idx
                    )
                except ValueError as e:
                    warnings.warn(
                        f"Failed dQtrafo3w_hv/ds for meas trafo3w {m_idx}, "
                        f"chg trafo3w {c_idx}: {e}"
                    )
                    sensitivity_matrix[i, j] = 0.0
        
        return sensitivity_matrix, meas_trafo3w_indices, chg_trafo3w_indices
    
    def compute_dQtrafo3w_hv_dQ_shunt(
        self,
        trafo3w_idx: int,
        shunt_bus_idx: int,
        q_step_mvar: float = 1.0,
    ) -> float:
        """
        Compute HV-side Q sensitivity of a 3W transformer to a shunt
        reactive power injection.
        
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
        # Negate to account for load-convention sign
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
        
        This constructs the matrix used in the OFO optimisation problem
        using numerical finite differences.
        
        Input vector u = [Q_DER | s_OLTC_2w | s_OLTC_3w | state_shunt]^T
        Output vector y = [Q_trafo_2w | Q_trafo3w_hv | V_bus | I_branch]^T
        
        Parameters
        ----------
        der_bus_indices : List[int]
            Pandapower bus indices where DERs are connected.
        observation_bus_indices : List[int]
            Pandapower bus indices for voltage measurements.
        line_indices : List[int]
            Pandapower line indices for current measurements.
        trafo_indices : List[int], optional
            ``net.trafo`` indices for Q measurements (2W transformers).
        trafo3w_indices : List[int], optional
            ``net.trafo3w`` indices for HV-side Q measurements (3W transformers).
        oltc_trafo_indices : List[int], optional
            ``net.trafo`` indices with OLTC (2W).
        oltc_trafo3w_indices : List[int], optional
            ``net.trafo3w`` indices with OLTC (3W).
        shunt_bus_indices : List[int], optional
            Pandapower bus indices where switchable shunts are connected.
        shunt_q_steps_mvar : List[float], optional
            Reactive power step per state change for each shunt [Mvar].
        
        Returns
        -------
        H : NDArray[np.float64]
            Combined sensitivity matrix of shape ``(n_outputs, n_inputs)``.
        mappings : dict
            Dictionary containing the index mappings (same as analytical version).
        
        Raises
        ------
        ValueError
            If no valid inputs or outputs are found.
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
        
        print("[Numerical] Computing sensitivities via finite differences ...")
        print(f"  Perturbations: ΔQ={self.delta_q_mvar} Mvar, Δs={self.delta_s_tap} taps")
        
        # Part 1: DER Q sensitivities
        print("  [1/7] Computing ∂(Q_trafo_2w)/∂(Q_DER) ...")
        if trafo_indices:
            dQtr2w_dQder, trafo_map, _ = self.compute_dQtrafo_dQ_der_matrix(
                trafo_indices, der_bus_indices
            )
        else:
            trafo_map = []
            dQtr2w_dQder = np.zeros((0, len(der_bus_indices)))
        
        print("  [2/7] Computing ∂(Q_trafo3w_hv)/∂(Q_DER) ...")
        if trafo3w_indices:
            dQtr3w_dQder, t3w_map, _ = self.compute_dQtrafo3w_hv_dQ_der_matrix(
                trafo3w_indices, der_bus_indices
            )
        else:
            t3w_map = []
            dQtr3w_dQder = np.zeros((0, len(der_bus_indices)))
        
        print("  [3/7] Computing ∂V/∂(Q_DER) ...")
        dV_dQder, obs_bus_map, _ = self.compute_dV_dQ_der(
            der_bus_indices, observation_bus_indices
        )
        
        print("  [4/7] Computing ∂I/∂(Q_DER) ...")
        dI_dQder, line_map, _ = self.compute_dI_dQ_der_matrix(
            line_indices, der_bus_indices
        )
        
        n_der = len(der_bus_indices)
        n_trafo2w_out = dQtr2w_dQder.shape[0]
        n_trafo3w_out = dQtr3w_dQder.shape[0]
        n_bus_out = dV_dQder.shape[0]
        n_line_out = dI_dQder.shape[0]
        n_outputs = n_trafo2w_out + n_trafo3w_out + n_bus_out + n_line_out
        
        # Row offsets
        row_q2w = 0
        row_q3w = n_trafo2w_out
        row_v = n_trafo2w_out + n_trafo3w_out
        row_i = row_v + n_bus_out
        
        # Part 2: 2W OLTC sensitivities
        print("  [5/7] Computing 2W OLTC sensitivities ...")
        dV_ds2w_mat, _, oltc2w_map = self.compute_dV_ds_matrix(
            oltc_trafo_indices, observation_bus_indices
        )
        
        dQtr2w_ds2w_mat, _ = self.compute_dQtrafo_ds_matrix(oltc_trafo_indices)
        
        # 3W Q w.r.t. 2W OLTC (cross-type, set to zero)
        dQtr3w_ds2w_mat = np.zeros((n_trafo3w_out, len(oltc2w_map)))
        
        # Current w.r.t. 2W OLTC (not implemented)
        dI_ds2w_mat = np.zeros((n_line_out, len(oltc2w_map)))
        
        # Part 3: 3W OLTC sensitivities
        print("  [6/7] Computing 3W OLTC sensitivities ...")
        dV_ds3w_mat, _, oltc3w_map = self.compute_dV_ds_trafo3w_matrix(
            oltc_trafo3w_indices, observation_bus_indices
        )
        
        # 2W Q w.r.t. 3W OLTC (cross-type, set to zero)
        dQtr2w_ds3w_mat = np.zeros((n_trafo2w_out, len(oltc3w_map)))
        
        dQtr3w_ds3w_mat, _, _ = self.compute_dQtrafo3w_hv_ds_matrix(
            trafo3w_indices if trafo3w_indices else [], oltc3w_map
        )
        
        # Current w.r.t. 3W OLTC (not implemented)
        dI_ds3w_mat = np.zeros((n_line_out, len(oltc3w_map)))
        
        # Part 4: Shunt sensitivities
        print("  [7/7] Computing shunt sensitivities ...")
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
            except ValueError as e:
                warnings.warn(f"Failed shunt sensitivity for bus {shunt_bus}: {e}")
                continue
        
        # Part 5: Assemble H matrix
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
        
        # DER Q columns
        H[row_q2w:row_q2w + n_trafo2w_out, col_der:col_der + n_der] = dQtr2w_dQder
        H[row_q3w:row_q3w + n_trafo3w_out, col_der:col_der + n_der] = dQtr3w_dQder
        H[row_v:row_v + n_bus_out, col_der:col_der + n_der] = dV_dQder
        H[row_i:row_i + n_line_out, col_der:col_der + n_der] = dI_dQder
        
        # 2W OLTC columns
        if n_oltc2w > 0:
            H[row_q2w:row_q2w + n_trafo2w_out, col_oltc2w:col_oltc2w + n_oltc2w] = dQtr2w_ds2w_mat
            H[row_q3w:row_q3w + n_trafo3w_out, col_oltc2w:col_oltc2w + n_oltc2w] = dQtr3w_ds2w_mat
            H[row_v:row_v + n_bus_out, col_oltc2w:col_oltc2w + n_oltc2w] = dV_ds2w_mat
            H[row_i:row_i + n_line_out, col_oltc2w:col_oltc2w + n_oltc2w] = dI_ds2w_mat
        
        # 3W OLTC columns
        if n_oltc3w > 0:
            H[row_q2w:row_q2w + n_trafo2w_out, col_oltc3w:col_oltc3w + n_oltc3w] = dQtr2w_ds3w_mat
            H[row_q3w:row_q3w + n_trafo3w_out, col_oltc3w:col_oltc3w + n_oltc3w] = dQtr3w_ds3w_mat
            H[row_v:row_v + n_bus_out, col_oltc3w:col_oltc3w + n_oltc3w] = dV_ds3w_mat
            H[row_i:row_i + n_line_out, col_oltc3w:col_oltc3w + n_oltc3w] = dI_ds3w_mat
        
        # Shunt columns
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
        
        print(f"[Numerical] Sensitivity matrix H: shape {H.shape}")
        print(f"  Inputs: {n_der} DER + {n_oltc2w} OLTC2w + {n_oltc3w} OLTC3w + {n_shunt_actual} shunt")
        print(f"  Outputs: {n_trafo2w_out} Q2w + {n_trafo3w_out} Q3w + {n_bus_out} V + {n_line_out} I")
        
        return H, mappings
