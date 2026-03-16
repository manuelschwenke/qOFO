#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Winding Transformer Sensitivity Validation Test
======================================================

This script validates the analytical sensitivities computed in jacobian.py
by comparing them against numerical sensitivities computed using finite
differences (perturbation method).

The following sensitivities are tested:
1. dQ_trafo3w_hv/ds: HV reactive power w.r.t. tap position changes
2. dQ_trafo3w_hv/dQ_der: HV reactive power w.r.t. DER reactive power
3. dV/ds_trafo3w: Bus voltages w.r.t. 3W transformer tap positions
4. dV/dQ_der: Bus voltages w.r.t. DER reactive power (for reference)

Test Method
-----------
For each sensitivity, the test:
1. Computes analytical sensitivity from Jacobian (cached state)
2. Perturbs the input variable by ±ε and runs power flow
3. Computes numerical sensitivity via finite differences
4. Compares analytical vs. numerical and reports errors

Author: Manuel Schwenke
Date: 2026-02-11
Version: 1.1 (Fixed unpacking error and complex warnings)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple
import pandapower as pp
import copy

# Import network builder and sensitivity calculator
from network.build_tuda_net import build_tuda_net
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.index_helper import get_ppc_trafo3w_branch_indices


# ==============================================================================
# NUMERICAL SENSITIVITY COMPUTATION (FINITE DIFFERENCES)
# ==============================================================================

def compute_numerical_dQtrafo3w_ds(
    net: pp.pandapowerNet,
    trafo3w_idx: int,
    perturbation: int = 1,
) -> float:
    """
    Compute numerical sensitivity dQ_HV/ds using finite differences.

    Uses central difference: dQ/ds ≈ (Q(s+Δs) - Q(s-Δs)) / (2Δs)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    trafo3w_idx : int
        Three-winding transformer index.
    perturbation : int
        Tap position perturbation (default: ±1 tap step).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity dQ_HV/ds [Mvar/tap].

    Raises
    ------
    RuntimeError
        If power flow fails for perturbed states.
    """
    if trafo3w_idx not in net.trafo3w.index:
        raise ValueError(f"Transformer {trafo3w_idx} not found in net.trafo3w.")

    # Store original tap position
    s_original = int(net.trafo3w.at[trafo3w_idx, "tap_pos"])

    # Get tap bounds
    s_min = int(net.trafo3w.at[trafo3w_idx, "tap_min"])
    s_max = int(net.trafo3w.at[trafo3w_idx, "tap_max"])

    # Check if perturbation is feasible
    if s_original + perturbation > s_max or s_original - perturbation < s_min:
        raise ValueError(
            f"Perturbation ±{perturbation} exceeds tap limits "
            f"[{s_min}, {s_max}] for tap position {s_original}."
        )

    # Create deep copies for perturbation
    net_plus = copy.deepcopy(net)
    net_minus = copy.deepcopy(net)

    # Perturb tap position: s + Δs
    net_plus.trafo3w.at[trafo3w_idx, "tap_pos"] = s_original + perturbation
    try:
        pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s + Δs: {e}")

    if not net_plus.converged:
        raise RuntimeError("Power flow did not converge for s + Δs.")

    q_hv_plus = float(net_plus.res_trafo3w.at[trafo3w_idx, "q_hv_mvar"])

    # Perturb tap position: s - Δs
    net_minus.trafo3w.at[trafo3w_idx, "tap_pos"] = s_original - perturbation
    try:
        pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s - Δs: {e}")

    if not net_minus.converged:
        raise RuntimeError("Power flow did not converge for s - Δs.")

    q_hv_minus = float(net_minus.res_trafo3w.at[trafo3w_idx, "q_hv_mvar"])

    # Compute finite difference
    dQ_ds_numerical = (q_hv_plus - q_hv_minus) / (2.0 * perturbation)

    return dQ_ds_numerical


def compute_numerical_dQtrafo3w_dQder(
    net: pp.pandapowerNet,
    trafo3w_idx: int,
    der_bus_idx: int,
    perturbation_mvar: float = 1.0,
) -> float:
    """
    Compute numerical sensitivity dQ_HV/dQ_DER using finite differences.

    Uses central difference: dQ/dQ_DER ≈ (Q(Q_DER+ΔQ) - Q(Q_DER-ΔQ)) / (2ΔQ)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    trafo3w_idx : int
        Three-winding transformer index (measurement point).
    der_bus_idx : int
        Bus index where DER is connected.
    perturbation_mvar : float
        DER reactive power perturbation [Mvar] (default: ±1.0 Mvar).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity dQ_HV/dQ_DER [dimensionless].

    Raises
    ------
    RuntimeError
        If power flow fails for perturbed states.
    ValueError
        If DER sgen cannot be found at the specified bus.
    """
    # Find DER sgen at the specified bus (exclude boundary sgens)
    sgen_mask = (
        (net.sgen["bus"] == der_bus_idx) &
        ~net.sgen["name"].astype(str).str.startswith("BOUND_")
    )

    if not sgen_mask.any():
        raise ValueError(
            f"DER sgen at bus {der_bus_idx} not found in net.sgen."
        )

    sgen_idx = net.sgen.index[sgen_mask][0]
    q_der_original = float(net.sgen.at[sgen_idx, "q_mvar"])

    # Create deep copies for perturbation
    net_plus = copy.deepcopy(net)
    net_minus = copy.deepcopy(net)

    # Find sgen in perturbed networks
    sgen_mask_plus = (
        (net_plus.sgen["bus"] == der_bus_idx) &
        ~net_plus.sgen["name"].astype(str).str.startswith("BOUND_")
    )
    sgen_idx_plus = net_plus.sgen.index[sgen_mask_plus][0]

    sgen_mask_minus = (
        (net_minus.sgen["bus"] == der_bus_idx) &
        ~net_minus.sgen["name"].astype(str).str.startswith("BOUND_")
    )
    sgen_idx_minus = net_minus.sgen.index[sgen_mask_minus][0]

    # Perturb DER Q: Q + ΔQ
    net_plus.sgen.at[sgen_idx_plus, "q_mvar"] = q_der_original + perturbation_mvar
    try:
        pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for Q_DER + ΔQ: {e}")

    if not net_plus.converged:
        raise RuntimeError("Power flow did not converge for Q_DER + ΔQ.")

    q_hv_plus = float(net_plus.res_trafo3w.at[trafo3w_idx, "q_hv_mvar"])

    # Perturb DER Q: Q - ΔQ
    net_minus.sgen.at[sgen_idx_minus, "q_mvar"] = q_der_original - perturbation_mvar
    try:
        pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for Q_DER - ΔQ: {e}")

    if not net_minus.converged:
        raise RuntimeError("Power flow did not converge for Q_DER - ΔQ.")

    q_hv_minus = float(net_minus.res_trafo3w.at[trafo3w_idx, "q_hv_mvar"])

    # Compute finite difference
    dQ_dQder_numerical = (q_hv_plus - q_hv_minus) / (2.0 * perturbation_mvar)

    return dQ_dQder_numerical


def compute_numerical_dV_ds_trafo3w(
    net: pp.pandapowerNet,
    trafo3w_idx: int,
    observation_bus_indices: List[int],
    perturbation: int = 1,
) -> Tuple[NDArray[np.float64], List[int]]:
    """
    Compute numerical voltage sensitivity dV/ds for 3W transformer using finite differences.

    Uses central difference: dV/ds ≈ (V(s+Δs) - V(s-Δs)) / (2Δs)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    trafo3w_idx : int
        Three-winding transformer index.
    observation_bus_indices : List[int]
        Bus indices where voltages are observed.
    perturbation : int
        Tap position perturbation (default: ±1 tap step).

    Returns
    -------
    dV_ds : NDArray[np.float64]
        Numerical sensitivity vector [p.u./tap] of shape (n_obs,).
    bus_mapping : List[int]
        Ordered list of observation bus indices.

    Raises
    ------
    RuntimeError
        If power flow fails for perturbed states.
    """
    if trafo3w_idx not in net.trafo3w.index:
        raise ValueError(f"Transformer {trafo3w_idx} not found in net.trafo3w.")

    # Store original tap position
    s_original = int(net.trafo3w.at[trafo3w_idx, "tap_pos"])

    # Get tap bounds
    s_min = int(net.trafo3w.at[trafo3w_idx, "tap_min"])
    s_max = int(net.trafo3w.at[trafo3w_idx, "tap_max"])

    # Check if perturbation is feasible
    if s_original + perturbation > s_max or s_original - perturbation < s_min:
        raise ValueError(
            f"Perturbation ±{perturbation} exceeds tap limits "
            f"[{s_min}, {s_max}] for tap position {s_original}."
        )

    # Create deep copies for perturbation
    net_plus = copy.deepcopy(net)
    net_minus = copy.deepcopy(net)

    # Perturb tap position: s + Δs
    net_plus.trafo3w.at[trafo3w_idx, "tap_pos"] = s_original + perturbation
    try:
        pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s + Δs: {e}")

    if not net_plus.converged:
        raise RuntimeError("Power flow did not converge for s + Δs.")

    # Perturb tap position: s - Δs
    net_minus.trafo3w.at[trafo3w_idx, "tap_pos"] = s_original - perturbation
    try:
        pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s - Δs: {e}")

    if not net_minus.converged:
        raise RuntimeError("Power flow did not converge for s - Δs.")

    # Compute finite differences for all observation buses
    bus_mapping = []
    dV_ds_list = []

    for bus_idx in observation_bus_indices:
        if bus_idx not in net.res_bus.index:
            continue

        v_plus = float(net_plus.res_bus.at[bus_idx, "vm_pu"])
        v_minus = float(net_minus.res_bus.at[bus_idx, "vm_pu"])

        dV_ds_numerical = (v_plus - v_minus) / (2.0 * perturbation)

        dV_ds_list.append(dV_ds_numerical)
        bus_mapping.append(bus_idx)

    return np.array(dV_ds_list, dtype=np.float64), bus_mapping


def compute_numerical_dV_dQder(
    net: pp.pandapowerNet,
    der_bus_idx: int,
    observation_bus_indices: List[int],
    perturbation_mvar: float = 1.0,
) -> Tuple[NDArray[np.float64], List[int]]:
    """
    Compute numerical voltage sensitivity dV/dQ_DER using finite differences.

    Uses central difference: dV/dQ_DER ≈ (V(Q_DER+ΔQ) - V(Q_DER-ΔQ)) / (2ΔQ)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    der_bus_idx : int
        Bus index where DER is connected.
    observation_bus_indices : List[int]
        Bus indices where voltages are observed.
    perturbation_mvar : float
        DER reactive power perturbation [Mvar] (default: ±1.0 Mvar).

    Returns
    -------
    dV_dQder : NDArray[np.float64]
        Numerical sensitivity vector [p.u./Mvar] of shape (n_obs,).
    bus_mapping : List[int]
        Ordered list of observation bus indices.

    Raises
    ------
    RuntimeError
        If power flow fails for perturbed states.
    """
    # Find DER sgen at the specified bus (exclude boundary sgens)
    sgen_mask = (
        (net.sgen["bus"] == der_bus_idx) &
        ~net.sgen["name"].astype(str).str.startswith("BOUND_")
    )

    if not sgen_mask.any():
        raise ValueError(
            f"DER sgen at bus {der_bus_idx} not found in net.sgen."
        )

    sgen_idx = net.sgen.index[sgen_mask][0]
    q_der_original = float(net.sgen.at[sgen_idx, "q_mvar"])

    # Create deep copies for perturbation
    net_plus = copy.deepcopy(net)
    net_minus = copy.deepcopy(net)

    # Find sgen in perturbed networks
    sgen_mask_plus = (
        (net_plus.sgen["bus"] == der_bus_idx) &
        ~net_plus.sgen["name"].astype(str).str.startswith("BOUND_")
    )
    sgen_idx_plus = net_plus.sgen.index[sgen_mask_plus][0]

    sgen_mask_minus = (
        (net_minus.sgen["bus"] == der_bus_idx) &
        ~net_minus.sgen["name"].astype(str).str.startswith("BOUND_")
    )
    sgen_idx_minus = net_minus.sgen.index[sgen_mask_minus][0]

    # Perturb DER Q: Q + ΔQ
    net_plus.sgen.at[sgen_idx_plus, "q_mvar"] = q_der_original + perturbation_mvar
    try:
        pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for Q_DER + ΔQ: {e}")

    if not net_plus.converged:
        raise RuntimeError("Power flow did not converge for Q_DER + ΔQ.")

    # Perturb DER Q: Q - ΔQ
    net_minus.sgen.at[sgen_idx_minus, "q_mvar"] = q_der_original - perturbation_mvar
    try:
        pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for Q_DER - ΔQ: {e}")

    if not net_minus.converged:
        raise RuntimeError("Power flow did not converge for Q_DER - ΔQ.")

    # Compute finite differences for all observation buses
    bus_mapping = []
    dV_dQder_list = []

    for bus_idx in observation_bus_indices:
        if bus_idx not in net.res_bus.index:
            continue

        v_plus = float(net_plus.res_bus.at[bus_idx, "vm_pu"])
        v_minus = float(net_minus.res_bus.at[bus_idx, "vm_pu"])

        dV_dQder_numerical = (v_plus - v_minus) / (2.0 * perturbation_mvar)

        dV_dQder_list.append(dV_dQder_numerical)
        bus_mapping.append(bus_idx)

    return np.array(dV_dQder_list, dtype=np.float64), bus_mapping


# ==============================================================================
# TEST EXECUTION AND REPORTING
# ==============================================================================

def run_sensitivity_tests(
    verbose: bool = True,
    tap_perturbation: int = 1,
    q_perturbation_mvar: float = 1.0,
) -> Dict:
    """
    Run comprehensive sensitivity validation tests for 3-winding transformers.

    Parameters
    ----------
    verbose : bool
        If True, print detailed test results (default: True).
    tap_perturbation : int
        Tap position perturbation for numerical derivatives (default: ±1 tap).
    q_perturbation_mvar : float
        Reactive power perturbation for numerical derivatives (default: ±1.0 Mvar).

    Returns
    -------
    results : dict
        Dictionary containing test results for all sensitivities.
        Keys: 'dQ_ds', 'dQ_dQder', 'dV_ds', 'dV_dQder'
        Each entry contains analytical, numerical values and errors.

    Raises
    ------
    RuntimeError
        If network building or power flow fails.
    """
    if verbose:
        print("=" * 80)
        print("THREE-WINDING TRANSFORMER SENSITIVITY VALIDATION TEST")
        print("=" * 80)
        print()
        print("[1/6] Building combined 380/110/20 kV network...")

    # Build network
    net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

    if verbose:
        print("[2/6] Running converged power flow...")

    # Run power flow
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    if not net.converged:
        raise RuntimeError("Initial power flow did not converge.")

    if verbose:
        print("[3/6] Identifying 3-winding transformers and DER buses...")

    # Identify 3-winding transformers (coupler transformers)
    trafo3w_indices = list(meta.coupler_trafo3w_indices)

    if len(trafo3w_indices) == 0:
        raise RuntimeError("No 3-winding transformers found in network.")

    # Identify DER buses (distribution-connected sgens, exclude boundary sgens)
    der_bus_indices = []
    for sidx in net.sgen.index:
        name = str(net.sgen.at[sidx, "name"])
        if name.startswith("BOUND_"):
            continue
        subnet = str(net.sgen.at[sidx, "subnet"])
        if subnet == "DN":
            der_bus_indices.append(int(net.sgen.at[sidx, "bus"]))

    if len(der_bus_indices) == 0:
        raise RuntimeError("No DER buses found in network.")

    # Observation buses: all 110 kV buses
    observation_bus_indices = sorted([
        int(b) for b in net.bus.index
        if 100.0 <= float(net.bus.at[b, "vn_kv"]) < 200.0
        and str(net.bus.at[b, "subnet"]) == "DN"
    ])

    if len(observation_bus_indices) == 0:
        raise RuntimeError("No observation buses found in network.")

    if verbose:
        print(f"    Found {len(trafo3w_indices)} three-winding transformers")
        print(f"    Found {len(der_bus_indices)} DER buses")
        print(f"    Monitoring {len(observation_bus_indices)} voltage buses (110 kV)")
        print()
        print("[4/6] Computing analytical sensitivities from Jacobian...")

    # Initialise Jacobian sensitivity calculator
    jac_sens = JacobianSensitivities(net)

    # Storage for test results
    results = {
        "dQ_ds": [],
        "dQ_dQder": [],
        "dV_ds": [],
        "dV_dQder": [],
    }

    # ==============================================================================
    # TEST 1: dQ_HV/ds (3W transformer reactive power w.r.t. its own tap position)
    # ==============================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST 1: dQ_HV/ds (3W transformer HV reactive power sensitivity to tap)")
        print("=" * 80)

    for i, trafo3w_idx in enumerate(trafo3w_indices):
        if verbose:
            print(f"Testing 3W transformer {trafo3w_idx} (self-sensitivity)...")

        try:
            # Analytical sensitivity
            dQ_ds_analytical = jac_sens.compute_dQtrafo3w_hv_ds(
                meas_trafo3w_idx=trafo3w_idx,
                chg_trafo3w_idx=trafo3w_idx,
            )

            # Numerical sensitivity
            dQ_ds_numerical = compute_numerical_dQtrafo3w_ds(
                net=net,
                trafo3w_idx=trafo3w_idx,
                perturbation=tap_perturbation,
            )

            # Compute error metrics
            abs_error = abs(dQ_ds_analytical - dQ_ds_numerical)
            rel_error = abs_error / max(abs(dQ_ds_numerical), 1e-9) * 100.0

            results["dQ_ds"].append({
                "trafo3w_idx": trafo3w_idx,
                "analytical": dQ_ds_analytical,
                "numerical": dQ_ds_numerical,
                "abs_error": abs_error,
                "rel_error_pct": rel_error,
            })

            if verbose:
                print(f"  Analytical: {dQ_ds_analytical:12.6f} Mvar/tap")
                print(f"  Numerical:  {dQ_ds_numerical:12.6f} Mvar/tap")
                print(f"  Abs. Error: {abs_error:12.6e} Mvar/tap")
                print(f"  Rel. Error: {rel_error:12.4f} %")

                if rel_error < 5.0:
                    print("  Status:     ✓ PASS (< 5% error)")
                elif rel_error < 10.0:
                    print("  Status:     ~ WARNING (5-10% error)")
                else:
                    print("  Status:     ✗ FAIL (> 10% error)")

        except Exception as e:
            if verbose:
                print(f"  Status:     ✗ FAILED with exception: {e}")
            results["dQ_ds"].append({
                "trafo3w_idx": trafo3w_idx,
                "error": str(e),
            })

    # ==============================================================================
    # TEST 2: dQ_HV/dQ_DER (3W transformer reactive power w.r.t. DER Q)
    # ==============================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST 2: dQ_HV/dQ_DER (3W transformer HV reactive power sensitivity to DER Q)")
        print("=" * 80)

    # Test first 3W transformer with first DER for brevity
    trafo3w_test_idx = trafo3w_indices[0]
    der_test_idx = der_bus_indices[0]

    if verbose:
        print(f"Testing 3W transformer {trafo3w_test_idx} w.r.t. DER at bus {der_test_idx}...")

    try:
        # Analytical sensitivity
        dQ_dQder_analytical = jac_sens.compute_dQtrafo3w_hv_dQ_der(
            trafo3w_idx=trafo3w_test_idx,
            der_bus_idx=der_test_idx,
        )

        # Numerical sensitivity
        dQ_dQder_numerical = compute_numerical_dQtrafo3w_dQder(
            net=net,
            trafo3w_idx=trafo3w_test_idx,
            der_bus_idx=der_test_idx,
            perturbation_mvar=q_perturbation_mvar,
        )

        # Compute error metrics
        abs_error = abs(dQ_dQder_analytical - dQ_dQder_numerical)
        rel_error = abs_error / max(abs(dQ_dQder_numerical), 1e-9) * 100.0

        results["dQ_dQder"].append({
            "trafo3w_idx": trafo3w_test_idx,
            "der_bus_idx": der_test_idx,
            "analytical": dQ_dQder_analytical,
            "numerical": dQ_dQder_numerical,
            "abs_error": abs_error,
            "rel_error_pct": rel_error,
        })

        if verbose:
            print(f"  Analytical: {dQ_dQder_analytical:12.6f} [dimensionless]")
            print(f"  Numerical:  {dQ_dQder_numerical:12.6f} [dimensionless]")
            print(f"  Abs. Error: {abs_error:12.6e}")
            print(f"  Rel. Error: {rel_error:12.4f} %")

            if rel_error < 5.0:
                print("  Status:     ✓ PASS (< 5% error)")
            elif rel_error < 10.0:
                print("  Status:     ~ WARNING (5-10% error)")
            else:
                print("  Status:     ✗ FAIL (> 10% error)")

    except Exception as e:
        if verbose:
            print(f"  Status:     ✗ FAILED with exception: {e}")
        results["dQ_dQder"].append({
            "trafo3w_idx": trafo3w_test_idx,
            "der_bus_idx": der_test_idx,
            "error": str(e),
        })

    # ==============================================================================
    # TEST 3: dV/ds_trafo3w (Bus voltages w.r.t. 3W transformer tap position)
    # ==============================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST 3: dV/ds_trafo3w (Bus voltage sensitivity to 3W transformer tap)")
        print("=" * 80)

    # Test first 3W transformer with a subset of observation buses
    trafo3w_test_idx = trafo3w_indices[0]
    obs_buses_subset = observation_bus_indices[:5]  # Test first 5 buses

    if verbose:
        print(f"Testing 3W transformer {trafo3w_test_idx} with {len(obs_buses_subset)} observation buses...")

    try:
        # Analytical sensitivity
        dV_ds_analytical, obs_map_analytical = jac_sens.compute_dV_ds_trafo3w(
            trafo3w_idx=trafo3w_test_idx,
            observation_bus_indices=obs_buses_subset,
        )

        # Numerical sensitivity
        dV_ds_numerical, obs_map_numerical = compute_numerical_dV_ds_trafo3w(
            net=net,
            trafo3w_idx=trafo3w_test_idx,
            observation_bus_indices=obs_buses_subset,
            perturbation=tap_perturbation,
        )

        # Compare sensitivities for each bus
        for k, bus_idx in enumerate(obs_map_analytical):
            if bus_idx not in obs_map_numerical:
                if verbose:
                    print(f"  Bus {bus_idx}: Not in numerical mapping, skipping.")
                continue

            k_num = obs_map_numerical.index(bus_idx)

            analytical_val = dV_ds_analytical[k]
            numerical_val = dV_ds_numerical[k_num]
            abs_error = abs(analytical_val - numerical_val)
            rel_error = abs_error / max(abs(numerical_val), 1e-9) * 100.0

            results["dV_ds"].append({
                "trafo3w_idx": trafo3w_test_idx,
                "bus_idx": bus_idx,
                "analytical": analytical_val,
                "numerical": numerical_val,
                "abs_error": abs_error,
                "rel_error_pct": rel_error,
            })

            if verbose:
                print(f"  Bus {bus_idx:3d}:")
                print(f"    Analytical: {analytical_val:12.6e} p.u./tap")
                print(f"    Numerical:  {numerical_val:12.6e} p.u./tap")
                print(f"    Abs. Error: {abs_error:12.6e} p.u./tap")
                print(f"    Rel. Error: {rel_error:12.4f} %")

                if rel_error < 5.0:
                    print("    Status:     ✓ PASS")
                elif rel_error < 10.0:
                    print("    Status:     ~ WARNING")
                else:
                    print("    Status:     ✗ FAIL")

    except Exception as e:
        if verbose:
            print(f"  Status:     ✗ FAILED with exception: {e}")
        results["dV_ds"].append({
            "trafo3w_idx": trafo3w_test_idx,
            "error": str(e),
        })

    # ==============================================================================
    # TEST 4: dV/dQ_DER (Bus voltages w.r.t. DER reactive power - for reference)
    # ==============================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST 4: dV/dQ_DER (Bus voltage sensitivity to DER reactive power - reference)")
        print("=" * 80)

    # Test first DER with a subset of observation buses
    der_test_idx = der_bus_indices[0]
    obs_buses_subset = observation_bus_indices[:5]

    if verbose:
        print(f"Testing DER at bus {der_test_idx} with {len(obs_buses_subset)} observation buses...")

    try:
        # Analytical sensitivity - FIXED: unpacking 3 values instead of 2
        dV_dQder_analytical, obs_map_analytical, der_map_analytical = jac_sens.compute_dV_dQ_der(
            der_bus_indices=[der_test_idx],
            observation_bus_indices=obs_buses_subset,
        )

        # Extract column for this DER
        dV_dQder_analytical_col = dV_dQder_analytical[:, 0]

        # Numerical sensitivity
        dV_dQder_numerical, obs_map_numerical = compute_numerical_dV_dQder(
            net=net,
            der_bus_idx=der_test_idx,
            observation_bus_indices=obs_buses_subset,
            perturbation_mvar=q_perturbation_mvar,
        )

        # Compare sensitivities for each bus
        for k, bus_idx in enumerate(obs_map_analytical):
            if bus_idx not in obs_map_numerical:
                if verbose:
                    print(f"  Bus {bus_idx}: Not in numerical mapping, skipping.")
                continue

            k_num = obs_map_numerical.index(bus_idx)

            analytical_val = dV_dQder_analytical_col[k]
            numerical_val = dV_dQder_numerical[k_num]
            abs_error = abs(analytical_val - numerical_val)
            rel_error = abs_error / max(abs(numerical_val), 1e-9) * 100.0

            results["dV_dQder"].append({
                "der_bus_idx": der_test_idx,
                "bus_idx": bus_idx,
                "analytical": analytical_val,
                "numerical": numerical_val,
                "abs_error": abs_error,
                "rel_error_pct": rel_error,
            })

            if verbose:
                print(f"  Bus {bus_idx:3d}:")
                print(f"    Analytical: {analytical_val:12.6e} p.u./Mvar")
                print(f"    Numerical:  {numerical_val:12.6e} p.u./Mvar")
                print(f"    Abs. Error: {abs_error:12.6e} p.u./Mvar")
                print(f"    Rel. Error: {rel_error:12.4f} %")

                if rel_error < 5.0:
                    print("    Status:     ✓ PASS")
                elif rel_error < 10.0:
                    print("    Status:     ~ WARNING")
                else:
                    print("    Status:     ✗ FAIL")

    except Exception as e:
        if verbose:
            print(f"  Status:     ✗ FAILED with exception: {e}")
        results["dV_dQder"].append({
            "der_bus_idx": der_test_idx,
            "error": str(e),
        })

    # ==============================================================================
    # SUMMARY
    # ==============================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        # Count pass/fail for each test
        for test_name, test_results in results.items():
            passed = sum(1 for r in test_results if "error" not in r and r.get("rel_error_pct", 100) < 5.0)
            warnings = sum(1 for r in test_results if "error" not in r and 5.0 <= r.get("rel_error_pct", 100) < 10.0)
            failed = sum(1 for r in test_results if "error" in r or r.get("rel_error_pct", 100) >= 10.0)
            total = len(test_results)

            print(f"{test_name}:")
            print(f"  Total tests:  {total}")
            print(f"  Passed:       {passed} (< 5% error)")
            print(f"  Warnings:     {warnings} (5-10% error)")
            print(f"  Failed:       {failed} (> 10% error or exception)")

        print()
        print("=" * 80)
        print("All sensitivity tests completed.")
        print("=" * 80)

    return results


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Execute all sensitivity validation tests.
    """
    results = run_sensitivity_tests(
        verbose=True,
        tap_perturbation=1,
        q_perturbation_mvar=1.0,
    )

    # Optional: Save results to file
    # import json
    # with open("sensitivity_test_results.json", "w") as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
