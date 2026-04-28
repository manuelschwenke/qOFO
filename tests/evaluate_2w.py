#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Winding Machine Transformer Sensitivity Validation Test
=============================================================

This script validates the analytical sensitivities for 2-winding machine
transformers (generator step-up transformers with OLTC) by comparing them
against numerical sensitivities computed using finite differences.

The following sensitivities are tested:

1. dV/ds (2W): Bus voltages w.r.t. machine transformer tap position
2. dQ_trafo/ds (2W): Transformer Q w.r.t. its own tap position

Test Method
-----------
For each sensitivity, the test:
1. Computes analytical sensitivity from Jacobian (cached state)
2. Perturbs the input variable by ±ε and runs power flow
3. Computes numerical sensitivity via finite differences
4. Compares analytical vs. numerical and reports errors

Author: Manuel Schwenke
Date: 2026-02-13
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


# ==============================================================================
# NUMERICAL SENSITIVITY COMPUTATION (FINITE DIFFERENCES)
# ==============================================================================

def compute_numerical_dQtrafo_2w_ds(
    net: pp.pandapowerNet,
    meas_trafo_idx: int,
    chg_trafo_idx: int,
    perturbation: int = 1,
) -> float:
    """
    Compute numerical sensitivity dQ_trafo/ds using finite differences.

    Uses central difference: dQ/ds ≈ (Q(s+Δs) - Q(s-Δs)) / (2Δs)

    The reactive power is measured at the HV side of the measurement
    transformer.

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    meas_trafo_idx : int
        Two-winding transformer index where Q is measured.
    chg_trafo_idx : int
        Two-winding transformer index where tap is changed.
    perturbation : int
        Tap position perturbation (default: ±1 tap step).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity dQ/ds [Mvar/tap].

    Raises
    ------
    RuntimeError
        If power flow fails for perturbed states.
    """
    if chg_trafo_idx not in net.trafo.index:
        raise ValueError(f"Transformer {chg_trafo_idx} not found in net.trafo.")
    if meas_trafo_idx not in net.trafo.index:
        raise ValueError(f"Transformer {meas_trafo_idx} not found in net.trafo.")

    # Store original tap position
    s_original = int(net.trafo.at[chg_trafo_idx, "tap_pos"])

    # Get tap bounds
    s_min = int(net.trafo.at[chg_trafo_idx, "tap_min"])
    s_max = int(net.trafo.at[chg_trafo_idx, "tap_max"])

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
    net_plus.trafo.at[chg_trafo_idx, "tap_pos"] = s_original + perturbation
    try:
        pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s + Δs: {e}")

    if not net_plus.converged:
        raise RuntimeError("Power flow did not converge for s + Δs.")

    q_hv_plus = float(net_plus.res_trafo.at[meas_trafo_idx, "q_hv_mvar"])

    # Perturb tap position: s - Δs
    net_minus.trafo.at[chg_trafo_idx, "tap_pos"] = s_original - perturbation
    try:
        pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s - Δs: {e}")

    if not net_minus.converged:
        raise RuntimeError("Power flow did not converge for s - Δs.")

    q_hv_minus = float(net_minus.res_trafo.at[meas_trafo_idx, "q_hv_mvar"])

    # Compute finite difference
    dQ_ds_numerical = (q_hv_plus - q_hv_minus) / (2.0 * perturbation)

    return dQ_ds_numerical


def compute_numerical_dV_ds_2w(
    net: pp.pandapowerNet,
    trafo_idx: int,
    observation_bus_indices: List[int],
    perturbation: int = 1,
) -> Tuple[NDArray[np.float64], List[int]]:
    """
    Compute numerical voltage sensitivity dV/ds for 2W transformer.

    Uses central difference: dV/ds ≈ (V(s+Δs) - V(s-Δs)) / (2Δs)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    trafo_idx : int
        Two-winding transformer index.
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
    if trafo_idx not in net.trafo.index:
        raise ValueError(f"Transformer {trafo_idx} not found in net.trafo.")

    # Store original tap position
    s_original = int(net.trafo.at[trafo_idx, "tap_pos"])

    # Get tap bounds
    s_min = int(net.trafo.at[trafo_idx, "tap_min"])
    s_max = int(net.trafo.at[trafo_idx, "tap_max"])

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
    net_plus.trafo.at[trafo_idx, "tap_pos"] = s_original + perturbation
    try:
        pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    except Exception as e:
        raise RuntimeError(f"Power flow failed for s + Δs: {e}")

    if not net_plus.converged:
        raise RuntimeError("Power flow did not converge for s + Δs.")

    # Perturb tap position: s - Δs
    net_minus.trafo.at[trafo_idx, "tap_pos"] = s_original - perturbation
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


# ==============================================================================
# TEST EXECUTION AND REPORTING
# ==============================================================================

def _print_result(
    label: str,
    analytical: float,
    numerical: float,
    unit: str,
    indent: int = 2,
) -> Dict:
    """Print and return comparison metrics for a single sensitivity value."""
    abs_error = abs(analytical - numerical)
    rel_error = abs_error / max(abs(numerical), 1e-9) * 100.0

    prefix = " " * indent
    print(f"{prefix}Analytical: {analytical:12.6e} {unit}")
    print(f"{prefix}Numerical:  {numerical:12.6e} {unit}")
    print(f"{prefix}Abs. Error: {abs_error:12.6e} {unit}")
    print(f"{prefix}Rel. Error: {rel_error:12.4f} %")

    if rel_error < 5.0:
        print(f"{prefix}Status:     PASS (< 5% error)")
    elif rel_error < 10.0:
        print(f"{prefix}Status:     WARNING (5-10% error)")
    else:
        print(f"{prefix}Status:     FAIL (> 10% error)")

    return {
        "analytical": analytical,
        "numerical": numerical,
        "abs_error": abs_error,
        "rel_error_pct": rel_error,
    }


def run_sensitivity_tests(
    verbose: bool = True,
    tap_perturbation: int = 1,
) -> Dict:
    """
    Run comprehensive sensitivity validation tests for 2-winding machine
    transformers.

    Parameters
    ----------
    verbose : bool
        If True, print detailed test results (default: True).
    tap_perturbation : int
        Tap position perturbation for numerical derivatives (default: ±1 tap).

    Returns
    -------
    results : dict
        Dictionary containing test results for all sensitivities.
    """
    if verbose:
        print("=" * 80)
        print("TWO-WINDING MACHINE TRANSFORMER SENSITIVITY VALIDATION TEST")
        print("=" * 80)
        print()
        print("[1/5] Building combined 380/110/20 kV network...")

    # Build network
    net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

    if verbose:
        print("[2/5] Running converged power flow...")

    # Run power flow
    pp.runpp(net, run_control=True, calculate_voltage_angles=True)

    if not net.converged:
        raise RuntimeError("Initial power flow did not converge.")

    if verbose:
        print("[3/5] Identifying machine transformers and network elements...")

    # Identify machine transformers
    machine_trafo_indices = list(meta.machine_trafo_indices)

    if len(machine_trafo_indices) == 0:
        raise RuntimeError("No machine transformers found in network.")

    # Print machine transformer info
    if verbose:
        for mt_idx in machine_trafo_indices:
            hv_bus = int(net.trafo.at[mt_idx, "hv_bus"])
            lv_bus = int(net.trafo.at[mt_idx, "lv_bus"])
            name = str(net.trafo.at[mt_idx, "name"])
            v_lv = float(net.res_bus.at[lv_bus, "vm_pu"])
            v_hv = float(net.res_bus.at[hv_bus, "vm_pu"])
            tap_pos = int(net.trafo.at[mt_idx, "tap_pos"])
            print(f"    Machine trafo {mt_idx} ({name}):")
            print(f"      HV bus {hv_bus} (V={v_hv:.4f} pu), "
                  f"LV bus {lv_bus} (V_gen={v_lv:.4f} pu), tap={tap_pos}")

    # Observation buses: all 110 kV and 380 kV buses
    observation_bus_indices = sorted([
        int(b) for b in net.bus.index
        if float(net.bus.at[b, "vn_kv"]) >= 100.0
    ])

    if verbose:
        print(f"    Found {len(machine_trafo_indices)} machine transformer(s)")
        print(f"    Monitoring {len(observation_bus_indices)} voltage buses "
              f"(110 kV + 380 kV)")

    if verbose:
        print()
        print("[4/5] Computing analytical sensitivities from Jacobian...")

    # Initialise Jacobian sensitivity calculator
    jac_sens = JacobianSensitivities(net)

    results = {
        "dV_ds_standard": [],
        "dQ_ds_standard": [],
    }

    # ==================================================================
    # TEST 1: dV/ds (standard method, cached V_j from power flow)
    # ==================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST 1: dV/ds_2w (standard — cached V_j from power flow)")
        print("=" * 80)

    obs_subset = observation_bus_indices[:8]

    for mt_idx in machine_trafo_indices:
        if verbose:
            print(f"\nMachine trafo {mt_idx} ({net.trafo.at[mt_idx, 'name']}):")

        try:
            dV_ds_ana, obs_map_ana = jac_sens.compute_dV_ds_2w(
                trafo_idx=mt_idx,
                observation_bus_indices=obs_subset,
            )

            dV_ds_num, obs_map_num = compute_numerical_dV_ds_2w(
                net=net,
                trafo_idx=mt_idx,
                observation_bus_indices=obs_subset,
                perturbation=tap_perturbation,
            )

            for k, bus_idx in enumerate(obs_map_ana):
                if bus_idx not in obs_map_num:
                    continue
                k_num = obs_map_num.index(bus_idx)

                if verbose:
                    print(f"  Bus {bus_idx:3d}:")
                r = _print_result(
                    f"Bus {bus_idx}", dV_ds_ana[k], dV_ds_num[k_num],
                    "p.u./tap", indent=4,
                )
                r["bus_idx"] = bus_idx
                r["trafo_idx"] = mt_idx
                results["dV_ds_standard"].append(r)

        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
            results["dV_ds_standard"].append({"trafo_idx": mt_idx, "error": str(e)})

    # ==================================================================
    # TEST 2: dQ_trafo/ds (self-sensitivity, standard)
    # ==================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST 2: dQ_trafo/ds (self-sensitivity: meas = chg)")
        print("=" * 80)

    for mt_idx in machine_trafo_indices:
        if verbose:
            print(f"\n  Standard: trafo {mt_idx} (self)")

        try:
            dQ_ds_ana = jac_sens.compute_dQtrafo_2w_ds(
                meas_trafo_idx=mt_idx, chg_trafo_idx=mt_idx,
            )
            dQ_ds_num = compute_numerical_dQtrafo_2w_ds(
                net=net,
                meas_trafo_idx=mt_idx, chg_trafo_idx=mt_idx,
                perturbation=tap_perturbation,
            )
            r = _print_result("dQ/ds std", dQ_ds_ana, dQ_ds_num, "Mvar/tap")
            r["trafo_idx"] = mt_idx
            results["dQ_ds_standard"].append(r)
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
            results["dQ_ds_standard"].append({"trafo_idx": mt_idx, "error": str(e)})

    # ==================================================================
    # SUMMARY
    # ==================================================================

    if verbose:
        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        test_descriptions = {
            "dV_ds_standard":
                "dV/ds (standard, cached V_j)",
            "dQ_ds_standard":
                "dQ/ds (standard, cached V_j)",
        }

        total_pass = 0
        total_warn = 0
        total_fail = 0

        for test_name, description in test_descriptions.items():
            test_results = results[test_name]
            passed = sum(
                1 for r in test_results
                if "error" not in r and r.get("rel_error_pct", 100) < 5.0
            )
            warnings = sum(
                1 for r in test_results
                if "error" not in r
                and 5.0 <= r.get("rel_error_pct", 100) < 10.0
            )
            failed = sum(
                1 for r in test_results
                if "error" in r or r.get("rel_error_pct", 100) >= 10.0
            )
            total = len(test_results)

            total_pass += passed
            total_warn += warnings
            total_fail += failed

            print(f"\n{description}:")
            print(f"  Total:    {total}")
            print(f"  Passed:   {passed} (< 5% error)")
            print(f"  Warnings: {warnings} (5-10% error)")
            print(f"  Failed:   {failed} (> 10% error or exception)")

        print()
        print("-" * 40)
        print(f"Overall: {total_pass} passed, {total_warn} warnings, "
              f"{total_fail} failed")
        print("=" * 80)

    return results


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """Execute all 2W machine transformer sensitivity validation tests."""
    results = run_sensitivity_tests(
        verbose=True,
        tap_perturbation=1,
    )


if __name__ == "__main__":
    main()
