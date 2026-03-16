#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Branch Current Sensitivity Validation Test
============================================

This script validates the analytical current sensitivities computed in
``jacobian.py`` by comparing them against numerical sensitivities computed
using central finite differences (perturbation method).

The following sensitivities are tested:

1. ``dI/dQ_DER``   – Branch current magnitude w.r.t. DER reactive power
2. ``dI/dQ_shunt`` – Branch current magnitude w.r.t. shunt reactive power
3. ``dI/ds_2w``    – Branch current magnitude w.r.t. 2W OLTC tap position
4. ``dI/dVgen``    – Branch current magnitude w.r.t. PV generator voltage setpoint

Test Method
-----------
For each sensitivity, the test:

1. Computes the analytical sensitivity from the cached Jacobian state.
2. Perturbs the input variable by ±ε and runs a power flow.
3. Computes the numerical sensitivity via central finite differences.
4. Compares analytical vs. numerical and reports relative errors.

The test structure mirrors ``evaluate_3w.py``.

Author: Manuel Schwenke
Date: 2026-02-26
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import pandapower as pp

# Project imports
from network.build_tuda_net import build_tuda_net
from sensitivity.jacobian import JacobianSensitivities
from sensitivity.index_helper import pp_bus_to_ppc_bus


# ==============================================================================
# NUMERICAL SENSITIVITY COMPUTATION (FINITE DIFFERENCES)
# ==============================================================================

def compute_numerical_dI_dQder(
    net: pp.pandapowerNet,
    line_idx: int,
    der_bus_idx: int,
    perturbation_mvar: float = 1.0,
) -> float:
    """
    Compute numerical sensitivity d|I|/dQ_DER using central finite differences.

    Uses: d|I|/dQ ≈ (|I|(Q+ΔQ) − |I|(Q−ΔQ)) / (2ΔQ)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    line_idx : int
        Pandapower line index for the current measurement.
    der_bus_idx : int
        Bus index where the DER is connected.
    perturbation_mvar : float
        DER reactive power perturbation [Mvar] (default: ±1.0).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity d|I|/dQ_DER [kA/Mvar].
    """
    # Find DER sgen at the specified bus (exclude boundary sgens)
    sgen_mask = (
        (net.sgen["bus"] == der_bus_idx)
        & ~net.sgen["name"].astype(str).str.startswith("BOUND_")
    )
    if not sgen_mask.any():
        raise ValueError(f"No DER sgen found at bus {der_bus_idx}.")

    sgen_idx = net.sgen.index[sgen_mask][0]
    q_original = float(net.sgen.at[sgen_idx, "q_mvar"])

    # +ΔQ
    net_plus = copy.deepcopy(net)
    idx_plus = net_plus.sgen.index[
        (net_plus.sgen["bus"] == der_bus_idx)
        & ~net_plus.sgen["name"].astype(str).str.startswith("BOUND_")
    ][0]
    net_plus.sgen.at[idx_plus, "q_mvar"] = q_original + perturbation_mvar
    pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    if not net_plus.converged:
        raise RuntimeError("PF did not converge for Q + dQ.")
    i_plus = float(net_plus.res_line.at[line_idx, "i_from_ka"])

    # −ΔQ
    net_minus = copy.deepcopy(net)
    idx_minus = net_minus.sgen.index[
        (net_minus.sgen["bus"] == der_bus_idx)
        & ~net_minus.sgen["name"].astype(str).str.startswith("BOUND_")
    ][0]
    net_minus.sgen.at[idx_minus, "q_mvar"] = q_original - perturbation_mvar
    pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    if not net_minus.converged:
        raise RuntimeError("PF did not converge for Q - dQ.")
    i_minus = float(net_minus.res_line.at[line_idx, "i_from_ka"])

    return (i_plus - i_minus) / (2.0 * perturbation_mvar)


def compute_numerical_dI_dQshunt(
    net: pp.pandapowerNet,
    line_idx: int,
    shunt_bus_idx: int,
    q_step_mvar: float = 1.0,
    perturbation_steps: int = 1,
) -> float:
    """
    Compute numerical sensitivity d|I|/d(shunt_state) using finite differences.

    Uses: d|I|/ds ≈ (|I|(step+Δ) − |I|(step−Δ)) / (2Δ)

    The perturbation is applied to the shunt step count. The analytical
    sensitivity ``compute_dI_dQ_shunt`` returns the derivative w.r.t. one
    shunt switching step, so we compare to the same quantity.

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    line_idx : int
        Line index for the current measurement.
    shunt_bus_idx : int
        Bus where the shunt is connected.
    q_step_mvar : float
        Rated reactive power step of the shunt [Mvar].
    perturbation_steps : int
        Number of step increments for finite difference (default: 1).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity d|I|/d(shunt_state) [kA/step].
    """
    shunt_mask = net.shunt["bus"] == shunt_bus_idx
    if not shunt_mask.any():
        raise ValueError(f"No shunt found at bus {shunt_bus_idx}.")

    shunt_idx = net.shunt.index[shunt_mask][0]
    step_original = int(net.shunt.at[shunt_idx, "step"])

    # +Δstep
    net_plus = copy.deepcopy(net)
    net_plus.shunt.at[shunt_idx, "step"] = step_original + perturbation_steps
    pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    if not net_plus.converged:
        raise RuntimeError("PF did not converge for step + d.")
    i_plus = float(net_plus.res_line.at[line_idx, "i_from_ka"])

    # −Δstep
    net_minus = copy.deepcopy(net)
    net_minus.shunt.at[shunt_idx, "step"] = step_original - perturbation_steps
    pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    if not net_minus.converged:
        raise RuntimeError("PF did not converge for step - d.")
    i_minus = float(net_minus.res_line.at[line_idx, "i_from_ka"])

    return (i_plus - i_minus) / (2.0 * perturbation_steps)


def compute_numerical_dI_ds_2w(
    net: pp.pandapowerNet,
    line_idx: int,
    trafo_idx: int,
    perturbation_steps: int = 1,
) -> float:
    """
    Compute numerical sensitivity d|I|/d(tap_pos) for a 2W OLTC using finite differences.

    Uses: d|I|/ds ≈ (|I|(s+Δ) − |I|(s−Δ)) / (2Δ)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    line_idx : int
        Pandapower line index for the current measurement.
    trafo_idx : int
        Pandapower transformer index (2W with OLTC).
    perturbation_steps : int
        Number of tap step increments for finite difference (default: 1).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity d|I|/d(tap_step) [kA/step].
    """
    tap_original = int(net.trafo.at[trafo_idx, "tap_pos"])

    # +Δstep
    net_plus = copy.deepcopy(net)
    net_plus.trafo.at[trafo_idx, "tap_pos"] = tap_original + perturbation_steps
    pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    if not net_plus.converged:
        raise RuntimeError("PF did not converge for tap_pos + d.")
    i_plus = float(net_plus.res_line.at[line_idx, "i_from_ka"])

    # −Δstep
    net_minus = copy.deepcopy(net)
    net_minus.trafo.at[trafo_idx, "tap_pos"] = tap_original - perturbation_steps
    pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    if not net_minus.converged:
        raise RuntimeError("PF did not converge for tap_pos - d.")
    i_minus = float(net_minus.res_line.at[line_idx, "i_from_ka"])

    return (i_plus - i_minus) / (2.0 * perturbation_steps)


def compute_numerical_dI_dVgen(
    net: pp.pandapowerNet,
    line_idx: int,
    gen_idx: int,
    perturbation_pu: float = 0.01,
) -> float:
    """
    Compute numerical sensitivity d|I|/dV_gen using central finite differences.

    Uses: d|I|/dV ≈ (|I|(V+ΔV) − |I|(V−ΔV)) / (2ΔV)

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    line_idx : int
        Pandapower line index for the current measurement.
    gen_idx : int
        Pandapower generator index (index into net.gen).
    perturbation_pu : float
        Voltage magnitude perturbation [p.u.] (default: 0.01).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity d|I|/dV_gen [kA/p.u.].
    """
    vm_original = float(net.gen.at[gen_idx, "vm_pu"])

    # +ΔV
    net_plus = copy.deepcopy(net)
    net_plus.gen.at[gen_idx, "vm_pu"] = vm_original + perturbation_pu
    pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    if not net_plus.converged:
        raise RuntimeError("PF did not converge for V + dV.")
    i_plus = float(net_plus.res_line.at[line_idx, "i_from_ka"])

    # −ΔV
    net_minus = copy.deepcopy(net)
    net_minus.gen.at[gen_idx, "vm_pu"] = vm_original - perturbation_pu
    pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    if not net_minus.converged:
        raise RuntimeError("PF did not converge for V - dV.")
    i_minus = float(net_minus.res_line.at[line_idx, "i_from_ka"])

    return (i_plus - i_minus) / (2.0 * perturbation_pu)


def compute_numerical_dI_ds_3w(
    net: pp.pandapowerNet,
    line_idx: int,
    trafo3w_idx: int,
    perturbation_steps: int = 1,
) -> float:
    """
    Compute numerical sensitivity d|I|/d(tap_pos) for a 3W OLTC using finite differences.

    Parameters
    ----------
    net : pp.pandapowerNet
        Converged pandapower network.
    line_idx : int
        Pandapower line index for the current measurement.
    trafo3w_idx : int
        Pandapower ``net.trafo3w`` index.
    perturbation_steps : int
        Number of tap step increments for finite difference (default: 1).

    Returns
    -------
    sensitivity : float
        Numerical sensitivity d|I|/d(tap_step) [kA/step].
    """
    tap_original = int(net.trafo3w.at[trafo3w_idx, "tap_pos"])

    # +d step
    net_plus = copy.deepcopy(net)
    net_plus.trafo3w.at[trafo3w_idx, "tap_pos"] = tap_original + perturbation_steps
    pp.runpp(net_plus, run_control=False, calculate_voltage_angles=True)
    if not net_plus.converged:
        raise RuntimeError("PF did not converge for 3W tap_pos + d.")
    i_plus = float(net_plus.res_line.at[line_idx, "i_from_ka"])

    # -d step
    net_minus = copy.deepcopy(net)
    net_minus.trafo3w.at[trafo3w_idx, "tap_pos"] = tap_original - perturbation_steps
    pp.runpp(net_minus, run_control=False, calculate_voltage_angles=True)
    if not net_minus.converged:
        raise RuntimeError("PF did not converge for 3W tap_pos - d.")
    i_minus = float(net_minus.res_line.at[line_idx, "i_from_ka"])

    return (i_plus - i_minus) / (2.0 * perturbation_steps)


# ==============================================================================
# TEST EXECUTION AND REPORTING
# ==============================================================================

def run_current_sensitivity_tests(
    verbose: bool = True,
    q_perturbation_mvar: float = 1.0,
    max_lines_per_test: int = 5,
    max_ders_per_test: int = 3,
) -> Dict:
    """
    Run comprehensive current sensitivity validation tests.

    Parameters
    ----------
    verbose : bool
        If True, print detailed results (default: True).
    q_perturbation_mvar : float
        Reactive power perturbation for DER finite differences [Mvar].
    max_lines_per_test : int
        Maximum number of lines to test per sensitivity type.
    max_ders_per_test : int
        Maximum number of DERs to test per line.

    Returns
    -------
    results : dict
        Dictionary containing test results.
        Keys: ``'dI_dQder_TN'``, ``'dI_dQder_DN'``, ``'dI_dQshunt_TN'``,
        ``'dI_dQshunt_DN'``, ``'dI_ds2w'``, ``'dI_ds3w'``, ``'dI_dVgen'``
    """
    if verbose:
        print("=" * 80)
        print("BRANCH CURRENT SENSITIVITY VALIDATION TEST")
        print("=" * 80)
        print()
        print("[1/8] Building combined 380/110/20 kV network ...")

    # Build network
    net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

    if verbose:
        print("[2/8] Running converged power flow ...")

    pp.runpp(net, run_control=True, calculate_voltage_angles=True)
    if not net.converged:
        raise RuntimeError("Initial power flow did not converge.")

    if verbose:
        print("[3/8] Identifying monitored lines, DERs, shunts, OLTCs, and gens ...")

    # TN lines (TSO-monitored)
    tn_lines = sorted(
        int(li) for li in net.line.index
        if str(net.line.at[li, "subnet"]) == "TN"
    )
    # DN lines (DSO-monitored)
    dn_lines = sorted(
        int(li) for li in net.line.index
        if str(net.line.at[li, "subnet"]) == "DN"
    )

    dn_buses = {int(b) for b in net.bus.index if str(net.bus.at[b, "subnet"]) == "DN"}

    # TSO DERs (TS-connected, not boundary)
    tso_der_buses = list(dict.fromkeys(
        int(net.sgen.at[s, "bus"]) for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) not in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ))

    # DSO DERs (DN-connected, not boundary)
    dso_der_buses = list(dict.fromkeys(
        int(net.sgen.at[s, "bus"]) for s in net.sgen.index
        if int(net.sgen.at[s, "bus"]) in dn_buses
        and not str(net.sgen.at[s, "name"]).startswith("BOUND_")
    ))

    # 380 kV shunts (TN)
    tn_shunt_buses = [
        int(net.shunt.at[s, "bus"]) for s in meta.tn_shunt_indices
        if s in net.shunt.index
    ]
    tn_shunt_q = [
        float(net.shunt.at[s, "q_mvar"]) for s in meta.tn_shunt_indices
        if s in net.shunt.index
    ]
    # Tertiary shunts (DN)
    dn_shunt_buses = [
        int(net.shunt.at[s, "bus"]) for s in meta.tertiary_shunt_indices
        if s in net.shunt.index
    ]
    dn_shunt_q = [
        float(net.shunt.at[s, "q_mvar"]) for s in meta.tertiary_shunt_indices
        if s in net.shunt.index
    ]

    # 2W OLTC transformers (machine transformers with tap changers)
    oltc_2w_indices = list(meta.machine_trafo_indices)

    # 3W OLTC transformers (coupler transformers)
    oltc_3w_indices = list(meta.coupler_trafo3w_indices)

    # PV generators
    gen_indices = list(net.gen.index)
    gen_buses_pp = [int(net.gen.at[g, "bus"]) for g in gen_indices]

    if verbose:
        print(f"    TN lines: {len(tn_lines)},  DN lines: {len(dn_lines)}")
        print(f"    TSO DERs: {len(tso_der_buses)},  DSO DERs: {len(dso_der_buses)}")
        print(f"    TN shunts: {len(tn_shunt_buses)},  DN shunts: {len(dn_shunt_buses)}")
        print(f"    2W OLTCs: {len(oltc_2w_indices)},  3W OLTCs: {len(oltc_3w_indices)},  PV gens: {len(gen_indices)}")
        print()
        print("[4/8] Computing analytical sensitivities from Jacobian ...")

    jac_sens = JacobianSensitivities(net)

    results: Dict = {
        "dI_dQder_TN": [],
        "dI_dQder_DN": [],
        "dI_dQshunt_TN": [],
        "dI_dQshunt_DN": [],
        "dI_ds2w": [],
        "dI_ds3w": [],
        "dI_dVgen": [],
    }

    # ==========================================================================
    # TEST 1: dI/dQ_DER on TN lines (TSO actuator sensitivities)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 1: d|I|/dQ_DER on TN lines (TSO DER -> TN branch currents)")
        print("=" * 80)

    test_tn_lines = tn_lines[:max_lines_per_test]
    test_tso_ders = tso_der_buses[:max_ders_per_test]

    for line_idx in test_tn_lines:
        for der_bus in test_tso_ders:
            try:
                # Analytical
                analytical = jac_sens.compute_dI_dQ_der(line_idx, der_bus)

                # Numerical
                numerical = compute_numerical_dI_dQder(
                    net, line_idx, der_bus, q_perturbation_mvar,
                )

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_dQder_TN"].append({
                    "line_idx": line_idx,
                    "der_bus": der_bus,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- DER bus {der_bus:3d}:  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- DER bus {der_bus:3d}:  EXCEPTION: {e}")
                results["dI_dQder_TN"].append({
                    "line_idx": line_idx, "der_bus": der_bus, "error": str(e),
                })

    # ==========================================================================
    # TEST 2: dI/dQ_DER on DN lines (DSO actuator sensitivities)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 2: d|I|/dQ_DER on DN lines (DSO DER -> DN branch currents)")
        print("=" * 80)

    test_dn_lines = dn_lines[:max_lines_per_test]
    test_dso_ders = dso_der_buses[:max_ders_per_test]

    for line_idx in test_dn_lines:
        for der_bus in test_dso_ders:
            try:
                analytical = jac_sens.compute_dI_dQ_der(line_idx, der_bus)
                numerical = compute_numerical_dI_dQder(
                    net, line_idx, der_bus, q_perturbation_mvar,
                )

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_dQder_DN"].append({
                    "line_idx": line_idx,
                    "der_bus": der_bus,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- DER bus {der_bus:3d}:  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- DER bus {der_bus:3d}:  EXCEPTION: {e}")
                results["dI_dQder_DN"].append({
                    "line_idx": line_idx, "der_bus": der_bus, "error": str(e),
                })

    # ==========================================================================
    # TEST 3: dI/dQ_shunt on TN lines (TSO shunt -> TN branch currents)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 3: d|I|/dQ_shunt on TN lines (TN shunt -> TN branch currents)")
        print("=" * 80)

    for line_idx in test_tn_lines:
        for k, shunt_bus in enumerate(tn_shunt_buses):
            q_step = tn_shunt_q[k]
            try:
                # Analytical: compute_dI_dQ_shunt returns (vector, line_mapping)
                dI_vec, line_map = jac_sens.compute_dI_dQ_shunt(
                    shunt_bus_idx=shunt_bus,
                    line_indices=[line_idx],
                    q_step_mvar=q_step,
                )
                if len(dI_vec) == 0:
                    raise ValueError("No valid sensitivity returned.")
                analytical = float(dI_vec[0])

                # Numerical
                numerical = compute_numerical_dI_dQshunt(
                    net, line_idx, shunt_bus, q_step,
                )

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_dQshunt_TN"].append({
                    "line_idx": line_idx,
                    "shunt_bus": shunt_bus,
                    "q_step_mvar": q_step,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- Shunt bus {shunt_bus:3d} "
                          f"(q_step={q_step:.1f} Mvar):  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- Shunt bus {shunt_bus:3d}:  EXCEPTION: {e}")
                results["dI_dQshunt_TN"].append({
                    "line_idx": line_idx, "shunt_bus": shunt_bus, "error": str(e),
                })

    # ==========================================================================
    # TEST 4: dI/dQ_shunt on DN lines (DN shunt -> DN branch currents)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 4: d|I|/dQ_shunt on DN lines (DN shunt -> DN branch currents)")
        print("=" * 80)

    for line_idx in test_dn_lines:
        for k, shunt_bus in enumerate(dn_shunt_buses):
            q_step = dn_shunt_q[k]
            try:
                dI_vec, line_map = jac_sens.compute_dI_dQ_shunt(
                    shunt_bus_idx=shunt_bus,
                    line_indices=[line_idx],
                    q_step_mvar=q_step,
                )
                if len(dI_vec) == 0:
                    raise ValueError("No valid sensitivity returned.")
                analytical = float(dI_vec[0])

                numerical = compute_numerical_dI_dQshunt(
                    net, line_idx, shunt_bus, q_step,
                )

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_dQshunt_DN"].append({
                    "line_idx": line_idx,
                    "shunt_bus": shunt_bus,
                    "q_step_mvar": q_step,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- Shunt bus {shunt_bus:3d} "
                          f"(q_step={q_step:.1f} Mvar):  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- Shunt bus {shunt_bus:3d}:  EXCEPTION: {e}")
                results["dI_dQshunt_DN"].append({
                    "line_idx": line_idx, "shunt_bus": shunt_bus, "error": str(e),
                })

    # ==========================================================================
    # TEST 5: dI/ds_2w on TN lines (2W OLTC tap -> TN branch currents)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 5: d|I|/ds_2w on TN lines (2W OLTC tap -> TN branch currents)")
        print("=" * 80)

    for line_idx in test_tn_lines:
        for trafo_idx in oltc_2w_indices:
            try:
                analytical = jac_sens.compute_dI_ds_2w(line_idx, trafo_idx)
                numerical = compute_numerical_dI_ds_2w(net, line_idx, trafo_idx)

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_ds2w"].append({
                    "line_idx": line_idx,
                    "trafo_idx": trafo_idx,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- Trafo {trafo_idx:3d}:  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- Trafo {trafo_idx:3d}:  EXCEPTION: {e}")
                results["dI_ds2w"].append({
                    "line_idx": line_idx, "trafo_idx": trafo_idx, "error": str(e),
                })

    # ==========================================================================
    # TEST 6: dI/dVgen on TN lines (PV generator voltage -> TN branch currents)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 6: d|I|/dV_gen on TN lines (PV gen voltage -> TN branch currents)")
        print("=" * 80)

    for line_idx in test_tn_lines:
        for g_idx, gen_bus_pp in zip(gen_indices, gen_buses_pp):
            try:
                gen_bus_ppc = pp_bus_to_ppc_bus(net, gen_bus_pp)
                analytical = jac_sens.compute_dI_dVgen(line_idx, gen_bus_ppc)
                numerical = compute_numerical_dI_dVgen(net, line_idx, g_idx)

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_dVgen"].append({
                    "line_idx": line_idx,
                    "gen_idx": g_idx,
                    "gen_bus": gen_bus_pp,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- Gen bus {gen_bus_pp:3d}:  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- Gen bus {gen_bus_pp:3d}:  EXCEPTION: {e}")
                results["dI_dVgen"].append({
                    "line_idx": line_idx, "gen_bus": gen_bus_pp, "error": str(e),
                })

    # ==========================================================================
    # TEST 7: dI/ds_3w on TN+DN lines (3W OLTC tap -> branch currents)
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST 7: d|I|/ds_3w on TN+DN lines (3W OLTC tap -> branch currents)")
        print("=" * 80)

    # Test on both TN and DN lines (3W couplers affect both)
    test_lines_3w = (test_tn_lines[:3] + test_dn_lines[:3])

    for line_idx in test_lines_3w:
        for trafo3w_idx in oltc_3w_indices:
            try:
                analytical = jac_sens.compute_dI_ds_3w(line_idx, trafo3w_idx)
                numerical = compute_numerical_dI_ds_3w(net, line_idx, trafo3w_idx)

                abs_err = abs(analytical - numerical)
                denom = max(abs(numerical), 1e-9)
                rel_err = abs_err / denom * 100.0

                results["dI_ds3w"].append({
                    "line_idx": line_idx,
                    "trafo3w_idx": trafo3w_idx,
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

                if verbose:
                    status = _status_str(rel_err)
                    print(f"  Line {line_idx:3d} <- Trafo3w {trafo3w_idx:3d}:  "
                          f"anal={analytical:+12.6e}  "
                          f"num={numerical:+12.6e}  "
                          f"rel_err={rel_err:8.3f}%  {status}")

            except Exception as e:
                if verbose:
                    print(f"  Line {line_idx:3d} <- Trafo3w {trafo3w_idx:3d}:  EXCEPTION: {e}")
                results["dI_ds3w"].append({
                    "line_idx": line_idx, "trafo3w_idx": trafo3w_idx, "error": str(e),
                })

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    if verbose:
        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        for test_name, test_results in results.items():
            if len(test_results) == 0:
                print(f"{test_name}: (no tests)")
                continue

            passed = sum(
                1 for r in test_results
                if "error" not in r and r.get("rel_error_pct", 100) < 5.0
            )
            warnings = sum(
                1 for r in test_results
                if "error" not in r and 5.0 <= r.get("rel_error_pct", 100) < 10.0
            )
            failed = sum(
                1 for r in test_results
                if "error" in r or r.get("rel_error_pct", 100) >= 10.0
            )
            total = len(test_results)

            # Median and max relative error (excluding exceptions)
            valid = [r["rel_error_pct"] for r in test_results if "error" not in r]
            median_err = float(np.median(valid)) if valid else float("nan")
            max_err = float(np.max(valid)) if valid else float("nan")

            print(f"{test_name}:")
            print(f"  Total:    {total}")
            print(f"  Passed:   {passed} (< 5% error)")
            print(f"  Warnings: {warnings} (5-10% error)")
            print(f"  Failed:   {failed} (> 10% error or exception)")
            print(f"  Median error: {median_err:.3f}%,  Max error: {max_err:.3f}%")

        print()
        print("=" * 80)
        print("All current sensitivity tests completed.")
        print("=" * 80)

    return results


def _status_str(rel_err: float) -> str:
    """Return a pass/warning/fail status string."""
    if rel_err < 5.0:
        return "PASS"
    elif rel_err < 10.0:
        return "~ WARNING"
    else:
        return "FAIL"


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Execute all current sensitivity validation tests."""
    results = run_current_sensitivity_tests(
        verbose=True,
        q_perturbation_mvar=1.0,
        max_lines_per_test=5,
        max_ders_per_test=3,
    )


if __name__ == "__main__":
    main()
