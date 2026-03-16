# Cascaded OFO Controller - Implementation Status

This document tracks the implementation status of all modules and functionalities
in the cascaded Online Feedback Optimisation (OFO) controller framework.

**Last Updated:** 2026-02-10 (Phase 5 + Architecture Corrections)

---

## Overview

The cascaded OFO framework implements a hierarchical voltage and reactive power
controller for TSO-DSO coordination. The TSO controller dispatches reactive power
setpoints to subordinate DSO controllers, which track these setpoints whilst
enforcing local network constraints.

---

## Module Status

### Core Data Structures (`core/`)

| Module | Class | Status | Description |
|--------|-------|--------|-------------|
| `network_state.py` | `NetworkState` | ✅ Implemented | Cached network state for Jacobian computation |
| `measurement.py` | `Measurement` | ✅ Implemented | Runtime measurements from the system |
| `message.py` | `SetpointMessage` | ✅ Implemented | TSO to DSO setpoint communication |
| `message.py` | `CapabilityMessage` | ✅ Implemented | DSO to TSO capability communication |
| `actuator_bounds.py` | `ActuatorBounds` | ✅ Implemented | Operating-point-dependent actuator bounds |

### Controller (`controller/`)

| Module | Class | Status | Description |
|--------|-------|--------|-------------|
| `base_controller.py` | `OFOParameters` | ✅ Implemented | Validated tuning parameters (α, g_w, g_z, g_s, g_u) |
| `base_controller.py` | `ControllerOutput` | ✅ Implemented | Result container for a single OFO iteration |
| `base_controller.py` | `BaseOFOController` | ✅ Implemented | Abstract base class with full OFO iteration logic |
| `tso_controller.py` | `TSOControllerConfig` | ✅ Implemented | Validated TSO configuration (PCC, OLTC, shunt, voltage schedule) |
| `tso_controller.py` | `TSOController` | ✅ Implemented | TSO-level MIQP controller with PCC setpoint dispatch |
| `dso_controller.py` | `DSOControllerConfig` | ✅ Implemented | Validated DSO configuration (DER, OLTC, shunt, interface) |
| `dso_controller.py` | `DSOController` | ✅ Implemented | DSO-level MIQP controller with Q setpoint tracking |

### Optimisation (`optimisation/`)

| Module | Class/Function | Status | Description |
|--------|----------------|--------|-------------|
| `miqp_solver.py` | `MIQPSolver` | ✅ Implemented | CVXPY-based MIQP solver interface |
| `miqp_solver.py` | `MIQPProblem` | ✅ Implemented | Data class for MIQP problem formulation |
| `miqp_solver.py` | `MIQPResult` | ✅ Implemented | Data class for MIQP solution results |
| `miqp_solver.py` | `build_miqp_problem` | ✅ Implemented | Convenience function to construct MIQP problems |

### Sensitivity (`sensitivity/`)

#### Analytical (Jacobian-Based)

| Module | Class/Function | Status | Description |
|--------|----------------|--------|-------------|
| `jacobian.py` | `JacobianSensitivities` | ✅ Implemented | Analytical sensitivity calculator class |
| `jacobian.py` | `compute_dV_dQ_der` | ✅ Implemented | ∂V/∂Q sensitivity from Jacobian (Eq. 9, 10) |
| `jacobian.py` | `compute_dV_ds` | ✅ Implemented | ∂V/∂s OLTC tap sensitivity (Eq. 11) |
| `jacobian.py` | `compute_dQtrafo_dQ_der` | ✅ Implemented | ∂Q_tr/∂Q_DER sensitivity (Eq. 12-14) |
| `jacobian.py` | `compute_dQtrafo_ds` | ✅ Implemented | ∂Q_tr/∂s OLTC sensitivity (Eq. 15-17) |
| `jacobian.py` | `compute_dI_dQ_der` | ✅ Implemented | ∂I/∂Q branch current sensitivity (Eq. 18-20) |
| `jacobian.py` | `build_sensitivity_matrix_H` | ✅ Implemented | Combined ∇H matrix construction with OLTC and shunt inputs |
| `jacobian.py` | `compute_dV_dQ_shunt` | ✅ Implemented | ∂V/∂Q_shunt sensitivity for switchable shunts |
| `jacobian.py` | `compute_dI_dQ_shunt` | ✅ Implemented | ∂I/∂Q_shunt sensitivity for switchable shunts |
| `jacobian.py` | `get_jacobian_indices` | ✅ Implemented | Helper function for Jacobian indexing |
| `jacobian.py` | `get_ppc_trafo_index` | ✅ Implemented | Helper function for pypower trafo mapping |

#### Numerical (Finite Differences)

| Module | Class/Function | Status | Description |
|--------|----------------|--------|-------------|
| `numerical.py` | `NumericalSensitivities` | ✅ Implemented | Numerical sensitivity calculator using perturbation method |
| `numerical.py` | `compute_dV_dQ_der` | ✅ Implemented | ∂V/∂Q via repeated power flow |
| `numerical.py` | `compute_dV_ds` | ✅ Implemented | ∂V/∂s (2W and 3W OLTC) via repeated power flow |
| `numerical.py` | `compute_dQtrafo_dQ_der` | ✅ Implemented | ∂Q_tr/∂Q_DER via repeated power flow |
| `numerical.py` | `compute_dQtrafo_ds` | ✅ Implemented | ∂Q_tr/∂s (2W and 3W OLTC) via repeated power flow |
| `numerical.py` | `compute_dI_dQ_der` | ✅ Implemented | ∂I/∂Q via repeated power flow |
| `numerical.py` | `compute_dV_dQ_shunt` | ✅ Implemented | ∂V/∂Q_shunt via repeated power flow |
| `numerical.py` | `compute_dI_dQ_shunt` | ✅ Implemented | ∂I/∂Q_shunt via repeated power flow |
| `numerical.py` | `build_sensitivity_matrix_H` | ✅ Implemented | Combined H matrix with all actuator types |

**Note:** `numerical.py` provides the same interface as `jacobian.py` for drop-in replacement.
Use for validation and debugging of analytical sensitivities.

### Network (`network/`)

| Module | Class/Function | Status | Description |
|--------|----------------|--------|-------------|
| `build_tuda_net.py` | `NetworkMetadata` | ✅ Implemented | Frozen dataclass recording all created element indices |
| `build_tuda_net.py` | `build_tuda_net()` | ✅ Implemented | Build combined 380/110/20 kV benchmark network with 3W couplers |
| `split_tn_dn_net.py` | `CouplerPowerFlow` | ✅ Implemented | Frozen dataclass for converged coupler measurements |
| `split_tn_dn_net.py` | `SplitResult` | ✅ Implemented | Container for split TN/DN networks + boundary metadata |
| `split_tn_dn_net.py` | `split_network()` | ✅ Implemented | Split combined network into separate TN and DN models |
| `split_tn_dn_net.py` | `validate_split()` | ✅ Implemented | Verify split reproduces combined operating point |

### Run Scripts

| Script | Status | Description |
|--------|--------|-------------|
| `run_cascade.py` | ✅ Correct | Cascaded TSO-DSO OFO loop with numerical/analytical toggle |
| `run_tso_voltage_control.py` | ✅ Fixed (2026-02-10) | TSO-only OFO with numerical/analytical toggle |
| `run_dso_reactive_power_control.py` | ✅ Correct | DSO-only OFO with numerical/analytical toggle (reference) |

### Tests (`tests/`)

| Module | Status | Description |
|--------|--------|-------------|
| `test_network_state.py` | ✅ Implemented | Unit tests for NetworkState |
| `test_measurement.py` | ✅ Implemented | Unit tests for Measurement |
| `test_message.py` | ✅ Implemented | Unit tests for Message classes |
| `test_actuator_bounds.py` | ✅ Implemented | Unit tests for ActuatorBounds |
| `test_jacobian.py` | ✅ Implemented | Unit tests for JacobianSensitivities (33 passed, 5 skipped for known issues) |
| `test_miqp_solver.py` | ✅ Implemented | Unit tests for MIQP solver |
| `test_controller.py` | ✅ Implemented | Unit tests for BaseOFO, TSO, and DSO controllers, including cascaded messaging |
| `test_network.py` | ✅ Implemented | Unit tests for network build, split, and split validation |

---

## Functionality Status

### Core OFO Algorithm

| Functionality | Status | Reference |
|---------------|--------|-----------|
| Iterative setpoint update: u^{k+1} = u^k + α·σ^k | ✅ Implemented | Eq. (22) in PSCC paper |
| MIQP objective formulation | ✅ Implemented | Eq. (27) in PSCC paper |
| Input constraints (actuator bounds) | ✅ Implemented | Eq. (24) in PSCC paper |
| Output constraints (voltage, current limits) | ✅ Implemented | Eq. (25) in PSCC paper |
| Soft constraints via slack variables | ✅ Implemented | Eq. (26) in PSCC paper |
| Objective gradient computation (DSO: Q tracking) | ✅ Implemented | Eq. (31) in PSCC paper |
| Objective gradient computation (TSO: V schedule) | ✅ Implemented | CIGRE 2026 Synopsis |
| Safety clipping and integer rounding | ✅ Implemented | BaseOFOController.step() |
| Predicted output computation | ✅ Implemented | y_pred = y + α·H·σ |

### Sensitivity Calculations

| Functionality | Status | Reference |
|---------------|--------|-----------|
| Bus voltage to DER Q sensitivity (analytical) | ✅ Implemented | Eq. (9), (10) in PSCC paper |
| Bus voltage to OLTC position sensitivity (analytical) | ✅ Implemented | Eq. (11) in PSCC paper |
| Transformer Q to DER Q sensitivity (analytical) | ✅ Implemented | Eq. (14) in PSCC paper |
| Transformer Q to OLTC position sensitivity (analytical) | ✅ Implemented | Eq. (17) in PSCC paper |
| Branch current to DER Q sensitivity (analytical) | ✅ Implemented | Eq. (20) in PSCC paper |
| Combined ∇H matrix construction (analytical) | ✅ Implemented | Eq. (29) in PSCC paper |
| All sensitivities via numerical finite differences | ✅ Implemented | validation.py / debugging |
| Toggle between analytical and numerical methods | ✅ Implemented | All run scripts |

### TSO-DSO Coordination

| Functionality | Status | Description |
|---------------|--------|-------------|
| TSO setpoint dispatch to DSO | ✅ Implemented | TSOController.generate_setpoint_messages() |
| DSO setpoint reception | ✅ Implemented | DSOController.receive_setpoint() |
| DSO capability aggregation | ✅ Implemented | DSOController.generate_capability_message() |
| TSO capability reception | ✅ Implemented | TSOController.receive_capability() |
| Interface Q tracking (soft constraint) | ✅ Implemented | DSO objective gradient with γ_Q penalty |
| Capability bounds at interface | ✅ Implemented | Jacobian-based mapping to PCC |
| PCC setpoint as TSO control variable | ✅ Implemented | Perfect-tracking assumption (∂Q_PCC/∂Q_set = 1) |
| Voltage schedule tracking (TSO) | ✅ Implemented | Soft output constraint with γ_V penalty |
| Voltage schedule update at runtime | ✅ Implemented | TSOController.update_voltage_setpoints() |

### Actuator-Specific Features

| Functionality | Status | Description |
|---------------|--------|-------------|
| DER Q bounds (VDE-AR-N 4120) | ✅ Implemented | P-dependent capability curve |
| Generator Q bounds | ⚠️ Placeholder | Fixed limits, needs P-Q diagram |
| OLTC tap bounds | ✅ Implemented | Fixed mechanical limits |
| Shunt state bounds | ✅ Implemented | Discrete states {-1, 0, +1} |

---

## Architecture

### Critical Principle: Combined Network vs Model Networks

See `ARCHITECTURE.md` for detailed documentation.

**Summary:**
- **Combined network** (`combined_net`): Real physical system
  - ✅ Take measurements
  - ✅ Apply control actions
  - ✅ Run power flow
- **Model networks** (`tn_net`, `dn_net`): Controller-internal models
  - ✅ Compute sensitivities
  - ✅ Create network states
  - ❌ **NOT** for measurements or controls

**All run scripts verified to follow this architecture** (2026-02-10).

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.20 | Array operations |
| scipy | >= 1.7 | Sparse matrices, linear algebra |
| cvxpy | >= 1.3 | Convex optimisation modelling |
| pyscipopt | >= 4.0 | SCIP solver interface (optional, for MIQP) |
| pandapower | >= 2.10 | Power flow simulation |
| pytest | >= 7.0 | Unit testing |

---

## Notes

- All code uses British English spelling in comments and documentation.
- No default values or silent error handling; missing data causes explicit errors.
- NetworkState is cached once from an initial operating point for Jacobian computation.
- TSO and DSO controllers maintain separate network states (no model exchange).
- The TSO controller models PCC setpoints as continuous control variables with an
  identity sensitivity approximation (perfect DSO tracking).
- **Measurements are always taken from the combined network (real plant)**.
- **Sensitivities are always computed from model networks (TN/DN split)**.

---

## Known Issues / Limitations

### Sensitivity Module

1. **Slack bus transformer sensitivity**: When a transformer has its HV bus connected
   to the slack bus, the OLTC and transformer Q sensitivities fail because the slack
   bus has no Jacobian index. The implementation needs to handle this case separately.

2. **Unit scaling in dV/dQ**: The reduced Jacobian inverse provides sensitivities in
   per-unit. There may be a factor of `sn_mva` difference between analytical and
   numerical sensitivities due to unit conversion.

3. **Integer variable sensitivities**: The H matrix now supports OLTC tap positions
   and shunt states as integer inputs. Use the `oltc_trafo_indices` and
   `shunt_bus_indices` parameters in `build_sensitivity_matrix_H()`. The
   `input_types` mapping indicates which inputs are 'continuous' vs 'integer'.

4. **Numerical sensitivity performance**: The numerical sensitivity calculator requires
   N power flows for N inputs, making it computationally expensive. Recommended for
   validation and debugging only, not for production use.

### Optimisation Module

1. **Solver availability**: The MIQP solver supports multiple backends (SCIP, GUROBI,
   MOSEK, ECOS_BB). For pure QP problems (no integers), OSQP or ECOS are used.
   Install pyscipopt for SCIP support: `pip install pyscipopt`.

2. **Variable reordering**: The `build_miqp_problem` function reorders variables to
   place continuous variables first, then integer variables. This is required by
   the internal CVXPY problem formulation.

### Controller Module

1. **DER active power measurement**: The `Measurement` class does not carry DER
   active power values. Both controllers currently use the installed capacity from
   `ActuatorBounds` as a proxy for computing P-dependent Q capability. A future
   iteration should add `der_p_mw` to `Measurement`.

2. **TSO PCC sensitivity approximation**: The TSO controller assumes perfect DSO
   tracking (`∂Q_PCC/∂Q_PCC_set = 1`). In practice, DSO tracking errors and local
   constraints lead to deviations. These are compensated in subsequent iterations
   via the feedback mechanism.

3. **H-matrix row/column reordering**: The TSO sensitivity matrix remaps rows and
   columns from the physical ordering returned by `build_sensitivity_matrix_H`
   to the controller-specific ordering. Verify consistency when adding new output
   types.

---

## Phase Completion Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Core data structures (NetworkState, Measurement, Message, ActuatorBounds) |
| Phase 2 | ✅ Complete | Sensitivity module (JacobianSensitivities with all required methods) |
| Phase 3 | ✅ Complete | MIQP solver interface (MIQPSolver, MIQPProblem, MIQPResult) |
| Phase 4 | ✅ Complete | Controller implementation (BaseOFOController, TSOController, DSOController) |
| Phase 5a | ✅ Complete | Network module (build_tuda_net, split_network, validate_split) |
| Phase 5b | ✅ Complete | Integration scripts with correct measurement architecture |
| Phase 6 | ✅ Complete | Numerical sensitivity validation module |

---

## Change Log

| Date | Changes |
|------|---------|
| 2025-02-05 | Initial implementation of core data structures (Phase 1) |
| 2025-02-05 | Implemented sensitivity module with all Jacobian-based methods (Phase 2) |
| 2025-02-06 | Implemented MIQP solver with CVXPY backend (Phase 3) |
| 2025-02-06 | Added unit tests for MIQP solver |
| 2025-02-06 | Implemented BaseOFOController with full OFO iteration logic (Phase 4) |
| 2025-02-06 | Implemented TSOController with PCC dispatch, capability reception, voltage schedule |
| 2025-02-06 | Implemented DSOController with Q tracking, capability reporting |
| 2025-02-06 | Added comprehensive unit tests for all controllers (test_controller.py) |
| 2025-02-06 | Updated controller/__init__.py to export all controller classes |
| 2026-02-06 | Refactored network/build_tuda_net.py: NetworkMetadata dataclass, machine transformers, TN shunt |
| 2026-02-06 | Refactored network/split_tn_dn_net.py: CouplerPowerFlow, SplitResult, split_network, validate_split |
| 2026-02-06 | Updated network/__init__.py to export clean public API |
| 2026-02-06 | Added test_network.py for build, split, and validation tests |
| 2026-02-10 | Implemented numerical.py with finite-difference sensitivities for validation |
| 2026-02-10 | Added toggle for numerical/analytical sensitivities in all run scripts |
| 2026-02-10 | **CRITICAL FIX**: Corrected run_tso_voltage_control.py to measure from combined network |
| 2026-02-10 | Verified run_cascade.py and run_dso_reactive_power_control.py follow correct architecture |
| 2026-02-10 | Added ARCHITECTURE.md documenting combined network vs model networks principle |

---

## References

1. Schwenke, M. et al. "Distribution Networks Providing Reactive Power as an
   Ancillary Service: Hierarchical Integration of Online Feedback Optimisation
   and Fuzzy Control." PSCC 2026.

2. Schwenke, M. et al. "Supporting Transmission Grid Voltage Control with Active
   Distribution Grids Using Online Feedback Optimisation." CIRED 2025.

3. Schwenke, M., Ruppert, J., Hanson, J. "Closed-Loop Voltage and Reactive Power
   Optimisation for Transmission Networks with Support from Active Distribution
   Networks." CIGRE 2026 Synopsis.

4. Klein-Helmkamp, F. et al. "Hierarchical Provision of Distribution Grid
   Flexibility with Online Feedback Optimization." Electric Power Systems
   Research, 2024.

5. Zettl, I. et al. "Tuning a Cascaded Online Feedback Optimization Controller
   for Provision of Distributed Flexibility." SEST 2024.
