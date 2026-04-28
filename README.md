# Cascaded Online Feedback Optimisation (qOFO)

A scientific implementation of a cascaded Online Feedback Optimisation (OFO) controller
for voltage and reactive power control across TSO-DSO interfaces.

## Overview

This project implements a hierarchical OFO control framework where:

- **Upper layer (TSO)**: MIQP-based OFO controller managing EHV-level actuators and
  issuing reactive power setpoints to subordinate DSO controllers
- **Lower layer (DSO)**: MIQP-based OFO controllers tracking TSO setpoints whilst
  enforcing local constraints

The framework is designed for scientific research and PhD thesis work. It follows a
fail-fast principle: missing values or invalid states result in explicit errors rather
than silent defaults.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TSO Controller (EHV)                        │
│  - Controls: Q_gen, Q_STATCOM, shunts, OLTC, Q_setpoints to DSO │
│  - Measures: V_EHV, Q_tie, Q_interface, I_lines                 │
│  - Objective: Voltage control, loss minimisation                │
└─────────────────────────────────────────────────────────────────┘
                          ▲
                          │ SetpointMessage (Q_set)
                          │ CapabilityMessage (Q_min, Q_max)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DSO Controller (HV)                         │
│  - Controls: Q_DER, OLTC, shunts                                │
│  - Measures: V_HV, Q_interface, I_lines                         │
│  - Objective: Track Q_setpoint, enforce local constraints       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Network State Separation

TSO and DSO controllers maintain separate network state representations:

- **TSO view**: DSO areas represented as PQ nodes (fixed P, Q injection)
- **DSO view**: TSO interface represented as PV node (fixed P, V) with one slack bus

This reflects real-world operational boundaries where TSO and DSO do not exchange
detailed network models.

### OFO Iteration

Each controller solves an MIQP at iteration k:

```
u^{k+1} = u^k + α · σ(u^k, d^k, y^k)
```

where σ is the solution to the quadratic programme projecting the gradient onto
the feasible set.

## Dependencies

- Python >= 3.10
- NumPy
- SciPy
- CVXPY (with SCIP solver)
- pandapower

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
qOFO/
├── core/                 # Core data structures
│   ├── network_state.py  # Cached network state for Jacobian computation
│   ├── measurement.py    # Runtime measurements from the system
│   ├── message.py        # Inter-controller messages
│   └── actuator_bounds.py# Operating-point-dependent actuator bounds
│
├── controller/           # Controller implementations
│   ├── base_controller.py# Abstract base OFO controller
│   ├── tso_controller.py # TSO-level MIQP controller
│   └── dso_controller.py # DSO-level MIQP controller
│
├── optimisation/         # MIQP solver interface
│   └── miqp_solver.py    # Quadratic programme formulation
│
├── sensitivity/          # Sensitivity calculations
│   └── jacobian.py       # Jacobian-based sensitivity matrices
│
└── tests/                # Unit tests
```

## References

- Schwenke, M., Ruppert, J., Hanson, J. (2026). "Closed-Loop Voltage and Reactive Power
  Optimisation for Transmission Networks with Support from Active Distribution Networks."
  CIGRE Calgary.

- Schwenke, M., Hanson, J. (2026). "Distribution Networks Providing Reactive Power as
  an Ancillary Service: Hierarchical Integration of Online Feedback Optimisation and
  Fuzzy Control." PSCC Limassol.

- Schwenke, M., Korff, F., Hanson, J. (2025). "Supporting Transmission Grid Voltage
  Control with Active Distribution Grids Using Online Feedback Optimisation."
  CIRED Geneva.

## Author

Manuel Schwenke  
Technical University of Darmstadt  
Institute of Electrical Power Supply with Integration of Renewable Energy

## License

[To be determined]
