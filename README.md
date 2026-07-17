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
qOFO_GH/
├── configs/              # Multi-system and cascade configuration
├── core/                 # Measurements, messages, and cached state
├── controller/           # TSO/DSO OFO controllers
├── network/              # Transmission and distribution test systems
├── sensitivity/          # Cached-model Jacobians and sensitivities
├── experiments/          # Three active entry points plus CIGRE_2026
├── sbx_h/                # Horizontal scheduled-boundary coordination
├── sbx_v/                # Vertical band/request/grant coordination
├── visualisation/        # Live and publication plotting
├── docs/                 # Architecture, status, tuning, archive, daily log
└── tests/                # Maintained regression and unit tests
```

The supported horizontal coordination comparison is **none versus SBX-H**.
Tie-line Q remains a measured/recorded controlled-system output. Removed
BME, delta-V-reference, and weighted tie-Q mechanisms remain recoverable
from Git history and their design documents are retained under
`docs/_archive/`.

Experiment outputs use
`results/<experiment>/<NNNN>_<YYYY-MM-DD_HHMMSS>/` with `config.pkl`,
`config.json`, and `meta.json` provenance files.

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
