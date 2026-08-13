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
- CVXPY (with SCIP solver; GUROBI is used when licensed — the two are **not**
  comparable, so the solver and version are recorded in each run's metadata)
- pandapower
- PowerFactory with its Python API — required only for the RMS co-simulation;
  every quasi-static path runs without it

## Installation

```bash
pip install -r requirements.txt
```

PowerFactory permits **one** session at a time. A second `connect()` terminates
the first, so a co-simulation in flight will be killed by any other process that
attaches to PowerFactory. Quasi-static runs and all analyses are unaffected and
can be run concurrently.

## Running a simulation

Every experiment is a wrapper around one call,
`run_multi_tso_dso(config, plant_factory=...)`, where `plant_factory` decides
which plant the *same* controller stack faces:

| Entry point | Static (QSS) leg | PowerFactory RMS leg | Results directory |
|---|---|---|---|
| `experiments/run_multi_system_ofo.py` | yes | no | `results/multi_system_ofo/` |
| `experiments/run_rms_cosim.py` | no | yes | `results/rms_cosim/` |
| `experiments/run_comparison_rms_cosim_qss.py` | yes | yes | `results/rms_phase6_replay/` |
| `experiments/run_openloop_qss_to_rms.py` | records | replays | — |

The quasi-static plant solves an algebraic power flow after every actuator
write, so it answers *which operating point the cascade reaches*; the RMS
co-simulation drives PowerFactory and additionally shows *how it gets there*.
The controllers cannot tell the difference: they act only on cached
sensitivities and their own measurements.

**Step-by-step description of all three:
[`docs/architecture/simulation_workflows.md`](docs/architecture/simulation_workflows.md)**,
including the structure of the shared runner and the constraints the RMS plant
imposes (features that mutate the pandapower network directly are rejected,
because for that plant the network is only a measurement mirror).

## Project Structure

```
qOFO_GH/
├── configs/              # MultiTSOConfig / CascadeConfig — also the run provenance record
├── core/                 # Plant interface, measurements, messages, profiles
├── controller/           # TSO/DSO OFO controllers
├── optimisation/         # MIQP solver layer
├── network/              # IEEE 39-bus build, HV underlays, scenarios
├── sensitivity/          # Cached-model Jacobians — the only plant model a controller has
├── experiments/          # Entry points; runners/ holds the shared closed-loop driver
├── pf/                   # PowerFactory integration (untracked in git)
├── analysis/             # Post-processing of stored runs
├── tuning/               # Offline Bayesian optimisation of controller weights
├── sbx_h/                # Horizontal scheduled-boundary coordination
├── sbx_v/                # Vertical band/request/grant coordination
├── visualisation/        # Live and publication plotting
├── tools/                # Repository utilities (documentation audit)
├── docs/                 # Architecture, packages, status, tuning, daily log
└── tests/                # Maintained regression and unit tests
```

Per-package overviews:
[`docs/architecture/packages/`](docs/architecture/packages/README.md).
Documentation coverage:
[`docs/architecture/doc_coverage.md`](docs/architecture/doc_coverage.md)
(regenerate both with `python tools/doc_audit.py && python tools/gen_package_docs.py`).

## Comparing runs

Results from different configurations are **not** interchangeable, and the
codebase treats this as a correctness concern rather than a convention. Each run
serialises its full configuration to `config.json`, and each analysis carries an
explicit admission filter over that block — scenario, DER capability model,
profile use, droop slope, per-DSO scenario multipliers, and whether a
disturbance was injected. `analysis/deadband_selection.py` is the worked
example; its `ADMIT` dictionary documents why each key is present, several of
them added after a mismatched run was found in a study.

In particular `base_410` and `rural_700` (410 vs 700 MW installed DER per DSO)
share the transmission-side build but their results are not comparable, and the
scenario is always passed explicitly rather than relying on a default.

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
