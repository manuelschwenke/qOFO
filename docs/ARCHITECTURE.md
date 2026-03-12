# OFO Controller Architecture

## Critical Principle: Combined Network vs Model Networks

This document describes the **fundamental architecture** of the cascaded OFO controller implementation.

---

## Two-Network Architecture

### 1. Combined Network (Real Plant)

The **combined network** (`combined_net`) represents the **real physical power system**.

**Purpose:**
- ✅ Take measurements (voltages, currents, reactive power, tap positions, etc.)
- ✅ Apply control actions (Q setpoints, tap changes, shunt states)
- ✅ Run power flow simulations (represents physical system behaviour)

**Key Point:** 
> All measurements **MUST** come from the combined network because this represents what sensors in the real system would measure.

### 2. Model Networks (TN and DN Split)

The **model networks** (`tn_net`, `dn_net`) are split versions used **only for controller calculations**.

**Purpose:**
- ✅ Compute sensitivity matrices (Jacobians)
- ✅ Create network states for sensitivity calculations
- ❌ **NOT** used for measurements
- ❌ **NOT** used for control application
- ❌ **NOT** used for power flow simulation

**Key Point:**
> Model networks enable each controller (TSO/DSO) to compute sensitivities from their own network view, but all physical interactions happen through the combined network.

---

## Implementation Pattern

All run scripts (`run_cascade.py`, `run_tso_voltage_control.py`, `run_dso_reactive_power_control.py`) follow this pattern:

### Step 1: Build Networks

```python
# 1) Build combined network - this is the REAL PLANT
combined_net, meta = build_tuda_net(ext_grid_vm_pu=1.06, pv_nodes=True)

# 2) Run initial power flow on combined plant
pp.runpp(combined_net, run_control=True, calculate_voltage_angles=True)

# 3) Split to create model networks (for sensitivities only)
split_result = split_network(combined_net, meta, dn_slack_coupler_index=0)
tn_net_model = split_result.tn_net  # MODEL, not plant!
dn_net_model = split_result.dn_net  # MODEL, not plant!
```

### Step 2: Configure Controllers

```python
# Identify actuators from COMBINED network
# (these are the real physical devices)
tso_der_buses = [...] # from combined_net
dso_der_buses = [...] # from combined_net

# Compute sensitivities from MODEL networks
tso_sens = JacobianSensitivities(tn_net_model)
dso_sens = JacobianSensitivities(dn_net_model)

# Network states from MODEL networks
tso_ns = network_state_from_net(tn_net_model, ...)
dso_ns = network_state_from_net(dn_net_model, ...)

# Actuator bounds from COMBINED network (real ratings)
tso_bounds = ActuatorBounds(...)  # ratings from combined_net
dso_bounds = ActuatorBounds(...)  # ratings from combined_net
```

### Step 3: Control Loop

```python
for iteration in range(1, n_iterations + 1):
    # ✅ MEASURE from COMBINED network (real plant)
    measurement = measurement_from_combined(combined_net, config, iteration)
    
    # ✅ COMPUTE control with sensitivities from MODEL
    output = controller.step(measurement)
    
    # ✅ APPLY controls to COMBINED network (real plant)
    apply_controls(combined_net, output, config)
    
    # ✅ RUN power flow on COMBINED network (real plant)
    pp.runpp(combined_net, run_control=False, calculate_voltage_angles=True)
```

---

## Why This Architecture?

### 1. Realistic Measurements

Measurements from the combined network include **all interactions** between transmission and distribution systems:
- Voltage coupling through transformers
- Reactive power flows across TN-DN boundaries
- Generator responses
- Load behaviour

This matches what **real sensors** in the field would measure.

### 2. Separate Controller Views

Each controller computes sensitivities from its own network model:
- TSO uses `tn_net_model` → sees TN-only sensitivities
- DSO uses `dn_net_model` → sees DN-only sensitivities

This reflects that:
- TSO doesn't know DN internal topology
- DSO doesn't know TN internal topology
- Controllers coordinate through interface measurements (PCC Q)

### 3. Model-Plant Mismatch

In reality, controllers never have a perfect model of the system. This architecture naturally introduces **model-plant mismatch**:
- Model networks are decoupled (split)
- Combined network includes all couplings
- Feedback compensates for mismatch over iterations

This tests the **robustness** of the OFO algorithm.

---

## Consequences of Violating This Architecture

❌ **WRONG: Taking measurements from model networks**

```python
# DON'T DO THIS!
measurement = measurement_from_tn(tn_net_model, ...)  # ❌ WRONG!
```

**Problem:** Model networks are decoupled. Measurements from `tn_net_model` would:
- Miss TN-DN interactions
- Not reflect boundary power flows
- Not represent physical system state
- Lead to incorrect controller behaviour

❌ **WRONG: Applying controls to model networks**

```python
# DON'T DO THIS!
apply_controls(tn_net_model, output, ...)  # ❌ WRONG!
pp.runpp(tn_net_model)  # ❌ WRONG!
```

**Problem:** Changes to model networks don't affect the "real" system. The physical combined network would remain unchanged.

---

## Verification Checklist

For each run script, verify:

- [ ] Combined network is built first
- [ ] Initial power flow runs on combined network
- [ ] Model networks are created by splitting combined network
- [ ] Measurement functions take `combined_net` as input
- [ ] Control application functions take `combined_net` as input
- [ ] Power flow always runs on `combined_net`
- [ ] Sensitivities are computed from model networks (`tn_net`, `dn_net`)
- [ ] Network states are created from model networks
- [ ] Actuator bounds use ratings from combined network

---

## Summary Table

| Operation | Network | Reason |
|-----------|---------|--------|
| **Measurements** | `combined_net` | Represents real sensors |
| **Control application** | `combined_net` | Represents real actuators |
| **Power flow** | `combined_net` | Represents physical system |
| **Sensitivity calculation** | `tn_net_model`, `dn_net_model` | Controller's internal model |
| **Network state** | `tn_net_model`, `dn_net_model` | For Jacobian computation |
| **Actuator ratings** | `combined_net` | Real device ratings |

---

## Status of Run Scripts

| Script | Status | Notes |
|--------|--------|-------|
| `run_dso_reactive_power_control.py` | ✅ Correct | Reference implementation |
| `run_cascade.py` | ✅ Correct | Measures from combined, sensitivities from models |
| `run_tso_voltage_control.py` | ✅ Fixed (2026-02-10) | Updated to match correct architecture |

---

## Related Documentation

- `ABSTRACT_BASE_FOR_TSO_DSO_OFO.pdf`: Controller algorithm description
- `pscc2026_schwenke.pdf`: Online Feedback Optimisation mathematical formulation
- `implementation_status.md`: Module implementation status
- `network/split_tn_dn_net.py`: Network splitting implementation

---

**Last Updated:** 2026-02-10  
**Author:** Manuel Schwenke
