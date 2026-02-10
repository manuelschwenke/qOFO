# Critical Fix: Measurement Architecture (2026-02-10)

## Problem Identified

The original `run_cascade.py` implementation had a **fundamental architectural error** in how measurements were extracted from the system:

### Incorrect Pattern (BEFORE)
- Measurements were extracted from **split networks** (tn_net, dn_net)
- Element indices were pre-filtered by network type
- This violated the principle that split networks are **models only**

### Consequences
- Measurements did not reflect the actual physical system state
- Index mismatches between plant and controller models
- Potential feedback errors in the closed-loop control

## Solution Implemented

Following the **correct pattern** from `run_dso_reactive_power_control.py`, the architecture now properly separates:

### 1. Real Plant (combined_net)
```python
combined_net = build_tuda_net(...)  # The PHYSICAL SYSTEM
```
- **All measurements taken from here**
- **All controls applied to here**
- **All power flows run on this**
- This is the "real world" that the controllers interact with

### 2. Controller Models (tn_net, dn_net)
```python
split = split_network(combined_net, ...)
tn_net = split.tn_net  # TSO MODEL (sensitivities only)
dn_net = split.dn_net  # DSO MODEL (sensitivities only)
```
- **Used ONLY for sensitivity calculations**
- Used for NetworkState creation
- **Never used for measurements or control application**
- These are simplified models for gradient computation

## Key Changes in run_cascade.py

### Function: `measurement_from_combined_tn_side()`

**BEFORE:**
```python
def measurement_from_combined_tn_side(net, ...):
    # Pre-filtered TN buses
    tn_bus_indices = [b for b in net.bus.index 
                      if b not in split.dn_only_bus_indices]
    all_bus = np.array(sorted(tn_bus_indices), dtype=np.int64)
    # ...
```

**AFTER:**
```python
def measurement_from_combined_tn_side(combined_net, ...):
    # ALL bus indices from combined network (real plant)
    all_bus = np.array(sorted(combined_net.res_bus.index), dtype=np.int64)
    vm = combined_net.res_bus.loc[all_bus, "vm_pu"].values
    # ...
```

### Function: `measurement_from_combined_dn_side()`

**BEFORE:**
```python
def measurement_from_combined_dn_side(net, ...):
    # DN bus indices (pre-filtered)
    dn_bus_indices = list(split.dn_only_bus_indices)
    all_bus = np.array(sorted(dn_bus_indices), dtype=np.int64)
    # ...
```

**AFTER:**
```python
def measurement_from_combined_dn_side(combined_net, ...):
    # ALL bus indices from combined network (real plant)
    all_bus = np.array(sorted(combined_net.res_bus.index), dtype=np.int64)
    vm = combined_net.res_bus.loc[all_bus, "vm_pu"].values
    # ...
```

### Element Index Identification

**BEFORE:**
- Indices identified from split networks
- Risk of inconsistency between plant and models

**AFTER:**
```python
# ALL element indices identified from COMBINED NETWORK (real plant)
tso_der_buses = []
for sidx in combined_net.sgen.index:  # <-- combined_net, not tn_net!
    bus = int(combined_net.sgen.at[sidx, "bus"])
    if bus not in split.dn_only_bus_indices:
        # ...
```

## Control Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    COMBINED NETWORK                         │
│                  (Real Physical Plant)                      │
│                                                             │
│  • 380/110/20 kV buses                                      │
│  • Lines, transformers, generators                          │
│  • DER, loads, shunts                                       │
│                                                             │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │ Measurements │        │   Controls   │                  │
│  │  (READ)      │        │   (WRITE)    │                  │
│  └──────┬───────┘        └──────▲───────┘                  │
└─────────┼───────────────────────┼──────────────────────────┘
          │                       │
          │                       │
          ▼                       │
  ┌──────────────┐       ┌───────┴────────┐
  │ TSO Meas.    │       │  TSO Control   │
  │              │       │     Output     │
  └──────┬───────┘       └───────▲────────┘
         │                       │
         │     ┌─────────────────┴─────────────────┐
         │     │    TSO CONTROLLER                 │
         └────►│                                   │
               │  Sensitivities from: TN_NET       │◄────┐
               │  (model only)                     │     │
               └───────────────────────────────────┘     │
                                                          │
  ┌──────────────┐       ┌───────────────┐          ┌───┴────┐
  │ DSO Meas.    │       │  DSO Control  │          │ TN_NET │
  │              │       │     Output    │          │ (model)│
  └──────┬───────┘       └───────▲───────┘          └────────┘
         │                       │
         │     ┌─────────────────┴─────────────────┐
         │     │    DSO CONTROLLER                 │
         └────►│                                   │
               │  Sensitivities from: DN_NET       │◄────┐
               │  (model only)                     │     │
               └───────────────────────────────────┘     │
                                                          │
                                                      ┌───┴────┐
                                                      │ DN_NET │
                                                      │ (model)│
                                                      └────────┘
```

## Verification

The fix can be verified by checking:

1. **Measurement source**: All `measurement_from_*()` calls use `combined_net`
2. **Control target**: All `apply_*_controls()` calls modify `combined_net`
3. **Power flow**: `pp.runpp(combined_net, ...)` operates on the plant
4. **Sensitivities**: Created from `JacobianSensitivities(tn_net)` or `JacobianSensitivities(dn_net)`
5. **NetworkState**: Created from model networks (`tn_net`, `dn_net`)

## Benefits of This Architecture

1. **Clear separation of concerns**:
   - Plant = reality (measurements, control effects)
   - Models = simplified representations (gradient computation)

2. **Realistic simulation**:
   - Measurements reflect true system state after all control actions
   - Controllers receive feedback from the actual plant

3. **Model-plant mismatch**:
   - Controllers use simplified models for sensitivity computation
   - Any model errors are corrected by feedback in subsequent iterations
   - This is closer to real-world operation

4. **Consistency**:
   - All run scripts now follow the same pattern
   - Easier to understand and maintain

## Testing Recommendations

1. **Run cascade simulation**:
   ```bash
   python run_cascade.py
   ```
   - Check that it executes without errors
   - Verify voltage convergence

2. **Compare with numerical sensitivities**:
   ```python
   use_numerical = True  # in run_cascade.py main()
   ```
   - Validates that sensitivity calculations are correct

3. **Check measurement consistency**:
   - Add debug prints to verify measurements come from combined_net
   - Verify element indices match between plant and controllers

## Related Files

- `run_cascade.py` - **Fixed** ✅
- `run_dso_reactive_power_control.py` - **Correct reference** ✅
- `run_tso_voltage_control.py` - **Needs review** ⚠️

## Authors

- Original implementation: Manuel Schwenke
- Critical fix identified and implemented: 2026-02-10

## References

- PSCC 2026 paper (Schwenke et al.)
- CIGRÉ 2026 Synopsis
- Implementation status document
