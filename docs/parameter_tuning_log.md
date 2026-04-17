# Manual Parameter Tuning Log -- Multi-TSO/DSO (IEEE 39-bus)

**Date:** 2026-04-13
**Script:** `run/run_M_TSO_M_DSO.py`
**Network:** IEEE 39-bus, 3 zones (fixed literature partition), 3 HV sub-networks (DSO_1..DSO_3)
**Stability criterion:** Theorem 3.3 (C1 DSO inner loops, C2 multi-zone continuous, C3 discrete small-gain)

## Starting configuration

| Parameter       | Value   | Role                                  |
|-----------------|---------|---------------------------------------|
| g_v             | 5000    | TSO voltage tracking weight           |
| g_q             | 30      | DSO Q-interface tracking weight       |
| dso_g_v         | 10000   | DSO voltage tracking weight           |
| g_w_der         | 10      | TSO DER Q regularisation              |
| g_w_gen         | 1e7     | TSO generator AVR regularisation      |
| g_w_pcc         | 20      | TSO PCC Q setpoint regularisation     |
| g_w_tso_oltc    | 2       | TSO OLTC tap regularisation           |
| g_w_dso_der     | 10      | DSO DER Q regularisation              |
| g_w_dso_oltc    | 100     | DSO OLTC tap regularisation           |
| dso_period_s    | 10      | DSO control period [s]                |
| live_plot       | True    | Real-time plotting                    |

Initial stability verdict: **UNSTABLE** (all three conditions failing).

## Tuning procedure

### Step 1: Disable live plotting

Set `live_plot=False` to get a clean baseline and faster iteration.
Shortened simulation to 60 min for rapid feedback.

### Step 2: Baseline run (run 1)

**Result:** UNSTABLE

| Condition | Status | Key metric                            |
|-----------|--------|---------------------------------------|
| C1        | FAIL   | DSO rho = 306--361 (wildly unstable)  |
| C2        | pass   | rho(M_full^c) = 0.986                 |
| C3        | FAIL   | rho(Gamma) = 1.35                     |

**Diagnosis:**
- C1: DSO g_w_dso_der = 10 is far too small.  Per-actuator deficits up to 350.
- C3: TSO OLTC g_w = 2 violates the G sizing rule.  Zone 2 OLTC_0 needs >= 6.54,
  Zone 3 OLTC_2 needs >= 2.63.
- C2: Zone 2 at rho = 0.986 (lam_max = 1.08, kappa = 77.2) -- marginal but passing.

### Step 3: Fix C3 -- increase g_w_tso_oltc (run 2)

Changed `g_w_tso_oltc = 2 -> 10`.  This satisfies the G sizing rule for all zones
with comfortable margin (Zone 2 OLTC_0 needs 6.43, now has 10.0).

**Result:** C3 passes (rho(Gamma) = 0.27).

### Step 4: Fix C1 -- increase g_w_dso_der iteratively (runs 2--8)

The DSO inner-loop stability requires rho(M_cont) < 1 for each DSO.  The dominant
eigenvalue of the DSO iteration matrix scales roughly inversely with g_w_dso_der.

| Run | g_w_dso_der | rho(DSO_1) | rho(DSO_2) | rho(DSO_3) | C1 |
|-----|-------------|------------|------------|------------|--------|
| 2   | 10          | 336.7      | 361.0      | 306.4      | FAIL   |
| 3   | 400         | 7.44       | 8.05       | 6.68       | FAIL   |
| 4   | 3200        | 0.98       | 0.97       | 0.98       | pass   |

At g_w_dso_der = 3200, all three conditions pass.  However, DSO Q-tracking quality
is very poor (final |err| = 26--34 Mvar, ~45% of setpoints).

**Root cause of poor Q-tracking:**  With dso_g_v = 10000 and g_q = 30, the DSO
Hessian is dominated by the voltage term (333x larger than Q-tracking).  The required
g_w_dso_der must be large enough to damp the dso_g_v-driven eigenvalue, but this
also prevents the DER from responding to Q setpoint changes.  The steady-state
tracking error ratio is approximately g_w / (g_w + h^2 * g_q), which at g_w = 3200
and g_q = 30 gives ~50% error.

### Step 5: Improve Q-tracking -- reduce dso_g_v proportionally (runs 9--12)

**Key insight:** Reducing dso_g_v reduces the dominant Hessian eigenvalue, allowing a
proportional reduction of g_w_dso_der while maintaining C1 stability.  The voltage
hard constraints (g_z_voltage = 1e-12) prevent voltage violations regardless of
dso_g_v.  The parameter dso_g_v only controls how aggressively voltages track V_set
*within* the constraint band.

Also reduced g_w_dso_oltc from 100 to 50 (let OLTCs handle voltage more actively,
freeing DER for Q-tracking) and dso_period_s from 10 to 5 s (36 inner iterations
per TSO step instead of 18).

| Run | dso_g_v | g_w_dso_der | rho(DSO_2) | Mean Q err [Mvar] | C1     |
|-----|---------|-------------|------------|-------------------|--------|
| 4   | 10000   | 3200        | 0.97       | 38--43            | pass   |
| 5   | 10000   | 1900        | 0.95       | 31--35            | pass   |
| 9   | 3000    | 700         | 0.98       | 19--21            | pass   |
| 10  | 2000    | 500         | 0.98       | 16--17            | pass   |
| 11  | 1000    | 300         | 1.03       | 12--13            | FAIL   |
| 12  | 1000    | 350         | 0.96       | 13--14            | pass   |

The boundary for C1 stability at dso_g_v = 1000 is g_w_dso_der ~ 330.  Setting
g_w_dso_der = 350 provides adequate margin (rho_max = 0.96).

### Step 6: Full verification (run 13)

Restored n_total_s to 720 min (12 hours) with contingencies:
- t = 120 min: trip gen[2] (Zone 3)
- t = 300 min: restore gen[2]

**Stability:**

| Condition | Status | Metric                   |
|-----------|--------|--------------------------|
| C1        | pass   | rho = 0.95--0.98         |
| C2        | pass   | rho(M_full^c) = 0.986    |
| C3        | pass   | rho(Gamma) = 0.27        |

Cascade margins: DSO_1 = 0.59, DSO_2 = 0.78, DSO_3 = 0.60.

**Voltage tracking (final, t = 720 min):**

| Zone   | V_mean [p.u.] | |V - V_set| [p.u.] |
|--------|---------------|---------------------|
| Zone 1 | 1.039         | 0.009               |
| Zone 2 | 1.033         | 0.003               |
| Zone 3 | 1.031         | 0.001               |

All voltages remained within [0.9, 1.1] p.u. throughout the simulation, including
during the gen-2 trip/restore transient.

**DSO Q-tracking quality:**

| DSO   | Mean |err| [Mvar] | Final |err| [Mvar] | Max |err| [Mvar] |
|-------|---------------------|---------------------|-------------------|
| DSO_1 | 9.3                 | 8.4                 | 47.9 (cold start) |
| DSO_2 | 10.7                | 10.0                | 48.0 (cold start) |
| DSO_3 | 5.8                 | 1.0                 | 47.9 (cold start) |

The max errors of ~48 Mvar occur only during the initial cold-start transient
(first TSO step).  Steady-state tracking errors are 1--10 Mvar.

**Contingency response:** Gen-2 trip at t = 120 min caused Zone 3 voltages to drop
from [1.028, 1.037] to [1.020, 1.029] p.u. -- handled smoothly without oscillation.
Gen-2 restore at t = 300 min recovered voltages to [1.028, 1.037] within one TSO
step.  No divergence or constraint violations during either transient.

## Step 7: Diagnose structural Q-tracking barriers (runs 14--16)

A diagnostic printout (Step 7d in the simulation) was added to reveal the
underlying network quantities at t=0.

### Finding 1: Transmission line lengths are unrealistic

The IEEE 39-bus line lengths between coupling buses are **1.0 km** at 345 kV,
essentially bus bars.  In a real EHV network, substations are 30--100+ km apart.
The pandapower case39 uses per-unit impedances without meaningful physical
distances.

HV sub-network lines span 180--240 km total at 110 kV -- realistic for regional
distribution, but the impedance ratio HV/TN is extreme.  DSO_2's higher impedance
(scale=1.0 vs 0.75) explains its worse Q-tracking: the sensitivity dQ_interface/dQ_DER
is attenuated by the longer electrical paths.

### Finding 2: Load Q consumption is nearly zero

Q_load = -6 Mvar at P = 96 MW (cos_phi ~ 0.998).  Real HV/MV substations
aggregating distribution feeders typically have cos_phi ~ 0.95--0.97, consuming
20--30 Mvar at this P level.  The nearly-unity power factor is a consequence of the
simbench "mv_rural_qload" profile at this timestep.

### Finding 3: Q-tracking limited by multi-objective trade-off, not capacity

DER Q capacity is [-135, +168] Mvar per DSO.  Setpoints of 50--150 Mvar use only
30--90% of the positive range -- ample capacity.

The **root cause** of the ~10 Mvar steady-state tracking error was the DSO's
multi-objective trade-off.  At MIQP equilibrium the gradient condition is:

    g_q * H_q^T * e_q + g_v * H_v^T * e_v = 0

With the old ratio dso_g_v / g_q = 1000 / 30 = 33:1, even small voltage deviations
dominated the gradient, leaving persistent Q error.  Since voltage limits are already
enforced by the hard constraint (g_z_voltage = 1e-12), the dso_g_v term is redundant
for safety -- it only controls how aggressively V tracks V_set within the safe band.

### Fix: shift DSO objective toward Q-tracking

The weights g_q and dso_g_v are NOT comparable at face value because their
outputs live on different physical scales.  Typical deviations are ~0.02 p.u.
for voltage and ~10 Mvar for Q.  The per-output cost contributions scale as:

    J_v ~ g_v * (0.02)^2 = g_v * 4e-4       (per bus)
    J_q ~ g_q * (10)^2   = g_q * 100         (per interface)

With the old g_v=1000, g_q=30: J_v ~ 0.4 per bus (x10 buses = 4.0) vs
J_q ~ 3000 per interface (x3 = 9000).  Q already dominated numerically,
but the voltage term was still large enough to bias the gradient at small
V errors, creating ~10 Mvar steady-state Q error.

Setting g_v=100 effectively removes the soft voltage objective.  Voltage
control now relies entirely on the hard output constraint (g_z_voltage)
and the DSO OLTCs.  This is acceptable because the DSO's primary mandate
is Q-interface tracking; voltage is a constraint, not an objective.

| Parameter    | Before | After | Ratio change |
|--------------|--------|-------|-------------|
| g_q          | 30     | 100   | 3.3x up     |
| dso_g_v      | 1000   | 100   | 10x down    |
| g_w_dso_der  | 350    | 650   | 1.9x up (for C1) |

**Result** (720-min, gen trip/restore):

| DSO   | Mean |err| old | Mean |err| new | Improvement |
|-------|---------------------|---------------------|-------------|
| DSO_1 | 9.3 Mvar            | 2.3 Mvar            | 4.0x        |
| DSO_2 | 10.7 Mvar           | 2.1 Mvar            | 5.1x        |
| DSO_3 | 5.8 Mvar            | 2.1 Mvar            | 2.8x        |

DSO stability margins also improved: rho = 0.82--0.85 (was 0.95--0.98).

## Final tuned parameters

```python
cfg = MultiTSOConfig(
    dso_period_s=5.0,          # 5 s (was 10 s)
    g_v=5000.0,                # unchanged
    g_q=100,                   # was 30
    dso_g_v=100.0,             # was 10000
    g_w_der=10,                # unchanged
    g_w_gen=1e7,               # unchanged
    g_w_pcc=20,                # unchanged
    g_w_tso_oltc=10,           # was 2
    g_w_dso_der=650,           # was 10
    g_w_dso_oltc=50,           # was 100
    live_plot=False,           # was True
)
```

## Network realism observations

1. **TN line lengths:** IEEE 39-bus lines use 1.0 km with inflated ohm/km values
   to encode the standard per-unit impedances.  The actual impedances correspond
   to realistic EHV distances (16--135 km equivalent at 0.32 ohm/km for 345 kV
   overhead lines).  Example: line 2--3 has x=25.4 ohm total, equivalent to 79 km;
   line 7--8 has x=43.2 ohm, equivalent to 135 km.

2. **HV/MV substation Q:** At the initial operating point, aggregated loads consume
   only -6 Mvar (slightly capacitive) at P = 96 MW.  Real substations would draw
   20--30 Mvar inductive at this power level (cos_phi ~ 0.95).  This is a
   consequence of the simbench ``mv_rural_qload`` profile at this timestep.

3. **DSO_2 impedance asymmetry:** DSO_2 uses line_length_scale = 1.0 vs 0.75 for
   DSO_1/DSO_3.  This 33% higher impedance attenuates Q sensitivity and explains
   DSO_2's systematically higher tracking errors at equal parameter settings.

## Open points

1. **C2 condition number:** Zone 2 has kappa = 77.7, meaning one control direction
   converges ~78x slower than the fastest.  Per-actuator g_w scaling could improve
   this but requires individual sensitivity analysis.

2. **Cold-start transient:** The ~48 Mvar max Q-tracking error during the first TSO
   step could be reduced by a warmup phase that ramps Q setpoints gradually.

3. **Load Q profiles:** The time-varying Q-load profile (mv_rural_qload) may create
   periods with higher or lower Q demand that temporarily stress the Q-tracking
   equilibrium.
