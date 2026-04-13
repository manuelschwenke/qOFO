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
| auto_tune_gw    | True    | Eigenvector pump auto-tuning          |
| live_plot       | True    | Real-time plotting                    |

Initial stability verdict: **UNSTABLE** (all three conditions failing).

## Tuning procedure

### Step 1: Disable auto-tune and live plotting

Set `auto_tune_gw=False`, `live_plot=False` to get a clean baseline and faster iteration.
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

## Final tuned parameters

```python
cfg = MultiTSOConfig(
    dso_period_s=5.0,          # 5 s (was 10 s)
    g_v=5000.0,                # unchanged
    g_q=30,                    # unchanged
    dso_g_v=1000.0,            # was 10000
    g_w_der=10,                # unchanged
    g_w_gen=1e7,               # unchanged
    g_w_pcc=20,                # unchanged
    g_w_tso_oltc=10,           # was 2
    g_w_dso_der=350,           # was 10
    g_w_dso_oltc=50,           # was 100
    auto_tune_gw=False,        # was True
    live_plot=False,           # was True
)
```

## Open points

1. **C2 condition number:** Zone 2 has kappa = 77.7, meaning one control direction
   converges ~78x slower than the fastest.  This is driven by the smallest eigenvalue
   of the Zone 2 TSO sub-problem (likely a weakly coupled Q_PCC direction).  Per-actuator
   g_w scaling could improve this but requires individual sensitivity analysis.

2. **DSO Q-tracking ceiling:** With dso_g_v = 1000 and g_q = 30, the fundamental
   trade-off between C1 stability (needs high g_w) and Q-tracking (needs low g_w)
   limits the achievable steady-state tracking error to ~5--10 Mvar.  Further
   improvement would require either a DSO step-size alpha < 1 (to decouple
   convergence rate from g_w) or structural changes to the DSO controller.

3. **Cold-start transient:** The ~48 Mvar max Q-tracking error during the first TSO
   step could be reduced by a warmup phase that ramps Q setpoints gradually.
