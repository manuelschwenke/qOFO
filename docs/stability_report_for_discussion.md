# OFO-MIQP Stability Framework — Full Technical Report

**Purpose**: Reference document for discussing the stability analysis and auto-tuning with Claude. Covers the mathematical theory, controller architecture, actuator variables, control objectives, and the current conditioning-pump tuning algorithm.

---

## 1. Controller Architecture

### 1.1 Hierarchical Structure

```
N TSO zones (decentralised, no direct communication)
  |
  |-- Zone i runs TSO OFO-MIQP controller
  |     |
  |     |-- Sends q_set to M_i DSO areas
  |     |-- Receives v_TSO, i_line measurements from transmission grid
  |     |
  |     +-- DSO_j OFO-MIQP controller (j = 1..M_i)
  |           |
  |           |-- Tracks q_set from TSO (primary)
  |           |-- Tracks v_set locally (secondary)
  |           |-- Controls DER Q, OLTCs, shunts in distribution grid
  |           |
  |           +-- Distribution network plant
  |
  +-- Shared transmission grid (coupling between zones)
```

### 1.2 Timescale Hierarchy

```
tau_plant  <<  T_DSO  <<  T_TSO

tau_plant:  plant settling time (seconds)
T_DSO:      DSO OFO period (e.g. 20 seconds)
T_TSO:      TSO OFO period (e.g. 3 minutes)

N_inner = T_TSO / T_DSO  (DSO iterations per TSO step, e.g. 9)
```

The DSO must converge within N_inner iterations before the TSO takes its next step. This is checked by the cascade margin: rho(M_cont)^N_inner << 1.

---

## 2. Actuator Variables

### 2.1 TSO Control Variables (per zone)

Column ordering in H matrix: `[Q_DER | Q_PCC_set | V_gen | s_OLTC | s_shunt]`

| Type | Symbol | Nature | Description |
|------|--------|--------|-------------|
| Q_DER | u_DER in R^n_der | Continuous | Reactive power setpoints of transmission-connected DERs |
| Q_PCC_set | u_PCC in R^n_pcc | Continuous | Reactive power setpoints at TSO-DSO interface transformers (sent to DSOs) |
| V_gen | u_gen in R^n_gen | Continuous | Generator AVR voltage setpoints |
| s_OLTC | d_OLTC in Z^n_oltc | Discrete | Machine-transformer OLTC tap positions (step +/-1 per iteration) |
| s_shunt | d_shunt in Z^n_shunt | Discrete | Switchable shunt stages |

### 2.2 DSO Control Variables (per DSO)

Column ordering: `[Q_DER | s_OLTC | s_shunt]`

| Type | Symbol | Nature | Description |
|------|--------|--------|-------------|
| Q_DER | u_DER in R^n_der | Continuous | DER reactive power in distribution network |
| s_OLTC | d_OLTC in Z^n_oltc | Discrete | Distribution coupler OLTC taps |
| s_shunt | d_shunt in Z^n_shunt | Discrete | Tertiary winding shunts |

### 2.3 Output Variables

**TSO outputs**: `y_TSO = [V_bus (p.u.) | I_line (p.u.)]`

**DSO outputs**: `y_DSO = [Q_interface (Mvar) | V_bus (p.u.) | I_line (p.u.)]`

Current rows (I_line) have weight 0 in the objective — they are constraint-only, not tracked.

---

## 3. Control Objectives

### 3.1 Quadratic Tracking Cost

The general cost function for any controller layer:

```
f(u, y) = (y - y_set)^T Q (y - y_set) + (u - u_ref)^T R (u - u_ref)
```

where:
- Q = diag(q_1, ..., q_p) >= 0 : per-output tracking weight
- R = diag(r_1, ..., r_n) >= 0 : input regularisation (typically 0)

### 3.2 TSO Objective

```
Q_obj,TSO = diag(g_v, ..., g_v, 0, ..., 0)
                 |--- n_v ---|  |-- n_i --|
```

Voltages tracked with weight g_v. Currents are constraint-only (weight 0).

The gradient is:
```
nabla_f = 2 * g_v * (V - V_set)^T * dV/du
```

### 3.3 DSO Objective

Three components in the gradient:

1. **Q-interface tracking** (primary):
   ```
   nabla_f_q = 2 * g_q * (Q_interface - Q_set)^T * dQ/du
   ```

2. **Integral Q-tracking** (leaky integrator, optional):
   ```
   s_{k+1} = lambda_qi * s_k + (Q_interface - Q_set)
   s_k = clip(s_k, [-q_max, +q_max])         (anti-windup)
   nabla_f_qi = 2 * g_qi * s^T * dQ/du
   ```

3. **Voltage tracking** (secondary):
   ```
   nabla_f_v = 2 * dso_g_v * (V - V_set)^T * dV/du
   ```

The per-output weight vector is:
```
Q_obj,DSO = diag(g_q, ..., g_q, dso_g_v, ..., dso_g_v, 0, ..., 0)
                 |-- n_iface --|  |---- n_v ----|  |-- n_i --|
```

---

## 4. OFO-MIQP Iteration

### 4.1 Update Rule

At each iteration k, the controller solves:

```
u^{k+1} = u^k + sigma^k
```

where sigma^k solves the MIQP:

```
min_{w, z}  w^T G_w w + nabla_f^T w + z^T G_z z

subject to:
  u_lower <= u^k + w <= u_upper              (input bounds)
  y_lower - z <= y^k + H * w <= y_upper + z  (output bounds with slack)
  z >= 0                                      (slack non-negativity)
  w_a in Z  for a in discrete                 (integrality)
  |w_a| <= 1  for a in discrete               (max one step per iteration)
```

### 4.2 Weight Matrices

**G_w** (control change penalty):
- For continuous variables: `G_w[i,i] = g_w[i] + alpha^2 * g_u[i]`
- For integer variables: `G_w[i,i] = g_w[i] + g_u[i]` (no alpha factor)

**G_z** (slack penalty):
- g_z ~ 1e12 for voltage/Q constraints (near-hard)
- g_z ~ 1e3 for current constraints (soft, weakly controllable)

### 4.3 Step-Size Alpha

Continuous variables use alpha as a step-size multiplier:
```
sigma_continuous = alpha * w_continuous
sigma_discrete = w_discrete            (always alpha=1 for integers)
```

### 4.4 Sensitivity Matrix H

```
H = [dV/dQ_DER | dV/dQ_PCC | dV/dV_gen | dV/ds_OLTC | dV/ds_shunt]
    [dI/dQ_DER | dI/dQ_PCC | dI/dV_gen | dI/ds_OLTC | dI/ds_shunt]
```

For DSO:
```
H = [dQ_iface/dQ_DER | dQ_iface/ds_OLTC | dQ_iface/ds_shunt]
    [dV/dQ_DER       | dV/ds_OLTC       | dV/ds_shunt      ]
    [dI/dQ_DER       | dI/ds_OLTC       | dI/ds_shunt      ]
```

Computed from the power flow Jacobian via Schur complement.

---

## 5. Stability Theory (Theorem 3.3)

### 5.1 Main Result

The complete multi-TSO/DSO hierarchy is stable iff three conditions hold simultaneously:

```
C1:  rho(I - alpha_j * M_cont,j) < 1     for each DSO j
C2:  rho(I - alpha * M_full^c) < 1        (multi-zone TSO, continuous only)
C3:  rho(Gamma) < 1                        (multi-zone TSO, discrete small-gain)
```

### 5.2 C1: DSO Inner-Loop Stability (Theorem 1.2)

For each DSO j, partition H_dso into continuous and discrete columns:
```
H_dso = [K_c | K_d]
```

Build the continuous curvature of the reduced cost:
```
C_c = R_c + (K_c)^T * Q * K_c
```

where R_c = diag(g_u) is the input regularisation (typically zeros for DSO).

Note: The Hessian is nabla^2 f = 2*C_c, but the factor 2 cancels with the 1/2
from the MIQP first-order condition (w* = -grad_f / (2*G_w)).  The effective
iteration gain is alpha * G_w^{-1} * C_c (not 2*C_c).

Precondition via the similarity transform:
```
M_cont = G_c^{-1/2} * C_c * G_c^{-1/2}
```

This is symmetric PSD with eigenvalues lambda_i(M_cont).

**Contraction condition**: The fixed-point iteration converges iff
```
rho(I - alpha * M_cont) = max_i |1 - alpha * lambda_i| < 1
```

which requires all `alpha * lambda_i in (0, 2)`.

**Optimal step size**:
```
alpha_opt = 2 / (lambda_max + lambda_min)
rho_opt = (kappa - 1) / (kappa + 1)    where kappa = lambda_max / lambda_min
```

**Per-actuator necessary condition** (Gershgorin):
```
g_w[i] > alpha * Phi_c[i,i] / 2
```

**Cascade margin** (Assumption 2.2):
```
rho(M_cont)^N_inner << 1
```
where N_inner = T_TSO / T_DSO. If this holds, the DSO has approximately converged before the TSO takes its next step.

**Discrete variables**: For a single DSO (no cross-zone interaction), discrete variables settle after finitely many switches (Proposition 1.3). No additional discrete condition is needed.

### 5.3 C2: Multi-Zone Continuous Stability (Theorem 3.1)

With all discrete variables frozen across all zones, extract only continuous columns of H_ij:

```
K_ij^c = H_ij[:, continuous_columns_of_zone_j]
```

Build the block curvature:
```
[C_full^c]_ij = (K_ii^c)^T * Q_i * K_ij^c             (off-diagonal, i != j)
[C_full^c]_ii = R_i^c + (K_ii^c)^T * Q_i * K_ii^c     (diagonal)
```

**Important**: The cross-curvature C_ij uses H_ii^T (zone i's OWN sensitivity) on the left, not H_ij^T. This means C_ij != C_ji^T in general -- the system matrix M_full^c is NOT symmetric.

Precondition:
```
M_ij^c = G_w,i^{-1/2} * [Phi_full^c]_ij * G_w,j^{-1/2}
```

Assemble the full block matrix:
```
M_full^c = [[M_ij^c]]     (N_zones x N_zones blocks)
```

**Contraction condition**:
```
rho(I - alpha * M_full^c) < 1
```

**Decentralised sufficient condition** (Corollary 3.1, small-gain):
```
gamma = max_i { rho_i + sum_{j!=i} alpha * ||M_ij^c||_2 } < 1
```

### 5.4 C3: Multi-Zone Discrete Small-Gain (Theorem 3.2)

Prevents cross-zone tap cycling. Extract discrete columns:
```
K_ij^d = H_ij[:, discrete_columns_of_zone_j]
```

Build the cross-zone discrete interaction matrix:
```
P_ij = (K_ii^d)^T * Q_i * K_ij^d     in R^{n_d_i x n_d_j}
```

Entry [P_ij]_{ab} measures how strongly zone j's tap b affects zone i's gradient for tap a (Q-weighted correlation).

Build the normalised gain matrix Gamma (N_zones x N_zones):
```
Gamma_ij = max_a (2 / g_{i,a}) * sum_b |[P_ij]_{ab}|     for i != j
Gamma_ii = 0
```

The factor 2 is a rounding-threshold margin: an integer variable triggers when
the continuous relaxation |w| >= 0.5, so g_w > 2 * row_L1 prevents triggering.
(The Hessian factor 2 and MIQP factor 1/2 cancel, as in the continuous case.)

**Stability condition**:
```
rho(Gamma) < 1
```

**G sizing rule** (Corollary 3.2) — closed-form lower bound on discrete g_w:
```
g_{i,a} > 2 * sum_{j!=i} ||[P_ij]_{a,.}||_1
```

This is constructive: compute the interaction, apply safety factor, done.

### 5.5 What Each Condition Prevents

| Condition | What it prevents | Design lever |
|-----------|-----------------|--------------|
| C1: rho(I - alpha*M_cont,j) < 1 | DSO continuous oscillation | DSO g_w (conditioning), DSO alpha (stability) |
| C2: rho(I - alpha*M_full^c) < 1 | Multi-zone TSO continuous oscillation | TSO g_w^c (conditioning), TSO alpha (stability) |
| C3: rho(Gamma) < 1 | Cross-zone tap ping-pong (discrete cycling) | TSO g_w^d (via sizing rule) |

---

## 6. Current Auto-Tuning Algorithm

### 6.1 Design Principle

**g_w controls conditioning** (relative eigenvalue scaling, kappa).
**alpha controls stability** (overall step size, rho).

These are distinct roles. The tuner must NOT use g_w to brute-force stability — that makes actuators infinitely slow.

### 6.2 Three-Phase Tuning

The auto_tune() function runs: DSO (C1) -> continuous TSO (C2) -> discrete TSO (C3).

#### Phase 1: Gershgorin Preconditioning

For each actuator i, compute the self-curvature:
```
C_ii = diag(H^T Q H)_i
```

Set g_w proportional to curvature:
```
g_w[i] = safety_factor * C_ii / 2
```

This normalises M so that diagonal entries are O(1). The safety_factor is:
- 2.0 for continuous actuators (DER, PCC, V_gen)
- 3.0 for discrete actuators (OLTC, shunt)

User-specified g_w acts as a floor: `g_w = max(g_w_gersh, g_w_user)`.

#### Phase 2: Conditioning Pump (C1 and C2)

If kappa = lambda_max / lambda_min > kappa_target (default 50), iteratively boost g_w for actuators in the fastest mode:

```
for iteration = 1..max_iters:
    M, eigs = build_M(gw)
    kappa = lambda_max / lambda_min
    if kappa <= kappa_target: break

    v = eigenvector of lambda_max(M)
    v_sq = v^2 / sum(v^2)             (participation weights)
    boost = 1 + (1.3 - 1) * v_sq / max(v_sq)
    gw = gw * boost                   (boost fastest-mode actuators)
```

This reduces lambda_max by making the fast actuators heavier, without touching lambda_min (which involves different actuators).

After the pump, compute alpha from the eigenspectrum:
```
alpha = find_alpha_for_target_rho(eigs, rho_target=0.95)
```

The alpha search uses ternary search for the minimum rho, then binary search for the largest alpha achieving rho <= 0.95.

#### Phase 3: Discrete G Sizing Rule (C3)

Closed-form, no iteration needed:
```
g_{i,a} = safety_factor * 2 * sum_{j!=i} ||[P_ij]_{a,.}||_1
```

where P_ij = (K_ii^d)^T Q_i K_ij^d.

Safety factor default: 1.5.

### 6.3 Current Issues / Open Questions

1. **DSO g_w still inflates significantly** (x20 from init). The Gershgorin bound uses the full curvature C_ii = (H^T Q H)_ii, which can be large when Q (objective weight) is large. Is the Gershgorin bound too conservative, or is the user's init g_w genuinely too small?

2. **Conditioning pump targets kappa_target = 50**. Is this the right target? For kappa = 50, rho_opt = 49/51 = 0.96, which is slow. Should we target kappa = 10 (rho_opt = 0.82)?

3. **Alpha interaction with G_w**: In the MIQP, continuous variables use alpha as a step-size: `sigma = alpha * w`. But the MIQP objective includes `G_w[i,i] = g_w[i] + alpha^2 * g_u[i]`. So alpha appears both as a step-size multiplier and inside the quadratic weight. The stability analysis currently handles this by checking `rho(I - alpha * M)` where M is built from g_w (not g_w + alpha^2 * g_u). Is this consistent?

4. **Zones 1 and 3 show lambda_max = 0, kappa = 1, rho = 0** in the stability report. These zones have only V_gen and OLTC actuators (no DER, no PCC). When V_gen is excluded from stability (config.exclude_from_stability = {'gen'}), these zones have zero continuous actuators and are trivially stable. Is this correct, or should excluded actuators still be reported?

5. **The conditioning pump only runs when kappa > 50**. After Gershgorin preconditioning, kappa is often already < 50, so the pump does nothing (0 iterations). The stability failure then comes from alpha being too small, not from poor conditioning. Should the pump also target an absolute lambda_max bound?

---

## 7. Sensitivity Computation

Sensitivities are computed from the power flow Jacobian via Schur complement:

```
J = [[J_Ptheta  J_PV ]     (power flow Jacobian at current operating point)
     [J_Qtheta  J_QV ]]

J_reduced = J_QV - J_Qtheta * J_Ptheta^{-1} * J_PV    (Schur complement)

dV/dQ = J_reduced^{-1}    (voltage sensitivity to reactive power injection)
```

Other sensitivities (dV/ds_OLTC, dV/dV_gen, dI/dQ, etc.) are derived from this base computation with appropriate chain rules.

### 7.1 Cross-Zone Sensitivities

For multi-zone analysis, the coordinator computes H_ij blocks:
```
H_ij = [dV_i/dQ_DER_j  |  dV_i/dQ_PCC_j  |  dV_i/dV_gen_j  |  dV_i/ds_OLTC_j]
       [dI_i/dQ_DER_j  |  dI_i/dQ_PCC_j  |  dI_i/dV_gen_j  |  dI_i/ds_OLTC_j]
```

where the observation rows (V_i, I_i) belong to zone i and the control columns belong to zone j.

---

## 8. Notation Summary

| Symbol | Definition | Typical values |
|--------|-----------|----------------|
| H_ij | Sensitivity: zone i outputs to zone j inputs | from Jacobian |
| K_ij^c | Continuous columns of H_ij | DER, PCC, V_gen columns |
| K_ij^d | Discrete columns of H_ij | OLTC, shunt columns |
| Q_i | Per-output objective weight diagonal | g_v for voltages, 0 for currents |
| G_w | Per-actuator change weight diagonal | 2-50 for DER, 1e6 for V_gen |
| G_u = R | Input regularisation diagonal | typically 0 |
| alpha | Step-size (continuous actuators only) | 0.3 - 1.3 |
| C_c | Continuous curvature: R + (K^c)^T Q K^c | |
| M | Preconditioned curvature: G_w^{-1/2} C_c G_w^{-1/2} | eigenvalues in (0, 2/alpha) for stability |
| kappa | Condition number: lambda_max / lambda_min | target < 50 |
| rho | Spectral radius: max |1 - alpha * lambda_i| | < 1 for stability |
| P_ij | Discrete interaction: (K_ii^d)^T Q_i K_ij^d | |
| Gamma | Normalised discrete gain matrix | rho(Gamma) < 1 for C3 |
| N_inner | T_TSO / T_DSO | e.g. 9 |
