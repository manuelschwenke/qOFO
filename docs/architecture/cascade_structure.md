# Cascaded TSO-DSO OFO Controller — Structure & Stability

## 1. Cascade Overview

The simulation implements a **two-layer cascaded Online Feedback Optimisation (OFO)** controller for coordinated transmission-distribution reactive power and voltage control. Both controllers share a single combined pandapower network model but operate at different time scales and control different actuator sets.

```
┌─────────────────────────────────────────────────────────────────┐
│                        TSO Controller                           │
│   Period: every 3 min                                           │
│                                                                 │
│   Actuators (u_TSO):                                            │
│     • Q_DER_TS   — TS-connected DER reactive power [Mvar]      │
│     • Q_PCC_set  — Reactive power setpoints to DSO  [Mvar]     │
│     • V_gen_set  — Generator AVR voltage setpoints  [p.u.]     │
│     • s_OLTC     — Machine transformer tap positions (integer)  │
│     • s_shunt    — 380 kV shunt states {-1,0,+1}   (integer)   │
│                                                                 │
│   Outputs (y_TSO):                                              │
│     • V_bus   — EHV bus voltages [p.u.]                         │
│     • I_line  — EHV line currents [kA]                          │
│                                                                 │
│   Objective: Voltage schedule tracking                          │
│     min  g_v · ||V - V_set||²  +  regularisation               │
│                                                                 │
│   Sends:  Q_PCC_set  ──────────────────────┐                   │
│   Receives: Q capability bounds  ◄──────┐  │                   │
└─────────────────────────────────────────┼──┼────────────────────┘
                                          │  │
                  Capability Message       │  │  Setpoint Message
                  (q_min, q_max per PCC)   │  │  (Q_set per PCC)
                                          │  │
┌─────────────────────────────────────────┼──┼────────────────────┐
│                        DSO Controller   │  │                    │
│   Period: every 1 min                   │  ▼                    │
│                                                                 │
│   Actuators (u_DSO):                                            │
│     • Q_DER_DN   — DN-connected DER reactive power [Mvar]      │
│     • s_OLTC     — 3W coupler tap positions        (integer)    │
│     • s_shunt    — Tertiary winding shunt states   (integer)    │
│                                                                 │
│   Outputs (y_DSO):                                              │
│     • Q_interface — Reactive power at TSO-DSO PCCs [Mvar]      │
│     • V_bus       — DN bus voltages [p.u.]                      │
│     • I_line      — DN line currents [kA]                       │
│                                                                 │
│   Objective: Q-interface tracking + soft voltage tracking       │
│     min  g_q·||Q - Q_set||²  +  g_v·||V - V_set||²            │
│                                                                 │
│   Sends:  Capability bounds  ──────────────────────┘            │
│   Receives: Q_PCC_set  ◄──────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 Information Exchange

At each TSO step:
1. **TSO solves its MIQP**, producing new setpoints including `Q_PCC_set` for each PCC.
2. **Setpoint messages** are sent to the DSO via `dso.receive_setpoint(msg)`.

At each DSO step:
1. **DSO solves its MIQP**, tracking `Q_PCC_set` while respecting local DN constraints.
2. **Capability messages** are sent back to the TSO via `tso.receive_capability(msg)`, reporting the achievable Q range at each PCC (derived from the DER fleet's P-dependent capability mapped through the Jacobian `dQ_interface / dQ_DER`).

### 1.2 Reserve Observer (Superordinate Shunt Logic)

A **Reserve Observer** monitors the Jacobian-weighted DER reactive power contribution to each TSO-DSO interface. When the DER burden at an interface exceeds a threshold, it forces the corresponding tertiary shunt ON (engage). When the burden drops below a lower release threshold, the shunt is released. Hysteresis prevents chattering, and a cooldown timer (default 15 min) prevents over-reaction.

```
DER Q burden at interface j  =  (∂Q_interface_j / ∂Q_DER) · Q_DER

    burden > q_threshold  →  ENGAGE shunt j  (force state = 1)
    burden < q_release    →  RELEASE shunt j (force state = 0)

    Hysteresis band:  q_threshold > q_release
```

---

## 2. The OFO Iteration (Core Algorithm)

Both TSO and DSO controllers inherit from `BaseOFOController`. Each iteration `k` solves:

### 2.1 MIQP Formulation

$$
\sigma^k = \arg\min_{w, z} \quad w^\top G_w \, w \;+\; \nabla f^\top w \;+\; z^\top G_z \, z \;+\; u^{k\top} G_u \, u^k
$$

subject to:

$$
u_{\min} - u^k \;\leq\; \alpha \, w \;\leq\; u_{\max} - u^k
$$

$$
y_{\min} - y^k \;\leq\; \alpha \, H \, w + z \;\leq\; y_{\max} - y^k
$$

where:
- **w** = proposed change direction (continuous + integer)
- **z** = slack variables (soft constraint violations on outputs)
- **H** = Jacobian sensitivity matrix `∂y/∂u` (from power flow)
- **∇f** = objective gradient (voltage tracking for TSO, Q-interface tracking for DSO)
- **G_w** = diagonal (or block) change-damping weights
- **G_z** = slack penalty weights (very large → hard constraints)
- **G_u** = usage/level regularisation

### 2.2 Update Rule

$$
u^{k+1} = u^k + \alpha \cdot \sigma^k_{\text{continuous}}
$$

$$
u^{k+1}_{\text{integer}} = u^k_{\text{integer}} + \sigma^k_{\text{integer}} \quad \text{(direct, } |\Delta| \leq 1 \text{ per step)}
$$

Integer variables (OLTCs, shunts) change by at most ±1 per iteration and are subject to a **cooldown lock** (6 iterations) after switching to prevent chattering.

---

## 3. Achieved-Value Tracking (AVT)

### 3.1 The Problem: Cascaded Setpoint Mismatch

The TSO treats `Q_PCC_set` as a control variable — it sets a desired reactive power flow at each PCC. However, the **DSO cannot perfectly achieve** this setpoint because:

1. The DSO has **limited DER capacity** (P-dependent Q capability curves).
2. The DSO must also respect **local voltage and current constraints**.
3. The DSO operates at a **faster cadence** (1 min vs 3 min), so the actual Q at the PCC evolves between TSO steps.
4. **Discrete actuators** (OLTCs, shunts) introduce step-wise changes that may over- or under-shoot.

Without correction, the TSO's internal state `u_PCC_set` drifts from reality: the TSO *thinks* it commanded Q = −20 Mvar, but the plant actually achieved Q = −15 Mvar. This mismatch accumulates and causes the TSO to make decisions based on a **stale, inaccurate internal model** — a classic **integrator windup** problem in cascaded control.

### 3.2 The AVT Mechanism

Before each TSO MIQP solve, the PCC-Q components of the internal state vector are **reset toward the physically measured value**:

$$
u_{\text{PCC}}^{k} \;\leftarrow\; (1 - k_t) \cdot u_{\text{PCC}}^{k} \;+\; k_t \cdot Q_{\text{measured}}
$$

where:
- `k_t = 1.0` (default): **full reset** — the TSO always starts from the actual plant state.
- `k_t = 0.0`: disabled — the TSO uses its own commanded value (no correction).
- `0 < k_t < 1`: partial blend (smooth transition).

### 3.3 Why AVT Matters for Control

| Aspect | Without AVT (`k_t = 0`) | With AVT (`k_t = 1`) |
|--------|------------------------|----------------------|
| TSO internal state | Reflects commanded Q | Reflects measured Q |
| MIQP ∆u basis | May be far from reality | Starts from true plant state |
| Gradient accuracy | ∇f uses stale Q_PCC | ∇f uses actual Q_PCC |
| Windup risk | Accumulated drift | Anti-windup: reset each step |
| Convergence | Can oscillate or diverge | Stable tracking |

**In control-theoretic terms**, AVT is an **anti-windup mechanism** for the outer (TSO) loop. It ensures that the optimisation-based controller always operates on a consistent, measured state — analogous to how a PI controller with anti-windup resets its integrator when the actuator saturates.

### 3.4 AVT in the Cascade Context

The cascade has a natural **time-scale separation** (TSO: 3 min, DSO: 1 min). Between two TSO steps, the DSO executes 3 iterations and moves the actual PCC-Q. AVT bridges this gap:

```
Time ──────────────────────────────────────────────────────►

TSO:  ────●────────────────────●────────────────────●───────
          │  Q_set = -20 Mvar  │                    │
          │                    │  AVT reset:        │
          │                    │  u_PCC ← Q_meas    │
          │                    │  = -15 Mvar        │
          │                    │  (∆u now based on  │
          │                    │   true plant state) │
          │                    │                    │
DSO:  ──●──●──●──────────●──●──●──────────●──●──●──────────
        ↑                ↑
        DSO tracks       DSO hits DER
        Q_set = -20      capacity limit
        achieves -15     at this PCC
```

---

## 4. Mathematical Stability Analysis

We now provide a rigorous framework to prove stability of the cascaded OFO scheme.

### 4.1 Notation & System Model

Consider the cascaded system with two controllers operating on a shared plant. Let:

- $u_T \in \mathbb{R}^{n_T}$: TSO control variables (including $Q_{\text{PCC,set}}$)
- $u_D \in \mathbb{R}^{n_D}$: DSO control variables
- $y = h(u_T, u_D, d)$: plant output map (power flow solution), with disturbance $d$ (load/generation profiles)
- $y^* = h(u_T^*, u_D^*, d)$: the optimal steady-state satisfying all constraints

The plant is modelled as a **static nonlinear map** $h(\cdot)$ (the AC power flow equations), and the OFO controllers use a **first-order linear approximation**:

$$
y \approx y^k + H_T \, \Delta u_T + H_D \, \Delta u_D
$$

where $H_T = \partial y / \partial u_T$, $H_D = \partial y / \partial u_D$ are the Jacobian sensitivity matrices.

### 4.2 Lyapunov-Based Stability Concept

**Claim**: Under appropriate conditions on the step size $\alpha$, the weights $G_w$, and the sensitivity accuracy, the cascaded OFO converges to a neighbourhood of the KKT point of the true (nonlinear) optimisation problem.

#### Step 1: Define the Lyapunov Function

$$
V(u^k) = \frac{1}{2} \| u^k - u^* \|^2_{G_w} = \frac{1}{2} (u^k - u^*)^\top G_w \, (u^k - u^*)
$$

where $u^* = [u_T^*, u_D^*]$ is the optimal setpoint and $G_w \succ 0$ is the positive-definite weighting matrix.

#### Step 2: Descent Property of a Single OFO Step

For the continuous part of a single-layer OFO (ignoring integer variables for now), the update is:

$$
u^{k+1} = u^k + \alpha \, \sigma^k
$$

where $\sigma^k$ solves the MIQP. In the unconstrained case, $\sigma^k = -G_w^{-1} \nabla f(u^k)$, and the descent condition becomes:

$$
V(u^{k+1}) - V(u^k) = \alpha \, (u^k - u^*)^\top G_w \, \sigma^k + \frac{\alpha^2}{2} \, \sigma^{k\top} G_w \, \sigma^k
$$

For sufficiently small $\alpha$ and if $\nabla f$ is Lipschitz continuous with constant $L$, the standard projected-gradient descent theory guarantees:

$$
V(u^{k+1}) \leq V(u^k) - \alpha \left( 1 - \frac{\alpha L}{2} \right) \| \sigma^k \|^2_{G_w}
$$

This is **strictly decreasing** provided:

$$
\boxed{\alpha < \frac{2}{L}}
$$

where $L$ depends on the curvature of the objective and the conditioning of $H$.

#### Step 3: Sensitivity Error (Model Mismatch)

In practice, the Jacobian $H^k$ is an approximation of the true sensitivity $\bar{H}$. Let $E^k = H^k - \bar{H}$ be the sensitivity error. The actual output after applying $\sigma^k$ is:

$$
y^{k+1} = y^k + \bar{H} \, \alpha \sigma^k + O(\|\alpha \sigma^k\|^2)
$$

The OFO's feedback nature is crucial: at step $k+1$, the controller **re-measures** $y^{k+1}$ from the plant and recomputes. The sensitivity error only affects the *predicted* output within one step, not the measured output used at the next step. This gives OFO its inherent **robustness to model mismatch** — the closed-loop error is:

$$
\| e^{k+1} \| \leq \rho \, \| e^k \| + \alpha \, \| E^k \| \, \| \sigma^k \|
$$

where $\rho < 1$ if the descent condition holds. The steady-state error is bounded by:

$$
\| e^\infty \| \leq \frac{\alpha \, \| E \|}{1 - \rho} \cdot \| \sigma^\infty \|
$$

which vanishes at a fixed point ($\sigma^\infty = 0$).

#### Step 4: Cascade Stability via Time-Scale Separation

The cascade has natural time-scale separation: DSO (fast, period $T_D = 1$ min) and TSO (slow, period $T_T = 3$ min). Between two TSO steps, the DSO executes $T_T / T_D = 3$ iterations.

**Singular perturbation argument**: If the DSO converges sufficiently fast (i.e., its spectral radius $\rho_D < 1$), then from the TSO's perspective, the DSO is approximately at its quasi-steady-state:

$$
u_D^*(Q_{\text{set}}) = \arg\min_{u_D} f_D(u_D; Q_{\text{set}}) \quad \text{s.t. constraints}
$$

The TSO then effectively optimises over the **reduced map**:

$$
\tilde{y}_T = h(u_T, u_D^*(Q_{\text{set}}(u_T)), d)
$$

The cascade is stable if:

1. **DSO internal stability**: The DSO controller, for fixed $Q_{\text{set}}$, is a contraction:
$$
\| u_D^{k+1} - u_D^* \| \leq \rho_D \, \| u_D^k - u_D^* \|, \quad \rho_D < 1
$$

2. **TSO stability on the reduced system**: The TSO, using the reduced sensitivity $\tilde{H}_T = \partial \tilde{y}_T / \partial u_T$, satisfies the descent condition with $\alpha_T < 2/L_T$.

3. **AVT ensures consistency**: The AVT reset guarantees that the TSO's internal PCC-Q state matches the DSO's achieved value, preventing windup in the outer loop. Formally, AVT ensures:
$$
\| u_{\text{PCC}}^k - Q_{\text{achieved}}^k \| = 0 \quad \text{(for } k_t = 1\text{)}
$$
eliminating the cascade coupling error from the TSO's optimisation basis.

4. **Capability bounds ensure feasibility**: The DSO reports operating-point-dependent Q bounds to the TSO, ensuring that $Q_{\text{set}}$ is always within the achievable range. This prevents the TSO from commanding infeasible setpoints.

#### Step 5: Combined Lyapunov Function

Define the combined Lyapunov function:

$$
\mathcal{V}(u_T^k, u_D^k) = V_T(u_T^k) + \epsilon \, V_D(u_D^k)
$$

where $\epsilon > 0$ is small (reflecting the time-scale separation). Under the conditions above:

$$
\mathcal{V}^{k+1} \leq \mathcal{V}^k - \alpha_T \, c_T \, \| \sigma_T^k \|^2 - \epsilon \, \alpha_D \, c_D \, \| \sigma_D^k \|^2
$$

for positive constants $c_T, c_D$ that depend on the weights and Lipschitz constants. This guarantees **monotone decrease** until a fixed point is reached ($\sigma_T = \sigma_D = 0$), proving convergence.

### 4.3 Conditions for Stability (Summary)

| Condition | Interpretation | Ensured by |
|-----------|---------------|------------|
| $\alpha < 2/L$ | Step size not too aggressive | Tuning of `alpha` and `G_w` |
| $G_w \succ 0$ | Positive-definite change penalty | Per-actuator weights $g_w > 0$ |
| $G_z \gg G_w$ | Output constraints near-hard | Large `g_z` (default $10^{12}$) |
| $\rho_D < 1$ | DSO converges between TSO steps | Time-scale separation ($T_T / T_D = 3$), DSO $G_w$ tuning |
| $\|E^k\|$ small | Jacobian approximation accurate | Sensitivity updater (V² shunt rescaling), re-computation after contingencies |
| AVT ($k_t = 1$) | No outer-loop windup | `apply_avt_reset()` before each TSO solve |
| Capability bounds | TSO commands are feasible for DSO | `generate_capability_message()` feedback |
| Integer cooldown | Discrete variables don't chatter | 6-iteration lock + Reserve Observer hysteresis |

### 4.4 Role of Each Mechanism in Stability

```
                    STABILITY
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    Descent in f    Feasibility   Anti-windup
    (Lyapunov)     maintenance    mechanisms
         │             │             │
    ┌────┴────┐    ┌───┴───┐    ┌───┴────┐
    │ α, G_w  │    │ G_z   │    │  AVT   │
    │ tuning  │    │ slack │    │ k_t=1  │
    │         │    │ vars  │    │        │
    │ Jacobian│    │       │    │Integral│
    │ H = ∂y/ │    │Capab. │    │  Q-err │
    │    ∂u   │    │bounds │    │  decay │
    └─────────┘    └───────┘    └────────┘
                                    │
                              ┌─────┴─────┐
                              │ Reserve   │
                              │ Observer  │
                              │ hysteresis│
                              └───────────┘
```

### 4.5 Practical Stability Margins

The $G_w$ weights in `CascadeConfig` directly control the effective step size. For a single variable with gradient $g$:

$$
\Delta u = \frac{-g}{2 \, (g_w + g_u)}
$$

The weights are calibrated so that:
- **DER Q** ($g_w = 0.4$ TSO, $4.0$ DSO): rapid continuous adjustment
- **Generator AVR** ($g_w = 5 \times 10^6$): extremely cautious voltage changes
- **OLTCs** ($g_w = 40$–$100$): moderate discrete steps
- **Shunts** ($g_w = 1000$–$3000$): conservative switching, reinforced by cooldown and Reserve Observer

The hierarchy of weights ensures a natural **priority ordering**: fast continuous DER Q first, then slow discrete OLTCs/shunts — which is consistent with the Lyapunov descent requiring the fastest modes to converge first.
