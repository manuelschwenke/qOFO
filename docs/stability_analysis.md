# Stability Analysis for Cascaded OFO Controllers

## 1. Overview

The stability analysis in `analysis/stability_analysis.py` computes theoretical
stability bounds for the cascaded TSO-DSO Online Feedback Optimisation (OFO)
controller.  It is based on the contraction mapping framework from
[Hauswirth et al. (2021)](https://arxiv.org/abs/2103.11329) and uses the
actual Jacobian sensitivity matrices $H$ from the TUDa network.

**Key output:**
- Maximum stable step size $\alpha_{\max}$
- Per-actuator stability margins
- DSO cascade contraction rate $\rho_D$
- Eigenvalue diagnostics showing *which actuator types drive instability*

---

## 2. Mathematical Foundation

### 2.1 OFO Update Rule

The MIQP at each OFO step solves (in the unconstrained case):

$$
\sigma^* = \arg\min_w \; w^\top G_w \, w + \nabla f^\top w
$$

giving $\sigma^* = -\tfrac{1}{2} G_w^{-1} \nabla f$.  The actuator update is:

$$
u^{k+1} = u^k + \alpha \, \sigma^* = u^k - \frac{\alpha}{2} \, G_w^{-1} \nabla f(u^k)
$$

This is **preconditioned gradient descent** with step size $\alpha/2$ and
preconditioner $G_w^{-1}$.

### 2.2 Tracking Objective and Gradient

The tracking objective for each layer is a weighted sum of squared deviations:

| Layer | Objective | Gradient |
|-------|-----------|----------|
| **TSO** | $f_T = g_v \|V - V_{\text{set}}\|^2$ | $\nabla f_T = 2\, H_V^\top \, g_v \, (V - V_{\text{set}})$ |
| **DSO** | $f_D = g_q \|Q - Q_{\text{set}}\|^2 + g_v^{\text{DSO}} \|V - V_{\text{set}}\|^2$ | $\nabla f_D = 2\, H_Q^\top g_q (Q - Q_{\text{set}}) + 2\, H_V^\top g_v^{\text{DSO}} (V - V_{\text{set}})$ |

The Hessian is $\nabla^2 f = 2 \, H^\top Q_{\text{obj}} \, H$, where $Q_{\text{obj}}$ is the
**per-output diagonal weight matrix** — NOT a scalar.

### 2.3 The Per-Output Weight Matrix $Q_{\text{obj}}$

The sensitivity matrix $H$ has rows for different output types, ordered as
$[Q_{\text{trafo}} \;|\; V_{\text{bus}} \;|\; I_{\text{line}}]$.

**Critical insight:**  Only outputs that appear in the tracking objective
contribute to the gradient's curvature.  Current limits are enforced via
constraints (slack variables with $G_z$), NOT via the tracking objective.
Therefore **current rows get weight zero** in $Q_{\text{obj}}$.

| Layer | Q rows | V rows | I rows |
|-------|--------|--------|--------|
| **TSO** | 0 (no trafo Q) | $g_v$ | 0 |
| **DSO** | $g_q$ (PCC interface) | $g_v^{\text{DSO}}$ | 0 |

$$
Q_{\text{obj}}^{\text{DSO}} = \text{diag}(\underbrace{g_q, \ldots, g_q}_{n_Q}, \;\underbrace{g_v^{\text{DSO}}, \ldots, g_v^{\text{DSO}}}_{n_V}, \;\underbrace{0, \ldots, 0}_{n_I})
$$

### 2.4 Why the Old Formula Was Wrong

The old analysis used:

$$
L_{\text{eff}} = 2 \, g_{\text{obj}} \cdot \sigma_{\max}(H)^2, \qquad
\alpha_{\max} = \frac{2}{L_{\text{eff}}}
$$

This has **two errors:**

1. **$\sigma_{\max}(H)$ mixes incompatible output types.**
   The SVD treats Q (Mvar), V (p.u.), and I (kA) rows with equal weight.
   But current rows have **zero weight** in the objective — they're
   constraint-only.  The DSO's $\sigma_{\max} = 57$ was dominated by
   I/OLTC entries that don't contribute to the gradient at all.

2. **$G_w$ preconditioning was ignored.**
   The update is preconditioned by $G_w^{-1}$, so the effective step size
   depends on the ratio of curvature to damping, not curvature alone.
   Since $G_w$ has very different values per actuator type (DER: 4–10,
   OLTC: 100–120, Shunt: 3000–5000), this dramatically changes $\alpha_{\max}$.

---

## 3. Correct Stability Condition

### 3.1 Curvature Matrix

Define the **curvature matrix** ($n_u \times n_u$, symmetric PSD):

$$
C = H^\top Q_{\text{obj}} \, H
$$

Efficiently computed as $(Q^{1/2} H)^\top (Q^{1/2} H)$.

### 3.2 Preconditioned Curvature

The **preconditioned curvature matrix** accounts for $G_w$:

$$
M = G_w^{-1/2} \, C \, G_w^{-1/2}
$$

This is symmetric, so it has real eigenvalues $\lambda_1 \leq \cdots \leq \lambda_{n_u}$.

### 3.3 Contraction Condition (Hauswirth)

The fixed-point iteration $u^{k+1} = u^k - (\alpha/2) G_w^{-1} \nabla f(u^k)$
is a contraction iff all eigenvalues of $\alpha \, G_w^{-1} C$ lie in $(0, 2)$.
Through the similarity transform, this is equivalent to:

$$
\boxed{0 < \alpha \, \lambda_i(M) < 2 \quad \text{for all } i}
$$

The **maximum stable step size** is:

$$
\alpha_{\max} = \frac{2}{\lambda_{\max}(M)}
$$

### 3.4 Per-Actuator Necessary Condition (Gershgorin)

From the Gershgorin disk theorem applied to $\alpha G_w^{-1} C$,
the diagonal entry for actuator $i$ must satisfy:

$$
\frac{\alpha \, C_{ii}}{g_{w,i}} < 2
\quad\Longleftrightarrow\quad
g_{w,i} > \frac{\alpha \, C_{ii}}{2}
$$

where $C_{ii} = (H^\top Q_{\text{obj}} H)_{ii}$ is the **per-actuator self-curvature**.

- **Threshold** $= \alpha \, C_{ii} / 2$: minimum $g_w$ for that actuator
- **Margin** $= g_{w,i} - \text{threshold}$: positive = safe, negative = unstable

> **Note:**  The per-actuator check is *necessary but not sufficient*.
> Cross-coupling between actuators (off-diagonal entries of $C$) can make
> the eigenvalue condition $\alpha \lambda_{\max}(M) < 2$ tighter than any
> individual diagonal check.

### 3.5 Spectral Contraction Rate

The contraction rate per iteration is:

$$
\rho = \max_i \; |1 - \alpha \, \lambda_i(M)|
$$

- $\rho < 1$: contracting (stable)
- $\rho = 0$: optimal step size ($\alpha = 1/\lambda_i$)
- $\rho \geq 1$: divergent (unstable)

---

## 4. Cascade Stability

### 4.1 Time-Scale Separation

The cascade has:
- TSO period $T_T$ (e.g. 3 min = 180 s)
- DSO period $T_D$ (e.g. 20 s)
- Ratio $T_T / T_D = 9$ DSO iterations per TSO step

### 4.2 DSO Convergence Between TSO Steps

When the TSO sends new $Q_{\text{set}}$, the DSO must converge to
quasi-steady-state within $T_T / T_D$ iterations.  After $N$ iterations,
the remaining error is $\rho_D^N$ times the initial error.

| $\rho_D$ | $\rho_D^9$ | Error remaining | Quality |
|-----------|-----------|-----------------|---------|
| 0.0 | 0.0 | 0% | Perfect (1-step convergence) |
| 0.5 | 0.002 | 0.2% | Excellent |
| 0.8 | 0.13 | 13% | Good |
| 0.9 | 0.39 | 39% | Acceptable |
| 0.95 | 0.63 | 63% | Tight |
| 1.0 | 1.0 | 100% | Boundary |
| >1.0 | diverges | — | Unstable |

**Cascade margin** $= 1 - \rho_D^{T_T/T_D}$:
- Must be **> 0** (necessary)
- Ideally **> 0.5** (DSO settles within half the TSO period)

### 4.3 Degenerate Modes and ρ_D Filtering

When $H$ is rank-deficient (e.g. co-located DERs on the same bus),
some eigenvalues of $M$ are near zero.  These give $|1 - \alpha \cdot 0| = 1$,
which would make $\rho_D = 1$ and the cascade margin 0.

**These modes are physically benign:**  A near-zero eigenvalue means the
gradient is zero in that direction — the iterate doesn't move, but it also
doesn't need to, because the error there is negligible.  Co-located DERs
changing Q in equal-and-opposite amounts produce no net effect on the outputs.

The analysis **filters out degenerate modes** with $\lambda < 0.01 \cdot \lambda_{\max}$
and computes $\rho_D$ only from the active modes.  Both the active and raw
(unfiltered) contraction rates are reported for transparency.

### 4.4 Three Conditions for Cascade Stability

| # | Condition | Formula | Meaning |
|---|-----------|---------|---------|
| 1 | TSO contraction | $\alpha \, \lambda_{\max}(M_T) < 2$ | TSO converges on its own |
| 2 | DSO contraction | $\alpha \, \lambda_{\max}(M_D) < 2$ | DSO converges on its own |
| 3 | DSO fast enough | $\rho_D^{T_T/T_D} \ll 1$ | DSO reaches quasi-steady-state between TSO steps |
| 4 | PI stability | $\rho_{\text{aug}} < 1$ | Augmented (u, q_int) system stable |

---

## 5. Augmented PI Analysis (Integral Q-Tracking)

### 5.1 Motivation

The DSO has an optional **integral Q-tracking** term with parameters:
- `g_qi` — integral weight (0 = disabled)
- `lambda_qi` — leaky integrator decay (0 < λ < 1)
- `q_integral_max_mvar` — clamp on the integral state

The integral state accumulates Q-interface tracking error:

$$q_{\text{int}}^{k+1} = \lambda \, q_{\text{int}}^k + (Q^k - Q_{\text{set}})$$

and adds a persistent pressure term to the gradient:

$$\nabla f_{\text{integral}} = 2 \, g_{qi} \, H_Q^\top \, q_{\text{int}}$$

This turns the memoryless gradient descent into **PI control**.

### 5.2 Augmented Contraction Matrix

The joint state is $[e_u, \, e_{qi}]$ where $e_u = u - u^*$ and
$e_{qi} = q_{\text{int}} - q_{\text{int}}^*$. The linearised iteration is:

$$
\begin{bmatrix} e_u^{k+1} \\ e_{qi}^{k+1} \end{bmatrix}
= \underbrace{\begin{bmatrix}
  I - \alpha \, G_w^{-1} C_{\text{prop}} & -\alpha \, g_{qi} \, G_w^{-1} H_Q^\top \\
  H_Q & \lambda \, I
\end{bmatrix}}_{A_{\text{aug}}}
\begin{bmatrix} e_u^k \\ e_{qi}^k \end{bmatrix}
$$

where $C_{\text{prop}} = H^\top Q_{\text{obj}} H$ is the proportional curvature
(same as the memoryless $C$).

### 5.3 Stability Condition

The augmented system is stable iff:

$$
\rho_{\text{aug}} = \max_i |\lambda_i(A_{\text{aug}})| < 1
$$

Note that $A_{\text{aug}}$ is **not symmetric** (the off-diagonal blocks have
different shapes), so eigenvalues may be **complex**.  We compute all
eigenvalues of the $(n_u + n_q) \times (n_u + n_q)$ matrix and take the
maximum absolute value.

### 5.4 Effect of Integral Parameters

- **g_qi = 0**: reduces to memoryless case (no integral coupling)
- **Small g_qi** (e.g. 0.1): mild integral pressure, barely affects ρ_aug
- **Large g_qi**: strong coupling between u and q_int, can destabilise
- **λ_qi close to 1** (e.g. 0.95): slow decay, integral accumulates more,
  steady-state gain = g_qi / (1 − λ) can be large
- **λ_qi close to 0**: fast decay, integral has little memory

The steady-state gain of the integrator is $g_{qi} / (1 - \lambda)$.
With g_qi = 0.2, λ = 0.95: gain = 0.2 / 0.05 = 4.0, effectively
quintupling the Q-interface curvature at steady state.

### 5.5 Caveat

The augmented analysis is **linear**.  The `q_integral_max_mvar` clamp is
a nonlinear saturation that the eigenvalue analysis cannot capture.  In
practice, the clamp prevents unbounded integral growth and can stabilise
a system that the linear analysis predicts as marginally unstable.

---

## 6. Eigenvalue Mode Diagnostics

### 5.1 What They Show

For each of the top eigenvalues of $M$, the report shows:
- $\lambda_i(M)$ — the eigenvalue
- $\alpha \lambda_i$ — should be $< 2$
- $|1 - \alpha \lambda_i|$ — contraction factor for that mode
- **Actuator-type participation** — fraction of the eigenvector weight per type

The participation comes from the squared eigenvector components $|v_i|^2$,
grouped by actuator type (DER, OLTC, Shunt).  Since $v$ is normalised,
the participations sum to 100%.

### 5.2 How to Read Them

**Example output:**
```
  mode        λ(M)      α·λ    |1−α·λ|   actuator-type participation
     1       1.055     1.055     0.0547   Q_DER_DN: 44%  Shunt_DN: 42%  OLTC_DN: 13%
     2       0.497     0.497     0.5030   OLTC_DN: 91%  Q_DER_DN: 5%  Shunt_DN: 4%
     3      0.3635    0.3635     0.6365   OLTC_DN: 83%  Shunt_DN: 14%  Q_DER_DN: 3%
          ...
  slow      0.0236    0.0236     0.9764   Q_DER_DN: 88%  OLTC_DN: 12% ◄ slow
```

- **Mode 1**: Fastest mode, dominated by DER Q and shunt coupling.
  $|1 - \alpha\lambda| = 0.05$ — converges in ~1-2 iterations.

- **Slow mode** (◄ slow): This is the **bottleneck** for ρ_D.
  $\lambda = 0.024$ is small → the gradient in this direction is weak →
  convergence is slow ($|1 - \alpha\lambda| = 0.976$).
  88% DER Q-driven, likely a near-degenerate direction from similarly-located DERs.
  To improve: this specific mode may be benign if the excitation in that
  direction is small in practice.

### 5.3 Tuning Strategy

1. **Identify the critical mode** (largest $|1 - \alpha \lambda|$, marked with ◄)
2. **Read the participation**: which actuator type dominates?
3. **Increase $g_w$ for that type**: this divides $\lambda(M)$ for that direction
4. **Or reduce $\alpha$**: shifts all modes away from the boundary

The participation tells you *where the cross-coupling is strongest*.
Even if the per-actuator diagonal margins are all positive, cross-coupling
through the Q/V outputs can push an eigenvalue close to 2.

---

## 7. Ill-Conditioned H Matrix

### 6.1 When It Happens

The raw $H$ matrix is often ill-conditioned ($\kappa > 10^6$) because:
- **Co-located DERs**: Multiple DER sgens on the same bus produce
  near-identical columns in $H$ (same sensitivity to all outputs).
- **Scale mismatch**: Some outputs (Q in Mvar) have much larger sensitivities
  than others (V in p.u.).

### 6.2 Is It a Problem?

**For stability: No.**  The $G_w$ regularisation ensures the preconditioned
matrix $M = G_w^{-1/2} C G_w^{-1/2}$ is well-conditioned.  Near-duplicate
columns create near-zero eigenvalues in $C$ (and $M$), which are benign —
they represent directions in which the gradient is zero and the iterate
doesn't move.

**For sensitivity accuracy:**  If the Jacobian $H$ is computed via finite
differences, the small singular values are numerically unreliable.  The
analytical Jacobian (`JacobianSensitivities`) is more robust.  In any case,
the *large* singular values (which determine stability) are accurate.

---

## 8. Units in the Sensitivity Matrix

The $H$ matrix has mixed physical units:

| | Q_DER (Mvar) | s_OLTC (tap) | Shunt (state) |
|---|---|---|---|
| **Q_PCC (Mvar)** | ∂Q/∂Q ≈ 1 (unitless) | ∂Q/∂s (Mvar/tap) | ∂Q/∂state (Mvar/step) |
| **V_bus (p.u.)** | ∂V/∂Q (p.u./Mvar) | ∂V/∂s (p.u./tap) | ∂V/∂state (p.u./step) |
| **I_line (kA)** | ∂I/∂Q (kA/Mvar) | ∂I/∂s (kA/tap) | ∂I/∂state (kA/step) |

The raw $\sigma_{\max}(H)$ is physically meaningless because it mixes
these units.  The **curvature matrix** $C = H^\top Q_{\text{obj}} H$
properly weights each row by its objective importance, making $L_{\text{eff}}$
and $\alpha_{\max}$ physically meaningful.

---

## 9. Running the Analysis

```bash
# With default TUDa config (from _tuda_config()):
python -m analysis.run_stability_tuda

# Override parameters:
python -m analysis.run_stability_tuda --alpha 0.8 --gw_dso 15 --dso_g_v 50000
```

**Programmatically:**
```python
from analysis import run_stability_analysis
from core.cascade_config import CascadeConfig

config = CascadeConfig(alpha=1.0, g_v=250000, g_q=1.0, dso_g_v=100000)
result = run_stability_analysis(config, verbose=True)

# Access results
print(result.dso.alpha_max)
print(result.dso.eigenvalue_diagnostics[0])  # top mode
```

---

## 10. References

- **Hauswirth, A., Bolognani, S., Hug, G., & Dörfler, F.** (2021).
  *Optimization Algorithms as Robust Feedback Controllers.*
  Annual Reviews in Control. [arXiv:2103.11329](https://arxiv.org/abs/2103.11329)

- **cascade_structure.md** — Full cascade architecture and Lyapunov argument.

- **stability_analysis.py** — Implementation of the analysis described here.
