# Switched-shunt dispatch in the TS MIQP: method

Status: 21 October 2026. Describes the implemented `shunt_dispatch='hybrid'`
path. Written for external review; §9 separates what is established from what
is hypothesis.

## 1. Scope and notation

One TS (Layer-1) control zone, solved every TS period `T_TS` (180 s in the runs
below). Subordinate STS (Layer-2) controllers track delegated interface
reactive-power setpoints on a faster period (20 s). No controller reads the
plant; each acts on measurements and cached sensitivity models.

| symbol | meaning |
|---|---|
| `u ∈ R^{n_u}` | TS control vector `[Q_DER \| Q_PCC,set \| V_AVR \| s_OLTC]` |
| `y ∈ R^{n_y}` | TS outputs, ordered `[V \| Q_PCC \| I \| Q_gen]` |
| `H̃ = ∂y/∂u` | cached sensitivity matrix |
| `∇_y f` | gradient of the shared TS objective w.r.t. outputs |
| `w` | control *increment* decided by one MIQP solve |
| `α` | OFO step size, applied to continuous increments only |
| `m` | switched-shunt banks in the zone (MSC/MSR, one each per subordinate system) |
| `ℓ_j ∈ {0,…,L_j}` | committed level of bank `j`; `Δq_j` its step size [Mvar] |
| `h_j ∈ R^{n_y}` | output change per **one level** of bank `j` (cached, voltage-rescaled) |

The shared objective is quadratic in the outputs; only its gradient enters.
With the weights used here,

```
∇_y f = [ 2·g_v·(V − V_set) ; 2·g_q,pcc·(Q_PCC − Q_PCC,set) ; 2·g_loss·c·|I| ; 2·g_res,sg·(Q_gen − Q_mid)/Q_half² ]
```

so each block is active only where its weight is non-zero.

## 2. The base TS MIQP

Each TS instant solves

```
min_{w, z}   wᵀ G_w w  +  ∇f ᵀ w  +  zᵀ G_z z
s.t.         u_lo ≤ u + w ≤ u_hi
             y_lo − z ≤ y + α·H_c w_c + H_i w_i ≤ y_hi + z,     z ≥ 0
             w_i integral for i ∈ I            (OLTC taps, and the banks of §3)
```

with `H_c`, `H_i` the continuous and integer column blocks of `H̃`. The output
constraint carries `α` on the continuous part only, because a discrete actuator
cannot move a fraction of a step. **In this project `α = 1` throughout** — the
step size is absorbed into the per-actuator `G_w` instead — so the constraint
reduces to `y + H̃w` and every number reported here is unaffected. The scaling
is kept explicit because the implementation retains it and the support-transfer
tests exercise `α = 0.2`.

with `∇f = H̃ᵀ ∇_y f` plus the direct control-space terms. There is **no factor
½**: `G_w` is the OFO step-size regularisation, so the realised step of an
unconstrained continuous actuator is `−∇f_i / (2 G_w,ii)`. The applied control
change is

```
Δu = [ α·w_continuous ; w_integer ]
```

— integers are applied in full, continuous increments scaled by `α`. `G_w` is a
trust-region device, **not** part of `f`; this matters in §6.

## 3. Bank augmentation

The `m` bank levels become genuine integer variables of the same problem:

```
n_integer ← n_integer + m
u_cur     ← [u ; ℓ]
u_lo,j    ← ℓ_j + min(0, D_j(t)),     u_hi,j ← ℓ_j + max(0, D_j(t))
∇f        ← [∇f ; (∇_y f ᵀ h_1, …, ∇_y f ᵀ h_m)]
H̃         ← [H̃ | h_1 … h_m]
G_w       ← blkdiag(G_w, g_w^sh · I_m)
```

`D_j(t) ⊆ {−1,+1}` are the directions admissible for bank `j` at time `t` under
its **shared** guards: level limits, dwell time since its last switch, and
switching budget. A bank whose guards block both directions is pinned.

Two restrictions are added to the solve:

```
‖w_s‖_1 ≤ 1                           (at most one bank step per zone and instant)
y_lo,V ≤ y_V + (H̃w)_V ≤ y_hi,V        as HARD rows (no slack), with
y_lo,V ← max(y_lo,V, max_j y_h,min,j),  y_hi,V ← min(y_hi,V, min_j y_h,max,j)
```

If the augmented problem is infeasible, the **original** problem is solved with
the banks held, so the ordinary recovery action for the continuous actuators is
never blocked.

## 4. Path A — the priced (regular) decision

Path A is an incentive mechanism: the bank is an ordinary variable of the
objective and the optimiser decides. Holding everything else fixed, a one-level
step on bank `j` with `Δs_j ∈ {−1,0,+1}` changes the objective by

```
ΔJ = g_w^sh·Δs_j²  +  g_j·Δs_j ,        g_j := ∇_y f ᵀ h_j
```

which is minimised at `Δs_j = −sgn(g_j)`, giving `ΔJ = g_w^sh − |g_j|`. Hence

> **an isolated one-level step lowers the objective iff `|g_j| > g_w^sh`.**

So `g_w^sh` is an **admission threshold in objective units per level**, not an
accumulation rate.

This is a statement about the *isolated* move, and it is neither necessary nor
sufficient in the constrained joint problem. It omits two effects: a bank step
that relieves a binding output constraint can pay for itself through a reduced
slack penalty `zᵀG_z z`, and it can reduce the continuous movement cost
`w_cᵀ G_w,c w_c` that the other actuators would otherwise incur — so a step can
be selected below the threshold. Conversely, constraints can block a step above
it. Empirically the rule is a good predictor: reconstructing the admitted set
this way, with levels tracked from the commit list and the dwell and level
limits applied, reproduced all ten commits of a recorded run with none
unexplained, and the two admitted-but-uncommitted instants were explained by
dwell. That is evidence, not proof.

## 5. Path B — supervisory proposals

Path B exists for demands the instantaneous optimisation does not express:
small but persistent demand, and reserve margins. It is **not** added to the
objective. Three supervisory costs are defined, each a function of measured or
communicated quantities only, with `D_c(x) := sgn(x)·max(|x|−c, 0)` applied
componentwise:

**(a) Subordinate tracking.** With interface error `e = q_req − q_meas`,

```
J_trk = w_trk · ‖D_d(e)‖² / q_s² ,      ∂J_trk/∂s_j = −2·w_trk·D_d(e)ᵀ h_q,j / q_s²
```

**(b) Subordinate reserve.** From the reported capability band `[q_min, q_max]`
with centre `m` and half-width `b`, and `ρ = (q − m)/b`,

```
J_sub = w_sub · ‖D_ρ0(ρ)‖² ,            ∂J_sub/∂s_j = −2·w_sub·D_ρ0(ρ)ᵀ (h_q,j / b)
```

**(c) TS fleet reserve.** From the zone's own synchronous machines, with
`Q_mid = ½(Q_max+Q_min)`, `Q_half = ½(Q_max−Q_min)` from the same capability
source as `∇_y f`, and `r = (Q_gen − Q_mid)/Q_half`,

```
J_fleet = w_f · ‖D_r0(r)‖² ,            ∂J_fleet/∂s_j = 2·w_f·D_r0(r)ᵀ (h_gen,j / Q_half)
```

All three are zero inside their deadband, so none exerts centring pressure in
normal operation. (a) additionally requires fresh feedback and a reference old
enough for the subordinate loop to have responded; (c) requires neither,
because it uses the TS's own measurements at the same instant.

**Memory.** Each (bank, reason) pair holds a signed accumulator in nameplate
Mvar, integrated once per TS observation:

```
a ← clip( a − (∂J_k/∂s_j) / (2·g_w^mem),  −Δq_j,  +Δq_j )
```

cleared on sign reversal, on demand disappearing, on stale feedback, on an
observation gap above `T_gap`, and after any physical commit in the zone.

**Readiness.** A proposal exists iff the sign has held since `t_since` with

```
t − t_since ≥ T_p      and      |a| ≥ Δq_j/2 + δ
```

and its direction `d` is the memory's sign — it is not searched.

**Candidate problem.** For each ready proposal, the *same* augmented problem is
re-solved with the step imposed:

```
s_k = ℓ_k   for all k ≠ j        (all other banks pinned)
s_j = ℓ_j + d                    (the proposal, forced)
```

so every continuous actuator and OLTC is re-optimised **around** the step,
under the same objective and the same hard corridor. For reasons (a) and (b)
the delegated requests are additionally pinned to `u*_pcc − d·h_q,j`, see §6.

**Acceptance.** Let `y_reg` and `y_cand` be the predicted outputs of the
regular and candidate solutions, both via `Δu = [α w_c ; w_i]`.

For the fleet reason the benefit is re-evaluated on the **complete candidate**,
not on the bank-only prediction the request used:

```
ΔJ_fleet = J_fleet(y_reg,gen) − J_fleet(y_cand,gen)   must be > 0
```

Without this, an AVR or delegated-Q adjustment made while re-optimising around
the forced step could cancel the relief the request predicted, and the bank
could be credited with a recovery the ordinary controller achieves by itself.
A refusal here is recorded as `no_fleet_relief`, with both the bank-only and
the full-candidate figures. (Added 21 October after external review; the
subordinate reasons still use their own request-level benefit, where the
delegated transfer pins the quantity the cost is defined on.)

The candidate is accepted iff

```
∇_y f ᵀ (y_cand − y_reg)  ≤  χ · ( J_k(before) − J_k(after) )          (gate)
max_i [ excess_i(y_cand) − excess_i(y_reg) ] ≤ 0                        (no new violation)
```

with `excess(y) = max(y_lo − y, 0) + max(y − y_hi, 0)` and `χ ≥ 0` the exchange
weight (default 0: support may act only when the shared objective does not
worsen). The left-hand side is the **first-order change of the shared
objective**, deliberately *not* a difference of MIQP objective values: `G_w` is
step-size regularisation and a candidate is forced to move its delegated
request by a whole bank step, so an objective-value comparison is dominated by
that price. (Measured: `5.5·10¹⁰` against support reductions of order 0.5.)

An alternative gate bounds the predicted worst-bus voltage tracking-error
increase by a per-unit allowance instead; it is retained for reproducibility of
earlier runs and is **not** recommended, because its threshold in Mvar moves
with the band and it refused candidates that *improve* the shared objective.

**Ranking.** Accepted candidates are ordered by `(priority_k, −relative
benefit, candidate objective value, bank index)` with priority
tracking → subordinate reserve → fleet reserve, and the first is dispatched.

**Cost.** One regular solve plus one solve per ready proposal. Measured:
42–47 solves per hour against 21 regular ones over three zones, i.e. ≈0.7 extra
solves per zone-instant; TS solve time p95 rose from 0.002 s to 0.048 s against
a 180 s period.

## 6. Reference conventions

A committed step changes the interface flow the subordinate controller
measures. A persistent per-interface feedforward offset keeps it from
counteracting:

```
dispatched setpoint = u_pcc + o,     o ← o + d·h_q,j   at each commit
reported band       [q+δ_min, q+δ_max]  shifted by −o before it bounds u_pcc
```

the second line being required so the *emitted* setpoint stays inside the
reported band. Two conventions then follow:

* **Offset only** (path A and fleet reserve): `o` absorbs the step, `u_pcc` stays
  free, so the subordinate target moves with the bank and the subordinate
  controller does not fight it.
* **Paired transfer** (subordinate tracking and reserve): additionally
  `u_pcc ← u_pcc − d·h_q,j`, so `u_pcc + o` is invariant and the total
  subordinate reference is held fixed while the bank removes part of an
  unresolved error.

The transfer is correct for tracking and wrong for reserve: holding the target
fixed forces the subordinate DER to undo the step, which a recorded commit
shows them doing (aggregate DER Q −47.9 → −50.9 Mvar while the flow recovered
−16.1 → −13.8 over six minutes).

## 7. Shared guards

Both paths commit through one physical level, one dwell clock and one switching
budget per bank; at most one step per zone and instant; the hard predicted
voltage corridor applies to every bank commit. The budget uses reset windows
rather than a rolling 24-hour counter.

## 8. Parameters

| parameter | role | status |
|---|---|---|
| `g_w^sh` (`g_w_tso_shunt`) | path-A admission threshold, objective units per level | derived: p99 of the per-level gradient distribution; the path is currently **parked** above its observed maximum |
| `g_w^mem` (`shunt_int_g_w`) | memory integration gain | inherited from the previous integrator design |
| `T_p`, `δ`, `T_gap` | confirmation time, hysteresis half-width, gap tolerance | chosen, not derived |
| `w_trk`, `d`, `q_s` | tracking cost weight, deadband, scale | trial values |
| `w_sub`, `ρ0` | subordinate reserve weight and band start | **inert**: the signal never fired; it needed a margin below ≈7 Mvar and the margin never fell below 18.1 |
| `w_f` (revised) | fleet reserve weight | **derived: 100**, for readiness in three TS observations at the median gradient of the mildest scenario that fires. The path then commits in closed loop, but the relief of the complete candidate is ~0.03% of the bank-only prediction |
| `w_f`, `r0` (first attempt) | fleet reserve weight and band start | **inert at `r0 = 0.8`**: the gradient was zero at all 168 samples. The worst machine reached `\|r\| = 0.513` (zone 2) and `0.373` (zone 3), median 0.41 and 0.34, so the fleet never approached its rails. Firing at all needs `r0 < 0.513`, which converts the cost from a near-limit guard into a mild centring pressure |
| `χ` | gate exchange weight | 0 by default; never exercised in the positive direction |

## 9. Established, versus open

**Established by construction or by test** (210 focused tests pass): the
admission identity of §4 and its agreement with recorded commits; that the
supervisory costs do not reach the continuous actuators (pinned by a test
requiring the regular solve to be identical with a supervisory cost on and
off); that the gate measure is invariant to both step-size prices; that both
paths share one level, dwell clock and budget; and the reference-convention
algebra of §6.

**Open.**

1. **No single-objective claim for path B.** The dispatched solution is the
   better of two solves under an explicit exchange test, not the minimiser of
   one function. Standard OFO convergence arguments do not transfer, and no
   stability statement is offered for the switching logic.
1b. **The gate is first order, so it is blind to curvature.** At perfect
   tracking `∇_y f = 0`, so a candidate that introduces an error incurs zero
   predicted cost. Since `f` is quadratic in `y`, evaluating
   `f(y_cand) − f(y_reg)` exactly costs nothing beyond the cached prediction and
   would close this. The counter-consideration is consistency: the MIQP itself
   optimises a first-order model, so an exactly-quadratic gate would judge
   candidates by a criterion the solver is not minimising. Not yet decided.
1c. **Path A pre-empts path B.** If the regular solve moves a bank, the
   supervisor is skipped entirely, so B is an opportunistic fallback rather
   than a dependable reserve supervisor. With at most one step per zone and
   instant this is not arbitrary — A has already taken the one step available —
   but it is an arbitration *policy*, and the ranking rule above orders only
   candidates *within* B. Settling A-versus-B arbitration, and whether a
   reserve-restoring step should be protected from immediate reversal by A, is
   an open design decision.
2. **Linearity of the reported band.** `Σ_j s_ij (q_bound,j − q_j)` presumes all
   subordinate DER can be driven to their rails at fixed sensitivities; it
   over-states what is deliverable as individual units saturate.
3. **Path A has no demonstrated useful operating point** in the scenario tested:
   it improved the TS voltage term by 0.6% while degrading subordinate tracking
   by 54%, and no threshold value changes that exchange — only how often it is
   taken.
4. **No release was ever observed.** Engagement and release are not symmetric in
   the evidence, and the mechanism for release is understood only for the
   parked path A.
5. **Scenario coverage.** One hour, one disturbance pair, quasi-static AC, one
   network. Earlier runs additionally used a diagnostic parameter set in which
   interface-Q compensation is disabled and the delegated request frozen, so
   their magnitudes do not transfer.
6. **Campaign validity.** The analytical Stage-I screen omits the bank columns
   and rejects this mode outright; adopting it requires a revised screen and a
   repeated closed-loop selection with a new baseline.
