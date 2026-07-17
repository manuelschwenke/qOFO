# TSO–TSO Tie Coordination — Concept & Critical Review

**Author:** Manuel Schwenke (with Claude Code)
**Date:** 2026-06-26
**Status of feature:** implemented, `enable_tie_coordination` defaults **False**
(only valuable under inter-zone voltage divergence); validated on CIGRE.
**Code:** `controller/tie_coordinator.py`, `core/message.py`
(`TieCoordinationMessage`), `controller/tso_controller.py`
(`receive_tie_coordination` / `report_tie_boundary_voltage`, Q_tie band),
`experiments/007_TIE_COORDINATION.py`, `visualisation/plot_tie_coordination.py`.
**History:** `docs/daily_log/2026-06-25_tso_tso_v_coordination.md` (first, failed
price design) and `docs/daily_log/2026-06-25_tso_tso_two_loop_refactor.md`
(current design).

---

## 1. Purpose and scope

This is a **horizontal** coordination layer between peer TSO control zones across
their shared **tie lines**, added on top of the existing **vertical** TSO–DSO
cascade. Its goal is to **remove the avoidable, voltage-divergence-driven part of
the inter-zone reactive exchange** while *improving* (never degrading) each zone's
own nodal-voltage tracking, using only lightweight communication and **no shared
plant model**.

It is a supervisory layer over Layer 1 (the per-zone EHV MIQP controllers). Each
zone keeps its own local voltage-tracking objective and its own actuators; the
coordinator never solves an OPF and never sees the plant — it observes only two
scalar measurements per tie and emits two scalar setpoints per tie.

**Controlled outputs (TSO level):** nodal voltages within bounds *(primary,
unchanged)*; reactive flow at zone boundaries *(secondary, shaped indirectly)*.
**Actuators (unchanged):** AVR setpoints (continuous), OLTCs (discrete), MSC/MSR
shunts (discrete), TS-DER (continuous). The coordinator actuates **none of these
directly** — it only redirects a voltage *setpoint* that the zone's own MIQP then
tracks.

---

## 2. The central design choice: coordinate **voltage**, not tie-flow

The single most important decision is *what scalar the two zones negotiate over*.

| | Negotiate $Q_{\text{tie}}$ (reactive flow) | Negotiate boundary voltage $V$ *(implemented)* |
|---|---|---|
| Controllability | Weak: $\partial Q_{\text{tie}}/\partial u$ acts only through $V_i-V_j$, with an **unobservable common-mode null space** (both zones raising $V$ together moves the boundary voltages but not the flow) | Strong and direct: $\partial V/\partial u$ is large and already in each zone's cached sensitivity matrix $H$ |
| Observability | Jointly determined by both zones + angle/active transfer | Locally measured at each boundary bus |
| Model need | Requires caching $\partial Q_{\text{tie}}/\partial u$ as a coordination object | Reuses the **existing** $g_v$ voltage-tracking term — no new sensitivity |

**Rationale.** Voltage is the directly controllable, locally observable
primitive. Driving the two endpoints of a tie toward a **common anchor voltage**
parks the reactive exchange at its *irreducible, active-flow-driven minimum* —
subsidiarity achieved *physically* rather than by chasing a flow target. The
reactive exchange itself is then bounded by a separate, **local** soft cap
(`q_tie_band_mvar`), treated as a guardrail, **not** a tracked reference.

> **Important caveat (kept honest):** equal boundary voltages do **not** imply
> $Q_{\text{tie}}=0$. $Q_{\text{tie}}$ also depends on the angle difference
> $\theta_i-\theta_j$ (active transfer) and the line reactance, which this scheme
> does *not* control. The coordinator removes only the **reducible**, voltage-
> divergence part of the exchange; the structural, active-dispatch-driven part is
> left untouched. This is confirmed empirically (§6).

---

## 3. What is measured, exchanged, and agreed

Per tie line $e=(i,j)$ with fixed orientation $\text{zone}_i < \text{zone}_j$
(the sign of $\Delta V_{\text{ref}}$ and the per-side split depend on it):

### 3.1 Measured (zone → coordinator)
- $V_i, V_j$ — realised boundary-bus voltages [p.u.] at the in-zone endpoint of
  the tie in each zone. Each endpoint must be a **monitored voltage bus** of its
  zone. Read via `report_tie_boundary_voltage` (the coordinator never sees the
  plant directly).
- $Q_{\text{tie},e}$ — realised tie-line reactive flow [Mvar], recorded post-PF
  for the soft-cap guardrail and diagnostics (not used by the negotiation).

### 3.2 Exchanged (the only inter-zone communication)
- **Coordinator → zone:** a `TieCoordinationMessage` carrying, per incident tie,
  the per-side boundary-voltage setpoint $V_{\text{ref}}$ at the boundary bus.
  **No price / no dual is transmitted.** One aggregated message per zone.
- **Zone → coordinator:** the two boundary-voltage measurements above.

This is a **gather-and-broadcast** pattern: gather $\{V_i,V_j\}_e$, update one
scalar of state per tie, broadcast $\{V_{\text{ref},i},V_{\text{ref},j}\}_e$.
Communication per TSO scales with its number of *incident interfaces*, not with
system size.

### 3.3 Agreed (coordinator state)
- **One scalar per tie:** the **agreed boundary-voltage difference**
  $\Delta V_{\text{ref}} \;=\; \text{target}(V_i - V_j)$.
- Derived per-side setpoints around a **common anchor**
  $V_{\text{anchor}}=\tfrac12(V_{\text{nom},i}+V_{\text{nom},j})$ (the midpoint of
  the two zones' *scheduled* boundary voltages):

$$
V_{\text{ref},i} = V_{\text{anchor}} + \kappa\,\Delta V_{\text{ref}},\qquad
V_{\text{ref},j} = V_{\text{anchor}} - (1-\kappa)\,\Delta V_{\text{ref}},
$$

so that $V_{\text{ref},i} - V_{\text{ref},j} = \Delta V_{\text{ref}}$ for any
split $\kappa\in[0,1]$ ($\kappa=0.5$, symmetric, recommended). $\Delta
V_{\text{ref}}$ is initialised to $0$ (full subsidiarity = zero agreed exchange).

---

## 4. The algorithm: two loops

The scheme is a **two-timescale** controller. The separation is realised through
the relaxation *gain* (a fraction per step), not through a different sampling rate
— in the current wiring both loops advance once per Layer-1 (TSO) step.

### 4.1 Inner loop (fast, per zone) — tracking, no price
Each zone tracks its agreed per-side setpoint $V_{\text{ref}}$ through its
**existing primary voltage objective** $g_v$ — i.e. the coordinator merely
*redirects* the boundary bus's $V$-setpoint; it adds **no new term** to the
objective. Consequently:

- The coordination signal is a **bounded setpoint**, so it can never diverge the
  power flow.
- An **unreachable** setpoint is just a bounded tracking error — the worst case is
  benign.

### 4.2 Outer loop (slow, coordinator) — negotiate $\Delta V_{\text{ref}}$

$$
\boxed{\;
\Delta V_{\text{ref}} \;\leftarrow\;
\Pi_{[\pm\Delta_{\max}]}\!\Big\{
\underbrace{(1-\text{relax})\,\Delta V_{\text{ref}} + \text{relax}\,\Delta V_{\text{realized}}}_{\text{marginal feedback}}
\;-\;
\underbrace{\text{anchor}\cdot \mathrm{DB}_\Delta(\Delta V_{\text{ref}})}_{\text{subsidiarity pull}\to 0}
\Big\}
\;}
$$

with $\Delta V_{\text{realized}} = V_i - V_j$, deadband soft-threshold
$\mathrm{DB}_\Delta(x)=\operatorname{sign}(x)\max(0,|x|-\Delta)$, and the clip
$|\Delta V_{\text{ref}}|\le \Delta_{\max}$ (`dvref_max`).

**Two competing pressures:**
1. **Relaxation toward the realised difference** — pulls the *agreed* difference
   toward what the zones actually achieved.
2. **Subsidiarity anchor toward 0** — the decoupling pressure, deadbanded so that
   *small* differences ($|\Delta V_{\text{ref}}|\le\Delta$) are tolerated and only
   larger ones are pushed back to zero exchange.

### 4.3 The marginal — the "dual-like" signal (diagnostic only)
The relaxation term is exactly **gradient descent on the envelope-theorem
marginal**. With $\mathrm{err}_i = V_i - V_{\text{ref},i}$ and the combined
($g_v$-stripped) marginal

$$
m_e \;=\; \kappa\,\mathrm{err}_i - (1-\kappa)\,\mathrm{err}_j
\;\overset{\kappa=1/2}{=}\; \tfrac12\big(\Delta V_{\text{realized}} - \Delta V_{\text{ref}}\big),
$$

a step $\Delta V_{\text{ref}} \leftarrow \Delta V_{\text{ref}} + \text{step}\cdot
m_e$ reproduces the relaxation term. The common $g_v$ is **folded into the gain**
($\text{relax}:=\alpha\,g_v\in(0,1]$), so — unlike the failed price design — no
parameter has to track $g_v$. $m_e$ is the **consistency residual** the
negotiation rides on; its sign says whether the realised difference exceeds
($>0$) or trails ($<0$) the agreed one. It is *exposed for diagnostics, not
injected as a price.*

### 4.4 The Q_tie soft cap (orthogonal guardrail)
Separately and **locally**, each zone may tighten its $Q_{\text{tie}}$ output
bound from wide-open to $[-\text{band},+\text{band}]$ (`q_tie_band_mvar`),
enforced as a soft slack penalised by `g_z_q_tie` (which must be $>0$ to bite).
This bounds the emergent reactive exchange **without tracking** it, and is
independent of the voltage negotiation.

### 4.5 Per-round sequencing (one Layer-1 step)
```
measure V_i, V_j at every tie endpoint        (report_tie_boundary_voltage)
   → coordinator.update({tie: (V_i, V_j)})     (advance ΔV_ref, store m_e)
   → coordinator.generate_messages()           (per-side V_ref, no price)
   → each zone.receive_tie_coordination(msg)   (redirect boundary V-setpoint)
   → each zone MIQP solve / actuate / PF        (inner g_v tracks the new setpoint)
```

---

## 5. Designed behaviour

- **Controllable tie:** zones track the commanded difference ⇒ $\Delta
  V_{\text{realized}}\approx\Delta V_{\text{ref}}$ ⇒ the relaxation term vanishes
  ⇒ the **anchor drives $\Delta V_{\text{ref}}\to 0$** ⇒ both boundaries converge
  to $V_{\text{anchor}}$ ⇒ the reducible reactive exchange decouples.
- **Structurally pinned tie** (one end a stiff source, e.g. bus 39): the zone
  cannot reach the setpoint ⇒ $\mathrm{err}$ (hence $m_e$) stays large ⇒
  relaxation pulls $\Delta V_{\text{ref}}$ back toward the *structural* difference,
  balancing the anchor ⇒ the tie is **correctly left near its natural value**, no
  instability (inner loop only ever carries a bounded tracking error).

### Empirical status (CIGRE, last 30 min of a 70-min run)
- **Uniform-schedule base case** (all zones 1.03 p.u.): coordination is
  **neutral** — boundaries already near-consensus, tie flows are *structural*
  (active-dispatch driven), nothing to decouple. (Not a failure; a scenario
  artefact.)
- **Divergent-schedule scenario** (zones 1/2/3 at 1.05 / 1.03 / 1.01):
  coordination **works** — $\Sigma|Q_{\text{tie}}|$ **−10…−12 %**, controllable
  ties −6…−21 %, **V-RMS −8 %** (a *win–win*: the large divergence-driven exchange
  was *stressing* voltage), structurally-pinned L14 (bus 39↔bus 9) correctly
  resists. Stable throughout; diminishing returns past `anchor ≈ 0.5` (boundaries
  hit the zones' control limits).

---

## 6. Parameters

| Knob (config) | Symbol | Role | Default |
|---|---|---|---|
| `tie_relax` | $\text{relax}=\alpha g_v$ | marginal-feedback gain $\in(0,1]$ | 0.5 |
| `tie_anchor` | anchor | subsidiarity pull $\to 0$ ($\ge 0$) | 0.2 |
| `tie_deadband_v_pu` | $\Delta$ | deadband on the anchor [p.u.] | 0.002 |
| `tie_kappa` | $\kappa$ | per-side split of $\Delta V_{\text{ref}}$ | 0.5 |
| `tie_dvref_max` | $\Delta_{\max}$ | clip on agreed difference [p.u.] | 0.06 |
| `tie_q_band_mvar` | band | orthogonal $Q_{\text{tie}}$ soft cap [Mvar] | — |
| `enable_tie_coordination` | — | master switch | **False** |
| `zone_v_setpoints_pu` | $V_{\text{nom},i}$ | per-zone schedule override | — |

---

## 7. Critical review against the PDGP-OFO transfer proposal

The proposed transfer (a measurement-feedback **primal–dual gradient projection**
OFO, with an **interface dual on the measured $Q_{\text{tie}}$ residual** injected
as a **price into each zone's primal gradient**) is a sound *methodological
template*. The implemented design departs from it deliberately on the two points
that matter physically, while **agreeing** on the lightweight, measurement-feedback,
gather-and-broadcast architecture. Point by point:

### 7.1 Where the implementation diverges — and why it is the stronger choice

**(a) Coordinated variable: $V$ (primal) vs $Q_{\text{tie}}$ (dual).**
The proposal puts the dual on the $Q_{\text{tie}}$ residual $r_e = Q_e^m - Q_e^\star$.
Its own §7 then warns that tie-line reactive flow is only weakly controllable
(through $V_i-V_j$, taps, AVR) and recommends regulating "only $Q_{\text{tie}}$
and maybe boundary-voltage magnitudes." **The implementation takes that warning to
its logical conclusion:** it promotes boundary *voltage* to the **primal
coordination variable** and demotes $Q_{\text{tie}}$ to a local soft cap. This
sidesteps the unobservable common-mode null space of $\partial Q_{\text{tie}}/
\partial u$ entirely, and reuses sensitivities the zones already cache. *Verdict:
the divergence is justified and arguably more defensible than the literal
transfer.*

**(b) Price-in-objective vs setpoint negotiation — the decisive empirical lesson.**
The proposal's core update adds $s_{i,e}H_{i,e}^\top\lambda_e$ as a **price term in
each zone's primal gradient**. *This exact design was implemented first and it
failed:* inert at $\lambda\ll g_v$, and **PF-diverging** at $\lambda\sim g_v$.
Root cause: a secondary *linear* price with no curvature of its own, competing in
the primary objective's gradient on an output (voltage) with strong sensitivity,
overdrives that output. The fix moves coordination **out of the gradient** and
into a **bounded setpoint** tracked by the existing $g_v$ — which *structurally*
cannot diverge the PF. The proposal does gesture at this risk (its §8
descent-preserving cone, §9 augmented/damped variant), but those are softer,
*still gradient-based* safeguards; the setpoint reformulation removes the failure
mode by construction. **This is the single most important caveat for any thesis
write-up: do not transfer the price-in-objective update naively.**

**(c) It is still primal–dual *in spirit*, but the "dual" lives in primal space.**
Worth stating precisely so the thesis label is correct: the relaxation term *is*
gradient descent on the envelope marginal $m_e\propto(\Delta V_{\text{realized}}-
\Delta V_{\text{ref}})$, and the deadbanded anchor is the **proximal operator of a
subsidiarity penalty**. So the method is measurement-feedback and
marginal-driven — but the dual is realised as a **primal setpoint correction**,
not a price in the objective. I would therefore **avoid calling it "primal–dual
OFO"** without qualification; a precise label is *"two-timescale measurement-
feedback boundary-voltage consensus with a subsidiarity anchor."*

**(d) No cross-sensitivity needed.** The proposal's price requires caching
$H_{i,e}=\partial Q_{\text{tie}}/\partial u$ per zone. The negotiation here needs
**only the boundary-voltage measurement** — the $\partial V/\partial u$ it relies
on is already in each zone's $H$. One fewer model object to cache/maintain,
consistent with "controllers only know the plant through cached sensitivities."
($Q_{\text{tie}}$ sensitivities are still used by the orthogonal soft cap, but
locally.)

### 7.2 Where the two **agree**

- **Deadband to avoid fighting over harmless small exchanges** — both adopt it.
  *Note the different signal:* the proposal deadbands the **dual update on the
  $Q$ residual**; the implementation deadbands the **subsidiarity anchor on
  $\Delta V_{\text{ref}}$** (the relaxation/marginal term has no deadband — the
  agreed difference always reflects reality; only the push-to-zero is tolerant).
- **Zero-exchange default.** The anchor $\to 0$ is the $V$-space analogue of the
  proposal's $Q^\star=0$, and both come with the same honest physical caveat
  (equal voltages / zero $Q$ removes only the *reducible* exchange — §2).
- **Gather-and-broadcast, parallel, privacy-preserving, neighbour-local
  communication** — identical architecture (the proposal's Option A).
- **Not ADMM.** Both correctly decline the ADMM label (no local copies, no
  consensus constraint, no augmented penalty). The implementation is in fact
  *further* from ADMM than the proposal's augmented variant.
- **Stability needs damping / a contraction argument.** Both flag it; see §8.

### 7.3 Where the proposal is genuinely better — candidates to adopt

- **Honouring a nonzero reactive-exchange schedule $Q^\star\neq 0$.** The
  proposal's $Q$-residual dual directly supports a contracted reactive exchange.
  The implemented $\Delta V_{\text{ref}}$ negotiation cannot express a $Q$ target
  directly — it would need a nonzero anchor target $\Delta V_{\text{ref}}^\star$
  derived from the desired $Q^\star$, or a thin second outer loop. (Already on the
  backlog as "marginal-coordination extension: negotiate $\Delta V_{\text{ref}}
  \neq 0$ for cost-based reserve sharing.")
- **Formal local-objective protection.** The proposal's §8 descent-preserving cone
  gives a *provable* "coordination never degrades local voltage tracking"
  guarantee. The implementation achieves this *structurally* (coordination is one
  bounded setpoint weighed against the zone's other voltage targets by the same
  $g_v$) but without a formal descent certificate — the cone is a defensible
  alternative if a guarantee is wanted.
- **Notation for the $M$-TSO general case.** The incidence notation
  ($\delta(i)$, interface set $\mathcal E$, $\sum_{e\in\delta(i)}$) is a clean way
  to write up 3–4 TSOs in the thesis even though the per-tie mechanism differs.

### 7.4 A scaling subtlety the proposal's formulation handles more cleanly

The proposal's price contributions at a shared control are **additive**
($\sum_{e\in\delta(i)}s_{i,e}H_{i,e}^\top\lambda_e$). The implementation's
**setpoint redirect is not additive** — it overrides the $V$-setpoint *per
boundary bus*. If a single boundary bus serves **multiple ties** (meshed
interconnection), the per-side setpoints from different ties **conflict at that
bus**. For the current CIGRE case (distinct boundary buses per tie, radial
inter-zone structure) this is a non-issue, but it is a real limitation to flag
before claiming general $M$-TSO scalability. The proposal's additive price
composes naturally there; a setpoint scheme would need an explicit aggregation
rule (e.g. average the per-tie $V_{\text{ref}}$ contributions at a shared bus).

---

## 8. Open questions / risks

- **Two-timescale contraction.** No formal non-oscillation proof yet for two (or
  more) neighbours both relaxing $\Delta V_{\text{ref}}$. The *PF-divergence*
  failure mode is structurally gone (bounded setpoints), but inter-coordinator
  oscillation across coupled ties is not yet ruled out. The proposal's warning on
  **systematic communication delays** (oscillation/failure, vs. benign random
  losses) applies and is undischarged.
- **Shared-boundary-bus conflict** under meshing (§7.4) — needs an aggregation
  rule before general $M\ge 3$ meshed use.
- **Nonzero $Q^\star$ schedules** not expressible directly (§7.3).
- **Topology changes** alter the structural drop; the anchor is built from
  *scheduled* nominal voltages. Behaviour is graceful (relax tracks the new
  realised difference; pinned ties are left alone), but an unnoticed topology
  change shifts the implicit reference — consistent with the proposal's caveat on
  hidden topology changes.
- **`anchor` Pareto sweep** (flow reduction vs voltage tracking) not yet mapped;
  returns diminish past `anchor ≈ 0.5`.

---

## Appendix — symbol table

| Symbol | Meaning |
|---|---|
| $e=(i,j)$ | tie line between zones $i<j$ (fixed orientation) |
| $V_i, V_j$ | measured boundary-bus voltages [p.u.] |
| $V_{\text{nom},i}$ | zone $i$'s scheduled boundary voltage [p.u.] |
| $V_{\text{anchor}}$ | $\tfrac12(V_{\text{nom},i}+V_{\text{nom},j})$, the per-tie anchor |
| $\Delta V_{\text{ref}}$ | **agreed** boundary-voltage difference (the coordinator state) |
| $\Delta V_{\text{realized}}$ | $V_i-V_j$, the measured difference |
| $V_{\text{ref},i/j}$ | per-side setpoints, $V_{\text{anchor}}\pm(\text{split})\Delta V_{\text{ref}}$ |
| $\mathrm{err}_{i/j}$ | $V_{i/j}-V_{\text{ref},i/j}$, boundary tracking error |
| $m_e$ | combined envelope marginal (the "dual-like" residual, diagnostic) |
| $\kappa$ | per-side split (0.5 = symmetric) |
| relax | $\alpha g_v$, marginal-feedback gain $\in(0,1]$ |
| anchor | subsidiarity pull toward zero exchange $\ge 0$ |
| $\Delta$ | deadband on the anchor [p.u.] |
| $\Delta_{\max}$ | clip on $\Delta V_{\text{ref}}$ [p.u.] |
| $g_v$ | the zone's primary voltage-tracking weight |
| $Q_{\text{tie},e}$ | tie reactive flow [Mvar]; bounded by the local soft cap, not tracked |
