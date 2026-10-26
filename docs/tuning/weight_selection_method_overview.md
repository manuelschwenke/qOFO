# Controller weight selection — method overview

Short reference to the procedure frozen in
[`2026-10-25_weight_selection_procedure.md`](2026-10-25_weight_selection_procedure.md)
(the binding text) and executed in
`results/THESIS_ch8_weight_selection_band_20261025/`.

## 1. Purpose

Select the ten objective and step weights of the cascaded OFO controllers,
TS-MIQP per zone and STS-MIQP per HV network, so that **TS voltage tracking
is as good as possible**, subject to:

- operational limits (wear, voltage corridor, AVR smoothness);
- a contraction (stability) screen;
- a settling requirement of the subordinate loop.

STS support for the TS matters only through its effect on TS voltage
tracking. The STS's own voltages enter as a tie-break, and through a DSO
operating band of 0.92–1.08 pu that is part of the configuration, not of the
search.

## 2. How it is planned

- **Frozen before use.** Every rule, range, seed and limit is fixed in
  writing before the procedure runs. It is applied **once**, and its output is
  used as it comes out.
- **Development history is separate.** Preliminary campaigns used to develop
  the procedure are recorded, but they are not part of it.
- **Deterministic and recorded.** Seeds are fixed (Sobol, measurement noise),
  and the code state is recorded at launch. Re-simulating any run reproduces
  every stored metric exactly; this was checked on 2026-10-24.

## 3. Name and classification

A **simulation-based, derivative-free two-phase search**:

1. analytic initialisation (Stage I);
2. **quasi-Monte-Carlo screening** with a scrambled Sobol design (Stage II-a);
3. **local random search with successive region reduction** around the
   incumbent, of the Luus–Jaakola type (Stage II-b).

Candidates are judged by **constraints as a filter** followed by a
**lexicographic order with an indifference band** (a lexicographic
semiorder). There is no composite cost and no scoring weights.

## 4. Why this fits the problem

| Property of the problem | Consequence in the procedure |
|---|---|
| Closed-loop performance exists only as the output of a nonlinear AC simulation with integer actuators (taps, banks), disturbances and measurement noise: a black box, non-smooth, with no gradients. | Derivative-free search on simulated windows. |
| One evaluation costs ≈ 40 min per 12-h window. | A cheap analytic screen discards ≈ 77 % of draws before any simulation; a space-filling design covers the box with few points; the local stage concentrates the remaining budget. |
| The MIQP objective is positively homogeneous in the whole weight vector, so a common rescaling leaves the trajectory unchanged. | g_v is fixed as the gauge; nine ratios are searched, in log space because weights span decades. |
| The criteria are incommensurable (pu, Mvar, operations/day), and some are gameable. Minimising the interface-Q error can be done by freezing the PCC setpoint. | Operational limits act as hard filters. The ranking is lexicographic on the primary goal. e_q is bounded, never minimised. |
| A slow layer drives a faster subordinate layer. | A contraction screen per layer, and a settling requirement on the isolated subordinate loop. |

## 5. Parameters

Gauge g_v = 4·10⁴ (fixed). The nine searched weights, with their Stage-I
values and Stage II-a ranges (Stage-I value × 10^(±span)):

| Weight | Meaning | Stage I | Span [decades] | II-a range |
|---|---|---|---|---|
| `g_q` | STS interface-Q tracking | 0.04 | ±1.5 | 0.00127 – 1.27 |
| `dso_g_v` | STS voltage tracking | 1.0·10⁴ | ±1.5 | 316 – 3.16·10⁵ |
| `g_w_der` | TS DER step price | 0.0931 | ±1.0 | 0.0093 – 0.931 |
| `g_w_pcc` | PCC setpoint step price | 0.0931 | ±1.0 | 0.0093 – 0.931 |
| `g_w_dso_der` | STS DER step price | 0.502 | ±1.0 | 0.050 – 5.02 |
| `g_w_gen` | AVR setpoint step price | 1.0·10⁸ | ±1.0 | 10⁷ – 10⁹ |
| `g_w_tso_oltc` | TS tap price | 21.37 | ±0.5 | 6.76 – 67.6 |
| `g_w_tso_shunt` | MSC/MSR bank level price | 27.21 | ±0.5 | 8.61 – 86.1 |
| `g_w_dso_oltc` | STS coupler tap price | 16.64 | ±0.5 | 5.26 – 52.6 |

Stage II-b boxes are centred on the current best candidate and are **not**
confined to the II-a ranges.

## 6. Rules

**Stage I: analytic starting point**, no simulation; six operating points
W1–W6.
- **Objective weights:** priorities π = g·σ² = 1, with σ_v,TS = 0.005 pu,
  σ_v,DS = 0.010 pu and σ_Q = 5 Mvar. This gives g_v = 4·10⁴,
  dso_g_v = 10⁴ and g_q = 0.04. τ = 1 sets g_w_der = g_w_pcc. g_w_gen = 10⁸
  is fixed by AVR smoothness.
- **Integer step prices** (taps, banks) come from the commit rule
  g_w = 2 g_V |Σ_j H_V[j,i]| Δv_eng, taking the strongest column. Δv_eng is
  0.0075 pu for TS taps, 0.015 pu for STS taps and 0.010 pu for banks.
- **Continuous scale per layer:** bisected so that the worst area and
  condition has λ_max(B) = 1, with
  B = Q^{1/2} H G_w^{-1} H^T Q^{1/2}. B counts one bank column per STS (the
  stronger of MSC and MSR), because the MSC/MSR exclusion lets only one of
  them move in a given direction.

**Stage II-a: global screening**
- 512 scrambled Sobol draws, seed 20260910, in the box of §5.
- Analytic screen: worst λ_max(B) ≤ 1.6/1.035 = 1.546 over 6 conditions ×
  7 layers (3 TS zones, 4 STSs). This leaves 119 survivors.
- The survivors and the Stage-I point are simulated 12 h in W1–W4 and W6.

**Stage II-b: local refinement**
- In round r = 1, 2, …: 64 Sobol draws (seed 20260910 + r) in a box centred
  on the current best candidate, with spans f × (II-a spans), f = 0.5 in
  round 1. Draws are screened and simulated as in II-a.
- After each round the selection rule is applied, without the settling
  check. A new winner recentres the box with the same f; otherwise f is
  halved.
- The stage stops when f would fall below 0.125, or after 4 rounds.

**Selection over all evaluated candidates**

1. **Filter**, on W1–W4 and W6:
   - taps ≤ 30 operations per transformer and day;
   - banks ≤ 12 commits per bank and day;
   - no hunting of taps or banks (opposite moves of one device < 2 h apart
     with no new disturbance);
   - AVR setpoint step ≤ 0.005 pu;
   - no bus outside 0.90–1.10 pu;
   - no power-flow failure;
   - λ_max(B) ≤ 1.6;
   - e_q ≤ 3 × e_q of the Stage-I point.
2. **Rank** on e_v(TS). Candidates within 2 % of the best are ordered by
   e_v(DS).
3. **Settling requirement**, checked in rank order on the isolated-STS step
   bank (120 interface-Q steps, W1–W4, W6, steps ≤ 10 Mvar):
   - the 95th percentile of the settling iterations n_s is at most
     3 N_inner = 27, with N_inner = T_TS/T_STS = 9;
   - the median residual is at most 1 Mvar.
   
   A failed candidate is removed and the ranking recomputed. The first
   candidate that passes is the result.

**Metrics**
- e_v(TS), e_v(DS): time mean of the per-bus spatial RMS deviation from the
  controllers' voltage references, as a quadratic mean over zones or STSs and
  over windows.
- e_q: time-weighted integral of the absolute interface-Q tracking error.

## 7. Execution

| Item | Value |
|---|---|
| Network | IEEE 39-bus TS with four 110-kV STSs, `rural_700`, DSO_3 ×2 |
| Banks | 2 × 25 Mvar MSC and MSR per STS, integer inputs of the TS-MIQP, holding time 1800 s |
| DSO band | 0.92–1.08 pu, soft constraint with slack weight 10⁹ |
| Cadence | T_TS = 180 s, T_STS = 20 s, dt = 20 s |
| Windows (start, 2016) | W1 05 Jan 08:00, W2 22 Feb 13:00, W3 04 Jan 08:00, W4 26 Nov 10:00, W6 18 Jul 14:00, 12 h each. W5 (03 Feb 20:00, no DER reactive capability) is simulated for the result only. |
| Disturbance per window | W1, W3: generator trip at 4.8 h, restored after 1 h. W2, W4: four load steps of +120 MW / +60 Mvar at bus 5 from 5.4 h, 20 min apart. W6: three load steps of −110 MW / −55 Mvar at bus 27 from 5.4 h. |
| Compute | II-a: 600 runs on 38 workers (≈ 12 h). II-b: ≤ 4 rounds on 30 workers (≈ 4 h each). Settling: ≈ 5 h. |
| Result | `configs/paramsets/ch9_selected_band_2026_10.json` |

## 8. Related literature

- **OFO as feedback control:** Hauswirth, Bolognani, Hug, Dörfler,
  *Optimization algorithms as robust feedback controllers*, Annual Reviews in
  Control, 2024 (arXiv:2103.11329).
- **Sobol sequences and scrambling:**
  - Sobol', *On the distribution of points in a cube and the approximate
    evaluation of integrals*, USSR Comput. Math. Math. Phys. 7(4), 1967;
  - Owen, *Scrambling Sobol' and Niederreiter–Xing points*, J. Complexity
    14(4), 1998;
  - Roy, Owen, Balandat, Haberland, *Quasi-Monte Carlo methods in Python*,
    JOSS 8(84), 2023 (the `scipy.stats.qmc` implementation used).
- **Space-filling designs for expensive simulators:** Santner, Williams,
  Notz, *The Design and Analysis of Computer Experiments*, 2nd ed., Springer,
  2018.
- **Region-reduction random search (Stage II-b):** Luus, Jaakola,
  *Optimization by direct search and systematic reduction of the size of
  search region*, AIChE J. 19(4), 1973.
- **Direct search and derivative-free optimisation:**
  - Kolda, Lewis, Torczon, *Optimization by direct search: new perspectives
    on some classical and modern methods*, SIAM Review 45(3), 2003;
  - Conn, Scheinberg, Vicente, *Introduction to Derivative-Free
    Optimization*, SIAM, 2009.
- **Simulation optimisation:** Amaran, Sahinidis, Sharda, Bury, *Simulation
  optimization: a review of algorithms and applications*, Ann. Oper. Res.
  240, 2016.
- **Lexicographic selection with a threshold:**
  - Ehrgott, *Multicriteria Optimization*, 2nd ed., Springer, 2005;
  - Tversky, *Intransitivity of preferences*, Psychological Review 76(1),
    1969 (the lexicographic semiorder).

Volumes and pages are given from memory and should be checked before they
are cited in the thesis.

## 9. Limits

- **No optimality claim.** The procedure finds the best candidate among
  those it evaluates; it does not prove a global optimum. In the 2026-10-24
  application Stage II-b ended at its round cap, while its last round no
  longer improved e_v(TS).
- **Validity is bounded.** Results hold for the stated network, windows and
  disturbances. The contraction screen is a linearised, relaxed condition, so
  quantisation and holding times are outside it.
