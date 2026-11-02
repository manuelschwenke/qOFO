# Controller weight selection, procedure v2: subordinate layer first (Ch 8, Experiment B)

**Status:** FROZEN 2026-11-02, before the first simulation of its campaign. It is applied
once, by `experiments/ch_09_parameter_selection/ch_9_8_procedure_v2.py`. Its output is
the weight set of the commitment base, whatever it is.

**Why v2.** Procedure v1 (`2026-10-28_weight_selection_procedure.md`, with amendments 2
and 3) produced no settling-capable winner on the commitment base. 29 settling checks
gave one pass. Diagnosis and options: `2026-11-02_procedure_v2_proposal.md`.

**Author decisions of 2026-11-02:**
- option B, the cascade design;
- DSO band 0.92–1.08;
- 8 workers on e5wimi and 36 on e5power;
- TS tap engagement designed by Claude, with the condition that the "OLTCs should not be
  just inert";
- hunting: 0 preferred, one event acceptable.

## Configuration

- **As v1 on the commitment base:** `rural_700` with DSO_3 ×2; merit-order unit commitment;
  slack Q limit 1200 Mvar; generator V_max 1.05 pu.
- **Banks:** MSC/MSR in the TS-MIQP with per-STS exclusion and a 1800 s holding time;
  TS integer cooldown 4 iterations; tap lock 5 iterations.
- **Timing:** T_TS 180 s, T_STS 20 s, dt 20 s; measurement noise as `make_config()`.
- **Voltage limits:**
  - DSO controllers hold an operating band of 0.92–1.08 pu (soft, slack price 10⁹);
  - TS controllers keep 0.90–1.10 pu.
- **Windows:** W1–W4 and W6, 12 h each. W5 is simulated only for the result.
- **Operating points:** the 12-h captures of run `0001_2026-10-31_135000_ch_9_6_full_procedure`
  (cache key `netf8c419bf`). The capture is weight-independent, so it is reused.

## Stage I-TS: supervisory starting point (unchanged from v1)

- Gauge g_v = 4·10⁴ (π_v,TS = 1 at σ = 0.005 pu); τ = 1; g_w_gen = 10⁸.
- TS contraction target λ_max(B_TS) = 1.
- Integer step prices from the commit rule g_w = 2 g_V |Σ_j H_V[j,i]| Δv_eng:
  - Δv_eng,TS = 0.0075 pu (TS taps);
  - 0.010 pu (MSC/MSR banks).

## Stage I-STS: subordinate starting point (new)

- **Fixed design values:**
  - π_v,DS = 1;
  - Δv_eng,STS = 0.03 pu, so θ_STS,min = 0.03 and the strongest coupler commits at
    1.06 pu under a uniform deviation;
  - STS contraction target λ_max(B_STS) = 1.4.
- **Three variants** π_q ∈ {10, 20, 30}, giving ρ = dso_g_v / g_q = 2.5·10⁴, 1.25·10⁴ and
  8.3·10³.
  - Each variant is written as a parameter set `stage1_v2_uc_2026_11_pq<π_q>`.
  - Its TS part is Stage I-TS.
- **Verification.**
  - Each variant gets the settling test of v1: isolated-STS step bank, 120 steps in W1–W4
    and W6, steps capped at 10 Mvar.
  - Choice: the variant with the largest ρ whose **measured** p95 of the settling
    iterations is ≤ 25 and whose median residual is ≤ 0.8 Mvar. These are margins of 2
    iterations and 0.2 Mvar to the acceptance criterion below.
  - If no variant qualifies, the campaign stops and is reported; no search follows.
- **The chosen variant is the Stage-I point.**

## Stage II-a: global screening (supervisory weights only)

- 512 scrambled Sobol draws, seed 20260910, in a log box centred on the Stage-I point.
- Spans:
  - ±1.0 decade for g_w_der, g_w_pcc and g_w_gen;
  - ±0.5 decade for g_w_tso_oltc and g_w_tso_shunt.
- **Fixed:** g_v and the STS weights (g_q, dso_g_v, g_w_dso_der, g_w_dso_oltc).
- **Analytic screen.** A draw is discarded if worst λ_max(B) > 1.546 (= 1.6 / 1.035), or if
  θ_TS,min > 0.009 pu (definition below).
- The survivors and the Stage-I point are simulated in W1–W4 and W6.

## Stage II-b: local refinement (as v1)

- Each round r works around the current best: the winner of the selection rule below
  (without the settling check) applied to everything evaluated so far.
- Per round: 64 Sobol draws, seed 20260910 + r, spans f × the II-a spans, f = 0.5 in
  round 1. Same fixed weights, same analytic screen.
- After each round: a new winner recentres the box with the same f; otherwise f is halved.
- The stage ends when f < 0.125 or after 4 rounds.

## Selection, over every candidate evaluated in II-a and II-b

1. **Filter** (W1–W4, W6). A candidate is admitted only if all of the following hold:
   - the v1 limits: tap operations ≤ 30 per transformer and day; bank commits ≤ 12 per
     bank and day; AVR step ≤ 0.005 pu; no voltage outside 0.90–1.10 pu; no power-flow
     failure; λ_max(B) ≤ 1.6; interface-Q error ≤ 3 × that of the Stage-I point;
   - **hunting:** no hunting of taps or banks (opposite moves of one device < 2 h apart
     with no new disturbance).
     - Only if no candidate is feasible with zero hunting, a tolerance of one tap
       hunting reversal applies (`--max-hunting 1`).
     - Bank hunting tolerance stays 0.
   - **STS engagement:** 0.02 ≤ θ_STS,min ≤ 0.04 pu. This holds by construction (0.03)
     and is kept as a check.
   - **TS taps not inert:** θ_TS,min ≤ 0.009 pu, with

     $$
     \theta_{\mathrm{TS,min}} = \min_{c \in \{W1..W6\}} \min_{i \in \text{TS taps}}
     \frac{g_{w,\mathrm{tso\_oltc}}}{2\, g_V\,|\sum_j H_V^{(c)}[j,i]|}
     $$

     on the Stage-I operating points, with the Stage-I commit-rule machinery.
     - In run 0001, TS taps priced above θ_TS,min ≈ 0.009 pu moved almost only in W6:
       952 of 964 moves, and on average in ≤ 0.3 of the windows W1–W4.
2. **Rank** on e_v(TS), the per-bus RMS deviation from the voltage reference in the TS.
   Candidates within 2 % of the best are ordered by e_v(DS).
3. **Settling confirmation**, in rank order.
   - Same test as the verification. Acceptance: p95 of the settling iterations ≤
     3 N_inner = 27, and median residual ≤ 1 Mvar.
   - One check at a time. A candidate that fails is removed and the ranking recomputed
     among the rest.

**The result** is the first candidate that passes, written as
`configs/paramsets/ch9_selected_uc_2026_11_v2.json`, after its W5 simulation for the
report.

## Execution

- **Hosts.** e5wimi is the master, with 8 simulation workers. e5power has 36 workers:
  shards of the same code, checked against the master's checkout before the start.
- **Verification batch:** the three variants in parallel on e5power, 12 workers each.
- **Confirmation:** on e5power, 36 workers.
- **Stage II** units are split over the hosts in proportion to the workers (8 : 36).

## Execution record (results only; the procedure above is unchanged)

- **2026-11-02 09:19:** run `0004_2026-11-02_091930_ch_9_8_procedure_v2` started
  (commit f51dffc).
- **09:47:** resumed with 10 workers on e5wimi (author). The driver was stopped while it
  waited for the verification shards, which kept running and were not relaunched.
  - Worker counts do not enter the results: the units are deterministic and resume from
    disk.
