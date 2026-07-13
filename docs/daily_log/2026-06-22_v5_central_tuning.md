# 2026-06-22 — Tune V5 (central OFO) into a valid upper bound

**Author:** Manuel Schwenke / Claude Code
**Scope:** `experiments/005_CIGRE_MULTI.py` (CIGRE 2026 case study), V5 = single
centralized OFO reference vs V4 = proposed cascaded TS-OFO + STS-OFO.

## Problem

V5 (`control_scope='central'`) is meant to be the **best-case upper bound**, but
performed *worse* than the well-tuned V4 on the headline voltage metric. The user
asked (a) how to compare fairly when the variants need different tuning, and
(b) whether V5 can be tuned to match V4.

## Diagnosis (established fact)

One OFO tick (unconstrained, slack/usage dropped) is
`σ* = −G_w⁻¹ Hᵀ diag(g_v)(V−V*)`, so the per-tick voltage-error map is
`e_{k+1} = (I − M) e_k` with **`M = H_V G_w⁻¹ H_Vᵀ diag(g_v)`**; OFO is stable iff
`eig(M) ⊂ (0,2)`, well-damped for `λ_max(M) ≲ 1`.

Going monolithic (V5) changes `M`: every shared actuator column of `H` now hits
*both* TN and HV voltage rows, so the same `G_w` produces larger steps. Measured
with the new curvature probe:

- V5 at the previous tuning (`g_v=5E7`, ~5× V4): `λ_max(M) ≈ 6 ≫ 2` → **unstable /
  oscillatory** — the real cause of "performs worse" (not suboptimality).
- V5 at V4's gain ratios (`g_v=1E7`): `λ_max(M) = 1.237` — stable but over-driven.
- `cond⁺(M) ≈ 4e10`: some voltage directions are nearly uncontrollable by the
  actuator set — structural (shared by V4's union of actuators), not tuning.

## Method (decision: per-variant tuning to a common metric; hand-tune ratios + global κ)

Common evaluation metric = `rms_v_ts_pu` (unweighted system voltage RMS, already
decoupled from any controller's `g_v`). Each variant tuned separately to it.

V5 retuned to be a valid upper bound:
1. **Match V4 per-class gain ratios** `g_v/g_w`: `g_v=1E7`, `central_dso_g_v=1E5`,
   `g_w_der=100`, `g_w_dso_der=1000`, `g_w_gen=5E9`, `g_w_tso_oltc=1E4`,
   `g_w_dso_oltc=200`.
2. **Remove the cadence handicap:** `central_period_s=180 → 60` (= `dt_s`, fire every
   step) so V5 matches V4's per-step STS loop.
3. **Global κ cooling of the whole `g_w` block:** `KAPPA_V5=1.25` (since `M ∝ G_w⁻¹`,
   κ scales `λ_max` by 1/κ) → `λ_max(M) = 1.237 → 0.99` (well-damped). One scalar
   preserves V4's inter-class balance.

## Code changes

- `configs/multi_tso_config.py`: added read-only `debug_central_curvature` flag.
- `experiments/runners/multi_tso_dso.py`: `_dump_central_curvature()` prints
  `λ_max/λ_min/cond/suggested-κ` from the exact expanded `H`, `g_v_per_bus`, and the
  per-variable `g_w` the MIQP uses (symmetric similarity form, read-only); called once
  after `central_controller.initialise(...)`.
- `experiments/diag_v5_curvature.py` (new): fast probe — sets the flag and a truthy
  `pre_loop_hook` so the 600-step loop is skipped (setup-time only).
- `experiments/005_CIGRE_MULTI.py`: V5 overrides retuned (above) + module-level
  `KAPPA_V5` scaling of the V5 `g_w` block; re-enabled V4 in `VARIANTS`; added
  `per_variant_params_block()` and wired it into `write_tables` so Table `tab:params`
  reports each variant's *own* tuned weights (the methodology requires it).

## Result (360-min run, wind_replace, 6 contingencies, 600 steps each)

V5 evaluated at two cadences (same weights, `λ_max(M)=0.99` in both — cadence does
not change the per-tick step map):

| metric | V4 (proposed) | V5 @ 60 s (every step) | V5 @ 180 s (= V4 TSO) |
|---|---|---|---|
| `rms_v_ts_pu` ↓ | 0.006244 | **0.006121** | 0.006951 |
| `m_bar_pu` (reserve) ↑ | 0.366 | 0.352 | 0.350 |
| `res_util` | 0.129 | 0.191 | 0.209 |
| `rms_e_sts_mvar` | 1.25 | — | — (no iface-Q tracking) |
| `rms_q_tie_mvar` | 26.3 | 27.4 | 33.6 |
| `n_sw` | 19 | 39 | 41 |

All converged (600/600 records, no `ReachabilityViolation`).

**Key finding — the cadence decides whether V5 is an upper bound:**
- **V5 @ 60 s (every step):** weakly dominates V4 (0.006121 < 0.006244). This is the
  true best-case reference — the central controller given maximal actuator freedom
  (config docstring: `central_period_s=None` ⇒ every step ⇒ "correct best-case cadence").
- **V5 @ 180 s (= `tso_period_s`):** *loses* to V4 (0.006951 > 0.006244) and is worse on
  tie-Q and switching. At 180 s the monolithic controller updates **all** actuators only
  every 3 min, whereas V4's cascade keeps a **fast DSO inner loop firing every 60 s**
  (`dso_period_s=20 ≤ dt_s=60`). That fast inner loop is a genuine architectural
  advantage of the cascade; a slow centralized controller gives it up, and no weight
  tuning recovers it (a cadence gap is not a curvature gap).

**Committed config:** `central_period_s=180` (user decision: cadence-matched to V4's
TSO frequency). Consequence: V5 is no longer an upper bound under this setting — see the
open decision below.

## Open decision (for the user / paper narrative)

The two desiderata "cadence-matched to V4's TSO (180 s)" and "V5 is a valid upper bound
(≥ V4)" **conflict**, because V4's fast 60 s DSO loop is doing real work:

- **Option A — V5 @ every step (60 s):** present V5 as the genuine centralized upper
  bound. Supports "V4 nearly attains the centralized optimum." (V5 wins by ~2%.)
- **Option B — V5 @ 180 s (current):** present a cadence-matched architectural
  comparison. Then the headline is the opposite and arguably stronger: *the cascade's
  fast inner loop lets a decomposed controller beat a slow centralized one* — i.e. the
  decomposition is not just near-optimal, it is advantageous at equal top-layer rate.
- **Option C — cadence-matched *all layers*:** also slow V4's DSO to 180 s. Cleanest
  isolation of pure decomposition, but it changes the *proposed* method's numbers, so
  it is the least attractive.

## Other risks

- κ calibrated from the `t=0` cached `H`; `λ_max=0.99` leaves margin and both runs
  stayed stable across all 6 contingencies.
- The reserve / switching trade should be **framed**, not hidden: V5 is a *voltage*
  reference, not a Pareto-dominant point, at either cadence.
- If a future change makes V5 trail V4 *at 60 s*, suspect structure (frozen `shared_jac`,
  or the zeroed 3W-OLTC `Q_gen` columns in `_build_oltc3w_columns`) before tuning.
