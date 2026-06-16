# 2026-06-05 — Togglable explicit reactive-reserve term in the TS controller

**Timestamp:** 2026-06-05 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
The TSO (Layer-1) controller so far minimised reactive-power reserve only
**implicitly** — by preferring to route Q through the DSO cascade rather than
loading its own EHV actuators. The objective had no explicit term keeping TS
actuators centred in their capability bands. This adds an **optional, togglable**
explicit reserve penalisation, matching the formulation sketched in the thesis
(`eq:reserveTS`):

```
+ r_SG^T  G_res^SG  r_SG        (synchronous generators)
+ r_DER^T G_res^DER r_DER       (TS-connected DER)
```

where `r_i = (Q_i − Q_mid,i) / Q_half,i` is the **normalised** distance of each
actuator's reactive output from the midpoint of its (state-dependent) capability
band, `Q_mid = ½(Q_min+Q_max)`, `Q_half = ½(Q_max−Q_min)`. Minimising `Σ r_i²`
keeps actuators centred → symmetric reserve in both directions. Two **separate**
weights let the operator prefer one resource class over the other.

Toggle semantics mirror the tie-line tracking (`tso_g_q_tie`): when a weight is
`0.0` the corresponding block is skipped and the term is **not part of the
objective** (this is the default → fully backward-compatible).

**Scope:** TS synchronous generators and TS DER only. **DSO-connected DER reserve
is intentionally NOT penalised** here (it belongs to the Layer-2 DSO controllers).
Grid-forming converters / STATCOMs (`gridforming_gen_*`) are **not** covered — the
request named only SG and DER (see Open / next).

## Method / where the gradient lands
The TS objective is supplied to the MIQP as a linearised gradient in
control-variable space (`TSOController._compute_objective_gradient`). The penalty
`g · Σ r_i²` has gradient `2·g·(Q_i − Q_mid,i)/Q_half,i²` w.r.t. `Q_i`. The two
resource classes differ in how `Q` relates to the control vector `u`:

- **DER reserve** — `Q_DER` is a *direct control variable* (first `n_der` columns
  of `u`). The coefficient lands straight on `grad_f[:n_der]`; no sensitivity
  mapping needed. Bands from `ActuatorBounds.compute_der_q_bounds(der_p)`
  (VDE-AR-N-4120). Current `Q_DER` read from `self._u_current[:n_der]`.
- **SG reserve** — `Q_gen` is an *output* (the `Q_gen` row block of `H`, offset
  `n_v + n_pcc + n_i`). The per-gen coefficient is mapped to control space via
  `coeff_sg @ dQ_gen/du`, exactly like the existing Q_PCC / Q_tie tracking terms.
  Bands from `ActuatorBounds.compute_gen_q_bounds(gen_p, gen_v)` (Milano PQ
  curve); current `Q_gen` from `measurement.gen_q_mvar`.

**Guards:** bands narrower than `_RES_HALF_EPS = 1e-6` Mvar (e.g. the DER P/S_n<0.1
dead-zone, or a collapsed band) are treated as having no reserve preference
(coefficient 0) to avoid division by zero. OOS generators contribute nothing
(their `Q_gen` H-row is already zeroed; the coefficient is zeroed too for safety).

## What was changed

### `controller/tso_controller.py`
- **`TSOControllerConfig`:** two new fields `g_res_sg: float = 0.0` and
  `g_res_der: float = 0.0` (with docstrings), placed after the Q_tie block.
- **`_compute_objective_gradient`:** new "Component 5" implementing the SG and DER
  reserve gradients described above. Skipped entirely when the respective weight
  is `0.0`.

### `configs/multi_tso_config.py`
- **`MultiTSOConfig`:** two new scalar fields `tso_g_res_sg: float = 0.0` and
  `tso_g_res_der: float = 0.0` (near `tso_g_q_tie`), routed to the per-zone config.

### `experiments/runners/multi_tso_dso.py`
- Per-zone `TSOControllerConfig(...)` construction now passes
  `g_res_sg=config.tso_g_res_sg`, `g_res_der=config.tso_g_res_der`.

## Validation
- `py_compile` clean on all three files; configs instantiate with the new
  defaults (`0.0`) and accept overrides.
- Numerical unit check (saturation test fixture, after `initialise`):
  - **DER:** `der_p=10`, `S_n=100` → band `[−10,+10]`, `Q_mid=0`, `Q_half=10`.
    With `g_res_der=2`, `Q_DER=+5` → `grad[DER]=0.200 = 2·2·5/10²` (sign >0 ⇒ pushes
    Q down toward mid); `Q_DER=−7` → grad <0; V_gen column untouched; dead-zone
    (`der_p=0`) → grad `0.0` (guard, finite).
  - **SG (injected H, `dQ_gen/dV_gen=100`):** band `[−34.9,+86.6]`, `Q_mid=25.8`,
    `Q_half=60.8`. With `g_res_sg=3`, `Q_gen=Q_mid+20` →
    `grad[V_gen]=3.249 = 2·3·20/60.8²·100` (>0 ⇒ QP lowers V_gen ⇒ lowers Q toward
    mid); below-mid flips sign; DER column untouched.
  - Both: weight `0.0` ⇒ gradient identically zero (toggle off).
- Regression: `pytest tests/test_tso_saturation.py tests/test_controller.py
  tests/test_tso_tertiary_shunt.py` → **78 passed, 1 skipped**.

## Open / next
- **Stability diagnostics not updated.** The decentralised contraction check
  (`MultiTSOCoordinator.check_contraction`, via `ZoneDefinition.q_obj_diagonal`)
  estimates curvature from output weights `H^T Q_obj H`. The reserve term adds
  curvature too — `g_res_sg/Q_half²` on the `Q_gen` output, and a *control-space*
  `g_res_der/Q_half²` (a `g_u`-like term) for DER that the output-weight form does
  not represent. With non-zero reserve weights the contraction LHS is therefore
  *approximate* (it under-counts reserve curvature). For the default `0.0` it is
  exact. If reserve weights are used aggressively, revisit whether to fold the SG
  reserve curvature into `q_obj_diagonal` and/or add a `g_u` channel for DER.
- **Grid-forming / STATCOM reserve** (`gridforming_gen_*`) is not penalised. If a
  third class is wanted, add a `g_res_gf` weight mapping through the `Q_gf` output
  rows (offset `n_v + n_pcc + n_i + n_gen`) — same pattern as SG.
- **Tuning:** weights are bare (un-normalised across classes beyond the `Q_half`
  normalisation). Suggest a small-value sweep (start `~1.0`, like the `g_q_tie`
  Phase-B start) and compare against the implicit-only baseline on a 30-min smoke.

## Addendum (2026-06-05, later) — "no influence at g_res_sg=10000" diagnosis

Added an **opt-in** diagnostic `TSOController._debug_reserve_term` (gated by env
`QOFO_DEBUG_RESERVE`, ≤6 prints/controller, **no dispatch effect**). Ran a short
V4 (`tso_mode=ofo`, 12 min ≈ 4 TS firings) of `005_CIGRE_MULTI` with
`tso_g_res_sg` forced to 1e4. Findings:

1. **Wiring is correct** — the term fires in all three zones and lands on usable
   actuator columns (DER `g_w=20`, PCC `g_w=150`, OLTC `g_w=100`) plus the gens'
   own `V_gen` (`g_w_gen=5e7`).
2. **It only runs for TSO-OFO variants.** V1/V2 use `tso_mode="local"` (Q(V)
   droop, no TSO MIQP) and V5 is `control_scope="central"` (separate controller),
   so the term is a structural no-op there. Only **V3/V4** can use it.
3. **Root cause is scale, not (only) the AVR freeze.** Measured
   `||reserve_contrib|| ≈ 90–260` against `||grad_f|| ≈ 1e3–2e5` ⇒ the reserve
   term is **~0.1–2 %** of the total objective gradient, which is dominated by
   voltage tracking (`g_v=3e5`). The reserve gradient scales as
   `2·g_res·r/Q_half`; for transmission machines `Q_half ~ 1e2–1e3` Mvar, so its
   *natural* magnitude is `~g_res/Q_half` — small unless `g_res` is large. The
   implied reserve-only step on the most responsive lever (a DER) was only
   **~0.08–0.19 Mvar/TS-step**. Generators *did* have reserve to recover
   (`max|r| ≈ 0.47–0.53` in zone 1).
4. The strongest physical lever (each gen's own AVR, `∂Q_gen/∂V_gen`) is
   **frozen** by `g_w_gen=5e7`, so reserve can only be recovered indirectly via
   the weaker cross-couplings `∂Q_gen/∂{DER,PCC,OLTC}`.

**Implication / recommendation:** to make the term bite, raise `tso_g_res_sg` by
~1–2 orders of magnitude (≈ 1e5–1e6) and/or lower `g_w_gen`; verify on V3/V4 and
watch `m_bar_pu`/`res_util` (should improve) vs `rms_v` (may worsen). Note the
NOTE at line 128 currently sits at `tso_g_res_sg=0` in the repo.

**TODO:** decide whether to keep `_debug_reserve_term` or strip it once tuned.

### Proof the term changes dispatch (re-sim, not --replot)
Ran V4 twice in one process (weight `0` vs `1e6`, 12-min horizon) and diffed the
records directly:

| signal | max|Δ| (0 → 1e6) |
|---|---|
| `zone_q_gen` | 17.8 Mvar |
| `gen_q_headroom_mvar` | 17.6 Mvar |
| `zone_v_rms_err_pu` | 1.3 mpu |
| `zone_v_gen` | 0.8 mpu |
| `zone_tso_objective` | 3.4e5 |
| `zone_tso_der_p_mw` | 0 (term moves DER **Q**, not P) |

⇒ the term **does** shift generator reactive dispatch (~18 Mvar) and voltage
tracking (~1.3 mpu). So a report of *exactly identical* metrics means the run was
**not re-simulated**: `005_CIGRE_MULTI.py --replot` reloads cached `log.pkl`
(no sim); `--only`/`--skip` may exclude the affected variant; and V1/V2
(`tso_mode="local"`) and V5 (central) never invoke the term. To see the effect,
re-run the OFO variant fresh: `python experiments/005_CIGRE_MULTI.py --only V4`
(NOT `--replot`).
