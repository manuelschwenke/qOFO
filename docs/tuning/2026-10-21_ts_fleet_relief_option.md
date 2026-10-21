# Dispatching the banks for TS fleet relief

Date: 21 October 2026. Status: assessment and proposed test scope. Nothing
implemented, nothing reconfigured.

## The offset mechanism, read out

`q_itf_sh_offset[t]` is a persistent per-interface feedforward offset in Mvar,
accumulated over commits, and it acts in two paired places in
`experiments/runners/multi_tso_dso.py`:

* **Dispatch** (`q_adj`, line 4521): the subordinate controller receives
  `u_pcc + offset`. Its stated purpose, at line 3262, is *"so the DSO does not
  counteract a committed switch (it then rejects only the residual)"*.
* **Band** (line 4660): the reported capability deltas are shifted by `−offset`
  before reaching the TS MIQP, so that `u_pcc ∈ [q_now + dmin − off,
  q_now + dmax − off]`, which is exactly `u_pcc + off ∈ [q_now + dmin,
  q_now + dmax]`: the *emitted* setpoint stays inside the reported band. Without
  this shift, DSO_2 once settled at rail plus offset with a −57.7 Mvar
  unclosable gap (2026-08-13).

Two consequences follow.

First, the margin computed in
`docs/tuning/2026-10-21_reserve_trigger_margin_derivation.md` is consistent
after all: the recorded band is the DSO's own unshifted report and the recorded
setpoint is the dispatched value including the offset, which is precisely the
pair the invariant above constrains. The −0.9 Mvar excursion at `g_w = 5`
therefore is a small residual violation of an invariant the 2026-08-13 note
says should hold to about 0.1 Mvar, and remains unexplained.

Second, and more important: the offset exists **so the subordinate controller
does not counteract the switch**. The tracking-support transfer
(`delegated -= delta_q` alongside `offset += delta_q`) holds the dispatched
value fixed and therefore cancels exactly that purpose — it makes the
subordinate controller counteract the switch, which is what the 2880 s trace
shows. The two intents are opposite. For tracking support the transfer is
deliberate, since the aim is to hold the total reference while the bank removes
an unresolved error. For reserve it is the wrong convention, and the
offset-only convention is the one that matches the mechanism.

## What the selected parameter set actually weights

From the stored configuration of the 20 October runs:

| Weight | Value | Role |
|---|---:|---|
| `g_v` | 40000 | TS nodal voltage tracking |
| `g_q` | **0** | TS interface/boundary Q tracking |
| `tso_g_res_sg` | **0** | synchronous-generator reactive reserve |
| `tso_g_res_der` | **0** | TS DER reactive reserve |
| `g_z_voltage` | 1e9 | voltage *limit* slack |
| `g_z_q_pcc` | 1e6 | PCC-Q *limit* slack |
| `g_z_q_gen` | 100 | generator-Q *limit* slack |

So the TS objective in every run analysed so far contains **one tracking term,
voltage, and no reserve term of any kind**. Interface Q is bounded by a
slack-penalised limit, not tracked. This reframes the earlier results:

* The bank's regular gradient is priced *purely* on its voltage effect, because
  the `Q_PCC` block of `∇_y f` is identically zero and the `Q_gen` block is
  zero while `tso_g_res_sg = 0`. The ratchet was not a mis-weighting of a
  trade-off; there was no competing term to trade against.
* The `interface_error_rmse_mvar` that degraded from 4.37 to 8.06 Mvar is the
  *subordinate* controller's tracking of its delegated setpoint. With `g_q = 0`
  that degradation is invisible to the TS objective, which is why no value of
  `g_w_tso_shunt` could price it.

## The proposed option, assessed

Dispatching the banks to relieve the TS's own fleet — synchronous generators
and TS-connected DER — fits the existing formulation without new machinery, and
it avoids the entire cascade problem diagnosed yesterday: the quantity to be
relieved is a TS output the TS measures itself, so no inter-layer report and no
assumption about a subordinate response is needed.

What already exists:

* `Q_gen` is part of the TS output vector `[V | Q_PCC | I | Q_gen]`, and
  `∇_y f` already carries the block `2·g_res_sg·(Q_gen − Q_mid)/Q_half²`.
* The hybrid bank column already includes `∂Q_gen/∂Q_shunt`, via
  `compute_dQgen_dQ_shunt_matrix` in `experiments/helpers/shunt_hybrid.py`.
* Therefore, with `tso_g_res_sg > 0`, a bank step's effect on generator reserve
  is **already priced in the regular bank gradient**. No code change.
* `tso_g_res_der` is a control-space term on `Q_DER`, so it enters the joint
  MIQP directly and the solver trades bank steps against DER movement. Also no
  code change.
* The evaluation metrics already exist and are the paper's own:
  `m_bar_mvar`, `m_bar_pu` and `res_util = mean_g |Q|/(|Q| + headroom)` in
  `experiments/helpers/comparison_metrics.py`, with
  `gen_q_headroom_series` behind them. This is in sharp contrast to the
  subordinate-reserve case, where no deliverable metric exists yet.

So the option is a **configuration experiment, not an implementation**. That
also makes its risk profile different: it changes the selected controller's
objective, so it changes the campaign baseline.

Open points specific to this option:

* `g_res_sg` must be scaled against `g_v = 40000`. The reserve term is
  normalised by `Q_half²`, so the weight carries its own scale and has to be
  calibrated. The admission-threshold method applies unchanged: with
  `g_res_sg > 0` the bank gradient gains a `Q_gen` component, and
  `tuning/shunt_weight_calibration.py` shows how the distribution moves and
  therefore what `g_w_tso_shunt` should become. The movement weight must be
  re-derived, not reused.
* With `g_q = 0`, interface-Q tracking remains unpriced, so a fleet-relief
  dispatch will again trade subordinate tracking away unless `g_q` is also
  enabled. Whether the thesis wants interface Q *tracked* or merely *bounded*
  at Layer 1 is a modelling decision, not a tuning one, and it should be
  settled before the campaign rather than after.
* Relieving generators and relieving TS DER are different objectives with
  different actuator competition, so `tso_g_res_sg` and `tso_g_res_der` should
  be varied separately before being combined.

## Proposed test scope

A full campaign under the 16 October protocol is 118 analytical survivors plus
a baseline on six 12-hour windows, 714 candidate-window simulations. That is
not an appropriate instrument for an exploratory question, and the Stage-I
analytical screen currently rejects hybrid mode outright because it omits the
bank columns. A focused comparison is the right size:

| Stage | Runs | Cost |
|---|---|---|
| Screen | 1 h diagnostic scenario, 6 variants | ~30 min |
| Confirm | one 12-hour window, the 3 surviving variants | ~3 h |

Variants for the screen, with the banks un-parked and `g_w_tso_shunt`
re-derived for each:

1. baseline, banks parked, weights as selected;
2. `tso_g_res_sg` low;
3. `tso_g_res_sg` high;
4. `tso_g_res_der` alone;
5. both reserve terms;
6. `tso_g_res_sg` high together with non-zero `g_q`, to see whether subordinate
   tracking can be protected while the fleet is relieved.

Primary metrics: `m_bar_pu` and `res_util` for the fleet relief that is the
point of the exercise; `interface_error_rmse_mvar` and the one-sided margins as
the cost side; commits, TS and STS voltage ranges, and solver status as
hygiene. The screen is diagnostic only. A campaign is warranted once a variant
shows fleet relief at an acceptable subordinate-tracking cost, and it then
needs the Stage-I extension that hybrid mode still lacks.

The screen runs in two passes. Pass 1 holds the banks parked under each
weighting, which gives a clean bank-gradient distribution to re-derive
`g_w_tso_shunt` from and, separately, shows what the new weights do on their
own. Pass 2 repeats with the banks active at those derived weights. The weight
cannot be derived from a run that has already switched, hence two passes.

## Screen pass 1: banks parked

`results/fleet_relief_screen_20261021`, one simulated hour each, `--cases
regular` so both slow-support weights are zero, `g_w_tso_shunt` at its parked
default. All six report 0 commits and 720 optimal subordinate solves.

| Variant | `g_q` | `g_res_sg` | `g_res_der` | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] | DER utilisation p95 | TS V max [pu] |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_selected` | 0 | 0 | 0 | 4.36584 | 11.72540 | 0.39897 | 1.0500642 |
| `v2_gq` | 0.0125 | 0 | 0 | **3.08006** | **7.40132** | 0.37027 | 1.0500600 |
| `v3_sg_low` | 0.0125 | 1 | 0 | 3.07879 | 7.39522 | 0.37017 | 1.0500572 |
| `v4_sg_high` | 0.0125 | 10 | 0 | 3.06750 | 7.34125 | 0.36930 | 1.0500324 |
| `v5_der` | 0.0125 | 0 | 1 | 3.07920 | 7.39186 | 0.37019 | 1.0500301 |
| `v6_both` | 0.0125 | 10 | 1 | **3.06650** | **7.33204** | **0.36923** | 1.0500024 |

Raising `g_q` from 0 to 0.0125 improves subordinate tracking by 29.4% in RMSE
and 36.9% in p95, with no bank action and no measurable voltage cost: the TS
voltage range is unchanged to six decimals and aggregate DER utilisation p95
improves from 0.39897 to 0.37027.

**That result must not be read as a deficiency of the selected controller, and
an earlier version of this note did so wrongly.** Two corrections:

* `g_q` is the **subordinate** controller's interface-Q tracking weight
  (`DSOControllerConfig.g_q`, consumed at `controller/dso_controller.py:1270`),
  not a TS-side price. Raising it makes each STS follow its delegated setpoint
  harder; it does not make the TS accountable for what it delegates.
* The zero is a property of the **diagnostic** parameter set these runs use,
  `ch9_selected_pcc27_gqceil_comp_q_off`, not of the selection. That file
  differs from the selected `ch9_selected_pcc27_gqceil` in exactly two fields:

  | field | selected | `comp_q_off` |
  |---|---:|---:|
  | `g_q` | 0.49769 | **0** |
  | `g_w_pcc` | 0.16431 | **1e8** |

  with `dso_g_q_per_area` (DSO_1 0.7747, DSO_2 0.7206, DSO_3 0.8046) identical
  in both. So "comp_q_off" means the subordinate interface-Q objective is
  switched off *and* the TS's delegated-request control is frozen. At 0.0125
  the screen is still 40× below the selected value.

The consequence reaches every shunt result of the last three days: they were all
measured with interface-Q compensation deliberately disabled and the delegated
request immobile. Measured over an hour on a parked-bank run, the delegated
request moves by at most 3·10⁻⁸ Mvar — it is a constant. The mechanisms
established in that configuration are weight-independent and stand: the
admission identity, the shared-objective gate, the margin derivation. The
*numbers* do not transfer: the ratchet, the margin tightening and "bank steps
worsen tracking" were all obtained in a configuration where the TS cannot
re-dispatch the interface and the subordinate controller does not track it.

The genuinely unpriced TS-side term is `tso_g_q_pcc`, which is **0.0 in both**
parameter sets. Its own docstring describes the shunt-relevant mechanism — "TSO
mildly prefers to cancel shunt-induced Q displacement at the interface via
`Q_PCC_set` adjustments rather than overload the DSO. Set 0.0 to keep the rows
informational only" — and it has not been tested.

The two reserve terms add little on top with the banks parked: `g_res_sg = 10`
buys a further 0.4% of RMSE, `g_res_der = 1` nothing measurable, both together
0.44%. That is expected rather than disappointing — with the banks held, a
reserve term can only re-centre generator Q through actuators that are already
dispatched for voltage and interface Q. Their actual test is pass 2, where the
banks can act on them, and the fleet metrics `m_bar_pu` and `res_util` are the
ones that will show it; the table above does not yet report them.

A caveat on reading the small differences: the runs are deterministic at a
fixed seed, so the 0.4% spread between `v2` and `v6` is real rather than noise,
but it is one hour of one scenario and well inside what a different window
could reverse.

### The reserve weights of this screen are too small by about two orders of magnitude

Re-deriving `g_w_tso_shunt` per variant from the parked-bank gradients gives
essentially the same distribution six times over:

| Variant | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `v1_selected` | 2.218 | 12.339 | 28.455 | 33.702 | 34.370 |
| `v2_gq` | 2.059 | 12.332 | 28.444 | 33.690 | 34.355 |
| `v3_sg_low` | 2.067 | 12.333 | 28.453 | 33.696 | 34.357 |
| `v4_sg_high` | 2.134 | 12.338 | 28.537 | 33.752 | 34.377 |
| `v5_der` | 2.027 | 12.001 | 28.311 | 33.519 | 34.187 |
| `v6_both` | 2.084 | 12.014 | 28.402 | 33.583 | 34.209 |

The spread across variants is under 1%, so the bank gradient is still
essentially the voltage term alone. The reason is the normalisation. The
`Q_gen` block of the output gradient is

    2 · g_res_sg · (Q_gen − Q_mid) / Q_half²,

and with `Q_half` of the order of hundreds of Mvar, `Q_half²` is 10⁴–10⁵. At
`g_res_sg = 10` the block contributes of the order of 0.3 to a bank gradient
whose voltage part is 30. The weights 1 and 10 used above were chosen without
computing that normalisation, and they cannot influence a bank decision; pass 2
at those values would have produced six near-identical runs and spent the
compute for nothing. `g_res_der` is slightly more visible — it shifts p90 from
12.34 to 12.00 and changes the sample count — because it acts on the control
gradient directly rather than through a normalised output block, but it is
still far too small to compete.

The scale therefore has to be derived like `g_q` was, not guessed. Three probe
runs with the banks parked, at `g_res_sg` of 10³, 10⁴ and 10⁵, were run to find
where the term begins to bite.

### Probe result: the useful window is narrow and the term fights voltage tracking

Banks parked throughout, so every effect below is the continuous actuators and
OLTCs responding to the changed objective. All three report 0 commits.

| `g_res_sg` | Interface Q RMSE | p95 | DER util. p95 | TS V min | TS V max | STS V min |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (`v2_gq`) | 3.0801 | 7.4013 | 0.37027 | 0.978877 | 1.050060 | 0.984352 |
| 10³ | **2.5786** | **6.0685** | **0.32752** | 0.971950 | 1.047609 | 0.988204 |
| 10⁴ | 4.0277 | 9.5930 | 0.39184 | 0.962478 | **1.061049** | 0.980268 |
| 10⁵ | 4.6797 | 11.3453 | 0.46675 | 0.963916 | 1.052988 | 0.981391 |

The response is **non-monotonic**. At 10³ the term helps on every count — 16%
better interface tracking than `g_q` alone and DER utilisation down from 0.370
to 0.328 — at the price of 0.007 pu of TS minimum voltage. At 10⁴ and above it
overrides voltage tracking and everything degrades together, including a TS
maximum of 1.0610 pu, an excursion past the 1.05 that every other run of this
investigation respected.

That is a direct argument for taking the reserve term **out** of the shared
objective. Placed there it competes with voltage tracking across every
continuous actuator, and only a narrow window near 10³ is beneficial, bounded
above by a voltage-band failure rather than by diminishing returns. A reserve
cost attached to the bank decision alone does not have that competition to lose,
which is the design the author proposed on 21 October.

Caveat: these three runs use the `comp_q_off` diagnostic parameter set, so the
subordinate interface-Q objective is off and the delegated request is frozen.
The qualitative conclusion — shared-objective placement makes the reserve term
compete with voltage, with a narrow useful window — does not depend on that,
but the numbers do not transfer to the selected controller.
