# Offset mechanism read out, and the TS fleet-relief option

Timestamp: 2026-10-21, 11:20 CEST.

Reason: the author asked for the shunt offset mechanism to be read, and
proposed a different basis for bank dispatch — relieving the TS's own
synchronous generators and TS-connected DER instead of the subordinate DER —
together with a question about testing it in a campaign.

Changes:

- Added `docs/tuning/2026-10-21_ts_fleet_relief_option.md` with the offset read,
  the weight audit, the assessment and a proposed test scope.
- Added `--g-q`, `--g-res-sg` and `--g-res-der` to
  `experiments/shunt_hybrid_check.py`, applied after the parameter set so the
  screen can override the three weights the audit found at zero. No default or
  control law changed; omitting a flag keeps the parameter-set value.
- Added `docs/handover/2026-10-21_fleet_relief_confirm_stage.md`, the brief for
  running the 12-hour confirm stage on the many-core host, with the Gurobi
  node-lock warning, the reason the Stage-II harness cannot be used in hybrid
  mode, and the command pattern.

**Correction entered 21 October, 14:10.** The `g_q` reasoning below was written
on the belief that `g_q` is a TS-side price and that its zero belonged to the
selected controller. Both are wrong. `g_q` is the *subordinate* controller's
interface-Q tracking weight (`controller/dso_controller.py:1270`), and the zero
belongs to the diagnostic parameter set these runs use,
`ch9_selected_pcc27_gqceil_comp_q_off`, which differs from the selected
`ch9_selected_pcc27_gqceil` in exactly two fields: `g_q` 0.49769 → 0 and
`g_w_pcc` 0.16431 → 1e8. So the screen's 29.4% RMSE gain is "turning
compensation partly back on where it was deliberately switched off", at a value
still 40× below the selected one — not a free improvement the selection missed.
The TS-side analogue is `tso_g_q_pcc`, which is 0.0 in both sets and untested.
Every shunt result of the last three days was measured with interface-Q
compensation off and the delegated request frozen to within 3e-8 Mvar per hour.
The mechanisms stand; the numbers do not transfer.

The original reasoning, retained for the record: `g_q` was settled as a
structural yes — the TS issues the interface setpoint, so with `g_q = 0` any
subordinate tracking error inside the reported band is free to it, an unpriced
cross-layer externality. The value is bracketed rather than
fixed: equalising the mean squared cost at the baseline gives `g_q = 0.0805`
(1 Mvar interface error equals 0.00142 pu voltage error at one bus), but at that
weight the first-order gradient of a 25 Mvar bank step is dominated 6.45 : 1 by
the Q term, so the screen uses `g_q = 0.0125`, the equal-gradient anchor, with
0.003 and 0.08 as the bracket.

The feedforward offset stays and needs no change: its documented purpose is that
the subordinate controller does not counteract a committed switch, which is what
fleet relief requires. The screen therefore runs the `regular` case only, with
both slow-support weights at zero, so the paired transfer never applies.

Findings:

- The offset acts in two paired places: the subordinate controller is
  dispatched `u_pcc + offset`, and the reported capability deltas are shifted
  by `−offset` before reaching the TS MIQP, so the emitted setpoint stays
  inside the reported band. Its stated purpose is that the subordinate
  controller does **not** counteract a committed switch.
- The tracking-support transfer therefore cancels the offset's purpose: holding
  the dispatched value fixed is what makes the subordinate controller counteract
  the switch. The two intents are opposite. For tracking that is deliberate;
  for reserve the offset-only convention is the one that matches the mechanism.
- The margin derivation of this morning is consistent with the mechanism: the
  recorded band is the unshifted DSO report and the recorded setpoint is the
  dispatched value, which is the pair the invariant constrains.
- **Weight audit of the selected set: `g_q = 0`, `tso_g_res_sg = 0`,
  `tso_g_res_der = 0`, `g_v = 40000`.** The TS objective has one tracking term,
  voltage, and no reserve term at all; interface Q is bounded by a
  slack-penalised limit rather than tracked. The bank's regular gradient is
  therefore priced purely on its voltage effect, because the `Q_PCC` block of
  `grad_y f` is identically zero and the `Q_gen` block is zero while
  `tso_g_res_sg = 0`. The ratchet was not a mis-priced trade-off; there was no
  competing term to trade against. The interface error that degraded from 4.37
  to 8.06 Mvar is the subordinate controller's own tracking error and is
  invisible to the TS objective, which is why no movement weight could price it.
- The fleet-relief option needs **no new machinery**: `Q_gen` is already a TS
  output, `grad_y f` already carries `2·g_res_sg·(Q_gen − Q_mid)/Q_half²`, and
  the hybrid bank column already includes `dQ_gen/dQ_shunt` through
  `compute_dQgen_dQ_shunt_matrix`. With `tso_g_res_sg > 0` a bank step's effect
  on generator reserve is already priced in the regular bank gradient.
  `tso_g_res_der` enters the joint solve directly. The evaluation metrics also
  already exist — `m_bar_mvar`, `m_bar_pu`, `res_util` — unlike the
  subordinate-reserve case. It is a configuration experiment, not an
  implementation.

Open: `g_res_sg` must be scaled against `g_v` and `g_w_tso_shunt` re-derived
for each setting, since the bank gradient gains a `Q_gen` component. With
`g_q = 0` a fleet-relief dispatch will again trade subordinate tracking away
unless `g_q` is enabled; whether Layer 1 should *track* or only *bound*
interface Q is a modelling decision to settle before any campaign. Enabling
either reserve term changes the selected controller and therefore the campaign
baseline. The proposed scope is a 6-variant one-hour screen followed by one
12-hour window for the survivors, not the 714-simulation protocol, which in any
case needs the Stage-I extension hybrid mode still lacks.
