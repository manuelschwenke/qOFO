# 2026-11-02 — TS live window: Q_PCC setpoint includes the shunt feedforward

## Problem (author, 2026-11-02)
The "Q_PCC to STS" tile of the TS live window plotted Σ u_PCC per zone, the
TSO MIQP decision.  The STS is dispatched u_PCC + q_itf_sh_offset: after a
committed MSC/MSR level change the runner adds the switched-shunt
feedforward (from the hosting DSO's own 3W model) to every coupler setpoint
of that STS, and the cached models are corrected by the rank-1 update.  The
metered interface flow tracks the shifted value, so after the MSC switch at
t ≈ 120 min the tile showed a gap of order 60 Mvar (zone 2) although the
tracking was good.

## Change
- `experiments/helpers/records.py`: new field
  `zone_q_pcc_sh_offset: Dict[int, NDArray]`, parallel to `zone_q_pcc_set`.
- `experiments/runners/multi_tso_dso.py`: on each TSO tick, immediately
  before the setpoint dispatch (after this step's shunt commits), record
  `q_itf_sh_offset` per `pcc_trafo_indices` of each zone.
- `visualisation/live_ts.py`: the tile's setpoint trace is now
  Σ (u_PCC + offset), i.e. the setpoint the STS actually receives.

`zone_q_pcc_set` itself is unchanged (it remains the MIQP decision variable
and is used by `ch_9_3_compare_runs.py` and `018_SBXV_E1.py`).

## Verification
Headless 6 h run, selected paramset, `make_config()` scenario: three MSCs
in zone 2 switch 0→1 at t ≈ 120 min and 1→0 at t ≈ 330 min.  Zone-2 offset
non-zero on 81 of 121 TSO ticks, range [−77.6, +15.7] Mvar; zones 1 and 3
zero.  In the redrawn tile the setpoint lies within a few Mvar of the
metered flow throughout; before the change the gap was the offset itself.

## Open points
- The offset does not return to zero when the banks return to level 0:
  the switch-on steps summed to −77.6 Mvar over the three couplers of each
  STS, the switch-off steps to +93.3 Mvar (each evaluated at the then
  current operating point and tertiary voltage), leaving +15.7 Mvar of
  feedforward with all banks at 0.  This is a property of the runner's
  accumulation, not of the plot; not changed here.  Investigated below.

## Investigation: why the offset does not return to zero
Diagnostic (scratch script, not committed): same 6 h run; at every commit a
copy of the plant was solved with the old and with the new bank level (all
other actuators as written), giving the true interface step; every third
plant solve while a bank was engaged, a copy with all banks at 0 gave the
true bank contribution.  Analysis only; no controller sees these flows.

Per STS (sum over its three couplers), Mvar:

| commit           | ff (DSO model) | TS row (host) | plant secant |
|------------------|---------------:|--------------:|-------------:|
| DSO_1 MSC 0→1    | −25.97 | −25.86 | −27.16 |
| DSO_2 MSC 0→1    | −25.85 | −25.69 | −25.44 |
| DSO_3 MSC 0→1    | −25.81 | −25.74 | −24.64 |
| DSO_1 MSC 1→0    | +30.88 | +31.17 | +27.87 |
| DSO_2 MSC 1→0    | +30.94 | +31.06 | +25.63 |
| DSO_3 MSC 1→0    | +31.48 | +31.76 | +26.77 |
| Σ cycle          | +15.7  |        | +3.0   |

Engagement is accurate to ≈ 1 Mvar per STS; release overestimates by
3–5 Mvar per STS (+13 Mvar in total), which is the residual.

Mechanism (model fact, checked against the numbers): the feedforward is
−S·q_step·V_meas², with S = ∂Q_itf/∂Q_inj from the hosting DSO's cached
model before the switch.  On engagement that model does not contain the
bank: S ≈ 1.00 (back-computed 0.99–1.00).  On release it contains the bank
(rank-1 update at engagement), so S includes the bank's own response
2·b·V·ΔV to the injected ΔQ: back-computed S ≈ 1.06–1.07, against
1 + 2·b·V²·X_th ≈ 1 + 2·0.25·1.16·0.127 ≈ 1.07 from the plant
(b = 0.25 pu, X_th = ΔV/ΔQ at the tertiary from the secant solves).  The
released bank's total injection −b·V² is removed *and* its voltage response
counted again: a first-order double count.  The TSO's own H column has the
same property (TS row 31.2 vs plant 27.9), so the MIQP prices a release
≈ 10–15 % too high.

While all three MSCs were engaged, the offset (−77.6) exceeded the true
joint contribution (−71 … −74) by 4–7 Mvar.  Causes: the three commits of
one instant are superposed at the pre-switch models (the joint effect is not
additive), and the plant-side Q(V) loops absorb part of the step, which the
models do not represent.

Consequence under the selected settings (`tso_g_q_pcc = 0`,
`tso_pcc_capability_on_output = False`, the capability band shifted by the
same offset): the residual acts as a gauge shift of u_PCC and does not bias
the TSO step; it does not vanish and grows with every on/off cycle (sign
follows the device: + for MSC, − for MSR).  At the release instant the STS
is asked for 3–5 Mvar more than the plant delivers, which its own actuators
have to supply.

Options (not implemented, for decision):
1. Ledger per bank: store each committed level's per-coupler feedforward;
   a release removes the stored value of that level.  Offset becomes a
   function of the levels, 0 with all banks at 0; release error here
   −2.6 Mvar in total (77.6 against the true 80.3) instead of +13.
2. Evaluate a release on the model without the bank (apply the DSO's rank-1
   removal first).  Removes the double count; the V² asymmetry remains
   (estimated residual ≈ +10 Mvar here).
3. Re-evaluate the whole offset each tick from the current levels and model.
   Tracks the operating point, but the STS setpoint then moves without a
   switch.

## Decision (author, 2026-11-02): option 1, per-bank ledger
- `configs/config.py`: `shunt_ff_ledger: bool = True`.  `False` reproduces
  runs made before 2026-11-02 (release on the pre-switch model).  The new
  field changes `config_fingerprint`, so cached results recompute.
- `experiments/runners/_multi_tso_helpers.py`: `_ledger_feedforward()`.
  Per bank, one `{coupler: dQ}` entry per engaged level.  Engagement stores
  and returns the model step; release returns the negated top entry and
  drops it.  Fails fast if the ledger depth differs from the bank level or
  the coupler set has changed.
- `experiments/runners/multi_tso_dso.py`: `sh_ff_ledger` next to
  `q_itf_sh_offset`; the commit block applies the helper after the model
  evaluation.  `rec.shunt_commits[*]["delta_q_mvar"]` is the applied step,
  new `"delta_q_mvar_dso_model"` the model evaluation (kept on release, so
  the double count stays measurable).
- Unchanged: engagement feedforward; the TSO MIQP's shunt column (still
  prices a release ≈ 10–15 % high); the central path (no DSO interface);
  `shunt_dispatch="local"` (never writes the offset).
- `tests/test_shunt_ff_ledger.py`: on/on/off/off returns to zero, banks
  independent, fail-fast cases.

### Verification
- Same 6 h diagnostic with the ledger: commits at the same instants;
  engagement feedforward unchanged; release applies +25.97 / +25.85 /
  +25.81 Mvar per STS (Σ 77.6 against the plant's 80.3, error −2.6 Mvar,
  previously +13.0).  Zone-2 offset after the release: 0.00 Mvar
  (previously +15.66).
- Test suite (without `tests/tuning`): 891 passed, 19 skipped, 1 failed.
  The failure, `test_case2_variants::test_m_is_ready_and_has_the_cascade_timescale`
  (900 vs 1080 s), also fails on a clean checkout of HEAD 40f4e14; it is not
  caused by this change.
- Analyses that compare `zone_q_pcc_set` with metered interface flow must add
  `zone_q_pcc_sh_offset` as well; `dso_trafo_q_set_mvar` already holds the
  shifted value on the STS side.
