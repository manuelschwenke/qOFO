# Gate-E: DSO coupler-tap runaway fixed by raising `g_w_dso_oltc` (150 → 1000)

**Date:** 2026-07-22
**Author:** Manuel Schwenke (with research assistant)
**Area:** Phase-6 RMS closed-loop replay / DSO Layer-2 MIQP tuning
**Runs:** baseline `results/rms_phase6_replay/0036` (g_w_dso_oltc=150), fix `0038` (g_w_dso_oltc=1000); plant floor `results/rms_openloop_uy/0003`

---

## What was changed

The DSO (Layer-2) OLTC switch-penalty weight in the local MIQP objective was raised
from **`g_w_dso_oltc = 150`** to **`1000`**. The TSO (Layer-1) OLTC weight was left
untouched at `g_w_tso_oltc = 5000`. No plant, sensitivity, or measurement code changed —
this is a pure controller-tuning change applied identically to both the static and the
RMS closed loop.

Mechanism of application: the CLI flag `--dso-oltc-switch-cost` on
`experiments/run_rms_phase6_replay.py` sets **only** `_cfg.g_w_dso_oltc` on both
`runner_static` and `runner_rms` configs (an earlier version of the flag also lowered
the TSO weight to 1000 — that confounded run, 0037, was aborted and is not used).

## Why

Under `g_w_dso_oltc = 150` the RMS closed loop exhibited a **DSO_4 coupler-tap runaway**:
the RMS plant's network-coupling transformers marched monotonically downward in tap
position while the static plant's identical couplers stayed put. This was previously
mis-attributed to a plant/load-application defect; the `u→y` open-loop test
(`rms_openloop_uy`, replaying the static run's *exact* actuator+profile timeline into
the RMS plant) refuted that — with identical `u` the plant residual is only
**interface_q RMSE ≈ 2.94 Mvar**. The runaway was therefore a genuine **closed-loop
control divergence**: a ~0.019 pu RMS/static voltage offset, amplified through discrete
tap decisions, because a tap buys ~250 objective units of interface-Q improvement while
the switch penalty (150) was too cheap to prevent the two plants from making *different*
discrete choices.

Raising the switch penalty makes both loops prefer the continuous DER actuator over the
discrete tap, so the discrete decisions of the two plants re-align.

## Key structure of the change

- `configs/config.py` default `g_w_dso_oltc` is `1.0` but is **overridden by
  `make_config()`** in `experiments/run_multi_system_ofo.py` (line ~256) to `150`. The
  Gate-E replay's `make_gate_e_config` inherits that, then `--dso-oltc-switch-cost`
  overrides it to the test value. **The operative baseline is 150, not the config
  default 1.0.**
- The weight enters the DSO MIQP as the L1/switch penalty on `|Δtap|` per interval; the
  DER effort weight is `g_w_dso_der = 800`. Raising the OLTC weight above the
  per-tap interface-Q gain (~250) is what removes the incentive to chatter.

## Results (600 s, 30 intervals, profiles on, ±1.0 pu DER capability override)

**Coupler tap trajectories (RMS plant):**

| coupler        | baseline (150) RMS      | fix (1000) RMS   |
|----------------|-------------------------|------------------|
| DSO_4·trafo_9  | −2 → −6, 4 switches     | −1, 0 switches   |
| DSO_4·trafo_11 | −2 → −6, 4 switches     | −1, 0 switches   |
| DSO_2·trafo_5  | −3 (static −2)          | −2 (= static)    |
| DSO_3·trafo_8  | −3 (static −2)          | −2 (= static)    |
| all others     | matched static          | matched static   |

Under the fix, **every** DSO coupler tap is frozen and identical between the static and
RMS plants (0 switches, span 0). The runaway is eliminated and *all* discrete decisions
re-aligned, not only DSO_4's.

**RMS-vs-static reproduction (the actual objective — "RMS controller nearly reproduces
the static controller"):**

| quantity          | plant floor (u→y) | baseline (150) | fix (1000) |
|-------------------|-------------------|----------------|------------|
| interface_q RMSE  | 2.94 Mvar         | **4.50 Mvar**  | **2.57 Mvar** |
| interface_q max\|e\| | 5.73 Mvar      | 15.17 Mvar     | 6.30 Mvar  |
| zone_voltage RMSE | 0.00298 pu        | 0.00447 pu     | 0.00447 pu |

The closed-loop RMS-vs-static interface-Q difference collapses from 4.50 → **2.57 Mvar**,
i.e. onto (marginally below) the ~2.94 Mvar open-loop **plant floor**. The residual is now
essentially the irreducible plant-model difference; the controller-divergence term
(baseline ≈ 4.50 − 2.94 = 1.6 Mvar on top of the floor) is gone. Zone-voltage tracking was
never the problem and is unchanged.

**No over-freezing / no hidden constraint violation:**

- RMS DSO-side voltages: fix **[1.004, 1.049] pu** vs baseline [0.980, 1.047] pu — freezing
  the down-marching couplers keeps the band better-centred (the baseline taps were dragging
  voltage toward 0.98).
- DSO MIQP output slack `max|z| = 0.000` at every interval in both runs — no soft
  constraint (voltage/current/capability) is being absorbed instead of corrected. The frozen
  taps are genuinely optimal, not a masked violation.

## Risks / unresolved

1. **Tracking offset unchanged (separate issue).** The DSO interface-Q *tracking* error
   (q_actual vs q_set, ~14–26 Mvar) persists and is **identical in static and RMS**
   (e.g. DSO_4·trafo_10: static +2.13, RMS +5.70, both vs set −6.93). It is not created or
   worsened by this change; it is the previously filed steady-state DSO offset, rooted in
   the QVPRE anchor / achievable capability at the operating voltage, not in tap freezing.
2. **Tested under the ±1.0 pu DER capability override** (`der_q_capability_override_pu=1.0`),
   which must be reverted to VDE-AR-N-4120 before publishing (filed task). The 1000 weight
   should be re-validated once real capability is restored — with tighter DER headroom the
   loop may *need* a tap occasionally, so the sweet spot could sit lower.
3. **Only 600 s / one operating point.** The runaway was a slow monotonic drift; 600 s shows
   4 baseline switches. A longer horizon (the 5 h profile run) should be repeated at 1000 to
   confirm the freeze holds and does not merely delay the drift.
4. **Weight not swept.** 1000 works cleanly here, but the sweet spot (above the ~250 per-tap
   interface-Q gain, below where genuine voltage excursions can no longer move a tap) was not
   bracketed. If real-capability re-validation shows under-actuation, sweep 300–800.
