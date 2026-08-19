# 2026-08-13 — RESOLVED: the switched-shunt feedforward moved the setpoint but not its bound

**Author:** Manuel Schwenke / Claude Code
**Timestamp:** 2026-08-13 (rewritten twice same day — see "Retraction" below)
**Reason:** A persistent setpoint-vs-metered gap on the TSO–DSO interface that
opens at every switched-shunt commit and never closes.

## Resolution (Manuel's diagnosis)

The TSO dispatches `u_pcc + q_itf_sh_offset` to the DSO, but the MIQP bounds
`u_pcc` itself against the reported capability band. **The quantity actually
requested was never the quantity constrained**, so the dispatched setpoint
escaped the band by exactly the accumulated offset. Requiring the *emitted*
setpoint to be reachable:

```
  u_pcc + off  in  [q_now + dmin,       q_now + dmax]
<=>  u_pcc     in  [q_now + dmin - off, q_now + dmax - off]
```

so the offset is now subtracted from the reported deltas before the TSO uses
them. Verified numerically: without the shift DSO_2 settled at `rail + offset`
(setpoint −137.46 against a reported rail of −89.5, cumulative offset −48.23,
matched to within 0.3 Mvar over five samples).

| arm | DSO_2 t3w3 final gap | DSO_3 t3w6 |
|---|---|---|
| feedforward ON, band unshifted (the fault) | −57.68 | −16.19 |
| feedforward OFF entirely | +1.60 | −3.60 |
| **feedforward ON + band shifted (implemented)** | **+1.67** | **−7.28** |

With the shift the setpoint tracks the reported rail to ~0.1 Mvar (−92.02 vs
−92.0; −93.67 vs −93.6; −88.62 vs −88.7). Removing the feedforward fixes the
gap equally well but gives up the mechanism's purpose — the DER not jumping at
a switch — so the shift is the correct repair: the design intent was right,
only the bookkeeping was half-applied.

**The between-commit behaviour reverses sign**, which is the structural
evidence that the fix works rather than merely offsetting the symptom:

| DSO_2 t3w3 | at commits | drift between | rate |
|---|---|---|---|
| unshifted | −10.70 | −21.56 | −3.60/h |
| shifted | −13.37 | **+14.96** | **+2.50/h** |

The gap still opens at each commit (unchanged feedforward estimate error, see
Open) but now *closes* between them instead of accumulating.

Source implementation reproduces the monkeypatch prototype exactly (+1.67 /
−7.28 on both), same commits at the same instants.

**Comparison hygiene:** an intermediate pair of runs reporting DSO_2 at −32.18
/ −32.31 was confounded — they executed while the (since-reverted) T' capability
patch narrowed the band and so limited how far `u_pcc` could wander. Disregard
those; the clean fault baseline is −57.68 with the original capability code.

**Changed:** `experiments/runners/multi_tso_dso.py` — capability-band shift in
the DSO step, explicitly cross-referenced to the `q_adj` block in the TSO
setpoint dispatch so the two cannot drift apart again.

**Status of other changes:** none survive. Working tree also carries
`g_z_q_pcc: 1e-2 -> 1e6` (Manuel's edit).

## Retraction

An earlier version of this file claimed the DSO capability report over-stated
its reachable interface range by **2.23x**, and a patch was written to post-
multiply the reported envelope by T'. **That claim is wrong and the patch is
reverted.** The number came from a probe that drove `q_set` to the DER rail
**once** and measured where the realised Q landed (36-50 % of the way). That
measures a *single-step* response, not capability.

It is not capability because `apply_qw_reset` reanchors the DER block of
`_u_current` onto the measured Q every DSO tick, while the input bound
`[q_min, q_max]` is absolute. So each tick the controller may command from the
*current realised* Q all the way to the rail, realise ~44 % of that gap,
reanchor, and close 44 % of the remainder next tick — geometric convergence to
the rail. The local Q(V) droop slows the approach; it does not cap it.

**CAIR as written is correct**: physical DER rail x open-loop `dQ_iface/dQ_DER`,
no closed-loop transform. Manuel's position throughout. Recorded here because
the wrong version was briefly committed to this log and because the
single-step-vs-multi-step distinction is easy to get wrong again.

## Established by measurement (all at 2016-01-05 08:00 unless noted)

### The local Q(V) droop is installed but INERT in closed loop
```
q_mode counts:        {'qv': 44}      -- every DER
qv_slope_pu:          0.06 (uniform)
net.controller rows:  44 -> {'QVLocalLoop': 44}
```
so it is present on every DER. But it never fires: measured over 150-172 min,
`Q_realised == q_set` exactly with a droop term of 0.000 Mvar on every DER at
every sample. `write_der_q_set` re-anchors `qv_vref_anchor_pu` at each DSO
apply, which holds `V - V_anchor` at ~5e-4 pu — inside `qv_deadband_pu` — so
the piecewise characteristic returns `Q = q_set` identically.

**Therefore T' = I in the operating regime**, and
`apply_qv_h_transform = False` is correct rather than an oversight: the DSO's
`H = dy/dQ_DER` is the right Jacobian for what it commands. This also means
`docs/daily_log/06_2026/2026-06-23_zigzag_investigation_v3v4v5.md`'s record of
T' as "a no-op, BIT-IDENTICAL" still holds — for the right reason.

An earlier probe here reported a 0.36-0.50 "reach fraction" and inferred a
2.3x actuator attenuation. That was an artefact of driving `q_set` to the rail
in **one shot**, which throws V far outside the deadband and activates a droop
that never fires under normal re-anchored stepping. Do not cite that number.

### The rank-1 SMW refresh after a commit is accurate
`receive_disturbance_message` freezes the cached operating point (no power
flow) while the plant moves ~20 Mvar at the interface, so it was a prime
suspect. It is clean. Cached `dQ_iface/dQ_DER` after SMW vs a full rebuild of
the reduced net at the post-switch converged state:

| DSO | after/truth | sign flips |
|---|---|---|
| DSO_1 | 1.013, 1.006, 1.004, 1.004 | 0/4 |
| DSO_2 | 1.014, 1.003, 1.001, 1.001 | 0/4 |
| DSO_3 | 1.012, 1.006, 1.005, 1.005 | 0/4 |
| DSO_4 | 1.016, 1.008, 1.007, 1.006 | 0/4 |

Worst case 1.6 %.

### The shunt feedforward moves the plant as intended
Per-commit, metered interface Q against the `dQ_itf` estimate (background
|delta| on non-commit steps: median 0.03, p95 0.17 Mvar):

| commit | est | setpoint delta | metered delta | gap opened |
|---|---|---|---|---|
| MSC bus 73 -> DSO_2 t3w3 | −23.31 | −23.85 | −18.74 | 5.11 |
| MSC bus 73 -> DSO_2 t3w3 | −24.92 | −25.23 | −19.64 | 5.59 |
| MSC bus 93 -> DSO_3 t3w6 | −20.47 | −20.78 | −21.33 | −0.55 |

Aggregate DER Q moves ≤ 4.4 Mvar at a commit, i.e. the DSO does **not**
counteract the switch — the feedforward achieves its stated purpose. The
per-commit error is DSO-specific: ~22 % over-statement on DSO_2's bank,
essentially exact on DSO_3's.

### Gap decomposition (6 h, `shunt_int_g_w=100`)
| trafo | final gap | at commits | drift between | rate |
|---|---|---|---|---|
| DSO_2 t3w3 | −32.18 | −10.70 | **−21.56** | −3.60/h |
| DSO_3 t3w6 | −8.23 | +0.55 | **−8.91** | −1.49/h |
| DSO_1, DSO_4 (no commits) | ≤ 0.74 | 0.00 | ≤ 1.39 | ≤ 0.23/h |

**Drift is ~67 % of the gap, appears only on DSOs that took a commit, and is
insensitive to everything tried.**

### `g_z_q_pcc` (Manuel's edit), controlled on that one variable
| | DSO_2 t3w3 | DSO_3 t3w6 |
|---|---|---|
| 1e-2 | −50.21 | −4.97 |
| 1e6 | −32.18 | −8.23 |

Helps the large gap by 36 %, hurts the small one. Drift unchanged
(−21.56 vs −21.76). Context: `g_z_voltage = 1e9`, so at 1e-2 the capability
slack was eleven orders of magnitude cheaper than the voltage slack and the
bound was effectively unenforced.

## Ruled out as the cause of the drift

CAIR formula; `dQ/dQ`; the SMW refresh; the `q_set` input bound; `g_q`
(Manuel tested — no effect); `g_z_q_pcc` (partial, drift untouched); sibling
feedforward compensation (measured: reduces the *at-commit* sibling gap
26-48 %, no effect on the offset transformer or the drift — reverted on
Manuel's instruction to keep the first-order update to the affected
transformer only).

## Open

* **NOT VERIFIED: that the fix preserves the DER-don't-jump property.** This is
  the entire reason to prefer the band shift over simply deleting the
  feedforward, and no run measures it — the gap traces cannot distinguish
  "DERs held steady" from "DERs moved and the TSO re-anchored", because the
  samples are ~0.5 h against a 180 s TSO period. The check is DER Q per DSO at
  commit-step resolution over the ~9 DSO ticks after a commit, in both arms.
* **Residual, ~13 Mvar.** With the shift, DSO_2's metered flow sits at −80.66
  against a reported rail of −93.6, i.e. the DSO leaves ~13 Mvar of its own
  band unused. Separate and smaller effect. Two of the three candidates are
  eliminated by measurement:

  - *DERs railed* — **no**. DSO_2's DER block sits at 41-77 % of span
    throughout (final −7.88 against rails [−151.8, +188.6]), i.e. ~144 Mvar of
    downward headroom against a 13 Mvar interface shortfall, and it moves
    62 Mvar over the last 30 ticks. Neither headroom nor immobility.
  - *Actuator attenuated by the local Q(V) droop* — **no**. Measured over the
    150-172 min window: `Q_realised == q_set` exactly, droop term 0.000 Mvar,
    zero DERs with an active contribution, including across the 159 min
    commit. `V - V_anchor ~ 5e-4 pu` — re-anchoring every DSO step keeps the
    deviation inside `qv_deadband_pu`, so the characteristic returns
    `Q = q_set` identically. **T' = I in the operating regime**, so
    `apply_qv_h_transform=False` is CORRECT and `H = dy/dQ_DER` is the right
    Jacobian. (The 0.36-0.50 reach fraction measured earlier is an artefact of
    a one-shot excursion to the rail, which drives V far outside the deadband
    and activates a droop that never fires in closed loop. That number should
    not be cited.)

  Remaining candidate: an equilibrium trade-off against the voltage objective
  (`dso_g_v = 1E5` vs `g_q = 250`). Untested — the earlier `g_q` test was run
  while the setpoint was still unreachable and is void. Next step is a `g_q`
  sweep on the fixed code.
* Two shunt-feedforward defects measured and left unfixed: the MSR estimate
  over-states by up to 36 % (a reactor depresses V and a constant-susceptance
  device then delivers `B·V²`, while the linear estimate is symmetric), and
  1-8 % of each step lands on sibling transformers whose setpoints are not
  compensated. Compensating the siblings was measured to reduce their
  at-commit gap 26-48 % but was reverted — first-order update stays on the
  affected transformer only (Manuel).
* The offset is cumulative and never reconciled against measurement, so the
  band shift inherits that: any error between the estimated `dQ_itf` and the
  true interface change (~22 % over-statement on DSO_2's bank, ~0 on DSO_3's)
  accumulates in the setpoint and the bound together. Consistent, but both
  drift from physics at the same rate.

## Method note

Two experiments in this investigation were invalidated by avoidable errors;
both are worth guarding against:

1. **Edit race.** An A/B pair was run by editing the source between two
   background launches. The first run had not yet finished importing when the
   edit landed, so both arms ran the *same* (new) code and returned bit-
   identical results. Gate behavioural A/B on a runtime flag, never on file
   state.
2. **Single-step probe read as a steady-state property.** See Retraction.
