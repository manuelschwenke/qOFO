# Derivation of the reserve trigger margin `k`

Date: 21 October 2026. Status: derivation only, nothing implemented.

Design decisions taken by the author on 21 October: report the **binding unit**
rather than a fleet mean; extend the upward report with **directional DER
margins in Mvar** (option C) when the interface is changed; **start with option
A**, which needs no interface change; and derive the threshold `k` from the
recordings rather than choosing it.

## Option A, restated

The reserve trigger uses the DSO-reported interface capability band one-sidedly
instead of as a centred interval. With the absolute band `[lo, hi]`, that is
`q_iface_now` plus the reported delta, and the dispatched request `q_set`:

    margin_up = hi − q_set,    margin_dn = q_set − lo,
    binding margin = min(margin_up, margin_dn),

and the trigger fires when the binding margin falls below `k · q_step_mvar`.
The threshold is therefore expressed in bank steps, a scale the actuator itself
defines, instead of the dimensionless `shunt_sts_reserve_start = 0.8` on a
normalised band position. Both quantities are already recorded as
`dso_trafo_q_cap_min_mvar`, `dso_trafo_q_cap_max_mvar` and
`dso_trafo_q_set_mvar`, so the derivation needs no new run.

## Distribution

Source: `results/shunt_hybrid_diag_20261020/allow_default/slow`, the held-bank
baseline of the 20 October diagnostic hour. 12 interfaces, 21 TS decision
instants, 252 TS-interface-samples. `q_step_mvar = 25`.

Binding margin, in bank steps:

| min | p5 | p25 | median | max |
|---:|---:|---:|---:|---:|
| 0.725 | 1.023 | 1.331 | 1.845 | 2.392 |

TS-interface-samples below the threshold, and the interfaces they belong to:

| `k` | samples | interfaces |
|---:|---:|---|
| 0.50 | 0 | – |
| 0.75 | 1 | DSO_2 ×1 |
| **1.00** | **9** | **DSO_2 ×6, DSO_4 ×3** |
| 1.25 | 39 | DSO_1 ×6, DSO_2 ×12, DSO_4 ×21 |
| 1.50 | 85 | DSO_1 ×21, DSO_1 ×2, DSO_2 ×21, DSO_3 ×20, DSO_4 ×21 |

**`margin_up` is the binding direction at every one of the 252 samples**
(its minimum is 0.725 against 1.624 for `margin_dn`). The one-sided form is
therefore not a refinement but a necessity: a centred measure averages a
systematically tight direction with a systematically loose one.

## Proposed value

**`k = 1.0`.** It fires on 9 of 252 TS-interface-samples, 3.6%, on the two
tightest interfaces, which is frequent enough to confirm a 360 s persistence
and far from permanent. It also has a physical reading that the empirical
choice and the mechanism agree on: *the subordinate system has less than one
full bank step of headroom left in the binding direction*, which is exactly the
condition under which a bank step can take that load over without pushing the
delegated request through the rail.

`k ≤ 0.5` never fires in this hour. `k ≥ 1.25` makes at least one interface
(DSO_4) demand a step at **21 of 21** instants, that is permanently, which is
the ratchet failure mode already diagnosed for the regular path: a
permanently-true trigger requests a step at every opportunity until dwell or
budget stops it. The usable window is thus narrow and `k = 1.0` sits inside it
with a physical anchor, unlike `g_w_tso_shunt`, whose value had to be placed in
an empirical gap.

## Why the present proxy never fired

With `r = (q_requested − m)/b` on the centred band, `margin_up = b(1 − r)`, so
`r > 0.8` requires `margin_up < 0.2 b`. The observed band half-width is 1.42 to
2.57 bank steps, so the trigger needed a margin below roughly 0.28 bank steps,
about 7 Mvar. The margin never fell below 0.725 bank steps, 18.1 Mvar — a
factor of 2.5. This is the quantitative reason the `sts_reserve` memory has
never become non-zero in any recorded run, and it also shows the structural
defect of the normalised form: its threshold in Mvar moves with the band, so a
wide band makes the same absolute margin look safe.

## Finding that bears on the whole reserve concept

Over the full 20 s record granularity, 2160 interface-samples per run:

| Run | Commits | Min binding margin [steps] | Samples < 1 step |
|---|---:|---:|---:|
| `legacy`, banks held | 0 | 0.725 | 85 |
| `regular`, `g_w = 5` | 10 objective | **−0.037** | 148 |
| `regular`, `g_w = 20` | 3 objective | 0.204 | 92 |
| `exchange0`, slow gate | 1 tracking | 0.725 | 85 |

An earlier reading of this table as "bank engagement tightens the margin" was
too coarse: the two reference conventions behave differently and must be
separated. Tracing the records around individual commits shows the following.

**Objective commit** (`g_w = 20`, 720 s, DSO_1, MSC). The dispatched setpoints
move with the bank: `trafo_1` from −6.6 to −26.6 Mvar, `trafo_0` from −4.1 to
−6.8, `trafo_2` from −7.8 to −11.3, together about one 25 Mvar step. The
measured flow follows, −6.3 to −25.9 on `trafo_1`, and the reported band shifts
down with it, (−86.9, 45.9) to (−110.2, 24.5). The margin is therefore almost
unchanged: 52.5 before, 51.1 after. Aggregate DSO_1 DER Q goes from −18.9 to a
settled −25 against a pre-commit drift already in progress, so the DER are
**not relieved**; the offset convention tells the subordinate controller not to
fight the bank, and its own production stays where it was.

**Tracking support commit** (`exchange0`, 2880 s, DSO_4, MSC). The dispatched
setpoints are invariant by construction — `offset += delta_q` and
`delegated -= delta_q` — and the records confirm it: `trafo_10` stays at
−4.0 Mvar throughout. The measured flow jumps from +7.8 to −16.1, so the
subordinate controller now sees a 12 Mvar error against an unchanged target and
**works against the bank**: over the following six minutes the flow recovers to
−13.8 while aggregate DER Q moves from −47.9 to −50.9. The band's anchor
dropped with the measured flow while the setpoint did not, so this interface's
`margin_up` fell from 51.7 to 28.5 Mvar. Asymptotically the subordinate loop
cancels the bank, having spent DER headroom to do it.

So neither convention relieves the DER, and the support transfer is actively
counterproductive for reserve: it holds the target fixed and therefore requires
the DER to undo the bank's contribution. **Reserve restoration through bank
switching is not demonstrated on this network, and the mechanism by which it
could work is not yet established.** The trigger derived above decides when to
ask; it says nothing about whether asking helps.

Which variant would actually unload the DER depends on where the bank's
reactive power enters relative to the metered HV port of the three-winding
interface transformer, and on the paired offset and `q_adj` logic in the
runner. Both commits moved the measured HV-side flow by about a full bank step,
so the tertiary injection reaches that port, but the division between
"substitutes for DER production" and "appears as a flow the DER must track"
has not been read out of the model. That reading, plus a targeted experiment
that holds everything else fixed and toggles the convention while measuring
aggregate DER Q, is a precondition for the reserve path to mean anything.

Separately, the `g_w = 5` run reaches a binding margin of −0.037 bank steps,
about 0.9 Mvar: the dispatched setpoint fell marginally **outside** the reported
band.

Separately, the `g_w = 5` run reaches a binding margin of −0.037 bank steps,
about 0.9 Mvar: the dispatched setpoint fell marginally **outside** the reported
band. The paired capability-shift and `q_adj` logic exists to prevent exactly
that, so this is either measurement noise and voltage rescaling or a residual in
that pairing, and it should be checked before the reserve path relies on the
band as a bound.

## What should be reported, revised

The author's objection to a per-DER maximum is accepted. The reserve that
matters to this controller is **interface-referred and deliverable**: the TS
delegates interface reactive power, so the relevant question is how much more
the subordinate system can still deliver at that interface, not whether some
individual park is near its own rail. A single small exhausted park is a plant
observation, not a loss of controllable reserve.

That collapses the earlier option list. Deliverability requires the sensitivity
weighting `Σ_j s_ij (q_bound_j − q_j)`, which is exactly what
`CapabilityMessage` already reports. Option B, the fleet-maximum utilisation,
is therefore not a trigger. Option C as written, an unweighted sum of DER
margins, is also not the right quantity, because an unweighted margin is not
deliverable at the interface. **Option A is the correct signal**, and the
`CapabilityMessage` needs no extension for the trigger: what has to change is
only that the reported band be used one-sidedly and thresholded in Mvar rather
than as a normalised position.

The per-DER distribution keeps one legitimate use, which is not a trigger: the
reported band is linear and presumes every DER can be driven to its rail with
fixed sensitivities. As individual units saturate, the linear sum over-states
what is still deliverable. A validity indicator — how much of the reported band
is backed by DER already close to their rails — would therefore qualify the
band rather than replace it, and is worth considering only after the mechanism
question above is settled.

## Reproduction

The quantities are recorded, so this is an offline analysis over
`dso_trafo_q_set_mvar` against `dso_trafo_q_cap_min_mvar` /
`dso_trafo_q_cap_max_mvar`, restricted to records whose
`zone_tso_shunt_hybrid` is populated, which marks the TS decision instants.
`tuning/shunt_weight_calibration.py` is the model for turning this into a
committed procedure over several campaign windows, which has not been done:
the numbers above are one hour of one scenario.
