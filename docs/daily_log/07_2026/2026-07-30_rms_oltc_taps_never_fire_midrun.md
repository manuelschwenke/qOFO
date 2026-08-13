# 2026-07-30 — RMS OLTC taps: never fire mid-run, and inverted in sign

**What:** Two independent defects in the PowerFactory plant's OLTC path, found
after the converged-seam fix cleared the noise floor that had hidden them.
**Why:** Gate-E run 0095 commanded a −1 tap that the RMS plant did not respond
to (0.04 mpu across the event instant, against −3.93 mpu on the static plant).
**Timestamp:** 2026-07-30, ~18:30–19:30.
**Status:** defect A open. Defect B narrowed to a **one-tap-per-element-per-
calculation** limit and PARTIALLY fixed (`pf/screening.py::add_tap_event` no
longer pools tap events); a general mid-run tap mechanism is still missing.

---

## Defect A — the tap sign convention is inverted

Same tap step, both models, measured directly on the same elements:

| element | pandapower `tap_pos = −1` | PowerFactory `EvtTap ntap = −1` |
|---|---:|---:|
| machine `MT_g0_t0` / `trafo 0`, HV bus 1 | **−8.02 mpu** | **+6.02 mpu** |
| network `NT_t3` / `trafo 3`, HV bus 11 | **−5.76 mpu** | **+5.41 mpu** |
| coupler `NC3W_DSO_1_t0` / `trafo3w 0`, tertiary | **+10.24 mpu** | **−9.08 mpu** |

Opposite sign on all three transformer classes; magnitudes agree to within
what RMS exciter dynamics versus a static gen holding LV exactly would explain.

`pf/plant.py::_dispatch_taps` computes `step = 1 if delta > 0 else -1` from the
pandapower tap delta and passes it straight to `EvtTap.ntap`, assuming the two
share a sign convention. They do not.

## Defect B — CORRECTED SCOPE: only `ElmTr2` taps are dead mid-run

> **Retraction.** This section first concluded "an EvtTap armed mid-calculation
> never takes effect", generalised from `ElmTr2` alone. That is **wrong**.
> Mid-run `EvtTap` was then tested on the other two classes that use the same
> code path:
>
> | element class | actuator | mid-run EvtTap |
> |---|---|---|
> | `ElmTr3` (DSO coupler OLTC) | Layer 2 | **works**, MV step −8.16 mpu (pre-run: −9.08) |
> | `ElmShnt` (MSC/MSR) | TSO tertiary | **works**, bus step +37.1 mpu |
> | `ElmTr2` (TSO machine + network OLTC) | Layer 1 | **dead**, 0.03 mpu |
>
> So the DSO coupler taps and the tertiary shunts — the Layer-2 actuator and
> the TSO's switched compensation — have been reaching the RMS plant all along.
> Only the TSO 2W OLTCs are affected. Run 0095 is consistent: the tap that
> failed was `z1[1]`, a TSO zone OLTC, and no coupler tap ever diverged.

### The original (over-generalised) evidence, on `ElmTr2`

The dominant one. Reproduced three ways on `MT_g0_t0` (HV terminal `TN_bus1`,
where a real tap moves the voltage ~6 mpu):

| arming context | slot | barrier | response |
|---|---|---|---|
| right after ComInc, before the first `simulate()` | fresh | — | **+6.02 mpu** |
| right after ComInc, before the first `simulate()` | pooled | — | **+6.02 mpu** |
| after 3 `simulate()` calls (t = 65 s) | fresh | none | +0.033 mpu |
| after 7 `simulate()` calls (t = 145 s) | fresh | none | +0.031 mpu |
| after 3 `simulate()` calls (t = 65 s) | pooled | **2 admission barriers** | +0.037 mpu |
| after 5 `simulate()` calls (t = 105 s) | pooled | **2 admission barriers** | +0.037 mpu |

The discriminator is **when** the event is armed, not slot provenance and not
admission:

* pooled and fresh behave identically in both contexts, so the persistent pool
  is not implicated;
* forcing `admit_new_events` (the mechanism that makes mid-run `EvtParam` work)
  does **not** help — 2 barriers ran and the tap still did not move.

This is the same shape as the documented `EvtParam`-on-tap finding ("taps are
read at init only — the zero-response finding of run 083416"): the tap position
appears to be consumed at initialisation, and `EvtTap` is honoured only while
the event list is still being read, i.e. before the first `ComSim.Execute`.

`pf/probe_tap_avr.py` verified `EvtTap` on `ElmTr3` and `ElmShnt` — but only in
the arm-then-run configuration, which is the one that works. The mid-run case
was never covered, and `ElmTr2` was never covered at all.

### Consequence for every RMS replay run to date

**Only a TSO 2W OLTC tap commanded in the very first dispatch interval can
reach the RMS plant. Every later `ElmTr2` tap command is silently ignored.**
Coupler (`ElmTr3`) and shunt (`ElmShnt`) commands are unaffected. `_dispatch_taps` writes
`net.trafo.at[idx, "tap_pos"] = target` into the mirror net regardless, so:

* the runner records, and `analysis/gate_e_diagnostics.py` plots, the
  **commanded** tap, not the plant's;
* every "RMS tap = X" in an actuator-divergence table is a command;
* apparent tap *agreement* between the plants can mean the RMS simply never
  moved.

Run 0095's zone-1 behaviour is exactly this: both loops commanded `z1[1] → −1`
at t = 180 s, the static plant moved −3.93 mpu, the RMS plant moved +0.25 mpu,
and the +4.08 mpu zone-1 error that followed is ≈ `0 − (−3.93)` — the signature
of the RMS doing nothing, not of an inverted tap (which would have given ≈ +7.9).

A scan of all 30 monitored TN buses at the event instant confirms it: the
largest step anywhere is 0.32 mpu, against the ~6 mpu a real tap produces.

## Why this surfaced only now

It needs a *same-tap* comparison. Until the converged-seam fix
(`2026-07-30_gate_e_snapshot_not_qv_fixed_point.md`) removed the Q(V) anchor
residual, the two loops never commanded the same tap at the same time — the
discrete decisions always diverged first, and that divergence was attributed to
the controllers. With the seed removed, both loops made identical decisions and
the plant-level difference became visible.

## Remedies — for discussion, none implemented

1. **Re-initialise on every tap** (ComInc) — correct but destroys the dynamic
   state and the whole point of a continuous RMS run.
2. **Represent the OLTC as a controllable element** (station controller / DSL
   tap input) that PF honours mid-run, instead of an event.
3. **Find an event class/attribute PF does honour** for transformer taps
   mid-run; unknown whether one exists.
4. **Accept taps as initialisation-only** and restrict Gate-E claims to the
   continuous actuators, documenting the limitation prominently.

Defect A must be fixed too, but on its own it would be a silent no-op for every
interval after the first, so B is the gating item. Fixing A alone would also be
actively dangerous: taps would then fire with the correct sign only in the one
context that works, making the inconsistency operating-point dependent.

## Risks / unresolved

- **Scope of invalidation not yet assessed.** Every archived Gate-E replay that
  involved a tap after interval 1 has an RMS plant that did not follow its own
  controller. How much that changes each conclusion is unquantified.
- Whether `ElmShnt` step events (MSC/MSR, same `EvtTap` class) suffer the same
  mid-run failure is **untested** and uses the identical code path. The tertiary
  shunts sat at step 0 in every run examined today, so it has not shown up.
- Defect A's magnitudes were compared across two different solution regimes
  (RMS with exciter dynamics vs static PF with a gen holding LV). The sign is
  unambiguous; the ~25 % magnitude difference is not separately explained.
- The pandapower-side derivative was measured on the unprofiled build state;
  the sign does not depend on the operating point, but the magnitude does.

## Files

- No source changes.
- Scratchpad probes (not committed): `probe_tr2_tapchanger.py` (tap-changer
  configuration audit — `itapch=1`, HV side, 1.25 %, ±9, i.e. the PF model is
  correct), `probe_tr2_tap_rms.py` (ElmTr2 vs ElmTr3 response),
  `probe_tap_pool_vs_fresh.py`, `probe_tap_midrun.py`.


---

# Addendum (same day, later): defect B narrowed and partially fixed

## What now works

`add_tap_event` no longer routes `EvtTap` through the persistent pool. It
creates a fresh event and lets `admit_new_events` admit it. With that, **the
first tap on a given element fires mid-run**, which it never did before:

| context | before | after |
|---|---|---|
| tap on `MT_g0_t0` mid-run | 0.03 mpu | **+6.20 mpu** |

`_track_persistent_arm` is also skipped for taps: they are one-shot objects,
and retiring them writes `time` back onto an event PF has already fired.

## What still does not work

**A second tap on the SAME element in the same calculation does not fire.**
Four taps on `MT_g0_t0` at 40 s spacing, empty event folder, no pooling:

| tap | 1st | 2nd | 3rd | 4th |
|---|---:|---:|---:|---:|
| step [mpu] | **+6.20** | 0.03 | 0.02 | 0.06 |

Taps on *different* elements each fire once — `NT_t3` then `MT_g0_t0` both
moved (+5.70, +6.14), and `NC3W_DSO_1_t0` then `SH_MSC_DSO_1_s0` both moved
(−8.16, +37.1). The consistent reading is **one honoured tap per element per
RMS calculation**, with one unexplained exception (a first `NT_t3` tap that
followed two dead `MT` events was itself dead, suggesting an unfired event
poisons later ones).

## Hypotheses tested and refuted

| hypothesis | verdict |
|---|---|
| mid-run arming is impossible for taps | refuted — first tap fires mid-run |
| `ElmTr2` is special (vs `ElmTr3` / `ElmShnt`) | refuted — all three behave alike |
| the persistent pool is at fault | partially — pooling blocks it, but removing pooling does not give repeat taps |
| `_retire_fired_events` corrupts later events | refuted — skipping it changed nothing |
| admission barriers are missing | refuted — forcing them does not help |
| event offset (`TAP_MECH_DELAY_S` = 5 s) outside the 0.5 s barrier horizon | refuted — 1.0 s behaves identically |
| leftover inert pool events in the folder | refuted — empty folder behaves identically |
| `EvtTap` has an absolute/relative mode | refuted — attributes are only `ntap`, `time`, `p_target`, `outserv` |

## Where to go next

1. **DIgSILENT documentation / support** on repeated `EvtTap` on one element
   during a single RMS calculation. The black-box search is exhausted.
2. **Re-`ComInc` per tap** — the first tap always fires, so re-initialising
   restores the capability. Costs the dynamic state; may be acceptable given
   Gate D's 20 s timescale separation, but that is a methodology change.
3. **Controllable ratio element** — represent the OLTC by something PF varies
   continuously (DSL-driven ratio), the robust answer and the most work.

**The partial fix must not be read as "taps now work."** A replay still
silently drops every tap after the first on each transformer, and
`_dispatch_taps` still writes the commanded position into the mirror net, so
records continue to show taps that the plant did not take.


---

# Addendum 2 — root cause found in the DIgSILENT documentation

## Defect A: solved and FIXED — it was never a sign convention

`EvtTap` carries a **Tap Action** attribute `i_tap`; `ntap` alone is ignored.
User Manual 13.9.15 ("The Tap Action can then be specified") and the DPL
example in the scripting chapter:

```
event(enable, trigger, 'create=EvtTap target=TransformerSlot
      name=TapEvent_Decrease i_tap=1')
```
"...will decrease the tap position, because the parameter `i_tap` is set to 1."

`i_tap` defaults to **0 = increase**. `add_tap_event` set only `ntap`, so
**every tap ever dispatched to the RMS plant was an increase, regardless of the
commanded direction** — an up-tap and a down-tap were indistinguishable to the
plant. Measured at `MT_g0_t0`'s HV terminal:

| | response |
|---|---:|
| `i_tap=0` (what was being sent) | +6.02 mpu |
| `i_tap=1` | **−6.26 mpu** |
| pandapower `tap_pos=-1` (reference) | −8.02 mpu |

**Fixed** in `pf/screening.py::add_tap_event`: `i_tap = 0 if ntap > 0 else 1`.
Verified through the real code path in the plant's own configuration —
commanding −1 now moves `c:nntap` from 0 to −1 at the scheduled instant.

There is no sign inversion between pandapower and PowerFactory. Addendum 1's
"opposite sign on all three transformer classes" was an artefact of always
sending an increase; retract it.

## Defect B: `EvtTap` is not the supported RMS mechanism

`TechRef_2-W-Transformer_3Phase.pdf`, §5 *RMS-Simulation*:

> "The model used by the RMS simulation is identical to the load flow model.
> However, **tap controller definitions are not considered. For the simulation
> of tap controllers, a separate dynamic model must be defined that can be
> interfaced with the transformer using the input variable `nntapin`
> (tap-input).**"

That explains every symptom. Driving taps by `EvtTap` during an RMS run is
outside the supported path, and its behaviour there is accordingly erratic:

| commanded | landed |
|---|---|
| t = 25 s | 25.03 s ✓ |
| t = 65 s | **125.00 s** (+60) |
| t = 105 s | **165.01 s** (+60) |
| t = 145 s | never (run ended at 180 s) |

The +60 s deferral is reproducible and **independent of** event offset
(0.1 s / 1 s / 5 s), pooling, pre-ComInc existence, admission barriers, and
event-folder contents. Deactivating a fired event (`outserv=1`) makes it worse:
only the first tap then applies at all.

`c:nntap` is the correct observable and should be monitored in every future tap
test — inferring tap state from bus voltage is what produced two wrong
conclusions in this session.

## The remedy is now documented, not speculative

Per the TechRef: give each controllable transformer a **DSL model driving
`nntapin`** through a composite frame, then command it mid-run with
`EvtParam` on that model's parameter — the mechanism already validated for
`REEC_D.Qext` and the AVR `usetp`, and the one the persistent pool and
admission barriers were built and proven for.

`nntapin` is a DSL *signal*, not a data attribute (`GetAttribute('nntapin')`
raises on both `ElmTr2` and `ElmTr3`), so it is reachable only via composite
wiring — the same shape as the existing `QVPRE` layer in
`pf/wecc_apply.py`, which is the natural template.

Work outline: one `BlkDef` frame (transformer slot + tap-controller slot), one
`ElmDsl` per controllable transformer holding the commanded position, an
`ElmComp` per transformer wiring the two, and a `_dispatch_taps` that writes the
DSL parameter instead of creating an `EvtTap`. 12 couplers + 8 two-winding
units = 20 composites.

## Status

* Direction: **fixed and verified**.
* Timing: **root cause identified**, remedy specified, **not implemented** —
  it is a model-building change on the scale of the QVPRE rollout and needs to
  be scoped deliberately.
* Until then, a replay's second and later taps on any transformer land tens of
  seconds late or not at all, and `_dispatch_taps` still records the commanded
  position in the mirror net.


---

# Addendum 3 — DSL tap-control layer built and proven, one manual step left

`pf/tap_ctrl.py` (new) implements the vendor-documented mechanism. Everything
scriptable is done and verified; one wire must be drawn once in the GUI.

## Verified on `MT_g0_t0` (Tmech = 5 s, pooled EvtParam)

| commanded | `nntapin` |
|---|---|
| −1 @ 25 s | −0.28 @27, −0.95 @40, **−1.000 @64** |
| −2 @ 65 s | −1.28 @67, −1.95 @80, **−2.000 @104** |
| 0 @ 105 s | −1.45 @107, −0.11 @120, **−0.002 @139** |

On time, both directions, repeatedly — exactly what `EvtTap` cannot do. Note
the dispatch path matters: a **fresh** `EvtParam` created mid-run showed the
same ~60 s lag as `EvtTap`, while a **pooled** slot pre-created before ComInc
lands on time. That is the design rationale the pool was built on, now
confirmed to apply to this layer too.

## Two PF requirements discovered the hard way

1. **The output needs its own `inc()`.** `inc(x1)=ntapcmd` plus `nntapin=x1` is
   not enough — ComInc fails with *"Output 'nntapin' not initialised"*. Adding
   `inc(nntapin)=ntapcmd` fixes it.
2. **A frame must declare its internal signals** in `sIntern`, or ComInc
   rejects it. The working WECC frame carries every `BlkSig` name there.

## The remaining manual step

The frame's *connection topology* lives in its `IntGrfnet` graphic, which the
API cannot author — the same reason `wecc_apply._project_blockdef` refuses to
create frames. `ensure_frame` creates the two slots and the signal; the wire
must be drawn once:

1. open `Frame Tap Control qOFO` in User Defined Models,
2. connect `Tap Control`.`nntapin` (output) to `Transformer`.`nntapin` (input),
3. save.

**Symptom while the wire is missing** — and it is silent, which is why it is
called out here: ComInc passes, the DSL runs, `s:nntapin` follows the command
exactly, and the transformer ratio and bus voltage do not move at all
(measured: 0.3 mpu for a commanded 2-step tap, against ~12.5 mpu for a real
one). `tap_ctrl.frame_is_wired()` detects this state.

## Still to do after the wire exists

* roll out to all 20 controllable transformers (8 `ElmTr2` + 12 `ElmTr3`
  couplers) — `controllable_transformers()` enumerates them;
* verify the `ElmTr3` couplers accept `nntapin` the same way (untested; the
  3W tap attribute is `n3tap_h`, and the input signal name may differ);
* switch `PowerFactoryPlant._dispatch_taps` from `add_tap_event` to a pooled
  `EvtParam` on the composite's `ntapcmd`, and pre-allocate those slots in
  `_preallocate_event_slots`;
* decide `Tmech`: 5 s models the mechanical delay; the static plant taps
  instantaneously, so a smaller value makes the two plants more comparable at
  the 20 s interval endpoint.

The `i_tap` fix in `add_tap_event` stays regardless — it is correct, and it
keeps the event path usable for the first tap and for shunt steps.


---

# Addendum 4 — WORKING for 2-winding OLTCs

The frame was hand-authored in the GUI (New Object -> **Composite model frame**,
then two slots and one wire). The API route used earlier produced a `BlkDef`
with no `IntGrfnet`, which is why its Diagram button did nothing and the signal
never reached the transformer.

## Frame as built

| slot | seq | class | signal | "Local, stored inside" |
|---|---|---|---|---|
| `Transformer` | 0 | `ElmTr2` | **input** `nntapin` | unticked |
| `Tap Control` | 1 | `ElmDsl` / `TAPCTRL_qOFO` | **output** `nntapin` | ticked |

`Type` is left empty on both — a network-element slot has no BlkDef, and the
working WECC frame likewise has `typ_id = None` on all nine of its slots.

## Verified end to end on `MT_g0_t0`

Commanded through pooled `EvtParam` on `ntapcmd` (Tmech = 5 s):

| t [s] | `nntapin` | `c:nntap` | ΔV [mpu] |
|---|---:|---:|---:|
| 0 | 0.000 | 0.00 | 0.00 |
| 30 | −0.605 | −0.60 | −4.44 |
| 64 | −1.000 | **−1.00** | **−6.58** |
| 80 | −1.946 | −1.95 | −11.58 |
| 104 | −2.000 | **−2.00** | **−11.77** |
| 139 | −0.002 | −0.00 | +0.24 |

On time, both directions, repeatedly, with the correct sign and magnitude
(−6.58 mpu/step against pandapower's −8.02 and the single-shot `EvtTap`'s
−6.26; two steps give −11.8 mpu, i.e. linear). **This is the mechanism the
replay needs**, and it is what `EvtTap` could never deliver.

## Open: the 3-winding couplers

`TechRef_3-W-Transformer_3Phase.pdf` Table 5.1 lists **three** tap inputs —
"Tap position (HV/MV/LV), controller input" — all printed as plain `nntapin`,
the winding qualifier lost in the table layout.

Binding an `ElmTr3` into the 2W frame **compiles and initialises without any
warning, and does nothing**: commanding −2 moved the MV bus by +0.96 mpu over
100 s (drift), against the ~20 mpu a real 2-step tap gives. So the plain
`nntapin` declaration reaches none of the three windings, and PF reports
nothing — the same silent-failure shape as the missing wire.

Unresolved: the exact HV-winding input name (`nntapin_h`? positional order?).
Candidates can be tested without further GUI work by copying the working frame
with `AddCopy` (which carries the `IntGrfnet`), then editing the slot's
`sInput` and the matching `BlkSig` name — the couplers are 12 of the 20
controllable transformers, so this matters.

## Remaining integration work

* resolve the 3W signal name, then a second frame for `ElmTr3`;
* roll out composites to all 20 transformers;
* switch `PowerFactoryPlant._dispatch_taps` from `add_tap_event` to a pooled
  `EvtParam` on `ntapcmd`, and pre-allocate those slots in
  `_preallocate_event_slots`;
* choose `Tmech` (5 s models the mechanical delay; the static plant taps
  instantaneously, so a smaller value is more comparable at the 20 s endpoint);
* re-run Gate E and re-assess every multi-tap conclusion.


## Addendum 4b — 3W coupler signal-name sweep: all five candidates fail

`AddCopy` on the working frame carries the `IntGrfnet`, so candidates can be
tried without GUI work.  Renaming a slot's signal plus the matching `BlkSig`
is **safe**: the copy renamed back to `nntapin` reproduces the original's
-11.83 mpu on the 2W exactly, so the wire survives renaming and the sweep
below is valid.

Coupler `NC3W_DSO_1_t0`, commanding `n3tap_h` -> -2, measured at MV bus
`DSO_1_bus43` (a real 2-step tap is ~20 mpu):

| slot input signal | dV [mpu] |
|---|---:|
| `nntapin` | +0.59 |
| `nntapin_h` | +0.59 |
| `nntapinh` | +0.59 |
| `n3tapin_h` | +0.59 |
| `nntapin_hv` | +0.59 |

Identical to two decimals across all five - i.e. baseline drift, nothing
connected, and PF issues no warning in any case.  The same frame and the same
`nntapin` name drives an `ElmTr2` at -11.83 mpu, so this is specific to
`ElmTr3`.

**Still open.** Untried ideas, in order of promise:

1. declare **all three** winding inputs on the slot as one comma-joined string
   (`nntapin_h,nntapin_m,nntapin_l` or whatever the true names are) and wire
   only the HV one - PF's convention for multi-signal slots, cf. the WECC
   `Generator` slot's `['u1r_in,u1i_in,id_ref,iq_ref']`.  Changing the port
   count probably needs the wire redrawn in the GUI;
2. read the true names off **Figure 5.1** of
   `TechRef_3-W-Transformer_3Phase.pdf` (an image; `pdftotext` lost the
   subscripts that Table 5.2's `I0rDelta h/m/l` shows are there);
3. check whether the 3W tap input requires the tap changer to be declared on
   that winding in the *type* (`TypTr3`), analogous to `TypTr2.itapch`.

Until this is resolved the DSL layer covers the 8 two-winding OLTCs but not
the 12 DSO couplers, i.e. not the Layer-2 actuator.


## Addendum 4c — SOLVED for both transformer classes

The 4b sweep was **invalid** and its conclusion is retracted. It renamed the
signal on *both* slots plus the `BlkSig`, so the `Tap Control` slot declared an
output the bound DSL does not emit (`TAPCTRL_qOFO.sOutput` is always
`nntapin`) -- nothing was produced, which is why all five candidates returned
an identical no-effect. Only the **transformer** side may vary.

Corrected sweep on `NC3W_DSO_1_t0` (DSL output left as `nntapin`, only the
transformer slot's `sInput` varied), commanding `n3tap_h` -> -2:

| transformer slot input | dV at MV [mpu] |
|---|---:|
| **`nntapin_h`** | **+15.29 — MOVES** |
| `nntapin h` | +0.59 |
| `nntapin_m` | +0.59 |
| `nntapin_l` | +0.59 |

`_m` / `_l` are accepted and silently do nothing, which is correct: our
couplers carry the tap changer on the HV winding only. The sign matches
pandapower (tap -1 -> MV **+**9.84 mpu there).

The names are `nntapin_h` / `nntapin_m` / `nntapin_l` with **underscores** --
legible in `TechRef_3-W-Transformer_1Phase.pdf`, where the 3-Phase edition's
Table 5.1 loses the subscripts. Same variables for both editions.

### Final configuration

| class | frame | transformer slot input | DSL output |
|---|---|---|---|
| `ElmTr2` | `Frame Tap Control qOFO` | `nntapin` | `nntapin` |
| `ElmTr3` | `Frame Tap Control 3W qOFO` | `nntapin_h` | `nntapin` |

The 3W frame was produced with `AddCopy` of the hand-drawn 2W frame (which
carries the `IntGrfnet`) and one attribute edit -- no second GUI session
needed. Renaming a slot signal is safe: the copy renamed back to `nntapin`
reproduced the original's -11.83 mpu exactly.

### Rollout verified

24 composites built (12 `ElmTr2` + 12 `ElmTr3`); ComInc passes with all of
them present. Commanding -2 on one of each:

| | dV |
|---|---:|
| 2W `MT_g0_t0` | **-11.99 mpu**, `c:nntap` -> -2.00 |
| 3W `NC3W_DSO_1_t0` | **+13.22 mpu** |

Note `controllable_transformers()` over-selects: it takes every `MT_`/`NT_`
`ElmTr2` (12) where the snapshot's actuator list names only 8
(`machine_oltc_trafo_indices` 6 + `network_oltc_trafo_indices` 2). The four
extra composites are inert -- they hold the tap at its load-flow value -- but
the selector should be narrowed to the actuator set.

### Remaining

* switch `PowerFactoryPlant._dispatch_taps` from `add_tap_event` to a pooled
  `EvtParam` on the composite's `ntapcmd`, and pre-allocate those slots in
  `_preallocate_event_slots`;
* narrow `controllable_transformers()` to the snapshot actuator list;
* choose `Tmech` (5 s = mechanical delay; the static plant taps instantly);
* re-run Gate E and revisit every multi-tap conclusion.
