# 2026-07-31 — DSL tap control integrated: RMS now tracks the QSS trajectory

**What:** Wired `PowerFactoryPlant` to the DSL tap-control layer and re-ran
Gate E. **Why:** taps were the last known plant defect — `EvtTap` could not move
a tap reliably mid-run, so the RMS plant never followed its controller's OLTC
decisions. **Status:** done, Gate E PASS, matched-pair improvement measured.

---

## Changes

* `pf/screening.py::add_tap_event` — sets `i_tap` (the Tap Action; `ntap` alone
  is ignored, default 0 = increase). Retained for shunt steps.
* `pf/plant.py::_dispatch_taps` — no longer fires `EvtTap`. Commands the
  **absolute** tap position as `EvtParam` on the `TAPCTRL` DSL's `ntapcmd`,
  on the pooled path that lands on time. Raises if a transformer has no
  `TAPC_` composite instead of silently dropping the command.
* `pf/plant.py::_resolve_handles` — resolves `TAPC_<transformer>` composites to
  their `TAPCTRL` `ElmDsl`.
* `pf/plant.py::_preallocate_event_slots` — reserves `ntapcmd` slots before
  ComInc, seeded with the current tap so an unused slot cannot yank it to zero.
* `pf/tap_ctrl.py` — `frame_for()`, per-class frames and `TAP_INPUT_SIGNAL`.

`Tmech = 5 s` kept: it reproduces the mechanical delay the event path modelled,
and the tap is 98 % settled at the 20 s interval endpoint, so it costs almost
nothing in static-vs-RMS comparability.

## Result — run 0096 vs 0095, matched pair

Identical in every respect (1080 s, `rural_700`, DSO_3 x2, 6 TSO periods)
except that taps now reach the plant:

| | interface Q rmse | max | zone V rmse | max |
|---|---:|---:|---:|---:|
| 0095, taps dead | 0.4638 | 0.9079 | 0.0021 | 0.0041 |
| **0096, taps working** | **0.1295** | **0.3947** | **0.0002** | **0.0004** |

**3.6x better on interface Q, 10x on zone voltage.** No actuator divergences.

The zone-1 step that 0095 could not remove is gone:

| t [s] | 180 | 360 | 540 | 720 | 900 | 1080 |
|---|---:|---:|---:|---:|---:|---:|
| 0095 zone 1 [mpu] | 4.08 | 3.89 | 3.79 | 3.71 | 3.64 | 3.44 |
| **0096 zone 1 [mpu]** | **0.28** | **0.11** | **0.12** | **0.11** | **0.10** | **-0.06** |

All three zones now hold within ~0.3 mpu for the whole horizon, and zone 1 no
longer diverges after the first TSO tap — because both plants now execute it.

## The arc

| run | what changed | interface Q rmse | max |
|---|---|---:|---:|
| 0088 | starting point (pre-re-sync) | 4.556 | 22.15 |
| 0093 | DSO model change + PF re-sync | 2.187 | 8.46 |
| 0095 | converged-seam fix (Q(V) anchor) | 0.464 | 0.91 |
| **0096** | **DSL tap control** | **0.130** | **0.39** |

Two independent plant defects, each masking the next: the Q(V) anchor residual
made the two loops tap at different times, which hid the fact that RMS taps
never executed at all.

## Risks / unresolved

- `controllable_transformers()` still over-selects: 24 composites built where
  the snapshot's actuator list names 20. The four extras are inert (never
  commanded) but should be narrowed.
- `TAP_MECH_DELAY_S` is now unused in `_dispatch_taps` but still imported.
- The frames are hand-authored artefacts: `Frame Tap Control qOFO` (drawn in
  the GUI) and `Frame Tap Control 3W qOFO` (`AddCopy` + one attribute edit).
  Neither can be rebuilt by script — `tap_ctrl.frame_for()` raises with an
  explicit message if either is missing or unwired.
- Every Gate-E result before 0096 was produced with a plant that did not
  execute its controller's taps after the first one per transformer.


---

## Addendum — MSC/MSR shunt steps have the SAME EvtTap defect

Asked whether the tap finding transfers to other event-fed signals. Inventory
of what we push into the RMS plant:

| # | signal | mechanism | known before the run? |
|---|---|---|---|
| 1 | DER Q (QVPRE `qset`,`Vanchor`) | pooled `EvtParam`, 88/interval | no — controller output |
| 2 | machine AVR V-ref (`usetp`) | pooled `EvtParam` | no |
| 3 | OLTC taps (`TAPCTRL.ntapcmd`) | pooled `EvtParam` (since today) | no |
| 4 | **MSC/MSR steps (`ElmShnt.ncapa`)** | **fresh `EvtTap`** | no |
| 5 | load P/Q profiles | `ElmFile` (default) / `EvtLod` | **yes** |
| 6 | DER P profiles (PV+wind) | `ElmFile` / `EvtParam` on `WTGWGO_A.Pref_in` | **yes** |

1-4 are controller outputs: the value does not exist until the MIQP runs, so an
event is inherent. What a frame/DSL buys there is not fewer events but **what
the event writes** -- a DSL parameter lands on time, a raw element attribute may
not. 5-6 are exogenous and fully known in advance, so they need no events at
all; `ElmFile` already exploits that.

### Measured: shunt steps are broken the same way

`SH_MSC_DSO_1_s0`, plant's exact dispatch path (pooled context,
`add_tap_event` with a relative delta, `_EVENT_EPS_S` offset), commands
+1/+1/-1/-1:

| commanded | landed |
|---|---|
| +1 @ 20.5 s | 20.53 ✓ |
| +1 @ 60.5 s | **120.50** (+60 s) |
| -1 @ 100.5 s | **160.51** (+60 s) |
| -1 @ 140.5 s | **never** |

Final `c:ncapa` = +1 where +1+1-1-1 should give 0: one command lost, two three
dispatch intervals late. Exactly the OLTC signature (first on time, the rest at
`t_event + 60 s`). Since taps no longer use `EvtTap`, this establishes the
defect as a property of **`EvtTap` itself**, not of any element class.

**No DSL escape exists for the shunt.** `TechRef_Shunt` S6 lists EMT state
variables only -- there is no RMS input signal analogous to the transformer's
`nntapin`. The integrated shunt controller is RMS-capable but is a *local*
voltage regulator, not an externally commanded setpoint, so it does not fit an
OFO-dispatched actuator.

### Why this has not bitten yet, and the guard

Every run examined has `zone_tso_shunt_states` at 0 for the whole horizon, so
no second step was ever dispatched. `pf/plant.py` now counts shunt events and
warns loudly (console + `logger.warning`) on the second one: a run whose
integrator moves a bank more than once is not trustworthy on its MSC/MSR
trajectory.

### Options, none taken

1. accept + document the ~60 s deferral (a 3-interval lag on a discrete
   actuator -- hard to defend in a control study);
2. model each step as its own single-step `ElmShnt` and switch it with
   `EvtSwitch` (a different event class, untested for this defect) -- a model
   change, 8 banks x ncapx steps;
3. find the root cause of the +60 s. Two element classes now show it
   identically, and offset, pooling, pre-ComInc existence, admission barriers
   and event-folder contents are all ruled out. The remaining suspect is
   `EvtTap` semantics themselves ("it is also fired for every calculation
   point after its execution time", User Manual 13.9.15).

### Profiles (5-6): postponed deliberately

`ElmLod` exposes `Pext` [MW] / `Qext` [Mvar] as RMS inputs, and DSL supports
`lapprox`/`lapprox2` lookups against `IntMat` objects stored inside the model,
so a per-load DSL could generate the profile from simulation time with **zero
events**. DER P has the same hook via the WECC frame's `Pref_in` signal.
Judged **not worth it now**: `ElmFile` is already event-free and is PF's
supported feature for this, so the gain is uniformity rather than correctness,
against ~157 composites plus 157 `IntMat` objects of unmeasured ComInc cost.
Revisit if the ElmFile build becomes the bottleneck or the file format cannot
express a needed profile.


### Option 2 (EvtSwitch) TESTED and REJECTED

Before rebuilding 8 banks as single-step shunts, the mechanism was tested on
the existing `SH_MSC_DSO_1_s0` with `ncapa = 1` (bank in service, so switching
it is visible): three alternating `EvtSwitch` events (`i_switch` 1/0/1 at
20.5 / 60.5 / 100.5 s).

Result — **two independent failures**:

1. **The same +60 s deferral.** The bank's Q sat at -28.07 Mvar through the
   first command at 20.5 s and only moved at **t = 120.51 s** — the 60.5 s
   command, 60 s late. `EvtSwitch` is therefore no escape: the deferral is not
   specific to `EvtTap` but common to at least two event classes.
2. **Numerical failure.** Immediately after the queued events landed together
   at 120.5 s (five Q jumps within 0.06 s: -28.1 -> -35.9 -> -23.6 -> -10.5 ->
   -62.5 -> -37.5) the simulation diverged:
   `G 07: System matrix inversion failed. Simulation interrupted.` Switching a
   shunt in and out mid-RMS is an abrupt topology change the network model does
   not survive here.

So option 2 would replace a silent lateness with a hard crash, and would not
even fix the lateness. **Not pursued.**

### What that leaves

* **Option 1 — accept + document.** The guard now warns on the second shunt
  step; a run that moves a bank more than once is flagged unreliable.
* **Option 3 — chase the `EvtTap`/`EvtSwitch` root cause.** Now the stronger
  candidate, since the +60 s is shared across event classes and across element
  classes, i.e. it is a property of how events are admitted into a running
  calculation, not of any one mechanism. Everything under our control has been
  ruled out (offset, pooling, pre-ComInc existence, admission barriers,
  event-folder contents, retirement).
* **Option 4 (new, invasive) — replace each bank with a DSL-driven reactive
  source.** A DSL with the terminal voltage as input and the step count as a
  parameter can reproduce a switched capacitor exactly (`Q = n * Q_step * u^2`,
  preserving the V-squared dependence a constant-Q source would lose), driven
  on the proven pooled-`EvtParam` path. Cost: the MSC/MSR would no longer be
  `ElmShnt` in PF, which breaks the `pf_sync`/Gate-C parity structure that maps
  `net.shunt` -> `ElmShnt` and checks shunt P/Q. Not a small change.


---

## Option 3 RESOLVED — PF applies event times MODULO A 60 s WINDOW

The `+60 s` deferral is not a defect in `EvtTap`, not element-specific, and not
a property of the persistent pool. **Once an RMS calculation is running,
PowerFactory interprets a simulation-event time modulo a 60 s window.** An
event scheduled at absolute time `te` fires at `te mod 60` inside the current
window, which *presents* as a deferral of `60 * floor(t_clock / 60)` when
absolute times are used.

### Evidence

Arming one `EvtTap` per interval on `SH_MSC_DSO_1_s0`, `DT = 20 s`, absolute
scheduling:

| armed at clock | wanted | landed |
|---:|---:|---:|
| 0 / 20 / 40 | 0.5 / 20.5 / 40.5 | on time |
| 60 / 80 / 100 | 60.5 / 80.5 / 100.5 | 120.5 / 140.5 / 160.5 (+60 s each) |

With `sched = te - 60*floor(t_clock/60)` (i.e. `te mod 60`), twelve consecutive
events over four windows to t = 220 s **all landed on time**:

| armed | wanted | scheduled | landed |
|---:|---:|---:|---:|
| 0 / 20 / 40 | 0.5 / 20.5 / 40.5 | 0.5 / 20.5 / 40.5 | exact |
| 60 / 80 / 100 | 60.5 / 80.5 / 100.5 | 0.5 / 20.5 / 40.5 | exact |
| 120 / 140 / 160 | 120.5 / 140.5 / 160.5 | 0.5 / 20.5 / 40.5 | exact |
| 180 / 200 / 220 | 180.5 / 200.5 / 220.5 | 0.5 / 20.5 / 40.5 | exact |

### It retro-explains every observation of the last two days

| observation | explanation |
|---|---|
| "only the first tap fires" | the first was armed at clock < 60 s |
| `DT <= 15 s` never showed it | every arming clock stayed below 60 s |
| `DT = 40 s` deferred by 120 s, not 60 | arming clock 120 -> two windows |
| arming all events up-front worked | clock 0; all events inside window 1 |
| the first backdating attempt failed | it backdated the clock<60 event too, into the past |
| `EvtSwitch` deferred identically | same scheduler, not an `EvtTap` property |
| pool / retirement / barriers / offset all irrelevant | none of them touch the window |

`ComInc` and `ComSim` expose no attribute equal to 60; the origin of the
constant is unknown and undocumented.

### Fix — IMPLEMENTED 2026-07-31

In `ScreeningContext.add_tap_event` (and any other absolute-time event on a
running calculation):

```python
sched = t_event - 60.0 * math.floor(current_sim_time / 60.0)
```

Implemented in `pf/screening.py`:

* `EVENT_WINDOW_S = 60.0`, documented with the evidence;
* `ScreeningContext` tracks the calculation clock (`_sim_time`, updated by
  `simulate()` **and** by the admission barriers, which also advance it);
* `add_tap_event` writes `time = t_event - 60*floor(clock/60)`.

**And the assertion that must accompany it** — `PowerFactoryPlant.
_verify_shunt_steps`, called after every `advance()`: it compares each bank's
commanded shadow step against the position PF actually holds (`c:ncapa`) and
raises on any mismatch. Without it the rule fails *silently*, because both the
shadow store and the mirror net keep the commanded value — which is precisely
how the tap defect hid for two days. Shunt steps are instantaneous (no
mechanical lag), so after a completed advance the two must agree exactly.

The earlier "second shunt step" warning is removed: it claimed the trajectory
was unreliable, which is no longer true, and the verifier is a strictly better
guard (it checks what happened, not what might have).

### Verified end to end

The exact case that failed before, through the real dispatch path:

| commanded | before the fix | after |
|---|---|---|
| +1 @ 20.5 s | 20.53 | **20.50** |
| +1 @ 60.5 s | 120.50 (+60) | **60.50** |
| −1 @ 100.5 s | 160.51 (+60) | **100.50** |
| −1 @ 140.5 s | never | **140.50** |
| final `c:ncapa` | +1 (wrong) | **0 (correct)** |

### Remaining caveat

The rule is **empirical**. It fits 30+ measurements across `ElmTr2`, `ElmTr3`
and `ElmShnt` via `EvtTap` and on `EvtSwitch`, and is validated on-time over
four windows to t = 220 s. A 1080 s replay spans 18 windows and has not been
exercised — but it no longer needs to be trusted blindly, because
`_verify_shunt_steps` will stop the run the moment the plant disagrees with the
command.

### Correction to the tap write-up

`EvtTap` was **usable all along** — this window rule would have fixed the OLTC
taps too. The earlier framing ("EvtTap is outside the supported path for RMS
tap control") was too strong: the TechRef statement about `nntapin` stands, but
the failure actually observed was this scheduling quirk, not an inherent
limitation of the event.

The DSL tap layer is **kept** regardless: it is the vendor-documented mechanism,
it gives a real mechanical time constant instead of an instantaneous jump, and
it does not depend on an undocumented quirk. Its measured Gate-E benefit
(run 0096: interface-Q rmse 0.4638 -> 0.1295, zone-V 0.0021 -> 0.0002) is
unaffected by this discovery.
