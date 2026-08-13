# Phase E — dead-band selection under contingencies (plan)

**Status:** plan only, nothing implemented. Raised 2026-07-31.
**Naming:** phase D is already taken (δ = 0.02/0.03 on window 1, for the
symmetric grid). This is **phase E**.

---

## 1. The question

Every run in the dead-band study is contingency-free and noise-free
(`contingencies = []`, `measurement_noise.enabled = False`,
`enable_reachability_guard = False`). δ\* is therefore selected against profile
evolution alone. Two distinct questions follow:

- **E1 — robustness.** Does a δ chosen under profiled operation remain
  acceptable when a contingency occurs, or does it degrade recovery?
- **E2 — selection.** Does including contingencies *move* δ\*?

E1 is the weaker and cheaper claim, and is probably what the chapter needs. E2
requires the contingency to be part of the objective the dead band is selected
against, which is a different experiment and arguably a different design rule.

## 2. Blocker: the RMS plant cannot apply contingencies

`experiments/runners/multi_tso_dso.py:2827-2843`:

```python
if not isinstance(plant, PandapowerStaticPlant):
    _unsupported = [
        label for cond, label in (
            (bool(contingencies), "contingency events"),
            (_oltc_limiter_active, "local OLTC tap-rate limiter"),
            (gen_dispatch is not None, "zonal generator dispatch schedule"),
        ) if cond
    ]
    if _unsupported:
        raise NotImplementedError(
            "non-static plant does not support: " + ", ".join(_unsupported))
```

The guard is deliberate: these features mutate `net` directly, and for a
non-static plant `net` is only a measurement mirror, so mutating it would
produce a silently inconsistent co-simulation. `_apply_contingency(net, ev, …)`
sets `in_service = False` on a pandapower row — the PowerFactory model would
never see it.

Note that profiles were once on this list and were removed on 2026-07-21 once
`Plant.apply_exogenous` existed (EvtLod for loads, EvtParam on the WECC
`Pref_in` for DER P). **That is the precedent to follow.**

## 3. Two tracks

### Track A — static-only contingency sweep (available today)

`run_multi_tso_dso(cfg)` with no `plant_factory` uses `PandapowerStaticPlant`,
which supports contingencies fully; the CIGRE Monte-Carlo experiments already
rely on this.

* **Needs no new code**, only a driver script.
* **Does not touch PowerFactory**, so it can run *concurrently* with the RMS
  phases B/C/D without any risk of terminating them. This is the practical
  argument for doing it first.
* **Limitation:** no electromechanical dynamics. It answers "what
  post-contingency equilibrium does the cascade reach, and how many dispatch
  intervals does it take", not "what does the transient look like". For a dead
  band — which acts on a voltage *magnitude* deviation, not on a transient — the
  quasi-static answer is a substantial part of the question, but it is not the
  whole of it, and the QSS/RMS divergence analysis
  (`docs/qss_rms_divergence_analysis.tex`) is precisely a caution against
  assuming the two agree in the presence of a dead zone.

### Track B — contingency support in the RMS plant (implementation)

Scope, in dependency order:

1. **Plant interface.** Add a contingency entry point alongside
   `apply_exogenous` (e.g. `Plant.apply_contingency(event)`), with the static
   plant delegating to the existing `_apply_contingency`.
2. **PowerFactory side.** Realise a trip as an `EvtSwitch` / `EvtOutage` on the
   mapped element. **Events must be pre-allocated**: PF admits only a couple of
   mid-run-created events before firing dies altogether — validated 2026-07-23,
   where a default of one slot froze every actuator at t ≈ 41 s. The event pool
   therefore needs `n_contingencies + margin` extra slots reserved at build
   time.
3. **Element mapping.** pandapower row index → PF object, for `line` at
   minimum. `pf_sync` already carries mappings for the actuator classes; this
   needs the line/trafo set.
4. **Mirror consistency.** `net` must be updated so the controllers'
   measurement image and every direct `net` read downstream stay valid, without
   the mutation itself driving the plant.
5. **Validation.** A Gate-E-style static-versus-RMS comparison across the
   contingency, to establish whether the two plants agree on the
   post-contingency operating point. Given the dead-zone multi-equilibrium
   finding, they may not — and that would itself be a result.

**Estimated effort:** this is a multi-session piece of work, not an overnight
add-on. Step 2 carries the most risk (PF event semantics under a topology
change during an RMS simulation are not yet probed).

## 4. Open design question worth deciding first

A contingency changes the plant Jacobian, but the controllers know the plant
only through **cached sensitivities** computed pre-contingency. So the
experiment must decide:

* **(a) No re-linearisation.** Controllers keep the pre-contingency
  sensitivities. This is the honest reading of the project's premise — the
  controller never sees the plant — and makes the experiment a test of
  *model mismatch under topology change*.
* **(b) Re-linearisation on detection.** Controllers recompute sensitivities
  after the event, which tests recovery under a correct model but presumes a
  detection-and-rebuild mechanism that is not currently part of the
  architecture.

**(a) is the more interesting and more defensible experiment**, and it is also
the one that needs no extra machinery. It should be stated explicitly in the
chapter either way, because the result means different things under each.

Note that the dead band interacts with this directly: a wider δ means the local
droop absorbs less of the post-contingency voltage step, leaving more for a
controller working from a stale model.

## 5. Proposed experiment (once a track is chosen)

| Item | Proposal |
|---|---|
| Windows | 2 — `2016-01-05 08:00` (mild, sharp interior optimum at δ = 0.005) and `2016-12-18 14:00` (stressed, shallow optimum) |
| δ grid | reduced: 0.005, 0.01, 0.015 — the range the profiled study identifies as relevant. Extend only if the response is strong |
| Contingency | a single line trip, N-1 secure, chosen to produce a measurable voltage step at the EHV–HV interfaces. Candidate set must be screened for convergence first (see §6) |
| Timing | fire mid-run, after the cascade has settled, leaving ≥ 5 dispatch intervals for recovery. Implies a longer horizon than the 300 s used so far — 600 s gives 30 intervals |
| Metrics | the existing three, computed **separately pre- and post-event**, plus: peak interface-Q excursion, peak nodal-voltage excursion, and intervals-to-resettle within the 2 % band |
| Size | 2 windows × 3 δ = 6 runs |

The pre/post split matters: averaging across the event would let a large
transient dominate the mean and hide the steady-state behaviour the earlier
phases measured.

## 6. Screening the contingency (do this before anything else)

The candidate line must (i) produce a voltage step large enough to be
informative at the DSO interfaces, (ii) leave a convergent post-contingency
power flow, and (iii) not be so severe that the controllers saturate, which
would mask δ entirely — the failure mode already seen with window
2016-07-15 03:00, where zero DER capability made δ irrelevant.

This screening runs on the static plant, costs minutes, and needs no
PowerFactory. It should decide the contingency before any RMS work is
attempted.

## 7. Recommendation

1. **Screen candidate contingencies** on the static plant (§6) — cheap, no PF,
   can run now.
2. **Run track A** (static-only sweep) concurrently with the RMS phases already
   queued. It answers E1 in the quasi-static sense and establishes whether the
   effect is large enough to justify track B.
3. **Decide on track B afterwards**, informed by (2). If the static result shows
   δ barely affects post-contingency recovery, the RMS implementation is hard to
   justify for this chapter; if it shows a strong interaction, the case is made
   and the implementation is scoped in §3.

Decision (a) versus (b) in §4 should be settled before step 2, since it changes
what the static result means.
