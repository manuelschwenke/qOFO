# Fleet reserve as directional Mvar margins, including the TS DER

Timestamp: 2026-10-21, 23:10 CEST.

Reason: the author asked to extend the reserve term to DER headroom and to
express it as directional Mvar margins rather than a normalised band position.
Both the author and the external review had made the directional argument for
the subordinate case; it applies equally to the TS fleet, and the DER are the
larger reactive resource in this model (±725/+901 Mvar of VDE capability at a
high-wind hour, against a few hundred per machine).

Changes:

- `controller/shunt_support.py`: `TSFleetReserveState` and its normalised cost
  replaced by `FleetMargins`, `fleet_margin_cost` and
  `fleet_reserve_direction`. Margins are directional and in Mvar:
  `R_up = Σ (Q_max − Q)` and `R_down = Σ (Q − Q_min)` over the eligible units.
  The cost is the squared shortfall of either direction below a threshold
  expressed in **bank steps**, so the threshold carries the actuator's own
  scale. `FleetMargins` holds machines first and DER second and records
  `n_gen`, because the two halves are predicted differently.
- `controller/tso_controller.py`: `fleet_reserve_state` now returns both
  blocks. The machine block keeps the capability source, band definition,
  epsilon and out-of-service mask of the `Q_gen` objective block; the DER block
  uses the same `compute_der_q_bounds` call and the same control ordering as
  `_get_control_limits`, so the margins refer to exactly the variables the MIQP
  bounds.
- `controller/shunt_hybrid.py`: the fleet path becomes a **threshold trigger**.
  This is forced, not preferred: `Q_DER` is a *control*, so there is no
  `dQ_DER/dlevel` sensitivity and no gradient can be formed for the DER half.
  The per-bank memory therefore acts as a confirmation timer — one observation
  saturates the accumulator and readiness is governed by `t_persistence_s`
  alone. The direction comes from which side is short and from whether the bank
  injects (MSC) or absorbs (MSR); both sides short yields no request, since one
  bank cannot relieve both. The condition is re-checked at commit time, because
  a memory can be ready from earlier observations while the margin has
  recovered.
- The benefit is still measured on the complete candidate, now over both
  halves: machines from the predicted outputs, DER from the candidate's chosen
  controls. Diagnostics record `fleet_cost_regular`, `fleet_cost_candidate`,
  the reduction, and the two directional margins at every TS instant.
- `configs/config.py`: `shunt_ts_reserve_start` replaced by
  `shunt_ts_reserve_margin_steps` (default 1.0), wired through the runner and
  `experiments/shunt_hybrid_check.py`.

Consequence of the weight: it no longer decides whether the path fires — the
margin threshold does. It prices the benefit in the acceptance test only. The
derivation of 100 from the 21 October gradients therefore no longer applies to
the trigger, and the weight now needs calibrating only against the
shared-objective cost it is compared with.

Verification: 177 tests pass with the one pre-existing skip. Seven fleet cases
cover a commit when a direction is short; **the DER margin counting**, where a
machine alone is short but a roomy DER lifts the fleet above the threshold and
no request is made; no request with room; no credit for relief the ordinary
controls already give; invariance of the regular solve to the supervisory
weight; admission with the delegated control pinned; and the memory behaving as
a confirmation timer.

One fixture defect found and fixed while writing those tests: the DER *control*
bounds must equal its *capability* bounds, as `_get_control_limits` sets them.
With a mismatch the solver moves the DER merely to re-enter its own box, which
healed the margin by itself and masked the path.

## The reformulation makes the signal live, and the threshold derivable

The capacitive −600 Mvar scenario, re-run against the new margins with the
banks parked:

| | min | p50 | max | samples below 25 Mvar |
|---|---:|---:|---:|---:|
| `R_up` | 952.80 | 1264.52 | 1427.58 | 0 / 42 |
| `R_down` | **24.57** | 739.92 | 1156.42 | 1 / 42 |

As expected, the DER capability dominates the *up* direction — never below
950 Mvar. But the *down* direction in zone 2 becomes genuinely tight, because a
capacitive disturbance forces every resource to absorb: it reaches 24.57 Mvar,
that is 98% of the way to the one-step threshold, against a normalised band
position that had never exceeded 64% of its own threshold. **The directional
Mvar form is therefore materially more sensitive than the form it replaces**,
which is the concrete gain from the author's suggestion.

One instant short cannot confirm a 360 s persistence, so no candidate was
created. The low tail of the zone-2 `R_down` margin over the 21 TS instants is

| t [s] | 1440 | 1620 | 1800 | 1980 | 1260 |
|---|---:|---:|---:|---:|---:|
| `R_down` [Mvar] | 29.97 | 25.10 | 24.57 | 49.75 | 51.04 |

— a monotone squeeze over three **consecutive** instants into the disturbance,
then a jump to ~50. Thresholds in bank steps:

| `k` | 1 | 1.5 | 2 | 3 | 5 |
|---|---:|---:|---:|---:|---:|
| instants short | 1 | **3** | 4 | 14 | 15 |

**`shunt_ts_reserve_margin_steps = 1.5` is derived**, 37.5 Mvar. It captures
exactly the three-instant squeeze, which is confirmable with the 360 s
persistence; it sits in the empirical gap between 29.97 and 49.75, so the choice
inside `k ∈ [1.2, 2.0]` is insensitive; and it reads physically as *less room
left than one and a half bank commits*. By `k = 3` two thirds of the hour is
short, which is the saturation that would produce a ratchet.

### Verification at k = 1.5

One fleet candidate, at **t = 1800 s**, refused as `no_fleet_relief` with
`fleet_cost_regular = 0`. The trigger behaved exactly as specified: the
condition first holds at 1440 s, the memory starts there, and 360 s of
persistence makes it ready at 1800 — not at 1620, which an earlier note of mine
predicted by miscounting the confirmation as "two instants short" rather than
"360 s elapsed since the condition began".

The refusal is the substantive result. The trigger reads the **measured** fleet
state at the start of the instant, where `R_down` is 24.6 Mvar; the cost is then
evaluated on the **regular solution's predicted** state, where the continuous
actuators have already moved and the margin is back above 37.5 Mvar. The
ordinary controller restores the margin inside the same control instant, so the
bank has nothing to add and is correctly denied credit for it.

That now holds across both formulations — the normalised band position and the
directional Mvar margin — and in both the extreme and the intermediate
scenarios. The conclusion is structural rather than parametric: **in this system
the continuous actuators restore reactive margin within one control instant, so
a switched shunt cannot add reserve.** The supervisory machinery is correct,
derived and verified; what it would deliver on this plant is nothing, and the
full-candidate check is what makes that visible instead of plausible-looking.

For completeness, the subordinate DER sit at 99.4% utilisation in this scenario,
so it is an extreme operating point in every respect.

Open: including the DER makes the condition **harder** to reach, not easier,
because their capability dominates the aggregate — which is physically correct
and consistent with the deliverable, interface-referred principle the author
established for the subordinate case, but it means the trigger will fire only
in genuine system-wide scarcity. At a low-wind hour the DER drop out as
ineligible (capability below the epsilon), so the margin reverts to
machines-only. Whether any credible operating point of this system reaches one
bank step of aggregate shortfall is still open; the capacitive scenario is being
re-run against the new margins.
