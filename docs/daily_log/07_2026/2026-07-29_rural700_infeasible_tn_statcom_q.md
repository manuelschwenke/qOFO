# 2026-07-29 — `rural_700` does not initialise: single-slack export limit at step [6]

**Timestamp:** 2026-07-29 late evening → 2026-07-30 early hours
**Reason:** The 18-run dead-band × operating-window sweep was to run overnight on
scenario `rural_700`. Runs 0076–0078 all died before producing records. Diagnosis
requested, with authorisation to start the sweep autonomously *if* the failure was
fixed. A validated candidate fix now exists but it changes results for `base_410`
too, so it is **held for discussion** and the sweep was **not** started.

> **Revision note.** An earlier version of this log attributed the failure to an
> infeasible TN STATCOM reactive seed and concluded that "no feasible dimensioning
> of the present devices supplies it". The author's counter-hypothesis — that this
> is a transformer tap-initialisation matter — was tested and is **correct** for the
> overvoltage. That part is retracted below and marked. The residual blocker is a
> different mechanism (single-slack active-power export).

## Symptom

`pandapower.auxiliary.LoadflowNotConverged` from `multi_tso_dso.py:930`
(`shared_jac = JacobianSensitivities(net)`) — runner step [6], the first power flow
of the run. This precedes DER Q(V) installation at step [10.3], so the dead band δ
is irrelevant and all 18 runs would have failed identically.

## Established facts

1. **No `rural_700` run has ever completed.** Every result in the existing U-curve
   series (runs 0065–0075) records `scenario=wind_replace`. Runs 0076 (`base_410`),
   0077 and 0078 (`rural_700`) produced no `rms_records.pkl`. The 700 MW-per-DSO
   network is unvalidated, not regressed.

2. **The failing solve is the no-distributed-slack re-converge.**
   `JacobianSensitivities.__init__` (`sensitivity/jacobian.py:338`) deliberately
   re-solves with `distributed_slack` **off**, so the stored Newton Jacobian has the
   `[P_PV, P_PQ, Q_PQ]` structure the sensitivity code expects. On `rural_700` that
   solve diverges from `init="results"` (50/500 it), `init="flat"` (200 it),
   `init="dc"` (200/1000 it), each also with `enforce_q_lims=False`. `base_410`
   succeeds. **No solver flag fixes it.**

3. **~~The TN STATCOM reactive seed is the cause.~~ RETRACTED — it is a tap-init
   artefact.** The build-time seed *is* infeasible: `hv_networks.py:825-834` sizes
   each `WP_STATCOM` Q by substituting a PV-gen holding 1.03 pu and reading back the
   Q, and the power flow on line 831 passes no `enforce_q_lims`, so the substitute
   is not held to the `max_q_mvar=sn` set on line 827. That yields
   Q = 2808 Mvar total for `rural_700` (bus 5: **1655 Mvar from a 500 MVA
   converter, 3.46×**) against 1200 Mvar for `base_410`, at identical TN P
   (2198 MW) and Sn (2198 MVA) — even though the transmission-side wind replacement
   is documented as *shared* between scenarios (`scenarios/__init__.py:9-11`).
   **But this is transient.** The build-time coupling OLTCs sit at `tap_pos=0`
   (`hv_networks.py:172-177`, range ±13 × 1.25%), and Phase 1
   (`multi_tso_dso.py:1643-1704`) re-derives the STATCOM Q with the same temp-PV-gen
   trick under `run_control=True` **and** `enforce_q_lims=True`, so the seed was
   always meant to be replaced. Measured on `rural_700`:

   | state | V range | buses >1.1 | TN Q | bus-5 device |
   |---|---|---|---|---|
   | build-time (what step [6] sees) | [0.9985, **1.1485**] | **33 of 118** | 2808 Mvar | 3.46× |
   | after Phase 1 (6/8 machine 2W taps move) | [0.9213, 1.0472] | **0** | 1126 Mvar | 1.41× |
   | after Phase 2 (12/12 coupler 3W taps → [−5..2]) | [0.9532, 1.0997] | **0** | 1126 Mvar | 1.41× |

   Tap init removes the overvoltage entirely and bounds the STATCOM Q at ±sn.
   Note `net.controller` is **empty** after `add_hv_networks` — the
   `DiscreteTapControl`s are installed by the runner in Phases 1/2, so a bare
   `runpp(run_control=True)` on a freshly built net is a no-op on taps.

4. **The step-[6] failure survives tap init.** From the fully tap-initialised
   V[0.9532, 1.0997] state, the no-distributed-slack solve still fails on
   `rural_700` while `base_410` succeeds. Taps are therefore *not* the blocker.

5. **The blocker is a single-slack active-power export limit.** Removing slack
   distribution reverts every participating machine to its *scheduled* `p_mw`,
   dumping the whole shared imbalance on gen 9. Measured against DSO DER scaling
   (after tap init):

   | DSO DER | `rural_700` single-slack | required slack P |
   |---|---|---|
   | 2800 MW (k=1.0, as configured) | **FAIL** | — |
   | 2520 MW (k=0.9) | **FAIL** | — |
   | 2240 MW (k=0.8) | OK | −1118 MW |
   | 1960 MW (k=0.7) | OK | −771 MW |
   | 1400 MW (k=0.5) | OK | −157 MW |

   The threshold lies between 2240 and 2520 MW; `rural_700` sits at 2800 MW, 12–25%
   beyond it. Under distributed slack the same case needs only −878 MW.
   `base_410` needs −736 MW and converges. Extrapolated, `rural_700` at k=1.0 would
   demand ≈−1.8 GW from one machine.

6. **`rural_700` roughly doubles DER against near-unchanged load:** DSO-side DER
   1640 → 2800 MW, total load 5984 → 5997 MW.

## Fixes applied (2026-07-30, author instruction "do all fixes")

Three changes, all verified against the real `JacobianSensitivities` code path:

### 1. Dispatch freeze before the single-slack re-converge — `sensitivity/jacobian.py`

In `JacobianSensitivities.__init__`, each non-reference machine's achieved
`res_gen.p_mw` is written back into `gen.p_mw` before the `distributed_slack`-off
re-converge. The reference machine is skipped so it can still absorb the residual.
Nets already solved single-slack (the reduced per-zone nets from
`build_tso_local_net`) are a no-op, since `res_gen == gen` there.

### 2. `enforce_q_lims` mirrored from the incoming solve — `sensitivity/jacobian.py`

The re-converge now passes `enforce_q_lims=net._options["enforce_q_lims"]` instead
of silently taking pandapower's default (`False`). The Jacobian is therefore taken
under the same limit regime as the plant solution. A saturated machine stops
regulating voltage and pandapower moves it PV → PQ in `_ppc['internal']`, which the
existing bus-type extraction reads, so the Jacobian structure follows automatically.

### 3. Rating-respecting build-time seed — `network/ieee39/hv_networks.py`

Two parts, after a first attempt that had to be redesigned:

- The Q is still *derived* with limits off. Enforcing them on the derivation PF
  destabilises it — pandapower's PV→PQ limit loop does not converge on `rural_700`
  at the neutral-tap build point — and the original comment was right that this PF
  is robust precisely because the substitute PV-gens may absorb any mismatch as Q.
- The inherited value is instead **clamped to ±sn on write-back**, with a
  `verbose >= 1` line naming the device and both values.

New helper `_reinit_pf` gives both build-time solves a start-strategy ladder
(`init="auto"` → `init="dc"` → `init="dc"` + `distributed_slack=True`) and raises a
`RuntimeError` naming the cause if all fail. Needed because a rating-limited seed at
neutral taps is genuinely unsolvable single-slack on `rural_700`. `base_410` still
succeeds on the first rung, so its build-time state is unchanged apart from the
clamp.

### Measured result

| | `base_410` | `rural_700` |
|---|---|---|
| build-time TN seed Q | 1200 → **1036** Mvar | 2808 → **1653** Mvar |
| worst seed S/Sn | 1.66 → **1.41** | 3.46 → **1.41** |
| `JacobianSensitivities` | OK | **FAIL → OK** |
| max \|dV\| vs plant solution | 0.05788 → **0.00000** pu | FAIL → **0.00000** pu |
| stored `J` | 253×253, finite | 255×255, finite |
| pv / pq buses | 5 / 124 | 3 / 126 |

`rural_700`'s 0.03572 pu residual predicted for fix 1 alone vanished once fix 2 was
added, confirming that residual was the `enforce_q_lims` mismatch. Its lower PV
count reflects machines correctly held at their Q limits.

The remaining `S/Sn = 1.41` is expected: the clamp enforces the *declared*
`max_q_mvar=sn`, which ignores P, and these devices are built with `sn_mva == p_mw`.
Reducing it further requires deciding their true capability — see open questions.

**Verified end-to-end**: a full replay run on `rural_700`
(`--duration 300 --profiles --physical-capability --der-deadband 0.005
--start-time "2016-01-05 08:00"`) passes steps [6], [7], Phase 1, Phase 2 and enters
the RMS stepping loop with DSO dispatches, where every previous attempt died at
step [6].

**Side effect on the single-slack export limit (fact 5):** with the dispatch frozen,
the reference machine only takes the loss residual, so the −1.8 GW demand that broke
`rural_700` no longer arises. The 12–25% margin noted under Risks is therefore no
longer the binding constraint it was.

### 4. Orphaned event-pool slots — `pf/screening.py` (surfaced by the fix above)

Not one of the three authorised fixes; it only became reachable once `rural_700`
cleared step [6] and got as far as the PowerFactory leg, where it failed with

```
pf.session.PFSessionError: persistent event 'qofo_pool_p_0001100' has no p_target
```

`prepare_persistent_event_pool` discovers qOFO-owned events that persist in the
project between runs and raised if any lacked a `p_target`. That state needs no
defect in the module to arise: `_new_pool_event` creates the object and
`_append_param_slot` assigns `p_target` as a *separate* step, so a run that dies
between the two — or whose target element is later removed — leaves an unarmable
slot behind. Runs 0076–0079 all aborted, and the leftover slot then blocked *every*
subsequent run, including `base_410`.

Such a slot is now **deleted** rather than raised on, in the same category as the
unmanaged events the function already removes, and counted separately
(`stats["orphaned_removed"]`, plus a `[event_pool]` line) so genuine pool corruption
stays visible. This is the "proper fix for stale fired pool slots" that has been on
the open-items list.

Caught during review: the warning was first written as `logger.warning`, but
`pf/screening.py` defines no `logger` and reports via `print()` — a `NameError` that
`py_compile` cannot see, on a path that only executes when orphans exist. Switched to
`print()` and re-checked the module for undefined names.

### No downstream reader of scheduled `gen.p_mw`

Checked before adopting fix 1: nothing reads `jac.net.gen.p_mw`. The only
`.net.gen` accesses in `sensitivity/`, `controller/`, `optimisation/`,
`experiments/runners/` and `core/` read `bus`, `in_service`, or `len()`
(`multi_tso_coordinator.py:637-644`, `multi_tso_dso.py:2046/2071`), and
`gen.p_mw` appears nowhere in the sensitivity computation itself.

## Candidate fix (superseded by the section above — kept for the record)

Write the distributed-slack result back into `net.gen.p_mw` before the single-slack
re-solve, so each machine keeps the output it actually had and the reference machine
is left with only the loss residual:

| scenario | current (as-scheduled) | candidate (frozen P) |
|---|---|---|
| `base_410` | OK, but **0.05788 pu** from the plant solution | OK, **0.00000 pu** — identical point |
| `rural_700` | **FAIL** | **OK**, 0.03572 pu |

Jacobian `J` is stored in both cases. This is not merely a convergence patch: the
current code evaluates the sensitivity Jacobian **0.058 pu away** (on `base_410`)
from the operating point it is supposed to linearise about, because dropping slack
distribution *moves the operating point*. Freezing P makes the Jacobian coincide
with the plant state, which is what the sensitivity model assumes (machines hold P
and V). It is a fidelity improvement independent of `rural_700`.

`rural_700`'s residual 0.036 pu is a **separate** pre-existing inconsistency:
`JacobianSensitivities` passes no `enforce_q_lims`, so pandapower's default
(`False`) applies, while the plant solves with `enforce_q_lims=True`. On
`rural_700` at least one machine sits on a Q limit in the plant solution and is
released in the Jacobian solve; on `base_410` none is, hence 0.000 pu.

## Open questions — decision required

- **Adopt the frozen-P fix?** It changes the Jacobian for **every** scenario, so the
  existing `base_410`/`wind_replace` results (the whole U-curve series) would need
  re-running. Held per the standing rule that architectural changes are discussed
  first.
- **`enforce_q_lims` mismatch** between plant and Jacobian solves — fix in the same
  change, or keep separate to isolate its effect?
- **`hv_networks.py:831`** should get `enforce_q_lims=True` (or an explicit clamp)
  regardless, so the build-time seed cannot silently exceed rating. Cosmetic given
  Phase 1 replaces it, but it is what made this failure mode unreadable.
- **`sn_mva == p_mw` for all four `WP_STATCOM` devices**, so their true converter
  reactive headroom √(sn²−p²) is zero and `max_q_mvar=sn` ignores P. Post-Phase-1
  they run at 1.41× (`rural_700`) and 1.36× (`base_410`) apparent power. Pre-existing
  and *not* `rural_700`-specific; flagged for the record, not blocking.

## Changes to code

**None retained.** A third `init="dc"` fallback added to
`runpp_with_stored_jacobian` (`sensitivity/jacobian.py`) earlier in the session was
**reverted**: fact 2 refutes the premise, and it could only convert a *raised*
divergence into a silently accepted DC-initialised solution, possibly on a different
solution branch. Since power-flow non-uniqueness is itself a subject of this thesis
(the P2 dead-band-edge multi-equilibrium finding), a fallback that may silently
select another branch is the wrong default. `sensitivity/jacobian.py` is back to its
known-good state.

## Risks / unresolved

- The existing U-curve (0065–0075) is on `wind_replace`, i.e. the **410 MW**
  network, and on the topology *before* the parallel session's change. No
  `rural_700` figure will be comparable to it.
- Adopting the frozen-P fix shifts `base_410` too → the U-curve series needs
  re-running on whichever variant is adopted.
- `deadband_vq_multi.py` and `deadband_band_fig.py` guard on
  `scenario == "rural_700"` and therefore currently admit **zero** runs. Correct as
  written; there is simply nothing yet to plot.
- Frozen P is written into the `JacobianSensitivities` deep copy, so
  `jac.net.gen.p_mw` would hold *actual* rather than *scheduled* outputs. Any
  downstream reader expecting scheduled values needs checking before adoption.
- Whether `rural_700` remains solvable across the whole profile year (not just the
  four probed windows) is untested; fact 5 shows it sits only 12–25% inside the
  single-slack limit, and profile-driven DER peaks could exceed it mid-run even with
  the fix.
