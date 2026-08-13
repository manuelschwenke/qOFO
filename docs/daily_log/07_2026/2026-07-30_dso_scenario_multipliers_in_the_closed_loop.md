# 2026-07-30 — Per-DSO scenario multipliers in the closed loop, and why DSO_3 ×2 will not initialise

**What:** Made the per-DSO DER/load multipliers of the annual characterisation
available to the multi-TSO/DSO runner (and therefore to the PowerFactory RMS
replay), fixed a broken `--no-profiles`, and diagnosed why `DSO_3 ×2` cannot
start.
**Why:** Requested — "is the DSO_3 ×2 multiplier active? if not, do so" — after
the 0089 trajectories still did not match.
**Timestamp:** 2026-07-30, ~14:30–15:30.
**Status:** multipliers implemented and verified; **`DSO_3 ×2` blocked at runner
step [6]**, root cause established, fix proposed and held for discussion.

---

## 1. State of the three flags in run 0089 (asked directly)

| flag | run 0089 | wanted |
|---|---|---|
| DSO_3 ×2 multiplier | **not active** | active |
| contingencies | `[]` — not active | not active ✓ |
| profiles | `use_profiles: true`, ZIP, ElmFile delivery | active ✓ |

Also off in 0089: `measurement_noise.enabled`, `enable_reachability_guard`,
`use_zonal_gen_dispatch`.

The multiplier was never reachable from the closed loop: it lived only in
`analysis/annual_dso_pq_characterization.py::_apply_dso_overrides`, behind that
script's own CLI. Every replay so far ran four symmetric DSOs.

## 2. Implemented

* **`network/ieee39/dso_overrides.py`** (new) — `apply_dso_overrides` moved here
  verbatim from the analysis module, which now imports it. The runner could not
  import from `analysis/` (that module pulls in matplotlib), and a second copy of
  validated model logic was the worse option. `analysis` tests: 6 passed.
* **`configs/config.py`** — `dso_der_scale`, `dso_load_p_scale`,
  `dso_load_q_profile_base_mvar`, `dso_line_std_type`.
* **`experiments/runners/multi_tso_dso.py`** — applies them directly after
  `add_hv_networks`, before any power flow / ZIP model / droop tagging /
  operating-point init, since all of those read `p_mw`, `base_p_mw`, `sn_mva`
  and the reactive-load base. Prints what it applied and warns that results are
  not comparable with an unscaled run.
* **`experiments/run_rms_phase6_replay.py`** — `--dso-der-scale`,
  `--dso-load-p-scale`, `--dso-load-q-base` (repeatable `DSO_x=value`).

Verified against the handover's numbers:

| | DER | load ref |
|---|---:|---:|
| DSO_1/2/4 | 700.0 MW | 261.80 MW |
| DSO_3 (×2) | **1400.0 MW** | **523.61 MW** |

`net["dso_overrides"]` records what was applied. Because the RMS snapshot is
taken downstream, `pf_sync` carries the scaled ratings into PowerFactory
automatically — no separate PF step.

### `--no-profiles` was broken

Only the `if args.profiles:` branch existed, so `--no-profiles` left
`make_gate_e_config`'s `use_profiles = True` standing **and** restored
`make_config`'s `use_zonal_gen_dispatch = True`, which builds a machine-P
schedule the PF plant cannot follow — the runner then refuses the RMS leg. The
flag never disabled profiles; it broke the run. Now the flag is authoritative
for both plants and zonal dispatch is off in either branch.

## 3. Why `DSO_3 ×2` does not initialise

Both attempts (profiles off, and profiles on with `--profile-settle 5`) died
identically:

```
[6] Initialising TSOControllers ...
  multi_tso_dso.py:955  shared_jac = JacobianSensitivities(net)
  pandapower.auxiliary.LoadflowNotConverged: nr did not converge after 200 iterations
```

**Step [6] runs at line 955; profiles are applied at line 1579.** The shared
Jacobian is therefore always built on the **build-time, un-profiled state at
neutral taps** — nameplate DER, no profile de-rating — regardless of
`use_profiles`. That is why turning profiles off changed nothing: it was never
the variable.

Measured on that exact state (distributed slack, `enforce_q_lims=True`):

| DSO_3 DER | DSO_3 load | Σ sgen P | Σ load P | result |
|---|---|---:|---:|---|
| ×1.0 | ×1.0 | 4998.0 | 5859.8 | OK, V ∈ [0.9356, 1.0709] |
| ×1.25 | ×1.25 | 5173.0 | 5923.3 | OK, V ∈ [0.9079, 1.0610] |
| ×1.5 | ×1.5 | 5348.0 | 5986.9 | OK, V ∈ [**0.8581**, 1.0426] |
| **×2.0** | **×2.0** | 5698.0 | 6114.0 | **FAIL** |
| ×2.0 | ×1.0 | 5698.0 | 5859.8 | **FAIL** |
| ×1.0 | ×2.0 | 4998.0 | 6114.0 | OK, V ∈ [0.9763, 1.0848] |

Three readings:

1. **The DER doubling is the blocker, not the load doubling** — ×2 DER alone
   fails, ×2 load alone is fine and actually *improves* the voltage profile.
2. It is a **voltage-collapse threshold, not a discrete defect**: V_min walks
   0.9356 → 0.9079 → 0.8581 → diverge as the multiplier rises. ×1.5 already sits
   at 0.858 pu, well outside any acceptable band.
3. **The profiled state is fine.** At the 2016-01-05 08:00 operating point the
   same `DSO_3 ×2` network converges under both distributed and single slack
   (Σ sgen 2324.5 MW vs 5698.0 un-profiled — profiles de-rate DER by ~60 % at
   that instant). So the scenario is not infeasible; only the build-time proxy
   for it is.

## 4. Fix applied — and a retraction

> **Retracted.** An earlier version of this section proposed *"build the shared
> Jacobian after the profile application"*, on the argument that the controllers
> were linearising about a point the plant never occupies. **That argument was
> wrong**, and it was wrong because I read only the first of the three
> `JacobianSensitivities` build sites. Kept visible because the reordering was
> authorised on the strength of it.

The line-955 build is an explicitly documented **bootstrap that is discarded**:

```
# Build one full-network Jacobian at the current (pre-profile) operating
# point ...  This snapshot is replaced by a fresh post-Phase-2 one below
# (see "Rebuild shared Jacobian"), so all controllers eventually operate on
# the same post-init cached plant model.  Avoids 8 redundant deep-copy +
# pp.runpp + dense-inversion calls inside the construction loops.
```

and line 1958 does exactly that — after the profile application and after
Phase 2 — additionally invalidating every controller's `_H_cache` and the
`seed_qv_equilibrium` LU cache. **The controllers already use a post-profile
Jacobian.** No reordering is needed and none was done.

The actual defect is narrower: a build whose numerical content is thrown away
was aborting the entire run. Fixed at `multi_tso_dso.py:955` — on
`LoadflowNotConverged`, and only when `use_profiles` is set, the bootstrap is
retried on a **deep copy** carrying the profiled start instant:

```python
try:
    shared_jac = JacobianSensitivities(net)
except LoadflowNotConverged:
    if not config.use_profiles:
        raise
    _probe_net = copy.deepcopy(net)
    snapshot_base_values(_probe_net)
    apply_profiles(_probe_net, _probe_profiles, config.start_time)
    shared_jac = JacobianSensitivities(_probe_net)
```

Properties that make this safe:

* `net` is untouched, so steps [6]–[8] keep the documented pre-profile state;
* the fallback only triggers on a divergence that is fatal today, so **every run
  that converges now is bit-for-bit unaffected** — no archived result is
  invalidated;
* the fallback point is the state the run actually starts from, so the bootstrap
  is *more* representative than the one it replaces, not less;
* without profiles the exception still propagates — there is no better point to
  fall back to, and silently continuing would hide a genuinely infeasible case.

## 5. Risks / unresolved

- **The `×2` case is unblocked but not validated.** The fallback lets the run
  start; whether the profiled `DSO_3 ×2` case stays solvable across Phase 1/2/3
  and the whole horizon is untested. `--no-profiles` at `×2` remains
  infeasible by construction and now fails with the original exception.
- The ×1.5 build point at 0.858 pu suggests the un-profiled build state is
  already marginal at symmetric `rural_700` (0.9356 pu) — worth knowing
  independently of the multiplier, since every run's sensitivity model is built
  there.
- The `--profile-settle 5` experiment (RMS controllers reading the post-profile
  state like the static ones, isolating the pre/post-profile read asymmetry as a
  cause of the 0089 shape mismatch) **has not run** — it died at step [6] before
  reaching the plant. It remains the outstanding test for the trajectory
  mismatch and does not depend on the multiplier.
- `analysis/annual_dso_pq_characterization.py` was edited (function removed, now
  imported). A parallel session owns that file; the change is two lines plus the
  deletion, but it is a conflict candidate.

## 6. Files

- `network/ieee39/dso_overrides.py` — new, holds `apply_dso_overrides`.
- `analysis/annual_dso_pq_characterization.py` — function removed, imported.
- `configs/config.py` — four per-DSO override fields.
- `experiments/runners/multi_tso_dso.py` — import + application after
  `add_hv_networks`.
- `experiments/run_rms_phase6_replay.py` — three CLI flags; `--no-profiles` fix.
