# 2026-07-30 — Gate E static-vs-RMS bias: the exported snapshot is not a Q(V) fixed point

**What:** Root-cause diagnosis of the `run_rms_phase6_replay` static-vs-RMS
divergence that reappeared on scenario `rural_700` (run 0088, 1800 s).
**Why:** The replay reproduced the quasi-static (QSS) interface-Q and voltage
trajectories well on `base_410`/`wind_replace`; on `rural_700` a persistent,
non-decaying offset returned.
**Timestamp:** 2026-07-30, midday.
**Status:** cause identified and reproduced in a static-only probe. **No code
changed** — the fix touches the initialisation contract and is held for
discussion per the standing rule.

---

## 1. Symptom (run 0088, `rural_700`, 1800 s, ±1.0 pu capability, δ = 0.005 pu)

* Interface Q: RMS sits above the static equilibrium on 9 of 12 interfaces by a
  roughly constant 4–8 Mvar that establishes itself inside the first ~200 s and
  never decays. Endpoint RMSE 4.56 Mvar, max 22.2 Mvar.
* TS zone-mean voltage: RMS above static by +0.0037 / +0.0004 / +0.0024 pu
  (zones 1/2/3).
* Discrete actuators diverge downstream: TSO `z1[1]` taps at 190 s / 1440 s in
  the RMS but only at 540 s in the static; `DSO_4|trafo_10` diverges likewise.
* Per DSO the picture is internally consistent: DER Q is ~10–14 Mvar lower in
  the RMS and the interface imports ~10–13 Mvar more, i.e. the DSO reactive
  balance is preserved and the *setpoints* differ, not the tracking.

## 2. What was ruled out

| hypothesis | verdict | evidence |
|---|---|---|
| PF↔pandapower initialisation mismatch | **refuted** | RMS `u_TN_bus*` at t = 0 matches the snapshot solution to max 1.55e-6 pu over all 30 TN buses |
| Machines on Q limits (parity LDF runs `iopt_lim = 0`, plant runs `enforce_q_lims=True`) | **refuted for this run** | no gen within 0.5 Mvar of `min_q_mvar`/`max_q_mvar` in the snapshot |
| One-interval profile lag (`rms_profile_settle_s = 0`) | **refuted as the driver** | the ElmFile profile changes ~0.03 % per 20 s interval; far too small for a 5 mpu step |
| DER capability model mismatch | **refuted** | both plants clip through `der_qv_local_loop._qv_capability` and the same `DER_Q_CAPABILITY_OVERRIDE_PU` global |
| Stale DSO line types in PF | **not this run** | `sync_full` → `sync_lines` pushes `rline/xline/cline/bline/sline` per line, and run 0088 predates the line-type edit (see §5) |

## 3. Cause

`experiments/runners/multi_tso_dso.py:1846` closes initialisation with

```python
pp.runpp(net, run_control=False, ...)   # "the QVLocalLoops ... iterate
                                        #  inside the first main-loop runpp"
```

so the post-initialisation state is **deliberately not a fixed point of the
plant-side DER Q(V) law**. Phase 3 (`:1791-1812`) has just reset
`q_set_mvar = 0`, left `qv_vref_anchor_pu = NaN` (cold-start anchor falls back
to the nominal `qv_vref_pu = 1.03`) and seeded `q_mvar` with the *single-pass
linear* equilibrium of `seed_qv_equilibrium`.

That state is what `pf/replay.py` exports as the PowerFactory snapshot. The two
plants then do opposite things with the residual:

* **static plant** — its first `advance()` (the pre-controller profile settle at
  `:2814`) runs `pp.runpp(run_control=True)` and *removes* the residual;
* **RMS plant** — `pf/plant.py::_anchor_qv_precontrollers` runs a load flow on
  the synced state and sets each QVPRE block to `qset = q_lf / S_n`,
  `Vanchor = v_lf`, i.e. it re-anchors the droop characteristic **through the
  non-equilibrium point** and thereby freezes the residual in permanently.

From then on the two plants carry different Q(V) characteristics — vertically
offset per park by that park's pending correction — for the whole run.

### Measured, static-only probe — pre-13:37 network (the one run 0088 used)

Instrumented `PandapowerStaticPlant` recording `res_sgen.q_mvar` /
`res_bus.vm_pu` at the snapshot instant and after the first two `advance()`
calls:

| | Σ DER Q [Mvar] | Δ vs snapshot |
|---|---|---|
| snapshot (= handed to PF) | 384.75 | — |
| after advance #1 | 301.84 | **−82.90** |
| after advance #2 | 330.69 | −54.06 |

Bus voltage movement over the first `advance()`: **mean 4.95 mpu, max 21.5 mpu**
(DSO_4 park bus 1.0588 → 1.0373).

Per group, and matching the closed-loop first-interval divergence sign for sign:

| group | q snapshot | after adv #1 | run 0088 static − RMS at t = 20 s |
|---|---|---|---|
| WP_STATCOM (TN) | 261.5 | 198.7 | — |
| DSO_1 | 47.4 | 40.1 | static below RMS |
| DSO_2 | 35.7 | 24.2 | static below RMS |
| DSO_3 | 3.8 | 16.2 | static **above** RMS |
| DSO_4 | 36.3 | 22.7 | static below RMS |

DSO_3 is the one group whose pending correction has the opposite sign, and it is
the one group where the closed-loop RMS ends up *below* the static. The recorded
pre-control measurements at step 1 corroborate the magnitude: static DSO_1
`v_meas_mean` 1.019576 vs RMS 1.024811 (5.2 mpu), against the probe's mean
4.95 mpu.

### Why `rural_700` is worse than `base_410`

The pending correction per park is `R · (V_local − V_anchor)` with
`R = S_n / qv_slope_pu`. `rural_700` raises **both** factors:

* `S_n` per DSO 410 → 700 MW (×1.7), and
* the post-Phase-2 DSO voltages sit further from the 1.03 cold-start anchor —
  DSO_4's parks are at 1.0567–1.0588 pu, i.e. 26–29 mpu of anchor error against
  a 5 mpu dead band.

Both scale the frozen-in error, so the same structural defect that was tolerable
at 410 MW is not at 700 MW. `--der-deadband 0.005` (the script default, half the
nominal 0.01) widens the affected set further.

### Re-measured on the post-re-sync network, with the candidate fix

Repeated after the 13:30/13:37 DSO model change and the PowerFactory re-sync
(`probe_converged_snapshot_fix.py`; the "fix" case runs one
`pp.runpp(run_control=True, max_iter=300)` at the plant seam, in the factory,
with no source change). Movement over the **first** `advance()`, i.e. the
movement the RMS plant freezes in:

| case | Σ DER Q at seam | ΔQ, advance #1 | mean \|ΔV\| | max \|ΔV\| |
|---|---:|---:|---:|---:|
| `base_410`, as-is | −4.42 Mvar | **+9.41** | 0.00095 | 0.00488 |
| `rural_700`, as-is | +7.39 Mvar | **+17.68** | 0.00229 | 0.02699 |
| `rural_700`, seam converged | +26.35 Mvar | **−1.20** | 0.00005 | 0.00014 |

Two things to read off this:

1. **The scenario dependence is confirmed and quantified.** `rural_700`'s
   frozen-in residual is 1.9× `base_410`'s in DER Q, 2.4× in mean bus voltage
   and 5.5× in worst-bus voltage. That is the "good on `base_410`, worse on
   `rural_700`" observation, measured directly.
2. **The candidate fix works.** Converging the seam cuts the residual by ~15×
   in Q, ~45× in mean voltage and ~190× in worst-bus voltage; the seam becomes a
   genuine fixed point of the Q(V) law. The remaining −1.20 Mvar is the
   nonlinear residual the linear seed cannot remove. Movement on advance #2
   (+21.7 Mvar, 4 mpu) is the controller's first dispatch and is expected.

The new DSO model reduced the defect on its own (the pre-13:37 network showed
−82.9 Mvar / 4.95 mpu mean at `rural_700`, and the sign flipped): removing the
constant reactive load makes the DSOs capacitive on average and the stiffer
conductors move the local voltages toward the 1.03 anchor. It did **not** remove
it, and the `rural_700` / `base_410` ratio survives.

**The naive form of the fix does not work**, exactly as the `:1838-1844` comment
warns: `pp.runpp(run_control=True)` with pandapower's default `max_iter = 30`
raises `ControllerNotConverged` after 31 controller calls on `rural_700`. It
needs the run-control cap `PandapowerStaticPlant.advance` already uses
(`max_iter = 300`, against `max_iteration = 50` for the inner NR). With that it
converges. So the comment's concern is real but is a solver-settings matter, not
a reason the seam cannot be converged.

### Open-loop u → y confirmation (`rural_700`, post-re-sync, run 0017)

`run_rms_openloop_uy.py --duration 300 --profiles --scenario rural_700
--der-deadband 0.005` replays the static run's captured actuator + profile
timeline verbatim to the RMS plant, so the controllers are removed as a factor.
Plant-only endpoint residual:

| quantity | rmse | mae | max |
|---|---:|---:|---:|
| interface Q | 1.160 Mvar | 0.872 | 4.938 |
| zone voltage | 0.00165 pu | 0.00110 | 0.00390 |

### Matched closed-loop run (0089) — and a correction

Run 0089 repeats 0017's settings exactly (300 s, `rural_700`, δ = 0.005,
±1.0 pu capability, post-re-sync model) with the full closed loop. **Gate E
PASS.** The matched pair:

| | interface Q rmse | mae | max | zone V rmse | max |
|---|---:|---:|---:|---:|---:|
| closed loop (0089) | 1.665 | 1.216 | 8.585 | 0.00157 | 0.00280 |
| open loop, identical u (0017) | 1.160 | 0.872 | 4.938 | 0.00165 | 0.00390 |

**Correction.** On first seeing 0017 I compared it against run 0088 and wrote
that "most of the closed-loop divergence is the controllers deciding
differently". The matched pair refutes that: closed-loop amplification is only
**1.4×** in interface-Q rmse, and the zone voltages are marginally *better*
closed-loop than open-loop. The plant-level residual accounts for roughly 70 %
of the closed-loop rmse — the divergence is plant-dominated, not
controller-dominated. The 4× figure came from an unmatched pair (0088 is 1800 s
and predates both the model change and the re-sync) and should be discarded.

**The divergence is also much smaller than what prompted this investigation:**

| | interface Q rmse | max | zone V rmse |
|---|---:|---:|---:|
| 0088 (pre-re-sync, 1800 s) | 4.556 | 22.15 | 0.00274 |
| 0089 (post-re-sync, 300 s) | **1.665** | **8.59** | **0.00157** |

2.7× better in rmse, 2.6× in worst case, with no controller or initialisation
change — purely the DSO model change (zero constant Q, stiffer conductors, DSO_3
parallel circuit) and the PowerFactory re-sync. Not a clean pair either (1800 s
vs 300 s), but 0088's per-interval error was roughly flat in time (5.36 Mvar
mean at the first interval, 4.61 at the last), so horizon truncation does not
explain the bulk of it.

### Where the residual sits in time (0089, closed loop)

| t [s] | 20 | 40 | 60 | 100 | 160 | 180 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean \|Δ interface Q\| [Mvar] | 1.62 | 2.20 | 1.22 | 1.13 | 0.69 | 1.17 | 1.21 |
| max \|Δ interface Q\| [Mvar] | 5.59 | **8.59** | 3.39 | 3.79 | 1.40 | 1.81 | 1.62 |

The **worst errors of the entire run occur in intervals 1–2** and then decay by
~3× — which is exactly the anchor defect's signature, and exactly what the
converged-seam fix removes. The step at t = 180 s is the first TSO dispatch
(`tso_period_s = 180`), after which the trace is flat.

Note also that the open-loop **linear drift** (§ above, reaching 1.68 Mvar mean
at t = 300 s) is *attenuated* in closed loop, which settles at 1.21 Mvar: the OFO
integrators reject it as the slowly-varying plant disturbance it is. That lowers
its priority — it is a plant-fidelity question, not a closed-loop accuracy one.

The residual's **time profile** separates two effects:

| t [s] | 20 | 40 | 60 | 100 | 200 | 300 |
|---|---:|---:|---:|---:|---:|---:|
| mean \|Δ interface Q\| [Mvar] | 1.51 | 0.44 | 0.06 | 0.31 | 0.97 | 1.68 |

* an **interval-1 spike** (1.51 mean / 4.94 max Mvar) that collapses to 0.06 by
  t = 60 s — the anchor defect, washed out once the OFO has commanded `q_set`
  and re-anchored both plants;
* a **linear drift** from t ≈ 60 s onward, ~0.6 Mvar per 100 s in the mean.
  This is a **separate, unexplained** plant-level effect. It is not the anchor
  defect (that one decays) and it is not visible in the zone voltages, which
  stay at 1.0–1.5 mpu. Candidates not yet discriminated: ElmFile playback vs
  `apply_profiles` evaluating the profile at different instants, slow
  governor/AVR dynamics with no static counterpart, or the RMS not fully
  settling inside 20 s so a per-interval residual accumulates. **Open.**

## 4. Candidate fixes — for discussion, not implemented

1. **Make the exported snapshot a Q(V) fixed point.** One converged
   `run_control=True, max_iter=300` solve after the Phase-3 seed and before
   `plant_factory(...)`. **Measured to work** (see above): residual −1.20 Mvar /
   0.00014 pu max, from +17.68 Mvar / 0.02699 pu. The `:1838-1844` comment's
   concern is real but is a solver-settings matter — the default `max_iter = 30`
   raises `ControllerNotConverged`, 300 (what the static plant itself uses)
   converges. This is the minimal fix and it also removes a 2.3 mpu artefact
   from the *static* runs' first interval.

   Open design question: apply it unconditionally at the seam (both plants, so
   quasi-static results change too) or only when `plant_factory is not None`
   (RMS runs only, leaving the static baseline bit-identical to the archive).
   The first is the physically consistent choice — the seam state should be a
   fixed point of the plant law regardless of which plant reads it — and it is
   the one I would take, but it invalidates more archived results.
2. **`--seed-der-anchor` alone is not sufficient and would make it worse.**
   It sets the static anchor to `v_local` while leaving `q_set = 0`, so the
   static law becomes `Q = −R·dz(V − v_local)` against the RMS's
   `Q = q_lf − R·dz(V − v_local)`: the two then differ by the constant `q_lf`.
   It is only meaningful *combined* with fix 1.
3. **Residual second-order issue, independent of 1 and 2.** Even at a shared
   fixed point the dead zones sit at different voltages — static at
   `qv_vref_pu = 1.03` until the first OFO apply re-anchors it, RMS at `v_lf`
   from t = 0. With δ = 5 mpu and anchor errors up to 29 mpu this is a
   first-order *gain* difference (0 vs `R`) in the first interval. It resolves
   itself once the first dispatch re-anchors the static side.

## 5. Interaction with the DSO model changes landed today

`network/ieee39/constants.py` and `hv_networks.py` were edited at 11:52 and
again at 13:30/13:37, i.e. **after** run 0088 (10:28). Run 0088 therefore
predates all of them and none of them explains its divergence.

The authoritative description is
`2026-07-30_ieee39_dso_powerfactory_sync_handover.md`. Persistent plant changes,
all three confirmed present in the tree as of 13:37:

1. per-DSO HV line types — 305-AL1/39-ST1A for DSO 1/2/4, 490-AL1/64-ST1A for
   DSO_3, replacing 184-AL1/30-ST1A everywhere;
2. DSO_3 corridor HV bus 5–6 as two parallel 490 mm² circuits
   (`parallel = 2`, 15.6 km);
3. zero constant reactive load in every DSO — `Q_load(t) = 500 Mvar ·
   mv_rural_qload(t)`, i.e. signed and capacitive on average (mean −24.9 Mvar).

The DSO_3 ×2 DER/active-load multiplier is **experiment-only**
(`_apply_dso_overrides` in `analysis/annual_dso_pq_characterization.py`, behind
`--dso-der-scale` / `--dso-load-p-scale`). It is not in the builder and
therefore does not enter any RMS replay; it was deliberately not pushed to
PowerFactory.

PF state audit before the re-sync (read-only, `01_LDF_Parity`, full stack):

* Every `ElmLne` owns its own `TypLne` — no sharing, so `sync_lines`' per-line
  type writes cannot alias between DSOs. (Worth checking, because `sync_lines`
  writes `rline/xline/cline` onto `obj.typ_id`; a shared type would have made
  DSO_3's 490 mm² data overwrite the others'.)
* All 44 DSO lines held the **old** 184-AL1 data (r = 0.1571 Ω/km, x = 0.4,
  c = 8.8 nF/km, i_max = 0.535 kA).
* `sync_full` → `sync_lines` writes exactly these attributes, so an RMS run
  self-heals them at its own start; the manual re-sync was still done, to
  refresh `export/snapshots/` and re-establish the Gate-C checkpoint. See
  `2026-07-30_pf_resync_dso_lines_qload.md`.

## 6. Risks / unresolved

* The candidate fix changes the initial state of **every** scenario, so all
  archived Gate-E and quasi-static results predate it. Same re-run cost as the
  frozen-P Jacobian fix discussed on 2026-07-29.
* The open-loop `u → y` test (`experiments/run_rms_openloop_uy.py`) has never
  been run on `rural_700`; it is the clean confirmation that nothing *else*
  differs between the plants once the anchor defect is removed.
* Whether the residual dead-zone-position difference (§4.3) matters after fix 1
  is untested.
* Gate D (20 s timescale separation) still predates the 2026-07-29 geometry
  change and now the line-type change.

## 7. Files

* No source changes.
* Probes (scratchpad, not committed): `probe_snapshot_qv_fixedpoint.py` (the
  static-only fixed-point measurement), `probe_pf_line_types.py` (read-only PF
  line/type audit).
