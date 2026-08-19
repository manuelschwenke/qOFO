# Results — Ch 9 §9.1 experiments A and B

**Run 2026-08-19 on the server.** Companion to
`00_daily_log/2026-08-19_handoff_ch9_experiments_A_B.md`. Method and code
changes are in `docs/daily_log/08_2026/2026-08-19_ch9_settling_table_emitter_rework.md`
(Task A) and `..._ch9_ts_period_sweep_and_ninner.md` (Task B).

Everything below is measured unless labelled otherwise. Where a number is
projected or carried over from an older run it says so.

---

## Short answer

- **Task A did not produce numbers.** The code is done and verified offline,
  but the RMS model no longer initialises at an equilibrium: a flat 60 s run
  drifts `1.36e-02` pu against `1.41e-10` pu in the run of record. The
  preflight refused, correctly — that drift is 14x the `1e-3` pu band the
  table is measured against, so every settling time would have been
  contaminated. Root cause is diagnosed and the next step is identified, but
  it is a model-data decision and was left to you.
- **Task B2 (the `T_TS` sweep) is complete**, 60/60 runs. The headline: the
  **censoring fraction is flat at 0.19–0.21 across every period from
  `N_inner = 3` to `15`**. Giving the subordinate layer five times as many
  iterations does not change how often it fails to settle. Whatever fails,
  fails for a reason that is not the iteration count.
- **Task B1 (isolated `N_inner`) is complete**, 240/240 steps. 88.8 % of
  band-edge steps never settle -- but the censored cases end a median
  **11 Mvar** (23 % of the commanded step) away, so this is reachability, not
  a tight band. Where the setpoint *is* reachable the loop settles within
  **9 iterations, median 8**.
- **`N_inner = 9` is supported, conditionally, and is not the binding
  quantity.** It is sufficient for reachable setpoints. What limits tracking is
  that the reported capability band overstates what the subordinate layer can
  deliver. Recommendation: keep `T_TS = 180 s`; move the defect into the
  capability-reporting discussion.

---

## Task A — blocked, and why that is the right outcome

### The defect the handoff identified is confirmed

`TABLE_ROWS` emitted 8 rows; the thesis prints a different 7. The table was
therefore filled by hand and a transcription error came with it:

| thesis row | thesis | measured `20260807-085455` |
|---|---|---|
| OLTC coupling transformer, one step | `11.13` s, "STS 1 B00" | **`16.28` s** at `u_DSO_1_bus43` |

`11.13` appears nowhere under `results/timescale/`. Consequences: the binding
row is the **coupler tap at 16.28 s**, not the machine transformer at
15.13 s; the margin at `T_STS = 20 s` is **3.72 s**, not 4.87 s; the ordering
sentence inverts.

The emitter now produces exactly the thesis rows, one line each, with the
location column taken from the measured worst signal — never typed.

### Why the run is blocked

| `ElmFile` state | `ComLdf` | `ComInc` | preflight drift |
|---|---|---|---|
| pointing at deleted `0543` | fail | fail | never reached |
| out of service | 0 | 0 | `1.36e-02` pu |
| repointed at `0566` (a trajectory) | 0 | 0 | `1.42e-02` pu |
| repointed at a frozen t0 profile | 0 | 0 | `1.36e-02` pu |

Two separate problems, found in sequence.

**First**, the study case's 97 `ElmFile` profile sources pointed at
`results/rms_phase6_replay/0543_.../rms_profiles_elmfile.txt`, which no longer
exists. `pf/profile_playback.py:365` writes an *absolute* path, so every
replay run repoints the shared case at its own snapshot — which dies at the
next `results/` prune. The run of record (08:54) predates replay `0543`
(14:29 the same day), which is why it worked then. Fixed; `ComInc` succeeds.

**Second**, and still open: the model initialises exactly on the load flow and
then relaxes `1.4e-2` pu away. Established by read-only diagnostics:

- `ComInc` initialises exactly on the load flow — `sum|dP| = 0.000` MW,
  `sum|dQ| = 0.000` Mvar across all 163 injectors and loads.
- All 44 `QVPRE` anchors match the load flow to machine precision, and
  `db = 0.5` pu means the Q(V) droop cannot engage.
- No tap or shunt moves.
- The move is a **step in the first sample**, then flat.
- **Active power redistributes**: the four TSO parks each shed 0.43–0.56 MW
  while the machines gain 7.4 MW (G 01 alone `+5.3` MW) and machine Q falls.
  That is a governor response to a real-power imbalance.

**Leading hypothesis, not verified:** the profile channels encode replay
`0566`'s park dispatch rather than the dispatch the current static model
carries, so the parks are driven to a P the load flow did not solve for.
Confirming it needs the per-source channel scaling (97 sources share one
10-channel file); fixing it means re-exporting a profile from the current
model state, which changes what the benchmark is.

### QVPRE — option (a) taken, and it is the faithful one

Keep the physics, reword the caption to "no secondary dispatch; primary
control (AVR, governors, local Q(V)) active".

The handoff expected the battery to differ from closed loop because `Vanchor`
is not re-anchored during it. Reading the code, it does not: closed loop
writes **both** `qset` and `Vanchor` at every dispatch (`pf/plant.py:727-733`),
re-anchoring to the voltage measured at that instant, and the battery anchors
at the load-flow point which the preflight holds. So `veff ≈ 0` at the step in
both cases. The battery *reproduces* the closed-loop anchoring condition
rather than degrading it. Neutralising `QVPRE` would measure a plant that
never operates.

---

## Task B2 — the `T_TS` sweep, 60/60 runs, exit 0

12 design windows × 5 periods, `T_STS = 20 s` fixed, weights the §9.3
selection (rebuilt and asserted bit-for-bit against
`tier1_aa4f6d4a8654.json`).

| `T_TS` [s] | `N_inner` | intervals | scored | ρ med | ρ p95 | `n_k` unc. med / p95 | censored | taps/interval | V viol frac |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60 | 3 | 13104 | 838 | 0.662 | 4.13 | 0.0 / 0.0 | **0.214** | 0.009 | 0.0071 |
| 120 | 6 | 6624 | 878 | 0.375 | 2.95 | 0.0 / 1.0 | **0.194** | 0.018 | 0.0000 |
| 180 | 9 | 4464 | 855 | 0.308 | 3.10 | 0.0 / 3.0 | **0.189** | 0.029 | 0.0000 |
| 240 | 12 | 3312 | 780 | 0.273 | 3.20 | 0.0 / 4.0 | **0.200** | 0.035 | 0.0000 |
| 300 | 15 | 2736 | 740 | 0.215 | 2.97 | 0.0 / 5.0 | **0.192** | 0.039 | 0.0000 |

`ρ_k = r_k / Δ_k` is the fraction of the commanded correction still
outstanding when the next dispatch lands. Voltage band is the configured
`0.9–1.1` pu. `n_k` quantiles are over **uncensored** intervals only.

### What it says

1. **The censoring floor is period-independent.** ~19–21 % of dispatch
   intervals never bring the tracking error inside 1 Mvar and keep it there,
   at every period from 3 to 15 subordinate iterations. **This is the central
   result.** `N_inner` is not the constraint.
2. **When the loop settles, it settles immediately** — uncensored median
   `n_k = 0` at every period. The typical interval is already in band at the
   first subordinate sample.
3. **The median residual improves monotonically** (0.662 → 0.215) but the
   **p95 plateaus after 120 s** (2.95, 3.10, 3.20, 2.97). Longer periods buy
   median performance, not tail performance.
4. **Tap lockout is not the explanation for 60 s.** Occupancy is ~0.02
   everywhere and taps barely move (0.009 per interval at 60 s). B.6's
   hypothesis is not supported. Per unit time the tap rate is roughly constant
   across the sweep, so the period choice carries no wear penalty.
5. **Voltages are essentially clean** against the configured band: 0.7 % of
   intervals at 60 s, zero at every longer period.

### What it does NOT say — and this matters for the chapter

**ρ shows no U-shape, but ρ cannot show one.** It measures how well the
subordinate layer executed the correction it was *told* to make, not whether
that correction was still right when it landed. The stale-setpoint cost lives
in the supervisory tracking objective (`f_ts`, `f_q`), which this script does
not compute.

**So the selection argument for 180 s is not yet measured.** The sweep
establishes that nothing breaks between 60 and 300 s and that the subordinate
layer is not the limiting factor; it does not establish an optimum. If you
want the U-shape, the missing experiment is the same sweep scored on the
supervisory objective rather than on interface tracking.

### Corrections to the brief, from the code

1. **The coupler OLTC cooldown is not 60 s.** `oltc_cooldown_s = 30.0`
   (wall clock), `oltc_cooldown_s_mt = 180.0`, and `int_cooldown = 6`
   *iterations* = 120 s. The binding lockout is the iteration count, so at
   `T_TS = 60 s` the changer is nominally unavailable for **twice** the
   interval. The concern in B.6 is real but understated and driven by a
   different mechanism than named. (Measured occupancy is ~0.02 regardless,
   so it does not bite here.)
2. **`dataclasses.replace(spec, tso_period_s=X)` is not "nothing else".** The
   selected design runs `coordination_mode = "sbx_h"` and `SBXConfig` carries
   its own `tso_period_s` the runner requires to match
   (`multi_tso_dso.py:1280`). Six of nine pilot runs raised on it. The
   coupling is real: `k_sched = 2` is a cycle length in **TSO iterations**, so
   sweeping `T_TS` also scales the SBX-H settlement cycle from 2 min to 10 min.
   **That is a confound in this sweep.** `--sbx-cycle` offers both readings;
   the default holds `k_sched`, i.e. holds the controller's configuration
   fixed and lets the wall-clock consequence follow — what changing `T_TS` in
   service would do.
3. **`rural_700` is not a different network.** Same IEEE 39 + four 110 kV
   sub-networks as `base_410`, differing only in installed DER capacity per
   DSO (460 MW wind + 240 MW PV against 270 + 140). Not a finding. But
   `ScenarioSpec` defaults to `base_410`, so the thesis should state which
   capacity scenario Ch 8's benchmark table lists.

---

## Task B1 — isolated `N_inner` (running)

### Two design faults found, both of which would have produced wrong numbers

**Fault 1 — the VDE dead zone.** The first probe returned `N_inner = 0`
everywhere. Not a fast loop: `band_w = 0.00`. The window was
`d_quiet_summer`, whose stratum is `none`. `WINDOW_META`'s stratum is exactly
the DER reactive-capability tier, as the sweep's own measured band widths
confirm:

| stratum | windows | median reported band |
|---|--:|--:|
| `full` | 6 | ~190 Mvar |
| `partial` | 4 | ~55–85 Mvar |
| `none` | 2 | ~0.1–0.7 Mvar |

Those two windows are now excluded and **reported as exclusions, not
failures** — counting them would put a spurious censoring fraction into
eq. (9.2).

**Fault 2 — the capability band is not additive across a DSO's interfaces.**
The second probe stepped all three interfaces of a DSO to their reported
edges simultaneously. The reported capability is *per interface*, but the DER
reactive power backing it is *shared*, so those bands are individually
feasible and jointly are not: `DSO_4|trafo_9` ended **87–101 Mvar** from a
37–44 Mvar setpoint while its two siblings tracked to within 1.5–7 Mvar. The
subordinate MIQP allocated the shared capability and starved one interface,
and every such case came back censored — measuring the infeasibility of the
request, not the speed of the loop.

This is the same defect class as the known voltage-blind over-reporting of the
capability message; here it appears as **non-additivity across the interfaces
of one DSO**. Now one interface steps at a time, siblings held.

### Result — 240/240 steps, 10 windows x 4 DSOs x 3 interfaces x 2 directions

**88.8 % of band-edge steps (213 of 240) never settle to 1 Mvar within 45
subordinate iterations.**

The decisive follow-up is *how far off* those censored cases end, because a
censored case sitting 1.1 Mvar from setpoint would mean the band decided the
result, while one sitting 30 Mvar away means the setpoint was never reachable.

| final residual of the censored cases | [Mvar] | as a fraction of the commanded step |
|---|--:|--:|
| p05 | 1.84 | 0.031 |
| p25 | 5.60 | 0.070 |
| **p50** | **10.98** | **0.226** |
| p75 | 20.96 | 0.554 |
| p95 | 33.66 | 1.027 |

**It is not a band artefact.** Widening the band rescues little: at 2 Mvar
only 7 % of censored cases end inside, at 5 Mvar 22 %, at 10 Mvar 48 %. (These
are an *upper* bound -- the end-of-window residual does not re-check the "and
stays" condition.) The median censored case ends **11 Mvar**, i.e. 23 % of the
correction it was asked to make, away from a setpoint placed at 95 % of the
capability the subordinate layer itself reported.

**When the loop does settle, it settles fast.** The 27 uncensored counts are
sharply bimodal, with nothing at all between 9 and 29:

```
1 1 1 2 2 2 2 3 3 3 4 4 7 8 9 9  |  29 31 33 33 33 34 35 35 43 43 45
```

16 of 27 settle within **9 iterations** (median 8). The gap is structural, not
sampling noise.

| | n | censored | residual med [Mvar] | step med [Mvar] | band med [Mvar] |
|---|--:|--:|--:|--:|--:|
| DSO_1 | 60 | 0.95 | 10.57 | 49.70 | 102.7 |
| DSO_2 | 60 | 0.92 | 8.04 | 44.84 | 97.3 |
| DSO_3 | 60 | **1.00** | 15.70 | 54.51 | 114.6 |
| DSO_4 | 60 | 0.68 | 2.84 | 45.56 | 96.7 |

Direction is not the discriminator (censored 0.90 down vs 0.88 up). **The tap
actuator is barely used**: 38-53 % of cases move no tap at all across the whole
45-iteration window, and DSO_3 -- the worst area, 100 % censored -- has a
median of **zero** tap moves. Its `trafo_6` recurs in every one of the ten
worst cases, ending ~35 Mvar from a ~55 Mvar step with no tap action.

### What this means for eq. (9.2)

**`N_inner = 9` is supported, conditionally, and it is not the binding
quantity.** Where the subordinate layer can reach the commanded setpoint it
does so within 9 iterations (median 8) -- exactly the guess. What fails is not
the *speed* of the loop but the *reachability* of the setpoint: the reported
capability band systematically overstates what is deliverable, by a median
23 % of the commanded step at the band edge.

That is the same defect as the known voltage-blind over-reporting of the
capability message, measured here directly and at the interface level. It also
explains B2's period-independent censoring floor: ~20 % of dispatch intervals
fail to settle at *every* period because the setpoint, not the time budget, is
the problem.

**Recommendation for the chapter.** Keep `T_TS = 180 s`. The evidence does not
support lowering it (the subordinate layer is not iteration-starved) and does
not support raising it (more iterations do not reduce the censoring floor).
State `N_inner = 9` as sufficient for reachable setpoints, cite the measured
median of 8, and put the reachability defect where it belongs -- in the
capability-reporting discussion, not in the timescale selection.

**The circularity still stands**: `G_w` was calibrated with `N_inner = 9`
assumed, so this is a check on the guess surviving its own consequence, not an
independent measurement. On this evidence it survives.

---

## Open items

1. **Task A: restore RMS/LF consistency** — the one thing standing between the
   reworked emitter and Table 9.1.
2. **The staleness metric** — the sweep scored on the supervisory objective,
   which is the missing half of the `T_TS = 180 s` argument.
3. **Why the reported CAIR collapses to zero width** at some instants (minimum
   0.00 Mvar for all twelve transformers) while its median is 67–181 Mvar.
4. **The SBX-H confound** — decide whether the chapter reports the
   hold-`k_sched` reading or the hold-wall-clock one.
5. **Why the subordinate layer barely taps.** 38-53 % of band-edge steps move
   no tap at all in 45 iterations, and DSO_3 -- 100 % censored -- has a median
   of zero. If the tap is available and would help, the weights are not buying
   it; if it is not available, the capability report should not be counting it.
6. **`N_inner` is answered** (9 is sufficient for reachable setpoints,
   measured median 8). What remains open is the **capability over-reporting**
   that makes 89 % of band-edge setpoints unreachable in the first place.
