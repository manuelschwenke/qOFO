# Declaration: Case-2 scenario F, area-2 deficit far beyond its own TS resources (written before any run)

Date: 2026-11-05, 22:35, before the first run of scenario F or FM (committed in 3197fc1, launched 22:37).
Request (author, 2026-11-05 evening, while the E rerun with fixed machine taps was starting):
"Could we even have a more drastical scenario F with e.g. two subsequent 400 Mvar steps?",
"Maybe this would also fit in the night simulation plan", "A scenario where we see a huge
difference and really need sts resources".
Decided by the bound below; the author has not seen the numbers before the launch.

## Why a further scenario

- E's deficit is restorable with 17 % of the area-2 STS DER capability (bound on the current
  base, below). E shows the cascade gain, but STS access is a modest top-up there.
- F tests the same mechanism at a deficit that only STS access can close. It is a declared stress
  test, not a typical operating case; A stays the illustrative scenario and E the reference run of
  ch:case3:nominal.
- Two steps instead of one make the run its own control. After the first step area 2's own TS
  resources still suffice; after the second they do not.

## Schedule

| time | event | element | phase in the 180-s TS period |
|---|---|---|---|
| 30:40 | G3 out (area 2's only machine, with its machine transformer) | gen 1 | 40 s |
| 121:40 | line 9–39 out (area 2's tie to the G1 equivalent) | line 14 | 100 s |
| 212:20 | **step 1:** 400 Mvar connected at IEEE 4 (location rule, as D and E) | load bus 3 | 140 s |
| 271:10 | **step 2:** 400 Mvar connected at IEEE 8 (load bus of B) | load bus 7 | 70 s |

- Up to 212:20 the schedule equals E's, so F reproduces E's first 212 min (same seed, period and
  weights); the runs differ only from step 1 on.
- Step 2 is 59 min after step 1 (settling p95 of the v4 weights: 23 min) and leaves 89 min to the
  end of the 360-min horizon.
- **Two buses.** The second step cannot go to IEEE 4 as a second 400-Mvar connect: connect events
  with equal (bus, P, Q) share one dormant load (`experiments/helpers/contingency.py`), so the
  second would do nothing. IEEE 8 is the load bus of B (declared 2026-10-27), so no new location
  choice is introduced.
- Weights `ch9_selected_uc_2026_11_v4` (061dadc05755); D1 dead band 0.010 pu (run 0014); machine
  taps fixed in L2, S1, S2 (`fix_machine_taps`); noise seed 20261027; all six variants.
- **F is reported whatever the outcome.**

**FM (fallback, declared now):** the same schedule with 2 × 350 Mvar. It runs at the same time on
e5power. It is reported if a variant of F does not complete, and otherwise next to F as a milder
point of the same mechanism.

## Scarcity bound (plant model only; no controller result used)

Scripts and outputs are in the shared notes folder, `sts_access_gain_scripts/`:
`scarcity_F_bounds2.py/.out`, `scarcity_F_seq_frozen.py/.out`, `scarcity_F_alpha_warm.py/.out`.

- **Start point.** The runner's plant at t = 0 of the REF period, with the 360-min commitment
  (G3 committed), current code base (`res/commitment-base-2026-10-31`).
- **Bounds.** The definitions of E's declaration: TS-a2 (area 2's own TS resources at their
  raising limit), TS-all (every TS resource at its limit, STS DER frozen), +STS α (TS-a2 plus the
  area-2 STS DER at α · Q_max(P), coupler taps frozen).
  - "Level" is the area-2 mean deviation from 1.03 pu, in mpu.
  - The area-2 STS DER capability is 754 Mvar.
- **Warm start.** Each state is solved from the state before it, as the runner meets each event.
  - A cold start from t = 0 with both steps applied diverges for every candidate of 700 Mvar or
    more (`scarcity_F_bounds2.out`).
  - Every warm-started state converges.

| end state (after G3, 9–39) | frozen, banks out (min V) | TS-a2 level (min V) | TS-all level | STS to restore | STS V max there |
|---|---|---|---|---|---|
| E: 600 @ 4 | −73.8 (0.924) | −28.5 (0.964) | +26.2 | α = 0.17, +132 Mvar (17 %) | 1.067 |
| step 1 of F: 400 @ 4 | −37.4 (0.968) | **+12.0** (1.011) | not computed | — (not scarce) | — |
| FM: 350 @ 4, 350 @ 8 | −108.9 (0.893) | −67.7 (0.941) | −16.4 | α = 0.41, +314 Mvar (42 %) | 1.081 |
| **F: 400 @ 4, 400 @ 8** | −132.4 (0.865) | **−89.6** (0.914) | **−41.2** | α ≈ 0.55, ≈ +420 Mvar (≈ 56 %) | ≈ 1.09 |

For F, the α row is interpolated between α = 0.5 (level −8.0, +382 Mvar) and α = 0.6 (+6.7, +457 Mvar).

Reading:
- After step 1, area 2 can restore its level with its own TS resources.
- After step 2, no controller of area 2 without access can. The area-2 minimum stays below
  0.95 pu even at the TS-a2 bound.
- Even every TS resource of all areas (TS-all) leaves the level 41 mpu low.
- With access, α ≈ 0.2 (about 155 Mvar) already lifts the area-2 minimum above 0.95 pu. Full level
  restoration needs about half of the STS DER capability, at STS voltages within 1.10 pu
  (`dso_v_max_pu`).

## Expected results (hypotheses, written before the run)

1. **Between the steps (212:20–271:10):** OFO+ − OFO in area-2 e_v is small (|Δ| ≤ 2 mpu), as in
   the non-scarce B/D (TS-a2 bound +12 mpu).
2. **After step 2 (271:10–360):**
   - OFO's area-2 level error is ≥ 40 mpu, with area-2 voltages below 0.95 pu.
   - OFO+ keeps it ≤ 15 mpu.
   - OFO+ − OFO in area-2 e_v is ≤ −25 mpu.
   - The margins to the bound allow for the bound's t = 0 operating point and for OFO trading the
     level against the spatial shape.
3. **SVR+ − SVR < 0** after 271:10, and larger in magnitude than in E.
4. **LOC** reaches area-2 minimum voltages near or below 0.90 pu after step 2 (frozen bound
   0.865–0.909 pu). Its run may not converge; a non-converged run is a result and is reported as
   such.
5. **No sign is predicted for OFO+ − CEN.** CEN has the neighbouring areas' machines, but the
   TS-all bound (−41 mpu) shows that they alone do not suffice.

If a result contradicts a hypothesis, that is reported as such.

## Limits of the statement

- **800 Mvar is very large.** It is several times the base reactive load of IEEE 4 and IEEE 8, and
  F is a stress test of the scarcity condition. Cold-start divergence and frozen minimum voltages
  of 0.87–0.91 pu show that the post-step states lie near the solvability limit of the plant model.
- **The bounds are static and optimistic for OFO** (one operating point, taps frozen).
- **One period, one noise seed.** The cross-scenario statement is left to the Monte Carlo
  campaign.

## Amendment 2026-11-06, 07:25: step 2 did not fire in runs 0020/0021; moved onto the step grid

- **What happened.** Runs 0020 (F) and 0021 (FM) completed with all six variants "ok" (pulled
  02:21–02:22). But step 2 never fired:
  - The runner fires an event only at a step whose time equals the event time
    (`experiments/runners/multi_tso_dso.py:3877`, tolerance 1e-9 s; step `dt_s` = 20 s).
  - Step 2 was scheduled at 271:10 = 16270 s, which is not a multiple of 20 s. The first three
    events are at 1840, 7300 and 12740 s, all on the grid.
  - Evidence: every variant's `events_fired` lists only three events. The area-2 load Q is
    558.9 Mvar at 268 min and 558.4 Mvar at 275 min (F, L2).
- **What 0020/0021 are.** E's schedule with 400 Mvar (0020) or 350 Mvar (0021) at IEEE 4 instead
  of 600 Mvar. They are not runs of F or FM and are not reported as such. They are kept as runs of
  a single 400 / 350 Mvar step; a note in each run folder says so.
- **What was seen before this amendment.** The windowed metrics of 0020/0021
  (`sts_access_gain_scripts/F_windows.py/.out`), i.e. the behaviour after step 1 only.
  - Hypothesis 1 can be read from them, because the rerun reproduces the first 271 min (same
    seed, period, weights; the windows before step 1 equal E's run 0019 to 0.1 mpu):
    O3 − O2 in area-2 e_v between the steps is +0.1 mpu (F) and +0.1 mpu (FM), within ±2 mpu.
  - Hypotheses 2–5 concern the state after step 2, which no run has produced yet. They stand
    unchanged.
- **Change.** Step 2 moves from 271:10 to **271:00** (16260 s, phase 60 s): on the step grid,
  10 s earlier, and still at a phase distinct from the other events. Loads, buses, weights, seed
  and the FM fallback rule are unchanged. The static bounds do not depend on the 10-s shift.
- **Guard.** `ref/scenarios._at` now rejects a time off the 20-s grid (`DT_S`), and a test checks
  that every event of every scenario lies on it. The MC campaign's event phases (20–160 s in
  20-s steps) are on the grid.
- **Reruns.** F and FM again on e5power, tags v4_F_r2 and v4_FM_r2 (run numbers in the daily log).
