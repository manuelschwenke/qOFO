# 2026-10-07: Part III Monte-Carlo driver, reference-period characterisation, S1g

**Author:** Manuel Schwenke / Claude Code. **Time:** 09:00–11:50 CEST.
**Thesis side:** `latex_diss_ms/00_daily_log/2026-10-07_mc_harness_reference_period.md`.

## What changed

### 1. New: `experiments/part3_mc/run_mc.py` (+ `README_server.md`)

**Why.** The paired statistics of Case 1 (S1 vs O2) and later of Case 2 need an ensemble that tests the Chapter-8 controller. The old `ch_10_case_study/ch_10_6_ladder_mc.py` did not do that:
- it built its configuration with `make_thesis_config` and the old pinned `g_w_gen = 1e9`;
- it gated on all seven variants, which rejected two thirds of the draws;
- it took the HV networks' bus lists (`hv.bus_indices`) as complete, so its load-bus catalogue contained 110 kV buses.

**Structure.**
- **Episode = seed.** `draw_episode(seed)` gives the start time and events. It is deterministic, and every variant task re-draws and asserts it.
- **Prescreen task.** L2, no events, dt = 180 s. The JSON is always written, and a task crash counts as an invalid draw.
- **Variant task.** One (seed, variant) pair.
  - Outcomes: `ok`, `fail_unreachable`, `fail_abort`, `fail_pf`, `fail_v_end`, `fail_osc` (tap reversals ≥ 2 in the last 30 min).
  - Metrics come from `report_case1.area_metrics` with the absolute range [0.95, 1.05], plus `epost_max`.
  - The compact npz series are stored.
- **Phase 1** prescreens in seed order until the first N valid seeds are known. **Phase 2** runs the variants, longest first, and is resumable.
- **Pool.** `_pool_map` rebuilds a broken process pool and requeues the tasks that were in flight.
- **Element catalogue (`enumerate_elements`).** Transmission elements are selected by nominal voltage ≥ 200 kV. A line is admissible only if the 345 kV graph stays connected without it. Result: 33 of 37 lines (excluded: 21, 29, 35, 36), 5 machines (slack never), 20 load buses.
- **Report.**
  - Outcomes per variant, excluding infeasible draws, with one-sided Clopper–Pearson 95 % bounds.
  - Medians and p95 per variant and area.
  - Paired O2 − S1 per area:
    - non-inferiority on `e_vp`: bootstrap 95 % upper bound of the median < δ = 1 mpu, plus a Wilcoxon test on d − δ;
    - superiority on `e_v`, `worst`, `t_out`: one-sided Wilcoxon with Holm correction over 9 tests.
- **Pre-flight.** A Gurobi licence check, so O2 cannot silently fall back to SCIP. Provenance (git head, status, diff, constants, element catalogue) is recorded per launch.
- **Declared constants.**
  - Master seed 20261007; N = 100; horizon 180 min.
  - Slots of 30 min with p = 0.25; every event reversed after 90 min.
  - Reactive load step: P = 0, Q ~ U[100, 400] Mvar.
  - The days of W1–W6 are excluded.

**Smoke test.** `--smoke --jobs 4`: 2 episodes × 10 min, L2/S1/O2 all `ok`; wall time L2 0.3 min, O2 0.4 min, S1 1.6 min (incl. calibration).

### 2. New: `experiments/reference_period/characterise_reference_period.py`

A one-off characterisation of 05 Jan 2016 08:00 + 360 min (thesis `ch:setup:scenarios:reference`).
- **Exogenous inputs** are computed from net + profiles without a power flow. Buses are classified as TS area or STS through the line-connected components of each 110 kV network, because `hv.bus_indices` is incomplete.
- **Power-flow quantities** come from the K1 L2 log.
- **Outputs** (in `results/THESIS_reference_period/`): CSV/JSON/MD and the figure.
- **Checked against the log.** STS load and DER agree to about 1 %. The TS totals differ by the ZIP anchoring: the runner divides the setpoints by 1.03, while the script reports the power served at the anchor voltage.

### 3. Changed

- **`experiments/case1_svr/report_case1.py`:** `t_out` now measures time outside an absolute range, `--v-lo 0.95 --v-hi 1.05` (it was `--band 0.025` around v_ref). Declared by the author on 2026-10-07.
- **`experiments/case1_svr/figures_case1.py`:** the legend of the equivalence panel (b) moved below the axes, because it covered the S1g curve.

### 4. Runs

- **`results/THESIS_case1_svr/20261006_K1/S1g`** completed, 1080 records, wall time 146 min.
  - **Why:** S1g separates the safeguard from the P term.
  - **Result:** the safeguard alone removes the post-event withdrawal (max growth 1.3 mpu, against 16.7 mpu for S1m; S1: 0.5 mpu), at AVR travel 0.34 pu (S1m 1.37, S1 0.35).
  - **Incident:** a duplicate S1g launch at 11:31 was killed after 15 s. My mistake: the original's buffered stdout looked stalled. The only effect is that the duplicate rewrote `config_repr.txt` with identical content.
- **Profiling of SVR runtime:** `profile_S1_10min`, `profile_O2_10min`.
  - In a 10-min run, the one-off calibration takes about half the time.
  - Per step, S1 needs two plant advances (the RPR acts every step), each a run_control Q(V) fixed point of about 40 PFs. About 70 % of each PF is `_pd2ppc`. O2 skips the end-of-step PF on 8 of 9 steps.
  - S1g's console shows coupler-OLTC reverts by `_OLTCRateLimiter` on 848 of 1020 steps after the gen trip, which adds PFs. Not changed.

## Open

- Run the campaign on the server; see `experiments/part3_mc/README_server.md`.
- After S2/O3/M are implemented, rerun three S1/O2 seeds to check that they reproduce identically.
- Settling-based failure criterion: evaluate offline from the npz series.

## Afternoon (14:00–): S1 integral-only; reference-figure lines

**Why.** The author decided to drop the PI form of S1 from the thesis and the investigation. On the reference run the P paths add little once the safeguard is on (pilot 4.8 vs 5.0 mpu, same AVR travel), and the integral form makes S1⁻ → S1 a one-mechanism step.

**`experiments/case1_svr/run_case1.py`**
- `VARIANTS["S1"]`: K_p = 0 on both loops, guard on (was K_p = 0.2). `S1g` is removed (identical to the new S1).
- `PARETO`: `S1_T112` (112.5 s, b = 1.6), `S1_T137` (137.4 s, b = 1.31), `S1_T600`.
- New `LEGACY` dict (`S1PI`, `S1PI_T150/T198/T600`): resolvable by `resolved()`, reported via `K1_EXTRA`, never part of the thesis variants.
- `STEP_ALLOWED`: (S1m, S1) = {guard}.
- **Verified:** resolved configs equal the stored `config_repr.txt` for S1 (ex-S1g), S1PI (ex-S1), S1PI_T150 and S1m.

**`results/THESIS_case1_svr/20261006_K1`** (local and share): folders and console logs renamed: S1 → S1PI, S1g → S1, S1_T150/T198/T600 → S1PI_T150/T198/T600. The note `RENAMED_2026-10-07.md` records the mapping.

**`experiments/case1_svr/figures_case1.py`:** S1 label "integral + safeguard"; S1g removed; the front is S1_T112/T137/S1/T600.

**`experiments/reference_period/characterise_reference_period.py`**
- New `tripped_gen_series()`.
- Panel (a) shows G_4 alone and the total synchronous generation minus G_4 ("other machines + equivalent"), plus the equivalent alone.
- Disturbances are numbered above the panel.

**`experiments/part3_mc/README_server.md`:** states the S1 definition and a check.

**Runs.** S1_T112, S1_T137, S1_T600 (360 min, reference period) started at 14:09. The smoke test of `run_mc` was rerun with the new S1: L2/S1/O2 all `ok`.

All files are synced to the share.

## 9. Evening (17:19–17:40): integral S1 front complete, Ch 9 §3 filled

**Runs.** S1_T112, S1_T137 and S1_T600 completed: 1080 records each, wall time 177–190 min each with three in parallel on the laptop. Afterwards `report_case1` was run and both Case-1 figures regenerated. Results synced to the share.

RMS over the areas [mpu]; per-area pilot errors as area 1 / 2 / 3:

| Variant | T_RVR | b | AVR travel [pu] | pilot | all-bus | per-area pilot |
|---|---|---|---|---|---|---|
| S1_T112 | 112.5 s | 1.6 | 0.383 | 3.5 | 11.7 | 3.0 / 5.0 / 1.4 |
| S1_T137 | 137.4 s | 1.31 | 0.368 | 3.7 | 11.7 | 3.4 / 5.2 / 1.4 |
| S1 (deployed) | 300 s | 0.6 | 0.339 | 5.0 | 11.9 | 5.5 / 6.3 / 2.3 |
| S1_T600 | 600 s | 0.3 | 0.295 | 6.8 | 12.4 | 7.6 / 8.2 / 4.0 |
| O1 ×0.01 | | | 0.409 | 4.2 | 9.3 | 5.6 / 4.2 / 1.9 |
| O2 (all tunings) | | | | 3.8–4.0 | 6.2–6.6 | area 1: 5.5–5.7 |

**Finding (one trajectory, design window W1).**
- The integral classical scheme tuned hard is tighter on the pilot than the PI front was (PI at 150 s: 4.0 mpu at 0.46 pu).
- The OFO front therefore stays behind on the pilot error:
  - O1: 0.5–0.7 mpu behind at comparable effort;
  - O2: 0.3–0.5 mpu behind, which is within δ in aggregate;
  - area 1: 2.5 mpu behind.
- O2 beats the deployed S1 on the pilot (4.0 vs 5.0 mpu) at 1/40 of the travel.
- Over the whole reference set, the OFO is better at every effort, and O2 nearly halves the error.
- The Ch 9 fourth discussion statement was revised accordingly.

**For the MC.** On this run, O2 − S1(deployed) in area 1 is +0.02 mpu. The non-inferiority test against the deployed comparator is therefore likely to pass there, but with little room.

## 10. Figures in TU colours (author request, 17:50)

All three new figures now use the TU Darmstadt corporate palette:
- the thesis preamble defines it (`ClassicThesis.tex`: `tured`, `tuamber`, `tuteal`, ...);
- `configs/color_config.py` mirrors it.

**Variant colours.** These follow the validated Part III palette of `ch_10_case_study/ch_10_2_figures.PALETTE`:

| Variant | Colour |
|---|---|
| S1 | red #B90F22 |
| O1 | amber #D28700 |
| O2 | teal #008877 |
| O3 | dark blue #004E8A |
| L2 | grey |
| S1⁻ = O0 | TU purple #611C73 (11c; not used in that palette) |

**Reference-period figure.** Colours by quantity:

| Quantity | Colour |
|---|---|
| other machines + equivalent | dark blue |
| equivalent alone | mid blue, dashed |
| G_4 | red |
| wind | olive #7FAB16 |
| photovoltaic | gold #D7AC00 |
| exchange | magenta #951169, dashed |
| load | black |

**Changed files.** `figures_case1.py` and `characterise_reference_period.py`, regenerated into `graphics/case_svr/` and `graphics/setup/`.
