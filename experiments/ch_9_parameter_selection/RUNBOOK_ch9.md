# Chapter 9 runbook: A → B → C → D → E → F1 → F2

Written 2026-09-15. Companion to:

* `docs/handoff/2026-09-15_ch9_experiment_order_rerun_plan.md` — the experiment plan this
  runbook executes (section numbers below refer to it);
* `docs/daily_log/09_2026/2026-09-15_ch9_experiment_order.md` — the chapter reordering;
* `docs/daily_log/09_2026/2026-09-15_ch9_paramset_unification.md` — the code change that
  makes this runbook executable, and the evidence found while auditing it.

Every step below resolves its controller as

```python
from configs.paramsets import apply_paramset
from experiments.run_multi_system_ofo import make_config
cfg = apply_paramset(make_config(), "ch9_selected")
```

which is what `_ch9_selected_design.build_selected_config()` now returns, plus the frozen
DSO_3 multipliers. Pass `--paramset stage1_start` for the analytic legs; never edit
`SELECTED_PARAMSET` to switch.

> **Status 2026-09-15.** B is derived on the batch network (`stage1_start_dso3x2`).
> **C is complete**: 120/120 cases, `N_inner` median **20.5**, p95 33.0, max 41, zero
> censored, zero capability-limited. The chapter's `N_inner = 9` is **not supported**.
> An E probe on the same case gives 15–17 against C's 28–29, so the tuned weights are
> ~1.8x faster and still short of 9. E and a plots-off re-run of C are running.
> D must be re-run on the batch network. See
> `docs/daily_log/09_2026/2026-09-15b_ch9_network_decision_and_f1_instrumentation.md`.

## Live plotting must be OFF in every campaign

`make_config()` is the **interactive** runner configuration and sets
`live_plot_controller=True` / `live_plot_cascade=True`. Building the Chapter-9 lineage on
it (2026-09-15) therefore turned them on in every pool worker, where
`live_plot_controller` constructs a matplotlib figure per scenario and redraws it every
step — for nobody.

This is now forced off in two places, so it cannot be inherited by accident:
`build_selected_config(..., live_plots=False)` (the default) and
`tuning.runner.run_scenario_with_records`, which is the shared batch harness and also
covers the Stage-II search. The plotters only read state, so no past result is invalid —
but any leg run with them on differs from a leg run without them in a resolved-config
field that is not the weights, and the C/E comparison requires those to be identical.

## Two decisions taken 2026-09-15, both now defaults

**Network: doubled DSO_3.** `build_selected_config(apply_frozen_network=True)` is the
default, because the Chapter-9 batch is declared on that network and the PowerFactory
project stores it. `--symmetric-network` opts out.

> This is a **stated mismatch, not an oversight**: the Stage-I/II weight campaign that
> produced `ch9_selected` ran on the *symmetric* model, because `make_config()` sets no
> DSO multipliers. The response experiments therefore evaluate the selected weights on a
> network the search never saw. The chapter must say so. Every run banner and every
> `resolved_config.json` records which network was used, and the `summary.md` provenance
> block states the mismatch in words.

**Window bank: W1–W6.** `--bank conditions` is the default, matching the thesis conditions
table and F1/F2, which are already specified on W1–W6. W5 has structurally zero DER
reactive capability at both levels, so it is excluded by `DEAD_ZONE_STRATA` and **five
windows are measured** — which is what the chapter already says about Experiment B.
`--bank design12` runs the twelve `tier1_design_set()` search windows of the
288-planned / 240-measured archive, as a broader coverage check.

---

## 0. The scheduling fact that shapes this plan

**Only A, F1 and F2 need PowerFactory.** B, C, D and E are quasi-steady-state and run on
pandapower alone:

| step | entry point | PowerFactory? |
|---|---|---|
| A (retained, no rerun) | `ch_9_1_timescale_seperation.py` | **yes** |
| B | `tuning/stage1_screen.py` | no |
| C, E | `ch_9_1_ninner_isolated_sts.py` | no |
| D | `ch_9_3_stage2_screen.py`, `ch_9_3_select.py` | no |
| D (cadence check) | `ch_9_1_ts_period_sweep.py` | no |
| F1, F2 | `run_rms_cosim.py` / `run_comparison_rms_cosim_qss.py` | **yes** |

So a blocked licence costs **nothing** on the critical path B → C → D → E. Run that whole
chain locally now; hand over only F1/F2, which the gate in §6 forbids starting before the
chain closes anyway.

### Machines

* **L — local** (`qOFO_2027` host): 20 cores / 40 threads, 256 GB.
  Python `F:\python_environments\qOFO_clean\python.exe`.
  Repo `Z:\Python_Projekte\qOFO_2027` = `\\e5server\homefolders$\mschwenke\Python_Projekte\qOFO_2027`.
* **P — the PowerFactory delegate.** Requirements, all of which must be confirmed before
  anything is handed over:
  1. PowerFactory **2025 SP4** with the Python 3.12 API at
     `C:\Program Files\DIgSILENT\PowerFactory 2025 SP4\Python\3.12`
     (`pf/session.py:DEFAULT_PF_PYTHON_PATH`), and a working licence;
  2. project `qOFO\IEEE39_qOFO` in that machine's PF database — **it is not in the repo**;
  3. a Python 3.12 env equivalent to `qOFO_clean` (`requirements/`);
  4. write access to the share above.
* Run machine P **from the share**, not from a copy: `results/` is gitignored but lives
  inside the repo, so outputs land back on L automatically. Budget for network I/O
  (§7 sizes it).

Smoke test on P before anything else:

```bash
python -X utf8 pf/hello_pf.py
```

The CLI itself is inspectable without a licence on either machine, which is how P's
operator can check a command line before burning a licence slot on it:

```bash
python -X utf8 -m experiments.run_rms_cosim --help
```

**Commit before launching any campaign.** Both Chapter-9.1 entry points stamp
`[WORKING TREE DIRTY]` into `run_meta.json`, and a dirty tree makes a run record
unreproducible — the `git_commit` it names is not the code that ran.

---

## 1. Freeze the two manifests (L, minutes)

Section 3 of the experiment plan: one complete resolved manifest per controller,
differing **in the declared weights only**.

```bash
python -X utf8 experiments/ch_9_parameter_selection/_ch9_selected_design.py --paramset stage1_start --manifest > /tmp/manifest_initial.json
```

```bash
python -X utf8 experiments/ch_9_parameter_selection/_ch9_selected_design.py --paramset ch9_selected --manifest > /tmp/manifest_final.json
```

Diff them. Expected differences: the nine weights, `shunt_int_g_w`, and the three cadence
fields `stage1_start` pins (`tso_period_s` 180, `dso_period_s` 20, `dt_s` 20 — identical
values to `make_config()`, so they are a restatement, not a change). Anything else is a
defect; stop and fix it rather than recording it as an assumption.

---

## 2. B — analytical starting weights (L, minutes)

`stage1_start` already exists. Verify rather than regenerate: rebuild its local screen on
the frozen configuration and confirm the recorded `worst_lambda_max_B = 1.0` and the
42 area-condition values.

**Archive the screen bound actually used.** `stage1_start.json` records
`screen_ceiling: 1.6` and `select_ceiling: 1.5459`; `ch9_selected.json` records
`limits.lambda_max_b: 1.6`. The 1.5 screen of the older campaign and the 1.6 screen here
belong to different campaign records — do not combine their numbers.

No RMS campaign. No simulation.

---

## 3. C — initial subordinate response (L, ~30 min)

The request bank at the **analytic** weights. This is the run the chapter has never had:
there is no verified initial-weight response measurement, and an old final-weight test
cannot be relabelled as one.

```bash
python -X utf8 experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py --paramset stage1_start --label C_initial_weights --workers 16 --max-step-mvar 10
```

Defaults now supply `--bank conditions` and the frozen network; pass them explicitly in
the recorded command if you prefer the run log to be self-describing.

Protocol as coded: 600 s settle, 900 s observe, `T_STS = 20 s`, both directions, step to
0.95 of the reported capability band capped at 10 Mvar, flatness `|Δq| ≤ 0.1 Mvar`.
Grid: 5 windows (W1–W4, W6; W5 excluded) × 4 DSO × 3 interfaces × 2 directions =
**120 cases**.

Cost model: 2120 s of simulated time per case at `dt = 20 s` = 106 QSS steps.
**Measured 2.16 s/step** on this configuration — one 90 min window (`d_quiet_spring`,
`--paramset ch9_selected`, 270 records) took 583 s wall on machine L at one worker, with
another job competing, from the network share. So ≈ 230 s/case, ≈ 7.7 h serial,
≈ **30 min at 16 workers** if parallel scaling holds.

For reference the 12 h Stage-II windows imply 0.91 s/step (mean 1960 s for 2160 steps at
12 workers). The two differ by 2.4×, so treat either as an order of magnitude and read the
real rate off the `--probe` run before committing the grid — the point is that C and E are
each well under an hour, and the licence block therefore costs nothing here.

The 240-case archive was the twelve-window bank. Re-running `--bank design12` for the
coverage comparison is another ~1 h per weight leg if that comparison is wanted; it is a
supplement, not the reported result.

Report, per §5 of the experiment plan: per-group coverage by STS and direction, the
no-band/censoring table, and every clipped or unavailable case — not a pooled percentile.
The old `n = 9` figure is 227/240 = 94.6 %; it is neither 100 % coverage nor a ≥ 95 %
result, and must not be quoted as either.

**Run it first with `--probe`** (one window, one DSO, both directions) to confirm the band
is traversable before spending the grid.

---

## 4. D — reconcile the existing search (L)

Do **not** repeat the Stage-II search to change the chapter's ordering. What genuinely
needs settling:

1. **The factory now agrees; the network deliberately does not.** Since 2026-09-15 C/E and
   the Stage-II search both resolve `apply_paramset(make_config(), ...)`, so shunts,
   boundary equivalent, local sensitivities and zone setpoints are identical across them.
   The DSO_3 multipliers are the one remaining difference and it is **chosen**: the batch
   is declared doubled, the search ran symmetric. State it in the chapter; do not present
   D's aggregate and C/E's response as measurements on one network. If a like-for-like
   comparison is wanted, `--symmetric-network` gives it for ~30 min of compute per leg.
2. **Shunt governance is open.** `results/THESIS_ch9_3_shunt_live/` already re-runs
   `ch9_selected` over all six windows for 12 h with `shunt_int_g_w = 0.03` live. The
   selection verdict survives (e_v(TS) 0.0016515 → 0.0013994, taps and hunting unchanged,
   still feasible), but `aggregate()` carries no shunt field and `Limits` has no shunt
   bound, while W4 shows 49 commits / 32 reversals and W5 shows 112 / 80 per 12 h.
   Decide: add a shunt limit and re-score, or state that bulk shunt switching is outside
   the selection's scope. Do not leave it implicit.
3. **W5's exclusion rationale needs restating.** It is excluded as zero-capability on the
   grounds that no weight can allocate anything there. With the shunt integrator live
   that is no longer true.
4. **The network rerun is now mandatory, and the cadence question is open.** Verified
   2026-09-15 from the code path: `tuning/stage1_screen.py:822` and
   `ch_9_3_stage2_screen.py::_run_one_window` both build on `make_config()`, which sets no
   per-DSO multipliers, and `ScenarioSpec.overlay_on` does not touch them. Stage I and
   Stage II ran **symmetric** throughout. The batch is doubled-DSO_3, so D's 12 h metrics
   have to be re-measured on it.

   Two things make that cheaper than it sounds. The analytic design point barely moves
   with the network — only `kappa_TS` (`g_w_der`, `g_w_pcc`) shifts, by 4.4 %, everything
   else under 0.1 % — so the search box, which is centred on it, is effectively unchanged
   and the existing shortlist is a sound starting set. And C's result means the cadence
   itself is in question, so D should be re-run **together with** a period decision rather
   than at a cadence C does not support.

5. **Cadence**, now required rather than optional:

```bash
python -X utf8 experiments/ch_9_parameter_selection/ch_9_1_ts_period_sweep.py --paramset ch9_selected --label D_cadence --workers 16
```

4 periods (180, 200, 240, 360 s) × 6 windows × 90 min = 24 runs at the measured 583 s
≈ 3.9 h serial, ≈ 15–25 min at 16 workers. If C supports `n = 9`, this is not required
for chronology. `--bank` must match whatever C and E were measured on.

A single-case smoke run on 2026-09-15 (`d_quiet_spring @ 180 s`) returned
ρ median 0.012, ρ p95 0.015, n_k median 0, censoring 0.00, CAIR width 193.1 Mvar —
i.e. the machinery works end to end on the unified configuration. One window is not a
result.

---

## 5. E — period confirmation at final weights (L, ~30 min)

The same bank as C, at the selected weights, so the two are directly comparable.

```bash
python -X utf8 experiments/ch_9_parameter_selection/ch_9_1_ninner_isolated_sts.py --paramset ch9_selected --label E_final_weights --workers 16 --max-step-mvar 10
```

**E cannot be reused from the archive.** Every archived `ninner` run was measured on the
pre-2026-09-15 lineage — MSC/MSR banks absent, PQ boundary equivalent, local sensitivities
off, doubled-DSO_3 network — so it measures a controller that is not the one the chapter
reports. This is a rerun, not a confirmation of a transfer.

Use the same case manifest as C — same `--bank`, same network flag, same
`--max-step-mvar`. Anything that differs between the two legs other than `--paramset`
makes the comparison meaningless.

**The move from the twelve design windows to W1–W6 is a declared scope change**, decided
2026-09-15. The chapter must say that the reported subordinate-response bank is the
conditions table, not the search bank, and must not quote the archived 227/240 = 94.6 %
coverage figure alongside a 120-case result as though they were the same measurement.

---

## 6. Execution gate

> Do not launch F1/F2 while weights, cadence, local-model construction or shunt settings
> are still changing.

Concretely, F may start once: §1 manifests are frozen; C has run and its ratio is stated;
§4.2 and §4.3 have an answer in writing; E passes at that ratio.

---

## 7. P2 — repair and instrument RMS delivery (write on L, verify on P)

This is the prerequisite F1 cannot be run honestly without.

**What is missing.** `csv/rms_der_raw.csv` carries `qDER_*` (delivered Q) and `uDER_*`
(terminal voltage) and nothing else. F1's stated measurement is
`max |U_filtered − U_anchor|` between each device's **actual effective anchor updates**,
and the anchor is not exported. It exists — `Vanchor` on the PF Q(V) pre-controller
(`pf/plant.py:659`) and `qv_vref_anchor_pu` in the static plant (`core/plant.py:142`) —
so this is an export gap, not a modelling gap. Add to
`pf/screening.py:monitored_outputs(..., include_der=True)`:

* the anchor `Vanchor` per park, and its update instants;
* the filtered dead-zone input;
* the **unsaturated** incremental droop command;
* the saturated reference;
* delivered Q (already present as `qDER_*`), kept separate from the above.

Reporting `Q_actual − Q_set` as if it were the droop signal is wrong: converter tracking
and filter dynamics live in that difference.

**Launcher hygiene for new campaigns.** The eight archived `.ps1` scripts under
`experiments/` are the record of past campaigns and are deliberately left as they are. A
new campaign must not reuse them:

* drop `--dso-oltc-switch-cost 200` — it replaces the selected `g_w_dso_oltc = 15.7164`
  with 200 (×12.7) and, being an assignment rather than `dataclasses.replace`, leaves the
  derived per-DSO voltage relief on the parameter set's base;
* pass `--paramset ch9_selected` explicitly even though it is now the default, so the
  command line records the controller;
* keep `--physical-capability`. Without it `der_q_capability_override_pu = 1.0`, which is
  a diagnostic, not a physical model, and no published result may use it.

**One diagnostic event before the F2 bank.** Verify from the plant itself: the realised
event time, load Q before/after, target identity, installed dead band. Identical output
hashes across nominal events are an investigation trigger; differing hashes alone are not
proof an event occurred. The previous F2 exports were identical across events and widths
and are not valid disturbance-response evidence.

---

## 8. F1 — smallest width with no routine droop activation (P)

Twelve mandatory RMS traces, eighteen with the narrower-width comparison.

* **F1-reference**, 6 × 2 h droop-suppressed, OFO active, same cached-model construction
  and update policy as the inactive-droop model. Measures the per-device maximum
  `|U_filtered − U_anchor|` between actual anchor updates.
* **F1-confirmation**, 6 × 2 h with the selected width active **from initialisation** —
  not switched in at an event. Verify zero dead-zone activation and zero incremental
  droop command within declared tolerances. Report zero-headroom devices separately:
  saturation must not hide an active dead-zone command.
* Recommended: 6 more at the adjacent narrower width, to show the onset of interaction.

Start the six traces at the **W1–W6** timestamps and run uninterrupted for 2 h, rather
than cold-starting at the later hours. `CONDITIONS` in `tuning_mc/scenarios_mc_v2.py` is
the tracked source of those timestamps:

| | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| start | 2016-01-05 08:00 | 2016-02-22 13:00 | 2016-01-04 08:00 | 2016-11-26 10:00 | 2016-02-03 20:00 | 2016-07-18 14:00 |
| stratum | full | full | partial | partial | **none** | full |

Declare the startup exclusion **before** reduction, apply the same rule in every leg, and
report startup separately. Do not infer convergence from 180 s having elapsed. Freeze the
numerical/measurement margin and the pass tolerances before comparing candidates. Do not
preselect 0.01 pu — start from the former ladder as a comparison and refine around the
measured maximum. A trace with no ordinary switching cannot establish switching
non-interference; extend the duration or declare an ordinary operating scenario rather
than removing the actuator.

Use **`run_rms_cosim.py`**, not the comparison script: F1 is RMS-versus-RMS, the QSS twin
is not the reference, and dropping it roughly halves the wall time.

```bash
python -X utf8 -m experiments.run_rms_cosim --paramset ch9_selected --physical-capability --duration 7200 --stride 100 --scenario rural_700 --start-time "2016-01-05 08:00" --profiles --profile-delivery elmfile --tso-deadband 0.5 --dso-deadband 0.5 --no-pdf --verbose 1
```

`--stride 100` is 1 s rows. F1 measures slow drift between anchor updates, so 1 s is
ample and it cuts `rms_der_raw.csv` from ~310 MB to ~31 MB per trace — 12 traces is then
~0.4 GB on the share instead of ~3.7 GB. **F2 must stay at `--stride 10`**: it measures a
post-event peak.

**Calibrate before committing the bank.** The only measured figure is ~25 min/run at
600 s for the *comparison* script (2026-08-02). `run_rms_cosim` is roughly half that, and
a 2 h trace is 12× the simulated time of a 600 s one, so the plausible range is wide.
Time one full F1-reference trace on P and multiply; do not plan from the estimate.

---

## 9. F2 — disturbance demonstration (P)

**6 windows × 3 disturbances × 3 legs = 54 RMS cells**, including all nine W5 cells and
their references. W5 is reported as a zero-capability limitation, not dropped.

* Disturbances: +200 Mvar at TS bus 7; +200 and −200 Mvar at TS bus 11
  (`--q-step-bus` / `--q-step-mvar`, which leave topology unchanged).
* Legs: F1-selected width, one wider width, droop disabled.
* Fixed slope and converter ratings; `S_n/m` with capability saturation kept separate.
* Event time by a **stated phase** relative to actual dispatch/re-anchoring — e.g. 0.5 s
  after a supervisory update, following a common run-up. Record the realised time.
* The candidate width is active throughout the operational run. An event-time switch is a
  separate instrument and must be labelled as one, with a continuous-operation
  confirmation retained.
* A −200 Mvar step that would make a nominal load capacitive must be checked and described
  physically, not silently clipped.
* Continue three legs of two representative cases for ≥ 3 supervisory periods past the
  event to show OFO handover; those six extended cells are part of the 54 if planned
  before execution.

```bash
python -X utf8 -m experiments.run_comparison_rms_cosim_qss --paramset ch9_selected --physical-capability --duration 600 --stride 10 --scenario rural_700 --start-time "2016-01-05 08:00" --profiles --profile-delivery elmfile --tso-deadband 0.01 --dso-deadband 0.01 --q-step-bus 7 --q-step-mvar 200 --trip-time 200 --no-pdf --verbose 1
```

Two reading rules for whoever runs machine P:

1. **Exit code 1 is expected.** `run_comparison_rms_cosim_qss` returns
   `0 if gate_ok else 1`, and Gate E (QSS↔RMS equivalence) legitimately fails through a
   disturbance. A genuine failure looks different: **no `rms_records.pkl`**. That is the
   check to apply.
2. Metrics to export per cell: actual dead-zone activation delay, unsaturated and
   saturated incremental command, delivered incremental Q, available directional
   headroom, time-weighted `|U − U_pre|` through the next subordinate dispatch, and the
   **unblanked** voltage extrema and time outside the corridor. If post-switching
   blanking is useful, report it as a separate metric and keep the raw extrema. Quote
   ratios against the no-droop leg only where the denominator is meaningful; always keep
   the absolute values.

No universal 25 % recovery condition is imposed. If a mandatory contingency requirement is
adopted later it must be declared explicitly and evaluated without retrospectively
changing F1.

---

## 10. Order of operations

```
L, now        1 freeze manifests → 2 B verify → 3 C (probe, then 120 cases)
L, then       4 D reconcile (shunt governance + W5 wording; cadence only if C moves it)
L, then       5 E (120 cases)
              ── gate §6 ──
L, meanwhile  7 P2 instrumentation (written without a licence, verified with one)
P, on licence 7 one diagnostic event → 8 F1 (12, or 18 with the narrow comparison)
P, then       9 F2 (54 cells)
              10 update tables from the complete manifests; keep old evidence historical
```

Steps 1-5 are roughly 1.5 hours of compute on L and need no licence. Nothing on P can
honestly start before the gate, so the licence block is not on the critical path.
