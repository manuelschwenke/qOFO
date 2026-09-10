# 2026-09-08 — Ch. 9 PowerFactory batch, Phase 0: network provenance and a broken design guard

## Purpose

Phase 0 of the Chapter-9 RMS/QSS batch (dispatch periods, droop dead band,
controller weights) gates every later phase. It asks three questions: which
network Experiment A ran on, which network the weight campaign evaluated on, and
what the one frozen configuration is. All three were answered read-only; no
simulation was started.

## Established facts

**Experiment A ran on the un-doubled network.**
`export/snapshots/full_t0_20160105-0800.json` has DSO_3 identical to DSO_1/2/4
in `sn_mva` (700.0), sgen `base_p_mw` (700.0), load `base_p_mw` (498.7217) and
`total_ref_p_mw` (261.80375). 700 MVA is the unscaled `rural_700` nameplate
(250 wind + 240 PV + 3 x 70 coupling). `apply_dso_overrides` scales those
columns in place, and all `scaling` entries are 1.0, so the override is not
hidden elsewhere. The sweep's `run_meta.json` names this file by SHA-256
(`9379a396...3266b`) and the file on disk matches.

**`campaign_0815` evaluated on the un-doubled network**, and so does the
actually-selected campaign. `stage_1_search.evaluate_one` goes
`load_config_yaml(baseline)` -> `build_config` -> `ScenarioSpec.overlay_on`;
the baseline `tuning/scripts/configs/baseline_ieee39_thevenin.yaml` carries
`dso_der_scale: {}` / `dso_load_p_scale: {}`; neither overlay touches those
fields; `tuning/` and `tuning_mc/` contain no assignment to them and no CLI flag;
and `multi_tso_dso.py:552` applies overrides only for non-empty maps. The 0.8
ratio in `window_selection.json` is the DSO DER nameplate ratio 2800/3500 —
per-MVA capability is 0.74 on both readings.

**The C1 RMS runs, by contrast, ran doubled.** Every `config.json` in
`results/rms_phase6_replay/` records `dso_der_scale = dso_load_p_scale =
{DSO_3: 2.0}`, with `g_w_dso_der = 800` / `g_w_dso_oltc = 200.0` (runs
0421–0566) or 1097.2 / 183 (0567–0570). So the weights, Experiment A and the C1
lineage do not share one network.

## Defect found: the relief default leaks into the Stage-0 design

`_ch9_selected_design.py` no longer reproduces its archived candidate:

```
g_w_dso_der:  rebuilt 552.532902778696   != archived 549.881810527467
g_w_dso_oltc: rebuilt 1241.6505006240188 != archived 392.6443639860221
```

Mechanism. Commit `ccc8a54` gave `MultiTSOConfig.dso_v_relief_factors` the
non-empty default `{"DSO_2": 10.0, "DSO_4": 10.0}`, installed by `__post_init__`
on every construction. `design_payload` runs `stage_0_preconditioning` as a
subprocess against the same YAML baseline, so Stage 0 now designs against an
already-relieved config; `build_config` then applies the relief a second time.

Measured from the preserved payloads: DSO_2 and DSO_4 per-area OLTC weights are
exactly x10.0000 while DSO_1 and DSO_3 are bit-identical, so the geometric-mean
global moves by 10^(2/4) = 3.1622776601683777, i.e. sqrt(10) to 15 significant
figures. TSO-side weights are unchanged, which confines the defect to the DSO
block.

The `stage_0_preconditioning.py` diff across that commit is comment-only. Only
its SHA-256 fingerprint changed, which invalidated the design cache and wrote
the contaminated payload to
`results/tuning_mc/stage1/designs/stage0_fe010aa3ead1.json`. The cache's
supersede-don't-overwrite rule preserved the two earlier payloads, which is the
only reason the original design is still recoverable.

Fingerprint map (each blob hashed from git):

| commit | date | fingerprint | design |
|---|---|---|---|
| `53865c3` | 2026-08-18 | `bca4dedc80a9` | 549.8818 / 392.6444 |
| `5d8e0f3` | 2026-08-21 | `34729bd4f470` | 549.8818 / 392.6444 |
| `ccc8a54` | 2026-08-24 | `ea11afdf8a2a` | 552.5329 / 1241.6505 |

Because `_ch9_selected_design.py` raises rather than proceeding, every §9.1
experiment importing it is currently blocked, not only this batch.

## What was changed

At the time of the Phase-0 report: nothing in the simulation or weight-design
code (the Stage-0 fix below came after the author's sign-off). Added, all new
files:

- `results/THESIS_ch9_PF_batch_2026-09-08/README.md` — the Phase 0 report;
- `results/THESIS_ch9_PF_batch_2026-09-08/phase0/export_frozen_cfg.py` — a
  read-only export that pins the Stage-0 payload to the preserved
  `stage0_fe010aa3ead1.json.fp_34729bd4f470`, re-runs every assertion
  `_ch9_selected_design.py` makes (all pass), and dumps the resolved config;
- `results/THESIS_ch9_PF_batch_2026-09-08/phase0/frozen_config.json` — its
  output: periods `T_TS = 180 s` / `T_STS = 20 s`, the six weight classes, the
  x10 DSO_2/DSO_4 relief, the 828.3682036634343 DSO_4 damping correction, droop
  slope 0.06 pu and dead band 0.01 pu, actuator bounds, sensitivity policy and
  solver settings.

## Decisions taken and implemented (same day)

Both gating questions were answered by the author:

1. **Freeze the un-doubled network** (`dso_der_scale = {}`).
2. **Fix the Stage-0 relief leak at source**, payload pin as interim.

`tuning_mc/stage_0_preconditioning.py` now strips `dso_v_relief_factors` on the
YAML-baseline path before designing -- and deliberately not on `--from-runner`,
where the per-area overrides are the object of study. The factor map is cleared
in the same `dataclasses.replace` as the generated maps, because `__post_init__`
runs on every copy and would otherwise reinstall what was just removed.

`build_selected_config()` builds again and reports `archived weights reproduced;
DSO-4 correction asserted`. The regenerated payload (fingerprint `f96fb3bda5da`)
is equal as parsed JSON to the preserved `fp_34729bd4f470`, so the fix restores
the archived design exactly rather than merely satisfying the assertion.
`tests/test_dso_v_relief_pairing.py` 20 passed / 6 skipped;
`tests/test_gw_precondition.py` 17 passed. The full run of
`tests/test_dso_v_relief_pairing.py tests/test_gw_precondition.py tests/tuning/`
finished 1 failed, 202 passed, 6 skipped (52 min).

The one failure,
`tests/tuning/stability_certificate/test_hierarchy.py::test_default_factory_reads_run_multi_system_ofo_parameters`,
is **pre-existing and unrelated**: it asserts `config.g_w_der == 50` and gets
14.4, which is the value `make_config()` sets at HEAD
(`experiments/run_multi_system_ofo.py:305`) -- the only uncommitted change to
that file is a contingency minute. The test imports one symbol from
`tuning.stability_certificate.hierarchy` and never reaches
`stage_0_preconditioning` (verified: the module is absent from `sys.modules`
after importing it), and the Stage-0 edit lives inside `main()` behind
`if not args.from_runner`, so it cannot run at import time. A stale expected
value left by the weight refactor; flagged separately, not fixed here.

## Two measurements that changed the reading

**The network choice does not move the converter margins.** Doubling DSO_3 at W1
raises DSO_3's own worst converter voltage by +0.0009 pu and leaves DSO_4 --
where the binding converter is -- unchanged at four decimals, because
`dso_load_p_scale` doubles the load in the same step. The decision matters for
the DER Q capability totals and hence the weight design, not for R1.

**The supplied calibration margins are droop-suppressed measurements.** With the
droop active the binding converter at W2 is a subordinate DER; with the dead band
at 0.5 pu (the C1 state) it is `WP_STATCOM|grid_bus24`, the calibrated device,
and W1 keeps `DSO_4|Wind_1`, also as calibrated. The specification's own
0.5 x 0.0355 = 0.0177 pu confirms it. R1 is therefore evaluated on the
droop-suppressed state.

## Pre-run deliverable: W1-W6 initial states and margins

Six conditions solved in both droop states, exported to
`results/THESIS_ch9_PF_batch_2026-09-08/phase0/initial_states{,_droop_suppressed}/`.
Droop-suppressed minima: W1 0.0311, W6 0.0414, W3 0.0536, W4 0.0763, W2 0.0774,
W5 0.0906 pu, all on the upper side.

- **No condition is below the 0.02 pu flag threshold**, so the 0.01 pu dead-band
  candidate stands.
- **W1 is the binding condition, not W6.** Binding R1 bound `dV_db <= 0.0156 pu`.
  That already removes 0.02 pu from the C1 candidate set and leaves 0.015 pu
  within 0.0006 pu of its bound.
- The computed DSO Q ranges reproduce `window_selection.json` exactly (2072.0 /
  738.5 / 710.4 / 0.0), an independent confirmation that the screening ran on the
  un-doubled network.

## Phase 1 inputs

Five snapshots generated for W2-W6 (`export/make_snapshots.py`, round-trip
verified). `make_snapshots.py` has no doubling path at all, which independently
confirms the Item 1 finding. All twelve Phase-1 case identifiers resolve in the
archived W1 sweep, and every quoted settling time reproduces -- but from two
different columns: `results.csv::t_settle_s` is the superseded pooled maximum,
and the quoted values live in `results_voltage_only.csv` and
`results_by_family.csv::t_settle_dso_q_s`. Checked over all 70 archived cases:
worst controlled `max(V, TS-STS interface Q)` is 12.472 s, while the excluded
inter-area tie-line Q already reaches 17.172 s -- 2.83 s from `T_STS = 20 s`.

## Phase 1 blocker found by the smoke test

`ch_9_1_actuator_location_sweep.py` never pushes its `--snapshot` into
PowerFactory. The argument feeds `load_snapshot_to_pandapower` for the pandapower
QSS screen and the candidate/output construction; the RMS half only calls
`connect(project, study_case=RMS_STUDY_CASE)`. The script has zero references to
`pf_sync`, and `sync_full` is called only from `pf/replay.py` and `pf_sync.py`'s
own CLI.

Measured confirmation: preflight worst drift ratio `0.9807497617988759` on
`Q_TSO_Z2_Z1_TN_line14` for the archived W1 run, for W2 and for W5 alike --
bit-identical across three profile timestamps. W5 is the extreme point (56
candidates against W2's 144, all 44 converters at zero Q capability) and still
returns the same sixteen figures. The plant does not change with `--snapshot`.

Had Phase 1 run as specified, all 53 RMS runs would have shared one operating
point while appearing normal in every output. The W3-W5 rows in particular exist
to probe reduced and zero converter capability, which lives entirely in the plant
state. The archived W1 sweep is unaffected: there the snapshot and the stored PF
state coincide.

Remedy: `python -m pf.pf_sync <snapshot> --phase full --study-case 02_RMS_CoSim`
before each condition. Not done unilaterally -- it mutates the shared PF project
(Network Data is project-level) and overwrites the operating point the archived
W1 sweep is reproducible against.

## Correction: PowerFactory stores the DOUBLED model, so Experiment A is a hybrid

The `pf_sync --dry-run` reports at all six snapshots (`--phase full
--study-case 02_RMS_CoSim`) are parameter-only -- created 0, renamed 0,
deleted 0 throughout -- with 116 updates at W1 against 235-257 elsewhere.
Classifying the W1 residual by subnetwork settles what the project holds:

- **DSO_3: 46 of 52 changes are a clean factor of exactly 2.000000** -- `sgn`,
  `Srated`, load `plini`, DER `pgini` alike (e.g. `DER_DSO_3_s25_b88.sgn:
  200.0 -> 100.0`). DSO_1, DSO_2 and DSO_4 show none; their only changes are
  reactive power to zero. **The PF project holds the doubled-DSO_3 network.**
- **Zero transmission-load changes at W1**, against 116 at every other
  condition, so the stored profile instant is W1.
- The 64 `qgini`/`qsetp` -> 0 changes present at every condition are the
  controller-free snapshot convention, not an operating-point difference.

So the archived Experiment A is a hybrid: commands, magnitudes and QSS ranking
from the **un-doubled** snapshot, RMS plant **doubled**. This corrects the
Phase-0 statement that the A settling table belongs to the un-doubled network --
the snapshot does, the measurements do not. It is also a defect in its own right:
DER-Q command magnitudes were sized from un-doubled capability, so for DSO_3 they
are half what the doubled plant can deliver.

It further undermines "keep the existing 70-command W1 sweep": under the
un-doubled freeze, W2-W6 would run on a synced un-doubled plant while the
retained W1 sweep sits on the doubled one.

Second discrepancy, visible even at W1: the stored zonal generator dispatch does
not match the snapshot's. Totals differ by only 98.2 MW (2635.9 vs 2734.0) but
the distribution is entirely different -- G 01 100.6 -> 1248.7 MW, G 09
740.1 -> 124.9 MW, G 03 578.0 -> 223.9 MW. The snapshot's G 01 at ~70 % of
capacity is the dispatch already known to ramp 1.3 Hz in outage studies for want
of a governor; Phase 1 injects no contingencies, but later phases would.

## Phase 1 executed (2026-09-08 14:04 -- 2026-09-09 13:43)

Network frozen **doubled** (second decision, after the dry-runs showed
PowerFactory stores the doubled model and Experiment A's settling times were
measured on it). `export/make_snapshots.py` gained an opt-in `--dso-der-scale` /
`--dso-load-p-scale` path, verified a no-op by default (the un-doubled W1
snapshot rebuilds byte-identical apart from `created_utc`). The project was
exported to `phase1/IEEE39_qOFO_pre-sync_2026-09-08.pfd` before the first write.

Two campaigns, each syncing its own snapshot into PowerFactory before sweeping:

- **six conditions x 8-10 commands**, 655 min, 56 cases, 0 censored;
- **W1 x 70 commands** on the synced plant, 720 min, 70/70, 0 failures.

Results: worst voltage settling **12.962 s** (W3, `tap_p1_MT_g0_t0`,
`V_TN_bus1`), leaving 7.038 s to `T_STS = 20 s`; worst TS-STS interface Q
**8.612 s** (W2). At the reference point the 70-command sweep gives 12.672 s and
8.512 s against the archived 12.472 s and 8.412 s -- same binding cases, same
limiting signal, median shift -0.075 s over the 66 common cases.

**The limiting actuator is not the same at every operating point.**
`tap_p1_MT_g0_t0` sits at 10.1-10.3 s at five conditions and jumps to 12.962 s at
W3, where it binds. The single-point study could not observe this.

**The excluded inter-area tie-line Q now exceeds the dispatch period**: 23.872 s
on `Q_TSO_Z1_Z2_TN_line2`, at five of six conditions (archived 17.172 s). Not a
controlled output, but worth stating rather than leaving implied.

Four cases differ from the archived sweep, all DSO_3 DER-Q steps: the archive
sized them from the un-doubled snapshot while acting on the doubled plant, so
they were half the plant's capability. Corrected here.

Export: `docs/data/2026-09-09_ch9_settling_six_conditions/`. Proposed section 9.1
text changes drafted, not applied:
`results/THESIS_ch9_PF_batch_2026-09-08/phase1/ch9_1_text_changes.md`.

## Incident: a stale completion marker cost W2 three cases

The follow-on job was queued with the check `"finished" in manifest.json`. That
key was left behind by an earlier `--only W1` invocation, so the waiter read the
campaign as complete, opened a second PowerFactory session at 16:53 and synced
the plant back to W1 while W2 was on case 6 of 8. PowerFactory permits one
session; W2's was terminated ("User session has been terminated", rc=6002).

Cost: three cases, ~45 min of rework. Cases 1-5 had completed on the correctly
synced W2 plant and stayed valid; the sweep resumes from its own `results.csv`,
so only the lost cases re-ran. No data was corrupted.

Fixed in two places, both commented with the reason: the driver now clears its
own completion marker at start, and the chain waits on the driver's **process
id** rather than any file-based marker.

## Verification that each condition ran on its own plant

Checked rather than assumed, since identical numbers are what a silent sync
failure looks like:

- all six preflight drift ratios differ;
- W4 and W5 share 7 of 10 settling times, but their sync logs wrote **all 164
  common values differently** and the same command reports a different limiting
  bus (`V_TN_bus9` vs `V_DSO_3_bus87`). The coincidence is the 50 ms sampling
  grid on commands whose dominant dynamics are machine-side;
- the 70-command W1 sweep reproduces the 8-command W1 preflight ratio exactly
  (0.396178998668395), so the plant re-synced to a repeatable state;
- the four fastest cases (shunt 0.022 s, TS-DER 0.172/0.872 s) all report
  `event_live=True`, `actual_delta == requested_delta`, diagnostics within
  tolerance and `subband=False` -- genuinely fast, not unapplied.

## Open questions for the next session

1. Whether the chapter should report the excluded inter-area tie-line Q settling
   alongside the controlled 12.472 s. The margin to `T_STS` is 7.5 s on the
   controlled outputs and 2.83 s on the tie line.
2. `tests/tuning/` had not finished when this was written; the two directly
   relevant unit-test files pass.

## Loose ends

`results/rms_phase6_replay/0538` (cited as the W1 C1 export run) does not exist,
and `0537_2026-08-07_131025` (the W2 import run) is an empty directory. The
baseline YAML carries `g_w_dso_oltc: 150`; the 200.0 quoted for the archived C1
runs is the runner's value, and both differ from the frozen 392.6444.
