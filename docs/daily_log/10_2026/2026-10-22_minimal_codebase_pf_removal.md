# Minimal codebase: the PowerFactory adapter, Nordic 32, tools and the campaign launchers removed

Timestamp: 2026-10-22, afternoon, Europe/Berlin.

Reason: the author asked for a minimal codebase carrying what the thesis
describes, naming `tools`, `pf` and the PowerFactory-adapter experiments, the
Nordic 32 network, `export`, many of the older experiments, and a per-file
audit of `core`, `controller`, `configs` and `analysis`. Done autonomously on
branch `minimal/thesis-core`, every step its own commit, so nothing is
unrecoverable.

Method note: the work is in a **separate worktree** at
`C:\Users\Manuel Schwenke\qOFO-minimal`, not in `Z:\Python_Projekte\qOFO`. The
shared checkout had another session's uncommitted work in it, and switching
branches there would have moved HEAD under that session. The branch lives in
the same repository, so `git log minimal/thesis-core` works from either place.

Changes: commits `94319df` (PowerFactory), `38f3557` (Nordic 32, tools,
launchers), `36235aa` (dead Q(V) channels, the audit, colloquium).

Baseline: the full suite passes at `100de96`, the last commit before this
branch diverges — **862 passed, 15 skipped, nothing failed**, run in an
isolated clone at that pinned SHA (measured by the parallel session). That
includes the experiment restore and the index regeneration from the morning.

## The headline consequence

**This removes the project's only non-quasi-static validation path.** After it,
every controller claim in the codebase — the switched-shunt law included — is
verifiable only against the static pandapower plant, and the RMS
co-simulation comparison cannot be re-run at all. The reasoning and the
measured findings survive in the daily logs; as *runnable evidence* they do
not.

That is a defensible choice for a thesis-only codebase, and it is the author's
decision. It should be a knowing one. Two things follow from it that are easy
to miss:

- the quasi-static plant is now both the thing under test and the only
  reference it can be tested against, so a defect shared by controller and
  plant model has nothing left to expose it; and
- it also discards the `base`-phase repair made this morning as a
  *PowerFactory* fix, although that repair survives in another role — see the
  state builder below.

## What was removed, and what it cost

**The PowerFactory adapter and the RMS co-simulation**, about 16.5k lines:
`pf/` (43 modules), `export/dynamic_snapshot.py` with its 12 generated
snapshots, the four RMS entry points, the two `rms_*` helpers, two
`pf`-importing modal analyses, `analysis/gate_e_diagnostics.py`, and their
tests (`tests/pf/`, `tests/export/`, four more).

Two consequences the author should weigh, neither of them avoidable while
removing `pf`:

- **Two chapter-9 experiments went with it.**
  `ch_9_1_timescale_seperation.py` produced Table 9.1, the settling battery
  that fixes `T_DS` and `T_TS`; `ch_9_1_actuator_location_sweep.py` was the
  actuator-location sweep. Both drove PowerFactory through `pf/screening.py`,
  so keeping them would have meant keeping the adapter. Their recorded results
  stand; the package docstring now says this instead of listing a module that
  is gone.
- **The chapter-9 dead-band study can no longer be re-run.** Its sweeps
  generated data through the RMS co-simulation: nine of the removed launchers
  drove `experiments.run_comparison_rms_cosim_qss`. The eleven
  `analysis/deadband_*` scripts are kept and still read and report the recorded
  runs under `results/`; their two "run the sweep first" hints now say the
  driver was removed rather than naming a file that no longer exists.

**Kept deliberately: the network-state builder.** `export/make_snapshots.py`
had two halves. The dumper was PowerFactory's; `build_snapshot_state` builds
the configured network at a timestamp and brings it to the initial operating
point, which is the only way to obtain "the state every quasi-static
experiment starts from" without running a controller — what three plant tests
measure against (`test_zip_load_model`, `test_plant_static`,
`test_full_dso_aux_load_buses`). It is now
`experiments/helpers/snapshot_state.py`, trimmed of the CLI that was the sole
user of `export.dynamic_snapshot`.

That also keeps this morning's `BARE_SCENARIO` repair load-bearing rather than
moot: those surviving tests build the `base` phase, the bare IEEE 39 case that
fix restored. Verified: 40 buses, empty `sgen` table.

**`network/nordic32/`** — the second test system. Genuinely isolated: nothing
imported it, and its own `convert_from_pf.py` converted the network out of a
PowerFactory project, so it had no builder left anyway. The TUDA network
(`build_tuda_net`) is untouched: it is still imported by
`optimisation/ac_opf_reference.py`, `experiments/runners/cascade.py` and two
tests.

**`tools/`** — four standalone scripts, none imported anywhere.
`check_reduction_fidelity.py` was already broken (it imported the deleted
`rms_cosim_config`). `doc_audit.py` and `gen_package_docs.py` generated the
architecture inventories, so `docs/architecture/packages/` (40 files) and
`doc_coverage.md` went with them — nothing can regenerate those now and they
describe deleted packages. The hand-written architecture docs stay.

**19 campaign launchers** — machine- and night-specific drivers, each tested
against what it actually invokes rather than removed on sight: nine drove the
deleted RMS comparison or the deleted `tools.missing_cells`,
`RUN_ON_SECOND_ACCOUNT.cmd` launched a `.ps1` that is not in the tree and was
already dead, and the rest are one-shot operational scaffolding (resuming a
sweep that died on a full disk, one night split across two accounts, a watcher
polling for logs of a batch that finished in August).

**The colloquium scripts** — `analysis/colloquium_run_report.py`,
`experiments/run_colloquium_gifs.py`, `visualisation/colloquium_gifs.py` and
the gif test. Presentation artefacts, not thesis ones.

## The single-file audit

An import-graph pass over `core/`, `controller/`, `configs/` and `analysis/`
answered the author's question directly: **`core`, `controller` and `configs`
contain no unimported modules** — every file in them is reachable. The
suspicion of dead single files there is not borne out.

`analysis/` has nineteen modules that nothing imports, but that is the wrong
test for it: these are standalone `python -m analysis.X` result producers, not
libraries, and "unimported" does not mean unused. Eleven are the chapter-9
dead-band family. Only the two tied to removed subsystems were deleted
(`gate_e_diagnostics`, which compared a static run against an RMS run, and the
colloquium report). **The remaining sixteen are left for the author**: deciding
which analyses a thesis chapter still needs is not a judgement to make from the
import graph, which is the mistake recorded below.

## Three write-only globals, and a config knob that had gone inert

The audit turned up real dead state in `core/actuator_bounds.py`:
`DER_QV_DEADBAND_BY_SGEN_PU`, `DER_QV_SLOPE_BY_SGEN_PU` and
`DER_QV_DEADBAND_OVERRIDE_PU` were all **written and never read**. Their only
reader had been `pf.plant` — their own docstrings said so ("Precedence in
``pf.plant``"). The runner still published into the two maps at step [4a] and
printed that it had.

The third one matters beyond tidiness. `config.der_qv_deadband_override_pu`
reached only the RMS plant: the static side takes its deadband from
`net.sgen.qv_deadband_pu`, which `tag_der_q_modes` writes from
`tso_`/`dso_qv_deadband_pu`. So the knob had become **silently inert**, while
the runner went on printing "DER Q(V) deadband OVERRIDDEN to X pu on both
plants". That is the failure mode this project has paid for before, so the
runner now **raises** if the field is set, and the field is kept (not deleted)
so a stored config carrying it fails loudly rather than being silently
reinterpreted.

### Which recorded runs believed it

"The runner printed that it had overridden the deadband on both plants" is the
kind of statement that gets quoted, so the scope needs stating rather than
worrying about. It is a **bounded, checkable question**: every archived run
carries its resolved `config.json`, so the runs that believed an override that
was never applied are exactly those whose config has a non-`None`
`der_qv_deadband_override_pu`.

Two things are known so far, both measured by the parallel session rather than
here:

- the four switched-shunt acceptance configs (pre- and post-repair, both arms)
  all carry `der_qv_deadband_override_pu = None`, with
  `der_qv_deadband_pu_overrides`, `der_qv_slope_pu_overrides` and
  `der_qv_vref_pu_overrides` empty. **The accepted figures never depended on
  it**, and the false claim in the log output did not reach them.
- the earlier campaigns are the open part. Once the archive copy finishes, the
  whole set sits at
  `Z:\Python_Projekte\qOFO_results_archive\vault_results_2026-10-22\`, and
  `docs/data/results_manifest_2026-10-22.md` maps each campaign to the
  documents that cite it — so the grep answers not just which runs were
  affected but which written claims rest on them.

Worth noting what class of defect this was, because it is the same class the
section above warns about: a claim the system made about itself with nothing in
the codebase able to contradict it. The static plant could not disagree,
because the override never reached it; the RMS plant could have, and is now
gone.

## Risks and open points

- The non-quasi-static validation path is gone (see above). The EvtTap and
  event-admission measurements of 2026-07-30 survive as reasoning in the daily
  logs but not as runnable code.
- Table 9.1 and the actuator-location sweep are no longer reproducible, and
  neither is the dead-band sweep data.
- Sixteen `analysis/` scripts are kept pending the author's judgement.
- `sbx_h` and `sbx_v` are kept per the author's decision (chapter 6 uses
  them); simplifying them is a later task.
- `docs/` was left alone apart from the generated inventories. It holds the
  record of the removed work, including the PowerFactory notes, and deleting it
  would destroy the provenance of results the thesis still reports.

## `results/` was never in reach, and that is worth understanding

Author instruction during this work: leave `results/` **entirely** alone until
it has been reviewed. It was never at risk here, and for a reason worth
recording rather than relying on: `.gitignore` excludes `results/`, so a fresh
worktree never receives it at all. The directory does not exist in
`C:\Users\Manuel Schwenke\qOFO-minimal`, and no commit on this branch touches
that path.

The same property is the hazard. **"Nothing was deleted, it is all in git" does
not hold for `results/`** — nothing in it is tracked, so a deletion there
leaves no git trace and cannot be reverted. The reassurance that covers every
other removal on this branch covers none of it.

Two inventories, taken by the parallel session:

- the code clone's `results/` holds only the two 6-hour shunt acceptance runs
  (26.6 MB, 20 files). Both are already preserved — provenance tracked in the
  vault under `docs/data/2026-10-22_shunt_fold_acceptance/`, records archived
  outside both repositories — so there is nothing left to rescue there, and
  equally nothing to gain by deleting it: being untracked, it is not part of
  the thesis codebase in git terms at all.
- the **vault** at `Z:\Python_Projekte\qOFO_2027` has its own ignored
  `results/` holding **3.06 GB in 7,488 files across 39 campaign folders**,
  including the runs behind the 21 October decisions (`three_way_20261021`,
  `confirm6h_20261022`, `ts_fleet_margins_20261021`, `stripped6h_20261022`).
  Vault documents cite those runs as their evidence and not one of those files
  is tracked anywhere. That directory is outside this sweep's path; it is
  recorded here so that nobody later reasons about it as if git protected it.

## The lesson, third time today

Earlier today an `experiments/archived/` deletion broke a test because the
check for importers was a static `from`/`import` grep, which cannot see
`importlib.import_module` with a string literal — and four of those files
turned out to be current experiments misfiled under a folder named `archived`.
The same pattern repeated twice more here: a test that imported a deleted
*experiment module* rather than `pf` (caught by collection, not by grep), and
`analysis/`'s nineteen unimported-but-live scripts.

In this repository a file's location and name are weak evidence of its status.
The reliable references are the live documentation, the printed runbook
strings, the dynamic imports and the test collector. Each removal above was
checked against those rather than against the directory it sat in.
