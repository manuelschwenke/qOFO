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
launchers), and the dead-channel and colloquium removals that follow.

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

## Risks and open points

- Removing `pf/` removes the ability to reproduce the RMS-versus-QSS
  validation at all. The reasoning and the measured findings survive in the
  daily logs — the EvtTap and event-admission measurements of 2026-07-30 among
  them — but not as runnable code.
- Table 9.1 and the actuator-location sweep are no longer reproducible, and
  neither is the dead-band sweep data.
- Sixteen `analysis/` scripts are kept pending the author's judgement.
- `sbx_h` and `sbx_v` are kept per the author's decision (chapter 6 uses
  them); simplifying them is a later task.
- `docs/` was left alone apart from the generated inventories. It holds the
  record of the removed work, including the PowerFactory notes, and deleting it
  would destroy the provenance of results the thesis still reports.

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
