# Nine pre-existing test failures resolved, archives removed, daily log consolidated

Timestamp: 2026-10-22, Europe/Berlin.

Reason: second part of the codebase cleanup, covering everything outside the
switched-shunt controller (that part ran in a parallel session and is recorded
separately). The author's brief was: triage the nine pre-existing test
failures, settle the two competing daily-log conventions, and decide what to do
with the frozen archive trees. The failures predate the cleanup — they were
reproduced identically from a clone of `main` at `c23e67b`.

Changes: commits `ed2d010`, `37be8e3`, `694c153`, `fae83da`, `01f3a5c`, plus
the archive deletion that landed in `202a9cc` (see "Index collision" below).

## Findings: the nine failures were two defects, not nine

**Eight of nine were one production-code regression, not stale tests.** All six
collection errors in `tests/export/test_dynamic_snapshot_roundtrip.py` and both
`tests/test_zip_load_model.py` failures raised
`ValueError: Unknown scenario: 'base'` from `network/ieee39/build.py`, because
all eight route through `build_snapshot_state("base", ...)`.

- `"base"` was **dropped, not renamed.** Commit `5266850` replaced the no-op
  registry entry `"base": lambda net, meta, **kw: (net, meta)` with
  `"base_410": apply_wind_replace`. `export/make_snapshots.py:359` still asked
  for `"base"`, so the phase its own CLI help describes as "always builds the
  bare IEEE 39 case" could not be built at all.
- It is the **same defect `tuning/scenarios.py` repaired on 2026-07-31**, in a
  second call site missed then. There, `"base"` was drawn for ~20 % of
  validation specs, `build_ieee39_net` raised, `run_one` swallowed each into a
  sentinel cost, and a fifth of every campaign was silently scored as a
  power-flow failure.
- **`base_410` is not a substitute**, so the tempting one-line alias would have
  been wrong twice over: it applies the wind replacement, which removes
  G2/G5/G6/G8 (and `test_removed_generators_match_scenario` requires `removed
  == []` for this phase) and populates `sgen` (and `pf_sync.sync_base` rejects a
  non-empty `sgen` table outright).
- **The `base` phase is live, so the tests were correct and are kept.** It is
  the PowerFactory Gate-0 parity baseline: `pf_parity._PHASE_VARIATIONS["base"]
  = ()` is the empty variation stack, and `--phase base` is the `pf_sync` CLI
  default.

Fix: the bare case is now the explicit non-study build target
`network.ieee39.scenarios.BARE_SCENARIO` (`"bare"`), resolved in
`build_ieee39_net` before the registry check and deliberately **kept out of**
`SCENARIO_REGISTRY`. `tuning.scenarios.VALID_NETWORK_SCENARIOS` is
`frozenset(SCENARIO_REGISTRY)` and `probe_dso_boundary_conditions` offers
`sorted(SCENARIO_REGISTRY)` as CLI choices, so a no-op entry there would have
broken the 2026-07-31 regression guard and re-admitted the silent-failure path
into the tuning draws. Writing the name into `net["ieee39_scenario"]` keeps
`add_hv_networks` fail-fast on the bare case rather than defaulting to a DER
capacity it was never given.

**Reconciliation with the 2026-10-14 record.** That entry records the author's
decision to migrate five test files off the removed `"base"` scenario onto
`rural_700`. The two resolutions are not competing, and the distinction is the
part worth keeping: `"base"` as *a network to build* is incidental to what
those five assert, so a study scenario substitutes cleanly; `"base"` as a
*build phase* is load-bearing through the parity contract above, where no study
scenario can stand in.

**The ninth was a genuinely stale test.** In
`tests/pf/test_screening_event_pool.py`,
`test_persistent_pool_grows_admits_and_retires_events` still asserted the
pre-2026-07-30 semantics in which a tap event came from the persistent pool:
`created == 3`, every object named `qofo_pool_*`, `retired == 3`. Two measured
findings had since removed `EvtTap` from both paths, with the numbers recorded
in `add_tap_event`: a re-armed pooled slot does not move the tap mid-run
(0.033 mpu dead against +6.14 mpu for a fresh object), and retirement writes
`time` back onto an already-fired `EvtTap`, which kills every subsequent tap in
the calculation (of four mid-run taps only the first fired). The test also
predated the required `horizon_s` parameter of `admit_new_events`. Updated to
assert the split explicitly — a tap counts in `pending_admission` but not in
`created` or `retired`, and keeps its armed timestamp — so both findings now
have the regression guard they lacked.

Validation: the four files, 49 passed / 2 skipped (both reasoned in code —
`test_missing_required_field_raises` needs only one phase). Full suite collects
876 tests with zero collection errors, where six were failing to collect
before.

## Daily log: one location

The log was kept in two places, **both tracked in this same repository**:
`docs/daily_log/` (212 entries in `MM_YYYY/` folders with `INDEX.md` and
`_build_index.py`) and `00_daily_log/` at the root (10 entries, flat). The
global convention named the second and the project `.claude/CLAUDE.md` the
first, so entries landed in whichever the writing session had read.

Worth correcting for the record: `Z:\Python_Projekte\qOFO_2027` is **not a
separate obsidian vault** but a second clone of this repository on another
branch, with the same top-level tree. Both folders exist in both clones, so
this was an in-repo consolidation rather than a transfer between places.

Author decision: `00_daily_log/` is the single location. Moves only, nothing
deleted or rewritten — the month folders and index tooling moved across, and
the ten formerly flat entries were sorted into their month folders so the whole
log follows one `MM_YYYY/YYYY-MM-DD_slug.md` convention. A signpost remains at
`docs/daily_log/README.md`.

- One filename existed in both places with **different content**: a short
  summary in `00_daily_log/` whose body pointed at the detailed change log of
  the same name under `docs/daily_log/10_2026/`. A summary/detail pair, not a
  duplicate, so both are kept and the summary is now
  `10_2026/2026-10-20_hybrid_shunt_dispatch_summary.md`.
- **A pre-existing hole in `_build_index.py`**, found while verifying the move
  lost nothing: `LOG_FILE` required `_` or `.` directly after the date, so an
  entry whose date carries a letter suffix for a second entry on the same day
  was silently skipped. The committed index read "211 entries" for 212 files,
  and `2026-09-15b_ch9_network_decision_and_f1_instrumentation.md` had been
  absent from every generated index since 2026-09-15. The regenerated index
  lists all 222 entries.

Open: the project `.claude/CLAUDE.md` still instructs sessions to log to
`docs/daily_log` "in this obsidian vault", which would send the next session to
the retired location. Left to the author, since it is an instruction file.

## Archives removed

Author decision: delete both frozen archives wholesale; git history holds every
file at the commit it was frozen, which is what a reference copy is for.
Removed `_archive/sbx_h_v5/` (31 files, the pre-v6 SBX-H deal layer frozen
2026-07-12), `_archive/observer_phase3/` (broken at import since the
`experiments/000_M_TSO_M_DSO.py` rename), `experiments/archived/` (11 files),
and the six generated package docs that described only those trees. Checked
first: nothing imports `experiments.archived` or `_archive` anywhere in the
tree.

The negative results that justified the SBX-H freeze stay in `STATUS_SBX.md`
and the run directories under `results/`: the 015 campaign found the deal layer
unverifiable at quantum scale (G3), physically marginal (G4, a +0.166 pu·step
ceiling at delivery ratio 0.16) and never armed by an honest exhaustion test
(G7 — the contract tier alone recovers every constructed scenario).

**One deliberate loss of function:** `experiments/archived/run_rms_phase6_replay.py`
was not dead code but a live deprecation shim, kept since the 2026-07-31 rename
"so existing shell history and notes keep working". It goes with the rest per
the same decision, so an old invocation of that name now fails with a
missing-file error instead of delegating to
`run_comparison_rms_cosim_qss.py`.

## Doc tooling was broken, which had frozen the architecture docs

Regenerating the generated docs after the deletion showed the pipeline had not
been runnable for some time: `tools/doc_audit.py` died with
`KeyError: 'module_doc'`. Two pre-existing defects, independent of the cleanup:

- `experiments/case1_svr/report_case1.py` was saved with a **UTF-8 BOM**.
  `analyse` read it as `utf-8`, which keeps the BOM as a literal U+FEFF, and
  `ast.parse` rejects that. Python's import machinery strips a BOM, so the
  module always worked — only tooling that parses the text tripped. Now read
  with `utf-8-sig`, and the stray bytes removed from the file.
- `analyse`'s `SyntaxError` branch returned a record with only three keys,
  while `summarise` and the JSON payload index `module_doc` unconditionally.
  **One unparseable file anywhere killed the whole audit**, although
  `write_report` already had a branch for reporting syntax errors. The branch
  now returns the full key set.

Because the audit had been failing, `tools/doc_coverage.json` was stale and the
package docs were generated from it. Consequences now visible: **nine packages
had never been documented** (`configs/paramsets`, `experiments/case1_svr`,
`experiments/ch_9_parameter_selection`, `experiments/ch_10_case_study`,
`experiments/part3_mc`, `experiments/reference_period`, `tests/experiments`,
`tuning_mc`, `00_daily_log`), and **five generated docs described packages that
no longer exist** (`tuning/reports`, `tuning/scripts`,
`tuning/stability_certificate`, `tests/tuning/stability_certificate`, and the
orphaned `docs/daily_log`), kept alive only by the stale JSON. Those are
deleted with their dangling `PURPOSE` entries. The audit now reports 359
modules across 40 packages. Seven packages remain flagged uncurated, which is
the generator's intended way of making a missing purpose paragraph visible.

## Index collision between the two sessions

Both sessions worked in one checkout, which shares one git index. A `git add -A`
in the parallel session staged this session's in-progress
`network/ieee39/` work (unstaged again, nothing lost), and later swept this
session's 48 staged archive deletions into its own commit `202a9cc`, whose
message is about the ch9/ch10 dispatch axis and says nothing about them. The
reasoning for the deletion is therefore recorded in `694c153` and here rather
than with the deletion itself. A `pytest` run in that session also died with
`FileNotFoundError: _archive` when this session removed the directory while
pytest was walking it; that session moved to an isolated clone. Both sessions
switched to explicit pathspecs, which is what made the recovery clean.

## Risks and open points

- `base_410` is now reachable only by being named explicitly; the 2026-10-14
  decision was that only `rural_700` is needed for now. Nothing in this work
  changes which study scenario the runners use.
- The deleted shim means any note or shell history invoking
  `run_rms_phase6_replay.py` now errors rather than delegating.
- Seven packages have generated overviews without a curated purpose paragraph.
- The `00_daily_log` move is a rename-only commit, but the second clone at
  `qOFO_2027` is on another branch; daily-log work there should be committed
  before the branches merge.
- The `shunt_dispatch` / `shunt_int_*` configuration in the CIGRE scripts and
  the chapter-9/10 design tables was left entirely to the parallel session,
  which owns those files.
