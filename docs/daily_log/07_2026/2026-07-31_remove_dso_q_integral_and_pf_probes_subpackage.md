# 2026-07-31 — Remove DSO integral Q-tracking; group PF probes into `pf/probes/`

**Timestamp:** 2026-07-31
**Scope:** refactor only — no change to any controller behaviour that was
active in a thesis configuration.

---

## 1. Removed: DSO integral Q-tracking (leaky integrator)

### Reason

The integral term was pinned off (`g_qi = 0`) in every experiment, baseline
YAML and BO trial in the repository, and was explicitly excluded from the
BO search space. It nevertheless remained wired through five layers
(controller → config dataclasses → runners → tuning → YAML), where it had to
be validated, serialised, pinned in `FIXED_OVERRIDES`, and carried in the
search-space fingerprint. Dead configuration surface with a stability
footnote attached (`docs/tuning/stability_analysis.md` §"augmented analysis")
but no active code path.

### What the mechanism was

A leaky integrator on the interface-Q error, contributing an extra gradient
term to the DSO MIQP objective:

$$
s^{k+1} = \lambda\, s^{k} + (Q_\text{if}^{k} - Q_\text{set}),\qquad
s \leftarrow \operatorname{clip}(s, \pm s_\text{max})
$$

$$
\nabla f \mathrel{+}= 2\, g_{Q,I}\; s^\top \frac{\partial Q}{\partial u}
$$

Intended to build pressure for discrete switching (OLTC, MSC/MSR) when the
continuous DERs alone could not close the Q-interface error. Documented in
`docs/architecture/PI_CHARACTERISTIC_OFO.md`.

### Method of change

Deleted the state, the three config fields, their validation, and the
gradient block; then swept every call site so no `TypeError` is left behind.

| File | Change |
|---|---|
| `controller/dso_controller.py` | dropped `g_qi`, `lambda_qi`, `q_integral_max_mvar` fields + docstrings + `__post_init__` validation; dropped `self._q_error_integral`; dropped `reset_integral()`; dropped the integral gradient block in the objective-gradient assembly |
| `configs/config.py` | dropped `dso_g_qi`, `dso_lambda_qi`, `dso_q_integral_max_mvar` from `MultiTSOConfig` |
| `configs/cascade_config.py` | dropped `g_qi`, `lambda_qi`, `q_integral_max_mvar` from `CascadeConfig` and from `to_dict()` |
| `experiments/runners/multi_tso_dso.py` | dropped the three kwargs at the `DSOControllerConfig` construction site |
| `experiments/runners/cascade.py` | same |
| `experiments/run_multi_system_ofo.py` (2 configs), `experiments/CIGRE_2026/005`, `006`, `experiments/archived/001`, `002`, `analysis/observer/analyse_observer_run.py` | dropped the kwargs from the experiment configs |
| `tuning/parameters.py` | dropped the three `FIXED_OVERRIDES` entries and the module-docstring paragraph |
| `tests/tuning/test_parameters.py` | the diverging-baseline fixture used `dso_g_qi=99.0` to prove `FIXED_OVERRIDES` wins; now uses `dso_gamma_oltc_q=0.9` |
| `configs/baseline_002.yaml`, `tuning/scripts/configs/baseline_002.yaml`, `tuning/scripts/configs/baseline_002_ieee39.yaml` | dropped the three keys |
| `tuning/_io.py` | **new** `_RETIRED_FIELDS` frozenset, filtered out in `load_config_yaml` |
| `docs/tuning/tuning_strategy.md` | §2 no longer describes the parameters as "excluded"; states they were removed |
| `docs/architecture/PI_CHARACTERISTIC_OFO.md` | status banner: document is now historical w.r.t. the I-part |

### Backward compatibility

`load_config_yaml` does `MultiTSOConfig(**d)`, so any `config.yaml` written
by an earlier run (every study directory under `results/`, every archived
baseline) would have raised `TypeError: unexpected keyword argument
'dso_g_qi'`. `_RETIRED_FIELDS` drops such keys on load. Verified by
injecting the three keys into a copy of `configs/baseline_002.yaml` and
loading it.

`CascadeConfig.from_dict` already filtered unknown keys against
`__dataclass_fields__`, so stored `config.json` files need no change.

### Verification

- `pytest tests/tuning/test_parameters.py` → 14 passed
- `pytest tests/test_controller.py tests/test_dso_qv_sensitivity.py tests/test_tag_der_q_modes.py` → 70 passed, 1 skipped
- YAML round-trip of all three baselines, plus a legacy-key YAML
- repo-wide grep for `g_qi|lambda_qi|q_integral_max|_q_error_integral|reset_integral`: only the `_RETIRED_FIELDS` entries remain in `.py`; zero in `.yaml`

---

## 2. New subpackage: `pf/probes/`

### Reason

`pf/` had accumulated 33 modules in one flat namespace, 12 of which were
one-off read-only probes — scripts that answer a single question about the
PowerFactory API or RMS solver semantics, print findings, and restore state.
They are not part of the co-simulation import graph, so they were pure noise
next to the load-bearing modules (`plant.py`, `screening.py`, `pf_sync.py`,
`session.py`, `replay.py`).

### Method of change

Moved, then rewrote the path bootstrap one level deeper and repointed the
intra-probe imports:

- Moved 11 `pf/probe_*.py` → `pf/probes/`.
- Moved the stray root-level `probe_wecc_frame_connections.py` → `pf/probes/`.
- Moved `pf/probe_data/` → `pf/probes/probe_data/`, which keeps
  `probe_rms_elmfile_profile.py`'s `Path(__file__).parent / "probe_data"`
  correct without editing the expression.
- `sys.path.insert(0, ... .parents[1])` → `parents[2]` in all moved modules.
- `from pf.probe_event_* import ...` → `from pf.probes.probe_event_* import ...`
  (the `probe_event_*` family reuses helpers from `probe_event_rearm`).
- Usage lines in the module docstrings updated to `python pf\probes\...`.
- New `pf/probes/__init__.py` documenting the probe families and the rule
  that probe bodies stay behind a `__main__` guard.
- `probe_wecc_frame_connections.py` additionally: replaced the hardcoded
  `sys.path.insert(0, r"Z:\Python_Projekte\qOFO_GH")` with the repo's
  `parents[2]` idiom, added a module docstring, and wrapped the
  top-level script body in `main()` behind a `__main__` guard (it was the
  only probe that executed on import).
- Path references updated in `pf/__init__.py`, `pf/screening.py` (docstring
  provenance note) and `docs/RMS_governor_droop_parity_prompt.md`.
  Daily-log references to the old paths are historical and left alone.

### Verification

Probes cannot run off the PF machine, so verification is static:
`py_compile` on all 12 modules, and `importlib.util.find_spec` on
`pf.probes` and the cross-importing modules. Both clean.

`pf/` is untracked in git, so this was a filesystem move, not `git mv` — no
history to preserve yet (see risks).

---

## 3. Housekeeping (approved from the proposal list the same day)

| Item | Change |
|---|---|
| **B** | Deleted 6 Codex scratch artifacts at the repo root (`.codex_tmp_postinit_probe.py`, `.codex_tmp_prototype_v8.py{,.orig,.rej}`, `.codex_tmp_v7.patch`, `.codex_tmp_v7_remove.patch`) and `network/ieee39/meta.py.orig`. All were dated 2026-07-20/21 and referenced `prototype_qv_controller_v6/v7` plus a per-session `~/.codex/visualizations/...` path, neither of which is in this repo. Added `*.orig`, `*.rej`, `*.patch`, `.codex_tmp_*` to `.gitignore`. |
| **C** | `.gitignore` excluded `.claude/`, so the project instructions were unversioned while root `AGENTS.md` was tracked. Changed to `.claude/*` + `!.claude/CLAUDE.md` + `!.claude/claude.md`. **`.claude/` (the directory) had to become `.claude/*` (its contents)** — git cannot re-include a path inside an excluded directory, so the negation was dead until that changed. Both spellings are listed because the file is `claude.md` on disk and only matches `CLAUDE.md` while `core.ignorecase=true` (true here, not on a Linux clone). `settings*.json` and `scheduled_tasks.lock` stay ignored. |
| **D** | `analysis/observer/analyse_observer_run.py` → `_archive/observer_phase3/`, with a README stating why it was archived rather than repointed: it monkey-patches `mod.attach_observer` and calls `mod.run_multi_tso_dso(CFG)`, but the current runner never imports `attach_observer` (that factory lives in `analysis/observer/stability_integration_ieee39.py`), so patching it is a no-op and the script would hit its own `RuntimeError("observer was never instantiated")` guard. Reviving it needs the observer hook re-integrated into the runner — a design decision, not a path fix. |
| **E** | `analysis/observer/test_stability_observer.py` and `test_stability_tuning.py` → `tests/observer/` (+ `__init__.py`, matching the existing `tests/pf`, `tests/tuning` layout). They were never collected by `pytest tests/`, so **13 tests were silently outside the suite**; all 13 pass now. Dropped their manual `_REPO_ROOT` `sys.path` bootstrap, which `tests/__init__.py` makes redundant. |
| **F** | Deliberately skipped (Manuel: "not now"). |
| **G** | Deferred — a PF run is in flight and `pf/` is read from this share by the PF machine. |
| **I** | 145 daily logs filed into `MM_YYYY` folders (`06_2026` 28, `07_2026` 117). New `docs/daily_log/INDEX.md`, generated by new `docs/daily_log/_build_index.py` (idempotent; reads each log's first `#` heading; regenerate after adding logs). 26 cross-references to the old flat paths were rewritten across `.md`/`.py` (`daily_log/2026-07-19_x.md` → `daily_log/07_2026/2026-07-19_x.md`); repo-wide grep for `daily_log[/\\]2026-` is now empty. |

**New convention:** daily logs go in `docs/daily_log/MM_YYYY/YYYY-MM-DD_slug.md`, and
`python docs\daily_log\_build_index.py` refreshes the index.

## 4. Risks / unresolved points

1. **BO study resume will refuse.** `tuning.parameters.search_space_fingerprint()`
   hashes `FIXED_OVERRIDES`, so removing the three pinned entries changes the
   digest and `tuning.tune.main` will refuse to resume any existing study.
   The refusal is a false positive here — the removed entries pinned the
   integrator *off*, which is what every trial already ran — but resuming a
   persisted study now needs a deliberate decision (bump
   `SEARCH_SPACE_VERSION` and re-stamp, or accept a fresh study).
2. `docs/tuning/stability_analysis.md` and
   `stability_report_for_discussion.md` still derive the augmented iteration
   matrix with the $q_\text{int}$ state. Left as written: they are dated
   analysis reports, not live specification. The conclusion they reach
   ($g_{Q,I} = 0$ reduces to the memoryless case) is exactly the
   configuration that now holds by construction.
3. The whole `pf/` package (33 modules, ~450 kB of Phase 0–6 RMS work) is
   **untracked**. Nothing in this refactor is recoverable from git.
4. **Another process was editing this working tree during the session.** Three
   `.orig` files present at the start vanished unprompted; `tuning/_io.py`
   gained a `_NESTED_DATACLASS_FIELDS` block between two reads;
   `experiments/run_rms_phase6_replay.py` → `run_rms_cosim.py`,
   `postprocess_rms_phase6_replay.py` → `postprocess_rms_cosim.py`,
   `run_rms_openloop_uy.py` → `run_openloop_qss_to_rms.py`, plus new
   `run_comparison_rms_cosim_qss.py`; and a second 2026-07-31 daily log
   (`2026-07-31_bo_tuning_audit.md`, `2026-07-31_rms_tap_control_gate_e_result.md`)
   appeared. Everything above was re-verified against the final on-disk state,
   but concurrent sessions in one tree can silently drop each other's edits.
5. `analysis/observer/` now holds only library code plus
   `demo_wind_replace_observer.py`, two `.md` notes and two `.png` figures.
   Figures checked in beside source are a separate question, not touched here.
