# Handover: re-running the dead-band selection experiment

**To:** a fresh Claude Code session on `Z:\Python_Projekte\qOFO_GH`
**From:** qOFO_GH session, 2026-07-31
**Purpose:** produce `deadband_selection.png`, `deadband_band.png` and the backing
CSVs for thesis Ch. 8 §2 (selection of the DER Q(V) dead-zone half-width δ).

Everything needed is now **in the repository**. Earlier versions of these scripts
lived in a session scratchpad and carried hard-coded numbers; both problems are
fixed (see §7).

---

## 1. What the experiment shows

δ is a **two-sided design choice**, and the figure is the argument for the value:

- **δ too narrow** → the local droop answers every small profile variation. The DER
  chatter and the OFO keeps re-anchoring against them.
- **δ too wide** → the droop is inactive between dispatches, so there is no local
  support and the interface Q drifts until the next OFO step.

Both extremes degrade the controlled quantities, so the useful δ is the **interior
minimum**. Three quantities are measured on the *same* runs:

| metric | meaning | unit |
|---|---|---|
| interface Q | mean \|Q_act − Q_set\| over the TS–DSO interfaces | Mvar |
| TS voltage | per-zone RMS voltage error | pu |
| DS voltage | RMS deviation of each DSO group's mean V from 1.03 pu | pu |

The study is run at several **profile windows** so it can be shown whether δ\* is a
system property or an artefact of one operating point.

**Actuators:** DER reactive power (continuous, Q(V) droop with dead zone),
coupler 3W OLTCs, machine 2W OLTCs, MSC/MSR shunts.
**Controlled outputs:** interface Q at the EHV–HV boundary, nodal voltages.
**Disturbance class:** *profiled operation* — the annual load/DER profiles, not an
injected step. This matters: it is the realistic small-signal regime the dead band
is meant to filter.

---

## 2. Prerequisites

- **Python:** `F:\python_environments\qOFO_clean\python.exe`
  (the workstation path in `.claude/CLAUDE.md` is **not** valid on this server).
- **PowerFactory must be free.** The runs drive PF, and a second `connect()` kills
  the first session ("User session has been terminated", exit 114). Do not start
  the sweep while anyone has the PF GUI open, and do not connect to PF while it
  runs.
- Working directory: `Z:\Python_Projekte\qOFO_GH`.

---

## 3. Step 1 — run the sweep

```bash
powershell -File experiments\run_deadband_sweep.ps1
```

Defaults: scenario `rural_700`, 3 windows × 5 dead bands = **15 runs, ~28 min each
≈ 7 h**. Fail-fast: it aborts on the first non-zero exit rather than burning the
night on a broken configuration. Logs go to
`results\deadband_selection\logs\<scenario>_<window>_db<delta>.log`.

Useful overrides:

```bash
powershell -File experiments\run_deadband_sweep.ps1 -Scenario base_410
powershell -File experiments\run_deadband_sweep.ps1 -Deltas 0.005,0.01
```

**Check run 1 before walking away.** A configuration error shows up in the first
run, and the whole matrix is worthless if it is wrong. Look for `[Gate E] PASS` at
the end of the first log.

### δ = 0 is excluded by default — and that default is wrong

The exclusion was justified by a recorded cost of roughly **4.9 h** per δ = 0
run (static-leg initialisation), against ~28 min for every other δ.

**That figure is void.** Measured 2026-07-31, the three δ = 0 runs took
**12.3, 12.8 and 13.2 min** — indistinguishable from every other cell, and ~23×
below the recorded estimate. All three phases of that evening averaged ~13 min
per run regardless of δ.

This mattered: δ = 0 turned out to be the single most informative point in the
study. It is where the interface-Q error is **3× its optimum** in every live
window, and it is what makes the two-sided argument evidential rather than
suggestive — without it the narrow-side branch rested on a ~5 % difference that
is not separable from run-to-run variation.

**Include `'0'` in `$Deltas`.** It costs ~13 min like anything else.

### Scenario must always be explicit

The config default is `base_410`. Relying on it silently produced one `base_410`
run in the middle of a `rural_700` series on 2026-07-29. `base_410` and
`rural_700` differ in installed DSO DER (410 vs 700 MW per DSO) and **their results
are not comparable**. The sweep script always passes `--scenario`, and the analysis
additionally filters on the scenario each run recorded in its own `config.json` — a
run-number threshold would not have caught that mistake.

---

## 4. Step 2 — CSVs and figures

```bash
F:\python_environments\qOFO_clean\python.exe -X utf8 -m analysis.deadband_selection
```

Reads every run under `results/rms_phase6_replay/`, admits only those matching the
study configuration, and writes into `results/deadband_selection/`:

| file | content |
|---|---|
| `deadband_metrics.csv` | one row per run: window, excursion, δ, run id, the 3 metrics |
| `deadband_optima.csv` | argmin δ per window per metric |
| `figures/deadband_selection.png` / `.pdf` | per-window U-curves, one panel, twin axes; each minimum ringed |
| `figures/deadband_band.png` / `.pdf` | min/max band across windows, each metric normalised to its own best |

It also prints the per-window tables and a verdict on whether δ\* moves with the
operating point. Options: `--scenario`, `--results-root`, `--out`, `--no-figures`.

Both figures degrade gracefully: `deadband_selection` needs ≥2 dead bands in a
window, `deadband_band` needs ≥2 windows sharing ≥2 dead bands. With less, it says
so and skips rather than drawing something misleading — so it can be run mid-sweep
to watch the curve fill in.

**Nothing is hard-coded.** Every number comes from `rms_records.pkl`.

---

## 5. Admission filter

A run enters the study only if its own `config.json` → `runner_static` matches
(`ADMIT` in `analysis/deadband_selection.py`):

| key | required |
|---|---|
| `scenario` | the `--scenario` argument |
| `der_q_capability_override_pu` | `None` (physical VDE capability, not a ±1.0 pu stub) |
| `use_profiles` | `True` |
| `dso_qv_slope_pu` | `0.06` |
| `seed_der_anchor_to_local_v` | `False` |
| `disable_qv_seed` | `False` |

Runs without `rms_records.pkl` (aborted) are skipped silently. The admitted/skipped
counts are printed — check them.

---

## 6. Current state (2026-07-31, 13:00)

- **The 15-run sweep is running.** Started 12:59:42, first run dir `0098`;
  projected finish ≈ 20:00. Watch
  `results\deadband_selection\logs\sweep_master.log`.
- **`rural_700` now has _no_ valid run predating this sweep.** `0080` was believed
  to be one valid cell, but it records **no** `dso_der_scale` / `dso_load_p_scale`
  — it predates the DSO_3 ×2 default and is unscaled, so it is not comparable
  with the sweep (the runner prints this warning itself). It is excluded; the
  study now stands entirely on the 15 sweep runs.
- Runs `0081`–`0086` are incomplete and skipped automatically. Runs `0087`–`0097`
  are `rural_700` but carry `der_q_capability_override_pu = 1.0` (run without
  `--physical-capability`) — a different experiment, correctly excluded by the
  admission filter. `0097` was cancelled manually at 10:52.

### The entry point moved on 2026-07-31

`experiments/run_rms_phase6_replay.py` was split the same morning by a concurrent
session. The sweep script now calls **`experiments.run_comparison_rms_cosim_qss`**;
the old `-m` target aborts immediately with `No module named ...`.

Do **not** repoint it at `run_rms_cosim` — the RMS-only entry point. It takes the
same flags and is about twice as fast, but it writes to `results/rms_cosim/` and
stores `runner_static = None`, and the admission filter (§5) reads the
`runner_static` block. The sweep would run 7 h, exit 0, and admit **zero** runs.
See `docs/daily_log/07_2026/2026-07-31_deadband_sweep_runner_repoint.md`.

### The DSO_3 ×2 multipliers are part of the study configuration

All 15 runs carry `dso_der_scale = dso_load_p_scale = {"DSO_3": 2.0}` (2× installed
DER and 2× active-load base on DSO_3), from the defaults in
`experiments/helpers/rms_cosim_config.py`. The sweep passes neither
`--symmetric-dso` nor explicit scales. Both keys are now in the admission filter
(§5), because the runner states a scaled run is *not comparable* with an unscaled
one and nothing else caught it.

`analysis/deadband_selection.py` also now prints a line when two admitted runs land
on the same (window, δ) cell. It previously overwrote silently — which is the only
reason `0080` did not corrupt the curve: the sweep re-runs its exact cell and would
have replaced it without a word.

**No reproduction check against `0080` is possible.** It differs from the sweep in
both scaling *and* code version, so the 2026-07-31 runner split cannot be validated
empirically against it. The RMS runners are untracked in git and the pre-split file
is gone, so that boundary rests on the daily log and on reading the source.

### The old numbers are void

The previous U-curve (runs `0067`–`0075`, δ\* = 0.005) is superseded twice over:

1. It is on scenario `wind_replace` (410 MW/DSO) and on the **topology before** the
   2026-07-29 change.
2. The 2026-07-30 fixes changed the **Jacobian for every scenario** — see
   `docs/daily_log/2026-07-29_rural700_infeasible_tn_statcom_q.md`. Most
   significantly, `JacobianSensitivities` was linearising about a point up to
   **0.058 pu away** from the plant solution; it is now exact.

Do **not** mix pre- and post-fix runs in one curve. The admission filter does not
catch this — it is a code-version boundary, not a config difference. Everything
from run `0080` onward is post-fix.

`docs/thesis_ch8_deadband_selection_handover.md` describes the *argument* to make
in the thesis and is still useful for that, but **its numbers are void**.

---

## 7. What changed to make this reproducible

- `analysis/deadband_selection.py` — **new**. One command: collect → CSV → both
  figures. Replaces four scratchpad scripts (`deadband_fig.py`, `deadband_fig2.py`,
  `deadband_vq_multi.py`, `deadband_band_fig.py`).
  - The old `deadband_fig2.py` had the metrics as **hard-coded literals** from runs
    0067–0073 and never read the results, so "re-running" it reproduced the old
    figure regardless of new data.
  - Validated by re-deriving those literals from the pickles: 2.846 / 1.242 /
    1.104 / 1.496 / 1.652 / 3.013 / 2.912 Mvar reproduced **exactly**.
- `experiments/run_deadband_sweep.ps1` — **new**, parameterised
  (`-Scenario`, `-Deltas`, `-Windows`, `-Duration`, `-Python`, `-LogDir`).
- Both previously lived in a session-specific scratchpad whose path embeds a
  session id, so they were unreachable from any other session.

---

## 8. Gotchas

- **Stopping a run:** `TaskStop` kills the PowerShell wrapper but *has* left the
  python child alive in the past. Check for an orphan holding the PF session before
  starting anything new.
- **Orphaned PF event slots:** killing a run mid-flight leaves event-pool slots with
  no `p_target`. These used to abort every *subsequent* run; since 2026-07-30 they
  are deleted automatically with an `[event_pool] removed N orphaned pool slot(s)`
  line. On 2026-07-30 there were **13,080** of them, which took ~10 min to clear —
  if a run seems to hang early with no output, that is probably what it is doing.
- **Run numbering** is global and increments even for aborted runs, so gaps are
  normal.
- **`--physical-capability`** is what selects the real VDE capability diagrams; the
  sweep script always passes it. Without it the DER get a ±1.0 pu stub override and
  the run is excluded by the admission filter.

---

## 9. Open decisions (not blocking)

- **δ = 0** — include it and spend ~15 h, or leave δ = 0.0025 as the narrow anchor?
- **TN STATCOM capability** — the four `WP_STATCOM` devices are built with
  `sn_mva == p_mw`, so `max_q_mvar = sn` ignores P and they operate at
  S/Sn ≈ 1.41. The build-time seed is now clamped to the *declared* ±sn, but their
  true capability (separate STATCOM rating / oversized converter /
  `q_max = √(sn²−p²)`) is undecided. Affects both scenarios; deliberately left to
  the author.
- Whether the thesis quotes `rural_700`, `base_410`, or both.
