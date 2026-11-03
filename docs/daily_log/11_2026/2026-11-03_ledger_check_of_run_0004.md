# 2026-11-03 — Shunt-feedforward ledger: check of the run-0004 selection (auto-launch armed)

Timestamp: 2026-11-03, ~09:30–10:15 Europe/Berlin.

## Reason

Run 0004 (`ch_9_8_procedure_v2.py`, thesis Ch 8 weight selection) simulates on
`res/commitment-base-2026-10-31` (`F:\qOFO_res`), which predates `ac69918`. A bank
release there is evaluated on the hosting STS's pre-switch model and is about 7 % too
large (`shunt_ff_ledger` did not exist yet, which is equivalent to `False`). Every
campaign after the selection (D1/D2, Case 1, Case-2 REF/MC) runs on
`minimal/thesis-core`, where `shunt_ff_ledger = True`.

**Author decision (2026-11-03): ledger on for all later campaigns, plus a check that
the selection holds.**

## Finding that shaped the check

- `_ledger_feedforward` changes nothing until a bank level is **released**: an engagement
  adds the same feedforward with or without the ledger. A window without a release runs
  identically under both settings.
- In run 0004, as of II-b round 3 (300 admitted candidates), **no feasible candidate
  releases a bank** in W1–W4 or W6. The top nine never switch a bank at all. A re-run of
  the top-ranked candidates, the first idea, would therefore show nothing by construction.
- The ledger can change the selection only by admitting a **rejected** candidate whose
  rejection came from a release. Inside the 2 % band there are 23 such candidates (62
  windows with a release). Several have a lower e_v_ts than the current leader, for
  example `ca6b2b5e2088` (0.00809 / 0.00813) and `60ef3f84da65` (0.00822 / 0.00801). All
  23 are rejected for tap or bank hunting, several of them heavily (up to 45 reversals).
- Simulation code differs between the two checkouts in more than the ledger: the
  live-plot rework `3f1577e` touches the controllers and the runner. The diff reads as
  recording-only (`objective_terms`), but that is not verified.

## Changes

- `experiments/ch_09_parameter_selection/ch_9_3_stage2_screen.py`:
  - new repeatable option `--config-set FIELD=VALUE`. It sets a `MultiTSOConfig` field
    in every simulated window, after the paramset, the weights and `--frozen-network`;
    VALUE is parsed as JSON, else taken as a string, and an unknown field is rejected;
  - recorded in `run_meta.json` and in each window JSON (`config_set`);
  - default behaviour is unchanged.
- New `experiments/ch_09_parameter_selection/ch_9_9_ledger_check.py` (thesis run via
  `start_thesis_run`, target `ch:timescales:weights`). It reads run 0004's final selection
  report and union, then runs two legs **concurrently** through the screen with run 0004's
  flags, Stage-I paramset and operating points (copied):
  - **off**: the selected candidate, all five windows, `shunt_ff_ledger=false`. It must
    reproduce run 0004 (relative tolerance 1e-9 on 15 window metrics), which shows that
    the two checkouts differ only in the ledger.
  - **on**: every union candidate with e_v_ts ≤ 1.02 × the best feasible one, plus the
    Stage-I point, ledger on. Windows with a release in run 0004 are re-simulated; the
    others are copied from run 0004 and marked `copied_from`, and the screen skips
    existing outputs. Then a re-selection with `ch_9_3_select.py` under run 0004's limits.
  - Outputs: `compare.json`, `compare.md`, and a verdict of HOLDS / CHANGES /
    INCONCLUSIVE. If another candidate ranks first, the verdict notes that it has not
    passed the settling confirmation (Experiment C is not repeated: it holds the TS
    controllers, so no bank switches).
  - Other options: `--plan-only` (nothing simulated); `--band`; `--workers` (default 11;
    the off leg gets min(5, workers/2)). The Stage-I paramset is imported from run 0004's
    checkout if missing, and never replaced if it differs.

## Verification (no simulation)

- `--dry-run` of the screen with `--config-set shunt_ff_ledger=false`: parsed and
  recorded. The six copied operating points were found as cached (net key `f8c419bf`),
  so nothing was rebuilt. An unknown field is rejected.
- `dataclasses.replace(make_config(), shunt_ff_ledger=False)` works on `MultiTSOConfig`.
- `--plan-only` on round 3: winner `1b945ad40b42`; 114 candidates in the band; 23 with a
  release; 62 windows.
- End-to-end test in the session scratchpad. All windows were pre-populated from run
  0004, so the screen ran zero units and only assembled the 115 records; then select and
  compare ran. Result: off leg "reproduced", ranking identical to run 0004, verdict HOLDS.
  This tests the plumbing, not the ledger.

## Auto-launch (author 2026-11-03, ~10:20: "Yes, auto-launch")

- Changes to the driver for an unattended launch:
  - **Final winner at the end.** The verdict re-reads run 0004's `DONE.txt` and
    `expC/criterion_results.json` once the screens finish. The leader at launch only
    sets the off-leg candidate: any candidate without a release shows the code
    neutrality equally well. Candidates that failed settling are dropped from the on-leg
    ranking. If the winner lies outside the band, the verdict is INCONCLUSIVE. While
    run 0004 is still unfinished, the verdict is marked "provisional".
  - **Workers.** Default 8: e5wimi's standing budget of 10 minus the 2 of run 0004's
    local W5 report. The split between the legs is proportional to their units (about
    1 off, 7 on).
  - Tested on fake data in three cases: leader holds; leader fails settling and the
    next one is selected; winner outside the band.
- **Watcher, detached:**
  - script `scratchpad\launch_ledger_check.ps1` of session 92595fe8, started 10:29:52
    as pid 62504;
  - log `scratchpad\ledger_check.log`;
  - it waits for `stage II-b ends after` in run 0004's `driver.log`, then starts
    `F:\python_environments\qOFO_clean\python.exe -u -X utf8 -m experiments.ch_09_parameter_selection.ch_9_9_ledger_check --workers 8 --note "..."`
    from `Z:\Python_Projekte\qOFO`, with stdout to `scratchpad\ledger_check.out`;
  - it launches nothing if `FAILED.txt` or `STOPPED_BY_AUTHOR.txt` appears, if a check
    is already running, or after 24 h.
  - Both paths (launch and FAILED) were tested on fake run folders.

## Not done / open

- Expected timing:
  - II-b round 4 ends ≈13:15–13:45, and the check starts then.
  - Cost on round-3 numbers: ≈41 + 3 worker-hours, so ≈6 h at 8 workers (the check has
    no remote shards). The verdict is expected ≈19:00–20:00.
  - Run 0004's settling confirmation and W5 run meanwhile on e5power and here (2).
- **Gate:** if the verdict is CHANGES, the selected paramset is in question, and D1/D2
  and everything after must wait. Recommendation: hold REF/MC (expensive) until the
  verdict; D1/D2 (≈1.3 h) may run in parallel at risk.
- Importing `stage1_v2_uc_2026_11_pq10.json` into `Z:` creates an untracked file. A
  later merge that adds the same file on the branch would stop on it: delete the
  identical untracked copy before such a merge.
- Nothing committed.
