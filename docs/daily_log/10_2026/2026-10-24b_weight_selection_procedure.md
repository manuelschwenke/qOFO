# Weight selection: one frozen two-phase procedure, applied once

Timestamp: 2026-10-24, ~15:00 Europe/Berlin.

Reason: author decision. Instead of a selection that rests on amendments, the
weights are to come from one well-designed procedure, applied once, whose
output is used as it is. The campaigns since 2026-10-16 (both Experiment B
protocols, run 1 and Amendment 1-alt) are the development of that procedure.

## What changed

| File | Change |
|---|---|
| `docs/tuning/2026-10-24_weight_selection_procedure.md` | New, frozen: Stage I (analytic point), Stage II-a (512 Sobol draws, screen, 12 h W1–W6), Stage II-b (local Sobol rounds of 64 draws around the current best, spans f × II-a spans, f = 0.5, recentre on a new best, halve otherwise, stop below 0.125 or after 4 rounds, seed 20260910 + r), selection over everything evaluated (filter incl. bank wear, ranking, settling requirement in rank order). Result name `ch9_selected_final_2026_10`. |
| `results/THESIS_ch8_weight_selection_20261024/driver.py`, `repro_unit.py` | Untracked campaign driver: copies the Stage II-a records, runs the reproduction check, Stage II-b, selection and settling checks, writes the result. |

No code in the repository changed.

## Why Stage II-a is reused

Stage II-a with seed 20260910 on this configuration is deterministic, so its
records of 2026-10-22/23 are what a re-run would produce. The driver checks
that before anything else: two windows (Stage-I point W1; II-a winner W5, with
bank commits and tap hunting) re-simulated must reproduce every stored metric
exactly, else it stops. Code since that run (a3ce523..HEAD): the only `.py`
changes on the simulation and selection path are docstrings and comments
(`ch_9_1_ninner_isolated_sts.py`, `tuning/parameters.py`,
`ch_9_parameter_selection/__init__.py`); the experiment-folder reorganisation
does not touch it. Settling results already measured for II-a candidates are
reused on the same grounds.

## Checks before launch

- Loop logic tested offline with stubbed selection and simulation: recentring,
  halving, stop below f = 0.125, cap at 4 rounds, settling in rank order with
  the band re-anchored after a failure.
- Round 1 draw-and-screen on the real operating points: 32 / 64 draws pass
  (seed 20260911), the Stage-I point's 6 window records are copied for the
  explicit-mode baseline.

## Expected cost

Reproduction check ~40 min; Stage II-b ≤ 4 rounds × ~4–5 h; settling checks
~1.5–3 h.
