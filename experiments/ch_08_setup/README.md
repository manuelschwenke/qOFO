# Ch. 8 — `ch:setup`: benchmark networks, reference models, evaluation methodology

Chapter number as of 2026-10-23; the label is the stable key. Results go to
`results/ch_08_setup/` (see `results/README.md` for the layout and
`experiments/thesis_runs.py` for the run folder).

| Sub-experiment | Script | Thesis target | Notes |
|---|---|---|---|
| `reference_period/` | `characterise_reference_period.py` | `ch:setup:scenarios:reference` → `fig:setup:reference`, `tab:setup:reference:events` | reads the L2 log of a `run_case1` run (default: the newest complete one); `--fig-dir` exports the figure into the thesis |

The Monte-Carlo protocol (`ch:setup:montecarlo`) is measured by
`experiments/ch_11_case2_multisystem/mc/run_mc.py`, which carries both labels.

Moved here from `experiments/reference_period/` on 2026-10-23.
