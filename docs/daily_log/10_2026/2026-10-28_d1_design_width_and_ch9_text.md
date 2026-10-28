# D1 design width 0.010 pu (run 0007) and the Ch 9 D1/D2 text

Timestamp: 2026-10-28, ~09:20–10:40 Europe/Berlin.

**Reason:** author decision (2026-10-28): "then lets use 0.01". The
drift-based rule gives the smallest admissible half-width, 0.0075 pu. The
design width is one candidate step above it, as a declared margin for:
- what the QSS traces do not contain (infeed fluctuations below the 15-min
  profile step, measurement noise);
- the fact that 0.010 pu is the width at which Experiment B selected the
  weights.

## Code (qOFO, 8def736)

`ch_9_4_deadband.py`:
- `DESIGN_STEPS = 1`;
- `selection.json` records `delta_rule_pu` (0.0075) and `delta_design_pu`
  (0.010);
- stage 2 tests the design width: D1 confirmation at 0.010 against its
  narrower neighbour 0.0075, and D2 legs at 0.010 / 0.020 / off.

## Run of record

`results/ch_09_parameter_selection/0007_2026-10-28_093138_ch_9_4_deadband`
(09:31–10:05; stage 1 reused from 0006, identical traces). Run 0006 is
archived, superseded by 0007.

- **D1:**
  - Drift max 0.00359 pu, giving Δ\* = 0.0075 pu and design ΔU_db = 0.010 pu.
    The rule as written would give 0.020 pu.
  - Confirmation at 0.010: 0 drift activations. Discrete-event activations
    by STS converters only, ≤ 20 s each: W2 2 converters (≤ 2.0 Mvar), W4 6
    (< 0.01 Mvar), W5 10 (0 Mvar).
  - W1, W3 and W6 are identical to the reference; elsewhere the deviation
    is ≤ 0.0046 pu and 0.39 Mvar, with the same tap and bank operations.
  - At 0.0075: 8 / 10 / 22 converters, deviation ≤ 0.0067 pu and 0.71 Mvar.
- **D2 at 0.010 (W1–W3, W6):**

  | Disturbance | Support | Reduction of the voltage step |
  |---|---|---|
  | Bus 8 (pp 7) | 78–85 Mvar | 31–34 % |
  | Bus 12 (pp 11) | 70–82 Mvar | 14–22 % |

  - At 0.020: 33–40 and 17–28 Mvar, reduction 14–16 and 5–12 %.
  - W4: 46–52 Mvar. W5: 0.
  - Hand-over ≤ 360 s.
  - With the droop off, an STS bus reaches 1.114 / 1.110 pu after the first
    TS dispatch (inductive events, W2 and W6); with the droop on it stays
    ≤ 1.083 pu.

## Thesis (`Z:\latex_diss_clone\Chapters\Chapter09.tex`)

- **Method text of `ch:timescales:deadband`:**
  - Normal operation = continuous movement plus discrete steps; the dead
    band is sized for the continuous movement.
  - The start-up phase (30 min) is reported separately; an interval is
    evaluated only up to a discrete step.
  - ε_U = 0.003 pu (HV measurement chain, class 0.2 transformer + class
    0.1 meter).
  - Δ\* is the smallest admissible half-width; the design ΔU_db is one step
    above it.
  - The confirmation runs at ΔU_db and Δ\*.
  - D2: legs ΔU_db, 2ΔU_db and off; inductive and capacitive 200 Mvar steps
    at **IEEE buses 8 and 12** (the code's pandapower indices 7 and 11),
    10 s after a dispatch; the tested width is active from the end of
    start-up.
- **Overview table, D1 row:** adapted.
- **D1 and D2 results:** filled in, plus the new table
  `tab:param:deadband:d2`.
- **Provenance:** the block of run 0007 is under the section heading.
  `thesis_runs check`: 3 blocks, 0 failing.
- **No project history in the text** (author convention). The 0.020 pu of
  the rule as written appears as a sensitivity in the results; the history
  of the decision is in a LaTeX comment.
- **Open (left as `\todo`):** the D2 trajectory figure for two
  representative cases.

## Notes

- **Test build:** a first attempt used `latexmk -pdf`, which forces
  pdfLaTeX. The repo's `.latexmkrc` selects LuaLaTeX, and the build failed
  on `\symbf`. That attempt also rewrote the tracked `ClassicThesis.bbl` and
  `.blg`; both were restored with `git checkout`, so no tracked file other
  than `Chapter09.tex` changed. The build was redone in a `git archive` copy
  with the repo's own `.latexmkrc`, because that rc copies the finished PDF
  back beside `ClassicThesis.tex`.
- **Noted for the author, not changed:**
  - The thesis noise section (`ch:setup:scenarios:noise`) describes Gaussian
    σ_v, while the code uses accuracy classes split into bias and a random
    part.
  - The Ch 9 source comments name the weight-selection disturbance buses
    with pandapower indices (5, 27 = IEEE 6, 28).
  - The W5 sentence at Ch 9 l. ~371 ("no weight can change its outcome") is
    still unchanged.
