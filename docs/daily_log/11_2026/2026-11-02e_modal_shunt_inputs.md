# 2026-11-02 — Experiment A: MSC/MSR step added to the modal settling analysis

## Task (author, 2026-11-02)
Table 8.2 (`tab:param:timescales:actuators`, `ch:timescales:plant`) had no row for a
switched-shunt step. The text already claimed that "a shunt [is stepped] by one step",
with the comment "the shunt step has not been run". The README of the 2026-09-23 bundle
lists "Shunt (MSC/MSR) inputs" as *not established*: "their admittance scale in PF's
network equations is not established".

## Approach
No new PowerFactory run. The Jacobians of the 2026-09-23 modal campaign (W1–W6 × db_wide/
db_zero, `qOFO_2027/results/screening/W*_db_*/20260923-*/matl/`) are still on disk.
The shunt step is added there as an extra input column.

1. **Admittance scale (the open point).** PF writes each terminal's network equations as
   Re/Im current balances (rows `ur`, `ui`).
   - Compared with pandapower's `makeYbus` (baseMVA 100, star bus of the trafo3w
     Kron-reduced): the PF diagonal at all 12 coupler tertiaries is 100.00002 × the
     pandapower value. The off-diagonals give the same ratio.
   - So PF's admittances are on a 1 MVA base.
   - The row structure (G, −B; B, G) holds to 1e-10.
   - A bank step ΔB adds jΔB to Y_kk, so the input column is ΔB·(−u_i, u_r) on the two
     terminal rows.
2. **PF vs Python model.**
   - Bank data agree: 25 Mvar per step, 2 steps, 20 kV rated = terminal nominal voltage.
   - **Location differs.** The PF model of the campaign still has MSC and MSR co-located
     on the T0 tertiary (before 2026-10-15). The thesis has MSR on T0 and MSC on T1.
   - Every bank is at step 0 in all six windows, so the operating point does not depend
     on the location. The MSC step is therefore applied at the T1 tertiary.
   - The code checks every one of these points and raises an error on a mismatch.
3. **Analysis.** Same outputs (70 V, 12 interface Q), bands (1e-3 pu, 1 Mvar), residue
   settling and ODE cross-check as the existing rows; taps-out system.

## Code changes
- `analysis/modal_observability.py`:
  - module doc: shunt paragraph replacing "Shunts are not included".
  - constants `PF_ADMITTANCE_BASE_MVA = 1.0` and `SHUNT_SCALE_RTOL`.
  - `shunt_placement()`: placement and step size from the `MultiTSOConfig` defaults.
  - `shunt_inputs()`: input columns, per-terminal scale check, parity table PF vs thesis.
  - `shunt_lf_check()`: final value vs pandapower load flow, one-step and central
    difference.
  - `run_shunts()`: shunt inputs only; writes into its own folder.
  - CLI flag `--shunts`.
  - `run()` and the dispatch input set are unchanged.
- `experiments/ch_09_parameter_selection/ch_9_1_modal_shunt_inputs.py` (new): documented
  driver (`start_thesis_run`, target `ch:timescales:plant`, feeds
  `tab:param:timescales:actuators`), 3 worker processes, aggregation and checks.

## Run
`results/ch_09_parameter_selection/0012_2026-11-02_163334_ch_9_1_modal_shunt_inputs/` (0011 archived, see below)

| input | max T_s | T_d + T_s | binding pair | mode at the crossing |
|:--|--:|--:|:--|:--|
| MSC +25 Mvar | 6.06 s | 7.06 s | DSO_4 -> V_TN_bus18, W4/W5 droop inert | G 04 field (τ 6.2 s) |
| MSR −25 Mvar | 6.10 s | 7.10 s | DSO_4 -> V_TN_bus18, W5 droop inert | G 04 field (τ 6.2 s) |

- At the 6.1 s crossing, G 04 field = 0.85 × band and the 0.77 Hz mode = 0.38 × band. The
  0.77 Hz mode is only the dom@20 s mode (0.04 × band).
- Droop on its slope: ≤ 0.5 s, except W4 (1.58 s) and W5 (identical spectra).
- T_STS = 20 s is unchanged; the binding row is still the machine-transformer tap at 17.7 s.

Checks:
- A vs `Amat` ≤ 1.9e-12.
- Scale deviation 2.03e-7.
- ODE ≤ 9.1e-5 of the band.
- 0 unstable modes.
- Load flow, central difference: ≤ 0.085 Mvar and ≤ 0.58 mpu. The one-step difference
  (≤ 1.0 Mvar) carries the second-order ΔB·ΔV term.

## Dissertation
- `Chapters/Chapter08.tex`:
  - new table row "MSC or MSR, one step of 25 Mvar | Field winding, G4 | 6.2 | 1 + 6.1";
  - the method sentence now names the 25 Mvar step and states how the step enters the
    linearisation;
  - one result sentence;
  - the "needs a re-run" comment is replaced by a provenance comment;
  - the RESULT PROVENANCE header is pasted above the Experiment A paragraph.
- `thesis_runs check`: 8 blocks, 0 failing.
- Data: `graphics/Ch8/data/2026-11-02_modal_shunt/` (README, CSVs).

## Provenance fix and re-run (author: "commit and re-run")
- Run 0011 recorded no code state ("Code: unavail"): git refused the UNC checkout
  (`safe.directory`) inside `thesis_runs._git`, and the driver was untracked.
- `experiments/thesis_runs.py::_git` now calls `git -c safe.directory=* ...` for that call
  only (read-only commands). `tests/experiments/test_thesis_runs.py`: 15 passed.
- Commit 1013a73 holds the three code files.
- Run of record: `0012_2026-11-02_163334_ch_9_1_modal_shunt_inputs`.
  - Code: 1013a73 on minimal/thesis-core, dirty with 4 files from another session
    (INDEX, visualisation/live_*.py), none of which enters the analysis.
  - Numbers are identical to 0011 (max difference 0.0); 0011 is archived with `--by` 0012.
- Branch check (author's question):
  - minimal/thesis-core is 81 ahead of and 0 behind its upstream (`origin` = qOFO_2027).
  - Not merged: 3 commits on res/commitment-base-2026-10-31 (ch_9_6 resume, scarcity
    figure) and 1 docs commit on case2/ref. None touches the modal code, the snapshot
    loader or `thesis_runs`.
  - At HEAD, `analysis/modal_observability.py`, `pf/modal_participation.py`,
    `export/dynamic_snapshot.py` and `pf/naming.py` equal the campaign commit 781a819,
    ignoring line endings.

## Tap rows below 0.5 s (author question, 2026-11-02)
The caption of Table 8.2 called 0.5 s "the time resolution of the evaluation". That was
inaccurate: 0.5 s is `T_MIN`, the start of the modal evaluation. The grid is 0.02 s up to
60 s. Below 0.5 s the fast repeated eigenvalues (e.g. 84 identical 10 ms lags, cluster
condition up to 1e16) cannot be decomposed reliably. The coupler and bus-12 tap rows
("5 + 0.5") are therefore upper bounds.

**Check (scratch computation, not a documented run).**
- Method: the taps-in system of each window, exactly as in `run()` (`tapsin_system`,
  slack-machine tap skipped), coupler (`TAP_NC3W`) and bus-12 (`TAP_other`) taps only.
  Exact zero-order-hold discretisation with `expm([[A, B], [0, 0]] * 0.01)`, no
  eigen-decomposition. Simulated `y = C x + D` for 0–3 s against `y_inf` of `analyse()`.
- Coverage: W1–W6, both droop cases, 14 taps × 82 outputs = 13 776 pairs.

| tap | droop inert | droop on its slope | pairs outside the band after 0.5 s |
|:--|--:|--:|--:|
| coupler | 0.00 s | ≤ 0.10 s (W4, TAP_NC3W_DSO_3_t7 -> its own Q: jump 22 × band, final 14 × band) | 0 |
| bus 12 | 0.00 s | ≤ 0.03 s | 0 |

The largest error at 0.5 s is 0.87 × band (W4, droop on its slope).

**Author decision: fix the wording, keep 0.5 s.**
- Chapter08.tex: the caption now reads "Settling is evaluated from 0.5 s after the step
  on; shorter settling times are given as ≤ 0.5 s".
- The cells now read `$5 + {\leq}\,0.5$`.
- The text sentence ("settle within 0.5 s of contact") was already correct as a bound.
- A LaTeX comment records the check. The values 0.10 s / 0.03 s are not quoted in the text,
  for two reasons: there is no documented run, and below ~0.1 s the RMS model (phasors,
  instantaneous tap contact, 10–20 ms converter lags) does not support that precision.
- T_STS is unaffected: 5 + ≤ 0.5 ≤ 5.5 s, against the binding 17.7 s.

## Open
- **Base.** All rows of Table 8.2 come from the 2026-09-23 linearisations (snapshots of
  2026-09-08). The commitment base of 2026-10-31 is not linearised.
- The PF project still carries the co-located banks. A re-sync from a current snapshot is
  needed before any RMS time-domain run with switched banks.
