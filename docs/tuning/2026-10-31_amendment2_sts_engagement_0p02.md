# Amendment 2 to the weight-selection procedure: STS coupler engagement at 0.02 pu

**Status:** declared 2026-10-31, ~15:20, before any selection of the run it applies to.

**Decision.** The author asked for this declaration ("Declare 0.02 now").

**Amends:** `2026-10-28_weight_selection_procedure.md`, filter rule "the strongest STS coupler
engages inside the DSO operating band" (amendment 1).

**Applies to:** the re-application on the unit-commitment base, run
`results/ch_09_parameter_selection/0001_2026-10-31_135000_ch_9_6_full_procedure` (e5wimi).
It does not change the earlier result `ch9_selected_band_2026_10_a1` (c08b1d3d33ab).

## Rule

The bound of amendment 1 is tightened:

$$
\theta_{\mathrm{STS,min}}
= \min_{c \in \{W1..W6\}} \min_{i \in \mathrm{couplers}}
\frac{g_{w,\mathrm{dso\_oltc}}}{2\, g_{V,\mathrm{DS}}\, \big|\sum_j H_V^{(c)}[j,i]\big|}
\;\le\; 0.02\ \mathrm{pu}
\qquad (\text{was } V_{\max,\mathrm{band}} - V_\mathrm{ref} = 0.05\ \mathrm{pu}).
$$

- H_V comes from the operating points of the run, i.e. the 12-h captures of its Stage I.
- θ is evaluated with `tuning.stage1_screen.engage_deviation`, as before.
- The bound acts in **every** selection of the run:
  - after Stage II-a;
  - after each II-b round, where it chooses the centre of the next round;
  - in the final ranking before settling.
- Everything else is unchanged.

## Rationale

- **What θ means.** θ is the uniform deviation of the observed STS voltages from 1.03 pu at
  which the voltage gradient alone commits one step of the strongest coupler.
- **Why 0.05 is too weak.** At the old bound, a coupler commits only at the upper band edge
  (1.08 pu), where the soft band constraint takes over anyway. The bound therefore excluded
  only candidates that cannot engage inside the band at all; fde2e81af6b1, at 0.080, is one.
  It did not secure the author's intent stated with amendment 1: "the tap changers engage when
  many STS voltages (...) are far from 1.03 pu".
- **Why 0.02.** It is the level at which the author's check of intent (run 0008) defines
  "far": a far episode is |median deviation| ≥ 0.02 pu for ≥ 10 min.
  - With θ ≤ 0.02, a uniform deviation of that size suffices to commit the strongest coupler.
  - A non-uniform deviation needs more, because the gradient weighs the buses by sensitivity.

## Known when declared (stated for transparency)

- **Analytic θ values only** (no simulation needed), on the run's captures:
  - Stage-I point 0.0150, c08b1d3d33ab 0.0195, fde2e81af6b1 0.0816;
  - of the 115 Stage II-a survivors, 26 have θ ≤ 0.02 and 47 have θ ≤ 0.05 (median 0.068).
- **c08b1d3d33ab lies just below the new bound.** The value 0.02 comes from the intent check's
  "far" level, not from c08b. The author knew c08b's θ (0.019, run 0008) when deciding.
- **No simulation result of this run had been looked at.** That covers the ranking and the
  metrics of the Stage II-a units, which were running.
- **Known from before:** the re-validation run 0010 (14 candidates of run 0008, new base) and
  the toy-size smoke test, whose numbers carry no meaning.
- **A second amendment.** With amendment 1 the author wrote that, if the amended result did not
  show the intended engagement, the remedy would be "a re-application with a different
  strategy (...), not a further amendment".
  - This run is such a re-application, on a new base, and its rule is fixed before its first
    selection.
  - The thesis must nevertheless report both amendments, their dates and what was known at
    each.

## Execution

- The driver `experiments/ch_09_parameter_selection/ch_9_6_full_procedure.py` carries the bound
  as `THETA_MAX = 0.02`.
- The master was restarted with it during Stage II-a, about 9 h before its first selection.
  The remote shard kept running; the local shard resumed from its unit files.
- The result paramset records the bound in its provenance.
