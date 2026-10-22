# 2026-07-03 — Experiments-folder audit + BME ladder evaluation figures

**Session:** parallel to the BME Phase 6c sweep session (which owns the
D2 pairing / oracle-rung wiring in `011_BME_LADDER.py`; this session's
changes to 011 are purely additive — the plotting layer).

## 1. Experiments audit (which are necessary / deletable)

Reviewed all 12 numbered entries in `experiments/`. **Verdict: none
deletable.** 000–006 are the vertical-cascade / CIGRE-2026-paper line
(005's `make_cigre_config()` is the shared scenario imported by 006,
007, 011 and several diagnostics); 004/004b document the sensitivity
trade-off the BME design cites; 007–010 document the vref lineage
(divergent-schedule case study, mutual-gradient demo, loss×coordinator
sweep, heterogeneous strategies incl. the sticky-OLTC evidence) that
the BME chapter's motivation rests on. Classification recorded in the
new `experiments/README.md`.

## 2. Renames (scheme tagging, `git mv`, history preserved)

| Old | New |
|---|---|
| `007_TIE_COORDINATION.py` | `007_VREF_TIE_COORDINATION.py` |
| `008_TIE_MUTUAL_GRADIENT_DEMO.ipynb` | `008_VREF_MUTUAL_GRADIENT_DEMO.ipynb` |
| `009_TSO_LOSS_TIE_SWEEP.py` | `009_TSO_LOSS_VREF_SWEEP.py` |
| `010_TSO_HETEROGENEOUS_STRATEGIES_DEMO.ipynb` | `010_VREF_HETEROGENEOUS_STRATEGIES_DEMO.ipynb` |

Safety checked first: none of the four is imported anywhere (005 IS
imported by module name in 6+ places and was therefore NOT renamed;
011 already carries the BME tag). Self-referencing docstrings in the
two `.py` files updated with a "renamed from" note; result folders
(`results/009_loss_tie_sweep/` etc.) unchanged; old names in daily
logs / notebook markdown left as historical record.

## 3. Evaluation figures in `experiments/011_BME_LADDER.py`

New plotting layer (functions `fig1…fig5`, `make_all_figures`, CLI
`--plot`; figures also regenerate automatically after every `--rung`
run). Design decisions:

- **Rung-discovery from disk**: every figure includes exactly the rungs
  whose `records_<rung>.pkl` exists — the oracle rung (d) and its
  gap-to-oracle annotations activate automatically once the Phase 6c
  session lands it; no plotting change needed then.
- **Fixed colour identity per rung** (never re-assigned when a rung is
  missing): none = grey, vref = aqua, bme = blue, bme_loss = orange,
  oracle = black dashed (colours from the validated colourblind-safe
  categorical palette; annotation text in ink greys).
- `fig1_phi_losses` — Φ_global(t) and plant losses(t), 15-min rolling
  mean over faint raw traces, contingency markers from the shared 005
  schedule, last-hour window shaded, last-hour means at the right
  margin (collision-staggered via `_spread_positions`). Answers "does
  BME lower the common objective vs none / vs oracle".
- `fig2_voltage_envelope` — small multiples (shared y): per-rung
  system V envelope (min/max over zones) against the D2 soft band
  [1.01, 1.05]. Makes the bme_loss voltage escape (1.156 pu) vs the
  banded bme rung (1.063) directly visible — the D2 story figure.
- `fig3_discrete` — cumulative OLTC tap moves (all rungs) +
  hygiene-gate decision breakdown (accepted / ε-reject / slot-blocked
  from the ledger CSVs) + predicted-vs-realised ΔΦ scatter per accepted
  switch (§3.10.2 premise data; predictions converted from w_Φ-scaled
  units, sign-agreement annotated: bme 0.75, bme_loss 0.89 on the
  current 360-min pickles).
- `fig4_summary` — headline dashboard: last-hour Φ, loss reduction vs
  none (positive = better), OLTC moves, whole-run V range dumbbells vs
  the D2 band; per-rung gap-to-oracle annotation once oracle exists.
- `fig5_zone_phi` — per-zone Φ_i last-hour means (Phulpin fairness
  premise). Skips gracefully on the current pickles (they predate the
  `bme_phi_zone_mw` record field; populates on the next rung re-run).

Validated against the existing Phase 6b 360-min pickles
(`results/011_BME_LADDER/`): all figures render headless (Agg),
PNG + PDF.

**Reason:** Manuel's request — clean up / classify the experiments
folder and provide the "does BME work, how good vs central and vs
none" evaluation plots for the ladder.
