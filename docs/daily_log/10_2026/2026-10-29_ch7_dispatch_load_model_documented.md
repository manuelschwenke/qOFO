# 2026-10-29 — Ch 7 (`ch:setup`): active-power schedule and ZIP load model written up

**Type:** documentation (thesis text), no code change
**Chapter:** `Chapters/Chapter07.tex` (label `ch:setup`; was `Chapter08.tex` until the 2026-10-28 renumbering)
**Deliverable:** [`docs/ch7_dispatch_and_load_model.tex`](../../ch7_dispatch_and_load_model.tex)

## What was missing

The chapter states that an imbalance is shared over the machines "in proportion to
their rating" (`ch:setup:multi`, `ch:setup:multi:dynamic`) but never says **how the
scheduled active power of each machine is determined**. Neither the dispatch rule nor
the voltage-dependent load model appears anywhere in the thesis:

| Item | Status before |
|---|---|
| Load time series (SimBench, 40/60 split) | documented (`ch:setup:multi`, App. B) |
| Post-outage sharing ∝ rating | documented (`ch:setup:multi:dynamic`), verification table `[TBD]` |
| Active-power dispatch rule | **absent from the whole thesis** |
| ZIP load model (κ_p=1, κ_q=2 @ 1.03 pu) | **absent as equations** (only "ZIP-load separation" in App. B) |

Consequence: a reader cannot reproduce any operating point of Part III, and would
reasonably assume the published IEEE 39 base-case dispatch — which
`compute_zonal_gen_dispatch` overwrites in full.

## Method of the schedule (as implemented)

Per control area `a`, per step, from the profiles alone. `L_a` / `E_a` include the
buses of the STS attached to `a`:

- residual `R_a(t) = Σ_l P_l^0(t) − Σ_m P_m^ref λ(t)`
- pro rata on nameplate, capped: `P_i = min(R_a · S_n,i / Σ_j S_n,j , α S_n,i)`, `α = 0.85`, `P_min = 0`
- clipping residue redistributed ∝ remaining headroom, first intra-area (≤5 passes),
  then system-wide (≤5 passes, deficit re-measured **system-wide** — the per-zone
  measure was the historical over-dispatch bug)
- losses + anything above `Σ α S_n` → distributed slack

Post-event sharing is a **separate** mechanism: `slack_weight = sn_mva`, so
`ΔP_i = ΔP · S_n,i / Σ_in-service S_n,j`. Profile-borne changes (incl. the scripted
`load_step_*`) are re-dispatched; contingencies are not (contingency loads are created
with `base_p_mw = base_q_mvar = 0` precisely so the pre-computed table is untouched).

## Verification

The law reproduces the chapter's own dataset (`graphics/setup/reference_period_pf.csv`)
exactly:

| Check | Predicted | File |
|---|---|---|
| `ne_P` at t=0 (area-1 pro rata + loss share) | 1240.0 MW | 1240.0 |
| `trip_gen_P` at t=29.3 min (G4 = R_3·800/1500 + 9.58) | 562.4 MW | 562.4 |
| Equivalent's share of the 552.5 MW pickup | 74.1 % (10000/13500) | 73.7 % (Δ = loss change) |

## Corrections folded in

- **Generator Q bound is P- and V-dependent**, not the flat ±0.5 S_n I first stated.
  `core/actuator_bounds.py:compute_generator_q_limits` implements the stator / rotor /
  UEL limits of `ch3:sec3:gen` at the measured `(P,V)`. The flat ±0.5 S_n in
  `build.py:167` is only what the *plant* power flow enforces (`enforce_q_lims_plant`),
  and it coincides with the stator circle at the dispatch cap (p=0.85 → q=0.527).
  App. B needs a clause; see amendment 1 in the draft.
- **Not an economic dispatch.** Pro rata on nameplate has no merit-order justification
  and is not citable as one. Framed instead against `ch1:sec:scope:dispatch`
  ("Active-power setpoints are not dispatched"), which makes the schedule a declared
  boundary condition rather than a weak contribution.

## Open points found in passing (not fixed)

1. `multi_tso_dso.py:1836` — comment says "20 % of P_max", code computes `* 0.0`, so
   `P_min = 0`. Comment is wrong; behaviour is intended.
2. `data/profiles.csv` row 806, `09.01.2016 09:30`: `mv_rural_qload = "4,00E-06"`
   (German decimal comma) → parses to NaN → `apply_profiles` skips the Q update and the
   previous step's value is held. Benign mid-run; an MC episode *starting* there gets
   `Q_STS = 0` for its first step. One-character fix in the CSV.
3. `LOAD_PEAK_BOOST = 1.0`, so the documented +20 % stress headroom is inactive. Any
   claim of loads above the IEEE base case rests on the profiles alone.
4. `tab:setup:multi:outage` frequency row still `[TBD]`; Δf ≈ −0.100 Hz predicted
   (see amendment 5).

## Why the draft is here and not in the thesis repo

`Z:\latex_diss_clone` is 24 commits behind `origin/main` (which carries the 2026-10-28
renumbering and the N1–N21 notation sweep) **and** has uncommitted `Chapters/Chapter11.tex`
+ `graphics/Ch11/*` work. Origin's "Renumber graphics folders to chapter numbers"
commit (`d838ad3`) moves `graphics/Ch11`, so a pull would conflict with that uncommitted
work. The draft is written against `origin/main`'s notation and is ready to paste once
the clone is reconciled. Placement: new `\subsection` of `ch:setup:benchmarks`, after
`ch:setup:multi:configs`, before `ch:setup:multi:dynamic`.

## Inserted into the thesis (2026-10-29, later the same day)

The clone was reconciled with `origin/main` first (ff to `5054ae2`; the pre-existing
Ch 11 WIP preserved on `wip/ch11-step-tests-2026-10-29` = `130b826` and re-applied to
the renumbered `Chapters/Chapter10.tex` / `graphics/Ch10`). The draft then went in
directly, so `docs/ch7_dispatch_and_load_model.tex` is now a record, not a pending
deliverable.

Five edits, all in `Z:\latex_diss_clone`:

| # | File | Change |
|---|---|---|
| 1 | `Chapters/Chapter07.tex` | New `\subsection{Load Model, Time Series, and Active-Power Schedule}`, label `ch:setup:operation`, between `ch:setup:multi` and `ch:setup:multi:dynamic`. Equations `eq:setup:zip`, `eq:setup:profile_split`, `eq:setup:residual`, `eq:setup:dispatch`, `eq:setup:sharing`, `eq:setup:df`. |
| 2 | `Chapters/Chapter07.tex` | `ch:setup:multi` "Six modifications": rating clause now points at `eq:setup:sharing`. |
| 3 | `Chapters/Chapter07.tex` | `ch:setup:multi:dynamic`: the prose sharing law + its four conditions replaced by a pointer to `eq:setup:sharing` (they were duplicated by edit 1). |
| 4 | `Chapters/Chapter07.tex` | `ch:setup:scenarios:reference`: "as stated in `ch:setup:multi`" → "by `eq:setup:sharing`". |
| 5 | `Appendices/AppendixB_Benchmarks.tex` | The ±0.5 S_n envelope is now stated as the *power-flow* box, identified as the stator limit at the dispatch cap, with the controllers' operating-point-dependent curve (`ch3:sec3:gen`) named as what they actually use. |

Pre-insertion check: all 11 external `\cref` targets and the one citation key
(`meinecke.2020`) verified present in the current build.

Amendments 5 and 6 of the draft header are **not** done: the Δf ≈ −0.100 Hz prediction
for the `[TBD]` row of `tab:setup:multi:outage` still needs the PowerFactory run, and
the limitation sentence about a merit-order schedule relocating the active injections
(for `ch:conclusions:limitations`) is proposed but not written.

## Reframed as a simplified merit order (author, same day)

Committed as three commits on `main` in the thesis repo, none pushed:

| Commit | Content |
|---|---|
| `1034c5e` | Ch 10: the re-applied interface-error figure and Step Tests section |
| `c36ac26` | Ch 7 + App. B: the schedule and load model as equations |
| `0fe3ff6` | Ch 7: the schedule stated as a simplified merit order |

The rule is unchanged; its cost model is now stated, which turns the schedule from
a convention into an optimum. Three declared assumptions: IBG at zero marginal cost,
taken at availability; identical cost characteristic **per unit of rating** for every
SG; each area covers its own residual.

The middle one is what yields the implemented rule exactly. For `C_i ∝ P_i²/S_n,i`,
identical across machines, `dC_i/dP_i ∝ P_i/S_n,i`, so equal incremental cost gives
`P_i ∝ S_n,i`. Note the per-unit qualifier is essential: identical *absolute* cost
curves would give equal MW, not proportional MW. The third assumption is a constraint
on the merit order, not part of it — equivalent to zero scheduled inter-area active
exchange, which is the property the reactive-power study actually relies on.

### Correction to `c36ac26`

That text claimed the clipping shortfall is redistributed headroom-proportionally
"first inside the area and then across all machines". Wrong — the two passes differ:

- **intra-area** (`zonal_balancing.py`, `free_cap = sum(gen_p_max)`): proportional to
  `gen_p_max = α·S_n`, i.e. to **rating**. The equal-λ property survives clipping.
- **cross-area spill** (`head = gen_p_max - dispatch`): proportional to **headroom**.
  No cost model yields this; it is the single deviation from the merit order, and it
  exists so a capped area is served by other areas' machines rather than by the
  10 GVA equivalent alone.

### Reviewer caveat now stated

`GEN_NAMEPLATE` assigns fuel types (Nuclear, Coal, Hydro, Equivalent) and
`tab:app:ieee39:machines` reports them, but they parameterise `x_d` / `i_f_max`, not a
cost. The text now says the merit order does not distinguish them, instead of leaving
it as an apparent inconsistency.

Build after each step: `latexmk` exit 0, no hard errors, no undefined references in
the final pass.
