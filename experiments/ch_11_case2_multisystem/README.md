# Ch. 11 — `ch:case3`: Case Study 2, multi-system OFO in a three-area configuration

Chapter number as of 2026-10-23; the label is the stable key. Results go to
`results/ch_11_case2_multisystem/<sub-experiment>/` (see `results/README.md`).
Plan of the campaign: `~/.claude/plans/i-now-want-to-composed-island.md`
(approved 2026-10-27); change record `docs/daily_log/10_2026/2026-10-27e_*`.

| Sub-experiment | Script | Thesis target | Status |
|---|---|---|---|
| — | `variants.py` | `tab:case3:variants` | the variant registry: L2, S1, O2, O3, M on one base (selected paramset, DSO_3 ×2, minimum-class metering for every controller and the droop, no event inherited from `make_config`), pair guard `assert_case2_controlled`. S2 waits for the author (design note `docs/architecture/2026-10-27_S2_design_note.md`) |
| — | `metrics_case2.py` | `ch:case3:setup:metrics` | area, STS, interface, recruited-flow and event metrics (Δe_post from the pre-control baseline) |
| `ref/` | `run_ref.py`, `scenarios.py` | `ch:case3:nominal` → `fig:case3:vtrack`, `fig:case3:qtrack`, `fig:case3:tieflow` | live; the reference scenario (A/B/C, one machine + one line + one load step between dispatches) is chosen by the author after a screening. One worker per process; on e5power one worker per WinRM session until detached launch works |
| `mc/` | `run_mc.py` | `ch:case3:montecarlo`, `ch:setup:montecarlo` → `tab:case3:mc`, `fig:case3:mc`, `tab:case_svr:stats` | to be moved onto `variants.py` (Phase 5 of the plan); the 2026-10-07 slice (old paramset, noise on the OFO only) cannot be paired with new runs |
| `mc/` | `area1_pilot_weight.py` | `ch:case3:montecarlo` | diagnostic of the first slice |
| `ladder/` | `variant_ladder.py`, `figures.py`, `preview.py`, `export_tikz_data.py`, `s1_pi_screen.py`, `s1_pi_candidates.py`, `ladder_mc.py` | — | **superseded** seven-rung design, see `ladder/README.md` |

**M (central bound).** At the selected weights the single central loop has
λ_max = 5.11 and does not contract; its whole `g_w` block is scaled by one
factor κ to the selected design's `lambda_max_B` from the paramset provenance
(κ = 3.615 → λ_max 1.414 with `ch9_selected_final_2026_10`). Author decision
2026-10-27; κ is printed and recorded (`central_gw_kappa`).

The paramset blocker of 2026-10-23 is resolved: `run_case1.PARAMSET` follows
`SELECTED_PARAMSET`.

Moved here on 2026-10-23: `mc/` from `experiments/part3_mc/`, `ladder/` from
`experiments/ch_10_case_study/` (files renamed without their `ch_10_N_`
prefixes).
