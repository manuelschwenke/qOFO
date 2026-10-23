# Ch. 11 — `ch:case3`: Case Study 2, multi-system OFO in a three-area configuration

Chapter number as of 2026-10-23; the label is the stable key. Results go to
`results/ch_11_case2_multisystem/<sub-experiment>/` (see `results/README.md`).

| Sub-experiment | Script | Thesis target | Status |
|---|---|---|---|
| `mc/` | `run_mc.py` | `ch:case3:montecarlo`, `ch:setup:montecarlo` → `tab:case3:mc`, `fig:case3:mc`, `tab:case_svr:stats` | live; one run folder per campaign, later variants (S2, O3, M) join it as invocations; server procedure in `mc/README_server.md` |
| `mc/` | `area1_pilot_weight.py` | `ch:case3:montecarlo` | diagnostic of the first slice |
| `ladder/` | `variant_ladder.py`, `figures.py`, `preview.py`, `export_tikz_data.py`, `s1_pi_screen.py`, `s1_pi_candidates.py`, `ladder_mc.py` | — | **superseded** seven-rung design, see `ladder/README.md` |

**Blocked on 2026-10-23:** the campaign uses the Case 1 configuration, which
needs a parameter set with `g_w_tso_shunt` (see `../ch_10_case1_svr/README.md`).

Moved here on 2026-10-23: `mc/` from `experiments/part3_mc/`, `ladder/` from
`experiments/ch_10_case_study/` (files renamed without their `ch_10_N_`
prefixes).
