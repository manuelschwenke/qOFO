# Ch. 10 — `ch:case_svr`: Case Study 1, OFO in place of pilot-node SVR

Chapter number as of 2026-10-23; the label is the stable key. Results go to
`results/ch_10_case1_svr/` (see `results/README.md`).

| Script | Role | Thesis target |
|---|---|---|
| `run_case1.py` | producer: K1-REF (L2, S1m, S1, O0, O1, O2) and K1-PAR, one run folder with a sub-folder per variant | `ch:case_svr` → `tab:case_svr:reference`, `fig:case_svr:equivalence`, `fig:case_svr:pareto` |
| `report_case1.py` | metrics, post-event growth, equivalence; writes into the run it reads | |
| `figures_case1.py` | the two figures; `--out <thesis>/graphics/case_svr` | |

`run_case1.resolved()` is also the configuration of the Monte-Carlo campaign
(`ch_11_case2_multisystem/mc/run_mc.py`) and of the reference-period
characterisation (`ch_08_setup/reference_period/`).

**Blocked on 2026-10-23:** paramset `ch9_selected` has no `g_w_tso_shunt`,
which has been a required weight since 2026-10-22, so every variant stops at
the start.

Moved here from `experiments/case1_svr/` on 2026-10-23.
