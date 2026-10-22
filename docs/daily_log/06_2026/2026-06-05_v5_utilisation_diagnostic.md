# 2026-06-05 — Diagnostic: why V5 gen-Q utilisation > V4

**Timestamp:** 2026-06-05 (local)
**Author:** Manuel Schwenke / Claude Code

## Reason / motivation
In 006 Monte-Carlo (n=100) the centralized V5 shows higher generator-Q
utilisation (`res_util` 0.21) than the cascaded V4 (0.14), despite **identical**
generator penalties (`g_w_gen=5e7`, `g_z_q_gen=1e2`). Decompose the cause by
controlled re-runs on already-accepted 006 scenarios (paired by seed).

## What was added
New throwaway diagnostic **`experiments/diag_v5_util.py`** (no changes to 005/006
beyond the earlier `m_bar_pu` work). Reproduces a scenario from its seed
(`random_start_time` → `build_random_schedule`, exactly as 006), runs a variant,
and computes from the full in-memory log: Table-3 metrics (via
`cigre_summary_table`) **plus** a breadth measurement — `frac_engaged` (share of
machines with per-machine util > 0.05) and per-machine time-mean utilisation
keyed `z<zone>k<k>`, annotated with `S_n`. One variant per process; aggregate
with `--aggregate`. Outputs under `experiments/results/diag_v5/`.

Variants: `V4`, `V5` (refs); `V5_slowgen` (`central_period_s=180`, cadence test);
`V4_notie` (`tso_g_q_tie=0`, inter-zone tie-Q test); `V5_noHVv`
(`central_dso_g_v=0`, HV-voltage-tracking test).

## Result (5 paired seeds, all converged)
| variant | res_util | m_bar_pu | frac_eng | tieQ | n_sw | rms_v |
|---|---|---|---|---|---|---|
| V4 | 0.138 | 0.375 | 0.76 | 28.6 | 11.6 | 0.00731 |
| V4_notie | 0.140 | 0.374 | 0.76 | 27.6 | 12.4 | 0.00727 |
| V5 | 0.226 | 0.375 | 0.81 | 28.1 | 7.6 | 0.00687 |
| V5_slowgen | 0.233 | 0.370 | 0.80 | 29.0 | 8.2 | 0.00755 |
| V5_noHVv | 0.213 | 0.376 | 0.79 | 28.3 | 5.6 | 0.00680 |

Per-machine util: z2k0 (800 MVA) 0.17→0.41 and z3k0 (800 MVA) 0.08→0.26 between
V4 and V5; big z1k1 (10000 MVA) and z1k0 (1000 MVA) unchanged (~0.05 / ~0.15).

## Conclusion
- **Cadence: refuted** (V5_slowgen ≈ V5).
- **Inter-zone tie-Q: refuted** (V4_notie ≈ V4).
- **HV-voltage tracking: minor** (~15% of the gap; V5_noHVv 0.213).
- **Structural interface-Q decomposition: the cause (~85%).** The extra reactive
  is concentrated on the TN machines over the load-heavy zones (z2 = 68 loads/1
  machine; z3). The cascade dispatches PCC/interface-Q setpoints (DSO tracks
  them), mediating the TN↔DS boundary and shielding those machines; the
  centralized OFO has no interface-Q variable and freely draws reactive from the
  overlying TN machine into the distribution networks. Not a uniform breadth
  effect (frac_engaged saturated at 5 machines).

## Open / next
- Distinguish "interface-Q decomposition" from "V5 under-uses DSO DER" by adding
  a DER-utilisation measurement to the diagnostic.
- `experiments/diag_v5_util.py` + `experiments/results/diag_v5/` are diagnostic
  artifacts; delete if not wanted in the repo.
