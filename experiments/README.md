# Experiments

The active entry points are deliberately small and descriptive:

- `run_multi_system_ofo.py` — manual multi-zone TSO-DSO OFO run.
- `comparison_none_vs_sbxh.py` — D0/D1/D2 comparison of no horizontal
  coordination, SBX-H contracts, and SBX-H planned support.
- `demonstrate_sbxv.py` — stressed SBX-V vertical-band demonstrator with
  live metering, request, grant, and remuneration visualization.
- `CIGRE_2026/005_CIGRE_MULTI.py` — V1-V5 control-architecture ladder.
- `CIGRE_2026/006_CIGRE_MONTECARLO.py` — paired Monte-Carlo extension.

`helpers/` and `runners/` contain shared implementation infrastructure.
`results_io.py` creates immutable numbered run directories with exact and
human-readable configuration snapshots. `archived/` contains frozen
historical experiments and is not part of the supported execution surface.

The active TSO-TSO comparison is `none` versus SBX-H. Physical tie-line
reactive power remains measured and recorded, but is not a separate
weight-based coordination objective.
