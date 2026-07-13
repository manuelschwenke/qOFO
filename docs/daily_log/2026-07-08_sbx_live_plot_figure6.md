# 2026-07-08 — SBX Figure 6 live plot + single-run demo (014)

**Task:** Manuel's request: a single-simulation script visualising SBX
requests and deliveries with the no-remuneration band, as a live plot
toggleable in config like the existing live plots; demonstrating the
013 scenario cases.

**Changed:**

- `visualisation/plot_sbx.py` (new): `SBXMechanismLivePlotter` —
  corridor tiles (measured reference-end flow, q_sched/q_std
  staircases, tier-1 band shading, deal/unwind/scarcity markers,
  need-flag strips), surplus staircases, cumulative payments. Window
  opens immediately with a placeholder; axes built at the
  contract-freeze tick (fixes "live plot not opening" — during warmup
  no corridors exist).
- `configs/multi_tso_config.py`: `live_plot_sbx: bool = False`.
- `experiments/runners/multi_tso_dso.py`: plotter construction beside
  the other live plotters (fail-fast unless coordination_mode="sbx"),
  per-step update passing the adapter and the per-step REFERENCE-END
  corridor flows computed from `net.res_line` at each tie's `bus_a`
  endpoint (q_from/q_to per orientation); handle exposed via
  `sbx_runtime["live_plotter"]`.
- `experiments/014_SBX_SINGLE_DEMO.py` (new): one run of any 013
  scenario, calibrated per-scenario band defaults, saves final PNG +
  settlement outputs + cycle table. Headless validation on `asym_z3`
  (150 min): exit 0, figure correct.

**Key finding (F7, recorded in STATUS §7.4):** `rec.zone_tie_q_mvar` /
`rec.tie_q_mvar` use −q_from for tie lines oriented from the higher
zone — ignores line charging (~107 Mvar on line 14), so the Figure-1
tie tile / 011 heat-map / zigzag diagnostics misstate that corridor's
at-endpoint flow. Not fixed (BME-era consumers); Figure 6 computes its
own flows. Docstring in `records.py` does not match the computation
for flipped charged lines.

**Also observed:** identical back-to-back 014 runs differ at the
~0.4 Mvar level (Gurobi tie-breaking nondeterminism); mechanism
behaviour unaffected, deal timing can shift by one cycle near
threshold.

**Status:** Figure 6 + 014 demo done and validated.
