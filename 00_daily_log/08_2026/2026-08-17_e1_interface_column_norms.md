# 2026-08-17 — E1: interface vs DS-DER sensitivity columns (the aggregation claim, measured)

**Timestamp:** 2026-08-17, Europe/Berlin
**New file:** `analysis/e1_interface_column_norms.py`
**Outputs:** `results/e1_column_norms/{e1_column_norms.csv, e1_summary.csv, e1_summary.txt, e1_columns.tex}`
**Reason:** thesis checklist item E1. The architecture argument in Ch 6 claimed that a cascade recruits distribution-side reactive capability a flat controller leaves idle *because* an individual DS-connected DER has a weak sensitivity column for transmission bus voltages while an interface setpoint has a strong one. That claim had never been measured.

## Method

Build the plant and controllers of variant V4 at t0 via `run_multi_tso_dso(cfg, pre_loop_hook=...)`, aborting before the time loop. Then, on the combined plant, take central finite differences of ±1 Mvar reactive injection at each actuator bus and record the response of two row sets:

- **far field** — the 30 monitored TS voltage buses of the TSO controllers;
- **local** — the monitored HV buses of the actuator's own subordinate network.

The probe is an auxiliary zero-P load created at the bus, so the perturbation is identical regardless of what is connected there. `run_control=False` throughout: no local Q(V) loop and no controller reacts, so the numbers are the pure network response — which is what a claim phrased as "physics, not tuning" has to be tested against.

Inventory: 12 interfaces, 40 DS-connected DERs, 30 TS buses, 40 HV buses. Runtime ~40 s including the 31 s controller build.

## Result — the claim as stated is false

| | far field (TS rows) | local (own HV rows) | local / far |
|---|---|---|---|
| interface, median | 3.56e-4 pu/Mvar | 3.98e-4 | **1.10** |
| DS DER, median | 3.46e-4 pu/Mvar | 1.05e-3 | **3.23** (2.11 – 7.84) |
| ratio | **1.03** | 2.64 | |

**The far-field columns are the same size.** Reactive power injected behind a coupling transformer arrives in the transmission network as reactive power; the far-field voltage response is set by the TS network's own impedances and does not record where behind the boundary the injection originated. Electrical distance attenuates the *local* response, not the far-field one — and the far-field one is what the supervisory objective is written on.

## What the mechanism actually is

The families differ in **collateral cost, not in delivered effect**. One Mvar at an individual DER disturbs that unit's own HV buses ~3.2× as much as it moves the TS buses it was meant to help; the same Mvar commanded at the interface disturbs them ~1.1×, because the subordinate controller realises the boundary flow with whatever internal combination leaves its own network where it wants it.

A controller carrying both TS and HV buses in one aggregate objective therefore prices every DER move against a larger local excursion — the move is partly self-cancelling. The cascade never poses that trade-off. **The mechanism is objective structure and delegated realisation, not attenuation by electrical distance.**

A second surviving asymmetry, not measured here: an interface column is bounded by the CAIR interval of a whole distribution network, each DER column by its own device rating (Σ sn_mva = 1640 MVA over the 40 units). Equal column magnitudes do not imply equal reach.

## Follow-up sweep (same day)

`--sweep 9` repeats the whole measurement at nine instants spanning the campaign's profile window (2016-01-05 08:00-13:00, 4.6-4.8 GW load). Eight converged; 09:52:30 did not converge under `run_control=False` and was skipped.

| | median | range over instants |
|---|---|---|
| far-field ratio interface/DER | **1.037** | 1.029 - 1.043 |
| collateral ratio, DS DER | **3.25** | 3.23 - 3.31 |
| collateral ratio, interface | **1.11** | 1.10 - 1.11 |

Both statistics are near-constant over the window. The falsification is therefore not a property of t0: the far-field columns are the same size at every operating point tested, and the collateral asymmetry is a structural property of where an actuator sits relative to the boundary. Ch 6 now quotes these as ranges.

## Risks / unresolved

- One instant of nine failed to converge. Not investigated; it is a plain profile draw, so a convergence failure there is itself worth a look (the reachability guard exists for exactly this).
- `run_control=False` was deliberate. The closed-loop variant (`--closed-loop`) includes the local Q(V) reaction and will shift both families; it has not been run.
- The measurement does not by itself establish that V5 leaves capability idle — it establishes the *reason* a flat objective would. The behavioural claim is the V4-vs-V5 comparison of the case study.
- The 40 DERs are the synthetic HV sub-network units; their placement is a modelling choice of the benchmark, so the collateral ratio inherits that choice.

## Consequence for the thesis

`Chapters/Chapter06.tex`, `ch:architectures:cascade:aggregation` was rewritten the same day: it now reports the falsified conjecture explicitly, gives the measured numbers, and rests the argument on the collateral asymmetry. The prediction handed to the case study is unchanged.

## Closed-loop channel (`--closed-loop --sweep 5`), same day

Repeated with `run_control=True`, so the local q(v) layer reacts during each perturbation. This is the channel a controller's H is actually built against, since the droop layer belongs to the plant (thesis `ch:droop:qcor`). Results in `results/e1_column_norms_cl/`.

| instant | load | far-field ratio | collateral DER | collateral iface |
|---|---:|---:|---:|---:|
| 08:00 | 4743 MW | 1.087 | 3.40 | 1.07 |
| 09:15 | 4807 | 1.427 | 5.83 | 0.94 |
| 10:30 | 4860 | 1.879 | 7.18 | 0.78 |
| 11:45 | 4673 | 1.246 | 3.49 | 1.00 |
| 13:00 | 4729 | 1.412 | 5.61 | 0.93 |
| **median** | | **1.412** | **5.61** | **0.94** |

**With the local layer active the families DO separate** — 1.09–1.88, widening with load — and very unevenly: at t0 the weakest DER column is 5.6e-5 pu/Mvar against a median of 2.9e-4, about a fifth.

The cause is still not electrical distance. It is **self-cancellation through the droop layer**: a unit that injects reactive power raises its own terminal voltage, its droop-controlled neighbours answer by absorbing, and part of the injection never leaves the distribution network. An interface command does not suffer this, because what the subordinate controller delivers at the boundary is a coordinated move of many units rather than one unit acting alone. Same local channel as the collateral measurement, now acting through the droop rather than through the objective.

Note the interface collateral ratio drops **below one** (0.78–1.07) in this channel: commanding the boundary disturbs the HV buses less than it moves the TS buses.

**Both channels are reported in the thesis.** The open-loop one kills the distance argument; the closed-loop one is more favourable to the architecture, and quoting only it would turn the measurement into advocacy.
