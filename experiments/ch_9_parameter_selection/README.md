# Ch. 9 — Parameterisation of the control hierarchy

Entry points producing the numbers of thesis Chapter 9. One module per
section, named after the section it fills, so that every reported value has a
single traceable producer.

| section | quantity fixed | script | status |
|---|---|---|---|
| 9.1 | `T_DS`, `T_TS` (Table 9.1) | `ch_9_1_timescale_seperation.py` | ready, **not yet run** |
| 9.1 | `N_inner` (eq. 9.2) | — | **no script yet**, see below |
| 9.2 | dead-band half-width `ΔU_db` | `experiments/run_deadband_n1_multiwindow.ps1` (not moved) | done, thesis filled |
| 9.3 | objective / step-size weights | `tuning/` | pending |
| 9.4 | shunt persistent-need threshold | — | pending |

---

## 9.1 — Timescale separation (Table 9.1)

`ch_9_1_timescale_seperation.py` is the open-loop settling battery. It drives
the RMS benchmark one actuator at a time and measures how long every
controlled output takes to re-enter a fixed band around its final value,
measured **from the dispatch instant**, so a tap changer's mechanical travel
is counted against the period it consumes.

Bands (quoted verbatim in the caption of Table 9.1): `1e-3` pu on voltages,
`1` Mvar on interface flows.

### Two things it does, one it does not

* **Dispatch rows** — the largest single command per actuator class. These
  bound `T_DS`: the period must outlast the transient the controller's own
  action excites.
* **Disturbance rows** — machine outage, load step. These bound *nothing*.
  They establish for how many dispatch periods after an event a controller
  still samples a transient rather than a settled plant.
* **`N_inner` is not measured here.** It is a *closed-loop* property of the
  isolated DSO-OFO (parent silent, capability-band-traversing setpoint step)
  and needs its own entry point in this folder. The summary prints
  `T_TS/T_DS = 9` as the **configured** ratio and says explicitly that it is
  not a measurement — do not quote that line for eq. (9.2).

### Running it

Needs PowerFactory 2025 SP4 with a **free licence seat**, project
`IEEE39_qOFO`, study case `02_RMS_CoSim`, and the PF GUI closed. Expect
30–60 min wall clock.

Check the post-processing without a seat (no PF, no licence, runs anywhere):

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --self-test
```

Check the case resolution without a seat (imports PF nothing, prints the
generator-index resolution that has bitten before):

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --dry-run
```

The run itself:

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --label full_t0_wecc --save-trajectories
```

### Outputs — `results/timescale/<label>/<stamp>/`

| file | use |
|---|---|
| `timescale_table.tex` | the tabular **body** of Table 9.1 — paste over the `[TBD]` rows |
| `timescale_summary.md` | the same numbers, the derived design quantities, the provenance block |
| `timescale_summary.csv` | machine readable, one row per case |
| `cases.csv` | the resolved catalogue, written *before* the battery runs |
| `run_meta.json` | argv, git commit + dirty flag, interpreter, PF project, every constant entering a number |
| `run.log` | full console transcript |
| `traj_<case>.csv` | per-signal time series (`--save-trajectories`) |

`results/timescale/<label>/_latest.txt` names the newest stamp, so "which run
is Table 9.1 from" has one answer.

### Exit codes

| code | meaning |
|---|---|
| 0 | complete battery, every table row filled, nothing censored |
| 1 | no case succeeded; nothing written |
| 2 | ran, but the result is **not thesis-ready**: a case failed, a table row is unfilled, or a settling time is censored |

A censored row is one where the trajectory was still outside the band at the
last sample of the run: the reported `T_s` is then a lower bound set by the
horizon, not a measurement, and it is marked `>` in the `.tex` body. Re-run
those with a longer `--dispatch-horizon` / `--disturbance-horizon`.

### Filling the thesis

1. Replace the `[TBD]` rows of `tab:param:timescales:settling` with the
   contents of `timescale_table.tex`.
2. Fill the bracketed margin in *"The selected values are …"* from the
   **binding row** line of `timescale_summary.md`.
3. Delete the `\todo{Fill from …}` under the table and the `% NUMBERS: NONE
   YET` block in the section's comment header. Leave the `\todo` for
   `N_inner` — this script does not answer it.

Do **not** quote the older `results/screening/full_t0_wecc/20260720-*` run: it
predates the `EVENT_WINDOW_S = 60 s` finding of 2026-07-31, so its event
timing is not established.

### Known gaps, deliberately not fixed here

* **The machine-transformer tap is measured but has no row in Table 9.1.**
  The catalogue runs `tap_+1_MT_*` and the thesis text distinguishes a
  coupler tap from a machine-transformer one, but the table carries a single
  "OLTC tap, one step" row (the coupler). The MT row is listed under
  *"Measured but not in Table 9.1"* in the summary and it **does** enter the
  bound. Adding a table row is an author decision, not a script change.
* **No combined (multi-device) tap dispatch.** The battery moves one actuator
  at a time. The per-device rate limits do not prevent several changers
  tapping in the *same* iteration, and the cross-coupling penalty
  (`gw_oltc_cross_tso`) is disabled in the configuration used here, so an
  aggregate tap dispatch is realisable and is not measured. This is the
  `\todo` already standing in the chapter.
* **One operating point.** The periods are placed from one snapshot; envelope
  sensitivity is named as out of scope in the section's *Status of this
  choice* paragraph.

### Things that have bitten before

See `docs/handover_timescale_study.md` §5 — generator numbering offset
(`gen[1]` is `G 03`, `gen[7]` is `G 09`, `G 01` is the slack equivalent and is
refused), the 60 s PF event window, `purge_events` needing a calculation
reset, tripped elements losing their `m:` variables, and machine transformers
being resolved topologically rather than by name.
