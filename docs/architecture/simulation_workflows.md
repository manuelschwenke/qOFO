# Simulation workflows

How a run actually executes: the quasi-static simulation, the PowerFactory RMS
co-simulation, and the shared closed-loop runner both of them drive.

Written 2026-07-31 against the code as it stands. Structural line references are
to `experiments/runners/multi_tso_dso.py` (4,279 lines) and will drift; the
section names are stable.

---

## 1. The one function everything goes through

Every experiment in this repository is a thin wrapper around a single call:

```python
log = run_multi_tso_dso(config, plant_factory=None)
```

* `config` is a `MultiTSOConfig` — the complete experiment definition, which is
  also what gets serialised into each run's `config.json`.
* `plant_factory` decides **which plant the same controllers face**:
  * `None` → `core.plant.PandapowerStaticPlant`, the quasi-static (QSS) plant.
  * a `PowerFactoryReplayFactory` → `pf.plant.PowerFactoryPlant`, the RMS plant.
* the return value is a list of `MultiTSOIterationRecord`, one per simulation
  step. Everything the analyses read is derived from this list, pickled as
  `static_records.pkl` / `rms_records.pkl`.

The controller stack is identical in both cases. This is the point of the
design: the controllers cannot tell which plant they are driving, because they
only ever see their cached sensitivities and their own measurements.

### Which entry point to use

| Script | Static leg | RMS leg | Writes to | Use for |
|---|---|---|---|---|
| `experiments/run_multi_system_ofo.py` | yes | no | `results/multi_system_ofo/` | the authoritative hand-tuned configuration; pure QSS studies |
| `experiments/run_rms_cosim.py` | no | yes | `results/rms_cosim/` | exploratory RMS runs; roughly half the wall time |
| `experiments/run_comparison_rms_cosim_qss.py` | yes | yes | `results/rms_phase6_replay/` | the Gate-E comparison; **what the dead-band study uses** |
| `experiments/run_openloop_qss_to_rms.py` | records | replays | — | open-loop `u → y` plant equivalence |

Two traps worth stating explicitly:

* **`run_rms_cosim.py` stores `runner_static = None`.** Analyses whose admission
  filter reads the `runner_static` block will reject every run it produces. If a
  study collects from `results/rms_phase6_replay/`, it needs the *comparison*
  script.
* **The former `run_rms_phase6_replay.py` was a misnomer** — it never replayed
  anything. It is now a deprecation shim in `experiments/archived/` delegating to
  `run_comparison_rms_cosim_qss.py`. The genuine replay is
  `run_openloop_qss_to_rms.py`.

---

## 2. The quasi-static (QSS) simulation

The default path, used when `plant_factory is None`.

The plant is `PandapowerStaticPlant`: a pandapower network solved to an
algebraic power-flow equilibrium after every actuator write. There are no
electromechanical dynamics and no time constants. "Advancing time" means
nothing more than moving to the next dispatch instant and re-solving.

Per step:

1. Apply the exogenous profile sample for this instant (loads, DER active
   power).
2. Solve a power flow → this is the new plant state.
3. Read measurements off `net.res_*`.
4. Run whichever controllers are due this tick.
5. Write actuator commands into `net`.
6. Solve a power flow again so the recorded state reflects the commands.

Because the plant equilibrates instantly and exactly, the QSS run answers *what
operating point does the cascade converge to*, and says nothing about how it
gets there. It is fast — no PowerFactory, no licence, no RMS solver — so it is
the right tool for parameter sweeps, tuning, and anything needing many runs.

The `net` object is also the **measurement image** for the controllers, which
matters for the RMS path below.

---

## 3. The PowerFactory RMS co-simulation

`run_comparison_rms_cosim_qss.py` runs the cascade **twice**, once per plant,
and compares the two.

### 3.1 Step by step

1. **Build two independent configs** from `make_cosim_config(duration)` — one
   for the static leg, one for the RMS leg — and apply every CLI override to
   both, so the two legs differ *only* in their plant.
2. **Optionally raise the PowerFactory desktop** up front. `App.Show()` costs
   20–30 s and the desktop cannot paint while the engine is busy, so it is
   raised during the pure-pandapower static leg rather than next to the RMS
   build. Engine mode permits one session, so the handle is passed to the
   factory.
3. **Create the run directory** via `new_run_dir("rms_phase6_replay", spec)`,
   which stamps `config.json` with both configs. This is the provenance record
   every analysis admission filter later reads.
4. **Run the static leg**: `run_multi_tso_dso(static_cfg)` → `static_records.pkl`.
5. **Build the RMS plant** through `PowerFactoryReplayFactory`. The critical
   parameter is `event_pool_slots = duration/dt + 5`: PowerFactory admits only a
   couple of *mid-run* created events before event firing stops working
   altogether, so one slot per firing per target must be pre-created before
   `ComInc`. A default of one slot once froze every actuator at t ≈ 41 s.
6. **Run the RMS leg**: `run_multi_tso_dso(rms_cfg, plant_factory=factory)` →
   `rms_records.pkl`.
7. **Assert the plant reached the requested horizon** (`plant.t == duration`),
   then report event-pool usage.
8. **Bulk-export the RMS trajectories** through `ComRes` into
   `csv/rms_comres_full.csv`, then split into interface-Q/voltage monitors and
   per-park DER records.
9. **Compare**: endpoint errors, per-interval settling statistics, overlay
   figures, and actuator/voltage diagnostics.
10. **Write the verdict** into `gate_e_summary.json` / `.md` and exit non-zero if
    it fails, so a sweep's fail-fast catches it.

### 3.2 What the comparison does and does not establish

The two legs are **independent closed loops** — each plant feeds its own
controller stack. The comparison therefore measures *closed-loop* equivalence,
not open-loop plant equivalence. If the two disagree, that can be the plant
models differing **or** the loops diverging from a shared initial point. The
open-loop `u → y` test (`run_openloop_qss_to_rms.py`) is the one that isolates
the plant.

The verdict is `PASS` only if every interval settles **and** the DER Q(V)
actuator law is equivalent on both plants (`der_qv_local_control_equivalent`).
Without that flag the endpoint comparison is diagnostic only, because the RMS
parks would be holding `Qext` constant between dispatches while the static plant
applies the re-anchored characteristic.

### 3.3 What the RMS plant cannot do

For a non-static plant, `net` is only a **measurement mirror** — the real state
lives in PowerFactory. Anything that mutates `net` directly is therefore
rejected up front rather than silently producing an inconsistent
co-simulation:

* contingency events
* the local OLTC tap-rate limiter
* the zonal generator-dispatch schedule

Time-series profiles *were* on that list and came off on 2026-07-21, once
`Plant.apply_exogenous` existed to deliver them natively (`EvtLod` for loads,
`EvtParam` on the WECC `Pref_in` for DER active power). That is the pattern any
future feature must follow: reach the plant through the `Plant` interface, not
through `net`.

The **exogenous load step** used by the disturbance-rejection study is
deliberately built this way. It perturbs the interpolated profile frame, so both
plants receive it through paths they already support, and it is *not* a
contingency — no element is switched.

---

## 4. Inside `run_multi_tso_dso`

One function, executed in three parts: build, initialise, loop. Section names
below match the comments in the source.

### 4.1 Build (steps 1–8)

| Step | What happens |
|---|---|
| 1 | Build the base IEEE 39-bus network for the configured scenario |
| 2 | Zone partitioning — relabel zones by generator count |
| 3 | Attach the HV sub-networks (110 kV underlays) |
| 3 (cont.) | **Apply per-DSO scenario multipliers** — before any power flow, load model or droop tagging, because they rewrite `p_mw`, `base_p_mw`, `sn_mva` and the reactive-load base that all of those read |
| 3b | Plant load model: constant-PQ vs anchored ZIP |
| 4 | Tag every DER with its `q_mode` |
| 5 | Build `ZoneDefinition`s and `TSOControllerConfig`s: partition generators, DER, OLTCs and tertiary shunts per zone; propagate the voltage schedule |
| 6 | Instantiate one `TSOController` per zone, including the `G_w` diagonal and any SBX-H / SBX-V coordination adapters |
| 7 | Instantiate one DSO controller per HV sub-network |
| 8 | Build the `MultiTSOCoordinator`; optionally build the single centralised controller instead (`control_scope='central'`) |

### 4.2 Initialise (steps 9–12)

| Step | What happens |
|---|---|
| 9 | Load the annual profiles, interpolate to `dt_s`, **apply the load step if configured**, clip to the simulation window, compute the zonal generator dispatch |
| 10 | Combined operating-point initialisation in three phases, ending with the plant-side DER `Q(V)` loops converged **before** the plant seam |
| 10 (cont.) | **Plant seam** — from here every actuator write, plant response and measurement refresh goes through the `Plant` interface. The static default reproduces the legacy behaviour bit-for-bit; a substituted RMS plant keeps `net` as its mirror |
| 11 | Build the cached sensitivities: shared or per-controller Jacobians, optional finite-difference `H`, optional curvature-based `g_w` preconditioning; send the initial DSO capability messages upward |
| 12 | Q-tracking capacity diagnostic |

The ordering in step 10 is load-bearing. The `Q(V)` fixed point is solved at the
seam so that both plants start from the same electrical state; the cached
sensitivities in step 11 are linearised about that point.

### 4.3 The main loop (step 13)

Once per `dt_s`, in this order:

1. **Profiles** — apply this instant's exogenous sample (and the load step, if
   past its instant).
2. **Contingencies** — static plant only.
3. **TSO tick** (every `tso_period_s`): optional SBX-H horizontal round and
   SBX-V vertical round *before* the zones solve, then each zone's MIQP,
   producing AVR setpoints, OLTC moves, shunt commands and the `Q_PCC` setpoints
   sent downward. Switched-shunt integrator dispatch runs here in integrator
   mode.
4. **DSO tick** (every `dso_period_s`): each DSO controller tracks its `Q_PCC`
   setpoint with local actuators and reports capability and tracking error back
   up.
5. **End-of-step power flow** — skipped when nothing wrote actuator commands and
   no contingency fired, since the post-profile solution already reflects the
   final state.
6. **Record** everything the analyses need: interface Q set/actual, per-zone
   voltage errors, DSO group voltages, actuator positions, losses, slack
   saturation, and the live-plot observables.

The timescale separation is configuration, not structure: `tso_period_s = 180`,
`dso_period_s = 20`, `dt_s = 20` gives the intended ~9:1 ratio. Setting
`dso_period_s` below `dt_s` does not make the DSO faster — it makes it fire
every plant step, which is how a whole tuning campaign was once run with no
timescale separation at all.

### 4.4 Final summary (step 14)

Prints per-zone voltage means and DSO Q-tracking quality — the
`mean|err|` figures that the dead-band study's interface-Q metric aggregates.

---

## 5. Reading a completed run

```
results/<experiment>/<NNNN>_<YYYY-MM-DD_HHMMSS>/
├── config.json              provenance: runner_static / runner_rms blocks
├── static_records.pkl       QSS leg  (comparison entry point only)
├── rms_records.pkl          RMS leg  → what most analyses read
├── gate_e_summary.{json,md} verdict
├── csv/                     trajectories, settling, endpoint comparison
├── figures/
└── snapshot/                PowerFactory model snapshot
```

`config.json → runner_static` is the block admission filters read. Two runs are
comparable only if it matches on scenario, DER capability model, profile use,
droop slope and dead band, seeding options, per-DSO scenario multipliers, and
(for undisturbed studies) the absence of a load step. `analysis/deadband_selection.py`
carries the worked example of such a filter, including why each key is in it.

Run numbering is global and increments even for aborted runs, so gaps are
normal. A run without `rms_records.pkl` was aborted and is skipped silently.
