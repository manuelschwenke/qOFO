# 2026-07-17 — RMS co-simulation Phase 0: dynamic snapshot exporter (Gate 0 green)

**Context.** Start of the PowerFactory RMS co-simulation build
(docs/RMS_IEEE39_PowerFactory_Build_Plan.md). Phase 0 freezes the scenario and
creates the hand-over artefact between the pandapower builder (single source of
truth) and the PF sync script. Decisions taken today: G10 (Hydro) retained
(code already consistent — only stale comments cleaned); static reference
snapshots are still required (RMS initial condition + parity oracle) but chosen
automatically (`t0`, `peakres`) rather than curated.

## New files

- **`export/dynamic_snapshot.py`** — `dump_dynamic_snapshot(net, meta,
  zone_map, label, out_dir, *, solver_options, …)` serialises `model` (every
  element keyed by pandapower index; explicit per-table field lists),
  `solution` (converged res_* tables = parity target), `meta`, `zone_map`,
  `removed_generators`, `actuators`, `solver_options`, provenance.
  `load_snapshot_to_pandapower()` rebuilds a runnable net from the JSON alone;
  `verify_roundtrip()` reruns the stored solver options and compares.
  **Fail-Fast completeness contract**: unknown columns raise
  `SnapshotSchemaError`; required-field NaN raises; referential integrity
  checked; must-be-empty tables (ext_grid, switch, ward, …) enforced;
  controllers must be dropped before dumping.
- **`export/make_snapshots.py`** — CLI driver mirroring the plant-relevant
  runner steps ([1]–[4], [9], [10.1]–[10.2] of
  `experiments/runners/multi_tso_dso.py`): build → fixed 3-zone partition →
  `add_hv_networks` (full phase) → `tag_der_q_modes` (MultiTSOConfig
  defaults) → profiles at the snapshot instant → zonal gen dispatch →
  STATCOM-Q/OLTC init (DiscreteTapControl phases 1+2, then controllers
  dropped) → final plain `runpp` with the canonical solver options
  (`distributed_slack=True, enforce_q_lims=True, init='auto',
  calculate_voltage_angles=True, max_iteration=50`). Phases: `base` /
  `wind_replace` / `full`; timestamps: `t0` = 05.01.2016 08:00 (experiment
  default) and `peakres` = full-year argmax of Σ P_load − Σ P_sgen.
- **`tests/export/test_dynamic_snapshot_roundtrip.py`** — Gate 0: per phase,
  dump → rebuild → rerun → solution must match (vm/va tol 1e-8, flows 1e-6);
  element-count, meta, zone-map, removed-gen and tamper (Fail-Fast) checks.
  Result: **16 passed, 2 skipped**; deviations are exactly **0.0**
  (bit-identical rebuild) on all three phases.
- **`docs/RMS_IEEE39_PowerFactory_Build_Plan.md`** — the working plan, plus a
  "Repo reality check" appendix (see findings below).

## Changed files

- **`network/ieee39/helpers.py` — `remove_generators()`**: now also drops
  loads stranded on removed 10.5 kV terminal buses and raises on any other
  dangling reference. *Discovery*: the case39 load at IEEE bus 31 (0-idx 30,
  split into const 3.68 MW + profile half) sat on G2's terminal bus; after
  wind_replace it referenced a non-existent bus and pandapower silently
  excluded it (res_load = 0), so `net.load` overstated served demand by
  ≈ 6.5 MW. The fix is provably behaviour-neutral for the power flow (those
  rows already contributed nothing; zone dispatch never counted bus 30) —
  confirmed by 38 pre-existing builder-dependent tests passing unchanged.
  **Open point for discussion**: physically, should that local demand instead
  *move to the grid bus* when the plant is replaced? That would change every
  wind_replace operating point slightly — not done today.
- **`network/ieee39/scenarios/wind_replace.py`**: stale "+ 37 maybe?"
  comments removed; G10-retained decision recorded (code unchanged).

## Reference snapshots generated (`export/snapshots/`, all round-trip OK)

| file | phase | note |
|---|---|---|
| base_t0_20160105-0800.json | base | Gate A oracle |
| base_peakres_20160413-1100.json | base | Gate A stress |
| wind_replace_t0_20160105-0800.json | wind_replace | Gate B oracle |
| full_t0_20160105-0800.json | full | Gate C oracle + RMS initial condition |
| full_peakres_20160413-0900.json | full | Gate C stress |

**Finding — wind_replace @ peakres does not converge** (NR, also with dc init
and 100 iterations): at 13.04.2016 11:00 the intermediate model (4 machines
removed, full TN load still attached, no HV underlays) needs ≈ 1780 MW
cross-zone spill and exceeds the feasible envelope with `enforce_q_lims`.
Acceptable: the intermediate is a translation-debug artefact; Gate B uses t0.
The full model converges at its own peak (13.04.2016 09:00).

## pandapower 3.4.0 schema facts captured (server env replication)

- Load ZIP shares are split per axis: `const_z_p/q_percent`,
  `const_i_p/q_percent` (all 0.0 here — constant-power oracle).
- `sgen`/`gen` carry capability-curve columns
  (`reactive_capability_curve`, `curve_style`,
  `id_q_capability_characteristic`) — asserted unused.
- `trafo3w` carries zero-sequence columns (`vk0_*`, `vkr0_*`,
  `vector_group`) — asserted unused (symmetric RMS/load flow only).

## Environment note (server)

`qOFO_clean` did not exist on this machine; replicated at
**`F:\python_environments\qOFO_clean`** (conda-registered, Python 3.12.13)
via `pip install --no-deps -r requirements/requirements-lock.txt`
(the lock is globally inconsistent for pip's resolver — numpy 2.4.6 vs
declared ceilings of numba/cvxpy/… — so `--no-deps` is the correct exact
replication; all 109 pins installed, pandapower 3.4.0 / numpy 2.4.6 /
pandas 2.3.3 verified). The CLAUDE.md interpreter path refers to the
workstation, not this server.

## Next (Phase 1)

PF template preparation on the licensed machine (import "39 Bus New England
System", reference machine G 01, G 05 rating check, PSS decision, constant-P
load model) + `pf/session.py` API scaffolding + `docs/pf_naming.md`.
