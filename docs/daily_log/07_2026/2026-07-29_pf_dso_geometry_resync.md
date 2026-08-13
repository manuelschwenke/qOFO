# 2026-07-29 — PowerFactory re-sync after the DSO↔TS coupling-geometry change

**What:** Propagated the `SUBNET_DEFS` calibration (per-DSO `scale`, `hv_buses`
reorder) into the PowerFactory model `qOFO\IEEE39_qOFO`, plus two `pf_sync`
defect fixes that the propagation exposed.
**Why:** Follow-up to `2026-07-29_dso_ts_coupling_geometry.md`. PF is a derived
artefact of the pandapower builder (build plan, guiding principle), so the
constants change had to be pushed and the Gate-C parity re-established.
**Timestamp:** 2026-07-29
**Status:** applied and verified. Gate C green at both operating points.

---

## 1. What the change looks like on the PF side

Regenerating the two `full` snapshots (`export/make_snapshots.py --phase full`,
round-trip verified) isolates the model delta to four things:

| delta | count | PF objects |
|---|---|---|
| DSO HV line lengths | 44 | `ElmLne.dline` |
| coupler MV-side bus | 8 | `ElmTr3.busmv` |
| coupling-bus STATCOM parks renamed + moved | 8 | `ElmGenstat` (+ `ElmStactrl`, `ElmComp`) |
| coupler OLTC tap positions from the init pass | 3 (t0) / 2 (peakres) | `ElmTr3.n3tap_h` |

`load`, `gen`, `shunt` and every TN element are untouched — the DSO geometry is
the only thing that moved. Resulting route lengths in PF:

| DSO | scale | route | circuit range |
|---|---|---|---|
| DSO_1 | 0.82 | 196.8 km | 8.2 – 32.8 km |
| DSO_2 | 1.40 | 336.0 km | 14.0 – 56.0 km |
| DSO_3 | 0.52 | 124.8 km | 5.2 – 20.8 km |
| DSO_4 | 2.44 | 585.6 km | 24.4 – 97.6 km |

The `hv_buses` reorder keeps the coupling-bus *set*, so couplers 0 and 1 of every
DSO swap onto each other's 110 kV bus — `mv_bus` 46↔43, 66↔63, 86↔83, 106↔103.

## 2. Two defects in `pf_sync` that this exposed

### 2.1 Connections were never re-pointed (silent divergence)

Cubicles are created exactly once, when the element is created. For element
classes whose `loc_name` does **not** encode their bus — `ElmLne`
(`DSO_1_line37`), `ElmTr3` (`NC3W_DSO_1_t0`), `ElmShnt` (`SH_MSC_DSO_1_s0`) —
find-or-create by name matched, every pushed attribute compared equal, and the
sync reported *unchanged* while the object stayed wired to its old terminal.
The eight `busmv` moves above would have gone through silently.

This is the dangerous shape of bug for a derived model: no error, no diff, and
the parity gate is the only thing that would have caught it (it would have, but
as an unexplained failure, not as a located one).

Fix: `_reconnect(ctx, obj, attr, bus_idx, tag, label)` re-points one connection
attribute at the snapshot's terminal, reports the move, and deletes the vacated
`StaCubic` when PF has released it. Wired into `sync_lines` (`bus1`/`bus2`),
`sync_trafo3w` (`bushv`/`busmv`/`buslv`) and `sync_shunts` (`bus1`).

`ElmTr2` is deliberately **checked, not re-pointed**: template adoption matches
on the unordered endpoint pair, so an adopted unit may carry the HV/LV
orientation reversed with respect to pandapower and a blind reconnect would move
the tap changer to the other winding. An unordered endpoint comparison raises on
a genuine bus reassignment instead.

Loads and static generators need none of this — their names carry the bus, so a
bus move renames them and the existing create/delete path handles it.

### 2.2 A deleted park left its dependants behind

`delete_stale_sgens` removed the renamed `ElmGenstat` but not the objects named
after it. The audit found **12 orphan `ElmStactrl`** — `CTRL_DER_DSO_*`, from the
2026-07-21 `WPC_* → DER_*` role reclassification, i.e. **pre-existing, not caused
by today's change** (their bus suffixes are the pre-reorder ones). All twelve were
in service, `i_ctrl = 1`, with non-zero `qsetp`, and a `psym` machine list
containing nothing but a dead handle.

They had no effect on the load flow — a station controller with no machine has
nothing to actuate, and removing them left the parity numbers bit-identical
(1.548e-5 pu / 4.013e-4 deg before and after). They were still live objects
referencing deleted ones inside a validated model.

Two fixes:

- `delete_stale_sgens` now takes the park's `ElmStactrl` and its
  `WECC_<park>` `ElmComp` with it.
- `delete_orphan_station_controllers` sweeps controllers that satisfy **both**
  (i) the name is `CTRL_<park>` for a park absent from this snapshot and
  (ii) no machine in `psym` still exists. Both conditions are required so the
  sweep cannot misfire in the `wind_replace` phase, where the snapshot names
  only the TSO parks while the DSO controllers and their parks are all alive.
  `_psym_has_live_member` fails safe: anything unreadable counts as live.

## 3. Applied sequence

Recovery point first:
`\mschwenke.IntUser\qOFO\IEEE39_qOFO.IntPrj\Versions.IntVersionman\pre_dso_geometry_resync_20260729-133346.IntVersion`.

```
python -m export.make_snapshots --phase full --auto t0,peakres --verify
python pf/pf_sync.py export/snapshots/full_t0_20160105-0800.json --phase full --dry-run
python pf/pf_sync.py export/snapshots/full_t0_20160105-0800.json --phase full
python pf/wecc_apply.py --only WPC_
python pf/pf_parity.py export/snapshots/full_t0_20160105-0800.json --interfaces
python pf/pf_sync.py export/snapshots/full_peakres_20160413-0900.json --phase full
python pf/pf_parity.py export/snapshots/full_peakres_20160413-0900.json --interfaces
```

First `full_t0` sync: created 32 (8 parks + 8 cubicles + 8 controllers + 8 new
3W MV cubicles), updated 246, deleted 32 (8 orphan parks with their 8
controllers and 8 WECC composites, plus 8 vacated cubicles). A second run
reports 0/0/0/0 — idempotent. The subsequent orphan sweep deleted the 12
pre-existing controllers; the run after that is clean again.

The `peakres` sync creates nothing and changes no connection (171 attribute
updates, all operating point) — confirming the topology fix is shared between
the two snapshots rather than re-applied per snapshot.

**`wecc_apply --only WPC_`** was needed because the eight deleted orphan parks
took their RMS composites with them. All 12 coupling parks were rebuilt and
re-anchored; the RMS layer is complete again (44 parks / 44 composites).

## 4. Verification

**Gate C parity** (`iopt_lim = 0`, load voltage dependency ON, no automatic
taps/shunts):

| | full_t0 | full_peakres | Gate C, 2026-07-19 |
|---|---|---|---|
| max \|Δvm\| | **1.548e-5 pu** | **1.620e-5 pu** | 1.545e-5 / 1.609e-5 |
| max \|Δva\| | **4.013e-4 deg** | **5.302e-4 deg** | 4.059e-4 / 5.377e-4 |
| worst 3W interface P/Q | **2.372e-3 MW** | **2.445e-3 MW** | 2.673e-3 |
| verdict | **PASS** | **PASS** | — |

All three families are at or slightly inside the pre-change Gate C figures, so
the geometry change costs no parity: the two models moved together. The residual
is the known float32 impedance-storage floor, not a modelling difference.

**Structural audit** (`scratchpad/pf_verify_model.py`, one-off): all 12 `ElmTr3`
windings on the snapshot's buses; 109 line lengths within 1.2e-5 km; 44 parks /
44 controllers / 44 WECC composites with no orphans of any class; every park
terminal matching the snapshot. The 130 empty `StaCubic` are the DIgSILENT
template's own spare cubicles (`Cub_1`…`Cub_4` on TN buses) and are untouched.

**Unit tests**: `tests/pf/test_pf_sync_reconnect.py` (new, 9 cases) covers the
move, the no-op, the still-occupied cubicle, the dry run, the unconnected
attribute, and the four `_psym_has_live_member` branches. `tests/pf/` +
`tests/export/`: 53 passed, 2 skipped, 1 failed —
`test_screening_event_pool.py::test_persistent_pool_grows_admits_and_retires_events`
fails on an `admit_new_events(..., horizon_s)` signature mismatch in
`pf/screening.py`, which this change does not touch. **Pre-existing, unrelated,
still open.**

Project left synced to `full_t0`, parity re-confirmed on that exact state.

## 5. Risks / unresolved

- **Re-tuning is still outstanding** — unchanged from the pandapower log. Every
  TN–DS impedance moved, so `results/tuned_params_t0min.json`, all cached
  sensitivities and every archived RMS/quasi-static result predate this. No
  result produced before 2026-07-29 is comparable with one produced after.
- **All RMS evidence predates the change.** Gate D (20 s timescale separation,
  2026-07-20) and every Phase-6 replay ran on the `scale = 1.00`, `(3,0,8)`
  geometry. DSO_4's route length nearly tripled (196.8 → 585.6 km) and DSO_3's
  halved, which shifts the DSO electrical distances and therefore the local
  Q(V)/OLTC interaction the screening measured. Gate D should be re-run before
  its verdict is cited against post-change results.
- **`op_diagram` divergence, latent.** The regenerated snapshots carry
  `op_diagram = 'VDE-AR-N-4120-v2'` on 16 sgens where the 2026-07-19 snapshots
  said `'STATCOM'` (working-tree drift from `core/actuator_bounds.py`, not from
  this change). `pf_sync` writes `q_min/q_max = ±1.0 pu` on every `ElmGenstat`
  regardless, so the PF Q capability is a full circle. Parity is unaffected —
  the parity load flow runs `iopt_lim = 0` with constant-Q controllers, so no
  limit binds — but the two models do not agree on DER capability, and anything
  that lets a limit bind (RMS, or a load flow with `iopt_lim = 1`) would diverge.
  Not fixed here.
- The 12 orphan controllers were deleted on the argument that a controller with
  no machine cannot actuate, evidenced by the bit-identical parity. If a PF
  release ever gave such a controller a default action, the pre-2026-07-29 RMS
  results would have been affected by them; nothing in the archived runs
  suggests it.
- Dry-run fidelity: `--dry-run` reported only 4 of the 8 orphan parks. Newly
  created parks are not instantiated in a dry run, so they cannot claim their
  terminal, and the "shares a terminal with a claimed park" rule under-reports.
  The real run found all 8. Cosmetic, but the dry run is not a complete preview
  of deletions.

## 6. Files

- `pf/pf_sync.py` — `_reconnect`, `_endpoint_term_names`,
  `delete_orphan_station_controllers`, `_psym_has_live_member`,
  `RMS_COMPOSITE_PREFIX`; `delete_stale_sgens` extended; reconnection wired into
  `sync_lines` / `sync_trafo3w` / `sync_shunts`; endpoint check in `sync_trafos`.
- `tests/pf/test_pf_sync_reconnect.py` — new.
- `export/snapshots/full_t0_20160105-0800.json`,
  `export/snapshots/full_peakres_20160413-0900.json` — regenerated.
