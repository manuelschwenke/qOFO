# 2026-07-30 — PowerFactory re-sync: DSO conductors, DSO_3 parallel circuit, zero constant Q

**What:** Propagated the three persistent DSO plant changes of
`2026-07-30_ieee39_dso_powerfactory_sync_handover.md` into `qOFO\IEEE39_qOFO`,
and moved the reference snapshots from the (deprecated) `wind_replace` alias to
the scenario the study actually runs.
**Why:** PowerFactory is a derived artefact of the pandapower builder, and the
RMS replays must not run against 184-AL1 conductors and a constant-Q DSO load
model that no longer exist on the pandapower side.
**Timestamp:** 2026-07-30, ~13:45–14:10.
**Status:** applied and verified. Gate C green at both operating points; project
left synced to `full_t0` on `rural_700`.

---

## 1. What was pushed

| change | PF objects | count |
|---|---|---|
| conductor type 184-AL1/30-ST1A → 305-AL1/39-ST1A (DSO 1/2/4) | `TypLne.rline/xline/cline/bline/sline` | 33 lines |
| conductor type 184-AL1/30-ST1A → 490-AL1/64-ST1A (DSO_3) | same | 11 lines |
| DSO_3 corridor HV 5–6 doubled | `ElmLne.nlnum` 1 → 2 (`DSO_3_line78`, 15.6 km) | 1 |
| zero constant reactive load | `ElmLod.qlini` 5.386 → 0.0 on every `*_const` row | 40 |
| profile-Q base 500 Mvar aggregate | `ElmLod.qlini` on every `*_var` row | 40 |
| installed DER 410 → 700 MW per DSO (`rural_700`) | `ElmGenstat.sgn/pgini` | 40 |

Old → new conductor data, as PF held it against what the tree builds:

| | r [Ω/km] | x [Ω/km] | c [µF/km] | i_max [kA] |
|---|---:|---:|---:|---:|
| PF before (all DSOs) | 0.1571 | 0.400 | 0.0088 | 0.535 |
| 305-AL1/39-ST1A (DSO 1/2/4) | 0.0949 | 0.380 | 0.0092 | 0.740 |
| 490-AL1/64-ST1A (DSO_3) | 0.0590 | 0.370 | 0.00975 | 0.960 |

**Not pushed, deliberately:** the DSO_3 ×2 DER / active-load multiplier. Per the
handover it is an experiment-only knob (`_apply_dso_overrides` in
`analysis/annual_dso_pq_characterization.py`, CLI `--dso-der-scale DSO_3=2
--dso-load-p-scale DSO_3=2`), not builder state, so it never reaches an RMS
replay and must not be baked into the shared model.

## 2. Pre-check that made the type write safe

`pf_sync.sync_lines` writes `rline/xline/cline/bline/sline` onto `obj.typ_id`
— the `TypLne` the line already points at. Had several `ElmLne` shared one
`TypLne`, the last write would have won and DSO_3's 490 mm² data would have
silently overwritten the 305 mm² lines (or vice versa), with no diff and no
error. A read-only audit first confirmed **every line owns its own TypLne**
(`TYP_DSO_1_line37(1)`, …), so the per-line writes cannot alias.

## 3. Code changes

* `export/make_snapshots.py` — new `--scenario` option, and
  `build_snapshot_state(..., scenario=...)`. The non-base phases previously
  hardcoded `scenario = "wind_replace"`, which now resolves through the
  deprecation alias to `base_410`. With installed capacity a scenario choice,
  that silently produced a 410 MW reference snapshot while the study
  (`MultiTSOConfig.scenario`, now `"rural_700"`) runs at 700 MW — i.e. the PF
  checkpoint would have been parity-validated against a model the RMS runs do
  not use. The default is now `MultiTSOConfig().scenario`, so the reference
  snapshot and the study it validates cannot drift apart unnoticed.
* `experiments/run_rms_openloop_uy.py` — new `--scenario` option, mirroring
  `run_rms_phase6_replay.py` (which already warns that results from different
  scenarios are not comparable). Without it the open-loop test silently took
  the config default.

## 4. Applied sequence

Recovery point first:
`\mschwenke.IntUser\qOFO\IEEE39_qOFO.IntPrj\Versions.IntVersionman\pre_dso_lines_qload_resync_20260730-1344.IntVersion`
(created through `IntPrj.CreateVersion`).

```
python -m export.make_snapshots --phase full --auto t0 --verify
python pf/pf_sync.py export/snapshots/full_t0_20160105-0800.json --phase full --dry-run
python pf/pf_sync.py export/snapshots/full_t0_20160105-0800.json --phase full
python pf/pf_parity.py export/snapshots/full_t0_20160105-0800.json --interfaces
python -m export.make_snapshots --phase full --auto peakres --verify
python pf/pf_sync.py export/snapshots/full_peakres_20160413-0900.json --phase full
python pf/pf_parity.py export/snapshots/full_peakres_20160413-0900.json --interfaces
python pf/pf_sync.py export/snapshots/full_t0_20160105-0800.json --phase full
python pf/pf_parity.py export/snapshots/full_t0_20160105-0800.json --interfaces
```

`full_t0` sync: **0 created, 0 renamed, 319 updated, 0 deleted**; a second run
reports 0/0/0/0 — idempotent. Nothing was created, deleted or re-connected, so
`wecc_apply` was **not** needed this time (unlike 2026-07-29, where deleted
parks took their RMS composites with them). Confirmed afterwards: 44
`ElmGenstat` / 44 `ElmStactrl` / 44 `WECC_*` `ElmComp`, no orphans in either
direction.

`peakres` sync: 165 updates, all operating point, no connection change —
confirming again that the topology/type fix is shared between snapshots rather
than re-applied per snapshot. Peak residual load resolved to 13.04.2016 09:00
under `rural_700` (the `base_410` series used the same timestamp).

## 5. Verification

Gate C parity (`iopt_lim = 0`, load voltage dependency ON, no automatic
taps/shunts):

| | full_t0 | full_peakres | 2026-07-29 (pre-change) |
|---|---|---|---|
| max \|Δvm\| | **1.595e-5 pu** | **1.764e-5 pu** | 1.548e-5 / 1.620e-5 |
| max \|Δva\| | **5.931e-4 deg** | **5.037e-4 deg** | 4.013e-4 / 5.302e-4 |
| worst 3W interface P/Q | **1.84e-3 MW** | — | 2.372e-3 / 2.445e-3 |
| verdict | **PASS** | **PASS** | — |

All families remain at the known float32 impedance-storage floor, so the two
models moved together across a conductor change, a parallel circuit and a load
model change. Snapshot round-trips: OK, all deltas 0.000e+00.

Structural spot check after the sync: `DSO_3_line78` reads `nlnum = 2`,
`dline = 15.6 km`, `rline = 0.059` — the parallel circuit and the reinforced
conductor both arrived.

## 6. Risks / unresolved

- **Every DSO impedance moved again.** `results/tuned_params_t0min.json`, all
  cached sensitivities and every archived RMS/quasi-static result predate this.
  Nothing produced before 2026-07-30 14:00 is comparable with anything after.
  This is the second such invalidation in two days (the 2026-07-29 geometry
  re-sync was the first).
- **Signed DSO load Q is now the model.** Aggregate DSO Q swings −69.98 …
  +172.13 Mvar with mean −24.94 Mvar, i.e. the DSOs are capacitive on average
  and there is no constant Q floor. Handover risk 1 (is the capacitive period
  intended in PF?) is pushed as-is and still wants an explicit confirmation.
- **`op_diagram` divergence, still latent.** Unchanged from 2026-07-29:
  `pf_sync` writes `q_min/q_max = ±1.0 pu` on every `ElmGenstat` regardless of
  the snapshot's `op_diagram`, so PF's DER Q capability is a full circle while
  pandapower's is the VDE box. Parity is unaffected (`iopt_lim = 0`), but the
  RMS path binds limits — there it is `pf/plant.py::_anchor_qv_precontrollers`
  that mirrors the box onto QVPRE, so the `ElmGenstat` limits are not what acts.
  Still two models disagreeing on paper.
- **Gate D is now two model generations stale** (pre-geometry, pre-conductor).
- The parity gate runs `iopt_lim = 0`; it cannot see a limit-binding
  disagreement. It also does not exercise the parallel circuit's ampacity — only
  its impedance shows up in the flows.

## 7. Files

- `export/make_snapshots.py` — `--scenario`, `DEFAULT_DSO_DER_SCENARIO`,
  `build_snapshot_state(scenario=...)`.
- `experiments/run_rms_openloop_uy.py` — `--scenario`.
- `export/snapshots/full_t0_20160105-0800.json`,
  `export/snapshots/full_peakres_20160413-0900.json` — regenerated on
  `rural_700`.
- Scratchpad probes (not committed): `probe_pf_line_types.py` (read-only
  TypLne-sharing audit), `pf_recovery_point.py` (IntVersion helper).
