# Building the RMS IEEE 39 Model in PowerFactory

## A step-by-step working plan with Claude Code prompts

**Goal.** Obtain a validated RMS (phasor) model of the modified IEEE 39-bus system (wind_replace scenario, 3 TS zones, 4 × 110 kV STS underlays) in DIgSILENT PowerFactory, derived automatically from the qOFO_GH pandapower model, ready for OFO-in-the-loop co-simulation to verify the timescale-separation assumption at the STS dispatch cadence (**20 s** — confirmed 2026-07-20; the original 10 s figure throughout earlier drafts is superseded, matching `MultiTSOConfig.dso_period_s = 20`).

**Definition of done.**

- [x] PF load flow of the full model matches pandapower at both full snapshots (Gate C, 2026-07-19): max voltage error 1.609e-5 pu, max angle error 5.377e-4 deg, and every one of the 12 coupler HV/MV P/Q interface quantities within 2.673e-3 MW/Mvar.
- [x] A 60 s no-disturbance RMS run stays flat (2026-07-20: drift 6.8e-12 pu — genuine equilibrium).
- [x] Modal analysis at the operating point produces a documented list of all modes with settling time > ~2 s (2026-07-20: 837 modes with WECC DER dynamics, 0 unstable, slow set = electromechanical band 0.78–1.34 Hz, ζ ≈ 0.04).
- [x] A scripted co-simulation loop (`PowerFactoryPlant`) dispatches OFO setpoints every 20 s and harvests full trajectories per dispatch interval (2026-07-20: `pf/plant.py`, smoke 5/5; closed-loop replay outstanding — Gate E).

**Guiding principle.** The pandapower builder remains the **single source of truth**. PowerFactory is a _derived artefact_, generated and updated by a sync script. Never fix anything manually in the PF GUI that the script should own — every manual edit is a divergence the parity gates will have to catch later.

**Working with Claude Code.** Claude Code cannot execute the PowerFactory Python API (PF runs only on the licensed Windows machine). The workflow is therefore: Claude Code writes/refactors the code in the repo → run it on the PF machine → paste the error output / parity report back into Claude Code → iterate.

---

## Pipeline overview

```
qOFO_GH (pandapower)                      PowerFactory (Windows + licence)
─────────────────────                     ────────────────────────────────
build_ieee39_net()                          39-bus template project (DIgSILENT)
  └ apply_wind_replace()                      └ Variation "wind_replace"
      └ add_hv_networks()                         └ Grids DSO_1 … DSO_4
          └ profiles @ snapshot t
              └ runpp  ──► snapshot.json ──► pf_sync.py ──► PF model
                              │                                │
                              └────────── parity gates ◄───────┘
                                               │
                                   PowerFactoryPlant co-sim loop
```

---

## Phase 0 — Freeze the scenario and build the snapshot exporter

_Everything downstream replicates whatever the code does — so the code must be unambiguous first._

### Steps

1. **Resolve the wind_replace discrepancy.** ✅ *Resolved 2026-07-17: G10 (Hydro, term 29) is retained.* `_z1_gens_to_remove_term = {36}` (G8 only) — code and docstring agree. The dynamic model therefore contains a hydro machine (G10, 1 GVA) in Zone 1.

2. **Reference snapshots.** Decision 2026-07-17: static reference snapshots **are** still required — (i) every RMS run initialises from a converged load flow, (ii) parity Gates A–C need a frozen oracle, (iii) modal analysis is evaluated at an operating point. But no curated battery is needed up front; the exporter is parameterised by timestamp and more snapshots can be dumped at any time. Default reference set (chosen automatically by `export/make_snapshots.py`):
    - `t0` — 05.01.2016 08:00 (default experiment start, `make_config()`).
    - `peakres` — full-year maximum of the system residual load Σ P_load − Σ P_sgen (stress point for the synchronous fleet).
    The 35 136-step profile machinery stays in pandapower; at co-simulation time the P/Q references are fed from the pandapower ground truth per dispatch step, so the PF model remains "time-series" without ever owning the profiles.

3. **`dump_dynamic_snapshot(net, meta, zone_map, label, out_dir, ...)`** ✅ implemented in `export/dynamic_snapshot.py`; JSON schema with `model` / `solution` / `meta` / `zone_map` / `removed_generators` / `actuators` / `solver_options` sections, every element keyed by pandapower index, Fail-Fast completeness contract (unknown columns raise).

4. **Round-trip validation** ✅ `load_snapshot_to_pandapower()` + `verify_roundtrip()` + `tests/export/test_dynamic_snapshot_roundtrip.py` (all three phases: base, wind_replace, full).

### Gate 0

- [x] Discrepancy resolved and committed.
- [x] Round-trip test green for all chosen snapshots (base, wind_replace, and regenerated full t0/peak-residual oracles).

---

## Phase 1 — Prepare the PowerFactory template (one-off, mostly GUI)

### Steps

1. **Import** the DIgSILENT "39 Bus New England System" project; make a copy `IEEE39_qOFO`. Run the predefined load-flow study case and verify against Appendix B / Table 10 of the PF documentation PDF — confirms a pristine starting point.

2. **Template fixes** (from the documentation review):
    - Reference machine → **G 01** (the 10 GVA equivalent; the slack anchor).
    - **Verify G 05 rating**: the PDF prints Sr = 300 MVA, but H = 4.333 s on machine base only reconciles with H = 26.0 s (100 MVA base) if Sr = 600 MVA. Check the actual object; correct to 600 if needed.
    - **PSS decision**: the template ships with PSSs _disabled_. Record the decision: start disabled, revisit in Phase 5 when the modal screen shows the actual damping.
    - **Load model** — superseded 2026-07-17: the pandapower oracle itself now carries the anchored ZIP model (kpu = 1, kqu = 2 at 1.03 pu; `network/ieee39/load_model.py`), so PF keeps voltage-dependent loads and the LDF flag "consider voltage dependence of loads" is ON with exponents (1, 2) and u0 = 1.0 pu (see docs/pf_api_notes.md §4).

3. **Naming convention.** Every scripted object embeds the pandapower index in `loc_name`: `TN_bus15`, `MT_g4` (machine trafo), `WP_TSO_bus35`, `DSO_2_bus07`, `DSO_2_line03`, `NC3W_DSO_1_b6` (3-winding coupler), `DER_DSO_3_wp1`, … Write the convention down in `docs/pf_naming.md`; the sync script's find-or-create logic depends on it.

4. **Python API hello-world** on the PF machine: `import powerfactory`, `GetApplicationExt()`, activate project, execute `ComLdf`, print one bus voltage. Pin the Python version the PF release supports and record it.

### Gate 1

- [x] Template load flow reproduces Table 10 exactly — verified 2026-07-17 against `docs/39_Bus_New_England_System.pdf` (all 39 buses match to the printed precision; see docs/pf_gate1_record.md). **Gate 1 complete.**
- [x] Hello-world script runs on the PF machine; Python version pinned (2026-07-17: PF 2025 SP4, Python 3.12.13 = qOFO_clean, engine mode, project `\mschwenke.IntUser\qOFO\IEEE39_qOFO`).

---

## Phase 2 — Align the static 345 kV grid (sync script core + Gate A)

_This is the grind phase. Per-unit, tap-side and voltage-level conventions always cost more time than expected — budget 1–2 weeks._

### Steps

1. **Build the sync script core** (`pf_sync.py`): reads `snapshot.json`, find-or-create by `loc_name`, updates attributes; `--rebuild` flag deletes and regenerates all script-owned objects; `--dry-run` prints the diff. Object mapping:

    |pandapower|PowerFactory|
    |---|---|
    |`bus`|`ElmTerm` (+ `StaBar`/`StaCubic` plumbing)|
    |`line`|`ElmLne` + `TypLne` (per-km R′, X′, C′)|
    |`trafo`|`ElmTr2` + `TypTr2` (uk = `vk_percent`, ukr = `vkr_percent`, tap data)|
    |`trafo3w`|`ElmTr3` + `TypTr3` (per-winding uk/ukr, tap on HV, star-point 20 kV tertiary)|
    |`load`|`ElmLod`|
    |`sgen`|`ElmGenstat` + `ElmStactrl`|
    |`gen`|`ElmSym` (template-owned; script only sets dispatch / outserv)|
    |`shunt`|`ElmShnt`|

2. **Push the line data**: the `LINE_LENGTHS_KM` lengths and per-km values onto the existing TN lines (totals agree with Pai on both sides, but push ours so the models are numerically identical; include `LINE_LENGTH_FACTOR` if ≠ 1).

3. **Replicate machine-trafo changes**: LV winding 10.5 kV, OLTC ±9 × 1.25 % HV side, the bus 19→20→34 two-trafo chain collapsed to one unit, and G1 (bus 39): a **new** 10.5 kV terminal + machine trafo created by the builder (case39 has no step-up there). ⚠ Changing the terminal voltage level requires editing **both** the `TypTr2` LV voltage **and** the machine type's rated voltage.

4. **Set dispatch** for all retained gens (P from zonal dispatch and V setpoints from the snapshot).

5. **Write all TN loads** from the snapshot (const + var halves are already resolved to plain P/Q numbers there).

6. **Build the parity tool** (`pf_parity.py`): runs ComLdf, extracts all bus voltages/angles and branch flows, compares against `snapshot.json` `solution`, prints a sorted table of worst deviations and exits non-zero above tolerance.

### Gate A

- [x] `pf_parity.py` green on the **base** scenario for all snapshots (2026-07-17): `base_t0` max |Δvm| = 2.1·10⁻⁶ pu / |Δva| = 6.2·10⁻³ °, `base_peakres` 7.3·10⁻⁶ pu / 9.0·10⁻³ ° — both ~50× and ~1.4× inside the gates; all flows within 0.2 MW (float32 impedance storage sets the parity floor). **Gate A complete**; project left synced to `base_t0`.

---

## Phase 3 — wind_replace in PowerFactory (Gate B)

### Steps

1. **Out-of-service, don't delete**: removed machines (G2, G5, G6, G8 — see `removed_generators` in the snapshot) and their step-up trafos get `outserv = 1` inside a PF **Variation** named `wind_replace`.

2. **Create the wind parks**: one `ElmGenstat` per replacement at the 345 kV grid bus; `sgn` = `wp_sn`; P from the snapshot; Q limits ±sn (full STATCOM circle).

3. **Set Q directly from the snapshot** (`qgini`) — the snapshot already contains the converged Q.

4. **Attach one `ElmStactrl` per park** in Q-setpoint mode. These are the OFO write handles.

5. **RMS layer, staged**: start with the plain static generator; upgrade selected parks to WECC REGC/REEC/REPC in Phase 5 as a sensitivity.

### Gate B

- [x] Parity green on **wind_replace without underlays** at t0 and peak residual, including per-park Q (Gate B closed 2026-07-19; max voltage error 3.118e-8 pu, max angle error 5.407e-6 deg).

---

## Phase 4 — The four DSO underlays (Gate C)

_Pure generation from `SUBNET_DEFS` + `HV_LINE_TOPOLOGY` — this is where scripting pays for itself._

### Steps

1. **One `ElmNet` grid folder per subnet** (`DSO_1` … `DSO_4`).

2. **Buses and lines**: per DSO, 10 physical 110 kV buses and 11 physical lines from `HV_LINE_TOPOLOGY`, plus 3 explicit 20 kV tertiary buses and 7 internal `DN_AUX` load buses/links. The auxiliary links use r = x = 0.01 ohm over 1 km and keep ZIP loads electrically separate from fixed sgens.

3. **Coupling transformers — ⚠ three-winding**: 3 per subnet, `ElmTr3` + `TypTr3`, 345/110/20 kV (300/300/75 MVA), vk_hv/mv/lv = 12/8/10 %, shift_lv = 150°, OLTC ±13 × 1.25 % on the HV side; snapshot tap positions; addressable via `meta.hv_networks[*].coupling_trafo_indices` (pandapower `net.trafo3w`).

4. **TSO tertiary shunts**: MSC/MSR (or bipolar) banks at the first 20 kV tertiary bus of each subnet (`ElmShnt`, snapshot `step` state; kinds and levels in `meta.tso_tertiary_shunt_*`).

5. **Loads** straight from the snapshot. Loads co-located with fixed static generators are placed on deterministic internal auxiliary buses so pandapower and PF apply the ZIP factor only to the load; physical-parent zone ownership is retained.

6. **DERs**: TUDA wind parks, PV plants, 40 MVA coupling-bus STATCOM WPs — `ElmGenstat` + `ElmStactrl` each (roles `DSO-DER` / `DSO-COUPLING-WP` in the snapshot).

7. **TN-side bookkeeping is automatic**: deleted profile-half loads and removed coupling-bus sgens are already absent from the snapshot.

### Gate C

- [x] Full-system parity green on both full snapshots (2026-07-19): t0 max dV 1.545e-5 pu / max angle 4.059e-4 deg; peak residual max dV 1.609e-5 pu / max angle 5.377e-4 deg.
- [x] `--interfaces` table green for all 12 coupler HV/MV P/Q flows at both operating points; worst 3W deviation 2.673e-3 MW/Mvar. **Gate C complete.** Evidence: `docs/daily_log/07_2026/2026-07-19_rms_phase4_gate_c_closed.md`.

---

## Phase 5 — Dynamic wiring and screening

_From here on it is RMS work, not model translation._

### Steps

1. **Load models**: baseline = anchored ZIP (kpu = 1, kqu = 2 at 1.03 pu — parity-comparable by construction since the oracle uses the same model). Stress variant = induction-motor share. Record both as study-case variants.

2. **q(v) droop layer**: check whether the PF version's `ElmStactrl` supports a Q(V) characteristic **with deadband** (slope and deadband per sgen in the snapshot: `qv_slope_pu`, `qv_deadband_pu`). If not — this is the _single_ place a small DSL frame may be unavoidable; find out at Phase 5 start.

3. **OLTC policy**: every automatic tap controller OFF. Taps move only via scheduled events with a mechanical delay (τ ≈ 5 s); multi-tap moves are sequential events 5 s apart.

4. **Machine layer decisions**: PSS on/off (revisit after first modal results); wind parks — upgrade 1–2 parks to WECC templates and compare step responses.

5. **Screening battery** (scripted, per snapshot): 60 s flat run (drift < 10⁻⁴ pu); modal analysis (table of ζ, f, settling time, participation); open-loop worst-case steps per actuator class.

### Gate D

- [x] Flat run green; modal table reviewed and archived; step battery verdict on the **20 s** assumption documented, including the OLTC sequential-tap case. **Gate D complete (2026-07-20): PASS for every actuator class**, worst case 13.2 s (+60 Mvar TSO-park Q dispatch; 6.8 s margin); tap responses are delay-dominated (≈ 5k s for k sequential taps). Evidence: `docs/daily_log/07_2026/2026-07-20_rms_phase5_event_accumulation_fix.md`, `results/screening/full_t0_wecc/20260720-143633/`. ⚠ Battery hygiene: PF silently refuses event deletion while a calculation is active — `purge_events` must `ResetCalculation()` first and verify the folder is empty.

---

## Phase 6 — Co-simulation hookup

### Steps

1. **Plant abstraction in qOFO_GH**: `Plant` protocol with `read_y()`, `apply_u()`, `advance(T)`; `PandapowerStaticPlant` reproduces today's behaviour bit-for-bit (regression test).

2. **`PowerFactoryPlant`**: `apply_u` → `EvtParam` on `REEC_D.Qext` (DERs), AVR `usetp` (machines), sequential tap events at t + 5 s (2W/3W OLTCs), and immediate MSC/MSR step events; `advance(T)` continues `ComSim`; `read_y` refreshes the pandapower measurement image from paused-state attributes; trajectories are harvested per chunk from `ElmRes`/`ComRes`.

3. **Event hygiene**: reset the active calculation, delete the simulation-event folder contents between runs, and verify it is empty before initialisation.

4. **Closed-loop replay**: one CIGRE-paper scenario window with the full 20 s STS / 180 s TS cascade against the dynamic plant; overlay y(t_k⁻) on the quasi-static pandapower trajectory — the headline validation figure.

5. **Settling statistics**: per dispatch interval, per signal: 2 %-band settling time → distribution, max and 95th percentile vs the 20 s line.

### Gate E

- [x] Static-plant and runner regression tests green (refactor changed nothing).
- [x] A 900 s closed-loop baseline replay completed using `run_multi_system_ofo.make_config`; overlay figures and settling tables were generated (`results/rms_phase6_replay/0005_2026-07-20_163715`).
- [ ] **Plant-law equivalence blocker:** the static plant applies the re-anchored Q(V) characteristic after each `q_set`, while the current RMS plant holds raw `REEC_D.Qext`. The baseline is diagnostic and is marked `BLOCKED_DER_QV_MISMATCH`; it is not the headline validation result.
- [ ] Implement and verify a separate RMS plant-side Q(V) pre-controller, then repeat the 900 s replay and close Gate E only if the equivalent-plant settling/endpoint criteria pass.

---

## Risk register

|Risk|Phase|Mitigation|
|---|---|---|
|Per-unit / tap-convention mismatches stall Gate A|2|Parity tool sorts worst-first; debug element-wise; budget 1–2 weeks|
|3W coupler model mismatch (star-point handling, 150° tertiary shift)|4|Dedicated `--interfaces` parity column from day one; compare per-winding flows, not just HV|
|Template PSSs disabled → poorly damped inter-area modes ring > 10 s|5|Modal screen _before_ closed loop; the damping shift from wind_replace is reportable in its own right|
|Load voltage dependency (kpu=1, kqu=2) ≠ constant-PQ oracle|1/5|Constant power for parity; ZIP/IM only as labelled stress variant|
|RMS DER model lacks the static re-anchored Q(V) actuator law|5/6|`ElmStactrl` Q(V) is load-flow-only; add one minimal, separately tested plant-side DSL/pre-controller layer ahead of constant-Q `REEC_D`|
|Stale simulation events corrupt runs|6|Purge event folder in the plant constructor; log all created events|
|Sequential tap events exceed the 10 s window|5/6|That is a _finding_, not a bug — feeds back into the OFO design|
|GUI edits diverge from script|all|Script-owned objects only via sync; re-run `pf_parity.py` after any GUI session|
|Licence seat blocks Monte Carlo scale-out|6|Curated representative cases for RMS; full MC stays quasi-static in pandapower|

## Suggested order of attack (effort-calibrated)

1. Phase 0 (2–3 days) — exporter + round-trip test. ✅ 2026-07-17
2. Phase 1 (1–2 days GUI + API hello-world).
3. Phase 2 (1–2 weeks) — sync core + Gate A. _The grind; everything after is faster._
4. Phase 3 (2–3 days) — Gate B.
5. Phase 4 (3–5 days) — Gate C.
6. Phase 5 (1–2 weeks) — q(v) feasibility check first, then screening battery.
7. Phase 6 (1 week) — plant abstraction first (pure Python, PF-independent), then the PF plant.

Go/no-go checkpoint for the dissertation timeline: end of Phase 5.

---

## Appendix — Repo reality check (2026-07-17, Phase 0)

Findings from the code audit that amend the original plan text:

1. **TN–HV coupling is three-winding.** `_create_hv_subnetwork` couples each subnet via `pp.create_transformer3w_from_parameters` (345/110/20 kV, 300/300/75 MVA, vk 12/8/10 %, vkr 0.30/0.20/0.25 %, pfe 80 kW, i0 0.04 %, shift_mv 0°, shift_lv 150°, tap HV ±13 × 1.25 %). The plan's original "NC transformers (`ElmTr2`)" wording is superseded: PF Phase 4 must create `ElmTr3` + `TypTr3`, and interface parity compares `res_trafo3w` per-winding flows. 4 subnets × 3 couplers = 12 units.
2. **G10 decision already in code.** `wind_replace.py` removes only G8 (term 36) in Zone 1; G10 (Hydro, term 29, 1 GVA) is retained. Removed fleet: G2 (term 30), G5 (33), G6 (34), G8 (36) → four wind parks. Retained synchronous fleet: G1 (slack anchor, 10 GVA), G3, G4, G7, G9, G10.
3. **G1 has a scripted machine trafo.** case39 has no step-up at bus 39; the builder creates a new 10.5 kV terminal bus + 2W trafo (`MachineTrf|gen…`). The PF template's G 01 sits directly on bus 39 — the sync script must either create the same trafo or (better) replicate the builder exactly, since the snapshot solution is computed with it.
4. **TSO tertiary shunts.** MSC/MSR (or legacy bipolar) banks at the first 20 kV tertiary of each subnet; kinds, per-step Mvar, level counts and step states are in `meta.tso_tertiary_shunt_*` and `model.shunt`.
5. **Slack is distributed.** There is **no** `ext_grid`; bus 38 hosts a `slack=True` gen (G1) and every machine carries `slack_weight = sn_mva`. All experiment power flows (and the stored snapshot solutions) run with `distributed_slack=True, enforce_q_lims=True`. PF's single reference machine G 01 approximates this only at the parity operating point because the snapshot stores the *converged* per-machine P (post-slack-allocation) — push `res_gen.p_mw`, not `gen.p_mw`, as `pgini`.
6. **Droop layer parameters** (`q_mode`, `qv_slope_pu`, `qv_vref_pu`, `qv_deadband_pu`, per sgen) are serialised in `model.sgen` — defaults: slope 0.06 pu, vref 1.03 pu, deadband 0.01 pu.
7. **Snapshot exporter artefacts**: `export/dynamic_snapshot.py` (schema + round-trip), `export/make_snapshots.py` (CLI driver, phases `base` / `wind_replace` / `full`, auto timestamps `t0` / `peakres`, OLTC init on by default), `tests/export/test_dynamic_snapshot_roundtrip.py` (Gate 0 — green 2026-07-17, rebuilds are bit-identical).
8. **Stranded-load fix.** `remove_generators` now drops loads left on removed 10.5 kV terminal buses (case39's IEEE bus-31 load on G2's terminal; pandapower had been silently zeroing them, `net.load` overstated served demand by ≈ 6.5 MW). Behaviour-neutral for the power flow; open question whether that demand should instead move to the grid bus (would shift every wind_replace operating point — deliberately not done).
9. **wind_replace @ peakres is infeasible.** The intermediate model (4 machines removed, full TN load, no underlays) does not converge at the yearly residual peak (13.04.2016 11:00; ≈ 1780 MW cross-zone spill, Q-limits binding) — also with dc init and 100 NR iterations. Gate B therefore uses `t0`; the **full** model converges at its own peak (13.04.2016 09:00).
10. **Reference snapshots** (`export/snapshots/`, all round-trip verified): `base`, `wind_replace`, `full` × {`t0`, `peakres`}. Full-model inventory: 88 buses, 79 lines, 8 × 2W trafos (6 machine OLTC + 2 network OLTC), 12 × 3W couplers, 113 loads, 44 sgens (4 TSO-WP, 28 DSO-DER, 12 DSO-COUPLING-WP), 6 machines, 8 MSC/MSR shunts.

### Phase-1 decisions and progress (2026-07-17)

11. **PF project confirmed**: `qOFO\IEEE39_qOFO` (folder `qOFO`, copy of the DIgSILENT template). **PowerFactory 2025**, scripts in **external engine mode**. Study-case layout proposed in docs/pf_api_notes.md §3 (`01_LDF_Parity`, `02_RMS_CoSim`, later `03_RMS_Screening`) — separate cases because active variations are stored per study case.
12. **Load model decision: anchored ZIP on both sides.** `MultiTSOConfig.load_model = "zip"` (default) applies `network/ieee39/load_model.py::apply_zip_load_model` — exact exponent image (kpu, kqu) = (1, 2) anchored at 1.03 pu, realised as 100 % const-I P / 100 % const-Z Q with the anchor folded into the bases. `"const_pq"` replays the legacy plant. ⚠ Every experiment operating point shifts under the new default; existing tunings/results predate this and are only reproducible with `load_model="const_pq"`.
13. **ZIP side effect**: `wind_replace @ peakres` (13.04.2016 11:00) now **converges** — load relief at depressed voltages rescues exactly the case that was infeasible under constant power (finding no. 9 partially superseded; the constant-P infeasibility remains documented as evidence of how close to the envelope that intermediate operating point is). Gate B now has a stress snapshot.
14. **Phase-1 scaffolding**: `pf/session.py` (engine/embedded session, Fail-Fast, DEBUG-logged), `pf/naming.py` + `docs/pf_naming.md` (loc_name convention, proven total + collision-free on all six snapshots by `tests/pf/test_naming.py`), `pf/hello_pf.py` + `docs/pf_api_notes.md` (Gate-1 manual test).
15. **Gate-1 execution (2026-07-17, on the PF machine)**: engine mode green on the first run — PF 2025 SP4 + Python 3.12.13 (qOFO_clean at `F:`), project activated, ComLdf converged. Machine names `G 01`…`G 10` confirmed → `TEMPLATE_MACHINE_NAMES` verified (`TEMPLATE_NAMES_VERIFIED = True`). Evidence: docs/pf_gate1_record.md (all 39 voltages/angles for the Table-10 check, TypSym data, ComLdf flags as found). Study cases `01_LDF_Parity` and `02_RMS_CoSim` created as scripted copies of `1. Power Flow` (`pf/setup_study_cases.py`, idempotent). Findings: template angle reference is at Bus 31 / G 02 (move to G 01 = Phase-2 sync action); **G 05 ships with Sr = 300 MVA, H = 4.333 s** — inconsistency confirmed but deferred (G 05 is removed in wind_replace; decide the fix at Phase-5 start after reading TypSym xd); no machine within 1 Mvar of a Q limit in any reference snapshot → `iopt_lim = 0` is exact for parity.
