# PowerFactory Python API — setup notes and Phase-1 manual test

Status: prepared 2026-07-17 (dev server has no PF licence — everything here
runs on the PF machine only). Project: **`qOFO\IEEE39_qOFO`** (user-created,
copy of the DIgSILENT "39 Bus New England System" template).

Decisions 2026-07-17: **PowerFactory 2025**, **external engine mode**,
**anchored ZIP load model on both sides** (kpu = 1, kqu = 2 at 1.03 pu; see
§4 and `network/ieee39/load_model.py`).

## 1. One-off setup on the PF machine

1. Find the PF installation directory, e.g.
   `C:\Program Files\DIgSILENT\PowerFactory 2024 SP4\`.
2. List `…\Python\` — it contains one folder per supported Python version.
   **Recorded 2026-07-17 (Gate 1 run, green):**
   - PF version: **PowerFactory 2025 SP4** (`C:\Program Files\DIgSILENT\PowerFactory 2025 SP4`; 2024 SP7 also installed; API folders 3.9–3.13)
   - Python pin: **3.12.13** = `F:\python_environments\qOFO_clean\python.exe` (qOFO_clean env)
   - PF database user: `mschwenke.IntUser`
3. For **external engine mode** (our default), the PF Python path is baked
   into `pf/session.py::DEFAULT_PF_PYTHON_PATH` (2025 SP4 / 3.12); the
   environment variable `QOFO_PF_PYTHON_PATH` overrides it when set.
4. Licence note: engine mode consumes a licence seat. If the GUI is open at
   the same time and only one seat exists, `GetApplicationExt()` fails —
   close the GUI first (the error text will say so).
5. Engine-mode processes may only call `GetApplication*` once;
   `pf/session.py` caches the handle accordingly. If a script crashes and
   the licence seems stuck, wait ~30 s or check Task Manager for orphaned
   `python.exe`.

## 2. The Phase-1 smoke test (Gate 1, second checkbox)

From the repo root on the PF machine:

```bat
python pf\hello_pf.py
```

Expected output blocks:

1. Interpreter + module provenance (DEBUG log shows the `powerfactory`
   module path).
2. Active project `…\qOFO\IEEE39_qOFO` + the study case list.
3. All `ElmSym` machines with `outserv` state → **verify
   `TEMPLATE_MACHINE_NAMES` in `pf/naming.py`**, correct spellings if they
   differ, then set `TEMPLATE_NAMES_VERIFIED = True`.
4. `ComLdf` executes and the first terminal voltages print.
5. The load-voltage-dependency flag of `ComLdf` (attribute probe) — record
   its name and default for the parity option set (Phase 2).

Paste the full console output back into the Claude Code session. Failures
raise `PFSessionError` with the PF error text — paste that too.

## 3. Study-case layout for the co-simulation (to create once in the GUI)

Keep the imported template study case untouched as the pristine reference
(Gate 1 Table-10 check runs there). Create additionally:

| Study case | Purpose | Notes |
|---|---|---|
| `01_LDF_Parity` | pf_sync + pf_parity (Gates A–C) | **created 2026-07-17** (scripted copy of `1. Power Flow` via `pf/setup_study_cases.py`); the sync script activates the variation set matching the requested `--phase` and owns the ComLdf parity options |
| `02_RMS_CoSim` | RMS initialisation + OFO-in-the-loop (Phases 5–6) | **created 2026-07-17** (same copy); own `ElmRes`; simulation-events folder is purged by the plant wrapper; all automatic tap controllers OFF (taps move via events only) |
| `03_RMS_Screening` | Phase-5 battery (flat run / modal / steps) | created when Phase 5 starts |

Variations are project-global, while their active state is stored per study
case. The derived model is layered and must be activated in order:
`base -> wind_replace -> full`. `wind_replace` owns the TN wind delta;
`full` owns only the four DSO underlays and is never a stand-alone alternative.
This is exactly why co-simulation gets its own cases: parity and RMS runs can
never silently inherit each other's model state.

⚠ **Recording-stage ordering (fixed 2026-07-19).** When multiple layers are
active together, PF records new objects in the *latest* expansion stage whose
activation time is `<=` the study-case time (`1401264000`, 2014-05-28). Each
layer therefore needs a **strictly increasing** stage epoch, defined in
`pf/session.py::STAGE_EPOCHS` (`wind_replace` = 2000-01-01, `full` =
2005-01-01); equal epochs made PF record the `full` DSO underlay into the
`wind_replace` stage, leaking ~567 MW of DSO load into wind_replace-alone and
breaking Gate B. **Regression guard:** after every `full` sync, re-run the
`wind_replace` parity — a Gate-B break is the signature of a recording-stage
mis-target.

## 4. Known parity-relevant PF options (collected for Phase 2)

- **Load model (decision 2026-07-17): anchored ZIP on both sides.** The
  pandapower oracle serves `P = p_mw·V` and `Q = q_mvar·V²` (V in pu of
  nominal), with the 1.03 pu anchor already folded into the snapshot's
  `p_mw`/`q_mvar` values (`network/ieee39/load_model.py`). On the PF side:
  per-load voltage exponents kpu = 1, kqu = 2, `plini`/`qlini` taken 1:1
  from the snapshot, and the ComLdf option "consider voltage dependence of
  loads" **ON**. ⚠ Verify during Gate A that PF's LDF dependency is
  normalised at u0 = 1.0 pu (TypLod input); if the type carries a
  different u0, set it to 1.0 — otherwise the two anchors diverge and the
  parity gate will show a uniform ~3 % load offset.
- **Three-winding tap placement (verified 2026-07-19, PF 2025 SP4):**
  pandapower `tap_at_star_point=False` maps to `TypTr3.itapos=1` (tap at
  winding terminal); `itapos=0` models the tap at the star point. The solved
  ratio `ElmTr3.m:t_h` is otherwise exactly `1 + n*du/100`. The placement
  changes which side of the winding impedance sees the ratio and is required
  for Gate-C parity.
- Automatic tap adjustment OFF in `01_LDF_Parity` — tap positions come from
  the snapshot.
- Reactive power limits: the snapshot solutions were computed with
  `enforce_q_lims=True`, but **no machine is within 1 Mvar of a limit in
  any of the six reference snapshots** (checked 2026-07-17) — interior
  solutions, so `ComLdf.iopt_lim = 0` matches the oracle exactly at these
  operating points.
- Reference machine / slack: the snapshot stores **converged per-machine P**
  (`solution.gen.p_mw`, post-distributed-slack); push those as `pgini` so
  PF's single reference machine only covers losses mismatch. ⚠ The template
  currently has its angle reference at **Bus 31 / G 02** (phi = 0 there in
  the Gate-1 record); the planned move to **G 01** is a pf_sync Phase-2
  action (`ip_ctrl` flag), scripted, not a GUI edit.
- ComLdf flags as found in the template (Gate-1 record): `iopt_net=0` (AC
  balanced), `iopt_pq=0`, `iopt_at=0`, `iopt_asht=0`, `iopt_lim=0`,
  `iopt_plim=0`. Parity set changes only `iopt_pq → 1` (anchored ZIP) plus
  per-load exponents.
- **G 05 type data — corrected 2026-07-17**: `G 05.ngnum = 2` (two
  parallel 300 MVA units). Plant kinetic energy 2·4.333·300 = 2600 MVA·s
  matches the literature exactly → **H is correct as shipped**; instead,
  every per-unit reactance is **half** its correct half-plant value
  (xd 2.01 vs 4.02 = 0.67·600/100, same factor on xq/x′/x″/xl). Fix if
  base-RMS fidelity is needed: double all `Type Gen 05` reactances, touch
  nothing else. Deferred (G 05 removed in wind_replace); irrelevant to LDF
  parity. ⚠ Scripting rule derived from this: `pgini`/limits are
  **per parallel unit** (divide plant values by `ngnum`), while PF
  **result** attributes (`m:P:bus1`) are plant totals.
