# 2026-07-17 (3rd entry) — Gate 1 executed on the PF machine

**Context.** First live contact with PowerFactory. Session moved to the PF
machine; `pf/session.py` was adapted so the 2025 SP4 API path is the
built-in default (`DEFAULT_PF_PYTHON_PATH`, overridable via the
`QOFO_PF_PYTHON_PATH` environment variable — the constant previously held
the env-var *name*, and the hardcoded path would have broken the lookup).

## Environment reality (recorded in docs/pf_api_notes.md)

- PowerFactory **2025 SP4** (2024 SP7 also installed); API folders
  Python 3.9–3.13.
- Interpreter: `F:\python_environments\qOFO_clean\python.exe`
  (**3.12.13**) matches the `Python\3.12` API folder — engine mode works.
- Project: `\mschwenke.IntUser\qOFO\IEEE39_qOFO` (DIgSILENT 39-bus
  template copy), 9 template study cases.

## Gate-1 runs (all green, first attempt)

1. **`pf/hello_pf.py`**: project activation, study-case inventory, all ten
   `ElmSym` machines (`G 01` … `G 10`, in service), ComLdf converged,
   voltages read. → `TEMPLATE_MACHINE_NAMES` in `pf/naming.py` verified
   as-is; `TEMPLATE_NAMES_VERIFIED = True`.
2. **`pf/gate1_record.py`** (new, read-only evidence collector) →
   `docs/pf_gate1_record.md`: all 39 bus voltages/angles for the manual
   Table-10 comparison (still a user task), per-machine TypSym data,
   ComLdf flags as found (`iopt_net/pq/at/asht/lim/plim` all 0).
3. **`pf/setup_study_cases.py`** (new, idempotent): created
   `01_LDF_Parity` and `02_RMS_CoSim` as `AddCopy` duplicates of
   `1. Power Flow` (copies inherit grid activation; active variations are
   stored per study case — the isolation argument). Template case left
   active.

## Findings

- **G 05 inconsistency confirmed**: TypSym `sgn = 300 MVA`, `h = 4.333 s`
  → H·S is half the IEEE task-force value (26 s · 100 MVA). Two
  consistent fixes (Sr→600 with pu-impedance rescale, or H→8.666 s at
  Sr = 300). **Deferred to Phase-5 start**: G 05 is removed in
  wind_replace, so only base-scenario RMS work is affected; the choice
  needs TypSym xd read first to see which base the template's pu data
  assumes.
- **Template angle reference sits at Bus 31 / G 02** (phi = 0 there;
  u = 0.982 = the case39 ex-slack setpoint). Moving the reference to
  G 01 is a Phase-2 `pf_sync` action (`ip_ctrl`), not a GUI edit.
- **Q-limit check across all six reference snapshots**: no machine within
  1 Mvar of its limit → `ComLdf.iopt_lim = 0` reproduces the
  `enforce_q_lims=True` oracle exactly at these operating points.
- Parity option delta vs template defaults is therefore minimal:
  `iopt_pq → 1` (anchored ZIP) + per-load exponents (1, 2) with u0 = 1.0;
  everything else stays as shipped.

## Gate 1 status

- [x] Hello-world + Python pin (3.12.13 / PF 2025 SP4, engine mode).
- [x] Study cases for the co-simulation created.
- [x] Table-10 comparison — user provided
  `docs/39_Bus_New_England_System.pdf`; automated comparison of all 39
  buses: **PASS to full printed precision** (max |Δu| = 0.5e-4 pu,
  |Δφ| = 0.5e-2 deg = rounding). Template pristine. **Gate 1 complete.**

## Addendum — G 05 inertia resolved from the PDF (Tables 7/8)

Cross-checking all ten units: Table 8 (PF model, machine base) follows
from Table 7 (100 MVA literature base) via x·(Sr/100) and H·(100/Sr)
exactly — except G 05's inertia. Its reactances convert on Sr = 300 MVA
(xd 0.67·3 = 2.010 ✓, x′d 0.132·3 = 0.396 ✓) but H = 4.333 s
= 26.0·100/**600**: the inertia alone was converted with the wrong base.
Correct minimal fix (when base-scenario RMS needs G 05): TypSym
`Type Gen 05`.h → **8.667 s** at Sr = 300. The build plan's tentative
"Sr → 600" route is **ruled out** (it would corrupt the reactances by 2×).
Not applied — G 05 is out of service in wind_replace; the template stays
pristine until the fix is needed.

## Next (Phase 2)

`pf/pf_sync.py` core (SyncHandler per element type, find-or-create by
loc_name against `pf/naming.py`, ChangeReport, `--dry-run`/`--rebuild`,
phase `base` first) + `pf/pf_parity.py` (vm/va/flow deviation report,
tolerances 1e-4 pu / 0.01°) against `export/snapshots/base_t0…json` in
study case `01_LDF_Parity`. Now iterable live from this machine.
