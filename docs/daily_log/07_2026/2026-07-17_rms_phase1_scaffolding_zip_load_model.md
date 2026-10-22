# 2026-07-17 (2nd entry) — RMS Phase 1 scaffolding + anchored ZIP load model

**Context.** Continuation of the RMS co-simulation build after Gate 0.
User decisions this session: PF project **`qOFO\IEEE39_qOFO`** exists
(folder `qOFO`); **PowerFactory 2025**; **external engine mode**; and the
load-model question resolved as **"ZIP both sides now"**.

## A. Anchored ZIP load model (plant-model change, both sides)

**Decision.** Both the pandapower oracle and the PF model use
voltage-dependent loads P = P_prof·(V/1.03)¹, Q = Q_prof·(V/1.03)²
(SimBench profiles are voltage-agnostic scalings; the 1.03 pu anchor is our
convention — it preserves the pre-ZIP power balance at the voltage
setpoint). Key enabling fact: PF's RMS load model re-anchors at the initial
load-flow point, and the exponent pair (1, 2) has an **exact** pandapower
image: 100 % const-I on P / 100 % const-Z on Q with bases divided by 1.03
and 1.03² (no approximation; verified numerically).

**Changes.**

- `network/ieee39/load_model.py` (new): `apply_zip_load_model(net,
  anchor_vm_pu=1.03)` — sets shares, rescales `p_mw/q_mvar` and
  `base_p_mw/base_q_mvar`, records `zip_anchor_vm_pu` per load; raises on
  double application (rescale must not compound).
- `configs/config.py`: `load_model: str = "zip"` (**new default — changes
  every experiment**), `load_zip_anchor_vm_pu: float = 1.03`;
  `"const_pq"` replays the legacy constant-power plant.
- `experiments/runners/multi_tso_dso.py`: step [3b] applies the model after
  `add_hv_networks`, before q-mode tagging.
- `export/make_snapshots.py`: same wiring + `--load-model` CLI;
  `export/dynamic_snapshot.py`: `zip_anchor_vm_pu` serialised.
- `tests/test_zip_load_model.py` (new, 5 tests): anchor identity at
  V = 1.03, exact exponent image off-anchor, double-apply guard, full-build
  consistency (every load obeys P = p·V, Q = q·V²), legacy option.

**Conventions.** Contingency stress loads stay constant-PQ (specified
disturbance magnitudes). `compute_zonal_gen_dispatch` now targets the
1.0-pu-equivalent nominal (≈ profile/1.03); the ±3 % voltage-dependence
remainder lands on the slack — inherent to voltage-dependent load, accepted.
`rec.total_load_p_mw` (runner) records the anchored nominal, not served
power; served sums come from `res_load` as before.

**Consequences.** All quasi-static operating points shift slightly;
existing tunings/campaign results predate the default flip and reproduce
only with `load_model="const_pq"`. Re-validation of headline experiments is
an open task for the user.

**Side effect (physical).** `wind_replace @ peakres` (13.04.2016 11:00),
infeasible under constant-P, now **converges** — load relief at depressed
voltages. All six reference snapshots (3 phases × {t0, peakres}) exist and
round-trip bit-identically.

## B. Phase-1 scaffolding (PF side prepared; runs on the PF machine)

- `pf/session.py` (new, stdlib-only): engine/embedded application handle
  (`GetApplicationExt` with fallback, cached singleton), `connect()` with
  project activation (default `qOFO\IEEE39_qOFO`), study-case activation,
  `get_by_name` (exact-match, Fail-Fast on 0/>1), `run_ldf` with logged
  option application, `get_attr`/`set_attrs` wrappers. Engine-mode Python
  path via env var `QOFO_PF_PYTHON_PATH`.
- `pf/naming.py` + `docs/pf_naming.md` (new): loc_name convention embedding
  pandapower indices; `build_name_map(snapshot)` proven total and
  collision-free on all six snapshots (`tests/pf/test_naming.py`, 11+
  tests). Template machines are *not* renamed — addressed via
  `TEMPLATE_MACHINE_NAMES` (placeholder spellings, verified in Gate 1;
  `TEMPLATE_NAMES_VERIFIED` gate flag for pf_sync).
- `pf/hello_pf.py` + `docs/pf_api_notes.md` (new): Gate-1 manual smoke test
  (interpreter/module provenance, study cases, ElmSym names, ComLdf, LDF
  voltage-dependency flag probe) + setup notes (Python pin procedure,
  licence-seat behaviour, study-case layout `01_LDF_Parity` /
  `02_RMS_CoSim`, parity-relevant option collection — LDF load voltage
  dependency now **ON** with exponents (1, 2), u0 = 1.0 verification note).

## Test status

`tests/export` + `tests/pf` + `tests/test_zip_load_model.py` +
`tests/test_network.py` + `tests/test_msc_msr_banks.py` +
`tests/test_tag_der_q_modes.py`: **65 passed, 2 skipped**.

## Next

User runs `python pf\hello_pf.py` on the PF-2025 machine (after setting
`QOFO_PF_PYTHON_PATH`), pastes the output; then: verify Table 10, fix
`TEMPLATE_MACHINE_NAMES`, create the two study cases → Gate 1 complete →
Phase 2 (`pf_sync.py` core + Gate A).
