# 2026-07-17 (5th entry) — RMS Phase 2: pf_sync + pf_parity, **Gate A green**

**Context.** Phase 2 of the RMS build plan, executed live on the PF machine
(engine mode). Deliverables: `pf/pf_sync.py` (base-phase sync core),
`pf/pf_parity.py` (Gate A–C comparator), `pf/probe_api.py` (attribute-reality
probe). SCIP is now the MIQP backend on every machine (user edit;
provenance comment added in `optimisation/miqp_solver.py`).

## Result

**Gate A complete on the first parity day** (plan budget: 1–2 weeks):

| snapshot | max \|Δvm\| [pu] | max \|Δva\| [deg] | flows |
|---|---|---|---|
| base_t0 | 2.1e-6 | 6.2e-3 | ≤ 0.19 MW |
| base_peakres | 7.3e-6 | 9.0e-3 | ≤ 0.11 MW |

Gates: 1e-4 pu / 0.01°. Loads match to 1e-4 MW — the anchored-ZIP mapping
(pandapower const-I/const-Z ↔ PF `General Load Type` u¹/u² with
`iopt_pq=1`) is exact. Residual ~0.2 MW slack-P/flow pattern = PF float32
attribute storage (rel. 1e-7 on impedances), far below gate. The sync is
idempotent (rerun: 0 operations) and re-targets operating points
(t0 ↔ peakres: 57 updates each way). Project left synced to `base_t0`.

## Architecture

- `pf/pf_sync.py`: SyncContext (dry-run indirection, ChangeReport,
  terminal alias map), template **adoption** (rename `Bus NN`/`Line NN-NN`
  to the pf/naming.py convention, push snapshot parameters into the
  existing objects/types), creation of model-divergent objects, deletion
  of superseded template objects, template-owned `ElmSym` handling
  (dispatch = converged `solution.gen.p` per parallel unit, `usetp`,
  `ip_ctrl` reference flag on G 01, `TypSym.ugn` rebase to 10.5 kV, G 01 +
  G 07 reconnection to new GT terminals via fresh `StaCubic`).
- `pf/pf_parity.py`: authoritative ComLdf parity option set (`iopt_pq=1`,
  everything automatic OFF), slack-terminal angle alignment, per-family
  max deviations + worst-N table, exit code on the vm/va gate.

## Model-difference findings (case39/pandapower vs DIgSILENT template)

1. **IEEE 23–36 is a line in case39** (ratio-1.0 branch) but a trafo in the
   template → template `Trf 23 - 36` deleted, `TN_line29` created, and G 07
   (like G 1) sits behind a *builder-created* machine trafo on a new
   terminal (`GT_bus40`/`GT_bus39`).
2. **case39 has 21 loads** (adds IEEE buses 1 and 9) vs 19 in the template.
3. Template fixed ratios (Table 5) are tap-encoded (`dutap`·`nntap`); our
   snapshot tap model overwrites them — pandapower discarded those ratios
   at build time, and the oracle is self-consistent without them.

## PF 2025 API quirks captured (the classic Gate-A time sinks)

1. **`TypTr2.uktr`/`uktrr` are derived read-only views**; writes are
   silently ignored. The storage is the pu pair **`r1pu`/`x1pu`**
   (r1 = vkr/100, x1 = sqrt((vk/100)² − r1²)) — write-through verified.
2. **Attributes are float32**: idempotency comparison needs rel. tol ≥ 1e-6
   (`_REL_EPS = 2e-6`); parity floor ~1e-6 pu.
3. **`ElmSym` input vs result convention**: `pgini`/limits are per parallel
   unit (`ngnum`), results (`m:P:bus1`) are plant totals.
4. Study-case `Activate()` returns 1 when already active → idempotent
   activation in `pf/session.py`.
5. First-run ordering bug (fixed): branch matching parsed template bus
   names *after* the bus adoption had renamed them → duplicate lines were
   created; matcher now resolves endpoints through the naming reverse map,
   stale `Line`/`Trf` remnants are deleted, and the fixed sync repaired
   the project state itself.

## G 05 record corrected

`G 05.ngnum = 2` (probe): plant kinetic energy is correct as shipped
(2·4.333·300 = 2600 MVA·s); the earlier "H → 8.667 s" conclusion is
withdrawn. Actual template inconsistency: every G 05 per-unit reactance is
half its correct half-plant value (xd 2.01 vs 4.02). Fix (deferred, G 05
removed in wind_replace): double all `Type Gen 05` reactances; never touch
H or Sr. docs/pf_gate1_record.md and docs/pf_api_notes.md updated.

## Next (Phase 3, Gate B)

Extend pf_sync with phase `wind_replace`: PF Variation handling (create /
activate variation, outserv the removed machines G2/G5/G6/G8 + their
machine trafos), `ElmGenstat` + `ElmStactrl` creation for the four TSO wind
parks, parity incl. per-park Q against `wind_replace_t0` / `_peakres`.
