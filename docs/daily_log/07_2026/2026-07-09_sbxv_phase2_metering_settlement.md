# 2026-07-09 — SBX-V Phase 2: metering + settlement

**What:** Implemented Phase 2 of the SBX-V build plan: `sbxv/metering.py` and
`sbxv/settlement.py`, plus `tests/sbxv/test_metering_settlement.py` (33 tests; SBX-V suite now
70, all green). Both regulatory PDFs under `docs/regulatory/` were read; the settlement rules
were implemented from the Leitfaden text directly (relevant extracts: §4.7 tolerance, §5.2/§8.3
Aggregationsgebiete/Saldierung, §7 Verrechnung complete, Tabellen 8.1/8.2, worked example §8.2).

**Method / structure:**

- Metering: four-quadrant register model per NVP (signed `q_hv` split by sign into
  (Q1+Q2)/(Q3+Q4) work registers per 15-min window), Saldierung per AggregationArea per
  [LF §8.3] — the numeric example of Abb. 8.3 is a verbatim unit test. Fail-fast interval
  recording (contiguity, no straddling, coverage assert, partial tail never settled).
- Settlement: `SettlementEngine.settle(observations, grants, incapabilities)` → window rows
  (energy, case-classified), day rows (Leistungspreis accrual per [LF §7.4] with full/pro-rata
  suspension), totals; CSV writers with documented schema. Both worlds: Tabelle 8.2
  (DSO delivers, opposite-edge reference; worked example VH = 200 Mvar) and Tabelle 8.1
  (TSO delivers, same-side reference, in-band deduction, case 3a via `IncapabilityRecord`).

**Documented interpretations** (STATUS_SBXV.md §2.2): ad-hoc downstream delivery on a logged
call without grant (own-edge reference, Grenzpreise) vs. self-initiated exceedance (Tabelle 8.1
case 4); re-issued same-day grants accrue one day of Vorhaltung; sub-day grants pay the full
day.

**One bug found during testing:** window-boundary lookup in `AreaMeter` double-applied the snap
tolerance and misclassified an interval ending exactly on a boundary as straddling — fixed.

**Why:** Plan §9 Phase 2; V-D8 settlement fidelity is the ground truth against which the V-D9
MIQP incentive may only approximate. R4 established (re-run every later phase).

**Next:** Phase 3 — request pipeline + grants ledger (`need_flag.py`, `feasibility.py`,
`pipeline.py`, `grants_ledger.py`, `potentials.py` incl. day-ahead forecast and the
missing-message substitute).
