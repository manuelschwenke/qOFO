# 2026-07-19 — Gate B regression fix: deterministic variation stage ordering

**Context.** Independent re-confirmation of the Phase-4 (Gate C) closure
surfaced a regression: Gate C (full) passed at both operating points, but
**Gate B (wind_replace-alone) failed at vm 4.5e-2 / va 14.7°**, with the
slack at 2131 MW vs 1564 expected (+567 MW). Base (Gate A) stayed clean.

## Root cause

The layered build activates `base -> wind_replace -> full` and records each
delta in its own PowerFactory expansion stage (`IntSstage`). But
`ensure_variation` gave **every** stage the same activation time
(`tAcTime = 946684800`, 2000-01-01). When several variations are active at
once, PF records new objects in the *latest* stage whose activation time is
`<=` the study-case time (`1401264000`, 2014-05-28); with equal times the
resolution is ambiguous and PF picked **wind_replace**. So the entire
Phase-4 DSO underlay — 28 auxiliary buses + ~567 MW of DSO load, created
while both variations were active — was recorded in the `wind_replace`
stage instead of `full`. It therefore leaked into wind_replace-alone
(confirmed: `app.GetRecordingStage()` returned `wind_replace/stage1` with
both active; `AUX_DSO_1_bus56` present whenever wind_replace was active;
base state had 0 aux terminals). Phase 4's summary only re-checked Gate C,
so the regression went unnoticed.

## Fix (`pf/session.py`)

Give each layer a **strictly increasing** stage epoch so the topmost active
layer deterministically owns the recording stage:

```
STAGE_EPOCHS = {"wind_replace": 946684800,   # 2000-01-01
                "full":         1104537600}   # 2005-01-01  (> wind_replace, < study time)
```

`ensure_variation` now reads `stage_epoch_for(name)` and, for a
pre-existing stage whose time disagrees, corrects it in place (self-heals
older projects). No change to `pf_sync.py` / `pf_parity.py` was needed —
they already call `ensure_variation`.

## Rebuild + verification (live, PF 2025 SP4)

Nuked both variations (back to clean base), then re-synced each layer:

| Gate | Snapshot | vm | va | Notes |
|---|---|---|---|---|
| A | base_t0 | 2.091e-6 | 6.2e-3° | 40 terminals, 0 aux |
| B | wind_replace_t0 | **3.118e-8** | 5.4e-6° | 2 TN aux only |
| B (after full sync) | wind_replace_t0 | **3.118e-8** | 5.4e-6° | **regression check — stays clean** |
| C | full_t0 | 1.545e-5 | 4.06e-4° | 30 aux (28 DSO + 2 TN) |
| C | full_peakres | 1.609e-5 | 5.38e-4° | 169-attr op-point retarget |

Layer separation now correct: wind_replace-alone = 2 aux terminals
(`AUX_TN_bus41/42`); full = 30 (28 DSO + 2 TN). Full sync creates 728
objects, all in the `full` stage. `full_t0` dry-run idempotent (all zeros).
`tests/pf` + `tests/export`: 30 passed, 2 skipped.

**Key regression guard for the future:** after any `full` sync, re-run the
`wind_replace` parity. Gate B breaking is the signature of a recording-stage
mis-target. Model left restored to `full_t0`.
