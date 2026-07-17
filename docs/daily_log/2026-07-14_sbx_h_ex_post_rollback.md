# SBX-H rollback to ex-post remuneration

**Timestamp:** 2026-07-14 (Europe/Berlin)  
**Reason:** User-approved rollback after the automatic restoration candidate
failed the predefined acceleration gate.

## Architectural result

The active automatic request, temporary voltage command, attributed runtime
settlement, automatic validation module, and automatic/planned-support
experiment variants were removed. The accepted active mechanism is:

1. planning-phase terminal-voltage schedules define the contractual voltage
   reference;
2. operation uses no real-time horizontal request or temporary voltage
   command;
3. useful physical reactive support is determined and remunerated ex post
   relative to the scheduled-voltage baseline;
4. for the current experiments, every area has the constant planning
   reference 1.0300 pu for the full horizon.

The generic contract schedule composer remains available for later genuine
planning studies, but experiment 014 sets `sbx_support_intervals = None` and
does not vary the 1.03 pu references.

## Code structure changed

- Removed `sbx_h/runtime_support.py` and `sbx_h/validation.py` and their tests.
- Removed runtime-request configuration, adapter integration, settlement
  attribution, plotting, and validation-report surfaces.
- Reworked `experiments/archived/014_SBX_SINGLE_DEMO.py` into one constant-
  1.03-pu ex-post demonstration.
- Replaced the rejected runtime-support specification with
  `docs/architecture/SBX_H_EX_POST_SETTLEMENT_SPEC.md`.

## Verification

- `python -m pytest tests/sbx_h -q`: 50 passed in 26.60 s.
- An 8 min headless experiment-014 smoke run completed 24 plant steps; all
  TSO solves were accepted. The written report records schedule source
  `controller_intent`, constant 1.0300 pu, planning changes disabled, and
  real-time requests/commands absent.
- Smoke artefacts:
  `results/014_SBX_H_DEMO/rollback_ex_post_smoke/`.
- The short smoke horizon ended before a settlement window was emitted;
  retained regression tests cover ex-post state, sign, and payment logic.

## Preserved modelling assumptions

- Controllers act through their cached sensitivities/models and do not see
  the plant directly.
- Existing Layer-1 actuators and constraints are unchanged: AVR/DER reactive
  setpoints, OLTCs, and shunt compensation subject to their modelled limits.
- Controlled outputs remain internal/terminal voltages and reactive boundary
  flows; SBX-H changes accounting and planning references, not plant physics.
