# Optional shunt persistence and STS support

Timestamp: 2026-10-20, 10:05 CEST. Work started in an isolated snapshot on
2026-10-19 and continued on 2026-10-20.

Reason: the author requested refinement and tests of persistent shunt demand,
assistance when the STS does not track its reference, and reserve restoration.
The author chose reserve action near capability limits. The author also asked
whether the tuning campaign must be repeated.

Changes:

- Added opt-in confirmation and cancellation of pending demand in
  `controller/shunt_integrator.py`. Existing configurations retain the
  original accumulation rule.
- Added measured-feedback tracking and near-limit reserve costs in
  `controller/shunt_support.py`, with freshness, reference-age, direction
  and full-step improvement checks.
- Added the paired delegated-request and shunt-offset transfer through
  `TSOController.transfer_pcc_request_to_shunt`. Support commits keep the total
  STS reference fixed. The runner propagates the bank effect to all interfaces
  of the affected STS.
- Added configuration fields, early checks of unsupported controller modes,
  actual-drive/reason/commit diagnostics, behavioural tests, AC directional
  tests and `experiments/shunt_refinement_check.py`.

Validation: 73 focused tests pass. Four one-hour AC comparisons at the current
runner parameter set completed. They show similar tracking and no STS support
commit within that horizon. Separate reduced-loop tests verify assistance and
reserve relief. Separate AC tests verify MSC/MSR action direction. Additional
comparison results at the October selection are recorded in the report.

The changes are experimental. Default controllers, selected weights and study
configuration files are unchanged. No full tuning campaign was launched.
Tracking and reserve support use separate supervisory costs, which must be
stated if this mode is adopted in the thesis.

The detailed design, exact trial parameters, limitations and tuning reuse/rerun
assessment are in
`docs/tuning/2026-10-20_shunt_refinement_and_rerun_scope.md`.
The implementation was prepared locally and applied only after checking source
hashes to preserve pre-existing edits.
