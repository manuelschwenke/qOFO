# Shunt controller refinement

Timestamp: 2026-10-20, 10:05 CEST.

Implemented the author's requested optional persistence, STS tracking assistance
and reserve action near capability limits. The method combines demand
confirmation, measured-feedback support costs and paired reference/offset
updates. Added behavioural and AC directional tests and a reproducible
comparison experiment. Existing study defaults and selected weights are
unchanged.

73 focused tests pass. Test results, assumptions and the required empirical
tuning rerun are documented in
`docs/tuning/2026-10-20_shunt_refinement_and_rerun_scope.md`.
The repository's chronological record is
`docs/daily_log/10_2026/2026-10-20_shunt_persistence_and_sts_support.md`.
