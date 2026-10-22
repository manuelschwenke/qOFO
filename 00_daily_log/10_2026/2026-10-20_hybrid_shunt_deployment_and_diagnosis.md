# Hybrid shunt dispatch: deployment and refusal diagnosis

Timestamp: 2026-10-20, 16:05 CEST.

Reason: the hybrid design of the afternoon existed only in an isolated snapshot
under `.../latex_diss_ms/tmp/shunt_hybrid_20261020/`, prepared for a
hash-checked deployment that had never been applied. The recorded comparisons
showed no benefit, and the reason why five slow-support proposals were refused
was unresolved.

Changes:

- Applied the prepared deployment: 8 modified and 8 new files. Every `before`
  hash matched the live file, so no concurrent edit was overwritten. Backups of
  the replaced files remain in the snapshot's `originals/`.
- Filled the pending results section of
  `docs/tuning/2026-10-20_hybrid_shunt_design_and_tests.md` from the recorded
  runs, and added the commit-sequence and refusal diagnoses below.
- No control law was changed in this session. Defaults, selected parameter sets
  and study configurations remain untouched; hybrid mode stays opt-in.

Method: the five refusals were diagnosed by re-running only the `slow` case with
the instrumented solver, which records a per-candidate rejection reason together
with `max_voltage_error_increase_pu` and `max_output_violation_increase`. The
earlier records predate those fields. A second run repeated the case with a
deliberately wide 0.02 pu allowance to separate an over-tight guard from a
genuine trade-off.

Verification: 198 tests pass in this checkout with the one pre-existing skip,
in 46.87 s, including the expected infeasible-or-unbounded warning from the
deliberate fallback test. The re-run with the default allowance reproduces the
snapshot's `slow` metrics exactly.

Findings:

- All five refusals are `voltage_tracking` at 0.00540–0.00577 pu, that is 5.4 to
  5.8 times the 0.001 pu allowance, with zero output-limit violation. The
  proposals were feasible; a 25 Mvar step simply moves the worst observed bus by
  about the size of the TS mean spatial tracking error.
- With a 0.02 pu allowance one commit is admitted at 2880 s on zone 3 bank 6
  (MSC). Interface-Q RMSE improves 4.36584 → 4.22566 Mvar and the absolute-error
  p95 11.72540 → 10.75032 Mvar. DER utilisation p95 rises 1.1%. The hourly TS and
  STS voltage ranges are unchanged to all printed digits. The slow path is
  therefore functional and was gated off, not idle.
- The ten regular-path commits are all engagements, one per TS instant from
  t = 20 s, eight minutes before the load step, and include every bank of
  zone 2, MSC and MSR alike. With `g_v = 40000` dominating the objective and a
  movement cost of 5, every step is profitable: a ratchet, not a regulating
  action.
- The `sts_reserve` memory never became non-zero in any run, while aggregate DER
  utilisation p95 reached 0.498 in the regular variant.

Open: whether to derive `g_w_tso_shunt` from the recorded gradient distribution
first, or to replace the pu voltage allowance by a shared-objective comparison
for slow candidates. Both are described in the design document. The diagnostic
runs are stored under `results/shunt_hybrid_diag_20261020/`.
