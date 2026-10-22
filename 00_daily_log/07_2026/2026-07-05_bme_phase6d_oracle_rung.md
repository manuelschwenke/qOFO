# BME Phase 6d — oracle rung (d): single-zone BME oracle

**Date:** 2026-07-05 (work begun 2026-07-03, interrupted, resumed)
**Author:** Manuel Schwenke (with research assistant)
**Scope:** Phase 6 item (4): the ladder's oracle rung (DECISION D8).
Manuel chose the "both" interpretation (2026-07-03): the single-zone BME
oracle now, the V5-style full-set Φ oracle as the optional later bound.

## Key design idea

The oracle is a CONFIGURATION, not a new controller: collapsing the zone
partition to one zone (union of the fixed 3-area TN bus sets) under
`coordination_mode="bme"` yields the centralised per-step OFO-MIQP with
global Φ by construction — no ties, empty boundary registry, port-frozen
operators degenerate to total-response operators, hence
g_bme = dΦ/du exactly. The single-area identity test (spec §3.5 test 1)
is therefore the oracle's correctness proof; the actuator universe, the
DSO cascade, the solver, the step logic and the D6 hygiene are identical
to the distributed bme rung — the gap-to-oracle isolates precisely the
value of decomposition/communication.

## What was changed

* `configs/multi_tso_config.py`: `single_zone_partition` flag (docstring
  records the D8 interpretation and the V5-Φ deferral).
* `experiments/runners/multi_tso_dso.py`: single-zone partition branch
  (union of the fixed 3-area lists — NOT a fresh partition, so the bus
  universe is bitwise-identical); `_hv_zone()` remap for HV sub-network
  grouping, PCC/DSO maps and MIQP tertiary-shunt ownership (without the
  remap the shunt banks silently vanish from the oracle's control
  vector); degenerate BME path when |zones| = 1 (no CoordinationBus, no
  receivers, no notice publish/consume — guarded, the two-zone bus
  fail-fast stands for genuine multi-zone runs).
* `sensitivity/boundary_sensitivity.py`: an EMPTY boundary registry is
  now the accepted single-area degenerate mode (zero-row H_{b,i}); the
  "all boundary buses pinned" fail-fast for non-empty registries stands.
* `experiments/011_BME_LADDER.py`: rung "oracle" (bme-family config +
  `single_zone_partition=True`).

## Verification

* 37 tests green (`test_boundary_sensitivity`, `test_bme_gradient_identity`
  — incl. the single-area identity, `test_discrete_hygiene`).
* 15-min smoke: runs end-to-end; ledger shows the single-decision-maker
  signature (1 accepted commit, zero slot-blocks); figures regenerate
  over all five rungs automatically.
* Per-step cost ≈ 12× the distributed rung (one large MIQP + global
  Jacobian rebuild per tick) — acceptable for a reference rung.
* 120-min oracle run: results recorded in `docs/BME_STATUS.md` §6d
  (comparable to the §6c sweep table: none 60.65 MW, distributed bme at
  the D2-final pairing 57.38 MW last-hour losses).

## Reason

Spec §5 Phase 6 rung (d) / DECISION D8; interpretation resolved with
Manuel 2026-07-03 (single-zone now, V5-Φ later — recorded in
BME_STATUS.md §6d).
