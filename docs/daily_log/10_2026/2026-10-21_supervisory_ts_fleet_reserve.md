# Supervisory TS fleet reserve for the shunt dispatcher

Timestamp: 2026-10-21, 16:30 CEST.

Reason: the author proposed that the reserve objective drive **only** the
shunts, leaving the TS generators and TS DER on their voltage task. The 21
October probes had shown why: placed in the shared objective, `tso_g_res_sg`
moves every continuous actuator, helps only in a narrow window near 10³ and
breaks the TS voltage band above it (1.0610 pu at 10⁴). The author chose the
supervisory route over a distorted bank gradient, so the joint MIQP keeps a
single objective.

Changes:

- `controller/shunt_support.py`: added `TSFleetReserveState` and
  `ts_fleet_reserve_request`. The cost is `w · ||D_r0(r)||²` with
  `r = (Q_g − Q_mid,g)/Q_half,g` over the zone's synchronous machines and `D_r0`
  the outside-band operator, so it is zero while every machine stays inside the
  central part of its band. A whole physical bank step must reduce it. Unlike
  the subordinate costs it needs no freshness or reference-age test: these are
  the TS's own measurements at the same instant.
- `controller/tso_controller.py`: added `fleet_reserve_state`, which returns the
  machine state from the **same** capability source, band definition, half-width
  epsilon and out-of-service mask as the `Q_gen` block of
  `_output_objective_gradient`, so the supervisory cost cannot drift from the
  controller's own reserve term.
- `controller/shunt_hybrid.py`: a third memory kind `ts_reserve` alongside
  `sts_tracking` and `sts_reserve`, observed and proposed per bank. The
  candidate for it **does not** pin the delegated requests: leaving them free
  keeps the feedforward offset-only convention, whose documented purpose is that
  the subordinate controller does not counteract the switch. The subordinate
  kinds keep the paired transfer. Admission for all three is unchanged —
  the hard voltage corridor, no increase in a predicted output-limit violation,
  and the configured gate.
- `controller/shunt_integrator.py`: `ts_reserve` accepted as a commit reason.
- `configs/config.py`: `shunt_ts_reserve_weight` (0.0, off) and
  `shunt_ts_reserve_start` (0.8), with fail-fast validation and a guard that a
  positive weight requires hybrid dispatch. Wired through the runner and
  `experiments/shunt_hybrid_check.py` (`--ts-reserve-weight`,
  `--ts-reserve-start`).

Method: the supervisory cost is evaluated only for the bank decision. The
shared objective keeps `tso_g_res_sg = 0`, so the continuous actuators never
see the reserve term; a test pins that by requiring the regular solve to be
identical with the supervisory cost on and off.

Verification: 210 tests pass with the one pre-existing skip, in 158 s; the
focused hybrid file passes 39. Six new cases cover a commit when a machine
leaves its band, no request inside the band, invariance of the regular solve to
the supervisory weight, admission with the delegated control pinned (the
offset-only convention), and validation of a negative weight and an
out-of-range band start.

## Tuning outcome, 17:40 CEST

The weight could not be tuned, for a reason that is itself the result. Two
one-hour runs on the **selected** parameter set `ch9_selected_pcc27_gqceil`,
banks parked, both subordinate weights zero, shared-objective gate:

- At unit weight the fleet-reserve gradient is **zero at all 168 samples**.
- With `fleet_max_abs_r` added to the diagnostics, the worst machine reaches
  `|r| = 0.5126` in zone 2 and `0.3731` in zone 3, with medians 0.4148 and
  0.3393. The fleet never approaches its rails in this hour, so at
  `r0 = 0.8` no weight can make the path fire.
- Firing at all would need `r0 < 0.513`; firing regularly, below about 0.41.
  At those values the cost is no longer a near-limit guard but a mild centring
  pressure over much of the hour — the behaviour the author objected to, except
  that it would now act only on the banks and not on the generators. That is a
  design decision, not a tuning one.

## Scenario search, 19:50 CEST

The author chose to look for a scenario that genuinely exhausts the reserve
rather than lower `r0`. Three levers were tried; **all three failed**, and the
highest worst-machine band position seen in any run today is 0.5126.

| Scenario | zone 2 max \|r\| | zone 3 max \|r\| | TS V min | gradient |
|---|---:|---:|---:|---:|
| 150 Mvar transient (original) | 0.5126 | 0.3731 | 0.9790 | 0/168 |
| 300 Mvar sustained | 0.4343 | 0.3723 | 0.9358 | 0/168 |
| 600 Mvar sustained | 0.4343 | 0.3821 | **0.8606** | 0/168 |
| 1000 Mvar sustained | — | — | — | run failed |
| trip gen 2 (G4, 800 MVA, zone 3) | 0.4343 | 0.3076 | 1.0060 | 0/168 |
| trip gen 5 (G7, 700 MVA, zone 3) | 0.4343 | 0.3378 | 1.0166 | 0/168 |

* **Reactive loading does not load the machines.** `max |r|` stays flat or
  falls while the TS voltage minimum collapses to 0.8606 pu at 600 Mvar.
  Voltage is the binding resource here, not generator reactive margin, and
  `compute_gen_q_bounds` depends on V, so the capability box moves with the sag.
* **A generator trip makes it *better*, not worse.** Tripping either zone-3
  machine *lowers* that zone's worst position, to 0.308 and 0.338 against
  0.373 un-tripped, because the trip also sheds 464–530 MW to the slack and the
  flow redistributes. Interface tracking improved as well, to 0.398 and
  0.310 Mvar RMSE, and voltages rose to 1.006–1.049 pu. Both runs are clean,
  720 optimal subordinate solves each.
* Zone 2's worst position sits at 0.424–0.434 in **every** run regardless of the
  disturbance: its single machine, G3_bus31, is held at a steady reactive
  loading by its voltage duty.

The likely cause is resource abundance rather than machine size: the zone holds
four TS DER of 508–650 MVA in Q(V) mode, plus the subordinate systems, and the
controller uses them in preference to the machines. The remaining lever is
therefore a **resource-scarcity** scenario — make the TS DER unavailable so the
machines must carry the reactive duty. That is a physically meaningful stress
rather than a threshold fudge, unlike lowering `r0` to produce switching, but it
is a scenario design decision.

## Weight derived, mechanism verified, service not demonstrated — 21:30 CEST

The author proposed a **capacitive** disturbance, to drive the machines toward
their nearer under-excited rail (VDE V2 is asymmetric: −0.33 against +0.41 S_n)
and to raise rather than sag the voltage. It is the first lever that produced a
signal.

| q [Mvar] | commits | Q RMSE | TS V range | z2 max \|r\| | gradient |
|---:|---:|---:|---|---:|---:|
| −300 | 0 | 2.50 | 1.0166–1.1157 | 0.661 | 0/168 |
| −600 | 2 objective | 5.92 | 0.9873–**1.1879** | **1.048** | 48/168 |
| −900 | 0 | 7.20 | 1.0137–**1.2859** | 1.297 | 51/168 |

With non-zero gradients the weight follows analytically from
`w ≥ (0.5·q_step + δ)·2·g_w^mem / (N·|g₁|)`. At the −600 onset, median
`|g₁| = 0.00866` per unit weight and `g_w^mem = 0.06`: N = 2 → 156, **N = 3 →
104**, N = 5 → 62. **`shunt_ts_reserve_weight = 100`** was adopted — readiness
in three TS observations, 540 s, just above the 360 s persistence floor, and it
scales with severity, since the −900 gradients are three times larger.

**Verification at w = 100 on the −600 scenario: the path works end to end.**
Six fleet candidates, four refused as `shared_objective` and two accepted,
committing MSR steps at 2520 s and 3060 s in zone 2 — reactors engaging against
an over-voltage, the right device. The gate did real work: the refusals carried
`shared_objective_increase` of +6.2 to +6.7, the acceptances −44.9 and −14.6, so
at exchange 0 only steps that pay for themselves were admitted.

**But the full-candidate relief check exposes that the service is nearly
absent.** For every candidate the bank-only relief was 0.77–0.96 while the
relief of the complete candidate was 0.00025–0.00030 — the re-optimised ordinary
controls cancel about **99.97%** of it. The machines' reserve is restored by the
continuous actuators, not by the bank; a 25 Mvar step buys 0.03% of the reserve
cost. The two commits that passed the gate did so because they *improved the
shared objective*, with the fleet relief as a rounding error, so they are steps
the regular path would have wanted had its weight not been parked.

Without the check accepted from the external review on 21 October, the
acceptance test would have weighed a shared-objective cost of +6.7 against a
claimed benefit of 0.77, and any exchange weight above ~9 would have admitted
steps whose real relief was 0.0003. The check turned a 3000-fold overstatement
into the measured number, and it is the reason this conclusion is available at
all.

**The realistic low-wind window shows no scarcity.** At 2016-11-26 10:00, where
the whole 2198 MVA of TS wind parks sits in the VDE dead zone with zero
reactive capability, the worst machine reaches only `|r| = 0.215` (zone 2) and
`0.241` (zone 3) — the **lowest** of any run — with voltages 0.9741–1.0438 and
no fleet candidate at all. The reason is that capability and duty share a
driver: at `P/S_n = 0.01` the parks produce almost no active power either, so
the network is lightly loaded and there is little reactive duty to carry.
Starving the DER of capability also removes the need for it.

So: the supervisory mechanism is correct and verified in closed loop; it fires
only at 1.19 pu, outside any admissible corridor; and where it fires, the bank's
contribution to fleet reserve survives re-optimisation at the 0.03% level. The
service is **not demonstrated as useful on this test system**, and the two
reasons are structural rather than tuning: the VDE capability of 2198 MVA of
TS parks outweighs the machines wherever there is duty to carry, and where that
capability vanishes the duty vanishes with it.

Also recorded from these runs, on the selected parameter set rather than the
`comp_q_off` diagnostic one: interface-Q RMSE 1.3093 Mvar and absolute-error
p95 1.2103 Mvar, against 4.3658 and 11.7254. The selected controller tracks the
interface about three times better in RMSE and ten times better in p95, which
is the quantitative confirmation that every earlier shunt run was measured on a
crippled configuration.

Note: the experiment script's `--slow-weight` default overrides the parameter
set's `shunt_int_g_w`, so these runs used 0.01 rather than the selected 0.06.
Immaterial while the gradient is zero, but it must be set deliberately before
any tuned number is quoted.

A method description with the formulas and the MIQP integration is at
`docs/method/shunt_dispatch_method.md`, written for external review.

Original open note: `shunt_ts_reserve_weight` is **not** tuned. It scales the gradient that
drives the per-bank memory, so readiness after `N` TS periods needs
`|gradient| ≥ (0.5·q_step + delta)·2·shunt_int_g_w / N`. Since the gradient is
linear in the weight, one run at unit weight fixes it analytically. That run is
in progress on the **selected** parameter set `ch9_selected_pcc27_gqceil`, not
the `comp_q_off` diagnostic set, so the tuning is not done on a configuration
whose interface compensation is disabled.
