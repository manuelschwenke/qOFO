# Banks held, local autonomous, or coordinated: a three-way comparison

Date: 21 October 2026. One simulated hour, selected parameter set
`ch9_selected_pcc27_gqceil`, symmetric benchmark, real profiles, seed 20260717,
TS/STS periods 180/20 s, 25 Mvar banks with 2 levels and 1800 s dwell,
horizontal coordination disabled, a 150 Mvar reactive load connected at minute
10 and removed at minute 30. Results under `results/three_way_20261021/run1`.

| Arm | Switches | Interface Q RMSE [Mvar] | Absolute error p95 [Mvar] | DER utilisation p95 | TS V range [pu] | TS mean spatial V RMS [pu] | STS mean spatial V RMS [pu] |
|---|---:|---:|---:|---:|---|---:|---:|
| `off` — banks installed, budget zero | 0 | 1.3093 | **1.2103** | **0.2821** | 0.9790–1.0502 | 0.005838 | 0.008076 |
| `local` — autonomous, deadbanded local voltage | 2 | 1.4898 | 1.6440 | 0.3021 | 0.9790–1.0508 | 0.005772 | **0.007631** |
| `coord` — integer banks in the TS MIQP | 3 | **1.2971** | 1.3210 | 0.2954 | 0.9790–1.0502 | **0.005725** | 0.007904 |

Arms: `off` installs the banks and sets the switching budget to zero, so the
network is identical and only the switching is removed. `local` drives the
existing integrating dispatcher from the bank's own deadbanded bus voltage
(`shunt_int_drive='local_voltage'`, gain 4.2·10⁵ derived from a unit-gain probe,
0.01 pu deadband, 1.03 pu reference), which is the accumulated-deviation rule
of CIGRE CSE 026 §2.6 with the interface and reserve terms excluded. `coord`
makes the bank levels integer variables of the TS MIQP at
`g_w_tso_shunt = 20`, inside the gap of the recorded gradient distribution
(p90 = 9.77, p95 = 28.49), with both supervisory weights at zero.

## What it says

**Coordination is a small win, not the disaster the earlier runs suggested.**
Against `off` it improves TS mean spatial voltage RMS by 1.9% and interface-Q
RMSE by 0.9%, at the cost of 9% on the interface error p95 and 4.7% on DER
utilisation, with identical voltage ranges and three switches in the hour.

**The 54% interface degradation reported on 20–21 October does not transfer.**
That figure came from `ch9_selected_pcc27_gqceil_comp_q_off`, which differs from
the selected set in exactly two fields: the subordinate interface-Q objective
is switched off (`g_q` 0.4977 → 0) and the delegated request is frozen
(`g_w_pcc` 0.1643 → 1e8). With the cascade actually closed, coordinated bank
dispatch is roughly neutral. The earlier number was a property of the
diagnostic configuration, not of coordination, and the conclusion drawn from it
— that the shared objective under-prices the interface it delegates — is not
supported on the selected controller.

**Local autonomous control is the weakest arm on the interface.** It degrades
interface-Q RMSE by 13.8% and the p95 by 35.8% against `off`, for the best STS
spatial voltage RMS of the three (−5.5%). That is the expected signature of an
uncoordinated actuator: it serves its own bus and the subordinate tracking pays.
It is also operating close to its dead zone — the deviations reaching the banks'
own buses barely exceed 0.01 pu — so two switches in an hour is a thin basis.

**One `coord` commit is clearly spurious.** It fires at t = 20 s, eight minutes
before the disturbance, because the regular path commits on the first instant
its threshold is crossed. That is what `shunt_hybrid_regular_confirmations`
addresses, and it is the subject of the campaign below.

## Campaign: movement weight against confirmation count

`coord` only, same hour and parameter set, varying `g_w_tso_shunt` and
`shunt_hybrid_regular_confirmations`. Results under
`results/three_way_20261021/gw*_c*`.

| `g_w` | confirmations | Switches | Q RMSE | p95 | DER util. | TS vrms | STS vrms | commit times [s] |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| – (`off`) | – | 0 | 1.3093 | 1.2103 | 0.2821 | 0.005838 | 0.008076 | – |
| 20 | 1 | 3 | **1.2971** | 1.3210 | 0.2954 | **0.005725** | 0.007904 | 20, 720, 1800 |
| 20 | 2 | 0 | 1.3093 | 1.2103 | 0.2821 | 0.005838 | 0.008076 | – |
| 20 | 3 | 0 | 1.3093 | 1.2103 | 0.2821 | 0.005838 | 0.008076 | – |
| 30 | 2 | 0 | 1.3093 | 1.2103 | 0.2821 | 0.005838 | 0.008076 | – |
| **10** | **2** | **1** | 1.3162 | 1.2276 | **0.2748** | 0.005770 | **0.007709** | 1980 |

Two things follow, and they interact.

**At the derived threshold, confirmation removes every commit.** None of the
three steps at `g_w = 20` survives even two consecutive instants: the MIQP asks
once and, by the next instant, the continuous actuators have resolved the
deviation or the gradient has fallen below the threshold. By the author's own
criterion — commit only on a deviation *not recovered directly by the other
actuators* — all three were unnecessary, and the t = 20 s commit eight minutes
before the disturbance was only the most obvious of them. Raising the threshold
to 30 changes nothing, because the binding test is now persistence.

**A persistence test is what makes a lower threshold safe.** At `g_w = 10` the
gradient distribution admits eight bank-instants in the hour, which without
confirmation would ratchet. With two confirmations it yields **one** commit, at
1980 s, just after the load is removed — and that commit improves three metrics
against banks held: DER utilisation p95 by 2.6% (0.2748, the best of any arm),
TS mean spatial voltage RMS by 1.2% and STS mean spatial voltage RMS by 4.5%,
at a cost of 0.5% on interface-Q RMSE and 1.4% on its p95.

So the pair `g_w_tso_shunt = 10` with two confirmations is the configuration
that matches the specification: one switch per hour, on a sustained deviation,
improving both layers' voltage quality and the subordinate DER loading for
under 2% of interface tracking. The magnitude threshold alone could not reach
it — at one confirmation the same weight would switch on transients, and at a
weight high enough to avoid that it switches on nothing useful. The two
parameters are not independent knobs; the persistence test is what buys the
freedom to lower the threshold.

## Limits

One hour, one scenario, one network, one seed; quasi-static AC. The differences
between arms are 1–5% on most metrics, which is well inside what a different
window could reverse, so this ranks the arms on this hour and nothing more. The
`local` arm's gain and deadband are a first derivation, not a tuned pair, and
its reference is the zone setpoint rather than a per-bank target. No arm was
evaluated on the campaign's ranking metrics.
