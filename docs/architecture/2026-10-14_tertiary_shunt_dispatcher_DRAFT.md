# DRAFT — one switched-shunt dispatcher per tertiary (MSC + MSR as one device)

**Status:** ON HOLD (2026-10-15). The MSC now sits on the second coupler's tertiary (`docs/daily_log/10_2026/2026-10-15_msc_on_second_coupler_tertiary.md`), so each tertiary carries one device and this per-tertiary dispatcher reduces to the existing per-bank integrator. Author decisions recorded for a possible per-DSO version: Q1 legacy switch yes, Q2 keep `x`, Q3 shared dwell/budget per group, Q4 decide g_w from the 0.03 vs 0.06 rerun. Revisit only if the placement validation shows pairing or both-on across the two tertiaries; the group would then be MSC + MSR of one DSO, with per-device pressures, since `h_C = −h_R` no longer holds exactly.

**Original status:** draft for author review, 2026-10-14. Nothing implemented.
**Evidence:** `docs/daily_log/10_2026/2026-10-14_shunt_gw_and_msc_msr_sibling_cycle.md`.

## 1. Problem (established, measured)

Each DSO tertiary carries one MSC (`N_C` × `q_C`) and one MSR (`N_R` × `q_R`), currently 2 × 25 Mvar each, on the **same bus**. Each is dispatched by its own `ShuntBank` integrator. Their pressures are exact negatives (`h_C = −h_R`), but clamp, hysteresis, dwell and budget act per bank. Measured on W4/W5 (12 h, evented) and the 6 h `make_config` scenario:

| defect | evidence |
|---|---|
| D1 capacitor and reactor energised together (net ≈ 0, both worn) | W4: 309 tertiary-min (g_w 0.03), 423 (0.06); 6 h: 51 |
| D2 the pair moves 50 Mvar in one instant (one releases, the other engages) | W5: 96 % of commits in the same instant as the sibling, 54 such pairs |
| D3 a reversal is executed by the sibling while the first bank is in dwell | W4: 24, W5: 40 sibling reversals, invisible to the per-bank metric |

D2/D3 produce a limit cycle at the dwell rate (median reversal gap = 30 min). `shunt_int_g_w` slows it but does not remove it.

## 2. Goals and non-goals

**Goals**
- G1 Never both devices energised at one tertiary: `min(ℓ_C, ℓ_R) = 0` at all times.
- G2 At most one physical step (one device, one level) per tertiary per TS iteration.
- G3 Dwell, daily budget, hysteresis and clamp act on the **tertiary**, so a reversal cannot be routed through the sibling.
- G4 The integrator law is unchanged: `Δx = −g_H / (2 g_w)`, clamp one step either side of the committed level, round-to-nearest with hysteresis `δ`, `x` carried across a commit.
- G5 Unchanged downstream: record format of `zone_tso_shunt_states` (per device, MSC = −level, MSR = +level, same order), and the runner's per-commit side effects (plant toggle, interface feed-forward, rank-1 SMW refresh of TSO and DSO models, `ShuntDisturbanceMessage`).

**Non-goals**
- No change of the update law. The virtual-effect correction `g̃ = g + κ (x − Q(n))` (option D) stays open.
- No new tuning weights, no MIQP change, no network-build change.

## 3. Device model

Net signed level, inductive positive (matches the recorder convention):

    n ∈ {−N_C, …, 0, …, +N_R},   ℓ_C = max(−n, 0),   ℓ_R = max(n, 0)

Net nameplate injection (load convention, inductive positive):

    Q(n) = q_R · n   for n ≥ 0,      Q(n) = q_C · n   for n < 0

Step sizes to the neighbours:

    q_up(n) = q_R if n ≥ 0 else q_C          (n → n+1)
    q_dn(n) = q_C if n ≤ 0 else q_R          (n → n−1)

**Pressure.** One projected gradient per tertiary along the inductive column:

    g = h_Rᵀ ∇_y f,   h_R = ∂y/∂Q_ind   (voltage, Q_PCC and SG-reserve parts, rescaled by (V_meas/V_0)²)

Computed once per tertiary with the MSR sign. The MSC pressure is `−g`, so no information is lost and half of the sensitivity evaluations are saved.

**Update and commit** (per TS iteration):

    x ← clip( x − g / (2 g_w),  Q(n) − q_dn(n),  Q(n) + q_up(n) )
    up   if  x ≥ Q(n) + q_up(n)/2 + δ   and  n < N_R
    down if  x ≤ Q(n) − q_dn(n)/2 − δ   and  n > −N_C

followed by the shared guards: dwell since the tertiary's last commit, daily budget of the tertiary, overshoot guard `v_meas + h_v,R · (±q) ∈ [y_min, y_max]`.

**Physical action** — exactly one device, one step:

| move | n before | action | emitted `ShuntCommit` |
|---|---|---|---|
| up (more inductive) | n < 0 | release one MSC step | kind MSC, direction −1 |
| up | n ≥ 0 | engage one MSR step | kind MSR, direction +1 |
| down (more capacitive) | n > 0 | release one MSR step | kind MSR, direction −1 |
| down | n ≤ 0 | engage one MSC step | kind MSC, direction +1 |

Release always precedes engage, which gives G1 by construction. `ShuntCommit` keeps its fields and device-convention `direction`, so the runner's feed-forward `dQ_itf = sign(kind) · direction · ∂Q_itf/∂Q · r_c` and SMW refresh are untouched.

**Special case.** A tertiary with one device only (`N_C = 0` or `N_R = 0`) reduces exactly to today's `ShuntBank`. That is the regression test.

## 4. Planned code changes

| File | Change |
|---|---|
| `controller/shunt_integrator.py` | New `TertiaryShuntGroup` built from the existing `ShuntBankConfig`s of one tertiary (validates: same bus, interface, DSO; at most one MSC and one MSR; equal `g_w`, `δ`, dwell, budget, band). State `n`, `x`, shared dwell/budget. `step(grad_g_ind, v_meas, h_v_ind, t_now) -> Optional[ShuntCommit]`. `ShuntIntegrator.from_configs(configs, per_tertiary=True)` groups by bus, preserving the build order; `update()` takes one input per group; `banks` stays a flattened per-device view (levels kept in sync) for the recorder. `ShuntBank` and the per-bank path stay as they are. |
| `configs/config.py` | `shunt_int_per_tertiary: bool = True`. `False` reproduces runs before the change (pattern of `shunt_int_v_rescale`). Structural, so a config field, not a parameter-set field. |
| `experiments/runners/multi_tso_dso.py` | Setup (~l. 3257): pass `per_tertiary`. Gradient loop (~l. 4192): iterate dispatch units; for a group compute `g`, `h_v` once with sign +1 (MSR). Commit block and recorder unchanged. Verbose print names the net level. |
| `tuning/stage2_search.py::shunt_summary` | Add `shunt_net_reversals` (per tertiary, net-level direction flips ≤ 2 h) and `shunt_both_on_s`. Analysis only; makes old (per-bank) and new runs comparable. Existing keys unchanged. |
| `tests/test_shunt_integrator.py` | New tests (below). |

Estimated size: ~200 lines integrator, ~40 runner, ~150 tests.

## 5. Tests

1. Single-device group reproduces `ShuntBank` trajectory, commit for commit (regression).
2. Invariant `min(ℓ_C, ℓ_R) = 0` under 10⁴ random pressure sequences, with and without dwell.
3. At most one commit per group per `update()`.
4. Crossing zero: from `n = +1` under sustained capacitive pressure the MSR is released first, the MSC engaged only after the dwell.
5. Shared dwell: a reversal within `t_dwell_s` is blocked whichever device would execute it.
6. Feed-forward sign: emitted `(kind, direction)` gives the same `dQ_itf` sign as the net move.
7. Grouping validation: mismatched bus / `g_w` / two MSCs on one bus raise `ValueError`.
8. Recorder order and sign convention unchanged (`[−ℓ_C, +ℓ_R]` per tertiary).
9. Existing `test_shunt_integrator*.py`, `test_shunt_voltage_rescaling.py`, `test_tso_tertiary_shunt.py` still pass.

## 6. Validation plan

Re-run W4, W5 (12 h, evented, DSO_3 ×2) and the 6 h `make_config` scenario with `shunt_int_per_tertiary=True`, at `shunt_int_g_w` 0.03 **and** 0.06. Today's runs serve as the `False` reference. Report commits, per-bank and net reversals, both-on time (must be 0), e_v(TS), e_v(DS), e_q, v_worst. ~45 min wall for 6 parallel units.

## 7. Decisions for the reviewer

- **Q1** Keep a legacy switch `shunt_int_per_tertiary` (default `True`)? *Recommended: yes.*
- **Q2** After a commit keep `x` where it is (today's law), or re-centre on `Q(n)`? *Recommended: keep (G4); re-centring is a law change.*
- **Q3** Daily budget per tertiary or per device? *Recommended: per tertiary (G3); it is effectively off at 1000 anyway.*
- **Q4** Keep `shunt_int_g_w = 0.06` in the script after the fix? *Recommended: decide from the 0.03 vs 0.06 rerun; part of the reason for 0.06 may disappear.*
- **Q5** Thesis Ch 4 (shunt dispatcher) must describe the per-tertiary formulation. Draft the text change as well?

## 8. Expected effect and residual risk

- **Hypothesis:** D1–D3 vanish by construction; the paired 50 Mvar swings (96 % of W5 commits) go with them.
- **Not addressed:** a single integrator can still cycle between two adjacent net levels when the continuous optimum lies between them (quantised-integral limit cycle), now at most one 25 Mvar step per dwell. That is option D.
- All Part III shunt results so far used per-bank dispatch (same caveat as the 2026-10-14 voltage rescaling).
