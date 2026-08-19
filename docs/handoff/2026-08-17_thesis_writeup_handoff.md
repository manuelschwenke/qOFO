# Handoff — writing the weight-selection campaign into the dissertation

Everything the thesis needs from the 0815 campaign. Every number is measured and
final; nothing is projected, and nothing is carried over from an earlier
campaign.

Source of record: `results/tuning_mc/campaign_0815/FINAL/` (regenerate with
`python -m tuning_mc.export_final`). Narrative and derivations:
`docs/daily_log/08_2026/2026-08-15_campaign_0815_tiered_bank_and_recalibration.md`.

---

## 1 — What to do with the two files

**Appendix D.** `docs/AppendixD_BO_Hyperparams.FILLED.tex` is the author's own
Appendix D with every campaign `[TBD]` filled (88 lines changed, structure and
labels untouched). Copy its contents over
`Appendices/AppendixD_BO_Hyperparams.tex`. Four `[TBD]`s remain, all prose in
comment blocks — lines 26, 28, 45 and 179 read oddly now that the tables they
describe are filled, and need a manual edit.

**Chapter 9, §9.3.** Carries no numbers by design and needs none. It needs
**one correction**, in the comment block at line ~421, which currently states:

> the 2026-08-14 numbers ... measure a block bound on the cached model, not
> lambda_max(B) ... so carrying them across would be a category error

**That is false for this configuration**, and as written it blocks the
campaign's numbers from being used at all. Proof, verified numerically:

* With `X = Q^(1/2) H G_w^(-1/2)`, the chapter's
  `B = Q^(1/2) H G_w^(-1) H^T Q^(1/2) = X X^T`, while the coordinator forms
  `M_ii = G_w^(-1/2) H^T Q H G_w^(-1/2) = X^T X`.
  `X X^T` and `X^T X` share their non-zero spectra.
* Measured on the real matrices at the selected point:

  | zone | lambda_max(M_ii) | lambda_max(B) | rel. diff | coupling sum |
  |---|---|---|---|---|
  | 1 | 0.682702817 | 0.682702817 | 4.9e-16 | 0.000000 |
  | 2 | 1.379997178 | 1.379997178 | 0.0 | 0.000000 |
  | 3 | 1.027761397 | 1.027761397 | 2.2e-16 | 0.000000 |

* The coupling term is **identically zero**, because `local_sensitivities_tso`
  zeroes every off-diagonal `H_ij` before the criterion is formed
  (`experiments/runners/multi_tso_dso.py:2473`). So
  `zone_contraction_lhs = lambda_max(M_ii) + 0 = lambda_max(B)`, and the
  campaign's `rho_emp_p95` **is** `lambda_max(B)`, measured along the trajectory
  as a p95 over windows and control areas.

The block bound and `lambda_max(B)` coincide here precisely because the
architecture zeroes the coupling. The general caution in that comment is correct
and worth keeping as a caution; the specific claim that these numbers measure
something else is not.

---

## 2 — The result

```
v_e 0.018 pu   lambda_TS 0.371   lambda_DS 1.6   tau 1.0
engage_ds 0.025 pu   r_v 1.5
```

| weight | selected | in service |
|---|---|---|
| `g_w_der` | 5.83429 | 20 |
| `g_w_pcc` | 22.1478 | 60 |
| `g_w_dso_der` | 617.152 | 800 |
| `g_w_tso_oltc` | 4740.10 | 5000 |
| `g_w_dso_oltc` | 183.112 | 150 |
| `dso_g_v` | 1.5e5 | 1e5 |

Per-area spreads, since the controllers receive per-area sets: `g_w_der`
0.937–30.57, `g_w_pcc` 9.998–47.55, `g_w_dso_der` 473.5–929.7, `g_w_tso_oltc`
1497–10870, `g_w_dso_oltc` 128.3–224.2.

Pinned as the numeraire and not designed: `g_w_gen` 1e9, `g_v` 1e7, `g_q` 250,
`tso_g_q_pcc` 0, `shunt_int_g_w` 100. `precondition_g_w: false` — the
preconditioner must stay off or it overwrites the designed weights.

---

## 3 — The screen and its margins

```
lambda_max(B) < 2                      isolated symmetric per-area block  [CITE-1]
lambda_max(B) <= 0.75 * 2 = 1.5        25 % reservation, a declared design choice
select at     <= 1.5 / 1.035 = 1.449   finite-bank allowance, eps_op = 0.035
verify        <= 1.5                   on the confirmation bank
```

`eps_op = 0.035` is the **worst** of six measured design-to-confirmation shifts
(+2.94, +3.11, +3.24, +3.28, +3.29, +3.49 %), corroborated independently by
resampling all C(18,12) twelve-window banks from the pooled windows (3.1 %).
The worst rather than the mean, because a candidate selected at 1.4554 under a
3 % allowance measured 1.5031 out of sample and had to be rejected.

**State the screen as a screen.** It is the exact convergence condition of the
*isolated, symmetric, unconstrained, continuous* per-area loop, used to bound the
parameter region before simulation. Stability of the constrained, integer,
multi-rate cascade is demonstrated empirically, not certified by it. Supporting
figure for the limitation sentence: re-evaluating the same locally-designed loop
with the off-diagonal blocks retained raises the contraction by a factor of
**1.24 to 1.29** across twelve measurements
(`00_method/coupling_check.json`), stable to ±0.5 % across operating points
spanning the profile year.

---

## 4 — The two design equations

```
g_w_oltc(v_e) = 318846 * v_e - 999                  max residual 0.33 (0.005 %)
lambda_max(B) = 4104 / g_w_oltc(v_e) + 1.5846 * lambda_TS
                                                     max residual 0.0142
```

The second is fitted over 47 raw measured points with `v_e >= 0.015 pu` and
`lambda_max(B) <= 1.6`. **State that validity range**: including points far
outside it — down to `v_e` 0.005 pu, where `lambda_max(B)` reaches 7.7 — inflates
the residual to 0.16.

Admissible region at `lambda_bar = 1.449`:

| `v_e` [pu] | `g_w_oltc` | `lambda_TS` admissible |
|---|---|---|
| 0.015 | 3783 | <= 0.230 |
| 0.017 | 4421 | <= 0.329 |
| 0.018 | 4740 | <= 0.368 |
| 0.020 | 5378 | <= 0.433 |
| 0.025 | 6972 | <= 0.543 |
| 0.030 | 8566 | <= 0.612 |

Below `v_e` ≈ 0.012 pu the floor alone exceeds the ceiling, so no continuous
weight is admissible. That is a lower bound from the contraction criterion, not
from the plant corridor — worth stating, because it is a stronger and more
defensible bound than the one currently given.

---

## 5 — Measured tables

### 5.1 Commit threshold, on the constraint boundary

`lambda_DS` 1.6, `tau` 1.0, `engage_ds` 0.025, `r_v` 1.5. Every row spends the
same stability budget, so the comparison is on cost alone.

| `v_e` | `lambda_TS` | rho | f_ts |
|---|---|---|---|
| 0.016 | 0.288 | 1.4506 | 1.24498 |
| 0.017 | 0.330 | 1.4453 | 1.24240 |
| **0.018** | **0.371** | **1.4480** | **1.23905** |
| 0.019 | 0.407 | 1.4509 | 1.24909 |
| 0.020 | 0.437 | 1.4508 | 1.25069 |

Interior minimum at 0.018, both neighbours worse.

### 5.2 Subordinate gain at the selected point

| `lambda_DS` | f_q | f_ts |
|---|---|---|
| 1.4 | 0.07987 | 1.23987 |
| **1.6** | **0.07710** | 1.23905 |
| 1.8 | 0.08672 | 1.23916 |

Interior minimum. `rho` is **bit-identical** across the whole `lambda_DS` range,
and in the wider sweep across a 12.7x change in `g_w_dso_der`. That invariance
is what licenses fixing the two gains in sequence rather than jointly, and it
was confirmed at four different `lambda_TS`.

### 5.3 Subordinate trade at the selected point

`f_ts` varies **0.41 %** over the entire DSO parameter space while `f_ds` varies
by a factor of 1.9. The DSO coordinates are invisible to the supervisory
objective and set only the `f_q` / `f_ds` trade.

| `engage_ds` | `r_v` | f_ts | f_q | f_ds | worst gap |
|---|---|---|---|---|---|
| **0.025** | **1.5** | 1.23905 | 0.07710 | 0.22315 | **12.99 %** |
| 0.0175 | 1.0 | 1.24034 | 0.07925 | 0.22569 | 14.27 % |
| 0.035 | 2.5 | 1.24244 | 0.08502 | 0.19750 | 19.00 % |
| 0.050 | 2.5 | 1.24190 | 0.08130 | 0.26950 | 36.46 % |
| 0.035 | 1.5 | 1.23974 | 0.07177 | 0.29250 | 48.11 % |
| 0.025 | 1.0 | 1.23966 | 0.07442 | 0.29841 | 51.10 % |
| 0.050 | 1.5 | 1.23725 | 0.07145 | 0.38200 | 93.42 % |

Selection rule, worth stating explicitly: the three costs have no exchange rate,
so the point minimising the **largest relative shortfall** from the best
achievable value on any criterion is taken (a compromise / Chebyshev choice).

### 5.4 Confirmation — 9 windows, even ISO weeks, never tuned on

| | f_ts | f_q | f_ds | lambda_max(B) |
|---|---|---|---|---|
| baseline, design bank | 1.25502 | 0.08106 | 0.29252 | 1.4743 |
| selected, design bank | 1.23905 | 0.07710 | 0.22315 | 1.4480 |
| baseline, confirmation | 1.77045 | 0.07755 | 0.33553 | **1.5201 — over** |
| selected, confirmation | 1.73986 | 0.08332 | 0.23132 | **1.4949 — ok** |

Design-bank deltas against the baseline: `f_ts` −1.27 %, `f_q` −4.9 %, `f_ds`
−23.7 %. Confirmation `f_ts` −1.7 %. Bank difficulty ratio 1.40.

**The baseline is inadmissible out of sample and the selected point is not** —
the single most useful sentence for the result section.

### 5.5 Switching wear — 4 x 12 h, per transformer

| window class | operations per transformer per day | share of activity |
|---|---|---|
| quiet | 6 | 14 % |
| contingency-carrying | 42 | 86 % |
| all windows | 42 | 100 % |

The 30/day budget is a **routine-day** figure (author's statement, 2026-08-17),
so the selected point uses 20 % of it. Report both rows: 94 % of tap activity
falls in windows carrying an injected contingency, and a 90-minute window cannot
resolve the budget at all — one tap operation there is 0.667/h.

The constraint is carried by **one transformer**: `DSO_4|trafo_10` at 36–42/day
while seventeen of nineteen sit far below. A per-area DSO tap price
(`dso_g_w_class`, already supported by the config, never used in this campaign)
is the obvious follow-up.

---

## 6 — Three findings worth putting in the text

**The commit threshold sets the contraction floor.** `rho <= 1` is reachable —
0.8829 at `v_e` = 0.030 pu. Any statement that the integer columns impose an
irreducible contraction must name the commit threshold it was measured at. The
floor of ≈1.09 is a property of `v_e` = 0.015 pu, not of the plant.

**A coordinate-wise search will not find the design point.** A compass search
over all six coordinates, run twice, moved only `r_v`. Polling one coordinate at
a time cannot find a move that needs two together: raising `v_e` alone worsens
`f_ts`, raising `lambda_TS` alone breaches the budget, raising both improves
`f_ts` and `f_q` at once. The coordinates a local search leaves alone are
exactly the ones that need a dedicated sweep.

**`tau` was settled by the carry-forward rule, not by the design bank.** The
design bank prefers `tau = 0.70` (worst gap 0.34 % against 1.25 %); the
confirmation bank prefers `tau = 1.0` (2.78 % against 3.30 %). The `f_q`
advantage of `tau = 0.70` **reverses sign** out of sample, so it is not carried
forward. A clean illustration of the rule doing real work.

One correction for the method description: `tau` sits on the **constraint
surface** together with `v_e` and `lambda_TS` — it rotates `g_w_der` against
`g_w_pcc` and therefore moves `lambda_max(B)`. The three supervisory
coordinates must be optimised jointly on `rho = lambda_bar`; only `lambda_DS`
and the DSO pair are genuine one-dimensional scans, and that is verified rather
than assumed (`rho` bit-identical across their whole range).

---

## 7 — Citations still needed

| tag | for |
|---|---|
| CITE-1 | `0 < lambda_r(B) < 2` — Boyd & Vandenberghe §9.3.1, already cited in Ch. 9 |
| CITE-2 | the curvature / preconditioning rule of Ch. 8 |
| CITE-3 | VDE-AR-N-4120 capability diagram — already `VDARN4120.2018` |
| CITE-4 | **the tap-changer maintenance budget** — an operator statement with no traceable source yet, and its routine-day scope is what makes the wear verdict a pass |

CITE-4 is the load-bearing one. If the budget turns out to be an envelope
covering contingency days, the selected point uses 140 % of it and the DSO trade
has to be re-picked toward less tapping.

---

## 8 — Scope limits to state

One benchmark (`rural_700`), one boundary model (Thevenin), one dynamic
parameter set, and a load model without recovery dynamics. Both design equations
and the coupling factor are properties of the network, its area partition and
the boundary model, and must be re-measured if any of those changes.

18.6 % of the profile year has exactly zero DER reactive capability. The scenario
bank samples all three capability strata in proportion to the year
(none 17 %, partial 33 %, full 50 % against 18.6 / 33.7 / 47.7), and costs are
reported per stratum as well as in aggregate — in the zero-capability stratum
`tau`, `lambda_DS` and `r_v` have nothing to allocate, and the aggregate would
otherwise average a signal with a constant.
