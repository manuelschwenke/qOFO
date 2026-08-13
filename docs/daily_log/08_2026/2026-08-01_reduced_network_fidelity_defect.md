# 2026-08-01 — The reduced sensitivity networks do not reproduce the plant

**Timestamp:** 2026-08-01
**Scope:** diagnosis only. New tool `tools/check_reduction_fidelity.py`.
**No production code was changed** — three attempted fixes were reverted, and
the reduction behaves exactly as before (verified bit-identical).

---

## 1. What was found

Every controller acts on cached sensitivities derived from a Ward-reduced
network (`sensitivity/network_reduction.py`). That reduction is only valid if
the reduced network, solved on its own, lands on the **same operating point**
as the combined network at the buses it retains. It does not.

Measured at `2016-01-05 08:00` (scenario `rural_700`, a run that is *in* the
completed dead-band study and converged normally):

| reduced net | buses | Ward stubs | max ΔV [pu] | mean ΔV [pu] |
|---|---|---|---|---|
| TSO zone 0 | 14 | 3 | 0.026019 | 0.005594 |
| TSO zone 1 | 35 | 13 | **0.000000** | 0.000000 |
| TSO zone 2 | 21 | 6 | 0.015127 | 0.006815 |
| DSO 1 | 16 | **0** | **0.286996** | 0.205789 |
| DSO 2 | 16 | **0** | **0.322923** | 0.217048 |
| DSO 3 | 16 | **0** | **0.340569** | 0.229379 |
| DSO 4 | 16 | **0** | **0.102194** | 0.028130 |

Reproduced identically at `2016-12-18 14:00` and `2016-01-15 03:00`, so it is
systematic and not a property of one operating point.

**The four DSO reduced networks are off by 0.10–0.36 pu.** A bus at 1.03 pu in
the plant is linearised near 0.70 pu. TSO zone 1 is exact; zones 0 and 2 are off
by 0.01–0.07 pu.

Consequence: every DSO controller in every run of this project has been acting
on sensitivities linearised about an operating point far from the plant's. This
is not confined to the dead-band study — it is the DSO layer's model of its own
network, and it is a candidate explanation for behaviour previously attributed
elsewhere.

## 2. Why this was not noticed

The reduction is guarded by convergence, not by fidelity. A reduced net with
stranded boundary flows converges perfectly happily — to a different operating
point. Nothing downstream compares it against the combined solution.

The related failure at `2016-05-01 16:00` (§4) is the same defect in its acute
form: there the reduced net has no solution at all, so the run aborts loudly
instead of proceeding with a wrong linearisation quietly.

## 3. The measurement

`tools/check_reduction_fidelity.py` wraps the reduction's own solve, captures
`res_bus` immediately before it (still the combined solution, carried through
the deepcopy), and compares against the solved reduced net. Static plant only;
no PowerFactory, so it can run beside a co-simulation.

```
python tools/check_reduction_fidelity.py                       # study windows
python tools/check_reduction_fidelity.py --window "2016-05-01 16:00"
```

Pass criterion: every retained bus within ~1e-3 pu. **Convergence is not the
test** — two of the three fixes below passed a convergence check while being
plainly wrong.

## 4. The acute case: 2016-05-01 16:00

Net infeed +2200 MW. The combined power flow converges (V 1.0014–1.1371) but
every run aborts with `LoadflowNotConverged` inside `build_tso_local_net`.

Established by direct test, all through the real runner path:

- **Not reactive shortage.** The ±1.0 pu DER capability override changes
  nothing, with or without the DSO_3 multiplier.
- **Not the DSO_3 multiplier.** ×2, ×1.75, ×1.5, ×1.25 and symmetric all fail.
- **Not dispatch or slack.** `distributed_slack=False` fails identically; the
  combined solution puts 0 MW on the ext_grid; curtailing the synchronous
  machines ×0.75 → ×0.0 leaves the *combined* flow converging throughout.
- **Not initialisation.** results / flat / dc / q-limits-off all diverge at
  16:00 and all converge at 16:30, so a solution that merely was hard to reach
  would have been found.

Zone 0's reduced net asks its promoted slack machine to swing from **+1000 MW
generation to −1760 MW absorption**: ~2.8 GW of boundary exchange is
unrepresented. The same zone carries the same defect at the *converging*
instant (2.2 GW) and merely stays solvable.

Feasibility boundary, empirically: **+1547 MW converges, +1670 MW does not**
(2016-05-01 16:30 vs 15:30). `2016-05-01 16:30` is therefore a usable
high-export window with no code change.

## 5. Three fixes attempted, all reverted

| attempt | idea | outcome |
|---|---|---|
| 1 | match each retained bus's **full-net injection** (`res_bus.p_mw`) | Wrong. By KCL that injection equals the flow into *all* branches; the reduced net has fewer, so it needs injection minus deleted-branch flows. Converged to a different operating point (V 1.0300–1.1046 vs 1.0014–1.1371) — the tell I initially missed. |
| 2 | one stub per **deleted branch**, carrying its cached flow | Right construction in principle, but broke `2016-05-01 16:30`, which had worked. |
| 3 | give the DSO boundary gens their cached `p_hv_mw` instead of `p_mw = 0` | Marginal (0.287 → 0.269 pu) and **made the TSO side worse** — zone 1 went from exact to 0.292 pu. |

All reverted; fidelity numbers verified bit-identical to the starting state.

The observation behind attempt 3 still looks sound and is worth revisiting: in
`build_dso_local_net` only the first primary bus becomes slack, the others are
PV gens created with `p_mw = 0.0`, so a multi-coupler DSO is forced to push its
entire real-power exchange through one transformer. That cannot match the
combined solution. It is evidently not the whole story, since fixing it alone
barely moved the error.

## 6. Cause found, and a partial fix

The DSO reductions were not imbalanced at all — the model is sound. The **power
flow was converging to the low-voltage root.**

Anatomy of one DSO reduced net at `2016-01-05 08:00` (16 buses, 3 couplers):

| bus | role | V cached | V solved | ΔV |
|---|---|---|---|---|
| 4, 6, 7 | PRIMARY (TN) | 1.0222 / 1.0200 / 1.0196 | identical | **0.00000** |
| 43–52 | MV + internal HV | 1.023 … 1.036 | 0.737 … 0.777 | ≈ 0.26 |
| 53–55 | LV / tertiary | 1.022 … 1.026 | **0.0000** | — |

The boundary is exact, the transformers are in service at `tap = 0.0`, no bus
is out of service, and the whole interior sits uniformly ~0.26 pu low with the
tertiaries collapsed to zero. That is a second, physically valid solution of the
same equations, not a modelling error — the multi-equilibrium behaviour already
documented in `docs/qss_rms_divergence_analysis.tex`, here reached by the
solver rather than by the plant.

Re-solving the identical net from different starts:

```
init=flat     V 0.0000..1.0222   max dV vs combined = 1.0259
init=dc       V 1.0004..1.0325   max dV vs combined = 0.0232
```

`build_dso_local_net` passed `init_sequence=(("flat", 100), ("dc", 200))`, and
`runpp_with_stored_jacobian` **stops at the first attempt that converges**. Flat
converges — to the wrong root — so the DC start was never reached. The DC
fallback had been added for the case where flat *diverges*; nobody checked the
case where it *converges wrongly*.

### Change made

`sensitivity/network_reduction.py`: the DSO ladder is now
`(("dc", 200), ("flat", 100))`. One line, no new machinery.

| reduced net | before | after |
|---|---|---|
| DSO 1 | 0.286996 | **0.023200** |
| DSO 2 | 0.322923 | **0.043914** |
| DSO 3 | 0.340569 | **0.043724** |
| DSO 4 | 0.102194 | 0.102194 |

7–12× better on three of four DSOs, reproduced at all four study windows. The
TSO reductions are untouched (separate code path, separate ladder).

### Second DSO defect: the boundary real-power split

DSO 4 did not respond to the DC reorder — its flat start *diverges*, so it was
already using DC. Its anatomy shows the other defect plainly. Cached coupler
flows at `2016-01-05 08:00` were **−26.5 / −64.9 / +7.9 MW** (two exporting,
one importing), yet all three boundary gens were created with `p_mw = 0.0`.

Only the first primary bus becomes the slack; the rest are **PV gens, whose P is
fixed at creation**. Leaving them at zero forces a multi-coupler DSO to push its
entire real-power exchange through one transformer. The error map confirmed it:
buses behind the slack coupler were 0.08–0.10 pu off while those behind the
other two were within 0.015 pu.

`sensitivity/network_reduction.py`: boundary gens are now created with
`p_mw = res_trafo3w.p_hv_mw` of their own coupler (power into the trafo at the
primary bus — after the TN is deleted, the boundary gen is the only other
element there, so it must inject exactly that).

### Third DSO defect: part of the DSO was missing

Even with both fixes the boundary slack solved at **−137.0 MW against a cached
−26.5 MW**, while the two PV boundary gens sat exactly on their cached values.
The arithmetic did not close: internal sgen \(265.6\) − internal load \(49.7\)
= \(215.9\) MW of surplus against a coupler outflow (Σ `p_mv`) of only
\(83.8\) MW. **132 MW of the DSO's own injection was unaccounted for.**

`HVNetworkInfo` carries `internal_aux_bus_indices` (with matching
`internal_aux_parent_buses` and `internal_aux_line_indices`), and the keep-set
was built from `bus_indices`, the coupling buses and the primaries only. The
auxiliary buses were therefore never kept, and `pp.drop_buses` removed them
together with their loads and sgens. **The reduced net was not a reduction of
the DSO; it was a different network.** Bus count 16 → 23 once restored.

### Result: the DSO reductions are now exact

Max |ΔV| against the combined solution:

| reduced net | original | + DC-first | + P-split | + aux buses |
|---|---|---|---|---|
| DSO 1 | 0.286996 | 0.023200 | 0.012360 | **0.000000** |
| DSO 2 | 0.322923 | 0.043914 | 0.016676 | **0.000000** |
| DSO 3 | 0.340569 | 0.043724 | 0.021587 | **0.000000** |
| DSO 4 | 0.102194 | 0.102194 | 0.048207 | **0.000000** |

All 16 DSO reductions (4 DSOs × 4 study windows) reproduce the combined
solution to 0.000000 pu. No TSO figure moved — separate code path.

### All three changes are necessary

Verified by reverting each in turn with the other two in place
(`2016-01-05 08:00`):

| configuration | DSO error |
|---|---|
| all three | **0.000000** |
| without the boundary P-split | 0.0056 – 0.0144 |
| without DC-first | 0.2581 – 0.2712 |

None is redundant. Each addresses a distinct defect: the wrong power-flow root,
the misrouted boundary real power, and the missing sub-network.

**Every input used is local.** The DC ordering carries no information at all;
`p_hv_mw` is the DSO's own PCC measurement; `internal_aux_bus_indices` is its
own topology metadata. Nothing beyond the boundary is consulted, so the
information boundary the reduction exists to model is intact.

## 7. The TSO zones: characterised, NOT fixed

Zone 1 reproduces the combined solution **exactly** (0.000000) at every window.
Zones 0 and 2 are off by 0.012–0.067 pu, concentrated in one corner: at
`2016-01-05 08:00` zone 0's error sits at buses 0, 1, 2 (0.026 / 0.015 / 0.015)
while the other ten buses are within 0.004.

Two hypotheses were tested and **both rejected**:

1. **Voltage-pinned tie boundary.** TSO zones represent every tie far-end as a
   pure PQ load, which removes the external system's voltage support; the DSO
   reduction pins its boundary and is exact there. Converting the `WARD_TIE`
   loads to PV gens at the cached voltage gave only 0.026 → 0.0228 (zone 0) and
   0.015 → 0.0125 (zone 2). Not the cause.
2. **Exhaustive per-bus stubs.** `WARD_TIE` is computed from *the tie line's own
   flow*, whereas what the far bus loses is the sum of **all** branches deleted
   at it. Those coincide only when the bus has one other connection — e.g. zone
   0 bus 2 loses 240.6 MW but gets a 14.8 MW stub, and bus 24 loses 159.2 MW
   with no stub at all. Replacing the category stubs with one aggregated stub
   per bus nevertheless made things **worse**: zone 1 went from exact to 0.292,
   zone 0 to 0.036, zone 2 to 0.034.

Result 2 is the important one, and it refutes the premise rather than the
implementation: **zone 1 shows large apparent "lost" flows by that accounting
and yet is exactly right**, so the deleted-branch sum is not the quantity the
stub should carry.

The element audit later explained why. At a tie far-end bus the **load is
dropped too** — zone 0 bus 2 carries 220.4 MW in the plant and keeps only the
95.9 MW stub — and the tie-flow-based stub value already nets the branches *and*
the local load together. The category stubs are therefore correct, and the
exhaustive branch sum double-counts. That closes the question raised above:
**the TSO Ward stubs are right and should be left alone.**

### The real TSO defect: scheduled setpoint vs actual output

The plant solves with ``distributed_slack=True``, so a generator's real output
differs from its ``p_mw`` setpoint — the machines share the slack burden through
their participation factors. The reduced net solves with
``distributed_slack=False``, which pins every PV generator exactly at ``p_mw``.
Carrying the setpoint across therefore injects power the plant never produced.
Zone 0 at `2016-01-05 08:00`:

| gen | bus | setpoint | actual (`res_gen`) |
|---|---|---|---|
| 0 | 29 | 250.00 | 160.06 |
| 7 | 37 | 830.00 | 740.06 |
| 9 | 40 | 1000.00 | 100.62 (slack) |

180 MW of phantom injection, which the slack then had to absorb (−214.0 MW
solved against +100.6 MW cached).

`sensitivity/network_reduction.py` now overwrites each retained generator's
``p_mw`` with its cached ``res_gen.p_mw`` before the reduced net is solved.
``res_gen.p_mw`` is the TSO's own machine telemetry, so this consults nothing
beyond the zone boundary.

### TSO result: improved, not solved

Max |ΔV|, all four study windows:

| window | zone 0 before | zone 0 after | zone 1 | zone 2 before | zone 2 after |
|---|---|---|---|---|---|
| 2016-02-22 13:00 | 0.011749 | 0.011460 | 0.000000 | 0.017103 | 0.018135 |
| 2016-01-05 08:00 | 0.026019 | **0.009666** | 0.000000 | 0.015127 | 0.016817 |
| 2016-01-15 03:00 | 0.050081 | **0.015242** | 0.000000 | 0.012838 | 0.014812 |
| 2016-12-18 14:00 | 0.067448 | **0.012966** | 0.000000 | 0.011578 | 0.014313 |

Zone 0 improves up to 5.2×; zone 1 stays exact; **zone 2 is consistently
~15 % worse**. The change is kept because it is unambiguously more correct —
using a machine's actual output rather than its setpoint — and zone 2's
regression indicates a second, compensating error there that the setpoint
mismatch had been partially cancelling.

Zone 2's residual has a clean signature: a **uniform +0.013 pu lift on every bus
except the two voltage-pinned generators**, with its promoted slack solving at
−29.97 MW against a cached 560.05 MW. Roughly 590 MW is unaccounted for, and a
less-loaded network sits higher, which is exactly what a uniform lift looks
like. Voltage-pinning the tie far-ends (tested again after the generator fix)
moves it only 0.0168 → 0.0132, so that is not the cause either.

## 8. Risks / unresolved points

## 7. Risks / unresolved points

1. **The reduction is wrong and remains wrong.** Nothing is fixed. The DSO
   reductions are the priority: their error is ~20× the TSO one and they build
   no boundary stubs at all.
2. **Re-running the completed study is premature.** A re-run is only worth its
   ~17 h (41 undisturbed + 36 disturbance runs) once the reduction passes the
   fidelity test. Re-running now would reproduce the same flaw at greater cost.
3. **The completed dead-band results are not invalidated by this, but their
   basis is weaker than stated.** All 77 runs share the same defective
   reduction, so comparisons *between* them remain like-for-like; what is in
   question is how well any of them represents a correctly linearised cascade.
4. Whether `local_sensitivities_dso` should be used at all until this is fixed
   is now a live question — the shared full-net Jacobian has no such defect,
   though it breaches the information boundary the reduction exists to model.
