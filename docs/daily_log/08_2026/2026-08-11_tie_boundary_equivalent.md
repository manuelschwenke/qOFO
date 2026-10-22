# 2026-08-11 — Tie-line boundary equivalent: PQ vs PV vs Z

**Author:** Manuel Schwenke / Claude Code
**Reason:** The reduced zone net condenses a neighbouring TSO area behind the
tie-line far-end bus into a constant PQ load. That choice was never compared
against an alternative. PQ is the infinite-Thevenin-impedance extreme (the
neighbour offers no voltage support); PV is the zero-impedance extreme (it
holds the boundary perfectly). The true equivalent has finite impedance and
therefore lies between them, so the spread between the two measures the
modelling uncertainty the choice introduces. Since the neighbouring area's
machines are AVR-regulated, the truth is expected to be stiffer than PQ,
which would mean the current model over-states the area's own authority at
the boundary and the MIQP over-steps there.

## Changed

### `sensitivity/network_reduction.py`
`build_tso_local_net(...)` gained a keyword-only `tie_boundary: str = "pq"`.
Default reproduces the previous behaviour exactly, so every existing run is
unaffected.

* **Structure.** `tie_load_specs` now carries `(bus, p_inj, q_inj, v_cached)`
  instead of `(bus, p_load, q_load)` — the far-end voltage at the cached
  operating point is captured in the same pass that reads the corridor flow,
  before the tables are edited. The sign convention was flipped to *injection
  into the far-end bus* so all three variants read from one quantity.
* **Step 7** branches on `tie_boundary`:
  - `"pq"` — `pp.create_load(p_mw=-p_inj, q_mvar=-q_inj)`, unchanged.
  - `"pv"` — `pp.create_gen(p_mw=p_inj, vm_pu=v_cached, slack=False,
    min/max_q_mvar=±1e6)`. Not a slack: the in-zone reference of step 9 still
    supplies the angle, so the only thing that changes between variants is the
    voltage-magnitude boundary condition.
  - `"z"` — `pp.create_shunt(p_mw=-p_inj/v², q_mvar=-q_inj/v²)`, i.e. a
    constant admittance matched at the cached voltage. Guarded: applied only
    where both components are non-negative (a genuinely passive absorber).
    Where the equivalent is a net source the constant-Z form would inject
    *more* as voltage rises — softer than constant power, i.e. further from
    the truth rather than closer — so those stubs fall back to PQ and the
    count is printed at `verbose >= 1`.
* **Step 9c bug fix (pre-existing, exposed by the new gens).** The cached-P
  overwrite loop iterated every row of `sub.gen` and copied
  `net.res_gen.at[g, "p_mw"]` by index. Boundary gens added in step 7 get
  freshly assigned indices, and because the reduction drops rows,
  `create_gen` reuses a gap — so a boundary gen could land on an index that
  belongs to a *different* machine in the plant's `res_gen` and have its
  in-feed silently overwritten. The loop now skips rows whose `name` starts
  with `WARD_`.

### `configs/config.py`
`MultiTSOConfig.tie_boundary_equivalent: str = "pq"`. Read only when
`local_sensitivities_tso=True`.

### `experiments/runners/multi_tso_dso.py`
`_build_tso_local_jac` forwards
`tie_boundary=getattr(config, "tie_boundary_equivalent", "pq")`.
`getattr` with a default so older pickled configs still load.

### `experiments/CIGRE_2026/007_TIE_BOUNDARY_COMPARE.py` (new)
Measures H fidelity rather than closed-loop performance, deliberately:

* All three variants reproduce the same far-end operating point by
  construction, so they differ **only** in the derivative — which is the
  entire content of H. Comparing H is therefore the direct measurement.
* `G_w` was BO-tuned with the PQ model in place. A closed-loop swap would
  measure tuning mismatch, not the equivalent. Comparing H removes that
  confound entirely.

Method: `run_multi_tso_dso(cfg, pre_loop_hook=...)` returns the production
post-init state, so no setup is replicated. `compute_numerical_h_tso` is then
applied with identical settings to (a) the full interconnected plant net —
the truth every reduced model approximates — and (b) each variant's reduced
net. One estimator throughout, so analytical-formula bias cancels.

Both plant and reduced copies have their controller table emptied first
(`_freeze`): the reduced nets are deep copies and inherit controllers
referencing dropped elements, which `compute_numerical_h_tso`'s
`run_control=True` baseline would trip over. DER Q(V) droop is therefore
frozen at its converged setpoints in every net; synchronous-machine AVR
action is retained everywhere, being inherent in the PV-bus model. Note the
direction of that simplification: active droop would stiffen the boundary
*further* toward PV, so the truth used here is if anything biased toward PQ.

Metrics per zone and variant: relative Frobenius error over the whole matrix,
the same restricted to the corridor-terminal voltage rows (what BRC-H
tracks), and the median column-wise gain ratio on those rows — the last being
the unit-free one, and the one that says whether the model over- or
under-states the area's boundary authority.

## Results

`results/007_tie_boundary/h_fidelity.csv`. Operating point 2016-01-05 08:00,
V4 configuration, `local_sensitivities_tso=True`.

| zone | variant | relF | relF_corr | gain |
|---|---|---|---|---|
| 1 | pq | 0.407 | 0.325 | **1.414** |
| 1 | pv | 0.759 | 0.245 | 0.645 |
| 1 | z  | 0.394 | 0.317 | 1.391 |
| 2 | pq | 1.285 | 3.611 | **4.475** |
| 2 | pv | 0.213 | 0.404 | 0.636 |
| 2 | z  | 1.051 | 3.142 | 3.999 |
| 3 | pq | 0.559 | 1.649 | **2.619** |
| 3 | pv | 0.164 | 0.535 | 0.475 |
| 3 | z  | 0.559 | 1.649 | 2.619 |

Mean over zones — pq: relF_corr 1.862, gain 2.836; pv: 0.395, 0.586;
z: 1.703, 2.670.

1. **The predicted direction holds in every zone.** PQ over-states the
   corridor gain (1.41 / 4.47 / 2.62, all > 1); PV under-states it
   (0.65 / 0.64 / 0.48, all < 1). The two do bracket the truth.
2. **The bracket is strongly asymmetric.** PQ is high by up to 4.5x, PV low
   by at most ~2x, so the true equivalent sits much nearer the PV end — the
   neighbouring area behaves close to a stiff voltage source, which is what
   AVR-regulated machines behind the corridor should do. On the corridor
   rows PV cuts the mean error by 4.7x (1.86 -> 0.39).
3. **The PQ error scales inversely with in-zone machine count** (from
   `007b`): zone 1 has 3 machines *and holds the system slack* -> gain 1.41;
   zone 3 has 2 machines, slack promoted -> 2.62; zone 2 has 1 machine,
   slack promoted -> 4.47. Monotone. Mechanism: the PQ convention leaves the
   area's voltage entirely to its own machines, so the fewer it has, the
   more it over-estimates its own authority at the boundary.
4. **Zone 1 is the one place PV is worse on the whole matrix** (relF 0.76 vs
   0.41) while still better on the corridor rows (0.245 vs 0.325). Zone 1
   already holds the system slack, so a PV boundary adds a *second* stiff
   anchor and over-stiffens; the degradation shows up in the Q_gen block,
   where a free-Q boundary absorbs reactive power the area's own machines
   would really have to supply. PV therefore risks under-stating machine
   reactive loading — relevant because those rows carry the SG capability
   soft constraints.
5. **Constant-Z does not deliver on this system.** Only 3 of 10 far-end
   stubs are passive absorbers; the rest are net sources and fall back to
   PQ. Zone 3 got zero Z stubs, which is why its `z` row is bit-identical to
   `pq`. Negative result: drop Z as a candidate here.
6. **Side finding, quantified.** The all-PQ convention forces a slack
   promotion in every zone that does not hold the system slack, and the
   promoted machine's OLTC column is masked out of the controller's model:
   zone 1 loses 0 of 2 OLTC columns, zone 2 loses 1 of 3, zone 3 loses 1 of
   2 — half its tap actuators. Anchoring the angle reference at a far-end
   tie bus instead would remove this, independently of the PQ/PV question.

### Caveats

* `relF` over the whole matrix is dominated by the Q_gen rows, whose units
  (Mvar/pu) are ~1e3 larger than the voltage rows. Do not lean on it; the
  corridor-row metrics are the meaningful ones.
* The column-wise `gain` median survives few columns after the relevance
  filter (5 / 1 / 3). Zone 2's value rests on a single column. `relF_corr`
  uses all columns and tells the same story, so the ranking is robust even
  where the gain figure is not.
* One operating point only.
* DER Q(V) droop frozen in all nets (see `_freeze` above). Active droop
  would stiffen the boundary further toward PV, so this truth is biased
  toward PQ — i.e. PV's advantage here is a lower bound.

## Follow-up: Thevenin boundary (same day)

`build_tso_local_net` gained `tie_boundary="thevenin"` + `tie_thevenin_k`,
`build_dso_local_net` gained `boundary="thevenin"` + `thevenin_k`, both via a
shared `add_thevenin_boundary()` helper: auxiliary bus, series branch, voltage
source, with the EMF and its in-feed back-solved so the cached operating point
at the boundary bus is reproduced exactly for ANY impedance. The whole family
is therefore a one-parameter sweep that matches at the linearisation point and
differs only in the derivative. Defaults (`"pq"` / `"slack"`) are unchanged.

Scripts: `007d_THEVENIN_SWEEP.py` (TS side; validates a reconstructed
`build_tso_local_net` argument set against the runner's own net before
sweeping, so one setup run covers all `k`), `007e_DS_THEVENIN.py` (DS side).

### TS horizontal boundary — corridor-row relF vs k = Z_th / Z_line

| k | zone 1 | zone 2 | zone 3 | mean |
|---|---|---|---|---|
| PQ (inf) | 0.325 | 3.611 | 1.649 | 1.862 |
| 4 | 0.113 | 0.721 | 0.344 | 0.393 |
| 2 | **0.092** | 0.342 | **0.040** | 0.158 |
| 1.5 | 0.105 | 0.251 | 0.110 | **0.155** |
| 1 | 0.132 | **0.206** | 0.230 | 0.189 |
| 0.5 | 0.177 | 0.260 | 0.370 | 0.269 |
| PV (0) | 0.245 | 0.404 | 0.535 | 0.395 |

Pronounced U-shape in every zone, so the finite-impedance model genuinely
beats both extremes rather than interpolating between them. Optimal k lies in
[1, 2] in all three zones despite the zones differing in machine count and
slack ownership. A **single default k = 1.5** gives mean 0.155 against PQ's
1.862 — a 12x reduction from one dimensionless constant and no per-corridor
data exchange. k = 1 minimises the worst case (0.230). Physical reading: the
neighbouring area presents roughly one to two tie-line impedances.

Zone 2 fails to converge at k >= 8 — a nearly-open boundary on a one-machine
zone. Consistent with it being the weak zone.

### DS vertical boundary

`dV/dQ` at the coupling-transformer primary buses:

* `boundary="slack"`: **UNAVAILABLE** (raises) at all 12 primary buses across
  all four DSOs. Not zero — structurally absent. The slack bus is eliminated
  from the reduced Jacobian and PV buses have no `d|V|/dQ` row, so the DSO
  cannot monitor, constrain, or evaluate a sensitivity at its own
  transmission-side terminals.
* `boundary="thevenin"`: available at all 12, scaling ~linearly with k
  (~1.8e-4 / 3.4e-4 / 6.3e-4 pu/Mvar at k = 0.5 / 1 / 2), as a Thevenin
  should.

Fidelity, mean relF over the four DSOs: slack 0.0751, k=0.5 0.1041,
k=1 0.2000, k=2 0.3460. The current slack model is already near-optimal here
and Thevenin degrades monotonically — which is exactly what the unification
argument predicts, since the TS is very stiff relative to a 110 kV DS, so
Z_th -> 0 is the right physics at this boundary. DSO_2 and DSO_4 do improve
slightly at k=0.5 (0.102 -> 0.068 and 0.102 -> 0.055).

### DS boundary, decided on the rows that matter

The primary-bus voltage sensitivity turns out to be a capability the DSO does
not need: its tracked output is the interface reactive flow, which is a BRANCH
flow of the retained coupler, not a nodal voltage. So the decision rests on the
`Q_iface` rows alone. Splitting them out:

| variant | relF (Q_iface) | gain (Q_iface) |
|---|---|---|
| slack (current) | 0.0751 | **1.069** |
| th k=0.25 | **0.0626** | 0.988 |
| th k=0.5 | 0.1041 | 0.920 |
| th k=1 | 0.2000 | 0.812 |

`relF` over all rows equals `relF` over the `Q_iface` rows to four digits: the
V and I blocks are ~1e-4 pu/Mvar against ~1 Mvar/Mvar on the Q rows, so they
contribute nothing to the Frobenius norm. The whole-matrix figures reported
above were therefore already the interface figures.

Reading: the slack model over-states the DSO's interface authority by ~7 %,
the same DIRECTION as the TS-side error (too-soft external system -> own
authority over-stated) but 7 % against 180 %. A small Thevenin (k ~ 0.25)
trims the gain to ~1.00 and improves mean interface fidelity by ~17 %, but the
improvement is not uniform -- DSO_3 gets worse (0.037 -> 0.084) while the other
three improve. Marginal, non-uniform, and it adds a parameter.

**Decision: keep `boundary="slack"` on the DS side.** The one argument for
changing it -- restoring primary-bus voltage sensitivity -- is moot given the
DSO only needs the interface flow, and the residual 7 % gain bias does not
justify a new tuning parameter. The asymmetry with the TS recommendation is
the physics, not an inconsistency.

### The fitted k IS the physical Thevenin impedance (`007f_ZTH_PER_CORRIDOR.py`)

The sweep's `k` was a fit. This measures the impedance the neighbouring system
actually presents at each corridor far-end bus and asks whether it agrees.

Method: delete the zone from the plant (its buses, its DSOs, and by cascade the
tie lines) -- what remains is exactly the external system that zone condenses.
Restore a slack if the zone owned the system one. Then probe the far-end bus:
inject +/- dQ and +/- dP, re-solve, read d|V|. For a source behind
`Z = R + jX`, `X ~ V dV/dQ` and `R ~ V dV/dP` in per unit.

Measuring by perturbation rather than from Ybus is deliberate: it is the same
definition the H comparison uses, and it embeds the neighbour's AVR behaviour
automatically, since PV buses hold their magnitude and therefore act as
sources. That is the correct stiffness for a steady-state QV sensitivity and it
is NOT the fault-study `Sk''`, which embeds `Xd''` and would come out too
stiff. Linearity check at two perturbation sizes: 0.0 % deviation on every
corridor.

| zone | k_phys mean | k_phys median | fitted k* (007d) |
|---|---|---|---|
| 1 | 1.93 | -- | 2.0 |
| 2 | 1.05 | 1.01 | 1.0 |
| 3 | 1.60 | 1.40 | 2.0 |

**All corridors: k_phys mean 1.48, median 1.32.** The independent H-error sweep
put the best single default at k = 1.5. Two unrelated measurements -- one from
sensitivity-matrix fidelity, one from a voltage-perturbation probe of the
external network -- land on the same number. `k ~ 1.5` is therefore not a fit;
it is the measured physical Thevenin impedance, and the sweep rediscovered it.

Per-corridor spread is wide: 0.05 to 2.73.

* zone 2 / line 14 / far bus 38: `k_phys = 0.05`, `|Z_th| = 1.46 ohm`. Bus 38
  sits at an AVR-regulated machine, so the corridor terminates essentially ON a
  voltage source and the neighbour is almost perfectly stiff there. This single
  corridor pulls zone 2's mean down; its median (1.01) is the representative
  figure.
* zone 1 / line 14 / far bus 8: `k_phys = 2.73`, the softest -- a corridor
  terminating deep in the neighbour's network.

So the physical picture is: corridors ending near a machine are stiff, corridors
ending deep in the neighbour are soft, and the population mean is ~1.5. The 50x
spread means a per-corridor value would beat a uniform one -- which is the
argument for the one-number-per-corridor BRC-H extension. The uniform 1.5
already captures the bulk of the improvement (12x on corridor H error).

### Retraction

The earlier note that the first (slack-touching) coupler's OLTC column is
systematically ~25 % weaker than its siblings does NOT reproduce. That reading
came from the controller's cached `_H_cache`, whose OLTC block sits on a
different scale. With `_build_sensitivity_matrix()` rebuilt against each
variant's Jacobian the spread is 1.05-1.10 and is the same under both
boundary conventions. Claim withdrawn.

## Closed-loop test (2026-08-12, `007g_CLOSEDLOOP_BOUNDARY.py`)

Everything above is open loop. This asks whether the better boundary model
actually controls better: 2 h of QSS on the tuned V4 cascade, identical in
every respect except the boundary the TSO controllers linearised on.

### Design, and the confound it is built around

`G_w` was BO-tuned with the PQ model in place. If PQ inflates the corridor rows
of H by ~2.8x, that tuning has already absorbed the inflation, so swapping in a
truer H at the SAME `G_w` leaves the corridor loop ~2.8x slower than intended.
A naive two-arm test would measure tuning mismatch rather than model quality --
and could plausibly show the BETTER model performing WORSE. Hence three arms:

* `pq` -- state of the art: PQ boundary, nominal `G_w`.
* `th` -- Thevenin k = 1.5, nominal `G_w`. The honest "just switch the flag"
  result, gain mismatch included.
* `th_gw` -- Thevenin k = 1.5 with the TSO `G_w` block scaled by
  `kappa = 1/2.84` to restore the loop gain. Only the four TSO weights
  (`g_w_der`, `g_w_gen`, `g_w_pcc`, `g_w_tso_oltc`) are scaled -- the DSO model
  is untouched by this experiment, so moving its tuning would confound the
  comparison.

`th` and `th_gw` bracket the effect. Both beating `pq` = robust to the
confound. Only `th_gw` beating it = the gain matters more than the model, which
is itself worth knowing.

k = 1.5 is used because the H-error sweep (`007d`) and the physical impedance
measurement (`007f`, k_phys mean 1.48) independently agree on it.

### Metric

`rms_v_ts_pu` from `cigre_summary_table`: unweighted across-zone RMS of
(V - v_set) on TS buses, time-averaged. Deliberately NOT the controllers' own
objective, so it cannot flatter whichever arm shares its weighting. Switching
count and interface tracking are reported alongside, because an arm that simply
moves less will look calmer without controlling better.

Split at minute 60, where the inherited schedule trips a generator. Over a 2 h
horizon it never restores, so the second hour runs post-contingency on a model
frozen at t = 0 -- the regime where a truer boundary should pay off most.

Per-zone breakdown is the mechanism test: open loop the PQ gain over-statement
was 1.41 / 4.47 / 2.62 in zones 1 / 2 / 3, scaling inversely with in-zone
machine count. A real benefit should follow that ordering. An aggregate that
improves while the per-zone pattern does not would mean the gain came from
somewhere other than the boundary.

### Results (2 h, V4 cascade, gen trip at minute 60)

| arm | rms_v_ts_pu | vs pq | pre-trip | post-trip | n_sw | rms_q_tie | rms_e_sts |
|---|---|---|---|---|---|---|---|
| `pq` (state of the art) | 0.00764 | -- | 0.00646 | 0.00881 | 5 | 30.9 | 1.94 |
| `th` (same G_w) | 0.01069 | **+39.9 %** | 0.00742 | 0.01393 | 1 | 37.9 | 1.95 |
| `th_gw` (gain restored) | 0.00683 | **-10.7 %** | 0.00550 | 0.00813 | 7 | 26.8 | 2.19 |

Per zone (open-loop PQ gain over-statement: z1 1.41, z2 4.47, z3 2.62):

| arm | zone 1 | zone 2 | zone 3 |
|---|---|---|---|
| `pq` | 0.00658 | 0.00745 | 0.00842 |
| `th` vs pq | +4.8 % | +35.4 % | +60.7 % |
| `th_gw` vs pq | **-26.2 %** | -2.5 % | -8.1 % |

**1. The drop-in swap is 40 % worse, and the reason is gain, not model.** `n_sw`
collapses 5 -> 1: a truer (smaller) H against a `G_w` tuned for an inflated one
gives an under-driven loop that barely moves. The confound anticipated in the
design is real and large. Practical consequence: the boundary model cannot be
changed without re-tuning.

**2. The `th` arm CONFIRMS the open-loop mechanism.** Its per-zone degradation
tracks the over-statement ordering -- zone 1 barely moves (+4.8 %, bias only
1.41), zones 2 and 3 degrade heavily (+35 %, +61 %). The loop slows in
proportion to how much its gain had been inflated, which is exactly what a
corrected H should do.

**3. The `th_gw` improvement is NOT yet attributable to the boundary.** Its
per-zone pattern is the REVERSE of the prediction: largest gain in zone 1
(-26.2 %, smallest bias) and smallest in zone 2 (-2.5 %, largest bias).
Diagnosis: `kappa = 1/2.84` is a GLOBAL scale applied to ZONE-SPECIFIC biases
(1.41 / 4.47 / 2.62). It over-compensates zone 1 -- which now runs faster than
it ever did -- and under-compensates zone 2, which stays sluggish. That
predicts the observed ordering exactly, so the 10.7 % may be a retuning
artefact of the arm design rather than a model effect.

**4. Costs.** Interface tracking degrades (`rms_e_sts` 1.94 -> 2.19 Mvar,
+13 %) and switching rises 5 -> 7, against a 13 % drop in inter-area reactive
exchange (`rms_q_tie` 30.9 -> 26.8 Mvar). Reserve unchanged (`m_bar_pu` 0.367).

**5. Where the gain sits in time.** The improvement is larger PRE-contingency
(-14.9 %) than post (-7.7 %), the opposite of the expectation that a frozen
model would hurt most after a disturbance. Coherent reading: post-trip the
error is dominated by the disturbance, which no boundary model addresses, while
pre-trip the residual IS the model-driven compromise -- the "over-determined
compromise" channel, which is precisely what a truer H should fix.

### Arms added to settle point 3

* `th_gwz` -- Thevenin with PER-ZONE `kappa_z = 1/gain_z`.
* `pq_gwz` -- **PQ with the same per-zone `kappa_z`**, the control. If it
  reproduces `th_gwz`, the effect is pure retuning and the boundary model
  contributes nothing in closed loop; if `th_gwz` beats it, the model
  contributes something the tuning cannot reach.

First attempt raised `FrozenInstanceError`: `OFOParameters` is a frozen
dataclass, so `ctrl.params.g_w = ...` fails. Fixed with
`dataclasses.replace(ctrl.params, g_w=scaled)` applied in the `pre_loop_hook`
(which returns falsy so the main loop still runs). Applied scalings:
zone 1 x0.707, zone 2 x0.224, zone 3 x0.382.

### Full arm set

| arm | rms_v_ts_pu | vs pq | rms_e_sts | n_sw | rms_q_tie |
|---|---|---|---|---|---|
| `pq` | 0.00764 | -- | 1.94 | 5 | 30.9 |
| `th` | 0.01069 | +39.9 % | 1.95 | 1 | 37.9 |
| `th_gw` | 0.00683 | -10.7 % | 2.19 | 7 | 26.8 |
| `th_gwz` | 0.00723 | **-5.4 %** | 2.26 | 4 | 30.9 |
| `pq_gwz` | 0.00633 | **-17.2 %** | 2.55 | 8 | 29.8 |

Per zone vs `pq`:

| arm | zone 1 | zone 2 | zone 3 |
|---|---|---|---|
| `th_gwz` | +0.6 % | -8.1 % | -7.8 % |
| `pq_gwz` | +0.3 % | -29.8 % | -19.5 % |

### Correction to the arm design, and what it means

`pq_gwz` was intended as a matched control and IS NOT ONE. Step magnitude goes
as `H / g_w`. Writing `H_pq = gain_z * H_true`:

* `th_gwz`: `H_true` against `g_w / gain_z` -> step ~ `gain_z * H_true / g_w`,
  i.e. the SAME step magnitude as `pq`. This is the clean comparison.
* `pq_gwz`: `gain_z * H_true` against `g_w / gain_z` -> step ~
  `gain_z^2 * H_true / g_w`, i.e. `gain_z` times MORE aggressive than `pq`
  (4.5x in zone 2, 2.6x in zone 3). A different operating point, not a control.

So the readings are:

**1. The clean model comparison is `pq` vs `th_gwz`** -- matched step magnitude,
different H direction. Thevenin is **5.4 % better** on TS voltage tracking, and
the per-zone pattern NOW matches the mechanism: zone 1 neutral (+0.6 %, smallest
open-loop bias 1.41), zones 2 and 3 improved (-8.1 %, -7.8 %, biases 4.47 and
2.62). Real, in the predicted direction, and modest.

**2. A retune of the EXISTING model beats it three-fold.** `pq_gwz` reaches
-17.2 % without touching the boundary at all. The dominant finding is therefore
not the boundary model: it is that the case study's uniform `G_w` leaves zones 2
and 3 badly under-driven, and simply making them more aggressive helps far more
than fixing their sensitivity model. The boundary work found this by accident.

**3. Neither is a free lunch.** Every arm that improves TS voltage tracking
degrades interface tracking, monotonically: `rms_e_sts` 1.94 (pq) -> 2.26
(th_gwz) -> 2.55 (pq_gwz), i.e. +16 % and +31 %. `pq_gwz` also raises switching
5 -> 8. `th_gwz` is the only improving arm that LOWERS switching (5 -> 4). There
is a trade-off axis here, not a dominance ordering, and which end is preferable
is a design choice rather than a measurement.

### Bottom line

On this trajectory the Thevenin boundary is worth about 5 % of TS voltage
tracking at matched loop gain, and it cannot be adopted without re-tuning --
dropped in naively it costs 40 %. That is a real but small control benefit,
and it is dominated by per-zone gain tuning on the existing model. The stronger
case for the Thevenin model remains the open-loop and structural one (12x H
fidelity, one derived object replacing three asserted conventions), not the
closed-loop one.

## Tuning frontier, 5 h (2026-08-12, `007h_BOUNDARY_FRONTIER.py`)

Single-point comparisons kept collapsing into the tuning confound, so this
sweeps the SAME gain ladder on both boundaries and compares the resulting
CURVES in the (voltage tracking, interface tracking) plane. A genuinely better
model must produce a frontier lying inside the other's. That question has no
tuning confound left in it.

| arm | kappa | rms_v_ts | rms_e_sts | set_move | n_sw | q_tie |
|---|---|---|---|---|---|---|
| `pq_k1`   | 1.00 | 0.00764 | 1.531 | 1.858 | 15 | 30.19 |
| `pq_k0.5` | 0.50 | 0.00645 | 2.082 | 2.419 | 22 | 24.73 |
| `pq_k0.3` | 0.30 | 0.00549 | 2.442 | 3.292 | 27 | 24.91 |
| `pq_k0.2` | 0.20 | 0.00518 | 2.912 | 4.621 | 30 | 24.03 |
| `th_k1`   | 1.00 | 0.00936 | 1.666 | 1.420 | 12 | 35.04 |
| `th_k0.5` | 0.50 | 0.00742 | 1.908 | 1.617 | 14 | 29.14 |
| `th_k0.3` | 0.30 | 0.00664 | 2.125 | 1.855 | 20 | 30.49 |
| `th_k0.2` | 0.20 | 0.00617 | 2.392 | 2.105 | 25 | 27.27 |

### 1. The PQ frontier dominates

Pairing by interface error (the confound-free axis), PQ reaches lower voltage
error at every level within the overlapping range:

| e_sts | pq rms_v | th rms_v |
|---|---|---|
| ~1.6 | 0.00764 @ 1.53 | 0.00936 @ 1.67 |
| ~1.9-2.1 | 0.00645 @ 2.08 | 0.00742 @ 1.91 |
| ~2.1-2.4 | 0.00549 @ 2.44 | 0.00664 @ 2.13 |
| ~2.4 | 0.00549 @ 2.44 | 0.00617 @ 2.39 |

**On this system, in closed loop, the Thevenin boundary does not improve
control -- it is consistently worse.** At equal kappa it is 15-22 % worse on TS
voltage tracking; on the frontier it is dominated. This stands against a 12x
open-loop H-fidelity advantage, so open-loop fidelity is NOT predicting
closed-loop performance here.

**Caveat -- the Thevenin frontier is truncated.** Its H is smaller, so at equal
kappa it drives far less: set_move spans 1.42-2.11 against PQ's 1.86-4.62.
Reaching PQ's aggressive end would need kappa ~ 0.1-0.05, which was not run.
The dominance claim therefore holds WITHIN the overlapping interface-error
range (1.67-2.39) and is unverified outside it.

### 2. The interface question: my earlier "exonerated" reading was WRONG

Within each boundary, interface error tracks setpoint movement strongly -- the
loop-gain mechanism is real. But the two boundaries sit on DIFFERENT lines:

| boundary | e_sts / set_move |
|---|---|
| pq | 0.824, 0.861, 0.742, 0.630 |
| thevenin | 1.173, 1.180, 1.146, 1.136 |

At matched setpoint drive Thevenin incurs ~40 % more interface error. The
overall r = 0.864 is driven by the within-boundary trends and masks the offset
between them. So there IS a boundary-specific interface penalty, and the
partial-data conclusion (drawn from five arms, before any Thevenin point had a
PQ neighbour at similar drive) was premature.

Likely mechanism, untested: the boundary also enters the Q_PCC output rows of
H, and 007 only ever measured the corridor-VOLTAGE rows (`relF_corr`) and the
unit-dominated whole-matrix `relF`. The Q_PCC rows were never broken out. If
Thevenin degrades those while improving the corridor rows, the closed-loop
result follows directly. Cheap to check with the existing 007 harness.

### 3. The dominant lever is still gain, not the boundary

The PQ curve alone: 0.00764 -> 0.00518 from kappa 1.0 to 0.2, a **32 %**
improvement, and still falling at the end of the ladder. No boundary variant
comes near that. The shipped tuning is well off its own optimum. Cost:
switching 15 -> 30 and interface error 1.53 -> 2.91.

### 4. Per zone, Thevenin fails exactly where it should have helped most

| arm | zone 1 | zone 2 | zone 3 |
|---|---|---|---|
| `pq_k0.2` | 0.00462 | **0.00500** | 0.00551 |
| `th_k0.2` | 0.00461 | **0.00764** | 0.00532 |

Zones 1 and 3 are a wash. Zone 2 -- the one-machine zone with the LARGEST
open-loop PQ bias (4.47), where the Thevenin correction should pay off most --
is 53 % worse, and it barely responds to gain at all under Thevenin
(0.01009 -> 0.00764 across the whole ladder, against pq's 0.00839 -> 0.00500).
No confirmed explanation. Note k = 1.5 was applied uniformly while 007f
measured zone 2's physical k_phys at 1.05, the lowest of the three, so zone 2
is the zone whose boundary is furthest from its measured value -- but that
should bias it TOWARDS PQ behaviour, not away, so this does not explain it.

### Bottom line

The Thevenin boundary is well supported open loop (12x H fidelity, a physical
k_phys that two independent measurements agree on, one derived object replacing
three asserted conventions) and NOT supported in closed loop on this system.
Those are compatible: OFO acts on measurements, so a more faithful H buys less
than its fidelity suggests, and here it costs something on the interface rows
that the corridor-row metric never saw. Report it as a modelling contribution
with an honest negative control result, not as a performance improvement.

## H fidelity by row block (`007i_H_ROW_BLOCKS.py`)

Test of the one candidate mechanism for 007h's closed-loop interface penalty:
does Thevenin degrade the Q_PCC rows -- the TSO's model of its own TS-DS
interface flows -- while improving the corridor rows? Rows split as
`[V_bus | Q_PCC | I_line | Q_gen]`, each compared against the numerical truth
on the full plant.

relF against truth (lower better); `gain` = ||H_var|| / ||H_truth||:

| zone | block | n | pq relF | pq gain | th relF | th gain |
|---|---|---|---|---|---|---|
| 1 | V_all | 8 | 0.2724 | 1.183 | **0.0867** | 0.972 |
| 1 | V_corridor | 3 | 0.3251 | 1.195 | **0.1049** | 0.969 |
| 1 | I_line | 8 | 1.1978 | 1.589 | **0.9852** | 1.414 |
| 1 | Q_gen | 3 | 0.4070 | 0.721 | **0.1607** | 1.066 |
| 2 | V_all | 12 | 1.8535 | 2.770 | **0.0926** | 1.046 |
| 2 | V_corridor | 3 | 3.6113 | 4.475 | **0.2512** | 1.100 |
| 2 | Q_PCC | 9 | 1.0006 | 0.0353 | 1.0000 | 0.0348 |
| 2 | I_line | 12 | 1.7304 | 0.810 | **0.8313** | 0.586 |
| 2 | Q_gen | 1 | 1.2855 | 0.286 | **0.0183** | 0.982 |
| 3 | V_all | 10 | 0.7246 | 1.597 | **0.0491** | 0.967 |
| 3 | V_corridor | 3 | 1.6493 | 2.639 | **0.1102** | 0.904 |
| 3 | Q_PCC | 3 | 1.0017 | 0.0347 | 1.0030 | 0.0308 |
| 3 | I_line | 10 | 1.0894 | 0.647 | **0.9489** | 0.640 |
| 3 | Q_gen | 2 | 0.5586 | 0.821 | **0.0208** | 0.994 |

### 1. The candidate mechanism is REFUTED

Q_PCC: pq 1.0006 / th 1.0000 (zone 2) and pq 1.0017 / th 1.0030 (zone 3). Tied
to four decimals. Thevenin does NOT degrade the interface rows, so the ~40 %
closed-loop interface penalty of 007h is **not** explained by this and remains
open.

### 2. Thevenin is better on EVERY other block, often by an order of magnitude

Not just the corridor rows it was designed for: `Q_gen` improves 1.29 -> 0.018
in zone 2 (70x) and 0.56 -> 0.021 in zone 3 (27x); `V_all` 1.85 -> 0.093 (20x)
in zone 2. Its gain ratios sit at 0.90-1.10 across the voltage blocks against
PQ's 1.18-4.47, i.e. the Thevenin H is close to unbiased where PQ is not.

**So the puzzle deepens rather than resolves.** Thevenin has a strictly better
or equal H on every row block, and controls worse. The honest headline is that
in this loop **H fidelity does not predict closed-loop performance** -- which
is evidence for, not against, the standing intuition that OFO tolerates
sensitivity error, and which makes the boundary question much less consequential
than the open-loop numbers suggested.

### 3. Incidental: the Q_PCC rows are ~97 % missing in BOTH models

`gain` ~ 0.035 on the Q_PCC block for both variants: the reduced model captures
about 3.5 % of the open-loop interface response the plant shows.

**This is partly by design and must not be reported as a defect without the
qualification.** Ch 6 models the subordinate DS as a frozen PQ whose response is
carried by the virtual actuator (unit gain), explicitly stating that the passive
equivalent "contributes nothing to this response". The truth used here has the
DS physically attached with its controllers frozen, so a TSO voltage change
propagates into the DS and moves Q_PCC -- propagation the reduced model omits on
purpose, on the assumption that the DSO controller counteracts it. So this truth
is arguably the wrong reference for THIS block, and the number measures the
modelling assumption rather than an error. Worth a separate check with the DSO
loops active; not actionable as stated.

### 4. Incidental: the I_line rows are poor in both

relF 0.83-1.20 across all zones and both boundaries. These are the
current-constraint rows. Pre-existing, boundary-independent, and unexamined.

## Corridor vs interior (`007j_CORRIDOR_VS_INTERIOR.py`)

Third hypothesis for the closed-loop loss: shared-boundary under-provision --
both areas track the same corridor, a Thevenin boundary correctly tells each
that the other is helping, both correctly reduce effort, the shared bus is
under-served. Predicts the loss is LOCALISED AT THE CORRIDORS.

**Refuted, in the opposite direction.** Thevenin vs PQ, mean over the gain
ladder: corridor terminals **+13.9 %**, interior buses **+27.0 %**. And the two
move opposite ways as the loop is driven harder:

| kappa | corridor | interior |
|---|---|---|
| 1.00 | +26.0 % | +22.5 % |
| 0.50 | +15.7 % | +17.6 % |
| 0.30 | +8.6 % | +32.5 % |
| 0.20 | +5.4 % | +35.5 % |

Thevenin CATCHES UP at the corridors (26 -> 5 %) and falls further behind in the
interior (22 -> 36 %). Per zone, interior, Thevenin vs PQ:

| kappa | zone 1 | zone 2 | zone 3 |
|---|---|---|---|
| 1.00 | +12.6 % | +20.2 % | +29.0 % |
| 0.50 | +8.2 % | +33.9 % | -6.8 % |
| 0.30 | +16.9 % | +60.3 % | -0.3 % |
| 0.20 | -1.8 % | **+96.2 %** | **-13.8 %** |

At the aggressive end two of three zones are neutral-to-BETTER under Thevenin in
the interior. The entire closed-loop deficit is **zone 2's interior**, and it
grows with gain.

### What actually explains it

Not physics, and not H accuracy -- 007i showed Thevenin's zone-2 V_all block is
20x MORE faithful (relF 0.093 vs 1.854). It is the optimiser.

The MIQP trades predicted output improvement against the step cost
`w' G_w w`. Inflating H by a factor gamma inflates the predicted benefit per
unit of actuator movement by gamma, so the optimiser will spend gamma times more
effort for the same predicted gain. Feedback then delivers the real (smaller)
gain -- but the effort was still spent, and in an under-tuned loop spending more
effort helps. **An over-stated H acts as a discount on actuator cost.**

PQ's inflation is not uniform: measured gain ratios are 1.18 (zone 1), 2.77
(zone 2), 1.60 (zone 3) on V_all, and 1.20 / 4.47 / 2.64 at the corridors. So PQ
hands the largest accidental gain boost to zone 2 -- the ONE-MACHINE zone, the
one least able to help itself. Thevenin removes that boost, and zone 2 has no
tuning headroom of its own to replace it.

### Why no kappa sweep could ever have settled this

A scalar kappa scales every zone and every actuator class alike, so the ~4.5x
PQ-vs-Thevenin drive ratio in zone 2 is preserved at EVERY kappa -- which is
exactly why the gap never closed across the ladder. Equalising would need a
per-zone and per-output-block gain, which `G_w` (per actuator class) cannot
express.

**So the comparison as constructed cannot answer "which boundary controls
better".** The boundary change is inseparable from a per-zone, per-block
effective-gain change that the controller's tuning structure cannot compensate.
PQ's apparent advantage is an accidental, unevenly distributed gain boost, not a
control-theoretic property of the boundary model.

Constructive reading: fix the tuning per zone (the 32 % headroom found in 007h)
and the boundary choice should largely stop mattering.

## Scored on the BO's own objective (`007k_SCORE_ON_BO_OBJECTIVE.py`)

### RETRACTION: the "32 % tuning headroom" was wrong

That figure was `rms_v_ts` alone. The BO optimises a scalar in which TS voltage
RMS and interface-Q carry EQUAL weight (`PerfWeights`: v_rms_ts 1.0, q_pcc 1.0,
v_worst_ts 0.5, v_band_ts 1.0, v_rms_ds 0.3, pcc_underutil 0.3; switching sits
in the constraint vector, not here). Over the same sweep q_pcc got 85 % worse.

Scored properly:

| arm | TOTAL | v_rms_ts | q_pcc | pcc_underutil |
|---|---|---|---|---|
| `pq_k1` (shipped) | 1.697 | 0.380 | 0.638 | 0.057 |
| `pq_k0.5` | **1.621** | 0.321 | 0.895 | 0.001 |
| `pq_k0.3` | 1.646 | 0.273 | 1.058 | 0.000 |
| `pq_k0.2` | 1.762 | 0.257 | 1.181 | 0.000 |
| `th_k1` | 3.109 | 0.466 | 0.936 | **0.960** |
| `th_k0.5` | 2.215 | 0.369 | 0.945 | 0.256 |
| `th_k0.3` | **2.042** | 0.330 | 1.084 | 0.038 |
| `th_k0.2` | 2.186 | 0.307 | 1.319 | 0.000 |

`pq_k0.2` is **3.8 % WORSE** than shipped on the BO scalar, not 32 % better.
The voltage gain was bought at a price the tuner counts equally.

### The shipped tuning is validated

The PQ curve is U-shaped with its minimum at kappa = 0.5, only **4.5 %** below
the shipped point. So the BO landed within ~5 % of the best point on this
ladder. **No crisis in the published V1-V5 results, and the zone-2 alarm raised
earlier is withdrawn** -- zone 2's modest voltage tracking is a deliberate trade
against interface-Q, not an accident.

### And it identifies why Thevenin loses -- measured, not inferred

`pcc_underutil` is 0.960 for `th_k1` against 0.057 for `pq_k1`, a 17x gap that
dominates Thevenin's total, and it collapses as gain rises (0.960 -> 0.256 ->
0.038 -> 0.000). The TSO is UNDER-DISPATCHING the DSO interface: Thevenin's H
tells it there is less to gain, so declared capability sits unused. This is the
same under-drive seen in `set_move` (1.42 vs 1.86) and in the 007j interior
result, now visible as a named term in the tuner's own cost.

Thevenin's best (2.042) remains 26 % worse than PQ's best (1.621) on the correct
objective, so the frontier conclusion survives the change of metric.

## Drift vs type (`007l_DRIFT_VS_TYPE.py`)

Models built at 08:00, scored against the plant at 13:00. Mean over zones,
corridor rows:

| | frozen@t0 | refreshed@t1 |
|---|---|---|
| pq | 1.8649 | 1.8831 |
| th | 0.1080 | 0.1042 |

* error removed by fixing the TYPE (pq -> th, both frozen): **+1.757**
* error removed by REFRESHING pq: **-0.018** (marginally worse)
* error removed by REFRESHING th: **+0.004**

**Type dominates by roughly 100x. Drift is a non-issue over 5 h.**

The null is not degenerate: measured over the 5 h `pq_k1` run the plant moves
max 0.0227 pu end-to-end and up to 0.0352 pu per-bus range, so the operating
point genuinely travels 2-3.5 % and refreshing the anchor still buys nothing.

**The "build once and freeze" choice is vindicated.** Chapter06.tex:161-162
currently asserts it; it can now cite a measurement.

Caveat: the t1 plant is the profile-driven state from a fresh initialisation,
not the state a controlled 5 h run reaches, so this isolates profile drift and
excludes control-induced drift. Profile drift dominates over this horizon, so
the figure is a lower bound on total drift -- but it is 100x too small to
change the ranking.

## Best-tuned vs best-tuned (`007m_PERZONE_TUNE_COMPARE.py`) -- SETTLES IT

Every earlier comparison was confounded: the two boundaries imply different loop
gains and a scalar kappa cannot equalise a per-zone bias. This gives BOTH
boundaries the same per-zone freedom -- one scale per zone on `params.g_w`,
coordinate pass over the ladder (1.0, 0.6, 0.35, 0.2) -- and searches each to
its own optimum, scored on the BO's performance scalar. Search at 2 h, winners
verified at 5 h.

| | optimum scales (z1,z2,z3) | score @2 h | score @5 h |
|---|---|---|---|
| pq | 0.35, 0.35, 0.2 | **0.789** | **1.672** |
| thevenin | 0.2, 0.2, 0.2 | 1.065 | 2.158 |

**Thevenin remains +35 % (2 h) / +29 % (5 h) behind at each one's own optimum.**
PQ's advantage is therefore NOT an accidental gain boost -- the central open
question of the last several rounds -- and it survives giving the tuning full
freedom to compensate.

### The mechanism, finally identified

Breakdown at 5 h, each at its optimum:

| term | pq | thevenin |
|---|---|---|
| v_rms_ts | 0.278 | 0.308 |
| v_worst_ts | 0.165 | 0.271 |
| **v_band_ts** | **0.002** | **0.149** |
| q_pcc | 1.077 | 1.291 |
| **pcc_underutil** | **0.000** | **0.000** |

`pcc_underutil` is now ZERO for BOTH. The per-zone re-tuning eliminated the
under-dispatch, so **under-dispatch is refuted as the explanation** -- and the
gap survives anyway. What survives with it is `v_band_ts`: two orders of
magnitude worse under Thevenin (0.002 -> 0.149).

Output constraints are imposed on the linearised prediction `y + H w`, so they
are evaluated through `H` itself and NOT through the step weights -- no
re-tuning can touch them. An over-stated `H` over-states how far a candidate
step moves the monitored voltages, so the optimiser holds a wider margin from
the band than the linear prediction alone requires. **That margin absorbs the
model error and disturbance arriving between iterations.** The calibrated
boundary predicts truthfully, steps up to the band, and gets carried across.

So the over-statement is not a harmless error the feedback absorbs. It is an
**implicit back-off on the output constraints**, and removing it removes the
back-off. This also refutes the earlier speculation (2026-08-12) that a
gain-matched Thevenin should be BETTER near the bands because PQ is
"needlessly conservative" -- the conservatism is doing work.

### Consequence

Retain constant PQ, on engineering grounds rather than convenience. It is also
the cheaper of the two in information: PQ needs only the area's own boundary
measurements, while the calibrated form additionally needs an agreed
per-corridor impedance, which is NOT locally observable (identification from the
terminal (V,Q) locus is biased exactly when the neighbour regulates that
terminal, which under BRC-H it always does).

Left open, and it is a constraint-formulation question rather than a boundary
one: whether an explicit robustness margin on the constraint rows would let the
calibrated boundary be used without the band-excess penalty.

## Not changed / open

* The DSO-side boundary (`build_dso_local_net`) is untouched — it is already a
  voltage-pinned source, i.e. the PV convention.
* Single operating point (2016-01-05 08:00). How the error grows as the plant
  drifts from the linearisation point is a separate question.
* The leftover `SYNTH_TSO_TERTIARY_SHUNT` path in `build_tso_local_net` step 8
  is unrelated to this change but still duplicates the tertiary bank in the
  reduced net; the module docstring still describes the old
  tertiary-is-dropped behaviour. Not addressed here.
