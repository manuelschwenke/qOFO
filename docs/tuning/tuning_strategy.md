# Offline Bayesian Tuning of the Multi-Zone OFO Controller

> Thesis section draft.  Written to be lifted directly into the
> dissertation chapter on controller synthesis.  Notation follows the
> stability theory developed earlier in the thesis (Theorem 1.2 for
> per-DSO C1, Theorem 3.1 for multi-zone continuous C2, Theorem 3.2 for
> the discrete small-gain C3, Theorem 3.3 for the combined verdict).

## 1. Motivation: the conservatism gap

The closed-loop stability of the cascaded MIQP-OFO controller is
guaranteed by Theorem 3.3, which combines three sufficient conditions:

* **C1** -- per-DSO inner-loop stability,
  $\rho(\mathbf{M}_{\mathrm{cont},j}) < 1$;
* **C2** -- multi-zone TSO continuous stability,
  $\rho(\mathbf{M}_{\mathrm{full}}^c) < 1$;
* **C3** -- multi-zone TSO discrete small-gain,
  $\rho(\boldsymbol{\Gamma}) < 1$.

Each condition produces an *analytical floor* on the proximal
regularisation weights $g_{w,a}$ above which the contraction
certificate holds.  In particular, the discrete small-gain rule
(Corollary 3.2) gives, per zone $i$ and discrete actuator $a$,
$$g_{w,i,a} \;\geq\; 2\sum_{j \neq i} \big\|[\mathbf{P}_{ij}]_{a,\cdot}\big\|_{1}.$$
Above this floor, stability is *sufficient*: the certificate guarantees
that the iteration cannot diverge.  Below the floor the certificate is
silent --- but the controller may still operate stably and, crucially,
*more responsively*, because the proximal step is less heavily damped.

This sufficiency-versus-necessity gap is well known in OFO literature.
It motivates a two-stage tuning strategy: (i) compute the analytical
floor as an *upper bound* on the search, and (ii) explore the region
*below* the floor empirically, using high-fidelity closed-loop
simulation to choose the best operating point.

The remainder of this chapter describes the offline Bayesian-optimisation
(BO) procedure that implements this two-stage approach.  The full
implementation lives in the project's `tuning/` package; design
decisions are stated here in source-language form.

## 2. Decision space

The controller exposes nine continuous weights that materially affect
both control quality and closed-loop stability.  All other dataclass
fields of `MultiTSOConfig` are either structural choices (e.g. integer
step caps, slack-variable penalties) or run-time switches (live plots,
verbose logging) and are therefore not tuned.

| Symbol            | Field name        | Role                                          |
|-------------------|-------------------|-----------------------------------------------|
| $g_v$             | `g_v`             | TSO voltage tracking weight                   |
| $g_q$             | `g_q`             | DSO Q-interface tracking weight               |
| $g_v^{(\mathrm{DSO})}$ | `dso_g_v`    | DSO voltage tracking weight (secondary)       |
| $g_{w,\mathrm{DER}}$  | `g_w_der`     | TSO DER Q proximal regularisation             |
| $g_{w,\mathrm{PCC}}$  | `g_w_pcc`     | TSO Q-PCC setpoint regularisation             |
| $g_{w,\mathrm{OLTC}}^{(\mathrm{TSO})}$ | `g_w_tso_oltc` | TSO machine-trafo OLTC regularisation |
| $g_{w,\mathrm{shunt}}^{(\mathrm{TSO})}$ | `g_w_tso_shunt` | TSO MSC/MSR shunt regularisation     |
| $g_{w,\mathrm{DER}}^{(\mathrm{DSO})}$  | `g_w_dso_der`  | DSO DER Q proximal regularisation             |
| $g_{w,\mathrm{OLTC}}^{(\mathrm{DSO})}$ | `g_w_dso_oltc` | DSO interface-trafo OLTC regularisation       |

Each parameter is sampled in **logarithmic space** because the
controller responds to ratios rather than absolute values: doubling
$g_v$ has the same effect on the dimensionless eigenvalues
of $\mathbf{M}_{\mathrm{full}}^c$ as halving every $g_{w,a}$ on the
relevant block.  The lower bounds are chosen far below any
empirically reasonable value
($g_{w,a}^{\min} = 10^{-1}$, $g_v^{\min} = 10^{2}$,
$g_q^{\min} = 1$) so that BO can probe the high-aggressiveness
regime where the certificate is silent.

The integral Q-tracking parameters
(`dso_g_qi`, `dso_lambda_qi`, `dso_q_integral_max_mvar`) are excluded
from this thesis configuration, as is the generator-AVR weight
$g_{w,\mathrm{gen}}$, which is pinned at a deliberately conservative
$10^{7}$ (the AVR is locally stable by construction; this thesis does
not tune it).

## 3. Stage one: LMI ceiling extraction

Before the empirical search starts, the analytical floor is computed
once at the *baseline* operating point.  We run a single TSO step of the
closed-loop simulator and read back
$\mathbf{M}_{\mathrm{full}}^c$, $\boldsymbol{\Gamma}$, and the per-DSO
$\mathbf{M}_{\mathrm{cont},j}$ matrices from the existing
`analyse_multi_zone_stability` output.  From these we extract per-block
ceilings by exploiting the leading-order scaling

$$
\mathbf{M}_{ij}^c \;=\; \mathbf{G}_{w,i}^{-1/2}
   \big(\mathbf{R}_i + \mathbf{K}_{ii}^{\top}\mathbf{Q}_i\mathbf{K}_{ij}\big)
   \mathbf{G}_{w,j}^{-1/2}.
$$

Increasing every continuous $g_{w,a}$ by factor $f$ scales every entry
of $\mathbf{M}_{\mathrm{full}}^c$ by $1/f$, so its spectral radius
scales as $1/f$.  Conversely, scaling $g_v$ by $f$ scales $\mathbf{Q}$
by $f$ and therefore $\rho(\mathbf{M}_{\mathrm{full}}^c)$ also by $f$.
The threshold (where $\rho \approx 1$) is therefore

$$
g_{w,a}^{\mathrm{LMI}} \;=\; g_{w,a}^{\mathrm{baseline}}\cdot
   \rho\!\left(\mathbf{M}_{\mathrm{full}}^c\Big|_{\text{baseline}}\right),
\qquad
g_v^{\mathrm{LMI}} \;=\; g_v^{\mathrm{baseline}} \,\big/\,
   \rho\!\left(\mathbf{M}_{\mathrm{full}}^c\Big|_{\text{baseline}}\right).
$$

For the discrete actuators (TSO OLTCs and shunts) the analytical bound
is read directly from `c3_discrete.g_min_required`, which already
implements Corollary 3.2 with the row-sum norm of $\mathbf{P}_{ij}$ on
the chosen actuator.  We take the maximum across zones --- the worst
zone determines the effective ceiling.  For DSO OLTCs, no analytical
bound is available because Proposition 1.3 establishes finite-time
settlement of the discrete inner loop without a continuous LMI; the
ceiling is set to $\infty$ and BO falls back to a generous
`fallback_high` value.

The full ceiling vector
$\boldsymbol{\theta}^{\mathrm{LMI}} = (g_v^{\mathrm{LMI}},
g_{w,\mathrm{DER}}^{\mathrm{LMI}},\,\dots,\,g_{w,\mathrm{DER},\mathrm{DSO}}^{\mathrm{LMI}})$
is cached on disk keyed by a SHA-256 hash of
the operating-point-defining inputs (scenario string, fixed-zone flag,
voltage setpoint, the objective weights, slack penalties, and profile
configuration).  The $g_{w,*}$ values are *deliberately excluded* from
the hash because they are precisely the BO decision variables.

> **Approximation, Stage 1.5 (open).**  The isotropic-scaling argument
> above pretends a single block can be scaled in isolation without
> affecting the others.  In reality, scaling one $g_{w,a}$ touches
> only the corresponding rows and columns of $\mathbf{M}_{\mathrm{full}}^c$
> and gives a tighter per-block bound.  The current implementation
> documents the approximation in the returned `Ceilings.notes` and
> reverts to $\infty$ where uncertain; the refined per-block solve is
> filed as Task 1.5.

## 4. Stage two: empirical Bayesian optimisation

### 4.1 Algorithm

We use the Tree-structured Parzen Estimator (TPE) sampler
[Bergstra et al., 2011] in its multivariate variant with grouped
parameters (`optuna.samplers.TPESampler(multivariate=True, group=True)`).
TPE is well suited to expensive, derivative-free optimisation in the
$n \leq 10$ regime: the first
$N_{\mathrm{startup}} = 15$ trials are sampled from the search prior
(approximately Sobol-like in log-space) to seed the surrogate, after
which the bandit allocates new trials by maximising the expected
improvement of a kernel-density ratio estimator.  The optimisation is
*resumable*: trials are persisted in a SQLite RDB
(`results/tuning/studies.db`), and re-running with the same
`--study-name` continues from where the previous run left off.  A
typical thesis run uses $N = 80$ trials.

### 4.2 Inner loop: scenario-averaged cost

Each BO trial samples a parameter vector
$\boldsymbol{\theta} \in \Theta$ and evaluates it across a *design
set* of five deterministic scenarios designed to span the operating
envelope without redundant simulation:

1. **`nominal_quiet`** -- 60 min of unperturbed mid-load operation.
2. **`gen_trip_recovery`** -- 90 min, generator 5 trip at $t=20$
   min, restoration at $t=60$ min.
3. **`load_step`** -- 60 min, $+300\,\mathrm{MW}/+100\,\mathrm{Mvar}$
   load connection at bus 5 between $t=15$ and $t=45$ min.
4. **`dual_disturbance`** -- 120 min, simultaneous gen trip and
   load step at bus 27.
5. **`winter_peak`** -- 60 min, winter-evening loading
   (2016-01-14, 18:00).

The total simulated time is $\sim 6$ hours, chosen so that one BO
trial takes a few wall-clock minutes on a modern workstation.  Each
scenario produces a list of `MultiTSOIterationRecord` from which a
trajectory metric is computed (\S~5).  The scenarios are deliberately
chosen to be diverse but *cheap*: each disturbance is small enough to
be handled within the simulated horizon, allowing the steady-state
behaviour to be observed.

### 4.3 Across-scenario aggregation: CVaR-25

For each trial, the per-scenario costs $\{J_k\}_{k=1}^{5}$ are
combined into a single scalar via the conditional value-at-risk at
the 75th percentile:

$$
J_{\mathrm{agg}}(\boldsymbol{\theta})
   \;=\; \mathrm{CVaR}_{0.75}\big(\{J_k\}\big)
   \;=\; \mathbb{E}\big[J \mid J \geq Q_{0.75}(J)\big].
$$

Concretely, with five scenarios, $J_{\mathrm{agg}}$ is the mean of the
two largest costs.  This choice lies between two natural extremes:

* The **mean** is too optimistic: a single bad scenario can be
  averaged out by four nominal ones, leaving the BO search blind to
  fragility.
* The **maximum** is too noisy: a single outlier --- which may be a
  rare combination of scenario and parameter --- dominates the
  signal.

The CVaR aggregate is the standard robust-optimisation choice and
matches the thesis's "safe across the operating envelope" goal.  It
is implemented in `tuning.objective.cvar_aggregate`; the percentile is
exposed via `--cvar-pct` so that the choice can be sensitivity-tested
in future work.

## 5. Trajectory metrics and composite cost

For each scenario we extract a single scalar $J_k$ from the
`MultiTSOIterationRecord` log via `tuning.metrics.extract_metrics`.
The composite cost is

$$
J_k \;=\; w_v\,(\widetilde{\mathrm{ITAE}}_v^{\mathrm{TS}}
                + \widetilde{\mathrm{ITAE}}_v^{\mathrm{DS}})
       \;+\; w_q\,\widetilde{\mathrm{ITAE}}_q^{\mathrm{PCC}}
       \;+\; w_{\mathrm{osc}}\,\widetilde{N}_{\mathrm{osc}}
       \;+\; w_{\mathrm{tap}}\,\widetilde{N}_{\mathrm{tap}}
       \;+\; w_{\mathrm{viol}}\,\widetilde{N}_{\mathrm{viol}}
       \;+\; w_{\mathrm{pf}}\,N_{\mathrm{pf-fail}},
$$

where each $\widetilde{(\cdot)}$ denotes a normalised quantity with
$\mathcal{O}(1)$ magnitude under nominal conditions.  The components
are:

* **Tracking error** -- integral of time-weighted absolute error
  $\mathrm{ITAE}(t,e) = \int t\,\lvert e(t)\rvert\,dt$ on (i) the
  TSO mean-zone voltage error, (ii) the DSO mean-group voltage
  error, and (iii) the DSO Q-interface tracking error
  $\sum_{\mathrm{DSO}}\lvert Q_{\mathrm{set}} - Q_{\mathrm{actual}}\rvert$.
  ITAE is preferred over ISE because it discounts initial transients
  (which any controller exhibits at $t=0$) and emphasises late-time
  behaviour.

* **Constraint health** -- the per-step count of zones / DSO groups
  whose voltage envelope falls outside $[0.95, 1.05]$ p.u.,
  aggregated over the trajectory.  Violations are weighted heavily
  ($w_{\mathrm{viol}} = 10$) because they represent operational
  failures rather than performance shortfalls.

* **Actuator activity** -- per-class oscillation count
  $N_{\mathrm{osc}}$, defined as the number of sign changes in
  $\Delta u(k)$ where both $\lvert\Delta u(k)\rvert$ and
  $\lvert\Delta u(k-1)\rvert$ exceed a *noise floor*.  Floors are
  per-actuator-class (e.g. $5\,\mathrm{Mvar}$ for DER, $10^{-3}$
  p.u. for $V_{\mathrm{gen}}$) so that bona fide steady-state noise
  does not register as oscillation.

* **Switching count** -- the total $\sum_k\lvert\Delta\tau_k\rvert$
  for OLTC / shunt taps, which captures discrete-actuator wear.

* **Power-flow failure** --- a heavy ($w_{\mathrm{pf}} = 10^{3}$)
  penalty for any record reporting a non-finite voltage.  An empty
  log (simulator divergence at $t=0$) returns a sentinel cost
  $J = 10^{3}$ so that BO drives away from this region without
  ever crashing.

The cost weights $\boldsymbol{w}$ are *meta-parameters distinct from
the controller weights*.  They encode the engineer's relative
preference between competing objectives and are not subject to
BO; defaults are chosen so that
$w_{\mathrm{pf}} \gg w_{\mathrm{viol}} > w_{\mathrm{osc}} >
 w_{\mathrm{tap}} \geq w_v$ at unit normalised metrics.

In addition to the cost, several *diagnostic* metrics are recorded
but not minimised: the empirical contraction percentile
$\widehat{\rho}_{p95} = Q_{0.95}\big(\{\,\text{LHS}_{\text{contraction},i}(t)\,\}\big)$,
the mean active-power loss, and the per-actuator-class counts.
These appear in the validation report (\S~7) for post-hoc
diagnosis.

## 6. Robustness: graceful failure handling

Every BO sample is a candidate parameter vector; some will produce
unstable closed-loop dynamics, MIQP infeasibility, or pandapower
power-flow failure.  The runner contract is therefore:
**`run_one` never raises**.  Any exception during simulation is caught,
recorded in `RunResult.failure_reason`, and a sentinel
`TrajectoryMetrics` with $J = 10^{3}$ is returned.  This guarantees
that BO never crashes on a bad sample and instead receives a strong
signal to avoid the offending region.  The only exceptions
deliberately *not* caught are validation errors from
`apply_to_config` (e.g. NaN in a parameter), which signal programmer
bugs rather than poor BO samples.

This design has been verified end-to-end: in the test
`test_run_one_handles_bad_params_gracefully`, a deliberately
pathological $g_{w,\mathrm{DER}} = 10^{-6}$ is fed into the runner;
the simulator raises, the runner returns a sentinel, and the test
asserts $J \geq 10^{3}$ and `pf_failures` $\geq 1$.

## 7. Validation and the certificate-ratio diagnostic

After the BO loop converges, the tuned vector
$\boldsymbol{\theta}^{\star}$ is written to YAML and passed through a
*post-hoc* validation phase: $N_{\mathrm{val}} = 200$ randomised
scenarios drawn from a reproducible RNG-seeded distribution
(`tuning.scenarios.validation_set`) covering the full annual
calendar and three duration buckets ($30/60/90$ minutes), with
$0$--$2$ contingencies per scenario and an 80/20 split between the
`wind_replace` and `base` network configurations.

Two artefacts emerge:

1. The **tuning report** (HTML).  Its central diagnostic is the
   *certificate-ratio table*, which lists, for each of the nine BO
   parameters,
   $$
   r_a \;=\; \frac{g_{w,a}^{\mathrm{LMI}}}{g_{w,a}^{\star}},\qquad
   r_v \;=\; \frac{g_v^{\star}}{g_v^{\mathrm{LMI}}}.
   $$
   $r_a$ quantifies the *conservatism gap*: $r_a \gg 1$ means the
   empirically-optimal regulariser is much smaller than the
   sufficient bound, i.e. the certificate is loose.  $r_a \approx 1$
   would mean BO converged toward the LMI bound itself, indicating
   that the analytical guarantee is in fact necessary.  The
   distribution of $\{r_a\}$ across actuator classes is the central
   thesis-defensible result of this chapter.

2. The **validation report** (HTML), summarising the tuned
   controller's behaviour over $N_{\mathrm{val}}$ unseen scenarios:
   the histogram of $\widehat{\rho}_{p95}$ (sustained empirical
   contraction below unity is the empirical analogue of C2), the
   distribution of $J_k$, the per-class oscillation and switching
   counts, and a breakdown of any scenarios that failed
   power-flow or hit the soft sentinel.

A successful tuning run is characterised by:

* Tuning report: $r_a > 1$ for every BO weight (the certificate is
  not tight) AND no recent best-trial improvement (BO has converged).
* Validation report: zero PF failures across all
  $N_{\mathrm{val}}$ scenarios, and
  $\widehat{\rho}_{p95}^{(\text{p95 over scenarios})} < 1$
  (sustained contraction across the operating envelope).

## 8. Limitations and open points

* **Approximate per-block ceilings.**  The Stage-1 ceiling extraction
  uses an isotropic scaling that is conservative.  Refining it to a
  per-block solve (Task 1.5) would tighten the upper bound on the BO
  search space and improve sample efficiency.

* **Stationary design set.**  The five design scenarios are
  deterministic by intent, but their power profiles still come from
  the SimBench `mv_rural_qload` time series.  A BO run that
  inadvertently overfits to this profile would only become apparent
  in the validation phase; if so, the design set should be
  rotated or augmented.

* **Single-objective scalarisation.**  The composite cost reduces a
  multi-objective problem (tracking, oscillation, switching) to one
  scalar via fixed weights $\boldsymbol{w}$.  This works when the
  Pareto frontier is well-behaved but masks trade-offs at the
  extremes.  A multi-objective BO formulation (e.g.
  `optuna.multi_objective`) returning $(\mathrm{ITAE}_v, N_{\mathrm{osc}})$
  pairs is straightforward to retrofit if the scalar approach proves
  inadequate.

* **Offline only.**  The methodology is deliberately offline: the
  controller is tuned once, deployed, and not adapted online.
  Online adaptation would couple controller dynamics with parameter
  dynamics on the same timescale, breaking the time-scale separation
  that the cascade depends on.  The thesis does not address online
  re-tuning.

* **Cost-weight calibration.**  The default weights $\boldsymbol{w}$
  in `CostWeights` are calibrated by physical reasoning, not
  data-driven validation.  The first thesis-quality BO run is
  expected to expose mis-calibration (e.g. $w_{\mathrm{tap}}$ too
  small relative to $w_{\mathrm{osc}}$, leaving residual chattering),
  after which they should be re-calibrated.

## 9. Reproduction recipe

The following commands reproduce a thesis-quality tuning run:

```bash
# 1.  Persist the chosen baseline once.
python -c "
from pathlib import Path
from tuning._io import save_config_yaml
from configs.multi_tso_config import MultiTSOConfig
save_config_yaml(MultiTSOConfig(), Path('configs/baseline.yaml'))
"

# 2.  Tune (overnight; ~80 trials × 5 scenarios ≈ 8 h on the lab box).
python -m tuning.tune \
    --baseline configs/baseline.yaml \
    --n-trials 80 --study-name v1 \
    --storage sqlite:///results/tuning/studies.db \
    --output configs/tuned_params.yaml \
    --report results/tuning/tuning_report.html

# 3.  Validate (post-tune, ~200 scenarios).
python -m tuning.validate \
    --params configs/tuned_params.yaml \
    --baseline configs/baseline.yaml \
    --n-scenarios 200 --seed 42 \
    --report results/tuning/validation_report.html
```

If the validation reveals a class of scenarios that the design set
missed, append them to `tuning.scenarios.design_set()` and re-run the
tune step --- Optuna resumes from the existing study, so no trials are
wasted.

The thesis-defensible artefacts are then `tuning_report.html`
(certificate ratios) and `validation_report.html`
($\widehat{\rho}_{p95}$ distribution).  The two together close the
loop between Theorems 1.2 / 3.1 / 3.2 / 3.3 (sufficient bounds) and
the empirically chosen operating point.
