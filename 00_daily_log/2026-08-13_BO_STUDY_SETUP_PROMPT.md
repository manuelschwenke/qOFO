# Prompt: set up and run the controller tuning study

Paste the block below to the agent on the always-on machine.

---

## Task

Run a Bayesian-optimisation tuning study for the cascaded multi-TSO / multi-DSO
OFO controller in `qOFO_GH`, using the existing `tuning/` machinery. Do not
build a new tuner — configure and launch the one that is there, then report.

## Background you need

The controller weights carry an **exact redundant direction**: scaling every
objective weight, every `g_w`, every `g_z` and the shunt integrator's gain by a
common factor reproduces the trajectory to ~4e-10, including the integer OLTC
tap sequence. This was measured on 2026-07-31 and is documented at the top of
`tuning/reparam.py`. Searching raw weights therefore wastes budget on a
direction that provably changes nothing, and an earlier raw-weight box also
excluded the known-good hand-tuned point (`g_v = 1e7` against a box of
`[1e2, 1e5]`).

`tuning/reparam.py` fixes both: it pins a **gauge** at a reference config and
searches dimensionless ratios about it. Use that space (`--reparam`), not the
legacy raw-weight one.

The four coordinates are:

| coordinate | range | meaning |
|---|---|---|
| `tso_lambda` | 0.05–1.20, linear | TSO loop gain, as target `lambda_max(M)` over continuous columns. ~0.9 is well-damped, 2.0 is the hard OFO bound. |
| `dso_lambda` | 0.05–1.20, linear | same for the DSO layer |
| `tau_der_pcc` | 1/64–64, log | relative damping of DER vs PCC inside the TSO block; 1.0 = analytic column-norm preconditioner |
| `dso_v_priority` | log window | DSO voltage-schedule vs interface-Q trade-off, as a multiple of the reference `dso_g_v` |

`g_v` and `g_q` are **the gauge** and are pinned on purpose — that is what
quotients out the redundancy. The objective trade-off is still tuned, through
`dso_v_priority`. Do not add `g_v`/`g_q` back as free dimensions.

## Steps

1. **Pin the baseline first.** `experiments/run_multi_system_ofo.py`'s
   `make_config()` has been edited repeatedly during experiments (`g_w_pcc`
   moved 80 -> 150 -> 80 -> 150 within one day). Commit it before starting and
   record the commit hash; a study whose baseline moves mid-run is
   uninterpretable. The last known-good snapshot is
   `results/007_tie_boundary/BEST_SO_FAR_2026-08-13.params.json`
   (Thevenin boundary, `zone_g_w_scale` uniform 0.3, `g_v=1e7`, `g_q=250`,
   `g_w_pcc=150`, git `b97badb`).

2. **Read the CLI** — `python tuning/tune.py --help`. Confirm how the baseline
   config is selected and which scenario set is used; do not assume. With
   `--reparam` the scenario set defaults to `tune_v2`.

3. **Check the reference is sane before spending compute.** The gauge is taken
   from the baseline config via `Gauge.from_config`, and `tune.py` prints the
   pinned gauge and the reference priorities at startup. Verify those lines
   look right. In particular:
   - `tso_g_q_pcc` is **zero** in the reference, i.e. the TSO interface-Q
     objective is off, so there is no TSO-side objective trade-off in the
     search. If a TSO voltage-vs-interface trade-off should be tuned, that
     requires switching `tso_g_q_pcc` on and adding a coordinate — see the
     "Deliberately absent" block in `tuning/reparam.py`. Decide before
     launching, not after.
   - `shunt_int_g_w` belongs to the gauge group, so shunt engagement behaviour
     is inherited from the reference and is **not** tuned. Shunts have not been
     engaging; if they should, fix `shunt_int_g_w` (currently 150; smaller
     commits sooner), `shunt_int_delta_mvar` (10.0) and `shunt_int_t_dwell_s`
     (1800 s) in the reference first. Note `g_w_tso_shunt` is inert while
     `shunt_dispatch="integrator"` — it is only read on the MIQP path.

4. **Decide the boundary equivalent and state it.** Tuning is per-boundary; the
   optimum differs measurably between `tie_boundary_equivalent="pq"` and
   `"thevenin"`. Tune the one that will ship. If both are wanted, run two
   studies with different `--study-name`, not one mixed study.

5. **Launch**, e.g.

       python tuning/tune.py --reparam --study-name <name> --n-trials 200 --seed 1

   Optuna persists the study, so it is resumable; prefer several shorter
   invocations over one long one. Budget roughly 3-6 min per trial at a 2 h
   horizon.

6. **Watch for bound-hugging.** `tune.py` warns when the best trial sits on a
   box edge. An optimum on the boundary is not an optimum — it means the box is
   binding. Report it rather than accepting the point. (This exact mistake
   invalidated an earlier comparison: a per-zone search reported a 29 % gap that
   fell to 3 % once the ladder was extended past the edge.)

## Report back

- study name, git commit of the pinned baseline, number of trials, seed
- best feasible trial: coordinates, resulting config weights, objective value
  and its term breakdown
- whether any coordinate hugged a bound
- constraint-violation rate across trials (how much of the box is infeasible)
- the resolved config of the best trial, saved as JSON alongside the study

## Guardrails

- Do not edit `make_config()` or any config factory while the study runs.
- Save the resolved parameters with every result —
  `experiments/helpers/run_params.py::dump_params` does this and handles tuple
  keys, numpy types and nested dataclasses.
- Do not tune against `rms_v_ts` alone. The scoring scalar in
  `tuning/objectives_v2.py` weights TS voltage and interface-Q equally, plus
  worst-bus, band excess, DS voltage and PCC under-utilisation. Optimising one
  term produced a spurious "32 % headroom" earlier that vanished when scored
  properly.
- Feasibility is handled by Optuna constraints, not by penalty terms in the
  scalar. Do not fold constraints back into the objective.
