# Stability-observer Pareto sweep over $(g_v, g_q)$

## Setup
6 runs of `experiments/000_M_TSO_M_DSO.py` `run_multi_tso_dso` at the canonical scenario-B config (otherwise identical to `main()` line 2079: g_v/g_q overridden, all g_w left at the smoke-run values, no contingencies, 30 min horizon, profiles enabled, IEEE 39 with `wind_replace`).

Grid:
- $g_v \in \{1500,\ 15000,\ 150000\}$
- $g_q \in \{20,\ 200\}$
- 11 observer snapshots per zone per run (TSO cadence = 3 min).

Artefacts:
- Per-run observer JSON / PNG / markdown under `results/observer_pareto_sweep/gv{gv}_gq{gq}/`.
- Aggregate plot: `analysis/observer/pareto_sweep_gv_gq.png`.
- Sweep driver: `_tmp_observer_pareto.py` at the repo root.

## Theoretical prediction

The observer's spectral matrix is

$$
M = g_v \cdot M_V + g_q \cdot M_Q, \qquad M_V = H_V^{\!\top} W_V^{2} H_V, \qquad M_Q = H_Q^{\!\top} W_Q^{2} H_Q.
$$

`stability_observer._split_H` currently returns $H_Q = 0_{0\times m}$ (the Q-gen rows are not yet exposed by the controller; see scope note in the docstring). Therefore $M_Q = 0$ and

$$
M(g_v) = g_v \cdot M_V \quad \Longrightarrow \quad g_w^{\min}(g_v) = g_v \cdot g_w^{\min}(1).
$$

In log-log:

$$
\log_{10}\,g_w^{\min} = 1 \cdot \log_{10} g_v + \mathrm{const} \qquad (\text{slope} = 1, \ g_q\text{-axis degenerate}).
$$

## Empirical result

Linear fit of $\log_{10} g_w^{\min,\text{p95}}$ vs $\log_{10} g_v$, per zone × block × $g_q$:

| Zone | Block | $g_q$ | slope | $R^2$ | n |
|---|---|---:|---:|---:|---:|
| 1 | DER | 20 | 0.999 | 1.000 | 3 |
| 1 | DER | 200 | 0.999 | 1.000 | 3 |
| 1 | V_gen | 20 | 0.999 | 1.000 | 3 |
| 1 | V_gen | 200 | 0.999 | 1.000 | 3 |
| 1 | OLTC | 20 | 0.999 | 1.000 | 3 |
| 1 | OLTC | 200 | 0.999 | 1.000 | 3 |
| 2 | DER | 20 | 1.000 | 1.000 | 3 |
| 2 | DER | 200 | 0.998 | 1.000 | 3 |
| 2 | PCC | 20 | 1.000 | 1.000 | 3 |
| 2 | PCC | 200 | 0.998 | 1.000 | 3 |
| 2 | V_gen | 20 | 1.000 | 1.000 | 3 |
| 2 | V_gen | 200 | 0.998 | 1.000 | 3 |
| 2 | OLTC | 20 | 1.000 | 1.000 | 3 |
| 2 | OLTC | 200 | 0.998 | 1.000 | 3 |
| 3 | DER | 20 | 0.999 | 1.000 | 3 |
| 3 | DER | 200 | 0.999 | 1.000 | 3 |
| 3 | PCC | 20 | 0.999 | 1.000 | 3 |
| 3 | PCC | 200 | 0.999 | 1.000 | 3 |
| 3 | V_gen | 20 | 0.999 | 1.000 | 3 |
| 3 | V_gen | 200 | 0.999 | 1.000 | 3 |
| 3 | OLTC | 20 | 0.999 | 1.000 | 3 |
| 3 | OLTC | 200 | 0.999 | 1.000 | 3 |

All 22 fits sit within $0.998 \le \text{slope} \le 1.000$ at $R^2 = 1.000$.

## Interpretation

- **Slope $\approx 1.0$ and $R^2 = 1.0$** confirm the observer's `_assemble_M` and `compute_min_gw_per_block` chain is internally consistent: the $g_v$ pre-factor passes through cleanly, dominant-eigenvalue scaling is linear, and the per-block ratio-priors preserve the $\text{OLTC} > V_\text{gen} > \text{PCC} > \text{DER}$ ordering across the full $g_v$ range.
- **The dashed ($g_q=20$) and solid ($g_q=200$) markers coincide pixel-perfect on the plot.** With the current $H_Q = 0$ split, $g_q$ has zero observable effect — exactly as predicted. This will change once Feature A (Q_gen as soft-constrained TSO output) populates $H_Q$; the plot then becomes a true 2-D Pareto with non-degenerate $g_q$ axis. **This sweep is the regression baseline against which the post-Feature-A observer should be checked.**
- **Block magnitudes for scenario B at $g_v = 150{,}000$** match the smoke run (Z2 OLTC $\approx 5.2 \cdot 10^6$, Z3 OLTC $\approx 4.2 \cdot 10^6$ etc.) → repeatability confirmed across runs.

## Caveat

Per the prompt's section §3.1, this sweep does **not** prove anything about the closed-loop OFO stability. It only verifies the observer's tuning-math pipeline is consistent. The empirical contraction rate $\rho_k$ (Phase 3.2.4) is the actual controller-level evidence and is still pending.
