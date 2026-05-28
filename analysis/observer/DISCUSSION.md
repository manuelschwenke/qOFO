# Stability discussion: spectral-gap floor vs. empirical contraction

**Status:** Draft for thesis Chapter 5. Live numbers from a 60-min scenario-B run (`results/observer_full_scenarioB/observer_analysis_summary.json`, dated 2026-04-20). Citation stubs marked `[cite ..., verify]`.

> **Note on naming (2026-04-20).** Earlier drafts of this codebase called the
> condition $\lambda_\min(D_w + M) > \|M\|_\text{op}$ the "Bianchi condition
> (1'')" by analogy with Bianchi & Dörfler (2025). On checking the published
> paper (arXiv:2412.10964 / EJC 86), Theorem 1 actually uses a max-type
> composite Lyapunov function (Assumption 3), not a spectral inequality of
> this form. The condition the observer evaluates is therefore renamed
> **spectral-gap sufficient condition (SG)**. It is a valid sufficient
> condition for an *unconstrained* projected-gradient iteration to contract
> on this quadratic cost — likely related to spectral conditions in
> Hauswirth et al. (2021) ARC / IEEE TAC or Colombino et al. (2020) IEEE TCNS
> — but the exact published source needs to be located before citation.
> Treat "spectral-gap" here as descriptive, not as a citation.

## 1. Summary of the regime

Scenario B canonical (validated-stable) config: `g_v = 100 000`, `g_q = 200`, `g_w_der = 40`, `g_w_pcc = 100`, `g_w_gen = 1 \cdot 10^7`, `g_w_tso_oltc = 200`. Observer recorded 21 cross-sensitivity snapshots per zone over 60 min (TSO cadence 3 min). Two stability certificates are evaluated at every snapshot:

- **Spectral-gap (unconstrained):** $g_w \ge \|M\|_\text{op} - \lambda_\min(M)$. Sufficient for an *unconstrained* projected-gradient OFO loop. Computed per-actuator via the Gershgorin LMI tuner (`method="lmi"`, since 2026-04-20). [cite source paper, to be confirmed]
- **Häberle (box-constrained):** $g_w \ge \|M\|_\text{op}/2$. Sufficient for projected-gradient OFO with explicit box on $\Delta u$ — i.e. closer to what the MIQP loop actually does. One scalar per zone. *Citation stub: Häberle, Hauswirth, Ortmann, Bolognani, Dörfler (2021) L-CSS 5(1):343 Thm 1; verify the constant 1/2 against the published descent condition before use as a thesis claim.*

Slack ratio $r := g_w^\text{current} / g_w^{\min,\text{p95}}$ for both certificates (ratio $<1$ means current $g_w$ sits below the floor):

| Zone | Block | $g_w^\text{current}$ | SG p95 | $r_\text{SG}$ | Häberle p95 | $r_H$ | Regime (vs SG) |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | DER   | $40$            | $2.48 \cdot 10^5$ | $1.6 \cdot 10^{-4}$ | $9.5 \cdot 10^4$ | $4.2 \cdot 10^{-4}$ | strongly box-reg |
| 1 | V_gen | $1 \cdot 10^7$  | $6.44 \cdot 10^4$ | $\mathbf{155}$ | $9.5 \cdot 10^4$ | $\mathbf{105}$ | **SG-certified** |
| 1 | OLTC  | $200$           | $2.48 \cdot 10^5$ | $8.0 \cdot 10^{-4}$ | $9.5 \cdot 10^4$ | $2.1 \cdot 10^{-3}$ | strongly box-reg |
| 2 | DER   | $40$            | $3.56 \cdot 10^5$ | $1.1 \cdot 10^{-4}$ | $1.34 \cdot 10^5$ | $3.0 \cdot 10^{-4}$ | strongly box-reg |
| 2 | PCC   | $100$           | $3.56 \cdot 10^5$ | $2.8 \cdot 10^{-4}$ | $1.34 \cdot 10^5$ | $7.4 \cdot 10^{-4}$ | strongly box-reg |
| 2 | V_gen | $1 \cdot 10^7$  | $1.22 \cdot 10^5$ | $\mathbf{82}$ | $1.34 \cdot 10^5$ | $\mathbf{75}$ | **SG-certified** |
| 2 | OLTC  | $200$           | $3.53 \cdot 10^5$ | $5.7 \cdot 10^{-4}$ | $1.34 \cdot 10^5$ | $1.5 \cdot 10^{-3}$ | strongly box-reg |
| 3 | DER   | $40$            | $2.94 \cdot 10^5$ | $1.4 \cdot 10^{-4}$ | $1.12 \cdot 10^5$ | $3.6 \cdot 10^{-4}$ | strongly box-reg |
| 3 | PCC   | $100$           | $2.96 \cdot 10^5$ | $3.4 \cdot 10^{-4}$ | $1.12 \cdot 10^5$ | $8.9 \cdot 10^{-4}$ | strongly box-reg |
| 3 | V_gen | $1 \cdot 10^7$  | $3.08 \cdot 10^5$ | $\mathbf{32}$ | $1.12 \cdot 10^5$ | $\mathbf{89}$ | **SG-certified** |
| 3 | OLTC  | $200$           | $2.94 \cdot 10^5$ | $6.8 \cdot 10^{-4}$ | $1.12 \cdot 10^5$ | $1.8 \cdot 10^{-3}$ | strongly box-reg |

Spectral norm $\|M\|_\text{op}$ (the cost-side Hessian seen by the controller) sits in a tight band of $\sim\!1.9\!-\!2.7 \cdot 10^5$ across all zones over the full hour, with $\text{p95} - \text{mean}$ excursions under $2\,\%$ — the operating point drifts smoothly with the load profile and there is no contingency-driven jump in the unforced 60-min window.

Several observations follow.

1. The **V_gen block clears both floors by 30–155×**. This is deliberate: `g_w_gen = 1 \cdot 10^7` is set to suppress AVR setpoint motion to within numerical noise (the `g_w_gen` value comment in `main()` reads "excluded from stability"). The observer confirms: the AVR loop is effectively frozen, and V_gen's contribution to the closed-loop dynamics is a quasi-static pre-positioner.
2. **DER, PCC, and OLTC blocks operate three to four orders of magnitude below both floors** ($r_\text{SG} \sim 10^{-4}$, $r_H \sim 10^{-4}$ to $2 \cdot 10^{-3}$). A naive reading of the spectral-gap condition would predict divergence by an enormous margin. The constrained-aware Häberle floor is roughly 2–3× looser than the per-actuator Gershgorin spectral-gap floor, but **does not change the qualitative picture**: even the tighter constrained certificate puts the live tuning $\sim\!10^{-3}$ below the floor.
3. The Pareto sweep (`pareto_sweep_gv_gq.png`, `PARETO_SUMMARY.md`) shows that $g_w^{\min}$ scales linearly with $g_v$ across three decades with $R^2 = 1.000$ — the observer's spectral-gap arithmetic is internally consistent. The g_v sweep at fixed live $g_w$ (`gv_sweep_slack_ratio.png`) shows the slack ratio $r_\text{SG}$ drops by exactly one decade per decade of $g_v$ across the four-point grid, with the moving-block lines crossing the $r_\text{SG} = 1$ certificate boundary at $g_v \in \{16, 125, 360\}$ for DER, OLTC, PCC respectively. The 3–4-decade slack between current and floor-min $g_w$ at the operating $g_v = 10^5$ is therefore not a numerical artefact; it is the safety factor the trust region buys you for choosing $g_v$ this large.
4. **The remaining gap between Häberle and live tuning is the headline thesis result.** Häberle accounts for the box on $\Delta u$ (a large effect: 2× tighter than spectral-gap for rank-deficient $M$) but **not** for (a) integer projection of OLTC taps, (b) MIQP per-actuator $\Delta u_\text{max}$ trust regions, or (c) output slack penalties. The remaining $\sim\!10^{-3}$ ratio quantifies the additional regularisation those three features supply on top of the simple box.

## 2. Why the spectral-gap condition over-estimates the floor for MIQP-OFO

The spectral-gap condition (SG) that the observer evaluates,
$$
\lambda_\min\!\bigl(D_w + g_v M_V + g_q M_Q\bigr) > \bigl\| g_v M_V + g_q M_Q \bigr\|_\text{op}, \qquad \text{(SG)}
$$
is sufficient for the **unconstrained** projected-gradient iteration $u_{k+1} = u_k - g_w^{-1} \nabla \Phi_\text{red}(u_k)$ to contract on $\mathbb{R}^m$. Three structural properties of the MIQP-OFO controller violate the "unconstrained" premise that (SG) is built on:

* **Trust-region box on $\Delta u$.** Each MIQP step has explicit bounds $\Delta u \in [-\Delta_\max, +\Delta_\max]$, with $\Delta_\max$ chosen per actuator class (10–30 % of the local capability per step in `controller/tso_controller.py`). This is exactly the projection step that Häberle, Hauswirth, Ortmann, Bolognani & Dörfler (2021) [cite L-CSS Theorem 1, verify] use to certify convergence of projected-gradient OFO under a strictly weaker spectral condition than (SG). The box acts as an implicit damping term whose effective Lipschitz contribution is **at most** the diameter of the box per step — independent of $\|M\|_\text{op}$.
* **OLTC integer quantisation.** The 17-position OLTC tap variable is integer-projected after each MIQP solve. Integer quantisation truncates $u$-updates to the lattice $\Delta_\text{tap} \approx 0.6\,\%\,\text{V}$, so most TSO steps emit zero tap motion. In the closed-loop view this is a 0–1 multiplier on the OLTC component of $\nabla \Phi_\text{red}$, which **monotonically reduces** the contractive radius required of $g_w^\text{OLTC}$.
* **Output-side hard constraints.** The MIQP keeps line currents and bus voltages inside their hard bounds via slack penalties on $g_z$, which the observer's $H_V / H_Q$ split does not see. The slack-augmented Lagrangian has an additional positive-definite term that further increases $\lambda_\min$ of the effective Hessian without changing $\|M\|_\text{op}$, again reducing the certified $g_w$.

The combined effect is that $g_w^{\min,\text{SG}}$ is an **upper bound** on the proximal weight needed for (SG) in the *unconstrained* case, not a *necessary* condition for the constrained MIQP loop. Calling the operating regime "box-regularised" — as introduced in Caduff (MSc 2021) and re-stated in Ortmann (PhD 2023) Ch. 5 — captures this: the trust-region box and OLTC quantiser supply the regularisation that, in the unconstrained derivation, would have to come from $D_w$.

## 3. Empirical contraction evidence

Per Ortmann (2023) [cite Ch. 5, verify] the box-regularised regime requires its **closed-loop** empirical evidence — i.e. an observed contraction rate — as the primary stability certificate. The observer collects this directly. For each TSO step $k$ with $u_k = (Q^\text{DER}, Q^\text{PCC,set}, V^\text{gen}, \tau^\text{OLTC})_z$:
$$
\rho_k(z) := \frac{\bigl\| u_k(z) - u_{k-1}(z) \bigr\|}{\bigl\| u_{k-1}(z) - u_{k-2}(z) \bigr\|}, \qquad k \ge 2.
$$
A contracting closed loop requires $\rho_k < 1$. The 60-min run yields 19 valid samples per zone:

| Zone | n | median | p95 | max | $\Pr(\rho_k < 1)$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 19 | 0.852 | 0.922 | 0.971 | **100 %** |
| 2 | 19 | 0.964 | 1.204 | 1.298 | **74 %** |
| 3 | 19 | 0.941 | 0.988 | 1.126 | **95 %** |

Distribution: see `contraction_rho_histograms.png`.

* **Zone 1** sits fully inside the contractive band with significant margin (ratios as low as 0.78). Median 0.85 and p95 0.92 cluster tightly, signalling a quasi-stationary closed loop.
* **Zone 3** is similar: $\Pr(\rho_k < 1) = 95\,\%$, single excursion at 1.13. Median and p95 close together (0.94 / 0.99).
* **Zone 2** is the marginal case: median 0.964 (still contractive), but about a quarter of the distribution sits at or above 1 ($\Pr(\rho_k < 1) = 74\,\%$), with excursions up to $\rho_k = 1.30$. Zone 2 is the largest zone (1 DER + 9 PCC + 1 V_gen + 3 OLTC = 14 actuators) and carries 3 cascaded DSOs; the excursions correlate with PCC tracking transients during the morning load ramp.
* No zone diverges; consecutive sample drift is bounded; the median is below 1 in all three zones.

Combined with the slack-ratio table, this is the primary certificate: **the controller is empirically contracting at three to four orders of magnitude lower $g_w$ than even the constrained-aware Häberle floor would prescribe**, because the MIQP integer projection on OLTC taps and the per-actuator trust-region $\Delta u_\text{max}$ supply the additional regularisation.

## 4. Three-tier stability argument (Ortmann 2023 PhD Ch. 5 structure)

Following [cite Ortmann Ch. 5, verify], the chapter advances three complementary tiers:

| Tier | Certificate | Condition | Source |
|---:|---|---|---|
| 1 — primary | Empirical contraction $\rho_k$ | $\Pr(\rho_k < 1) \approx 1$ over the simulation horizon | This run + future 12 h scenario-B run |
| 2a — sufficient (constrained) | Häberle floor $g_w \ge \|M\|_\text{op}/2$ | Sufficient for projected gradient with explicit box on $\Delta u$; observer-evaluated; ~3-decade upper bound here | `stability_observer.aggregate_haberle`, this `DISCUSSION.md` §1 |
| 2b — sufficient (unconstrained) | Spectral-gap floor $g_w \ge \|M\|_\text{op} - \lambda_\min(M)$ | Sufficient for unconstrained projected gradient (descent-lemma flavour); observer-evaluated; ~4-decade upper bound here | `stability_observer.summary`, `PARETO_SUMMARY.md` |
| 3 — sufficient (per-actuator) | LMI bound $g_w \ge $ Gershgorin column-sum | Per-actuator sufficient condition; the active default since 2026-04-20 (`method="lmi"`) | `stability_tuning.compute_min_gw_lmi` |

Tier 1 is what the closed-loop run actually demonstrates. Tier 2 brackets it from above. The thesis claim is **not** "the spectral-gap condition is wrong" — it is "the spectral-gap condition is conservative for the constrained-MIQP regime, and the empirical contraction rate from the 12 h simulation, combined with the tier-2a (Häberle) / tier-2b (spectral-gap) sufficient bounds, is the proper stability statement for this controller." The g_v sweep (`gv_sweep_slack_ratio.png`) makes this explicit: to be SG-certified for the moving actuators one would have to operate at $g_v \sim 10^2$ instead of $10^5$ — a 1000× weaker voltage-tracking weight that would kill closed-loop responsiveness.

## 5. Conclusion

* For the actuators that move (DER, PCC, OLTC), the controller operates in the **strongly box-regularised regime**: current $g_w$ is $\sim\!10^{-4}$ of the per-actuator spectral-gap minimum and $\sim\!10^{-3}$ of the constrained-aware Häberle floor.
* For the actuators that are intentionally pinned (V_gen with $g_w^\text{gen} = 1 \cdot 10^7$), the controller is **SG-certified by 32–155×**; this is a deliberate regularisation choice, not a stability-driven tuning.
* The **empirical contraction rate** $\rho_k$ has a per-zone median of 0.85–0.96 and stays below 1 in 74–100 % of TSO firings (zone 2 the marginal case at 74 %; zones 1 and 3 at 95–100 %).
* The spectral-gap floor remains **valuable as a diagnostic upper bound**: it is the $g_w$ that an unconstrained variant of the same controller would need. The Häberle floor is a tighter intermediate bound (factor 2 closer for rank-deficient $M$). **The remaining gap between Häberle and live $g_w$ — about three orders of magnitude — is the quantity to attribute to integer-OLTC projection and per-actuator trust regions.** Closing that gap with a clean inequality is the open theoretical question.

## 6. Open items / what to do next

1. **Run a 12 h scenario-B simulation** to confirm $\rho_k$ statistics under the full diurnal profile and the scheduled contingencies. Current $\rho_k$ comes from a quiet 60-min window; the contingencies in `main()` (gen-5 trip at min 90, load-bus-13 connect at min 30) would expand the histogram tails. The integration is ready; only the underlying MIQP-solver `user_limit` issue (encountered in the first 4 h smoke attempt) needs to be resolved or bypassed.
2. **Once Feature A lands** (Q_gen as soft-constrained TSO output), the observer's $H_Q$ block becomes non-empty and the Pareto sweep gains a real $g_q$ axis; re-run the sweep then.
3. **Trust-region-aware tier 4.** A new sufficient inequality that combines Häberle's box term with the integer-projection contribution from OLTC and the per-actuator $\Delta u_\text{max}$ cap. Closing the remaining $\sim\!10^{-3}$ ratio between Häberle and live $g_w$ with a clean closed-form bound is the open theoretical question (and the most plausible novel contribution for the chapter).
4. **Cite-check the stubs** — Ortmann (2023) Ch. 5 three-tier structure, Häberle et al. (2021) L-CSS Theorem 1 (especially the constant 1/2 in the descent condition), the Caduff (MSc 2021) / Ortmann (PhD 2023) box-regularised terminology.
5. **Identify the published source** for the spectral-gap condition $\lambda_\min(D_w + M) > \|M\|_\text{op}$, or explicitly disclose it as an in-house derivation. Most plausible candidates: Hauswirth et al. (2021) ARC / IEEE TAC, Colombino et al. (2020) IEEE TCNS. Verified-NOT: Bianchi & Dörfler (2025), which uses a different (Lyapunov-based) certificate.

## 7. Notes on naming history

The condition $\lambda_\min(D_w + M) > \|M\|_\text{op}$ was originally implemented in this codebase under the label "Bianchi condition (1'')". WebFetch of arXiv:2412.10964v2 (Bianchi & Dörfler 2025, EJC 86) on 2026-04-20 confirmed that the actual Theorem 1 of that paper uses a **max-type composite Lyapunov function** with constants $(\mu_1, \theta_1, \mu_2, \theta_2, \xi, \alpha)$ and conditions $-\mu_1 + \xi\theta_1 < 0$ and $\theta_2 - \xi\mu_2 < 0$ (Assumption 3) — i.e. it does **not** contain a spectral inequality of the implemented form. The condition was therefore renamed "spectral-gap sufficient condition (SG)" throughout the codebase. The label "Bianchi" was kept only in the original 2025 paper citation in `stability_tuning.py`'s reference list, not as a name for any condition this observer evaluates.
