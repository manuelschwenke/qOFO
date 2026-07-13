# 2026-07-06 — BME method section written into thesis Chapter 7

**Where:** thesis repository (`latex_diss_ms`), not this code repository. Logged here
because the section is generated from this repo's BME design/implementation record.

## What was changed

`Chapters/Chapter07.tex` (Multi-TSO Multi-DSO OFO Control):

1. **New section** `\section{Boundary Marginal Exchange: Cooperative Coordination on a
   Common Objective}` (`ch:architectures:multitso:bme`), appended after the tie-line
   coordinator section, with subsections:
   - `:bme:boundary` — boundary registry, separator assumption (asserted, not assumed),
     stacked coordinates x_b = [v_b | θ_b] ∈ R^{2|B|} incl. the PMU observability
     assumption (D7 REVISED rationale: loss objectives are angle-coupled).
   - `:bme:objective` — Φ = Σ Φ_i (w_loss·P_loss,i + Σ φ_band), ownership partition
     (D1: interior by owner, ties 50/50, band per bus owner), partition invariant,
     TS-only scope (Q7), objective replacement (g_v tracking off, g_q_tie off — Q1/Q3),
     C¹ hinge caveat, Phulpin exchange-rate distinction.
   - `:bme:marginals` — μ_j = dΦ_j/dx_b (eq. with port-frozen internal-response chain),
     local computability, sparsity, price interpretation, bus/delay d, per-sender
     low-pass β_μ (D3), self-marginal undelayed/unfiltered.
   - `:bme:gradient` — Convention-A price term ∇f^bme = w_Φ(g_own|_{x_b} + H_{b,i}ᵀ Σ μ)
     hooked into eq:ofo_miqp; exactness identity (distributed = centralised gradient at
     d=0, β=1) + single-area identity as correctness anchors; no-double-counting remark
     (Convention A vs B); H_{b,i} concession + access-restriction + online-estimation
     realism story; w_Φ as a units choice; integer externality pricing vs
     relax–round–resolve.
   - `:bme:hygiene` — switch notices (H_{b,i}·Δu_int, horizontal generalisation of the
     vertical MSC/MSR feedforward), round-robin commit slotting (D5), ε-acceptance vs
     frozen-integer QP (D6) + switching ledger as the finite-switching premise data.
   - `:bme:protocol` — the step-by-step per-step sequence (measure → receive/filter →
     price/publish → assemble → solve MIQP → gate integers → apply/notify → settle),
     cold-start and hold-last-filtered drop policies.
   - `:bme:properties` — information-accounting table, vref as quadratic-boundary
     special case (γ = scalar normalised private counterpart of μ), decomposition
     positioning (goal coordination / feedback equilibrium seeking, single solve per
     step, no ADMM inner iterations), symmetrisation statement (scoped to synchronous
     ideal + fixed hinge active set; empirical characterisation deferred), fairness
     (normalised overcost; redistribution is a property of Φ, not of decentralisation),
     closing `definition_own` box.
2. **Intro hooks (marked `% DRAFT`):** spectrum paragraph "two points" → "three points"
   + BME sentence; new paragraph "The cooperative coordination concept."; Organisation
   paragraph extended with the new section.

`FrontBackmatter/Glossary.tex`: added `\newabbreviation{ADMM}` (BME entry already
existed).

## Method / structure of the change

Content transcribed from `docs/BME_SPEC.md` §3 and `docs/BME_STATUS.md` (Phases 0–6),
translated to thesis conventions: sentence-per-line, `\VEC`/`\MAT`/`\Transp`/
`\ReelleZahlen`, `\gls{}`, chapter-7 label scheme, hooks into the existing per-step
MIQP `eq:ofo_miqp` (Ch. 4) and tie-coordinator equations (`eq:multitso:gamma`).
Notation choices to avoid chapter-internal collisions: β_μ (filter; β is the
subsidiarity weight in eq:multitso:update), ε_sw/c_sw (ε is the tie-coordinator cap),
subscript "int" for the integer block (i is the zone index; ch.4 uses subscript i).
Numerical results were deliberately kept OUT of the methods chapter (forward
references to ch:case3 + `% TODO[refs]` placeholders for phulpin.2009 and the
TSO-scale ADMM reference, which are not yet in References.bib).

## Verification

One `lualatex -draftmode -halt-on-error` pass on `ClassicThesis.tex`: exit 0, no
errors; only first-pass undefined-reference warnings for the new labels and 7
pre-existing undefined citations in other chapters (pages 58–111).

## Reason

Manuel's request 2026-07-06: "write down how the BME method works, step by step …
in chapter 7". Chapter 7 previously ended at the tie-line coordinator; the BME
successor (spec Phases 0–6 implemented, MC campaign §6f running) had no thesis text.

## Addendum (same day) — protocol step 1 corrected after Manuel's review

Manuel flagged "refresh the local sensitivities at the measured operating point"
in the per-step protocol: his standing premise (and the repo default,
`refresh_shared_jac_on_tso=False`) is a one-time cached Jacobian. The clause had
elevated the v1 implementation choice (bme rung forces per-tick re-linearisation,
fail-fast otherwise; BME_STATUS.md Phase 4: "measurement-evaluated gradients on a
frozen model are noted future work") into the method definition.

**Fix:** refresh clause removed from step 1 (Measure = measurements only); new
`\paragraph{Model maintenance.}` added after the protocol enumerate: measured
VALUES (loss-gradient loadings, hinge active set, boundary state) are per-step
feedback; the OPERATORS (J_int,i, H_{b,i}) are model objects whose refresh
cadence is a policy choice consistent with the cached-sensitivity premise;
caching ⇒ exact values through approximate directions = the σ_H error class;
reference implementation = re-linearised at coordination cadence (exact-model
configuration, enables the plant-side identity check); frozen-model variant
recorded as open. Field reading: J_int local (re-linearisable from own state
estimation at will), H_{b,i} an online estimate that necessarily lags (closer to
cached than fresh). Recompiled: 0 errors.

**Open (code side, for later):** (i) mode="bme" currently fail-fasts on
`refresh_shared_jac_on_tso=False` — needs relaxing if a frozen-model BME rung is
ever run; (ii) ladder asymmetry worth an ablation: bme runs per-tick refreshed
full-net sensitivities while none/vref run frozen (Ward) models — a
`none + refresh` control (or frozen-model bme) would separate coordination gain
from model-freshness gain.

## Addendum 2 (same day) — measurement requirements clarified after Manuel's question

Question: "do I really only need the complex voltages at the tie lines to minimise
losses? for the other buses voltage magnitude measurements are sufficient? or
current measurements?"

Answer written into a new `\paragraph{Measurement requirements.}` in
`:bme:boundary` (after the stacked-coordinates paragraph):

- Interior: magnitudes suffice for the band term only; the LOSS term additionally
  needs branch loadings — magnitude snapshots are blind to losses. Two sufficient
  standard routes: (a) branch-current magnitudes with ∂P/∂|I| = 2R|I| chained
  through the model's I-rows (= the existing form-B loss-objective mechanism);
  (b) the zone's state estimate (interior angles reconstructed from conventional
  P/Q/V telemetry, loss depends only on frame-independent angle differences).
  Interior PMUs required by neither route.
- Boundary: PMUs buy the CONSISTENCY OF THE EXCHANGE, not loss evaluability —
  (i) the price arithmetic H_{b,i}ᵀ Σ μ_j combines quantities produced by
  different operators, and angle coordinates are only comparable in one
  synchronised frame (each zone's SE has its own arbitrary angle reference);
  (ii) notice correction + online H_{b,i} identification act on measured boundary
  angle trajectories. μ_i itself needs no measured boundary angle (internal SE +
  port-frozen model).
- `% TODO[refs]`: cross-reference the current-based loss objective once it gets a
  section in ch. 4 (009 work not yet written up in the thesis).

Recompiled: 0 errors.
