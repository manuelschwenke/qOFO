# 2026-07-15 - Offline IQC/LMI stability certificate

Timestamp: 2026-07-15 13:56 CEST

Reason: Evaluate the Lessard-Recht-Packard IQC/LMI framework for stability and parameter tuning of the cached-model multi-TSO/multi-DSO OFO hierarchy, without adding a separate step-size parameter.

## Changes

Added tuning/stability_certificate:

- iqc.py: sector-IQC rate LMI for projected gradient descent and a multiplicative cached-gradient/model-error IQC, ||u-w|| <= delta ||w||.
- linear.py: exact frozen linear-map Lyapunov LMI, firm-nonexpansive projection IQC, and real-Schur diagnostic for non-neutral modes.
- hierarchy.py: automatic extraction from experiments.run_multi_system_ofo:make_config using the existing one-step hook and controller-cached sensitivities.
- report.py, CLI, data models, and focused tests under tests/tuning/stability_certificate.

No online controller, update rule, experiment configuration, or actuator dispatch logic was changed.

## Convention and method

The implementation fixes alpha = 1; step amplitude remains entirely in positive diagonal G_w. For active local curvature bounds 0 < m <= L, it solves the nominal 2x2 Lessard sector LMI. For a uniform multiplier s on the current continuous G_w, stability requires s > L/2, with ideal quadratic scale s_opt = (m+L)/2. This is a whole-controller scale, not a per-actuator-class optimum.

The nonsymmetric coupled map is checked separately as A = I-M, optionally followed by a fixed convex metric projection. Integer variables are frozen in all continuous LMIs.

## Automatic result for run_multi_system_ofo.py

Current values include g_w_der = 50, g_w_pcc = 200, g_w_gen = 5e9, g_w_dso_der = 1000, and g_w_dso_oltc = 150.

- Full 22-state frozen projected hierarchy: NOT CERTIFIED. Eight numerical/null-curvature directions remain; spectral radius is approximately 1 and spectral norm is 1.0004005.
- Symmetry defect: 0.250418, so a common-potential gradient interpretation is not justified.
- Conditional 14-dimensional non-neutral frozen linear subspace: CERTIFIED, rho = 0.9999763 and condition(P) = 1.135. This assumes locally inactive constraints and is not a full projected-map certificate.
- TSO local active-mode rates are approximately 0.9999 to 0.999999. Uniform ideal scales are 0.0059, 0.0340, and 0.0217. These are not direct recommendations because they also scale the deliberately huge generator weight.
- Four instantiated DSO controllers were extracted. Active-mode rates are 0.870 to 0.883; uniform ideal scales are 0.455 to 0.472 of current G_w, with idealized rates 0.725 to 0.744. These are plausible BO priors.
- At hypothetical relative cached-gradient errors delta = 0.01, 0.05, and 0.10, all DSO active-mode rates remain below 0.896. TSO zones 2 and 3 are not certified at delta >= 0.05.
- Lessard IQCs are NOT APPLICABLE to MIQP integer decisions. The separate project C3 model reports rho(Gamma) = 0.79072 < 1, but its seven assembled discrete variables have a caveat: configured TSO shunts use the hysteretic integrator, so C3 does not certify that logic.

## Validation

- Final focused pytest result: 10 passed (two expected CVXPY inaccurate-solution warnings); covers nominal/robust IQCs, unstable-step rejection, exact linear/projection LMIs, configuration loading, and neutral-mode handling.
- The automatic CLI completed successfully twice and wrote final Markdown/JSON reports under results/stability_certificate.
- Returned SDP matrices are post-validated by eigenvalues. CVXPY emitted OPTIMAL_INACCURATE warnings for some SCS solves; unvalidated matrices are not accepted.

## Interpretation / next work

Use this method now as an offline diagnostic and source of BO priors, especially for continuous DSO weights. Do not yet use it as a hard full-hierarchy BO feasibility constraint. A future BO extension should vary actuator-class G_w, rebuild cached curvature per candidate and operating-point/topology snapshot, and retain only candidates with a feasible full projected-map certificate. Integer switching and hysteretic shunts require a hybrid/dwell-time or finite-state argument rather than this continuous IQC.

## 2026-07-15 14:28 CEST - Cached-curvature LMI BO continuation

Reason: Implement per-candidate cached-curvature reconstruction and LMI evaluation while preserving the deliberately slow generator/AVR weight.

### Changes

- Added tuning/stability_certificate/snapshot.py. One short plant run captures H blocks, objective diagonals, actual coordinate-level G_w vectors, actuator-class labels, and DSO models. A versioned configuration-keyed local cache avoids repeating the plant extraction.
- Added tuning/stability_certificate/candidate.py. Each candidate rescales only its actuator classes, rebuilds C1/C2/C3, and reruns local fixed-rate sector LMIs plus the coupled non-neutral Lyapunov LMI.
- Added tuning/stability_certificate/bo.py. Optuna searches g_w_der, g_w_pcc, and g_w_dso_der in baseline-relative log ranges, with a small log-distance penalty from the empirical baseline.
- Corrected tuning/parameters.py: g_w_gen is no longer silently overwritten by FIXED_OVERRIDES. Because it is excluded from BO_DIMS, it now inherits the baseline exactly.
- Added regression tests for class scaling, changed-generator rejection, candidate LMI execution, the short BO loop, and baseline generator preservation.

Fixed throughout the study:

- alpha = 1.
- g_w_gen = 5e9.
- g_w_tso_oltc = 5000.
- g_w_tso_shunt = 12000.
- g_w_dso_oltc = 150.
- shunt_int_g_w = 150.
- Objective weights and the cached sensitivity snapshot.

### Automatic 16-trial result

Baseline objective: 4.415297. Best objective: 4.121396.

Best continuous weights:

- g_w_der: 50 -> 47.21048 (ratio 0.9442).
- g_w_pcc: 200 -> 78.53980 (ratio 0.3927).
- g_w_dso_der: 1000 -> 814.24624 (ratio 0.8142).
- g_w_gen remained exactly 5e9 in all trials.

The coupled non-neutral rate improved from 0.9999763 to 0.9999492. DSO active rates improved from 0.870-0.883 to 0.840-0.857. All nominal best-candidate screening LMIs passed. Eight coupled neutral modes remain, so the full projected state is still not certified.

The final high-accuracy certificate confirms the conditional non-neutral rate. Under hypothetical cached-gradient errors, all DSO rates remain below 0.872 at delta = 0.10. TSO zones remain poorly conditioned and do not all retain robust certificates at delta >= 0.05.

### MIQP interpretation

The exact modeled discrete condition remains rho(Gamma) = 0.7907196 < 1 for baseline and best because discrete weights were fixed. This is a positive C3 certificate for the discrete interconnection represented by Gamma. A failed row-sum or per-actuator sizing condition is only failure of a stronger sufficient shortcut and does not invalidate the exact spectral-radius result. The separately dispatched hysteretic shunt integrator remains outside this Gamma/MIQP certificate.

Final validation: 14 stability-certificate/BO tests and 14 tuning-parameter tests passed. Four warnings were limited to Optuna experimental sampler flags and CVXPY inaccurate-solution notices; accepted SDP matrices remain residual-checked.

### Limitations

This is a stability-only cached-model study. It does not establish time-domain tracking quality or justify applying the best weights directly. In particular, the stability best g_w_pcc = 78.54 exceeds the separate performance-BO cap of 30, which was introduced to prevent a gameable sluggish-PCC solution. The candidate must therefore be tested in the multi-scenario performance simulation before adoption.
