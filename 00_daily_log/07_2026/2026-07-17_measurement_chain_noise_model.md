# Measurement-chain noise model revision

**Timestamp:** 2026-07-17 15:46:22 +02:00  
**Reason:** Replace the first independent bounded-uniform V/I/P/Q model with a metrologically defensible component-wise model after rechecking IEC accuracy-class interpretation and 380-kV RMS voltage measurement.

## Short answer

Controller-facing analogue measurements now contain persistent, channel-specific VT/CT/PMD error plus a small reproducible per-sample contribution. Voltage magnitude is level-specific. P and Q are no longer perturbed independently; they share gain and phase-displacement errors through complex power. Plant result tables remain exact and are never exposed to a controller.

## Assumptions and profile components

The class number is a permissible component-error half-width, not a Gaussian standard deviation. A rectangular Type-B component with half-width a has standard uncertainty a/sqrt(3).

| Component | minimum | conservative |
|---|---:|---:|
| EHV VT/CVT, nominal voltage >= 220 kV | +/-0.1% | +/-0.1% |
| lower-voltage VT | +/-0.2% | +/-0.2% |
| voltage PMD | +/-0.1% | +/-0.2% |
| CT | +/-0.2% | +/-0.5% |
| current PMD | +/-0.1% | +/-0.2% |
| power-meter gain | +/-0.2% | +/-0.5% |
| power phase displacement | +/-0.1 deg | +/-atan(0.02) = 1.14576 deg |

The 220-kV threshold maps the IEEE-39 345-kV grid to the intended 380-kV EHV class. Lower-voltage buses, including the 110-kV DSO systems, use the class-0.2 VT assumption.

Variance-equivalent aggregate rectangular half-widths are reported for information:

- minimum: V_EHV 0.141%, V_HV 0.224%, I 0.224%, P_EHV 0.300%;
- conservative: V_EHV 0.224%, V_HV 0.283%, I 0.539%, P_EHV 0.714%.

The runtime draws components separately; these aggregate values do not replace the convolution of component distributions.

## Model structure

For every bounded component, the default split is 90% persistent calibration bias and 10% per-sample variation. The absolute sub-bounds sum to the original accuracy-class bound:

    e_c(k) = (1-f) a_c b_c + f a_c w_c(k),  f = 0.10,

where b_c and w_c are reproducible U(-1,1) draws. Bias is stable for a physical channel throughout the run; the sample part is shared by all packets reading that channel at the same controller instant.

RMS voltage magnitude is multiplicative:

    V_m(k) = V(k) (1 + e_VT(k)) (1 + e_PMD,V(k)).

Current uses the explicit CT primary rating when a line column named ct_primary_i_ka, measurement_i_nom_ka, or rated_i_ka exists. Below 20% of CT rating, the error is referenced to 0.2 I_n. Existing networks lacking CT metadata can explicitly fall back to line.max_i_ka.

Power uses one coupled channel:

    S_m = (1 + e_VT)(1 + e_CT)(1 + e_PMD,S) exp(j delta) S.

Consequently,

    P_m approx P(1+g) - Q delta,
    Q_m approx Q(1+g) + P delta.

This retains non-zero Q error near Q=0 when active power flows, instead of incorrectly making Q noise vanish.

## Files changed

- configs/config.py
  - replaced four aggregate profile bounds with measurement-chain component profiles;
  - added level threshold, persistent/sample split, CT-rating metadata, overrides, validation, and equivalent-bound reporting.
- core/measurement_noise.py
  - introduced deterministic physical-channel draws;
  - introduced persistent bias and per-sample caches;
  - implemented multiplicative RMS voltage and CT/PMD current measurement;
  - implemented coupled complex-power error for interface, DER, generator, and tie-line power.
- experiments/runners/multi_tso_dso.py
  - changed startup reporting to show equivalent EHV/HV bounds, phase bound, and persistent share.
- tests/test_measurement_noise.py
  - replaced independent-channel tests with component-bound, level-class, CT-rating, temporal-persistence, and complex-power invariants.

## Controller assumptions, constraints, actuators, and controlled outputs

- Controllers still know the plant only through cached sensitivities/models and received packets.
- No actuator model changed: AVR setpoints, DER Q, OLTC positions, and shunt states are untouched by measurement noise.
- Digital actuator states and commanded references remain exact.
- Controlled voltage magnitudes, line currents, PCC/interface Q, DER/generator P/Q, and tie-line P/Q receive the revised metering errors.
- Voltage-angle packet fields remain exact because the requested voltage measurement is RMS magnitude; phase error is used internally only to form measured P/Q.
- Plant truth remains available only to post-control system recording and the tracking truth overlay.

## Verification

- tests/test_measurement_noise.py: 8 passed.
- Staged config, core model, runner, and tests compiled under the qOFO_clean Python 3.12 environment.

## Sources and rationale

- NIST Type-B rectangular uncertainty: https://www.itl.nist.gov/div898/handbook/mpc/section5/mpc541.htm
- IEC 61869-5 CVT scope for Um >= 72.5 kV: https://webstore.iec.ch/en/publication/6054
- Siemens SICAM P850/P855 measurement classes and RMS aggregation: https://support.industry.siemens.com/cs/attachments/109752151/SICAM_P85x_7KG85_MAN_US_3.pdf
- Schneider ION9000 class-0.1 V/I/P: https://eshop.se.com/za/powerlogictm-ion9000-meter-din-mount-no-display-hw-kit-metseion92030.html
- GE iSTAT I500 class-0.2 P/Q and 0.1-degree angle: https://www.gevernova.com/grid-solutions/sites/default/files/resources/products/manuals/i500sp-en-m-d.pdf

## Unresolved points

- Add real ct_primary_i_ka values to network data when known; max_i_ka remains a compatibility fallback, not a claim about installed CT ratios.
- Replace generic component classes with substation-specific VT/CT/PMD nameplate data if these become available.
- The 90/10 temporal split is an explicit modelling assumption because accuracy-class standards do not specify repeatability versus persistent bias.
