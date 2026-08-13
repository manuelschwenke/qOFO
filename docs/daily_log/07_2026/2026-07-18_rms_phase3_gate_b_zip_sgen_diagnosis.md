# 2026-07-18 — RMS Phase 3: Gate-B root cause isolated

- **Timestamp:** 2026-07-18 23:07 CEST
- **Reason:** continue the RMS/PowerFactory build after the Phase-3 parity
  investigation stopped at the per-branch-loss comparison.
- **Code added:** `pf/gate_b_diagnostics.py`, a read-only diagnostic for
  system active-power balance and per-branch P/Q-loss comparison.

## Established result

The clean `wind_replace_t0_20160105-0800` state remains reproducible:

- max bus-voltage magnitude deviation: `1.038e-3 pu`;
- max aligned angle deviation: `0.3709 deg`;
- PowerFactory slack-P deviation: `12.670 MW`.

The passive PowerFactory model is not the source of the difference.
PowerFactory satisfies its active-power balance exactly:

| quantity | pandapower snapshot | PowerFactory |
|---|---:|---:|
| generation minus served load | 21.162785 MW | 36.225529 MW |
| summed line + transformer loss | 35.676677 MW | 36.225529 MW |

The snapshot therefore has a `14.513892 MW` accounting/KCL gap.  It is
concentrated exactly at the two buses that contain both an anchored-ZIP load
and a constant-PQ wind `sgen`:

| bus index | wind P | vm | missing P = P_wind (vm - 1) |
|---:|---:|---:|---:|
| 18 | 230.382572 MW | 1.03050243 | 7.027228 MW |
| 24 | 244.894860 MW | 1.03057093 | 7.486663 MW |

Total: `14.513891 MW`, equal to the system gap to numerical precision.

## Cause in pandapower 3.4.0 (also reproduced with isolated 3.5.4)

During pp-to-ppc conversion, pandapower first assigns the load's bus-level
current/impedance fractions and then aggregates *all* PQ elements, including
static generators, into the same `PD/QD`.  Consequently, at a bus with
constant-current P load and constant-P wind generation, the solved equation
is

\[
P_{\mathrm{pp}}(V)=(P_L^0-P_G)V,
\]

whereas the element semantics and exported result tables state

\[
P_{\mathrm{intended}}(V)=P_L^0V-P_G.
\]

For Q, the anchored load is constant impedance, so the analogous erroneous
term is `Q_G (V² - 1)`.  Thus `res_sgen`, `res_load`, and `res_bus` do not
represent the injections that produced the stored branch flows at the two
co-located buses.  The JSON round-trip test reproduces the same internal
equations and therefore cannot detect this cross-table KCL defect.

A reversible PowerFactory test replaced the two parks' setpoints only for one
load-flow run by the effective pandapower injections
`P_eff=P_sgen*V` and `Q_eff=Q_sgen*V²`.  It gave:

- max `|d vm| = 2.884e-8 pu`;
- max `|d va| = 2.542e-6 deg`;
- max line-flow deviation `< 6.2e-5 MW`;
- max transformer-flow deviation `< 1.5e-4 MW`.

The original constant-PQ setpoints were restored in a `finally` block; a
subsequent `pf_sync --dry-run` reported zero changes.

## Assumptions, constraints, actuators, controlled outputs

- **Assumption:** wind parks are constant-PQ injections during the parity
  LDF, as specified by Phase 3 and by `ElmGenstat.av_mode='constq'`.
- **Constraint:** Gate B requires `<1e-4 pu` voltage and `<0.01 deg` angle
  parity without changing the physical meaning of wind P/Q setpoints.
- **Actuators:** four TSO wind parks (`ElmGenstat`; later Q setpoints through
  their station-controller write handles).  No controller action is present
  in the parity run.
- **Controlled outputs checked:** all TN bus voltage magnitudes/angles and
  line/transformer flows; per-park P/Q are parity observables.

## Architectural decision required before a model fix

No model-equation workaround was committed.  The scientifically defensible
choices are:

1. use the build plan's constant-PQ load model for Gates A–C and retain ZIP
   as a labelled Phase-5 stress variant;
2. introduce a tested project-local correction/fork of pandapower's
   bus-level ZIP aggregation so ZIP loads and constant-PQ injections remain
   separate;
3. introduce auxiliary electrical buses/near-zero branches to separate
   co-located load and generation (not recommended: artificial topology and
   numerical-conditioning risk).

Scaling the PowerFactory wind setpoints to the erroneous effective
pandapower injections is rejected as a permanent fix: it makes bus/flow
parity green but violates the specified constant-PQ wind-park powers and
fails per-park P/Q parity.

