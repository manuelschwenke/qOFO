# 2026-08-21 ? DSO-4 continuous-loop damping correction and corrected B1/B2 reruns

## Purpose

Diagnose the DSO-4 non-settling cases in Chapter 9 Experiment B, distinguish
continuous-control oscillation from OLTC activity and network structure, apply a
narrow correction, and rerun both Chapter 9 experiments:

- B1: isolated subordinate-system measurement of (N_\mathrm{inner});
- B2: closed-loop sweep of (T_\mathrm{TS}).

The simulations were run from a clean archive of commit
`5d8e0f31c2aa5b4256782f6709ce4516142a2bdf`, with the exact runner and
selected-design helper copied into each result folder. This avoided the
unrelated live config refactor in the main worktree.

## Established diagnosis

The problematic DSO-4 mode was not an OLTC limit cycle.

- Every problematic B1 case recorded zero DSO tap moves.
- In the instrumented period-2 case, `tap_pos` was identically zero.
- The interface-Q tracking error alternated sign while its magnitude slowly
  increased. A scalar AR(1) fit gave (-0.99974) over the complete trace and
  approximately (-1.0057) over its late portion.
- Stage 0 designed the pooled DSO DER weight as
  (g_{w,\mathrm{DSO-DER}}=549.8818). DSO 4 alone required a geometric mean
  of (610.9483) and a maximum column requirement of (828.3682).
- Scaling the DSO-4 Stage-0 eigenvalue by the pooled/area weight ratio gives the
  approximate effective value
  (\lambda_\mathrm{DSO4}\simeq1.99649), hence the predicted discrete pole
  (1-\lambda\simeq-0.99649). This matches the measured period-2 mode.

DSO 4's long, weak network is the structural reason its cached sensitivity
columns are larger and more heterogeneous: its modeled circuit length is
585.6 km and its measured internal voltage spread is 0.147 p.u. (DSO 2:
336 km and 0.117 p.u.). The proximate control error, however, was the pooled
movement weight being too small for DSO 4. The taps did not oscillate.

## Correction

The Chapter-9 selected-design helper now rebuilds and verifies archived
candidate `fe010aa3ead1` first, then applies an explicitly labelled
post-selection safety correction:

1. reduce the paired DSO-2/DSO-4 voltage-relief factor from 20 to 10;
2. preserve (g_v/g_{w,\mathrm{OLTC}}) by re-deriving both OLTC weights;
3. set only DSO 4's continuous-DER class weight to
   (828.3682036634343), the maximum of its ten Stage-0 column requirements.

The global DSO DER weight remains (549.881810527467) for DSO 1?3. The
correction is implemented and asserted in
`experiments/ch_9_parameter_selection/_ch9_selected_design.py`; it is not
misrepresented as part of the archived campaign optimum.

### Diagnostic screens

At relief factor 10 and DSO-4 weight 610.9483 (area geometric mean), the
30-case DSO-4 screen still had 1/30 censored, median 11, p95 38.65 and maximum
46. Thus the geometric mean was not robust enough.

At relief factor 10 and DSO-4 weight 828.3682 (area maximum), the same 30-case
screen had 0/30 censored, median 8.5, p95 16 and maximum 17 under the stricter
0.5 Mvar flatness criterion. It also had zero tap moves and zero voltage
violations.

## Corrected B1 ? isolated (N_\mathrm{inner})

Artifact:
`docs/data/2026-08-21_ch9_b1_b2_corrected/B1_ninner_10mvar/20260821-180548/`

Protocol:

- ten admissible operating windows; the two structural VDE dead-zone windows
  were excluded because they have no reactive capability band to traverse;
- four DSOs and three interfaces per DSO;
- balanced half-fraction direction design: 120 cases, preserving every
  window/DSO/interface cell;
- maximum applied setpoint step 10 Mvar;
- (T_\mathrm{STS}=20\) s;
- (N_\mathrm{inner}) is the first iteration after which
  (|q_\mathrm{PCC}(k)-q_\mathrm{PCC}(k-1)|\le1\) Mvar remains true;
- frozen supervisory parent, so the subordinate response is isolated.

Result:

| Metric | Corrected B1 |
|---|---:|
| cases | 120 |
| censored | 0 |
| median (N_\mathrm{inner}) | 2 |
| p90 | 6.2 |
| p95 | 10 |
| maximum | 18 |
| (N_\mathrm{inner}\le9) | 113/120 (94.2%) |
| capability-limited flags | 0 |
| tap moves | 0 |
| voltage violations | 0 |
| maximum command-delivery error | 0 Mvar |
| steps capped at 10 Mvar | 113/120 |

Per-area p90 values were DSO 1: 6.0, DSO 2: 11.1, DSO 3: 4.0 and DSO 4: 3.0.
Thus (N_\mathrm{inner}=9) is supported as a pooled approximately-90th-percentile
engineering choice (and covers 94.2% empirically), but it is not a worst-case
guarantee. A strict pooled p95 choice would be 10.

The 10 Mvar cap is an empirical supervisory envelope, not a formal controller
bound: in the archived (T_\mathrm{TS}=180) s closed-loop run over these ten
windows, 3597/3600 command increments (99.9167%) were at most 10 Mvar; only
three exceeded 10 Mvar and two exceeded 20 Mvar.

## Corrected B2 ? (T_\mathrm{TS}) sweep

Artifact:
`docs/data/2026-08-21_ch9_b1_b2_corrected/B2_ts_period_sweep/20260821-183927/`

Protocol: 12 operating windows, (T_\mathrm{STS}=20) s fixed,
(T_\mathrm{TS}\in\{60,120,180,240,300\}) s, 60/60 runs completed.

| (T_\mathrm{TS}) [s] | configured ratio | (ho) median | (ho) p95 | censored | tap moves | voltage violations |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 3 | 0.4407 | 2.5140 | 0.1574 | 0 | 6 |
| 120 | 6 | 0.3095 | 2.7782 | 0.1552 | 1 | 0 |
| 180 | 9 | 0.2572 | 3.5164 | 0.1597 | 1 | 0 |
| 240 | 12 | 0.2221 | 2.9677 | 0.1621 | 1 | 0 |
| 300 | 15 | 0.1909 | 2.5977 | 0.1659 | 0 | 0 |

Relative to the superseded factor-20/global-weight sweep, the corrected
controller reduced the median tracking ratio at every period and reduced
censoring from about 22?23% to about 15.5?16.6%.

The six voltage flags are three interface copies of two time intervals in
`d_ramp_down_spring` at (T_\mathrm{TS}=60) s. The maximum recorded voltage
was 1.100187 p.u.; maximum voltage slack was (8.6\times10^{-5}) p.u. All
periods of 120 s or longer had zero voltage violations. Across B2 there were
three tap moves in 30,240 interface intervals.

The median (ho) decreases monotonically with (T_\mathrm{TS}). This is
expected because (ho) measures how well the subordinate layer executes the
received correction, not whether that correction is stale when it arrives.
Therefore B2 is useful tracking/timescale evidence but does not, by itself,
establish an optimum at 180 s. A selection claim would require the supervisory
objective or an explicit staleness metric. B2's `n_k` must not replace B1's
flatness-based (N_\mathrm{inner}).

## Code change

Changed:

- `experiments/ch_9_parameter_selection/_ch9_selected_design.py`
- `configs/paramsets/tuned.json`

Method:

- retain the archived-candidate reconstruction and equality guard;
- derive and assert the DSO-4 correction from the archived Stage-0 per-area
  payload;
- clear/re-derive the paired voltage-relief maps compatibly with both the
  historical and current config implementations;
- record the correction in run provenance;
- carry the same DSO-4-only `dso_der` override in the canonical tuned
  parameter set. Its loader test produced `dso_der=828.3682` and the paired
  factor-10 `dso_oltc=1500`.

Reason: eliminate the DSO-4 period-2 continuous-loop instability without
changing DSO 1?3 or the OLTC loop-gain ratio.

## Risks and unresolved points

- The correction is a post-selection safety overlay, not a full Stage-1
  re-optimisation. B1 and B2 validate the Chapter-9 timescale experiments, but
  a future controller-wide tuning campaign should include the per-area DSO-4
  weight as a first-class parameter.
- The six (T_\mathrm{TS}=60) s voltage flags are numerically small but are
  reported rather than rounded to zero.
- B2 does not contain the supervisory staleness objective needed to select
  (T_\mathrm{TS}=180) s uniquely.

