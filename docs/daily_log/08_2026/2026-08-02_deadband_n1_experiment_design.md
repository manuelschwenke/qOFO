# 2026-08-02 — Dead-band selection as a DETECTOR problem, under an N-1 outage

**Design intent (stated by MS, 2026-08-02).** The DER Q(V) layer is a
disturbance-rejection mechanism. *Profile-induced* voltage shifts should remain
**inside** the dead band — they are the OFO's job. *Event-driven* excursions
should fall **outside** it and be compensated effectively. This holds equally
for TS- and DS-connected DER.

That makes δ a **detector threshold**, not a quantity with an optimum, and it
explains why the 1D study never found a reproducible argmin (δ\* CV 0.715 across
five windows): there is no minimum to find. The deliverable is an admissible
**interval**, bounded below by the profile-drift distribution and above by the
smallest credible event excursion.

## The quantity δ is actually compared against

The droop law is `|V − V_anchor| > δ`, and `qv_vref_anchor_pu` is re-anchored to
the measured voltage every time the OFO writes that park's setpoint
(`core/plant.py`). So δ does **not** discriminate voltage levels — it
discriminates **drift since that park's last dispatch**. Dispatch periods are
asymmetric: TSO 180 s, DSO 20 s.

## Measured inputs to the design (2026-08-02)

Anchor-referenced drift, undisturbed, at wide δ (droop silent, so open loop):

| | window | median | p90 | max |
|---|---|---|---|---|
| TS parks | 180 s | 0.00087 | 0.00197 | 0.00313 |
| DS parks | 20 s | 0.00051 | 0.00131 | **0.01106** |

N-1 excursion at the worst park (static scan):

| gen | P [MW] | TS max \|ΔV\| | DS max \|ΔV\| |
|---|---|---|---|---|
| 7 | 830 | 0.0104 | 0.0122 |
| 0 | 250 | 0.0539 | 0.0220 |
| 5 | 560 | 0.0618 | 0.0679 |
| 1 | 650 | 0.0717 | 0.0854 |
| 2 | 632 | 0.1025 | 0.0456 |

Impact tracks **lost AVR voltage support**, not MW: gen 7 is the largest unit
and the weakest disturbance. gen 9 (1000 MW) diverges and is excluded.

**The distributions overlap**: DS drift max 0.0111 exceeds the mildest event
(gen 7 at 0.0104). So no δ is simultaneously always-quiet and
always-responsive, and the design is a quantile trade-off. If the admissible set
comes out empty at a level, that is a structural statement about the
architecture, not a tuning failure.

A prediction of mine was **refuted** on the way: I argued δ_TS ≈ 9·δ_DS from the
period ratio. The measured ratio of medians is **1.71**, and on the tail the DS
parks drift *more* despite a 9× shorter window. Drift saturates rather than
accumulating with the dispatch period.

## Why RMS and not QSS

1. The pp-vs-RMS gap is itself a **dead-band-edge** phenomenon (2026-07-24: "a
   genuine solver-vs-solver" divergence; the droop is multi-valued at the edge
   and the two solvers settle in different basins). Selecting a dead band on
   QSS would measure the parameter with an instrument known to be unreliable at
   that parameter's critical point.
2. The design parameter is a **peak** deviation. QSS's first post-event sample
   is 20 s later — the electromechanical transient is over by then.
3. QSS redistributes the lost machine instantly through distributed slack; RMS
   does it through governor droop and AVR response.

QSS keeps one honest role: the undisturbed drift statistics, which are
quasi-steady.

## Experiment

`experiments/run_deadband_n1.ps1`, launched 2026-08-02 22:52.

- Window 2016-01-05 08:00, `rural_700`, DSO_3 ×2, physical VDE capability, rev-2
  sensitivities.
- Horizon **600 s**, trip at **t = 200 s** — 20 s *after* a TSO dispatch (TSO
  fires at 0/180/360/540), which maximises the droop-only exposure to 160 s.
  That is the inter-OFO-step stress the study is about. Two post-trip TSO
  dispatches (360, 540).
- δ ladder placed on the measured distributions, not round numbers:
  `0, 0.001` (droop-dominant) · `0.005, 0.01` (drift tail and the gen-7
  boundary, where detection is decided) · `0.025, 0.05, 0.075` (severe-event
  range) · `0.15` (droop disabled reference).
- δ_TS = δ_DS in this stage; both driven through
  `--tso-deadband`/`--dso-deadband` so stage 1 and a later 2D stage share one
  code path (the per-sgen map) and stay comparable.
- 8 δ × {twin, gen 7, gen 1} = **24 runs**, ~25 min each ≈ 10 h. Ordered twins →
  gen 7 → gen 1 so an interrupted night still yields a complete, usable dataset.

Twins are not optional: at gen 7 the event excursion is the same order as
ordinary drift, so the response must be referenced to the same-δ undisturbed run
or drift is counted as rejection.

## Readout (`analysis/deadband_n1.py`)

A δ is admissible for an event when all three hold, per level:

| criterion | metric |
|---|---|
| profile drift stays inside | `faopen_*` = fraction of inter-dispatch windows whose **open-loop** drift exceeds δ |
| event falls outside | `detected_*`, from the open-loop excursion (the droop-disabled cell) |
| event is compensated | `comp_eff_*` = 1 − peak(δ)/peak(δ_disabled) |

False activation is reported twice on purpose: `fa_*` from the cell's own twin
(closed loop — what actually happens at that setting, where a narrow δ
suppresses the drift being measured) and `faopen_*` from the widest-δ twin (open
loop — the detector view, independent of the setting being chosen). A design
rule is stated against the latter.

## Verification before launch

- δ = 0.15 reaches the RMS plant: `[qvpre] deadbands applied: [0.15] pu`.
- Trip at t = 200 s fires correctly against a **different** event window than
  the earlier probe (window 180 vs 60): `EvtOutage ... armed at RMS t=180.5s,
  i.e. within the interval reported for the nominal trip at t=200s`.
- PowerFactory genuinely tripped the machine: `('gen', 1) left service
  (m:P:bus1 unavailable)` — the only reliable detector, since `EvtOutage` does
  not update `outserv`.
- Severe outage (gen 1) with the droop disabled completes without divergence,
  ~25 min/run at 600 s.

Gate E FAILs for an N-1. That is expected physics, not a defect: Gate E
validates QSS/RMS equivalence and the two plants have genuinely different
dynamics through a topology change. The verdict is carried alongside the metrics
rather than used to reject runs.

## Reading the sweep log: "FAILED" is expected for every trip run

`experiments/run_comparison_rms_cosim_qss.py` ends with
`return 0 if gate_ok else 1`, so **exit = 1 means Gate E failed, not that the
run failed**. Gate E validates QSS/RMS equivalence, which a topology change
legitimately breaks, so all 16 trip runs report exit = 1 and the sweep's closing
summary will read "16 of 24 run(s) FAILED". Verified on run 9 (0293): the log
reaches the final summary, the DSO Q-tracking table and the result export, and
both `rms_records.pkl` (172 kB) and `csv/rms_der_raw.csv` (26 MB) are written.
The data is complete and admitted by the analysis.

A genuine failure looks different: no `rms_records.pkl`. That is the check to
apply, not the exit code.

Two log-reading traps met while diagnosing this:

* the per-run logs are **UTF-16** (PowerShell 5.1 `*>` redirection), so an
  ASCII `grep` for `Traceback` silently matches nothing on a log that plainly
  contains errors;
* Python warnings on stderr are wrapped by PowerShell as `NativeCommandError`
  and look like fatal errors in the log while being harmless (here: "sgen[7]
  ... cannot act as a Q actuator", a zero-Q-capability warning under the VDE
  diagram).

## Open-loop admission: a self-fulfilling test, corrected

The first version admitted a twin to the open-loop drift reference when its
`delta` exceeded **its own** observed drift maximum. That is self-fulfilling: a
droop which successfully suppresses drift below its threshold then looks
inactive. Measured 2026-08-03:

```
twin 0292 d=0.15    drift_max=0.01101     <- genuinely silent
twin 0291 d=0.075   drift_max=0.01101
twin 0290 d=0.05    drift_max=0.01101
twin 0289 d=0.025   drift_max=0.01097
twin 0288 d=0.01    drift_max=0.00375     <- SUPPRESSED, would have been admitted
twin 0287 d=0.005   drift_max=0.00365     <- SUPPRESSED, would have been admitted
```

Admission is now against the **widest** twin's maximum. This changed the pooled
DS drift maximum from an understated 0.0030 to the correct **0.0110** pu -- the
number the whole design rule turns on. Pooled reference: 4 twins,
TS n = 48, DS n = 3360.

## Contamination found and fixed

The gen-trip probe (run 0283) had **silently superseded** study run 0201 in the
δ = 0.01 cell of the 1D study: it is undisturbed in every key `ADMIT` checked
(no load step, profiled, rev 2, scaled DSO_3, diagonal dead band), so the curve
was briefly built from a run containing a generator outage. The duplicate-cell
warning caught it; the guard was missing.

Added `undisturbed_topology()` (contingency-free) and `n_total_s` to the 1D
`ADMIT`, wired into `deadband_selection`, `deadband_disturbance` and
`deadband_threshold` (collection *and* twin lookup). Verified: admitted counts
restored to 47 undisturbed / 41 stepped, duplicate warning gone.

## FINAL RESULTS — three windows (2026-08-04 11:53, 57 runs, 21 h 15 min)

**The single-window recommendation of δ = 0.005 pu is REFUTED by replication.**

Three operating windows, net infeed −117 / +409 / +1367 MW. 56 of 57 runs
usable (0316 lost its trace export).

### False activation — the binding evidence

| δ | TS: −117 / +409 / +1367 | DS: −117 / +409 / +1367 |
|---|---|---|
| 0.0025 | 0.450 / 0.167 / 0.083 | 0.060 / 0.030 / 0.046 |
| **0.005** | **0.133** / 0.000 / 0.000 | 0.009 / 0.006 / 0.014 |
| **0.01** | **0.000 / 0.000 / 0.000** | 0.000 / 0.001 / 0.005 |
| 0.025 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

δ = 0.005 was selected on the +409 window, where TS false activation is exactly
zero. **That zero is a property of the window, not of the system**: at the
import window the TS parks drift 2.4× further (max 0.00752 vs ≈0.0031) and
δ = 0.005 fires on 13.3 % of TS inter-dispatch windows.

- zero TS false activation at ALL windows → **δ ≥ 0.01**
- zero at BOTH levels → δ ≥ 0.0125
- cost of 0.005 → 0.01 ≈ 0.05 of compensation

**Recommendation: δ ≈ 0.01 pu.**

### Admissible intervals (all 12 combinations)

| window | gen 1 TS / DS | gen 7 TS / DS |
|---|---|---|
| −117 | [0.01, 0.05] / [0.01, 0.1] | [0.01, 0.1] / [0.01, 0.1] |
| +409 | [0.005, 0.075] / [0.025, 0.075] | [0.005, 0.15] / [0.025, 0.15] |
| +1367 | [0.005, 0.05] / [0.025, 0.05] | [0.005, 0.1] / [0.025, 0.1] |

The import window binds from below in every case.

### What reproduces

- **DS drift maximum**: 0.01101 / 0.00746 / 0.01107 pu
- **Compensation flat to δ ≈ 0.025** at every window, then decays
- **gen 7 severe everywhere**: no-droop peak 0.176 / 0.201 / 0.273 pu

### Unexplained: TS drift varies 2.4× across windows

Three mechanisms tested, **all refuted** — recorded so the next reader does not
repeat them:

| hypothesis | refutation |
|---|---|
| less DER capability at low infeed | capability *identical* (901.2 Mvar) at all three windows; P/S_n = 0.38–0.60, far above the 0.2 knee of the VDE diagram |
| Q saturating on the operating diagram | peak utilisation 12 %; zero samples above 95 % of capability |
| OLTC tap steps driving the drift | tap windows show 5–14× *lower* drift; only 1–3 tap events per 600 s run |

### Assets

`results/deadband_n1/deadband_n1_metrics.csv` (54 rows, per-window keyed);
`figures/fig_n1x_*` (3 cross-window) and `fig_n1_*_<tag>` (15 per-window);
`docs/ch8_deadband_selection.tex` §`sec:n1` rewritten;
`docs/handover_deadband_n1_data.md` for the write-up session.

---

## Interim results, window 1 only (2026-08-03 07:11) — superseded above

### Profile-driven drift — the lower bound

Open-loop pool, 4 provably-silent twins (δ ≥ 0.025): TS n = 48, DS n = 3360.

| | p90 | max |
|---|---|---|
| TS parks | 0.00285 | 0.00304 |
| DS parks | 0.00108 | **0.01101** |

False-activation rate (windows in which the droop fires on ordinary drift):

| δ | TS | DS |
|---|---|---|
| 0 | 48/48 | 3360/3360 |
| 0.001 | 28/48 | 499/3360 |
| 0.005 | 0/48 | 20/3360 (0.6 %) |
| 0.01 | 0/48 | 4/3360 (0.12 %) |
| **0.025** | **0/48** | **0/3360** |

The DS distribution has a **10× tail** (p90 0.0011, max 0.0110): only 4 windows
in 3360 exceed 0.01. The DS dead band is set by rare outliers, not by typical
operation — which is why the choice is a quantile decision, not an optimum.

### Event rejection — the upper bound

Minimum **absolute** DER terminal voltage after the trip (twin baseline 1.0152):

| δ | gen 7 | gen 1 |
|---|---|---|
| 0 – 0.025 | 0.933 – 0.939 | 0.972 – 1.006 |
| 0.05 | 0.9162 | 0.9656 |
| 0.075 | **0.8800** | 0.9514 |
| 0.15 | **0.8161** | 0.9369 |

With the droop disabled, a gen 7 trip drives a DSO_1 park to **0.816 pu**. The
Q(V) layer prevents that violation, but only while δ ≲ 0.05.

### The answer

    strict (zero false activation):   delta in [0.025, 0.05] pu
    with 1 % false-activation budget: delta in [0.005, 0.05] pu
    recommended:                      delta ~ 0.025 pu

Lower bound from the DS drift maximum (0.011), upper bound from the 0.9 pu
undervoltage limit under the worst N-1. Compensation is best at the lower end
(gen 1: 0.49/0.32 at δ = 0.025 against 0.08/0.19 at δ = 0.075).

**One uniform δ suffices** here: the TS admissible set ([0.005, 0.075]) contains
the DS set ([0.025, 0.075]), so DS binds and TS has room to spare. δ_TS ≠ δ_DS
is not required at this operating point.

### The static screening was inverted — why RMS was necessary

| | static scan | RMS, no droop |
|---|---|---|
| gen 1 (650 MW) | 0.0854 | 0.104 |
| gen 7 (830 MW) | **0.0104** | **0.224** |

Statically gen 7 looked like the *mildest* event; dynamically it is by far the
**most severe** — 20× the predicted excursion, 5× the DER activity, and the only
case producing an undervoltage violation. The static scan redistributes the lost
830 MW instantly through distributed slack and so sees almost nothing. Had the
factorial been run on QSS as originally proposed, the disturbance selection and
the recommendation would have rested on an inverted ranking.

### Verification of the headline number

* pre-trip |ΔV| between each trip run and its twin is **exactly 0.00000**, so
  the twin referencing contributes no artifact;
* the gen 7 peak is at t = 191.0 s, 10.5 s after the trip, with **880 of 6247**
  samples above 0.05 pu — sustained, not a spike;
* absolute voltages confirm it: 0.8161 pu at the worst park.

## Open

- The TS drift sample is small (12 windows per 600 s twin, 4 parks × 3 windows).
  Adequate for a rate, thin for a tail.
- Single window. Reproducibility across operating points is the main remaining
  weakness, and is the natural stage-3 extension.
- Stage 2 (δ_TS ≠ δ_DS) is deliberately deferred until this stage locates the
  interesting region, rather than spending 36 cells discovering the plane is
  mostly flat.
