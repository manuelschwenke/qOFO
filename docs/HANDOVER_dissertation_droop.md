# Mini handover — dead band x droop data, for the dissertation session

**You have the dissertation but no PowerFactory.** Everything below is already
computed and stored; nothing here needs a simulator. If you find you need
another simulation, that is a request back to the PowerFactory machine.

**Environment** (the path in `.claude/CLAUDE.md` is a workstation path and is
wrong on this machine):

```
F:\python_environments\qOFO_clean\python.exe      # run from the project root
```

---

## 1. READ THIS FIRST — the analysis module cannot see the new data

`analysis/deadband_n1.py` was written for a single droop. It has

```python
ADMIT = { ... "dso_qv_slope_pu": 0.06, ... }
```

and keys cells by `(window, delta, gen)` with **no droop dimension**. Run as-is
it silently returns only the older droop-0.06 study and ignores all 96 new runs.

Two fixes are needed before it can process this data:

1. remove `dso_qv_slope_pu` from `ADMIT` (or make it a parameter), and
2. add droop to the cell key: `(droop, window, delta, gen)` — twins, the
   no-droop reference and `comp_eff` must all be resolved **within one droop
   level**, exactly as was done for `window` earlier.

Until that is done, use the numbers in §3, which were computed directly from the
run directories.

---

## 2. Where the data is

| | path |
|---|---|
| runs | `results/rms_phase6_replay/0*` — one directory per cell |
| per-run traces | `<run>/csv/rms_der_raw.csv` (~26 MB, `uDER_*` = terminal V, `qDER_*` = park Q) |
| per-run records | `<run>/rms_records.pkl` (interface Q, zone voltages, taps) |
| per-run config | `<run>/config.json` → `runner_static` block |
| completeness helper | `tools/missing_cells.py` |
| existing figures | `results/deadband_n1/figures/` (droop-0.06 study) |
| chapter draft | `docs/ch8_deadband_selection.tex` §`sec:n1` — **written for droop 0.06 only** |
| earlier handovers | `docs/handover_deadband_n1_data.md`, `docs/handover_thesis_writeup_deadband.md` |

**Identify a cell** from `config.json` → `runner_static`:

```
tso_qv_slope_pu     droop   (0.05, 0.06, 0.10)   <- NOTE: this IS the droop
tso_qv_deadband_pu  delta
start_time          window
contingencies[0].element_index   tripped generator; absent => undisturbed twin
n_total_s           600.0 for this study
```

**A run is usable only if `csv/rms_der_raw.csv` exists and is >= 20 MB.** Every
good trace is 25.7–27.6 MB; anything smaller is a truncated write. Do **not**
use the exit code — every N-1 run exits 1 because Gate E validates QSS/RMS
equivalence, which a topology change legitimately breaks.

Per-run `.log` files under `results/deadband_n1/logs/` are **UTF-16**; an ASCII
grep silently matches nothing in them.

## Matrix

- droop **0.05** and **0.10**: 48 cells each — 2 windows x 8 deltas x {twin, gen 1, gen 5}
- droop **0.06**: 84 cells — 3 windows, older ladder, events gen 1 and gen 7
- windows: `2016-01-05 08:00` (+409 MW net infeed), `2016-02-22 13:00` (−117 MW),
  and for droop 0.06 also `2016-12-18 14:00` (+1367 MW)
- deltas: 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, **0.5 = no droop reference**
- events: gen 1 (650 MW, zone 2, open-loop peak 0.083 pu),
  gen 5 (560 MW, zone 3, 0.051 pu). gen 7 (830 MW, 0.22 pu) appears only at
  droop 0.06 and was dropped for saturating — every delta below 0.05 behaves alike.
- horizon 600 s, trip at t = 200 s, `rural_700`, DSO_3 x2, rev-2 sensitivities

---

## 3. The result to write

**delta is a detector threshold**, not a quantity with an optimum: profile-driven
voltage shifts belong to the OFO and must stay INSIDE the dead band; event-driven
excursions must fall outside it and be compensated.

### The lower bound is droop-independent — measured, not argued

At delta = 0.5 the droop never engages, so profile drift is a property of the
plant. The false-activation rates come out **bit-identical** at droop 0.05 and
0.10:

| delta | TS | DS |
|---|---|---|
| 0.0025 | 0.333 | 0.045 |
| 0.005 | 0.083 | 0.007 |
| 0.0075 | 0.000 | 0.001 |
| 0.01 | 0.000 | 0.001 |
| >= 0.025 | 0.000 | 0.000 |

(worst of the two windows; drift maxima TS 0.0075 at −117 and 0.0029 at +409,
DS 0.0110 at +409 and 0.0061 at −117)

So the droop cannot move the lower bound. Only the **upper** bound is at stake.

### Rejection is droop-dependent, in the expected direction

Compensation `1 - peak(delta)/peak(no droop)` at the worst park:

| delta | gen 1 @+409 | gen 1 @−117 | gen 5 @+409 | gen 5 @−117 |
|---|---|---|---|---|
| | 0.05 → 0.10 | 0.05 → 0.10 | 0.05 → 0.10 | 0.05 → 0.10 |
| 0.005 | 0.68 → 0.55 | 0.79 → 0.75 | 0.69 → 0.58 | 0.83 → 0.74 |
| **0.01** | **0.60 → 0.49** | **0.74 → 0.68** | **0.45 → 0.41** | **0.74 → 0.71** |
| 0.025 | 0.33 → 0.27 | 0.54 → 0.50 | 0.41 → 0.39 | 0.58 → 0.51 |
| 0.05 | 0.27 → 0.28 | 0.32 → 0.30 | 0.01 → 0.01 | 0.43 → 0.42 |

No-droop peaks (delta = 0.5): gen 1 → 0.1039 (+409) / 0.1328 (−117);
gen 5 → 0.0513 (+409) / 0.1074 (−117).

### Headline

**delta ~ 0.01 pu is admissible at both droops.** A 10 % droop costs roughly
10 percentage points of compensation against a 5 % droop but does **not** move
the admissible interval. Since the lower bound is droop-independent and the
upper bound stays far above 0.01, the recommendation holds across the tested
part of the 5–15 % band the grid code permits.

Also worth stating: gen 5 at +409 collapses to compensation 0.01 by delta = 0.05
because its whole excursion is only 0.051 pu — the dead band has reached the
event size and the droop stops engaging. That is the detector logic visible
directly in the data.

---

## 4. Figures to make (cross-droop versions do not exist yet)

`analysis/deadband_n1_figures_multiwindow.py` plots per-window across windows;
there is no droop dimension yet. The figures that carry this chapter:

1. **false activation vs delta, one line per droop** — they overlie exactly,
   which is the visual proof that the lower bound is droop-independent
2. **compensation vs delta, one line per (droop, window)** — shows the ~10 pp
   penalty and that the curves stay well above zero at delta = 0.01
3. **peak |dV| vs delta per droop**, with each no-droop level as a dashed
   reference

Style, from `analysis/deadband_n1_figures.py::style()`: minimal, **no titles**
(captions belong in the document), log x-axis for delta, consistent colours.

`docs/ch8_deadband_selection.tex` already has
`\graphicspath{{../results/deadband_selection/figures/}{../results/deadband_n1/figures/}}`.
If the dissertation is a different file, adjust that path or copy the PDFs.

---

## 5. Caveats to carry into the text

- **Two droops tested (0.05, 0.10), not three.** The grid code permits 5–15 %;
  0.15 is untested. The trend across 0.05 → 0.10 is monotone and shallow, so
  extrapolating to 0.15 is plausible but **not measured** — say so.
- **Two windows for the droop study** (+409, −117); the droop-0.06 study has a
  third (+1367).
- `qv_slope_pu` **is the droop**, not a gain: it divides the voltage error
  (static `R = S_n/slope`, RMS `Kdroop = 1/slope`), so 0.10 pu of deviation
  commands full rated Q. Define it once as *droop* or a reader will take it for
  a gain of 0.06.
- **No measurement noise** anywhere. A dead zone's textbook justification is
  noise rejection and that mechanism is absent, so every delta here is a
  *lower* bound for a noisy plant. Argued, not measured.
- **The excursion is a transient**, not an offset: it peaks ~10 s after the trip
  and settles within ~120 s, i.e. inside one 180 s TSO interval — which is why
  the droop, not the OFO, is what acts on it. Never compare the RMS peak with a
  static scan's settled value as a ratio; they are different quantities.
- **delta_TS = delta_DS throughout.** Whether the levels want different values
  is untested.
