# Handover — N-1 dead-band data: where it is, what to plot, what to avoid

**For a session that will plot the results and write them into the thesis.**
Written 2026-08-04 10:30, while the last 5 of 57 runs are still executing.

---

## 0. First thing: regenerate

**Environment — the path in `.claude/CLAUDE.md` is WRONG on this machine.** It
names a workstation Miniconda path that does not exist here. Use:

```
F:\python_environments\qOFO_clean\python.exe
```

and run from the project root `Z:\Python_Projekte\qOFO_GH` (the modules are
imported as `analysis.*`, so the cwd must be the root).

`powershell.exe` is also not on `PATH`; it is at
`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`.

The CSV is only as complete as the last analysis run. **Always start with:**

```bash
cd /z/Python_Projekte/qOFO_GH
"F:/python_environments/qOFO_clean/python.exe" -X utf8 -m analysis.deadband_n1
"F:/python_environments/qOFO_clean/python.exe" -X utf8 -m analysis.deadband_n1_figures
```

`-X utf8` matters: without it the console encoding mangles the output on this
host.

`analysis/deadband_n1.py` scans the run directories, admits by config, pairs each
trip run with its same-δ twin, and writes the CSV. `--fa-tol` sets the tolerated
false-activation rate used by the admissibility verdict (default 0 = the droop
must be completely silent on profiles).

---

## 1. What exists

| | path |
|---|---|
| **metrics CSV** | `results/deadband_n1/deadband_n1_metrics.csv` ← **plot from this** |
| figures | `results/deadband_n1/figures/fig_n1_*_<WINDOWTAG>.{pdf,png}` |
| raw runs | `results/rms_phase6_replay/0285…0341` (traces `<run>/csv/rms_der_raw.csv`, records `<run>/rms_records.pkl`) |
| sweep logs | `results/deadband_n1/logs/` (`_multiwindow_master.log` = progress) |
| analysis | `analysis/deadband_n1.py`, `analysis/deadband_n1_figures.py` |
| sweep drivers | `experiments/run_deadband_n1.ps1`, `run_deadband_n1_multiwindow.ps1` |
| draft prose | `docs/ch8_deadband_selection.tex` §`sec:n1` — **already rewritten for the three-window result**; §`sec:n1choice` holds the recommendation |
| writeup guidance | `docs/handover_thesis_writeup_deadband.md` (keep/compress/drop of older sections) |
| design record | `docs/daily_log/08_2026/2026-08-02_deadband_n1_experiment_design.md` |

### Three operating windows

| window | net infeed | label to use | δ ladder |
|---|---|---|---|
| `2016-02-22T13:00` | −117 MW | import | 8 values |
| `2016-01-05T08:00` | +409 MW | reference | **11** values |
| `2016-12-18T14:00` | +1367 MW | high export | 8 values |

**The ladders differ.** Window +409 was measured first and carries three extra
points (0.001, 0.075, 0.15). The **common ladder for cross-window plots** is:

```
0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5
```

A blank cell at 0.001 / 0.075 / 0.15 for the other two windows is **not a failed
run** — that δ was never scheduled there.

`δ = 0.5` is the **no-droop reference**; label it as such, not as a dead band.
It is what `comp_eff` is measured against.

Setup common to every run: `rural_700`, DSO_3 ×2, physical VDE-AR-N-4120
capability, rev-2 sensitivities, RMS co-simulation, 600 s horizon, trip at
t = 200 s, δ_TS = δ_DS.

---

## 2. Column dictionary

| column | meaning |
|---|---|
| `window`, `delta_pu`, `gen` | cell identity (`gen` 7 = severe, 1 = milder) |
| `run`, `twin` | run numbers, for traceability |
| `faopen_ts` / `faopen_ds` | **false-activation rate** — fraction of inter-dispatch windows whose OPEN-LOOP drift exceeds δ. **This is the one to plot.** |
| `fa_ts` / `fa_ds` | same but from the cell's own twin (closed loop). Diagnostic only — at narrow δ the droop suppresses the drift being measured, so it understates. |
| `faopen_p90_*`, `faopen_max_*` | the drift distribution itself |
| `peak_dv_ts_pu` / `_ds_pu` | post-trip peak \|ΔV\| vs twin, worst park |
| `resid_dv_*` | \|ΔV\| still standing at the next TSO dispatch |
| `comp_eff_ts` / `_ds` | `1 − peak(δ)/peak(δ=0.5)` — fraction of the no-droop excursion removed |
| `detected_ts` / `_ds` | whether the event's open-loop excursion exceeds δ |
| `traverse_ts` / `_ds` | actuator motion, Mvar per park per interval |
| `ifq_post_mvar` | interface-Q tracking error after the trip |

---

## 3. Known data issues — read before plotting

1. **Run 0316 (window +409, δ = 0.1, gen 1) had no traces.** The PowerFactory
   `ComRes` export died mid-write; `rms_records.pkl` existed but
   `csv/rms_der_raw.csv` did not, leaving `ifq_post_mvar` populated and **NaN
   for every trace metric** (peak, resid, comp_eff, traverse).

   **Re-simulated on the PowerFactory machine 2026-08-04.** Nothing to do here:
   regenerating the CSV (§0) picks the new run up automatically, and the
   duplicate-cell line in the `[admit]` output will name it superseding 0316.

   *If a NaN row ever reappears:* it needs a PowerFactory re-run, which **cannot
   be done from this machine** — see §7. Drop the point and say so; δ = 0.1 is
   above the admissible range and bracketed by 0.075 and 0.15/0.5, so no
   conclusion depends on it.
2. **`exit=1` is normal.** The entry point returns Gate E's verdict, and Gate E
   validates QSS/RMS *equivalence*, which a topology change legitimately breaks.
   Every N-1 run exits 1. **The real failure test is a missing
   `csv/rms_der_raw.csv`**, which is how 0316 was found.
3. **`comp_eff` is only valid once that (window, gen) ladder is complete**,
   because the reference is the widest δ *present*. While window −117 gen 1 was
   still running, its `comp_eff` was computed against δ = 0.005 and read
   0.27 / 0.10 / 0.00 — an artifact, not a result. **Re-run the analysis after
   the sweep finishes** and the column becomes correct. Sanity check: the
   δ = 0.5 row must show `comp_eff = 0` in every (window, gen) group.
4. Per-run logs are **UTF-16** (PowerShell `*>`), so an ASCII `grep` for
   `Traceback` silently matches nothing on a log full of errors.

---

## 4. The result to write

**δ is a detector threshold**, not a quantity with an optimum: profile-driven
shifts must stay inside the dead band (they belong to the OFO), event-driven
excursions must fall outside it and be compensated.

### Headline: δ = 0.005 does NOT survive replication; use δ ≈ 0.01

False activation, TS parks:

| δ | −117 | +409 | +1367 |
|---|---|---|---|
| 0.0025 | 0.450 | 0.167 | 0.083 |
| **0.005** | **0.133** | 0.000 | 0.000 |
| **0.01** | **0.000** | **0.000** | **0.000** |

δ = 0.005 was chosen on the +409 window, where TS false activation is exactly
zero. That property is **not reproducible**: at the import window TS drift is
2.4× larger (max 0.0075 vs ~0.0031) and δ = 0.005 fires on 13.3 % of TS
inter-dispatch windows.

- zero TS false activation at **all** windows requires **δ ≥ 0.01**
- zero at **both levels** requires **δ ≥ 0.0125** (DS at δ = 0.01 is 0–0.5 %)
- cost of 0.005 → 0.01 is ≈ 0.05 of compensation — small

### What DOES reproduce

- **DS drift maximum**: 0.01101 / 0.00893 / 0.01107 across the three windows
  (24 % spread)
- **Compensation is flat up to δ ≈ 0.025** at every window, then decays
- **gen 7 is the severe case everywhere**: no-droop peak 0.176 / 0.201 /
  0.273 pu

### Unexplained, and to be written as such

TS drift is 2.4× larger at the import window. Three candidate mechanisms were
tested and **all refuted** — state them, because they tell a reader where not to
look:

| hypothesis | why it fails |
|---|---|
| less DER capability at low infeed | capability is *identical* (901.2 Mvar) at all three windows; P/Sₙ = 0.38–0.60, far above the 0.2 knee of the VDE diagram |
| Q saturating on the operating diagram | peak utilisation 12 %; **zero** samples above 95 % of capability |
| OLTC tap steps driving the drift | windows containing a tap step have 5–14× *lower* drift, and there are only 1–3 tap events per 600 s run |

Do not offer a fourth mechanism without testing it.

---

## 5. Figures

Two generators, both already written:

```bash
"F:/python_environments/qOFO_clean/python.exe" -X utf8 -m analysis.deadband_n1_figures             # per-window
"F:/python_environments/qOFO_clean/python.exe" -X utf8 -m analysis.deadband_n1_figures_multiwindow # cross-window
```

**The cross-window set carries the thesis argument** — the case for δ is a
*replication* case, so the figures that matter put all three operating points on
one axis:

| figure | shows |
|---|---|
| `fig_n1x_false_activation` | **the headline.** False-activation rate vs δ, one line per window, TS and DS panels. This is what rules out δ = 0.005 and justifies 0.01 |
| `fig_n1x_compensation` | compensation vs δ per window, one panel per tripped machine — the flat region and where it decays |
| `fig_n1x_peak` | post-trip peak \|ΔV\| vs δ, with each window's no-droop level dashed |

Both modules share `style()` and `save()` from `analysis/deadband_n1_figures.py`
(minimal, no titles, log-x). Windows are labelled by **net infeed**, not date —
that is the physically meaningful ordering — and colours are fixed in the
`WINDOWS` dict. `DELTA_STAR = 0.01` draws the recommended value as a guide line;
change it there if the recommendation moves.

Only the **common ladder** is drawn (8 values). The reference window's three
extra dead bands are excluded on purpose: plotting them would show phantom gaps
in the other two series.

### Placing them in the dissertation

The figures live in `results/deadband_n1/figures/`. `docs/ch8_deadband_selection.tex`
already has

```latex
\graphicspath{{../results/deadband_selection/figures/}%
              {../results/deadband_n1/figures/}}
```

**If the dissertation is a different file/repo, that path must be adjusted** —
it is relative to the tex file's location, and I do not know where the
dissertation lives. Either extend its `\graphicspath` or copy the PDFs in.
Filenames are stable across re-runs, so a copy stays valid.

---

## 6. Caveats for the text

- **Three windows, one scenario** (`rural_700`, DSO_3 ×2). Not an annual or
  frequency-weighted statement.
- **Two machines tripped** (gen 7, gen 1) of six. `gen 9` diverges in the static
  screening and is excluded. No N-1 completeness claim; line outages are not yet
  deliverable by the RMS adapter.
- **No measurement noise** anywhere. A dead zone's textbook justification is
  noise rejection, and that mechanism is absent — so every δ here is a *lower*
  bound for a noisy plant. Argued, not measured.
- **The excursion is a transient**, not a sustained offset: gen 7 peaks ~10 s
  after the trip, exceeds 0.05 pu for ~90 s, and settles below 0.01 pu. It opens
  and closes *inside one 180 s TSO interval*, which is exactly why the droop
  (not the OFO) is what acts on it. Do not compare the RMS peak with the static
  scan's settled value as a ratio — they are different quantities.
- **δ_TS = δ_DS throughout.** Whether the two levels want different values is
  untested; the per-level machinery exists
  (`--tso-deadband` / `--dso-deadband`) and a 2D sweep is the natural follow-up.

---

## 7. Machine split — what CANNOT be done from here

Your machine has **the dissertation but no PowerFactory**; the simulator is on a
different machine. The split matters:

| task | where |
|---|---|
| analysis, figures, writing | **here** — everything needed is in the CSV and the two figure modules; no simulator involved |
| any new or repeated **simulation** | **the PowerFactory machine only** |

So **do not plan work that requires new runs.** If the write-up turns out to
need another cell, another dead band, another operating window, or the 2D
δ_TS ≠ δ_DS sweep, that is a request back to the PowerFactory machine — it
cannot be satisfied here. The failure mode to avoid is quietly plotting from a
stale or incomplete CSV instead.

Everything required for the write-up is already computed and stored: 57 runs,
3 windows, 54 metric rows. The single command in §0 regenerates the CSV and all
18 figures from the existing runs, offline.
