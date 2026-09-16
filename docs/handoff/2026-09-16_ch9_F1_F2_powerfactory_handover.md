# Handover: run Chapter-9 experiments F1 and F2 on PowerFactory

You are picking up a PhD experiment campaign on hierarchical multi-zone reactive power
control. Everything that can be done without a PowerFactory licence has been done on
another machine. **Your job is F1 and F2, which need RMS co-simulation.** Do not re-run
the quasi-steady-state chain; it is finished and its results are in the repo.

Read `.claude/CLAUDE.md` first — it states the controller hierarchy, the controlled
outputs and the house rules for how results must be reported. Then read
`experiments/ch_9_parameter_selection/RUNBOOK_ch9.md` (sections 7–9 are yours) and
`docs/daily_log/09_2026/2026-09-15b_ch9_network_decision_and_f1_instrumentation.md`,
which records every decision below and why.

---

## 0. Before anything: two blocking checks

### 0.1 Confirm your machine

* PowerFactory **2025 SP4** with the Python 3.12 API at
  `C:\Program Files\DIgSILENT\PowerFactory 2025 SP4\Python\3.12`
  (`pf/session.py::DEFAULT_PF_PYTHON_PATH`), and a working licence.
* Project **`qOFO\IEEE39_qOFO`** in that machine's PowerFactory database.
  **It is not in the repo.** If it is missing, stop and ask.
* A Python 3.12 environment matching `requirements/`. On the origin machine it is
  `F:\python_environments\qOFO_clean\python.exe`; yours will differ. Note that
  `.claude/CLAUDE.md` records an interpreter path that does not exist — ignore it.
* Run **from the share**, `Z:\Python_Projekte\qOFO_2027`. `results/` is gitignored but
  lives inside the repo, so outputs land back with the rest of the campaign.

Smoke test: `python -X utf8 pf/hello_pf.py`

### 0.2 Commit before you run anything

The working tree is **dirty**: 21 modified files, 16 untracked, including all the
Chapter-9 changes made on 2026-09-15. Every run of the QSS chain is stamped
`[WORKING TREE DIRTY]` in its `run_meta.json`, meaning its `git_commit` does not identify
the code that produced it. That is already a defect in the record; do not add F1 and F2 to
it.

Ask the author before committing, then commit, then run. Your runs must be reproducible
from their hash — they are the chapter's disturbance-response evidence.

---

## 1. What is already decided — do not relitigate

| decision | value | where it came from |
|---|---|---|
| Controller weights | `ch9_selected` (candidate `399e30d3b60b`) | re-confirmed on the batch network 2026-09-16, `results/THESIS_ch9_3_stage2_dso3x2/refine_evented/selection_report.json` |
| Network | **doubled DSO_3** (`dso_der_scale` / `dso_load_p_scale` = 2.0) | author's decision; the PF project stores this model |
| Window bank | **W1–W6**, the thesis conditions table | author's decision |
| `T_STS` | 20 s | Experiment A; binding open-loop settling 12.96 s, margin 7.04 s |
| `T_TS` | **180 s** | `N_inner` p95 = 6.0 at the declared tolerance; ρ flat across 180–360 s so no longer period earns its latency |
| "Interface Q settled" | **0.5 Mvar** | ≈10× the measured 0.045–0.05 Mvar measurement-chain resolution |
| Interface Q means | **coupling transformers only** (`Q_DSO_NC3W_*`), *not* TS tie lines | author |
| Bulk shunt switching | reported, **not constrained** — out of the selection's scope | author; see `tuning/stage2_search.py::Limits` |

W1–W6 start times are in `tuning_mc/scenarios_mc_v2.py::CONDITIONS`:

| | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| start | 2016-01-05 08:00 | 2016-02-22 13:00 | 2016-01-04 08:00 | 2016-11-26 10:00 | 2016-02-03 20:00 | 2016-07-18 14:00 |
| stratum | full | full | partial | partial | **none** | full |

W5 has zero DER reactive capability at both levels. It is **kept** in F1/F2 and reported
as a zero-capability limitation — it is not excluded here the way it is from the weight
selection.

---

## 2. Your first run is a verification, not a measurement

**The F1 instrumentation has never touched a live PowerFactory.** It was written and
unit-tested on the origin machine, which has no licence. This is the single most likely
thing to fail, and it fails silently if you are not careful.

The dead-band experiment needs six signals per DER park from the `QVPRE` DSL block
(`pf/wecc_apply.py::QVPRE_EQUATIONS`), registered in
`pf/screening.py::QV_INTERNAL_SIGNALS`:

```
s:Vanchor   the Q(V) anchor; its CHANGES are the effective anchor-update instants
s:x1        Tf-filtered terminal voltage — the dead-zone INPUT
s:veff      x1 - Vanchor — literally |U_filtered - U_anchor|, the F1 quantity
s:qcorr     dead-zone output; non-zero iff the dead zone acted
s:qraw      qset - Kdroop*qcorr — UNSATURATED droop command
s:Qext      clip(qraw, qmin, qmax) — SATURATED reference
```

plus `s:qset`, which is a long-standing registered monitor that no entry point ever
exported (both harvest filters admit `qDER_`, and `"qsetDER_x".startswith("qDER_")` is
`False`). The `s:qset` precedent is why the `s:<name>` convention is believed correct —
but **believed is not verified**.

Run one short diagnostic:

```bash
python -X utf8 -m experiments.run_rms_cosim --paramset ch9_selected --physical-capability --qv-internals --duration 600 --stride 10 --scenario rural_700 --start-time "2016-01-05 08:00" --profiles --profile-delivery elmfile --tso-deadband 0.5 --dso-deadband 0.5 --no-pdf --verbose 1
```

Then check **all** of:

1. The console prints `[qvpre] recording N Q(V) dead-band diagnostic channels` with
   `N = 7 × number of parks` (~44 parks, so ~300).
2. `csv/rms_qv_internals.csv` exists and is non-empty.
3. **`veffDER_* == vfiltDER_* - vanchorDER_*` sample-wise**, to numerical tolerance. If
   this fails, a monitor name is wrong and every F1 number would be garbage.
4. `vanchorDER_*` actually changes during the run — the anchor is written by pooled
   `EvtParam` events, and if it never moves the anchor-update detection has nothing to key
   on.

The code raises rather than returning empty at three points (`pf/plant.py` if no QVPRE
monitor registers, and both RMS entry points if the export comes back empty), so a wrong
name should fail loudly. Do not trust that; check items 3 and 4 yourself.

If a name is wrong, the fix is in `pf/screening.py::QV_INTERNAL_SIGNALS` — the block's
signal names are authoritative in `pf/wecc_apply.py`.

The analysis that consumes these is `analysis/deadband_anchor.py`
(`python -m analysis.deadband_anchor --self-test` → 15/15 PASS offline).

---

## 3. F1 — smallest dead-band width with no routine droop activation

12 mandatory RMS traces, 18 with the recommended narrow-width comparison.

**Use `run_rms_cosim`, not the comparison script.** F1 is RMS-versus-RMS; the QSS twin is
not the reference and dropping it roughly halves the wall time.

### 3.1 F1-reference — six droop-suppressed 2 h traces

```bash
python -X utf8 -m experiments.run_rms_cosim --paramset ch9_selected --physical-capability --qv-internals --duration 7200 --stride 100 --scenario rural_700 --start-time "2016-01-05 08:00" --profiles --profile-delivery elmfile --tso-deadband 0.5 --dso-deadband 0.5 --no-pdf --verbose 1
```

Repeat for each of the six W start times. `--tso-deadband 0.5 --dso-deadband 0.5` is the
**no-droop reference** (0.5 pu is unreachable by a factor of two; the earlier 0.15 was
not, and every gen-7 figure computed against it is a lower bound).

`--stride 100` is 1 s rows — ample for slow drift, and it cuts the per-park CSV from
~310 MB to ~31 MB per trace.

Then:

```bash
python -X utf8 -m analysis.deadband_anchor --mode reference --run <dir1> --run <dir2> --run <dir3> --run <dir4> --run <dir5> --run <dir6> --startup-exclude-s <declared> --margin <declared> --out results/deadband_anchor/f1_reference.csv
```

**Declare the startup exclusion and the margin before you reduce any data**, use the same
values in every leg, and report startup separately. Do not infer convergence from 180 s
having elapsed.

### 3.2 F1-confirmation — six matched traces at the selected width

Same commands with `--tso-deadband <δ> --dso-deadband <δ>` at the width
`select_half_width()` returns, **active from initialisation** — not switched in at an
event. Then `--mode confirm --delta <δ>`.

Pass criteria: zero dead-zone activation and zero incremental droop command within the
declared tolerances. **Zero-headroom parks are reported separately** — a park at a
capability limit shows zero incremental Q while its dead zone is fully active, and the
analysis flags those as `zero_headroom` rather than counting them quiet. Check that column.

### 3.3 Recommended: six more at the adjacent narrower width

Shows the onset of interaction. Without it you have a width that works and no evidence it
is the *smallest* one that works.

### 3.4 Cost

**Unknown — measure it.** The only figure on record is ~25 min/run at 600 s for the
*comparison* script (2026-08-02). `run_rms_cosim` is roughly half that, and a 2 h trace is
12× the simulated time. **Time one full F1-reference trace before committing the bank**,
and report the actual rate.

---

## 4. F2 — disturbance demonstration

**6 windows × 3 disturbances × 3 legs = 54 RMS cells**, W5 included and reported as a
zero-capability limitation.

* Disturbances: **+200 Mvar at TS bus 7**; **+200 and −200 Mvar at TS bus 11**. Use
  `--q-step-bus` / `--q-step-mvar`, which leave topology unchanged so Gate E stays
  meaningful.
* Legs: F1-selected width, one wider width, droop disabled.
* **Keep `--stride 10`** here. F2 measures a post-event peak.
* Activate the candidate width **throughout the operational run**. If you use an
  event-time switch for isolation, label it as a separate instrument and retain a
  continuous-operation confirmation.
* Event time by a **stated phase** relative to actual dispatch/re-anchoring — e.g. 0.5 s
  after a supervisory update, following a common run-up. **Record the realised time**, do
  not assume the nominal one.
* A −200 Mvar step that would make a nominal load capacitive must be checked and described
  physically, not silently clipped.
* Continue three legs of two representative cases for ≥3 supervisory periods past the
  event to show OFO handover. Plan those six extended cells **before** execution if you
  want them inside the 54.

```bash
python -X utf8 -m experiments.run_comparison_rms_cosim_qss --paramset ch9_selected --physical-capability --qv-internals --duration 600 --stride 10 --scenario rural_700 --start-time "2016-01-05 08:00" --profiles --profile-delivery elmfile --tso-deadband <δ> --dso-deadband <δ> --q-step-bus 7 --q-step-mvar 200 --trip-time 200 --no-pdf --verbose 1
```

### 4.1 Verify ONE event before the full bank

From the plant itself, confirm: the realised event time, load Q before and after, the
target identity, and the installed dead band. The previous F2 exports were **identical
across nominal events and widths** and are not valid disturbance-response evidence.
Identical output hashes are an investigation trigger; differing hashes alone are not proof
an event occurred.

### 4.2 Metrics

Actual dead-zone activation delay; unsaturated and saturated incremental command;
delivered incremental Q; available directional headroom; time-weighted `|U − U_pre|`
through the next subordinate dispatch; and the **unblanked** voltage extrema and time
outside the corridor. If post-switching blanking helps, report it as a separate metric and
keep the raw extrema. Quote ratios against the no-droop leg only where the denominator is
meaningful; always keep absolute values.

No universal 25 % recovery condition is imposed.

---

## 5. Reading rules — get these wrong and you will misread the whole campaign

1. **Exit code 1 is expected** from `run_comparison_rms_cosim_qss`. It returns
   `0 if gate_ok else 1`, and Gate E (QSS↔RMS equivalence) legitimately fails through a
   disturbance. **A genuine failure looks different: no `rms_records.pkl`.** That is the
   check to apply.
2. **Do not reuse the eight archived `.ps1` launchers** under `experiments/`. They pass
   `--dso-oltc-switch-cost 200`, which replaces the selected `g_w_dso_oltc = 15.7164` with
   200 — a factor of 12.7 — and, being a plain assignment rather than
   `dataclasses.replace`, leaves the derived per-DSO voltage relief on the parameter set's
   base. They also pass no `--paramset`. They are the record of past campaigns; leave them
   alone.
3. **`--physical-capability` is mandatory.** Without it `der_q_capability_override_pu = 1.0`,
   which is a diagnostic, not a physical model, and no published result may use it.
4. **`--paramset ch9_selected` explicitly**, even though it is now the default, so the
   command line records the controller.
5. Watch disk: 54 cells at `--stride 10` is ~1.4 GB of per-park CSV, on a network share.

---

## 6. What NOT to do

* Do not re-run A, B, C, D, E or the cadence sweep. They are complete on one unified
  configuration.
* Do not modify `configs/paramsets/ch9_selected.json`. Its `fields` are correct and
  independently re-confirmed; only its `provenance` block is stale, and updating that is
  the author's call.
* Do not write into `results/THESIS_ch9_3_stage2/` (the symmetric archive) or
  `results/THESIS_Ch9_A_plant_settling/` in `qOFO_GH`. Both are historical records.
* Do not change `T_TS`, `T_STS`, the network, the bank or the 0.5 Mvar resolution. If your
  results suggest one of them is wrong, **report it and stop** — those are chapter-level
  decisions.

---

## 7. Known-open items you may hit

1. **W5 shows 19 hunting reversals and ~1.1–1.25 taps/h** at the selected weights in the
   12 h QSS bank. It is excluded from the weight-selection aggregate as zero-capability,
   but that exclusion rationale predates the shunt integrator being live. If W5 behaves
   oddly in F1/F2, this is why — report it, do not work around it.
2. **Bulk shunt switching is high and deliberately unconstrained**: 210 commits / 144
   reversals over 6×12 h at the selected weights, with W4 (a ranked window) at 126
   commits/day. Out of scope for the selection; if it interferes with F1's
   "no routine switching interference" claim, that is a finding worth reporting.
3. **A run with no ordinary switching cannot establish switching non-interference.** If an
   F1 trace happens to contain no normal tap or shunt operation, extend the duration or
   declare an ordinary operating scenario — do not quietly remove the actuator.

---

## 8. When you are done

Write a daily-log entry under `docs/daily_log/09_2026/` per `.claude/CLAUDE.md`: what was
changed, the method, a timestamp, and the reason. State assumptions, constraints, actuators
and controlled outputs when discussing results. Distinguish established model facts from
hypotheses from open questions. Report the actual wall-clock cost per trace so the next
campaign can be planned from measurement rather than estimate.

If the instrumentation check in section 2 fails, **stop and report** rather than
working around it. An F1 result computed from a reconstructed anchor is exactly the defect
this instrumentation was built to remove.
