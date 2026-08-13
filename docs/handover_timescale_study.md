# Handover: timescale study for thesis §9.1

**Goal.** Produce the settling numbers that fill Table 9.1 of the dissertation
(`Chapters/Chapter09.tex`, `tab:param:timescales:settling`), which is currently
all `[TBD]`. One script does it. Expect ~30–60 min wall clock.

**Run this on the PowerFactory machine.** Nothing here works elsewhere: the
script drives a live PF session.

> **Moved 2026-08-06.** The script now lives at
> `experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py`, with
> the other Ch. 9 parameter-selection entry points; see that folder's
> `README.md`. `pf\timescale_study.py` is a forwarding shim and still works.
> Everything below applies unchanged except the path.

---

## 1. Preconditions

- PowerFactory 2025 SP4 with a free licence seat, project `IEEE39_qOFO`.
- Python env `qOFO_clean` (3.12) with the `powerfactory` module importable.
- Repo at `\\130.83.232.108\homefolders$\mschwenke\Python_Projekte\qOFO_GH`
  (or a local clone — run from the repo root either way).
- **PowerFactory must be closed in the GUI**, or the script cannot activate
  the study case.

Check the environment first — these need no licence and fail fast:

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --dry-run
```

Expected: a list of the cases that would run, the two outage targets resolving
to `G 03` and `G 09`, and the commit the run would be recorded against. If the
resolution prints anything else, stop and report it (see §5).

The post-processing (settling → table → derived quantities) can be checked
without PowerFactory at all:

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --self-test
```

---

## 2. The run

```bash
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --label full_t0_wecc --save-trajectories
```

What it does, in order:

1. **Preflight** — 60 s flat run, asserts max bus-voltage drift below
   tolerance. *If this fails, stop.* It means the RMS initialisation did not
   land on the load-flow operating point, and every settling time afterwards
   would be that drift plus the response. Do not pass `--skip-preflight` to
   get past it.
2. **Dispatch battery** — largest single command per actuator class
   (DER/park Q, AVR V-ref, coupler OLTC ×1 and ×2 sequential, machine-trafo
   OLTC, MSC), 60 s each.
3. **Disturbance battery** — machine outages `gen[1]`/`gen[7]` plus unit
   transformers, and load steps at ±10 %, +25 %, **600 s each**.
4. Writes the outputs and prints the derived design quantities.

A single failing case is caught and skipped, so the battery always completes;
check the log for `FAILED:` lines afterwards.

---

## 3. Outputs

Written to `results/timescale/full_t0_wecc/<stamp>/`:

| file | use |
|---|---|
| `timescale_table.tex` | the tabular **body** of Table 9.1 — paste over the `[TBD]` rows |
| `timescale_summary.md` | same numbers plus the derived design quantities and the provenance block |
| `timescale_summary.csv` | machine readable |
| `cases.csv` | the resolved catalogue, written *before* the battery runs |
| `run_meta.json` | argv, git commit + dirty flag, interpreter, PF project, every constant entering a number |
| `run.log` | full console transcript |
| `traj_<case>.csv` | per-signal time series |

`results/timescale/<label>/_latest.txt` names the newest stamp.

`timescale_summary.md` reports what the thesis text needs:

- `T_s^cont` — worst *continuous* dispatch settling
- `T_s^tap` — single-tap settling, split into `T_mech` (per step, measured as
  the two-step minus one-step difference, not assumed) and `T_elec`
- the **binding row** — the worst over *every realisable* dispatch, which is
  what enters eq. (9.1) — and the margin at `T_DS = 20 s`
- worst disturbance settling, in dispatch intervals

No `|dtau|_max` is derived: the controller caps taps at one step per
subordinate iteration and then locks the changer out for its cooldown, so a
multi-step command cannot arise and the cap is 1 by configuration.

**Exit codes.** `0` = complete and thesis-ready. `1` = no case succeeded.
`2` = it ran, but a case failed, a table row is unfilled, or a settling time
is censored (still outside the band at the last sample of the run — the
reported value is then a lower bound set by the horizon, marked `>` in the
`.tex`). Do not paste the table on a `2`.

---

## 4. Filling the thesis

In `Chapters/Chapter09.tex`, section `ch:param:timescales`:

1. Replace the `[TBD]` rows of `tab:param:timescales:settling` with the
   contents of `timescale_table.tex`.
2. Fill the bracketed margin in the sentence beginning *"The selected values
   are"* from the **binding row** line of `timescale_summary.md`. (The
   `|\Delta\tau|_{\max}` that an earlier version of this handover named is
   gone from the chapter: the step cap is 1 by configuration, not derived.)
3. Delete the `\todo{Fill from ...}` under the table and the `% NUMBERS: NONE
   YET` block in the section's comment header.

**`N_inner` is not answered by this script.** Eq. (9.2) needs a *closed-loop*
measurement on the isolated DSO-OFO (parent silent, capability-band-traversing
setpoint step), and this battery is open-loop. The summary prints
`T_TS/T_DS = 9` as the **configured** ratio and says so; leave the chapter's
`N_inner` `\todo` standing until a second script measures it.

Do **not** quote the older `results/screening/full_t0_wecc/20260720-*` run: it
predates the `EVENT_WINDOW_S = 60 s` finding of 2026-07-31, so its event
timing is not established. That is why this study exists.

---

## 5. Things that have bitten before

- **Generator numbering is offset.** The contingency configs address machines
  by *pandapower* `net.gen` index; PowerFactory addresses them by name.
  `gen[1]` is **`G 03`**, `gen[7]` is **`G 09`**. The machine literally named
  `G 01` is `gen[8]` — it is the 10 GVA interconnection equivalent and carries
  the angle reference, so tripping it is not a contingency; the script refuses
  it. The script prints every resolution: read those lines.
- **Events fire modulo a 60 s window.** PF applies event times mod 60 once the
  calculation runs. The script's events all sit inside the first window, so
  this should not bite, but if a tap or outage appears not to fire, that is
  the first thing to check.
- **`purge_events` must reset the calculation first.** PF silently refuses
  event deletion while a calculation is active. Handled inside
  `ScreeningContext`, but if cases start contaminating each other, look here.
- **A tripped element's `m:` variables disappear.** That is the documented way
  to detect an outage — not `outserv`, which the event never updates. The
  script tolerates it for outage cases only.
- **Machine transformers are resolved topologically**, not by name: they are
  called `MT_g<i>_t0` after a build index, so `MT_g0_t0` sits under `G 01`.
  If you see `WARNING: no ElmTr2 found on the LV terminal of ...`, the trip
  will leave the unit transformer energised and differ from the closed-loop
  contingency. Report it rather than ignoring it.

---

## 6. What to report back

- The contents of `timescale_summary.md` (it now carries its own provenance
  block: commit, dirty flag, bands, PF project).
- The exit code. Anything other than `0` means the table is not thesis-ready.
- Any `FAILED:` lines from `run.log`, and any `WARNING:` about unresolved
  unit transformers.
- Whether preflight passed and with what drift figure (also in
  `run_meta.json` as `preflight_drift_pu`).

Optional, only if a seat is still free and time allows — it answers a
separate open question (whether the slow eigenvalues are visible in the
controlled outputs at all, and what kind of modes they are):

```bash
python pf\probes\probe_modal_residue.py
```

```bash
python pf\screening.py modal --label full_t0_wecc --export-matrices
```

The probe is read-only and reports whether participation factors or the
system matrices can be extracted. Until they can, no eigenvalue may be given
a physical label — §9.1 deliberately avoids the word "electromechanical" for
that reason.
