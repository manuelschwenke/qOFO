# 2026-08-06 — Ch. 9.1 timescale battery: relocated, made reproducible, bound fixed

**Timestamp:** 2026-08-06
**Scope:** thesis §9.1 (`ch:param:timescales`, Table 9.1 `tab:param:timescales:settling`)
**Trigger:** the script producing Table 9.1 had to be located and made a
steady, reproducible entry point before the run. No PowerFactory seat was
free, so nothing was simulated; everything below is offline work.

---

## What was changed

### 1. Relocation

`pf/timescale_study.py` → `experiments/ch_9_parameter_selection/ch_9_1_timescale_seperation.py`.

New folder `experiments/ch_9_parameter_selection/` holds one entry point per
Ch. 9 section, named after the section it fills, with a `README.md` mapping
section → script → status. `pf/timescale_study.py` remains as a forwarding
shim (three lines of logic, no duplication) because `docs/handover_timescale_study.md`
and the `Chapter09.tex` comment header both name that path.

The PowerFactory driver infrastructure (`ScreeningContext`, the step and
disturbance catalogues, `settling_metrics`, the generator-index resolution)
stays in `pf/screening.py`. It is shared with the RMS build and the dead-band
study; duplicating it into `experiments/` would let the two drift apart.

### 2. Defect fixed: the bound excluded two realisable dispatches

`derive()` computed

    binding = max( T_s^cont over the `param` rows , T_s of the coupler tap )

which leaves out **the MSC switch-in** and **the machine-transformer tap**.
Both are dispatches the controller issues, both are `kind="tap"` and neither
is a `param` row, so a slow one would have set no bound and left no symptom —
the margin in eq. (9.1) would simply have been too large.

Now the binding row is the worst over *every realisable* dispatch, excluding
only the two-step tap, and only because the controller cannot issue it
(`local_oltc_max_step_per_dt = 1` plus the cooldown lockout make it an
instrument for splitting `T_mech` from `T_elec`, never a command). `T_s^cont`
and `T_s^tap` are still reported separately — the chapter text names them —
but the margin is taken against the true worst dispatch, and the summary
states which row binds.

*No thesis number changes, because the study has not been run yet.*

### 3. Reproducibility

Each run directory now also carries:

| file | content |
|---|---|
| `run_meta.json` | argv, git commit + **dirty flag**, interpreter, platform, PF project + study case, and every constant that enters a number (bands, RMS step, read stride, horizons, periods, OLTC rate limits); after the battery, also the preflight drift and the per-case failure list |
| `cases.csv` | the resolved catalogue, written **before** the battery runs, so an aborted run still records what it attempted |
| `run.log` | full console transcript (stdout tee) |
| `_latest.txt` | one level up: the stamp of the newest run, so "which run is Table 9.1 from" has a single answer |

The summary's provenance block says explicitly when the working tree was dirty
at run time, i.e. when the result is *not* reproducible from the commit alone.

### 4. Silent-failure modes made loud

- **Exit codes.** `0` = complete and thesis-ready; `1` = nothing succeeded;
  `2` = ran but not thesis-ready (a case failed, a Table 9.1 row is unfilled,
  or a settling time is censored). Previously a battery that lost cases exited
  `0` and emitted a `.tex` body with `[not run]` rows that read like
  deliberate omissions.
- **Censoring.** A settling time equal to the run horizon is not a
  measurement — the trajectory was still outside the band at the last sample,
  so the true value is unknown and ≥ horizon. Such rows are flagged, marked
  `$>$` in the `.tex`, listed in the summary, and force exit `2`. This is the
  one way the horizon could otherwise set the bound of eq. (9.1) itself.
- **Measured but not tabulated.** The battery runs more cases than Table 9.1
  has rows — the machine-transformer tap and the further load steps. They are
  now listed in the summary under their own heading instead of vanishing.
- **`N_inner`.** The summary used to print `T_TS/T_DS = 9` next to the
  measured quantities, which invites reading it as a measurement. It is the
  *configured* ratio; eq. (9.2) needs a closed-loop measurement on the
  isolated DSO-OFO that this open-loop battery does not perform. The line now
  says so, and the chapter's `\todo` stays.

### 5. Offline verification

`--self-test` exercises the whole post-processing chain on synthetic settling
rows: no PowerFactory, no licence seat. It guards the three failures that
would put a wrong number in the thesis — a table row silently matching
nothing (the `+`-in-a-regex trap of 2026-08-04), the bound missing a
realisable dispatch, and the tap split computed the wrong way round.

The same guards are in `tests/experiments/test_ch9_timescale_seperation.py`
(12 tests, all passing) so they run in the suite rather than only on demand.

---

## Verified today

```
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --self-test   # ALL PASS, exit 0
python experiments\ch_9_parameter_selection\ch_9_1_timescale_seperation.py --dry-run     # gen[1]->G 03, gen[7]->G 09, exit 0
python pf\timescale_study.py --dry-run                                                   # shim forwards, exit 0
pytest tests/experiments/test_ch9_timescale_seperation.py -q                             # 12 passed
```

## Not done

- **The study itself.** No PowerFactory seat was free. Table 9.1 is still all
  `[TBD]` and no number has been produced or quoted.
- **`N_inner`** (eq. 9.2) — needs its own closed-loop entry point in the same
  folder: isolated DSO-OFO, parent silent, capability-band-traversing
  setpoint step, reported as a distribution over the design windows.
- **The combined multi-device tap dispatch** the chapter's `\todo` asks for.
  The battery still moves one actuator at a time; several changers tapping in
  the *same* iteration is realisable (the `gw_oltc_cross_tso` cross-coupling
  penalty is disabled in this configuration) and remains unmeasured.
- **A table row for the machine-transformer tap.** Measured, enters the
  bound, reported in the summary, but Table 9.1 carries a single "OLTC tap,
  one step" row (the coupler). Adding a row is an author decision.
