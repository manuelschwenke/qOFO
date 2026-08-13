# Prompt for the Claude Code session on the ms_admin account

Paste everything below the line into that session.

---

You are running one half of a PowerFactory RMS co-simulation sweep. The other
half runs concurrently in a session on the `mschwenke` account. Do not
duplicate its work.

**Project path on this account:** `V:\` — that drive letter is mapped directly
to the `Python_Projekte\qOFO_GH` folder, so the project root IS `V:\` (not
`V:\Python_Projekte\qOFO_GH`). The other account reaches the same folder as
`Z:\Python_Projekte\qOFO_GH`. **It is the same folder** — same code, same
results directory.

**Python:** `F:\python_environments\qOFO_clean\python.exe`
(the path in `.claude/CLAUDE.md` is a workstation path and is wrong on this
machine). `powershell.exe` is at
`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`.

## Run exactly this

```
powershell -NoProfile -ExecutionPolicy Bypass -File V:\experiments\RUN_ON_SECOND_ACCOUNT.ps1 -Prj 'V:\'
```

That is the whole job: **36 runs, roughly 13 hours.** It runs a 20-minute smoke
test first and aborts if PowerFactory is not usable from this account — let it.

## What it does, and why the split is what it is

| account | dead bands δ | runs |
|---|---|---|
| **this one (ms_admin)** | 0.025, 0.05, 0.1, 0.5 | 36 |
| mschwenke | 0.0025, 0.005, 0.0075, 0.01 | 36 |

Fixed for every run: **droop 0.10 pu**, horizon 600 s, trip at t = 200 s,
scenario `rural_700`, three operating windows
(2016-01-05 08:00, 2016-12-18 14:00, 2016-02-22 13:00), and for each δ a
`{twin, gen 1 trip, gen 5 trip}` triple.

Each δ cell is self-contained (its own twin), so this half can be analysed even
if the other half fails.

## Things that will look like failures and are not

1. **Every N-1 run exits 1.** The entry point returns Gate E's verdict, and
   Gate E validates QSS/RMS *equivalence*, which a generator trip legitimately
   breaks. The sweep prints `!!! FAILED run n` for each — expected.
   **The real health test is whether `<run>/csv/rms_der_raw.csv` was written.**
2. **Per-run logs are UTF-16** (PowerShell `*>` redirection), so an ASCII grep
   for `Traceback` silently matches nothing on a log full of errors. Decode as
   utf-16 when reading them.
3. The shared results folder is safe for two concurrent writers: run-directory
   allocation uses `mkdir()` without `exist_ok` (atomic create-if-not-exists)
   and retries with the next number on collision. No merge step is needed.

## Rules

- **Do not edit any file under the project while runs are in flight.** Both
  accounts import the same modules from the same folder. Editing a module
  mid-run crashed a run earlier today (a half-applied edit produced a
  `NameError` in the RMS anchor pass). If something needs changing, say so and
  wait.
- **Do not start any other PowerFactory work.** One session per account, and
  the other account is using its own.
- Do not run the analysis until both halves finish; it would report a partial
  matrix as if complete.

## Report back

When finished, report: how many of the 36 runs wrote `rms_der_raw.csv`, any run
that did not, and the total wall-clock time. Also confirm the smoke test printed
a line containing `droops applied: [0.1]` — if it printed a different droop, or
none, stop and report immediately, because the whole sweep would be invalid.
