# Handover prompt for a Claude Code session on the server

Paste everything below the line into a Claude Code session on the server.

---

You are setting up and launching a long Monte-Carlo simulation campaign for my PhD thesis on this machine. The campaign is fully specified; your job is to run it **exactly as specified**. Do not modify code, constants, variants or the thesis.

**Source.** The code is the repository `\\e5server\homefolders$\mschwenke\Python_Projekte\qOFO_2027`. That is a git working copy with intended uncommitted changes: do not commit, push, stash or reset anything.

**Read first:** `experiments\part3_mc\README_server.md` and the module docstring of `experiments\part3_mc\run_mc.py`. They define the campaign:
- L2, S1 and O2 on N = 100 valid episodes of 180 min each;
- master seed 20261007;
- discard only on the base-case prescreen.

## Steps

1. **Copy the repository to a local disk**, not under `AppData` and not on the network share. For example:
   ```bat
   robocopy \\e5server\homefolders$\mschwenke\Python_Projekte\qOFO_2027 D:\qOFO_2027 /E /XD results __pycache__ /XF *.pyc
   ```
   Keep `.git`: the driver records `git status` and `git diff` as provenance. All later commands run from that copy. If the machine runs Linux, translate the paths and commands accordingly.

2. **Python environment** `qOFO_clean` (Python 3.12).
   - If it does not exist, create it from `requirements\environment.yml`.
   - In either case, make these exact versions hold, because they are what the laptop reference runs used:
     - `numpy==2.3.5` (the lock file says 2.4.6; override it)
     - `pandapower==3.4.0`, `scipy==1.16.3`, `pandas==2.3.3`
     - `cvxpy==1.8.1`, `gurobipy==12.0.1`, `networkx==3.6.1`, `numba==0.65.1`
   - Print the installed versions to confirm.

3. **Gurobi licence.** Run:
   ```bat
   python -c "import gurobipy as gp; m = gp.Model(); print('gurobi ok')"
   ```
   If this fails, **stop and tell me**. I have to install a licence on this machine (`grbgetkey`). Never run the campaign with `--allow-no-gurobi`.

4. **Check the variant definition**, from the repo root:
   ```bat
   python -c "from experiments.case1_svr.run_case1 import resolved; c = resolved('S1', 180); assert c.svr_k_p_rvr == 0.0 and c.svr_k_p_rpr == 0.0 and c.svr_rpr_voltage_priority; print('S1 ok: integral + safeguard')"
   ```

5. **Smoke test** (a few minutes):
   ```bat
   python -m experiments.part3_mc.run_mc --smoke --jobs 4
   ```
   Then open `results\THESIS_part3_mc_smoke\report.md`. All three variants must show outcome `ok` for both episodes. If not, stop and report the logs in `results\THESIS_part3_mc_smoke\logs\`.

6. **Choose `--jobs`.**
   - Start from the number of physical cores minus 2.
   - Limit it so that the workers have about 1 GB of free RAM each.
   - Tell me the number and the reason.

7. **Launch the campaign detached**, so that it survives the end of this session. In PowerShell, from the repo root:
   ```powershell
   Start-Process -FilePath (Get-Command python).Source -ArgumentList '-u','-m','experiments.part3_mc.run_mc','--jobs','<N>' -WorkingDirectory (Get-Location) -RedirectStandardOutput results\part3_mc_console.log -RedirectStandardError results\part3_mc_console.err -WindowStyle Hidden -PassThru
   ```
   Report the PID. Do not run the campaign in the foreground of this session.

8. **Verify it is running.** After about 10 minutes, check:
   - `results\THESIS_part3_mc\prescreen\` fills up;
   - the console log shows `[run]` lines;
   - the driver wrote `provenance_*.json`.

   Then report:
   - the number of valid prescreens;
   - the first run outcomes;
   - an estimated finish time. Expect about 160 CPU-hours in total; an S1 episode takes about 75 min per core.

## Rules

- **Resume.** If the machine restarts or the pool breaks, run the same command again. It skips everything that already has a JSON and reruns `task_crash` entries.
- **Do not change anything in the campaign.** That means the code, the constants (seed, N, horizon, δ, voltage range, discard rule) and the variant definitions. Do not delete results.
- **If something looks like a bug,** stop and report the traceback from `results\THESIS_part3_mc\logs\`. Examples: many `task_crash` entries, or every run of one variant failing.
- **When the campaign has finished:**
  1. Run `python -m experiments.part3_mc.run_mc --report`.
  2. Write `results\THESIS_part3_mc\SERVER_RUN_NOTES.md`. It should contain: machine (CPU, cores, RAM, OS), `--jobs`, start and end time, package versions, the number of valid episodes and of discarded draws, outcomes per variant, and anything unusual.
  3. Copy `results\THESIS_part3_mc` (without `logs\` and `_scratch\`) to `\\e5server\homefolders$\mschwenke\Python_Projekte\qOFO_2027\results\THESIS_part3_mc`.
